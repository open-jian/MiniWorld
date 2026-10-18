"""Inference-only residual gate/add fusion for channel-major video tokens.

The eager expression materializes a BF16 gate/update product before adding the
BF16 residual. This kernel keeps that rounding boundary while avoiding the
temporary tensor. It also retains the residual layout required by fused
RMSNorm/AdaLN; making this output row-major would disable that optimization.
"""

from functools import lru_cache
import importlib.util
from typing import Optional

import torch
from torch import Tensor


def _gate_layout(
    gate: Tensor, batch: int, tokens: int, channels: int,
    tokens_per_frame: Optional[int],
) -> Optional[tuple[int, int, int]]:
    """Return broadcast-aware batch/row/channel strides without expanding gate."""
    if any(stride < 0 for stride in gate.stride()):
        return None
    if tokens_per_frame is not None:
        if gate.ndim != 4 or gate.shape[2] != 1:
            return None
        rows = tokens // tokens_per_frame
    else:
        # Match the eager three-dimensional broadcast exactly. In particular,
        # do not reinterpret an ambiguous two-dimensional gate as batch-wise.
        if gate.ndim != 3:
            return None
        rows = tokens
    if (gate.shape[0] not in (1, batch) or gate.shape[1] not in (1, rows)
            or gate.shape[-1] != channels):
        return None
    return (
        gate.stride(0) if gate.shape[0] != 1 else 0,
        gate.stride(1) if gate.shape[1] != 1 else 0,
        gate.stride(-1),
    )


def _fits_index_span(value: Tensor) -> bool:
    # Tensor pointers already incorporate storage_offset. Relative offsets and
    # all output/grid calculations must fit the kernel's signed 32-bit indices.
    return sum((size - 1) * stride for size, stride in zip(value.shape, value.stride())) < 2**31


@lru_cache(maxsize=1)
def _has_triton() -> bool:
    # Keep CPU-only use and installations without the optional package working.
    return importlib.util.find_spec("triton") is not None


@lru_cache(maxsize=None)
def _supports_bfloat16(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device)[0] >= 8


def can_fuse_residual_gate_add(
    residual: Tensor, update: Tensor, gate: Tensor,
    tokens_per_frame: Optional[int] = None,
) -> bool:
    """Whether inputs satisfy the deliberately narrow inference contract.

    The caller must additionally enforce its eval/optimization-switch policy.
    Unsupported layouts, precision, broadcasting, or enabled gradients retain
    the caller's eager path; no full-tensor copy or precision change is hidden
    in this predicate. Dense and compact per-frame gates are both supported.
    """
    # Tensor subclasses can override multiply/add via dispatch. Bypassing that
    # behavior with raw pointers would change their semantics, even in eval.
    if any(type(value) is not Tensor for value in (residual, update, gate)):
        return False
    # Sparse/nested tensors do not expose the strided storage assumed below.
    # Lazy negation/conjugation also changes values without changing storage;
    # a raw-pointer kernel must not ignore that metadata or materialize a copy.
    if any(
        value.layout != torch.strided or value.is_nested or value.is_neg() or value.is_conj()
        for value in (residual, update, gate)
    ):
        return False
    if torch.is_grad_enabled() or not residual.is_cuda or torch.version.hip is not None:
        return False
    if residual.dtype != torch.bfloat16 or residual.ndim != 3:
        return False
    batch, tokens, channels = residual.shape
    if (min(batch, tokens, channels) <= 0 or channels > 4096 or batch > 65535
            or batch * tokens * channels >= 2**31):
        return False
    if (residual.stride(1) != 1 or residual.stride(2) != tokens
            or residual.stride(0) < tokens * channels):
        return False
    if update.shape != residual.shape:
        return False
    for value in (residual, update, gate):
        if (value.device != residual.device or value.dtype != torch.bfloat16
                or any(stride < 0 for stride in value.stride())
                or not _fits_index_span(value)):
            return False
    if tokens_per_frame is not None:
        if (isinstance(tokens_per_frame, bool) or not isinstance(tokens_per_frame, int)
                or tokens_per_frame <= 0 or tokens % tokens_per_frame):
            return False
    if _gate_layout(gate, batch, tokens, channels, tokens_per_frame) is None:
        return False
    return _has_triton() and _supports_bfloat16(residual.device)


@lru_cache(maxsize=1)
def _kernel():
    # Import Triton only after an eligible CUDA call actually requests fusion.
    import triton
    import triton.language as tl

    @triton.jit
    def residual_gate_add(
        X, UPDATE, GATE, OUT,
        N: tl.constexpr, D: tl.constexpr, TOKENS_PER_ROW: tl.constexpr,
        XB: tl.constexpr, UB: tl.constexpr, UN: tl.constexpr, UD: tl.constexpr,
        GB: tl.constexpr, GR: tl.constexpr, GD: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        batch = tl.program_id(2)
        tokens = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        channels = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        valid = (tokens[:, None] < N) & (channels[None, :] < D)
        # The residual is channel-major and the normal projection output is
        # row-major. A two-dimensional tile reads both real layouts directly.
        residual = tl.load(
            X + batch * XB + tokens[:, None] + channels[None, :] * N,
            valid, other=0,
        ).to(tl.float32)
        update = tl.load(
            UPDATE + batch * UB + tokens[:, None] * UN + channels[None, :] * UD,
            valid, other=0,
        ).to(tl.float32)
        gate = tl.load(
            GATE + batch * GB + (tokens[:, None] // TOKENS_PER_ROW) * GR
            + channels[None, :] * GD,
            valid, other=0,
        ).to(tl.float32)
        # Eager multiplication writes BF16 before the separate residual add.
        # This explicit narrowing is essential: x + gate * update evaluated
        # entirely in FP32 (or as one FMA) would change the model arithmetic.
        product = (gate * update).to(tl.bfloat16).to(tl.float32)
        result = residual + product
        tl.store(
            OUT + batch * N * D + tokens[:, None] + channels[None, :] * N,
            result, valid,
        )

    return triton, residual_gate_add


def fused_residual_gate_add(
    residual: Tensor, update: Tensor, gate: Tensor,
    tokens_per_frame: Optional[int] = None,
) -> Tensor:
    """Return a fresh BF16 channel-major result without changing any inputs.

    Only the gate/update temporary is removed. BF16 multiplication rounding
    precedes BF16 addition rounding, as in the eager expression. There are no
    persistent buffers, in-place writes, cache ownership, or cross-stream state.
    Unsupported calls must use the caller's eager implementation; compilation
    and execution errors are not silently hidden behind a fallback.
    """
    if not can_fuse_residual_gate_add(residual, update, gate, tokens_per_frame):
        raise ValueError("Unsupported fused residual gate/add input; use the eager path")
    batch, tokens, channels = residual.shape
    gate_layout = _gate_layout(gate, batch, tokens, channels, tokens_per_frame)
    triton, kernel = _kernel()
    # Launch on the tensor's device/current stream, restoring the caller's
    # current device afterward. Never rely on Triton to infer the device.
    with torch.cuda.device(residual.device):
        output = torch.empty_strided(
            residual.shape, (tokens * channels, 1, tokens),
            device=residual.device, dtype=torch.bfloat16,
        )
        kernel[(triton.cdiv(tokens, 32), triton.cdiv(channels, 32), batch)](
            residual, update, gate, output, tokens, channels, tokens_per_frame or 1,
            residual.stride(0), update.stride(0), update.stride(1), update.stride(2),
            *gate_layout, 32, 32, num_warps=4, enable_fp_fusion=False,
        )
    return output
