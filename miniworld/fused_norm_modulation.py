"""Inference-only RMSNorm/AdaLN kernels for channel-major video tokens.

The supported residual layout is ``(B, N, D)`` with adjacent tokens in memory,
as produced by MiniWorld's patch embedding. No full-tensor ``contiguous`` copy
is made. Unsupported calls should use the eager model path instead; this module
does not conceal compilation or execution failures behind a silent fallback.
"""

from functools import lru_cache
import importlib.util
import math
from typing import Optional

import torch
from torch import Tensor


def _modulation_layout(
    value: Tensor, batch: int, tokens: int, channels: int,
    tokens_per_frame: Optional[int],
) -> Optional[tuple[int, int, int]]:
    """Return batch/frame-or-token/channel strides without expanding values."""
    if any(stride < 0 for stride in value.stride()):
        return None
    if tokens_per_frame is not None:
        if value.ndim != 4 or value.shape[2] != 1:
            return None
        rows = tokens // tokens_per_frame
        if value.shape[1] not in (1, rows):
            return None
        row_stride = value.stride(1) if value.shape[1] != 1 else 0
    elif value.ndim == 3:
        if value.shape[1] not in (1, tokens):
            return None
        row_stride = value.stride(1) if value.shape[1] != 1 else 0
    elif value.ndim == 2:
        row_stride = 0
    else:
        return None
    if value.shape[0] not in (1, batch) or value.shape[-1] != channels:
        return None
    batch_stride = value.stride(0) if value.shape[0] != 1 else 0
    return batch_stride, row_stride, value.stride(-1)


@lru_cache(maxsize=1)
def _has_triton() -> bool:
    # Discovering the optional package does not import Triton or initialize CUDA.
    return importlib.util.find_spec("triton") is not None


@lru_cache(maxsize=None)
def _supports_bfloat16(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device)[0] >= 8


def _fits_index_span(value: Tensor) -> bool:
    # Kernel offsets use signed 32-bit arithmetic. Storage offsets are already
    # incorporated in tensor pointers; only the reachable relative span matters.
    return sum((size - 1) * stride for size, stride in zip(value.shape, value.stride())) < 2**31


def can_fuse_norm_modulation(
    x: Tensor, weight: Tensor, shift: Optional[Tensor], scale: Tensor,
    conditioning_tokens_per_frame: Optional[int] = None,
) -> bool:
    """Whether this call has the explicitly supported inference contract.

    This is deliberately narrower than generic RMSNorm: training, non-NVIDIA
    devices, other residual dtypes/layouts, and other weight dtypes retain their
    existing eager behavior. Modulation may be dense or compact per frame, and
    its non-contiguous/zero strides are respected rather than materialized.
    """
    if torch.is_grad_enabled() or not x.is_cuda or torch.version.hip is not None:
        return False
    if x.dtype != torch.bfloat16 or x.ndim != 3:
        return False
    batch, tokens, channels = x.shape
    # CUDA grid.y is limited to 65535 and all output/input offsets must fit the
    # kernels' signed 32-bit indexing. Large or padded views use the eager path.
    if (min(batch, tokens, channels) <= 0 or channels > 4096 or batch > 65535
            or batch * tokens * channels >= 2**31 or not _fits_index_span(x)):
        return False
    if x.stride(1) != 1 or x.stride(2) != tokens or x.stride(0) < tokens * channels:
        return False
    if weight.device != x.device or weight.dtype != torch.float32:
        return False
    if weight.shape != (channels,) or weight.stride(0) <= 0:
        return False
    if not _fits_index_span(weight):
        return False
    if conditioning_tokens_per_frame is not None:
        if (isinstance(conditioning_tokens_per_frame, bool)
                or not isinstance(conditioning_tokens_per_frame, int)
                or conditioning_tokens_per_frame <= 0
                or tokens % conditioning_tokens_per_frame):
            return False
    for value in (scale, shift):
        if value is None:
            continue
        if value.device != x.device or value.dtype not in (torch.bfloat16, torch.float32):
            return False
        if _modulation_layout(value, batch, tokens, channels, conditioning_tokens_per_frame) is None:
            return False
        if not _fits_index_span(value):
            return False
    return _has_triton() and _supports_bfloat16(x.device)


@lru_cache(maxsize=1)
def _kernels():
    # Lazy imports keep CPU use and installations without Triton working.
    import triton
    import triton.language as tl

    @triton.jit
    def inverse_rms(
        X, INV, N: tl.constexpr, D: tl.constexpr, XB: tl.constexpr,
        EPS: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        batch = tl.program_id(1)
        tokens = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        channels = tl.arange(0, BLOCK_D)
        # A tile spans adjacent tokens, so the channel-major input is read in
        # its real layout. Only one FP32 inverse RMS per token is materialized.
        values = tl.load(
            X + batch * XB + channels[:, None] * N + tokens[None, :],
            (channels[:, None] < D) & (tokens[None, :] < N), other=0,
        ).to(tl.float32)
        mean_square = tl.sum(values * values, axis=0) / D
        inverse = tl.rsqrt(mean_square + EPS)
        tl.store(INV + batch * N + tokens, inverse, tokens < N)

    @triton.jit
    def apply_norm_modulation(
        X, W, SHIFT, SCALE, INV, OUT,
        N: tl.constexpr, D: tl.constexpr, XB: tl.constexpr, WS: tl.constexpr,
        SCALE_B: tl.constexpr, SCALE_R: tl.constexpr, SCALE_D: tl.constexpr,
        SHIFT_B: tl.constexpr, SHIFT_R: tl.constexpr, SHIFT_D: tl.constexpr,
        TOKENS_PER_ROW: tl.constexpr, HAS_SHIFT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(1)
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < N * D
        tokens = offsets % N
        channels = offsets // N
        rows = tokens // TOKENS_PER_ROW
        values = tl.load(X + batch * XB + offsets, valid, other=0).to(tl.float32)
        inverse = tl.load(INV + batch * N + tokens, valid, other=0)
        weights = tl.load(W + channels * WS, valid, other=0)
        scales = tl.load(
            SCALE + batch * SCALE_B + rows * SCALE_R + channels * SCALE_D,
            valid, other=0,
        )
        # These casts are part of the eager numerical contract. RMSNorm rounds
        # its normalized result to BF16 BEFORE multiplying its FP32 weight.
        normalized = (values * inverse).to(tl.bfloat16).to(tl.float32)
        weighted = normalized * weights
        # Likewise, (1 + scale) rounds to the scale's original dtype before
        # modulation. Omitting either narrowing changes the model arithmetic.
        factor = (1.0 + scales.to(tl.float32)).to(scales.dtype).to(tl.float32)
        if HAS_SHIFT:
            shifts = tl.load(
                SHIFT + batch * SHIFT_B + rows * SHIFT_R + channels * SHIFT_D,
                valid, other=0,
            ).to(tl.float32)
            # Scalar PTX with explicit rounding also prevents the Blackwell
            # packed-FP32 instruction path from contracting these operations.
            # enable_fp_fusion=False alone did not retain the eager rounding
            # boundary for this multiply/add pair on the profiled SM120 GPU.
            result = tl.inline_asm_elementwise(
                "{ .reg .f32 product; mul.rn.f32 product, $1, $2; "
                "add.rn.f32 $0, product, $3; }",
                constraints="=f,f,f,f", args=[weighted, factor, shifts],
                dtype=tl.float32, is_pure=True, pack=1,
            )
        else:
            result = weighted * factor
        tl.store(OUT + batch * N * D + offsets, result, valid)

    return triton, inverse_rms, apply_norm_modulation


def fused_norm_modulation(
    x: Tensor, weight: Tensor, shift: Optional[Tensor], scale: Tensor,
    eps: float = 1e-6,
    conditioning_tokens_per_frame: Optional[int] = None,
) -> Tensor:
    """Fuse the supported eager RMSNorm and AdaLN operations into two kernels.

    The result is FP32 with a channel-major ``(B, N, D)`` layout. FP32 reduction
    order can differ from PyTorch's mean, so this is not a bitwise equivalence
    guarantee; the explicit BF16 rounding boundaries and epsilon are preserved.
    No input is modified and no temporary buffer is shared across calls/streams.
    """
    if not can_fuse_norm_modulation(x, weight, shift, scale, conditioning_tokens_per_frame):
        raise ValueError("Unsupported fused RMSNorm/AdaLN input; use the eager path")
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm epsilon must be a finite positive number")
    batch, tokens, channels = x.shape
    scale_layout = _modulation_layout(scale, batch, tokens, channels, conditioning_tokens_per_frame)
    shift_layout = ((0, 0, 0) if shift is None else
                    _modulation_layout(shift, batch, tokens, channels, conditioning_tokens_per_frame))
    triton, inverse_rms, apply_norm_modulation = _kernels()
    # Triton launches on the current device/stream, rather than deriving these
    # from its tensor arguments. Respect x's device and restore the caller's.
    with torch.cuda.device(x.device):
        inverse = torch.empty((batch, tokens), device=x.device, dtype=torch.float32)
        output = torch.empty_strided(
            x.shape, (tokens * channels, 1, tokens), device=x.device, dtype=torch.float32,
        )
        # Avoid FMA contraction: eager multiply and add are separate FP32 operations.
        # The two kernels avoid full-size intermediate tensors and a layout copy.
        inverse_rms[(triton.cdiv(tokens, 16), batch)](
            x, inverse, tokens, channels, x.stride(0), eps,
            triton.next_power_of_2(channels), 16, num_warps=8, enable_fp_fusion=False,
        )
        apply_norm_modulation[(triton.cdiv(tokens * channels, 1024), batch)](
            x, weight, scale if shift is None else shift, scale, inverse, output,
            tokens, channels, x.stride(0), weight.stride(0),
            *scale_layout, *shift_layout, conditioning_tokens_per_frame or 1,
            shift is not None, 1024, num_warps=4, enable_fp_fusion=False,
        )
    return output
