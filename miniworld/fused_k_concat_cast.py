"""Inference-only stateless FP32 K concatenation and BF16 attention cast.

The canonical history and current tensors stay FP32. This removes only the
intermediate FP32 concatenation: every visible history element is still read
and a fresh BF16 attention buffer is still written on each call.
"""

from functools import lru_cache
import importlib.util

import torch
from torch import Tensor


def can_fuse_k_concat_cast(past: Tensor, current: Tensor) -> bool:
    """Check the narrow NVIDIA, no-grad, FP32 BHND tensor contract.

    The caller additionally enforces eval, BF16 autocast, and cached chunkwise
    FlashAttention policy. No input is converted or copied by this predicate."""
    values = (past, current)
    if torch.is_grad_enabled() or torch.version.hip is not None:
        return False
    if any(type(value) is not Tensor for value in values):
        return False
    if any(value.layout != torch.strided or value.is_nested
           or value.is_neg() or value.is_conj() for value in values):
        return False
    if any(not value.is_cuda or value.dtype != torch.float32 or value.ndim != 4
           for value in values):
        return False
    if past.device != current.device:
        return False
    b, h, n, d = current.shape
    if b != 1 or h != 12 or n <= 0 or d != 128:
        return False
    if past.shape[:2] != (b, h) or past.shape[-1] != d:
        return False
    if b * h * (past.shape[2] + n) * d >= 2**31:
        return False
    for value in values:
        strides = value.stride()
        if (strides[-1] != 1 or any(s < 0 or s >= 2**31 for s in strides)
                or sum(max(0, size - 1) * stride
                       for size, stride in zip(value.shape, strides)) >= 2**31):
            return False
    return _has_triton() and _supports_device(current.device)


@lru_cache(maxsize=1)
def _has_triton() -> bool:
    # Optional-package discovery does not import Triton or initialize CUDA.
    return importlib.util.find_spec("triton") is not None


@lru_cache(maxsize=None)
def _supports_device(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device)[0] >= 8


@lru_cache(maxsize=1)
def _kernel():
    # Import and compile lazily only after an eligible call requests fusion.
    import triton
    import triton.language as tl

    @triton.jit
    def concat_cast_kernel(
        PAST, CURRENT, OUT,
        NP: tl.constexpr, NC: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
        PH: tl.constexpr, PN: tl.constexpr, CH: tl.constexpr, CN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        channels = offsets % D
        tokens = offsets // D % (NP + NC)
        heads = offsets // (D * (NP + NC))
        valid = offsets < H * (NP + NC) * D
        is_past = tokens < NP
        history = tl.load(PAST + heads * PH + tokens * PN + channels,
                          valid & is_past, other=0)
        present = tl.load(CURRENT + heads * CH + (tokens - NP) * CN + channels,
                          valid & ~is_past, other=0)
        value = tl.where(is_past, history, present)
        # Triton performs the same round-to-nearest-even FP32 -> BF16 cast as
        # PyTorch. Explicit NaN canonicalization below preserves Torch bits.
        converted = value.to(tl.bfloat16)
        nan_bits = tl.full((BLOCK,), 0x7FFF, tl.uint16).to(tl.bfloat16, bitcast=True)
        converted = tl.where(value != value, nan_bits, converted)
        tl.store(OUT + offsets, converted, valid)

    return triton, concat_cast_kernel


def fused_k_concat_cast(past: Tensor, current: Tensor) -> Tensor:
    """Return fresh, BHND-contiguous BF16 K without modifying either input.

    The return layout exactly matches the old FP32 cat followed by its BF16
    cast. The existing FlashAttention helper transposes this to BNHD. There is
    no resident BF16 copy, cache invalidation, in-place write, or fallback that
    could hide a failed GPU kernel. Guard failures belong in the caller.
    """
    if not can_fuse_k_concat_cast(past, current):
        raise ValueError("Unsupported fused K concatenation/cast input; use the eager path")
    b, h, nc, d = current.shape
    np = past.shape[2]
    triton, kernel = _kernel()
    with torch.cuda.device(current.device):
        out = torch.empty((b, h, np + nc, d), device=current.device,
                          dtype=torch.bfloat16)
        kernel[(triton.cdiv(out.numel(), 1024),)](
            past, current, out, np, nc, h, d,
            past.stride(1), past.stride(2),
            current.stride(1), current.stride(2),
            BLOCK=1024, num_warps=4,
        )
    return out
