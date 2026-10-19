"""Inference-only RoPE arithmetic for normalized FP32 video Q/K tensors.

The existing coefficient tensors are consumed without rebuilding or caching
them. Only ``t * cos + rotate_half(t) * sin`` is fused: QK normalization,
attention casts, canonical FP32 K, and cache eviction remain separate paths.
Unsupported inputs must retain the caller's eager implementation.
"""

from functools import lru_cache
import importlib.util

import torch
from torch import Tensor


def _fits_index_span(value: Tensor) -> bool:
    # Tensor pointers already include storage_offset. Bound every stride and
    # reachable relative offset for the kernel's signed 32-bit indexing.
    strides = value.stride()
    return (
        all(0 <= stride < 2**31 for stride in strides)
        and sum((size - 1) * stride for size, stride in zip(value.shape, strides)) < 2**31
    )


@lru_cache(maxsize=1)
def _has_triton() -> bool:
    # Optional-package discovery does not import Triton or initialize CUDA.
    return importlib.util.find_spec("triton") is not None


@lru_cache(maxsize=None)
def _supports_device(device: torch.device) -> bool:
    # Match the NVIDIA generation required by the BF16 inference path.
    return torch.cuda.get_device_capability(device)[0] >= 8


def can_fuse_rope(t: Tensor, cos: Tensor, sin: Tensor) -> bool:
    """Whether inputs satisfy the deliberately narrow no-grad CUDA contract.

    The caller must also enforce its eval and optimization-switch policy.
    Normalized Q/K use logical B,H,N,D with token-major physical storage and
    head dimension 128. Existing BF16 dynamic or FP32 precomputed coefficients
    may have noncontiguous/zero strides; no contiguous copy is inserted.
    """
    values = (t, cos, sin)
    # Raw-pointer kernels cannot honor subclass dispatch or lazy value bits.
    # Check storage metadata before querying sparse/nested shapes or strides.
    if any(type(value) is not Tensor for value in values):
        return False
    if any(
        value.layout != torch.strided or value.is_nested or value.is_neg() or value.is_conj()
        for value in values
    ):
        return False
    if torch.is_grad_enabled() or not t.is_cuda or torch.version.hip is not None:
        return False
    if t.dtype != torch.float32 or t.ndim != 4:
        return False
    batch, heads, tokens, channels = t.shape
    if (min(batch, heads, tokens) <= 0 or channels != 128
            or batch * heads * tokens * channels >= 2**31):
        return False
    if (t.stride(0) < tokens * heads * channels or t.stride(1) != channels
            or t.stride(2) != heads * channels or t.stride(3) != 1):
        return False
    if cos.shape != (tokens, channels) or sin.shape != (tokens, channels):
        return False
    if cos.dtype != sin.dtype or cos.dtype not in (torch.bfloat16, torch.float32):
        return False
    if any(value.device != t.device or not _fits_index_span(value) for value in values):
        return False
    return _has_triton() and _supports_device(t.device)


@lru_cache(maxsize=1)
def _kernel():
    # Import and compile lazily only after an eligible call requests fusion.
    import triton
    import triton.language as tl

    @triton.jit
    def apply_rope(
        X, COS, SIN, OUT, TOTAL: tl.constexpr,
        N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
        XB: tl.constexpr, XH: tl.constexpr, XN: tl.constexpr,
        COS_N: tl.constexpr, COS_D: tl.constexpr,
        SIN_N: tl.constexpr, SIN_D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < TOTAL
        channels = offsets % D
        heads = offsets // D % H
        tokens = offsets // (D * H) % N
        batch = offsets // (D * H * N)
        source = batch * XB + heads * XH + tokens * XN
        value = tl.load(X + source + channels, valid, other=0)
        partner = tl.load(X + source + (channels ^ 1), valid, other=0)
        # MiniWorld pairs adjacent channels as [-odd, even], not two halves.
        rotated = tl.where(channels % 2 == 0, -partner, partner)
        cosine = tl.load(COS + tokens * COS_N + channels * COS_D, valid, other=0).to(tl.float32)
        sine = tl.load(SIN + tokens * SIN_N + channels * SIN_D, valid, other=0).to(tl.float32)
        # Both eager FP32 products round before the FP32 addition. Explicit
        # scalar PTX also prevents packed-FP32 contraction on Blackwell; the
        # compiler's enable_fp_fusion=False alone is not the precision contract.
        result = tl.inline_asm_elementwise(
            "{ .reg .f32 p0, p1; mul.rn.f32 p0, $1, $2; "
            "mul.rn.f32 p1, $3, $4; add.rn.f32 $0, p0, p1; }",
            constraints="=f,f,f,f,f", args=[value, cosine, rotated, sine],
            dtype=tl.float32, is_pure=True, pack=1,
        )
        tl.store(OUT + offsets, result, valid)

    return triton, apply_rope


def fused_rope(t: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Return a fresh FP32 rotation with eager-compatible output layout.

    There are no in-place writes, persistent tensor buffers, frequency caches,
    or new attention casts. Unsupported calls raise instead of changing their
    precision/layout; compilation and execution failures are never concealed
    behind an eager retry.
    """
    if not can_fuse_rope(t, cos, sin):
        raise ValueError("Unsupported fused RoPE input; use the eager path")
    batch, heads, tokens, channels = t.shape
    # Singleton axes have no physical indexing effect, but eager elementwise
    # outputs give them contiguous strides. Preserve that observable metadata.
    output_stride = (
        tokens * heads * channels,
        channels if heads > 1 else tokens * channels,
        heads * channels if tokens > 1 else channels,
        1,
    )
    triton, kernel = _kernel()
    # Use the input device's current stream and restore the caller's device.
    with torch.cuda.device(t.device):
        output = torch.empty_strided(t.shape, output_stride, device=t.device, dtype=torch.float32)
        kernel[(triton.cdiv(t.numel(), 1024),)](
            t, cos, sin, output, t.numel(), tokens, heads, channels,
            t.stride(0), t.stride(1), t.stride(2),
            cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1),
            BLOCK=1024, num_warps=4, enable_fp_fusion=False,
        )
    return output
