"""Inference-only post-GEMM bias and SwiGLU fusion for BF16 projections.

The W12 matrix multiplication stays separate. Its BF16 result receives BF16
bias, rounds back to BF16, and then passes through SiLU and multiplication.
Keeping those intermediate roundings is necessary for the original arithmetic;
adding bias to a GEMM's FP32 accumulator would implement a different operation.
"""

from functools import lru_cache
import importlib.util

import torch
from torch import Tensor


def _fits_index_span(value: Tensor) -> bool:
    # Tensor pointers already incorporate storage_offset. Reject strides and
    # reachable relative offsets outside signed 32-bit kernel indexing.
    strides = value.stride()
    return (
        all(0 <= stride < 2**31 for stride in strides)
        and sum((size - 1) * stride for size, stride in zip(value.shape, strides)) < 2**31
    )


@lru_cache(maxsize=1)
def _has_triton() -> bool:
    # Keep CPU imports and installations without the optional package working.
    return importlib.util.find_spec("triton") is not None


@lru_cache(maxsize=None)
def _supports_device(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device)[0] >= 8


def can_fuse_bias_swiglu_input(x: Tensor, weight: Tensor, bias: Tensor) -> bool:
    """Check tensor eligibility BEFORE replacing the original biased Linear.

    Only the observed batch-one, channel-major FP32 input under CUDA BF16
    autocast is accepted. A contiguous input can select an addmm bias epilogue
    with different rounding, so it must not be sent through this optimization.
    The caller additionally checks module hooks/overrides, eval policy, and
    environment options that could force a different Linear implementation.
    """
    if (type(x) is not Tensor
            or type(weight) not in (Tensor, torch.nn.Parameter)
            or type(bias) not in (Tensor, torch.nn.Parameter)):
        return False
    values = (x, weight, bias)
    if any(
        value.layout != torch.strided or value.is_nested or value.is_neg() or value.is_conj()
        for value in values
    ):
        return False
    if torch.is_grad_enabled() or not x.is_cuda or torch.version.hip is not None:
        return False
    if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16:
        return False
    if x.dtype != torch.float32 or x.ndim != 3:
        return False
    batch, tokens, channels = x.shape
    if batch != 1 or tokens <= 1 or channels <= 1:
        return False
    if (x.stride(0) < tokens * channels or x.stride(1) != 1
            or x.stride(2) != tokens or x.is_contiguous()):
        return False
    if weight.ndim != 2 or weight.shape[1] != channels:
        return False
    doubled_hidden = weight.shape[0]
    if doubled_hidden <= 0 or doubled_hidden % 2 or tokens * doubled_hidden >= 2**31:
        return False
    if bias.shape != (doubled_hidden,):
        return False
    if any(value.dtype != torch.float32 or value.device != x.device or not _fits_index_span(value)
           for value in values):
        return False
    if not weight.is_contiguous() or not bias.is_contiguous():
        return False
    return _has_triton() and _supports_device(x.device)


def can_fuse_bias_swiglu(projection: Tensor, bias: Tensor) -> bool:
    """Whether a raw W12 projection satisfies the narrow inference contract.

    The caller must check eval policy and GEMM eligibility BEFORE omitting the
    Linear bias. This predicate validates its BF16 contiguous B,N,2H output and
    the original FP32 bias parameter. It never copies inputs or changes dtype.
    """
    # Native Parameter is the expected bias type. Other Tensor subclasses may
    # override dispatch and must not be bypassed by raw-pointer arithmetic.
    if type(projection) is not Tensor or type(bias) not in (Tensor, torch.nn.Parameter):
        return False
    values = (projection, bias)
    if any(
        value.layout != torch.strided or value.is_nested or value.is_neg() or value.is_conj()
        for value in values
    ):
        return False
    if torch.is_grad_enabled() or not projection.is_cuda or torch.version.hip is not None:
        return False
    if projection.dtype != torch.bfloat16 or projection.ndim != 3:
        return False
    batch, tokens, doubled_hidden = projection.shape
    if (min(batch, tokens, doubled_hidden) <= 0 or doubled_hidden % 2
            or projection.numel() >= 2**31):
        return False
    if bias.shape != (doubled_hidden,) or bias.dtype != torch.float32:
        return False
    if any(
        value.device != projection.device or not value.is_contiguous() or not _fits_index_span(value)
        for value in values
    ):
        return False
    return _has_triton() and _supports_device(projection.device)


@lru_cache(maxsize=1)
def _kernel():
    # Delay optional imports and compilation until an eligible CUDA invocation.
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def bias_swiglu(
        PROJECTION, BIAS, OUT, TOTAL: tl.constexpr, H: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < TOTAL
        rows = offsets // H
        channels = offsets % H
        first = tl.load(PROJECTION + rows * (2 * H) + channels, valid, other=0).to(tl.float32)
        second = tl.load(PROJECTION + rows * (2 * H) + H + channels, valid, other=0).to(tl.float32)
        # Linear autocast originally converts FP32 bias to BF16 BEFORE add_.
        first_bias = tl.load(BIAS + channels, valid, other=0).to(tl.bfloat16).to(tl.float32)
        second_bias = tl.load(BIAS + H + channels, valid, other=0).to(tl.bfloat16).to(tl.float32)
        first = (first + first_bias).to(tl.bfloat16).to(tl.float32)
        second = (second + second_bias).to(tl.bfloat16).to(tl.float32)
        denominator = 1.0 + libdevice.exp(-first)
        # Preserve FP32 subnormals: libdevice.div_rn lowered to a flushing
        # div.rn.ftz on the measured GPU. Explicit scalar PTX has no FTZ suffix.
        activated = tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;", constraints="=f,f,f",
            args=[first, denominator], dtype=tl.float32, is_pure=True, pack=1,
        )
        # Eager SiLU materializes BF16 before the separate BF16 multiplication.
        activated = activated.to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + offsets, activated * second, valid)

    return triton, bias_swiglu


def fused_bias_swiglu(projection: Tensor, bias: Tensor) -> Tensor:
    """Return fresh contiguous BF16 B,N,H output without modifying inputs.

    No bias or activation buffer persists between calls. Unsupported inputs
    must use the caller's original path; compilation and execution failures
    are not hidden by retrying a different implementation.
    """
    if not can_fuse_bias_swiglu(projection, bias):
        raise ValueError("Unsupported fused bias/SwiGLU input; use the eager path")
    batch, tokens, doubled_hidden = projection.shape
    hidden = doubled_hidden // 2
    triton, kernel = _kernel()
    # Use the input device's current stream and restore the caller's device.
    with torch.cuda.device(projection.device):
        output = torch.empty((batch, tokens, hidden), device=projection.device, dtype=torch.bfloat16)
        kernel[(triton.cdiv(output.numel(), 1024),)](
            projection, bias, output, output.numel(), hidden,
            BLOCK=1024, num_warps=4, enable_fp_fusion=False,
        )
    return output
