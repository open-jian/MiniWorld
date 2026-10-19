"""Fused current-QKV head packing for heterogeneous inference attention."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


class CurrentPackWorkspace:
    """Scratch owned by one request and shared by its sequential layer calls.

    Each CUDA stream gets a separate buffer, preventing simultaneous streams
    from overwriting one another. Sharing is safe after attention consuming
    the previous pack has been enqueued on that same stream.
    """
    def __init__(self):
        self.buffers = {}
        self.fused_calls = 0
        self.fallback_calls = 0

    def get(self, q, dtype):
        stream = torch.cuda.current_stream(q.device).cuda_stream if q.is_cuda else None
        key = (q.device, dtype, stream)
        needed = 3*q.numel()
        buffer = self.buffers.get(key)
        if buffer is None or buffer.numel() < needed:
            buffer = torch.empty(needed, device=q.device, dtype=dtype)
            self.buffers[key] = buffer
        return buffer[:needed].view(3, *q.shape)


if triton is not None:
    @triton.jit
    def _pack_qkv_kernel(Q, K, V, HEADS, OUT,
                         TOTAL: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
                         QB: tl.constexpr, QH: tl.constexpr, QN: tl.constexpr, QD: tl.constexpr,
                         KB: tl.constexpr, KH: tl.constexpr, KN: tl.constexpr, KD: tl.constexpr,
                         VB: tl.constexpr, VH: tl.constexpr, VN: tl.constexpr, VD: tl.constexpr,
                         BLOCK: tl.constexpr):
        ids = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
        valid = ids < TOTAL
        d = ids%D
        n = (ids//D)%N
        packed_head = (ids//(D*N))%H
        b = ids//(D*N*H)
        source_head = tl.load(HEADS+packed_head, mask=valid, other=0)
        q = tl.load(Q+b*QB+source_head*QH+n*QN+d*QD, mask=valid, other=0)
        k = tl.load(K+b*KB+source_head*KH+n*KN+d*KD, mask=valid, other=0)
        v = tl.load(V+b*VB+source_head*VH+n*VN+d*VD, mask=valid, other=0)
        tl.store(OUT+ids, q, mask=valid)
        tl.store(OUT+TOTAL+ids, k, mask=valid)
        tl.store(OUT+2*TOTAL+ids, v, mask=valid)


@torch.no_grad()
def pack_current_qkv(q, k, v, heads, workspace, dtype):
    """One pass gathers all groups and casts actual FA inputs to its dtype."""
    packed = workspace.get(q, dtype)
    if triton is not None and q.is_cuda:
        _pack_qkv_kernel[(triton.cdiv(q.numel(), 1024),)](
            q, k, v, heads, packed, q.numel(), q.shape[1], q.shape[2], q.shape[3],
            *q.stride(), *k.stride(), *v.stride(), BLOCK=1024, num_warps=4)
        workspace.fused_calls += 1
    else:
        for target, source in zip(packed, (q, k, v)):
            target.copy_(source.index_select(1, heads))
        workspace.fallback_calls += 1
    return packed.unbind(0)
