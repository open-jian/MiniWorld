"""Inference-only, resident KV storage for one streaming rollout / CFG branch.

Each layer owns a fixed arena laid out as
``[committed sink + history | current forward's scratch KV | unused]``.
Only ``length`` tokens are committed. Denoising overwrites scratch in place;
only an explicit commit after a successful clean (t=0) forward promotes it.
This removes repeated history concatenation, not attention's history reads,
the attention dtype conversion, or the original eviction / RoPE work.

An instance belongs to one model, batch, device, precision and execution stream.
Do not share it between CFG branches, simultaneous rollouts or CUDA streams.
Prefix views alias mutable storage: consume them before the next mutation, and
clone explicitly if a diagnostic needs a persistent snapshot. This is not an
nn.Module and is deliberately absent from model weights / checkpoints.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor


class ResidentKVCache:
    def __init__(self, depth: int, capacity_tokens: int) -> None:
        if depth <= 0 or capacity_tokens <= 0:
            raise ValueError("depth and capacity_tokens must be positive")
        self.depth = int(depth)
        self.capacity_tokens = int(capacity_tokens)
        self._buffers: List[Optional[Tuple[Tensor, Tensor]]] = [None] * self.depth
        # One reusable V scratch arena is enough: layers are compacted serially
        # on the same stream. K already has an independent RoPE result tensor.
        self._value_scratch: Optional[Tensor] = None
        self._length = 0
        self._invalidate_stage()

    @property
    def length(self) -> int:
        """Committed token count, INCLUDING sink, excluding scratch / padding."""
        return self._length

    def _invalidate_stage(self) -> None:
        self._forward_tokens: Optional[int] = None
        self._allow_commit = False
        self._complete = False
        self._written = [False] * self.depth

    @staticmethod
    def _require_inference() -> None:
        if torch.is_grad_enabled():
            raise RuntimeError("ResidentKVCache requires torch.no_grad() or inference_mode()")

    def reset(self) -> None:
        """Start another same-shape rollout without reallocating or zeroing.

        Uninitialized / stale tokens are never included in a returned prefix.
        Construct a new cache to change batch, head shape, dtype or device.
        """
        self._length = 0
        self._invalidate_stage()

    def discard_pending(self) -> None:
        """Invalidate scratch without touching committed history.

        Model entry calls this before input validation / projection so even an
        early failed forward cannot leave a previous clean stage committable.
        """
        self._invalidate_stage()

    def begin_forward(self, n_current: int, *, allow_commit: bool = False) -> None:
        # Invalidate first, including on a failed new attempt. A previous clean
        # forward must not remain accidentally committable after another call.
        self._invalidate_stage()
        self._require_inference()
        if n_current <= 0 or self.length + n_current > self.capacity_tokens:
            raise ValueError(
                f"KV capacity exceeded or empty forward: history={self.length}, "
                f"current={n_current}, capacity={self.capacity_tokens}"
            )
        self._forward_tokens = int(n_current)
        self._allow_commit = bool(allow_commit)

    def write_layer(self, layer_idx: int, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Write ONLY current KV and return history+current as prefix views."""
        self._require_inference()
        if self._forward_tokens is None or self._complete:
            raise RuntimeError("begin_forward() must precede layer writes")
        if not 0 <= layer_idx < self.depth or self._written[layer_idx]:
            raise ValueError("invalid or duplicate cache layer write")
        if (
            k.ndim != 4 or k.shape != v.shape or k.device != v.device
            or k.shape[-2] != self._forward_tokens
        ):
            raise ValueError("expected matching (B, heads, current_tokens, D) K/V on one device")

        buffers = self._buffers[layer_idx]
        if buffers is None:
            shape = (*k.shape[:2], self.capacity_tokens, k.shape[-1])
            # Allocate from the actual projected tensors, NOT the model/input
            # dtype: QK norm / RoPE currently produce FP32 K while V is BF16.
            buffers = (k.new_empty(shape), v.new_empty(shape))
            self._buffers[layer_idx] = buffers
        k_buffer, v_buffer = buffers
        for buffer, current in zip(buffers, (k, v)):
            if (
                buffer.shape[:2] != current.shape[:2]
                or buffer.shape[-1] != current.shape[-1]
                or buffer.dtype != current.dtype or buffer.device != current.device
            ):
                raise ValueError("cache batch/head shape, dtype and device must remain fixed")

        end = self.length + self._forward_tokens
        k_buffer[..., self.length:end, :].copy_(k)
        v_buffer[..., self.length:end, :].copy_(v)
        self._written[layer_idx] = True
        # Keep the capacity-based strides. Calling contiguous() here would
        # silently reintroduce a complete history copy on every forward.
        return k_buffer[..., :end, :], v_buffer[..., :end, :]

    def finish_forward(self) -> None:
        """Call only after the ENTIRE model forward has returned successfully."""
        if self._forward_tokens is None or self._complete or not all(self._written):
            raise RuntimeError("cannot finish an incomplete or already finished KV forward")
        self._complete = True

    def commit(self) -> None:
        """Promote a successful explicit KV-producing forward, without copying.

        The caller must use the original t=0 commit forward, not the last noisy
        denoising forward. For CFG, finish both branches at the OLD position
        before committing either branch. No tensor views are returned here.
        """
        if not self._complete or not self._allow_commit or self._forward_tokens is None:
            raise RuntimeError("commit requires a completed KV-producing forward")
        self._length += self._forward_tokens
        self._invalidate_stage()

    def history(self) -> List[Optional[Tuple[Tensor, Tensor]]]:
        """Borrow committed prefix views for inspection; they are NOT snapshots."""
        return [
            None if kv is None else (kv[0][..., :self.length, :], kv[1][..., :self.length, :])
            for kv in self._buffers
        ]

    def evict_and_shift(
        self, drop_frames: int, tokens_per_frame: int, rope_module, sink_frames: int = 0,
    ) -> None:
        """Drop oldest post-sink frames, keeping the original RoPE arithmetic.

        This intentionally still moves surviving history ONCE at eviction.
        A ring buffer alone would not preserve the existing positional rebase,
        and rebuilding a contiguous ring view would bring back the copies.
        If compaction raises, the cache is empty and must be prefilled again;
        partially moved layers must never be reused as committed history.
        """
        self._require_inference()
        if drop_frames < 0 or sink_frames < 0 or tokens_per_frame <= 0:
            raise ValueError("invalid eviction frame counts")
        if drop_frames == 0:
            return
        sink_tokens = sink_frames * tokens_per_frame
        drop_tokens = drop_frames * tokens_per_frame
        if sink_tokens + drop_tokens > self.length:
            raise ValueError("eviction would extend beyond committed post-sink history")
        self._invalidate_stage()
        old_length = self.length
        new_length = old_length - drop_tokens
        remaining = new_length - sink_tokens
        # Publish the new length only after every layer succeeds. On exception
        # the zero length hides any partially rebased / moved data, rather than
        # allowing a caller to retry attention with inconsistent layer positions.
        self._length = 0
        for kv in self._buffers:
            if kv is None:
                raise RuntimeError("committed cache has an uninitialized layer")
            k_buffer, v_buffer = kv
            if remaining:
                src = slice(sink_tokens + drop_tokens, old_length)
                dst = slice(sink_tokens, new_length)
                # The original RoPE routine is out-of-place. Preserve its exact
                # dtype/rounding; never rotate the fixed sink. Its output also
                # makes the subsequent overlapping K compaction safe.
                shifted = rope_module.rope_shift_time(-drop_frames, k_buffer[..., src, :])
                k_buffer[..., dst, :].copy_(shifted)
                scratch = self._value_scratch
                if (
                    scratch is None or scratch.shape != v_buffer.shape
                    or scratch.dtype != v_buffer.dtype or scratch.device != v_buffer.device
                ):
                    scratch = torch.empty_like(v_buffer)
                    self._value_scratch = scratch
                temporary = scratch[..., :remaining, :]
                # copy_(overlapping views) is NOT memmove, on CPU or CUDA.
                temporary.copy_(v_buffer[..., src, :])
                v_buffer[..., dst, :].copy_(temporary)
        self._length = new_length
