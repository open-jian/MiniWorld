"""Physical Forcing-KV history for MiniWorld's rolling diffusion sampler.

The official implementation selects low-cosine contiguous flattened segments
from three candidate frames, using layer 1 as a shared selector. MiniWorld
commits two frames at a time: candidates are the previous boundary frame and
all but the last new frame. The final new frame becomes the next boundary.
Previously selected segments are replaced, not recursively accumulated.

Selection is request/CFG owned, and each layer gathers its own K/V. Static
heads see sink + immediate predecessor + their current chunk; dynamic heads
see compact history + the legal current prefix. The latter is the adaptation
needed for MiniWorld's concurrently denoised chunks. Only clean cache updates
score/gather history. Denoising reuses packed history and fused current QKV.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from miniworld.kv_cache import CompactKVCache, flash_attn_with_kvcache
from miniworld.kv_kernels import CurrentPackWorkspace, pack_current_qkv


GROUPS = ("static", "dynamic")


@dataclass
class _SparseHistory:
    heads: torch.Tensor
    tokens: tuple[int, ...]
    k: torch.Tensor
    v: torch.Tensor
    version: int = 0


class ForcingKVCache(CompactKVCache):
    """One layer's independently owned sparse history and packed attention.

    ``selection_state`` must be a different dictionary for every request and
    CFG branch. Construct/update the selector before other dynamic layers.
    Scores are reduced with minimum across batch, so a change in any sample
    can retain a region. Only indices are shared: no cross-layer/batch KV.
    """

    @classmethod
    @torch.no_grad()
    def from_dense(cls, kv, groups, *, logical_frames, height, width, chunk_size,
                   sink_frames=1, segments=6, retention=.33, max_frames=48,
                   selection_state, selector_layer=False):
        self = cls()
        k, v = kv
        if k.ndim != 4 or k.shape != v.shape:
            raise ValueError("KV must have matching [B,H,N,D] shapes")
        if min(height, width, chunk_size, segments) <= 0:
            raise ValueError("grid, chunk, and segment dimensions must be positive")
        if min(logical_frames, sink_frames, max_frames) < 0:
            raise ValueError("frame counts must be nonnegative")
        if not 0 <= retention <= 1 or not math.isfinite(retention):
            raise ValueError("retention must be finite and in [0,1]")
        if not isinstance(selection_state, dict):
            raise TypeError("selection_state must be a request/CFG-owned dictionary")
        self.height, self.width = int(height), int(width)
        self.tpf = self.height*self.width
        self.segments, self.retention = int(segments), float(retention)
        if self.tpf % self.segments:
            raise ValueError("frame tokens must be divisible by segment count")
        self.segment_tokens = self.tpf//self.segments
        self.chunk_size, self.sink_frames = int(chunk_size), int(sink_frames)
        self.max_frames, self.logical_frames = int(max_frames), int(logical_frames)
        self.batch, self.num_heads, n, self.head_dim = k.shape
        self.device = k.device
        if n != self.logical_frames*self.tpf:
            raise ValueError("dense KV length differs from logical_frames * grid area")
        if set(groups) != set(GROUPS):
            raise ValueError(f"groups must have exactly {GROUPS}")
        self.groups = {name: torch.as_tensor(groups[name], device=k.device, dtype=torch.long)
                       for name in GROUPS}
        if any(x.ndim != 1 for x in self.groups.values()):
            raise ValueError("head indices must be one-dimensional")
        self._pack_heads = torch.cat(tuple(self.groups.values()))
        if sorted(self._pack_heads.cpu().tolist()) != list(range(self.num_heads)):
            raise ValueError("groups must be a disjoint exhaustive partition of heads")
        ns = self.groups['static'].numel()
        self._pack_slices = {'static': slice(0, ns), 'dynamic': slice(ns, self.num_heads)}
        self._current_workspace = CurrentPackWorkspace()
        self.fused_current_pack = True
        self._workspaces, self._metadata, self._histories = {}, {}, {}
        self.selection_state = selection_state
        self.selector_layer = bool(selector_layer)
        self._selection_seen = -1
        self._version = 0
        self._selector_recent = None
        self._selected_tokens = ()
        # Initial conversion uses the same short candidate horizon as later
        # updates; it does not select from the full previously dense history.
        candidate_frames = tuple(range(max(self.sink_frames, self.logical_frames-self.chunk_size-1),
                                       max(self.sink_frames, self.logical_frames-1)))
        candidate_tokens = tuple(f*self.tpf+p for f in candidate_frames for p in range(self.tpf))
        candidate_index = torch.tensor(candidate_tokens, device=k.device, dtype=torch.long)
        selected = self._select(k.index_select(2, candidate_index), k[:, :, -self.tpf:] if n else k,
                                candidate_frames, self.logical_frames, initial=True)
        selected_tokens = self._absolute_selection(candidate_frames, selected)
        sink_tokens = tuple(range(min(self.logical_frames, self.sink_frames)*self.tpf))
        recent_tokens = (tuple(range((self.logical_frames-1)*self.tpf, n))
                         if self.logical_frames > self.sink_frames else ())
        for name, heads in self.groups.items():
            if not heads.numel():
                continue
            tokens = sink_tokens + (selected_tokens if name == 'dynamic' else ()) + recent_tokens
            index = torch.tensor(tokens, device=k.device, dtype=torch.long)
            self._histories[name] = _SparseHistory(
                heads, tokens, k.index_select(1, heads).index_select(2, index).float(),
                v.index_select(1, heads).index_select(2, index))
        self._selected_tokens = selected_tokens if self.groups['dynamic'].numel() else ()
        if self.selector_layer and not self.groups['dynamic'].numel():
            self._selector_recent = k[:, :, -self.tpf:].float().clone() if n else k.float().clone()
        return self

    def _select(self, candidates, boundary, frames, generation, *, initial=False):
        """Select segments once at the proxy layer; other layers reuse indices."""
        count = len(frames)*self.segments
        keep = min(count, max(0, int(round(count*self.retention))))
        state = self.selection_state
        if self.selector_layer:
            heads = self.groups['dynamic']
            if heads.numel():
                candidates, boundary = candidates.index_select(1, heads), boundary.index_select(1, heads)
            if count:
                chain = torch.cat((candidates, boundary), dim=2).float()
                chain = (chain.reshape(self.batch, chain.shape[1], len(frames)+1,
                                       self.segments, self.segment_tokens, self.head_dim)
                         .permute(0, 2, 3, 4, 1, 5).flatten(-2))
                # Official score: mean token cosine between adjacent frames,
                # flattened over the proxy layer's dynamic head channels.
                scores = F.cosine_similarity(chain[:, :-1], chain[:, 1:], dim=-1).mean(-1).amin(0)
                indices = scores.flatten().topk(keep, largest=False).indices.sort().values
            else:
                scores = candidates.new_empty((0, self.segments), dtype=torch.float32)
                indices = torch.empty(0, device=self.device, dtype=torch.long)
            # One small synchronization per clean commit, not once per layer
            # or denoising call. CPU IDs support exact sparse RoPE bookkeeping.
            indices_cpu = tuple(indices.cpu().tolist())
            token_indices = (indices[:, None]*self.segment_tokens
                             + torch.arange(self.segment_tokens, device=self.device)[None, :]).flatten()
            state.update(generation=int(generation), candidate_frames=frames,
                         indices=indices.detach(), indices_cpu=indices_cpu,
                         token_indices=token_indices,
                         scores=scores.detach(), selection_updates=int(state.get('selection_updates', 0))+1,
                         batch_reduction='minimum', proxy_all_heads=not bool(heads.numel()))
        elif not self.groups['dynamic'].numel():
            return ()
        elif (state.get('generation') != generation or state.get('candidate_frames') != frames
              or (not initial and int(state.get('selection_updates', -1)) <= self._selection_seen)):
            raise RuntimeError("Forcing-KV selector layer must update before other dynamic layers")
        self._selection_seen = int(state['selection_updates'])
        selected = state['indices_cpu']
        if len(selected) != keep:
            raise RuntimeError("Forcing-KV layers disagree on retention or candidate geometry")
        return selected

    def _absolute_selection(self, frames, selected):
        return tuple(frames[index//self.segments]*self.tpf + (index%self.segments)*self.segment_tokens+p
                     for index in selected for p in range(self.segment_tokens))

    def _recent_all_heads(self):
        if self._selector_recent is not None:
            return self._selector_recent
        # _select expects all-head indexing. Only the selector consumes these
        # values; gather the proxy group's tail into its original head slots.
        dynamic = self._histories.get('dynamic')
        if dynamic is not None and self.selector_layer:
            tail = dynamic.k[:, :, -self.tpf:] if self.logical_frames else dynamic.k
            result = tail.new_empty((self.batch, self.num_heads, tail.shape[2], self.head_dim))
            result.index_copy_(1, dynamic.heads, tail)
            return result
        return None

    @torch.no_grad()
    def append(self, k, v):
        if k.ndim != 4 or k.shape != v.shape or k.shape[:2] != (self.batch, self.num_heads):
            raise ValueError("appended KV must match cache batch and heads")
        if k.shape[-1] != self.head_dim or k.shape[-2] % self.tpf:
            raise ValueError("committed KV must contain complete spatial frames")
        frames = k.shape[2]//self.tpf
        if not frames:
            return self
        end = self.logical_frames+frames
        candidate_frames = tuple(range(max(self.sink_frames, self.logical_frames-1),
                                       max(self.sink_frames, end-1)))
        # Build proxy candidates only once. Other layers gather from their own
        # recent frame and newly committed K/V after the selector has run.
        if self.selector_layer:
            recent = self._recent_all_heads()
            old_part = recent if self.logical_frames > self.sink_frames else k[:, :, :0]
            new_begin = max(0, self.sink_frames-self.logical_frames)*self.tpf
            candidates = torch.cat((old_part, k[:, :, new_begin:-self.tpf]), dim=2)
        else:
            candidates = k[:, :, :0]
        selected = self._select(candidates, k[:, :, -self.tpf:], candidate_frames, end)
        selected_tokens = self._absolute_selection(candidate_frames, selected)
        selected_ids = self.selection_state.get('token_indices')
        for name, history in self._histories.items():
            kk, vv = k.index_select(1, history.heads).float(), v.index_select(1, history.heads)
            old_sink = min(self.logical_frames, self.sink_frames)*self.tpf
            new_sink = min(frames, max(0, self.sink_frames-self.logical_frames))*self.tpf
            sink_k = torch.cat((history.k[:, :, :old_sink], kk[:, :, :new_sink]), dim=2)
            sink_v = torch.cat((history.v[:, :, :old_sink], vv[:, :, :new_sink]), dim=2)
            pieces_k, pieces_v = [sink_k], [sink_v]
            tokens = tuple(range(old_sink+new_sink))
            if name == 'dynamic' and selected:
                old_k = history.k[:, :, -self.tpf:] if self.logical_frames > self.sink_frames else kk[:, :, :0]
                old_v = history.v[:, :, -self.tpf:] if self.logical_frames > self.sink_frames else vv[:, :, :0]
                new_begin = max(0, self.sink_frames-self.logical_frames)*self.tpf
                candidates_k = torch.cat((old_k, kk[:, :, new_begin:-self.tpf]), dim=2)
                candidates_v = torch.cat((old_v, vv[:, :, new_begin:-self.tpf]), dim=2)
                pieces_k.append(candidates_k.index_select(2, selected_ids))
                pieces_v.append(candidates_v.index_select(2, selected_ids))
                tokens += selected_tokens
            if end > self.sink_frames:
                pieces_k.append(kk[:, :, -self.tpf:])
                pieces_v.append(vv[:, :, -self.tpf:])
                tokens += tuple(range((end-1)*self.tpf, end*self.tpf))
            history.k, history.v = torch.cat(pieces_k, dim=2), torch.cat(pieces_v, dim=2)
            history.tokens, history.version = tokens, history.version+1
        if self._selector_recent is not None:
            self._selector_recent = k[:, :, -self.tpf:].float().clone()
        self._selected_tokens = selected_tokens if self.groups['dynamic'].numel() else ()
        self.logical_frames = end
        self._version += 1
        if self.max_frames <= 0 or self.logical_frames <= self.max_frames:
            self._metadata.clear()
        return self

    @torch.no_grad()
    def evict(self, drop, rope):
        drop = int(drop)
        if drop < 0 or drop > max(0, self.logical_frames-self.sink_frames):
            raise ValueError("invalid number of post-sink frames to evict")
        if not drop:
            return self
        sink, boundary = self.sink_frames*self.tpf, (self.sink_frames+drop)*self.tpf
        for history in self._histories.values():
            pinned = [i for i, token in enumerate(history.tokens) if token < sink]
            moving = [i for i, token in enumerate(history.tokens) if token >= boundary]
            indices = torch.tensor(pinned+moving, device=self.device, dtype=torch.long)
            kk, vv = history.k.index_select(2, indices), history.v.index_select(2, indices)
            if moving:
                kk[:, :, len(pinned):] = rope.rope_shift_time(-drop, kk[:, :, len(pinned):])
            history.tokens = tuple(history.tokens[i] if history.tokens[i] < sink else history.tokens[i]-drop*self.tpf
                                   for i in pinned+moving)
            history.k, history.v, history.version = kk, vv, history.version+1
        if self._selector_recent is not None:
            if self.logical_frames-1 >= self.sink_frames+drop:
                self._selector_recent = rope.rope_shift_time(-drop, self._selector_recent)
            elif self.logical_frames-drop <= self.sink_frames:
                # Every post-sink frame was evicted. Retain the last pinned
                # frame as the next old boundary, or an empty frame if none.
                h = self._histories.get('static')
                self._selector_recent = (h.k[:, :, -self.tpf:].clone() if h is not None and h.tokens
                                         else self._selector_recent[:, :, :0].clone())
        self._selected_tokens = tuple(t if t < sink else t-drop*self.tpf
                                      for t in self._selected_tokens if t < sink or t >= boundary)
        self.logical_frames -= drop
        self._version += 1
        if self.logical_frames != self.max_frames:
            self._metadata.clear()
        return self

    def _static_flash(self, history, q, k, v, chunk_tokens, dtype):
        """Batch independent chunk queries with sink + predecessor + own KV."""
        if self.logical_frames < self.sink_frames:
            return self._static_initial_flash(history, q, k, v, chunk_tokens, dtype)
        b, h, n, d = q.shape
        chunks = math.ceil(n/chunk_tokens)
        padding = chunks*chunk_tokens-n
        n_sink = min(self.sink_frames, self.logical_frames)*self.tpf
        history_prev = self.tpf if self.logical_frames > self.sink_frames else 0
        name = ('static', chunks)
        old = self._workspaces.get(name)
        if old is None or old['version'] != history.version or old['k'].dtype != dtype:
            pk = history.k.transpose(1, 2).repeat_interleave(chunks, dim=0)
            pv = history.v.transpose(1, 2).repeat_interleave(chunks, dim=0)
        else:
            pk, pv = old['k'][:, :history.k.shape[2]], old['v'][:, :history.v.shape[2]]
        kk, vv, _ = self._workspace(name, pk, pv, self.tpf+chunk_tokens, dtype, history.version)
        capacity = kk.shape[1]
        kk5, vv5 = kk.view(b, chunks, capacity, h, d), vv.view(b, chunks, capacity, h, d)

        def chunk_batch(t):
            if padding:
                t = F.pad(t, (0, 0, 0, padding))
            return t.reshape(b, h, chunks, chunk_tokens, d).permute(0, 2, 3, 1, 4)

        ck, cv = chunk_batch(k), chunk_batch(v)
        first = n_sink+history_prev
        kk5[:, 0, first:first+chunk_tokens].copy_(ck[:, 0])
        vv5[:, 0, first:first+chunk_tokens].copy_(cv[:, 0])
        key = ('static_metadata', b, n, chunk_tokens, n_sink, history_prev)
        metadata = self._metadata.get(key)
        if metadata is None:
            previous = (torch.arange(1, chunks, device=self.device)[:, None]*chunk_tokens-self.tpf
                        + torch.arange(self.tpf, device=self.device)[None, :]).flatten()
            sizes = torch.full((chunks,), chunk_tokens, device=self.device, dtype=torch.int32)
            sizes[-1] = n-(chunks-1)*chunk_tokens
            sizes += n_sink+self.tpf
            sizes[0] -= self.tpf-history_prev
            metadata = previous, sizes.repeat(b)
            self._metadata[key] = metadata
        previous, lengths = metadata
        if chunks > 1:
            for target, source, current in ((kk5, k, ck), (vv5, v, cv)):
                prev = source.index_select(2, previous).reshape(b, h, chunks-1, self.tpf, d).permute(0, 2, 3, 1, 4)
                target[:, 1:, n_sink:n_sink+self.tpf].copy_(prev)
                target[:, 1:, n_sink+self.tpf:n_sink+self.tpf+chunk_tokens].copy_(current[:, 1:])
        queries = chunk_batch(q).reshape(b*chunks, chunk_tokens, h, d)
        result = flash_attn_with_kvcache(queries, kk, vv, cache_seqlens=lengths, causal=False)
        return result.reshape(b, chunks*chunk_tokens, h, d)[:, :n].transpose(1, 2)

    def _static_initial_flash(self, history, q, k, v, chunk_tokens, dtype):
        """Keep the initial sink even while it is still an in-flight frame.

        This short startup path gathers disjoint per-chunk intervals. The
        common committed-sink path above keeps persistent packed history.
        """
        b, h, n, d = q.shape
        chunks = math.ceil(n/chunk_tokens)
        offset = self.logical_frames*self.tpf
        key = ('initial_static', b, n, chunk_tokens, offset)
        metadata = self._metadata.get(key)
        if metadata is None:
            rows = []
            for start in range(0, n, chunk_tokens):
                end = offset+min(start+chunk_tokens, n)
                sink_end = min(self.sink_frames*self.tpf, end)
                tail_start = max(sink_end, offset+start-self.tpf)
                rows.append(list(range(sink_end))+list(range(tail_start, end)))
            capacity = max(map(len, rows))
            lengths = torch.tensor([len(row) for row in rows], device=self.device, dtype=torch.int32).repeat(b)
            indices = torch.tensor([row+[0]*(capacity-len(row)) for row in rows],
                                   device=self.device, dtype=torch.long).flatten()
            metadata = indices, lengths, capacity
            self._metadata[key] = metadata
        indices, lengths, capacity = metadata
        empty = q.new_empty((b*chunks, 0, h, d), dtype=dtype)
        kk, vv, _ = self._workspace(('initial_static', chunks), empty, empty,
                                    capacity, dtype, history.version)
        for target, current, past in ((kk, k, history.k), (vv, v, history.v)):
            source = torch.cat((past.to(dtype), current.to(dtype)), dim=2) if offset else current.to(dtype)
            packed = (source.index_select(2, indices).reshape(b, h, chunks, capacity, d)
                      .permute(0, 2, 3, 1, 4).reshape(b*chunks, capacity, h, d))
            target[:, :capacity].copy_(packed)
        if n < chunks*chunk_tokens:
            q = F.pad(q, (0, 0, 0, chunks*chunk_tokens-n))
        queries = q.reshape(b, h, chunks, chunk_tokens, d).permute(0, 2, 3, 1, 4).reshape(b*chunks, chunk_tokens, h, d)
        result = flash_attn_with_kvcache(queries, kk, vv, cache_seqlens=lengths, causal=False)
        return result.reshape(b, chunks*chunk_tokens, h, d)[:, :n].transpose(1, 2)

    def _reference(self, name, history, q, k, v, chunk_tokens):
        past_ids = torch.tensor(history.tokens, device=self.device, dtype=torch.long)
        offset = self.logical_frames*self.tpf
        ids = torch.cat((past_ids, torch.arange(q.shape[2], device=self.device)+offset))
        keys = torch.cat((history.k, k), dim=2).to(q.dtype)
        values = torch.cat((history.v, v), dim=2).to(q.dtype)
        output = []
        for start in range(0, q.shape[2], chunk_tokens):
            end = min(start+chunk_tokens, q.shape[2])
            visible = ids < offset+end
            if name == 'static':
                visible &= ((ids < self.sink_frames*self.tpf)
                            | (ids >= max(0, offset+start-self.tpf)))
            output.append(F.scaled_dot_product_attention(q[:, :, start:end], keys, values,
                                                        attn_mask=visible[None, :], dropout_p=0.0))
        return torch.cat(output, dim=2)

    @torch.no_grad()
    def attention(self, q, k_current, v_current, chunk_tokens):
        if q.ndim != 4 or q.shape != k_current.shape or q.shape != v_current.shape:
            raise ValueError("Q/current K/current V must match [B,H,N,D]")
        if q.shape[:2] != (self.batch, self.num_heads) or q.shape[-1] != self.head_dim:
            raise ValueError("attention shapes differ from cache")
        if q.shape[2] == 0 or chunk_tokens <= 0:
            raise ValueError("query length and chunk size must be positive")
        dtype = (torch.get_autocast_dtype(q.device.type)
                 if torch.is_autocast_enabled(q.device.type) else v_current.dtype)
        flash = (q.is_cuda and flash_attn_with_kvcache is not None
                 and dtype in (torch.bfloat16, torch.float16)
                 and self.head_dim <= 256 and self.head_dim % 8 == 0)
        packed = (pack_current_qkv(q, k_current, v_current, self._pack_heads, self._current_workspace, dtype)
                  if flash and self.fused_current_pack else None)
        output = torch.empty_like(q, dtype=dtype if flash else q.dtype)
        for name, history in self._histories.items():
            if packed is None:
                qq, kk, vv = (t.index_select(1, history.heads) for t in (q, k_current, v_current))
                if flash:
                    qq = qq.to(dtype)
            else:
                qq, kk, vv = (t[:, self._pack_slices[name]] for t in packed)
            if not flash or (name == 'static' and (chunk_tokens % self.tpf or chunk_tokens < self.tpf)):
                result = self._reference(name, history, qq, kk, vv, chunk_tokens)
            elif name == 'static':
                result = self._static_flash(history, qq, kk, vv, chunk_tokens, dtype)
            else:
                result = self._ordinary_flash(name, history, qq, kk, vv, chunk_tokens, dtype)
            output.index_copy_(1, history.heads, result.to(output.dtype))
        return output

    def storage_bytes(self, seen=None):
        seen = set() if seen is None else seen
        result = 0

        def visit(obj):
            nonlocal result
            if isinstance(obj, torch.Tensor):
                storage = obj.untyped_storage()
                key = str(obj.device), storage.data_ptr()
                if key not in seen:
                    seen.add(key)
                    result += storage.nbytes()
            elif isinstance(obj, _SparseHistory):
                visit((obj.heads, obj.k, obj.v))
            elif isinstance(obj, dict):
                for value in obj.values():
                    visit(value)
            elif isinstance(obj, (list, tuple)):
                for value in obj:
                    visit(value)

        visit((self.groups, self._pack_heads, self._histories, self._workspaces, self._metadata,
               self._selector_recent, self.selection_state, self._current_workspace.buffers))
        return result

    def summary(self):
        return {
            'logical_frames': self.logical_frames,
            'groups': {name: heads.numel() for name, heads in self.groups.items()},
            'stored_history_tokens_per_head': {name: len(h.tokens) for name, h in self._histories.items()},
            'selected_dynamic_tokens_per_head': len(self._selected_tokens),
            'selection_updates': int(self.selection_state.get('selection_updates', 0)),
            'selection_batch_reduction': 'minimum',
            'selector_layer': self.selector_layer,
            'proxy_all_heads': bool(self.selection_state.get('proxy_all_heads', False)),
            'proxy_extra_bytes': (0 if self._selector_recent is None else
                                  self._selector_recent.numel()*self._selector_recent.element_size()),
            'storage_bytes': self.storage_bytes(),
            'canonical_kv_bytes': sum(h.k.numel()*h.k.element_size()+h.v.numel()*h.v.element_size()
                                      for h in self._histories.values()),
            'packed_scratch_bytes': sum(w['k'].numel()*w['k'].element_size()+w['v'].numel()*w['v'].element_size()
                                        for w in self._workspaces.values()),
            'byte_accounting': 'storage_bytes deduplicates storage; canonical/scratch may overlap through V aliases',
        }
