"""Request-owned, physically heterogeneous inference KV caches.

Canonical keys retain their incoming precision for MiniWorld's eviction RoPE
shift. FlashAttention history is packed/cast only after a clean cache update;
each denoising call writes just its current K/V into reusable scratch tails.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from miniworld.kv_kernels import CurrentPackWorkspace, pack_current_qkv

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None


GROUPS = ("global", "sink", "dummy", "local", "spatial")


@dataclass
class _History:
    heads: torch.Tensor
    frames: tuple
    k: torch.Tensor
    v: torch.Tensor
    version: int = 0


class CompactKVCache:
    """One layer's compact history; never attached to model/checkpoint state."""

    @classmethod
    @torch.no_grad()
    def from_dense(cls, kv, groups, *, logical_frames, height, width, chunk_size,
                   sink_frames=1, dummy_frames=1, local_frames=4,
                   tile_height=5, tile_width=5, max_frames=48):
        self = cls()
        k, v = kv
        if k.ndim != 4 or k.shape != v.shape:
            raise ValueError("KV must have matching [B,H,N,D] shapes")
        if min(height, width, chunk_size, tile_height, tile_width) <= 0:
            raise ValueError("grid, chunk, and tile dimensions must be positive")
        if min(logical_frames, sink_frames, dummy_frames, local_frames, max_frames) < 0:
            raise ValueError("frame counts must be nonnegative")
        self.height, self.width = int(height), int(width)
        self.tpf = self.height * self.width
        self.chunk_size = int(chunk_size)
        self.logical_frames = int(logical_frames)
        self.sink_frames = int(sink_frames)
        self.dummy_frames, self.local_frames = int(dummy_frames), int(local_frames)
        self.tile_height, self.tile_width = int(tile_height), int(tile_width)
        self.max_frames = int(max_frames)
        self.batch, self.num_heads, n, self.head_dim = k.shape
        self.device = k.device
        if n != self.logical_frames * self.tpf:
            raise ValueError("dense KV length differs from logical_frames * grid area")
        if set(groups) != set(GROUPS):
            raise ValueError(f"groups must have exactly {GROUPS}")
        ids = {name: torch.as_tensor(groups[name], device=k.device, dtype=torch.long)
               for name in GROUPS}
        if any(t.ndim != 1 for t in ids.values()):
            raise ValueError("head indices must be one-dimensional")
        all_ids = torch.cat(tuple(ids.values())).cpu().tolist()
        if sorted(all_ids) != list(range(self.num_heads)):
            raise ValueError("groups must be a disjoint exhaustive partition of heads")
        self.groups = ids
        self._pack_heads = torch.cat(tuple(ids.values()))
        self._pack_slices = {}
        start = 0
        for name in GROUPS:
            self._pack_slices[name] = slice(start, start+ids[name].numel())
            start += ids[name].numel()
        self._current_workspace = CurrentPackWorkspace()
        self.fused_current_pack = True
        self._histories = {}
        for name in GROUPS:
            heads = ids[name]
            if not heads.numel():
                continue
            frames = self._select_frames(name, tuple(range(self.logical_frames)))
            first = frames[0]*self.tpf if frames else 0
            last = (frames[-1]+1)*self.tpf if frames else 0
            # index_select gives independently owned compact storage, including
            # an all-global cache; it must not keep a dense source tensor alive.
            self._histories[name] = _History(
                heads, frames, k[:, :, first:last].index_select(1, heads),
                v[:, :, first:last].index_select(1, heads))
        self._version = 0
        self._workspaces = {}
        self._metadata = {}
        buckets = {}
        tile_ids = [0] * self.tpf
        tile_id = 0
        for y in range(0, self.height, self.tile_height):
            for x in range(0, self.width, self.tile_width):
                positions = [yy*self.width+xx
                             for yy in range(y, min(y+self.tile_height, self.height))
                             for xx in range(x, min(x+self.tile_width, self.width))]
                buckets.setdefault(len(positions), []).append(positions)
                for p in positions:
                    tile_ids[p] = tile_id
                tile_id += 1
        self._tiles = tuple(torch.tensor(p, device=k.device, dtype=torch.long)
                            for p in buckets.values())
        self._tile_ids = torch.tensor(tile_ids, device=k.device, dtype=torch.long)
        self._all_global = ids["global"].numel() == self.num_heads and all_ids == list(range(self.num_heads))
        return self

    def share_current_workspace(self, workspace):
        if not isinstance(workspace, CurrentPackWorkspace):
            raise TypeError("expected a request-owned CurrentPackWorkspace")
        self._current_workspace = workspace
        return self

    def _select_frames(self, name, frames):
        if name == "sink":
            return tuple(f for f in frames if f < self.sink_frames)
        if name in ("dummy", "local"):
            budget = self.dummy_frames if name == "dummy" else self.local_frames
            return frames[-budget:] if budget else ()
        return frames

    @torch.no_grad()
    def append(self, k, v):
        if k.ndim != 4 or k.shape != v.shape or k.shape[:2] != (self.batch, self.num_heads):
            raise ValueError("appended KV must match cache batch and heads")
        if k.shape[-1] != self.head_dim or k.shape[-2] % self.tpf:
            raise ValueError("committed KV must contain complete spatial frames")
        n_frames = k.shape[-2] // self.tpf
        if not n_frames:
            return self
        new_frames = tuple(range(self.logical_frames, self.logical_frames+n_frames))
        overflow = self.max_frames > 0 and self.logical_frames+n_frames > self.max_frames
        for name, history in self._histories.items():
            if name == 'sink' and self.logical_frames >= self.sink_frames:
                continue
            frames = history.frames + new_frames
            # Keep old recent candidates until an overflow eviction has removed
            # middle frames; pinned frames can then enter the recent budget.
            wanted = frames if overflow and name in ('dummy', 'local') else self._select_frames(name, frames)
            if name == 'sink':
                first, last = 0, len(wanted)
            else:
                first, last = len(frames)-len(wanted), len(frames)
            old_count = len(history.frames)
            old_start, old_end = min(first, old_count)*self.tpf, min(last, old_count)*self.tpf
            new_start, new_end = max(0, first-old_count)*self.tpf, max(0, last-old_count)*self.tpf
            old_k, old_v = history.k[:, :, old_start:old_end], history.v[:, :, old_start:old_end]
            if new_end > new_start:
                new_k = k[:, :, new_start:new_end].index_select(1, history.heads)
                new_v = v[:, :, new_start:new_end].index_select(1, history.heads)
                kk = torch.cat((old_k, new_k), dim=2) if old_end > old_start else new_k
                vv = torch.cat((old_v, new_v), dim=2) if old_end > old_start else new_v
            else:
                kk, vv = old_k.clone(), old_v.clone()
            history.frames, history.k, history.v = wanted, kk, vv
            history.version += 1
        self.logical_frames += n_frames
        self._version += 1
        # At capacity, append is followed by eviction back to the same geometry.
        # Keep cached tile IDs / FA sequence metadata for that repeated geometry.
        if self.max_frames <= 0 or self.logical_frames <= self.max_frames:
            self._metadata.clear()
        return self

    @torch.no_grad()
    def evict(self, drop_frames, rope_module):
        drop_frames = int(drop_frames)
        if drop_frames < 0 or drop_frames > max(0, self.logical_frames-self.sink_frames):
            raise ValueError("invalid number of post-sink frames to evict")
        if not drop_frames:
            return self
        boundary = self.sink_frames + drop_frames
        for name, history in self._histories.items():
            if name == 'sink' or not history.frames:
                continue
            keep = [(i, f) for i, f in enumerate(history.frames)
                    if f < self.sink_frames or f >= boundary]
            if name in ('dummy', 'local'):
                budget = self.dummy_frames if name == 'dummy' else self.local_frames
                keep = keep[-budget:] if budget else []
            pinned = [i for i, f in keep if f < self.sink_frames]
            moving = [i for i, f in keep if f >= boundary]
            # Both lists describe contiguous spans of compact storage. There
            # is no per-token Python list, CPU->GPU index copy, or index_select.
            kp = history.k[:, :, pinned[0]*self.tpf:(pinned[-1]+1)*self.tpf] if pinned else None
            vp = history.v[:, :, pinned[0]*self.tpf:(pinned[-1]+1)*self.tpf] if pinned else None
            if moving:
                begin, end = moving[0]*self.tpf, (moving[-1]+1)*self.tpf
                km = rope_module.rope_shift_time(-drop_frames, history.k[:, :, begin:end])
                vm = history.v[:, :, begin:end]
                kk = torch.cat((kp, km), dim=2) if pinned else km
                # Clone the tail when alone: canonical V may alias a persistent
                # buffer whose prefix will be overwritten on the next forward.
                vv = torch.cat((vp, vm), dim=2) if pinned else vm.clone()
            elif pinned:
                kk, vv = kp.clone(), vp.clone()
            else:
                kk, vv = history.k[:, :, :0].clone(), history.v[:, :, :0].clone()
            history.frames = tuple(f if f < self.sink_frames else f-drop_frames for _, f in keep)
            history.k, history.v = kk, vv
            history.version += 1
        self.logical_frames -= drop_frames
        self._version += 1
        if self.logical_frames != self.max_frames:
            self._metadata.clear()
        return self

    def _workspace(self, name, history_k, history_v, current_length, dtype, version):
        """BLHD history; cached buffers have room for current scratch tails."""
        b, n, h, d = history_k.shape
        required = n + current_length
        old = self._workspaces.get(name)
        if (old is None or old['k'].shape[0] != b or old['k'].shape[2:] != (h, d)
                or old['k'].shape[1] < required or old['k'].dtype != dtype):
            capacity = max(required, 1)
            old = {'k': torch.empty((b, capacity, h, d), device=self.device, dtype=dtype),
                   'v': torch.empty((b, capacity, h, d), device=self.device, dtype=dtype),
                   'version': -1, 'n_past': n}
            self._workspaces[name] = old
        if old['version'] != version or old['n_past'] != n:
            old['k'][:, :n].copy_(history_k)
            old['v'][:, :n].copy_(history_v)
            old['version'], old['n_past'] = version, n
        return old['k'], old['v'], n

    def _range_metadata(self, name, batch, n_current, chunk_tokens, n_past, back=None):
        key = (name, batch, n_current, chunk_tokens, n_past, back)
        metadata = self._metadata.get(key)
        if metadata is None:
            chunks = math.ceil(n_current/chunk_tokens)
            starts = torch.arange(chunks, device=self.device, dtype=torch.int32)*chunk_tokens
            ends = (starts+chunk_tokens).clamp_max(n_current)
            lengths = (ends+n_past).repeat(batch)
            left = None if back is None else (starts+n_past-back).clamp_min(0).repeat(batch)
            indices = torch.arange(batch, device=self.device, dtype=torch.int32).repeat_interleave(chunks)
            metadata = (indices, lengths, left)
            self._metadata[key] = metadata
        return metadata

    def _flash_ranges(self, q, cache_k, cache_v, n_past, chunk_tokens, name, back=None):
        # q: [B,H,N,D]; queries become chunk batches sharing persistent KV.
        b, h, n, d = q.shape
        chunks = math.ceil(n/chunk_tokens)
        if n != chunks*chunk_tokens:
            q = F.pad(q, (0, 0, 0, chunks*chunk_tokens-n))
        queries = (q.reshape(b, h, chunks, chunk_tokens, d).permute(0, 2, 3, 1, 4)
                   .reshape(b*chunks, chunk_tokens, h, d))
        indices, lengths, left = self._range_metadata(name, b, n, chunk_tokens, n_past, back)
        output = flash_attn_with_kvcache(
            queries, cache_k, cache_v, cache_batch_idx=indices,
            cache_seqlens=lengths, cache_leftpad=left, causal=False)
        return output.reshape(b, chunks*chunk_tokens, h, d)[:, :n].transpose(1, 2)

    def _ordinary_flash(self, name, history, q, k, v, chunk_tokens, dtype):
        kk, vv, n_past = self._workspace(
            name, history.k.transpose(1, 2), history.v.transpose(1, 2), q.shape[2], dtype, history.version)
        if history.v.dtype == dtype:
            # Values need no RoPE update. Reuse the history portion of packed
            # scratch as canonical V; append/evict materialize new survivors.
            history.v = vv[:, :n_past].transpose(1, 2)
        kk[:, n_past:n_past+k.shape[2]].copy_(k.transpose(1, 2))
        vv[:, n_past:n_past+v.shape[2]].copy_(v.transpose(1, 2))
        back = None
        if name in ('dummy', 'local'):
            back = (self.dummy_frames if name == 'dummy' else self.local_frames)*self.tpf
        return self._flash_ranges(q, kk, vv, n_past, chunk_tokens, name, back)

    def _sink_flash(self, history, q, k, v, chunk_tokens, dtype):
        # Sink + own query chunk is noncontiguous in a shared current prefix.
        # Batch chunks; duplicate only the small fixed sink, once per geometry.
        b, h, n, d = q.shape
        chunks = math.ceil(n/chunk_tokens)
        padding = chunks*chunk_tokens-n
        def chunk_batch(t):
            if padding:
                t = F.pad(t, (0, 0, 0, padding))
            return t.reshape(b, h, chunks, chunk_tokens, d).permute(0, 2, 3, 1, 4).reshape(b*chunks, chunk_tokens, h, d)
        old = self._workspaces.get(('sink', chunks))
        if old is None or old['version'] != history.version or old['k'].dtype != dtype:
            history_k = history.k.transpose(1, 2).repeat_interleave(chunks, dim=0)
            history_v = history.v.transpose(1, 2).repeat_interleave(chunks, dim=0)
        else:
            history_k = old['k'][:, :history.k.shape[2]]
            history_v = old['v'][:, :history.v.shape[2]]
        kk, vv, n_past = self._workspace(('sink', chunks), history_k, history_v, chunk_tokens, dtype, history.version)
        if history.v.dtype == dtype:
            history.v = vv.reshape(b, chunks, vv.shape[1], h, d)[:, 0, :n_past].transpose(1, 2)
        kk[:, n_past:n_past+chunk_tokens].copy_(chunk_batch(k))
        vv[:, n_past:n_past+chunk_tokens].copy_(chunk_batch(v))
        meta_key = ('sink_lengths', b, n, chunk_tokens, n_past)
        lengths = self._metadata.get(meta_key)
        if lengths is None:
            sizes = torch.full((chunks,), chunk_tokens, dtype=torch.int32, device=self.device)
            sizes[-1] = n-(chunks-1)*chunk_tokens
            lengths = (sizes+n_past).repeat(b)
            self._metadata[meta_key] = lengths
        out = flash_attn_with_kvcache(chunk_batch(q), kk, vv, cache_seqlens=lengths, causal=False)
        return out.reshape(b, chunks*chunk_tokens, h, d)[:, :n].transpose(1, 2)

    def _spatial_flash(self, history, q, k, v, chunk_tokens, dtype):
        b, h, n, d = q.shape
        out = torch.empty_like(q)
        frames = n//self.tpf
        past_frames = len(history.frames)
        for bucket, positions in enumerate(self._tiles):
            tiles, area = positions.shape
            meta_key = ('spatial_ids', bucket, frames, past_frames)
            ids = self._metadata.get(meta_key)
            if ids is None:
                cur_ids = (torch.arange(frames, device=self.device)[None, :, None]*self.tpf
                           + positions[:, None, :]).reshape(tiles, -1)
                past_ids = (torch.arange(past_frames, device=self.device)[None, :, None]*self.tpf
                            + positions[:, None, :]).reshape(tiles, -1)
                ids = (cur_ids, past_ids)
                self._metadata[meta_key] = ids
            cur_ids, past_ids = ids
            def tile(t, token_ids):
                return (t.index_select(2, token_ids.flatten()).reshape(b, h, tiles, -1, d)
                        .permute(0, 2, 3, 1, 4).reshape(b*tiles, -1, h, d))
            workspace_name = ('spatial', bucket)
            old = self._workspaces.get(workspace_name)
            if old is None or old['version'] != history.version or old['k'].dtype != dtype:
                pk, pv = tile(history.k, past_ids), tile(history.v, past_ids)
            else:
                # No history gather/cast on a denoising step with unchanged KV.
                pk, pv = old['k'][:, :past_frames*area], old['v'][:, :past_frames*area]
            kk, vv, n_past = self._workspace(workspace_name, pk, pv, frames*area, dtype, history.version)
            kk[:, n_past:n_past+frames*area].copy_(tile(k, cur_ids))
            vv[:, n_past:n_past+frames*area].copy_(tile(v, cur_ids))
            query = tile(q, cur_ids).transpose(1, 2)
            result = self._flash_ranges(query, kk, vv, n_past,
                                        chunk_tokens//self.tpf*area, workspace_name)
            result = (result.reshape(b, tiles, h, frames*area, d).permute(0, 2, 1, 3, 4)
                      .reshape(b, h, tiles*frames*area, d))
            out.index_copy_(2, cur_ids.flatten(), result)
        return out

    def _reference(self, name, history, q, k, v, chunk_tokens):
        """Exact compact-cache SDPA fallback, also handles partial spatial frames."""
        past_tokens = [f*self.tpf+p for f in history.frames for p in range(self.tpf)]
        past_ids = torch.tensor(past_tokens, dtype=torch.long, device=q.device)
        current_ids = torch.arange(q.shape[2], device=q.device)+self.logical_frames*self.tpf
        key_ids = torch.cat((past_ids, current_ids))
        keys, values = torch.cat((history.k, k), dim=2), torch.cat((history.v, v), dim=2)
        if keys.dtype != q.dtype:
            keys = keys.to(q.dtype)
        if values.dtype != q.dtype:
            values = values.to(q.dtype)
        outputs = []
        offset = self.logical_frames*self.tpf
        for start in range(0, q.shape[2], chunk_tokens):
            end = min(start+chunk_tokens, q.shape[2])
            visible = key_ids < offset+end
            if name == 'sink':
                visible &= (key_ids < min(self.sink_frames, self.logical_frames)*self.tpf) | (key_ids >= offset+start)
            elif name in ('dummy', 'local'):
                back = (self.dummy_frames if name == 'dummy' else self.local_frames)*self.tpf
                visible &= key_ids >= max(0, offset+start-back)
            if name == 'spatial':
                qids = torch.arange(start, end, device=q.device)%self.tpf
                visible = visible[None, :] & (self._tile_ids[qids, None] == self._tile_ids[key_ids%self.tpf][None, :])
            else:
                visible = visible[None, :]
            outputs.append(F.scaled_dot_product_attention(q[:, :, start:end], keys, values,
                                                         attn_mask=visible, dropout_p=0.0))
        return torch.cat(outputs, dim=2)

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
        packed = None
        if flash and not self._all_global and self.fused_current_pack:
            packed = pack_current_qkv(q, k_current, v_current, self._pack_heads,
                                      self._current_workspace, dtype)
        elif flash:
            q = q.to(dtype)
        output = None if self._all_global else torch.empty_like(q, dtype=dtype if flash else q.dtype)
        for name, history in self._histories.items():
            if self._all_global:
                qq, kk, vv = q, k_current, v_current
            elif packed is not None:
                group_slice = self._pack_slices[name]
                qq, kk, vv = (tensor[:, group_slice] for tensor in packed)
            else:
                qq = q.index_select(1, history.heads)
                kk, vv = k_current.index_select(1, history.heads), v_current.index_select(1, history.heads)
            if not flash or (name == 'spatial' and (q.shape[2]%self.tpf or chunk_tokens%self.tpf)):
                result = self._reference(name, history, qq, kk, vv, chunk_tokens)
            elif name == 'sink':
                result = self._sink_flash(history, qq, kk, vv, chunk_tokens, dtype)
            elif name == 'spatial':
                result = self._spatial_flash(history, qq, kk, vv, chunk_tokens, dtype)
            else:
                result = self._ordinary_flash(name, history, qq, kk, vv, chunk_tokens, dtype)
            if self._all_global:
                return result
            output.index_copy_(1, history.heads, result.to(output.dtype))
        return output

    def storage_bytes(self, seen=None):
        """Unique storage bytes; optional shared set deduplicates across caches.

        With a caller-owned set, return only newly encountered storage so the
        request-shared current pack is counted once across layers and CFG.
        """
        if seen is None:
            seen = set()
        newly_seen = {}
        def visit(obj):
            if isinstance(obj, torch.Tensor):
                storage = obj.untyped_storage()
                key = (str(obj.device), storage.data_ptr())
                if key not in seen:
                    seen.add(key)
                    newly_seen[key] = storage.nbytes()
            elif isinstance(obj, dict):
                for value in obj.values():
                    visit(value)
            elif isinstance(obj, (tuple, list)):
                for value in obj:
                    visit(value)
            elif isinstance(obj, _History):
                visit((obj.heads, obj.k, obj.v))
        visit((self._histories, self.groups, self._workspaces, self._metadata,
               self._tiles, self._tile_ids, self._pack_heads, self._current_workspace.buffers))
        return sum(newly_seen.values())

    def summary(self):
        return {'logical_frames': self.logical_frames,
                'groups': {name: ids.numel() for name, ids in self.groups.items()},
                'stored_history_tokens_per_head': {name: len(h.frames)*self.tpf for name, h in self._histories.items()},
                'storage_bytes': self.storage_bytes(),
                'canonical_kv_bytes': sum(h.k.numel()*h.k.element_size()+h.v.numel()*h.v.element_size()
                                          for h in self._histories.values()),
                'packed_scratch_bytes': sum(w['k'].numel()*w['k'].element_size()+w['v'].numel()*w['v'].element_size()
                                            for w in self._workspaces.values()),
                'byte_accounting': 'storage_bytes deduplicates storage; canonical and scratch byte subtotals can overlap through aliased V'}
