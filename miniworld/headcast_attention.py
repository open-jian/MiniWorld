"""Grouped history attention with unchanged chunk-causal in-flight visibility.

All inputs use the model's existing RoPE positions. This module only selects
keys for reading; it never mutates or repacks the persistent KV cache.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    from flash_attn import flash_attn_func
except ImportError:  # The reference path also supports CPU-only test installs.
    flash_attn_func = None
try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None


class SpatialLayout:
    """Non-overlapping tiles, bucketed by their number of real spatial tokens."""

    def __init__(self, height: int, width: int, tile_height: int, tile_width: int, device):
        self.height = int(height)
        self.width = int(width)
        self.tile_height = int(tile_height)
        self.tile_width = int(tile_width)
        if min(self.height, self.width, self.tile_height, self.tile_width) <= 0:
            raise ValueError("spatial and tile dimensions must be positive")
        self.device = torch.device(device)
        self.tokens_per_frame = self.height * self.width
        tile_ids = [0] * self.tokens_per_frame
        buckets = {}
        tile_id = 0
        for row in range(0, self.height, self.tile_height):
            for col in range(0, self.width, self.tile_width):
                positions = [
                    r * self.width + c
                    for r in range(row, min(row + self.tile_height, self.height))
                    for c in range(col, min(col + self.tile_width, self.width))
                ]
                for position in positions:
                    tile_ids[position] = tile_id
                buckets.setdefault(len(positions), []).append(positions)
                tile_id += 1
        self.tile_ids = torch.tensor(tile_ids, dtype=torch.long, device=self.device)
        self.device = self.tile_ids.device
        # Each tensor has shape [number of equal-area tiles, valid tile area].
        self._buckets = tuple(
            torch.tensor(positions, dtype=torch.long, device=self.device)
            for positions in buckets.values()
        )


def spatial_history_mask(query_ids: Tensor, n_past: int, visible_end: int,
                         layout: SpatialLayout) -> Tensor:
    """Keep same-tile history plus all ``current[:visible_end]`` keys.

    ``query_ids`` are positions relative to the current sequence. The caller
    supplies the prefix ending at the query's chunk, including the entire chunk.
    """
    if n_past < 0 or visible_end < 0:
        raise ValueError("key lengths must be nonnegative")
    query_ids = query_ids.to(device=layout.device, dtype=torch.long)
    if query_ids.ndim != 1:
        raise ValueError("query_ids must be one-dimensional")
    query_tiles = layout.tile_ids[query_ids.remainder(layout.tokens_per_frame)]
    history_ids = torch.arange(n_past, device=layout.device)
    history_tiles = layout.tile_ids[history_ids.remainder(layout.tokens_per_frame)]
    history_visible = query_tiles[:, None] == history_tiles[None, :]
    current_visible = torch.ones(
        (query_ids.numel(), visible_end), device=layout.device, dtype=torch.bool
    )
    return torch.cat((history_visible, current_visible), dim=1)


def _dense_attention(q: Tensor, k: Tensor, v: Tensor, use_flash: bool) -> Tensor:
    """One normalization over every supplied key, with no token causal mask."""
    if use_flash:
        return flash_attn_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=False
        ).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


def _prefix_attention(q: Tensor, k: Tensor, v: Tensor, query_chunk_tokens: int,
                      visible_lengths: Tuple[int, ...], use_flash: bool) -> Tensor:
    """Attend each query chunk to its explicit KV prefix without copying KV.

    Full chunks share packed KV through repeated cache batch indices. The
    optional final short chunk uses the ordinary attention call; no padding
    enters a softmax. The cache API is read-only because no new K/V is passed.
    """
    b, h, n_query, d = q.shape
    full_chunks = n_query // query_chunk_tokens
    outputs = []
    start = 0
    if (use_flash and flash_attn_with_kvcache is not None and full_chunks > 1
            and not (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)))):
        start = full_chunks * query_chunk_tokens
        q_batch = (q[:, :, :start].reshape(b, h, full_chunks, query_chunk_tokens, d)
                   .permute(0, 2, 3, 1, 4).reshape(b * full_chunks, query_chunk_tokens, h, d))
        cache_indices = torch.arange(b, dtype=torch.int32, device=q.device).repeat_interleave(full_chunks)
        cache_lengths = torch.tensor(
            visible_lengths[:full_chunks], dtype=torch.int32, device=q.device
        ).repeat(b)
        result = flash_attn_with_kvcache(
            q_batch, k.transpose(1, 2), v.transpose(1, 2),
            cache_batch_idx=cache_indices, cache_seqlens=cache_lengths, causal=False,
        )
        outputs.append(result.reshape(b, start, h, d).transpose(1, 2))
    for chunk_start in range(start, n_query, query_chunk_tokens):
        end = min(chunk_start + query_chunk_tokens, n_query)
        visible_end = visible_lengths[chunk_start // query_chunk_tokens]
        outputs.append(_dense_attention(
            q[:, :, chunk_start:end], k[:, :, :visible_end], v[:, :, :visible_end], use_flash
        ))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)


def _chunk_attention(q: Tensor, k_current: Tensor, v_current: Tensor,
                     k_past: Tensor, v_past: Tensor, chunk_tokens: int,
                     use_flash: bool) -> Tensor:
    n_past = k_past.shape[2]
    k = torch.cat((k_past, k_current), dim=2)
    v = torch.cat((v_past, v_current), dim=2)
    visible_lengths = tuple(n_past + min(start + chunk_tokens, q.shape[2])
                            for start in range(0, q.shape[2], chunk_tokens))
    return _prefix_attention(q, k, v, chunk_tokens, visible_lengths, use_flash)


def _merged_global_temporal_attention(
    q: Tensor, k_current: Tensor, v_current: Tensor,
    k_past: Tensor, v_past: Tensor, chunk_tokens: int,
    temporal_heads: Tensor, sink_tokens: int, recent_tokens: int,
) -> Tensor:
    """Read global and temporal heads in one inference-only KV-cache call.

    Heads and query chunks become batch entries. Repeated cache indices share
    each head's scratch KV, while per-query lengths retain exactly its legal
    current prefix. Only scratch copies are packed; source caches stay intact.
    """
    b, h, n_current, d = q.shape
    n_past = k_past.shape[2]
    cache_k = torch.cat((k_past, k_current), dim=2).reshape(b * h, n_past + n_current, 1, d)
    cache_v = torch.cat((v_past, v_current), dim=2).reshape(b * h, n_past + n_current, 1, d)
    anchor_end = min(sink_tokens, n_past)
    recent_start = max(anchor_end, n_past - recent_tokens)
    temporal_history = anchor_end + n_past - recent_start
    history_lengths = torch.full((b, h), n_past, dtype=torch.int32, device=q.device)
    head_batches = (torch.arange(b, device=q.device)[:, None] * h
                    + temporal_heads[None, :]).flatten()
    source_ids = torch.cat((torch.arange(anchor_end, device=q.device),
                            torch.arange(recent_start, n_past + n_current, device=q.device)))
    # Advanced indexing materializes each selected source before overwriting
    # the scratch prefix, including when anchor/recent budgets overlap.
    packed_k = cache_k[head_batches[:, None], source_ids[None, :], 0]
    packed_v = cache_v[head_batches[:, None], source_ids[None, :], 0]
    cache_k[head_batches, :source_ids.numel(), 0] = packed_k
    cache_v[head_batches, :source_ids.numel(), 0] = packed_v
    history_lengths[:, temporal_heads] = temporal_history

    n_chunks = (n_current + chunk_tokens - 1) // chunk_tokens
    if n_current % chunk_tokens:
        queries = q.new_zeros((b, h, n_chunks * chunk_tokens, d))
        queries[:, :, :n_current] = q
    else:
        queries = q.contiguous()
    queries = queries.reshape(b * h * n_chunks, chunk_tokens, 1, d)
    visible_ends = torch.arange(1, n_chunks + 1, dtype=torch.int32, device=q.device)
    visible_ends = visible_ends.mul(chunk_tokens).clamp_max(n_current)
    lengths = (history_lengths.reshape(b * h, 1) + visible_ends[None, :]).flatten()
    cache_indices = torch.arange(b * h, dtype=torch.int32, device=q.device).repeat_interleave(n_chunks)
    output = flash_attn_with_kvcache(
        queries, cache_k, cache_v, cache_batch_idx=cache_indices,
        cache_seqlens=lengths, causal=False,
    )
    # Padding creates query outputs only; no padded keys enter a softmax.
    return output.reshape(b, h, n_chunks * chunk_tokens, d)[:, :, :n_current]


def _spatial_reference(q: Tensor, k_current: Tensor, v_current: Tensor,
                       k_past: Tensor, v_past: Tensor, chunk_tokens: int,
                       layout: SpatialLayout) -> Tensor:
    """Exact fallback for flattened sequences that end inside spatial frames."""
    n_past = k_past.shape[2]
    k = torch.cat((k_past, k_current), dim=2)
    v = torch.cat((v_past, v_current), dim=2)
    outputs = []
    for start in range(0, q.shape[2], chunk_tokens):
        end = min(start + chunk_tokens, q.shape[2])
        query_ids = torch.arange(start, end, device=q.device)
        mask = spatial_history_mask(query_ids, n_past, end, layout)
        outputs.append(F.scaled_dot_product_attention(
            q[:, :, start:end], k[:, :, :n_past + end], v[:, :, :n_past + end],
            attn_mask=mask, dropout_p=0.0, is_causal=False,
        ))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2)


def _spatial_attention(q: Tensor, k_current: Tensor, v_current: Tensor,
                       k_past: Tensor, v_past: Tensor, chunk_tokens: int,
                       layout: SpatialLayout, use_flash: bool) -> Tensor:
    b, h, n_cur, d = q.shape
    n_past = k_past.shape[2]
    tpf = layout.tokens_per_frame
    if n_past % tpf or n_cur % tpf or chunk_tokens % tpf:
        return _spatial_reference(q, k_current, v_current, k_past, v_past, chunk_tokens, layout)

    past_offsets = torch.arange(n_past // tpf, device=q.device) * tpf
    current_offsets = torch.arange(n_cur // tpf, device=q.device) * tpf
    output = torch.empty_like(q)
    for positions in layout._buckets:
        tiles, area = positions.shape
        past_ids = (positions[:, None, :] + past_offsets[None, :, None]).reshape(tiles, -1)
        current_ids = (positions[:, None, :] + current_offsets[None, :, None]).reshape(tiles, -1)
        past_len = past_ids.shape[1]
        current_len = current_ids.shape[1]

        def pack(tensor: Tensor, ids: Tensor, length: int) -> Tensor:
            return (tensor.index_select(2, ids.reshape(-1))
                    .reshape(b, h, tiles, length, d).permute(0, 2, 1, 3, 4))

        # Packing is per bucket/forward, not repeated for each query chunk.
        q_tiles = pack(q, current_ids, current_len)
        history_k = pack(k_past, past_ids, past_len)
        history_v = pack(v_past, past_ids, past_len)
        # Each tile shares the full current sequence. Pack it once, then slice
        # the legal prefix per chunk so history and current share one softmax.
        k_batch = torch.cat((
            history_k, k_current[:, None].expand(b, tiles, h, n_cur, d)
        ), dim=3).reshape(b * tiles, h, past_len + n_cur, d)
        v_batch = torch.cat((
            history_v, v_current[:, None].expand(b, tiles, h, n_cur, d)
        ), dim=3).reshape(b * tiles, h, past_len + n_cur, d)
        visible_lengths = tuple(past_len + min(start + chunk_tokens, n_cur)
                                for start in range(0, n_cur, chunk_tokens))
        tile_output = _prefix_attention(
            q_tiles.reshape(b * tiles, h, current_len, d), k_batch, v_batch,
            chunk_tokens // tpf * area, visible_lengths, use_flash,
        )
        tile_output = (tile_output.reshape(b, tiles, h, current_len, d)
                       .permute(0, 2, 1, 3, 4).reshape(b, h, tiles * current_len, d))
        output.index_copy_(2, current_ids.reshape(-1), tile_output)
    return output


def grouped_attention(q: Tensor, k_current: Tensor, v_current: Tensor,
                      past_kv: Optional[Tuple[Tensor, Tensor]], chunk_tokens: int,
                      groups: Dict[str, Tensor], layout: SpatialLayout,
                      sink_tokens: int, recent_tokens: int) -> Tensor:
    """Read global, anchor/recent, or tile-local history for each head group.

    All tensors have shape ``[B, H, N, D]``. ``groups`` must be a disjoint,
    exhaustive partition of heads into ``global``, ``temporal`` and ``spatial``;
    validate that partition when constructing it, outside this hot path.
    Every group keeps the full legal in-flight prefix. Returned tensors use the
    active attention dtype; source tensors and their RoPE coordinates are intact.
    """
    if q.ndim != 4 or q.shape != k_current.shape or q.shape != v_current.shape:
        raise ValueError("Q/current K/current V must have matching [B,H,N,D] shapes")
    if chunk_tokens <= 0 or q.shape[2] == 0:
        raise ValueError("chunk_tokens and current query length must be positive")
    if sink_tokens < 0 or recent_tokens < 0:
        raise ValueError("history budgets must be nonnegative")
    if layout.tile_ids.device != q.device:
        raise ValueError("SpatialLayout and attention tensors must use the same device")
    if set(groups) != {"global", "temporal", "spatial"}:
        raise ValueError("groups must contain global, temporal and spatial")
    if any(ids.ndim != 1 or ids.dtype != torch.long for ids in groups.values()):
        raise ValueError("head groups must be one-dimensional long tensors")
    if sum(ids.numel() for ids in groups.values()) != q.shape[1]:
        raise ValueError("head groups must cover every attention head")

    attn_dtype = (torch.get_autocast_dtype(q.device.type)
                  if torch.is_autocast_enabled(q.device.type) else v_current.dtype)
    q = q.to(dtype=attn_dtype)
    k_current = k_current.to(device=q.device, dtype=attn_dtype)
    v_current = v_current.to(device=q.device, dtype=attn_dtype)
    if past_kv is None:
        k_past, v_past = k_current[:, :, :0], v_current[:, :, :0]
    else:
        k_past, v_past = past_kv
        if (k_past.ndim != 4 or k_past.shape != v_past.shape
                or k_past.shape[:2] != q.shape[:2] or k_past.shape[3] != q.shape[3]):
            raise ValueError("past K/V must match batch, head count and head dimension")
        k_past = k_past.to(device=q.device, dtype=attn_dtype)
        v_past = v_past.to(device=q.device, dtype=attn_dtype)
    use_flash = (flash_attn_func is not None and q.is_cuda
                 and attn_dtype in (torch.float16, torch.bfloat16)
                 and q.shape[3] <= 256 and q.shape[3] % 8 == 0
                 and torch.cuda.get_device_capability(q.device)[0] >= 8)
    n_past = k_past.shape[2]
    if n_past == 0 or groups["global"].numel() == q.shape[1]:
        return _chunk_attention(q, k_current, v_current, k_past, v_past, chunk_tokens, use_flash)

    merge_global_temporal = (use_flash and flash_attn_with_kvcache is not None
                             and not torch.is_grad_enabled()
                             and groups["global"].numel() > 0
                             and groups["temporal"].numel() > 0)
    output = torch.empty_like(q)
    if merge_global_temporal:
        heads = torch.cat((groups["global"], groups["temporal"])).to(device=q.device)
        n_global = groups["global"].numel()
        temporal_heads = torch.arange(n_global, heads.numel(), device=q.device)
        result = _merged_global_temporal_attention(
            q.index_select(1, heads), k_current.index_select(1, heads),
            v_current.index_select(1, heads), k_past.index_select(1, heads),
            v_past.index_select(1, heads), chunk_tokens, temporal_heads,
            int(sink_tokens), int(recent_tokens),
        )
        output.index_copy_(1, heads, result)

    # The two ranges are disjoint, including when their requested budgets overlap.
    temporal_ids = None
    if not merge_global_temporal and groups["temporal"].numel():
        anchor_end = min(int(sink_tokens), n_past)
        recent_start = max(anchor_end, n_past - int(recent_tokens))
        temporal_ids = torch.cat((torch.arange(anchor_end, device=q.device),
                                  torch.arange(recent_start, n_past, device=q.device)))
    remaining_groups = ("spatial",) if merge_global_temporal else ("global", "temporal", "spatial")
    for name in remaining_groups:
        heads = groups[name]
        if heads.numel() == 0:
            continue
        heads = heads.to(device=q.device)
        q_group = q.index_select(1, heads)
        k_group, v_group = k_current.index_select(1, heads), v_current.index_select(1, heads)
        past_k_group, past_v_group = k_past.index_select(1, heads), v_past.index_select(1, heads)
        if name == "spatial":
            result = _spatial_attention(q_group, k_group, v_group, past_k_group, past_v_group,
                                        chunk_tokens, layout, use_flash)
        else:
            if name == "temporal":
                past_k_group = past_k_group.index_select(2, temporal_ids)
                past_v_group = past_v_group.index_select(2, temporal_ids)
            result = _chunk_attention(q_group, k_group, v_group, past_k_group, past_v_group,
                                      chunk_tokens, use_flash)
        output.index_copy_(1, heads, result)
    return output
