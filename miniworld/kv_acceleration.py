"""Opt-in HeadCast/Dummy-Forcing cache lifecycle for one MiniWorld rollout.

No checkpoint parameters or global model state are changed. Logical video
positions stay independent of the physically retained cache entries.
"""
from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class KVAccelerationConfig:
    mode: str = 'combined'
    calibration_frames: int = 8
    max_relative_rmse: float = .10
    min_cosine: float = .95
    spatial_score_threshold: float = .755
    query_samples_per_chunk: int = 32
    sink_frames: int = 1
    dummy_frames: int = 1
    local_frames: int = 4
    tile_height: int = 5
    tile_width: int = 5
    dummy_fraction: float = .5

    def __post_init__(self):
        if self.mode not in ('dense', 'headcast', 'headcast-source', 'dummy', 'combined'):
            raise ValueError(f'Unknown KV acceleration mode: {self.mode}')
        for name in ('calibration_frames', 'query_samples_per_chunk', 'dummy_frames',
                     'local_frames', 'tile_height', 'tile_width'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(self.sink_frames, int) or self.sink_frames < 0:
            raise ValueError('sink_frames must be a nonnegative integer')
        if not math.isfinite(self.max_relative_rmse) or self.max_relative_rmse < 0:
            raise ValueError('max_relative_rmse must be finite and nonnegative')
        if not math.isfinite(self.min_cosine) or not -1 <= self.min_cosine <= 1:
            raise ValueError('min_cosine must be in [-1,1]')
        if not math.isfinite(self.spatial_score_threshold):
            raise ValueError('spatial_score_threshold must be finite')
        if not math.isfinite(self.dummy_fraction) or not 0 <= self.dummy_fraction <= 1:
            raise ValueError('dummy_fraction must be in [0,1]')


def fast_dense_attention(q, k_current, v_current, past_kv, chunk_tokens):
    """Exact chunk visibility with batched launches, including the warmup."""
    from miniworld.headcast_attention import _prefix_attention, flash_attn_with_kvcache
    dtype = (torch.get_autocast_dtype(q.device.type)
             if torch.is_autocast_enabled(q.device.type) else v_current.dtype)
    q, k, v = q.to(dtype), k_current.to(dtype), v_current.to(dtype)
    n_past = 0
    if past_kv is not None:
        pk, pv = past_kv
        n_past = pk.shape[2]
        # Cast before concatenation: exactly the production kernel inputs, but
        # no temporary full FP32 concatenation followed by another full cast.
        k = torch.cat((pk.to(dtype), k), dim=2)
        v = torch.cat((pv.to(dtype), v), dim=2)
    use_flash = (flash_attn_with_kvcache is not None and q.is_cuda
                 and dtype in (torch.float16, torch.bfloat16)
                 and q.shape[-1] <= 256 and q.shape[-1] % 8 == 0
                 and torch.cuda.get_device_capability(q.device)[0] >= 8)
    lengths = tuple(n_past + min(i + chunk_tokens, q.shape[2])
                    for i in range(0, q.shape[2], chunk_tokens))
    return _prefix_attention(q, k, v, chunk_tokens, lengths, use_flash)


class KVAccelerationRuntime:
    def __init__(self, config, *, depth, num_heads, height, width, chunk_size,
                 max_cache_frames, branches=('cond', 'uncond')):
        self.config = config
        self.depth, self.num_heads = depth, num_heads
        self.height, self.width = height, width
        self.tpf, self.chunk_size = height * width, chunk_size
        self.chunk_tokens = self.tpf * chunk_size
        self.max_cache_frames = max_cache_frames
        self.branches = tuple(branches)
        self.completed_branches = set()
        self.groups = None
        self.policy = None
        self.calibration_forwards = 0
        self.routed_attention_calls = 0
        self.compact_attention_calls = 0
        self.commit_attention_calls = 0
        self.dense_attention_calls = 0
        self.converted_caches = 0
        self.calibration_observations = []
        self.last_storage_bytes = 0
        self.last_logical_frames = 0
        self.current_pack_workspace = None
        if config.mode != 'dense':
            from miniworld.kv_policy import KVPolicy
            self.policy = KVPolicy(config, depth, num_heads, height, width,
                                   chunk_size, branches=self.branches)

    def begin_forward(self, branch, cache_frames, t, *, commit=False):
        if branch not in self.branches:
            raise ValueError(f'Unknown CFG branch {branch}')
        if self.config.mode == 'dense' and self.groups is None:
            self.groups = [dict(global_=torch.arange(self.num_heads, device=t.device))
                           for _ in range(self.depth)]
            self.groups = [{'global': row['global_'], **{
                name: row['global_'][:0] for name in ('sink', 'dummy', 'local', 'spatial')}}
                for row in self.groups]
        phase, chunks = 'route' if self.groups is not None else 'warmup', ()
        if (not commit and self.groups is None
                and branch not in self.completed_branches
                and cache_frames >= self.config.calibration_frames
                and t.shape[1] >= self.chunk_size and bool((t > 0).any())):
            values = [t[:, i:i+self.chunk_size].float().mean().item()
                      for i in range(0, t.shape[1], self.chunk_size)]
            order = sorted(range(len(values)), key=values.__getitem__)
            chunks = tuple(dict.fromkeys((order[0], order[len(order)//2], order[-1])))
            phase = 'calibrate'
            self.calibration_observations.append(dict(
                branch=branch, cache_frames=cache_frames, chunk_indices=list(chunks),
                timesteps=[values[i] for i in chunks]))
        return KVContext(self, branch, phase, chunks, commit)

    def finish_forward(self, context):
        if context.phase != 'calibrate':
            return
        if context.observed_layers != set(range(self.depth)):
            raise RuntimeError('Incomplete KV classification forward')
        self.completed_branches.add(context.branch)
        self.calibration_forwards += 1
        if self.completed_branches == set(self.branches):
            self.groups = self.policy.finalize()

    def compact(self, cache, logical_frames, *, branch=None):
        """One-time transition; discard dense history once both CFGs classify."""
        if self.groups is None or not cache:
            return cache
        from miniworld.kv_cache import CompactKVCache
        if self.current_pack_workspace is None:
            from miniworld.kv_kernels import CurrentPackWorkspace
            self.current_pack_workspace = CurrentPackWorkspace()
        result = []
        for i, item in enumerate(cache):
            if item is not None and not isinstance(item, CompactKVCache):
                item = CompactKVCache.from_dense(
                    item, self.groups[i], logical_frames=logical_frames,
                    height=self.height, width=self.width, chunk_size=self.chunk_size,
                    sink_frames=self.config.sink_frames,
                    dummy_frames=self.config.dummy_frames,
                    local_frames=self.config.local_frames,
                    tile_height=self.config.tile_height, tile_width=self.config.tile_width,
                    max_frames=self.max_cache_frames)
                item.share_current_workspace(self.current_pack_workspace)
                self.converted_caches += 1
            result.append(item)
        return result

    def record_storage(self, *caches):
        total = 0
        seen = set()
        for cache in caches:
            for item in cache:
                if item is None:
                    continue
                if hasattr(item, 'storage_bytes'):
                    total += item.storage_bytes(seen)
                    self.last_logical_frames = item.logical_frames
                else:
                    total += sum(t.numel() * t.element_size() for t in item)
                    self.last_logical_frames = item[0].shape[-2] // self.tpf
        self.last_storage_bytes = total

    def summary(self):
        groups = [] if self.groups is None else [
            {name: ids.cpu().tolist() for name, ids in layer.items()} for layer in self.groups]
        return dict(
            mode=self.config.mode, enabled=True,
            status='active' if self.groups is not None else 'warmup', config=asdict(self.config),
            groups=groups, group_counts={name: sum(len(layer[name]) for layer in groups)
                for name in ('global', 'sink', 'dummy', 'local', 'spatial')},
            calibration_forwards=self.calibration_forwards,
            calibration_observations=self.calibration_observations,
            routed_attention_calls=self.routed_attention_calls,
            compact_attention_calls=self.compact_attention_calls,
            commit_attention_calls=self.commit_attention_calls,
            dense_attention_calls=self.dense_attention_calls,
            converted_caches=self.converted_caches,
            fused_current_pack_calls=getattr(self.current_pack_workspace, 'fused_calls', 0),
            final_cache_storage_bytes=self.last_storage_bytes,
            final_logical_history_frames=self.last_logical_frames,
            policy=None if self.policy is None else self.policy.summary(),
            position_semantics='logical cache frames; sink pinned, post-sink keys rotated on eviction',
            scope='per-rollout inference runtime; physically compact history and sparse past/current attention')


class KVContext:
    def __init__(self, runtime, branch, phase, chunks, commit):
        self.runtime, self.branch, self.phase = runtime, branch, phase
        self.chunks, self.commit = chunks, commit
        self.observed_layers = set()

    def for_layer(self, layer):
        return KVLayerContext(self, layer)


class KVLayerContext:
    def __init__(self, context, layer):
        self.context, self.layer = context, layer

    def attention(self, q, k_current, v_current, past_kv):
        c, r = self.context, self.context.runtime
        r.routed_attention_calls += 1
        if c.commit:
            r.commit_attention_calls += 1
        if past_kv is not None and hasattr(past_kv, 'attention'):
            r.compact_attention_calls += 1
            return past_kv.attention(q, k_current, v_current, r.chunk_tokens)
        r.dense_attention_calls += 1
        output = fast_dense_attention(q, k_current, v_current, past_kv, r.chunk_tokens)
        if c.phase == 'calibrate':
            if self.layer in c.observed_layers:
                raise RuntimeError('Duplicate layer in calibration')
            k_full, v_full = k_current, v_current
            if past_kv is not None:
                k_full = torch.cat((past_kv[0], k_current), dim=2)
                v_full = torch.cat((past_kv[1], v_current), dim=2)
            r.policy.observe(self.layer, c.branch, q, k_full, v_full, output, c.chunks)
            c.observed_layers.add(self.layer)
        return output
