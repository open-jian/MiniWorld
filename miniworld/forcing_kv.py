"""Independent, opt-in Forcing-KV policy/lifecycle for MiniWorld inference.

The compact-cache protocol and QKV packing primitives are shared with the other
backends; Forcing-KV classification and cache contents are separate. All mutable
state belongs to this rollout, including separate CFG selection state.
"""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import torch

from miniworld.kv_acceleration import KVAccelerationRuntime, KVContext, KVLayerContext


SOURCE_COMMIT = '50ac1f454348e7f0a2e54bde71e1652b20c09fe6'


def checkpoint_identity(path):
    """Stable local checkpoint identity, obtained before generation timing.

Use a content hash instead for a caller that needs profiles portable across
checkpoint paths. The standard sampling entry point uses path/size/mtime.
"""
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    return json.dumps(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns), sort_keys=True)


@dataclass(frozen=True)
class ForcingKVConfig:
    mode: str = 'forcing-kv'
    calibration_frames: int = 8
    threshold: float = .8
    query_samples_per_chunk: int = 32
    sink_frames: int = 1
    segments: int = 6
    retention: float = .33
    profile_path: str = ''
    profile_output: str = ''
    model_identity: str = ''
    profile_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode != 'forcing-kv':
            raise ValueError('ForcingKVConfig only accepts forcing-kv mode')
        for name in ('calibration_frames', 'query_samples_per_chunk', 'segments'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.sink_frames) is not int or self.sink_frames < 0:
            raise ValueError('sink_frames must be a nonnegative integer')
        for name in ('threshold', 'retention'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and in [0,1]')
        if self.profile_path and self.profile_output:
            raise ValueError('Choose a profile input or a calibration export, not both')


class ForcingKVRuntime(KVAccelerationRuntime):
    """Reuse only the common calibration schedule and cache accounting hooks."""
    def __init__(self, config, *, depth, num_heads, height, width, chunk_size,
                 max_cache_frames, branches=('cond', 'uncond')):
        from miniworld.forcing_kv_policy import ForcingKVPolicy
        if height*width % config.segments:
            raise ValueError('Forcing-KV segments must divide spatial tokens per frame')
        self.config = config
        self.depth, self.num_heads = depth, num_heads
        self.height, self.width = height, width
        self.tpf, self.chunk_size = height*width, chunk_size
        self.chunk_tokens = self.tpf*chunk_size
        self.max_cache_frames = max_cache_frames
        self.branches = tuple(branches)
        self.completed_branches = set()
        self.groups = None
        self.policy = ForcingKVPolicy(config, depth, num_heads, height, width,
                                      chunk_size, branches=self.branches)
        self.calibration_forwards = 0
        self.routed_attention_calls = self.compact_attention_calls = 0
        self.commit_attention_calls = self.dense_attention_calls = 0
        self.converted_caches = 0
        self.calibration_observations = []
        self.last_storage_bytes = self.last_logical_frames = 0
        self.current_pack_workspace = None
        self.selection_states = {branch: {} for branch in self.branches}
        self.empty_caches = {branch: {} for branch in self.branches}
        self.cache_summaries = {}
        self.profile_loaded = False

    def begin_forward(self, branch, cache_frames, t, *, commit=False):
        if self.config.profile_path and self.groups is None:
            self.groups = self.policy.load_profile(device=t.device)
            self.profile_loaded = True
        context = super().begin_forward(branch, cache_frames, t, commit=commit)
        return ForcingKVContext(self, branch, context.phase, context.chunks, commit)

    def _make_cache(self, kv, layer, logical_frames, branch):
        from miniworld.forcing_kv_cache import ForcingKVCache
        from miniworld.kv_kernels import CurrentPackWorkspace
        if self.current_pack_workspace is None:
            self.current_pack_workspace = CurrentPackWorkspace()
        cache = ForcingKVCache.from_dense(
            kv, self.groups[layer], logical_frames=logical_frames,
            height=self.height, width=self.width, chunk_size=self.chunk_size,
            sink_frames=self.config.sink_frames, segments=self.config.segments,
            retention=self.config.retention, max_frames=self.max_cache_frames,
            selection_state=self.selection_states[branch],
            selector_layer=layer == min(1, self.depth-1))
        cache.share_current_workspace(self.current_pack_workspace)
        return cache

    def compact(self, cache, logical_frames, *, branch=None):
        if branch not in self.branches:
            raise ValueError('Forcing-KV cache conversion requires its CFG branch')
        if self.groups is None or not cache:
            return cache
        from miniworld.forcing_kv_cache import ForcingKVCache
        result = []
        for layer, item in enumerate(cache):
            if item is not None and not isinstance(item, ForcingKVCache):
                item = self._make_cache(item, layer, logical_frames, branch)
                self.converted_caches += 1
            result.append(item)
        if any(item is not None for item in result):
            self.empty_caches[branch].clear()
        return result

    def record_storage(self, *caches):
        super().record_storage(*caches)
        self.cache_summaries = {branch: [item.summary() for item in cache
                                         if hasattr(item, 'summary')]
                                for branch, cache in zip(self.branches, caches)}

    def summary(self):
        groups = [] if self.groups is None else [
            {name: ids.cpu().tolist() for name, ids in layer.items()} for layer in self.groups]
        updates = {branch: int(state.get('selection_updates', 0))
                   for branch, state in self.selection_states.items()}
        return dict(mode='forcing-kv', enabled=True,
            status='active' if self.groups is not None else 'warmup',
            config=asdict(self.config), groups=groups,
            group_counts={name: sum(len(layer[name]) for layer in groups) for name in ('static', 'dynamic')},
            calibration_forwards=self.calibration_forwards,
            calibration_observations=self.calibration_observations,
            profile_loaded=self.profile_loaded, policy=self.policy.summary(),
            routed_attention_calls=self.routed_attention_calls,
            compact_attention_calls=self.compact_attention_calls,
            commit_attention_calls=self.commit_attention_calls,
            dense_attention_calls=self.dense_attention_calls,
            converted_caches=self.converted_caches,
            fused_current_pack_calls=getattr(self.current_pack_workspace, 'fused_calls', 0),
            selection_updates=sum(updates.values()), selection_updates_per_branch=updates,
            final_cache_storage_bytes=self.last_storage_bytes,
            final_logical_history_frames=self.last_logical_frames,
            cache_summaries=self.cache_summaries, source_commit=SOURCE_COMMIT,
            scope='Independent Forcing-KV strategy with physical grouped caches; no weight changes',
            adaptation='2-frame chunks; separate CFG selection; per-chunk static anchors; dynamic heads retain the legal in-flight prefix; optional MiniWorld-calibrated offline head profile',
            position_semantics='Original logical positions; pinned sink; identical temporal RoPE rebasing on eviction')


class ForcingKVContext(KVContext):
    def for_layer(self, layer):
        return ForcingKVLayerContext(self, layer)


class ForcingKVLayerContext(KVLayerContext):
    def attention(self, q, k_current, v_current, past_kv):
        c, r = self.context, self.context.runtime
        if r.groups is None or (past_kv is not None and hasattr(past_kv, 'attention')):
            return super().attention(q, k_current, v_current, past_kv)
        # An offline profile routes from the first forward, including before
        # any history exists. Reuse empty-cache workspaces during this prefix.
        if past_kv is None:
            cache = r.empty_caches[c.branch].get(self.layer)
            if cache is None:
                cache = r._make_cache((k_current[:, :, :0], v_current[:, :, :0]),
                                      self.layer, 0, c.branch)
                r.empty_caches[c.branch][self.layer] = cache
        else:
            cache = r._make_cache(past_kv, self.layer, past_kv[0].shape[-2]//r.tpf, c.branch)
        r.routed_attention_calls += 1
        r.compact_attention_calls += 1
        if c.commit:
            r.commit_attention_calls += 1
        return cache.attention(q, k_current, v_current, r.chunk_tokens)
