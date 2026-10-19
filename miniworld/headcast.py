"""Per-generation, inference-only HeadCast adaptation for MiniWorld.

Inspired by HeadCast's one-time output-based classification. MiniWorld keeps
its existing dense backing KV cache, clean commits, RoPE and chunk causality.
The selectable paths are sink+recent history, tiled history, and full history;
all paths preserve the entire legal in-flight prefix. Calibration evaluates
the same tiles used at inference, across both CFG branches and mixed-noise
chunks, before any sparse output is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F

from miniworld.headcast_attention import SpatialLayout, grouped_attention, spatial_history_mask


@dataclass(frozen=True)
class HeadCastConfig:
    enabled: bool = False
    tile_height: int = 5
    tile_width: int = 5
    recent_chunks: int = 2
    max_relative_rmse: float = 0.10
    min_cosine: float = 0.95
    query_samples_per_chunk: int = 32

    def __post_init__(self):
        for name in ('tile_height', 'tile_width', 'recent_chunks', 'query_samples_per_chunk'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(self.max_relative_rmse) or self.max_relative_rmse < 0:
            raise ValueError('max_relative_rmse must be finite and nonnegative')
        if not math.isfinite(self.min_cosine) or not -1 <= self.min_cosine <= 1:
            raise ValueError('min_cosine must be in [-1, 1]')


def _sample_queries(start, end, height, width, count, device):
    """Stratum centres in every current frame; no random-number consumption."""
    tpf = height * width
    if end - start <= count:
        return torch.arange(start, end, device=device)
    frames = max(1, math.ceil((end - start) / tpf))
    per_frame = max(1, count // frames)
    ny = min(height, max(1, int(math.sqrt(per_frame))))
    nx = min(width, max(1, math.ceil(per_frame / ny)))
    ys = ((torch.arange(ny, device=device) + .5) * height / ny).long()
    xs = ((torch.arange(nx, device=device) + .5) * width / nx).long()
    spatial = (ys[:, None] * width + xs[None, :]).flatten()[:per_frame]
    ids = (torch.arange(start, end, tpf, device=device)[:, None] + spatial).flatten()
    return ids[ids < end][:count]


class HeadCastRuntime:
    """Plain Python state owned by one streaming call; never a model module."""
    def __init__(self, config: HeadCastConfig, depth: int, num_heads: int,
                 grid_height: int, grid_width: int, chunk_size: int,
                 sink_frames: int, max_cache_frames: int,
                 branches=('cond', 'uncond')):
        self.config = config
        self.depth, self.num_heads = depth, num_heads
        self.height, self.width = grid_height, grid_width
        self.chunk_size = chunk_size
        self.tpf = grid_height * grid_width
        self.chunk_tokens = chunk_size * self.tpf
        self.sink_tokens = sink_frames * self.tpf
        self.recent_tokens = config.recent_chunks * self.chunk_tokens
        self.max_cache_frames = max_cache_frames
        self.branches = tuple(branches)
        if not self.branches or len(set(self.branches)) != len(self.branches):
            raise ValueError('branches must be nonempty and unique')
        self.completed_branches = set()
        self.groups = None
        self.layout = None
        self.metrics = {}
        self.calibration_forwards = 0
        self.routed_attention_calls = 0
        self.calibration_observations = []

    def begin_forward(self, branch, cache_frames, t):
        if branch not in self.branches:
            raise ValueError(f'Unknown HeadCast branch: {branch}')
        if not self.config.enabled:
            return None
        if self.groups is not None:
            return HeadCastContext(self, branch, 'route', ())
        if (self.max_cache_frames <= 0 or cache_frames < self.max_cache_frames
                or branch in self.completed_branches):
            return None
        # Calls are explicitly denoising-only. Also reject a fully clean window
        # so a caller cannot classify from a t=0 cache commit by accident.
        if t.ndim != 2 or t.shape[1] == 0 or not bool((t > 0).any()):
            return None
        values = [t[:, start:start+self.chunk_size].float().mean().item()
                  for start in range(0, t.shape[1], self.chunk_size)]
        order = sorted(range(len(values)), key=values.__getitem__)
        chunks = tuple(dict.fromkeys((order[0], order[len(order)//2], order[-1])))
        if self.layout is None:
            self.layout = SpatialLayout(self.height, self.width, self.config.tile_height,
                                        self.config.tile_width, t.device)
        self.calibration_observations.append({
            'branch': branch, 'cache_frames': cache_frames,
            'chunk_indices': list(chunks), 'timesteps': [values[i] for i in chunks],
        })
        return HeadCastContext(self, branch, 'calibrate', chunks)

    def finish_forward(self, context):
        if context is None or context.phase != 'calibrate':
            return
        if context.runtime is not self:
            raise ValueError('HeadCast context belongs to another generation')
        if context.observed_layers != set(range(self.depth)):
            raise RuntimeError('Incomplete HeadCast calibration forward')
        if context.branch in self.completed_branches:
            raise RuntimeError('HeadCast branch calibrated more than once')
        self.completed_branches.add(context.branch)
        self.calibration_forwards += 1
        if self.completed_branches != set(self.branches):
            return
        self.groups = []
        for layer in range(self.depth):
            metrics = self.metrics[layer]
            temporal = ((metrics['temporal']['relative_rmse'] <= self.config.max_relative_rmse)
                        & (metrics['temporal']['cosine'] >= self.config.min_cosine))
            spatial = ((metrics['spatial']['relative_rmse'] <= self.config.max_relative_rmse)
                       & (metrics['spatial']['cosine'] >= self.config.min_cosine) & ~temporal)
            self.groups.append({
                'temporal': temporal.nonzero(as_tuple=True)[0],
                'spatial': spatial.nonzero(as_tuple=True)[0],
                'global': (~temporal & ~spatial).nonzero(as_tuple=True)[0],
            })

    def summary(self):
        groups = [] if self.groups is None else [
            {name: ids.cpu().tolist() for name, ids in layer.items()} for layer in self.groups]
        return {
            'enabled': self.config.enabled,
            'status': 'active' if self.groups is not None else ('calibrating' if self.completed_branches else 'warmup'),
            'config': asdict(self.config),
            'groups': groups,
            'group_counts': {name: sum(len(layer[name]) for layer in groups)
                             for name in ('global', 'temporal', 'spatial')},
            'calibration_forwards': self.calibration_forwards,
            'routed_attention_calls': self.routed_attention_calls,
            'calibration_observations': self.calibration_observations,
            'completed_branches': sorted(self.completed_branches),
            'classification': 'temporal priority, then spatial, otherwise global; all observed chunks/batches/CFG branches must pass',
            'spatial_pattern': 'non-overlapping tiles in committed history; complete legal in-flight prefix',
            'kv_storage': 'unchanged dense backing cache; no KV memory compression claimed',
        }


class HeadCastContext:
    def __init__(self, runtime, branch, phase, chunks):
        self.runtime, self.branch, self.phase, self.chunks = runtime, branch, phase, chunks
        self.observed_layers = set()

    @property
    def routing(self):
        return self.phase == 'route'

    def for_layer(self, layer):
        if not 0 <= layer < self.runtime.depth:
            raise ValueError('Layer index outside HeadCast runtime')
        return HeadCastLayerContext(self, layer)


class HeadCastLayerContext:
    def __init__(self, context, layer):
        self.context, self.layer = context, layer
        self.chunk_tokens = context.runtime.chunk_tokens

    @property
    def routing(self):
        if self.context.phase != 'route':
            return False
        groups = self.context.runtime.groups[self.layer]
        return bool(groups['temporal'].numel() + groups['spatial'].numel())

    def route(self, q, k_current, v_current, past_kv, chunk_tokens):
        runtime = self.context.runtime
        if not self.routing:
            raise RuntimeError('Cannot route before completed calibration or for an all-global layer')
        runtime.routed_attention_calls += 1
        return grouped_attention(q, k_current, v_current, past_kv, chunk_tokens,
                                 runtime.groups[self.layer], runtime.layout,
                                 runtime.sink_tokens, runtime.recent_tokens)

    @torch.no_grad()
    def observe(self, q, k_full, v_full, chunk_tokens, full_output):
        if self.context.phase != 'calibrate':
            return
        runtime = self.context.runtime
        if self.layer in self.context.observed_layers:
            raise RuntimeError('A layer was observed twice in one HeadCast forward')
        if chunk_tokens != runtime.chunk_tokens:
            raise ValueError('HeadCast chunk size differs from the sampler')
        n_cur = q.shape[-2]
        n_past = k_full.shape[-2] - n_cur
        if q.shape[1] != runtime.num_heads or n_past < 0:
            raise ValueError('Unexpected HeadCast attention shape')
        dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else v_full.dtype
        with torch.autocast(q.device.type, enabled=False):
            k = k_full.to(dtype).float()
            v = v_full.to(dtype).float()
            for chunk in self.context.chunks:
                start, end = chunk * chunk_tokens, min((chunk+1) * chunk_tokens, n_cur)
                ids = _sample_queries(start, end, runtime.height, runtime.width,
                                      runtime.config.query_samples_per_chunk, q.device)
                qq = q[:, :, ids].to(dtype).float()
                reference = full_output[:, :, ids].float()
                scores = (qq @ k[:, :, :n_past+end].transpose(-1, -2)) * q.shape[-1] ** -.5
                keys = torch.arange(n_past+end, device=q.device)
                # sink_tokens=0 naturally becomes recent-only; never promote
                # the FIFO's first frame to a semantic sink when none exists.
                temporal = ((keys < min(runtime.sink_tokens, n_past))
                            | (keys >= max(0, n_past-runtime.recent_tokens)))
                masks = {'temporal': temporal[None, :],
                         'spatial': spatial_history_mask(ids, n_past, end, runtime.layout)}
                for name, mask in masks.items():
                    output = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v[:, :, :n_past+end]
                    mse = (output-reference).square().mean(dim=(-1, -2))
                    energy = reference.square().mean(dim=(-1, -2))
                    relative = (mse / energy.clamp_min(1e-20)).sqrt().amax(0)
                    cosine = F.cosine_similarity(output, reference, dim=-1, eps=1e-12).mean(-1).amin(0)
                    finite = torch.isfinite(relative) & torch.isfinite(cosine)
                    relative = torch.where(finite, relative, torch.full_like(relative, torch.inf))
                    cosine = torch.where(finite, cosine, torch.full_like(cosine, -torch.inf))
                    current = runtime.metrics.setdefault(self.layer, {}).get(name)
                    if current is None:
                        runtime.metrics[self.layer][name] = {'relative_rmse': relative, 'cosine': cosine}
                    else:
                        current['relative_rmse'] = torch.maximum(current['relative_rmse'], relative)
                        current['cosine'] = torch.minimum(current['cosine'], cosine)
        self.context.observed_layers.add(self.layer)
