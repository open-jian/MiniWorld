"""One-time head classification for compact streaming KV caches.

The fidelity-gated modes compare the exact routed masks with dense attention.
The ``dummy`` mode instead implements Dummy Forcing's released globally
normalized attention-mass ranking. Its forced head budget is intentionally
not an error guarantee; the same output-error diagnostics are still reported.

All positions are logical positions before physical cache packing. In
particular, earlier in-flight chunks are history relative to a later query
chunk. No candidate may read a future chunk.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


GROUP_NAMES = ("global", "sink", "dummy", "local", "spatial")
CANDIDATE_NAMES = ("sink", "dummy", "spatial", "local")


def _sample_queries(start, end, height, width, count, device):
    """Deterministic spatial stratum centres in each current frame."""
    if end - start <= count:
        return torch.arange(start, end, device=device)
    tpf = height * width
    frames = max(1, math.ceil((end - start) / tpf))
    per_frame = max(1, count // frames)
    ny = min(height, max(1, int(math.sqrt(per_frame))))
    nx = min(width, max(1, math.ceil(per_frame / ny)))
    ys = ((torch.arange(ny, device=device) + 0.5) * height / ny).long()
    xs = ((torch.arange(nx, device=device) + 0.5) * width / nx).long()
    spatial = (ys[:, None] * width + xs[None, :]).flatten()[:per_frame]
    ids = (torch.arange(start, end, tpf, device=device)[:, None] + spatial).flatten()
    return ids[ids < end][:count]


def _errors(output, reference):
    """Per-head worst-batch RRMSE and mean-query cosine."""
    difference = (output - reference).square().mean(dim=(-1, -2))
    energy = reference.square().mean(dim=(-1, -2))
    relative = (difference / energy.clamp_min(1e-20)).sqrt().amax(dim=0)
    cosines = F.cosine_similarity(output, reference, dim=-1, eps=1e-12)
    both_zero = (output.square().sum(-1) == 0) & (reference.square().sum(-1) == 0)
    cosines = torch.where(both_zero, torch.ones_like(cosines), cosines)
    cosine = cosines.mean(dim=-1).amin(dim=0)
    finite = torch.isfinite(relative) & torch.isfinite(cosine)
    return (torch.where(finite, relative, torch.full_like(relative, torch.inf)),
            torch.where(finite, cosine, torch.full_like(cosine, -torch.inf)))


def _source_stats(output, reference):
    """Official HeadCast statistics, reduced over every batch observation.

    Compute p05 - 5*MSE before reducing across batch/CFG/chunks. Combining
    separately worst p05 and MSE can manufacture a failure absent in any
    observation. Source cosine uses PyTorch's default epsilon, including its
    zero-vector behavior.
    """
    cosine = F.cosine_similarity(output, reference, dim=-1)
    mean = cosine.mean(-1)
    p05 = torch.quantile(cosine, 0.05, dim=-1)
    mse = (output-reference).square().mean(dim=(-1, -2))
    score = p05 - 5.0*mse
    finite = torch.isfinite(mean) & torch.isfinite(p05) & torch.isfinite(mse)
    mean = torch.where(finite, mean, torch.full_like(mean, -torch.inf))
    p05 = torch.where(finite, p05, torch.full_like(p05, -torch.inf))
    mse = torch.where(finite, mse, torch.full_like(mse, torch.inf))
    score = torch.where(finite, score, torch.full_like(score, -torch.inf))
    return {"source_cosine": mean.amin(0), "cosine_p05": p05.amin(0),
            "mean_squared_error": mse.amax(0), "source_spatial_score": score.amin(0)}


class KVPolicy:
    """Generation-local classifier; it owns no model parameters or KV tensors.

    ``observe`` is called once for every layer and CFG branch during dense
    calibration. ``finalize`` returns device-resident int64 head indices for
    the compact-cache backend. Every fidelity-gated candidate must pass every
    supplied chunk, branch and batch item.
    """

    def __init__(self, config, depth, num_heads, height, width, chunk_size,
                 branches=("cond", "uncond")):
        self.config = config
        self.mode = getattr(config, "mode", "headcast")
        if self.mode not in ("headcast", "headcast-source", "dummy", "combined"):
            raise ValueError("KVPolicy mode must be headcast, headcast-source, dummy, or combined")
        self.depth, self.num_heads = int(depth), int(num_heads)
        self.height, self.width, self.chunk_size = int(height), int(width), int(chunk_size)
        if min(self.depth, self.num_heads, self.height, self.width, self.chunk_size) < 1:
            raise ValueError("model dimensions and chunk size must be positive")
        self.tpf = self.height * self.width
        self.chunk_tokens = self.chunk_size * self.tpf
        self.branches = tuple(branches)
        if not self.branches or len(set(self.branches)) != len(self.branches):
            raise ValueError("branches must be nonempty and unique")
        self.max_relative_rmse = float(getattr(config, "max_relative_rmse", 0.1))
        self.min_cosine = float(getattr(config, "min_cosine", 0.95))
        self.query_samples = int(getattr(config, "query_samples_per_chunk", 32))
        self.sink_frames = int(getattr(config, "sink_frames", 1))
        self.dummy_frames = int(getattr(config, "dummy_frames", 1))
        self.local_frames = int(getattr(config, "local_frames", 4))
        self.tile_height = int(getattr(config, "tile_height", 5))
        self.tile_width = int(getattr(config, "tile_width", 5))
        self.dummy_fraction = float(getattr(config, "dummy_fraction", 0.5))
        self.spatial_score_threshold = float(getattr(config, "spatial_score_threshold", 0.755))
        if not math.isfinite(self.spatial_score_threshold):
            raise ValueError("spatial_score_threshold must be finite")
        if (not math.isfinite(self.max_relative_rmse) or self.max_relative_rmse < 0
                or not math.isfinite(self.min_cosine) or not -1 <= self.min_cosine <= 1):
            raise ValueError("invalid output-error thresholds")
        if min(self.query_samples, self.tile_height, self.tile_width) < 1:
            raise ValueError("sample count and tile dimensions must be positive")
        if min(self.sink_frames, self.dummy_frames, self.local_frames) < 0:
            raise ValueError("history frame counts must be nonnegative")
        if not math.isfinite(self.dummy_fraction) or not 0 <= self.dummy_fraction <= 1:
            raise ValueError("dummy_fraction must be in [0, 1]")
        self.metrics = {}
        self.controls = {}
        self.observations = []
        self._observed = set()
        self._mass_sums = {}
        self._mass_weights = {}
        self._cost_sums = {}
        self._cost_weights = {}
        self._device = None
        self.groups = None
        self._ranking = None

    def _masks(self, query_ids, n_past, start, end):
        """Masks over the exact dense key prefix for one current query chunk."""
        device = query_ids.device
        keys = torch.arange(n_past + end, device=device)
        chunk_start = n_past + start
        own_chunk = keys >= chunk_start
        sink = (keys < min(n_past, self.sink_frames * self.tpf)) | own_chunk
        dummy = keys >= max(0, chunk_start - self.dummy_frames * self.tpf)
        local = keys >= max(0, chunk_start - self.local_frames * self.tpf)
        query_spatial = query_ids.remainder(self.tpf)
        key_spatial = keys.remainder(self.tpf)
        tile_cols = math.ceil(self.width / self.tile_width)
        q_tile = ((query_spatial // self.width) // self.tile_height * tile_cols
                  + (query_spatial % self.width) // self.tile_width)
        k_tile = ((key_spatial // self.width) // self.tile_height * tile_cols
                  + (key_spatial % self.width) // self.tile_width)
        spatial = q_tile[:, None] == k_tile[None, :]
        return {"sink": sink[None], "dummy": dummy[None],
                "local": local[None], "spatial": spatial}

    @torch.no_grad()
    def observe(self, layer, branch, q, k_full, v_full, full_output, chunk_indices):
        if self.groups is not None:
            raise RuntimeError("cannot observe after policy finalization")
        layer = int(layer)
        if not 0 <= layer < self.depth or branch not in self.branches:
            raise ValueError("unexpected layer or CFG branch")
        if (layer, branch) in self._observed:
            raise RuntimeError("layer and branch were already observed")
        if q.ndim != 4 or k_full.ndim != 4 or k_full.shape != v_full.shape:
            raise ValueError("expected Q,K,V shaped [B,H,N,D] with matching K/V")
        b, heads, n_cur, dim = q.shape
        if (heads != self.num_heads or n_cur < 1 or b < 1 or dim < 1
                or k_full.shape[:2] != q.shape[:2] or k_full.shape[-1] != dim
                or full_output.shape != q.shape):
            raise ValueError("unexpected attention tensor shape")
        n_past = k_full.shape[-2] - n_cur
        if n_past < 0 or n_past % self.tpf:
            raise ValueError("history must contain complete spatial frames")
        if any(t.device != q.device for t in (k_full, v_full, full_output)):
            raise ValueError("all attention tensors must be on the same device")
        if self._device is not None and self._device != q.device:
            raise ValueError("all observations must use the same device")
        chunks = tuple(dict.fromkeys(int(c) for c in chunk_indices))
        n_chunks = math.ceil(n_cur / self.chunk_tokens)
        if not chunks or min(chunks) < 0 or max(chunks) >= n_chunks:
            raise ValueError("chunk_indices must select existing current chunks")
        self._device = q.device
        # Mirror the production attention cast before promoting for diagnostics.
        dtype = (torch.get_autocast_dtype("cuda")
                 if q.is_cuda and torch.is_autocast_enabled("cuda") else v_full.dtype)
        layer_metrics = self.metrics.setdefault(layer, {})
        costs = self._cost_sums.setdefault(layer, {name: 0.0 for name in CANDIDATE_NAMES})
        with torch.autocast(q.device.type, enabled=False):
            k, v = k_full.to(dtype).float(), v_full.to(dtype).float()
            for chunk in chunks:
                start = chunk * self.chunk_tokens
                end = min(start + self.chunk_tokens, n_cur)
                ids = _sample_queries(start, end, self.height, self.width,
                                      self.query_samples, q.device)
                qq = q[:, :, ids].to(dtype).float()
                visible = n_past + end
                scores = (qq @ k[:, :, :visible].transpose(-1, -2)) * dim ** -0.5
                weights = scores.softmax(-1)
                reference = weights @ v[:, :, :visible]
                control_rmse, control_cos = _errors(full_output[:, :, ids].float(), reference)
                previous_control = self.controls.get(layer)
                if previous_control is None:
                    self.controls[layer] = {"relative_rmse": control_rmse, "cosine": control_cos}
                else:
                    previous_control["relative_rmse"] = torch.maximum(previous_control["relative_rmse"], control_rmse)
                    previous_control["cosine"] = torch.minimum(previous_control["cosine"], control_cos)
                masks = self._masks(ids, n_past, start, end)
                for name, mask in masks.items():
                    candidate = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v[:, :, :visible]
                    relative, cosine = _errors(candidate, reference)
                    previous = layer_metrics.get(name)
                    if previous is None:
                        layer_metrics[name] = {"relative_rmse": relative, "cosine": cosine}
                    else:
                        previous["relative_rmse"] = torch.maximum(previous["relative_rmse"], relative)
                        previous["cosine"] = torch.minimum(previous["cosine"], cosine)
                    if self.mode == "headcast-source":
                        source = _source_stats(candidate, reference)
                        for key, value in source.items():
                            stored = layer_metrics[name].get(key)
                            if stored is None:
                                layer_metrics[name][key] = value
                            elif key == "mean_squared_error":
                                layer_metrics[name][key] = torch.maximum(stored, value)
                            else:
                                layer_metrics[name][key] = torch.minimum(stored, value)
                    # Spatial edge tiles can have different cardinality. This
                    # records actual sampled-query cost rather than nominal area.
                    costs[name] += float(mask.sum(-1).float().mean().item()) * ids.numel()
                self._cost_weights[layer] = self._cost_weights.get(layer, 0) + ids.numel()
                # Released Dummy Forcing ranks sink vs *all* preceding context,
                # excluding own current chunk. Earlier in-flight chunks count
                # as history here, including the immediate PAF preceding frame.
                sink_end = min(n_past, self.sink_frames * self.tpf)
                own_start = n_past + start
                mass = torch.stack((weights[..., :sink_end].sum(-1),
                                    weights[..., sink_end:own_start].sum(-1),
                                    weights[..., own_start:].sum(-1)), dim=-1)
                mass_sum = mass.sum(dim=(0, 2))
                self._mass_sums[layer] = self._mass_sums.get(layer, 0) + mass_sum
                self._mass_weights[layer] = self._mass_weights.get(layer, 0) + b * ids.numel()
        self._observed.add((layer, branch))
        self.observations.append({"layer": layer, "branch": branch, "batch_size": b,
                                  "past_frames": n_past // self.tpf,
                                  "current_tokens": n_cur, "chunks": list(chunks),
                                  "attention_input_dtype": str(dtype)})

    def _passes(self, layer, name):
        metric = self.metrics[layer][name]
        if self.mode == "headcast-source":
            if name in ("sink", "dummy"):
                return metric["source_cosine"] >= self.min_cosine
            if name == "spatial":
                return ((metric["source_cosine"] > self.min_cosine)
                        & (metric["source_spatial_score"] >= self.spatial_score_threshold))
            return torch.zeros_like(metric["source_cosine"], dtype=torch.bool)
        return ((metric["relative_rmse"] <= self.max_relative_rmse)
                & (metric["cosine"] >= self.min_cosine))

    def finalize(self):
        if self.groups is not None:
            return self.groups
        required = {(layer, branch) for layer in range(self.depth) for branch in self.branches}
        if self._observed != required:
            missing = sorted(required - self._observed)
            raise RuntimeError(f"incomplete KV calibration: missing {missing}")
        assignment = torch.zeros((self.depth, self.num_heads), dtype=torch.long, device=self._device)
        if self.mode == "dummy":
            masses = torch.stack([self._mass_sums[layer] / self._mass_weights[layer]
                                  for layer in range(self.depth)])
            if not bool(torch.isfinite(masses).all()):
                raise RuntimeError("nonfinite attention mass cannot define Dummy Forcing groups")
            flat = masses.reshape(-1, 3)
            sink_norm = flat[:, 0] / flat[:, 0].sum().clamp_min(1e-20)
            neighbor_norm = flat[:, 1] / flat[:, 1].sum().clamp_min(1e-20)
            costs = torch.maximum(sink_norm, neighbor_norm)
            # floor matches a fixed integer budget supplied to the source code.
            num_dummy = int(self.depth * self.num_heads * self.dummy_fraction)
            chosen = torch.argsort(costs, stable=True)[:num_dummy]
            flat_assignment = torch.where(sink_norm >= neighbor_norm, 1, 3)
            flat_assignment[chosen] = 2
            assignment = flat_assignment.reshape_as(assignment)
            self._ranking = {"attention_mass": masses, "sink_normalized": sink_norm.reshape_as(assignment),
                             "neighbor_normalized": neighbor_norm.reshape_as(assignment),
                             "opportunity_cost": costs.reshape_as(assignment),
                             "num_dummy": num_dummy}
        else:
            for layer in range(self.depth):
                priority = ["sink", "dummy", "spatial"]
                if self.mode == "combined":
                    priority = sorted(CANDIDATE_NAMES, key=lambda name: self._cost_sums[layer][name])
                free = torch.ones(self.num_heads, dtype=torch.bool, device=self._device)
                for name in priority:
                    selected = free & self._passes(layer, name)
                    assignment[layer, selected] = GROUP_NAMES.index(name)
                    free &= ~selected
        self.groups = [{name: (assignment[layer] == index).nonzero(as_tuple=True)[0]
                        for index, name in enumerate(GROUP_NAMES)} for layer in range(self.depth)]
        return self.groups

    def summary(self):
        def values(tensor):
            return tensor.detach().cpu().tolist()

        groups = [] if self.groups is None else [
            {name: values(ids) for name, ids in group.items()} for group in self.groups]
        metrics = []
        for layer in sorted(self.metrics):
            candidates = {}
            for name in CANDIDATE_NAMES:
                metric = self.metrics[layer][name]
                candidates[name] = {"relative_rmse": values(metric["relative_rmse"]),
                                    "cosine": values(metric["cosine"]),
                                    "passes": values(self._passes(layer, name)),
                                    "mean_retained_keys": self._cost_sums[layer][name] / self._cost_weights[layer]}
                for key in ("source_cosine", "cosine_p05", "mean_squared_error", "source_spatial_score"):
                    if key in metric:
                        candidates[name][key] = values(metric[key])
            metrics.append({"layer": layer, "candidates": candidates,
                            "dense_reference_control": {key: values(value)
                                                        for key, value in self.controls[layer].items()}})
        result = {
            "mode": self.mode, "status": "active" if self.groups is not None else "calibrating",
            "groups": groups,
            "group_counts": {name: sum(len(group[name]) for group in groups) for name in GROUP_NAMES},
            "fidelity_guarded": self.mode != "dummy",
            "thresholds": {"max_relative_rmse": self.max_relative_rmse, "min_cosine": self.min_cosine,
                           "spatial_score_threshold": self.spatial_score_threshold},
            "relative_rmse_guarded": self.mode in ("headcast", "combined"),
            "source_score_aggregation": "minimum paired p05-minus-five-MSE score over sampled chunks, batches and CFG branches",
            "aggregation": ("minimum source mean cosine and paired spatial p05-minus-five-MSE score across every sampled chunk, batch item and CFG branch"
                            if self.mode == "headcast-source" else "max RRMSE and min mean-query cosine across every sampled chunk, batch item and CFG branch"),
            "reference": "FP32 dense attention evaluated on actual low-precision kernel Q/K/V inputs",
            "mask_semantics": {
                "global": "all committed history plus legal current prefix",
                "sink": "first sink frames of committed history plus own current chunk",
                "dummy": "immediately preceding dummy frames of logical prefix plus own current chunk",
                "local": "immediately preceding local frames of logical prefix plus own current chunk",
                "spatial": "same non-overlapping spatial tile across history and legal current prefix",
            },
            "selection": ("released Dummy Forcing global normalized sink/neighbor mass rank; fixed dummy budget; no output-error guard"
                          if self.mode == "dummy" else "official HeadCast cosine and spatial p05-minus-five-MSE gates; sink, dummy, spatial, global priority"
                          if self.mode == "headcast-source" else "sink, dummy, spatial, global priority"
                          if self.mode == "headcast" else "minimum sampled mean retained-key count among passing candidates; global fallback"),
            "observations": list(self.observations), "metrics": metrics,
        }
        if self._ranking is not None:
            result["dummy_ranking"] = {key: values(value) if torch.is_tensor(value) else value
                                       for key, value in self._ranking.items()}
            result["dummy_ranking"]["source"] = "csguoh/DummyForcing@28128ed6204dcb4c572bf8719955124a5f2dbe20/wan/modules/dummyforcing.py:28-63"
            result["dummy_ranking"]["adaptation"] = "mean mass over all sampled chunks/batches/CFG; actual sink boundaries; all earlier in-flight tokens are neighbor history"
        return result
