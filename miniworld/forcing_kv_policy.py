"""MiniWorld head profiling adapted from the released Forcing-KV classifier.

The source classifies heads by non-sink attention mass in the current chunk
and the immediately preceding frame. Its three-frame Wan chunk therefore uses
LAST_K=4; MiniWorld uses ``chunk_size + 1`` rather than copying that constant or
the published Wan head lists. We average scores across sampled noise chunks,
batch items and CFG branches, then keep the allocation fixed for a generation.

An exported profile can be loaded for subsequent generations. Profiles bind the
head allocation to model identity, conditioning contract and attention geometry.
They contain head indices, never a model's K/V tensors.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

import torch

from .kv_policy import _sample_queries


SOURCE_COMMIT = "50ac1f454348e7f0a2e54bde71e1652b20c09fe6"
PROFILE_FORMAT = "miniworld_forcing_kv_profile"
PROFILE_VERSION = 1
GROUP_NAMES = ("static", "dynamic")


class ForcingKVPolicy:
    """Generation-local static/dynamic head classifier and offline profile I/O.

    ``load_profile(device=...)`` bypasses online observation and returns the
    allocation on the requested device. Otherwise observe each layer/branch
    once, then call ``finalize()``. A configured ``profile_output`` is written
    atomically by finalize. Persisted profiles require a nonempty model identity.
    """

    def __init__(self, config, depth, num_heads, height, width, chunk_size,
                 branches=("cond", "uncond")):
        self.config = config
        self.depth, self.num_heads = int(depth), int(num_heads)
        self.height, self.width, self.chunk_size = int(height), int(width), int(chunk_size)
        if min(self.depth, self.num_heads, self.height, self.width, self.chunk_size) < 1:
            raise ValueError("model dimensions and chunk size must be positive")
        self.tpf = self.height * self.width
        self.chunk_tokens = self.chunk_size * self.tpf
        self.branches = tuple(branches)
        if (not self.branches or len(set(self.branches)) != len(self.branches)
                or any(not isinstance(branch, str) or not branch for branch in self.branches)):
            raise ValueError("branches must be nonempty unique strings")
        self.threshold = float(getattr(config, "threshold", 0.8))
        self.query_samples = int(getattr(config, "query_samples_per_chunk", 32))
        self.sink_frames = int(getattr(config, "sink_frames", 1))
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be finite and in [0, 1]")
        if self.query_samples < 1 or self.sink_frames < 0:
            raise ValueError("sample count must be positive and sink_frames nonnegative")
        self.profile_path = str(getattr(config, "profile_path", "") or "")
        self.profile_output = str(getattr(config, "profile_output", "") or "")
        self.model_identity = str(getattr(config, "model_identity", "") or "")
        metadata = getattr(config, "profile_metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("profile_metadata must be a JSON dictionary")
        try:
            self.profile_metadata = json.loads(json.dumps(metadata, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("profile_metadata must contain finite JSON values") from exc
        self.groups = None
        self.profile_loaded = False
        self.observations = []
        self._observed = set()
        self._score_sums = {}
        self._score_counts = {}
        self._scores = None
        self._device = None

    def profile_contract(self):
        """Return the exact compatibility contract used for persisted profiles."""
        return {"depth": self.depth, "num_heads": self.num_heads,
                "height": self.height, "width": self.width,
                "chunk_size": self.chunk_size, "sink_frames": self.sink_frames,
                "threshold": self.threshold,
                "query_samples_per_chunk": self.query_samples,
                "branches": list(self.branches), "model_identity": self.model_identity,
                "profile_metadata": self.profile_metadata,
                "first_layer_policy": "all-static"}

    @torch.no_grad()
    def observe(self, layer, branch, q, k_full, v_full, full_output, chunk_indices):
        if self.groups is not None:
            raise RuntimeError("cannot observe after policy finalization or profile load")
        layer = int(layer)
        if not 0 <= layer < self.depth or branch not in self.branches:
            raise ValueError("unexpected layer or CFG branch")
        if (layer, branch) in self._observed:
            raise RuntimeError("layer and branch were already observed")
        if q.ndim != 4 or k_full.ndim != 4:
            raise ValueError("expected Q,K shaped [B,H,N,D]")
        b, heads, n_cur, dim = q.shape
        if (heads != self.num_heads or min(b, n_cur, dim) < 1
                or k_full.shape[:2] != q.shape[:2] or k_full.shape[-1] != dim):
            raise ValueError("unexpected attention tensor shape")
        if v_full is not None and v_full.shape != k_full.shape:
            raise ValueError("K and V must have matching shapes")
        if full_output is not None and full_output.shape != q.shape:
            raise ValueError("full_output must match Q shape")
        tensors = (k_full, v_full, full_output)
        if any(t is not None and t.device != q.device for t in tensors):
            raise ValueError("all attention tensors must be on the same device")
        if self._device is not None and self._device != q.device:
            raise ValueError("all observations must use the same device")
        n_past = k_full.shape[-2] - n_cur
        if n_past < 0 or n_past % self.tpf:
            raise ValueError("history must contain complete spatial frames")
        chunks = tuple(dict.fromkeys(int(c) for c in chunk_indices))
        if not chunks or min(chunks) < 0 or max(chunks) >= math.ceil(n_cur / self.chunk_tokens):
            raise ValueError("chunk_indices must select existing current chunks")
        dtype = (torch.get_autocast_dtype("cuda")
                 if q.is_cuda and torch.is_autocast_enabled("cuda")
                 else v_full.dtype if v_full is not None else q.dtype)
        score_sum = torch.zeros(self.num_heads, device=q.device, dtype=torch.float32)
        score_count = 0
        samples = []
        with torch.autocast(q.device.type, enabled=False):
            k = k_full.to(dtype).float()
            for chunk in chunks:
                start = chunk * self.chunk_tokens
                end = min(start + self.chunk_tokens, n_cur)
                ids = _sample_queries(start, end, self.height, self.width,
                                      self.query_samples, q.device)
                own_start, visible = n_past + start, n_past + end
                sink_end = min(n_past, self.sink_frames * self.tpf)
                local_start = max(sink_end, own_start - self.tpf)
                qq = q[:, :, ids].to(dtype).float()
                logits = (qq @ k[:, :, :visible].transpose(-1, -2)) * dim ** -0.5
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("nonfinite attention logits cannot define Forcing-KV groups")
                # The source divides summed local mass by summed non-sink mass
                # across query rows. A plain mean of per-query ratios changes
                # that statistic when sink affinity differs between queries.
                # Log-space query weighting preserves the ratio of sums while
                # avoiding cancellation/underflow for strongly sink-heavy heads.
                nonsink = logits[..., sink_end:]
                weights = nonsink.softmax(-1)
                nonsink_log_mass = nonsink.logsumexp(-1) - logits.logsumexp(-1)
                query_weights = nonsink_log_mass.softmax(-1)
                scores = (weights[..., local_start - sink_end:].sum(-1) * query_weights).sum(-1)
                score_sum += scores.sum(0)
                score_count += b
                samples.append({"chunk": chunk, "query_count": ids.numel(),
                                "visible_tokens": visible,
                                "sink_tokens": sink_end,
                                "local_start_token": local_start})
        self._device = q.device
        self._score_sums[layer] = self._score_sums.get(layer, 0) + score_sum
        self._score_counts[layer] = self._score_counts.get(layer, 0) + score_count
        self._observed.add((layer, branch))
        self.observations.append({"layer": layer, "branch": branch,
                                  "batch_size": b, "past_frames": n_past // self.tpf,
                                  "current_tokens": n_cur, "chunks": list(chunks),
                                  "samples": samples, "attention_input_dtype": str(dtype)})

    def finalize(self):
        if self.groups is not None:
            return self.groups
        required = {(layer, branch) for layer in range(self.depth) for branch in self.branches}
        if self._observed != required:
            raise RuntimeError(f"incomplete Forcing-KV calibration: missing {sorted(required - self._observed)}")
        self._scores = torch.stack([self._score_sums[layer] / self._score_counts[layer]
                                    for layer in range(self.depth)])
        if not bool(torch.isfinite(self._scores).all()):
            raise RuntimeError("nonfinite attention scores cannot define Forcing-KV groups")
        static = self._scores >= self.threshold
        # Released Wan keeps block 0 on its short dense path regardless of the
        # offline head list. MiniWorld's static group implements that short path.
        static[0] = True
        self.groups = [{"static": static[layer].nonzero(as_tuple=True)[0],
                        "dynamic": (~static[layer]).nonzero(as_tuple=True)[0]}
                       for layer in range(self.depth)]
        if self.profile_output:
            self.save_profile()
        return self.groups

    def save_profile(self, path=None):
        """Atomically save a finalized allocation; return the saved path."""
        if self.groups is None:
            raise RuntimeError("finalize the Forcing-KV policy before saving a profile")
        if not self.model_identity:
            raise ValueError("model_identity is required to save a reusable profile")
        destination = str(path or self.profile_output)
        if not destination:
            raise ValueError("profile output path is required")
        payload = {"format": PROFILE_FORMAT, "version": PROFILE_VERSION,
                   "source_commit": SOURCE_COMMIT, "contract": self.profile_contract(),
                   "groups": [{name: ids.cpu().tolist() for name, ids in group.items()}
                              for group in self.groups],
                   "scores": self._scores.cpu().tolist(),
                   "observations": self.observations}
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                             prefix=f".{destination.name}.", suffix=".tmp",
                                             delete=False) as stream:
                temporary = stream.name
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, destination)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        return str(destination)

    def load_profile(self, path=None, device=None):
        """Load a compatible MiniWorld allocation, failing on any mismatch."""
        if self.groups is not None or self._observed:
            raise RuntimeError("load a profile before observations or policy finalization")
        if not self.model_identity:
            raise ValueError("model_identity is required to load a reusable profile")
        source = str(path or self.profile_path)
        if not source:
            raise ValueError("profile input path is required")
        with open(source, encoding="utf-8") as stream:
            payload = json.load(stream)
        if (not isinstance(payload, dict) or payload.get("format") != PROFILE_FORMAT
                or payload.get("version") != PROFILE_VERSION):
            raise ValueError("unsupported Forcing-KV profile format or version")
        expected = self.profile_contract()
        actual = payload.get("contract")
        if not isinstance(actual, dict) or actual != expected:
            changed = sorted(set(expected) | set(actual or {})) if isinstance(actual, dict) else ["contract"]
            if isinstance(actual, dict):
                changed = [key for key in changed if actual.get(key) != expected.get(key)]
            raise ValueError(f"incompatible Forcing-KV profile: {', '.join(changed)}")
        groups = payload.get("groups")
        if not isinstance(groups, list) or len(groups) != self.depth:
            raise ValueError("profile groups must contain exactly one entry per layer")
        for layer, group in enumerate(groups):
            if (not isinstance(group, dict) or set(group) != set(GROUP_NAMES)
                    or any(not isinstance(group[name], list) for name in GROUP_NAMES)):
                raise ValueError(f"invalid profile head groups at layer {layer}")
            ids = group["static"] + group["dynamic"]
            if (any(type(head) is not int for head in ids)
                    or sorted(ids) != list(range(self.num_heads))):
                raise ValueError(f"profile heads must form a disjoint complete partition at layer {layer}")
        if groups[0]["dynamic"]:
            raise ValueError("profile layer 0 must use the source short dense/static path")
        try:
            scores = torch.tensor(payload["scores"], dtype=torch.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("profile scores must be a numeric layer/head matrix") from exc
        if (scores.shape != (self.depth, self.num_heads) or not bool(torch.isfinite(scores).all())
                or bool(((scores < -1e-6) | (scores > 1 + 1e-6)).any())):
            raise ValueError("profile scores must be finite values in [0, 1] for every head")
        for layer in range(1, self.depth):
            selected = (scores[layer] >= self.threshold).nonzero(as_tuple=True)[0].tolist()
            if sorted(groups[layer]["static"]) != selected:
                raise ValueError(f"profile head groups disagree with scores at layer {layer}")
        observations = payload.get("observations")
        if not isinstance(observations, list):
            raise ValueError("profile observations must be a list")
        self._device = torch.device(device or "cpu")
        self._scores = scores.to(self._device)
        self.groups = [{name: torch.tensor(ids, dtype=torch.long, device=self._device)
                        for name, ids in group.items()} for group in groups]
        self.observations = observations
        self.profile_loaded = True
        self.profile_path = source
        return self.groups

    def summary(self):
        groups = [] if self.groups is None else [
            {name: ids.detach().cpu().tolist() for name, ids in group.items()}
            for group in self.groups]
        return {"mode": "forcing-kv", "status": "active" if self.groups is not None else "calibrating",
                "profile_loaded": self.profile_loaded, "profile_path": self.profile_path,
                "profile_output": self.profile_output, "groups": groups,
                "group_counts": {name: sum(len(group[name]) for group in groups) for name in GROUP_NAMES},
                "scores": [] if self._scores is None else self._scores.detach().cpu().tolist(),
                "threshold": self.threshold, "fidelity_guarded": False,
                "aggregation": "ratio of summed local and non-sink attention masses over sampled query rows, then equal mean over sampled chunks, batch items and CFG branches",
                "mask_semantics": "own current chunk plus immediately preceding frame, excluding sink from both mass terms; no future chunks",
                "first_layer_override": "all heads static, matching source block-0 short dense path",
                "source": {"repository": "https://github.com/zju-jiyicheng/Forcing-KV",
                           "commit": SOURCE_COMMIT,
                           "classifier": "configs_head/head_profile.py:31-98",
                           "offline_allocation": "wan/modules/causal_model_forcingkv.py:845-968"},
                "adaptation": "MiniWorld chunk_size+1 frames replace Wan LAST_K=4; deterministic spatial samples from low/mid/high-noise chunks and both CFG branches; Wan head indices are never reused",
                "contract": self.profile_contract(), "observations": list(self.observations)}
