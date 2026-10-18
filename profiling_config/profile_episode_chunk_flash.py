"""Diagnostic CUDA-event profile for episode 002352's two attention backends.

This deliberately runs separately from the uninstrumented end-to-end benchmark:
Python hooks and CUDA events can perturb execution and are never E2E timings.
All history sizes exclude the fixed one-latent sink and current query tokens.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import miniworld.miniworld as mw
from miniworld.denoiser import DiffusionForcingDenoiser
import benchmark_episode_chunk_flash as bench


class Recorder:
    """Time disjoint cache-related operations and inclusive model forwards."""

    def __init__(self, model, backend):
        self.model, self.backend = model, backend
        self.events = []
        self.context = None
        self.attention_layer = None
        self.forward_id = 0
        self.captures = {}
        self.originals = {}

    @staticmethod
    def cache_meta(cache):
        present = [pair for pair in (cache or []) if pair is not None]
        frames = int(present[0][0].shape[-2] // 300) if present else 0
        return {
            "cache_latents_including_sink": frames,
            "history_chunks_excluding_sink": max(0, frames - 1) / 2,
            "cache_logical_bytes_one_cfg_branch": sum(t.numel() * t.element_size() for pair in present for t in pair),
        }

    def timed(self, category, func, args, kwargs, metadata=None):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        meta = dict(metadata if metadata is not None else (self.context or {}))
        if self.attention_layer is not None:
            meta["layer"] = self.attention_layer
        start.record()
        result = func(*args, **kwargs)
        end.record()
        self.events.append((category, meta, start, end))
        return result

    def install(self):
        self.originals = {
            "forward": self.model.net.forward_with_cache,
            "attention": mw.Attention.forward,
            "sdpa": F.scaled_dot_product_attention,
            "flash": mw._chunkwise_flash_attention,
            "cat": torch.cat,
            "mask": mw._build_cached_block_causal_mask,
            "append": DiffusionForcingDenoiser._append_kv_cache,
            "evict": DiffusionForcingDenoiser._evict_and_shift_cache,
        }
        layers = {id(block.attn): i for i, block in enumerate(self.model.net.blocks)}

        def forward(*args, **kwargs):
            old_context = self.context
            x = args[0] if args else kwargs["x"]
            self.context = {
                "forward_id": self.forward_id,
                "role": "commit" if kwargs.get("return_kv", False) else "denoise",
                "current_latents": int(x.shape[2]),
                **self.cache_meta(kwargs.get("past_kv_list")),
            }
            self.forward_id += 1
            kwargs["use_chunk_flash_attn"] = self.backend == "chunk_flash"
            try:
                return self.timed("model_forward", self.originals["forward"], args, kwargs)
            finally:
                self.context = old_context

        def attention(module, *args, **kwargs):
            old_layer = self.attention_layer
            self.attention_layer = layers.get(id(module))
            try:
                return self.originals["attention"](module, *args, **kwargs)
            finally:
                self.attention_layer = old_layer

        def sdpa(*args, **kwargs):
            if self.context is None:
                return self.originals["sdpa"](*args, **kwargs)
            cache_frames = self.context["cache_latents_including_sink"]
            # Retain these existing immutable tensors, with no GPU copy inside
            # the timed region, for a later identical-input kernel comparison.
            if (self.backend == "baseline" and self.attention_layer == 0
                    and self.context["role"] == "denoise"
                    and self.context["current_latents"] == 16
                    and cache_frames in (8, 24, 48) and cache_frames not in self.captures):
                self.captures[cache_frames] = (args[:3], kwargs["attn_mask"], dict(self.context))
            return self.timed("cached_attention", self.originals["sdpa"], args, kwargs)

        def flash(*args, **kwargs):
            return self.timed("cached_attention", self.originals["flash"], args, kwargs)

        def cat(*args, **kwargs):
            dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            if self.attention_layer is not None and dim == -2:
                return self.timed("internal_kv_concat", self.originals["cat"], args, kwargs)
            return self.originals["cat"](*args, **kwargs)

        def mask(*args, **kwargs):
            return self.timed("dense_mask_build", self.originals["mask"], args, kwargs)

        def append(*args, **kwargs):
            cache = args[0] if args else kwargs["cache"]
            return self.timed("cache_append", self.originals["append"], args, kwargs, self.cache_meta(cache))

        def evict(*args, **kwargs):
            cache = args[0] if args else kwargs["cache"]
            return self.timed("cache_evict_and_rope_rebase", self.originals["evict"], args, kwargs, self.cache_meta(cache))

        self.model.net.forward_with_cache = forward
        mw.Attention.forward = attention
        F.scaled_dot_product_attention = sdpa
        mw._chunkwise_flash_attention = flash
        torch.cat = cat
        mw._build_cached_block_causal_mask = mask
        DiffusionForcingDenoiser._append_kv_cache = staticmethod(append)
        DiffusionForcingDenoiser._evict_and_shift_cache = staticmethod(evict)

    def restore(self):
        self.model.net.forward_with_cache = self.originals["forward"]
        mw.Attention.forward = self.originals["attention"]
        F.scaled_dot_product_attention = self.originals["sdpa"]
        mw._chunkwise_flash_attention = self.originals["flash"]
        torch.cat = self.originals["cat"]
        mw._build_cached_block_causal_mask = self.originals["mask"]
        DiffusionForcingDenoiser._append_kv_cache = staticmethod(self.originals["append"])
        DiffusionForcingDenoiser._evict_and_shift_cache = staticmethod(self.originals["evict"])

    def rows(self):
        # One synchronization after the whole rollout; individual operations
        # remain asynchronous and preserve the sampler's scheduling behavior.
        torch.cuda.synchronize()
        return [{"category": category, "backend": self.backend, **meta,
                 "duration_ms": start.elapsed_time(end)} for category, meta, start, end in self.events]


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row["duration_ms"])
    return [{**dict(zip(keys, group)), "count": len(values), "total_ms": sum(values),
             "mean_ms": statistics.mean(values), "median_ms": statistics.median(values)}
            for group, values in groups.items()]


@torch.inference_mode()
def replay(captures):
    rows = []
    for cache_frames, (tensors, mask, meta) in sorted(captures.items()):
        q, k, v = tensors
        functions = {
            "baseline": lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
            "chunk_flash": lambda: mw._chunkwise_flash_attention(q, k, v, 600),
        }
        samples = defaultdict(list)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(4):
                for function in functions.values():
                    function()
            reference, actual = functions["baseline"](), functions["chunk_flash"]()
            error = (reference.float() - actual.float())
            rmse, max_abs = error.square().mean().sqrt().item(), error.abs().max().item()
            for repetition in range(20):
                for name in (("baseline", "chunk_flash") if repetition % 2 == 0 else ("chunk_flash", "baseline")):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    functions[name]()
                    end.record()
                    torch.cuda.synchronize()
                    samples[name].append(start.elapsed_time(end))
        medians = {name: statistics.median(values) for name, values in samples.items()}
        rows.append({
            **meta, "layer": 0, "q_shape": list(q.shape), "k_shape": list(k.shape),
            "q_dtype": str(q.dtype), "k_dtype": str(k.dtype), "v_dtype": str(v.dtype),
            "baseline_median_ms": medians["baseline"], "chunk_flash_median_ms": medians["chunk_flash"],
            "speedup": medians["baseline"] / medians["chunk_flash"],
            "output_rmse": rmse, "output_max_abs": max_abs,
            "samples_ms": dict(samples),
        })
    return rows


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=bench.OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model, _ = bench.load_model()
    prepared = bench.load_prepared(device="cuda")
    all_rows, backends, replay_rows = [], {}, []
    for backend in ("baseline", "chunk_flash"):
        print(f"Diagnostic backend={backend}", flush=True)
        # Warm the actual model and chosen backend without changing a rollout
        # input or cache. The end-to-end benchmark has its own independent warmup.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(2):
                model.net.forward_with_cache(
                    prepared["noise"][:, :, :16], torch.ones(1, 16, device="cuda"),
                    prepared["cond_seq"][:, :16], chunk_size=2,
                    use_chunk_flash_attn=backend == "chunk_flash",
                )
        torch.cuda.synchronize()
        recorder = Recorder(model, backend)
        recorder.install()
        started = time.perf_counter()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = bench.generate(model, prepared, decoder=None)
            torch.cuda.synchronize()
        finally:
            recorder.restore()
        diagnostic_wall = time.perf_counter() - started
        del result
        rows = recorder.rows()
        all_rows.extend(rows)
        write_csv(args.output_dir / f"kv_profile_{backend}_events.csv", rows)
        summary = aggregate(rows, ["category"])
        backends[backend] = {"diagnostic_wall_sec_not_e2e": diagnostic_wall, "categories": summary}
        print(json.dumps(backends[backend], indent=2), flush=True)
        if recorder.captures:
            replay_rows = replay(recorder.captures)
            (args.output_dir / "same_input_attention_replay.json").write_text(json.dumps(replay_rows, indent=2) + "\n")
            write_csv(args.output_dir / "same_input_attention_replay.csv", [
                {key: value for key, value in row.items() if key not in ("samples_ms", "q_shape", "k_shape")}
                for row in replay_rows
            ])
        del recorder
        torch.cuda.empty_cache()
    bins = aggregate(all_rows, ["backend", "category", "role", "history_chunks_excluding_sink", "current_latents"])
    write_csv(args.output_dir / "kv_profile_by_history_chunks.csv", bins)
    write_csv(args.output_dir / "kv_profile_category_totals.csv", aggregate(all_rows, ["backend", "category"]))
    report = {
        "episode": "002352", "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "scope": "Diagnostic full autoregressive rollout without VAE; instrumentation wall time is NOT end-to-end latency",
        "history_definition": "cached latents minus one sink latent, divided by two; excludes current query/inflight tokens",
        "unchanged_sampler_capacity": "original cache=48 latent including 1 sink, so max sliding history=23.5 chunks; current inflight up to 8 chunks",
        "categories": {
            "cached_attention": "SDPA includes autocast, or chunk helper includes casts, all chunk attention calls, and output concatenation",
            "internal_kv_concat": "two K/V concatenations inside each Attention.forward, across all layers and both CFG branches",
            "dense_mask_build": "once per baseline model forward; fast backend omits the mask",
            "cache_append": "all layers outside model forward, both CFG branches",
            "cache_evict_and_rope_rebase": "all layers outside model forward, includes sink-preserving eviction and surviving-key RoPE shift",
            "model_forward": "inclusive denominator; DO NOT add it to its nested attention/concat/mask categories",
        },
        "timing_caveat": "CUDA event intervals include stream execution and possible host submission gaps; hooks add overhead, so use the separate uninstrumented runs for E2E",
        "backends": backends, "identical_input_first_layer_replays": replay_rows,
    }
    (args.output_dir / "kv_profile_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved diagnostic profile to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
