"""Bounded pretrained-model A/B benchmark for cached chunk FlashAttention.

This measures a complete ``forward_with_cache`` on deterministic synthetic
latents/actions, including projections, RoPE, KV concatenation, and attention.
It is not an autoregressive rollout or a PSNR/SSIM/LPIPS quality evaluation.

Example (keep artifacts outside the source checkout)::

    CUDA_VISIBLE_DEVICES=2 python profiling_config/benchmark_chunk_flash_attention.py \
        --checkpoint /data2/jian/checkpoints/miniworld/MiniWorld_1b_droid.pt \
        --output-dir /data2/jian/output/Miniworld/DROID_long_horizon/chunk_flash_attention
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from miniworld.denoiser import Denoiser, DenoiserConfig  # noqa: E402


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-latents", type=int, nargs="+", default=[0, 1, 9, 47, 48])
    parser.add_argument("--current-latents", type=int, nargs="+", default=[2, 16])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if min(args.cache_latents) < 0 or min(args.current_latents) <= 0:
        parser.error("cache lengths must be nonnegative and current lengths positive")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("use at least one warmup and two timed repetitions")
    return args


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _error(reference, actual):
    reference = reference.float()
    delta = reference - actual.float()
    rms = reference.square().mean().sqrt().item()
    rmse = delta.square().mean().sqrt().item()
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rmse": rmse,
        "reference_rms": rms,
        "relative_rmse": rmse / max(rms, 1e-12),
        "finite": bool(torch.isfinite(actual).all()),
    }


def _kv_error(reference, actual):
    result = {}
    for component, index in (("k", 0), ("v", 1)):
        errors = [_error(a[index], b[index]) for a, b in zip(reference, actual)]
        result[component] = {
            "max_abs": max(error["max_abs"] for error in errors),
            "rmse": math.sqrt(statistics.mean(error["rmse"] ** 2 for error in errors)),
            "finite": all(error["finite"] for error in errors),
            "dtype_unchanged": all(a[index].dtype == b[index].dtype for a, b in zip(reference, actual)),
            "shape_unchanged": all(a[index].shape == b[index].shape for a, b in zip(reference, actual)),
        }
    return result


def _summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples),
        "samples_ms": samples,
    }


@torch.inference_mode()
def main():
    args = _args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = raw.get("ema_model", raw.get("model", raw))
    # Infer only checkpoint properties; no sampler/sink/window settings change.
    cond_dim = int(weights["net.null_action"].shape[-1])
    trained_latents = int(weights["net.feat_rope.freqs_cos"].shape[0] // 300)
    if max(args.cache_latents) + max(args.current_latents) > trained_latents:
        raise ValueError("This bounded benchmark stays within the trained latent window")
    cfg = DenoiserConfig(
        wm_model="1B", latent_size=(15, 20), latent_channels=48,
        latent_frames=trained_latents, cond_dim=cond_dim,
        cond_per_token=False, cond_dropout_prob=0.1, wm_use_checkpoint=False,
        df_chunk_size=2,
    )
    model = Denoiser(cfg)
    incompatible = model.load_state_dict(weights, strict=True)
    del weights, raw
    model = model.cuda().eval()
    net = model.net
    max_cache = max(args.cache_latents)
    prefix = torch.randn(1, 48, max_cache, 15, 20, device="cuda")
    prefix_cond = 0.2 * torch.randn(1, max_cache, cond_dim, device="cuda")
    past = None
    prefix_offset = 0
    print("Building common baseline cache from synthetic prefix (seed latent + two-latent chunks)", flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        while prefix_offset < max_cache:
            count = 1 if prefix_offset == 0 else min(2, max_cache - prefix_offset)
            _, current_kv = net.forward_with_cache(
                prefix[:, :, prefix_offset:prefix_offset + count],
                torch.zeros(1, count, device="cuda"),
                prefix_cond[:, prefix_offset:prefix_offset + count],
                past_kv_list=past, current_position_offset=prefix_offset,
                return_kv=True, chunk_size=2, use_chunk_flash_attn=False,
            )
            past = current_kv if past is None else [
                (torch.cat((old[0], new[0]), dim=-2), torch.cat((old[1], new[1]), dim=-2))
                for old, new in zip(past, current_kv)
            ]
            prefix_offset += count
    del prefix, prefix_cond
    torch.cuda.synchronize()

    report = {
        "scope": "pretrained complete forward_with_cache; synthetic inputs; NOT rollout quality",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "checkpoint_load_strict": True,
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "production_source_sha256": hashlib.sha256((REPO_ROOT / "miniworld/miniworld.py").read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "weights_dtype": str(next(model.parameters()).dtype),
        "compute_autocast_dtype": "torch.bfloat16",
        "cpu_threads": torch.get_num_threads(),
        "seed": args.seed,
        "warmups_per_backend": args.warmup,
        "repeats_per_backend": args.repeats,
        "tokens_per_latent": 300,
        "latent_per_chunk": 2,
        "trained_latents": trained_latents,
        "cache_definition": "cache_latents includes the one-latent sink; historical window excludes sink and current/inflight tokens",
        "cache_generation": "baseline masked SDPA on common synthetic prefix, first latent then two-latent chunks",
        "timing_includes": "complete model forward_with_cache; projections, conditioning, RoPE, internal KV concatenation, attention, final layer",
        "timing_excludes": "cache generation/eviction, CFG second pass, sampler integration, VAE, file I/O",
        "cases": [],
    }

    for cache_latents in args.cache_latents:
        # Both backends read identical baseline-produced tensors, not their own
        # outputs. Prefix views preserve dtype, positions, and cache contents.
        cache = None if cache_latents == 0 else [
            (k[:, :, :cache_latents * 300], v[:, :, :cache_latents * 300]) for k, v in past
        ]
        cache_bytes = 0 if cache is None else sum(t.numel() * t.element_size() for pair in cache for t in pair)
        for current_latents in args.current_latents:
            current = torch.randn(1, 48, current_latents, 15, 20, device="cuda")
            condition = 0.2 * torch.randn(1, current_latents, cond_dim, device="cuda")
            # The commit call has clean t=0; in-flight timesteps are constant
            # within each two-latent chunk, as in diffusion forcing sampling.
            commit = current_latents == 2
            timestep = torch.zeros(1, 2, device="cuda") if commit else torch.linspace(
                0.05, 0.95, math.ceil(current_latents / 2), device="cuda",
            ).repeat_interleave(2)[:current_latents].unsqueeze(0)

            def call(fast):
                return net.forward_with_cache(
                    current, timestep, condition, past_kv_list=cache,
                    current_position_offset=cache_latents, return_kv=commit,
                    chunk_size=2, use_chunk_flash_attn=fast,
                )

            with torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(args.warmup):
                    call(False)
                    call(True)
                reference, reference_kv = call(False)
                actual, actual_kv = call(True)
                output_error = _error(reference, actual)
                cache_error = _kv_error(reference_kv, actual_kv) if commit else None
                del reference, actual, reference_kv, actual_kv
                events = {False: [], True: []}
                walls = {False: [], True: []}
                peaks = {False: [], True: []}
                for repeat in range(args.repeats):
                    # Alternate order to reduce temperature/clock/order bias.
                    for fast in ((False, True) if repeat % 2 == 0 else (True, False)):
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        base_memory = torch.cuda.memory_allocated()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        wall_start = time.perf_counter()
                        start.record()
                        result = call(fast)
                        end.record()
                        torch.cuda.synchronize()
                        walls[fast].append((time.perf_counter() - wall_start) * 1000)
                        events[fast].append(start.elapsed_time(end))
                        peaks[fast].append(torch.cuda.max_memory_allocated() - base_memory)
                        del result
            baseline = _summarize(events[False])
            fast_result = _summarize(events[True])
            case = {
                "cache_latents_including_sink": cache_latents,
                "sink_latents": min(1, cache_latents),
                "history_chunks_excluding_sink": max(0, cache_latents - 1) / 2,
                "cache_tokens": cache_latents * 300,
                "cache_logical_bytes_one_cfg_branch": cache_bytes,
                "cache_logical_gib_one_cfg_branch": cache_bytes / 2**30,
                "first_layer_k_shape": list(cache[0][0].shape) if cache is not None else None,
                "first_layer_k_dtype": str(cache[0][0].dtype) if cache is not None else None,
                "first_layer_v_dtype": str(cache[0][1].dtype) if cache is not None else None,
                "current_latents": current_latents,
                "current_chunks": math.ceil(current_latents / 2),
                "role": "commit" if commit else "denoise",
                "return_kv": commit,
                "original_masked_sdpa_cuda": baseline,
                "chunk_flash_attention_cuda": fast_result,
                "original_masked_sdpa_wall": _summarize(walls[False]),
                "chunk_flash_attention_wall": _summarize(walls[True]),
                "median_speedup_cuda": baseline["median_ms"] / fast_result["median_ms"],
                "median_speedup_wall": statistics.median(walls[False]) / statistics.median(walls[True]),
                "peak_incremental_allocated_bytes_original": max(peaks[False]),
                "peak_incremental_allocated_bytes_chunk_flash": max(peaks[True]),
                "output_error": output_error,
                "returned_current_kv_error": cache_error,
            }
            report["cases"].append(case)
            (args.output_dir / "pretrained_forward_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"cache={cache_latents:2d} latent, current={current_latents:2d} latent, "
                f"{case['role']:7s}: {baseline['median_ms']:.3f} -> {fast_result['median_ms']:.3f} ms "
                f"({case['median_speedup_cuda']:.3f}x), RMSE={output_error['rmse']:.6g}, "
                f"relative RMSE={output_error['relative_rmse']:.6g}", flush=True,
            )

    csv_path = args.output_dir / "pretrained_forward_benchmark.csv"
    fields = [
        "cache_latents_including_sink", "history_chunks_excluding_sink", "current_latents", "role",
        "original_median_ms", "chunk_flash_median_ms", "speedup", "output_max_abs", "output_rmse",
        "output_relative_rmse", "cache_gib_one_cfg_branch",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in report["cases"]:
            writer.writerow({
                **{key: case[key] for key in fields[:4]},
                "original_median_ms": case["original_masked_sdpa_cuda"]["median_ms"],
                "chunk_flash_median_ms": case["chunk_flash_attention_cuda"]["median_ms"],
                "speedup": case["median_speedup_cuda"],
                "output_max_abs": case["output_error"]["max_abs"],
                "output_rmse": case["output_error"]["rmse"],
                "output_relative_rmse": case["output_error"]["relative_rmse"],
                "cache_gib_one_cfg_branch": case["cache_logical_gib_one_cfg_branch"],
            })
    print(f"Saved {csv_path} and JSON", flush=True)


if __name__ == "__main__":
    main()
