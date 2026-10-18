"""Pretrained A/B validation of framewise conditioning projections.

Both paths use Chunkwise FlashAttention, identical weights, and BF16 autocast.
Forward timings include the complete cached model call with shared history.
The optional rollout uses the episode identified by its preparation manifest,
keeping the original sampler settings and fixed noise. Its timer starts with prepared GPU
latents/actions and ends with decoded RGB: input loading/VAE encoding, model
loading, warmup, saving, and quality metrics are explicitly excluded.

Example::

    CUDA_VISIBLE_DEVICES=2 python profiling_config/benchmark_framewise_conditioning.py \
        --prepared-dir /path/to/episode_002352/prepared \
        --episode-id 002352 --rollout-latents 125 \
        --output-dir /path/to/episode_002352/results
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark_chunk_flash_attention import _error, _kv_error, _summarize  # noqa: E402
from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode  # noqa: E402
from miniworld.optimization_config import OptimizationConfig  # noqa: E402
from miniworld.vae.codec import StreamingVAEDecoder, load_wan22_vae  # noqa: E402


PREPARED_DIR = Path(
    "/data2/jian/output/Miniworld/DROID_long_horizon/chunk_flash_attention/episode_002352"
)
CANDIDATE = "framewise"
SWITCH_NAME = "framewise_conditioning"
COMPARISON_LABEL = "Framewise conditioning"
FIXED_OPTIMIZATIONS = {"chunkwise_flash_attention": True, "fused_norm_modulation": False}
MODES = {"original": False, CANDIDATE: True}


def configure_comparison(candidate, switch_name, label, fixed_optimizations):
    """Configure the shared protocol before parsing or constructing any models."""
    global CANDIDATE, SWITCH_NAME, COMPARISON_LABEL, FIXED_OPTIMIZATIONS, MODES
    if candidate == "original" or switch_name in fixed_optimizations:
        raise ValueError("The candidate and varied switch must be distinct from the fixed settings")
    CANDIDATE, SWITCH_NAME, COMPARISON_LABEL = candidate, switch_name, label
    FIXED_OPTIMIZATIONS = dict(fixed_optimizations)
    MODES = {"original": False, CANDIDATE: True}

# Validate the archived protocol instead of silently substituting sampler defaults.
SAMPLING_SETTINGS = {
    "history_len": 1, "stream_max_cache_chunks": 24, "stream_inflight_chunks": 8,
    "stream_sink_size": 1, "df_chunk_size": 2, "num_sampling_steps": 100,
    "df_ardiff_step": 5, "cfg_scale": 2.0, "cfg_interval_min": 0.2,
    "cfg_interval_max": 1.0, "timestep_baseshift": 2.667, "timestep_shift": -1.0,
}


def parse_args(defaults=None, description=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/data2/jian/checkpoints/miniworld/MiniWorld_1b_droid.pt"))
    parser.add_argument("--vae-checkpoint", type=Path,
                        default=Path("/data2/jian/checkpoints/miniworld/Wan2.2_VAE.pth"))
    parser.add_argument("--prepared-dir", type=Path, default=PREPARED_DIR)
    parser.add_argument("--episode-id", type=int,
                        help="Optional expected episode ID; a manifest mismatch is an error")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/data2/jian/output/Miniworld/framewise_conditioning"))
    parser.add_argument("--cache-latents", type=int, nargs="+", default=[0, 48])
    parser.add_argument("--current-latents", type=int, nargs="+", default=[2, 16])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--rollout-latents", type=int, default=0,
                        help="0 disables rollout; 125 covers 497 RGB frames with cache eviction")
    parser.add_argument("--skip-forward", action="store_true",
                        help="Skip standalone forward timing; both rollout paths still receive warmup")
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollout-order", nargs=2, choices=tuple(MODES),
                        default=list(MODES))
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    if min(args.cache_latents) < 0 or min(args.current_latents) <= 0:
        parser.error("cache lengths must be nonnegative and current lengths positive")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("use at least one warmup and two timed repetitions")
    if args.rollout_latents and args.rollout_latents < 17:
        parser.error("rollout length must be at least 17 latent frames, or 0 to disable")
    if args.skip_forward and not args.rollout_latents:
        parser.error("--skip-forward requires a rollout")
    if len(set(args.rollout_order)) != 2:
        parser.error("--rollout-order must include each implementation once")
    if args.output_dir.resolve().is_relative_to(REPO_ROOT.resolve()):
        parser.error("Write benchmark artifacts outside the source checkout")
    return args


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def validate_inputs(args, manifest, data):
    """Fail before measuring if episode identity, sampler settings, or inputs differ."""
    episode = int(manifest["episode"])
    if args.episode_id is not None and episode != args.episode_id:
        raise ValueError(f"Expected episode {args.episode_id:06d}, found {episode:06d}")
    for key, expected in SAMPLING_SETTINGS.items():
        actual = manifest["args"].get(key)
        if actual != expected:
            raise ValueError(f"Archived sampler setting {key}: expected {expected!r}, got {actual!r}")
    latent_count = int(manifest["latent_frames"])
    expected_shape = (1, 48, latent_count, 15, 20)
    for key in ("latents", "noise"):
        if tuple(data[key].shape) != expected_shape:
            raise ValueError(f"Invalid {key} shape: {tuple(data[key].shape)} != {expected_shape}")
    if tuple(data["cond_seq"].shape) != (1, latent_count, 28):
        raise ValueError("Expected framewise DROID actions with shape (1, latent_count, 28)")
    for key, value in data.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite prepared input: {key}")
    if torch.count_nonzero(data["latents"][:, :, 1:]).item():
        raise ValueError("Autoregressive input must contain only the initial GT latent; later latents must be zero")
    if "gt_latents" in data:
        if tuple(data["gt_latents"].shape) != expected_shape:
            raise ValueError("GT latent shape must match the rollout input")
        if not torch.equal(data["gt_latents"][:, :, :1], data["latents"][:, :, :1]):
            raise ValueError("Initial observation must match the separately stored GT latents")
    required = max(args.rollout_latents, 0 if args.skip_forward else max(args.cache_latents) + max(args.current_latents))
    if required > latent_count:
        raise ValueError(f"Prepared input has {latent_count} latent frames, but {required} are required")
    gt = np.load(args.prepared_dir / "gt_rgb.npy", mmap_mode="r")
    if tuple(gt.shape) != (4 * (latent_count - 1) + 1, 240, 320, 3) or gt.dtype != np.uint8:
        raise ValueError("GT RGB shape/dtype does not match the preparation manifest")


def load_models(args, manifest):
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = raw.get("ema_model", raw.get("model", raw))
    trained_latents = int(weights["net.feat_rope.freqs_cos"].shape[0] // 300)
    if max(args.cache_latents) + max(args.current_latents) > trained_latents:
        raise ValueError("Cached forward cases must stay inside the trained RoPE range")
    models = {}
    for name, enabled in MODES.items():
        options = dict(manifest["args"])
        options.update(
            latent_size=(15, 20), latent_channels=48, latent_frames=trained_latents,
            cond_dim=int(weights["net.null_action"].shape[-1]), wm_use_checkpoint=False,
            optimization_config=OptimizationConfig(
                **FIXED_OPTIMIZATIONS, **{SWITCH_NAME: enabled},
            ),
        )
        model = build_denoiser_from_mode(DenoiserConfig(**options))
        model.load_state_dict(weights, strict=True)
        models[name] = model.cuda().eval()
    return models, trained_latents


def build_shared_cache(model, data, count):
    """Use only the original path to construct identical shared forward history."""
    clean_latents = data.get("gt_latents", data["latents"])
    cache = None
    offset = 0
    while offset < count:
        length = min(2, count - offset)
        _, update = model.net.forward_with_cache(
            clean_latents[:, :, offset:offset + length],
            torch.zeros(1, length, device="cuda"),
            data["cond_seq"][:, offset:offset + length],
            past_kv_list=cache, current_position_offset=offset,
            return_kv=True, chunk_size=2,
        )
        cache = update if cache is None else [
            (torch.cat((old[0], new[0]), dim=-2), torch.cat((old[1], new[1]), dim=-2))
            for old, new in zip(cache, update)
        ]
        offset += length
    return cache


def projection_shapes(model, call):
    """Record actual projected row counts outside all measured iterations."""
    rows = []
    handles = []
    for name, module in model.net.named_modules():
        if isinstance(module, torch.nn.Linear) and (
            name.startswith("shared_mod.") or ".modulation." in name
            or name.startswith("final_layer.adaLN_modulation.")
        ):
            def record(_module, inputs, name=name):
                tensor = inputs[0]
                rows.append({"module": name, "input_shape": list(tensor.shape),
                             "projected_rows": tensor.numel() // tensor.shape[-1]})
            handles.append(module.register_forward_pre_hook(record))
    try:
        call()
    finally:
        for handle in handles:
            handle.remove()
    return rows


def benchmark_forward(args, models, data, report, diagnostics=None):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        shared = build_shared_cache(models["original"], data, max(args.cache_latents))
        for cache_length in args.cache_latents:
            cache = None if cache_length == 0 else [
                (k[:, :, :cache_length * 300], v[:, :, :cache_length * 300])
                for k, v in shared
            ]
            cache_bytes = 0 if cache is None else sum(
                tensor.numel() * tensor.element_size() for pair in cache for tensor in pair
            )
            for length in args.current_latents:
                commit = length == 2
                timestep = torch.zeros(1, length, device="cuda") if commit else torch.linspace(
                    0.05, 0.95, math.ceil(length / 2), device="cuda",
                ).repeat_interleave(2)[:length].unsqueeze(0)
                clean = data.get("gt_latents", data["latents"])[:, :, cache_length:cache_length + length]
                noise = data["noise"][:, :, cache_length:cache_length + length]
                t_view = timestep[:, None, :, None, None]
                current = (1 - t_view) * clean + t_view * noise
                condition = data["cond_seq"][:, cache_length:cache_length + length]

                def call(name):
                    return models[name].net.forward_with_cache(
                        current, timestep, condition, past_kv_list=cache,
                        current_position_offset=cache_length,
                        return_kv=commit, chunk_size=2,
                    )

                for _ in range(args.warmup):
                    for name in MODES:
                        call(name)
                baseline, baseline_kv = call("original")
                actual, actual_kv = call(CANDIDATE)
                error = _error(baseline, actual)
                kv_error = _kv_error(baseline_kv, actual_kv) if commit else None
                if not error["finite"]:
                    raise RuntimeError(f"{COMPARISON_LABEL} output contains non-finite values")
                del baseline, actual, baseline_kv, actual_kv
                shapes = {name: projection_shapes(models[name], lambda name=name: call(name))
                          for name in MODES}
                execution = {} if diagnostics is None else {
                    name: diagnostics(name, models[name], lambda name=name: call(name))
                    for name in MODES
                }
                cuda_times = {name: [] for name in MODES}
                wall_times = {name: [] for name in MODES}
                peak_bytes = {name: [] for name in MODES}
                for repeat in range(args.repeats):
                    # Alternating execution order reduces systematic clock/order bias.
                    order = list(MODES) if repeat % 2 == 0 else list(reversed(MODES))
                    for name in order:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        memory_before = torch.cuda.memory_allocated()
                        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        wall_start = time.perf_counter()
                        start.record()
                        result = call(name)
                        end.record()
                        torch.cuda.synchronize()
                        wall_times[name].append((time.perf_counter() - wall_start) * 1000)
                        cuda_times[name].append(start.elapsed_time(end))
                        peak_bytes[name].append(torch.cuda.max_memory_allocated() - memory_before)
                        del result
                case = {
                    "cache_latents_including_sink": cache_length,
                    "history_chunks_excluding_sink": max(0, cache_length - 1) / 2,
                    "cache_logical_bytes_one_cfg_branch": cache_bytes,
                    "current_latents": length, "role": "commit" if commit else "denoise",
                    "return_kv": commit, "output_error": error,
                    "returned_current_kv_error": kv_error,
                    "projection_shapes": shapes,
                    "execution_diagnostics": execution,
                    "cuda": {name: _summarize(values) for name, values in cuda_times.items()},
                    "wall": {name: _summarize(values) for name, values in wall_times.items()},
                    "peak_incremental_allocated_bytes": {name: max(values) for name, values in peak_bytes.items()},
                    "median_speedup_cuda": statistics.median(cuda_times["original"]) / statistics.median(cuda_times[CANDIDATE]),
                    "median_speedup_wall": statistics.median(wall_times["original"]) / statistics.median(wall_times[CANDIDATE]),
                }
                report["forward_cases"].append(case)
                dump(args.output_dir / "benchmark.json", report)
                print(f"cache={cache_length}, current={length}: "
                      f"{case['cuda']['original']['median_ms']:.3f} -> "
                      f"{case['cuda'][CANDIDATE]['median_ms']:.3f} ms "
                      f"({case['median_speedup_cuda']:.3f}x), relative RMSE={error['relative_rmse']:.6g}", flush=True)
    write_csv(args.output_dir / "forward_summary.csv", [{
        "history_chunks_excluding_sink": case["history_chunks_excluding_sink"],
        "current_latents": case["current_latents"], "role": case["role"],
        "original_median_ms": case["cuda"]["original"]["median_ms"],
        f"{CANDIDATE}_median_ms": case["cuda"][CANDIDATE]["median_ms"],
        "speedup": case["median_speedup_cuda"],
        "output_relative_rmse": case["output_error"]["relative_rmse"],
        "output_max_abs": case["output_error"]["max_abs"],
    } for case in report["forward_cases"]])


def generate(model, data, total_len, decoder):
    return model.generate_eval_latents_streaming(
        data["latents"], data["cond_seq"], total_len=total_len, history_len=1,
        max_cache_chunks=24, inflight_chunks=8, sink_frames=1,
        noise=data["noise"], stream_decoder=decoder, collect_stream_timing=False,
    )


class EvictionRecorder:
    """Record only CPU shape metadata; never synchronize the timed CUDA stream."""

    def __init__(self, model):
        self.model = model
        self.original = model._evict_and_shift_cache
        self.rows = []

    def __enter__(self):
        def record(cache, drop_frames, tokens_per_frame, rope_module, sink_frames=0):
            result = self.original(cache, drop_frames, tokens_per_frame, rope_module, sink_frames)
            before = cache[0][0].shape[-2] // tokens_per_frame
            after = result[0][0].shape[-2] // tokens_per_frame
            self.rows.append({"before_latents": before, "after_latents": after,
                              "drop_latents": drop_frames, "sink_latents": sink_frames})
            return result
        self.model._evict_and_shift_cache = record
        return self

    def __exit__(self, *_):
        # Restore class lookup, rather than leaving a benchmark function attached.
        del self.model._evict_and_shift_cache


def save_video(path, frames, fps):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def benchmark_rollout(args, models, data, report):
    vae = load_wan22_vae({"vae_checkpoint": args.vae_checkpoint})
    fixed_inputs = {key: data[key].clone() for key in ("latents", "noise", "cond_seq")}
    # Both implementations execute the complete same short sampler/VAE path before
    # timing. Standalone forward warmup does not exercise CFG, scheduling, or decode.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for name in args.rollout_order:
            print(f"Warming {name} sampler on 17 latent frames", flush=True)
            warm = generate(models[name], data, 17, StreamingVAEDecoder(vae))
            del warm
    torch.cuda.synchronize()
    report["rollout"] = {
        "latent_frames": args.rollout_latents,
        "rgb_frames": 4 * (args.rollout_latents - 1) + 1,
        "mode": "normal generated-history autoregression; GT initial latent only",
        "order": args.rollout_order,
        "warmup": "one 17-latent complete generation + streaming decode per implementation",
        "timing_scope": "prepared GPU latents/actions to RGB; generation + streaming VAE decode",
        "timing_excludes": ["input loading", "VAE encoding", "model loading", "warmup", "file saving", "metrics"],
        "sampling": {"history_len": 1, "max_cache_chunks": 24, "inflight_chunks": 8,
                     "sink_frames": 1, "chunk_size": 2, "num_sampling_steps": 100,
                     "df_ardiff_step": 5, "cfg_scale": 2.0},
        "limitation": "one seed and one timed run per path for this episode; wall timing is subject to run-to-run variation",
        "diagnostic_overhead": "CPU-only cache eviction shape metadata; no diagnostic CUDA synchronization",
        "runs": {},
    }
    for name in args.rollout_order:
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with EvictionRecorder(models[name]) as cache_recorder:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latents, rgb = generate(models[name], data, args.rollout_latents, StreamingVAEDecoder(vae))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated()
        metadata = dict(models[name].last_eval_meta)
        if metadata["effective_steps"] != 40:
            raise AssertionError("Rollout must retain the original 40 effective steps")
        checks = {
            "latents_finite": bool(torch.isfinite(latents).all()),
            "rgb_finite": bool(torch.isfinite(rgb).all()),
            "latent_shape_correct": tuple(latents.shape) == (1, 48, args.rollout_latents, 15, 20),
            "rgb_shape_correct": tuple(rgb.shape) == (1, 3, report["rollout"]["rgb_frames"], 240, 320),
            "initial_latent_preserved": bool(torch.equal(latents[:, :, :1], data["latents"][:, :, :1])),
            "shared_inputs_unmodified": all(torch.equal(data[key], original) for key, original in fixed_inputs.items()),
            "chunkwise_flash_attention_enabled": metadata["optimization_config"]["requested"]["chunkwise_flash_attention"] is True,
            f"{CANDIDATE}_switch_correct": metadata["optimization_config"]["requested"][SWITCH_NAME] is MODES[name],
            **{f"fixed_{key}_correct": metadata["optimization_config"]["requested"][key] is value
               for key, value in FIXED_OPTIMIZATIONS.items()},
            "sink_argument_one_in_every_eviction": all(row["sink_latents"] == 1 for row in cache_recorder.rows),
            "cache_capacity_after_every_eviction": all(row["after_latents"] == 48 for row in cache_recorder.rows),
            "eviction_length_consistent": all(row["before_latents"] - row["after_latents"] == row["drop_latents"] for row in cache_recorder.rows),
            "eviction_exercised_when_rollout_exceeds_cache": args.rollout_latents <= 49 or bool(cache_recorder.rows),
        }
        if not all(checks.values()):
            raise AssertionError(f"Rollout validation failed for {name}: {checks}")
        pixels = ((rgb[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()
        np.save(args.output_dir / f"{name}_rgb.npy", pixels)
        torch.save(latents.cpu(), args.output_dir / f"{name}_latents.pt")
        report["rollout"]["runs"][name] = {
            "generation_stream_decode_seconds": elapsed,
            "peak_allocated_bytes": peak_allocated,
            "sampler_meta": metadata,
            "checks": checks,
            "cache_eviction_calls_both_cfg_branches": len(cache_recorder.rows),
            "cache_capacity_latents_including_sink": 48,
            "history_capacity_chunks_excluding_sink": 23.5,
            "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        }
        write_csv(args.output_dir / f"{name}_cache_evictions.csv", cache_recorder.rows)
        dump(args.output_dir / "benchmark.json", report)
        print(f"{name} rollout: {pixels.shape[0]} RGB frames in {elapsed:.3f} s", flush=True)
        if args.save_videos:
            save_video(args.output_dir / f"{name}.mp4", pixels, report["save_fps"])
        del latents, rgb, pixels
    runs = report["rollout"]["runs"]
    report["rollout"]["speedup"] = runs["original"]["generation_stream_decode_seconds"] / runs[CANDIDATE]["generation_stream_decode_seconds"]
    if args.save_videos:
        count = report["rollout"]["rgb_frames"]
        arrays = [np.load(args.prepared_dir / "gt_rgb.npy", mmap_mode="r")[:count]] + [
            np.load(args.output_dir / f"{name}_rgb.npy", mmap_mode="r") for name in MODES
        ]
        # Concatenate one frame at a time instead of allocating a full triptych.
        import av
        with av.open(str(args.output_dir / f"gt_original_{CANDIDATE}.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=report["save_fps"])
            stream.width, stream.height, stream.pix_fmt = 960, 240, "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}
            for frame in range(count):
                pixels = np.concatenate([array[frame] for array in arrays], axis=1)
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


def quality(args, report):
    import lpips
    from skimage.metrics import structural_similarity

    count = report["rollout"]["rgb_frames"]
    gt = np.load(args.prepared_dir / "gt_rgb.npy", mmap_mode="r")[:count]
    metric = lpips.LPIPS(net="alex", version="0.1").cuda().eval()
    metric.requires_grad_(False)
    rows = []
    for name in MODES:
        pixels = np.load(args.output_dir / f"{name}_rgb.npy", mmap_mode="r")
        # Metric definitions match the earlier episode 002352 comparisons.
        for start in range(0, count, 8):
            pred = torch.from_numpy(np.array(pixels[start:start + 8])).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
            target = torch.from_numpy(np.array(gt[start:start + 8])).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
            perceptual = metric(pred, target).flatten().cpu().tolist()
            for offset, distance in enumerate(perceptual):
                frame = start + offset
                mse = np.mean((pixels[frame].astype(np.float64) - gt[frame]) ** 2)
                rows.append({
                    "implementation": name, "frame_index": frame,
                    "PSNR": 10 * math.log10(255 ** 2 / mse) if mse > 0 else float("inf"),
                    "SSIM": float(structural_similarity(
                        gt[frame], pixels[frame], data_range=255, channel_axis=-1,
                        gaussian_weights=True, sigma=1.5, win_size=11, use_sample_covariance=False,
                    )),
                    "LPIPS": float(distance),
                })
    summaries = [{"implementation": name, "first_rgb_frame": 1, "last_rgb_frame": count - 1,
                  **{key: statistics.mean(row[key] for row in rows if row["implementation"] == name and row["frame_index"] > 0)
                     for key in ("PSNR", "SSIM", "LPIPS")}} for name in MODES]
    write_csv(args.output_dir / "quality_per_frame.csv", rows)
    write_csv(args.output_dir / "quality_summary.csv", summaries)
    segments = []
    for name in MODES:
        for index, segment in enumerate(np.array_split(np.arange(1, count), 4)):
            first, last = int(segment[0]), int(segment[-1])
            selected = [row for row in rows if row["implementation"] == name and first <= row["frame_index"] <= last]
            segments.append({"implementation": name, "quarter": index + 1, "first_rgb_frame": first,
                             "last_rgb_frame": last,
                             **{key: statistics.mean(row[key] for row in selected) for key in ("PSNR", "SSIM", "LPIPS")}})
    write_csv(args.output_dir / "quality_by_quarter.csv", segments)
    latent = {name: torch.load(args.output_dir / f"{name}_latents.pt", map_location="cpu", weights_only=True) for name in MODES}
    report["quality"] = {
        "summary": summaries,
        "quarters": segments,
        f"delta_{CANDIDATE}_minus_original": {key: summaries[1][key] - summaries[0][key] for key in ("PSNR", "SSIM", "LPIPS")},
        "latent_error": _error(latent["original"], latent[CANDIDATE]),
        "averaging": "arithmetic per-frame means excluding conditioning RGB frame 0",
        "definitions": "PSNR RGB range255; SSIM gaussian sigma1.5 win11 population covariance RGB mean; LPIPS AlexNet v0.1 FP32",
    }
    print("QUALITY", json.dumps(report["quality"]), flush=True)


@torch.inference_mode()
def main(args=None, report_metadata=None, forward_diagnostics=None):
    args = args or parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "benchmark.json").exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir} already has benchmark.json")
    torch.set_num_threads(4)
    manifest = json.loads((args.prepared_dir / "experiment_manifest.json").read_text())
    torch.manual_seed(manifest["seed"])
    torch.cuda.manual_seed_all(manifest["seed"])
    data = torch.load(args.prepared_dir / "prepared_inputs.pt", map_location="cpu", weights_only=True)
    validate_inputs(args, manifest, data)
    data = {name: value.cuda() for name, value in data.items()}
    models, trained_latents = load_models(args, manifest)
    report = {
        "episode": f"{int(manifest['episode']):06d}", "seed": manifest["seed"],
        "task": manifest.get("task"), "save_fps": int(manifest["args"].get("save_fps", 8)),
        "prepared_manifest": manifest,
        "prepared_dir": str(args.prepared_dir.resolve()),
        "comparison": {"candidate": CANDIDATE, "switch": SWITCH_NAME,
                       "label": COMPARISON_LABEL, "fixed_optimizations": FIXED_OPTIMIZATIONS},
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_load_strict": True,
        "trained_latents": trained_latents, "tokens_per_latent": 300,
        "latent_frames_per_chunk": 2, "sink_latents": 1,
        "window_definition": "historical window only, excludes sink and currently denoised chunks",
        "prepared_input_sha256": hash_file(args.prepared_dir / "prepared_inputs.pt"),
        "source_video": manifest["source"], "source_actions": manifest["actions"],
        "optimization_configs": {name: model.net.optimization_config.snapshot() for name, model in models.items()},
        "weights_dtype": str(next(models["original"].parameters()).dtype),
        "compute_autocast_dtype": "torch.bfloat16", "gpu": torch.cuda.get_device_name(),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "gpu_compute_capability": list(torch.cuda.get_device_capability()),
        "host": socket.gethostname(), "cpu_threads": torch.get_num_threads(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "source_sha256": {str(path.relative_to(REPO_ROOT)): hash_file(path) for path in (
            REPO_ROOT / "miniworld/miniworld.py", REPO_ROOT / "miniworld/optimization_config.py", Path(__file__).resolve(),
        )},
        "warmup_per_case": args.warmup, "timed_repeats_per_case": args.repeats,
        "forward_scope": "complete cached forward incl. projections, attention, RoPE, KV concatenation, final layer; common baseline-built cache",
        "forward_history_inputs": "encoded GT latents" if "gt_latents" in data else "legacy prepared seed plus zero-filled later latents (not GT history)",
        "forward_excludes": "cache construction/eviction, sampler, CFG second pass, VAE, I/O",
        "forward_cases": [],
    }
    if report_metadata is not None:
        report.update(report_metadata(args, models, data))
    dump(args.output_dir / "benchmark.json", report)
    if not args.skip_forward:
        benchmark_forward(args, models, data, report, diagnostics=forward_diagnostics)
    if args.rollout_latents:
        benchmark_rollout(args, models, data, report)
        del models, data
        gc.collect()
        torch.cuda.empty_cache()
        quality(args, report)
    report["complete"] = True
    dump(args.output_dir / "benchmark.json", report)
    print(f"Saved {args.output_dir / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
