"""Bounded paired real-trajectory smoke validation for opt-in HeadCast.

Reuses episode 002352's fixed noise, initial observation, and action conditions.
Runs the existing chunk-Flash baseline and HeadCast on the same loaded model.
Timers cover complete latent generation including calibration, but exclude
model loading, VAE, and file output. One pair is not a throughput benchmark.
"""

from __future__ import annotations

import time

PROCESS_START = time.monotonic()

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_INPUT = Path(
    "/data2/jian/output/Miniworld/DROID_long_horizon/"
    "chunk_flash_attention/episode_002352"
)
DEFAULT_OUTPUT = Path(
    "/home/jian/Documents/Codex/2026-10-19/hello/work/headcast_migration"
)


class BudgetExpired(Exception):
    pass


def json_safe(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def error_stats(reference, actual):
    reference, actual = reference.float(), actual.float()
    delta = actual - reference
    rms = reference.square().mean().sqrt().item()
    rmse = delta.square().mean().sqrt().item()
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "rmse": rmse,
        "reference_rms": rms,
        "relative_rmse": rmse / max(rms, 1e-12),
        "bitwise_equal": bool(torch.equal(reference, actual)),
    }


def group_audit(summary, depth, heads):
    groups = summary.get("groups", [])
    valid = len(groups) == depth
    counts = {mode: 0 for mode in ("global", "temporal", "spatial")}
    for layer in groups:
        flattened = []
        for mode in counts:
            indices = list(layer.get(mode, []))
            flattened.extend(indices)
            counts[mode] += len(indices)
        valid = valid and sorted(flattened) == list(range(heads))
    return {"complete_disjoint_head_partition": valid, "computed_group_counts": counts}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-latents", type=int, default=72)
    parser.add_argument("--budget-seconds", type=float, default=300)
    parser.add_argument("--tile-height", type=int, default=5)
    parser.add_argument("--tile-width", type=int, default=5)
    parser.add_argument("--recent-chunks", type=int, default=2)
    parser.add_argument("--max-relative-rmse", type=float, default=0.1)
    parser.add_argument("--min-cosine", type=float, default=0.95)
    parser.add_argument("--query-samples-per-chunk", type=int, default=32)
    args = parser.parse_args()
    if not 66 <= args.total_latents <= 125:
        parser.error("use 66..125 latents to reach a full 48-frame cache and sparse steps")
    if not 0 < args.budget_seconds <= 300:
        parser.error("budget must be positive and at most 300 seconds")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "validation.json"
    if report_path.exists():
        raise FileExistsError(f"Choose a fresh output directory; {report_path} exists")
    deadline = PROCESS_START + args.budget_seconds

    def expired(signum, frame):
        raise BudgetExpired("300-second validation budget exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, deadline - time.monotonic()))
    report = {
        "status": "running", "scope": "one paired real generated-history latent rollout; no VAE",
        "arguments": vars(args), "runs": {},
        "timing_includes": "complete latent sampler, warm-up trajectory, cache maintenance, and HeadCast classification",
        "timing_excludes": "model/input loading, VAE, diagnostic parity replays, saving outputs, and metrics",
        "memory_scope": "includes one retained full conditional cache for parity replay; not production peak memory",
        "limits": "one episode and seed; one baseline/on pair; latent error is not decoded-video quality",
    }
    original_forward = None
    model = None
    try:
        from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode

        manifest = json.loads((args.input_dir / "experiment_manifest.json").read_text())
        torch.set_num_threads(4)
        torch.manual_seed(manifest["seed"])
        raw = torch.load(manifest["args"]["checkpoint"], map_location="cpu", weights_only=True)
        weights = raw.get("ema_model", raw.get("model", raw))
        options = dict(manifest["args"])
        options.update(
            latent_size=(15, 20),
            latent_frames=int(weights["net.feat_rope.freqs_cos"].shape[0] // 300),
            wm_use_checkpoint=False, headcast=False,
            headcast_tile_height=args.tile_height, headcast_tile_width=args.tile_width,
            headcast_recent_chunks=args.recent_chunks,
            headcast_max_relative_rmse=args.max_relative_rmse,
            headcast_min_cosine=args.min_cosine,
            headcast_query_samples_per_chunk=args.query_samples_per_chunk,
        )
        config = DenoiserConfig(**options)
        if not hasattr(config, "headcast"):
            raise RuntimeError("HeadCast config interface is absent; refusing a vacuous baseline/on comparison")
        model = build_denoiser_from_mode(config)
        model.load_state_dict(weights, strict=True)
        del weights, raw
        model = model.cuda().eval()
        inputs = torch.load(args.input_dir / "prepared_inputs.pt", map_location="cpu", weights_only=True)
        inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
        torch.cuda.synchronize()
        report.update(
            episode=manifest["episode"], seed=manifest["seed"], checkpoint=options["checkpoint"],
            gpu=torch.cuda.get_device_name(), visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch_version=torch.__version__, load_seconds=time.monotonic() - PROCESS_START,
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            source_sha256={name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in (
                "miniworld/miniworld.py", "miniworld/denoiser.py", "miniworld/headcast.py",
                "miniworld/headcast_attention.py",
            )},
            cfg_scale=model.cfg_scale,
            sampler=dict(total_len=args.total_latents, history_len=1, max_cache_chunks=24,
                         inflight_chunks=8, sink_frames=1, stream_decoder=None,
                         collect_stream_timing=False),
        )
        write_report(report_path, report)
        original_forward = model.net.forward_with_cache
        probe = {}

        def capture_one_full_cache(*positional, **keywords):
            # Existing concatenated cache tensors are immutable across forwards;
            # retaining their references avoids a multi-GB copy inside the timer.
            past = keywords.get("past_kv_list")
            if (not probe and not keywords.get("return_kv", False) and
                    keywords.get("cond_drop") is None and past and past[0] is not None and
                    past[0][0].shape[-2] == 48 * 300):
                probe["args"] = tuple(value.clone() for value in positional)
                probe["kwargs"] = dict(keywords)
                probe["kwargs"]["past_kv_list"] = list(past)
            return original_forward(*positional, **keywords)

        model.net.forward_with_cache = capture_one_full_cache
        outputs = {}
        off_reference = None
        for name, enabled in (("baseline", False), ("headcast", True)):
            model.cfg.headcast = enabled
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.generate_eval_latents_streaming(
                    inputs["latents"], inputs["cond_seq"],
                    noise=inputs["noise"], **report["sampler"],
                )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            outputs[name] = result.detach().cpu()
            summary = json_safe(model.last_eval_meta)
            report["runs"][name] = {
                "latent_generation_seconds": elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "latent_shape": list(result.shape),
                "finite": bool(torch.isfinite(result).all()),
                "sampler_meta": summary,
            }
            torch.save(outputs[name], args.output_dir / f"{name}_latents.pt")
            del result
            head_counts = summary.get("headcast", {}).get("group_counts")
            print(f"{name}: {elapsed:.3f} s, head_counts={head_counts}", flush=True)
            write_report(report_path, report)
            if not report["runs"][name]["finite"]:
                raise AssertionError(f"Non-finite latent output from {name}")
            if name == "baseline":
                model.net.forward_with_cache = original_forward
                if not probe:
                    raise AssertionError("Baseline did not reach a full cache")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    off_reference = original_forward(*probe["args"], **probe["kwargs"])[0].detach().cpu()

        model.cfg.headcast = False
        with torch.autocast("cuda", dtype=torch.bfloat16):
            off_after = original_forward(*probe["args"], **probe["kwargs"])[0].detach().cpu()
        report["disabled_after_enabled_same_input_parity"] = error_stats(off_reference, off_after)
        report["latent_error"] = error_stats(outputs["baseline"], outputs["headcast"])
        report["last_24_latent_error"] = error_stats(outputs["baseline"][:, :, -24:], outputs["headcast"][:, :, -24:])
        report["first_48_latent_error"] = error_stats(outputs["baseline"][:, :, :48], outputs["headcast"][:, :, :48])
        report["seed_frame_equal"] = bool(torch.equal(outputs["baseline"][:, :, :1], outputs["headcast"][:, :, :1]))
        report["one_pair_time_ratio_baseline_over_headcast"] = (
            report["runs"]["baseline"]["latent_generation_seconds"] /
            report["runs"]["headcast"]["latent_generation_seconds"]
        )
        headcast = report["runs"]["headcast"]["sampler_meta"].get("headcast", {})
        report["group_audit"] = group_audit(headcast, model.net.depth, model.net.num_heads)
        checks = {
            "finite_outputs": all(run["finite"] for run in report["runs"].values()),
            "headcast_active": headcast.get("status") == "active",
            "sparse_paths_used": int(headcast.get("routed_attention_calls", 0)) > 0,
            "head_partition_valid": report["group_audit"]["complete_disjoint_head_partition"],
            "disabled_parity_bitwise": report["disabled_after_enabled_same_input_parity"]["bitwise_equal"],
            "warmup_prefix_bitwise": report["first_48_latent_error"]["bitwise_equal"],
            "seed_frame_preserved": report["seed_frame_equal"],
        }
        report["checks"] = checks
        rows = []
        for frame in range(args.total_latents):
            rows.append({"latent_frame": frame, **error_stats(
                outputs["baseline"][:, :, frame], outputs["headcast"][:, :, frame],
            )})
        with (args.output_dir / "latent_error_by_frame.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report["status"] = "complete" if all(checks.values()) else "failed_checks"
    except BudgetExpired as error:
        report["status"] = "budget_exhausted"
        report["error"] = str(error)
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if model is not None:
            model.cfg.headcast = False
            if original_forward is not None:
                model.net.forward_with_cache = original_forward
        report["elapsed_seconds_including_load_and_validation"] = time.monotonic() - PROCESS_START
        write_report(report_path, report)
        print(f"Validation {report['status']}: {report_path}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
