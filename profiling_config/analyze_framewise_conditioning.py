"""Summarize paired framewise-conditioning rollouts without using a GPU.

Example::

    python profiling_config/analyze_framewise_conditioning.py \
        --input-root /data2/jian/output/Miniworld/framewise_conditioning_four

Each episode must contain benchmark.json and the quality CSV files written by
benchmark_framewise_conditioning.py. Frame zero is an observation and is excluded
from every reported quality average. No quality acceptance threshold is assumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CANDIDATE = "framewise"
SWITCH_NAME = "framewise_conditioning"
COMPARISON_LABEL = "Framewise conditioning"
SHORT_LABEL = "Framewise"
FIXED_OPTIMIZATIONS = {"chunkwise_flash_attention": True}
MODES = ("original", CANDIDATE)
METRICS = ("PSNR", "SSIM", "LPIPS")
LABELS = {"PSNR": "PSNR (dB, higher is better)",
          "SSIM": "SSIM (higher is better)",
          "LPIPS": "LPIPS (lower is better)"}
COLORS = {"original": "#3668a8", "framewise": "#df761d"}


def configure_comparison(candidate, switch_name, label, short_label, fixed_optimizations):
    """Reuse the archived analysis with an explicitly named comparison."""
    global CANDIDATE, SWITCH_NAME, COMPARISON_LABEL, SHORT_LABEL, FIXED_OPTIMIZATIONS, MODES, COLORS
    CANDIDATE, SWITCH_NAME = candidate, switch_name
    COMPARISON_LABEL, SHORT_LABEL = label, short_label
    FIXED_OPTIMIZATIONS = dict(fixed_optimizations)
    MODES = ("original", CANDIDATE)
    COLORS = {"original": "#3668a8", CANDIDATE: "#df761d"}


def delta_key(field):
    return f"delta_{field}_{CANDIDATE}_minus_original"


TIMING_SCOPE = "Generation + streaming VAE decode; prepared GPU inputs to decoded RGB"
TIMING_EXCLUDES = "Input/model loading, VAE encoding, warmup, file saving, and quality metrics"
SAMPLING = {"history_len": 1, "max_cache_chunks": 24, "inflight_chunks": 8,
            "sink_frames": 1, "chunk_size": 2, "num_sampling_steps": 100,
            "df_ardiff_step": 5, "cfg_scale": 2.0}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values):
    """Do not silently drop invalid values and bias an aggregate toward success."""
    values = list(values)
    if not values or any(value is None or not math.isfinite(value) for value in values):
        return None
    return statistics.mean(values)


def delta(after, before):
    return None if after is None or before is None else after - before


def quality_means(frames, indices):
    return {metric: mean(frames[index][metric] for index in indices) for metric in METRICS}


def analyze_episode(path):
    report = json.loads(path.read_text())
    if report.get("complete") is not True:
        raise ValueError(f"Incomplete benchmark: {path}")
    episode = f"{int(report['episode']):06d}"
    rollout = report["rollout"]
    count = int(rollout["rgb_frames"])
    if count < 5:
        raise ValueError(f"Episode {episode} needs at least four predicted frames")
    issues = []
    frames = {mode: {} for mode in MODES}
    for row in read_csv(path.parent / "quality_per_frame.csv"):
        mode, index = row["implementation"], int(row["frame_index"])
        if mode not in frames or index in frames[mode]:
            raise ValueError(f"Unknown implementation or duplicate frame in {path.parent}: {row}")
        frames[mode][index] = {key: float(row[key]) for key in METRICS}
    for mode in MODES:
        if set(frames[mode]) != set(range(count)):
            raise ValueError(f"Episode {episode}, {mode}: missing or unexpected quality frames")
        for index in range(1, count):
            for metric, value in frames[mode][index].items():
                if not math.isfinite(value):
                    issues.append(f"{mode}, frame {index}: non-finite {metric}={value}")

    predicted = np.arange(1, count)
    quarters = np.array_split(predicted, 4)
    result = {
        "episode": episode, "seed": report["seed"], "task": report.get("task"),
        "benchmark_json": str(path.resolve()), "git_head": report.get("git_head"),
        "host": report.get("host"), "gpu": report.get("gpu"),
        "predicted_rgb_frames": count - 1, "first_evaluated_rgb_frame": 1,
        "last_evaluated_rgb_frame": count - 1,
        "first_quarter_rgb_frames": [int(quarters[0][0]), int(quarters[0][-1])],
        "last_quarter_rgb_frames": [int(quarters[-1][0]), int(quarters[-1][-1])],
        "sampling": rollout.get("sampling"), "rollout_order": rollout.get("order"),
        "implementations": {}, "issues": issues,
    }
    if rollout.get("sampling") != SAMPLING:
        issues.append(f"Sampler settings differ from the archived protocol: {rollout.get('sampling')}")
    for mode, enabled in zip(MODES, (False, True)):
        run = rollout["runs"][mode]
        checks = run.get("checks", {})
        if not checks:
            issues.append(f"{mode}: benchmark validation checks are missing")
        issues.extend(f"{mode}: validation check failed: {key}={value!r}"
                      for key, value in checks.items() if value is not True)
        meta = run.get("sampler_meta", {})
        requested = meta.get("optimization_config", {}).get("requested", {})
        if requested.get("chunkwise_flash_attention") is not True:
            issues.append(f"{mode}: Chunkwise FlashAttention was not enabled in the model snapshot")
        if requested.get(SWITCH_NAME) is not enabled:
            issues.append(f"{mode}: unexpected {SWITCH_NAME} value in model snapshot")
        for key, expected in FIXED_OPTIMIZATIONS.items():
            if requested.get(key) is not expected:
                issues.append(f"{mode}: unexpected fixed {key} value in model snapshot")
        if meta.get("effective_steps") != 40:
            issues.append(f"{mode}: expected 40 effective sampler steps, got {meta.get('effective_steps')}")
        elapsed = float(run["generation_stream_decode_seconds"])
        if not math.isfinite(elapsed) or elapsed <= 0:
            issues.append(f"{mode}: invalid measured duration {elapsed}")
            elapsed = None
        result["implementations"][mode] = {
            "mean_quality": quality_means(frames[mode], predicted),
            "first_quarter_quality": quality_means(frames[mode], quarters[0]),
            "last_quarter_quality": quality_means(frames[mode], quarters[-1]),
            "generation_stream_decode_seconds": elapsed,
            "peak_allocated_GiB": run["peak_allocated_bytes"] / 2 ** 30,
            "checks": checks, "requested_optimization_config": requested,
            "cache_eviction_calls_both_cfg_branches": run.get("cache_eviction_calls_both_cfg_branches"),
            "history_capacity_chunks_excluding_sink": run.get("history_capacity_chunks_excluding_sink"),
        }
    # Cross-check the published CSV instead of trusting separately computed means.
    saved = {row["implementation"]: row for row in read_csv(path.parent / "quality_summary.csv")}
    if set(saved) != set(MODES):
        raise ValueError(f"Unexpected quality summary implementations: {path.parent}")
    for mode in MODES:
        if int(saved[mode]["first_rgb_frame"]) != 1 or int(saved[mode]["last_rgb_frame"]) != count - 1:
            issues.append(f"{mode}: quality_summary.csv uses a different frame range")
        for metric in METRICS:
            recomputed = result["implementations"][mode]["mean_quality"][metric]
            if recomputed is not None and not math.isclose(recomputed, float(saved[mode][metric]), rel_tol=1e-9, abs_tol=1e-9):
                issues.append(f"{mode}: recomputed {metric} differs from quality_summary.csv")
    old, new = (result["implementations"][mode] for mode in MODES)
    old_config, new_config = (dict(item["requested_optimization_config"]) for item in (old, new))
    old_config.pop(SWITCH_NAME, None)
    new_config.pop(SWITCH_NAME, None)
    if old_config != new_config:
        issues.append(f"Optimization configurations differ beyond the {SWITCH_NAME} switch")
    if report.get("quality", {}).get("latent_error", {}).get("finite") is not True:
        issues.append("Missing or failed finite check for the paired generated latent outputs")
    for field in ("mean_quality", "first_quarter_quality", "last_quarter_quality"):
        result[delta_key(field)] = {
            metric: delta(new[field][metric], old[field][metric]) for metric in METRICS
        }
    result["last_minus_first_quarter_change_in_delta"] = {
        metric: delta(result[delta_key("last_quarter_quality")][metric],
                      result[delta_key("first_quarter_quality")][metric])
        for metric in METRICS
    }
    a, b = (item["generation_stream_decode_seconds"] for item in (old, new))
    result["generation_stream_decode_speedup"] = a / b if a and b else None
    result["recorded_checks_and_metrics_finite"] = not issues
    return result, frames


def aggregate(episodes, expected):
    issues = [f"Episode {item['episode']}: {issue}" for item in episodes for issue in item["issues"]]
    if len(episodes) != expected:
        issues.append(f"Expected {expected} episodes, found {len(episodes)}")
    if len({item["episode"] for item in episodes}) != len(episodes):
        issues.append("Duplicate episode IDs; the macro average does not represent distinct episodes")
    for key in ("git_head", "gpu"):
        if len({item[key] for item in episodes}) != 1:
            issues.append(f"Episodes were measured with different {key} values")
    macro = {}
    for mode in MODES:
        macro[mode] = {
            field: {metric: mean(item["implementations"][mode][field][metric] for item in episodes)
                    for metric in METRICS}
            for field in ("mean_quality", "first_quarter_quality", "last_quarter_quality")
        }
        macro[mode]["mean_generation_stream_decode_seconds"] = mean(
            item["implementations"][mode]["generation_stream_decode_seconds"] for item in episodes)
    a, b = (macro[mode]["mean_generation_stream_decode_seconds"] for mode in MODES)
    return {
        "episode_count": len(episodes), "expected_episode_count": expected,
        "episode_ids": [item["episode"] for item in episodes],
        "timing_scope": TIMING_SCOPE, "timing_excludes": TIMING_EXCLUDES,
        "quality_averaging": "Arithmetic per-frame average excludes observation frame 0; macro average weights each episode equally",
        "delta_definition": f"{SHORT_LABEL} ON minus OFF; positive PSNR/SSIM and negative LPIPS indicate improvement",
        "tail_definition": "Fourth np.array_split quarter of predicted RGB frame indices; compare paired ON-minus-OFF deltas across quarters",
        "history_window_definition": "Historical chunks only; excludes the sink and chunks currently being denoised",
        "history_capacity_chunks_excluding_sink": 23.5,
        "sink_latents": 1, "currently_denoised_chunks": 8, "latent_frames_per_chunk": 2,
        "limitations": ["One seed and one timed rollout per implementation per episode",
                        "Runtime includes CPU-only cache eviction diagnostics and is subject to timing variation",
                        "Requested optimization flags come from model snapshots; no per-kernel backend trace is collected",
                        "No quality acceptance threshold was specified; raw deltas do not establish equivalence",
                        "Late-minus-early quality deltas are descriptive and do not alone identify error accumulation"],
        "macro_average": macro,
        f"macro_quality_delta_{CANDIDATE}_minus_original": {
            metric: delta(macro[CANDIDATE]["mean_quality"][metric], macro["original"]["mean_quality"][metric])
            for metric in METRICS},
        "ratio_of_total_generation_stream_decode_times": a / b if a and b else None,
        "mean_per_episode_speedup": mean(item["generation_stream_decode_speedup"] for item in episodes),
        "recorded_checks_and_metrics_finite": not issues,
        "issues": issues, "episodes": episodes,
    }


def format_number(value, digits=6):
    return "invalid" if value is None else f"{value:.{digits}f}"


def write_outputs(destination, summary, frames):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    rows = []
    for item in summary["episodes"]:
        row = {key: item[key] for key in ("episode", "seed", "predicted_rgb_frames", "generation_stream_decode_speedup",
                                        "recorded_checks_and_metrics_finite")}
        for mode in MODES:
            run = item["implementations"][mode]
            row[f"{mode}_generation_stream_decode_seconds"] = run["generation_stream_decode_seconds"]
            row[f"{mode}_cache_eviction_calls_both_cfg_branches"] = run["cache_eviction_calls_both_cfg_branches"]
            for segment in ("mean_quality", "first_quarter_quality", "last_quarter_quality"):
                for metric in METRICS:
                    row[f"{mode}_{segment}_{metric}"] = run[segment][metric]
        for segment in ("mean_quality", "first_quarter_quality", "last_quarter_quality"):
            for metric in METRICS:
                row[f"delta_{segment}_{metric}"] = item[delta_key(segment)][metric]
        for metric in METRICS:
            row[f"last_minus_first_quarter_change_in_delta_{metric}"] = item["last_minus_first_quarter_change_in_delta"][metric]
        rows.append(row)
    with (destination / "per_episode_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"{COMPARISON_LABEL}: paired OFF / ON validation", "",
             f"Episodes: {', '.join(summary['episode_ids'])}",
             "Chunkwise FlashAttention enabled in both model configurations.",
             "Quality: arithmetic means over predicted RGB frames, excluding observation frame 0.",
             "History capacity: 23.5 chunks, excluding 1 latent sink and 8 denoising chunks; 2 latent frames/chunk.",
             f"Timing: {TIMING_SCOPE}.", f"Timing excludes: {TIMING_EXCLUDES}.", "",
             "Episode  OFF seconds  ON seconds  Speedup  Delta PSNR (dB)  Delta SSIM  Delta LPIPS"]
    for item in summary["episodes"]:
        a, b = (item["implementations"][mode]["generation_stream_decode_seconds"] for mode in MODES)
        differences = item[delta_key("mean_quality")]
        lines.append(f"{item['episode']}  {format_number(a, 3)}  {format_number(b, 3)}  "
                     f"{format_number(item['generation_stream_decode_speedup'], 3)}x  "
                     + "  ".join(format_number(differences[metric]) for metric in METRICS))
    lines.extend(["", "Equal-episode macro averages (OFF -> ON):"])
    for metric in METRICS:
        a, b = (summary["macro_average"][mode]["mean_quality"][metric] for mode in MODES)
        lines.append(f"{metric}: {format_number(a)} -> {format_number(b)}; delta {format_number(delta(b, a))}")
    lines.extend([f"Ratio of total generation + decode times: {format_number(summary['ratio_of_total_generation_stream_decode_times'], 3)}x",
                  "", "Last-quarter deltas (ON minus OFF), followed by last-minus-first-quarter change in delta:"])
    for item in summary["episodes"]:
        tail = item[delta_key("last_quarter_quality")]
        change = item["last_minus_first_quarter_change_in_delta"]
        lines.append(f"{item['episode']}: " + "; ".join(
            f"{metric} {format_number(tail[metric])} (change {format_number(change[metric])})" for metric in METRICS))
    lines.extend(["", f"Recorded validation checks and finite metrics: {summary['recorded_checks_and_metrics_finite']}"])
    lines.extend(f"ISSUE: {issue}" for issue in summary["issues"])
    lines.extend(["", *summary["limitations"]])
    (destination / "summary.txt").write_text("\n".join(lines) + "\n")
    plot_quality(destination, summary, frames)
    plot_comparison(destination, summary)


def plot_quality(destination, summary, frames):
    episodes = summary["episodes"]
    fig, axes = plt.subplots(len(episodes), 3, figsize=(17, 3.0 * len(episodes)), squeeze=False)
    for row, item in enumerate(episodes):
        episode = item["episode"]
        indices = np.arange(1, item["last_evaluated_rgb_frame"] + 1)
        for column, metric in enumerate(METRICS):
            axis = axes[row, column]
            for mode in MODES:
                axis.plot(indices, [frames[episode][mode][int(index)][metric] for index in indices],
                          color=COLORS[mode], linewidth=0.8, alpha=0.85,
                          label=f"{SHORT_LABEL} {'OFF' if mode == 'original' else 'ON'}")
            axis.axvspan(*item["last_quarter_rgb_frames"], color="#888888", alpha=0.08, label="Last quarter")
            axis.set_title(f"Episode {episode} | {metric}")
            axis.set_xlabel("RGB frame index (observation frame 0 excluded)")
            axis.set_ylabel(LABELS[metric])
            axis.ticklabel_format(axis="y", useOffset=False)
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.935))
    fig.suptitle("Generated-history autoregression | Chunkwise FlashAttention ON in both paths\n"
                 "History: 23.5 chunks, excluding 1 latent sink and 8 denoising chunks", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(destination / "quality_vs_frame.png", dpi=160)
    plt.close(fig)


def plot_comparison(destination, summary):
    episodes = summary["episodes"]
    labels = [item["episode"] for item in episodes] + ["Macro mean"]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for axis, metric in zip(axes.flat, (*METRICS, "seconds")):
        for mode, offset in zip(MODES, (-0.19, 0.19)):
            if metric == "seconds":
                values = [item["implementations"][mode]["generation_stream_decode_seconds"] for item in episodes]
                values.append(summary["macro_average"][mode]["mean_generation_stream_decode_seconds"])
            else:
                values = [item["implementations"][mode]["mean_quality"][metric] for item in episodes]
                values.append(summary["macro_average"][mode]["mean_quality"][metric])
            axis.bar(positions + offset, [np.nan if value is None else value for value in values],
                     width=0.38, color=COLORS[mode], label=f"{SHORT_LABEL} {'OFF' if mode == 'original' else 'ON'}")
        axis.set_xticks(positions, labels)
        axis.set_ylabel("Generation + streaming decode (seconds)" if metric == "seconds" else LABELS[metric])
        axis.set_title("Generation + decode latency" if metric == "seconds" else f"Mean {metric} over predicted frames")
        axis.grid(axis="y", alpha=0.2)
        if metric == "seconds":
            speedups = [item["generation_stream_decode_speedup"] for item in episodes] + [summary["ratio_of_total_generation_stream_decode_times"]]
            axis.margins(y=0.2)
            for index, speedup in enumerate(speedups):
                axis.text(index, 0.98, f"{format_number(speedup, 3)}x", transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle(f"{COMPARISON_LABEL} OFF / ON | Chunkwise FlashAttention ON in both paths", fontsize=13)
    fig.text(0.5, 0.014, "Quality excludes observation frame 0. Timing excludes input/model loading, VAE encoding, warmup, saving, and metrics.\n"
             "One timed rollout per path per episode. Macro mean weights episodes equally; no quality acceptance threshold assumed.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(destination / "mean_quality_and_latency.png", dpi=160)
    plt.close(fig)


def plot_contact_sheet(path, destination, episode):
    """Read only a few frames from memory-mapped RGB arrays, never entire videos."""
    report = json.loads(path.read_text())
    prepared = Path(report.get("prepared_dir", path.parent.parent / "prepared"))
    paths = [prepared / "gt_rgb.npy"] + [
        path.parent / f"{mode}_rgb.npy" for mode in MODES
    ]
    if not all(item.is_file() for item in paths):
        return None
    arrays = [np.load(item, mmap_mode="r") for item in paths]
    last = min(array.shape[0] for array in arrays) - 1
    indices = [index for index in (1, 124, 248, 372, 496) if index <= last]
    if last < 496:
        indices = sorted(set(np.linspace(1, last, min(5, last), dtype=int).tolist()))
    fig, axes = plt.subplots(3, len(indices), figsize=(3.5 * len(indices), 8.2), squeeze=False)
    row_labels = ("GT", f"{SHORT_LABEL} OFF", f"{SHORT_LABEL} ON")
    for row, (array, label) in enumerate(zip(arrays, row_labels)):
        for column, index in enumerate(indices):
            axis = axes[row, column]
            axis.imshow(array[index])
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"RGB frame {index}")
            if column == 0:
                axis.set_ylabel(label, fontsize=11)
    fig.suptitle(f"Episode {episode} | Generated-history autoregression\n"
                 "Chunkwise FlashAttention ON in both paths; identical initial observation and noise", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    target = destination / f"episode_{episode}" / "contact_sheet.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return str(target.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Default: the input root")
    parser.add_argument("--expected-episodes", type=int, default=4)
    args = parser.parse_args()
    paths = sorted(args.input_root.rglob("benchmark.json"))
    if not paths:
        parser.error(f"No benchmark.json files under {args.input_root}")
    episodes, frames = [], {}
    for path in paths:
        result, curves = analyze_episode(path)
        episodes.append(result)
        if result["episode"] in frames:
            raise ValueError(f"Duplicate episode ID: {result['episode']}")
        frames[result["episode"]] = curves
    summary = aggregate(episodes, args.expected_episodes)
    destination = args.output_dir or args.input_root
    write_outputs(destination, summary, frames)
    for path, episode in zip(paths, episodes):
        contact_sheet = plot_contact_sheet(path, destination, episode["episode"])
        if contact_sheet:
            print(f"Contact sheet saved to {contact_sheet}")
    print((destination / "summary.txt").read_text())
    print(f"Analysis saved to {destination.resolve()}")
    if summary["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
