"""Analyze the four paired RoPE-only rollouts using the archived quality protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import analyze_framewise_conditioning as analysis
from analyze_fused_norm_modulation import validate_provenance
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_ROOT = Path(
    "/data2/jian/output/Miniworld/incremental_rebuild_20261018/stage1_reuse_rope_frequencies"
)
EXPECTED_EPISODES = {"002352", "004354", "007217", "007616"}
FORWARD_CASES = ((0, 2), (0, 16), (48, 2), (48, 16))


def configure_analysis():
    analysis.configure_comparison(
        "rope", "reuse_rope_frequencies", "RoPE frequency reuse", "RoPE reuse",
        {"chunkwise_flash_attention": True, "framewise_conditioning": True,
         "fused_norm_modulation": True},
    )


def analyze_episode(path):
    result, frames = analysis.analyze_episode(path)
    report = json.loads(path.read_text())
    issues = result["issues"]
    if report.get("execution_mode") != "no_grad" or report.get("autocast_cache_enabled") is not True:
        issues.append("Expected official no_grad execution with the autocast weight cache enabled")
    if report["rollout"]["rgb_frames"] != 497 or report["rollout"]["latent_frames"] != 125:
        issues.append("Expected the full 125-latent / 497-RGB rollout")
    result["source_fingerprint_sha256"] = validate_provenance(report, path, issues)
    result["source_provenance"] = report.get("source_provenance")
    expected_properties = {"all_normal_tensors": True, "all_require_grad": True, "all_float32": True}
    for name in analysis.MODES:
        if report.get("parameter_properties", {}).get(name) != expected_properties:
            issues.append(f"{name}: expected ordinary FP32 parameters with requires_grad=True")
    cases = report.get("forward_cases", [])
    if sorted((case["cache_latents_including_sink"], case["current_latents"]) for case in cases) != sorted(FORWARD_CASES):
        issues.append("Missing or duplicate forward timing cases")
    kv_equal = True
    for case in cases:
        location = f"history={case['history_chunks_excluding_sink']}, current={case['current_latents'] / 2} chunks"
        if case.get("output_error", {}).get("finite") is not True:
            issues.append(f"{location}: failed finite output check")
        for name, expected in (("original", 56), ("rope", 1)):
            observed = case.get("execution_diagnostics", {}).get(name, {})
            requests = observed.get("requests", [])
            expected_builds = sum(item["calls"] for item in requests
                                  if item["num_frames"] != report["trained_latents"] or item["start_frame"] != 0)
            counts_valid = (observed.get("rope_get_calls") == observed.get("expected_get_calls") == expected
                            and sum(item["calls"] for item in requests) == expected
                            and observed.get("trained_num_frames") == report["trained_latents"]
                            and observed.get("rope_build_calls") == observed.get("expected_build_calls") == expected_builds)
            if observed.get("checks_passed") is not True or not counts_valid:
                issues.append(f"{location}, {name}: missing or incorrect RoPE request/build counts")
            samples = case["cuda"][name]["samples_ms"]
            if len(samples) < 2 or any(not math.isfinite(value) or value <= 0 for value in samples):
                issues.append(f"{location}, {name}: invalid CUDA timing samples")
        if case.get("return_kv"):
            for component in ("k", "v"):
                checks = case.get("returned_current_kv_error", {}).get(component, {})
                if any(checks.get(key) is not True for key in ("finite", "dtype_unchanged", "shape_unchanged")):
                    issues.append(f"{location}, {component}: failed returned-KV validation")
                kv_equal = kv_equal and checks.get("max_abs") == 0
    result["forward_cases"] = cases
    result["forward_outputs_bitwise_equal"] = bool(cases) and all(case["output_error"]["max_abs"] == 0 for case in cases)
    result["returned_current_kv_bitwise_equal"] = kv_equal
    result["generated_latents_bitwise_equal"] = report["quality"]["latent_error"]["max_abs"] == 0
    result["generated_rgb_identical"] = len({run["rgb_sha256"] for run in report["rollout"]["runs"].values()}) == 1
    for key in ("forward_outputs_bitwise_equal", "returned_current_kv_bitwise_equal",
                "generated_latents_bitwise_equal", "generated_rgb_identical"):
        if not result[key]:
            issues.append(f"Expected exact RoPE-only equality: {key}")
    result["recorded_checks_and_metrics_finite"] = not issues
    return result, frames


def write_forward_outputs(destination, episodes):
    rows = []
    for episode in episodes:
        for case in episode["forward_cases"]:
            row = {"episode": episode["episode"],
                   "history_chunks_excluding_sink": case["history_chunks_excluding_sink"],
                   "current_chunks": case["current_latents"] / 2,
                   "speedup": case["median_speedup_cuda"], "output_max_abs": case["output_error"]["max_abs"]}
            for name in analysis.MODES:
                for key in ("median_ms", "min_ms", "max_ms", "stdev_ms"):
                    row[f"{name}_cuda_{key}"] = case["cuda"][name][key]
                row[f"{name}_wall_median_ms"] = case["wall"][name]["median_ms"]
                for key in ("rope_get_calls", "rope_build_calls"):
                    row[f"{name}_{key}_untimed"] = case["execution_diagnostics"][name][key]
            rows.append(row)
    with (destination / "forward_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, (cached, current) in zip(axes.flat, FORWARD_CASES):
        history = max(0, cached - 1) / 2
        selected = [row for row in rows if row["history_chunks_excluding_sink"] == history
                    and row["current_chunks"] == current / 2]
        positions = np.arange(len(selected))
        for mode, offset in zip(analysis.MODES, (-0.19, 0.19)):
            axis.bar(positions + offset, [row[f"{mode}_cuda_median_ms"] for row in selected],
                     width=0.38, color=analysis.COLORS[mode],
                     label=f"RoPE reuse {'OFF' if mode == 'original' else 'ON'}")
        axis.set_xticks(positions, [row["episode"] for row in selected])
        axis.set_title(f"History {history:g} chunks | Current {current / 2:g} chunks")
        axis.set_ylabel("Complete cached forward, CUDA median ms")
        axis.grid(axis="y", alpha=0.2)
        axis.margins(y=0.2)
        for index, row in enumerate(selected):
            axis.text(index, 0.98, f"{row['speedup']:.3f}x", transform=axis.get_xaxis_transform(), ha="center", va="top")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("RoPE frequency reuse | no_grad and persistent BF16 autocast in both paths")
    fig.text(0.5, 0.01, "History excludes the 1-latent sink and current chunks; 2 latent frames per chunk.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    fig.savefig(destination / "forward_latency.png", dpi=160)
    plt.close(fig)


def main():
    configure_analysis()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=4)
    args = parser.parse_args()
    paths = sorted(args.input_root.glob("episode_*/results/benchmark.json"))
    if not paths:
        parser.error("No episode_*/results/benchmark.json files found")
    episodes, frames = [], {}
    for path in paths:
        result, curves = analyze_episode(path)
        if result["episode"] in frames:
            raise ValueError(f"Duplicate episode: {result['episode']}")
        episodes.append(result)
        frames[result["episode"]] = curves
    summary = analysis.aggregate(episodes, args.expected_episodes)
    if args.expected_episodes == 4 and set(summary["episode_ids"]) != EXPECTED_EPISODES:
        summary["issues"].append("Episode IDs differ from the four-episode protocol")
    fingerprints = {item["source_fingerprint_sha256"] for item in episodes}
    if len(fingerprints) != 1 or None in fingerprints:
        summary["issues"].append("Measured source fingerprints differ or are missing")
    summary["source_fingerprints_sha256"] = sorted(value for value in fingerprints if value)
    for key in ("forward_outputs_bitwise_equal", "returned_current_kv_bitwise_equal",
                "generated_latents_bitwise_equal", "generated_rgb_identical"):
        summary[f"all_{key}"] = all(item[key] for item in episodes)
    summary["limitations"].append("RoPE requests/builds counted in separate untimed forwards; no rollout-wide instrumentation")
    summary["recorded_checks_and_metrics_finite"] = not summary["issues"]
    destination = args.output_dir or args.input_root
    analysis.write_outputs(destination, summary, frames)
    write_forward_outputs(destination, episodes)
    for path, item in zip(paths, episodes):
        analysis.plot_contact_sheet(path, destination, item["episode"])
    print((destination / "summary.txt").read_text())
    print("Bitwise forward/KV/latent/RGB:", *(summary[f"all_{key}"] for key in (
        "forward_outputs_bitwise_equal", "returned_current_kv_bitwise_equal",
        "generated_latents_bitwise_equal", "generated_rgb_identical")))
    if summary["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
