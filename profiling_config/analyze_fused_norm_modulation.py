"""Analyze four paired fused RMSNorm/AdaLN rollouts without using a GPU.

Reuse the framewise experiment's quality definitions, quarter summaries, videos,
and plots while checking this experiment's fixed optimizations, source snapshot,
497 RGB frame protocol, and actual fused calls from untimed validation forwards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import analyze_framewise_conditioning as analysis
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_ROOT = Path("/data2/jian/output/Miniworld/fused_norm_modulation_4episodes_rack3")
EXPECTED_EPISODES = {"002352", "004354", "007217", "007616"}
FORWARD_CASES = ((0, 2), (0, 16), (48, 2), (48, 16))


def configure_analysis():
    analysis.configure_comparison(
        candidate="fused", switch_name="fused_norm_modulation",
        label="Fused RMSNorm/AdaLN", short_label="Fusion",
        fixed_optimizations={"chunkwise_flash_attention": True, "framewise_conditioning": True},
    )


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_provenance(report, path, issues):
    provenance = report.get("source_provenance", {})
    snapshot = path.parent / "source_snapshot"
    if not provenance:
        issues.append("Source provenance is missing")
        return None
    if provenance.get("git_head") != report.get("git_head"):
        issues.append("Source provenance HEAD differs from the benchmark HEAD")
    tracked = snapshot / "working_tree.diff"
    if not tracked.is_file() or sha256_file(tracked) != provenance.get("tracked_diff_sha256"):
        issues.append("Saved tracked diff is missing or its SHA256 differs")
    sources = provenance.get("source_sha256", {})
    if "miniworld/fused_norm_modulation.py" not in sources:
        issues.append("Fused production kernel is missing from source hashes")
    untracked = provenance.get("untracked_source_sha256", {})
    for relative, expected in untracked.items():
        saved = snapshot / "untracked" / relative
        if not saved.is_file() or sha256_file(saved) != expected or sources.get(relative) != expected:
            issues.append(f"Untracked source snapshot/hash mismatch: {relative}")
    dirty_manifest = {"tracked_diff_sha256": provenance.get("tracked_diff_sha256"),
                      "untracked_source_sha256": untracked}
    dirty_hash = hashlib.sha256(json.dumps(dirty_manifest, sort_keys=True).encode()).hexdigest()
    if dirty_hash != provenance.get("dirty_diff_including_untracked_sha256"):
        issues.append("Combined tracked/untracked dirty diff SHA256 differs")
    fingerprint = {key: provenance.get(key) for key in (
        "git_head", "dirty_diff_including_untracked_sha256", "source_sha256",
    )}
    expected = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    if expected != provenance.get("source_fingerprint_sha256"):
        issues.append("Source fingerprint SHA256 differs")
    return provenance.get("source_fingerprint_sha256")


def validate_diagnostics(diagnostics, location, issues):
    for mode in analysis.MODES:
        result = diagnostics.get(mode, {})
        actual = result.get("fused_norm_modulation_calls")
        expected = result.get("expected_fused_norm_modulation_calls")
        valid = type(actual) is int and type(expected) is int and actual == expected
        valid = valid and (expected > 0 if mode == "fused" else expected == 0)
        if not valid or result.get("all_expected_calls_observed") is not True:
            issues.append(f"{location}, {mode}: missing or incorrect actual fused-call count")
        if mode == "fused" and sum(row.get("calls", 0) for row in result.get("input_signatures", [])) != actual:
            issues.append(f"{location}, {mode}: input signatures do not account for every fused call")


def analyze_episode(path):
    result, frames = analysis.analyze_episode(path)
    report = json.loads(path.read_text())
    issues = result["issues"]
    if report["rollout"]["rgb_frames"] != 497 or report["rollout"]["latent_frames"] != 125:
        issues.append("Expected the full 125 latent / 497 RGB frame rollout")
    result["source_fingerprint_sha256"] = validate_provenance(report, path, issues)
    result["source_provenance"] = report.get("source_provenance")
    startup = report.get("execution_validation", {}).get("startup_forward", {})
    validate_diagnostics(startup, "startup forward", issues)
    result["startup_execution_diagnostics"] = startup
    cases = report.get("forward_cases", [])
    keys = [(case["cache_latents_including_sink"], case["current_latents"]) for case in cases]
    if sorted(keys) != sorted(FORWARD_CASES):
        issues.append(f"Expected four forward cases {FORWARD_CASES}, got {keys}")
    for case in cases:
        location = f"history={case['cache_latents_including_sink']}, current={case['current_latents']}"
        validate_diagnostics(case.get("execution_diagnostics", {}), location, issues)
        if case.get("output_error", {}).get("finite") is not True:
            issues.append(f"{location}: missing/failed finite forward output check")
        if case.get("return_kv"):
            for component in ("k", "v"):
                checks = case.get("returned_current_kv_error", {}).get(component, {})
                for flag in ("finite", "dtype_unchanged", "shape_unchanged"):
                    if checks.get(flag) is not True:
                        issues.append(f"{location}, {component}: missing/failed {flag} KV check")
        for mode in analysis.MODES:
            measured = case["cuda"][mode]
            samples = measured["samples_ms"]
            if len(samples) < 2 or not all(math.isfinite(x) and x > 0 for x in samples):
                issues.append(f"{location}, {mode}: invalid CUDA timing samples")
    result["forward_cases"] = cases
    result["recorded_checks_and_metrics_finite"] = not issues
    return result, frames


def write_forward_outputs(destination, episodes):
    rows = []
    for episode in episodes:
        for case in episode["forward_cases"]:
            counts = case.get("execution_diagnostics", {})
            rows.append({
                "episode": episode["episode"],
                "cache_latents_including_sink": case["cache_latents_including_sink"],
                "history_chunks_excluding_sink": case["history_chunks_excluding_sink"],
                "current_latents": case["current_latents"],
                "current_chunks": case["current_latents"] / 2,
                "original_median_ms": case["cuda"]["original"]["median_ms"],
                "fused_median_ms": case["cuda"]["fused"]["median_ms"],
                "speedup": case["median_speedup_cuda"],
                "output_relative_rmse": case["output_error"]["relative_rmse"],
                "output_max_abs": case["output_error"]["max_abs"],
                "original_fused_calls_untimed": counts.get("original", {}).get("fused_norm_modulation_calls"),
                "fused_fused_calls_untimed": counts.get("fused", {}).get("fused_norm_modulation_calls"),
            })
    if rows:
        with (destination / "forward_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    for axis, (history, current) in zip(axes.flat, FORWARD_CASES):
        selected = [row for row in rows if row["cache_latents_including_sink"] == history
                    and row["current_latents"] == current]
        positions = np.arange(len(selected))
        for mode, offset in zip(analysis.MODES, (-0.19, 0.19)):
            axis.bar(positions + offset, [row[f"{mode}_median_ms"] for row in selected],
                     width=0.38, color=analysis.COLORS[mode],
                     label=f"Fusion {'OFF' if mode == 'original' else 'ON'}")
        axis.set_xticks(positions, [row["episode"] for row in selected])
        history_chunks = max(0, history - 1) / 2
        role = "commit" if current == 2 else "denoising"
        axis.set_title(f"History {history_chunks:g} chunks | Current {current / 2:g} {role} chunks")
        axis.set_ylabel("Complete cached forward (CUDA median ms)")
        axis.grid(axis="y", alpha=0.2)
        axis.margins(y=0.2)
        for index, row in enumerate(selected):
            axis.text(index, 0.98, f"{row['speedup']:.3f}x", transform=axis.get_xaxis_transform(),
                      ha="center", va="top", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Fused RMSNorm/AdaLN | Chunkwise FlashAttention and framewise conditioning ON")
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
    paths = sorted(args.input_root.rglob("benchmark.json"))
    if not paths:
        parser.error(f"No benchmark.json files under {args.input_root}")
    episodes, frames = [], {}
    for path in paths:
        result, curves = analyze_episode(path)
        if result["episode"] in frames:
            raise ValueError(f"Duplicate episode ID: {result['episode']}")
        episodes.append(result)
        frames[result["episode"]] = curves
    summary = analysis.aggregate(episodes, args.expected_episodes)
    if args.expected_episodes == 4 and set(summary["episode_ids"]) != EXPECTED_EPISODES:
        summary["issues"].append(f"Expected episode IDs {sorted(EXPECTED_EPISODES)}")
    fingerprints = {item["source_fingerprint_sha256"] for item in episodes}
    if len(fingerprints) != 1 or None in fingerprints:
        summary["issues"].append("Episodes were measured with different or missing source fingerprints")
    summary["source_fingerprints_sha256"] = sorted(value for value in fingerprints if value is not None)
    summary["execution_diagnostics_scope"] = "Actual fused calls in separate untimed forwards; no rollout hot-path instrumentation"
    summary["limitations"] = [line for line in summary["limitations"] if "no per-kernel backend trace" not in line]
    summary["limitations"].append("Fused launches are verified in untimed forwards; rollout-wide kernel counts are not collected")
    summary["recorded_checks_and_metrics_finite"] = not summary["issues"]
    destination = args.output_dir or args.input_root
    analysis.write_outputs(destination, summary, frames)
    write_forward_outputs(destination, episodes)
    for path, episode in zip(paths, episodes):
        analysis.plot_contact_sheet(path, destination, episode["episode"])
    print((destination / "summary.txt").read_text())
    print(f"Analysis saved to {destination.resolve()}")
    if summary["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
