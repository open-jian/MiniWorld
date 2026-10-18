"""Validate fused RMSNorm/AdaLN against the same model with fusion disabled.

Both paths keep Chunkwise FlashAttention and framewise conditioning enabled.
The shared benchmark preserves the prepared episode, noise, 40 effective
sampling steps, cache eviction, BF16 autocast, and generation/decode timer.
The default 125 latent frames produce the complete 497 RGB frame rollout.

Example::

    CUDA_VISIBLE_DEVICES=0 python profiling_config/benchmark_fused_norm_modulation.py \
        --episode-id 002352 \
        --prepared-dir /path/to/episode_002352/prepared \
        --output-dir /path/to/episode_002352/results
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import torch

import benchmark_framewise_conditioning as protocol
import miniworld.miniworld as model_module


OUTPUT_ROOT = Path("/data2/jian/output/Miniworld/fused_norm_modulation_4episodes_rack3")
PREPARED_ROOT = Path("/data2/jian/output/Miniworld/framewise_conditioning_4episodes_rack3")


def configure_protocol():
    protocol.configure_comparison(
        candidate="fused", switch_name="fused_norm_modulation",
        label="Fused RMSNorm/AdaLN",
        fixed_optimizations={"chunkwise_flash_attention": True, "framewise_conditioning": True,
                             "reuse_rope_frequencies": False},
    )


def execution_diagnostics(name, model, call):
    """Count completed kernel calls in a separate, untimed model forward."""
    original = model_module.fused_norm_modulation
    signatures = Counter()

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        x = args[0]
        signatures[(tuple(x.shape), tuple(x.stride()), str(x.dtype))] += 1
        return result

    model_module.fused_norm_modulation = record
    try:
        result = call()
        del result
    finally:
        model_module.fused_norm_modulation = original
    expected = 2 * len(model.net.blocks) + 1 if name == "fused" else 0
    calls = sum(signatures.values())
    if calls != expected:
        raise AssertionError(f"{name}: expected {expected} fused launches, observed {calls}")
    return {
        "fused_norm_modulation_calls": calls,
        "expected_fused_norm_modulation_calls": expected,
        "all_expected_calls_observed": True,
        "scope": "one separate forward; excluded from all timings and rollout",
        "input_signatures": [
            {"shape": list(shape), "stride": list(stride), "dtype": dtype, "calls": count}
            for (shape, stride, dtype), count in signatures.items()
        ],
    }


def source_provenance(output_dir):
    """Fingerprint dirty tracked changes and preserve untracked Python sources."""
    root = protocol.REPO_ROOT
    snapshot = output_dir / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    tracked_diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"], cwd=root)
    (snapshot / "working_tree.diff").write_bytes(tracked_diff)
    tracked_diff_hash = hashlib.sha256(tracked_diff).hexdigest()
    source_paths = sorted({
        *root.joinpath("miniworld").rglob("*.py"),
        *root.joinpath("profiling_config").glob("*.py"),
        *root.joinpath("tests").glob("*.py"),
        root / "optimization_config.toml",
    })
    source_hashes = {str(path.relative_to(root)): protocol.hash_file(path)
                     for path in source_paths if path.is_file()}
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root,
    ).decode().split("\0")
    untracked_hashes = {}
    for relative in sorted(set(untracked) & source_hashes.keys()):
        untracked_hashes[relative] = source_hashes[relative]
        target = snapshot / "untracked" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    dirty_manifest = {"tracked_diff_sha256": tracked_diff_hash,
                      "untracked_source_sha256": untracked_hashes}
    dirty_hash = hashlib.sha256(json.dumps(dirty_manifest, sort_keys=True).encode()).hexdigest()
    result = {
        "git_head": protocol.git("rev-parse", "HEAD"),
        "git_status": protocol.git("status", "--short"),
        "tracked_diff_sha256": tracked_diff_hash,
        "untracked_source_sha256": untracked_hashes,
        "dirty_diff_including_untracked_sha256": dirty_hash,
        "source_sha256": source_hashes,
        "snapshot_dir": str(snapshot.resolve()),
    }
    fingerprint = {key: result[key] for key in (
        "git_head", "dirty_diff_including_untracked_sha256", "source_sha256",
    )}
    result["source_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode(),
    ).hexdigest()
    protocol.dump(snapshot / "manifest.json", result)
    return result


def report_metadata(args, models, data):
    provenance = source_provenance(args.output_dir)
    startup = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for name, model in models.items():
            startup[name] = execution_diagnostics(name, model, lambda model=model: model.net.forward_with_cache(
                data["latents"][:, :, :2], torch.zeros(1, 2, device="cuda"),
                data["cond_seq"][:, :2], return_kv=True, chunk_size=2,
            ))
    return {
        "source_provenance": provenance,
        "source_sha256": provenance["source_sha256"],
        "execution_validation": {
            "counter": "successful calls of miniworld.miniworld.fused_norm_modulation",
            "scope": "separate untimed forwards; rollout has no fused-call instrumentation",
            "startup_forward": startup,
        },
    }


def main():
    configure_protocol()
    args = protocol.parse_args(defaults={
        "prepared_dir": PREPARED_ROOT / "episode_002352" / "prepared",
        "output_dir": OUTPUT_ROOT / "episode_002352" / "results",
        "rollout_latents": 125,
    }, description=__doc__)
    protocol.main(args, report_metadata=report_metadata, forward_diagnostics=execution_diagnostics)


if __name__ == "__main__":
    main()
