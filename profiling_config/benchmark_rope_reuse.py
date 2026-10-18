"""Compare only per-forward RoPE frequency reuse with the official inference path.

Both implementations use no_grad, ordinary FP32 parameters, persistent BF16
autocast, Chunkwise FlashAttention, framewise conditioning, and fused RMSNorm/AdaLN.
Only reuse_rope_frequencies changes. Prepared inputs, noise, sampler settings,
cache eviction, streaming VAE decode, and quality metrics use the shared protocol.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

import benchmark_framewise_conditioning as protocol
from benchmark_fused_norm_modulation import source_provenance


OUTPUT_ROOT = Path(
    "/data2/jian/output/Miniworld/incremental_rebuild_20261018/stage1_reuse_rope_frequencies"
)
PREPARED_ROOT = Path("/data2/jian/output/Miniworld/framewise_conditioning_4episodes_rack3")
FIXED = {"chunkwise_flash_attention": True, "framewise_conditioning": True,
         "fused_norm_modulation": True}


def configure_protocol():
    protocol.configure_comparison("rope", "reuse_rope_frequencies", "RoPE frequency reuse", FIXED)


def execution_diagnostics(name, model, call):
    """Count actual frequency requests/builds in a separate untimed forward."""
    rope = model.net.feat_rope
    original_get, original_build = rope._get_freqs, rope._build_freqs
    counts = {"rope_get_calls": 0, "rope_build_calls": 0}
    requests = Counter()

    def get(num_frames, device, start_frame=0):
        counts["rope_get_calls"] += 1
        requests[(num_frames, start_frame)] += 1
        return original_get(num_frames, device, start_frame=start_frame)

    def build(*args, **kwargs):
        counts["rope_build_calls"] += 1
        return original_build(*args, **kwargs)

    rope._get_freqs, rope._build_freqs = get, build
    try:
        result = call()
        del result
    finally:
        del rope._get_freqs
        del rope._build_freqs
    expected = 1 if name == "rope" else 2 * len(model.net.blocks)
    expected_builds = sum(count for (frames, start), count in requests.items()
                          if frames != rope._num_frames or start != 0)
    if counts["rope_get_calls"] != expected or counts["rope_build_calls"] != expected_builds:
        raise AssertionError(f"{name}: expected {expected} requests/{expected_builds} builds, got {counts}")
    return {**counts, "expected_get_calls": expected, "expected_build_calls": expected_builds,
            "trained_num_frames": rope._num_frames,
            "requests": [{"num_frames": frames, "start_frame": start, "calls": count}
                         for (frames, start), count in sorted(requests.items())], "checks_passed": True,
            "scope": "one separate untimed forward; measured forwards and rollouts have no counters"}


def report_metadata(args, models, data):
    provenance = source_provenance(args.output_dir)
    properties = {}
    for name, model in models.items():
        parameters = list(model.parameters())
        properties[name] = {
            "all_normal_tensors": all(not parameter.is_inference() for parameter in parameters),
            "all_require_grad": all(parameter.requires_grad for parameter in parameters),
            "all_float32": all(parameter.dtype == torch.float32 for parameter in parameters),
        }
        if not all(properties[name].values()):
            raise AssertionError(f"{name}: expected ordinary FP32 parameters with requires_grad=True")
    return {
        "source_provenance": provenance, "source_sha256": provenance["source_sha256"],
        "parameter_properties": properties,
        "autocast_scope": "One BF16 autocast context across repeated forwards; one context per complete rollout",
        "timing_definitions": {
            "cuda": "CUDA event span of the complete forward, including stream idle gaps",
            "wall": "Host wall time through CUDA synchronization; not asynchronous submission time",
        },
        "execution_validation": "RoPE requests/builds counted only in separate untimed forwards",
    }


def main():
    configure_protocol()
    args = protocol.parse_args(defaults={
        "prepared_dir": PREPARED_ROOT / "episode_002352" / "prepared",
        "output_dir": OUTPUT_ROOT / "episode_002352" / "results",
        "rollout_latents": 125,
    }, description=__doc__)
    protocol.main(args, report_metadata=report_metadata, forward_diagnostics=execution_diagnostics,
                  execution_mode="no_grad")


if __name__ == "__main__":
    main()
