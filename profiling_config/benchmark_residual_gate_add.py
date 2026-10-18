"""Compare only fused residual gate/add with the accepted inference baseline.

Both paths keep Chunkwise FlashAttention, framewise conditioning, fused
RMSNorm/AdaLN, and RoPE frequency reuse enabled. The shared protocol preserves
prepared inputs, fixed noise, all sampler settings, eviction, streaming VAE,
and the generation-plus-streaming-decode timer used in previous full rollouts.
The default 125 latents produce 497 RGB frames. Repeated rollouts are separate
processes/output directories; the external launcher balances their OFF/ON order.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import benchmark_framewise_conditioning as protocol
from benchmark_fused_norm_modulation import source_provenance
import miniworld.miniworld as model_module


OUTPUT_ROOT = Path("/data2/jian/output/Miniworld/fused_residual_gate_add_4episodes_rack3")
PREPARED_ROOT = Path("/data2/jian/output/Miniworld/framewise_conditioning_4episodes_rack3")
FIXED = {
    "chunkwise_flash_attention": True,
    "framewise_conditioning": True,
    "fused_norm_modulation": True,
    "reuse_rope_frequencies": True,
}


def configure_protocol():
    protocol.configure_comparison(
        "gate", "fused_residual_gate_add", "Fused residual gate/add", FIXED,
    )


def tensor_fingerprint(tensor):
    """Hash actual storage bits, including signed zeros, outside all timings."""
    if tensor is None:
        return None
    bits = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
    }


def output_fingerprints(output):
    prediction, current_kv = output
    return {
        "prediction": tensor_fingerprint(prediction),
        "current_kv": None if current_kv is None else [
            [tensor_fingerprint(key), tensor_fingerprint(value)] for key, value in current_kv
        ],
    }


def execution_diagnostics(name, model, call):
    """Count completed fused calls in a separate forward and hash its outputs."""
    original = model_module.fused_residual_gate_add
    signatures = Counter()

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        residual, update, gate = args[:3]
        signature = tuple(
            (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
            for tensor in (residual, update, gate)
        )
        signatures[signature] += 1
        return result

    model_module.fused_residual_gate_add = record
    try:
        output = call()
        fingerprints = output_fingerprints(output)
        del output
    finally:
        model_module.fused_residual_gate_add = original
    expected = 2 * len(model.net.blocks) if name == "gate" else 0
    calls = sum(signatures.values())
    if calls != expected:
        raise AssertionError(f"{name}: expected {expected} completed fused calls, observed {calls}")
    return {
        "fused_residual_gate_add_calls": calls,
        "expected_fused_residual_gate_add_calls": expected,
        "all_expected_calls_observed": True,
        "scope": "one separate untimed forward; no instrumentation in measured forwards or rollout",
        "output_fingerprints": fingerprints,
        "input_signatures": [
            {"residual": {"shape": list(signature[0][0]), "stride": list(signature[0][1]), "dtype": signature[0][2]},
             "update": {"shape": list(signature[1][0]), "stride": list(signature[1][1]), "dtype": signature[1][2]},
             "gate": {"shape": list(signature[2][0]), "stride": list(signature[2][1]), "dtype": signature[2][2]},
             "calls": count}
            for signature, count in signatures.items()
        ],
    }


def report_metadata(args, models, data):
    """Save provenance only when the root has frozen the measured source tree."""
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
    with torch.autocast("cuda", dtype=torch.bfloat16):
        startup = {
            name: execution_diagnostics(name, model, lambda model=model: model.net.forward_with_cache(
                data["latents"][:, :, :2], torch.zeros(1, 2, device="cuda"),
                data["cond_seq"][:, :2], return_kv=True, chunk_size=2,
            )) for name, model in models.items()
        }
    if startup["original"]["output_fingerprints"] != startup["gate"]["output_fingerprints"]:
        raise AssertionError("Startup prediction or returned current K/V differs in its stored bits")
    return {
        "source_provenance": provenance, "source_sha256": provenance["source_sha256"],
        "parameter_properties": properties,
        "autocast_scope": "One BF16 autocast context across repeated forwards; one context per complete rollout",
        "timing_definitions": {
            "cuda": "CUDA event span of one complete model forward, in milliseconds",
            "wall": "Host wall time through CUDA synchronization, in milliseconds",
            "rollout": "Generation plus streaming VAE decode from prepared GPU inputs to RGB, in seconds",
        },
        "execution_validation": {"startup_forward": startup,
            "counter": "successful calls of miniworld.miniworld.fused_residual_gate_add",
            "scope": "separate untimed forwards; no instrumentation during rollout"},
    }


def validate_saved_outputs(args):
    """Check saved rollout arrays byte-for-byte after shared timing and metrics."""
    path = args.output_dir / "benchmark.json"
    report = json.loads(path.read_text())
    checks = {}
    if args.rollout_latents:
        latents = {
            name: torch.load(args.output_dir / f"{name}_latents.pt", map_location="cpu", weights_only=True)
            for name in protocol.MODES
        }
        fingerprints = {name: tensor_fingerprint(value) for name, value in latents.items()}
        rgb = {name: np.load(args.output_dir / f"{name}_rgb.npy", mmap_mode="r")
               for name in protocol.MODES}
        checks.update({
            "latent_fingerprints": fingerprints,
            "generated_latents_bitwise_equal": fingerprints["original"] == fingerprints["gate"],
            "generated_rgb_identical": bool(np.array_equal(rgb["original"], rgb["gate"])),
        })
    checks["forward_output_bits_equal"] = all(
        case["execution_diagnostics"]["original"]["output_fingerprints"] ==
        case["execution_diagnostics"]["gate"]["output_fingerprints"]
        for case in report.get("forward_cases", [])
    )
    checks["scope"] = "Saved latent bit hashes, RGB arrays, and separately measured forward output hashes"
    report["exact_output_validation"] = checks
    protocol.dump(path, report)
    protocol.dump(args.output_dir / "exact_output_validation.json", checks)
    required = [checks["forward_output_bits_equal"]]
    if args.rollout_latents:
        required.extend((checks["generated_latents_bitwise_equal"], checks["generated_rgb_identical"]))
    if not all(required):
        raise AssertionError(f"Residual gate/add exact-output validation failed: {checks}")


def main():
    configure_protocol()
    args = protocol.parse_args(defaults={
        "prepared_dir": PREPARED_ROOT / "episode_002352" / "prepared",
        "output_dir": OUTPUT_ROOT / "episode_002352" / "repeat_01" / "results",
        "rollout_latents": 125, "warmup": 7, "repeats": 31,
    }, description=__doc__)
    protocol.main(args, report_metadata=report_metadata, forward_diagnostics=execution_diagnostics,
                  execution_mode="no_grad")
    validate_saved_outputs(args)


if __name__ == "__main__":
    main()
