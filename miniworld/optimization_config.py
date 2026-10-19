"""Validated, per-model inference switches; independent of checkpoint weights.

Read once when a model is constructed, never inside the denoising loop. Each
model keeps its own immutable snapshot so editing the file or building another
model cannot change an already running experiment. Python 3.11's TOML parser
also rejects duplicate keys, and no extra configuration dependency is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


# Resolve against this checkout, not the caller's working directory. Explicit
# relative paths, in contrast, are intentionally relative to the caller's cwd.
DEFAULT_OPTIMIZATION_CONFIG = Path(__file__).resolve().parents[1] / "optimization_config.toml"


@dataclass(frozen=True)
class OptimizationConfig:
    """Only implemented execution optimizations belong here.

    These are requests: the existing device/dtype guards may still select the
    SDPA fallback. Sampling parameters, precision and model structure are kept
    in their existing configuration rather than hidden behind a speed switch.
    """

    chunkwise_flash_attention: bool = True
    # Provenance only; it is not a supported TOML key or a checkpoint tensor.
    source_path: str | None = field(default=None, repr=False, compare=False)
    # Disable explicitly for the reference path; GEMM shape affects BF16 rounding.
    framewise_conditioning: bool = True
    # Fuse only supported cached inference calls; reference arithmetic remains available.
    fused_norm_modulation: bool = True
    # Reuse RoPE coefficients within one cached inference forward, never across calls.
    reuse_rope_frequencies: bool = True
    # Fuse BF16 residual gating/addition without removing the product's rounding.
    fused_residual_gate_add: bool = True
    # Fuse current Q/K rotation arithmetic; frequency reuse remains independent.
    fused_rope: bool = True
    # Fuse the separate W12 bias add, SiLU, and product with BF16 boundaries intact.
    fused_bias_swiglu: bool = True

    def __post_init__(self) -> None:
        # bool("false") is True: never coerce strings/numbers into switches.
        for name in (
            "chunkwise_flash_attention", "framewise_conditioning", "fused_norm_modulation",
            "reuse_rope_frequencies", "fused_residual_gate_add", "fused_rope",
            "fused_bias_swiglu",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a TOML/Python boolean")

    def as_dict(self) -> dict:
        """Serializable requested options, separate from backend fallback."""
        return {
            "schema_version": 1,
            "chunkwise_flash_attention": self.chunkwise_flash_attention,
            "framewise_conditioning": self.framewise_conditioning,
            "fused_norm_modulation": self.fused_norm_modulation,
            "reuse_rope_frequencies": self.reuse_rope_frequencies,
            "fused_residual_gate_add": self.fused_residual_gate_add,
            "fused_rope": self.fused_rope,
            "fused_bias_swiglu": self.fused_bias_swiglu,
        }

    def snapshot(self) -> dict:
        """Provenance and values for run logs, not the model state_dict."""
        return {"source_path": self.source_path, "requested": self.as_dict()}


def load_optimization_config(
    source: OptimizationConfig | str | Path | None = None,
) -> OptimizationConfig:
    """Load the default file, a chosen TOML file, or an explicit Python config.

    Absent switches use the current defaults (implemented optimizations enabled).
    Set a switch to False explicitly when a reference execution path is needed.
    An absent/invalid *file*, version, or unknown key is an error, not a silent
    fallback. Add future switches here together with their actual call sites;
    unsupported optimizations must not appear as enabled no-op placeholders.
    """
    if isinstance(source, OptimizationConfig):
        return source
    path = (DEFAULT_OPTIMIZATION_CONFIG if source is None else Path(source)).expanduser().resolve()
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    unknown = set(values) - {
        "schema_version", "chunkwise_flash_attention", "framewise_conditioning", "fused_norm_modulation",
        "reuse_rope_frequencies", "fused_residual_gate_add", "fused_rope",
        "fused_bias_swiglu",
    }
    if unknown:
        raise ValueError(f"Unknown optimization config keys in {path}: {', '.join(sorted(unknown))}")
    version = values.get("schema_version")
    if type(version) is not int or version != 1:
        raise ValueError(f"{path}: schema_version must be the integer 1")
    return OptimizationConfig(
        chunkwise_flash_attention=values.get("chunkwise_flash_attention", True),
        source_path=str(path),
        framewise_conditioning=values.get("framewise_conditioning", True),
        fused_norm_modulation=values.get("fused_norm_modulation", True),
        reuse_rope_frequencies=values.get("reuse_rope_frequencies", True),
        fused_residual_gate_add=values.get("fused_residual_gate_add", True),
        fused_rope=values.get("fused_rope", True),
        fused_bias_swiglu=values.get("fused_bias_swiglu", True),
    )
