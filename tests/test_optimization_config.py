"""Configuration regressions for inference optimization switches.

Run with ``python -m unittest discover -s tests -v`` from the repository root.
The CUDA cases exercise actual attention dispatch with nonzero model outputs;
the remaining cases also run on a CPU-only machine.
"""

import os
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from miniworld import miniworld as mw
from miniworld.denoiser import Denoiser, DenoiserConfig
from miniworld.optimization_config import (
    DEFAULT_OPTIMIZATION_CONFIG,
    OptimizationConfig,
    load_optimization_config,
)
from test_chunk_flash_attention import small_model


class OptimizationConfigLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "optimizations.toml"

    def write_config(self, text):
        self.path.write_text(text, encoding="utf-8")
        return self.path

    def test_file_and_instance_inputs_have_same_effective_values(self):
        self.write_config("schema_version = 1\nchunkwise_flash_attention = false\n")
        from_path = load_optimization_config(self.path)
        from_string = load_optimization_config(str(self.path))
        from_instance = load_optimization_config(OptimizationConfig(False))
        expected = {
            "schema_version": 1,
            "chunkwise_flash_attention": False,
            "framewise_conditioning": True,
            "fused_norm_modulation": True,
            "reuse_rope_frequencies": True,
            "fused_residual_gate_add": True,
            "fused_rope": True,
            "fused_bias_swiglu": True,
            "fused_k_concat_cast": True,
        }
        for config in (from_path, from_string, from_instance):
            self.assertEqual(config.as_dict(), expected)
        self.assertEqual(Path(from_path.source_path).resolve(), self.path.resolve())
        self.assertIsNone(from_instance.source_path)

    def test_missing_switch_preserves_existing_enabled_default(self):
        self.write_config("schema_version = 1\n")
        self.assertIs(load_optimization_config(self.path).chunkwise_flash_attention, True)
        self.assertIs(load_optimization_config(self.path).framewise_conditioning, True)
        self.assertIs(load_optimization_config(self.path).fused_norm_modulation, True)
        self.assertIs(load_optimization_config(self.path).reuse_rope_frequencies, True)
        self.assertIs(load_optimization_config(self.path).fused_residual_gate_add, True)
        self.assertIs(load_optimization_config(self.path).fused_rope, True)
        self.assertIs(load_optimization_config(self.path).fused_bias_swiglu, True)
        self.assertIs(load_optimization_config(self.path).fused_k_concat_cast, True)

    def test_fused_k_concat_cast_is_independent_and_serialized(self):
        self.assertIs(OptimizationConfig().fused_k_concat_cast, True)
        self.write_config("schema_version = 1\nfused_k_concat_cast = false\n")
        config = load_optimization_config(self.path)
        self.assertIs(config.fused_k_concat_cast, False)
        self.assertIs(config.snapshot()["requested"]["fused_k_concat_cast"], False)
        previous = ("chunkwise_flash_attention", "framewise_conditioning",
                    "fused_norm_modulation", "reuse_rope_frequencies",
                    "fused_residual_gate_add", "fused_rope", "fused_bias_swiglu")
        for name in previous:
            self.assertIs(getattr(config, name), True)
        self.write_config("schema_version = 1\nfused_k_concat_cast = true\n" +
                          "".join(f"{name} = false\n" for name in previous))
        config = load_optimization_config(self.path)
        self.assertIs(config.fused_k_concat_cast, True)
        for name in previous:
            self.assertIs(getattr(config, name), False)

    def test_fused_bias_swiglu_is_independent_and_serialized(self):
        self.assertIs(OptimizationConfig().fused_bias_swiglu, True)
        self.write_config("schema_version = 1\nfused_bias_swiglu = false\n")
        config = load_optimization_config(self.path)
        self.assertIs(config.fused_bias_swiglu, False)
        self.assertIs(config.snapshot()["requested"]["fused_bias_swiglu"], False)
        previous = ("chunkwise_flash_attention", "framewise_conditioning",
                    "fused_norm_modulation", "reuse_rope_frequencies",
                    "fused_residual_gate_add", "fused_rope")
        for name in previous:
            self.assertIs(getattr(config, name), True)
        self.write_config("schema_version = 1\nfused_bias_swiglu = true\n" +
                          "".join(f"{name} = false\n" for name in previous))
        config = load_optimization_config(self.path)
        self.assertIs(config.fused_bias_swiglu, True)
        for name in previous:
            self.assertIs(getattr(config, name), False)

    def test_fused_rope_is_independent_and_serialized(self):
        self.assertIs(OptimizationConfig().fused_rope, True)
        self.write_config("schema_version = 1\nfused_rope = false\n")
        config = load_optimization_config(self.path)
        self.assertIs(config.fused_rope, False)
        self.assertIs(config.snapshot()["requested"]["fused_rope"], False)
        previous = ("chunkwise_flash_attention", "framewise_conditioning",
                    "fused_norm_modulation", "reuse_rope_frequencies", "fused_residual_gate_add")
        for name in previous:
            self.assertIs(getattr(config, name), True)
        self.write_config("schema_version = 1\nfused_rope = true\n" +
                          "".join(f"{name} = false\n" for name in previous))
        config = load_optimization_config(self.path)
        self.assertIs(config.fused_rope, True)
        for name in previous:
            self.assertIs(getattr(config, name), False)

    def test_residual_gate_add_is_independent_and_serialized(self):
        self.assertIs(OptimizationConfig().fused_residual_gate_add, True)
        self.write_config("schema_version = 1\nfused_residual_gate_add = false\n")
        config = load_optimization_config(self.path)
        self.assertIs(config.fused_residual_gate_add, False)
        self.assertIs(config.snapshot()["requested"]["fused_residual_gate_add"], False)
        for name in ("chunkwise_flash_attention", "framewise_conditioning",
                     "fused_norm_modulation", "reuse_rope_frequencies"):
            self.assertIs(getattr(config, name), True)
        self.write_config(
            "schema_version = 1\nchunkwise_flash_attention = false\n"
            "framewise_conditioning = false\nfused_norm_modulation = false\n"
            "reuse_rope_frequencies = false\nfused_residual_gate_add = true\n"
        )
        config = load_optimization_config(self.path)
        self.assertIs(config.fused_residual_gate_add, True)
        for name in ("chunkwise_flash_attention", "framewise_conditioning",
                     "fused_norm_modulation", "reuse_rope_frequencies"):
            self.assertIs(getattr(config, name), False)

    def test_rope_reuse_is_independent_and_serialized(self):
        self.assertIs(OptimizationConfig().reuse_rope_frequencies, True)
        self.write_config("schema_version = 1\nreuse_rope_frequencies = false\n")
        config = load_optimization_config(self.path)
        self.assertIs(config.reuse_rope_frequencies, False)
        self.assertIs(config.chunkwise_flash_attention, True)
        self.assertIs(config.framewise_conditioning, True)
        self.assertIs(config.fused_norm_modulation, True)
        self.assertIs(config.snapshot()["requested"]["reuse_rope_frequencies"], False)
        self.write_config(
            "schema_version = 1\nchunkwise_flash_attention = false\n"
            "framewise_conditioning = false\nfused_norm_modulation = false\n"
            "reuse_rope_frequencies = true\n"
        )
        config = load_optimization_config(self.path)
        self.assertIs(config.reuse_rope_frequencies, True)
        self.assertIs(config.chunkwise_flash_attention, False)
        self.assertIs(config.framewise_conditioning, False)
        self.assertIs(config.fused_norm_modulation, False)

    def test_fused_norm_is_independent_and_serialized(self):
        self.assertIs(OptimizationConfig().fused_norm_modulation, True)
        self.write_config(
            "schema_version = 1\n"
            "fused_norm_modulation = false\n"
        )
        config = load_optimization_config(self.path)
        self.assertIs(config.fused_norm_modulation, False)
        self.assertIs(config.framewise_conditioning, True)
        self.assertIs(config.chunkwise_flash_attention, True)
        self.assertIs(config.snapshot()["requested"]["fused_norm_modulation"], False)
        self.write_config(
            "schema_version = 1\nchunkwise_flash_attention = false\n"
            "framewise_conditioning = false\nfused_norm_modulation = true\n"
        )
        config = load_optimization_config(self.path)
        self.assertIs(config.fused_norm_modulation, True)
        self.assertIs(config.framewise_conditioning, False)
        self.assertIs(config.chunkwise_flash_attention, False)

    def test_framewise_conditioning_is_independent_and_enabled_by_default(self):
        # File and Python defaults agree; an explicit false selects the baseline.
        self.write_config("schema_version = 1\nchunkwise_flash_attention = true\n")
        self.assertIs(load_optimization_config(self.path).framewise_conditioning, True)
        self.assertIs(OptimizationConfig().framewise_conditioning, True)
        self.write_config(
            "schema_version = 1\n"
            "chunkwise_flash_attention = false\n"
            "framewise_conditioning = true\n"
        )
        config = load_optimization_config(self.path)
        self.assertIs(config.chunkwise_flash_attention, False)
        self.assertIs(config.framewise_conditioning, True)
        self.assertIs(config.snapshot()["requested"]["framewise_conditioning"], True)
        self.write_config("schema_version = 1\nframewise_conditioning = false\n")
        self.assertIs(load_optimization_config(self.path).framewise_conditioning, False)
        self.assertIs(load_optimization_config(self.path).chunkwise_flash_attention, True)

    def test_unknown_switch_fails_instead_of_silently_ignoring_a_typo(self):
        for unsupported in ("chunkwise_flash_atention", "resident_kv_cache"):
            with self.subTest(unsupported=unsupported):
                self.write_config(f"schema_version = 1\n{unsupported} = true\n")
                with self.assertRaises(ValueError):
                    load_optimization_config(self.path)

    def test_missing_and_unsupported_schema_versions_fail(self):
        for text in (
            "chunkwise_flash_attention = true\n",
            "schema_version = 2\n",
            "schema_version = true\n",
            'schema_version = "1"\n',
        ):
            with self.subTest(text=text):
                self.write_config(text)
                with self.assertRaises(ValueError):
                    load_optimization_config(self.path)

    def test_strings_and_numbers_are_not_boolean_switches(self):
        # bool("false") and bool(1) would both silently turn the switch on.
        for name in ("chunkwise_flash_attention", "framewise_conditioning", "fused_norm_modulation", "reuse_rope_frequencies", "fused_residual_gate_add", "fused_rope", "fused_bias_swiglu", "fused_k_concat_cast"):
            for value in ('"false"', "0", "1", "[]"):
                with self.subTest(name=name, value=value):
                    self.write_config(f"schema_version = 1\n{name} = {value}\n")
                    with self.assertRaises(TypeError):
                        load_optimization_config(self.path)

    def test_python_config_also_rejects_non_boolean_switches(self):
        for name in ("chunkwise_flash_attention", "framewise_conditioning", "fused_norm_modulation", "reuse_rope_frequencies", "fused_residual_gate_add", "fused_rope", "fused_bias_swiglu", "fused_k_concat_cast"):
            for value in ("false", 0, 1, None):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(TypeError):
                        OptimizationConfig(**{name: value})

    def test_duplicate_key_and_missing_file_fail(self):
        self.write_config(
            "schema_version = 1\n"
            "chunkwise_flash_attention = true\n"
            "chunkwise_flash_attention = false\n"
        )
        with self.assertRaises(ValueError):
            load_optimization_config(self.path)
        with self.assertRaises(FileNotFoundError):
            load_optimization_config(self.path.with_name("missing.toml"))

    def test_snapshot_is_immutable_and_serialization_is_detached(self):
        config = OptimizationConfig(chunkwise_flash_attention=False)
        with self.assertRaises(FrozenInstanceError):
            config.chunkwise_flash_attention = True
        serialized = config.as_dict()
        serialized["chunkwise_flash_attention"] = True
        self.assertIs(config.chunkwise_flash_attention, False)
        snapshot = config.snapshot()
        self.assertEqual(snapshot, {"source_path": None, "requested": config.as_dict()})
        snapshot["requested"]["chunkwise_flash_attention"] = True
        self.assertIs(config.chunkwise_flash_attention, False)

    def test_default_file_does_not_depend_on_working_directory(self):
        expected = load_optimization_config(DEFAULT_OPTIMIZATION_CONFIG)
        old_directory = Path.cwd()
        try:
            # A local file with the same name must not shadow the repository's
            # default when inference is launched from another directory.
            local = Path(self.directory.name) / Path(DEFAULT_OPTIMIZATION_CONFIG).name
            local.write_text("schema_version = 999\n", encoding="utf-8")
            os.chdir(self.directory.name)
            actual = load_optimization_config()
        finally:
            os.chdir(old_directory)
        self.assertEqual(actual.as_dict(), expected.as_dict())
        self.assertEqual(actual.source_path, expected.source_path)

    def test_editing_file_affects_new_models_but_not_running_model(self):
        self.write_config("schema_version = 1\nchunkwise_flash_attention = false\n")
        existing = small_model(optimization_config=self.path)
        self.write_config("schema_version = 1\nchunkwise_flash_attention = true\n")
        following = small_model(optimization_config=self.path)
        self.assertIs(existing.optimization_config.chunkwise_flash_attention, False)
        self.assertIs(following.optimization_config.chunkwise_flash_attention, True)
        self.assertIs(load_optimization_config(self.path).chunkwise_flash_attention, True)


class OptimizationConfigIntegrationTests(unittest.TestCase):
    def test_denoiser_config_forwards_explicit_optimization_config(self):
        config = OptimizationConfig(
            chunkwise_flash_attention=False, framewise_conditioning=True, fused_norm_modulation=False,
        )

        def construct(**kwargs):
            return small_model(optimization_config=kwargs["optimization_config"])

        factory = mock.Mock(side_effect=construct)
        with mock.patch.dict(mw.MiniWorldModels, {"optimization-config-test": factory}):
            denoiser = Denoiser(DenoiserConfig(
                wm_model="optimization-config-test",
                latent_size=(2, 3),
                latent_channels=2,
                latent_frames=8,
                optimization_config=config,
            ))
        self.assertIs(factory.call_args.kwargs["optimization_config"], config)
        self.assertIs(denoiser.net.optimization_config.chunkwise_flash_attention, False)
        self.assertIs(denoiser.net.optimization_config.framewise_conditioning, True)
        self.assertIs(denoiser.net.optimization_config.fused_norm_modulation, False)

    def test_per_call_override_rejects_ambiguous_values(self):
        model = small_model(optimization_config=OptimizationConfig(False))
        x = torch.randn(1, 2, 1, 2, 3)
        t = torch.rand(1, 1)
        for override in ("false", 0, 1):
            with self.subTest(override=override):
                with self.assertRaises(TypeError):
                    model.forward_with_cache(x, t, use_chunk_flash_attn=override)

    def test_switch_is_not_checkpoint_state_and_cpu_fallback_preserves_outputs(self):
        enabled = small_model(optimization_config=OptimizationConfig(True))
        disabled = small_model(optimization_config=OptimizationConfig(False))
        self.assertEqual(set(enabled.state_dict()), set(disabled.state_dict()))
        self.assertFalse(any("optimization" in name for name in enabled.state_dict()))
        loaded = disabled.load_state_dict(enabled.state_dict(), strict=True)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        self.assertIs(disabled.optimization_config.chunkwise_flash_attention, False)

        x = torch.randn(1, 2, 3, 2, 3)
        t = torch.rand(1, 3)
        y = torch.randn(1, 3, 3)
        with mock.patch.object(
            mw, "_chunkwise_flash_attention", side_effect=AssertionError("CPU used FlashAttention")
        ):
            expected, _ = disabled.forward_with_cache(x, t, y, chunk_size=2)
            actual, _ = enabled.forward_with_cache(x, t, y, chunk_size=2)
        self.assertGreater(expected.std().item(), 0.01)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    "FlashAttention requires a CUDA GPU with compute capability >= 8.0",
)
class OptimizationConfigCudaTests(unittest.TestCase):
    def test_model_local_defaults_and_per_call_overrides_dispatch_correctly(self):
        enabled = small_model("cuda", optimization_config=OptimizationConfig(True))
        disabled = small_model("cuda", optimization_config=OptimizationConfig(False))
        disabled.load_state_dict(enabled.state_dict(), strict=True)
        x = torch.randn(1, 2, 3, 2, 3, device="cuda")
        t = torch.rand(1, 3, device="cuda")
        y = torch.randn(1, 3, 3, device="cuda")
        seed = torch.randn(1, 2, 1, 2, 3, device="cuda")
        seed_t = torch.rand(1, 1, device="cuda")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, past = disabled.forward_with_cache(seed, seed_t, chunk_size=2, return_kv=True)
            kwargs = dict(
                y=y, past_kv_list=past, current_position_offset=1,
                chunk_size=2, return_kv=True,
            )
            expected, expected_cache = disabled.forward_with_cache(x, t, **kwargs)
            # Interleave models: a process-global flag would make these cases
            # interfere. Overrides must also leave the stored config intact.
            cases = [
                (enabled, None, True),
                (disabled, None, False),
                (disabled, True, True),
                (enabled, False, False),
                (enabled, None, True),
                (disabled, None, False),
            ]
            for model, override, expect_flash in cases:
                with self.subTest(config=model.optimization_config.as_dict(), override=override):
                    call_kwargs = dict(kwargs)
                    if override is not None:
                        call_kwargs["use_chunk_flash_attn"] = override
                    with mock.patch.object(
                        mw, "_chunkwise_flash_attention", wraps=mw._chunkwise_flash_attention
                    ) as flash, mock.patch.object(
                        mw, "_build_cached_block_causal_mask", wraps=mw._build_cached_block_causal_mask
                    ) as dense_mask:
                        actual, actual_cache = model.forward_with_cache(x, t, **call_kwargs)
                    self.assertEqual(flash.call_count, model.depth if expect_flash else 0)
                    if expect_flash:
                        dense_mask.assert_not_called()
                    else:
                        self.assertGreater(dense_mask.call_count, 0)
                    self.assertGreater(actual.std().item(), 0.01)
                    tolerance = 0.025 if expect_flash else 0
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                    for actual_kv, expected_kv in zip(actual_cache, expected_cache):
                        for actual_tensor, expected_tensor in zip(actual_kv, expected_kv):
                            torch.testing.assert_close(
                                actual_tensor, expected_tensor, rtol=tolerance, atol=tolerance
                            )
        self.assertIs(enabled.optimization_config.chunkwise_flash_attention, True)
        self.assertIs(disabled.optimization_config.chunkwise_flash_attention, False)


if __name__ == "__main__":
    unittest.main()
