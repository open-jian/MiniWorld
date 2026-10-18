"""Regression checks for compact, inference-only conditioning projections.

Run with ``python -m unittest discover -s tests -v`` from the repository root.
The projections, gates, and output weights are nonzero: identity blocks or an
all-zero output cannot make these parity and spatial-sensitivity checks pass.
"""

from contextlib import contextmanager
import unittest

import torch

from miniworld.miniworld import MiniWorldModel
from miniworld.optimization_config import OptimizationConfig


MODES = ("adaln_lora", "fully_shared", "per_block")
SPATIAL_TOKENS = 6


def conditioning_model(enabled, *, mode="adaln_lora", pose=False, device="cpu"):
    torch.manual_seed(510)
    model = MiniWorldModel(
        in_channels=2,
        hidden_size=64,
        cond_dim=3,
        depth=2,
        num_heads=2,
        patch_size=1,
        input_size=(2, 3),
        num_frames=5,
        mlp_ratio=2,
        adaln_mode=mode,
        adaln_lora_dim=8,
        cond_per_token=pose,
        optimization_config=OptimizationConfig(
            chunkwise_flash_attention=True,
            framewise_conditioning=enabled,
            fused_norm_modulation=False,
            fused_residual_gate_add=False,
        ),
    )
    with torch.no_grad():
        model.shared_mod[-1].weight.normal_(std=0.025)
        model.shared_mod[-1].bias.normal_(std=0.05)
        for block in model.blocks:
            if mode == "adaln_lora":
                block.modulation.lora[-1].weight.normal_(std=0.025)
            elif mode == "per_block":
                block.modulation.full[-1].weight.normal_(std=0.025)
                block.modulation.full[-1].bias.normal_(std=0.05)
        if model.action_encoder is not None:
            model.null_action.normal_(std=0.15)
            model.action_encoder.to_mod[-1].weight.normal_(std=0.025)
            model.action_encoder.to_mod[-1].bias.normal_(std=0.05)
        else:
            model.pose_encoder.to_mod.weight.normal_(std=0.025)
            model.pose_encoder.to_mod.bias.normal_(std=0.05)
        model.final_layer.adaLN_modulation[-1].weight.normal_(std=0.025)
        model.final_layer.adaLN_modulation[-1].bias.normal_(std=0.05)
        model.final_layer.linear.weight.normal_(std=0.075)
        model.final_layer.linear.bias.normal_(std=0.05)
    return model.to(device).eval()


def model_pair(**kwargs):
    reference = conditioning_model(False, **kwargs)
    compact = conditioning_model(True, **kwargs)
    compact.load_state_dict(reference.state_dict(), strict=True)
    return reference, compact


def inputs(frames=5, *, pose=False, device="cpu", batch=2):
    x = torch.randn(batch, 2, frames, 2, 3, device=device)
    t = torch.rand(batch, frames, device=device)
    shape = (batch, frames, 3, 2, 3) if pose else (batch, frames, 3)
    y = torch.randn(*shape, device=device)
    return x, t, y


@contextmanager
def projection_rows(model):
    """Observe actual linear inputs, rather than trusting intermediate labels."""
    seen = {}
    modules = {"final": model.final_layer.adaLN_modulation[-1]}
    if model.adaln_mode != "per_block":
        modules["shared"] = model.shared_mod[-1]
    for index, block in enumerate(model.blocks):
        if model.adaln_mode == "adaln_lora":
            modules[f"block_{index}_down"] = block.modulation.lora[1]
            modules[f"block_{index}_up"] = block.modulation.lora[2]
        elif model.adaln_mode == "per_block":
            modules[f"block_{index}"] = block.modulation.full[-1]
    if model.pose_encoder is not None:
        modules["pose"] = model.pose_encoder.to_mod
    handles = []
    for name, module in modules.items():
        def record(_module, args, key=name):
            seen.setdefault(key, []).append(tuple(args[0].shape))
        handles.append(module.register_forward_pre_hook(record))
    try:
        yield seen
    finally:
        for handle in handles:
            handle.remove()


class FramewiseConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_output_and_cache_close(self, actual, expected, *, rtol=2e-5, atol=3e-6):
        output, cache = actual
        reference_output, reference_cache = expected
        self.assertGreater(reference_output.std().item(), 0.01)
        torch.testing.assert_close(output, reference_output, rtol=rtol, atol=atol)
        if reference_cache is None:
            self.assertIsNone(cache)
        else:
            self.assertEqual(len(cache), len(reference_cache))
            for actual_pair, expected_pair in zip(cache, reference_cache):
                for actual_tensor, expected_tensor in zip(actual_pair, expected_pair):
                    torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol)

    def assert_rows(self, seen, *, batch, frames, compact, pose=False):
        self.assertIn("final", seen)
        for name, shapes in seen.items():
            rows = frames if compact and name != "pose" else frames * SPATIAL_TOKENS
            self.assertEqual(len(shapes), 1, (name, shapes))
            self.assertEqual(shapes[0][:-1], (batch, rows), (name, shapes))
        if pose:
            self.assertIn("pose", seen)

    def test_all_adaln_modes_actions_cached_offset_and_partial_chunk(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                reference, compact = model_pair(mode=mode)
                seed_x, seed_t, seed_y = inputs(frames=1)
                seed = reference.forward_with_cache(seed_x, seed_t, seed_y, return_kv=True)
                # A shared, nonempty history isolates the changed conditioning
                # path. Five frames also exercise a partial final size-2 chunk.
                x, t, y = inputs(frames=5)
                kwargs = dict(
                    past_kv_list=seed[1], current_position_offset=1,
                    return_kv=True, chunk_size=2,
                    cond_drop=torch.tensor([False, True]),
                )
                expected = reference.forward_with_cache(x, t, y, **kwargs)
                actual = compact.forward_with_cache(x, t, y, **kwargs)
                self.assert_output_and_cache_close(actual, expected)

    def test_compact_rows_reach_every_projection_in_all_modes(self):
        for mode in MODES:
            for pose in (False, True):
                for enabled in (False, True):
                    with self.subTest(mode=mode, pose=pose, enabled=enabled):
                        model = conditioning_model(enabled, mode=mode, pose=pose)
                        x, t, y = inputs(frames=3, pose=pose)
                        with projection_rows(model) as seen:
                            model.forward_with_cache(x, t, y, chunk_size=2)
                        self.assert_rows(seen, batch=2, frames=3, compact=enabled, pose=pose)

    def test_pose_remains_spatial_and_cfg_drop_removes_only_dropped_sample(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                reference, compact = model_pair(mode=mode, pose=True)
                x, t, y = inputs(frames=3, pose=True)
                kwargs = dict(chunk_size=2, cond_drop=torch.tensor([False, True]))
                expected = reference.forward_with_cache(x, t, y, **kwargs)
                actual = compact.forward_with_cache(x, t, y, **kwargs)
                self.assert_output_and_cache_close(actual, expected)
                changed_pose = y.clone()
                changed_pose[..., 0, 0] += 2.0
                changed, _ = compact.forward_with_cache(x, t, changed_pose, **kwargs)
                self.assertGreater((changed[0] - actual[0][0]).abs().max().item(), 1e-4)
                torch.testing.assert_close(changed[1], actual[0][1], rtol=0, atol=0)

    def test_scalar_t_null_action_broadcast_action_and_first_frame_semantics(self):
        for mode in MODES:
            reference, compact = model_pair(mode=mode)
            x, t, y = inputs(frames=3)
            for action in (None, y[:, :1], y):
                for offset in (0, 4):
                    with self.subTest(mode=mode, action_shape=None if action is None else action.shape, offset=offset):
                        kwargs = dict(current_position_offset=offset, chunk_size=2)
                        expected = reference.forward_with_cache(x, t[:, 0], action, **kwargs)
                        actual = compact.forward_with_cache(x, t[:, 0], action, **kwargs)
                        self.assert_output_and_cache_close(actual, expected)
            # Global frame zero uses null_action; a later window uses y[:, 0].
            changed = y.clone()
            changed[:, 0] += 2.0
            first, _ = compact.forward_with_cache(x, t, y, current_position_offset=0)
            first_changed, _ = compact.forward_with_cache(x, t, changed, current_position_offset=0)
            torch.testing.assert_close(first, first_changed, rtol=0, atol=0)
            later, _ = compact.forward_with_cache(x, t, y, current_position_offset=4)
            later_changed, _ = compact.forward_with_cache(x, t, changed, current_position_offset=4)
            self.assertGreater((later - later_changed).abs().max().item(), 1e-4)

    def test_per_token_timesteps_fall_back_without_collapsing_spatial_values(self):
        for mode in MODES:
            for pose in (False, True):
                with self.subTest(mode=mode, pose=pose):
                    reference, compact = model_pair(mode=mode, pose=pose)
                    x, t, y = inputs(frames=3, pose=pose)
                    repeated_t = t.repeat_interleave(SPATIAL_TOKENS, dim=1)
                    frame_result = reference.forward_with_cache(x, t, y)
                    token_result = reference.forward_with_cache(x, repeated_t, y)
                    self.assert_output_and_cache_close(token_result, frame_result)
                    spatial_t = repeated_t.clone()
                    spatial_t[:, 1::SPATIAL_TOKENS] += 7.0
                    expected = reference.forward_with_cache(x, spatial_t, y)
                    with projection_rows(compact) as seen:
                        actual = compact.forward_with_cache(x, spatial_t, y)
                    self.assert_rows(seen, batch=2, frames=3, compact=False, pose=pose)
                    self.assert_output_and_cache_close(actual, expected, rtol=0, atol=0)
                    self.assertGreater((actual[0] - token_result[0]).abs().max().item(), 1e-4)

    def test_regular_forward_training_and_gradients_keep_original_path(self):
        reference, compact = model_pair()
        reference.train()
        compact.train()
        x, t, y = inputs(frames=5)
        kwargs = dict(temporal_causal=True, chunk_size=2, cond_drop=torch.tensor([False, True]))
        expected = reference(x, t, y, **kwargs)
        with projection_rows(compact) as seen:
            actual = compact(x, t, y, **kwargs)
        self.assert_rows(seen, batch=2, frames=5, compact=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for (name, param), (_, other) in zip(reference.named_parameters(), compact.named_parameters()):
            if param.grad is None:
                self.assertIsNone(other.grad, name)
            else:
                torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0, msg=name)
        with projection_rows(compact) as seen:
            cached = compact.forward_with_cache(x, t, y, chunk_size=2)
        self.assert_rows(seen, batch=2, frames=5, compact=False)
        self.assert_output_and_cache_close(cached, reference.forward_with_cache(x, t, y, chunk_size=2), rtol=0, atol=0)
        compact.eval()
        with projection_rows(compact) as seen:
            compact(x, t, y, temporal_causal=True)
        self.assert_rows(seen, batch=2, frames=5, compact=False)

    def test_optimization_preserves_checkpoint_keys_and_strict_loading(self):
        for mode in MODES:
            for pose in (False, True):
                with self.subTest(mode=mode, pose=pose):
                    reference, compact = model_pair(mode=mode, pose=pose)
                    self.assertEqual(set(reference.state_dict()), set(compact.state_dict()))
                    self.assertFalse(any("conditioning_tokens" in key for key in compact.state_dict()))
                    result = compact.load_state_dict(reference.state_dict(), strict=True)
                    self.assertEqual(result.missing_keys, [])
                    self.assertEqual(result.unexpected_keys, [])


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU")
class FramewiseConditioningCudaTests(unittest.TestCase):
    assert_output_and_cache_close = FramewiseConditioningTests.assert_output_and_cache_close

    def test_bfloat16_nonzero_model_outputs_and_cache(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                reference, compact = model_pair(mode=mode, device="cuda")
                x, t, y = inputs(frames=3, device="cuda")
                kwargs = dict(return_kv=True, chunk_size=2, current_position_offset=2)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = reference.forward_with_cache(x, t, y, **kwargs)
                    actual = compact.forward_with_cache(x, t, y, **kwargs)
                self.assertTrue(torch.isfinite(actual[0]).all())
                # Different GEMM row counts can change BF16 rounding. Tight
                # absolute bounds catch semantic/layout errors; real-checkpoint
                # quality and speed measurements are separate from this test.
                self.assert_output_and_cache_close(actual, expected, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
