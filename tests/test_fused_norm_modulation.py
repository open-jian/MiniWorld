"""Numerical and dispatch checks for inference RMSNorm/AdaLN fusion.

These checks exercise the actual channel-major residual layout, nonzero norm
weights and modulation, and the BF16 rounding stages in the reference graph.
End-to-end video quality and latency are measured separately on real episodes.
"""

from dataclasses import replace
import unittest
from unittest import mock

import torch

from miniworld import miniworld as mw
from miniworld.fused_norm_modulation import (
    can_fuse_norm_modulation,
    fused_norm_modulation,
)
from test_framewise_conditioning import conditioning_model, inputs


def reference_norm_modulation(x, weight, shift, scale, *, eps=1e-6, tokens_per_frame=None):
    """Use the unfused model's arithmetic, including its intermediate casts."""
    norm = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    return mw.modulate(norm * weight, shift, scale, tokens_per_frame)


def channel_major(batch, rows, channels, *, device="cuda", dtype=torch.bfloat16):
    return torch.randn(batch, channels, rows, device=device, dtype=dtype).transpose(1, 2)


def optimization_pair(*, device="cpu", framewise=True, mode="adaln_lora", pose=False):
    reference = conditioning_model(framewise, mode=mode, pose=pose, device=device)
    fused = conditioning_model(framewise, mode=mode, pose=pose, device=device)
    reference.optimization_config = replace(reference.optimization_config, fused_norm_modulation=False)
    fused.optimization_config = replace(fused.optimization_config, fused_norm_modulation=True)
    # Unit weights can hide an incorrectly moved BF16 cast around RMSNorm.
    with torch.no_grad():
        for module in reference.modules():
            if isinstance(module, mw.RMSNorm):
                module.weight.uniform_(0.75, 1.25)
    fused.load_state_dict(reference.state_dict(), strict=True)
    return reference, fused


class FusedNormModulationFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cpu_is_ineligible_even_with_matching_layout_and_dtype(self):
        x = channel_major(2, 15, 65, device="cpu")
        weight = torch.rand(65)
        scale = torch.rand(2, 3, 1, 65, dtype=torch.bfloat16)
        with torch.no_grad():
            self.assertFalse(can_fuse_norm_modulation(x, weight, None, scale, 5))
            with self.assertRaises(ValueError):
                fused_norm_modulation(x, weight, None, scale, conditioning_tokens_per_frame=5)

    def test_cpu_cached_forward_and_checkpoint_keys_keep_reference_behavior(self):
        reference, fused = optimization_pair()
        x, t, y = inputs(frames=3)
        kwargs = dict(return_kv=True, chunk_size=2)
        expected = reference.forward_with_cache(x, t, y, **kwargs)
        actual = fused.forward_with_cache(x, t, y, **kwargs)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        self.assertGreater(actual[0].std().item(), 0.01)
        for actual_pair, expected_pair in zip(actual[1], expected[1]):
            for a, b in zip(actual_pair, expected_pair):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(set(reference.state_dict()), set(fused.state_dict()))

    def test_regular_training_forward_and_gradients_are_unchanged(self):
        reference, fused = optimization_pair()
        reference.train()
        fused.train()
        x, t, y = inputs(frames=5)
        kwargs = dict(temporal_causal=True, chunk_size=2, cond_drop=torch.tensor([False, True]))
        expected = reference(x, t, y, **kwargs)
        actual = fused(x, t, y, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for (name, param), (_, other) in zip(reference.named_parameters(), fused.named_parameters()):
            if param.grad is None:
                self.assertIsNone(other.grad, name)
            else:
                torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0, msg=name)


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU")
class FusedNormModulationCudaTests(unittest.TestCase):
    def assert_numerically_close(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.isfinite(actual).all().item())
        error = (actual - expected).float()
        relative_rmse = error.square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-12)
        # FP32 reduction order can very rarely cross a BF16 midpoint. A global
        # error bound plus a separate maximum catches layout/rounding mistakes
        # without incorrectly requiring a bitwise-identical reduction tree.
        self.assertLess(relative_rmse.item(), 3e-4)
        self.assertLess(error.abs().max().item(), 0.04)

    @torch.no_grad()
    def test_real_strides_compact_dense_and_broadcast_modulation(self):
        torch.manual_seed(4101)
        for batch, frames, spatial, channels in ((2, 3, 5, 65), (1, 1, 17, 64), (1, 2, 300, 1536)):
            rows = frames * spatial
            x = channel_major(batch, rows, channels)
            weight = torch.randn(channels, device="cuda") * 0.2 + 0.9
            original_x = x.clone()
            for dtype in (torch.bfloat16, torch.float32):
                # Real modulation comes from a chunk of a 6D projection, so
                # its row stride is larger than D even in the compact path.
                projected = torch.randn(batch, frames, 6 * channels, device="cuda", dtype=dtype) * 0.1
                compact_shift, compact_scale = [p.unsqueeze(2) for p in projected.chunk(6, -1)[:2]]
                dense_shift = compact_shift.expand(-1, -1, spatial, -1).reshape(batch, rows, channels)
                dense_scale = compact_scale.expand(-1, -1, spatial, -1).reshape(batch, rows, channels)
                global_shift = compact_shift[:, 0, 0]
                global_scale = compact_scale[:, 0, 0]
                cases = (
                    ("compact", compact_shift, compact_scale, spatial),
                    ("compact_no_shift", None, compact_scale, spatial),
                    ("dense", dense_shift, dense_scale, None),
                    ("global", global_shift, global_scale, None),
                    ("singleton_row", global_shift[:, None], global_scale[:, None], None),
                    ("broadcast_batch", dense_shift[:1], dense_scale[:1], None),
                    ("broadcast_zero_stride", dense_shift[:1, :1].expand(batch, rows, -1),
                     dense_scale[:1, :1].expand(batch, rows, -1), None),
                )
                for label, shift, scale, tokens in cases:
                    with self.subTest(shape=tuple(x.shape), dtype=dtype, case=label):
                        self.assertTrue(can_fuse_norm_modulation(x, weight, shift, scale, tokens))
                        expected = reference_norm_modulation(x, weight, shift, scale, tokens_per_frame=tokens)
                        actual = fused_norm_modulation(x, weight, shift, scale, conditioning_tokens_per_frame=tokens)
                        self.assertEqual(actual.stride(), x.stride())
                        self.assert_numerically_close(actual, expected)
            torch.testing.assert_close(x, original_x, rtol=0, atol=0)

    @torch.no_grad()
    def test_storage_offset_and_custom_epsilon_are_respected(self):
        torch.manual_seed(4102)
        storage = torch.randn(2 * 17 * 64 + 7, device="cuda", dtype=torch.bfloat16)
        x = storage[7:].view(2, 64, 17).transpose(1, 2)
        x.mul_(1e-4)
        self.assertEqual(x.storage_offset(), 7)
        weight = torch.linspace(0.7, 1.3, 64, device="cuda")
        scale = torch.full((2, 1, 64), 0.2, device="cuda", dtype=torch.bfloat16)
        outputs = []
        for epsilon in (1e-6, 1e-3):
            expected = reference_norm_modulation(x, weight, None, scale, eps=epsilon)
            actual = fused_norm_modulation(x, weight, None, scale, eps=epsilon)
            self.assert_numerically_close(actual, expected)
            outputs.append(actual)
        self.assertGreater((outputs[0] - outputs[1]).abs().max().item(), 0.1)

    @torch.no_grad()
    def test_zero_epsilon_keeps_the_supported_eager_model_path(self):
        x = channel_major(1, 17, 64)
        norm = mw.RMSNorm(64, eps=0.0).cuda()
        scale = torch.full((1, 17, 64), 0.2, device="cuda", dtype=torch.bfloat16)
        expected = mw.modulate(norm(x), None, scale)
        with mock.patch.object(mw, "fused_norm_modulation", side_effect=AssertionError("unsupported epsilon")):
            actual = mw._norm_modulate(norm, x, None, scale, use_fused=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.no_grad()
    def test_bfloat16_norm_and_scale_rounding_happen_before_fp32_products(self):
        # Repeated values keep the reduction away from a BF16 midpoint, while
        # 1 + scale deliberately rounds in BF16. An all-FP32 rewrite must fail.
        pattern = torch.tensor([0.5, 1.0, 1.5, 2.0], device="cuda", dtype=torch.bfloat16)
        x = pattern.repeat(16).view(1, 64, 1).expand(1, 64, 3).contiguous().transpose(1, 2)
        weight = torch.linspace(0.73, 1.19, 64, device="cuda")
        scale = torch.full((1, 3, 64), 0.005859375, device="cuda", dtype=torch.bfloat16)
        shift = torch.full_like(scale, 0.03125)
        expected = reference_norm_modulation(x, weight, shift, scale)
        actual = fused_norm_modulation(x, weight, shift, scale)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        norm_fp32 = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        wrong_norm_rounding = norm_fp32 * weight * (1 + scale) + shift
        wrong_scale_rounding = norm_fp32.to(torch.bfloat16).float() * weight * (1 + scale.float()) + shift
        self.assertGreater((actual - wrong_norm_rounding).abs().max().item(), 1e-3)
        self.assertGreater((actual - wrong_scale_rounding).abs().max().item(), 1e-3)

    @torch.no_grad()
    def test_fp32_modulation_product_rounds_before_shift_addition(self):
        # The profiled Blackwell packed FP32 path contracted this pair despite
        # enable_fp_fusion=False. Exact values expose the one-ULP difference.
        x = torch.ones(1, 64, 3, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        weighted = (torch.tensor(0.466796875) * torch.tensor(1.2808101177215576)).item()
        weight = torch.full((64,), weighted, device="cuda")
        scale = torch.full((1, 3, 64), 0.0634765625, device="cuda", dtype=torch.bfloat16)
        shift = torch.full_like(scale, -0.25390625)
        expected = reference_norm_modulation(x, weight, shift, scale)
        actual = fused_norm_modulation(x, weight, shift, scale)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # FP64 can represent this FP32 product and sum exactly before the final
        # cast, so it models the undesired single-rounding multiply-add here.
        contracted = (weight.double() * (1 + scale).double() + shift.double()).float()
        self.assertTrue((expected != contracted).all().item())

    @torch.no_grad()
    def test_nondefault_stream_preserves_input_dependencies(self):
        torch.manual_seed(4103)
        x = channel_major(1, 17, 64)
        weight = torch.linspace(0.7, 1.3, 64, device=x.device)
        scale = torch.zeros(1, 1, 64, device=x.device, dtype=torch.bfloat16)
        # Compile before creating the dependency; compilation must not make a
        # mistakenly launched default-stream kernel appear synchronized.
        fused_norm_modulation(x, weight, None, scale)
        torch.cuda.synchronize(x.device)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(stream):
            torch.cuda._sleep(5_000_000)
            x[:, 0, 0].add_(2)
            expected = reference_norm_modulation(x, weight, None, scale)
            actual = fused_norm_modulation(x, weight, None, scale)
            copied = actual.clone()
        stream.synchronize()
        self.assert_numerically_close(copied, expected)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two visible CUDA GPUs")
    @torch.no_grad()
    def test_tensor_device_is_used_and_callers_current_device_is_restored(self):
        previous = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            x = channel_major(2, 17, 64, device="cuda:1")
            weight = torch.linspace(0.7, 1.3, 64, device=x.device)
            scale = torch.zeros(2, 1, 64, device=x.device, dtype=torch.bfloat16)
            self.assertEqual(torch.cuda.current_device(), 0)
            expected = reference_norm_modulation(x, weight, None, scale)
            actual = fused_norm_modulation(x, weight, None, scale)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(actual.device, x.device)
            torch.cuda.synchronize(x.device)
            self.assert_numerically_close(actual, expected)
        finally:
            torch.cuda.set_device(previous)

    def test_unsupported_dtypes_layout_and_gradient_mode_are_ineligible(self):
        x = channel_major(2, 15, 65)
        weight = torch.ones(65, device="cuda")
        scale = torch.zeros(2, 15, 65, device="cuda", dtype=torch.bfloat16)
        with torch.enable_grad():
            self.assertFalse(can_fuse_norm_modulation(x, weight, None, scale))
        with torch.no_grad():
            self.assertTrue(can_fuse_norm_modulation(x, weight, None, scale))
            cases = (
                (x.contiguous(), weight, scale),
                (x.float(), weight, scale),
                (x.half(), weight, scale),
                (x, weight.bfloat16(), scale),
                (x, weight, scale.half()),
            )
            for residual, norm_weight, modulation in cases:
                with self.subTest(residual_dtype=residual.dtype, stride=residual.stride(),
                                  weight_dtype=norm_weight.dtype, scale_dtype=modulation.dtype):
                    self.assertFalse(can_fuse_norm_modulation(residual, norm_weight, None, modulation))
                    with self.assertRaises(ValueError):
                        fused_norm_modulation(residual, norm_weight, None, modulation)

    @torch.no_grad()
    def test_cached_model_outputs_cache_and_actual_kernel_dispatch(self):
        for framewise in (False, True):
            for mode, pose in (("adaln_lora", False), ("fully_shared", False), ("per_block", False),
                               ("adaln_lora", True)):
                with self.subTest(framewise=framewise, mode=mode, pose=pose):
                    reference, fused = optimization_pair(device="cuda", framewise=framewise, mode=mode, pose=pose)
                    seed = inputs(frames=1, pose=pose, device="cuda")
                    x, t, y = inputs(frames=5, pose=pose, device="cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, history = reference.forward_with_cache(*seed, return_kv=True)
                        saved_history = [(k.clone(), v.clone()) for k, v in history]
                        kwargs = dict(past_kv_list=history, current_position_offset=1,
                                      return_kv=True, chunk_size=2,
                                      cond_drop=torch.tensor([False, True], device="cuda"))
                        expected = reference.forward_with_cache(x, t, y, **kwargs)
                        with mock.patch.object(mw, "fused_norm_modulation", wraps=fused_norm_modulation) as kernel:
                            actual = fused.forward_with_cache(x, t, y, **kwargs)
                        self.assertGreater(kernel.call_count, 0)
                    self.assertGreater(expected[0].std().item(), 0.01)
                    torch.testing.assert_close(actual[0], expected[0], rtol=0.01, atol=0.01)
                    for actual_pair, expected_pair in zip(actual[1], expected[1]):
                        for a, b in zip(actual_pair, expected_pair):
                            torch.testing.assert_close(a, b, rtol=0.01, atol=0.01)
                    for unchanged, saved in zip(history, saved_history):
                        for a, b in zip(unchanged, saved):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)

    @torch.no_grad()
    def test_training_cached_forward_and_fp32_inference_do_not_dispatch(self):
        reference, fused = optimization_pair(device="cuda")
        x, t, y = inputs(frames=3, device="cuda")
        for training in (False, True):
            reference.train(training)
            fused.train(training)
            with mock.patch.object(mw, "fused_norm_modulation", side_effect=AssertionError("unexpected fused kernel")):
                if training:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        expected = reference.forward_with_cache(x, t, y, chunk_size=2)
                        actual = fused.forward_with_cache(x, t, y, chunk_size=2)
                else:
                    expected = reference.forward_with_cache(x, t, y, chunk_size=2)
                    actual = fused.forward_with_cache(x, t, y, chunk_size=2)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
