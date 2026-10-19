"""Arithmetic and dispatch regressions for inference bias/SwiGLU fusion.

The reference preserves three BF16 boundaries: the cast of the FP32 bias,
the projected bias sum, and SiLU's result before the final multiplication.
Full MLP tests also check the separately rounded no-bias GEMM output.
"""

from dataclasses import replace
import os
import subprocess
import sys
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from miniworld import miniworld as mw
from miniworld import fused_bias_swiglu as swiglu_module
from miniworld.fused_bias_swiglu import can_fuse_bias_swiglu, fused_bias_swiglu
from miniworld.optimization_config import OptimizationConfig
from test_chunk_flash_attention import small_model
from test_framewise_conditioning import inputs


def channel_major(batch, tokens, channels, *, device="cuda", dtype=torch.float32):
    return torch.randn(batch, channels, tokens, device=device, dtype=dtype).transpose(1, 2)


def reference_activation(projection, bias):
    biased = projection + bias.to(torch.bfloat16)
    first, second = biased.chunk(2, dim=-1)
    return F.silu(first) * second


def assert_bits(test, actual, expected, *, layout=False):
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.shape, expected.shape)
    if layout:
        test.assertEqual(actual.stride(), expected.stride())
    integer = torch.int32 if actual.dtype == torch.float32 else torch.int16
    test.assertTrue(torch.equal(actual.view(integer), expected.view(integer)))


def model_pair(*, device="cpu"):
    config = OptimizationConfig(fused_bias_swiglu=False)
    reference = small_model(device, head_dim=128, optimization_config=config)
    optimized = small_model(device, head_dim=128, optimization_config=replace(config, fused_bias_swiglu=True))
    optimized.load_state_dict(reference.state_dict(), strict=True)
    return reference, optimized


def assert_result_exact(test, actual, expected):
    assert_bits(test, actual[0], expected[0])
    test.assertGreater(expected[0].std().item(), 0.01)
    if expected[1] is None:
        test.assertIsNone(actual[1])
    else:
        test.assertEqual(len(actual[1]), len(expected[1]))
        for actual_pair, expected_pair in zip(actual[1], expected[1]):
            for actual_value, expected_value in zip(actual_pair, expected_pair):
                assert_bits(test, actual_value, expected_value)


class BiasSwiGLUFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @torch.no_grad()
    def test_cpu_fallback_and_checkpoint_keys_are_unchanged(self):
        projection = torch.randn(1, 17, 130, dtype=torch.bfloat16)
        bias = torch.nn.Parameter(torch.randn(130))
        self.assertFalse(can_fuse_bias_swiglu(projection, bias))
        with self.assertRaises(ValueError):
            fused_bias_swiglu(projection, bias)
        mlp = mw.SwiGLUFFN(64, 65).eval()
        value = channel_major(1, 17, 64, device="cpu")
        with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("CPU dispatch")):
            assert_bits(self, mlp(value, use_fused=True), mlp(value))
            reference, optimized = model_pair()
            values = inputs(frames=3, batch=1)
            assert_result_exact(self, optimized.forward_with_cache(*values, chunk_size=2, return_kv=True),
                                reference.forward_with_cache(*values, chunk_size=2, return_kv=True))
        self.assertEqual(set(reference.state_dict()), set(optimized.state_dict()))
        optimized.load_state_dict(reference.state_dict(), strict=True)

    def test_training_and_ordinary_forward_preserve_reference_gradients(self):
        reference, optimized = model_pair()
        reference.train()
        optimized.train()
        values = inputs(frames=8, batch=1)
        kwargs = dict(chunk_size=2, temporal_causal=True, cond_drop=torch.tensor([False]))
        with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Training dispatch")):
            expected = reference(*values, **kwargs)
            actual = optimized(*values, **kwargs)
        assert_bits(self, actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for (name, parameter), (_, peer) in zip(reference.named_parameters(), optimized.named_parameters()):
            if parameter.grad is None:
                self.assertIsNone(peer.grad, name)
            else:
                torch.testing.assert_close(peer.grad, parameter.grad, rtol=0, atol=0, msg=name)
        with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Training cached dispatch")):
            expected = reference.forward_with_cache(*values, chunk_size=2, return_kv=True)
            actual = optimized.forward_with_cache(*values, chunk_size=2, return_kv=True)
        assert_result_exact(self, actual, expected)
        self.assertFalse(actual[0].requires_grad)


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU")
class BiasSwiGLUCudaTests(unittest.TestCase):
    @torch.no_grad()
    def test_real_projection_shapes_tail_and_storage_offsets(self):
        torch.manual_seed(1850)
        for batch, tokens, hidden in ((2, 17, 65), (1, 600, 4096), (1, 4800, 4096)):
            with self.subTest(batch=batch, tokens=tokens, hidden=hidden):
                storage = torch.randn(batch * tokens * 2 * hidden + 7, device="cuda", dtype=torch.bfloat16)
                projection = storage[7:].view(batch, tokens, 2 * hidden)
                bias_storage = torch.randn(2 * hidden + 11, device="cuda")
                bias = torch.nn.Parameter(bias_storage[11:])
                snapshots = [item.clone() for item in (storage, bias_storage)]
                self.assertTrue(can_fuse_bias_swiglu(projection, bias))
                actual = fused_bias_swiglu(projection, bias)
                assert_bits(self, actual, reference_activation(projection, bias), layout=True)
                self.assertTrue(actual.is_contiguous())
                self.assertNotEqual(actual.data_ptr(), projection.data_ptr())
                for item, saved in zip((storage, bias_storage), snapshots):
                    assert_bits(self, item, saved)

    @torch.no_grad()
    def test_all_finite_bfloat16_activation_values_with_bias(self):
        # Enumerate BF16 storage patterns directly, including both signs of zero
        # and subnormals. This catches approximate-exp and flush-to-zero drift.
        patterns = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        domain = patterns.view(torch.bfloat16)
        domain = domain[torch.isfinite(domain)]
        self.assertEqual(domain.numel(), 65280)
        projection = torch.ones(1, domain.numel(), 2, device="cuda", dtype=torch.bfloat16)
        projection[0, :, 0] = domain
        for first, second in ((0.0, 0.0), (0.01, 0.0), (-0.01, 0.0), (0.5, -0.3),
                              (-0.5, 1.0), (1.00390625, 0.00390625), (-1.00390625, -0.5)):
            with self.subTest(bias=(first, second)):
                bias = torch.tensor([first, second], device="cuda")
                assert_bits(self, fused_bias_swiglu(projection, bias), reference_activation(projection, bias), layout=True)

    @torch.no_grad()
    def test_bias_cast_and_activation_rounding_are_not_removed(self):
        # Bias must narrow before addition: rounding (-1 + FP32 midpoint)
        # would retain a nonzero result instead of the original BF16 zero.
        projection = torch.tensor([[[-1.0, 1.0]]], device="cuda", dtype=torch.bfloat16)
        bias = torch.tensor([1.00390625, 0.0], device="cuda")
        expected = reference_activation(projection, bias)
        wrong_biased = (projection.float() + bias).bfloat16()
        wrong = F.silu(wrong_biased[..., :1]) * wrong_biased[..., 1:]
        self.assertEqual(expected.item(), 0.0)
        self.assertNotEqual(wrong.item(), 0.0)
        assert_bits(self, fused_bias_swiglu(projection, bias), expected)

        # Keeping SiLU in FP32 until after multiplication changes BF16 results.
        values = (torch.arange(-640, 641, device="cuda") / 128).bfloat16()
        projection = torch.empty(1, values.numel(), 2, device="cuda", dtype=torch.bfloat16)
        projection[0, :, 0] = values
        projection[0, :, 1] = 1.015625
        bias = torch.zeros(2, device="cuda")
        expected = reference_activation(projection, bias)
        wrong = (F.silu(projection[..., :1].float()) * projection[..., 1:].float()).bfloat16()
        self.assertTrue((wrong.view(torch.int16) != expected.view(torch.int16)).any().item())
        assert_bits(self, fused_bias_swiglu(projection, bias), expected)

    @torch.no_grad()
    def test_unsupported_kernel_metadata_and_optional_dependencies(self):
        projection = torch.randn(1, 17, 130, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(130, device="cuda")
        oversized_stride = projection.as_strided(projection.shape, (2**31, 130, 1))
        cases = (
            (projection.float(), bias), (projection.half(), bias),
            (projection, bias.bfloat16()), (projection, bias.half()),
            (projection[..., :129], bias[:129]), (projection[:, :0], bias),
            (projection[..., ::2], bias[::2]), (projection, bias.repeat(2)[::2]),
            (oversized_stride, bias), (projection, bias.cpu()),
            (projection.to_sparse(), bias), (projection, bias.to_sparse()),
            (torch._neg_view(projection), bias), (projection, torch._neg_view(bias)),
            (torch.nn.Parameter(projection), bias),
        )
        for args in cases:
            with self.subTest(dtypes=[item.dtype for item in args], shapes=[tuple(item.shape) for item in args]):
                self.assertFalse(can_fuse_bias_swiglu(*args))
                with self.assertRaises(ValueError):
                    fused_bias_swiglu(*args)
        with mock.patch.object(swiglu_module, "_has_triton", return_value=False):
            self.assertFalse(can_fuse_bias_swiglu(projection, bias))
        with mock.patch.object(swiglu_module, "_supports_device", return_value=False):
            self.assertFalse(can_fuse_bias_swiglu(projection, bias))

    @torch.no_grad()
    def test_full_mlp_no_bias_gemm_matches_original_projection(self):
        torch.manual_seed(1851)
        for tokens, channels, hidden in ((17, 64, 65), (600, 1536, 4096), (4800, 1536, 4096)):
            with self.subTest(tokens=tokens, channels=channels, hidden=hidden):
                mlp = mw.SwiGLUFFN(channels, hidden).cuda().eval()
                value = channel_major(1, tokens, channels)
                snapshots = {name: parameter.clone() for name, parameter in mlp.named_parameters()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    original_projection = mlp.w12(value)
                    raw = F.linear(value, mlp.w12.weight, None)
                    assert_bits(self, raw + mlp.w12.bias.bfloat16(), original_projection)
                    expected = mlp(value)
                    with mock.patch.object(mw, "fused_bias_swiglu", wraps=fused_bias_swiglu) as fused:
                        actual = mlp(value, use_fused=True)
                    self.assertEqual(fused.call_count, 1)
                assert_bits(self, actual, expected, layout=True)
                for name, parameter in mlp.named_parameters():
                    assert_bits(self, parameter, snapshots[name])

    def test_gradient_enabled_eval_retains_input_and_parameter_gradients(self):
        reference = mw.SwiGLUFFN(64, 65).cuda().eval()
        optimized = mw.SwiGLUFFN(64, 65).cuda().eval()
        optimized.load_state_dict(reference.state_dict(), strict=True)
        value = channel_major(1, 17, 64).detach().requires_grad_()
        peer = value.detach().clone().requires_grad_()
        projection = torch.randn(1, 17, 130, device="cuda", dtype=torch.bfloat16)
        self.assertFalse(can_fuse_bias_swiglu(projection, reference.w12.bias))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = reference(value)
            with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Gradient dispatch")):
                actual = optimized(peer, use_fused=True)
        assert_bits(self, actual, expected)
        expected.float().square().mean().backward()
        actual.float().square().mean().backward()
        torch.testing.assert_close(peer.grad, value.grad, rtol=0, atol=0)
        for (name, parameter), (_, other) in zip(reference.named_parameters(), optimized.named_parameters()):
            torch.testing.assert_close(other.grad, parameter.grad, rtol=0, atol=0, msg=name)

    @torch.no_grad()
    def test_unsupported_mlp_paths_training_and_disabled_flag_use_original(self):
        mlp = mw.SwiGLUFFN(64, 65).cuda().eval()
        values = (
            channel_major(2, 17, 64), channel_major(1, 1, 64),
            channel_major(1, 17, 64).contiguous(), channel_major(1, 17, 64, dtype=torch.bfloat16),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for value in values:
                with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Unsupported MLP dispatch")):
                    assert_bits(self, mlp(value, use_fused=True), mlp(value))
            value = channel_major(1, 17, 64)
            with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Disabled dispatch")):
                assert_bits(self, mlp(value, use_fused=False), mlp(value))
            mlp.train()
            with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Training dispatch")):
                assert_bits(self, mlp(value, use_fused=True), mlp(value))
            mlp.eval()
            unbiased = mw.SwiGLUFFN(64, 65, bias=False).cuda().eval()
            with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("No bias dispatch")):
                assert_bits(self, unbiased(value, use_fused=True), unbiased(value))
            with mock.patch.object(swiglu_module, "_has_triton", return_value=False):
                with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Missing Triton dispatch")):
                    assert_bits(self, mlp(value, use_fused=True), mlp(value))
        with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("FP32 autocast disabled dispatch")):
            assert_bits(self, mlp(value, use_fused=True), mlp(value))
        with torch.autocast("cuda", dtype=torch.float16):
            with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("FP16 autocast dispatch")):
                assert_bits(self, mlp(value, use_fused=True), mlp(value))

    @torch.no_grad()
    def test_forced_linear_flatten_environment_preserves_original_projection(self):
        # PyTorch caches this setting on first use. Isolate each environment
        # scenario instead of contaminating subsequent tests in this process.
        script = """
import os
from unittest import mock
import torch
from miniworld import miniworld as mw
mlp = mw.SwiGLUFFN(64, 65).cuda().eval()
value = torch.randn(1, 64, 17, device='cuda').transpose(1, 2)
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    with mock.patch.object(mw, 'fused_bias_swiglu', side_effect=AssertionError('Forced flatten dispatch')):
        for setting in ('1', '0'):
            os.environ['TORCH_LINEAR_FLATTEN_3D'] = setting
            actual = mlp(value, use_fused=True)
            expected = mlp(value)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
"""
        for startup in ("0", "1"):
            with self.subTest(startup=startup):
                environment = dict(os.environ, TORCH_LINEAR_FLATTEN_3D=startup)
                result = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @torch.no_grad()
    def test_full_model_and_current_kv_with_all_prior_optimizations(self):
        reference, optimized = model_pair(device="cuda")
        values = inputs(frames=5, batch=1, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, past = reference.forward_with_cache(*values, chunk_size=2, return_kv=True)
            saved = [(key.clone(), value.clone()) for key, value in past]
            for frames, offset, history in ((8, 0, None), (3, 0, None), (1, 5, past), (3, 5, past)):
                with self.subTest(frames=frames, offset=offset):
                    values = inputs(frames=frames, batch=1, device="cuda")
                    kwargs = dict(chunk_size=2, return_kv=True, past_kv_list=history, current_position_offset=offset)
                    expected = reference.forward_with_cache(*values, **kwargs)
                    with mock.patch.object(mw, "fused_bias_swiglu", wraps=fused_bias_swiglu) as fused:
                        actual = optimized.forward_with_cache(*values, **kwargs)
                    self.assertEqual(fused.call_count, optimized.depth)
                    assert_result_exact(self, actual, expected)
            for actual_pair, expected_pair in zip(past, saved):
                for actual, expected in zip(actual_pair, expected_pair):
                    assert_bits(self, actual, expected)
            ordinary_values = inputs(frames=8, batch=1, device="cuda")
            with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Ordinary model dispatch")):
                assert_bits(self, optimized(*ordinary_values, temporal_causal=True, chunk_size=2),
                            reference(*ordinary_values, temporal_causal=True, chunk_size=2))

    @torch.no_grad()
    def test_hooks_and_custom_projection_forward_are_honored(self):
        mlp = mw.SwiGLUFFN(64, 65).cuda().eval()
        value = channel_major(1, 17, 64)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for target in (mlp, mlp.w12):
                observed = []
                handle = target.register_forward_pre_hook(lambda _module, _args, kwargs: observed.append(dict(kwargs)), with_kwargs=True)
                try:
                    expected = mw._swiglu(mlp, value, use_fused=False)
                    with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Hooked module dispatch")):
                        actual = mw._swiglu(mlp, value, use_fused=True)
                    assert_bits(self, actual, expected)
                    self.assertEqual(observed, [{}, {}])
                finally:
                    handle.remove()
            handle = mlp.w12.register_forward_hook(lambda _module, _args, output: output + 0.25)
            try:
                with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Output hook dispatch")):
                    assert_bits(self, mw._swiglu(mlp, value, use_fused=True), mw._swiglu(mlp, value))
            finally:
                handle.remove()
            original = mlp.w12.forward
            with mock.patch.object(mlp.w12, "forward", side_effect=lambda x: original(x) + 0.25) as custom:
                with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Custom projection dispatch")):
                    assert_bits(self, mw._swiglu(mlp, value, use_fused=True), mw._swiglu(mlp, value))
                self.assertEqual(custom.call_count, 2)
            handle = torch.nn.modules.module.register_module_forward_pre_hook(lambda _module, _args: None)
            try:
                with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Global hook dispatch")):
                    assert_bits(self, mw._swiglu(mlp, value, use_fused=True), mw._swiglu(mlp, value))
            finally:
                handle.remove()
            # W3 still executes normally; its hooks need not disable W12 fusion.
            handle = mlp.w3.register_forward_hook(lambda _module, _args, output: output + 0.125)
            try:
                expected = mw._swiglu(mlp, value)
                with mock.patch.object(mw, "fused_bias_swiglu", wraps=fused_bias_swiglu) as fused:
                    actual = mw._swiglu(mlp, value, use_fused=True)
                self.assertEqual(fused.call_count, 1)
                assert_bits(self, actual, expected)
            finally:
                handle.remove()

    @torch.no_grad()
    def test_legacy_ffn_and_linear_class_instance_and_subclass_overrides(self):
        class LegacyFFN(mw.SwiGLUFFN):
            def forward(self, value):
                return super().forward(value) + 0.125

        class LegacyLinear(torch.nn.Linear):
            def forward(self, value):
                return super().forward(value) + 0.25

        value = channel_major(1, 17, 64)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mlp = LegacyFFN(64, 65).cuda().eval()
            with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Subclass FFN dispatch")):
                assert_bits(self, mw._swiglu(mlp, value, use_fused=True), mw._swiglu(mlp, value))
            mlp = mw.SwiGLUFFN(64, 65).cuda().eval()
            original_ffn = mw.SwiGLUFFN.forward

            def legacy_forward(module, x):
                return original_ffn(module, x) + 0.125

            with mock.patch.object(mw.SwiGLUFFN, "forward", new=legacy_forward):
                with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Class FFN dispatch")):
                    assert_bits(self, mw._swiglu(mlp, value, use_fused=True), mw._swiglu(mlp, value))
            original = mlp.forward
            with mock.patch.object(mlp, "forward", side_effect=lambda x: original(x) + 0.125):
                with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Instance FFN dispatch")):
                    assert_bits(self, mw._swiglu(mlp, value, use_fused=True), mw._swiglu(mlp, value))
            original_linear = torch.nn.Linear.forward

            def legacy_linear(module, x):
                return original_linear(module, x) + 0.25

            with mock.patch.object(torch.nn.Linear, "forward", new=legacy_linear):
                with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Class Linear dispatch")):
                    assert_bits(self, mw._swiglu(mlp, value, use_fused=True), mw._swiglu(mlp, value))
            replacement = LegacyLinear(64, 130).cuda().eval()
            replacement.load_state_dict(mlp.w12.state_dict(), strict=True)
            mlp.w12 = replacement
            with mock.patch.object(mw, "fused_bias_swiglu", side_effect=AssertionError("Subclass Linear dispatch")):
                assert_bits(self, mw._swiglu(mlp, value, use_fused=True), mw._swiglu(mlp, value))

    def test_nondefault_stream_and_inference_mode(self):
        stream = torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(stream):
            projection = torch.randn(2, 17, 130, device="cuda", dtype=torch.bfloat16)
            bias = torch.randn(130, device="cuda")
            self.assertTrue(can_fuse_bias_swiglu(projection, bias))
            actual = fused_bias_swiglu(projection, bias)
            expected = reference_activation(projection, bias)
            done = stream.record_event()
        torch.cuda.current_stream().wait_event(done)
        assert_bits(self, actual, expected, layout=True)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two visible CUDA GPUs")
    @torch.no_grad()
    def test_input_device_and_callers_device_are_preserved(self):
        with torch.cuda.device(0):
            projection = torch.randn(1, 17, 130, device="cuda:1", dtype=torch.bfloat16)
            bias = torch.randn(130, device="cuda:1")
            actual = fused_bias_swiglu(projection, bias)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(actual.device, projection.device)
            assert_bits(self, actual, reference_activation(projection, bias), layout=True)


if __name__ == "__main__":
    unittest.main()
