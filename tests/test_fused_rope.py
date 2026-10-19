"""Exact arithmetic, layout and dispatch regressions for inference RoPE fusion.

These checks isolate rotation from normalization and coefficient construction.
Real-video quality and end-to-end timing are validated separately.
"""

from dataclasses import replace
import unittest
from unittest import mock

import torch

from miniworld import miniworld as mw
from miniworld import fused_rope as rope_module
from miniworld.denoiser import DiffusionForcingDenoiser
from miniworld.fused_rope import can_fuse_rope, fused_rope
from miniworld.optimization_config import OptimizationConfig
from test_chunk_flash_attention import small_model
from test_framewise_conditioning import inputs


def token_major(batch, heads, tokens, *, device="cuda", dtype=torch.float32):
    return torch.randn(batch, tokens, heads, 128, device=device, dtype=dtype).permute(0, 2, 1, 3)


def reference_rope(value, cosine, sine):
    return value * cosine + mw.rotate_half(value) * sine


def assert_bits(test, actual, expected, *, layout=False):
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.shape, expected.shape)
    if layout:
        test.assertEqual(actual.stride(), expected.stride())
    integer = torch.int32 if actual.dtype == torch.float32 else torch.int16
    test.assertTrue(torch.equal(actual.view(integer), expected.view(integer)))


def model_pair(device="cpu", *, reuse=True):
    config = OptimizationConfig(reuse_rope_frequencies=reuse, fused_rope=False)
    reference = small_model(device, head_dim=128, optimization_config=config)
    optimized = small_model(device, head_dim=128, optimization_config=replace(config, fused_rope=True))
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
            for value, reference in zip(actual_pair, expected_pair):
                assert_bits(test, value, reference)


class FusedRopeFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @torch.no_grad()
    def test_cpu_module_fallback_and_checkpoint_state(self):
        value = token_major(2, 3, 17, device="cpu")
        cosine, sine = (torch.randn(17, 128) for _ in range(2))
        rope = mw.VisionRotaryEmbeddingFast3D(128, 1, 1, 17).eval()
        self.assertFalse(can_fuse_rope(value, cosine, sine))
        with self.assertRaises(ValueError):
            fused_rope(value, cosine, sine)
        with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("CPU dispatch")):
            actual = rope(value, freqs=(cosine, sine), use_fused=True)
        assert_bits(self, actual, reference_rope(value, cosine, sine), layout=True)
        reference, optimized = model_pair()
        self.assertEqual(set(reference.state_dict()), set(optimized.state_dict()))
        values = inputs(frames=8)
        with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("CPU dispatch")):
            expected = reference.forward_with_cache(*values, chunk_size=2, return_kv=True)
            actual = optimized.forward_with_cache(*values, chunk_size=2, return_kv=True)
        assert_result_exact(self, actual, expected)
        optimized.load_state_dict(reference.state_dict(), strict=True)

    def test_training_regular_and_cached_forward_preserve_gradients(self):
        reference, optimized = model_pair()
        reference.train()
        optimized.train()
        values = inputs(frames=8)
        for cached in (False, True):
            with self.subTest(cached=cached):
                reference.zero_grad(set_to_none=True)
                optimized.zero_grad(set_to_none=True)
                kwargs = dict(chunk_size=2, cond_drop=torch.tensor([False, True]))
                with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("Training dispatch")):
                    if cached:
                        expected = reference.forward_with_cache(*values, **kwargs)[0]
                        actual = optimized.forward_with_cache(*values, **kwargs)[0]
                    else:
                        expected = reference(*values, temporal_causal=True, **kwargs)
                        actual = optimized(*values, temporal_causal=True, **kwargs)
                assert_bits(self, actual, expected)
                # The public cached entry point is already decorated no_grad;
                # training gradients belong to the ordinary forward path.
                if cached:
                    self.assertFalse(expected.requires_grad)
                    self.assertFalse(actual.requires_grad)
                else:
                    expected.square().mean().backward()
                    actual.square().mean().backward()
                    for (name, parameter), (_, peer) in zip(reference.named_parameters(), optimized.named_parameters()):
                        if parameter.grad is None:
                            self.assertIsNone(peer.grad, name)
                        else:
                            torch.testing.assert_close(peer.grad, parameter.grad, rtol=0, atol=0, msg=name)


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU")
class FusedRopeCudaTests(unittest.TestCase):
    @torch.no_grad()
    def test_real_qk_shapes_both_coefficient_dtypes_and_singleton_layouts(self):
        torch.manual_seed(1841)
        shapes = ((2, 3, 17), (1, 12, 600), (1, 12, 4800), (2, 1, 17), (2, 3, 1), (1, 1, 1))
        for shape in shapes:
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(shape=shape, coefficients=dtype):
                    value = token_major(*shape)
                    cosine, sine = (torch.randn(shape[2], 128, device="cuda", dtype=dtype) for _ in range(2))
                    saved = [x.clone() for x in (value, cosine, sine)]
                    self.assertTrue(can_fuse_rope(value, cosine, sine))
                    expected = reference_rope(value, cosine, sine)
                    actual = fused_rope(value, cosine, sine)
                    assert_bits(self, actual, expected, layout=True)
                    self.assertNotEqual(actual.data_ptr(), value.data_ptr())
                    for item, snapshot in zip((value, cosine, sine), saved):
                        assert_bits(self, item, snapshot)

    @torch.no_grad()
    def test_offsets_batch_padding_and_strided_broadcast_coefficients(self):
        batch, heads, tokens = 2, 3, 17
        batch_stride = tokens * heads * 128 + 31
        storage = torch.randn(batch * batch_stride + 7, device="cuda")
        value = storage.as_strided((batch, heads, tokens, 128), (batch_stride, 128, heads * 128, 1), 7)
        saved = storage.clone()
        for dtype in (torch.bfloat16, torch.float32):
            cosine = torch.randn(tokens, 260, device="cuda", dtype=dtype)[:, 2:258:2]
            sine = torch.randn(1, 128, device="cuda", dtype=dtype).expand(tokens, 128)
            self.assertGreater(cosine.storage_offset(), 0)
            self.assertEqual(sine.stride(0), 0)
            self.assertTrue(can_fuse_rope(value, cosine, sine))
            assert_bits(self, fused_rope(value, cosine, sine), reference_rope(value, cosine, sine), layout=True)
        assert_bits(self, storage, saved)

    @torch.no_grad()
    def test_separate_fp32_rounding_signed_zero_and_subnormal_values(self):
        value = token_major(1, 2, 3)
        cosine, sine = (torch.ones(3, 128, device="cuda") for _ in range(2))
        # The eager products round before addition. A contracted multiply/add
        # would retain 2**-24 in the even channel instead of producing +0.
        value[..., ::2] = 1 + 2**-12
        value[..., 1::2] = 1 + 2**-11
        cosine[..., ::2] = 1 + 2**-12
        expected = reference_rope(value, cosine, sine)
        contracted = reference_rope(value.double(), cosine.double(), sine.double()).float()
        self.assertTrue((expected[..., ::2] == 0).all().item())
        self.assertTrue((contracted[..., ::2] != expected[..., ::2]).all().item())
        assert_bits(self, fused_rope(value, cosine, sine), expected, layout=True)

        # Include subnormals below the normal FP32 range and both signs of zero.
        pattern = torch.tensor([0.0, -0.0, 2**-149, -2**-149, 2**-126, -2**-126, 2**-127, -2**-127], device="cuda")
        value.copy_(pattern.repeat(16).view(1, 1, 1, 128).expand_as(value))
        for dtype in (torch.bfloat16, torch.float32):
            cosine = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 0.5, 1.0, -1.0], device="cuda", dtype=dtype).repeat(16).expand(3, 128)
            sine = cosine.flip(-1)
            assert_bits(self, fused_rope(value, cosine, sine), reference_rope(value, cosine, sine), layout=True)

    @torch.no_grad()
    def test_unsupported_metadata_and_optional_dependency_retain_eager_path(self):
        value = token_major(1, 2, 3)
        cosine, sine = (torch.randn(3, 128, device="cuda") for _ in range(2))
        rope = mw.VisionRotaryEmbeddingFast3D(128, 1, 1, 3).cuda().eval()
        # An unused huge batch stride exercises the index guard without a huge allocation.
        oversized_stride = value.as_strided(value.shape, (2**31, 128, 256, 1))
        compatible = (
            (value.contiguous(), cosine, sine),
            (value.bfloat16(), cosine, sine),
            (value.half(), cosine.half(), sine.half()),
            (value, cosine.bfloat16(), sine),
            (oversized_stride, cosine, sine),
            (torch._neg_view(value), cosine, sine),
        )
        for args in compatible:
            with self.subTest(dtype=args[0].dtype, stride=args[0].stride(), lazy_negative=args[0].is_neg()):
                self.assertFalse(can_fuse_rope(*args))
                with self.assertRaises(ValueError):
                    fused_rope(*args)
                with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("Unsupported dispatch")):
                    actual = rope(args[0], freqs=args[1:], use_fused=True)
                assert_bits(self, actual, reference_rope(*args), layout=True)
        for args in ((value, cosine[:, :64], sine), (value, cosine.cpu(), sine),
                     (value.to_sparse(), cosine, sine), (torch.nn.Parameter(value), cosine, sine),
                     (value[..., :64], cosine[:, :64], sine[:, :64]), (value[:, :, :0], cosine[:0], sine[:0])):
            self.assertFalse(can_fuse_rope(*args))
        with mock.patch.object(rope_module, "_has_triton", return_value=False):
            self.assertFalse(can_fuse_rope(value, cosine, sine))
            with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("Missing Triton dispatch")):
                assert_bits(self, rope(value, freqs=(cosine, sine), use_fused=True), reference_rope(value, cosine, sine))
        with mock.patch.object(rope_module, "_supports_device", return_value=False):
            self.assertFalse(can_fuse_rope(value, cosine, sine))
        with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("Disabled/training dispatch")):
            assert_bits(self, rope(value, freqs=(cosine, sine)), reference_rope(value, cosine, sine))
            rope.train()
            assert_bits(self, rope(value, freqs=(cosine, sine), use_fused=True), reference_rope(value, cosine, sine))

    def test_grad_enabled_eval_keeps_value_and_coefficient_gradients(self):
        value = token_major(2, 3, 17).detach().requires_grad_()
        cosine, sine = (torch.randn(17, 128, device="cuda", requires_grad=True) for _ in range(2))
        peers = [v.detach().clone().requires_grad_() for v in (value, cosine, sine)]
        rope = mw.VisionRotaryEmbeddingFast3D(128, 1, 1, 17).cuda().eval()
        self.assertFalse(can_fuse_rope(value, cosine, sine))
        expected = reference_rope(value, cosine, sine)
        with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("Gradient dispatch")):
            actual = rope(peers[0], freqs=peers[1:], use_fused=True)
        assert_bits(self, actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for item, peer in zip((value, cosine, sine), peers):
            torch.testing.assert_close(peer.grad, item.grad, rtol=0, atol=0)

    @torch.no_grad()
    def test_cached_model_flags_are_independent_and_existing_fusions_stay_exact(self):
        for reuse in (False, True):
            reference, optimized = model_pair("cuda", reuse=reuse)
            keys = set(optimized.state_dict())
            for frames, offset in ((8, 0), (3, 0), (1, 2)):
                with self.subTest(reuse=reuse, frames=frames, offset=offset):
                    values = inputs(frames=frames, device="cuda")
                    kwargs = dict(chunk_size=2, return_kv=True, current_position_offset=offset)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        expected = reference.forward_with_cache(*values, **kwargs)
                        with mock.patch.object(mw, "fused_rope", wraps=fused_rope) as fused:
                            with mock.patch.object(optimized.feat_rope, "_get_freqs", wraps=optimized.feat_rope._get_freqs) as get_freqs:
                                actual = optimized.forward_with_cache(*values, **kwargs)
                    self.assertEqual(fused.call_count, 2 * optimized.depth)
                    self.assertEqual(get_freqs.call_count, 1 if reuse else 2 * optimized.depth)
                    expected_dtype = torch.float32 if frames == 8 else torch.bfloat16
                    self.assertTrue(all(call.args[1].dtype == expected_dtype for call in fused.call_args_list))
                    assert_result_exact(self, actual, expected)
            self.assertEqual(set(optimized.state_dict()), keys)

    @torch.no_grad()
    def test_sink_preserving_eviction_and_current_kv_are_unchanged(self):
        reference, optimized = model_pair("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, history = reference.forward_with_cache(*inputs(frames=5, device="cuda"), chunk_size=2, return_kv=True)
            with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("Eviction must retain its own RoPE path")):
                evicted = DiffusionForcingDenoiser._evict_and_shift_cache(history, drop_frames=2, tokens_per_frame=6, rope_module=optimized.feat_rope, sink_frames=1)
            saved = [(k.clone(), v.clone()) for k, v in evicted]
            for before, after in zip(history, evicted):
                for old, new in zip(before, after):
                    assert_bits(self, new[..., :6, :], old[..., :6, :])
            for frames in (3, 1):
                values = inputs(frames=frames, device="cuda")
                kwargs = dict(past_kv_list=evicted, current_position_offset=3, chunk_size=2, return_kv=True)
                assert_result_exact(self, optimized.forward_with_cache(*values, **kwargs), reference.forward_with_cache(*values, **kwargs))
            for actual_pair, expected_pair in zip(evicted, saved):
                for actual, expected in zip(actual_pair, expected_pair):
                    assert_bits(self, actual, expected)

    @torch.no_grad()
    def test_hooks_custom_forward_and_regular_forward_keep_original_calls(self):
        reference, optimized = model_pair("cuda")
        values = inputs(frames=3, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = reference.forward_with_cache(*values, chunk_size=2)
            observed = []
            handle = optimized.feat_rope.register_forward_pre_hook(lambda _module, _args, kwargs: observed.append(dict(kwargs)), with_kwargs=True)
            try:
                with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("Hooked module dispatch")):
                    assert_result_exact(self, optimized.forward_with_cache(*values, chunk_size=2), expected)
            finally:
                handle.remove()
            self.assertEqual(len(observed), 2 * optimized.depth)
            self.assertTrue(all("freqs" not in kwargs and "use_fused" not in kwargs for kwargs in observed))

            original = optimized.feat_rope.forward

            def custom(value, num_frames_override=None, start_frame=0):
                return original(value, num_frames_override=num_frames_override, start_frame=start_frame)

            with mock.patch.object(optimized.feat_rope, "forward", side_effect=custom) as replaced:
                with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("Custom forward dispatch")):
                    assert_result_exact(self, optimized.forward_with_cache(*values, chunk_size=2), expected)
            self.assertEqual(replaced.call_count, 2 * optimized.depth)

            handle = torch.nn.modules.module.register_module_forward_pre_hook(lambda _module, _args: None)
            try:
                with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("Global hook dispatch")):
                    assert_result_exact(self, optimized.forward_with_cache(*values, chunk_size=2), expected)
            finally:
                handle.remove()
            regular_values = inputs(frames=8, device="cuda")
            with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("Ordinary forward dispatch")):
                assert_bits(self, optimized(*regular_values, temporal_causal=True, chunk_size=2), reference(*regular_values, temporal_causal=True, chunk_size=2))

    @torch.no_grad()
    def test_custom_rope_subclass_preserves_legacy_forward_signature(self):
        class LegacyRope(mw.VisionRotaryEmbeddingFast3D):
            def forward(self, value, num_frames_override=None, start_frame=0):
                return super().forward(value, num_frames_override=num_frames_override, start_frame=start_frame)

        reference, optimized = model_pair("cuda")
        legacy = LegacyRope(128, 8, 2, 3).cuda().eval()
        legacy.load_state_dict(optimized.feat_rope.state_dict(), strict=True)
        optimized.feat_rope = legacy
        values = inputs(frames=3, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = reference.forward_with_cache(*values, chunk_size=2)
            with mock.patch.object(mw, "fused_rope", side_effect=AssertionError("Subclass dispatch")):
                assert_result_exact(self, optimized.forward_with_cache(*values, chunk_size=2), expected)

    @torch.no_grad()
    def test_class_level_forward_override_preserves_pre_fusion_signature(self):
        original = mw.VisionRotaryEmbeddingFast3D.forward
        observed = []

        # A class-level replacement can support the prior frequency-reuse API
        # without accepting the newly introduced arithmetic-fusion keyword.
        def legacy_forward(module, value, num_frames_override=None, start_frame=0, freqs=None):
            observed.append(freqs is not None)
            return original(module, value, num_frames_override=num_frames_override,
                            start_frame=start_frame, freqs=freqs)

        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                reference, optimized = model_pair("cuda", reuse=reuse)
                values = inputs(frames=3, device="cuda")
                observed.clear()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = reference.forward_with_cache(*values, chunk_size=2)
                    with mock.patch.object(mw.VisionRotaryEmbeddingFast3D, "forward", new=legacy_forward):
                        with mock.patch.object(mw, "fused_rope", wraps=fused_rope) as fused:
                            with mock.patch.object(optimized.feat_rope, "_get_freqs", wraps=optimized.feat_rope._get_freqs) as get_freqs:
                                actual = optimized.forward_with_cache(*values, chunk_size=2)
                self.assertEqual(fused.call_count, 0)
                self.assertEqual(get_freqs.call_count, 1 if reuse else 2 * optimized.depth)
                self.assertEqual(observed, [reuse] * (2 * optimized.depth))
                assert_result_exact(self, actual, expected)

    def test_nondefault_stream_and_inference_mode(self):
        stream = torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(stream):
            value = token_major(2, 3, 17)
            cosine, sine = (torch.randn(17, 128, device="cuda") for _ in range(2))
            self.assertTrue(can_fuse_rope(value, cosine, sine))
            actual = fused_rope(value, cosine, sine)
            expected = reference_rope(value, cosine, sine)
            done = stream.record_event()
        torch.cuda.current_stream().wait_event(done)
        assert_bits(self, actual, expected, layout=True)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two visible CUDA GPUs")
    @torch.no_grad()
    def test_input_device_is_used_and_callers_device_is_restored(self):
        with torch.cuda.device(0):
            value = token_major(2, 3, 17, device="cuda:1")
            cosine, sine = (torch.randn(17, 128, device="cuda:1") for _ in range(2))
            self.assertEqual(torch.cuda.current_device(), 0)
            actual = fused_rope(value, cosine, sine)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(actual.device, value.device)
            assert_bits(self, actual, reference_rope(value, cosine, sine), layout=True)


if __name__ == "__main__":
    unittest.main()
