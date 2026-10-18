"""Exactness and lifecycle checks for per-forward RoPE frequency reuse."""

from dataclasses import replace
import unittest
from unittest import mock

import torch

from miniworld.denoiser import DiffusionForcingDenoiser
from test_framewise_conditioning import conditioning_model, inputs


def rope_model(enabled=False, *, device="cpu"):
    model = conditioning_model(True, device=device)
    model.optimization_config = replace(model.optimization_config, reuse_rope_frequencies=enabled)
    return model


def assert_result_exact(test, actual, expected):
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    test.assertGreater(expected[0].std().item(), 0.01)
    if expected[1] is None:
        test.assertIsNone(actual[1])
    else:
        for actual_pair, expected_pair in zip(actual[1], expected[1]):
            for a, b in zip(actual_pair, expected_pair):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


class RopeReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_build_counts_trained_length_and_checkpoint_state_are_preserved(self):
        reference = rope_model()
        cases = [(5, 0), (3, 0), (3, 2), (1, 7), (3, 0)]
        for enabled in (False, True):
            model = rope_model(enabled)
            self.assertEqual(set(model.state_dict()), set(reference.state_dict()))
            model.load_state_dict(reference.state_dict(), strict=True)
            before = {name: (value.data_ptr(), value.clone()) for name, value in model.named_buffers()}
            for frames, offset in cases:
                x, t, y = inputs(frames=frames)
                kwargs = dict(current_position_offset=offset, return_kv=True, chunk_size=2)
                expected = reference.forward_with_cache(x, t, y, **kwargs)
                with mock.patch.object(model.feat_rope, "_get_freqs", wraps=model.feat_rope._get_freqs) as get_freqs:
                    with mock.patch.object(model.feat_rope, "_build_freqs", wraps=model.feat_rope._build_freqs) as build:
                        actual = model.forward_with_cache(x, t, y, **kwargs)
                with self.subTest(enabled=enabled, frames=frames, offset=offset):
                    assert_result_exact(self, actual, expected)
                    expected_gets = 1 if enabled else 2 * model.depth
                    self.assertEqual(get_freqs.call_count, expected_gets)
                    expected_builds = 0 if frames == 5 and offset == 0 else expected_gets
                    self.assertEqual(build.call_count, expected_builds)
            # Reuse lasts only for one forward; no checkpoint buffers or table
            # mutations may accumulate as the window changes between calls.
            after = dict(model.named_buffers())
            self.assertEqual(set(before), set(after))
            for name, (pointer, value) in before.items():
                self.assertEqual(after[name].data_ptr(), pointer)
                torch.testing.assert_close(after[name], value, rtol=0, atol=0)

    def test_frequency_changes_and_checkpoint_reload_are_visible_on_next_call(self):
        reference = rope_model()
        optimized = rope_model(True)
        x, t, y = inputs(frames=3)
        kwargs = dict(current_position_offset=2, chunk_size=2, return_kv=True)
        previous = optimized.forward_with_cache(x, t, y, **kwargs)
        with torch.no_grad():
            reference.feat_rope.base_freqs_t.mul_(1.7)
        optimized.load_state_dict(reference.state_dict(), strict=True)
        expected = reference.forward_with_cache(x, t, y, **kwargs)
        with mock.patch.object(optimized.feat_rope, "_get_freqs", wraps=optimized.feat_rope._get_freqs) as get_freqs:
            actual = optimized.forward_with_cache(x, t, y, **kwargs)
        self.assertEqual(get_freqs.call_count, 1)
        assert_result_exact(self, actual, expected)
        self.assertGreater((actual[0] - previous[0]).abs().max().item(), 1e-6)

    def test_hooks_observe_original_arguments_and_disable_reuse(self):
        reference = rope_model()
        optimized = rope_model(True)
        observed = []
        handle = optimized.feat_rope.register_forward_pre_hook(
            lambda _module, _args, kwargs: observed.append(dict(kwargs)), with_kwargs=True,
        )
        try:
            x, t, y = inputs(frames=3)
            expected = reference.forward_with_cache(x, t, y, chunk_size=2)
            with mock.patch.object(optimized.feat_rope, "_get_freqs", wraps=optimized.feat_rope._get_freqs) as get_freqs:
                actual = optimized.forward_with_cache(x, t, y, chunk_size=2)
            self.assertEqual(get_freqs.call_count, 2 * optimized.depth)
            self.assertEqual(len(observed), 2 * optimized.depth)
            self.assertTrue(all("freqs" not in kwargs for kwargs in observed))
            assert_result_exact(self, actual, expected)
        finally:
            handle.remove()

    def test_custom_forward_retains_original_signature(self):
        reference = rope_model()
        optimized = rope_model(True)
        original = optimized.feat_rope.forward

        def custom(value, num_frames_override=None, start_frame=0):
            return original(value, num_frames_override=num_frames_override, start_frame=start_frame)

        x, t, y = inputs(frames=3)
        expected = reference.forward_with_cache(x, t, y, chunk_size=2)
        with mock.patch.object(optimized.feat_rope, "forward", side_effect=custom) as custom_forward:
            with mock.patch.object(optimized.feat_rope, "_get_freqs", wraps=optimized.feat_rope._get_freqs) as get_freqs:
                actual = optimized.forward_with_cache(x, t, y, chunk_size=2)
        self.assertEqual(custom_forward.call_count, 2 * optimized.depth)
        self.assertEqual(get_freqs.call_count, 2 * optimized.depth)
        assert_result_exact(self, actual, expected)

    def test_training_cached_and_regular_forward_keep_original_calls_and_gradients(self):
        reference = rope_model()
        optimized = rope_model(True)
        reference.train()
        optimized.train()
        x, t, y = inputs(frames=5)
        kwargs = dict(chunk_size=2, cond_drop=torch.tensor([False, True]))
        for cached in (False, True):
            reference_call = reference.forward_with_cache if cached else reference
            optimized_call = optimized.forward_with_cache if cached else optimized
            extra = {} if cached else {"temporal_causal": True}
            expected = reference_call(x, t, y, **kwargs, **extra)
            with mock.patch.object(optimized.feat_rope, "_get_freqs", wraps=optimized.feat_rope._get_freqs) as get_freqs:
                actual = optimized_call(x, t, y, **kwargs, **extra)
            self.assertEqual(get_freqs.call_count, 2 * optimized.depth)
            if cached:
                assert_result_exact(self, actual, expected)
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected.square().mean().backward()
                actual.square().mean().backward()
                for (name, parameter), (_, other) in zip(reference.named_parameters(), optimized.named_parameters()):
                    if parameter.grad is None:
                        self.assertIsNone(other.grad, name)
                    else:
                        torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0, msg=name)


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU")
class RopeReuseCudaTests(unittest.TestCase):
    @torch.no_grad()
    def test_output_and_current_kv_remain_bitwise_equal_after_sink_preserving_eviction(self):
        reference = rope_model(device="cuda")
        optimized = rope_model(True, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, history = reference.forward_with_cache(*inputs(frames=5, device="cuda"), return_kv=True, chunk_size=2)
            evicted = DiffusionForcingDenoiser._evict_and_shift_cache(
                history, drop_frames=2, tokens_per_frame=6, rope_module=reference.feat_rope, sink_frames=1,
            )
            for before, after in zip(history, evicted):
                for old, new in zip(before, after):
                    torch.testing.assert_close(new[..., :6, :], old[..., :6, :], rtol=0, atol=0)
            saved = [(k.clone(), v.clone()) for k, v in evicted]
            cases = ((5, 0, None), (3, 0, None), (3, 3, evicted), (1, 3, evicted))
            for frames, offset, past in cases:
                with self.subTest(frames=frames, offset=offset):
                    x, t, y = inputs(frames=frames, device="cuda")
                    kwargs = dict(past_kv_list=past, current_position_offset=offset, return_kv=True, chunk_size=2)
                    expected = reference.forward_with_cache(x, t, y, **kwargs)
                    with mock.patch.object(optimized.feat_rope, "_get_freqs", wraps=optimized.feat_rope._get_freqs) as get_freqs:
                        actual = optimized.forward_with_cache(x, t, y, **kwargs)
                    self.assertEqual(get_freqs.call_count, 1)
                    assert_result_exact(self, actual, expected)
            for unchanged, snapshot in zip(evicted, saved):
                for a, b in zip(unchanged, snapshot):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
