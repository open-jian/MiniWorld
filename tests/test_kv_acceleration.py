"""Integration regressions for opt-in compact caching and rollout isolation."""
import unittest
from unittest import mock

import torch

from miniworld.kv_acceleration import KVAccelerationConfig, KVAccelerationRuntime
from miniworld.kv_cache import CompactKVCache
from test_headcast import CpuTestCase, small_denoiser


class CompactStreamingTests(CpuTestCase):
    def inputs(self, model):
        model.cfg.kv_calibration_frames = 2
        model.cfg.kv_tile_height = 1
        model.cfg.kv_tile_width = 2
        model.cfg.kv_query_samples_per_chunk = 6
        return torch.randn(1, 2, 10, 2, 3), torch.randn(1, 10, 3), torch.randn(1, 2, 10, 2, 3)

    def rollout(self, model, inputs):
        latents, conditions, noise = inputs
        return model.generate_eval_latents_streaming(
            latents, conditions, total_len=10, history_len=1,
            max_cache_chunks=2, inflight_chunks=3, sink_frames=1, noise=noise,
        )

    def test_off_never_constructs_runtime_and_is_bitwise_unchanged_after_enabled_run(self):
        model = small_denoiser()
        inputs = self.inputs(model)
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        with mock.patch('miniworld.kv_acceleration.KVAccelerationRuntime',
                        side_effect=AssertionError('off must not construct runtime')):
            baseline = self.rollout(model, inputs)
        model.cfg.kv_acceleration = 'dummy'
        self.rollout(model, inputs)
        model.cfg.kv_acceleration = 'off'
        actual = self.rollout(model, inputs)
        torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
        self.assertEqual(set(weights), set(model.state_dict()))
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, weights[key], atol=0, rtol=0)

    def test_exact_dense_backend_survives_commits_and_rope_eviction(self):
        model = small_denoiser()
        inputs = self.inputs(model)
        baseline = self.rollout(model, inputs)
        model.cfg.kv_acceleration = 'dense'
        actual = self.rollout(model, inputs)
        torch.testing.assert_close(actual, baseline, atol=2e-6, rtol=2e-5)
        summary = model.last_eval_meta['kv_acceleration']
        self.assertEqual(summary['group_counts']['global'], 4)
        self.assertEqual(summary['converted_caches'], 4)
        self.assertEqual(summary['final_logical_history_frames'], 2)
        self.assertGreater(summary['compact_attention_calls'], 0)
        self.assertGreater(summary['commit_attention_calls'], 0)

    def test_sparse_modes_route_clean_commits_and_reset_per_rollout(self):
        for mode in ('headcast', 'headcast-source', 'dummy', 'combined'):
            with self.subTest(mode=mode):
                model = small_denoiser()
                inputs = self.inputs(model)
                model.cfg.kv_acceleration = mode
                # Make this lifecycle test exercise sparse groups even with
                # random toy weights; fidelity gating has separate policy tests.
                model.cfg.kv_max_relative_rmse = 100.0
                model.cfg.kv_min_cosine = -1.0
                state = torch.random.get_rng_state().clone()
                with mock.patch.object(model.net, 'forward_with_cache',
                                       wraps=model.net.forward_with_cache) as forwarding:
                    first = self.rollout(model, inputs)
                summary = model.last_eval_meta['kv_acceleration']
                self.assertEqual(summary['status'], 'active')
                self.assertEqual(summary['calibration_forwards'], 2)
                self.assertLess(summary['group_counts']['global'], 4)
                compact_commits = [c for c in forwarding.call_args_list
                    if c.kwargs.get('return_kv') and c.kwargs.get('kv_context') is not None
                    and isinstance(c.kwargs['past_kv_list'][0], CompactKVCache)]
                self.assertGreater(len(compact_commits), 0)
                second = self.rollout(model, inputs)
                torch.testing.assert_close(second, first, atol=0, rtol=0)
                self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_no_cfg_and_unreached_calibration_are_supported(self):
        model = small_denoiser()
        inputs = self.inputs(model)
        model.cfg_scale = 1.0
        model.cfg.kv_acceleration = 'combined'
        self.rollout(model, inputs)
        self.assertEqual(model.last_eval_meta['kv_acceleration']['calibration_forwards'], 1)
        model.cfg.kv_calibration_frames = 8
        result = self.rollout(model, inputs)
        self.assertTrue(torch.isfinite(result).all())
        summary = model.last_eval_meta['kv_acceleration']
        self.assertEqual(summary['status'], 'warmup')
        self.assertEqual(summary['converted_caches'], 0)

    def test_legacy_and_new_modes_cannot_both_be_enabled(self):
        model = small_denoiser(enabled=True)
        inputs = self.inputs(model)
        model.cfg.kv_acceleration = 'combined'
        with self.assertRaisesRegex(ValueError, 'not both'):
            self.rollout(model, inputs)


if __name__ == '__main__':
    unittest.main()
