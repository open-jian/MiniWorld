"""Rollout integration, CFG isolation and offline profile regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from miniworld.forcing_kv import ForcingKVConfig, ForcingKVRuntime
from test_headcast import CpuTestCase, small_denoiser


class ForcingKVIntegrationTests(CpuTestCase):
    def setup_model(self):
        model = small_denoiser()
        model.cfg.kv_calibration_frames = 2
        model.cfg.kv_query_samples_per_chunk = 6
        model.cfg.forcing_kv_segments = 3
        model.cfg.forcing_kv_model_identity = 'deterministic-unit-test-checkpoint'
        data = (torch.randn(1, 2, 11, 2, 3), torch.randn(1, 11, 3), torch.randn(1, 2, 11, 2, 3))
        return model, data

    def rollout(self, model, data):
        latent, condition, noise = data
        return model.generate_eval_latents_streaming(latent, condition, total_len=11,
            history_len=1, max_cache_chunks=2 if model.df_chunk_size == 1 else 1,
            inflight_chunks=3, sink_frames=1, noise=noise)

    def test_default_off_and_state_restored_after_multiple_enabled_rollouts(self):
        model, data = self.setup_model()
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        with mock.patch('miniworld.forcing_kv.ForcingKVRuntime', side_effect=AssertionError('off runtime')):
            baseline = self.rollout(model, data)
        model.cfg.kv_acceleration = 'forcing-kv'
        rng = torch.random.get_rng_state().clone()
        first = self.rollout(model, data)
        meta = model.last_eval_meta['kv_acceleration']
        self.assertEqual(meta['status'], 'active')
        self.assertEqual(meta['calibration_forwards'], 2)
        self.assertEqual(meta['converted_caches'], model.net.depth*2)
        self.assertGreater(meta['compact_attention_calls'], 0)
        self.assertGreater(meta['commit_attention_calls'], 0)
        self.assertGreater(min(meta['selection_updates_per_branch'].values()), 1)
        self.assertEqual(sum(meta['group_counts'].values()), model.net.depth*model.net.num_heads)
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(self.rollout(model, data), first, atol=0, rtol=0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        model.cfg.kv_acceleration = 'off'
        torch.testing.assert_close(self.rollout(model, data), baseline, atol=0, rtol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], atol=0, rtol=0)

    def test_profile_export_load_and_checkpoint_mismatch(self):
        model, data = self.setup_model()
        model.cfg.kv_acceleration = 'forcing-kv'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'heads.json'
            model.cfg.forcing_kv_profile_output = str(path)
            self.rollout(model, data)
            groups = model.last_eval_meta['kv_acceleration']['groups']
            self.assertEqual(json.loads(path.read_text())['groups'], groups)
            model.cfg.forcing_kv_profile_output = ''
            model.cfg.forcing_kv_profile = str(path)
            first = self.rollout(model, data)
            info = model.last_eval_meta['kv_acceleration']
            self.assertEqual(info['groups'], groups)
            self.assertEqual(info['calibration_forwards'], 0)
            self.assertTrue(info['profile_loaded'])
            torch.testing.assert_close(self.rollout(model, data), first, atol=0, rtol=0)
            model.cfg_interval_min += .01
            with self.assertRaises(ValueError):
                self.rollout(model, data)
            model.cfg_interval_min -= .01
            model.cfg.forcing_kv_model_identity = 'different-checkpoint'
            with self.assertRaises(ValueError):
                self.rollout(model, data)

    def test_no_cfg_warmup_only_and_legacy_conflict(self):
        model, data = self.setup_model()
        model.cfg.kv_acceleration = 'forcing-kv'
        model.cfg_scale = 1.
        self.rollout(model, data)
        info = model.last_eval_meta['kv_acceleration']
        self.assertEqual(info['calibration_forwards'], 1)
        self.assertEqual(set(info['selection_updates_per_branch']), {'cond'})
        model.cfg.kv_calibration_frames = 100
        self.assertTrue(torch.isfinite(self.rollout(model, data)).all())
        self.assertEqual(model.last_eval_meta['kv_acceleration']['status'], 'warmup')
        model.cfg.headcast = True
        with self.assertRaisesRegex(ValueError, 'not both'):
            self.rollout(model, data)

    def test_two_frame_chunks_partial_tail_and_offline_empty_history(self):
        model, data = self.setup_model()
        model.df_chunk_size = model.cfg.df_chunk_size = 2
        model.cfg.kv_acceleration = 'forcing-kv'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'heads.json'
            model.cfg.forcing_kv_profile_output = str(path)
            generated = self.rollout(model, data)
            self.assertEqual(generated.shape[2], 11)
            self.assertTrue(torch.isfinite(generated).all())
            model.cfg.forcing_kv_profile_output = ''
            model.cfg.forcing_kv_profile = str(path)
            first = self.rollout(model, data)
            self.assertTrue(model.last_eval_meta['kv_acceleration']['profile_loaded'])
            self.assertEqual(model.last_eval_meta['kv_acceleration']['calibration_forwards'], 0)
            torch.testing.assert_close(self.rollout(model, data), first, atol=0, rtol=0)

    def test_configuration_rejects_invalid_geometry_and_budgets(self):
        for setting in ({'threshold': float('nan')}, {'retention': 1.01}, {'segments': 0},
                        {'profile_path': 'a', 'profile_output': 'b'}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                ForcingKVConfig(**setting)
        with self.assertRaisesRegex(ValueError, 'segments'):
            ForcingKVRuntime(ForcingKVConfig(segments=4), depth=2, num_heads=2,
                             height=2, width=3, chunk_size=2, max_cache_frames=4)


if __name__ == '__main__':
    unittest.main()
