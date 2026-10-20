"""CPU regressions for TeaCache's gate, residual and rollout lifecycle."""
import math
import unittest
from unittest import mock

import torch

from miniworld.teacache import TeaCacheConfig, TeaCacheRuntime, fit_calibration


class TeaCacheRuntimeTests(unittest.TestCase):
    def runtime(self, **kwargs):
        options = dict(enabled=True, threshold=.1, coefficients=(1., 0.))
        options.update(kwargs)
        return TeaCacheRuntime(TeaCacheConfig(**options))

    def full(self, runtime, value=1., *, branch='cond', context=(0, 16, 0),
             residual=3., proxy=None, **kwargs):
        value = torch.full((2, 4), value) if not isinstance(value, torch.Tensor) else value
        proxy = value if proxy is None else proxy
        decision = runtime.evaluate(branch, context, proxy, value, **kwargs)
        self.assertTrue(decision.should_compute)
        output = value + residual
        runtime.store(decision, value, output)
        return value, output

    def test_explicit_disabled_path_never_touches_tensors(self):
        runtime = TeaCacheRuntime(TeaCacheConfig())
        decision = runtime.evaluate('arbitrary', object(), object(), object())
        self.assertTrue(decision.should_compute)
        runtime.store(decision, object(), object())
        self.assertEqual(runtime.summary()['cached_tensor_bytes'], 0)
        self.assertEqual(runtime.summary()['full_forwards'], 0)

    def test_requires_own_finite_coefficients_and_valid_configuration(self):
        invalid = [dict(enabled=True), dict(enabled=1), dict(threshold=float('nan')),
                   dict(threshold=-1), dict(coefficients=(float('inf'),)),
                   dict(max_reuse_steps=-1), dict(max_reuse_steps=1.5),
                   dict(calibration_samples=0), dict(max_trace_records=0)]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                TeaCacheConfig(**options)
        TeaCacheConfig(enabled=True, calibration=True)

    def test_reuses_residual_with_fresh_input_not_previous_output(self):
        runtime = self.runtime()
        original, output = self.full(runtime, residual=7.)
        current = original + 2.
        decision = runtime.evaluate('cond', (0, 16, 0), original, current)
        self.assertFalse(decision.should_compute)
        torch.testing.assert_close(decision.reuse(current), current + 7.)
        self.assertFalse(torch.equal(decision.reuse(current), output))
        torch.testing.assert_close(current, original + 2.)
        self.assertEqual(runtime.summary()['reused_forwards'], 1)

    def test_accumulates_adjacent_changes_and_refresh_resets_accumulator(self):
        runtime = self.runtime(threshold=.11)
        self.full(runtime, 100.)
        for value in (105., 110.25):
            x = torch.full((2, 4), value)
            decision = runtime.evaluate('cond', (0, 16, 0), x, x)
            self.assertFalse(decision.should_compute)
            decision.reuse(x)
        x = torch.full((2, 4), 115.7625)
        decision = runtime.evaluate('cond', (0, 16, 0), x, x)
        self.assertTrue(decision.should_compute)
        self.assertEqual(decision.reason, 'threshold')
        runtime.store(decision, x, x + 10.)
        next_x = x * 1.05
        next_decision = runtime.evaluate('cond', (0, 16, 0), next_x, next_x)
        self.assertFalse(next_decision.should_compute)
        torch.testing.assert_close(next_decision.reuse(next_x), next_x + 10.)

    def test_cfg_branches_keep_different_residuals_and_reset_drops_both(self):
        runtime = self.runtime()
        x, _ = self.full(runtime, branch='cond', residual=10.)
        self.full(runtime, branch='uncond', residual=20.)
        for branch, residual in (('cond', 10.), ('uncond', 20.)):
            decision = runtime.evaluate(branch, (0, 16, 0), x, x)
            self.assertFalse(decision.should_compute)
            torch.testing.assert_close(decision.reuse(x), x + residual)
        runtime.reset()
        self.assertEqual(runtime.summary()['cached_tensor_bytes'], 0)
        self.assertEqual(runtime.summary()['reused_forwards'], 0)
        self.full(runtime, branch='cond')
        self.full(runtime, branch='uncond')

    def test_same_shape_window_or_history_change_invalidates(self):
        runtime = self.runtime()
        self.full(runtime, context=(0, 16, 0))
        # The tensor's shape stays fixed while actual frames/history change.
        self.full(runtime, context=(2, 18, 0))
        self.full(runtime, context=(2, 18, 1))
        self.assertEqual(runtime.summary()['invalidations'], 2)
        self.assertEqual(runtime.summary()['reused_forwards'], 0)

    def test_dtype_and_partial_window_geometry_change_invalidates(self):
        runtime = self.runtime()
        self.full(runtime)
        self.full(runtime, torch.ones(2, 4, dtype=torch.float64))
        self.full(runtime, torch.ones(1, 4, dtype=torch.float64))
        self.assertEqual(runtime.summary()['invalidations'], 2)

    def test_kv_commit_bypasses_gate_and_cannot_write_reuse_state(self):
        runtime = self.runtime()
        x, _ = self.full(runtime)
        self.full(runtime, branch='uncond', residual=5.)
        decision = runtime.evaluate('cond', object(), object(), object(), commit=True)
        self.assertTrue(decision.should_compute)
        runtime.store(decision, object(), object())
        self.full(runtime, branch='cond')
        other = runtime.evaluate('uncond', (0, 16, 0), x, x)
        self.assertFalse(other.should_compute)
        self.assertEqual(runtime.summary()['commit_forwards'], 1)

    def test_force_full_and_zero_threshold_never_reuse(self):
        for options in ({'threshold': 0.}, {}):
            with self.subTest(options=options):
                runtime = self.runtime(**options)
                x, _ = self.full(runtime)
                decision = runtime.evaluate('cond', (0, 16, 0), x, x,
                                            force_full=bool(not options))
                self.assertTrue(decision.should_compute)
                output = x * 1.234567
                expected = output.clone()
                runtime.store(decision, x, output)
                torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_reuse_limit_and_nonfinite_gate_force_full(self):
        runtime = self.runtime(max_reuse_steps=1)
        x, _ = self.full(runtime)
        self.assertFalse(runtime.evaluate('cond', (0, 16, 0), x, x).should_compute)
        decision = runtime.evaluate('cond', (0, 16, 0), x, x)
        self.assertEqual(decision.reason, 'reuse_limit')
        runtime.store(decision, x, x + 3.)
        invalid = runtime.evaluate('cond', (0, 16, 0), x * float('nan'), x)
        self.assertTrue(invalid.should_compute)
        self.assertEqual(invalid.reason, 'nonfinite_gate')

    def test_negative_polynomial_does_not_reduce_accumulated_change(self):
        runtime = self.runtime(coefficients=(-1.,), threshold=0.)
        x, _ = self.full(runtime)
        decision = runtime.evaluate('cond', (0, 16, 0), x, x)
        self.assertTrue(decision.should_compute)
        self.assertEqual(decision.reason, 'threshold')

    def test_calibration_uses_true_output_change_and_bounded_scalar_records(self):
        runtime = self.runtime(calibration=True, calibration_samples=4, max_trace_records=1)
        x = torch.ones(2, 4)
        self.full(runtime, x, residual=4., proxy=x)
        output = x * 3 + 4
        decision = runtime.evaluate('cond', (0, 16, 0), x * 1.1, x * 3)
        self.assertTrue(decision.should_compute)
        expected = output.clone()
        runtime.store(decision, x * 3, output)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        row = runtime.summary()['records'][0]
        self.assertAlmostEqual(row['proxy_relative_l1'], .1, places=6)
        self.assertAlmostEqual(row['output_relative_l1'], .4, places=6)
        self.assertEqual(row['residual_relative_l1'], 0.)
        self.assertLessEqual(row['sample_elements'], 4)
        self.full(runtime, x * 4, residual=4.)
        self.assertEqual(len(runtime.summary()['records']), 1)
        self.assertEqual(runtime.summary()['trace_records_dropped'], 1)
        self.assertIsNone(runtime._states['cond'].residual)
        # New membership creates a fresh calibration pair, never a cross-window row.
        self.full(runtime, x * 5, context=(2, 18, 1))
        self.assertEqual(runtime.summary()['trace_records_dropped'], 1)
        self.assertEqual(runtime.summary()['reused_forwards'], 0)

    def test_stale_foreign_and_incorrect_store_calls_are_rejected(self):
        runtime = self.runtime()
        x, _ = self.full(runtime)
        reused = runtime.evaluate('cond', (0, 16, 0), x, x)
        with self.assertRaisesRegex(RuntimeError, 'skipped'):
            runtime.store(reused, x, x)
        runtime.invalidate('cond')
        with self.assertRaisesRegex(RuntimeError, 'Stale'):
            reused.reuse(x)
        current = runtime.evaluate('cond', (0, 16, 0), x, x)
        with self.assertRaisesRegex(RuntimeError, 'Foreign'):
            self.runtime().store(current, x, x)
        with self.assertRaisesRegex(ValueError, 'shape/dtype/device'):
            runtime.store(current, x, x[:1])

    def test_fit_targets_output_change_and_records_fit_domain(self):
        records = [dict(proxy_relative_l1=x / 1000,
                        output_relative_l1=.003 + 2. * x / 1000,
                        residual_relative_l1=100.) for x in range(1, 51)]
        records.append(dict(proxy_relative_l1=float('nan'), output_relative_l1=1.))
        result = fit_calibration(records)
        coefficients, metadata = result['coefficients'], result['fit_metadata']
        self.assertEqual(metadata['samples'], 50)
        self.assertEqual(metadata['target'], 'output_relative_l1')
        self.assertLess(metadata['rmse'], 1e-10)
        prediction = 0.
        for coefficient in coefficients:
            prediction = prediction * .025 + coefficient
        self.assertAlmostEqual(prediction, .053, places=10)
        self.assertTrue(all(math.isfinite(value) for value in coefficients))
        self.assertEqual(fit_calibration(records[:1])['fit_metadata']['degree'], 0)
        with self.assertRaisesRegex(ValueError, 'No finite'):
            fit_calibration([])

    def test_inference_trace_tracks_reuse_sources_gate_and_commits(self):
        runtime = self.runtime(trace=True, threshold=.11)
        x, _ = self.full(runtime, 100., trace_metadata={'outer_step': 4, 'num_layers': 28})
        x = x * 1.05
        reused = runtime.evaluate('cond', (0, 16, 0), x, x,
                                  trace_metadata={'outer_step': 5, 'num_layers': 28})
        reused.reuse(x)
        self.full(runtime, x * 1.1, trace_metadata={'outer_step': 6, 'num_layers': 28})
        runtime.evaluate('cond', (0, 2, 0), x, x, commit=True,
                         trace_metadata={'phase': 'kv_commit', 'outer_step': 7})
        self.full(runtime, x, context=(2, 18, 1), trace_metadata={'outer_step': 7})
        rows = runtime.summary()['decision_records']
        self.assertEqual([row['call_index'] for row in rows], list(range(5)))
        self.assertEqual([row['decision'] for row in rows], ['full', 'reuse', 'full', 'full', 'full'])
        self.assertEqual([row['reason'] for row in rows],
                         ['cold_start', 'reuse', 'threshold', 'kv_commit', 'cold_start'])
        row = rows[1]
        self.assertEqual(row['reference_call_index'], 0)
        self.assertEqual(row['reference_outer_step'], 4)
        self.assertEqual(row['reuse_streak'], 1)
        self.assertEqual(row['num_layers'], 28)
        self.assertAlmostEqual(row['proxy_relative_l1'], .05, places=6)
        self.assertAlmostEqual(row['predicted_change'], .05, places=6)
        self.assertAlmostEqual(row['accumulated_after_gate'], .05, places=6)
        self.assertAlmostEqual(rows[2]['accumulated_after_gate'], .15, places=6)
        self.assertEqual(rows[2]['accumulated_after'], 0.)
        self.assertIsNone(rows[4]['last_full_call_index'])
        self.assertEqual(rows[4]['history_epoch'], 1)

    def test_trace_has_no_additional_gate_reductions_and_is_bounded(self):
        from miniworld import teacache
        for trace in (False, True):
            runtime = self.runtime(trace=trace, max_trace_records=1)
            with mock.patch.object(teacache, '_relative_l1', wraps=teacache._relative_l1) as reduce:
                x, _ = self.full(runtime)
                runtime.evaluate('cond', (0, 16, 0), x, x).reuse(x)
                runtime.evaluate('cond', (0, 16, 0), x, x).reuse(x)
            self.assertEqual(reduce.call_count, 2)
            self.assertEqual(len(runtime.decision_records), 1 if trace else 0)
            self.assertEqual(runtime.summary()['decision_records_dropped'], 2 if trace else 0)


if __name__ == '__main__':
    unittest.main()
