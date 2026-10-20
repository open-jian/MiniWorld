"""CPU integration oracles for opt-in TeaCache trunk reuse.

The controlled decision source below deliberately requests a reuse on the
second call. This tests which model computations run independently of the
runtime's threshold policy, whose own tests cover the gate and lifecycle.
"""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from miniworld.teacache import TeaCacheConfig, TeaCacheContext, TeaCacheRuntime
from test_chunk_flash_attention import small_model
from test_headcast import CpuTestCase, small_denoiser


class ControlledRuntime:
    """Store actual trunk residuals but make the test's skip choice explicit."""

    def __init__(self):
        self.residuals = {}
        self.evaluations = []

    def evaluate(self, *, branch, context, proxy, trunk_input,
                 force_full=False, commit=False):
        key = (branch, context)
        self.evaluations.append((key, proxy.detach().clone(), commit))
        residual = self.residuals.get(key)
        return SimpleNamespace(
            key=key,
            should_compute=bool(force_full or commit or residual is None),
            reuse=lambda current: current + residual,
        )

    def store(self, decision, trunk_input, trunk_output):
        self.residuals[decision.key] = (trunk_output - trunk_input).detach().clone()


def context(runtime, *, branch="cond", key=(0, 3, 0), force_full=False):
    return SimpleNamespace(runtime=runtime, branch=branch, key=key,
                           force_full=force_full)


class TeaCacheModelTests(CpuTestCase):
    def inputs(self):
        return (torch.randn(1, 2, 3, 2, 3),
                torch.tensor([[0.9, 0.6, 0.3]]),
                torch.randn(1, 3, 3))

    def test_forced_full_preserves_original_forward_and_checkpoint(self):
        model = small_model()
        x, t, y = self.inputs()
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        baseline, _ = model.forward_with_cache(x, t, y, chunk_size=1)
        runtime = ControlledRuntime()
        actual, _ = model.forward_with_cache(
            x, t, y, chunk_size=1,
            teacache_context=context(runtime, force_full=True),
        )
        torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
        self.assertGreater(actual.std().item(), 0.01)
        self.assertEqual(set(weights), set(model.state_dict()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], atol=0, rtol=0)

    def test_reuse_skips_every_block_but_recomputes_current_output_head(self):
        model = small_model()
        x, t, y = self.inputs()
        runtime = ControlledRuntime()
        embedded = []
        head_inputs = []
        head_outputs = []

        def save_embed(module, inputs, output):
            embedded.append(output.detach().clone())

        def save_head_input(module, inputs):
            head_inputs.append(tuple(value.detach().clone() for value in inputs))

        def save_head_output(module, inputs, output):
            head_outputs.append(output.detach().clone())

        with ExitStack() as stack:
            stack.callback(model.x_embedder.register_forward_hook(save_embed).remove)
            stack.callback(model.final_layer.register_forward_pre_hook(save_head_input).remove)
            stack.callback(model.final_layer.register_forward_hook(save_head_output).remove)
            blocks = [stack.enter_context(mock.patch.object(
                block, "forward", wraps=block.forward)) for block in model.blocks]
            model.forward_with_cache(x, t, y, teacache_context=context(runtime))
            actual, new_kv = model.forward_with_cache(
                x + 0.015, t * 0.8, y,
                teacache_context=context(runtime),
            )

        self.assertEqual([block.call_count for block in blocks], [1] * len(model.blocks))
        self.assertEqual(len(head_inputs), 2)
        self.assertEqual(len(head_outputs), 2)
        old_residual = head_inputs[0][0] - embedded[0]
        torch.testing.assert_close(head_inputs[1][0], embedded[1] + old_residual,
                                   atol=0, rtol=0)
        self.assertFalse(torch.equal(head_inputs[0][1], head_inputs[1][1]))
        self.assertFalse(torch.equal(head_outputs[0], head_outputs[1]))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertIsNone(new_kv)

    def test_clean_kv_commit_runs_every_layer_even_with_a_warm_context(self):
        model = small_model()
        x, t, y = self.inputs()
        runtime = ControlledRuntime()
        ctx = context(runtime)
        model.forward_with_cache(x, t, y, teacache_context=ctx)
        clean_t = torch.zeros_like(t)
        expected, expected_kv = model.forward_with_cache(
            x + 0.02, clean_t, y, return_kv=True,
        )
        with ExitStack() as stack:
            blocks = [stack.enter_context(mock.patch.object(
                block, "forward", wraps=block.forward)) for block in model.blocks]
            actual, actual_kv = model.forward_with_cache(
                x + 0.02, clean_t, y, return_kv=True, teacache_context=ctx,
            )
        self.assertEqual([block.call_count for block in blocks], [1] * len(model.blocks))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(len(actual_kv), model.depth)
        for actual_pair, expected_pair in zip(actual_kv, expected_kv):
            for actual_tensor, expected_tensor in zip(actual_pair, expected_pair):
                torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)

    def test_context_forwarding_keeps_cfg_branches_and_window_keys_distinct(self):
        model = small_model()
        x, t, y = self.inputs()
        runtime = ControlledRuntime()
        contexts = [context(runtime), context(runtime, branch="uncond"),
                    context(runtime, key=(1, 4, 1))]
        with ExitStack() as stack:
            blocks = [stack.enter_context(mock.patch.object(
                block, "forward", wraps=block.forward)) for block in model.blocks]
            for ctx in contexts:
                model.forward_with_cache(x, t, y, teacache_context=ctx)
        self.assertEqual([block.call_count for block in blocks], [3] * len(model.blocks))
        self.assertEqual(len(runtime.residuals), 3)

    def test_combining_tea_with_attention_acceleration_is_rejected(self):
        model = small_model()
        x, t, y = self.inputs()
        for name in ("kv_context", "headcast_context"):
            with self.subTest(context=name):
                with self.assertRaisesRegex(ValueError, "TeaCache|teacache"):
                    model.forward_with_cache(
                        x, t, y, teacache_context=context(ControlledRuntime()),
                        **{name: object()},
                    )

    def test_real_runtime_reuses_bfloat16_activations_without_dtype_promotion(self):
        model = small_model()
        x, t, y = self.inputs()
        runtime = TeaCacheRuntime(TeaCacheConfig(
            enabled=True, threshold=100., coefficients=(1., 0.)))
        ctx = TeaCacheContext(runtime, "cond", (0, 3, 0))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected, _ = model.forward_with_cache(x, t, y)
            first, _ = model.forward_with_cache(x, t, y, teacache_context=ctx)
            second, _ = model.forward_with_cache(
                x + 0.015, t * 0.8, y, teacache_context=ctx)
        torch.testing.assert_close(first, expected, atol=0, rtol=0)
        self.assertEqual(second.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(second).all())
        self.assertEqual(runtime.summary()["reused_forwards"], 1)


class TeaCacheStreamingTests(CpuTestCase):
    def fixture(self):
        model = small_denoiser()
        # Multiple denoising calls per window and enough chunks to saturate
        # and evict the historical cache; one call per window cannot reuse.
        model.df_ardiff_step = 2
        model.steps = 6
        model.cfg.teacache_coefficients = (1., 0.)
        model.cfg.teacache_threshold = 100.
        inputs = (torch.randn(1, 2, 10, 2, 3), torch.randn(1, 10, 3),
                  torch.randn(1, 2, 10, 2, 3))
        return model, inputs

    def rollout(self, model, inputs):
        latents, actions, noise = inputs
        return model.generate_eval_latents_streaming(
            latents, actions, total_len=10, history_len=1,
            max_cache_chunks=2, inflight_chunks=3, sink_frames=1, noise=noise,
        )

    def test_off_never_constructs_runtime_and_enabled_rollouts_do_not_leak(self):
        model, inputs = self.fixture()
        self.assertFalse(model.cfg.teacache)
        self.assertFalse(model.cfg.teacache_calibration)
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        with mock.patch("miniworld.teacache.TeaCacheRuntime",
                        side_effect=AssertionError("TeaCache off constructed a runtime")):
            baseline = self.rollout(model, inputs)
        model.cfg.teacache = True
        rng_before = torch.random.get_rng_state().clone()
        first = self.rollout(model, inputs)
        first_summary = model.last_eval_meta["teacache"]
        self.assertGreater(first_summary["reused_forwards"], 0)
        self.assertGreater(first_summary["commit_forwards"], 0)
        second = self.rollout(model, inputs)
        torch.testing.assert_close(second, first, atol=0, rtol=0)
        self.assertEqual(model.last_eval_meta["teacache"], first_summary)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        model.cfg.teacache = False
        with mock.patch("miniworld.teacache.TeaCacheRuntime",
                        side_effect=AssertionError("TeaCache off constructed a runtime")):
            actual = self.rollout(model, inputs)
        torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
        self.assertNotIn("teacache", model.last_eval_meta)
        self.assertEqual(set(model.state_dict()), set(weights))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], atol=0, rtol=0)

    def test_calibration_is_dense_and_preserves_the_baseline_bitwise(self):
        model, inputs = self.fixture()
        baseline = self.rollout(model, inputs)
        model.cfg.teacache_calibration = True
        model.cfg.teacache_coefficients = ()
        actual = self.rollout(model, inputs)
        torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
        summary = model.last_eval_meta["teacache"]
        self.assertEqual(summary["reused_forwards"], 0)
        self.assertEqual(summary["calibration_forwards"], summary["denoising_forwards"])
        self.assertGreater(summary["commit_forwards"], 0)
        self.assertGreater(len(summary["records"]), 0)
        self.assertEqual({record["branch"] for record in summary["records"]},
                         {"cond", "uncond"})

    def test_sampler_uses_global_window_identity_after_cache_length_saturates(self):
        model, inputs = self.fixture()
        model.cfg.teacache = True
        with ExitStack() as stack:
            forwarding = stack.enter_context(mock.patch.object(
                model.net, "forward_with_cache", wraps=model.net.forward_with_cache))
            blocks = [stack.enter_context(mock.patch.object(
                block, "forward", wraps=block.forward)) for block in model.net.blocks]
            self.rollout(model, inputs)
        summary = model.last_eval_meta["teacache"]
        self.assertEqual([block.call_count for block in blocks],
                         [summary["full_forwards"]] * len(model.net.blocks))
        saturated_keys = set()
        commit_calls = 0
        for call in forwarding.call_args_list:
            ctx = call.kwargs["teacache_context"]
            self.assertEqual(ctx.key[1] - ctx.key[0], call.args[0].shape[2])
            if call.kwargs["return_kv"]:
                commit_calls += 1
                self.assertTrue(ctx.force_full)
            elif call.kwargs["current_position_offset"] == 2:
                saturated_keys.add(ctx.key)
        self.assertEqual(commit_calls, summary["commit_forwards"])
        self.assertGreater(len(saturated_keys), 1)
        self.assertGreater(len({key[2] for key in saturated_keys}), 1)
        self.assertGreater(max(key[0] for key in saturated_keys), 2)
        self.assertGreater(summary["invalidations"], 0)

    def test_no_cfg_uses_only_the_conditional_cache(self):
        model, inputs = self.fixture()
        model.cfg_scale = 1.
        model.cfg.teacache = True
        actual = self.rollout(model, inputs)
        self.assertTrue(torch.isfinite(actual).all())
        summary = model.last_eval_meta["teacache"]
        self.assertEqual(set(summary["branches"]), {"cond"})
        self.assertGreater(summary["reused_forwards"], 0)

    def test_profile_policy_is_used_unless_explicitly_overridden(self):
        model, inputs = self.fixture()
        model.cfg.teacache = True
        model.cfg.teacache_threshold = None
        model.cfg.teacache_max_reuse_steps = None
        with tempfile.TemporaryDirectory(prefix="miniworld-teacache-profile-") as directory:
            profile_path = Path(directory) / "profile.json"
            profile_path.write_text(json.dumps({
                "coefficients": [1., 0.], "threshold": .75, "max_reuse_steps": 1,
            }))
            model.cfg.teacache_profile = str(profile_path)
            self.rollout(model, inputs)
            effective = model.last_eval_meta["teacache"]["config"]
            self.assertEqual(effective["coefficients"], (1., 0.))
            self.assertEqual(effective["threshold"], .75)
            self.assertEqual(effective["max_reuse_steps"], 1)

            # Zero is an explicit override: it must not be replaced by a
            # profile value through a truthiness-based fallback.
            model.cfg.teacache_threshold = 0.
            model.cfg.teacache_max_reuse_steps = 0
            self.rollout(model, inputs)
            summary = model.last_eval_meta["teacache"]
            self.assertEqual(summary["config"]["threshold"], 0.)
            self.assertEqual(summary["config"]["max_reuse_steps"], 0)
            self.assertEqual(summary["reused_forwards"], 0)

    def test_sampler_rejects_other_acceleration_modes(self):
        for calibration in (False, True):
            for other in ("headcast", "dummy", "forcing-kv"):
                with self.subTest(calibration=calibration, other=other):
                    model, inputs = self.fixture()
                    model.cfg.teacache = not calibration
                    model.cfg.teacache_calibration = calibration
                    if other == "headcast":
                        model.cfg.headcast = True
                    else:
                        model.cfg.kv_acceleration = other
                    with self.assertRaisesRegex(ValueError, "TeaCache"):
                        self.rollout(model, inputs)

    def test_decision_trace_preserves_outputs_and_covers_every_model_forward(self):
        model, inputs = self.fixture()
        model.cfg.teacache = True
        expected = self.rollout(model, inputs)
        untraced = model.last_eval_meta['teacache']
        self.assertEqual(untraced['decision_records'], [])
        model.cfg.teacache_trace = True
        with ExitStack() as stack:
            blocks = [stack.enter_context(mock.patch.object(
                block, 'forward', wraps=block.forward)) for block in model.net.blocks]
            actual = self.rollout(model, inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        summary = model.last_eval_meta['teacache']
        rows = summary['decision_records']
        self.assertEqual(len(rows), summary['full_forwards'] + summary['reused_forwards'])
        self.assertEqual([row['call_index'] for row in rows], list(range(len(rows))))
        self.assertEqual([block.call_count for block in blocks],
                         [summary['full_forwards']] * len(model.net.blocks))
        for key in ('full_forwards', 'reused_forwards', 'commit_forwards', 'reason_counts', 'branches'):
            self.assertEqual(summary[key], untraced[key])
        self.assertTrue(any(row['phase'] == 'prefill' for row in rows))
        self.assertTrue(any(row['phase'] == 'kv_commit' for row in rows))
        for row in rows:
            self.assertEqual(row['num_layers'], model.net.depth)
            self.assertEqual(row['latent_frame_ids'], list(range(row['window_start'], row['window_end'])))
            self.assertTrue(set(row['updated_latent_frames']).issubset(row['latent_frame_ids']))
            self.assertNotIn(0, row['updated_latent_frames'])
            if row['phase'] != 'denoise':
                self.assertEqual(row['decision'], 'full')
                self.assertEqual(row['updated_latent_frames'], [])
            if row['decision'] == 'reuse':
                source = rows[row['reference_call_index']]
                self.assertLess(source['call_index'], row['call_index'])
                self.assertEqual(source['decision'], 'full')
                for key in ('branch', 'window_start', 'window_end', 'history_epoch'):
                    self.assertEqual(source[key], row[key])
        for branch in ('cond', 'uncond'):
            for frame in range(1, 10):
                updates = [row['denoise_update_indices'][row['latent_frame_ids'].index(frame)]
                           for row in rows if row['branch'] == branch and frame in row['updated_latent_frames']]
                self.assertEqual(updates, list(range(1, len(updates) + 1)))

    def test_trace_masks_pinned_context_and_trace_output_enables_recording(self):
        model, inputs = self.fixture()
        model.df_chunk_size = 2
        model.cfg.trained_num_frames = 12
        model.trained_num_frames = 12
        model.cfg.teacache = True
        with tempfile.TemporaryDirectory(prefix='miniworld-teacache-trace-') as directory:
            destination = Path(directory) / 'decisions.json'
            model.cfg.teacache_trace_output = str(destination)
            self.rollout(model, inputs)
            summary = json.loads(destination.read_text())
        self.assertTrue(summary['config']['trace'])
        rows = summary['decision_records']
        pinned_rows = [row for row in rows if 0 in row['latent_frame_ids']]
        self.assertTrue(pinned_rows)
        for row in pinned_rows:
            self.assertIn(0, row['context_latent_frames'])
            self.assertNotIn(0, row['updated_latent_frames'])
            self.assertEqual(row['timestep'][0], 0.)
            self.assertEqual(row['next_timestep'][0], 0.)
        for row in rows:
            classified = (row['updated_latent_frames'] + row['context_latent_frames']
                          + row['pending_latent_frames'] + row['finished_latent_frames'])
            self.assertEqual(sorted(classified), row['latent_frame_ids'])


if __name__ == "__main__":
    unittest.main()
