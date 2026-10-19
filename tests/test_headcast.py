"""HeadCast calibration, generation lifecycle, and checkpoint regressions.

These tests use CPU-sized models; attention-kernel mask oracles live in the
separate HeadCast attention tests. Run with ``python -m unittest discover -s
tests -v`` in the MiniWorld environment.
"""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from miniworld import miniworld as mw
from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
from miniworld.headcast import HeadCastConfig, HeadCastRuntime


def small_denoiser(*, enabled=False, cond_dropout_prob=0.1):
    """Use active gates/output, so output regressions cannot pass vacuously."""
    torch.manual_seed(713)

    def factory(**kwargs):
        return mw.MiniWorldModel(
            hidden_size=64, depth=2, num_heads=2, patch_size=1,
            adaln_lora_dim=8, **kwargs,
        )

    cfg = DenoiserConfig(
        wm_model="headcast_unit_test", latent_size=(2, 3), latent_channels=2,
        latent_frames=8, trained_num_frames=8, cond_dim=3,
        wm_mlp_ratio=2, wm_use_checkpoint=False, df_chunk_size=1,
        df_ardiff_step=1, num_sampling_steps=3, cfg_scale=2.0,
        cond_dropout_prob=cond_dropout_prob, headcast=enabled,
        headcast_tile_height=1, headcast_tile_width=2,
        headcast_recent_chunks=1, headcast_query_samples_per_chunk=6,
    )
    with mock.patch.dict(mw.MiniWorldModels, {"headcast_unit_test": factory}):
        model = build_denoiser_from_mode(cfg).eval()
    with torch.no_grad():
        model.net.shared_mod[-1].weight.normal_(std=0.02)
        model.net.shared_mod[-1].bias.fill_(0.2)
        model.net.action_encoder.to_mod[-1].weight.normal_(std=0.02)
        model.net.final_layer.linear.weight.normal_(std=0.1)
        model.net.final_layer.adaLN_modulation[-1].weight.normal_(std=0.02)
    return model


def runtime_fixture(*, branches=("cond", "uncond"), enabled=True):
    config = HeadCastConfig(
        enabled=enabled, tile_height=1, tile_width=2, recent_chunks=1,
        query_samples_per_chunk=6,
    )
    return HeadCastRuntime(
        config, depth=2, num_heads=2, grid_height=2, grid_width=3,
        chunk_size=1, sink_frames=1, max_cache_frames=4, branches=branches,
    )


def uniform_observation(context, *, batch=2):
    """Every nonempty candidate has the exact same nonzero output."""
    q = torch.zeros(batch, 2, 18, 8)
    k = torch.zeros(batch, 2, 42, 8)
    v = torch.ones_like(k)
    full_output = torch.ones_like(q)
    originals = [tensor.clone() for tensor in (q, k, v, full_output)]
    for layer in range(2):
        layer_context = context.for_layer(layer)
        assert not layer_context.routing, "calibration must keep the dense path"
        layer_context.observe(q, k, v, 6, full_output)
    for actual, original in zip((q, k, v, full_output), originals):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def history_dependent_observation(context):
    """Only batch item 1 needs an old, spatially isolated history token."""
    q = torch.zeros(2, 2, 18, 8)
    k = torch.zeros(2, 2, 42, 8)
    v = torch.ones_like(k)
    v[1].zero_()
    # Second historical frame: outside both the sink and recent-frame tail.
    v[1, :, 6, :] = 100.0
    rows = torch.arange(18)
    legal_end = 24 + (rows // 6 + 1) * 6
    mask = torch.arange(42)[None, :] < legal_end[:, None]
    full_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    for layer in range(2):
        context.for_layer(layer).observe(q, k, v, 6, full_output)


class CpuTestCase(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)


class HeadCastRuntimeTests(CpuTestCase):
    def test_disabled_and_warmup_do_not_start_calibration(self):
        t = torch.tensor([[0.1, 0.5, 1.0]])
        disabled = runtime_fixture(enabled=False)
        self.assertIsNone(disabled.begin_forward("cond", 4, t))
        runtime = runtime_fixture()
        for cache_frames in (0, 1, 3):
            self.assertIsNone(runtime.begin_forward("cond", cache_frames, t))
        self.assertIsNone(runtime.begin_forward("cond", 4, torch.zeros_like(t)))

    def test_calibration_keeps_inputs_and_dense_output_untouched(self):
        runtime = runtime_fixture(branches=("cond",))
        t = torch.tensor([[0.1, 0.5, 1.0], [0.2, 0.6, 0.9]])
        context = runtime.begin_forward("cond", 4, t)
        self.assertIsNotNone(context)
        uniform_observation(context)
        runtime.finish_forward(context)
        context = runtime.begin_forward("cond", 4, t)
        self.assertIsNotNone(context)
        self.assertTrue(context.for_layer(0).routing)
        self.assertEqual(runtime.summary()["calibration_forwards"], 1)
        for groups in runtime.summary()["groups"]:
            # Both candidates passed; temporal wins the documented priority.
            self.assertEqual(set(groups["temporal"]), {0, 1})
            self.assertEqual(groups["spatial"], [])

    def test_waits_for_both_cfg_branches_without_recalibrating_first(self):
        runtime = runtime_fixture()
        t = torch.tensor([[0.1, 0.5, 1.0], [0.2, 0.6, 0.9]])
        conditional = runtime.begin_forward("cond", 4, t)
        uniform_observation(conditional)
        runtime.finish_forward(conditional)
        self.assertIsNone(runtime.begin_forward("cond", 4, t))
        unconditional = runtime.begin_forward("uncond", 4, t)
        uniform_observation(unconditional)
        runtime.finish_forward(unconditional)
        self.assertTrue(runtime.begin_forward("cond", 4, t).for_layer(0).routing)
        self.assertTrue(runtime.begin_forward("uncond", 4, t).for_layer(0).routing)
        self.assertEqual(runtime.summary()["calibration_forwards"], 2)

    def test_candidate_must_pass_every_cfg_branch_and_batch_item(self):
        runtime = runtime_fixture()
        t = torch.tensor([[0.1, 0.5, 1.0], [0.2, 0.6, 0.9]])
        conditional = runtime.begin_forward("cond", 4, t)
        uniform_observation(conditional)
        runtime.finish_forward(conditional)
        unconditional = runtime.begin_forward("uncond", 4, t)
        history_dependent_observation(unconditional)
        runtime.finish_forward(unconditional)
        # Three of the four (branch, batch) combinations passed. The one
        # failure must protect the history for both branches and every sample.
        for branch in ("cond", "uncond"):
            context = runtime.begin_forward(branch, 4, t)
            for layer in range(2):
                self.assertFalse(context.for_layer(layer).routing)
        for groups in runtime.summary()["groups"]:
            self.assertEqual(set(groups["global"]), {0, 1})
            self.assertEqual(groups["temporal"], [])
            self.assertEqual(groups["spatial"], [])

    def test_incomplete_forward_cannot_activate_partial_layer_assignments(self):
        runtime = runtime_fixture(branches=("cond",))
        context = runtime.begin_forward("cond", 4, torch.tensor([[0.1, 0.5, 1.0]]))
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            runtime.finish_forward(context)
        self.assertEqual(runtime.summary()["calibration_forwards"], 0)
        self.assertEqual(runtime.summary()["groups"], [])

    def test_middle_noise_chunk_failure_is_not_hidden_by_high_noise_pass(self):
        runtime = runtime_fixture(branches=("cond",))
        context = runtime.begin_forward("cond", 4, torch.tensor([[0.1, 0.5, 1.0]]))
        q = torch.zeros(1, 2, 18, 8)
        k = torch.zeros(1, 2, 42, 8)
        v = torch.ones_like(k)
        k[:, :, 6, 0] = 1.0
        v[:, :, 6, :] = 100.0
        # Low/high chunks suppress that old token; the middle chunk needs it.
        q[:, :, :6, 0] = -100.0
        q[:, :, 12:, 0] = -100.0
        rows = torch.arange(18)
        mask = torch.arange(42)[None, :] < (24 + (rows // 6 + 1) * 6)[:, None]
        full = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        torch.testing.assert_close(full[:, :, :6], torch.ones_like(full[:, :, :6]))
        torch.testing.assert_close(full[:, :, 12:], torch.ones_like(full[:, :, 12:]))
        self.assertGreater(full[:, :, 6:12].min().item(), 3.0)
        for layer in range(2):
            context.for_layer(layer).observe(q, k, v, 6, full)
        runtime.finish_forward(context)
        for groups in runtime.summary()["groups"]:
            self.assertEqual(set(groups["global"]), {0, 1})


class HeadCastModelIsolationTests(CpuTestCase):
    def test_mismatched_context_is_rejected_before_attention(self):
        model = small_denoiser(enabled=True)
        x = torch.randn(1, 2, 3, 2, 3)
        t = torch.tensor([[0.1, 0.5, 1.0]])
        conditions = torch.randn(1, 3, 3)
        mismatches = (
            # Same token count but different image geometry must still fail.
            ({"grid_height": 3, "grid_width": 2}, "spatial grid"),
            ({"chunk_size": 2}, "chunk size"),
            ({"depth": 1}, "depth"),
            ({"num_heads": 1}, "head count"),
        )
        for overrides, message in mismatches:
            with self.subTest(overrides=overrides):
                params = dict(depth=2, num_heads=2, grid_height=2, grid_width=3,
                              chunk_size=1, sink_frames=1, max_cache_frames=4,
                              branches=("cond",))
                params.update(overrides)
                runtime = HeadCastRuntime(HeadCastConfig(enabled=True), **params)
                context = runtime.begin_forward("cond", 4, t)
                self.assertIsNotNone(context)
                with mock.patch.object(
                    model.net.blocks[0], "forward", side_effect=AssertionError("invalid context reached attention"),
                ), self.assertRaisesRegex(ValueError, message):
                    model.net.forward_with_cache(
                        x, t, conditions, chunk_size=1, headcast_context=context,
                    )

    def test_routing_context_cannot_change_forward_chunk_visibility(self):
        model = small_denoiser(enabled=True)
        runtime = runtime_fixture(branches=("cond",))
        t = torch.tensor([[0.1, 0.5, 1.0]])
        calibration = runtime.begin_forward("cond", 4, t)
        uniform_observation(calibration)
        runtime.finish_forward(calibration)
        context = runtime.begin_forward("cond", 4, t)
        self.assertTrue(context.for_layer(0).routing)
        with self.assertRaisesRegex(ValueError, "chunk size"):
            model.net.forward_with_cache(
                torch.randn(1, 2, 3, 2, 3), t, torch.randn(1, 3, 3),
                chunk_size=2, headcast_context=context,
            )

    def test_enabled_config_preserves_checkpoint_and_training_forward(self):
        baseline = small_denoiser(enabled=False, cond_dropout_prob=0.0)
        enabled = small_denoiser(enabled=True, cond_dropout_prob=0.0)
        self.assertEqual(set(baseline.state_dict()), set(enabled.state_dict()))
        enabled.load_state_dict(baseline.state_dict(), strict=True)
        x = torch.randn(2, 2, 8, 2, 3)
        t = torch.rand(2, 8)
        y = torch.randn(2, 8, 3)
        baseline.net.train()
        enabled.net.train()
        with torch.no_grad():
            expected = baseline.net(x, t, y, temporal_causal=True, chunk_size=1)
            actual = enabled.net(x, t, y, temporal_causal=True, chunk_size=1)
        self.assertGreater(expected.std().item(), 0.01)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_clean_commit_ignores_even_an_explicit_routing_context(self):
        model = small_denoiser(enabled=True)
        x = torch.randn(1, 2, 2, 2, 3)
        t = torch.zeros(1, 2)
        conditions = torch.randn(1, 2, 3)
        expected, expected_kv = model.net.forward_with_cache(
            x, t, conditions, chunk_size=1, return_kv=True,
        )
        context = mock.Mock()
        context.for_layer.side_effect = AssertionError("clean commit used HeadCast")
        actual, actual_kv = model.net.forward_with_cache(
            x, t, conditions, chunk_size=1, return_kv=True,
            headcast_context=context,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_layer, expected_layer in zip(actual_kv, expected_kv):
            for actual_tensor, expected_tensor in zip(actual_layer, expected_layer):
                torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)

    def test_streaming_disabled_never_constructs_runtime(self):
        model = small_denoiser(enabled=False)
        latents = torch.randn(1, 2, 10, 2, 3)
        conditions = torch.randn(1, 10, 3)
        noise = torch.randn_like(latents)
        with mock.patch(
            "miniworld.headcast.HeadCastRuntime",
            side_effect=AssertionError("disabled generation constructed HeadCast"),
        ):
            actual = model.generate_eval_latents_streaming(
                latents, conditions, total_len=10, history_len=1,
                max_cache_chunks=2, inflight_chunks=3, sink_frames=1, noise=noise,
            )
        self.assertEqual(actual.shape, latents.shape)
        self.assertTrue(torch.isfinite(actual).all())

    def test_streaming_uses_explicit_cfg_branches_resets_and_keeps_commits_dense(self):
        for dropout in (0.0, 0.1):
            with self.subTest(cond_dropout_prob=dropout):
                model = small_denoiser(enabled=True, cond_dropout_prob=dropout)
                state_keys = set(model.state_dict())
                latents = torch.randn(1, 2, 10, 2, 3)
                conditions = torch.randn(1, 10, 3)
                noise = torch.randn_like(latents)
                runtimes = []

                def construct_runtime(*args, **kwargs):
                    runtime = HeadCastRuntime(*args, **kwargs)
                    runtime.begin_forward = mock.Mock(wraps=runtime.begin_forward)
                    runtimes.append(runtime)
                    return runtime

                with mock.patch("miniworld.headcast.HeadCastRuntime", side_effect=construct_runtime), mock.patch.object(
                    model.net, "forward_with_cache", wraps=model.net.forward_with_cache,
                ) as forwarding:
                    results = [
                        model.generate_eval_latents_streaming(
                            latents, conditions, total_len=10, history_len=1,
                            max_cache_chunks=2, inflight_chunks=3, sink_frames=1,
                            noise=noise,
                        )
                        for _ in range(2)
                    ]
                self.assertEqual(len(runtimes), 2)
                self.assertIsNot(runtimes[0], runtimes[1])
                for runtime in runtimes:
                    branches = {call.args[0] if call.args else call.kwargs["branch"]
                                for call in runtime.begin_forward.call_args_list}
                    self.assertEqual(branches, {"cond", "uncond"})
                    self.assertEqual(runtime.summary()["status"], "active")
                commits = [call for call in forwarding.call_args_list if call.kwargs.get("return_kv")]
                self.assertGreater(len(commits), 0)
                for call in commits:
                    self.assertIsNone(call.kwargs.get("headcast_context"))
                routed_calls = [call for call in forwarding.call_args_list
                                if call.kwargs.get("headcast_context") is not None]
                self.assertGreater(len(routed_calls), 0)
                self.assertEqual(set(model.state_dict()), state_keys)
                torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
