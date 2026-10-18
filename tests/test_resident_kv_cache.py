"""Resident-cache regressions against the unchanged concatenate-based path.

Run from the repository root with the MiniWorld environment::

    CUDA_VISIBLE_DEVICES=6 python -m unittest discover -s tests -v

The small models have nonzero output/gating weights, so equality cannot pass
merely because both models predict zero. CPU tests exercise the state machine,
RoPE/sink eviction and complete AR rollouts; CUDA additionally exercises the
capacity-strided views passed to FlashAttention with FP32 K and BF16 V.
"""

import contextlib
import unittest
from unittest import mock

import torch

from miniworld import miniworld as mw
from miniworld.denoiser import DenoiserConfig, DiffusionForcingDenoiser
from miniworld.kv_cache import ResidentKVCache
from test_chunk_flash_attention import small_model


class InferenceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.previous_threads = torch.get_num_threads()
        # Small attention matrices are otherwise dominated by CPU thread-pool
        # startup on the many-core profiling host.
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        torch.manual_seed(137)
        self.inference = torch.no_grad()
        self.inference.__enter__()
        self.addCleanup(self.inference.__exit__, None, None, None)

    def assert_kv_equal(self, actual, expected, *, atol=0, rtol=0):
        self.assertEqual(len(actual), len(expected))
        for actual_layer, expected_layer in zip(actual, expected):
            if expected_layer is None:
                self.assertIsNone(actual_layer)
                continue
            self.assertIsNotNone(actual_layer)
            for actual_tensor, expected_tensor in zip(actual_layer, expected_layer):
                torch.testing.assert_close(actual_tensor, expected_tensor, atol=atol, rtol=rtol)


def snapshot(cache):
    return [None if kv is None else tuple(t.clone() for t in kv) for kv in cache]


def storage_pointers(cache):
    return [tuple(t.untyped_storage().data_ptr() for t in kv) for kv in cache]


def stage(cache, layers, *, allow_commit=False):
    cache.begin_forward(layers[0][0].shape[-2], allow_commit=allow_commit)
    result = [cache.write_layer(i, k, v) for i, (k, v) in enumerate(layers)]
    cache.finish_forward()
    return result


def random_kv(n_tokens, *, depth=2, device="cpu", batch=2):
    return [
        (torch.randn(batch, 2, n_tokens, 32, device=device),
         torch.randn(batch, 2, n_tokens, 32, device=device, dtype=torch.bfloat16))
        for _ in range(depth)
    ]


class ResidentCacheStateTests(InferenceTestCase):
    def test_staging_overwrites_without_copying_or_promoting_history(self):
        cache = ResidentKVCache(depth=2, capacity_tokens=48)
        self.assertEqual(cache.length, 0)
        self.assertEqual(cache.capacity_tokens, 48)
        self.assertEqual(cache.history(), [None, None])
        original = random_kv(12)
        stage(cache, original, allow_commit=True)
        cache.commit()
        saved = snapshot(cache.history())
        pointers = storage_pointers(cache.history())

        for current_len in (18, 6, 18):
            current = random_kv(current_len)
            prefix = stage(cache, current)
            self.assertEqual(cache.length, 12)
            self.assertEqual(storage_pointers(prefix), pointers)
            self.assert_kv_equal(cache.history(), saved)
            for prefix_layer, old_layer, current_layer in zip(prefix, original, current):
                for combined, old, new in zip(prefix_layer, old_layer, current_layer):
                    torch.testing.assert_close(combined, torch.cat((old, new), dim=-2), rtol=0, atol=0)
                    self.assertEqual(combined.stride(-1), 1)
                    # A valid-prefix view retains its arena's capacity stride;
                    # compacting it would recreate the very history copy saved.
                    self.assertFalse(combined.is_contiguous())
            self.assertEqual(prefix[0][0].dtype, torch.float32)
            self.assertEqual(prefix[0][1].dtype, torch.bfloat16)
        with self.assertRaises((RuntimeError, ValueError)):
            cache.commit()

    def test_reset_reuses_allocations_and_hides_previous_episode(self):
        cache = ResidentKVCache(depth=2, capacity_tokens=48)
        stage(cache, random_kv(12), allow_commit=True)
        cache.commit()
        pointers = storage_pointers(cache.history())
        cache.reset()
        self.assertEqual(cache.length, 0)
        self.assertTrue(all(k.shape[-2] == v.shape[-2] == 0 for k, v in cache.history()))
        with self.assertRaises((RuntimeError, ValueError)):
            cache.commit()
        replacement = random_kv(6)
        prefix = stage(cache, replacement, allow_commit=True)
        cache.commit()
        self.assertEqual(storage_pointers(prefix), pointers)
        self.assert_kv_equal(cache.history(), replacement)

    def test_eviction_matches_legacy_rope_and_keeps_sink_bitwise(self):
        rope = small_model().feat_rope
        for sink_frames, drop_frames in ((1, 1), (0, 2), (1, 4), (0, 5)):
            with self.subTest(sink_frames=sink_frames, drop_frames=drop_frames):
                cache = ResidentKVCache(depth=2, capacity_tokens=48)
                stage(cache, random_kv(30), allow_commit=True)
                cache.commit()
                before = snapshot(cache.history())
                pointers = storage_pointers(cache.history())
                expected = DiffusionForcingDenoiser._evict_and_shift_cache(
                    before, drop_frames, 6, rope, sink_frames=sink_frames
                )
                cache.evict_and_shift(drop_frames, 6, rope, sink_frames=sink_frames)
                self.assertEqual(cache.length, 30 - drop_frames * 6)
                self.assertEqual(storage_pointers(cache.history()), pointers)
                self.assert_kv_equal(cache.history(), expected)
                if sink_frames:
                    for actual, old in zip(cache.history(), before):
                        for actual_tensor, old_tensor in zip(actual, old):
                            torch.testing.assert_close(actual_tensor[..., :6, :], old_tensor[..., :6, :], atol=0, rtol=0)
                # Writing the next current region must not expose dropped data.
                current = random_kv(6)
                prefix = stage(cache, current)
                reference = [tuple(torch.cat((old, new), dim=-2) for old, new in zip(old_layer, new_layer))
                             for old_layer, new_layer in zip(expected, current)]
                self.assert_kv_equal(prefix, reference)

    def test_incomplete_noncommit_and_double_commit_are_rejected(self):
        cache = ResidentKVCache(depth=2, capacity_tokens=48)
        layers = random_kv(6)
        with self.assertRaises((RuntimeError, ValueError)):
            cache.commit()
        with self.assertRaises((RuntimeError, ValueError)):
            cache.write_layer(0, *layers[0])
        cache.begin_forward(6, allow_commit=True)
        cache.write_layer(0, *layers[0])
        with self.assertRaises((RuntimeError, ValueError)):
            cache.finish_forward()
        with self.assertRaises((RuntimeError, ValueError)):
            cache.commit()
        # A new forward deliberately abandons incomplete/failed staging.
        stage(cache, layers, allow_commit=True)
        cache.commit()
        with self.assertRaises((RuntimeError, ValueError)):
            cache.commit()
        stage(cache, layers, allow_commit=False)
        with self.assertRaises((RuntimeError, ValueError)):
            cache.commit()
        self.assertEqual(cache.length, 6)

    def test_later_layer_eviction_failure_hides_partial_cache_and_allows_reuse(self):
        cache = ResidentKVCache(depth=2, capacity_tokens=48)
        stage(cache, random_kv(30), allow_commit=True)
        cache.commit()
        pointers = storage_pointers(cache.history())
        rope = small_model().feat_rope
        original_shift = rope.rope_shift_time
        calls = 0

        def fail_after_first_layer(delta, keys):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("test later-layer RoPE failure")
            return original_shift(delta, keys)

        # Layer 0 has already moved when layer 1 fails. Re-exposing the old
        # length would silently mix two different temporal layouts.
        with mock.patch.object(rope, "rope_shift_time", side_effect=fail_after_first_layer):
            with self.assertRaisesRegex(RuntimeError, "test later-layer RoPE failure"):
                cache.evict_and_shift(1, 6, rope, sink_frames=1)
        self.assertEqual(calls, 2)
        self.assertEqual(cache.length, 0)
        self.assertTrue(all(k.shape[-2] == v.shape[-2] == 0 for k, v in cache.history()))
        with self.assertRaises((RuntimeError, ValueError)):
            cache.commit()
        replacement = random_kv(6)
        stage(cache, replacement, allow_commit=True)
        cache.commit()
        self.assertEqual(storage_pointers(cache.history()), pointers)
        self.assert_kv_equal(cache.history(), replacement)

    def test_capacity_shape_dtype_and_grad_misuse_fail_before_commit(self):
        cache = ResidentKVCache(depth=2, capacity_tokens=18)
        layers = random_kv(12)
        stage(cache, layers, allow_commit=True)
        cache.commit()
        before = snapshot(cache.history())
        with self.assertRaises((RuntimeError, ValueError)):
            cache.begin_forward(7)
        for wrong in (random_kv(6, batch=1)[0],
                      (layers[0][0][..., :6, :].to(torch.float64), layers[0][1][..., :6, :]),
                      (layers[0][0][..., :6, :], layers[0][1][..., :6, :].float())):
            cache.begin_forward(6)
            with self.assertRaises((RuntimeError, ValueError)):
                cache.write_layer(0, *wrong)
        self.assertEqual(cache.length, 12)
        self.assert_kv_equal(cache.history(), before)
        with torch.enable_grad(), self.assertRaises((RuntimeError, ValueError)):
            cache.begin_forward(6)


class ResidentModelParityTests(InferenceTestCase):
    def compare_model(self, *, device, flash):
        model = small_model(device)
        cache = ResidentKVCache(model.depth, capacity_tokens=72)
        legacy = None
        batch = 2
        # First write, uncommitted denoising overwrite, a clean commit, and an
        # incomplete final current chunk exercise changing visible prefixes.
        operations = ((2, True), (5, False), (3, False), (2, True), (1, False))
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()
        with context:
            for frames, commit in operations:
                current_offset = cache.length // 6
                x = torch.randn(batch, 2, frames, 2, 3, device=device)
                t = torch.zeros(batch, frames, device=device) if commit else torch.rand(batch, frames, device=device)
                y = torch.randn(batch, frames, 3, device=device)
                drop = torch.tensor([False, True], device=device)
                kwargs = dict(current_position_offset=current_offset, return_kv=True,
                              chunk_size=2, cond_drop=drop, use_chunk_flash_attn=flash)
                expected, expected_kv = model.forward_with_cache(x, t, y, past_kv_list=legacy, **kwargs)
                actual, actual_kv = model.forward_with_cache(x, t, y, resident_cache=cache, **kwargs)
                # Same backend and values: the resident path only changes strides.
                atol = 0.003 if device == "cuda" else 0
                torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)
                self.assertGreater(actual.std().item(), 0.01)
                self.assert_kv_equal(actual_kv, expected_kv, atol=atol, rtol=atol)
                returned_snapshot = snapshot(actual_kv)
                if commit:
                    cache.commit()
                    legacy = DiffusionForcingDenoiser._append_kv_cache(
                        [None] * model.depth if legacy is None else legacy, expected_kv
                    )
                    self.assert_kv_equal(cache.history(), legacy, atol=atol, rtol=atol)
                    if cache.length > 12:
                        cache.evict_and_shift(2, 6, model.feat_rope, sink_frames=1)
                        legacy = DiffusionForcingDenoiser._evict_and_shift_cache(
                            legacy, 2, 6, model.feat_rope, sink_frames=1
                        )
                        self.assert_kv_equal(cache.history(), legacy, atol=atol, rtol=atol)
                # The public return_kv contract returns independent current K/V,
                # not aliases that a later denoising step silently overwrites.
                model.forward_with_cache(x, t, y, resident_cache=cache,
                                         current_position_offset=cache.length // 6,
                                         return_kv=False, chunk_size=2, use_chunk_flash_attn=flash)
                self.assert_kv_equal(actual_kv, returned_snapshot)

    def test_cpu_dense_model_parity(self):
        self.compare_model(device="cpu", flash=False)

    def test_invalid_offset_and_two_cache_sources_are_rejected(self):
        model = small_model()
        x, t = torch.randn(1, 2, 2, 2, 3), torch.zeros(1, 2)
        cache = ResidentKVCache(model.depth, capacity_tokens=48)
        for kwargs in (dict(current_position_offset=1), dict(past_kv_list=[None] * model.depth)):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, RuntimeError, AssertionError)):
                model.forward_with_cache(x, t, resident_cache=cache, **kwargs)
        model.forward_with_cache(x, t, resident_cache=cache, return_kv=True)
        cache.commit()
        with self.assertRaises((ValueError, RuntimeError, AssertionError)):
            model.forward_with_cache(x, t, resident_cache=cache, current_position_offset=0)

    def test_failed_new_forward_cannot_commit_an_older_clean_stage(self):
        model = small_model()
        x, t = torch.randn(1, 2, 2, 2, 3), torch.zeros(1, 2)
        cache = ResidentKVCache(model.depth, capacity_tokens=48)
        for bad_x, bad_kwargs in ((x, dict(current_position_offset=1)),
                                  (x[..., :1, :], {}),
                                  (x, dict(past_kv_list=[None] * model.depth))):
            with self.subTest(shape=tuple(bad_x.shape), kwargs=bad_kwargs):
                cache.reset()
                model.forward_with_cache(x, t, resident_cache=cache, return_kv=True)
                with self.assertRaises((ValueError, RuntimeError, AssertionError)):
                    model.forward_with_cache(bad_x, t, resident_cache=cache, **bad_kwargs)
                with self.assertRaises((ValueError, RuntimeError)):
                    cache.commit()
                self.assertEqual(cache.length, 0)

        wrong_depth = ResidentKVCache(model.depth + 1, capacity_tokens=48)
        stage(wrong_depth, random_kv(6, depth=model.depth + 1), allow_commit=True)
        with self.assertRaises((ValueError, RuntimeError, AssertionError)):
            model.forward_with_cache(x, t, resident_cache=wrong_depth)
        with self.assertRaises((ValueError, RuntimeError)):
            wrong_depth.commit()

    def test_final_layer_failure_does_not_finish_or_promote_staging(self):
        model = small_model()
        cache = ResidentKVCache(model.depth, capacity_tokens=48)
        x, t = torch.randn(1, 2, 2, 2, 3), torch.zeros(1, 2)
        model.forward_with_cache(x, t, resident_cache=cache, return_kv=True)
        cache.commit()
        saved = snapshot(cache.history())
        with mock.patch.object(model.final_layer, "forward", side_effect=RuntimeError("test output failure")):
            with self.assertRaisesRegex(RuntimeError, "test output failure"):
                model.forward_with_cache(x, t, resident_cache=cache, return_kv=True,
                                         current_position_offset=2)
        with self.assertRaises((ValueError, RuntimeError)):
            cache.commit()
        self.assertEqual(cache.length, 12)
        self.assert_kv_equal(cache.history(), saved)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
                         "CUDA FlashAttention needs compute capability >= 8.0")
    def test_cuda_dense_and_flash_model_parity(self):
        for flash in (False, True):
            with self.subTest(flash=flash):
                self.compare_model(device="cuda", flash=flash)


def tiny_denoiser(device, cfg_scale, null_cfg):
    def factory(**kwargs):
        model = mw.MiniWorldModel(hidden_size=64, depth=2, num_heads=2,
                                 patch_size=1, adaln_lora_dim=8, **kwargs)
        with torch.no_grad():
            model.shared_mod[-1].weight.normal_(std=0.02)
            model.shared_mod[-1].bias.fill_(0.2)
            model.action_encoder.to_mod[-1].weight.normal_(std=0.02)
            model.final_layer.linear.weight.normal_(std=0.1)
            model.final_layer.adaLN_modulation[-1].weight.normal_(std=0.02)
        return model

    cfg = DenoiserConfig(wm_model="resident_cache_test", latent_size=(2, 3),
                         latent_channels=2, latent_frames=8, trained_num_frames=8,
                         wm_mlp_ratio=2, wm_use_checkpoint=False, cond_dim=3,
                         cond_dropout_prob=0.2 if null_cfg else 0,
                         num_sampling_steps=3, df_chunk_size=2, df_ardiff_step=2,
                         cfg_scale=cfg_scale, cfg_interval_min=0.2)
    with mock.patch.dict(mw.MiniWorldModels, {"resident_cache_test": factory}), \
            mock.patch("miniworld.denoiser._print0"):
        return DiffusionForcingDenoiser(cfg).to(device).eval()


class ResidentSamplerParityTests(InferenceTestCase):
    def compare_rollout(self, *, device, cfg_scale, history_len, max_cache_chunks,
                        inflight_chunks, sink_frames, total_len, null_cfg=True):
        denoiser = tiny_denoiser(device, cfg_scale, null_cfg)
        latents = torch.randn(1, 2, total_len, 2, 3, device=device)
        condition = torch.randn(1, total_len, 3, device=device)
        noise = torch.randn_like(latents)
        noise_before = noise.clone()
        kwargs = dict(total_len=total_len, history_len=history_len,
                      max_cache_chunks=max_cache_chunks, inflight_chunks=inflight_chunks,
                      sink_frames=sink_frames, noise=noise)
        traces, outputs, metadata = [], [], []
        original_forward = denoiser.net.forward_with_cache
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()
        with context, mock.patch("miniworld.denoiser._print0"):
            for resident in (False, True):
                trace = []

                def forward(*args, **kw):
                    # Do not let an unimplemented toggle swallowed by **kwargs
                    # turn the full-rollout comparison into legacy vs legacy.
                    self.assertEqual(kw.get("resident_cache") is not None, resident)
                    drop = kw.get("cond_drop")
                    trace.append((args[0].shape[2], kw.get("current_position_offset", 0),
                                  kw.get("return_kv", False), args[1].clone(),
                                  None if drop is None else drop.clone()))
                    return original_forward(*args, **kw)

                with mock.patch.object(denoiser.net, "forward_with_cache", side_effect=forward):
                    outputs.append(denoiser.generate_eval_latents_streaming(
                        latents, condition, use_resident_kv_cache=resident, **kwargs
                    ))
                metadata.append(dict(denoiser.last_eval_meta))
                traces.append(trace)
        atol = 0.006 if device == "cuda" else 0
        torch.testing.assert_close(outputs[1], outputs[0], atol=atol, rtol=atol)
        torch.testing.assert_close(outputs[1][:, :, :history_len], latents[:, :, :history_len], atol=0, rtol=0)
        torch.testing.assert_close(noise, noise_before, atol=0, rtol=0)
        self.assertEqual(outputs[1].shape[2], total_len)
        self.assertGreater((outputs[1][:, :, history_len:] - noise[:, :, history_len:]).abs().max().item(), 0.01)
        for key in ("total_chunks", "num_outer_steps", "effective_steps", "cfg_enabled"):
            self.assertEqual(metadata[1][key], metadata[0][key])
        self.assertEqual(len(traces[1]), len(traces[0]))
        for actual, expected in zip(traces[1], traces[0]):
            self.assertEqual(actual[:3], expected[:3])
            torch.testing.assert_close(actual[3], expected[3], atol=0, rtol=0)
            if expected[4] is None:
                self.assertIsNone(actual[4])
            else:
                torch.testing.assert_close(actual[4], expected[4], atol=0, rtol=0)
        if total_len > inflight_chunks * 2:
            # A non-vacuous long rollout must include clean commit forwards,
            # not merely exercise a single in-flight model call.
            commits = [entry for entry in traces[1] if entry[2]]
            self.assertGreaterEqual(len(commits), 2)
            self.assertTrue(all(torch.count_nonzero(entry[3]).item() == 0 for entry in commits))

    def test_cpu_full_rollouts_cover_sink_prefill_cfg_and_empty_cache(self):
        cases = [
            dict(cfg_scale=1, history_len=1, max_cache_chunks=2, inflight_chunks=2, sink_frames=1, total_len=13),
            dict(cfg_scale=2, history_len=3, max_cache_chunks=2, inflight_chunks=2, sink_frames=1, total_len=13),
            dict(cfg_scale=2, history_len=1, max_cache_chunks=2, inflight_chunks=2, sink_frames=0, total_len=13, null_cfg=False),
            dict(cfg_scale=1, history_len=1, max_cache_chunks=0, inflight_chunks=2, sink_frames=0, total_len=9),
            dict(cfg_scale=2, history_len=1, max_cache_chunks=0, inflight_chunks=4, sink_frames=0, total_len=7),
            dict(cfg_scale=1, history_len=1, max_cache_chunks=1, inflight_chunks=2, sink_frames=2, total_len=13),
            dict(cfg_scale=1, history_len=4, max_cache_chunks=2, inflight_chunks=2, sink_frames=1, total_len=11),
        ]
        for case in cases:
            with self.subTest(**case):
                self.compare_rollout(device="cpu", **case)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
                         "CUDA FlashAttention needs compute capability >= 8.0")
    def test_cuda_autocast_full_rollout_with_cfg_sink_and_repeated_eviction(self):
        self.compare_rollout(device="cuda", cfg_scale=2, history_len=3,
                             max_cache_chunks=2, inflight_chunks=2, sink_frames=1, total_len=13)


if __name__ == "__main__":
    unittest.main()
