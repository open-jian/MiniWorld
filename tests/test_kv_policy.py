"""CPU mask, branch-consensus and official-ranking checks for compact KV policy."""
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from miniworld.kv_policy import KVPolicy, _source_stats


def policy(mode="combined", *, depth=1, heads=2, branches=("cond",), **overrides):
    config = dict(mode=mode, max_relative_rmse=0.1, min_cosine=0.95,
                  query_samples_per_chunk=32, sink_frames=1, dummy_frames=1,
                  local_frames=2, tile_height=1, tile_width=1, dummy_fraction=0.5)
    config.update(overrides)
    return KVPolicy(SimpleNamespace(**config), depth, heads, 2, 2, 1, branches)


def fixture(*, batch=1, heads=2, current=12, past=16):
    q = torch.zeros(batch, heads, current, 4)
    k = torch.zeros(batch, heads, past + current, 4)
    v = torch.ones_like(k)
    return q, k, v


def dense(q, k, v, chunk_tokens=4):
    past = k.shape[-2] - q.shape[-2]
    rows = torch.arange(q.shape[-2])
    end = (rows // chunk_tokens + 1) * chunk_tokens
    mask = torch.arange(k.shape[-2])[None] < (past + end)[:, None]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class KVPolicyTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_source_cosine_gate_accepts_amplitude_change_without_rmse_guard(self):
        q, k, v = fixture()
        v[:, :, :4] = 10
        reference = dense(q, k, v)
        source = policy("headcast-source")
        guarded = policy("headcast")
        for p in (source, guarded):
            p.observe(0, "cond", q, k, v, reference, [0, 1, 2])
        self.assertEqual(source.finalize()[0]["sink"].tolist(), [0, 1])
        self.assertEqual(guarded.finalize()[0]["sink"].tolist(), [])
        self.assertTrue((source.metrics[0]["sink"]["relative_rmse"] > .1).all())
        self.assertFalse(source.summary()["relative_rmse_guarded"])

    def test_source_spatial_score_penalizes_mse_and_strict_cosine_boundary(self):
        reference = torch.ones(1, 2, 32, 4)
        close = _source_stats(reference*1.1, reference)
        far = _source_stats(reference*2.0, reference)
        torch.testing.assert_close(close["source_spatial_score"], torch.full((2,), .95))
        torch.testing.assert_close(far["source_spatial_score"], torch.full((2,), -4.0))
        p = policy("headcast-source")
        p.metrics[0] = {"spatial": close, "sink": {"source_cosine": torch.full((2,), .95)}}
        self.assertTrue(p._passes(0, "spatial").all())
        p.metrics[0]["spatial"] = far
        self.assertFalse(p._passes(0, "spatial").any())
        close["source_cosine"] = torch.full((2,), .95)
        p.metrics[0]["spatial"] = close
        self.assertFalse(p._passes(0, "spatial").any())
        self.assertTrue(p._passes(0, "sink").all())

    def test_source_score_keeps_p05_and_mse_paired_per_batch(self):
        reference = torch.zeros(2, 1, 32, 2)
        reference[..., 0] = 1
        candidate = reference.clone()
        # Batch0: p05=.9, MSE=.0125, score=.8375. Batch1: p05=1,
        # MSE=.04, score=.8. Separate worstp05/worstMSE would give.7.
        theta = torch.acos(torch.tensor(.9))
        candidate[0, 0, :4, 0] = torch.cos(theta)
        candidate[0, 0, :4, 1] = torch.sin(theta)
        candidate[1, :, :, 0] = 1 + (0.08 ** .5)
        stats = _source_stats(candidate, reference)
        expected0 = .9 - 5 * .0125
        expected1 = 1 - 5 * .04
        self.assertAlmostEqual(stats["source_spatial_score"].item(), min(expected0, expected1), places=5)
        separate_extrema = stats["cosine_p05"] - 5 * stats["mean_squared_error"]
        self.assertGreater(stats["source_spatial_score"].item(), separate_extrema.item() + .05)

    def test_source_policy_requires_all_cfg_branches_and_batches(self):
        p = policy("headcast-source", branches=("cond", "uncond"))
        q, k, v = fixture(batch=2)
        p.observe(0, "cond", q, k, v, dense(q, k, v), [0, 1, 2])
        v[1].zero_()
        v[1, :, 5] = 100
        p.observe(0, "uncond", q, k, v, dense(q, k, v), [0, 1, 2])
        self.assertEqual(p.finalize()[0]["global"].tolist(), [0, 1])

    def test_masks_use_own_chunk_and_query_relative_history(self):
        p = policy()
        masks = p._masks(torch.tensor([8, 9]), n_past=16, start=8, end=12)
        self.assertEqual(masks["sink"][0].nonzero().flatten().tolist(), list(range(4)) + list(range(24, 28)))
        self.assertEqual(masks["dummy"][0].nonzero().flatten().tolist(), list(range(20, 28)))
        self.assertEqual(masks["local"][0].nonzero().flatten().tolist(), list(range(16, 28)))
        self.assertEqual(masks["spatial"][0].nonzero().flatten().tolist(), list(range(0, 28, 4)))
        self.assertEqual(masks["spatial"][1].nonzero().flatten().tolist(), list(range(1, 28, 4)))

    def test_no_rng_or_input_mutation_and_headcast_priority(self):
        p = policy("headcast")
        q, k, v = fixture()
        output = dense(q, k, v)
        originals = [x.clone() for x in (q, k, v, output)]
        rng = torch.random.get_rng_state().clone()
        p.observe(0, "cond", q, k, v, output, [0, 1, 2])
        groups = p.finalize()
        self.assertEqual(groups[0]["sink"].tolist(), [0, 1])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for original, actual in zip(originals, (q, k, v, output)):
            self.assertTrue(torch.equal(original, actual))
        self.assertIs(p.finalize(), groups)

    def test_combined_chooses_lower_actual_spatial_cost(self):
        p = policy()
        q, k, v = fixture()
        p.observe(0, "cond", q, k, v, dense(q, k, v), [0, 1, 2])
        groups = p.finalize()
        self.assertEqual(groups[0]["spatial"].tolist(), [0, 1])
        self.assertEqual(p.summary()["group_counts"]["spatial"], 2)

    def test_every_cfg_branch_and_batch_item_must_pass(self):
        p = policy(branches=("cond", "uncond"))
        q, k, v = fixture(batch=2)
        p.observe(0, "cond", q, k, v, dense(q, k, v), [0, 1, 2])
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            p.finalize()
        # This historical token is outside sink, recent and three spatial tiles.
        v[1].zero_()
        v[1, :, 5] = 100
        p.observe(0, "uncond", q, k, v, dense(q, k, v), [0, 1, 2])
        groups = p.finalize()
        self.assertEqual(groups[0]["global"].tolist(), [0, 1])

    def test_middle_chunk_failure_is_retained(self):
        p = policy("headcast")
        q, k, v = fixture()
        k[:, :, 5, 0] = 1
        v[:, :, 5] = 100
        q[:, :, :4, 0] = -100
        q[:, :, 8:, 0] = -100
        p.observe(0, "cond", q, k, v, dense(q, k, v), [0, 1, 2])
        self.assertEqual(p.finalize()[0]["global"].tolist(), [0, 1])

    def test_global_dummy_budget_uses_released_column_normalization(self):
        p = policy("dummy", depth=2, heads=2, dummy_fraction=0.25)
        q, k, v = fixture()
        for layer in range(2):
            p.observe(layer, "cond", q, k, v, dense(q, k, v), [0])
        # Independent analytical ranking oracle. Raw max would rank head(1,0)
        # first; column normalization instead selects head(0,1) first.
        masses = torch.tensor([[[0.8, 0.1, 0.1], [0.2, 0.01, 0.79]],
                               [[0.05, 0.15, 0.8], [0.4, 0.2, 0.4]]])
        for layer in range(2):
            p._mass_sums[layer] = masses[layer]
            p._mass_weights[layer] = 1
        flat = masses.reshape(-1, 3)
        sn = flat[:, 0] / flat[:, 0].sum()
        hn = flat[:, 1] / flat[:, 1].sum()
        chosen = torch.argsort(torch.maximum(sn, hn), stable=True)[:1].tolist()
        groups = p.finalize()
        actual = [layer * 2 + head for layer, group in enumerate(groups) for head in group["dummy"].tolist()]
        self.assertEqual(sorted(actual), sorted(chosen))
        self.assertEqual(sum(group["dummy"].numel() for group in groups), 1)
        self.assertFalse(p.summary()["fidelity_guarded"])

    def test_reference_uses_low_precision_inputs_not_supplied_output(self):
        p = policy("headcast")
        q, k, v = fixture()
        v = v.to(torch.bfloat16)
        wrong_output = torch.full_like(q, 2)
        p.observe(0, "cond", q, k, v, wrong_output, [0, 1, 2])
        self.assertEqual(p.finalize()[0]["sink"].tolist(), [0, 1])
        control = p.summary()["metrics"][0]["dense_reference_control"]
        torch.testing.assert_close(torch.tensor(control["relative_rmse"]), torch.ones(2))

    def test_summary_is_available_during_warmup(self):
        p = policy()
        summary = p.summary()
        self.assertEqual(summary["status"], "calibrating")
        self.assertEqual(summary["metrics"], [])
        self.assertEqual(summary["group_counts"], {name: 0 for name in ("global", "sink", "dummy", "local", "spatial")})

    def test_duplicate_missing_and_postfinalize_observations_rejected(self):
        p = policy()
        q, k, v = fixture()
        args = (0, "cond", q, k, v, dense(q, k, v), [0])
        p.observe(*args)
        with self.assertRaisesRegex(RuntimeError, "already observed"):
            p.observe(*args)
        p.finalize()
        with self.assertRaisesRegex(RuntimeError, "after policy"):
            p.observe(*args)

    def test_partial_final_chunk_and_no_history_keep_nonempty_masks(self):
        p = policy()
        q, k, v = fixture(current=6, past=0)
        p.observe(0, "cond", q, k, v, dense(q, k, v), [0, 1])
        groups = p.finalize()[0]
        self.assertEqual(sum(ids.numel() for ids in groups.values()), 2)
        self.assertTrue(all(torch.isfinite(metric["relative_rmse"]).all()
                            for metric in p.metrics[0].values()))


if __name__ == "__main__":
    unittest.main()
