"""Analytical attention-mass, causal-prefix and reusable-profile checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from miniworld.forcing_kv_policy import ForcingKVPolicy


def policy(*, depth=2, heads=2, height=1, width=1, chunk_size=2,
           branches=("cond", "uncond"), **overrides):
    config = dict(threshold=.8, query_samples_per_chunk=32, sink_frames=1,
                  profile_path="", profile_output="", model_identity="fixture-sha256",
                  profile_metadata={"conditioning": "actions-28", "cfg_scale": 2.0})
    config.update(overrides)
    return ForcingKVPolicy(SimpleNamespace(**config), depth, heads, height, width,
                           chunk_size, branches)


def fixture(batch=1, current=2, past=4):
    q = torch.ones(batch, 2, current, 1)
    k = torch.zeros(batch, 2, past + current, 1)
    k[:, 0, past - 1:] = 6
    k[:, 1, 1:past - 1] = 6
    return q, k, torch.zeros_like(k), torch.zeros_like(q)


def observe_all(p, tensors=None, chunks=(0,)):
    tensors = tensors or fixture()
    for layer in range(p.depth):
        for branch in p.branches:
            p.observe(layer, branch, *tensors, chunks)
    return p.finalize()


def attention_mass_oracle(q, k, chunk, chunk_tokens, sink_tokens, tpf):
    """Independent explicit dense-softmax ratio, no policy helpers."""
    past = k.shape[-2] - q.shape[-2]
    start, end = chunk * chunk_tokens, min((chunk + 1) * chunk_tokens, q.shape[-2])
    probabilities = ((q[:, :, start:end].double() @ k[:, :, :past + end].double().transpose(-1, -2))
                     * q.shape[-1] ** -.5).softmax(-1)
    sink_end = min(past, sink_tokens)
    local_start = max(sink_end, past + start - tpf)
    return (probabilities[..., local_start:].sum((-1, -2))
            / probabilities[..., sink_end:].sum((-1, -2)))


class ForcingKVPolicyTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_source_mass_ratio_matches_independent_dense_oracle(self):
        p = policy()
        generator = torch.Generator().manual_seed(7)
        q = torch.randn(2, 2, 6, 3, generator=generator)
        k = torch.randn(2, 2, 10, 3, generator=generator) * 2
        # Unequal sink affinity distinguishes ratio-of-sums from mean-of-ratios.
        k[:, :, 0] *= 5
        observe_all(p, (q, k, torch.zeros_like(k), torch.zeros_like(q)), (0, 1, 2))
        oracle = torch.stack([attention_mass_oracle(q, k, chunk, 2, 1, 1)
                              for chunk in (0, 1, 2)]).mean((0, 1))
        torch.testing.assert_close(p._scores[1].double(), oracle, rtol=1e-5, atol=1e-6)

    def test_two_cfg_branches_and_all_layers_required(self):
        p = policy()
        inputs = fixture()
        p.observe(0, "cond", *inputs, [0])
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            p.finalize()
        for layer, branch in ((0, "uncond"), (1, "cond"), (1, "uncond")):
            p.observe(layer, branch, *inputs, [0])
        groups = p.finalize()
        self.assertEqual(groups[0]["static"].tolist(), [0, 1])
        self.assertEqual(groups[0]["dynamic"].tolist(), [])
        self.assertEqual(groups[1]["static"].tolist(), [0])
        self.assertEqual(groups[1]["dynamic"].tolist(), [1])

    def test_branch_scores_use_mean_not_worst_case(self):
        p = policy(threshold=.5)
        original = fixture()
        reversed_inputs = tuple(t.flip(1) for t in original)
        for layer in range(p.depth):
            p.observe(layer, "cond", *original, [0])
            p.observe(layer, "uncond", *reversed_inputs, [0])
        p.finalize()
        expected = (attention_mass_oracle(*original[:2], 0, 2, 1, 1)
                    + attention_mass_oracle(*reversed_inputs[:2], 0, 2, 1, 1)).mean(0) / 2
        torch.testing.assert_close(p._scores[1].double(), expected)
        self.assertEqual(p.groups[1]["static"].tolist(), [0, 1])

    def test_no_query_chunk_reads_future_keys(self):
        q, k, v, output = fixture(current=6)
        reference = policy()
        changed = policy()
        observe_all(reference, (q, k, v, output), [0])
        k_changed = k.clone()
        k_changed[:, :, 6:] = 100000
        observe_all(changed, (q, k_changed, v, output), [0])
        torch.testing.assert_close(reference._scores, changed._scores, rtol=0, atol=0)

    def test_sink_overlap_is_excluded_once_and_no_history_is_valid(self):
        for past in (0, 1, 2):
            p = policy(sink_frames=3)
            q = torch.zeros(1, 2, 3, 1)
            k = torch.zeros(1, 2, past + 3, 1)
            observe_all(p, (q, k, None, None), [0])
            torch.testing.assert_close(p._scores, torch.ones(2, 2))

    def test_sink_dominated_heads_do_not_lose_non_sink_statistic(self):
        p = policy()
        q, k, v, output = fixture()
        k[:, :, 0] = 10000
        observe_all(p, (q, k, v, output), [0])
        self.assertTrue(torch.isfinite(p._scores).all())
        self.assertEqual(p.groups[1]["static"].tolist(), [0])
        self.assertEqual(p.groups[1]["dynamic"].tolist(), [1])

    def test_partial_chunk_and_spatial_samples_are_deterministic(self):
        p = policy(height=2, width=3, query_samples_per_chunk=5)
        q = torch.randn(1, 2, 18, 3)
        k = torch.randn(1, 2, 42, 3)
        inputs = (q, k, torch.zeros_like(k), torch.zeros_like(q))
        copies = [tensor.clone() for tensor in inputs]
        random_before = torch.random.get_rng_state().clone()
        observe_all(p, inputs, (0, 1))
        self.assertTrue(torch.equal(random_before, torch.random.get_rng_state()))
        for original, tensor in zip(copies, inputs):
            self.assertTrue(torch.equal(original, tensor))
        self.assertEqual(p.observations[0]["samples"][-1]["visible_tokens"], 42)
        self.assertTrue(all(item["query_count"] <= 5 for item in p.observations[0]["samples"]))

    def test_duplicate_postfinalize_and_nonfinite_observations_rejected(self):
        p = policy()
        args = (0, "cond", *fixture(), [0])
        p.observe(*args)
        with self.assertRaisesRegex(RuntimeError, "already observed"):
            p.observe(*args)
        finalized = policy()
        groups = observe_all(finalized)
        self.assertIs(finalized.finalize(), groups)
        with self.assertRaisesRegex(RuntimeError, "after policy"):
            finalized.observe(*args)
        bad = policy()
        q, k, v, output = fixture()
        k[0, 0, 0] = torch.nan
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            bad.observe(0, "cond", q, k, v, output, [0])

    def test_offline_profile_roundtrip_bypasses_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "profile.json")
            calibrated = policy(profile_output=path)
            groups = observe_all(calibrated)
            loaded = policy(profile_path=path)
            restored = loaded.load_profile(device="cpu")
            self.assertTrue(loaded.profile_loaded)
            self.assertEqual(loaded._observed, set())
            self.assertIs(loaded.finalize(), restored)
            self.assertTrue(loaded.summary()["profile_loaded"])
            self.assertEqual(loaded.summary()["contract"], calibrated.summary()["contract"])
            for expected, actual in zip(groups, restored):
                for name in ("static", "dynamic"):
                    torch.testing.assert_close(actual[name], expected[name])
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            json.dumps(loaded.summary(), allow_nan=False)

    def test_profile_contract_rejects_checkpoint_conditioning_geometry_and_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "profile.json")
            observe_all(policy(profile_output=path))
            for overrides, key in [({"model_identity": "another-checkpoint"}, "model_identity"),
                                   ({"profile_metadata": {"conditioning": "pose"}}, "profile_metadata"),
                                   ({"height": 2}, "height"),
                                   ({"chunk_size": 3}, "chunk_size"),
                                   ({"threshold": .9}, "threshold"),
                                   ({"sink_frames": 0}, "sink_frames"),
                                   ({"branches": ("cond",)}, "branches")]:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    policy(profile_path=path, **overrides).load_profile()

    def test_profile_rejects_duplicate_heads_source_wan_list_and_inconsistent_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            observe_all(policy(profile_output=str(path)))
            original = json.loads(path.read_text())
            invalid = json.loads(json.dumps(original))
            invalid["groups"][1]["dynamic"] = [0]
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "partition"):
                policy(profile_path=str(path)).load_profile()
            invalid = json.loads(json.dumps(original))
            invalid["groups"][1] = {"static": [1], "dynamic": [0]}
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "disagree with scores"):
                policy(profile_path=str(path)).load_profile()
            path.write_text(json.dumps({"format": "forcingkv_offline", "layers": []}))
            with self.assertRaisesRegex(ValueError, "format or version"):
                policy(profile_path=str(path)).load_profile()

    def test_reusable_profiles_require_model_identity_and_summary_works_before_observe(self):
        p = policy(model_identity="")
        self.assertEqual(p.summary()["status"], "calibrating")
        self.assertEqual(p.summary()["group_counts"], {"static": 0, "dynamic": 0})
        with self.assertRaisesRegex(ValueError, "model_identity"):
            p.load_profile("unused.json")
        observe_all(p)
        with self.assertRaisesRegex(ValueError, "model_identity"):
            p.save_profile("unused.json")


if __name__ == "__main__":
    unittest.main()
