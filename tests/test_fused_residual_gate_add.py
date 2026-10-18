"""Exactness and dispatch regressions for residual gate/add fusion.

Inputs exercise MiniWorld's channel-major residual, packed conditioning gates,
and the BF16 multiplication boundary before the residual addition. Video quality
and end-to-end timing are measured separately on real episodes.
"""

from dataclasses import replace
import unittest
from unittest import mock

import torch

from miniworld import miniworld as mw
from miniworld import fused_residual_gate_add as gate_module
from miniworld.fused_residual_gate_add import (
    can_fuse_residual_gate_add,
    fused_residual_gate_add,
)
from test_framewise_conditioning import conditioning_model, inputs


def channel_major(batch, tokens, channels, *, device="cuda", dtype=torch.bfloat16):
    return torch.randn(batch, channels, tokens, device=device, dtype=dtype).transpose(1, 2)


def reference_gate_add(residual, update, gate, tokens_per_frame=None):
    return residual + mw._apply_gate(update, gate, tokens_per_frame)


def model_pair(*, device="cpu", framewise=True, mode="adaln_lora"):
    reference = conditioning_model(framewise, device=device, mode=mode)
    optimized = conditioning_model(framewise, device=device, mode=mode)
    reference.optimization_config = replace(reference.optimization_config, fused_residual_gate_add=False)
    optimized.optimization_config = replace(optimized.optimization_config, fused_residual_gate_add=True)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    return reference, optimized


def assert_result_exact(test, actual, expected):
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    test.assertGreater(expected[0].std().item(), 0.01)
    if expected[1] is None:
        test.assertIsNone(actual[1])
    else:
        test.assertEqual(len(actual[1]), len(expected[1]))
        for actual_pair, expected_pair in zip(actual[1], expected[1]):
            for value, reference in zip(actual_pair, expected_pair):
                test.assertEqual(value.dtype, reference.dtype)
                torch.testing.assert_close(value, reference, rtol=0, atol=0)


class ResidualGateAddFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cpu_helper_preserves_reference_and_direct_kernel_rejects(self):
        residual = channel_major(2, 15, 65, device="cpu")
        update = torch.randn_like(residual.contiguous())
        gate = torch.randn(2, 3, 1, 65, dtype=torch.bfloat16)
        with torch.no_grad():
            self.assertFalse(can_fuse_residual_gate_add(residual, update, gate, 5))
            with self.assertRaises(ValueError):
                fused_residual_gate_add(residual, update, gate, 5)
            expected = reference_gate_add(residual, update, gate, 5)
            with mock.patch.object(mw, "fused_residual_gate_add", side_effect=AssertionError("CPU kernel dispatch")):
                actual = mw._residual_gate_add(residual, update, gate, 5, use_fused=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cpu_cached_forward_and_checkpoint_keys_are_unchanged(self):
        reference, optimized = model_pair()
        keys_before = set(optimized.state_dict())
        self.assertEqual(keys_before, set(reference.state_dict()))
        values = inputs(frames=3)
        with mock.patch.object(mw, "fused_residual_gate_add", side_effect=AssertionError("CPU kernel dispatch")):
            expected = reference.forward_with_cache(*values, chunk_size=2, return_kv=True)
            actual = optimized.forward_with_cache(*values, chunk_size=2, return_kv=True)
        assert_result_exact(self, actual, expected)
        self.assertEqual(set(optimized.state_dict()), keys_before)
        optimized.load_state_dict(reference.state_dict(), strict=True)

    def test_regular_training_forward_and_gradients_use_reference(self):
        reference, optimized = model_pair()
        reference.train()
        optimized.train()
        values = inputs(frames=5)
        kwargs = dict(temporal_causal=True, chunk_size=2, cond_drop=torch.tensor([False, True]))
        with mock.patch.object(mw, "fused_residual_gate_add", side_effect=AssertionError("training kernel dispatch")):
            expected = reference(*values, **kwargs)
            actual = optimized(*values, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for (name, parameter), (_, peer) in zip(reference.named_parameters(), optimized.named_parameters()):
            if parameter.grad is None:
                self.assertIsNone(peer.grad, name)
            else:
                torch.testing.assert_close(peer.grad, parameter.grad, rtol=0, atol=0, msg=name)


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU")
class ResidualGateAddCudaTests(unittest.TestCase):
    def assert_exact(self, actual, expected):
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(actual.stride(), expected.stride())
        # Compare storage bits as well as values, including signed zero.
        self.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))

    @torch.no_grad()
    def test_compact_dense_broadcast_and_actual_update_layouts(self):
        torch.manual_seed(6201)
        for batch, frames, spatial, channels in ((2, 3, 5, 65), (1, 1, 17, 64), (1, 2, 300, 1536)):
            tokens = frames * spatial
            residual = channel_major(batch, tokens, channels)
            projected = torch.randn(batch, frames, 6 * channels, device="cuda", dtype=torch.bfloat16)
            compact = projected.chunk(6, -1)[2].unsqueeze(2)
            dense = torch.randn(batch, tokens, 6 * channels, device="cuda", dtype=torch.bfloat16).chunk(6, -1)[2]
            cases = (
                ("compact", compact, spatial),
                ("compact_batch_broadcast", compact[:1], spatial),
                ("compact_frame_broadcast", compact[:, :1], spatial),
                ("compact_expanded_zero_stride", compact[:1, :1].expand(batch, frames, 1, channels), spatial),
                ("dense", dense, None),
                ("dense_batch_broadcast", dense[:1], None),
                ("dense_row_broadcast", dense[:, :1], None),
                ("dense_expanded_zero_stride", dense[:1, :1].expand(batch, tokens, channels), None),
            )
            updates = (
                ("row_major", torch.randn(batch, tokens, channels, device="cuda", dtype=torch.bfloat16)),
                ("channel_major", channel_major(batch, tokens, channels)),
            )
            for update_label, update in updates:
                for gate_label, gate, tokens_per_frame in cases:
                    with self.subTest(shape=tuple(residual.shape), update=update_label, gate=gate_label):
                        snapshots = [value.clone() for value in (residual, update, gate)]
                        self.assertTrue(can_fuse_residual_gate_add(residual, update, gate, tokens_per_frame))
                        expected = reference_gate_add(residual, update, gate, tokens_per_frame)
                        actual = fused_residual_gate_add(residual, update, gate, tokens_per_frame)
                        self.assert_exact(actual, expected)
                        self.assertNotEqual(actual.data_ptr(), residual.data_ptr())
                        for value, saved in zip((residual, update, gate), snapshots):
                            torch.testing.assert_close(value, saved, rtol=0, atol=0)

    @torch.no_grad()
    def test_storage_offsets_and_padded_update_gate_strides(self):
        batch, tokens, channels = 2, 15, 65
        residual_storage = torch.randn(batch * tokens * channels + 7, device="cuda", dtype=torch.bfloat16)
        residual = residual_storage[7:].view(batch, channels, tokens).transpose(1, 2)
        update_storage = torch.randn(batch * tokens * channels * 2 + 11, device="cuda", dtype=torch.bfloat16)
        update = update_storage[11:].view(batch, tokens, channels * 2)[..., ::2]
        gate_storage = torch.randn(batch * 3 * 6 * channels + 13, device="cuda", dtype=torch.bfloat16)
        gate = gate_storage[13:].view(batch, 3, 6 * channels)[..., 2 * channels:3 * channels].unsqueeze(2)
        self.assertEqual(residual.storage_offset(), 7)
        self.assertEqual(update.storage_offset(), 11)
        self.assertGreater(gate.storage_offset(), 13)
        originals = [v.clone() for v in (residual_storage, update_storage, gate_storage)]
        self.assertTrue(can_fuse_residual_gate_add(residual, update, gate, 5))
        self.assert_exact(fused_residual_gate_add(residual, update, gate, 5),
                          reference_gate_add(residual, update, gate, 5))
        for value, saved in zip((residual_storage, update_storage, gate_storage), originals):
            torch.testing.assert_close(value, saved, rtol=0, atol=0)

    @torch.no_grad()
    def test_bfloat16_product_rounds_before_residual_addition(self):
        # 1.015625 squared is 1.031494140625, which rounds to 1.03125 in BF16.
        # Adding -1 before that narrowing produces a different BF16 result.
        residual = torch.full((2, 65, 15), -1, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        update = torch.full((2, 15, 65), 1.015625, device="cuda", dtype=torch.bfloat16)
        gate = torch.full((2, 3, 1, 65), 1.015625, device="cuda", dtype=torch.bfloat16)
        expected = reference_gate_add(residual, update, gate, 5)
        actual = fused_residual_gate_add(residual, update, gate, 5)
        wrong = (residual.float() + mw._apply_gate(update.float(), gate.float(), 5)).bfloat16()
        self.assertTrue((expected == 0.03125).all().item())
        self.assertTrue((wrong != expected).all().item())
        self.assert_exact(actual, expected)

    def test_gradient_enabled_helper_keeps_original_gradients(self):
        residual = channel_major(2, 15, 65).detach().requires_grad_()
        update = torch.randn(2, 15, 65, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        gate = torch.randn(2, 3, 1, 65, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        peers = [v.detach().clone().requires_grad_() for v in (residual, update, gate)]
        self.assertFalse(can_fuse_residual_gate_add(residual, update, gate, 5))
        expected = reference_gate_add(residual, update, gate, 5)
        with mock.patch.object(mw, "fused_residual_gate_add", side_effect=AssertionError("gradient kernel dispatch")):
            actual = mw._residual_gate_add(*peers, 5, use_fused=True)
        self.assert_exact(actual, expected)
        expected.float().square().mean().backward()
        actual.float().square().mean().backward()
        for value, peer in zip((residual, update, gate), peers):
            torch.testing.assert_close(peer.grad, value.grad, rtol=0, atol=0)

    @torch.no_grad()
    def test_unsupported_inputs_and_disabled_switch_fall_back(self):
        residual = channel_major(2, 15, 65)
        update = torch.randn(2, 15, 65, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(2, 15, 65, device="cuda", dtype=torch.bfloat16)
        compatible_fallbacks = (
            (residual.contiguous(), update, gate),
            (residual.float(), update.float(), gate.float()),
            (residual.half(), update.half(), gate.half()),
            (residual, update.float(), gate),
            (residual, update, gate.float()),
        )
        for values in compatible_fallbacks:
            with self.subTest(dtypes=[v.dtype for v in values], stride=values[0].stride()):
                self.assertFalse(can_fuse_residual_gate_add(*values))
                with self.assertRaises(ValueError):
                    fused_residual_gate_add(*values)
                with mock.patch.object(mw, "fused_residual_gate_add", side_effect=AssertionError("unsupported kernel dispatch")):
                    actual = mw._residual_gate_add(*values, use_fused=True)
                torch.testing.assert_close(actual, reference_gate_add(*values), rtol=0, atol=0)
        with mock.patch.object(mw, "fused_residual_gate_add", side_effect=AssertionError("disabled kernel dispatch")):
            actual = mw._residual_gate_add(residual, update, gate, use_fused=False)
        self.assert_exact(actual, reference_gate_add(residual, update, gate))
        compact = gate[:, :3].unsqueeze(2)
        for invalid in (0, -1, True, 2.5, 4):
            with self.subTest(tokens_per_frame=invalid):
                self.assertFalse(can_fuse_residual_gate_add(residual, update, compact, invalid))
                with self.assertRaises(ValueError):
                    fused_residual_gate_add(residual, update, compact, invalid)
        self.assertFalse(can_fuse_residual_gate_add(residual, update[..., :64], gate))
        self.assertFalse(can_fuse_residual_gate_add(residual, update, gate[..., :64]))
        self.assertFalse(can_fuse_residual_gate_add(residual, update, gate.cpu()))

    @torch.no_grad()
    @unittest.skipUnless(hasattr(torch, "_neg_view"), "Requires lazy negative views")
    def test_lazy_negative_views_use_reference_values_without_kernel_dispatch(self):
        residual = channel_major(2, 15, 65)
        update = torch.randn(2, 15, 65, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(2, 3, 1, 65, device="cuda", dtype=torch.bfloat16)
        for index, name in enumerate(("residual", "update", "gate")):
            with self.subTest(lazy_negative_input=name):
                values = [residual, update, gate]
                # A lazy view changes tensor values without changing its raw pointer.
                values[index] = torch._neg_view(values[index])
                self.assertTrue(values[index].is_neg())
                self.assertFalse(can_fuse_residual_gate_add(*values, 5))
                with self.assertRaises(ValueError):
                    fused_residual_gate_add(*values, 5)
                expected = reference_gate_add(*values, 5)
                with mock.patch.object(mw, "fused_residual_gate_add", side_effect=AssertionError("lazy view kernel dispatch")):
                    actual = mw._residual_gate_add(*values, 5, use_fused=True)
                self.assert_exact(actual, expected)

    @torch.no_grad()
    def test_sparse_inputs_are_rejected_before_reading_strides(self):
        residual = channel_major(2, 15, 65)
        update = torch.randn(2, 15, 65, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(2, 3, 1, 65, device="cuda", dtype=torch.bfloat16)
        for index, name in enumerate(("residual", "update", "gate")):
            with self.subTest(sparse_input=name):
                values = [residual, update, gate]
                values[index] = values[index].to_sparse()
                self.assertFalse(can_fuse_residual_gate_add(*values, 5))
                with self.assertRaises(ValueError):
                    fused_residual_gate_add(*values, 5)

    @torch.no_grad()
    def test_cached_outputs_current_kv_and_partial_chunks_are_exact(self):
        for framewise in (False, True):
            for mode in ("adaln_lora", "fully_shared", "per_block"):
                reference, optimized = model_pair(device="cuda", framewise=framewise, mode=mode)
                keys = set(optimized.state_dict())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, history = reference.forward_with_cache(*inputs(frames=1, device="cuda"), return_kv=True)
                    history_before = [(k.clone(), v.clone()) for k, v in history]
                    for frames, past in ((1, history), (3, history), (5, None)):
                        with self.subTest(framewise=framewise, mode=mode, frames=frames, cached=past is not None):
                            values = inputs(frames=frames, device="cuda")
                            kwargs = dict(past_kv_list=past, current_position_offset=1 if past else 0,
                                          return_kv=True, chunk_size=2,
                                          cond_drop=torch.tensor([False, True], device="cuda"))
                            with mock.patch.object(mw, "fused_residual_gate_add", side_effect=AssertionError("disabled model dispatch")):
                                expected = reference.forward_with_cache(*values, **kwargs)
                            with mock.patch.object(mw, "fused_residual_gate_add", wraps=fused_residual_gate_add) as kernel:
                                actual = optimized.forward_with_cache(*values, **kwargs)
                            self.assertEqual(kernel.call_count, 2 * optimized.depth)
                            assert_result_exact(self, actual, expected)
                for pair, saved_pair in zip(history, history_before):
                    for value, saved in zip(pair, saved_pair):
                        torch.testing.assert_close(value, saved, rtol=0, atol=0)
                self.assertEqual(set(optimized.state_dict()), keys)

    @torch.no_grad()
    def test_cached_training_and_fp32_inference_do_not_dispatch(self):
        reference, optimized = model_pair(device="cuda")
        values = inputs(frames=3, device="cuda")
        for training in (True, False):
            reference.train(training)
            optimized.train(training)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=training):
                with mock.patch.object(mw, "fused_residual_gate_add", side_effect=AssertionError("ineligible model dispatch")):
                    expected = reference.forward_with_cache(*values, chunk_size=2, return_kv=True)
                    actual = optimized.forward_with_cache(*values, chunk_size=2, return_kv=True)
            assert_result_exact(self, actual, expected)

    @torch.no_grad()
    def test_regular_eval_forward_keeps_reference_under_bfloat16_autocast(self):
        reference, optimized = model_pair(device="cuda")
        values = inputs(frames=5, device="cuda")
        kwargs = dict(temporal_causal=True, chunk_size=2,
                      cond_drop=torch.tensor([False, True], device="cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with mock.patch.object(mw, "fused_residual_gate_add", side_effect=AssertionError("regular eval dispatch")):
                expected = reference(*values, **kwargs)
                actual = optimized(*values, **kwargs)
        self.assertGreater(expected.std().item(), 0.01)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.no_grad()
    def test_gate_fusion_retains_fused_norm_dispatch_and_exact_model_values(self):
        reference, optimized = model_pair(device="cuda")
        for model in (reference, optimized):
            model.optimization_config = replace(model.optimization_config, fused_norm_modulation=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, history = reference.forward_with_cache(*inputs(frames=1, device="cuda"), return_kv=True)
            values = inputs(frames=3, device="cuda")
            kwargs = dict(past_kv_list=history, current_position_offset=1, return_kv=True, chunk_size=2)
            expected = reference.forward_with_cache(*values, **kwargs)
            with mock.patch.object(mw, "fused_norm_modulation", wraps=mw.fused_norm_modulation) as norm:
                with mock.patch.object(mw, "fused_residual_gate_add", wraps=fused_residual_gate_add) as gate:
                    actual = optimized.forward_with_cache(*values, **kwargs)
            self.assertGreater(norm.call_count, 0)
            self.assertEqual(gate.call_count, 2 * optimized.depth)
        assert_result_exact(self, actual, expected)

    @torch.no_grad()
    def test_missing_triton_uses_reference_helper_without_dispatch(self):
        residual = channel_major(2, 15, 65)
        update = torch.randn(2, 15, 65, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(2, 3, 1, 65, device="cuda", dtype=torch.bfloat16)
        expected = reference_gate_add(residual, update, gate, 5)
        with mock.patch.object(gate_module, "_has_triton", return_value=False):
            self.assertFalse(can_fuse_residual_gate_add(residual, update, gate, 5))
            with mock.patch.object(mw, "fused_residual_gate_add", side_effect=AssertionError("Triton unavailable")):
                actual = mw._residual_gate_add(residual, update, gate, 5, use_fused=True)
        self.assert_exact(actual, expected)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two visible CUDA GPUs")
    @torch.no_grad()
    def test_tensor_device_is_used_and_callers_current_device_is_restored(self):
        previous = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            residual = channel_major(2, 15, 65, device="cuda:1")
            update = torch.randn(2, 15, 65, device=residual.device, dtype=torch.bfloat16)
            gate = torch.randn(2, 3, 1, 65, device=residual.device, dtype=torch.bfloat16)
            self.assertEqual(torch.cuda.current_device(), 0)
            expected = reference_gate_add(residual, update, gate, 5)
            actual = fused_residual_gate_add(residual, update, gate, 5)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(actual.device, residual.device)
            torch.cuda.synchronize(residual.device)
            self.assert_exact(actual, expected)
        finally:
            torch.cuda.set_device(previous)

    @torch.no_grad()
    def test_nondefault_stream_and_inference_mode_preserve_values(self):
        residual = channel_major(2, 15, 65)
        update = torch.randn(2, 15, 65, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(2, 3, 1, 65, device="cuda", dtype=torch.bfloat16)
        fused_residual_gate_add(residual, update, gate, 5)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(5_000_000)
            residual[:, 0, 0].add_(2)
            expected = reference_gate_add(residual, update, gate, 5)
            actual = fused_residual_gate_add(residual, update, gate, 5)
            copied = actual.clone()
        stream.synchronize()
        self.assert_exact(copied, expected)
        with torch.inference_mode():
            values = [v.clone() for v in (residual, update, gate)]
            self.assertTrue(can_fuse_residual_gate_add(*values, 5))
            self.assert_exact(fused_residual_gate_add(*values, 5), reference_gate_add(*values, 5))


if __name__ == "__main__":
    unittest.main()
