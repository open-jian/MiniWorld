"""Exact K cast arithmetic, layout, and cached-attention dispatch regressions.

The fused buffer belongs only to the attention computation. The returned K and
the canonical history must retain FP32 values, storage, and cache semantics.
"""

from dataclasses import replace
import unittest
from unittest import mock

import torch

from miniworld import miniworld as mw
from miniworld import fused_k_concat_cast as concat_module
from miniworld.fused_k_concat_cast import can_fuse_k_concat_cast, fused_k_concat_cast
from miniworld.optimization_config import OptimizationConfig
from test_chunk_flash_attention import small_model
from test_framewise_conditioning import inputs


def current_key(tokens, *, device="cuda"):
    return torch.randn(1, tokens, 12, 128, device=device).transpose(1, 2)


def reference(past, current):
    return torch.cat((past.to(dtype=current.dtype, device=current.device), current), dim=-2).bfloat16()


def assert_bits(test, actual, expected, *, layout=False):
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.shape, expected.shape)
    if layout:
        test.assertEqual(actual.stride(), expected.stride())
    integer = torch.int32 if actual.dtype == torch.float32 else torch.int16
    test.assertTrue(torch.equal(actual.view(integer), expected.view(integer)))


def attention_fixture(*, device="cuda", tokens=17, history=31, heads=12, head_dim=128):
    attention = mw.Attention(heads * head_dim, num_heads=heads, qkv_bias=True, qk_norm=True).to(device).eval()
    value = torch.randn(1, tokens, heads * head_dim, device=device)
    past = (torch.randn(1, heads, history, head_dim, device=device),
            torch.randn(1, heads, history, head_dim, device=device, dtype=torch.bfloat16))
    return attention, value, past


def assert_attention_exact(test, actual, expected):
    assert_bits(test, actual[0], expected[0])
    test.assertGreater(actual[0].float().std().item(), 0.01)
    for got, wanted in zip(actual[1], expected[1]):
        assert_bits(test, got, wanted, layout=True)
    test.assertEqual(actual[1][0].dtype, torch.float32)


def checkpoint_shape_model(config):
    # Retain the real head count/dimension but use one block and few video tokens
    # to verify configuration propagation without loading the 1B checkpoint.
    model = mw.MiniWorldModel(
        in_channels=2, hidden_size=1536, cond_dim=3, depth=1, num_heads=12,
        patch_size=1, input_size=(2, 3), num_frames=8, mlp_ratio=1,
        adaln_lora_dim=8, optimization_config=config,
    ).cuda().eval()
    with torch.no_grad():
        model.shared_mod[-1].weight.normal_(std=0.005)
        model.shared_mod[-1].bias.fill_(0.2)
        model.action_encoder.to_mod[-1].weight.normal_(std=0.005)
        model.final_layer.linear.weight.normal_(std=0.05)
        model.final_layer.adaLN_modulation[-1].weight.normal_(std=0.005)
    return model


class KConcatCastFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @torch.no_grad()
    def test_cpu_kernel_and_attention_fallback(self):
        past = torch.randn(1, 12, 5, 128)
        current = current_key(3, device="cpu")
        self.assertFalse(can_fuse_k_concat_cast(past, current))
        with self.assertRaises(ValueError):
            fused_k_concat_cast(past, current)
        attention, value, history = attention_fixture(device="cpu", tokens=3, history=5)
        expected = attention(value, past_kv=history, return_kv=True)
        with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("CPU dispatch")):
            actual = attention(value, past_kv=history, return_kv=True, use_fused_k_concat_cast=True)
        assert_attention_exact(self, actual, expected)

    def test_regular_training_keeps_reference_gradients_and_checkpoint_keys(self):
        config = OptimizationConfig(fused_k_concat_cast=False)
        reference_model = small_model(optimization_config=config).train()
        optimized = small_model(optimization_config=replace(config, fused_k_concat_cast=True)).train()
        optimized.load_state_dict(reference_model.state_dict(), strict=True)
        self.assertEqual(set(reference_model.state_dict()), set(optimized.state_dict()))
        values = inputs(frames=8, batch=1)
        kwargs = dict(temporal_causal=True, chunk_size=2, cond_drop=torch.tensor([False]))
        expected = reference_model(*values, **kwargs)
        with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Training dispatch")):
            actual = optimized(*values, **kwargs)
        assert_bits(self, actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for (name, parameter), (_, peer) in zip(reference_model.named_parameters(), optimized.named_parameters()):
            if parameter.grad is None:
                self.assertIsNone(peer.grad, name)
            else:
                torch.testing.assert_close(peer.grad, parameter.grad, rtol=0, atol=0, msg=name)


@unittest.skipUnless(torch.cuda.is_available(), "Requires a CUDA GPU")
class KConcatCastCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @torch.no_grad()
    def test_real_cache_shapes_preserve_bits_layout_and_inputs(self):
        torch.manual_seed(1861)
        for history_tokens, current_tokens in ((0, 1), (300, 17), (14400, 600), (14400, 4800)):
            with self.subTest(history=history_tokens, current=current_tokens):
                past = torch.randn(1, 12, history_tokens, 128, device="cuda")
                current = current_key(current_tokens)
                saved = [value.clone() for value in (past, current)]
                versions = [value._version for value in (past, current)]
                self.assertTrue(can_fuse_k_concat_cast(past, current))
                actual = fused_k_concat_cast(past, current)
                assert_bits(self, actual, reference(past, current), layout=True)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertNotEqual(actual.data_ptr(), current.data_ptr())
                self.assertEqual(versions, [value._version for value in (past, current)])
                for value, snapshot in zip((past, current), saved):
                    assert_bits(self, value, snapshot)

    @torch.no_grad()
    def test_padding_sliced_history_and_nonzero_offsets(self):
        history_storage = torch.randn(1, 12, 14437, 132, device="cuda")
        current_storage = torch.randn(1, 4827, 12, 136, device="cuda")
        past = history_storage[:, :, 19:14419, 2:130]
        current = current_storage[:, 11:4811, :, 3:131].transpose(1, 2)
        saved = [value.clone() for value in (history_storage, current_storage)]
        self.assertGreater(past.storage_offset(), 0)
        self.assertGreater(current.storage_offset(), 0)
        self.assertTrue(can_fuse_k_concat_cast(past, current))
        assert_bits(self, fused_k_concat_cast(past, current), reference(past, current), layout=True)
        for value, snapshot in zip((history_storage, current_storage), saved):
            assert_bits(self, value, snapshot)

    @torch.no_grad()
    def test_fp32_rounding_boundaries_nan_signed_zero_and_subnormals(self):
        # Every BF16 high word is paired with five lower FP32 words around the
        # nearest-even boundary. NaN payload/sign, infinity, signed-zero, and
        # subnormal patterns are compared as integer storage, never float ==.
        high = torch.arange(65536, dtype=torch.int64)
        lows = torch.tensor([0, 0x7FFF, 0x8000, 0x8001, 0xFFFF], dtype=torch.int64)
        patterns = ((high[:, None] << 16) | lows).to(torch.int32).flatten()
        values = patterns.view(torch.float32).cuda()
        self.assertEqual(values.numel(), 327680)
        tokens = (values.numel() + 1535) // 1536
        current = torch.zeros(1, 12, tokens, 128, device="cuda")
        current.flatten()[:values.numel()] = values
        past = current[:, :, 3:7]
        saved = current.clone()
        expected = reference(past, current)
        actual = fused_k_concat_cast(past, current)
        assert_bits(self, actual, expected, layout=True)
        assert_bits(self, current, saved)
        self.assertTrue(torch.isnan(expected).any().item())
        self.assertTrue((expected.view(torch.int16)[torch.isnan(expected)] == 0x7FFF).all().item())

    @torch.no_grad()
    def test_unsupported_metadata_and_missing_optional_dependency(self):
        past = torch.randn(1, 12, 31, 128, device="cuda")
        current = current_key(17)
        oversized_stride = current.as_strided(current.shape, (2**31, 128, 1536, 1))
        cases = (
            (past.bfloat16(), current), (past, current.bfloat16()),
            (past.half(), current.half()), (past.cpu(), current),
            (past, current.cpu()), (past[:, :11], current),
            (past.expand(2, -1, -1, -1), current.expand(2, -1, -1, -1)),
            (past, current[:, :, :0]), (past[..., :64], current[..., :64]),
            (past[..., ::2], current[..., ::2]), (past, oversized_stride),
            (past.to_sparse(), current), (past, current.to_sparse()),
            (torch.nn.Parameter(past), current), (past, torch.nn.Parameter(current)),
            (torch._neg_view(past), current), (past, torch._neg_view(current)),
        )
        for pair in cases:
            with self.subTest(shapes=[tuple(value.shape) for value in pair], dtypes=[value.dtype for value in pair]):
                self.assertFalse(can_fuse_k_concat_cast(*pair))
                with self.assertRaises(ValueError):
                    fused_k_concat_cast(*pair)
        with mock.patch.object(concat_module, "_has_triton", return_value=False):
            self.assertFalse(can_fuse_k_concat_cast(past, current))
            with self.assertRaises(ValueError):
                fused_k_concat_cast(past, current)
        with mock.patch.object(concat_module, "_supports_device", return_value=False):
            self.assertFalse(can_fuse_k_concat_cast(past, current))
        with mock.patch.object(torch.version, "hip", "test-rocm"):
            self.assertFalse(can_fuse_k_concat_cast(past, current))
        with mock.patch.object(concat_module, "_kernel", side_effect=RuntimeError("test compilation failure")):
            with self.assertRaisesRegex(RuntimeError, "test compilation failure"):
                fused_k_concat_cast(past, current)

    def test_nondefault_stream_and_inference_mode(self):
        stream = torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(stream):
            past = torch.randn(1, 12, 300, 128, device="cuda")
            current = current_key(600)
            actual = fused_k_concat_cast(past, current)
            expected = reference(past, current)
            done = stream.record_event()
        torch.cuda.current_stream().wait_event(done)
        assert_bits(self, actual, expected, layout=True)

    @torch.no_grad()
    def test_attention_dispatch_and_fp32_returned_cache_remain_exact(self):
        attention, value, history = attention_fixture()
        versions = [item._version for item in history]
        snapshots = [item.clone() for item in history]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = attention(value, past_kv=history, return_kv=True, cached_chunk_tokens=8)
            with mock.patch.object(mw, "fused_k_concat_cast", wraps=fused_k_concat_cast) as fused:
                actual = attention(value, past_kv=history, return_kv=True,
                                   cached_chunk_tokens=8, use_fused_k_concat_cast=True)
            self.assertEqual(fused.call_count, 1)
            assert_attention_exact(self, actual, expected)
            self.assertEqual(versions, [item._version for item in history])
            for item, snapshot in zip(history, snapshots):
                assert_bits(self, item, snapshot)
            with mock.patch.object(mw, "fused_k_concat_cast", side_effect=RuntimeError("test execution failure")):
                with self.assertRaisesRegex(RuntimeError, "test execution failure"):
                    attention(value, past_kv=history, cached_chunk_tokens=8, use_fused_k_concat_cast=True)

    @torch.no_grad()
    def test_attention_switch_history_training_and_backend_fallbacks(self):
        attention, value, history = attention_fixture()
        empty = tuple(item[:, :, :0] for item in history)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for past in (None, empty):
                expected = attention(value, past_kv=past, return_kv=True, cached_chunk_tokens=8)
                with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Empty history dispatch")):
                    actual = attention(value, past_kv=past, return_kv=True,
                                       cached_chunk_tokens=8, use_fused_k_concat_cast=True)
                assert_attention_exact(self, actual, expected)
            with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Disabled dispatch")):
                attention(value, past_kv=history, cached_chunk_tokens=8, use_fused_k_concat_cast=False)
            attention.train()
            expected = attention(value, past_kv=history, return_kv=True, cached_chunk_tokens=8)
            with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Training dispatch")):
                actual = attention(value, past_kv=history, return_kv=True,
                                   cached_chunk_tokens=8, use_fused_k_concat_cast=True)
            assert_attention_exact(self, actual, expected)
            attention.eval()
            expected = attention(value, past_kv=history, return_kv=True)
            with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("SDPA dispatch")):
                actual = attention(value, past_kv=history, return_kv=True, use_fused_k_concat_cast=True)
            assert_attention_exact(self, actual, expected)
            with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Masked dispatch")):
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    attention(value, past_kv=history, attn_mask=torch.zeros(17, 48, device="cuda"),
                              cached_chunk_tokens=8, use_fused_k_concat_cast=True)
            with mock.patch.object(concat_module, "_has_triton", return_value=False):
                with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Missing Triton dispatch")):
                    actual = attention(value, past_kv=history, return_kv=True,
                                       cached_chunk_tokens=8, use_fused_k_concat_cast=True)
                assert_attention_exact(self, actual, attention(value, past_kv=history, return_kv=True, cached_chunk_tokens=8))
        for dtype in (None, torch.float16):
            with self.subTest(autocast=dtype):
                with torch.autocast("cuda", dtype=torch.float16, enabled=dtype is not None):
                    # An uncast FP32 input uses SDPA; explicit FlashAttention
                    # supports FP16 here, but this BF16-only fusion does not.
                    kwargs = dict(past_kv=history, return_kv=True,
                                  cached_chunk_tokens=8 if dtype is not None else None)
                    expected = attention(value, **kwargs)
                    with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Precision dispatch")):
                        actual = attention(value, use_fused_k_concat_cast=True, **kwargs)
                    assert_attention_exact(self, actual, expected)

    def test_gradient_enabled_attention_retains_original_gradients(self):
        attention, value, history = attention_fixture(tokens=5, history=3)
        value.requires_grad_()
        peer = value.detach().clone().requires_grad_()
        self.assertFalse(can_fuse_k_concat_cast(history[0], current_key(5)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = attention(value, past_kv=history, cached_chunk_tokens=3)
        # Separate autocast contexts avoid sharing the parameter-cast autograd
        # graph between the two independently backpropagated reference calls.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Gradient dispatch")):
                actual = attention(peer, past_kv=history, cached_chunk_tokens=3, use_fused_k_concat_cast=True)
        assert_bits(self, actual, expected)
        expected.float().square().mean().backward()
        expected_gradients = {name: parameter.grad.clone() for name, parameter in attention.named_parameters()}
        attention.zero_grad(set_to_none=True)
        actual.float().square().mean().backward()
        torch.testing.assert_close(peer.grad, value.grad, rtol=0, atol=0)
        for name, parameter in attention.named_parameters():
            torch.testing.assert_close(parameter.grad, expected_gradients[name], rtol=0, atol=0, msg=name)

    @torch.no_grad()
    def test_model_config_reaches_cached_attention_and_is_independent(self):
        torch.manual_seed(1862)
        config = OptimizationConfig(fused_k_concat_cast=False)
        reference_model = checkpoint_shape_model(config)
        optimized = checkpoint_shape_model(replace(config, fused_k_concat_cast=True))
        optimized.load_state_dict(reference_model.state_dict(), strict=True)
        self.assertEqual(set(reference_model.state_dict()), set(optimized.state_dict()))
        self.assertTrue(OptimizationConfig().fused_k_concat_cast)
        values = inputs(frames=8, device="cuda", batch=1)
        prefix = tuple(value[:, :, :2] if index == 0 else value[:, :2]
                       for index, value in enumerate(values))
        current = tuple(value[:, :, 2:5] if index == 0 else value[:, 2:5]
                        for index, value in enumerate(values))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, history = reference_model.forward_with_cache(*prefix, return_kv=True, chunk_size=2)
            kwargs = dict(past_kv_list=history, current_position_offset=2, return_kv=True, chunk_size=2)
            expected = reference_model.forward_with_cache(*current, **kwargs)
            with mock.patch.object(mw, "fused_k_concat_cast", wraps=fused_k_concat_cast) as fused:
                actual = optimized.forward_with_cache(*current, **kwargs)
            self.assertEqual(fused.call_count, optimized.depth)
            assert_bits(self, actual[0], expected[0])
            self.assertGreater(actual[0].std().item(), 0.01)
            for pair, saved in zip(actual[1], expected[1]):
                for item, wanted in zip(pair, saved):
                    assert_bits(self, item, wanted)
                self.assertEqual(pair[0].dtype, torch.float32)
            # The ordinary forward has no temporary cached K concatenation.
            with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Ordinary forward dispatch")):
                assert_bits(self, optimized(*values, temporal_causal=True, chunk_size=2),
                            reference_model(*values, temporal_causal=True, chunk_size=2))
            # Disabling the attention backend must also disable this cast fusion.
            expected_dense = reference_model.forward_with_cache(*current, use_chunk_flash_attn=False, **kwargs)
            with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Dense model dispatch")):
                actual_dense = optimized.forward_with_cache(*current, use_chunk_flash_attn=False, **kwargs)
            assert_bits(self, actual_dense[0], expected_dense[0])
            # Existing legacy custom methods must not receive the new keyword.
            original = optimized.blocks[0].attn.forward
            def legacy(value, rope=None, attn_mask=None, past_kv=None, return_kv=False, cached_chunk_tokens=None):
                return original(value, rope=rope, attn_mask=attn_mask, past_kv=past_kv,
                                return_kv=return_kv, cached_chunk_tokens=cached_chunk_tokens)
            with mock.patch.object(optimized.blocks[0].attn, "forward", new=legacy):
                with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Custom forward dispatch")):
                    actual = optimized.forward_with_cache(*current, **kwargs)
            assert_bits(self, actual[0], expected[0])
            for observed_module in (optimized.blocks[0], optimized.blocks[0].attn):
                observed_kwargs = []
                handle = observed_module.register_forward_pre_hook(
                    lambda _module, _args, call_kwargs: observed_kwargs.append(dict(call_kwargs)),
                    with_kwargs=True,
                )
                try:
                    with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Hooked forward dispatch")):
                        actual = optimized.forward_with_cache(*current, **kwargs)
                finally:
                    handle.remove()
                self.assertEqual(len(observed_kwargs), 1)
                self.assertNotIn("use_fused_k_concat_cast", observed_kwargs[0])
                assert_bits(self, actual[0], expected[0])
            original_block = optimized.blocks[0].forward
            def legacy_block(value, emb, shared_mod, pose_mod, feat_rope=None,
                             attn_mask=None, past_kv=None, return_kv=False,
                             cached_chunk_tokens=None, conditioning_tokens_per_frame=None,
                             use_fused_norm_modulation=False, use_fused_residual_gate_add=False,
                             use_fused_bias_swiglu=False):
                return original_block(
                    value, emb, shared_mod, pose_mod, feat_rope, attn_mask, past_kv, return_kv,
                    cached_chunk_tokens=cached_chunk_tokens,
                    conditioning_tokens_per_frame=conditioning_tokens_per_frame,
                    use_fused_norm_modulation=use_fused_norm_modulation,
                    use_fused_residual_gate_add=use_fused_residual_gate_add,
                    use_fused_bias_swiglu=use_fused_bias_swiglu,
                )
            with mock.patch.object(optimized.blocks[0], "forward", new=legacy_block):
                with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Custom block dispatch")):
                    actual = optimized.forward_with_cache(*current, **kwargs)
            assert_bits(self, actual[0], expected[0])
            # Class replacements also predate the newly introduced keyword.
            attention_forward = mw.Attention.forward
            def legacy_class_attention(module, value, rope=None, attn_mask=None,
                                       past_kv=None, return_kv=False, cached_chunk_tokens=None):
                return attention_forward(module, value, rope=rope, attn_mask=attn_mask,
                                         past_kv=past_kv, return_kv=return_kv,
                                         cached_chunk_tokens=cached_chunk_tokens)
            block_forward = mw.MiniWorldBlock.forward
            def legacy_class_block(module, *args, **call_kwargs):
                self.assertNotIn("use_fused_k_concat_cast", call_kwargs)
                return block_forward(module, *args, **call_kwargs)
            for cls, method in ((mw.Attention, legacy_class_attention),
                                (mw.MiniWorldBlock, legacy_class_block)):
                with mock.patch.object(cls, "forward", new=method):
                    with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Class override dispatch")):
                        actual = optimized.forward_with_cache(*current, **kwargs)
                assert_bits(self, actual[0], expected[0])
            class LegacyAttention(mw.Attention):
                def forward(self, value, rope=None, attn_mask=None, past_kv=None,
                            return_kv=False, cached_chunk_tokens=None):
                    return super().forward(value, rope=rope, attn_mask=attn_mask,
                                           past_kv=past_kv, return_kv=return_kv,
                                           cached_chunk_tokens=cached_chunk_tokens)
            original_attention = optimized.blocks[0].attn
            subclass_attention = LegacyAttention(1536, num_heads=12, qkv_bias=True, qk_norm=True).cuda().eval()
            subclass_attention.load_state_dict(original_attention.state_dict(), strict=True)
            optimized.blocks[0].attn = subclass_attention
            try:
                with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Attention subclass dispatch")):
                    actual = optimized.forward_with_cache(*current, **kwargs)
            finally:
                optimized.blocks[0].attn = original_attention
            assert_bits(self, actual[0], expected[0])
            class LegacyBlock(mw.MiniWorldBlock):
                def forward(self, *args, **call_kwargs):
                    if "use_fused_k_concat_cast" in call_kwargs:
                        raise AssertionError("New keyword reached a legacy block subclass")
                    return super().forward(*args, **call_kwargs)
            original_module = optimized.blocks[0]
            subclass_block = LegacyBlock(1536, 12, "adaln_lora", mlp_ratio=1,
                                         use_qknorm=True, lora_dim=8).cuda().eval()
            subclass_block.load_state_dict(original_module.state_dict(), strict=True)
            optimized.blocks[0] = subclass_block
            try:
                with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Block subclass dispatch")):
                    actual = optimized.forward_with_cache(*current, **kwargs)
            finally:
                optimized.blocks[0] = original_module
            assert_bits(self, actual[0], expected[0])
            # Global observers also see the old call signature. Compute both
            # paths under the hook because other accepted fusions independently
            # fall back when a global observer is installed.
            global_kwargs = []
            def observe(module, _args, call_kwargs, _output):
                if module is optimized.blocks[0] or module is optimized.blocks[0].attn:
                    global_kwargs.append(dict(call_kwargs))
            handle = torch.nn.modules.module.register_module_forward_hook(observe, with_kwargs=True)
            try:
                expected_hooked = reference_model.forward_with_cache(*current, **kwargs)
                with mock.patch.object(mw, "fused_k_concat_cast", side_effect=AssertionError("Global hook dispatch")):
                    actual = optimized.forward_with_cache(*current, **kwargs)
            finally:
                handle.remove()
            self.assertEqual(len(global_kwargs), 2)
            self.assertTrue(all("use_fused_k_concat_cast" not in item for item in global_kwargs))
            assert_bits(self, actual[0], expected_hooked[0])


if __name__ == "__main__":
    unittest.main()
