"""Regression checks for the cached, chunk-wise FlashAttention path.

Run from the repository root with the MiniWorld environment::

    CUDA_VISIBLE_DEVICES=1 python -m unittest discover -s tests -v

CUDA tests need flash-attn and a supported GPU; CPU fallback tests always run.
"""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from miniworld import miniworld as mw
from miniworld.optimization_config import OptimizationConfig


def small_model(device="cpu", head_dim=32, optimization_config=None):
    """Activate the zero-initialized gates/output so parity is not vacuous."""
    torch.manual_seed(71)
    model = mw.MiniWorldModel(
        in_channels=2,
        hidden_size=2 * head_dim,
        cond_dim=3,
        depth=2,
        num_heads=2,
        patch_size=1,
        input_size=(2, 3),
        num_frames=8,
        mlp_ratio=2,
        adaln_lora_dim=8,
        # Test semantics must not depend on a user's editable default file.
        optimization_config=(
            OptimizationConfig(framewise_conditioning=False)
            if optimization_config is None else optimization_config
        ),
    )
    with torch.no_grad():
        model.shared_mod[-1].weight.normal_(std=0.02)
        model.shared_mod[-1].bias.fill_(0.2)
        model.action_encoder.to_mod[-1].weight.normal_(std=0.02)
        model.final_layer.linear.weight.normal_(std=0.1)
        model.final_layer.adaLN_modulation[-1].weight.normal_(std=0.02)
    return model.to(device).eval()


def explicit_reference(q, k, v, chunk_tokens):
    """Build visibility independently of the implementation's mask builder."""
    n_cur, n_total = q.shape[-2], k.shape[-2]
    n_past = n_total - n_cur
    visible = torch.empty((n_cur, n_total), dtype=torch.bool, device=q.device)
    for row in range(n_cur):
        end = n_past + min((row // chunk_tokens + 1) * chunk_tokens, n_cur)
        visible[row] = torch.arange(n_total, device=q.device) < end
    return F.scaled_dot_product_attention(q, k, v, attn_mask=visible)


class FallbackTests(unittest.TestCase):
    def test_cpu_fp32_retains_dense_sdpa(self):
        model = small_model()
        x = torch.randn(2, 2, 5, 2, 3)
        t = torch.rand(2, 5)
        y = torch.randn(2, 5, 3)
        expected, expected_cache = model.forward_with_cache(
            x, t, y, chunk_size=2, return_kv=True, use_chunk_flash_attn=False
        )
        with mock.patch.object(
            mw, "_chunkwise_flash_attention", side_effect=AssertionError("CPU used FlashAttention")
        ):
            actual, actual_cache = model.forward_with_cache(
                x, t, y, chunk_size=2, return_kv=True
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertGreater(actual.std().item(), 0.01)
        self.assertEqual(actual.dtype, torch.float32)
        for actual_kv, expected_kv in zip(actual_cache, expected_cache):
            for actual_tensor, expected_tensor in zip(actual_kv, expected_kv):
                torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
                self.assertEqual(actual_tensor.dtype, torch.float32)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    "FlashAttention requires a CUDA GPU with compute capability >= 8.0",
)
class ChunkFlashAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(137)

    def test_independent_dense_reference(self):
        # Empty history, an unaligned sink, a full chunk commit, batch > 1,
        # and an incomplete final chunk all have distinct indexing risks.
        cases = [(1, 0, 12, 4), (2, 3, 10, 4), (2, 19, 4, 4), (1, 7, 1, 4)]
        for dtype in (torch.bfloat16, torch.float16):
            for batch, n_past, n_cur, chunk_tokens in cases:
                with self.subTest(dtype=dtype, batch=batch, n_past=n_past, n_cur=n_cur):
                    q = torch.randn(batch, 2, n_cur, 32, device="cuda")
                    k = torch.randn(batch, 2, n_past + n_cur, 32, device="cuda")
                    v = torch.randn_like(k, dtype=dtype)
                    with torch.autocast("cuda", dtype=dtype):
                        actual = mw._chunkwise_flash_attention(q, k, v, chunk_tokens)
                        expected = explicit_reference(q, k, v, chunk_tokens)
                    self.assertEqual(actual.shape, q.shape)
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.008)

    def test_visibility_is_chunk_causal_and_history_is_always_visible(self):
        # Zero Q/K make each visible V equiprobable, giving a direct oracle.
        n_past, n_cur, chunk_tokens = 3, 10, 4
        q = torch.zeros(1, 1, n_cur, 32, device="cuda")
        k = torch.zeros(1, 1, n_past + n_cur, 32, device="cuda")
        v = torch.zeros_like(k, dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zero = mw._chunkwise_flash_attention(q, k, v, chunk_tokens)

            # The last token of chunk 1 must affect its first query. A token-
            # causal kernel would incorrectly hide this value from that query.
            within = v.clone()
            within[:, :, n_past + chunk_tokens - 1] = 7
            actual = mw._chunkwise_flash_attention(q, k, within, chunk_tokens)
            torch.testing.assert_close(actual[:, :, :chunk_tokens], torch.ones_like(actual[:, :, :chunk_tokens]), rtol=0, atol=0)

            # Chunk 2 may not affect any query in chunk 1.
            future = v.clone()
            future[:, :, n_past + chunk_tokens :] = 64
            actual = mw._chunkwise_flash_attention(q, k, future, chunk_tokens)
            torch.testing.assert_close(actual[:, :, :chunk_tokens], zero[:, :, :chunk_tokens], rtol=0, atol=0)
            self.assertGreater(actual[:, :, chunk_tokens:].min().item(), 0)

            # A sink/history token remains visible, even though its count is
            # not a multiple of the current chunk size.
            history = v.clone()
            history[:, :, 0] = 7
            actual = mw._chunkwise_flash_attention(q, k, history, chunk_tokens)
            self.assertGreater(actual.min().item(), 0)
            expected = explicit_reference(q, k, history, chunk_tokens)
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.004)

    def test_cuda_fp32_falls_back_without_autocast(self):
        model = small_model("cuda")
        x = torch.randn(1, 2, 3, 2, 3, device="cuda")
        t = torch.rand(1, 3, device="cuda")
        expected, _ = model.forward_with_cache(x, t, chunk_size=2, use_chunk_flash_attn=False)
        with mock.patch.object(
            mw, "_chunkwise_flash_attention", side_effect=AssertionError("FP32 was downcast")
        ):
            actual, _ = model.forward_with_cache(x, t, chunk_size=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)

    def test_unsupported_head_dimension_falls_back(self):
        # FlashAttention 2 does not support dimensions above 256.
        model = small_model("cuda", head_dim=264)
        x = torch.randn(1, 2, 3, 2, 3, device="cuda")
        t = torch.rand(1, 3, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected, _ = model.forward_with_cache(x, t, chunk_size=2, use_chunk_flash_attn=False)
            with mock.patch.object(
                mw, "_chunkwise_flash_attention", side_effect=AssertionError("Unsupported head used FlashAttention")
            ):
                actual, _ = model.forward_with_cache(x, t, chunk_size=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_full_model_rope_and_cache_parity(self):
        model = small_model("cuda")
        for past_frames in (0, 1, 3):
            for explicit_empty in (False, True) if past_frames == 0 else (False,):
                with self.subTest(past_frames=past_frames, explicit_empty=explicit_empty):
                    batch, current_frames = 2, 5
                    past = None
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if past_frames:
                            seed = torch.randn(batch, 2, past_frames, 2, 3, device="cuda")
                            seed_t = torch.rand(batch, past_frames, device="cuda")
                            _, past = model.forward_with_cache(
                                seed, seed_t, chunk_size=2, return_kv=True, use_chunk_flash_attn=False
                            )
                        elif explicit_empty:
                            past = [
                                (torch.empty(batch, 2, 0, 32, device="cuda"),
                                 torch.empty(batch, 2, 0, 32, device="cuda", dtype=torch.bfloat16))
                                for _ in model.blocks
                            ]
                        snapshots = None if past is None else [(k.clone(), v.clone()) for k, v in past]
                        x = torch.randn(batch, 2, current_frames, 2, 3, device="cuda")
                        t = torch.rand(batch, current_frames, device="cuda")
                        y = torch.randn(batch, current_frames, 3, device="cuda")
                        kwargs = dict(
                            y=y, past_kv_list=past, current_position_offset=past_frames,
                            chunk_size=2, return_kv=True, cond_drop=torch.tensor([False, True], device="cuda")
                        )
                        expected, expected_cache = model.forward_with_cache(
                            x, t, use_chunk_flash_attn=False, **kwargs
                        )
                        # The accelerated path must also avoid allocating the
                        # dense mask, including when the past cache is absent.
                        with mock.patch.object(
                            mw, "_build_cached_block_causal_mask", side_effect=AssertionError("Fast path built dense mask")
                        ), mock.patch.object(
                            mw, "_chunkwise_flash_attention", wraps=mw._chunkwise_flash_attention
                        ) as flash:
                            actual, actual_cache = model.forward_with_cache(x, t, **kwargs)
                        self.assertEqual(flash.call_count, model.depth)

                    self.assertGreater(expected.std().item(), 0.01)
                    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.025)
                    for layer, (actual_kv, expected_kv) in enumerate(zip(actual_cache, expected_cache)):
                        self.assertEqual(actual_kv[0].dtype, torch.float32)
                        self.assertEqual(actual_kv[1].dtype, torch.bfloat16)
                        self.assertEqual(actual_kv[0].shape[-2], current_frames * 6)
                        for actual_tensor, expected_tensor in zip(actual_kv, expected_kv):
                            # Layer 0 projected K/V must be bitwise untouched;
                            # later layers include the attention roundoff.
                            tolerance = 0 if layer == 0 else 0.025
                            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=tolerance, atol=tolerance)
                    if past is not None:
                        for original_kv, saved_kv in zip(past, snapshots):
                            for original, saved in zip(original_kv, saved_kv):
                                torch.testing.assert_close(original, saved, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
