"""Independent visibility checks for HeadCast's inference attention paths.

CPU checks use an explicit token oracle; CUDA checks compare grouped
FlashAttention against that same oracle, including irregular tile edges.
"""

import math
import unittest

import torch

import miniworld.headcast_attention as headcast_attention
from miniworld.headcast_attention import SpatialLayout, grouped_attention, spatial_history_mask


def explicit_mask(n_current, n_past, chunk_tokens, height, width,
                  tile_height, tile_width, mode, sink_tokens, recent_tokens):
    """Build visibility without calling any production layout/mask helper."""
    tokens_per_frame = height * width
    visible = torch.zeros(n_current, n_past + n_current, dtype=torch.bool)
    for query in range(n_current):
        end = min((query // chunk_tokens + 1) * chunk_tokens, n_current)
        visible[query, n_past:n_past + end] = True
        q_position = query % tokens_per_frame
        q_tile = ((q_position // width) // tile_height,
                  (q_position % width) // tile_width)
        for key in range(n_past):
            if mode == "global":
                visible[query, key] = True
            elif mode == "temporal":
                visible[query, key] = key < sink_tokens or key >= n_past - recent_tokens
            elif mode == "spatial":
                k_position = key % tokens_per_frame
                k_tile = ((k_position // width) // tile_height,
                          (k_position % width) // tile_width)
                visible[query, key] = q_tile == k_tile
            else:
                raise ValueError(mode)
    return visible


def explicit_attention(q, k_current, v_current, past, chunk_tokens, groups,
                       height, width, tile_height, tile_width, sink_tokens, recent_tokens):
    n_past = 0 if past is None else past[0].shape[-2]
    k = k_current if past is None else torch.cat((past[0], k_current), dim=-2)
    v = v_current if past is None else torch.cat((past[1], v_current), dim=-2)
    out = torch.empty_like(q, dtype=torch.float32)
    with torch.autocast(q.device.type, enabled=False):
        for mode, head_ids in groups.items():
            mask = explicit_mask(
                q.shape[-2], n_past, chunk_tokens, height, width,
                tile_height, tile_width, mode, sink_tokens, recent_tokens,
            ).to(q.device)
            for head in head_ids.tolist():
                logits = q[:, head].float() @ k[:, head].float().transpose(-1, -2)
                logits = logits / math.sqrt(q.shape[-1])
                probabilities = logits.masked_fill(~mask, -float("inf")).softmax(-1)
                out[:, head] = probabilities @ v[:, head].float()
    return out


def make_groups(device):
    # Deliberately interleave and reverse head IDs to exercise output placement.
    return {name: torch.tensor(ids, dtype=torch.long, device=device) for name, ids in (
        ("global", [2, 0]), ("temporal", [4, 1]), ("spatial", [5, 3]),
    )}


class HeadCastMaskTests(unittest.TestCase):
    def test_nondivisible_tile_edges_and_spatial_proxy(self):
        height, width, tile_height, tile_width = 3, 5, 2, 3
        layout = SpatialLayout(height, width, tile_height, tile_width, torch.device("cpu"))
        expected_ids = [
            (row // tile_height) * math.ceil(width / tile_width) + column // tile_width
            for row in range(height) for column in range(width)
        ]
        self.assertEqual(layout.tokens_per_frame, 15)
        self.assertEqual(layout.tile_ids.tolist(), expected_ids)

        # All queries belong to chunk 1; its whole legal in-flight prefix is
        # retained, including current tokens outside each query's spatial tile.
        queries = torch.tensor([30, 34, 39, 44, 47, 59])
        n_past, visible_end = 45, 60
        actual = spatial_history_mask(queries, n_past, visible_end, layout)
        expected = explicit_mask(
            60, n_past, 30, height, width, tile_height, tile_width,
            "spatial", 15, 30,
        )[queries]
        self.assertEqual(actual.dtype, torch.bool)
        self.assertEqual(tuple(actual.shape), (len(queries), n_past + visible_end))
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        self.assertTrue(bool(actual[:, n_past:].all()))

    def test_cpu_grouped_attention_matches_independent_oracle(self):
        torch.manual_seed(935)
        device = torch.device("cpu")
        layout = SpatialLayout(3, 5, 2, 3, device)
        groups = make_groups(device)
        for batch, n_past, n_current in ((1, 0, 31), (2, 15, 67), (2, 15, 75), (1, 45, 30)):
            with self.subTest(batch=batch, n_past=n_past, n_current=n_current):
                q = torch.randn(batch, 6, n_current, 32)
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                past = None if not n_past else (
                    torch.randn(batch, 6, n_past, 32),
                    torch.randn(batch, 6, n_past, 32),
                )
                snapshots = None if past is None else tuple(t.clone() for t in past)
                actual = grouped_attention(q, k, v, past, 30, groups, layout, 15, 30)
                expected = explicit_attention(q, k, v, past, 30, groups, 3, 5, 2, 3, 15, 30)
                self.assertEqual(tuple(actual.shape), tuple(q.shape))
                self.assertTrue(bool(torch.isfinite(actual).all()))
                torch.testing.assert_close(actual.float(), expected, rtol=2e-5, atol=2e-6)
                if snapshots is not None:
                    for original, snapshot in zip(past, snapshots):
                        torch.testing.assert_close(original, snapshot, rtol=0, atol=0)

    def test_future_chunks_cannot_leak_and_legal_prefix_is_not_spatially_pruned(self):
        device = torch.device("cpu")
        layout = SpatialLayout(3, 5, 2, 3, device)
        groups = make_groups(device)
        q = torch.zeros(1, 6, 67, 32)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        past = (torch.zeros(1, 6, 45, 32), torch.zeros(1, 6, 45, 32))

        # A large future value must never reach either of the first two chunks.
        future = v.clone()
        future[:, :, 60:] = 1000
        actual = grouped_attention(q, k, future, past, 30, groups, layout, 15, 15)
        torch.testing.assert_close(actual[:, :, :60], torch.zeros_like(actual[:, :, :60]), rtol=0, atol=0)
        self.assertGreater(actual[:, :, 60:].min().item(), 0)

        # Token 29 is in another spatial tile than query 0, but is in the same
        # two-frame chunk and must remain visible even on the spatial path.
        within = v.clone()
        within[:, :, 29] = 100
        actual = grouped_attention(q, k, within, past, 30, groups, layout, 15, 15)
        self.assertGreater(actual[:, :, 0].min().item(), 0)
        # Earlier in-flight chunks must also remain visible to later chunks.
        self.assertGreater(actual[:, :, 60].min().item(), 0)
        expected = explicit_attention(q, k, within, past, 30, groups, 3, 5, 2, 3, 15, 15)
        torch.testing.assert_close(actual.float(), expected, rtol=2e-5, atol=2e-6)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    "CUDA FlashAttention checks require compute capability >= 8.0",
)
class HeadCastCudaTests(unittest.TestCase):
    @torch.inference_mode()
    def test_repeated_cache_indices_use_each_query_chunks_prefix(self):
        if headcast_attention.flash_attn_with_kvcache is None:
            self.skipTest("FlashAttention KV-cache API is unavailable")
        q = torch.zeros(2, 2, 10, 32, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(2, 2, 12, 32, device="cuda", dtype=torch.bfloat16)
        v = (torch.arange(12, device="cuda", dtype=torch.bfloat16)[None, None, :, None]
             .expand_as(k).clone())
        v[1] += 100
        snapshots = tuple(t.clone() for t in (k, v))

        # Each cache batch is reused by two full query chunks with different
        # prefix lengths; a final short chunk sees all twelve keys. Zero logits
        # make their outputs the independently known prefix means.
        actual = headcast_attention._prefix_attention(q, k, v, 4, (5, 9, 12), True)
        expected = (torch.tensor([2.] * 4 + [4.] * 4 + [5.5] * 2, device="cuda")
                    [None, None, :, None].expand_as(actual).clone())
        expected[1] += 100
        torch.testing.assert_close(actual.float(), expected, rtol=0, atol=0)
        for source, snapshot in zip((k, v), snapshots):
            torch.testing.assert_close(source, snapshot, rtol=0, atol=0)

    @torch.inference_mode()
    def test_batched_bf16_partial_chunks_and_all_global_reference(self):
        torch.manual_seed(1729)
        device = torch.device("cuda")
        layout = SpatialLayout(3, 5, 2, 3, device)
        for dtype in (torch.bfloat16, torch.float16):
            for n_past in (0, 15, 45):
                # 75 is a partial two-frame chunk, but has complete spatial
                # frames and therefore exercises the deployed tiled kernel.
                # 67 additionally tests the sub-frame reference fallback.
                for n_current in (75, 67):
                    with self.subTest(dtype=dtype, n_past=n_past, n_current=n_current):
                        q = torch.randn(2, 6, n_current, 32, device=device, dtype=dtype)
                        k, v = torch.randn_like(q), torch.randn_like(q)
                        past = None if not n_past else (
                            torch.randn(2, 6, n_past, 32, device=device, dtype=dtype),
                            torch.randn(2, 6, n_past, 32, device=device, dtype=dtype),
                        )
                        snapshots = None if past is None else tuple(t.clone() for t in past)
                        for groups in (make_groups(device), {
                            "global": torch.arange(6, device=device),
                            "temporal": torch.empty(0, dtype=torch.long, device=device),
                            "spatial": torch.empty(0, dtype=torch.long, device=device),
                        }):
                            actual = grouped_attention(q, k, v, past, 30, groups, layout, 15, 30)
                            expected = explicit_attention(q, k, v, past, 30, groups, 3, 5, 2, 3, 15, 30)
                            self.assertEqual(actual.dtype, dtype)
                            torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.008)
                            if snapshots is not None:
                                for source, snapshot in zip(past, snapshots):
                                    torch.testing.assert_close(source, snapshot, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
