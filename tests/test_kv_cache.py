"""Independent dense-mask oracles for physical compact KV caches."""
import math
import unittest

import torch

from miniworld.kv_cache import CompactKVCache, GROUPS, flash_attn_with_kvcache


def groups(device, all_global=False):
    ids = {'global': [5], 'sink': [3], 'dummy': [0], 'local': [4], 'spatial': [2, 1]}
    if all_global:
        ids = {name: list(range(6)) if name == 'global' else [] for name in GROUPS}
    return {name: torch.tensor(value, device=device, dtype=torch.long) for name, value in ids.items()}


def oracle(q, current_k, current_v, past, head_groups, chunk, height=3, width=5,
           tile_height=2, tile_width=3, sink_frames=1, dummy_frames=1, local_frames=3):
    k = torch.cat((past[0], current_k), dim=2).float()
    v = torch.cat((past[1], current_v), dim=2).float()
    n_past, n_cur = past[0].shape[2], q.shape[2]
    area = height*width
    output = torch.empty_like(q, dtype=torch.float32)
    for name, heads in head_groups.items():
        mask = torch.zeros(n_cur, n_past+n_cur, device=q.device, dtype=torch.bool)
        for qi in range(n_cur):
            start = qi//chunk*chunk
            end = min(start+chunk, n_cur)
            for ki in range(n_past+end):
                visible = True
                if name == 'sink':
                    visible = ki < min(n_past, sink_frames*area) or ki >= n_past+start
                elif name in ('dummy', 'local'):
                    back = (dummy_frames if name == 'dummy' else local_frames)*area
                    visible = ki >= max(0, n_past+start-back)
                elif name == 'spatial':
                    qsp, ksp = qi%area, ki%area
                    visible = (qsp//width//tile_height, qsp%width//tile_width) == (
                        ksp//width//tile_height, ksp%width//tile_width)
                mask[qi, ki] = visible
        for head in heads.tolist():
            scores = q[:, head].float() @ k[:, head].transpose(-1, -2) / math.sqrt(q.shape[-1])
            output[:, head] = scores.masked_fill(~mask, -float('inf')).softmax(-1) @ v[:, head]
    return output


class AdditiveRope:
    def rope_shift_time(self, delta, values):
        result = values.clone()
        result[..., 0] += delta
        return result


class CompactKVTests(unittest.TestCase):
    def make_cache(self, past, head_groups, **kwargs):
        return CompactKVCache.from_dense(past, head_groups, logical_frames=past[0].shape[2]//15,
                                         height=3, width=5, chunk_size=2,
                                         tile_height=2, tile_width=3, local_frames=3, **kwargs)

    def test_cpu_visibility_partial_chunks_and_no_history(self):
        torch.manual_seed(819)
        for batch, n_past, n_cur in ((1, 0, 31), (2, 15, 67), (1, 75, 75)):
            for all_global in (False, True):
                with self.subTest(batch=batch, past=n_past, current=n_cur, global_only=all_global):
                    q = torch.randn(batch, 6, n_cur, 16)
                    k, v = torch.randn_like(q), torch.randn_like(q)
                    past = (torch.randn(batch, 6, n_past, 16), torch.randn(batch, 6, n_past, 16))
                    hg = groups('cpu', all_global)
                    cache = self.make_cache(past, hg)
                    out = cache.attention(q, k, v, 30)
                    expected = oracle(q, k, v, past, hg, 30)
                    torch.testing.assert_close(out, expected, rtol=2e-5, atol=2e-6)

    def test_append_evict_preserves_rope_and_compact_lifetimes(self):
        torch.manual_seed(42)
        hg = groups('cpu')
        past = (torch.randn(1, 6, 75, 16), torch.randn(1, 6, 75, 16))
        source_storage = {t.untyped_storage().data_ptr() for t in past}
        cache = self.make_cache(past, hg)
        for history in cache._histories.values():
            self.assertNotIn(history.k.untyped_storage().data_ptr(), source_storage)
            self.assertNotIn(history.v.untyped_storage().data_ptr(), source_storage)
        rope = AdditiveRope()
        for _ in range(3):
            new = (torch.randn(1, 6, 30, 16), torch.randn(1, 6, 30, 16))
            cache.append(*new)
            past = tuple(torch.cat((old, inc), dim=2) for old, inc in zip(past, new))
            cache.evict(2, rope)
            past = (torch.cat((past[0][:, :, :15], rope.rope_shift_time(-2, past[0][:, :, 45:])), dim=2),
                    torch.cat((past[1][:, :, :15], past[1][:, :, 45:]), dim=2))
            self.assertEqual(cache.logical_frames, 5)
            q = torch.randn(1, 6, 67, 16)
            k, v = torch.randn_like(q), torch.randn_like(q)
            expected = oracle(q, k, v, past, hg, 30)
            torch.testing.assert_close(cache.attention(q, k, v, 30), expected, rtol=2e-5, atol=2e-6)
        self.assertLess(cache.summary()['canonical_kv_bytes'], sum(t.numel()*t.element_size() for t in past))
        self.assertGreaterEqual(cache.storage_bytes(), cache.summary()['canonical_kv_bytes'])

    def test_zero_history_budgets_and_invalid_partition(self):
        past = (torch.randn(1, 6, 15, 16), torch.randn(1, 6, 15, 16))
        hg = groups('cpu')
        cache = self.make_cache(past, hg, sink_frames=0, dummy_frames=0)
        q = torch.randn(1, 6, 31, 16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        torch.testing.assert_close(cache.attention(q, k, v, 30),
                                   oracle(q, k, v, past, hg, 30, sink_frames=0, dummy_frames=0),
                                   rtol=2e-5, atol=2e-6)
        hg['sink'] = hg['dummy']
        with self.assertRaises(ValueError):
            self.make_cache(past, hg)

    def test_recent_budget_spans_pinned_sink_after_overflow(self):
        # Tail selection before eviction would lose pinned frames0,1 here.
        hg = {name: torch.tensor([0] if name == 'local' else [], dtype=torch.long) for name in GROUPS}
        past = (torch.zeros(1, 1, 4, 8), torch.arange(4.).view(1, 1, 4, 1).expand(1, 1, 4, 8).clone())
        cache = CompactKVCache.from_dense(past, hg, logical_frames=4, height=1, width=1,
                                         chunk_size=2, sink_frames=3, local_frames=4, max_frames=4)
        cache.append(torch.zeros(1, 1, 2, 8), torch.tensor([4., 5.]).view(1, 1, 2, 1).expand(1, 1, 2, 8))
        class IdentityRope:
            def rope_shift_time(self, delta, tensor): return tensor
        cache.evict(2, IdentityRope())
        current = torch.tensor([6., 7.]).view(1, 1, 2, 1).expand(1, 1, 2, 8)
        actual = cache.attention(torch.zeros_like(current), torch.zeros_like(current), current, 2)
        torch.testing.assert_close(actual, torch.full_like(actual, 3.5))

    def test_constructor_does_not_retain_autograd_graph(self):
        past = (torch.randn(1, 6, 15, 16, requires_grad=True), torch.randn(1, 6, 15, 16, requires_grad=True))
        cache = self.make_cache(past, groups('cpu'))
        self.assertTrue(all(not h.k.requires_grad and not h.v.requires_grad for h in cache._histories.values()))


@unittest.skipUnless(torch.cuda.is_available() and flash_attn_with_kvcache is not None,
                     'requires CUDA FlashAttention')
class CompactKVCudaTests(unittest.TestCase):
    make_cache = CompactKVTests.make_cache
    @torch.inference_mode()
    def test_cuda_matches_mask_and_workspace_reuse(self):
        torch.manual_seed(71)
        for n_past, n_cur in ((0, 31), (15, 75), (75, 67), (75, 75)):
            for all_global in (False, True):
                with self.subTest(past=n_past, current=n_cur, global_only=all_global):
                    q = torch.randn(2, 6, n_cur, 32, device='cuda', dtype=torch.bfloat16)
                    k, v = torch.randn_like(q), torch.randn_like(q)
                    # Canonical keys are FP32, matching MiniWorld QK norm/RoPE.
                    past = (torch.randn(2, 6, n_past, 32, device='cuda', dtype=torch.bfloat16).float(),
                            torch.randn(2, 6, n_past, 32, device='cuda', dtype=torch.bfloat16))
                    hg = groups('cuda', all_global)
                    cache = self.make_cache(past, hg)
                    expected = oracle(q, k, v, past, hg, 30)
                    actual = cache.attention(q, k.float(), v, 30)
                    torch.testing.assert_close(actual.float(), expected, rtol=.025, atol=.008)
                    pointers = {name: (w['k'].data_ptr(), w['v'].data_ptr()) for name, w in cache._workspaces.items()}
                    second = cache.attention(q, k.float(), v, 30)
                    torch.testing.assert_close(second, actual, rtol=0, atol=0)
                    self.assertEqual(pointers, {name: (w['k'].data_ptr(), w['v'].data_ptr()) for name, w in cache._workspaces.items()})
                    # No per-step mutation of canonical history.
                    for history in cache._histories.values():
                        self.assertEqual(history.k.dtype, torch.float32)

    @torch.inference_mode()
    def test_cuda_append_evict_invalidates_packed_history_and_value_alias(self):
        from miniworld.miniworld import VisionRotaryEmbeddingFast3D
        torch.manual_seed(63)
        hg = groups('cuda')
        past = (torch.randn(1, 6, 75, 32, device='cuda'),
                torch.randn(1, 6, 75, 32, device='cuda', dtype=torch.bfloat16))
        cache = self.make_cache(past, hg, max_frames=5)
        rope = VisionRotaryEmbeddingFast3D(32, 8, 3, 5).cuda()
        for n_current in (60, 75, 31):
            q = torch.randn(1, 6, n_current, 32, device='cuda', dtype=torch.bfloat16)
            k, v = torch.randn_like(q).float(), torch.randn_like(q)
            expected = oracle(q, k, v, (past[0].bfloat16(), past[1]), hg, 30)
            torch.testing.assert_close(cache.attention(q, k, v, 30).float(), expected, rtol=.025, atol=.008)
            history = cache._histories['global']
            self.assertEqual(history.v.untyped_storage().data_ptr(), cache._workspaces['global']['v'].untyped_storage().data_ptr())
            new = (torch.randn(1, 6, 30, 32, device='cuda'),
                   torch.randn(1, 6, 30, 32, device='cuda', dtype=torch.bfloat16))
            cache.append(*new).evict(2, rope)
            past = tuple(torch.cat((old, inc), dim=2) for old, inc in zip(past, new))
            past = (torch.cat((past[0][:, :, :15], rope.rope_shift_time(-2, past[0][:, :, 45:])), dim=2),
                    torch.cat((past[1][:, :, :15], past[1][:, :, 45:]), dim=2))

    @torch.inference_mode()
    def test_fused_current_pack_handles_projection_strides_and_shared_lifetime(self):
        from miniworld.kv_kernels import CurrentPackWorkspace, pack_current_qkv
        raw = torch.randn(2, 75, 3, 6, 32, device='cuda', dtype=torch.bfloat16)
        q, k, v = raw.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = q.float(), k.float()
        # V keeps the fused QKV projection's noncontiguous strides.
        order = torch.tensor([5, 3, 0, 4, 2, 1], device='cuda')
        workspace = CurrentPackWorkspace()
        packed = pack_current_qkv(q, k, v, order, workspace, torch.bfloat16)
        for actual, original in zip(packed, (q, k, v)):
            torch.testing.assert_close(actual, original.index_select(1, order).bfloat16(), rtol=0, atol=0)
        pointer = packed[0].untyped_storage().data_ptr()
        again = pack_current_qkv(q, k, v, order, workspace, torch.bfloat16)
        self.assertEqual(pointer, again[0].untyped_storage().data_ptr())
        self.assertEqual(workspace.fused_calls, 2)
        self.assertEqual(workspace.fallback_calls, 0)

        past = (torch.randn(2, 6, 75, 32, device='cuda'),
                torch.randn(2, 6, 75, 32, device='cuda', dtype=torch.bfloat16))
        hg = groups('cuda')
        a = self.make_cache(past, hg).share_current_workspace(workspace)
        b = self.make_cache(past, hg).share_current_workspace(workspace)
        first = a.attention(q, k, v, 30)
        snapshot = first.clone()
        b.attention(q+1, k+1, v+1, 30)
        # A subsequent layer's pack must not alias or change prior outputs.
        torch.testing.assert_close(first, snapshot, rtol=0, atol=0)
        a.fused_current_pack = False
        torch.testing.assert_close(first, a.attention(q, k, v, 30), rtol=0, atol=0)
        seen = set()
        total = a.storage_bytes(seen)+b.storage_bytes(seen)
        scratch_bytes = sum(t.untyped_storage().nbytes() for t in workspace.buffers.values())
        self.assertLessEqual(total, a.storage_bytes()+b.storage_bytes()-scratch_bytes)


if __name__ == '__main__':
    unittest.main()
