"""Independent dense-position oracles for the physical Forcing-KV cache."""
import math
import unittest

import torch

from miniworld.forcing_kv_cache import ForcingKVCache, flash_attn_with_kvcache
from miniworld.kv_kernels import CurrentPackWorkspace


AREA, SEGMENTS = 12, 3


def groups(device='cpu', all_static=False, all_dynamic=False):
    values = {'static': [2, 0], 'dynamic': [3, 1]}
    if all_static:
        values = {'static': [0, 1, 2, 3], 'dynamic': []}
    if all_dynamic:
        values = {'static': [], 'dynamic': [0, 1, 2, 3]}
    return {name: torch.tensor(ids, device=device, dtype=torch.long) for name, ids in values.items()}


def expected_selection(k, head_groups, frames, boundary, retention=.5, area=AREA, segments=SEGMENTS):
    """Direct per-token cosine loops, independent of cache reshape/gathers."""
    heads = head_groups['dynamic'].tolist() or list(range(k.shape[1]))
    chain = list(frames)+[boundary]
    scores = []
    for i, frame in enumerate(frames):
        for segment in range(segments):
            batch_scores = []
            for batch in range(k.shape[0]):
                token_scores = []
                for token in range(segment*area//segments, (segment+1)*area//segments):
                    a = k[batch, heads, frame*area+token].float().flatten()
                    b = k[batch, heads, chain[i+1]*area+token].float().flatten()
                    token_scores.append((a*b).sum()/max(float(a.norm()*b.norm()), 1e-16))
                batch_scores.append(torch.stack(token_scores).mean())
            scores.append(torch.stack(batch_scores).min())
    keep = round(len(scores)*retention)
    chosen = sorted(torch.stack(scores).topk(keep, largest=False).indices.tolist()) if scores else []
    return tuple(frames[index//segments]*area+(index%segments)*area//segments+offset
                 for index in chosen for offset in range(area//segments))


def oracle(q, current_k, current_v, dense_past, head_groups, selected, chunk=24, sink=1, area=AREA):
    """Dense all-token attention with explicit masks, never using cache data."""
    k = torch.cat((dense_past[0], current_k), dim=2).float()
    v = torch.cat((dense_past[1], current_v), dim=2).float()
    past, n = dense_past[0].shape[2], q.shape[2]
    result = torch.empty_like(q, dtype=torch.float32)
    sink_end = min(past+n, sink*area)
    selected = set(selected)
    for name, heads in head_groups.items():
        mask = torch.zeros(n, past+n, device=q.device, dtype=torch.bool)
        for qi in range(n):
            start = qi//chunk*chunk
            end = min(n, start+chunk)
            for ki in range(past+end):
                if name == 'static':
                    visible = ki < sink_end or ki >= max(0, past+start-area)
                else:
                    visible = ki < sink_end or ki in selected or ki >= max(0, past-area)
                mask[qi, ki] = visible
        for head in heads.tolist():
            scores = q[:, head].float() @ k[:, head].transpose(-1, -2)/math.sqrt(q.shape[-1])
            result[:, head] = scores.masked_fill(~mask, -float('inf')).softmax(-1) @ v[:, head]
    return result


class AdditiveRope:
    def rope_shift_time(self, delta, values):
        result = values.clone()
        result[..., 0] += delta
        return result


class ForcingCacheTests(unittest.TestCase):
    def make_cache(self, past, head_groups=None, state=None, selector=True, **kwargs):
        return ForcingKVCache.from_dense(
            past, head_groups if head_groups is not None else groups(past[0].device),
            logical_frames=past[0].shape[2]//AREA, height=3, width=4, chunk_size=2,
            segments=SEGMENTS, retention=kwargs.pop('retention', .5),
            selection_state={} if state is None else state, selector_layer=selector, **kwargs)

    def test_independent_selection_and_dense_attention(self):
        torch.manual_seed(17)
        for batch, frames, n in ((1, 0, 31), (2, 1, 49), (2, 6, 60), (1, 6, 31)):
            for all_static, all_dynamic in ((False, False), (True, False), (False, True)):
                with self.subTest(batch=batch, history=frames, current=n, static=all_static, dynamic=all_dynamic):
                    past = tuple(torch.randn(batch, 4, frames*AREA, 8) for _ in range(2))
                    hg = groups(all_static=all_static, all_dynamic=all_dynamic)
                    cache = self.make_cache(past, hg)
                    candidates = list(range(max(1, frames-3), max(1, frames-1)))
                    selected = expected_selection(past[0], hg, candidates, frames-1)
                    if not all_static:
                        self.assertEqual(cache._selected_tokens, selected)
                    q, k, v = (torch.randn(batch, 4, n, 8) for _ in range(3))
                    torch.testing.assert_close(cache.attention(q, k, v, 24),
                                               oracle(q, k, v, past, hg, selected), rtol=2e-5, atol=2e-6)

    def test_append_replaces_segments_and_evict_shifts_only_survivors(self):
        torch.manual_seed(72)
        past = tuple(torch.randn(2, 4, 6*AREA, 8) for _ in range(2))
        hg = groups()
        cache = self.make_cache(past, hg, max_frames=6)
        rope = AdditiveRope()
        for append_frames in (2, 2, 1):
            old_frames = past[0].shape[2]//AREA
            new = tuple(torch.randn(2, 4, append_frames*AREA, 8) for _ in range(2))
            past = tuple(torch.cat((old, inc), dim=2) for old, inc in zip(past, new))
            selected = expected_selection(past[0], hg, list(range(old_frames-1, old_frames+append_frames-1)),
                                          old_frames+append_frames-1)
            cache.append(*new)
            self.assertEqual(cache._selected_tokens, selected)
            self.assertTrue(all(t >= (old_frames-1)*AREA for t in selected))
            cache.evict(append_frames, rope)
            past = (torch.cat((past[0][:, :, :AREA], rope.rope_shift_time(-append_frames, past[0][:, :, (append_frames+1)*AREA:])), dim=2),
                    torch.cat((past[1][:, :, :AREA], past[1][:, :, (append_frames+1)*AREA:]), dim=2))
            selected = tuple(t-append_frames*AREA for t in selected)
            self.assertEqual(cache.logical_frames, 6)
            self.assertEqual(cache._selected_tokens, selected)
            for history in cache._histories.values():
                expected_k = past[0].index_select(1, history.heads)[:, :, history.tokens]
                torch.testing.assert_close(history.k, expected_k, rtol=0, atol=0)
            q, k, v = (torch.randn(2, 4, 31, 8) for _ in range(3))
            torch.testing.assert_close(cache.attention(q, k, v, 24), oracle(q, k, v, past, hg, selected),
                                       rtol=2e-5, atol=2e-6)
        self.assertEqual(cache.summary()['selection_updates'], 4)
        self.assertLess(cache.summary()['canonical_kv_bytes'], sum(t.numel()*t.element_size() for t in past))

    def test_layer_indices_shared_but_kv_and_cfg_independent(self):
        torch.manual_seed(94)
        a = tuple(torch.randn(2, 4, 6*AREA, 8) for _ in range(2))
        b = tuple(torch.randn_like(t) for t in a)
        state = {}
        proxy = self.make_cache(a, state=state)
        other = self.make_cache(b, state=state, selector=False)
        unrelated = self.make_cache(b)
        self.assertEqual(proxy._selected_tokens, other._selected_tokens)
        self.assertIsNot(proxy.selection_state, unrelated.selection_state)
        for history in other._histories.values():
            torch.testing.assert_close(history.k, b[0].index_select(1, history.heads)[:, :, history.tokens])
        new_a = tuple(torch.randn(2, 4, 2*AREA, 8) for _ in range(2))
        new_b = tuple(torch.randn_like(t) for t in new_a)
        with self.assertRaisesRegex(RuntimeError, 'selector layer'):
            other.append(*new_b)
        proxy.append(*new_a)
        other.append(*new_b)
        self.assertEqual(proxy._selected_tokens, other._selected_tokens)
        self.assertEqual(unrelated.summary()['selection_updates'], 1)
        with self.assertRaisesRegex(RuntimeError, 'selector layer'):
            self.make_cache(b, state={}, selector=False)

    def test_static_proxy_and_empty_append_sink_deduplication(self):
        torch.manual_seed(89)
        empty = (torch.empty(1, 4, 0, 8), torch.empty(1, 4, 0, 8))
        state = {}
        proxy = self.make_cache(empty, groups(all_static=True), state=state)
        other = self.make_cache(empty, state=state, selector=False)
        full = empty
        for n in (1, 2, 1):
            new = tuple(torch.randn(1, 4, n*AREA, 8) for _ in range(2))
            full = tuple(torch.cat((old, inc), dim=2) for old, inc in zip(full, new))
            proxy.append(*new)
            other.append(*new)
            for cache in (proxy, other):
                for history in cache._histories.values():
                    self.assertEqual(len(history.tokens), len(set(history.tokens)))
                    torch.testing.assert_close(history.k, full[0].index_select(1, history.heads)[:, :, history.tokens])
            q, k, v = (torch.randn(1, 4, 49, 8) for _ in range(3))
            torch.testing.assert_close(other.attention(q, k, v, 24),
                                       oracle(q, k, v, full, groups(), other._selected_tokens), rtol=2e-5, atol=2e-6)
        self.assertTrue(proxy.summary()['proxy_all_heads'])
        self.assertEqual(proxy.summary()['proxy_extra_bytes'], 1*4*AREA*8*4)

    def test_no_graph_dense_alias_or_future_leak_and_zero_retention(self):
        torch.manual_seed(78)
        past = tuple(torch.randn(1, 4, 6*AREA, 8, requires_grad=True) for _ in range(2))
        cache = self.make_cache(past, retention=0)
        source = {t.untyped_storage().data_ptr() for t in past}
        for history in cache._histories.values():
            self.assertFalse(history.k.requires_grad or history.v.requires_grad)
            self.assertNotIn(history.k.untyped_storage().data_ptr(), source)
            self.assertNotIn(history.v.untyped_storage().data_ptr(), source)
        self.assertEqual(cache._selected_tokens, ())
        q, k, v = (torch.randn(1, 4, 60, 8) for _ in range(3))
        before = cache.attention(q, k, v, 24)
        k[:, :, 24:] += 100
        v[:, :, 24:] -= 100
        after = cache.attention(q, k, v, 24)
        torch.testing.assert_close(before[:, :, :24], after[:, :, :24], rtol=0, atol=0)
        self.assertEqual(cache.summary()['selection_updates'], 1)

    def test_initial_inflight_sink_survives_for_later_chunks(self):
        empty = (torch.empty(1, 4, 0, 8), torch.empty(1, 4, 0, 8))
        cache = self.make_cache(empty, groups(all_static=True))
        values = torch.arange(6.).repeat_interleave(AREA).view(1, 1, 6*AREA, 1).expand(1, 4, 6*AREA, 8)
        result = cache.attention(torch.zeros_like(values), torch.zeros_like(values), values, 24)
        # Third chunk sees frame0 + predecessor3 + own frames4,5 exactly once.
        torch.testing.assert_close(result[:, :, 4*AREA:], torch.full_like(result[:, :, 4*AREA:], 3.0))


@unittest.skipUnless(torch.cuda.is_available() and flash_attn_with_kvcache is not None, 'requires CUDA FlashAttention')
class ForcingCudaTests(unittest.TestCase):
    make_cache = ForcingCacheTests.make_cache

    @torch.inference_mode()
    def test_flash_matches_dense_masks_and_reuses_workspace(self):
        torch.manual_seed(39)
        for frames, n in ((0, 31), (1, 49), (6, 60), (6, 31)):
            for all_static, all_dynamic in ((False, False), (True, False), (False, True)):
                with self.subTest(history=frames, current=n, static=all_static, dynamic=all_dynamic):
                    past = (torch.randn(2, 4, frames*AREA, 32, device='cuda'),
                            torch.randn(2, 4, frames*AREA, 32, device='cuda', dtype=torch.bfloat16))
                    hg = groups('cuda', all_static, all_dynamic)
                    cache = self.make_cache(past, hg)
                    q, k, v = (torch.randn(2, 4, n, 32, device='cuda', dtype=torch.bfloat16) for _ in range(3))
                    expected = oracle(q, k, v, (past[0].bfloat16(), past[1]), hg, cache._selected_tokens)
                    actual = cache.attention(q, k.float(), v, 24)
                    torch.testing.assert_close(actual.float(), expected, rtol=.025, atol=.008)
                    ptrs = {name: (w['k'].data_ptr(), w['v'].data_ptr()) for name, w in cache._workspaces.items()}
                    again = cache.attention(q, k.float(), v, 24)
                    torch.testing.assert_close(again, actual, rtol=0, atol=0)
                    self.assertEqual(ptrs, {name: (w['k'].data_ptr(), w['v'].data_ptr()) for name, w in cache._workspaces.items()})
                    self.assertEqual(cache.summary()['selection_updates'], 1)
                    self.assertTrue(all(h.k.dtype == torch.float32 for h in cache._histories.values()))

    @torch.inference_mode()
    def test_flash_sparse_eviction_and_request_scratch_lifetimes(self):
        from miniworld.miniworld import VisionRotaryEmbeddingFast3D
        torch.manual_seed(90)
        past = (torch.randn(1, 4, 6*AREA, 32, device='cuda'),
                torch.randn(1, 4, 6*AREA, 32, device='cuda', dtype=torch.bfloat16))
        workspace = CurrentPackWorkspace()
        a = self.make_cache(past, max_frames=6).share_current_workspace(workspace)
        b = self.make_cache(past).share_current_workspace(workspace)
        rope = VisionRotaryEmbeddingFast3D(32, 8, 3, 4).cuda()
        for _ in range(2):
            q = torch.randn(1, 4, 49, 32, device='cuda', dtype=torch.bfloat16)
            k, v = torch.randn_like(q).float(), torch.randn_like(q)
            expected = oracle(q, k, v, (past[0].bfloat16(), past[1]), groups('cuda'), a._selected_tokens)
            actual = a.attention(q, k, v, 24)
            torch.testing.assert_close(actual.float(), expected, rtol=.025, atol=.008)
            snapshot = actual.clone()
            b.attention(q+1, k+1, v+1, 24)
            torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)
            new = (torch.randn(1, 4, 2*AREA, 32, device='cuda'),
                   torch.randn(1, 4, 2*AREA, 32, device='cuda', dtype=torch.bfloat16))
            a.append(*new).evict(2, rope)
            past = tuple(torch.cat((old, inc), dim=2) for old, inc in zip(past, new))
            past = (torch.cat((past[0][:, :, :AREA], rope.rope_shift_time(-2, past[0][:, :, 3*AREA:])), dim=2),
                    torch.cat((past[1][:, :, :AREA], past[1][:, :, 3*AREA:]), dim=2))
        seen = set()
        self.assertLess(a.storage_bytes(seen)+b.storage_bytes(seen), a.storage_bytes()+b.storage_bytes())


if __name__ == '__main__':
    unittest.main()
