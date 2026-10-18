"""Separate diagnostic timings for the two KV storage paths on episode 002352.

Instrumentation is deliberately excluded from the E2E benchmark. Event regions
include CPU submission gaps and overlap as documented in the JSON report; do
not sum inclusive model/attention/RoPE regions with their children.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from unittest import mock

import torch
import benchmark_episode_chunk_flash as bench
from profile_episode_chunk_flash import aggregate, write_csv
from miniworld import miniworld as mw
from miniworld.denoiser import DiffusionForcingDenoiser
from miniworld.kv_cache import ResidentKVCache


class Recorder:
    def __init__(self, model, name):
        self.model, self.name = model, name
        self.events = []
        self.context = {}
        self.layer = None
        self.cast_index = None
        self.forward_id = 0
        self.stack = ExitStack()

    def timed(self, category, fn, *args, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        metadata = dict(self.context, layer=self.layer)
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        self.events.append((category, metadata, start, end))
        return result

    def __enter__(self):
        original_forward = self.model.net.forward_with_cache
        original_attention = mw.Attention.forward
        original_flash = mw._chunkwise_flash_attention
        original_cat, original_to = torch.cat, torch.Tensor.to
        layers = {id(block.attn): idx for idx, block in enumerate(self.model.net.blocks)}

        def forward(*args, **kwargs):
            cache = kwargs.get('resident_cache')
            if cache is not None:
                past = cache.length
            else:
                first = next((kv for kv in kwargs.get('past_kv_list') or [] if kv is not None), None)
                past = 0 if first is None else first[0].shape[-2]
            self.context = dict(forward_id=self.forward_id,
                                role='commit' if kwargs.get('return_kv') else 'denoise',
                                current_latents=args[0].shape[2],
                                cache_latents_including_sink=past // 300,
                                history_chunks_excluding_sink=max(0, past // 300 - 1) / 2)
            self.forward_id += 1
            kwargs['use_chunk_flash_attn'] = True
            try:
                return self.timed('model_forward_inclusive', original_forward, *args, **kwargs)
            finally:
                self.context = {}

        def attention(module, *args, **kwargs):
            self.layer = layers[id(module)]
            try:
                return original_attention(module, *args, **kwargs)
            finally:
                self.layer = None

        def flash(*args, **kwargs):
            self.cast_index = 0
            try:
                return self.timed('attention_inclusive', original_flash, *args, **kwargs)
            finally:
                self.cast_index = None

        def to(tensor, *args, **kwargs):
            if self.cast_index is not None:
                index = self.cast_index
                self.cast_index += 1
                # The existing helper calls .to on Q, K, V in that order.
                # Observe exactly the original conversion; do not replace its
                # stride policy or force contiguous buffers for measurement.
                if index == 1:
                    return self.timed('k_dtype_conversion_nested_in_attention', original_to,
                                      tensor, *args, **kwargs)
            return original_to(tensor, *args, **kwargs)

        def cat(*args, **kwargs):
            dim = kwargs.get('dim', args[1] if len(args) > 1 else 0)
            if self.layer is not None and dim == -2:
                return self.timed('history_current_cat', original_cat, *args, **kwargs)
            return original_cat(*args, **kwargs)

        def patch(obj, attr, replacement):
            self.stack.enter_context(mock.patch.object(obj, attr, replacement))

        patch(self.model.net, 'forward_with_cache', forward)
        patch(mw.Attention, 'forward', attention)
        patch(mw, '_chunkwise_flash_attention', flash)
        patch(torch, 'cat', cat)
        patch(torch.Tensor, 'to', to)
        for obj, attr, category, is_static in (
            (ResidentKVCache, 'write_layer', 'current_kv_write', False),
            (ResidentKVCache, 'evict_and_shift', 'evict_and_rope_inclusive', False),
            (DiffusionForcingDenoiser, '_append_kv_cache', 'commit_cat', True),
            (DiffusionForcingDenoiser, '_evict_and_shift_cache', 'evict_and_rope_inclusive', True),
            (mw.VisionRotaryEmbeddingFast3D, 'rope_shift_time', 'rope_rebase_nested_in_eviction', False),
        ):
            original = getattr(obj, attr)
            def wrapped(*args, _fn=original, _category=category, **kwargs):
                return self.timed(_category, _fn, *args, **kwargs)
            patch(obj, attr, staticmethod(wrapped) if is_static else wrapped)
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def rows(self):
        torch.cuda.synchronize()
        return [dict(backend=self.name, category=category, **meta,
                     duration_ms=start.elapsed_time(end))
                for category, meta, start, end in self.events]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    cli = parser.parse_args()
    cli.out.mkdir(parents=True, exist_ok=True)
    if (cli.out / 'summary.json').exists():
        raise FileExistsError('choose a new output directory')
    torch.set_num_threads(4)
    model, _ = bench.load_model()
    data = bench.load_prepared()
    production = [bench.ROOT / 'miniworld' / name for name in ('miniworld.py', 'denoiser.py', 'kv_cache.py')]
    report = dict(
        episode='002352', device=torch.cuda.get_device_name(), torch=torch.__version__,
        scope='Full 125-latent rollout WITHOUT VAE, with diagnostic hooks; NOT E2E timing',
        settings=dict(max_cache_chunks=24, sink_frames=1, inflight_chunks=8,
                      chunk_size=2, seed=bench.SEED, history_len=1),
        caveat='Events include stream execution and host submission gaps. Model includes attention and staging; attention includes K cast; eviction includes RoPE. Do not double count.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in production},
        results={},
    )
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(3):
            model.net.forward_with_cache(data['noise'][:, :, :16], torch.ones(1, 16, device='cuda'),
                                         data['cond_seq'][:, :16], chunk_size=2)
    for name, resident in (('concatenated', False), ('resident', True)):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with Recorder(model, name) as recorder, torch.autocast('cuda', dtype=torch.bfloat16):
            result = model.generate_eval_latents_streaming(
                data['latents'], data['cond_seq'], total_len=125, history_len=1,
                max_cache_chunks=24, inflight_chunks=8, sink_frames=1,
                noise=data['noise'], use_resident_kv_cache=resident,
            )
        torch.cuda.synchronize()
        diagnostic_wall = time.perf_counter() - started
        reference_path = cli.out.parents[1] / f'{name}_latents.pt'
        reference = torch.load(reference_path, map_location='cpu', weights_only=True)
        actual = result.cpu()
        rows = recorder.rows()
        write_csv(cli.out / f'{name}_events.csv', rows)
        summary = aggregate(rows, ['category'])
        report['results'][name] = dict(diagnostic_wall_sec=diagnostic_wall,
                                       identical_to_uninstrumented=bool(torch.equal(actual, reference)),
                                       latent_max_abs_diff=float((actual-reference).abs().max()),
                                       categories=summary)
        assert report['results'][name]['identical_to_uninstrumented']
        bench.dump(cli.out / 'partial.json', report)
        print(json.dumps(report['results'][name]), flush=True)
        del rows, recorder, result, reference, actual
        torch.cuda.empty_cache()
    bench.dump(cli.out / 'summary.json', report)


if __name__ == '__main__':
    main()
