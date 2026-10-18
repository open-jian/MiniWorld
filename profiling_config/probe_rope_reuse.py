"""Bounded synthetic microbenchmark of exact RoPE coefficient reuse.

Production code is not modified. Timed values are operator measurements, NOT
predicted end-to-end savings: host launch gaps and GPU overlap can differ in
the complete model. Profile metadata supplies actual episode shape weights.
"""

from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miniworld.miniworld import VisionRotaryEmbeddingFast3D, rotate_half

ROOT = Path('/data2/jian/output/Miniworld/DROID_long_horizon/resident_kv/episode_002352')
OUT = ROOT / 'rope_potential'
EVENTS = ROOT / 'implementation_audit/full_profile/concatenated_events.csv'
CHECKPOINT = Path('/data2/jian/checkpoints/miniworld/MiniWorld_1b_droid.pt')


def measure(funcs, rounds=5, repeats=10):
    samples = {name: {'cuda_ms': [], 'wall_ms': []} for name in funcs}
    for func in funcs.values():
        for _ in range(5):
            func()
    torch.cuda.synchronize()
    for rnd in range(rounds):
        names = list(funcs) if rnd % 2 == 0 else list(reversed(funcs))
        for name in names:
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            for _ in range(repeats):
                result = funcs[name]()
            end.record()
            end.synchronize()
            samples[name]['wall_ms'].append((time.perf_counter() - started) * 1000 / repeats)
            samples[name]['cuda_ms'].append(begin.elapsed_time(end) / repeats)
    for result in samples.values():
        result['cuda_median_ms'] = statistics.median(result['cuda_ms'])
        result['wall_median_ms'] = statistics.median(result['wall_ms'])
    return samples


@torch.inference_mode()
def main():
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(20263368)
    OUT.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with EVENTS.open() as handle:
        for row in csv.DictReader(handle):
            if row['category'] == 'model_forward_inclusive':
                counts[(int(row['cache_latents_including_sink']), int(row['current_latents']))] += 1
    rope = VisionRotaryEmbeddingFast3D(128, 64, 15, 20)
    # mmap reads only the small RoPE buffers, not the model's multi-GB weights.
    raw = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    weights = raw.get('ema_model', raw.get('model', raw))
    candidates = [key[:-len('base_freqs_t')] for key in weights if key.endswith('base_freqs_t')]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one RoPE module, found {candidates}')
    prefix = candidates[0]
    copied_buffers = []
    for name, buffer in rope.named_buffers():
        key = prefix + name
        if key in weights and weights[key].shape == buffer.shape:
            buffer.copy_(weights[key])
            copied_buffers.append(key)
    rope = rope.cuda()
    del weights, raw
    # Checkpoint tables may have been built on another device; rebuilding once
    # on the execution GPU avoids assuming CPU/GPU cos/sin are bitwise equal.
    gpu_full = rope._build_freqs(64, rope.base_freqs_t, rope.base_freqs_h,
                               rope.base_freqs_w, 15, 20, 128)
    report = {
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cpu_threads': 4,
        'checkpoint': str(CHECKPOINT), 'checkpoint_buffers_loaded': copied_buffers,
        'model_forwards': sum(counts.values()), 'unique_shapes': len(counts),
        'layers': 28, 'old_coefficient_build_calls': sum(counts.values()) * 28 * 2,
        'protocol': 'Five alternating rounds, ten eager repetitions; CUDA events outside repetition loop. Wall includes final synchronization. Hot synthetic inputs; no full model execution.',
        'frequency_builds': [], 'qk_rotation': [],
    }
    # Obtain precise frequency-build costs for every shape in the actual trace.
    # Full-table slices are also checked against independent table construction.
    for offset, frames in sorted(counts):
        cos, sin = rope._get_freqs(frames, torch.device('cuda'), start_frame=offset)
        sliced_cos = rope.freqs_cos[offset * 300:(offset + frames) * 300]
        sliced_sin = rope.freqs_sin[offset * 300:(offset + frames) * 300]
        equal = torch.equal(cos, sliced_cos) and torch.equal(sin, sliced_sin)
        rebuilt = tuple(table[offset * 300:(offset + frames) * 300] for table in gpu_full)
        rebuilt_equal = all(torch.equal(x, y) for x, y in zip((cos, sin), rebuilt))
        result = measure({'build': lambda: rope._get_freqs(frames, torch.device('cuda'), start_frame=offset)})
        report['frequency_builds'].append({
            'offset': offset, 'current_latents': frames, 'forward_count': counts[offset, frames],
            'old_build_count': counts[offset, frames] * 28 * 2,
            'full_table_slice_bitwise_equal': equal, **result,
            'gpu_rebuilt_full_table_slice_bitwise_equal': rebuilt_equal,
            'checkpoint_table_max_abs_difference': max(
                (cos - sliced_cos).abs().max().item(), (sin - sliced_sin).abs().max().item()),
        })
    for offset, frames in ((48, 16), (48, 15), (48, 2), (0, 16), (0, 2)):
        # Match norm/RoPE input's token-major physical order with B,H,N,D view.
        q = torch.randn((1, frames * 300, 12, 128), device='cuda').transpose(1, 2)
        k = torch.randn_like(q)
        cos, sin = rope._get_freqs(frames, q.device, start_frame=offset)
        def apply(t, c=cos, s=sin):
            return t * c + rotate_half(t) * s
        def old_pair():
            return (rope(q, num_frames_override=frames, start_frame=offset),
                    rope(k, num_frames_override=frames, start_frame=offset))
        def build_once_pair():
            c, s = rope._get_freqs(frames, q.device, start_frame=offset)
            return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s
        def cached_pair():
            return apply(q), apply(k)
        old, once, cached = old_pair(), build_once_pair(), cached_pair()
        equal = all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(old, once, cached))
        result = measure({'old_qk_two_builds': old_pair, 'qk_one_build': build_once_pair,
                          'qk_cached_coefficients': cached_pair, 'one_rotation_only': lambda: apply(q)})
        row = {'offset': offset, 'current_latents': frames,
               'output_bitwise_equal': equal, **result}
        report['qk_rotation'].append(row)
        print(json.dumps(row), flush=True)
    def weighted(name):
        # Arithmetic operator-cost estimate only; overlap/launch gaps mean this
        # is not a promise of equivalent E2E wall-clock savings.
        return sum(row['old_build_count'] * row['build'][name] for row in report['frequency_builds']) / 1000
    report['weighted_old_frequency_construction_seconds'] = {
        'cuda_event_sum_estimate': weighted('cuda_median_ms'),
        'wall_sum_estimate': weighted('wall_median_ms'),
        'warning': 'Microbenchmark costs multiplied by exact call counts; NOT measured or predicted E2E savings.',
    }
    report['all_frequency_slices_bitwise_equal'] = all(r['full_table_slice_bitwise_equal'] for r in report['frequency_builds'])
    report['all_gpu_rebuilt_frequency_slices_bitwise_equal'] = all(
        r['gpu_rebuilt_full_table_slice_bitwise_equal'] for r in report['frequency_builds'])
    report['all_qk_outputs_bitwise_equal'] = all(r['output_bitwise_equal'] for r in report['qk_rotation'])
    representative = {(r['offset'], r['current_latents']): r for r in report['qk_rotation']}
    weighted_pair_delta = 0.0
    for (offset, frames), count in counts.items():
        row = representative.get((offset, frames), representative[48, frames])
        delta = row['old_qk_two_builds']['cuda_median_ms'] - row['qk_cached_coefficients']['cuda_median_ms']
        weighted_pair_delta += count * 28 * delta / 1000
    report['representative_pair_delta_seconds_estimate'] = weighted_pair_delta
    report['representative_pair_delta_caveat'] = (
        'Shape-count-weighted old-QK versus cached-QK microbenchmark deltas; '
        'unmeasured intermediate offsets use offset48 of the same current length. '
        'NOT measured or predicted E2E savings. Host/GPU overlap explains why '
        'this differs from summing isolated coefficient-build times.')
    report['elapsed_seconds'] = time.perf_counter() - started
    (OUT / 'rope_reuse_microbench.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('frequency_builds', 'qk_rotation')}), flush=True)


if __name__ == '__main__':
    main()
