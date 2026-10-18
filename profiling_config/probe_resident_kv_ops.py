"""Read-only production-path microbenchmarks for resident KV implementation audit.

Synthetic values reproduce QKV-projection strides and actual episode shapes.
These hot-input operator timings are diagnostic, not an end-to-end speedup.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from flash_attn import flash_attn_func

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miniworld.kv_cache import ResidentKVCache
from miniworld.miniworld import _chunkwise_flash_attention, VisionRotaryEmbeddingFast3D, RMSNorm
from miniworld.denoiser import DiffusionForcingDenoiser


def compare(funcs, *, repeats=20, rounds=7):
    samples = {name: [] for name in funcs}
    wall = {name: [] for name in funcs}
    for func in funcs.values():
        for _ in range(10):
            func()
    torch.cuda.synchronize()
    for rnd in range(rounds):
        names = list(funcs) if rnd % 2 == 0 else list(reversed(funcs))
        for name in names:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            begin.record()
            for _ in range(repeats):
                result = funcs[name]()
            end.record()
            end.synchronize()
            wall[name].append((time.perf_counter() - started) * 1000 / repeats)
            samples[name].append(begin.elapsed_time(end) / repeats)
    return {name: {
        'cuda_median_ms': statistics.median(values),
        'cuda_rounds_ms': values,
        'wall_median_ms': statistics.median(wall[name]),
    } for name, values in samples.items()}


def precast_flash(q, k, v, history, current):
    outputs = [flash_attn_func(q[:, start:start + 600], k[:, :history + start + 600],
                              v[:, :history + start + 600], causal=False)
               for start in range(0, current, 600)]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eviction-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20263368)
    device = torch.device('cuda')
    rope = VisionRotaryEmbeddingFast3D(128, 64, 15, 20).to(device)
    norm = RMSNorm(128).to(device)
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'capacity_tokens': 19200, 'notes': 'Hot synthetic values; alternating seven rounds of twenty eager repeats; timings are not E2E.',
              'shapes': []}
    for history in (() if args.eviction_only else (0, 2400, 7200, 14400)):
        for current in (600, 4800):
            # Match qkv linear output -> reshape -> permute -> unbind. V retains
            # its interleaved QKV stride; Q/K follow actual FP32 norm and RoPE.
            qkv = torch.randn((1, current, 3, 12, 128), device=device, dtype=torch.bfloat16)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            q = rope(norm(q), num_frames_override=current // 300, start_frame=history // 300)
            k = rope(norm(k), num_frames_override=current // 300, start_frame=history // 300)
            past_k = torch.randn((1, 12, history, 128), device=device)
            past_v = torch.randn((1, 12, history, 128), device=device, dtype=torch.bfloat16)
            bank = ResidentKVCache(1, 19200)
            if history:
                bank.begin_forward(history, allow_commit=True)
                bank.write_layer(0, past_k, past_v)
                bank.finish_forward()
                bank.commit()

            def old_store():
                if not history:
                    return k, v
                return torch.cat((past_k, k), dim=-2), torch.cat((past_v, v), dim=-2)

            def new_store():
                bank.begin_forward(current)
                pair = bank.write_layer(0, k, v)
                bank.finish_forward()
                return pair

            old_k, old_v = old_store()
            new_k, new_v = new_store()
            assert torch.equal(old_k, new_k) and torch.equal(old_v, new_v)
            old_qb = q.transpose(1, 2).to(torch.bfloat16)
            old_kb = old_k.transpose(1, 2).to(torch.bfloat16)
            new_kb = new_k.transpose(1, 2).to(torch.bfloat16)
            old_vb, new_vb = old_v.transpose(1, 2), new_v.transpose(1, 2)
            old_result = _chunkwise_flash_attention(q, old_k, old_v, 600)
            new_result = _chunkwise_flash_attention(q, new_k, new_v, 600)
            row = {'history_tokens': history, 'current_tokens': current,
                   'input_k_stride': k.stride(), 'input_v_stride': v.stride(),
                   'old_k_stride': old_k.stride(), 'resident_k_stride': new_k.stride(),
                   'old_cast_k_stride': old_kb.stride(), 'resident_cast_k_stride': new_kb.stride(),
                   'output_equal': torch.equal(old_result, new_result),
                   'max_output_difference': (old_result.float() - new_result.float()).abs().max().item()}
            row['store'] = compare({'old_cat': old_store, 'resident_write': new_store})
            row['cast_k'] = compare({
                'old': lambda: old_k.transpose(1, 2).to(torch.bfloat16),
                'resident': lambda: new_k.transpose(1, 2).to(torch.bfloat16),
            })
            row['flash_precast'] = compare({
                'old': lambda: precast_flash(old_qb, old_kb, old_vb, history, current),
                'resident': lambda: precast_flash(old_qb, new_kb, new_vb, history, current),
            })
            row['attention_helper'] = compare({
                'old': lambda: _chunkwise_flash_attention(q, old_k, old_v, 600),
                'resident': lambda: _chunkwise_flash_attention(q, new_k, new_v, 600),
            })
            def old_combined():
                kk, vv = old_store()
                return _chunkwise_flash_attention(q, kk, vv, 600)
            def new_combined():
                kk, vv = new_store()
                return _chunkwise_flash_attention(q, kk, vv, 600)
            row['store_plus_attention'] = compare({'old': old_combined, 'resident': new_combined})
            if history in (7200, 14400) and current == 4800:
                for label, func in [('old', old_combined), ('resident', new_combined)]:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=True, profile_memory=True) as prof:
                        func()
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(args.output / f'trace_h{history}_{label}.json'))
                    (args.output / f'profiler_h{history}_{label}.txt').write_text(
                        prof.key_averages(group_by_input_shape=True).table(
                            sort_by='self_cuda_time_total', row_limit=80))
            report['shapes'].append(row)
            (args.output / 'microbench.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)

    # A full-window commit transiently has 50 latent frames, then drops 2.
    # Restore buffers outside each timed interval: eviction is a mutation, and
    # timing repeated eviction on an already-shortened cache would be invalid.
    k_source = torch.randn((1, 12, 15000, 128), device=device)
    v_source = torch.randn((1, 12, 15000, 128), device=device, dtype=torch.bfloat16)
    cache = ResidentKVCache(1, 19200)
    def restore():
        cache.reset()
        cache.begin_forward(15000, allow_commit=True)
        cache.write_layer(0, k_source, v_source)
        cache.finish_forward()
        cache.commit()
    def old_evict():
        return DiffusionForcingDenoiser._evict_and_shift_cache(
            [(k_source, v_source)], 2, 300, rope, sink_frames=1)
    def new_evict():
        cache.evict_and_shift(2, 300, rope, sink_frames=1)
    restore()
    old = old_evict()
    new_evict()
    new = cache.history()
    equal = all(torch.equal(a, b) for a, b in zip(old[0], new[0]))
    for _ in range(10):
        restore()
        old_evict()
        new_evict()
    times = {'old': [], 'resident': []}
    for rnd in range(7):
        for name in (('old', 'resident') if rnd % 2 == 0 else ('resident', 'old')):
            events = []
            for _ in range(20):
                restore()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                result = old_evict() if name == 'old' else new_evict()
                end.record()
                events.append((begin, end))
            torch.cuda.synchronize()
            times[name].append(statistics.mean(a.elapsed_time(b) for a, b in events))
    eviction = {'output_equal': equal, 'layers': 1,
                'old_frames_including_sink': 50, 'new_frames_including_sink': 48,
                'notes': 'Per-operation CUDA events, source restored outside timed intervals.',
                'times': {name: {'cuda_median_ms': statistics.median(values), 'cuda_rounds_ms': values}
                          for name, values in times.items()}}
    (args.output / 'eviction.json').write_text(json.dumps(eviction, indent=2))
    print(json.dumps(eviction), flush=True)


if __name__ == '__main__':
    main()
