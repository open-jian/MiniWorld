"""Summarize uninstrumented latency separately from diagnostic KV timings."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('/data2/jian/output/Miniworld/DROID_long_horizon/chunk_flash_attention/episode_002352')
BACKENDS = ('baseline', 'chunk_flash')
COLORS = ('#596579', '#108d86')
STRICT = ('internal_kv_concat', 'cache_append', 'cache_evict_and_rope_rebase')
BROAD = STRICT + ('cached_attention', 'dense_mask_build')


def main():
    runs = {f'gpu{gpu}_{backend}': json.loads((OUT / f'gpu{gpu}_{backend}_timing.json').read_text())
            for gpu in (4, 5) for backend in BACKENDS}
    profile = json.loads((OUT / 'kv_profile_report.json').read_text())
    quality = json.loads((OUT / 'quality_analysis.json').read_text())
    times = {}
    for backend in BACKENDS:
        selected = [runs[f'gpu{gpu}_{backend}'] for gpu in (4, 5)]
        times[backend] = {
            key: statistics.mean(row[key] for row in selected)
            for key in ('end_to_end_sec', 'generation_stream_decode_sec', 'source_load_transfer_sec', 'vae_encode_condition_sec')
        }
        times[backend]['end_to_end_samples_sec'] = [row['end_to_end_sec'] for row in selected]
        times[backend]['peak_allocated_gib_max'] = max(row['peak_allocated_bytes'] for row in selected) / 2**30
    paired = {f'gpu{gpu}': runs[f'gpu{gpu}_baseline']['end_to_end_sec'] / runs[f'gpu{gpu}_chunk_flash']['end_to_end_sec']
              for gpu in (4, 5)}
    totals = {backend: {row['category']: row['total_ms'] / 1000 for row in profile['backends'][backend]['categories']}
              for backend in BACKENDS}
    comparisons = []
    for name, categories in [('kv_maintenance_only', STRICT), ('cached_attention_subsystem', BROAD)]:
        values = {backend: sum(totals[backend].get(key, 0) for key in categories) for backend in BACKENDS}
        comparisons.append(dict(component=name, categories=categories, **values,
                                speedup=values['baseline']/values['chunk_flash'], scope='separate instrumented rollout CUDA event sums'))
    for name in ('cached_attention', 'internal_kv_concat', 'dense_mask_build', 'cache_append', 'cache_evict_and_rope_rebase', 'model_forward'):
        values = {backend: totals[backend].get(name, 0) for backend in BACKENDS}
        comparisons.append(dict(component=name, **values,
                                speedup=values['baseline']/values['chunk_flash'] if values['chunk_flash'] else None,
                                scope='separate instrumented rollout CUDA event sums'))
    report = {
        'episode': '002352', 'seed': 20263368, 'processed_rgb_frames': 497,
        'sampler_mode': 'normal autoregressive generated history',
        'history_chunks_excluding_sink_at_capacity': 23.5,
        'fixed_sink_latents': 1, 'inflight_chunks': 8,
        'timing': times, 'same_gpu_paired_speedups': paired,
        'mean_end_to_end_speedup': times['baseline']['end_to_end_sec'] / times['chunk_flash']['end_to_end_sec'],
        'kv_component_comparisons': comparisons,
        'persistent_kv_gib_both_cfg_branches': 48 * 300 * 1536 * 28 * (4 + 2) * 2 / 2**30,
        'quality': quality['backend_means'],
        'limits': [
            'One episode and one seed; two runs per backend check GPU/order repeatability, not statistical generalization.',
            'Warm E2E includes source decode/actions, transfer, VAE encode, conditioning, generation/cache management and streaming VAE decode.',
            'Model loading, warmup, output video encoding and quality analysis excluded from E2E.',
            'Warmup exercises VAE encode/decode and both attention backends at cached full/commit/tail shapes; collect_stream_timing=False for both timed arms.',
            'KV module CUDA timings come from separate instrumented runs; host submission gaps/instrumentation can affect these sums.',
            'model_forward contains attention/concat/mask; never add that inclusive total to its nested components.',
            'Original max_cache_chunks=24 includes sink; capacity is preserved and equals 23.5 chunks of sliding history plus one latent sink.',
        ],
    }
    (OUT / 'comparison_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    with (OUT / 'performance_summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['component', 'baseline_sec', 'chunk_flash_sec', 'speedup', 'scope'])
        writer.writeheader()
        for key in ('end_to_end_sec', 'generation_stream_decode_sec'):
            writer.writerow(dict(component=key, baseline_sec=times['baseline'][key], chunk_flash_sec=times['chunk_flash'][key],
                                 speedup=times['baseline'][key]/times['chunk_flash'][key], scope='uninstrumented two paired runs'))
        for row in comparisons:
            writer.writerow(dict(component=row['component'], baseline_sec=row['baseline'], chunk_flash_sec=row['chunk_flash'],
                                 speedup=row['speedup'], scope=row['scope']))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    def bars(ax, title, values, note):
        ax.bar(['Original', 'Chunk FlashAttention'], values, color=COLORS, width=0.58)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('Seconds (lower is better)')
        ax.set_ylim(0, max(values)*1.23)
        for index, value in enumerate(values):
            ax.text(index, value, f'{value:.2f} s', ha='center', va='bottom')
        ax.text(.98, .95, f'{values[0]/values[1]:.2f}x', transform=ax.transAxes, ha='right', va='top', fontsize=15, weight='bold')
        ax.text(.5, -.19, note, transform=ax.transAxes, ha='center', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    bars(axes[0, 0], 'Warm end-to-end pipeline', [times[b]['end_to_end_sec'] for b in BACKENDS], 'Two same-GPU pairs; no diagnostic instrumentation')
    broad = comparisons[1]
    bars(axes[0, 1], 'Cached-attention subsystem', [broad[b] for b in BACKENDS], 'Attention + mask + KV concatenation/append/eviction; diagnostic')
    strict = comparisons[0]
    bars(axes[1, 0], 'KV maintenance alone', [strict[b] for b in BACKENDS], 'Concatenation + append + eviction/RoPE rebase; diagnostic')
    replays = profile['identical_input_first_layer_replays']
    for backend, color in zip(BACKENDS, COLORS):
        axes[1, 1].plot([row['history_chunks_excluding_sink'] for row in replays],
                        [row[f'{backend}_median_ms'] for row in replays], marker='o', color=color,
                        label='Original' if backend == 'baseline' else 'Chunk FlashAttention')
    axes[1, 1].set(title='Same-input attention kernel replay', xlabel='History window (chunks; excludes sink and query)', ylabel='Milliseconds per layer (8 query chunks)')
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(alpha=.2)
    fig.suptitle('DROID episode 002352 | 497 RGB frames | seed 20263368', fontsize=15)
    fig.savefig(OUT / 'performance_comparison.png', dpi=180)
    fig.savefig(OUT / 'performance_comparison.pdf')
    plt.close(fig)
    lines = [f"Episode 002352, seed 20263368, 497 RGB frames, same official sampler configuration.",
             f"Warm E2E: {times['baseline']['end_to_end_sec']:.3f}s -> {times['chunk_flash']['end_to_end_sec']:.3f}s; {report['mean_end_to_end_speedup']:.3f}x.",
             f"Paired GPU speedups: {paired}",
             f"KV memory both CFG branches: {report['persistent_kv_gib_both_cfg_branches']:.4f} GiB (unchanged storage dtypes/capacity).",
             '', 'Separate diagnostic component totals (seconds):']
    lines += [f"{row['component']}: {row['baseline']:.4f} -> {row['chunk_flash']:.4f}; speedup {row['speedup']}" for row in comparisons]
    lines += ['', 'Quality, pre-compression uint8, frame 0 excluded:']
    lines += [str(row) for row in quality['backend_means']]
    lines += ['', 'Limits:'] + report['limits']
    (OUT / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
