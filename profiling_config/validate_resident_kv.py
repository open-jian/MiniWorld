#!/usr/bin/env python3
"""Validate resident KV against concatenated KV on the complete episode 002352.

Run with the MiniWorld environment and CUDA_VISIBLE_DEVICES=4. Both paths use
chunkwise FlashAttention and the archived sampler settings. The warm E2E timer
includes source loading, transfers, VAE encoding, conditioning, generation and
streaming VAE decoding; model loading, warmup, saving and metrics are excluded.
This is one matched pair, not a statistically stable speed benchmark.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

import benchmark_episode_chunk_flash as benchmark
from miniworld.denoiser import DiffusionForcingDenoiser


DEFAULT_OUT = Path(
    '/data2/jian/output/Miniworld/DROID_long_horizon/resident_kv/episode_002352'
)
MODES = {'concatenated': False, 'resident': True}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def generate(model, data, resident, *, total_len=benchmark.TOTAL_LATENTS, decoder=None):
    # Explicitly pass the cache switch for both paths. The imported attention
    # benchmark's generate() is pinned to the legacy cache for its own A/B test.
    return model.generate_eval_latents_streaming(
        data['latents'], data['cond_seq'], total_len=total_len, history_len=1,
        max_cache_chunks=24, inflight_chunks=8, sink_frames=1,
        stream_decoder=decoder, collect_stream_timing=False, noise=data['noise'],
        use_resident_kv_cache=resident,
    )


def write_manifest(out, args, entry, order):
    files = [
        benchmark.ROOT / 'miniworld/miniworld.py',
        benchmark.ROOT / 'miniworld/denoiser.py',
        Path(__file__).resolve(), Path(benchmark.__file__).resolve(),
        benchmark.OUT / 'prepared_inputs.pt', benchmark.OUT / 'gt_rgb.npy',
        benchmark.OLD / 'worker_5_environment.json',
        Path(args.checkpoint), Path(entry['source']), Path(entry['parquet']),
    ]
    # Include any standalone resident cache implementation added alongside the
    # integration points. Paths are recorded as well as hashes for auditability.
    files += list((benchmark.ROOT / 'miniworld').glob('*cache*.py'))
    source_hashes = {str(path): sha256_file(path) for path in dict.fromkeys(files)}
    payload = {
        'episode': '002352', 'seed': benchmark.SEED, 'order': order,
        'rgb_frames': benchmark.TOTAL_RGB, 'latent_frames': benchmark.TOTAL_LATENTS,
        'mode': 'generated-history autoregression; only initial latent is GT',
        'backend': 'chunkwise FlashAttention for both cache implementations',
        'sampler_args': vars(args),
        'overrides': {'total_len': 125, 'history_len': 1, 'max_cache_chunks': 24,
                      'inflight_chunks': 8, 'sink_frames': 1,
                      'collect_stream_timing': False},
        'window_definition': 'history only, excludes sink and current denoising area',
        'capacity_including_sink_latents': 48,
        'sink_latents': 1, 'saturated_history_chunks_excluding_sink': 23.5,
        'latent_frames_per_chunk': 2,
        'timing_scope': 'source/actions load + transfers + VAE encode + conditioning '
                        '+ generation with streaming VAE decode',
        'timing_exclusions': ['model loading', 'warmup', 'file output', 'metrics'],
        'warmup': 'existing benchmark shape/VAE warmup, then 3-latent generation '
                  'with each explicit cache switch',
        'allocator_protocol': 'gc + empty_cache before each timed run; model, VAE '
                              'and fixed inputs remain resident; reset peak stats',
        'diagnostic_hooks': False,
        'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'git_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=benchmark.ROOT, text=True).strip(),
        'git_status': subprocess.check_output(
            ['git', 'status', '--short'], cwd=benchmark.ROOT, text=True),
        'sha256': source_hashes,
        'limitation': 'one episode and seed, one timed run per cache path on one GPU',
    }
    benchmark.dump(out / 'manifest.json', payload)


def quality_and_parity(out):
    # Analysis runs after all timed generation has completed. Reuse exactly the
    # previous metric definitions, including exclusion of conditioning frame 0.
    import lpips
    import analyze_episode_chunk_flash as analysis

    arrays = {name: np.load(out / f'{name}_rgb.npy', mmap_mode='r') for name in MODES}
    gt = np.load(benchmark.OUT / 'gt_rgb.npy', mmap_mode='r')
    latents = {name: torch.load(out / f'{name}_latents.pt', weights_only=True,
                               map_location='cpu') for name in MODES}
    latent_diff = (latents['resident'].float() - latents['concatenated'].float()).abs()
    pixel_diff = arrays['resident'].astype(np.int16) - arrays['concatenated'].astype(np.int16)
    identical = bool(np.array_equal(arrays['resident'], arrays['concatenated']))
    parity = {
        'latents_bitwise_equal': bool(torch.equal(latents['resident'], latents['concatenated'])),
        'latent_max_abs_diff': float(latent_diff.max()),
        'latent_mean_abs_diff': float(latent_diff.mean()),
        'rgb_bitwise_equal': identical,
        'rgb_max_abs_diff_uint8': int(np.abs(pixel_diff).max()),
        'rgb_mean_abs_diff_uint8': float(np.abs(pixel_diff).mean()),
        'rgb_changed_scalar_fraction': float(np.mean(pixel_diff != 0)),
    }
    device = torch.device('cuda')
    metric = lpips.LPIPS(net='alex', version='0.1').eval().to(device)
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    per_frame = []
    summary = []
    old_rows = None
    for name in MODES:
        rows = old_rows if name == 'resident' and identical else analysis.measure_pair(
            gt, arrays[name], metric, device, batch_size=16)
        if name == 'concatenated':
            old_rows = rows
        per_frame.extend({'cache': name, **row} for row in rows)
        summary.append({'cache': name, **analysis.summarize(rows, 1, benchmark.TOTAL_RGB - 1)})
    analysis.write_csv(out / 'quality_per_frame.csv', per_frame)
    analysis.write_csv(out / 'quality_summary.csv', summary)
    benchmark.dump(out / 'quality_and_parity.json', analysis.json_safe({
        'parity': parity, 'quality_vs_gt': summary,
        'quality_delta_resident_minus_concatenated': {
            metric_name: summary[1][metric_name] - summary[0][metric_name]
            for metric_name in analysis.METRICS
        },
        'metrics': 'arithmetic frame means over RGB 1..496; PSNR RGB range255; '
                   'SSIM gaussian sigma1.5 win11 population covariance RGB mean; '
                   'LPIPS AlexNet v0.1 FP32',
        'identical_outputs_share_one_metric_evaluation': identical,
    }))
    return parity, summary


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--order', nargs=2, choices=tuple(MODES),
                        default=['concatenated', 'resident'])
    cli = parser.parse_args()
    if len(set(cli.order)) != 2:
        parser.error('--order must contain each cache path exactly once')
    # Fail closed: the old sampler accepted unknown **kwargs, which could turn
    # an apparently successful A/B comparison into two copies of the same run.
    signature = inspect.signature(DiffusionForcingDenoiser.generate_eval_latents_streaming)
    if 'use_resident_kv_cache' not in signature.parameters:
        raise RuntimeError('Sampler has no explicit use_resident_kv_cache switch yet')
    cli.out.mkdir(parents=True, exist_ok=True)
    if any((cli.out / f'{name}_timing.json').exists() for name in MODES):
        raise FileExistsError(f'Completed run already exists under {cli.out}; choose a fresh --out')
    torch.set_num_threads(4)
    model, args = benchmark.load_model()
    vae = benchmark.load_wan22_vae(args)
    dataset, entry = benchmark.make_dataset()
    fixed = benchmark.load_prepared()
    write_manifest(cli.out, args, entry, cli.order)
    # Pin the attention implementation for BOTH cache paths before sampler calls.
    benchmark.set_backend(model, 'chunk_flash')
    video, actions = benchmark.decode_source(dataset, entry)
    benchmark.encode_inputs(vae, video, actions)
    del video, actions
    benchmark.warmup(model, vae, fixed)
    for resident in MODES.values():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            generate(model, fixed, resident, total_len=3)
    torch.cuda.synchronize()
    results = {}
    for name in cli.order:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        benchmark.dump(cli.out / 'state.json', {'phase': 'running', 'cache': name})
        started = time.perf_counter()
        video, actions = benchmark.decode_source(dataset, entry)
        torch.cuda.synchronize()
        loaded = time.perf_counter()
        data = benchmark.encode_inputs(vae, video, actions)
        data['noise'] = fixed['noise']
        torch.cuda.synchronize()
        encoded = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            latents, rgb = generate(model, data, MODES[name],
                                    decoder=benchmark.StreamingVAEDecoder(vae))
        torch.cuda.synchronize()
        ended = time.perf_counter()
        result = {
            'cache': name, 'use_resident_kv_cache': MODES[name], 'order': cli.order,
            'source_load_transfer_sec': loaded - started,
            'vae_encode_condition_sec': encoded - loaded,
            'generation_stream_decode_sec': ended - encoded,
            'end_to_end_sec': ended - started,
            'initial_allocated_bytes': initial_allocated,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'sampler_meta': model.last_eval_meta.copy(),
            'prepared_latents_identical': bool(torch.equal(data['latents'], fixed['latents'])),
            'conditioning_identical': bool(torch.equal(data['cond_seq'], fixed['cond_seq'])),
        }
        assert result['prepared_latents_identical'] and result['conditioning_identical']
        assert result['sampler_meta']['effective_steps'] == 40
        pixels = ((rgb[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()
        assert pixels.shape == (benchmark.TOTAL_RGB, 240, 320, 3)
        np.save(cli.out / f'{name}_rgb.npy', pixels)
        torch.save(latents.cpu(), cli.out / f'{name}_latents.pt')
        result['rgb_array_sha256'] = hashlib.sha256(pixels.tobytes()).hexdigest()
        result['latents_file_sha256'] = sha256_file(cli.out / f'{name}_latents.pt')
        benchmark.dump(cli.out / f'{name}_timing.json', result)
        results[name] = result
        print('COMPLETED', json.dumps(result), flush=True)
        del video, actions, data, latents, rgb, pixels
    # Release generation resources before LPIPS to keep analysis memory outside
    # the measured generation peaks and avoid unnecessary GPU residency.
    del model, vae, fixed
    gc.collect()
    torch.cuda.empty_cache()
    benchmark.dump(cli.out / 'state.json', {'phase': 'quality_analysis'})
    parity, quality = quality_and_parity(cli.out)
    summary = {
        'timings': results, 'parity': parity, 'quality_vs_gt': quality,
        'end_to_end_speedup': results['concatenated']['end_to_end_sec'] / results['resident']['end_to_end_sec'],
        'generation_stream_decode_speedup': results['concatenated']['generation_stream_decode_sec'] / results['resident']['generation_stream_decode_sec'],
        'peak_allocated_change_bytes': results['resident']['peak_allocated_bytes'] - results['concatenated']['peak_allocated_bytes'],
        'limitation': 'single matched pair; small timing differences require repetitions to establish',
    }
    benchmark.dump(cli.out / 'summary.json', summary)
    benchmark.dump(cli.out / 'state.json', {'phase': 'complete'})
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
