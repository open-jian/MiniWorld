"""Paired, decoded-video evaluation of opt-in MiniWorld Forcing-KV.

One worker owns one GPU and one episode. The wall clock includes fresh initial
RGB/action reads, input preparation and generation with streaming VAE decode.
Loading/warmup, reference-video decoding, metrics and exports are outside it.
"""
from __future__ import annotations

import time
PROCESS_START = time.monotonic()
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from profiling_config.kv_episode_inputs import (episode_manifest, load_episode_cpu,
                                               prepare_inputs, input_hashes, LEGACY_ROOT)
from profiling_config.validate_kv_acceleration import (ForwardTimer, fixed_probe,
    source_identity, error_stats, write_report, json_safe)
from miniworld.vae.codec import StreamingVAEDecoder, load_wan22_vae


class TimedDecoder(StreamingVAEDecoder):
    """CUDA events only: no synchronization between decoded chunks."""
    def __init__(self, vae):
        super().__init__(vae)
        self.events = []

    def step(self, latents_chunk):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        result = super().step(latents_chunk)
        end.record()
        self.events.append((start, end, int(latents_chunk.shape[2]), int(result.shape[2])))
        return result

    def summary(self):
        return dict(gpu_seconds=sum(a.elapsed_time(b) for a, b, _, _ in self.events)/1000,
                    calls=len(self.events), latent_frames=sum(n for _, _, n, _ in self.events),
                    rgb_frames=sum(n for _, _, _, n in self.events),
                    definition='Sum of CUDA decode-step event intervals; not a separate generation benchmark')


def code_identity():
    hashes = source_identity()
    for name in ('evaluate_forcing_kv.py', 'kv_episode_inputs.py', 'kv_video_metrics.py'):
        path = REPO/'profiling_config'/name
        hashes[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('codec.py', 'wan22.py'):
        path = REPO/'miniworld/vae'/name
        if path.exists():
            hashes[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def metric_tail(metric, count=128):
    data = metric['per_frame']
    mse = math.fsum(data['mse_rgb'][-count:])/min(count, len(data['mse_rgb']))
    return dict(frame_start=data['rgb_frame'][-count], frame_end=data['rgb_frame'][-1],
                frame_count=min(count, len(data['rgb_frame'])), mse_global=mse,
                psnr_db_global=None if mse == 0 else -10*math.log10(mse),
                ssim_mean=float(np.mean(data['ssim'][-count:])),
                lpips_alex_mean=float(np.mean(data['lpips_alex'][-count:])))


def set_mode(model, mode, options):
    settings = dict(headcast=False, kv_acceleration='off' if mode == 'baseline' else 'forcing-kv',
                    forcing_kv_threshold=options.threshold, forcing_kv_segments=options.segments,
                    forcing_kv_retention=options.retention,
                    forcing_kv_profile=str(options.profile) if options.profile else None,
                    forcing_kv_profile_output=str(options.profile_output) if options.profile_output else None)
    for key, value in settings.items():
        if not hasattr(model.cfg, key):
            raise RuntimeError(f'Forcing-KV configuration is missing {key}')
        setattr(model.cfg, key, value)
    return settings


def runtime_audit(mode, meta, depth, heads, phase_timing, profile_loaded=False):
    """Require actual physical compaction and all-layer/CFG routing, not a flag."""
    if mode == 'baseline':
        return {'baseline_has_no_acceleration_runtime': not meta.get('kv_acceleration') and not meta.get('headcast')}
    info = meta.get('kv_acceleration', {})
    groups = info.get('groups', [])
    partition = len(groups) == depth
    counts = dict(static=0, dynamic=0)
    for row in groups:
        partition = partition and isinstance(row, dict) and set(row) == set(counts)
        if not partition:
            break
        partition = sorted(row['static'] + row['dynamic']) == list(range(heads))
        for name in counts:
            counts[name] += len(row[name])
    branches = 2 if meta.get('cfg_enabled', True) else 1
    routed = int(info.get('routed_attention_calls', 0))
    compact = int(info.get('compact_attention_calls', 0))
    dense = int(info.get('dense_attention_calls', 0))
    commits = int(info.get('commit_attention_calls', 0))
    checks = dict(runtime_summary_present=bool(info), enabled=info.get('enabled') is True,
                  status_active=info.get('status') == 'active', requested_mode_matches=info.get('mode') == 'forcing-kv',
                  complete_head_partition=partition, reported_group_counts_match=info.get('group_counts') == counts,
                  routed_attention_used=routed > 0, compact_attention_used=compact > 0,
                  all_layer_branch_caches_converted=int(info.get('converted_caches', 0)) >= depth * branches,
                  routed_call_accounting=routed == dense + compact,
                  cache_storage_recorded=int(info.get('final_cache_storage_bytes', 0)) > 0,
                  full_logical_cache_reached=int(info.get('final_logical_history_frames', 0)) == 48,
                  all_context_calls_counted=routed == int(phase_timing['kv_context_forward_count']) * depth,
                  commit_attention_used_and_counted=commits > 0 and commits == int(phase_timing['kv_context_commit_forward_count']) * depth,
                  all_branches_calibrated=int(info.get('calibration_forwards', 0)) == (0 if profile_loaded else branches),
                  expected_profile_loading=info.get('profile_loaded') is profile_loaded,
                  selector_updated=int(info.get('selection_updates', 0)) > 0,
                  selector_updated_both_branches=len(info.get('selection_updates_per_branch', {})) == branches
                      and all(int(value) > 0 for value in info.get('selection_updates_per_branch', {}).values()),
                  fused_current_packing_used=int(info.get('fused_current_pack_calls', 0)) > 0)
    policy = info.get('policy', {})
    checks['policy_active_and_matches_runtime'] = (policy.get('status') == 'active'
        and policy.get('groups') == groups and policy.get('group_counts') == counts)
    cache_summaries = info.get('cache_summaries', {})
    checks['all_layer_branch_cache_summaries'] = (len(cache_summaries) == branches
        and all(len(caches) == depth for caches in cache_summaries.values()))
    checks['history_physically_reduced'] = bool(cache_summaries) and all(
        item.get('logical_frames') == 48
        and bool(item.get('stored_history_tokens_per_head'))
        and all(0 < tokens < 48*300 for tokens in item['stored_history_tokens_per_head'].values())
        for caches in cache_summaries.values() for item in caches)
    return checks


def export_contact_sheet(output_dir, gt, outputs):
    from PIL import Image, ImageDraw
    names = ['GT', 'baseline', 'forcing-kv']
    chosen = [0, 96, 196, 296, 396, 496]
    images = {'GT': gt, **outputs}
    height, width = 240, 320
    sheet = Image.new('RGB', (width*len(chosen), (height+24)*len(names)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(names):
        for col, index in enumerate(chosen):
            frame = (images[name][index].permute(1, 2, 0)*255).round().clamp(0, 255).byte().numpy()
            x, y = col*width, row*(height+24)
            draw.text((x+5, y+5), f'{name} | RGB {index}', fill='black')
            sheet.paste(Image.fromarray(frame), (x, y+24))
    path = output_dir/'comparison.png'
    sheet.save(path)
    return str(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--order', default='baseline,forcing-kv')
    parser.add_argument('--threshold', type=float, default=.8)
    parser.add_argument('--segments', type=int, default=6)
    parser.add_argument('--retention', type=float, default=.33)
    parser.add_argument('--profile', type=Path, help='Reusable head profile; absent means online calibration')
    parser.add_argument('--profile-output', type=Path, help='Export the online profile outside the public PDF directory')
    parser.add_argument('--budget-seconds', type=float, default=600)
    args = parser.parse_args()
    order = args.order.split(',')
    if sorted(order) != sorted(['baseline', 'forcing-kv']):
        parser.error('--order must be a permutation of baseline,forcing-kv')
    if args.profile and args.profile_output:
        parser.error('--profile and --profile-output are mutually exclusive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'evaluation.json'
    if path.exists():
        raise FileExistsError(path)
    def expired(signum, frame):
        raise TimeoutError(f'Worker budget {args.budget_seconds:g} seconds exhausted')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, args.budget_seconds-(time.monotonic()-PROCESS_START)))
    report = dict(schema_version=1, status='running', episode=f'{args.episode:06d}',
                  modes=order, runs=[], metrics={}, checks={},
                  forcing_kv=dict(threshold=args.threshold, segments=args.segments,
                      retention=args.retention, profile=str(args.profile) if args.profile else None,
                      profile_output=str(args.profile_output) if args.profile_output else None),
                  timing=dict(primary='end_to_end_seconds',
                    includes='Initial RGB/actions reads, resize, transfers, initial-frame VAE encode, conditioning/noise construction, complete sampler and streaming VAE RGB decode; GPU output ready',
                    excludes='Model loading, common operation warmup, full GT decode, hashes, quality metrics, CPU output transfer and file export',
                    instrumentation='CUDA events per DiT forward and VAE step, without per-call synchronization; wall time authoritative',
                    repetitions=1, comparison='Same GPU within each episode, mode order reversed across the two episodes',
                    warmup='Dense 48-frame cache with current lengths 16/2/1, both CFG branches; initial-frame VAE encode and streaming VAE decode; no full-rollout warmup',
                    candidate_first_use='Calibration and any candidate kernel first-use compilation remain inside its timed rollout'),
                  metric_protocol=dict(frames='RGB 1..496 inclusive; conditioning frame 0 excluded',
                    video_shape=[497, 3, 240, 320], format='FP32 decoded RGB clamped to [0,1] before evaluation; no video compression',
                    references=['baseline', 'gt'], psnr='-10 log10(MSE over all evaluated RGB pixels)',
                    ssim='Gaussian 11x11 sigma 1.5, valid centers, data_range=1',
                    lpips='Official pretrained calibrated AlexNet v0.1', temporal_subsampling=False))
    original_forward = model = None
    try:
        from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
        from miniworld.forcing_kv import checkpoint_identity
        torch.set_num_threads(4)
        torch.manual_seed(20261016+args.episode)
        report['source'] = episode_manifest(args.episode, 125)
        report['source_sha256'] = code_identity()
        report['gpu'] = torch.cuda.get_device_name()
        report['visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        report['torch_version'] = torch.__version__
        report['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        manifest = json.loads((LEGACY_ROOT/'experiment_manifest.json').read_text())
        options = dict(manifest['args'])
        options['forcing_kv_model_identity'] = checkpoint_identity(options['checkpoint'])
        report['checkpoints'] = {name: options[name] for name in ('checkpoint', 'vae_checkpoint')}
        report['model_identity'] = options['forcing_kv_model_identity']
        raw = torch.load(options['checkpoint'], map_location='cpu', weights_only=True)
        weights = raw.get('ema_model', raw.get('model', raw))
        options.update(latent_size=(15, 20), latent_frames=int(weights['net.feat_rope.freqs_cos'].shape[0]//300),
                       wm_use_checkpoint=False, headcast=False, kv_acceleration='off')
        model = build_denoiser_from_mode(DenoiserConfig(**options))
        model.load_state_dict(weights, strict=True)
        del raw, weights
        model = model.cuda().eval()
        vae = load_wan22_vae(options)
        original_forward = model.net.forward_with_cache
        # Decode the reference once, outside every generation clock.
        gt_data = load_episode_cpu(args.episode, include_gt=True)
        gt = torch.from_numpy(gt_data['gt_rgb']).permute(0, 3, 1, 2).float().div_(255)
        np.save(args.output_dir/'gt_rgb.npy', gt_data['gt_rgb'])
        report['load_and_reference_seconds'] = time.monotonic()-PROCESS_START
        started = time.perf_counter()
        warm_inputs = prepare_inputs(gt_data, vae, 'cuda')
        baseline_probe = torch.load('/home/jian/Documents/Codex/2026-10-19/hello/work/headcast_migration/run_001/baseline_latents.pt', map_location='cpu', weights_only=True).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            off_reference = fixed_probe(model, baseline_probe, warm_inputs['cond_seq'], warm_inputs['noise'], 1)
            decoder = StreamingVAEDecoder(vae)
            decoder.begin()
            try:
                for start, stop in ((0, 2), (2, 4), (4, 5)):
                    decoded = decoder.step(baseline_probe[:, :, start:stop])
                    del decoded
            finally:
                decoder.end()
        torch.cuda.synchronize()
        report['warmup_seconds'] = time.perf_counter()-started
        del decoder, warm_inputs, gt_data
        sampler = dict(total_len=125, history_len=1, max_cache_chunks=24,
                       inflight_chunks=8, sink_frames=1, collect_stream_timing=False)
        report['sampler'] = sampler
        outputs = {}
        write_report(path, report)
        for index, mode in enumerate(order):
            print(f'[{args.episode:06d}] starting {mode} ({index+1}/2)', flush=True)
            settings = set_mode(model, mode, args)
            timer = ForwardTimer(original_forward)
            model.net.forward_with_cache = timer
            decoder = TimedDecoder(vae)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                cpu_data = load_episode_cpu(args.episode, include_gt=False)
                inputs = prepare_inputs(cpu_data, vae, 'cuda')
                torch.cuda.synchronize()
                prep_end = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred_latents, pred_rgb = model.generate_eval_latents_streaming(
                        inputs['latents'], inputs['cond_seq'], noise=inputs['noise'],
                        stream_decoder=decoder, **sampler)
                torch.cuda.synchronize()
                ended = time.perf_counter()
            finally:
                model.net.forward_with_cache = original_forward
            row = dict(mode=mode, order=index, configuration=settings,
                       end_to_end_seconds=ended-started, input_prep_seconds=prep_end-started,
                       sampler_stream_seconds=ended-prep_end,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                       latent_shape=list(pred_latents.shape), rgb_shape=list(pred_rgb.shape),
                       finite=bool(torch.isfinite(pred_latents).all() and torch.isfinite(pred_rgb).all()),
                       input_hashes=input_hashes(inputs), sampler_meta=json_safe(model.last_eval_meta),
                       phase_timing=timer.summary(), vae_decode=decoder.summary())
            row['phase_timing']['interval_definition'] += '; this streaming run also includes interleaved VAE decoding'
            row['runtime_checks'] = runtime_audit(mode, row['sampler_meta'], model.net.depth,
                                                  model.net.num_heads, row['phase_timing'], bool(args.profile))
            row['runtime_active'] = all(row['runtime_checks'].values())
            if not row['finite'] or not row['runtime_active']:
                report['runs'].append(row)
                write_report(path, report)
                raise AssertionError(f'Invalid rollout: {mode} {row["runtime_checks"]}')
            if row['rgb_shape'] != [1, 3, 497, 240, 320]:
                raise AssertionError(f'Unexpected video length or shape: {row["rgb_shape"]}')
            rgb = pred_rgb[0].permute(1, 0, 2, 3).float().add(1).mul(.5).clamp(0, 1).cpu()
            outputs[mode] = rgb
            latent_path = args.output_dir/f'{mode}_latents.pt'
            torch.save(pred_latents.cpu(), latent_path)
            if mode == 'baseline':
                prior_path = Path('/home/jian/Documents/Codex/2026-10-19/hello/work/kv_multiepisode')/f'episode_{args.episode:06d}'/'baseline_latents.pt'
                if prior_path.exists():
                    previous = torch.load(prior_path, map_location='cpu', weights_only=True)
                    report['previous_baseline_sanity'] = dict(path=str(prior_path),
                        **error_stats(previous, pred_latents.cpu()))
                    del previous
            rgb_path = args.output_dir/f'{mode}_rgb_fp32.npy'
            np.save(rgb_path, rgb.numpy())
            row['outputs'] = dict(latents=str(latent_path), rgb=str(rgb_path))
            report['runs'].append(row)
            write_report(path, report)
            print(f'[{args.episode:06d}] {mode}: E2E {row["end_to_end_seconds"]:.3f}s, sampler+VAE {row["sampler_stream_seconds"]:.3f}s, VAE GPU {row["vae_decode"]["gpu_seconds"]:.3f}s', flush=True)
            del pred_latents, pred_rgb, rgb, cpu_data, timer, decoder
        report['checks']['all_mode_input_hashes_equal'] = all(row['input_hashes'] == report['runs'][0]['input_hashes'] for row in report['runs'])
        set_mode(model, 'baseline', args)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            off_after = fixed_probe(model, baseline_probe, inputs['cond_seq'], inputs['noise'], 1)
        report['off_after_on_probe'] = error_stats(off_reference, off_after)
        report['checks']['off_after_on_bitwise_equal'] = report['off_after_on_probe']['bitwise_equal']
        report['checks']['all_rollouts_valid'] = all(row['finite'] and row['runtime_active'] for row in report['runs'])
        del model, vae, baseline_probe, inputs, off_after, off_reference, original_forward
        model = original_forward = None
        gc.collect()
        torch.cuda.empty_cache()
        report['status'] = 'metrics_running'
        write_report(path, report)
        from profiling_config.kv_video_metrics import VideoMetricCalculator
        calculator = VideoMetricCalculator(device='cuda:0', batch_size=8)
        for mode in ('baseline', 'forcing-kv'):
            report['metrics'][mode] = {}
            for reference in ('baseline', 'gt'):
                if mode == 'baseline' and reference == 'baseline':
                    continue
                print(f'[{args.episode:06d}] metrics {mode} vs {reference}', flush=True)
                metric = calculator(outputs['baseline'] if reference == 'baseline' else gt, outputs[mode])
                metric['reference'] = reference
                metric['tail_128'] = metric_tail(metric)
                report['metrics'][mode][reference] = metric
                write_report(path, report)
        fields = ['episode', 'mode', 'reference', 'rgb_frame', 'mse_rgb', 'psnr_db', 'psnr_is_positive_infinity', 'ssim', 'lpips_alex']
        csv_path = args.output_dir/'per_frame_metrics.csv'
        with csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for mode, references in report['metrics'].items():
                for reference, metric in references.items():
                    for index in range(metric['summary']['frame_count']):
                        writer.writerow(dict(episode=report['episode'], mode=mode, reference=reference,
                            **{name: values[index] for name, values in metric['per_frame'].items()}))
        report['artifacts'] = dict(per_frame_csv=str(csv_path), contact_sheet=export_contact_sheet(args.output_dir, gt, outputs))
        report['checks']['source_unchanged'] = code_identity() == report['source_sha256']
        report['checks']['all_metric_frames_complete'] = all(metric['summary']['frame_count'] == 496 for refs in report['metrics'].values() for metric in refs.values())
        if not all(report['checks'].values()):
            raise AssertionError(f'Evaluation checks failed: {report["checks"]}')
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if model is not None and original_forward is not None:
            model.net.forward_with_cache = original_forward
        report['worker_wall_seconds'] = time.monotonic()-PROCESS_START
        write_report(path, report)
    print(f'[{args.episode:06d}] status={report["status"]}; worker {report["worker_wall_seconds"]:.1f}s', flush=True)
    if report['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
