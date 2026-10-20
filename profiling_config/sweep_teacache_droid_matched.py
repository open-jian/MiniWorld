"""Matched-length DROID TeaCache sweep, with fresh baselines for every episode.

The default 64 latents produce 253 RGB frames, matching the RE10K sweep.
Each worker loads its models once and measures all thresholds independently
with identical source inputs/noise. Historical 497-frame runs are never imported.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from profiling_config.evaluate_teacache import (
    BASELINE_PROBE, SAMPLER, code_identity, runtime_audit, set_mode)
from profiling_config.evaluate_kv_episodes import TimedDecoder, metric_tail
from profiling_config.kv_episode_inputs import (
    LEGACY_ROOT, episode_manifest, input_hashes, load_episode_cpu, prepare_inputs)
from profiling_config.validate_kv_acceleration import (
    ForwardTimer, error_stats, fixed_probe, json_safe, write_report)
from miniworld.vae.codec import StreamingVAEDecoder, load_wan22_vae


def threshold_key(threshold):
    return 'baseline' if threshold is None else 'tau_' + format(threshold, '.9g').replace('.', 'p')


def ordered_thresholds(thresholds, episode):
    """Deterministic rotations and direction; independent of measured results."""
    values = sorted(set(thresholds))
    rotation = int(episode) % len(values)
    values = values[rotation:] + values[:rotation]
    return values if (int(episode) // len(values)) % 2 == 0 else list(reversed(values))


def identities():
    result = code_identity()
    result[str(Path(__file__).relative_to(REPO))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def validate_reference(report, source, profile, total_latents):
    keys = ('source_video_sha256', 'source_actions_sha256', 'normalization_stats_sha256',
            'modality_sha256', 'seed', 'total_latents', 'requested_rgb_frames', 'resize_hw')
    checks = dict(source_matches_current=all(report['source'][key] == source[key] for key in keys),
                  checkpoints_match_profile=report['checkpoints'] == profile['checkpoints'],
                  sampler_matches_current=report['sampler'] == dict(total_len=total_latents, **SAMPLER),
                  current_measurement=report['provenance']['kind'] == 'current',
                  prior_checks_pass=bool(report.get('checks')) and all(report['checks'].values()),
                  has_gt_metrics='gt' in report['metrics'],
                  all_metric_frames_complete=all(
                      item['summary']['frame_count'] == 4*(total_latents-1)
                      for item in report['metrics'].values()))
    if not all(checks.values()):
        raise ValueError(f'Resumed reference protocol mismatch: {checks}')
    return checks


def load_complete(path, resume, threshold, profile):
    if not path.exists():
        return None
    if not resume:
        raise FileExistsError(f'{path}; use --resume to preserve completed points')
    result = json.loads(path.read_text())
    if result.get('status') != 'complete':
        return None
    if result['threshold'] != threshold:
        raise ValueError(f'Existing threshold differs: {path}')
    if threshold is not None:
        settings = result['run']['configuration']
        if (list(settings['teacache_coefficients']) != profile['coefficients']
                or settings['teacache_threshold'] != threshold
                or settings['teacache_max_reuse_steps'] != profile.get('max_reuse_steps', 0)):
            raise ValueError(f'Existing TeaCache policy differs: {path}')
    for filename in result['run']['outputs'].values():
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)
    return result


TIMING = dict(primary='end_to_end_seconds', repetitions=1,
    includes='Fresh initial RGB/action reads, resize, transfers, initial-frame VAE encode, conditioning/noise construction, complete sampler and streaming VAE RGB decode; GPU output ready',
    excludes='Model loading, common warmup, calibration/profile fitting, full GT decode, hashes, quality metrics, CPU output transfer and exports',
    instrumentation='CUDA events per DiT forward and VAE step; no per-call synchronization added by timer',
    resident_metric_model='LPIPS AlexNet loaded once before all new measured rollouts; resident allocation but no metric kernels run during generation',
    comparison='One fixed-noise full rollout per episode/threshold; all original and accelerated points are freshly measured at the same video length')


def metric_protocol(total_latents):
    rgb_frames = 4*(total_latents-1)+1
    return dict(frames=f'RGB 1..{rgb_frames-1} inclusive; conditioning frame zero excluded',
        video_shape=[rgb_frames, 3, 240, 320],
        format='Uncompressed decoded FP32 RGB clamped to [0,1]',
        references=['gt', 'baseline'], psnr='-10 log10(MSE over all evaluated RGB pixels)',
        ssim='Gaussian 11x11 sigma1.5, valid centers, data_range1',
        lpips='Official pretrained calibrated AlexNet v0.1', temporal_subsampling=False)


def call_counts(run):
    rows = run['phase_timing']['by_shape']
    return {kind: sum(row['count'] for row in rows if row['kind'] == kind)
            for kind in ('denoise', 'commit')}


@torch.inference_mode()
def measure_point(model, vae, calculator, args, profile, episode, threshold,
                  point_dir, gt, baseline, source, worker_source, order):
    mode = 'baseline' if threshold is None else 'teacache'
    point_dir.mkdir(parents=True, exist_ok=True)
    path = point_dir / 'point.json'
    settings_profile = dict(profile)
    if threshold is not None:
        settings_profile['threshold'] = threshold
    settings = set_mode(model, mode, args, settings_profile)
    result = dict(schema_version=1, status='running', episode=f'{episode:06d}',
                  dataset='droid', mode=mode, threshold=threshold, label=threshold_key(threshold),
                  source=source, source_sha256=worker_source, metrics={}, checks={},
                  checkpoints=profile['checkpoints'], sampler=dict(total_len=args.total_latents, **SAMPLER),
                  timing=TIMING, metric_protocol=metric_protocol(args.total_latents),
                  provenance=dict(kind='current', gpu=torch.cuda.get_device_name(),
                      visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      torch_version=torch.__version__, git_head=args.git_head,
                      imported_wall_time_is_historical=False))
    write_report(path, result)
    original_forward = model.net.forward_with_cache
    timer, decoder = ForwardTimer(original_forward), TimedDecoder(vae)
    try:
        if identities() != worker_source:
            raise AssertionError('Source changed before measured rollout')
        print(f'[{episode:06d}] start {result["label"]}', flush=True)
        model.net.forward_with_cache = timer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            cpu_data = load_episode_cpu(episode, args.total_latents, include_gt=False)
            inputs = prepare_inputs(cpu_data, vae, 'cuda')
            torch.cuda.synchronize()
            prep_end = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                latents, video = model.generate_eval_latents_streaming(
                    inputs['latents'], inputs['cond_seq'], noise=inputs['noise'],
                    stream_decoder=decoder, **result['sampler'])
            torch.cuda.synchronize()
            ended = time.perf_counter()
        finally:
            model.net.forward_with_cache = original_forward
        run = dict(mode=mode, order=order, configuration=settings,
                   end_to_end_seconds=ended-started, input_prep_seconds=prep_end-started,
                   sampler_stream_seconds=ended-prep_end,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                   latent_shape=list(latents.shape), rgb_shape=list(video.shape),
                   finite=bool(torch.isfinite(latents).all() and torch.isfinite(video).all()),
                   input_hashes=input_hashes(inputs), sampler_meta=json_safe(model.last_eval_meta),
                   phase_timing=timer.summary(), vae_decode=decoder.summary())
        result['run'] = run
        checks = runtime_audit(mode, run['sampler_meta'], run['phase_timing'])
        # A small threshold may correctly trigger full computation on every call.
        checks.pop('residual_reuse_used', None)
        checks.update(finite=run['finite'], expected_rgb_shape=run['rgb_shape'] == [1, 3, args.rgb_frames, 240, 320],
                      source_unchanged=identities() == worker_source)
        run['call_counts'] = call_counts(run)
        if baseline is not None:
            checks['inputs_match_baseline'] = run['input_hashes'] == baseline['run']['input_hashes']
            checks['call_counts_match_baseline'] = run['call_counts'] == call_counts(baseline['run'])
        if mode == 'teacache':
            info = run['sampler_meta']['teacache']
            checks['denoising_calls_match_timer'] = info['denoising_forwards'] == run['call_counts']['denoise']
            checks['commit_calls_match_timer'] = info['commit_forwards'] == run['call_counts']['commit']
        result['checks'].update(checks)
        run['runtime_checks'] = checks
        if not all(checks.values()):
            raise AssertionError(f'Invalid rollout: {checks}')
        rgb = video[0].permute(1, 0, 2, 3).float().add(1).mul(.5).clamp(0, 1).cpu()
        latent_path, rgb_path = point_dir / 'latents.pt', point_dir / 'rgb_fp32.npy'
        torch.save(latents.cpu(), latent_path)
        np.save(rgb_path, rgb.numpy())
        run['outputs'] = dict(latents=str(latent_path), rgb=str(rgb_path))
        result['status'] = 'metrics_running'
        write_report(path, result)
        del inputs, cpu_data, video, latents, timer, decoder
        metric = calculator(gt, rgb)
        metric.update(reference='gt', tail_128=metric_tail(metric, min(128, args.rgb_frames-1)))
        result['metrics']['gt'] = metric
        if mode == 'teacache':
            baseline_array = np.load(baseline['run']['outputs']['rgb'], mmap_mode='c')
            baseline_rgb = torch.from_numpy(baseline_array)
            metric = calculator(baseline_rgb, rgb)
            metric.update(reference='baseline', tail_128=metric_tail(metric, min(128, args.rgb_frames-1)))
            result['metrics']['baseline'] = metric
            del baseline_array, baseline_rgb
        result['checks']['all_metric_frames_complete'] = all(
            item['summary']['frame_count'] == args.rgb_frames-1 for item in result['metrics'].values())
        if not all(result['checks'].values()):
            raise AssertionError(f'Point validation failed: {result["checks"]}')
        result['status'] = 'complete'
        write_report(path, result)
        tea = run['sampler_meta'].get('teacache') or {}
        print(f'[{episode:06d}] complete {result["label"]}: {run["end_to_end_seconds"]:.3f}s; '
              f'reused {tea.get("reused_forwards", 0)}; '
              f'GT PSNR {result["metrics"]["gt"]["summary"]["psnr_db_global"]:.4f}', flush=True)
        return result
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        write_report(path, result)
        raise
    finally:
        model.net.forward_with_cache = original_forward


@torch.inference_mode()
def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, nargs='+', required=True)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.025, .05, .1, .2, .4])
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--total-latents', type=int, default=64)
    parser.add_argument('--budget-seconds', type=float, default=1200)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.total_latents < 64:
        parser.error('total-latents must be at least 64 for the common 48-history/16-current probe')
    args.rgb_frames = 4*(args.total_latents-1)+1
    if not all(math.isfinite(x) and x >= 0 for x in args.thresholds):
        parser.error('thresholds must be finite and nonnegative')
    if len(set(args.episodes)) != len(args.episodes):
        parser.error('episodes must be distinct within each worker')
    args.threshold, args.max_reuse_steps = None, None
    args.git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker_path = args.output_root / ('worker_' + '_'.join(f'{ep:06d}' for ep in args.episodes) + '.json')
    worker_source = identities()
    profile = json.loads(args.profile.read_text())
    if not profile.get('coefficients'):
        raise ValueError('Profile requires fixed calibrated coefficients')
    worker = dict(schema_version=1, status='running', dataset='droid', episodes=args.episodes,
                  total_latents=args.total_latents, rgb_frames=args.rgb_frames,
                  thresholds=sorted(set(args.thresholds)), profile=profile, profile_path=str(args.profile),
                  source_sha256=worker_source, points=[], timing=TIMING, checks={},
                  threshold_order={str(ep): ordered_thresholds(args.thresholds, ep) for ep in args.episodes})
    def expired(signum, frame):
        raise TimeoutError(f'Worker exceeded {args.budget_seconds:g} second wall-time budget')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.budget_seconds)
    write_report(worker_path, worker)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(20261016 + args.episodes[0])
        from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
        from profiling_config.kv_video_metrics import VideoMetricCalculator
        options = dict(json.loads((LEGACY_ROOT / 'experiment_manifest.json').read_text())['args'])
        for key in ('checkpoint', 'vae_checkpoint'):
            if options[key] != profile['checkpoints'][key]:
                raise ValueError(f'Profile checkpoint mismatch: {key}')
        raw = torch.load(options['checkpoint'], map_location='cpu', weights_only=True)
        weights = raw.get('ema_model', raw.get('model', raw))
        options.update(latent_size=(15, 20), latent_frames=int(weights['net.feat_rope.freqs_cos'].shape[0]//300),
                       wm_use_checkpoint=False, headcast=False, kv_acceleration='off', teacache=False,
                       teacache_calibration=False)
        model = build_denoiser_from_mode(DenoiserConfig(**options))
        model.load_state_dict(weights, strict=True)
        del raw, weights
        model = model.cuda().eval()
        vae = load_wan22_vae(options)
        calculator = VideoMetricCalculator(device='cuda:0', batch_size=8)
        probe = torch.load(BASELINE_PROBE, map_location='cpu', weights_only=True).cuda()
        for episode in args.episodes:
            episode_dir = args.output_root / f'episode_{episode:06d}'
            episode_dir.mkdir(parents=True, exist_ok=True)
            source = dict(episode_manifest(episode, args.total_latents), dataset='droid')
            baseline = None
            existing = {}
            for threshold in [None, *sorted(set(args.thresholds))]:
                path = episode_dir / threshold_key(threshold) / 'point.json'
                point = load_complete(path, args.resume, threshold, profile)
                if point is not None:
                    validate_reference(point, source, profile, args.total_latents)
                    if point['source_sha256'] != worker_source:
                        raise ValueError(f'Resumed point source code differs: {path}')
                    existing[threshold] = point
                    worker['points'].append(str(path))
                    if threshold is None:
                        baseline = point
            gt_data = load_episode_cpu(episode, args.total_latents, include_gt=True)
            gt = torch.from_numpy(gt_data['gt_rgb']).permute(0, 3, 1, 2).float().div_(255)
            np.save(episode_dir / 'gt_rgb.npy', gt_data['gt_rgb'])
            write_report(episode_dir / 'source.json', source)
            set_mode(model, 'baseline', args, profile)
            warm_inputs = prepare_inputs(gt_data, vae, 'cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                off_reference = fixed_probe(model, probe, warm_inputs['cond_seq'], warm_inputs['noise'], 1)
                decoder = StreamingVAEDecoder(vae)
                decoder.begin()
                try:
                    for begin, end in ((0, 2), (2, 4), (4, 5)):
                        decoder.step(probe[:, :, begin:end])
                finally:
                    decoder.end()
            torch.cuda.synchronize()
            del decoder, gt_data
            jobs = ([] if baseline is not None else [None]) + [
                value for value in ordered_thresholds(args.thresholds, episode) if value not in existing]
            for order, threshold in enumerate(jobs):
                point_dir = episode_dir / threshold_key(threshold)
                point = measure_point(model, vae, calculator, args, profile, episode, threshold,
                                      point_dir, gt, baseline, source, worker_source, order)
                if threshold is None:
                    baseline = point
                worker['points'].append(str(point_dir / 'point.json'))
                write_report(worker_path, worker)
            set_mode(model, 'baseline', args, profile)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                off_after = fixed_probe(model, probe, warm_inputs['cond_seq'], warm_inputs['noise'], 1)
            parity = error_stats(off_reference, off_after)
            worker['checks'][f'{episode:06d}_off_after_on_bitwise_equal'] = parity['bitwise_equal']
            if not parity['bitwise_equal']:
                raise AssertionError(f'Off-after-on parity failed on {episode}')
            write_report(episode_dir / 'isolation.json', parity)
            del gt, warm_inputs, off_reference, off_after
            gc.collect()
        worker['checks']['source_unchanged'] = identities() == worker_source
        if not all(worker['checks'].values()):
            raise AssertionError(f'Worker checks failed: {worker["checks"]}')
        worker['status'] = 'complete'
    except BaseException as error:
        worker.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        worker['worker_wall_seconds'] = time.monotonic() - started
        write_report(worker_path, worker)
    print(f'worker status={worker["status"]}; {worker["worker_wall_seconds"]:.1f}s', flush=True)
    if worker['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
