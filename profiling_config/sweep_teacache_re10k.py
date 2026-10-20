"""Paired RE10K TeaCache threshold sweep with real camera conditioning.

Calibration uses a separate explicit clip and never reuses residuals. Evaluation
uses only the first RGB frame plus the recorded camera trajectory as inputs;
all future RGB is held out for metrics. Model defaults remain untouched.
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
from miniworld.conditioning.actions import ConditioningConfig, build_cond_seq_for_batch
from miniworld.vae.codec import StreamingVAEDecoder, load_wan22_vae, vae_encode
from profiling_config.evaluate_teacache import (
    SAMPLER, code_identity, runtime_audit, screen_thresholds, set_mode)
from profiling_config.evaluate_kv_episodes import TimedDecoder, metric_tail
from profiling_config.kv_episode_inputs import input_hashes, sha256_file
from profiling_config.sweep_teacache_thresholds import ordered_thresholds, threshold_key
from profiling_config.validate_kv_acceleration import (
    ForwardTimer, error_stats, fixed_probe, json_safe, write_report)


def clip_seed(clip):
    return 20261019 + int(hashlib.sha256(clip.encode()).hexdigest()[:8], 16)


def identities():
    result = code_identity()
    for name in ('profiling_config/sweep_teacache_re10k.py',
                 'profiling_config/sweep_teacache_thresholds.py',
                 'miniworld/conditioning/actions.py', 'miniworld/conditioning/poses.py',
                 'miniworld/data/re10k.py'):
        result[name] = hashlib.sha256((REPO / name).read_bytes()).hexdigest()
    return result


def source_manifest(args, clip, total_latents):
    if Path(clip).name != clip or clip in ('.', '..'):
        raise ValueError('Clips must be file stems, not paths')
    video, pose = args.data_root / f'{clip}.mp4', args.pose_dir / f'{clip}.pt'
    if not video.is_file() or not pose.is_file():
        raise FileNotFoundError(f'Missing paired RE10K files: {video}, {pose}')
    count = 4 * (total_latents - 1) + 1
    return dict(dataset='re10k', episode=clip, clip=clip, seed=clip_seed(clip),
        source_video=str(video), source_poses=str(pose),
        source_video_sha256=sha256_file(video), source_poses_sha256=sha256_file(pose),
        total_latents=total_latents, requested_rgb_frames=count,
        reference_frame_indices=list(range(count)), pose_frame_indices=list(range(count)),
        resize_hw=[240, 320], resize_implementation='decord.VideoReader(width=320,height=240)',
        rgb_normalization='CPU FP32 uint8 /255 -> *2 -1; no augmentation',
        pose_layout='18D stored pose ->16D: concatenate columns[:4] and[6:]; normalized intrinsics and world-to-camera extrinsics',
        conditioning='Official MiniWorld four-pose-per-latent ray encoding, 15 frequencies; cond_dim720 per token; relative to initial pose; normalize_trans=False',
        initial_latent_policy='Only initial RGB encoded; all future latent placeholders zero; future GT never used by generation',
        noise_policy='Isolated CUDA generator with deterministic SHA256-derived clip seed',
        dataset_split=args.dataset_split)


def load_clip_cpu(source, *, include_gt):
    from decord import VideoReader, cpu
    count = source['requested_rgb_frames']
    reader = VideoReader(source['source_video'], ctx=cpu(0), width=320, height=240, num_threads=2)
    if len(reader) < count:
        raise ValueError(f"Clip {source['clip']} has {len(reader)} RGB frames; need {count}")
    poses = torch.load(source['source_poses'], map_location='cpu', weights_only=True)
    if not isinstance(poses, torch.Tensor) or poses.ndim != 2 or poses.shape[1] != 18 or len(poses) < count:
        raise ValueError(f"Invalid RE10K stored poses for {source['clip']}: {getattr(poses, 'shape', None)}")
    poses = torch.cat([poses[:count, :4], poses[:count, 6:]], dim=-1).float().unsqueeze(0)
    if not torch.isfinite(poses).all() or not (poses[..., :2] > 0).all():
        raise ValueError('Poses must be finite with positive focal lengths')
    rgb = reader.get_batch(list(range(count)) if include_gt else [0]).asnumpy()
    return dict(gt_rgb=rgb, poses=poses, manifest=source,
                source_rgb_frames=len(reader), source_fps=float(reader.get_avg_fps()))


@torch.inference_mode()
def prepare_inputs(data, vae, device):
    device = torch.device(device)
    source = data['manifest']
    initial = torch.from_numpy(np.ascontiguousarray(data['gt_rgb'][:1]))
    video = initial.float().div(255).mul(2).sub(1).permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        first = vae_encode(vae, video)
    if first.ndim != 5 or first.shape[2:] != (1, 15, 20):
        raise RuntimeError(f'Unexpected seed encoding {tuple(first.shape)}')
    length = source['total_latents']
    latents = first.new_zeros((first.shape[0], first.shape[1], length, 15, 20))
    latents[:, :, :1].copy_(first)
    cond = build_cond_seq_for_batch(cfg=ConditioningConfig(use_pose_cond=True, pose_enc_freq=15),
        poses=data['poses'].to(device), actions=None, t_latent=length, h_lat=15, w_lat=20)
    if list(cond.shape) != [1, length, 720, 15, 20]:
        raise RuntimeError(f'Unexpected camera conditioning {tuple(cond.shape)}')
    generator = torch.Generator(device=device).manual_seed(source['seed'])
    noise = torch.randn(latents.shape, device=device, dtype=torch.float32, generator=generator)
    return dict(latents=latents, cond_seq=cond, noise=noise)


TIMING = dict(primary='end_to_end_seconds', repetitions=1,
    includes='Fresh initial RGB and pose reads, resize, transfers, initial-frame VAE encoding, camera ray encoding/noise construction, complete sampler and streaming VAE decode to GPU-ready RGB',
    excludes='Model loading, common warmup, calibration/profile fitting, full GT decode, hashes, metrics, CPU output transfer and artifact writes',
    instrumentation='CUDA events per DiT forward and VAE step; no per-call synchronization added by timer',
    resident_metric_model='LPIPS AlexNet resident throughout measured rollouts; no metric kernels overlap generation',
    comparison='One fixed-noise full rollout per clip/threshold; baseline newly measured on same worker/GPU')


def metric_protocol(count):
    return dict(frames=f'RGB 1..{count-1}; conditioning frame zero excluded',
        video_shape=[count, 3, 240, 320], format='Uncompressed decoded FP32 RGB clamped to[0,1]',
        references=['gt', 'baseline'], psnr='-10 log10(MSE over all evaluated RGB pixels)',
        ssim='Gaussian11x11 sigma1.5 valid centers data_range1',
        lpips='Official pretrained calibrated AlexNet v0.1', temporal_subsampling=False)


def resume_point(path, args, source, threshold, profile, worker_source, checkpoints):
    if not path.exists():
        return None
    if not args.resume:
        raise FileExistsError(f'{path}; use --resume to retain completed points')
    result = json.loads(path.read_text())
    if result.get('status') != 'complete':
        return None
    if (result['source'] != source or result['threshold'] != threshold
            or result['checkpoints'] != checkpoints or result['source_sha256'] != worker_source
            or result['sampler'] != dict(total_len=source['total_latents'], **SAMPLER)
            or not all(result['checks'].values())):
        raise ValueError(f'Existing point has incompatible source/settings: {path}')
    if threshold is not None:
        settings = result['run']['configuration']
        if (settings['teacache_threshold'] != threshold
                or list(settings['teacache_coefficients']) != profile['coefficients']
                or settings['teacache_max_reuse_steps'] != profile.get('max_reuse_steps', 0)):
            raise ValueError(f'Existing point has incompatible TeaCache policy: {path}')
    for filename in result['run']['outputs'].values():
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)
    return result


@torch.inference_mode()
def measure_point(model, vae, calculator, args, profile, source, threshold,
                  point_dir, gt, baseline, worker_source, checkpoints, order,
                  profile_path=None):
    mode = 'baseline' if threshold is None else 'teacache'
    point_dir.mkdir(parents=True, exist_ok=True)
    path = point_dir / 'point.json'
    effective = dict(profile)
    if threshold is not None:
        effective['threshold'] = threshold
    settings = set_mode(model, mode, args, effective)
    count = source['requested_rgb_frames']
    result = dict(schema_version=1, status='running', dataset='re10k', episode=source['clip'],
        mode=mode, threshold=threshold, label=point_dir.name,
        comparison_group='transferred_droid_profile' if point_dir.name.startswith('transfer_') else 'dataset_calibrated',
        source=source, source_sha256=worker_source, checks={}, metrics={},
        checkpoints=checkpoints, sampler=dict(total_len=source['total_latents'], **SAMPLER),
        timing=TIMING, metric_protocol=metric_protocol(count),
        provenance=dict(kind='current', gpu=torch.cuda.get_device_name(),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=torch.__version__,
            git_head=args.git_head, imported_wall_time_is_historical=False,
            profile_path=str(profile_path or args.profile), profile_dataset=profile.get('dataset', 'droid'),
            transferred_profile=profile.get('checkpoints') != checkpoints))
    write_report(path, result)
    original = model.net.forward_with_cache
    timer, decoder = ForwardTimer(original), TimedDecoder(vae)
    try:
        if identities() != worker_source:
            raise AssertionError('Source changed before rollout')
        print(f"[{source['clip']}] start {threshold_key(threshold)}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        model.net.forward_with_cache = timer
        started = time.perf_counter()
        try:
            cpu_data = load_clip_cpu(source, include_gt=False)
            inputs = prepare_inputs(cpu_data, vae, 'cuda')
            torch.cuda.synchronize()
            prep_end = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                latents, video = model.generate_eval_latents_streaming(inputs['latents'], inputs['cond_seq'],
                    noise=inputs['noise'], stream_decoder=decoder, **result['sampler'])
            torch.cuda.synchronize()
            ended = time.perf_counter()
        finally:
            model.net.forward_with_cache = original
        run = dict(mode=mode, order=order, configuration=settings, end_to_end_seconds=ended-started,
            input_prep_seconds=prep_end-started, sampler_stream_seconds=ended-prep_end,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            latent_shape=list(latents.shape), rgb_shape=list(video.shape),
            finite=bool(torch.isfinite(latents).all() and torch.isfinite(video).all()),
            input_hashes=input_hashes(inputs), sampler_meta=json_safe(model.last_eval_meta),
            phase_timing=timer.summary(), vae_decode=decoder.summary())
        result['run'] = run
        checks = runtime_audit(mode, run['sampler_meta'], run['phase_timing'])
        checks.pop('residual_reuse_used', None)
        checks.update(finite=run['finite'], expected_rgb_shape=run['rgb_shape'] == [1, 3, count, 240, 320],
            source_unchanged=identities() == worker_source)
        if baseline is not None:
            checks['inputs_match_baseline'] = run['input_hashes'] == baseline['run']['input_hashes']
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
        metric.update(reference='gt', tail_128=metric_tail(metric))
        result['metrics']['gt'] = metric
        if mode == 'teacache':
            reference = torch.from_numpy(np.load(baseline['run']['outputs']['rgb'], mmap_mode='c'))
            metric = calculator(reference, rgb)
            metric.update(reference='baseline', tail_128=metric_tail(metric))
            result['metrics']['baseline'] = metric
        result['checks']['all_metric_frames_complete'] = all(
            item['summary']['frame_count'] == count-1 for item in result['metrics'].values())
        if not all(result['checks'].values()):
            raise AssertionError(f"Point validation failed: {result['checks']}")
        result['status'] = 'complete'
        write_report(path, result)
        tea = run['sampler_meta'].get('teacache') or {}
        print(f"[{source['clip']}] complete {threshold_key(threshold)}: {run['end_to_end_seconds']:.3f}s; "
              f"reused {tea.get('reused_forwards', 0)}; GT PSNR {result['metrics']['gt']['summary']['psnr_db_global']:.4f}", flush=True)
        return result
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        write_report(path, result)
        raise
    finally:
        model.net.forward_with_cache = original


def model_options(args, weights, meta):
    from miniworld.sample import resolve_latent_frames, resolve_wm_model
    args.resize_h, args.resize_w, args.spatial_downsample = 240, 320, 16
    return dict(wm_model=resolve_wm_model(meta, args),
        latent_frames=resolve_latent_frames(weights, meta, args), latent_size=(15, 20), latent_channels=48,
        wm_use_checkpoint=False, cond_dim=720, cond_per_token=True, adaln_mode='adaln_lora',
        cond_dropout_prob=.1, df_chunk_size=2, df_ardiff_step=5, timestep_baseshift=2.667,
        timestep_shift=-1., num_sampling_steps=100, cfg_scale=2., cfg_interval_min=.2,
        cfg_interval_max=1., headcast=False, kv_acceleration='off', teacache=False,
        teacache_calibration=False, checkpoint=str(args.checkpoint), vae_checkpoint=str(args.vae_checkpoint))


@torch.inference_mode()
def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips', nargs='+', default=[])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pose-dir', type=Path, required=True)
    parser.add_argument('--dataset-split', default='test')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--vae-checkpoint', type=Path, required=True)
    parser.add_argument('--wm-model', default='1B')
    parser.add_argument('--total-latents', type=int, default=64)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.025, .05, .1, .2, .4])
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--allow-profile-transfer', action='store_true')
    parser.add_argument('--transfer-profile', type=Path,
                        help='Optional fixed DROID coefficients evaluated separately from the dataset-calibrated grid')
    parser.add_argument('--transfer-thresholds', type=float, nargs='+', default=[.1])
    parser.add_argument('--calibrate-only', action='store_true')
    parser.add_argument('--calibration-clip')
    parser.add_argument('--calibration-latents', type=int, default=64)
    parser.add_argument('--budget-seconds', type=float, default=1200)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.total_latents < 64:
        parser.error('--total-latents must be >=64 for full active-window warmup')
    if len(args.clips) != len(set(args.clips)) or not all(math.isfinite(x) and x >= 0 for x in args.thresholds + args.transfer_thresholds):
        parser.error('Clips must be distinct and thresholds finite/nonnegative')
    if args.calibrate_only:
        if not args.calibration_clip or args.calibration_latents < 17:
            parser.error('Calibration needs --calibration-clip and at least17latents')
    elif not args.profile or not args.clips:
        parser.error('Evaluation needs --profile and --clips')
    args.threshold, args.max_reuse_steps = None, None
    args.git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    args.output_root.mkdir(parents=True, exist_ok=True)
    worker_path = args.output_root / ('calibration.json' if args.calibrate_only else 'worker_' + '_'.join(args.clips) + '.json')
    if args.calibrate_only and worker_path.exists():
        raise FileExistsError(worker_path)
    worker_source = identities()
    checkpoints = dict(checkpoint=str(args.checkpoint), vae_checkpoint=str(args.vae_checkpoint))
    profile = json.loads(args.profile.read_text()) if args.profile else {}
    transfer = json.loads(args.transfer_profile.read_text()) if args.transfer_profile else None
    if transfer is not None and not transfer.get('coefficients'):
        raise ValueError('Transfer profile requires fixed coefficients')
    if not args.calibrate_only:
        if not profile.get('coefficients'):
            raise ValueError('Profile requires fixed calibrated coefficients')
        if profile.get('checkpoints') != checkpoints and not args.allow_profile_transfer:
            raise ValueError('Profile checkpoints differ; use matching calibration or explicit --allow-profile-transfer')
        if profile.get('calibration_episode') in args.clips:
            raise ValueError('Calibration clip must be held out from evaluation')
    worker = dict(schema_version=1, status='running', dataset='re10k', episodes=args.clips,
        thresholds=sorted(set(args.thresholds)), profile=profile, profile_path=str(args.profile),
        source_sha256=worker_source, checkpoints=checkpoints, points=[], checks={}, timing=TIMING,
        transfer_profile=transfer, transfer_profile_path=str(args.transfer_profile),
        total_latents=args.total_latents, threshold_order={clip: ordered_thresholds(args.thresholds, clip_seed(clip)) for clip in args.clips})
    def expired(signum, frame):
        raise TimeoutError(f'Worker exceeded {args.budget_seconds:g}second budget')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.budget_seconds)
    write_report(worker_path, worker)
    try:
        from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
        torch.set_num_threads(4)
        torch.manual_seed(20261019)
        raw = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        weights = raw.get('ema_model', raw.get('model', raw))
        options = model_options(args, weights, raw.get('meta', {}))
        worker['model_options'] = options
        model = build_denoiser_from_mode(DenoiserConfig(**options))
        model.load_state_dict(weights, strict=True)
        del raw, weights
        model = model.cuda().eval()
        vae = load_wan22_vae(options)
        if args.calibrate_only:
            from miniworld.teacache import DEFAULT_TEACACHE_THRESHOLD, fit_calibration
            source = source_manifest(args, args.calibration_clip, args.calibration_latents)
            data = load_clip_cpu(source, include_gt=False)
            inputs = prepare_inputs(data, vae, 'cuda')
            set_mode(model, 'calibration', args)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model.generate_eval_latents_streaming(inputs['latents'], inputs['cond_seq'],
                    noise=inputs['noise'], total_len=args.calibration_latents, **SAMPLER)
            torch.cuda.synchronize()
            info = json_safe(model.last_eval_meta['teacache'])
            if info['reused_forwards'] != 0:
                raise AssertionError('Dense calibration reused residuals')
            records = info['records']
            profile = fit_calibration(records, degree=4)
            screen = screen_thresholds(records, profile['coefficients'])
            predicted = np.maximum(0., np.polyval(profile['coefficients'], [row['proxy_relative_l1'] for row in records]))
            quantiles = {str(q): float(np.quantile(predicted, q)) for q in (0., .1, .25, .5, .75, .9, 1.)}
            profile.update(dataset='re10k', threshold=DEFAULT_TEACACHE_THRESHOLD, max_reuse_steps=0,
                calibration_episode=args.calibration_clip, calibration_latents=args.calibration_latents,
                checkpoints=checkpoints, source_sha256=worker_source, source=source, threshold_screen=screen,
                sampler=dict(total_len=args.calibration_latents, **SAMPLER), predicted_change_quantiles=quantiles)
            write_report(args.output_root / 'teacache_profile.json', profile)
            worker.update(profile=profile, source=source, calibration_summary=info,
                calibration_seconds=time.perf_counter()-begin, included_in_timed_rollouts=False)
            worker['checks']['calibration_no_reuse'] = True
        else:
            from profiling_config.kv_video_metrics import VideoMetricCalculator
            calculator = VideoMetricCalculator(device='cuda:0', batch_size=8)
            for clip in args.clips:
                episode_dir = args.output_root / f'episode_{clip}'
                episode_dir.mkdir(parents=True, exist_ok=True)
                source = source_manifest(args, clip, args.total_latents)
                gt_data = load_clip_cpu(source, include_gt=True)
                gt = torch.from_numpy(gt_data['gt_rgb']).permute(0, 3, 1, 2).float().div_(255)
                np.save(episode_dir / 'gt_rgb.npy', gt_data['gt_rgb'])
                write_report(episode_dir / 'source.json', source)
                set_mode(model, 'baseline', args, profile)
                warm = prepare_inputs(gt_data, vae, 'cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    off_reference = fixed_probe(model, warm['noise'], warm['cond_seq'], warm['noise'], 1)
                    decoder = StreamingVAEDecoder(vae)
                    decoder.begin()
                    try:
                        for begin, end in ((0, 2), (2, 4), (4, 5)):
                            decoder.step(warm['noise'][:, :, begin:end])
                    finally:
                        decoder.end()
                torch.cuda.synchronize()
                del decoder, gt_data
                baseline = None
                order = [None, *ordered_thresholds(args.thresholds, clip_seed(clip))]
                for index, threshold in enumerate(order):
                    point_dir = episode_dir / threshold_key(threshold)
                    point = resume_point(point_dir / 'point.json', args, source, threshold,
                                         profile, worker_source, checkpoints)
                    if point is None:
                        point = measure_point(model, vae, calculator, args, profile, source, threshold,
                            point_dir, gt, baseline, worker_source, checkpoints, index)
                    if threshold is None:
                        baseline = point
                    worker['points'].append(str(point_dir / 'point.json'))
                    write_report(worker_path, worker)
                if transfer is not None:
                    for index, threshold in enumerate(ordered_thresholds(args.transfer_thresholds, clip_seed(clip))):
                        point_dir = episode_dir / ('transfer_' + threshold_key(threshold))
                        point = resume_point(point_dir / 'point.json', args, source, threshold,
                                             transfer, worker_source, checkpoints)
                        if point is None:
                            measure_point(model, vae, calculator, args, transfer, source, threshold,
                                point_dir, gt, baseline, worker_source, checkpoints, len(order)+index,
                                profile_path=args.transfer_profile)
                        worker['points'].append(str(point_dir / 'point.json'))
                        write_report(worker_path, worker)
                set_mode(model, 'baseline', args, profile)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    off_after = fixed_probe(model, warm['noise'], warm['cond_seq'], warm['noise'], 1)
                parity = error_stats(off_reference, off_after)
                worker['checks'][f'{clip}_off_after_on_bitwise_equal'] = parity['bitwise_equal']
                write_report(episode_dir / 'isolation.json', parity)
                if not parity['bitwise_equal']:
                    raise AssertionError(f'Off-after-on parity failed: {clip}')
                del gt, warm, off_reference, off_after
                gc.collect()
        worker['checks']['source_unchanged'] = identities() == worker_source
        if not all(worker['checks'].values()):
            raise AssertionError(f"Worker checks failed: {worker['checks']}")
        worker['status'] = 'complete'
    except BaseException as error:
        worker.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        worker['worker_wall_seconds'] = time.monotonic()-started
        write_report(worker_path, worker)
    print(f"worker status={worker['status']}; {worker['worker_wall_seconds']:.1f}s", flush=True)
    if worker['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
