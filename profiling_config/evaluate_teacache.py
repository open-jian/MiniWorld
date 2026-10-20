"""Paired RGB-quality and actual end-to-end evaluation of opt-in TeaCache.

The optional short dense calibration is separate from both measured rollouts.
Every measured rollout starts from the same source RGB/actions and fixed noise.
The default is one pilot episode; this script never launches the five-episode set.
"""
from __future__ import annotations

import time
PROCESS_START = time.monotonic()
import argparse
import csv
import gc
import hashlib
import json
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
from profiling_config.kv_episode_inputs import (
    LEGACY_ROOT, episode_manifest, input_hashes, load_episode_cpu, prepare_inputs)
from profiling_config.evaluate_kv_episodes import TimedDecoder, metric_tail
from profiling_config.validate_kv_acceleration import (
    ForwardTimer, error_stats, fixed_probe, json_safe, source_identity, write_report)
from miniworld.vae.codec import StreamingVAEDecoder, load_wan22_vae
from miniworld.teacache import DEFAULT_TEACACHE_THRESHOLD


SAMPLER = dict(history_len=1, max_cache_chunks=24, inflight_chunks=8,
               sink_frames=1, collect_stream_timing=False)
BASELINE_PROBE = Path('/home/jian/Documents/Codex/2026-10-19/hello/work/'
                      'headcast_migration/run_001/baseline_latents.pt')


def code_identity():
    identity = source_identity()
    for name in ('profiling_config/evaluate_teacache.py',
                 'profiling_config/evaluate_kv_episodes.py',
                 'profiling_config/kv_episode_inputs.py',
                 'profiling_config/kv_video_metrics.py', 'miniworld/teacache.py',
                 'miniworld/vae/codec.py', 'miniworld/vae/wan22.py'):
        path = REPO / name
        if path.exists():
            identity[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identity


def set_mode(model, mode, args, profile=None):
    """Keep all other acceleration runtimes off in every measured mode."""
    profile = profile or {}
    threshold = (profile.get('threshold', DEFAULT_TEACACHE_THRESHOLD)
                 if args.threshold is None else args.threshold)
    max_reuse_steps = (profile.get('max_reuse_steps', 0)
                       if args.max_reuse_steps is None else args.max_reuse_steps)
    settings = dict(headcast=False, kv_acceleration='off', teacache=mode != 'baseline',
                    teacache_calibration=mode == 'calibration',
                    teacache_threshold=float(threshold),
                    teacache_coefficients=tuple(profile.get('coefficients', ())),
                    teacache_profile='', teacache_trace=False, teacache_trace_output='',
                    teacache_max_reuse_steps=max_reuse_steps)
    for key, value in settings.items():
        if not hasattr(model.cfg, key):
            raise RuntimeError(f'TeaCache integration does not expose {key}')
        setattr(model.cfg, key, value)
    return settings


def screen_thresholds(records, coefficients, target_skip_rate=.4, max_reuse_steps=0):
    """Replay the dense trace on CPU; this does not predict closed-loop quality.

    Trace rows describe adjacent full computations within an unchanged global
    frame window/history. Each new branch/window starts with one forced full
    call. Predictions are clamped nonnegative, matching the runtime.
    """
    valid = [row for row in records if row.get('proxy_relative_l1') is not None]
    if not valid:
        raise ValueError('Dense calibration produced no comparable adjacent calls')
    values = np.maximum(0., np.polyval(coefficients,
                                     [row['proxy_relative_l1'] for row in valid]))
    positive = values[values > 0]
    if not len(positive):
        raise ValueError('Calibrated polynomial predicts no positive changes')
    thresholds = np.unique(np.concatenate((
        np.geomspace(max(float(positive.min()) * .5, 1e-6),
                     max(float(positive.max()) * 5, 1e-5), 100),
        np.asarray([.01, .025, .05, .075, .1, .15, .2, .3, .5]))))
    trials = []
    for threshold in thresholds:
        states, full, reuse, max_streak = {}, 0, 0, 0
        for row, change in zip(valid, values):
            key = json.dumps([row.get('branch'), row.get('context')], sort_keys=True)
            if key not in states:
                states[key] = [0., 0]
                full += 1
            accumulator, streak = states[key]
            accumulator += float(change)
            force = bool(max_reuse_steps and streak >= max_reuse_steps)
            if force or accumulator >= threshold:
                full += 1
                states[key] = [0., 0]
            else:
                reuse += 1
                streak += 1
                max_streak = max(max_streak, streak)
                states[key] = [accumulator, streak]
        trials.append(dict(threshold=float(threshold), full=full, reuse=reuse,
                           skip_fraction=reuse/(full+reuse), max_reuse_streak=max_streak))
    chosen = min(trials, key=lambda row: (abs(row['skip_fraction']-target_skip_rate),
                                         row['threshold']))
    return dict(selected=chosen, target_skip_fraction=target_skip_rate, candidates=trials,
                interpretation='Dense-trace CPU gate replay only; not a quality or speed prediction; excludes commits')


def runtime_audit(mode, meta, phase_timing):
    checks = dict(other_acceleration_runtimes_off=not meta.get('kv_acceleration')
                  and not meta.get('headcast'))
    info = meta.get('teacache')
    if mode == 'baseline':
        checks['baseline_has_no_teacache_runtime'] = not info
        return checks
    info = info or {}
    full = int(info.get('full_forwards', 0))
    reused = int(info.get('reused_forwards', 0))
    commits = int(info.get('commit_forwards', 0))
    observed_commits = sum(row['count'] for row in phase_timing['by_shape']
                           if row['kind'] == 'commit')
    checks.update(runtime_summary_present=bool(info),
                  full_computations_used=full > 0,
                  residual_reuse_used=reused > 0,
                  full_commit_calls_counted=commits == observed_commits and commits > 0,
                  all_forward_calls_counted=full+reused == phase_timing['forward_count'])
    return checks


def export_contact_sheet(output_dir, gt, outputs):
    from PIL import Image, ImageDraw
    names, chosen = ['GT', 'baseline', 'teacache'], [0, 96, 196, 296, 396, 496]
    images = {'GT': gt, **outputs}
    width, height = 320, 240
    sheet = Image.new('RGB', (width*len(chosen), (height+24)*len(names)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(names):
        for column, index in enumerate(chosen):
            frame = (images[name][index].permute(1, 2, 0)*255).round().byte().numpy()
            x, y = column*width, row*(height+24)
            draw.text((x+5, y+5), f'{name} | RGB {index}', fill='black')
            sheet.paste(Image.fromarray(frame), (x, y+24))
    path = output_dir/'comparison.png'
    sheet.save(path)
    return str(path)


def pilot_decision(report):
    runs = {row['mode']: row for row in report['runs']}
    baseline, candidate = (report['metrics'][mode]['gt']['summary']
                           for mode in ('baseline', 'teacache'))
    agreement = report['metrics']['teacache']['baseline']['summary']
    saved_fraction = 1-runs['teacache']['end_to_end_seconds']/runs['baseline']['end_to_end_seconds']
    deltas = {key: candidate[key]-baseline[key]
              for key in ('psnr_db_global', 'ssim_mean', 'lpips_alex_mean')}
    checks = dict(e2e_time_saved_at_least_10_percent=saved_fraction >= .1,
                  gt_psnr_loss_at_most_half_db=deltas['psnr_db_global'] >= -.5,
                  gt_ssim_loss_at_most_0015=deltas['ssim_mean'] >= -.015,
                  gt_lpips_increase_at_most_001=deltas['lpips_alex_mean'] <= .01,
                  baseline_agreement_ssim_at_least_095=agreement['ssim_mean'] >= .95,
                  baseline_agreement_lpips_at_most_003=agreement['lpips_alex_mean'] <= .03)
    return dict(expansion_recommended=all(checks.values()), checks=checks,
                time_saved_fraction=saved_fraction,
                speedup=runs['baseline']['end_to_end_seconds']/runs['teacache']['end_to_end_seconds'],
                candidate_minus_baseline_gt=deltas,
                interpretation='Engineering screening criteria for this pilot; not a statistical quality guarantee. Five-episode evaluation is launched separately.')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, default=2352)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--order', default='baseline,teacache')
    parser.add_argument('--profile', type=Path, help='Reuse MiniWorld calibration; otherwise collect a short dense trace')
    parser.add_argument('--threshold', type=float,
                        help='Override the profile threshold (default 0.4); lower it if quality loss is too large')
    parser.add_argument('--max-reuse-steps', type=int, default=None,
                        help='Override the saved reuse cap; default inherits the profile, then zero (unlimited)')
    parser.add_argument('--calibration-latents', type=int, default=65)
    parser.add_argument('--calibrate-only', action='store_true')
    parser.add_argument('--target-skip-rate', type=float, default=.4)
    parser.add_argument('--budget-seconds', type=float, default=720)
    args = parser.parse_args()
    order = args.order.split(',')
    if sorted(order) != ['baseline', 'teacache']:
        parser.error('--order must be baseline,teacache or teacache,baseline')
    if not 0 < args.target_skip_rate < 1:
        parser.error('--target-skip-rate must be strictly between zero and one')
    if args.threshold is not None and args.threshold < 0:
        parser.error('--threshold must be nonnegative')
    if args.calibration_latents < 17 or (args.max_reuse_steps is not None and args.max_reuse_steps < 0):
        parser.error('calibration needs at least 17 latents; max-reuse-steps must be nonnegative')
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
                  timing=dict(primary='end_to_end_seconds', repetitions=1,
                    includes='Fresh initial RGB/action reads, resize, transfers, initial-frame VAE encode, conditioning/noise construction, complete sampler and streaming VAE RGB decode; GPU output ready',
                    excludes='Model loading, common warmup, dense calibration/profile fitting, full GT decode, hashes, quality metrics, CPU output transfer and exports',
                    instrumentation='CUDA events per DiT forward and VAE step; no per-call synchronization added by timer; gate synchronizations remain inside candidate E2E',
                    comparison='Same GPU, same checkpoint, same source inputs and fixed noise; current baseline measured in this process',
                    warmup='Dense full-cache lengths 16/2/1, both CFG branches; input VAE encode and streaming decode'),
                  metric_protocol=dict(frames='RGB 1..496 inclusive; conditioning frame zero excluded',
                    video_shape=[497, 3, 240, 320], format='Uncompressed decoded FP32 RGB clamped to [0,1]',
                    references=['gt', 'baseline'], psnr='-10 log10(MSE over all evaluated RGB pixels)',
                    ssim='Gaussian 11x11 sigma1.5, valid centers, data_range1',
                    lpips='Official pretrained calibrated AlexNet v0.1', temporal_subsampling=False))
    original_forward = model = None
    try:
        from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
        from miniworld.teacache import fit_calibration
        torch.set_num_threads(4)
        torch.manual_seed(20261016+args.episode)
        report['source'] = episode_manifest(args.episode, 125)
        report['source_sha256'] = code_identity()
        report['gpu'] = torch.cuda.get_device_name()
        report['visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        report['torch_version'] = torch.__version__
        report['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        options = dict(json.loads((LEGACY_ROOT/'experiment_manifest.json').read_text())['args'])
        report['checkpoints'] = {key: options[key] for key in ('checkpoint', 'vae_checkpoint')}
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
        original_forward = model.net.forward_with_cache
        gt_data = load_episode_cpu(args.episode, include_gt=True)
        gt = torch.from_numpy(gt_data['gt_rgb']).permute(0, 3, 1, 2).float().div_(255)
        np.save(args.output_dir/'gt_rgb.npy', gt_data['gt_rgb'])
        report['load_and_reference_seconds'] = time.monotonic()-PROCESS_START
        started = time.perf_counter()
        warm_inputs = prepare_inputs(gt_data, vae, 'cuda')
        baseline_probe = torch.load(BASELINE_PROBE, map_location='cpu', weights_only=True).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            off_reference = fixed_probe(model, baseline_probe, warm_inputs['cond_seq'], warm_inputs['noise'], 1)
            decoder = StreamingVAEDecoder(vae)
            decoder.begin()
            try:
                for start, stop in ((0, 2), (2, 4), (4, 5)):
                    decoder.step(baseline_probe[:, :, start:stop])
            finally:
                decoder.end()
        torch.cuda.synchronize()
        report['warmup_seconds'] = time.perf_counter()-started
        del decoder, warm_inputs, gt_data
        if args.profile:
            profile = json.loads(args.profile.read_text())
            if not profile.get('coefficients'):
                raise ValueError('Profile has no fitted coefficients')
            report['calibration'] = dict(profile_source=str(args.profile), reused=True,
                                         included_in_timed_rollouts=False)
        else:
            print(f'[{args.episode:06d}] dense calibration ({args.calibration_latents} latents; outside E2E)', flush=True)
            set_mode(model, 'calibration', args)
            calibration_data = load_episode_cpu(args.episode, args.calibration_latents, include_gt=False)
            calibration_inputs = prepare_inputs(calibration_data, vae, 'cuda')
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                calibration_latents = model.generate_eval_latents_streaming(
                    calibration_inputs['latents'], calibration_inputs['cond_seq'],
                    noise=calibration_inputs['noise'], total_len=args.calibration_latents, **SAMPLER)
            torch.cuda.synchronize()
            info = json_safe(model.last_eval_meta.get('teacache', {}))
            report['calibration'] = dict(elapsed_seconds=time.perf_counter()-started,
                total_latents=args.calibration_latents, source=calibration_data['manifest'],
                included_in_timed_rollouts=False, summary=info)
            if int(info.get('reused_forwards', 0)) != 0:
                raise AssertionError('Calibration unexpectedly reused residuals')
            records = info.get('records', [])
            profile = fit_calibration(records, degree=4)
            screen = screen_thresholds(records, profile['coefficients'],
                                       args.target_skip_rate, args.max_reuse_steps or 0)
            # Keep the replay recommendation as diagnostic metadata; newly
            # calibrated profiles use the same manually adjustable 0.4 default.
            profile.update(threshold=DEFAULT_TEACACHE_THRESHOLD,
                           calibration_episode=report['episode'],
                           calibration_latents=args.calibration_latents,
                           checkpoints=report['checkpoints'], source_sha256=report['source_sha256'],
                           threshold_screen=screen, max_reuse_steps=args.max_reuse_steps or 0)
            del calibration_data, calibration_inputs, calibration_latents
        if args.threshold is not None:
            profile['threshold'] = args.threshold
        profile['max_reuse_steps'] = (profile.get('max_reuse_steps', 0)
                                     if args.max_reuse_steps is None else args.max_reuse_steps)
        profile_path = args.output_dir/'teacache_profile.json'
        write_report(profile_path, profile)
        report['teacache_profile'] = dict(path=str(profile_path), **profile)
        write_report(path, report)
        if args.calibrate_only:
            report['status'] = 'complete'
            report['checks']['calibration_profile_written'] = profile_path.exists()
            return
        sampler = dict(total_len=125, **SAMPLER)
        report['sampler'] = sampler
        outputs = {}
        for index, mode in enumerate(order):
            print(f'[{args.episode:06d}] starting {mode} ({index+1}/2)', flush=True)
            settings = set_mode(model, mode, args, profile)
            timer, decoder = ForwardTimer(original_forward), TimedDecoder(vae)
            model.net.forward_with_cache = timer
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
            row['runtime_checks'] = runtime_audit(mode, row['sampler_meta'], row['phase_timing'])
            row['runtime_active'] = all(row['runtime_checks'].values())
            if not row['finite'] or not row['runtime_active']:
                report['runs'].append(row)
                write_report(path, report)
                raise AssertionError(f'Invalid rollout: {mode} {row["runtime_checks"]}')
            if row['rgb_shape'] != [1, 3, 497, 240, 320]:
                raise AssertionError(f'Unexpected video shape {row["rgb_shape"]}')
            rgb = pred_rgb[0].permute(1, 0, 2, 3).float().add(1).mul(.5).clamp(0, 1).cpu()
            outputs[mode] = rgb
            latent_path, rgb_path = (args.output_dir/f'{mode}_latents.pt',
                                     args.output_dir/f'{mode}_rgb_fp32.npy')
            torch.save(pred_latents.cpu(), latent_path)
            np.save(rgb_path, rgb.numpy())
            row['outputs'] = dict(latents=str(latent_path), rgb=str(rgb_path))
            if mode == 'baseline':
                previous_path = (Path('/home/jian/Documents/Codex/2026-10-19/hello/work/kv_multiepisode')
                                 /f'episode_{args.episode:06d}'/'baseline_latents.pt')
                if previous_path.exists():
                    previous = torch.load(previous_path, map_location='cpu', weights_only=True)
                    report['previous_baseline_sanity'] = dict(path=str(previous_path),
                        **error_stats(previous, pred_latents.cpu()))
                    del previous
            report['runs'].append(row)
            write_report(path, report)
            print(f'[{args.episode:06d}] {mode}: E2E {row["end_to_end_seconds"]:.3f}s', flush=True)
            del pred_latents, pred_rgb, rgb, cpu_data, timer, decoder
        report['checks']['all_mode_input_hashes_equal'] = all(
            row['input_hashes'] == report['runs'][0]['input_hashes'] for row in report['runs'])
        set_mode(model, 'baseline', args, profile)
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
        for mode in ('baseline', 'teacache'):
            report['metrics'][mode] = {}
            for reference in ('baseline', 'gt'):
                if mode == reference:
                    continue
                print(f'[{args.episode:06d}] metrics {mode} vs {reference}', flush=True)
                metric = calculator(outputs['baseline'] if reference == 'baseline' else gt, outputs[mode])
                metric.update(reference=reference, tail_128=metric_tail(metric))
                report['metrics'][mode][reference] = metric
                write_report(path, report)
        csv_path = args.output_dir/'per_frame_metrics.csv'
        fields = ['episode', 'mode', 'reference', 'rgb_frame', 'mse_rgb', 'psnr_db',
                  'psnr_is_positive_infinity', 'ssim', 'lpips_alex']
        with csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for mode, references in report['metrics'].items():
                for reference, metric in references.items():
                    for index in range(metric['summary']['frame_count']):
                        writer.writerow(dict(episode=report['episode'], mode=mode, reference=reference,
                            **{name: values[index] for name, values in metric['per_frame'].items()}))
        report['artifacts'] = dict(per_frame_csv=str(csv_path),
                                  contact_sheet=export_contact_sheet(args.output_dir, gt, outputs))
        report['checks']['source_unchanged'] = code_identity() == report['source_sha256']
        report['checks']['all_metric_frames_complete'] = all(
            metric['summary']['frame_count'] == 496
            for refs in report['metrics'].values() for metric in refs.values())
        if not all(report['checks'].values()):
            raise AssertionError(f'Evaluation checks failed: {report["checks"]}')
        report['pilot_decision'] = pilot_decision(report)
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
