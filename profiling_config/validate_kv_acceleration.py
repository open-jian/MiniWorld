"""Bounded paired rollout validation for MiniWorld KV acceleration.

The timed region is the complete latent sampler (both CFG branches, calibration,
cache commits/eviction and kernel launches). Model loading, warmup, diagnostic
replays, VAE decoding and file I/O are excluded and reported separately. A short
causally decoded shared tail is exported for visual inspection. By default one
pair is measured; use --pairs 2 to reverse mode order on the second pair.
"""
from __future__ import annotations

import time
PROCESS_START = time.monotonic()

import argparse
from collections import defaultdict
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import traceback

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
WORK = Path('/home/jian/Documents/Codex/2026-10-19/hello/work')
INPUT = Path('/data2/jian/output/Miniworld/DROID_long_horizon/chunk_flash_attention/episode_002352')
MODE_CONFIG = {
    'baseline': {'headcast': False, 'kv_acceleration': 'off'},
    'headcast': {'headcast': True, 'kv_acceleration': 'off'},
    'full': {'headcast': False, 'kv_acceleration': 'combined'},
    'combined': {'headcast': False, 'kv_acceleration': 'combined'},
    'dense': {'headcast': False, 'kv_acceleration': 'dense'},
    'dummy': {'headcast': False, 'kv_acceleration': 'dummy'},
    'new-headcast': {'headcast': False, 'kv_acceleration': 'headcast'},
    'headcast-source': {'headcast': False, 'kv_acceleration': 'headcast-source'},
}


class BudgetExpired(Exception):
    pass


def json_safe(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_report(path, report):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def error_stats(reference, actual):
    bitwise_equal = bool(torch.equal(reference, actual))
    reference, actual = reference.float(), actual.float()
    delta = actual - reference
    rms = reference.square().mean().sqrt().item()
    rmse = delta.square().mean().sqrt().item()
    finite = bool(torch.isfinite(actual).all())
    return dict(finite=finite, max_abs=delta.abs().max().item() if finite else None,
                mean_abs=delta.abs().mean().item() if finite else None,
                rmse=rmse if finite else None, reference_rms=rms,
                relative_rmse=rmse / max(rms, 1e-12) if finite else None,
                bitwise_equal=bitwise_equal)


def summarize(values):
    return dict(count=len(values), total_ms=sum(values), median_ms=statistics.median(values),
                mean_ms=statistics.mean(values), min_ms=min(values), max_ms=max(values))


def set_mode(model, mode, overrides):
    settings = {**overrides, **MODE_CONFIG[mode]}
    missing = [key for key in settings if not hasattr(model.cfg, key)]
    if missing:
        raise RuntimeError(f'Unsupported configuration options: {missing}; acceleration integration is absent or incompatible')
    for key, value in settings.items():
        setattr(model.cfg, key, value)
    return settings


class ForwardTimer:
    """Record GPU events without synchronizing between forwards.

    Full-cache interval includes all GPU work between its boundary events,
    including sampler-side cache maintenance; it is not a sum of forward times.
    The per-forward summaries exclude work outside forward_with_cache.
    """
    def __init__(self, original, cache_frames=48, current_frames=16):
        self.original = original
        self.cache_frames = cache_frames
        self.current_frames = current_frames
        self.rows = []
        self.steady_start = None
        self.steady_end = None
        self.steady_last_end = None

    def __call__(self, *args, **kwargs):
        current = int(args[0].shape[2])
        past_frames = int(kwargs.get('current_position_offset', 0))
        commit = bool(kwargs.get('return_kv', False))
        branch = 'uncond' if kwargs.get('cond_drop') is not None else 'cond'
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        full = not commit and past_frames == self.cache_frames and current == self.current_frames
        if full and self.steady_start is None:
            self.steady_start = start
        elif self.steady_start is not None and self.steady_end is None and not commit and not full:
            self.steady_end = start
        result = self.original(*args, **kwargs)
        end.record()
        if full:
            self.steady_last_end = end
        self.rows.append((dict(branch=branch, kind='commit' if commit else 'denoise',
                               cache_frames=past_frames, current_frames=current,
                               kv_context=kwargs.get('kv_context') is not None), start, end))
        return result

    def summary(self):
        groups = defaultdict(list)
        totals = defaultdict(float)
        for row, start, end in self.rows:
            elapsed = start.elapsed_time(end)
            key = (row['branch'], row['kind'], row['cache_frames'], row['current_frames'])
            groups[key].append(elapsed)
            totals[row['kind']] += elapsed
        grouped = [dict(branch=key[0], kind=key[1], cache_frames=key[2], current_frames=key[3],
                        **summarize(values)) for key, values in groups.items()]
        full = [row for row in grouped if row['kind'] == 'denoise' and
                row['cache_frames'] == self.cache_frames and row['current_frames'] == self.current_frames]
        end = self.steady_end if self.steady_end is not None else self.steady_last_end
        span = self.steady_start.elapsed_time(end) if self.steady_start is not None and end is not None else None
        return dict(forward_count=len(self.rows),
                    kv_context_forward_count=sum(row['kv_context'] for row, _, _ in self.rows),
                    kv_context_commit_forward_count=sum(row['kv_context'] and row['kind'] == 'commit'
                                                        for row, _, _ in self.rows),
                    forward_gpu_ms_by_kind=dict(totals),
                    saturated_full_window_interval_gpu_ms=span,
                    saturated_full_window_forward_summaries=full, by_shape=grouped,
                    interval_definition='First 48-cache/16-current denoise entry through first shrinking-window denoise entry; includes intervening commits, classification and maintenance; excludes initial prefill and final flush',
                    instrumentation='Two CUDA events per network call; no per-call synchronization; sampler wall time is authoritative')


def source_identity():
    files = [REPO / 'miniworld' / name for name in ('miniworld.py', 'denoiser.py', 'headcast.py', 'headcast_attention.py')]
    files += list((REPO / 'miniworld').glob('*kv*.py'))
    files += [Path(__file__)]
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(files)) if path.exists()}


@torch.inference_mode()
def fixed_probe(model, baseline, condition, noise, warmups=0):
    """Recreate the exact same baseline probe, retaining no rollout cache."""
    net = model.net
    t = torch.linspace(.275887, 1., 8, device=baseline.device).repeat_interleave(2).unsqueeze(0)
    current = baseline[:, :, 48:64] * (1 - t[:, None, :, None, None]) + noise[:, :, 48:64] * t[:, None, :, None, None]
    reference = None
    for branch in ('cond', 'uncond'):
        ctx_condition = condition[:, :48]
        if branch == 'uncond':
            ctx_condition, drop = model._make_uncond(ctx_condition)
        else:
            drop = None
        _, past = net.forward_with_cache(baseline[:, :, :48], torch.zeros(1, 48, device=baseline.device),
                                         ctx_condition, return_kv=True, chunk_size=2, cond_drop=drop)
        for count in (16, 2, 1):
            cond = condition[:, 48:48 + count]
            if branch == 'uncond':
                cond, drop = model._make_uncond(cond)
            for _ in range(max(1, warmups)):
                output = net.forward_with_cache(current[:, :, :count], t[:, :count], cond,
                                                past_kv_list=past, current_position_offset=48,
                                                return_kv=False, chunk_size=2, cond_drop=drop)[0]
            if branch == 'cond' and count == 16:
                reference = output.detach().cpu()
        del past
    torch.cuda.synchronize()
    return reference


def partition_audit(summary, depth, heads):
    groups = summary.get('groups', [])
    if not groups or not isinstance(groups, list):
        return {'available': False, 'complete_disjoint_head_partition': False}
    names = ('global', 'temporal', 'spatial') if 'temporal' in groups[0] else ('global', 'sink', 'dummy', 'local', 'spatial')
    counts = {name: 0 for name in names}
    valid = len(groups) == depth
    for layer in groups:
        if not isinstance(layer, dict) or set(layer) != set(names):
            return {'available': True, 'complete_disjoint_head_partition': False}
        flattened = []
        for name in names:
            indices = layer[name]
            if not isinstance(indices, list) or any(type(index) is not int for index in indices):
                return {'available': True, 'complete_disjoint_head_partition': False}
            counts[name] += len(indices)
            flattened.extend(indices)
        valid = valid and sorted(flattened) == list(range(heads))
    return dict(available=True, complete_disjoint_head_partition=valid, counts=counts,
                reported_group_counts_match=summary.get('group_counts') == counts)


def runtime_audit(mode, meta, depth, heads, phase_timing=None):
    if mode == 'baseline':
        return {'baseline_has_no_acceleration_runtime': not meta.get('kv_acceleration') and not meta.get('headcast')}
    key = 'headcast' if mode == 'headcast' else 'kv_acceleration'
    info = meta.get(key)
    if not isinstance(info, dict):
        return {'runtime_summary_present': False}
    partition = partition_audit(info, depth, heads)
    checks = dict(runtime_summary_present=True, status_active=info.get('status') == 'active',
                  complete_head_partition=partition.get('complete_disjoint_head_partition', False),
                  reported_group_counts_match=partition.get('reported_group_counts_match', False),
                  routed_attention_used=int(info.get('routed_attention_calls', 0)) > 0)
    if mode == 'headcast':
        return checks
    branches = 2 if meta.get('cfg_enabled', True) else 1
    routed = int(info.get('routed_attention_calls', 0))
    compact = int(info.get('compact_attention_calls', 0))
    dense = int(info.get('dense_attention_calls', 0))
    commits = int(info.get('commit_attention_calls', 0))
    checks.update(enabled=info.get('enabled') is True,
                  requested_mode_matches=info.get('mode') == MODE_CONFIG[mode]['kv_acceleration'],
                  compact_attention_used=compact > 0,
                  all_layer_branch_caches_converted=int(info.get('converted_caches', 0)) >= depth * branches,
                  routed_call_accounting=routed == dense + compact,
                  cache_storage_recorded=int(info.get('final_cache_storage_bytes', 0)) > 0,
                  full_logical_cache_reached=int(info.get('final_logical_history_frames', 0)) == 48)
    if phase_timing is not None:
        expected_calls = int(phase_timing['kv_context_forward_count']) * depth
        expected_commits = int(phase_timing['kv_context_commit_forward_count']) * depth
        checks['all_context_calls_counted'] = routed == expected_calls
        checks['commit_attention_used_and_counted'] = commits == expected_commits and (commits > 0 if expected_commits else True)
    else:
        checks['commit_attention_used'] = commits > 0
    if mode == 'dense':
        checks['dense_mode_all_heads_global'] = partition.get('counts', {}).get('global') == depth * heads
        checks['dense_mode_has_no_policy'] = info.get('policy') is None
    else:
        policy = info.get('policy')
        checks['all_branches_calibrated'] = int(info.get('calibration_forwards', 0)) == branches
        checks['policy_active_and_matches_runtime'] = (isinstance(policy, dict)
            and policy.get('status') == 'active' and policy.get('groups') == info.get('groups')
            and policy.get('group_counts') == info.get('group_counts')
            and policy.get('mode') == info.get('mode'))
    return checks


def check_runtime(mode, meta, depth=28, heads=12, phase_timing=None):
    return all(runtime_audit(mode, meta, depth, heads, phase_timing).values())


@torch.inference_mode()
def decode_quality(args, manifest, outputs, report):
    from miniworld.vae.codec import load_wan22_vae
    import numpy as np
    from PIL import Image, ImageDraw
    start = time.monotonic()
    vae = load_wan22_vae(manifest['args'])
    names = list(outputs)
    cutoff_latent = max(0, args.total_latents - args.decode_tail_latents)
    decoded = {}
    for name in names:
        vae.decode_stream_begin()
        parts = []
        try:
            for offset in range(0, args.total_latents, 2):
                piece = vae.decode_stream_step(outputs[name][:, :, offset:offset + 2].cuda())
                # Each latent after the first contributes four RGB frames. This
                # trims exact global coordinates even when cutoff is odd.
                rgb_start = 0 if offset == 0 else 1 + 4 * (offset - 1)
                rgb_cutoff = 0 if cutoff_latent == 0 else 1 + 4 * (cutoff_latent - 1)
                keep_from = max(0, rgb_cutoff - rgb_start)
                if keep_from < piece.shape[2]:
                    parts.append(piece[:, :, keep_from:].float().cpu())
                del piece
        finally:
            vae.decode_stream_end()
        decoded[name] = torch.cat(parts, dim=2)
    baseline = decoded['baseline']
    metrics = {}
    for name in names:
        metrics[name] = error_stats(baseline, decoded[name])
        mse = ((baseline - decoded[name]).float() * .5).square().mean().item()
        metrics[name]['psnr_db_range_0_1'] = -10 * math.log10(mse) if mse > 0 else None
    # Contact sheet: identical global times, separate labeled rows for every mode.
    rgb = {name: ((value[0].permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()
           for name, value in decoded.items()}
    frames = baseline.shape[2]
    chosen = np.linspace(0, frames - 1, min(6, frames), dtype=int).tolist()
    height, width = rgb['baseline'].shape[1:3]
    sheet = Image.new('RGB', (len(chosen) * width, len(names) * (height + 28)), 'white')
    draw = ImageDraw.Draw(sheet)
    rgb_cutoff = 0 if cutoff_latent == 0 else 1 + 4 * (cutoff_latent - 1)
    for row, name in enumerate(names):
        y = row * (height + 28)
        for column, index in enumerate(chosen):
            draw.text((column * width + 5, y + 7), f'{name} | RGB frame {rgb_cutoff + index}', fill='black')
            sheet.paste(Image.fromarray(rgb[name][index]), (column * width, y + 28))
    sheet_path = args.output_dir / 'decoded_tail_comparison.png'
    sheet.save(sheet_path)
    video_path = args.output_dir / 'decoded_tail_comparison.mp4'
    video_error = None
    try:
        video = np.concatenate([rgb[name] for name in names], axis=2)
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{video.shape[2]}x{video.shape[1]}', '-r', str(args.video_fps),
            '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            str(video_path),
        ], input=video.tobytes(), check=True, capture_output=True, timeout=30)
    except Exception as error:
        video_error = f'{type(error).__name__}: {error}'
    torch.cuda.synchronize()
    report['decoded_quality'] = dict(status='complete', scope='Same-coordinate generated tail, after causally decoding the complete prefix separately for each mode',
                                     latent_start=cutoff_latent, latent_end=args.total_latents,
                                     rgb_start=rgb_cutoff, rgb_frames=int(frames), modes=names,
                                     metrics=metrics, contact_sheet=str(sheet_path),
                                     video=str(video_path) if video_error is None else None,
                                     video_export_error=video_error,
                                     wall_seconds_including_vae_load=time.monotonic() - start,
                                     limits='Pixel errors and visual comparison on one episode/seed; no perceptual metric or broad quality claim')
    del vae
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--modes', default='baseline,full')
    parser.add_argument('--pairs', type=int, default=1)
    parser.add_argument('--total-latents', type=int, default=72)
    parser.add_argument('--budget-seconds', type=float, default=360)
    parser.add_argument('--warmup-forwards', type=int, default=2)
    parser.add_argument('--baseline-latents', type=Path,
                        default=WORK / 'headcast_migration/run_001/baseline_latents.pt')
    parser.add_argument('--config-json', type=Path)
    parser.add_argument('--decode-tail-latents', type=int, default=8,
                        help='0 disables VAE quality export; default decodes the full prefix but exports only the final 8 latents')
    parser.add_argument('--video-fps', type=int, default=8)
    parser.add_argument('--seed', type=int, help='Regenerate fixed noise if different from manifest seed')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    args.modes = args.modes.split(',')
    if len(set(args.modes)) != len(args.modes) or any(mode not in MODE_CONFIG for mode in args.modes):
        parser.error(f'--modes must be unique names from {list(MODE_CONFIG)}')
    if 'baseline' not in args.modes:
        parser.error('--modes must include baseline')
    if not 66 <= args.total_latents <= 125:
        parser.error('--total-latents must be 66..125 to reach saturation')
    if args.pairs not in (1, 2, 3):
        parser.error('--pairs must be 1, 2 or 3; increase budget deliberately for additional pairs')
    if not 0 < args.budget_seconds <= 600:
        parser.error('--budget-seconds must be positive and at most 600')
    if args.decode_tail_latents < 0 or args.decode_tail_latents > args.total_latents:
        parser.error('--decode-tail-latents must be 0..total-latents')
    if args.warmup_forwards < 1:
        parser.error('--warmup-forwards must be at least 1')
    return args


def self_test():
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    assert error_stats(x, x.clone())['bitwise_equal']
    assert error_stats(x, x + 1)['rmse'] == 1
    assert not error_stats(x, x + 1)['bitwise_equal']
    assert check_runtime('baseline', {})
    assert not check_runtime('full', {})
    assert not check_runtime('full', {'kv_acceleration': {'status': 'active', 'mode': 'combined'}})
    groups = [{'global': [0], 'sink': [], 'dummy': [1], 'local': [], 'spatial': []}]
    summary = {'groups': groups, 'group_counts': {'global': 1, 'sink': 0, 'dummy': 1, 'local': 0, 'spatial': 0}}
    assert partition_audit(summary, 1, 2)['complete_disjoint_head_partition']
    summary['groups'][0]['dummy'] = [0]
    assert not partition_audit(summary, 1, 2)['complete_disjoint_head_partition']
    assert not check_runtime('headcast', {'headcast': {'status': 'active', 'routed_attention_calls': 0}})
    assert summarize([1., 2., 3.])['median_ms'] == 2
    print('CPU metric/config/report self-tests passed', flush=True)


@torch.inference_mode()
def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / 'validation.json'
    if report_path.exists():
        raise FileExistsError(f'Choose a fresh output directory: {report_path} exists')
    deadline = PROCESS_START + args.budget_seconds
    def expired(signum, frame):
        raise BudgetExpired(f'{args.budget_seconds:g}-second process budget exhausted')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline - time.monotonic()))
    report = dict(status='running', arguments=vars(args), runs=[], checks={},
                  timing_includes='Complete latent sampler, both CFG branches, calibration, commits, cache maintenance, sampler integration and CUDA-event recording',
                  timing_excludes='Input/model loading, common dense operation warmup, parity probes, VAE decoding, metrics and file output',
                  limits='One episode and fixed noise; short generated-history trajectory; a single pair is a smoke benchmark, not a confidence interval',
                  warmup_scope='Dense network/cache primitives at current lengths 16/2/1 and both CFG branches. Candidate first-use compilation/calibration remains included in its first measured rollout.',
                  phase_timing_note='Full sampler wall time is primary. Per-forward events exclude sampler-side work; saturated interval includes intervening sampler-side GPU work.')
    original_forward = None
    model = None
    outputs = {}
    try:
        from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
        manifest = json.loads((args.input_dir / 'experiment_manifest.json').read_text())
        seed = int(manifest['seed'] if args.seed is None else args.seed)
        torch.set_num_threads(4)
        torch.manual_seed(seed)
        raw = torch.load(manifest['args']['checkpoint'], map_location='cpu', weights_only=True)
        weights = raw.get('ema_model', raw.get('model', raw))
        options = dict(manifest['args'])
        options.update(latent_size=(15, 20), latent_frames=int(weights['net.feat_rope.freqs_cos'].shape[0] // 300),
                       wm_use_checkpoint=False, headcast=False, kv_acceleration='off')
        overrides = json.loads(args.config_json.read_text()) if args.config_json else {}
        if not isinstance(overrides, dict):
            raise TypeError('--config-json must contain an object of DenoiserConfig overrides')
        options.update(overrides)
        options.update(headcast=False, kv_acceleration='off')
        config = DenoiserConfig(**options)
        missing = [key for key in overrides if not hasattr(config, key)]
        if missing:
            raise RuntimeError(f'Unknown --config-json options would be silently ignored: {missing}')
        model = build_denoiser_from_mode(config)
        model.load_state_dict(weights, strict=True)
        del raw, weights
        model = model.cuda().eval()
        original_forward = model.net.forward_with_cache
        inputs_cpu = torch.load(args.input_dir / 'prepared_inputs.pt', map_location='cpu', weights_only=True)
        if seed != int(manifest['seed']):
            generator = torch.Generator(device='cpu').manual_seed(seed)
            inputs_cpu['noise'] = torch.randn(inputs_cpu['noise'].shape, generator=generator, dtype=inputs_cpu['noise'].dtype)
        noise_hash = hashlib.sha256(inputs_cpu['noise'].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        inputs = {name: value.cuda() for name, value in inputs_cpu.items()}
        baseline_probe = torch.load(args.baseline_latents, map_location='cpu', weights_only=True).cuda()
        torch.cuda.synchronize()
        report.update(episode=manifest['episode'], seed=seed, checkpoint=manifest['args']['checkpoint'],
                      noise_sha256=noise_hash, gpu=torch.cuda.get_device_name(),
                      visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_version=torch.__version__,
                      git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                      source_sha256=source_identity(), load_seconds=time.monotonic() - PROCESS_START,
                      model_shape=dict(layers=model.net.depth, heads=model.net.num_heads), cfg_scale=model.cfg_scale,
                      sampler=dict(total_len=args.total_latents, history_len=1, max_cache_chunks=24,
                                   inflight_chunks=8, sink_frames=1, stream_decoder=None, collect_stream_timing=False),
                      config_overrides=overrides)
        set_mode(model, 'baseline', overrides)
        started = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            off_reference = fixed_probe(model, baseline_probe, inputs['cond_seq'], inputs['noise'], args.warmup_forwards)
        report['warmup_seconds'] = time.monotonic() - started
        del baseline_probe
        torch.cuda.empty_cache()
        write_report(report_path, report)
        for pair in range(args.pairs):
            order = args.modes if pair % 2 == 0 else list(reversed(args.modes))
            for mode in order:
                settings = set_mode(model, mode, overrides)
                timer = ForwardTimer(original_forward)
                model.net.forward_with_cache = timer
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                try:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = model.generate_eval_latents_streaming(inputs['latents'], inputs['cond_seq'],
                                                                      noise=inputs['noise'], **report['sampler'])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                finally:
                    model.net.forward_with_cache = original_forward
                memory = dict(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                output = result.detach().cpu()
                del result
                meta = json_safe(model.last_eval_meta)
                row = dict(pair=pair, order=order.index(mode), mode=mode, configuration=settings,
                           latent_generation_seconds=elapsed, latent_shape=list(output.shape),
                           finite=bool(torch.isfinite(output).all()), sampler_meta=meta,
                           phase_timing=timer.summary(), **memory)
                row['runtime_checks'] = runtime_audit(mode, meta, model.net.depth, model.net.num_heads, row['phase_timing'])
                row['runtime_active'] = all(row['runtime_checks'].values())
                runtime = meta.get('headcast' if mode == 'headcast' else 'kv_acceleration', {})
                row['head_partition_audit'] = partition_audit(runtime, model.net.depth, model.net.num_heads)
                if mode in outputs:
                    row['repeat_latent_parity'] = error_stats(outputs[mode], output)
                else:
                    outputs[mode] = output
                torch.save(output, args.output_dir / f'{mode}_pair_{pair}_latents.pt')
                report['runs'].append(row)
                print(f'pair {pair} {mode}: {elapsed:.3f} s; runtime_active={row["runtime_active"]}', flush=True)
                write_report(report_path, report)
                if not row['finite']:
                    raise AssertionError(f'Non-finite output in {mode}')
                if not row['runtime_active']:
                    raise AssertionError(f'{mode} runtime summary absent or inactive; refusing a vacuous timing comparison')
                del timer, output
        set_mode(model, 'baseline', overrides)
        baseline_probe = torch.load(args.baseline_latents, map_location='cpu', weights_only=True).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            off_after = fixed_probe(model, baseline_probe, inputs['cond_seq'], inputs['noise'], 1)
        report['disabled_after_enabled_same_input_parity'] = error_stats(off_reference, off_after)
        report['checks']['disabled_parity_bitwise'] = report['disabled_after_enabled_same_input_parity']['bitwise_equal']
        report['checks']['all_finite'] = all(row['finite'] for row in report['runs'])
        report['checks']['requested_runtimes_active'] = all(row['runtime_active'] for row in report['runs'])
        report['source_sha256_after_rollouts'] = source_identity()
        report['checks']['source_unchanged_during_rollouts'] = report['source_sha256'] == report['source_sha256_after_rollouts']
        report['latent_comparison'] = {}
        frame_rows = []
        for mode, output in outputs.items():
            if mode == 'baseline':
                continue
            report['latent_comparison'][mode] = dict(full=error_stats(outputs['baseline'], output),
                first_8=error_stats(outputs['baseline'][:, :, :8], output[:, :, :8]),
                first_48=error_stats(outputs['baseline'][:, :, :48], output[:, :, :48]),
                last_24=error_stats(outputs['baseline'][:, :, -24:], output[:, :, -24:]),
                seed_frame_equal=bool(torch.equal(outputs['baseline'][:, :, :1], output[:, :, :1])))
            report['checks'][f'{mode}_seed_frame_preserved'] = report['latent_comparison'][mode]['seed_frame_equal']
            for frame in range(args.total_latents):
                frame_rows.append(dict(mode=mode, latent_frame=frame,
                    **error_stats(outputs['baseline'][:, :, frame], output[:, :, frame])))
        if frame_rows:
            with (args.output_dir / 'latent_error_by_frame.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
                writer.writeheader()
                writer.writerows(frame_rows)
        ratios = defaultdict(list)
        comparisons = []
        for pair in range(args.pairs):
            rows = {row['mode']: row for row in report['runs'] if row['pair'] == pair}
            base = rows['baseline']
            for mode, row in rows.items():
                if mode == 'baseline':
                    continue
                ratio = base['latent_generation_seconds'] / row['latent_generation_seconds']
                ratios[mode].append(ratio)
                baseline_interval = base['phase_timing']['saturated_full_window_interval_gpu_ms']
                candidate_interval = row['phase_timing']['saturated_full_window_interval_gpu_ms']
                comparisons.append(dict(pair=pair, mode=mode, speedup_baseline_over_candidate=ratio,
                    wall_time_reduction_fraction=1 - 1 / ratio,
                    saturated_interval_speedup=(baseline_interval / candidate_interval
                        if baseline_interval and candidate_interval else None)))
        report['paired_timing_comparisons'] = comparisons
        report['median_pair_speedup'] = {mode: statistics.median(values) for mode, values in ratios.items()}
        report['latent_status'] = 'complete' if all(report['checks'].values()) else 'failed_checks'
        write_report(report_path, report)
        model.net.forward_with_cache = original_forward
        model = None
        del original_forward, baseline_probe, inputs, config
        original_forward = None
        gc.collect()
        torch.cuda.empty_cache()
        if args.decode_tail_latents:
            decode_quality(args, manifest, outputs, report)
        else:
            report['decoded_quality'] = dict(status='not_requested')
        report['status'] = report['latent_status']
    except BudgetExpired as error:
        report['status'] = 'budget_exhausted'
        report['error'] = str(error)
    except Exception:
        report['status'] = 'error'
        report['error'] = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if model is not None:
            model.cfg.headcast = False
            if hasattr(model.cfg, 'kv_acceleration'):
                model.cfg.kv_acceleration = 'off'
            if original_forward is not None:
                model.net.forward_with_cache = original_forward
        report['elapsed_seconds_including_load_diagnostics_decode'] = time.monotonic() - PROCESS_START
        write_report(report_path, report)
        print(f'Validation {report["status"]}: {report_path}', flush=True)
    if report['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
