"""Replay fixed TeaCache evaluations to collect reuse decisions, without rebenchmarking.

The existing paired evaluation remains the speed/quality evidence. This replay
omits RGB decoding and metrics, checks every input and generated latent against
that evidence, and counts actual executions of each transformer block.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from profiling_config.evaluate_teacache import SAMPLER, code_identity, set_mode
from profiling_config.kv_episode_inputs import LEGACY_ROOT, input_hashes, load_episode_cpu, prepare_inputs
from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
from miniworld.vae.codec import load_wan22_vae


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', action='append', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=float, default=480)
    args = parser.parse_args()
    def expired(signum, frame):
        raise TimeoutError('Reuse trace worker exceeded its execution budget')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.budget_seconds)
    torch.set_num_threads(4)
    options = dict(json.loads((LEGACY_ROOT/'experiment_manifest.json').read_text())['args'])
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
    for evaluation_path in args.evaluation:
        evaluation = json.loads(evaluation_path.read_text())
        episode = evaluation['episode']
        archived = next(r for r in evaluation['runs'] if r['mode'] == 'teacache')
        expected = archived['sampler_meta']['teacache']
        out = args.output_root/f'episode_{episode}'
        out.mkdir(parents=True, exist_ok=True)
        if (out/'replay.json').exists():
            raise FileExistsError(out/'replay.json')
        identity = code_identity()
        identity[str(Path(__file__).relative_to(REPO))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report = dict(schema_version=1, status='running', episode=episode,
                      evaluation_source=str(evaluation_path.resolve()),
                      gpu=torch.cuda.get_device_name(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      source_sha256=identity, checks={},
                      purpose='Diagnostic replay only; existing paired E2E/quality results are unchanged',
                      omitted='No original rollout, RGB decoding, quality metrics, or calibration')
        hooks = []
        try:
            print(f'[{episode}] preparing fixed inputs and tracing TeaCache', flush=True)
            profile = evaluation['teacache_profile']
            set_mode(model, 'teacache', argparse.Namespace(threshold=None, max_reuse_steps=None), profile)
            model.cfg.teacache_trace = True
            model.cfg.teacache_trace_output = ''
            inputs = prepare_inputs(load_episode_cpu(int(episode), include_gt=False), vae, 'cuda')
            report['input_hashes'] = input_hashes(inputs)
            report['checks']['inputs_identical_to_measured_run'] = report['input_hashes'] == archived['input_hashes']
            if not report['checks']['inputs_identical_to_measured_run']:
                raise AssertionError('Replay inputs do not match the measured TeaCache run')
            counts = [0]*len(model.net.blocks)
            def make_hook(index):
                def count_forward(module, arguments):
                    counts[index] += 1
                return count_forward
            hooks = [block.register_forward_pre_hook(make_hook(index))
                     for index, block in enumerate(model.net.blocks)]
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model.generate_eval_latents_streaming(
                    inputs['latents'], inputs['cond_seq'], noise=inputs['noise'], total_len=125, **SAMPLER)
            torch.cuda.synchronize()
            report['diagnostic_sampler_seconds'] = time.perf_counter()-started
            trace = model.last_eval_meta['teacache']
            write_json(out/'trace.json', trace)
            report['trace_path'] = str((out/'trace.json').resolve())
            report['layer_forward_counts'] = counts
            old = torch.load(archived['outputs']['latents'], map_location='cpu', weights_only=True)
            result_cpu = result.cpu()
            report['checks']['latents_bitwise_identical_to_measured_run'] = bool(torch.equal(result_cpu, old))
            report['latent_max_abs_difference'] = float((result_cpu-old).abs().max())
            report['checks']['aggregate_decisions_match_measured_run'] = all(
                trace[key] == expected[key] for key in
                ('full_forwards', 'reused_forwards', 'denoising_forwards', 'commit_forwards', 'reason_counts', 'branches'))
            records = trace['decision_records']
            report['checks']['every_forward_has_a_trace_record'] = len(records) == trace['full_forwards']+trace['reused_forwards']
            report['checks']['record_decisions_match_summary'] = (
                Counter(r['reason'] for r in records) == Counter(trace['reason_counts'])
                and sum(r['decision'] == 'reuse' for r in records) == trace['reused_forwards'])
            report['checks']['every_layer_executes_on_full_calls_only'] = all(n == trace['full_forwards'] for n in counts)
            report['checks']['all_kv_writes_compute_fully'] = all(r['decision'] == 'full' for r in records if r['phase'] != 'denoise')
            indexed = {r['call_index']: r for r in records}
            report['checks']['reuse_sources_are_prior_full_calls_in_same_context'] = all(
                r['reference_call_index'] in indexed
                and r['reference_call_index'] < r['call_index']
                and indexed[r['reference_call_index']]['decision'] == 'full'
                and all(indexed[r['reference_call_index']][k] == r[k]
                        for k in ('branch', 'window_start', 'window_end', 'history_epoch'))
                for r in records if r['decision'] == 'reuse')
            report['checks']['updated_frames_are_in_window_and_exclude_conditioning'] = all(
                set(r['updated_latent_frames']).issubset(set(r['latent_frame_ids']))
                and 0 not in r['updated_latent_frames'] for r in records if r['phase'] == 'denoise')
            update_counts = Counter((r['branch'], f) for r in records if r['phase'] == 'denoise'
                                    for f in r['updated_latent_frames'])
            report['checks']['all_generated_frames_have_40_updates_per_branch'] = (
                len(update_counts) == 124*2 and set(update_counts.values()) == {40})
            report['checks']['only_teacache_acceleration_active'] = not model.last_eval_meta.get('headcast') and not model.last_eval_meta.get('kv_acceleration')
            report['checks']['source_stable_during_replay'] = all(
                hashlib.sha256((REPO/name).read_bytes()).hexdigest() == digest for name, digest in identity.items())
            report['original_counts'] = {key:expected[key] for key in
                ('full_forwards', 'reused_forwards', 'denoising_forwards', 'commit_forwards', 'reason_counts', 'branches')}
            if not all(report['checks'].values()):
                raise AssertionError('Reuse replay validation failed: '+str(report['checks']))
            report['status'] = 'complete'
            print(f'[{episode}] PASS: {len(records)} decisions; {trace["reused_forwards"]} reused; '
                  f'all {len(counts)} layers checked; output bitwise identical', flush=True)
        except Exception as error:
            report.update(status='failed', error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            for hook in hooks:
                hook.remove()
            write_json(out/'replay.json', report)
        del inputs, result, result_cpu, old
    signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
