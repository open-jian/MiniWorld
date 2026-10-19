"""Compare production fused_k_concat_cast against the seven-optimization baseline.

The shared protocol keeps fixed inputs, noise, sampling, cache management, and
streaming VAE unchanged. Each process first warms both complete 125-latent DiT
paths, then executes the shared 17-latent sampler/VAE warmup and paired timed
rollout. Compilation and the added kernel/fingerprint diagnostics are outside the
reported rollout time; the shared CPU-only eviction recorder remains inside.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import benchmark_framewise_conditioning as protocol
from benchmark_fused_bias_swiglu import output_fingerprints, tensor_fingerprint
from benchmark_fused_norm_modulation import source_provenance
import miniworld.miniworld as model_module

OUTPUT_ROOT = Path('/data2/jian/output/Miniworld/fused_k_concat_cast_integration_rack3')
PREPARED_ROOT = Path('/data2/jian/output/Miniworld/framewise_conditioning_4episodes_rack3')
RESEARCH_ROOT = Path('/data2/jian/output/Miniworld/fused_k_concat_cast_research_20261018/rollout')
ARCHIVED_ROOT = Path('/data2/jian/output/Miniworld/fused_bias_swiglu_4episodes_rack3')
FIXED = {name: True for name in (
    'chunkwise_flash_attention', 'framewise_conditioning', 'fused_norm_modulation',
    'reuse_rope_frequencies', 'fused_residual_gate_add', 'fused_rope', 'fused_bias_swiglu',
)}
SAMPLING = {'history_len': 1, 'max_cache_chunks': 24, 'inflight_chunks': 8,
            'sink_frames': 1, 'chunk_size': 2, 'num_sampling_steps': 100,
            'df_ardiff_step': 5, 'cfg_scale': 2.0}


def configure_protocol():
    # Use the production OptimizationConfig and actual production call site.
    protocol.configure_comparison('concat_cast', 'fused_k_concat_cast',
                                  'Fused K concatenation and cast', FIXED)


@contextmanager
def count_completed_calls():
    """Observe the actual kernel symbol; restore it before any timing begins."""
    original = model_module.fused_k_concat_cast
    signatures = Counter()

    def record(past, current, *args, **kwargs):
        result = original(past, current, *args, **kwargs)
        signature = tuple((tuple(value.shape), tuple(value.stride()), str(value.dtype),
                           value.data_ptr() % 16 == 0) for value in (past, current, result))
        signatures[signature] += 1
        return result

    model_module.fused_k_concat_cast = record
    try:
        yield signatures
    finally:
        model_module.fused_k_concat_cast = original


def signature_rows(signatures):
    return [dict(zip(('past', 'current', 'output'), [
        {'shape': list(shape), 'stride': list(stride), 'dtype': dtype,
         'pointer_aligned_16': aligned} for shape, stride, dtype, aligned in signature]), calls=count)
        for signature, count in signatures.items()]


def execution_diagnostics(name, model, call):
    original_get_freqs = model_module.VisionRotaryEmbeddingFast3D._get_freqs
    frequency_calls = 0

    def record_frequencies(self, *args, **kwargs):
        nonlocal frequency_calls
        result = original_get_freqs(self, *args, **kwargs)
        frequency_calls += 1
        return result

    model_module.VisionRotaryEmbeddingFast3D._get_freqs = record_frequencies
    try:
        with count_completed_calls() as signatures:
            output = call()
            fingerprints = output_fingerprints(output)
            dtypes = sorted({str(key.dtype) for key, _ in output[1]})
            del output
    finally:
        model_module.VisionRotaryEmbeddingFast3D._get_freqs = original_get_freqs
    expected = len(model.net.blocks) if name == 'concat_cast' else 0
    calls = sum(signatures.values())
    if calls != expected or frequency_calls != 1 or dtypes != ['torch.float32']:
        raise AssertionError(f'{name}: invalid kernel calls {calls}/{expected}, frequency calls {frequency_calls}, or K dtypes {dtypes}')
    return {'fused_k_concat_cast_calls': calls, 'expected_fused_k_concat_cast_calls': expected,
            'rope_frequency_lookup_calls': frequency_calls, 'expected_rope_frequency_lookup_calls': 1,
            'current_k_dtypes': dtypes, 'all_expected_calls_observed': True,
            'output_fingerprints': fingerprints, 'input_signatures': signature_rows(signatures)}


def full_shape_warmup(args, models, data):
    """Warm all actual history/current/stride specializations without VAE decode."""
    inputs = {key: tensor_fingerprint(data[key]) for key in ('latents', 'noise', 'cond_seq')}
    records = {}
    for name in args.rollout_order:
        print(f'Full-shape compile warmup: {name}, {args.rollout_latents} latents, no VAE; excluded from reported time', flush=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with count_completed_calls() as signatures:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                warm = protocol.generate(models[name], data, args.rollout_latents, decoder=None)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        if tuple(warm.shape) != (1, 48, args.rollout_latents, 15, 20):
            raise AssertionError(f'{name}: incorrect warmup output shape')
        latent_bits = tensor_fingerprint(warm)
        del warm
        calls = sum(signatures.values())
        if (name == 'original' and calls != 0) or (name == 'concat_cast' and calls <= 0):
            raise AssertionError(f'{name}: invalid completed warmup fused calls {calls}')
        if models[name].last_eval_meta.get('effective_steps') != 40:
            raise AssertionError(f'{name}: warmup changed effective sampler steps')
        if any(tensor_fingerprint(data[key]) != value for key, value in inputs.items()):
            raise AssertionError(f'{name}: warmup modified prepared inputs')
        records[name] = {'elapsed_seconds_excluded_from_rollout': elapsed,
            'latent_fingerprint': latent_bits, 'completed_fused_k_concat_cast_calls': calls,
            'unique_fused_tensor_signatures': len(signatures), 'input_signatures': signature_rows(signatures),
            'prepared_inputs_unchanged': True, 'sampler_effective_steps': 40}
        print(f'Full-shape warmup finished: {name}; {elapsed:.3f} excluded seconds; {calls} fused calls / {len(signatures)} signatures', flush=True)
    if records['original']['latent_fingerprint'] != records['concat_cast']['latent_fingerprint']:
        raise AssertionError('Complete warmup latent bits differ between OFF and ON')
    result = {'scope': 'complete same-input latent generation per path, before shared 17-latent VAE warmup and timed full rollout',
        'latent_frames_per_path': args.rollout_latents, 'stream_decoder': None,
        'collect_stream_timing': False, 'sampling': SAMPLING, 'order': list(args.rollout_order),
        'runs': records, 'paired_warmup_latents_bitwise_equal': True,
        'all_diagnostic_wrappers_restored_before_timing': True}
    protocol.dump(args.output_dir/'full_shape_compile_warmup.json', result)
    return result


def capture_benchmark_helpers(output_dir, provenance):
    """Snapshot external launch/analysis code as well as shared repository sources."""
    paths = sorted(OUTPUT_ROOT.glob('*.py'))
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    payload = {'source_fingerprint_sha256': provenance['source_fingerprint_sha256'],
               'external_source_sha256': sources}
    value = {**payload, 'benchmark_fingerprint_sha256': hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()}
    destination = output_dir/'source_snapshot'/'external'
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copyfile(path, destination/path.name)
    protocol.dump(destination/'manifest.json', value)
    return value


def report_metadata(args, models, data):
    # The shared source scanner already includes all miniworld Python modules,
    # profiling drivers, tests, and optimization_config.toml, including new files.
    provenance = source_provenance(args.output_dir)
    required = ('miniworld/fused_k_concat_cast.py', 'tests/test_fused_k_concat_cast.py',
                'profiling_config/benchmark_fused_k_concat_cast.py')
    if any(name not in provenance['source_sha256'] for name in required):
        raise AssertionError('Production kernel, tests, or benchmark missing from source provenance')
    helpers = capture_benchmark_helpers(args.output_dir, provenance)
    properties = {}
    for name, model in models.items():
        parameters = list(model.parameters())
        properties[name] = {'all_normal_tensors': all(not value.is_inference() for value in parameters),
                            'all_require_grad': all(value.requires_grad for value in parameters),
                            'all_float32': all(value.dtype == torch.float32 for value in parameters)}
        if not all(properties[name].values()):
            raise AssertionError(f'{name}: ordinary FP32 parameters are required')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        history = protocol.build_shared_cache(models['original'], data, 2)
        before = [[tensor_fingerprint(value) for value in pair] for pair in history]
        versions = [(key._version, value._version) for key, value in history]
        startup = {name: execution_diagnostics(name, model, lambda model=model: model.net.forward_with_cache(
            data['gt_latents'][:, :, 2:18], torch.zeros(1, 16, device='cuda'), data['cond_seq'][:, 2:18],
            past_kv_list=history, current_position_offset=2, return_kv=True, chunk_size=2,
        )) for name, model in models.items()}
        if before != [[tensor_fingerprint(value) for value in pair] for pair in history]:
            raise AssertionError('Canonical history storage bits were modified')
        if versions != [(key._version, value._version) for key, value in history]:
            raise AssertionError('Canonical history versions were modified')
        del history
    if startup['original']['output_fingerprints'] != startup['concat_cast']['output_fingerprints']:
        raise AssertionError('Startup prediction/current K/V bits differ')
    warmup = full_shape_warmup(args, models, data)
    return {'source_provenance': provenance, 'source_sha256': provenance['source_sha256'],
        'benchmark_helper_provenance': helpers, 'production_integration': True,
        'parameter_properties': properties, 'full_shape_compile_warmup': warmup,
        'autocast_scope': 'one BF16 autocast context per complete rollout',
        'timing_definitions': {'rollout': 'generation plus streaming VAE decode, seconds; full-shape warmup excluded'},
        'execution_validation': {'startup_forward': startup, 'history_latents': 2, 'current_latents': 16,
            'canonical_history_bits_unchanged': True, 'canonical_history_versions_unchanged': True,
            'counter': 'miniworld.miniworld.fused_k_concat_cast',
            'scope': 'separate untimed forwards/warmup; added kernel wrappers restored before measured rollout'}}


def validate_saved_outputs(args):
    path = args.output_dir/'benchmark.json'
    report = json.loads(path.read_text())
    fingerprints = {mode: tensor_fingerprint(torch.load(args.output_dir/f'{mode}_latents.pt',
        map_location='cpu', weights_only=True)) for mode in protocol.MODES}
    rgb = {mode: np.load(args.output_dir/f'{mode}_rgb.npy', mmap_mode='r') for mode in protocol.MODES}
    checks = {'latent_fingerprints': fingerprints,
        'generated_latents_bitwise_equal': fingerprints['original'] == fingerprints['concat_cast'],
        'generated_rgb_identical': bool(np.array_equal(rgb['original'], rgb['concat_cast'])),
        'forward_output_bits_equal': True,
        'timed_latents_match_full_shape_warmup': all(fingerprints[mode] ==
            report['full_shape_compile_warmup']['runs'][mode]['latent_fingerprint'] for mode in protocol.MODES)}
    # Output folders retain episode/repetition names so the same seed, prepared
    # input, and saved historical arrays can be compared without pairing guesses.
    relative = Path(f"episode_{report['episode']}")/args.output_dir.parent.name/'results'
    archives = [('research_off', RESEARCH_ROOT/relative, 'original', 'original'),
                ('research_on', RESEARCH_ROOT/relative, 'concat_cast', 'concat_cast'),
                ('archived_bias_swiglu_on', ARCHIVED_ROOT/relative, 'bias_swiglu', 'original')]
    for label, folder, archived_mode, mode in archives:
        archived = json.loads((folder/'benchmark.json').read_text())
        if archived['seed'] != report['seed'] or archived['prepared_input_sha256'] != report['prepared_input_sha256']:
            raise AssertionError(f'{label}: archived seed or prepared input changed')
        checks[f'latent_bits_match_{label}'] = fingerprints[mode] == archived[
            'exact_output_validation']['latent_fingerprints'][archived_mode]
        checks[f'rgb_matches_{label}'] = bool(np.array_equal(rgb[mode],
            np.load(folder/f'{archived_mode}_rgb.npy', mmap_mode='r')))
    report['exact_output_validation'] = checks
    protocol.dump(path, report)
    protocol.dump(args.output_dir/'exact_output_validation.json', checks)
    if any(value is not True for key, value in checks.items() if key != 'latent_fingerprints'):
        raise AssertionError(f'Production concat/cast exact-output validation failed: {checks}')


def main():
    configure_protocol()
    args = protocol.parse_args(defaults={
        'prepared_dir': PREPARED_ROOT/'episode_002352'/'prepared',
        'output_dir': OUTPUT_ROOT/'episode_002352'/'repeat_01'/'results',
        'rollout_latents': 125, 'skip_forward': True, 'warmup': 7, 'repeats': 31,
    }, description=__doc__)
    if not args.skip_forward or args.rollout_latents != 125:
        raise ValueError('This integration driver requires a full 125-latent rollout; standalone forwards are separate')
    protocol.main(args, report_metadata=report_metadata, execution_mode='no_grad')
    validate_saved_outputs(args)


if __name__ == '__main__':
    main()
