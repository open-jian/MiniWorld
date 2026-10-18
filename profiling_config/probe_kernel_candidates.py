"""Diagnostic isolated torch.compile fusion prototypes; NOT production changes.

Uses MiniWorld's actual BF16 activation / FP32 norm-weight arithmetic and
projection strides. Compilation and warmup are excluded from steady timings.
Logical tensor traffic estimates and allocator requests are NOT measured HBM.
"""

import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miniworld.miniworld import VisionRotaryEmbeddingFast3D, modulate, rotate_half, BlockModulation

OUT = Path('/data2/jian/output/Miniworld/DROID_long_horizon/kernel_audit/micro')
CHECKPOINT = Path('/data2/jian/checkpoints/miniworld/MiniWorld_1b_droid.pt')


def rms(x, weight):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).type_as(x) * weight


def norm_modulation(x, weight, shift, scale):
    return modulate(rms(x, weight), shift, scale)


def swiglu(z):
    a, b = z.chunk(2, dim=-1)
    return F.silu(a) * b


def residual(x, gate, update):
    return x + gate * update


def qk_norm_rope_cast(q, k, qw, kw, cos, sin):
    q, k = rms(q, qw), rms(k, kw)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    # FP32 K is a required cache output; do not benchmark an implementation
    # which silently drops it or changes the precision of subsequent eviction.
    return q.to(torch.bfloat16), k, k.to(torch.bfloat16)


def timings(funcs, args, repeats=30, rounds=7):
    samples = {name: {'cuda_ms': [], 'wall_ms': []} for name in funcs}
    for func in funcs.values():
        for _ in range(10):
            result = func(*args)
    torch.cuda.synchronize()
    for rnd in range(rounds):
        for name in (list(funcs) if rnd % 2 == 0 else list(reversed(funcs))):
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record()
            for _ in range(repeats):
                result = funcs[name](*args)
            end.record()
            end.synchronize()
            samples[name]['cuda_ms'].append(begin.elapsed_time(end) / repeats)
            samples[name]['wall_ms'].append((time.perf_counter() - started) * 1000 / repeats)
    for row in samples.values():
        row['cuda_median_ms'] = statistics.median(row['cuda_ms'])
        row['wall_median_ms'] = statistics.median(row['wall_ms'])
    return samples


def parity(reference, candidate):
    refs = reference if isinstance(reference, tuple) else (reference,)
    vals = candidate if isinstance(candidate, tuple) else (candidate,)
    rows = []
    for old, new in zip(refs, vals):
        difference = (old.float() - new.float()).abs()
        denominator = old.float().abs()
        stable = denominator > 1e-3
        rows.append({'dtype': str(old.dtype), 'shape': list(old.shape),
                     'reference_stride': list(old.stride()), 'candidate_stride': list(new.stride()),
                     'bitwise_equal': torch.equal(old, new),
                     'different_fraction': (old != new).float().mean().item(),
                     'max_abs': difference.max().item(), 'mean_abs': difference.mean().item(),
                     'rms_error': difference.square().mean().sqrt().item(),
                     'max_rel_where_abs_reference_gt_1e_3': (difference[stable] / denominator[stable]).max().item(),
                     'all_finite': torch.isfinite(new).all().item()})
    return rows


def profile(name, func, args):
    # Capture after compile/warmup. Three calls improve kernel-count checks;
    # absolute profiler times are not used as the performance result.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, profile_memory=True) as prof:
        for _ in range(3):
            result = func(*args)
        torch.cuda.synchronize()
    trace = OUT / f'{name}_trace.json'
    prof.export_chrome_trace(str(trace))
    (OUT / f'{name}_operators.txt').write_text(prof.key_averages(group_by_input_shape=True).table(
        sort_by='self_cuda_time_total', row_limit=80))
    events = json.loads(trace.read_text())['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    allocations = [e for e in events if e.get('name') == '[memory]' and e.get('args', {}).get('Bytes', 0) > 0]
    counts = {}
    for event in kernels:
        counts[event['name']] = counts.get(event['name'], 0) + 1
    return {'kernel_count_per_call': len(kernels) / 3,
            'kernel_names_counts_for_three_calls': counts,
            'positive_allocator_bytes_per_call': sum(e['args']['Bytes'] for e in allocations) / 3,
            'allocator_request_count_per_call': len(allocations) / 3,
            'trace': str(trace),
            'allocator_note': 'Positive profiler allocation requests, not physical HBM traffic or peak live memory.'}


@torch.no_grad()
def probe_frame_modulation():
    """Compute identical per-frame modulation before spatial expansion.

The compared optimized output is expanded to the same dense token shape, so
this includes the broadcast/gather materialization cost. Different GEMM row
counts may change rounding; report rather than assume numerical equivalence.
"""
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20263368)
    ordinary_parameters = '--ordinary-parameters' in sys.argv
    context_label = 'ordinary' if ordinary_parameters else 'inference_parameters'
    result_file = 'frame_modulation_ordinary_parameters.json' if ordinary_parameters else 'frame_modulation.json'
    raw = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    weights = raw.get('ema_model', raw.get('model', raw))
    frames = torch.randn((1, 16, 1536), device='cuda', dtype=torch.float32)
    frame_ids = torch.arange(4800, device='cuda') // 300
    tokens = frames[:, frame_ids, :]
    shared_frames = torch.randn((1, 16, 9216), device='cuda', dtype=torch.bfloat16) * .25
    shared_tokens = shared_frames[:, frame_ids, :]
    report = {'production_modified': False,
              'method': 'Actual checkpoint weights; FP32 normalized embeddings, BF16 autocast GEMMs. Optimized path computes16 frames then materializes the same4800-token output. No pose/per-token-timestep case implied.',
              'execution_context': ('torch.no_grad, ordinary parameters, one enclosing BF16 autocast context.'
                                    if ordinary_parameters else
                                    'torch.inference_mode includes parameter construction, matching existing episode benchmark; one enclosing BF16 autocast context.'),
              'rows': []}
    specs = [('shared_mod', 'net.shared_mod.', None, 9216),
             ('block0_mod', 'net.blocks.0.modulation.', 0, 9216),
             ('block14_mod', 'net.blocks.14.modulation.', 14, 9216),
             ('block27_mod', 'net.blocks.27.modulation.', 27, 9216),
             ('final_mod', 'net.final_layer.adaLN_modulation.', None, 3072)]
    with torch.inference_mode(not ordinary_parameters), torch.autocast('cuda', dtype=torch.bfloat16):
        for name, prefix, layer, out_features in specs:
            is_block = layer is not None
            module = (BlockModulation(1536, 'adaln_lora', lora_dim=256) if is_block else
                      torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(1536, out_features)))
            state = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
            module.load_state_dict(state, strict=True)
            module = module.cuda().eval()
            def old():
                return module(tokens, shared_tokens, None) if is_block else module(tokens)
            def frame_first():
                result = module(frames, shared_frames, None) if is_block else module(frames)
                return result[:, frame_ids, :]
            reference, candidate = old(), frame_first()
            row = {'name': name, 'weight_prefix': prefix,
                   'parameters_are_inference_tensors': next(module.parameters()).is_inference(),
                   'per_frame_input_shape': list(frames.shape), 'token_input_shape': list(tokens.shape),
                   'output_shape': list(reference.shape), 'output_dtype': str(reference.dtype),
                   'timings': timings({'original_per_token': old, 'frame_first_including_expand': frame_first}, ()),
                   'numerical_difference': parity(reference, candidate),
                   'original_profile': profile(name + '_' + context_label + '_token', old, ()),
                   'frame_first_profile': profile(name + '_' + context_label + '_frame_first', frame_first, ())}
            per_row_madds = 1536 * 256 + 256 * 9216 if is_block else 1536 * out_features
            row['gemm_flops_original'] = 2 * 4800 * per_row_madds
            row['gemm_flops_frame_first'] = 2 * 16 * per_row_madds
            row['isolated_speedup'] = row['timings']['original_per_token']['cuda_median_ms'] / row['timings']['frame_first_including_expand']['cuda_median_ms']
            report['rows'].append(row)
            (OUT / result_file).write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20263368)
    actual_layout = '--actual-layout-only' in sys.argv
    device = torch.device('cuda')
    raw = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    weights = raw.get('ema_model', raw.get('model', raw))
    weight = weights['net.blocks.0.norm1.weight'].to(device)
    qw = weights['net.blocks.0.attn.q_norm.weight'].to(device)
    kw = weights['net.blocks.0.attn.k_norm.weight'].to(device)
    rope = VisionRotaryEmbeddingFast3D(128, 64, 15, 20)
    for name, buffer in rope.named_buffers():
        buffer.copy_(weights['net.feat_rope.' + name])
    rope = rope.to(device)
    cos, sin = rope._get_freqs(16, device, start_frame=48)
    del weights, raw
    x = (torch.randn((1, 1536, 4800), device=device, dtype=torch.bfloat16).transpose(1, 2)
         if actual_layout else torch.randn((1, 4800, 1536), device=device, dtype=torch.bfloat16))
    # Actual .chunk(6) modulation vectors retain a 6*C row stride.
    modulation = torch.randn((1, 4800, 6 * 1536), device=device, dtype=torch.bfloat16) * .25
    shift, scale, gate, *_ = modulation.chunk(6, dim=-1)
    # Attention projection produces contiguous B,N,C even when the residual
    # stream retains patch-embedding's B,C,N physical storage.
    update = torch.randn((1, 4800, 1536), device=device, dtype=torch.bfloat16)
    z = torch.randn((1, 4800, 8192), device=device, dtype=torch.bfloat16)
    qkv = torch.randn((1, 4800, 3, 12, 128), device=device, dtype=torch.bfloat16)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    e, mlp_e = x.numel(), z.numel() // 2
    candidates = [
        ('norm_modulation', norm_modulation, (x, weight, shift, scale), {
            'approx_eager_logical_bytes': 62 * e, 'fused_minimum_unique_input_output_bytes': 10 * e + weight.numel() * 4,
            'traffic_note': 'Elementwise-stage tensor reads+writes; small reduction vectors omitted. Minimum counts each unique input once; does not model cache reuse or physical HBM.'}),
        ('swiglu_activation', swiglu, (z,), {
            'approx_eager_logical_bytes': 10 * mlp_e, 'fused_minimum_unique_input_output_bytes': 6 * mlp_e,
            'traffic_note': 'BF16 SiLU intermediate then BF16 multiply: 10 bytes/output vs fused input+output 6 bytes/output; not measured HBM.'}),
        ('residual_gate_add', residual, (x, gate, update), {
            'approx_eager_logical_bytes': 12 * e, 'fused_minimum_unique_input_output_bytes': 8 * e,
            'traffic_note': 'Two BF16 elementwise stages vs three BF16 inputs+one output; not measured HBM.'}),
        ('qk_norm_rope_cast', qk_norm_rope_cast, (q, k, qw, kw, cos, sin), {
            'approx_eager_logical_bytes': int(168 * e + 16 * e / 12),
            'fused_minimum_unique_input_output_bytes': int(12 * e + 8 * e / 12 + (qw.numel() + kw.numel()) * 4),
            'traffic_note': 'Approximate explicit eager intermediates for two norms/rotations/casts. FP32 K remains an output. Coefficients counted once per table per stage; not measured HBM.'}),
    ]
    report = {'production_modified': False, 'gpu': torch.cuda.get_device_name(),
              'residual_layout': ('actual hooked stride [7372800,1,4800]' if actual_layout else
                                  'diagnostic contiguous B,N,C; does not match hooked residual layout'),
              'torch': torch.__version__, 'protocol': 'Seven alternating-order rounds of 30 eager calls, events outside inner loop; compilation excluded. Actual checkpoint norm weights and RoPE frequencies; synthetic nonzero activations.',
              'candidates': []}
    for name, original, args, traffic in candidates:
        if actual_layout and name not in ('norm_modulation', 'residual_gate_add'):
            continue
        trace_name = name + ('_actual_layout' if actual_layout else '')
        print(f'Compiling {name}', flush=True)
        compiled = torch.compile(original, fullgraph=True, dynamic=False)
        before = time.perf_counter()
        compiled(*args)
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - before
        # Default Inductor can remove intermediate BF16 rounding and enable
        # FP contraction. Also test its precision-emulation mode, keeping the
        # distinction explicit instead of calling default fusion equivalent.
        strict = torch.compile(original, fullgraph=True, dynamic=False,
                               options={'emulate_precision_casts': True})
        before = time.perf_counter()
        strict(*args)
        torch.cuda.synchronize()
        strict_compile_seconds = time.perf_counter() - before
        row = {'name': name, 'initial_compile_and_first_call_seconds': compile_seconds,
               'strict_initial_compile_and_first_call_seconds': strict_compile_seconds,
               'inputs': [{'shape': list(t.shape), 'stride': list(t.stride()), 'dtype': str(t.dtype)} for t in args],
               'timings': timings({'eager': original, 'compiled_prototype': compiled,
                                   'compiled_precision_emulation': strict}, args),
               'numerical_difference': parity(original(*args), compiled(*args)),
               'precision_emulation_numerical_difference': parity(original(*args), strict(*args)),
               'logical_traffic': traffic,
               'eager_profile': profile(trace_name + '_eager', original, args),
               'compiled_profile': profile(trace_name + '_compiled', compiled, args),
               'precision_emulation_profile': profile(trace_name + '_precision_emulation', strict, args)}
        row['isolated_speedup'] = row['timings']['eager']['cuda_median_ms'] / row['timings']['compiled_prototype']['cuda_median_ms']
        row['precision_emulation_isolated_speedup'] = row['timings']['eager']['cuda_median_ms'] / row['timings']['compiled_precision_emulation']['cuda_median_ms']
        report['candidates'].append(row)
        report_file = 'kernel_candidates_actual_layout.json' if actual_layout else 'kernel_candidates.json'
        (OUT / report_file).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    probe_frame_modulation() if '--modulation-only' in sys.argv else main()
