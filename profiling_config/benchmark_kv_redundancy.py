"""Bounded cost replay and packing-inclusive KV-redundancy microbenchmarks.

No production modules or generation paths are modified. The full-model replay
uses previously generated clean latents; it is a cost experiment, not a rollout
quality evaluation. Candidate selection and index scheduling are excluded from
microbenchmarks, while gathering tensors and assembling outputs are included.
"""

from __future__ import annotations

import time

PROCESS_START = time.monotonic()

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import traceback

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
WORK = Path('/home/jian/Documents/Codex/2026-10-19/hello/work')
INPUT = Path('/data2/jian/output/Miniworld/DROID_long_horizon/chunk_flash_attention/episode_002352')


def summarize(values):
    return dict(median_ms=statistics.median(values), mean_ms=statistics.mean(values),
                min_ms=min(values), max_ms=max(values), samples_ms=values)


def measure(call, repeats=7, warmups=3):
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    events = []
    for _ in range(repeats):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        result = call()
        end.record()
        events.append((start, end))
        del result
    torch.cuda.synchronize()
    return summarize([a.elapsed_time(b) for a, b in events])


def alternating(calls, repeats=10, warmups=2):
    for call in calls.values():
        for _ in range(warmups):
            call()
    torch.cuda.synchronize()
    events = defaultdict(list)
    items = list(calls.items())
    for repeat in range(repeats):
        for name, call in (items if repeat % 2 == 0 else list(reversed(items))):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            result = call()
            end.record()
            events[name].append((start, end))
            del result
    torch.cuda.synchronize()
    return {name: summarize([a.elapsed_time(b) for a, b in pairs]) for name, pairs in events.items()}


def write_json(path, report):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def full_forward_cost(args):
    from miniworld.denoiser import DenoiserConfig, build_denoiser_from_mode
    import miniworld.miniworld as core

    manifest = json.loads((args.input_dir / 'experiment_manifest.json').read_text())
    raw = torch.load(manifest['args']['checkpoint'], map_location='cpu', weights_only=True)
    weights = raw.get('ema_model', raw.get('model', raw))
    options = dict(manifest['args'])
    options.update(latent_size=(15, 20), latent_frames=weights['net.feat_rope.freqs_cos'].shape[0] // 300,
                   wm_use_checkpoint=False, headcast=False)
    model = build_denoiser_from_mode(DenoiserConfig(**options))
    model.load_state_dict(weights, strict=True)
    del raw, weights
    model = model.cuda().eval()
    net = model.net
    baseline = torch.load(args.baseline_latents, map_location='cpu', weights_only=True).cuda()
    prepared = torch.load(args.input_dir / 'prepared_inputs.pt', map_location='cpu', weights_only=True)
    condition = prepared['cond_seq'].cuda()
    noise = prepared['noise'][:, :, 48:64].cuda()
    del prepared
    # The eight chunks occupy different noise levels. This is deliberately a
    # replay for cost, not a claim that these mixed latents form a sampler state.
    t = torch.linspace(.275887, 1., 8, device='cuda').repeat_interleave(2).unsqueeze(0)
    current = baseline[:, :, 48:64] * (1 - t[:, None, :, None, None]) + noise * t[:, None, :, None, None]
    rows = {}
    past = None
    for branch in ('cond', 'uncond'):
        drop = None if branch == 'cond' else torch.ones(1, device='cuda', dtype=torch.bool)
        if past is not None:
            del past
        _, past = net.forward_with_cache(
            baseline[:, :, :48], torch.zeros(1, 48, device='cuda'), condition[:, :48],
            return_kv=True, chunk_size=2, cond_drop=drop,
        )
        def call(commit=False):
            return net.forward_with_cache(
                baseline[:, :, 48:50] if commit else current,
                torch.zeros(1, 2, device='cuda') if commit else t,
                condition[:, 48:50] if commit else condition[:, 48:64],
                past_kv_list=past, current_position_offset=48, return_kv=commit,
                chunk_size=2, cond_drop=drop,
            )
        unhooked = measure(call)
        commit = measure(lambda: call(True), repeats=5, warmups=2)
        events = defaultdict(list)
        handles = []
        def add_hooks(module, name):
            stack = []
            def before(mod, inputs):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                stack.append(event)
            def after(mod, inputs, output):
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                events[name].append((stack.pop(), event))
            handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))
        for block in net.blocks:
            for module, name in ((block.attn.qkv, 'qkv_projection'),
                                 (block.attn.proj, 'attention_output_projection'),
                                 (block.attn, 'attention_module_inclusive'),
                                 (block.mlp, 'mlp')):
                add_hooks(module, name)
        original = core._chunkwise_flash_attention
        def timed_attention(*positional, **keywords):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            out = original(*positional, **keywords)
            end.record()
            events['attention_primitive'].append((start, end))
            return out
        core._chunkwise_flash_attention = timed_attention
        by_repeat = defaultdict(list)
        totals = []
        try:
            for _ in range(7):
                events.clear()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                result = call()
                end.record()
                torch.cuda.synchronize()
                totals.append(start.elapsed_time(end))
                for name, pairs in events.items():
                    by_repeat[name].append(sum(a.elapsed_time(b) for a, b in pairs))
                del result
        finally:
            core._chunkwise_flash_attention = original
            for handle in handles:
                handle.remove()
        for index, total in enumerate(totals):
            attn = by_repeat['attention_module_inclusive'][index]
            by_repeat['attention_other_cache_rope_norm'].append(attn - sum(by_repeat[key][index] for key in
                ('qkv_projection', 'attention_output_projection', 'attention_primitive')))
            by_repeat['other_model'].append(total - attn - by_repeat['mlp'][index])
        rows[branch] = dict(unhooked=unhooked, instrumented=summarize(totals), clean_commit_2_frames=commit,
                            components={name: summarize(values) for name, values in by_repeat.items()},
                            instrumentation_overhead_fraction=statistics.median(totals) / unhooked['median_ms'] - 1,
                            persistent_kv_bytes=sum(t.numel() * t.element_size() for pair in past for t in pair))
        print(f'{branch} full forward: {unhooked["median_ms"]:.3f} ms', flush=True)
    output = dict(branches=rows, num_layers=net.depth, num_heads=net.num_heads, hidden_size=net.hidden_size,
                  cache_frames=48, current_frames=16, current_tokens=4800, past_tokens=14400,
                  noise_levels=t[0, ::2].tolist(),
                  scope='fixed-state cost replay, generated clean first48 prefilled in two-frame block-causal order; not a streaming trajectory',
                  timing_excludes='prefill, model loading, sampler integration, cache eviction, VAE',
                  instrumentation_note='disjoint components sum to instrumented total; inclusive attention_module is not additive',
                  baseline_latents=str(args.baseline_latents), checkpoint=manifest['args']['checkpoint'])
    del model, net, past, baseline, current, condition, noise
    torch.cuda.empty_cache()
    return output


def build_masked_call(q, k, v, keep, n_past, chunk_tokens):
    from flash_attn import flash_attn_with_kvcache
    # Batch each head independently; real lengths prevent padding tokens from
    # entering softmax. The static gather schedule is outside the timed call.
    b, h, n_current, d = q.shape
    counts = keep.sum(-1).to(torch.int32)
    max_history = int(counts.max())
    index = torch.zeros(h, max_history + n_current, device=q.device, dtype=torch.long)
    for head in range(h):
        selected = keep[head].nonzero().flatten()
        count = selected.numel()
        index[head, :count] = selected
        index[head, count:count + n_current] = torch.arange(n_past, n_past + n_current, device=q.device)
    gather_index = index[None, :, :, None].expand(b, h, -1, d)
    lengths = [(counts + end).repeat(b) for end in range(chunk_tokens, n_current + chunk_tokens, chunk_tokens)]
    def call():
        packed_k = torch.gather(k, 2, gather_index).reshape(b * h, -1, 1, d)
        packed_v = torch.gather(v, 2, gather_index).reshape(b * h, -1, 1, d)
        flat_q = q.contiguous().reshape(b * h, n_current, 1, d)
        results = []
        for number, start in enumerate(range(0, n_current, chunk_tokens)):
            end = min(start + chunk_tokens, n_current)
            valid = lengths[number] if end - start == chunk_tokens else (counts + end).repeat(b)
            results.append(flash_attn_with_kvcache(flat_q[:, start:end], packed_k, packed_v,
                                                  cache_seqlens=valid, causal=False))
        return torch.cat(results, dim=1).reshape(b, h, n_current, d)
    return call


@torch.autocast('cuda', enabled=False)
def validate_sampled(q, k, v, actual, keep, n_past, chunk_tokens):
    # Independently evaluate the exact retained-key softmax for four spatial
    # samples in every query chunk. This also detects future-prefix leakage.
    positions = []
    for start in range(0, q.shape[2], chunk_tokens):
        end = min(start + chunk_tokens, q.shape[2])
        positions.extend(torch.linspace(start, end - 1, 4, device=q.device).long().tolist())
    ids = torch.tensor(positions, device=q.device)
    scores = torch.matmul(q[:, :, ids].float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    allowed_history = keep[:, None, :].expand(-1, len(positions), -1)
    visible = ((ids // chunk_tokens + 1) * chunk_tokens).clamp_max(q.shape[2])
    current_mask = torch.arange(q.shape[2], device=q.device)[None, :] < visible[:, None]
    allowed = torch.cat((allowed_history, current_mask[None].expand(q.shape[1], -1, -1)), dim=-1)
    reference = torch.softmax(scores.masked_fill(~allowed[None], -torch.inf), dim=-1) @ v.float()
    observed = actual[:, :, ids].float()
    error = ((observed - reference).square().mean() / reference.square().mean().clamp_min(1e-12)).sqrt().item()
    if not math.isfinite(error) or error > .015:
        raise AssertionError(f'Packed attention disagrees with exact retained-key reference: RRMSE={error}')
    return dict(relative_rmse_vs_exact_retained_softmax=error, sampled_queries=len(positions))


def projection_cost(probe):
    x = probe['input_x'].cuda()
    weight = probe['qkv_weight'].cuda().to(torch.bfloat16)
    bias = probe.get('qkv_bias')
    bias = None if bias is None else bias.cuda().to(torch.bfloat16)
    width = weight.shape[0] // 3
    heads = probe['q'].shape[1]
    dim = width // heads
    old = F.linear(x, weight, bias)
    old_k, old_v = old[..., width:2 * width].contiguous(), old[..., 2 * width:].contiguous()
    calls = {'fused_qkv': lambda: F.linear(x, weight, bias)}
    for reuse_fraction in (.25, .5, .75, 1.):
        fresh_heads = round(heads * (1 - reuse_fraction))
        fresh_width = fresh_heads * dim
        fresh_weight = torch.cat((weight[width:width + fresh_width], weight[2 * width:2 * width + fresh_width]))
        fresh_bias = None if bias is None else torch.cat((bias[width:width + fresh_width], bias[2 * width:2 * width + fresh_width]))
        def call(n=fresh_width, fw=fresh_weight, fb=fresh_bias):
            q_new = F.linear(x, weight[:width], None if bias is None else bias[:width])
            if n:
                kv_new = F.linear(x, fw, fb)
                k_new, v_new = kv_new.split(n, dim=-1)
                return torch.cat((q_new, k_new, old_k[..., n:], v_new, old_v[..., n:]), dim=-1)
            return torch.cat((q_new, old_k, old_v), dim=-1)
        calls[f'reuse_{reuse_fraction:.2f}'] = call
    timing = alternating(calls)
    base = timing['fused_qkv']['median_ms']
    for name, row in timing.items():
        row['fraction_time_saved_vs_fused'] = 1 - row['median_ms'] / base
        assembled = calls[name]()
        with torch.autocast('cuda', enabled=False):
            assembly_error = ((assembled.float() - old.float()).square().mean() /
                              old.float().square().mean().clamp_min(1e-12)).sqrt().item()
        if not math.isfinite(assembly_error) or assembly_error > .015:
            raise AssertionError(f'Raw QKV output assembly mismatch: {name}: {assembly_error}')
        row['same_input_assembly_relative_rmse'] = assembly_error
        del assembled
    return dict(cases=timing, input_shape=list(x.shape), stale_kv_storage_bytes=old_k.numel() * old_k.element_size() * 2,
                scope='projection only; all K/V in stale storage, contiguous fresh-head prefix, Q recomputed; output assembly and old KV reads included',
                excludes='selection, Q/K normalization and RoPE, attention, output projection, MLP; no whole-layer skip',
                implementation='separate Q and fresh-KV GEMMs with prepacked static weights; BF16 weights match warmed CUDA autocast',
                accuracy_note='cost-only surrogate old K/V from same x; eligibility and stale error must come from trajectory profile')


def microbenchmarks(args):
    import miniworld.miniworld as core
    # torch.__version__ is a str subclass in recent PyTorch releases. The local
    # collector includes it in metadata; narrowly allowlist that harmless type
    # while retaining weights_only rather than enabling arbitrary unpickling.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(args.probes, map_location='cpu', weights_only=True)
    rows = []
    for probe in saved['probes']:
        q, k, v = (probe[name].cuda().to(torch.bfloat16) for name in ('q', 'k_full', 'v_full'))
        n_past, chunk_tokens = probe['n_past'], probe['chunk_tokens']
        n_cur, heads = q.shape[2], q.shape[1]
        dense_keep = torch.ones(heads, n_past, device=q.device, dtype=torch.bool)
        calls = {'dense_native': lambda: core._chunkwise_flash_attention(q, k, v, chunk_tokens),
                 'dense_head_batch': build_masked_call(q, k, v, dense_keep, n_past, chunk_tokens)}
        masks = {'dense_native': dense_keep, 'dense_head_batch': dense_keep}
        for name, keep in probe.get('recent_masks', {}).items():
            keep = keep.cuda()
            if not torch.equal(keep, keep[:1].expand_as(keep)):
                raise ValueError('Recent-retention masks must be identical across heads for native layout')
            index = torch.cat((keep[0].nonzero().flatten(), torch.arange(n_past, n_past + n_cur, device=q.device)))
            calls[name] = lambda index=index: core._chunkwise_flash_attention(q, k.index_select(2, index), v.index_select(2, index), chunk_tokens)
            masks[name] = keep
        for name, keep in probe.get('similarity_masks', {}).items():
            keep = keep.cuda()
            calls[name] = build_masked_call(q, k, v, keep, n_past, chunk_tokens)
            masks[name] = keep
        timing = alternating(calls)
        for name, call in calls.items():
            actual = call()
            timing[name].update(validate_sampled(q, k, v, actual, masks[name], n_past, chunk_tokens))
            counts = masks[name].sum(-1)
            timing[name].update(history_retained_fraction=counts.float().mean().item() / n_past,
                                retained_tokens_per_head=counts.tolist(),
                                fraction_attention_time_saved=1 - timing[name]['median_ms'] / timing['dense_native']['median_ms'])
            del actual
        rows.append(dict(layer=int(probe['layer']), state=probe.get('state'), step=probe.get('step'),
                         branch=probe.get('branch'), q_shape=list(q.shape), kv_shape=list(k.shape),
                         attention=timing, projection=projection_cost(probe)))
        print(f'Layer {probe["layer"]} microbenchmarks finished', flush=True)
        del q, k, v, calls, masks
        torch.cuda.empty_cache()
    return dict(probes=rows, input_file=str(args.probes), metadata=saved.get('metadata', {}),
                timing_includes='all gathers, KV packing, legal-current-prefix attention, output assembly',
                timing_excludes='similarity decision, index-schedule construction, persistent-cache update, projections for attention tests',
                semantics='retained historical keys plus all legal current keys through current query chunk; one joint softmax, no future chunks',
                limits='single full-cache state and representative layers; not end-to-end speed or rollout quality')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=INPUT)
    parser.add_argument('--baseline-latents', type=Path, default=WORK / 'headcast_migration/run_001/baseline_latents.pt')
    parser.add_argument('--probes', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--budget-seconds', type=float, default=120.)
    parser.add_argument('--skip-model', action='store_true')
    args = parser.parse_args()
    if not 0 < args.budget_seconds <= 120:
        parser.error('budget must be between 0 and 120 seconds')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'timing.json'
    if path.exists():
        raise FileExistsError(path)
    def expired(signum, frame):
        raise TimeoutError('120-second process budget exhausted')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, args.budget_seconds - (time.monotonic() - PROCESS_START)))
    report = dict(status='running', units='milliseconds unless marked otherwise',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    try:
        torch.set_num_threads(4)
        report.update(gpu=torch.cuda.get_device_name(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      torch_version=torch.__version__, compute_dtype='bfloat16', headcast_enabled=False)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if not args.skip_model:
                report['full_forward'] = full_forward_cost(args)
                write_json(path, report)
            report['microbenchmarks'] = microbenchmarks(args)
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report['process_seconds'] = time.monotonic() - PROCESS_START
        write_json(path, report)
        print(f'Timing report: {path}, status={report["status"]}, seconds={report["process_seconds"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
