"""Read-only kernel audit of the unchanged episode-002352 resident-KV rollout.

Profiles real model calls at five history/current shapes and one eviction. The
extra labels and CUPTI instrumentation perturb wall time: these are NOT E2E
benchmarks. GPU kernel durations are disjoint; enclosing CPU scopes are labels,
not extra GPU time. Production modules are patched only within this process.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
from unittest import mock

import torch
from torch.profiler import ProfilerActivity, profile, record_function
import benchmark_episode_chunk_flash as bench
from miniworld import miniworld as mw
from miniworld.kv_cache import ResidentKVCache


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_meta(t):
    if not isinstance(t, torch.Tensor):
        return None
    return dict(shape=list(t.shape), dtype=str(t.dtype), stride=list(t.stride()))


def summarize_trace(path):
    trace = json.loads(path.read_text())
    events = trace['traceEvents']
    cpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') in ('cpu_op', 'user_annotation')]
    scopes = [e for e in cpu if e['name'].startswith('audit/')]
    # Kineto correlation IDs link device activity to its launching ATen op.
    # Find the innermost explicitly labelled CPU scope containing that op.
    launches = {e.get('args', {}).get('correlation'): e for e in cpu
                if 'correlation' in e.get('args', {})}
    externals = {e.get('args', {}).get('External id'): e for e in cpu
                 if 'External id' in e.get('args', {})}
    kernels = []
    for e in events:
        if e.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        args = e.get('args', {})
        op = launches.get(args.get('correlation')) or externals.get(args.get('External id'))
        parents = [] if op is None else [s for s in scopes if s['tid'] == op['tid']
                    and s['ts'] <= op['ts'] and s['ts'] + s['dur'] >= op['ts'] + op['dur'] - 0.01]
        parents.sort(key=lambda s: s['dur'])
        kernels.append(dict(name=e['name'], category=e['cat'], duration_us=e.get('dur', 0),
                            ts=e['ts'], op=op['name'] if op else 'UNLINKED',
                            scope=parents[0]['name'] if parents else 'unlabelled',
                            scope_chain=' > '.join(s['name'] for s in reversed(parents)),
                            grid=str(args.get('grid', '')), block=str(args.get('block', '')),
                            registers_per_thread=args.get('registers per thread', ''),
                            shared_memory=args.get('shared memory', ''), stream=args.get('stream', '')))
    grouped = defaultdict(lambda: [0, 0.0])
    names = defaultdict(lambda: [0, 0.0])
    for e in kernels:
        grouped[e['scope']][0] += 1
        grouped[e['scope']][1] += e['duration_us']
        names[e['name']][0] += 1
        names[e['name']][1] += e['duration_us']
    total = sum(k['duration_us'] for k in kernels)
    def rows(g, key):
        return [{key: k, 'count': v[0], 'gpu_ms': v[1] / 1000,
                 'gpu_percent': v[1] / total * 100 if total else 0}
                for k, v in sorted(g.items(), key=lambda item: -item[1][1])]
    ranges = sorted((k['ts'], k['ts'] + k['duration_us']) for k in kernels)
    busy = 0
    if ranges:
        start, end = ranges[0]
        for a, b in ranges[1:]:
            if a > end:
                busy += end - start
                start, end = a, b
            else:
                end = max(end, b)
        busy += end - start
    runtime = [e for e in events if e.get('cat') == 'cuda_runtime']
    result = dict(kernel_count=len(kernels), gpu_sum_ms=total/1000, gpu_union_busy_ms=busy/1000,
                  gpu_first_to_last_ms=(ranges[-1][1]-ranges[0][0])/1000 if ranges else 0,
                  linked_kernel_count=sum(k['op'] != 'UNLINKED' for k in kernels),
                  runtime_calls=dict(Counter(e['name'] for e in runtime)),
                  scopes=rows(grouped, 'scope'), kernel_names=rows(names, 'kernel'))
    write_csv(path.with_name(path.stem + '_kernels.csv'), kernels)
    write_csv(path.with_name(path.stem + '_scopes.csv'), result['scopes'])
    bench.dump(path.with_name(path.stem + '_summary.json'), result)
    with path.open('rb') as src, gzip.open(str(path) + '.gz', 'wb') as dest:
        dest.write(src.read())
    path.unlink()
    return result


class Audit:
    def __init__(self, model, out, nsys=False):
        self.model, self.out, self.nsys = model, out, nsys
        self.stack = ExitStack()
        self.active = False
        self.done = {}
        self.forward_id = 0
        self.counts = Counter()
        self.tensors = {}
        self.attention_saved = False

    def scoped(self, label, fn):
        def wrapped(*args, **kwargs):
            if not self.active:
                return fn(*args, **kwargs)
            with record_function('audit/' + label):
                return fn(*args, **kwargs)
        return wrapped

    def patch(self, obj, attr, label):
        replacement = self.scoped(label, getattr(obj, attr))
        if isinstance(inspect.getattr_static(obj, attr), staticmethod):
            replacement = staticmethod(replacement)
        self.stack.enter_context(mock.patch.object(obj, attr, replacement))

    def capture(self, name, fn, meta, *args, **kwargs):
        torch.cuda.synchronize()
        self.active = True
        self.tensors = {}
        if self.nsys:
            torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.nvtx.range(name):
                result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            details = {}
        else:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True, profile_memory=True, with_stack=False) as prof:
                with record_function('audit/' + name):
                    result = fn(*args, **kwargs)
                torch.cuda.synchronize()
            self.active = False
            path = self.out / (name + '.json')
            prof.export_chrome_trace(str(path))
            details = summarize_trace(path)
            # Memory is allocator accounting, NOT measured physical DRAM bytes.
            op_rows = []
            for e in prof.key_averages(group_by_input_shape=True):
                op_rows.append(dict(op=e.key, shapes=str(e.input_shapes), count=e.count,
                                    self_cpu_us=e.self_cpu_time_total,
                                    self_gpu_us=e.self_device_time_total,
                                    self_device_alloc_bytes=e.self_device_memory_usage))
            write_csv(self.out / (name + '_operators.csv'), op_rows)
        self.active = False
        tensor_info = {}
        for key, tensor in self.tensors.items():
            tensor_info[key] = tensor_meta(tensor)
            if tensor is not None and tensor.ndim == 3 and tensor.shape[1] % 300 == 0:
                frame = tensor.reshape(tensor.shape[0], -1, 300, tensor.shape[-1])
                tensor_info[key]['identical_within_each_frame'] = bool(torch.equal(frame, frame[:, :, :1].expand_as(frame)))
        self.tensors.clear()
        self.done[name] = dict(metadata=meta, tensors=tensor_info, **details)
        bench.dump(self.out / 'partial.json', self.done)
        print('CAPTURED', name, json.dumps({k:details.get(k) for k in ('kernel_count','gpu_sum_ms','gpu_first_to_last_ms')}), flush=True)
        return result

    def __enter__(self):
        for obj, attr, label in (
            (mw.RMSNorm, 'forward', 'rmsnorm'), (mw, 'modulate', 'adaln_pointwise'),
            (mw.SwiGLUFFN, 'forward', 'ffn'),
            (mw.BlockModulation, 'forward', 'block_modulation'),
            (mw.MiniWorldBlock, 'forward', 'block_residual_and_other'),
            (mw.VisionRotaryEmbeddingFast3D, 'forward', 'rope_apply'),
            (mw.VisionRotaryEmbeddingFast3D, '_build_freqs', 'rope_coefficients'),
            (mw.VisionRotaryEmbeddingFast3D, 'rope_shift_time', 'rope_rebase'),
            (mw, '_chunkwise_flash_attention', 'attention_cast_cat_and_flash'),
            (ResidentKVCache, 'write_layer', 'kv_current_write'),
            (mw.MiniWorldModel, '_build_conditioning', 'conditioning'),
            (mw.FinalLayer, 'forward', 'final_layer'),
            (mw.Attention, 'forward', 'attention_layout_and_other'),
        ):
            self.patch(obj, attr, label)
        for name, module in self.model.net.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Group identical layers while retaining their surrounding scope.
                category = ('modulation_gemm' if '.modulation.' in name else
                            'qkv_gemm' if name.endswith('attn.qkv') else
                            'attention_output_gemm' if name.endswith('attn.proj') else
                            'ffn_gate_up_gemm' if name.endswith('mlp.w12') else
                            'ffn_down_gemm' if name.endswith('mlp.w3') else 'conditioning_final_gemm')
                self.patch(module, 'forward', category)

        def block_inputs(module, args):
            if self.active and not self.tensors:
                for key, value in zip(('residual_x', 'emb', 'shared_mod', 'pose_mod'), args):
                    self.tensors[key] = value
        self.stack.callback(self.model.net.blocks[0].register_forward_pre_hook(block_inputs).remove)
        original_flash = mw._chunkwise_flash_attention
        def flash(q, k, v, chunk_tokens):
            if self.active and not self.attention_saved and q.shape[-2] == 4800 and k.shape[-2] == 19200:
                # CPU saving occurs AFTER this capture to keep copies out of its
                # measured kernels. Cloning is unnecessary until capture ends.
                self.snapshot = (q, k, v, chunk_tokens)
                self.attention_saved = True
            return original_flash(q, k, v, chunk_tokens)
        self.stack.enter_context(mock.patch.object(mw, '_chunkwise_flash_attention', flash))

        original_forward = self.model.net.forward_with_cache
        def forward(*args, **kwargs):
            cache = kwargs.get('resident_cache')
            h = cache.length // 300 if cache else 0
            c = args[0].shape[2]
            role = 'commit' if kwargs.get('return_kv') else 'denoise'
            self.counts[(h, c, role)] += 1
            idx = self.forward_id
            self.forward_id += 1
            target = (h, c, role) in ((0,16,'denoise'), (24,16,'denoise'), (48,16,'denoise'), (48,2,'commit'), (48,15,'denoise'))
            name = f'{role}_history{h}_current{c}'
            if self.nsys:
                target = (h,c,role) == (48,16,'denoise')
            if target and name not in self.done:
                result = self.capture(name, original_forward, dict(forward_id=idx, committed_latents_including_sink=h,
                                      history_chunks_excluding_sink=max(0,h-1)/2, current_latents=c, role=role), *args, **kwargs)
                if hasattr(self, 'snapshot'):
                    q,k,v,chunk = self.snapshot
                    torch.save(dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),chunk_tokens=chunk), self.out / 'attention_snapshot.pt')
                    del self.snapshot
                return result
            return original_forward(*args, **kwargs)
        self.stack.enter_context(mock.patch.object(self.model.net, 'forward_with_cache', forward))
        original_evict = ResidentKVCache.evict_and_shift
        def evict(cache, *args, **kwargs):
            name = 'eviction_history50_to48'
            # Count all bank evictions, even after the sample is captured.
            # Otherwise the third-sample selection count looks like a rollout
            # total and can produce incorrect extrapolated costs.
            self.counts[('evictions',)] += 1
            if not self.nsys and name not in self.done:
                # Skip the allocation-heavy first eviction, profile the third
                # bank eviction once reusable compaction scratch is allocated.
                if self.counts[('evictions',)] == 3:
                    return self.capture(name, original_evict, dict(old_tokens=cache.length), cache, *args, **kwargs)
            return original_evict(cache, *args, **kwargs)
        self.stack.enter_context(mock.patch.object(ResidentKVCache, 'evict_and_shift', evict))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--nsys', action='store_true')
    p.add_argument('--attention-snapshot', type=Path)
    cli = p.parse_args()
    cli.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if cli.attention_snapshot:
        data = torch.load(cli.attention_snapshot, weights_only=True)
        q,k,v = [data[key].cuda() for key in ('q','k','v')]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(3):
                mw._chunkwise_flash_attention(q,k,v,data['chunk_tokens'])
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            mw._chunkwise_flash_attention(q,k,v,data['chunk_tokens'])
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        return
    if (cli.out / 'summary.json').exists():
        raise FileExistsError('Use a fresh audit directory')
    model, _ = bench.load_model()
    data = bench.load_prepared()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(3):
            model.net.forward_with_cache(data['noise'][:,:,:16], torch.ones(1,16,device='cuda'), data['cond_seq'][:,:16],chunk_size=2)
        with Audit(model,cli.out,cli.nsys) as audit:
            result = model.generate_eval_latents_streaming(
                data['latents'],data['cond_seq'],total_len=125,history_len=1,
                max_cache_chunks=24,inflight_chunks=8,sink_frames=1,noise=data['noise'],
                use_resident_kv_cache=True)
    torch.cuda.synchronize()
    reference = torch.load('/data2/jian/output/Miniworld/DROID_long_horizon/resident_kv/episode_002352/resident_latents.pt',map_location='cpu',weights_only=True)
    props = torch.cuda.get_device_properties(0)
    report = dict(episode='002352',seed=bench.SEED,gpu=props.name,sm=f'{props.major}{props.minor}',
                  multiprocessors=props.multi_processor_count,torch=torch.__version__,
                  git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=bench.ROOT,text=True).strip(),
                  production_sha256={f:hashlib.sha256((bench.ROOT/'miniworld'/f).read_bytes()).hexdigest()
                                     for f in ('miniworld.py','denoiser.py','kv_cache.py')},
                  settings=dict(chunk_latents=2,sink_latents=1,cache_including_sink_latents=48,inflight_chunks=8),
                  caveat='Diagnostic instrumented single-call samples; no VAE. Kernel sum is not E2E latency or measured DRAM traffic.',
                  latent_bitwise_equal=bool(torch.equal(result.cpu(),reference)),
                  calls=[dict(history_current_role=list(k),count=v) for k,v in audit.counts.items()],captures=audit.done)
    bench.dump(cli.out/'summary.json',report)
    print('DONE parity=',report['latent_bitwise_equal'],flush=True)
    assert report['latent_bitwise_equal']


if __name__ == '__main__':
    main()
