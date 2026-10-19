"""Bounded observational KV-redundancy profile on an unchanged MiniWorld rollout.

All diagnostics use the exact BF16 Q/K/V presented to the production attention
kernel, and FP32 reductions. Production outputs are never modified. JSON lists:
step_records: same-frame K/V descriptors and K-only/V-only/KV stale substitutions;
layer_records: same-index cross-layer descriptors and full-KV substitutions;
deletion_records: exact local-output effects of history-only deletion policies;
layer_matrices: descriptive 28-by-28 same-coordinate cosine matrices;
history_similarity: adjacent disjoint historical-frame-pair cosine distributions;
controls: sampled FP32 reference versus the unchanged production output.
"""
from __future__ import annotations

import time
PROCESS_START = time.monotonic()

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_INPUT = Path('/data2/jian/output/Miniworld/DROID_long_horizon/chunk_flash_attention/episode_002352')
DEFAULT_OUTPUT = Path('/home/jian/Documents/Codex/2026-10-19/hello/work/kv_redundancy')


class BudgetExpired(Exception):
    pass


class ProfileComplete(Exception):
    pass


def sampled_ids(start, end, height, width, device):
    """Same 4-by-4 stratum centers in each of the two query frames."""
    ys = ((torch.arange(4, device=device) + .5) * height / 4).long()
    xs = ((torch.arange(4, device=device) + .5) * width / 4).long()
    spatial = (ys[:, None] * width + xs[None, :]).flatten()
    ids = (torch.arange(start, end, height * width, device=device)[:, None] + spatial).flatten()
    return ids[ids < end]


def descriptor(reference, candidate):
    """[heads,tokens,channels] -> [heads,5]: cosine mean/p05, RRMSE, norm ratio, RMS."""
    reference, candidate = reference.float(), candidate.float()
    cos = F.cosine_similarity(reference, candidate, dim=-1, eps=1e-12)
    energy = reference.square().mean((-1, -2)).clamp_min(1e-20)
    return torch.stack((cos.mean(-1), torch.quantile(cos, .05, dim=-1),
                        ((candidate-reference).square().mean((-1,-2))/energy).sqrt(),
                        (candidate.square().mean((-1,-2))/energy).sqrt(), energy.sqrt()), -1)


def output_metrics(reference, candidate):
    reference, candidate = reference.float(), candidate.float()
    energy = reference.square().mean((-1, -2)).clamp_min(1e-20)
    cos = F.cosine_similarity(reference, candidate, dim=-1, eps=1e-12)
    return torch.stack((((candidate-reference).square().mean((-1,-2))/energy).sqrt(),
                        cos.mean(-1), torch.quantile(cos, .05, dim=-1)), -1)


DESCRIPTOR_FIELDS = ('cos_mean', 'cos_p05', 'relative_rmse', 'norm_ratio', 'reference_rms')
OUTPUT_FIELDS = ('relative_rmse', 'cos_mean', 'cos_p05')


def row_dict(fields, row):
    return dict(zip(fields, row))


def columns(fields, rows, offset=0):
    return {field:[row[offset+i] for row in rows] for i,field in enumerate(fields)}


def deletion_masks(k, v, n_past, tpf, sink_frames=1, protect_recent_frames=4):
    """History-only keep masks [H,Npast]. Similarity pairs exclude sink/recent."""
    heads, _, dim = k.shape
    frames = n_past // tpf
    assert n_past % tpf == 0
    masks = {}
    frame_ids = torch.arange(n_past, device=k.device) // tpf
    for label, fraction in (('recent_075', .75), ('recent_050', .5),
                            ('recent_025', .25), ('recent_0125', .125)):
        # Fraction denotes the entire history budget including its pinned sink.
        keep_frames = max(sink_frames, math.ceil(frames * fraction))
        recent_frames = max(0, keep_frames - sink_frames)
        keep = (frame_ids < sink_frames) | (frame_ids >= frames - recent_frames)
        masks[label] = keep[None].expand(heads, -1)
    pair_start = torch.arange(sink_frames, max(sink_frames, frames-protect_recent_frames-1),
                              2, device=k.device)
    kk = k[:, :n_past].reshape(heads, frames, tpf, dim).float()
    vv = v[:, :n_past].reshape(heads, frames, tpf, dim).float()
    ck = F.cosine_similarity(kk[:, pair_start], kk[:, pair_start+1], dim=-1).mean(-1)
    cv = F.cosine_similarity(vv[:, pair_start], vv[:, pair_start+1], dim=-1).mean(-1)
    for label, threshold in (('similarity_095', .95), ('similarity_099', .99)):
        keep_frames = torch.ones(heads, frames, device=k.device, dtype=torch.bool)
        if len(pair_start):
            keep_frames[:, pair_start] = ~((ck >= threshold) & (cv >= threshold))
        masks[label] = keep_frames.repeat_interleave(tpf, -1)
    return masks, pair_start, ck, cv


class BranchTracker:
    """Explicit sampler-order tracking, independent of conditioning implementation."""
    def __init__(self, use_cfg=True):
        self.branches = ('cond', 'uncond') if use_cfg else ('cond',)
        self.counts = {False: 0, True: 0}
        self.committed_frames = 0
        self.denoise_steps = 0

    def begin(self, commit):
        count = self.counts[commit]
        branch = self.branches[count % len(self.branches)]
        self.counts[commit] += 1
        if not commit and branch == 'cond':
            self.denoise_steps += 1
        return branch

    def finish_commit(self, branch, frames):
        if branch == 'cond':
            self.committed_frames += frames


class KVProfiler:
    def __init__(self, output_dir, metadata, deadline, net):
        self.output_dir, self.metadata, self.deadline, self.net = output_dir, metadata, deadline, net
        self.payload = {'metadata': metadata, 'status': 'running', **{name: [] for name in (
            'step_records', 'layer_records', 'deletion_records', 'layer_matrices',
            'history_similarity', 'controls')}}
        self.tracker = BranchTracker(use_cfg=True)
        self.context = None
        self.layer = -1
        self.input_x = None
        self.completed = {'short': 0, 'full': 0}
        self.anchors = {}
        self.ring = {}
        self.previous_layers = {}
        self.matrix_samples = {}
        self.capture_states = []
        self.benchmark_probes = []
        self.probe_seconds = 0.0
        self.saved_counts = {name:0 for name in self.payload if name not in ('metadata','status')}

    def check(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExpired('Profile exceeded its process budget')

    def pre_hook(self, layer):
        def hook(module, args):
            self.layer = layer
            self.input_x = args[0] if self.context is not None else None
        return hook

    def base(self, head=None, chunk=None):
        c = self.context
        result = {key: c[key] for key in ('state','step','branch','outer_step','cache_frames','global_window_start')}
        result['layer'] = self.layer
        if head is not None:
            result['head'] = head
        else:
            result['heads'] = list(range(self.net.num_heads))
        if chunk is not None:
            result.update(noise_label=chunk['noise_label'], chunk_index=chunk['index'],
                          timestep=chunk['t'], current_global_frames=chunk['global_frames'])
        return result

    def probe(self, q, k, v, chunk_tokens, production_output):
        if self.context is None:
            return
        self.check()
        started = time.monotonic()
        c, layer = self.context, self.layer
        n_past = k.shape[-2] - q.shape[-2]
        assert n_past == c['cache_frames'] * 300
        assert q.shape[0] == 1
        with torch.autocast('cuda', enabled=False):
            # Match the casts done once per layer by _chunkwise_flash_attention.
            qq_all, kk_all, vv_all = q[0].to(torch.bfloat16), k[0].to(torch.bfloat16), v[0].to(torch.bfloat16)
            current_k, current_v = kk_all[:, n_past:].clone(), vv_all[:, n_past:].clone()
            key = (c['branch'], layer)
            previous_steps = self.ring.setdefault(key, [])
            spatial = sampled_ids(0, 300, 15, 20, q.device)
            # Small same-coordinate sample at all frame locations for layer matrix.
            if c['step'] in (0,4):
                history_ids = (torch.arange(0,n_past,300,device=q.device)[:,None] + spatial).flatten()
                current_ids = (torch.arange(n_past,k.shape[-2],300,device=q.device)[:,None] + spatial).flatten()
                self.matrix_samples[layer] = {
                    'history_k': kk_all[:,history_ids].float(), 'history_v': vv_all[:,history_ids].float(),
                    'current_k': kk_all[:,current_ids].float(), 'current_v': vv_all[:,current_ids].float(),
                }
                masks, pairs, pair_k, pair_v = deletion_masks(kk_all, vv_all, n_past, 300)
                pair_cpu = torch.stack((pair_k,pair_v),-1).cpu().tolist()
                pair_ids = pairs.cpu().tolist()
                # Absolute history frames are contiguous at these pre-eviction anchors.
                self.payload['history_similarity'].append({**self.base(),
                    'frame_pairs': [[int(p),int(p+1)] for p in pair_ids],
                    'cos_k': [[a[0] for a in row] for row in pair_cpu],
                    'cos_v': [[a[1] for a in row] for row in pair_cpu],
                    'pair_frame_coordinate': 'cache-relative; first48-frame capture precedes first eviction'})
            else:
                masks = {}
            kf, vf = kk_all.float(), vv_all.float()
            for chunk in c['chunks']:
                self.check()
                start = chunk['index'] * chunk_tokens
                end = min(start + chunk_tokens, q.shape[-2])
                ids = sampled_ids(start,end,15,20,q.device)
                legal_end = n_past + end
                qf = qq_all[:,ids].float()
                scores = (qf @ kf[:,:legal_end].transpose(-1,-2)) * q.shape[-1] ** -.5
                probs = scores.softmax(-1)
                ref_fp32 = probs @ vf[:,:legal_end]
                reference = production_output[0,:,ids].float()
                control = output_metrics(reference,ref_fp32).cpu().tolist()
                self.payload['controls'].append({**self.base(chunk=chunk),
                    'fp32_vs_production':columns(OUTPUT_FIELDS,control), 'sampled_queries':len(ids)})
                for lag in (1,2,4):
                    if len(previous_steps) < lag:
                        continue
                    old_k, old_v = previous_steps[-lag]
                    dk = descriptor(current_k[:,ids],old_k[:,ids])
                    dv = descriptor(current_v[:,ids],old_v[:,ids])
                    stale_scores = scores.clone()
                    stale_scores[:,:,n_past:] = (qf @ old_k[:,:end].float().transpose(-1,-2)) * q.shape[-1] ** -.5
                    stale_probs = stale_scores.softmax(-1)
                    stale_v = torch.cat((vf[:,:n_past],old_v[:,:end].float()),dim=1)
                    outputs = (stale_probs @ vf[:,:legal_end], probs @ stale_v, stale_probs @ stale_v)
                    errors = torch.stack([output_metrics(reference,out) for out in outputs],1)
                    values = torch.cat((dk,dv,errors.flatten(1)),1).cpu().tolist()
                    self.payload['step_records'].append({**self.base(chunk=chunk), 'lag':lag,
                        'k':columns(DESCRIPTOR_FIELDS,values), 'v':columns(DESCRIPTOR_FIELDS,values,5),
                        'candidates':{mode:columns(OUTPUT_FIELDS,values,10+3*i)
                                      for i,mode in enumerate(('K','V','KV'))}})
                if c['step'] in (0,4):
                    for lag in (1,2,4):
                        if layer-lag not in self.previous_layers:
                            continue
                        old_k, old_v = self.previous_layers[layer-lag]
                        # Functional replacement is in actual same head/channel coordinates.
                        candidate = ((qf @ old_k[:,:legal_end].float().transpose(-1,-2)) * q.shape[-1]**-.5).softmax(-1) @ old_v[:,:legal_end].float()
                        h_ids = sampled_ids(0,n_past,15,20,q.device)
                        dkc = descriptor(current_k[:,ids],old_k[:,n_past+ids])
                        dvc = descriptor(current_v[:,ids],old_v[:,n_past+ids])
                        dkh = descriptor(kk_all[:,h_ids],old_k[:,h_ids])
                        dvh = descriptor(vv_all[:,h_ids],old_v[:,h_ids])
                        values = torch.cat((dkc,dvc,dkh,dvh,output_metrics(reference,candidate)),1).cpu().tolist()
                        self.payload['layer_records'].append({**self.base(chunk=chunk),'lag':lag,'source_layer':layer-lag,
                            'current_k':columns(DESCRIPTOR_FIELDS,values), 'current_v':columns(DESCRIPTOR_FIELDS,values,5),
                            'history_k':columns(DESCRIPTOR_FIELDS,values,10), 'history_v':columns(DESCRIPTOR_FIELDS,values,15),
                            'output':columns(OUTPUT_FIELDS,values,20)})
                    for mode,history_mask in masks.items():
                        mask = torch.cat((history_mask,torch.ones(q.shape[1],end,device=q.device,dtype=torch.bool)),dim=1)
                        candidate = scores.masked_fill(~mask[:,None],-torch.inf).softmax(-1) @ vf[:,:legal_end]
                        deleted = (~history_mask).sum(-1).float()
                        values = torch.cat((output_metrics(reference,candidate),
                                            (deleted/n_past)[:,None],(deleted/legal_end)[:,None]),1).cpu().tolist()
                        self.payload['deletion_records'].append({**self.base(chunk=chunk),'mode':mode,
                            'output':columns(OUTPUT_FIELDS,values),'deleted_history_fraction':[r[3] for r in values],
                            'removed_legal_key_fraction':[r[4] for r in values],'history_tokens':n_past,'legal_keys':legal_end})
            previous_steps.append((current_k,current_v))
            if len(previous_steps)>5:
                del previous_steps[0]
            if c['step'] in (0,4):
                self.previous_layers[layer] = (kk_all,vv_all)
                for old_layer in list(self.previous_layers):
                    if old_layer < layer-3:
                        del self.previous_layers[old_layer]
            if c['state']=='full' and c['step']==4 and c['branch']=='cond' and layer in (0,13,27):
                attn = self.net.blocks[layer].attn
                self.benchmark_probes.append({**self.base(), 'q':qq_all[None].cpu(),
                    'k_full':kk_all[None].cpu(),'v_full':vv_all[None].cpu(),
                    'input_x':self.input_x.detach().cpu(),'qkv_weight':attn.qkv.weight.detach().cpu(),
                    'qkv_bias':None if attn.qkv.bias is None else attn.qkv.bias.detach().cpu(),
                    'n_past':n_past,'chunk_tokens':chunk_tokens,'grid':[15,20],
                    'similarity_masks':{name:mask.cpu() for name,mask in masks.items() if name.startswith('similarity')},
                    'recent_masks':{name:mask.cpu() for name,mask in masks.items() if name.startswith('recent')}})
        self.probe_seconds += time.monotonic()-started

    def layer_matrices(self):
        for region in ('history','current'):
            for kind in ('k','v'):
                # Average cosine over matching heads and sampled token coordinates.
                rows = torch.stack([F.normalize(self.matrix_samples[i][f'{region}_{kind}'],dim=-1)
                                    for i in range(self.net.depth)])
                matrix = torch.einsum('lhtd,mhtd->lm',rows,rows)/(rows.shape[1]*rows.shape[2])
                self.payload['layer_matrices'].append({**{k:self.context[k] for k in ('state','step','branch')},
                    'region':region,'kind':kind,'matrix':matrix.cpu().tolist(),
                    'definition':'mean same-index cosine; independently learned layer bases, no alignment fitted'})

    def wrap_forward(self, original):
        def forward(x,t,y=None,**kwargs):
            self.check()
            commit = bool(kwargs.get('return_kv',False))
            branch = self.tracker.begin(commit)
            cache_frames = int(kwargs.get('current_position_offset',0))
            self.context = None
            state = {8:'short',48:'full'}.get(cache_frames)
            if not commit and state and self.completed[state]<5 and x.shape[2]==16:
                step = self.completed[state]
                anchor = (self.tracker.committed_frames,cache_frames,int(x.shape[2]))
                if state not in self.anchors:
                    assert branch=='cond' and step==0
                    self.anchors[state] = anchor
                    self.ring.clear()
                assert self.anchors[state]==anchor, f'Window moved during five-step capture: {state}, {anchor}'
                values = t[0].reshape(-1,kwargs['chunk_size']).mean(-1).float().cpu().tolist()
                self.context = dict(state=state,step=step,branch=branch,outer_step=self.tracker.denoise_steps-1,
                    cache_frames=cache_frames,global_window_start=self.tracker.committed_frames,
                    global_window_frames=list(range(self.tracker.committed_frames,self.tracker.committed_frames+x.shape[2])),
                    t_per_frame=t[0].float().cpu().tolist(),
                    chunks=[{'index':i,'noise_label':label,'t':values[i],
                             'global_frames':list(range(self.tracker.committed_frames+i*2,self.tracker.committed_frames+(i+1)*2))}
                            for label,i in (('low',0),('mid',4),('high',7))])
                self.previous_layers.clear()
                self.matrix_samples.clear()
                print(f'[KVProfile] capture {state} step={step} {branch}; global={anchor[0]}, cache={cache_frames}',flush=True)
            active = self.context
            try:
                result = original(x,t,y,**kwargs)
                if active is not None:
                    if active['step'] in (0,4):
                        self.layer_matrices()
                    self.capture_states.append(dict(active))
                    if branch==self.tracker.branches[-1]:
                        self.completed[state] += 1
                    if self.completed[state]==5:
                        self.ring.clear()
                    if len(self.benchmark_probes)==3 and not (self.output_dir/'benchmark_probes.pt').exists():
                        torch.save({'metadata':self.metadata,'probes':self.benchmark_probes},self.output_dir/'benchmark_probes.pt')
                    self.save('running')
                    if self.completed['full']==5:
                        raise ProfileComplete()
                if commit:
                    self.tracker.finish_commit(branch,int(x.shape[2]))
                return result
            finally:
                self.context = None
                self.input_x = None
                self.previous_layers.clear()
                self.matrix_samples.clear()
        return forward

    def save(self,status):
        self.metadata.update(elapsed_seconds=time.monotonic()-PROCESS_START,
            probe_seconds=self.probe_seconds,captured_forwards=self.capture_states,
            completed_steps=self.completed,conditional_denoise_forwards=self.tracker.denoise_steps,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None)
        self.payload['status'] = status
        counts = {name:len(self.payload[name]) for name in self.saved_counts}
        pending = {name:self.payload[name][self.saved_counts[name]:] for name in self.saved_counts
                   if counts[name]>self.saved_counts[name]}
        if pending:
            with (self.output_dir/'records.jsonl').open('a') as stream:
                stream.write(json.dumps(pending,separators=(',',':'),allow_nan=False)+'\n')
            self.saved_counts.update(counts)
        # Tiny checkpoints per forward; complete records are streamed once and
        # materialized as profile.json only on completion/error/budget exit.
        saved_payload = self.payload if status!='running' else {
            'metadata':self.metadata,'status':status,'record_counts':counts}
        tmp = self.output_dir/'profile.json.tmp'
        tmp.write_text(json.dumps(saved_payload,separators=(',',':'),allow_nan=False)+'\n')
        tmp.replace(self.output_dir/'profile.json')


def self_test():
    torch.manual_seed(13)
    q,k,v = torch.randn(3,5,8),torch.randn(3,40,8),torch.randn(3,40,8)
    masks,pairs,ck,cv = deletion_masks(k,v,32,4)
    assert pairs.tolist()==[1]
    for mask in masks.values():
        assert mask.shape==(3,32) and mask[:,:4].all()
    assert masks['recent_050'].sum(-1).eq(16).all()
    assert masks['similarity_095'][:,-16:].all()
    scores=q@k.transpose(-1,-2)/8**.5
    mask=torch.cat((masks['recent_050'],torch.ones(3,8,dtype=torch.bool)),1)
    actual=scores.masked_fill(~mask[:,None],-torch.inf).softmax(-1)@v
    expected=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None])
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(output_metrics(actual,actual)[:,0],torch.zeros(3))
    torch.testing.assert_close(descriptor(k,k)[:,0],torch.ones(3))
    # Adjacent exact duplicates are dropped only outside protected history.
    kd,vd=k.clone(),v.clone()
    kd[:,4:8]=kd[:,8:12]; vd[:,4:8]=vd[:,8:12]
    dm,_,_,_=deletion_masks(kd,vd,32,4)
    assert (~dm['similarity_099'][:,4:8]).all() and dm['similarity_099'][:,-16:].all()
    tracker=BranchTracker()
    assert [tracker.begin(False),tracker.begin(False)]==['cond','uncond']
    assert tracker.begin(True)=='cond'
    tracker.finish_commit('cond',2)
    assert tracker.begin(True)=='uncond'
    tracker.finish_commit('uncond',2)
    assert tracker.committed_frames==2 and tracker.begin(False)=='cond'
    assert sampled_ids(0,600,15,20,'cpu').shape==(32,)
    print('Self-test passed: masks, explicit branch bookkeeping, metrics, sampled positions, duplicate protection.')


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--budget-seconds',type=float,default=300)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if not 0<args.budget_seconds<=300:
        parser.error('--budget-seconds must be in (0,300]')
    return args


@torch.inference_mode()
def main():
    args=parse_args()
    if args.self_test:
        self_test(); return
    manifest=json.loads((args.input_dir/'experiment_manifest.json').read_text())
    seed=manifest['seed'] if args.seed is None else args.seed
    output_dir=args.output_dir or DEFAULT_OUTPUT/f'seed_{seed}'
    output_dir.mkdir(parents=True,exist_ok=True)
    if (output_dir/'profile.json').exists():
        raise FileExistsError(f'Use fresh output directory: {output_dir}')
    deadline=PROCESS_START+args.budget_seconds
    def expired(signum,frame):
        raise BudgetExpired('300-second process budget exhausted')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-time.monotonic()))
    metadata={
        'scope':'observational same-input local attention substitutions; not rollout quality or measured speedup',
        'episode':manifest['episode'],'seed':seed,'checkpoint':manifest['args']['checkpoint'],
        'input_dir':str(args.input_dir),'budget_seconds':args.budget_seconds,'headcast':False,
        'attention_input_dtype':'bfloat16','statistics_dtype':'float32','latent_grid':[15,20],
        'layers':28,'heads':12,'lags':[1,2,4],'history_already_cached':True,
        'state_selection':'first five consecutive conditional/unconditional denoise forwards with 8 then 48 cached frames; unchanged window asserted',
        'query_sampling':'32 deterministic spatial-stratum queries in each of chunks 0,4,7; chunks have differing diffusion times',
        'step_similarity':'same absolute frame, spatial position, layer/head, branch, and RoPE position; K is post-RoPE actual kernel input',
        'step_substitution':'replace only legal in-flight K/V prefix with an earlier-step copy; keep current Q and committed history unchanged',
        'layer_similarity':'same-index learned coordinates are descriptive; functional test directly substitutes earlier-layer full K/V',
        'deletion':'history-only drop; sink and all legal in-flight positions retained; similarity-pair policies additionally protect newest 4 history frames; no multiplicity correction or merge',
        'recent_fraction_definition':'budget is ceil(fraction*history_frames), includes sink; remaining budget assigned to newest frames',
        'similarity_deletion_definition':'disjoint adjacent historical frames; drop older member when both per-head mean spatial K cosine and V cosine reach threshold',
        'relative_rmse_definition':'sqrt(mean((candidate-production)^2)/mean(production^2)) over sampled queries and channels per head',
        'descriptor_fields':list(DESCRIPTOR_FIELDS),'output_fields':list(OUTPUT_FIELDS),
        'schema':'metric vectors are indexed by heads=[0..11]; history_similarity arrays are [head][frame_pair]',
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'source_sha256':{name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in ('miniworld/miniworld.py','miniworld/denoiser.py')},
        'profiler_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'gpu':torch.cuda.get_device_name(),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,
    }
    profiler=None; hooks=[]; original_attention=None; original_forward=None; status='error'
    try:
        torch.set_num_threads(4); torch.manual_seed(seed)
        from miniworld.denoiser import DenoiserConfig,build_denoiser_from_mode
        import miniworld.miniworld as model_module
        raw=torch.load(manifest['args']['checkpoint'],map_location='cpu',weights_only=True)
        weights=raw.get('ema_model',raw.get('model',raw))
        options=dict(manifest['args'])
        options.update(latent_size=(15,20),latent_frames=int(weights['net.feat_rope.freqs_cos'].shape[0]//300),
                       wm_use_checkpoint=False,headcast=False)
        model=build_denoiser_from_mode(DenoiserConfig(**options))
        model.load_state_dict(weights,strict=True)
        del raw,weights
        model=model.cuda().eval()
        assert model.cfg_scale>1, 'Explicit two-branch collector expects CFG enabled'
        inputs=torch.load(args.input_dir/'prepared_inputs.pt',map_location='cpu',weights_only=True)
        inputs={key:value.cuda() for key,value in inputs.items()}
        if seed!=manifest['seed']:
            generator=torch.Generator(device='cuda').manual_seed(seed)
            inputs['noise']=torch.randn(inputs['noise'].shape,device='cuda',dtype=inputs['noise'].dtype,generator=generator)
        torch.cuda.synchronize()
        metadata.update(load_seconds=time.monotonic()-PROCESS_START,sampler_args=options,
                        noise_source='saved original noise' if seed==manifest['seed'] else 'new CUDA Gaussian noise with explicit generator seed',
                        noise_sha256=hashlib.sha256(inputs['noise'].cpu().numpy().tobytes()).hexdigest())
        profiler=KVProfiler(output_dir,metadata,deadline,model.net)
        profiler.save('running')
        print(f'[KVProfile] loaded in {metadata["load_seconds"]:.2f}s; seed={seed}',flush=True)
        original_attention=model_module._chunkwise_flash_attention
        def attention(q,k,v,chunk_tokens):
            result=original_attention(q,k,v,chunk_tokens)
            profiler.probe(q,k,v,chunk_tokens,result)
            return result
        model_module._chunkwise_flash_attention=attention
        hooks=[block.attn.register_forward_pre_hook(profiler.pre_hook(i)) for i,block in enumerate(model.net.blocks)]
        original_forward=model.net.forward_with_cache
        model.net.forward_with_cache=profiler.wrap_forward(original_forward)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            model.generate_eval_latents_streaming(inputs['latents'],inputs['cond_seq'],
                total_len=inputs['latents'].shape[2],history_len=1,max_cache_chunks=24,inflight_chunks=8,
                sink_frames=1,noise=inputs['noise'],stream_decoder=None,collect_stream_timing=False)
        status='trajectory_finished'
    except ProfileComplete:
        status='complete'
    except BudgetExpired as exc:
        status='budget_exhausted'; metadata['error']=str(exc)
    except Exception:
        metadata['error']=traceback.format_exc()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if original_attention is not None:
            model_module._chunkwise_flash_attention=original_attention
        if original_forward is not None:
            model.net.forward_with_cache=original_forward
        for hook in hooks:
            hook.remove()
        if profiler is not None:
            profiler.save(status)
        else:
            (output_dir/'profile.json').write_text(json.dumps({'metadata':metadata,'status':status}))
        print(f'[KVProfile] {status}; elapsed {time.monotonic()-PROCESS_START:.2f}s',flush=True)


if __name__=='__main__':
    main()
