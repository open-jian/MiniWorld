"""Profile the unmodified official sampler; adapters only for paths/checkpoint/video IO."""
import os, sys, json, time, argparse, traceback, subprocess
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from miniworld import sample
from miniworld.vae.codec import load_wan22_vae, vae_encode, StreamingVAEDecoder
from miniworld.conditioning.actions import ConditioningConfig, build_cond_seq_for_batch
from miniworld.data.droid import LeRobotActionDataset

ROOT=Path('/data2/jian/output/Miniworld/DROID_long_horizon')
DATA=Path('/data2/jian/data/DROID_long_horizon')
CKPT=Path('/data2/jian/checkpoints/miniworld')

def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,ensure_ascii=False))

def cache_info(cache):
 pairs=[p for p in (cache or []) if p is not None]
 return dict(cache_bytes=sum(t.numel()*t.element_size() for p in pairs for t in p),
  cache_tokens=pairs[0][0].shape[-2] if pairs else 0,
  cache_shapes=[list(t.shape) for t in pairs[0]] if pairs else [],
  cache_dtypes=[str(t.dtype) for t in pairs[0]] if pairs else [])

class Recorder:
 def __init__(self,model):
  self.rows=[];self.evictions=[];self.decodes=[]; self.step=-1
  orig=model.net.forward_with_cache
  def forward(*args,**kw):
   commit=kw.get('return_kv',False);branch='uncond' if kw.get('cond_drop') is not None else 'cond'
   if not commit and branch=='cond':self.step+=1
   row=dict(call=len(self.rows),step=self.step,role='commit' if commit else 'denoise',branch=branch,
    current_frames=args[0].shape[2],cache_frames=kw.get('current_position_offset',0),
    allocated_before=torch.cuda.memory_allocated(),**cache_info(kw.get('past_kv_list')))
   a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   a.record();out=orig(*args,**kw);b.record()
   row.update(allocated_after=torch.cuda.memory_allocated(),events=(a,b))
   self.rows.append(row)
   return out
  model.net.forward_with_cache=forward
  orig_evict=model._evict_and_shift_cache
  def evict(*args,**kw):
   before=cache_info(args[0]);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   a.record();out=orig_evict(*args,**kw);b.record()
   self.evictions.append(dict(step=self.step,drop_frames=args[1],before=before,after=cache_info(out),events=(a,b)))
   return out
  model._evict_and_shift_cache=evict
 def clear(self):self.rows=[];self.evictions=[];self.decodes=[];self.step=-1
 def finish(self):
  torch.cuda.synchronize()
  for row in self.rows+self.evictions:
   a,b=row.pop('events');row['gpu_ms']=a.elapsed_time(b)


def save_video(path,frames):
 import av
 with av.open(str(path),'w') as container:
  stream=container.add_stream('libx264',rate=15);stream.width=frames.shape[2];stream.height=frames.shape[1]
  stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast'}
  for frame in frames:
   for packet in stream.encode(av.VideoFrame.from_ndarray(frame,format='rgb24')):container.mux(packet)
  for packet in stream.encode():container.mux(packet)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,default=0);parser.add_argument('--workers',type=int,default=8);parser.add_argument('--limit',type=int,default=0);parser.add_argument('--smoke',action='store_true');parser.add_argument('--queue',action='store_true')
 opt=parser.parse_args(); torch.set_num_threads(4)
 sys.argv=[sys.argv[0],'--dataset','droid','--checkpoint',str(CKPT/'MiniWorld_1b_droid.pt'),'--vae_checkpoint',str(CKPT/'Wan2.2_VAE.pth')]
 args=sample.parse_args()
 dataset=LeRobotActionDataset('/data2/jian/benchmark/droid',num_frames=497,frame_interval=1,randomize=False,color_aug=False,max_keep=1)
 sample.resolve_conditioning(args,dataset)
 weights=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
 meta=weights.get('meta',{}); weights=weights.get('ema_model',weights.get('model',weights))
 args.latent_frames=sample.resolve_latent_frames(weights,meta,args)
 model=sample.build_denoiser(args).cuda().eval();sample.load_weights(weights,model);del weights
 vae=load_wan22_vae(args);rec=Recorder(model)
 dump(ROOT/f'worker_{opt.worker}_environment.json',dict(args=vars(args),torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,parameter_dtype=str(next(model.parameters()).dtype),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),worker=opt.worker,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),adapter=str(Path(__file__).resolve())))
 entries=json.loads((DATA/'selection.json').read_text())['videos']
 # Interleave lengths, then distribute; each GPU gets multiple length groups.
 entries=sorted(entries,key=lambda x:(next(i for i,e in enumerate([r for r in entries if r['target_frames']==x['target_frames']]) if e['episode_index']==x['episode_index']),x['target_frames']))
 entries=entries if opt.queue else entries[opt.worker::opt.workers]
 if opt.limit:entries=entries[:opt.limit]
 warmed=False
 for index,item in enumerate(entries):
  ep=item['episode_index'];dest=ROOT/('smoke' if opt.smoke else 'episodes')/f'episode_{ep:06d}'
  if (dest/'summary.json').exists():continue
  claim=ROOT/'claims'/f'{ep}.json'
  if opt.queue:
   claim.parent.mkdir(exist_ok=True)
   try:
    with claim.open('x') as fp:json.dump(dict(pid=os.getpid(),episode_index=ep,worker=opt.worker),fp)
   except FileExistsError:continue
  dest.mkdir(parents=True,exist_ok=True)
  n=(item['actual_frames']-1)//4*4+1
  if opt.smoke:n=257
  dataset.num_frames=n;dataset._parquet_path=lambda _,p=item['parquet']:Path(p)
  start=time.perf_counter();video,ids=dataset._decode_clip(Path(item['source']));actions=dataset._load_actions(ep,ids).unsqueeze(0).cuda()
  load_sec=time.perf_counter()-start
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   video_gpu=video.unsqueeze(0).cuda().permute(0,4,1,2,3).contiguous();torch.cuda.synchronize();start=time.perf_counter()
   latents=vae_encode(vae,video_gpu);torch.cuda.synchronize();encode_sec=time.perf_counter()-start;del video_gpu
   cond=build_cond_seq_for_batch(cfg=ConditioningConfig(False,True,args.pose_enc_freq),poses=None,actions=actions,t_latent=latents.shape[2],h_lat=latents.shape[3],w_lat=latents.shape[4])
   if not warmed:
    print('[Warmup] 64 latent frames, default cache/window',flush=True)
    model.generate_eval_latents_streaming(latents[:,:,:64],cond[:,:64],total_len=64,history_len=1,max_cache_chunks=24,inflight_chunks=8,sink_frames=1,stream_decoder=StreamingVAEDecoder(vae),collect_stream_timing=False)
    torch.cuda.synchronize();rec.clear();torch.cuda.empty_cache();warmed=True
   generation_start_unix=time.time();rec.clear();torch.manual_seed(20261016+ep);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
   pred_latents,pred_rgb=model.generate_eval_latents_streaming(latents,cond,total_len=latents.shape[2],history_len=1,max_cache_chunks=24,inflight_chunks=8,sink_frames=1,stream_decoder=StreamingVAEDecoder(vae),collect_stream_timing=True)
   torch.cuda.synchronize();gen_sec=time.perf_counter()-start
   peak_alloc=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved();rec.finish()
  dump(dest/'forward_calls.json',rec.rows);dump(dest/'cache_evictions.json',rec.evictions);dump(dest/'stream_timing.json',model.last_eval_meta)
  start=time.perf_counter();rgb=((pred_rgb[0].permute(1,2,3,0).clamp(-1,1)+1)*127.5).to(torch.uint8).cpu().numpy()
  save_video(dest/'prediction.mp4',rgb)
  gt=((video.clamp(-1,1)+1)*127.5).to(torch.uint8).numpy()
  save_video(dest/'comparison_gt_left_prediction_right.mp4',np.concatenate([gt,rgb],axis=2))
  mse=np.mean((rgb.astype(np.float32)-gt.astype(np.float32))**2,axis=(1,2,3));psnr=10*np.log10(255**2/np.maximum(mse,1e-10))
  np.savetxt(dest/'per_frame_psnr.csv',np.column_stack((np.arange(n),psnr)),delimiter=',',header='frame,psnr_db',comments='')
  summary=dict(**item,processed_rgb_frames=n,latent_frames=latents.shape[2],dropped_tail_frames=item['actual_frames']-n,worker=opt.worker,generation_start_unix=generation_start_unix,generation_end_unix=generation_start_unix+gen_sec,seed=20261016+ep,load_sec=load_sec,vae_encode_sec=encode_sec,generation_sec=gen_sec,output_write_and_metrics_sec=time.perf_counter()-start,generated_rgb_fps=(n-1)/gen_sec,peak_allocated_bytes=peak_alloc,peak_reserved_bytes=peak_reserved,max_cache_bytes_per_branch=max(r['cache_bytes'] for r in rec.rows),psnr_mean_excluding_seed_db=float(psnr[1:].mean()))
  dump(dest/'summary.json',summary); print('[DONE] '+json.dumps(summary),flush=True)
  del video,actions,latents,cond,pred_latents,pred_rgb,rgb,gt;rec.clear();torch.cuda.empty_cache()
  if opt.queue:claim.unlink(missing_ok=True)

if __name__=='__main__':main()
