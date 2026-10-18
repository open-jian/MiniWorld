"""Length extrapolation experiment. Official weights/core; runtime guard override documented."""
import os,sys,json,time,argparse,subprocess,concurrent.futures,inspect,textwrap,types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark_500_frames import Recorder,save_video,dump
from miniworld import sample
from miniworld.data.droid import LeRobotActionDataset
from miniworld.vae.codec import load_wan22_vae,vae_encode,StreamingVAEDecoder
from miniworld.conditioning.actions import ConditioningConfig,build_cond_seq_for_batch
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/generated_history_5episodes_3seeds_window0_24')
C=Path('/data2/jian/checkpoints/miniworld')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--episode',type=int,required=True);parser.add_argument('--worker',type=int,default=0);parser.add_argument('--seed-base',type=int,required=True);opt=parser.parse_args();ep=opt.episode;seed=opt.seed_base+ep
 artifact=lambda mode,name:R/f'episode_{ep:06d}_seed{seed}_history{mode:03d}chunks_{name}'
 phase=lambda value:dump(R/f'episode_{ep:06d}_seed{seed}_worker{opt.worker}_state.json',dict(episode=ep,phase=value,time=time.time(),pid=os.getpid()))
 torch.set_num_threads(4)
 entry=next(x for x in json.loads((R/'selection.json').read_text())['videos'] if x['episode_index']==ep)
 sys.argv=[sys.argv[0],'--dataset','droid','--checkpoint',str(C/'MiniWorld_1b_droid.pt'),'--vae_checkpoint',str(C/'Wan2.2_VAE.pth')]
 args=sample.parse_args();dataset=LeRobotActionDataset('/data2/jian/benchmark/droid',num_frames=1277,frame_interval=1,randomize=False,color_aug=False,max_keep=1)
 sample.resolve_conditioning(args,dataset)
 ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False);meta=ckpt.get('meta',{});weights=ckpt.get('ema_model',ckpt.get('model',ckpt))
 args.latent_frames=sample.resolve_latent_frames(weights,meta,args);assert args.latent_frames==64
 model=sample.build_denoiser(args).cuda().eval();sample.load_weights(weights,model);del ckpt,weights
 # Exact capacity semantics: sliding history N chunks PLUS unchanged sink.
 original_method=type(model).generate_eval_latents_streaming
 method_source=textwrap.dedent(inspect.getsource(original_method))
 old_line='max_cache_frames = max_cache_chunks * chunk_size'
 new_line='max_cache_frames = max_cache_chunks * chunk_size + max(0, int(sink_frames))'
 assert method_source.count(old_line)==1
 method_source=method_source.replace(old_line,new_line)
 namespace=dict(original_method.__wrapped__.__globals__)
 exec(compile(method_source,'<history_chunks_excluding_sink>','exec'),namespace)
 model.generate_eval_latents_streaming=types.MethodType(namespace['generate_eval_latents_streaming'],model)
 dump(R/f'episode_{ep:06d}_seed{seed}_worker{opt.worker}_capacity_adapter.json',dict(old=old_line,new=new_line,changes=1,core_source_modified=False))
 vae=load_wan22_vae(args)
 # Audit clean generated-history commits; pass original x through unchanged.
 original_forward=model.net.forward_with_cache
 history_audit={'cursor':0,'pending':None,'records':[]}
 def audited_forward(*inputs,**kw):
  if kw.get('return_kv',False):
   x,t,c=inputs;branch='uncond' if kw.get('cond_drop') is not None else 'cond'
   assert torch.count_nonzero(t).item()==0
   if branch=='cond':
    start=history_audit['cursor'];end=start+x.shape[2]
    assert end<=latents.shape[2]
    history_audit['pending']=(start,end)
   else:
    assert history_audit['pending'] is not None
    start,end=history_audit['pending']
   history_audit['records'].append(dict(branch=branch,start_latent=start,end_latent_exclusive=end,cache_position=kw.get('current_position_offset',0),source='generated latent (unchanged sampler input)'))
   result=original_forward(x,t,c,**kw)
   if branch=='uncond':
    history_audit['cursor']=end;history_audit['pending']=None
   return result
  return original_forward(*inputs,**kw)
 model.net.forward_with_cache=audited_forward
 rec=Recorder(model)
 evict_original=model._evict_and_shift_cache
 sink_checks=[]
 def checked_evict(*args,**kw):
  assert kw['sink_frames']==1
  out=evict_original(*args,**kw)
  for before,after in zip(args[0],out):
   if before is not None:
    for x,y in zip(before,after):sink_checks.append((x[...,:300,:]==y[...,:300,:]).all())
  return out
 model._evict_and_shift_cache=checked_evict
 dump(R/f'episode_{ep:06d}_seed{seed}_worker{opt.worker}_environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],checkpoint_training_latent_frames=64,max_history_window_chunks=24,total_capacity_latent_max=65,extension='History window excludes sink and query; runtime sampler capacity expression adds fixed sink. Training guard raised only as needed. Core files and weights unchanged.',args=vars(args),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()))
 dataset._parquet_path=lambda _:Path(entry['parquet'])
 video,ids=dataset._decode_clip(Path(entry['source']));actions=dataset._load_actions(ep,ids).unsqueeze(0).cuda()
 gt=((video+1)*127.5).round().clamp(0,255).to(torch.uint8).numpy()
 np.save(R/f'episode_{ep:06d}_seed{seed}_ground_truth_rgb.npy',gt)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  latents=vae_encode(vae,video.unsqueeze(0).cuda().permute(0,4,1,2,3).contiguous());assert latents.shape[2]==320
  cond=build_cond_seq_for_batch(cfg=ConditioningConfig(False,True,15),poses=None,actions=actions,t_latent=320,h_lat=15,w_lat=20)
  latents[:,:,1:]=0 # Only the first GT latent is available to the generation sampler.
  phase('warmup');history_audit.update(cursor=0,pending=None,records=[])
  model.generate_eval_latents_streaming(latents[:,:,:64],cond[:,:64],total_len=64,history_len=1,max_cache_chunks=23,inflight_chunks=8,sink_frames=1,stream_decoder=StreamingVAEDecoder(vae),collect_stream_timing=False)
  torch.cuda.synchronize();rec.clear();torch.cuda.empty_cache()
  import lpips
  from skimage.metrics import structural_similarity
  perceptual=lpips.LPIPS(net='alex',version='0.1').cuda().float().eval()
  def ssim(pair):return structural_similarity(pair[0],pair[1],data_range=255,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11)
  for window in reversed(list(range(0,25))):
   if artifact(window,'complete.json').exists():continue
   claim=R/'.claims'/f'{ep:06d}_{seed}_{window:03d}.json'
   try:
    with claim.open('x') as fp:json.dump(dict(pid=os.getpid(),worker=opt.worker,window=window),fp)
   except FileExistsError:continue
   model.trained_num_frames=max(64,17+2*window) # Total: sink 1 + sliding history 2*N + query 16.
   rec.clear();decoded=[];sink_checks.clear();history_audit.update(cursor=0,pending=None,records=[])
   class Decoder(StreamingVAEDecoder):
    def step(self,z):
     out=super().step(z)
     latest=rec.rows[-1]
     decoded.append(dict(chunk_index=len(decoded),step=rec.step,cache_frames=latest['cache_frames'],query_frames=latest['current_frames'],active_window=latest['cache_frames']+latest['current_frames'],cache_bytes_both=2*latest['cache_bytes'],rgb_count=int(out.shape[2])))
     return out
   torch.manual_seed(seed);torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
   phase(f'generation_w{window:03d}');start_unix=time.time();start=time.perf_counter()
   pred_latents,pred=model.generate_eval_latents_streaming(latents,cond,total_len=320,history_len=1,max_cache_chunks=window,inflight_chunks=8,sink_frames=1,stream_decoder=Decoder(vae),collect_stream_timing=True)
   torch.cuda.synchronize();seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();rec.finish()
   assert sink_checks and bool(torch.stack(sink_checks).all().item()), 'Sink changed during eviction'
   assert all(row['cache_frames']<=1+2*window for row in rec.rows if row['role']=='denoise')
   assert any(row['cache_frames']==1+2*window for row in rec.rows if row['role']=='denoise')
   for event in rec.evictions:
    assert event['after']['cache_tokens']==(1+2*window)*300
   phase(f'writing_history{window:03d}chunks')
   rgb=((pred[0].permute(1,2,3,0).clamp(-1,1)+1)*127.5).to(torch.uint8).cpu().numpy();assert rgb.shape==gt.shape
   np.save(artifact(window,'prediction_rgb.npy'),rgb)
   save_video(artifact(window,'prediction.mp4'),rgb)
   save_video(artifact(window,'comparison.mp4'),np.concatenate([gt,rgb],axis=2))
   dump(artifact(window,'history_audit_commits.json'),history_audit['records']);dump(artifact(window,'forward_calls.json'),rec.rows);dump(artifact(window,'evictions.json'),rec.evictions);dump(artifact(window,'stream_timing.json'),model.last_eval_meta)
   frame_end=-1
   for row in decoded:
    row['rgb_start']=frame_end+1;frame_end+=row['rgb_count'];row['rgb_end']=frame_end
   dump(artifact(window,'decoded_chunks.json'),decoded)
   dump(artifact(window,'summary.json'),dict(episode_index=ep,seed=seed,history_window_chunks=window,window_definition="sliding history only; excludes sink and in-flight query",total_capacity_latent=17+2*window,cache_capacity_latent=1+2*window,query_latent=16,sink_frames=1,effective_sampling_steps=40,checkpoint_training_latent=64,processed_rgb_frames=1277,latent_frames=320,generation_sec=seconds,generation_start_unix=start_unix,generated_rgb_fps=1276/seconds,peak_allocated_bytes=peak,peak_reserved_bytes=reserved,max_kv_bytes_both=max(r['cache_bytes'] for r in rec.rows)*2,history_source='Generated latent committed unchanged at t=0 in both CFG branches; no GT history replacement',physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
   print('DONE GENERATION',ep,window,seconds,flush=True)
   del pred_latents,pred,rgb;rec.clear();torch.cuda.empty_cache()
   phase(f'quality_w{window:03d}')
   with concurrent.futures.ThreadPoolExecutor(4) as pool,torch.inference_mode(),torch.autocast('cuda',enabled=False):
    rgb=np.load(artifact(window,'prediction_rgb.npy'),mmap_mode='r')
    mse=np.mean((gt.astype(np.float64)-rgb.astype(np.float64))**2,axis=(1,2,3))
    ss=list(pool.map(ssim,zip(gt,rgb)));lp=[]
    for start in range(0,len(gt),32):
     a=torch.from_numpy(gt[start:start+32]).permute(0,3,1,2).cuda().float()/127.5-1
     b=torch.tensor(rgb[start:start+32].copy()).permute(0,3,1,2).cuda().float()/127.5-1
     lp.extend(perceptual(a,b).flatten().cpu().tolist())
    df=pd.DataFrame(dict(frame=np.arange(len(gt)),psnr_db=10*np.log10(255**2/np.maximum(mse,1e-12)),ssim=ss,lpips_alex=lp))
    df.to_csv(artifact(window,'quality_per_frame.csv'),index=False)
   dump(artifact(window,'complete.json'),dict(episode_index=ep,seed=seed,window=window,status='complete',time=time.time()))
   print('DONE QUALITY',ep,window,flush=True)
 dump(R/f'episode_{ep:06d}_seed{seed}_complete.json',dict(episode_index=ep,seed=seed,status='complete',windows=list(range(0,25))))
 phase('complete')
if __name__=='__main__':main()
