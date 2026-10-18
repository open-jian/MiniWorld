"""Small-query supplement: zero retained history; original sampler and weights."""
import os,sys,json,time,argparse,subprocess,concurrent.futures
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
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/window_016')
C=Path('/data2/jian/checkpoints/miniworld')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--episode',type=int,required=True);opt=parser.parse_args();ep=opt.episode
 artifact=lambda mode,name:R/f'episode_{ep:06d}_w{mode:03d}_{name}'
 phase=lambda value:dump(R/f'episode_{ep:06d}_state.json',dict(episode=ep,phase=value,time=time.time(),pid=os.getpid()))
 torch.set_num_threads(4)
 entry=next(x for x in json.loads((R/'selection.json').read_text())['videos'] if x['episode_index']==ep)
 sys.argv=[sys.argv[0],'--dataset','droid','--checkpoint',str(C/'MiniWorld_1b_droid.pt'),'--vae_checkpoint',str(C/'Wan2.2_VAE.pth')]
 args=sample.parse_args();dataset=LeRobotActionDataset('/data2/jian/benchmark/droid',num_frames=253,frame_interval=1,randomize=False,color_aug=False,max_keep=1)
 sample.resolve_conditioning(args,dataset)
 ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False);meta=ckpt.get('meta',{});weights=ckpt.get('ema_model',ckpt.get('model',ckpt))
 args.latent_frames=sample.resolve_latent_frames(weights,meta,args);assert args.latent_frames==64
 model=sample.build_denoiser(args).cuda().eval();sample.load_weights(weights,model);del ckpt,weights
 vae=load_wan22_vae(args);rec=Recorder(model)
 dump(R/f'episode_{ep:06d}_environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],checkpoint_training_latent_frames=64,permitted_experiment_window=16,extension='No guard override. Small-query experiment: no retained history or sink, query=window; effective denoising steps=5*window/2. Not fixed-query cache-growth experiment.',args=vars(args),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()))
 dataset._parquet_path=lambda _:Path(entry['parquet'])
 video,ids=dataset._decode_clip(Path(entry['source']));actions=dataset._load_actions(ep,ids).unsqueeze(0).cuda()
 gt=((video+1)*127.5).round().clamp(0,255).to(torch.uint8).numpy();np.save(R/f'episode_{ep:06d}_ground_truth_rgb.npy',gt)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  latents=vae_encode(vae,video.unsqueeze(0).cuda().permute(0,4,1,2,3).contiguous());assert latents.shape[2]==64
  cond=build_cond_seq_for_batch(cfg=ConditioningConfig(False,True,15),poses=None,actions=actions,t_latent=64,h_lat=15,w_lat=20)
  phase('warmup')
  model.generate_eval_latents_streaming(latents[:,:,:64],cond[:,:64],total_len=64,history_len=1,max_cache_chunks=0,inflight_chunks=8,sink_frames=0,stream_decoder=StreamingVAEDecoder(vae),collect_stream_timing=False)
  torch.cuda.synchronize();rec.clear();torch.cuda.empty_cache()
  for window in range(2,17,2):
   if artifact(window,'summary.json').exists():continue
   assert model.trained_num_frames==64
   rec.clear();decoded=[]
   class Decoder(StreamingVAEDecoder):
    def step(self,z):
     out=super().step(z)
     latest=rec.rows[-1]
     decoded.append(dict(chunk_index=len(decoded),step=rec.step,cache_frames=latest['cache_frames'],query_frames=latest['current_frames'],active_window=latest['cache_frames']+latest['current_frames'],cache_bytes_both=2*latest['cache_bytes'],rgb_count=int(out.shape[2])))
     return out
   torch.manual_seed(20261016+ep);torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
   phase(f'generation_w{window:03d}');start_unix=time.time();start=time.perf_counter()
   pred_latents,pred=model.generate_eval_latents_streaming(latents,cond,total_len=64,history_len=1,max_cache_chunks=0,inflight_chunks=window//2,sink_frames=0,stream_decoder=Decoder(vae),collect_stream_timing=True)
   torch.cuda.synchronize();seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();rec.finish()
   phase(f'writing_w{window:03d}')
   rgb=((pred[0].permute(1,2,3,0).clamp(-1,1)+1)*127.5).to(torch.uint8).cpu().numpy();assert rgb.shape==gt.shape
   np.save(artifact(window,'prediction_rgb.npy'),rgb)
   save_video(artifact(window,'prediction.mp4'),rgb)
   save_video(artifact(window,'comparison.mp4'),np.concatenate([gt,rgb],axis=2))
   dump(artifact(window,'forward_calls.json'),rec.rows);dump(artifact(window,'evictions.json'),rec.evictions);dump(artifact(window,'stream_timing.json'),model.last_eval_meta)
   frame_end=-1
   for row in decoded:
    row['rgb_start']=frame_end+1;frame_end+=row['rgb_count'];row['rgb_end']=frame_end
   dump(artifact(window,'decoded_chunks.json'),decoded)
   dump(artifact(window,'summary.json'),dict(episode_index=ep,window=window,cache_capacity_latent=0,query_latent=window,effective_sampling_steps=5*(window//2),sink_frames=0,checkpoint_training_latent=64,processed_rgb_frames=253,latent_frames=64,generation_sec=seconds,generation_start_unix=start_unix,generated_rgb_fps=252/seconds,peak_allocated_bytes=peak,peak_reserved_bytes=reserved,max_kv_bytes_both=max(r['cache_bytes'] for r in rec.rows)*2,seed=20261016+ep,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
   print('DONE GENERATION',ep,window,seconds,flush=True)
   del pred_latents,pred,rgb;rec.clear();torch.cuda.empty_cache()
 phase('quality')
 import lpips
 from skimage.metrics import structural_similarity
 perceptual=lpips.LPIPS(net='alex',version='0.1').cuda().eval()
 def ssim(pair):return structural_similarity(pair[0],pair[1],data_range=255,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11)
 metrics={};predictions={}
 with concurrent.futures.ThreadPoolExecutor(4) as pool,torch.inference_mode():
  for window in range(2,17,2):
   rgb=np.load(artifact(window,'prediction_rgb.npy'),mmap_mode='r');predictions[window]=rgb
   mse=np.mean((gt.astype(np.float64)-rgb.astype(np.float64))**2,axis=(1,2,3))
   ss=list(pool.map(ssim,zip(gt,rgb)));lp=[]
   for start in range(0,len(gt),32):
    a=torch.from_numpy(gt[start:start+32]).permute(0,3,1,2).cuda().float()/127.5-1
    b=torch.tensor(rgb[start:start+32].copy()).permute(0,3,1,2).cuda().float()/127.5-1
    lp.extend(perceptual(a,b).flatten().cpu().tolist())
   df=pd.DataFrame(dict(frame=np.arange(len(gt)),psnr_db=10*np.log10(255**2/np.maximum(mse,1e-12)),ssim=ss,lpips_alex=lp));metrics[window]=df
   df.to_csv(artifact(window,'quality_per_frame.csv'),index=False)
   for frame in [28,60,92,124,156,188,220,252]:
    sheet=np.concatenate([gt[frame],rgb[frame]],axis=1)
    Image.fromarray(sheet).save(artifact(window,f'rgb{frame:03d}_comparison.png'))
 dump(R/f'episode_{ep:06d}_complete.json',dict(episode_index=ep,status='complete',windows=list(range(2,17,2))))
 phase('complete');print('EPISODE COMPLETE',ep,flush=True)
if __name__=='__main__':main()
