"""Synthetic shape/memory feasibility check, NOT a rollout or quality experiment."""
import sys,json,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from miniworld import sample
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/window_512')
torch.set_num_threads(4)
sys.argv=[sys.argv[0],'--dataset','droid','--checkpoint','/data2/jian/checkpoints/miniworld/MiniWorld_1b_droid.pt','--vae_checkpoint','/data2/jian/checkpoints/miniworld/Wan2.2_VAE.pth']
a=sample.parse_args();a.use_action_cond=True;a.use_pose_cond=False;a.cond_dim=28;a.cond_per_token=False
weights=torch.load(a.checkpoint,map_location='cpu',weights_only=False);meta=weights.get('meta',{});weights=weights.get('ema_model',weights.get('model',weights));a.latent_frames=sample.resolve_latent_frames(weights,meta,a)
m=sample.build_denoiser(a).cuda().eval();sample.load_weights(weights,m);del weights
result=dict(kind='synthetic memory feasibility only',window_latent=512,cache_latent=496,query_latent=16,vae_loaded=False,gpu=torch.cuda.get_device_name(),gpu_bytes=torch.cuda.get_device_properties(0).total_memory,expected_both_branches_kv_bytes=2*28*496*300*1536*6)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 x=torch.zeros(1,48,16,15,20,device='cuda');t=torch.zeros(1,16,device='cuda');cond=torch.zeros(1,16,28,device='cuda')
 m.net.forward_with_cache(x,t,cond,current_position_offset=0,chunk_size=2)
 torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 try:
  caches=[[ (torch.zeros(1,12,496*300,128,device='cuda',dtype=torch.float32),torch.zeros(1,12,496*300,128,device='cuda',dtype=torch.bfloat16)) for _ in range(28)] for _ in range(2)]
  result['cache_allocated']=True
  m.net.forward_with_cache(x,t,cond,past_kv_list=caches[0],current_position_offset=496,chunk_size=2)
  torch.cuda.synchronize();result['forward_succeeded']=True;result['forward_peak_bytes']=torch.cuda.max_memory_allocated()
  new=[(torch.zeros(1,12,600,128,device='cuda',dtype=torch.float32),torch.zeros(1,12,600,128,device='cuda',dtype=torch.bfloat16)) for _ in range(28)]
  try:
   appended=m._append_kv_cache(caches[0],new);torch.cuda.synchronize();result['original_append_succeeded']=True
  except torch.OutOfMemoryError:
   result['original_append_succeeded']=False;result['append_failure']='CUDA out of memory with both CFG caches resident; official append allocates a new full branch while old tensors remain alive.'
 except torch.OutOfMemoryError as ex:result['failure']=str(ex)
result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
(R/'memory_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
