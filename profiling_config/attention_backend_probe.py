import sys,json,time
from pathlib import Path
import torch
from torch.profiler import profile,ProfilerActivity
sys.path.insert(0,'/data2/jian/project/MiniWorld')
from miniworld.miniworld import _build_cached_block_causal_mask
from flash_attn import flash_attn_func
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/acceleration_assessment')
torch.set_num_threads(4);torch.manual_seed(42)
results=[]
for window in [24,120]:
 past=(1+2*window)*300;n=4800
 q=torch.randn(1,12,n,128,device='cuda',dtype=torch.float32)
 k=torch.randn(1,12,past+n,128,device='cuda',dtype=torch.float32)
 v=torch.randn(1,12,past+n,128,device='cuda',dtype=torch.bfloat16)
 mask=_build_cached_block_causal_mask(past,n,300,2,q.device,torch.bfloat16)
 qb=q.bfloat16();kb=k.bfloat16()
 def original():
  with torch.autocast('cuda',dtype=torch.bfloat16):return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
 def normalized():return torch.nn.functional.scaled_dot_product_attention(qb,kb,v,attn_mask=mask)
 def split_flash():
  outs=[]
  for i in range(8):
   end=past+(i+1)*600
   outs.append(flash_attn_func(qb[:,:,i*600:(i+1)*600].transpose(1,2),kb[:,:,:end].transpose(1,2),v[:,:,:end].transpose(1,2),causal=False))
  return torch.cat(outs,dim=1).transpose(1,2)
 def bench(fn):
  for _ in range(3):fn()
  torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record()
  for _ in range(10):out=fn()
  b.record();torch.cuda.synchronize();return a.elapsed_time(b)/10,out
 with torch.inference_mode():
  t,out=bench(original)
  with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:original();torch.cuda.synchronize()
  operators=[dict(name=e.key,cuda_us=e.device_time_total) for e in prof.key_averages() if any(x in e.key.lower() for x in ['attention','flash','cudnn','efficient'])]
  row=dict(history_chunks=window,original_ms=t,sdpa_operators=operators)
  for label,fn in [('precast_bf16',normalized),('eight_chunk_flash',split_flash)]:
   try:
    ms,y=bench(fn);diff=(out.float()-y.float());row[label]=dict(ms=ms,speedup=t/ms,rmse=float(diff.square().mean().sqrt()),max_abs=float(diff.abs().max()))
   except Exception as e:row[label]=dict(error=str(e))
  results.append(row);print(json.dumps(row),flush=True)
 del q,k,v,mask,qb,kb,out;torch.cuda.empty_cache()
(R/'attention_backend_probe.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,synthetic=True,scope='Single attention layer only, random tensors, prebuilt mask, no model/quality inference. Split flash preserves chunkwise causal visibility.',results=results),indent=2))
