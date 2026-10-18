"""Evaluate saved predictions consistently against source RGB; excludes seed frame."""
from pathlib import Path
import json,time,concurrent.futures
import numpy as np
import pandas as pd
import torch,av,lpips,skimage
from skimage.metrics import structural_similarity
from decord import VideoReader,cpu
ROOT=Path('/data2/jian/output/Miniworld/DROID_long_horizon/500_frames')
torch.set_num_threads(4)
model=lpips.LPIPS(net='alex',version='0.1').cuda().eval()
rows=[];all_frames=[]
def ssim(pair):
 return structural_similarity(pair[0],pair[1],data_range=255,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool,torch.inference_mode():
 for file in sorted(ROOT.glob('episode_*_summary.json')):
  info=json.loads(file.read_text());ep=info['episode_index'];n=info['processed_rgb_frames'];start=time.time()
  vr=VideoReader(info['source'],ctx=cpu(0),width=320,height=240,num_threads=4)
  gt=vr.get_batch(list(range(n))).asnumpy()
  with av.open(str(ROOT/f'episode_{ep:06d}_prediction.mp4')) as c:pred=np.stack([f.to_ndarray(format='rgb24') for f in c.decode(video=0)])
  assert gt.shape==pred.shape==(497,240,320,3),(gt.shape,pred.shape)
  mse=np.mean((gt.astype(np.float64)-pred.astype(np.float64))**2,axis=(1,2,3))
  psnr=10*np.log10(255**2/np.maximum(mse,1e-12))
  ss=np.asarray(list(pool.map(ssim,zip(gt,pred))))
  distances=[]
  for offset in range(0,n,32):
   a=torch.from_numpy(gt[offset:offset+32]).permute(0,3,1,2).cuda().float()/127.5-1
   b=torch.from_numpy(pred[offset:offset+32]).permute(0,3,1,2).cuda().float()/127.5-1
   distances.extend(model(a,b).flatten().cpu().tolist())
  frame=pd.DataFrame(dict(episode_index=ep,frame=np.arange(n),included_in_aggregate=np.arange(n)>0,psnr_db=psnr,ssim=ss,lpips_alex=distances))
  assert np.isfinite(frame[['psnr_db','ssim','lpips_alex']].to_numpy()).all()
  frame.to_csv(ROOT/f'episode_{ep:06d}_quality_metrics.csv',index=False);all_frames.append(frame)
  selected=frame[frame.included_in_aggregate]
  row=dict(episode_index=ep,evaluated_frames=len(selected),psnr_db=float(selected.psnr_db.mean()),ssim=float(selected.ssim.mean()),lpips_alex=float(selected.lpips_alex.mean()),evaluation_sec=time.time()-start)
  rows.append(row);pd.DataFrame(rows).to_csv(ROOT/'quality_metrics_per_video.csv',index=False)
  print(json.dumps(row),flush=True)
per=pd.DataFrame(rows);frames=pd.concat(all_frames,ignore_index=True);selected=frames[frames.included_in_aggregate]
frames.to_csv(ROOT/'quality_metrics_per_frame.csv',index=False)
result=dict(videos=len(rows),evaluated_frames=len(selected),psnr_db=float(selected.psnr_db.mean()),ssim=float(selected.ssim.mean()),lpips_alex=float(selected.lpips_alex.mean()),per_video_std={k:float(per[k].std()) for k in ['psnr_db','ssim','lpips_alex']},protocol=dict(prediction='Saved libx264 CRF18 prediction.mp4 decoded as RGB24 with PyAV; includes export compression',reference='Original source video decoded/resized with decord to 240x320, same resize path as inference loader',frames='Indices 1..496; index 0 conditioning seed excluded',aggregation='Arithmetic mean of frame metrics; same frame count per video, so also equals mean of video means',psnr='RGB 8-bit, data_range=255, per-frame MSE then PSNR',ssim='skimage RGB channel mean; Gaussian weights sigma=1.5, win_size=11, use_sample_covariance=False, data_range=255',lpips='lpips 0.1.4, learned AlexNet LPIPS v0.1, FP32, RGB normalized to [-1,1]',skimage_version=skimage.__version__,note='Earlier PSNR measured pre-export uint8 predictions; this report recomputes all three metrics on the same saved videos.'))
(ROOT/'quality_metrics_summary.json').write_text(json.dumps(result,indent=2))
table='\n'.join(f"| {int(r.episode_index):06d} | {r.psnr_db:.3f} | {r.ssim:.4f} | {r.lpips_alex:.4f} |" for _,r in per.iterrows())
report=f'''# 500 帧组图像质量指标

10 条视频，每条评估生成帧 1–496，排除首帧条件，共 {len(selected)} 帧。

| 指标 | 全部生成帧均值 |
|---|---:|
| PSNR ↑ | {result['psnr_db']:.3f} dB |
| SSIM ↑ | {result['ssim']:.4f} |
| LPIPS-Alex ↓ | {result['lpips_alex']:.4f} |

| Episode | PSNR dB ↑ | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|
{table}

统一比较已导出的 prediction.mp4 与原始输入视频经相同 decord 路径缩放后的 RGB，分辨率 240×320。预测视频经过 H.264 CRF18 压缩，因此本报告与此前导出前 PSNR 可能略有差异；导出前预测张量未保存，不能将这些指标标为无损原始模型输出指标。

PSNR 为每帧 RGB MSE 转 dB 后取均值。SSIM 使用 Gaussian sigma=1.5、11×11 窗口、总体协方差、RGB 三通道均值。LPIPS 使用官方 lpips 包的 AlexNet v0.1 校准权重，输入 [-1,1]，FP32。三项均先逐帧计算再取算术均值。逐帧和逐视频数据分别在 quality_metrics_per_frame.csv 和 quality_metrics_per_video.csv；完整协议在 quality_metrics_summary.json。
'''
(ROOT/'quality_metrics_report.md').write_text(report)
print('FINAL '+json.dumps(result),flush=True)
