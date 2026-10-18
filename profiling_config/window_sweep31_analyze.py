from pathlib import Path
import json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/window_sweep31_gt_episode022694')
ep=22694;windows=list(range(16,257,8));metrics=['psnr_db','ssim','lpips_alex'];labels=['PSNR (dB) ↑','SSIM ↑','LPIPS-Alex ↓'];rows=[];frames=[]
for w in windows:
 prefix=f'episode_{ep:06d}_w{w:03d}_'
 assert (R/(prefix+'complete.json')).exists(),w
 q=pd.read_csv(R/(prefix+'quality_per_frame.csv'));q=q[q.frame>0].copy();assert len(q)==1276
 assert np.isfinite(q[metrics].to_numpy()).all()
 s=json.loads((R/(prefix+'summary.json')).read_text());assert s['sink_frames']==0 and s['query_latent']==16 and s['effective_sampling_steps']==40
 records=json.loads((R/(prefix+'gt_history_commits.json')).read_text())
 for branch in ['cond','uncond']:
  commits=[v for v in records if v['branch']==branch];assert all(v['start_latent']==i*2 and v['end_latent_exclusive']==i*2+2 for i,v in enumerate(commits))
 rows.append(dict(episode_index=ep,window=w,history_capacity_latent=w-16,**q[metrics].mean().to_dict(),generation_sec=s['generation_sec']))
 q['window']=w;frames.append(q)
a=pd.DataFrame(rows);a.to_csv(R/'quality_by_window.csv',index=False);f=pd.concat(frames,ignore_index=True);f.to_csv(R/'quality_by_frame_all31.csv',index=False)
plt.style.use('seaborn-v0_8-whitegrid')
fig,axes=plt.subplots(1,3,figsize=(16,5))
for ax,m,label in zip(axes,metrics,labels):
 ax.plot(a.window,a[m],'o-',ms=3);ax.axvline(64,color='gray',ls='--',lw=1);ax.set(xlabel='Window capacity (latent frames)',ylabel=label,xticks=[16,64,128,192,256])
fig.suptitle('Episode 22694 | GT cached history | 31 independent window capacities')
fig.text(.06,.015,'Mean over identical RGB frames 1–1276. Query=16; effective steps=40; sink=0. Above 64: length extrapolation. One episode; no cross-episode error bars.',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94))
for ext in ['png','pdf']:fig.savefig(R/f'quality_vs_window.{ext}',dpi=200)
plt.close(fig)
colors=plt.colormaps['turbo'](np.linspace(0,1,31))
fig,axes=plt.subplots(3,1,figsize=(16,13),sharex=True)
for ax,m,label in zip(axes,metrics,labels):
 for w,color in zip(windows,colors):
  q=f[f.window==w];ax.plot(q.frame,q[m],color=color,lw=.8,alpha=.85,label=str(w))
 ax.set(ylabel=label);assert len(ax.lines)==31
axes[-1].set(xlabel='RGB frame index (seed frame 0 excluded)',xlim=(1,1276))
fig.suptitle('Episode 22694 | GT cached history | All 31 window capacities (raw per-frame quality)')
handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,title='Window capacity (latent)',loc='lower center',ncol=11,fontsize=9)
fig.tight_layout(rect=(0,.10,1,.96))
for ext in ['png','pdf']:fig.savefig(R/f'quality_vs_frame_31_curves.{ext}',dpi=200)
plt.close(fig)
report='''# Episode 22694：31 档窗口容量实验

窗口总容量为16、24、…、256 latent，共31次独立推理，历史KV容量为0、8、…、240 latent。固定去噪区16 latent、有效去噪40步、sink=0、种子20283710、相同动作及GT条件首帧。每次使用前1277 RGB帧（320 latent）；质量评估帧1–1276，排除条件首帧。

历史提交时用对应GT VAE latent在t=0重新计算两个CFG分支的KV；当前去噪区仍为生成内容，输出不替换GT。这是GT缓存历史teacher-forcing对照。所有档统一关闭sink，因此不复用之前sink=1的64/256实验。256等大于64的窗口仍属训练长度之外的外推。

quality_vs_window.png/pdf：每项指标一条31点曲线，每点是相同1276帧的逐帧指标平均。
quality_vs_frame_31_curves.png/pdf：每项指标31条线，横轴原始RGB帧序号，不平滑。每条线是一档独立运行；图例标窗口容量。

PSNR先逐帧计算再平均；SSIM Gaussian sigma1.5、win11、总体协方差；LPIPS Alex v0.1 FP32；全部基于视频编码前uint8 RGB。仅一个episode，不能解释为数据集总体结论。视频初段缓存未满，各档实际历史长度可能相同；横轴窗口图表示容量配置，不代表每帧缓存都满。

原始预测、对比视频、逐帧指标、提交审计、前向计时与日志平铺本目录。
'''
(R/'report.md').write_text(report)
(R/'index.html').write_text('<meta charset="utf-8"><title>Episode 22694 window sweep</title><style>body{max-width:1500px;margin:30px auto;font-family:system-ui}img{max-width:100%}pre{white-space:pre-wrap}</style><img src="quality_vs_window.png"><img src="quality_vs_frame_31_curves.png"><pre>'+html.escape(report)+'</pre>')
print(a.to_string(index=False))
