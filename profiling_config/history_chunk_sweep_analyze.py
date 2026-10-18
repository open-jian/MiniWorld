from pathlib import Path
import json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/history_chunk_sweep31_gt_episode022694')
ep=22694;windows=list(range(0,121,4));metrics=['psnr_db','ssim','lpips_alex'];labels=['PSNR (dB) ↑','SSIM ↑','LPIPS-Alex ↓'];rows=[];frames=[]
for w in windows:
 prefix=f'episode_{ep:06d}_history{w:03d}chunks_'
 assert (R/(prefix+'complete.json')).exists(),w
 q=pd.read_csv(R/(prefix+'quality_per_frame.csv'));q=q[q.frame>0].copy();assert len(q)==1276
 assert np.isfinite(q[metrics].to_numpy()).all()
 s=json.loads((R/(prefix+'summary.json')).read_text());assert s['sink_frames']==1 and s['query_latent']==16 and s['effective_sampling_steps']==40
 records=json.loads((R/(prefix+'gt_history_commits.json')).read_text())
 for branch in ['cond','uncond']:
  commits=[v for v in records if v['branch']==branch];assert all(v['start_latent']==i*2 and v['end_latent_exclusive']==i*2+2 for i,v in enumerate(commits))
 rows.append(dict(episode_index=ep,history_window_chunks=w,sliding_history_latent=2*w,sink_latents=1,total_capacity_latent=17+2*w,**q[metrics].mean().to_dict(),generation_sec=s['generation_sec']))
 q['history_window_chunks']=w;frames.append(q)
a=pd.DataFrame(rows);a.to_csv(R/'quality_by_window.csv',index=False);f=pd.concat(frames,ignore_index=True);f.to_csv(R/'quality_by_frame_all31.csv',index=False)
plt.style.use('seaborn-v0_8-whitegrid')
fig,axes=plt.subplots(1,3,figsize=(16,5))
for ax,m,label in zip(axes,metrics,labels):
 ax.plot(a.history_window_chunks,a[m],'o-',ms=3);ax.set(xlabel='Sliding history window (chunks; excludes sink and query)',ylabel=label,xticks=[0,20,40,60,80,100,120])
fig.suptitle('Episode 22694 | GT cached history | 31 sliding-history windows; sink=1 latent; query=8 chunks')
fig.text(.06,.015,'Mean over identical RGB frames 1–1276. Effective steps=40; fixed sink preserved. Total attention = 1 + 2N + 16 latent. One episode.',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94))
for ext in ['png','pdf']:fig.savefig(R/f'quality_vs_window.{ext}',dpi=200)
plt.close(fig)
colors=plt.colormaps['turbo'](np.linspace(0,1,31))
fig,axes=plt.subplots(3,1,figsize=(16,13),sharex=True)
for ax,m,label in zip(axes,metrics,labels):
 for w,color in zip(windows,colors):
  q=f[f.history_window_chunks==w];ax.plot(q.frame,q[m],color=color,lw=.8,alpha=.85,label=str(w))
 ax.set(ylabel=label);assert len(ax.lines)==31
axes[-1].set(xlabel='RGB frame index (seed frame 0 excluded)',xlim=(1,1276))
fig.suptitle('Episode 22694 | GT cached history | 31 sliding-history windows (chunks); sink=1 latent; query=8 chunks')
handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,title='Sliding history window (chunks; excludes sink and query)',loc='lower center',ncol=11,fontsize=9)
fig.tight_layout(rect=(0,.10,1,.96))
for ext in ['png','pdf']:fig.savefig(R/f'quality_vs_frame_31_curves.{ext}',dpi=200)
plt.close(fig)
report='# Episode 22694：31 档历史窗口实验（chunk）\n\n窗口统一定义：只包含可滑动历史，单位chunk，不包含固定sink或正在去噪的区域。历史窗口N=0、4、8、…、120 chunk，共31档独立运行。每chunk=2 latent；sink固定1 latent；去噪区固定8 chunk。历史KV总容量为1+2N latent，注意力总容量上限为1+2N+16 latent。N=0仍保留sink和去噪区。\n\n保持原sink保留和淘汰逻辑；实验脚本仅将原采样函数的缓存容量表达式改为max_cache_chunks*chunk_size+sink_frames，以精确实现sink之外N个chunk的容量，不把sink隐含减进窗口。核心源码文件不改。每次淘汰逐层核查sink K/V逐元素保持不变，并断言去噪使用的历史容量不超上限且实际达到上限。训练窗口原为64 latent，总容量超过64仍属长度外推；只放宽运行时检查，不修改权重或RoPE频率。\n\n同episode 22694、同种子20283710、同动作、40有效去噪步、1277 RGB帧。历史提交用GT VAE latent重新计算CFG两分支KV，去噪区与输出保持预测。逐帧比较RGB 1–1276，排除条件首帧，采用MP4编码前uint8预测。PSNR逐帧计算再平均；SSIM Gaussian sigma1.5、win11、总体协方差；LPIPS Alex v0.1 FP32。\n\nquality_vs_window：横轴历史窗口chunk，纵轴相同评估帧平均质量，每项指标31点。quality_vs_frame_31_curves：横轴RGB帧序号，每项指标31条原始未平滑曲线，图例为历史窗口chunk。所有图标题标明episode。仅一个episode，不外推为数据集结论。缓存未满时不同容量可有相同实际历史，窗口图横轴是容量设置。\n\n不复用先前sink=0的数据。所有预测视频、无损数组、逐帧指标、GT提交记录和日志均在本目录。sink逐元素检查额外消耗算力，因此本实验的端到端耗时不作为纯性能结果。\n'
(R/'report.md').write_text(report)
(R/'index.html').write_text('<meta charset="utf-8"><title>Episode 22694 window sweep</title><style>body{max-width:1500px;margin:30px auto;font-family:system-ui}img{max-width:100%}pre{white-space:pre-wrap}</style><img src="quality_vs_window.png"><img src="quality_vs_frame_31_curves.png"><pre>'+html.escape(report)+'</pre>')
print(a.to_string(index=False))
