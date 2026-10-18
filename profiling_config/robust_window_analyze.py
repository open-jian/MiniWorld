from pathlib import Path
import json,html
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/generated_history_5episodes_3seeds_window0_24');cfg=json.loads((R/'experiment.json').read_text());metrics=['psnr_db','ssim','lpips_alex'];labels=['PSNR (dB) ↑','SSIM ↑','LPIPS-Alex ↓'];rows=[];frames=[]
for ep in cfg['episodes']:
 for base in cfg['seed_bases']:
  seed=ep+base
  for w in range(25):
   prefix=f'episode_{ep:06d}_seed{seed}_history{w:03d}chunks_';assert (R/(prefix+'complete.json')).exists()
   s=json.loads((R/(prefix+'summary.json')).read_text());assert s['seed']==seed and s['sink_frames']==1 and s['query_latent']==16 and s['effective_sampling_steps']==40
   q=pd.read_csv(R/(prefix+'quality_per_frame.csv'));q=q[q.frame>0];assert len(q)==1276 and np.isfinite(q[metrics].to_numpy()).all()
   rows.append(dict(episode=ep,seed=seed,seed_base=base,window_chunks=w,**q[metrics].mean().to_dict()))
   frames.append(q.assign(episode=ep,seed=seed,seed_base=base,window_chunks=w))
a=pd.DataFrame(rows);f=pd.concat(frames,ignore_index=True);a.to_csv(R/'quality_per_episode_seed_window.csv',index=False);f.to_csv(R/'quality_per_frame_all_runs.csv',index=False)
e=a.groupby(['episode','window_chunks'])[metrics].mean().reset_index();e.to_csv(R/'quality_per_episode_seedmean.csv',index=False)
stats=e.groupby('window_chunks')[metrics].agg(['mean','std']);stats.to_csv(R/'quality_by_window_mean_episode_sd.csv')
within=a.groupby(['episode','window_chunks'])[metrics].std();within.to_csv(R/'quality_seed_sd_per_episode.csv')
best=[]
for (ep,seed),q in a.groupby(['episode','seed']):
 for m in metrics:
  row=q.loc[q[m].idxmin() if m=='lpips_alex' else q[m].idxmax()]
  best.append(dict(episode=ep,seed=seed,metric=m,best_window_chunks=int(row.window_chunks),value=row[m]))
best=pd.DataFrame(best);best.to_csv(R/'best_window_per_episode_seed.csv',index=False);best.groupby(['metric','best_window_chunks']).size().to_csv(R/'best_window_counts.csv',header=['runs'])
plt.style.use('seaborn-v0_8-whitegrid')
fig,axes=plt.subplots(1,3,figsize=(17,5.5))
for ax,m,label in zip(axes,metrics,labels):
 z=stats[m];x=z.index.to_numpy();y=z['mean'].to_numpy();sd=z['std'].to_numpy();ax.plot(x,y,'o-',ms=3);ax.fill_between(x,y-sd,y+sd,alpha=.18)
 ax.set(xlabel='Sliding history window (chunks)',ylabel=label,xticks=range(0,25,2))
fig.suptitle('Generated history | 5 episodes × 3 seeds | fixed sink=1 latent, query=8 chunks',color='red')
fig.text(.05,.02,'Seed means within each episode, then mean ± SD across 5 episodes. All evaluate RGB 1–1276. Window excludes sink and query.',color='red',fontsize=10)
fig.tight_layout(rect=(0,.06,1,.94))
for ext in ['png','pdf']:fig.savefig(R/f'quality_vs_window.{ext}',dpi=200)
plt.close(fig)
colors=plt.colormaps['turbo'](np.linspace(0,1,25))
# Frame-by-frame means: 15 rollouts overall; 3 seeds for each individual episode.
for ep in [None]+cfg['episodes']:
 data=f if ep is None else f[f.episode==ep]
 z=data.groupby(['window_chunks','frame'])[metrics].mean().reset_index()
 name='all_episodes' if ep is None else f'episode_{ep:06d}'
 z.to_csv(R/f'{name}_per_frame_mean.csv',index=False)
 fig,axes=plt.subplots(3,1,figsize=(17,14),sharex=True)
 for ax,m,label in zip(axes,metrics,labels):
  for w,c in zip(range(25),colors):
   q=z[z.window_chunks==w];ax.plot(q.frame,q[m],color=c,lw=1,label=str(w))
  ax.set(ylabel=label);assert len(ax.lines)==25
 axes[-1].set(xlabel='RGB frame index (seed 0 excluded)',xlim=(1,1276))
 title='5 episodes × 3 seeds mean' if ep is None else f'Episode {ep}: mean of 3 seeds'
 fig.suptitle(f'Generated history | {title} | sink=1 latent, query=8 chunks',color='red')
 h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,title='Sliding history window (chunks; excludes sink and query)',loc='lower center',ncol=13)
 fig.tight_layout(rect=(0,.09,1,.96))
 for ext in ['png','pdf']:fig.savefig(R/f'{name}_quality_vs_frame_25_curves.{ext}',dpi=200)
 plt.close(fig)
report='''# 正常自回归：5 episode × 3 seeds，历史窗口0–24 chunk

Episodes: 22694、46584、19464、27206、12835。种子基数20261016/20261017/20261018，实际随机种子=基数+episode；同episode与种子下各窗口使用相同随机数种子。窗口0到24逐chunk取点，共375次独立生成。窗口只指滑动历史，不包含固定sink=1 latent和去噪区8 chunk。保持有效去噪40步、条件首帧、动作及其他采样设置。每条生成1277 RGB帧，评估1–1276。

历史由模型生成，不用GT替换。GT只用于首帧条件及输出质量评估。原模型训练总长度64 latent；24 chunk历史+1 sink+16 query=65 latent，最后一档略超过训练长度，运行时放宽检查，权重和RoPE频率不变。

主窗口图先对每episode的3个种子取均值，再对5个episode取均值；阴影是5个episode均值的样本标准差，不是置信区间。逐帧总图是15条rollout在相同帧序号上的均值（不同episode不是同一场景），另附每episode三种子平均的逐帧图。均未做时间平滑。

best_window_per_episode_seed.csv和best_window_counts.csv给出15个episode/种子组合各自的最佳窗口，用于观察峰值是否稳定；quality_seed_sd_per_episode.csv记录种子波动。PSNR逐帧计算后平均，SSIM Gaussian sigma1.5/win11/总体协方差，LPIPS Alex v0.1 FP32。均使用MP4压缩前uint8 RGB。

仅5条选定episode、3个种子，结论应限定于该样本。每次缓存淘汰逐层检查sink保持不变；计时包含额外检查开销，不作纯性能结论。
'''
(R/'report.md').write_text(report)
(R/'index.html').write_text('<meta charset="utf-8"><style>body{max-width:1600px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}</style><img src="quality_vs_window.png"><img src="all_episodes_quality_vs_frame_25_curves.png"><pre>'+html.escape(report)+'</pre>')
print(stats.to_string());print(best.groupby(['metric','best_window_chunks']).size().to_string())
