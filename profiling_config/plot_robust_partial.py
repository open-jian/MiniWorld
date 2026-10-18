from pathlib import Path
import json,html
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/generated_history_5episodes_3seeds_window0_24');O=R/'partial_figures';O.mkdir(exist_ok=True)
metrics=['psnr_db','ssim','lpips_alex'];labels=['PSNR (dB) ↑','SSIM ↑','LPIPS-Alex ↓'];rows=[];frames=[]
for p in sorted(R.glob('episode_*_history???chunks_complete.json')):
 v=json.loads(p.read_text());ep=v['episode_index'];seed=v['seed'];w=v['window'];prefix=p.name.removesuffix('complete.json')
 q=pd.read_csv(R/(prefix+'quality_per_frame.csv'));q=q[q.frame>0].copy();assert len(q)==1276 and np.isfinite(q[metrics].to_numpy()).all()
 rows.append(dict(episode=ep,seed=seed,window_chunks=w,**q[metrics].mean().to_dict()));frames.append(q.assign(episode=ep,seed=seed,window_chunks=w))
a=pd.DataFrame(rows);f=pd.concat(frames,ignore_index=True);a.to_csv(O/'all_completed_run_means.csv',index=False);f.to_csv(O/'all_completed_per_frame.csv',index=False)
pairs=sorted(set(zip(a.episode,a.seed)));sets=[set(a[(a.episode==ep)&(a.seed==seed)].window_chunks) for ep,seed in pairs];common=sorted(set.intersection(*sets));assert common==list(range(19,25))
matched=a[a.window_chunks.isin(common)];e=matched.groupby(['episode','window_chunks'])[metrics].mean().reset_index();stats=e.groupby('window_chunks')[metrics].agg(['mean','std']);stats.to_csv(O/'matched_window_mean_episode_sd.csv');matched.to_csv(O/'matched_runs.csv',index=False)
plt.style.use('seaborn-v0_8-whitegrid')
red='#c51b1b'
fig,axes=plt.subplots(1,3,figsize=(16,5.5))
for ax,m,label in zip(axes,metrics,labels):
 z=stats[m];x=z.index.to_numpy();y=z['mean'].to_numpy();sd=z['std'].to_numpy();ax.errorbar(x,y,yerr=sd,fmt='o-',capsize=3)
 ax.set(xlabel='Sliding history window (chunks)',ylabel=label,xticks=common)
fig.suptitle('PARTIAL RESULTS | Generated history | Window size → Mean quality',color=red,fontsize=17)
fig.text(.5,.89,'Matched windows 19–24 only: same 8 episode/seed pairs, covering 5 episodes',ha='center',color=red)
fig.text(.5,.055,'Mean seeds within each episode, then equal-weight mean ± SD across 5 episodes; only 1–2 seeds per episode.',ha='center',color=red,fontsize=10)
fig.text(.5,.02,'Fixed sink=1 latent, query=8 chunks; window excludes both. RGB frames 1–1276. Windows 0–16 have no completed runs.',ha='center',color=red,fontsize=10)
fig.tight_layout(rect=(0,.12,1,.86))
for ext in ['png','pdf']:fig.savefig(O/f'partial_quality_vs_window.{ext}',dpi=180)
plt.close(fig)
fmatch=f[f.window_chunks.isin(common)];ef=fmatch.groupby(['episode','window_chunks','frame'])[metrics].mean().reset_index();gf=ef.groupby(['window_chunks','frame'])[metrics].mean().reset_index();gf.to_csv(O/'matched_mean_quality_per_frame.csv',index=False)
fig,axes=plt.subplots(3,1,figsize=(16,12),sharex=True)
for ax,m,label in zip(axes,metrics,labels):
 for w in common:
  q=gf[gf.window_chunks==w];ax.plot(q.frame,q[m],lw=1,label=f'{w} chunks')
 ax.set(ylabel=label);ax.legend(ncol=3,fontsize=9)
axes[-1].set(xlabel='RGB frame index (seed frame 0 excluded)',xlim=(1,1276))
fig.suptitle('PARTIAL RESULTS | Generated history | Frame index → Quality',color=red,fontsize=17)
fig.text(.5,.935,'Same 8 episode/seed pairs at all 6 windows; seed means first, then equal-weight mean across 5 episodes.',ha='center',color=red,fontsize=11)
fig.text(.5,.025,'Sink=1 latent; query=8 chunks. No smoothing. Different episodes at the same frame index represent different scenes.',ha='center',color=red,fontsize=10)
fig.tight_layout(rect=(0,.05,1,.92))
for ext in ['png','pdf']:fig.savefig(O/f'partial_quality_vs_frame.{ext}',dpi=180)
plt.close(fig)
cfg=json.loads((R/'experiment.json').read_text());allpairs=[(ep,ep+base) for ep in cfg['episodes'] for base in cfg['seed_bases']];coverage=np.zeros((15,25),dtype=int)
for i,(ep,seed) in enumerate(allpairs):
 for w in a[(a.episode==ep)&(a.seed==seed)].window_chunks:coverage[i,int(w)]=1
pd.DataFrame(coverage,index=[f'{ep}/{seed}' for ep,seed in allpairs],columns=range(25)).to_csv(O/'completion_coverage.csv')
fig,ax=plt.subplots(figsize=(14,7));ax.imshow(coverage,cmap='Greens',vmin=0,vmax=1,aspect='auto');ax.set(xticks=range(25),yticks=range(15),yticklabels=[f'Ep {ep} / seed {seed}' for ep,seed in allpairs],xlabel='Sliding history window (chunks)',title=f'Completion coverage: {len(a)} / 375 runs (green = complete, white = missing)');ax.title.set_color(red)
fig.tight_layout();fig.savefig(O/'completion_coverage.png',dpi=180);plt.close(fig)
report=f'''部分结果，实验已按用户要求停止，未恢复推理。
完整结果共{len(a)}次。19–24 chunk各8次（相同组合），18 chunk为7次，17 chunk为1次。0–16无完整结果。
主图严格限制在共同的19–24 chunk范围。5条episode为22694、46584、19464、27206、12835；前三者各2个种子，后两者各1个种子。先平均同episode内已有种子，再对5条episode等权平均；误差条为5条episode均值的样本标准差，不是置信区间。帧序号图沿用相同分层平均。
17、18的额外结果完整保存在all_completed_run_means.csv及all_completed_per_frame.csv，不混入主曲线。completion_coverage展示所有计划375次运行的完成情况。
正常生成历史；sink=1 latent，去噪区8 chunk不变；窗口排除二者。评估RGB帧1–1276。当前部分结果不能用于判断4 chunk是否最佳，也不能称为已完成5 episode×3 seeds实验。
'''
(O/'report.txt').write_text(report)
(O/'index.html').write_text('<meta charset="utf-8"><style>body{max-width:1600px;margin:auto}img{width:100%}pre{white-space:pre-wrap}</style><img src="partial_quality_vs_window.png"><img src="partial_quality_vs_frame.png"><img src="completion_coverage.png"><pre>'+html.escape(report)+'</pre>')
print(stats.to_string());print('OUTPUT',O)
