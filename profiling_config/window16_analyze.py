from pathlib import Path
import json,html,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/window_016')
B=Path(__file__).resolve().parent
rows=[];progress=[]
for ep in [22694,46584,19464,27206]:
 assert (R/f'episode_{ep:06d}_complete.json').exists()
 for w in range(2,17,2):
  p=f'episode_{ep:06d}_w{w:03d}_'
  s=json.loads((R/(p+'summary.json')).read_text())
  f=pd.read_json(R/(p+'forward_calls.json'))
  f=f[(f.role=='denoise')&(f.current_frames==w)]
  assert (f.cache_frames==0).all() and (f.cache_bytes==0).all()
  g=f.groupby('step').agg(gpu_ms=('gpu_ms','sum'),branches=('branch','nunique'))
  assert (g.branches==2).all()
  # Discard the first five steps at each query shape; exclude shrinking tail above.
  g=g.iloc[5:];assert len(g)>0
  q=pd.read_csv(R/(p+'quality_per_frame.csv')).set_index('frame')
  rows.append(dict(episode_index=ep,window=w,query_latent=w,cache_latent=0,kv_gib=0,effective_sampling_steps=s['effective_sampling_steps'],cfg_gpu_ms=g.gpu_ms.mean(),cfg_gpu_ms_p50=g.gpu_ms.median(),cfg_gpu_ms_p95=g.gpu_ms.quantile(.95),measured_steps=len(g),generation_sec=s['generation_sec'],rgb_fps=s['generated_rgb_fps'],peak_allocated_gib=s['peak_allocated_bytes']/2**30,**q.loc[1:,['psnr_db','ssim','lpips_alex']].mean().to_dict()))
  for end in range(28,253,32):
   progress.append(dict(episode_index=ep,window=w,rgb_end=end,**q.loc[max(1,end-31):end,['psnr_db','ssim','lpips_alex']].mean().to_dict()))
a=pd.DataFrame(rows);a.to_csv(R/'per_episode_summary.csv',index=False)
pd.DataFrame(progress).to_csv(R/'quality_every8_latent.csv',index=False)
z=a.groupby('window').agg(cfg_gpu_ms_mean=('cfg_gpu_ms','mean'),cfg_gpu_ms_std=('cfg_gpu_ms','std'),episodes=('episode_index','count'),effective_sampling_steps=('effective_sampling_steps','first'),kv_gib=('kv_gib','mean'),generation_sec=('generation_sec','mean'),rgb_fps=('rgb_fps','mean'),peak_allocated_gib=('peak_allocated_gib','max'),psnr_db=('psnr_db','mean'),ssim=('ssim','mean'),lpips_alex=('lpips_alex','mean')).reset_index()
z.to_csv(R/'summary.csv',index=False)
old=pd.read_csv(R.parent/'window_256/latency_by_window.csv');old=old[old.capacity==256]
combined=pd.concat([z[['window','cfg_gpu_ms_mean','cfg_gpu_ms_std']].rename(columns={'window':'active_window','cfg_gpu_ms_mean':'gpu_ms_mean','cfg_gpu_ms_std':'gpu_ms_std'}).assign(regime='vary_query_no_history'),old[['active_window','gpu_ms_mean','gpu_ms_std']].assign(regime='fixed_query_16_grow_history')],ignore_index=True)
combined.to_csv(R/'latency_2_to_256.csv',index=False)
plt.style.use('seaborn-v0_8-whitegrid')
fig,(ax,zoom)=plt.subplots(1,2,figsize=(14,5),gridspec_kw={'width_ratios':[2,1]})
for pane in [ax,zoom]:
 x=z.window.to_numpy();y=z.cfg_gpu_ms_mean.to_numpy();sd=z.cfg_gpu_ms_std.to_numpy()
 pane.errorbar(x,y,yerr=sd,fmt='o--',color='#dc7b22',ms=4,label='Query 2–16; no retained history')
 if pane is ax:
  pane.plot(old.active_window,old.gpu_ms_mean,'o-',color='#1477AB',ms=3,label='Query 16; history 0–240 (previous run)')
  pane.fill_between(old.active_window,old.gpu_ms_mean-old.gpu_ms_std,old.gpu_ms_mean+old.gpu_ms_std,color='#1477AB',alpha=.15)
  pane.set(xlim=(0,260),xticks=[0,16,64,128,192,256],title='Combined view: two explicitly different regimes')
 else:
  anchor=old[old.active_window==16].iloc[0]
  pane.scatter([16],[anchor.gpu_ms_mean],color='#1477AB',marker='s',label='Previous run at window 16',zorder=5)
  pane.set(xlim=(0,17),xticks=np.arange(0,17,2),title='Supplement: small query, zero retained KV')
 pane.set(xlabel='Actual active window (latent frames)',ylabel='GPU latency per CFG step (two forwards), ms',ylim=(0,None));pane.legend(fontsize=8)
fig.text(.08,.015,'4 episodes, mean ± episode SD. No measurement at 0. Timing excludes commits, eviction, VAE and I/O.',fontsize=9)
fig.tight_layout(rect=(0,.04,1,1))
for suffix in ['png','pdf']:fig.savefig(R/f'window_size_latency_0_to_256.{suffix}',dpi=180)
plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,4.5))
for ax,metric,title in zip(axes,['psnr_db','ssim','lpips_alex'],['PSNR ↑ (dB)','SSIM ↑','LPIPS-Alex ↓']):
 t=a.groupby('window')[metric].agg(['mean','std']);ax.errorbar(t.index,t['mean'],yerr=t['std'],fmt='o-',color='#dc7b22');ax.set(xlabel='Query / active window (latent)',ylabel=title,xticks=np.arange(2,17,2))
fig.suptitle('Small-window supplement: quality over the same 252 generated RGB frames')
fig.text(.05,.015,'Zero retained history / no sink. Official schedule: 5–40 effective denoising steps as query grows 2–16. Not an isolated KV-size effect.',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94))
for suffix in ['png','pdf']:fig.savefig(R/f'quality_small_window.{suffix}',dpi=180)
plt.close(fig)
table='\n'.join(f'| {int(r.window)} | {r.cfg_gpu_ms_mean:.2f} | {int(r.effective_sampling_steps)} | {r.psnr_db:.3f} | {r.ssim:.4f} | {r.lpips_alex:.4f} |' for r in z.itertuples())
report=f'''# 0–16 latent 补充实验

仅补左侧；256–512 已取消。0 没有可执行的网络输入，因此不填测量值。实测 2、4、6、8、10、12、14、16 latent；同四条 episode 22694、46584、19464、27206，物理 GPU 4–7。

## 方法

原实验的去噪区固定为 16 latent，实际窗口最小为 16，无法在保持此配置时测到更小窗口。本补充实验设置 inflight_chunks=window/2、max_cache_chunks=0、sink_frames=0，history_len=1（首帧条件），训练窗口检查仍为原始 64。所有官方核心代码、模型权重不变。两段实验使用不同颜色，16 处保留各自实测点，不强行平移或拟合拼接。

每条使用相同前 253 RGB 帧（64 latent）、相同种子 20261016+episode。先预热，再分别生成 8 个配置。采样器原有 residence cap 使 effective_steps 随窗口从 5 增至 40，因此质量同时受去噪区、去噪步数、缺少历史及 sink 影响；不能解释为单纯 KV 容量的因果效应。此质量图和旧的增长阶段检查点图不拼接。

计时用 CUDA events，对条件/无条件前向按 step 求和，仅统计 current_frames 等于目标窗口且缓存为 0 的去噪步骤；剔除每个配置最初 5 步和尾部缩小查询。每条先取步骤均值，再取四条均值和样本标准差。生成总耗时另含 VAE、提交及淘汰，见 CSV。保留历史 KV=0，但提交时仍会临时生成 KV 再淘汰；0 GiB 不表示网络没有临时 K/V 或不消耗显存。

## 四条平均

| 窗口 latent | CFG 两次前向 ms | 有效去噪步数 | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|---:|---:|---:|
{table}

质量在 MP4 编码前的 uint8 RGB 上评估，排除条件首帧，逐帧指标取均值。SSIM Gaussian sigma=1.5、win=11、use_sample_covariance=False；LPIPS AlexNet v0.1。每隔 8 个生成 latent 保存一张 GT/预测对比图及最多 32 RGB 帧区间指标，见 quality_every8_latent.csv。完整预测、GT/预测视频、原始预测数组、逐次计时、逐帧质量、GPU 遥测和日志均平铺于本目录。
'''
(R/'report.md').write_text(report)
(R/'index.html').write_text('<meta charset="utf-8"><title>MiniWorld 0–16 supplement</title><style>body{max-width:1400px;margin:30px auto;font-family:system-ui}img{max-width:100%}pre{white-space:pre-wrap}</style><img src="window_size_latency_0_to_256.png"><img src="quality_small_window.png"><pre>'+html.escape(report)+'</pre>')
for name in ['window16_worker.py','window16_runner.py','window16_analyze.py']:shutil.copy(B/name,R/name)
print(z.to_string(index=False))
