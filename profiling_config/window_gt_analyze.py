from pathlib import Path
import json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/window_256_gt_history')
entries=json.loads((R/'selection.json').read_text())['videos'];all_lat=[];summaries=[];windows=[];progress=[];totals=[];prefix=[]
for entry in entries:
 ep=entry['episode_index'];artifact=lambda mode,name:R/f'episode_{ep:06d}_w{mode:03d}_{name}'
 if not (R/f'episode_{ep:06d}_complete.json').exists():continue
 prefix.append(json.loads((R/f'episode_{ep:06d}_complete.json').read_text()))
 w=pd.read_csv(R/f'episode_{ep:06d}_quality_by_active_window.csv')
 for mode in [64,256]:
  summaries.append(json.loads(artifact(mode,'summary.json').read_text()))
  f=pd.read_json(artifact(mode,'forward_calls.json'));f=f[(f.role=='denoise')&(f.current_frames==16)]
  g=f.groupby(['step','cache_frames']).agg(gpu_ms=('gpu_ms','sum'),branches=('branch','nunique'),kv_bytes_per_branch=('cache_bytes','max')).reset_index();assert (g.branches==2).all()
  g=g.groupby('cache_frames').agg(gpu_ms=('gpu_ms','mean'),kv_bytes_per_branch=('kv_bytes_per_branch','max')).reset_index();g['active_window']=g.cache_frames+16;g['capacity']=mode;g['episode_index']=ep;g['kv_gib_both']=g.kv_bytes_per_branch*2/2**30
  all_lat.append(g[g.active_window%8==0])
  q=pd.read_csv(artifact(mode,'quality_per_frame.csv')).set_index('frame')
  for metric in ['psnr_db','ssim','lpips_alex']:w[f'w{mode}_snapshot_{metric}']=w.rgb_frame.map(q[metric])
  totals.append(dict(episode_index=ep,capacity=mode,**{metric:float(q.loc[1:,metric].mean()) for metric in ['psnr_db','ssim','lpips_alex']}))
 w.to_csv(R/f'episode_{ep:06d}_quality_by_active_window.csv',index=False);windows.append(w)
 progress.append(pd.read_csv(R/f'episode_{ep:06d}_quality_every8_latent.csv'))
assert len(windows)==4,'Expected all four episodes'
l=pd.concat(all_lat,ignore_index=True);w=pd.concat(windows,ignore_index=True);p=pd.concat(progress,ignore_index=True);s=pd.DataFrame(summaries);t=pd.DataFrame(totals)
l.to_csv(R/'latency_by_window_per_episode.csv',index=False);w.to_csv(R/'quality_by_window_per_episode.csv',index=False);p.to_csv(R/'quality_every8_latent.csv',index=False);s.to_csv(R/'generation_summary.csv',index=False);t.to_csv(R/'quality_overall_per_episode.csv',index=False)
lstat=l.groupby(['capacity','active_window']).agg(gpu_ms_mean=('gpu_ms','mean'),gpu_ms_std=('gpu_ms','std'),kv_gib_both=('kv_gib_both','mean'),episodes=('episode_index','count')).reset_index();lstat.to_csv(R/'latency_by_window.csv',index=False)
wstat=w.groupby('active_window').mean(numeric_only=True).drop(columns=['episode_index']);wstat.to_csv(R/'quality_by_window.csv')
qstat=t.groupby('capacity').mean(numeric_only=True).drop(columns=['episode_index']);qstat.to_csv(R/'quality_overall.csv')
plt.style.use('seaborn-v0_8-whitegrid')
fig,ax=plt.subplots(figsize=(11,6.5))
for mode,color in [(256,'#1477AB')]:
 a=lstat[lstat.capacity==mode];x=a.active_window.to_numpy();y=a.gpu_ms_mean.to_numpy();sd=a.gpu_ms_std.to_numpy()
 ax.plot(x,y,'o-',ms=3,color=color,label=f'Capacity {mode} latent');ax.fill_between(x,y-sd,y+sd,color=color,alpha=.15)
ax.axvline(64,color='#888',ls='--');ax.set(xlabel='Actual active window (cached history + 16 in-flight latent frames)',ylabel='GPU latency per CFG step (two forwards), ms',title='MiniWorld 1B: Window Size vs. Latency — Four Episodes',xticks=np.arange(16,257,24));ax.legend()
fig.text(.12,.015,'Fixed query = 16 latent; mean ± video SD. Capacity 256 extrapolates beyond the 64-latent training window.',fontsize=9)
fig.tight_layout(rect=(0,.035,1,1));fig.savefig(R/'window_size_latency.png',dpi=180);fig.savefig(R/'window_size_latency.pdf');plt.close(fig)
for snapshot in [False,True]:
 fig,axes=plt.subplots(1,3,figsize=(15,4.8))
 for ax,metric,title in zip(axes,['psnr_db','ssim','lpips_alex'],['PSNR ↑ (dB)','SSIM ↑','LPIPS-Alex ↓']):
  for mode,color in [(256,'#1477AB')]:
   column=f'w{mode}_'+('snapshot_' if snapshot else '')+metric
   z=w.groupby('active_window')[column].agg(['mean','std']);xx=z.index.to_numpy();yy=z['mean'].to_numpy();sd=z['std'].to_numpy()
   ax.plot(xx,yy,'o-',ms=3,color=color,label=f'Capacity {mode}');ax.fill_between(xx,yy-sd,yy+sd,color=color,alpha=.12)
  ax.axvline(64,ls='--',color='#888');ax.set(xlabel='Actual active window (latent frames)',ylabel=title,xticks=[16,64,128,192,256]);ax.legend(fontsize=9)
 fig.suptitle('GT-history Run: Quality vs. Actual Window Size',fontsize=15)
 note='Exact snapshot frame at each checkpoint.' if snapshot else 'Mean over up to 32 generated RGB frames ending at each checkpoint.'
 fig.text(.06,.015,note+' Shading: ± video SD across 4 episodes. Window size and rollout time increase together.',fontsize=9)
 fig.tight_layout(rect=(0,.05,1,.93));name='snapshot_quality_vs_window' if snapshot else 'quality_vs_window';fig.savefig(R/f'{name}.png',dpi=180);fig.savefig(R/f'{name}.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(15,4.8))
for ax,metric,title in zip(axes,['psnr_db','ssim','lpips_alex'],['PSNR ↑ (dB)','SSIM ↑','LPIPS-Alex ↓']):
 for mode in [64,256]:
  z=p.groupby('rgb_end')[f'w{mode}_{metric}'].agg(['mean','std']);x=z.index.to_numpy();y=z['mean'].to_numpy();sd=z['std'].to_numpy();ax.plot(x,y,label=f'Capacity {mode} latent');ax.fill_between(x,y-sd,y+sd,alpha=.12)
 ax.set(xlabel='Generated RGB frames (excluding seed)',ylabel=title,xticks=[0,256,512,768,1024,1276]);ax.legend()
fig.suptitle('Capacity 64 vs. 256: Quality at the Same Generation Progress');fig.text(.06,.02,'X = generated RGB frames, not window size. Mean over up to 32 frames per checkpoint; shading = ± video SD.',fontsize=9);fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(R/'quality_vs_progress.png',dpi=180);fig.savefig(R/'quality_vs_progress.pdf');plt.close(fig)
endpoint=lstat[(lstat.capacity==256)&(lstat.active_window==256)].iloc[0]
summary=dict(episodes=4,rollout_rgb=1277,checkpoint_training_latent=64,extended_active_window=256,max_kv_gib=float(s[s.window==256].max_kv_bytes_both.max()/2**30),peak_allocated_gib=float(s[s.window==256].peak_allocated_bytes.max()/2**30),full_window_cfg_step_ms=float(endpoint.gpu_ms_mean),overall_quality=qstat.reset_index().to_dict(orient='records'),prefix_checks=prefix)
(R/'aggregate.json').write_text(json.dumps(summary,indent=2))
qtable='\n'.join(f"| {int(cap)} | {row.psnr_db:.3f} | {row.ssim:.4f} | {row.lpips_alex:.4f} |" for cap,row in qstat.iterrows())
report=f'''# MiniWorld GT 历史缓存对照实验

4 条 episode：22694、46584、19464、27206；仅 GPU 4–7。每条从第 0 帧开始，生成 320 latent / 1277 RGB。每条分别使用窗口上限 64 和 256，固定相同随机种子与动作条件。

## 配置与解释

窗口单位为 latent 帧。去噪区固定 8 chunk = 16 latent；64 窗口的历史缓存为 48 latent，256 窗口为 240 latent。后者历史缓存最多 120 chunk；CFG 双分支持久 KV 最大 {summary['max_kv_gib']:.4f} GiB，生成峰值 allocated {summary['peak_allocated_gib']:.2f} GiB。满 256 窗口时，一次 CFG 两分支前向合计平均 {summary['full_window_cfg_step_ms']:.2f} ms。

原权重仅训练于 64 latent。官方代码有 active_frames <= trained_num_frames 检查；实验脚本仅将运行时该检查属性设为 256，RoPE 按原始频率动态计算，不缩放、不插值、不修改模型权重或核心源码。因此这属于长度外推实验，不能称为原版支持范围内的默认推理。

窗口大小横轴的耗时图和质量图仅绘制 256 扩展组；64 与 256 的质量对照仅在以生成 RGB 帧数为横轴的 quality_vs_progress 图中绘制。64 对照组的实际窗口从未超过 64。

这是一个 rollout 中活动窗口从 16 长到 256 的实验，不是 31 个不同窗口上限的独立实验。每增加 8 latent，在首次完成输出的对应位置记录检查点。窗口大小和生成进度同时增长，因此加入同 episode、同随机种子、同 RGB 帧位置的 64 窗口对照。生成前 197 RGB 帧在发生首次不同淘汰前应一致，实测差异见 aggregate.json 的 prefix_checks。

## 全序列画面质量均值

| 窗口上限 latent | PSNR ↑ dB | SSIM ↑ | LPIPS-Alex ↓ |
|---|---:|---:|---:|
{qtable}

每条排除首帧条件，评估生成帧 1–1276。使用导出 MP4 前的 uint8 RGB（prediction_rgb.npy），不含有损视频编码误差。PSNR 为逐帧 RGB MSE 转 dB 后均值；SSIM Gaussian sigma1.5、11×11 窗口、总体协方差、RGB 通道均值；LPIPS 官方 AlexNet v0.1 校准权重，FP32，输入 [-1,1]。

## 每隔 8 的质量与图片

- quality_by_window.csv：active_window 为扩展组的实际窗口，w64 列为同帧位置对照的质量而非其窗口大小。窗口 16、24、…、256 共 31 个检查点，4 episode 平均。常规指标为检查点前最多 32 个生成 RGB 帧的均值；snapshot 指标为截图那一帧的指标，rgb_frame 给出准确帧位置。
- quality_by_window_per_episode.csv：逐 episode、逐窗口结果，可直接比较 w64 与 w256 同帧的差异。
- episode_XXXXXX_activeYYY_comparison.png：31 张无损检查点图片/episode，左 GT、中窗口64、右窗口256。
- quality_every8_latent.csv：额外按生成进度每 8 latent 记录一次，共 40 次/episode，包含满 256 窗口后的阶段。
- window_size_latency.png/pdf：实际窗口与两分支前向 GPU 耗时；固定 16 latent 查询，不含 VAE、缓存提交/淘汰、质量计算或写盘。
- quality_vs_window.png：仅扩展组，检查点区间均值；snapshot_quality_vs_window.png：仅扩展组，截图帧质量；quality_vs_progress.png/pdf：64 与 256 对照，横轴为生成 RGB 帧数。

完整视频、未压缩预测数组、逐帧指标、原始计时、显存、配置及 GPU 进程记录均在本目录。图片取样与质量计算均在生成计时结束后进行。预热不计入正式统计。
'''
(R/'report.md').write_text(report)
page='<!doctype html><meta charset="utf-8"><title>MiniWorld window 256</title><style>body{font-family:system-ui;max-width:1200px;margin:30px auto;padding:15px}img{max-width:100%}pre{white-space:pre-wrap;background:#f5f7fa;padding:20px}summary{padding:10px;cursor:pointer}a{color:#1463ac}</style><h1>MiniWorld 64 → 256 latent</h1>'
for name in ['window_size_latency','quality_vs_window','snapshot_quality_vs_window','quality_vs_progress']:page+=f'<img src="{name}.png">'
for entry in entries:
 ep=entry['episode_index'];page+=f'<h2>Episode {ep}</h2><p>{html.escape(entry["task"])}</p>'
 for mode in [64,256]:page+=f'<a href="episode_{ep:06d}_w{mode:03d}_comparison.mp4">GT / 窗口 {mode} 对比视频</a> · '
 for window in range(16,257,8):page+=f'<details><summary>扩展组活动窗口 {window} latent，同帧位置：GT / 上限64 / 上限256</summary><img loading="lazy" src="episode_{ep:06d}_active{window:03d}_comparison.png"></details>'
page+='<h2>完整报告</h2><pre>'+html.escape(report)+'</pre>'
(R/'index.html').write_text(page)
print(json.dumps(summary,indent=2))

# Compare history sources at identical frame positions and capacity.
free=R.parent/'window_256'
comparison=[]
for ep in [22694,46584,19464,27206]:
 for cap in [64,256]:
  for source,root in [('generated',free),('GT',R)]:
   q=pd.read_csv(root/f'episode_{ep:06d}_w{cap:03d}_quality_per_frame.csv')
   for end in range(28,1277,32):
    v=q[(q.frame>=max(1,end-31))&(q.frame<=end)]
    comparison.append(dict(episode_index=ep,capacity=cap,history_source=source,rgb_end=end,**v[['psnr_db','ssim','lpips_alex']].mean().to_dict()))
   if source=='GT':
    commits=json.loads((root/f'episode_{ep:06d}_w{cap:03d}_gt_history_commits.json').read_text())
    cond=[v for v in commits if v['branch']=='cond'];uncond=[v for v in commits if v['branch']=='uncond']
    assert [(v['start_latent'],v['end_latent_exclusive']) for v in cond]==[(v['start_latent'],v['end_latent_exclusive']) for v in uncond]
    assert all(v['start_latent']==i*2 and v['end_latent_exclusive']==i*2+2 for i,v in enumerate(cond))
c=pd.DataFrame(comparison);c.to_csv(R/'history_source_comparison_per_episode.csv',index=False)
c.groupby(['capacity','history_source','rgb_end'])[['psnr_db','ssim','lpips_alex']].mean().to_csv(R/'history_source_comparison.csv')
overall=[]
for cap in [64,256]:
 for source,root in [('generated',free),('GT',R)]:
  vals=[]
  for ep in [22694,46584,19464,27206]:
   q=pd.read_csv(root/f'episode_{ep:06d}_w{cap:03d}_quality_per_frame.csv');vals.append(q[q.frame>0][['psnr_db','ssim','lpips_alex']].mean())
  overall.append(dict(capacity=cap,history_source=source,**pd.DataFrame(vals).mean().to_dict()))
pd.DataFrame(overall).to_csv(R/'history_source_overall.csv',index=False)
fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True)
for axrow,cap in zip(axes,[64,256]):
 for ax,metric in zip(axrow,['psnr_db','ssim','lpips_alex']):
  for source in ['generated','GT']:
   z=c[(c.capacity==cap)&(c.history_source==source)].groupby('rgb_end')[metric].agg(['mean','std']);x=z.index.to_numpy();y=z['mean'].to_numpy();sd=z['std'].to_numpy()
   ax.plot(x,y,label=source+' history');ax.fill_between(x,y-sd,y+sd,alpha=.12)
  ax.set(title=f'Capacity {cap}: {metric}',xlabel='Generated RGB frames (excluding seed)');ax.legend()
fig.suptitle('GT vs. Generated Cached History: Same Frames, Same Capacity, Same Seeds')
fig.tight_layout()
fig.savefig(R/'gt_vs_generated_history.png',dpi=180);fig.savefig(R/'gt_vs_generated_history.pdf');plt.close(fig)
note='\n\n## GT 历史对照方法与限制\n\n本目录所有新实验在每次 t=0 历史提交时，将该 chunk 的生成 latent 替换为相应 GT VAE latent，分别重新计算条件和无条件 KV；保留同样的缓存淘汰、sink、位置策略和调度。当前去噪区以及所有预测输出不替换为 GT。使用 GT 的潜在编码属于 teacher-forcing/oracle 评估，不是可部署的纯自回归生成，也不等于去噪区内所有较早时间位置都使用 GT。\n\nGT 的 Wan VAE 时间编码具有因果性与否并未另行验证，故结果仅解释为给定 GT 编码历史的对照。比较使用相同 episode、帧位置、种子、动作、容量、inflight=8、effective_steps=40。256 仍是超出原训练64的长度外推。记录 gt_history_commits.json 可核查所有提交的帧索引、两个 CFG 分支、GT 与原生成 latent 的差异。\n\nhistory_source_overall.csv 为四条全序列均值，gt_vs_generated_history.png 为相同生成进度的 GT/生成历史质量曲线；这是判断历史来源影响的主要图。单条增长曲线仍混合预测距离和缓存长度，不单独解释为容量因果效应。提交中增加 GT 差异核查和同步开销，整段端到端耗时不能直接作为无开销性能对比；去噪前向 CUDA event 时间仍单独记录。\n'
with (R/'report.md').open('a') as f:f.write(note)
with (R/'index.html').open('a') as f:f.write('<img src="gt_vs_generated_history.png"><pre>'+html.escape(note)+'</pre>')
print(pd.DataFrame(overall).to_string(index=False))
