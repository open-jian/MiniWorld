"""Aggregate raw measurements; exclude warmup and separate fixed query sizes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/data2/jian/output/Miniworld/DROID_long_horizon')
OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
summaries=[];calls=[];chunks=[];evictions=[]
for p in sorted((ROOT/'episodes').glob('*/summary.json')):
 s=json.loads(p.read_text());summaries.append(s);ep=s['episode_index'];group=s['target_frames']
 f=pd.DataFrame(json.loads((p.parent/'forward_calls.json').read_text()));f['episode_index']=ep;f['target_frames']=group;f['worker']=s['worker'];f['raw_recorder_step']=f.step;f.loc[f.role=='commit','step']+=1;calls.append(f)
 timing=json.loads((p.parent/'stream_timing.json').read_text())['stream_timing']
 v=pd.DataFrame(timing['vae_chunk_events']);v['episode_index']=ep;v['target_frames']=group
 v['chunk_interval_sec']=v.end_sec.diff();v['rgb_frames']=4*(v.end_frame-v.start_frame);v.loc[v.start_frame==0,'rgb_frames']-=3
 v['chunk_rgb_fps']=v.rgb_frames/v.chunk_interval_sec
 c=f[(f.role=='denoise')&(f.branch=='cond')].set_index('step')
 v['cache_frames']=v.step_idx.map(c.cache_frames);v['current_frames']=v.step_idx.map(c.current_frames)
 v['cache_total_gib']=v.step_idx.map(c.cache_bytes)*2/2**30
 v['forward_gpu_ms']=v.step_idx.map(f.groupby('step').gpu_ms.sum())
 chunks.append(v)
 for e in json.loads((p.parent/'cache_evictions.json').read_text()):
  evictions.append(dict(episode_index=ep,step=e['step']+1,gpu_ms=e['gpu_ms'],before_bytes=e['before']['cache_bytes'],after_bytes=e['after']['cache_bytes']))
if not summaries:raise SystemExit('No complete episodes')
s=pd.DataFrame(summaries);f=pd.concat(calls,ignore_index=True);v=pd.concat(chunks,ignore_index=True);e=pd.DataFrame(evictions)
s.to_csv(OUT/'episodes.csv',index=False);f.to_csv(OUT/'forward_calls.csv',index=False);v.to_csv(OUT/'chunks.csv',index=False);e.to_csv(OUT/'evictions.csv',index=False)
# Compare fixed 16-frame query width so startup window growth does not confound cache growth.
d=f[(f.role=='denoise')&(f.current_frames==16)]
per=d.groupby(['episode_index','cache_frames']).gpu_ms.mean().reset_index()
bins=per.groupby('cache_frames').gpu_ms.agg(['count','mean','std','median','min','max']).reset_index()
bins.to_csv(OUT/'cache_growth_fixed_query.csv',index=False)
reg=[]
for ep,g in per.groupby('episode_index'):
 slope,intercept=np.polyfit(g.cache_frames,g.gpu_ms,1)
 residual=((g.gpu_ms-(intercept+slope*g.cache_frames))**2).sum()
 total=((g.gpu_ms-g.gpu_ms.mean())**2).sum()
 reg.append(dict(episode_index=ep,slope_ms_per_cached_latent=slope,intercept_ms=intercept,r_squared=1-residual/total,full_over_empty=g.loc[g.cache_frames==48,'gpu_ms'].iloc[0]/g.loc[g.cache_frames==0,'gpu_ms'].iloc[0]))
pd.DataFrame(reg).to_csv(OUT/'cache_growth_regression.csv',index=False)
groups=s.groupby('target_frames').agg(n=('episode_index','count'),actual_rgb_mean=('processed_rgb_frames','mean'),generation_mean_sec=('generation_sec','mean'),generation_std_sec=('generation_sec','std'),fps_mean=('generated_rgb_fps','mean'),fps_std=('generated_rgb_fps','std'),peak_alloc_max=('peak_allocated_bytes','max'),psnr_mean_db=('psnr_mean_excluding_seed_db','mean'))
groups['realtime_factor_vs_15fps']=15/groups.fps_mean
groups.to_csv(OUT/'length_groups.csv')
# Within-episode early and late plateau means; avoid tail and variable query widths.
plateau=[]
stable=d[(d.cache_frames==48)&(d.step < d.episode_index.map(f.groupby('episode_index').step.max())-35)]
for ep,g in stable.groupby('episode_index'):
 steps=sorted(g.step.unique());n=len(steps);a=g[g.step.isin(steps[:max(1,n//3)])].gpu_ms.mean();b=g[g.step.isin(steps[-max(1,n//3):])].gpu_ms.mean()
 plateau.append(dict(episode_index=ep,early_ms=a,late_ms=b,late_over_early=b/a))
plateau=pd.DataFrame(plateau);plateau.to_csv(OUT/'plateau_early_vs_late.csv',index=False)
# Additional latency percentiles and time components.
components=[]
for ep,g in f.groupby('episode_index'):
 c=v[v.episode_index==ep]; ee=e[e.episode_index==ep] if len(e) else e
 steady=c[(c.cache_frames==48)&(c.current_frames==16)&(c.step_idx<g.step.max()-35)]
 components.append(dict(episode_index=ep,first_output_sec=float(c.end_sec.iloc[0]),denoise_gpu_sec=float(g[g.role=='denoise'].gpu_ms.sum()/1000),commit_gpu_sec=float(g[g.role=='commit'].gpu_ms.sum()/1000),evict_gpu_sec=float(ee.gpu_ms.sum()/1000) if len(ee) else 0,vae_decode_wall_sec=float(c.duration_sec.sum()),full_cache_interval_mean_sec=float(steady.chunk_interval_sec.mean()),full_cache_interval_p50_sec=float(steady.chunk_interval_sec.quantile(.5)),full_cache_interval_p95_sec=float(steady.chunk_interval_sec.quantile(.95)),full_cache_interval_p99_sec=float(steady.chunk_interval_sec.quantile(.99))))
pd.DataFrame(components).to_csv(OUT/'latency_and_components.csv',index=False)
rng=np.random.default_rng(20261016)
ratios=plateau.late_over_early.to_numpy()
ci=np.quantile(rng.choice(ratios,(5000,len(ratios)),replace=True).mean(axis=1),[.025,.975])
(OUT/'plateau_bootstrap.json').write_text(json.dumps(dict(unit='episode',replicates=5000,mean_ratio=float(ratios.mean()),ci95_mean_ratio=ci.tolist()),indent=2))
plt.style.use('seaborn-v0_8-whitegrid')
fig,axes=plt.subplots(2,2,figsize=(14,10))
ax=axes[0,0];ax.errorbar(bins.cache_frames,bins['mean'],yerr=bins['std'],marker='o',capsize=2);ax.set(xlabel='Cached latent frames (fixed query = 16 latent frames)',ylabel='GPU ms per branch forward',title='KV growth vs denoiser latency (mean ± episode SD)')
ax2=ax.twiny();ax2.set_xlim(ax.get_xlim());ticks=[0,12,24,36,48];ax2.set_xticks(ticks,[f'{x*2*28*300*1536*6/2**30:.2f}' for x in ticks]);ax2.set_xlabel('Both CFG branches KV payload (GiB; FP32 K + BF16 V)')
for group,g in v.groupby('target_frames'):
 q=g.groupby('chunk_idx').chunk_interval_sec.median();axes[0,1].plot(q.index,q.values,label=str(group))
axes[0,1].set(xlabel='Output chunk index',ylabel='Median output interval (s)',title='Streaming output latency by video length');axes[0,1].legend(title='Nominal RGB frames')
for group,g in v.groupby('target_frames'):
 q=g.groupby('chunk_idx').cache_total_gib.median();axes[1,0].plot(q.index,q.values,label=str(group))
axes[1,0].set(xlabel='Output chunk index',ylabel='Both branches KV payload (GiB)',title='Measured KV size saturates')
axes[1,1].bar(groups.index.astype(str),groups.fps_mean,yerr=groups.fps_std,capsize=3);axes[1,1].set(xlabel='Nominal RGB frames',ylabel='Generated RGB frames/s',title='Whole rollout incl. stream VAE, excl. input encode / file IO')
fig.tight_layout();fig.savefig(OUT/'overview.png',dpi=180);fig.savefig(OUT/'overview.pdf');plt.close(fig)
low=bins.iloc[0];high=bins[bins.cache_frames==48].iloc[0]
maxbytes=int(f.cache_bytes.max())*2
stats=dict(completed=len(s),total_rgb_frames=int(s.processed_rgb_frames.sum()),total_generation_gpu_seconds=float(s.generation_sec.sum()),weighted_rgb_fps=float((s.processed_rgb_frames-1).sum()/s.generation_sec.sum()),max_persistent_kv_bytes_both_branches=maxbytes,max_persistent_kv_gib=maxbytes/2**30,first_fixed_query_cache_frames=int(low.cache_frames),first_fixed_query_forward_ms=float(low['mean']),full_cache_forward_ms=float(high['mean']),full_vs_early_forward_ratio=float(high['mean']/low['mean']),plateau_late_over_early_mean=float(plateau.late_over_early.mean()),plateau_late_over_early_median=float(plateau.late_over_early.median()),peak_allocated_gib=float(s.peak_allocated_bytes.max()/2**30),peak_reserved_gib=float(s.peak_reserved_bytes.max()/2**30),eviction_mean_ms_per_branch=float(e.gpu_ms.mean()) if len(e) else None)
(OUT/'aggregate.json').write_text(json.dumps(stats,indent=2))
rows='\n'.join(f"| {idx} | {r.n:.0f} | {r.actual_rgb_mean:.1f} | {r.generation_mean_sec:.2f} ± {r.generation_std_sec:.2f} | {r.fps_mean:.2f} ± {r.fps_std:.2f} | {r.peak_alloc_max/2**30:.2f} |" for idx,r in groups.iterrows())
report=f'''# MiniWorld 原版 DROID 长序列推理实测

完成 {len(s)}/50 条；总输出 {stats['total_rgb_frames']:,} RGB 帧。原始逐次测量保存在各 episode 目录，聚合 CSV 位于 analysis。

## 核心结果

- 双 CFG 分支持久 KV 张量最大 **{maxbytes/2**30:.4f} GiB**（{maxbytes:,} bytes）。这是 KV payload，不能与整个进程显存混为一谈。
- 固定 16 latent 查询帧，每分支前向在缓存 {int(low.cache_frames)} latent 时为 **{low['mean']:.2f} ms**，48 latent 满缓存时为 **{high['mean']:.2f} ms**，比值 **{stats['full_vs_early_forward_ratio']:.3f}×**。按每条视频先平均，再跨视频汇总；图中误差线是视频间标准差。
- 满缓存后，同一视频稳定区后 1/3 对前 1/3 的前向耗时比：均值 **{stats['plateau_late_over_early_mean']:.3f}×**，中位数 **{stats['plateau_late_over_early_median']:.3f}×**；以视频为单位 bootstrap 5000 次得到均值比值 95% 区间 **[{ci[0]:.3f}, {ci[1]:.3f}]**。已限制查询长度为 16 latent，避免尾段与启动窗口大小混淆。
- 整段生成吞吐量按总帧/总耗时计算：**{stats['weighted_rgb_fps']:.2f} RGB FPS**；包含流式 VAE 解码和官方采样循环，不含输入视频读取、输入 VAE 编码和 MP4 写盘。相对于原始 15 FPS 视频，实时因子约 **{15/stats['weighted_rgb_fps']:.2f}×**（生成一秒视频需要的墙钟秒数）。
- PyTorch 生成阶段峰值 allocated **{stats['peak_allocated_gib']:.2f} GiB**，reserved **{stats['peak_reserved_gib']:.2f} GiB**。输入视频全量 VAE 编码单独计时；这些值不是模型 KV 的大小。官方实现保留完整 latent 和已解码 RGB 输出，最终再拼接；因此总峰值随视频长度略增，即便 KV 已固定。

## 各长度组

| 目标帧数 | 样本数 | 实际处理帧数均值 | 生成秒数 均值 ± SD | RGB FPS 均值 ± SD | 峰值 allocated GiB |
|---|---|---|---|---|---|
{rows}

## KV-cache 到底是什么

默认 max_cache_chunks=24、chunk_size=2，因此常驻缓存最多 48 latent 帧。每 latent 空间 15×20=300 tokens；28 层、12 heads、head_dim=128。每分支每层 K/V 各 [1,12,14400,128]，实际 dtype 见 forward_calls.json。实测 K 为 FP32，V 为 BF16；原版 FP32 RMSNorm 权重及 RoPE 运算使 K 保持 FP32。双分支大小公式：2(CFG)×28×14400×1536×(4+2) bytes = 7,431,782,400 bytes = 6.9214 GiB。纯 BF16 的 4.6143 GiB 只是错误 dtype 假设下的估计，不是本次原版实测值。

满缓存时每层单分支 K 为 84.375 MiB、V 为 42.1875 MiB，合计 126.5625 MiB；28 层单分支 3.4607 GiB，CFG 双分支 6.9214 GiB。提交后、淘汰前，两分支缓存变量合计短暂为 50 latent，即 7.2098 GiB；此外还可能同时持有旧张量和新 KV，不能用这一数值推算进程显存峰值。

另有 8 个 chunk = 16 latent 帧同时去噪，当前前向会临时构造其 K/V、与历史拼接以及 attention mask；这部分不是持久 KV。缓存满后每次提交再加入 2 latent，单分支临时达到 50 latent，然后淘汰最老的非 sink 帧；保留第一个 latent 作为 sink。淘汰记录包含实际前后字节数和耗时。

因此 64 latent = 48 缓存 + 16 去噪，是注意力活动窗口上限，253 RGB 是该 latent 长度对应的整段 VAE 输出长度，**不是数据集视频长度上限，也不是最大 rollout 长度**。长视频可持续生成，但超过窗口后不会保留全部早期历史。

## 测量协议与限制

官方代码 commit 见 worker_*_environment.json；核心 miniworld/ 源码未改动。使用官方 MiniWorld_1b_droid.pt + Wan2.2_VAE.pth。仅外部适配原始 state_dict、改名后的数据路径及 PyAV 视频写出。batch=1，240×320，FP32 模型参数 + BF16 autocast，CFG=2，history=1，AR step=5，num_sampling_steps=100（官方调度实际 effective_steps=40），cache=24/inflight=8/sink=1。每张 GPU 正式测量前跑 64 latent warmup；CUDA events 计每次完整网络前向和淘汰。官方 collect_stream_timing 会在输出 chunk 时同步，计时开销与官方同步开销均包含在端到端结果中。

原始 recorder 的 commit/evict step 表示上一 denoise step；聚合 CSV 已加 1 对齐官方循环实际 step，保留 raw_recorder_step 可核对。

按用户要求清空此前所有结果后，仅使用物理 GPU 4、5、6、7 从头重跑全部 50 条。调度器每 2 秒检查 GPU 进程，在这四张卡中选择无其他计算进程的 GPU；进程记录见 logs/exclusive_process_checks.jsonl，硬件状态见 logs/gpu_telemetry.csv。

4 GPU 各运行独立 batch=1 工作进程，长度混合分配；不是多卡并行加速单条视频。统计是该硬件和软件环境下的测量，GPU 状态见 logs/gpu_telemetry.csv。每条仅运行一次，50条间差异不能替代同视频多次重复的置信区间；满缓存早晚配对比较用于观察长时漂移。

从每个视频第 0 帧开始，不降采样，保留原长度中最大 4n+1 帧（舍弃末尾 0–3 帧）。仅首帧为视觉条件，后续使用真实机器人动作。原版入口会先编码完整输入视频，我们保留此步骤并单独计时；未来真实帧不会作为生成视觉条件。对比视频左边真实、右边预测，15 FPS 播放。像素 PSNR 是参考诊断，不能直接代表世界模型感知质量、动作成功率或长期一致性；未测 FVD/LPIPS/机器人任务成功率。

## 文件

- `analysis/overview.png` / `.pdf`：主要图表。
- `analysis/episodes.csv`：每条数据、输入编码、生成、写盘、显存、PSNR。
- `analysis/forward_calls.csv`：每次前向角色、分支、查询长度、KV 字节数和 GPU 耗时。
- `analysis/chunks.csv`：输出逐 chunk 间隔、VAE 解码耗时和缓存大小。
- `analysis/cache_growth_fixed_query.csv`：控制查询长度后的缓存增长对比。
- `analysis/plateau_early_vs_late.csv`：满缓存后早晚配对对比；plateau_bootstrap.json 为按视频 bootstrap 区间。
- `analysis/latency_and_components.csv`：首输出延迟、满缓存 chunk 间隔 P50/P95/P99、前向/commit/evict/解码分项。
- `analysis/cache_growth_regression.csv`：每条视频缓存增长阶段的线性斜率、R² 与满/空缓存比。
- `analysis/distributions.png`：耗时、显存、稳定阶段漂移及逐帧 PSNR 诊断图。
- `episodes/episode_XXXXXX/`：预测、左右对比、原始计时、淘汰记录及逐帧 PSNR。
- `logs/`：完整 worker 日志与 GPU 遥测；每个进程先预热，预热不计入统计。
'''
(OUT/'report.md').write_text(report)
print(json.dumps(stats,indent=2))
# Distribution and per-episode diagnostics, complementary to mean curves.
fig,axes=plt.subplots(2,2,figsize=(14,10))
for group,g in s.groupby('target_frames'):
 axes[0,0].scatter(g.processed_rgb_frames,g.generation_sec,label=str(group))
axes[0,0].set(xlabel='Actual processed RGB frames',ylabel='Generation wall time (s)',title='Each point = one video');axes[0,0].legend()
axes[0,1].hist(plateau.late_over_early,bins=15);axes[0,1].axvline(1,color='r',ls='--');axes[0,1].set(xlabel='Late / early full-cache forward latency',ylabel='Videos',title='Same-video plateau drift')
for group,g in s.groupby('target_frames'):
 axes[1,0].scatter(g.processed_rgb_frames,g.peak_allocated_bytes/2**30,label=str(group))
axes[1,0].axhline(maxbytes/2**30,color='r',ls='--',label='Persistent KV');axes[1,0].set(xlabel='Actual processed RGB frames',ylabel='GiB',title='Total peak allocation vs persistent KV');axes[1,0].legend()
for group,g in s.groupby('target_frames'):
 curves=[]
 for ep in g.episode_index:
  q=pd.read_csv(ROOT/'episodes'/f'episode_{ep:06d}'/'per_frame_psnr.csv');curves.append(np.interp(np.linspace(0,1,101),q.frame/q.frame.max(),q.psnr_db))
 axes[1,1].plot(np.linspace(0,100,101),np.mean(curves,axis=0),label=str(group))
axes[1,1].set(xlabel='Rollout progress (%)',ylabel='PSNR (dB)',title='Pixel agreement diagnostic (not perceptual quality)');axes[1,1].legend()
fig.tight_layout();fig.savefig(OUT/'distributions.png',dpi=180);fig.savefig(OUT/'distributions.pdf');plt.close(fig)
# Local HTML gallery, all paths relative so moving the complete output stays usable.
import html
cards=[]
for row in summaries:
 ep=row['episode_index'];folder=f'../episodes/episode_{ep:06d}'
 cards.append(f'<tr><td>{ep}</td><td>{row["target_frames"]}</td><td>{row["processed_rgb_frames"]}</td><td>{row["generation_sec"]:.2f}</td><td>{row["generated_rgb_fps"]:.2f}</td><td><a href="{folder}/prediction.mp4">预测</a> · <a href="{folder}/comparison_gt_left_prediction_right.mp4">GT / 预测对比</a> · <a href="{folder}/summary.json">原始统计</a></td></tr>')
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MiniWorld DROID long horizon benchmark</title><style>body{max-width:1200px;margin:30px auto;font-family:system-ui;line-height:1.6;color:#182334;padding:16px}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;background:#f4f6f8;padding:24px}a{color:#1463ac}</style><h1>MiniWorld · DROID 长序列实测</h1>'''
page+=f'<p>完成 {len(s)}/50 条。双分支持久 KV：{maxbytes/2**30:.3f} GiB；总帧加权吞吐：{stats["weighted_rgb_fps"]:.2f} FPS。</p><img src="overview.png"><img src="distributions.png"><h2>逐视频输出</h2><p>对比视频：左侧真实、右侧预测；15 FPS。各项耗时含义与限制见下方报告。</p><table><tr><th>Episode</th><th>目标帧</th><th>处理帧</th><th>生成秒</th><th>RGB FPS</th><th>输出</th></tr>'+''.join(cards)+'</table><h2>完整报告</h2><pre>'+html.escape(report)+'</pre></html>'
(OUT/'index.html').write_text(page)
