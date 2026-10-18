from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/window_256')
a=pd.read_csv(R/'latency_by_window.csv');a=a[a.capacity==256].copy()
a['cache_latents']=a.active_window-16
a['resident_sink_latents']=np.where(a.cache_latents>0,1,0)
a['history_window_chunks']=(a.cache_latents-a.resident_sink_latents)/2
cols=['history_window_chunks','gpu_ms_mean','gpu_ms_std','episodes','kv_gib_both','cache_latents','resident_sink_latents','active_window']
a[cols].to_csv(R/'latency_by_history_chunks.csv',index=False)
plt.style.use('seaborn-v0_8-whitegrid')
fig,ax=plt.subplots(figsize=(12,6.5))
x=a.history_window_chunks.to_numpy();y=a.gpu_ms_mean.to_numpy();sd=a.gpu_ms_std.to_numpy()
ax.plot(x,y,'o-',ms=4,color='#1477AB');ax.fill_between(x,y-sd,y+sd,color='#1477AB',alpha=.16)
ax.set(xlabel='Sliding history window (chunks; excludes sink and in-flight denoising)',ylabel='GPU latency per CFG step (two model forwards), ms',title='MiniWorld 1B | Historical window size vs. inference latency',xlim=(0,121),xticks=np.arange(0,121,20))
fig.text(.08,.07,'Fixed in-flight region = 8 chunks; sink = 1 latent once cached. Mean ± SD across 4 episodes.',fontsize=10)
fig.text(.08,.04,'History chunks = (cached latent frames − resident sink) / 2. Original maximum = 119.5 chunks.',fontsize=10)
fig.text(.08,.01,'Replotted original measurements. Excludes VAE, cache commits/evictions and I/O; not isolated memory-operation latency.',fontsize=9)
fig.tight_layout(rect=(0,.10,1,1))
for ext in ['png','pdf']:fig.savefig(R/f'latency_vs_history_chunks.{ext}',dpi=200)
(R/'latency_vs_history_chunks_notes.txt').write_text('Original generated-history capacity-256 run, 4 episodes. Only original blue fixed-query branch retained. In-flight = 16 latent = 8 chunks. Subtract resident sink (1 latent when cache nonempty), then divide by 2. Empty cache maps to 0, and has not yet committed a sink; this is not a sink-only steady-state measurement. Historical sample positions are 0, 3.5, 7.5, ...,119.5 chunks because the old cache cap included the 1-latent sink. Do not round to integer or relabel as new 0–120-chunk sweep. No inference rerun.\n')
print(a[cols].iloc[[0,-1]].to_string(index=False))
