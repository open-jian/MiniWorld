from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/500_frames')
f=pd.read_csv(R/'forward_calls.csv')
f=f[(f.role=='denoise')&(f.current_frames==16)]
g=f.groupby(['episode_index','step','cache_frames']).agg(step_gpu_ms=('gpu_ms','sum'),branches=('branch','nunique')).reset_index()
assert (g.branches==2).all()
e=g.groupby(['episode_index','cache_frames']).step_gpu_ms.mean().reset_index()
s=e.groupby('cache_frames').step_gpu_ms.agg(['mean','std','count']).reset_index()
s['active_latent_frames']=s.cache_frames+16;s['active_chunks']=s.active_latent_frames/2
s=s.rename(columns={'mean':'cfg_step_gpu_ms_mean','std':'cfg_step_gpu_ms_episode_std','count':'videos'})
s.to_csv(R/'window_size_latency.csv',index=False)
x=s.active_latent_frames.to_numpy();y=s.cfg_step_gpu_ms_mean.to_numpy();sd=s.cfg_step_gpu_ms_episode_std.to_numpy()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.right':False,'axes.spines.top':False})
fig,ax=plt.subplots(figsize=(10,6.3));fig.subplots_adjust(left=.12,right=.96,bottom=.23,top=.78)
ax.fill_between(x,y-sd,y+sd,color='#2374AB',alpha=.16,label='Across-video ±1 SD')
ax.plot(x,y,'o-',color='#176A9C',lw=2.3,ms=4.5,label='Mean of 10 videos')
ax.axvline(64,color='#B45309',ls='--',lw=1.2)
ax.set(xlim=(14,67),ylim=(0,750),xticks=[16,24,32,40,48,56,64],xlabel='Active window size (latent frames = cached history + 16 in-flight)',ylabel='DiT denoising step GPU latency (ms)')
ax.grid(axis='y',alpha=.2)
top=ax.secondary_xaxis('top',functions=(lambda x:x-16,lambda x:x+16));top.set_xlabel('Cached history (latent frames)');top.set_xticks([0,8,16,24,32,40,48])
ax.annotate(f'{y[0]:.1f} ms',xy=(x[0],y[0]),xytext=(18,240),arrowprops={'arrowstyle':'-','color':'#176A9C'},color='#176A9C',weight='bold')
ax.annotate(f'{y[-1]:.1f} ms\nFull window: 64 latent / 32 chunks',xy=(64,y[-1]),xytext=(38,690),arrowprops={'arrowstyle':'-','color':'#B45309'},color='#92400E',fontsize=11)
ax.legend(loc='lower right',frameon=False,fontsize=10)
fig.suptitle('MiniWorld 1B · Active Window Size vs. Latency',fontsize=17,weight='bold',y=.97)
fig.text(.12,.10,'One denoising step = conditional + unconditional network forwards (CFG = 2).',fontsize=10,color='#374151')
fig.text(.12,.065,'Batch 1 · 240×320 · fixed 16-latent query · excludes VAE, cache commit/eviction and file I/O.',fontsize=9.5,color='#374151')
fig.text(.12,.03,'Measured during cache growth; configured capacity stays fixed at 64 latent frames.',fontsize=9.5,color='#374151')
fig.savefig(R/'window_size_latency.png',dpi=200);fig.savefig(R/'window_size_latency.pdf');plt.close(fig)
print(s.iloc[[0,-1]].to_string(index=False))
