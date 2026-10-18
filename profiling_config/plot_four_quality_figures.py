from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker
ROOT=Path('/data2/jian/output/Miniworld/DROID_long_horizon')
OUT=ROOT/'four_quality_figures';OUT.mkdir(exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black','xtick.color':'black','ytick.color':'black'})
metrics=['psnr_db','ssim','lpips_alex'];labels=['PSNR (dB) ↑','SSIM ↑','LPIPS-Alex ↓'];windows=list(range(0,121,4));colors=plt.colormaps['turbo'](np.linspace(0,1,31))
for source,title,slug,meaning in [('generated','Diffusion forcing','diffusion_forcing','Generated history / autoregressive inference'),('gt','Teacher forcing','teacher_forcing','GT history / teacher-forced cache')]:
 r=ROOT/f'history_chunk_sweep31_{source}_episode022694'
 a=pd.read_csv(r/'quality_by_window.csv');f=pd.read_csv(r/'quality_by_frame_all31.csv')
 assert a.history_window_chunks.tolist()==windows
 for kind in ['window','frame']:
  if kind=='window':
   fig,axes=plt.subplots(1,3,figsize=(17,6))
   for ax,m,label in zip(axes,metrics,labels):
    ax.plot(a.history_window_chunks,a[m],'o-',ms=4)
    ax.set(xlabel='Sliding history window (chunks)',ylabel=label,xticks=[0,20,40,60,80,100,120])
    ax.xaxis.label.set_color('black');ax.yaxis.label.set_color('black')
   plot_title='Window size → Mean quality'
   fig.tight_layout(rect=(0,.09,1,.78))
   note='Mean of per-frame metrics over RGB frames 1–1276 (seed frame 0 excluded).'
  else:
   fig,axes=plt.subplots(3,1,figsize=(17,14),sharex=True)
   for ax,m,label in zip(axes,metrics,labels):
    for w,c in zip(windows,colors):
     q=f[f.history_window_chunks==w];assert len(q)==1276
     ax.plot(q.frame,q[m],color=c,lw=.9,alpha=.9,label=str(w))
    assert len(ax.lines)==31
    ax.set(ylabel=label);ax.yaxis.label.set_color('black')
   axes[-1].set(xlabel='RGB frame index',xlim=(1,1276));axes[-1].xaxis.label.set_color('black')
   handles,names=axes[0].get_legend_handles_labels()
   legend=fig.legend(handles,names,title='History window (chunks): 31 curves per metric',loc='lower center',bbox_to_anchor=(.5,.025),ncol=11,fontsize=10)
   legend.get_title().set_color('black')
   plot_title='Frame index → Quality (31 curves per metric)'
   fig.tight_layout(rect=(0,.13,1,.89))
   note='Raw per-frame metrics; no smoothing. Seed frame 0 excluded.'
  heading=HPacker(children=[TextArea(title,textprops=dict(color='red',fontsize=20,fontweight='bold')),TextArea(' | '+plot_title,textprops=dict(color='black',fontsize=20,fontweight='bold'))],align='center',pad=0,sep=0)
  fig.add_artist(AnchoredOffsetbox(loc='upper center',child=heading,pad=0,frameon=False,bbox_to_anchor=(.5,.995),bbox_transform=fig.transFigure,borderpad=0))
  y=.925 if kind=='window' else .95
  fig.text(.5,y,f'Episode 22694 | {meaning}',ha='center',color='black',fontsize=13)
  fig.text(.5,y-(.045 if kind=='window' else .023),'Window = sliding history only: 0, 4, …, 120 chunks; excludes sink and in-flight denoising.',ha='center',color='black',fontsize=12)
  fig.text(.5,y-(.09 if kind=='window' else .046),'Fixed sink = 1 latent | In-flight denoising = 8 chunks | 1 chunk = 2 latent | Effective steps = 40',ha='center',color='black',fontsize=12)
  fig.text(.5,.025 if kind=='window' else .008,note,ha='center',color='black',fontsize=10)
  for ext in ['png','pdf']:fig.savefig(OUT/f'{slug}_quality_vs_{kind}.{ext}',dpi=200)
  plt.close(fig)
(OUT/'index.html').write_text('<meta charset="utf-8"><title>Four quality figures</title><style>body{max-width:1700px;margin:auto}img{width:100%}</style>'+''.join(f'<img src="{s}_quality_vs_{k}.png">' for s in ['diffusion_forcing','teacher_forcing'] for k in ['window','frame']))
(OUT/'labels.txt').write_text('Diffusion forcing: generated-history autoregressive evaluation. Teacher forcing: GT-history cache evaluation. Both use the diffusion-forcing sampler; group labels distinguish history source. No distillation training. Only forcing group names are red; other text is black. Data unchanged; episode 22694.\n')
print(OUT)
