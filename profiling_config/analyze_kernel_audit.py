"""Build disjoint kernel-time tables and figures for the diagnostic audit.

This analysis never scales isolated microbenchmarks into claimed E2E gains.
Linear scopes include casts/bias, and GPU activity counts include memsets.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path('/data2/jian/output/Miniworld/DROID_long_horizon/kernel_audit')
GROUPS={
 'Attention (incl. cast/cat)': ['attention_cast_cat_and_flash'],
 'Main Linear (incl. casts/bias)': ['qkv_gemm','attention_output_gemm','ffn_gate_up_gemm','ffn_down_gemm'],
 'Condition/modulation': ['modulation_gemm','block_modulation','conditioning_final_gemm','conditioning'],
 'RMSNorm + AdaLN pointwise': ['rmsnorm','adaln_pointwise'],
 'RoPE build + apply': ['rope_coefficients','rope_apply'],
 'Residual gate/add': ['block_residual_and_other'],
 'SwiGLU activation': ['ffn'],
 'Current KV writes': ['kv_current_write'],
}

def load(path):
    return json.loads((OUT/path).read_text())

def main():
    profile=load('torch_profile/summary.json')
    captures=profile['captures']
    results={}
    for name,cap in captures.items():
        rows=list(csv.DictReader((OUT/'torch_profile'/f'{name}_kernels.csv').open()))
        scopes={r['scope'].removeprefix('audit/'):r['gpu_ms'] for r in cap['scopes']}
        groups={key:sum(scopes.get(s,0) for s in parts) for key,parts in GROUPS.items()}
        groups['Other/layout']=cap['gpu_sum_ms']-sum(groups.values())
        results[name]=dict(gpu_activity_count=len(rows),compute_kernel_count=sum(r['category']=='kernel' for r in rows),
                           memset_count=sum(r['category']=='gpu_memset' for r in rows),
                           gpu_sum_ms=cap['gpu_sum_ms'],groups_ms=groups)
    (OUT/'kernel_breakdown.json').write_text(json.dumps(results,indent=2)+'\n')
    names=['denoise_history0_current16','denoise_history24_current16','denoise_history48_current16']
    keys=list(results[names[0]]['groups_ms'])
    colors=['#276FBF','#3A7D44','#A56DB4','#EAAA4A','#EF8354','#5B8E7D','#A8B0C9','#B8336A','#ABB3B7']
    fig,axes=plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios':[1,1.25]})
    bottom=np.zeros(3)
    for key,color in zip(keys,colors):
        vals=np.array([results[name]['groups_ms'][key] for name in names])
        axes[0].bar(np.arange(3),vals,bottom=bottom,label=key,color=color,width=.62)
        bottom+=vals
    for i,y in enumerate(bottom): axes[0].text(i,y+2,f'{y:.1f} ms',ha='center',fontsize=11)
    axes[0].set_xticks(range(3),['0','11.5','23.5'])
    axes[0].set_xlabel('History window (chunks; excludes sink/current)')
    axes[0].set_ylabel('Summed GPU activity duration per forward (ms)')
    axes[0].set_ylim(0,185)
    axes[0].set_title('Real rollout samples; current = 8 chunks')
    full=results[names[-1]]
    values=list(full['groups_ms'].values())
    axes[1].barh(np.arange(len(keys)),values,color=colors)
    axes[1].set_yticks(np.arange(len(keys)),keys)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0,87)
    axes[1].set_xlabel('GPU activity duration (ms)')
    axes[1].set_title('Full history: 23.5 chunks + fixed sink')
    for i,x in enumerate(values):axes[1].text(x+.5,i,f'{x:.2f} / {x/full["gpu_sum_ms"]*100:.1f}%',va='center',fontsize=9)
    fig.suptitle('MiniWorld / episode 002352 / unchanged sampling and precision',fontsize=15)
    fig.text(.5,.015,'Single instrumented calls, no VAE; disjoint GPU time, NOT end-to-end latency. Linear includes casts and bias.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.05,1,.94])
    fig.savefig(OUT/'kernel_breakdown.png',dpi=180)
    fig.savefig(OUT/'kernel_breakdown.pdf')
    plt.close(fig)

    actual=load('micro/kernel_candidates_actual_layout.json')['candidates']
    ordinary=load('micro/kernel_candidates.json')['candidates']
    selected=[next(c for c in (actual if name in ('norm_modulation','residual_gate_add') else ordinary) if c['name']==name)
              for name in ('norm_modulation','residual_gate_add','swiglu_activation','qk_norm_rope_cast')]
    labels=['Norm + AdaLN','Residual gate + add','SwiGLU activation','QK norm + cached RoPE + cast']
    fig,ax=plt.subplots(figsize=(11,5.7))
    before=[c['timings']['eager']['cuda_median_ms']*1000 for c in selected]
    after=[c['timings']['compiled_precision_emulation']['cuda_median_ms']*1000 for c in selected]
    y=np.arange(4)
    ax.barh(y-.18,before,height=.34,label='Current eager chain',color='#6b83a5')
    ax.barh(y+.18,after,height=.34,label='Isolated fusion prototype',color='#48a999')
    for i,(a,b) in enumerate(zip(before,after)):
        ax.text(a+5,i-.18,f'{a:.1f} us',va='center')
        ax.text(b+5,i+.18,f'{b:.1f} us  ({a/b:.2f}x)',va='center')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,max(before)*1.25)
    ax.set_xlabel('Median isolated operation time (microseconds)')
    ax.legend(loc='lower right')
    ax.set_title('Actual-layout fusion probes: kernel speedup is NOT E2E speedup')
    fig.text(.5,.015,'Synthetic activations + checkpoint weights. Residual/SwiGLU bitwise equal on tested input; Norm/QK differ.\nPrecision-emulation enabled; no production deployment or optimized full-video quality claim.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(OUT/'fusion_candidates.png',dpi=180);fig.savefig(OUT/'fusion_candidates.pdf');plt.close(fig)
    print(json.dumps(full,indent=2))


if __name__=='__main__':main()
