import os,json,time,subprocess,collections
from pathlib import Path
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/generated_history_5episodes_3seeds_window0_24');B=Path(__file__).resolve().parent
cfg=json.loads((R/'experiment.json').read_text());jobs=collections.deque((ep,base) for base in cfg['seed_bases'] for ep in cfg['episodes']);active={};stamp=int(time.time())
def status(state,**extra):
 count=len(list(R.glob('episode_*_history???chunks_complete.json')))
 v=dict(status=state,time=time.time(),completed_runs=count,total_runs=375,queued_episode_seed_jobs=len(jobs),active={g:dict(pid=p.pid,episode=ep,seed=base+ep) for g,(p,ep,base) in active.items()},**extra)
 t=R/'status.tmp';t.write_text(json.dumps(v,indent=2));t.replace(R/'status.json')
def busy():
 ids={row.split(',')[1].strip():int(row.split(',')[0]) for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()};out=set()
 for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines():out.add(ids[row.split(',')[1].strip()])
 return out
try:
 while jobs or active:
  for g,(p,ep,base) in list(active.items()):
   if p.poll() is not None:
    if p.returncode:raise RuntimeError(f'GPU {g}, episode {ep}, seed {base+ep}: exit {p.returncode}')
    del active[g];print('JOB COMPLETE',g,ep,base+ep,flush=True)
  occupied=busy()
  for g in range(8):
   if not jobs:break
   if g in occupied or g in active:continue
   ep,base=jobs.popleft();seed=ep+base
   if (R/f'episode_{ep:06d}_seed{seed}_complete.json').exists():continue
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
   with (R/f'gpu{g}_episode{ep}_seed{seed}_{stamp}.log').open('w') as f:
    p=subprocess.Popen(['/home/jian/.local/share/mamba/envs/miniworld/bin/python','-u',str(B/'robust_window_worker.py'),'--episode',str(ep),'--seed-base',str(base),'--worker',str(g)],env=env,stdout=f,stderr=subprocess.STDOUT)
   active[g]=(p,ep,base);print('START',g,ep,seed,p.pid,flush=True)
  status('running');time.sleep(5)
 assert len(list(R.glob('episode_*_history???chunks_complete.json')))==375
 status('analyzing')
 with (R/'analysis.log').open('w') as f:subprocess.run(['/home/jian/anaconda3/bin/python3',str(B/'robust_window_analyze.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 status('complete');print('375 RUNS AND ANALYSIS COMPLETE',flush=True)
except Exception as e:status('failed',error=str(e));raise
