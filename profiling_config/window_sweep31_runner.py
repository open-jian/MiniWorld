import os,sys,time,json,subprocess
from pathlib import Path
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/window_sweep31_gt_episode022694');B=Path(__file__).resolve().parent
PYTHON='/home/jian/.local/share/mamba/envs/miniworld/bin/python'
active={};finished=set();run_id=int(time.time())
def status(state,**extra):
 v=dict(status=state,time=time.time(),episode_index=22694,completed_windows=len(list(R.glob('episode_022694_w???_complete.json'))),total_windows=31,active={g:p.pid for g,p in active.items()},**extra)
 tmp=R/'status.tmp';tmp.write_text(json.dumps(v,indent=2));tmp.replace(R/'status.json')
def gpu_processes():
 devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
 ids={v.split(',')[1].strip():int(v.split(',')[0]) for v in devices.splitlines()}
 pids={g:[] for g in range(4,8)}
 lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
 for line in lines.splitlines():
  pid,uid=line.split(',');g=ids[uid.strip()]
  if g in pids:pids[g].append(int(pid))
 return pids
try:
 while len(finished)<4:
  occupied=gpu_processes()
  for slot,g in enumerate(range(4,8)):
   if slot in finished:continue
   if g in active:
    p=active[g]
    if p.poll() is not None:
     del active[g]
     if p.returncode:raise RuntimeError(f'GPU {g} worker exited {p.returncode}')
     finished.add(slot);continue
    if any(pid!=p.pid for pid in occupied[g]):
     p.terminate();p.wait();del active[g]
     print('PAUSED for other process on GPU',g,flush=True)
    continue
   if occupied[g]:continue
   if (R/f'episode_022694_worker{slot}_complete.json').exists():finished.add(slot);continue
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
   with (R/f'gpu{g}_{run_id}_{int(time.time())}.log').open('w') as log:
    active[g]=subprocess.Popen([PYTHON,'-u',str(B/'window_sweep31_worker.py'),'--episode','22694','--worker',str(slot)],env=env,stdout=log,stderr=subprocess.STDOUT)
   print('START',g,active[g].pid,flush=True)
  status('running')
  time.sleep(5)
 status('analyzing')
 with (R/'analysis.log').open('w') as log:subprocess.run(['/home/jian/anaconda3/bin/python3',str(B/'window_sweep31_analyze.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
 status('complete');print('ALL 31 WINDOWS + BOTH FIGURES COMPLETE',flush=True)
except Exception as e:
 status('failed',error=str(e));raise
