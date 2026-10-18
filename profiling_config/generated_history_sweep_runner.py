import os,signal,time,json,subprocess
from pathlib import Path
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/history_chunk_sweep31_generated_episode022694');B=Path(__file__).resolve().parent
legacy={};active={};finished=set();stamp=int(time.time())
def done(w):return (R/f'episode_022694_history{w:03d}chunks_complete.json').exists()
def alive(pid):
 try:
  state=Path(f'/proc/{pid}/stat').read_text().split()[2];return state!='Z'
 except FileNotFoundError:return False
def status(state,**extra):
 v=dict(status=state,time=time.time(),episode_index=22694,completed_windows=sum(done(w) for w in range(0,121,4)),total_windows=31,active={**{g:v['pid'] for g,v in legacy.items()},**{g:p.pid for g,p in active.items()}},gpus=list(range(8)),**extra)
 t=R/'status.tmp';t.write_text(json.dumps(v,indent=2));t.replace(R/'status.json')
def occupied():
 ids={row.split(',')[1].strip():int(row.split(',')[0]) for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()}
 v={g:[] for g in range(8)}
 for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines():
  pid,uid=row.split(',');v[ids[uid.strip()]].append(int(pid))
 return v
for v in legacy.values():os.kill(v['pid'],signal.SIGCONT)
try:
 while True:
  for g,v in list(legacy.items()):
   if done(v['window']):
    if alive(v['pid']):os.kill(v['pid'],signal.SIGTERM)
    while alive(v['pid']):time.sleep(.1)
    del legacy[g];print('LEGACY WINDOW FINISHED',g,v['window'],flush=True)
   elif not alive(v['pid']):raise RuntimeError(f'Legacy worker GPU {g} exited before checkpoint')
  for g,p in list(active.items()):
   if p.poll() is not None:
    del active[g]
    if p.returncode:raise RuntimeError(f'GPU {g} exited {p.returncode}')
    finished.add(g)
  if all(done(w) for w in range(0,121,4)) and not active and not legacy:break
  busy=occupied()
  for g in range(8):
   if g in active or g in legacy or g in finished or busy[g]:continue
   # Slot 0 remains on GPU4, preserving its GT export and state namespace.
   slot=g-4 if g>=4 else g+4
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
   with (R/f'gpu{g}_queue8_{stamp}.log').open('w') as f:
    active[g]=subprocess.Popen(['/home/jian/.local/share/mamba/envs/miniworld/bin/python','-u',str(B/'generated_history_sweep_worker.py'),'--episode','22694','--worker',str(slot)],env=env,stdout=f,stderr=subprocess.STDOUT)
   print('START QUEUE',g,active[g].pid,flush=True)
  status('running');time.sleep(1)
 status('analyzing')
 with (R/'analysis.log').open('w') as f:subprocess.run(['/home/jian/anaconda3/bin/python3',str(B/'generated_history_sweep_analyze.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 status('complete');print('ALL 31 WINDOWS + BOTH FIGURES COMPLETE',flush=True)
except Exception as e:
 status('failed',error=str(e));raise
