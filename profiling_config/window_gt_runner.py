import os,json,time,subprocess
from pathlib import Path
R=Path('/data2/jian/output/Miniworld/DROID_long_horizon/window_256_gt_history');B=Path(__file__).resolve().parent
assignments={4:22694,5:46584,6:19464,7:27206};active={};attempts={g:0 for g in assignments}
query=lambda fields:subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True)
uuids={s.split(',')[1].strip():int(s.split(',')[0]) for s in query('index,uuid').splitlines()}
last_telemetry=0
with (R/'gpu_processes.jsonl').open('a') as audit,(R/'gpu_telemetry.csv').open('w') as telemetry:
 telemetry.write('timestamp,index,utilization_pct,memory_MiB,temperature_C,power_W,sm_clock_MHz\n')
 while True:
  pids={g:[] for g in assignments}
  for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines():
   pid,uuid=line.split(',');g=uuids[uuid.strip()]
   if g in pids:pids[g].append(int(pid))
  audit.write(json.dumps(dict(time=time.time(),gpu_pids=pids,our_pids={g:p.pid for g,p in active.items()}))+'\n');audit.flush()
  for g,p in list(active.items()):
   if p.poll() is not None:
    del active[g]
    if p.returncode!=0:
     (R/'status.json').write_text(json.dumps(dict(status='failed',gpu=g,episode=assignments[g],exit_code=p.returncode)))
     raise RuntimeError(f'GPU {g} worker failed, inspect its log')
   elif any(pid!=p.pid for pid in pids[g]):
    p.terminate();p.wait();del active[g];ep=assignments[g]
    state=json.loads((R/f'episode_{ep:06d}_state.json').read_text())
    if state['phase'].startswith(('generation_','writing_')):
     mode=state['phase'].split('_')[-1]
     for a in R.glob(f'episode_{ep:06d}_{mode}_*'):a.unlink()
    print('PAUSED due to other GPU process',g,flush=True)
  for g,ep in assignments.items():
   if g in active or pids[g] or (R/f'episode_{ep:06d}_complete.json').exists():continue
   attempts[g]+=1;log=open(R/f'gpu{g}_episode{ep}_attempt{attempts[g]}.log','w')
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
   active[g]=subprocess.Popen(['/home/jian/.local/share/mamba/envs/miniworld/bin/python','-u',str(B/'window_gt_worker.py'),'--episode',str(ep)],env=env,stdout=log,stderr=subprocess.STDOUT)
   print('START',g,ep,active[g].pid,flush=True)
  complete=len(list(R.glob('episode_*_complete.json')))
  (R/'status.json').write_text(json.dumps(dict(time=time.time(),status='running',completed_episodes=complete,total_episodes=4,active={g:p.pid for g,p in active.items()},gpu_episode_assignments=assignments),indent=2))
  if time.time()-last_telemetry>=10:
   telemetry.write(query('timestamp,index,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm'));telemetry.flush();last_telemetry=time.time()
  if complete==4 and not active:break
  time.sleep(2)
with (R/'analysis.log').open('w') as f:subprocess.run(['python3',str(B/'window_gt_analyze.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
(R/'status.json').write_text(json.dumps(dict(time=time.time(),status='complete',completed_episodes=4,total_episodes=4,gpu_episode_assignments=assignments),indent=2))
print('ALL FOUR EPISODES + ANALYSIS COMPLETE',flush=True)
