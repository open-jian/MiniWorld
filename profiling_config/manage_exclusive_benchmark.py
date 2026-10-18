import os,time,json,subprocess,shutil,signal
from pathlib import Path
root=Path('/data2/jian/output/Miniworld/DROID_long_horizon');claims=root/'claims';claims.mkdir(exist_ok=True)
entries=json.loads((root/'selection.json').read_text())['videos']
entries=sorted(entries,key=lambda x:(next(i for i,e in enumerate([r for r in entries if r['target_frames']==x['target_frames']]) if e['episode_index']==x['episode_index']),x['target_frames']))
allowed_gpus=[4,5,6,7]
active={};children={};attempt=0;last_report=0;last_telemetry=0
query=lambda fields:subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True)
uuids={x.split(',')[1].strip():int(x.split(',')[0]) for x in query('index,uuid').splitlines()}
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False

def release(pid,contended=False):
 for p in claims.glob('*.json'):
  x=json.loads(p.read_text())
  if x['pid']!=pid:continue
  dest=root/'episodes'/f"episode_{x['episode_index']:06d}"
  if contended and dest.exists() and not (dest/'summary.json').exists():
   moved=root/'contended_runs'/f"{dest.name}_pid{pid}_{int(time.time())}";shutil.move(str(dest),str(moved))
  p.unlink()

with (root/'logs/exclusive_process_checks.jsonl').open('a') as audit:
 while True:
  raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
  gpu_pids={i:[] for i in range(8)}
  for line in raw.splitlines():
   a,b=line.split(',');gpu_pids[uuids[b.strip()]].append(int(a))
  audit.write(json.dumps(dict(time=time.time(),gpu_pids=gpu_pids,our_pids=active))+'\n');audit.flush()
  for gpu,pid in list(active.items()):
   if not alive(pid):release(pid);active.pop(gpu);continue
   external=[p for p in gpu_pids[gpu] if p!=pid]
   if external:
    print('CONTENTION',gpu,pid,external,flush=True);os.kill(pid,signal.SIGTERM);time.sleep(1);release(pid,True);active.pop(gpu)
  count=len(list((root/'episodes').glob('*/summary.json')))
  if time.time()-last_telemetry>=10:
   fields='timestamp,index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap'
   with (root/'logs/gpu_telemetry.csv').open('a') as fp:fp.write(query(fields))
   last_telemetry=time.time()
  if count>=last_report+5 or (count==50 and not active):
   with (root/'logs/analysis.log').open('a') as fp:
    result=subprocess.run(['python3',str(Path(__file__).with_name('analyze_long_horizon.py'))],stdout=fp,stderr=subprocess.STDOUT)
   if result.returncode==0:last_report=count
  if count==50 and not active:
   if last_report!=50:raise RuntimeError('Final analysis failed; see logs/analysis.log')
   break
  # Launch only on an unoccupied device; claims prevent duplicates.
  pending=[r for r in entries if not (root/'episodes'/f"episode_{r['episode_index']:06d}"/'summary.json').exists() and not (claims/f"{r['episode_index']}.json").exists()]
  for gpu in allowed_gpus:
   if not pending or gpu in active or gpu_pids[gpu]:continue
   attempt+=1
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
   fp=open(root/f'logs/queue_gpu{gpu}_attempt{attempt}.log','w')
   p=subprocess.Popen(['/home/jian/.local/share/mamba/envs/miniworld/bin/python','-u',str(Path(__file__).with_name('benchmark_long_horizon.py')),'--queue','--worker',str(gpu)],stdout=fp,stderr=subprocess.STDOUT,env=env)
   children[p.pid]=p;active[gpu]=p.pid
   print('START',gpu,p.pid,flush=True)
  for p in children.values():p.poll()
  (root/'managed_status.json').write_text(json.dumps(dict(time=time.time(),allowed_gpus=allowed_gpus,completed=count,active=active,pending_unclaimed=len(pending)),indent=2))
  time.sleep(2)
(root/'managed_status.json').write_text(json.dumps(dict(time=time.time(),allowed_gpus=allowed_gpus,completed=50,active={},status='complete'),indent=2))
print('ALL 50 COMPLETE',flush=True)
