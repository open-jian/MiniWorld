import os,json,time,subprocess,signal
from pathlib import Path
ROOT=Path('/data2/jian/output/Miniworld/DROID_long_horizon/500_frames')
BASE=Path(__file__).resolve().parent
PYTHON='/home/jian/.local/share/mamba/envs/miniworld/bin/python'
GPUS=[4,5,6,7]
active={};attempt=0;last_report=0;last_telemetry=0
claims=ROOT/'.claims';claims.mkdir(exist_ok=True)
entries=json.loads((ROOT/'selection.json').read_text())['videos'];assert len(entries)==10
query=lambda fields:subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True)
uuids={line.split(',')[1].strip():int(line.split(',')[0]) for line in query('index,uuid').splitlines()}
def release(pid,discard=False):
 for p in claims.glob('*.json'):
  data=json.loads(p.read_text())
  if data['pid']!=pid:continue
  if discard:
   for artifact in ROOT.glob(f"episode_{data['episode_index']:06d}_*"):artifact.unlink()
  p.unlink()
with (ROOT/'gpu_process_checks.jsonl').open('w') as audit,(ROOT/'gpu_telemetry.csv').open('w') as telemetry:
 telemetry.write('timestamp,index,uuid,utilization.gpu_pct,memory.used_MiB,temperature.gpu_C,power.draw_W,clocks.sm_MHz,sw_thermal,hw_thermal,sw_power_cap\n')
 while True:
  gpu_pids={i:[] for i in GPUS}
  raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
  for line in raw.splitlines():
   pid,uuid=line.split(',');gpu=uuids[uuid.strip()]
   if gpu in GPUS:gpu_pids[gpu].append(int(pid))
  audit.write(json.dumps(dict(time=time.time(),gpu_pids=gpu_pids,our_pids={g:p.pid for g,p in active.items()}))+'\n');audit.flush()
  for gpu,p in list(active.items()):
   code=p.poll()
   if code is not None:
    release(p.pid);del active[gpu]
    if code!=0:
     (ROOT/'status.json').write_text(json.dumps(dict(status='failed',gpu=gpu,exit_code=code)))
     raise RuntimeError(f'Worker GPU {gpu} failed with {code}')
   elif any(pid!=p.pid for pid in gpu_pids[gpu]):
    print('Pausing due to other GPU process:',gpu,flush=True);p.terminate();p.wait();release(p.pid,True);del active[gpu]
  count=len(list(ROOT.glob('episode_*_summary.json')))
  if time.time()-last_telemetry>=10:
   telemetry.write(query('timestamp,index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap'));telemetry.flush();last_telemetry=time.time()
  if count>=last_report+5 or (count==10 and not active):
   with (ROOT/'analysis.log').open('a') as fp:subprocess.run(['python3',str(BASE/'analyze_500_frames.py')],stdout=fp,stderr=subprocess.STDOUT,check=True)
   last_report=count
  if count==10 and not active:break
  pending=[e for e in entries if not (ROOT/f"episode_{e['episode_index']:06d}_summary.json").exists() and not (claims/f"{e['episode_index']}.json").exists()]
  for gpu in GPUS:
   if not pending or gpu in active or gpu_pids[gpu]:continue
   attempt+=1
   fp=open(ROOT/f'gpu{gpu}_attempt{attempt}.log','w')
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
   active[gpu]=subprocess.Popen([PYTHON,'-u',str(BASE/'benchmark_500_frames.py'),'--queue','--worker',str(gpu)],stdout=fp,stderr=subprocess.STDOUT,env=env)
   print('Started GPU',gpu,'PID',active[gpu].pid,flush=True)
  (ROOT/'status.json').write_text(json.dumps(dict(time=time.time(),status='running',completed=count,total=10,allowed_gpus=GPUS,active={g:p.pid for g,p in active.items()}),indent=2))
  time.sleep(2)
claims.rmdir()
(ROOT/'status.json').write_text(json.dumps(dict(time=time.time(),status='complete',completed=10,total=10,allowed_gpus=GPUS),indent=2))
print('All 10 videos and final analysis complete.',flush=True)
