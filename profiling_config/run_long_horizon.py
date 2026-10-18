import subprocess,os,time,json
from pathlib import Path
root=Path('/data2/jian/output/Miniworld/DROID_long_horizon');script=Path(__file__).with_name('benchmark_long_horizon.py');python='/home/jian/.local/share/mamba/envs/miniworld/bin/python'
processes=[None]*8
for i in [1,2,3,4,5,6,7,0]:
 if i==0:
  while Path("/proc/1080328").exists():time.sleep(5)
 log=open(root/f'logs/worker_{i}.log','w')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 p=subprocess.Popen([python,'-u',str(script),'--worker',str(i)],stdout=log,stderr=subprocess.STDOUT,env=env)
 processes[i]=p
with open(root/'logs/gpu_telemetry.csv','w') as f:
 f.write('timestamp, index, uuid, utilization.gpu_pct, memory.used_MiB, temperature.gpu_C, power.draw_W, clocks.sm_MHz\n')
 while any(p.poll() is None for p in processes):
  r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader,nounits'],capture_output=True,text=True)
  f.write(r.stdout);f.flush()
  (root/'status.json').write_text(json.dumps(dict(time=time.time(),workers=[dict(worker=i,pid=p.pid,exit_code=p.poll()) for i,p in enumerate(processes)],completed=len(list((root/'episodes').glob('*/summary.json')))),indent=2))
  time.sleep(10)
(root/'status.json').write_text(json.dumps(dict(time=time.time(),workers=[dict(worker=i,pid=p.pid,exit_code=p.returncode) for i,p in enumerate(processes)],completed=len(list((root/'episodes').glob('*/summary.json')))),indent=2))
print('Exit codes:',[p.returncode for p in processes],flush=True)
