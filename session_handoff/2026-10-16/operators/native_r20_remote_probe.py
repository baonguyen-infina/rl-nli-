import json,os,pathlib
import subprocess
run=os.environ['NATIVE_RUN_ID'];base=pathlib.Path('/content')
for name in [run+'_setup_status.json',run+'_pipeline_status.json',run+'_test_status.json',run+'_launch_receipt.json',run+'_checkpoint100_status.json',run+'_checkpoint200_status.json']:
 p=base/name
 if p.exists():
  try: print(json.dumps(json.loads(p.read_text()),ensure_ascii=False))
  except Exception: print(name,'unreadable')
p=base/(('G0'+run+'P')+'.log')
if p.exists(): print('smoke_log_tail',p.read_text(errors='replace')[-500:])
p=base/(run+'.log')
if p.exists(): print('main_log_tail',p.read_text(errors='replace')[-1000:])
launch=base/(run+'_launch_receipt.json')
if launch.exists():
 for key in ('pipeline_pid','test_worker_pid'):
  pid=json.loads(launch.read_text()).get(key)
  if pid: print('process',key,subprocess.run(['ps','-p',str(pid),'-o','pid=,etime=,stat=,comm='],capture_output=True,text=True).stdout.strip())
gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True)
if gpu.returncode==0: print('gpu',gpu.stdout.strip())
receipt=base/(run+'_checkpoint100_observer_receipt.json')
if receipt.exists():
 pid=json.loads(receipt.read_text())['observer_pid']
 print('process observer_pid',subprocess.run(['ps','-p',str(pid),'-o','pid=,etime=,stat=,comm='],capture_output=True,text=True).stdout.strip())
