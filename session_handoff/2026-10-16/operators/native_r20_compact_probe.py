import json,os,pathlib,subprocess
run=os.environ['NATIVE_RUN_ID'];assert run in ('N20R20','N21R20')
base=pathlib.Path('/content')
state={}
for key,suffix in [('pipeline','_pipeline_status.json'),('checkpoint100','_checkpoint100_status.json'),('checkpoint200','_checkpoint200_status.json'),('test','_test_status.json')]:
 p=base/(run+suffix)
 if p.exists():state[key]=json.loads(p.read_text())
processes=subprocess.run(['ps','-eo','pid,ppid,etime,stat,args'],capture_output=True,text=True,check=True).stdout.splitlines()
state['actual_train_processes']=[x.strip() for x in processes if '-m scripts.train_worker' in x and '--run '+run+' ' in x]
state['actual_eval_processes']=[x.strip() for x in processes if '-m eval.predict' in x and '--run '+run+' ' in x]
receipt=base/(run+'_launch_receipt.json')
if receipt.exists():
 for key,pid in json.loads(receipt.read_text()).items():
  if key.endswith('_pid'):
   state[key]=subprocess.run(['ps','-p',str(pid),'-o','pid=,etime=,stat=,comm='],capture_output=True,text=True).stdout.strip()
for step in (100,200):
 receipt=base/(run+'_checkpoint'+str(step)+'_observer_receipt.json')
 if receipt.exists():
  pid=json.loads(receipt.read_text())['observer_pid']
  state['observer_'+str(step)+'_process']=subprocess.run(['ps','-p',str(pid),'-o','pid=,etime=,stat=,comm='],capture_output=True,text=True).stdout.strip()
gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True)
state['gpu']=gpu.stdout.strip();print(json.dumps({'run':run,**state}),flush=True)
