import json,os,pathlib,subprocess,sys,time
run=os.environ['NATIVE_RUN_ID'];assert run in ('N20R20','N21R20')
base=pathlib.Path('/content');receipt=base/(run+'_checkpoint200_observer_receipt.json')
assert not receipt.exists(),'Observer already registered; inspect its recorded PID'
log=base/(run+'_checkpoint200_observer.log')
with log.open('w') as stream:
 p=subprocess.Popen([sys.executable,'/content/native_r20_checkpoint200_observer.py'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1'})
value={'run':run,'observer_pid':p.pid,'started_at':time.time(),'test_accessed':False,'checkpoint':200}
receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
