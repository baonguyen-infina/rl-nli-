"""Register, smoke-test, and start the run plus its once-only test watcher."""
import hashlib,json,os,pathlib,subprocess,sys,time
root=pathlib.Path('/content/rl-nli');os.chdir(root);sys.path.insert(0,str(root))
from common.io import read_json,write_json
run=os.environ['NATIVE_RUN_ID'];smoke='G0'+run+'P'
status=pathlib.Path('/content')/(run+'_setup_status.json')
def report(phase,**extra):
    value={'run':run,'phase':phase,'observed_at':time.time(),**extra}
    write_json(status,value);print(json.dumps(value),flush=True)
assert run in ('N20R20','N21R20')
out=root/'results'/(run+'_Qwen3.5-2B_s0')
assert not out.exists(),'Refuse to overwrite a previous run'
assert pathlib.Path('/content') .joinpath(run+'_test_registration.json').exists()
assert read_json('/content/'+run+'_test_registration.json')['candidate_fixed_before_test'] is True
assert read_json('/content/native_r9_cache_ready.json')['revision']=='15852e8c16360a2fea060d615a32b45270f8a8fc'
prepared=read_json('/content/native_r18_prepared_data_hashes.json')['prepared_files']
for name,item in prepared.items():
    path=root/name
    assert path.stat().st_size==item['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'], 'Prepared train/dev data changed'
env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1'}
with open('/content/'+smoke+'.log','w') as stream:
    result=subprocess.run([sys.executable,'-m','scripts.run','--run',smoke,'--seed','0','--phase','train'],
                          cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
if result.returncode:raise RuntimeError('One-step smoke run failed; inspect '+smoke+'.log')
smoke_out=root/'results'/(smoke+'_Qwen3.5-2B_s0')
gradient=read_json(smoke_out/'gradient_smoke.json')
needed=['policy_norm','answer_ce_norm','allocation_norm']
assert all(isinstance(gradient[k],(int,float)) and gradient[k]>0 for k in needed)
assert gradient['allocation_active_score_max_difference']==0.
reference=read_json(smoke_out/'reference_audit.json')
assert reference['frozen'] and reference['unchanged']
assert not read_json(smoke_out/'dropout_audit.json')['active_dropout_modules']
write_json('/content/'+run+'_setup_ready.json',{'passed':True,'smoke_run':smoke,
    'teacher':False,'test_accessed':False,'gradient':gradient,'reference_unchanged':True,
    'prepared_data_hashes_verified':True,'finished_at':time.time()})
test=subprocess.Popen([sys.executable,'/content/native_r20_test_worker.py'],cwd='/content',
    env={**env,'NATIVE_RUN_ID':run},stdout=open('/content/'+run+'_test_worker.log','w'),
    stderr=subprocess.STDOUT,start_new_session=True)
pipeline=subprocess.Popen([sys.executable,'/content/native_r20_pipeline.py'],cwd='/content',
    env={**env,'NATIVE_RUN_ID':run},stdout=open('/content/'+run+'_pipeline.log','w'),
    stderr=subprocess.STDOUT,start_new_session=True)
write_json('/content/'+run+'_launch_receipt.json',{'run':run,'pipeline_pid':pipeline.pid,
    'test_worker_pid':test.pid,'test_stage':'registered_waiting_for_training',
    'started_at':time.time(),'teacher':False})
report('launched',pipeline_pid=pipeline.pid,test_worker_pid=test.pid,test_accessed=False)
