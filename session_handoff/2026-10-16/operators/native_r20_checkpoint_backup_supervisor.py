"""Back up both Colab arms with bounded memory and no local checkpoint files.

All model computation and archive creation stay on Colab. This local operator
only reads metadata and streams archives to the owned private draft release.
"""
import argparse,json,pathlib,subprocess,sys,time
sys.path.insert(0,'/private/tmp')
from native_github_archive_relay import relay,verify_destination,SESSIONS

ROOT=pathlib.Path('/Users/infina/Documents/rl-nli/artifacts/analysis')
STATE=pathlib.Path('/private/tmp/native_github_archive_state.json')
KEY='/private/tmp/native_r13_ssh_key'
LANES={'N20R20':'N20R20','N21R20':'N21R20'}

def log_event(observation,now,last_log_at,failure_stage_seen):
    stage=observation.get('failure_stage')
    if (stage and stage!=failure_stage_seen) or now-last_log_at>=900:
        return 'logs-'+str(int(now))
    return None

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def remote(run,source,timeout=100):
    command=['ssh','-T','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=15',
        '-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null','-o','LogLevel=ERROR',
        '-o','ProxyCommand=colab ssh --proxy-mode -s '+SESSIONS[run]+' -i '+KEY,
        'root@colab-runtime','python -']
    result=subprocess.run(command,input=source,capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError('Remote archive operation exited '+str(result.returncode))
    return json.loads(result.stdout)

def observe(run):
    return remote(run,"""import json,pathlib
root=pathlib.Path('/content/rl-nli');run=%r;lane=%r;out=root/'results'/(run+'_Qwen3.5-2B_s0')
def read(path):return json.loads(path.read_text()) if path.exists() else {}
ready=[]
for step in [100,200,300,400]:
    folder=out/'grpo_ckpts'/('checkpoint-'+str(step))
    if read(folder/'trainer_state.json') and (folder/'adapter_model.safetensors').is_file():ready.append('checkpoint-'+str(step))
verified=read(out/'test_verification.json')
final=(verified.get('split')=='test' and verified.get('n')==1000 and
    read(out/'.test_claim/receipt.json').get('status')=='completed' and
    read(out/'cost_test.json').get('status')=='complete' and read(out/'cost.json').get('status')=='complete')
# Final archives are handled by the separate once-only test archiver.
pipeline=read(pathlib.Path('/content/'+run+'_pipeline_status.json'))
test=read(pathlib.Path('/content/'+run+'_test_status.json'))
failure=('training_failed' if pipeline.get('phase')=='failed' or read(out/'cost.json').get('status') in ['failed','aborted']
    else 'test_failed' if test.get('stage')=='test_failed' else None)
print(json.dumps({'run':run,'ready':ready,'test_verified':bool(final),'failure_stage':failure,
    'pipeline_finished':read(pathlib.Path('/content/'+run+'_pipeline_finished.json')).get('complete') is True,
    'test_terminal':test.get('stage') in ['test_verified_archived','test_verified_backup_pending','test_failed']}))
""" % (run,LANES[run]))

def pack(run,event):
    return remote(run,"""import json,subprocess,sys
result=subprocess.run([sys.executable,'/content/native_r20_checkpoint_backup_pack.py','--run',%r,'--event',%r],capture_output=True,text=True)
if result.returncode:raise SystemExit(result.returncode)
print(json.dumps(json.loads(result.stdout)))
""" % (run,event),timeout=150)

def commit_receipt(run,event,receipt):
    # The final guard is checked on Colab again before unblocking its parent.
    return remote(run,"""import json,pathlib,time
run=%r;event=%r;receipt=%r;lane=%r
root=pathlib.Path('/content/rl-nli');out=root/'results'/(run+'_Qwen3.5-2B_s0')
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\\n');temporary.replace(path)
assert receipt['provider']=='github-private-draft-release' and receipt['repo_private'] and receipt['release_draft']
assert receipt['github_digest']=='sha256:'+receipt['sha256']
save(out/'github_backup'/(event+'.receipt.json'),receipt)
if event=='final':
    verified=json.loads((out/'test_verification.json').read_text())
    assert verified['split']=='test' and verified['n']==1000
    assert json.loads((out/'.test_claim/receipt.json').read_text())['status']=='completed'
    assert json.loads((out/'cost_test.json').read_text())['status']=='complete'
    assert json.loads((out/'cost.json').read_text())['status']=='complete'
    receipt.update(run_id=run,event='complete',scope='best adapter, frozen sources, all captured training/evaluation logs, and verified test predictions',committed_at=time.time())
    save(root/'artifacts/remote_receipts'/(run+'-complete.json'),receipt)
    save(pathlib.Path('/content/native_log_archives')/(lane+'-terminal.receipt.json'),receipt)
print(json.dumps({'run':run,'event':event,'receipt_committed':True,'provider':receipt['provider']}))
""" % (run,event,receipt,LANES[run]))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');options=parser.parse_args()
    state=json.loads(STATE.read_text());verify_destination(state)
    final=set();final_archived=set();verified=set();last_log_at={};failure_seen={};deadline=time.time()+12*3600
    while time.time()<deadline and len(final)<len(LANES):
        for run in LANES:
            if run in final:continue
            try:
                observation=observe(run)
                save(ROOT/'github_backup_status'/(run+'.json'),{**observation,'observed_at':time.time()})
                now=time.time();event=log_event(observation,now,last_log_at.get(run,0),failure_seen.get(run))
                terminal=(observation['pipeline_finished'] and observation['test_terminal'])
                if terminal:event='logs-'+str(int(now))
                if event:
                    spec=pack(run,event)
                    save(ROOT/'github_backup_specs'/(run+'_'+event+'.json'),spec)
                    receipt=relay(spec,state)
                    save(ROOT/'github_backup_receipts'/(run+'_'+event+'.json'),receipt)
                    commit_receipt(run,event,receipt)
                    last_log_at[run]=now;failure_seen[run]=observation.get('failure_stage')
                    print(json.dumps({'logs_backed_up':run,'event':event,'bytes':spec['bytes'],
                        'terminal_snapshot':terminal,'sha256':spec['sha256']}),flush=True)
                    if terminal:final.add(run);continue
                for event in observation['ready']:
                    if (run,event) in verified:continue
                    spec=pack(run,event)
                    save(ROOT/'github_backup_specs'/(run+'_'+event.replace('-','')+'.json'),spec)
                    receipt=relay(spec,state)
                    save(ROOT/'github_backup_receipts'/(run+'_'+event.replace('-','')+'.json'),receipt)
                    committed=commit_receipt(run,event,receipt)
                    verified.add((run,event))
                    print(json.dumps({**committed,'bytes':spec['bytes'],'sha256':spec['sha256'],
                        'model_bytes_written_locally':0,'max_buffer_bytes':receipt['max_buffer_bytes']}),flush=True)
                    if event=='final':final_archived.add(run)
            except Exception as exc:
                # Exception messages from network libraries can contain private URLs.
                failure={'run':run,'error_type':type(exc).__name__,'observed_at':time.time()}
                save(ROOT/'github_backup_status'/(run+'_last_error.json'),failure)
                print(json.dumps({'retry_backup':True,**failure}),flush=True)
        if options.once:break
        if len(final)<len(LANES):time.sleep(120)
    save(ROOT/'github_r20_backup_supervisor_result.json',{'final_backups_verified':sorted(final),
        'checkpoint_and_final_receipts':len(verified),'local_ml':False,'model_bytes_written_locally':0,
        'finished_at':time.time(),'all_done':len(final)==len(LANES)})

if __name__=='__main__':main()
