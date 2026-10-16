"""Relay a verified test archive from Colab to the owned private draft release."""
import json,pathlib,subprocess,sys,time
sys.path.insert(0,'/private/tmp')
from native_github_archive_relay import relay,validate_spec
from native_github_archive_supervisor import remote

ROOT=pathlib.Path('/Users/infina/Documents/rl-nli/artifacts/analysis')
STATE=pathlib.Path('/private/tmp/native_github_archive_state.json')
CLI='/Users/infina/.local/share/uv/tools/google-colab-cli/bin/colab'
run=sys.argv[1]
session={'N20R20':'rl-nli-resume-r18-g4','N21R20':'rl-nli-r19-native-swap-2b-g4'}[run]
source=pathlib.Path('/private/tmp/native_r20_pack.py')
subprocess.run([CLI,'upload','--session',session,str(source),'/content/native_r20_pack.py'],check=True)
result=subprocess.run([CLI,'exec','--session',session,'--file',str(source),'--timeout','1800',
                       '--env','NATIVE_RUN_ID='+run],capture_output=True,text=True,timeout=1900)
if result.returncode:raise RuntimeError('Remote packing failed with exit '+str(result.returncode))
lines=[line.strip() for line in result.stdout.splitlines() if line.strip().startswith('{')]
if not lines:raise RuntimeError('Remote packer returned no archive manifest')
spec=json.loads(lines[-1]);validate_spec(spec)
ROOT.joinpath('github_backup_specs').mkdir(parents=True,exist_ok=True)
ROOT.joinpath('github_backup_receipts').mkdir(parents=True,exist_ok=True)
ROOT.joinpath('github_backup_specs',run+'_final.json').write_text(json.dumps(spec,indent=2)+'\n')
state=json.loads(STATE.read_text())
receipt=relay(spec,state)
ROOT.joinpath('github_backup_receipts',run+'_final.json').write_text(json.dumps(receipt,indent=2)+'\n')
commit={**receipt,'run_id':run,'event':'complete','scope':'selected adapter, frozen sources, all captured run logs, and verified test predictions','committed_at':time.time()}
src='''import json,pathlib,time
run=%r;receipt=%r
root=pathlib.Path('/content/rl-nli');out=root/'results'/(run+'_Qwen3.5-2B_s0')
state=pathlib.Path('/content')/(run+'_test_status.json')
value=json.loads(state.read_text());assert value['stage']=='test_verified_backup_pending'
test=value['summary'];assert test['split']=='test' and test['n']==1000
assert receipt['provider']=='github-private-draft-release' and receipt['repo_private'] and receipt['release_draft']
assert receipt['github_digest']=='sha256:'+receipt['sha256']
def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
 temp.write_text(json.dumps(obj,indent=2)+'\\n');temp.replace(path)
save(out/'github_backup'/'final.receipt.json',receipt)
value.update(stage='test_verified_archived',archive_id=receipt['id'],archive_bytes=receipt['size'],archive_sha256=receipt['sha256'],off_vm_backup_verified=True,archived_at=time.time())
save(state,value)
save(pathlib.Path('/content/native_log_archives')/(run+'-terminal.receipt.json'),receipt)
save(root/'artifacts/remote_receipts'/(run+'-complete.json'),receipt)
print(json.dumps({'run':run,'stage':value['stage'],'archive_id':receipt['id'],'off_vm_backup_verified':True}))
'''%(run,receipt)
committed=remote(run,src,timeout=100)
ROOT.joinpath('github_backup_receipts',run+'_commit.json').write_text(json.dumps(committed,indent=2)+'\n')
print(json.dumps({'run':run,'spec':{k:spec[k] for k in ('name','bytes','sha256','files')},
    'receipt':{k:receipt[k] for k in ('id','size','sha256','github_digest','max_buffer_bytes')},
    'remote_commit':committed}),flush=True)
