import json,os,pathlib,tarfile
run=os.environ['NATIVE_RUN_ID'];assert run in ('N20R20','N21R20')
paths=sorted(pathlib.Path('/content/native_github_staging').glob(run+'_s0_logs-*.tar.gz'))
assert paths,'No existing periodic archive'
p=paths[-1]
with tarfile.open(p,'r:gz') as archive:
 manifest=json.load(archive.extractfile('backup_manifest.json'))
 candidates=[(name,value) for name,value in manifest['files'].items() if name.endswith('/rollouts_train.jsonl')]
 found=[(name,value) for name,value in candidates if (run+'_Qwen3.5-2B_s0/rollouts_train.jsonl') in name]
print(json.dumps({'run':run,'archive':p.name,'contains_rollouts':bool(found),'rollout_bytes':found[0][1]['bytes'] if found else 0,'candidate_paths':[name for name,value in candidates]}),flush=True)
assert len(found)==1 and found[0][1]['bytes']>0,'Periodic log archive omitted training rollouts'
