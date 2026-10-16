"""Create a small-metadata, integrity-checked test archive on Colab itself."""
import hashlib,io,json,os,pathlib,sys,tarfile,time
run=os.environ['NATIVE_RUN_ID'];assert run in ('N20R20','N21R20')
root=pathlib.Path('/content/rl-nli');out=root/'results'/(run+'_Qwen3.5-2B_s0')
status=pathlib.Path('/content')/(run+'_test_status.json')
summary=json.loads((out/'test_verification.json').read_text())
claim=json.loads((out/'.test_claim/receipt.json').read_text())
assert summary['split']=='test' and summary['n']==1000 and claim['status']=='completed'
assert json.loads((out/'cost.json').read_text())['status']=='complete'
assert json.loads((out/'cost_test.json').read_text())['status']=='complete'
assert json.loads((status).read_text())['stage']=='test_verified_backup_pending'
paths=[]
for name in ['config.yaml','cost.json','cost_test.json','ckpt_scores.json','metrics_dev.json',
             'predictions_dev.jsonl','metrics_test.json','predictions_test.jsonl',
             'locked_test_protocol.json','test_registration.json','test_verification.json',
             'source_snapshot.json','execution_provenance.json','training_source_hashes.json',
             'dataset_test_receipt.json','eval_runtime_test.json','train_log.csv','rollouts_train.jsonl',
             'dropout_audit.json','reference_audit.json','gradient_smoke.json','group_audit.json','environment.json',
             'checkpoint200_review_policy.json']:
    path=out/name
    if path.is_file():paths.append((path,run+'_Qwen3.5-2B_s0/'+name))
for base in ['best_adapter','ckpt_eval','initial_eval','internal_diagnostics']:
    folder=out/base
    if folder.exists():
        paths.extend((p,run+'_Qwen3.5-2B_s0/'+p.relative_to(out).as_posix()) for p in folder.rglob('*') if p.is_file())
paths.extend((pathlib.Path('/content')/name,'runtime/'+name) for name in [
    run+'.log',run+'_test.log',run+'_pipeline.log',run+'_test_worker.log',run+'_setup_status.json',
    run+'_pipeline_status.json',run+'_test_status.json',run+'_launch_receipt.json',
    'G0'+run+'P.log',run+'_checkpoint100_status.json',run+'_checkpoint100_observer.log',
    run+'_checkpoint100_failure.log',run+'_checkpoint200_status.json',run+'_checkpoint200_observer.log',
    run+'_checkpoint200_failure.log'] if (pathlib.Path('/content')/name).is_file())
for name in json.loads((out/'source_snapshot.json').read_text()):
    path=root/name
    if path.is_file():paths.append((path,'source/'+name))
assert len(paths)>20
stage=pathlib.Path('/content/native_github_staging');stage.mkdir(exist_ok=True)
name=run+'_s0_final.tar.gz';archive=stage/name;temporary=stage/(name+'.tmp')
manifest={'run':run,'event':'final','created_at':time.time(),'scope':'verified test predictions and metrics, selected adapter, all run logs and source; credentials and raw test data excluded','files':{}}
with tarfile.open(temporary,'w:gz',compresslevel=1) as packed:
    seen=set()
    for path,arcname in sorted(paths,key=lambda item:item[1]):
        assert arcname not in seen;seen.add(arcname)
        assert not any(word in path.name.casefold() for word in ['credentials','restore_manifest','oauth','token.json','targets'])
        payload=path.read_bytes();info=tarfile.TarInfo(arcname);info.size=len(payload);info.mtime=int(time.time())
        packed.addfile(info,io.BytesIO(payload))
        manifest['files'][arcname]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    payload=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('backup_manifest.json');info.size=len(payload)
    packed.addfile(info,io.BytesIO(payload))
temporary.replace(archive)
def digest(path,algorithm):
    h=hashlib.new(algorithm)
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
spec={'run':run,'event':'final','name':name,'path':str(archive),'bytes':archive.stat().st_size,
      'sha256':digest(archive,'sha256'),'md5':digest(archive,'md5'),'files':len(manifest['files']),
      'created_at':time.time(),'model_bytes_written_locally':0}
(stage/(run+'_s0_final.json')).write_text(json.dumps(spec,indent=2)+'\n')
print(json.dumps(spec),flush=True)
