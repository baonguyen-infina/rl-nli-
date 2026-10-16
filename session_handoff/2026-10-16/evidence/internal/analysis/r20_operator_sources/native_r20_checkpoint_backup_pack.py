"""Pack a checkpoint and full captured logs on Colab; emit only small metadata."""
import argparse,hashlib,io,json,pathlib,re,sys,tarfile,time
parser=argparse.ArgumentParser();parser.add_argument('--run',choices=['N20R20','N21R20'],required=True)
parser.add_argument('--event',required=True)
args=parser.parse_args();root=pathlib.Path('/content/rl-nli');sys.path.insert(0,str(root))
if not re.fullmatch(r'checkpoint-(100|200|300|400)|final|logs-[0-9]+',args.event):parser.error('Unknown archive event')
from scripts.remote_artifacts import event_ready,snapshot_files,digest
from scripts.remote_logs import collect_logs
run=args.run;lane=run
smoke='G0'+run+'P'
out=root/'results'/(run+'_Qwen3.5-2B_s0')
if args.event.startswith('logs-'):
    event='complete-metadata'
elif args.event=='final':
    receipt=json.loads((out/'.test_claim/receipt.json').read_text())
    assert receipt['status']=='completed' and receipt['n']==1000
    assert json.loads((out/'cost.json').read_text())['status']=='complete'
    assert json.loads((out/'cost_test.json').read_text())['status']=='complete'
    assert json.loads((out/'test_verification.json').read_text())['split']=='test'
    event='complete'
else:
    event=args.event.replace('checkpoint-','checkpoint-adapter-',1)
    assert event_ready(out,event),'Checkpoint is not completely saved'
stage=pathlib.Path('/content/native_github_staging');stage.mkdir(exist_ok=True)
name=run+'_s0_'+args.event+'.tar.gz';archive=stage/name;spec_path=archive.with_suffix('.json')
if archive.exists():
    spec=json.loads(spec_path.read_text());assert digest(archive,'sha256')==spec['sha256']
    print(json.dumps(spec),flush=True);raise SystemExit(0)
files={str(path.relative_to(root)):path for path in snapshot_files(out,event)}
files.update(collect_logs('/content',[smoke,run],lane))
for suffix in ['.log','_test.log','_pipeline.log','_test_worker.log','_failure.log','_test_failure.log','_setup_status.json','_pipeline_status.json','_test_status.json','_launch_receipt.json','_checkpoint100_status.json','_checkpoint100_observer.log','_checkpoint100_failure.log']:
    path=pathlib.Path('/content')/(run+suffix)
    if path.is_file():files['runtime/'+path.name]=path
path=pathlib.Path('/content')/(smoke+'.log')
if path.is_file():files['runtime/'+path.name]=path
snapshot=out/'source_snapshot.json'
for suffix in ['_checkpoint200_status.json','_checkpoint200_observer.log','_checkpoint200_failure.log']:
    path=pathlib.Path('/content')/(run+suffix)
    if path.is_file():files['runtime/'+path.name]=path
if snapshot.is_file():
    for source_name in json.loads(snapshot.read_text()):
        path=root/source_name
        if path.is_file():files['source/'+source_name]=path
operator_paths=['/content/native_'+lane+'_pipeline.py','/content/native_'+lane+'_final_test.py',
    '/content/native_'+lane+'_launch.py','/content/native_'+lane+'_test_registration.json',
    '/content/native_r18_prepared_data_hashes.json','/content/native_checkpoint_backup_pack.py']
operator_paths+=['/content/native_resume_source.tar.gz'] if lane=='r18' else ['/content/native_r19_source.tar.gz']
operator_paths += ['/content/native_r20_pipeline.py','/content/native_r20_test_worker.py','/content/native_r20_launch.py','/content/native_r20_checkpoint_observer.py','/content/native_r20_checkpoint_backup_pack.py']
operator_paths.append('/content/native_r20_checkpoint200_observer.py')
for name_path in operator_paths:
    path=pathlib.Path(name_path)
    if path.is_file():files['operator/'+path.name]=path
manifest={'run':run,'event':args.event,'created_at':time.time(),
    'scope':('all available experiment logs and source; no weights; credentials excluded'
        if args.event.startswith('logs-') else 'selected adapter plus all available experiment logs and source; credentials excluded'),
    'live_snapshot':args.event!='final','files':{}}
temporary=archive.with_suffix('.tmp')
with tarfile.open(temporary,'w:gz',compresslevel=1) as packed:
    for name_path,path in sorted(files.items()):
        assert not path.is_symlink()
        assert not any(word in path.name.casefold() for word in ['credentials','targets','restore_manifest','oauth','token.json'])
        payload=path.read_bytes();entry=tarfile.TarInfo(name_path);entry.size=len(payload);entry.mtime=int(time.time())
        packed.addfile(entry,io.BytesIO(payload))
        manifest['files'][name_path]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    payload=json.dumps(manifest,indent=2).encode();entry=tarfile.TarInfo('backup_manifest.json');entry.size=len(payload)
    packed.addfile(entry,io.BytesIO(payload))
temporary.replace(archive)
spec={'run':run,'event':args.event,'name':archive.name,'path':str(archive),
    'bytes':archive.stat().st_size,'sha256':digest(archive,'sha256'),'md5':digest(archive,'md5'),
    'files':len(manifest['files']),'created_at':time.time(),'model_bytes_written_locally':0}
spec_path.write_text(json.dumps(spec,indent=2)+'\n')
print(json.dumps(spec),flush=True)
