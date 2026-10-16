"""Persist Colab artifacts to preauthorized, file-scoped Drive upload URLs.

This module runs only on the remote VM. It receives no account OAuth token.
Upload URLs must stay outside run configs, source archives and logged output.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import time


def event_ready(run_dir, event):
    root=Path(run_dir)
    try:
        if event.startswith('checkpoint-'):
            checkpoint_event=event.replace('checkpoint-adapter-','checkpoint-',1)
            for folder in ['sft_ckpts','grpo_ckpts']:
                p=root/folder/checkpoint_event/'trainer_state.json'
                if p.exists():
                    json.loads(p.read_text())
                    return True
            return False
        if event in ('complete','complete-metadata'):
            p=root/'cost.json'
            return p.exists() and json.loads(p.read_text()).get('status') in ('complete','failed','aborted','trained')
        if event=='initial_eval':
            p=root/'initial_eval/metrics_dev.json'
            return p.exists() and isinstance(json.loads(p.read_text()),dict)
        if event.startswith('progress-'):
            p=root/'train_log.csv'
            if not p.exists():return False
            with p.open() as f:rows=list(csv.DictReader(f))
            steps=[int(float(r['step'])) for r in rows if r.get('step')]
            return bool(steps) and max(steps)>=int(event.split('-')[1])
    except (ValueError,OSError):
        return False  # A trainer may still be writing the marker.
    raise ValueError('Unknown artifact event')


def snapshot_files(run_dir,event):
    root=Path(run_dir)
    excluded={'sft_ckpts','grpo_ckpts','best_adapter','final_adapter'}
    files=[]
    for p in root.rglob('*'):
        if not p.is_file() or p.is_symlink():continue
        relative=p.relative_to(root)
        if any(part in excluded for part in relative.parts):continue
        if p.suffix in ('.json','.yaml','.csv') or (p.suffix=='.jsonl' and (event=='complete' or p.name!='rollouts_train.jsonl')):
            files.append(p)
    if event.startswith('checkpoint-'):
        compact=event.startswith('checkpoint-adapter-')
        checkpoint_event=event.replace('checkpoint-adapter-','checkpoint-',1)
        for folder in ['sft_ckpts','grpo_ckpts']:
            checkpoint=root/folder/checkpoint_event
            if not checkpoint.exists():continue
            if compact:
                files.extend(p for p in checkpoint.iterdir() if p.is_file() and not p.is_symlink()
                             and p.suffix in ('.safetensors','.json','.model'))
            else:
                files.extend(p for p in checkpoint.rglob('*') if p.is_file() and not p.is_symlink()
                             and (folder=='grpo_ckpts' or p.suffix not in ('.pt','.bin')))
    if event=='complete':
        best=root/'best_adapter'
        if best.exists():
            files.extend(p for p in best.iterdir() if p.is_file() and p.suffix in ('.safetensors','.json','.model'))
    return sorted(set(files))


def pack_snapshot(run_dir,event,output):
    root=Path(run_dir)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():return output  # Resumed uploads must keep identical bytes.
    temp=output.with_suffix(output.suffix+'.tmp')
    with tarfile.open(temp,'w:gz',compresslevel=1) as archive:
        for p in snapshot_files(root,event):
            archive.add(p,arcname=str(Path(root.name)/p.relative_to(root)),recursive=False)
    temp.replace(output)
    return output


def digest(path,name):
    h=hashlib.new(name)
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def put_resumable(path,uri):
    import requests
    from urllib.parse import urlparse
    parsed=urlparse(uri)
    if parsed.scheme!='https' or parsed.hostname!='www.googleapis.com':
        raise ValueError('Upload target must be a Google Drive session')
    size=Path(path).stat().st_size
    offset=0
    try:
        with Path(path).open('rb') as f:
            # Query first, so retries after a network interruption resume safely.
            r=requests.put(uri,data=b'',headers={'Content-Range':f'bytes */{size}','Content-Length':'0'},timeout=60)
            if r.status_code in (200,201):
                result=r.json()
                if int(result['size'])!=size or result.get('md5Checksum')!=digest(path,'md5'):
                    raise RuntimeError('Remote archive integrity mismatch')
                return result
            if r.status_code!=308:raise RuntimeError('Upload session status HTTP '+str(r.status_code))
            if 'Range' in r.headers:offset=int(r.headers['Range'].rsplit('-',1)[1])+1
            while offset<size:
                f.seek(offset);chunk=f.read(8*1024*1024)
                r=requests.put(uri,data=chunk,headers={'Content-Length':str(len(chunk)),
                               'Content-Range':f'bytes {offset}-{offset+len(chunk)-1}/{size}'},timeout=90)
                if r.status_code in (200,201):
                    result=r.json()
                    if int(result['size'])!=size or result.get('md5Checksum')!=digest(path,'md5'):
                        raise RuntimeError('Remote archive integrity mismatch')
                    return result
                if r.status_code!=308:raise RuntimeError('Archive upload HTTP '+str(r.status_code))
                if 'Range' not in r.headers:raise RuntimeError('Upload acknowledged no bytes')
                offset=int(r.headers['Range'].rsplit('-',1)[1])+1
        raise RuntimeError('Upload ended without completion acknowledgment')
    except requests.RequestException as e:
        # Requests exceptions can include a secret upload-session URL.
        raise RuntimeError('Archive transport error: '+type(e).__name__) from None


def watch(root,manifest_path):
    root=Path(root)
    manifest=json.loads(Path(manifest_path).read_text())
    receipts=root/'artifacts/remote_receipts';receipts.mkdir(parents=True,exist_ok=True)
    staging=root/'artifacts/remote_archives';staging.mkdir(parents=True,exist_ok=True)
    while True:
        for run,targets in manifest['runs'].items():
            dirs=list((root/'results').glob(run+'_*_s0'))
            if len(dirs)!=1:continue
            run_dir=dirs[0]
            for event,uri in targets.items():
                receipt=receipts/f'{run}-{event}.json'
                if receipt.exists() or not event_ready(run_dir,event):continue
                try:
                    archive=staging/f'{run}-{event}.tar.gz'
                    pack_snapshot(run_dir,event,archive)
                    result=put_resumable(archive,uri)
                    result.update(run_id=run,event=event,sha256=digest(archive,'sha256'),saved_at=time.time())
                    temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(receipt)
                    archive.unlink()
                    print(json.dumps({'saved':run,'event':event,'file_id':result['id'],'bytes':result['size']}),flush=True)
                except Exception as e:
                    print(json.dumps({'retry':run,'event':event,'error_type':type(e).__name__}),flush=True)
        if all(all((receipts/f'{run}-{event}.json').exists() for event in targets
                   if event in ('complete','complete-metadata')) for run,targets in manifest['runs'].items()):return
        time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--targets',required=True)
    args=parser.parse_args();watch(args.root,args.targets)
