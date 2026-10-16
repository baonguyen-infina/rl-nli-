"""Fail-closed, hash-locked held-out evaluation and once-only result receipts."""
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import yaml
from common.io import ROOT


def _digest(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('Malformed SHA256')
    return value


def sha256_path(path):
    path=Path(path)
    if path.is_symlink() or not path.exists():raise ValueError('Missing or symlinked artifact')
    def file_hash(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    if path.is_file():return file_hash(path)
    if not path.is_dir():raise ValueError('Unsupported artifact type')
    files=sorted(path.rglob('*'))
    if any(p.is_symlink() for p in files):raise ValueError('Symlink inside adapter tree')
    files=[p for p in files if p.is_file()]
    if not files:raise ValueError('Empty adapter tree')
    h=hashlib.sha256()
    for p in files:
        h.update(p.relative_to(path).as_posix().encode()+b'\0'+file_hash(p).encode()+b'\n')
    return h.hexdigest()


def _artifact(spec,root):
    if not isinstance(spec,dict):raise ValueError('Artifact specification required')
    raw=spec.get('path')
    if not isinstance(raw,str) or '\\' in raw:raise ValueError('Invalid artifact path')
    rel=PurePosixPath(raw)
    if rel.is_absolute() or '..' in rel.parts or len(rel.parts)<3 or rel.parts[0]!='results':
        raise ValueError('Artifact must stay inside results')
    results=root/'results';path=results
    for part in rel.parts[1:]:
        path=path/part
        if path.is_symlink():raise ValueError('Symlinked artifact path')
    if not path.resolve().is_relative_to(results.resolve()):raise ValueError('Escaped results')
    if sha256_path(path)!=_digest(spec.get('sha256')):raise ValueError('Artifact SHA256 changed')
    return path


def validate_protocol(manifest,*,root=ROOT):
    root=Path(root)
    if not isinstance(manifest,dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version']!=1:
        raise ValueError('Unsupported protocol schema')
    if manifest.get('locked') is not True or manifest.get('selection_split')!='dev' or manifest.get('report_split')!='test':
        raise ValueError('Protocol must be locked after dev selection for test reporting')
    rows=manifest.get('runs')
    if not isinstance(rows,list) or not rows:raise ValueError('Nonempty run selection required')
    selected=[];seen=set()
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Run entry required')
        rid=row.get('run_id');seed=row.get('seed');name=row.get('model')
        if not isinstance(rid,str) or not re.fullmatch('[A-Za-z0-9]+',rid):raise ValueError('Invalid run ID')
        if type(seed) is not int or seed<0 or not isinstance(name,str) or not name:raise ValueError('Invalid seed/model')
        if (rid,seed) in seen:raise ValueError('Duplicate selected run/seed')
        seen.add((rid,seed))
        folder=root/'results'/f'{rid}_{name.split("/")[-1]}_s{seed}'
        source=_artifact(row.get('source_config'),root)
        if source.parent!=folder:raise ValueError('Source config outside named run')
        config=yaml.safe_load(source.read_text())
        if not isinstance(config,dict) or config.get('run_id')!=rid or config.get('model',{}).get('name')!=name:
            raise ValueError('Source run identity mismatch')
        if seed not in config.get('seeds',[config.get('seed',seed)]):raise ValueError('Source training seed mismatch')
        adapter=_artifact(row.get('adapter'),root)
        adapter_run=adapter.relative_to(root/'results').parts[0]
        if not adapter_run.endswith(f'_s{seed}'):raise ValueError('Adapter training seed mismatch')
        effective=source
        if 'evaluation_config' in row:
            effective=_artifact(row['evaluation_config'],root)
            if effective.parent!=folder:raise ValueError('Evaluation config outside named run')
        evaluation=yaml.safe_load(effective.read_text())
        if not isinstance(evaluation,dict) or evaluation.get('run_id')!=rid or evaluation.get('model',{}).get('name')!=name:
            raise ValueError('Evaluation run identity mismatch')
        if any(evaluation.get(k)!=row.get(k) for k in ('format','recipe')):
            raise ValueError('Evaluation format/recipe mismatch')
        if row.get('format') not in ('label','cot','align'):raise ValueError('Invalid evaluation format')
        selected.append({**row,'source_config_path':source,'evaluation_config_path':effective,
                         'adapter_path':adapter,'source_config_data':config,'evaluation_config_data':evaluation})
    return selected


def _read_receipt(output_dir,manifest_sha256):
    try:receipt=json.loads((output_dir/'.test_claim/receipt.json').read_text())
    except (OSError,ValueError) as exc:raise ValueError('Pending or malformed test claim') from exc
    if receipt.get('manifest_sha256')!=manifest_sha256:raise ValueError('Different locked protocol')
    return receipt


def _atomic_receipt(output_dir,receipt):
    path=output_dir/'.test_claim/receipt.json'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(receipt,indent=2));temporary.replace(path)


def _verified_outputs(output_dir):
    names=['predictions_test.jsonl','metrics_test.json']
    if (output_dir/'cost_test.json').exists():names.append('cost_test.json')
    try:
        rows=[json.loads(line) for line in (output_dir/names[0]).read_text().splitlines() if line.strip()]
        metric=json.loads((output_dir/names[1]).read_text())
    except (OSError,ValueError) as exc:raise ValueError('Missing or malformed test outputs') from exc
    if not rows or type(metric.get('n')) is not int or metric['n']!=len(rows):raise ValueError('Test prediction count mismatch')
    ids=[r.get('id') for r in rows]
    if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):raise ValueError('Invalid or duplicate test prediction IDs')
    return len(rows),{name:sha256_path(output_dir/name) for name in names}


def claim_test_run(output_dir,manifest_sha256):
    _digest(manifest_sha256);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    claim=out/'.test_claim'
    if claim.exists():
        receipt=_read_receipt(out,manifest_sha256)
        if receipt.get('status')!='completed':raise ValueError('Test evaluation already in progress')
        n,artifacts=_verified_outputs(out)
        if receipt.get('n')!=n or receipt.get('artifacts')!=artifacts:raise ValueError('Completed test artifacts changed')
        return {'reused':True,'receipt':receipt}
    if any((out/name).exists() for name in ('predictions_test.jsonl','metrics_test.json','cost_test.json')):
        raise ValueError('Unattributed existing test results')
    claim.mkdir()
    receipt={'schema_version':1,'manifest_sha256':manifest_sha256,'status':'in_progress'}
    _atomic_receipt(out,receipt)
    return {'reused':False,'receipt':receipt}


def complete_test_run(output_dir,manifest_sha256):
    _digest(manifest_sha256);out=Path(output_dir)
    receipt=_read_receipt(out,manifest_sha256)
    if receipt.get('status')!='in_progress':raise ValueError('Test claim is not in progress')
    n,artifacts=_verified_outputs(out)
    receipt.update(status='completed',n=n,artifacts=artifacts)
    _atomic_receipt(out,receipt)
    return receipt
