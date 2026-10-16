"""Execute one previously claimed held-out evaluation from a hash-locked manifest."""
import argparse
import json
import os
from pathlib import Path
import re
from common.io import ROOT,read_json,read_jsonl,write_json,write_jsonl
from eval.final_protocol import validate_protocol,sha256_path


def validate_dataset_spec(spec):
    if not isinstance(spec,dict) or spec.get('repo')!='uitnlp/ViANLI' or spec.get('file')!='vianli_test.jsonl':
        raise ValueError('Held-out protocol requires the official ViANLI test file')
    if not isinstance(spec.get('revision'),str) or not re.fullmatch('[0-9a-f]{40}',spec['revision']):
        raise ValueError('Pinned dataset revision required')
    if type(spec.get('expected_n')) is not int or spec['expected_n']!=1000:
        raise ValueError('Official test count must be1000')
    return spec


def load_locked_test_rows(spec):
    from data.prepare import _guard,prepare_rows,check_counts
    from huggingface_hub import hf_hub_download
    _guard('test',True)
    spec=validate_dataset_spec(spec)
    path=hf_hub_download(repo_id=spec['repo'],filename=spec['file'],revision=spec['revision'],repo_type='dataset')
    rows=prepare_rows(read_jsonl(path))
    check_counts({'test':len(rows)})
    if len({row['id'] for row in rows})!=len(rows):raise ValueError('Duplicate held-out IDs')
    return rows,{'repo':spec['repo'],'revision':spec['revision'],'file':spec['file'],
                 'sha256':sha256_path(Path(path).resolve()),'n':len(rows)}


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--protocol',required=True)
    args=ap.parse_args(argv);path=Path(args.protocol)
    manifest=read_json(path);entries=validate_protocol(manifest,root=ROOT)
    selected=[e for e in entries if e['run_id']==args.run and e['seed']==args.seed]
    if len(selected)!=1:raise ValueError('Run/seed is not registered in the locked protocol')
    entry=selected[0];out=entry['source_config_path'].parent
    try:receipt=read_json(out/'.test_claim/receipt.json')
    except (OSError,ValueError) as exc:raise ValueError('A parent test claim is required') from exc
    if receipt.get('status')!='in_progress' or receipt.get('manifest_sha256')!=sha256_path(path):
        raise ValueError('Invalid or completed test claim')
    spec=validate_dataset_spec(manifest.get('dataset'))
    with (out/'.test_claim/execution_started').open('x') as marker:marker.write(sha256_path(path))
    previous=os.environ.get('G4_UNLOCK')
    try:
        os.environ['G4_UNLOCK']='1';rows,data_receipt=load_locked_test_rows(spec)
    finally:
        if previous is None:os.environ.pop('G4_UNLOCK',None)
        else:os.environ['G4_UNLOCK']=previous
    from formats.prompts import build_messages
    from eval.predict import generate,assemble_records
    from eval.metrics import compute_metrics
    cfg=entry['evaluation_config_data'];fmt=cfg['format'];guess=cfg.get('guess',False);ev=cfg['eval']
    messages=[build_messages(r['premise'],r['hypothesis'],fmt,guess,cfg.get('schema_examples','single')) for r in rows]
    options=cfg['validation_eval']['engine_options'] if cfg.get('validation_protocol')=='fixed_single_sequence' else ev.get('engine_options')
    outputs=generate(cfg['model']['name'],messages,ev['max_new_tokens'][fmt],entry['adapter_path'],
        ev['gpu_memory_utilization'],ev['max_model_len'],
        reasoning_budget=cfg['native_pilot']['reasoning_budget'] if cfg.get('bounded_reasoning') else None,
        engine_options=options,backend=ev.get('backend','vllm'),model_revision=ev.get('model_revision'))
    if len(outputs)!=len(rows):raise ValueError('Incomplete test generation')
    records=assemble_records(rows,outputs,fmt,guess);metric=compute_metrics(records)
    write_jsonl(out/'predictions_test.jsonl',records);write_json(out/'metrics_test.json',metric)
    write_json(out/'dataset_test_receipt.json',data_receipt)
    write_json(out/'eval_runtime_test.json',{'backend':ev.get('backend','vllm'),'engine_options':options or {},
        'temperature':0.,'split':'test','format':fmt,'seed':args.seed,'n':len(rows),
        'model_revision':ev.get('model_revision'),'manifest_sha256':sha256_path(path)})
    print({k:metric[k] for k in ('n','macro_f1','accuracy','parse_rate')})


if __name__=='__main__':main()
