"""Collect completed runs from frozen configs and recorded metrics; never infer missing results."""
import argparse
import csv
import json
import re
from pathlib import Path
import yaml
from common.io import RESULTS, ARTIFACTS, read_json, read_jsonl


def collect_runs(root=RESULTS, split='dev'):
    rows=[]
    for out in sorted(Path(root).iterdir()):
        cost_name='cost_test.json' if split=='test' else 'cost.json'
        if not out.is_dir() or not all((out/p).exists() for p in ['config.yaml',cost_name,f'metrics_{split}.json']):
            continue
        if split=='test':
            receipt_path=out/'.test_claim/receipt.json'
            if not receipt_path.exists():continue
            receipt=read_json(receipt_path)
            if receipt.get('status')!='completed':continue
            from eval.final_protocol import _verified_outputs
            n,artifacts=_verified_outputs(out)
            if receipt.get('n')!=n or receipt.get('artifacts')!=artifacts:
                raise ValueError(f'Changed held-out artifacts: {out}')
        cost=read_json(out/cost_name)
        if cost.get('status')!='complete':continue
        cfg=yaml.safe_load((out/'config.yaml').read_text())
        match=re.search(r'_s(\d+)$',out.name)
        if not match:continue
        m=read_json(out/f'metrics_{split}.json')
        row={'run_id':cfg['run_id'],'model':cfg['model']['name'],'seed':int(match[1]),
             'recipe':cfg['recipe'],'format':cfg['format'],'split':split,
             **{k:m[k] for k in ['n','accuracy','macro_f1','parse_rate']},
             'wall_seconds':cost.get('wall_seconds'),'rule_consistency':None,'structure_rate':None,
             'source':str(out)}
        protocol=cfg.get('validation_protocol')
        engine=cfg.get('validation_eval' if protocol=='fixed_single_sequence' else 'eval',{}).get('engine_options',{})
        initial=out/'initial_eval'/f'metrics_{split}.json'
        selection=read_json(out/'ckpt_scores.json') if (out/'ckpt_scores.json').exists() else {}
        row.update(init_from=cfg.get('init_from'),native_mode=cfg.get('native_mode'),
                   scoring_protocol=cfg.get('scoring_protocol'),validation_protocol=protocol,
                   eval_engine_options=json.dumps(engine,sort_keys=True),
                   initial_macro_f1=read_json(initial).get('macro_f1') if initial.exists() else None,
                   selected_checkpoint=selection.get('best'),selection_split=selection.get('selection_split'),
                   selection_metric=selection.get('selection_metric'))
        predictions=out/f'predictions_{split}.jsonl'
        if cfg['format']=='align' and predictions.exists():
            records=read_jsonl(predictions)
            if len(records)!=m['n']:raise ValueError(f'Prediction count mismatch: {out}')
            row['rule_consistency']=sum(r.get('rule_label') is not None and r['rule_label']==r.get('pred') for r in records)/len(records)
            row['structure_rate']=sum(r.get('grounded') is True and (r.get('coverage') or 0)>=.5 and r.get('rule_label') is not None and r['rule_label']==r.get('pred') for r in records)/len(records)
        rows.append(row)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=RESULTS)
    ap.add_argument('--runs',nargs='+')
    ap.add_argument('--split',choices=['dev','test'],default='dev')
    ap.add_argument('--out',type=Path,default=ARTIFACTS/'analysis'/'runs.csv');a=ap.parse_args()
    rows=collect_runs(a.results,a.split)
    if a.runs:rows=[row for row in rows if row['run_id'] in a.runs]
    if not rows:raise SystemExit('No completed runs with metrics')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'{len(rows)} runs -> {a.out}')

if __name__=='__main__':main()
