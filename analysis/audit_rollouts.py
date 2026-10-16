"""Reproducible training-rollout audit, distinct from held-out evaluation."""
import argparse
import random
from collections import Counter
from pathlib import Path
from common.io import read_jsonl,write_jsonl,write_json
from formats.parser import parse


def select_rows(records,min_step=150,n=50,seed=42):
    latest={r['id']:r for r in records if r['step']>=min_step}
    if len(latest)<n:raise ValueError(f'Only {len(latest)} unique late prompts; require {n}')
    return random.Random(seed).sample(list(latest.values()),n)


def main():
    from data.prepare import get_rows
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True,type=Path)
    ap.add_argument('--min-step',default=150,type=int);a=ap.parse_args()
    rows=select_rows(read_jsonl(a.run_dir/'rollouts_train.jsonl'),a.min_step)
    sources={r['id']:r for r in get_rows('train')}
    for r in rows:
        r.update(premise=sources[r['id']]['premise'],hypothesis=sources[r['id']]['hypothesis'])
    counts=Counter(line.rel for r in rows for line in (parse(r['raw_output'],'align',True).align_lines or []))
    summary={'n':len(rows),'selection':f'latest completion per unique prompt, step >= {a.min_step}; uniform seed 42',
             'gold_counts':dict(Counter(r['gold'] for r in rows)),'status_counts':dict(counts)}
    for name,fn in {
        'accuracy':lambda r:r['pred']==r['gold'],
        'guess_accuracy':lambda r:r['guess']==r['gold'],
        'parse_rate':lambda r:r['parse_ok'],
        'grounded_rate':lambda r:r['grounded'] is True,
        'rule_consistency':lambda r:r['rule_label'] is not None and r['rule_label']==r['pred'],
        'coverage_pass_rate':lambda r:(r['coverage'] or 0)>=.5,
    }.items():summary[name]=sum(fn(r) for r in rows)/len(rows)
    out=a.run_dir/'audits';write_jsonl(out/'pilot50.jsonl',rows);write_json(out/'pilot50_summary.json',summary)
    print(summary)

if __name__=='__main__':main()
