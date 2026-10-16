"""CPU-only audit of already recorded training advantages; never read benchmark test data."""
import argparse,collections,csv,hashlib,json,math,os,pathlib,statistics,time

from types import SimpleNamespace
run=os.environ['NATIVE_RUN_ID'];assert run in ('N20R20','N15R18')
options=SimpleNamespace(run=run,steps=20)
root=pathlib.Path('/content/rl-nli')
out=root/'results'/(options.run+'_Qwen3.5-2B_s0')
frozen=root/'train/native_grpo.py'
before=hashlib.sha256(frozen.read_bytes()).hexdigest()
groups=collections.defaultdict(list)
invalid_lines=0
with (out/'rollouts_train.jsonl').open() as stream:
    for line in stream:
        try:record=json.loads(line)
        except json.JSONDecodeError:
            invalid_lines+=1
            continue
        if int(record['step'])>=options.steps:break
        if not record['slot_weight']:continue
        value=float(record['advantage'])
        assert math.isfinite(value)
        groups[(int(record['step']),record['id'])].append(
            (record.get('pred')==record['gold'],value))
assert groups and all(len(rows)==4 for rows in groups.values())
mixed={key:rows for key,rows in groups.items() if len({correct for correct,_ in rows})==2}
wrong=[adv for rows in mixed.values() for correct,adv in rows if not correct]
correct=[adv for rows in mixed.values() for is_correct,adv in rows if is_correct]
rows=list(csv.DictReader((out/'train_log.csv').open()))
traces={}
for name in ['native/trace_gap','native/trace_gap_std','native/rotated_logq','native/accuracy',
             'native/policy_allocation_bonus','native/replay_active_score_max_difference']:
    values=[float(row[name]) for row in rows if row.get(name) and row.get('step')
            and float(row['step'])<=options.steps]
    if values:traces[name]={'n':len(values),'min':min(values),'mean':statistics.fmean(values),'max':max(values)}
result={'run':options.run,'observed_at':time.time(),'split':'train','paper_eligible':False,
        'test_accessed':False,'source_steps':[0,options.steps-1],
        'prompt_rollout_groups':len(groups),'mixed_correctness_groups':len(mixed),
        'wrong_samples_in_mixed_groups':len(wrong),'correct_samples_in_mixed_groups':len(correct),
        'wrong_positive_advantage':sum(value>1e-8 for value in wrong),
        'correct_negative_advantage':sum(value< -1e-8 for value in correct),
        'wrong_advantage_max':max(wrong,default=None),'correct_advantage_min':min(correct,default=None),
        'training_metric_summary':traces,'partial_lines_skipped':invalid_lines,
        'training_sources_changed':False,'native_grpo_sha256':before,
        'diagnostic_source_sha256':hashlib.sha256(pathlib.Path('/content/native_r20_training_credit_diagnostic.py').read_bytes()).hexdigest(),
        'limitation':'Stored scalar training advantages diagnose credit transport only, not semantic faithfulness or held-out performance.'}
assert hashlib.sha256(frozen.read_bytes()).hexdigest()==before
destination=out/'training_credit_diagnostic.json'
temporary=destination.with_suffix('.tmp')
temporary.write_text(json.dumps(result,indent=2)+'\n')
temporary.replace(destination)
print(json.dumps(result),flush=True)
