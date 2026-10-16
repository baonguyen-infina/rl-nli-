"""Read-only comparison of existing dev predictions; never opens test artifacts."""
import collections,hashlib,json,math,os,pathlib
run=os.environ['DIAGNOSTIC_RUN_ID']
assert run in ('N20R20','N21R20')
out=pathlib.Path('/content/rl-nli/results')/(run+'_Qwen3.5-2B_s0')
paths=[out/'initial_eval/predictions_dev.jsonl',out/'ckpt_eval/checkpoint-100/predictions_dev.jsonl']
rows=[{x['id']:x for x in (json.loads(line) for line in p.read_text().splitlines() if line.strip())} for p in paths]
a,b=rows
assert len(a)==len(b)==1000 and a.keys()==b.keys()
counts=collections.Counter();classes=collections.defaultdict(collections.Counter)
for key,x in a.items():
 y=b[key];assert x['gold']==y['gold']
 old=x['pred']==x['gold'];new=y['pred']==y['gold']
 name='both_correct' if old and new else 'regression' if old else 'repair' if new else 'both_wrong'
 counts[name]+=1;classes[x['gold']][name]+=1
 counts['old_to_new/'+str(x['pred'])+'/'+str(y['pred'])]+=1
n=counts['repair']+counts['regression'];k=min(counts['repair'],counts['regression'])
p=min(1.,2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1.
metrics=[json.loads((p.parent/'metrics_dev.json').read_text()) for p in paths]
report={'run':run,'split':'dev','internal_only':True,'test_accessed':False,'n':1000,
 'paired_counts':dict(counts),'by_gold':{k:dict(v) for k,v in classes.items()},
 'mcnemar_exact_p':p,'baseline':{k:metrics[0][k] for k in ('accuracy','macro_f1','parse_rate')},
 'checkpoint100':{k:metrics[1][k] for k in ('accuracy','macro_f1','parse_rate')},
 'source_sha256':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
report['checkpoint']=100
p=out/'internal_diagnostics/checkpoint100_vs_warm.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False),flush=True)
