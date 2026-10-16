"""Stop only the owned N20 training worker after verified dev and adapter backup."""
import csv,hashlib,json,os,pathlib,signal,time
run=os.environ['NATIVE_RUN_ID'];assert run=='N20R20'
root=pathlib.Path('/content/rl-nli');out=root/'results'/(run+'_Qwen3.5-2B_s0')
read=lambda path:json.loads(path.read_text())
status=read(pathlib.Path('/content')/(run+'_checkpoint100_status.json'))
assert status['stage']=='checkpoint100_dev_verified' and status['n']==1000
assert not (out/'.test_claim').exists() and not (out/'predictions_test.jsonl').exists()
backup=read(out/'github_backup/checkpoint-100.receipt.json')
assert backup['repo_private'] and backup['release_draft'] and backup['github_digest']=='sha256:'+backup['sha256']
assert backup['model_bytes_written_locally']==0
registration=read(pathlib.Path('/content')/(run+'_test_registration.json'))
for name,expected in registration['source_hashes'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
pid=85155
args=pathlib.Path('/proc')/str(pid)/'cmdline'
parts=args.read_bytes().decode().strip('\0').split('\0')
assert '-m' in parts and parts[parts.index('-m')+1]=='scripts.train_worker'
assert parts[parts.index('--run')+1]==run and parts[parts.index('--seed')+1]=='0'
metrics=read(out/'ckpt_eval/checkpoint-100/metrics_dev.json')
rows=list(csv.DictReader((out/'train_log.csv').open()))
steps=[int(float(r['step'])) for r in rows if r.get('step')]
receipt={'run':run,'method_status':'early_stop_requested','observed_at':time.time(),'stop_signal':'SIGINT',
 'worker_pid':pid,'last_observed_step':max(steps),'original_budget_steps':400,'retained_checkpoint':100,
 'reason':'Exploratory dev-based compute allocation: no compelling gain over the faster N21 arm at checkpoint100; unrestricted credit diagnostic remains unresolved.',
 'stopping_rule_preregistered':False,'test_accessed':False,'paper_eligible':False,
 'checkpoint100_dev_metrics':{k:metrics[k] for k in ('n','accuracy','macro_f1','parse_rate')},
 'checkpoint100_archive_id':backup['id'],'checkpoint100_archive_sha256':backup['sha256'],
 'preserve_all_logs':True,'do_not_retry_as_training_failure':True}
p=out/'early_stop_receipt.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(p)
os.kill(pid,signal.SIGINT)
print(json.dumps({k:receipt[k] for k in ('run','method_status','last_observed_step','worker_pid','retained_checkpoint','test_accessed','checkpoint100_archive_id')}),flush=True)
