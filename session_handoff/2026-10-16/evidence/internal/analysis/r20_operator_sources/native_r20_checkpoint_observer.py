"""Evaluate checkpoint100 on dev with the frozen run config; reuse at final selection."""
import hashlib,json,os,pathlib,subprocess,sys,time,traceback
run=os.environ['NATIVE_RUN_ID'];assert run in ('N20R20','N21R20')
root=pathlib.Path('/content/rl-nli');os.chdir(root);sys.path.insert(0,str(root))
from common.io import read_json,read_jsonl,write_json
from eval.metrics import compute_metrics
from scripts.remote_artifacts import event_ready
from scripts.run import _predict_cli
out=root/'results'/(run+'_Qwen3.5-2B_s0');status=pathlib.Path('/content')/(run+'_checkpoint100_status.json')
def report(stage,**extra):write_json(status,dict(run=run,stage=stage,observed_at=time.time(),test_accessed=False,**extra))
try:
 deadline=time.time()+6*3600
 while not event_ready(out,'checkpoint-adapter-100'):
  cost=read_json(out/'cost.json') if (out/'cost.json').exists() else {}
  if cost.get('status') in ('failed','aborted'):raise RuntimeError('Training is terminal')
  if time.time()>deadline:raise TimeoutError('Checkpoint100 not ready')
  report('waiting_checkpoint100');time.sleep(30)
 registration=read_json('/content/'+run+'_test_registration.json')
 def verify():
  for name,expected in registration['source_hashes'].items():
   assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
 verify()
 import yaml
 cfg=yaml.safe_load((out/'config.yaml').read_text())
 assert cfg['eval']['split']=='dev' and cfg['format']=='label' and cfg['eval']['backend']=='transformers'
 ev=out/'ckpt_eval/checkpoint-100';ev.mkdir(parents=True,exist_ok=True)
 if not (ev/'metrics_dev.json').exists():
  free=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  assert int(free.stdout.strip())>20000,'Insufficient memory for concurrent dev readout'
  (ev/'.early_evaluation_lock').mkdir()
  report('evaluating_checkpoint100',concurrent_training=True)
  _predict_cli(run,0,'dev',ev,adapter=out/'grpo_ckpts/checkpoint-100',config_path=out/'config.yaml')
 verify()
 rows=read_jsonl(ev/'predictions_dev.jsonl');metrics=read_json(ev/'metrics_dev.json');actual=compute_metrics(rows)
 assert len(rows)==len({row['id'] for row in rows})==1000
 assert all(abs(metrics[k]-actual[k])<1e-12 for k in ('macro_f1','accuracy','parse_rate'))
 report('checkpoint100_dev_verified',n=1000,internal_only=True,primary_selection_unchanged=True,
        metrics_path=str(ev/'metrics_dev.json'),predictions_sha256=hashlib.sha256((ev/'predictions_dev.jsonl').read_bytes()).hexdigest())
except Exception as exc:
 with open('/content/'+run+'_checkpoint100_failure.log','a') as f:traceback.print_exc(file=f)
 report('observer_failed',error_type=type(exc).__name__)
 raise
