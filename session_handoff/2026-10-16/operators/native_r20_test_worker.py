"""Wait for dev selection, lock one checkpoint, and execute its sole test pass."""
import datetime, hashlib, json, os, pathlib, subprocess, sys, time, traceback
root=pathlib.Path('/content/rl-nli');os.chdir(root);sys.path.insert(0,str(root))
from common.io import read_json,read_jsonl,write_json
from eval.final_protocol import sha256_path,validate_protocol
from eval.metrics import compute_metrics
from scripts.run import select_checkpoint

run=os.environ['NATIVE_RUN_ID']
out=root/'results'/(run+'_Qwen3.5-2B_s0')
registration=pathlib.Path('/content')/(run+'_test_registration.json')
status=pathlib.Path('/content')/(run+'_test_status.json')
def report(stage,**extra):
    val={'run':run,'stage':stage,'observed_at':time.time(),**extra}
    write_json(status,val);print(json.dumps(val),flush=True)
try:
    reg=read_json(registration)
    assert reg['run']==run and reg['seed']==0 and reg['candidate_fixed_before_test']
    report('registered_waiting_for_training',test_accessed=False)
    deadline=time.time()+8*3600
    while True:
        cost=read_json(out/'cost.json') if (out/'cost.json').exists() else {}
        if cost.get('status')=='failed': raise RuntimeError('Training failed; test remains unopened')
        if cost.get('status')=='complete': break
        if time.time()>deadline: raise TimeoutError('Training timed out; test remains unopened')
        time.sleep(30)
    for name,expected in reg['source_hashes'].items():
        assert sha256_path(root/name)==expected, 'Frozen source changed: '+name
    cfg=json.loads(json.dumps(__import__('yaml').safe_load((out/'config.yaml').read_text())))
    assert cfg['run_id']==run and cfg['format']=='label' and cfg['recipe']=='native'
    assert cfg['eval']['model_revision']==reg['model_revision']
    selection=read_json(out/'ckpt_scores.json')
    assert selection['selection_split']=='dev' and set(selection['scores'])=={
        'checkpoint-100','checkpoint-200','checkpoint-300','checkpoint-400'}
    checkpoints=[out/'grpo_ckpts'/k for k in selection['scores']]
    assert select_checkpoint(checkpoints,selection['scores']).name==selection['best']
    assert read_json(out/'metrics_dev.json')['n']==1000
    assert not (out/'.test_claim').exists() and not (out/'predictions_test.jsonl').exists()
    def artifact(path):return {'path':str(path.relative_to(root)),'sha256':sha256_path(path)}
    manifest={'schema_version':1,'locked':True,'selection_split':'dev','report_split':'test',
        'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'dataset':reg['dataset'],'model_revision':reg['model_revision'],'training_seeds':[0],
        'primary_method':run,'exploratory_followup_after_prior_test_cohort':True,
        'rules':'Select by full dev only, then report this registered test result regardless of outcome; no test-based retuning.',
        'runs':[{'run_id':run,'seed':0,'model':cfg['model']['name'],'recipe':cfg['recipe'],'format':cfg['format'],
            'source_config':artifact(out/'config.yaml'),'adapter':artifact(out/'best_adapter'),
            'checkpoint':selection['best'],'checkpoint_selection':'full_dev_greedy_macro_f1',
            'inference_view':'single_sequence_greedy_label'}]}
    validate_protocol(manifest,root=root)
    protocol=out/'locked_test_protocol.json';write_json(protocol,manifest)
    write_json(out/'test_registration.json',reg);digest=sha256_path(protocol)
    report('testing',test_accessed=True,protocol_sha256=digest,checkpoint=selection['best'])
    log=pathlib.Path('/content')/(run+'_test.log')
    with log.open('w') as stream:
        p=subprocess.run([sys.executable,'-m','scripts.run','--run',run,'--seed','0','--phase','test','--protocol',str(protocol)],
            cwd=root,env={**os.environ,'HF_HUB_OFFLINE':'0','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1'},
            stdout=stream,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError('Once-only test command failed; preserve claim for audit')
    receipt=read_json(out/'.test_claim/receipt.json')
    assert receipt['status']=='completed' and receipt['n']==1000 and receipt['manifest_sha256']==digest
    rows=read_jsonl(out/'predictions_test.jsonl');metrics=read_json(out/'metrics_test.json')
    assert len(rows)==len({r['id'] for r in rows})==1000
    recomputed=compute_metrics(rows)
    for key in ('macro_f1','accuracy','parse_rate'): assert abs(metrics[key]-recomputed[key])<1e-12
    assert read_json(out/'cost_test.json')['status']=='complete'
    summary={'run':run,'split':'test','seed':0,'n':1000,
        **{k:metrics[k] for k in ('macro_f1','accuracy','parse_rate')},'target_macro_f1':.58,
        'target_met':metrics['macro_f1']>=.58,'protocol_sha256':digest,
        'predictions_sha256':sha256_path(out/'predictions_test.jsonl'),
        'paper_eligible':True,'scope':'single-seed exploratory follow-up; no A/A* acceptance claim'}
    write_json(out/'test_verification.json',summary)
    report('test_verified_backup_pending',test_accessed=True,summary=summary)
except Exception as exc:
    with open('/content/'+run+'_test_failure.log','a') as stream:traceback.print_exc(file=stream)
    report('test_failed' if (out/'.test_claim').exists() else 'training_failed',
           test_accessed=bool((out/'.test_claim/execution_started').exists()),error_type=type(exc).__name__)
    raise
