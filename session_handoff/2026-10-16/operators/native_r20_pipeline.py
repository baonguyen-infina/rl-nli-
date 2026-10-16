"""Run a registered 400-step teacher-free RL candidate on an existing G4."""
import hashlib, json, os, pathlib, subprocess, sys, time, traceback

root = pathlib.Path('/content/rl-nli')
os.chdir(root)
sys.path.insert(0, str(root))
from common.io import load_config, read_json, write_json

run = os.environ['NATIVE_RUN_ID']
status = pathlib.Path('/content') / (run + '_pipeline_status.json')
out = root / 'results' / (run + '_Qwen3.5-2B_s0')
source_names = [
    'common/io.py', 'common/labels.py', 'configs/base.yaml', 'configs/matrix.py',
    f'configs/runs/{run}.yaml', 'data/prepare.py', 'eval/final_protocol.py',
    'eval/final_predict.py', 'eval/metrics.py', 'eval/records.py', 'eval/predict.py',
    'eval/hf_generate.py', 'formats/parser.py', 'formats/prompts.py', 'scripts/run.py',
    'scripts/train_worker.py', 'train/common.py', 'train/audit.py',
    'train/native_grpo.py', 'train/native_reward.py', 'train/native_training.py',
    'train/native_groups.py', 'train/native_allocation.py', 'train/native_rollout.py',
    'train/native_trace_swap.py', 'train/native_saliency.py', 'train/label_marginal.py',
]
sources = {name: {'sha256': hashlib.sha256((root/name).read_bytes()).hexdigest()}
           for name in source_names}

def report(phase, **extra):
    row = {'run': run, 'phase': phase, 'observed_at': time.time(), **extra}
    write_json(status, row)
    print(json.dumps(row), flush=True)

def check_sources():
    for name, item in sources.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == item['sha256'], name

try:
    cfg = load_config(run)
    assert cfg['recipe'] == 'native' and cfg['model']['name'] == 'Qwen/Qwen3.5-2B'
    assert cfg['init_from'] == 'D3R9' and cfg['eval']['split'] == 'dev'
    assert cfg['native_pilot']['max_steps'] == 400
    assert cfg['native_mode'] in ('trace_saliency', 'marginal_f1')
    assert cfg['eval']['model_revision'] == '15852e8c16360a2fea060d615a32b45270f8a8fc'
    assert read_json('/content/'+run+'_setup_ready.json')['passed']
    assert not out.exists(), f'Refusing to overwrite {out}'
    free = __import__('shutil').disk_usage('/content').free
    assert free > 20*1024**3, f'Insufficient remote disk space: {free}'
    env = {**os.environ, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'PYTHONUNBUFFERED':'1'}
    log = pathlib.Path('/content')/(run+'.log')
    with log.open('w') as stream:
        child = subprocess.Popen([sys.executable,'-m','scripts.run','--run',run,'--seed','0','--phase','all'],
                                cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        deadline = time.time()+900
        while child.poll() is None and not (out/'config.yaml').exists():
            if time.time()>deadline: raise TimeoutError('Run config was not written')
            time.sleep(2)
        if out.exists():
            write_json(out/'source_snapshot.json', sources)
            write_json(out/'execution_provenance.json', {
                'method': cfg['native_mode'], 'seed':0, 'teacher':False,
                'test_split_accessed':False, 'source_hashes':{k:v['sha256'] for k,v in sources.items()},
                'smoke_run': 'G0'+run+'P', 'smoke_gate_passed':True,
                'paper_metrics_split':'test only; dev stays internal'})
        while child.poll() is None:
            check_sources()
            rows=[]
            path=out/'train_log.csv'
            if path.exists():
                import csv
                try:
                    with path.open() as f: rows=list(csv.DictReader(f))
                except (OSError, csv.Error): pass
            last=next((r.get('step') for r in reversed(rows) if r.get('step')),None)
            report('training_or_internal_selection', step=last)
            time.sleep(60)
    check_sources()
    if child.returncode: raise RuntimeError(f'training/eval exited {child.returncode}')
    assert read_json(out/'cost.json')['status']=='complete'
    selection=read_json(out/'ckpt_scores.json')
    assert selection['selection_split']=='dev' and len(selection['scores'])==4
    report('training_complete', checkpoint=selection['best'], test_accessed=False)
    write_json('/content/'+run+'_pipeline_finished.json', {
        'run':run,'complete':True,'training_returncode':0,'test_accessed':False,'finished_at':time.time()})
    report('complete', checkpoint=selection['best'], test_accessed=False)
except Exception as exc:
    with open('/content/'+run+'_failure.log','a') as f: traceback.print_exc(file=f)
    report('failed', error_type=type(exc).__name__, test_accessed=False)
    raise
