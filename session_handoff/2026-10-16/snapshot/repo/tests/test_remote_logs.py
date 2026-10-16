"""Full remote logs must survive VM release without including credentials or weights."""
import hashlib,json,tarfile
from pathlib import Path
import pytest


def helper():
    try:
        from scripts.remote_logs import collect_logs,pack_logs
    except ModuleNotFoundError:
        pytest.fail('Full remote log archive policy has not been implemented')
    return collect_logs,pack_logs


def fixture(tmp_path):
    run='N10R13';out=tmp_path/'rl-nli/results/N10R13_Qwen3.5-2B_s0';out.mkdir(parents=True)
    (out/'train_log.csv').write_text('step,loss\n20,0.1\n')
    (out/'rollouts_train.jsonl').write_text('{"reason":"evidence"}\n')
    (out/'config.yaml').write_text('seed: 0\n')
    (tmp_path/'N10R13.log').write_text('train stdout\neval stdout\n')
    (tmp_path/'r13_attempt2_dependencies.log').write_text('setup stdout\n')
    (tmp_path/'native_r13_primary_pipeline.log').write_text('pipeline stdout\n')
    return run,out


def test_full_log_archive_includes_stdout_and_unfiltered_rollout_evidence(tmp_path):
    collect,_=helper();run,out=fixture(tmp_path)
    files=collect(tmp_path,[run],'primary')
    assert files['runtime/N10R13.log']==tmp_path/'N10R13.log'
    assert files['runtime/r13_attempt2_dependencies.log']==tmp_path/'r13_attempt2_dependencies.log'
    assert files['results/N10R13_Qwen3.5-2B_s0/rollouts_train.jsonl']==out/'rollouts_train.jsonl'


def test_log_collection_excludes_weights_secrets_and_symlinks(tmp_path):
    collect,_=helper();run,out=fixture(tmp_path)
    (out/'adapter_model.safetensors').write_bytes(b'weights')
    (out/'optimizer.pt').write_bytes(b'optimizer')
    (out/'targets.json').write_text('{"secret":"upload capability"}')
    (tmp_path/'native_restore_manifest.json').write_text('{"secret":"oauth"}')
    (out/'secret_link.log').symlink_to(tmp_path/'native_restore_manifest.json')
    files=collect(tmp_path,[run],'primary')
    assert 'results/N10R13_Qwen3.5-2B_s0/adapter_model.safetensors' not in files
    assert 'results/N10R13_Qwen3.5-2B_s0/optimizer.pt' not in files
    assert 'results/N10R13_Qwen3.5-2B_s0/targets.json' not in files
    assert 'results/N10R13_Qwen3.5-2B_s0/secret_link.log' not in files
    assert tmp_path/'native_restore_manifest.json' not in files.values()


def test_log_bundle_records_each_payload_size_and_hash_independently(tmp_path):
    _,pack=helper();run,out=fixture(tmp_path)
    archive=tmp_path/'bundle.tar.gz'
    pack(tmp_path,[run],'primary',archive,event='progress-20')
    with tarfile.open(archive) as t:
        manifest=json.load(t.extractfile('logs_manifest.json'))
        assert manifest['event']=='progress-20' and manifest['scope']=='logs only; no weights'
        expected=b'train stdout\neval stdout\n'
        assert t.extractfile('runtime/N10R13.log').read()==expected
        assert manifest['files']['runtime/N10R13.log']=={'bytes':len(expected),'sha256':hashlib.sha256(expected).hexdigest()}
        assert t.extractfile('results/N10R13_Qwen3.5-2B_s0/rollouts_train.jsonl').read()==b'{"reason":"evidence"}\n'


def test_retry_reuses_original_bytes_even_if_live_logs_grew(tmp_path):
    _,pack=helper();run,out=fixture(tmp_path)
    archive=tmp_path/'bundle.tar.gz';pack(tmp_path,[run],'primary',archive,event='progress-20')
    original=archive.read_bytes();(tmp_path/'N10R13.log').write_text('later stdout\n')
    pack(tmp_path,[run],'primary',archive,event='progress-20')
    assert archive.read_bytes()==original


def test_checkpoint_training_state_is_logged_without_copying_tokenizer_assets(tmp_path):
    collect,_=helper();run,out=fixture(tmp_path)
    checkpoint=out/'grpo_ckpts/checkpoint-100';checkpoint.mkdir(parents=True)
    (checkpoint/'trainer_state.json').write_text('{"global_step":100}')
    (checkpoint/'tokenizer.json').write_text('{"large_vocabulary":"not a log"}')
    files=collect(tmp_path,[run],'primary')
    assert 'results/N10R13_Qwen3.5-2B_s0/grpo_ckpts/checkpoint-100/trainer_state.json' in files
    assert 'results/N10R13_Qwen3.5-2B_s0/grpo_ckpts/checkpoint-100/tokenizer.json' not in files


def test_readout_probe_setup_tests_and_cache_logs_are_preserved(tmp_path):
    collect,_=helper();run,out=fixture(tmp_path)
    for name in ['r14_probe_dependencies.log','r14_probe_red.log','r14_probe_green.log',
                 'native_r14_probe_pipeline.log','full_logs_probe_watcher.log']:
        (tmp_path/name).write_text('complete diagnostic execution evidence\n')
        assert 'runtime/'+name in collect(tmp_path,[run],'probe')


@pytest.mark.parametrize('run,lane,prefix',[
    ('N13R16','r16traceswap','r16'),
    ('N14R17','r17tracesaliency','r17'),
])
def test_trace_followups_archive_setup_regression_and_pipeline_stdout(tmp_path,run,lane,prefix):
    collect,pack=helper()
    names=[prefix+'_run_full_tests.log',prefix+'_run_configs.log',
           prefix+'_run_dependencies.log','native_'+prefix+'_pipeline.log']
    for name in names:
        (tmp_path/name).write_text('remote evidence: '+name+'\n')
    files=collect(tmp_path,[run],lane)
    assert all('runtime/'+name in files for name in names)
    archive=pack(tmp_path,[run],lane,tmp_path/'logs.tar.gz',event='terminal')
    with tarfile.open(archive) as saved:
        for name in names:
            assert saved.extractfile('runtime/'+name).read()==('remote evidence: '+name+'\n').encode()
