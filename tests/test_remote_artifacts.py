"""Remote autosave separates metadata, completed checkpoints and final adapters."""
from pathlib import Path
import json
import tarfile

from scripts.remote_artifacts import event_ready, snapshot_files, pack_snapshot


def fixture_run(tmp_path):
    root=tmp_path/'R_Qwen3.5-2B_s0';root.mkdir()
    (root/'config.yaml').write_text('run_id: R\n')
    (root/'train_log.csv').write_text('step,loss\n20,0.1\n')
    (root/'rollouts_train.jsonl').write_text('{}\n')
    for step in [100,200]:
        p=root/'grpo_ckpts'/f'checkpoint-{step}';p.mkdir(parents=True)
        (p/'adapter_model.safetensors').write_bytes(b'model')
        (p/'optimizer.pt').write_bytes(b'optimizer')
    return root


def test_checkpoint_is_not_ready_while_its_state_is_unwritten(tmp_path):
    root=fixture_run(tmp_path)
    assert not event_ready(root,'checkpoint-100')
    (root/'grpo_ckpts/checkpoint-100/trainer_state.json').write_text('{}')
    assert event_ready(root,'checkpoint-100')


def test_progress_snapshot_never_copies_model_or_optimizer(tmp_path):
    root=fixture_run(tmp_path)
    files=snapshot_files(root,'progress-20')
    assert root/'config.yaml' in files
    assert root/'train_log.csv' in files
    assert all(p.suffix not in ('.safetensors','.pt') for p in files)
    assert root/'rollouts_train.jsonl' not in files


def test_checkpoint_snapshot_keeps_exact_checkpoint_and_excludes_another(tmp_path):
    root=fixture_run(tmp_path)
    files=snapshot_files(root,'checkpoint-100')
    assert root/'grpo_ckpts/checkpoint-100/adapter_model.safetensors' in files
    assert root/'grpo_ckpts/checkpoint-100/optimizer.pt' in files
    assert not any('checkpoint-200' in p.parts for p in files)


def test_completed_archive_keeps_selected_adapter_and_evidence(tmp_path):
    root=fixture_run(tmp_path)
    (root/'cost.json').write_text(json.dumps({'status':'complete'}))
    best=root/'best_adapter';best.mkdir()
    (best/'adapter_model.safetensors').write_bytes(b'best')
    (best/'optimizer.pt').write_bytes(b'irrelevant')
    files=snapshot_files(root,'complete')
    assert best/'adapter_model.safetensors' in files
    assert best/'optimizer.pt' not in files
    assert root/'rollouts_train.jsonl' in files
    output=tmp_path/'archive.tar.gz';pack_snapshot(root,'complete',output)
    with tarfile.open(output) as t:
        names=t.getnames()
    assert 'R_Qwen3.5-2B_s0/best_adapter/adapter_model.safetensors' in names
    assert not any('checkpoint-200' in name for name in names)


def test_initial_eval_and_terminal_failure_are_durable_events(tmp_path):
    root=fixture_run(tmp_path)
    assert not event_ready(root,'initial_eval')
    p=root/'initial_eval';p.mkdir();(p/'metrics_dev.json').write_text('{}')
    assert event_ready(root,'initial_eval')
    assert event_ready(root,'progress-20')
    assert not event_ready(root,'progress-40')
    (root/'cost.json').write_text('{"status":"failed"}')
    assert event_ready(root,'complete')


def test_retry_preserves_the_original_archive_bytes(tmp_path):
    root=fixture_run(tmp_path)
    output=tmp_path/'archive.tar.gz'
    pack_snapshot(root,'progress-20',output)
    original=output.read_bytes()
    (root/'config.yaml').write_text('run_id: a changed live file\n')
    pack_snapshot(root,'progress-20',output)
    assert output.read_bytes()==original


def test_final_metadata_can_be_retrieved_without_model_weights(tmp_path):
    root=fixture_run(tmp_path)
    (root/'cost.json').write_text('{"status":"complete"}')
    best=root/'best_adapter';best.mkdir();(best/'adapter_model.safetensors').write_bytes(b'best')
    assert event_ready(root,'complete-metadata')
    files=snapshot_files(root,'complete-metadata')
    assert root/'cost.json' in files
    assert not any(p.suffix=='.safetensors' for p in files)
    assert root/'rollouts_train.jsonl' not in files


def test_compact_checkpoint_waits_for_valid_trainer_state(tmp_path):
    root=fixture_run(tmp_path)
    assert not event_ready(root,'checkpoint-adapter-100')
    state=root/'grpo_ckpts/checkpoint-100/trainer_state.json'
    state.write_text('{')
    assert not event_ready(root,'checkpoint-adapter-100')
    state.write_text('{}')
    assert event_ready(root,'checkpoint-adapter-100')


def test_compact_checkpoint_keeps_student_and_state_without_optimizer_or_reference(tmp_path):
    root=fixture_run(tmp_path)
    checkpoint=root/'grpo_ckpts/checkpoint-100'
    (checkpoint/'trainer_state.json').write_text('{}')
    (checkpoint/'adapter_config.json').write_text('{}')
    reference=checkpoint/'reference';reference.mkdir()
    (reference/'adapter_model.safetensors').write_bytes(b'frozen reference')
    files=snapshot_files(root,'checkpoint-adapter-100')
    for name in ('adapter_model.safetensors','adapter_config.json','trainer_state.json'):
        assert checkpoint/name in files
    assert checkpoint/'optimizer.pt' not in files
    assert not any('reference' in p.parts or 'checkpoint-200' in p.parts for p in files)
    full=snapshot_files(root,'checkpoint-100')
    assert checkpoint/'optimizer.pt' in full
    assert reference/'adapter_model.safetensors' in full
