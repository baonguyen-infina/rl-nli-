"""Runner phải chờ tiến trình train thoát trước eval và bảo toàn config cũ."""
import json
import sys
from pathlib import Path
import pytest
from scripts.run import run_training_process, select_checkpoint


def test_training_child_exits_before_parent_continues(tmp_path):
    worker = tmp_path / 'worker.py'
    marker = tmp_path / 'alive'
    worker.write_text('import pathlib,sys,time\np=pathlib.Path(sys.argv[1]);p.write_text("owned")\ntime.sleep(.05)\np.unlink()\n')
    run_training_process([sys.executable, str(worker), str(marker)])
    assert not marker.exists()


def test_training_failure_propagates():
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):
        run_training_process([sys.executable, '-c', 'raise SystemExit(7)'])


def test_checkpoint_selection_uses_greedy_full_dev_macro_f1(tmp_path):
    ckpts = [tmp_path/'checkpoint-9', tmp_path/'checkpoint-100']
    for p in ckpts: p.mkdir()
    # Loss/accuracy không được thay thế Macro-F1; tie ưu tiên bước số lớn hơn.
    scores = {'checkpoint-9': {'macro_f1': .55, 'accuracy': .9},
              'checkpoint-100': {'macro_f1': .55, 'accuracy': .6}}
    assert select_checkpoint(ckpts, scores).name == 'checkpoint-100'


def test_resumed_eval_reads_saved_config_not_current_hyperparameters(tmp_path):
    from scripts.run import frozen_config
    path=tmp_path/'config.yaml'
    path.write_text('model:\n  name: archived-model\neval:\n  max_model_len: 4096\n')
    cfg=frozen_config(path)
    assert cfg['model']['name'] == 'archived-model'
    assert cfg['eval']['max_model_len'] == 4096


def test_predict_subprocess_receives_frozen_config(tmp_path, monkeypatch):
    import scripts.run as runner
    seen = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda cmd, check: seen.append(cmd))
    runner._predict_cli('D9', 0, 'dev', tmp_path, config_path=tmp_path/'config.yaml')
    assert '--config' in seen[0]
    assert str(tmp_path/'config.yaml') in seen[0]


def test_prompt_only_sft_control_uses_the_same_seed_adapter(tmp_path):
    from scripts.run import resolve_initial_adapter
    for seed in [0,1]:
        (tmp_path/f'D3_Qwen3.5-2B_s{seed}'/'best_adapter').mkdir(parents=True)
    assert resolve_initial_adapter({'init_from':'D3'},1,tmp_path)==tmp_path/'D3_Qwen3.5-2B_s1'/'best_adapter'
    with pytest.raises(FileNotFoundError):resolve_initial_adapter({'init_from':'D3'},2,tmp_path)
    assert resolve_initial_adapter({},0,tmp_path) is None
