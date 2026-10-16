from pathlib import Path
import yaml
import pytest


def test_native_full_pass_uses_shared_profile_without_changing_other_parameters(tmp_path,monkeypatch):
    import common.io as io
    monkeypatch.setattr(io,'ROOT',tmp_path)
    monkeypatch.setattr(io,'RUN_CONFIGS',tmp_path/'configs/runs')
    (tmp_path/'configs/runs').mkdir(parents=True)
    base={'native_pilot':{'max_steps':200,'lr':2e-6,'reasoning_budget':64},
          'native_schedules':{'full_pass':{'max_steps':400}}}
    (tmp_path/'configs/base.yaml').write_text(yaml.safe_dump(base))
    for name in ('N1R7','N4R7','N5R7'):
        (tmp_path/'configs/runs'/f'{name}.yaml').write_text(yaml.safe_dump({'native_schedule':'full_pass'}))
        assert io.load_config(name)['native_pilot']=={'max_steps':400,'lr':2e-6,'reasoning_budget':64}
    (tmp_path/'configs/runs/legacy.yaml').write_text('{}')
    assert io.load_config('legacy')['native_pilot']['max_steps']==200


def test_unknown_native_schedule_raises(tmp_path,monkeypatch):
    import common.io as io
    monkeypatch.setattr(io,'ROOT',tmp_path)
    monkeypatch.setattr(io,'RUN_CONFIGS',tmp_path/'configs/runs')
    (tmp_path/'configs/runs').mkdir(parents=True)
    (tmp_path/'configs/base.yaml').write_text('native_pilot: {}\n')
    (tmp_path/'configs/runs/x.yaml').write_text('native_schedule: missing\n')
    with pytest.raises(ValueError):io.load_config('x')


def test_new_protocol_profiles_resolve_all_budgets_without_changing_legacy(tmp_path,monkeypatch):
    import common.io as io
    monkeypatch.setattr(io,'ROOT',tmp_path)
    monkeypatch.setattr(io,'RUN_CONFIGS',tmp_path/'configs/runs')
    (tmp_path/'configs/runs').mkdir(parents=True)
    base={'sft':{'epochs':4,'lr':2e-5,'early_stopping_patience':3},
          'native_pilot':{'max_steps':200,'lr':2e-6,'reasoning_budget':64,'max_completion_length':96,'score_microbatch':4},
          'eval':{'max_new_tokens':{'label':16,'cot':96}},
          'sft_schedules':{'full_budget':{'early_stopping_patience':None}},
          'native_schedules':{'full_reasoning':{'max_steps':400,'reasoning_budget':128,'max_completion_length':160,'score_microbatch':12}},
          'eval_schedules':{'full_reasoning':{'max_new_tokens':{'cot':160}}}}
    (tmp_path/'configs/base.yaml').write_text(yaml.safe_dump(base))
    (tmp_path/'configs/runs/new.yaml').write_text(yaml.safe_dump({'sft_schedule':'full_budget','native_schedule':'full_reasoning','eval_schedule':'full_reasoning'}))
    cfg=io.load_config('new')
    assert cfg['sft']=={'epochs':4,'lr':2e-5,'early_stopping_patience':None}
    assert cfg['native_pilot']['reasoning_budget']==128
    assert cfg['native_pilot']['max_steps']==400 and cfg['native_pilot']['lr']==2e-6
    assert cfg['native_pilot']['score_microbatch']==12
    assert cfg['eval']['max_new_tokens']=={'label':16,'cot':160}
    (tmp_path/'configs/runs/legacy.yaml').write_text('{}')
    legacy=io.load_config('legacy')
    assert legacy['sft']['early_stopping_patience']==3 and legacy['native_pilot']['reasoning_budget']==64


@pytest.mark.parametrize('selector',['sft_schedule','eval_schedule'])
def test_unknown_additional_profile_is_rejected(tmp_path,monkeypatch,selector):
    import common.io as io
    monkeypatch.setattr(io,'ROOT',tmp_path)
    monkeypatch.setattr(io,'RUN_CONFIGS',tmp_path/'configs/runs')
    (tmp_path/'configs/runs').mkdir(parents=True)
    (tmp_path/'configs/base.yaml').write_text('{}')
    (tmp_path/'configs/runs/x.yaml').write_text(yaml.safe_dump({selector:'missing'}))
    with pytest.raises(ValueError):io.load_config('x')


def test_prefix_budget_preserves_long_schedule_and_stops_after_saved_step():
    from train.native_training import stop_after_callback
    from types import SimpleNamespace
    callback=stop_after_callback(200)
    control=SimpleNamespace(should_training_stop=False,should_save=False)
    args=SimpleNamespace(max_steps=400)
    callback.on_step_end(args,SimpleNamespace(global_step=199),control)
    assert not control.should_training_stop
    callback.on_step_end(args,SimpleNamespace(global_step=200),control)
    assert control.should_training_stop and control.should_save
    assert args.max_steps==400


@pytest.mark.parametrize('limit',[0,-1,True,2.5])
def test_prefix_budget_rejects_nonpositive_or_noninteger_limits(limit):
    from train.native_training import stop_after_callback
    with pytest.raises(ValueError):stop_after_callback(limit)


def test_random_group_ablation_keeps_gated_policy_and_direct_channels_and_lr_horizon():
    from common.io import load_config
    native=load_config('N6R11');random=load_config('N7R12')
    assert native['native_mode']==random['native_mode']=='gated_joint'
    assert random['native_grouping']=='random'
    assert random['init_from']==native['init_from']=='D3R9'
    assert random['native_pilot']['max_steps']==native['native_pilot']['max_steps']==400
    control=dict(random['native_pilot']);assert control.pop('stop_after_steps')==200
    assert control==native['native_pilot']
