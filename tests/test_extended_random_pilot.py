"""The next full-budget RL run preserves training semantics and declares its readout."""
from common.io import load_config
from train.native_groups import build_native_blocks,build_scheduled_random_blocks


def test_full_random_run_uses_original_horizon_reference_and_label_readout():
    original=load_config('N7R12');full=load_config('N12R15')
    assert full['native_pilot']=={k:v for k,v in original['native_pilot'].items() if k!='stop_after_steps'}
    assert full['native_pilot']['max_steps']==400
    assert full['init_from']=='D3R9' and full.get('init_checkpoint') is None
    assert full['native_mode']==original['native_mode']=='gated_joint'
    assert full['native_grouping']==original['native_grouping']=='random'
    assert full['training_format']=='cot' and full['format']=='label'
    assert full['bounded_reasoning'] is False
    assert full['eval']['backend']=='transformers'
    assert full['eval']['model_revision']=='15852e8c16360a2fea060d615a32b45270f8a8fc'
    smoke=load_config('G0N12R15P')
    assert smoke['native_pilot']['max_steps']==1 and smoke['native_pilot']['save_steps']==1
    assert smoke['training_format']=='cot' and smoke['format']=='label'


def test_trace_swap_followup_changes_only_credit_assignment_and_keeps_fixed_readout():
    base=load_config('N12R15');trace=load_config('N13R16')
    assert trace['init_from']==base['init_from']=='D3R9'
    assert trace['native_mode']=='trace_swap'
    assert trace['native_grouping']==base['native_grouping']=='random'
    assert trace['native_pilot']==base['native_pilot']
    assert trace['training_format']==base['training_format']=='cot'
    assert trace['format']==base['format']=='label'
    assert trace['validation_protocol']==base['validation_protocol']=='fixed_single_sequence'
    assert trace['eval']==base['eval']
    smoke=load_config('G0N13R16P')
    assert smoke['native_mode']=='trace_swap' and smoke['native_pilot']['max_steps']==1


def test_extending_random_construction_keeps_each_previous_update_unchanged():
    rows=[{'id':str(group)+'-'+label,'premise':'P'+str(group),'hypothesis':'H'+str(group)+label,
           'label':label} for group in range(24) for label in ['entailment','neutral','contradiction']]
    native=build_native_blocks(rows,seed=0)
    order=[(i*7)%24 for i in range(24)]
    pilot=build_scheduled_random_blocks(native,order[:12],groups_per_batch=4,seed=0)
    full=build_scheduled_random_blocks(native,order,groups_per_batch=4,seed=0)
    for group in order[:12]:assert full[group]==pilot[group]
    assert sorted(r['id'] for group in full for r in group)==sorted(r['id'] for r in rows)


def test_r15_setup_and_pipeline_stdout_are_collected_without_model_download(tmp_path):
    from scripts.remote_logs import collect_logs
    for name in ['r15_run_dependencies.log','r15_run_red.log','r15_run_full_tests.log',
                 'native_r15_fullrandom_pipeline.log']:
        (tmp_path/name).write_text('setup and experiment stdout\n')
        assert 'runtime/'+name in collect_logs(tmp_path,['N12R15'],'fullrandom')


def test_trace_saliency_followup_is_matched_except_for_credit_assignment_and_horizon():
    base=load_config('N12R15');trace=load_config('N14R17')
    assert trace['init_from']==base['init_from']=='D3R9'
    assert trace['native_mode']=='trace_saliency'
    assert trace['native_grouping']==base['native_grouping']=='random'
    assert trace['training_format']==base['training_format']=='cot'
    assert trace['format']==base['format']=='label'
    assert trace['validation_protocol']==base['validation_protocol']=='fixed_single_sequence'
    assert trace['eval']==base['eval']
    assert trace['native_pilot']['stop_after_steps']==200 and trace['native_pilot']['max_steps']==base['native_pilot']['max_steps']
    assert trace['native_pilot']['groups_per_update']==base['native_pilot']['groups_per_update']
    smoke=load_config('G0N14R17P')
    assert smoke['native_mode']=='trace_saliency' and smoke['native_pilot']['max_steps']==1


def test_bounded_counterfactual_followup_keeps_registered_full_budget_and_readout():
    base=load_config('N12R15');bounded=load_config('N15R18')
    assert bounded['native_mode']=='gated_trace_swap'
    assert bounded['native_pilot']==base['native_pilot']
    assert bounded['init_from']==base['init_from']=='D3R9'
    assert bounded['native_grouping']==base['native_grouping']=='random'
    assert bounded['training_format']==base['training_format']=='cot'
    assert bounded['format']==base['format']=='label'
    assert bounded['eval']==base['eval']
    smoke=load_config('G0N15R18P')
    assert smoke['native_mode']=='gated_trace_swap' and smoke['native_pilot']['max_steps']==1


def test_native_swap_comparison_changes_only_training_group_membership():
    random=load_config('N15R18');native=load_config('N16R19')
    assert native['native_grouping']=='native' and random['native_grouping']=='random'
    assert {k:v for k,v in native.items() if k not in ('run_id','native_grouping')}=={
        k:v for k,v in random.items() if k not in ('run_id','native_grouping')}
    smoke=load_config('G0N16R19P')
    assert smoke['native_mode']==native['native_mode']=='gated_trace_swap'
    assert smoke['native_grouping']=='native' and smoke['native_pilot']['max_steps']==1
