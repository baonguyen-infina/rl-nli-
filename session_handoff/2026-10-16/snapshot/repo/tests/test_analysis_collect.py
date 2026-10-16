import json
import yaml
from analysis.collect import collect_runs


def test_collect_only_completed_frozen_runs_and_preserve_seed(tmp_path):
    for seed,status in [(0,'complete'),(1,'failed')]:
        out=tmp_path/f'D18_Qwen3.5-2B_s{seed}';out.mkdir()
        (out/'config.yaml').write_text(yaml.safe_dump({'run_id':'D18','recipe':'gta','format':'align','model':{'name':'Qwen/Qwen3.5-2B'}}))
        (out/'cost.json').write_text(json.dumps({'status':status,'wall_seconds':42}))
        (out/'metrics_dev.json').write_text(json.dumps({'n':2,'accuracy':.5,'macro_f1':.4,'parse_rate':.5}))
        records=[{'pred':'neutral','rule_label':'neutral','grounded':True,'coverage':.6}, {'pred':None,'rule_label':None,'grounded':None,'coverage':None}]
        (out/'predictions_dev.jsonl').write_text('\n'.join(map(json.dumps,records)))
    rows=collect_runs(tmp_path)
    assert len(rows)==1 and rows[0]['seed']==0
    assert rows[0]['macro_f1']==.4 and rows[0]['rule_consistency']==.5
    assert rows[0]['structure_rate']==.5


def test_collect_preserves_inference_protocol_and_adapter_parent(tmp_path):
    out=tmp_path/'V4R4B_Qwen3.5-2B_s0';out.mkdir()
    cfg={'run_id':'V4R4B','recipe':'zs','format':'cot',
         'model':{'name':'Qwen/Qwen3.5-2B'},'init_from':'N4R4',
         'native_mode':'allocation','scoring_protocol':'deterministic',
         'validation_protocol':'fixed_single_sequence',
         'validation_eval':{'engine_options':{'max_num_seqs':1,'seed':0}}}
    (out/'config.yaml').write_text(yaml.safe_dump(cfg))
    (out/'cost.json').write_text(json.dumps({'status':'complete'}))
    (out/'metrics_dev.json').write_text(json.dumps({'n':1000,'accuracy':.55,'macro_f1':.56,'parse_rate':1.}))
    row=collect_runs(tmp_path)[0]
    assert row['init_from']=='N4R4' and row['native_mode']=='allocation'
    assert row['validation_protocol']=='fixed_single_sequence'
    assert json.loads(row['eval_engine_options'])=={'max_num_seqs':1,'seed':0}
    assert row['initial_macro_f1'] is None and row['selected_checkpoint'] is None


def test_collect_initial_score_and_selection_from_recorded_artifacts(tmp_path):
    out=tmp_path/'N4R4_Qwen3.5-2B_s0';out.mkdir()
    cfg={'run_id':'N4R4','recipe':'native','format':'cot',
         'model':{'name':'Qwen/Qwen3.5-2B'},'init_from':'D3R3'}
    (out/'config.yaml').write_text(yaml.safe_dump(cfg))
    (out/'cost.json').write_text(json.dumps({'status':'complete'}))
    (out/'metrics_dev.json').write_text(json.dumps({'n':1000,'accuracy':.55,'macro_f1':.56,'parse_rate':1.}))
    (out/'initial_eval').mkdir()
    (out/'initial_eval/metrics_dev.json').write_text(json.dumps({'macro_f1':.52}))
    (out/'ckpt_scores.json').write_text(json.dumps({'best':'checkpoint-100','selection_split':'dev','selection_metric':'macro_f1'}))
    row=collect_runs(tmp_path)[0]
    assert row['initial_macro_f1']==.52 and row['selected_checkpoint']=='checkpoint-100'
    assert row['selection_split']=='dev' and row['selection_metric']=='macro_f1'
    assert row['validation_protocol'] is None


def test_test_collection_uses_completed_test_cost_and_receipt(tmp_path):
    from tests.test_final_protocol import _fixture,_outputs,PROTOCOL_SHA
    from eval.final_protocol import claim_test_run,complete_test_run
    manifest,out=_fixture(tmp_path)
    claim_test_run(out,PROTOCOL_SHA);_outputs(out)
    (out/'cost_test.json').write_text(json.dumps({'status':'complete','wall_seconds':5}))
    complete_test_run(out,PROTOCOL_SHA)
    rows=collect_runs(tmp_path/'results','test')
    assert len(rows)==1 and rows[0]['split']=='test'
    assert rows[0]['wall_seconds']==5 and rows[0]['macro_f1']==.5


def test_test_collection_ignores_unclaimed_results_even_with_completed_training(tmp_path):
    from tests.test_final_protocol import _fixture,_outputs
    _,out=_fixture(tmp_path);_outputs(out)
    (out/'cost.json').write_text(json.dumps({'status':'complete'}))
    (out/'cost_test.json').write_text(json.dumps({'status':'complete'}))
    assert collect_runs(tmp_path/'results','test')==[]
