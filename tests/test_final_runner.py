"""The experiment entry point prepares locked eval views and never silently reruns test."""
import json
import pytest


def test_prepare_phase_freezes_config_without_training_or_evaluating(tmp_path,monkeypatch):
    import scripts.run as run
    import common.io as io
    monkeypatch.setattr(io,'RESULTS',tmp_path)
    monkeypatch.setattr(run,'load_config',lambda _: {'run_id':'T1','recipe':'zs','format':'label',
        'model':{'name':'test/model'},'seeds':[0]})
    monkeypatch.setattr(run,'run_training_process',lambda _:pytest.fail('no training'))
    monkeypatch.setattr(run,'_predict_cli',lambda *a,**kw:pytest.fail('no inference'))
    run.main(['--run','T1','--seed','0','--phase','prepare'])
    out=tmp_path/'T1_model_s0'
    assert (out/'config.yaml').exists()
    assert json.loads((out/'cost.json').read_text())['status']=='prepared'


def test_test_phase_requires_a_protocol_before_any_execution(monkeypatch):
    import scripts.run as run
    monkeypatch.setattr(run,'load_config',lambda _:pytest.fail('must reject before model/config loading'))
    with pytest.raises(ValueError,match='protocol'):
        run.main(['--run','T1','--phase','test'])


def test_final_test_reuses_receipt_and_preserves_development_cost(tmp_path,monkeypatch):
    import scripts.run as run
    from tests.test_final_protocol import _fixture,_outputs
    manifest,out=_fixture(tmp_path)
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(manifest))
    (out/'cost.json').write_text('{"status":"complete","wall_seconds":123}')
    previous=(out/'cost.json').read_bytes();calls=[]
    monkeypatch.setattr(run,'RESULTS',tmp_path/'results')
    def child(entry,path):
        calls.append((entry['run_id'],path));_outputs(out)
    monkeypatch.setattr(run,'_predict_final_cli',child)
    run.run_final_test('N4R9',0,protocol)
    assert len(calls)==1
    assert (out/'cost.json').read_bytes()==previous
    assert json.loads((out/'cost_test.json').read_text())['status']=='complete'
    run.run_final_test('N4R9',0,protocol)
    assert len(calls)==1


def test_unknown_run_is_rejected_without_reading_test(tmp_path,monkeypatch):
    import scripts.run as run
    from tests.test_final_protocol import _fixture
    manifest,out=_fixture(tmp_path)
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(manifest))
    monkeypatch.setattr(run,'RESULTS',tmp_path/'results')
    monkeypatch.setattr(run,'_predict_final_cli',lambda *a:pytest.fail('must not execute'))
    with pytest.raises(ValueError):run.run_final_test('UNKNOWN',0,protocol)
