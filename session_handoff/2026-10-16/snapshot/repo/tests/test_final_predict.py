"""Held-out loader is pinned and never reached from an unclaimed execution."""
import json
import pytest


def test_dataset_spec_requires_vianli_test_and_pinned_revision():
    from eval.final_predict import validate_dataset_spec
    spec={'repo':'uitnlp/ViANLI','file':'vianli_test.jsonl','revision':'a'*40,'expected_n':1000}
    assert validate_dataset_spec(spec)==spec
    for key,value in [('repo','other/data'),('file','vianli_train.jsonl'),('revision','main'),('expected_n',1)]:
        with pytest.raises(ValueError):validate_dataset_spec({**spec,key:value})


def test_child_cannot_open_test_without_parent_claim(tmp_path,monkeypatch):
    from tests.test_final_protocol import _fixture
    import eval.final_predict as predict
    manifest,out=_fixture(tmp_path)
    manifest['dataset']={'repo':'uitnlp/ViANLI','file':'vianli_test.jsonl','revision':'a'*40,'expected_n':1000}
    path=tmp_path/'protocol.json';path.write_text(json.dumps(manifest))
    monkeypatch.setattr(predict,'ROOT',tmp_path)
    monkeypatch.setattr(predict,'load_locked_test_rows',lambda *a:pytest.fail('must reject before test read'))
    with pytest.raises(ValueError):predict.main(['--run','N4R9','--seed','0','--protocol',str(path)])


def test_locked_loader_hashes_hf_cache_target_and_counts_real_rows(tmp_path,monkeypatch):
    import huggingface_hub
    from eval.final_predict import load_locked_test_rows
    records=[{'uid':f'uit_Adver_{i}_1_11_01','premise':'P','hypothesis':'H','label':'entailment'} for i in range(1000)]
    blob=tmp_path/'blob';blob.write_text(''.join(json.dumps(r)+'\n' for r in records))
    link=tmp_path/'snapshot.jsonl';link.symlink_to(blob)
    monkeypatch.setenv('G4_UNLOCK','1')
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',lambda **kw:str(link))
    rows,receipt=load_locked_test_rows({'repo':'uitnlp/ViANLI','file':'vianli_test.jsonl','revision':'a'*40,'expected_n':1000})
    assert len(rows)==1000 and receipt['n']==1000
    assert len(receipt['sha256'])==64
