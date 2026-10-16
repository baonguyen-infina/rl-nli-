"""Final inference views are fixed before held-out access, including model revision."""
import pytest


def test_final_views_share_backend_revision_and_fixed_format():
    from common.io import load_config
    for source in ('D3R9','N1R9','N4R9','N5R10','N6R11','N0R11','N7R12'):
        cfg=load_config('H'+source+'L')
        assert cfg['init_from']==source and cfg['format']=='label'
        assert cfg['eval']['backend']=='transformers'
        assert cfg['eval']['model_revision']=='15852e8c16360a2fea060d615a32b45270f8a8fc'
        assert cfg['recipe']=='zs' and cfg['seeds']==[0]
        assert not cfg.get('bounded_reasoning')
    cfg=load_config('HN6R11P200L')
    assert cfg['init_from']=='N6R11' and cfg['init_checkpoint']==200


def test_explicit_ablation_checkpoint_stays_in_same_seed_run(tmp_path):
    from scripts.run import resolve_initial_adapter
    path=tmp_path/'N6R11_Qwen3.5-2B_s0/grpo_ckpts/checkpoint-200'
    path.mkdir(parents=True)
    assert resolve_initial_adapter({'init_from':'N6R11','init_checkpoint':200},0,tmp_path)==path
    with pytest.raises(FileNotFoundError):
        resolve_initial_adapter({'init_from':'N6R11','init_checkpoint':200},1,tmp_path)
    for value in ('../../x',True,-1,0):
        with pytest.raises(ValueError):
            resolve_initial_adapter({'init_from':'N6R11','init_checkpoint':value},0,tmp_path)


def test_pinned_model_snapshot_only_uses_local_cached_revision(monkeypatch):
    import huggingface_hub
    from eval.hf_generate import resolve_model_snapshot
    calls=[]
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda **kw:calls.append(kw) or '/cached/pinned')
    assert resolve_model_snapshot('Qwen/Qwen3.5-2B','a'*40)=='/cached/pinned'
    assert calls==[dict(repo_id='Qwen/Qwen3.5-2B',revision='a'*40,local_files_only=True)]
    assert resolve_model_snapshot('Qwen/Qwen3.5-2B',None)=='Qwen/Qwen3.5-2B'
    with pytest.raises(ValueError):resolve_model_snapshot('Qwen/Qwen3.5-2B','main')


def test_pinned_revision_reaches_transformers_backend(monkeypatch):
    import eval.hf_generate as hf
    from eval.predict import generate
    seen={}
    def fake(*args,**kwargs):seen.update(kwargs);return []
    monkeypatch.setattr(hf,'generate_transformers',fake)
    generate('Qwen/Qwen3.5-2B',[],16,backend='transformers',model_revision='a'*40)
    assert seen['model_revision']=='a'*40
