"""Keep legacy eval defaults and prespecify reproducible runtime probes."""
import sys
from types import ModuleType,SimpleNamespace
import pytest


@pytest.fixture
def engine(monkeypatch):
    calls=[]
    fake=ModuleType('vllm')
    class LLM:
        def __init__(self,**kwargs):calls.append(kwargs)
        def chat(self,messages,params,**kwargs):
            return [SimpleNamespace(outputs=[SimpleNamespace(text='<answer>neutral</answer>',token_ids=[1,2])])
                    for message in messages]
    fake.LLM=LLM;fake.SamplingParams=lambda **kw:SimpleNamespace(**kw)
    request=ModuleType('vllm.lora.request');request.LoRARequest=lambda *args:args
    monkeypatch.setitem(sys.modules,'vllm',fake)
    monkeypatch.setitem(sys.modules,'vllm.lora.request',request)
    return calls


def test_optional_engine_controls_reach_vllm_and_preserve_greedy_outputs(engine):
    from eval.predict import generate
    options=dict(max_num_seqs=1,enforce_eager=True,enable_prefix_caching=False,seed=0,max_num_batched_tokens=2048)
    outputs=generate('test/model',[[{'role':'user','content':'P H'}]],16,engine_options=options)
    assert outputs==[('<answer>neutral</answer>',2)]
    assert all(engine[0][key]==value for key,value in options.items())
    assert engine[0]['language_model_only'] is True
    assert engine[0]['dtype']=='bfloat16'


def test_legacy_eval_keeps_default_scheduler_arguments(engine):
    from eval.predict import generate
    generate('test/model',[[{'role':'user','content':'P H'}]],16)
    assert not {'max_num_seqs','enforce_eager','seed','enable_prefix_caching'}&set(engine[0])


@pytest.mark.parametrize('options',[{'max_num_seqs':0},{'max_num_seqs':-1},{'model':'unexpected'},
    {'max_num_batched_tokens':0},{'seed':-1},{'enforce_eager':'false'}])
def test_invalid_engine_controls_fail_before_initializing_gpu(engine,options):
    from eval.predict import generate
    with pytest.raises(ValueError):generate('test/model',[],16,engine_options=options)
    assert engine==[]


def test_fixed_validation_protocol_uses_shared_base_preset(monkeypatch,tmp_path):
    import eval.predict as predict
    captured={}
    monkeypatch.setattr(predict,'get_rows',lambda split:[])
    def fake_generate(*args,**kwargs):captured.update(kwargs);return []
    monkeypatch.setattr(predict,'generate',fake_generate)
    monkeypatch.setattr(predict,'compute_metrics',lambda records:{'n':0})
    controls=dict(max_num_seqs=1,enforce_eager=True,enable_prefix_caching=False,seed=0,max_num_batched_tokens=2048)
    cfg=dict(model={'name':'test/model'},format='label',eval={'max_new_tokens':{'label':16},
        'gpu_memory_utilization':.65,'max_model_len':2048},validation_protocol='fixed_single_sequence',
        validation_eval={'engine_options':controls})
    predict.predict_split(cfg,'dev',tmp_path)
    assert captured['engine_options']==controls


def test_explicit_transformers_backend_never_initializes_vllm(engine,monkeypatch):
    import eval.hf_generate as hf
    from eval.predict import generate
    captured={}
    def fake(*args,**kwargs):
        captured.update(kwargs)
        return [('<answer>neutral</answer>',3)]
    monkeypatch.setattr(hf,'generate_transformers',fake)
    assert generate('test/model',[],16,backend='transformers')==[('<answer>neutral</answer>',3)]
    assert engine==[]


def test_unknown_eval_backend_rejected_before_loading_model(engine):
    from eval.predict import generate
    with pytest.raises(ValueError):generate('test/model',[],16,backend='unknown')
    assert engine==[]
