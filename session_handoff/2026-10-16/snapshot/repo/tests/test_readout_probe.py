"""Frozen readout probes preserve exact greedy prefixes and separate decoding protocols."""
import math
from types import SimpleNamespace
import pytest
import torch


class Tokenizer:
    eos_token_id=999
    pad_token_id=0
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
    def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids if i not in (0,999))
    def apply_chat_template(self,messages,**kwargs):return [80,72]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(1000,1)
        self.answers=iter(['why</think>','contradiction</answer>'])
        self.scored=[]
    def get_input_embeddings(self):return self.embedding
    def generate(self,input_ids,**kwargs):
        return torch.cat([input_ids,torch.tensor([[ord(c) for c in next(self.answers)]])],dim=1)
    def forward(self,input_ids,**kwargs):
        self.scored.append(input_ids[0].tolist())
        keep=kwargs.get('logits_to_keep',input_ids.shape[1])
        return SimpleNamespace(logits=torch.zeros(1,keep,1000))


def helper():
    try:
        from eval.readout_probe import canonical_answer_scores,probe_indices,probe_summary
    except ModuleNotFoundError:
        pytest.fail('Frozen answer-readout diagnostic is not implemented')
    return canonical_answer_scores,probe_indices,probe_summary


def test_scores_include_every_canonical_suffix_token_and_eos():
    score,_,_=helper();model=Model();tok=Tokenizer()
    with torch.inference_mode():result=score(model,tok,[80,72])
    assert result['scores']['neutral']==pytest.approx(-17*math.log(1000),abs=1e-4)
    assert result['scores']['entailment']==pytest.approx(-20*math.log(1000),abs=1e-4)
    assert result['scores']['contradiction']==pytest.approx(-23*math.log(1000),abs=1e-4)
    assert result['map_label']=='neutral'
    assert all(row[:2]==[80,72] and row[-1]==999 for row in model.scored)


def test_probe_keeps_the_original_greedy_answer_and_exact_reasoning_prefix():
    helper()
    from eval.hf_generate import greedy_outputs
    model=Model();tok=Tokenizer();audits=[]
    output=greedy_outputs(model,tok,[[{'role':'user','content':'pair'}]],100,
                          reasoning_budget=20,answer_scores=audits)
    assert output[0][0]=='<think>why</think><answer>contradiction</answer>'
    prefix=tok.encode('PH<think>why</think><answer>')
    assert len(audits)==1 and audits[0]['prefix_length']==len(prefix)
    assert audits[0]['map_label']=='neutral'
    assert all(row[:len(prefix)]==prefix for row in model.scored)


def test_subset_selection_is_fixed_label_blind_and_never_repeats_a_row():
    _,select,_=helper()
    assert select(5,3,42)==[0,2,4]
    assert select(2,9,42)==[0,1]
    with pytest.raises(ValueError):select(5,0,42)


def test_summary_reports_repairs_and_regressions_without_hiding_greedy_results():
    _,_,summary=helper()
    greedy=[dict(id='a',gold='neutral',pred='contradiction',parse_ok=True,topic='01'),
            dict(id='b',gold='entailment',pred='entailment',parse_ok=True,topic='01')]
    mapped=[{**greedy[0],'pred':'neutral'},{**greedy[1],'pred':'neutral'}]
    result=summary(greedy,mapped)
    assert result['repairs']==1 and result['regressions']==1 and result['net_repairs']==0
    assert result['greedy']['accuracy']==result['canonical_map']['accuracy']==.5
    assert result['selection_allowed'] is False and result['split']=='dev_subset'
    with pytest.raises(ValueError):summary(greedy,list(reversed(mapped)))


def test_nonuniform_scores_match_an_independent_full_sequence_causal_sum():
    score,_,_=helper();tok=Tokenizer();prefix=[80,72,19]
    class Nonuniform(Model):
        def full(self,ids):
            positions=torch.arange(len(ids),dtype=torch.float32)[:,None]
            vocab=torch.arange(1000,dtype=torch.float32)[None,:]
            return torch.sin((positions+1)*(vocab+1)*.017)*2.3
        def forward(self,input_ids,**kwargs):
            logits=self.full(input_ids[0].tolist())
            return SimpleNamespace(logits=logits[-kwargs['logits_to_keep']:][None])
    model=Nonuniform();result=score(model,tok,prefix)
    expected={}
    for label in ['entailment','neutral','contradiction']:
        suffix=tok.encode(label+'</answer>')+[tok.eos_token_id]
        full=model.full(prefix+suffix).log_softmax(-1)
        expected[label]=sum(float(full[len(prefix)-1+j,token]) for j,token in enumerate(suffix))
    assert result['scores']==pytest.approx(expected,abs=1e-4)
    assert result['map_label']==max(expected,key=expected.get)


def test_probe_subset_files_cannot_be_collected_as_full_dev(tmp_path,monkeypatch):
    helper()
    from eval.readout_probe import predict_probe
    from analysis.collect import collect_runs
    from common.io import write_json
    import yaml
    rows=[dict(id='safe-row',dataset='vianli',topic='01',premise='P',hypothesis='H',label='neutral')]
    monkeypatch.setattr('data.prepare.get_rows',lambda split:rows)
    def fake_generate(*args,answer_scores,**kwargs):
        answer_scores.append({'map_label':'neutral','candidate_lengths':{'neutral':5},
                              'completion_prefix_length':3,'prefix_length':20})
        return [('<think>x</think><answer>neutral</answer>',8)]
    monkeypatch.setattr('eval.hf_generate.generate_transformers',fake_generate)
    cfg={'run_id':'P0R14','format':'cot','recipe':'zs','bounded_reasoning':True,
         'model':{'name':'Qwen/Qwen3.5-2B'},'eval':{'backend':'transformers','max_new_tokens':{'cot':160}},
         'readout_probe':{'n':1,'subset_seed':42},'native_pilot':{'reasoning_budget':128}}
    out=tmp_path/'P0R14_Qwen3.5-2B_s0';out.mkdir()
    (out/'config.yaml').write_text(yaml.safe_dump(cfg));write_json(out/'cost.json',{'status':'complete'})
    result=predict_probe(cfg,'dev',out,None)
    assert result['n']==1 and (out/'metrics_dev_subset.json').exists()
    assert (out/'predictions_dev_subset.jsonl').exists()
    assert not (out/'metrics_dev.json').exists() and collect_runs(tmp_path)==[]
