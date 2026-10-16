"""Two-stage rollout boundary and complete-group sampler contracts."""
import pytest

from train.native_rollout import assemble_rollout, block_sample_indices


def test_marginal_rollout_excludes_sampled_answer_from_policy_and_kl(monkeypatch):
    from types import SimpleNamespace
    import train.native_rollout as module
    class Tok:
        def encode(self,text,add_special_tokens=False):return [10] if text=='<think>' else [1]
    row=assemble_rollout([10],[20,21],[-.2,-.3],[30],[40],[-.4],[50])
    row['reasoning_tokens']=2
    monkeypatch.setattr(module,'bounded_generate',lambda *args,**kwargs:[row])
    trainer=SimpleNamespace(processing_class=Tok(),vllm_generation=SimpleNamespace(llm=None),
        native_cfg={'reasoning_budget':64,'max_completion_length':96},temperature=1.,
        _native_active=[1],label_marginal_enabled=True)
    result=module.rollout_func(['prompt'],trainer)
    assert result['env_mask']==[[0,1,1,0,0,0]]


def test_inserted_delimiters_have_no_policy_likelihood():
    out = assemble_rollout([1], [2, 3], [-0.2, -0.3], [4], [5], [-0.5], [6])
    assert out['completion_ids'] == [1, 2, 3, 4, 5, 6]
    assert out['env_mask'] == [0, 1, 1, 0, 1, 0]
    assert out['logprobs'] == [0.0, -0.2, -0.3, 0.0, -0.5, 0.0]
    assert out['prefix_ids'] == [1, 2, 3, 4]


def test_custom_rollout_likelihood_shape_matches_trl_token_matrix():
    torch = pytest.importorskip('torch')
    from trl.trainer.utils import pad
    rows = [assemble_rollout([1], [2], [-.2], [3], [4], [-.4], [5]),
            assemble_rollout([1], [2, 6], [-.2, -.6], [3], [4], [-.4], [5])]
    ids = pad([torch.tensor(r['completion_ids']) for r in rows])
    likelihoods = pad([torch.tensor(r['logprobs']) for r in rows])
    assert likelihoods.shape == ids.shape


def test_rollout_rejects_more_than_reasoning_budget_or_total_budget():
    with pytest.raises(ValueError):
        assemble_rollout([1], list(range(65)), [-1.] * 65, [4], [5], [-1.], [6])
    with pytest.raises(ValueError):
        assemble_rollout([1], [2], [-1.], [4], list(range(93)), [-1.] * 93, [6])


def test_sampler_preserves_complete_groups_and_prompt_repeats():
    indices = block_sample_indices(10, slots=3, rollouts=4, groups_per_batch=8, seed=0)
    assert len(indices) == 8 * 3 * 4
    blocks = [indices[i:i+12] for i in range(0, len(indices), 12)]
    assert len({block[0] // 3 for block in blocks}) == 8
    for block in blocks:
        g = block[0] // 3
        assert block == [g * 3 + s for s in range(3) for _ in range(4)]
    assert indices == block_sample_indices(10, slots=3, rollouts=4, groups_per_batch=8, seed=0)


def test_sampler_no_partial_batch():
    assert len(block_sample_indices(17, slots=3, rollouts=4, groups_per_batch=8, seed=2)) == 192


def test_sampler_repeats_the_whole_generation_batch_for_trl_microsteps():
    ordinary = block_sample_indices(17, seed=0)
    repeated = block_sample_indices(17, seed=0, repeat_count=24)
    assert len(repeated) == len(ordinary) * 24
    for update in range(2):
        target = ordinary[update * 96:(update + 1) * 96]
        for micro in range(24):
            start = (update * 24 + micro) * 96
            assert repeated[start:start+96] == target


@pytest.mark.parametrize('first_stop', [11, 99])
def test_vllm_retained_stop_ids_are_removed_before_second_stage(monkeypatch, first_stop):
    import sys
    from types import SimpleNamespace
    from train.native_rollout import bounded_generate
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    class Tok:
        eos_token_id = 99
        def encode(self, text, add_special_tokens=False):
            return {'<think>': [10], '</think>': [11], '<answer>': [12],
                    '</think><answer>': [11, 12], '</answer>': [13]}[text]
        def decode(self, ids, **kw):
            return str(ids)
    class Engine:
        calls = 0
        def generate(self, prompts, params, **kw):
            self.calls += 1
            if self.calls == 1:
                assert prompts == [{'prompt_token_ids': [1, 10]}]
                ids, text = [21, first_stop], 'reason'
            else:
                assert prompts == [{'prompt_token_ids': [1, 10, 21, 11, 12]}]
                ids, text = [31, 13], 'neutral'
            sample = SimpleNamespace(token_ids=ids, text=text,
                     logprobs=[{i: SimpleNamespace(logprob=-.1)} for i in ids])
            return [SimpleNamespace(outputs=[sample])]
    result = bounded_generate(Engine(), [[1]], Tok())[0]
    assert result['completion_ids'] == [10, 21, 11, 12, 31, 13, 99]
    assert result['env_mask'] == [0, 1, 0, 0, 1, 0, 0]


def test_stop_marker_inside_merged_token_is_censored_without_losing_answer(monkeypatch):
    import sys
    from types import SimpleNamespace
    from train.native_rollout import bounded_generate
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    class Tok:
        eos_token_id = 99
        def encode(self, text, add_special_tokens=False):
            return {'<think>': [10], '</think>': [11], '<answer>': [12],
                    '</think><answer>': [11, 12], '</answer>': [14, 15, 17]}[text]
        def decode(self, ids, **kw):
            parts={10:'<think>',11:'</think>',12:'<answer>',14:'</',15:'answer',17:'>',16:'>_',21:'reason',31:'neutral',99:'EOS'}
            return ''.join(parts[i] for i in ids)
    class Engine:
        calls = 0
        def generate(self, prompts, params, **kw):
            self.calls += 1
            ids=[21,11] if self.calls==1 else [31,14,15,16]
            sample=SimpleNamespace(token_ids=ids,logprobs=[{i:SimpleNamespace(logprob=-.1)} for i in ids])
            return [SimpleNamespace(outputs=[sample])]
    row=bounded_generate(Engine(),[[1]],Tok())[0]
    assert row['completion_ids']==[10,21,11,12,31,14,15,17,99]
    assert row['raw_output']=='<think>reason</think><answer>neutral</answer>'


def test_staged_decoder_does_not_sample_another_open_think_tag(monkeypatch):
    import sys
    from types import SimpleNamespace
    from train.native_rollout import bounded_generate
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=lambda **kw:SimpleNamespace(**kw)))
    class Tok:
        eos_token_id=99
        def encode(self,text,**kw):
            return {'<think>':[10],'</think>':[11],'<answer>':[12],'</think><answer>':[11,12],'</answer>':[13]}[text]
        def decode(self,ids,**kw): return str(ids)
    class Engine:
        calls=0
        def generate(self,prompts,params,**kw):
            self.calls+=1
            if self.calls==1: assert '<think>' in params.bad_words
            ids=[21] if self.calls==1 else [31]
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids,logprobs=[{i:SimpleNamespace(logprob=-.1)} for i in ids])])]
    bounded_generate(Engine(),[[1]],Tok())
