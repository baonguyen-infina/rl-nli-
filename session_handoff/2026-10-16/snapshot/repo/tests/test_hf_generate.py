"""Greedy HF evaluation uses identical bounded syntax without sampling or UID inputs."""
import torch
import pytest
from eval.hf_generate import greedy_outputs, trim_generated_tokens


class Tokenizer:
    eos_token_id = 999
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids if i not in (0, 999))
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        return [80, 72]


class Generator(torch.nn.Module):
    def __init__(self, continuations):
        super().__init__()
        self.embedding = torch.nn.Embedding(1000, 1)
        self.continuations = iter(continuations)
        self.calls = []
    def get_input_embeddings(self):
        return self.embedding
    def generate(self, input_ids, **kwargs):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert kwargs['do_sample'] is False
        self.calls.append((input_ids[0].tolist(), kwargs))
        ids = [ord(c) for c in next(self.continuations)]
        return torch.cat((input_ids, torch.tensor([ids], device=input_ids.device)), dim=1)


def test_trim_stops_at_first_marker_or_eos():
    tok=Tokenizer()
    assert tok.decode(trim_generated_tokens(tok.encode('why</think>extra'),tok,['</think>','<answer>']))=='why'
    assert trim_generated_tokens([65,999,66],tok,[])==[65]


def test_label_generation_is_greedy_and_keeps_output_schema():
    model=Generator(['<answer>neutral</answer>']);tok=Tokenizer()
    result=greedy_outputs(model,tok,[[{'role':'user','content':'pair'}]],32)
    assert result==[('<answer>neutral</answer>',24)]
    assert model.calls[0][0]==[80,72]
    assert model.calls[0][1]['max_new_tokens']==32


def test_bounded_generation_uses_same_two_stages_and_counts_inserted_tokens():
    model=Generator(['why</think>','neutral</answer>']);tok=Tokenizer()
    result=greedy_outputs(model,tok,[[{'role':'user','content':'pair'}]],100,reasoning_budget=20)
    assert result[0][0]=='<think>why</think><answer>neutral</answer>'
    assert result[0][1]==len(result[0][0])+1
    assert tok.decode(model.calls[0][0])=='PH<think>'
    assert tok.decode(model.calls[1][0])=='PH<think>why</think><answer>'
    assert model.calls[0][1]['max_new_tokens']==20
    expected=100-len('<think>why</think><answer></answer>')-1
    assert model.calls[1][1]['max_new_tokens']==expected


@pytest.mark.parametrize('budget,reasoning',[(0,None),(10,20),(50,0),(50,-1)])
def test_bad_budgets_fail_before_generation(budget,reasoning):
    model=Generator([])
    with pytest.raises(ValueError):greedy_outputs(model,Tokenizer(),[],budget,reasoning_budget=reasoning)
    assert not model.calls
