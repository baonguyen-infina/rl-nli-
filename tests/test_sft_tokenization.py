"""Pin the inference prefix and target mask without GPU libraries."""
from train.common import tokenize_sft_example

class Tokenizer:
    eos_token_id = 99
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and enable_thinking is False
        return [10, 11, 12]
    def encode(self, text, add_special_tokens):
        assert text == '<answer>neutral</answer>' and add_special_tokens is False
        return [20, 21]


def test_training_masks_prompt_and_retains_complete_answer():
    row = {'prompt':[{'role':'user','content':'p'}], 'completion':[{'role':'assistant','content':'<answer>neutral</answer>'}]}
    item = tokenize_sft_example(row, Tokenizer(), max_length=6)
    assert item['input_ids'] == [10,11,12,20,21,99]
    assert item['labels'] == [-100,-100,-100,20,21,99]
    assert item['attention_mask'] == [1,1,1,1,1,1]


def test_long_examples_are_rejected_instead_of_cutting_answer():
    import pytest
    row = {'prompt':[{'role':'user','content':'p'}], 'completion':[{'role':'assistant','content':'<answer>neutral</answer>'}]}
    with pytest.raises(ValueError, match='max_length'):
        tokenize_sft_example(row, Tokenizer(), max_length=5)


def test_transformers_batch_encoding_prefix_contains_token_ids():
    class DictTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return {'input_ids':[10,11,12], 'attention_mask':[1,1,1]}
    row={'prompt':[{'role':'user','content':'p'}], 'completion':[{'role':'assistant','content':'<answer>neutral</answer>'}]}
    assert tokenize_sft_example(row, DictTokenizer(), 6)['input_ids'] == [10,11,12,20,21,99]


def test_custom_collator_columns_are_not_removed(monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    from train.sft import _make_sft_config
    fake = ModuleType('trl')
    fake.SFTConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, 'trl', fake)
    cfg = _make_sft_config({'lr':2e-5,'epochs':4,'scheduler':'cosine','per_device_batch':32,'grad_accum':1,'max_len':512},0,tmp_path)
    assert cfg['remove_unused_columns'] is False
