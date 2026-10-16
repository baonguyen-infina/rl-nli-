import pytest
from train.gta import guess_content_mask, use_supervised_loss


def test_mask_guess_content_but_keep_format_and_reasoning():
    text='<guess>neutral</guess><think>x</think><answer>entailment</answer>'
    spans=[(i,i+1) for i in range(len(text))]
    mask=guess_content_mask(text,spans)
    assert mask[:7] == [1]*7
    assert mask[7:14] == [0]*7
    assert mask[14:] == [1]*(len(text)-14)


def test_partial_guess_is_masked_to_end():
    text='<guess>neut'
    assert guess_content_mask(text,[(0,7),(7,11)]) == [1,0]


def test_absent_guess_does_not_hide_answer_gradient():
    assert guess_content_mask('<answer>neutral</answer>',[(0,8),(8,15),(15,24)]) == [1,1,1]


def test_conflicting_gradients_keep_rl_only():
    assert not use_supervised_loss(-1.0,2.0,3.0,cos_constraint=True,flat_fallback=True)


def test_aligned_gradients_combine():
    assert use_supervised_loss(1.0,2.0,3.0,cos_constraint=True,flat_fallback=False)


def test_flat_rl_still_learns_from_gold_without_teacher():
    assert use_supervised_loss(0.0,0.0,2.0,cos_constraint=True,flat_fallback=True)
    assert not use_supervised_loss(0.0,0.0,2.0,cos_constraint=True,flat_fallback=False)


def test_cosine_ablation_combines_conflicting_gradients():
    assert use_supervised_loss(-1.0,2.0,3.0,cos_constraint=False,flat_fallback=False)
