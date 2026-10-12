"""Test reward trên completion dạng chuỗi và dạng message list của TRL.
"""
import pytest
from rewards.core import total_reward, r_acc, r_fmt, r_struct
from rewards.trl_rewards import make_reward_funcs, completion_text, REWARD_WEIGHTS
from formats.parser import parse

PREM = "Hà Nội là thủ đô của Việt Nam và có hơn tám triệu dân."
HYP = "Hà Nội là thủ đô của Việt Nam."
ALIGN_GOOD = ('<align>\nH1: "Hà Nội là thủ đô của Việt Nam" <-> "Hà Nội là thủ đô của Việt Nam" | match\n'
              '</align><answer>entailment</answer>')


def test_label_rewards():
    assert total_reward("<answer>neutral</answer>", "label", False, "neutral", PREM, HYP) == 1.2
    assert total_reward("<answer>neutral</answer>", "label", False, "entailment", PREM, HYP) == 0.2
    assert total_reward("blah", "label", False, "neutral", PREM, HYP) == 0.0


def test_align_struct_bonus():
    assert total_reward(ALIGN_GOOD, "align", False, "entailment", PREM, HYP) == pytest.approx(1.5)
    # wrong gold: acc 0, struct bonus not granted
    assert total_reward(ALIGN_GOOD, "align", False, "neutral", PREM, HYP) == pytest.approx(0.2)


def test_struct_zero_for_non_align_and_bad_format():
    p = parse("<answer>neutral</answer>", "label")
    assert r_struct(p, "label", PREM, HYP) == 0.0
    assert r_struct(parse("x", "align"), "align", PREM, HYP) == 0.0
    assert r_acc(parse("x", "label"), "neutral") == 0.0
    assert r_fmt(p) == 1.0


def test_completion_text_both_shapes():
    assert completion_text("abc") == "abc"
    assert completion_text([{"role": "assistant", "content": "abc"}]) == "abc"
    with pytest.raises(TypeError):
        completion_text(123)


def test_trl_funcs_string_and_message_completions():
    f_acc, f_fmt, f_struct = make_reward_funcs("align")
    assert (f_acc.__name__, f_fmt.__name__, f_struct.__name__) == ("r_acc", "r_fmt", "r_acc_struct")
    assert REWARD_WEIGHTS == [1.0, 0.2, 0.3]
    kw = dict(gold=["entailment", "neutral"], premise=[PREM, PREM], hypothesis=[HYP, HYP])
    comps = [ALIGN_GOOD, [{"role": "assistant", "content": ALIGN_GOOD}]]
    assert f_acc(["p", "p"], comps, **kw) == [1.0, 0.0]
    assert f_fmt(["p", "p"], comps, **kw) == [1.0, 1.0]
    assert f_struct(["p", "p"], comps, **kw) == [1.0, 0.0]
