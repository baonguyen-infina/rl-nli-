"""CPU tests for a bounded, correctness-gated native allocation policy bonus."""

import math

import pytest


def reward_helper():
    from train.native_reward import bounded_correct_allocation_rewards
    return bounded_correct_allocation_rewards


def test_bonus_uses_exact_group_rollout_probability_and_sampled_correctness():
    reward = reward_helper()
    result = reward(
        [1.2, 0.2, 1.0, 0.5, 8.0],
        ["A", "A", "A", "B", "pad"], [0, 0, 1, 0, 0],
        {("A", 0): math.log(0.25), ("A", 1): math.log(0.75), ("B", 0): 0.0},
        [1, 0, 1, 1, 1],
        {("A", 0): 1, ("A", 1): 1, ("B", 0): 0},
        active=[1, 1, 1, 1, 0], coefficient=0.2,
    )
    assert result == pytest.approx([1.25, 0.2, 1.15, 0.5, 0.0])


def test_wrong_answer_gets_no_bonus_even_for_perfect_native_allocation():
    reward = reward_helper()
    result = reward([0.2, 1.2], [0, 0], [0, 0], {(0, 0): 0.0},
                    [0, 1], {(0, 0): 1}, coefficient=0.2)
    assert result == pytest.approx([0.2, 1.4])


def test_uninformative_groups_keep_base_without_logq_lookup():
    reward = reward_helper()
    # Empty/singleton/same-label groups have no assignment discrimination.
    assert reward([1.2, 0.2], [0, 1], [0, 0], {}, [1, 0],
                  {(0, 0): False, (1, 0): False}) == [1.2, 0.2]


def test_padding_has_zero_reward_and_does_not_require_any_group_key():
    reward = reward_helper()
    assert reward([99.0, -7.0], ["missing", "missing"], [9, 10], {},
                  [1, 0], {}, active=[False, False]) == [0.0, 0.0]


def test_zero_coefficient_is_ordinary_reward_with_zero_padding():
    reward = reward_helper()
    assert reward([1.2, 0.2, 9.0], [0, 0, "pad"], [0, 1, 0],
                  {(0, 0): -0.1, (0, 1): -3.0}, [1, 0, 1],
                  {(0, 0): 1, (0, 1): 1}, active=[1, 1, 0],
                  coefficient=0.0) == [1.2, 0.2, 0.0]


@pytest.mark.parametrize("logq", [0.0, -0.1, -10.0, -100.0, -1e6])
def test_bonus_is_nonnegative_and_at_most_configured_coefficient(logq):
    reward = reward_helper()
    result = reward([0.0], [0], [0], {(0, 0): logq}, [1], {(0, 0): 1}, coefficient=0.3)
    assert 0.0 <= result[0] <= 0.3
    assert result[0] == pytest.approx(0.3 * math.exp(logq))


def test_finite_extreme_negative_logq_underflows_to_zero_without_error():
    reward = reward_helper()
    assert reward([1.2], [0], [0], {(0, 0): -1e6}, [1], {(0, 0): 1}) == [1.2]


@pytest.mark.parametrize("roundoff", [1e-12, 1e-7, 1e-6])
def test_tiny_positive_logq_roundoff_is_clamped_to_zero(roundoff):
    reward = reward_helper()
    assert reward([0.0], [0], [0], {(0, 0): roundoff}, [1], {(0, 0): 1}) == pytest.approx([0.2])


@pytest.mark.parametrize("logq", [1.1e-6, 0.1, float("inf"), float("-inf"), float("nan")])
def test_invalid_informative_logq_is_rejected_even_for_wrong_answers(logq):
    reward = reward_helper()
    with pytest.raises(ValueError):
        reward([0.2], [0], [0], {(0, 0): logq}, [0], {(0, 0): 1})


@pytest.mark.parametrize("which", ["group", "rollout", "correct", "active"])
def test_misaligned_vectors_are_rejected_instead_of_silently_zip_truncated(which):
    reward = reward_helper()
    groups, rollouts, correct, active = [0, 0], [0, 1], [1, 1], [1, 1]
    if which == "group":
        groups.pop()
    elif which == "rollout":
        rollouts.pop()
    elif which == "correct":
        correct.pop()
    else:
        active.pop()
    with pytest.raises(ValueError):
        reward([1.0, 1.0], groups, rollouts, {(0, 0): 0.0, (0, 1): 0.0},
               correct, {(0, 0): 1, (0, 1): 1}, active=active)


@pytest.mark.parametrize("which,value", [
    ("correct", -1), ("correct", 0.5), ("correct", float("nan")),
    ("active", 2), ("active", 0.5), ("active", float("nan")),
    ("informative", -1), ("informative", 0.5), ("informative", float("nan")),
])
def test_gates_must_be_binary(which, value):
    reward = reward_helper()
    correct, active, informative = [1], [1], {(0, 0): 1}
    if which == "correct":
        correct[0] = value
    elif which == "active":
        active[0] = value
    else:
        informative[(0, 0)] = value
    with pytest.raises(ValueError):
        reward([1.0], [0], [0], {(0, 0): 0.0}, correct, informative, active=active)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_base_is_rejected_including_padding(value):
    reward = reward_helper()
    with pytest.raises(ValueError):
        reward([value], ["pad"], [0], {}, [0], {}, active=[0])


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), float("-inf")])
def test_bonus_coefficient_must_be_finite_and_nonnegative(value):
    reward = reward_helper()
    with pytest.raises(ValueError):
        reward([1.0], [0], [0], {(0, 0): 0.0}, [1], {(0, 0): 1}, coefficient=value)


def test_missing_active_informativeness_fails_clearly():
    reward = reward_helper()
    with pytest.raises(ValueError, match="(?i)informative|missing|key"):
        reward([1.0], [0], [0], {(0, 0): 0.0}, [1], {})


def test_missing_active_informative_logq_fails_even_if_answer_is_wrong():
    reward = reward_helper()
    with pytest.raises(ValueError, match="(?i)logq|missing|key"):
        reward([0.2], [0], [0], {}, [0], {(0, 0): 1})


def test_empty_reward_batch_returns_empty_without_group_lookups():
    reward = reward_helper()
    assert reward([], [], [], {}, [], {}) == []


def test_reward_inputs_are_not_mutated():
    reward = reward_helper()
    base, correct, active = [1.0], [1], [True]
    scores, informative = {(0, 0): 1e-7}, {(0, 0): True}
    assert reward(base, [0], [0], scores, correct, informative, active=active) == pytest.approx([1.2])
    assert base == [1.0] and correct == [1] and active == [True]
    assert scores == {(0, 0): 1e-7} and informative == {(0, 0): True}


def test_native_policy_channel_control_is_explicit():
    from train.native_reward import policy_loss_coefficient
    assert policy_loss_coefficient('allocation_only')==0.
    assert policy_loss_coefficient('ce')==0.
    assert policy_loss_coefficient('gated_joint')==1.
    assert policy_loss_coefficient('joint')==1.


def test_bounded_swap_uses_probability_difference_and_correctness_gate():
    from train.native_reward import bounded_correct_swap_rewards
    actual=bounded_correct_swap_rewards([1.2,.2,1.],['A']*3,[0,0,1],
        {('A',0):math.log(.8),('A',1):math.log(.1)},
        {('A',0):math.log(.3),('A',1):math.log(.9)},
        [1,0,1],{('A',0):1,('A',1):1},coefficient=.2)
    assert actual==pytest.approx([1.3,.2,.84])


@pytest.mark.parametrize('own,swapped',[(0.,-1e6),(-1e6,0.),(-.1,-10.),(-10.,-.1),(-1e6,-1e6)])
def test_bounded_swap_preserves_raw_correctness_dominance(own,swapped):
    from train.native_reward import bounded_correct_swap_rewards
    actual=bounded_correct_swap_rewards([1.,.2],[0,0],[0,0],{(0,0):own},
        {(0,0):swapped},[1,0],{(0,0):1})
    assert .8-1e-12<=actual[0]<=1.2+1e-12
    assert actual[1]==.2 and actual[0]>actual[1]


def test_identical_swap_scores_cancel_and_padding_needs_no_lookup():
    from train.native_reward import bounded_correct_swap_rewards
    assert bounded_correct_swap_rewards([1.2,.2,99.],[0,0,'pad'],[0,0,0],
        {(0,0):-.5},{(0,0):-.5},[1,0,1],{(0,0):1},active=[1,1,0])==pytest.approx([1.2,.2,0.])
    assert bounded_correct_swap_rewards([1.2],[0],[0],{}, {},[1],{(0,0):0})==[1.2]


@pytest.mark.parametrize('swapped',[.1,float('nan'),float('inf'),float('-inf')])
def test_bounded_swap_rejects_invalid_counterfactual_probability(swapped):
    from train.native_reward import bounded_correct_swap_rewards
    with pytest.raises(ValueError):
        bounded_correct_swap_rewards([1.],[0],[0],{(0,0):-.5},
            {(0,0):swapped},[1],{(0,0):1})


def test_bounded_swap_reduces_to_existing_gated_reward_when_counterfactual_has_zero_probability():
    from train.native_reward import bounded_correct_swap_rewards,policy_loss_coefficient
    args=([1.2,.2],[0,0],[0,0],{(0,0):math.log(.7)})
    actual=bounded_correct_swap_rewards(*args,{(0,0):-1e6},[1,0],{(0,0):1})
    expected=reward_helper()(*args,[1,0],{(0,0):1})
    assert actual==pytest.approx(expected)
    assert policy_loss_coefficient('gated_trace_swap')==1.
