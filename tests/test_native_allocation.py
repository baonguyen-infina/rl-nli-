"""CPU regression tests for replaying the native allocation score gradient.

The replay may split siblings across arbitrary microbatches, but must preserve
the full mean loss value and score derivative at the cached score point.
"""

from itertools import permutations
import math

import pytest


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def helpers():
    from train.native_allocation import allocation_linearization, allocation_surrogate
    return allocation_linearization, allocation_surrogate


def replay(surrogate, live, cache):
    return surrogate(live, cache["scores"], cache["jacobian"],
                     cache["value_share"], cache["group_count"])


def mixed_example(torch):
    # Groups deliberately have interleaved rows and unequal active sizes.
    scores = torch.tensor([
        [0.2, -0.4, 1.1], [1.0, 0.2, -0.8], [2.0, -1.0, 0.0],
        [-0.9, 0.4, 0.7], [0.1, 1.7, -0.3], [0.2, 0.8, 1.9],
        [-0.4, 0.3, 0.2], [1.7, 0.5, -0.4], [0.6, 0.1, -0.6],
        [0.0, -0.1, 0.3], [0.4, 0.8, 0.2], [1.0, 0.0, -1.0],
    ], dtype=torch.float64)
    members = {"triple": [7, 3, 9], "repeat": [0, 5, 10],
               "pair": [1, 4, 8], "same": [2, 6, 11]}
    gold = ["neutral", "entailment", "contradiction", "neutral",
            "neutral", "neutral", "contradiction", "entailment",
            "ignored-padding", "contradiction", "contradiction", "ignored-padding"]
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]
    return scores, members, gold, weights


def enumeration_loss(scores, members, gold, weights, torch):
    """Independent exhaustive assignment oracle; does not use production helpers."""
    classes = ("entailment", "neutral", "contradiction")
    losses = []
    for indices in members.values():
        active = [i for i in indices if weights[i]]
        labels = tuple(classes.index(gold[i]) for i in active)
        assignments = sorted(set(permutations(labels)))
        if len(set(labels)) <= 1:
            losses.append(scores.sum() * 0.0)
            continue
        candidate = torch.stack([
            sum((scores[i, label] for i, label in zip(active, assignment)),
                scores.new_zeros(()))
            for assignment in assignments
        ])
        target = sum((scores[i, label] for i, label in zip(active, labels)),
                     scores.new_zeros(()))
        losses.append(torch.logsumexp(candidate, dim=0) - target)
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def test_pair_value_and_jacobian_match_hand_derived_softplus(torch):
    linearize, surrogate = helpers()
    scores = torch.tensor([[4.0, 0.0, -10.0], [1.0, 0.0, -10.0],
                           [99.0, -99.0, 5.0]], dtype=torch.float64)
    cache = linearize(scores, {"pair": [0, 1, 2]},
                      ["entailment", "neutral", "ignored"], [1, 1, 0])
    probability = 1.0 / (1.0 + math.exp(3.0))
    expected = torch.tensor([[-probability, probability, 0.0],
                             [probability, -probability, 0.0],
                             [0.0, 0.0, 0.0]], dtype=torch.float64)
    assert cache["value_share"].tolist() == pytest.approx([
        math.log1p(math.exp(-3.0)) / 2, math.log1p(math.exp(-3.0)) / 2, 0.0,
    ])
    assert torch.allclose(cache["jacobian"], expected, atol=1e-12, rtol=1e-12)
    live = scores.clone().requires_grad_()
    result = replay(surrogate, live, cache)
    assert result.item() == pytest.approx(math.log1p(math.exp(-3.0)))
    result.backward()
    assert torch.allclose(live.grad, expected, atol=1e-12, rtol=1e-12)


def test_arbitrary_shuffled_microbatches_preserve_mean_value_and_full_gradient(torch):
    linearize, surrogate = helpers()
    scores, members, gold, weights = mixed_example(torch)
    exact = scores.clone().requires_grad_()
    target = enumeration_loss(exact, members, gold, weights, torch)
    target.backward()
    cache = linearize(scores, members, gold, weights)
    live = scores.clone().requires_grad_()
    total = live.new_zeros(())
    # A microbatch can cut through every native sibling group.
    for indices in ([9, 1], [11, 7, 0, 4], [5], [8, 2, 10, 3, 6]):
        index = torch.tensor(indices)
        piece = surrogate(live[index], cache["scores"][index],
                          cache["jacobian"][index], cache["value_share"][index],
                          cache["group_count"])
        total = total + piece
    assert total.item() == pytest.approx(target.item(), abs=1e-12)
    total.backward()
    assert torch.allclose(live.grad, exact.grad, atol=1e-12, rtol=1e-12)


def test_score_gradient_matches_existing_differentiable_allocation_utility(torch):
    from train.native_groups import tensor_log_allocation
    linearize, surrogate = helpers()
    scores, members, gold, weights = mixed_example(torch)
    rows = list(members.values())
    label_index = {"entailment": 0, "neutral": 1, "contradiction": 2}
    exact = scores.clone().requires_grad_()
    grouped = exact[torch.tensor(rows)].unsqueeze(1)
    grouped_gold = torch.tensor([[label_index.get(gold[i], -1) for i in row] for row in rows])
    grouped_weights = torch.tensor([[weights[i] for i in row] for row in rows])
    direct = -tensor_log_allocation(grouped, grouped_gold, grouped_weights).mean()
    direct.backward()
    cache = linearize(scores, members, gold, weights)
    live = scores.clone().requires_grad_()
    result = replay(surrogate, live, cache)
    result.backward()
    assert result.item() == pytest.approx(direct.item(), abs=1e-12)
    assert torch.allclose(live.grad, exact.grad, atol=1e-12, rtol=1e-12)


def test_normalization_counts_no_contrast_groups_and_padding_has_zero_gradient(torch):
    linearize, surrogate = helpers()
    scores, members, gold, weights = mixed_example(torch)
    cache = linearize(scores, members, gold, weights)
    assert cache["group_count"].ndim == 0
    assert cache["group_count"].item() == 4
    assert torch.equal(cache["jacobian"][[2, 6, 11, 8]], torch.zeros(4, 3, dtype=torch.float64))
    assert cache["value_share"][[2, 6, 11, 8]].tolist() == [0.0] * 4
    live = scores.clone().requires_grad_()
    replay(surrogate, live, cache).backward()
    assert torch.equal(live.grad[[2, 6, 11, 8]], torch.zeros(4, 3, dtype=torch.float64))


def test_repeated_label_assignments_have_no_factorial_overcount(torch):
    linearize, surrogate = helpers()
    scores = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    cache = linearize(scores, {0: [0, 1, 2]},
                      ["entailment", "neutral", "neutral"], [1, 1, 1])
    assert replay(surrogate, scores, cache).item() == pytest.approx(math.log(3.0))


def test_row_and_class_bias_cancel_in_value_and_score_jacobian(torch):
    linearize, _ = helpers()
    scores, members, gold, weights = mixed_example(torch)
    base = linearize(scores, members, gold, weights)
    row_bias = torch.arange(len(gold), dtype=torch.float64).unsqueeze(-1) * 11.0
    class_bias = torch.tensor([1000.0, -2000.0, 3000.0], dtype=torch.float64)
    shifted = linearize(scores + row_bias + class_bias, members, gold, weights)
    assert torch.allclose(base["value_share"], shifted["value_share"], atol=1e-11, rtol=1e-11)
    assert torch.allclose(base["jacobian"], shifted["jacobian"], atol=1e-11, rtol=1e-11)
    assert torch.allclose(base["jacobian"].sum(-1), torch.zeros(len(gold), dtype=torch.float64), atol=1e-12)
    for indices in members.values():
        assert torch.allclose(base["jacobian"][indices].sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-12)


def test_cache_is_detached_independent_storage_and_works_under_outer_no_grad(torch):
    linearize, _ = helpers()
    scores, members, gold, weights = mixed_example(torch)
    source = scores.clone().requires_grad_()
    with torch.no_grad():
        cache = linearize(source, members, gold, weights)
        assert not torch.is_grad_enabled()
    assert source.grad is None
    for tensor in cache.values():
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
    saved = cache["scores"].clone()
    with torch.no_grad():
        source.add_(10.0)
    assert torch.equal(cache["scores"], saved)


def test_cached_jacobian_and_values_cannot_receive_gradient_from_replay(torch):
    linearize, surrogate = helpers()
    scores, members, gold, weights = mixed_example(torch)
    cache = linearize(scores, members, gold, weights)
    old = cache["scores"].clone().requires_grad_()
    jacobian = cache["jacobian"].clone().requires_grad_()
    shares = cache["value_share"].clone().requires_grad_()
    count = cache["group_count"].double().requires_grad_()
    live = scores.clone().requires_grad_()
    surrogate(live, old, jacobian, shares, count).backward()
    assert live.grad is not None and live.grad.abs().sum().item() > 0
    assert all(value.grad is None for value in (old, jacobian, shares, count))


def test_extreme_scores_and_nonfinite_padding_remain_finite(torch):
    linearize, surrogate = helpers()
    scores = torch.tensor([[10000.0, -10000.0, 0.0],
                           [-10000.0, 10000.0, 0.0],
                           [float("nan"), float("inf"), float("-inf")]], dtype=torch.float64)
    cache = linearize(scores, {0: [0, 1, 2]},
                      ["neutral", "entailment", "invalid-padding"], [1, 1, 0])
    live = scores.clone().requires_grad_()
    result = replay(surrogate, live, cache)
    assert result.item() == pytest.approx(40000.0)
    assert torch.isfinite(result).item()
    result.backward()
    assert torch.isfinite(live.grad).all().item()
    assert torch.equal(live.grad[2], torch.zeros(3, dtype=torch.float64))


def test_empty_layout_has_finite_graph_connected_zero(torch):
    linearize, surrogate = helpers()
    scores = torch.empty(0, 3, dtype=torch.float64, requires_grad=True)
    cache = linearize(scores, {}, [], [])
    assert cache["group_count"].item() == 0
    result = replay(surrogate, scores, cache)
    assert result.item() == 0.0
    result.backward()
    assert scores.grad.shape == (0, 3)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_half_precision_assignment_math_is_promoted_and_gradient_survives(torch, dtype_name):
    linearize, surrogate = helpers()
    scores = torch.zeros(3, 3, dtype=getattr(torch, dtype_name), requires_grad=True)
    cache = linearize(scores, {0: [0, 1, 2]},
                      ["entailment", "neutral", "contradiction"], [1, 1, 1])
    assert cache["jacobian"].dtype == torch.float32
    assert cache["value_share"].dtype == torch.float32
    result = replay(surrogate, scores, cache)
    assert result.item() == pytest.approx(math.log(6.0), abs=1e-6)
    result.backward()
    assert torch.isfinite(scores.grad).all().item()
    assert scores.grad.abs().sum().item() > 0


@pytest.mark.parametrize("members,gold,weights", [
    ({0: [0, 1]}, ["entailment", "neutral", "neutral"], [1, 1, 1]),
    ({0: [0, 1], 1: [1, 2]}, ["entailment", "neutral", "neutral"], [1, 1, 1]),
    ({0: [0, 1, 3]}, ["entailment", "neutral", "neutral"], [1, 1, 1]),
    ({0: []}, ["entailment", "neutral", "neutral"], [1, 1, 1]),
    ({0: [0, 1, 2]}, ["entailment", "invalid", "neutral"], [1, 1, 1]),
    ({0: [0, 1, 2]}, ["entailment", "neutral", "neutral"], [1, 0.5, 1]),
    ({0: [0, 1, 2]}, ["entailment", "neutral"], [1, 1, 1]),
])
def test_linearization_rejects_malformed_or_incomplete_group_layout(torch, members, gold, weights):
    linearize, _ = helpers()
    with pytest.raises(ValueError):
        linearize(torch.zeros(3, 3), members, gold, weights)


def test_linearization_rejects_nonfinite_active_scores_and_wrong_score_shape(torch):
    linearize, _ = helpers()
    gold, weights, members = ["entailment", "neutral"], [1, 1], {0: [0, 1]}
    for scores in (torch.zeros(2, 2), torch.zeros(2, 3, dtype=torch.long),
                   torch.tensor([[0.0, float("nan"), 0.0], [0.0, 0.0, 0.0]])):
        with pytest.raises(ValueError):
            linearize(scores, members, gold, weights)


@pytest.mark.parametrize("bad_field", ["old_scores", "jacobian", "value_share", "group_count"])
def test_surrogate_rejects_invalid_replay_shape_or_normalizer(torch, bad_field):
    linearize, surrogate = helpers()
    scores, members, gold, weights = mixed_example(torch)
    cache = linearize(scores, members, gold, weights)
    args = {"old_scores": cache["scores"], "jacobian": cache["jacobian"],
            "value_share": cache["value_share"], "group_count": cache["group_count"]}
    args[bad_field] = (torch.tensor(-1) if bad_field == "group_count" else args[bad_field][:-1])
    with pytest.raises(ValueError):
        surrogate(scores, **args)
