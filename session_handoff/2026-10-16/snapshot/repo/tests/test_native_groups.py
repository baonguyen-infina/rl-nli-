"""CPU tests for bounded native blocks and conditional label allocation."""

from collections import Counter
from copy import deepcopy
import math
import random

import pytest

from train.native_groups import (
    build_native_blocks,
    build_random_blocks,
    log_allocation,
    pad_blocks,
    tensor_log_allocation,
    build_scheduled_random_blocks,
)


def test_random_control_preserves_actual_exposure_at_every_update_prefix():
    native = [[row(f'{g}-{label}', f'Premise {g}', label)
               for label in ('entailment', 'neutral', 'contradiction')] for g in range(11)]
    order = [7, 1, 5, 3, 10, 0, 9, 4, 8, 2, 6]
    control = build_scheduled_random_blocks(native, order, groups_per_batch=4, seed=0)
    for start in (0, 4):
        cohort = order[start:start+4]
        assert Counter(id(r) for g in cohort for r in control[g]) == Counter(id(r) for g in cohort for r in native[g])
        assert all(len({r['premise'] for r in control[g]}) == len(control[g]) for g in cohort)
    for g in order[8:]:
        assert control[g] == native[g]
    assert [[r['label'] for r in b] for b in control] == [[r['label'] for r in b] for b in native]
    assert Counter(id(r) for b in control for r in b) == Counter(id(r) for b in native for r in b)


def row(name, premise, label):
    return {
        "id": name,
        "dataset": "vianli",
        "premise": premise,
        "hypothesis": "Một giả thuyết.",
        "label": label,
        "topic": "01",
        "premise_id": "analysis-only",
        "intended_label": "analysis-only",
    }


def test_native_blocks_normalize_nfc_whitespace_without_casefolding():
    rows = [
        row("a", "Cà phê", "entailment"),
        row("b", "Ca\u0300\t  phe\u0302", "neutral"),
        row("c", "  Cà\nphê  ", "contradiction"),
        row("d", "CÀ PHÊ", "neutral"),
        row("e", "Cà phê!", "neutral"),
    ]
    blocks = build_native_blocks(rows, seed=0)
    assert {frozenset(r["id"] for r in block) for block in blocks} == {
        frozenset({"a", "b", "c"}), frozenset({"d"}), frozenset({"e"}),
    }


def test_native_blocks_preserve_every_original_object_without_mutation():
    rows = [row(str(i), "P", "neutral") for i in range(7)]
    before = deepcopy(rows)
    blocks = build_native_blocks(rows, seed=7)
    assert sorted(map(len, blocks)) == [1, 3, 3]
    assert Counter(id(r) for block in blocks for r in block) == Counter(map(id, rows))
    assert rows == before


def test_native_partition_is_label_blind_and_does_not_reset_global_rng():
    rows = [row(str(i), "P", "neutral") for i in range(8)]
    relabeled = deepcopy(rows)
    for i, item in enumerate(relabeled):
        item["label"] = ("entailment", "neutral", "contradiction")[i % 3]
    rng_state = random.getstate()
    first = build_native_blocks(rows, seed=31)
    second = build_native_blocks(relabeled, seed=31)
    assert [[r["id"] for r in b] for b in first] == [[r["id"] for r in b] for b in second]
    assert first == build_native_blocks(rows, seed=31)
    assert random.getstate() == rng_state


@pytest.mark.parametrize("max_size", [0, 4])
def test_native_blocks_reject_out_of_bound_block_size(max_size):
    with pytest.raises(ValueError):
        build_native_blocks([row("a", "P", "neutral")], max_size=max_size)


def test_empty_native_input_has_no_blocks():
    assert build_native_blocks([]) == []


def test_random_blocks_match_slots_exposure_and_distinct_premises():
    native = [
        [row("p-e", "P", "entailment"), row("p-n", "P", "neutral"), row("p-c", "P", "contradiction")],
        [row("q-e", "Q", "entailment"), row("q-n", "Q", "neutral"), row("q-c", "Q", "contradiction")],
        [row("r-e", "R", "entailment"), row("r-n", "R", "neutral"), row("r-c", "R", "contradiction")],
        [row("s-e", "S", "entailment")],
    ]
    before = deepcopy(native)
    random_blocks = build_random_blocks(native, seed=8)
    assert [len(b) for b in random_blocks] == [3, 3, 3, 1]
    assert [[r["label"] for r in b] for b in random_blocks] == [
        ["entailment", "neutral", "contradiction"],
        ["entailment", "neutral", "contradiction"],
        ["entailment", "neutral", "contradiction"],
        ["entailment"],
    ]
    assert Counter(id(r) for b in random_blocks for r in b) == Counter(id(r) for b in native for r in b)
    assert all(len({r["premise"] for r in b}) == len(b) for b in random_blocks)
    assert random_blocks == build_random_blocks(native, seed=8)
    assert native == before


def test_random_blocks_detect_infeasibility_after_premise_normalization():
    native = [[row("a", "Cà phê", "entailment"), row("b", "Ca\u0300  phe\u0302", "neutral")]]
    with pytest.raises(ValueError, match="(?i)infeasible|different.premise"):
        build_random_blocks(native)


def test_random_blocks_do_not_silently_keep_an_infeasible_label_tail():
    native = [
        [row("p-e", "P", "entailment"), row("p-c", "P", "contradiction")],
        [row("q-n", "Q", "neutral"), row("q-n2", "Q", "neutral")],
    ]
    with pytest.raises(ValueError, match="(?i)infeasible|different.premise"):
        build_random_blocks(native)


def test_padding_has_zero_weight_and_preserves_active_rows():
    native = [
        [row("a", "P", "entailment"), row("b", "P", "neutral")],
        [row("c", "Q", "contradiction")],
    ]
    before = deepcopy(native)
    padded = pad_blocks(native)
    assert [(r["group_key"], r["slot"], r["slot_weight"]) for r in padded] == [
        (0, 0, 1), (0, 1, 1), (0, 2, 0),
        (1, 0, 1), (1, 1, 0), (1, 2, 0),
    ]
    assert [r["id"] for r in padded if r["slot_weight"]] == ["a", "b", "c"]
    assert all(r["premise"] and r["hypothesis"] for r in padded)
    assert native == before
    assert pad_blocks([]) == []


def test_padding_never_injects_metadata_into_an_existing_prompt():
    original = row("a", "P", "neutral")
    original["prompt"] = [{"role": "user", "content": "P\nMột giả thuyết."}]
    before = deepcopy(original)
    padded = pad_blocks([[original]])
    assert all(r["prompt"] == before["prompt"] for r in padded)
    assert original == before


def test_padding_rejects_an_oversized_or_empty_block():
    with pytest.raises(ValueError):
        pad_blocks([[row(str(i), "P", "neutral") for i in range(4)]])
    with pytest.raises(ValueError):
        pad_blocks([[]])


def test_pair_allocation_is_exact_negative_softplus_of_gold_swap_margin():
    scores = [[4.0, 0.0, -10.0], [1.0, 0.0, -10.0]]
    assert log_allocation(scores, [0, 1], [1, 1]) == pytest.approx(-math.log1p(math.exp(-3.0)))
    assert log_allocation(scores, ["entailment", "neutral"], [1, 1]) == pytest.approx(-math.log1p(math.exp(-3.0)))


def test_repeated_gold_labels_count_unique_assignments_only():
    scores = [[0.0, 0.0, 0.0]] * 3
    assert log_allocation(scores, [0, 1, 1], [1, 1, 1]) == pytest.approx(-math.log(3.0))
    assert log_allocation(scores, [0, 1, 2], [1, 1, 1]) == pytest.approx(-math.log(6.0))


@pytest.mark.parametrize("gold,mask", [([1, 1, 1], [1, 1, 1]), ([1, -1, -1], [1, 0, 0]), ([-1, -1, -1], [0, 0, 0])])
def test_no_contrast_returns_zero_and_ignores_inactive_gold(gold, mask):
    assert log_allocation([[4.0, 0.0, 2.0]] * 3, gold, mask) == 0.0


def test_allocation_is_invariant_to_shared_class_offsets():
    scores = [[4.0, 0.0, -10.0], [1.0, 0.0, -10.0]]
    bias = [10000.0, -20000.0, 30000.0]
    shifted = [[value + bias[c] for c, value in enumerate(slot)] for slot in scores]
    assert log_allocation(shifted, [0, 1], [1, 1]) == pytest.approx(log_allocation(scores, [0, 1], [1, 1]), abs=1e-9)


def test_allocation_remains_finite_for_extreme_and_padded_scores():
    scores = [[10000.0, -10000.0, 0.0], [-10000.0, 10000.0, 0.0], [float("nan")] * 3]
    assert log_allocation(scores, [1, 0, -1], [1, 1, 0]) == pytest.approx(-40000.0)


@pytest.mark.parametrize("scores,gold,mask", [
    ([[0.0, 0.0]], [0], [1]),
    ([[0.0, 0.0, 0.0]], [0, 1], [1]),
    ([[0.0, 0.0, 0.0]], [3], [1]),
    ([[0.0, 0.0, 0.0]], [0], [0.5]),
])
def test_allocation_rejects_invalid_active_inputs(scores, gold, mask):
    with pytest.raises(ValueError):
        log_allocation(scores, gold, mask)


def test_tensor_allocation_keeps_group_rollout_shape_and_zero_padding_gradient():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([
        [[[4.0, 0.0, -10.0], [1.0, 0.0, -10.0], [10000.0, -10000.0, 0.0]],
         [[0.0, 4.0, -10.0], [0.0, 1.0, -10.0], [-10000.0, 10000.0, 0.0]]],
        [[[1.0, 4.0, 0.0], [2.0, 3.0, 0.0], [10000.0, 2.0, 0.0]],
         [[3.0, 2.0, 0.0], [4.0, 1.0, 0.0], [1.0, 0.0, 10000.0]]],
    ], dtype=torch.float64, requires_grad=True)
    gold = torch.tensor([[0, 1, -1], [1, 1, -1]])
    weights = torch.tensor([[1, 1, 0], [1, 1, 0]])
    result = tensor_log_allocation(scores, gold, weights)
    assert result.shape == (2, 2)
    assert result[0].detach().tolist() == pytest.approx([-math.log1p(math.exp(-3.0)), -math.log1p(math.exp(3.0))])
    assert result[1].detach().tolist() == [0.0, 0.0]
    (-result.sum()).backward()
    assert scores.grad[:, :, 2, :].abs().sum().item() == 0.0
    assert scores.grad[1].abs().sum().item() == 0.0
    assert scores.grad[0, 0, 0, 0].item() < 0.0
    assert scores.grad[0, 0, 1, 0].item() > 0.0
    assert scores.grad[0, 0, :, 0].sum().item() == pytest.approx(0.0, abs=1e-12)


def test_tensor_allocation_ignores_nonfinite_padding_and_has_zero_empty_gradient():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([[[[0.0, 0.0, 0.0], [float("nan")] * 3, [float("nan")] * 3]]], requires_grad=True)
    result = tensor_log_allocation(scores, torch.tensor([[0, -1, -1]]), torch.tensor([[1, 0, 0]]))
    assert result.item() == 0.0
    result.sum().backward()
    assert torch.equal(scores.grad, torch.zeros_like(scores))


def test_tensor_allocation_rejects_wrong_shapes_and_nonbinary_weights():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        tensor_log_allocation(torch.zeros(2, 3, 3), torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 3))
    with pytest.raises(ValueError):
        tensor_log_allocation(torch.zeros(1, 2, 3, 3), torch.zeros(1, 3, dtype=torch.long), torch.tensor([[1.0, 0.5, 0.0]]))
