"""Boundaries for native RL rewards and student-prefix answer supervision.

The fake tokenizer supplies only explicit token boundaries. Model loading and
TRL integration belong to the remote smoke test; these tests never load weights.
"""

import math

import pytest

from train.native_training import (
    answer_prefix,
    build_answer_ce_rows,
    redistribute_group_rewards,
    token_content_mask,
    weighted_answer_ce,
)


class BoundaryTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    pieces = {
        "<answer>": [33],
        "<think>student reason</think><answer>": [31, 32, 33],
        "neutral</answer>": [41, 42],
        "entailment</answer>": [43, 44, 42],
        "contradiction</answer>": [45, 46, 47, 42],
    }

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, enable_thinking
    ):
        assert messages == [{"role": "user", "content": "premise and hypothesis"}]
        assert tokenize and add_generation_prompt and enable_thinking is False
        return [11, 12]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(self.pieces[text])


PROMPT = [{"role": "user", "content": "premise and hypothesis"}]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", "<answer>"),
        ("<answer>neutral</answer>", "<answer>"),
        (
            "<think>student reason</think>",
            "<think>student reason</think><answer>",
        ),
        (
            "<think>student reason</think><answer>contradiction</answer>"
            "<answer>entailment</answer>",
            "<think>student reason</think><answer>",
        ),
    ],
)
def test_answer_prefix_discards_sampled_answer_and_keeps_one_marker(raw, expected):
    assert answer_prefix(raw) == expected


def test_gold_ce_labels_only_answer_suffix_and_eos_after_student_reasoning():
    rows = build_answer_ce_rows(
        [PROMPT],
        ["<think>student reason</think><answer>contradiction</answer>"],
        ["neutral"],
        BoundaryTokenizer(),
    )

    assert len(rows) == 1
    assert rows[0]["input_ids"] == [11, 12, 31, 32, 33, 41, 42, 99]
    assert rows[0]["labels"] == [-100, -100, -100, -100, -100, 41, 42, 99]
    assert rows[0]["attention_mask"] == [1, 1, 1, 1, 1, 1, 1, 1]
    assert rows[0]["weight"] == 1.0


def test_gold_ce_retains_complete_multitoken_canonical_answer():
    rows = build_answer_ce_rows(
        [PROMPT], ["<answer>neutral</answer>"], ["contradiction"], BoundaryTokenizer()
    )

    assert rows[0]["input_ids"] == [11, 12, 33, 45, 46, 47, 42, 99]
    assert rows[0]["labels"] == [-100, -100, -100, 45, 46, 47, 42, 99]


def test_gold_ce_inactive_rows_have_no_supervised_tokens_or_weight():
    rows = build_answer_ce_rows(
        [PROMPT, PROMPT],
        ["<answer>neutral</answer>", "<answer>neutral</answer>"],
        ["neutral", "entailment"],
        BoundaryTokenizer(),
        active=[True, False],
    )

    assert rows[0]["labels"] == [-100, -100, -100, 41, 42, 99]
    assert rows[0]["weight"] == 1.0
    assert rows[1]["labels"] == [-100] * len(rows[1]["input_ids"])
    assert rows[1]["weight"] == 0.0


def test_gold_ce_rejects_misaligned_inputs_instead_of_silently_dropping_rows():
    with pytest.raises(ValueError):
        build_answer_ce_rows(
            [PROMPT, PROMPT], ["<answer>neutral</answer>"], ["neutral"], BoundaryTokenizer()
        )


def test_reasoning_loss_cap_preserves_delimiters_and_sampled_answer():
    ending = "</think><answer>neutral</answer>"
    raw = "<think>" + "x" * 66 + ending
    offsets = [(i, i + 1) for i in range(len(raw))] + [(0, 0)]

    mask = token_content_mask(raw, offsets, max_reasoning_tokens=64)

    # Seven opening-tag tokens, 64 reasoning tokens, two excluded reasoning
    # tokens, and all closing/answer tokens. This masks loss, not generation.
    assert mask == [1.0] * 71 + [0.0, 0.0] + [1.0] * len(ending) + [0.0]


def test_reasoning_cap_counts_tokens_instead_of_characters():
    raw = "<think>" + "x" * 66 + "</think><answer>neutral</answer>"
    offsets = [(0, 7), (7, 73), (73, len(raw)), (0, 0)]

    assert token_content_mask(raw, offsets, max_reasoning_tokens=64) == [
        1.0, 1.0, 1.0, 0.0
    ]


def test_unclosed_reasoning_still_obeys_loss_cap():
    raw = "<think>" + "x" * 66
    offsets = [(i, i + 1) for i in range(len(raw))]

    assert token_content_mask(raw, offsets, max_reasoning_tokens=64) == (
        [1.0] * 71 + [0.0, 0.0]
    )


def test_inactive_completion_masks_every_token_for_rl_and_kl():
    raw = "<think>x</think><answer>neutral</answer>"
    offsets = [(i, i + 1) for i in range(len(raw))] + [(0, 0)]

    assert token_content_mask(raw, offsets, active=False) == [0.0] * len(offsets)


def test_answer_only_completion_keeps_answer_tokens_and_excludes_padding():
    raw = "<answer>neutral</answer>"
    offsets = [(0, 8), (8, 15), (15, 24), (0, 0)]

    assert token_content_mask(raw, offsets) == [1.0, 1.0, 1.0, 0.0]


def test_group_reward_uses_explicit_group_and_joint_rollout_keys():
    rewards = redistribute_group_rewards(
        [1.0, 2.0, 3.0, 4.0, 9.0, 8.0],
        ["A", "B", "A", "B", "pad", "A"],
        [0, 1, 1, 0, 0, 0],
        {("A", 0): -1.0, ("A", 1): -2.0, ("B", 0): -4.0, ("B", 1): -3.0},
        active=[True, True, True, True, False, True],
        coefficient=0.2,
    )

    assert rewards == pytest.approx([0.8, 1.4, 2.6, 3.2, 0.0, 7.8])


def test_zero_allocation_coefficient_retains_ordinary_reward_and_zero_padding():
    rewards = redistribute_group_rewards(
        [1.0, 0.2, 1.2],
        ["A", "A", "pad"],
        [0, 1, 0],
        {("A", 0): -10.0, ("A", 1): -20.0},
        active=[True, True, False],
        coefficient=0.0,
    )

    assert rewards == pytest.approx([1.0, 0.2, 0.0])


def test_missing_active_joint_rollout_score_cannot_silently_borrow_another_draw():
    with pytest.raises((KeyError, ValueError)):
        redistribute_group_rewards(
            [1.0, 1.0], ["A", "A"], [0, 1], {("A", 0): -1.0}
        )


def test_group_reward_rejects_misaligned_metadata():
    with pytest.raises(ValueError):
        redistribute_group_rewards(
            [1.0, 1.0], ["A"], [0, 1], {("A", 0): -1.0, ("A", 1): -2.0}
        )


@pytest.mark.parametrize("all_labels_ignored", [True, False])
def test_answer_ce_empty_active_denominator_has_connected_zero_gradient(all_labels_ignored):
    torch = pytest.importorskip("torch")
    logits = torch.zeros((2, 2, 3), requires_grad=True)
    targets = (
        torch.full((2, 2), -100, dtype=torch.long)
        if all_labels_ignored
        else torch.tensor([[0, 1], [2, 0]])
    )
    row_weights = torch.ones(2) if all_labels_ignored else torch.zeros(2)

    loss = weighted_answer_ce(logits, targets, row_weights)

    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_answer_ce_weights_valid_tokens_and_excludes_inactive_and_ignored_targets():
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[20.0, -20.0], [20.0, -20.0]],
            [[0.0, math.log(3.0)], [0.0, math.log(3.0)]],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([[0, -100], [1, 1], [0, 0]])
    row_weights = torch.tensor([1.0, 0.0, 0.5])

    loss = weighted_answer_ce(logits, targets, row_weights)

    # The first active target has loss log(2). The two half-weight targets each
    # have loss log(4), giving a weighted valid-token denominator of two.
    assert loss.item() == pytest.approx((math.log(2.0) + math.log(4.0)) / 2.0)
    loss.backward()
    expected_gradient = torch.tensor(
        [
            [[-0.25, 0.25], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[-0.1875, 0.1875], [-0.1875, 0.1875]],
        ]
    )
    assert torch.allclose(logits.grad, expected_gradient)
