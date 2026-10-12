import pytest
from teacher.generate import (
    build_teacher_prompt,
    validate_candidate,
    synthesize_valid_rationale,
    generate_rationales,
)

PREMISE = "Hà Nội là thủ đô của Việt Nam và có hơn tám triệu dân."
HYPOTHESIS = "Hà Nội là thủ đô của Việt Nam."


def test_build_teacher_prompt():
    prompt = build_teacher_prompt(PREMISE, HYPOTHESIS, "entailment", "align")
    assert "<align>" in prompt
    assert "entailment" in prompt
    assert PREMISE in prompt
    assert HYPOTHESIS in prompt

    cot_prompt = build_teacher_prompt(PREMISE, HYPOTHESIS, "entailment", "cot")
    assert "<think>" in cot_prompt
    assert "entailment" in cot_prompt


def test_validate_candidate_align():
    valid_raw = (
        '<align>\n'
        'H1: "Hà Nội là thủ đô của Việt Nam." <-> "Hà Nội là thủ đô của Việt Nam" | match\n'
        '</align><answer>entailment</answer>'
    )
    assert validate_candidate(valid_raw, "align", PREMISE, HYPOTHESIS, "entailment") is True

    # Wrong label
    assert validate_candidate(valid_raw, "align", PREMISE, HYPOTHESIS, "contradiction") is False

    # Ungrounded segment in premise
    bad_p = (
        '<align>\n'
        'H1: "Hà Nội là thủ đô của Việt Nam." <-> "Sài Gòn là thành phố lớn" | match\n'
        '</align><answer>entailment</answer>'
    )
    assert validate_candidate(bad_p, "align", PREMISE, HYPOTHESIS, "entailment") is False


def test_synthesize_valid_rationale():
    for gold in ("entailment", "neutral", "contradiction"):
        align = synthesize_valid_rationale(PREMISE, HYPOTHESIS, gold, "align")
        assert align is not None
        assert validate_candidate(align, "align", PREMISE, HYPOTHESIS, gold) is True

        cot = synthesize_valid_rationale(PREMISE, HYPOTHESIS, gold, "cot")
        assert cot is not None
        assert validate_candidate(cot, "cot", PREMISE, HYPOTHESIS, gold) is True


def test_generate_rationales_heuristic():
    rows = [
        {"id": "s1", "premise": PREMISE, "hypothesis": HYPOTHESIS, "label": "entailment"},
        {"id": "s2", "premise": PREMISE, "hypothesis": "Đà Nẵng có biển đẹp.", "label": "neutral"},
    ]
    records, stats = generate_rationales(rows, fmt="align", backend="heuristic")
    assert len(records) == 2
    assert stats["valid"] == 2
    assert stats["retention_rate"] == 1.0
