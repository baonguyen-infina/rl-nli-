import pytest
from formats.prompts import (build_prompt, build_messages, EXAMPLE_OUTPUTS,
                             EXAMPLE_PREMISE, EXAMPLE_HYPOTHESIS)
from formats.parser import parse
from formats.align import struct_ok

CASES = [("label", False), ("cot", False), ("cot", True), ("align", False), ("align", True)]


@pytest.mark.parametrize("fmt,guess", CASES)
def test_examples_parse_under_own_format(fmt, guess):
    p = parse(EXAMPLE_OUTPUTS[(fmt, guess)], fmt, guess)
    assert p.format_ok and p.label == "entailment"


@pytest.mark.parametrize("guess", [False, True])
def test_align_example_is_structurally_valid(guess):
    p = parse(EXAMPLE_OUTPUTS[("align", guess)], "align", guess)
    assert struct_ok(p.align_lines, EXAMPLE_PREMISE, EXAMPLE_HYPOTHESIS, p.label)


@pytest.mark.parametrize("fmt,guess", CASES)
def test_prompt_contains_inputs_and_no_uid(fmt, guess):
    pr = build_prompt("Premise XYZ", "Hypothesis ABC", fmt, guess)
    assert "Premise XYZ" in pr and "Hypothesis ABC" in pr
    assert "uit_Adver" not in pr
    assert ("<guess>" in pr) == guess


def test_prompt_mentions_format_tags():
    assert "<think>" in build_prompt("a", "b", "cot")
    assert "<align>" in build_prompt("a", "b", "align")
    assert "<think>" not in build_prompt("a", "b", "label")


def test_messages_shape():
    m = build_messages("a", "b", "label")
    assert len(m) == 1 and m[0]["role"] == "user"


def test_label_with_guess_rejected():
    with pytest.raises(ValueError):
        build_prompt("a", "b", "label", guess=True)


@pytest.mark.parametrize('guess',[False,True])
def test_balanced_schema_examples_are_grounded_and_cover_all_labels(guess):
    from formats.prompts import balanced_alignment_examples
    examples=balanced_alignment_examples(guess)
    assert {e['label'] for e in examples}=={'entailment','neutral','contradiction'}
    for e in examples:
        p=parse(e['output'],'align',guess)
        assert p.format_ok and p.label==e['label']
        assert struct_ok(p.align_lines,e['premise'],e['hypothesis'],p.label)
    text=build_prompt('p','h','align',guess,schema_examples='balanced')
    assert '∅ | missing' in text and 'uit_Adver' not in text
