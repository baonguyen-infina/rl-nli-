"""Test parser cho cả 3 format và biến thể guess (output hợp lệ / lỗi).
"""
import pytest
from formats.parser import parse, Parsed

ALIGN_OK = ('<align>\nH1: "Hà Nội" <-> "Hà Nội" | match\n'
            'H2: "thủ đô" <-> ∅ | missing\n</align><answer>neutral</answer>')


def test_label_ok():
    p = parse("<answer>neutral</answer>", "label")
    assert p.format_ok and p.label == "neutral" and p.guess is None


def test_label_failures():
    for bad in ["neutral", "<answer>maybe</answer>", "<answer>neutral</answer> thêm",
                "<answer>neutral</answer><answer>neutral</answer>", "", "<answer>Neutral</answer>"]:
        p = parse(bad, "label")
        assert not p.format_ok and p.label is None, bad


def test_label_with_guess_is_error():
    with pytest.raises(ValueError):
        parse("<answer>neutral</answer>", "label", guess=True)


def test_cot_ok_and_multiline():
    p = parse("<think>Dòng 1\nDòng 2</think><answer>entailment</answer>", "cot")
    assert p.format_ok and p.label == "entailment" and p.reasoning == "Dòng 1\nDòng 2"


def test_cot_failures():
    for bad in ["<answer>entailment</answer>",
                "<think></think><answer>entailment</answer>",
                "<think>a</think><think>b</think><answer>entailment</answer>",
                "<think>a</think><answer>entailment</answer> xong",
                "<answer>entailment</answer><think>a</think>"]:
        assert not parse(bad, "cot").format_ok, bad


def test_guess_variants():
    ok = "<guess>neutral</guess><think>x</think><answer>contradiction</answer>"
    p = parse(ok, "cot", guess=True)
    assert p.format_ok and p.guess == "neutral" and p.label == "contradiction"
    # guess required when guess=True
    assert not parse("<think>x</think><answer>neutral</answer>", "cot", guess=True).format_ok
    # guess forbidden when guess=False
    assert not parse(ok, "cot", guess=False).format_ok
    # guess must be a valid label
    assert not parse("<guess>foo</guess><think>x</think><answer>neutral</answer>", "cot", guess=True).format_ok


def test_align_ok():
    p = parse(ALIGN_OK, "align")
    assert p.format_ok and p.label == "neutral"
    assert [ln.rel for ln in p.align_lines] == ["match", "missing"]


def test_align_with_guess():
    p = parse("<guess>neutral</guess>" + ALIGN_OK, "align", guess=True)
    assert p.format_ok and p.guess == "neutral"


def test_align_failures():
    # missing without ∅
    bad = '<align>\nH1: "a" <-> "b" | missing\n</align><answer>neutral</answer>'
    assert not parse(bad, "align").format_ok
    # no align block
    assert not parse("<answer>neutral</answer>", "align").format_ok
    # extra text before
    assert not parse("Kết quả: " + ALIGN_OK, "align").format_ok


def test_unknown_format():
    with pytest.raises(ValueError):
        parse("<answer>neutral</answer>", "xyz")
