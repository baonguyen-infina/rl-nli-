import pytest
from common.labels import LABELS, INTENDED_CODE, normalize_label
from common.text import normalize, match_key, syllables
from common.io import deep_merge, run_name, read_jsonl, write_jsonl, write_train_log


def test_labels():
    assert LABELS == ("entailment", "neutral", "contradiction")
    assert INTENDED_CODE == {"11": "entailment", "21": "neutral", "31": "contradiction"}


def test_normalize_label():
    assert normalize_label(" Entailment ") == "entailment"
    with pytest.raises(ValueError):
        normalize_label("maybe")


def test_text_normalization():
    assert normalize("  Hà   Nội\n") == "Hà Nội"
    # NFD input must equal NFC input after normalization
    assert match_key("Vie\u0302\u0323t") == match_key("Việt")
    assert match_key("HÀ Nội") == "hà nội"


def test_syllables_strip_punctuation():
    assert syllables("Hà Nội là thủ đô.") == ["hà", "nội", "là", "thủ", "đô"]
    assert syllables("   ") == []


def test_deep_merge():
    a = {"x": {"a": 1, "b": 2}, "y": 1}
    b = {"x": {"b": 3}, "z": 4}
    assert deep_merge(a, b) == {"x": {"a": 1, "b": 3}, "y": 1, "z": 4}
    assert a["x"]["b"] == 2  # inputs not mutated


def test_run_name():
    assert run_name("E3", "Qwen/Qwen3.5-2B", 0) == "E3_Qwen3.5-2B_s0"


def test_jsonl_roundtrip(tmp_path):
    rows = [{"a": 1, "b": "Việt"}, {"a": 2, "b": None}]
    p = tmp_path / "x.jsonl"
    write_jsonl(p, rows)
    assert read_jsonl(p) == rows


def test_write_train_log(tmp_path):
    p = tmp_path / "train_log.csv"
    write_train_log([{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.4, "lr": 1e-5}], p)
    lines = p.read_text().strip().splitlines()
    assert lines[0].split(",") == ["step", "loss", "lr"]
    assert len(lines) == 3
