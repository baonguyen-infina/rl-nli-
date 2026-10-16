import pytest
from eval.metrics import compute_metrics
from eval.records import build_record, encoder_record

KEYS = {"id", "dataset", "topic", "gold", "pred", "guess", "parse_ok",
        "rule_label", "grounded", "coverage", "n_tokens", "raw_output"}
ROW = {"id": "x1", "dataset": "vianli", "topic": "03", "label": "entailment",
       "premise": "Hà Nội là thủ đô của Việt Nam và có hơn tám triệu dân.",
       "hypothesis": "Hà Nội là thủ đô của Việt Nam."}


def test_metrics_basic():
    recs = [
        {"gold": "entailment", "pred": "entailment", "topic": "01", "parse_ok": True},
        {"gold": "neutral", "pred": "entailment", "topic": "01", "parse_ok": True},
        {"gold": "contradiction", "pred": None, "topic": "02", "parse_ok": False},
        {"gold": "contradiction", "pred": "contradiction", "topic": "02", "parse_ok": True},
    ]
    m = compute_metrics(recs)
    assert m["n"] == 4 and m["accuracy"] == 0.5 and m["parse_rate"] == 0.75
    assert m["acc_by_topic"] == {"01": 0.5, "02": 0.5}
    assert m["confusion"]["contradiction"] == {"contradiction": 1, "unparsed": 1}
    # entailment: tp=1 fp=1 fn=0 -> F1 2/3 ; neutral: 0 ; contradiction: tp=1 fn=1 -> F1 2/3
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0 + 2 / 3) / 3)


def test_build_record_label_and_failure():
    r = build_record(ROW, "<answer>entailment</answer>", "label", False, 5)
    assert set(r) == KEYS and r["pred"] == "entailment" and r["parse_ok"] and r["gold"] == "entailment"
    assert r["rule_label"] is None and r["grounded"] is None and r["coverage"] is None
    bad = build_record(ROW, "???", "label", False, 3)
    assert bad["pred"] is None and bad["parse_ok"] is False and bad["raw_output"] == "???"


def test_build_record_align_fills_structure_fields():
    out = ('<align>\nH1: "Hà Nội là thủ đô của Việt Nam" <-> "Hà Nội là thủ đô của Việt Nam" | match\n'
           '</align><answer>entailment</answer>')
    r = build_record(ROW, out, "align", False, 40)
    assert r["rule_label"] == "entailment" and r["grounded"] is True and r["coverage"] == 1.0


def test_encoder_record():
    r = encoder_record(ROW, "neutral")
    assert set(r) == KEYS and r["pred"] == "neutral" and r["parse_ok"] is True
