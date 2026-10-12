from scripts.check_g1 import evaluate_gate


def _m(acc, parse=1.0):
    return {"accuracy": acc, "parse_rate": parse}


def test_gate_pass():
    ok, lines = evaluate_gate({"E0": _m(0.4), "E1": _m(0.45), "E2": _m(0.42), "E3": _m(0.6), "B1": _m(0.5)})
    assert ok


def test_gate_fail_not_better_than_zs():
    ok, _ = evaluate_gate({"E0": _m(0.4), "E1": _m(0.65), "E2": _m(0.42), "E3": _m(0.6), "B1": _m(0.5)})
    assert not ok


def test_gate_fail_parse_rate():
    ok, _ = evaluate_gate({"E0": _m(0.4), "E1": _m(0.4), "E2": _m(0.4), "E3": _m(0.6, 0.98), "B1": _m(0.5)})
    assert not ok


def test_gate_fail_missing_run():
    ok, lines = evaluate_gate({"E0": _m(0.4), "E3": _m(0.6)})
    assert not ok and any("thiếu" in l for l in lines)
