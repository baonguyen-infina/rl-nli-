from eval.predict import assemble_records, split_allowed

ROWS = [{"id": "a", "dataset": "vianli", "topic": "01", "label": "neutral", "premise": "p", "hypothesis": "h"},
        {"id": "b", "dataset": "vianli", "topic": "02", "label": "entailment", "premise": "p", "hypothesis": "h"}]


def test_assemble_records():
    recs = assemble_records(ROWS, [("<answer>neutral</answer>", 7), ("garbage", 3)], "label", False)
    assert [r["pred"] for r in recs] == ["neutral", None]
    assert [r["n_tokens"] for r in recs] == [7, 3]
    assert [r["id"] for r in recs] == ["a", "b"]


def test_split_allowed():
    assert split_allowed("dev") and split_allowed("dev_mini")
    assert not split_allowed("test") and not split_allowed("train")
