"""Test tách uid, số đếm, dev-mini cố định theo seed.
"""
import pytest
from data.topics import TOPICS
from data.prepare import (parse_uid, prepare_rows, check_counts, make_dev_mini,
                          load_raw, get_rows, EXPECTED_COUNTS)


def test_topics():
    assert sorted(TOPICS) == [f"{i:02d}" for i in range(1, 14)]


def test_parse_uid():
    d = parse_uid("uit_Adver_123_2_21_07")
    assert d == {"premise_id": "123", "k": "2", "intended_label": "neutral", "topic": "07"}
    for bad in ["uit_Adver_123_2_99_07", "uit_Adver_123_2_21_14", "foo", "uit_Adver_1_2_21"]:
        with pytest.raises(ValueError):
            parse_uid(bad)


def _raw(n):
    labels = ["entailment", "neutral", "contradiction"]
    return [{"uid": f"uit_Adver_{i}_1_{['11','21','31'][i % 3]}_{(i % 13) + 1:02d}",
             "premise": f"p{i}", "hypothesis": f"h{i}", "label": labels[i % 3]}
            for i in range(n)]


def test_prepare_rows_schema():
    rows = prepare_rows(_raw(6))
    assert set(rows[0]) == {"id", "dataset", "premise", "hypothesis", "label",
                            "topic", "premise_id", "intended_label"}
    assert rows[0]["dataset"] == "vianli" and rows[0]["id"] == "uit_Adver_0_1_11_01"
    assert rows[0]["intended_label"] == "entailment"


def test_check_counts():
    check_counts({"train": 8012, "dev": 1000})
    with pytest.raises(AssertionError):
        check_counts({"train": 8011})
    assert EXPECTED_COUNTS == {"train": 8012, "dev": 1000, "test": 1000}


def test_dev_mini_fixed_and_stratified():
    rows = prepare_rows(_raw(900))
    a = make_dev_mini(rows, n=300, seed=42)
    b = make_dev_mini(rows, n=300, seed=42)
    assert a == b and len(a) == 300 and a == sorted(a)
    by_label = {}
    for r in rows:
        by_label[r["id"]] = r["label"]
    counts = {}
    for i in a:
        counts[by_label[i]] = counts.get(by_label[i], 0) + 1
    assert set(counts.values()) == {100}
    assert make_dev_mini(rows, n=300, seed=1) != a


def test_test_split_is_guarded(monkeypatch):
    monkeypatch.delenv("G4_UNLOCK", raising=False)
    with pytest.raises(PermissionError):
        load_raw("test")
    with pytest.raises(PermissionError):
        get_rows("test")
    with pytest.raises(PermissionError):
        load_raw("test", allow_test=True)  # env var also required
