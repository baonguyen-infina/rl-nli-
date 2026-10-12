import pytest
from train.common import build_sft_dataset, build_grpo_dataset

ROWS = [{"id": "uit_Adver_1_1_11_01", "dataset": "vianli", "premise": "P1", "hypothesis": "H1",
         "label": "neutral", "topic": "01", "premise_id": "1", "intended_label": "entailment"}]


def test_sft_dataset_label():
    ds = build_sft_dataset(ROWS, "label")
    assert ds[0]["completion"] == [{"role": "assistant", "content": "<answer>neutral</answer>"}]
    assert ds[0]["prompt"][0]["role"] == "user" and "P1" in ds[0]["prompt"][0]["content"]
    assert set(ds[0]) == {"prompt", "completion"}


def test_no_uid_or_intended_label_leak():
    for ds in (build_sft_dataset(ROWS, "label"), build_grpo_dataset(ROWS, "label")):
        blob = str(ds)
        assert "uit_Adver" not in blob
        assert "intended" not in blob


def test_grpo_dataset_columns():
    ds = build_grpo_dataset(ROWS, "label")
    assert ds[0]["gold"] == "neutral" and ds[0]["premise"] == "P1" and ds[0]["hypothesis"] == "H1"
    assert set(ds[0]) == {"prompt", "gold", "premise", "hypothesis"}


def test_sft_other_formats_not_yet_supported():
    with pytest.raises(NotImplementedError):
        build_sft_dataset(ROWS, "align")
