from train.sft import pick_best


def test_pick_best_max_accuracy():
    assert pick_best({"checkpoint-100": 0.5, "checkpoint-200": 0.7, "checkpoint-300": 0.6}) == "checkpoint-200"


def test_pick_best_tie_prefers_latest():
    assert pick_best({"checkpoint-100": 0.7, "checkpoint-200": 0.7}) == "checkpoint-200"


def test_make_sft_config_compatibility():
    import sys
    from pathlib import Path
    from types import ModuleType
    from train.sft import _make_sft_config

    cfg_sft = {
        "lr": 2e-4, "epochs": 3, "warmup_ratio": 0.03, "scheduler": "cosine",
        "per_device_batch": 8, "grad_accum": 4, "max_len": 1024,
    }

    # Case 1: SFTConfig expects warmup_ratio (v4)
    called_args_v4 = {}
    class FakeSFTConfigV4:
        def __init__(self, **kwargs):
            if "warmup_ratio" not in kwargs:
                raise TypeError("expected warmup_ratio")
            called_args_v4.update(kwargs)

    fake_trl = ModuleType("trl")
    fake_trl.SFTConfig = FakeSFTConfigV4
    sys.modules["trl"] = fake_trl
    try:
        _make_sft_config(cfg_sft, 0, Path("/tmp/out"))
        assert called_args_v4["warmup_ratio"] == 0.03
    finally:
        sys.modules.pop("trl", None)

    # Case 2: SFTConfig rejects warmup_ratio and expects warmup_steps (v5)
    called_args_v5 = {}
    class FakeSFTConfigV5:
        def __init__(self, **kwargs):
            if "warmup_ratio" in kwargs:
                raise TypeError("unexpected keyword argument 'warmup_ratio'")
            called_args_v5.update(kwargs)

    fake_trl = ModuleType("trl")
    fake_trl.SFTConfig = FakeSFTConfigV5
    sys.modules["trl"] = fake_trl
    try:
        _make_sft_config(cfg_sft, 0, Path("/tmp/out"))
        assert called_args_v5["warmup_steps"] == 0.03
    finally:
        sys.modules.pop("trl", None)


def test_make_compute_metrics():
    import numpy as np
    from types import SimpleNamespace
    from train.sft import make_compute_metrics, preprocess_logits_for_metrics

    class DummyTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            mapping = {1: "<answer>entailment</answer>", 2: "<answer>neutral</answer>", 3: "<answer>contradiction</answer>"}
            return mapping.get(int(token_ids[0]), "") if len(token_ids) > 0 else ""

    cm = make_compute_metrics(DummyTokenizer(), "label")
    preds = np.array([[0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]])
    labels = np.array([[-100, -100, 1, -100], [-100, -100, 2, -100], [-100, -100, 3, -100]])
    eval_pred = SimpleNamespace(predictions=preds, label_ids=labels)
    res = cm(eval_pred)
    assert "macro_f1" in res and "accuracy" in res
    assert res["accuracy"] == 1.0 and res["macro_f1"] == 1.0

    # Test align format with prompt containing label names
    class AlignTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            mapping = {
                1: 'Prompt: entailment, neutral, contradiction.\n<align>\nH1: "a" <-> "b" | match\n</align>\n<answer>neutral</answer>',
                2: 'Prompt: entailment, neutral, contradiction.\n<align>\nH1: "a" <-> "b" | match\n</align>\n<answer>contradiction</answer>',
            }
            return mapping.get(int(token_ids[0]), "")

    cm_align = make_compute_metrics(AlignTokenizer(), "align", dev_golds=["neutral", "entailment"])
    preds_align = np.array([[0, 1, 0, 0], [0, 2, 0, 0]])
    labels_align = np.array([[-100, -100, 1, -100], [-100, -100, 2, -100]])
    eval_pred_align = SimpleNamespace(predictions=preds_align, label_ids=labels_align)
    res_align = cm_align(eval_pred_align)
    # sample 0 matches ('neutral' == 'neutral'), sample 1 differs ('contradiction' != 'entailment')
    assert res_align["accuracy"] == 0.5
    assert res_align["parse_rate"] == 1.0
