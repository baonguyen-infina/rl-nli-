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
