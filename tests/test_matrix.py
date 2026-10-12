import yaml
from configs.matrix import RUNS, build_run_config, write_run_configs
from common.io import load_config


def test_runs_defined():
    assert set(RUNS) == {"G0", "E0", "E1", "E2", "E3", "B1"}
    assert [RUNS[k]["format"] for k in ("E0", "E1", "E2")] == ["label", "cot", "align"]
    assert RUNS["E3"]["recipe"] == "sft" and RUNS["E3"]["format"] == "label"
    assert RUNS["B1"]["recipe"] == "encoder"
    assert all(RUNS[k]["recipe"] == "zs" for k in ("E0", "E1", "E2"))


def test_build_run_config_g0_overrides_steps():
    cfg = build_run_config("G0")
    assert cfg["run_id"] == "G0" and cfg["model"]["name"] == "Qwen/Qwen3.5-2B"
    assert cfg["grpo"]["max_steps"] == 1


def test_write_and_load_config(tmp_path, monkeypatch):
    import common.io as cio
    monkeypatch.setattr(cio, "RUN_CONFIGS", tmp_path)
    import configs.matrix as mx
    monkeypatch.setattr(mx, "RUN_CONFIGS", tmp_path)
    paths = write_run_configs()
    assert len(paths) == len(RUNS)
    cfg = load_config("E3")
    assert cfg["recipe"] == "sft" and cfg["lora"]["r"] == 32  # merged from base.yaml
    assert cfg["grpo"]["beta"] == 0.0 and cfg["grpo"]["max_steps"] == 500
    g0 = load_config("G0")
    assert g0["grpo"]["max_steps"] == 1 and g0["grpo"]["num_generations"] == 8
    assert yaml.safe_load(open(tmp_path / "E3.yaml"))["run_id"] == "E3"
