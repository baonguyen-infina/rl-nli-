"""Định nghĩa ma trận E0–E13, A1–A6, B1–B3 (recipe, format, guess, init_from, cờ ablation, priority, seeds) và sinh configs/runs/<run_id>.yaml.
"""
import yaml

from common.io import RUN_CONFIGS, deep_merge

BASE_MODEL = "Qwen/Qwen3.5-2B"
ENCODER_MODEL = "uitnlp/CafeBERT"

RUNS = {
    "G0": dict(recipe="grpo", format="label", guess=False, model=BASE_MODEL, init_from=None,
               priority="P0", seeds=[0], overrides={"grpo": {"max_steps": 1}}),
    "E0": dict(recipe="zs", format="label", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "E1": dict(recipe="zs", format="cot", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "E2": dict(recipe="zs", format="align", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "E3": dict(recipe="sft", format="label", guess=False, model=BASE_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
    "B1": dict(recipe="encoder", format="label", guess=False, model=ENCODER_MODEL, init_from=None,
               priority="P1", seeds=[0], overrides={}),
}


def build_run_config(run_id: str) -> dict:
    spec = dict(RUNS[run_id])
    overrides = spec.pop("overrides")
    spec["model"] = {"name": spec["model"]}
    spec["run_id"] = run_id
    return deep_merge(spec, overrides)


def write_run_configs() -> list:
    RUN_CONFIGS.mkdir(parents=True, exist_ok=True)
    paths = []
    for run_id in RUNS:
        p = RUN_CONFIGS / f"{run_id}.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(build_run_config(run_id), f, allow_unicode=True, sort_keys=False)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in write_run_configs():
        print(p)
