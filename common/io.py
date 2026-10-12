"""Đường dẫn (artifacts/, results/, configs/runs/), đọc/ghi JSONL/JSON, đặt tên run `<run_id>_<model>_s<seed>`.
"""
import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
RUN_CONFIGS = ROOT / "configs" / "runs"


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_name(run_id: str, model: str, seed: int) -> str:
    return f"{run_id}_{model.split('/')[-1]}_s{seed}"


def run_dir(run_id: str, model: str, seed: int) -> Path:
    return RESULTS / run_name(run_id, model, seed)


def deep_merge(a: dict, b: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in a.items()}
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(run_id: str) -> dict:
    with open(ROOT / "configs" / "base.yaml", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    with open(RUN_CONFIGS / f"{run_id}.yaml", encoding="utf-8") as f:
        run = yaml.safe_load(f) or {}
    return deep_merge(base, run)


def write_train_log(log_history: list[dict], path) -> None:
    fields: list[str] = []
    for row in log_history:
        for k in row:
            if k not in fields:
                fields.append(k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in log_history:
            w.writerow(row)
