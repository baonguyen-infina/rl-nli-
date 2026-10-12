"""Điểm vào duy nhất: --run E11 --seed 0 -> đọc config, gọi đúng trainer, rồi predict trên dev.
"""
import argparse
import shutil
import subprocess
import sys
import time

import yaml

from common.io import load_config, read_json, run_dir, write_json, write_jsonl


def _predict_cli(run_id, seed, split, out_dir, adapter=None):
    cmd = [sys.executable, "-m", "eval.predict", "--run", run_id, "--seed", str(seed),
           "--split", split, "--out-dir", str(out_dir)]
    if adapter:
        cmd += ["--adapter", str(adapter)]
    subprocess.run(cmd, check=True)  # tiến trình riêng để giải phóng GPU sau khi train


def _run_sft(run_id, seed, cfg, out):
    from train.sft import train_sft
    ckpts = train_sft(cfg, seed, out)
    best = out / "best_adapter"
    if not best.exists() and ckpts:
        best = ckpts[-1]
    _predict_cli(run_id, seed, "dev", out, adapter=best)


def _run_encoder(cfg, seed, out):
    from data.prepare import get_rows
    from eval.metrics import compute_metrics
    from eval.records import encoder_record
    from common.labels import LABELS
    from train.encoder import train_and_predict
    dev = get_rows("dev")
    probs = train_and_predict(get_rows("train"), dev, cfg, mode="nli", seed=seed)
    recs = [encoder_record(r, LABELS[int(p.argmax())]) for r, p in zip(dev, probs)]
    write_jsonl(out / "predictions_dev.jsonl", recs)
    write_json(out / "metrics_dev.json", compute_metrics(recs))


def main(argv=None) -> None:
    try:
        from train.common import patch_torchao_compatibility, patch_vllm_sync_weights_for_qwen35
        patch_torchao_compatibility()
        patch_vllm_sync_weights_for_qwen35()
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = load_config(a.run)
    out = run_dir(a.run, cfg["model"]["name"], a.seed)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    t0 = time.time()
    recipe = cfg["recipe"]
    if recipe == "zs":
        _predict_cli(a.run, a.seed, cfg["eval"]["split"], out)
    elif recipe == "sft":
        _run_sft(a.run, a.seed, cfg, out)
    elif recipe == "grpo":
        from train.grpo import train_grpo
        train_grpo(cfg, a.seed, out)
        if a.run != "G0":
            adapter_path = out / "best_adapter"
            _predict_cli(a.run, a.seed, cfg["eval"]["split"], out,
                         adapter=adapter_path if adapter_path.exists() else None)
    elif recipe == "encoder":
        _run_encoder(cfg, a.seed, out)
    else:
        raise ValueError(f"recipe không hỗ trợ: {recipe}")
    cost = {"wall_seconds": round(time.time() - t0, 1)}
    try:
        import torch
        cost["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        cost["gpu"] = None
    write_json(out / "cost.json", cost)


if __name__ == "__main__":
    main()
