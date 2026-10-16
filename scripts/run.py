"""Điểm vào duy nhất: --run E11 --seed 0 -> đọc config, gọi đúng trainer, rồi predict trên dev.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import yaml

from common.io import RESULTS, load_config, read_json, run_dir, write_json, write_jsonl


def resolve_initial_adapter(cfg, seed, results=RESULTS):
    if not cfg.get('init_from'):
        return None
    checkpoint=cfg.get('init_checkpoint')
    if checkpoint is not None and (type(checkpoint) is not int or checkpoint<1):
        raise ValueError('init_checkpoint must be a positive integer')
    suffix='best_adapter' if checkpoint is None else f'grpo_ckpts/checkpoint-{checkpoint}'
    candidates = list(results.glob(f"{cfg['init_from']}_*_s{seed}/{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"Missing same-seed warm start: {cfg['init_from']} seed {seed}")
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous warm start: {candidates}")
    return candidates[0]


def run_training_process(cmd):
    """All training GPU allocations belong to a child that exits before eval."""
    subprocess.run(cmd, check=True)


def select_checkpoint(checkpoints, scores):
    return max(checkpoints, key=lambda p: (scores[p.name]["macro_f1"],
                                          int(p.name.rsplit("-", 1)[-1])))


def frozen_config(path):
    """Read the immutable config copied into a run directory."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid frozen run config: {path}")
    return cfg


def _predict_cli(run_id, seed, split, out_dir, adapter=None, config_path=None):
    cmd = [sys.executable, "-m", "eval.predict", "--run", run_id, "--seed", str(seed),
           "--split", split, "--out-dir", str(out_dir)]
    if adapter:
        cmd += ["--adapter", str(adapter)]
    if config_path:
        cmd += ["--config", str(config_path)]
    subprocess.run(cmd, check=True)  # tiến trình riêng để giải phóng GPU sau khi train


def _predict_final_cli(entry,protocol_path):
    env={**os.environ,'HF_HUB_OFFLINE':'0','TRANSFORMERS_OFFLINE':'1'}
    subprocess.run([sys.executable,'-m','eval.final_predict','--run',entry['run_id'],
                    '--seed',str(entry['seed']),'--protocol',str(protocol_path)],check=True,env=env)


def run_final_test(run_id,seed,protocol_path):
    from eval.final_protocol import validate_protocol,sha256_path,claim_test_run,complete_test_run
    protocol_path=__import__('pathlib').Path(protocol_path)
    entries=validate_protocol(read_json(protocol_path),root=RESULTS.parent)
    selected=[e for e in entries if e['run_id']==run_id and e['seed']==seed]
    if len(selected)!=1:raise ValueError('Run/seed absent from locked test protocol')
    entry=selected[0];out=entry['source_config_path'].parent
    digest=sha256_path(protocol_path)
    claim=claim_test_run(out,digest)
    if claim['reused']:
        print({'test_results_reused':True,'run':run_id});return
    t0=time.time();cost={'status':'in_progress','manifest_sha256':digest}
    try:
        _predict_final_cli(entry,protocol_path)
        cost.update(status='complete',wall_seconds=round(time.time()-t0,1))
        write_json(out/'cost_test.json',cost)
        complete_test_run(out,digest)
        original=read_json(out/'cost.json') if (out/'cost.json').exists() else {}
        if original.get('status')=='prepared':
            write_json(out/'cost.json',{**original,'status':'complete','last_phase':'test',
                                      'wall_seconds':cost['wall_seconds']})
    except Exception as exc:
        cost.update(status='failed',error=f'{type(exc).__name__}: {exc}',wall_seconds=round(time.time()-t0,1))
        write_json(out/'cost_test.json',cost)
        raise


def _evaluate_checkpoints(run_id, seed, cfg, out):
    recipe = cfg["recipe"]
    folder = "sft_ckpts" if recipe == "sft" else "grpo_ckpts"
    ckpts = sorted((out / folder).glob("checkpoint-*"))
    if ckpts:
        scores = {}
        for ck in ckpts:
            ev = out / "ckpt_eval" / ck.name
            if not (ev / "metrics_dev.json").exists():
                _predict_cli(run_id, seed, "dev", ev, adapter=ck,
                             config_path=out / "config.yaml")
            scores[ck.name] = read_json(ev / "metrics_dev.json")
        best = select_checkpoint(ckpts, scores)
        write_json(out / "ckpt_scores.json", {"scores": scores, "best": best.name,
                                              "selection_split": "dev", "selection_metric": "macro_f1"})
        if (out / "best_adapter").exists():
            shutil.rmtree(out / "best_adapter")
        shutil.copytree(best, out / "best_adapter")
        for name in ("predictions_dev.jsonl", "metrics_dev.json"):
            shutil.copy(out / "ckpt_eval" / best.name / name, out / name)
    else:
        best = out / "best_adapter"
        _predict_cli(run_id, seed, "dev", out, adapter=best,
                     config_path=out / "config.yaml")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase", choices=["all", "train", "eval", "prepare", "test"], default="all")
    ap.add_argument('--protocol',default=None)
    a = ap.parse_args(argv)
    if a.phase=='test':
        if not a.protocol:raise ValueError('A locked test protocol is required')
        run_final_test(a.run,a.seed,a.protocol);return
    cfg = load_config(a.run)
    out = run_dir(a.run, cfg["model"]["name"], a.seed)
    if a.phase == "eval":
        if not (out / "config.yaml").exists():
            raise FileNotFoundError(f"No frozen run config in {out}")
        cfg = frozen_config(out / "config.yaml")
    elif (out / "config.yaml").exists():
        raise FileExistsError(f"Run exists: {out}; use --phase eval or a new matrix ID")
    out.mkdir(parents=True, exist_ok=True)
    if a.phase != "eval":
        with open(out / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    if a.phase=='prepare':
        write_json(out/'cost.json',{'status':'prepared','wall_seconds':0.,'last_phase':'prepare'})
        return
    t0 = time.time()
    recipe = cfg["recipe"]
    cost = read_json(out / "cost.json") if (out / "cost.json").exists() else {}
    try:
        if a.phase != "eval" and recipe != "zs":
            if recipe == 'native' and not a.run.startswith('G0'):
                _predict_cli(a.run, a.seed, 'dev', out / 'initial_eval',
                             adapter=resolve_initial_adapter(cfg, a.seed),
                             config_path=out / 'config.yaml')
            run_training_process([sys.executable, "-m", "scripts.train_worker", "--run", a.run,
                                  "--seed", str(a.seed)])
        if a.phase != "train":
            if recipe == "zs":
                _predict_cli(a.run, a.seed, cfg["eval"]["split"], out,
                             adapter=resolve_initial_adapter(cfg, a.seed),
                             config_path=out / "config.yaml")
            elif recipe in ("sft", "grpo", "gta", 'native') and a.run != "G0":
                _evaluate_checkpoints(a.run, a.seed, cfg, out)
        cost["status"] = "trained" if a.phase == "train" else "complete"
    except Exception as exc:
        cost.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        cost["wall_seconds"] = cost.get("wall_seconds", 0) + round(time.time() - t0, 1)
        cost["last_phase"] = a.phase
        write_json(out / "cost.json", cost)


if __name__ == "__main__":
    main()
