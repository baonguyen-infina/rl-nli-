"""GPU worker owned by scripts.run; its exit releases train and rollout engines."""
import argparse
import yaml
from common.io import load_config, run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    lookup = load_config(a.run)
    out = run_dir(a.run, lookup["model"]["name"], a.seed)
    with open(out / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg["recipe"] == "sft":
        from train.sft import train_sft
        train_sft(cfg, a.seed, out)
    elif cfg["recipe"] in ("grpo", "gta"):
        from train.grpo import train_grpo
        train_grpo(cfg, a.seed, out)
    elif cfg['recipe'] == 'native':
        from train.native_grpo import train_native
        train_native(cfg, a.seed, out)
    elif cfg["recipe"] == "encoder":
        from scripts.run import _run_encoder
        _run_encoder(cfg, a.seed, out)
    else:
        raise ValueError(cfg["recipe"])


if __name__ == "__main__":
    main()
