"""SFT (TRL SFTTrainer), loss chỉ trên completion; dùng cho E3–E5.
"""
from pathlib import Path

from common.io import write_train_log
from data.prepare import get_rows
from train.common import (build_sft_dataset, load_model_and_tokenizer, make_lora_config,
                          patch_torchao_compatibility)


def pick_best(scores: dict[str, float]) -> str:
    best = max(scores.values())
    return sorted(k for k, v in scores.items() if v == best)[-1]


def _make_sft_config(s: dict, seed: int, out_dir: Path):
    from trl import SFTConfig
    eval_steps = s.get("eval_steps", 100)
    save_steps = s.get("save_steps", 100)
    kw = {
        "output_dir": str(out_dir / "sft_ckpts"), "seed": seed,
        "learning_rate": s["lr"], "num_train_epochs": s["epochs"],
        "lr_scheduler_type": s["scheduler"], "per_device_train_batch_size": s["per_device_batch"],
        "per_device_eval_batch_size": s.get("per_device_eval_batch", s["per_device_batch"]),
        "gradient_accumulation_steps": s["grad_accum"], "max_length": s["max_len"],
        "bf16": True, "fp16": False, "logging_steps": 10,
        "save_strategy": "steps", "save_steps": save_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": [],
    }
    ratio = s.get("warmup_ratio", 0.05)
    for try_assistant in (True, False):
        for eval_key in ("eval_strategy", "evaluation_strategy"):
            curr_kw = dict(kw)
            curr_kw[eval_key] = "steps"
            curr_kw["eval_steps"] = eval_steps
            if try_assistant:
                curr_kw["assistant_only_loss"] = True
            try:
                return SFTConfig(warmup_ratio=ratio, **curr_kw)
            except TypeError:
                try:
                    return SFTConfig(warmup_steps=ratio, **curr_kw)
                except TypeError:
                    continue
    # Minimal fallback
    try:
        return SFTConfig(warmup_ratio=ratio, **kw)
    except TypeError:
        return SFTConfig(**kw)


def train_sft(cfg: dict, seed: int, out_dir) -> list[Path]:
    patch_torchao_compatibility()
    from datasets import Dataset
    from trl import SFTTrainer
    out_dir = Path(out_dir)
    s = cfg["sft"]
    model, tok = load_model_and_tokenizer(cfg["model"]["name"])

    fmt, guess = cfg["format"], cfg.get("guess", False)
    train_raw = build_sft_dataset(get_rows("train"), fmt, guess)
    dev_raw = build_sft_dataset(get_rows("dev"), fmt, guess)

    train_ds = Dataset.from_list([{"messages": r["prompt"] + r["completion"]} for r in train_raw])
    dev_ds = Dataset.from_list([{"messages": r["prompt"] + r["completion"]} for r in dev_raw])

    args = _make_sft_config(s, seed, out_dir)

    collator = None
    if not getattr(args, "assistant_only_loss", False):
        try:
            from trl import DataCollatorForCompletionOnlyLM
            collator = DataCollatorForCompletionOnlyLM(
                response_template="<|im_start|>assistant\n",
                tokenizer=tok,
            )
        except Exception:
            collator = None

    patience = s.get("early_stopping_patience", 3)
    callbacks = []
    try:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    except Exception:
        pass

    kwargs = {
        "model": model, "args": args, "train_dataset": train_ds, "eval_dataset": dev_ds,
        "processing_class": tok, "peft_config": make_lora_config(cfg),
        "callbacks": callbacks,
    }
    if collator is not None:
        kwargs["data_collator"] = collator

    trainer = SFTTrainer(**kwargs)
    trainer.train()
    trainer.save_model(str(out_dir / "best_adapter"))
    write_train_log(trainer.state.log_history, out_dir / "train_log.csv")
    return sorted((out_dir / "sft_ckpts").glob("checkpoint-*"))
