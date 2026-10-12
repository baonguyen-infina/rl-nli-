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
    kw = {
        "output_dir": str(out_dir / "sft_ckpts"), "seed": seed,
        "learning_rate": s["lr"], "num_train_epochs": s["epochs"],
        "lr_scheduler_type": s["scheduler"], "per_device_train_batch_size": s["per_device_batch"],
        "gradient_accumulation_steps": s["grad_accum"], "max_length": s["max_len"],
        "bf16": True, "logging_steps": 10, "save_strategy": "epoch", "save_total_limit": None,
        "report_to": [],
    }
    ratio = s.get("warmup_ratio", 0.05)
    for try_assistant in (True, False):
        curr_kw = dict(kw)
        if try_assistant:
            curr_kw["assistant_only_loss"] = True
        try:
            return SFTConfig(warmup_ratio=ratio, **curr_kw)
        except TypeError:
            try:
                return SFTConfig(warmup_steps=ratio, **curr_kw)
            except TypeError:
                try:
                    return SFTConfig(**curr_kw)
                except TypeError:
                    continue
    return SFTConfig(**kw)


def train_sft(cfg: dict, seed: int, out_dir) -> list[Path]:
    patch_torchao_compatibility()
    from datasets import Dataset
    from trl import SFTTrainer
    out_dir = Path(out_dir)
    s = cfg["sft"]
    model, tok = load_model_and_tokenizer(cfg["model"]["name"])
    raw_rows = build_sft_dataset(get_rows("train"), cfg["format"], cfg.get("guess", False))
    formatted_data = [{"messages": r["prompt"] + r["completion"]} for r in raw_rows]
    ds = Dataset.from_list(formatted_data)
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

    kwargs = {
        "model": model, "args": args, "train_dataset": ds, "processing_class": tok,
        "peft_config": make_lora_config(cfg),
    }
    if collator is not None:
        kwargs["data_collator"] = collator

    trainer = SFTTrainer(**kwargs)
    trainer.train()
    write_train_log(trainer.state.log_history, out_dir / "train_log.csv")
    return sorted((out_dir / "sft_ckpts").glob("checkpoint-*"))
