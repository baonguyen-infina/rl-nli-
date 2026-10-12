"""SFT (TRL SFTTrainer), loss chỉ trên completion; dùng cho E3–E5.
"""
from pathlib import Path

from common.io import write_train_log
from common.labels import LABELS
from data.prepare import get_rows
from formats.parser import parse
from train.common import (build_sft_dataset, load_model_and_tokenizer, make_lora_config,
                          patch_torchao_compatibility)


def pick_best(scores: dict[str, float]) -> str:
    best = max(scores.values())
    return sorted(k for k, v in scores.items() if v == best)[-1]


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def make_compute_metrics(tok, fmt, guess=False, dev_golds=None):
    import re

    def compute_metrics(eval_pred):
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(labels, tuple):
            labels = labels[0]
        if hasattr(preds, "shape") and hasattr(labels, "shape") and len(preds.shape) == 2 and len(labels.shape) == 2:
            preds = preds[:, :-1]
            labels = labels[:, 1:]
        golds, predicted_labels, parse_oks = [], [], []
        for i in range(len(labels)):
            gl = None
            if dev_golds is not None and i < len(dev_golds):
                gl = dev_golds[i]
            if gl is None:
                mask_g = (labels[i] != -100)
                if mask_g.any():
                    gold_toks = labels[i][mask_g]
                    if hasattr(gold_toks, "tolist"):
                        gold_toks = gold_toks.tolist()
                    gold_text = tok.decode(gold_toks, skip_special_tokens=True).strip()
                    m_gold = re.search(r"<answer>\s*(entailment|neutral|contradiction)\s*</answer>", gold_text)
                    if m_gold:
                        gl = m_gold.group(1)
            if gl is None:
                continue
            golds.append(gl)

            mask_p = (labels[i] != -100)
            pred_toks = preds[i][mask_p] if mask_p.any() else preds[i]
            if hasattr(pred_toks, "tolist"):
                pred_toks = pred_toks.tolist()
            pred_text = tok.decode(pred_toks, skip_special_tokens=True).strip()

            comp = pred_text
            if "<|im_start|>assistant" in pred_text:
                comp = pred_text.split("<|im_start|>assistant")[-1]
            elif fmt == "align" and "<align>" in pred_text:
                comp = pred_text[pred_text.rfind("<align>"):]
            elif fmt == "cot" and "<think>" in pred_text:
                comp = pred_text[pred_text.rfind("<think>"):]
            elif fmt == "label" and "<answer>" in pred_text:
                comp = pred_text[pred_text.rfind("<answer>"):]
            elif guess and "<guess>" in pred_text:
                comp = pred_text[pred_text.rfind("<guess>"):]
            comp = comp.strip()

            p_pred = parse(comp, fmt, guess)
            pl = p_pred.label
            parse_ok = p_pred.format_ok
            if pl is None:
                m_ans = re.search(r"<answer>\s*(entailment|neutral|contradiction)\s*</answer>", comp)
                if m_ans:
                    pl = m_ans.group(1)
            predicted_labels.append(pl)
            parse_oks.append(parse_ok)

        if not golds:
            return {"macro_f1": 0.0, "accuracy": 0.0, "parse_rate": 0.0}

        correct = sum(1 for g, p in zip(golds, predicted_labels) if g == p and p is not None)
        acc = correct / len(golds)
        f1s = []
        for lab in LABELS:
            tp = sum(1 for g, p in zip(golds, predicted_labels) if g == lab and p == lab)
            fp = sum(1 for g, p in zip(golds, predicted_labels) if g != lab and p == lab)
            fn = sum(1 for g, p in zip(golds, predicted_labels) if g == lab and p != lab)
            denom = 2 * tp + fp + fn
            f1s.append(2 * tp / denom if denom else 0.0)
        macro_f1 = sum(f1s) / len(f1s)
        parse_rate = sum(1 for ok in parse_oks if ok) / len(parse_oks) if parse_oks else 0.0
        return {
            "macro_f1": round(macro_f1, 4),
            "accuracy": round(acc, 4),
            "parse_rate": round(parse_rate, 4),
        }
    return compute_metrics


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
        "metric_for_best_model": "eval_macro_f1",
        "greater_is_better": True,
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
    dev_rows = get_rows("dev")
    train_raw = build_sft_dataset(get_rows("train"), fmt, guess)
    dev_raw = build_sft_dataset(dev_rows, fmt, guess)
    dev_golds = [r["label"] for r in dev_rows]

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
        "compute_metrics": make_compute_metrics(tok, fmt, guess, dev_golds=dev_golds),
        "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
        "callbacks": callbacks,
    }
    if collator is not None:
        kwargs["data_collator"] = collator

    trainer = SFTTrainer(**kwargs)
    trainer.train()
    trainer.save_model(str(out_dir / "best_adapter"))
    write_train_log(trainer.state.log_history, out_dir / "train_log.csv")
    del trainer, model
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return sorted((out_dir / "sft_ckpts").glob("checkpoint-*"))
