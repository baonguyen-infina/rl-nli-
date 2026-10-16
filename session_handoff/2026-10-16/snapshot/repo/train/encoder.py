"""Fine-tune encoder (CafeBERT) phân loại NLI (mode nli) hoặc chỉ hypothesis (mode hyp). Dùng cho B1 và hyp probe.
"""
import numpy as np

from common.labels import LABELS


def train_and_predict(train_rows, eval_rows, cfg: dict, mode: str, seed: int) -> np.ndarray:
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments, DataCollatorWithPadding)
    assert mode in ("nli", "hyp")
    e = cfg["encoder"]
    tok = AutoTokenizer.from_pretrained(e["name"])
    model = AutoModelForSequenceClassification.from_pretrained(e["name"], num_labels=len(LABELS))
    l2i = {l: i for i, l in enumerate(LABELS)}

    def enc(rows):
        ds = Dataset.from_list([{"premise": r["premise"], "hypothesis": r["hypothesis"],
                                 "labels": l2i[r["label"]]} for r in rows])

        def fn(b):
            if mode == "nli":
                return tok(b["premise"], b["hypothesis"], truncation=True, max_length=e["max_len"])
            return tok(b["hypothesis"], truncation=True, max_length=e["max_len"])
        return ds.map(fn, batched=True, remove_columns=["premise", "hypothesis"])

    args = TrainingArguments(
        output_dir="/tmp/encoder_out", seed=seed, learning_rate=e["lr"],
        num_train_epochs=e["epochs"], per_device_train_batch_size=e["batch"],
        per_device_eval_batch_size=e["batch"] * 2, bf16=torch.cuda.is_available(),
        save_strategy="no", report_to=[], logging_steps=50,
    )
    trainer = Trainer(model=model, args=args, train_dataset=enc(train_rows),
                      data_collator=DataCollatorWithPadding(tok))
    trainer.train()
    logits = trainer.predict(enc(eval_rows)).predictions
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)
