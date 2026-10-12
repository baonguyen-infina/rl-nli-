"""Suy luận greedy bằng vLLM (--language-model-only), ghi predictions_<split>.jsonl và metrics_<split>.json.
"""
import argparse
from pathlib import Path

from common.io import load_config, run_dir, write_json, write_jsonl
from data.prepare import get_rows
from eval.metrics import compute_metrics
from eval.records import build_record
from formats.prompts import build_messages


def split_allowed(split: str) -> bool:
    return split in ("dev", "dev_mini")  # test chỉ mở ở G4 qua đường riêng


def assemble_records(rows, outputs, fmt, guess) -> list[dict]:
    return [build_record(r, text, fmt, guess, n_tok) for r, (text, n_tok) in zip(rows, outputs)]


def generate(model_name, messages_list, max_new_tokens, adapter=None,
             gpu_mem=0.85, max_model_len=2048):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=model_name, language_model_only=True, dtype="bfloat16",
              enable_lora=adapter is not None, max_lora_rank=64,
              gpu_memory_utilization=gpu_mem, max_model_len=max_model_len)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    lora = LoRARequest("adapter", 1, str(adapter)) if adapter else None
    outs = llm.chat(messages_list, sp, lora_request=lora,
                    chat_template_kwargs={"enable_thinking": False})
    return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


def predict_split(cfg: dict, split: str, out_dir, adapter=None) -> dict:
    assert split_allowed(split), f"split {split!r} không được phép ở giai đoạn này"
    fmt, guess = cfg["format"], cfg.get("guess", False)
    rows = get_rows(split)
    msgs = [build_messages(r["premise"], r["hypothesis"], fmt, guess) for r in rows]
    ev = cfg["eval"]
    outputs = generate(cfg["model"]["name"], msgs, ev["max_new_tokens"][fmt], adapter,
                       ev["gpu_memory_utilization"], ev["max_model_len"])
    recs = assemble_records(rows, outputs, fmt, guess)
    out_dir = Path(out_dir)
    write_jsonl(out_dir / f"predictions_{split}.jsonl", recs)
    metrics = compute_metrics(recs)
    write_json(out_dir / f"metrics_{split}.json", metrics)
    return metrics


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.run)
    out = a.out_dir or run_dir(a.run, cfg["model"]["name"], a.seed)
    m = predict_split(cfg, a.split, out, a.adapter)
    print({k: m[k] for k in ("n", "accuracy", "macro_f1", "parse_rate")})


if __name__ == "__main__":
    main()
