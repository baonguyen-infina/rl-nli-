"""Nạp Qwen3.5 phần ngôn ngữ, LoRA (q,k,v,o_proj; in_proj_qkv, in_proj_z, out_proj; gate,up,down_proj), dựng dataset prompt-completion.
"""
from formats.prompts import build_messages


def build_sft_dataset(rows: list[dict], fmt: str, guess: bool = False) -> list[dict]:
    if fmt != "label":
        raise NotImplementedError("SFT cot/align cần rationale từ teacher (giai đoạn G2)")
    return [{"prompt": build_messages(r["premise"], r["hypothesis"], fmt, guess),
             "completion": [{"role": "assistant", "content": f"<answer>{r['label']}</answer>"}]}
            for r in rows]


def build_grpo_dataset(rows: list[dict], fmt: str, guess: bool = False) -> list[dict]:
    return [{"prompt": build_messages(r["premise"], r["hypothesis"], fmt, guess),
             "gold": r["label"], "premise": r["premise"], "hypothesis": r["hypothesis"]}
            for r in rows]


def load_model_and_tokenizer(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    return model, tok


def make_lora_config(cfg: dict):
    from peft import LoraConfig
    lo = cfg["lora"]
    return LoraConfig(r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
                      target_modules=lo["target_modules"], task_type="CAUSAL_LM")
