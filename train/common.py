"""Nạp Qwen3.5 phần ngôn ngữ, LoRA (q,k,v,o_proj; in_proj_qkv, in_proj_z, out_proj; gate,up,down_proj), dựng dataset prompt-completion.
"""
from common.io import ARTIFACTS, read_jsonl
from formats.prompts import build_messages


def chat_template_ids(tok, messages) -> list[int]:
    """Normalize Transformers' list and BatchEncoding chat-template outputs."""
    encoded = tok.apply_chat_template(messages, tokenize=True,
                                     add_generation_prompt=True, enable_thinking=False)
    if isinstance(encoded, dict):
        if "input_ids" not in encoded:
            raise ValueError("apply_chat_template returned a mapping without input_ids")
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError("Expected one chat-template sequence, got a batch")
        encoded = encoded[0]
    ids = list(encoded)
    if any(not isinstance(token, int) for token in ids):
        raise TypeError("Chat-template input_ids must contain integer token IDs")
    return ids


def tokenize_sft_example(row: dict, tok, max_length: int) -> dict:
    """Use the same non-thinking prefix as greedy eval; never truncate the target."""
    prefix = chat_template_ids(tok, row["prompt"])
    target = tok.encode(row["completion"][0]["content"], add_special_tokens=False)
    if not target or target[-1] != tok.eos_token_id:
        target.append(tok.eos_token_id)
    ids = list(prefix) + target
    if len(ids) > max_length:
        raise ValueError(f"SFT example needs {len(ids)} tokens; max_length={max_length}")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + target}


def collate_sft_examples(examples, pad_token_id):
    import torch
    n = max(len(x["input_ids"]) for x in examples)
    return {key: torch.tensor([x[key] + [pad] * (n - len(x[key])) for x in examples])
            for key, pad in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))}


def build_sft_dataset(rows: list[dict], fmt: str, guess: bool = False) -> list[dict]:
    if fmt == "label":
        return [{"prompt": build_messages(r["premise"], r["hypothesis"], fmt, guess),
                 "completion": [{"role": "assistant", "content": f"<answer>{r['label']}</answer>"}]}
                for r in rows]

    teacher_file = ARTIFACTS / "teacher" / f"rationales_{fmt}.jsonl"
    if not teacher_file.exists():
        raise NotImplementedError(
            f"SFT {fmt} cần rationale từ teacher. Không tìm thấy {teacher_file}. "
            f"Hãy chạy `python -m teacher.generate --format {fmt}` trước."
        )

    teacher_rows = read_jsonl(teacher_file)
    comp_map = {r["id"]: r["completion"] for r in teacher_rows if "id" in r and "completion" in r}

    out = []
    for r in rows:
        rid = r.get("id")
        content = comp_map.get(rid)
        if content is None:
            from teacher.generate import synthesize_valid_rationale
            content = synthesize_valid_rationale(r["premise"], r["hypothesis"], r["label"], fmt)
            if content is None:
                content = f"<answer>{r['label']}</answer>"
        if guess and not content.startswith("<guess>"):
            content = f"<guess>{r['label']}</guess>{content}"
        out.append({
            "prompt": build_messages(r["premise"], r["hypothesis"], fmt, guess),
            "completion": [{"role": "assistant", "content": content}],
        })
    return out


def build_grpo_dataset(rows: list[dict], fmt: str, guess: bool = False,
                       schema_examples: str = 'single') -> list[dict]:
    return [{"prompt": build_messages(r["premise"], r["hypothesis"], fmt, guess, schema_examples),
             "gold": r["label"], "premise": r["premise"], "hypothesis": r["hypothesis"]}
            for r in rows]


def load_model_and_tokenizer(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    return model, tok


def patch_torchao_compatibility() -> None:
    """Vá lỗi PEFT khi môi trường có torchao không tương thích (< 0.16.0)."""
    import sys
    try:
        import peft.import_utils as piu
        orig_fn = getattr(piu, "is_torchao_available", None)
        if orig_fn is not None:
            def safe_is_torchao_available():
                try:
                    return orig_fn()
                except Exception:
                    return False
            piu.is_torchao_available = safe_is_torchao_available
    except Exception:
        pass

    mod = sys.modules.get("peft.tuners.lora.torchao")
    if mod is not None:
        mod.is_torchao_available = lambda: False


def patch_vllm_sync_weights_for_qwen35() -> None:
    """Map Qwen3.5 LM weights to vLLM's language_model namespace, surfacing sync errors."""
    try:
        import trl.generation.vllm_generation as tvg
    except ImportError:
        return
    if getattr(tvg, "_rl_nli_qwen35_sync_patch", False):
        return

    original_fix = tvg.VLLMGeneration._fix_param_name_to_vllm

    def patched_fix(self, name: str, extra_prefixes: list[str] | None = None) -> str:
        name = original_fix(self, name, extra_prefixes)
        if name.startswith(("language_model.", "visual.")):
            return name
        return f"language_model.{name}"

    tvg.VLLMGeneration._fix_param_name_to_vllm = patched_fix

    # TRL imports LLM as a module-level symbol. Restrict language-only loading to
    # this trainer path and this model family; eval uses its own explicit setting.
    original_llm = getattr(tvg, "LLM", None)
    if original_llm is not None:
        def qwen35_language_only_llm(*args, **kwargs):
            model_name = kwargs.get("model", args[0] if args else "")
            if str(model_name).lower().startswith("qwen/qwen3.5"):
                kwargs["language_model_only"] = True
            return original_llm(*args, **kwargs)

        tvg.LLM = qwen35_language_only_llm

    tvg._rl_nli_qwen35_sync_patch = True


def make_lora_config(cfg: dict):
    patch_torchao_compatibility()
    from peft import LoraConfig
    lo = cfg["lora"]
    return LoraConfig(r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
                      target_modules=lo["target_modules"], task_type="CAUSAL_LM")
