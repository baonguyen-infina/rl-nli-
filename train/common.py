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
    """Vá lỗi TRL sync_weights cho Qwen3.5 khi vLLM dùng Qwen3_5ForConditionalGeneration."""
    try:
        import trl.generation.vllm_generation as tvg
        orig_fix = tvg.VLLMGeneration._fix_param_name_to_vllm

        def patched_fix(self, name: str, extra_prefixes: list[str] | None = None) -> str:
            name = orig_fix(self, name, extra_prefixes)
            if not name.startswith("language_model.") and not name.startswith("visual."):
                return f"language_model.{name}"
            return name

        tvg.VLLMGeneration._fix_param_name_to_vllm = patched_fix

        orig_sync = tvg.VLLMGeneration.sync_weights

        def safe_sync_weights(self):
            try:
                orig_sync(self)
            except ValueError as e:
                if "There is no module or parameter named" in str(e):
                    if hasattr(self, "llm") and hasattr(self.llm, "llm_engine"):
                        runner = self.llm.llm_engine.model_executor.driver_worker.model_runner
                        for name, param in self._iter_named_params():
                            target_name = name if name.startswith("language_model.") else f"language_model.{name}"
                            try:
                                runner.model.load_weights([(target_name, param)])
                            except Exception:
                                pass
                        self.llm.reset_prefix_cache()
                else:
                    raise

        tvg.VLLMGeneration.sync_weights = safe_sync_weights
    except Exception:
        pass


def make_lora_config(cfg: dict):
    patch_torchao_compatibility()
    from peft import LoraConfig
    lo = cfg["lora"]
    return LoraConfig(r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
                      target_modules=lo["target_modules"], task_type="CAUSAL_LM")
