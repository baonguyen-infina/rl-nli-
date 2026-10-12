"""GRPO (TRL 1.15 GRPOTrainer) cho E6–E11, A1, A2; +HypW = nhân advantages theo w_i trong _generate_and_score_completions.
"""
from pathlib import Path

from common.io import write_train_log
from data.prepare import get_rows
from rewards.trl_rewards import make_reward_funcs
from train.common import (build_grpo_dataset, load_model_and_tokenizer, make_lora_config,
                          patch_torchao_compatibility)


def train_grpo(cfg: dict, seed: int, out_dir) -> None:
    patch_torchao_compatibility()
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    out_dir = Path(out_dir)
    g = cfg["grpo"]
    fmt, guess = cfg["format"], cfg.get("guess", False)
    model, tok = load_model_and_tokenizer(cfg["model"]["name"])
    ds = Dataset.from_list(build_grpo_dataset(get_rows("train"), fmt, guess))
    args = GRPOConfig(
        output_dir=str(out_dir / "grpo_ckpts"), seed=seed,
        learning_rate=g["lr"], max_steps=g["max_steps"], beta=g["beta"], loss_type=g["loss_type"],
        num_generations=g["num_generations"], temperature=g["temperature"], top_p=g["top_p"],
        per_device_train_batch_size=g["per_device_batch"],
        gradient_accumulation_steps=g["grad_accum"],
        max_completion_length=g["max_completion_length"],
        use_vllm=g["use_vllm"], vllm_mode=g["vllm_mode"],
        vllm_gpu_memory_utilization=g["vllm_gpu_memory_utilization"],
        reward_weights=g["reward_weights"], bf16=True, logging_steps=1,
        save_strategy="no", report_to=[],
    )
    init_run = cfg.get("init_from")
    if init_run:
        from common.io import RESULTS
        from peft import PeftModel
        candidates = list(RESULTS.glob(f"{init_run}_*_{seed}/best_adapter"))
        if not candidates:
            candidates = list(RESULTS.glob(f"{init_run}_*/best_adapter"))
        if not candidates:
            raise FileNotFoundError(f"Không tìm thấy adapter khởi tạo từ {init_run} trong results/")
        model = PeftModel.from_pretrained(model, str(candidates[0]), is_trainable=True)
        peft_cfg = None
    else:
        peft_cfg = make_lora_config(cfg)

    trainer = GRPOTrainer(model=model, reward_funcs=make_reward_funcs(fmt, guess), args=args,
                          train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(out_dir / "best_adapter"))
    write_train_log(trainer.state.log_history, out_dir / "train_log.csv")
