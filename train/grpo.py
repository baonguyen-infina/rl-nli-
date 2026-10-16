"""GRPO (TRL 1.15 GRPOTrainer) cho E6–E11, A1, A2; +HypW = nhân advantages theo w_i trong _generate_and_score_completions.
"""
from pathlib import Path

from common.io import write_train_log
from data.prepare import get_rows
from rewards.trl_rewards import make_reward_funcs
from train.common import (build_grpo_dataset, load_model_and_tokenizer, make_lora_config,
                          patch_torchao_compatibility,
                          patch_vllm_sync_weights_for_qwen35)


def train_grpo(cfg: dict, seed: int, out_dir) -> None:
    patch_torchao_compatibility()
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    out_dir = Path(out_dir)
    g = cfg["grpo"]
    fmt, guess = cfg["format"], cfg.get("guess", False)
    if g["use_vllm"] and cfg["model"]["name"].lower().startswith("qwen/qwen3.5"):
        patch_vllm_sync_weights_for_qwen35()
    model, tok = load_model_and_tokenizer(cfg["model"]["name"])
    ds = Dataset.from_list(build_grpo_dataset(get_rows("train"), fmt, guess,
                                            cfg.get('schema_examples', 'single')))
    args = GRPOConfig(
        output_dir=str(out_dir / "grpo_ckpts"), seed=seed,
        learning_rate=g["lr"], max_steps=g["max_steps"], beta=g["beta"], loss_type=g["loss_type"],
        num_generations=g["num_generations"], temperature=g["temperature"], top_p=g["top_p"],
        per_device_train_batch_size=g["per_device_batch"],
        gradient_accumulation_steps=g["grad_accum"],
        max_completion_length=g["max_completion_length"],
        use_vllm=g["use_vllm"], vllm_mode=g["vllm_mode"],
        vllm_gpu_memory_utilization=g["vllm_gpu_memory_utilization"],
        reward_weights=g["reward_weights"], bf16=True, fp16=False, logging_steps=1,
        chat_template_kwargs={"enable_thinking": False},
        top_k=0, save_strategy="steps", save_steps=g.get("save_steps", 100),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
    )
    init_run = cfg.get("init_from")
    if init_run:
        from common.io import RESULTS
        from peft import PeftModel
        candidates = list(RESULTS.glob(f"{init_run}_*_s{seed}/best_adapter"))
        if not candidates:
            raise FileNotFoundError(
                f"Không tìm thấy adapter khởi tạo từ {init_run} trong results/. "
                f"Bạn cần chạy SFT cho {init_run} trước: `python -m scripts.run --run {init_run} --seed {seed}`"
            )
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous init adapter: {candidates}")
        model = PeftModel.from_pretrained(model, str(candidates[0]), is_trainable=True)
        peft_cfg = None
    else:
        peft_cfg = make_lora_config(cfg)

    trainer_cls = GRPOTrainer
    if guess:
        from train.gta import make_gta_trainer
        trainer_cls = make_gta_trainer(cfg)
    from train.audit import with_rollout_audit, train_log_callback
    trainer_cls = with_rollout_audit(trainer_cls, cfg, out_dir)
    trainer = trainer_cls(model=model, reward_funcs=make_reward_funcs(fmt, guess), args=args,
                          train_dataset=ds, processing_class=tok, peft_config=peft_cfg,
                          callbacks=[train_log_callback(out_dir)])
    try:
        trainer.train()
        trainer.save_model(str(out_dir / "final_adapter"))
    finally:
        write_train_log(trainer.state.log_history, out_dir / "train_log.csv")
