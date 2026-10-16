"""Suy luận greedy bằng vLLM (--language-model-only), ghi predictions_<split>.jsonl và metrics_<split>.json.
"""
import argparse
from pathlib import Path

import yaml

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
             gpu_mem=0.85, max_model_len=2048, reasoning_budget=None, engine_options=None, backend='vllm',
             model_revision=None):
    if backend=='transformers':
        from eval.hf_generate import generate_transformers
        return generate_transformers(model_name,messages_list,max_new_tokens,adapter,
                                     reasoning_budget=reasoning_budget,model_revision=model_revision)
    if backend!='vllm':raise ValueError('Unsupported evaluation backend')
    options = dict(engine_options or {})
    allowed = {'max_num_seqs', 'max_num_batched_tokens', 'enforce_eager', 'enable_prefix_caching', 'seed'}
    if set(options) - allowed:
        raise ValueError('Unsupported eval engine controls')
    for key in ('max_num_seqs', 'max_num_batched_tokens', 'seed'):
        if key in options and (type(options[key]) is not int or options[key] < (0 if key == 'seed' else 1)):
            raise ValueError('Invalid integer eval engine control: ' + key)
    for key in ('enforce_eager', 'enable_prefix_caching'):
        if key in options and type(options[key]) is not bool:
            raise ValueError('Invalid boolean eval engine control: ' + key)
    from tempfile import TemporaryDirectory
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    with TemporaryDirectory(prefix='rl-nli-vllm-adapter-') as temp:
        if adapter and model_name.lower().startswith('qwen/qwen3.5'):
            from eval.adapters import export_qwen35_adapter
            adapter = export_qwen35_adapter(adapter, temp)
        llm = LLM(model=model_name, language_model_only=True, dtype="bfloat16",
                  enable_lora=adapter is not None, max_lora_rank=64,
                  gpu_memory_utilization=gpu_mem, max_model_len=max_model_len, **options)
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        lora = LoRARequest("adapter", 1, str(adapter)) if adapter else None
        if reasoning_budget is not None:
            from train.common import chat_template_ids
            from train.native_rollout import bounded_generate
            tok = llm.get_tokenizer()
            prompts = [chat_template_ids(tok, messages) for messages in messages_list]
            rows = bounded_generate(llm, prompts, tok, temperature=0.,
                                    reasoning_budget=reasoning_budget, total_budget=max_new_tokens,
                                    lora_request=lora)
            return [(r['raw_output'], len(r['completion_ids'])) for r in rows]
        outs = llm.chat(messages_list, sp, lora_request=lora,
                       chat_template_kwargs={"enable_thinking": False})
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


def predict_split(cfg: dict, split: str, out_dir, adapter=None) -> dict:
    assert split_allowed(split), f"split {split!r} không được phép ở giai đoạn này"
    if cfg.get('readout_diagnostic'):
        from eval.readout_probe import predict_probe
        return predict_probe(cfg, split, out_dir, adapter)
    fmt, guess = cfg["format"], cfg.get("guess", False)
    rows = get_rows(split)
    msgs = [build_messages(r["premise"], r["hypothesis"], fmt, guess,
                           cfg.get('schema_examples', 'single')) for r in rows]
    ev = cfg["eval"]
    options = (cfg['validation_eval']['engine_options']
               if cfg.get('validation_protocol') == 'fixed_single_sequence' else ev.get('engine_options'))
    outputs = generate(cfg["model"]["name"], msgs, ev["max_new_tokens"][fmt], adapter,
                       ev["gpu_memory_utilization"], ev["max_model_len"],
                       reasoning_budget=cfg['native_pilot']['reasoning_budget'] if cfg.get('bounded_reasoning') else None,
                       engine_options=options,backend=ev.get('backend','vllm'),
                       model_revision=ev.get('model_revision'))
    recs = assemble_records(rows, outputs, fmt, guess)
    out_dir = Path(out_dir)
    write_jsonl(out_dir / f"predictions_{split}.jsonl", recs)
    metrics = compute_metrics(recs)
    write_json(out_dir / f"metrics_{split}.json", metrics)
    write_json(out_dir / 'eval_runtime.json', dict(protocol=cfg.get('validation_protocol','legacy_scheduler'),
               engine_options=options or {},backend=ev.get('backend','vllm'),temperature=0., split=split, n=len(rows)))
    if "report" in metrics:
        print(metrics["report"])
    return metrics


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--config", default=None,
                    help="Frozen config copied into the run directory")
    a = ap.parse_args(argv)
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid frozen run config: {a.config}")
    else:
        cfg = load_config(a.run)
    out = a.out_dir or run_dir(a.run, cfg["model"]["name"], a.seed)
    m = predict_split(cfg, a.split, out, a.adapter)
    print({k: m[k] for k in ("n", "accuracy", "macro_f1", "parse_rate")})


if __name__ == "__main__":
    main()
