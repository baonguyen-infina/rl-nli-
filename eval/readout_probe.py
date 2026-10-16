"""Development-only comparison of greedy and canonical closed-set answer decoding.

Both answers use the exact same frozen, greedily generated reasoning prefix.
This diagnostic never selects a checkpoint or changes the primary greedy result.
"""
import hashlib
import json
import math
import random
from pathlib import Path

from common.labels import LABELS
from eval.metrics import compute_metrics


def canonical_answer_scores(model, tok, prefix):
    """Sum causal log probabilities of each complete label, closer, and EOS."""
    import torch
    if not prefix or tok.eos_token_id is None:
        raise ValueError('A nonempty exact token prefix and EOS are required')
    prefix = list(prefix)
    device = model.get_input_embeddings().weight.device
    scores, lengths = {}, {}
    with torch.inference_mode():
        for label in LABELS:
            suffix = tok.encode(label + '</answer>', add_special_tokens=False) + [tok.eos_token_id]
            ids = torch.tensor([prefix + suffix], device=device, dtype=torch.long)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           logits_to_keep=len(suffix) + 1).logits[:, :-1].float()
            targets = ids[:, -len(suffix):]
            logps = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            score = float(logps.sum())
            if not math.isfinite(score):
                raise ValueError('Nonfinite canonical answer score')
            scores[label], lengths[label] = score, len(suffix)
    best = max(scores, key=scores.get)
    masses = {label: math.exp(value - scores[best]) for label, value in scores.items()}
    total = sum(masses.values())
    return {'scores': scores, 'map_label': best,
            'probabilities': {label: mass / total for label, mass in masses.items()},
            'candidate_lengths': lengths, 'prefix_length': len(prefix),
            'prefix_sha256': hashlib.sha256(json.dumps(prefix, separators=(',', ':')).encode()).hexdigest()}


def probe_indices(n, limit, seed):
    """Choose rows before scoring, without consulting their labels or contents."""
    if type(n) is not int or n < 1 or type(limit) is not int or limit < 1:
        raise ValueError('Positive dataset and subset sizes are required')
    return sorted(random.Random(seed).sample(range(n), min(n, limit)))


def probe_summary(greedy, mapped):
    if (not greedy or len(greedy) != len(mapped)
            or len({row['id'] for row in greedy}) != len(greedy)
            or any((left['id'], left['gold']) != (right['id'], right['gold'])
                   for left, right in zip(greedy, mapped))):
        raise ValueError('Readout comparisons require the same ordered unique examples and gold labels')
    repairs = sum(left['pred'] != left['gold'] and right['pred'] == right['gold']
                  for left, right in zip(greedy, mapped))
    regressions = sum(left['pred'] == left['gold'] and right['pred'] != right['gold']
                      for left, right in zip(greedy, mapped))
    return {'split': 'dev_subset', 'selection_allowed': False,
            'greedy': compute_metrics(greedy), 'canonical_map': compute_metrics(mapped),
            'repairs': repairs, 'regressions': regressions, 'net_repairs': repairs - regressions,
            'disagreements': sum(left['pred'] != right['pred'] for left, right in zip(greedy, mapped))}


def predict_probe(cfg, split, out_dir, adapter):
    """Write the paired diagnostic under the ordinary run entry point, dev only."""
    from common.io import write_json, write_jsonl
    from data.prepare import get_rows
    from eval.hf_generate import generate_transformers
    from eval.records import build_record
    from formats.prompts import build_messages
    if (split != 'dev' or cfg['format'] != 'cot' or cfg.get('guess')
            or not cfg.get('bounded_reasoning') or cfg['eval'].get('backend') != 'transformers'):
        raise ValueError('Readout diagnostic requires bounded Transformers CoT on dev only')
    out_dir = Path(out_dir)
    rows = get_rows('dev')
    profile = cfg['readout_probe']
    indices = probe_indices(len(rows), profile['n'], profile['subset_seed'])
    rows = [rows[index] for index in indices]
    # Persist membership before any model output is observed.
    write_json(out_dir / 'probe_protocol.json', {
        'split': 'dev_subset', 'indices': indices, 'subset_seed': profile['subset_seed'],
        'n': len(rows), 'selection_allowed': False, 'teacher': False,
        'primary_decoding': 'single_sequence_token_greedy',
        'secondary_decoding': 'canonical_suffix_logprob_sum_MAP_with_EOS'})
    messages = [build_messages(row['premise'], row['hypothesis'], 'cot', False,
                              cfg.get('schema_examples', 'single')) for row in rows]
    audit = []
    outputs = generate_transformers(cfg['model']['name'], messages,
        cfg['eval']['max_new_tokens']['cot'], adapter,
        reasoning_budget=cfg['native_pilot']['reasoning_budget'],
        model_revision=cfg['eval'].get('model_revision'), answer_scores=audit)
    if len(outputs) != len(rows) or len(audit) != len(rows):
        raise RuntimeError('Incomplete paired answer-readout diagnostic')
    greedy, mapped, records = [], [], []
    for row, (text, n_tokens), scores in zip(rows, outputs, audit):
        left = build_record(row, text, 'cot', False, n_tokens)
        boundary = text.rfind('<answer>')
        if boundary < 0:
            raise RuntimeError('Bounded greedy output is missing its answer opener')
        mapped_text = text[:boundary] + '<answer>' + scores['map_label'] + '</answer>'
        right = build_record(row, mapped_text, 'cot', False,
            scores['completion_prefix_length'] + scores['candidate_lengths'][scores['map_label']])
        greedy.append(left)
        mapped.append(right)
        records.append({'id': row['id'], 'gold': row['label'], 'greedy_pred': left['pred'],
                        'greedy_parse_ok': left['parse_ok'], **scores})
    summary = probe_summary(greedy, mapped)
    write_jsonl(out_dir / 'predictions_dev_subset.jsonl', greedy)
    write_json(out_dir / 'metrics_dev_subset.json', summary['greedy'])
    write_jsonl(out_dir / 'predictions_dev_subset_canonical_map.jsonl', mapped)
    write_json(out_dir / 'metrics_dev_subset_canonical_map.json', summary['canonical_map'])
    write_jsonl(out_dir / 'readout_audit.jsonl', records)
    write_json(out_dir / 'probe_summary.json', summary)
    write_json(out_dir / 'eval_runtime.json', {'split': 'dev_subset', 'n': len(rows),
        'backend': 'transformers', 'temperature': 0., 'selection_allowed': False,
        'protocol': 'frozen_greedy_reasoning_paired_readout'})
    return summary['greedy']
