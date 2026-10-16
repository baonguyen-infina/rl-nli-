"""Weak supervision from the gold NLI label, without annotated or generated rationales.

The independent categorical status distribution induces a distribution over the
three labels through the same conflict > missing > match rule used by the parser.
This is a factorized training objective, not a claim of semantic verification.
"""
import math
import re


def schema_status_support(lines):
    """Permitted status alternatives for the student's existing premise reference."""
    return [[False, True, False] if line.p is None else [True, False, True]
            for line in lines]


def rule_probabilities(rows):
    if not rows:
        raise ValueError('At least one alignment status is required')
    entailment = math.prod(row[0] for row in rows)
    no_conflict = math.prod(row[0] + row[1] for row in rows)
    return [entailment, no_conflict - entailment, 1 - no_conflict]


def alignment_status_positions(ids, tok, status_ids):
    raw = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    start = raw.find('<align>')
    end = raw.find('</align>', start + 7)
    if start < 0 or end < 0:
        return []
    spans = [m.span(1) for m in re.finditer(
        r'\|\s*(match|missing|conflict)(?=\s*(?:\n|</align>))', raw[start:end + 8])]
    spans = [(a + start, b + start) for a, b in spans]
    positions = []
    for index, token in enumerate(ids):
        if token not in status_ids:
            continue
        a = len(tok.decode(ids[:index], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        b = len(tok.decode(ids[:index + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        if any(a <= first and b >= last for first, last in spans):
            positions.append(index)
    return positions if len(positions) == len(spans) else []


def tensor_rule_probabilities(probs, mask):
    import torch
    default = torch.zeros_like(probs)
    default[..., 0] = 1
    probs = torch.where(mask[..., None], probs, default)
    match, missing, conflict = probs.unbind(-1)
    no_conflict = match + missing
    ones = torch.ones_like(match[:, :1])
    before_match = torch.cat([ones, match.cumprod(1)[:, :-1]], 1)
    before_no_conflict = torch.cat([ones, no_conflict.cumprod(1)[:, :-1]], 1)
    after_no_conflict = torch.cat([no_conflict.flip(1).cumprod(1).flip(1)[:, 1:], ones], 1)
    # Disjoint first-missing / first-conflict events avoid cancellation at tiny probabilities.
    return torch.stack([match.prod(1), (missing * before_match * after_no_conflict).sum(1),
                        (conflict * before_no_conflict).sum(1)], dim=-1)


def rule_supervision_loss(model, inputs, status_ids):
    import torch
    indices = inputs['rule_positions']
    mask = indices >= 0
    valid = mask.any(1)
    support = inputs.get('rule_support')
    if support is not None:
        has_missing = (mask & support[..., 1]).any(1)
        has_quote = (mask & support[..., 0]).any(1)
        gold = inputs['rule_gold']
        feasible = ((gold == 0) & ~has_missing) | ((gold == 1) & has_missing) | ((gold == 2) & has_quote)
        valid = valid & feasible
    if not valid.any():
        return None, 0.0
    prompt_width = inputs['prompt_ids'].size(1)
    positions = prompt_width + indices - 1
    union = torch.unique(positions[mask], sorted=True)
    output = model(input_ids=torch.cat([inputs['prompt_ids'], inputs['completion_ids']], 1),
                   attention_mask=torch.cat([inputs['prompt_mask'], inputs['completion_mask']], 1),
                   use_cache=False, logits_to_keep=union)
    lookup = torch.searchsorted(union, positions.clamp(min=int(union[0]), max=int(union[-1])))
    batch = torch.arange(indices.size(0), device=indices.device)[:, None]
    logits = output.logits[batch, lookup]
    status_logits = logits[..., status_ids].float()
    if support is not None:
        status_logits = status_logits.masked_fill(~support, float('-inf'))
    probs = status_logits.softmax(-1)
    induced = tensor_rule_probabilities(probs, mask)
    chosen = induced.gather(1, inputs['rule_gold'][:, None]).squeeze(1)
    return -chosen[valid].clamp(min=1e-8).log().mean(), float(valid.float().mean())
