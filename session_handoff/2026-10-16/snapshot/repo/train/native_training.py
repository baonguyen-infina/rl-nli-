"""Student-prefix answer supervision and native-group reward boundaries."""
from contextlib import contextmanager
from train.common import chat_template_ids


def stop_after_callback(limit):
    """Stop at an audited prefix budget without changing the LR horizon."""
    if type(limit) is not int or limit<1:raise ValueError('Positive integer prefix budget required')
    from transformers import TrainerCallback
    class PrefixBudget(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step>=limit:
                control.should_training_stop=True
                control.should_save=True
            return control
    return PrefixBudget()


@contextmanager
def deterministic_scoring(model):
    """Score detached rewards without dropout, restoring every module's mode."""
    import torch
    modes=[(module,module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        # Restore mixed frozen/trainable branch modes without recursive overrides.
        for module,was_training in modes:
            module.training=was_training


def answer_prefix(raw):
    marker = '<answer>'
    at = raw.find(marker)
    return raw[:at + len(marker)] if at >= 0 else raw + marker


def build_answer_ce_rows(prompts, raw_outputs, gold, tok, *, active=None):
    n = len(prompts)
    active = [True] * n if active is None else active
    if not (len(raw_outputs) == len(gold) == len(active) == n):
        raise ValueError('Mismatched answer CE inputs')
    rows = []
    for prompt, raw, label, use in zip(prompts, raw_outputs, gold, active):
        prefix = chat_template_ids(tok, prompt) + tok.encode(answer_prefix(raw), add_special_tokens=False)
        suffix = tok.encode(label + '</answer>', add_special_tokens=False) + [tok.eos_token_id]
        ids = prefix + suffix
        rows.append(dict(input_ids=ids, labels=([-100] * len(prefix) + suffix) if use else [-100] * len(ids),
                         attention_mask=[1] * len(ids), weight=float(use)))
    return rows


def token_content_mask(raw, offsets, *, active=True, max_reasoning_tokens=64):
    if not active:
        return [0.0] * len(offsets)
    start = raw.find('<think>')
    start = start + len('<think>') if start >= 0 else -1
    end = raw.find('</think>', start) if start >= 0 else -1
    end = len(raw) if end < 0 else end
    count, mask = 0, []
    for a, b in offsets:
        if a == b:
            mask.append(0.0)
            continue
        reasoning = start >= 0 and b > start and a < end
        count += int(reasoning)
        mask.append(float(not reasoning or count <= max_reasoning_tokens))
    return mask


def redistribute_group_rewards(base_rewards, group_ids, rollout_ids, group_logq,
                               *, active=None, coefficient=0.2):
    n = len(base_rewards)
    active = [True] * n if active is None else active
    if not (len(group_ids) == len(rollout_ids) == len(active) == n):
        raise ValueError('Mismatched reward metadata')
    return [float(base + coefficient * group_logq[(g, k)]) if use else 0.0
            for base, g, k, use in zip(base_rewards, group_ids, rollout_ids, active)]


def weighted_answer_ce(logits, targets, row_weights):
    import torch.nn.functional as F
    weights = row_weights.to(logits).unsqueeze(-1) * (targets != -100)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                           targets.reshape(-1), ignore_index=-100, reduction='none').reshape_as(targets)
    return (loss * weights).sum() / weights.sum().clamp(min=1.)
