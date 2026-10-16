"""Teacher-free native hypothesis contrast GRPO with answer-only auxiliary CE."""
import hashlib
import json
from pathlib import Path

from common.io import write_json, write_train_log
from common.labels import LABELS
from data.prepare import get_rows
from eval.records import build_record
from formats.parser import parse
from formats.prompts import build_messages
from train.common import (chat_template_ids, load_model_and_tokenizer,
                          patch_torchao_compatibility, patch_vllm_sync_weights_for_qwen35)
from train.native_groups import build_native_blocks, build_scheduled_random_blocks, pad_blocks, log_allocation
from train.native_rollout import block_sample_indices, rollout_func
from train.native_training import deterministic_scoring, redistribute_group_rewards, weighted_answer_ce
from train.native_allocation import allocation_linearization, allocation_surrogate
from train.native_saliency import native_token_credit, apply_reasoning_credit
from train.native_trace_swap import native_swap_indices
from train.native_reward import (bounded_correct_allocation_rewards, bounded_correct_swap_rewards,
                                 policy_loss_coefficient)
from train.label_marginal import label_marginal_linearization,example_rloo


def adapter_digest(model, name):
    from peft import get_peft_model_state_dict
    h = hashlib.sha256()
    state = get_peft_model_state_dict(model, adapter_name=name)
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().view(-1).view(dtype=__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def _pad_ce(rows, tok, device, min_length=0):
    import torch
    length = max(min_length, max(len(r['input_ids']) for r in rows))
    return {key: torch.tensor([[fill] * (length - len(r[key])) + r[key] for r in rows], device=device)
            for key, fill in [('input_ids', tok.pad_token_id), ('attention_mask', 0), ('labels', -100)]}


def _answer_row(prefix, label, tok, active=True):
    suffix = tok.encode(label + '</answer>', add_special_tokens=False) + [tok.eos_token_id]
    return dict(input_ids=prefix + suffix, attention_mask=[1] * (len(prefix) + len(suffix)),
                labels=[-100] * len(prefix) + suffix if active else [-100] * (len(prefix) + len(suffix)))


def make_native_trainer(cfg, out_dir):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Sampler
    from trl import GRPOTrainer
    from trl.models.utils import disable_gradient_checkpointing
    from train.audit import train_log_callback
    g = cfg['native_pilot']
    marginal_enabled=cfg['native_mode'] in ('marginal_accuracy','marginal_f1')
    allocation_enabled = cfg['native_mode'] in ('allocation', 'joint', 'saliency', 'trace_swap', 'trace_saliency',
                                               'gated_joint','gated_trace_swap','allocation_only') or marginal_enabled
    saliency_enabled = cfg['native_mode'] in ('saliency', 'trace_saliency')
    trace_enabled = cfg['native_mode'] in ('trace_swap', 'trace_saliency','gated_trace_swap')
    if allocation_enabled and not g.get('disable_dropout', False):
        raise ValueError('Direct allocation replay requires disabled dropout')

    class WholeBlockSampler(Sampler):
        def __init__(self, n):
            if n % 3:
                raise ValueError('Native dataset is not rectangular')
            self.groups, self.epoch = n // 3, 0
        def __iter__(self):
            indices = block_sample_indices(self.groups, groups_per_batch=g['groups_per_update'],
                                           rollouts=g['num_generations'], seed=self.seed + self.epoch,
                                           repeat_count=g['grad_accum'])
            self.epoch += 1
            return iter(indices)
        def __len__(self):
            return self.groups // g['groups_per_update'] * g['groups_per_update'] * 3 * g['num_generations'] * g['grad_accum']
        seed = cfg['seed']

    class NativeTrainer(GRPOTrainer):
        native_cfg = g
        label_marginal_enabled=marginal_enabled

        def _get_train_sampler(self, dataset=None):
            return WholeBlockSampler(len(self.train_dataset if dataset is None else dataset))

        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            if self.accelerator.num_processes != 1:
                raise ValueError('This prespecified pilot requires one GPU')
            device, tok = self.accelerator.device, self.processing_class
            keys, counters, members, rewards, ce_rows = [], {}, {}, [], []
            actual_groups = list(dict.fromkeys(row['group_key'] for row in inputs))
            start = self.state.global_step * g['groups_per_update']
            if actual_groups != self.native_group_order[start:start+g['groups_per_update']]:
                raise ValueError('Observed training exposure differs from prespecified update schedule')
            for row in inputs:
                slot_key = (row['group_key'], row['slot'])
                k = counters.get(slot_key, 0)
                counters[slot_key] = k + 1
                keys.append((row['group_key'], k))
                members.setdefault((row['group_key'], k), []).append(len(keys)-1)
                parsed = parse(row['raw_output'], 'cot', False)
                rewards.append(float(parsed.label == row['gold']) + .2 * float(parsed.format_ok))
                ce_rows.append(_answer_row(row['prompt_ids_native'] + row['prefix_ids'], row['gold'], tok, row['slot_weight']))
            if any(n != g['num_generations'] for n in counters.values()) or any(len(v) != 3 for v in members.values()):
                raise ValueError('Generation batch split or duplicated a hypothesis block')
            coef = g['allocation_weight'] if cfg['native_mode'] in ('native', 'random', 'joint', 'trace_swap', 'trace_saliency','gated_joint','gated_trace_swap') else 0.
            logq = {key: 0. for key in keys}
            self._native_allocation = None
            self._native_credit = None
            if coef or allocation_enabled:
                score_rows = [_answer_row(row['prompt_ids_native'] + row['prefix_ids'], lab, tok)
                              for row in inputs for lab in LABELS]
                # TRL shuffles rows before loss microbatching. Keep the same
                # sequence width during cache and replay: BF16 DeltaNet kernels
                # can otherwise change scores when left padding changes.
                score_length = max(len(row['input_ids']) for row in score_rows)
                suffix_length = max(len(_answer_row([],lab,tok)['input_ids']) for lab in LABELS)
                if trace_enabled:
                    # Every original/donor pairing fits this same geometry. The
                    # padding width must not become a numerical reward signal.
                    score_length=max(len(row['prompt_ids_native']) for row in inputs)+max(
                        len(row['prefix_ids']) for row in inputs)+suffix_length
                scores = []
                scoring_context = deterministic_scoring(self.model) if g.get('disable_dropout', False) else torch.no_grad()
                with scoring_context, disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                    for at in range(0, len(score_rows), g['score_microbatch']):
                        batch = _pad_ce(score_rows[at:at+g['score_microbatch']], tok, device,
                                        min_length=score_length)
                        keep = int((batch['labels'] != -100).sum(-1).max()) + 1
                        logits = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                            use_cache=False, logits_to_keep=keep).logits[:, :-1].float()
                        targets = batch['labels'][:, -(keep-1):]
                        token_lp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                                    ignore_index=-100, reduction='none').reshape_as(targets)
                        scores.extend(token_lp.sum(-1).cpu().tolist())
                scores = [scores[i:i+3] for i in range(0, len(scores), 3)]
                for key, indices in members.items():
                    logq[key] = log_allocation([scores[i] for i in indices], [inputs[i]['gold'] for i in indices],
                                              [inputs[i]['slot_weight'] for i in indices])
                if allocation_enabled:
                    if marginal_enabled:
                        cache=label_marginal_linearization(torch.tensor(scores,device=device),
                            [r['gold'] for r in inputs],[r['slot_weight'] for r in inputs],
                            [(r['group_key'],r['slot']) for r in inputs],[k[1] for k in keys],
                            objective='macro_f1' if cfg['native_mode']=='marginal_f1' else 'accuracy',
                            seed=cfg['seed']+self.state.global_step)
                        self._native_allocation={k:cache[k] for k in
                            ('scores','jacobian','value_share','group_count')}
                        self._label_marginal_rewards=cache['rewards'].cpu().tolist()
                        active_values=cache['rewards'][torch.tensor([r['slot_weight'] for r in inputs],
                                                                  device=device,dtype=torch.bool)]
                        self._metrics['train']['marginal/reward'].append(float(active_values.mean()))
                        self._metrics['train']['marginal/direct_score_gradient_norm'].append(
                            float(cache['jacobian'].norm()))
                    else:
                        self._native_allocation = allocation_linearization(
                            torch.tensor(scores, device=device), members, [r['gold'] for r in inputs],
                            [r['slot_weight'] for r in inputs])
                    prefixes = [r['prompt_ids_native'] + r['prefix_ids'] for r in inputs]
                    padded = _pad_ce([dict(input_ids=p, attention_mask=[1]*len(p), labels=[-100]*len(p))
                                      for p in prefixes], tok, device,min_length=score_length-suffix_length)
                    self._native_allocation['prefix_input_ids'] = padded['input_ids']
                    self._native_allocation['prefix_attention_mask'] = padded['attention_mask']
                    if saliency_enabled:
                        budget = g['reasoning_budget']
                        hidden = self.model.get_input_embeddings().weight.shape[-1]
                        contributions = torch.zeros(len(inputs), 3, budget, hidden, device=device)
                        jacobian = self._native_allocation['jacobian'].reshape(-1)
                        opener = len(tok.encode('<think>', add_special_tokens=False))
                        # Estimate local credit without accumulating parameter gradients.
                        with torch.enable_grad():
                            for at in range(0, len(score_rows), g['score_microbatch']):
                                chunk = score_rows[at:at+g['score_microbatch']]
                                batch = _pad_ce(chunk, tok, device,min_length=score_length)
                                embedded = self.model.get_input_embeddings()(batch['input_ids']).detach().requires_grad_()
                                keep = int((batch['labels'] != -100).sum(-1).max()) + 1
                                logits = self.model(inputs_embeds=embedded, attention_mask=batch['attention_mask'],
                                                    use_cache=False, logits_to_keep=keep).logits[:, :-1].float()
                                targets = batch['labels'][:, -(keep-1):]
                                logps = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                    ignore_index=-100, reduction='none').reshape_as(targets).sum(-1)
                                objective = (logps * jacobian[at:at+len(chunk)]).sum()
                                gradients = torch.autograd.grad(objective, embedded)[0]
                                for offset, row in enumerate(chunk):
                                    sample, label = divmod(at+offset, 3)
                                    length = inputs[sample]['reasoning_tokens']
                                    left_padding = embedded.shape[1] - len(row['input_ids'])
                                    start = left_padding + len(inputs[sample]['prompt_ids_native']) + opener
                                    contributions[sample, label, :length] = (
                                        gradients[offset, start:start+length].float() *
                                        embedded.detach()[offset, start:start+length].float())
                                del gradients, objective, logps, logits, embedded, batch
                        self._native_credit = native_token_credit(contributions,
                            [r['reasoning_tokens'] for r in inputs], [r['slot_weight'] for r in inputs],
                            floor=g['saliency_floor'])
                        values = [self._native_credit[i, :r['reasoning_tokens']] for i, r in enumerate(inputs)
                                  if r['slot_weight'] and r['reasoning_tokens']]
                        if values:
                            values = torch.cat(values)
                            self._metrics['train']['native/credit_std'].append(float(values.std(unbiased=False)))
                            self._metrics['train']['native/credit_max'].append(float(values.max()))
                            self._metrics['train']['native/credit_mean'].append(float(values.mean()))
                            if not (Path(out_dir) / 'credit_smoke.json').exists():
                                write_json(Path(out_dir) / 'credit_smoke.json', dict(mean=float(values.mean()),
                                    std=float(values.std(unbiased=False)), max=float(values.max()), min=float(values.min()),
                                    parameter_gradients_accumulated=any(p.grad is not None for p in self.model.parameters()),
                                    sampled_reasoning_tokens=len(values)))
            reward_logq = logq
            if trace_enabled:
                donors = native_swap_indices(members, [r['slot_weight'] for r in inputs])
                changed = [i for i, donor in enumerate(donors)
                           if inputs[i]['prefix_ids'] != inputs[donor]['prefix_ids']]
                # Identical inputs reuse the original score, giving exactly zero contrast.
                rotated_scores = [list(row) for row in scores]
                rotated_rows = [_answer_row(inputs[i]['prompt_ids_native'] + inputs[donors[i]]['prefix_ids'], lab, tok)
                                for i in changed for lab in LABELS]
                rotated_values = []
                with deterministic_scoring(self.model), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                    for at in range(0, len(rotated_rows), g['score_microbatch']):
                        batch = _pad_ce(rotated_rows[at:at+g['score_microbatch']], tok, device,
                                        min_length=score_length)
                        keep = int((batch['labels'] != -100).sum(-1).max()) + 1
                        logits = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                            use_cache=False, logits_to_keep=keep).logits[:, :-1].float()
                        targets = batch['labels'][:, -(keep-1):]
                        token_lp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                                    ignore_index=-100, reduction='none').reshape_as(targets)
                        rotated_values.extend(token_lp.sum(-1).cpu().tolist())
                for position, index in enumerate(changed):
                    rotated_scores[index] = rotated_values[3*position:3*position+3]
                rotated_logq = {key: log_allocation([rotated_scores[i] for i in indices],
                               [inputs[i]['gold'] for i in indices], [inputs[i]['slot_weight'] for i in indices])
                               for key, indices in members.items()}
                reward_logq = {key: logq[key] - rotated_logq[key] for key in members}
                self._native_trace_logq = dict(original=logq, rotated=rotated_logq)
                gaps = torch.tensor(list(reward_logq.values()), dtype=torch.float32)
                self._metrics['train']['native/trace_gap'].append(float(gaps.mean()))
                self._metrics['train']['native/trace_gap_std'].append(float(gaps.std(unbiased=False)))
                self._metrics['train']['native/rotated_logq'].append(sum(rotated_logq.values()) / len(rotated_logq))
                if not (Path(out_dir) / 'trace_swap_smoke.json').exists():
                    write_json(Path(out_dir) / 'trace_swap_smoke.json', dict(
                        changed_prefix_rows=len(changed), active_rows=sum(r['slot_weight'] for r in inputs),
                        groups=len(members), original_logq_mean=sum(logq.values())/len(logq),
                        rotated_logq_mean=sum(rotated_logq.values())/len(rotated_logq),
                        gap_mean=float(gaps.mean()), gap_std=float(gaps.std(unbiased=False)),
                        finite=bool(torch.isfinite(gaps).all()),
                        donor_rule='label-blind cyclic active slots within the same native group and rollout'))
            if marginal_enabled:
                total=self._label_marginal_rewards
            elif cfg['native_mode'] in ('gated_joint','gated_trace_swap'):
                informative={key:int(len({inputs[i]['gold'] for i in indices if inputs[i]['slot_weight']})>=2)
                             for key,indices in members.items()}
                correct=[int(parse(r['raw_output'],'cot').label==r['gold']) for r in inputs]
                if cfg['native_mode']=='gated_trace_swap':
                    total=bounded_correct_swap_rewards(rewards,[k[0] for k in keys],[k[1] for k in keys],logq,
                        rotated_logq,correct,informative,active=[r['slot_weight'] for r in inputs],coefficient=coef)
                else:
                    total=bounded_correct_allocation_rewards(rewards,[k[0] for k in keys],[k[1] for k in keys],logq,
                        correct,informative,active=[r['slot_weight'] for r in inputs],coefficient=coef)
                self._metrics['train']['native/policy_allocation_bonus'].append(
                    sum(value-base for value,base,row in zip(total,rewards,inputs) if row['slot_weight'])/
                    max(1,sum(r['slot_weight'] for r in inputs)))
            else:
                total = redistribute_group_rewards(rewards, [k[0] for k in keys], [k[1] for k in keys], reward_logq,
                                                   active=[r['slot_weight'] for r in inputs], coefficient=coef)
            self._native_ce = _pad_ce(ce_rows, tok, device)
            self._native_weights = torch.tensor([r['slot_weight'] for r in inputs], device=device, dtype=torch.float32)
            active = self._native_weights.bool()
            self._metrics['train']['native/accuracy'].append(sum(r['slot_weight'] * (parse(r['raw_output'], 'cot').label == r['gold']) for r in inputs) / max(1, int(active.sum())))
            self._metrics['train']['native/logq'].append(sum(logq.values()) / max(1, len(logq)))
            self._metrics['train']['native/active_examples'].append(int(active.sum()) / g['num_generations'])
            self._metrics['train']['native/reasoning_tokens'].append(sum(r['reasoning_tokens'] for r in inputs) / len(inputs))
            # A single scalar reward already has its approved weights applied.
            return torch.tensor(total, device=device).unsqueeze(-1)

        def _generate_and_score_completions(self, inputs):
            self._native_active = [r['slot_weight'] for r in inputs]
            result = super()._generate_and_score_completions(inputs)
            for key, value in self._native_ce.items():
                result['ce_' + key] = value
            result['ce_weights'] = self._native_weights
            result['ce_num_items'] = (self._native_ce['labels'] != -100).sum()
            if self._native_allocation is not None:
                for key, value in self._native_allocation.items():
                    result['allocation_' + key] = value
            # Rollout masks were applied BEFORE TRL's KL and sampling-ratio calculation.
            final_mask = result['completion_mask'] * result['tool_mask']
            result['num_items_in_batch'] = final_mask.sum()
            if marginal_enabled:
                advantages=example_rloo(self._label_marginal_rewards,
                    [(r['group_key'],r['slot']) for r in inputs],[r['slot_weight'] for r in inputs])
                result['advantages']=torch.tensor(advantages,device=self.accelerator.device,dtype=torch.float32)
                active_adv=result['advantages'][self._native_weights.bool()]
                self._metrics['train']['marginal/rloo_abs_mean'].append(float(active_adv.abs().mean()))
                self._metrics['train']['marginal/zero_advantage_fraction'].append(
                    float((active_adv.abs()<1e-12).float().mean()))
            sequence_advantages = result['advantages']
            if self._native_credit is not None:
                result['advantages'] = apply_reasoning_credit(sequence_advantages, self._native_credit,
                    [r['reasoning_tokens'] for r in inputs],
                    opener_length=len(self.processing_class.encode('<think>', add_special_tokens=False)),
                    completion_length=result['completion_mask'].shape[-1])
                result['reasoning_credit'] = self._native_credit
            if (final_mask[~self._native_weights.bool()] != 0).any():
                raise ValueError('Padding contributed policy/KL tokens')
            with (Path(out_dir) / 'rollouts_train.jsonl').open('a', encoding='utf-8') as f:
                for i, row in enumerate(inputs):
                    record = build_record(row, row['raw_output'], 'cot', False, int(result['completion_mask'][i].sum()))
                    record.update(step=self.state.global_step, group_key=row['group_key'], slot=row['slot'],
                                  slot_weight=row['slot_weight'], reasoning_tokens=row['reasoning_tokens'],
                                  advantage=float(sequence_advantages[i]))
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            return result

        def _compute_loss(self, model, inputs):
            policy = super()._compute_loss(model, inputs)
            labels = inputs['ce_labels']
            keep = int((labels != -100).sum(-1).max()) + 1
            keep = max(2, keep)
            logits = model(input_ids=inputs['ce_input_ids'], attention_mask=inputs['ce_attention_mask'],
                           use_cache=False, logits_to_keep=keep).logits[:, :-1]
            targets = labels[:, -(keep-1):]
            ce = weighted_answer_ce(logits, targets, inputs['ce_weights'])
            valid = ((targets != -100) * inputs['ce_weights'].unsqueeze(-1)).sum()
            # Sum across microbatches yields one globally token-normalized CE.
            auxiliary = ce * valid / inputs['ce_num_items'].clamp(min=1)
            total = policy_loss_coefficient(cfg['native_mode'])*policy + g['answer_ce_weight'] * auxiliary
            allocation = None
            replay_difference = None
            if allocation_enabled:
                tok = self.processing_class
                ids = inputs['allocation_prefix_input_ids']
                masks = inputs['allocation_prefix_attention_mask']
                prefixes = [row[mask.bool()].tolist() for row, mask in zip(ids, masks)]
                score_rows = [_answer_row(prefix, label, tok) for prefix in prefixes for label in LABELS]
                score_length = ids.shape[1] + max(len(_answer_row([], label, tok)['input_ids'])
                                                for label in LABELS)
                score_parts = []
                # Keep training mode and checkpointing enabled; dropout is audited as zero.
                for at in range(0, len(score_rows), g['score_microbatch']):
                    batch = _pad_ce(score_rows[at:at+g['score_microbatch']], tok, ids.device,
                                    min_length=score_length)
                    keep = int((batch['labels'] != -100).sum(-1).max()) + 1
                    live_logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                        use_cache=False, logits_to_keep=keep).logits[:, :-1].float()
                    live_targets = batch['labels'][:, -(keep-1):]
                    token_lp = -F.cross_entropy(live_logits.reshape(-1, live_logits.size(-1)),
                        live_targets.reshape(-1), ignore_index=-100, reduction='none').reshape_as(live_targets)
                    score_parts.append(token_lp.sum(-1))
                live_scores = torch.cat(score_parts).reshape(-1, 3)
                allocation = allocation_surrogate(live_scores, inputs['allocation_scores'],
                    inputs['allocation_jacobian'], inputs['allocation_value_share'], inputs['allocation_group_count'])
                replay_difference = (live_scores.detach()-inputs['allocation_scores']).abs()
                active_difference = replay_difference[inputs['ce_weights'].bool()]
                active_max_difference = float(active_difference.max()) if active_difference.numel() else 0.
                total = total + g['allocation_loss_weight'] * allocation
                self._metrics['train']['native/allocation_loss_contribution'].append(float(allocation.detach()))
                self._metrics['train']['native/replay_score_max_difference'].append(float(replay_difference.max()))
                self._metrics['train']['native/replay_active_score_max_difference'].append(active_max_difference)
            if not torch.isfinite(total).item():
                raise FloatingPointError('Nonfinite native RL loss')
            self._metrics['train']['native/answer_ce'].append(float(ce.detach()))
            if not hasattr(self, '_checked_gradients') and valid > 0 and (
                not allocation_enabled or inputs['allocation_jacobian'].abs().sum() > 0):
                params = [p for p in model.parameters() if p.requires_grad]
                pgrads = torch.autograd.grad(policy, params, retain_graph=True, allow_unused=True)
                cgrads = torch.autograd.grad(auxiliary, params, retain_graph=True, allow_unused=True)
                norm = lambda grads: sum(float(x.float().square().sum()) for x in grads if x is not None) ** .5
                audit = dict(policy_norm=norm(pgrads), answer_ce_norm=norm(cgrads),
                             active_tokens=int(inputs['num_items_in_batch']), ce_tokens=int(inputs['ce_num_items']))
                if allocation_enabled:
                    agrads = torch.autograd.grad(allocation, params, retain_graph=True, allow_unused=True)
                    audit.update(allocation_norm=norm(agrads), allocation_score_max_difference=float(replay_difference.max()),
                                 allocation_active_score_max_difference=active_max_difference,
                                 allocation_groups=int(inputs['allocation_group_count']))
                write_json(Path(out_dir) / 'gradient_smoke.json', audit)
                self._checked_gradients = True
            return total

    return NativeTrainer


def train_native(cfg, seed, out_dir):
    import torch
    from datasets import Dataset
    from peft import PeftModel, get_peft_model_state_dict
    from trl import GRPOConfig
    from scripts.run import resolve_initial_adapter
    from train.audit import train_log_callback
    patch_torchao_compatibility()
    patch_vllm_sync_weights_for_qwen35()
    cfg = {**cfg, 'seed': seed}
    out_dir = Path(out_dir)
    g = cfg['native_pilot']
    blocks = build_native_blocks(get_rows('train'), seed=seed)
    scheduled_indices = block_sample_indices(len(blocks), groups_per_batch=g['groups_per_update'],
                                             rollouts=g['num_generations'], seed=seed)
    stride = 3 * g['num_generations']
    group_order = [idx // 3 for idx in scheduled_indices[::stride]]
    if g['max_steps'] > len(group_order) // g['groups_per_update']:
        raise ValueError('Matched pilot is prespecified to at most one epoch')
    grouping=cfg.get('native_grouping','random' if cfg['native_mode']=='random' else 'native')
    if grouping not in ('native','random'):raise ValueError('Unknown native grouping')
    if grouping=='random':
        visited_steps=min(g['max_steps'],g.get('stop_after_steps',g['max_steps']))
        blocks = build_scheduled_random_blocks(blocks, group_order[:visited_steps*g['groups_per_update']],
                                               groups_per_batch=g['groups_per_update'], seed=seed)
    rows = pad_blocks(blocks)
    model, tok = load_model_and_tokenizer(cfg['model']['name'])
    import importlib.metadata
    from common.io import ROOT
    write_json(out_dir / 'environment.json', dict(
        packages={p: importlib.metadata.version(p) for p in ('torch', 'trl', 'transformers', 'peft', 'vllm')},
        gpu=torch.cuda.get_device_name(0), total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        model_revision=getattr(model.config, '_commit_hash', None),
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in ('train/native_grpo.py', 'train/native_reward.py', 'train/native_groups.py', 'train/native_rollout.py',
                                    'train/native_training.py', 'train/native_allocation.py', 'train/native_saliency.py', 'train/native_trace_swap.py',
                                    'train/label_marginal.py',
                                    'formats/parser.py', 'eval/predict.py')},
        seed=seed, decoding='two-stage; inserted boundaries excluded from policy likelihood'))
    model = PeftModel.from_pretrained(model, str(resolve_initial_adapter(cfg, seed)), is_trainable=True)
    data = []
    for row in rows:
        messages = build_messages(row['premise'], row['hypothesis'], 'cot', False)
        prompt_ids = chat_template_ids(tok, messages)
        # Standard string prompts avoid response-schema stripping of think tags.
        data.append({**row, 'gold': row['label'], 'prompt_ids_native': prompt_ids,
                     'prompt': tok.decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)})
    args = GRPOConfig(output_dir=str(out_dir / 'grpo_ckpts'), seed=seed, learning_rate=g['lr'], max_steps=g['max_steps'],
                      beta=g['beta'], loss_type='dapo', num_generations=g['num_generations'], temperature=1., top_p=1., top_k=0,
                      per_device_train_batch_size=g['per_device_batch'], gradient_accumulation_steps=g['grad_accum'],
                      generation_batch_size=g['groups_per_update'] * 3 * g['num_generations'],
                      max_completion_length=g['max_completion_length'], use_vllm=True, vllm_mode='colocate',
                      vllm_gpu_memory_utilization=g['vllm_gpu_memory_utilization'], vllm_max_model_length=2048,
                      reward_weights=[1.], bf16=True, logging_steps=1, save_steps=g['save_steps'],
                      save_strategy='steps', gradient_checkpointing=True,
                      gradient_checkpointing_kwargs={'use_reentrant': False}, report_to=[],
                      disable_dropout=g.get('disable_dropout', False),
                      chat_template_kwargs={'enable_thinking': False}, remove_unused_columns=False)
    cls = make_native_trainer(cfg, out_dir)
    callbacks=[train_log_callback(out_dir)]
    if g.get('stop_after_steps') is not None:
        from train.native_training import stop_after_callback
        callbacks.append(stop_after_callback(g['stop_after_steps']))
    trainer = cls(model=model, args=args, reward_funcs=lambda **kwargs: [],
                  train_dataset=Dataset.from_list(data), processing_class=tok, peft_config=None,
                  rollout_func=rollout_func, callbacks=callbacks)
    trainer.native_group_order = group_order
    dropout_modules={name:module.p for name,module in trainer.model.named_modules()
                     if isinstance(module,torch.nn.Dropout) and module.p>0}
    write_json(out_dir / 'dropout_audit.json', dict(disable_dropout_requested=g.get('disable_dropout',False),
               active_dropout_modules=dropout_modules,
               reward_scoring='detached eval' if g.get('disable_dropout',False) else 'detached train'))
    if g.get('disable_dropout',False) and dropout_modules:
        raise ValueError('Dropout remains active despite deterministic RL configuration')
    if trainer.args.steps_per_generation != g['grad_accum'] or trainer.num_iterations != 1:
        raise ValueError('Pilot requires one generation per complete optimizer update')
    if 'ref' not in trainer.model.peft_config:
        raise ValueError('TRL did not preserve the D3 reference adapter')
    default = get_peft_model_state_dict(trainer.model, adapter_name='default')
    reference = get_peft_model_state_dict(trainer.model, adapter_name='ref')
    if default.keys() != reference.keys() or any(not torch.equal(default[k], reference[k]) for k in default):
        raise ValueError('Reference is not an exact D3 copy')
    if any(p.requires_grad for n, p in trainer.model.named_parameters() if '.ref.' in n):
        raise ValueError('Reference adapter is trainable')
    digest = adapter_digest(trainer.model, 'ref')
    write_json(out_dir / 'reference_audit.json', dict(initial_sha256=digest, init_from=cfg['init_from'], frozen=True))
    write_json(out_dir / 'group_audit.json', dict(groups=len(blocks), active_rows=sum(r['slot_weight'] for r in rows),
               padded_rows=len(rows), mode=cfg['native_mode'], grouping=grouping,group_order=group_order,
               blocks=[[r['id'] for r in b] for b in blocks]))
    try:
        trainer.train()
        trainer.save_model(str(out_dir / 'final_adapter'))
        after = adapter_digest(trainer.model, 'ref')
        write_json(out_dir / 'reference_audit.json', dict(initial_sha256=digest, final_sha256=after,
                   init_from=cfg['init_from'], frozen=True, unchanged=after == digest))
        if after != digest:
            raise ValueError('Reference adapter changed during optimization')
    finally:
        write_train_log(trainer.state.log_history, out_dir / 'train_log.csv')
