"""Bounded two-stage reasoning; inserted syntax is masked out of policy loss.

Inference uses the identical two-stage procedure with greedy sampling. Neither
stage receives group membership, gold labels or UID metadata.
"""
import random


def block_sample_indices(n_groups, *, slots=3, rollouts=4, groups_per_batch=8, seed=0, repeat_count=1):
    groups = list(range(n_groups))
    random.Random(seed).shuffle(groups)
    groups = groups[:n_groups // groups_per_batch * groups_per_batch]
    indices = []
    for at in range(0, len(groups), groups_per_batch):
        batch = [g * slots + slot for g in groups[at:at+groups_per_batch]
                 for slot in range(slots) for _ in range(rollouts)]
        indices.extend(batch * repeat_count)
    return indices


def assemble_rollout(opener, reasoning, reasoning_logps, bridge, answer, answer_logps, ending,
                     *, reasoning_budget=64, total_budget=96):
    if len(reasoning) > reasoning_budget or len(reasoning) + len(answer) + len(opener) + len(bridge) + len(ending) > total_budget:
        raise ValueError('Rollout exceeds prespecified token budget')
    if len(reasoning) != len(reasoning_logps) or len(answer) != len(answer_logps):
        raise ValueError('Missing sampled-token likelihoods')
    prefix = opener + reasoning + bridge
    ids = prefix + answer + ending
    return dict(completion_ids=ids, prefix_ids=prefix,
                env_mask=[0] * len(opener) + [1] * len(reasoning) + [0] * len(bridge) + [1] * len(answer) + [0] * len(ending),
                logprobs=[0.] * len(opener) + reasoning_logps + [0.] * len(bridge) + answer_logps + [0.] * len(ending))


def bounded_generate(llm, prompt_ids, tok, *, temperature=1., reasoning_budget=64,
                     total_budget=96, lora_request=None):
    from vllm import SamplingParams
    opener = tok.encode('<think>', add_special_tokens=False)
    bridge = tok.encode('</think><answer>', add_special_tokens=False)
    ending = tok.encode('</answer>', add_special_tokens=False) + [tok.eos_token_id]
    def sampled(x, stops):
        o = x.outputs[0]
        ids = list(o.token_ids)
        ps = [float(entry[token].logprob) for entry, token in zip(o.logprobs, ids, strict=True)]
        cut = len(ids)
        # vLLM removes stop strings from text but retains their sampled IDs.
        for sequence in [[tok.eos_token_id]] + [tok.encode(s, add_special_tokens=False) for s in stops]:
            for at in range(len(ids) - len(sequence) + 1):
                if ids[at:at+len(sequence)] == sequence:
                    cut = min(cut, at)
                    break
        # A marker can cross token boundaries or share a token with trailing
        # text (e.g. '>_'); censor at the start of the decoded stop marker.
        decoded = tok.decode(ids[:cut], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        positions = [decoded.find(s) for s in stops if s in decoded]
        if positions:
            boundary = min(positions)
            for at in range(cut):
                prefix = tok.decode(ids[:at+1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                if len(prefix) > boundary:
                    cut = at
                    break
        return ids[:cut], ps[:cut]
    first = llm.generate([{'prompt_token_ids': p + opener} for p in prompt_ids],
                         SamplingParams(temperature=temperature, top_p=1., top_k=-1,
                                        max_tokens=reasoning_budget, stop=['</think>', '<answer>'],
                                        bad_words=['<think>', '</answer>'],
                                        logprobs=0, skip_special_tokens=False),
                         lora_request=lora_request, use_tqdm=False)
    reason_samples = [sampled(x, ['</think>', '<answer>']) for x in first]
    reasons = [ids for ids, ps in reason_samples]
    second_prompts = [p + opener + reason + bridge for p, reason in zip(prompt_ids, reasons)]
    params = [SamplingParams(temperature=temperature, top_p=1., top_k=-1,
                             max_tokens=total_budget - len(opener) - len(r) - len(bridge) - len(ending),
                             stop=['</answer>'], bad_words=['<think>', '</think>', '<answer>'],
                             logprobs=0, skip_special_tokens=False)
              for r in reasons]
    second = llm.generate([{'prompt_token_ids': p} for p in second_prompts], params,
                          lora_request=lora_request, use_tqdm=False)
    rows = []
    for p, (reason, rps), a in zip(prompt_ids, reason_samples, second, strict=True):
        answer, aps = sampled(a, ['</answer>'])
        row = assemble_rollout(opener, reason, rps, bridge, answer, aps, ending,
                               reasoning_budget=reasoning_budget, total_budget=total_budget)
        row['prompt_ids'] = p
        row['raw_output'] = tok.decode(row['completion_ids'][:-1], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
        row['reasoning_tokens'] = len(reason)
        rows.append(row)
    return rows


def rollout_func(prompts, trainer):
    from train.common import chat_template_ids
    pids = [chat_template_ids(trainer.processing_class, p) if not isinstance(p, str)
            else trainer.processing_class.encode(p, add_special_tokens=False) for p in prompts]
    g = trainer.native_cfg
    rows = bounded_generate(trainer.vllm_generation.llm, pids, trainer.processing_class,
                            temperature=trainer.temperature, reasoning_budget=g['reasoning_budget'],
                            total_budget=g['max_completion_length'])
    for row, active in zip(rows, trainer._native_active, strict=True):
        row['env_mask'] = [value * active for value in row['env_mask']]
        if getattr(trainer,'label_marginal_enabled',False):
            start=len(trainer.processing_class.encode('<think>',add_special_tokens=False))
            end=start+row['reasoning_tokens']
            row['env_mask']=[value if start<=i<end else 0 for i,value in enumerate(row['env_mask'])]
    return {key: [row[key] for row in rows] for key in rows[0]}
