"""Bounded correctness-gated native assignment bonus for sampled reasoning."""
import math


def bounded_correct_allocation_rewards(base_rewards,group_ids,rollout_ids,group_logq,
                                       correct,informative,*,active=None,coefficient=.2):
    n=len(base_rewards)
    active=[True]*n if active is None else active
    if any(len(values)!=n for values in (group_ids,rollout_ids,correct,active)):
        raise ValueError('Mismatched reward metadata')
    if not math.isfinite(coefficient) or coefficient<0:
        raise ValueError('Coefficient must be finite and nonnegative')
    if any(x not in (0,1) for values in (correct,active) for x in values):
        raise ValueError('Correctness and active gates must be binary')
    if any(not math.isfinite(x) for x in base_rewards):
        raise ValueError('Base rewards must be finite, including padding')
    result=[]
    for base,g,k,acc,use in zip(base_rewards,group_ids,rollout_ids,correct,active):
        if not use:
            result.append(0.);continue
        key=(g,k)
        if key not in informative or informative[key] not in (0,1):
            raise ValueError('Missing or nonbinary informative group key')
        if not informative[key]:
            result.append(float(base));continue
        if key not in group_logq:raise ValueError('Missing informative logQ key')
        value=group_logq[key]
        if not math.isfinite(value) or value>1e-6:
            raise ValueError('Informative logQ must be finite and nonpositive')
        bonus=coefficient*acc*math.exp(min(0.,value))
        result.append(float(base+bonus))
    return result


def bounded_correct_swap_rewards(base_rewards,group_ids,rollout_ids,group_logq,
                                 swapped_logq,correct,informative,*,active=None,coefficient=.2):
    """Bound swap credit by subtracting gated counterfactual assignment probability.

    This changes only the policy reward. It does not gate the auxiliary direct
    allocation objective or imply a bound on normalized advantages/gradients.
    """
    original=bounded_correct_allocation_rewards(base_rewards,group_ids,rollout_ids,group_logq,
        correct,informative,active=active,coefficient=coefficient)
    counterfactual=bounded_correct_allocation_rewards([0.]*len(base_rewards),group_ids,rollout_ids,swapped_logq,
        correct,informative,active=active,coefficient=coefficient)
    return [own-swapped for own,swapped in zip(original,counterfactual)]


def policy_loss_coefficient(mode):
    if mode not in ('ordinary','ce','native','random','allocation','joint','saliency',
                    'trace_swap','trace_saliency','gated_joint','gated_trace_swap',
                    'allocation_only','marginal_accuracy','marginal_f1'):
        raise ValueError('Unknown native mode')
    return 0. if mode in ('ce','allocation_only') else 1.
