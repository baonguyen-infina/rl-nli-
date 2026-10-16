"""Teacher-free finite-label utility and detached score-gradient replay.

Macro-F1 utility uses only training labels and independently sampled canonical
answers from the current student. It is a minibatch objective, not corpus F1.
"""
import math
from collections import defaultdict
from common.labels import LABELS


def _macro_f1(counts):
    value=0.
    for c in range(3):
        denominator=sum(counts[c])+sum(row[c] for row in counts)
        if denominator:value+=2*counts[c][c]/denominator
    return value/3


def counterfactual_utilities(gold,predictions,active,rollout_ids):
    """B times the F1 change from adding each possible answer to the other rows."""
    n=len(gold)
    if any(len(v)!=n for v in (predictions,active,rollout_ids)):
        raise ValueError('Mismatched counterfactual metadata')
    if any(w not in (0,1) for w in active):raise ValueError('Active mask must be binary')
    batches=defaultdict(list)
    for i,use in enumerate(active):
        if not use:continue
        if gold[i] not in range(3) or predictions[i] not in range(3):
            raise ValueError('Active labels must be canonical class indices')
        batches[rollout_ids[i]].append(i)
    utilities=[[0.,0.,0.] for _ in range(n)]
    for indices in batches.values():
        counts=[[0]*3 for _ in range(3)]
        for i in indices:counts[gold[i]][predictions[i]]+=1
        for i in indices:
            y,p=gold[i],predictions[i]
            counts[y][p]-=1
            baseline=_macro_f1(counts)
            for c in range(3):
                counts[y][c]+=1
                utilities[i][c]=len(indices)*(_macro_f1(counts)-baseline)
                counts[y][c]-=1
            counts[y][p]+=1
    return utilities


def example_rloo(rewards,example_ids,active):
    """Leave out the current reasoning sample; never standardize by its variance."""
    if not (len(rewards)==len(example_ids)==len(active)):
        raise ValueError('Mismatched RLOO metadata')
    if any(w not in (0,1) for w in active):raise ValueError('Active mask must be binary')
    groups=defaultdict(list)
    for i,use in enumerate(active):
        if use:
            if not math.isfinite(rewards[i]):raise ValueError('Active rewards must be finite')
            groups[example_ids[i]].append(i)
    advantages=[0.]*len(rewards)
    for indices in groups.values():
        if len(indices)<2:raise ValueError('RLOO needs at least two reasoning samples per example')
        total=sum(rewards[i] for i in indices)
        for i in indices:advantages[i]=rewards[i]-(total-rewards[i])/(len(indices)-1)
    return advantages


def label_marginal_linearization(class_scores,gold,weights,example_ids,rollout_ids,*,objective,seed):
    """Cache -E_q[utility] and its exact derivative with respect to label scores."""
    import torch
    if class_scores.ndim!=2 or class_scores.shape[-1]!=3 or not class_scores.is_floating_point():
        raise ValueError('Scores must be floating point [row,3]')
    n=len(class_scores)
    if any(len(v)!=n for v in (gold,weights,example_ids,rollout_ids)):
        raise ValueError('Mismatched label-marginal metadata')
    if any(w not in (0,1) for w in weights):raise ValueError('Weights must be binary')
    if objective not in ('accuracy','macro_f1'):raise ValueError('Unknown label utility')
    if type(seed) is not int or seed<0:raise ValueError('Sampling seed must be a nonnegative integer')
    slots=set();labels_by_example={}
    active=torch.tensor(weights,device=class_scores.device,dtype=torch.bool)
    labels=[]
    for i,use in enumerate(weights):
        if not use:labels.append(-1);continue
        if gold[i] not in LABELS:raise ValueError('Invalid active gold label')
        key=(example_ids[i],rollout_ids[i])
        if key in slots:raise ValueError('Duplicate example rollout slot')
        slots.add(key)
        if example_ids[i] in labels_by_example and labels_by_example[example_ids[i]]!=gold[i]:
            raise ValueError('Gold label changed across reasoning samples')
        labels_by_example[example_ids[i]]=gold[i]
        labels.append(LABELS.index(gold[i]))
    working=class_scores.detach().clone()
    if working.dtype in (torch.float16,torch.bfloat16):working=working.float()
    if not torch.isfinite(working[active]).all().item():raise ValueError('Active scores must be finite')
    working[~active]=0.
    probabilities=working.softmax(-1)
    sampled=torch.full((n,),-1,device=working.device,dtype=torch.long)
    if active.any():
        generator=torch.Generator(device='cpu').manual_seed(seed)
        sampled[active]=torch.multinomial(probabilities[active].double().cpu(),1,
                                        generator=generator).squeeze(-1).to(working.device)
    if objective=='accuracy':
        utilities=[[float(c==y) if use else 0. for c in range(3)] for y,use in zip(labels,weights)]
    else:
        utilities=counterfactual_utilities(labels,sampled.cpu().tolist(),weights,rollout_ids)
    utilities=torch.tensor(utilities,device=working.device,dtype=working.dtype).reshape(n,3)
    with torch.enable_grad():
        leaf=working.clone().requires_grad_()
        reward=(leaf.softmax(-1)*utilities).sum(-1)
        losses=-reward
        jacobian=torch.autograd.grad(losses.sum(),leaf)[0].detach()
    return dict(scores=working,jacobian=jacobian,value_share=losses.detach(),
                group_count=active.sum().detach(),utilities=utilities,
                rewards=reward.detach(),sampled_labels=sampled.detach())
