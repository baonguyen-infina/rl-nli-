"""Replay exact score derivatives of conditional native label allocation."""
from common.labels import LABELS
from train.native_groups import tensor_log_allocation


def allocation_linearization(class_scores,members,gold,weights):
    """Cache unnormalized group Jacobians; never retain a student model graph."""
    import torch
    if class_scores.ndim!=2 or class_scores.shape[1]!=3 or not class_scores.is_floating_point():
        raise ValueError('scores must be floating point [row,3]')
    n=class_scores.shape[0]
    if len(gold)!=n or len(weights)!=n or any(w not in (0,1) for w in weights):
        raise ValueError('gold and binary weights must match score rows')
    groups=list(members.values())
    if any(not 1<=len(indices)<=3 for indices in groups):
        raise ValueError('groups must contain one to three rows')
    flattened=[i for indices in groups for i in indices]
    if any(not isinstance(i,int) for i in flattened) or sorted(flattened)!=list(range(n)):
        raise ValueError('groups must partition every score row exactly once')
    if any(gold[i] not in LABELS for i in range(n) if weights[i]):
        raise ValueError('active gold label is invalid')
    working=class_scores.detach().clone()
    if working.dtype in (torch.float16,torch.bfloat16):working=working.float()
    active=torch.tensor(weights,device=working.device,dtype=torch.bool)
    if not torch.isfinite(working[active]).all().item():
        raise ValueError('active scores must be finite')
    # Inactive nonfinite scores cannot contaminate a zero-gradient replay.
    working[~active]=0.
    jacobian=torch.zeros_like(working)
    shares=working.new_zeros(n)
    with torch.enable_grad():
        for indices in groups:
            leaf=working[indices].clone().requires_grad_()
            labels=torch.tensor([[LABELS.index(gold[i]) if weights[i] else -1 for i in indices]],
                                device=working.device,dtype=torch.long)
            mask=torch.tensor([[weights[i] for i in indices]],device=working.device)
            loss=-tensor_log_allocation(leaf.unsqueeze(0).unsqueeze(0),labels,mask).sum()
            derivative=torch.autograd.grad(loss,leaf)[0]
            jacobian[indices]=derivative.detach()
            count=sum(weights[i] for i in indices)
            if count:
                for i in indices:
                    if weights[i]:shares[i]=loss.detach()/count
    return dict(scores=working,jacobian=jacobian,value_share=shares,
                group_count=torch.tensor(len(groups),device=working.device))


def allocation_surrogate(live_scores,old_scores,jacobian,value_share,group_count):
    """Match value and score derivative at the cached point across arbitrary slices."""
    import torch
    if live_scores.ndim!=2 or live_scores.shape[-1]!=3 or not live_scores.is_floating_point():
        raise ValueError('live scores must be floating point [row,3]')
    if old_scores.shape!=live_scores.shape or jacobian.shape!=live_scores.shape or value_share.shape!=(len(live_scores),):
        raise ValueError('replay fields must have matching row shapes')
    if group_count.ndim!=0 or not torch.isfinite(group_count).item() or group_count.item()<0:
        raise ValueError('group count must be a finite nonnegative scalar')
    jac=jacobian.detach()
    # Mask BEFORE multiplication so ignored NaN/Inf cannot produce 0*NaN.
    participates=jac!=0
    delta=torch.where(participates,live_scores,torch.zeros_like(live_scores))-torch.where(
        participates,old_scores.detach(),torch.zeros_like(old_scores))
    return ((jac*delta).sum()+value_share.detach().sum())/group_count.detach().clamp(min=1)
