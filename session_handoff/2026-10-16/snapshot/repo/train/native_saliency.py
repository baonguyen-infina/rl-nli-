"""Teacher-free reasoning credit from native score gradient activations."""
import math


def native_token_credit(label_vectors,lengths,active,*,floor=.25):
    """Sum label vectors BEFORE magnitude; normalize each active reasoning span."""
    import torch
    if label_vectors.ndim!=4 or label_vectors.shape[1]!=3 or not label_vectors.is_floating_point():
        raise ValueError('attribution vectors must be floating point [row,3,token,hidden]')
    rows,_,tokens,_=label_vectors.shape
    if not math.isfinite(floor) or not 0<=floor<=1:
        raise ValueError('credit floor must be finite in [0,1]')
    if len(lengths)!=rows or len(active)!=rows:
        raise ValueError('lengths and active weights must match rows')
    if any(type(n) is not int or not 0<=n<=tokens for n in lengths) or any(w not in (0,1) for w in active):
        raise ValueError('reasoning lengths and binary active weights are invalid')
    vectors=label_vectors.detach().float()
    result=vectors.new_zeros(rows,tokens)
    for row,(length,use) in enumerate(zip(lengths,active)):
        if not use or length==0:continue
        subset=vectors[row,:,:length]
        if not torch.isfinite(subset).all().item():raise ValueError('active attribution is nonfinite')
        peak=subset.abs().amax()
        subset=subset/peak if peak>0 else subset
        strength=subset.sum(0).abs().sum(-1)
        total=strength.sum()
        result[row,:length]=floor+(1-floor)*length*strength/total if total>0 else 1.
    return result


def apply_reasoning_credit(advantages,credit,lengths,*,opener_length,completion_length):
    """Preserve answer credit and insert detached reasoning weights in exact spans."""
    import torch
    if advantages.ndim!=1 or credit.ndim!=2 or len(advantages)!=len(credit) or len(lengths)!=len(credit):
        raise ValueError('scalar sequence advantages and reasoning credit rows must match')
    if type(opener_length) is not int or opener_length<0 or completion_length<opener_length:
        raise ValueError('invalid completion boundaries')
    if any(type(n) is not int or not 0<=n<=credit.shape[1] or opener_length+n>completion_length for n in lengths):
        raise ValueError('reasoning span exceeds completion')
    weights=torch.ones((len(advantages),completion_length),device=advantages.device,dtype=advantages.dtype)
    for row,length in enumerate(lengths):
        weights[row,opener_length:opener_length+length]=credit[row,:length].detach()
    return advantages.detach().unsqueeze(-1)*weights
