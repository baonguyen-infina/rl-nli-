"""Label-blind rotation of student prefixes within native rollout groups."""


def native_swap_indices(members, active):
    """Return one cyclic donor per active row; keep singleton/padding rows fixed.

    The complete row partition is supplied by the audited native sampler. This
    function deliberately accepts no label, UID, topic or prompt metadata.
    """
    n=len(active)
    if any(weight not in (0,1) for weight in active):
        raise ValueError('Active row weights must be binary')
    seen=set();mapping=list(range(n))
    for indices in members.values():
        if not 1<=len(indices)<=3:
            raise ValueError('Native groups require one to three rows')
        for index in indices:
            if type(index) is not int or not 0<=index<n or index in seen:
                raise ValueError('Membership must partition row indices exactly once')
            seen.add(index)
        selected=[index for index in indices if active[index]]
        if len(selected)>1:
            for position,index in enumerate(selected):
                mapping[index]=selected[(position+1)%len(selected)]
    if seen!=set(range(n)):
        raise ValueError('Membership does not cover all rows')
    return mapping
