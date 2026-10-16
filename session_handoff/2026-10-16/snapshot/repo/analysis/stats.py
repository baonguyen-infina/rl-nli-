"""Paired bootstrap 10k, McNemar, hiệu chỉnh Holm cho các so sánh đăng ký trước.
"""
import numpy as np
from scipy.stats import binomtest


def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "diff": float(d.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "p": float(min(p, 1.0)),
    }


def mcnemar(a, b) -> float:
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n = n10 + n01
    if n == 0:
        return 1.0
    return float(binomtest(min(n10, n01), n, 0.5).pvalue)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, pvals[i] * (m - rank)))
        adj[i] = running
    return adj


def paired_macro_f1_bootstrap(gold, pred_a, pred_b, n_boot=10000, seed=0):
    """Paired percentile CI over examples, with a fixed three-label denominator.

    Invalid predictions remain false negatives for their gold class. Arrays must
    already be aligned by example ID; callers must not compare unrelated rows.
    This interval measures example uncertainty, not variability across seeds.
    """
    from common.labels import LABELS
    if not len(gold) or not (len(gold)==len(pred_a)==len(pred_b)):
        raise ValueError('Nonempty aligned predictions required')
    if n_boot<1:raise ValueError('n_boot must be positive')
    y=np.array([LABELS.index(x) for x in gold])
    a=np.array([LABELS.index(x) if x in LABELS else -1 for x in pred_a])
    b=np.array([LABELS.index(x) if x in LABELS else -1 for x in pred_b])
    def score(indices,pred):
        actual=y[indices];guessed=pred[indices]
        f=[]
        for label in range(3):
            tp=((actual==label)&(guessed==label)).sum(axis=-1)
            denom=(actual==label).sum(axis=-1)+(guessed==label).sum(axis=-1)
            f.append(np.divide(2*tp,denom,out=np.zeros_like(tp,dtype=float),where=denom!=0))
        return np.stack(f).mean(axis=0)
    rng=np.random.default_rng(seed);diffs=[]
    for start in range(0,n_boot,256):
        ix=rng.integers(0,len(y),size=(min(256,n_boot-start),len(y)))
        diffs.extend((score(ix,a)-score(ix,b)).tolist())
    original=np.arange(len(y))[None,:]
    return {'diff':float((score(original,a)-score(original,b))[0]),
            'ci_low':float(np.percentile(diffs,2.5)),
            'ci_high':float(np.percentile(diffs,97.5)),
            'n_boot':n_boot,'seed':seed}


def paired_cluster_macro_f1_bootstrap(gold, pred_a, pred_b, groups,
                                      n_boot=10000, seed=0):
    """Paired percentile interval resampling whole premise clusters.

    Keep all hypotheses from a sampled premise together, including repeated
    copies when that premise is sampled more than once. This quantifies test
    sample uncertainty conditional on the fitted models, not training-seed
    variability. Invalid predictions count as false negatives for the gold
    label. The caller must align every array by example ID before calling.
    """
    from common.labels import LABELS
    if not len(gold) or not (len(gold) == len(pred_a) == len(pred_b) == len(groups)):
        raise ValueError('Nonempty aligned predictions and groups required')
    if type(n_boot) is not int or n_boot < 1:
        raise ValueError('n_boot must be a positive integer')
    if any(label not in LABELS for label in gold):
        raise ValueError('Unknown gold label')
    index = {}
    try:
        for group in groups:
            if group is None or group == '':
                raise ValueError('Missing premise cluster')
            if group not in index:
                index[group] = len(index)
    except TypeError as exc:
        raise ValueError('Premise clusters must be hashable') from exc
    counts = np.zeros((2, len(index), 3, 4), dtype=np.int64)
    for row, label in enumerate(gold):
        cluster, actual = index[groups[row]], LABELS.index(label)
        for method, predictions in enumerate((pred_a, pred_b)):
            prediction = predictions[row]
            guessed = LABELS.index(prediction) if prediction in LABELS else 3
            counts[method, cluster, actual, guessed] += 1

    def macro_f1(confusion):
        tp = np.diagonal(confusion[..., :3], axis1=-2, axis2=-1)
        denominator = confusion.sum(axis=-1) + confusion[..., :3].sum(axis=-2)
        return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float),
                         where=denominator != 0).mean(axis=-1)

    original = macro_f1(counts.sum(axis=1))
    rng = np.random.default_rng(seed)
    diffs = []
    for start in range(0, n_boot, 128):
        draw = rng.integers(0, len(index), size=(min(128, n_boot-start), len(index)))
        scores = macro_f1(counts[:, draw].sum(axis=2))
        diffs.extend((scores[0] - scores[1]).tolist())
    return {'diff': float(original[0] - original[1]),
            'ci_low': float(np.percentile(diffs, 2.5)),
            'ci_high': float(np.percentile(diffs, 97.5)),
            'n_boot': n_boot, 'seed': seed, 'n_clusters': len(index),
            'bootstrap_unit': 'premise_cluster'}
