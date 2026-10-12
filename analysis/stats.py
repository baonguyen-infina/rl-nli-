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
