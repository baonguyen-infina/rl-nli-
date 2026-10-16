import numpy as np
from data.hyp_probe import fold_indices


def test_fold_indices_partition():
    folds = fold_indices(103, 5, seed=0)
    assert len(folds) == 5
    all_val = np.concatenate([v for _, v in folds])
    assert sorted(all_val.tolist()) == list(range(103))
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va)) and len(tr) + len(va) == 103


def test_fold_indices_deterministic():
    a = fold_indices(50, 5, seed=3)
    b = fold_indices(50, 5, seed=3)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
