"""Test bootstrap/McNemar/Holm trên dữ liệu giả.
"""
import numpy as np
import pytest
from analysis.stats import paired_bootstrap, mcnemar, holm


def test_bootstrap_identical():
    a = np.array([1, 0, 1, 1, 0] * 20)
    r = paired_bootstrap(a, a.copy(), n_boot=2000, seed=0)
    assert r["diff"] == 0.0 and r["p"] == 1.0


def test_bootstrap_clear_difference():
    a = np.ones(200)
    b = np.zeros(200)
    r = paired_bootstrap(a, b, n_boot=2000, seed=0)
    assert r["diff"] == 1.0 and r["ci_low"] > 0 and r["p"] < 0.01


def test_bootstrap_deterministic():
    a = np.random.default_rng(1).integers(0, 2, 100)
    b = np.random.default_rng(2).integers(0, 2, 100)
    assert paired_bootstrap(a, b, 1000, 7) == paired_bootstrap(a, b, 1000, 7)


def test_mcnemar():
    a = np.array([1] * 10 + [0] * 10)
    b = np.array([0] * 10 + [0] * 10)
    assert mcnemar(a, b) < 0.01
    assert mcnemar(a, a.copy()) == 1.0
    # symmetric discordance -> p = 1
    assert mcnemar(np.array([1, 0]), np.array([0, 1])) == 1.0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]
    assert holm([0.04, 0.5]) == pytest.approx([0.08, 0.5])
