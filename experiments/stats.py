"""Statistics helpers: seed spread (training randomness) and bootstrap / McNemar (sample noise)."""
from math import erfc, sqrt

import numpy as np


def seed_mean_std(values):
    """Mean and sample standard deviation (ddof=1) across seeds."""
    v = np.asarray(values, dtype=np.float64)
    std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return float(v.mean()), std


def _as_2d(correct):
    return np.atleast_2d(np.asarray(correct, dtype=np.float64))


def bootstrap_ci(correct, n_boot=2000, rng=None, alpha=0.05):
    """Bootstrap CI of accuracy from per-sample correctness flags.

    correct: shape (n_samples,) or (n_seeds, n_samples). Each resample of sample indices is applied
    to every seed; the accuracy is averaged over seeds. Measures finite-evaluation-set noise only.
    """
    c = _as_2d(correct)
    rng = rng if rng is not None else np.random.default_rng(0)
    n = c.shape[1]
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = c[:, rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(correct_a, correct_b, n_boot=2000, rng=None, alpha=0.05):
    """Mean accuracy difference (a - b) and its paired bootstrap CI (same resample for both)."""
    a, b = _as_2d(correct_a), _as_2d(correct_b)
    assert a.shape == b.shape
    rng = rng if rng is not None else np.random.default_rng(0)
    n = a.shape[1]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[:, idx].mean() - b[:, idx].mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def mcnemar(correct_a, correct_b):
    """McNemar's test (continuity corrected) for two classifiers on the same samples.

    Returns (a_right_b_wrong, a_wrong_b_right, p_value).
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n_ab = int((a & ~b).sum())
    n_ba = int((~a & b).sum())
    n = n_ab + n_ba
    if n == 0:
        return n_ab, n_ba, 1.0
    chi2 = (abs(n_ab - n_ba) - 1) ** 2 / n
    return n_ab, n_ba, float(erfc(sqrt(chi2 / 2)))  # chi-square, 1 degree of freedom