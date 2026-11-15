import numpy as np

from experiments.stats import bootstrap_ci, mcnemar, paired_bootstrap_diff, seed_mean_std


def test_seed_mean_std_known_values():
    mean, std = seed_mean_std([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert abs(std - 1.0) < 1e-12
    assert seed_mean_std([5.0]) == (5.0, 0.0)


def test_bootstrap_ci_all_correct_has_zero_width():
    lo, hi = bootstrap_ci(np.ones((3, 500), dtype=bool), n_boot=200)
    assert lo == hi == 1.0


def test_bootstrap_ci_contains_true_accuracy_and_has_expected_width():
    rng = np.random.default_rng(1)
    flags = rng.random(10000) < 0.9
    lo, hi = bootstrap_ci(flags, n_boot=1000, rng=np.random.default_rng(2))
    assert lo < 0.9 < hi
    assert 0.008 < hi - lo < 0.016  # theory: about 2 * 1.96 * sqrt(0.09 / 10000) = 0.0118


def test_paired_bootstrap_identical_classifiers():
    flags = np.random.default_rng(0).random(2000) < 0.8
    diff, lo, hi = paired_bootstrap_diff(flags, flags, n_boot=200)
    assert diff == lo == hi == 0.0


def test_paired_bootstrap_clear_difference_excludes_zero():
    rng = np.random.default_rng(0)
    a = rng.random(5000) < 0.95
    b = rng.random(5000) < 0.90
    diff, lo, hi = paired_bootstrap_diff(a, b, n_boot=500)
    assert lo > 0.0 and diff > 0.03


def test_mcnemar_hand_computed_table():
    a = np.array([True] * 30 + [False] * 10 + [True] * 100)   # a right, b wrong: 30; a wrong, b right: 10
    b = np.array([False] * 30 + [True] * 10 + [True] * 100)
    n_ab, n_ba, p = mcnemar(a, b)
    assert (n_ab, n_ba) == (30, 10)
    # chi2 = (|30-10|-1)^2 / 40 = 9.025, p = erfc(sqrt(9.025 / 2)) = 0.00266
    assert abs(p - 0.00266) < 2e-4


def test_mcnemar_no_disagreements():
    f = np.array([True, False, True])
    assert mcnemar(f, f) == (0, 0, 1.0)