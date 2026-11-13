import numpy as np
import pytest

from data.loader import load_train_val

pytestmark = pytest.mark.slow


def test_shapes_and_range():
    xt, yt, xv, yv = load_train_val()
    assert xt.shape == (50000, 784) and xv.shape == (10000, 784)
    assert yt.shape == (50000,) and yv.shape == (10000,)
    assert xt.dtype == np.float32
    assert 0.0 <= xt.min() and xt.max() <= 1.0


def test_class_balance():
    _, yt, _, yv = load_train_val()
    for y in (yt, yv):
        share = np.bincount(y, minlength=10) / len(y)
        assert share.min() > 0.07 and share.max() < 0.13  # each class is about 10%


def test_no_overlap_between_train_and_val():
    xt, _, xv, _ = load_train_val()
    train_hashes = {row.tobytes() for row in xt}
    assert not any(row.tobytes() in train_hashes for row in xv)