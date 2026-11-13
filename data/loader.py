"""Load cached MNIST, verify checksums, and split 50k train / 10k validation."""
import hashlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "mnist.npz"
CHECKSUMS = HERE / "checksums.txt"
SPLIT_SEED = 1234  # fixed, so the 50k/10k split never changes


def _hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _load_verified():
    data = np.load(CACHE)
    expected = dict(line.split() for line in CHECKSUMS.read_text().splitlines())
    for name in data.files:
        if _hash(data[name]) != expected[name]:
            raise RuntimeError(f"Checksum mismatch for {name}: cache is corrupted or changed")
    return data


def load_train_val():
    data = _load_verified()
    x = data["x_train"].astype(np.float32) / 255.0
    y = data["y_train"].astype(np.int64)
    idx = np.random.default_rng(SPLIT_SEED).permutation(len(x))
    tr, va = idx[:50000], idx[50000:]
    return x[tr], y[tr], x[va], y[va]


def load_test():
    """Official test set. Only final_eval.py may call this."""
    data = _load_verified()
    return data["x_test"].astype(np.float32) / 255.0, data["y_test"].astype(np.int64)


def train_stats(x_train):
    """Global mean and std of the training pixels (scalars, not per-pixel)."""
    return float(x_train.mean()), float(x_train.std())