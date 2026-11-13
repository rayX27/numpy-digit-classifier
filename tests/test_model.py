import numpy as np
import pytest
from sklearn.datasets import load_digits

from config import Config
from nn.layers import Dense, ReLU
from nn.losses import SoftmaxCrossEntropy
from nn.model import Sequential


def build(sizes, seed=0, dtype=np.float32):
    rngs = Config(seed=seed).rngs()
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Dense(sizes[i], sizes[i + 1], rngs["init"], dtype))
        if i < len(sizes) - 2:
            layers.append(ReLU())
    return Sequential(layers)


def digits_batch(n):
    d = load_digits()
    return (d.data[:n] / 16.0).astype(np.float32), d.target[:n]


def sgd_step(model, loss_fn, x, y, lr):
    loss = loss_fn.forward(model.forward(x), y)
    model.backward(loss_fn.backward())
    for params, grads, name in model.params_and_grads():
        params[name] -= lr * grads[name]
    return loss


@pytest.mark.parametrize("sizes", [[8, 4], [8, 6, 4], [8, 6, 5, 4]])
@pytest.mark.parametrize("n", [1, 7, 64])
def test_shapes(sizes, n):
    model = build(sizes)
    x = np.random.default_rng(0).random((n, sizes[0])).astype(np.float32)
    out = model.forward(x)
    assert out.shape == (n, sizes[-1])
    assert model.backward(np.ones_like(out)).shape == (n, sizes[0])


def test_overfit_one_batch():
    x, y = digits_batch(32)
    model, loss_fn = build([64, 64, 10]), SoftmaxCrossEntropy()
    for _ in range(1000):
        loss = sgd_step(model, loss_fn, x, y, lr=0.5)
    assert loss < 0.05


def test_dtypes_stay_float32():
    x, y = digits_batch(16)
    model, loss_fn = build([64, 32, 10]), SoftmaxCrossEntropy()
    logits = model.forward(x)
    assert logits.dtype == np.float32
    loss_fn.forward(logits, y)
    grad = loss_fn.backward()
    assert grad.dtype == np.float32
    assert model.backward(grad).dtype == np.float32
    for params, grads, name in model.params_and_grads():
        assert grads[name].dtype == np.float32
        params[name] -= 0.1 * grads[name]
        assert params[name].dtype == np.float32


def test_extreme_logits_are_finite():
    logits = np.array([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 1000.0]], dtype=np.float32)
    y = np.array([0, 1])
    loss_fn = SoftmaxCrossEntropy()
    loss = loss_fn.forward(logits, y)
    assert np.isfinite(loss)
    assert np.all(np.isfinite(loss_fn.backward()))


def test_initial_loss_is_near_random_guessing():
    x, y = digits_batch(200)
    loss = SoftmaxCrossEntropy().forward(build([64, 32, 10]).forward(x), y)
    assert 1.5 < loss < 4.0  # ln(10) = 2.30; He init starts a bit higher


def test_same_seed_same_result():
    x, y = digits_batch(32)
    results = []
    for _ in range(2):
        model, loss_fn = build([64, 32, 10], seed=5), SoftmaxCrossEntropy()
        for _ in range(20):
            sgd_step(model, loss_fn, x, y, lr=0.5)
        results.append(model.layers[0].params["W"].copy())
    assert np.allclose(results[0], results[1])