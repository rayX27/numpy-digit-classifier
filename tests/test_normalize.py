import numpy as np

from config import Config
from nn.layers import Dense, ReLU, Normalize
from nn.losses import SoftmaxCrossEntropy
from nn.model import Sequential


def test_normalize_output_stats_and_no_params():
    x = np.random.default_rng(0).random((1000, 20)).astype(np.float32)
    layer = Normalize(x.mean(), x.std())
    out = layer.forward(x)
    assert abs(out.mean()) < 1e-3
    assert abs(out.std() - 1.0) < 1e-3
    assert out.dtype == np.float32
    assert layer.params == {}


def test_gradient_check_with_normalize_including_input():
    rng = np.random.default_rng(0)
    x = rng.random((8, 6))
    y = rng.integers(0, 3, size=8)
    init = Config(seed=0).rngs()["init"]
    model = Sequential([Normalize(0.3, 0.7), Dense(6, 5, init, np.float64), ReLU(),
                        Dense(5, 3, init, np.float64)])
    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(model.forward(x), y)
    dx = model.backward(loss_fn.backward())

    eps = 1e-6
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xp, xm = x.copy(), x.copy()
            xp[i, j] += eps
            xm[i, j] -= eps
            numerical = (loss_fn.forward(model.forward(xp), y)
                         - loss_fn.forward(model.forward(xm), y)) / (2 * eps)
            assert np.isclose(numerical, dx[i, j], rtol=1e-6, atol=1e-8), (i, j)