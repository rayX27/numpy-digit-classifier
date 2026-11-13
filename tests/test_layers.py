import numpy as np
from config import Config
from nn.layers import Dense, ReLU
from nn.losses import SoftmaxCrossEntropy
from nn.model import Sequential


def make_model(seed=0):
    rng = Config(seed=seed).rngs()["init"]
    return Sequential([Dense(6, 5, rng, dtype=np.float64), ReLU(),
                       Dense(5, 3, rng, dtype=np.float64)])


def test_gradient_check_whole_network():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((8, 6))
    y = rng.integers(0, 3, size=8)
    model = make_model()
    loss_fn = SoftmaxCrossEntropy()

    loss_fn.forward(model.forward(x), y)
    model.backward(loss_fn.backward())

    eps = 1e-6
    for params, grads, name in model.params_and_grads():
        flat = params[name].reshape(-1)  # a view, so edits change the weights
        g = grads[name].reshape(-1)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps
            l_plus = loss_fn.forward(model.forward(x), y)
            flat[i] = old - eps
            l_minus = loss_fn.forward(model.forward(x), y)
            flat[i] = old
            numerical = (l_plus - l_minus) / (2 * eps)
            assert np.isclose(numerical, g[i], rtol=1e-6, atol=1e-8), (name, i)