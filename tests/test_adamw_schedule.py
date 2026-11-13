import math

import numpy as np

from nn.optim import Adam, AdamW
from nn.schedulers import constant, cosine


class TwoParamModel:
    """One weight matrix 'W' (2-D) and one bias 'b' (1-D), zero gradients."""

    def __init__(self):
        self.params = {"W": np.ones((3, 2), dtype=np.float32), "b": np.ones(2, dtype=np.float32)}
        self.grads = {"W": np.zeros((3, 2), dtype=np.float32), "b": np.zeros(2, dtype=np.float32)}

    def params_and_grads(self):
        for name in self.params:
            yield self.params, self.grads, name


def test_weights_decay_but_biases_do_not():
    model = TwoParamModel()
    opt = AdamW(model, lr=0.1, weight_decay=0.5)
    for _ in range(10):
        opt.step()
    assert np.all(model.params["W"] < 1.0)
    assert np.array_equal(model.params["b"], np.ones(2, dtype=np.float32))
    assert model.params["W"].dtype == np.float32


def test_zero_weight_decay_equals_adam():
    a, b = TwoParamModel(), TwoParamModel()
    rng = np.random.default_rng(0)
    oa, ob = Adam(a, 0.01), AdamW(b, 0.01, weight_decay=0.0)
    for _ in range(5):
        g = {k: rng.standard_normal(v.shape).astype(np.float32) for k, v in a.params.items()}
        for k in g:
            a.grads[k], b.grads[k] = g[k].copy(), g[k].copy()
        oa.step()
        ob.step()
    for k in a.params:
        assert np.allclose(a.params[k], b.params[k])


def test_cosine_schedule_endpoints_and_shape():
    f = cosine(0.1, 30)
    assert math.isclose(f(0), 0.1)
    assert math.isclose(f(15), 0.05)
    assert abs(f(30)) < 1e-12
    values = [f(e) for e in range(31)]
    assert all(x >= y for x, y in zip(values, values[1:]))


def test_constant_schedule():
    f = constant(0.3)
    assert f(0) == f(17) == 0.3