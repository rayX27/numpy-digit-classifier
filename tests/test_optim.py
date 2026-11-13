import numpy as np
import pytest

from nn.optim import SGD, SGDMomentum, Adam

W_STAR = np.array([1.0, -2.0, 3.0], dtype=np.float32)
CURVATURE = np.array([1.0, 2.0, 5.0], dtype=np.float32)


class QuadraticModel:
    """f(w) = 0.5 * sum(a * (w - w_star)^2), minimum at w_star."""

    def __init__(self, dtype=np.float32):
        self.params = {"w": np.zeros(3, dtype=dtype)}
        self.grads = {"w": np.zeros(3, dtype=dtype)}

    def compute_grad(self):
        self.grads["w"] = (CURVATURE * (self.params["w"] - W_STAR)).astype(self.params["w"].dtype)

    def params_and_grads(self):
        yield self.params, self.grads, "w"


def run(opt, model, steps):
    for _ in range(steps):
        model.compute_grad()
        opt.step()


@pytest.mark.parametrize("make,steps,tol", [
    (lambda m: SGD(m, 0.1), 300, 1e-3),
    (lambda m: SGDMomentum(m, 0.05, 0.9), 400, 1e-3),
    (lambda m: Adam(m, 0.05), 1500, 1e-2),
])
def test_converges_to_known_minimum(make, steps, tol):
    model = QuadraticModel()
    run(make(model), model, steps)
    assert np.allclose(model.params["w"], W_STAR, atol=tol)


@pytest.mark.parametrize("make", [
    lambda m: SGD(m, 0.1),
    lambda m: SGDMomentum(m, 0.05),
    lambda m: Adam(m, 0.05),
])
def test_parameters_and_state_stay_float32(make):
    model = QuadraticModel(np.float32)
    opt = make(model)
    run(opt, model, 5)
    assert model.params["w"].dtype == np.float32
    for state in (getattr(opt, "v", None), getattr(opt, "m", None)):
        if state is not None:
            assert all(s.dtype == np.float32 for s in state)


def test_lr_attribute_can_be_changed_between_steps():
    model = QuadraticModel()
    opt = SGD(model, 0.1)
    model.compute_grad()
    opt.lr = 0.0
    before = model.params["w"].copy()
    opt.step()
    assert np.array_equal(before, model.params["w"])