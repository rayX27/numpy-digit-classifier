import numpy as np


class Layer:
    """Interface: forward(x, train=True), backward(grad_out), params and grads dicts."""

    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, x, train=True):
        raise NotImplementedError

    def backward(self, grad_out):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_in, n_out, rng, dtype=np.float32):
        super().__init__()
        W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)  # He init
        self.params["W"] = W.astype(dtype)
        self.params["b"] = np.zeros(n_out, dtype=dtype)
        self.x = None

    def forward(self, x, train=True):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out):
        self.grads["W"] = self.x.T @ grad_out
        self.grads["b"] = grad_out.sum(axis=0)
        return grad_out @ self.params["W"].T


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x, train=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask


class Normalize(Layer):
    """Fixed input standardization. Has no parameters, so optimizers never touch it."""

    def __init__(self, mean, std):
        super().__init__()
        self.mean = float(mean)  # Python floats keep float32 inputs in float32
        self.std = float(std)

    def forward(self, x, train=True):
        return (x - self.mean) / self.std

    def backward(self, grad_out):
        return grad_out / self.std