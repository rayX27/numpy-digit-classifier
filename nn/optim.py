"""Optimizers. Each takes a model exposing params_and_grads() and updates parameters in place."""
import numpy as np


class Optimizer:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def step(self):
        for params, grads, name in self.model.params_and_grads():
            params[name] -= self.lr * grads[name]


class SGDMomentum(Optimizer):
    def __init__(self, model, lr, beta=0.9):
        super().__init__(model, lr)
        self.beta = beta
        self.v = None

    def step(self):
        items = list(self.model.params_and_grads())
        if self.v is None:
            self.v = [np.zeros_like(p[n]) for p, _, n in items]
        for v, (params, grads, name) in zip(self.v, items):
            v *= self.beta
            v += grads[name]
            params[name] -= self.lr * v


class Adam(Optimizer):
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(model, lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self):
        items = list(self.model.params_and_grads())
        if self.m is None:
            self.m = [np.zeros_like(p[n]) for p, _, n in items]
            self.v = [np.zeros_like(p[n]) for p, _, n in items]
        self.t += 1
        c1 = 1.0 - self.beta1 ** self.t   # bias corrections
        c2 = 1.0 - self.beta2 ** self.t
        for m, v, (params, grads, name) in zip(self.m, self.v, items):
            g = grads[name]
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            params[name] -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)



class AdamW(Adam):
    """Adam with decoupled weight decay. Only weight matrices are decayed (ndim >= 2);
    biases (1-D) are excluded."""

    def __init__(self, model, lr, weight_decay=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(model, lr, beta1, beta2, eps)
        self.weight_decay = weight_decay

    def step(self):
        if self.weight_decay:
            for params, _, name in self.model.params_and_grads():
                if params[name].ndim >= 2:
                    params[name] *= 1.0 - self.lr * self.weight_decay
        super().step()