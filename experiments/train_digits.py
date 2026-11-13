"""Train 64-32-10 on 8x8 digits with the new nn/ code and log one result row."""
import time

import numpy as np
from sklearn.datasets import load_digits

from config import Config
from experiments.logger import log_result
from nn.layers import Dense, ReLU
from nn.losses import SoftmaxCrossEntropy
from nn.model import Sequential


def main():
    start = time.perf_counter()
    cfg = Config(seed=42, learning_rate=0.5, batch_size=1437, epochs=300, hidden_sizes=(32,))
    rngs = cfg.rngs()

    d = load_digits()
    X = (d.data / 16.0).astype(np.float32)
    y = d.target
    idx = rngs["data"].permutation(len(X))
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]

    model = Sequential([Dense(64, 32, rngs["init"]), ReLU(), Dense(32, 10, rngs["init"])])
    loss_fn = SoftmaxCrossEntropy()

    for _ in range(cfg.epochs):
        loss = loss_fn.forward(model.forward(X[tr]), y[tr])
        model.backward(loss_fn.backward())
        for params, grads, name in model.params_and_grads():
            params[name] -= cfg.learning_rate * grads[name]

    acc = float((model.forward(X[te], train=False).argmax(axis=1) == y[te]).mean())
    log_result("digits_64_32_10_new_core", {"test_accuracy": acc, "final_train_loss": float(loss)},
               cfg, seconds=time.perf_counter() - start)
    print("test accuracy:", acc)


if __name__ == "__main__":
    main()