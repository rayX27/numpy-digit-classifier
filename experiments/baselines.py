"""Baselines: majority-class predictor and softmax regression (no hidden layer)."""
import time

import numpy as np

from config import Config
from data.loader import load_train_val, train_stats
from experiments.logger import log_result
from nn.layers import Dense, Normalize
from nn.losses import SoftmaxCrossEntropy
from nn.model import Sequential


def majority_baseline(y_train, x_val, y_val):
    majority = int(np.bincount(y_train).argmax())
    acc = float((y_val == majority).mean())
    return majority, acc


def softmax_regression(x_tr, y_tr, x_va, y_va, cfg):
    rngs = cfg.rngs()
    mean, std = train_stats(x_tr)
    model = Sequential([Normalize(mean, std), Dense(784, 10, rngs["init"])])
    loss_fn = SoftmaxCrossEntropy()
    for _ in range(cfg.epochs):
        order = rngs["data"].permutation(len(x_tr))
        for i in range(0, len(order), cfg.batch_size):
            b = order[i:i + cfg.batch_size]
            loss = loss_fn.forward(model.forward(x_tr[b]), y_tr[b])
            model.backward(loss_fn.backward())
            for params, grads, name in model.params_and_grads():
                params[name] -= cfg.learning_rate * grads[name]
    val_acc = float((model.forward(x_va, train=False).argmax(axis=1) == y_va).mean())
    return val_acc, float(loss)


def main():
    x_tr, y_tr, x_va, y_va = load_train_val()

    start = time.perf_counter()
    majority, acc = majority_baseline(y_tr, x_va, y_va)
    cfg0 = Config(seed=42, learning_rate=0.0, batch_size=0, epochs=0, hidden_sizes=())
    log_result("baseline_majority_class",
               {"val_accuracy": acc, "predicted_class": majority}, cfg0,
               seconds=time.perf_counter() - start)
    print(f"majority class = {majority}, validation accuracy = {acc:.4f}")

    start = time.perf_counter()
    cfg = Config(seed=42, learning_rate=0.1, batch_size=256, epochs=30, hidden_sizes=())
    val_acc, last_loss = softmax_regression(x_tr, y_tr, x_va, y_va, cfg)
    log_result("baseline_softmax_regression",
               {"val_accuracy": val_acc, "final_batch_loss": last_loss}, cfg,
               seconds=time.perf_counter() - start)
    print(f"softmax regression, validation accuracy = {val_acc:.4f}")


if __name__ == "__main__":
    main()