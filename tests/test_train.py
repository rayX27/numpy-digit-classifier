import numpy as np

from config import Config
from nn.layers import Dense, ReLU
from nn.losses import SoftmaxCrossEntropy
from nn.model import Sequential
from nn.train import batches, evaluate, train


class PlainSGD:
    def __init__(self, model, lr):
        self.model, self.lr = model, lr

    def step(self):
        for params, grads, name in self.model.params_and_grads():
            params[name] -= self.lr * grads[name]


def toy_data(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    centers = np.eye(3, 8) * 3.0
    x = (centers[y] + 0.3 * rng.standard_normal((n, 8))).astype(np.float32)
    return x, y


def build(seed):
    rngs = Config(seed=seed).rngs()
    model = Sequential([Dense(8, 16, rngs["init"]), ReLU(), Dense(16, 3, rngs["init"])])
    return model, rngs


def test_batches_cover_each_sample_once():
    x = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    got = list(batches(x, y, 4, np.random.default_rng(0)))
    assert [len(b[1]) for b in got] == [4, 4, 2]
    assert sorted(np.concatenate([b[1] for b in got]).tolist()) == list(range(10))


def test_evaluate_matches_manual_computation():
    x, y = toy_data(50, 1)
    model, _ = build(0)
    loss_fn = SoftmaxCrossEntropy()
    loss, acc = evaluate(model, loss_fn, x, y, chunk=7)
    logits = model.forward(x, train=False)
    assert np.isclose(loss, loss_fn.forward(logits, y), rtol=1e-5)
    assert acc == (logits.argmax(axis=1) == y).mean()


def test_train_learns_and_is_reproducible():
    data = (*toy_data(300, 0), *toy_data(100, 1))
    histories = []
    for _ in range(2):
        model, rngs = build(3)
        opt = PlainSGD(model, 0.5)
        histories.append(train(model, SoftmaxCrossEntropy(), opt, data, 20, rngs, batch_size=32))
    h = histories[0]
    assert all(len(v) == 20 for v in h.values())
    assert h["val_loss"][-1] < h["val_loss"][0]
    assert h["val_acc"][-1] > 0.95
    assert np.allclose(h["val_loss"], histories[1]["val_loss"])