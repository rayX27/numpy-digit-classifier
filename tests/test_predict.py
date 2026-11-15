import numpy as np

from nn.layers import Dense
from nn.model import Sequential
from nn.train import predict


def test_predict_matches_argmax_across_chunks():
    rng = np.random.default_rng(0)
    model = Sequential([Dense(4, 3, rng)])
    x = rng.random((10, 4)).astype(np.float32)
    expected = model.forward(x, train=False).argmax(axis=1)
    got = predict(model, x, chunk=3)  # 10 samples, chunk 3: crosses chunk boundaries
    assert got.shape == (10,)
    assert np.array_equal(got, expected)