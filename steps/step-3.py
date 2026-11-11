import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data / 16.0   # shape (1797, 64), scaled to 0..1
y = digits.target        # shape (1797,), integers 0..9

rng = np.random.default_rng(42)   # fixed seed so results repeat
idx = rng.permutation(len(X))     # shuffled list of 0..1796
split = int(0.8 * len(X))
train_idx, test_idx = idx[:split], idx[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

def one_hot(labels, k=10):
    Y = np.zeros((len(labels), k))
    Y[np.arange(len(labels)), labels] = 1.0
    return Y

Y_train = one_hot(y_train)
Y_test = one_hot(y_test)

print(X_train.shape, X_test.shape)
print(y_train[:5])
print(Y_train[:5])