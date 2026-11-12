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


n_in, n_hidden, n_out = 64, 32, 10

W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2.0 / n_in)
b1 = np.zeros(n_hidden)
W2 = rng.standard_normal((n_hidden, n_out)) * np.sqrt(2.0 / n_hidden)
b2 = np.zeros(n_out)

print(W1.shape, b1.shape, W2.shape, b2.shape)
print(W1.size + b1.size + W2.size + b2.size)   # must print 2410


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)   # subtract max: avoids overflow
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    h = np.maximum(0, z1)
    z2 = h @ W2 + b2
    p = softmax(z2)
    return z1, h, p

def loss_fn(p, Y):
    return -np.mean(np.sum(Y * np.log(p + 1e-12), axis=1))

z1, h, p = forward(X_train, W1, b1, W2, b2)
print(z1.shape, h.shape, p.shape)
print(loss_fn(p, Y_train))
print(p.sum(axis=1)[:5])


def backward(X, Y, z1, h, p, W2):
    N = len(X)
    d2 = (p - Y) / N
    dW2 = h.T @ d2
    db2 = d2.sum(axis=0)
    d1 = (d2 @ W2.T) * (z1 > 0)
    dW1 = X.T @ d1
    db1 = d1.sum(axis=0)
    return dW1, db1, dW2, db2

z1, h, p = forward(X_train, W1, b1, W2, b2)
dW1, db1, dW2, db2 = backward(X_train, Y_train, z1, h, p, W2)
print(dW1.shape, db1.shape, dW2.shape, db2.shape)

# gradient check: compare my backward pass with a slow, trusted method
def numerical_grad(param, i, j, eps=1e-5):
    old = param[i, j]
    param[i, j] = old + eps
    l_plus = loss_fn(forward(X_train, W1, b1, W2, b2)[2], Y_train)
    param[i, j] = old - eps
    l_minus = loss_fn(forward(X_train, W1, b1, W2, b2)[2], Y_train)
    param[i, j] = old
    return (l_plus - l_minus) / (2 * eps)

for (i, j) in [(0, 0), (10, 5), (30, 20), (63, 31)]:
    num = numerical_grad(W1, i, j)
    ana = dW1[i, j]
    print("W1", (i, j), num, ana, abs(num - ana))

for (i, j) in [(0, 0), (15, 3), (31, 9)]:
    num = numerical_grad(W2, i, j)
    ana = dW2[i, j]
    print("W2", (i, j), num, ana, abs(num - ana))