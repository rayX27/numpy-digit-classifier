import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data / 16.0
y = digits.target

rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
split = int(0.8 * len(X))
X_train, y_train = X[idx[:split]], y[idx[:split]]
X_test, y_test = X[idx[split:]], y[idx[split:]]
Y_train = np.eye(10)[y_train]

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    h = np.maximum(0, z1)
    z2 = h @ W2 + b2
    return z1, h, softmax(z2)

def loss_fn(p, Y):
    return -np.mean(np.sum(Y * np.log(p + 1e-12), axis=1))

def backward(X, Y, z1, h, p, W2):
    N = len(X)
    d2 = (p - Y) / N
    dW2 = h.T @ d2
    db2 = d2.sum(axis=0)
    d1 = (d2 @ W2.T) * (z1 > 0)
    dW1 = X.T @ d1
    db1 = d1.sum(axis=0)
    return dW1, db1, dW2, db2

def accuracy(p, y):
    return np.mean(np.argmax(p, axis=1) == y)

def train(lr, n_hidden, epochs):
    r = np.random.default_rng(0)
    W1 = r.standard_normal((64, n_hidden)) * np.sqrt(2.0 / 64)
    b1 = np.zeros(n_hidden)
    W2 = r.standard_normal((n_hidden, 10)) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros(10)
    for _ in range(epochs):
        z1, h, p = forward(X_train, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X_train, Y_train, z1, h, p, W2)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    loss = loss_fn(forward(X_train, W1, b1, W2, b2)[2], Y_train)
    test_acc = accuracy(forward(X_test, W1, b1, W2, b2)[2], y_test)
    return loss, test_acc

print("Experiment: learning rate (hidden=32, epochs=300)")
for lr in [0.01, 0.1, 0.5, 2.0]:
    loss, acc = train(lr, 32, 300)
    print("lr", lr, "| final loss", round(loss, 4), "| test accuracy", round(acc, 4))

    print()
print("Experiment: hidden size (lr=0.5, epochs=300)")
for n in [4, 8, 32, 128]:
    loss, acc = train(0.5, n, 300)
    print("hidden", n, "| final loss", round(loss, 4), "| test accuracy", round(acc, 4))


    def train2(lr, n_hidden, epochs):
        r = np.random.default_rng(0)
        W1 = r.standard_normal((64, n_hidden)) * np.sqrt(2.0 / 64)
        b1 = np.zeros(n_hidden)
        W2 = r.standard_normal((n_hidden, 10)) * np.sqrt(2.0 / n_hidden)
        b2 = np.zeros(10)
        for _ in range(epochs):
            z1, h, p = forward(X_train, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward(X_train, Y_train, z1, h, p, W2)
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
        train_acc = accuracy(forward(X_train, W1, b1, W2, b2)[2], y_train)
        test_acc = accuracy(forward(X_test, W1, b1, W2, b2)[2], y_test)
        return train_acc, test_acc

print()
print("Experiment: epochs (lr=0.5, hidden=32)")
for ep in [20, 100, 300, 1000]:
    tr, te = train2(0.5, 32, ep)
    print("epochs", ep, "| train accuracy", round(tr, 4), "| test accuracy", round(te, 4))

def train3(lr, n_hidden, epochs, use_relu):
    r = np.random.default_rng(0)
    W1 = r.standard_normal((64, n_hidden)) * np.sqrt(2.0 / 64)
    b1 = np.zeros(n_hidden)
    W2 = r.standard_normal((n_hidden, 10)) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros(10)
    N = len(X_train)

    def predict(X):
        z1 = X @ W1 + b1
        h = np.maximum(0, z1) if use_relu else z1
        return softmax(h @ W2 + b2)

    for _ in range(epochs):
        z1 = X_train @ W1 + b1
        h = np.maximum(0, z1) if use_relu else z1
        p = softmax(h @ W2 + b2)
        d2 = (p - Y_train) / N
        dW2 = h.T @ d2
        db2 = d2.sum(axis=0)
        d1 = d2 @ W2.T
        if use_relu:
            d1 = d1 * (z1 > 0)
        dW1 = X_train.T @ d1
        db1 = d1.sum(axis=0)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return accuracy(predict(X_train), y_train), accuracy(predict(X_test), y_test)

print()
print("Experiment: ReLU on/off (hidden=32, epochs=300)")
for lr in [0.1, 0.5]:
    for use_relu in [True, False]:
        tr, te = train3(lr, 32, 300, use_relu)
        print("lr", lr, "| ReLU", use_relu, "| train accuracy", round(tr, 4), "| test accuracy", round(te, 4))