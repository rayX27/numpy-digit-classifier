"""Neural network in pure NumPy: classify 8x8 handwritten digits (64-32-10)."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

SEED = 42
LR = 0.5
EPOCHS = 300
N_IN, N_HIDDEN, N_OUT = 64, 32, 10


def load_data():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]
    return X[tr], y[tr], X[te], y[te], rng


def one_hot(labels, k=10):
    return np.eye(k)[labels]


def init_weights(rng):
    W1 = rng.standard_normal((N_IN, N_HIDDEN)) * np.sqrt(2.0 / N_IN)
    b1 = np.zeros(N_HIDDEN)
    W2 = rng.standard_normal((N_HIDDEN, N_OUT)) * np.sqrt(2.0 / N_HIDDEN)
    b2 = np.zeros(N_OUT)
    return W1, b1, W2, b2


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
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


def gradient_check(W1, b1, W2, b2, X, Y):
    """Largest difference between backprop and the slow numerical gradient."""
    z1, h, p = forward(X, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward(X, Y, z1, h, p, W2)
    worst = 0.0
    eps = 1e-5
    checks = [(W1, dW1, [(10, 5), (30, 20), (63, 31)]),
              (W2, dW2, [(15, 3), (31, 9)])]
    for param, grad, spots in checks:
        for i, j in spots:
            old = param[i, j]
            param[i, j] = old + eps
            l_plus = loss_fn(forward(X, W1, b1, W2, b2)[2], Y)
            param[i, j] = old - eps
            l_minus = loss_fn(forward(X, W1, b1, W2, b2)[2], Y)
            param[i, j] = old
            numerical = (l_plus - l_minus) / (2 * eps)
            worst = max(worst, abs(numerical - grad[i, j]))
    return worst


def main():
    X_train, y_train, X_test, y_test, rng = load_data()
    Y_train = one_hot(y_train)
    W1, b1, W2, b2 = init_weights(rng)

    print("parameters:", W1.size + b1.size + W2.size + b2.size)
    print("gradient check, largest difference:", gradient_check(W1, b1, W2, b2, X_train, Y_train))

    history = {"loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(EPOCHS):
        z1, h, p = forward(X_train, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X_train, Y_train, z1, h, p, W2)
        W1 -= LR * dW1
        b1 -= LR * db1
        W2 -= LR * dW2
        b2 -= LR * db2

        history["loss"].append(loss_fn(p, Y_train))
        history["train_acc"].append(accuracy(p, y_train))
        p_test = forward(X_test, W1, b1, W2, b2)[2]
        history["test_acc"].append(accuracy(p_test, y_test))
        if epoch % 50 == 0:
            print(epoch, round(history["loss"][-1], 4), round(history["test_acc"][-1], 4))

    p_test = forward(X_test, W1, b1, W2, b2)[2]
    pred = np.argmax(p_test, axis=1)
    print("final test accuracy:", accuracy(p_test, y_test))
    print("mistakes:", int(np.sum(pred != y_test)), "of", len(y_test))

    # picture 1: loss and accuracy curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(history["loss"])
    ax[0].set_title("training loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["test_acc"], label="test")
    ax[1].set_title("accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.close()
    

    # picture 2: predictions on unseen digits
    fig, axes = plt.subplots(3, 4, figsize=(7, 5.5))
    for ax, i in zip(axes.ravel(), range(12)):
        ax.imshow(X_test[i].reshape(8, 8), cmap="gray_r")
        ax.set_title(f"pred {pred[i]} (true {y_test[i]})",
                     color="green" if pred[i] == y_test[i] else "red", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    plt.close()

    # picture 3: confusion matrix
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y_test, pred), 1)
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("confusion matrix")
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    np.savez("model.npz", W1=W1, b1=b1, W2=W2, b2=b2)


if __name__ == "__main__":
    main()