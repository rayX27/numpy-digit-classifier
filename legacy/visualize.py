import numpy as np
import matplotlib.pyplot as plt
from train import load_data, forward

m = np.load("model.npz")
W1, b1, W2, b2 = m["W1"], m["b1"], m["W2"], m["b2"]
X_train, y_train, X_test, y_test, _ = load_data()

# Picture 1: what each of the 32 hidden neurons responds to
fig, axes = plt.subplots(4, 8, figsize=(12, 6.5))
vmax = np.abs(W1).max()
for k, ax in enumerate(axes.ravel()):
    ax.imshow(W1[:, k].reshape(8, 8), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.axis("off")
    ax.set_title(f"neuron {k}", fontsize=8)
fig.suptitle("Hidden neurons: red = pixel excites it, blue = pixel inhibits it")
plt.tight_layout()
plt.savefig("hidden_neurons.png", dpi=150)
plt.close()

# Picture 2: probabilities for 3 correct and 3 wrong predictions
p_test = forward(X_test, W1, b1, W2, b2)[2]
pred = p_test.argmax(axis=1)
right = list(np.where(pred == y_test)[0][:3])
wrong = list(np.where(pred != y_test)[0][:3])
chosen = right + wrong

fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for col, i in enumerate(chosen):
    ok = pred[i] == y_test[i]
    axes[0, col].imshow(X_test[i].reshape(8, 8), cmap="gray_r")
    axes[0, col].set_title(f"true {y_test[i]}, pred {pred[i]}",
                           color="green" if ok else "red", fontsize=10)
    axes[0, col].axis("off")
    colors = ["green" if d == y_test[i] else "gray" for d in range(10)]
    axes[1, col].bar(range(10), p_test[i], color=colors)
    axes[1, col].set_xticks(range(10))
    axes[1, col].set_ylim(0, 1)
axes[1, 0].set_ylabel("probability")
plt.tight_layout()
plt.savefig("probabilities.png", dpi=150)
plt.close()

print("saved hidden_neurons.png and probabilities.png")