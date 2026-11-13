"""Plot 20 random training images with labels, as a data sanity check."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.loader import load_train_val

ROOT = Path(__file__).resolve().parent.parent


def main():
    x, y, _, _ = load_train_val()
    idx = np.random.default_rng(0).choice(len(x), 20, replace=False)
    fig, axes = plt.subplots(4, 5, figsize=(7, 6))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(x[i].reshape(28, 28), cmap="gray_r")
        ax.set_title(f"label {y[i]}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    out = ROOT / "results" / "samples.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main()