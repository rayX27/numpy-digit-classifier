import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data / 16.0   # shape (1797, 64), scaled to 0..1
y = digits.target        # shape (1797,), integers 0..9

print(X.shape, y.shape, X.min(), X.max())
print(X[0])

# look at a few digits
fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
for ax, img, label in zip(axes.ravel(), digits.images, y):
    ax.imshow(img, cmap="gray_r")
    ax.set_title(label)
    ax.axis("off")
plt.show()