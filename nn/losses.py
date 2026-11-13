import numpy as np


class SoftmaxCrossEntropy:
    """Fused log-softmax + cross-entropy. Labels are integer class ids."""

    def forward(self, logits, y):
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        self.p = np.exp(log_probs)
        self.y = y
        return -log_probs[np.arange(len(y)), y].mean()

    def backward(self):
        grad = self.p.copy()
        grad[np.arange(len(self.y)), self.y] -= 1.0
        return grad / len(self.y)