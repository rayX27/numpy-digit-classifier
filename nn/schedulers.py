"""Learning-rate schedules: each returns a function epoch -> lr."""
import math


def constant(base_lr):
    return lambda epoch: base_lr


def cosine(base_lr, total_epochs):
    """Cosine decay from base_lr (epoch 0) towards 0 (epoch total_epochs)."""
    return lambda epoch: 0.5 * base_lr * (1.0 + math.cos(math.pi * epoch / total_epochs))