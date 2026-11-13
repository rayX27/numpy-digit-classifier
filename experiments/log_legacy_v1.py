"""Run the frozen v1 training once and log one result row with provenance."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "legacy"))

import train as v1  # type: ignore  # legacy/train.py, not modified
from config import Config
from experiments.logger import log_result


def main():
    start = time.perf_counter()
    X_train, y_train, X_test, y_test, rng = v1.load_data()
    Y_train = v1.one_hot(y_train)
    W1, b1, W2, b2 = v1.init_weights(rng)

    for _ in range(v1.EPOCHS):
        z1, h, p = v1.forward(X_train, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = v1.backward(X_train, Y_train, z1, h, p, W2)
        W1 -= v1.LR * dW1
        b1 -= v1.LR * db1
        W2 -= v1.LR * dW2
        b2 -= v1.LR * db2

    p_test = v1.forward(X_test, W1, b1, W2, b2)[2]
    acc = float(v1.accuracy(p_test, y_test))
    seconds = time.perf_counter() - start

    cfg = Config(seed=v1.SEED, learning_rate=v1.LR, batch_size=len(X_train),
                 epochs=v1.EPOCHS, hidden_sizes=(v1.N_HIDDEN,))
    log_result("legacy_v1_digits", {"test_accuracy": acc}, cfg, seconds=seconds)
    print("logged legacy v1 run, test accuracy:", acc)


if __name__ == "__main__":
    main()