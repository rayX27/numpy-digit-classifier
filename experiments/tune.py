"""Tune the learning rate of each optimizer on a 10k proxy subset (identical protocol for all).

Protocol: 12 log-uniform learning-rate draws (the same random draws for every optimizer, mapped
into each optimizer's range), 2 seeds, 10 epochs, constant learning rate, batch 256.
Score = mean final-epoch validation accuracy. Validation is the selection set.
Use --quick for a fast smoke test (2 trials, 1 seed, 2 epochs; writes no files).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config
from data.loader import load_train_val, train_stats
from experiments.logger import log_result
from nn.layers import Dense, Normalize, ReLU
from nn.losses import SoftmaxCrossEntropy
from nn.model import Sequential
from nn.optim import SGD, Adam, AdamW, SGDMomentum
from nn.train import train

N_TRIALS = 12
SEEDS = (0, 1)
EPOCHS = 10
BATCH_SIZE = 256
PROXY_SIZE = 10000
PROXY_SEED = 777
DRAW_SEED = 2026
HIDDEN = (256, 128)

RANGES = {"sgd": (1e-3, 1.0), "momentum": (1e-3, 0.5), "adam": (1e-4, 1e-2), "adamw": (1e-4, 1e-2)}
BUILDERS = {
    "sgd": lambda m, lr: SGD(m, lr),
    "momentum": lambda m, lr: SGDMomentum(m, lr, 0.9),
    "adam": lambda m, lr: Adam(m, lr),
    "adamw": lambda m, lr: AdamW(m, lr, weight_decay=0.01),
}


def build_model(rngs, mean, std):
    return Sequential([Normalize(mean, std),
                       Dense(784, HIDDEN[0], rngs["init"]), ReLU(),
                       Dense(HIDDEN[0], HIDDEN[1], rngs["init"]), ReLU(),
                       Dense(HIDDEN[1], 10, rngs["init"])])


def draw_lrs(name, n):
    """Same uniform draws for every optimizer, mapped log-uniformly into its own range."""
    u = np.random.default_rng(DRAW_SEED).random(n)
    lo, hi = RANGES[name]
    return np.sort(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))


def main(quick=False):
    n_trials, seeds, epochs = (2, SEEDS[:1], 2) if quick else (N_TRIALS, SEEDS, EPOCHS)
    x_tr, y_tr, x_va, y_va = load_train_val()
    mean, std = train_stats(x_tr)  # fixed normalization from the full training split
    keep = np.random.default_rng(PROXY_SEED).permutation(len(x_tr))[:PROXY_SIZE]
    data = (x_tr[keep], y_tr[keep], x_va, y_va)

    summary = {}
    for name in BUILDERS:
        scores = []
        for lr in draw_lrs(name, n_trials):
            accs = []
            for seed in seeds:
                cfg = Config(seed=seed, learning_rate=float(lr), batch_size=BATCH_SIZE,
                             epochs=epochs, hidden_sizes=HIDDEN)
                rngs = cfg.rngs()
                model = build_model(rngs, mean, std)
                start = time.perf_counter()
                with np.errstate(all="ignore"):  # diverged runs may overflow; they just score badly
                    hist = train(model, SoftmaxCrossEntropy(), BUILDERS[name](model, float(lr)),
                                 data, epochs, rngs, batch_size=BATCH_SIZE)
                accs.append(hist["val_acc"][-1])
                if not quick:
                    log_result(f"tune_{name}", {"final_val_acc": hist["val_acc"][-1],
                                                "final_val_loss": hist["val_loss"][-1]},
                               cfg, seconds=time.perf_counter() - start)
            scores.append((float(lr), float(np.mean(accs))))
            print(f"{name:9s} lr={lr:.5f}  mean val acc={np.mean(accs):.4f}", flush=True)
        best_lr, best_acc = max(scores, key=lambda t: t[1])
        edge = best_lr in (min(s[0] for s in scores), max(s[0] for s in scores))
        summary[name] = {"best_lr": best_lr, "best_val_acc": best_acc,
                         "best_is_at_edge_of_tried_range": edge, "trials": scores}

    print("\noptimizer  best lr     val acc   edge-of-range?")
    for name, s in summary.items():
        print(f"{name:9s}  {s['best_lr']:.5f}   {s['best_val_acc']:.4f}   {s['best_is_at_edge_of_tried_range']}")

    if not quick:
        (ROOT / "configs").mkdir(exist_ok=True)
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        frozen = {k: v["best_lr"] for k, v in summary.items()}
        (ROOT / "configs" / "frozen_lrs.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        print("wrote results/tuning_summary.json and configs/frozen_lrs.json")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)