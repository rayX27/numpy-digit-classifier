"""Time one training epoch for several batch sizes, dtypes and BLAS thread counts."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BATCH_SIZES = (64, 128, 256, 512)
THREADS = (1, 4)
DTYPES = ("float32", "float64")


def worker(dtype_name):
    import numpy as np
    from config import Config
    from data.loader import load_train_val
    from nn.layers import Dense, ReLU
    from nn.losses import SoftmaxCrossEntropy
    from nn.model import Sequential

    dtype = np.dtype(dtype_name)
    x, y, _, _ = load_train_val()
    x = x.astype(dtype)
    order_rng = np.random.default_rng(0)

    for bs in BATCH_SIZES:
        rngs = Config(seed=0).rngs()
        model = Sequential([Dense(784, 256, rngs["init"], dtype), ReLU(),
                            Dense(256, 128, rngs["init"], dtype), ReLU(),
                            Dense(128, 10, rngs["init"], dtype)])
        loss_fn = SoftmaxCrossEntropy()
        idx = order_rng.permutation(len(x))
        start = time.perf_counter()
        for i in range(0, len(x), bs):
            b = idx[i:i + bs]
            loss_fn.forward(model.forward(x[b]), y[b])
            model.backward(loss_fn.backward())
            for params, grads, name in model.params_and_grads():
                params[name] -= 0.01 * grads[name]
        print("RESULT " + json.dumps({"batch_size": bs, "seconds": time.perf_counter() - start}))


def main():
    from config import Config
    from experiments.logger import log_result

    rows = []
    for threads in THREADS:
        env = {**os.environ}
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[var] = str(threads)
        for dtype_name in DTYPES:
            print(f"running threads={threads} dtype={dtype_name} ...", flush=True)
            out = subprocess.run([sys.executable, __file__, "--worker", dtype_name],
                                 cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
            for line in out.splitlines():
                if line.startswith("RESULT "):
                    r = json.loads(line[7:])
                    rows.append((threads, dtype_name, r["batch_size"], r["seconds"]))
                    cfg = Config(batch_size=r["batch_size"], hidden_sizes=(256, 128))
                    log_result("profile_epoch",
                               {"seconds_per_epoch": r["seconds"], "threads": threads, "dtype": dtype_name},
                               cfg, seconds=r["seconds"])
    print("\nthreads  dtype     batch  sec/epoch")
    for t, d, bs, s in rows:
        print(f"{t:<8} {d:<9} {bs:<6} {s:.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()