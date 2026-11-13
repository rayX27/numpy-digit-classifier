# Compute budget

Hardware: Intel i5, 8 GB RAM, CPU only, Windows 10. Python 3.11.4, NumPy 2.4.6.
Model: 784-256-128-10, plain SGD, 50,000 training images, one epoch, training only (no evaluation).
Script: `experiments/profile_epoch.py` (rows also logged in `results/log.jsonl`). One epoch per setting, so timings are noisy (about plus or minus 30%).

| threads | dtype | batch | sec/epoch |
| --- | --- | --- | --- |
| 1 | float32 | 64 | 4.08 |
| 1 | float32 | 128 | 2.86 |
| 1 | float32 | 256 | 2.90 |
| 1 | float32 | 512 | 2.28 |
| 1 | float64 | 64 | 5.40 |
| 1 | float64 | 128 | 3.84 |
| 1 | float64 | 256 | 3.48 |
| 1 | float64 | 512 | 3.73 |
| 4 | float32 | 64 | 2.48 |
| 4 | float32 | 128 | 1.94 |
| 4 | float32 | 256 | 1.28 |
| 4 | float32 | 512 | 1.58 |
| 4 | float64 | 64 | 5.33 |
| 4 | float64 | 128 | 6.36 |
| 4 | float64 | 256 | 3.66 |
| 4 | float64 | 512 | 2.35 |

## Decisions

- dtype: float32 (about 1.5 to 3x faster than float64).
- BLAS threads: 4.
- Planning figure: 2 s per epoch (conservative, because the best measured value was 1.3 s and timings are noisy).
- Epoch budget: 30 epochs for the full comparison (about 1 minute per run, about 20 minutes for 4 optimizers x 5 seeds).
- Tuning: 10k proxy subset, 2 seeds, 10 epochs, 12 log-uniform learning-rate trials per optimizer (about 7 minutes in total).
- Batch size is chosen in Phase 3 together with the learning rate, not here. Batch 256 is the default for now.