# Hypotheses

Predictions written and pushed before running each experiment.

| Experiment | Prediction | Date pushed |
| --- | --- | --- |
| Majority-class predictor, validation accuracy | About 11.2% (digit 1 is the most common class in MNIST train; range 10.8 to 11.7%) | 2026-11-13 |
| Softmax regression (no hidden layer), validation accuracy | About 92% (range 91.5 to 92.7%) after 30 epochs, batch 256, normalized inputs | 2026-11-13 |
| Which optimizer reaches 97% validation accuracy in the fewest epochs | Adam, in about 5 to 8 epochs; AdamW about the same | 2026-11-13 |
| Which optimizer is slowest to reach 97% | Plain SGD, at about 15 to 30 epochs (it may not reach 97% within 30 epochs at a poorly tuned learning rate) | 2026-11-13 |
| Will AdamW differ from Adam by more than noise? | No. With small weight decay and only 30 epochs, the gap should be inside seed std and the bootstrap CI | 2026-11-13 |
| 784-256-128-10, validation accuracy after 30 epochs with the best optimizer | About 98.0% (range 97.8 to 98.3%) with no regularization | 2026-11-13 |
| Best learning rate range, plain SGD versus Adam | SGD about 0.05 to 0.3 (momentum about 0.01 to 0.1); Adam about 5e-4 to 3e-3 | 2026-11-13 |
| 784-128-10 (one hidden layer), validation accuracy after 30 epochs, best optimizer, 5 seeds | About 97.7% (range 97.4 to 98.0%), against about 98.0% for 784-256-128-10 | 2026-11-15 |
| Which architecture is better, and by how much? | 784-256-128-10, by about 0.3 percentage points (range 0.0 to 0.6); about 75% confident it is better at all | 2026-11-15 |
| Is that gap larger than the noise? | Larger than the seed std (about 0.1 points), so it should show in the means. Each accuracy's own bootstrap CI (about plus or minus 0.27 points) will probably still overlap, so the paired bootstrap difference is the test that decides | 2026-11-15 |
| Final training accuracy minus validation accuracy, best optimizer, 30 epochs | 784-128-10 about 1.5 to 2.5 points; 784-256-128-10 about 1.8 to 3 points (the larger model memorizes slightly more) | 2026-11-15 |
| Jump from softmax regression to 784-128-10 | About 5 to 6 points, much larger than the 128 to 256-128 gap | 2026-11-15 |

Note: all predictions above were made by Claude at my request, before the experiments they cover were run. The learning rates were tuned only on 784-256-128-10, so 784-128-10 may be slightly disadvantaged. Still to score after Session 7: the 97%-epochs, AdamW-vs-Adam, 98% and all architecture-comparison rows.
