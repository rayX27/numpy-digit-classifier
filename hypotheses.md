# Hypotheses

Predictions written and pushed before running each experiment.

| Majority-class predictor, validation accuracy | About 11.2% (digit 1 is the most common class in MNIST train; range 10.8 to 11.7%) | 2026-11-13 |
| Softmax regression (no hidden layer), validation accuracy | About 92% (range 91.5 to 92.7%) after 30 epochs, batch 256, normalized inputs | 2026-11-13 |
| Which optimizer reaches 97% validation accuracy in the fewest epochs | Adam, in about 5 to 8 epochs; AdamW about the same | 2026-11-13 |
| Which optimizer is slowest to reach 97% | Plain SGD, at about 15 to 30 epochs (it may not reach 97% within 30 epochs at a poorly tuned learning rate) | 2026-11-13 |
| Will AdamW differ from Adam by more than noise? | No. With small weight decay and only 30 epochs, the gap should be inside seed std and the bootstrap CI | 2026-11-13 |
| 784-256-128-10, validation accuracy after 30 epochs with the best optimizer | About 98.0% (range 97.8 to 98.3%) with no regularization | 2026-11-13 |
| Best learning rate range, plain SGD versus Adam | SGD about 0.05 to 0.3 (momentum about 0.01 to 0.1); Adam about 5e-4 to 3e-3 | 2026-11-13 |

Note: these predictions were made by Claude at my request, before any Phase 3 experiment was run.