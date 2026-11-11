# Experiments

All runs: 64-32-10 network, full-batch gradient descent, seed fixed. Test set = 360 unseen images.

## 1. Learning rate (hidden=32, epochs=300)
| lr | final loss | test accuracy |
|---|---|---|
| 0.01 | 1.6483 | 64.2% |
| 0.1 | 0.2251 | 93.3% |
| 0.5 | 0.0596 | 95.6% |
| 2.0 | 2.3021 | 6.9% |

Why: A learning rate that is too small (0.01) takes steps so tiny that 300 epochs is not enough, while one that is too large (2.0) overshoots the minimum and the network collapses to random guessing (loss 2.30 = ln 10); 0.5 was the sweet spot.

## 2. Hidden layer size (lr=0.5, epochs=300)
| hidden | final loss | test accuracy |
|---|---|---|
| 4 | 0.6764 | 80.3% |
| 8 | 0.2934 | 88.6% |
| 32 | 0.0596 | 95.6% |
| 128 | 0.0504 | 95.8% |

Why: More hidden neurons give the network more capacity to tell similar digits apart, so accuracy jumps from 4 to 8 to 32 neurons, but going from 32 to 128 adds only one image of accuracy, which shows diminishing returns.

## 3. Epochs (lr=0.5, hidden=32)
| epochs | train acc | test acc |
|---|---|---|
| 20 | 90.9% | 87.8% |
| 100 | 97.0% | 95.0% |
| 300 | 99.0% | 95.6% |
| 1000 | 100% | 97.2% |

Why: Training longer keeps raising train accuracy (to 100% at 1000 epochs) and test accuracy followed (up to 97.2%), and the steady gap of about 3 points between them is mild overfitting that did not get worse in this range.

## 4. ReLU on/off (hidden=32, epochs=300)
| lr | ReLU | train acc | test acc |
|---|---|---|---|
| 0.1 | on | 95.3% | 93.3% |
| 0.1 | off | 95.8% | 92.5% |
| 0.5 | on | 99.0% | 95.6% |
| 0.5 | off | 98.8% | 95.3% |

Why: Without ReLU the two layers collapse into a single linear layer, yet it still reached about 95% because these 8x8 digits are almost separable by straight boundaries, so ReLU's extra bend only helps by a small margin on this easy dataset.