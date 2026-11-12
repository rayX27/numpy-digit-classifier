from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Config:
    seed: int = 42
    learning_rate: float = 0.01   # placeholder until tuned in Phase 3
    batch_size: int = 128
    epochs: int = 10              # placeholder until profiling in Phase 2
    hidden_sizes: tuple = (256, 128)

    def rngs(self):
        """Fresh independent RNG streams. Call once per run."""
        names = ("init", "data", "dropout", "aug")
        children = np.random.SeedSequence(self.seed).spawn(len(names))
        return {n: np.random.default_rng(s) for n, s in zip(names, children)}

    def to_dict(self):
        return asdict(self)