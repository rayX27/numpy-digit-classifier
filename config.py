import numpy as np
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    learning_rate: float = 0.01
    batch_size: int = 128
    epochs: int = 10
    hidden_sizes: tuple = (256, 128)
    
    def __post_init__(self):
        # Spawn 4 independent RNG streams from the master seed
        sq = np.random.SeedSequence(self.seed)
        streams = sq.spawn(4)
        
        self.rng_init = np.random.default_rng(streams[0])
        self.rng_data = np.random.default_rng(streams[1])
        self.rng_dropout = np.random.default_rng(streams[2])
        self.rng_aug = np.random.default_rng(streams[3])

# Global configuration instance
config = Config()