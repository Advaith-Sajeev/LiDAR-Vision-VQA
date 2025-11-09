"""Custom sampler for deterministic single-process training"""

import torch
from torch.utils.data import Sampler, Dataset
from typing import Iterator


class SingleProcessDetSampler(Sampler[int]):
    """
    Deterministic sampler for single-process training with optional shuffling.
    """
    
    def __init__(self, data_source: Dataset, seed: int = 42, shuffle: bool = True):
        self.data_source = data_source
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        
    def set_epoch(self, epoch: int):
        """Set current epoch for deterministic shuffling."""
        self.epoch = epoch
        
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if not self.shuffle:
            return iter(range(n))
            
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(n, generator=g).tolist()
        return iter(order)
        
    def __len__(self) -> int:
        return len(self.data_source)
