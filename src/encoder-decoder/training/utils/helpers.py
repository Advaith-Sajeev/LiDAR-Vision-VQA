"""Helper utility functions"""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module) -> Tuple[int, int, float]:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (trainable_params, all_params, trainable_percentage)
    """
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    return t, a, (t / max(1, a) * 100.0)
