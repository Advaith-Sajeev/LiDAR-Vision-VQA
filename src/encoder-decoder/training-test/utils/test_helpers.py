import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from training.utils import helpers


def test_set_seed_reproducibility():
    """Test that set_seed produces reproducible results."""
    seed = 42
    
    # Set seed and generate random values
    helpers.set_seed(seed)
    random_val_1 = random.random()
    numpy_val_1 = np.random.rand()
    torch_val_1 = torch.rand(1).item()
    
    # Set same seed again and generate values
    helpers.set_seed(seed)
    random_val_2 = random.random()
    numpy_val_2 = np.random.rand()
    torch_val_2 = torch.rand(1).item()
    
    # Values should be identical
    assert random_val_1 == random_val_2
    assert numpy_val_1 == numpy_val_2
    assert torch_val_1 == torch_val_2


def test_set_seed_different_seeds_produce_different_results():
    """Test that different seeds produce different random values."""
    helpers.set_seed(42)
    torch_val_1 = torch.rand(1).item()
    
    helpers.set_seed(123)
    torch_val_2 = torch.rand(1).item()
    
    assert torch_val_1 != torch_val_2


def test_set_seed_affects_cuda(monkeypatch):
    """Test that set_seed calls cuda.manual_seed_all."""
    seed = 42
    cuda_seed_called = {"called": False, "seed": None}
    
    def _fake_manual_seed_all(s):
        cuda_seed_called["called"] = True
        cuda_seed_called["seed"] = s
    
    monkeypatch.setattr(torch.cuda, "manual_seed_all", _fake_manual_seed_all)
    
    helpers.set_seed(seed)
    
    assert cuda_seed_called["called"]
    assert cuda_seed_called["seed"] == seed


def test_count_trainable_params_all_trainable():
    """Test counting when all parameters are trainable."""
    model = nn.Sequential(
        nn.Linear(10, 5),  # 10*5 + 5 = 55 params
        nn.Linear(5, 2)     # 5*2 + 2 = 12 params
    )
    # Total: 67 params, all trainable
    
    trainable, total, percentage = helpers.count_trainable_params(model)
    
    assert trainable == 67
    assert total == 67
    assert percentage == 100.0


def test_count_trainable_params_some_frozen():
    """Test counting when some parameters are frozen."""
    model = nn.Sequential(
        nn.Linear(10, 5),  # 55 params
        nn.Linear(5, 2)     # 12 params
    )
    
    # Freeze first layer
    for param in model[0].parameters():
        param.requires_grad = False
    
    trainable, total, percentage = helpers.count_trainable_params(model)
    
    assert trainable == 12  # Only second layer trainable
    assert total == 67
    assert abs(percentage - (12 / 67 * 100)) < 0.01


def test_count_trainable_params_all_frozen():
    """Test counting when all parameters are frozen."""
    model = nn.Linear(10, 5)
    
    for param in model.parameters():
        param.requires_grad = False
    
    trainable, total, percentage = helpers.count_trainable_params(model)
    
    assert trainable == 0
    assert total == 55
    assert percentage == 0.0


def test_count_trainable_params_empty_model():
    """Test counting on a model with no parameters."""
    model = nn.Sequential()  # Empty model
    
    trainable, total, percentage = helpers.count_trainable_params(model)
    
    assert trainable == 0
    assert total == 0
    assert percentage == 0.0  # Should handle division by zero


def test_count_trainable_params_complex_model():
    """Test counting on a more complex model structure."""
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)  # 3*16*3*3 + 16 = 448 params
            self.bn = nn.BatchNorm2d(16)      # 16*2 = 32 params (weight + bias)
            self.fc = nn.Linear(16, 10)       # 16*10 + 10 = 170 params
            
        def forward(self, x):
            return self.fc(self.bn(self.conv(x)).flatten(1))
    
    model = CustomModel()
    # Total: 448 + 32 + 170 = 650 params
    
    # Freeze conv layer
    for param in model.conv.parameters():
        param.requires_grad = False
    
    trainable, total, percentage = helpers.count_trainable_params(model)
    
    assert trainable == 32 + 170  # bn + fc
    assert total == 650
    expected_percentage = (202 / 650) * 100
    assert abs(percentage - expected_percentage) < 0.01
