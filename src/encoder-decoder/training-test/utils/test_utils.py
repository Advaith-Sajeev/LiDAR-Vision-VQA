import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from training.utils import helpers


def test_set_seed_reproducible_across_libraries():
    # First run with a given seed
    helpers.set_seed(123)
    py_vals_1 = [random.randint(0, 10**6) for _ in range(5)]
    np_vals_1 = np.random.rand(5)
    torch_vals_1 = torch.rand(5)

    # Second run with the same seed should reproduce exactly
    helpers.set_seed(123)
    py_vals_2 = [random.randint(0, 10**6) for _ in range(5)]
    np_vals_2 = np.random.rand(5)
    torch_vals_2 = torch.rand(5)

    assert py_vals_1 == py_vals_2
    assert np.allclose(np_vals_1, np_vals_2)
    assert torch.allclose(torch_vals_1, torch_vals_2)


def test_set_seed_changes_sequence_for_different_seeds():
    helpers.set_seed(1)
    seq1 = [random.randint(0, 10**6) for _ in range(5)]

    helpers.set_seed(2)
    seq2 = [random.randint(0, 10**6) for _ in range(5)]

    # Extremely unlikely to be identical if seeding works correctly
    assert seq1 != seq2


class ToyModel(nn.Module):
    # Linear(2,3) -> 2*3 + 3 = 9 params
    # Linear(3,1) -> 3*1 + 1 = 4 params
    # Total = 13
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 3)
        self.l2 = nn.Linear(3, 1)


def test_count_trainable_params_all_trainable():
    model = ToyModel()
    t, a, pct = helpers.count_trainable_params(model)

    assert a == 13
    assert t == 13
    assert pct == pytest.approx(100.0)


def test_count_trainable_params_with_frozen_layers():
    model = ToyModel()

    # Freeze second layer
    for p in model.l2.parameters():
        p.requires_grad = False

    t, a, pct = helpers.count_trainable_params(model)

    # All params still counted in 'a'
    assert a == 13

    # Only first layer (9 params) is trainable
    assert t == 9
    assert pct == pytest.approx(9 / 13 * 100.0)
