"""Tests for data sampler"""

import pytest
import torch
from torch.utils.data import TensorDataset
from training.data.sampler import SingleProcessDetSampler


class TestSingleProcessDetSampler:
    """Tests for SingleProcessDetSampler"""
    
    def test_sampler_initialization(self):
        """Test sampler initialization"""
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = SingleProcessDetSampler(dataset, seed=42, shuffle=True)
        
        assert sampler.seed == 42
        assert sampler.shuffle == True
        assert sampler.epoch == 0
        assert len(sampler) == 100
    
    def test_sampler_no_shuffle(self):
        """Test sampler without shuffling"""
        dataset = TensorDataset(torch.randn(10, 5))
        sampler = SingleProcessDetSampler(dataset, seed=42, shuffle=False)
        
        indices = list(sampler)
        assert indices == list(range(10))
    
    def test_sampler_with_shuffle_deterministic(self):
        """Test that shuffling is deterministic with same seed"""
        dataset = TensorDataset(torch.randn(100, 10))
        
        sampler1 = SingleProcessDetSampler(dataset, seed=42, shuffle=True)
        indices1 = list(sampler1)
        
        sampler2 = SingleProcessDetSampler(dataset, seed=42, shuffle=True)
        indices2 = list(sampler2)
        
        assert indices1 == indices2
        assert indices1 != list(range(100))  # Should be shuffled
    
    def test_sampler_different_seeds(self):
        """Test that different seeds produce different shuffles"""
        dataset = TensorDataset(torch.randn(100, 10))
        
        sampler1 = SingleProcessDetSampler(dataset, seed=42, shuffle=True)
        indices1 = list(sampler1)
        
        sampler2 = SingleProcessDetSampler(dataset, seed=99, shuffle=True)
        indices2 = list(sampler2)
        
        assert indices1 != indices2
    
    def test_sampler_set_epoch(self):
        """Test set_epoch changes shuffle order"""
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = SingleProcessDetSampler(dataset, seed=42, shuffle=True)
        
        sampler.set_epoch(0)
        indices_epoch0 = list(sampler)
        
        sampler.set_epoch(1)
        indices_epoch1 = list(sampler)
        
        assert indices_epoch0 != indices_epoch1
        assert len(indices_epoch0) == len(indices_epoch1)
    
    def test_sampler_all_indices_present(self):
        """Test that all indices are present exactly once"""
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = SingleProcessDetSampler(dataset, seed=42, shuffle=True)
        
        indices = list(sampler)
        assert sorted(indices) == list(range(100))
    
    def test_sampler_length(self):
        """Test sampler length matches dataset"""
        for size in [10, 50, 100, 500]:
            dataset = TensorDataset(torch.randn(size, 5))
            sampler = SingleProcessDetSampler(dataset)
            assert len(sampler) == size
    
    def test_sampler_multiple_epochs(self):
        """Test that multiple epochs produce consistent but different orders"""
        dataset = TensorDataset(torch.randn(50, 10))
        sampler = SingleProcessDetSampler(dataset, seed=42, shuffle=True)
        
        epochs_indices = []
        for epoch in range(3):
            sampler.set_epoch(epoch)
            epochs_indices.append(list(sampler))
        
        # Each epoch should have different order
        assert epochs_indices[0] != epochs_indices[1]
        assert epochs_indices[1] != epochs_indices[2]
        
        # But same epoch should give same order
        sampler.set_epoch(0)
        indices_epoch0_again = list(sampler)
        assert indices_epoch0_again == epochs_indices[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
