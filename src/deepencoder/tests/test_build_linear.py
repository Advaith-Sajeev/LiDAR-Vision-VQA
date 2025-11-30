#!/usr/bin/env python3
"""
Tests for MlpProjector in build_linear.py

This module tests all projector types:
- identity
- linear
- mlp_gelu
- normlayer_downsample_mlp_gelu
- downsample_mlp_gelu
- low_high_hybrid_split_mlp_gelu
- hybrid_split_feature_mlp_gelu
- low_high_split_mlp_gelu

Also tests:
- Token pooling path
- Convolution fusion path
- get_flops_per_sample() static method
"""

import sys
import os
from pathlib import Path

# Add src/ directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest
import torch
import torch.nn as nn


class EasyDict(dict):
    """Simple dict that allows attribute access."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    
    def get(self, key, default=None):
        return super().get(key, default)


class TestIdentityProjector:
    """Test identity projector (pass-through)."""
    
    def test_identity_forward(self):
        """Test identity projector returns input unchanged."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="identity",
            input_dim=1024,
            n_embed=2048,
        )
        
        projector = MlpProjector(cfg)
        
        # Input tensor
        x = torch.randn(2, 256, 1024)  # [B, HW, C]
        
        output = projector(x)
        
        # Identity should return same tensor
        assert torch.allclose(output, x), "Identity projector should return input unchanged"
    
    def test_identity_shape_preserved(self):
        """Test identity projector preserves input shape."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="identity",
            input_dim=512,
            n_embed=1024,
        )
        
        projector = MlpProjector(cfg)
        
        x = torch.randn(4, 64, 512)
        output = projector(x)
        
        assert output.shape == x.shape


class TestLinearProjector:
    """Test linear projector."""
    
    def test_linear_forward(self):
        """Test linear projector with dimension change."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="linear",
            input_dim=1024,
            n_embed=2048,
        )
        
        projector = MlpProjector(cfg)
        
        x = torch.randn(2, 256, 1024)  # [B, HW, C]
        
        output = projector(x)
        
        assert output.shape == (2, 256, 2048), f"Expected (2, 256, 2048), got {output.shape}"
    
    def test_linear_no_nan(self):
        """Test linear projector produces no NaN values."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="linear",
            input_dim=512,
            n_embed=768,
        )
        
        projector = MlpProjector(cfg)
        
        x = torch.randn(1, 100, 512)
        output = projector(x)
        
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"


class TestMlpGeluProjector:
    """Test mlp_gelu projector with varying depths."""
    
    def test_mlp_gelu_depth_1(self):
        """Test mlp_gelu with depth=1 (single linear layer)."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="mlp_gelu",
            input_dim=1024,
            n_embed=2048,
            depth=1,
        )
        
        projector = MlpProjector(cfg)
        
        x = torch.randn(2, 256, 1024)
        output = projector(x)
        
        assert output.shape == (2, 256, 2048)
    
    def test_mlp_gelu_depth_2(self):
        """Test mlp_gelu with depth=2 (Linear + GELU + Linear)."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="mlp_gelu",
            input_dim=1024,
            n_embed=2048,
            depth=2,
        )
        
        projector = MlpProjector(cfg)
        
        x = torch.randn(2, 256, 1024)
        output = projector(x)
        
        assert output.shape == (2, 256, 2048)
    
    def test_mlp_gelu_depth_3(self):
        """Test mlp_gelu with depth=3."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="mlp_gelu",
            input_dim=512,
            n_embed=1024,
            depth=3,
        )
        
        projector = MlpProjector(cfg)
        
        x = torch.randn(1, 64, 512)
        output = projector(x)
        
        assert output.shape == (1, 64, 1024)
    
    def test_mlp_gelu_default_depth(self):
        """Test mlp_gelu uses default depth=1 if not specified."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="mlp_gelu",
            input_dim=256,
            n_embed=512,
        )
        
        projector = MlpProjector(cfg)
        
        x = torch.randn(1, 16, 256)
        output = projector(x)
        
        assert output.shape == (1, 16, 512)


class TestDownsampleMlpGeluProjector:
    """Test downsample_mlp_gelu projector."""
    
    def test_downsample_ratio_2(self):
        """Test downsample with ratio 2 (4 tokens -> 1)."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="downsample_mlp_gelu",
            input_dim=256,
            n_embed=512,
            downsample_ratio=2,
            depth=2,
            mlp_ratio=1,
        )
        
        projector = MlpProjector(cfg)
        
        # 16x16 = 256 tokens, after 2x2 downsample = 8x8 = 64 tokens
        x = torch.randn(1, 256, 256)  # [B, HW, C]
        output = projector(x)
        
        # After 2x downsample: 16/2 = 8, so 8*8 = 64 tokens
        assert output.shape == (1, 64, 512), f"Expected (1, 64, 512), got {output.shape}"
    
    def test_downsample_with_padding(self):
        """Test downsample handles non-divisible sizes with padding."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="downsample_mlp_gelu",
            input_dim=128,
            n_embed=256,
            downsample_ratio=2,
            depth=1,
            mlp_ratio=1,
        )
        
        projector = MlpProjector(cfg)
        
        # 7x7 = 49 tokens (not divisible by 2)
        # Will be padded to 8x8 = 64, then downsampled to 4x4 = 16
        x = torch.randn(1, 49, 128)
        output = projector(x)
        
        # Should produce some output without error
        assert output.shape[0] == 1
        assert output.shape[2] == 256


class TestNormlayerDownsampleMlpGeluProjector:
    """Test normlayer_downsample_mlp_gelu projector."""
    
    def test_with_layernorm(self):
        """Test downsample with LayerNorm before projection."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="normlayer_downsample_mlp_gelu",
            input_dim=256,
            n_embed=512,
            downsample_ratio=2,
            depth=2,
            mlp_ratio=1,
        )
        
        projector = MlpProjector(cfg)
        
        x = torch.randn(1, 256, 256)  # 16x16 tokens
        output = projector(x)
        
        # After 2x downsample: 8x8 = 64 tokens
        assert output.shape == (1, 64, 512)


class TestLowHighHybridSplitMlpGelu:
    """Test low_high_hybrid_split_mlp_gelu projector."""
    
    def test_hybrid_split_forward(self):
        """Test hybrid split projector with two feature sources."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="low_high_hybrid_split_mlp_gelu",
            input_dim=512,
            n_embed=1024,
            depth=2,
        )
        
        projector = MlpProjector(cfg)
        
        # This projector expects x as a tuple/list of [high_x, low_x]
        high_x = torch.randn(1, 64, 512)
        low_x = torch.randn(1, 64, 512)
        x = [high_x, low_x]
        
        output = projector(x)
        
        assert output.shape == (1, 64, 1024)


class TestHybridSplitFeatureMlpGelu:
    """Test hybrid_split_feature_mlp_gelu projector."""
    
    def test_hybrid_split_feature_forward(self):
        """Test hybrid feature split projector."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="hybrid_split_feature_mlp_gelu",
            input_dim=[256, 256],  # high and low dims
            n_embed=512,
            depth=2,
            channel_div=0.5,
        )
        
        projector = MlpProjector(cfg)
        
        # Input is concatenated features: [B, HW, high_dim + low_dim]
        x = torch.randn(1, 64, 512)  # 256 + 256 = 512
        
        output = projector(x)
        
        assert output.shape == (1, 64, 512)


class TestLowHighSplitMlpGelu:
    """Test low_high_split_mlp_gelu projector."""
    
    def test_low_high_split_forward(self):
        """Test low/high split projector with separate processing."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="low_high_split_mlp_gelu",
            input_dim=256,
            n_embed=512,
            depth=2,
        )
        
        projector = MlpProjector(cfg)
        
        # This projector expects x as a tuple/list of [high_x, low_x]
        high_x = torch.randn(1, 64, 256)  # n_embed // 2 = 256
        low_x = torch.randn(1, 64, 256)
        x = [high_x, low_x]
        
        output = projector(x)
        
        assert output.shape == (1, 64, 512)


class TestTokenPooling:
    """Test token pooling functionality."""
    
    def test_token_pooling_enabled(self):
        """Test projector with token pooling enabled."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="linear",
            input_dim=256,
            n_embed=512,
            token_pooling=True,
        )
        
        projector = MlpProjector(cfg)
        
        # With token_pooling, input goes through 2x2 pooling
        # 16x16 = 256 tokens -> 8x8 = 64 tokens (after 2x2 pooling)
        # input_dim stays the same (256) after token_pooling_layer
        x = torch.randn(1, 256, 256)
        output = projector(x)
        
        # After 2x2 pooling: 64 tokens with input_dim, then linear to n_embed
        # Note: token_pooling_layer outputs input_dim, then layers outputs n_embed
        assert output.shape == (1, 64, 512)


class TestConvFusion:
    """Test convolution fusion functionality."""
    
    def test_conv_fusion_enabled(self):
        """Test projector with conv fusion for high/low features."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="linear",
            input_dim=256,
            n_embed=512,
            conv_fusion_high_low_features=True,
        )
        
        projector = MlpProjector(cfg)
        
        # With conv_fusion, expects x[:, 0] and x[:, 1] as separate feature maps
        # fusion_layer(x[:, 0]) + x[:, 1] produces [B, input_dim] 
        # Then linear layer maps to n_embed
        x = torch.randn(1, 2, 256)  # [B, 2, C]
        
        output = projector(x)
        
        # After fusion: [B, n_embed]
        assert output.shape == (1, 512)


class TestGetFlopsPerSample:
    """Test get_flops_per_sample static method."""
    
    def test_linear_flops(self):
        """Test FLOPS calculation for linear projector."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="linear",
            input_dim=1024,
            n_embed=2048,
        )
        
        flops = MlpProjector.get_flops_per_sample(cfg)
        
        # Linear: 2 * input_dim * n_embed (forward, backward*2)
        expected = 2 * 1024 * 2048 * 3  # *3 for forward/backward
        assert flops == expected
    
    def test_mlp_gelu_flops(self):
        """Test FLOPS calculation for mlp_gelu projector."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="mlp_gelu",
            input_dim=512,
            n_embed=1024,
            depth=2,
        )
        
        flops = MlpProjector.get_flops_per_sample(cfg)
        
        # First layer + (depth-1) * n_embed * n_embed
        expected = (2 * 512 * 1024 + (2 - 1) * 2 * 1024 * 1024) * 3
        assert flops == expected
    
    def test_downsample_flops(self):
        """Test FLOPS calculation for downsample_mlp_gelu."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="downsample_mlp_gelu",
            input_dim=256,
            n_embed=512,
            downsample_ratio=2,
            depth=1,
        )
        
        flops = MlpProjector.get_flops_per_sample(cfg)
        
        # input_dim * downsample_ratio^2 for effective input dim
        effective_input = 256 * 2 * 2
        expected = (2 * effective_input * 512 + 0) * 3
        assert flops == expected
    
    def test_identity_flops(self):
        """Test FLOPS calculation for identity projector (should be 0)."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="identity",
            input_dim=256,
            n_embed=512,
        )
        
        flops = MlpProjector.get_flops_per_sample(cfg)
        
        assert flops == 0


class TestUnknownProjectorType:
    """Test error handling for unknown projector types."""
    
    def test_unknown_type_raises(self):
        """Test that unknown projector type raises ValueError."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="unknown_projector",
            input_dim=256,
            n_embed=512,
        )
        
        with pytest.raises(ValueError, match="Unknown projector type"):
            MlpProjector(cfg)


class TestProjectorGradients:
    """Test that projectors properly support gradient flow."""
    
    def test_linear_gradient_flow(self):
        """Test gradient flows through linear projector."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="linear",
            input_dim=64,
            n_embed=128,
        )
        
        projector = MlpProjector(cfg)
        
        x = torch.randn(1, 16, 64, requires_grad=True)
        output = projector(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_mlp_gelu_gradient_flow(self):
        """Test gradient flows through mlp_gelu projector."""
        from deepencoder.build_linear import MlpProjector
        
        cfg = EasyDict(
            projector_type="mlp_gelu",
            input_dim=64,
            n_embed=128,
            depth=2,
        )
        
        projector = MlpProjector(cfg)
        
        x = torch.randn(1, 16, 64, requires_grad=True)
        output = projector(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
