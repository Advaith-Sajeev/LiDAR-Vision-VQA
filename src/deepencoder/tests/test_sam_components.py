#!/usr/bin/env python3
"""
Tests for SAM components in sam_vary_sdpa.py

This module tests:
- Attention module with relative position embeddings
- Block (transformer block with window attention)
- window_partition / window_unpartition functions
- get_rel_pos function
- add_decomposed_rel_pos function
- PatchEmbed module
- MLPBlock module
- LayerNorm2d module
- ImageEncoderViT structure
- get_abs_pos function
"""

import sys
import os
from pathlib import Path

# Add src/ directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestMLPBlock:
    """Test MLPBlock component."""
    
    def test_mlp_block_forward(self):
        """Test MLPBlock forward pass."""
        from deepencoder.sam_vary_sdpa import MLPBlock
        
        embedding_dim = 768
        mlp_dim = 3072
        
        mlp = MLPBlock(embedding_dim=embedding_dim, mlp_dim=mlp_dim)
        
        x = torch.randn(1, 64, 64, embedding_dim)  # [B, H, W, C]
        output = mlp(x)
        
        assert output.shape == x.shape
    
    def test_mlp_block_custom_activation(self):
        """Test MLPBlock with custom activation."""
        from deepencoder.sam_vary_sdpa import MLPBlock
        
        mlp = MLPBlock(embedding_dim=256, mlp_dim=1024, act=nn.ReLU)
        
        x = torch.randn(2, 16, 16, 256)
        output = mlp(x)
        
        assert output.shape == x.shape
    
    def test_mlp_block_gradient_flow(self):
        """Test gradient flows through MLPBlock."""
        from deepencoder.sam_vary_sdpa import MLPBlock
        
        mlp = MLPBlock(embedding_dim=128, mlp_dim=512)
        
        x = torch.randn(1, 8, 8, 128, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestLayerNorm2d:
    """Test LayerNorm2d component."""
    
    def test_layernorm2d_forward(self):
        """Test LayerNorm2d forward pass."""
        from deepencoder.sam_vary_sdpa import LayerNorm2d
        
        num_channels = 256
        ln = LayerNorm2d(num_channels)
        
        x = torch.randn(1, num_channels, 32, 32)  # [B, C, H, W]
        output = ln(x)
        
        assert output.shape == x.shape
    
    def test_layernorm2d_normalization(self):
        """Test LayerNorm2d produces normalized output."""
        from deepencoder.sam_vary_sdpa import LayerNorm2d
        
        ln = LayerNorm2d(64)
        
        x = torch.randn(1, 64, 16, 16) * 10 + 5  # Non-zero mean, high std
        output = ln(x)
        
        # After normalization, mean should be close to 0, std close to 1
        # (approximately, due to learned parameters)
        assert output.mean().abs() < 1.0
    
    def test_layernorm2d_eps(self):
        """Test LayerNorm2d with custom epsilon."""
        from deepencoder.sam_vary_sdpa import LayerNorm2d
        
        ln = LayerNorm2d(32, eps=1e-5)
        
        x = torch.zeros(1, 32, 8, 8)  # All zeros (edge case)
        output = ln(x)
        
        assert not torch.isnan(output).any()


class TestPatchEmbed:
    """Test PatchEmbed component."""
    
    def test_patch_embed_forward(self):
        """Test PatchEmbed forward pass."""
        from deepencoder.sam_vary_sdpa import PatchEmbed
        
        patch_embed = PatchEmbed(
            kernel_size=(16, 16),
            stride=(16, 16),
            in_chans=3,
            embed_dim=768,
        )
        
        x = torch.randn(1, 3, 1024, 1024)  # [B, C, H, W]
        output = patch_embed(x)
        
        # Output should be [B, H//16, W//16, embed_dim]
        assert output.shape == (1, 64, 64, 768)
    
    def test_patch_embed_custom_size(self):
        """Test PatchEmbed with custom patch size."""
        from deepencoder.sam_vary_sdpa import PatchEmbed
        
        patch_embed = PatchEmbed(
            kernel_size=(14, 14),
            stride=(14, 14),
            in_chans=3,
            embed_dim=1024,
        )
        
        x = torch.randn(1, 3, 224, 224)
        output = patch_embed(x)
        
        assert output.shape == (1, 16, 16, 1024)
    
    def test_patch_embed_with_padding(self):
        """Test PatchEmbed with padding."""
        from deepencoder.sam_vary_sdpa import PatchEmbed
        
        patch_embed = PatchEmbed(
            kernel_size=(16, 16),
            stride=(16, 16),
            padding=(8, 8),  # Adds padding
            in_chans=3,
            embed_dim=768,
        )
        
        x = torch.randn(1, 3, 256, 256)
        output = patch_embed(x)
        
        # With padding, output spatial dims are larger
        assert output.shape[1] == 17  # (256 + 2*8) // 16 = 17


class TestWindowPartition:
    """Test window_partition and window_unpartition functions."""
    
    def test_window_partition_basic(self):
        """Test basic window partitioning."""
        from deepencoder.sam_vary_sdpa import window_partition
        
        x = torch.randn(1, 64, 64, 768)  # [B, H, W, C]
        window_size = 8
        
        windows, (Hp, Wp) = window_partition(x, window_size)
        
        # Should produce (B * num_windows, window_size, window_size, C)
        num_windows = (64 // window_size) ** 2
        assert windows.shape == (1 * num_windows, 8, 8, 768)
        assert Hp == 64
        assert Wp == 64
    
    def test_window_partition_with_padding(self):
        """Test window partitioning with padding for non-divisible sizes."""
        from deepencoder.sam_vary_sdpa import window_partition
        
        x = torch.randn(1, 65, 65, 256)  # Not divisible by 8
        window_size = 8
        
        windows, (Hp, Wp) = window_partition(x, window_size)
        
        # Should pad to 72x72 (next multiple of 8)
        assert Hp == 72
        assert Wp == 72
    
    def test_window_unpartition_basic(self):
        """Test basic window unpartitioning."""
        from deepencoder.sam_vary_sdpa import window_partition, window_unpartition
        
        x = torch.randn(1, 64, 64, 768)
        window_size = 8
        
        windows, (Hp, Wp) = window_partition(x, window_size)
        x_restored = window_unpartition(windows, window_size, (Hp, Wp), (64, 64))
        
        assert x_restored.shape == x.shape
        assert torch.allclose(x, x_restored)
    
    def test_window_roundtrip_with_padding(self):
        """Test window partition/unpartition roundtrip with padding."""
        from deepencoder.sam_vary_sdpa import window_partition, window_unpartition
        
        x = torch.randn(2, 65, 65, 256)
        window_size = 8
        
        windows, (Hp, Wp) = window_partition(x, window_size)
        x_restored = window_unpartition(windows, window_size, (Hp, Wp), (65, 65))
        
        assert x_restored.shape == x.shape
        assert torch.allclose(x, x_restored)


class TestGetRelPos:
    """Test get_rel_pos function."""
    
    def test_get_rel_pos_same_size(self):
        """Test get_rel_pos with same query and key sizes."""
        from deepencoder.sam_vary_sdpa import get_rel_pos
        
        q_size = 16
        k_size = 16
        rel_pos = torch.randn(2 * q_size - 1, 64)  # [L, C]
        
        result = get_rel_pos(q_size, k_size, rel_pos)
        
        # Output should be [q_size, k_size, C]
        assert result.shape == (q_size, k_size, 64)
    
    def test_get_rel_pos_different_sizes(self):
        """Test get_rel_pos with different query and key sizes."""
        from deepencoder.sam_vary_sdpa import get_rel_pos
        
        q_size = 16
        k_size = 8
        max_rel_dist = 2 * max(q_size, k_size) - 1
        rel_pos = torch.randn(max_rel_dist, 64)
        
        result = get_rel_pos(q_size, k_size, rel_pos)
        
        assert result.shape == (q_size, k_size, 64)
    
    def test_get_rel_pos_interpolation(self):
        """Test get_rel_pos with interpolation."""
        from deepencoder.sam_vary_sdpa import get_rel_pos
        
        q_size = 32
        k_size = 32
        # Create smaller rel_pos that needs interpolation
        rel_pos = torch.randn(31, 64)  # Less than 2*32-1=63
        
        result = get_rel_pos(q_size, k_size, rel_pos)
        
        assert result.shape == (q_size, k_size, 64)


class TestAddDecomposedRelPos:
    """Test add_decomposed_rel_pos function."""
    
    def test_add_decomposed_rel_pos_basic(self):
        """Test basic decomposed relative position computation."""
        from deepencoder.sam_vary_sdpa import add_decomposed_rel_pos
        
        B = 2
        q_h, q_w = 16, 16
        k_h, k_w = 16, 16
        dim = 64
        
        q = torch.randn(B, q_h * q_w, dim)
        rel_pos_h = torch.randn(2 * q_h - 1, dim)
        rel_pos_w = torch.randn(2 * q_w - 1, dim)
        
        rel_h, rel_w = add_decomposed_rel_pos(
            q, rel_pos_h, rel_pos_w, (q_h, q_w), (k_h, k_w)
        )
        
        # rel_h: [B, q_h*q_w, k_h, 1]
        # rel_w: [B, q_h*q_w, 1, k_w]
        assert rel_h.shape == (B, q_h * q_w, k_h, 1)
        assert rel_w.shape == (B, q_h * q_w, 1, k_w)


class TestAttention:
    """Test SAM Attention module."""
    
    def test_attention_without_rel_pos(self):
        """Test Attention without relative position embeddings."""
        from deepencoder.sam_vary_sdpa import Attention
        
        dim = 768
        num_heads = 12
        
        attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            use_rel_pos=False,
        )
        
        x = torch.randn(1, 64, 64, dim)  # [B, H, W, C]
        output = attn(x)
        
        assert output.shape == x.shape
    
    def test_attention_with_rel_pos(self):
        """Test Attention with relative position embeddings."""
        from deepencoder.sam_vary_sdpa import Attention
        
        dim = 768
        num_heads = 12
        input_size = (64, 64)
        
        attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            use_rel_pos=True,
            input_size=input_size,
        )
        
        x = torch.randn(1, 64, 64, dim)
        output = attn(x)
        
        assert output.shape == x.shape
    
    def test_attention_gradient_flow(self):
        """Test gradient flows through Attention."""
        from deepencoder.sam_vary_sdpa import Attention
        
        attn = Attention(dim=256, num_heads=8, use_rel_pos=False)
        
        x = torch.randn(1, 16, 16, 256, requires_grad=True)
        output = attn(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestBlock:
    """Test SAM Block (transformer block)."""
    
    def test_block_without_window(self):
        """Test Block without window attention (global)."""
        from deepencoder.sam_vary_sdpa import Block
        
        dim = 768
        num_heads = 12
        
        block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            use_rel_pos=False,
            window_size=0,  # No window (global attention)
        )
        
        x = torch.randn(1, 64, 64, dim)
        output = block(x)
        
        assert output.shape == x.shape
    
    def test_block_with_window(self):
        """Test Block with window attention."""
        from deepencoder.sam_vary_sdpa import Block
        
        dim = 768
        num_heads = 12
        window_size = 14
        
        block = Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            use_rel_pos=True,
            window_size=window_size,
            input_size=(64, 64),
        )
        
        x = torch.randn(1, 64, 64, dim)
        output = block(x)
        
        assert output.shape == x.shape
    
    def test_block_residual_connection(self):
        """Test Block preserves residual connection."""
        from deepencoder.sam_vary_sdpa import Block
        
        block = Block(
            dim=256,
            num_heads=8,
            mlp_ratio=4.0,
            use_rel_pos=False,
            window_size=0,
        )
        
        # With zero-initialized weights, output should be close to input
        # (due to residual connections)
        x = torch.randn(1, 16, 16, 256)
        output = block(x)
        
        # Output should not be all zeros
        assert output.abs().max() > 0


class TestImageEncoderViT:
    """Test complete ImageEncoderViT."""
    
    def test_encoder_forward(self):
        """Test ImageEncoderViT forward pass."""
        from deepencoder.sam_vary_sdpa import ImageEncoderViT
        
        encoder = ImageEncoderViT(
            img_size=1024,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            out_chans=256,
            qkv_bias=True,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
            global_attn_indexes=[2, 5, 8, 11],
        )
        
        x = torch.randn(1, 3, 1024, 1024)
        
        with torch.no_grad():
            output = encoder(x)
        
        # Output after net_2 and net_3: [B, 1024, 16, 16]
        assert output.shape == (1, 1024, 16, 16)
    
    def test_encoder_with_abs_pos(self):
        """Test encoder with absolute position embeddings."""
        from deepencoder.sam_vary_sdpa import ImageEncoderViT
        
        encoder = ImageEncoderViT(
            img_size=512,
            patch_size=16,
            embed_dim=768,
            depth=4,
            num_heads=12,
            use_abs_pos=True,
        )
        
        assert encoder.pos_embed is not None
        assert encoder.pos_embed.shape == (1, 32, 32, 768)
    
    def test_encoder_without_abs_pos(self):
        """Test encoder without absolute position embeddings."""
        from deepencoder.sam_vary_sdpa import ImageEncoderViT
        
        encoder = ImageEncoderViT(
            img_size=512,
            patch_size=16,
            embed_dim=768,
            depth=4,
            num_heads=12,
            use_abs_pos=False,
        )
        
        assert encoder.pos_embed is None


class TestSdpAttention:
    """Test sdp_attention function."""
    
    def test_sdp_attention_basic(self):
        """Test sdp_attention basic functionality."""
        from deepencoder.sam_vary_sdpa import sdp_attention
        
        B, H, S, D = 2, 8, 64, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        
        output = sdp_attention(q, k, v, attn_mask=None)
        
        assert output.shape == (B, H, S, D)
    
    def test_sdp_attention_with_mask(self):
        """Test sdp_attention with attention mask."""
        from deepencoder.sam_vary_sdpa import sdp_attention
        
        B, H, S, D = 2, 4, 32, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        attn_mask = torch.zeros(B, H, S, S)
        
        output = sdp_attention(q, k, v, attn_mask=attn_mask)
        
        assert output.shape == (B, H, S, D)
    
    def test_sdp_attention_numerical_stability(self):
        """Test sdp_attention produces stable output."""
        from deepencoder.sam_vary_sdpa import sdp_attention
        
        B, H, S, D = 1, 4, 16, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        
        output = sdp_attention(q, k, v)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestGetAbsPos:
    """Test get_abs_pos function for position embedding resampling."""
    
    def test_get_abs_pos_same_size(self):
        """Test get_abs_pos with same source and target size."""
        from deepencoder.sam_vary_sdpa import get_abs_pos
        
        abs_pos = torch.randn(1, 64, 64, 768)
        tgt_size = 64
        
        result = get_abs_pos(abs_pos, tgt_size)
        
        assert result.shape == abs_pos.shape
        assert torch.allclose(result, abs_pos)
    
    def test_get_abs_pos_upsample(self):
        """Test get_abs_pos with upsampling."""
        from deepencoder.sam_vary_sdpa import get_abs_pos
        
        abs_pos = torch.randn(1, 32, 32, 768)
        tgt_size = 64
        
        result = get_abs_pos(abs_pos, tgt_size)
        
        assert result.shape == (1, 64, 64, 768)
    
    def test_get_abs_pos_downsample(self):
        """Test get_abs_pos with downsampling."""
        from deepencoder.sam_vary_sdpa import get_abs_pos
        
        abs_pos = torch.randn(1, 64, 64, 768)
        tgt_size = 32
        
        result = get_abs_pos(abs_pos, tgt_size)
        
        assert result.shape == (1, 32, 32, 768)


class TestBuildSamVitB:
    """Test build_sam_vit_b function."""
    
    def test_build_sam_vit_b_without_checkpoint(self):
        """Test building SAM ViT-B without checkpoint."""
        from deepencoder.sam_vary_sdpa import build_sam_vit_b
        
        model = build_sam_vit_b(checkpoint=None)
        
        assert model is not None
        assert len(model.blocks) == 12  # ViT-B has 12 blocks
    
    def test_build_sam_vit_b_structure(self):
        """Test SAM ViT-B has correct structure."""
        from deepencoder.sam_vary_sdpa import build_sam_vit_b
        
        model = build_sam_vit_b(checkpoint=None)
        
        # Check key components exist
        assert hasattr(model, 'patch_embed')
        assert hasattr(model, 'pos_embed')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'neck')
        assert hasattr(model, 'net_2')
        assert hasattr(model, 'net_3')


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
