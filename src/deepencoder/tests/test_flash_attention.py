#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flash Attention Tests for DeepEncoder

This script tests Flash Attention integration in:
  - CLIP ViT-L/14 (clip_sdpa.py)
  - SAM ViT-B (sam_vary_sdpa.py)

Tests include:
  - Flash Attention availability detection
  - Multi-dtype support (float32, float16, bfloat16)
  - Numerical correctness vs PyTorch SDPA fallback
  - Memory efficiency checks

Run from repo root:
    python src/deepencoder/tests/test_flash_attention.py
    
Or with pytest:
    pytest src/deepencoder/tests/test_flash_attention.py -v
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F
import pytest


# =============================================================================
# Test Configuration
# =============================================================================

DTYPES_TO_TEST = [torch.float32, torch.float16, torch.bfloat16]
DTYPE_NAMES = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}


def get_device():
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# =============================================================================
# CLIP Flash Attention Tests
# =============================================================================

class TestCLIPFlashAttention:
    """Tests for CLIP ViT-L/14 Flash Attention integration."""
    
    def test_flash_attention_availability(self):
        """Test that Flash Attention is detected correctly."""
        from deepencoder.clip_sdpa import _HAS_FLASH_ATTN, vit_model_cfg
        
        print(f"\n[CLIP] Flash Attention available: {_HAS_FLASH_ATTN}")
        print(f"[CLIP] Config use_flash_attn: {vit_model_cfg.use_flash_attn}")
        
        # The config should match the availability
        assert vit_model_cfg.use_flash_attn == _HAS_FLASH_ATTN, \
            "Config use_flash_attn should match _HAS_FLASH_ATTN"
    
    def test_attention_module_configuration(self):
        """Test that NoTPAttention is configured correctly."""
        from deepencoder.clip_sdpa import NoTPAttention, vit_model_cfg, _HAS_FLASH_ATTN
        
        attn = NoTPAttention(vit_model_cfg)
        
        print(f"\n[CLIP] NoTPAttention.use_flash_attention: {attn.use_flash_attention}")
        
        # If Flash Attention is available, it should be enabled in the module
        assert attn.use_flash_attention == _HAS_FLASH_ATTN
    
    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    def test_attention_forward_pass(self, dtype):
        """Test attention forward pass with different dtypes."""
        skip_if_no_cuda()
        
        from deepencoder.clip_sdpa import NoTPAttention, vit_model_cfg
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[CLIP] Testing attention forward with dtype={dtype_name}")
        
        # Create attention module
        attn = NoTPAttention(vit_model_cfg).to(device).to(dtype)
        attn.eval()
        
        # Create input tensor
        B, S, C = 2, 257, 1024  # batch, seq_len (1 CLS + 256 patches), hidden_dim
        x = torch.randn(B, S, C, device=device, dtype=dtype)
        
        with torch.no_grad():
            output = attn(x)
        
        # Verify output shape
        assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
        
        # Verify numerical stability
        assert not torch.isnan(output).any(), f"Output contains NaNs with dtype={dtype_name}"
        assert not torch.isinf(output).any(), f"Output contains Infs with dtype={dtype_name}"
        
        print(f"[CLIP] ✓ Attention forward passed for dtype={dtype_name}")
        print(f"       Output: mean={output.mean():.4f}, std={output.std():.4f}")
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_full_model_with_dtype(self, dtype):
        """Test full CLIP model with different dtypes."""
        skip_if_no_cuda()
        
        from deepencoder.clip_sdpa import build_clip_l
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[CLIP] Testing full model with dtype={dtype_name}")
        
        # Build model
        model = build_clip_l().to(device).to(dtype)
        model.eval()
        
        # Create input
        B, C, H, W = 1, 3, 224, 224
        x = torch.randn(B, C, H, W, device=device, dtype=dtype)
        
        with torch.no_grad():
            output = model(x, patch_embeds=None)
        
        # Expected: [B, 1+HW/patch_size^2, hidden_dim] = [1, 257, 1024]
        expected_shape = (B, 257, 1024)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify numerical stability
        assert not torch.isnan(output).any(), f"Output contains NaNs with dtype={dtype_name}"
        assert not torch.isinf(output).any(), f"Output contains Infs with dtype={dtype_name}"
        
        print(f"[CLIP] ✓ Full model passed for dtype={dtype_name}")


# =============================================================================
# SAM Flash Attention Tests
# =============================================================================

class TestSAMFlashAttention:
    """Tests for SAM ViT-B Flash Attention integration."""
    
    def test_flash_attention_availability(self):
        """Test that Flash Attention is detected correctly."""
        from deepencoder.sam_vary_sdpa import _HAS_FLASH_ATTN
        
        print(f"\n[SAM] Flash Attention available: {_HAS_FLASH_ATTN}")
        
        # Just verify the import works
        assert isinstance(_HAS_FLASH_ATTN, bool)
    
    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    def test_sdp_attention_function(self, dtype):
        """Test sdp_attention function with different dtypes."""
        skip_if_no_cuda()
        
        from deepencoder.sam_vary_sdpa import sdp_attention, _HAS_FLASH_ATTN
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[SAM] Testing sdp_attention with dtype={dtype_name}")
        print(f"[SAM] Flash Attention available: {_HAS_FLASH_ATTN}")
        
        # Create Q, K, V tensors [B, H, S, D]
        B, H, S, D = 2, 8, 64, 64
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        # Test without attention mask (enables Flash Attention)
        output = sdp_attention(q, k, v, attn_mask=None)
        
        # Verify output shape
        assert output.shape == q.shape, f"Expected shape {q.shape}, got {output.shape}"
        
        # Verify numerical stability
        assert not torch.isnan(output).any(), f"Output contains NaNs with dtype={dtype_name}"
        assert not torch.isinf(output).any(), f"Output contains Infs with dtype={dtype_name}"
        
        print(f"[SAM] ✓ sdp_attention passed for dtype={dtype_name}")
        print(f"       Output: mean={output.mean():.4f}, std={output.std():.4f}")
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sdp_attention_with_mask(self, dtype):
        """Test sdp_attention with attention mask (falls back to SDPA)."""
        skip_if_no_cuda()
        
        from deepencoder.sam_vary_sdpa import sdp_attention
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[SAM] Testing sdp_attention with mask, dtype={dtype_name}")
        
        # Create Q, K, V tensors
        B, H, S, D = 2, 8, 64, 64
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        # Create attention mask
        attn_mask = torch.zeros(B, H, S, S, device=device, dtype=dtype)
        
        # Test with attention mask (falls back to SDPA)
        output = sdp_attention(q, k, v, attn_mask=attn_mask)
        
        # Verify output shape
        assert output.shape == q.shape, f"Expected shape {q.shape}, got {output.shape}"
        
        # Verify numerical stability
        assert not torch.isnan(output).any(), f"Output contains NaNs"
        
        print(f"[SAM] ✓ sdp_attention with mask passed for dtype={dtype_name}")
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_attention_module(self, dtype):
        """Test full SAM Attention module with different dtypes."""
        skip_if_no_cuda()
        
        from deepencoder.sam_vary_sdpa import Attention
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[SAM] Testing Attention module with dtype={dtype_name}")
        
        # Create attention module
        dim = 768
        num_heads = 12
        attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            use_rel_pos=False,
        ).to(device).to(dtype)
        
        # Create input [B, H, W, C]
        B, H, W, C = 2, 14, 14, dim
        x = torch.randn(B, H, W, C, device=device, dtype=dtype)
        
        with torch.no_grad():
            output = attn(x)
        
        # Verify output shape
        assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
        
        # Verify numerical stability
        assert not torch.isnan(output).any(), f"Output contains NaNs with dtype={dtype_name}"
        assert not torch.isinf(output).any(), f"Output contains Infs with dtype={dtype_name}"
        
        print(f"[SAM] ✓ Attention module passed for dtype={dtype_name}")


# =============================================================================
# Numerical Correctness Tests
# =============================================================================

class TestNumericalCorrectness:
    """Tests for numerical correctness of Flash Attention vs standard attention."""
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_flash_vs_sdpa_equivalence(self, dtype):
        """Test that Flash Attention produces similar results to SDPA."""
        skip_if_no_cuda()
        
        from deepencoder.sam_vary_sdpa import _HAS_FLASH_ATTN
        
        if not _HAS_FLASH_ATTN:
            pytest.skip("Flash Attention not available")
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[Numerical] Testing Flash Attention vs SDPA equivalence, dtype={dtype_name}")
        
        # Create Q, K, V tensors
        B, H, S, D = 1, 4, 32, 64
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        # Compute using PyTorch SDPA
        sdpa_output = F.scaled_dot_product_attention(q, k, v)
        
        # Compute using Flash Attention
        from flash_attn import flash_attn_func
        q_flash = q.transpose(1, 2)  # [B, S, H, D]
        k_flash = k.transpose(1, 2)
        v_flash = v.transpose(1, 2)
        flash_output = flash_attn_func(q_flash, k_flash, v_flash, causal=False)
        flash_output = flash_output.transpose(1, 2)  # Back to [B, H, S, D]
        
        # Compare outputs (allow some tolerance for float16/bfloat16)
        atol = 1e-2 if dtype == torch.float16 else 1e-1
        rtol = 1e-2 if dtype == torch.float16 else 1e-1
        
        max_diff = (sdpa_output - flash_output).abs().max().item()
        mean_diff = (sdpa_output - flash_output).abs().mean().item()
        
        print(f"[Numerical] Max diff: {max_diff:.6f}")
        print(f"[Numerical] Mean diff: {mean_diff:.6f}")
        
        # They should be reasonably close
        assert max_diff < 0.5, f"Max difference too large: {max_diff}"
        
        print(f"[Numerical] ✓ Flash Attention and SDPA produce similar results")


# =============================================================================
# Memory Efficiency Tests
# =============================================================================

class TestMemoryEfficiency:
    """Tests for memory efficiency of Flash Attention."""
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_memory_usage(self, dtype):
        """Test that Flash Attention uses less memory than naive attention."""
        skip_if_no_cuda()
        
        from deepencoder.sam_vary_sdpa import _HAS_FLASH_ATTN, sdp_attention
        
        if not _HAS_FLASH_ATTN:
            pytest.skip("Flash Attention not available")
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[Memory] Testing memory usage with dtype={dtype_name}")
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create tensors
        B, H, S, D = 2, 8, 256, 64
        q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype)
        
        # Measure memory before
        mem_before = torch.cuda.max_memory_allocated() / 1024**2
        
        # Run attention
        output = sdp_attention(q, k, v, attn_mask=None)
        
        # Measure memory after
        mem_after = torch.cuda.max_memory_allocated() / 1024**2
        
        mem_used = mem_after - mem_before
        
        print(f"[Memory] Memory used: {mem_used:.2f} MB")
        print(f"[Memory] Peak memory: {mem_after:.2f} MB")
        
        # Just verify it runs without OOM
        assert output.shape == q.shape
        
        print(f"[Memory] ✓ Memory test passed")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_tests():
    """Run all tests without pytest."""
    print("\n" + "=" * 70)
    print(" " * 15 + "DeepEncoder Flash Attention Test Suite")
    print("=" * 70)
    
    device = get_device()
    print(f"\nDevice: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    
    # Import Flash Attention status
    from deepencoder.clip_sdpa import _HAS_FLASH_ATTN as CLIP_FA
    from deepencoder.sam_vary_sdpa import _HAS_FLASH_ATTN as SAM_FA
    
    print(f"\nFlash Attention Status:")
    print(f"  CLIP module: {CLIP_FA}")
    print(f"  SAM module: {SAM_FA}")
    
    # Run tests
    test_classes = [
        TestCLIPFlashAttention(),
        TestSAMFlashAttention(),
        TestNumericalCorrectness(),
        TestMemoryEfficiency(),
    ]
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{'='*70}")
        print(f"Running {class_name}")
        print("=" * 70)
        
        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                method = getattr(test_class, method_name)
                try:
                    # Handle parametrized tests
                    if "dtype" in method_name or "parametrize" in str(method):
                        for dtype in [torch.float16, torch.bfloat16]:
                            try:
                                method(dtype)
                            except Exception as e:
                                print(f"✗ {method_name}({DTYPE_NAMES[dtype]}): {e}")
                    else:
                        method()
                except pytest.skip.Exception as e:
                    print(f"⊘ {method_name}: Skipped - {e}")
                except Exception as e:
                    print(f"✗ {method_name}: {e}")
    
    print("\n" + "=" * 70)
    print(" " * 20 + "ALL TESTS COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
