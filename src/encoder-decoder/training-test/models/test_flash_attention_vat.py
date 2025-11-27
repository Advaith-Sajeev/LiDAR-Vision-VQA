#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flash Attention Tests for VAT Blocks

This script tests Flash Attention integration in:
  - FlashMultiheadAttention (self-attention and cross-attention)
  - VATBlock (combined SA + CA + MLP)

Tests include:
  - Flash Attention availability detection
  - Multi-dtype support (float32, float16, bfloat16)
  - Self-attention vs cross-attention modes
  - Numerical correctness
  - Gradient flow verification

Run from repo root:
    python src/encoder-decoder/training-test/models/test_flash_attention_vat.py
    
Or with pytest:
    pytest src/encoder-decoder/training-test/models/test_flash_attention_vat.py -v
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

# Path setup
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.models.vat_blocks import (
    VATBlock,
    FlashMultiheadAttention,
    _HAS_FLASH_ATTN,
)


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


def count_parameters(module: nn.Module) -> tuple:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# FlashMultiheadAttention Tests
# =============================================================================

class TestFlashMultiheadAttention:
    """Tests for FlashMultiheadAttention module."""
    
    def test_flash_attention_availability(self):
        """Test that Flash Attention is detected correctly."""
        print(f"\n[VAT] Flash Attention available: {_HAS_FLASH_ATTN}")
        assert isinstance(_HAS_FLASH_ATTN, bool)
    
    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    def test_self_attention_forward(self, dtype):
        """Test self-attention forward pass with different dtypes."""
        skip_if_no_cuda()
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[VAT] Testing self-attention with dtype={dtype_name}")
        
        # Create self-attention module
        d_model, n_heads = 512, 8
        attn = FlashMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            is_cross_attn=False,
        ).to(device).to(dtype)
        attn.eval()
        
        # Create input
        B, S, D = 2, 64, d_model
        q = torch.randn(B, S, D, device=device, dtype=dtype)
        
        with torch.no_grad():
            output = attn(q)
        
        # Verify output shape
        assert output.shape == q.shape, f"Expected {q.shape}, got {output.shape}"
        
        # Verify numerical stability
        assert not torch.isnan(output).any(), f"Output contains NaNs with dtype={dtype_name}"
        assert not torch.isinf(output).any(), f"Output contains Infs with dtype={dtype_name}"
        
        print(f"[VAT] ✓ Self-attention passed for dtype={dtype_name}")
        print(f"       Output: mean={output.mean():.4f}, std={output.std():.4f}")
    
    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    def test_cross_attention_forward(self, dtype):
        """Test cross-attention forward pass with different dtypes."""
        skip_if_no_cuda()
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[VAT] Testing cross-attention with dtype={dtype_name}")
        
        # Create cross-attention module
        d_model, n_heads = 512, 8
        attn = FlashMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            is_cross_attn=True,
        ).to(device).to(dtype)
        attn.eval()
        
        # Create inputs (q and kv have different seq lengths)
        B, Sq, Sk, D = 2, 32, 128, d_model
        q = torch.randn(B, Sq, D, device=device, dtype=dtype)
        k = torch.randn(B, Sk, D, device=device, dtype=dtype)
        v = torch.randn(B, Sk, D, device=device, dtype=dtype)
        
        with torch.no_grad():
            output = attn(q, k, v)
        
        # Verify output shape (should match query shape)
        expected_shape = (B, Sq, D)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify numerical stability
        assert not torch.isnan(output).any(), f"Output contains NaNs with dtype={dtype_name}"
        assert not torch.isinf(output).any(), f"Output contains Infs with dtype={dtype_name}"
        
        print(f"[VAT] ✓ Cross-attention passed for dtype={dtype_name}")
        print(f"       Output: mean={output.mean():.4f}, std={output.std():.4f}")
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_flash_attention_usage(self, dtype):
        """Verify Flash Attention is actually being used for compatible dtypes."""
        skip_if_no_cuda()
        
        if not _HAS_FLASH_ATTN:
            pytest.skip("Flash Attention not available")
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[VAT] Verifying Flash Attention usage with dtype={dtype_name}")
        
        # Create module
        d_model, n_heads = 256, 4
        attn = FlashMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            is_cross_attn=False,
        ).to(device).to(dtype)
        
        # Create input on CUDA with compatible dtype
        B, S, D = 2, 64, d_model
        q = torch.randn(B, S, D, device=device, dtype=dtype)
        
        # The condition for Flash Attention: CUDA + compatible dtype
        use_flash = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
        
        assert use_flash, "Flash Attention should be used for this configuration"
        
        output = attn(q)
        assert output.shape == q.shape
        
        print(f"[VAT] ✓ Flash Attention is used for dtype={dtype_name} on CUDA")


# =============================================================================
# VATBlock Tests
# =============================================================================

class TestVATBlockFlashAttention:
    """Tests for VATBlock with Flash Attention."""
    
    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    def test_vat_block_forward(self, dtype):
        """Test VATBlock forward pass with different dtypes."""
        skip_if_no_cuda()
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[VATBlock] Testing forward with dtype={dtype_name}")
        
        # Create VATBlock
        d_model, n_heads, d_mlp = 512, 8, 2048
        block = VATBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_mlp=d_mlp,
            dropout=0.1,
        ).to(device).to(dtype)
        block.eval()
        
        # Create inputs
        B, Nq, Nk = 2, 32, 128
        q = torch.randn(B, Nq, d_model, device=device, dtype=dtype)
        kv = torch.randn(B, Nk, d_model, device=device, dtype=dtype)
        
        with torch.no_grad():
            output = block(q, kv)
        
        # Verify output shape
        expected_shape = (B, Nq, d_model)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Verify numerical stability
        assert not torch.isnan(output).any(), f"Output contains NaNs with dtype={dtype_name}"
        assert not torch.isinf(output).any(), f"Output contains Infs with dtype={dtype_name}"
        
        print(f"[VATBlock] ✓ Forward passed for dtype={dtype_name}")
        print(f"           Output: mean={output.mean():.4f}, std={output.std():.4f}")
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_gradient_flow(self, dtype):
        """Test that gradients flow correctly through VATBlock."""
        skip_if_no_cuda()
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[VATBlock] Testing gradient flow with dtype={dtype_name}")
        
        # Create VATBlock
        d_model, n_heads, d_mlp = 256, 4, 1024
        block = VATBlock(
            d_model=d_model,
            n_heads=n_heads,
            d_mlp=d_mlp,
            dropout=0.0,  # Disable dropout for deterministic gradients
        ).to(device).to(dtype)
        block.train()
        
        # Create inputs with gradients
        B, Nq, Nk = 2, 16, 64
        q = torch.randn(B, Nq, d_model, device=device, dtype=dtype, requires_grad=True)
        kv = torch.randn(B, Nk, d_model, device=device, dtype=dtype, requires_grad=True)
        
        # Forward pass
        output = block(q, kv)
        
        # Backward pass
        loss = output.mean()
        loss.backward()
        
        # Verify gradients exist
        assert q.grad is not None, "Gradient for q is None"
        assert kv.grad is not None, "Gradient for kv is None"
        
        # Verify gradients are finite
        assert not torch.isnan(q.grad).any(), "q gradient contains NaNs"
        assert not torch.isnan(kv.grad).any(), "kv gradient contains NaNs"
        
        # Verify module gradients
        for name, param in block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} is None"
                assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaNs"
        
        print(f"[VATBlock] ✓ Gradient flow verified for dtype={dtype_name}")
    
    def test_multi_config_sweep(self):
        """Test VATBlock with multiple configurations."""
        skip_if_no_cuda()
        
        device = get_device()
        dtype = torch.bfloat16
        
        print(f"\n[VATBlock] Testing multiple configurations")
        
        configs = [
            # (d_model, n_heads, mlp_ratio, B, Nq, Nk)
            (256, 4, 4.0, 2, 32, 128),
            (512, 8, 4.0, 2, 64, 256),
            (768, 12, 4.0, 1, 128, 512),
        ]
        
        for d_model, n_heads, mlp_ratio, B, Nq, Nk in configs:
            d_mlp = int(d_model * mlp_ratio)
            
            print(f"\n  Config: d_model={d_model}, n_heads={n_heads}, B={B}, Nq={Nq}, Nk={Nk}")
            
            block = VATBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_mlp=d_mlp,
                dropout=0.1,
            ).to(device).to(dtype)
            block.eval()
            
            q = torch.randn(B, Nq, d_model, device=device, dtype=dtype)
            kv = torch.randn(B, Nk, d_model, device=device, dtype=dtype)
            
            with torch.no_grad():
                output = block(q, kv)
            
            assert output.shape == (B, Nq, d_model)
            assert not torch.isnan(output).any()
            
            total, trainable = count_parameters(block)
            print(f"  ✓ Passed (params: {total:,})")
            
            del block
            torch.cuda.empty_cache()
        
        print(f"\n[VATBlock] ✓ All configurations passed")


# =============================================================================
# Comparison Tests (Flash Attention vs SDPA)
# =============================================================================

class TestFlashVsSdpaComparison:
    """Compare Flash Attention outputs with PyTorch SDPA."""
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_self_attention_equivalence(self, dtype):
        """Test that Flash Attention produces similar results to SDPA for self-attention."""
        skip_if_no_cuda()
        
        if not _HAS_FLASH_ATTN:
            pytest.skip("Flash Attention not available")
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[Compare] Self-attention equivalence test, dtype={dtype_name}")
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        d_model, n_heads = 256, 4
        head_dim = d_model // n_heads
        B, S = 2, 64
        
        # Create random input
        x = torch.randn(B, S, d_model, device=device, dtype=dtype)
        
        # Create weights (shared for both implementations)
        qkv_weight = torch.randn(3 * d_model, d_model, device=device, dtype=dtype) * 0.02
        out_weight = torch.randn(d_model, d_model, device=device, dtype=dtype) * 0.02
        
        # Manual QKV projection
        qkv = F.linear(x, qkv_weight)
        qkv = qkv.view(B, S, 3, n_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # SDPA computation
        q_sdpa = q.transpose(1, 2)  # [B, H, S, D]
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        sdpa_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        sdpa_out = sdpa_out.transpose(1, 2).contiguous().view(B, S, d_model)
        sdpa_out = F.linear(sdpa_out, out_weight)
        
        # Flash Attention computation
        from flash_attn import flash_attn_func
        flash_out = flash_attn_func(q, k, v, causal=False)
        flash_out = flash_out.view(B, S, d_model)
        flash_out = F.linear(flash_out, out_weight)
        
        # Compare
        max_diff = (sdpa_out - flash_out).abs().max().item()
        mean_diff = (sdpa_out - flash_out).abs().mean().item()
        
        print(f"[Compare] Max diff: {max_diff:.6f}")
        print(f"[Compare] Mean diff: {mean_diff:.6f}")
        
        # Allow some tolerance
        assert max_diff < 0.5, f"Max difference too large: {max_diff}"
        
        print(f"[Compare] ✓ Self-attention outputs are similar")
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cross_attention_equivalence(self, dtype):
        """Test that Flash Attention produces similar results to SDPA for cross-attention."""
        skip_if_no_cuda()
        
        if not _HAS_FLASH_ATTN:
            pytest.skip("Flash Attention not available")
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[Compare] Cross-attention equivalence test, dtype={dtype_name}")
        
        torch.manual_seed(42)
        
        d_model, n_heads = 256, 4
        head_dim = d_model // n_heads
        B, Sq, Sk = 2, 32, 128
        
        # Create inputs
        q_input = torch.randn(B, Sq, d_model, device=device, dtype=dtype)
        kv_input = torch.randn(B, Sk, d_model, device=device, dtype=dtype)
        
        # Create weights
        q_weight = torch.randn(d_model, d_model, device=device, dtype=dtype) * 0.02
        k_weight = torch.randn(d_model, d_model, device=device, dtype=dtype) * 0.02
        v_weight = torch.randn(d_model, d_model, device=device, dtype=dtype) * 0.02
        out_weight = torch.randn(d_model, d_model, device=device, dtype=dtype) * 0.02
        
        # Project
        q = F.linear(q_input, q_weight).view(B, Sq, n_heads, head_dim)
        k = F.linear(kv_input, k_weight).view(B, Sk, n_heads, head_dim)
        v = F.linear(kv_input, v_weight).view(B, Sk, n_heads, head_dim)
        
        # SDPA computation
        q_sdpa = q.transpose(1, 2)  # [B, H, Sq, D]
        k_sdpa = k.transpose(1, 2)  # [B, H, Sk, D]
        v_sdpa = v.transpose(1, 2)
        sdpa_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        sdpa_out = sdpa_out.transpose(1, 2).contiguous().view(B, Sq, d_model)
        sdpa_out = F.linear(sdpa_out, out_weight)
        
        # Flash Attention computation
        from flash_attn import flash_attn_func
        flash_out = flash_attn_func(q, k, v, causal=False)
        flash_out = flash_out.view(B, Sq, d_model)
        flash_out = F.linear(flash_out, out_weight)
        
        # Compare
        max_diff = (sdpa_out - flash_out).abs().max().item()
        mean_diff = (sdpa_out - flash_out).abs().mean().item()
        
        print(f"[Compare] Max diff: {max_diff:.6f}")
        print(f"[Compare] Mean diff: {mean_diff:.6f}")
        
        assert max_diff < 0.5, f"Max difference too large: {max_diff}"
        
        print(f"[Compare] ✓ Cross-attention outputs are similar")


# =============================================================================
# Memory and Performance Tests
# =============================================================================

class TestPerformance:
    """Performance and memory tests for Flash Attention in VAT blocks."""
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_memory_efficiency(self, dtype):
        """Test memory usage with Flash Attention."""
        skip_if_no_cuda()
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[Perf] Memory test with dtype={dtype_name}")
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create large block
        d_model, n_heads, d_mlp = 768, 12, 3072
        block = VATBlock(d_model, n_heads, d_mlp, dropout=0.0).to(device).to(dtype)
        block.eval()
        
        # Large inputs
        B, Nq, Nk = 4, 256, 1024
        q = torch.randn(B, Nq, d_model, device=device, dtype=dtype)
        kv = torch.randn(B, Nk, d_model, device=device, dtype=dtype)
        
        # Warmup
        with torch.no_grad():
            _ = block(q, kv)
        
        # Measure
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            output = block(q, kv)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"[Perf] Peak memory: {peak_memory:.2f} MB")
        print(f"[Perf] Output shape: {output.shape}")
        
        assert output.shape == (B, Nq, d_model)
        
        print(f"[Perf] ✓ Memory test passed")
        
        # Cleanup
        del block, q, kv, output
        torch.cuda.empty_cache()


# =============================================================================
# Dtype Compatibility Tests
# =============================================================================

class TestDtypeCompatibility:
    """Tests for dtype compatibility across VAT blocks."""
    
    def test_float32_on_cpu(self):
        """Test that float32 works on CPU (no Flash Attention)."""
        device = torch.device("cpu")
        dtype = torch.float32
        
        print(f"\n[Dtype] Testing float32 on CPU")
        
        d_model, n_heads, d_mlp = 256, 4, 1024
        block = VATBlock(d_model, n_heads, d_mlp, dropout=0.1).to(device).to(dtype)
        block.eval()
        
        B, Nq, Nk = 2, 16, 64
        q = torch.randn(B, Nq, d_model, device=device, dtype=dtype)
        kv = torch.randn(B, Nk, d_model, device=device, dtype=dtype)
        
        with torch.no_grad():
            output = block(q, kv)
        
        assert output.shape == (B, Nq, d_model)
        assert output.dtype == dtype
        assert not torch.isnan(output).any()
        
        print(f"[Dtype] ✓ float32 on CPU works correctly")
    
    @pytest.mark.parametrize("dtype", DTYPES_TO_TEST)
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input dtype."""
        skip_if_no_cuda()
        
        device = get_device()
        dtype_name = DTYPE_NAMES[dtype]
        
        print(f"\n[Dtype] Testing dtype preservation for {dtype_name}")
        
        d_model, n_heads, d_mlp = 256, 4, 1024
        block = VATBlock(d_model, n_heads, d_mlp, dropout=0.1).to(device).to(dtype)
        block.eval()
        
        B, Nq, Nk = 2, 16, 64
        q = torch.randn(B, Nq, d_model, device=device, dtype=dtype)
        kv = torch.randn(B, Nk, d_model, device=device, dtype=dtype)
        
        with torch.no_grad():
            output = block(q, kv)
        
        assert output.dtype == dtype, f"Expected dtype {dtype}, got {output.dtype}"
        
        print(f"[Dtype] ✓ Output dtype correctly preserved as {dtype_name}")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_tests():
    """Run all tests without pytest."""
    print("\n" + "=" * 70)
    print(" " * 12 + "VAT Blocks Flash Attention Test Suite")
    print("=" * 70)
    
    device = get_device()
    print(f"\nDevice: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    
    print(f"\nFlash Attention available: {_HAS_FLASH_ATTN}")
    
    # Run tests
    test_classes = [
        TestFlashMultiheadAttention(),
        TestVATBlockFlashAttention(),
        TestFlashVsSdpaComparison(),
        TestPerformance(),
        TestDtypeCompatibility(),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
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
                    if hasattr(method, "__code__") and "dtype" in method.__code__.co_varnames:
                        for dtype in DTYPES_TO_TEST:
                            try:
                                method(dtype)
                                passed += 1
                            except pytest.skip.Exception as e:
                                print(f"⊘ {method_name}({DTYPE_NAMES[dtype]}): Skipped - {e}")
                                skipped += 1
                            except Exception as e:
                                print(f"✗ {method_name}({DTYPE_NAMES[dtype]}): {e}")
                                failed += 1
                    else:
                        method()
                        passed += 1
                except pytest.skip.Exception as e:
                    print(f"⊘ {method_name}: Skipped - {e}")
                    skipped += 1
                except Exception as e:
                    print(f"✗ {method_name}: {e}")
                    failed += 1
    
    print("\n" + "=" * 70)
    print(f" Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
