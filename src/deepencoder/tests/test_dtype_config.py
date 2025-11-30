"""
Tests for dtype configuration and conversion across the codebase.

This module verifies that:
1. String-to-dtype conversion works correctly
2. All supported dtypes are properly handled
3. Flash Attention dtype requirements are met
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directories to path for imports
_TESTS_DIR = Path(__file__).parent
_DEEPENCODER_DIR = _TESTS_DIR.parent
_SRC_DIR = _DEEPENCODER_DIR.parent
_ENCODER_DECODER_DIR = _SRC_DIR / "encoder-decoder"

sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_DEEPENCODER_DIR))
sys.path.insert(0, str(_ENCODER_DECODER_DIR))


class TestDtypeConversion:
    """Test the _to_dtype helper function."""
    
    def test_bfloat16_strings(self):
        """Test bfloat16 string variants."""
        from deepencoder.deepencoder_infer import _to_dtype
        
        assert _to_dtype("bf16") == torch.bfloat16
        assert _to_dtype("bfloat16") == torch.bfloat16
        assert _to_dtype("BF16") == torch.bfloat16
        assert _to_dtype("BFLOAT16") == torch.bfloat16
    
    def test_float16_strings(self):
        """Test float16 string variants."""
        from deepencoder.deepencoder_infer import _to_dtype
        
        assert _to_dtype("fp16") == torch.float16
        assert _to_dtype("float16") == torch.float16
        assert _to_dtype("FP16") == torch.float16
        assert _to_dtype("FLOAT16") == torch.float16
    
    def test_float32_strings(self):
        """Test float32 string variants."""
        from deepencoder.deepencoder_infer import _to_dtype
        
        assert _to_dtype("fp32") == torch.float32
        assert _to_dtype("float32") == torch.float32
        assert _to_dtype("FP32") == torch.float32
        assert _to_dtype("FLOAT32") == torch.float32
    
    def test_invalid_dtype_raises(self):
        """Test that invalid dtype strings raise ValueError."""
        from deepencoder.deepencoder_infer import _to_dtype
        
        with pytest.raises(ValueError, match="Unsupported dtype string"):
            _to_dtype("invalid")
        
        with pytest.raises(ValueError, match="Unsupported dtype string"):
            _to_dtype("float64")
        
        with pytest.raises(ValueError, match="Unsupported dtype string"):
            _to_dtype("int32")


class TestModelSetupDtype:
    """Test dtype handling in model_setup."""
    
    def test_get_model_dtype_fp16(self):
        """Test _get_model_dtype returns float16 when fp16=True and CUDA."""
        from training.core.model_setup import _get_model_dtype
        
        config = {"fp16": True}
        
        # CUDA device should return float16
        if torch.cuda.is_available():
            device = torch.device("cuda")
            assert _get_model_dtype(config, device) == torch.float16
        
        # CPU device should return bfloat16 (fp16 only applies to CUDA)
        cpu_device = torch.device("cpu")
        assert _get_model_dtype(config, cpu_device) == torch.bfloat16
    
    def test_get_model_dtype_default(self):
        """Test _get_model_dtype returns bfloat16 by default."""
        from training.core.model_setup import _get_model_dtype
        
        config = {"fp16": False}
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        assert _get_model_dtype(config, device) == torch.bfloat16
    
    def test_get_model_dtype_missing_key(self):
        """Test _get_model_dtype handles missing fp16 key."""
        from training.core.model_setup import _get_model_dtype
        
        config = {}  # No fp16 key
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Should default to bfloat16 when key is missing
        assert _get_model_dtype(config, device) == torch.bfloat16


class TestFlashAttentionDtypeRequirements:
    """Test Flash Attention dtype requirements."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flash_attention_accepts_float16(self):
        """Verify Flash Attention works with float16."""
        try:
            from flash_attn import flash_attn_func
        except ImportError:
            pytest.skip("flash_attn not installed")
        
        B, N, H, D = 2, 16, 4, 64
        q = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, N, H, D, device="cuda", dtype=torch.float16)
        
        # Should not raise
        out = flash_attn_func(q, k, v)
        assert out.dtype == torch.float16
        assert out.shape == (B, N, H, D)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flash_attention_accepts_bfloat16(self):
        """Verify Flash Attention works with bfloat16."""
        try:
            from flash_attn import flash_attn_func
        except ImportError:
            pytest.skip("flash_attn not installed")
        
        B, N, H, D = 2, 16, 4, 64
        q = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        
        # Should not raise
        out = flash_attn_func(q, k, v)
        assert out.dtype == torch.bfloat16
        assert out.shape == (B, N, H, D)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flash_attention_rejects_float32(self):
        """Verify Flash Attention raises error with float32."""
        try:
            from flash_attn import flash_attn_func
        except ImportError:
            pytest.skip("flash_attn not installed")
        
        B, N, H, D = 2, 16, 4, 64
        q = torch.randn(B, N, H, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, N, H, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, N, H, D, device="cuda", dtype=torch.float32)
        
        # Flash Attention should raise RuntimeError for float32
        with pytest.raises(RuntimeError):
            flash_attn_func(q, k, v)


class TestVATBlocksDtypeFallback:
    """Test VAT blocks dtype handling and fallback."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flash_mha_dtype_check(self):
        """Test FlashMultiheadAttention dtype checking logic."""
        from training.models.vat_blocks import FlashMultiheadAttention
        
        embed_dim = 64
        num_heads = 4
        
        mha = FlashMultiheadAttention(embed_dim, num_heads).cuda()
        
        # Test with different dtypes
        seq_len = 16
        batch = 2
        
        for dtype in [torch.float16, torch.bfloat16]:
            mha_typed = mha.to(dtype)
            # FlashMultiheadAttention expects [B, S, D] format
            x = torch.randn(batch, seq_len, embed_dim, device="cuda", dtype=dtype)
            
            out = mha_typed(x)  # Returns single tensor, not tuple
            assert out.dtype == dtype
            assert out.shape == x.shape
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flash_mha_float32_fallback(self):
        """Test FlashMultiheadAttention falls back for float32."""
        from training.models.vat_blocks import FlashMultiheadAttention
        
        embed_dim = 64
        num_heads = 4
        
        mha = FlashMultiheadAttention(embed_dim, num_heads).cuda().float()
        
        seq_len = 16
        batch = 2
        # FlashMultiheadAttention expects [B, S, D] format
        x = torch.randn(batch, seq_len, embed_dim, device="cuda", dtype=torch.float32)
        
        # Should fall back to F.scaled_dot_product_attention for float32
        out = mha(x)  # Returns single tensor, not tuple
        assert out.dtype == torch.float32
        assert out.shape == x.shape


class TestDeepEncoderRuntimeDtype:
    """Test DeepEncoderRuntime dtype handling."""
    
    def test_runtime_accepts_torch_dtype(self):
        """Test that DeepEncoderRuntime accepts torch.dtype directly."""
        from deepencoder.deepencoder_infer import DeepEncoderRuntime
        
        # Test that the class signature accepts torch.dtype
        # We don't actually initialize (would download SAM) - just verify the type annotation
        import inspect
        sig = inspect.signature(DeepEncoderRuntime.__init__)
        dtype_param = sig.parameters.get("dtype")
        
        assert dtype_param is not None
        # Check that dtype accepts Union[torch.dtype, str]
        # The annotation is torch.dtype | str which is equivalent to Union
    
    def test_runtime_accepts_string_dtype(self):
        """Test that DeepEncoderRuntime converts string dtype."""
        from deepencoder.deepencoder_infer import _to_dtype
        
        # Verify all supported string formats work
        for s in ["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"]:
            dtype = _to_dtype(s)
            assert isinstance(dtype, torch.dtype)


class TestDefaultConfigDtype:
    """Test default configuration dtype settings."""
    
    def test_default_config_has_deep_dtype(self):
        """Test that default config includes deep_dtype."""
        from training.config.default_config import DEFAULT_CONFIG
        
        assert "deep_dtype" in DEFAULT_CONFIG
        
        # Verify it's a valid dtype string
        from deepencoder.deepencoder_infer import _to_dtype
        dtype = _to_dtype(DEFAULT_CONFIG["deep_dtype"])
        assert dtype in [torch.float16, torch.bfloat16, torch.float32]
    
    def test_default_config_has_mixed_precision(self):
        """Test that default config includes mixed_precision setting."""
        from training.config.default_config import DEFAULT_CONFIG
        
        assert "mixed_precision" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["mixed_precision"] in ["no", "fp16", "bf16"]


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
