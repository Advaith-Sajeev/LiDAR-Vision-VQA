#!/usr/bin/env python3
"""
Tests for DeepEncoderLoRAConfig in lora_config.py

This module tests:
- Default values for all config fields
- materialize_target_modules() method
- QLoRA settings validation
- Type validation for config fields
"""

import sys
import os
from pathlib import Path

# Add src/ directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest


class TestLoRAConfigDefaults:
    """Test default values for DeepEncoderLoRAConfig."""
    
    def test_default_values(self):
        """Test all default values are set correctly."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig()
        
        # Check defaults
        assert config.enabled is False, "LoRA should be disabled by default"
        assert config.r == 8, "Default rank should be 8"
        assert config.lora_alpha == 16, "Default lora_alpha should be 16"
        assert config.lora_dropout == 0.0, "Default dropout should be 0.0"
        assert config.bias == "none", "Default bias should be 'none'"
        assert config.target_modules is None, "Default target_modules should be None"
    
    def test_qlora_defaults(self):
        """Test QLoRA-specific default values."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig()
        
        assert config.use_qlora is False, "QLoRA should be disabled by default"
        assert config.qlora_quant_type == "nf4", "Default quant type should be 'nf4'"
        assert config.qlora_double_quant is True, "Double quantization should be enabled by default"
        assert config.qlora_compute_dtype == "bfloat16", "Default compute dtype should be 'bfloat16'"


class TestLoRAConfigCustomValues:
    """Test custom values for DeepEncoderLoRAConfig."""
    
    def test_custom_rank(self):
        """Test custom rank value."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(r=16)
        assert config.r == 16
    
    def test_custom_alpha(self):
        """Test custom alpha value."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(lora_alpha=32)
        assert config.lora_alpha == 32
    
    def test_custom_dropout(self):
        """Test custom dropout value."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(lora_dropout=0.1)
        assert config.lora_dropout == 0.1
    
    def test_enabled_config(self):
        """Test enabled LoRA config."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(enabled=True)
        assert config.enabled is True
    
    def test_custom_target_modules(self):
        """Test custom target modules list."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        targets = ["qkv_proj", "out_proj", "mlp.fc1"]
        config = DeepEncoderLoRAConfig(target_modules=targets)
        
        assert config.target_modules == targets
    
    def test_bias_options(self):
        """Test different bias options."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        for bias in ["none", "lora_only", "all"]:
            config = DeepEncoderLoRAConfig(bias=bias)
            assert config.bias == bias


class TestMaterializeTargetModules:
    """Test materialize_target_modules() method."""
    
    def test_materialize_with_explicit_targets(self):
        """Test materialize returns explicit target modules."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        targets = ["qkv_proj", "out_proj"]
        config = DeepEncoderLoRAConfig(target_modules=targets)
        
        result = config.materialize_target_modules()
        
        assert result == targets
        assert isinstance(result, list)
    
    def test_materialize_with_none_and_fallback(self):
        """Test materialize uses fallback when target_modules is None."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(target_modules=None)
        fallback = ["mlp.fc1", "mlp.fc2"]
        
        result = config.materialize_target_modules(fallback=fallback)
        
        assert result == fallback
    
    def test_materialize_with_none_and_no_fallback(self):
        """Test materialize returns empty list when no fallback provided."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(target_modules=None)
        
        result = config.materialize_target_modules()
        
        assert result == []
    
    def test_materialize_returns_list_copy(self):
        """Test materialize returns a list (not the original)."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        targets = ["qkv_proj"]
        config = DeepEncoderLoRAConfig(target_modules=targets)
        
        result = config.materialize_target_modules()
        
        # Modify result and check original is unchanged
        result.append("new_module")
        assert "new_module" not in config.target_modules


class TestQLoRAConfig:
    """Test QLoRA-specific configuration."""
    
    def test_qlora_enabled(self):
        """Test QLoRA enabled configuration."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(
            enabled=True,
            use_qlora=True,
            qlora_quant_type="nf4",
        )
        
        assert config.enabled is True
        assert config.use_qlora is True
        assert config.qlora_quant_type == "nf4"
    
    def test_qlora_fp4_quant_type(self):
        """Test QLoRA with fp4 quantization."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(
            use_qlora=True,
            qlora_quant_type="fp4",
        )
        
        assert config.qlora_quant_type == "fp4"
    
    def test_qlora_double_quant_disabled(self):
        """Test QLoRA with double quantization disabled."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(
            use_qlora=True,
            qlora_double_quant=False,
        )
        
        assert config.qlora_double_quant is False
    
    def test_qlora_compute_dtype_float16(self):
        """Test QLoRA with float16 compute dtype."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(
            use_qlora=True,
            qlora_compute_dtype="float16",
        )
        
        assert config.qlora_compute_dtype == "float16"


class TestLoRAConfigWithCLIPTargets:
    """Test LoRA config integration with CLIP target modules."""
    
    def test_with_clip_default_targets(self):
        """Test config works with CLIP's default LoRA targets."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        from deepencoder.clip_sdpa import clip_l_lora_default_targets
        
        default_targets = list(clip_l_lora_default_targets())
        
        config = DeepEncoderLoRAConfig(
            enabled=True,
            target_modules=default_targets,
        )
        
        assert config.target_modules == default_targets
        assert "qkv_proj" in config.target_modules
        assert "out_proj" in config.target_modules
    
    def test_materialize_with_clip_fallback(self):
        """Test materialize uses CLIP defaults as fallback."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        from deepencoder.clip_sdpa import clip_l_lora_default_targets
        
        config = DeepEncoderLoRAConfig(
            enabled=True,
            target_modules=None,
        )
        
        fallback = list(clip_l_lora_default_targets())
        result = config.materialize_target_modules(fallback=fallback)
        
        assert result == fallback


class TestLoRAConfigDataclass:
    """Test dataclass behavior of DeepEncoderLoRAConfig."""
    
    def test_config_is_dataclass(self):
        """Test config is a proper dataclass."""
        from dataclasses import is_dataclass
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        assert is_dataclass(DeepEncoderLoRAConfig)
    
    def test_config_repr(self):
        """Test config has readable repr."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(enabled=True, r=16)
        repr_str = repr(config)
        
        assert "enabled=True" in repr_str
        assert "r=16" in repr_str
    
    def test_config_equality(self):
        """Test config equality comparison."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config1 = DeepEncoderLoRAConfig(r=8, lora_alpha=16)
        config2 = DeepEncoderLoRAConfig(r=8, lora_alpha=16)
        config3 = DeepEncoderLoRAConfig(r=16, lora_alpha=32)
        
        assert config1 == config2
        assert config1 != config3


class TestLoRAConfigRealistic:
    """Test realistic LoRA configuration scenarios."""
    
    def test_typical_lora_config(self):
        """Test typical LoRA configuration for CLIP training."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(
            enabled=True,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["qkv_proj", "out_proj", "mlp.fc1", "mlp.fc2"],
        )
        
        assert config.enabled is True
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert len(config.target_modules) == 4
    
    def test_high_rank_lora_config(self):
        """Test high-rank LoRA for more capacity."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(
            enabled=True,
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
        )
        
        assert config.r == 64
        assert config.lora_alpha == 128
    
    def test_minimal_lora_config(self):
        """Test minimal LoRA with low rank for efficiency."""
        from deepencoder.lora_config import DeepEncoderLoRAConfig
        
        config = DeepEncoderLoRAConfig(
            enabled=True,
            r=4,
            lora_alpha=8,
        )
        
        assert config.r == 4
        assert config.lora_alpha == 8


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
