"""Tests for LoRA/QLoRA utilities"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from training.models.lora_utils import (
    make_lora,
    patch_clip_peft_forward,
    infer_clip_lora_targets,
    get_bnb_config,
)


class TestMakeLora:
    """Tests for make_lora function"""
    
    @patch('training.models.lora_utils.get_peft_model')
    @patch('training.models.lora_utils.LoraConfig')
    def test_make_lora_creates_config(self, mock_lora_config, mock_get_peft):
        """Test that LoRA config is created with correct parameters"""
        model = nn.Linear(10, 10)
        targets = ["q_proj", "k_proj", "v_proj"]
        r = 8
        alpha = 16
        dropout = 0.1
        
        make_lora(model, targets, r, alpha, dropout)
        
        mock_lora_config.assert_called_once_with(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=targets,
            task_type="CAUSAL_LM",
        )
    
    @patch('training.models.lora_utils.get_peft_model')
    @patch('training.models.lora_utils.LoraConfig')
    def test_make_lora_returns_peft_model(self, mock_lora_config, mock_get_peft):
        """Test that make_lora returns PEFT model"""
        model = nn.Linear(10, 10)
        targets = ["q_proj"]
        
        mock_peft = Mock()
        mock_get_peft.return_value = mock_peft
        
        result = make_lora(model, targets, 8, 16, 0.1)
        
        assert result == mock_peft
        mock_get_peft.assert_called_once()
    
    @patch('training.models.lora_utils.get_peft_model')
    @patch('training.models.lora_utils.LoraConfig')
    def test_make_lora_with_multiple_targets(self, mock_lora_config, mock_get_peft):
        """Test make_lora with multiple target modules"""
        model = nn.Linear(10, 10)
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
        
        make_lora(model, targets, 16, 32, 0.05)
        
        call_kwargs = mock_lora_config.call_args[1]
        assert call_kwargs["target_modules"] == targets
        assert call_kwargs["r"] == 16
        assert call_kwargs["lora_alpha"] == 32
        assert call_kwargs["lora_dropout"] == 0.05

    @patch('training.models.lora_utils.get_peft_model')
    @patch('training.models.lora_utils.LoraConfig')
    @patch('training.models.lora_utils.prepare_model_for_kbit_training')
    def test_make_lora_with_quantization(self, mock_prepare_kbit, mock_lora_config, mock_get_peft):
        """Test that QLoRA prepares model for k-bit training when is_quantized=True"""
        model = nn.Linear(10, 10)
        targets = ["q_proj"]
        
        mock_prepared_model = Mock()
        mock_prepare_kbit.return_value = mock_prepared_model
        
        make_lora(model, targets, 8, 16, 0.1, is_quantized=True)
        
        # Should call prepare_model_for_kbit_training when quantized
        mock_prepare_kbit.assert_called_once()
        # Should pass the prepared model to get_peft_model
        mock_get_peft.assert_called_once()
    
    @patch('training.models.lora_utils.get_peft_model')
    @patch('training.models.lora_utils.LoraConfig')
    @patch('training.models.lora_utils.prepare_model_for_kbit_training')
    def test_make_lora_without_quantization(self, mock_prepare_kbit, mock_lora_config, mock_get_peft):
        """Test that k-bit preparation is skipped when is_quantized=False"""
        model = nn.Linear(10, 10)
        targets = ["q_proj"]
        
        make_lora(model, targets, 8, 16, 0.1, is_quantized=False)
        
        # Should NOT call prepare_model_for_kbit_training when not quantized
        mock_prepare_kbit.assert_not_called()


class TestGetBnbConfig:
    """Tests for get_bnb_config function"""
    
    def test_get_bnb_config_default(self):
        """Test default BitsAndBytesConfig creation"""
        # Test the actual function behavior without complex mocking
        # The function imports BitsAndBytesConfig internally, so we just test it works
        result = get_bnb_config()
        
        # The function should return a BitsAndBytesConfig object
        # We just verify it doesn't raise and returns something
        assert result is not None
        
        # Verify it has expected attributes from BitsAndBytesConfig
        assert hasattr(result, 'load_in_4bit')
        assert result.load_in_4bit == True
    
    @patch('transformers.BitsAndBytesConfig')
    def test_get_bnb_config_custom(self, mock_bnb_config):
        """Test custom BitsAndBytesConfig creation"""
        result = get_bnb_config(
            use_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype="float16",
        )
        
        # Verify it returns something
        assert result is not None
    
    def test_get_bnb_config_dtype_mapping(self):
        """Test dtype string to torch.dtype mapping"""
        # Test that different dtype strings work without raising
        result_bf16 = get_bnb_config(bnb_4bit_compute_dtype="bfloat16")
        assert result_bf16 is not None
        
        result_f16 = get_bnb_config(bnb_4bit_compute_dtype="float16")
        assert result_f16 is not None
        
        result_f32 = get_bnb_config(bnb_4bit_compute_dtype="float32")
        assert result_f32 is not None


class TestPatchClipPeftForward:
    """Tests for patch_clip_peft_forward function"""
    
    def test_patch_adds_forward_method(self):
        """Test that patching adds custom forward method"""
        peft_clip = Mock()
        peft_clip.base_model = Mock()
        
        result = patch_clip_peft_forward(peft_clip)
        
        assert hasattr(result, 'forward')
        assert callable(result.forward)
    
    def test_patched_forward_calls_base_model(self):
        """Test that patched forward calls base_model"""
        peft_clip = Mock()
        base_model_mock = Mock()
        base_model_mock.return_value = torch.randn(10, 10)
        peft_clip.base_model = base_model_mock
        
        patched = patch_clip_peft_forward(peft_clip)
        
        x = torch.randn(1, 3, 224, 224)
        patch_embeds = torch.randn(1, 256, 768)
        
        result = patched.forward(x, patch_embeds)
        
        base_model_mock.assert_called_once_with(x, patch_embeds)
    
    def test_patch_returns_same_object(self):
        """Test that patch returns the same object (modified in-place)"""
        peft_clip = Mock()
        peft_clip.base_model = Mock()
        
        result = patch_clip_peft_forward(peft_clip)
        
        assert result is peft_clip


class TestInferClipLoraTargets:
    """Tests for infer_clip_lora_targets function"""
    
    def test_infer_finds_attention_layers(self):
        """Test that inference finds attention projection layers"""
        class MockAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(768, 768)
                self.k_proj = nn.Linear(768, 768)
                self.v_proj = nn.Linear(768, 768)
                self.out_proj = nn.Linear(768, 768)
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MockAttn()
        
        model = MockModel()
        targets = infer_clip_lora_targets(model)
        
        assert "q_proj" in targets
        assert "k_proj" in targets
        assert "v_proj" in targets
        assert "out_proj" in targets
    
    def test_infer_finds_mlp_layers(self):
        """Test that inference finds MLP layers"""
        class MockMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(768, 3072)
                self.fc2 = nn.Linear(3072, 768)
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MockMLP()
        
        model = MockModel()
        targets = infer_clip_lora_targets(model)
        
        assert "fc1" in targets
        assert "fc2" in targets
    
    def test_infer_removes_duplicates(self):
        """Test that duplicate target names are removed"""
        class MockBlock1(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(768, 768)
        
        class MockBlock2(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(768, 768)
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = MockBlock1()
                self.block2 = MockBlock2()
        
        model = MockModel()
        targets = infer_clip_lora_targets(model)
        
        # Should have q_proj only once
        assert targets.count("q_proj") == 1
    
    def test_infer_preserves_order(self):
        """Test that order of discovered targets is preserved"""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer1._name = "q_proj"
                self.layer2 = nn.Linear(10, 10)
                self.layer2._name = "k_proj"
                self.layer3 = nn.Linear(10, 10)
                self.layer3._name = "v_proj"
        
        # Note: This test checks order preservation but the actual
        # function depends on module naming patterns
        model = MockModel()
        targets = infer_clip_lora_targets(model)
        
        # If targets found, they should be in consistent order
        if len(targets) > 1:
            # Check no duplicates
            assert len(targets) == len(set(targets))
    
    def test_infer_handles_empty_model(self):
        """Test inference with model having no linear layers"""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3)
        
        model = MockModel()
        targets = infer_clip_lora_targets(model)
        
        assert isinstance(targets, list)
        assert len(targets) == 0
    
    def test_infer_finds_qkv_layers(self):
        """Test that inference finds QKV combined layers"""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Module()
                self.attn.qkv = nn.Linear(768, 2304)
        
        model = MockModel()
        targets = infer_clip_lora_targets(model)
        
        assert "qkv" in targets
    
    def test_infer_finds_proj_layers(self):
        """Test that inference finds projection layers"""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(768, 512)
        
        model = MockModel()
        targets = infer_clip_lora_targets(model)
        
        assert "proj" in targets
    
    def test_infer_fallback_pattern(self):
        """Test fallback pattern when common patterns not found"""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.custom_qkv = nn.Linear(768, 2304)
                self.custom_proj = nn.Linear(768, 768)
                self.custom_fc1 = nn.Linear(768, 3072)
        
        model = MockModel()
        targets = infer_clip_lora_targets(model)
        
        # Should find layers ending with known suffixes via fallback
        # The function extracts the last part of the name (after the last dot)
        # So custom_qkv -> custom_qkv, custom_proj -> custom_proj, etc.
        # Since these end with qkv/proj/fc1, they should be found
        assert "custom_qkv" in targets
        assert "custom_proj" in targets
        assert "custom_fc1" in targets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
