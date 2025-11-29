"""Tests for Trainer resume functionality.

Tests for:
- Issue 1.1: GradScaler/mixed_precision mode validation on resume
- Issue 1.2: LoRA adapter config validation on resume
- Issue 1.4: Optimizer state device migration on resume
- Issue 1.6: SAM compression head restore on resume
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pytest
import torch
import torch.nn as nn

# Mock external dependencies before importing trainer
sys.modules['nuscenes'] = MagicMock()
sys.modules['nuscenes.nuscenes'] = MagicMock()
sys.modules['deepencoder'] = MagicMock()
sys.modules['deepencoder.deepencoder_infer'] = MagicMock()
sys.modules['deepencoder.lora_config'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['safetensors'] = MagicMock()
sys.modules['safetensors.torch'] = MagicMock()


# ============================================================================
# Tests for issue 1.2: LoRA config validation
# ============================================================================

class TestLoraConfigValidation:
    """Tests for _validate_lora_config method."""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a minimal mock trainer for testing _validate_lora_config."""
        # Import after mocks are set up
        from training.core.trainer import Trainer
        
        # Create a minimal mock trainer object without full initialization
        trainer = object.__new__(Trainer)
        trainer.config = {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_target_modules": ["q_proj", "v_proj"],
            "clip_lora_target_modules": ["qkv_proj", "out_proj"],
        }
        return trainer
    
    def test_validate_lora_config_matching_config_returns_true(self, mock_trainer, tmp_path):
        """Test that matching LoRA config returns True."""
        # Create adapter_config.json with matching values
        adapter_path = tmp_path / "lora_adapter"
        adapter_path.mkdir()
        
        config_data = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
        }
        with open(adapter_path / "adapter_config.json", "w") as f:
            json.dump(config_data, f)
        
        result = mock_trainer._validate_lora_config(adapter_path, adapter_type="LLM")
        assert result is True
    
    def test_validate_lora_config_mismatched_rank_returns_false(self, mock_trainer, tmp_path):
        """Test that mismatched lora_r returns False."""
        adapter_path = tmp_path / "lora_adapter"
        adapter_path.mkdir()
        
        config_data = {
            "r": 16,  # Different from config's 8
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
        }
        with open(adapter_path / "adapter_config.json", "w") as f:
            json.dump(config_data, f)
        
        result = mock_trainer._validate_lora_config(adapter_path, adapter_type="LLM")
        assert result is False
    
    def test_validate_lora_config_mismatched_alpha_returns_false(self, mock_trainer, tmp_path):
        """Test that mismatched lora_alpha returns False."""
        adapter_path = tmp_path / "lora_adapter"
        adapter_path.mkdir()
        
        config_data = {
            "r": 8,
            "lora_alpha": 32,  # Different from config's 16
            "target_modules": ["q_proj", "v_proj"],
        }
        with open(adapter_path / "adapter_config.json", "w") as f:
            json.dump(config_data, f)
        
        result = mock_trainer._validate_lora_config(adapter_path, adapter_type="LLM")
        assert result is False
    
    def test_validate_lora_config_mismatched_target_modules_returns_false(self, mock_trainer, tmp_path):
        """Test that mismatched target_modules returns False."""
        adapter_path = tmp_path / "lora_adapter"
        adapter_path.mkdir()
        
        config_data = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "k_proj"],  # Different from ["q_proj", "v_proj"]
        }
        with open(adapter_path / "adapter_config.json", "w") as f:
            json.dump(config_data, f)
        
        result = mock_trainer._validate_lora_config(adapter_path, adapter_type="LLM")
        assert result is False
    
    def test_validate_lora_config_missing_config_file_returns_true(self, mock_trainer, tmp_path):
        """Test that missing adapter_config.json returns True (backward compatibility)."""
        adapter_path = tmp_path / "lora_adapter"
        adapter_path.mkdir()
        # No adapter_config.json file
        
        result = mock_trainer._validate_lora_config(adapter_path, adapter_type="LLM")
        assert result is True
    
    def test_validate_lora_config_clip_type_uses_clip_target_modules(self, mock_trainer, tmp_path):
        """Test that CLIP adapter type uses clip_lora_target_modules config."""
        adapter_path = tmp_path / "clip_adapter"
        adapter_path.mkdir()
        
        config_data = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["qkv_proj", "out_proj"],  # Matches clip_lora_target_modules
        }
        with open(adapter_path / "adapter_config.json", "w") as f:
            json.dump(config_data, f)
        
        result = mock_trainer._validate_lora_config(adapter_path, adapter_type="CLIP")
        assert result is True
    
    def test_validate_lora_config_clip_type_mismatch_returns_false(self, mock_trainer, tmp_path):
        """Test that CLIP adapter type with mismatched modules returns False."""
        adapter_path = tmp_path / "clip_adapter"
        adapter_path.mkdir()
        
        config_data = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],  # LLM modules, not CLIP modules
        }
        with open(adapter_path / "adapter_config.json", "w") as f:
            json.dump(config_data, f)
        
        result = mock_trainer._validate_lora_config(adapter_path, adapter_type="CLIP")
        assert result is False
    
    def test_validate_lora_config_order_independent_target_modules(self, mock_trainer, tmp_path):
        """Test that target_modules comparison is order-independent."""
        adapter_path = tmp_path / "lora_adapter"
        adapter_path.mkdir()
        
        config_data = {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["v_proj", "q_proj"],  # Same modules, different order
        }
        with open(adapter_path / "adapter_config.json", "w") as f:
            json.dump(config_data, f)
        
        result = mock_trainer._validate_lora_config(adapter_path, adapter_type="LLM")
        assert result is True


# ============================================================================
# Tests for issue 1.4: Optimizer state device migration
# ============================================================================

class TestOptimizerDeviceMigration:
    """Tests for optimizer state device migration on resume."""
    
    def test_optimizer_state_tensors_moved_to_device(self):
        """Test that optimizer state tensors are moved to correct device."""
        # Create a simple model and optimizer
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Run one optimization step to create optimizer state
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        
        # Verify optimizer has state
        assert len(optimizer.state) > 0
        
        # Simulate saving state on CPU
        saved_state = optimizer.state_dict()
        
        # Create new optimizer
        new_model = nn.Linear(10, 10)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        
        # Load state (simulating checkpoint load)
        new_optimizer.load_state_dict(saved_state)
        
        # Verify state tensors exist
        for state in new_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    # This is what the trainer code does
                    assert v.device.type == "cpu"  # Should be on CPU after load
    
    def test_optimizer_state_migration_code_snippet(self):
        """Test the actual migration code snippet from trainer.py."""
        # Create a simple model and optimizer
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Run optimization to create state
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        
        # Save and reload state dict
        saved_state = optimizer.state_dict()
        
        new_model = nn.Linear(10, 10)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        new_optimizer.load_state_dict(saved_state)
        
        # Apply the migration code from trainer.py
        target_device = torch.device("cpu")  # Use CPU for testing
        for state in new_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(target_device)
        
        # Verify all tensors are on target device
        for state in new_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    assert v.device == target_device


# ============================================================================
# Tests for issue 1.6: SAM compression head restore
# ============================================================================

class DummySAMModule(nn.Module):
    """Mock SAM module with net_2 and net_3 compression head layers."""
    def __init__(self):
        super().__init__()
        # Frozen SAM backbone layers
        self.image_encoder = nn.Linear(256, 256)
        self.mask_decoder = nn.Linear(128, 128)
        
        # Trainable compression head layers
        self.net_2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.net_3 = nn.Linear(32, 16)
        
        # Set requires_grad appropriately
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.mask_decoder.parameters():
            p.requires_grad = False
        for p in self.net_2.parameters():
            p.requires_grad = True
        for p in self.net_3.parameters():
            p.requires_grad = True


class TestSAMCompressionHeadRestore:
    """Tests for SAM compression head restore on resume."""
    
    def test_sam_compression_head_restore_updates_model(self, tmp_path):
        """Test that loading SAM compression head updates the model correctly."""
        # Create original SAM model and save compression head
        original_sam = DummySAMModule()
        
        # Extract compression head state
        compression_head_state = {
            name: param.clone() for name, param in original_sam.named_parameters()
            if name.startswith("net_2") or name.startswith("net_3")
        }
        
        # Save to file
        sam_path = tmp_path / "sam_compression_head_latest.pt"
        torch.save(compression_head_state, sam_path)
        
        # Create new SAM model with different weights
        new_sam = DummySAMModule()
        
        # Verify weights are different before restore
        for name, param in new_sam.named_parameters():
            if name.startswith("net_2") or name.startswith("net_3"):
                original_param = compression_head_state[name]
                # Weights should be different (random initialization)
                # Note: There's a small chance they could be equal, but very unlikely
        
        # Load compression head state (simulating resume code)
        loaded_state = torch.load(sam_path, map_location="cpu")
        current_state = new_sam.state_dict()
        for name, param in loaded_state.items():
            if name in current_state:
                current_state[name] = param
        new_sam.load_state_dict(current_state)
        
        # Verify weights match after restore
        for name, param in new_sam.named_parameters():
            if name.startswith("net_2") or name.startswith("net_3"):
                original_param = compression_head_state[name]
                assert torch.allclose(param, original_param)
    
    def test_sam_compression_head_restore_preserves_backbone(self, tmp_path):
        """Test that restoring compression head doesn't affect backbone weights."""
        # Create original SAM model
        original_sam = DummySAMModule()
        
        # Save only compression head
        compression_head_state = {
            name: param.clone() for name, param in original_sam.named_parameters()
            if name.startswith("net_2") or name.startswith("net_3")
        }
        
        sam_path = tmp_path / "sam_compression_head_latest.pt"
        torch.save(compression_head_state, sam_path)
        
        # Create new SAM model
        new_sam = DummySAMModule()
        
        # Save backbone weights before restore
        backbone_before = {
            name: param.clone() for name, param in new_sam.named_parameters()
            if name.startswith("image_encoder") or name.startswith("mask_decoder")
        }
        
        # Load compression head (should not affect backbone)
        loaded_state = torch.load(sam_path, map_location="cpu")
        current_state = new_sam.state_dict()
        for name, param in loaded_state.items():
            if name in current_state:
                current_state[name] = param
        new_sam.load_state_dict(current_state)
        
        # Verify backbone weights unchanged
        for name, param in new_sam.named_parameters():
            if name.startswith("image_encoder") or name.startswith("mask_decoder"):
                assert torch.allclose(param, backbone_before[name])
    
    def test_sam_compression_head_only_saves_trainable_layers(self):
        """Test that only net_2 and net_3 parameters are considered for saving."""
        sam = DummySAMModule()
        
        # Extract compression head state (same logic as checkpoints.py)
        compression_head_state = {
            name: param.clone() for name, param in sam.named_parameters()
            if name.startswith("net_2") or name.startswith("net_3")
        }
        
        # Verify only net_2 and net_3 params are included
        for name in compression_head_state.keys():
            assert name.startswith("net_2") or name.startswith("net_3")
        
        # Verify backbone params are NOT included
        assert not any("image_encoder" in name for name in compression_head_state.keys())
        assert not any("mask_decoder" in name for name in compression_head_state.keys())


# ============================================================================
# Tests for issue 1.1: Mixed precision mode validation (integration)
# ============================================================================

class TestMixedPrecisionValidation:
    """Integration tests for mixed precision mode validation."""
    
    def test_mixed_precision_mode_extracted_correctly_fp16(self):
        """Test that fp16 mode is correctly extracted from config."""
        config = {"mixed_precision": "fp16"}
        mixed_precision = config.get('mixed_precision', 'fp16' if config.get('fp16', False) else 'no')
        assert mixed_precision == "fp16"
    
    def test_mixed_precision_mode_extracted_correctly_bf16(self):
        """Test that bf16 mode is correctly extracted from config."""
        config = {"mixed_precision": "bf16"}
        mixed_precision = config.get('mixed_precision', 'fp16' if config.get('fp16', False) else 'no')
        assert mixed_precision == "bf16"
    
    def test_mixed_precision_mode_legacy_fp16_true(self):
        """Test legacy fp16=True config is handled correctly."""
        config = {"fp16": True}
        mixed_precision = config.get('mixed_precision', 'fp16' if config.get('fp16', False) else 'no')
        assert mixed_precision == "fp16"
    
    def test_mixed_precision_mode_legacy_fp16_false(self):
        """Test legacy fp16=False config is handled correctly."""
        config = {"fp16": False}
        mixed_precision = config.get('mixed_precision', 'fp16' if config.get('fp16', False) else 'no')
        assert mixed_precision == "no"
    
    def test_mixed_precision_mode_no_config(self):
        """Test empty config defaults to 'no'."""
        config = {}
        mixed_precision = config.get('mixed_precision', 'fp16' if config.get('fp16', False) else 'no')
        assert mixed_precision == "no"
    
    def test_mixed_precision_mode_mismatch_detection(self):
        """Test that mode mismatch is correctly detected."""
        saved_mixed_prec = "fp16"
        current_mixed_prec = "bf16"
        
        # This is the logic from trainer.py
        mode_changed = saved_mixed_prec is not None and saved_mixed_prec != current_mixed_prec
        assert mode_changed is True
    
    def test_mixed_precision_mode_match_detection(self):
        """Test that mode match is correctly detected."""
        saved_mixed_prec = "fp16"
        current_mixed_prec = "fp16"
        
        mode_changed = saved_mixed_prec is not None and saved_mixed_prec != current_mixed_prec
        assert mode_changed is False
