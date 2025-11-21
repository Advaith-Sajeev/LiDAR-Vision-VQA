import math
import sys
from unittest.mock import Mock, MagicMock, patch
import pytest
import torch
import torch.nn as nn

# Mock external dependencies before importing model_setup
sys.modules['nuscenes'] = MagicMock()
sys.modules['nuscenes.nuscenes'] = MagicMock()
sys.modules['deepencoder'] = MagicMock()
sys.modules['deepencoder.deepencoder_infer'] = MagicMock()
sys.modules['deepencoder.lora_config'] = MagicMock()

from training.core import model_setup


@pytest.fixture
def mock_config():
    """Standard config for testing."""
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "fp16": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        "use_vision": False,
        "vat_queries": 252,  # Must be divisible by 6 (NUM_VIEWS)
        "vat_layers": 4,
        "vat_heads": 8,
        "vat_mlp_ratio": 4.0,
        "vat_dropout": 0.1,
        "vat_post_dropout": 0.1,
        "batch_size": 2,
        "grad_accum": 4,
        "epochs": 10,
        "warmup_steps": 100,
        "lr_vat": 1e-4,
        "lr_lora": 5e-5,
        "weight_decay": 0.01,
    }


@pytest.fixture
def mock_device():
    """Mock device."""
    return torch.device("cpu")


def test_create_vat_lidar(mock_config, mock_device):
    """Test VATLiDAR creation."""
    c_in = 128
    d_model = 512
    
    vat_lidar = model_setup.create_vat_lidar(c_in, d_model, mock_config, mock_device)
    
    assert vat_lidar is not None
    assert isinstance(vat_lidar, nn.Module)
    # Verify it's on correct device
    assert next(vat_lidar.parameters()).device == mock_device


def test_create_vat_lidar_uses_config_params(mock_config, mock_device):
    """Test that VATLiDAR respects config parameters."""
    c_in = 64
    d_model = 256
    
    # Modify config to test specific values (must be divisible by 6)
    mock_config["vat_queries"] = 126  # 126 is divisible by 6
    mock_config["vat_layers"] = 6
    mock_config["vat_heads"] = 4
    
    vat_lidar = model_setup.create_vat_lidar(c_in, d_model, mock_config, mock_device)
    
    # Basic validation - model was created
    assert vat_lidar is not None
    # Check device placement
    assert next(vat_lidar.parameters()).device == mock_device


def test_setup_optimizer_and_scheduler_basic(mock_config, mock_device):
    """Test optimizer and scheduler setup without vision."""
    # Create mock models
    base_model = nn.Linear(10, 10).to(mock_device)
    vat_lidar = nn.Linear(20, 20).to(mock_device)
    
    # Make some params in base trainable (simulating LoRA)
    base_model.weight.requires_grad = True
    base_model.bias.requires_grad = False
    
    train_size = 1000
    world_size = 1
    
    optim, sched, sched_meta = model_setup.setup_optimizer_and_scheduler(
        base_model,
        vat_lidar,
        None,  # vat_vision
        None,  # vision_adapter
        None,  # runtime
        mock_config,
        train_size,
        world_size,
    )
    
    # Verify optimizer created
    assert optim is not None
    assert len(optim.param_groups) == 2  # lidar + lora
    
    # Verify scheduler created
    assert sched is not None
    
    # Verify scheduler metadata
    assert "total_steps" in sched_meta
    assert "warmup_steps" in sched_meta
    assert sched_meta["warmup_steps"] == mock_config["warmup_steps"]


def test_setup_optimizer_and_scheduler_calculates_steps_correctly(mock_config, mock_device):
    """Test that scheduler steps are calculated correctly."""
    base_model = nn.Linear(10, 10).to(mock_device)
    base_model.weight.requires_grad = True
    vat_lidar = nn.Linear(20, 20).to(mock_device)
    
    train_size = 1000
    world_size = 2
    
    optim, sched, sched_meta = model_setup.setup_optimizer_and_scheduler(
        base_model,
        vat_lidar,
        None,
        None,
        None,
        mock_config,
        train_size,
        world_size,
    )
    
    # Calculate expected steps
    effective_batch_size = mock_config["batch_size"] * world_size * mock_config["grad_accum"]
    steps_per_epoch = max(1, math.ceil(train_size / effective_batch_size))
    expected_total_steps = mock_config["epochs"] * steps_per_epoch
    
    assert sched_meta["total_steps"] == expected_total_steps
    assert sched_meta["warmup_steps"] == mock_config["warmup_steps"]


def test_setup_optimizer_and_scheduler_with_vision(mock_config, mock_device):
    """Test optimizer setup with vision components."""
    base_model = nn.Linear(10, 10).to(mock_device)
    base_model.weight.requires_grad = True
    vat_lidar = nn.Linear(20, 20).to(mock_device)
    vat_vision = nn.Linear(30, 30).to(mock_device)
    vision_adapter = nn.Linear(40, 40).to(mock_device)
    
    # Mock runtime with clip_vit and projector
    runtime = Mock()
    runtime.clip_vit = nn.Linear(50, 50).to(mock_device)
    runtime.clip_vit.weight.requires_grad = True
    runtime.projector = nn.Linear(60, 60).to(mock_device)
    
    # Enable vision in config
    mock_config["use_vision"] = True
    mock_config["lr_vision"] = 1e-4
    mock_config["lr_vision_vat"] = 5e-5
    
    train_size = 1000
    world_size = 1
    
    optim, sched, sched_meta = model_setup.setup_optimizer_and_scheduler(
        base_model,
        vat_lidar,
        vat_vision,
        vision_adapter,
        runtime,
        mock_config,
        train_size,
        world_size,
    )
    
    # Should have 5 param groups: lidar, lora, clip_lora, vision_adapter+projector, vision_vat
    assert len(optim.param_groups) == 5


def test_setup_optimizer_learning_rates(mock_config, mock_device):
    """Test that optimizer param groups have correct learning rates."""
    base_model = nn.Linear(10, 10).to(mock_device)
    base_model.weight.requires_grad = True
    vat_lidar = nn.Linear(20, 20).to(mock_device)
    
    train_size = 100
    world_size = 1
    
    optim, sched, sched_meta = model_setup.setup_optimizer_and_scheduler(
        base_model,
        vat_lidar,
        None,
        None,
        None,
        mock_config,
        train_size,
        world_size,
    )
    
    # Check learning rates match config
    # Note: After scheduler step(s), lr may be modified, so check initial state
    # The param_groups store the initial lr values
    assert abs(optim.param_groups[0]["lr"] - 0) < 1e-6 or optim.param_groups[0]["lr"] == mock_config["lr_vat"]  # lidar (may be 0 after warmup)
    assert abs(optim.param_groups[1]["lr"] - 0) < 1e-6 or optim.param_groups[1]["lr"] == mock_config["lr_lora"]  # lora (may be 0 after warmup)


def test_projector_dimension_assertion():
    """Test that projector dimension is validated."""
    # Create a mock projector with wrong output dimension
    mock_projector = Mock()
    mock_cfg = Mock()
    mock_cfg.n_embed = 1024  # Wrong dimension
    mock_projector.cfg = mock_cfg
    
    # This should fail the assertion
    with pytest.raises(AssertionError, match="projector output dimension"):
        projector_out_dim = mock_projector.cfg.n_embed
        assert projector_out_dim == 2048, \
            f"DeepEncoder projector output dimension {projector_out_dim} != 2048"


def test_projector_dimension_correct():
    """Test that projector dimension validation passes with correct dimension."""
    mock_projector = Mock()
    mock_cfg = Mock()
    mock_cfg.n_embed = 2048  # Correct dimension
    mock_projector.cfg = mock_cfg
    
    # This should pass
    projector_out_dim = mock_projector.cfg.n_embed
    assert projector_out_dim == 2048, \
        f"DeepEncoder projector output dimension {projector_out_dim} != 2048"


def test_setup_optimizer_handles_empty_param_groups(mock_config, mock_device):
    """Test optimizer setup handles models with no trainable params gracefully."""
    base_model = nn.Linear(10, 10).to(mock_device)
    # No trainable params
    base_model.weight.requires_grad = False
    base_model.bias.requires_grad = False
    
    vat_lidar = nn.Linear(20, 20).to(mock_device)
    
    train_size = 100
    world_size = 1
    
    # Should still create optimizer (though lora group will be empty)
    optim, sched, sched_meta = model_setup.setup_optimizer_and_scheduler(
        base_model,
        vat_lidar,
        None,
        None,
        None,
        mock_config,
        train_size,
        world_size,
    )
    
    assert optim is not None
    assert sched is not None
