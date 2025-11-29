"""Tests for epoch-level checkpoint utilities"""

import os
from pathlib import Path

import pytest
import torch

from training.utils import checkpoints


class DummyModule(torch.nn.Module):
    """Simple nn.Module used for state_dict-based saves."""
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)


class DummySavePretrainedModule(torch.nn.Module):
    """Module that mimics HuggingFace-style save_pretrained()."""
    def __init__(self):
        super().__init__()
        self.saved_paths = []

    def save_pretrained(self, path, save_embedding_layers=False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.saved_paths.append(path)


class DummyOpt:
    def __init__(self):
        self._state = {"opt": 1}

    def state_dict(self):
        return self._state


class DummySched:
    def __init__(self):
        self._state = {"sched": 1}

    def state_dict(self):
        return self._state


class DummyScaler:
    def __init__(self):
        self._state = {"scaler": 1}

    def state_dict(self):
        return self._state


def _assert_exists(path: Path):
    assert path.exists(), f"Expected to exist: {path}"


def _assert_not_exists(path: Path):
    assert not path.exists(), f"Expected to be removed: {path}"


def test_save_state_latest_creates_expected_files_and_state(tmp_path, monkeypatch):
    """Test that save_state creates all expected checkpoint files."""
    # Ensure CPU-only branch for RNG
    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    vat_lidar = DummyModule()
    vat_vision = DummyModule()
    vision_adapter = DummyModule()
    projector = DummyModule()
    base = DummySavePretrainedModule()
    clip_vit = DummySavePretrainedModule()
    optim = DummyOpt()
    sched = DummySched()
    scaler = DummyScaler()

    checkpoints.save_state(
        out_dir=tmp_path,
        tag="latest",
        step=5,
        epoch=2,
        global_step=4,
        epoch_losses=[0.1, 0.2],
        best_loss=0.05,
        best_step=99,
        optim=optim,
        sched=sched,
        scaler=scaler,
        vat_lidar=vat_lidar,
        vat_vision=vat_vision,
        base=base,
        clip_vit=clip_vit,
        vision_adapter=vision_adapter,
        projector=projector,
        sched_meta={"foo": "bar"},
        config={"lr": 1e-4},
        val_losses=[0.01],
        val_epochs=[1],
    )

    # Check checkpoint artifacts
    _assert_exists(tmp_path / "vat_lidar_latest.pt")
    _assert_exists(tmp_path / "vat_vision_latest.pt")
    _assert_exists(tmp_path / "vision_adapter_latest.pt")
    _assert_exists(tmp_path / "projector_latest.pt")
    _assert_exists(tmp_path / "qwen2_lora_adapter_latest")
    _assert_exists(tmp_path / "clip_lora_adapter_latest")

    state_path = tmp_path / "training_state_latest.pt"
    _assert_exists(state_path)

    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")

    # Core metadata
    assert state["epoch"] == 2
    assert state["global_step"] == 4
    assert state["epoch_losses"] == [0.1, 0.2]
    assert state["best_loss"] == 0.05
    assert state["best_step"] == 99
    assert state["val_losses"] == [0.01]
    assert state["val_epochs"] == [1]

    # Optimizer, scheduler, scaler state
    assert state["optimizer"] == optim.state_dict()
    assert state["scheduler"] == sched.state_dict()
    assert state["scaler"] == scaler.state_dict()

    # RNG + aux
    assert "rng" in state
    assert "sched_meta" in state and state["sched_meta"]["foo"] == "bar"
    assert "config" in state and state["config"]["lr"] == 1e-4


def test_save_state_handles_optional_modules(tmp_path, monkeypatch):
    """Test that save_state handles None for optional modules."""
    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    vat_lidar = DummyModule()
    base = DummySavePretrainedModule()
    optim = DummyOpt()
    sched = DummySched()

    checkpoints.save_state(
        out_dir=tmp_path,
        tag="latest",
        step=10,
        epoch=1,
        global_step=10,
        epoch_losses=[0.5],
        best_loss=0.5,
        best_step=None,
        optim=optim,
        sched=sched,
        scaler=None,
        vat_lidar=vat_lidar,
        vat_vision=None,
        base=base,
        clip_vit=None,
        vision_adapter=None,
        projector=None,
        sched_meta={},
        config={},
        val_losses=None,
        val_epochs=None,
    )

    _assert_exists(tmp_path / "vat_lidar_latest.pt")
    _assert_exists(tmp_path / "qwen2_lora_adapter_latest")
    _assert_exists(tmp_path / "training_state_latest.pt")

    _assert_not_exists(tmp_path / "vat_vision_latest.pt")
    _assert_not_exists(tmp_path / "vision_adapter_latest.pt")
    _assert_not_exists(tmp_path / "projector_latest.pt")
    _assert_not_exists(tmp_path / "clip_lora_adapter_latest")


def test_save_state_unwraps_ddp_wrapped_models(tmp_path, monkeypatch):
    """Ensure DDP-wrapped models are unwrapped via .module before saving."""
    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    class DummyDDP(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    monkeypatch.setattr(checkpoints.nn.parallel, "DistributedDataParallel", DummyDDP)

    vat_lidar = DummyDDP(DummyModule())
    base = DummyDDP(DummySavePretrainedModule())
    optim = DummyOpt()
    sched = DummySched()

    checkpoints.save_state(
        out_dir=tmp_path,
        tag="latest",
        step=0,
        epoch=0,
        global_step=0,
        epoch_losses=[],
        best_loss=0.0,
        best_step=None,
        optim=optim,
        sched=sched,
        scaler=None,
        vat_lidar=vat_lidar,
        vat_vision=None,
        base=base,
        clip_vit=None,
        vision_adapter=None,
        projector=None,
        sched_meta={},
        config={},
    )

    _assert_exists(tmp_path / "vat_lidar_latest.pt")
    _assert_exists(tmp_path / "qwen2_lora_adapter_latest")
    _assert_exists(tmp_path / "training_state_latest.pt")


def test_try_load_state_finds_latest(tmp_path):
    """Test that try_load_state finds the latest checkpoint."""
    latest_path = tmp_path / "training_state_latest.pt"
    torch.save({"marker": "latest", "epoch": 5}, latest_path)

    state, tag = checkpoints.try_load_state(tmp_path)
    assert tag == "latest"
    assert state["marker"] == "latest"
    assert state["epoch"] == 5


def test_try_load_state_returns_none_when_no_checkpoints(tmp_path):
    """Test that try_load_state returns None when no checkpoint exists."""
    state, tag = checkpoints.try_load_state(tmp_path)
    assert state is None
    assert tag == ""


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    """Test full save and load cycle with all components."""
    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    vat_lidar = DummyModule()
    base = DummySavePretrainedModule()
    optim = DummyOpt()
    sched = DummySched()
    scaler = DummyScaler()

    checkpoints.save_state(
        out_dir=tmp_path,
        tag="latest",
        step=100,
        epoch=5,
        global_step=100,
        epoch_losses=[0.5, 0.4, 0.3, 0.2, 0.1],
        best_loss=0.1,
        best_step=80,
        optim=optim,
        sched=sched,
        scaler=scaler,
        vat_lidar=vat_lidar,
        vat_vision=None,
        base=base,
        clip_vit=None,
        vision_adapter=None,
        projector=None,
        sched_meta={"total_steps": 200},
        config={"epochs": 10},
        val_losses=[0.15, 0.12, 0.10],
        val_epochs=[2, 4, 5],
    )

    state, tag = checkpoints.try_load_state(tmp_path)
    
    assert tag == "latest"
    assert state is not None
    
    assert state["epoch"] == 5
    assert state["global_step"] == 100
    assert state["epoch_losses"] == [0.5, 0.4, 0.3, 0.2, 0.1]
    assert state["best_loss"] == 0.1
    assert state["best_step"] == 80
    assert state["val_losses"] == [0.15, 0.12, 0.10]
    assert state["val_epochs"] == [2, 4, 5]
    assert state["sched_meta"]["total_steps"] == 200
    assert state["config"]["epochs"] == 10
    
    assert state["optimizer"] == {"opt": 1}
    assert state["scheduler"] == {"sched": 1}
    assert state["scaler"] == {"scaler": 1}
    
    assert "rng" in state
    assert "py_random" in state["rng"]
    assert "np_random" in state["rng"]
    assert "torch" in state["rng"]


# ============================================================================
# Tests for issue 1.1: mixed_precision mode saved in checkpoint
# ============================================================================

def test_save_state_includes_mixed_precision_fp16(tmp_path, monkeypatch):
    """Test that mixed_precision='fp16' is saved in checkpoint state."""
    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    vat_lidar = DummyModule()
    base = DummySavePretrainedModule()
    optim = DummyOpt()
    sched = DummySched()

    checkpoints.save_state(
        out_dir=tmp_path,
        tag="latest",
        step=10,
        epoch=1,
        global_step=10,
        epoch_losses=[0.5],
        best_loss=0.5,
        best_step=None,
        optim=optim,
        sched=sched,
        scaler=None,
        vat_lidar=vat_lidar,
        vat_vision=None,
        base=base,
        clip_vit=None,
        vision_adapter=None,
        projector=None,
        sched_meta={},
        config={"mixed_precision": "fp16"},
    )

    state, tag = checkpoints.try_load_state(tmp_path)
    assert state is not None
    assert state.get("mixed_precision") == "fp16"


def test_save_state_includes_mixed_precision_bf16(tmp_path, monkeypatch):
    """Test that mixed_precision='bf16' is saved in checkpoint state."""
    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    vat_lidar = DummyModule()
    base = DummySavePretrainedModule()
    optim = DummyOpt()
    sched = DummySched()

    checkpoints.save_state(
        out_dir=tmp_path,
        tag="latest",
        step=10,
        epoch=1,
        global_step=10,
        epoch_losses=[0.5],
        best_loss=0.5,
        best_step=None,
        optim=optim,
        sched=sched,
        scaler=None,
        vat_lidar=vat_lidar,
        vat_vision=None,
        base=base,
        clip_vit=None,
        vision_adapter=None,
        projector=None,
        sched_meta={},
        config={"mixed_precision": "bf16"},
    )

    state, tag = checkpoints.try_load_state(tmp_path)
    assert state is not None
    assert state.get("mixed_precision") == "bf16"


def test_save_state_handles_legacy_fp16_config(tmp_path, monkeypatch):
    """Test that legacy fp16=True config is converted to mixed_precision='fp16'."""
    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    vat_lidar = DummyModule()
    base = DummySavePretrainedModule()
    optim = DummyOpt()
    sched = DummySched()

    # Legacy config without mixed_precision but with fp16=True
    checkpoints.save_state(
        out_dir=tmp_path,
        tag="latest",
        step=10,
        epoch=1,
        global_step=10,
        epoch_losses=[0.5],
        best_loss=0.5,
        best_step=None,
        optim=optim,
        sched=sched,
        scaler=None,
        vat_lidar=vat_lidar,
        vat_vision=None,
        base=base,
        clip_vit=None,
        vision_adapter=None,
        projector=None,
        sched_meta={},
        config={"fp16": True},  # Legacy config
    )

    state, tag = checkpoints.try_load_state(tmp_path)
    assert state is not None
    assert state.get("mixed_precision") == "fp16"


def test_save_state_handles_no_mixed_precision_config(tmp_path, monkeypatch):
    """Test that config without mixed_precision defaults to 'no'."""
    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    vat_lidar = DummyModule()
    base = DummySavePretrainedModule()
    optim = DummyOpt()
    sched = DummySched()

    checkpoints.save_state(
        out_dir=tmp_path,
        tag="latest",
        step=10,
        epoch=1,
        global_step=10,
        epoch_losses=[0.5],
        best_loss=0.5,
        best_step=None,
        optim=optim,
        sched=sched,
        scaler=None,
        vat_lidar=vat_lidar,
        vat_vision=None,
        base=base,
        clip_vit=None,
        vision_adapter=None,
        projector=None,
        sched_meta={},
        config={},  # No mixed_precision or fp16
    )

    state, tag = checkpoints.try_load_state(tmp_path)
    assert state is not None
    assert state.get("mixed_precision") == "no"


# ============================================================================
# Tests for issue 1.6: SAM compression head save/restore
# ============================================================================

class DummySAMModule(torch.nn.Module):
    """Mock SAM module with net_2 and net_3 compression head layers."""
    def __init__(self):
        super().__init__()
        # Frozen SAM backbone layers (should NOT be saved)
        self.image_encoder = torch.nn.Linear(256, 256)
        self.mask_decoder = torch.nn.Linear(128, 128)
        
        # Trainable compression head layers (SHOULD be saved)
        self.net_2 = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
        )
        self.net_3 = torch.nn.Linear(32, 16)
        
        # Set requires_grad appropriately
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.mask_decoder.parameters():
            p.requires_grad = False
        for p in self.net_2.parameters():
            p.requires_grad = True
        for p in self.net_3.parameters():
            p.requires_grad = True


def test_save_state_saves_sam_compression_head(tmp_path, monkeypatch):
    """Test that SAM compression head (net_2, net_3) is saved."""
    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    vat_lidar = DummyModule()
    base = DummySavePretrainedModule()
    sam = DummySAMModule()
    optim = DummyOpt()
    sched = DummySched()

    checkpoints.save_state(
        out_dir=tmp_path,
        tag="latest",
        step=10,
        epoch=1,
        global_step=10,
        epoch_losses=[0.5],
        best_loss=0.5,
        best_step=None,
        optim=optim,
        sched=sched,
        scaler=None,
        vat_lidar=vat_lidar,
        vat_vision=None,
        base=base,
        clip_vit=None,
        vision_adapter=None,
        projector=None,
        sam=sam,
        sched_meta={},
        config={},
    )

    # SAM compression head should be saved
    sam_path = tmp_path / "sam_compression_head_latest.pt"
    _assert_exists(sam_path)
    
    # Load and verify only compression head params are saved
    sam_state = torch.load(sam_path, map_location="cpu")
    
    # Should have net_2 and net_3 params
    param_names = list(sam_state.keys())
    assert any("net_2" in name for name in param_names)
    assert any("net_3" in name for name in param_names)
    
    # Should NOT have image_encoder or mask_decoder params
    assert not any("image_encoder" in name for name in param_names)
    assert not any("mask_decoder" in name for name in param_names)


def test_save_state_sam_compression_head_none_does_not_create_file(tmp_path, monkeypatch):
    """Test that no SAM file is created when sam=None."""
    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    vat_lidar = DummyModule()
    base = DummySavePretrainedModule()
    optim = DummyOpt()
    sched = DummySched()

    checkpoints.save_state(
        out_dir=tmp_path,
        tag="latest",
        step=10,
        epoch=1,
        global_step=10,
        epoch_losses=[0.5],
        best_loss=0.5,
        best_step=None,
        optim=optim,
        sched=sched,
        scaler=None,
        vat_lidar=vat_lidar,
        vat_vision=None,
        base=base,
        clip_vit=None,
        vision_adapter=None,
        projector=None,
        sam=None,
        sched_meta={},
        config={},
    )

    # SAM compression head should NOT be created
    sam_path = tmp_path / "sam_compression_head_latest.pt"
    _assert_not_exists(sam_path)


def test_sam_compression_head_values_are_correct(tmp_path, monkeypatch):
    """Test that saved SAM compression head values match the original."""
    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    vat_lidar = DummyModule()
    base = DummySavePretrainedModule()
    sam = DummySAMModule()
    optim = DummyOpt()
    sched = DummySched()

    # Get original net_2 and net_3 state before saving
    original_net2_state = {k: v.clone() for k, v in sam.net_2.state_dict().items()}
    original_net3_state = {k: v.clone() for k, v in sam.net_3.state_dict().items()}

    checkpoints.save_state(
        out_dir=tmp_path,
        tag="latest",
        step=10,
        epoch=1,
        global_step=10,
        epoch_losses=[0.5],
        best_loss=0.5,
        best_step=None,
        optim=optim,
        sched=sched,
        scaler=None,
        vat_lidar=vat_lidar,
        vat_vision=None,
        base=base,
        clip_vit=None,
        vision_adapter=None,
        projector=None,
        sam=sam,
        sched_meta={},
        config={},
    )

    # Load saved state
    sam_path = tmp_path / "sam_compression_head_latest.pt"
    saved_state = torch.load(sam_path, map_location="cpu")

    # Verify values match
    for key, value in original_net2_state.items():
        saved_key = f"net_2.{key}"
        assert saved_key in saved_state
        assert torch.allclose(saved_state[saved_key], value)

    for key, value in original_net3_state.items():
        saved_key = f"net_3.{key}"
        assert saved_key in saved_state
        assert torch.allclose(saved_state[saved_key], value)
