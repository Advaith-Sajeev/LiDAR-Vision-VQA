import os
from pathlib import Path

import pytest
import torch

from training.utils import checkpoints


class DummyModule(torch.nn.Module):
    """Simple nn.Module used for state_dict-based saves."""
    def __init__(self):
        super().__init__()


class DummySavePretrainedModule(torch.nn.Module):
    """Module that mimics HuggingFace-style save_pretrained()."""
    def __init__(self):
        super().__init__()
        self.saved_paths = []

    def save_pretrained(self, path):
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


def _assert_exists(path: Path):
    assert path.exists(), f"Expected to exist: {path}"


def _assert_not_exists(path: Path):
    assert not path.exists(), f"Expected to be removed: {path}"


def test_save_state_latest_creates_expected_files_and_state(tmp_path, monkeypatch):
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

    checkpoints.save_state(
        out_dir=tmp_path,
        tag="latest",
        step=5,
        epoch=2,
        it_in_epoch=3,
        global_step=4,
        epoch_losses=[0.1, 0.2],
        best_loss=0.05,
        best_step=99,
        optim=optim,
        sched=sched,
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

    # state = torch.load(state_path, map_location="cpu")

    try:
    # For newer PyTorch where weights_only may be enforced / available
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        # For older PyTorch that doesn't support weights_only
        state = torch.load(state_path, map_location="cpu")

    # Core metadata
    assert state["epoch"] == 2
    assert state["it_in_epoch"] == 3
    assert state["global_step"] == 4
    assert state["epoch_losses"] == [0.1, 0.2]
    assert state["best_loss"] == 0.05
    assert state["best_step"] == 99
    assert state["val_losses"] == [0.01]
    assert state["val_epochs"] == [1]

    # Optimizer & scheduler state
    assert state["optimizer"] == optim.state_dict()
    assert state["scheduler"] == sched.state_dict()

    # RNG + aux
    assert "rng" in state
    assert "sched_meta" in state and state["sched_meta"]["foo"] == "bar"
    assert "config" in state and state["config"]["lr"] == 1e-4


def test_save_state_step_creates_step_files_and_handles_optionals(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    vat_lidar = DummyModule()
    base = DummySavePretrainedModule()
    optim = DummyOpt()
    sched = DummySched()
    step = 10

    # Optional modules set to None should be skipped without error
    checkpoints.save_state(
        out_dir=tmp_path,
        tag="non-latest",  # any value != "latest" goes to step-based branch
        step=step,
        epoch=0,
        it_in_epoch=0,
        global_step=0,
        epoch_losses=[],
        best_loss=0.0,
        best_step=None,
        optim=optim,
        sched=sched,
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

    _assert_exists(tmp_path / f"vat_lidar_step{step}.pt")
    _assert_exists(tmp_path / f"qwen2_lora_adapter_step{step}")
    _assert_exists(tmp_path / f"training_state_step{step}.pt")

    # Optional ones should not exist
    _assert_not_exists(tmp_path / f"vat_vision_step{step}.pt")
    _assert_not_exists(tmp_path / f"vision_adapter_step{step}.pt")
    _assert_not_exists(tmp_path / f"projector_step{step}.pt")
    _assert_not_exists(tmp_path / f"clip_lora_adapter_step{step}")


def test_save_state_unwraps_ddp_wrapped_models(tmp_path, monkeypatch):
    """Ensure DDP-wrapped models are unwrapped via .module before saving."""

    monkeypatch.setattr(checkpoints.torch.cuda, "is_available", lambda: False)

    class DummyDDP(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    # Patch the type used in isinstance(...)
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
        it_in_epoch=0,
        global_step=0,
        epoch_losses=[],
        best_loss=0.0,
        best_step=None,
        optim=optim,
        sched=sched,
        vat_lidar=vat_lidar,
        vat_vision=None,
        base=base,
        clip_vit=None,
        vision_adapter=None,
        projector=None,
        sched_meta={},
        config={},
    )

    # If unwrap failed, these would likely error or be missing.
    _assert_exists(tmp_path / "vat_lidar_latest.pt")
    _assert_exists(tmp_path / "qwen2_lora_adapter_latest")
    _assert_exists(tmp_path / "training_state_latest.pt")


def test_try_load_state_prefers_latest_over_steps(tmp_path):
    latest_path = tmp_path / "training_state_latest.pt"
    step_path = tmp_path / "training_state_step5.pt"

    torch.save({"marker": "step"}, step_path)
    torch.save({"marker": "latest"}, latest_path)

    state, tag = checkpoints.try_load_state(tmp_path)
    assert tag == "latest"
    assert state["marker"] == "latest"


def test_try_load_state_uses_max_step_if_no_latest(tmp_path):
    torch.save({"marker": "s1"}, tmp_path / "training_state_step1.pt")
    torch.save({"marker": "s5"}, tmp_path / "training_state_step5.pt")
    torch.save({"marker": "s3"}, tmp_path / "training_state_step3.pt")

    state, tag = checkpoints.try_load_state(tmp_path)
    assert tag == "step5"
    assert state["marker"] == "s5"


def test_try_load_state_returns_none_when_no_checkpoints(tmp_path):
    state, tag = checkpoints.try_load_state(tmp_path)
    assert state is None
    assert tag == ""


def test_prune_checkpoints_steps_keeps_last_n_and_best(tmp_path):
    # Create checkpoints for steps 1..4
    for st in [1, 2, 3, 4]:
        # The pruning logic discovers steps via vat_lidar_step*.pt
        (tmp_path / f"vat_lidar_step{st}.pt").write_text("vat_lidar")
        (tmp_path / f"vat_vision_step{st}.pt").write_text("vat_vision")
        (tmp_path / f"vision_adapter_step{st}.pt").write_text("vision_adapter")
        (tmp_path / f"projector_step{st}.pt").write_text("projector")
        (tmp_path / f"training_state_step{st}.pt").write_text("state")
        (tmp_path / f"qwen2_lora_adapter_step{st}").mkdir()
        (tmp_path / f"clip_lora_adapter_step{st}").mkdir()

    # Keep last 2 (steps 4,3) + best_step=2
    checkpoints.prune_checkpoints_steps(
        out_dir=tmp_path,
        keep_last_n=2,
        best_step=2,
    )

    # Step 1 should be fully removed
    for prefix in [
        "vat_lidar",
        "vat_vision",
        "vision_adapter",
        "projector",
        "training_state",
    ]:
        _assert_not_exists(tmp_path / f"{prefix}_step1.pt")

    _assert_not_exists(tmp_path / "qwen2_lora_adapter_step1")
    _assert_not_exists(tmp_path / "clip_lora_adapter_step1")

    # Steps 2,3,4 should remain
    for st in [2, 3, 4]:
        for prefix in [
            "vat_lidar",
            "vat_vision",
            "vision_adapter",
            "projector",
            "training_state",
        ]:
            _assert_exists(tmp_path / f"{prefix}_step{st}.pt")

        _assert_exists(tmp_path / f"qwen2_lora_adapter_step{st}")
        _assert_exists(tmp_path / f"clip_lora_adapter_step{st}")
