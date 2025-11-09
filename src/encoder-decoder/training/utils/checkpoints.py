"""Checkpoint management utilities"""

import random
import shutil
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional


def save_state(
    out_dir: Path,
    tag: str,
    *,
    step: int,
    epoch: int,
    it_in_epoch: int,
    global_step: int,
    epoch_losses: List[float],
    best_loss: float,
    best_step: Optional[int],
    optim,
    sched,
    vat_lidar: nn.Module,
    vat_vision: Optional[nn.Module],
    base: nn.Module,
    clip_vit: Optional[nn.Module],
    vision_adapter: Optional[nn.Module] = None,
    projector: Optional[nn.Module] = None,
    sched_meta: Dict,
    config: Dict,
    val_losses: Optional[List[float]] = None,
    val_epochs: Optional[List[int]] = None,
):
    """
    Save training state and model checkpoints.
    
    Args:
        out_dir: Output directory for checkpoints
        tag: Tag for checkpoint (e.g., "latest" or "step1000")
        step: Current step number
        epoch: Current epoch number
        it_in_epoch: Iteration within epoch
        global_step: Global training step
        epoch_losses: List of epoch losses
        best_loss: Best validation loss so far
        best_step: Step with best validation loss
        optim: Optimizer
        sched: Learning rate scheduler
        vat_lidar: LiDAR VAT model
        vat_vision: Vision VAT model (optional)
        base: Base LLM model
        clip_vit: CLIP model (optional)
        vision_adapter: Vision adapter model (optional)
        projector: Projector model (optional)
        sched_meta: Scheduler metadata
        config: Training configuration
        val_losses: Validation losses (optional)
        val_epochs: Epochs where validation was run (optional)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Unwrap DDP if needed
    def unwrap(model):
        return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    
    if tag == "latest":
        torch.save(unwrap(vat_lidar).state_dict(), out_dir / "vat_lidar_latest.pt")
        if vat_vision is not None:
            torch.save(unwrap(vat_vision).state_dict(), out_dir / "vat_vision_latest.pt")
        unwrap(base).save_pretrained(out_dir / "qwen2_lora_adapter_latest")
        if vision_adapter is not None:
            torch.save(unwrap(vision_adapter).state_dict(), out_dir / "vision_adapter_latest.pt")
        if projector is not None:
            torch.save(unwrap(projector).state_dict(), out_dir / "projector_latest.pt")
        if clip_vit is not None:
            unwrap(clip_vit).save_pretrained(out_dir / "clip_lora_adapter_latest")
        fname = "training_state_latest.pt"
    else:
        torch.save(unwrap(vat_lidar).state_dict(), out_dir / f"vat_lidar_step{step}.pt")
        if vat_vision is not None:
            torch.save(unwrap(vat_vision).state_dict(), out_dir / f"vat_vision_step{step}.pt")
        unwrap(base).save_pretrained(out_dir / f"qwen2_lora_adapter_step{step}")
        if vision_adapter is not None:
            torch.save(unwrap(vision_adapter).state_dict(), out_dir / f"vision_adapter_step{step}.pt")
        if projector is not None:
            torch.save(unwrap(projector).state_dict(), out_dir / f"projector_step{step}.pt")
        if clip_vit is not None:
            unwrap(clip_vit).save_pretrained(out_dir / f"clip_lora_adapter_step{step}")
        fname = f"training_state_step{step}.pt"

    # Save RNG states
    rng = {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    
    state = {
        "epoch": epoch,
        "it_in_epoch": it_in_epoch,
        "global_step": global_step,
        "epoch_losses": epoch_losses,
        "best_loss": best_loss,
        "best_step": best_step,
        "val_losses": val_losses,
        "val_epochs": val_epochs,
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict(),
        "rng": rng,
        "sched_meta": sched_meta,
        "config": config,
    }
    torch.save(state, out_dir / fname)


def try_load_state(out_dir: Path):
    """
    Try to load training state from checkpoint.
    
    Args:
        out_dir: Directory containing checkpoints
        
    Returns:
        Tuple of (state_dict, tag) if found, else (None, "")
    """
    p_latest = out_dir / "training_state_latest.pt"
    if p_latest.exists():
        st = torch.load(p_latest, map_location="cpu")
        return st, "latest"
        
    steps = []
    for p in out_dir.glob("training_state_step*.pt"):
        try:
            steps.append(int(p.stem.replace("training_state_step", "")))
        except:
            pass
            
    if steps:
        stp = max(steps)
        st = torch.load(out_dir / f"training_state_step{stp}.pt", map_location="cpu")
        return st, f"step{stp}"
        
    return None, ""


def prune_checkpoints_steps(out_dir: Path, keep_last_n: int, best_step: Optional[int]):
    """
    Remove old checkpoints, keeping only the last N and the best.
    
    Args:
        out_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        best_step: Step number of best checkpoint (always kept)
    """
    steps = []
    for f in out_dir.glob("vat_lidar_step*.pt"):
        try:
            steps.append(int(f.stem.replace("vat_lidar_step", "")))
        except:
            pass
            
    steps = sorted(steps, reverse=True)
    to_keep = set(steps[:keep_last_n])
    if best_step is not None:
        to_keep.add(best_step)
        
    for st in steps:
        if st in to_keep:
            continue
            
        for p in [
            out_dir / f"vat_lidar_step{st}.pt",
            out_dir / f"vat_vision_step{st}.pt",
            out_dir / f"vision_adapter_step{st}.pt",
            out_dir / f"projector_step{st}.pt",
            out_dir / f"training_state_step{st}.pt",
        ]:
            if p.exists():
                if p.suffix == ".pt":
                    p.unlink()
                    
        p_l = out_dir / f"qwen2_lora_adapter_step{st}"
        if p_l.exists():
            shutil.rmtree(p_l, ignore_errors=True)
            
        p_c = out_dir / f"clip_lora_adapter_step{st}"
        if p_c.exists():
            shutil.rmtree(p_c, ignore_errors=True)
