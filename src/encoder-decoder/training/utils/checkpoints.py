"""Checkpoint management utilities for epoch-level resumability"""

import random
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
    global_step: int,
    epoch_losses: List[float],
    best_loss: float,
    best_step: Optional[int],
    optim,
    sched,
    scaler=None,  # GradScaler for mixed precision
    vat_lidar: nn.Module,
    vat_vision: Optional[nn.Module],
    base: nn.Module,
    clip_vit: Optional[nn.Module],
    vision_adapter: Optional[nn.Module] = None,
    projector: Optional[nn.Module] = None,
    sam: Optional[nn.Module] = None,  # SAM model with trainable compression head
    sched_meta: Dict,
    config: Dict,
    val_losses: Optional[List[float]] = None,
    val_epochs: Optional[List[int]] = None,
    # Metrics history for live plotting (restore on resume)
    caption_metrics_history: Optional[Dict] = None,
    grounding_det_area_metrics_history: Optional[Dict] = None,
    grounding_det_object_metrics_history: Optional[Dict] = None,
    metrics_epochs: Optional[List[int]] = None,
    # Step-in-epoch for mid-epoch resume (None = end of epoch)
    step_in_epoch: Optional[int] = None,
    # Step-level loss tracking for detailed plots
    step_losses: Optional[List[float]] = None,
    step_loss_steps: Optional[List[int]] = None,
):
    """
    Save training state and model checkpoints.
    
    Supports both epoch-level saves (end of epoch) and step-level saves
    (mid-epoch checkpoints for crash recovery).
    
    Args:
        out_dir: Output directory for checkpoints
        tag: Tag for checkpoint (always "latest" for epoch-level)
        step: Current step number
        epoch: Current epoch number
        global_step: Global training step
        epoch_losses: List of epoch losses
        best_loss: Best validation loss so far
        best_step: Step with best validation loss
        optim: Optimizer
        sched: Learning rate scheduler
        scaler: GradScaler for mixed precision (optional)
        vat_lidar: LiDAR VAT model
        vat_vision: Vision VAT model (optional)
        base: Base LLM model
        clip_vit: CLIP model (optional)
        vision_adapter: Vision adapter model (optional)
        projector: Projector model (optional)
        sam: SAM model with trainable compression head (net_2, net_3) (optional)
        sched_meta: Scheduler metadata
        config: Training configuration
        val_losses: Validation losses (optional)
        val_epochs: Epochs where validation was run (optional)
        step_in_epoch: Step within current epoch for mid-epoch resume (None = end of epoch)
        step_losses: List of step-level losses for detailed plotting (optional)
        step_loss_steps: List of steps corresponding to step_losses (optional)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Unwrap DDP if needed
    def unwrap(model):
        return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    
    # Save model weights
    torch.save(unwrap(vat_lidar).state_dict(), out_dir / "vat_lidar_latest.pt")
    if vat_vision is not None:
        torch.save(unwrap(vat_vision).state_dict(), out_dir / "vat_vision_latest.pt")
    # save_embedding_layers=True: We resize embeddings for special tokens, so explicitly save them
    unwrap(base).save_pretrained(out_dir / "qwen2_lora_adapter_latest", save_embedding_layers=True)
    if vision_adapter is not None:
        torch.save(unwrap(vision_adapter).state_dict(), out_dir / "vision_adapter_latest.pt")
    if projector is not None:
        torch.save(unwrap(projector).state_dict(), out_dir / "projector_latest.pt")
    if clip_vit is not None:
        unwrap(clip_vit).save_pretrained(out_dir / "clip_lora_adapter_latest")
    
    # Save SAM compression head (net_2 and net_3 - the trainable DeepEncoder/VARY layers)
    if sam is not None:
        sam_model = unwrap(sam)
        sam_compression_head_state = {
            name: param.clone() for name, param in sam_model.named_parameters()
            if name.startswith("net_2") or name.startswith("net_3")
        }
        if sam_compression_head_state:
            torch.save(sam_compression_head_state, out_dir / "sam_compression_head_latest.pt")
            print(f"[checkpoint] Saved SAM compression head ({len(sam_compression_head_state)} parameters)")

    # Save RNG states for reproducibility
    rng = {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    
    # Extract mixed_precision mode for validation on resume
    mixed_precision = config.get('mixed_precision', 'fp16' if config.get('fp16', False) else 'no')
    
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "step_in_epoch": step_in_epoch,  # None = end of epoch, >0 = mid-epoch checkpoint
        "epoch_losses": epoch_losses,
        "best_loss": best_loss,
        "best_step": best_step,
        "val_losses": val_losses,
        "val_epochs": val_epochs,
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "mixed_precision": mixed_precision,  # Save for validation on resume
        "rng": rng,
        "sched_meta": sched_meta,
        "config": config,
        # Metrics history for live plotting
        "caption_metrics_history": caption_metrics_history,
        "grounding_det_area_metrics_history": grounding_det_area_metrics_history,
        "grounding_det_object_metrics_history": grounding_det_object_metrics_history,
        "metrics_epochs": metrics_epochs,
        # Step-level loss tracking for detailed plots
        "step_losses": step_losses,
        "step_loss_steps": step_loss_steps,
    }
    torch.save(state, out_dir / "training_state_latest.pt")
    
    # Log appropriate message based on checkpoint type
    if step_in_epoch is not None and step_in_epoch > 0:
        print(f"[checkpoint] Saved step checkpoint: epoch {epoch}, step {step_in_epoch} (global_step={global_step})")
    else:
        print(f"[checkpoint] Saved epoch {epoch} checkpoint (global_step={global_step})")


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
        st = torch.load(p_latest, map_location="cpu", weights_only=False)
        return st, "latest"
        
    return None, ""
