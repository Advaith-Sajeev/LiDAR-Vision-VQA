"""Plotting utilities for training visualization"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    val_epochs: List[int],
    out_dir: Path,
):
    """
    Plot train and validation losses with correct epoch alignment.
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss values
        val_epochs: Epoch numbers where validation was performed
        out_dir: Directory to save plot
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    
    train_epochs = list(range(1, len(train_losses) + 1))
    plt.plot(train_epochs, train_losses, label="train", linewidth=2, marker="o", markersize=3)
    
    if val_losses and val_epochs:
        plt.plot(val_epochs, val_losses, label="val", linewidth=2, marker="s", markersize=4)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=120)
    plt.close()


def plot_step_curve(step_losses: List[float], out_dir: Path):
    """
    Plot training loss per step.
    
    Args:
        step_losses: Loss value for each training step
        out_dir: Directory to save plot
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    xs = list(range(1, len(step_losses) + 1))
    plt.plot(xs, step_losses, linewidth=1)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss per Step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve_steps.png", dpi=120)
    plt.close()
