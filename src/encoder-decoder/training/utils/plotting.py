"""Plotting utilities for training visualization"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional


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


def plot_metric_curves(
    metrics_history: Dict[str, List],
    epochs: List[int],
    out_dir: Path,
    metric_type: str = "caption"
):
    """
    Plot individual metric curves over epochs.
    
    Args:
        metrics_history: Dictionary of metric names to lists of values
        epochs: Epoch numbers where metrics were computed
        out_dir: Directory to save plots
        metric_type: Type of metrics ("caption" or "grounding")
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not metrics_history or not epochs:
        return
    
    # Plot each metric separately
    for metric_name, values in metrics_history.items():
        if not values:
            continue
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, values, linewidth=2, marker="o", markersize=5, color='steelblue')
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{metric_type.capitalize()}: {metric_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with safe filename
        safe_name = metric_name.replace(" ", "_").replace("/", "_").lower()
        plt.savefig(out_dir / f"{metric_type}_{safe_name}.png", dpi=120)
        plt.close()
    
    # Also create a combined plot with all metrics (if multiple)
    if len(metrics_history) > 1:
        plt.figure(figsize=(12, 6))
        for metric_name, values in metrics_history.items():
            if values:
                plt.plot(epochs, values, linewidth=2, marker="o", markersize=4, label=metric_name)
        
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{metric_type.capitalize()} Metrics Over Time")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric_type}_metrics_combined.png", dpi=120)
        plt.close()


def plot_all_metrics(
    caption_metrics: Dict[str, List],
    grounding_metrics: Dict[str, List],
    epochs: List[int],
    out_dir: Path
):
    """
    Plot all caption and grounding metrics.
    
    Args:
        caption_metrics: Dictionary of caption metric histories
        grounding_metrics: Dictionary of grounding metric histories
        epochs: Epoch numbers
        out_dir: Output directory
    """
    # Create subdirectory for metric plots
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot caption metrics
    if caption_metrics:
        plot_metric_curves(caption_metrics, epochs, metrics_dir, "caption")
    
    # Plot grounding metrics
    if grounding_metrics:
        plot_metric_curves(grounding_metrics, epochs, metrics_dir, "grounding")
