"""Plotting utilities for training visualization"""

import math
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
    Dynamically adjusts grid size based on enabled metrics (non-zero data).
    
    Args:
        metrics_history: Dictionary of metric names to lists of values
        epochs: Epoch numbers where metrics were computed
        out_dir: Directory to save plots
        metric_type: Type of metrics ("caption", "grounding_det_area", or "grounding_det_object")
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not metrics_history or not epochs:
        return
    
    # Filter to only metrics with actual data (non-empty and not all zeros)
    active_metrics = {}
    for metric_name, values in metrics_history.items():
        if values and len(values) > 0:
            # Check if there's any non-zero value (metric was actually computed)
            if any(v != 0.0 for v in values):
                active_metrics[metric_name] = values
    
    if not active_metrics:
        print(f"[plotting] No active metrics to plot for {metric_type}")
        return
    
    # Plot each metric separately
    for metric_name, values in active_metrics.items():
        plt.figure(figsize=(8, 5))
        plt.plot(epochs[:len(values)], values, linewidth=2, marker="o", markersize=5, color='steelblue')
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{metric_type.replace('_', ' ').title()}: {metric_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with safe filename
        safe_name = metric_name.replace(" ", "_").replace("/", "_").lower()
        plt.savefig(out_dir / f"{metric_type}_{safe_name}.png", dpi=120)
        plt.close()
    
    # Create a grid subplot with all active metrics (if multiple)
    if len(active_metrics) > 1:
        # Calculate optimal grid dimensions
        n_metrics = len(active_metrics)
        
        # Dynamic grid sizing: prefer wider layouts
        if n_metrics <= 2:
            n_cols = n_metrics
            n_rows = 1
        elif n_metrics <= 4:
            n_cols = 2
            n_rows = 2
        elif n_metrics <= 6:
            n_cols = 3
            n_rows = 2
        else:
            n_cols = min(3, n_metrics)
            n_rows = math.ceil(n_metrics / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Flatten axes array for easier iteration
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = list(axes)
        elif n_cols == 1 and n_rows > 1:
            axes = list(axes)
        else:
            axes = axes.flatten()
        
        # Plot each metric in its own subplot
        for idx, (metric_name, values) in enumerate(active_metrics.items()):
            ax = axes[idx]
            ax.plot(epochs[:len(values)], values, linewidth=2, marker="o", markersize=4, color='steelblue')
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(metric_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        # Add overall title
        fig.suptitle(f"{metric_type.replace('_', ' ').title()} Metrics Over Time", 
                     fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric_type}_metrics_combined.png", dpi=120, bbox_inches='tight')
        plt.close()
    elif len(active_metrics) == 1:
        # Single metric - create a simple combined plot (same as individual)
        metric_name, values = list(active_metrics.items())[0]
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(epochs[:len(values)], values, linewidth=2, marker="o", markersize=5, color='steelblue')
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(f"{metric_type.replace('_', ' ').title()}: {metric_name}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric_type}_metrics_combined.png", dpi=120)
        plt.close()


def plot_all_metrics(
    caption_metrics: Dict[str, List],
    grounding_det_area_metrics: Dict[str, List],
    grounding_det_object_metrics: Dict[str, List],
    epochs: List[int],
    out_dir: Path
):
    """
    Plot all caption and grounding metrics for three dashboards.
    
    Args:
        caption_metrics: Dictionary of caption metric histories
        grounding_det_area_metrics: Dictionary of det_area metric histories (text + bbox)
        grounding_det_object_metrics: Dictionary of det_object metric histories (text only)
        epochs: Epoch numbers
        out_dir: Output directory
    """
    # Create subdirectory for metric plots
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot caption metrics
    if caption_metrics:
        plot_metric_curves(caption_metrics, epochs, metrics_dir, "caption")
    
    # Plot grounding det_area metrics (text + bbox)
    if grounding_det_area_metrics:
        plot_metric_curves(grounding_det_area_metrics, epochs, metrics_dir, "grounding_det_area")
    
    # Plot grounding det_object metrics (text only)
    if grounding_det_object_metrics:
        plot_metric_curves(grounding_det_object_metrics, epochs, metrics_dir, "grounding_det_object")
