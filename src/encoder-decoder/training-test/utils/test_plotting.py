from pathlib import Path

from training.utils import plotting as plotting_utils


def test_plot_loss_curve_creates_png_with_train_and_val(tmp_path):
    out_dir = tmp_path / "plots"
    train_losses = [1.0, 0.8, 0.6]
    val_losses = [0.9, 0.7]
    val_epochs = [1, 3]

    plotting_utils.plot_loss_curve(
        train_losses=train_losses,
        val_losses=val_losses,
        val_epochs=val_epochs,
        out_dir=out_dir,
    )

    out_file = out_dir / "loss_curve.png"
    # Directory and file should exist
    assert out_dir.exists()
    assert out_file.exists()
    # Should not be an empty file
    assert out_file.stat().st_size > 0


def test_plot_loss_curve_handles_no_val_data(tmp_path):
    out_dir = tmp_path / "plots_no_val"
    train_losses = [1.0, 0.9, 0.85]

    plotting_utils.plot_loss_curve(
        train_losses=train_losses,
        val_losses=[],
        val_epochs=[],
        out_dir=out_dir,
    )

    out_file = out_dir / "loss_curve.png"
    # Still should produce a plot with only train curve
    assert out_dir.exists()
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_plot_step_curve_creates_png(tmp_path):
    out_dir = tmp_path / "plots_steps"
    step_losses = [1.0, 0.95, 0.9, 0.85]

    plotting_utils.plot_step_curve(step_losses=step_losses, out_dir=out_dir)

    out_file = out_dir / "loss_curve_steps.png"
    assert out_dir.exists()
    assert out_file.exists()
    assert out_file.stat().st_size > 0
