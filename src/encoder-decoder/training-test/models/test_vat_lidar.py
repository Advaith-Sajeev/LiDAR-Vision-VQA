# file: src/encoder-decoder/training-test/models/test_vat_lidar.py
"""
VATLiDAR Testing & Visualization

This script:
  - Validates geometric feature generation and sector assignments.
  - Verifies view embedding usage and query grouping.
  - Runs forward passes across multiple configs / grid sizes.
  - Visualizes the 6-way sector partition used in VATLiDAR._grid.

Expected layout:
  - VATLiDAR:
        src/encoder-decoder/training/models/vat_lidar.py
  - This test:
        src/encoder-decoder/training-test/models/test_vat_lidar.py

Run (from repo root or any directory):
  python src/encoder-decoder/training-test/models/test_vat_lidar.py
"""

import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path setup so `from training.models.vat_lidar import VATLiDAR` works
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent                          # .../training-test/models
PROJECT_ROOT = THIS_DIR.parent.parent                               # .../encoder-decoder
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.models.vat_lidar import VATLiDAR  # noqa: E402  (import after sys.path tweak)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_sector_assignments(H: int = 64, W: int = 64, save_path: Path | None = None):
    """
    Visualize the 6-way sector (view) assignment from VATLiDAR._grid.

    Uses a lightweight VATLiDAR instance; only _grid is relevant here.
    """
    device = get_device()
    print(f"[viz] Using device: {device}")

    model = VATLiDAR(c_in=64, d_model=256, n_queries=576).to(device).eval()

    with torch.no_grad():
        geom, sid = model._grid(H, W, device)

    sid_grid = sid.cpu().numpy().reshape(H, W)

    sector_colors = ["red", "orange", "yellow", "cyan", "blue", "purple"]
    sector_names = ["Front", "Front-Right", "Front-Left",
                    "Back", "Back-Right", "Back-Left"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1) Sector ID heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(sid_grid, cmap="tab10", interpolation="nearest", vmin=0, vmax=5)
    ax1.set_title("Sector Assignment (View IDs)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("X (Width)")
    ax1.set_ylabel("Y (Height)")
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=range(6))
    cbar1.set_label("Sector ID")

    # 2) Colored sectors
    ax2 = axes[0, 1]
    colored = np.zeros((H, W, 3), dtype=np.float32)
    from matplotlib.colors import to_rgb
    for i in range(6):
        mask = sid_grid == i
        colored[mask] = to_rgb(sector_colors[i])
    ax2.imshow(colored)
    ax2.set_title("Sector Colors", fontsize=14, fontweight="bold")
    ax2.set_xlabel("X (Width)")
    ax2.set_ylabel("Y (Height)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=sector_colors[i], label=f"{i}: {sector_names[i]}")
        for i in range(6)
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # 3) Polar angle map (degrees)
    ax3 = axes[1, 0]
    geom_np = geom.cpu().numpy()
    x_coords = geom_np[:, 0]
    y_coords = geom_np[:, 1]
    theta = np.arctan2(y_coords, x_coords).reshape(H, W)
    im3 = ax3.imshow(np.degrees(theta), cmap="twilight", interpolation="nearest")
    ax3.set_title("Polar Angles (degrees)", fontsize=14, fontweight="bold")
    ax3.set_xlabel("X (Width)")
    ax3.set_ylabel("Y (Height)")
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label("Angle (degrees)")

    # 4) Sector counts
    ax4 = axes[1, 1]
    sector_counts = [int((sid_grid == i).sum()) for i in range(6)]
    bars = ax4.bar(range(6), sector_counts, color=sector_colors)
    ax4.set_title("Pixels per Sector", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Sector ID")
    ax4.set_ylabel("Pixel Count")
    ax4.set_xticks(range(6))
    ax4.set_xticklabels(
        [f"{i}\n{sector_names[i]}" for i in range(6)],
        rotation=0,
        fontsize=9,
    )
    ax4.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, sector_counts):
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            count,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[viz] Saved sector visualization to: {save_path}")

    plt.show()

    total = H * W
    print("\n" + "=" * 60)
    print("SECTOR ASSIGNMENT STATISTICS")
    print("=" * 60)
    print(f"Grid size: {H} x {W} = {total} pixels")
    for i in range(6):
        pct = 100.0 * sector_counts[i] / total
        print(f"  Sector {i} ({sector_names[i]:12s}): {sector_counts[i]:6d} px ({pct:5.2f}%)")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_geometric_features():
    """Check _grid shapes, ranges, sector coverage, and cache behavior."""
    print("\n" + "=" * 60)
    print("TEST: GEOMETRIC FEATURES (_grid)")
    print("=" * 60)

    device = get_device()
    model = VATLiDAR(c_in=256, d_model=512, n_queries=576).to(device).eval()

    H, W = 50, 50
    with torch.no_grad():
        geom, sid = model._grid(H, W, device)

    assert geom.shape == (H * W, 5), f"geom shape {geom.shape}, expected {(H*W, 5)}"
    assert sid.shape == (H * W,), f"sid shape {sid.shape}, expected {(H*W,)}"

    x, y, r, s, c = geom[:, 0], geom[:, 1], geom[:, 2], geom[:, 3], geom[:, 4]
    print(f"  x in [{x.min():.3f}, {x.max():.3f}] (expected ~[-1, 1])")
    print(f"  y in [{y.min():.3f}, {y.max():.3f}] (expected ~[-1, 1])")
    print(f"  r in [{r.min():.3f}, {r.max():.3f}] (expected [0, 1])")
    print(f"  sinθ in [{s.min():.3f}, {s.max():.3f}] (expected [-1, 1])")
    print(f"  cosθ in [{c.min():.3f}, {c.max():.3f}] (expected [-1, 1])")

    unique_sids = torch.unique(sid).cpu().tolist()
    print(f"  Unique sector IDs: {sorted(unique_sids)}")
    assert len(unique_sids) == 6, "Expected 6 unique sectors"
    assert all(0 <= v < 6 for v in unique_sids), "Sector IDs must be in [0, 5]"

    # Cache check
    with torch.no_grad():
        geom2, sid2 = model._grid(H, W, device)
    assert torch.equal(geom, geom2), "Cached geom mismatch"
    assert torch.equal(sid, sid2), "Cached sid mismatch"

    print("  ✓ Geometric features & caching OK\n")


def test_view_embedding_assignment():
    """Ensure view embeddings exist, split is correct, and forward runs."""
    print("\n" + "=" * 60)
    print("TEST: VIEW EMBEDDING ASSIGNMENT")
    print("=" * 60)

    device = get_device()
    model = VATLiDAR(c_in=256, d_model=512, n_queries=576, n_layers=1).to(device).eval()

    print(f"  n_queries        : {model.n_queries}")
    print(f"  queries per view : {model.nq_per_view}")
    print(f"  view_embed shape : {tuple(model.view_embed.shape)}")

    assert model.view_embed.requires_grad, "view_embed must be learnable"
    assert model.nq_per_view * 6 == model.n_queries, "Queries must split evenly into 6 views"

    B, H, W = 2, 40, 40
    bev = torch.randn(B, 256, H, W, device=device)

    with torch.no_grad():
        out = model(bev)

    print(f"  Output shape     : {tuple(out.shape)}")
    assert out.shape == (B, model.n_queries, model.d_model)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    print("  ✓ View embedding assignment OK\n")


def test_vat_lidar_forward():
    """
    Sweep several configs / shapes:
      - check output shape,
      - check numerical sanity (no NaN/Inf),
      - print simple stats.
    """
    print("\n" + "=" * 60)
    print("TEST: VATLiDAR FORWARD PASS")
    print("=" * 60)

    device = get_device()
    print(f"  Using device: {device}")
    torch.manual_seed(0)

    test_configs = [
        {"c_in": 256, "d_model": 512, "n_queries": 576, "n_layers": 4},
        {"c_in": 128, "d_model": 256, "n_queries": 384, "n_layers": 2},
        {"c_in": 512, "d_model": 768, "n_queries": 768, "n_layers": 3},
    ]
    batch_sizes = [1, 2]
    bev_sizes = [(50, 50), (64, 64), (96, 96)]

    for idx, cfg in enumerate(test_configs, start=1):
        print(f"\n--- Config {idx} --- "
              f"c_in={cfg['c_in']}, d_model={cfg['d_model']}, "
              f"n_queries={cfg['n_queries']}, n_layers={cfg['n_layers']}")

        model = VATLiDAR(**cfg).to(device).eval()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: total={total_params:,}, trainable={trainable_params:,}")

        for B in batch_sizes:
            for (H, W) in bev_sizes:
                bev = torch.randn(B, cfg["c_in"], H, W, device=device)
                with torch.no_grad():
                    out = model(bev)

                expected = (B, cfg["n_queries"], cfg["d_model"])
                assert out.shape == expected, f"Expected {expected}, got {tuple(out.shape)}"
                assert not torch.isnan(out).any(), "NaNs in output"
                assert not torch.isinf(out).any(), "Infs in output"

                mean = float(out.mean())
                std = float(out.std())
                min_v = float(out.min())
                max_v = float(out.max())

                print(f"  ✓ B={B}, H={H}, W={W} "
                      f"mean={mean:.4f}, std={std:.4f}, "
                      f"min={min_v:.4f}, max={max_v:.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n  ✓ All forward configs passed.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print(" " * 18 + "VATLiDAR Testing Suite")
    print("=" * 70)

    out_dir = THIS_DIR / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        test_geometric_features()
        test_view_embedding_assignment()
        test_vat_lidar_forward()

        print("\nGenerating sector assignment visualization (H=256, W=256)...")
        viz_path = out_dir / "sector_assignment_visualization.png"
        visualize_sector_assignments(H=256, W=256, save_path=viz_path)

        print("\n" + "=" * 70)
        print(" " * 18 + "ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70 + "\n")
    except Exception as e:
        print("\n" + "=" * 70)
        print("TEST FAILED")
        print("=" * 70)
        print(e)
        print("=" * 70 + "\n")
        raise


if __name__ == "__main__":
    main()
