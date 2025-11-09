# file: src/encoder-decoder/training-test/models/test_vat_block.py
"""
VATBlock Testing & Summary

This script:
  - Imports VATBlock from training.models.vat_blocks.
  - Prints a concise module summary (submodules + parameter counts).
  - Runs forward-pass tests for multiple configurations.

VATBlock API (from vat_blocks.py):
    VATBlock(d_model: int, n_heads: int, d_mlp: int, dropout: float)

Forward signature:
    forward(q, kv)
      q:  [B, Nq, d_model]
      kv: [B, Nk, d_model]
      →  [B, Nq, d_model]

Run from repo root:
    python src/encoder-decoder/training-test/models/test_vat_block.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Path setup so `from training.models.vat_blocks import VATBlock` works
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # .../training-test/models
PROJECT_ROOT = THIS_DIR.parent.parent               # .../encoder-decoder
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.models.vat_blocks import VATBlock  # noqa: E402


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_module_summary(module: nn.Module, name: str = "VATBlock"):
    """
    Print a compact summary:
      - class name
      - total/trainable params
      - direct child modules with their param counts
    """
    total_params, trainable_params = count_parameters(module)

    print("\n" + "=" * 70)
    print(f"{name} SUMMARY")
    print("=" * 70)
    print(f"Class           : {module.__class__.__name__}")
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print("-" * 70)
    print("Submodules:")
    for child_name, child in module.named_children():
        c_total, c_train = count_parameters(child)
        print(
            f"  {child_name:<18} "
            f"{child.__class__.__name__:<24} "
            f"params={c_total:8,d}, trainable={c_train:8,d}"
        )
    print("=" * 70 + "\n")


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_vat_block_single_config():
    """
    Smoke test:
      - one VATBlock config
      - forward pass
      - shape preservation + NaN/Inf check
    """
    print("\n" + "=" * 60)
    print("TEST: VATBlock single-config forward")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")

    d_model = 256
    n_heads = 8
    d_mlp = 1024  # e.g. 4x expansion
    dropout = 0.1

    # Match actual signature: (d_model, n_heads, d_mlp, dropout)
    block = VATBlock(d_model, n_heads, d_mlp, dropout).to(device)
    block.eval()

    print_module_summary(block, name="VATBlock (single config)")

    B = 2
    Nq = 128
    Nk = 512

    q = torch.randn(B, Nq, d_model, device=device)
    kv = torch.randn(B, Nk, d_model, device=device)

    with torch.no_grad():
        out = block(q, kv)

    print(f"Input  q  shape: {tuple(q.shape)}")
    print(f"Input  kv shape: {tuple(kv.shape)}")
    print(f"Output   shape: {tuple(out.shape)}")

    assert out.shape == q.shape, "Output must match q shape: [B, Nq, d_model]"
    assert not torch.isnan(out).any(), "Output contains NaNs"
    assert not torch.isinf(out).any(), "Output contains Infs"

    mean = float(out.mean())
    std = float(out.std())
    print(f"Output stats   : mean={mean:.4f}, std={std:.4f}")
    print("✓ Single-config VATBlock test passed.\n")


def test_vat_block_multi_configs():
    """
    Sweep multiple configurations and (B, Nq, Nk) shapes.

    Validates:
      - construction matches VATLiDAR usage,
      - output shape correctness,
      - numerical sanity.
    """
    print("\n" + "=" * 60)
    print("TEST: VATBlock multi-config sweep")
    print("=" * 60)

    device = get_device()
    torch.manual_seed(0)

    # Each config: (d_model, n_heads, mlp_ratio, dropout)
    configs = [
        (256, 8, 4.0, 0.1),
        (512, 8, 4.0, 0.1),
        (768, 12, 4.0, 0.1),
    ]

    # Test shapes: (B, Nq, Nk)
    shapes = [
        (1, 64, 256),
        (2, 128, 512),
        (4, 256, 1024),
    ]

    for idx, (d_model, n_heads, mlp_ratio, dropout) in enumerate(configs, start=1):
        d_mlp = int(mlp_ratio * d_model)
        print(
            f"\n--- Config {idx} --- "
            f"d_model={d_model}, n_heads={n_heads}, d_mlp={d_mlp}, dropout={dropout}"
        )

        block = VATBlock(d_model, n_heads, d_mlp, dropout).to(device)
        block.eval()

        total_params, trainable_params = count_parameters(block)
        print(f"  Params: total={total_params:,}, trainable={trainable_params:,}")

        for (B, Nq, Nk) in shapes:
            q = torch.randn(B, Nq, d_model, device=device)
            kv = torch.randn(B, Nk, d_model, device=device)

            with torch.no_grad():
                out = block(q, kv)

            assert out.shape == (B, Nq, d_model), (
                f"Expected {(B, Nq, d_model)}, got {tuple(out.shape)}"
            )
            assert not torch.isnan(out).any(), "NaNs in output"
            assert not torch.isinf(out).any(), "Infs in output"

            mean = float(out.mean())
            std = float(out.std())
            print(
                f"  ✓ B={B}, Nq={Nq}, Nk={Nk} "
                f"mean={mean:.4f}, std={std:.4f}"
            )

        del block
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n✓ All VATBlock configs passed.\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print(" " * 22 + "VATBlock Testing Suite")
    print("=" * 70)

    try:
        test_vat_block_single_config()
        test_vat_block_multi_configs()

        print("\n" + "=" * 70)
        print(" " * 24 + "ALL VATBlock TESTS COMPLETED")
        print("=" * 70 + "\n")
    except Exception as e:
        print("\n" + "=" * 70)
        print("VATBlock TEST FAILED")
        print("=" * 70)
        print(e)
        print("=" * 70 + "\n")
        raise


if __name__ == "__main__":
    main()
