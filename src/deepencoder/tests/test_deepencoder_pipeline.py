import os
import sys

# Ensure src/ is on sys.path so `import deepencoder` works when running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import deepencoder.deepencoder_infer as dei
from deepencoder.deepencoder_infer import (
    DeepEncoderRuntime,
    resize_and_pad_to_square,
    _pil_to_tensor_og_norm,
)

# ---------- Lightweight stand-ins for heavy models ----------

class DummySAM(nn.Module):
    """
    Minimal SAM-like encoder:
    Input : [B, 3, 1024, 1024]
    Output: [B, 1024, 16, 16]
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)                        # [B,1024,H,W]
        y = F.adaptive_avg_pool2d(y, (16, 16))  # [B,1024,16,16]
        return y


class DummyCLIP(nn.Module):
    """
    Minimal CLIP-ViT-like head:
    forward(x, patch_embeds) -> [B, 1 + 256, 1024]
    where 256 = 16*16.
    """
    def __init__(self):
        super().__init__()
        self.embed_dim = 1024

    def forward(self, x: torch.Tensor, patch_embeds: torch.Tensor) -> torch.Tensor:
        B, C, H, W = patch_embeds.shape
        assert C == 1024 and H == 16 and W == 16
        tokens = patch_embeds.flatten(2).transpose(1, 2)  # [B,256,1024]
        cls = torch.zeros(B, 1, self.embed_dim, device=x.device, dtype=x.dtype)
        return torch.cat([cls, tokens], dim=1)            # [B,257,1024]


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def test_image_path() -> os.PathLike:
    here = os.path.dirname(__file__)
    p = os.path.join(here, "test-input-images", "test-1.jpg")
    if not os.path.exists(p):
        pytest.skip(f"test-1.jpg not found at {p}")
    return p


@pytest.fixture
def patched_runtime(monkeypatch):
    """
    Construct DeepEncoderRuntime but:
      - avoid real SAM downloads
      - replace SAM + CLIP builders with small dummies
    """

    # 1) Avoid download / filesystem dependency
    def _fake_download_sam_if_needed(sam_ckpt, auto_download=True):
        # Return dummy path; not used by DummySAM.
        return "/tmp/dummy_sam_checkpoint.pth"

    monkeypatch.setattr(
        dei,
        "download_sam_if_needed",
        _fake_download_sam_if_needed,
        raising=True,
    )

    # 2) Plug in DummySAM for SAM backbone
    def _fake_build_sam_vit_b(checkpoint=None):
        return DummySAM()

    monkeypatch.setattr(
        dei,
        "build_sam_vit_b",
        _fake_build_sam_vit_b,
        raising=True,
    )

    # 3) Plug in DummyCLIP for CLIP-L
    def _fake_build_clip_l():
        return DummyCLIP()

    monkeypatch.setattr(
        dei,
        "build_clip_l",
        _fake_build_clip_l,
        raising=True,
    )

    runtime = DeepEncoderRuntime(
        sam_ckpt=None,
        auto_download_sam=False,
        device="cpu",
        dtype="float32",
        openclip_pretrained="openai",
    )
    runtime.eval()
    return runtime


# ---------- Test: full pipeline + shape trace ----------
def test_deepencoder_full_pipeline_shapes(patched_runtime, test_image_path):
    """
    Run test-1.jpg through the DeepEncoderRuntime pipeline and:
      - print tensor shapes after each major operation
      - save the resized/padded image for inspection
      - assert expected shapes
    """
    runtime = patched_runtime
    shapes = {}

    # 1) Original image
    img = Image.open(test_image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    shapes["original_image_hw"] = (img.size[1], img.size[0])  # (H, W)

    # 2) Resize + pad using package helper
    img_1024 = resize_and_pad_to_square(img, target=runtime.image_size)
    shapes["resized_padded_hw"] = (img_1024.size[1], img_1024.size[0])
    assert img_1024.size == (runtime.image_size, runtime.image_size)

    # ---- Save resized/padded image for debugging ----
    save_dir = os.path.join(os.path.dirname(__file__), "test-output-images")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "resized_padded_test1.jpg")
    img_1024.save(save_path)
    print(f"\n[Saved resized/padded image] -> {save_path}")

    # 3) Normalize using package helper
    x = _pil_to_tensor_og_norm(img_1024).to(device=runtime.device, dtype=runtime.dtype)
    shapes["normalized_tensor"] = tuple(x.shape)  # [1,3,1024,1024]

    # --- instrument _sam_features ---
    orig_sam_features = runtime._sam_features

    def wrapped_sam_features(x_in: torch.Tensor):
        out = orig_sam_features(x_in)
        shapes["sam_features"] = tuple(out.shape)  # [1,1024,16,16]
        return out

    runtime._sam_features = wrapped_sam_features

    # --- instrument CLIP forward ---
    orig_clip_forward = runtime.clip_vit.forward

    def wrapped_clip_forward(x_in, sam_feats, _orig=orig_clip_forward):
        out = _orig(x_in, sam_feats)
        shapes["clip_output"] = tuple(out.shape)                   # [1,257,1024]
        shapes["clip_tokens_wo_cls"] = tuple(out[:, 1:, :].shape)  # [1,256,1024]
        return out

    runtime.clip_vit.forward = wrapped_clip_forward  # type: ignore[assignment]

    # --- instrument projector forward ---
    orig_projector_forward = runtime.projector.forward

    def wrapped_projector_forward(fused_tokens: torch.Tensor, _orig=orig_projector_forward):
        shapes["fused_tokens"] = tuple(fused_tokens.shape)   # [1,256,2048]
        out = _orig(fused_tokens)
        shapes["vision_tokens"] = tuple(out.shape)           # [1,256,2048]
        return out

    runtime.projector.forward = wrapped_projector_forward  # type: ignore[assignment]

    # 4) Run the full pipeline
    out = runtime.encode_image(str(test_image_path))
    final_tokens = out["tokens"]  # [256,2048]
    shapes["final_tokens"] = tuple(final_tokens.shape)

    # ---- Print the trace ----
    print("\n[DeepEncoder pipeline shape trace]")
    for k, v in shapes.items():
        print(f"{k}: {v}")

    # ---- Assertions ----
    assert shapes["normalized_tensor"] == (1, 3, 1024, 1024)
    assert shapes["sam_features"] == (1, 1024, 16, 16)
    assert shapes["clip_output"] == (1, 1 + 16 * 16, 1024)
    assert shapes["clip_tokens_wo_cls"] == (1, 16 * 16, 1024)
    assert shapes["fused_tokens"] == (1, 16 * 16, 2048)
    assert shapes["vision_tokens"] == (1, 16 * 16, 2048)
    assert shapes["final_tokens"] == (16 * 16, 2048)
