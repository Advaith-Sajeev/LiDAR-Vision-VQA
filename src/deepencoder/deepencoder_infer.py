#!/usr/bin/env python3
"""
DeepEncoder inference (SAM ViT-B + CLIP ViT-L/14) with a fixed **global-only** view.
Changes vs OG DeepSeek-OCR vLLM path (and the earlier version of this script):
  • Image is always resized/letterboxed to **1024×1024**.
  • **OG normalization** is applied: x = (x - 0.5) / 0.5  (RGB channels, after [0,1] scaling).
  • No local tiles/crops; **global view only**.
  • Projector now maps **2048 → 2048** (kept for modality-mixing; downstream MLP will map to decoder d_model).
  • Encoder returns **[HW, 2048]** tokens with grid (16,16); row/newline + final separator are added **downstream**.

Usage:
    python deepencoder_infer.py

Notes:
  • If you integrate with a downstream LLM: after you receive tokens of shape [256, 2048],
    insert 16 row-delimiters (one after each 16 tokens) and an optional final view-separator downstream.
    If your delimiter embeddings live in d_model space, map 2048→d_model first (with your downstream MLP),
    then append delimiters; or append 2048-dim delimiters and map everything together—be consistent.
"""

import math
import os
from pathlib import Path
import urllib.request
import ssl

import torch
import torch.nn.functional as F  # noqa: F401  (kept for parity; not used directly)
from PIL import Image
import numpy as np

# --- import your package modules ---
from deepencoder.lora_config import DeepEncoderLoRAConfig
from deepencoder.sam_vary_sdpa import build_sam_vit_b
from deepencoder.clip_sdpa import build_clip_l, VitModel
from deepencoder.build_linear import MlpProjector


class EasyDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

# Optional: OpenCLIP to source CLIP-L/14 weights
try:
    import open_clip
    _HAS_OPENCLIP = True
except Exception:
    _HAS_OPENCLIP = False


# =========================
# CONFIG (edit these as needed)
# =========================
CONFIG = {
    # Input image path
    "image": "/home/j_bindu/fyp-26-grp-38/Datasets/nuscenes/train/samples/CAM_FRONT/n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915243512465.jpg",

    # If None or file missing, we'll auto-download to ~/.cache/deepencoder/sam_vit_b_01ec64.pth
    "sam_ckpt": None,

    # Device & dtype
    "device": "cuda",            # "cuda" or "cpu"
    "dtype": "float32",          # "bfloat16" | "float32"

    # OpenCLIP pretrained tag (e.g., "openai", "laion2b_s32b_b82k")
    "openclip_pretrained": "openai",

    # Optional path to save tokens as .npy (or None)
    "save_npy": None,

    # Auto-download SAM weights if missing
    "auto_download_sam": True,
}

# Official Meta Segment-Anything SAM ViT-B checkpoint
SAM_VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_DEFAULT_NAME = "sam_vit_b_01ec64.pth"

# Fixed target grid for 1024×1024 global-only pipeline
FIXED_IMAGE_SIZE = 1024
FIXED_GRID_SIDE = 16  # 1024 / (16*4) = 16  (patch=16, downsample=4)


# ------------------------------
# Utility: downloader
# ------------------------------
def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = downloaded / total_size * 100 if total_size > 0 else 0
    bar_len = 30
    fill = int(bar_len * percent / 100)
    bar = "#" * fill + "-" * (bar_len - fill)
    print(f"\r[DOWNLOAD] [{bar}] {percent:6.2f}% ({downloaded/1e6:.2f} / {total_size/1e6:.2f} MB)", end="")


def download_sam_if_needed(sam_ckpt: str | None, auto_download: bool = True) -> str:
    """Ensure SAM ViT-B weights are present. If not, download to ~/.cache/deepencoder/.
    Returns the local path to the checkpoint.
    """
    # If a valid path was provided, use it.
    if sam_ckpt is not None and Path(sam_ckpt).exists():
        print(f"[INFO] Using provided SAM checkpoint: {sam_ckpt}")
        return sam_ckpt

    if not auto_download:
        raise FileNotFoundError(
            "SAM checkpoint not found and auto-download is disabled. "
            "Set CONFIG['sam_ckpt'] to a valid file or enable auto-download."
        )

    cache_dir = Path(os.path.expanduser("~/.cache/deepencoder"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest_path = cache_dir / SAM_DEFAULT_NAME

    if dest_path.exists():
        print(f"[INFO] Found cached SAM checkpoint: {dest_path}")
        return str(dest_path)

    print(f"[INFO] Downloading SAM ViT-B weights to: {dest_path}")
    # Some environments need this to avoid SSL inspection issues
    ctx = ssl.create_default_context()  # noqa: F841 (placeholder for custom handlers)
    try:
        urllib.request.urlretrieve(SAM_VIT_B_URL, dest_path, _progress_hook)
        print("\n[INFO] Download complete.")
    except Exception as e:
        # Clean up partial file
        if dest_path.exists():
            try:
                dest_path.unlink()
            except Exception:
                pass
        raise RuntimeError(
            f"Failed to download SAM weights from {SAM_VIT_B_URL}. Error: {e}"
        ) from e

    return str(dest_path)


# ------------------------------
# Utility: resize / pad to fixed 1024×1024 (letterbox)
# ------------------------------
def _resize_pad_square_1024(im: Image.Image) -> Image.Image:
    """Letterbox to 1024×1024 preserving aspect (center padding)."""
    target = FIXED_IMAGE_SIZE
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    scale = min(target / w, target / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    im_resized = im.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new(im.mode, (target, target), color=0)
    pad_left = (target - new_w) // 2
    pad_top = (target - new_h) // 2
    canvas.paste(im_resized, (pad_left, pad_top))
    return canvas


def _pil_to_tensor_og_norm(im: Image.Image) -> torch.Tensor:
    """PIL RGB -> FloatTensor [1,3,H,W] with **OG normalization** to [-1, 1].
    Steps: convert to [0,1], then (x - 0.5) / 0.5
    """
    if im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.array(im).astype(np.float32) / 255.0  # [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    t = (t - 0.5) / 0.5  # OG normalization → [-1,1]
    return t


# ------------------------------
# CLIP weight loading into your VitModel
# ------------------------------
def load_openclip_vitl14_into_vitmodel(
    vit: VitModel,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    openclip_pretrained: str = "openai",
):
    """
    Best-effort load of CLIP ViT-L/14 weights into your VitModel:
      - transformer blocks (attn qkv/out, mlp, layer norms)
      - class embedding
      - positional embedding
    Your VitModel bypasses CLIP's patch embed when SAM patch features are provided,
    so we skip CLIP's patch conv.
    """
    if not _HAS_OPENCLIP:
        print("[WARN] open_clip not found; skipping CLIP weight loading (leaving random init).")
        return

    print("[INFO] Loading CLIP ViT-L/14 (OpenCLIP, pretrained=%s)..." % openclip_pretrained)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=openclip_pretrained, device=device
    )
    sd = model.visual.state_dict()

    with torch.no_grad():
        # class embedding
        if "class_embedding" in sd and hasattr(vit.embeddings, "class_embedding"):
            vit.embeddings.class_embedding.copy_(sd["class_embedding"].to(dtype))

        # positional embedding: [1, 257, 1024] -> Embedding(num_positions, dim)
        if "positional_embedding" in sd and hasattr(vit.embeddings, "position_embedding"):
            pe = sd["positional_embedding"].to(dtype)  # [1, 257, 1024]
            if vit.embeddings.num_positions == pe.shape[1]:
                vit.embeddings.position_embedding.weight.copy_(pe.squeeze(0))
            else:
                n = min(vit.embeddings.num_positions, pe.shape[1])
                vit.embeddings.position_embedding.weight[:n].copy_(pe.squeeze(0)[:n])

        # transformer blocks
        my_blocks = vit.transformer.layers
        for i, block in enumerate(my_blocks):
            prefix = f"transformer.resblocks.{i}."
            # qkv
            qkv_w = sd.get(prefix + "attn.in_proj_weight", None)
            qkv_b = sd.get(prefix + "attn.in_proj_bias", None)
            if qkv_w is not None:
                block.self_attn.qkv_proj.weight.copy_(qkv_w.to(dtype))
            if qkv_b is not None:
                block.self_attn.qkv_proj.bias.copy_(qkv_b.to(dtype))
            # attn out
            out_w = sd.get(prefix + "attn.out_proj.weight", None)
            out_b = sd.get(prefix + "attn.out_proj.bias", None)
            if out_w is not None:
                block.self_attn.out_proj.weight.copy_(out_w.to(dtype))
            if out_b is not None:
                block.self_attn.out_proj.bias.copy_(out_b.to(dtype))
            # MLP
            fc1_w = sd.get(prefix + "mlp.c_fc.weight", None)
            fc1_b = sd.get(prefix + "mlp.c_fc.bias", None)
            fc2_w = sd.get(prefix + "mlp.c_proj.weight", None)
            fc2_b = sd.get(prefix + "mlp.c_proj.bias", None)
            if fc1_w is not None:
                block.mlp.fc1.weight.copy_(fc1_w.to(dtype))
            if fc1_b is not None:
                block.mlp.fc1.bias.copy_(fc1_b.to(dtype))
            if fc2_w is not None:
                block.mlp.fc2.weight.copy_(fc2_w.to(dtype))
            if fc2_b is not None:
                block.mlp.fc2.bias.copy_(fc2_b.to(dtype))
            # LNs
            ln1_w = sd.get(prefix + "ln_1.weight", None)
            ln1_b = sd.get(prefix + "ln_1.bias", None)
            ln2_w = sd.get(prefix + "ln_2.weight", None)
            ln2_b = sd.get(prefix + "ln_2.bias", None)
            if ln1_w is not None:
                block.layer_norm1.weight.copy_(ln1_w.to(dtype))
            if ln1_b is not None:
                block.layer_norm1.bias.copy_(ln1_b.to(dtype))
            if ln2_w is not None:
                block.layer_norm2.weight.copy_(ln2_w.to(dtype))
            if ln2_b is not None:
                block.layer_norm2.bias.copy_(ln2_b.to(dtype))

    print("[INFO] CLIP weights loaded where shapes matched. Unmatched params stay randomly-initialized.")


# ------------------------------
# ------------------------------
# Inference-only helper (kept): SAM ➜ CLIP ➜ concat ➜ projector (2048→2048)
# This keeps @torch.no_grad() **only** for the standalone helper.
# Training should use DeepEncoderRuntime below.
@torch.no_grad()
def deepencoder_infer(
    image_path: str,
    sam_ckpt: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    openclip_pretrained: str = "openai",
):
    # 1) Load and prep image → fixed 1024×1024 with OG normalization
    img = Image.open(image_path)
    img = _resize_pad_square_1024(img)
    x = _pil_to_tensor_og_norm(img).to(device=device, dtype=dtype)  # [1,3,1024,1024], [-1,1]

    # 2) Build SAM ViT-B and load checkpoint
    sam = build_sam_vit_b(checkpoint=sam_ckpt).to(device=device, dtype=dtype)
    sam.eval()

    # 3) Build CLIP-L/14 and (optionally) load OpenCLIP weights
    clip_vit = build_clip_l().to(device=device, dtype=dtype)
    clip_vit.eval()
    load_openclip_vitl14_into_vitmodel(
        clip_vit, device=device, dtype=dtype, openclip_pretrained=openclip_pretrained
    )

    # 4) Build projector (now 2048 -> 2048)
    projector = MlpProjector(EasyDict(projector_type="linear", input_dim=2048, n_embed=2048)).to(device=device, dtype=dtype)
    projector.eval()

    # 5) SAM features (global-only)
    sam_feats = sam(x)  # [B, 1024, Hs, Ws]

    # 6) CLIP tokens conditioned on SAM patch features
    clip_y = clip_vit(x, sam_feats)  # [B, 1+HW, 1024]

    # 7) Fuse: concat(CLIP_no_CLS, SAM_flat) -> projector (2048→2048)
    clip_tokens = clip_y[:, 1:, :]                      # [B, HW, 1024]
    sam_tokens  = sam_feats.flatten(2).permute(0, 2, 1) # [B, HW, 1024]
    fused = torch.cat([clip_tokens, sam_tokens], dim=-1)      # [B, HW, 2048]
    vision_tokens = projector(fused)                          # [B, HW, 2048]

    side = FIXED_GRID_SIDE  # 16
    return {
        "vision_tokens": vision_tokens,     # [B, 256, 2048]
        "grid": (side, side),               # (16, 16)
        "image_size": FIXED_IMAGE_SIZE,     # 1024
        "normalization": "og_0.5_mean_0.5_std",
    }


def _to_dtype(s: str) -> torch.dtype:
    if s.lower() in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s.lower() in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {s}")


# ======== Optional: multi-view helpers (kept, now returning 2048-dim tokens) ========
from typing import List, Optional, Sequence, Tuple
from nuscenes.nuscenes import NuScenes

# Fixed 6-view order (front, front_right, front_left, back, back_right, back_left)
DEFAULT_VIEW_ORDER: Tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
)


def resolve_cam_image_paths(
    nusc: NuScenes,
    sample_token: str,
    view_order: Sequence[str] = DEFAULT_VIEW_ORDER,
) -> List[Optional[Path]]:
    """Resolve absolute image paths for the specified views from a nuScenes sample token."""
    sample = nusc.get("sample", sample_token)
    out: List[Optional[Path]] = []
    for cam in view_order:
        sd_tok = sample["data"].get(cam, None)
        if not sd_tok:
            out.append(None)
            continue
        sd = nusc.get("sample_data", sd_tok)
        p = (Path(nusc.dataroot) / sd["filename"]).resolve()
        out.append(p if p.exists() else None)
    return out

class DeepEncoderRuntime:
    """
    Train-ready runtime:
      • SAM is **frozen** and always runs under no_grad() + eval().
      • CLIP is **trainable**; optionally wrapped with LoRA (PEFT).
      • Projector is **trainable**.
      • encode_image / encode_views DO NOT disable grad for CLIP/projector.
    """

    def __init__(
        self,
        *,
        sam_ckpt: Optional[str] = None,
        auto_download_sam: bool = True,
        device: str = "cuda",
        dtype: torch.dtype | str = torch.bfloat16,
        openclip_pretrained: str = "openai",
        lora_config: DeepEncoderLoRAConfig = DeepEncoderLoRAConfig(),
        freeze_clip_backbone_when_lora_enabled: bool = True,
    ):
        self.image_size = FIXED_IMAGE_SIZE
        self.grid = (FIXED_GRID_SIDE, FIXED_GRID_SIDE)
        self.device = device
        self.dtype = _to_dtype(dtype) if isinstance(dtype, str) else dtype
        self.lora_config = lora_config
        self.freeze_clip_backbone_when_lora_enabled = freeze_clip_backbone_when_lora_enabled

        # Ensure SAM weights exist
        ckpt = download_sam_if_needed(sam_ckpt, auto_download=auto_download_sam)

        # -------- SAM (always frozen) --------
        self.sam = build_sam_vit_b(checkpoint=ckpt).to(device=self.device, dtype=self.dtype)
        for p in self.sam.parameters():
            p.requires_grad = False
        self.sam.eval()  # frozen

        # -------- CLIP (trainable, optionally LoRA) --------
        self.clip_vit = build_clip_l().to(device=self.device, dtype=self.dtype)

        # Optionally load OpenCLIP weights (unchanged)
        load_openclip_vitl14_into_vitmodel(
            self.clip_vit, device=self.device, dtype=self.dtype, openclip_pretrained=openclip_pretrained
        )

        if self.lora_config.enabled:
            if not _HAS_PEFT:
                raise RuntimeError("LoRA requested but 'peft' is not installed.")
            lcfg = LoraConfig(
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                bias=self.lora_config.bias,
                target_modules=self.lora_config.target_modules,
            )
            self.clip_vit = get_peft_model(self.clip_vit, lcfg)
            # Optionally freeze the non-LoRA CLIP backbone params:
            if self.freeze_clip_backbone_when_lora_enabled:
                for n, p in self.clip_vit.named_parameters():
                    # LoRA-added params have requires_grad=True already.
                    # We conservatively freeze everything else.
                    if "lora_" not in n:
                        p.requires_grad = False

        # -------- Projector (trainable) --------
        self.projector = MlpProjector(
            EasyDict(projector_type="linear", input_dim=2048, n_embed=2048)
        ).to(device=self.device, dtype=self.dtype)

    def train(self):
        """Set train mode for trainable parts."""
        # SAM stays eval (frozen)
        self.clip_vit.train()
        self.projector.train()
        return self

    def eval(self):
        """Set eval mode for all modules."""
        self.sam.eval()
        self.clip_vit.eval()
        self.projector.eval()
        return self

    def trainable_parameters(self):
        """Return only parameters that should be optimized (CLIP/LoRA + projector)."""
        params = []
        for p in self.clip_vit.parameters():
            if p.requires_grad:
                params.append(p)
        params += list(self.projector.parameters())
        return params

    @torch.no_grad()
    def _sam_features(self, x: torch.Tensor) -> torch.Tensor:
        """Private helper: SAM forward under no_grad()."""
        return self.sam(x)  # [B, 1024, Hs, Ws]

    def encode_image(self, image_path: str) -> dict:
        """Returns tokens for a single image (train-ready; grads flow through CLIP+projector)."""
        img = Image.open(image_path)
        img = _resize_pad_square_1024(img)
        x = _pil_to_tensor_og_norm(img).to(device=self.device, dtype=self.dtype)  # [1,3,1024,1024]

        # SAM features (frozen)
        sam_feats = self._sam_features(x)

        # CLIP tokens conditioned on SAM (trainable)
        clip_y = self.clip_vit(x, sam_feats)                # [B, 1+HW, 1024]
        clip_tokens = clip_y[:, 1:, :]                      # [B, HW, 1024]
        sam_tokens  = sam_feats.flatten(2).permute(0, 2, 1) # [B, HW, 1024]

        fused = torch.cat([clip_tokens, sam_tokens], dim=-1)      # [B, HW, 2048]
        vision_tokens = self.projector(fused)                      # [B, HW, 2048]

        vt = vision_tokens.squeeze(0)  # [HW, 2048]
        return {"tokens": vt, "grid": self.grid, "image_size": self.image_size}

    def encode_views(
        self,
        image_paths: Sequence[Optional[Path]],
        *,
        strict: bool = True,
        view_order: Sequence[str] = DEFAULT_VIEW_ORDER,
    ) -> dict:
        """Encode multiple camera views. Output tokens are **[HW, 2048]** per view.
        Missing views -> zeros (unless strict=True, which raises).
        TODO : Change actual 0s to fall-back incase of missing views
        """
        tokens_list: List[Optional[torch.Tensor]] = []
        present_mask: List[bool] = []
        first_shape: Optional[Tuple[int, int]] = None

        for p in image_paths:
            if p is not None and Path(p).exists():
                out_i = self.encode_image(str(p))
                t = out_i["tokens"]  # [HW, 2048]
                tokens_list.append(t)
                present_mask.append(True)
                if first_shape is None:
                    first_shape = tuple(t.shape)
            else:
                if strict:
                    raise FileNotFoundError(f"Missing view file: {p}")
                tokens_list.append(None)
                present_mask.append(False)

        if first_shape is None:
            raise RuntimeError("No available camera views to infer token shape.")

        HW, D = first_shape
        for i, t in enumerate(tokens_list):
            if t is None:
                tokens_list[i] = torch.zeros((HW, D), device=self.device, dtype=self.dtype)

        return {
            "tokens": tokens_list,
            "present_mask": present_mask,
            "view_names": list(view_order),
            "grid": self.grid,
            "image_size": self.image_size,
        }

def multiview_tokens_from_sample_token(
    sample_token: str,
    nusc: NuScenes,
    *,
    runtime: Optional[DeepEncoderRuntime] = None,
    view_order: Sequence[str] = DEFAULT_VIEW_ORDER,
    strict: bool = False,
    # The kwargs below are only used if runtime is None:
    sam_ckpt: Optional[str] = None,
    auto_download_sam: bool = True,
    device: str = "cuda",
    dtype: torch.dtype | str = torch.bfloat16,
    openclip_pretrained: str = "openai",
) -> dict:
    """Convenience helper that returns the same dict as DeepEncoderRuntime.encode_views(), plus the runtime."""
    if runtime is None:
        runtime = DeepEncoderRuntime(
            sam_ckpt=sam_ckpt,
            auto_download_sam=auto_download_sam,
            device=device,
            dtype=dtype,
            openclip_pretrained=openclip_pretrained,
        )

    img_paths = resolve_cam_image_paths(nusc, sample_token, view_order=view_order)
    out = runtime.encode_views(img_paths, strict=strict, view_order=view_order)
    out["runtime"] = runtime
    return out


if __name__ == "__main__":
    # Resolve config
    image_path = CONFIG["image"]
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Ensure SAM weights
    sam_ckpt = CONFIG.get("sam_ckpt", None)
    sam_ckpt = download_sam_if_needed(sam_ckpt, auto_download=CONFIG.get("auto_download_sam", True))

    device = CONFIG.get("device", "cuda")
    dtype = _to_dtype(CONFIG.get("dtype", "bfloat16"))
    openclip_pretrained = CONFIG.get("openclip_pretrained", "openai")
    save_npy = CONFIG.get("save_npy", None)

    out = deepencoder_infer(
        image_path=image_path,
        sam_ckpt=sam_ckpt,
        device=device,
        dtype=dtype,
        openclip_pretrained=openclip_pretrained,
    )

    vt = out["vision_tokens"].squeeze(0)  # [256, 2048]
    print(
        f"[OK] Vision tokens: shape={tuple(vt.shape)} grid={out['grid']} image_size={out['image_size']} norm={out['normalization']}"
    )
    if save_npy:
        np.save(save_npy, vt.detach().cpu().to(torch.float32).numpy())
        print(f"[SAVED] {save_npy}")
