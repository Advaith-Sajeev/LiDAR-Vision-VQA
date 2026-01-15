#!/usr/bin/env python3
"""
DeepEncoder inference (CLIP ViT-B/16, frozen) with 6-view global images.
Changes vs previous SAM+CLIP hybrid:
    • SAM is removed; only CLIP ViT-B/16 patch tokens are used (frozen backbone).
    • Images are resized with preserved aspect ratio and padded to 384×384.
    • CLIP normalization (OpenAI) is applied on 0-1 scale.
    • Projector maps **768 → output_dim** (matches LLM hidden size, e.g., 896).
    • Encoder returns **[576, output_dim]** tokens per view (24×24 grid); per-view delimiters stay downstream.

Usage:
        python deepencoder_infer.py

Notes:
    • Downstream keeps the same delimiter logic as before (start/end tokens per view).
"""

import math
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Sequence
import urllib.request
import ssl

import torch
import torch.nn.functional as F  # noqa: F401  (kept for parity; not used directly)
from PIL import Image
import numpy as np

# Import shared debug logger from training utilities
from deepencoder.debug import debug

_MODULE = "deepencoder"

# --- import your package modules ---
from deepencoder.clip_sdpa import build_clip_b16, VitModel, get_abs_pos
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
# Constants
# =========================

# Fixed target grid for 384x384 pipeline
FIXED_IMAGE_SIZE = 384
FIXED_GRID_SIDE = 24  # 24×24 = 576 tokens per view


# ------------------------------
# Utility: resize+pad to fixed 384×384
# ------------------------------
def resize_and_pad_to_square(im: Image.Image, target: int = FIXED_IMAGE_SIZE) -> Image.Image:
    """
    Resize with preserved aspect ratio so the image fits inside target×target,
    then pad with black pixels to get exactly target×target.
    No non-uniform scaling (no stretching).
    """
    if im.mode != "RGB":
        im = im.convert("RGB")

    w, h = im.size

    # Uniform scale factor so both dimensions <= target
    scale = min(target / w, target / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Clamp in case of rounding edge cases
    new_w = min(new_w, target)
    new_h = min(new_h, target)

    im_resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create square canvas and center the resized image
    canvas = Image.new("RGB", (target, target), color=0)
    pad_left = (target - new_w) // 2
    pad_top = (target - new_h) // 2
    canvas.paste(im_resized, (pad_left, pad_top))

    return canvas


# CLIP (OpenAI) normalization parameters (on 0-1 scale)
CLIP_PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]


def _pil_to_tensor_clip_norm(im: Image.Image, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """PIL RGB -> FloatTensor [1,3,H,W] with CLIP normalization."""
    if im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.array(im, dtype=np.float32) / 255.0  # [0,1]
    mean = np.array(CLIP_PIXEL_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(CLIP_PIXEL_STD, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    if dtype != torch.float32:
        t = t.to(dtype=dtype)
    return t



def load_and_preprocess_image(
    image_path: Optional[Path], 
    dtype: torch.dtype = torch.float32
) -> Optional[torch.Tensor]:
    """
    Load and preprocess a single image for vision encoding.
    This function is designed to be called in DataLoader workers (CPU-bound).
    
    Args:
        image_path: Path to the image file, or None if missing
        dtype: Target dtype for the tensor (default: float32)
               Pass target dtype (e.g., bfloat16) to save memory
        
    Returns:
        Preprocessed tensor [1, 3, 384, 384], or None if loading fails
    """
    if image_path is None or not Path(image_path).exists():
        return None
    try:
        img = Image.open(str(image_path))
        img = resize_and_pad_to_square(img)
        x = _pil_to_tensor_clip_norm(img, dtype=dtype)  # [1, 3, 384, 384]
        return x
    except Exception as e:
        debug.warn(_MODULE, f"Failed to load image {image_path}: {e}")
        return None


# ------------------------------
# CLIP weight loading into your VitModel
# ------------------------------
def load_openclip_vit_into_vitmodel(
    vit: VitModel,
    *,
    model_name: str = "ViT-B-16",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    openclip_pretrained: str = "openai",
):
    """Best-effort load of CLIP ViT weights (OpenCLIP) into our VitModel."""
    if not _HAS_OPENCLIP:
        debug.warn(_MODULE, "open_clip not found; skipping CLIP weight loading (leaving random init).")
        return

    debug.info(_MODULE, f"Loading CLIP {model_name} (OpenCLIP, pretrained={openclip_pretrained})...")
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=openclip_pretrained, device=device
        )
        sd = model.visual.state_dict()
        if not sd:
            debug.warn(_MODULE, "OpenCLIP returned empty state dict; skipping weight loading (leaving random init).")
            return
    except (StopIteration, RuntimeError, Exception) as e:
        debug.warn(_MODULE, f"Failed to load OpenCLIP weights: {e}; skipping weight loading (leaving random init).")
        return

    with torch.no_grad():
        if "class_embedding" in sd and hasattr(vit.embeddings, "class_embedding"):
            vit.embeddings.class_embedding.copy_(sd["class_embedding"].to(dtype))

        if "positional_embedding" in sd and hasattr(vit.embeddings, "position_embedding"):
            pe = sd["positional_embedding"].to(dtype)  # [Nsrc, C]
            pe = pe.unsqueeze(0)  # [1, Nsrc, C]
            pe_resampled = get_abs_pos(pe, vit.embeddings.num_positions)  # [1, Ntgt, C]
            pe_resampled = pe_resampled.squeeze(0)
            n = min(vit.embeddings.num_positions, pe_resampled.shape[0])
            vit.embeddings.position_embedding.weight[:n].copy_(pe_resampled[:n])

        my_blocks = vit.transformer.layers
        for i, block in enumerate(my_blocks):
            prefix = f"transformer.resblocks.{i}."
            qkv_w = sd.get(prefix + "attn.in_proj_weight", None)
            qkv_b = sd.get(prefix + "attn.in_proj_bias", None)
            if qkv_w is not None:
                block.self_attn.qkv_proj.weight.copy_(qkv_w.to(dtype))
            if qkv_b is not None:
                block.self_attn.qkv_proj.bias.copy_(qkv_b.to(dtype))

            out_w = sd.get(prefix + "attn.out_proj.weight", None)
            out_b = sd.get(prefix + "attn.out_proj.bias", None)
            if out_w is not None:
                block.self_attn.out_proj.weight.copy_(out_w.to(dtype))
            if out_b is not None:
                block.self_attn.out_proj.bias.copy_(out_b.to(dtype))

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

    debug.debug(_MODULE, "CLIP weights loaded where shapes matched. Unmatched params stay randomly-initialized.")

def _to_dtype(s: str) -> torch.dtype:
    """Convert string dtype specification to torch.dtype.
    
    Supported values:
        - "bf16", "bfloat16" → torch.bfloat16
        - "fp16", "float16" → torch.float16
        - "fp32", "float32" → torch.float32
    """
    if s.lower() in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s.lower() in ("fp16", "float16"):
        return torch.float16
    if s.lower() in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {s}. Supported: bf16/bfloat16, fp16/float16, fp32/float32")


# ======== Optional: multi-view helpers (kept, now returning output_dim tokens) ========
from nuscenes.nuscenes import NuScenes

# Import from centralized config
from configs.constants import DEFAULT_VIEW_ORDER, PROJECTOR_DIM


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
    """Train-ready runtime for the CLIP-only encoder (frozen backbone + trainable projector)."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        dtype: torch.dtype | str = torch.bfloat16,
        openclip_pretrained: str = "openai",
        output_dim: int = PROJECTOR_DIM,
    ):
        self.image_size = FIXED_IMAGE_SIZE
        self.grid = (FIXED_GRID_SIDE, FIXED_GRID_SIDE)
        self.device = device
        self.dtype = _to_dtype(dtype) if isinstance(dtype, str) else dtype
        self.output_dim = output_dim

        # -------- CLIP (frozen) --------
        self.clip_vit = build_clip_b16().to(device=self.device, dtype=self.dtype)
        
        # Use -quickgelu suffix for OpenAI weights to avoid warnings
        target_model_name = "ViT-B-16-quickgelu" if openclip_pretrained == "openai" else "ViT-B-16"
        
        load_openclip_vit_into_vitmodel(
            self.clip_vit,
            model_name=target_model_name,
            device=self.device,
            dtype=self.dtype,
            openclip_pretrained=openclip_pretrained,
        )
        for p in self.clip_vit.parameters():
            p.requires_grad = False
        self.clip_vit.eval()

        # -------- Projector (trainable, maps 768 → output_dim) --------
        self.projector = MlpProjector(
            EasyDict(projector_type="linear", input_dim=768, n_embed=self.output_dim)
        ).to(device=self.device, dtype=self.dtype)

    def train(self):
        """Set train mode for trainable parts."""
        self.clip_vit.eval()  # keep frozen
        self.projector.train()
        return self

    def eval(self):
        """Set eval mode for all modules."""
        self.clip_vit.eval()
        self.projector.eval()
        return self

    def parameters(self):
        """Return trainable parameters (projector only; CLIP is frozen)."""
        for p in self.projector.parameters():
            if p.requires_grad:
                yield p

    def named_parameters(self):
        """Return trainable named parameters."""
        for n, p in self.projector.named_parameters():
            if p.requires_grad:
                yield f"projector.{n}", p

    def encode_image(self, image_path: str) -> dict:
        """Returns tokens for a single image (projected CLIP patch tokens)."""
        debug.trace(_MODULE, f"📷 Loading image: {image_path}")
        img = Image.open(image_path)
        original_size = img.size
        img = resize_and_pad_to_square(img)
        x = _pil_to_tensor_clip_norm(img, dtype=self.dtype).to(device=self.device)  # [1,3,384,384]
        debug.trace(_MODULE, f"📐 Image: {original_size} → {img.size} → tensor {x.shape} dtype={x.dtype}")

        clip_model = self.clip_vit.base_model.model if hasattr(self.clip_vit, 'base_model') else self.clip_vit
        debug.trace(_MODULE, "🔄 Running CLIP encoder (frozen)...")
        clip_y = clip_model(x)  # [B, 1+HW, 768]
        debug.trace(_MODULE, f"📐 CLIP output: {clip_y.shape} dtype={clip_y.dtype}")

        clip_tokens = clip_y[:, 1:, :]  # [B, HW, 768] - remove CLS token
        vision_tokens = self.projector(clip_tokens)  # [B, HW, output_dim]
        debug.trace(_MODULE, f"📐 After projector (768→{self.output_dim}): {vision_tokens.shape}")

        vt = vision_tokens.squeeze(0)  # [HW, output_dim]
        debug.trace(_MODULE, f"✓ Vision encoding complete: {vt.shape}")
        return {"tokens": vt, "grid": self.grid, "image_size": self.image_size}

    def encode_views(
        self,
        image_paths: Sequence[Optional[Path]],
        *,
        strict: bool = True,
        view_order: Sequence[str] = DEFAULT_VIEW_ORDER,
    ) -> dict:
        """Encode multiple camera views. Output tokens are **[HW, output_dim]** per view.
        Missing views -> zeros (unless strict=True, which raises).
        TODO : Change actual 0s to fall-back incase of missing views
        """
        debug.debug(_MODULE, f"🎥 Encoding {len(image_paths)} camera views...")
        tokens_list: List[Optional[torch.Tensor]] = []
        present_mask: List[bool] = []
        first_shape: Optional[Tuple[int, int]] = None

        for i, p in enumerate(image_paths):
            view_name = view_order[i] if i < len(view_order) else 'unknown'
            debug.trace(_MODULE, f"Processing view {i+1}/{len(image_paths)}: {view_name}")
            if p is not None and Path(p).exists():
                out_i = self.encode_image(str(p))
                t = out_i["tokens"]  # [HW, output_dim]
                tokens_list.append(t)
                present_mask.append(True)
                debug.trace(_MODULE, f"✓ View {i+1} ({view_name}) encoded: {t.shape}")
                if first_shape is None:
                    first_shape = tuple(t.shape)
            else:
                if strict:
                    raise FileNotFoundError(f"Missing view file: {p}")
                debug.trace(_MODULE, f"⚠ View {i+1} ({view_name}) missing, using zeros")
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

    def encode_images_batch(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Batch encode multiple images through CLIP in a single forward pass.

        Args:
            images: List of tensors, each [1, 3, 384, 384] (preprocessed images)
                    OR a single batched tensor [N, 3, 384, 384]

        Returns:
            vision_tokens: [N, HW, output_dim] tensor of vision tokens
        """
        # Stack if list of tensors
        if isinstance(images, list):
            x = torch.cat(images, dim=0)  # [N, 3, 384, 384]
        else:
            x = images
        
        N = x.shape[0]
        debug.debug(_MODULE, f"🚀 Batch encoding {N} images...")

        clip_model = self.clip_vit.base_model.model if hasattr(self.clip_vit, 'base_model') else self.clip_vit
        clip_y = clip_model(x)  # [N, 1+HW, 768]
        debug.trace(_MODULE, f"📐 CLIP batch output: {clip_y.shape}")

        clip_tokens = clip_y[:, 1:, :]  # [N, HW, 768] - remove CLS token
        vision_tokens = self.projector(clip_tokens)  # [N, HW, output_dim]
        
        debug.debug(_MODULE, f"✓ Batch encoding complete: {vision_tokens.shape}")
        return vision_tokens

    def encode_preloaded_views_batch(
        self,
        batch_images: List[List[Optional[torch.Tensor]]],
        view_order: Sequence[str] = DEFAULT_VIEW_ORDER,
    ) -> List[List[torch.Tensor]]:
        """
        Encode pre-loaded image tensors (from DataLoader workers) through CLIP.
        This is the fast path - images are already loaded, just need GPU encoding.
        
        Args:
            batch_images: List of B samples, each containing V image tensors [1, 3, 384, 384]
                         None entries indicate missing views
            view_order: Order of camera views (for sizing)
        
        Returns:
            List of B samples, each containing V tensors [HW, output_dim]
        """
        B = len(batch_images)
        V = len(view_order)
        
        # Collect all valid images and their positions
        all_images = []
        valid_positions = []
        
        for b_idx, sample_images in enumerate(batch_images):
            for v_idx, img_tensor in enumerate(sample_images):
                if img_tensor is not None:
                    all_images.append(img_tensor)
                    valid_positions.append((b_idx, v_idx))
        
        if not all_images:
            # All missing - return zeros
            HW = self.grid[0] * self.grid[1]  # 576 tokens for 24x24 grid
            D = self.output_dim
            return [[torch.zeros((HW, D), device=self.device, dtype=self.dtype) for _ in range(V)] for _ in range(B)]
        
        debug.debug(_MODULE, f"🚀 Encoding {len(all_images)} pre-loaded images (batch={B}, views={V})")
        
        # Concatenate and move to GPU (single transfer)
        batch_tensor = torch.cat(all_images, dim=0).to(device=self.device, dtype=self.dtype)  # [N_valid, 3, 384, 384]
        
        # Batch encode all valid images on GPU
        all_tokens = self.encode_images_batch(batch_tensor)  # [N_valid, HW, output_dim]
        
        HW, D = all_tokens.shape[1], all_tokens.shape[2]
        
        # Initialize output with zeros
        results = [[torch.zeros((HW, D), device=self.device, dtype=self.dtype) for _ in range(V)] for _ in range(B)]
        
        # Fill in valid tokens
        for idx, (b_idx, v_idx) in enumerate(valid_positions):
            results[b_idx][v_idx] = all_tokens[idx]
        
        debug.debug(_MODULE, f"✓ Pre-loaded batch encoding complete")
        return results

    def encode_views_batch(
        self,
        batch_image_paths: List[List[Optional[Path]]],
        *,
        strict: bool = False,
        view_order: Sequence[str] = DEFAULT_VIEW_ORDER,
    ) -> List[List[torch.Tensor]]:
        """
        Batch encode vision tokens for multiple samples, each with multiple views.
        Processes all images in a single forward pass for efficiency.
        
        Args:
            batch_image_paths: List of B samples, each containing V image paths (6 views)
                              e.g., [[sample1_view1, ..., sample1_view6], [sample2_view1, ...], ...]
            strict: If True, raise error on missing views
            view_order: Order of camera views
        
        Returns:
            List of B samples, each containing V tensors [HW, output_dim]
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        B = len(batch_image_paths)
        V = len(view_order)
        
        # Helper function to load and preprocess a single image (CPU-bound, parallelizable)
        def load_and_preprocess(args):
            """Load image from path and preprocess (runs in thread pool)."""
            b_idx, v_idx, p = args
            if p is None or not Path(p).exists():
                return (b_idx, v_idx, None)
            try:
                img = Image.open(str(p))
                img = resize_and_pad_to_square(img)
                # Convert to tensor on CPU with target dtype
                # Using target dtype directly saves memory during torch.cat() later
                x = _pil_to_tensor_clip_norm(img, dtype=self.dtype)  # [1, 3, 384, 384] on CPU
                return (b_idx, v_idx, x)
            except Exception as e:
                debug.warn(_MODULE, f"Failed to load image {p}: {e}")
                return (b_idx, v_idx, None)
        
        # Collect all image paths with their indices
        load_tasks = []
        for b_idx, sample_paths in enumerate(batch_image_paths):
            for v_idx, p in enumerate(sample_paths):
                load_tasks.append((b_idx, v_idx, p))
        
        # Parallel image loading using ThreadPoolExecutor
        # I/O-bound tasks benefit from threading (GIL released during I/O)
        num_workers = min(16, len(load_tasks))  # Cap at 16 threads
        debug.debug(_MODULE, f"Loading {len(load_tasks)} images with {num_workers} threads...")
        
        all_images = []
        valid_positions = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(load_and_preprocess, task): task for task in load_tasks}
            
            # Collect results as they complete
            for future in as_completed(futures):
                b_idx, v_idx, tensor = future.result()
                if tensor is not None:
                    all_images.append(tensor)
                    valid_positions.append((b_idx, v_idx))
                elif strict:
                    task = futures[future]
                    raise FileNotFoundError(f"Missing view file: {task[2]}")
        
        if not all_images:
            raise RuntimeError("No valid images found in batch")
        
        debug.debug(_MODULE, f"Loaded {len(all_images)} valid images out of {B * V}")
        
        # Move to GPU (already in target dtype from load_and_preprocess)
        batch_tensor = torch.cat(all_images, dim=0).to(device=self.device)  # [N_valid, 3, 384, 384]
        
        # Batch encode all valid images on GPU
        all_tokens = self.encode_images_batch(batch_tensor)  # [N_valid, HW, output_dim]
        
        HW, D = all_tokens.shape[1], all_tokens.shape[2]
        
        # Initialize output with zeros
        results = [[torch.zeros((HW, D), device=self.device, dtype=self.dtype) for _ in range(V)] for _ in range(B)]
        
        # Fill in valid tokens
        for idx, (b_idx, v_idx) in enumerate(valid_positions):
            results[b_idx][v_idx] = all_tokens[idx]
        
        return results


def batch_multiview_tokens_from_sample_tokens(
    sample_tokens: List[str],
    nusc: NuScenes,
    *,
    runtime: DeepEncoderRuntime,
    view_order: Sequence[str] = DEFAULT_VIEW_ORDER,
    strict: bool = False,
) -> List[List[torch.Tensor]]:
    """
    Batch process multiple sample tokens to get vision tokens for all views.
    Much more efficient than calling multiview_tokens_from_sample_token in a loop.
    
    Args:
        sample_tokens: List of nuScenes sample tokens
        nusc: NuScenes dataset object
        runtime: DeepEncoderRuntime instance
        view_order: Camera view order
        strict: Raise on missing views
    
    Returns:
        List of B samples, each containing 6 view tensors [HW, output_dim]
    """
    # Resolve all image paths
    batch_paths = []
    for tok in sample_tokens:
        paths = resolve_cam_image_paths(nusc, tok, view_order=view_order)
        batch_paths.append(paths)
    
    # Batch encode all images
    return runtime.encode_views_batch(batch_paths, strict=strict, view_order=view_order)


def multiview_tokens_from_sample_token(
    sample_token: str,
    nusc: NuScenes,
    *,
    runtime: Optional[DeepEncoderRuntime] = None,
    view_order: Sequence[str] = DEFAULT_VIEW_ORDER,
    strict: bool = False,
    device: str = "cuda",
    dtype: torch.dtype | str = torch.bfloat16,
    openclip_pretrained: str = "openai",
    output_dim: int = PROJECTOR_DIM,
) -> dict:
    """Convenience helper that returns the same dict as DeepEncoderRuntime.encode_views(), plus the runtime."""
    if runtime is None:
        runtime = DeepEncoderRuntime(
            device=device,
            dtype=dtype,
            openclip_pretrained=openclip_pretrained,
            output_dim=output_dim,
        )

    img_paths = resolve_cam_image_paths(nusc, sample_token, view_order=view_order)
    out = runtime.encode_views(img_paths, strict=strict, view_order=view_order)
    out["runtime"] = runtime
    return out


if __name__ == "__main__":
    print("This module is intended to be imported, not run directly.")
    print("To test DeepEncoder inference, run:")
    print("  python tests/test_deepencoder_infer.py")
    print("\nFor usage examples, see the test file or import DeepEncoderRuntime:")
    print("  from deepencoder.deepencoder_infer import DeepEncoderRuntime")

