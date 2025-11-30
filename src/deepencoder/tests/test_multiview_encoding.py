#!/usr/bin/env python3
"""
Tests for multiview encoding functionality in deepencoder_infer.py

This module tests:
- encode_views() method
- encode_images_batch() method
- encode_views_batch() method
- encode_preloaded_views_batch() method
- resolve_cam_image_paths() function
- batch_multiview_tokens_from_sample_tokens() function
- Image preprocessing utilities
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

# Add src/ directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import tempfile


class EasyDict(dict):
    """Simple dict that allows attribute access."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


# ---------- Lightweight stand-ins for heavy models ----------

class DummySAM(nn.Module):
    """Minimal SAM-like encoder for testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, kernel_size=1)
        # Add net_2 and net_3 for the compression head
        self.net_2 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.net_3 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = F.adaptive_avg_pool2d(y, (16, 16))
        return y


class DummyCLIP(nn.Module):
    """Minimal CLIP-ViT-like head for testing."""
    def __init__(self):
        super().__init__()
        self.embed_dim = 1024
        self.embeddings = nn.Module()
        self.embeddings.class_embedding = nn.Parameter(torch.zeros(1024))
        self.embeddings.position_embedding = nn.Embedding(257, 1024)
        self.embeddings.num_positions = 257
        self.embeddings.patch_embedding = nn.Conv2d(3, 1024, kernel_size=14, stride=14)
        
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList()

    def forward(self, x: torch.Tensor, patch_embeds: torch.Tensor) -> torch.Tensor:
        B, C, H, W = patch_embeds.shape
        tokens = patch_embeds.flatten(2).transpose(1, 2)
        cls = torch.zeros(B, 1, self.embed_dim, device=x.device, dtype=x.dtype)
        return torch.cat([cls, tokens], dim=1)


class DummyProjector(nn.Module):
    """Minimal projector for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2048, 2048)
    
    def forward(self, x):
        return self.linear(x)


# ---------- Fixtures ----------

@pytest.fixture
def temp_test_images():
    """Create temporary test images for multiview testing."""
    images = {}
    temp_dir = tempfile.mkdtemp()
    
    for i, cam in enumerate(["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
                              "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]):
        img = Image.new("RGB", (1600, 900), color=(i * 40, i * 30, i * 20))
        path = Path(temp_dir) / f"{cam}.jpg"
        img.save(path)
        images[cam] = path
    
    yield images
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def patched_runtime(monkeypatch):
    """Create a patched DeepEncoderRuntime for testing."""
    import deepencoder.deepencoder_infer as dei
    
    # Avoid download / filesystem dependency
    def _fake_download_sam_if_needed(sam_ckpt, auto_download=True):
        return "/tmp/dummy_sam_checkpoint.pth"

    monkeypatch.setattr(dei, "download_sam_if_needed", _fake_download_sam_if_needed)

    # Plug in DummySAM
    def _fake_build_sam_vit_b(checkpoint=None):
        return DummySAM()

    monkeypatch.setattr(dei, "build_sam_vit_b", _fake_build_sam_vit_b)

    # Plug in DummyCLIP
    def _fake_build_clip_l():
        return DummyCLIP()

    monkeypatch.setattr(dei, "build_clip_l", _fake_build_clip_l)

    runtime = dei.DeepEncoderRuntime(
        sam_ckpt=None,
        auto_download_sam=False,
        device="cpu",
        dtype="float32",
        openclip_pretrained="openai",
    )
    runtime.eval()
    return runtime


# ---------- Test: Image Preprocessing ----------

class TestImagePreprocessing:
    """Test image preprocessing utilities."""
    
    def test_resize_and_pad_to_square(self):
        """Test resize_and_pad_to_square function."""
        from deepencoder.deepencoder_infer import resize_and_pad_to_square
        
        # Create a non-square image
        img = Image.new("RGB", (1600, 900), color=(255, 128, 64))
        
        result = resize_and_pad_to_square(img, target=1024)
        
        assert result.size == (1024, 1024), f"Expected (1024, 1024), got {result.size}"
        assert result.mode == "RGB"
    
    def test_resize_preserves_aspect_ratio(self):
        """Test that resize preserves aspect ratio."""
        from deepencoder.deepencoder_infer import resize_and_pad_to_square
        
        # Wide image
        wide = Image.new("RGB", (2000, 1000), color=(255, 0, 0))
        result_wide = resize_and_pad_to_square(wide, target=1024)
        
        # The image should be centered (padded top and bottom)
        # When downscaled: 1024 wide, 512 tall, centered
        assert result_wide.size == (1024, 1024)
    
    def test_resize_handles_square_input(self):
        """Test resize with already square input."""
        from deepencoder.deepencoder_infer import resize_and_pad_to_square
        
        square = Image.new("RGB", (500, 500), color=(0, 255, 0))
        result = resize_and_pad_to_square(square, target=1024)
        
        assert result.size == (1024, 1024)
    
    def test_pil_to_tensor_og_norm(self):
        """Test PIL to tensor conversion with OG normalization."""
        from deepencoder.deepencoder_infer import _pil_to_tensor_og_norm
        
        img = Image.new("RGB", (224, 224), color=(127, 127, 127))
        tensor = _pil_to_tensor_og_norm(img, dtype=torch.float32)
        
        assert tensor.shape == (1, 3, 224, 224)
        # OG normalization: (0.5 - 0.5) / 0.5 ≈ 0 for mid-gray
        assert tensor.mean().abs() < 0.1
    
    def test_pil_to_tensor_dtype(self):
        """Test PIL to tensor respects dtype."""
        from deepencoder.deepencoder_infer import _pil_to_tensor_og_norm
        
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            tensor = _pil_to_tensor_og_norm(img, dtype=dtype)
            assert tensor.dtype == dtype, f"Expected {dtype}, got {tensor.dtype}"
    
    def test_load_and_preprocess_image(self):
        """Test load_and_preprocess_image function."""
        from deepencoder.deepencoder_infer import load_and_preprocess_image
        
        # Create temp image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (800, 600), color=(200, 150, 100))
            img.save(f.name)
            temp_path = Path(f.name)
        
        try:
            tensor = load_and_preprocess_image(temp_path, dtype=torch.float32)
            
            assert tensor is not None
            assert tensor.shape == (1, 3, 1024, 1024)
        finally:
            temp_path.unlink()
    
    def test_load_and_preprocess_missing_image(self):
        """Test load_and_preprocess_image with missing file."""
        from deepencoder.deepencoder_infer import load_and_preprocess_image
        
        result = load_and_preprocess_image(Path("/nonexistent/path.jpg"))
        
        assert result is None
    
    def test_load_and_preprocess_none_path(self):
        """Test load_and_preprocess_image with None path."""
        from deepencoder.deepencoder_infer import load_and_preprocess_image
        
        result = load_and_preprocess_image(None)
        
        assert result is None


# ---------- Test: Single Image Encoding ----------

class TestSingleImageEncoding:
    """Test single image encoding through runtime."""
    
    def test_encode_image_shape(self, patched_runtime, temp_test_images):
        """Test encode_image returns correct shape."""
        runtime = patched_runtime
        image_path = temp_test_images["CAM_FRONT"]
        
        result = runtime.encode_image(str(image_path))
        
        assert "tokens" in result
        assert "grid" in result
        assert "image_size" in result
        
        tokens = result["tokens"]
        assert tokens.shape == (256, 2048), f"Expected (256, 2048), got {tokens.shape}"
    
    def test_encode_image_grid(self, patched_runtime, temp_test_images):
        """Test encode_image returns correct grid."""
        runtime = patched_runtime
        image_path = temp_test_images["CAM_FRONT"]
        
        result = runtime.encode_image(str(image_path))
        
        assert result["grid"] == (16, 16)
        assert result["image_size"] == 1024


# ---------- Test: Multiview Encoding ----------

class TestMultiviewEncoding:
    """Test multiview encoding functionality."""
    
    def test_encode_views_all_present(self, patched_runtime, temp_test_images):
        """Test encode_views with all views present."""
        runtime = patched_runtime
        view_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                      "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        
        image_paths = [temp_test_images[v] for v in view_order]
        
        result = runtime.encode_views(image_paths, strict=False, view_order=view_order)
        
        assert "tokens" in result
        assert "present_mask" in result
        assert "view_names" in result
        
        tokens = result["tokens"]
        assert len(tokens) == 6
        assert all(t.shape == (256, 2048) for t in tokens)
        assert all(result["present_mask"])
    
    def test_encode_views_missing_view(self, patched_runtime, temp_test_images):
        """Test encode_views with missing view (non-strict)."""
        runtime = patched_runtime
        view_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                      "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        
        # Include a missing path
        image_paths = [
            temp_test_images["CAM_FRONT"],
            None,  # Missing view
            temp_test_images["CAM_FRONT_RIGHT"],
            temp_test_images["CAM_BACK"],
            temp_test_images["CAM_BACK_LEFT"],
            temp_test_images["CAM_BACK_RIGHT"],
        ]
        
        result = runtime.encode_views(image_paths, strict=False, view_order=view_order)
        
        assert len(result["tokens"]) == 6
        assert result["present_mask"] == [True, False, True, True, True, True]
        
        # Missing view should be zeros
        missing_tokens = result["tokens"][1]
        assert torch.allclose(missing_tokens, torch.zeros_like(missing_tokens))
    
    def test_encode_views_strict_raises(self, patched_runtime, temp_test_images):
        """Test encode_views raises in strict mode with missing view."""
        runtime = patched_runtime
        
        image_paths = [
            temp_test_images["CAM_FRONT"],
            None,  # Missing view
        ]
        
        with pytest.raises(FileNotFoundError):
            runtime.encode_views(image_paths, strict=True, view_order=["CAM_FRONT", "CAM_FRONT_LEFT"])


# ---------- Test: Batch Encoding ----------

class TestBatchEncoding:
    """Test batch encoding functionality."""
    
    def test_encode_images_batch(self, patched_runtime):
        """Test encode_images_batch with multiple images."""
        runtime = patched_runtime
        
        # Create batch of images
        images = [torch.randn(1, 3, 1024, 1024) for _ in range(4)]
        
        result = runtime.encode_images_batch(images)
        
        assert result.shape == (4, 256, 2048)
    
    def test_encode_images_batch_single(self, patched_runtime):
        """Test encode_images_batch with single image."""
        runtime = patched_runtime
        
        images = [torch.randn(1, 3, 1024, 1024)]
        
        result = runtime.encode_images_batch(images)
        
        assert result.shape == (1, 256, 2048)
    
    def test_encode_images_batch_tensor_input(self, patched_runtime):
        """Test encode_images_batch with pre-stacked tensor."""
        runtime = patched_runtime
        
        batch = torch.randn(3, 3, 1024, 1024)
        
        result = runtime.encode_images_batch(batch)
        
        assert result.shape == (3, 256, 2048)


class TestBatchMultiviewEncoding:
    """Test batch multiview encoding functionality."""
    
    def test_encode_preloaded_views_batch(self, patched_runtime):
        """Test encode_preloaded_views_batch with preloaded tensors."""
        runtime = patched_runtime
        
        # Simulate 2 samples with 6 views each
        B = 2
        V = 6
        
        batch_images = []
        for _ in range(B):
            sample_images = [torch.randn(1, 3, 1024, 1024) for _ in range(V)]
            batch_images.append(sample_images)
        
        view_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                      "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        
        result = runtime.encode_preloaded_views_batch(batch_images, view_order=view_order)
        
        assert len(result) == B
        for sample_tokens in result:
            assert len(sample_tokens) == V
            for view_tokens in sample_tokens:
                assert view_tokens.shape == (256, 2048)
    
    def test_encode_preloaded_views_batch_with_missing(self, patched_runtime):
        """Test encode_preloaded_views_batch with some missing views."""
        runtime = patched_runtime
        
        view_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                      "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        
        batch_images = [
            [torch.randn(1, 3, 1024, 1024), None, torch.randn(1, 3, 1024, 1024),
             torch.randn(1, 3, 1024, 1024), torch.randn(1, 3, 1024, 1024), torch.randn(1, 3, 1024, 1024)],
        ]
        
        result = runtime.encode_preloaded_views_batch(batch_images, view_order=view_order)
        
        assert len(result) == 1
        assert len(result[0]) == 6
        
        # Missing view should be zeros
        missing_tokens = result[0][1]
        assert torch.allclose(missing_tokens, torch.zeros_like(missing_tokens))
    
    def test_encode_views_batch(self, patched_runtime, temp_test_images):
        """Test encode_views_batch with file paths."""
        runtime = patched_runtime
        
        view_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                      "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        
        # Two samples with all views
        batch_paths = [
            [temp_test_images[v] for v in view_order],
            [temp_test_images[v] for v in view_order],
        ]
        
        result = runtime.encode_views_batch(batch_paths, strict=False, view_order=view_order)
        
        assert len(result) == 2
        for sample_tokens in result:
            assert len(sample_tokens) == 6
            for view_tokens in sample_tokens:
                assert view_tokens.shape == (256, 2048)


# ---------- Test: NuScenes Integration ----------

class TestNuScenesIntegration:
    """Test nuScenes integration functions."""
    
    def test_resolve_cam_image_paths_mocked(self):
        """Test resolve_cam_image_paths with mocked NuScenes."""
        from deepencoder.deepencoder_infer import resolve_cam_image_paths
        
        # Create mock NuScenes object
        mock_nusc = Mock()
        mock_nusc.dataroot = "/data/nuscenes"
        
        # Mock sample data
        mock_nusc.get.side_effect = lambda table, token: {
            "sample": {
                "data": {
                    "CAM_FRONT": "front_token",
                    "CAM_BACK": "back_token",
                }
            },
            "sample_data": {
                "filename": "samples/CAM_FRONT/test.jpg"
            }
        }.get(table, {})
        
        view_order = ["CAM_FRONT", "CAM_BACK"]
        
        # This will fail on file existence check, but tests the resolution logic
        paths = resolve_cam_image_paths(mock_nusc, "sample_token", view_order=view_order)
        
        assert len(paths) == 2


# ---------- Test: Runtime Train/Eval Modes ----------

class TestRuntimeModes:
    """Test runtime train/eval mode switching."""
    
    def test_train_mode(self, patched_runtime):
        """Test switching to train mode."""
        runtime = patched_runtime
        
        runtime.train()
        
        # CLIP and projector should be in train mode
        assert runtime.clip_vit.training is True
        assert runtime.projector.training is True
        # SAM should stay in eval mode (frozen)
        assert runtime.sam.training is False
    
    def test_eval_mode(self, patched_runtime):
        """Test switching to eval mode."""
        runtime = patched_runtime
        
        runtime.train()  # First go to train
        runtime.eval()   # Then back to eval
        
        assert runtime.sam.training is False
        assert runtime.clip_vit.training is False
        assert runtime.projector.training is False


# ---------- Test: Trainable Parameters ----------

class TestTrainableParameters:
    """Test trainable_parameters method."""
    
    def test_trainable_parameters_returns_list(self, patched_runtime):
        """Test trainable_parameters returns a list."""
        runtime = patched_runtime
        
        params = runtime.trainable_parameters()
        
        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in params)
    
    def test_trainable_parameters_requires_grad(self, patched_runtime):
        """Test all trainable parameters have requires_grad=True."""
        runtime = patched_runtime
        
        params = runtime.trainable_parameters()
        
        assert all(p.requires_grad for p in params)


# ---------- Test: Dtype Conversion ----------

class TestDtypeConversion:
    """Test dtype string conversion."""
    
    def test_to_dtype_all_formats(self):
        """Test _to_dtype with all supported formats."""
        from deepencoder.deepencoder_infer import _to_dtype
        
        assert _to_dtype("bf16") == torch.bfloat16
        assert _to_dtype("bfloat16") == torch.bfloat16
        assert _to_dtype("fp16") == torch.float16
        assert _to_dtype("float16") == torch.float16
        assert _to_dtype("fp32") == torch.float32
        assert _to_dtype("float32") == torch.float32
    
    def test_to_dtype_case_insensitive(self):
        """Test _to_dtype is case insensitive."""
        from deepencoder.deepencoder_infer import _to_dtype
        
        assert _to_dtype("BF16") == torch.bfloat16
        assert _to_dtype("FP16") == torch.float16
        assert _to_dtype("FLOAT32") == torch.float32
    
    def test_to_dtype_invalid_raises(self):
        """Test _to_dtype raises for invalid dtype."""
        from deepencoder.deepencoder_infer import _to_dtype
        
        with pytest.raises(ValueError, match="Unsupported dtype string"):
            _to_dtype("invalid")


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
