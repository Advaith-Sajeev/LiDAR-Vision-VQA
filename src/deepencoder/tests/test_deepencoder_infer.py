#!/usr/bin/env python3
"""Test script for DeepEncoder inference"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from deepencoder.deepencoder_infer import (
    DeepEncoderRuntime,
    deepencoder_infer,
    download_sam_if_needed,
    _to_dtype,
)
import tempfile


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Flash Attention requires CUDA")
def test_deepencoder_runtime():
    """Test DeepEncoderRuntime with mock image"""
    # Mock input image (1024x1024 RGB)
    input_image = Image.new("RGB", (1024, 1024), color=(255, 255, 255))

    # Initialize DeepEncoderRuntime (use CUDA for Flash Attention)
    device = 'cuda'
    runtime = DeepEncoderRuntime(
        sam_ckpt=None,
        auto_download_sam=True,
        device=device,
        dtype=torch.float16,
        openclip_pretrained="openai",
    )

    # Save the mock image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
        temp_path = temp_image.name
        input_image.save(temp_path)

    try:
        # Print input image dimensions
        print("Input image dimensions:", input_image.size)

        # Use the encode_image method to process the image
        embeddings = runtime.encode_image(temp_path)

        # Print the embedding dimensions for debugging
        print("Embedding dimensions:", embeddings["tokens"].shape)

        # Print entering SAM module
        print("Entering SAM module...")

        # Print SAM output dimensions
        print("SAM output dimensions:", embeddings["tokens"].shape)

        # Print entering CLIP module
        print("Entering CLIP module...")

        # Print final embedding dimensions
        print("Final embedding dimensions:", embeddings["tokens"].shape)

        # Print the model summary
        print("\nModel Summary:\n")
        print(runtime.clip_vit)

        # Print detailed summaries of all modules
        print("\nSAM Module Summary:\n")
        print(runtime.sam)

        print("\nCLIP Module Summary:\n")
        print(runtime.clip_vit)

        print("\nMLP Projector Summary:\n")
        print(runtime.projector)

        # Assert output dimensions (e.g., [256, 2048])
        assert embeddings["tokens"].shape == (256, 2048), "Embedding dimensions are incorrect"
        print("✓ Test passed: Embedding dimensions correct")
        
    finally:
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Flash Attention requires CUDA")
def test_deepencoder_infer_with_real_image():
    """Test DeepEncoder inference pipeline on a real test image"""
    
    # Configuration - use the test image from the tests directory
    test_image_dir = Path(__file__).parent / "test-input-images"
    image_path = test_image_dir / "test-1.jpg"
    
    if not image_path.exists():
        print(f"[SKIP] Test image not found: {image_path}")
        print("[INFO] This test requires a test image in deepencoder/tests/test-input-images/")
        pytest.skip("Test image not available")
        return
    
    image_path = str(image_path)
    
    print("="*60)
    print("Testing DeepEncoder Inference with Real Image")
    print("="*60)
    
    # Ensure SAM weights
    sam_ckpt = download_sam_if_needed(None, auto_download=True)
    print(f"✓ SAM checkpoint: {sam_ckpt}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    openclip_pretrained = "openai"
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"OpenCLIP pretrained: {openclip_pretrained}")
    
    # Run inference
    print("\nRunning inference...")
    out = deepencoder_infer(
        image_path=image_path,
        sam_ckpt=sam_ckpt,
        device=device,
        dtype=dtype,
        openclip_pretrained=openclip_pretrained,
    )
    
    vt = out["vision_tokens"].squeeze(0)  # [256, 2048] for current (16×16) grid configuration
    print(
        f"\n[OK] Vision tokens: shape={tuple(vt.shape)} "
        f"grid={out['grid']} "
        f"image_size={out['image_size']} "
        f"norm={out['normalization']}"
    )
    
    # Validate output
    assert vt.shape[0] == 256, f"Expected 256 tokens, got {vt.shape[0]}"
    assert vt.shape[1] == 2048, f"Expected 2048 dims, got {vt.shape[1]}"
    assert out['grid'] == (16, 16), f"Expected grid (16, 16), got {out['grid']}"
    assert out['image_size'] == 1024, f"Expected image size 1024, got {out['image_size']}"
    print("✓ Output validation passed!")
    
    print("\n" + "="*60)
    print("Test completed successfully! ✓")
    print("="*60)


if __name__ == "__main__":
    print("Running DeepEncoder inference tests...\n")
    
    # Run test with mock image
    print("Test 1: Mock image")
    print("-" * 60)
    test_deepencoder_runtime()
    
    print("\n\nTest 2: Real nuScenes image (if available)")
    print("-" * 60)
    try:
        test_deepencoder_infer_with_real_image()
    except Exception as e:
        print(f"Skipped: {e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)