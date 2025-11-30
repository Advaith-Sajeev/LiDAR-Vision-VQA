#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR-Vision-LLM Modal Test Runner :: src/modal-trainer/modal-test.py

Runs tests in the deepencoder and training-test packages on Modal.

USAGE:
    # Run deepencoder tests only:
    modal run src/modal-trainer/modal-test.py::run_deepencoder_tests_only
    
    # Run training tests only:
    modal run src/modal-trainer/modal-test.py::run_training_tests_only
    
    # Interactive shell for debugging:
    modal shell src/modal-trainer/modal-test.py::run_deepencoder_tests_only
    
NOTE: Tests must be run separately due to import isolation requirements.
"""

import modal
from pathlib import Path

# ============================================================================
# MODAL SETUP - Reuse the same app name as training to avoid image rebuild
# ============================================================================

app = modal.App("lidar-vision-training")
volume = modal.Volume.from_name("lidar-llm", create_if_missing=False)

# ----------------------------------------------------------------------------
# IMAGE DEFINITION: Same as modal-train.py for consistency
# ----------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04", 
        add_python="3.11"
    )
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Asia/Kolkata"})
    .run_commands(
        "ln -snf /usr/share/zoneinfo/Asia/Kolkata /etc/localtime",
        "echo Asia/Kolkata > /etc/timezone"
    )
    .apt_install(
        "git", "wget", "build-essential", "ninja-build", 
        "clang", "llvm-dev", "libopencv-dev", "pkg-config"
    )
    .run_commands(
        "pip3 install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu126"
    )
    .pip_install("spconv-cu126")
    .pip_install("llvmlite", "numba")
    .pip_install(
        "tensorboardX", "easydict", "pyyaml",
        "scikit-image", "tqdm", "SharedArray", "opencv-python", "pyquaternion",
    )
    # Compile lidar-encoder (pcdet)
    .add_local_dir(
        local_path="./src/lidar-encoder",
        remote_path="/tmp/lidar-encoder",
        copy=True
    )
    .run_commands(
        "cd /tmp/lidar-encoder && pip install -e . --no-build-isolation",
        gpu="any", 
    )
    .pip_install(
        "transformers>=4.35.0",
        "peft>=0.6.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.25.0",
        "open-clip-torch>=2.20.0",
        "pillow>=10.0.0",
        "pycocotools>=2.0.6",
        "pycocoevalcap>=1.2",
        "bert-score>=0.3.13",
        "nuscenes-devkit>=1.1.0",
        "matplotlib>=3.7.0",
        "numpy>=2.0.0,<2.3.0",
        "optimum>=1.15.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-xdist>=3.5.0",  # For parallel test execution
        "sacrebleu>=2.3.0",
        "rouge-score>=0.1.2",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .run_commands("python -c 'import nltk; nltk.download(\"punkt\"); nltk.download(\"wordnet\")'")
    # Mount source code
    .add_local_dir(
        local_path="./src",
        remote_path="/root/src",
        copy=False
    )
)


# ============================================================================
# DEEPENCODER TESTS
# ============================================================================

@app.function(
    gpu="H200",
    cpu=24.0,
    memory=65536,       # 64 GB RAM
    image=image,
    volumes={"/data": volume},
    timeout=1800,
)
def run_deepencoder_tests_only(verbose: bool = True):
    """Run only deepencoder tests (faster for quick validation)."""
    import subprocess
    import sys
    import os
    
    print("=" * 80)
    print("🧪 DEEPENCODER TESTS")
    print("=" * 80)
    
    src_path = "/root/src"
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    os.environ["PYTHONPATH"] = src_path
    
    pytest_args = [
        "pytest",
        "/root/src/deepencoder/tests",
        "-v" if verbose else "-q",
        "--tb=short",
        "--color=yes",
        "-ra",
    ]
    
    print(f"🚀 Running: {' '.join(pytest_args)}")
    print("=" * 80)
    
    result = subprocess.run(pytest_args, cwd="/root/src", env=os.environ.copy())
    
    print("=" * 80)
    if result.returncode == 0:
        print("✅ DEEPENCODER TESTS PASSED")
    else:
        print(f"❌ DEEPENCODER TESTS FAILED (exit code: {result.returncode})")
    
    return result.returncode


# ============================================================================
# TRAINING TESTS
# ============================================================================

@app.function(
    gpu="H200",
    cpu=24.0,
    memory=65536,       # 64 GB RAM
    image=image,
    volumes={"/data": volume},
    timeout=2400,
)
def run_training_tests_only(verbose: bool = True):
    """Run only training-test package tests."""
    import subprocess
    import sys
    import os
    
    print("=" * 80)
    print("🧪 TRAINING TESTS")
    print("=" * 80)
    
    src_path = "/root/src"
    encoder_decoder_path = "/root/src/encoder-decoder"
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if encoder_decoder_path not in sys.path:
        sys.path.insert(0, encoder_decoder_path)
    
    os.environ["PYTHONPATH"] = f"{src_path}:{encoder_decoder_path}"
    
    pytest_args = [
        "pytest",
        "/root/src/encoder-decoder/training-test",
        "-v" if verbose else "-q",
        "--tb=short",
        "--color=yes",
        "-ra",
    ]
    
    print(f"🚀 Running: {' '.join(pytest_args)}")
    print("=" * 80)
    
    result = subprocess.run(pytest_args, cwd="/root/src", env=os.environ.copy())
    
    print("=" * 80)
    if result.returncode == 0:
        print("✅ TRAINING TESTS PASSED")
    else:
        print(f"❌ TRAINING TESTS FAILED (exit code: {result.returncode})")
    
    return result.returncode


# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main():
    """
    Default entrypoint - shows usage instructions.
    Tests must be run separately.
    """
    print("=" * 80)
    print("MODAL TEST RUNNER")
    print("=" * 80)
    print()
    print("Tests must be run separately due to import isolation requirements.")
    print()
    print("Usage:")
    print("  modal run src/modal-trainer/modal-test.py::run_deepencoder_tests_only")
    print("  modal run src/modal-trainer/modal-test.py::run_training_tests_only")
    print()
    print("=" * 80)


if __name__ == "__main__":
    print("Run with:")
    print("  modal run src/modal-trainer/modal-test.py::run_deepencoder_tests_only  # deepencoder tests")
    print("  modal run src/modal-trainer/modal-test.py::run_training_tests_only     # training tests")
