#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR-Vision-LLM Modal Training Script :: src/modal-trainer/modal-train.py

WEEK-LONG TRAINING SETUP (CUDA 12.6 + L4 GPU)
---------------------------------------------
This script is architected for long-running experiments that exceed Modal's 
24-hour execution limit. It uses a "Relay Race" pattern:
1. Run for 24 hours.
2. Modal triggers a Timeout.
3. Modal triggers a Retry (up to 10 times).
4. The new container detects the LATEST checkpoint and RESUMES automatically.

USAGE:
    modal run src/modal-trainer/modal-train.py # Run in foreground
    modal run --detach src/modal-trainer/modal-train.py # Run in background 
    
    # To monitor logs of a detached run: modal app logs lidar-vision-training
    # TO stop a detached run: modal app stop lidar-vision-training
    
    # To load an interactive terminal: modal shell lidar-vision-training
    
    # To delete all runs and checkpoints: modal volume rm lidar-llm /checkpoints --recursive    

Configuration is defined in: src/configs/modal_config.py
"""

import modal
from typing import Dict
from datetime import datetime
import os
from pathlib import Path

# ============================================================================
# MODAL SETUP - Define Cloud Environment
# ============================================================================

app = modal.App("lidar-vision-training")
volume = modal.Volume.from_name("lidar-llm", create_if_missing=False)
FLASH_ATTN_CACHE = Path("/data/build_cache/flash-attn")


def _ensure_flash_attn_cached() -> None:
    """Install flash-attn from a cached wheel on the Modal volume, building once if absent."""
    import subprocess
    import sys

    cache_dir = FLASH_ATTN_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _install_from_wheel(wheel_path: Path) -> None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", str(wheel_path), "--no-build-isolation"],
            check=True,
        )

    wheels = sorted(cache_dir.glob("flash_attn*.whl"))
    if wheels:
        wheel = wheels[-1]
        print(f"[modal_train] Installing cached flash-attn wheel → {wheel.name}")
        _install_from_wheel(wheel)
        return

    print("[modal_train] flash-attn wheel cache empty; compiling once for reuse")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "flash-attn",
            "--no-build-isolation",
            "-w",
            str(cache_dir),
        ],
        check=True,
    )
    wheels = sorted(cache_dir.glob("flash_attn*.whl"))
    if not wheels:
        raise RuntimeError("flash-attn wheel build failed; no wheel found in cache")
    _install_from_wheel(wheels[-1])

# ----------------------------------------------------------------------------
# IMAGE DEFINITION: CUDA 12.6 + PyTorch + Custom Compilation
# ----------------------------------------------------------------------------
image = (
    # Use NVIDIA's official CUDA 12.6 Devel image (Contains NVCC compiler)
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04", 
        add_python="3.11"
    )
    # Set timezone non-interactively to prevent build hangs
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Asia/Kolkata"})
    .run_commands(
        "ln -snf /usr/share/zoneinfo/Asia/Kolkata /etc/localtime",
        "echo Asia/Kolkata > /etc/timezone"
    )
    # Install system build tools and dependencies for compilation
    # - build-essential: gcc, g++, make (for C/C++ compilation)
    # - clang: Required by SharedArray
    # - llvm-dev: Required by llvmlite/numba
    # - libopencv-dev: OpenCV system libraries
    # - pkg-config: Used by many build systems
    .apt_install(
        "git", "wget", "build-essential", "ninja-build",
        "clang", "llvm-dev", "libopencv-dev", "pkg-config",
        "default-jre-headless",
    )
    # Install PyTorch with CUDA 12.6 support
    # NOTE: pip_install does not support pre_install_commands parameter
    # Use run_commands() before pip_install() instead
    .run_commands(
        "pip3 install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu126"
    )
    .pip_install("spconv-cu126")
    .pip_install(
        "llvmlite", "numba",
    )
    .pip_install(
        "tensorboardX", "easydict", "pyyaml",
        "scikit-image", "tqdm", "SharedArray", "opencv-python", "pyquaternion",
    )
    # Install remaining Python dependencies
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
        "numpy>=2.0.0,<2.3.0",  # Pin numpy 2.x to satisfy opencv-python
        "optimum>=1.15.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "nltk>=3.8.1",
        "sacrebleu>=2.3.0",
        "rouge-score>=0.1.2",
    )
    .run_commands("python -c 'import nltk; nltk.download(\"punkt\"); nltk.download(\"wordnet\"); nltk.download(\"omw-1.4\")'")
    # Add the entire src directory to the image (excluding lidar-encoder which was already added)
    .add_local_dir(
        local_path="./src",
        remote_path="/root/src",
        copy=False  # Files available at runtime, not needed at build time
    )
)


# Import configuration from centralized config module
# NOTE: This import happens at runtime inside the Modal container
# The config is imported inside train_model() to ensure /root/src is in sys.path
def get_modal_training_config() -> Dict:
    """
    Import and return config from centralized configs module.
    
    This wrapper exists because the import path is only valid inside 
    the Modal container where /root/src is mounted.
    """
    import sys
    # Add both paths to ensure imports work
    if "/root/src" not in sys.path:
        sys.path.insert(0, "/root/src")
    if "/root/src/modal-trainer" not in sys.path:
        sys.path.insert(0, "/root/src/modal-trainer")
    
    # Import from modal-trainer directory (where modal_config.py lives)
    from modal_config import get_modal_training_config as _get_config
    return _get_config()


# ============================================================================
# MODAL TRAINING FUNCTION
# ============================================================================

@app.function(
    # =========================================================================
    # HARDWARE CONFIGURATION
    # =========================================================================
    # PHASE 1 (Validation): Use T4 (cheap) + many CPU cores for data validation
    # PHASE 2 (Training):   Switch to H200 + fewer cores once validation passes
    # -------------------------------------------------------------------------
    # Phase 1 Config (validation - will OOM on model loading, that's OK):
    gpu="A100-80GB",  # H100 80GB for large model + data validation
    cpu=8,       # 40 cores (Modal max) for parallel validation
    memory=131072,  # 128 GB RAM
    # ------------------------------------------------------------------------
    
    image=image,
    volumes={"/data": volume},
    
    # --- LONG RUNNING CONFIGURATION ---
    # 24 Hours: The maximum duration for a single execution
    timeout=86400, 
    
    # Retries: If the function times out (or crashes), restart it.
    # 10 retries * 24 hours = ~10 Days of total runtime coverage.
    retries=modal.Retries(
        max_retries=10,
        initial_delay=60.0,
        backoff_coefficient=1.0
    ),
    
    # Max Inputs = 1 ensures that every retry gets a pristine, fresh container.
    max_inputs=1,
)
def train_model():
    """
    Main training function designed for resilience and auto-resuming.
    """
    import sys
    import os
    import torch
    from pathlib import Path
    
    # 0. Setup Model Cache on Persistent Volume
    # This ensures downloaded models (Qwen, CLIP, SAM) are reused across runs
    model_cache_dir = "/data/model_cache"
    os.makedirs(f"{model_cache_dir}/huggingface", exist_ok=True)
    os.makedirs(f"{model_cache_dir}/torch", exist_ok=True)
    os.makedirs(f"{model_cache_dir}/clip", exist_ok=True)
    os.makedirs(f"{model_cache_dir}/sam", exist_ok=True)
    
    # Set environment variables BEFORE importing any ML libraries
    os.environ["HF_HOME"] = f"{model_cache_dir}/huggingface"
    os.environ["HF_HUB_CACHE"] = f"{model_cache_dir}/huggingface"
    os.environ["TORCH_HOME"] = f"{model_cache_dir}/torch"
    os.environ["XDG_CACHE_HOME"] = model_cache_dir  # For open_clip
    
    # BERTScore caches the RoBERTa model here (~1.4GB download)
    # Point it to our persistent volume so it's cached between runs
    os.environ["BERT_SCORE_CACHE"] = f"{model_cache_dir}/bertscore"
    os.makedirs(f"{model_cache_dir}/bertscore", exist_ok=True)
    
    # Transformers cache (used by BERTScore internally) - relies on HF_HOME
    # os.environ["TRANSFORMERS_CACHE"] = f"{model_cache_dir}/huggingface"
    
    # CUDA memory optimization: use expandable segments to reduce fragmentation
    # This helps when reserved-but-unallocated memory is large
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    _ensure_flash_attn_cached()
    
    print(f"📦 Model cache directory: {model_cache_dir}")
    print(f"   HuggingFace cache: {os.environ['HF_HOME']}")
    print(f"   Torch cache: {os.environ['TORCH_HOME']}")
    print(f"   BERTScore cache: {os.environ['BERT_SCORE_CACHE']}")
    
    # 1. Setup Environment
    src_path = "/root/src"
    if src_path not in sys.path: sys.path.insert(0, src_path)
    
    # Add encoder-decoder path for imports
    encoder_decoder_path = "/root/src/encoder-decoder"
    if encoder_decoder_path not in sys.path: 
        sys.path.insert(0, encoder_decoder_path)
    
    try:
        from training.core import Trainer
    except ImportError as e:
        print(f"❌ Failed to import Trainer: {e}")
        print(f"sys.path: {sys.path}")
        raise

    # 2. Load Configuration
    config = get_modal_training_config()
    
    # 3. Directory Logic (The "Smart Resume" System)
    # Use the configured out_dir as the root for checkpoints
    root_ckpt_dir = Path(config["out_dir"])
    
    # Explicit check to create the directory if it's the very first run on this volume
    if not root_ckpt_dir.exists():
        print(f"✨ First Run: Creating checkpoints directory at {root_ckpt_dir}")
        
    root_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    timeout_threshold = 3600 # 1 Hour
    
    # --- SAFETY ASSERTION ---
    # If we are running a long job (>1h), we MUST be in resume mode.
    # If resume=False, a retry would wipe out progress by creating a new folder.
    if config["timeout"] > timeout_threshold and not config["resume"]:
        error_msg = (
            f"\n🛑 CONFIGURATION ERROR: Week-Long Training Safety 🛑\n"
            f"----------------------------------------------------------\n"
            f"You requested a long-running job (timeout={config['timeout']}s) with resume=False.\n\n"
            f"Why this is blocked:\n"
            f"When Modal triggers a retry (after 24h), the script runs from the top.\n"
            f"If resume=False, it would create a NEW 'run_YYYY...' folder on Day 2,\n"
            f"ignoring Day 1's progress. This wastes computation.\n\n"
            f"👉 FIX: Set config['resume'] = True\n"
            f"   (The script will automatically detect if it needs to start fresh\n"
            f"    or resume the latest run.)\n"
            f"----------------------------------------------------------\n"
        )
        raise ValueError(error_msg)

    # --- AUTO-DISCOVERY OF LATEST RUN ---
    latest_run_dir = None
    
    # Find all directories starting with "run_"
    run_dirs = [
        d for d in root_ckpt_dir.iterdir() 
        if d.is_dir() and d.name.startswith("run_")
    ]
    
    if run_dirs:
        # Sort by name (timestamp format ensures chronological order)
        # run_20251126_143052 comes after run_20251126_100000
        run_dirs.sort(key=lambda x: x.name)
        latest_run_dir = run_dirs[-1]
        print(f"🔎 Found {len(run_dirs)} existing runs.")
        print(f"👉 Latest run identified: {latest_run_dir.name}")
    else:
        print("🔎 No existing runs found in checkpoints folder.")

    # --- RESUME DECISION LOGIC ---
    if config["resume"] and latest_run_dir:
        # Case A: Resuming an existing run
        ckpt_file = latest_run_dir / "training_state_latest.pt"
        
        if ckpt_file.exists():
            print(f"✅ Auto-Resuming from: {latest_run_dir}")
            print(f"   Checkpoint found: {ckpt_file.name}")
            config["out_dir"] = str(latest_run_dir)
            config["resume_path"] = str(ckpt_file)
        else:
            print(f"⚠️  Latest run ({latest_run_dir.name}) has no 'training_state_latest.pt'.")
            print("   Assuming broken/empty run. Starting FRESH run...")
            # Fallback to creating a new one
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_run_dir = root_ckpt_dir / f"run_{timestamp}"
            new_run_dir.mkdir(parents=True, exist_ok=True)
            config["out_dir"] = str(new_run_dir)
            config["resume"] = False # Logic switch for Trainer
            
    else:
        # Case B: First run ever OR resume=False explicitly set (and timeout < 1h)
        print("🆕 Starting FRESH training run...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_run_dir = root_ckpt_dir / f"run_{timestamp}"
        new_run_dir.mkdir(parents=True, exist_ok=True)
        config["out_dir"] = str(new_run_dir)
        # Even if config['resume'] was True, we set it to False here 
        # so the Trainer initializes weights instead of looking for a file.
        config["resume"] = False 

    print(f"📁 Final Output Directory: {config['out_dir']}")

    # 5. Run Training
    try:
        print("\n🏋️ Initializing Trainer...")
        trainer = Trainer(config)
        
        print("🚀 Starting training loop...")
        trainer.train()
        
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Always commit volume changes so checkpoints persist
        volume.commit()
        print("✓ Volume committed")


# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main():
    import os
    from pathlib import Path
    
    print("=" * 80)
    print("MODAL DEPLOYMENT (Week-Long Training Setup)")
    print("=" * 80)
    
    if not Path("./src").exists():
        print("❌ ERROR: Run from project root containing ./src")
        raise SystemExit(1)
        
    print("✓ Project root verified")
    print("✓ Deploying to Modal...")
    
    try:
        train_model.remote()
    except Exception as e:
        print(f"\n❌ Deployment Failed: {e}")
        if "volume" in str(e).lower():
            print("💡 Tip: Ensure volume 'lidar-llm' exists: modal volume create lidar-llm")
        raise

if __name__ == "__main__":
    print("Run with: modal run src/modal-trainer/modal-train.py")