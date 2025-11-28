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

"""

import modal
from typing import Dict
from datetime import datetime
import os

# ============================================================================
# MODAL SETUP - Define Cloud Environment
# ============================================================================

app = modal.App("lidar-vision-training")
volume = modal.Volume.from_name("lidar-llm", create_if_missing=False)

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
        "clang", "llvm-dev", "libopencv-dev", "pkg-config"
    )
    # Install PyTorch with CUDA 12.6 support
    # NOTE: pip_install does not support pre_install_commands parameter
    # Use run_commands() before pip_install() instead
    .run_commands(
        "pip3 install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu126"
    )
    # Install spconv for CUDA 12.6 (Critical for L4 performance)
    .pip_install("spconv-cu126")
    
    # Dependencies for lidar-encoder (order matters for compilation)
    # Install packages that may require compilation first
    .pip_install(
        "llvmlite", "numba",  # Installed first as they have LLVM dependencies
    )
    .pip_install(
        "tensorboardX", "easydict", "pyyaml",
        "scikit-image", "tqdm", "SharedArray", "opencv-python", "pyquaternion",
    )
    # Compile lidar-encoder (pcdet)
    # copy=True ensures files are built into the image for subsequent build steps
    .add_local_dir(
        local_path="./src/lidar-encoder",
        remote_path="/tmp/lidar-encoder",
        copy=True
    )
    .run_commands(
        "cd /tmp/lidar-encoder && pip install -e . --no-build-isolation",
        gpu="any", 
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
        "sacrebleu>=2.3.0",
        "rouge-score>=0.1.2",
    )
    # Install flash-attn last
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .run_commands("python -c 'import nltk; nltk.download(\"punkt\"); nltk.download(\"wordnet\")'")
    # Add the entire src directory to the image (excluding lidar-encoder which was already added)
    .add_local_dir(
        local_path="./src",
        remote_path="/root/src",
        copy=False  # Files available at runtime, not needed at build time
    )
)


def get_modal_training_config() -> Dict:
    """
    Configuration for Week-Long Training on L4.
    """
    config = {
        # ╔════════════════════════════════════════════════════════════════╗
        # ║                    DEBUG LOGGING TOGGLE                        ║
        # ╚════════════════════════════════════════════════════════════════╝
        # 
        # Quick Enable/Disable Debug Mode:
        #   - Set "debug_mode" to True to enable extensive debug logging
        #   - Set "debug_mode" to False for normal training (no overhead)
        # 
        # Debug Levels (when debug_mode=True):
        #   1 = INFO  : High-level flow only (minimal output)
        #   2 = DEBUG : Detailed flow with data tracking (recommended)
        #   3 = TRACE : Very detailed with shapes, stats, timing (verbose)
        # 
        # Module Filtering:
        #   - [] (empty)              : Show all modules
        #   - ["trainer"]             : Show only trainer logs
        #   - ["trainer", "dataset"]  : Show multiple modules
        # 
        # Output is automatically saved to: <out_dir>/debug.log
        # Terminal output is color-coded for easy reading
        # 
        # Performance Impact:
        #   - debug_mode=False : 0% overhead (completely disabled)
        #   - debug_level=1    : <1% overhead
        #   - debug_level=2    : 1-3% overhead
        #   - debug_level=3    : 5-10% overhead
        # 
        # See DEBUG_GUIDE.md for complete documentation
        # ────────────────────────────────────────────────────────────────
        
        "debug_mode": False,      # ← SET TO True TO ENABLE DEBUG LOGGING
        "debug_level": 0,         # ← 0=DISABLED, 1=INFO, 2=DEBUG, 3=TRACE
        "debug_modules": [],      # ← [] = all modules, or ["trainer", "dataset"]
        
        
        # ==================== I/O Configuration ====================
        # Directories containing BEV feature .npy files (one per sample_token)
        "feature_dirs": ["/data/bev_feats"],
        
        # JSON/JSONL files with QA pairs for training and validation (nuCaption and nuGrounding)
        "jsons": [
            "/data/Datasets/nuScenes/external/nuCaption.json",
            "/data/Datasets/nuScenes/external/nuGrounding.json"
        ],
        
        # Output directory for checkpoints, logs, and plots
        # NOTE: This is overwritten at runtime by the Smart Resume logic
        # Actual checkpoints go to: /data/checkpoints/run_YYYYMMDD_HHMMSS/
        "out_dir": "/data/checkpoints",  # Placeholder - overwritten by train_model()
        
        # Maximum number of samples to use (None = use all data)
        # Set to small number (e.g., 10) for quick testing
        "max_samples": 25_000,  # None for full dataset
        
        
        # ==================== Training Configuration ====================
        # Number of training epochs
        "epochs": 50,
        
        # Batch size per GPU
        # NOTE: Reduced from 4 to 2 due to OOM on A100-40GB with full multimodal setup
        # Using grad_accum=2 to maintain effective batch size of 4
        "batch_size": 16,
        
        # Gradient accumulation steps (effective_batch = batch_size * grad_accum * num_gpus)
        # Increased to compensate for smaller batch_size (2 * 2 = 4 effective)
        "grad_accum": 1,
        
        # Number of DataLoader workers for parallel data loading
        # Higher = better GPU utilization (overlaps data loading with training)
        # Recommended: num_workers ≈ CPU_cores / 2 (with 24 cores, use 12-16)
        "num_workers": 16,
        
        # Prefetch factor: batches to prefetch per worker (requires num_workers > 0)
        # Higher = more memory usage but better GPU utilization
        "prefetch_factor": 4,
        
        # Random seed for reproducibility
        "seed": 42,
        
        # Mixed precision training mode
        # Options: "no" (disabled), "fp16" (float16), "bf16" (bfloat16)
        # bf16 is recommended for L4 GPUs (better numerical stability than fp16)
        "mixed_precision": "bf16",  # "no", "fp16", or "bf16"
        
        # Enable gradient checkpointing to save memory (trades compute for memory)
        # This recomputes activations during backward pass instead of storing them
        # Enables larger batch sizes but increases training time by ~20-30%
        "gradient_checkpointing": True,
        
        # Resume from checkpoint if available
        "resume": True,
        
        # Save checkpoint every N steps (0 = disable step-based saving)
        "save_every_steps": 0,
        
        # Keep only last N checkpoints (older ones are deleted)
        "keep_last_n": 3,
        
        # Plot loss curves every N epochs (currently not used in loop, always plots)
        "plot_every": 1,
        
        # Print tensor shapes during forward pass (for debugging)
        "debug_shapes": False, 
        
        
        # ==================== Validation Configuration ====================
        # Percentage of data to use for validation (0.1 = 10%, 0.05 = 5%)
        "val_split": 0.1,
        
        # Run validation every N epochs
        "validate_every": 1,
        
        # System prompt for the model (used in chat template)
        "system_prompt": "You are an autonomous driving assistant. Analyze the 3D LiDAR tokens and camera tokens to understand the driving scene. Provide accurate, concise reply for the question asked.",
        
        
        # ==================== Inference Sampling Configuration ====================
        # Generate predictions on validation samples every N epochs (0 = disable)
        "inference_sampling_every": 5,
        
        # Total number of samples to generate (must be divisible by 4 for equal distribution)
        # 50% caption, 25% det_area, 25% det_object
        "inference_samples_n": 20,
        
        # Test JSON files for inference sampling
        "inference_caption_json": "/data/Datasets/Test/LiDAR-LLM-Nu-Caption-val.json",
        "inference_grounding_json": "/data/Datasets/Test/LiDAR-LLM-Nu-Grounding-val.json",
        
        # Generation parameters for inference sampling
        "inference_max_tokens": 256,
        "inference_temperature": 0.1,
        "inference_top_p": 0.9,
        "inference_top_k": 50,
        "inference_do_sample": True,
        "inference_num_beams": 1,
        
        # ==================== Evaluation Metrics Toggles ====================
        # Enable/disable specific metrics for each dashboard
        # Caption Dashboard Metrics (text quality only)
        "eval_caption_bleu4": True,
        "eval_caption_cider": True,
        "eval_caption_spice": False,
        "eval_caption_bertscore": False,
        
        # Grounding Det Area Dashboard Metrics (text quality + bbox accuracy)
        "eval_det_area_bleu4": True,
        "eval_det_area_cider": True,
        "eval_det_area_spice": False,
        "eval_det_area_bertscore": False,
        "eval_det_area_top1_acc": True,      # Object class identification accuracy
        "eval_det_area_bev_iou": True,       # 2D Bird's Eye View IoU
        
        # Grounding Det Object Dashboard Metrics (text quality only)
        "eval_det_object_bleu4": True,
        "eval_det_object_cider": True,
        "eval_det_object_spice": False,
        "eval_det_object_bertscore": False,
        
        # Toggle components during training (for debugging/ablation studies)
        # WARNING: Disabling components during training will train a model that doesn't use them!
        "training_use_vision": True,    # Include vision tokens in training
        "training_use_lidar": True,     # Include LiDAR tokens in training
        
        # Toggle components during validation (for debugging/ablation studies)
        "validation_use_vision": True,  # Include vision tokens in validation
        "validation_use_lidar": True,   # Include LiDAR tokens in validation
        
        # Toggle components during inference sampling (for debugging/ablation studies)
        "inference_use_vision": True,   # Include vision tokens in inference
        "inference_use_lidar": True,    # Include LiDAR tokens in inference
        "inference_use_system": True,   # Include system prompt in inference
        
        
        # ==================== Model Configuration ====================
        # Hugging Face model ID for base LLM
        # Options: "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"
        "model_id": "Qwen/Qwen2.5-3B",
        
        # Field name in JSON containing target answer
        "target_field": "answer", # or answer_lidar
        
        # Maximum answer tokens (longer answers will be truncated)
        "max_ans_toks": 256,
        
        # Scale factor applied to VAT prompts before feeding to LLM
        # Smaller values (0.1-0.2) help stabilize training
        "prefix_scale": 0.2,
        
        
        # ==================== LiDAR VAT Configuration ====================
        # Number of learnable query tokens for LiDAR VAT
        # MUST be divisible by 6 (for 6 spatial sectors)
        # Recommended: 12 (testing), 576 (medium), 768 (large)
        "vat_queries": 576,
        
        # Number of transformer layers in LiDAR VAT
        "vat_layers": 4,
        
        # Number of attention heads in LiDAR VAT
        "vat_heads": 8,
        
        # MLP expansion ratio (d_mlp = d_model * vat_mlp_ratio)
        "vat_mlp_ratio": 4.0,
        
        # Dropout rate in transformer blocks
        "vat_dropout": 0.10,
        
        # Dropout rate after final projection
        "vat_post_dropout": 0.10,
        
        
        # ==================== Vision VAT Configuration ====================
        # Enable vision pipeline (multi-view cameras)
        "use_vision": True,
        
        # Number of learnable query tokens for Vision VAT
        # Only needs to be divisible by 6 if vision_per_view_query=True
        # Recommended: 12 (testing), 1536 (medium), 2304 (large)
        "vision_queries": 768,
        
        # Number of transformer layers in Vision VAT
        "vision_layers": 4,
        
        # Number of attention heads in Vision VAT
        "vision_heads": 8,
        
        # MLP expansion ratio for Vision VAT
        "vision_mlp_ratio": 4.0,
        
        # Dropout rate in Vision VAT transformer blocks
        "vision_dropout": 0.10,
        
        # Dropout rate after Vision VAT final projection
        "vision_post_dropout": 0.10,
        
        # Use separate query embeddings for each camera view
        "vision_per_view_query": False, # keep False 
        
        # If True, error when per-view not feasible; if False, auto-disable with warning
        "vision_strict_per_view": False, # keep False
        
        
        # ==================== QLoRA Configuration ====================
        # Enable 4-bit quantization (QLoRA) for memory-efficient training
        # When enabled, the base LLM is loaded in 4-bit precision
        "use_qlora": True,
        
        # Quantization type: "nf4" (normalized float4) or "fp4" (float4)
        # nf4 is recommended for better accuracy
        "qlora_quant_type": "nf4",
        
        # Enable double quantization for additional memory savings
        # Quantizes the quantization constants themselves
        "qlora_double_quant": True,
        
        # Compute dtype for QLoRA operations
        # Options: "bfloat16", "float16", "float32"
        "qlora_compute_dtype": "bfloat16",
        
        # LoRA rank (higher = more parameters, more expressive)
        # Typical values: 8-16 for QLoRA (can use higher rank due to memory savings)
        "lora_r": 16,
        
        # LoRA alpha (scaling factor, typically 2*r)
        "lora_alpha": 32,
        
        # LoRA dropout rate
        "lora_dropout": 0.05,
        
        # LLM LoRA target modules (which layers to apply LoRA to)
        # Common targets for Qwen/LLaMA-style models:
        #   - Attention: "q_proj", "k_proj", "v_proj", "o_proj"
        #   - MLP: "gate_proj", "up_proj", "down_proj"
        # Set to None to use defaults, or customize the list eg: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        "lora_target_modules":  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        
        # ==================== CLIP LoRA Configuration ====================
        # Enable LoRA fine-tuning for CLIP (if False, CLIP is fully frozen)
        "clip_lora_enabled": True,
        
        # CLIP LoRA target modules (which layers to apply LoRA to)
        # Common targets for CLIP ViT models:
        #   - Attention: "qkv_proj", "out_proj" (combined Q/K/V projection)
        #   - MLP: "mlp.fc1", "mlp.fc2"
        # Set to None to use auto-detected defaults from infer_clip_lora_targets()
        # Note: CLIP uses the same lora_r, lora_alpha, lora_dropout as LLM above
        "clip_lora_target_modules": ["qkv_proj", "out_proj", "mlp.fc1", "mlp.fc2"],  # None = auto-detect, or provide list for all attention+MLP layers
        
        
        # ==================== Optimization Configuration ====================
        # Learning rate for LiDAR VAT
        "lr_vat": 5e-4,
        
        # Learning rate for Vision VAT
        "lr_vision_vat": 5e-4,
        
        # Learning rate for LLM LoRA adapters
        "lr_lora": 3e-4,
        
        # Learning rate for vision components (VisionAdapter, DeepEncoder projector, CLIP LoRA)
        "lr_vision": 5e-4,
        
        # Weight decay for regularization
        "weight_decay": 0.01,
        
        # Number of warmup steps for learning rate scheduler
        "warmup_steps": 1000,
        
        # Gradient clipping norm (prevents exploding gradients)
        "clip_norm": 1.0,
        
        
        # ==================== nuScenes / DeepEncoder Configuration ====================
        # Path to nuScenes dataset root directory
        # Should contain folders: samples, sweeps, maps, etc.
        "nu_dataroot": "/data/Datasets/nuScenes",
        
        # nuScenes version
        # Options: "v1.0-trainval", "v1.0-mini", "v1.0-test"
        "nu_version": "v1.0-trainval",
        
        # Path to SAM checkpoint (None = auto-download if auto_download_sam=True)
        # Will be saved to /data/model_cache for persistence across runs
        # NOTE: Code uses SAM ViT-B, not ViT-H!
        "sam_ckpt": "/data/model_cache/sam/sam_vit_b_01ec64.pth",
        
        # Automatically download SAM weights if missing
        "auto_download_sam": True,
        
        # Data type for DeepEncoder processing
        # Options: "float32", "bfloat16" (bfloat16 faster but requires modern GPU)
        "deep_dtype": "bfloat16",
        
        # OpenCLIP pretrained weights
        # Options: "openai", "laion400m_e32", "laion2b_s32b_b79k"
        "openclip_pretrained": "openai",
        
        # ==================== Performance Optimizations ====================
        # Enable torch.compile() for VAT models (PyTorch 2.0+)
        # Provides 10-30% speedup on A100/H100 GPUs via TorchDynamo+TorchInductor
        # First epoch is slower due to compilation; disable for debugging
        # NOTE: Disabled due to CUDA graph conflicts with tensor caching in VATLiDAR._grid()
        # The caching mechanism creates tensors that conflict with CUDA graph memory management
        "use_torch_compile": False,
        
        # torch.compile mode
        # Options: "default", "reduce-overhead" (recommended for training), "max-autotune"
        "torch_compile_mode": "reduce-overhead",
        
        # --- ADDED FOR MODAL ---
        "checkpoints_root": "/data/checkpoints",  # Root directory for all training runs
        "timeout": 86400,  # 24 hours timeout
    }

    return config


# ============================================================================
# MODAL TRAINING FUNCTION
# ============================================================================

@app.function(
    # Hardware: NVIDIA H200 (141GB VRAM) + Beefy CPU
    gpu="H200",      
    cpu=24.0,      # 24 Physical Cores (increased for more DataLoader workers)
    memory=65536,  # 64 GB RAM
    
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
    
    print("=" * 80)
    print("🚀 MODAL TRAINING STARTED (CUDA 12.6 + A100)")
    print("=" * 80)
    
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
    
    # CUDA memory optimization: use expandable segments to reduce fragmentation
    # This helps when reserved-but-unallocated memory is large
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print(f"📦 Model cache directory: {model_cache_dir}")
    print(f"   HuggingFace cache: {os.environ['HF_HOME']}")
    print(f"   Torch cache: {os.environ['TORCH_HOME']}")
    
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
    root_ckpt_dir = Path(config["checkpoints_root"])
    
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
    print("✓ Configuration: A100 GPU, 10 Retries, 24h Timeout")
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