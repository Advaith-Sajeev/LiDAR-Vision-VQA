#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR-Vision-LLM Training Script
Comprehensive entry point with all configuration options
"""

import sys
import os

# Add the 'src' directory to the Python path BEFORE any local imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from typing import Dict
from training.core import Trainer


def get_training_config() -> Dict:
    """
    Get comprehensive training configuration with all available options.
    
    All options are explicitly shown here for easy customization.
    Modify values directly or uncomment sections as needed.
    """
    
    config = {
        # ==================== I/O Configuration ====================
        # Directories containing BEV feature .npy files (one per sample_token)
        "feature_dirs": ["./bev_feats/train"],
        
        # JSON/JSONL files with QA pairs (nuCaption, nuGrounding, etc.)
        "jsons": [
            "Dataset_subset/external/nuCaption.json",
            "Dataset_subset/external/nuGrounding.json"
        ],
        
        # Output directory for checkpoints, logs, and plots
        "out_dir": "./checkpoints_vat",
        
        # Maximum number of samples to use (None = use all data)
        # Set to small number (e.g., 10) for quick testing
        "max_samples": 10,  # None for full dataset
        
        
        # ==================== Training Configuration ====================
        # Number of training epochs
        "epochs": 10,
        
        # Batch size per GPU
        "batch_size": 1,
        
        # Gradient accumulation steps (effective_batch = batch_size * grad_accum * num_gpus)
        "grad_accum": 1,
        
        # Random seed for reproducibility
        "seed": 42,
        
        # Enable mixed precision (FP16) training - speeds up training on modern GPUs
        "fp16": False,
        
        # Resume from checkpoint if available
        "resume": True,
        
        # Save checkpoint every N steps (0 = disable step-based saving)
        "save_every_steps": 1000,
        
        # Keep only last N checkpoints (older ones are deleted)
        "keep_last_n": 5,
        
        # Plot loss curves every N epochs (currently not used in loop, always plots)
        "plot_every": 1,
        
        # Print tensor shapes during forward pass (for debugging)
        "debug_shapes": False,
        
        
        # ==================== Validation Configuration ====================
        # Percentage of data to use for validation (0.05 = 5%)
        "val_split": 0.05,
        
        # Run validation every N epochs
        "validate_every": 1,
        
        # Number of validation samples to save for inference inspection
        "val_inference_n": 10,
        
        
        # ==================== Model Configuration ====================
        # Hugging Face model ID for base LLM
        # Options: "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"
        "model_id": "Qwen/Qwen2.5-0.5B",
        
        # Field name in JSON containing target answer
        "target_field": "answer",
        
        # Maximum answer tokens (longer answers will be truncated)
        "max_ans_toks": 32,
        
        # Scale factor applied to VAT prompts before feeding to LLM
        # Smaller values (0.1-0.2) help stabilize training
        "prefix_scale": 0.2,
        
        
        # ==================== LiDAR VAT Configuration ====================
        # Number of learnable query tokens for LiDAR VAT
        # MUST be divisible by 6 (for 6 spatial sectors)
        # Recommended: 12 (testing), 576 (medium), 768 (large)
        "vat_queries": 12,
        
        # Number of transformer layers in LiDAR VAT
        "vat_layers": 1,
        
        # Number of attention heads in LiDAR VAT
        "vat_heads": 2,
        
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
        # MUST be divisible by 6 (for 6 camera views)
        # Recommended: 12 (testing), 1536 (medium), 2304 (large)
        "vision_queries": 12,
        
        # Number of transformer layers in Vision VAT
        "vision_layers": 1,
        
        # Number of attention heads in Vision VAT
        "vision_heads": 2,
        
        # MLP expansion ratio for Vision VAT
        "vision_mlp_ratio": 4.0,
        
        # Dropout rate in Vision VAT transformer blocks
        "vision_dropout": 0.10,
        
        # Dropout rate after Vision VAT final projection
        "vision_post_dropout": 0.10,
        
        # Use separate query embeddings for each camera view
        "vision_per_view_query": True,
        
        
        # ==================== LoRA Configuration ====================
        # LoRA rank (higher = more parameters, more expressive)
        # Typical values: 2-8 for small models, 8-16 for large models
        "lora_r": 2,
        
        # LoRA alpha (scaling factor, typically 2*r)
        "lora_alpha": 4,
        
        # LoRA dropout rate
        "lora_dropout": 0.05,
        
        
        # ==================== CLIP LoRA Configuration ====================
        # Enable LoRA fine-tuning for CLIP (if False, CLIP is fully frozen)
        "clip_lora_enabled": True,
        
        # Note: CLIP uses the same lora_r, lora_alpha, lora_dropout as LLM above
        # The LoRA is applied to CLIP's attention (qkv_proj, out_proj) and MLP (fc1, fc2) layers
        
        
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
        "nu_dataroot": "./nuscenes/train",
        
        # nuScenes version
        # Options: "v1.0-trainval", "v1.0-mini", "v1.0-test"
        "nu_version": "v1.0-trainval",
        
        # Path to SAM checkpoint (None = auto-download if auto_download_sam=True)
        "sam_ckpt": None,
        
        # Automatically download SAM weights if missing
        "auto_download_sam": True,
        
        # Data type for DeepEncoder processing
        # Options: "float32", "bfloat16" (bfloat16 faster but requires modern GPU)
        "deep_dtype": "float32",
        
        # OpenCLIP pretrained weights
        # Options: "openai", "laion400m_e32", "laion2b_s32b_b79k"
        "openclip_pretrained": "openai",
    }
    
    return config


def main():
    """
    Main training entry point.
    
    Modify the config in get_training_config() to customize training.
    """
    
    # Get comprehensive configuration
    config = get_training_config()
    
    # ==================== Quick Configuration Overrides ====================
    # Uncomment and modify these for quick experiments without editing the full config
    
    # Quick test (fast, minimal data)
    # config["max_samples"] = 10
    # config["epochs"] = 2
    # config["batch_size"] = 1
    # config["vat_queries"] = 12
    # config["vision_queries"] = 12
    
    # Full training (production)
    # config["max_samples"] = None  # Use all data
    # config["epochs"] = 50
    # config["batch_size"] = 2
    # config["grad_accum"] = 4
    # config["vat_queries"] = 576
    # config["vision_queries"] = 1536
    # config["fp16"] = True
    
    # LiDAR only (no vision)
    # config["use_vision"] = False
    
    # Large model
    # config["model_id"] = "Qwen/Qwen2.5-3B"
    # config["batch_size"] = 1
    # config["grad_accum"] = 8
    # config["fp16"] = True
    
    # High capacity VAT
    # config["vat_queries"] = 768
    # config["vat_layers"] = 6
    # config["vat_heads"] = 12
    # config["vision_queries"] = 2304
    # config["vision_layers"] = 6
    # config["vision_heads"] = 12
    
    # Custom learning rates
    # config["lr_vat"] = 1e-3
    # config["lr_vision_vat"] = 1e-3
    # config["lr_lora"] = 5e-4
    # config["warmup_steps"] = 500
    
    # Debug mode
    # config["debug_shapes"] = True
    # config["max_samples"] = 5
    # config["epochs"] = 1
    
    
    # ==================== Print Configuration ====================
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"\n{'='*30} I/O {'='*30}")
    print(f"Feature dirs: {config['feature_dirs']}")
    print(f"JSON files: {config['jsons']}")
    print(f"Output dir: {config['out_dir']}")
    print(f"Max samples: {config['max_samples']}")
    
    print(f"\n{'='*30} Training {'='*30}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Gradient accumulation: {config['grad_accum']}")
    print(f"Effective batch size: {config['batch_size'] * config['grad_accum']}")
    print(f"FP16: {config['fp16']}")
    print(f"Resume: {config['resume']}")
    print(f"Seed: {config['seed']}")
    
    print(f"\n{'='*30} Model {'='*30}")
    print(f"Base model: {config['model_id']}")
    print(f"Use vision: {config['use_vision']}")
    print(f"LiDAR VAT queries: {config['vat_queries']} (layers={config['vat_layers']}, heads={config['vat_heads']})")
    if config['use_vision']:
        print(f"Vision VAT queries: {config['vision_queries']} (layers={config['vision_layers']}, heads={config['vision_heads']})")
    print(f"LoRA rank: {config['lora_r']}, alpha: {config['lora_alpha']}")
    
    print(f"\n{'='*30} Optimization {'='*30}")
    print(f"LR VAT: {config['lr_vat']}")
    print(f"LR LoRA: {config['lr_lora']}")
    if config['use_vision']:
        print(f"LR Vision VAT: {config['lr_vision_vat']}")
        print(f"LR Vision: {config['lr_vision']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Warmup steps: {config['warmup_steps']}")
    print(f"Gradient clip norm: {config['clip_norm']}")
    
    print(f"\n{'='*30} Validation {'='*30}")
    print(f"Val split: {config['val_split']*100:.1f}%")
    print(f"Validate every: {config['validate_every']} epochs")
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    
    # ==================== Create Trainer and Run ====================
    try:
        trainer = Trainer(config)
        trainer.train()
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Training failed with error: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
