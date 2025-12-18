"""
Local Training configuration for LiDAR-Vision-LLM.

This module provides the configuration for local (non-Modal) training runs.
Modify values as needed for your local training setup.

Usage:
    from local_config import get_local_config
    config = get_local_config()
    # Modify config as needed
    config["epochs"] = 20
"""

from typing import Dict


def get_local_config() -> Dict:
    """
    Get local training configuration with all available options.
    
    All options are explicitly shown here for easy customization.
    Modify values as needed.
    
    IMPORTANT (Only for src\encoder-decoder\train.py):
    - If resume=False: Set out_dir to a base directory (e.g., "./checkpoints") 
                      The system will create a timestamped subdirectory
    - If resume=True:  Set out_dir to point to a specific run directory 
                      (e.g., "./checkpoints/run_20251110_143052") OR set to base 
                      directory and you'll be prompted to select a run
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
        
        "debug_mode": True,      # ← SET TO True TO ENABLE DEBUG LOGGING
        "debug_level": 3,         # ← 1=INFO, 2=DEBUG (recommended), 3=TRACE
        "debug_modules": [],      # ← [] = all modules, or ["trainer", "dataset"]
        
        

        
        # JSON/JSONL file paths for each dataset type
        "caption_json": "/home/j_bindu/fyp-26-grp-38/Dataset_subset/external/nuCaption.json",
        
        # Output directory for checkpoints, logs, and plots
        # If resume=False: Use base directory (e.g., "./checkpoints")
        # If resume=True: Use specific run directory OR base directory (will prompt)
        "out_dir": "./checkpoints",
        
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
        
        # Mixed precision training mode
        # Options: "no" (disabled), "fp16" (float16), "bf16" (bfloat16)
        # bf16 is recommended for modern GPUs (better numerical stability than fp16)
        "mixed_precision": "bf16",  # "no", "fp16", or "bf16"
        
        # Resume from checkpoint if available
        "resume": False,
        
        # Save checkpoint every N steps (0 = disable step-based saving)
        "save_every_steps": 1000,
        
        # Keep only last N checkpoints (older ones are deleted)
        "keep_last_n": 5,
        
        # Plot loss curves every N epochs (currently not used in loop, always plots)
        "plot_every": 1,
        

        
        
        # ==================== Validation Configuration ====================
        # Percentage of data to use for validation (0.05 = 5%)
        "val_split": 0.05,
        
        # Run validation every N epochs
        "validate_every": 1,
        
        # System prompt for the model (used in chat template)
        "system_prompt": "You are an expert autonomous driving assistant. Analyze the 3D LiDAR point cloud and camera images to understand the driving scene. Provide accurate, concise descriptions for the question asked.",
        
        
        # ==================== Inference Sampling Configuration ====================
        # Generate predictions on validation samples every N epochs
        "inference_sampling_every": 1,
        
        # Total number of samples to generate
        "inference_samples_n": 12,
        
        # Test JSON files for inference sampling
        # Set to None to disable (safe when not using that mode)
        # Default to training split unless explicitly overridden
        "inference_caption_json": None,

        
        # Generation parameters for inference sampling
        "inference_max_tokens": 64,
        "inference_temperature": 0.7,
        "inference_top_p": 0.9,
        "inference_top_k": 50,
        "inference_do_sample": True,
        "inference_num_beams": 1,
        "inference_batch_size": 2,
        
        # ==================== Evaluation Metrics Toggles ====================
        # Enable/disable specific metrics for each dashboard
        # Caption Dashboard Metrics (text quality only)
        "eval_caption_bleu4": True,
        "eval_caption_cider": True,
        "eval_caption_spice": True,
        "eval_caption_bertscore": True,
        

        

        
        
        # ==================== Model Configuration ====================
        # Hugging Face model ID for base LLM
        # Options: "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"
        "model_id": "Qwen/Qwen2.5-0.5B",
        
        # Field name in JSON containing target answer
        "target_field": "answer",
        
        # Maximum answer tokens (longer answers will be truncated)
        "max_ans_toks": 64,
        
        # DEPRECATED: prefix_scale is no longer used externally.
        # VAT models (VATLiDAR, VATVision) now include learnable output_scale parameters
        # initialized to 1.0 that adapt during training to match LLM embedding magnitudes.
        # Kept for backward compatibility with old checkpoints.
        "prefix_scale": 1.0,  # DEPRECATED - VAT models handle scaling internally
        
        

        
        
        # ==================== Vision Configuration ====================
        # Vision Adapter configuration
        # Dropout rate for Vision Adapter
        "vision_dropout": 0.10,
        
        
        # ==================== Tuning Configuration (LLM) ====================
        # "qlora" : 4-bit quantization + LoRA adapters (Default, most efficient)
        # "lora"  : BF16/FP16 + LoRA adapters (Standard LoRA)
        # "full"  : Full fine-tuning (Heavy memory usage)
        "tuning_mode": "qlora",
        
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
        # Typical values: 2-8 for small models, 8-16 for large models
        "llm_lora_r": 2,
        
        # LoRA alpha (scaling factor, typically 2*r)
        "llm_lora_alpha": 4,
        
        # LoRA dropout rate
        "llm_lora_dropout": 0.05,
        
        # LLM LoRA target modules (which layers to apply LoRA to)
        # Common targets for Qwen/LLaMA-style models:
        #   - Attention: "q_proj", "k_proj", "v_proj", "o_proj"
        #   - MLP: "gate_proj", "up_proj", "down_proj"
        # Set to None to use defaults, or customize the list eg: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        "llm_lora_targets": ["q_proj", "down_proj"],
        
        
        # ==================== CLIP LoRA Configuration ====================
        # Enable LoRA fine-tuning for CLIP (if False, CLIP is fully frozen)
        "clip_lora_enabled": True,

        # CLIP LoRA rank and alpha (decoupled from LLM)
        "clip_lora_r": 2,
        "clip_lora_alpha": 4,
        "clip_lora_dropout": 0.05,
        
        # CLIP LoRA target modules (which layers to apply LoRA to)
        # Common targets for CLIP ViT models:
        #   - Attention: "qkv_proj", "out_proj" (combined Q/K/V projection)
        #   - MLP: "mlp.fc1", "mlp.fc2"
        # Set to None to use auto-detected defaults from clip_l_lora_default_targets()
        "clip_lora_target_modules": ["qkv_proj", "out_proj"],  # None = auto-detect, or provide list like ["qkv_proj", "out_proj", "mlp.fc1", "mlp.fc2"]
        
        
        # ==================== Optimization Configuration ====================

        

        # Learning rate for LLM LoRA adapters
        "lr_lora": 3e-4,
        
        # Learning rate for vision components (VisionAdapter, DeepEncoder projector, CLIP LoRA)
        "lr_vision": 5e-4,
        
        # Weight decay for regularization
        "weight_decay": 0.01,
        
        # Number of warmup steps for learning rate scheduler
        "warmup_steps": 100,
        
        # Gradient clipping norm (prevents exploding gradients)
        "clip_norm": 1.0,
        
        
        # ==================== nuScenes / DeepEncoder Configuration ====================
        # Path to nuScenes dataset root directory
        # Should contain folders: samples, sweeps, maps, etc.
        "nu_dataroot": "/home/j_bindu/fyp-26-grp-38/Dataset_subset",
        
        # nuScenes version
        # Options: "v1.0-trainval", "v1.0-mini", "v1.0-test"
        "nu_version": "v1.0-trainval",
        
        # Path to SAM checkpoint (None = auto-download if auto_download_sam=True)
        "sam_ckpt": None,
        
        # Automatically download SAM weights if missing
        "auto_download_sam": True,
        
        # Data type for DeepEncoder processing
        # Options: "float32", "bfloat16" (bfloat16 faster but requires modern GPU)
        "deep_dtype": "bfloat16",
        
        # OpenCLIP pretrained weights
        # Options: "openai", "laion400m_e32", "laion2b_s32b_b79k"
        "openclip_pretrained": "openai",
    }
    
    # Auto-configure jsons list
    config["jsons"] = [config["caption_json"]]
    
    return config


__all__ = ["get_local_config"]
