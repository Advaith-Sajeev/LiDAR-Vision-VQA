"""
Training configuration template for LiDAR-Vision-LLM.

This module provides a comprehensive training configuration with all available options
explicitly documented. Modify values as needed for your training setup.

Usage:
    from configs import get_training_config
    config = get_training_config()
    # Modify config as needed
    config["epochs"] = 20
"""

from typing import Dict


def get_training_config() -> Dict:
    """
    Get comprehensive training configuration with all available options.
    
    All options are explicitly shown here for easy customization.
    Modify values as needed.
    
    IMPORTANT:
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
        
        
        # ==================== I/O Configuration ====================
        # Directories containing BEV feature .npy files (one per sample_token)
        "feature_dirs": ["/home/j_bindu/fyp-26-grp-38/bev_feats"],
        
        # =========================================================================
        # DATASET MODE SELECTION
        # =========================================================================
        # Choose which dataset(s) to use for training, validation, and inference:
        #   "caption"   - Only nuCaption dataset (scene descriptions)
        #   "grounding" - Only nuGrounding dataset (object detection/localization)
        #   "both"      - Both datasets combined (default, recommended for full training)
        # 
        # This setting affects:
        #   1. Training/validation data (jsons list)
        #   2. Inference sampling (which test JSONs to load)
        #   3. Evaluation metrics (caption vs grounding dashboards)
        # 
        # SAFETY: When using "caption" mode, grounding paths can be set to None
        #         to prevent any accidental data leakage.
        "dataset_mode": "both",  # "caption", "grounding", or "both"
        
        # JSON/JSONL file paths for each dataset type
        # These are used based on the dataset_mode setting above
        # Set to None to disable a dataset (safe when not using that mode)
        "caption_json": "/home/j_bindu/fyp-26-grp-38/Dataset_subset/external/nuCaption.json",
        "grounding_json": "/home/j_bindu/fyp-26-grp-38/Dataset_subset/external/nuGrounding.json",
        
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
        
        # Print tensor shapes during forward pass (for debugging)
        "debug_shapes": False, 
        
        
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
        # For "caption" mode: can be any positive integer
        # For "grounding" mode: must be divisible by 2
        # For "both" mode: must be divisible by 4
        "inference_samples_n": 12,
        
        # Test JSON files for inference sampling (used based on dataset_mode)
        # Set to None to disable (safe when not using that mode)
        # Default to training split unless explicitly overridden
        "inference_caption_json": None,
        "inference_grounding_json": None,
        
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
        
        # Grounding Det Area Dashboard Metrics (text quality + bbox accuracy)
        "eval_det_area_bleu4": True,
        "eval_det_area_cider": True,
        "eval_det_area_spice": True,
        "eval_det_area_bertscore": True,
        "eval_det_area_top1_acc": True,      # Object class identification accuracy
        "eval_det_area_bev_iou": True,       # 2D Bird's Eye View IoU
        
        # Grounding Det Object Dashboard Metrics (text quality only)
        "eval_det_object_bleu4": True,
        "eval_det_object_cider": True,
        "eval_det_object_spice": True,
        "eval_det_object_bertscore": True,
        
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
        "model_id": "Qwen/Qwen2.5-0.5B",
        
        # Field name in JSON containing target answer
        "target_field": "answer", # or answer_lidar
        
        # Maximum answer tokens (longer answers will be truncated)
        "max_ans_toks": 64,
        
        # DEPRECATED: prefix_scale is no longer used externally.
        # VAT models (VATLiDAR, VATVision) now include learnable output_scale parameters
        # initialized to 1.0 that adapt during training to match LLM embedding magnitudes.
        # Kept for backward compatibility with old checkpoints.
        "prefix_scale": 1.0,  # DEPRECATED - VAT models handle scaling internally
        
        
        # ==================== LiDAR VAT Configuration ====================
        # Number of learnable query tokens for LiDAR VAT
        # MUST be divisible by 6 (for 6 spatial sectors)
        # Recommended: 12 (testing), 576 (medium), 768 (large)
        "vat_queries": 6,
        
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
        "vision_queries": 2,
        
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
        "vision_per_view_query": False, # keep False 
        
        # If True, error when per-view not feasible; if False, auto-disable with warning
        "vision_strict_per_view": False, # keep False
        
        
        # ==================== QLoRA Configuration ====================
        # Enable 4-bit quantization (QLoRA) for memory-efficient training
        # When enabled, the base LLM is loaded in 4-bit precision
        "use_qlora": False,  # Set to True for memory-efficient training
        
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
        "lora_r": 2,
        
        # LoRA alpha (scaling factor, typically 2*r)
        "lora_alpha": 4,
        
        # LoRA dropout rate
        "lora_dropout": 0.05,
        
        # LLM LoRA target modules (which layers to apply LoRA to)
        # Common targets for Qwen/LLaMA-style models:
        #   - Attention: "q_proj", "k_proj", "v_proj", "o_proj"
        #   - MLP: "gate_proj", "up_proj", "down_proj"
        # Set to None to use defaults, or customize the list eg: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        "lora_target_modules": ["q_proj",  "down_proj"],
        
        
        # ==================== CLIP LoRA Configuration ====================
        # Enable LoRA fine-tuning for CLIP (if False, CLIP is fully frozen)
        "clip_lora_enabled": True,
        
        # CLIP LoRA target modules (which layers to apply LoRA to)
        # Common targets for CLIP ViT models:
        #   - Attention: "qkv_proj", "out_proj" (combined Q/K/V projection)
        #   - MLP: "mlp.fc1", "mlp.fc2"
        # Set to None to use auto-detected defaults from clip_l_lora_default_targets()
        # Note: CLIP uses the same lora_r, lora_alpha, lora_dropout as LLM above
        "clip_lora_target_modules": ["qkv_proj", "out_proj"],  # None = auto-detect, or provide list like ["qkv_proj", "out_proj", "mlp.fc1", "mlp.fc2"]
        
        
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
    
    # =========================================================================
    # BUILD DYNAMIC CONFIGS BASED ON DATASET MODE
    # =========================================================================
    # Automatically build the jsons list based on dataset_mode setting
    # Validates that required paths are configured for the selected mode
    mode = config["dataset_mode"]
    
    if mode == "caption":
        # Caption mode: only caption_json is required
        if not config.get("caption_json"):
            raise ValueError(
                "dataset_mode='caption' requires 'caption_json' to be set. "
                "Please configure a valid path to the caption JSON file."
            )
        config["jsons"] = [config["caption_json"]]
        
    elif mode == "grounding":
        # Grounding mode: only grounding_json is required
        if not config.get("grounding_json"):
            raise ValueError(
                "dataset_mode='grounding' requires 'grounding_json' to be set. "
                "Please configure a valid path to the grounding JSON file."
            )
        config["jsons"] = [config["grounding_json"]]
        
    elif mode == "both":
        # Both mode: both paths are required
        if not config.get("caption_json"):
            raise ValueError(
                "dataset_mode='both' requires 'caption_json' to be set. "
                "Please configure a valid path to the caption JSON file."
            )
        if not config.get("grounding_json"):
            raise ValueError(
                "dataset_mode='both' requires 'grounding_json' to be set. "
                "Please configure a valid path to the grounding JSON file."
            )
        config["jsons"] = [config["caption_json"], config["grounding_json"]]
        
    else:
        raise ValueError(
            f"Invalid dataset_mode: '{mode}'. "
            f"Must be one of: 'caption', 'grounding', 'both'"
        )
    
    return config


__all__ = ["get_training_config"]
