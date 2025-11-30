"""
Modal Training Configuration for LiDAR-Vision-LLM.

This provides the configuration for Modal cloud training runs.
Configuration is optimized for long-running experiments (24h+ with auto-resume).

Usage:
    from configs.modal_config import get_modal_training_config
    config = get_modal_training_config()
"""

from typing import Dict, List, Optional

# ============================================================================
# Modal Training Configuration
# ============================================================================

def get_modal_training_config() -> Dict:
    """
    Configuration for Week-Long Training on L4/H200.
    
    This config is optimized for:
    - Long-running training (24h sessions with auto-resume)
    - Modal cloud infrastructure
    - H200/L4 GPU memory constraints
    - Production-scale datasets
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
        "max_samples": 20_000,  # None for full dataset
        
        # =========================================================================
        # VALIDATION TOGGLE (for phased deployment)
        # =========================================================================
        # Phase 1: Run with skip_all_validation=False on cheap T4 to validate data
        # Phase 2: Set skip_all_validation=True and switch to H200 for training
        # When True, skips ALL data validation checks for faster startup
        "skip_all_validation": True,
        
        # BEV validation workers (for parallel shape/dtype checking of 37K+ files)
        # NOTE: Too many workers blocks Modal heartbeat! Keep at 16-20 max.
        # Phase 1: Use 16 workers (leaves CPU for system/heartbeat)
        # Phase 2: Not used when skip_all_validation=True
        "bev_validation_workers": 16,
        
        
        # ==================== Training Configuration ====================
        # Number of training epochs
        "epochs": 100,
        
        # Batch size per GPU
        # NOTE: Reduced from 4 to 2 due to OOM on A100-40GB with full multimodal setup
        # Using grad_accum=2 to maintain effective batch size of 4
        "batch_size": 20,
        
        # Gradient accumulation steps (effective_batch = batch_size * grad_accum * num_gpus)
        # Increased to compensate for smaller batch_size (2 * 2 = 4 effective)
        "grad_accum": 1,
        
        # Number of DataLoader workers for parallel data loading
        # Higher = better GPU utilization (overlaps data loading with training)
        # NOTE: Too many workers can block Modal heartbeat! Keep at 16-20 max.
        # Phase 1 (T4 validation): Use 16 workers (leaves CPU for Modal heartbeat)
        # Phase 2 (H200 training): Use 16 workers with 24 cores
        "num_workers": 16,
        
        # Prefetch factor: batches to prefetch per worker (requires num_workers > 0)
        # Higher = more memory usage but better GPU utilization
        "prefetch_factor": 2,
        
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
        "inference_samples_n": 40,
        
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
        "target_field": "answer",  # or answer_lidar
        
        # Maximum answer tokens (longer answers will be truncated)
        "max_ans_toks": 256,
        
        # DEPRECATED: prefix_scale is no longer used externally.
        # VAT models (VATLiDAR, VATVision) now include learnable output_scale parameters
        # initialized to 1.0 that adapt during training to match LLM embedding magnitudes.
        # Kept for backward compatibility with old checkpoints.
        "prefix_scale": 1.0,  # DEPRECATED - VAT models handle scaling internally
        
        
        # ==================== LiDAR VAT Configuration ====================
        # Number of learnable query tokens for LiDAR VAT
        # MUST be divisible by 6 (for 6 spatial sectors)
        # Recommended: 12 (testing), 576 (medium), 768 (large)
        "vat_queries": 768,
        
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
        "vision_per_view_query": False,  # keep False 
        
        # If True, error when per-view not feasible; if False, auto-disable with warning
        "vision_strict_per_view": False,  # keep False
        
        
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
        "lora_r": 32,
        
        # LoRA alpha (scaling factor, typically 2*r)
        "lora_alpha": 64,
        
        # LoRA dropout rate
        "lora_dropout": 0.05,
        
        # LLM LoRA target modules (which layers to apply LoRA to)
        # Common targets for Qwen/LLaMA-style models:
        #   - Attention: "q_proj", "k_proj", "v_proj", "o_proj"
        #   - MLP: "gate_proj", "up_proj", "down_proj"
        # Set to None to use defaults, or customize the list
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        
        # ==================== CLIP LoRA Configuration ====================
        # Enable LoRA fine-tuning for CLIP (if False, CLIP is fully frozen)
        "clip_lora_enabled": True,
        
        # CLIP LoRA target modules (which layers to apply LoRA to)
        # Common targets for CLIP ViT models:
        #   - Attention: "qkv_proj", "out_proj" (combined Q/K/V projection)
        #   - MLP: "mlp.fc1", "mlp.fc2"
        # Set to None to use auto-detected defaults from infer_clip_lora_targets()
        # Note: CLIP uses the same lora_r, lora_alpha, lora_dropout as LLM above
        "clip_lora_target_modules": ["qkv_proj", "out_proj", "mlp.fc1", "mlp.fc2"],
        
        
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
        "use_torch_compile": False,
        
        # torch.compile mode
        # Options: "default", "reduce-overhead" (recommended for training), "max-autotune"
        "torch_compile_mode": "reduce-overhead",
        
        
        # ==================== Modal-Specific Configuration ====================
        # Root directory for all training runs
        "checkpoints_root": "/data/checkpoints",
        
        # Timeout in seconds (24 hours)
        "timeout": 86400,
    }

    return config


__all__ = ["get_modal_training_config"]
