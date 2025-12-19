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
        
        "debug_mode": True,      # ← SET TO True TO ENABLE DEBUG LOGGING
        "debug_level": 3,         # ← 0=DISABLED, 1=INFO, 2=DEBUG, 3=TRACE
    
        # ──────────────────────────────────────────────────────────────────
        # JSON/JSONL file paths
        "caption_json": "/data/Datasets/nuScenes/external/vision_finetuning_dataset.json",
        # ──────────────────────────────────────────────────────────────────

                
        # ──────────────────────────────────────────────────────────────────
        # Output directory for checkpoints, logs, and plots
        "out_dir": "/data/checkpoints",
        # ──────────────────────────────────────────────────────────────────
        
        # Maximum number of samples to use (None = use all data)
        # Set to small number (e.g., 10) for quick testing
        "max_samples": 16,
        
        
        # =========================================================================
        # VALIDATION TOGGLE (for phased deployment)
        # =========================================================================
        # Phase 1: Run with skip_all_validation=False on cheap T4 to validate data
        # Phase 2: Set skip_all_validation=True and switch to H200 for training
        # When True, skips ALL data validation checks for faster startup
        "skip_all_validation": True,
        
        
        # ==================== Vision Toggle ====================
        # Enable vision processing (always True for this vision-only model)
        "use_vision": True,
        
        
        # ==================== Training Configuration ====================
        # Number of training epochs
        "epochs": 5,
        
        # Batch size per GPU
        # Using grad_accum=2 to maintain effective batch size of 32
        "batch_size": 16,
        
        # Gradient accumulation steps (effective_batch = batch_size * grad_accum * num_gpus)
        # Increased to compensate for smaller batch_size (2 * 16 = 32 effective)
        "grad_accum": 1,
        
        # Number of DataLoader workers for parallel data loading
        "num_workers": 16,
        
        # Prefetch factor: batches to prefetch per worker (requires num_workers > 0)
        # Higher = more CPU RAM usage but better GPU utilization
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
        # This is critical for long training runs - allows resuming from last saved step
        "save_every_steps": 10,
        
        # Keep only last N checkpoints (older ones are deleted)
        "keep_last_n": 3,
        
        # Plot loss curves every N epochs (currently not used in loop, always plots)
        "plot_every": 1,
        
        # ==================== Validation Configuration ====================
        # Percentage of data to use for validation (0.1 = 10%, 0.05 = 5%)
        "val_split": 0.05,
        
        # Run validation every N epochs
        "validate_every": 1,
        
        # System prompt for the model (used in chat template)
        "system_prompt": """You are an Autonomous Driving Perception Assistant analyzing six surround-view camera images (Front, Front-Left, Front-Right, Back, Back-Left, Back-Right). Your task is to fuse these views into a consistent scene understanding.

                            Follow the output template based on the requested mode:

                            1. @SCENE_SUMMARY_MODE
                            Output Template: A cohesive narrative text describing the driving scene.

                            2. @ENTITY_LIST_MODE
                            Output Template: Strictly valid JSON using the following schema:
                            {
                                "detected_entities": [
                                [ "Category", "Type", { "KeyAttributes": "Value" }, "Relative_Location" ]
                                ]
                            }""",
        
        
        # ==================== Inference Sampling Configuration ====================
        # Generate predictions on validation samples every N epochs (0 = disable)
        "inference_sampling_every": 2,
        
        # Total number of samples to generate
        "inference_samples_n": 5,
        
        # Test JSON files for inference sampling (used based on dataset_mode)
        "inference_caption_json": None,
        
        # Generation parameters for inference sampling
        "inference_max_tokens": 256,
        "inference_temperature": 0.0,
        "inference_do_sample": False,
        "inference_num_beams": 1,
        "inference_batch_size": 2,
        
        
        # ==================== Evaluation Metrics Toggles ====================
        # Enable/disable specific metrics for each dashboard
        # Caption Dashboard Metrics (text quality only)
        "eval_caption_bleu4": True,
        "eval_caption_cider": True,
        "eval_caption_spice": False,
        "eval_caption_bertscore": False,
        

        # ==================== Model Configuration ====================
        # Hugging Face model ID for base LLM
        # Options: "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"
        "model_id": "Qwen/Qwen2.5-3B",
        
        # Field name in JSON containing target answer
        "target_field": "answer",
        
        # Maximum answer tokens (longer answers will be truncated)
        "max_ans_toks": 256,
              
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
        
        # ==================== Tuning Configuration (LLM) ====================
        # "qlora" : 4-bit quantization + LoRA adapters (Default, most efficient)
        # "lora"  : BF16/FP16 + LoRA adapters (Standard LoRA)
        # "full"  : Full fine-tuning (Heavy memory usage)
        "tuning_mode": "qlora",
        
        # LoRA rank (higher = more parameters, more expressive)
        # Typical values: 8-16 for QLoRA (can use higher rank due to memory savings)
        "llm_lora_r": 8,
        
        # LoRA alpha (scaling factor, typically 2*r)
        "llm_lora_alpha": 16,
        
        # LoRA dropout rate
        "llm_lora_dropout": 0.10,
        
        # Target modules for LoRA application (Qwen specific)
        # Apply to all linear layers for best performance per QLoRA paper
        "llm_lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        
        # ==================== CLIP LoRA Configuration ====================
        # Enable LoRA fine-tuning for CLIP (if False, CLIP is fully frozen)
        "clip_lora_enabled": True,

        # CLIP LoRA rank and alpha (decoupled from LLM)
        "clip_lora_r": 8,
        "clip_lora_alpha": 16,
        "clip_lora_dropout": 0.10,
        
        # CLIP LoRA target modules (which layers to apply LoRA to)
        # Common targets for CLIP ViT models:
        #   - Attention: "qkv_proj", "out_proj" (combined Q/K/V projection)
        #   - MLP: "mlp.fc1", "mlp.fc2" ["qkv_proj", "out_proj", "mlp.fc1", "mlp.fc2"] 
        # Set to None to use auto-detected defaults from infer_clip_lora_targets()
        "clip_lora_target_modules": None,
        
        
        # ==================== Optimization Configuration ====================
        # Learning rate for LLM LoRA adapters
        "lr_lora": 3e-4,
        
        # Learning rate for vision components (VisionAdapter, DeepEncoder projector, CLIP LoRA)
        "lr_vision": 5e-4,
        
        # Weight decay for regularization
        "weight_decay": 0.05,  # Increased from 0.01 to combat overfitting
        
        # Number of warmup steps for learning rate scheduler
        "warmup_steps": 10,
        
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
        # NOTE: Disabled due to CUDA graph conflicts with tensor caching
        "use_torch_compile": False,
        
        # torch.compile mode
        # Options: "default", "reduce-overhead" (recommended for training), "max-autotune"
        # "torch_compile_mode": "reduce-overhead",
        
        
        # Timeout in seconds (24 hours)
        "timeout": 86400,
    }
    
    # Auto-configure jsons list (requires caption_json to be set)
    config["jsons"] = [config["caption_json"]]
    
    return config


__all__ = ["get_modal_training_config"]
