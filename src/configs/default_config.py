"""
Default configuration for LiDAR-Vision-LLM training.

This provides fallback values for all configuration options.
These defaults are used when specific values are not provided in the training config.
"""

from typing import Dict, Tuple
from enum import IntEnum


# ============================================================================
# Multimodal Sequence Position Configuration
# ============================================================================
# Defines the canonical ordering of modality components in LLM input sequences.
# This ordering is critical for:
# - Training: ensures consistent input structure across batches
# - Inference: must match training order for model to interpret inputs correctly
# - Checkpointing: position meanings must be stable across training runs
#
# SEQUENCE LAYOUT (when all modalities enabled):
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ [vision_start] [vision_tokens...] [vision_end]                          │
# │ [lidar_start]  [lidar_tokens...]  [lidar_end]                           │
# │ [text_prompt...]                                                         │
# │ [answer_tokens...]  (training only)                                      │
# └─────────────────────────────────────────────────────────────────────────┘
#
# If a modality is disabled, its positions are skipped but relative order preserved.

class ModalityPosition(IntEnum):
    """
    Explicit position markers for each modality component in the LLM input sequence.
    
    The numerical values define the canonical ordering - lower values come first.
    This enum ensures consistent sequence construction across training, validation,
    and inference, preventing subtle bugs from mismatched embedding order.
    
    Usage:
        - VISION_START/END: Delimiter tokens for vision embeddings
        - VISION_TOKENS: Projected vision features from VATVision
        - LIDAR_START/END: Delimiter tokens for LiDAR embeddings  
        - LIDAR_TOKENS: Projected LiDAR features from VATLiDAR
        - TEXT_PROMPT: Tokenized text question/instruction
        - ANSWER_TOKENS: Ground truth answer tokens (training only)
    
    Note: Vision always precedes LiDAR, and both precede text. This ordering
    was chosen to match the natural "see then reason" cognitive flow.
    """
    VISION_START = 0      # <vision_start> delimiter token
    VISION_TOKENS = 1     # Vision VAT output [n_queries, d_model]
    VISION_END = 2        # <vision_end> delimiter token
    LIDAR_START = 3       # <lidar_start> delimiter token
    LIDAR_TOKENS = 4      # LiDAR VAT output [n_queries, d_model]
    LIDAR_END = 5         # <lidar_end> delimiter token
    TEXT_PROMPT = 6       # Tokenized prompt embeddings [seq_len, d_model]
    ANSWER_TOKENS = 7     # Answer embeddings (training only) [ans_len, d_model]


# ============================================================================
# Camera View Configuration
# ============================================================================
# Fixed 6-view order for nuScenes cameras
# This order is used consistently across:
# - DeepEncoder (image encoding)
# - VisionAdapter (view embedding)
# - VATVision (per-view queries)
# - Dataset loading
DEFAULT_VIEW_ORDER: Tuple[str, ...] = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
)

# Alias for backward compatibility
CAM_VIEWS: Tuple[str, ...] = DEFAULT_VIEW_ORDER
NUM_VIEWS: int = len(DEFAULT_VIEW_ORDER)  # 6

# ============================================================================
# Vision Pipeline Constants
# ============================================================================
# These constants define the fixed dimensions in the vision encoding pipeline.
# They are derived from the DeepEncoder architecture and should NOT be changed
# unless the underlying model architecture changes.

# DeepEncoder outputs 16x16 grid of tokens per view
FIXED_GRID_SIDE: int = 16
TOKENS_PER_VIEW: int = FIXED_GRID_SIDE * FIXED_GRID_SIDE  # 256 tokens per camera view

# DeepEncoder projector output dimension (CLIP 1024 + SAM 1024 = 2048)
PROJECTOR_DIM: int = 2048

# Total vision tokens when all 6 views are concatenated
TOTAL_VISION_TOKENS: int = NUM_VIEWS * TOKENS_PER_VIEW  # 6 * 256 = 1536

# ============================================================================
# Default Training Configuration
# ============================================================================
DEFAULT_CONFIG: Dict = {
    # I/O
    "feature_dirs": ["./bev_feats/train"],      # list of directories containing <sample_token>.npy
    "jsons": ["Dataset_subset/external/nuCaption.json", "Dataset_subset/external/nuGrounding.json"],
    "out_dir": "./checkpoints_vat",
    "max_samples": 10,                          # int or None
    
    # =========================================================================
    # Data Validation Settings
    # =========================================================================
    # Validates data integrity during dataset initialization to catch issues early.
    # Prevents silent failures from inconsistent data (mixed model outputs, corruption, etc.)
    # All validations are ON by default and check the ENTIRE dataset for maximum safety.
    
    # BEV Feature Shape Validation
    "validate_bev_shapes": True,                # Enable BEV shape consistency check
    "validate_all_bev_shapes": True,            # Validate ALL files (thorough)
    "bev_validation_sample_fraction": 1.0,      # Check 100% of files
    "bev_validation_min_samples": 10,           # Minimum files to validate
    "bev_validation_max_samples": 100000,       # Effectively unlimited
    "bev_validation_workers": 16,               # Parallel workers for validation
    
    # JSON/QA Validation  
    "validate_json_schema": True,               # Validate JSON structure (required fields)
    "validate_token_coverage": True,            # Check that JSON tokens have matching BEV files
    
    # Image Validation (when use_vision=True)
    "validate_image_paths": True,               # Verify camera image paths exist
    "validate_image_shapes": True,              # Check image dimensions
    
    # BEV Value Validation (checks for corruption)
    "validate_bev_dtype": True,                 # Check dtype, NaN/Inf, value ranges

    # Train
    "epochs": 10,
    "batch_size": 1,
    "grad_accum": 1,
    "seed": 42,
    "mixed_precision": "bf16",  # "no", "fp16", or "bf16" (recommended for modern GPUs)
    "gradient_checkpointing": True,  # Trade compute for memory (enables larger batch sizes)
    "resume": True,
    "save_every_steps": 1000,
    "keep_last_n": 5,
    "plot_every": 1,
    "debug_shapes": False,                      # print tensor shapes at key points

    "val_split": 0.05,                          # percent of data to use for validation
    "validate_every": 1,                        # evaluate every N epochs
    "val_inference_n": 10,                      # number of validation samples to save

    # Model / tokens
    "model_id": "Qwen/Qwen2.5-0.5B",
    "target_field": "answer",
    "max_ans_toks": 32,
    
    # DEPRECATED: prefix_scale is no longer used externally.
    # VAT models (VATLiDAR, VATVision) now include learnable output_scale parameters
    # initialized to 1.0 that adapt during training to match LLM embedding magnitudes.
    # This replaced the arbitrary fixed scaling that didn't account for:
    # - Different magnitudes between LiDAR and Vision VAT outputs
    # - The scale relationship to LLM text embeddings
    # - The fact that VAT outputs are already LayerNorm'd
    # Keeping this config for backward compatibility with old checkpoints.
    "prefix_scale": 1.0,  # DEPRECATED - VAT models handle scaling internally

    # LiDAR VAT
    "vat_queries": 12,                          # must be divisible by 6
    "vat_layers": 1,
    "vat_heads": 2,
    "vat_mlp_ratio": 4.0,
    "vat_dropout": 0.10,
    "vat_post_dropout": 0.10,

    # Vision VAT
    "use_vision": True,
    "vision_queries": 12,                       # must be divisible by 6
    "vision_layers": 1,
    "vision_heads": 2,
    "vision_mlp_ratio": 4.0,
    "vision_dropout": 0.10,
    "vision_post_dropout": 0.10,
    "vision_per_view_query": True,
    "vision_strict_per_view": False,            # If True, error when per-view not feasible; if False, auto-disable

    # LoRA
    "lora_r": 2,
    "lora_alpha": 4,
    "lora_dropout": 0.05,

    # Optim
    "lr_vat": 5e-4,
    "lr_vision_vat": 5e-4,
    "lr_lora": 3e-4,
    "lr_vision": 5e-4,                          # VisionAdapter + DeepEncoder projector + CLIP LoRA
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "clip_norm": 1.0,

    # nuScenes / DeepEncoder
    "nu_dataroot": "./nuscenes/train",
    "nu_version": "v1.0-trainval",
    "sam_ckpt": None,                           # Path to SAM checkpoint, or None to auto-download
    "auto_download_sam": True,                  # Auto-download SAM weights if missing
    "deep_dtype": "bfloat16",                   # "bfloat16", "float16", or "float32" (bf16 recommended for modern GPUs)
    "openclip_pretrained": "openai",
    
    # Performance optimizations
    "num_workers": 4,                           # DataLoader workers (0=main process, >0=parallel loading, 4-8 recommended)
    "prefetch_factor": 2,                       # Batches to prefetch per worker (requires num_workers > 0)
    "use_torch_compile": False,                 # Enable torch.compile() for VAT models (PyTorch 2.0+, 10-30% speedup)
    "torch_compile_mode": "reduce-overhead",    # "default", "reduce-overhead", or "max-autotune"
    
    # CUDA optimizations
    "cudnn_benchmark": True,                    # Enable cuDNN autotuning (faster for fixed-size inputs)
    
    # Inference optimizations
    "inference_batch_size": 8,                  # Batch size for encoding during inference sampling (higher = faster but more VRAM)
}

__all__ = [
    "DEFAULT_CONFIG", 
    "DEFAULT_VIEW_ORDER", 
    "CAM_VIEWS", 
    "NUM_VIEWS",
    "FIXED_GRID_SIDE",
    "TOKENS_PER_VIEW",
    "PROJECTOR_DIM",
    "TOTAL_VISION_TOKENS",
    "ModalityPosition",
]
