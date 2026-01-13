"""
Shared constants for Vision-LLM training.

This module contains constants that are used across both Modal and local training configs.
These values define the core architecture and should not be changed without careful consideration.
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
# SEQUENCE LAYOUT (vision-only with per-view delimiters):
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ [cam_front_start] [256 tokens] [cam_front_end]                          │
# │ [cam_front_right_start] [256 tokens] [cam_front_right_end]              │
# │ [cam_front_left_start] [256 tokens] [cam_front_left_end]                │
# │ [cam_back_start] [256 tokens] [cam_back_end]                            │
# │ [cam_back_right_start] [256 tokens] [cam_back_right_end]                │
# │ [cam_back_left_start] [256 tokens] [cam_back_left_end]                  │
# │ [text_prompt...]                                                         │
# │ [answer_tokens...]  (training only)                                      │
# └─────────────────────────────────────────────────────────────────────────┘

class ModalityPosition(IntEnum):
    """
    Explicit position markers for each modality component in the LLM input sequence.
    
    The numerical values define the canonical ordering - lower values come first.
    This enum ensures consistent sequence construction across training, validation,
    and inference, preventing subtle bugs from mismatched embedding order.
    
    Each camera view has its own start delimiter, tokens, and end delimiter.
    Views are ordered: FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_RIGHT, BACK_LEFT
    """
    # CAM_FRONT (view 0)
    CAM_FRONT_START = 0
    CAM_FRONT_TOKENS = 1
    CAM_FRONT_END = 2
    
    # CAM_FRONT_RIGHT (view 1)
    CAM_FRONT_RIGHT_START = 3
    CAM_FRONT_RIGHT_TOKENS = 4
    CAM_FRONT_RIGHT_END = 5
    
    # CAM_FRONT_LEFT (view 2)
    CAM_FRONT_LEFT_START = 6
    CAM_FRONT_LEFT_TOKENS = 7
    CAM_FRONT_LEFT_END = 8
    
    # CAM_BACK (view 3)
    CAM_BACK_START = 9
    CAM_BACK_TOKENS = 10
    CAM_BACK_END = 11
    
    # CAM_BACK_RIGHT (view 4)
    CAM_BACK_RIGHT_START = 12
    CAM_BACK_RIGHT_TOKENS = 13
    CAM_BACK_RIGHT_END = 14
    
    # CAM_BACK_LEFT (view 5)
    CAM_BACK_LEFT_START = 15
    CAM_BACK_LEFT_TOKENS = 16
    CAM_BACK_LEFT_END = 17
    
    # Text and answer (after all vision)
    TEXT_PROMPT = 18
    ANSWER_TOKENS = 19


# ============================================================================
# Camera View Configuration
# ============================================================================
# Fixed 6-view order for nuScenes cameras
# This order is used consistently across:
# - DeepEncoder (image encoding)
# - VisionAdapter (view embedding)
# - Sequence building (per-view delimiters)
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

# Mapping from view name to ModalityPosition triplet (start, tokens, end)
VIEW_POSITIONS: Dict[str, Tuple[ModalityPosition, ModalityPosition, ModalityPosition]] = {
    "CAM_FRONT": (ModalityPosition.CAM_FRONT_START, ModalityPosition.CAM_FRONT_TOKENS, ModalityPosition.CAM_FRONT_END),
    "CAM_FRONT_RIGHT": (ModalityPosition.CAM_FRONT_RIGHT_START, ModalityPosition.CAM_FRONT_RIGHT_TOKENS, ModalityPosition.CAM_FRONT_RIGHT_END),
    "CAM_FRONT_LEFT": (ModalityPosition.CAM_FRONT_LEFT_START, ModalityPosition.CAM_FRONT_LEFT_TOKENS, ModalityPosition.CAM_FRONT_LEFT_END),
    "CAM_BACK": (ModalityPosition.CAM_BACK_START, ModalityPosition.CAM_BACK_TOKENS, ModalityPosition.CAM_BACK_END),
    "CAM_BACK_RIGHT": (ModalityPosition.CAM_BACK_RIGHT_START, ModalityPosition.CAM_BACK_RIGHT_TOKENS, ModalityPosition.CAM_BACK_RIGHT_END),
    "CAM_BACK_LEFT": (ModalityPosition.CAM_BACK_LEFT_START, ModalityPosition.CAM_BACK_LEFT_TOKENS, ModalityPosition.CAM_BACK_LEFT_END),
}

# Special tokens for per-view delimiters
VIEW_SPECIAL_TOKENS: Tuple[str, ...] = (
    "<cam_front_start>", "<cam_front_end>",
    "<cam_front_right_start>", "<cam_front_right_end>",
    "<cam_front_left_start>", "<cam_front_left_end>",
    "<cam_back_start>", "<cam_back_end>",
    "<cam_back_right_start>", "<cam_back_right_end>",
    "<cam_back_left_start>", "<cam_back_left_end>",
)

# Mapping from view name to special token pair (start_token, end_token)
VIEW_DELIMITER_TOKENS: Dict[str, Tuple[str, str]] = {
    "CAM_FRONT": ("<cam_front_start>", "<cam_front_end>"),
    "CAM_FRONT_RIGHT": ("<cam_front_right_start>", "<cam_front_right_end>"),
    "CAM_FRONT_LEFT": ("<cam_front_left_start>", "<cam_front_left_end>"),
    "CAM_BACK": ("<cam_back_start>", "<cam_back_end>"),
    "CAM_BACK_RIGHT": ("<cam_back_right_start>", "<cam_back_right_end>"),
    "CAM_BACK_LEFT": ("<cam_back_left_start>", "<cam_back_left_end>"),
}

# ============================================================================
# Vision Pipeline Constants
# ============================================================================
# These constants define the fixed dimensions in the vision encoding pipeline.
# They are derived from the DeepEncoder architecture and should NOT be changed
# unless the underlying model architecture changes.

# DeepEncoder outputs 6x6 grid of tokens per view (for 384x384)
FIXED_GRID_SIDE: int = 6
TOKENS_PER_VIEW: int = FIXED_GRID_SIDE * FIXED_GRID_SIDE  # 36 tokens per camera view

# DeepEncoder projector output dimension (now configurable to match d_model)
# Default is 2048 (CLIP 1024 + SAM 1024) but can be set to d_model directly
# For Qwen2.5-0.5B, d_model is 896
PROJECTOR_DIM: int = 896

# Total vision tokens when all 6 views are used (without delimiters)
TOTAL_VISION_TOKENS: int = NUM_VIEWS * TOKENS_PER_VIEW  # 6 * 36 = 216

# Total vision tokens including per-view delimiters (12 delimiter tokens)
TOTAL_VISION_WITH_DELIMITERS: int = TOTAL_VISION_TOKENS + NUM_VIEWS * 2  # 216 + 12 = 228


# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config(config: Dict, is_main: bool = True) -> None:
    """
    Validate configuration for conflicting or inefficient settings.
    Raises AssertionError for critical conflicts, prints warnings for inefficiencies.
    
    Args:
        config: Configuration dictionary
        is_main: Whether this is the main process (for printing warnings)
    """
    warnings = []
    
    # ===== CRITICAL CONFLICTS (Errors) =====
    
    # Vision must be enabled (this is a vision-only model)
    use_vision = config.get("use_vision", True)
    if not use_vision:
        raise AssertionError(
            "Config conflict: use_vision=False but this is a vision-only model. "
            "use_vision must be True."
        )
    
    # CLIP LoRA enabled but vision disabled - wasteful resource allocation
    clip_lora_enabled = config.get("clip_lora_enabled", False)
    if clip_lora_enabled and not use_vision:
        raise AssertionError(
            "Config conflict: clip_lora_enabled=True but use_vision=False. "
            "CLIP LoRA adapters will be created but never used. "
            "Either enable use_vision or disable clip_lora_enabled."
        )
    
    
    # ===== EFFICIENCY WARNINGS =====
    
    # QLoRA without gradient checkpointing - inefficient memory usage
    use_qlora = config.get("use_qlora", False)
    gradient_checkpointing = config.get("gradient_checkpointing", True)
    if use_qlora and not gradient_checkpointing:
        warnings.append(
            "Efficiency warning: use_qlora=True but gradient_checkpointing=False. "
            "QLoRA is typically used to save memory, but disabling gradient checkpointing "
            "negates much of this benefit. Consider enabling gradient_checkpointing=True."
        )
    
    # Very small effective batch size
    batch_size = config.get("batch_size", 1)
    grad_accum = config.get("grad_accum", 1)
    effective_batch = batch_size * grad_accum
    if effective_batch < 4:
        warnings.append(
            f"Efficiency warning: Very small effective batch size ({batch_size} × {grad_accum} = {effective_batch}). "
            f"Training may be unstable or slow to converge. "
            f"Consider increasing batch_size or grad_accum for effective_batch >= 4."
        )
    
    # num_workers > 0 but prefetch_factor not set optimally
    num_workers = config.get("num_workers", 4)
    prefetch_factor = config.get("prefetch_factor", 2)
    if num_workers > 0 and prefetch_factor < 2:
        warnings.append(
            f"Efficiency warning: num_workers={num_workers} but prefetch_factor={prefetch_factor}. "
            f"Consider prefetch_factor >= 2 for better data loading overlap."
        )
    
    # Mixed precision disabled on modern hardware
    mixed_precision = config.get("mixed_precision", "bf16")
    if mixed_precision == "no":
        warnings.append(
            "Efficiency warning: mixed_precision='no' (disabled). "
            "Training will use full FP32, which is 2x slower and uses 2x memory. "
            "Consider mixed_precision='bf16' for modern GPUs or 'fp16' for older GPUs."
        )
    
    # Print warnings on main process only
    if is_main and warnings:
        print("\n" + "=" * 60)
        print("CONFIG VALIDATION WARNINGS")
        print("=" * 60)
        for i, w in enumerate(warnings, 1):
            print(f"\n[{i}] {w}")
        print("\n" + "=" * 60 + "\n")


__all__ = [
    "DEFAULT_VIEW_ORDER", 
    "CAM_VIEWS", 
    "NUM_VIEWS",
    "VIEW_POSITIONS",
    "VIEW_SPECIAL_TOKENS",
    "VIEW_DELIMITER_TOKENS",
    "FIXED_GRID_SIDE",
    "TOKENS_PER_VIEW",
    "PROJECTOR_DIM",
    "TOTAL_VISION_TOKENS",
    "TOTAL_VISION_WITH_DELIMITERS",
    "ModalityPosition",
    "validate_config",
]
