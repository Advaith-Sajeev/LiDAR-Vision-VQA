"""
Centralized configuration module for LiDAR-Vision-VQA.

This module provides shared constants used across all training configurations.

Constants:
- DEFAULT_VIEW_ORDER: Standard camera view ordering for nuScenes
- CAM_VIEWS, NUM_VIEWS: Aliases for view configuration
- ModalityPosition: Enum for multimodal sequence position ordering
- VIEW_POSITIONS, VIEW_SPECIAL_TOKENS, VIEW_DELIMITER_TOKENS: Per-view delimiter config
- Vision pipeline constants: FIXED_GRID_SIDE, TOKENS_PER_VIEW, etc.
- validate_config: Configuration validation function

Config files are now located separately:
- Modal cloud training: src/modal-trainer/modal_config.py
- Local training: src/encoder-decoder/local_config.py
"""

from .constants import (
    DEFAULT_VIEW_ORDER,
    CAM_VIEWS,
    NUM_VIEWS,
    ModalityPosition,
    VIEW_POSITIONS,
    VIEW_SPECIAL_TOKENS,
    VIEW_DELIMITER_TOKENS,
    FIXED_GRID_SIDE,
    TOKENS_PER_VIEW,
    PROJECTOR_DIM,
    TOTAL_VISION_TOKENS,
    TOTAL_VISION_WITH_DELIMITERS,
    validate_config,
)

__all__ = [
    "DEFAULT_VIEW_ORDER",
    "CAM_VIEWS",
    "NUM_VIEWS",
    "ModalityPosition",
    "VIEW_POSITIONS",
    "VIEW_SPECIAL_TOKENS",
    "VIEW_DELIMITER_TOKENS",
    "FIXED_GRID_SIDE",
    "TOKENS_PER_VIEW",
    "PROJECTOR_DIM",
    "TOTAL_VISION_TOKENS",
    "TOTAL_VISION_WITH_DELIMITERS",
    "validate_config",
]
