"""
Centralized configuration module for LiDAR-Vision-VQA.

This module provides:
- DEFAULT_VIEW_ORDER: Standard camera view ordering for nuScenes
- CAM_VIEWS, NUM_VIEWS: Aliases for view configuration
- DEFAULT_CONFIG: Fallback configuration values
- ModalityPosition: Enum for multimodal sequence position ordering
- get_training_config: Local training configuration template
- get_modal_training_config: Modal cloud training configuration
"""

from .default_config import DEFAULT_CONFIG, DEFAULT_VIEW_ORDER, CAM_VIEWS, NUM_VIEWS, ModalityPosition
from .training_config import get_training_config
from .modal_config import get_modal_training_config

__all__ = [
    "DEFAULT_VIEW_ORDER",
    "CAM_VIEWS", 
    "NUM_VIEWS",
    "DEFAULT_CONFIG",
    "ModalityPosition",
    "get_training_config",
    "get_modal_training_config",
]
