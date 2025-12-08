"""Utility functions for training"""

from .distributed import (
    world_info,
    init_dist_if_needed,
    is_main_process,
)
from .logging_utils import Tee
from .helpers import set_seed, count_trainable_params
from .checkpoints import (
    save_state,
    try_load_state,
)
from .plotting import plot_loss_curve, plot_step_curve, plot_metric_curves, plot_all_metrics
from .metrics import (
    calculate_caption_metrics,
    calculate_grounding_metrics,
    calculate_metrics_by_type,
    calculate_sample_level_metrics,
)
from .debug_logger import (
    debug,
    set_debug_mode,
    set_debug_level,
    set_module_filter,
    set_log_file,
    is_debug_enabled,
    get_debug_level,
    DEBUG_DISABLED,
    DEBUG_INFO,
    DEBUG_DEBUG,
    DEBUG_TRACE,
)

__all__ = [
    # Distributed
    "world_info",
    "init_dist_if_needed",
    "is_main_process",
    # Logging
    "Tee",
    # Helpers
    "set_seed",
    "count_trainable_params",
    # Checkpoints
    "save_state",
    "try_load_state",
    # Plotting
    "plot_loss_curve",
    "plot_step_curve",
    "plot_metric_curves",
    "plot_all_metrics",
    # Metrics
    "calculate_caption_metrics",
    "calculate_grounding_metrics",
    "calculate_metrics_by_type",
    "calculate_sample_level_metrics",
    # Debug logging
    "debug",
    "set_debug_mode",
    "set_debug_level",
    "set_module_filter",
    "set_log_file",
    "is_debug_enabled",
    "get_debug_level",
    "DEBUG_DISABLED",
    "DEBUG_INFO",
    "DEBUG_DEBUG",
    "DEBUG_TRACE",
]
