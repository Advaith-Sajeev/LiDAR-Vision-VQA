"""
Centralized debugging logger for the entire package

This module provides a comprehensive debugging system with toggleable logs.
Enable/disable debugging via:
  - Environment variable: export DEBUG_MODE=1
  - Python code: debug.set_debug_mode(True)
  - Config file: config["debug_mode"] = True

Features:
  - Hierarchical debug levels (INFO, DEBUG, TRACE)
  - Module-specific filtering
  - Data flow tracking
  - Shape/tensor inspection
  - Performance timing
  - Color-coded output
"""

import os
import time
import functools
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime


class DebugLogger:
    """Centralized debug logger with hierarchical levels and module filtering."""
    
    # Debug levels
    DISABLED = 0
    INFO = 1      # High-level flow (e.g., "Starting training", "Loading model")
    DEBUG = 2     # Detailed flow (e.g., "Processing batch", "Forward pass")
    TRACE = 3     # Very detailed (e.g., shapes, tensor stats, timing)
    
    # ANSI color codes for terminal output
    COLORS = {
        'RESET': '\033[0m',
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'CYAN': '\033[96m',
        'WHITE': '\033[97m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
    }
    
    def __init__(self):
        """Initialize debug logger."""
        self._debug_level = self.DISABLED
        self._module_filters = set()  # Empty = all modules
        self._log_file = None
        self._file_handle = None
        self._use_colors = True
        self._timing_stack = []
        self._counters = {}
        
        # Check environment variable
        if os.environ.get('DEBUG_MODE', '0') == '1':
            self._debug_level = self.DEBUG
        if os.environ.get('DEBUG_TRACE', '0') == '1':
            self._debug_level = self.TRACE
    
    def set_debug_mode(self, enabled: bool, level: int = DEBUG):
        """
        Enable/disable debug mode.
        
        Args:
            enabled: Whether to enable debug mode
            level: Debug level (INFO, DEBUG, or TRACE)
        """
        if enabled:
            self._debug_level = level
            self.info("system", f"Debug logging enabled (level={level})")
        else:
            self._debug_level = self.DISABLED
    
    def get_debug_level(self) -> int:
        """Get current debug level."""
        return self._debug_level
    
    def is_enabled(self, level: int = INFO) -> bool:
        """Check if debug logging is enabled at specified level."""
        return self._debug_level >= level
    
    def set_module_filter(self, modules: Union[str, List[str]]):
        """
        Filter debug output to specific modules.
        
        Args:
            modules: Module name(s) to include (e.g., "trainer", ["dataset", "model"])
                    Empty list = show all modules
        """
        if isinstance(modules, str):
            modules = [modules]
        self._module_filters = set(modules)
        if modules:
            self.info("system", f"Debug filter: {', '.join(modules)}")
    
    def set_log_file(self, file_path: Union[str, Path]):
        """
        Enable logging to file.
        
        Args:
            file_path: Path to log file
        """
        if self._file_handle:
            self._file_handle.close()
        
        self._log_file = Path(file_path)
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = open(self._log_file, 'a', encoding='utf-8')
        self.info("system", f"Logging to: {self._log_file}")
    
    def _should_log(self, module: str, level: int) -> bool:
        """Check if message should be logged."""
        if self._debug_level < level:
            return False
        if self._module_filters and module not in self._module_filters:
            return False
        return True
    
    def _format_message(self, module: str, msg: str, level: int, color: str = None) -> str:
        """Format log message with timestamp and module."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Level indicators
        level_str = {
            self.INFO: "INFO ",
            self.DEBUG: "DEBUG",
            self.TRACE: "TRACE",
        }.get(level, "?????")
        
        # Color coding
        if self._use_colors and color:
            c = self.COLORS.get(color, '')
            reset = self.COLORS['RESET']
            dim = self.COLORS['DIM']
            return f"{dim}{timestamp}{reset} {c}[{level_str}]{reset} {c}[{module:12s}]{reset} {msg}"
        else:
            return f"{timestamp} [{level_str}] [{module:12s}] {msg}"
    
    def _log(self, module: str, msg: str, level: int, color: str = None):
        """Internal logging function."""
        if not self._should_log(module, level):
            return
        
        formatted = self._format_message(module, msg, level, color)
        print(formatted)
        
        # Also log to file (without colors)
        if self._file_handle:
            plain = self._format_message(module, msg, level, color=None)
            self._file_handle.write(plain + '\n')
            self._file_handle.flush()
    
    # High-level logging methods
    
    def info(self, module: str, msg: str):
        """Log INFO level message (high-level flow)."""
        self._log(module, msg, self.INFO, color='GREEN')
    
    def debug(self, module: str, msg: str):
        """Log DEBUG level message (detailed flow)."""
        self._log(module, msg, self.DEBUG, color='CYAN')
    
    def trace(self, module: str, msg: str):
        """Log TRACE level message (very detailed)."""
        self._log(module, msg, self.TRACE, color='BLUE')
    
    def warn(self, module: str, msg: str):
        """Log warning (always shown if debug enabled)."""
        if self._debug_level > self.DISABLED:
            self._log(module, f"⚠️  {msg}", self.INFO, color='YELLOW')
    
    def error(self, module: str, msg: str):
        """Log error (always shown if debug enabled)."""
        if self._debug_level > self.DISABLED:
            self._log(module, f"❌ {msg}", self.INFO, color='RED')
    
    # Specialized logging methods
    
    def shape(self, module: str, name: str, tensor: Union[torch.Tensor, np.ndarray, Any]):
        """Log tensor/array shape."""
        if not self._should_log(module, self.TRACE):
            return
        
        if isinstance(tensor, torch.Tensor):
            shape_str = f"{name}: {tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        elif isinstance(tensor, np.ndarray):
            shape_str = f"{name}: {tensor.shape} dtype={tensor.dtype}"
        elif hasattr(tensor, 'shape'):
            shape_str = f"{name}: {tensor.shape}"
        else:
            shape_str = f"{name}: {type(tensor).__name__}"
        
        self.trace(module, f"📐 {shape_str}")
    
    def tensor_stats(self, module: str, name: str, tensor: Union[torch.Tensor, np.ndarray]):
        """Log detailed tensor statistics."""
        if not self._should_log(module, self.TRACE):
            return
        
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            arr = tensor
        else:
            return
        
        stats = (
            f"{name}: shape={arr.shape} "
            f"min={arr.min():.4f} max={arr.max():.4f} "
            f"mean={arr.mean():.4f} std={arr.std():.4f}"
        )
        self.trace(module, f"📊 {stats}")
    
    def param_count(self, module: str, name: str, model: torch.nn.Module):
        """Log model parameter counts."""
        if not self._should_log(module, self.DEBUG):
            return
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        
        self.debug(
            module,
            f"🔢 {name}: total={total:,} trainable={trainable:,} frozen={frozen:,}"
        )
    
    def data_flow(self, module: str, stage: str, data_info: str):
        """Log data flow through pipeline."""
        self.debug(module, f"📦 {stage}: {data_info}")
    
    def memory_usage(self, module: str, label: str = ""):
        """Log GPU memory usage."""
        if not self._should_log(module, self.TRACE):
            return
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            msg = f"💾 {label} GPU memory: allocated={allocated:.2f}GB reserved={reserved:.2f}GB"
            self.trace(module, msg)
    
    # Timing utilities
    
    def start_timer(self, module: str, name: str):
        """Start a named timer."""
        if not self._should_log(module, self.TRACE):
            return
        
        self._timing_stack.append({
            'module': module,
            'name': name,
            'start': time.time()
        })
        self.trace(module, f"⏱️  Start: {name}")
    
    def end_timer(self, module: str, name: str = None):
        """End the most recent timer (or named timer)."""
        if not self._should_log(module, self.TRACE):
            return
        
        if not self._timing_stack:
            return
        
        timer = self._timing_stack.pop()
        elapsed = time.time() - timer['start']
        
        if name and timer['name'] != name:
            self.warn(module, f"Timer mismatch: expected '{name}', got '{timer['name']}'")
        
        self.trace(timer['module'], f"⏱️  End: {timer['name']} ({elapsed*1000:.2f}ms)")
    
    def counter(self, module: str, name: str, value: int = 1):
        """Increment/track a counter."""
        key = f"{module}.{name}"
        self._counters[key] = self._counters.get(key, 0) + value
        
        if self._should_log(module, self.TRACE):
            self.trace(module, f"🔢 {name}={self._counters[key]}")
    
    def get_counter(self, module: str, name: str) -> int:
        """Get counter value."""
        key = f"{module}.{name}"
        return self._counters.get(key, 0)
    
    def reset_counter(self, module: str, name: str = None):
        """Reset counter(s)."""
        if name:
            key = f"{module}.{name}"
            self._counters[key] = 0
        else:
            # Reset all counters for module
            keys_to_reset = [k for k in self._counters if k.startswith(f"{module}.")]
            for k in keys_to_reset:
                self._counters[k] = 0
    
    # Decorator for function tracing
    
    def trace_function(self, module: str):
        """
        Decorator to trace function entry/exit and timing.
        
        Usage:
            @debug.trace_function("module_name")
            def my_function(args):
                ...
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self._should_log(module, self.TRACE):
                    return func(*args, **kwargs)
                
                func_name = func.__name__
                self.trace(module, f"→ Enter: {func_name}")
                self.start_timer(module, func_name)
                
                try:
                    result = func(*args, **kwargs)
                    self.end_timer(module, func_name)
                    self.trace(module, f"← Exit: {func_name}")
                    return result
                except Exception as e:
                    self.error(module, f"Exception in {func_name}: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def section(self, module: str, title: str, level: int = DEBUG):
        """Print a section separator."""
        if not self._should_log(module, level):
            return
        
        sep = "=" * 80
        self._log(module, sep, level, color='BOLD')
        self._log(module, f"  {title}", level, color='BOLD')
        self._log(module, sep, level, color='BOLD')
    
    def close(self):
        """Close log file handle."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


# Global debug logger instance
_debug = DebugLogger()


# Convenience functions for external use
def set_debug_mode(enabled: bool, level: int = DebugLogger.DEBUG):
    """Enable/disable debug mode globally."""
    _debug.set_debug_mode(enabled, level)


def set_debug_level(level: int):
    """Set debug level (DISABLED, INFO, DEBUG, or TRACE)."""
    _debug.set_debug_mode(level > DebugLogger.DISABLED, level)


def set_module_filter(modules: Union[str, List[str]]):
    """Filter debug output to specific modules."""
    _debug.set_module_filter(modules)


def set_log_file(file_path: Union[str, Path]):
    """Enable logging to file."""
    _debug.set_log_file(file_path)


def is_debug_enabled(level: int = DebugLogger.INFO) -> bool:
    """Check if debug logging is enabled."""
    return _debug.is_enabled(level)


def get_debug_level() -> int:
    """Get current debug level."""
    return _debug.get_debug_level()


# Export the global instance for direct use
debug = _debug


# Export constants for convenience
DEBUG_DISABLED = DebugLogger.DISABLED
DEBUG_INFO = DebugLogger.INFO
DEBUG_DEBUG = DebugLogger.DEBUG
DEBUG_TRACE = DebugLogger.TRACE
