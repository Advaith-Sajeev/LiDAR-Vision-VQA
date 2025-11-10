# Debug Logging Guide

This guide explains how to use the comprehensive debug logging system added to the LiDAR-Vision-VQA package.

## Table of Contents
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Debug Levels](#debug-levels)
- [Module Filtering](#module-filtering)
- [Understanding Output](#understanding-output)
- [Environment Variables](#environment-variables)
- [Examples](#examples)

---

## Quick Start

### Enable Debug Mode in Training

Edit `src/encoder-decoder/train.py` and set:

```python
config = {
    "debug_mode": True,      # Enable debug logging
    "debug_level": 2,        # 1=INFO, 2=DEBUG, 3=TRACE
    "debug_modules": [],     # Empty = all modules, or ["trainer", "dataset"]
    # ... rest of config
}
```

### Run Training with Debug

```bash
# Option 1: Via config file (recommended)
python src/encoder-decoder/train.py  # Uses config["debug_mode"]

# Option 2: Via environment variable
export DEBUG_MODE=1
python src/encoder-decoder/train.py

# Option 3: Trace level (very detailed)
export DEBUG_TRACE=1
python src/encoder-decoder/train.py
```

---

## Configuration

### Config File Options

```python
config = {
    # Enable/disable debug logging
    "debug_mode": True,
    
    # Debug level (higher = more detailed)
    # 0 = DISABLED (no debug output)
    # 1 = INFO     (high-level flow only)
    # 2 = DEBUG    (detailed flow)
    # 3 = TRACE    (very detailed: shapes, stats, timing)
    "debug_level": 2,
    
    # Filter to specific modules (optional)
    # Empty list = show all modules
    # Example: ["trainer", "dataset", "model"]
    "debug_modules": [],
}
```

### Programmatic Control

```python
from training.utils import debug, set_debug_mode, DEBUG_INFO, DEBUG_DEBUG, DEBUG_TRACE

# Enable debug mode
set_debug_mode(True, DEBUG_DEBUG)

# Check if enabled
if debug.is_enabled():
    print("Debug is on!")

# Set debug level
debug.set_debug_mode(True, DEBUG_TRACE)

# Filter to specific modules
debug.set_module_filter(["trainer", "dataset"])

# Enable file logging
debug.set_log_file("logs/debug.log")
```

---

## Debug Levels

### Level 0: DISABLED
No debug output.

### Level 1: INFO
High-level flow tracking:
- Initialization steps
- Epoch progress
- Major milestones
- Warnings and errors

**Example output:**
```
23:15:42.123 [INFO ] [trainer     ] Device: cuda
23:15:42.456 [INFO ] [trainer     ] Initializing models...
23:15:45.789 [INFO ] [dataset     ] Dataset ready: 1000 samples
```

### Level 2: DEBUG
Detailed flow tracking:
- Data processing steps
- Model forward passes
- Component toggles
- Batch information

**Example output:**
```
23:15:48.123 [DEBUG] [trainer     ] Training toggles: vision=True, lidar=True
23:15:48.234 [DEBUG] [trainer     ] Batch size: 2
23:15:48.345 [DEBUG] [trainer     ] 📦 vision_start: Processing 2 samples
23:15:48.456 [DEBUG] [trainer     ] Input sequence order: vision_start → vision_tokens → vision_end → lidar_start → lidar_tokens → lidar_end → text_prompt
```

### Level 3: TRACE
Very detailed tracking:
- Tensor shapes
- Tensor statistics (min, max, mean, std)
- Memory usage
- Timing information
- Function entry/exit

**Example output:**
```
23:15:48.567 [TRACE] [trainer     ] 📐 bev: (2, 256, 64, 64) dtype=torch.float32 device=cuda:0
23:15:48.678 [TRACE] [trainer     ] 📊 prefix_lidar: shape=(2, 6, 512) min=-0.1234 max=0.5678 mean=0.0123 std=0.0987
23:15:48.789 [TRACE] [trainer     ] ⏱️  Start: forward_pass
23:15:49.012 [TRACE] [trainer     ] 💾 before_llm GPU memory: allocated=2.34GB reserved=3.12GB
23:15:49.345 [TRACE] [trainer     ] ⏱️  End: forward_pass (556.00ms)
```

---

## Module Filtering

Filter debug output to specific modules to reduce noise:

```python
# Show only trainer logs
config["debug_modules"] = ["trainer"]

# Show trainer and dataset logs
config["debug_modules"] = ["dataset", "trainer"]

# Show all logs (default)
config["debug_modules"] = []
```

### Available Modules

- `trainer` - Training loop, forward/backward passes
- `dataset` - Dataset loading and preprocessing  
- `model` - Model architecture and operations
- `validation` - Validation and inference
- `system` - System-level operations

---

## Understanding Output

### Output Format

```
TIMESTAMP [LEVEL] [MODULE      ] MESSAGE
```

- **TIMESTAMP**: Current time in HH:MM:SS.mmm format
- **LEVEL**: INFO, DEBUG, or TRACE
- **MODULE**: Component name (12 chars, padded)
- **MESSAGE**: Debug message

### Special Symbols

- 📦 Data flow marker
- 📐 Shape information
- 📊 Tensor statistics
- 🔢 Parameter/counter information
- 💾 Memory usage
- ⏱️  Timing information
- ⚠️  Warning
- ❌ Error

### Color Coding (Terminal)

- **Green**: INFO level
- **Cyan**: DEBUG level
- **Blue**: TRACE level
- **Yellow**: Warnings
- **Red**: Errors

---

## Environment Variables

### Quick Enable/Disable

```bash
# Enable debug mode (DEBUG level)
export DEBUG_MODE=1
python train.py

# Enable trace mode (TRACE level)
export DEBUG_TRACE=1
python train.py

# Disable (default)
unset DEBUG_MODE
unset DEBUG_TRACE
python train.py
```

Environment variables override config file settings.

---

## Examples

### Example 1: Basic Training with Debug

```python
# In train.py
config = {
    "debug_mode": True,
    "debug_level": 2,  # DEBUG level
    # ... rest of config
}

trainer = Trainer(config)
trainer.train()
```

**Output shows:**
- Initialization steps
- Batch processing details
- Forward/backward pass flow
- Loss values
- Data flow through components

### Example 2: Trace Level for Debugging Issues

```python
config = {
    "debug_mode": True,
    "debug_level": 3,  # TRACE level
    "debug_modules": ["trainer"],  # Only trainer
}
```

**Output shows:**
- Everything from DEBUG level
- Tensor shapes at each step
- Tensor statistics
- Memory usage
- Detailed timing

### Example 3: Debug Only Dataset Loading

```python
config = {
    "debug_mode": True,
    "debug_level": 2,
    "debug_modules": ["dataset"],  # Only dataset
}
```

**Output shows:**
- Feature directory scanning
- JSON loading progress
- Sample filtering
- BEV loading (at TRACE level)

### Example 4: Production Training (No Debug)

```python
config = {
    "debug_mode": False,  # Disabled
    # ... rest of config
}
```

**Output:**
- Normal training logs only
- No debug overhead
- Maximum performance

---

## Performance Impact

### Overhead by Level

- **DISABLED**: No overhead
- **INFO**: < 1% overhead
- **DEBUG**: 1-3% overhead
- **TRACE**: 5-10% overhead (includes tensor operations)

### Recommendations

- **Development**: Use DEBUG or TRACE level
- **Testing**: Use DEBUG level
- **Production**: Use DISABLED or INFO level
- **Debugging Issues**: Use TRACE level with module filtering

---

## Log Files

### Enable File Logging

Debug logs are automatically saved to `<out_dir>/debug.log` when `debug_mode=True`.

You can also manually specify a log file:

```python
from training.utils import set_log_file

set_log_file("my_custom_debug.log")
```

### Log File Format

Log files contain the same information as terminal output but without color codes.

---

## Data Flow Tracking

The debug system tracks data flow through the entire pipeline:

### Training Flow

```
1. Dataset Loading
   └─ Feature indexing
   └─ JSON loading
   └─ Sample filtering

2. Batch Processing
   └─ BEV loading
   └─ Vision processing (if enabled)
      └─ Multiview token extraction
      └─ Vision adapter
   └─ Embedding assembly
      └─ Vision tokens (if enabled)
      └─ LiDAR VAT
      └─ Vision VAT (if enabled)
      └─ Text embeddings

3. Forward Pass
   └─ LLM forward
   └─ Loss computation

4. Backward Pass
   └─ Gradient computation

5. Optimizer Step
   └─ Gradient clipping
   └─ Weight update
```

### Validation Flow

```
1. Validation Loop
   └─ Similar to training but without backward pass

2. Inference Sampling
   └─ Sample selection
   └─ Generation
   └─ Metric calculation
```

---

## Troubleshooting

### No Debug Output

**Problem**: Debug mode is enabled but no output appears.

**Solutions**:
1. Check `debug_mode` is `True` in config
2. Check `debug_level` is > 0
3. If using module filter, ensure module name is correct
4. Check environment variables aren't overriding config

### Too Much Output

**Problem**: Debug output is overwhelming.

**Solutions**:
1. Lower debug level (3 → 2 → 1)
2. Use module filtering
3. Redirect to file: `python train.py > debug_output.txt 2>&1`

### Performance Issues

**Problem**: Training is slower with debug enabled.

**Solutions**:
1. Lower debug level (TRACE → DEBUG → INFO)
2. Use module filtering
3. Disable for production runs

---

## Advanced Usage

### Custom Debug Messages

Add debug logging to your own code:

```python
from training.utils import debug

# Info message
debug.info("my_module", "Starting processing")

# Debug message  
debug.debug("my_module", "Processing batch 5")

# Trace message
debug.trace("my_module", "Detailed information")

# Warning
debug.warn("my_module", "Something unusual happened")

# Error
debug.error("my_module", "Something went wrong")

# Shape information
debug.shape("my_module", "my_tensor", tensor)

# Tensor statistics
debug.tensor_stats("my_module", "my_tensor", tensor)

# Data flow
debug.data_flow("my_module", "stage_name", "description")

# Timing
debug.start_timer("my_module", "operation")
# ... do work ...
debug.end_timer("my_module", "operation")
```

### Function Tracing Decorator

```python
from training.utils import debug

@debug.trace_function("my_module")
def my_function(args):
    # Function automatically traced
    pass
```

---

## Summary

The debug logging system provides comprehensive visibility into the training pipeline with minimal code changes. Key features:

✅ **Toggleable** - Enable/disable via config or environment  
✅ **Hierarchical** - INFO → DEBUG → TRACE levels  
✅ **Filtered** - Show only specific modules  
✅ **Detailed** - Shapes, stats, timing, memory  
✅ **Color-coded** - Easy to read terminal output  
✅ **File logging** - Saved to disk automatically  
✅ **Low overhead** - < 5% even at DEBUG level

For questions or issues, check the code in:
- `src/encoder-decoder/training/utils/debug_logger.py`
