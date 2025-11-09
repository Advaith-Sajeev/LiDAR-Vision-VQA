# LiDAR-Vision-LLM Training Package

Modular training package for LiDAR and Vision-enhanced Large Language Models.

## Package Structure

```
training/
├── __init__.py                 # Package initialization
├── config/                     # Configuration management
│   ├── __init__.py
│   └── default_config.py      # Default training configuration
├── data/                       # Data loading and processing
│   ├── __init__.py
│   ├── dataset.py             # MixedNuDataset class
│   ├── collate.py             # Batch collation
│   ├── sampler.py             # Custom samplers
│   └── utils.py               # Data utility functions
├── models/                     # Model architectures
│   ├── __init__.py
│   ├── vat_blocks.py          # VAT transformer blocks
│   ├── vat_lidar.py           # LiDAR VAT model
│   ├── vat_vision.py          # Vision VAT model
│   ├── vision_adapter.py      # Vision token adapter
│   └── lora_utils.py          # LoRA utilities
├── utils/                      # Training utilities
│   ├── __init__.py
│   ├── distributed.py         # Distributed training helpers
│   ├── logging.py             # Logging utilities
│   ├── helpers.py             # General helper functions
│   ├── checkpoints.py         # Checkpoint management
│   └── plotting.py            # Training visualization
└── core/                       # Core training logic
    ├── __init__.py
    ├── trainer.py             # Main Trainer class
    ├── validation.py          # Validation and inference
    └── model_setup.py         # Model initialization
```

## Quick Start

### Basic Usage

```python
from training.config import DEFAULT_CONFIG
from training.core import Trainer

# Use default configuration
config = DEFAULT_CONFIG.copy()
trainer = Trainer(config)
trainer.train()
```

### Custom Configuration

```python
from training.config import DEFAULT_CONFIG
from training.core import Trainer

# Customize configuration
config = DEFAULT_CONFIG.copy()
config["epochs"] = 20
config["batch_size"] = 2
config["lr_vat"] = 1e-4
config["use_vision"] = True

trainer = Trainer(config)
trainer.train()
```

### Using train.py Script

The simplest way to start training:

```bash
python train.py
```

Edit `train.py` to customize configuration before running.

## Configuration Options

### I/O Settings
- `feature_dirs`: List of directories containing BEV feature .npy files
- `jsons`: List of JSON/JSONL files with QA pairs
- `out_dir`: Output directory for checkpoints
- `max_samples`: Maximum samples to use (None for all)

### Training Settings
- `epochs`: Number of training epochs
- `batch_size`: Batch size per GPU
- `grad_accum`: Gradient accumulation steps
- `seed`: Random seed
- `fp16`: Enable mixed precision training
- `resume`: Resume from checkpoint if available

### Model Settings
- `model_id`: Hugging Face model ID (e.g., "Qwen/Qwen2.5-0.5B")
- `vat_queries`: Number of LiDAR VAT queries (must be divisible by 6)
- `vat_layers`: Number of LiDAR VAT layers
- `vision_queries`: Number of vision VAT queries (must be divisible by 6)
- `vision_layers`: Number of vision VAT layers
- `use_vision`: Enable vision pipeline

### Optimization Settings
- `lr_vat`: Learning rate for LiDAR VAT
- `lr_vision_vat`: Learning rate for Vision VAT
- `lr_lora`: Learning rate for LoRA adapters
- `lr_vision`: Learning rate for vision components
- `weight_decay`: Weight decay
- `warmup_steps`: Warmup steps for scheduler
- `clip_norm`: Gradient clipping norm

## Components

### Trainer

Main training orchestrator that handles:
- Model initialization and setup
- Data loading and batching
- Training loop with validation
- Checkpoint management
- Distributed training (DDP)

```python
from training.core import Trainer
from training.config import DEFAULT_CONFIG

trainer = Trainer(DEFAULT_CONFIG)
trainer.train()
```

### Models

#### VATLiDAR
View-Aware Transformer for BEV LiDAR features.

```python
from training.models import VATLiDAR

vat = VATLiDAR(
    c_in=256,           # Input channels
    d_model=896,        # Model dimension
    n_queries=576,      # Number of output queries
    n_layers=4,         # Transformer layers
    n_heads=8,          # Attention heads
)
```

#### VATVision
View-Aware Transformer for multi-view camera features.

```python
from training.models import VATVision

vat = VATVision(
    d_model=896,
    n_queries=1536,
    n_layers=4,
    n_heads=8,
)
```

### Data

#### MixedNuDataset
Dataset for nuScenes with BEV features and QA pairs.

```python
from training.data import MixedNuDataset

dataset = MixedNuDataset(
    json_paths=["data/nuCaption.json"],
    feature_dirs=["bev_feats/train"],
    target_field="answer",
)
```

### Utilities

#### Checkpointing

```python
from training.utils import save_state, try_load_state

# Save checkpoint
save_state(out_dir, tag="latest", ...)

# Load checkpoint
state, tag = try_load_state(out_dir)
```

#### Plotting

```python
from training.utils import plot_loss_curve

plot_loss_curve(train_losses, val_losses, val_epochs, out_dir)
```

## Advanced Usage

### Custom Training Loop

For more control, you can build your own training loop:

```python
from training.config import DEFAULT_CONFIG
from training.data import MixedNuDataset, make_collate
from training.models import VATLiDAR, VATVision
from training.core import setup_models

config = DEFAULT_CONFIG
device = torch.device("cuda")

# Setup models
tok, base, _, vat_vision, vision_adapter, runtime, nusc, d_model, _ = setup_models(
    config, device, True
)

# Create dataset
dataset = MixedNuDataset(
    config["jsons"],
    config["feature_dirs"],
    target_field=config["target_field"],
)

# Your custom training logic here...
```

### Extending the Package

Add custom models to `training/models/`:

```python
# training/models/my_model.py
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your model definition
        
    def forward(self, x):
        # Your forward pass
        return x
```

Update `training/models/__init__.py`:

```python
from .my_model import MyCustomModel

__all__ = [..., "MyCustomModel"]
```

## Distributed Training

### Single-GPU Training

```bash
python train.py
```

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=4 train.py
```

The trainer automatically detects and configures distributed training.

## Checkpoints

Checkpoints are saved to `config["out_dir"]` with the following structure:

```
checkpoints_vat/
├── vat_lidar_latest.pt              # Latest LiDAR VAT weights
├── vat_vision_latest.pt             # Latest Vision VAT weights
├── vision_adapter_latest.pt         # Latest Vision Adapter weights
├── projector_latest.pt              # Latest projector weights
├── qwen2_lora_adapter_latest/       # Latest LoRA adapters
├── clip_lora_adapter_latest/        # Latest CLIP LoRA
├── training_state_latest.pt         # Latest training state
├── vat_lidar_best.pt                # Best validation weights
├── loss_curve.png                   # Training curves
└── train.log                        # Training log
```

## Validation

Validation runs every `config["validate_every"]` epochs:
- Computes validation loss
- Saves best model based on validation performance
- Generates inference samples every 5 epochs

## Tips

1. **Memory Management**: Reduce `batch_size` or `vat_queries`/`vision_queries` if OOM
2. **Faster Training**: Increase `grad_accum` and `batch_size` × `grad_accum`
3. **Better Convergence**: Tune `warmup_steps` and learning rates
4. **Debugging**: Set `debug_shapes=True` to print tensor shapes

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `vat_queries` or `vision_queries`
- Enable `fp16=True`

### Slow Training
- Increase `num_workers` in DataLoader (edit source)
- Use larger `batch_size` with gradient accumulation
- Check data loading is not the bottleneck

### Poor Performance
- Increase model capacity (`vat_layers`, `vision_layers`)
- Tune learning rates
- Increase training `epochs`
- Check data quality and preprocessing

## License

See main repository LICENSE file.
