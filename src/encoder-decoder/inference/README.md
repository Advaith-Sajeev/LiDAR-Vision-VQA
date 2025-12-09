# Inference Package

This package provides inference capabilities for the trained LiDAR-Vision-LLM model.

## Components

### 1. `model_loader.py`
Handles loading of trained model checkpoints:
- Base LLM with LoRA adapters
- LiDAR VAT (View-Aware Transformer)
- Vision VAT and Vision Adapter (if enabled)
- DeepEncoder runtime (if vision is enabled)

### 2. `inference_engine.py`
High-level inference engine that:
- Formats prompts with special tokens
- Processes LiDAR BEV features
- Processes multi-view camera images (if enabled)
- Generates text responses

### 3. `utils.py`
Helper functions for:
- Loading BEV features
- Formatting prompts
- Loading/saving QA pairs
- Calculating metrics
- Formatting output

## Usage

### Basic Usage

```python
from inference import ModelLoader, InferenceEngine

# Load models
loader = ModelLoader("checkpoints_vat", device="cuda")
models = loader.load_all(c_in=256)

# Create engine
engine = InferenceEngine(models)

# Run inference
answer = engine.generate(
    question="What objects are in front of the vehicle?",
    bev="path/to/bev_feature.npy",
    sample_token="sample_token_123",  # Optional, for vision
    max_new_tokens=64,
    temperature=0.7
)

print(answer)
```

### Batch Inference

```python
questions = ["What is ahead?", "Is the road clear?"]
bevs = ["bev1.npy", "bev2.npy"]
sample_tokens = ["token1", "token2"]

answers = engine.generate_batch(
    questions=questions,
    bevs=bevs,
    sample_tokens=sample_tokens,
    max_new_tokens=64
)
```

### Using the CLI Script

See `infer.py` for a complete command-line interface.

#### Saving per-sample artifacts

Add `--artifact-dir path/to/archive` (and optionally `--ground-truth "..."` in interactive mode) to copy the BEV `.npy`, the six nuScenes camera images, and a `metadata.json` file containing the question, ground truth, and prediction into numbered subfolders for each inference run.

#### Random split sampling

Combine `--sample-random N` with `--json ... --feature-dirs ...` to run sequential inference on `N` random samples drawn from the same train/val split definition used in training (seed + `val_split` read from the checkpoint config). Use `--split val` to target the validation shard; otherwise it defaults to the training shard.

## Generation Parameters

- `max_new_tokens`: Maximum tokens to generate (default: 64)
- `temperature`: Sampling temperature, higher = more random (default: 0.7)
- `top_p`: Nucleus sampling threshold (default: 0.9)
- `top_k`: Top-k sampling threshold (default: 50)
- `do_sample`: Enable sampling vs greedy decoding (default: True)
- `num_beams`: Number of beams for beam search (default: 1)

## Special Tokens

The model uses special tokens to mark modal boundaries:
- `<vision_start>` / `<vision_end>`: Vision features
- `<lidar_start>` / `<lidar_end>`: LiDAR features

These are automatically inserted by the inference engine.

## Requirements

- Trained checkpoint directory with:
  - `config.json`: Training configuration
  - `qwen2_lora_adapter_latest/`: LLM LoRA adapter directory (or legacy `lora.pt`)
  - `vat_lidar_latest.pt`: LiDAR VAT weights
  - `vat_vision_latest.pt`: Vision VAT weights (if vision enabled)
  - `vision_adapter_latest.pt`: Vision adapter weights (if vision enabled)
  - `clip_lora_adapter_latest/`: CLIP LoRA adapter directory (if vision enabled)
  - `projector_latest.pt`: DeepEncoder projector weights (if vision enabled)
- `sam_compression_head_latest.pt`: Trainable SAM compression head weights (if vision enabled)
