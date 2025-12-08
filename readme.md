# 🚗 LiDAR-Vision-VQA: Multimodal Visual Question Answering for Autonomous Driving

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

> A state-of-the-art multimodal Visual Question Answering system that combines LiDAR Bird's Eye View (BEV) features and multi-camera vision for autonomous driving scene understanding.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [The Broader Vision](#-the-broader-vision)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Inference](#-inference)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Overview

**LiDAR-Vision-VQA** is an advanced multimodal Visual Question Answering system specifically designed for autonomous driving scenarios. It seamlessly integrates:

- **🔷 LiDAR BEV Features**: 3D spatial understanding through Bird's Eye View representations
- **📷 Multi-Camera Vision**: 6-camera surround-view perception using CLIP and SAM encoders
- **🤖 Large Language Model**: Natural language understanding and generation

The system can answer complex questions about driving scenes, such as:
- *"What objects are in front of the vehicle?"*
- *"Describe the current driving scene."*
- *"Is it safe to change lanes?"*

### Why LiDAR-Vision-VQA?

Traditional VQA systems often rely solely on camera images, missing crucial 3D spatial information. By fusing LiDAR BEV features with multi-view camera data, our system achieves:
- **Superior 3D spatial reasoning** through BEV representations
- **Robust scene understanding** across different lighting and weather conditions
- **Accurate object localization** combining vision and depth information

---

## 🌟 The Broader Vision

This project represents a **foundational step** in our broader research vision for developing next-generation autonomous driving systems with natural language interaction capabilities.

### Long-term Research Goals

**LiDAR-Vision-VQA** is the first milestone in a multi-phase research agenda:

#### Phase 1: Multimodal Scene Understanding (Current)
- ✅ Establishing robust fusion of LiDAR BEV and multi-camera vision
- ✅ Training View-Aware Transformers (VAT) for efficient multimodal compression
- ✅ Validating caption generation and scene description capabilities
- 🔄 Extending to object grounding and spatial reasoning tasks

#### Phase 2: Temporal Reasoning & Trajectory Prediction (Planned)
- 📋 Incorporating temporal sequences for motion understanding
- 📋 Predicting future trajectories of surrounding vehicles and pedestrians
- 📋 Reasoning about dynamic scene changes and their implications for driving decisions

#### Phase 3: Interactive Planning & Decision Making (Future Vision)
- 📋 Natural language-driven route planning and navigation
- 📋 Explaining autonomous driving decisions in human-understandable terms
- 📋 Interactive Q&A for real-time driving assistance and debugging

#### Phase 4: End-to-End Autonomous Driving System
- 📋 Integration with planning and control modules
- 📋 Real-time deployment on autonomous vehicle platforms
- 📋 Human-AI collaborative driving with natural language interfaces

### Why This Matters

The ability to **understand** and **communicate** about driving scenes in natural language is crucial for:
- **Trust & Transparency**: Passengers and operators can query the system about its perception and reasoning
- **Debugging & Development**: Engineers can quickly diagnose perception failures through conversational interfaces
- **Accessibility**: Making autonomous driving systems more interpretable to non-technical users
- **Safety**: Enabling better human-AI collaboration in complex or ambiguous driving scenarios

This project lays the groundwork by solving the fundamental challenge of multimodal scene understanding, which is essential for all downstream tasks in the broader vision.

---

## ✨ Key Features

### 🏗️ Advanced Architecture
- **View-Aware Transformers (VAT)**: Dedicated attention mechanisms for both vision and LiDAR modalities
- **Dual Vision Encoders**: CLIP ViT-L/14 + SAM ViT-B for comprehensive visual understanding
- **Efficient Training**: LoRA/QLoRA support for parameter-efficient fine-tuning
- **Flexible Modality Selection**: Train with vision-only, LiDAR-only, or both modalities

### 🚀 Production-Ready
- **Cloud Training**: Native Modal.com integration for scalable, long-running experiments
- **Smart Checkpointing**: Automatic resume from latest checkpoint with metadata tracking
- **Comprehensive Logging**: Debug modes, metrics dashboards, and training visualization
- **Distributed Training**: Multi-GPU support with DeepSpeed integration

### 📊 Evaluation & Metrics
- **Caption Metrics**: BLEU-4, CIDEr, BERTScore for scene description quality
- **Grounding Support**: Object detection and localization evaluation (nuGrounding dataset)
- **Sample-Level Analysis**: Per-sample metric tracking and visualization
- **Inference Visualization**: Automated generation of prediction visualizations

---

## 🏛️ Architecture

The system consists of three main components:

### 1. **LiDAR Encoder** (`src/lidar-encoder/`)
- Built on PCDet framework for 3D object detection
- Extracts BEV feature maps from LiDAR point clouds
- Outputs: `[B, C, H, W]` BEV tensors (e.g., 256×256×256)

### 2. **Vision Encoder** (`src/deepencoder/`)
- **CLIP ViT-L/14**: Semantic understanding (1024-dim per token)
- **SAM ViT-B**: Spatial and structural features (1024-dim per token)
- **Projector**: Fuses features to 2048-dim per-token representation
- Processes 6 camera views: Front, Front-Left, Front-Right, Back, Back-Left, Back-Right
- Outputs: `[6 views × 256 tokens, 2048-dim]` = 1536 tokens per sample

### 3. **Encoder-Decoder** (`src/encoder-decoder/`)
The core multimodal fusion and language generation module:

#### **VATVision** (Vision VAT)
- **Input**: 1536 vision tokens `[1536, 2048]` from 6 camera views
- **Processing**:
  - Spatial positional encodings (continuous grid coordinates)
  - View embeddings (discrete camera identifiers)
  - Learned query tokens with cross-attention
- **Output**: Compact vision representation `[n_queries, d_model]`

#### **VATLiDAR** (LiDAR VAT)
- **Input**: BEV feature maps `[B, C, H, W]`
- **Processing**:
  - Geometric positional encodings (x, y, r, sin θ, cos θ)
  - 6-sector view embeddings (spatial partitioning)
  - View-aligned query tokens with cross-attention
- **Output**: Compact BEV representation `[n_queries, d_model]`

#### **Sequence Construction**
```
[<vision_start>] [vision_queries] [<vision_end>]
[<lidar_start>]  [lidar_queries]  [<lidar_end>]
[text_prompt_tokens]
[answer_tokens]  # Training only
```

#### **Language Model Integration**
- Base LLM: Qwen2.5 (0.5B/1.5B/3B variants supported, 3B recommended for Modal)
- LoRA adaptation for efficient fine-tuning
- Flash Attention 2 for faster training
- Quantization support (4-bit, 8-bit)

### Architecture Diagram

For detailed architecture diagrams, see the `arch/` folder in this repository.

**High-Level Overview:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INPUT MODALITIES                                │
├─────────────────────────────────────────────────────────────────────────┤
│  📷 6 Camera Views                    🔷 LiDAR Point Cloud              │
│  (1024×1024 RGB images)                (Bird's Eye View)                │
└──────────┬───────────────────────────────────────┬──────────────────────┘
           │                                       │
           ▼                                       ▼
    ┌──────────────┐                      ┌──────────────┐
    │ CLIP ViT-L/14│                      │  BEV Feature │
    │  + SAM ViT-B │                      │  Extraction  │
    │   Encoders   │                      │   (PCDet)    │
    └──────┬───────┘                      └──────┬───────┘
           │                                     │
           │ [1536, 2048]                        │ [B, 256, H, W]
           ▼                                     ▼
    ┌──────────────┐                      ┌──────────────┐
    │  VATVision   │                      │  VATLiDAR    │
    │  (View-Aware │                      │  (View-Aware │
    │  Transformer)│                      │  Transformer)│
    └──────┬───────┘                      └──────┬───────┘
           │                                     │
           │ [n_v, d_model]                      │ [n_l, d_model]
           └────────────┬────────────────────────┘
                        ▼
              ┌─────────────────┐
              │  Sequence       │
              │  Builder        │
              └────────┬────────┘
                       │
                       │ [seq_len, d_model]
                       ▼
              ┌─────────────────┐
              │   Qwen2.5 LLM   │
              │   + LoRA Fine-  │
              │     tuning      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Generated     │
              │   Answer Text   │
              └─────────────────┘
```

---

## 🔧 Installation

### Prerequisites
- **Python**: 3.11
- **CUDA**: 12.0+ (12.6 recommended)
- **GPU**: NVIDIA GPU with 24GB+ VRAM (e.g., RTX 4090, A100, H100, L4)
- **Storage**: ~100GB for datasets and models

### Step 1: Create Conda Environment
```bash
conda create --name lidar-vqa python=3.11 -y
conda activate lidar-vqa
```

### Step 2: Install PyTorch
```bash
# For CUDA 12.6
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu120
```

### Step 3: Install LiDAR Encoder (PCDet)
```bash
cd src/lidar-encoder/

# Install dependencies
pip install llvmlite numba tensorboardX easydict pyyaml scikit-image tqdm SharedArray opencv-python pyquaternion

# Install spconv (adjust for your CUDA version)
pip install spconv-cu120  # or spconv-cu126

# Install PCDet in editable mode
pip install -e . --no-build-isolation

# Verify installation
python tools/verify_dependency.py

cd ../..
```

### Step 4: Install Core Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Download NLTK Data (for evaluation)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### Step 6: Install Flash Attention (Optional but Recommended)
```bash
# Must be installed after all other dependencies
pip install flash-attn --no-build-isolation
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from pcdet.version import __version__; print(f'PCDet: {__version__}')"
```

---

## 📂 Dataset Preparation

This project uses the **nuScenes** dataset with additional annotations from **nuCaption** and **nuGrounding**.

### Step 1: Download nuScenes Dataset
1. Register at [nuScenes.org](https://www.nuscenes.org/)
2. Download the following:
   - **Full dataset (v1.0)**: ~350GB
     - Trainval (v1.0-trainval)
     - Test (v1.0-test)
   - **Mini dataset**: ~4GB (for quick testing)

3. Organize directory structure:
```
/data/Datasets/nuScenes/
├── maps/
├── samples/           # Images from 6 cameras
├── sweeps/           # Intermediate frames
├── v1.0-trainval/    # Metadata
├── v1.0-test/
└── external/
    ├── nuCaption.json      # Scene description annotations
    └── nuGrounding.json    # Object grounding annotations
```

### Step 2: Download Annotations
```bash
# Download nuCaption and nuGrounding datasets
# Place JSON files in /data/Datasets/nuScenes/external/
```

### Step 3: Precompute BEV Features
```bash
cd src/get-data/

# Configure paths in precompute_bev_features.py
python precompute_bev_features.py \
    --nuscenes-root /data/Datasets/nuScenes \
    --output-dir /data/bev_feats \
    --version v1.0-trainval
```

This generates `.npy` files containing BEV features for each sample:
```
/data/bev_feats/
├── train/
│   ├── sample_token_1.npy
│   ├── sample_token_2.npy
│   └── ...
└── val/
    └── ...
```

### Step 4: (Optional) Create Dataset Subset
For faster experimentation:
```bash
python create_nuScenes_subset.py \
    --input /data/Datasets/nuScenes/external/nuCaption.json \
    --output /data/Datasets/nuScenes/external/nuCaption_mini.json \
    --num-samples 1000
```

---

## 🎓 Training

### Local Training

#### Basic Training
```bash
cd src/encoder-decoder/

python train.py
```

#### Configuration
Edit `train.py` to customize training configuration:
```python
config = {
    # Dataset
    "dataset_mode": "caption",  # "caption", "grounding", or "both"
    "caption_json": "/data/Datasets/nuScenes/external/nuCaption.json",
    "feature_dirs": ["/data/bev_feats/train"],
    "max_samples": None,  # None for full dataset
    
    # Model
    "model_id": "Qwen/Qwen2.5-0.5B",  # Default for local training
    # Other options: "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B" (3B recommended for Modal)
    "use_lora": True,
    "lora_r": 64,
    "lora_alpha": 128,
    
    # Training
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "num_epochs": 5,
    "warmup_steps": 500,
    
    # Hardware
    "use_flash_attn": True,
    "load_in_4bit": False,
    
    # Output
    "out_dir": "./checkpoints",
    "save_interval": 1000,
}
```

#### Resume Training
```bash
# Automatic resume from latest checkpoint
python train.py  # Will detect and prompt for available checkpoints
```

### Cloud Training (Modal)

For week-long training with auto-resume on Modal.com:

#### Setup Modal
```bash
pip install modal
modal setup  # Follow authentication steps
```

#### Create Modal Volume
```bash
# Create persistent volume for checkpoints
modal volume create lidar-llm

# Upload dataset (if not using Modal's dataset feature)
modal volume put lidar-llm /local/path/to/data /data
```

#### Configure Training
Edit `src/configs/modal_config.py`:
```python
config = {
    "dataset_mode": "caption",
    "caption_json": "/data/Datasets/nuScenes/external/nuCaption.json",
    "feature_dirs": ["/data/bev_feats"],
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "num_epochs": 10,
    # ... other settings
}
```

#### Start Training
```bash
cd src/modal-trainer/

# Run in foreground (monitor logs in real-time)
modal run modal-train.py

# Run in background (detached)
modal run --detach modal-train.py
```

#### Monitor Training
```bash
# View logs
modal app logs lidar-vision-training

# Stop training
modal app stop lidar-vision-training

# List checkpoints
modal volume ls lidar-llm /checkpoints
```

#### Features of Modal Training
- ✅ **Auto-Resume**: Automatically resumes from latest checkpoint after 24h timeout
- ✅ **Persistent Storage**: Checkpoints saved to Modal Volume
- ✅ **GPU Selection**: Configurable (L4, A100, H100)
- ✅ **Scalable**: Easy to scale to multiple GPUs
- ✅ **Cost-Effective**: Pay only for compute time used

---

## 🔮 Inference

### Interactive Inference
```bash
cd src/encoder-decoder/

# With LiDAR only
python infer.py \
    --checkpoint ./checkpoints/run_20241208_031041 \
    --bev /data/bev_feats/train/sample_token.npy \
    --question "What is ahead of the vehicle?"

# With vision + LiDAR
python infer.py \
    --checkpoint ./checkpoints/run_20241208_031041 \
    --sample-token abc123 \
    --bev /data/bev_feats/train/abc123.npy \
    --question "Describe the driving scene."
```

### Batch Inference
```bash
python infer.py \
    --checkpoint ./checkpoints/run_20241208_031041 \
    --json /data/Datasets/nuScenes/external/nuCaption.json \
    --feature-dirs /data/bev_feats/train \
    --output predictions.json \
    --num-samples 100
```

### Modal Inference
```bash
cd src/modal-trainer/

modal run modal-infer.py
```

### Generation Parameters
```bash
# Greedy decoding (deterministic)
python infer.py ... --no-sample --temperature 0.0

# Sampling with temperature
python infer.py ... --temperature 0.7 --top-p 0.9 --top-k 50

# Beam search
python infer.py ... --num-beams 5 --no-sample

# Control output length
python infer.py ... --max-tokens 256
```

### Visualization
After running inference, generate visualizations:
```bash
cd modal_inference_20251208_031041/20251208_031041/

python create_visualizations.py
```

This creates visualizations in `viz/` directory showing:
- Input camera images
- Ground truth captions
- Model predictions
- Comparison metrics

---

## 📁 Project Structure

```
LiDAR-Vision-VQA/
├── readme.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── src/                                # Source code
│   ├── configs/                        # Configuration files
│   │   ├── default_config.py          # Default training config
│   │   ├── modal_config.py            # Modal cloud training config
│   │   └── training_config.py         # Additional training configs
│   │
│   ├── lidar-encoder/                  # LiDAR BEV feature extraction
│   │   ├── pcdet/                     # PCDet framework (3D detection)
│   │   ├── tools/                     # Utility scripts
│   │   ├── readme.md                  # LiDAR encoder documentation
│   │   └── setup.py                   # Installation script
│   │
│   ├── deepencoder/                    # Vision feature extraction
│   │   ├── clip_sdpa.py               # CLIP ViT-L/14 encoder
│   │   ├── sam_vary_sdpa.py           # SAM ViT-B encoder
│   │   ├── deepencoder_infer.py       # Inference wrapper
│   │   └── build_linear.py            # Projection layers
│   │
│   ├── encoder-decoder/                # Main training & inference
│   │   ├── train.py                   # Local training script
│   │   ├── infer.py                   # Local inference script
│   │   ├── training/                  # Training modules
│   │   │   ├── core/                  # Core training logic
│   │   │   │   └── trainer.py         # Main Trainer class
│   │   │   ├── models/                # Model architectures
│   │   │   │   ├── vat_vision.py      # Vision VAT
│   │   │   │   ├── vat_lidar.py       # LiDAR VAT
│   │   │   │   ├── vat_blocks.py      # Shared VAT components
│   │   │   │   ├── vision_adapter.py  # Vision preprocessing
│   │   │   │   └── lora_utils.py      # LoRA utilities
│   │   │   ├── data/                  # Dataset and dataloading
│   │   │   │   ├── dataset.py         # Main dataset class
│   │   │   │   ├── collate.py         # Batch collation
│   │   │   │   ├── sampler.py         # Data sampling
│   │   │   │   └── utils.py           # Data utilities
│   │   │   └── utils/                 # Training utilities
│   │   │       ├── metrics.py         # Evaluation metrics
│   │   │       ├── checkpoints.py     # Checkpoint management
│   │   │       ├── logging.py         # Logging utilities
│   │   │       └── ...
│   │   ├── inference/                 # Inference modules
│   │   │   ├── engine.py              # Inference engine
│   │   │   ├── loader.py              # Model loader
│   │   │   └── utils.py               # Inference utilities
│   │   └── training-test/             # Unit tests
│   │       ├── models/                # Model tests
│   │       ├── data/                  # Data tests
│   │       └── utils/                 # Utility tests
│   │
│   ├── modal-trainer/                  # Modal cloud training
│   │   ├── modal-train.py             # Modal training entry point
│   │   ├── modal-infer.py             # Modal inference entry point
│   │   ├── modal-test.py              # Modal testing
│   │   └── requirements-modal.txt     # Modal-specific deps
│   │
│   └── get-data/                       # Data preprocessing
│       ├── precompute_bev_features.py # BEV feature extraction
│       ├── create_nuScenes_subset.py  # Dataset subsetting
│       └── get_nuscenes_with_extract.py # Data download helper
│
├── modal_inference_*/                  # Inference outputs (gitignored)
│   └── YYYYMMDD_HHMMSS/
│       ├── artifacts/                 # Per-sample JSON predictions
│       ├── viz/                       # Visualization images
│       ├── inference_sampling_*.json  # All predictions
│       ├── metrics_summary_*.json     # Aggregated metrics
│       └── create_visualizations.py   # Visualization script
│
└── checkpoints/                        # Training checkpoints (gitignored)
    └── run_YYYYMMDD_HHMMSS/
        ├── checkpoint_step_*/         # Per-step checkpoints
        ├── best_checkpoint/           # Best performing checkpoint
        ├── training.log               # Training logs
        └── config.json                # Training configuration
```

---

## 📊 Results

### Performance Metrics

#### Caption Generation (nuCaption Dataset)
| Metric | Score | Description |
|--------|-------|-------------|
| **BLEU-4** | 0.203 | N-gram overlap with reference captions |
| **CIDEr** | 0.598 | Consensus-based metric for caption quality |
| **BERTScore** | TBD | Semantic similarity using BERT embeddings |

**Test Configuration:**
- Checkpoint: Step 27,000
- Dataset: 40 samples from validation set
- Modality: Vision + LiDAR

### Sample Predictions

#### Example 1: Urban Intersection
**Image:** Front camera view at intersection  
**Question:** *"Describe the current driving scene."*  
**Ground Truth:** *"The vehicle is waiting at a traffic light intersection with cars ahead and pedestrians crossing."*  
**Prediction:** *"The ego vehicle is stopped at an intersection with traffic light ahead and multiple vehicles in front."*

#### Example 2: Highway Scenario
**Image:** Highway front view  
**Question:** *"What objects are visible ahead?"*  
**Ground Truth:** *"Multiple vehicles on a multi-lane highway with clear weather."*  
**Prediction:** *"Several cars and trucks are visible on the highway ahead, maintaining safe distances."*

### Ablation Studies
| Configuration | BLEU-4 | CIDEr | Notes |
|---------------|--------|-------|-------|
| Vision Only | 0.178 | 0.521 | Struggles with 3D spatial relationships |
| LiDAR Only | 0.165 | 0.489 | Lacks fine-grained object details |
| **Vision + LiDAR** | **0.203** | **0.598** | Best overall performance |

### Training Curves
Training metrics are logged during training and can be visualized:
```bash
# View training logs
tensorboard --logdir checkpoints/run_YYYYMMDD_HHMMSS/
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Reporting Bugs
- Use GitHub Issues
- Include system info (GPU, CUDA version, Python version)
- Provide minimal reproduction steps
- Attach relevant logs

### Suggesting Enhancements
- Open a GitHub Issue with the "enhancement" label
- Clearly describe the proposed feature
- Explain use cases and benefits

### Code Contributions
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Test** thoroughly
5. **Commit** with clear messages (`git commit -m 'Add amazing feature'`)
6. **Push** to your fork (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions/classes
- Write unit tests for new features
- Update documentation as needed
- Keep PRs focused and atomic

### Running Tests
```bash
cd src/encoder-decoder/

# Run all tests
pytest training-test/

# Run specific test module
pytest training-test/models/test_vat_vision.py

# Run with coverage
pytest --cov=training training-test/
```

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{lidar_vision_vqa2024,
  title={LiDAR-Vision-VQA: Multimodal Visual Question Answering for Autonomous Driving},
  author={Advaith Sajeev},
  year={2024},
  url={https://github.com/Advaith-Sajeev/LiDAR-Vision-VQA}
}
```

---

## 🙏 Acknowledgements

This project builds upon several excellent open-source projects:

- **[PCDet](https://github.com/open-mmlab/OpenPCDet)**: 3D object detection framework for LiDAR processing
- **[nuScenes](https://www.nuscenes.org/)**: Comprehensive autonomous driving dataset
- **[CLIP](https://github.com/openai/CLIP)**: Vision-language pre-training by OpenAI
- **[SAM](https://github.com/facebookresearch/segment-anything)**: Segment Anything Model by Meta
- **[Qwen2.5](https://github.com/QwenLM/Qwen2.5)**: High-quality open-source language model
- **[Modal](https://modal.com/)**: Cloud infrastructure for scalable training
- **[Hugging Face](https://huggingface.co/)**: Transformers library and model hub

Special thanks to:
- **nuScenes team** for the nuCaption and nuGrounding annotations
- The **autonomous driving research community** for inspiring this work

---

## 📞 Contact

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **GitHub**: [@Advaith-Sajeev](https://github.com/Advaith-Sajeev)

---

## 📄 License

This project is licensed under the MIT License. See [MIT License](https://opensource.org/licenses/MIT) for details.

---

## 🗺️ Roadmap

### Current Status (v1.0)
- ✅ Core architecture implementation
- ✅ Training pipeline with LoRA
- ✅ Modal cloud training support
- ✅ Caption generation (nuCaption)
- ✅ Basic inference and evaluation

### Upcoming Features (v1.1)
- 🔄 Object grounding evaluation (nuGrounding)
- 🔄 Multi-GPU distributed training optimization
- 🔄 Real-time inference optimization
- 🔄 Integration with ROS for robotics deployment
- 🔄 Expanded language model support (Llama 3, Mistral)

### Future Work (v2.0)
- 📋 Temporal modeling across video sequences
- 📋 Active learning for data-efficient training
- 📋 Cross-dataset generalization experiments
- 📋 Integration with end-to-end autonomous driving systems
- 📋 Web demo and interactive visualization tool

---

## 💡 Tips & Best Practices

### Training Tips
1. **Start Small**: Use `max_samples=1000` for quick experiments
2. **Monitor GPU Memory**: Adjust batch size and gradient accumulation accordingly
3. **Use LoRA**: Significantly reduces memory and enables larger batch sizes
4. **Enable Flash Attention**: 2-3x speedup with minimal setup
5. **Checkpoint Frequently**: Set `save_interval=1000` to avoid losing progress

### Debugging
```python
# Enable debug mode in config
config = {
    "debug_mode": True,
    "debug_level": 2,  # INFO=1, DEBUG=2, TRACE=3
    "debug_modules": ["trainer", "dataset"],  # Filter specific modules
}
```

### Performance Optimization
- **Mixed Precision**: Automatically enabled with `torch.cuda.amp`
- **Gradient Checkpointing**: Reduces memory at cost of compute
- **Data Prefetching**: Use `num_workers=4` in DataLoader
- **Pin Memory**: Enable for faster CPU-GPU transfer

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

**Made with ❤️ for the autonomous driving community**

</div>
