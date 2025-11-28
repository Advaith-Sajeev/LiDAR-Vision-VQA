"""
Default configuration for LiDAR-Vision-LLM training
"""

from typing import Dict, List, Optional

DEFAULT_CONFIG: Dict = {
    # I/O
    "feature_dirs": ["./bev_feats/train"],      # list of directories containing <sample_token>.npy
    "jsons": ["Dataset_subset/external/nuCaption.json", "Dataset_subset/external/nuGrounding.json"],
    "out_dir": "./checkpoints_vat",
    "max_samples": 10,                          # int or None

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
    "prefix_scale": 0.2,                        # scale on VAT prompts before feeding LLM

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
