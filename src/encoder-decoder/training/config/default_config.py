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
    "fp16": False,
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
    "deep_dtype": "float32",                    # "bfloat16" or "float32"
    "openclip_pretrained": "openai",
}
