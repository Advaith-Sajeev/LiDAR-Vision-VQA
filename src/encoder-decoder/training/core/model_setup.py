"""Model initialization and setup"""

import math
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from nuscenes.nuscenes import NuScenes
from typing import Dict, Tuple, Optional

from deepencoder.deepencoder_infer import DeepEncoderRuntime
from deepencoder.lora_config import DeepEncoderLoRAConfig
from ..models import (
    VATLiDAR,
    VATVision,
    VisionAdapter,
    make_lora,
)
from ..utils import count_trainable_params


def setup_models(config: Dict, device: torch.device, is_main: bool):
    """
    Initialize all models for training.
    
    Args:
        config: Training configuration
        device: Device to place models on
        is_main: Whether this is the main process
        
    Returns:
        Tuple of (tokenizer, base_model, vat_lidar, vat_vision, vision_adapter, runtime, nusc, d_model, c_in)
    """
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(config["model_id"], use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Add special tokens
    special_tokens = {
        "additional_special_tokens": [
            "<vision_start>",
            "<vision_end>",
            "<lidar_start>",
            "<lidar_end>",
        ]
    }
    added = tok.add_special_tokens(special_tokens)

    # Base LLM
    base = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=torch.float16 if (config["fp16"] and device.type == "cuda") else None,
        device_map=None,
    ).to(device)
    base.config.use_cache = False
    base.requires_grad_(False)
    base.gradient_checkpointing_enable()

    if added > 0:
        base.resize_token_embeddings(len(tok))

    # Apply LoRA to base model
    lora_targets = config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    if is_main:
        print(f"[LLM LoRA] Applying LoRA to target modules: {lora_targets}")
    base = make_lora(base, lora_targets, config["lora_r"], config["lora_alpha"], config["lora_dropout"])

    d_model = base.config.hidden_size

    # Vision pipeline
    if config["use_vision"]:
        if is_main:
            print("[vision] initializing DeepEncoder...")
            
        nusc = NuScenes(
            version=config["nu_version"],
            dataroot=str(Path(config["nu_dataroot"]).resolve()),
            verbose=False,
        )

        # Create LoRA configuration for CLIP
        clip_target_modules = config.get("clip_lora_target_modules", None)
        if is_main:
            if clip_target_modules is None:
                print("[CLIP LoRA] Using auto-detected target modules")
            else:
                print(f"[CLIP LoRA] Using configured target modules: {clip_target_modules}")
        
        clip_lora_config = DeepEncoderLoRAConfig(
            enabled=config.get("clip_lora_enabled", True),
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            target_modules=clip_target_modules,  # Can be None (auto-detect) or custom list
        )

        runtime = DeepEncoderRuntime(
            sam_ckpt=config.get("sam_ckpt", None),
            auto_download_sam=config.get("auto_download_sam", True),
            device=("cuda" if device.type == "cuda" else "cpu"),
            dtype=config["deep_dtype"],
            openclip_pretrained=config["openclip_pretrained"],
            lora_config=clip_lora_config,
            freeze_clip_backbone_when_lora_enabled=True,
        )

        # Freeze SAM (already done in DeepEncoderRuntime, but explicit for clarity)
        for p in runtime.sam.parameters():
            p.requires_grad = False

        # Verify projector dimension
        # Projector expects 2048-dim input (CLIP 1024 + SAM 1024 concatenated)
        test_input = torch.randn(1, 2048, device=device)
        test_output = runtime.projector(test_input)
        projector_out_dim = test_output.shape[-1]
        if projector_out_dim != 2048:
            raise ValueError(
                f"DeepEncoder projector output dimension {projector_out_dim} "
                f"does not match VisionAdapter expected input 2048"
            )

        # Enable gradients for projector
        for p in runtime.projector.parameters():
            p.requires_grad = True

        # Vision models
        # VisionAdapter expects (d_in, dropout) - d_in=2048 from DeepEncoder projector
        vision_adapter = VisionAdapter(2048, dropout=0.10).to(device)
        
        # VATVision new signature: takes d_in, d_model, n_input_tokens, compression_factor
        # n_input_tokens = 6 views * 256 tokens/view = 1536
        # n_queries is calculated as: n_input_tokens // compression_factor
        # To get desired n_queries from config, calculate compression_factor
        n_input_tokens = 6 * 256  # 1536 tokens from VisionAdapter
        desired_n_queries = config["vision_queries"]
        
        # Calculate compression factor (default to 2 if exact division not possible)
        if n_input_tokens % desired_n_queries == 0:
            compression_factor = n_input_tokens // desired_n_queries
        else:
            # Fall back to compression_factor=2 (gives 768 queries)
            compression_factor = 2
            if is_main:
                print(f"[VATVision] Warning: vision_queries={desired_n_queries} not compatible with n_input_tokens={n_input_tokens}")
                print(f"[VATVision] Using compression_factor={compression_factor}, resulting in {n_input_tokens // compression_factor} queries")
        
        vat_vision = VATVision(
            d_in=2048,  # Input dimension from VisionAdapter
            d_model=d_model,  # Target output dimension
            n_input_tokens=n_input_tokens,
            compression_factor=compression_factor,
            n_layers=config["vision_layers"],
            n_heads=config["vision_heads"],
            mlp_ratio=config["vision_mlp_ratio"],
            dropout=config["vision_dropout"],
            post_dropout=config["vision_post_dropout"],
            use_per_view_query=config["vision_per_view_query"],
        ).to(device)
    else:
        nusc = runtime = vision_adapter = vat_vision = None

    # LiDAR VAT (need to probe BEV shape first, handled by caller)
    # Return None for now, will be created after dataset is loaded
    return tok, base, None, vat_vision, vision_adapter, runtime, nusc, d_model, None


def create_vat_lidar(c_in: int, d_model: int, config: Dict, device: torch.device):
    """
    Create LiDAR VAT model.
    
    Args:
        c_in: Number of input channels
        d_model: Model dimension
        config: Training configuration
        device: Device to place model on
        
    Returns:
        VATLiDAR model
    """
    return VATLiDAR(
        c_in=c_in,
        d_model=d_model,
        n_queries=config["vat_queries"],
        n_layers=config["vat_layers"],
        n_heads=config["vat_heads"],
        mlp_ratio=config["vat_mlp_ratio"],
        dropout=config["vat_dropout"],
        post_dropout=config["vat_post_dropout"],
    ).to(device)


def setup_optimizer_and_scheduler(
    base,
    vat_lidar,
    vat_vision,
    vision_adapter,
    runtime,
    config: Dict,
    train_size: int,
    world_size: int,
):
    """
    Setup optimizer and learning rate scheduler.
    
    Args:
        base: Base LLM model
        vat_lidar: LiDAR VAT model
        vat_vision: Vision VAT model (optional)
        vision_adapter: Vision adapter (optional)
        runtime: DeepEncoder runtime (optional)
        config: Training configuration
        train_size: Size of training dataset
        world_size: Number of distributed processes
        
    Returns:
        Tuple of (optimizer, scheduler, scheduler_metadata)
    """
    lora_params = [p for p in base.parameters() if p.requires_grad]
    lidar_params = list(vat_lidar.parameters())

    optim_groups = [
        {"params": lidar_params, "lr": config["lr_vat"], "weight_decay": config["weight_decay"]},
        {"params": lora_params, "lr": config["lr_lora"], "weight_decay": config["weight_decay"]},
    ]

    if config["use_vision"]:
        clip_lora_params = [p for p in runtime.clip_vit.parameters() if p.requires_grad]
        va_params = list(vision_adapter.parameters())
        proj_params = list(runtime.projector.parameters())
        vision_vat_params = list(vat_vision.parameters())

        optim_groups.append(
            {"params": clip_lora_params, "lr": config["lr_vision"], "weight_decay": config["weight_decay"]}
        )
        optim_groups.append(
            {"params": va_params + proj_params, "lr": config["lr_vision"], "weight_decay": config["weight_decay"]}
        )
        optim_groups.append(
            {"params": vision_vat_params, "lr": config["lr_vision_vat"], "weight_decay": config["weight_decay"]}
        )

    optim = torch.optim.AdamW(optim_groups)

    # Calculate scheduler steps
    effective_batch_size = config["batch_size"] * max(1, world_size) * config["grad_accum"]
    steps_per_epoch = max(1, math.ceil(train_size / effective_batch_size))
    total_steps = config["epochs"] * steps_per_epoch

    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=config["warmup_steps"], num_training_steps=total_steps
    )

    sched_meta = {"total_steps": total_steps, "warmup_steps": config["warmup_steps"]}

    return optim, sched, sched_meta
