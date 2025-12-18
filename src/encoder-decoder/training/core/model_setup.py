"""Model initialization and setup for Vision-LLM training"""

import math
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from nuscenes.nuscenes import NuScenes
from typing import Dict, Tuple, Optional

from deepencoder.deepencoder_infer import DeepEncoderRuntime
from deepencoder.lora_config import DeepEncoderLoRAConfig
from configs.constants import (
    NUM_VIEWS,           # 6 camera views
    TOKENS_PER_VIEW,     # 256 tokens per view (16x16 grid)
    TOTAL_VISION_TOKENS, # 1536 (6 * 256)
    VIEW_SPECIAL_TOKENS, # Per-view delimiter tokens
)

from ..models import (
    VisionAdapter,
    make_lora,
    get_bnb_config,
)
from ..utils import count_trainable_params

# Check for Flash Attention availability
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


def maybe_compile_model(model: nn.Module, config: Dict, name: str = "") -> nn.Module:
    """
    Optionally compile model with torch.compile() for faster execution.
    
    torch.compile() uses TorchDynamo and TorchInductor to optimize PyTorch code,
    providing 10-30% speedup on modern GPUs (especially A100/H100).
    
    Args:
        model: The model to compile
        config: Training configuration (checks "use_torch_compile" key)
        name: Name for logging
        
    Returns:
        Compiled model if enabled, else original model
    
    Note:
        - Requires PyTorch 2.0+
        - First forward pass is slow (compilation)
        - Disable for debugging (set use_torch_compile: False)
        - mode="reduce-overhead" is best for training with varying shapes
    """
    if not config.get("use_torch_compile", False):
        return model
    
    if not hasattr(torch, 'compile'):
        print(f"[torch.compile] Skipping {name}: PyTorch < 2.0")
        return model
    
    try:
        # mode options: "default", "reduce-overhead", "max-autotune"
        # - "reduce-overhead": Good for training, handles dynamic shapes better
        # - "max-autotune": Slower compile, potentially faster runtime
        compile_mode = config.get("torch_compile_mode", "reduce-overhead")
        compiled = torch.compile(model, mode=compile_mode)
        print(f"[torch.compile] {name} compiled with mode={compile_mode}")
        return compiled
    except Exception as e:
        print(f"[torch.compile] Failed for {name}: {e}")
        return model


def _get_model_dtype(config: Dict, device: torch.device) -> torch.dtype:
    """
    Determine the dtype for LLM model based on config.
    
    Supports both legacy fp16 and new mixed_precision config.
    Priority:
        1. Check mixed_precision: "fp16" → float16, "bf16" → bfloat16
        2. Fallback to legacy fp16 boolean
        3. Default → bfloat16
        
    Note: Flash Attention requires float16 or bfloat16.
    
    Args:
        config: Training configuration with "mixed_precision" or "fp16" key
        device: Target device
        
    Returns:
        torch.dtype: Either torch.float16 or torch.bfloat16
    """
    # Handle new mixed_precision config (preferred)
    mixed_prec = config.get('mixed_precision', None)
    if mixed_prec == 'fp16':
        return torch.float16
    elif mixed_prec == 'bf16':
        return torch.bfloat16
    elif mixed_prec == 'no':
        return torch.bfloat16  # Even in "no" AMP mode, use bf16 for model weights
    
    # Fallback to legacy fp16 config
    if config.get("fp16", False) and device.type == "cuda":
        return torch.float16
    
    return torch.bfloat16


def setup_models(config: Dict, device: torch.device, is_main: bool):
    """
    Initialize all models for vision-only training.
    
    Args:
        config: Training configuration
        device: Device to place models on
        is_main: Whether this is the main process
        
    Returns:
        Tuple of (tokenizer, base_model, vision_adapter, runtime, nusc, d_model)
    """
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(config["model_id"], use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Add special tokens for per-view delimiters
    special_tokens = {
        "additional_special_tokens": list(VIEW_SPECIAL_TOKENS)
    }
    added = tok.add_special_tokens(special_tokens)
    if is_main:
        print(f"[tokenizer] Added {added} per-view delimiter tokens")

    # Determine attention implementation
    attn_implementation = None
    if _HAS_FLASH_ATTN and device.type == "cuda":
        attn_implementation = "flash_attention_2"
        if is_main:
            print("[LLM] Using Flash Attention 2 for faster training")
    elif is_main:
        print("[LLM] Flash Attention not available, using default attention")

    # Determine dtype using helper function
    model_dtype = _get_model_dtype(config, device)
    if is_main:
        print(f"[LLM] Using dtype: {model_dtype}")

    # Determine tuning mode
    tuning_mode = config.get("tuning_mode", "qlora")  # Default to qlora if missing
    use_qlora = (tuning_mode == "qlora")
    use_lora = (tuning_mode in ["qlora", "lora"])
    
    quantization_config = None
    
    if use_qlora:
        if is_main:
            print("[LLM] Enabling QLoRA with 4-bit quantization")
        
        qlora_compute_dtype_str = config.get("qlora_compute_dtype", "bfloat16")
        quantization_config = get_bnb_config(
            use_4bit=True,
            bnb_4bit_quant_type=config.get("qlora_quant_type", "nf4"),
            bnb_4bit_use_double_quant=config.get("qlora_double_quant", True),
            bnb_4bit_compute_dtype=qlora_compute_dtype_str,
        )
        if is_main:
            print(f"[LLM QLoRA] quant_type={config.get('qlora_quant_type', 'nf4')}, "
                  f"double_quant={config.get('qlora_double_quant', True)}, "
                  f"compute_dtype={qlora_compute_dtype_str}")
            
            # Warn if QLoRA compute dtype doesn't match model dtype
            qlora_dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
            qlora_compute_dtype = qlora_dtype_map.get(qlora_compute_dtype_str, torch.bfloat16)
            if qlora_compute_dtype != model_dtype:
                print(f"[LLM QLoRA] ⚠️  WARNING: QLoRA compute dtype ({qlora_compute_dtype}) differs from "
                      f"model dtype ({model_dtype}). This may cause dtype conversion overhead.")
                print(f"[LLM QLoRA]    Consider setting qlora_compute_dtype to match mixed_precision setting.")

    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=model_dtype,
        device_map={"": device},
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
    )
    
    # Resize embeddings for new special tokens
    base.resize_token_embeddings(len(tok))
    if is_main:
        print(f"[LLM] Resized embeddings to {len(tok)} tokens")
    
    # Apply LoRA adapters if QLoRA or LoRA is enabled
    # QLoRA = 4-bit quantization + LoRA adapters
    # LoRA = LoRA adapters only (no quantization)
    apply_lora = use_lora # use_lora is derived from tuning_mode above
    
    if apply_lora:
        lora_target_modules = config.get("llm_lora_targets", ["q_proj", "v_proj"])
        base = make_lora(
            base,
            targets=lora_target_modules,
            r=config["llm_lora_r"],
            alpha=config["llm_lora_alpha"],
            dropout=config["llm_lora_dropout"],
            is_quantized=use_qlora,
        )
        if is_main:
            mode = "QLoRA" if use_qlora else "LoRA"
            print(f"[LLM] {mode} adapters applied: r={config['llm_lora_r']}, alpha={config['llm_lora_alpha']}, targets={lora_target_modules}")
    else:
        if is_main:
            print("[LLM] Full fine-tuning mode (no LoRA/QLoRA)")
        # Enable gradients for all parameters when not using LoRA
        for p in base.parameters():
            p.requires_grad = True
    
    # Enable gradient checkpointing if configured
    if config.get("gradient_checkpointing", True):
        base.gradient_checkpointing_enable()
        if is_main:
            print("[LLM] Gradient checkpointing enabled")
    
    # Get model dimension from LLM config
    d_model = base.config.hidden_size
    if is_main:
        print(f"[LLM] Hidden dimension (d_model): {d_model}")

    # Vision pipeline (always enabled for vision-only model)
    # NuScenes
    nusc = NuScenes(version=config["nu_version"], dataroot=config["nu_dataroot"], verbose=False)
    if is_main:
        print(f"[nuScenes] Loaded {config['nu_version']}")

    # DeepEncoder with configurable output dimension
    clip_lora_config = None
    if config.get("clip_lora_enabled", False):
        clip_target_modules = config.get("clip_lora_target_modules", None)
        if clip_target_modules is None:
            from ..models import infer_clip_lora_targets
            clip_target_modules = infer_clip_lora_targets(config["openclip_pretrained"])
            if is_main:
                print(f"[CLIP LoRA] Auto-detected target modules: {clip_target_modules}")
        
        clip_lora_config = DeepEncoderLoRAConfig(
            enabled=True,  # Must set to True for LoRA to be applied
            r=config["clip_lora_r"],
            lora_alpha=config["clip_lora_alpha"],
            lora_dropout=config.get("clip_lora_dropout", 0.1),
            target_modules=clip_target_modules,
        )

    # Use model_dtype for DeepEncoder to ensure consistency
    deep_dtype_str = "bfloat16" if model_dtype == torch.bfloat16 else "float16"
    if is_main:
        print(f"[DeepEncoder] Using dtype: {deep_dtype_str} (matches LLM dtype)")
    
    runtime = DeepEncoderRuntime(
        sam_ckpt=config.get("sam_ckpt", None),
        auto_download_sam=config.get("auto_download_sam", True),
        device=("cuda" if device.type == "cuda" else "cpu"),
        dtype=deep_dtype_str,
        openclip_pretrained=config["openclip_pretrained"],
        lora_config=clip_lora_config,
        freeze_clip_backbone_when_lora_enabled=True,
        output_dim=d_model,  # Output directly in decoder's d_model dimension
    )

    # Freeze SAM backbone but keep compression heads (net_2, net_3) trainable
    for name, p in runtime.sam.named_parameters():
        # net_2 and net_3 are the DeepEncoder/VARY compression heads
        if name.startswith("net_2") or name.startswith("net_3"):
            p.requires_grad = True   # learnable compression heads
        else:
            p.requires_grad = False  # frozen SAM backbone

    # Enable gradient checkpointing for CLIP if configured
    if config.get("gradient_checkpointing", True):
        clip_model = runtime.clip_vit.base_model.model if hasattr(runtime.clip_vit, 'base_model') else runtime.clip_vit
        if hasattr(clip_model, 'gradient_checkpointing_enable'):
            clip_model.gradient_checkpointing_enable()
            if is_main:
                print(f"[CLIP ViT] Gradient checkpointing enabled")

    # Verify projector dimension
    projector_out_dim = runtime.projector.cfg.n_embed
    if is_main:
        print(f"[DeepEncoder] Projector output dimension: {projector_out_dim} (d_model={d_model})")

    # Enable gradients for projector
    for p in runtime.projector.parameters():
        p.requires_grad = True

    # VisionAdapter - takes DeepEncoder output, adds view embeddings, returns 6 separate tensors
    vision_adapter = VisionAdapter(d_model, dropout=0.10).to(device)
    vision_adapter = vision_adapter.to(dtype=model_dtype)
    if is_main:
        print(f"[VisionAdapter] Using dtype: {model_dtype}")
    
    # Apply torch.compile() for faster execution
    vision_adapter = maybe_compile_model(vision_adapter, config, "VisionAdapter")

    return tok, base, vision_adapter, runtime, nusc, d_model


def setup_optimizer_and_scheduler(
    base,
    vision_adapter,
    runtime,
    config: Dict,
    train_size: int,
    world_size: int,
):
    """
    Setup optimizer and learning rate scheduler for vision-only training.
    
    Args:
        base: Base LLM model
        vision_adapter: Vision adapter
        runtime: DeepEncoder runtime
        config: Training configuration
        train_size: Size of training dataset
        world_size: Number of distributed processes
        
    Returns:
        Tuple of (optimizer, scheduler, scheduler_metadata)
    """
    lora_params = [p for p in base.parameters() if p.requires_grad]
    
    # Vision parameters (CLIP + LoRA + Projector + SAM heads)
    vision_params = list(runtime.parameters())
    va_params = list(vision_adapter.parameters())

    optim_groups = [
        {"params": lora_params, "lr": config["lr_lora"], "weight_decay": config["weight_decay"]},
        {"params": vision_params, "lr": config["lr_vision"], "weight_decay": config["weight_decay"]},
        {"params": va_params, "lr": config["lr_vision"], "weight_decay": config["weight_decay"]},
    ]

    # Use fused AdamW if available (PyTorch 2.0+ on CUDA)
    use_fused = torch.cuda.is_available() and hasattr(torch.optim.AdamW, '__init__')
    try:
        optim = torch.optim.AdamW(optim_groups, fused=use_fused)
        if use_fused:
            print("[optimizer] Using fused AdamW (faster optimizer step)")
    except TypeError:
        optim = torch.optim.AdamW(optim_groups)
        print("[optimizer] Using standard AdamW (fused not available)")

    # Calculate scheduler steps
    effective_batch_size = config["batch_size"] * max(1, world_size) * config["grad_accum"]
    steps_per_epoch = max(1, math.ceil(train_size / effective_batch_size))
    total_steps = config["epochs"] * steps_per_epoch

    sched = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=config["warmup_steps"], num_training_steps=total_steps
    )

    sched_meta = {"total_steps": total_steps, "warmup_steps": config["warmup_steps"]}

    return optim, sched, sched_meta
