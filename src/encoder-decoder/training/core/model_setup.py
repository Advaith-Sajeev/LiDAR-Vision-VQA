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
from configs.default_config import (
    NUM_VIEWS,           # 6 camera views
    TOKENS_PER_VIEW,     # 256 tokens per view (16x16 grid)
    PROJECTOR_DIM,       # 2048 (CLIP + SAM fused)
    TOTAL_VISION_TOKENS, # 1536 (6 * 256)
)

from ..models import (
    VATLiDAR,
    VATVision,
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
    Initialize all models for training.
    
    Args:
        config: Training configuration
        device: Device to place models on
        is_main: Whether this is the main process
        
    Returns:
        Tuple of (tokenizer, base_model, vat_vision, vision_adapter, runtime, nusc, d_model)
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

    # Check if using QLoRA (4-bit quantization)
    use_qlora = config.get("use_qlora", False)
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
            
            # Warn if QLoRA compute dtype doesn't match model dtype (potential performance issue)
            qlora_dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
            qlora_compute_dtype = qlora_dtype_map.get(qlora_compute_dtype_str, torch.bfloat16)
            if qlora_compute_dtype != model_dtype:
                print(f"[LLM QLoRA] ⚠️  WARNING: QLoRA compute dtype ({qlora_compute_dtype}) differs from "
                      f"model dtype ({model_dtype}). This may cause dtype conversion overhead.")
                print(f"[LLM QLoRA]    Consider setting qlora_compute_dtype to match mixed_precision setting.")

    # Base LLM
    base = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        dtype=model_dtype,
        device_map="auto" if use_qlora else None,  # QLoRA needs device_map for quantization
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
    )
    
    # Move to device if not using QLoRA (QLoRA uses device_map="auto")
    if not use_qlora:
        base = base.to(device)
    
    base.config.use_cache = False
    
    # Gradient checkpointing is handled by prepare_model_for_kbit_training for QLoRA
    if not use_qlora:
        base.requires_grad_(False)
        base.gradient_checkpointing_enable()

    if added > 0:
        base.resize_token_embeddings(len(tok))

    # Apply LoRA to base model
    lora_targets = config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    if is_main:
        print(f"[LLM {'QLoRA' if use_qlora else 'LoRA'}] Applying LoRA to target modules: {lora_targets}")
    base = make_lora(
        base, 
        lora_targets, 
        config["lora_r"], 
        config["lora_alpha"], 
        config["lora_dropout"],
        is_quantized=use_qlora,
    )

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

        # Use model_dtype for DeepEncoder to ensure consistency
        # Override config["deep_dtype"] if it exists to maintain consistency
        deep_dtype_str = "bfloat16" if model_dtype == torch.bfloat16 else "float16"
        if is_main:
            print(f"[DeepEncoder] Using dtype: {deep_dtype_str} (matches LLM dtype)")
        
        runtime = DeepEncoderRuntime(
            sam_ckpt=config.get("sam_ckpt", None),
            auto_download_sam=config.get("auto_download_sam", True),
            device=("cuda" if device.type == "cuda" else "cpu"),
            dtype=deep_dtype_str,  # Use model_dtype for consistency
            openclip_pretrained=config["openclip_pretrained"],
            lora_config=clip_lora_config,
            freeze_clip_backbone_when_lora_enabled=True,
        )

        # Freeze SAM (already done in DeepEncoderRuntime, but explicit for clarity)
        for p in runtime.sam.parameters():
            p.requires_grad = False

        # Enable gradient checkpointing for CLIP if configured
        if config.get("gradient_checkpointing", True):
            # Access the base model if wrapped by PEFT/LoRA
            clip_model = runtime.clip_vit.base_model.model if hasattr(runtime.clip_vit, 'base_model') else runtime.clip_vit
            if hasattr(clip_model, 'gradient_checkpointing_enable'):
                clip_model.gradient_checkpointing_enable()
                if is_main:
                    print(f"[CLIP ViT] Gradient checkpointing enabled")

        # Verify projector dimension matches expected constant
        projector_out_dim = runtime.projector.cfg.n_embed
        assert projector_out_dim == PROJECTOR_DIM, \
            f"DeepEncoder projector output dimension {projector_out_dim} != {PROJECTOR_DIM}"

        # Enable gradients for projector
        for p in runtime.projector.parameters():
            p.requires_grad = True

        # Vision models
        # VisionAdapter: takes DeepEncoder output [TOKENS_PER_VIEW, PROJECTOR_DIM] per view,
        # adds view embeddings, concatenates NUM_VIEWS views → [TOTAL_VISION_TOKENS, PROJECTOR_DIM],
        # then projects to d_model → [TOTAL_VISION_TOKENS, d_model]
        vision_adapter = VisionAdapter(PROJECTOR_DIM, d_model, dropout=0.10).to(device)
        
        # Convert vision models to match LLM dtype for consistency
        vision_adapter = vision_adapter.to(dtype=model_dtype)
        if is_main:
            print(f"[VisionAdapter] Using dtype: {model_dtype}")
        
        # Apply torch.compile() for faster execution (PyTorch 2.0+)
        vision_adapter = maybe_compile_model(vision_adapter, config, "VisionAdapter")
        
        # VATVision: compresses tokens via cross-attention
        # Input: [B, n_input_tokens, d_model] from VisionAdapter
        # Output: [B, n_queries, d_model] where n_queries is configurable
        # 
        # Token count is derived from deepencoder grid size (FIXED_GRID_SIDE=16 → 256 tokens/view)
        # and number of camera views (NUM_VIEWS=6), avoiding hardcoded magic numbers.
        n_input_tokens = NUM_VIEWS * TOKENS_PER_VIEW  # 6 * 256 = 1536 tokens
        n_queries = config["vision_queries"]  # Any positive integer (no divisibility constraint)
        
        if is_main:
            print(f"[VATVision] n_input_tokens={n_input_tokens} → n_queries={n_queries}")
        
        # Note: d_in == d_model since VisionAdapter already projects to d_model
        # VATVision operates entirely in d_model dimension space
        vat_vision = VATVision(
            d_in=d_model,  # Input from VisionAdapter (already projected to d_model)
            d_model=d_model,  # Output dimension (same as input in current architecture)
            n_input_tokens=n_input_tokens,
            n_queries=n_queries,  # Direct: any positive integer allowed
            n_layers=config["vision_layers"],
            n_heads=config["vision_heads"],
            mlp_ratio=config["vision_mlp_ratio"],
            dropout=config["vision_dropout"],
            post_dropout=config["vision_post_dropout"],
            use_per_view_query=config["vision_per_view_query"],
            strict_per_view=config.get("vision_strict_per_view", False),
        ).to(device)
        
        # Convert vision VAT to match LLM dtype
        vat_vision = vat_vision.to(dtype=model_dtype)
        if is_main:
            print(f"[VATVision] Using dtype: {model_dtype}")
        
        # Enable gradient checkpointing if configured (trades compute for memory)
        if config.get("gradient_checkpointing", True):
            vat_vision.gradient_checkpointing_enable()
            if is_main:
                print(f"[VATVision] Gradient checkpointing enabled")
        
        # Apply torch.compile() for faster execution (PyTorch 2.0+)
        vat_vision = maybe_compile_model(vat_vision, config, "VATVision")
    else:
        nusc = runtime = vision_adapter = vat_vision = None

    # LiDAR VAT created later after probing BEV shape from dataset
    return tok, base, vat_vision, vision_adapter, runtime, nusc, d_model


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
    # Determine model dtype to match LLM
    model_dtype = _get_model_dtype(config, device)
    
    vat_lidar = VATLiDAR(
        c_in=c_in,
        d_model=d_model,
        n_queries=config["vat_queries"],
        n_layers=config["vat_layers"],
        n_heads=config["vat_heads"],
        mlp_ratio=config["vat_mlp_ratio"],
        dropout=config["vat_dropout"],
        post_dropout=config["vat_post_dropout"],
    ).to(device)
    
    # Convert to match LLM dtype
    vat_lidar = vat_lidar.to(dtype=model_dtype)
    
    # Enable gradient checkpointing if configured (trades compute for memory)
    if config.get("gradient_checkpointing", True):
        vat_lidar.gradient_checkpointing_enable()
        print(f"[VATLiDAR] Gradient checkpointing enabled")
    
    # Apply torch.compile() for faster execution (PyTorch 2.0+)
    vat_lidar = maybe_compile_model(vat_lidar, config, "VATLiDAR")
    
    return vat_lidar


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

    # Use fused AdamW if available (PyTorch 2.0+ on CUDA) for ~5-15% faster optimizer step
    # Fused optimizer combines multiple CUDA kernels into one, reducing kernel launch overhead
    use_fused = torch.cuda.is_available() and hasattr(torch.optim.AdamW, '__init__')
    try:
        optim = torch.optim.AdamW(optim_groups, fused=use_fused)
        if use_fused:
            print("[optimizer] Using fused AdamW (faster optimizer step)")
    except TypeError:
        # Fallback for older PyTorch versions that don't support fused parameter
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
