"""LoRA/QLoRA utilities for model adaptation"""

import re
import types
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Optional, Dict, Any


def make_lora(
    model,
    targets: List[str],
    r: int,
    alpha: int,
    dropout: float,
    is_quantized: bool = False,
):
    """
    Apply LoRA (or QLoRA) to a causal language model.
    
    Args:
        model: Base model to apply LoRA to
        targets: List of target module names
        r: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        is_quantized: Whether the model is quantized (4-bit/8-bit)
        
    Returns:
        PEFT model with LoRA applied
    """
    # Prepare model for k-bit training if quantized
    if is_quantized:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def get_bnb_config(
    use_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",
) -> Optional["BitsAndBytesConfig"]:
    """
    Create BitsAndBytesConfig for QLoRA quantization.
    
    Args:
        use_4bit: Enable 4-bit quantization
        bnb_4bit_quant_type: Quantization type ("nf4" or "fp4")
        bnb_4bit_use_double_quant: Use double quantization
        bnb_4bit_compute_dtype: Compute dtype string
        
    Returns:
        BitsAndBytesConfig for model loading
    """
    from transformers import BitsAndBytesConfig
    
    # Map string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(bnb_4bit_compute_dtype, torch.bfloat16)
    
    return BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def patch_clip_peft_forward(peft_clip):
    """
    Make PEFT-wrapped CLIP behave like VitModel(x, patch_embeds):
      forward(x, sam_feats) -> base_model(x, sam_feats)
    This keeps DeepEncoderRuntime.encode_image/encode_views working.
    
    Args:
        peft_clip: PEFT-wrapped CLIP model
        
    Returns:
        Patched PEFT model
    """
    def _forward(self, x, patch_embeds):
        return self.base_model(x, patch_embeds)

    peft_clip.forward = types.MethodType(_forward, peft_clip)
    return peft_clip


def infer_clip_lora_targets(model: nn.Module) -> List[str]:
    """
    Infer LoRA target module names for CLIP model.
    
    Args:
        model: CLIP model to analyze
        
    Returns:
        List of target module names for LoRA
    """
    names = []
    
    # First pass: look for common attention/MLP patterns
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if re.search(
                r"(attn\.(q_proj|k_proj|v_proj|out_proj|qkv)|mlp\.(fc1|fc2)|\.proj$)",
                name,
            ):
                names.append(name.split(".")[-1])
                
    # Second pass: fallback to common linear layer names
    if not names:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and name.endswith(
                ("qkv", "proj", "fc1", "fc2", "out_proj")
            ):
                names.append(name.split(".")[-1])
                
    # Remove duplicates while preserving order
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
            
    return uniq
