from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DeepEncoderLoRAConfig:
    """
    Training-time LoRA/QLoRA configuration for CLIP.
    All fields are read by DeepEncoderRuntime and forwarded to PEFT's LoraConfig.

    enabled: whether to enable LoRA on CLIP at all.
    r:       LoRA rank.
    lora_alpha: LoRA scaling (alpha).
    lora_dropout: dropout applied to LoRA paths.
    bias:    "none" | "lora_only" | "all" (see PEFT docs).
    target_modules: list of module name substrings to match (e.g., ["qkv_proj", "out_proj"]).
                    If None, DeepEncoderRuntime will use auto-detected defaults from clip_l_lora_default_targets().
    
    QLoRA-specific (optional, CLIP is small enough that bf16 is usually fine):
    use_qlora: Enable 4-bit quantization for CLIP (not recommended, CLIP is small)
    qlora_quant_type: Quantization type ("nf4" or "fp4")
    qlora_double_quant: Use double quantization
    qlora_compute_dtype: Compute dtype for quantized operations
    """
    enabled: bool = False
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    target_modules: Optional[List[str]] = None
    
    # QLoRA-specific (optional for CLIP - it's small enough that bf16 is usually fine)
    use_qlora: bool = False
    qlora_quant_type: str = "nf4"
    qlora_double_quant: bool = True
    qlora_compute_dtype: str = "bfloat16"

    def materialize_target_modules(self, fallback: Optional[List[str]] = None) -> List[str]:
        """
        Utility for callers: returns a concrete list of target module names,
        using provided fallback if target_modules is None.
        """
        if self.target_modules is not None:
            return list(self.target_modules)
        return list(fallback or [])
