from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DeepEncoderLoRAConfig:
    """
    Training-time LoRA configuration for CLIP.
    All fields are read by DeepEncoderRuntime and forwarded to PEFT's LoraConfig.

    enabled: whether to enable LoRA on CLIP at all.
    r:       LoRA rank.
    lora_alpha: LoRA scaling (alpha).
    lora_dropout: dropout applied to LoRA paths.
    bias:    "none" | "lora_only" | "all" (see PEFT docs).
    target_modules: list of module name substrings to match (e.g., ["qkv_proj", "out_proj"]).
                    If None, training code should supply defaults (see clip_l_lora_default_targets()).
    """
    enabled: bool = False
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    target_modules: Optional[List[str]] = None

    def materialize_target_modules(self, fallback: Optional[List[str]] = None) -> List[str]:
        """
        Utility for callers: returns a concrete list of target module names,
        using provided fallback if target_modules is None.
        """
        if self.target_modules is not None:
            return list(self.target_modules)
        return list(fallback or [])
