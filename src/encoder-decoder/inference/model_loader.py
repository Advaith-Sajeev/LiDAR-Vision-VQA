"""
Model loading utilities for inference
"""

import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from nuscenes.nuscenes import NuScenes
from typing import Dict, Optional, Tuple
from copy import deepcopy
import json

from deepencoder.deepencoder_infer import DeepEncoderRuntime
from deepencoder.lora_config import DeepEncoderLoRAConfig
from deepencoder import TOKENS_PER_VIEW  # 256 tokens per view (from FIXED_GRID_SIDE=16)
from configs.default_config import NUM_VIEWS, PROJECTOR_DIM  # 6 camera views, 2048 projector dim
from configs.training_config import (
    get_training_config as get_default_training_config,
)

from training.models import (
    VATLiDAR,
    VATVision,
    VisionAdapter,
    make_lora,
    get_bnb_config,
    infer_clip_lora_targets,
)


def _resolve_model_dtype(config: Dict, device: torch.device) -> torch.dtype:
    """Mirror training-time dtype resolution so inference matches checkpoints."""
    if device.type != "cuda":
        # CPU path relies on float32 kernels for reliability.
        return torch.float32
    mixed_prec = config.get("mixed_precision")
    if mixed_prec == "fp16":
        return torch.float16
    if mixed_prec == "bf16":
        return torch.bfloat16
    if mixed_prec == "no":
        return torch.bfloat16
    if config.get("fp16", False):
        return torch.float16
    return torch.bfloat16


def _dtype_to_deepencoder_str(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return "float32"


class ModelLoader:
    """
    Loads trained LiDAR-Vision-LLM models from checkpoint.
    
    Handles:
      - Base LLM with LoRA
      - LiDAR VAT
      - Vision VAT (if enabled)
      - Vision Adapter (if enabled)
      - DeepEncoder runtime (if enabled)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        device: Optional[str] = None,
        fallback_config: Optional[Dict] = None,
    ):
        """
        Initialize model loader.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            device: Device to load models on ('cuda', 'cpu', or None for auto)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._fallback_config = deepcopy(fallback_config) if fallback_config else None
        
        # Load config
        config_path = self.checkpoint_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"[loader] Loaded training config from {config_path}")
        else:
            if self._fallback_config is not None:
                self.config = deepcopy(self._fallback_config)
                print(
                    f"[loader] Warning: {config_path} not found; using provided fallback config"
                )
            else:
                print(
                    f"[loader] Warning: {config_path} not found; falling back to configs.training_config defaults"
                )
                self.config = get_default_training_config()
        
        self.model_dtype = _resolve_model_dtype(self.config, self.device)
        self.deep_dtype_str = _dtype_to_deepencoder_str(self.model_dtype)

        print(f"[loader] Loading models from {checkpoint_dir}")
        print(f"[loader] Using device: {self.device}")
        print(f"[loader] Target dtype: {self.model_dtype}")

    def _infer_c_in_from_checkpoint(self) -> Optional[int]:
        """Best-effort recovery of BEV channels from saved VAT weights."""
        vat_path = self.checkpoint_dir / "vat_lidar_latest.pt"
        if not vat_path.exists():
            return None
        try:
            state = torch.load(vat_path, map_location="cpu")
        except Exception as exc:
            print(f"[loader] Warning: failed to inspect {vat_path} for c_in: {exc}")
            return None
        weight = state.get("refine.0.weight")
        if isinstance(weight, torch.Tensor):
            return int(weight.shape[0])
        return None
        
    def load_tokenizer(self):
        """Load and configure tokenizer."""
        print("[loader] Loading tokenizer...")
        tok = AutoTokenizer.from_pretrained(self.config["model_id"], use_fast=True)
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
        tok.add_special_tokens(special_tokens)
        return tok
    
    def load_base_model(self, tokenizer):
        """Load base LLM with LoRA/QLoRA."""
        print("[loader] Loading base LLM...")
        
        # Check if model was trained with QLoRA
        use_qlora = self.config.get("use_qlora", False)
        quantization_config = None
        
        if use_qlora:
            print("[loader] Model was trained with QLoRA, loading with 4-bit quantization")
            compute_dtype_str = self.config.get(
                "qlora_compute_dtype",
                "bfloat16" if self.model_dtype == torch.bfloat16 else "float16",
            )
            quantization_config = get_bnb_config(
                use_4bit=True,
                bnb_4bit_quant_type=self.config.get("qlora_quant_type", "nf4"),
                bnb_4bit_use_double_quant=self.config.get("qlora_double_quant", True),
                bnb_4bit_compute_dtype=compute_dtype_str,
            )
        
        base_dtype = self.model_dtype if self.device.type == "cuda" else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            self.config["model_id"],
            dtype=base_dtype,
            device_map="auto" if use_qlora else None,
            quantization_config=quantization_config,
        )
        
        # Move to device if not using QLoRA
        if not use_qlora:
            base = base.to(self.device)
        
        base.config.use_cache = True  # Enable KV cache for faster inference
        
        # Resize embeddings if tokens were added
        if len(tokenizer) != base.config.vocab_size:
            base.resize_token_embeddings(len(tokenizer))
        
        # Apply LoRA
        # Use target modules from config (saved during training) or default
        lora_targets = self.config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        print(f"[loader] Applying {'QLoRA' if use_qlora else 'LoRA'} with targets: {lora_targets}")
        base = make_lora(
            base,
            lora_targets,
            self.config["lora_r"],
            self.config["lora_alpha"],
            self.config["lora_dropout"],
            is_quantized=use_qlora,
        )
        
        # Load LoRA weights - try PEFT adapter directory first, then legacy lora.pt
        # Training saves to: qwen2_lora_adapter_latest
        lora_adapter_path = self.checkpoint_dir / "qwen2_lora_adapter_latest"
        lora_legacy_path = self.checkpoint_dir / "lora.pt"
        
        if lora_adapter_path.exists():
            print(f"[loader] Loading LLM LoRA adapter from {lora_adapter_path}")
            # Load adapter weights using PEFT's set_peft_model_state_dict
            adapter_weights_path = lora_adapter_path / "adapter_model.safetensors"
            if adapter_weights_path.exists():
                from safetensors.torch import load_file
                adapter_state = load_file(str(adapter_weights_path))
            else:
                adapter_weights_path = lora_adapter_path / "adapter_model.bin"
                if adapter_weights_path.exists():
                    adapter_state = torch.load(adapter_weights_path, map_location=self.device)
                else:
                    print(f"[loader] Warning: No adapter weights found in {lora_adapter_path}")
                    adapter_state = None
            
            if adapter_state is not None:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(base, adapter_state)
                print(f"[loader] LLM LoRA adapter loaded successfully via set_peft_model_state_dict()")
        elif lora_legacy_path.exists():
            print(f"[loader] Loading LoRA weights from legacy {lora_legacy_path}")
            lora_state = torch.load(lora_legacy_path, map_location=self.device)
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(base, lora_state)
        else:
            print(f"[loader] Warning: LoRA weights not found at {lora_adapter_path} or {lora_legacy_path}")
        
        base.eval()
        return base
    
    def load_lidar_vat(self, d_model: int, c_in: int):
        """Load LiDAR VAT model."""
        print("[loader] Loading LiDAR VAT...")
        vat_lidar = VATLiDAR(
            c_in=c_in,
            d_model=d_model,
            n_queries=self.config["vat_queries"],
            n_layers=self.config["vat_layers"],
            n_heads=self.config["vat_heads"],
            mlp_ratio=self.config["vat_mlp_ratio"],
            dropout=self.config["vat_dropout"],
            post_dropout=self.config["vat_post_dropout"],
        ).to(self.device)
        
        # Load weights
        vat_path = self.checkpoint_dir / "vat_lidar_latest.pt"
        if vat_path.exists():
            print(f"[loader] Loading LiDAR VAT weights from {vat_path}")
            vat_state = torch.load(vat_path, map_location=self.device)
            vat_lidar.load_state_dict(vat_state)
        else:
            raise FileNotFoundError(f"LiDAR VAT weights not found: {vat_path}")
        
        vat_lidar = vat_lidar.to(dtype=self.model_dtype)
        vat_lidar.eval()
        return vat_lidar
    
    def load_vision_pipeline(self, d_model: int):
        """Load vision pipeline components (if enabled)."""
        if not self.config.get("use_vision", False):
            return None, None, None, None
        
        print("[loader] Loading vision pipeline...")
        
        # Initialize nuScenes
        nusc = NuScenes(
            version=self.config["nu_version"],
            dataroot=str(Path(self.config["nu_dataroot"]).resolve()),
            verbose=False,
        )
        
        # Create LoRA configuration for CLIP
        # Use target modules from config (saved during training) or auto-detect
        clip_target_modules = self.config.get("clip_lora_target_modules", None)
        if clip_target_modules is None:
            print(f"[loader] CLIP LoRA targets: auto-detecting...")
            # Note: Will be auto-detected by DeepEncoderLoRAConfig using infer_clip_lora_targets
        else:
            print(f"[loader] CLIP LoRA targets: {clip_target_modules}")
        
        clip_lora_config = DeepEncoderLoRAConfig(
            enabled=self.config.get("clip_lora_enabled", True),
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            bias="none",
            target_modules=clip_target_modules,
        )
        
        # Initialize DeepEncoder
        configured_dtype = self.config.get("deep_dtype")
        if configured_dtype and configured_dtype != self.deep_dtype_str:
            print(
                f"[loader] Warning: overriding deep_dtype ({configured_dtype}) with {self.deep_dtype_str} to match training dtype"
            )

        runtime = DeepEncoderRuntime(
            sam_ckpt=self.config.get("sam_ckpt", None),
            auto_download_sam=self.config.get("auto_download_sam", True),
            device=str(self.device),
            dtype=self.deep_dtype_str,
            openclip_pretrained=self.config["openclip_pretrained"],
            lora_config=clip_lora_config,
            freeze_clip_backbone_when_lora_enabled=True,
        )
        
        # Freeze SAM (already done in DeepEncoderRuntime)
        for p in runtime.sam.parameters():
            p.requires_grad = False
        runtime.sam.eval()

        sam_head_path = self.checkpoint_dir / "sam_compression_head_latest.pt"
        if sam_head_path.exists():
            print(f"[loader] Loading SAM compression head from {sam_head_path}")
            sam_state = torch.load(sam_head_path, map_location=self.device)
            current = runtime.sam.state_dict()
            current.update(sam_state)
            runtime.sam.load_state_dict(current)
        else:
            print("[loader] Warning: SAM compression head weights not found; using pretrained head")
        
        # Load CLIP LoRA weights if they exist
        # Note: The LoRA adapters are already applied by DeepEncoderRuntime
        # Training saves to: clip_lora_adapter_latest
        clip_lora_path = self.checkpoint_dir / "clip_lora_adapter_latest"
        if clip_lora_path.exists():
            print(f"[loader] Loading CLIP LoRA adapter from {clip_lora_path}")
            # PEFT's save_pretrained/from_pretrained handles loading
            # The runtime already wrapped CLIP with LoRA, just need to load weights
            try:
                from peft import PeftModel
                # Load the adapter weights into the existing PEFT model
                runtime.clip_vit = PeftModel.from_pretrained(
                    runtime.clip_vit.get_base_model(),
                    clip_lora_path,
                    is_trainable=False
                )
            except Exception as e:
                print(f"[loader] Warning: Could not load CLIP LoRA adapter: {e}")
                print("[loader] Continuing with initialized LoRA weights...")
        
        runtime.clip_vit.eval()
        runtime.clip_vit = runtime.clip_vit.to(dtype=self.model_dtype)
        
        # Load projector weights
        proj_path = self.checkpoint_dir / "projector_latest.pt"
        if proj_path.exists():
            print(f"[loader] Loading projector weights from {proj_path}")
            proj_state = torch.load(proj_path, map_location=self.device)
            runtime.projector.load_state_dict(proj_state)
        
        runtime.projector.eval()
        
        # Load Vision Adapter
        vision_adapter = VisionAdapter(PROJECTOR_DIM, d_model, dropout=0.10).to(self.device)
        va_path = self.checkpoint_dir / "vision_adapter_latest.pt"
        if va_path.exists():
            print(f"[loader] Loading vision adapter weights from {va_path}")
            va_state = torch.load(va_path, map_location=self.device)
            vision_adapter.load_state_dict(va_state)
        else:
            raise FileNotFoundError(f"Vision adapter weights not found: {va_path}")
        vision_adapter = vision_adapter.to(dtype=self.model_dtype)
        vision_adapter.eval()
        
        # Load Vision VAT
        # Token count derived from deepencoder grid (FIXED_GRID_SIDE) and camera views (NUM_VIEWS)
        n_input_tokens = NUM_VIEWS * TOKENS_PER_VIEW  # 6 * 256 = 1536 (default)
        n_queries = self.config["vision_queries"]  # Any positive integer allowed
        
        print(f"[loader] VATVision: n_input_tokens={n_input_tokens} → n_queries={n_queries}")
        
        # Note: d_in == d_model since VisionAdapter already projects to d_model
        # VATVision operates entirely in d_model dimension space
        vat_vision = VATVision(
            d_in=d_model,  # Input from VisionAdapter (already projected to d_model)
            d_model=d_model,  # Output dimension (same as input in current architecture)
            n_input_tokens=n_input_tokens,
            n_queries=n_queries,  # Direct: any positive integer allowed
            n_layers=self.config["vision_layers"],
            n_heads=self.config["vision_heads"],
            mlp_ratio=self.config["vision_mlp_ratio"],
            dropout=self.config["vision_dropout"],
            post_dropout=self.config["vision_post_dropout"],
            use_per_view_query=self.config["vision_per_view_query"],
            strict_per_view=self.config.get("vision_strict_per_view", False),
        ).to(self.device)
        
        vat_vision_path = self.checkpoint_dir / "vat_vision_latest.pt"
        if vat_vision_path.exists():
            print(f"[loader] Loading vision VAT weights from {vat_vision_path}")
            vat_vision_state = torch.load(vat_vision_path, map_location=self.device)
            vat_vision.load_state_dict(vat_vision_state)
        else:
            raise FileNotFoundError(f"Vision VAT weights not found: {vat_vision_path}")
        vat_vision = vat_vision.to(dtype=self.model_dtype)
        vat_vision.eval()
        
        runtime.projector = runtime.projector.to(dtype=self.model_dtype)
        return vat_vision, vision_adapter, runtime, nusc
    
    def load_all(self, c_in: Optional[int] = None) -> Dict:
        """
        Load all model components.
        
        Args:
            c_in: Number of input channels for LiDAR VAT (auto-detect if None)
            
        Returns:
            Dictionary containing all models and components
        """
        # Load tokenizer
        tokenizer = self.load_tokenizer()
        
        # Load base model
        base_model = self.load_base_model(tokenizer)
        d_model = base_model.config.hidden_size
        
        # Auto-detect c_in if not provided
        if c_in is None:
            c_in = self.config.get("c_in")
            if c_in is None:
                inferred = self._infer_c_in_from_checkpoint()
                if inferred is not None:
                    c_in = inferred
                    print(f"[loader] Inferred c_in={c_in} from vat_lidar_latest.pt")
            if c_in is None:
                c_in = 256
                print("[loader] Warning: c_in missing; falling back to 256 (check config.json)")
            else:
                print(f"[loader] Using c_in={c_in} (from config)")
        
        # Load LiDAR VAT
        vat_lidar = self.load_lidar_vat(d_model, c_in)
        
        # Load vision pipeline
        vat_vision, vision_adapter, runtime, nusc = self.load_vision_pipeline(d_model)
        
        print("[loader] All models loaded successfully!")
        
        return {
            "tokenizer": tokenizer,
            "base_model": base_model,
            "vat_lidar": vat_lidar,
            "vat_vision": vat_vision,
            "vision_adapter": vision_adapter,
            "runtime": runtime,
            "nusc": nusc,
            "config": self.config,
            "device": self.device,
            "d_model": d_model,
        }
