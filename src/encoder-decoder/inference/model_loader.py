"""
Model loading utilities for inference
"""

import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from nuscenes.nuscenes import NuScenes
from typing import Dict, Optional, Tuple
import json

from deepencoder.deepencoder_infer import DeepEncoderRuntime
from deepencoder.lora_config import DeepEncoderLoRAConfig
from training.models import (
    VATLiDAR,
    VATVision,
    VisionAdapter,
    make_lora,
)


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
    
    def __init__(self, checkpoint_dir: str, device: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            device: Device to load models on ('cuda', 'cpu', or None for auto)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load config
        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"[loader] Loading models from {checkpoint_dir}")
        print(f"[loader] Using device: {self.device}")
        
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
        """Load base LLM with LoRA."""
        print("[loader] Loading base LLM...")
        base = AutoModelForCausalLM.from_pretrained(
            self.config["model_id"],
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map=None,
        ).to(self.device)
        
        base.config.use_cache = True  # Enable KV cache for faster inference
        base.requires_grad_(False)
        
        # Resize embeddings if tokens were added
        if len(tokenizer) != base.config.vocab_size:
            base.resize_token_embeddings(len(tokenizer))
        
        # Apply LoRA
        # Use target modules from config (saved during training) or default
        lora_targets = self.config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        print(f"[loader] Applying LoRA with targets: {lora_targets}")
        base = make_lora(
            base,
            lora_targets,
            self.config["lora_r"],
            self.config["lora_alpha"],
            self.config["lora_dropout"]
        )
        
        # Load LoRA weights
        lora_path = self.checkpoint_dir / "lora.pt"
        if lora_path.exists():
            print(f"[loader] Loading LoRA weights from {lora_path}")
            lora_state = torch.load(lora_path, map_location=self.device)
            base.load_state_dict(lora_state, strict=False)
        else:
            print(f"[loader] Warning: LoRA weights not found at {lora_path}")
        
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
        vat_path = self.checkpoint_dir / "vat_lidar.pt"
        if vat_path.exists():
            print(f"[loader] Loading LiDAR VAT weights from {vat_path}")
            vat_state = torch.load(vat_path, map_location=self.device)
            vat_lidar.load_state_dict(vat_state)
        else:
            raise FileNotFoundError(f"LiDAR VAT weights not found: {vat_path}")
        
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
        # Use target modules from config (saved during training) or None for auto-detect
        clip_target_modules = self.config.get("clip_lora_target_modules", None)
        print(f"[loader] CLIP LoRA targets: {clip_target_modules if clip_target_modules else 'auto-detect'}")
        
        clip_lora_config = DeepEncoderLoRAConfig(
            enabled=self.config.get("clip_lora_enabled", True),
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            bias="none",
            target_modules=clip_target_modules,
        )
        
        # Initialize DeepEncoder
        runtime = DeepEncoderRuntime(
            sam_ckpt=self.config.get("sam_ckpt", None),
            auto_download_sam=self.config.get("auto_download_sam", True),
            device=str(self.device),
            dtype=self.config["deep_dtype"],
            openclip_pretrained=self.config["openclip_pretrained"],
            lora_config=clip_lora_config,
            freeze_clip_backbone_when_lora_enabled=True,
        )
        
        # Freeze SAM (already done in DeepEncoderRuntime)
        for p in runtime.sam.parameters():
            p.requires_grad = False
        runtime.sam.eval()
        
        # Load CLIP LoRA weights if they exist
        # Note: The LoRA adapters are already applied by DeepEncoderRuntime
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
        
        # Load projector weights
        proj_path = self.checkpoint_dir / "projector.pt"
        if proj_path.exists():
            print(f"[loader] Loading projector weights from {proj_path}")
            proj_state = torch.load(proj_path, map_location=self.device)
            runtime.projector.load_state_dict(proj_state)
        
        runtime.projector.eval()
        
        # Load Vision Adapter
        vision_adapter = VisionAdapter(2048, d_model).to(self.device)
        va_path = self.checkpoint_dir / "vision_adapter.pt"
        if va_path.exists():
            print(f"[loader] Loading vision adapter weights from {va_path}")
            va_state = torch.load(va_path, map_location=self.device)
            vision_adapter.load_state_dict(va_state)
        else:
            raise FileNotFoundError(f"Vision adapter weights not found: {va_path}")
        
        vision_adapter.eval()
        
        # Load Vision VAT
        vat_vision = VATVision(
            d_model=d_model,
            n_queries=self.config["vision_queries"],
            n_layers=self.config["vision_layers"],
            n_heads=self.config["vision_heads"],
            mlp_ratio=self.config["vision_mlp_ratio"],
            dropout=self.config["vision_dropout"],
            post_dropout=self.config["vision_post_dropout"],
            use_per_view_query=self.config["vision_per_view_query"],
        ).to(self.device)
        
        vat_vision_path = self.checkpoint_dir / "vat_vision.pt"
        if vat_vision_path.exists():
            print(f"[loader] Loading vision VAT weights from {vat_vision_path}")
            vat_vision_state = torch.load(vat_vision_path, map_location=self.device)
            vat_vision.load_state_dict(vat_vision_state)
        else:
            raise FileNotFoundError(f"Vision VAT weights not found: {vat_vision_path}")
        
        vat_vision.eval()
        
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
            # Try to load from config or detect from first BEV file
            c_in = self.config.get("c_in", 256)  # Default to 256
            print(f"[loader] Using c_in={c_in} (from config or default)")
        
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
