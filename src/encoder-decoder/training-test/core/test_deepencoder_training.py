"""
Tests for DeepEncoder component training setup.
Verifies that SAM is frozen, CLIP has LoRA, and projector is trainable.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add src/ directory to path (contains deepencoder package)
src_dir = Path(__file__).parent.parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Skip entire module if deepencoder not available
pytest.importorskip("deepencoder")

from deepencoder import DeepEncoderRuntime, DeepEncoderLoRAConfig


# ---------- Lightweight stand-ins for heavy models ----------

class DummySAM(nn.Module):
    """Minimal SAM-like encoder for testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, kernel_size=1)
        # Add net_2 and net_3 for the compression head (these should be trainable)
        self.net_2 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.net_3 = nn.Conv2d(1024, 1024, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = F.adaptive_avg_pool2d(y, (16, 16))
        return y


class DummyCLIP(nn.Module):
    """Minimal CLIP-ViT-like head for testing."""
    def __init__(self):
        super().__init__()
        self.embed_dim = 1024
        self.embeddings = nn.Module()
        self.embeddings.class_embedding = nn.Parameter(torch.zeros(1024))
        self.embeddings.position_embedding = nn.Embedding(257, 1024)
        self.embeddings.num_positions = 257
        self.embeddings.patch_embedding = nn.Conv2d(3, 1024, kernel_size=14, stride=14)
        
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList()
        
        # Add qkv_proj for LoRA targeting
        self.qkv_proj = nn.Linear(1024, 3072)

    def forward(self, x: torch.Tensor, patch_embeds: torch.Tensor) -> torch.Tensor:
        B, C, H, W = patch_embeds.shape
        tokens = patch_embeds.flatten(2).transpose(1, 2)
        cls = torch.zeros(B, 1, self.embed_dim, device=x.device, dtype=x.dtype)
        return torch.cat([cls, tokens], dim=1)


@pytest.fixture(scope="module")
def deepencoder_runtime():
    """Create DeepEncoder runtime with LoRA config for testing (with mocked models)."""
    import deepencoder.deepencoder_infer as dei
    
    # Mock download and model builders
    with patch.object(dei, 'download_sam_if_needed', return_value="/tmp/dummy_sam.pth"):
        with patch.object(dei, 'build_sam_vit_b', return_value=DummySAM()):
            with patch.object(dei, 'build_clip_l', return_value=DummyCLIP()):
                with patch.object(dei, 'load_openclip_vitl14_into_vitmodel'):
                    lora_config = DeepEncoderLoRAConfig(
                        enabled=True,
                        r=1,
                        lora_alpha=2,
                        lora_dropout=0.05,
                        target_modules=["qkv_proj"],
                    )
                    
                    runtime = DeepEncoderRuntime(
                        device="cpu",
                        dtype=torch.float32,
                        lora_config=lora_config,
                        freeze_clip_backbone_when_lora_enabled=True,
                    )
    
    return runtime


class TestDeepEncoderTrainingSetup:
    """Test DeepEncoder component training configuration."""
    
    def test_sam_is_frozen(self, deepencoder_runtime):
        """SAM should have only net_2 and net_3 trainable (compression head)."""
        # Identify trainable parameters in SAM
        trainable_params = [(n, p.numel()) for n, p in deepencoder_runtime.sam.named_parameters() if p.requires_grad]
        
        print("\n" + "="*80)
        print("SAM TRAINABLE PARAMETERS:")
        print("="*80)
        for name, count in trainable_params:
            print(f"  {name}: {count:,} parameters")
        
        total_trainable = sum(count for _, count in trainable_params)
        print(f"\nTotal trainable: {total_trainable:,} parameters")
        print("="*80)
        
        # Verify only net_2 and net_3 are trainable (the compression head)
        trainable_names = [n for n, _ in trainable_params]
        
        assert all(n.startswith("net_2") or n.startswith("net_3") for n in trainable_names), \
            f"SAM should only have net_2 and net_3 trainable, but found: {trainable_names}"
        
        # Verify net_2 and net_3 exist and are trainable
        has_net2 = any(n.startswith("net_2") for n in trainable_names)
        has_net3 = any(n.startswith("net_3") for n in trainable_names)
        
        assert has_net2, "SAM should have net_2 (compression head) trainable"
        assert has_net3, "SAM should have net_3 (compression head) trainable"
        
        print(f"✓ Verified: Only net_2 and net_3 are trainable ({total_trainable:,} parameters)")
        print("  These are the DeepEncoder/VARY compression head layers.")
    
    def test_clip_has_trainable_params(self, deepencoder_runtime):
        """CLIP should have trainable LoRA parameters."""
        clip_trainable = sum(p.numel() for p in deepencoder_runtime.clip_vit.parameters() if p.requires_grad)
        assert clip_trainable > 0, "CLIP should have trainable parameters"
    
    def test_clip_has_lora_params(self, deepencoder_runtime):
        """CLIP should have LoRA parameters when LoRA is enabled."""
        lora_params = [n for n, p in deepencoder_runtime.clip_vit.named_parameters() 
                       if 'lora_' in n and p.requires_grad]
        assert len(lora_params) > 0, "CLIP should have LoRA parameters"
    
    def test_projector_is_trainable(self, deepencoder_runtime):
        """Projector should be fully trainable."""
        proj_trainable = sum(p.numel() for p in deepencoder_runtime.projector.parameters() if p.requires_grad)
        proj_total = sum(p.numel() for p in deepencoder_runtime.projector.parameters())
        assert proj_trainable == proj_total, "Projector should be fully trainable"
    
    def test_trainable_parameters_method(self, deepencoder_runtime):
        """trainable_parameters() should return correct count."""
        trainable_from_method = sum(p.numel() for p in deepencoder_runtime.trainable_parameters())
        trainable_direct = sum(p.numel() for p in deepencoder_runtime.clip_vit.parameters() if p.requires_grad)
        trainable_direct += sum(p.numel() for p in deepencoder_runtime.projector.parameters() if p.requires_grad)
        assert trainable_from_method == trainable_direct, "trainable_parameters() count mismatch"
    
    def test_gradient_flow_sam_frozen(self, deepencoder_runtime):
        """SAM should not receive gradients."""
        deepencoder_runtime.train()
        
        # Forward pass
        dummy_img = torch.randn(1, 3, 1024, 1024, device=deepencoder_runtime.device, dtype=deepencoder_runtime.dtype)
        sam_feats = deepencoder_runtime._sam_features(dummy_img)
        
        # Access base model if wrapped with PEFT
        clip_model = deepencoder_runtime.clip_vit.base_model.model if hasattr(deepencoder_runtime.clip_vit, 'base_model') else deepencoder_runtime.clip_vit
        clip_y = clip_model(dummy_img, sam_feats)
        
        clip_tokens = clip_y[:, 1:, :]
        sam_tokens = sam_feats.flatten(2).permute(0, 2, 1)
        fused = torch.cat([clip_tokens, sam_tokens], dim=-1)
        vision_tokens = deepencoder_runtime.projector(fused)
        
        # Backward pass
        loss = vision_tokens.sum()
        loss.backward()
        
        # Check SAM backbone has no gradients (net_2 and net_3 are intentionally trainable)
        sam_backbone_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for n, p in deepencoder_runtime.sam.named_parameters()
            if not (n.startswith("net_2") or n.startswith("net_3"))
        )
        assert not sam_backbone_has_grad, "SAM backbone should not receive gradients (net_2/net_3 are trainable)"
    
    def test_gradient_flow_clip_trainable(self, deepencoder_runtime):
        """CLIP should receive gradients on trainable parameters."""
        deepencoder_runtime.train()
        
        # Forward pass
        dummy_img = torch.randn(1, 3, 1024, 1024, device=deepencoder_runtime.device, dtype=deepencoder_runtime.dtype)
        sam_feats = deepencoder_runtime._sam_features(dummy_img)
        
        # Access base model if wrapped with PEFT
        clip_model = deepencoder_runtime.clip_vit.base_model.model if hasattr(deepencoder_runtime.clip_vit, 'base_model') else deepencoder_runtime.clip_vit
        clip_y = clip_model(dummy_img, sam_feats)
        
        clip_tokens = clip_y[:, 1:, :]
        sam_tokens = sam_feats.flatten(2).permute(0, 2, 1)
        fused = torch.cat([clip_tokens, sam_tokens], dim=-1)
        vision_tokens = deepencoder_runtime.projector(fused)
        
        # Backward pass
        loss = vision_tokens.sum()
        loss.backward()
        
        # Check CLIP has gradients
        clip_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                           for p in deepencoder_runtime.clip_vit.parameters() if p.requires_grad)
        assert clip_has_grad, "CLIP should receive gradients on trainable parameters"
    
    def test_gradient_flow_projector_trainable(self, deepencoder_runtime):
        """Projector should receive gradients."""
        deepencoder_runtime.train()
        
        # Forward pass
        dummy_img = torch.randn(1, 3, 1024, 1024, device=deepencoder_runtime.device, dtype=deepencoder_runtime.dtype)
        sam_feats = deepencoder_runtime._sam_features(dummy_img)
        
        # Access base model if wrapped with PEFT
        clip_model = deepencoder_runtime.clip_vit.base_model.model if hasattr(deepencoder_runtime.clip_vit, 'base_model') else deepencoder_runtime.clip_vit
        clip_y = clip_model(dummy_img, sam_feats)
        
        clip_tokens = clip_y[:, 1:, :]
        sam_tokens = sam_feats.flatten(2).permute(0, 2, 1)
        fused = torch.cat([clip_tokens, sam_tokens], dim=-1)
        vision_tokens = deepencoder_runtime.projector(fused)
        
        # Backward pass
        loss = vision_tokens.sum()
        loss.backward()
        
        # Check projector has gradients
        proj_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                           for p in deepencoder_runtime.projector.parameters())
        assert proj_has_grad, "Projector should receive gradients"


def main():
    """Main function for standalone script execution."""
    # Run verification when executed as script
    print("="*80)
    print("DEEPENCODER TRAINING VERIFICATION")
    print("="*80)
    print("\nRunning pytest tests...")
    
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v"],
        capture_output=False
    )
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
