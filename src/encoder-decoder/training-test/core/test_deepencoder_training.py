"""
Verification script to check which DeepEncoder components are trainable.
This script analyzes the parameter setup to ensure correct training behavior.
"""

import sys
from pathlib import Path

# Add parent directories to path
root_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(root_dir / "src"))

import torch
from deepencoder import DeepEncoderRuntime, DeepEncoderLoRAConfig


def count_parameters(model, name="Model"):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    print(f"\n{name}:")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Frozen params:    {frozen:,}")
    print(f"  Trainable %:      {100 * trainable / total:.2f}%")
    
    return total, trainable, frozen


def verify_component_training_status(runtime):
    """Verify training status of each DeepEncoder component."""
    
    print("\n" + "="*80)
    print("DEEPENCODER COMPONENT TRAINING STATUS")
    print("="*80)
    
    # SAM - Should be FROZEN
    print("\n1. SAM (Segment Anything Model)")
    sam_trainable = sum(p.numel() for p in runtime.sam.parameters() if p.requires_grad)
    sam_total = sum(p.numel() for p in runtime.sam.parameters())
    
    if sam_trainable == 0:
        print(f"   ✓ CORRECTLY FROZEN: 0 / {sam_total:,} parameters trainable")
    else:
        print(f"   ✗ ERROR: {sam_trainable:,} / {sam_total:,} parameters are trainable!")
        print(f"   SAM should be completely frozen!")
    
    # CLIP - Should have trainable LoRA parameters
    print("\n2. CLIP Vision Transformer")
    clip_trainable = sum(p.numel() for p in runtime.clip_vit.parameters() if p.requires_grad)
    clip_total = sum(p.numel() for p in runtime.clip_vit.parameters())
    
    if clip_trainable > 0:
        print(f"   ✓ TRAINABLE: {clip_trainable:,} / {clip_total:,} parameters")
        print(f"   Trainable %: {100 * clip_trainable / clip_total:.2f}%")
        
        # Check for LoRA parameters
        lora_params = [n for n, p in runtime.clip_vit.named_parameters() if 'lora_' in n and p.requires_grad]
        if lora_params:
            print(f"   LoRA enabled: {len(lora_params)} LoRA parameter tensors found")
            print(f"   Sample LoRA params: {lora_params[:3]}")
        else:
            print(f"   WARNING: No LoRA parameters found, full CLIP training enabled")
    else:
        print(f"   ✗ ERROR: CLIP is completely frozen!")
    
    # Projector - Should be TRAINABLE
    print("\n3. Projector (CLIP+SAM -> 2048)")
    proj_trainable = sum(p.numel() for p in runtime.projector.parameters() if p.requires_grad)
    proj_total = sum(p.numel() for p in runtime.projector.parameters())
    
    if proj_trainable == proj_total:
        print(f"   ✓ FULLY TRAINABLE: {proj_trainable:,} / {proj_total:,} parameters")
    else:
        print(f"   ✗ ERROR: Only {proj_trainable:,} / {proj_total:,} parameters trainable!")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_trainable = sam_trainable + clip_trainable + proj_trainable
    total_params = sam_total + clip_total + proj_total
    
    print(f"\nTotal DeepEncoder parameters:     {total_params:,}")
    print(f"Total trainable parameters:       {total_trainable:,}")
    print(f"Total frozen parameters:          {total_params - total_trainable:,}")
    print(f"Trainable percentage:             {100 * total_trainable / total_params:.2f}%")
    
    # Verify expectations
    print("\n" + "-"*80)
    print("VERIFICATION CHECKS:")
    print("-"*80)
    
    checks_passed = 0
    checks_total = 3
    
    if sam_trainable == 0:
        print("✓ SAM is frozen (expected)")
        checks_passed += 1
    else:
        print("✗ SAM should be frozen!")
    
    if clip_trainable > 0:
        print("✓ CLIP has trainable parameters (expected)")
        checks_passed += 1
    else:
        print("✗ CLIP should have trainable parameters!")
    
    if proj_trainable == proj_total:
        print("✓ Projector is fully trainable (expected)")
        checks_passed += 1
    else:
        print("✗ Projector should be fully trainable!")
    
    print(f"\nVerification: {checks_passed}/{checks_total} checks passed")
    
    if checks_passed == checks_total:
        print("\n✓ ALL CHECKS PASSED - DeepEncoder training setup is correct!")
        return True
    else:
        print("\n✗ SOME CHECKS FAILED - Please review the setup!")
        return False


def main():
    """Main verification function."""
    
    print("="*80)
    print("DEEPENCODER TRAINING VERIFICATION")
    print("="*80)
    print("\nThis script verifies that DeepEncoder components are correctly")
    print("configured for training (SAM frozen, CLIP+Projector trainable).")
    
    # Create LoRA config (typical training setup)
    lora_config = DeepEncoderLoRAConfig(
        enabled=True,
        r=1,  # Small rank for testing
        lora_alpha=2,
        lora_dropout=0.05,
        target_modules=["qkv_proj"],  # Common target
    )
    
    print("\n" + "-"*80)
    print("CREATING DEEPENCODER RUNTIME")
    print("-"*80)
    print(f"\nLoRA Config:")
    print(f"  Enabled: {lora_config.enabled}")
    print(f"  Rank: {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Target modules: {lora_config.target_modules}")
    
    # Initialize runtime (will download models if needed)
    print("\nInitializing DeepEncoderRuntime...")
    print("(This may take a few moments for first-time model downloads)")
    
    try:
        runtime = DeepEncoderRuntime(
            device="cpu",  # Use CPU for verification
            dtype=torch.float32,
            lora_config=lora_config,
            freeze_clip_backbone_when_lora_enabled=True,
        )
        
        print("✓ Runtime initialized successfully")
        
        # Verify training status
        all_checks_passed = verify_component_training_status(runtime)
        
        # Additional check: ensure trainable_parameters() method returns correct params
        print("\n" + "="*80)
        print("TRAINABLE_PARAMETERS() METHOD CHECK")
        print("="*80)
        
        trainable_from_method = sum(p.numel() for p in runtime.trainable_parameters())
        trainable_direct = sum(p.numel() for p in runtime.clip_vit.parameters() if p.requires_grad)
        trainable_direct += sum(p.numel() for p in runtime.projector.parameters() if p.requires_grad)
        
        print(f"\nParameters from trainable_parameters(): {trainable_from_method:,}")
        print(f"Parameters counted directly:            {trainable_direct:,}")
        
        if trainable_from_method == trainable_direct:
            print("✓ trainable_parameters() method returns correct count")
        else:
            print("✗ trainable_parameters() method count mismatch!")
            all_checks_passed = False
        
        # Gradient flow test
        print("\n" + "="*80)
        print("GRADIENT FLOW TEST")
        print("="*80)
        
        print("\nTesting gradient flow through DeepEncoder components...")
        runtime.train()  # Set to training mode
        
        # Create dummy input
        dummy_img = torch.randn(1, 3, 1024, 1024, device=runtime.device, dtype=runtime.dtype)
        
        # Forward pass
        sam_feats = runtime._sam_features(dummy_img)
        clip_y = runtime.clip_vit(dummy_img, sam_feats)
        clip_tokens = clip_y[:, 1:, :]
        sam_tokens = sam_feats.flatten(2).permute(0, 2, 1)
        fused = torch.cat([clip_tokens, sam_tokens], dim=-1)
        vision_tokens = runtime.projector(fused)
        
        # Backward pass
        loss = vision_tokens.sum()
        loss.backward()
        
        # Check gradients
        sam_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in runtime.sam.parameters())
        clip_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in runtime.clip_vit.parameters() if p.requires_grad)
        proj_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in runtime.projector.parameters())
        
        print(f"\nGradient flow results:")
        if not sam_has_grad:
            print("  ✓ SAM: No gradients (correctly frozen)")
        else:
            print("  ✗ SAM: Has gradients (should be frozen!)")
            all_checks_passed = False
        
        if clip_has_grad:
            print("  ✓ CLIP: Has gradients (correctly trainable)")
        else:
            print("  ✗ CLIP: No gradients (should be trainable!)")
            all_checks_passed = False
        
        if proj_has_grad:
            print("  ✓ Projector: Has gradients (correctly trainable)")
        else:
            print("  ✗ Projector: No gradients (should be trainable!)")
            all_checks_passed = False
        
        # Final status
        print("\n" + "="*80)
        if all_checks_passed:
            print("SUCCESS: DeepEncoder is correctly configured for training!")
            print("="*80)
            print("\nDuring training:")
            print("  • SAM will remain frozen (no gradients)")
            print("  • CLIP will be trained via LoRA adapters")
            print("  • Projector will be fully trained")
            return 0
        else:
            print("FAILURE: Some checks did not pass!")
            print("="*80)
            print("\nPlease review the errors above and fix the configuration.")
            return 1
            
    except Exception as e:
        print(f"\n✗ ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
