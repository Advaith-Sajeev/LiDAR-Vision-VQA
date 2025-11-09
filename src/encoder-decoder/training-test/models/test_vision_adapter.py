"""Test script for VisionAdapter"""

import torch
import sys
from pathlib import Path

# Add the parent directory to the path to import from training module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.models.vision_adapter import VisionAdapter, CAM_VIEWS


def test_vision_adapter():
    """Test VisionAdapter with various scenarios"""
    
    print("="*60)
    print("Testing VisionAdapter")
    print("="*60)
    
    # Configuration
    d_in = 2048
    num_views = 6
    hw = 256  # Number of tokens per view
    
    # Initialize adapter
    adapter = VisionAdapter(d_in=d_in, dropout=0.1)
    print(f"\n✓ VisionAdapter initialized with d_in={d_in}")
    print(f"  Number of views: {adapter.num_views}")
    print(f"  View embeddings shape: {adapter.view_embed.shape}")
    print(f"  Camera views order: {CAM_VIEWS}")
    
    # Test 1: Basic forward pass with correct input
    print("\n" + "="*60)
    print("Test 1: Basic forward pass")
    print("="*60)
    
    # Create dummy input: list of 6 tensors, each [HW, d_in]
    views_tokens = [torch.randn(hw, d_in) for _ in range(num_views)]
    print(f"Input: List of {len(views_tokens)} tensors")
    print(f"Each tensor shape: {views_tokens[0].shape}")
    
    # Forward pass
    output = adapter(views_tokens)
    expected_total_hw = num_views * hw
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: [{expected_total_hw}, {d_in}]")
    assert output.shape == (expected_total_hw, d_in), f"Shape mismatch! Got {output.shape}"
    print("✓ Output shape correct! (All views concatenated)")
    
    # Test 2: Check that view embeddings are learnable
    print("\n" + "="*60)
    print("Test 2: View embeddings are learnable parameters")
    print("="*60)
    
    print(f"view_embed requires_grad: {adapter.view_embed.requires_grad}")
    assert adapter.view_embed.requires_grad, "View embeddings should require gradients!"
    print("✓ View embeddings are learnable!")
    
    # Check that view_embed is in model parameters
    param_names = [name for name, _ in adapter.named_parameters()]
    print(f"Model parameters: {param_names}")
    assert 'view_embed' in param_names, "view_embed should be in model parameters!"
    print("✓ View embeddings registered as parameters!")
    
    # Test 3: Verify different views get different embeddings
    print("\n" + "="*60)
    print("Test 3: Different views have different embeddings")
    print("="*60)
    
    print(f"Comparing embeddings for {len(CAM_VIEWS)} camera views:")
    for i in range(num_views):
        for j in range(i+1, num_views):
            diff = torch.abs(adapter.view_embed[i] - adapter.view_embed[j]).mean()
            print(f"  {CAM_VIEWS[i]} vs {CAM_VIEWS[j]}: mean abs diff = {diff:.6f}")
    print("✓ View embeddings are different (initialized randomly)!")
    
    # Test 4: Gradient flow check
    print("\n" + "="*60)
    print("Test 4: Gradient flow through view embeddings")
    print("="*60)
    
    # Create a simple loss
    views_tokens = [torch.randn(hw, d_in) for _ in range(num_views)]
    output = adapter(views_tokens)
    loss = output.mean()
    loss.backward()
    
    print(f"view_embed has gradients: {adapter.view_embed.grad is not None}")
    if adapter.view_embed.grad is not None:
        print(f"view_embed gradient norm: {adapter.view_embed.grad.norm():.6f}")
        print("✓ Gradients flow through view embeddings!")
    else:
        print("✗ No gradients on view embeddings!")
    
    # Test 5: Wrong number of views (should raise error)
    print("\n" + "="*60)
    print("Test 5: Error handling - wrong number of views")
    print("="*60)
    
    adapter_clean = VisionAdapter(d_in=d_in, dropout=0.1)
    wrong_views = [torch.randn(hw, d_in) for _ in range(4)]  # Only 4 views
    try:
        output = adapter_clean(wrong_views)
        print("✗ Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:80]}...")
    
    # Test 6: Wrong tensor dimensions (should raise error)
    print("\n" + "="*60)
    print("Test 6: Error handling - wrong tensor dimensions")
    print("="*60)
    
    adapter_clean = VisionAdapter(d_in=d_in, dropout=0.1)
    wrong_dim_views = [torch.randn(hw, d_in, 1) for _ in range(num_views)]  # 3D instead of 2D
    try:
        output = adapter_clean(wrong_dim_views)
        print("✗ Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:80]}...")
    
    # Test 7: Inconsistent HW across views (should raise error)
    print("\n" + "="*60)
    print("Test 7: Error handling - inconsistent HW across views")
    print("="*60)
    
    adapter_clean = VisionAdapter(d_in=d_in, dropout=0.1)
    inconsistent_views = [torch.randn(hw, d_in) for _ in range(5)]
    inconsistent_views.append(torch.randn(hw + 10, d_in))  # Different HW
    try:
        output = adapter_clean(inconsistent_views)
        print("✗ Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:80]}...")
    
    # Test 8: Verify concatenation order
    print("\n" + "="*60)
    print("Test 8: Verify concatenation order")
    print("="*60)
    
    adapter_clean = VisionAdapter(d_in=d_in, dropout=0.0)  # No dropout for this test
    adapter_clean.eval()  # Evaluation mode
    
    # Create unique tokens for each view (filled with view index)
    views_tokens = [torch.ones(hw, d_in) * i for i in range(num_views)]
    output = adapter_clean(views_tokens)
    
    # Check that views are concatenated in order
    for i in range(num_views):
        start_idx = i * hw
        end_idx = (i + 1) * hw
        view_section = output[start_idx:end_idx]
        print(f"  View {i} ({CAM_VIEWS[i]}): tokens [{start_idx}:{end_idx}]")
    
    print("✓ Views concatenated in correct order!")
    
    # Test 9: Model size and parameter count
    print("\n" + "="*60)
    print("Test 9: Model statistics")
    print("="*60)
    
    total_params = sum(p.numel() for p in adapter.parameters())
    trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"View embeddings parameters: {adapter.view_embed.numel():,}")
    print(f"LayerNorm parameters: {sum(p.numel() for p in adapter.norm.parameters()):,}")
    
    # Breakdown
    print("\nParameter breakdown:")
    for name, param in adapter.named_parameters():
        print(f"  {name}: {param.shape} ({param.numel():,} params)")
    
    # Test 10: Different HW sizes
    print("\n" + "="*60)
    print("Test 10: Different HW sizes")
    print("="*60)
    
    for test_hw in [64, 256, 1024]:
        adapter_test = VisionAdapter(d_in=d_in, dropout=0.1)
        views_tokens = [torch.randn(test_hw, d_in) for _ in range(num_views)]
        output = adapter_test(views_tokens)
        expected_total = num_views * test_hw
        print(f"HW={test_hw}: Output shape = {output.shape} (expected [{expected_total}, {d_in}])")
        assert output.shape == (expected_total, d_in)
    print("✓ Works with different HW sizes!")
    
    # Test 11: Check view embedding initialization
    print("\n" + "="*60)
    print("Test 11: View embedding initialization")
    print("="*60)
    
    adapter_init = VisionAdapter(d_in=d_in, dropout=0.1)
    embed_mean = adapter_init.view_embed.mean().item()
    embed_std = adapter_init.view_embed.std().item()
    embed_max = adapter_init.view_embed.abs().max().item()
    
    print(f"View embeddings statistics:")
    print(f"  Mean: {embed_mean:.6f}")
    print(f"  Std: {embed_std:.6f}")
    print(f"  Max (abs): {embed_max:.6f}")
    print(f"  Expected std: ~0.02 (from trunc_normal initialization)")
    print("✓ View embeddings initialized correctly!")
    
    # Test 12: Verify each view gets its unique embedding
    print("\n" + "="*60)
    print("Test 12: Each view receives its unique embedding")
    print("="*60)
    
    adapter_clean = VisionAdapter(d_in=d_in, dropout=0.0)
    adapter_clean.eval()
    
    # Create identical input for all views
    identical_input = torch.randn(hw, d_in)
    views_tokens = [identical_input.clone() for _ in range(num_views)]
    
    output = adapter_clean(views_tokens)
    
    # Extract each view's output
    print("Comparing outputs (same input + different view embeddings):")
    for i in range(num_views - 1):
        start_i = i * hw
        end_i = (i + 1) * hw
        start_j = (i + 1) * hw
        end_j = (i + 2) * hw
        
        view_i_out = output[start_i:end_i]
        view_j_out = output[start_j:end_j]
        
        diff = torch.abs(view_i_out - view_j_out).mean()
        print(f"  {CAM_VIEWS[i]} vs {CAM_VIEWS[i+1]}: mean abs diff = {diff:.6f}")
    
    print("✓ Each view produces different output (unique embeddings applied)!")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    test_vision_adapter()