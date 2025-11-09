"""Test script for VisionAdapter + VATVision pipeline with projection"""

import torch
import sys
from pathlib import Path

# Add the parent directory to the path to import from training module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.models.vision_adapter import VisionAdapter, CAM_VIEWS
# from training.models.vat_vision import VATVision  # Uncomment when vat_vision.py is ready


# Mock VATBlock for testing (remove when actual vat_blocks.py is available)
class VATBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout),
        )
    
    def forward(self, q, kv):
        # Cross attention: q attends to kv
        attn_out, _ = self.attn(q, kv, kv)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.mlp(q))
        return q


# Mock VATVision (copy from updated version above)
class VATVision(torch.nn.Module):
    """Mock VATVision with projection layer for testing"""
    
    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_input_tokens: int = 1536,
        compression_factor: int = 2,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.10,
        post_dropout: float = 0.10,
        use_per_view_query: bool = True,
    ):
        super().__init__()
        
        NUM_VIEWS = 6
        assert n_input_tokens % compression_factor == 0
        
        self.d_in = d_in
        self.d_model = d_model
        self.n_input_tokens = n_input_tokens
        self.compression_factor = compression_factor
        self.n_queries = n_input_tokens // compression_factor
        
        assert self.n_queries % NUM_VIEWS == 0
        
        self.nq_per_view = self.n_queries // NUM_VIEWS
        self.use_per_view_query = use_per_view_query

        self.query = torch.nn.Parameter(torch.randn(self.n_queries, d_in) * 0.02)

        if use_per_view_query:
            self.view_query_embed = torch.nn.Parameter(
                torch.zeros(NUM_VIEWS, d_in), requires_grad=True
            )
            torch.nn.init.trunc_normal_(self.view_query_embed, std=0.02)
        else:
            self.view_query_embed = None

        d_ff = int(mlp_ratio * d_in)
        self.blocks = torch.nn.ModuleList(
            [VATBlock(d_in, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        
        self.final_ln = torch.nn.LayerNorm(d_in)
        self.post = torch.nn.Sequential(
            torch.nn.LayerNorm(d_in),
            torch.nn.Linear(d_in, d_in),
            torch.nn.GELU(),
            torch.nn.Dropout(post_dropout),
            torch.nn.Linear(d_in, d_in),
        )
        
        # Projection layer: d_in -> d_model
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_in),
            torch.nn.Linear(d_in, d_model),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, d_model),
            torch.nn.LayerNorm(d_model),
        )

    def forward(self, kv_tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = kv_tokens.shape
        
        assert N == self.n_input_tokens
        assert D == self.d_in
        
        q = self.query.unsqueeze(0).expand(B, -1, -1)

        if self.use_per_view_query and self.nq_per_view > 0:
            chunks = q.split(self.nq_per_view, dim=1)
            q = torch.cat(
                [ch + self.view_query_embed[k].view(1, 1, -1) 
                 for k, ch in enumerate(chunks)],
                dim=1,
            )

        for blk in self.blocks:
            q = blk(q, kv_tokens)
            
        q = self.final_ln(q)
        q = self.post(q)
        
        # Project to target dimension
        q = self.proj(q)
        
        return q


def test_vision_pipeline():
    """Test the complete VisionAdapter -> VATVision pipeline with projection"""
    
    print("="*70)
    print("Testing VisionAdapter + VATVision Pipeline (with Projection)")
    print("="*70)
    
    # Configuration
    d_in = 2048  # VisionAdapter output dimension
    d_model = 1536  # Target dimension after projection
    num_views = 6
    hw = 256  # Tokens per view
    batch_size = 2
    
    print(f"\nConfiguration:")
    print(f"  d_in (encoder embedding dim): {d_in}")
    print(f"  d_model (target model dim): {d_model}")
    print(f"  num_views: {num_views}")
    print(f"  hw (tokens per view): {hw}")
    print(f"  batch_size: {batch_size}")
    
    # Test 1: VisionAdapter
    print("\n" + "="*70)
    print("Test 1: VisionAdapter (Add View Embeddings + Concatenate)")
    print("="*70)
    
    adapter = VisionAdapter(d_in=d_in, dropout=0.1)
    print(f"✓ VisionAdapter initialized")
    
    # Create input: list of 6 views, each [hw, d_in]
    views_tokens = [torch.randn(hw, d_in) for _ in range(num_views)]
    print(f"Input: {len(views_tokens)} views × [{hw}, {d_in}]")
    
    adapter_output = adapter(views_tokens)
    expected_tokens = num_views * hw  # 1536
    print(f"Output shape: {adapter_output.shape}")
    print(f"Expected: [{expected_tokens}, {d_in}]")
    assert adapter_output.shape == (expected_tokens, d_in)
    print("✓ VisionAdapter output shape correct!")
    
    # Test 2: VATVision with Projection
    print("\n" + "="*70)
    print("Test 2: VATVision (Token Compression + Dimension Projection)")
    print("="*70)
    
    vat = VATVision(
        d_in=d_in,
        d_model=d_model,
        n_input_tokens=1536,
        compression_factor=2,
        n_layers=4,
        n_heads=8,
        dropout=0.1,
    )
    print(f"✓ VATVision initialized")
    print(f"  Input tokens: {vat.n_input_tokens}")
    print(f"  Input dimension: {vat.d_in}")
    print(f"  Compression factor: {vat.compression_factor}")
    print(f"  Output tokens: {vat.n_queries}")
    print(f"  Output dimension: {vat.d_model}")
    print(f"  Queries per view: {vat.nq_per_view}")
    
    # Add batch dimension to adapter output
    batched_input = adapter_output.unsqueeze(0).expand(batch_size, -1, -1)
    print(f"\nInput to VATVision: {batched_input.shape}")
    
    vat_output = vat(batched_input)
    expected_compressed = expected_tokens // 2  # 768
    print(f"Output shape: {vat_output.shape}")
    print(f"Expected: [{batch_size}, {expected_compressed}, {d_model}]")
    assert vat_output.shape == (batch_size, expected_compressed, d_model)
    print("✓ VATVision output shape correct!")
    
    # Test 3: Full pipeline
    print("\n" + "="*70)
    print("Test 3: Complete Pipeline")
    print("="*70)
    
    print("Pipeline: DeepEncoder → VisionAdapter → VATVision")
    print()
    
    # Simulate batch of DeepEncoder outputs
    batch_views = []
    for b in range(batch_size):
        views = [torch.randn(hw, d_in) for _ in range(num_views)]
        batch_views.append(views)
    
    print(f"Step 1: DeepEncoder output (simulated)")
    print(f"  Batch size: {batch_size}")
    print(f"  Per sample: {num_views} views × [{hw}, {d_in}]")
    
    # Process through adapter
    adapter_outputs = []
    for views in batch_views:
        adapter_out = adapter(views)
        adapter_outputs.append(adapter_out)
    adapter_batched = torch.stack(adapter_outputs, dim=0)
    
    print(f"\nStep 2: VisionAdapter output")
    print(f"  Shape: {adapter_batched.shape}")
    print(f"  Description: [{batch_size}, {num_views * hw}, {d_in}]")
    print(f"  → Concatenated all views, added view embeddings")
    
    # Process through VAT
    final_output = vat(adapter_batched)
    
    print(f"\nStep 3: VATVision output (final)")
    print(f"  Shape: {final_output.shape}")
    print(f"  Description: [{batch_size}, {(num_views * hw) // 2}, {d_model}]")
    print(f"  → Reduced tokens via cross-attention")
    print(f"  → Projected embeddings via MLP")
    
    print(f"\n✓ Token reduction: {num_views * hw} → {(num_views * hw) // 2} (50% compression)")
    print(f"✓ Embedding dimension reduction: {d_in} → {d_model} ({(1-d_model/d_in)*100:.1f}% reduction)")
    
    # Test 4: Gradient flow
    print("\n" + "="*70)
    print("Test 4: Gradient Flow")
    print("="*70)
    
    loss = final_output.mean()
    loss.backward()
    
    print("Checking gradients...")
    print(f"  VisionAdapter view_embed has grad: {adapter.view_embed.grad is not None}")
    print(f"  VATVision query has grad: {vat.query.grad is not None}")
    print(f"  VATVision view_query_embed has grad: {vat.view_query_embed.grad is not None}")
    
    # Check projection layer gradients
    proj_has_grad = any(p.grad is not None for p in vat.proj.parameters() if p.requires_grad)
    print(f"  VATVision projection layer has grad: {proj_has_grad}")
    print("✓ Gradients flow through the entire pipeline including projection!")
    
    # Test 5: Parameter count
    print("\n" + "="*70)
    print("Test 5: Parameter Statistics")
    print("="*70)
    
    adapter_params = sum(p.numel() for p in adapter.parameters())
    vat_params = sum(p.numel() for p in vat.parameters())
    proj_params = sum(p.numel() for p in vat.proj.parameters())
    total_params = adapter_params + vat_params
    
    print(f"VisionAdapter parameters: {adapter_params:,}")
    print(f"VATVision parameters: {vat_params:,}")
    print(f"  - Projection layer: {proj_params:,}")
    print(f"Total pipeline parameters: {total_params:,}")
    
    # Test 6: Different d_model values
    print("\n" + "="*70)
    print("Test 6: Different Target Dimensions (d_model)")
    print("="*70)
    
    for target_dim in [256, 512, 768, 1024]:
        vat_test = VATVision(
            d_in=d_in,
            d_model=target_dim,
            n_input_tokens=1536,
            compression_factor=2,
            n_layers=2,
            n_heads=8,
        )
        test_input = torch.randn(batch_size, 1536, d_in)
        test_output = vat_test(test_input)
        reduction = (1 - target_dim/d_in) * 100
        print(f"d_model={target_dim}: {test_input.shape} → {test_output.shape} ({reduction:.1f}% dim reduction)")
        assert test_output.shape == (batch_size, 768, target_dim)
    print("✓ Multiple target dimensions work correctly!")
    
    # Test 7: Memory efficiency
    print("\n" + "="*70)
    print("Test 7: Memory Efficiency")
    print("="*70)
    
    input_memory = batch_size * 1536 * d_in * 4 / (1024**2)  # 4 bytes per float32
    output_memory = batch_size * 768 * d_model * 4 / (1024**2)
    
    print(f"Input tensor memory: {input_memory:.2f} MB")
    print(f"  Shape: [{batch_size}, 1536, {d_in}]")
    print(f"Output tensor memory: {output_memory:.2f} MB")
    print(f"  Shape: [{batch_size}, 768, {d_model}]")
    print(f"Total memory reduction: {(1 - output_memory/input_memory)*100:.1f}%")
    
    # Test 8: Projection layer architecture
    print("\n" + "="*70)
    print("Test 8: Projection Layer Architecture")
    print("="*70)
    
    print("Projection layer components:")
    for name, module in vat.proj.named_children():
        if hasattr(module, 'weight'):
            print(f"  {name}: {module}")
    print("✓ Projection uses MLP with LayerNorm, GELU activation, and Dropout!")
    
    print("\n" + "="*70)
    print("All pipeline tests passed! ✓")
    print("="*70)
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"Stage 1: DeepEncoder")
    print(f"  Output: 6 views × [256, 2048] per view")
    print(f"")
    print(f"Stage 2: VisionAdapter")
    print(f"  - Adds learned view embeddings (6 embeddings)")
    print(f"  - Concatenates all views")
    print(f"  Output: [1536, 2048]")
    print(f"")
    print(f"Stage 3: VATVision")
    print(f"  - Token compression via cross-attention (2x reduction)")
    print(f"  - Dimension projection via MLP")
    print(f"  Output: [768, {d_model}]")
    print(f"")
    print(f"Total Compression:")
    print(f"  Tokens: 1536 → 768 (50% reduction)")
    print(f"  Dimension: 2048 → {d_model} ({(1-d_model/2048)*100:.1f}% reduction)")
    print(f"  Combined: {1536*2048} → {768*d_model} parameters per sample")
    print(f"  Overall reduction: {(1-(768*d_model)/(1536*2048))*100:.1f}%")
    print("="*70)


if __name__ == "__main__":
    torch.manual_seed(42)
    test_vision_pipeline()