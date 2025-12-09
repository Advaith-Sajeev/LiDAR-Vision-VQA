"""Vision VAT - View-Aware Transformer for vision tokens"""

import torch
import torch.nn as nn

from .vat_blocks import VATBlock


# Import debug logger
try:
    from ..utils import debug
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False


NUM_VIEWS = 6


class VATVision(nn.Module):
    """
    Compresses vision tokens via cross-attention and refines embeddings.
    
    Data flow:
    1. DeepEncoder: Raw images → [256, 2048] tokens per view (CLIP + SAM features)
    2. VisionAdapter: 6 views concatenated + projected → [1536, d_model]
    3. VATVision (this): Cross-attention compression → [n_queries, d_model]
    
    Architecture:
    - Uses learnable query tokens that cross-attend to input vision tokens
    - Token reduction: 1536 → n_queries tokens (any value allowed)
    - Dimension preserved: d_model → d_model (d_in == d_model in current setup)

    Shapes:
      Input:  [B, 1536, d_model] from VisionAdapter
      Queries: [n_queries, d_model] learnable parameters
      After VAT blocks: [B, n_queries, d_model]
      Output: [B, n_queries, d_model]
    
    Note: In the current architecture, d_in == d_model since VisionAdapter already
    projects DeepEncoder's 2048-dim output to d_model before passing to VATVision.
    The final projection layer is effectively a refinement MLP (d_model → d_model).
    
    Args:
        d_in: Input dimension from VisionAdapter (equals d_model, e.g., 896 for Qwen2.5-0.5B)
        d_model: Target output dimension (same as d_in in current architecture)
        n_input_tokens: Total tokens from VisionAdapter (6 views * 256 tokens = 1536)
        n_queries: Number of output query tokens (any positive integer)
        n_layers: Number of VAT transformer blocks
        n_heads: Number of attention heads per block
        mlp_ratio: MLP hidden dimension expansion ratio
        dropout: Dropout rate in transformer blocks
        post_dropout: Dropout rate in final projection
        use_per_view_query: Enable view-specific query embeddings (requires n_queries divisible by NUM_VIEWS)
        strict_per_view: If True, raise error when per-view not feasible; if False, auto-disable with warning
    """
    
    def __init__(
        self,
        d_in: int,  # Input dimension from VisionAdapter (equals d_model, e.g., 896)
        d_model: int,  # Output dimension (same as d_in in current architecture)
        n_input_tokens: int = 1536,  # Total tokens from VisionAdapter (6 views * 256)
        n_queries: int = 768,  # Number of output query tokens (any positive integer)
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.10,
        post_dropout: float = 0.10,
        use_per_view_query: bool = False,
        strict_per_view: bool = False,  # If True, raise error when per-view not feasible; if False, auto-disable
    ):
        super().__init__()
        
        # Store dimensions directly - no divisibility constraint needed
        assert n_queries > 0, f"n_queries must be positive, got {n_queries}"
        
        self.d_in = d_in
        self.d_model = d_model
        self.n_input_tokens = n_input_tokens
        self.n_queries = n_queries
        
        # Check if per-view queries are feasible
        per_view_feasible = (
            NUM_VIEWS > 0 and 
            self.n_queries >= NUM_VIEWS and 
            self.n_queries % NUM_VIEWS == 0
        )
        
        if use_per_view_query and not per_view_feasible:
            if strict_per_view:
                # Strict mode: raise error
                raise ValueError(
                    f"Per-view queries requested but not feasible: "
                    f"n_queries={self.n_queries}, NUM_VIEWS={NUM_VIEWS}. "
                    f"Either increase n_queries to be divisible by {NUM_VIEWS}, "
                    f"or set use_per_view_query=False, or set strict_per_view=False for auto-disable."
                )
            else:
                # Auto-disable mode: print warning and continue
                print(f"[VATVision] Warning: use_per_view_query=True requested but not feasible:")
                print(f"             n_queries={self.n_queries}, NUM_VIEWS={NUM_VIEWS}")
                print(f"             Automatically disabling per-view queries.")
                use_per_view_query = False
        
        self.use_per_view_query = use_per_view_query

        # Only compute nq_per_view when using per-view queries
        if self.use_per_view_query:
            self.nq_per_view = self.n_queries // NUM_VIEWS
        else:
            self.nq_per_view = 0  # not used

        # Learnable query tokens (in d_in space for cross-attention)
        self.query = nn.Parameter(torch.randn(self.n_queries, d_in) * 0.02)

        # Optional per-view query embeddings
        if self.use_per_view_query:
            self.view_query_embed = nn.Parameter(torch.zeros(NUM_VIEWS, d_in))
            nn.init.trunc_normal_(self.view_query_embed, std=0.02)
        else:
            self.view_query_embed = None

        # VAT blocks for cross-attention (operate in d_in space)
        d_ff = int(mlp_ratio * d_in)
        self.blocks = nn.ModuleList(
            [VATBlock(d_in, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        
        # Final processing in d_in space
        self.final_ln = nn.LayerNorm(d_in)
        self.post = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Dropout(post_dropout),
            nn.Linear(d_in, d_in),
        )
        
        # Projection layer: d_in -> d_model (dimension reduction)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Learnable output scale for matching LLM embedding magnitudes.
        # Initialized to 1.0 - the model learns optimal scaling during training.
        # This replaces the arbitrary fixed prefix_scale (e.g., 0.2) that was applied
        # externally after VAT processing. Since VAT outputs are already LayerNorm'd,
        # the learned scale adapts to match LLM text embedding statistics.
        self.output_scale = nn.Parameter(torch.ones(1))
        
        # Gradient checkpointing flag
        self._gradient_checkpointing = False
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for memory-efficient training.
        This trades compute for memory by recomputing activations during backward pass.
        """
        self._gradient_checkpointing = True
        for blk in self.blocks:
            blk.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        for blk in self.blocks:
            blk.gradient_checkpointing = False

    def forward(self, kv_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress vision tokens via cross-attention.
        
        Args:
            kv_tokens: Vision tokens from VisionAdapter [B, 1536, d_model]
                       (6 views * 256 tokens, already projected from 2048 to d_model)
            
        Returns:
            Compressed tokens [B, 768, d_model] (1536 tokens compressed to 768)
        """
        if DEBUG_AVAILABLE:
            debug.trace("vat_vision", "=" * 40)
            debug.trace("vat_vision", "Vision VAT Forward Pass")
            debug.trace("vat_vision", "=" * 40)
            debug.shape("vat_vision", "input_kv_tokens", kv_tokens)
        
        B, N, D = kv_tokens.shape
        
        if DEBUG_AVAILABLE:
            debug.debug("vat_vision", f"Input: B={B}, N={N}, D={D}")
        
        # Validate input shape
        assert N == self.n_input_tokens, \
            f"Expected {self.n_input_tokens} input tokens, got {N}"
        assert D == self.d_in, \
            f"Expected d_in={self.d_in}, got {D}"
        
        # Initialize query tokens [B, n_queries, d_in]
        if DEBUG_AVAILABLE:
            debug.start_timer("vat_vision", "query_init")
        
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, 768, 2048]
        
        if DEBUG_AVAILABLE:
            debug.shape("vat_vision", "initialized_queries", q)
            debug.debug("vat_vision", f"Target queries: {self.n_queries} (from {self.n_input_tokens} input tokens)")

        # Add per-view query embeddings if enabled
        if self.use_per_view_query and self.nq_per_view > 0:
            if DEBUG_AVAILABLE:
                debug.debug("vat_vision", f"Adding per-view embeddings ({self.nq_per_view} queries/view)")
            
            # Split queries into 6 view-specific chunks
            chunks = q.split(self.nq_per_view, dim=1)  # 6 chunks of [B, 128, 2048]
            
            # Add view-specific embedding to each chunk
            q = torch.cat(
                [ch + self.view_query_embed[k].view(1, 1, -1) 
                 for k, ch in enumerate(chunks)],
                dim=1,
            )  # [B, 768, 2048]
            
            if DEBUG_AVAILABLE:
                debug.shape("vat_vision", "queries_with_view_embed", q)
        
        if DEBUG_AVAILABLE:
            debug.end_timer("vat_vision", "query_init")

        # Apply VAT blocks (cross-attention: queries attend to KV tokens)
        # This reduces the number of tokens: 1536 -> 768
        if DEBUG_AVAILABLE:
            debug.start_timer("vat_vision", "vat_blocks")
            debug.debug("vat_vision", f"Processing {len(self.blocks)} VAT blocks")
        
        for i, blk in enumerate(self.blocks):
            if DEBUG_AVAILABLE:
                debug.trace("vat_vision", f"Block {i+1}/{len(self.blocks)}")
            q = blk(q, kv_tokens)  # [B, 768, 2048]
        
        if DEBUG_AVAILABLE:
            debug.shape("vat_vision", "after_blocks", q)
            debug.end_timer("vat_vision", "vat_blocks")
            
        # Final normalization and projection in d_in space
        if DEBUG_AVAILABLE:
            debug.start_timer("vat_vision", "final_processing")
        
        q = self.final_ln(q)
        q = self.post(q)  # [B, 768, 2048]
        
        if DEBUG_AVAILABLE:
            debug.shape("vat_vision", "after_post", q)
            debug.tensor_stats("vat_vision", "before_projection", q)
        
        # Project to target dimension: d_in -> d_model
        # This reduces the embedding dimension: 2048 -> d_model
        q = self.proj(q)  # [B, 768, d_model]
        
        # Apply learned output scale to match LLM embedding magnitudes.
        # This replaces external prefix_scale multiplication.
        q = q * self.output_scale
        
        if DEBUG_AVAILABLE:
            debug.shape("vat_vision", "output", q)
            debug.tensor_stats("vat_vision", "output", q)
            debug.debug("vat_vision", f"Dimension reduction: {self.d_in} → {self.d_model}")
            debug.debug("vat_vision", f"output_scale: {self.output_scale.item():.4f}")
            debug.end_timer("vat_vision", "final_processing")
            debug.trace("vat_vision", "Vision VAT Complete")
        
        return q