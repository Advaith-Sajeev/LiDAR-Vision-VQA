"""Vision VAT - View-Aware Transformer for vision tokens"""

import torch
import torch.nn as nn

from .vat_blocks import VATBlock


NUM_VIEWS = 6


class VATVision(nn.Module):
    """
    Reduces vision tokens by half and projects embeddings to d_model dimension.
    
    Two-stage compression:
    1. Token reduction: Reduces number of tokens via cross-attention
    2. Dimension reduction: Projects token embeddings via MLP

    Shapes:
      Input:  [B, N_img_tokens (1536), d_in (2048)]
      After VAT: [B, n_queries (768), d_in (2048)]
      Output: [B, n_queries (768), d_model (e.g., 512)]
    """
    
    def __init__(
        self,
        d_in: int,  # Input dimension from VisionAdapter (e.g., 2048)
        d_model: int,  # Target output dimension (e.g., 512)
        n_input_tokens: int = 1536,  # Total tokens from VisionAdapter (6 * 256)
        compression_factor: int = 2,  # Reduce tokens by this factor
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.10,
        post_dropout: float = 0.10,
        use_per_view_query: bool = False,
    ):
        super().__init__()
        
        # Calculate n_queries from compression
        assert n_input_tokens % compression_factor == 0, \
            f"n_input_tokens ({n_input_tokens}) must be divisible by compression_factor ({compression_factor})"
        
        self.d_in = d_in
        self.d_model = d_model
        self.n_input_tokens = n_input_tokens
        self.compression_factor = compression_factor
        self.n_queries = n_input_tokens // compression_factor  # e.g., 1536 // 2 = 768
        
        self.use_per_view_query = use_per_view_query

        # Only enforce view divisibility when grouping by view
        if self.use_per_view_query:
            assert NUM_VIEWS > 0, "NUM_VIEWS must be > 0 when use_per_view_query=True"
            assert self.n_queries % NUM_VIEWS == 0, \
                f"n_queries ({self.n_queries}) must be divisible by NUM_VIEWS ({NUM_VIEWS})"
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

    def forward(self, kv_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress vision tokens and project to target dimension.
        
        Args:
            kv_tokens: Vision tokens from VisionAdapter [B, 1536, 2048]
            
        Returns:
            Compressed and projected tokens [B, 768, d_model]
        """
        B, N, D = kv_tokens.shape
        
        # Validate input shape
        assert N == self.n_input_tokens, \
            f"Expected {self.n_input_tokens} input tokens, got {N}"
        assert D == self.d_in, \
            f"Expected d_in={self.d_in}, got {D}"
        
        # Initialize query tokens [B, n_queries, d_in]
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, 768, 2048]

        # Add per-view query embeddings if enabled
        if self.use_per_view_query and self.nq_per_view > 0:
            # Split queries into 6 view-specific chunks
            chunks = q.split(self.nq_per_view, dim=1)  # 6 chunks of [B, 128, 2048]
            
            # Add view-specific embedding to each chunk
            q = torch.cat(
                [ch + self.view_query_embed[k].view(1, 1, -1) 
                 for k, ch in enumerate(chunks)],
                dim=1,
            )  # [B, 768, 2048]

        # Apply VAT blocks (cross-attention: queries attend to KV tokens)
        # This reduces the number of tokens: 1536 -> 768
        for blk in self.blocks:
            q = blk(q, kv_tokens)  # [B, 768, 2048]
            
        # Final normalization and projection in d_in space
        q = self.final_ln(q)
        q = self.post(q)  # [B, 768, 2048]
        
        # Project to target dimension: d_in -> d_model
        # This reduces the embedding dimension: 2048 -> d_model
        q = self.proj(q)  # [B, 768, d_model]
        
        return q