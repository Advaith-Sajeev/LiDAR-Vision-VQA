"""Vision VAT - View-Aware Transformer for vision tokens"""

import torch
import torch.nn as nn

from .vat_blocks import VATBlock


NUM_VIEWS = 6


class VATVision(nn.Module):
    """
    Learns n_queries vision prompts by attending to dense vision KV tokens.

    Shapes:
      kv_tokens: [B, N_img_tokens (~2400), d_model]
      output:    [B, n_queries (e.g.,1536), d_model]
    """
    
    def __init__(
        self,
        d_model: int,
        n_queries: int = 1536,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.10,
        post_dropout: float = 0.10,
        use_per_view_query: bool = True,
    ):
        super().__init__()
        assert n_queries % NUM_VIEWS == 0, "vision n_queries must be divisible by 6"
        self.d_model = d_model
        self.n_queries = n_queries
        self.nq_per_view = n_queries // NUM_VIEWS
        self.use_per_view_query = use_per_view_query

        self.query = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)

        if use_per_view_query:
            self.view_query_embed = nn.Parameter(
                torch.zeros(NUM_VIEWS, d_model), requires_grad=True
            )
            nn.init.zeros_(self.view_query_embed)
        else:
            self.view_query_embed = None

        d_ff = int(mlp_ratio * d_model)
        self.blocks = nn.ModuleList(
            [VATBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.post = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(post_dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, kv_tokens: torch.Tensor) -> torch.Tensor:
        """
        Process vision KV tokens through VAT.
        
        Args:
            kv_tokens: Vision tokens [B, N_img_tokens, d_model]
            
        Returns:
            Vision prompts [B, n_queries, d_model]
        """
        B, N, D = kv_tokens.shape
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B,nq,d]

        if self.use_per_view_query and self.nq_per_view > 0:
            chunks = q.split(self.nq_per_view, dim=1)  # 6 chunks
            q = torch.cat(
                [ch + self.view_query_embed[k].view(1, 1, -1) for k, ch in enumerate(chunks)],
                dim=1,
            )

        for blk in self.blocks:
            q = blk(q, kv_tokens)  # [B,nq,d]
            
        q = self.final_ln(q)
        q = self.post(q)  # [B,nq,d]
        return q
