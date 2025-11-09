"""VAT Transformer Block :: vat_blocks.py"""

import torch
import torch.nn as nn


class VATBlock(nn.Module):
    """
    Transformer block with SA + Cross-Attn (query attends to kv) + MLP
    
    Shapes:
      q:  [B, nq, d_model]
      kv: [B, N_kv, d_model]
      out:[B, nq, d_model]
    """
    
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float):
        super().__init__()
        self.sa_ln = nn.LayerNorm(d_model)
        self.sa = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ca_ln = nn.LayerNorm(d_model)
        self.ca = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.mlp_ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Self-attention
        q_norm = self.sa_ln(q)
        q = q + self.sa(q_norm, q_norm, q_norm, need_weights=False)[0]
        
        # Cross-attention
        q = q + self.ca(self.ca_ln(q), kv, kv, need_weights=False)[0]
        
        # MLP
        q = q + self.mlp(self.mlp_ln(q))
        
        return q
