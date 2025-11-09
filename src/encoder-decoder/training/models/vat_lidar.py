"""LiDAR VAT - View-Aware Transformer for BEV features"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .vat_blocks import VATBlock


NUM_VIEWS = 6


class VATLiDAR(nn.Module):
    """
    VAT over BEV tokens (C,H,W) → Nq query tokens.
    Adds geometric PE + unconditional 6-view embeddings to KV (projected BEV tokens).

    Shapes:
      bev:            [B, C, H, W]
      after proj:     [B, H*W, d_model]        # tokens (KV)
      learnable q:    [1, n_queries, d_model]  # broadcast to B
      output:         [B, n_queries, d_model]
    """
    
    def __init__(
        self,
        c_in: int,
        d_model: int,
        n_queries: int = 576,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.10,
        post_dropout: float = 0.10,
    ):
        super().__init__()
        assert n_queries % NUM_VIEWS == 0, "n_queries must be divisible by 6"
        self.d_model = d_model
        self.nq_per_view = n_queries // NUM_VIEWS

        self.refine = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in), nn.GELU()
        )
        self.proj = nn.Conv2d(c_in, d_model, 1, bias=True)
        self.norm_tokens = nn.LayerNorm(d_model)

        self.geo_mlp = nn.Sequential(
            nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        self.view_embed = nn.Parameter(torch.zeros(NUM_VIEWS, d_model), requires_grad=True)
        nn.init.zeros_(self.view_embed)

        self.query = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)

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

        self._cache: Dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _grid(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate geometric features and sector IDs for spatial grid."""
        key = (H, W, device)
        if key in self._cache:
            return self._cache[key]
            
        yv, xv = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        r = torch.clamp((xv**2 + yv**2).sqrt(), 0, 1)
        theta = torch.atan2(yv, xv)
        geom = torch.stack(
            [xv, yv, r, torch.sin(theta), torch.cos(theta)], dim=-1
        ).view(H * W, 5)

        # Assign sector IDs (0-5 for 6 views)
        ft = theta.view(-1)
        pi = math.pi
        sid = torch.full((H * W,), 5, dtype=torch.long, device=device)
        sid[(ft >= pi / 3) & (ft < 2 * pi / 3)] = 0  # front
        sid[(ft >= pi / 6) & (ft < pi / 3)] = 1  # front_right
        sid[(ft >= 2 * pi / 3) & (ft < 5 * pi / 6)] = 2  # front_left
        sid[(ft >= -2 * pi / 3) & (ft < -pi / 3)] = 3  # back
        sid[(ft >= -pi / 3) & (ft < -pi / 6)] = 4  # back_right
        sid[(ft < -2 * pi / 3) | (ft >= 5 * pi / 6)] = 5  # back_left
        
        self._cache[key] = (geom, sid)
        return geom, sid

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        B, C, H, W = bev.shape
        dev = bev.device
        
        # Refine and project BEV features
        x = self.refine(bev)  # [B,C,H,W]
        x = self.proj(x).permute(0, 2, 3, 1).reshape(B, H * W, self.d_model)  # [B,HW,d]
        x = self.norm_tokens(x)

        # Add geometric and view embeddings
        geom, sid = self._grid(H, W, dev)  # [HW,5], [HW]
        x = x + self.geo_mlp(geom).unsqueeze(0)  # [B,HW,d]
        x = x + self.view_embed[sid].unsqueeze(0)  # [B,HW,d]

        # Initialize learnable queries
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B,nq,d]
        
        # Add view embeddings to query chunks
        if self.nq_per_view > 0:
            chunks = q.split(self.nq_per_view, dim=1)
            q = torch.cat(
                [ch + self.view_embed[k].view(1, 1, -1) for k, ch in enumerate(chunks)],
                dim=1,
            )  # [B,nq,d]

        # Apply transformer blocks
        for blk in self.blocks:
            q = blk(q, x)  # [B,nq,d]
            
        q = self.final_ln(q)
        q = self.post(q)  # [B,nq,d]
        return q
