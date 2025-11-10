# file: vat_lidar.py
"""
LiDAR VAT - View-Aware Transformer for BEV features

Overview
--------
This module implements VATLiDAR, a view-aware transformer that:
- Consumes BEV feature maps: [B, C_in, H, W]
- Converts them into BEV tokens (K/V) enriched with:
    - continuous geometric positional encoding (x, y, r, sinθ, cosθ)
    - discrete 6-way view (sector) embeddings
- Uses learned query tokens, partitioned into 6 view-specific groups, that attend over BEV tokens
- Produces compact, view-aware tokens: [B, n_queries, d_model]

Key ideas
---------
- Geometric PE is continuous and resolution-agnostic (computed from normalized coordinates).
- View embeddings align BEV tokens and query tokens by sector.
- n_queries must be divisible by NUM_VIEWS (here 6); each view gets an equal share of queries.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .vat_blocks import VATBlock


# Import debug logger
try:
    from ..utils import debug
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False


NUM_VIEWS = 6 # front, front_right, front_left, back, back_right, back_left


class VATLiDAR(nn.Module):
    """
    View-Aware Transformer over BEV features.

    Input:
        bev: [B, C_in, H, W]

    Processing:
        1. Depthwise conv refinement on bev.
        2. 1x1 conv projection to d_model → BEV tokens (K/V): [B, H*W, d_model]
        3. Add geometric positional embeddings (MLP(x, y, r, sinθ, cosθ)).
        4. Add 6-way view embeddings based on polar angle sector.
        5. Initialize n_queries learned query tokens, split into 6 equal groups;
           each group tagged with its corresponding view embedding.
        6. Pass queries through stacked VATBlocks with BEV tokens as K/V.

    Output:
        tokens: [B, n_queries, d_model]
            - queries are view-aware summaries of the BEV.
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
        assert n_queries % NUM_VIEWS == 0, "n_queries must be divisible by NUM_VIEWS (6)."

        self.d_model = d_model
        self.n_queries = n_queries
        self.nq_per_view = n_queries // NUM_VIEWS

        # Lightweight local refinement of BEV features (depthwise conv).
        self.refine = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in),
            nn.GELU(),
        )

        # Project BEV features to transformer dimension.
        self.proj = nn.Conv2d(c_in, d_model, kernel_size=1, bias=True)
        self.norm_tokens = nn.LayerNorm(d_model)

        # Geometric positional encoding MLP:
        # [x, y, r, sinθ, cosθ] → d_model.
        self.geo_mlp = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable view embeddings: one per sector (0..5).
        # Shape: [6, d_model]
        self.view_embed = nn.Parameter(torch.zeros(NUM_VIEWS, d_model))

        # Learnable queries (later split into 6 equal view groups).
        # Shape: [n_queries, d_model]
        self.query = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)

        # VAT blocks (cross-attention style: queries attend over BEV tokens).
        d_ff = int(mlp_ratio * d_model)
        self.blocks = nn.ModuleList(
            [VATBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Final normalization + projection head.
        self.final_ln = nn.LayerNorm(d_model)
        self.post = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(post_dropout),
            nn.Linear(d_model, d_model),
        )

        # Cache for geometric grid (per (H, W, device)).
        # Maps (H, W, device) -> (geom: [HW, 5], sid: [HW])
        self._cache: Dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _grid(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build geometric features and sector IDs for an H×W BEV grid.

        Returns:
            geom: [HW, 5]
                - x, y in [-1, 1]
                - r in [0, 1]
                - sinθ, cosθ
            sid: [HW]
                - integer sector id in {0..5}, one of 6 non-overlapping angular bins.
        """
        key = (H, W, device)
        if key in self._cache:
            return self._cache[key]

        # Normalized coordinates: y in [-1, 1] (top→bottom), x in [-1, 1] (left→right).
        yv, xv = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing="ij",
        )

        r = torch.clamp((xv**2 + yv**2).sqrt(), 0.0, 1.0)
        theta = torch.atan2(yv, xv)  # [-pi, pi], atan2(y, x)

        # Geometric feature: [x, y, r, sinθ, cosθ]
        geom = torch.stack(
            (xv, yv, r, torch.sin(theta), torch.cos(theta)),
            dim=-1,
        ).view(H * W, 5)

        # ----- View / sector assignment (corrected) -----
        # 6 contiguous 60° sectors covering [-pi, pi], aligned with comments:
        #   0: front       ~ [ 60°, 120°)
        #   1: front_right ~ [  0°,  60°)
        #   2: front_left  ~ [120°, 180°]
        #   3: back        ~ [-120°, -60°)
        #   4: back_right  ~ [ -60°,   0°)
        #   5: back_left   ~ [-180°, -120°)
        ft = theta.view(-1)
        pi = math.pi
        sid = torch.empty(H * W, dtype=torch.long, device=device)

        # Front
        sid[(ft >= pi / 3) & (ft < 2 * pi / 3)] = 0
        # Front-right
        sid[(ft >= 0.0) & (ft < pi / 3)] = 1
        # Front-left (includes angles up to +pi)
        sid[(ft >= 2 * pi / 3) & (ft <= pi)] = 2
        # Back
        sid[(ft >= -2 * pi / 3) & (ft < -pi / 3)] = 3
        # Back-right
        sid[(ft >= -pi / 3) & (ft < 0.0)] = 4
        # Back-left (includes angles down to -pi)
        sid[(ft >= -pi) & (ft < -2 * pi / 3)] = 5

        self._cache[key] = (geom, sid)
        return geom, sid

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev: [B, C_in, H, W] BEV feature map.

        Returns:
            queries: [B, n_queries, d_model]
                View-aware latent tokens; n_queries is split evenly across 6 views.
        """
        if DEBUG_AVAILABLE:
            debug.trace("vat_lidar", "=" * 40)
            debug.trace("vat_lidar", "LiDAR VAT Forward Pass")
            debug.trace("vat_lidar", "=" * 40)
            debug.shape("vat_lidar", "input_bev", bev)
        
        B, C, H, W = bev.shape
        dev = bev.device
        
        if DEBUG_AVAILABLE:
            debug.debug("vat_lidar", f"Input: B={B}, C={C}, H={H}, W={W}")

        # 1) Local refinement.
        if DEBUG_AVAILABLE:
            debug.start_timer("vat_lidar", "refinement")
        
        x = self.refine(bev)  # [B, C_in, H, W]
        
        if DEBUG_AVAILABLE:
            debug.shape("vat_lidar", "refined", x)
            debug.end_timer("vat_lidar", "refinement")

        # 2) Project to d_model tokens.
        if DEBUG_AVAILABLE:
            debug.start_timer("vat_lidar", "projection")
        
        x = self.proj(x)                      # [B, d_model, H, W]
        x = x.permute(0, 2, 3, 1)             # [B, H, W, d_model]
        x = x.reshape(B, H * W, self.d_model) # [B, HW, d_model]
        x = self.norm_tokens(x)
        
        if DEBUG_AVAILABLE:
            debug.shape("vat_lidar", "projected_tokens", x)
            debug.end_timer("vat_lidar", "projection")

        # 3) Add geometric PE.
        if DEBUG_AVAILABLE:
            debug.start_timer("vat_lidar", "geometric_pe")
        
        geom, sid = self._grid(H, W, dev)         # geom: [HW,5], sid: [HW]
        geo_pe = self.geo_mlp(geom)               # [HW, d_model]
        x = x + geo_pe.unsqueeze(0)               # [B, HW, d_model]
        
        if DEBUG_AVAILABLE:
            debug.shape("vat_lidar", "with_geo_pe", x)
            debug.tensor_stats("vat_lidar", "geometric_pe", geo_pe)
            debug.end_timer("vat_lidar", "geometric_pe")

        # 4) Add view embeddings to BEV tokens.
        if DEBUG_AVAILABLE:
            debug.start_timer("vat_lidar", "view_embedding")
        
        x = x + self.view_embed[sid].unsqueeze(0)  # [B, HW, d_model]
        
        if DEBUG_AVAILABLE:
            debug.shape("vat_lidar", "with_view_embed", x)
            debug.debug("vat_lidar", f"View sectors: {len(torch.unique(sid))} unique sectors")
            debug.end_timer("vat_lidar", "view_embedding")

        # 5) Initialize queries and make them view-aware.
        if DEBUG_AVAILABLE:
            debug.start_timer("vat_lidar", "query_init")
        
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, n_queries, d_model]

        if self.nq_per_view > 0:
            # Split into 6 groups and tag each with its corresponding view embedding.
            chunks = q.split(self.nq_per_view, dim=1)  # list of 6 x [B, nq_per_view, d_model]
            q = torch.cat(
                [
                    ch + self.view_embed[k].view(1, 1, -1)
                    for k, ch in enumerate(chunks)
                ],
                dim=1,
            )  # [B, n_queries, d_model]
        
        if DEBUG_AVAILABLE:
            debug.shape("vat_lidar", "initialized_queries", q)
            debug.debug("vat_lidar", f"Queries per view: {self.nq_per_view}")
            debug.end_timer("vat_lidar", "query_init")

        # 6) VAT blocks: queries attend over BEV tokens.
        if DEBUG_AVAILABLE:
            debug.start_timer("vat_lidar", "vat_blocks")
            debug.debug("vat_lidar", f"Processing {len(self.blocks)} VAT blocks")
        
        for i, blk in enumerate(self.blocks):
            if DEBUG_AVAILABLE:
                debug.trace("vat_lidar", f"Block {i+1}/{len(self.blocks)}")
            q = blk(q, x)  # [B, n_queries, d_model]
        
        if DEBUG_AVAILABLE:
            debug.shape("vat_lidar", "after_blocks", q)
            debug.end_timer("vat_lidar", "vat_blocks")

        # 7) Final normalization + MLP head.
        if DEBUG_AVAILABLE:
            debug.start_timer("vat_lidar", "final_projection")
        
        q = self.final_ln(q)
        q = self.post(q)  # [B, n_queries, d_model]
        
        if DEBUG_AVAILABLE:
            debug.shape("vat_lidar", "output", q)
            debug.tensor_stats("vat_lidar", "output", q)
            debug.end_timer("vat_lidar", "final_projection")
            debug.trace("vat_lidar", "LiDAR VAT Complete")

        return q
