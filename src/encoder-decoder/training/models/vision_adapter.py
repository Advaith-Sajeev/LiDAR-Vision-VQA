"""Vision Adapter - projects DeepEncoder tokens to model dimension"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

# Default view order for nuScenes cameras (6 views)
# This matches deepencoder.deepencoder_infer.DEFAULT_VIEW_ORDER
DEFAULT_VIEW_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
]

CAM_VIEWS = tuple(DEFAULT_VIEW_ORDER)  # 6 views, fixed order


class VisionAdapter(nn.Module):
    """
    Projects DeepEncoder tokens [HW, 1280] per view → d_model,
    adds 2D PE + 6-view embeddings, returns KV tokens for VATVision.

    Shapes:
      per-view tokens: List[6 × (HW, 1280)]
      output:          [B=1, sum_views(HW), d_model]  ~ [1, ~2400, d_model]
    """
    
    def __init__(self, d_in: int, d_model: int, dropout: float = 0.10):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_in, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)

        self.geo_mlp = nn.Sequential(
            nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        self.view_embed = nn.Parameter(
            torch.zeros(len(CAM_VIEWS), d_model), requires_grad=True
        )
        nn.init.zeros_(self.view_embed)

        self.dropout = nn.Dropout(dropout)
        self._grid_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

    def _grid_feats(self, side: int, device: torch.device) -> torch.Tensor:
        """Generate 2D geometric features for vision tokens."""
        key = (side, side, device)
        if key in self._grid_cache:
            return self._grid_cache[key]
            
        H = W = side
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
        
        self._grid_cache[key] = geom
        return geom

    def forward(self, views_tokens: List[torch.Tensor], grid_side: int) -> torch.Tensor:
        """
        Process multi-view tokens.
        
        Args:
            views_tokens: List of 6 tensors, each [HW, 1280]
            grid_side: Spatial dimension of the grid (e.g., 20 for 20x20)
            
        Returns:
            Concatenated vision tokens [1, 6*HW, d_model]
        """
        device = views_tokens[0].device
        geom = self._grid_feats(grid_side, device)  # [HW,5]

        seqs = []
        for v_idx, t in enumerate(views_tokens):
            # t: [HW, 1280]  -> proj -> add geom PE + view embed
            x = self.proj(t)  # [HW,d]
            x = x + self.geo_mlp(geom)  # [HW,d]
            x = x + self.view_embed[v_idx].unsqueeze(0)  # [HW,d]
            x = self.norm(x)
            x = self.dropout(x)
            seqs.append(x.unsqueeze(0))  # [1,HW,d]

        out = torch.cat(seqs, dim=1)  # [1,6*HW,d]
        return out
