"""Vision Adapter - adds per-view embeddings and concatenates views.

This version:
- Takes 6 camera views (fixed order).
- Adds a learned embedding specific to each camera/view.
- Concatenates all views into a single sequence: [num_views * HW, d_in].
"""

import torch
import torch.nn as nn
from typing import List

# Default view order for nuScenes cameras (6 views)
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
    Adds learned per-view (camera-specific) embeddings to DeepEncoder tokens
    and concatenates all views into a single sequence.

    Inputs:
        views_tokens:
            List of length V (here 6), where each element is a tensor [HW, d_in]
            corresponding to one camera view in CAM_VIEWS order.

    Output:
        Tensor of shape [num_views * HW, d_in], where:
            - num_views * HW = total number of tokens (e.g., 6 * 256 = 1536)
            - d_in = original token dimension (unchanged)
    """

    def __init__(self, d_in: int, dropout: float = 0.10):
        super().__init__()
        self.d_in = d_in
        self.num_views = len(CAM_VIEWS)

        # Per-token normalization + regularization
        self.norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

        # One learnable embedding per camera/view: [num_views, d_in]
        self.view_embed = nn.Parameter(
            torch.zeros(self.num_views, d_in), requires_grad=True
        )

        # Init view embeddings (small random values)
        nn.init.trunc_normal_(self.view_embed, std=0.02)

    def forward(self, views_tokens: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            views_tokens: list of tensors, length == num_views (6),
                          each of shape [HW, d_in].

        Returns:
            out: tensor of shape [num_views * HW, d_in]
        """
        if len(views_tokens) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} views in order {CAM_VIEWS}, "
                f"got {len(views_tokens)}"
            )

        seqs = []
        expected_hw = None

        for v_idx, t in enumerate(views_tokens):
            if t.dim() != 2:
                raise ValueError(
                    f"Expected tensor of shape [HW, d_in] for view {v_idx}, "
                    f"got shape {tuple(t.shape)}"
                )

            hw, _ = t.shape
            if expected_hw is None:
                expected_hw = hw
            elif hw != expected_hw:
                raise ValueError(
                    f"All views must have same HW. Got {expected_hw} and {hw}."
                )

            # Add this view's embedding to all its tokens
            # view_embed[v_idx]: [d_in] -> broadcast to [HW, d_in]
            x = t + self.view_embed[v_idx].unsqueeze(0)

            # Normalize + dropout
            x = self.norm(x)
            x = self.dropout(x)

            seqs.append(x)  # each [HW, d_in]

        # Concatenate along the sequence dimension: [num_views * HW, d_in]
        out = torch.cat(seqs, dim=0)
        return out