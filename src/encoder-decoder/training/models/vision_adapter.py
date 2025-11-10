"""Vision Adapter - adds per-view embeddings and concatenates views.

This version:
- Takes 6 camera views (fixed order).
- Adds a learned embedding specific to each camera/view.
- Concatenates all views into a single sequence: [num_views * HW, d_in].
"""

import torch
import torch.nn as nn
from typing import List


# Import debug logger
try:
    from ..utils import debug
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False


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
        if DEBUG_AVAILABLE:
            debug.trace("vision_adapt", "=" * 40)
            debug.trace("vision_adapt", "Vision Adapter Forward Pass")
            debug.trace("vision_adapt", "=" * 40)
            debug.debug("vision_adapt", f"Input: {len(views_tokens)} views")
        
        if len(views_tokens) != self.num_views:
            error_msg = (
                f"Expected {self.num_views} views in order {CAM_VIEWS}, "
                f"got {len(views_tokens)}"
            )
            if DEBUG_AVAILABLE:
                debug.error("vision_adapt", error_msg)
            raise ValueError(error_msg)

        seqs = []
        expected_hw = None

        for v_idx, t in enumerate(views_tokens):
            if DEBUG_AVAILABLE:
                debug.trace("vision_adapt", f"Processing view {v_idx}: {CAM_VIEWS[v_idx]}")
                debug.shape("vision_adapt", f"view_{v_idx}_input", t)
            
            if t.dim() != 2:
                error_msg = (
                    f"Expected tensor of shape [HW, d_in] for view {v_idx}, "
                    f"got shape {tuple(t.shape)}"
                )
                if DEBUG_AVAILABLE:
                    debug.error("vision_adapt", error_msg)
                raise ValueError(error_msg)

            hw, _ = t.shape
            if expected_hw is None:
                expected_hw = hw
                if DEBUG_AVAILABLE:
                    debug.debug("vision_adapt", f"Expected HW per view: {expected_hw}")
            elif hw != expected_hw:
                error_msg = f"All views must have same HW. Got {expected_hw} and {hw}."
                if DEBUG_AVAILABLE:
                    debug.error("vision_adapt", error_msg)
                raise ValueError(error_msg)

            # Add this view's embedding to all its tokens
            # view_embed[v_idx]: [d_in] -> broadcast to [HW, d_in]
            x = t + self.view_embed[v_idx].unsqueeze(0)
            
            if DEBUG_AVAILABLE:
                debug.tensor_stats("vision_adapt", f"view_{v_idx}_with_embed", x)

            # Normalize + dropout
            x = self.norm(x)
            x = self.dropout(x)
            
            if DEBUG_AVAILABLE:
                debug.shape("vision_adapt", f"view_{v_idx}_output", x)

            seqs.append(x)  # each [HW, d_in]

        # Concatenate along the sequence dimension: [num_views * HW, d_in]
        out = torch.cat(seqs, dim=0)
        
        if DEBUG_AVAILABLE:
            debug.shape("vision_adapt", "concatenated_output", out)
            debug.tensor_stats("vision_adapt", "output", out)
            debug.debug("vision_adapt", f"Total tokens: {out.shape[0]} ({self.num_views} views × {expected_hw} tokens/view)")
            debug.trace("vision_adapt", "Vision Adapter Complete")
        
        return out