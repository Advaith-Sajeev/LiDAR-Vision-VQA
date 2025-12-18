"""Vision Adapter - adds per-view embeddings to DeepEncoder outputs.

Data flow:
- DeepEncoder produces [256, d_model] tokens per view (CLIP + SAM features, projected to d_model)

This module:
1. Takes 6 camera views as separate tensors in fixed order (CAM_FRONT, CAM_FRONT_RIGHT, etc.)
2. Adds a learned embedding specific to each camera/view
3. Applies LayerNorm + dropout for regularization
4. Returns 6 separate tensors - NO CONCATENATION

Output: List of 6 tensors, each [256, d_model], ready for per-view delimiter insertion.
"""

import torch
import torch.nn as nn
from typing import List

# Import camera view config from centralized config
try:
    from configs.constants import DEFAULT_VIEW_ORDER, CAM_VIEWS
except ImportError:
    # Fallback if configs not in path
    DEFAULT_VIEW_ORDER = (
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
        "CAM_BACK_LEFT",
    )
    CAM_VIEWS = DEFAULT_VIEW_ORDER


# Import debug logger
try:
    from ..utils import debug
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False


class VisionAdapter(nn.Module):
    """
    Adds learned per-view embeddings to DeepEncoder tokens.
    
    Since DeepEncoder now outputs directly in d_model dimension, no projection is needed.
    This module returns 6 SEPARATE tensors (not concatenated) so that per-view
    delimiters can be inserted by the sequence builder.

    Inputs:
        views_tokens: List of 6 tensors, each [256, d_model] from DeepEncoder
                      (one per camera view in CAM_VIEWS order)

    Output:
        List of 6 tensors, each [256, d_model] with view embeddings added
    
    Args:
        d_model: Dimension from DeepEncoder (now equals LLM d_model)
        dropout: Dropout rate after normalization
    """

    def __init__(self, d_model: int, dropout: float = 0.10):
        super().__init__()
        self.d_model = d_model
        self.num_views = len(CAM_VIEWS)

        # Per-token normalization + regularization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # One learnable embedding per camera/view: [num_views, d_model]
        self.view_embed = nn.Parameter(
            torch.zeros(self.num_views, d_model), requires_grad=True
        )

        # Init view embeddings (small random values)
        nn.init.trunc_normal_(self.view_embed, std=0.02)

    def forward(self, views_tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process each view separately and return as a list.
        
        Args:
            views_tokens: list of tensors, length == num_views (6),
                          each of shape [HW, d_model] (typically [256, d_model]).

        Returns:
            List of 6 tensors, each [HW, d_model] with view embeddings added
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

        outputs = []
        
        for i, view_tok in enumerate(views_tokens):
            if DEBUG_AVAILABLE:
                debug.trace("vision_adapt", f"View {i} ({CAM_VIEWS[i]}): input shape {tuple(view_tok.shape)}")
            
            # Add view-specific embedding (broadcasts across all tokens)
            # view_tok: [HW, d_model], view_embed[i]: [d_model]
            out = view_tok + self.view_embed[i]  # [HW, d_model]
            
            # Normalize and apply dropout
            out = self.norm(out)
            out = self.dropout(out)
            
            if DEBUG_AVAILABLE:
                debug.trace("vision_adapt", f"View {i} output shape: {tuple(out.shape)}")
            
            outputs.append(out)

        if DEBUG_AVAILABLE:
            debug.debug("vision_adapt", f"Output: {len(outputs)} separate tensors")

        return outputs

    def forward_batch(self, batch_views_tokens: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        """
        Batched processing for multiple samples.
        
        Args:
            batch_views_tokens: List of length B (batch), where each element
                               is a list of 6 tensors (one per view), each [HW, d_model]

        Returns:
            List of length B, where each element is a list of 6 tensors [HW, d_model]
        """
        if DEBUG_AVAILABLE:
            debug.trace("vision_adapt", "=" * 40)
            debug.trace("vision_adapt", "Vision Adapter Batched Forward Pass")
            debug.trace("vision_adapt", f"Batch size: {len(batch_views_tokens)}")
            debug.trace("vision_adapt", "=" * 40)
        
        batch_outputs = []
        
        for sample_views in batch_views_tokens:
            # Process each sample individually
            sample_outputs = self.forward(sample_views)
            batch_outputs.append(sample_outputs)
        
        if DEBUG_AVAILABLE:
            debug.debug("vision_adapt", f"Batch output: {len(batch_outputs)} samples, each with {self.num_views} views")
        
        return batch_outputs

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_views={self.num_views}, dropout={self.dropout.p}"


__all__ = ["VisionAdapter"]