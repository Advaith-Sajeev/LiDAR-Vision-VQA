"""Vision Adapter - adds per-view embeddings, concatenates views, and projects to d_model.

Data flow (upstream):
- DeepEncoder produces [256, 2048] tokens per view (CLIP 1024 + SAM 1024 features, projected)

This module:
1. Takes 6 camera views in fixed order (CAM_FRONT, CAM_FRONT_RIGHT, etc.)
2. Adds a learned embedding specific to each camera/view
3. Concatenates all views: 6 * [256, 2048] → [1536, 2048]
4. Projects from 2048 to d_model: [1536, 2048] → [1536, d_model]

Output goes to VATVision for token compression (1536 → 768 tokens).
"""

import torch
import torch.nn as nn
from typing import List, Union

# Import camera view config from centralized config
try:
    from configs.default_config import DEFAULT_VIEW_ORDER, CAM_VIEWS
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
    Adds learned per-view embeddings to DeepEncoder tokens, concatenates views,
    and projects to LLM embedding dimension.

    Inputs:
        views_tokens: List of 6 tensors, each [256, 2048] from DeepEncoder
                      (one per camera view in CAM_VIEWS order)

    Output:
        Tensor [1536, d_model] where:
            - 1536 = 6 views * 256 tokens per view
            - d_model = LLM embedding dimension (e.g., 896 for Qwen2.5-0.5B)
    
    Args:
        d_in: Input dimension from DeepEncoder (2048 = CLIP 1024 + SAM 1024)
        d_model: Output dimension matching LLM embeddings
        dropout: Dropout rate after normalization
    """

    def __init__(self, d_in: int, d_model: int, dropout: float = 0.10):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
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
        
        # Output projection to d_model
        self.proj = nn.Linear(d_in, d_model)

    def forward(self, views_tokens: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            views_tokens: list of tensors, length == num_views (6),
                          each of shape [HW, d_in].

        Returns:
            out: tensor of shape [num_views * HW, d_model]
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
            debug.tensor_stats("vision_adapt", "before_projection", out)
        
        # Project to d_model: [num_views * HW, d_in] -> [num_views * HW, d_model]
        out = self.proj(out)
        
        if DEBUG_AVAILABLE:
            debug.shape("vision_adapt", "final_output", out)
            debug.tensor_stats("vision_adapt", "output", out)
            debug.debug("vision_adapt", f"Total tokens: {out.shape[0]} ({self.num_views} views × {expected_hw} tokens/view)")
            debug.debug("vision_adapt", f"Projected from {self.d_in} to {self.d_model}")
            debug.trace("vision_adapt", "Vision Adapter Complete")
        
        return out

    def forward_batch(self, batch_views_tokens: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Batched forward pass - processes multiple samples in parallel.
        
        Args:
            batch_views_tokens: List of B samples, each containing V (6) tensors [HW, d_in]
                               Shape: List[B][V] of tensors [HW, d_in]
        
        Returns:
            out: tensor of shape [B, num_views * HW, d_model]
        """
        B = len(batch_views_tokens)
        
        if DEBUG_AVAILABLE:
            debug.debug("vision_adapt", f"Batched forward: {B} samples × {self.num_views} views")
        
        if B == 0:
            raise ValueError("Empty batch provided")
        
        # Validate and stack into batched tensor
        # Each sample has V views, each view is [HW, d_in]
        V = self.num_views
        first_sample = batch_views_tokens[0]
        
        if len(first_sample) != V:
            raise ValueError(f"Expected {V} views per sample, got {len(first_sample)}")
        
        HW = first_sample[0].shape[0]
        
        # Stack all views from all samples: [B, V, HW, d_in]
        # First, stack views within each sample, then stack samples
        stacked_samples = []
        for sample_views in batch_views_tokens:
            # Stack V views: [V, HW, d_in]
            sample_tensor = torch.stack(sample_views, dim=0)
            stacked_samples.append(sample_tensor)
        
        # Stack B samples: [B, V, HW, d_in]
        x = torch.stack(stacked_samples, dim=0)
        
        if DEBUG_AVAILABLE:
            debug.shape("vision_adapt", "batched_input", x)
        
        # Add view embeddings: view_embed is [V, d_in]
        # Broadcast to [B, V, HW, d_in]
        # view_embed[None, :, None, :] -> [1, V, 1, d_in]
        x = x + self.view_embed[None, :, None, :]
        
        # Reshape for LayerNorm: [B * V * HW, d_in]
        x = x.reshape(-1, self.d_in)
        
        # Normalize + dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        # Project: [B * V * HW, d_in] -> [B * V * HW, d_model]
        x = self.proj(x)
        
        # Reshape to [B, V * HW, d_model]
        x = x.reshape(B, V * HW, self.d_model)
        
        if DEBUG_AVAILABLE:
            debug.shape("vision_adapt", "batched_output", x)
            debug.debug("vision_adapt", f"Batched output: {B} samples × {V * HW} tokens × {self.d_model} dim")
        
        return x