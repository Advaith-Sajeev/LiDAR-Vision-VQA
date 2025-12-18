"""
Sequence Builder for LLM Embedding Assembly.

This module provides explicit position tracking for building multimodal
input sequences. The order is guaranteed and documented:

    SEQUENCE ORDER (vision-only with per-view delimiters):
    ┌─────────────────────────────────────────────────────────────────┐
    │ [cam_front_start] [256 tokens] [cam_front_end]                  │
    │ [cam_front_right_start] [256 tokens] [cam_front_right_end]      │
    │ [cam_front_left_start] [256 tokens] [cam_front_left_end]        │
    │ [cam_back_start] [256 tokens] [cam_back_end]                    │
    │ [cam_back_right_start] [256 tokens] [cam_back_right_end]        │
    │ [cam_back_left_start] [256 tokens] [cam_back_left_end]          │
    │ text_prompt [prompt_len, d_model]                               │
    │ answer_tokens [answer_len, d_model] (for training)              │
    └─────────────────────────────────────────────────────────────────┘

Each camera view has its own start and end delimiter tokens for explicit
position marking in the sequence.
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import torch

# Import from centralized configs
from configs.constants import (
    ModalityPosition, 
    DEFAULT_VIEW_ORDER,
    VIEW_POSITIONS,
    VIEW_DELIMITER_TOKENS,
    NUM_VIEWS,
)


@dataclass
class SequencePiece:
    """A single piece of the input sequence with explicit position."""
    position: ModalityPosition
    tensor: torch.Tensor
    name: str
    
    def __lt__(self, other: "SequencePiece") -> bool:
        """Enable sorting by position."""
        return self.position < other.position


@dataclass
class SequenceBuilder:
    """
    Builds LLM input sequences with explicit position tracking.
    
    Guarantees correct ordering regardless of the order pieces are added.
    
    Usage:
        builder = SequenceBuilder(batch_size=B, device=device, dtype=dtype)
        
        # Add per-view vision tokens with delimiters
        builder.add_per_view_vision(view_tokens_list, get_special_token_emb)
        
        # Add text
        builder.add_text_prompt(tok_emb)
        
        # Build concatenated sequence
        inp, piece_info = builder.build()
    """
    batch_size: int
    device: torch.device
    dtype: torch.dtype
    pieces: List[SequencePiece] = field(default_factory=list)
    
    def add_per_view_vision(
        self,
        view_tokens_list: List[torch.Tensor],
        get_special_token_emb: Callable[[str], torch.Tensor],
    ) -> "SequenceBuilder":
        """
        Add all 6 camera views with per-view delimiters.
        
        Each view gets:
        - <cam_X_start> delimiter
        - View tokens [B, 256, d_model]
        - <cam_X_end> delimiter
        
        Args:
            view_tokens_list: List of 6 tensors from VisionAdapter,
                             each [256, d_model] or [B, 256, d_model]
            get_special_token_emb: Function(str) -> [1, 1, d_model] for special tokens
        """
        if len(view_tokens_list) != NUM_VIEWS:
            raise ValueError(f"Expected {NUM_VIEWS} view tensors, got {len(view_tokens_list)}")
        
        for i, view_name in enumerate(DEFAULT_VIEW_ORDER):
            view_tokens = view_tokens_list[i]
            
            # Ensure batch dimension
            if view_tokens.dim() == 2:
                # [HW, d_model] -> [B, HW, d_model]
                view_tokens = view_tokens.unsqueeze(0).expand(self.batch_size, -1, -1)
            
            # Get delimiter tokens for this view
            start_token, end_token = VIEW_DELIMITER_TOKENS[view_name]
            start_pos, tokens_pos, end_pos = VIEW_POSITIONS[view_name]
            
            # Add start delimiter
            start_emb = get_special_token_emb(start_token).expand(self.batch_size, -1, -1)
            self.pieces.append(SequencePiece(
                position=start_pos,
                tensor=start_emb,
                name=start_token
            ))
            
            # Add view tokens
            self.pieces.append(SequencePiece(
                position=tokens_pos,
                tensor=view_tokens,
                name=f"{view_name.lower()}_tokens[{view_tokens.shape[1]}]"
            ))
            
            # Add end delimiter
            end_emb = get_special_token_emb(end_token).expand(self.batch_size, -1, -1)
            self.pieces.append(SequencePiece(
                position=end_pos,
                tensor=end_emb,
                name=end_token
            ))
        
        return self
    
    def add_text_prompt(self, tok_emb: torch.Tensor) -> "SequenceBuilder":
        """
        Add text prompt embeddings.
        
        Args:
            tok_emb: [B, seq_len, d_model] text token embeddings
        """
        self.pieces.append(SequencePiece(
            position=ModalityPosition.TEXT_PROMPT,
            tensor=tok_emb,
            name=f"text_prompt[{tok_emb.shape[1]}]"
        ))
        return self
    
    def add_answer(self, ans_emb: torch.Tensor) -> "SequenceBuilder":
        """
        Add answer embeddings (for training only).
        
        Args:
            ans_emb: [B, seq_len, d_model] answer token embeddings
        """
        self.pieces.append(SequencePiece(
            position=ModalityPosition.ANSWER_TOKENS,
            tensor=ans_emb,
            name=f"answer[{ans_emb.shape[1]}]"
        ))
        return self
    
    def build(self) -> Tuple[torch.Tensor, Dict]:
        """
        Build the final concatenated sequence.
        
        Returns:
            inp: [B, total_len, d_model] concatenated input tensor
            info: Dict with sequence metadata:
                - 'order': List of piece names in final order
                - 'lengths': Dict mapping piece names to their lengths
                - 'positions': Dict mapping ModalityPosition to (start, end) indices
        """
        if not self.pieces:
            raise ValueError("No pieces added to sequence builder")
        
        # Sort pieces by their explicit position
        sorted_pieces = sorted(self.pieces)
        
        # Build metadata
        order = []
        lengths = {}
        positions = {}
        current_pos = 0
        
        for piece in sorted_pieces:
            seq_len = piece.tensor.shape[1]
            order.append(piece.name)
            lengths[piece.name] = seq_len
            positions[piece.position] = (current_pos, current_pos + seq_len)
            current_pos += seq_len
        
        # Concatenate tensors
        tensors = [p.tensor for p in sorted_pieces]
        inp = torch.cat(tensors, dim=1)
        
        info = {
            'order': order,
            'lengths': lengths,
            'positions': positions,
            'total_length': current_pos,
        }
        
        return inp, info
    
    def get_sequence_description(self) -> str:
        """Return a human-readable description of the sequence order."""
        if not self.pieces:
            return "Empty sequence"
        
        sorted_pieces = sorted(self.pieces)
        return " → ".join(p.name for p in sorted_pieces)


def build_training_sequence(
    *,
    E,  # Embedding layer
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    tok_emb: torch.Tensor,
    ans_emb: torch.Tensor,
    view_tokens_list: Optional[List[torch.Tensor]] = None,
    get_special_token_emb: Callable[[str], torch.Tensor],
) -> Tuple[torch.Tensor, Dict]:
    """
    Build a training sequence with per-view delimiters.
    
    This is a convenience function that uses SequenceBuilder internally.
    
    Args:
        E: Embedding layer (for reference, not used directly)
        device: Target device
        dtype: Target dtype
        batch_size: Batch size
        tok_emb: [B, seq_len, d_model] text prompt embeddings
        ans_emb: [B, seq_len, d_model] answer embeddings
        view_tokens_list: List of 6 tensors from VisionAdapter, each [256, d_model]
        get_special_token_emb: Function(str) -> [1, 1, d_model] for special tokens
    
    Returns:
        inp: [B, total_len, d_model] full input sequence
        info: Dict with sequence metadata
    """
    builder = SequenceBuilder(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    
    # Add per-view vision tokens with delimiters
    if view_tokens_list is not None:
        builder.add_per_view_vision(view_tokens_list, get_special_token_emb)
    
    builder.add_text_prompt(tok_emb)
    builder.add_answer(ans_emb)
    
    return builder.build()


def build_inference_sequence(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    tok_emb: torch.Tensor,
    view_tokens_list: Optional[List[torch.Tensor]] = None,
    get_special_token_emb: Callable[[str], torch.Tensor],
) -> Tuple[torch.Tensor, Dict]:
    """
    Build an inference sequence with per-view delimiters (no answer tokens).
    
    Args:
        device: Target device
        dtype: Target dtype
        batch_size: Batch size
        tok_emb: [B, seq_len, d_model] text prompt embeddings
        view_tokens_list: List of 6 tensors from VisionAdapter, each [256, d_model]
        get_special_token_emb: Function(str) -> [1, 1, d_model] for special tokens
    
    Returns:
        inp: [B, total_len, d_model] input sequence (without answer)
        info: Dict with sequence metadata
    """
    builder = SequenceBuilder(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    
    if view_tokens_list is not None:
        builder.add_per_view_vision(view_tokens_list, get_special_token_emb)
    
    builder.add_text_prompt(tok_emb)
    
    return builder.build()


__all__ = [
    "ModalityPosition",
    "SequencePiece", 
    "SequenceBuilder",
    "build_training_sequence",
    "build_inference_sequence",
]
