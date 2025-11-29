"""
Sequence Builder for LLM Embedding Assembly.

This module provides explicit position tracking for building multimodal
input sequences. The order is guaranteed and documented:

    SEQUENCE ORDER (when all modalities enabled):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Position 0: <vision_start>                                       │
    │ Position 1: vision_tokens [n_vision_queries, d_model]            │
    │ Position 2: <vision_end>                                         │
    │ Position 3: <lidar_start>                                        │
    │ Position 4: lidar_tokens [n_lidar_queries, d_model]              │
    │ Position 5: <lidar_end>                                          │
    │ Position 6: text_prompt [prompt_len, d_model]                    │
    │ Position 7: answer_tokens [answer_len, d_model] (for training)   │
    └─────────────────────────────────────────────────────────────────┘

If a modality is disabled, its positions are skipped but the relative
order is preserved (vision always before LiDAR, LiDAR always before text).
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import torch

# Import ModalityPosition from centralized configs
from configs.default_config import ModalityPosition


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
        
        # Add pieces in any order - they'll be sorted correctly
        if use_vision:
            builder.add_vision(prefix_vision, vision_start_emb, vision_end_emb)
        if use_lidar:
            builder.add_lidar(prefix_lidar, lidar_start_emb, lidar_end_emb)
        builder.add_text_prompt(tok_emb)
        
        # Build concatenated sequence
        inp, piece_info = builder.build()
    """
    batch_size: int
    device: torch.device
    dtype: torch.dtype
    pieces: List[SequencePiece] = field(default_factory=list)
    
    def add_vision(
        self, 
        vision_tokens: torch.Tensor,
        vision_start_emb: torch.Tensor,
        vision_end_emb: torch.Tensor,
    ) -> "SequenceBuilder":
        """
        Add vision modality tokens with explicit position markers.
        
        Args:
            vision_tokens: [B, n_queries, d_model] vision VAT output
            vision_start_emb: [B, 1, d_model] start delimiter
            vision_end_emb: [B, 1, d_model] end delimiter
        """
        self.pieces.append(SequencePiece(
            position=ModalityPosition.VISION_START,
            tensor=vision_start_emb,
            name="<vision_start>"
        ))
        self.pieces.append(SequencePiece(
            position=ModalityPosition.VISION_TOKENS,
            tensor=vision_tokens,
            name=f"vision_tokens[{vision_tokens.shape[1]}]"
        ))
        self.pieces.append(SequencePiece(
            position=ModalityPosition.VISION_END,
            tensor=vision_end_emb,
            name="<vision_end>"
        ))
        return self
    
    def add_lidar(
        self,
        lidar_tokens: torch.Tensor,
        lidar_start_emb: torch.Tensor,
        lidar_end_emb: torch.Tensor,
    ) -> "SequenceBuilder":
        """
        Add LiDAR modality tokens with explicit position markers.
        
        Args:
            lidar_tokens: [B, n_queries, d_model] LiDAR VAT output
            lidar_start_emb: [B, 1, d_model] start delimiter
            lidar_end_emb: [B, 1, d_model] end delimiter
        """
        self.pieces.append(SequencePiece(
            position=ModalityPosition.LIDAR_START,
            tensor=lidar_start_emb,
            name="<lidar_start>"
        ))
        self.pieces.append(SequencePiece(
            position=ModalityPosition.LIDAR_TOKENS,
            tensor=lidar_tokens,
            name=f"lidar_tokens[{lidar_tokens.shape[1]}]"
        ))
        self.pieces.append(SequencePiece(
            position=ModalityPosition.LIDAR_END,
            tensor=lidar_end_emb,
            name="<lidar_end>"
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
    prefix_vision: Optional[torch.Tensor] = None,
    prefix_lidar: Optional[torch.Tensor] = None,
    get_special_token_emb,  # Callable to get special token embeddings
) -> Tuple[torch.Tensor, Dict]:
    """
    Build a training sequence with explicit position markers.
    
    This is a convenience function that uses SequenceBuilder internally.
    
    Args:
        E: Embedding layer (for reference, not used directly)
        device: Target device
        dtype: Target dtype
        batch_size: Batch size
        tok_emb: [B, seq_len, d_model] text prompt embeddings
        ans_emb: [B, seq_len, d_model] answer embeddings
        prefix_vision: [B, n_queries, d_model] vision VAT output, or None
        prefix_lidar: [B, n_queries, d_model] LiDAR VAT output, or None
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
    
    # Add modalities in explicit order (SequenceBuilder sorts by position anyway)
    if prefix_vision is not None:
        builder.add_vision(
            vision_tokens=prefix_vision,
            vision_start_emb=get_special_token_emb("<vision_start>").expand(batch_size, -1, -1),
            vision_end_emb=get_special_token_emb("<vision_end>").expand(batch_size, -1, -1),
        )
    
    if prefix_lidar is not None:
        builder.add_lidar(
            lidar_tokens=prefix_lidar,
            lidar_start_emb=get_special_token_emb("<lidar_start>").expand(batch_size, -1, -1),
            lidar_end_emb=get_special_token_emb("<lidar_end>").expand(batch_size, -1, -1),
        )
    
    builder.add_text_prompt(tok_emb)
    builder.add_answer(ans_emb)
    
    return builder.build()


def build_inference_sequence(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    tok_emb: torch.Tensor,
    prefix_vision: Optional[torch.Tensor] = None,
    prefix_lidar: Optional[torch.Tensor] = None,
    get_special_token_emb,  # Callable to get special token embeddings
) -> Tuple[torch.Tensor, Dict]:
    """
    Build an inference sequence with explicit position markers (no answer tokens).
    
    Args:
        device: Target device
        dtype: Target dtype
        batch_size: Batch size
        tok_emb: [B, seq_len, d_model] text prompt embeddings
        prefix_vision: [B, n_queries, d_model] vision VAT output, or None
        prefix_lidar: [B, n_queries, d_model] LiDAR VAT output, or None
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
    
    if prefix_vision is not None:
        builder.add_vision(
            vision_tokens=prefix_vision,
            vision_start_emb=get_special_token_emb("<vision_start>").expand(batch_size, -1, -1),
            vision_end_emb=get_special_token_emb("<vision_end>").expand(batch_size, -1, -1),
        )
    
    if prefix_lidar is not None:
        builder.add_lidar(
            lidar_tokens=prefix_lidar,
            lidar_start_emb=get_special_token_emb("<lidar_start>").expand(batch_size, -1, -1),
            lidar_end_emb=get_special_token_emb("<lidar_end>").expand(batch_size, -1, -1),
        )
    
    builder.add_text_prompt(tok_emb)
    
    return builder.build()


__all__ = [
    "ModalityPosition",
    "SequencePiece", 
    "SequenceBuilder",
    "build_training_sequence",
    "build_inference_sequence",
]
