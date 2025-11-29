"""
Tests for sequence_builder module.

Tests the SequenceBuilder class and helper functions for building
multimodal LLM input sequences with explicit position tracking.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(src_path))

# Add encoder-decoder to path
encoder_decoder_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(encoder_decoder_path))

from configs.default_config import ModalityPosition
from training.utils.sequence_builder import (
    SequenceBuilder,
    SequencePiece,
    build_training_sequence,
    build_inference_sequence,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Return float32 dtype for testing."""
    return torch.float32


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def d_model():
    """Model dimension for tests."""
    return 64


@pytest.fixture
def sample_tensors(batch_size, d_model, device, dtype):
    """Create sample tensors for testing."""
    return {
        "vision_start": torch.randn(batch_size, 1, d_model, device=device, dtype=dtype),
        "vision_tokens": torch.randn(batch_size, 12, d_model, device=device, dtype=dtype),
        "vision_end": torch.randn(batch_size, 1, d_model, device=device, dtype=dtype),
        "lidar_start": torch.randn(batch_size, 1, d_model, device=device, dtype=dtype),
        "lidar_tokens": torch.randn(batch_size, 8, d_model, device=device, dtype=dtype),
        "lidar_end": torch.randn(batch_size, 1, d_model, device=device, dtype=dtype),
        "text_prompt": torch.randn(batch_size, 20, d_model, device=device, dtype=dtype),
        "answer": torch.randn(batch_size, 10, d_model, device=device, dtype=dtype),
    }


# ============================================================================
# ModalityPosition Tests
# ============================================================================

class TestModalityPosition:
    """Tests for ModalityPosition enum."""
    
    def test_position_ordering(self):
        """Verify positions are in correct order."""
        assert ModalityPosition.VISION_START < ModalityPosition.VISION_TOKENS
        assert ModalityPosition.VISION_TOKENS < ModalityPosition.VISION_END
        assert ModalityPosition.VISION_END < ModalityPosition.LIDAR_START
        assert ModalityPosition.LIDAR_START < ModalityPosition.LIDAR_TOKENS
        assert ModalityPosition.LIDAR_TOKENS < ModalityPosition.LIDAR_END
        assert ModalityPosition.LIDAR_END < ModalityPosition.TEXT_PROMPT
        assert ModalityPosition.TEXT_PROMPT < ModalityPosition.ANSWER_TOKENS
    
    def test_position_values(self):
        """Verify exact position values."""
        assert ModalityPosition.VISION_START == 0
        assert ModalityPosition.VISION_TOKENS == 1
        assert ModalityPosition.VISION_END == 2
        assert ModalityPosition.LIDAR_START == 3
        assert ModalityPosition.LIDAR_TOKENS == 4
        assert ModalityPosition.LIDAR_END == 5
        assert ModalityPosition.TEXT_PROMPT == 6
        assert ModalityPosition.ANSWER_TOKENS == 7
    
    def test_all_positions_defined(self):
        """Verify all 8 positions are defined."""
        assert len(ModalityPosition) == 8


# ============================================================================
# SequencePiece Tests
# ============================================================================

class TestSequencePiece:
    """Tests for SequencePiece dataclass."""
    
    def test_piece_creation(self, device, dtype):
        """Test creating a sequence piece."""
        tensor = torch.randn(2, 10, 64, device=device, dtype=dtype)
        piece = SequencePiece(
            position=ModalityPosition.TEXT_PROMPT,
            tensor=tensor,
            name="test_piece"
        )
        assert piece.position == ModalityPosition.TEXT_PROMPT
        assert piece.name == "test_piece"
        assert torch.equal(piece.tensor, tensor)
    
    def test_piece_sorting(self, device, dtype):
        """Test that pieces sort correctly by position."""
        tensor = torch.randn(2, 5, 64, device=device, dtype=dtype)
        
        pieces = [
            SequencePiece(ModalityPosition.TEXT_PROMPT, tensor, "text"),
            SequencePiece(ModalityPosition.VISION_START, tensor, "vis_start"),
            SequencePiece(ModalityPosition.LIDAR_TOKENS, tensor, "lidar"),
        ]
        
        sorted_pieces = sorted(pieces)
        
        assert sorted_pieces[0].position == ModalityPosition.VISION_START
        assert sorted_pieces[1].position == ModalityPosition.LIDAR_TOKENS
        assert sorted_pieces[2].position == ModalityPosition.TEXT_PROMPT


# ============================================================================
# SequenceBuilder Tests
# ============================================================================

class TestSequenceBuilder:
    """Tests for SequenceBuilder class."""
    
    def test_empty_builder_raises(self, batch_size, device, dtype):
        """Test that building empty sequence raises error."""
        builder = SequenceBuilder(batch_size=batch_size, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="No pieces added"):
            builder.build()
    
    def test_text_only_sequence(self, batch_size, device, dtype, d_model):
        """Test building sequence with only text prompt."""
        builder = SequenceBuilder(batch_size=batch_size, device=device, dtype=dtype)
        
        text = torch.randn(batch_size, 20, d_model, device=device, dtype=dtype)
        builder.add_text_prompt(text)
        
        inp, info = builder.build()
        
        assert inp.shape == (batch_size, 20, d_model)
        assert info["total_length"] == 20
        assert len(info["order"]) == 1
        assert "text_prompt" in info["order"][0]
    
    def test_full_training_sequence(self, batch_size, device, dtype, sample_tensors):
        """Test building full training sequence with all modalities."""
        builder = SequenceBuilder(batch_size=batch_size, device=device, dtype=dtype)
        
        builder.add_vision(
            vision_tokens=sample_tensors["vision_tokens"],
            vision_start_emb=sample_tensors["vision_start"],
            vision_end_emb=sample_tensors["vision_end"],
        )
        builder.add_lidar(
            lidar_tokens=sample_tensors["lidar_tokens"],
            lidar_start_emb=sample_tensors["lidar_start"],
            lidar_end_emb=sample_tensors["lidar_end"],
        )
        builder.add_text_prompt(sample_tensors["text_prompt"])
        builder.add_answer(sample_tensors["answer"])
        
        inp, info = builder.build()
        
        # Check total length: 1+12+1 + 1+8+1 + 20 + 10 = 54
        expected_len = 1 + 12 + 1 + 1 + 8 + 1 + 20 + 10
        assert inp.shape == (batch_size, expected_len, sample_tensors["vision_tokens"].shape[-1])
        assert info["total_length"] == expected_len
    
    def test_order_independence(self, batch_size, device, dtype, sample_tensors):
        """Test that pieces can be added in any order and result is the same."""
        # Add in canonical order
        builder1 = SequenceBuilder(batch_size=batch_size, device=device, dtype=dtype)
        builder1.add_vision(
            sample_tensors["vision_tokens"],
            sample_tensors["vision_start"],
            sample_tensors["vision_end"],
        )
        builder1.add_lidar(
            sample_tensors["lidar_tokens"],
            sample_tensors["lidar_start"],
            sample_tensors["lidar_end"],
        )
        builder1.add_text_prompt(sample_tensors["text_prompt"])
        inp1, info1 = builder1.build()
        
        # Add in reverse order
        builder2 = SequenceBuilder(batch_size=batch_size, device=device, dtype=dtype)
        builder2.add_text_prompt(sample_tensors["text_prompt"])
        builder2.add_lidar(
            sample_tensors["lidar_tokens"],
            sample_tensors["lidar_start"],
            sample_tensors["lidar_end"],
        )
        builder2.add_vision(
            sample_tensors["vision_tokens"],
            sample_tensors["vision_start"],
            sample_tensors["vision_end"],
        )
        inp2, info2 = builder2.build()
        
        # Results should be identical
        assert torch.equal(inp1, inp2)
        assert info1["order"] == info2["order"]
        assert info1["total_length"] == info2["total_length"]
    
    def test_vision_before_lidar(self, batch_size, device, dtype, sample_tensors):
        """Test that vision tokens always come before LiDAR tokens."""
        builder = SequenceBuilder(batch_size=batch_size, device=device, dtype=dtype)
        
        # Add LiDAR first, then vision
        builder.add_lidar(
            sample_tensors["lidar_tokens"],
            sample_tensors["lidar_start"],
            sample_tensors["lidar_end"],
        )
        builder.add_vision(
            sample_tensors["vision_tokens"],
            sample_tensors["vision_start"],
            sample_tensors["vision_end"],
        )
        builder.add_text_prompt(sample_tensors["text_prompt"])
        
        inp, info = builder.build()
        
        # Check order: vision should be first
        assert "<vision_start>" in info["order"][0]
        assert "vision_tokens" in info["order"][1]
        assert "<vision_end>" in info["order"][2]
        assert "<lidar_start>" in info["order"][3]
    
    def test_position_metadata(self, batch_size, device, dtype, sample_tensors):
        """Test that position metadata is correctly computed."""
        builder = SequenceBuilder(batch_size=batch_size, device=device, dtype=dtype)
        
        builder.add_vision(
            sample_tensors["vision_tokens"],  # 12 tokens
            sample_tensors["vision_start"],   # 1 token
            sample_tensors["vision_end"],     # 1 token
        )
        builder.add_text_prompt(sample_tensors["text_prompt"])  # 20 tokens
        
        inp, info = builder.build()
        
        positions = info["positions"]
        
        # Vision start at position 0, length 1
        assert positions[ModalityPosition.VISION_START] == (0, 1)
        # Vision tokens at position 1, length 12
        assert positions[ModalityPosition.VISION_TOKENS] == (1, 13)
        # Vision end at position 13, length 1
        assert positions[ModalityPosition.VISION_END] == (13, 14)
        # Text prompt at position 14, length 20
        assert positions[ModalityPosition.TEXT_PROMPT] == (14, 34)
    
    def test_sequence_description(self, batch_size, device, dtype, sample_tensors):
        """Test human-readable sequence description."""
        builder = SequenceBuilder(batch_size=batch_size, device=device, dtype=dtype)
        
        # Empty builder
        assert builder.get_sequence_description() == "Empty sequence"
        
        # Add some pieces
        builder.add_vision(
            sample_tensors["vision_tokens"],
            sample_tensors["vision_start"],
            sample_tensors["vision_end"],
        )
        builder.add_text_prompt(sample_tensors["text_prompt"])
        
        desc = builder.get_sequence_description()
        assert "<vision_start>" in desc
        assert "vision_tokens" in desc
        assert "<vision_end>" in desc
        assert "text_prompt" in desc
        assert "→" in desc  # Arrow separator
    
    def test_method_chaining(self, batch_size, device, dtype, sample_tensors):
        """Test that builder methods return self for chaining."""
        builder = SequenceBuilder(batch_size=batch_size, device=device, dtype=dtype)
        
        result = (
            builder
            .add_vision(
                sample_tensors["vision_tokens"],
                sample_tensors["vision_start"],
                sample_tensors["vision_end"],
            )
            .add_text_prompt(sample_tensors["text_prompt"])
        )
        
        assert result is builder
        inp, info = builder.build()
        assert inp.shape[1] == 1 + 12 + 1 + 20  # vision_start + tokens + end + text


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestBuildTrainingSequence:
    """Tests for build_training_sequence helper function."""
    
    def test_full_sequence(self, batch_size, device, dtype, d_model):
        """Test building full training sequence."""
        # Create mock embedding function
        def get_special_token_emb(token_str):
            return torch.randn(1, 1, d_model, device=device, dtype=dtype)
        
        tok_emb = torch.randn(batch_size, 15, d_model, device=device, dtype=dtype)
        ans_emb = torch.randn(batch_size, 8, d_model, device=device, dtype=dtype)
        prefix_vision = torch.randn(batch_size, 12, d_model, device=device, dtype=dtype)
        prefix_lidar = torch.randn(batch_size, 6, d_model, device=device, dtype=dtype)
        
        inp, info = build_training_sequence(
            E=None,  # Not used
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            tok_emb=tok_emb,
            ans_emb=ans_emb,
            prefix_vision=prefix_vision,
            prefix_lidar=prefix_lidar,
            get_special_token_emb=get_special_token_emb,
        )
        
        # Expected: vis_start(1) + vis(12) + vis_end(1) + lid_start(1) + lid(6) + lid_end(1) + text(15) + ans(8) = 45
        expected_len = 1 + 12 + 1 + 1 + 6 + 1 + 15 + 8
        assert inp.shape == (batch_size, expected_len, d_model)
        assert info["total_length"] == expected_len
    
    def test_no_vision(self, batch_size, device, dtype, d_model):
        """Test training sequence without vision."""
        def get_special_token_emb(token_str):
            return torch.randn(1, 1, d_model, device=device, dtype=dtype)
        
        tok_emb = torch.randn(batch_size, 15, d_model, device=device, dtype=dtype)
        ans_emb = torch.randn(batch_size, 8, d_model, device=device, dtype=dtype)
        prefix_lidar = torch.randn(batch_size, 6, d_model, device=device, dtype=dtype)
        
        inp, info = build_training_sequence(
            E=None,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            tok_emb=tok_emb,
            ans_emb=ans_emb,
            prefix_vision=None,  # No vision
            prefix_lidar=prefix_lidar,
            get_special_token_emb=get_special_token_emb,
        )
        
        # Expected: lid_start(1) + lid(6) + lid_end(1) + text(15) + ans(8) = 31
        expected_len = 1 + 6 + 1 + 15 + 8
        assert inp.shape == (batch_size, expected_len, d_model)
        
        # Vision positions should not be in metadata
        assert ModalityPosition.VISION_START not in info["positions"]
    
    def test_no_lidar(self, batch_size, device, dtype, d_model):
        """Test training sequence without LiDAR."""
        def get_special_token_emb(token_str):
            return torch.randn(1, 1, d_model, device=device, dtype=dtype)
        
        tok_emb = torch.randn(batch_size, 15, d_model, device=device, dtype=dtype)
        ans_emb = torch.randn(batch_size, 8, d_model, device=device, dtype=dtype)
        prefix_vision = torch.randn(batch_size, 12, d_model, device=device, dtype=dtype)
        
        inp, info = build_training_sequence(
            E=None,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            tok_emb=tok_emb,
            ans_emb=ans_emb,
            prefix_vision=prefix_vision,
            prefix_lidar=None,  # No LiDAR
            get_special_token_emb=get_special_token_emb,
        )
        
        # Expected: vis_start(1) + vis(12) + vis_end(1) + text(15) + ans(8) = 37
        expected_len = 1 + 12 + 1 + 15 + 8
        assert inp.shape == (batch_size, expected_len, d_model)
        
        # LiDAR positions should not be in metadata
        assert ModalityPosition.LIDAR_START not in info["positions"]


class TestBuildInferenceSequence:
    """Tests for build_inference_sequence helper function."""
    
    def test_no_answer_tokens(self, batch_size, device, dtype, d_model):
        """Test that inference sequence has no answer tokens."""
        def get_special_token_emb(token_str):
            return torch.randn(1, 1, d_model, device=device, dtype=dtype)
        
        tok_emb = torch.randn(batch_size, 15, d_model, device=device, dtype=dtype)
        prefix_vision = torch.randn(batch_size, 12, d_model, device=device, dtype=dtype)
        prefix_lidar = torch.randn(batch_size, 6, d_model, device=device, dtype=dtype)
        
        inp, info = build_inference_sequence(
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            tok_emb=tok_emb,
            prefix_vision=prefix_vision,
            prefix_lidar=prefix_lidar,
            get_special_token_emb=get_special_token_emb,
        )
        
        # Expected: vis_start(1) + vis(12) + vis_end(1) + lid_start(1) + lid(6) + lid_end(1) + text(15) = 37
        expected_len = 1 + 12 + 1 + 1 + 6 + 1 + 15
        assert inp.shape == (batch_size, expected_len, d_model)
        
        # Answer position should not be in metadata
        assert ModalityPosition.ANSWER_TOKENS not in info["positions"]
    
    def test_text_only_inference(self, batch_size, device, dtype, d_model):
        """Test inference with only text prompt."""
        def get_special_token_emb(token_str):
            return torch.randn(1, 1, d_model, device=device, dtype=dtype)
        
        tok_emb = torch.randn(batch_size, 15, d_model, device=device, dtype=dtype)
        
        inp, info = build_inference_sequence(
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            tok_emb=tok_emb,
            prefix_vision=None,
            prefix_lidar=None,
            get_special_token_emb=get_special_token_emb,
        )
        
        # Only text: 15 tokens
        assert inp.shape == (batch_size, 15, d_model)
        assert info["total_length"] == 15


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_token_sequences(self, batch_size, device, dtype, d_model):
        """Test with single-token sequences."""
        builder = SequenceBuilder(batch_size=batch_size, device=device, dtype=dtype)
        
        single_token = torch.randn(batch_size, 1, d_model, device=device, dtype=dtype)
        builder.add_text_prompt(single_token)
        
        inp, info = builder.build()
        assert inp.shape == (batch_size, 1, d_model)
    
    def test_large_batch_size(self, device, dtype, d_model):
        """Test with large batch size."""
        large_batch = 32
        builder = SequenceBuilder(batch_size=large_batch, device=device, dtype=dtype)
        
        text = torch.randn(large_batch, 100, d_model, device=device, dtype=dtype)
        builder.add_text_prompt(text)
        
        inp, info = builder.build()
        assert inp.shape == (large_batch, 100, d_model)
    
    def test_tensor_device_consistency(self, batch_size, dtype, d_model):
        """Test that builder works with tensors on different devices."""
        device = torch.device("cpu")
        builder = SequenceBuilder(batch_size=batch_size, device=device, dtype=dtype)
        
        text = torch.randn(batch_size, 10, d_model, device=device, dtype=dtype)
        builder.add_text_prompt(text)
        
        inp, info = builder.build()
        assert inp.device == device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
