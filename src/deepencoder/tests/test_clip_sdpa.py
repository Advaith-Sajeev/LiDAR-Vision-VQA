import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from deepencoder.clip_sdpa import build_clip_l

import pytest
import torch

def test_vit_model_input_output():
    # Mock input tensor with expected dimensions (e.g., batch_size=2, channels=3, height=224, width=224)
    input_tensor = torch.randn(2, 3, 224, 224)

    # Initialize the VitModel using the builder function
    model = build_clip_l()

    # Forward pass
    output_tensor = model(input_tensor, patch_embeds=None)

    # Assert output dimensions (batch_size=2, sequence_length=257, hidden_size=1024)
    assert output_tensor.shape == (2, 257, 1024), "Output dimensions are incorrect"

# Add more tests for other components if needed