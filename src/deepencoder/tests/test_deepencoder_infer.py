import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import torch
from PIL import Image
from deepencoder.deepencoder_infer import DeepEncoderRuntime, CONFIG
import tempfile

def test_deepencoder_runtime():
    # Mock input image (1024x1024 RGB)
    input_image = Image.new("RGB", (1024, 1024), color=(255, 255, 255))

    # Initialize DeepEncoderRuntime
    runtime_config = {k: v for k, v in CONFIG.items() if k not in ['image', 'save_npy']}
    # Override device to use CPU
    runtime_config['device'] = 'cpu'
    runtime = DeepEncoderRuntime(**runtime_config)

    # Save the mock image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_image:
        input_image.save(temp_image.name)

        # Print input image dimensions
        print("Input image dimensions:", input_image.size)

        # Use the encode_image method to process the image
        embeddings = runtime.encode_image(temp_image.name)

        # Print the embedding dimensions for debugging
        print("Embedding dimensions:", embeddings["tokens"].shape)

        # Print entering SAM module
        print("Entering SAM module...")

        # Print SAM output dimensions
        print("SAM output dimensions:", embeddings["tokens"].shape)

        # Print entering CLIP module
        print("Entering CLIP module...")

        # Print final embedding dimensions
        print("Final embedding dimensions:", embeddings["tokens"].shape)

        # Print the model summary
        print("\nModel Summary:\n")
        print(runtime.clip_vit)

        # Print detailed summaries of all modules
        print("\nSAM Module Summary:\n")
        print(runtime.sam)

        print("\nCLIP Module Summary:\n")
        print(runtime.clip_vit)

        print("\nMLP Projector Summary:\n")
        print(runtime.projector)

    # Assert output dimensions (e.g., [256, 2048])
    assert embeddings["tokens"].shape == (256, 2048), "Embedding dimensions are incorrect"

# Add more tests for other components if needed