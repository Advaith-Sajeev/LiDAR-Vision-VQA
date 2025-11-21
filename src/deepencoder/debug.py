"""
Debug logger for deepencoder package.
Imports the shared debug logger from training utilities.
"""

import sys
from pathlib import Path

# Add encoder-decoder to path to access training utilities
_encoder_decoder_path = Path(__file__).parent.parent / "encoder-decoder"
if str(_encoder_decoder_path) not in sys.path:
    sys.path.insert(0, str(_encoder_decoder_path))

# Import shared debug logger from training
from training.utils.debug_logger import debug

__all__ = ["debug"]
