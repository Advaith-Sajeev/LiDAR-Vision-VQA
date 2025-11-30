#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest configuration for training-test tests.

This conftest.py ensures that:
1. The src/ directory is in sys.path for imports
2. The encoder-decoder/ directory is in sys.path for training imports
3. All tests can import from deepencoder.* and training.*
"""

import sys
import os
from pathlib import Path

# Determine paths relative to this file
_THIS_DIR = Path(__file__).resolve().parent           # training-test/
_ENCODER_DECODER_DIR = _THIS_DIR.parent               # encoder-decoder/
_SRC_DIR = _ENCODER_DECODER_DIR.parent                # src/

# Add paths to sys.path if not already present
for path in [_SRC_DIR, _ENCODER_DECODER_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Set PYTHONPATH for any subprocess calls
os.environ["PYTHONPATH"] = f"{_SRC_DIR}:{_ENCODER_DECODER_DIR}:" + os.environ.get("PYTHONPATH", "")


# Print confirmation during test collection (visible with -v flag)
def pytest_configure(config):
    """Called after command line options have been parsed."""
    print(f"\n[conftest.py] training-test configured")
    print(f"[conftest.py] src path: {_SRC_DIR}")
    print(f"[conftest.py] encoder-decoder path: {_ENCODER_DECODER_DIR}")
