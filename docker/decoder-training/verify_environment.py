"""Build-time decoder environment contract; never loads model weights."""

from __future__ import annotations

import importlib.metadata
import sys


EXPECTED = {
    "click": "8.3.1",
    "torch": "2.11.0+cu130",
    "transformers": "5.5.0",
    "trl": "0.24.0",
    "unsloth": "2026.8.22",
}

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"decoder image requires Python 3.12, found {sys.version}")
for package, expected in EXPECTED.items():
    actual = importlib.metadata.version(package)
    if actual != expected:
        raise RuntimeError(f"decoder image requires {package}=={expected}, found {actual}")

from document_ocr.decoder_training.config import DecoderTrainingConfig  # noqa: E402,F401
