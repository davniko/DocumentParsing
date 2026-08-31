"""Build-time contract for the digest-pinned CUDA synthesis image."""

from __future__ import annotations

import importlib
from importlib.metadata import version

import torch

from document_ocr.synthesis.sdv_harness import default_candidate_specs

EXPECTED_PACKAGES = {
    "ctgan": "0.12.1",
    "matplotlib": "3.11.1",
    "pandas": "2.3.3",
    "pydantic": "2.13.4",
    "pydantic-ai-slim": "2.36.0",
    "scipy": "1.18.1",
    "sdmetrics": "0.30.0",
    "sdv": "1.38.2",
    "seaborn": "0.13.2",
}


def main() -> None:
    actual = {name: version(name) for name in EXPECTED_PACKAGES}
    if actual != EXPECTED_PACKAGES:
        raise RuntimeError(
            f"CUDA synthesis dependency mismatch: expected={EXPECTED_PACKAGES}, actual={actual}"
        )
    for module in (
        "ctgan",
        "matplotlib",
        "pandas",
        "pydantic_ai",
        "scipy",
        "sdmetrics",
        "sdv",
        "seaborn",
    ):
        importlib.import_module(module)
    if torch.__version__.split("+", maxsplit=1)[0] != "2.13.0":
        raise RuntimeError(f"PyTorch version mismatch: {torch.__version__}")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"PyTorch CUDA build mismatch: {torch.version.cuda}")
    neural = default_candidate_specs(enable_gpu=True)[2:]
    if not all(candidate.parameters["enable_gpu"] is True for candidate in neural):
        raise RuntimeError("CUDA SDV candidates are not configured to request the GPU")


if __name__ == "__main__":
    main()
