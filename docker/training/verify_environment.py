"""Build-time contract for the digest-pinned KIE training image."""

from __future__ import annotations

import importlib
import importlib.metadata

import torch

EXPECTED_PACKAGES = {
    "accelerate": "1.14.0",
    "datasets": "5.0.1",
    "mlflow-skinny": "3.15.1",
    "nvidia-ml-py": "13.610.43",
    "peft": "0.19.1",
    "transformers": "5.15.0",
}


def main() -> None:
    actual = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    if actual != EXPECTED_PACKAGES:
        raise RuntimeError(
            f"training dependency mismatch: expected={EXPECTED_PACKAGES}, actual={actual}"
        )
    modules = {
        "accelerate": "accelerate",
        "datasets": "datasets",
        "mlflow-skinny": "mlflow",
        "nvidia-ml-py": "pynvml",
        "peft": "peft",
        "transformers": "transformers",
    }
    for module in modules.values():
        importlib.import_module(module)
    importlib.import_module("document_ocr.training.cli")
    if torch.__version__.split("+", maxsplit=1)[0] != "2.13.0":
        raise RuntimeError(f"PyTorch version mismatch: {torch.__version__}")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"PyTorch CUDA build mismatch: {torch.version.cuda}")


if __name__ == "__main__":
    main()
