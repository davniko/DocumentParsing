"""Named-PEFT best-checkpoint loading and final-export integrity."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)


def adapter_directory(checkpoint: Path, adapter_name: str) -> Path:
    """Resolve PEFT's native save layout without guessing another adapter."""
    directory = checkpoint if adapter_name == "default" else checkpoint / adapter_name
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (directory / filename).is_file():
            raise RuntimeError(f"checkpoint adapter file is missing: {directory / filename}")
    return directory


class PeftBestCheckpointMixin:
    """Load the one active training adapter, including PEFT's named subdirectory."""

    def _load_best_model(self) -> None:
        from peft import PeftModel

        model = self.accelerator.unwrap_model(  # type: ignore[attr-defined]
            self.model,
            keep_torch_compile=False,  # type: ignore[attr-defined]
        )
        if not isinstance(model, PeftModel):
            return super()._load_best_model()  # type: ignore[misc,no-any-return]
        if self.is_deepspeed_enabled or self.is_fsdp_enabled:  # type: ignore[attr-defined]
            raise RuntimeError("named-adapter best loading requires the single-process trainer")
        active = model.active_adapters
        if len(active) != 1 or set(model.peft_config) != set(active):
            raise RuntimeError("best-checkpoint loading requires exactly one training adapter")
        checkpoint = self.state.best_model_checkpoint  # type: ignore[attr-defined]
        if checkpoint is None:
            raise RuntimeError("best-checkpoint loading requires a selected checkpoint")
        adapter_name = active[0]
        directory = adapter_directory(Path(checkpoint), adapter_name)
        _LOGGER.info("Loading best adapter %s from %s", adapter_name, directory)
        result = model.load_adapter(
            str(directory),
            adapter_name,
            is_trainable=True,
            torch_device="cpu",
            local_files_only=True,
        )
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"best adapter did not load completely from {directory}: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )


def verify_best_adapter_export(
    *,
    checkpoint: Path,
    exported: Path,
    adapter_name: str,
) -> dict[str, Any]:
    """Check every exported tensor against the selected checkpoint, one tensor at a time."""
    import torch
    from safetensors import safe_open

    best_directory = adapter_directory(checkpoint, adapter_name)
    exported_directory = adapter_directory(exported, adapter_name)
    best_file = best_directory / "adapter_model.safetensors"
    exported_file = exported_directory / "adapter_model.safetensors"
    with (
        safe_open(best_file, framework="pt", device="cpu") as best,
        safe_open(exported_file, framework="pt", device="cpu") as final,
    ):
        names = set(best.keys())
        if not names or names != set(final.keys()):
            raise RuntimeError("final adapter tensor names differ from the best checkpoint")
        tensor_bytes = 0
        for name in sorted(names):
            expected, actual = best.get_tensor(name), final.get_tensor(name)
            if expected.dtype != actual.dtype or not torch.equal(expected, actual):
                raise RuntimeError(f"final adapter differs from the best checkpoint: {name}")
            tensor_bytes += expected.numel() * expected.element_size()
    return {
        "status": "verified",
        "best_checkpoint": str(checkpoint),
        "adapter_name": adapter_name,
        "tensor_count": len(names),
        "tensor_bytes": tensor_bytes,
    }
