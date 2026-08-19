"""Training collator boundary between sampler metadata and model tensors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

_SAMPLER_METADATA = frozenset(
    {
        "document_id",
        "input_length",
        "input_original_length",
        "target_length",
        "source_truncated",
    }
)


@dataclass(frozen=True, slots=True)
class MetadataStrippingCollator:
    """Keep cached lengths available to the sampler, then remove non-model metadata."""

    base_collator: Callable[[list[dict[str, Any]]], dict[str, Any]]

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty feature batch")
        cleaned = []
        for feature in features:
            missing = _SAMPLER_METADATA - feature.keys()
            if missing:
                raise ValueError(f"tokenized feature lacks sampler metadata: {sorted(missing)}")
            cleaned.append(
                {key: value for key, value in feature.items() if key not in _SAMPLER_METADATA}
            )
        return self.base_collator(cleaned)
