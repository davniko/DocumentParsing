"""Explicit extraction-only category extensions without invented application codes."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import sha256_bytes

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class SemanticCategoryExtension(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    noExactApplicationCodeReason: Text
    reviewProvenance: Text


class ApplicationRegistrySource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: Text
    sha256: Sha256
    entries: Annotated[int, Field(gt=0)]

    @field_validator("path")
    @classmethod
    def source_is_relative(cls, value: str) -> str:
        if Path(value).is_absolute():
            raise ValueError("application registry source path must be artifact-relative")
        return value


def validate_semantic_extensions(
    version: int,
    kind: str,
    unmapped: set[str],
    extensions: Mapping[str, SemanticCategoryExtension],
    source: ApplicationRegistrySource | None,
) -> None:
    """Version one is the application vocabulary; version two is reviewed extraction."""
    if version == 1:
        if unmapped or extensions or source is not None:
            raise ValueError("application registry v1 cannot contain semantic extensions")
    elif version == 2:
        if kind != "package" or source is None or not unmapped or unmapped != set(extensions):
            raise ValueError(
                "extraction registry v2 requires exact reviewed unmapped package entries"
            )
        if any(not token.startswith("PACKAGE_") for token in unmapped):
            raise ValueError("semantic package extensions require PACKAGE_ category tokens")
    else:
        raise ValueError("unsupported category registry schema version")


def verify_application_registry_source(
    registry_path: Path,
    source: ApplicationRegistrySource | None,
    entries: Sequence[dict],
) -> None:
    """Verify all mapped entry objects against the exact pinned application artifact."""
    if source is None:
        return  # Valid v1 application artifacts have no extraction-only parent.
    path = registry_path.parent / source.path
    if path.is_symlink():
        raise ValueError("application registry source must not be a symbolic link")
    raw = read_regular_file_bytes(path.resolve(strict=True))
    if sha256_bytes(raw) != source.sha256:
        raise ValueError("application registry source SHA-256 mismatch")
    base = json.loads(raw)
    if base.get("schemaVersion") != 1 or base.get("registryKind") != "package":
        raise ValueError("extraction registry requires a v1 package application source")
    mapped = [entry for entry in entries if entry["applicationCode"] is not None]
    if len(mapped) != source.entries or mapped != base.get("entries"):
        raise ValueError("extraction registry changes the pinned application entries")


def application_code(category: str, code: str | None) -> str:
    """Wire projection is a separate, explicit operation and never guesses a code."""
    if code is None:
        raise ValueError(f"category has no exact application-code mapping: {category}")
    return code
