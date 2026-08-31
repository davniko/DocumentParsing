"""Pinned package vocabulary and train-only printed-surface support.

The MPCI package registry is an application vocabulary, while the printed text
on a bill of lading is evidence.  These contracts keep those two layers
separate: category tokens are validated against the pinned registry and exact
printed variants remain an empirical, train-only inventory.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.hashing import sha256_bytes

CategoryToken = Annotated[
    str,
    StringConstraints(pattern=r"^PACKAGE_[A-Z0-9]+(?:_[A-Z0-9]+)*$", max_length=96),
]
ApplicationCode = Annotated[str, StringConstraints(pattern=r"^[A-Z0-9]{2}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]

_WHITESPACE = re.compile(r"\s+")
_READ_CHUNK_SIZE = 1024 * 1024
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class PackageRegistryEntry(BaseModel):
    """One exact application category and its deployed code."""

    model_config = _STRICT

    categoryToken: CategoryToken
    applicationCode: ApplicationCode
    displayName: NonEmptyText

    @model_validator(mode="after")
    def canonical_text(self) -> PackageRegistryEntry:
        if self.displayName != self.displayName.strip():
            raise ValueError("package displayName must not contain outer whitespace")
        if any(unicodedata.category(character) == "Cc" for character in self.displayName):
            raise ValueError("package displayName must not contain control characters")
        return self


class PackageRegistryPayload(BaseModel):
    """Strict payload stored in the pinned registry artifact."""

    model_config = _STRICT

    schemaVersion: Annotated[int, Field(ge=1)]
    registryKind: Annotated[str, StringConstraints(pattern=r"^package$")]
    sourceAuthority: NonEmptyText
    sourceRevision: NonEmptyText
    sourcePath: NonEmptyText
    sourceSha256: Sha256
    entries: tuple[PackageRegistryEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def entries_are_canonical(self) -> PackageRegistryPayload:
        tokens = tuple(row.categoryToken for row in self.entries)
        codes = tuple(row.applicationCode for row in self.entries)
        names = tuple(row.displayName for row in self.entries)
        if len(tokens) != len(set(tokens)):
            raise ValueError("package registry categoryToken values must be unique")
        if len(codes) != len(set(codes)):
            raise ValueError("package registry applicationCode values must be unique")
        if len(names) != len(set(names)):
            raise ValueError("package registry displayName values must be unique")
        return self


class LoadedPackageRegistry(BaseModel):
    """Registry payload paired with the verified artifact receipt."""

    model_config = _STRICT

    path: NonEmptyText
    sha256: Sha256
    payload: PackageRegistryPayload

    def entry(self, category_token: str) -> PackageRegistryEntry:
        """Return one category or fail closed for an unknown token."""

        for row in self.payload.entries:
            if row.categoryToken == category_token:
                return row
        raise ValueError(f"package category is absent from pinned registry: {category_token!r}")

    @property
    def category_tokens(self) -> frozenset[str]:
        return frozenset(row.categoryToken for row in self.payload.entries)


class PackageSurfaceVariant(BaseModel):
    """One exact printed spelling inside a normalized surface bucket."""

    model_config = _STRICT

    exact_surface: NonEmptyText
    occurrences: Annotated[int, Field(gt=0)]
    document_count: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def documents_do_not_exceed_occurrences(self) -> PackageSurfaceVariant:
        if self.document_count > self.occurrences:
            raise ValueError("printed-surface document count exceeds occurrences")
        return self


class PackageSurfaceBucket(BaseModel):
    """All formatting variants for one normalized printed surface."""

    model_config = _STRICT

    category_token: CategoryToken | None
    normalized_surface: NonEmptyText
    variants: tuple[PackageSurfaceVariant, ...] = Field(min_length=1)
    occurrences: Annotated[int, Field(gt=0)]
    document_count: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def counts_and_order_match_variants(self) -> PackageSurfaceBucket:
        exact = tuple(row.exact_surface for row in self.variants)
        if exact != tuple(sorted(set(exact))):
            raise ValueError("printed-surface variants must be unique and sorted")
        if sum(row.occurrences for row in self.variants) != self.occurrences:
            raise ValueError("printed-surface occurrence count does not balance")
        if max(row.document_count for row in self.variants) > self.document_count:
            raise ValueError("printed-surface bucket document count is inconsistent")
        if any(normalize_printed_surface(value) != self.normalized_surface for value in exact):
            raise ValueError("printed-surface variant differs from normalized bucket")
        return self


class PackageSurfaceInventory(BaseModel):
    """Train-only category-to-surface support with complete accounting."""

    model_config = _STRICT

    train_document_count: Annotated[int, Field(gt=0)]
    decision_count: Annotated[int, Field(ge=0)]
    typed_decision_count: Annotated[int, Field(ge=0)]
    fallback_decision_count: Annotated[int, Field(ge=0)]
    buckets: tuple[PackageSurfaceBucket, ...]

    @model_validator(mode="after")
    def inventory_balances(self) -> PackageSurfaceInventory:
        if self.typed_decision_count + self.fallback_decision_count != self.decision_count:
            raise ValueError("package surface decision counts do not balance")
        if sum(row.occurrences for row in self.buckets) != self.decision_count:
            raise ValueError("package surface buckets do not balance")
        keys = tuple((row.category_token or "", row.normalized_surface) for row in self.buckets)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("package surface buckets must be unique and sorted")
        owners: dict[str, str | None] = {}
        for bucket in self.buckets:
            previous = owners.setdefault(bucket.normalized_surface, bucket.category_token)
            if previous != bucket.category_token:
                raise ValueError(
                    "one normalized printed surface resolves to multiple package categories"
                )
        return self

    def surfaces_for(self, category_token: str | None) -> tuple[PackageSurfaceBucket, ...]:
        return tuple(row for row in self.buckets if row.category_token == category_token)


class _PackageSurfaceDecision(BaseModel):
    model_config = _STRICT

    categoryToken: CategoryToken | None
    groupId: Annotated[str, StringConstraints(pattern=r"^g[1-9][0-9]*$")]
    packageId: Annotated[str, StringConstraints(pattern=r"^p[1-9][0-9]*$")]
    sourceTypeDescription: NonEmptyText
    sourceTypeDescriptionSha256: Sha256

    @model_validator(mode="after")
    def description_digest_matches(self) -> _PackageSurfaceDecision:
        if sha256_bytes(self.sourceTypeDescription.encode("utf-8")) != (
            self.sourceTypeDescriptionSha256
        ):
            raise ValueError("printed package surface SHA-256 mismatch")
        normalize_printed_surface(self.sourceTypeDescription)
        return self


def normalize_printed_surface(value: str) -> str:
    """Canonical comparison key without discarding the exact printed variant."""

    if not isinstance(value, str):
        raise TypeError("printed surface must be a string")
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(character) == "Cc" for character in normalized):
        raise ValueError("printed surface must not contain control characters")
    normalized = _WHITESPACE.sub(" ", normalized).strip().upper()
    if not normalized:
        raise ValueError("printed surface must not be empty")
    return normalized


def load_package_registry(
    path: Path,
    *,
    expected_sha256: str,
    expected_entries: int,
) -> LoadedPackageRegistry:
    """Load a regular, pinned JSON registry without accepting path substitution."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected package registry SHA-256 must be lowercase hexadecimal")
    if expected_entries <= 0:
        raise ValueError("expected package registry entry count must be positive")
    if path.is_symlink():
        raise ValueError(f"package registry path must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"package registry path is not a regular file: {resolved}")
    digest = hashlib.sha256()
    payload_parts: list[bytes] = []
    with resolved.open("rb", buffering=_READ_CHUNK_SIZE) as stream:
        while chunk := stream.read(_READ_CHUNK_SIZE):
            digest.update(chunk)
            payload_parts.append(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"package registry SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )
    payload = PackageRegistryPayload.model_validate_json(b"".join(payload_parts), strict=True)
    if len(payload.entries) != expected_entries:
        raise ValueError(
            "package registry entry count mismatch: "
            f"expected {expected_entries}, found {len(payload.entries)}"
        )
    return LoadedPackageRegistry(
        path=str(resolved),
        sha256=actual_sha256,
        payload=payload,
    )


def build_package_surface_inventory(
    metadata_rows: Sequence[Mapping[str, Any]],
    *,
    train_document_ids: Sequence[str],
    registry: LoadedPackageRegistry,
) -> PackageSurfaceInventory:
    """Build exact train-only surface support from reviewed category decisions."""

    if not train_document_ids or len(train_document_ids) != len(set(train_document_ids)):
        raise ValueError("train document IDs must be non-empty and unique")
    requested = frozenset(train_document_ids)
    rows_by_document: dict[str, Mapping[str, Any]] = {}
    for row in metadata_rows:
        document_id = row.get("documentId")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("package category metadata has an invalid documentId")
        if document_id in rows_by_document:
            raise ValueError(f"duplicate package category metadata row: {document_id}")
        rows_by_document[document_id] = row
    missing = sorted(requested - rows_by_document.keys())
    if missing:
        raise ValueError(f"package category metadata is missing train documents: {missing[:5]}")

    occurrence_counts: dict[tuple[str | None, str, str], int] = defaultdict(int)
    occurrence_documents: dict[tuple[str | None, str, str], set[str]] = defaultdict(set)
    bucket_documents: dict[tuple[str | None, str], set[str]] = defaultdict(set)
    normalized_owners: dict[str, str | None] = {}
    typed = 0
    fallback = 0
    for document_id in train_document_ids:
        decisions = rows_by_document[document_id].get("packageDecisions")
        if not isinstance(decisions, list):
            raise ValueError(f"packageDecisions must be an array: {document_id}")
        seen_package_ids: set[str] = set()
        for raw_decision in decisions:
            decision = _PackageSurfaceDecision.model_validate(raw_decision, strict=True)
            if decision.packageId in seen_package_ids:
                raise ValueError(
                    f"duplicate package category decision {document_id}:{decision.packageId}"
                )
            seen_package_ids.add(decision.packageId)
            if decision.categoryToken is not None:
                registry.entry(decision.categoryToken)
                typed += 1
            else:
                fallback += 1
            normalized = normalize_printed_surface(decision.sourceTypeDescription)
            previous = normalized_owners.setdefault(normalized, decision.categoryToken)
            if previous != decision.categoryToken:
                raise ValueError(
                    "normalized package surface has conflicting reviewed categories: "
                    f"{normalized!r}"
                )
            exact_key = (
                decision.categoryToken,
                normalized,
                decision.sourceTypeDescription,
            )
            occurrence_counts[exact_key] += 1
            occurrence_documents[exact_key].add(document_id)
            bucket_documents[(decision.categoryToken, normalized)].add(document_id)

    grouped: dict[tuple[str | None, str], list[PackageSurfaceVariant]] = defaultdict(list)
    for key in sorted(occurrence_counts, key=lambda value: (value[0] or "", value[1], value[2])):
        category, normalized, exact = key
        grouped[(category, normalized)].append(
            PackageSurfaceVariant(
                exact_surface=exact,
                occurrences=occurrence_counts[key],
                document_count=len(occurrence_documents[key]),
            )
        )
    buckets = tuple(
        PackageSurfaceBucket(
            category_token=category,
            normalized_surface=normalized,
            variants=tuple(grouped[(category, normalized)]),
            occurrences=sum(row.occurrences for row in grouped[(category, normalized)]),
            document_count=len(bucket_documents[(category, normalized)]),
        )
        for category, normalized in sorted(grouped, key=lambda value: (value[0] or "", value[1]))
    )
    return PackageSurfaceInventory(
        train_document_count=len(train_document_ids),
        decision_count=typed + fallback,
        typed_decision_count=typed,
        fallback_decision_count=fallback,
        buckets=buckets,
    )
