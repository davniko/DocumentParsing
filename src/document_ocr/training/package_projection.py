"""Publish a fail-closed, task-facing package projection for B/L training.

The source relation label deliberately retains every OCR-grounded package level.
The MPCI form, however, has no parent/child package hierarchy, so teaching a
small model to reproduce pallet/skid and generic aggregate levels alongside a
more specific goods-package level adds an avoidable target ambiguity.  This
module creates a new immutable dataset instead of changing that source truth.

Only package roles named by a pinned policy are projected automatically. A
multi-package group with an untyped package is held for review; multiple
direct-goods types remain as legitimate repeated package rows. Removed facts
and the complete original targets remain in the package-metadata sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, cast

from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from document_ocr.atomic import atomic_publish_bytes, read_regular_file_bytes
from document_ocr.config import load_strict_yaml_mapping
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading import BillOfLadingLabel
from document_ocr.label_schemas.bill_of_lading_v3 import (
    BillOfLadingRelationExplicitLabel,
    project_relation_to_normal,
    validate_dual_cargo_consistency,
)
from document_ocr.label_schemas.common import LabelSchemaModel
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DOCUMENT_ID = re.compile(r"^doc_[0-9a-f]{64}$")
_GROUP_ID = re.compile(r"^g[1-9][0-9]*$")
_PACKAGE_ID = re.compile(r"^p[1-9][0-9]*$")


class PackageProjectionError(RuntimeError):
    """A pinned input or package projection violated its contract."""


class _ConfigModel(LabelSchemaModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def _absolute_path(value: str, field_name: str, *, directory: bool) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    normalized = Path(os.path.abspath(value))
    if directory and normalized == Path(normalized.anchor):
        raise ValueError(f"{field_name} must not be the filesystem root")
    return value


class PinnedSource(_ConfigModel):
    dataset_id: NonEmptyString
    root: NonEmptyString
    manifest_sha256: Sha256
    expected_records: PositiveInteger

    @field_validator("dataset_id")
    @classmethod
    def dataset_id_is_safe(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError("dataset_id contains unsupported characters")
        return value

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_path(value, "source root", directory=True)


class PackageRolePolicy(_ConfigModel):
    normalized_outer_types: tuple[NonEmptyString, ...] = Field(min_length=1)
    normalized_generic_types: tuple[NonEmptyString, ...] = Field(min_length=1)

    @field_validator("normalized_outer_types", "normalized_generic_types", mode="before")
    @classmethod
    def sequences_are_frozen(cls, value: Any) -> Any:
        if isinstance(value, tuple):
            return value
        if not isinstance(value, list):
            raise ValueError("package-role values must be YAML sequences")
        return tuple(value)

    @model_validator(mode="after")
    def values_are_normalized_unique_and_disjoint(self) -> PackageRolePolicy:
        outer = tuple(normalize_package_type(value) for value in self.normalized_outer_types)
        generic = tuple(normalize_package_type(value) for value in self.normalized_generic_types)
        if outer != self.normalized_outer_types or generic != self.normalized_generic_types:
            raise ValueError("package-role values must already be normalized")
        if len(outer) != len(set(outer)) or len(generic) != len(set(generic)):
            raise ValueError("package-role values must be unique")
        if set(outer) & set(generic):
            raise ValueError("outer and generic package-role values must be disjoint")
        return self


class ReviewedPackageRoleOverride(_ConfigModel):
    """One source-pinned semantic role decision that does not invent a type."""

    document_id: NonEmptyString
    group_id: NonEmptyString
    package_id: NonEmptyString
    expected_package_sha256: Sha256
    role: Literal["direct_goods"]
    rationale: NonEmptyString

    @field_validator("document_id")
    @classmethod
    def document_id_is_valid(cls, value: str) -> str:
        if _DOCUMENT_ID.fullmatch(value) is None:
            raise ValueError("reviewed document_id must be a canonical document ID")
        return value

    @field_validator("group_id")
    @classmethod
    def group_id_is_valid(cls, value: str) -> str:
        if _GROUP_ID.fullmatch(value) is None:
            raise ValueError("reviewed group_id must be a canonical cargo-group ID")
        return value

    @field_validator("package_id")
    @classmethod
    def package_id_is_valid(cls, value: str) -> str:
        if _PACKAGE_ID.fullmatch(value) is None:
            raise ValueError("reviewed package_id must be a canonical package ID")
        return value


class ProjectionOutput(_ConfigModel):
    root: NonEmptyString
    dataset_id: NonEmptyString

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_path(value, "output root", directory=True)

    @field_validator("dataset_id")
    @classmethod
    def dataset_id_is_safe(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError("output dataset_id contains unsupported characters")
        return value


class PackageProjectionConfig(_ConfigModel):
    schema_version: Literal[1]
    projection: Literal["mpci_bl_task_facing_packages_v1"]
    source: PinnedSource
    policy: PackageRolePolicy
    reviewed_role_overrides: tuple[ReviewedPackageRoleOverride, ...]
    output: ProjectionOutput

    @field_validator("reviewed_role_overrides", mode="before")
    @classmethod
    def reviewed_overrides_are_frozen(cls, value: Any) -> Any:
        if isinstance(value, tuple):
            return value
        if not isinstance(value, list):
            raise ValueError("reviewed_role_overrides must be a YAML sequence")
        return tuple(value)

    @model_validator(mode="after")
    def source_and_output_are_distinct(self) -> PackageProjectionConfig:
        if self.source.dataset_id == self.output.dataset_id:
            raise ValueError("source and output dataset IDs must differ")
        if Path(self.source.root).resolve(strict=False) == Path(self.output.root).resolve(
            strict=False
        ):
            raise ValueError("source and output roots must differ")
        keys = tuple(
            (row.document_id, row.group_id, row.package_id) for row in self.reviewed_role_overrides
        )
        if len(keys) != len(set(keys)):
            raise ValueError("reviewed_role_overrides contains a duplicate package key")
        return self


PackageRole = Literal["outer_transport", "generic_aggregate", "direct_goods", "unknown"]
GroupStatus = Literal[
    "projected",
    "retained_same_type_facts",
    "retained_non_hierarchical",
    "held_ambiguous",
]


@dataclass(frozen=True, slots=True)
class PackageDiagnosis:
    package_id: str
    normalized_type: str | None
    role: PackageRole
    role_source: Literal["policy", "reviewed_override"]
    value: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GroupDiagnosis:
    group_id: str
    status: GroupStatus
    reason: str
    packages: tuple[PackageDiagnosis, ...]
    retained_package_ids: tuple[str, ...]
    metadata_package_ids: tuple[str, ...]


def normalize_package_type(value: str) -> str:
    """Normalize only for matching against the pinned role policy."""

    return " ".join(re.sub(r"[^A-Z0-9]+", " ", value.upper()).split())


def classify_package_role(package: dict[str, Any], policy: PackageRolePolicy) -> PackageRole:
    value = package.get("typeDescription", package.get("typeCategory"))
    if value is None:
        return "unknown"
    if not isinstance(value, str):
        raise PackageProjectionError("package type must be text or null")
    normalized = normalize_package_type(value)
    # v5 category tokens name the same physical type as v4 descriptions. The
    # namespace prefix is not part of the package type, and treating it as one
    # silently made a pallet look like a direct goods package.
    if "typeDescription" not in package and normalized.startswith("PACKAGE "):
        normalized = normalized.removeprefix("PACKAGE ")
    if normalized in set(policy.normalized_outer_types):
        return "outer_transport"
    if normalized in set(policy.normalized_generic_types):
        return "generic_aggregate"
    return "direct_goods"


def diagnose_package_group(
    group_id: str,
    packages: Sequence[dict[str, Any]],
    policy: PackageRolePolicy,
    reviewed_role_overrides: Mapping[str, ReviewedPackageRoleOverride] | None = None,
) -> GroupDiagnosis:
    """Classify one source-ordered package group without inferring a hierarchy."""

    reviewed = reviewed_role_overrides or {}
    source_package_ids = {
        package.get("packageId") for package in packages if isinstance(package, dict)
    }
    extra_override_ids = set(reviewed) - source_package_ids
    if extra_override_ids:
        raise PackageProjectionError(
            f"reviewed package override is absent from {group_id}: {sorted(extra_override_ids)!r}"
        )
    diagnosed: list[PackageDiagnosis] = []
    for package in packages:
        package_id = package.get("packageId")
        if not isinstance(package_id, str) or _PACKAGE_ID.fullmatch(package_id) is None:
            raise PackageProjectionError(f"invalid package ID in {group_id}")
        value = package.get("typeDescription", package.get("typeCategory"))
        normalized = normalize_package_type(value) if isinstance(value, str) else None
        automatic_role = classify_package_role(package, policy)
        override = reviewed.get(package_id)
        role: PackageRole = automatic_role
        role_source: Literal["policy", "reviewed_override"] = "policy"
        if override is not None:
            if override.group_id != group_id or override.package_id != package_id:
                raise PackageProjectionError(
                    f"reviewed package override identity differs in {group_id}/{package_id}"
                )
            if automatic_role != "unknown":
                raise PackageProjectionError(
                    f"reviewed package override targets an already typed fact: "
                    f"{group_id}/{package_id}"
                )
            if sha256_bytes(canonical_json_bytes(package)) != override.expected_package_sha256:
                raise PackageProjectionError(
                    f"reviewed package override source hash differs: {group_id}/{package_id}"
                )
            role = override.role
            role_source = "reviewed_override"
        diagnosed.append(
            PackageDiagnosis(
                package_id=package_id,
                normalized_type=normalized,
                role=role,
                role_source=role_source,
                value=json.loads(json.dumps(package)),
            )
        )
    package_rows = tuple(diagnosed)
    all_ids = tuple(row.package_id for row in package_rows)
    if len(package_rows) <= 1:
        return GroupDiagnosis(
            group_id=group_id,
            status="retained_non_hierarchical",
            reason="The cargo group has at most one package fact.",
            packages=package_rows,
            retained_package_ids=all_ids,
            metadata_package_ids=(),
        )

    if any(row.role == "unknown" for row in package_rows):
        return GroupDiagnosis(
            group_id=group_id,
            status="held_ambiguous",
            reason=(
                "A multi-package group contains a quantity-only package fact; its role cannot "
                "be established without inventing a package type."
            ),
            packages=package_rows,
            retained_package_ids=(),
            metadata_package_ids=(),
        )

    direct = tuple(row for row in package_rows if row.role == "direct_goods")
    outer = tuple(row for row in package_rows if row.role == "outer_transport")
    generic = tuple(row for row in package_rows if row.role == "generic_aggregate")
    if direct and (outer or generic):
        return GroupDiagnosis(
            group_id=group_id,
            status="projected",
            reason=(
                "The pinned policy identifies direct-goods package facts; every direct fact "
                "remains source ordered while explicitly named outer transport and generic "
                "aggregate facts move to metadata."
            ),
            packages=package_rows,
            retained_package_ids=tuple(row.package_id for row in direct),
            metadata_package_ids=tuple(
                row.package_id
                for row in package_rows
                if row.role in {"outer_transport", "generic_aggregate"}
            ),
        )

    if outer and generic and not direct:
        return GroupDiagnosis(
            group_id=group_id,
            status="projected",
            reason=(
                "No inner type is printed; the specific outer transport package remains the "
                "task-facing package and only the generic aggregate fact moves to metadata."
            ),
            packages=package_rows,
            retained_package_ids=tuple(row.package_id for row in outer),
            metadata_package_ids=tuple(row.package_id for row in generic),
        )

    normalized_types = {row.normalized_type for row in package_rows}
    return GroupDiagnosis(
        group_id=group_id,
        status=(
            "retained_same_type_facts"
            if len(normalized_types) == 1
            else "retained_non_hierarchical"
        ),
        reason=(
            "Repeated facts carry the same normalized package type and remain source ordered."
            if len(normalized_types) == 1
            else "No pinned outer/generic hierarchy is present; all package facts remain."
        ),
        packages=package_rows,
        retained_package_ids=all_ids,
        metadata_package_ids=(),
    )


def load_package_projection_config(path: Path) -> PackageProjectionConfig:
    try:
        return PackageProjectionConfig.model_validate(load_strict_yaml_mapping(path), strict=True)
    except ValueError as error:
        raise PackageProjectionError(
            f"invalid package projection config: {path}: {error}"
        ) from error


def _strict_json(payload: bytes, *, context: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise PackageProjectionError(f"{context} is not strict JSON") from error
    if not isinstance(value, dict):
        raise PackageProjectionError(f"{context} must be a JSON object")
    return value


def _strict_jsonl(payload: bytes, *, context: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for row_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise PackageProjectionError(f"{context} has a blank row at {row_number}")
        rows.append(_strict_json(line, context=f"{context} row {row_number}"))
    return tuple(rows)


def _manifest_file(
    root: Path, manifest: dict[str, Any], *, kind: str
) -> tuple[Path, bytes, tuple[dict[str, Any], ...], dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise PackageProjectionError("source manifest files must be a list")
    entries = [row for row in files if isinstance(row, dict) and row.get("kind") == kind]
    if len(entries) != 1:
        raise PackageProjectionError(f"source manifest must contain one {kind} file")
    entry = cast(dict[str, Any], entries[0])
    relative = entry.get("path")
    digest = entry.get("sha256")
    byte_count = entry.get("bytes")
    row_count = entry.get("rows", entry.get("records"))
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise PackageProjectionError(f"source {kind} manifest entry is malformed")
    path = (root / relative).resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise PackageProjectionError(f"source {kind} path is invalid: {path}")
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != digest or len(payload) != byte_count:
        raise PackageProjectionError(f"source {kind} bytes/hash differ: {path}")
    rows = _strict_jsonl(payload, context=f"source {kind}")
    if len(rows) != row_count:
        raise PackageProjectionError(f"source {kind} row count differs")
    return path, payload, rows, entry


def _load_source(
    source: PinnedSource,
) -> tuple[
    bytes,
    dict[str, Any],
    tuple[dict[str, Any], ...],
    dict[str, dict[str, Any]],
]:
    root = Path(source.root).resolve(strict=True)
    if not root.is_dir():
        raise PackageProjectionError(f"source dataset root is not a directory: {root}")
    manifest_path = (root / "manifest.json").resolve(strict=True)
    if root not in manifest_path.parents or not manifest_path.is_file():
        raise PackageProjectionError("source manifest is absent")
    manifest_payload = read_regular_file_bytes(manifest_path)
    if sha256_bytes(manifest_payload) != source.manifest_sha256:
        raise PackageProjectionError("source manifest SHA-256 differs")
    manifest = _strict_json(manifest_payload, context="source manifest")
    if manifest.get("datasetId", manifest.get("dataset_id")) != source.dataset_id:
        raise PackageProjectionError("source dataset ID differs")
    _, _, records, _ = _manifest_file(root, manifest, kind="training_records")
    _, _, lineage_rows, _ = _manifest_file(root, manifest, kind="lineage")
    if len(records) != source.expected_records or len(lineage_rows) != source.expected_records:
        raise PackageProjectionError("source record/lineage count differs from config")
    lineage: dict[str, dict[str, Any]] = {}
    for row in lineage_rows:
        document_id = row.get("documentId")
        if not isinstance(document_id, str) or _DOCUMENT_ID.fullmatch(document_id) is None:
            raise PackageProjectionError("source lineage contains an invalid document ID")
        if document_id in lineage:
            raise PackageProjectionError(f"duplicate source lineage: {document_id}")
        lineage[document_id] = row
    return manifest_payload, manifest, records, lineage


def _canonical_labels(
    record: dict[str, Any], *, document_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        normal = BillOfLadingLabel.model_validate_json(
            canonical_json_bytes(record["normalTarget"]), strict=True
        )
        relation = BillOfLadingRelationExplicitLabel.model_validate_json(
            canonical_json_bytes(record["target"]), strict=True
        )
        validate_dual_cargo_consistency(normal, relation)
        projected_normal = project_relation_to_normal(relation)
    except (KeyError, TypeError, ValueError) as error:
        raise PackageProjectionError(
            f"source label validation failed for {document_id}: {error}"
        ) from error
    if projected_normal.canonical_target() != normal.canonical_target():
        raise PackageProjectionError(f"source normal/relation projection differs: {document_id}")
    return normal.canonical_target(), relation.canonical_target()


def _project_relation_target(
    target: dict[str, Any], diagnoses: Sequence[GroupDiagnosis]
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    projected = json.loads(json.dumps(target))
    patch = cast(dict[str, Any], projected["documentPatch"])
    diagnosis_by_group = {row.group_id: row for row in diagnoses}
    retained_old_ids = {
        package_id for diagnosis in diagnoses for package_id in diagnosis.retained_package_ids
    }
    source_packages = cast(list[dict[str, Any]], patch.get("cargoPackages", []))
    retained_packages = [
        row for row in source_packages if cast(str, row["packageId"]) in retained_old_ids
    ]
    old_to_new = {
        cast(str, row["packageId"]): f"p{index}"
        for index, row in enumerate(retained_packages, start=1)
    }
    for row in retained_packages:
        row["packageId"] = old_to_new[cast(str, row["packageId"])]
    if retained_packages:
        patch["cargoPackages"] = retained_packages
    else:
        patch.pop("cargoPackages", None)

    allocation_audit: list[dict[str, Any]] = []
    projected_allocations: list[dict[str, Any]] = []
    for source_allocation in cast(list[dict[str, Any]], patch.get("cargoAllocationGroups", [])):
        original = json.loads(json.dumps(source_allocation))
        group_id = cast(str, source_allocation["groupId"])
        diagnosis = diagnosis_by_group[group_id]
        removed = bool(diagnosis.metadata_package_ids)
        referenced_ids = set(source_allocation.get("packageIds", []))
        referenced_ids.update(
            allocation["packageId"]
            for allocation in source_allocation["allocations"]
            if allocation.get("packageId") is not None
        )
        # A printed allocation explicitly tied to a retained package remains
        # valid. Only an unlinked quantity or a link to removed metadata loses
        # its package-level meaning when the target is projected.
        must_downgrade = (
            removed
            and source_allocation["coverage"] != "container_membership_only"
            and (not referenced_ids or not referenced_ids <= set(diagnosis.retained_package_ids))
        )
        if must_downgrade:
            seen: set[str] = set()
            memberships: list[dict[str, Any]] = []
            for allocation in source_allocation["allocations"]:
                container = cast(str, allocation["containerNumber"])
                if container not in seen:
                    seen.add(container)
                    memberships.append({"containerNumber": container})
            replacement = {
                "groupId": group_id,
                "coverage": "container_membership_only",
                "packageIds": [],
                "allocations": memberships,
            }
            projected_allocations.append(replacement)
            allocation_audit.append(
                {
                    "groupId": group_id,
                    "decision": "downgraded_to_container_membership",
                    "reason": (
                        "The original allocation refers to package metadata removed from the "
                        "training target; container membership is preserved without transferring "
                        "an outer/aggregate quantity to an inner package."
                    ),
                    "original": original,
                    "projected": replacement,
                }
            )
            continue
        replacement = json.loads(json.dumps(source_allocation))
        replacement["packageIds"] = [
            old_to_new[package_id] for package_id in replacement.get("packageIds", [])
        ]
        for allocation in replacement["allocations"]:
            if allocation.get("packageId") is not None:
                allocation["packageId"] = old_to_new[allocation["packageId"]]
        projected_allocations.append(replacement)
        allocation_audit.append(
            {
                "groupId": group_id,
                "decision": "preserved",
                "reason": "The allocation does not depend on a removed package fact.",
                "original": original,
                "projected": replacement,
            }
        )
    if projected_allocations:
        patch["cargoAllocationGroups"] = projected_allocations
    else:
        patch.pop("cargoAllocationGroups", None)
    return projected, tuple(allocation_audit)


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _publish(path: Path, payload: bytes) -> None:
    if path.exists() and read_regular_file_bytes(path) != payload:
        raise PackageProjectionError(
            f"immutable publication already exists with different bytes: {path}"
        )
    atomic_publish_bytes(path, payload)


def _file_entry(path: Path, root: Path, payload: bytes, rows: int, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": path.relative_to(root).as_posix(),
        "rows": rows,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _diagnosis_json(diagnosis: GroupDiagnosis) -> dict[str, Any]:
    return {
        "groupId": diagnosis.group_id,
        "status": diagnosis.status,
        "reason": diagnosis.reason,
        "packages": [
            {
                "packageId": row.package_id,
                "normalizedType": row.normalized_type,
                "role": row.role,
                "roleSource": row.role_source,
                "value": row.value,
            }
            for row in diagnosis.packages
        ],
        "retainedPackageIds": list(diagnosis.retained_package_ids),
        "metadataPackageIds": list(diagnosis.metadata_package_ids),
    }


def _report(summary: dict[str, Any], dataset_id: str) -> bytes:
    vocabulary = cast(dict[str, dict[str, Any]], summary["packageRoleVocabulary"])
    direct = vocabulary["direct_goods"]
    outer = vocabulary["outer_transport"]
    generic = vocabulary["generic_aggregate"]
    unknown = vocabulary["unknown"]
    unchanged_documents = cast(int, summary["documentsUnchangedByPackageProjection"])
    text = f"""# MPCI B/L task-facing package projection

Dataset: `{dataset_id}`

This is a non-destructive training projection. The source labels remain unchanged, and
`package-metadata.jsonl` retains each complete original normal/relation target plus every package
decision. The MPCI form has multiple package rows but no parent/child package relation; therefore
only exact policy-classified pallet/skid/transport levels and generic aggregate `PACKAGE`/`PKG`
levels move to metadata when one or more explicit direct-goods package facts remain.

## Result

| Measure | Count |
|---|---:|
| Source records | {summary["sourceRecords"]:,} |
| Projected training records | {summary["projectedRecords"]:,} |
| Fail-closed held records | {summary["heldRecords"]:,} |
| Documents with a package projection | {summary["documentsWithPackageProjection"]:,} |
| Documents unchanged by the package projection | {unchanged_documents:,} |
| Multi-package documents | {summary["multiPackageDocuments"]:,} |
| Multi-package groups audited | {summary["multiPackageGroups"]:,} |
| Explicitly projected groups | {summary["projectedGroups"]:,} |
| Unchanged multi-package groups | {summary["unchangedMultiPackageGroups"]:,} |
| Ambiguous groups held | {summary["ambiguousGroups"]:,} |
| Reviewed quantity-only roles retained | {summary["reviewedRoleOverridesUsed"]:,} |
| Package facts moved to metadata | {summary["metadataPackageFacts"]:,} |
| Allocations downgraded to membership-only | {summary["allocationGroupsDowngraded"]:,} |

## Observed package-role vocabulary

| Role | Occurrences | Normalized printed types |
|---|---:|---:|
| Direct goods | {direct["occurrences"]:,} | {direct["normalizedTypes"]:,} |
| Outer transport | {outer["occurrences"]:,} | {outer["normalizedTypes"]:,} |
| Generic aggregate | {generic["occurrences"]:,} | {generic["normalizedTypes"]:,} |
| Untyped | {unknown["occurrences"]:,} | 0 |

## Fail-closed boundary

A document is held only when a multi-package group contains an untyped quantity whose role has not
been established by an exact, source-pinned review. A reviewed quantity-only direct-goods fact is
retained with its type absent; the projection never invents a type. Multiple direct-goods types are
legitimate repeated MPCI package rows and remain source ordered. Unresolved rows appear in
`review-queue.jsonl`; they are not silently discarded or used as training supervision.

When an original allocation names a removed outer/aggregate package, the projected target keeps
only the printed container membership. It never transfers the removed package quantity to a
retained inner fact. The complete allocation remains in the metadata sidecar.

`task-constraints.json` is published and schema-bound with empty package/container category-token
vocabularies. This removes the obsolete category branch from the prompt schema while retaining
printed `typeDescription` targets.
"""
    return text.encode("utf-8")


def project_package_hierarchy(config: PackageProjectionConfig) -> Path:
    source_manifest_payload, _, source_records, source_lineage = _load_source(config.source)
    projected_records: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    raw_hashes: set[str] = set()
    counters: Counter[str] = Counter()
    multi_documents: set[str] = set()
    changed_documents: set[str] = set()
    role_occurrences: Counter[str] = Counter()
    role_types: dict[str, set[str]] = defaultdict(set)
    override_index: dict[tuple[str, str], dict[str, ReviewedPackageRoleOverride]] = defaultdict(
        dict
    )
    configured_override_keys: set[tuple[str, str, str]] = set()
    used_override_keys: set[tuple[str, str, str]] = set()
    for override in config.reviewed_role_overrides:
        override_index[(override.document_id, override.group_id)][override.package_id] = override
        configured_override_keys.add((override.document_id, override.group_id, override.package_id))

    for source_row_number, record in enumerate(source_records, start=1):
        document_id = record.get("documentId")
        raw_text = record.get("joinedRawText")
        raw_hash = record.get("joinedRawTextSha256")
        if (
            not isinstance(document_id, str)
            or _DOCUMENT_ID.fullmatch(document_id) is None
            or not isinstance(raw_text, str)
            or not isinstance(raw_hash, str)
            or sha256_bytes(raw_text.encode("utf-8")) != raw_hash
        ):
            raise PackageProjectionError(
                f"source record identity/raw OCR is invalid at row {source_row_number}"
            )
        if document_id in document_ids or raw_hash in raw_hashes:
            raise PackageProjectionError(f"duplicate source document/raw OCR: {document_id}")
        document_ids.add(document_id)
        raw_hashes.add(raw_hash)
        if document_id not in source_lineage:
            raise PackageProjectionError(f"source lineage is absent: {document_id}")
        normal_target, relation_target = _canonical_labels(record, document_id=document_id)
        relation_patch = cast(dict[str, Any], relation_target["documentPatch"])
        packages_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for package in relation_patch.get("cargoPackages", []):
            packages_by_group[cast(str, package["groupId"])].append(package)
            role = classify_package_role(package, config.policy)
            role_occurrences[role] += 1
            value = package.get("typeDescription", package.get("typeCategory"))
            if isinstance(value, str):
                role_types[role].add(normalize_package_type(value))
        group_ids = [cast(str, row["groupId"]) for row in relation_patch.get("cargoGroups", [])]
        diagnoses_list: list[GroupDiagnosis] = []
        for group_id in group_ids:
            reviewed = override_index.get((document_id, group_id), {})
            diagnosis = diagnose_package_group(
                group_id,
                packages_by_group.get(group_id, ()),
                config.policy,
                reviewed,
            )
            diagnoses_list.append(diagnosis)
            for package in diagnosis.packages:
                if package.role_source == "reviewed_override":
                    used_override_keys.add((document_id, group_id, package.package_id))
        diagnoses = tuple(diagnoses_list)
        multi = tuple(row for row in diagnoses if len(row.packages) > 1)
        if multi:
            multi_documents.add(document_id)
        for diagnosis in multi:
            counters["multiPackageGroups"] += 1
            if diagnosis.status == "projected":
                counters["projectedGroups"] += 1
            elif diagnosis.status == "held_ambiguous":
                counters["ambiguousGroups"] += 1
            else:
                counters["unchangedMultiPackageGroups"] += 1
            inventory_rows.append(
                {
                    "documentId": document_id,
                    "joinedRawTextSha256": raw_hash,
                    **_diagnosis_json(diagnosis),
                }
            )
        is_held = any(row.status == "held_ambiguous" for row in diagnoses)
        if any(row.metadata_package_ids for row in diagnoses):
            changed_documents.add(document_id)
        original_record_hash = sha256_bytes(canonical_json_bytes(record))
        metadata: dict[str, Any] = {
            "documentId": document_id,
            "status": "held" if is_held else "projected",
            "sourceRecordRow": source_row_number,
            "sourceRecordCanonicalSha256": original_record_hash,
            "sourceNormalTarget": normal_target,
            "sourceRelationTarget": relation_target,
            "groupDiagnoses": [_diagnosis_json(row) for row in diagnoses],
            "allocationProjection": [],
        }
        if is_held:
            counters["heldRecords"] += 1
            review_row = {
                **metadata,
                "joinedRawText": raw_text,
                "joinedRawTextSha256": raw_hash,
            }
            review_rows.append(review_row)
            metadata_rows.append(metadata)
            metadata_hash = sha256_bytes(canonical_json_bytes(metadata))
            lineage_rows.append(
                {
                    "documentId": document_id,
                    "status": "held",
                    "sourceDatasetId": config.source.dataset_id,
                    "sourceDatasetManifestSha256": config.source.manifest_sha256,
                    "sourceRecordCanonicalSha256": original_record_hash,
                    "sourceLineage": source_lineage[document_id],
                    "packageMetadataCanonicalSha256": metadata_hash,
                }
            )
            continue

        projected_target, allocation_audit = _project_relation_target(relation_target, diagnoses)
        try:
            projected_relation = BillOfLadingRelationExplicitLabel.model_validate_json(
                canonical_json_bytes(projected_target), strict=True
            )
            projected_normal = project_relation_to_normal(projected_relation)
            validate_dual_cargo_consistency(projected_normal, projected_relation)
        except ValueError as error:
            raise PackageProjectionError(
                f"projected label validation failed for {document_id}: {error}"
            ) from error
        target = projected_relation.canonical_target()
        normal = projected_normal.canonical_target()
        projected_record = {
            "documentId": document_id,
            "joinedRawText": raw_text,
            "joinedRawTextSha256": raw_hash,
            "normalTarget": normal,
            "target": target,
        }
        metadata["allocationProjection"] = list(allocation_audit)
        metadata["projectedNormalTargetSha256"] = sha256_bytes(canonical_json_bytes(normal))
        metadata["projectedRelationTargetSha256"] = sha256_bytes(canonical_json_bytes(target))
        metadata_rows.append(metadata)
        projected_records.append(projected_record)
        counters["projectedRecords"] += 1
        moved = sum(len(row.metadata_package_ids) for row in diagnoses)
        counters["metadataPackageFacts"] += moved
        counters["outerMetadataPackageFacts"] += sum(
            1
            for row in diagnoses
            for package in row.packages
            if package.package_id in row.metadata_package_ids and package.role == "outer_transport"
        )
        counters["genericMetadataPackageFacts"] += sum(
            1
            for row in diagnoses
            for package in row.packages
            if package.package_id in row.metadata_package_ids
            and package.role == "generic_aggregate"
        )
        counters["allocationGroupsDowngraded"] += sum(
            row["decision"] == "downgraded_to_container_membership" for row in allocation_audit
        )
        metadata_hash = sha256_bytes(canonical_json_bytes(metadata))
        projected_hash = sha256_bytes(canonical_json_bytes(projected_record))
        lineage_rows.append(
            {
                "documentId": document_id,
                "status": "projected",
                "sourceDatasetId": config.source.dataset_id,
                "sourceDatasetManifestSha256": config.source.manifest_sha256,
                "sourceRecordCanonicalSha256": original_record_hash,
                "sourceLineage": source_lineage[document_id],
                "packageMetadataCanonicalSha256": metadata_hash,
                "projectedRecordCanonicalSha256": projected_hash,
            }
        )

    if len(document_ids) != config.source.expected_records:
        raise PackageProjectionError("source record count differs after validation")
    if len(projected_records) + len(review_rows) != len(source_records):
        raise PackageProjectionError("projected plus held records do not cover the source")
    if used_override_keys != configured_override_keys:
        unused = sorted(configured_override_keys - used_override_keys)
        unexpected = sorted(used_override_keys - configured_override_keys)
        raise PackageProjectionError(
            f"reviewed package overrides were not consumed exactly once: "
            f"unused={unused!r}, unexpected={unexpected!r}"
        )

    summary = {
        "schemaVersion": 1,
        "policy": config.projection,
        "sourceRecords": len(source_records),
        "projectedRecords": len(projected_records),
        "heldRecords": len(review_rows),
        "documentsWithPackageProjection": len(changed_documents),
        "documentsUnchangedByPackageProjection": (len(projected_records) - len(changed_documents)),
        "multiPackageDocuments": len(multi_documents),
        "multiPackageGroups": counters["multiPackageGroups"],
        "projectedGroups": counters["projectedGroups"],
        "unchangedMultiPackageGroups": counters["unchangedMultiPackageGroups"],
        "ambiguousGroups": counters["ambiguousGroups"],
        "reviewedRoleOverridesUsed": len(used_override_keys),
        "metadataPackageFacts": counters["metadataPackageFacts"],
        "outerMetadataPackageFacts": counters["outerMetadataPackageFacts"],
        "genericMetadataPackageFacts": counters["genericMetadataPackageFacts"],
        "allocationGroupsDowngraded": counters["allocationGroupsDowngraded"],
        "schemaValidatedProjectedRecords": len(projected_records),
        "packageRoleVocabulary": {
            role: {
                "occurrences": role_occurrences[role],
                "normalizedTypes": len(role_types[role]),
                "values": sorted(role_types[role]),
            }
            for role in (
                "outer_transport",
                "generic_aggregate",
                "direct_goods",
                "unknown",
            )
        },
    }

    output_root = Path(config.output.root).resolve(strict=False)
    if output_root == Path(output_root.anchor):
        raise PackageProjectionError("refusing to publish at filesystem root")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    task = get_training_task("bill_of_lading_relation_explicit_v3")
    empty_package_registry = {
        "registry": "package_category_tokens",
        "policy": "printed_type_description_only",
        "tokens": [],
    }
    empty_container_registry = {
        "registry": "container_category_tokens",
        "policy": "printed_type_description_only",
        "tokens": [],
    }
    constraints = RelationExplicitTaskConstraints.model_validate(
        {
            "schemaVersion": 1,
            "task": "bill_of_lading_relation_explicit_v3",
            "basePromptSchemaSha256": task.base_prompt_schema_sha256(),
            "targetSchemaSha256": sha256_bytes(
                canonical_json_bytes(
                    BillOfLadingRelationExplicitLabel.model_json_schema(mode="serialization")
                )
            ),
            "packageRegistrySha256": sha256_bytes(canonical_json_bytes(empty_package_registry)),
            "containerRegistrySha256": sha256_bytes(canonical_json_bytes(empty_container_registry)),
            "packageCategoryTokens": (),
            "containerCategoryTokens": (),
        },
        strict=True,
    )
    task.bind_constraints(constraints)
    constraints_payload = canonical_json_bytes(constraints.model_dump(mode="json")) + b"\n"
    payloads = {
        "records.jsonl": (
            _jsonl_bytes(projected_records),
            len(projected_records),
            "training_records",
        ),
        "lineage.jsonl": (_jsonl_bytes(lineage_rows), len(lineage_rows), "lineage"),
        "package-metadata.jsonl": (
            _jsonl_bytes(metadata_rows),
            len(metadata_rows),
            "package_metadata",
        ),
        "multi-package-audit.jsonl": (
            _jsonl_bytes(inventory_rows),
            len(inventory_rows),
            "multi_package_audit",
        ),
        "review-queue.jsonl": (
            _jsonl_bytes(review_rows),
            len(review_rows),
            "review_queue",
        ),
        "summary.json": (canonical_json_bytes(summary) + b"\n", 1, "summary"),
        "REPORT.md": (_report(summary, config.output.dataset_id), 1, "report"),
        "task-constraints.json": (
            constraints_payload,
            1,
            "training_task_constraints",
        ),
    }
    files: list[dict[str, Any]] = []
    for relative, (payload, rows, kind) in payloads.items():
        path = output_root / relative
        _publish(path, payload)
        files.append(_file_entry(path, output_root, payload, rows, kind))
    manifest = {
        "schemaVersion": 1,
        "datasetId": config.output.dataset_id,
        "task": "bill_of_lading_relation_single_source_v4_task_facing_packages",
        "records": len(projected_records),
        "trainingReady": True,
        "failClosedHeldRecords": len(review_rows),
        "sourceDataset": {
            "datasetId": config.source.dataset_id,
            "manifestSha256": sha256_bytes(source_manifest_payload),
            "records": len(source_records),
        },
        "packagePolicy": {
            "contract": config.projection,
            "outerTypes": list(config.policy.normalized_outer_types),
            "genericTypes": list(config.policy.normalized_generic_types),
            "ambiguousGroupsAreHeld": True,
            "reviewedRoleOverrides": [
                row.model_dump(mode="json") for row in config.reviewed_role_overrides
            ],
            "sourceTargetsRetainedInMetadata": True,
            "mpciParentChildPackageHierarchySupported": False,
        },
        "training_task_constraints": {
            "path": "task-constraints.json",
            "sha256": sha256_bytes(constraints_payload),
        },
        "audit": summary,
        "files": files,
    }
    manifest_path = output_root / "manifest.json"
    _publish(manifest_path, canonical_json_bytes(manifest) + b"\n")
    return manifest_path


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise PackageProjectionError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _ArgumentParser(
        prog="python -m document_ocr.training.package_projection",
        description="Publish a non-destructive task-facing B/L package projection.",
    )
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    config = load_package_projection_config(arguments.config)
    manifest = project_package_hierarchy(config)
    print(
        json.dumps(
            {"command": "package-project", "manifest": str(manifest), "status": "complete"},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
