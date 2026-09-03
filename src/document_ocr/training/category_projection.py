"""Publish registry-backed package categories over a task-facing B/L dataset.

The source dataset deliberately keeps printed package descriptions.  This
transform replaces a description with a readable category token only when an
exact, reviewed source-string decision selects one authoritative registry
entry.  Unresolved descriptions remain verbatim fallbacks, container types are
never categorized, and complete source targets remain in a metadata sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, cast

from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from document_ocr.atomic import atomic_publish_bytes, read_regular_file_bytes
from document_ocr.config import load_strict_yaml_mapping
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading import BillOfLadingLabel
from document_ocr.label_schemas.bill_of_lading_v3 import (
    BillOfLadingRelationExplicitLabel,
    CategoryToken,
    validate_dual_cargo_consistency,
)
from document_ocr.label_schemas.common import LabelSchemaModel
from document_ocr.semantic_v3.transform import CategoryAssignments, CategoryRegistry
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DOCUMENT_ID = re.compile(r"^doc_[0-9a-f]{64}$")


class CategoryProjectionError(RuntimeError):
    """A pinned categorical input or projection violated its contract."""


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


class PinnedDataset(_ConfigModel):
    dataset_id: NonEmptyString
    root: NonEmptyString
    manifest_sha256: Sha256
    expected_records: PositiveInteger

    @field_validator("dataset_id")
    @classmethod
    def dataset_id_is_safe(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError("source dataset_id contains unsupported characters")
        return value

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_path(value, "source root", directory=True)


class PinnedJson(_ConfigModel):
    path: NonEmptyString
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def path_is_absolute_json(cls, value: str) -> str:
        _absolute_path(value, "pinned JSON path", directory=False)
        if Path(value).suffix != ".json":
            raise ValueError("pinned JSON path must end in .json")
        return value


class PinnedJsonl(_ConfigModel):
    path: NonEmptyString
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def path_is_absolute_jsonl(cls, value: str) -> str:
        _absolute_path(value, "pinned JSONL path", directory=False)
        if Path(value).suffix != ".jsonl":
            raise ValueError("pinned JSONL path must end in .jsonl")
        return value


class RegistryInputs(_ConfigModel):
    package: PinnedJson
    container: PinnedJson
    prior_assignments: PinnedJson
    prior_projection_inventory: PinnedJsonl | None = None


class ExtensionResolutionGroup(_ConfigModel):
    category_token: CategoryToken
    source_type_descriptions: tuple[NonEmptyString, ...] = Field(min_length=1)
    rationale: NonEmptyString

    @field_validator("source_type_descriptions", mode="before")
    @classmethod
    def descriptions_are_frozen(cls, value: Any) -> Any:
        if isinstance(value, tuple):
            return value
        if not isinstance(value, list):
            raise ValueError("source_type_descriptions must be a YAML sequence")
        return tuple(value)

    @model_validator(mode="after")
    def descriptions_are_unique(self) -> ExtensionResolutionGroup:
        if len(self.source_type_descriptions) != len(set(self.source_type_descriptions)):
            raise ValueError("an extension group contains duplicate source descriptions")
        return self


class UnresolvedDescription(_ConfigModel):
    source_type_description: NonEmptyString
    rationale: NonEmptyString


class ReviewPolicy(_ConfigModel):
    reviewed_by: NonEmptyString
    reviewed_on: date
    expected_package_occurrences: PositiveInteger
    expected_package_variants: PositiveInteger
    expected_container_occurrences: PositiveInteger
    expected_container_variants: PositiveInteger
    extension_resolution_groups: tuple[ExtensionResolutionGroup, ...]
    unresolved_descriptions: tuple[UnresolvedDescription, ...]

    @field_validator(
        "extension_resolution_groups", "unresolved_descriptions", mode="before"
    )
    @classmethod
    def review_rows_are_frozen(cls, value: Any) -> Any:
        if isinstance(value, tuple):
            return value
        if not isinstance(value, list):
            raise ValueError("review decisions must be YAML sequences")
        return tuple(value)

    @model_validator(mode="after")
    def review_keys_are_unique(self) -> ReviewPolicy:
        resolved = tuple(
            description
            for group in self.extension_resolution_groups
            for description in group.source_type_descriptions
        )
        unresolved = tuple(row.source_type_description for row in self.unresolved_descriptions)
        if len(resolved) != len(set(resolved)):
            raise ValueError("an extension source description is resolved more than once")
        if len(unresolved) != len(set(unresolved)):
            raise ValueError("an unresolved source description is duplicated")
        if set(resolved) & set(unresolved):
            raise ValueError("a source description is both resolved and unresolved")
        return self


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


class CategoryProjectionConfig(_ConfigModel):
    schema_version: Literal[1]
    projection: Literal["mpci_bl_task_facing_package_categories_v1"]
    source: PinnedDataset
    registries: RegistryInputs
    review: ReviewPolicy
    output: ProjectionOutput

    @model_validator(mode="after")
    def roots_and_ids_are_distinct(self) -> CategoryProjectionConfig:
        if self.source.dataset_id == self.output.dataset_id:
            raise ValueError("source and output dataset IDs must differ")
        if Path(self.source.root).resolve(strict=False) == Path(self.output.root).resolve(
            strict=False
        ):
            raise ValueError("source and output roots must differ")
        return self


def load_category_projection_config(path: Path) -> CategoryProjectionConfig:
    try:
        return CategoryProjectionConfig.model_validate(
            load_strict_yaml_mapping(path), strict=True
        )
    except ValueError as error:
        raise CategoryProjectionError(
            f"invalid category projection config: {path}: {error}"
        ) from error


def _canonical_file(path: Path, *, context: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CategoryProjectionError(f"missing {context}: {path}") from error
    if resolved != absolute or not resolved.is_file() or resolved.is_symlink():
        raise CategoryProjectionError(f"{context} must be a canonical regular file: {path}")
    return resolved


def _strict_json(payload: bytes, *, context: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError) as error:
        raise CategoryProjectionError(f"{context} is not strict JSON") from error
    if not isinstance(value, dict):
        raise CategoryProjectionError(f"{context} must be a JSON object")
    return value


def _strict_jsonl(payload: bytes, *, context: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for row_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise CategoryProjectionError(f"{context} has a blank row at {row_number}")
        rows.append(_strict_json(line, context=f"{context} row {row_number}"))
    return tuple(rows)


def _load_pinned_model[ModelT: LabelSchemaModel](
    configured: PinnedJson, model: type[ModelT], *, context: str
) -> tuple[ModelT, bytes]:
    path = _canonical_file(Path(configured.path), context=context)
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != configured.sha256:
        raise CategoryProjectionError(f"{context} SHA-256 differs: {path}")
    try:
        value = model.model_validate_json(payload, strict=True)
    except ValueError as error:
        raise CategoryProjectionError(f"{context} failed schema validation: {path}") from error
    return value, payload


def _load_pinned_jsonl(
    configured: PinnedJsonl, *, context: str
) -> tuple[tuple[dict[str, Any], ...], bytes]:
    path = _canonical_file(Path(configured.path), context=context)
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != configured.sha256:
        raise CategoryProjectionError(f"{context} SHA-256 differs: {path}")
    return _strict_jsonl(payload, context=context), payload


def _manifest_file(
    root: Path, manifest: dict[str, Any], *, kind: str
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise CategoryProjectionError("source manifest files must be a list")
    entries = [row for row in files if isinstance(row, dict) and row.get("kind") == kind]
    if len(entries) != 1:
        raise CategoryProjectionError(f"source manifest must contain one {kind} file")
    entry = cast(dict[str, Any], entries[0])
    relative = entry.get("path")
    digest = entry.get("sha256")
    byte_count = entry.get("bytes")
    row_count = entry.get("rows", entry.get("records"))
    if (
        not isinstance(relative, str)
        or not isinstance(digest, str)
        or not isinstance(byte_count, int)
        or not isinstance(row_count, int)
    ):
        raise CategoryProjectionError(f"source {kind} manifest entry is malformed")
    path = (root / relative).resolve(strict=True)
    if root not in path.parents or not path.is_file() or path.is_symlink():
        raise CategoryProjectionError(f"source {kind} path is invalid: {path}")
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != digest or len(payload) != byte_count:
        raise CategoryProjectionError(f"source {kind} bytes/hash differ: {path}")
    rows = _strict_jsonl(payload, context=f"source {kind}")
    if len(rows) != row_count:
        raise CategoryProjectionError(f"source {kind} row count differs")
    return rows, entry


def _load_source(
    source: PinnedDataset,
) -> tuple[bytes, tuple[dict[str, Any], ...], dict[str, dict[str, Any]]]:
    root = Path(source.root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise CategoryProjectionError(f"source dataset root is invalid: {root}")
    manifest_path = _canonical_file(root / "manifest.json", context="source manifest")
    manifest_payload = read_regular_file_bytes(manifest_path)
    if sha256_bytes(manifest_payload) != source.manifest_sha256:
        raise CategoryProjectionError("source manifest SHA-256 differs")
    manifest = _strict_json(manifest_payload, context="source manifest")
    if manifest.get("datasetId") != source.dataset_id:
        raise CategoryProjectionError("source dataset ID differs")
    records, _ = _manifest_file(root, manifest, kind="training_records")
    lineage_rows, _ = _manifest_file(root, manifest, kind="lineage")
    if len(records) != source.expected_records:
        raise CategoryProjectionError("source training-record count differs from config")
    lineage: dict[str, dict[str, Any]] = {}
    for row in lineage_rows:
        document_id = row.get("documentId")
        if not isinstance(document_id, str) or _DOCUMENT_ID.fullmatch(document_id) is None:
            raise CategoryProjectionError("source lineage contains an invalid document ID")
        if document_id in lineage:
            raise CategoryProjectionError(f"duplicate source lineage: {document_id}")
        lineage[document_id] = row
    return manifest_payload, records, lineage


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
    except (KeyError, TypeError, ValueError) as error:
        raise CategoryProjectionError(
            f"source label validation failed for {document_id}: {error}"
        ) from error
    normal_target = normal.canonical_target()
    relation_target = relation.canonical_target()
    if normal_target != record["normalTarget"] or relation_target != record["target"]:
        raise CategoryProjectionError(f"source target is noncanonical: {document_id}")
    return normal_target, relation_target


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _publish(path: Path, payload: bytes) -> None:
    if path.exists() and read_regular_file_bytes(path) != payload:
        raise CategoryProjectionError(
            f"immutable publication already exists with different bytes: {path}"
        )
    atomic_publish_bytes(path, payload)


def _file_entry(
    path: Path, root: Path, payload: bytes, rows: int, kind: str
) -> dict[str, Any]:
    return {
        "bytes": len(payload),
        "kind": kind,
        "path": path.relative_to(root).as_posix(),
        "rows": rows,
        "sha256": sha256_bytes(payload),
    }


def _category_task_constraints(
    *, package_registry_sha256: str, container_registry_sha256: str, tokens: set[str]
) -> RelationExplicitTaskConstraints:
    task = get_training_task("bill_of_lading_relation_explicit_v3")
    target_schema_sha256 = sha256_bytes(
        canonical_json_bytes(task.target_model.model_json_schema(mode="serialization"))
    )
    return RelationExplicitTaskConstraints.model_validate(
        {
            "schemaVersion": 1,
            "task": task.name,
            "basePromptSchemaSha256": task.base_prompt_schema_sha256(),
            "targetSchemaSha256": target_schema_sha256,
            "packageRegistrySha256": package_registry_sha256,
            "containerRegistrySha256": container_registry_sha256,
            "packageCategoryTokens": tuple(sorted(tokens)),
            "containerCategoryTokens": (),
        },
        strict=True,
    )


def _report(summary: dict[str, Any], dataset_id: str) -> bytes:
    tokens = ", ".join(f"`{value}`" for value in summary["packageCategoryTokens"])
    typed = f'{summary["typedPackageOccurrences"]:,} / {summary["typedPackageVariants"]:,}'
    categorized = f'{summary["categoryOccurrences"]:,} / {summary["categoryVariants"]:,}'
    fallback = f'{summary["fallbackOccurrences"]:,} / {summary["fallbackVariants"]:,}'
    reused = f'{summary["reusedPriorOccurrences"]:,} / {summary["reusedPriorVariants"]:,}'
    extended = f'{summary["extensionOccurrences"]:,} / {summary["extensionVariants"]:,}'
    containers = (
        f'{summary["containerTypeOccurrences"]:,} / {summary["containerTypeVariants"]:,}'
    )
    text = f"""# MPCI B/L task-facing package categories

Dataset: `{dataset_id}`

This immutable dataset is a non-destructive projection over the task-facing package dataset.
Every exact package description is covered by a pinned decision. A readable, authoritative
registry token replaces the raw description only when that source string selects one category;
otherwise the original `typeDescription` remains the model target. Containers stay as printed
text and never receive inferred categories.

## Audit

| Measure | Count |
|---|---:|
| Source/projected records | {summary["records"]:,} |
| Package facts | {summary["packageFacts"]:,} |
| Typed package occurrences / variants | {typed} |
| Category occurrences / variants | {categorized} |
| Verbatim fallback occurrences / variants | {fallback} |
| Reused prior-review occurrences / variants | {reused} |
| Newly reviewed occurrences / variants | {extended} |
| Untyped quantity-only package facts | {summary["untypedPackageFacts"]:,} |
| Container type occurrences / variants (all printed) | {containers} |
| Schema and dual-view validations | {summary["validatedRecords"]:,} |

## Observed package category vocabulary

{tokens}

## Truth and lineage boundary

`category-metadata.jsonl` retains both complete source targets and an exact package-by-package
decision trace. `lineage.jsonl` pins the source dataset/record and that metadata row. The category
inventory records affected documents, occurrence counts, registry identity, review provenance,
and the exact unresolved rationale. Inventory drift, registry drift, or an unreviewed source
variant blocks publication.

`records.jsonl` is a single unsplit source suitable for the training runtime's seeded partition.
`task-constraints.json` contains exactly the category tokens observed in these projected records;
the container vocabulary is empty by policy.
"""
    return text.encode("utf-8")


def project_package_categories(config: CategoryProjectionConfig) -> Path:
    source_manifest_payload, source_records, source_lineage = _load_source(config.source)
    package_registry, _ = _load_pinned_model(
        config.registries.package, CategoryRegistry, context="package registry"
    )
    container_registry, _ = _load_pinned_model(
        config.registries.container, CategoryRegistry, context="container registry"
    )
    prior, _ = _load_pinned_model(
        config.registries.prior_assignments,
        CategoryAssignments,
        context="prior category assignments",
    )
    if package_registry.registryKind != "package" or container_registry.registryKind != "container":
        raise CategoryProjectionError("category registries have the wrong registryKind")
    if container_registry.entries:
        raise CategoryProjectionError(
            "container no-inference policy requires the pinned empty semantic registry"
        )
    if (
        prior.packageRegistrySha256 != config.registries.package.sha256
        or prior.containerRegistrySha256 != config.registries.container.sha256
    ):
        raise CategoryProjectionError("prior assignments target different registries")
    registry_by_token = {row.categoryToken: row for row in package_registry.entries}
    if len(registry_by_token) != len(package_registry.entries):
        raise CategoryProjectionError("package registry tokens are not unique")

    prior_map: dict[str, tuple[str | None, str]] = {}
    for row in prior.packageAssignments:
        if row.sourceTypeCode is not None or row.sourceTypeText is None:
            continue
        prior_map[row.sourceTypeText] = (row.categoryToken, row.reviewBasis)

    projection_prior_map: dict[str, tuple[str | None, str, str]] = {}
    prior_projection = config.registries.prior_projection_inventory
    if prior_projection is not None:
        prior_rows, _ = _load_pinned_jsonl(
            prior_projection, context="prior projection inventory"
        )
        for row_number, inventory_row in enumerate(prior_rows, start=1):
            description = inventory_row.get("sourceTypeDescription")
            token = inventory_row.get("categoryToken")
            basis = inventory_row.get("reviewBasis")
            rationale = inventory_row.get("rationale")
            if (
                not isinstance(description, str)
                or not description
                or (token is not None and not isinstance(token, str))
                or not isinstance(basis, str)
                or not basis
                or not isinstance(rationale, str)
                or not rationale
            ):
                raise CategoryProjectionError(
                    f"prior projection inventory row is malformed: {row_number}"
                )
            if description in projection_prior_map:
                raise CategoryProjectionError(
                    f"prior projection inventory repeats a source key: {description!r}"
                )
            if token is not None and token not in registry_by_token:
                raise CategoryProjectionError(
                    f"prior projection category is absent from registry: {token}"
                )
            if description in prior_map and prior_map[description][0] != token:
                raise CategoryProjectionError(
                    f"prior projection conflicts with prior assignments: {description!r}"
                )
            projection_prior_map[description] = (token, basis, rationale)

    extension_map: dict[str, tuple[str, str]] = {}
    for group in config.review.extension_resolution_groups:
        if group.category_token not in registry_by_token:
            raise CategoryProjectionError(
                f"extension category is absent from registry: {group.category_token}"
            )
        for description in group.source_type_descriptions:
            if description in prior_map or description in projection_prior_map:
                raise CategoryProjectionError(
                    f"extension redundantly re-reviews prior source key: {description!r}"
                )
            extension_map[description] = (group.category_token, group.rationale)
    unresolved_map = {
        row.source_type_description: row.rationale
        for row in config.review.unresolved_descriptions
    }
    if set(extension_map) & set(unresolved_map):
        raise CategoryProjectionError("extension review contains conflicting decisions")
    if (set(prior_map) | set(projection_prior_map)) & set(unresolved_map):
        raise CategoryProjectionError("prior-reviewed key is redundantly marked unresolved")

    package_counts: Counter[str] = Counter()
    package_documents: dict[str, set[str]] = defaultdict(set)
    container_counts: Counter[str] = Counter()
    container_documents: dict[str, set[str]] = defaultdict(set)
    untyped_package_facts = 0
    for record in source_records:
        document_id = record.get("documentId")
        if not isinstance(document_id, str) or _DOCUMENT_ID.fullmatch(document_id) is None:
            raise CategoryProjectionError("source record has an invalid document ID")
        patch = cast(dict[str, Any], cast(dict[str, Any], record["target"])["documentPatch"])
        for package in patch.get("cargoPackages", []):
            if package.get("typeCategory") is not None:
                raise CategoryProjectionError(
                    f"source unexpectedly contains a package category: {document_id}"
                )
            description = package.get("typeDescription")
            if description is None:
                untyped_package_facts += 1
            elif isinstance(description, str):
                package_counts[description] += 1
                package_documents[description].add(document_id)
            else:
                raise CategoryProjectionError(f"non-text package description: {document_id}")
        for container in patch.get("containers", []):
            if container.get("typeCategory") is not None:
                raise CategoryProjectionError(
                    f"source unexpectedly contains a container category: {document_id}"
                )
            description = container.get("typeDescription")
            if description is not None:
                if not isinstance(description, str):
                    raise CategoryProjectionError(
                        f"non-text container description: {document_id}"
                    )
                container_counts[description] += 1
                container_documents[description].add(document_id)

    review = config.review
    if (
        sum(package_counts.values()) != review.expected_package_occurrences
        or len(package_counts) != review.expected_package_variants
        or sum(container_counts.values()) != review.expected_container_occurrences
        or len(container_counts) != review.expected_container_variants
    ):
        raise CategoryProjectionError("observed category inventory differs from config")
    decided_keys = (
        set(prior_map)
        | set(projection_prior_map)
        | set(extension_map)
        | set(unresolved_map)
    )
    observed_keys = set(package_counts)
    missing = sorted(observed_keys - decided_keys)
    extra = sorted((set(extension_map) | set(unresolved_map)) - observed_keys)
    if missing or extra:
        raise CategoryProjectionError(
            f"package review does not exactly cover the observed extension: "
            f"missing={missing!r}, extra={extra!r}"
        )

    resolved_map: dict[str, str | None] = {}
    decision_rows: list[dict[str, Any]] = []
    for description in sorted(package_counts, key=lambda value: (-package_counts[value], value)):
        source_key_sha256 = sha256_bytes(
            canonical_json_bytes({"sourceTypeDescription": description})
        )
        if description in projection_prior_map:
            token, basis, rationale = projection_prior_map[description]
            provenance = "reused_prior_projection"
        elif description in prior_map:
            token, basis = prior_map[description]
            provenance = "reused_prior_review"
            rationale = (
                "The exact source string and decision are reused from the pinned prior "
                "category-assignment artifact."
            )
        elif description in extension_map:
            token, rationale = extension_map[description]
            basis = "manual_semantic_review"
            provenance = "current_exact_source_review"
        else:
            token = None
            basis = "insufficient_source_specificity"
            provenance = "current_exact_source_review"
            rationale = unresolved_map[description]
        resolved_map[description] = token
        registry_row = registry_by_token.get(token) if token is not None else None
        decision_rows.append(
            {
                "affectedDocuments": sorted(package_documents[description]),
                "applicationCode": (
                    registry_row.applicationCode if registry_row is not None else None
                ),
                "categoryToken": token,
                "displayName": registry_row.displayName if registry_row is not None else None,
                "decisionProvenance": provenance,
                "occurrences": package_counts[description],
                "rationale": rationale,
                "reviewBasis": basis,
                "reviewedBy": review.reviewed_by,
                "reviewedOn": review.reviewed_on.isoformat(),
                "sourceKeySha256": source_key_sha256,
                "sourceTypeDescription": description,
            }
        )

    container_rows = [
        {
            "affectedDocuments": sorted(container_documents[description]),
            "categoryToken": None,
            "decisionProvenance": "frozen_no_semantic_container_registry_policy",
            "occurrences": container_counts[description],
            "rationale": (
                "The authoritative pinned container semantic registry is empty; preserve the "
                "printed description and do not infer a category."
            ),
            "sourceKeySha256": sha256_bytes(
                canonical_json_bytes({"sourceTypeDescription": description})
            ),
            "sourceTypeDescription": description,
        }
        for description in sorted(
            container_counts, key=lambda value: (-container_counts[value], value)
        )
    ]

    projected_records: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    raw_hashes: set[str] = set()
    category_occurrences = 0
    fallback_occurrences = 0
    changed_documents = 0
    observed_tokens: set[str] = set()
    for source_row_number, record in enumerate(source_records, start=1):
        document_id = cast(str, record.get("documentId"))
        raw_text = record.get("joinedRawText")
        raw_hash = record.get("joinedRawTextSha256")
        if (
            _DOCUMENT_ID.fullmatch(document_id) is None
            or not isinstance(raw_text, str)
            or not isinstance(raw_hash, str)
            or sha256_bytes(raw_text.encode("utf-8")) != raw_hash
        ):
            raise CategoryProjectionError(
                f"source identity/raw OCR is invalid at row {source_row_number}"
            )
        if document_id in document_ids or raw_hash in raw_hashes:
            raise CategoryProjectionError(f"duplicate source document/raw OCR: {document_id}")
        if document_id not in source_lineage:
            raise CategoryProjectionError(f"source lineage is absent: {document_id}")
        document_ids.add(document_id)
        raw_hashes.add(raw_hash)
        normal_target, source_target = _canonical_labels(record, document_id=document_id)
        projected_target = json.loads(json.dumps(source_target))
        package_decisions: list[dict[str, Any]] = []
        changed = False
        for package in projected_target["documentPatch"].get("cargoPackages", []):
            description = package.get("typeDescription")
            if description is None:
                continue
            token = resolved_map[description]
            if token is None:
                fallback_occurrences += 1
            else:
                package.pop("typeDescription")
                package["typeCategory"] = token
                observed_tokens.add(token)
                category_occurrences += 1
                changed = True
            package_decisions.append(
                {
                    "categoryToken": token,
                    "groupId": package["groupId"],
                    "packageId": package["packageId"],
                    "sourceTypeDescription": description,
                    "sourceTypeDescriptionSha256": sha256_bytes(
                        description.encode("utf-8")
                    ),
                }
            )
        try:
            projected_relation = BillOfLadingRelationExplicitLabel.model_validate_json(
                canonical_json_bytes(projected_target), strict=True
            )
            normal = BillOfLadingLabel.model_validate_json(
                canonical_json_bytes(normal_target), strict=True
            )
            validate_dual_cargo_consistency(normal, projected_relation)
        except ValueError as error:
            raise CategoryProjectionError(
                f"projected target validation failed for {document_id}: {error}"
            ) from error
        canonical_projected = projected_relation.canonical_target()
        if normal.canonical_target() != normal_target:
            raise CategoryProjectionError(f"normal target changed: {document_id}")
        projected_record = {
            "documentId": document_id,
            "joinedRawText": raw_text,
            "joinedRawTextSha256": raw_hash,
            "normalTarget": normal_target,
            "target": canonical_projected,
        }
        metadata = {
            "documentId": document_id,
            "packageDecisions": package_decisions,
            "projectedRelationTargetSha256": sha256_bytes(
                canonical_json_bytes(canonical_projected)
            ),
            "sourceNormalTarget": normal_target,
            "sourceNormalTargetSha256": sha256_bytes(canonical_json_bytes(normal_target)),
            "sourceRecordCanonicalSha256": sha256_bytes(canonical_json_bytes(record)),
            "sourceRecordRow": source_row_number,
            "sourceRelationTarget": source_target,
            "sourceRelationTargetSha256": sha256_bytes(canonical_json_bytes(source_target)),
        }
        metadata_hash = sha256_bytes(canonical_json_bytes(metadata))
        projected_hash = sha256_bytes(canonical_json_bytes(projected_record))
        projected_records.append(projected_record)
        metadata_rows.append(metadata)
        lineage_rows.append(
            {
                "categoryMetadataCanonicalSha256": metadata_hash,
                "documentId": document_id,
                "projectedRecordCanonicalSha256": projected_hash,
                "sourceDatasetId": config.source.dataset_id,
                "sourceDatasetManifestSha256": config.source.manifest_sha256,
                "sourceLineage": source_lineage[document_id],
                "sourceRecordCanonicalSha256": metadata["sourceRecordCanonicalSha256"],
            }
        )
        changed_documents += int(changed)

    resolved_variants = {description for description, token in resolved_map.items() if token}
    fallback_variants = set(resolved_map) - resolved_variants
    reused_variants = set(package_counts) & (set(prior_map) | set(projection_prior_map))
    extension_variants = set(package_counts) - reused_variants
    constraints = _category_task_constraints(
        package_registry_sha256=config.registries.package.sha256,
        container_registry_sha256=config.registries.container.sha256,
        tokens=observed_tokens,
    )
    constraints_payload = canonical_json_bytes(constraints.model_dump(mode="json")) + b"\n"
    summary = {
        "categoryOccurrences": category_occurrences,
        "categoryVariants": len(resolved_variants),
        "changedDocuments": changed_documents,
        "containerTypeOccurrences": sum(container_counts.values()),
        "containerTypeVariants": len(container_counts),
        "extensionOccurrences": sum(package_counts[value] for value in extension_variants),
        "extensionVariants": len(extension_variants),
        "fallbackOccurrences": fallback_occurrences,
        "fallbackVariants": len(fallback_variants),
        "packageCategoryTokens": sorted(observed_tokens),
        "packageFacts": sum(package_counts.values()) + untyped_package_facts,
        "policy": config.projection,
        "records": len(projected_records),
        "reusedPriorOccurrences": sum(package_counts[value] for value in reused_variants),
        "reusedPriorVariants": len(reused_variants),
        "schemaVersion": 1,
        "typedPackageOccurrences": sum(package_counts.values()),
        "typedPackageVariants": len(package_counts),
        "untypedPackageFacts": untyped_package_facts,
        "validatedRecords": len(projected_records),
    }

    output_root = Path(config.output.root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root.resolve(strict=True)
    if output_root.is_symlink() or output_root == Path(output_root.anchor):
        raise CategoryProjectionError("output root is unsafe")
    payloads = {
        "records.jsonl": (
            _jsonl_bytes(projected_records),
            len(projected_records),
            "training_records",
        ),
        "lineage.jsonl": (_jsonl_bytes(lineage_rows), len(lineage_rows), "lineage"),
        "category-metadata.jsonl": (
            _jsonl_bytes(metadata_rows),
            len(metadata_rows),
            "category_metadata",
        ),
        "package-category-inventory.jsonl": (
            _jsonl_bytes(decision_rows),
            len(decision_rows),
            "package_category_inventory",
        ),
        "container-category-inventory.jsonl": (
            _jsonl_bytes(container_rows),
            len(container_rows),
            "container_category_inventory",
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
        "audit": summary,
        "categoryPolicy": {
            "contract": config.projection,
            "containerCategoriesInferred": False,
            "exactReviewedSourceKeysOnly": True,
            "rawFallbackForAmbiguousDescriptions": True,
            "sourceTargetsRetainedInMetadata": True,
        },
        "datasetId": config.output.dataset_id,
        "files": files,
        "records": len(projected_records),
        "registries": {
            "containerRegistrySha256": config.registries.container.sha256,
            "packageRegistrySha256": config.registries.package.sha256,
            "priorAssignmentsSha256": config.registries.prior_assignments.sha256,
            "priorProjectionInventorySha256": (
                config.registries.prior_projection_inventory.sha256
                if config.registries.prior_projection_inventory is not None
                else None
            ),
        },
        "schemaVersion": 1,
        "sourceDataset": {
            "datasetId": config.source.dataset_id,
            "manifestSha256": sha256_bytes(source_manifest_payload),
            "records": len(source_records),
        },
        "task": "bill_of_lading_relation_explicit_v3",
        "trainingReady": True,
        "training_task_constraints": {
            "path": "task-constraints.json",
            "sha256": sha256_bytes(constraints_payload),
        },
    }
    manifest_path = output_root / "manifest.json"
    _publish(manifest_path, canonical_json_bytes(manifest) + b"\n")
    return manifest_path


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CategoryProjectionError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _ArgumentParser(
        prog="python -m document_ocr.training.category_projection",
        description="Publish exact registry-backed task-facing B/L package categories.",
    )
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    config = load_category_projection_config(arguments.config)
    manifest = project_package_categories(config)
    print(
        json.dumps(
            {"command": "category-project", "manifest": str(manifest), "status": "complete"},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
