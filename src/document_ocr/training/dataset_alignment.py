"""Align the legacy B/L corpus to the current training contract and merge it.

The legacy semantic-v3 publication intentionally learned registry categories.
The current single-source contract instead retains package/container wording as
printed and leaves registry mapping downstream.  This module republishes the
legacy targets without mutating either source dataset, applies only explicitly
reviewed OCR-grounded corrections, and then joins disjoint validated records.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, cast

from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from document_ocr.atomic import atomic_publish_bytes, read_regular_file_bytes
from document_ocr.config import load_strict_yaml_mapping
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading import BillOfLadingAnnotation, BillOfLadingLabel
from document_ocr.label_schemas.bill_of_lading_v3 import (
    BillOfLadingDualCargoAnnotation,
    BillOfLadingRelationExplicitLabel,
    validate_dual_cargo_consistency,
)
from document_ocr.label_schemas.common import LabelSchemaModel
from document_ocr.labeling_agents.work_items import (
    AgentWorkItem,
    WorkItemError,
    page_texts,
    validate_annotation_evidence,
    validate_raw_ocr_evidence,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
DocumentId = Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]
NonNegativeInteger = Annotated[int, Field(ge=0)]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DatasetAlignmentError(RuntimeError):
    """A pinned input or aligned publication violated its contract."""


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

    @field_validator("dataset_id")
    @classmethod
    def dataset_id_is_safe(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError("dataset_id contains unsupported characters")
        return value

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_path(value, "dataset root", directory=True)


class PinnedAnnotationRoot(_ConfigModel):
    root_id: NonEmptyString
    root: NonEmptyString
    manifest_sha256: Sha256

    @field_validator("root_id")
    @classmethod
    def root_id_is_safe(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError("annotation root ID contains unsupported characters")
        return value

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_path(value, "annotation root", directory=True)


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


class LegacyAlignmentSource(_ConfigModel):
    normal_dataset: PinnedDataset
    relation_dataset: PinnedDataset
    annotation_roots: tuple[PinnedAnnotationRoot, ...] = Field(min_length=1)
    prior_corrections: PinnedJson
    policy_corrections: PinnedJson
    expected_source_records: PositiveInteger
    expected_aligned_records: PositiveInteger
    expected_relation_exclusions: PositiveInteger

    @field_validator("annotation_roots", mode="before")
    @classmethod
    def annotation_roots_are_frozen(cls, value: Any) -> Any:
        if isinstance(value, tuple):
            return value
        if not isinstance(value, list):
            raise ValueError("annotation_roots must be a YAML sequence")
        return tuple(value)

    @model_validator(mode="after")
    def sources_are_distinct(self) -> LegacyAlignmentSource:
        roots = (
            self.normal_dataset.root,
            self.relation_dataset.root,
            *(row.root for row in self.annotation_roots),
        )
        if len(roots) != len(set(roots)):
            raise ValueError("legacy dataset and annotation roots must be distinct")
        root_ids = tuple(row.root_id for row in self.annotation_roots)
        if len(root_ids) != len(set(root_ids)):
            raise ValueError("annotation root IDs must be unique")
        return self


class CurrentDatasetSource(_ConfigModel):
    dataset: PinnedDataset
    expected_training_records: PositiveInteger


class CrossSourceDuplicateResolution(_ConfigModel):
    joined_raw_text_sha256: Sha256
    legacy_document_id: DocumentId
    current_document_id: DocumentId
    retain: Literal["legacy", "current"]
    rationale: NonEmptyString


class AlignmentOutput(_ConfigModel):
    aligned_legacy_root: NonEmptyString
    combined_root: NonEmptyString
    aligned_legacy_dataset_id: NonEmptyString
    combined_dataset_id: NonEmptyString

    @field_validator("aligned_legacy_root", "combined_root")
    @classmethod
    def roots_are_absolute(cls, value: str) -> str:
        return _absolute_path(value, "alignment output root", directory=True)

    @field_validator("aligned_legacy_dataset_id", "combined_dataset_id")
    @classmethod
    def ids_are_safe(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError("alignment dataset ID contains unsupported characters")
        return value

    @model_validator(mode="after")
    def outputs_are_distinct(self) -> AlignmentOutput:
        if self.aligned_legacy_root == self.combined_root:
            raise ValueError("aligned and combined output roots must differ")
        if self.aligned_legacy_dataset_id == self.combined_dataset_id:
            raise ValueError("aligned and combined dataset IDs must differ")
        return self


class DatasetAlignmentConfig(_ConfigModel):
    schema_version: Literal[1]
    alignment: Literal["mpci_bl_current_single_source_contract_v1"]
    legacy: LegacyAlignmentSource
    current: CurrentDatasetSource
    cross_source_duplicate_resolutions: tuple[CrossSourceDuplicateResolution, ...] = ()
    output: AlignmentOutput

    @field_validator("cross_source_duplicate_resolutions", mode="before")
    @classmethod
    def duplicate_resolutions_are_frozen(cls, value: Any) -> Any:
        if isinstance(value, tuple):
            return value
        if not isinstance(value, list):
            raise ValueError("cross_source_duplicate_resolutions must be a YAML sequence")
        return tuple(value)

    @model_validator(mode="after")
    def identifiers_and_roots_are_unique(self) -> DatasetAlignmentConfig:
        ids = (
            self.legacy.normal_dataset.dataset_id,
            self.legacy.relation_dataset.dataset_id,
            self.current.dataset.dataset_id,
            self.output.aligned_legacy_dataset_id,
            self.output.combined_dataset_id,
        )
        if len(ids) != len(set(ids)):
            raise ValueError("all source and output dataset IDs must be unique")
        source_roots = {
            self.legacy.normal_dataset.root,
            self.legacy.relation_dataset.root,
            self.current.dataset.root,
            *(row.root for row in self.legacy.annotation_roots),
        }
        if self.output.aligned_legacy_root in source_roots:
            raise ValueError("aligned legacy output aliases a source root")
        if self.output.combined_root in source_roots:
            raise ValueError("combined output aliases a source root")
        resolutions = self.cross_source_duplicate_resolutions
        hashes = tuple(row.joined_raw_text_sha256 for row in resolutions)
        legacy_ids = tuple(row.legacy_document_id for row in resolutions)
        current_ids = tuple(row.current_document_id for row in resolutions)
        if len(hashes) != len(set(hashes)):
            raise ValueError("duplicate-resolution raw OCR hashes must be unique")
        if len(legacy_ids) != len(set(legacy_ids)):
            raise ValueError("duplicate-resolution legacy document IDs must be unique")
        if len(current_ids) != len(set(current_ids)):
            raise ValueError("duplicate-resolution current document IDs must be unique")
        return self


class CorrectionEvidence(_ConfigModel):
    pageNumber: PositiveInteger
    rawValue: NonEmptyString


class PolicyCorrection(_ConfigModel):
    correctionId: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
    documentId: DocumentId
    kind: Literal["cargo_description", "package_type"]
    goodsIndex: NonNegativeInteger
    packageIndex: NonNegativeInteger | None
    before: NonEmptyString
    after: NonEmptyString
    rawOcrEvidence: tuple[CorrectionEvidence, ...] = Field(min_length=1)
    normalizationRule: NonEmptyString

    @field_validator("rawOcrEvidence", mode="before")
    @classmethod
    def raw_ocr_evidence_is_frozen(cls, value: Any) -> Any:
        if isinstance(value, tuple):
            return value
        if not isinstance(value, list):
            raise ValueError("rawOcrEvidence must be a JSON array")
        return tuple(value)

    @model_validator(mode="after")
    def index_shape_matches_kind(self) -> PolicyCorrection:
        if self.kind == "cargo_description" and self.packageIndex is not None:
            raise ValueError("cargo-description correction cannot carry packageIndex")
        if self.kind == "package_type" and self.packageIndex is None:
            raise ValueError("package-type correction requires packageIndex")
        if self.before == self.after:
            raise ValueError("correction before and after values must differ")
        return self


class PolicyCorrections(_ConfigModel):
    schemaVersion: Literal[1]
    policy: Literal["printed_package_types_and_product_only_descriptions_v1"]
    corrections: tuple[PolicyCorrection, ...] = Field(min_length=1)

    @field_validator("corrections", mode="before")
    @classmethod
    def corrections_are_frozen(cls, value: Any) -> Any:
        if isinstance(value, tuple):
            return value
        if not isinstance(value, list):
            raise ValueError("corrections must be a JSON array")
        return tuple(value)

    @model_validator(mode="after")
    def corrections_are_unique(self) -> PolicyCorrections:
        ids = tuple(row.correctionId for row in self.corrections)
        targets = tuple(
            (row.documentId, row.kind, row.goodsIndex, row.packageIndex) for row in self.corrections
        )
        if len(ids) != len(set(ids)) or len(targets) != len(set(targets)):
            raise ValueError("correction IDs and target coordinates must be unique")
        return self


@dataclass(frozen=True, slots=True)
class LoadedRow:
    value: dict[str, Any]
    source_path: Path
    source_sha256: str
    row_number: int
    split: str | None


def load_dataset_alignment_config(path: Path) -> DatasetAlignmentConfig:
    try:
        return DatasetAlignmentConfig.model_validate(load_strict_yaml_mapping(path), strict=True)
    except ValueError as error:
        raise DatasetAlignmentError(f"invalid dataset alignment config: {path}: {error}") from error


def _json(payload: bytes, *, context: str) -> dict[str, Any]:
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
        raise DatasetAlignmentError(f"{context} is not strict JSON") from error
    if not isinstance(value, dict):
        raise DatasetAlignmentError(f"{context} must be a JSON object")
    return value


def _jsonl(payload: bytes, *, context: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for row_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise DatasetAlignmentError(f"{context} contains a blank row at {row_number}")
        rows.append(_json(line, context=f"{context} row {row_number}"))
    return tuple(rows)


def _root(path: str, *, context: str) -> Path:
    try:
        root = Path(path).resolve(strict=True)
    except OSError as error:
        raise DatasetAlignmentError(f"{context} is absent: {path}") from error
    if not root.is_dir():
        raise DatasetAlignmentError(f"{context} is not a directory: {root}")
    return root


def _file(path: Path, *, context: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DatasetAlignmentError(f"{context} is absent: {path}") from error
    if not resolved.is_file():
        raise DatasetAlignmentError(f"{context} is not a regular file: {resolved}")
    return resolved


def _contained(root: Path, relative: str, *, context: str) -> Path:
    path = _file(root / relative, context=context)
    if root not in path.parents:
        raise DatasetAlignmentError(f"{context} escapes its dataset root: {relative}")
    return path


def _pinned_json(configured: PinnedJson, *, context: str) -> tuple[bytes, dict[str, Any]]:
    path = _file(Path(configured.path), context=context)
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != configured.sha256:
        raise DatasetAlignmentError(f"{context} SHA-256 differs: {path}")
    return payload, _json(payload, context=context)


def _dataset_manifest(configured: PinnedDataset) -> tuple[Path, bytes, dict[str, Any]]:
    root = _root(configured.root, context=f"dataset {configured.dataset_id}")
    path = _contained(root, "manifest.json", context="dataset manifest")
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != configured.manifest_sha256:
        raise DatasetAlignmentError(f"dataset manifest SHA-256 differs: {path}")
    manifest = _json(payload, context=f"dataset manifest {path}")
    observed_id = manifest.get("dataset_id", manifest.get("datasetId"))
    if observed_id != configured.dataset_id:
        raise DatasetAlignmentError(f"dataset manifest ID differs: {path}")
    return root, payload, manifest


def _manifest_entry(manifest: dict[str, Any], *, kind: str) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise DatasetAlignmentError("dataset manifest files must be a list")
    matches = [row for row in files if isinstance(row, dict) and row.get("kind") == kind]
    if len(matches) != 1:
        raise DatasetAlignmentError(f"dataset manifest must contain one {kind} entry")
    return cast(dict[str, Any], matches[0])


def _entry_count(entry: dict[str, Any], *, context: str) -> int:
    value = entry.get("records", entry.get("rows"))
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetAlignmentError(f"{context} row count is invalid")
    return value


def _manifest_rows(
    root: Path,
    manifest: dict[str, Any],
    *,
    kinds: Sequence[tuple[str, str | None]],
) -> tuple[LoadedRow, ...]:
    loaded: list[LoadedRow] = []
    for kind, split in kinds:
        entry = _manifest_entry(manifest, kind=kind)
        relative = entry.get("path")
        expected_sha = entry.get("sha256")
        expected_bytes = entry.get("bytes")
        if not isinstance(relative, str) or not isinstance(expected_sha, str):
            raise DatasetAlignmentError(f"manifest {kind} path/hash is invalid")
        path = _contained(root, relative, context=f"dataset {kind}")
        payload = read_regular_file_bytes(path)
        if sha256_bytes(payload) != expected_sha or len(payload) != expected_bytes:
            raise DatasetAlignmentError(f"dataset {kind} bytes/hash differ: {path}")
        rows = _jsonl(payload, context=f"dataset {kind}")
        if len(rows) != _entry_count(entry, context=kind):
            raise DatasetAlignmentError(f"dataset {kind} row count differs: {path}")
        loaded.extend(
            LoadedRow(
                value=row,
                source_path=path,
                source_sha256=expected_sha,
                row_number=row_number,
                split=split,
            )
            for row_number, row in enumerate(rows, start=1)
        )
    return tuple(loaded)


def _document_id(row: LoadedRow, *, context: str) -> str:
    value = row.value.get("documentId")
    if not isinstance(value, str) or re.fullmatch(r"doc_[0-9a-f]{64}", value) is None:
        raise DatasetAlignmentError(f"{context} has an invalid documentId")
    return value


def _unique_rows(rows: Sequence[LoadedRow], *, context: str) -> dict[str, LoadedRow]:
    indexed: dict[str, LoadedRow] = {}
    joined_hashes: set[str] = set()
    for row in rows:
        document_id = _document_id(row, context=context)
        joined = row.value.get("joinedRawText")
        joined_hash = row.value.get("joinedRawTextSha256")
        if not isinstance(joined, str) or not isinstance(joined_hash, str):
            raise DatasetAlignmentError(f"{context} raw OCR fields are absent: {document_id}")
        if sha256_bytes(joined.encode("utf-8")) != joined_hash:
            raise DatasetAlignmentError(f"{context} raw OCR SHA-256 differs: {document_id}")
        if document_id in indexed or joined_hash in joined_hashes:
            raise DatasetAlignmentError(
                f"{context} contains duplicate ID or raw OCR: {document_id}"
            )
        indexed[document_id] = row
        joined_hashes.add(joined_hash)
    return indexed


def _pointer_tokens(pointer: str) -> tuple[str, ...]:
    if not pointer.startswith("/"):
        raise DatasetAlignmentError(f"JSON pointer must be absolute: {pointer}")
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/"))


def _pointer_value(value: Any, pointer: str) -> Any:
    current = value
    for token in _pointer_tokens(pointer):
        if isinstance(current, list):
            current = current[int(token)]
        elif isinstance(current, dict):
            current = current[token]
        else:
            raise DatasetAlignmentError(f"JSON pointer traverses a scalar: {pointer}")
    return current


def _pointer_set(value: Any, pointer: str, replacement: Any) -> None:
    tokens = _pointer_tokens(pointer)
    current = value
    for token in tokens[:-1]:
        current = current[int(token)] if isinstance(current, list) else current[token]
    last = tokens[-1]
    if isinstance(current, list):
        current[int(last)] = replacement
    else:
        current[last] = replacement


def _annotation_index(
    configured_roots: Sequence[PinnedAnnotationRoot],
) -> tuple[dict[tuple[str, str], tuple[str, Path]], tuple[dict[str, Any], ...]]:
    index: dict[tuple[str, str], tuple[str, Path]] = {}
    source_rows: list[dict[str, Any]] = []
    for configured in configured_roots:
        root = _root(configured.root, context=f"annotation root {configured.root_id}")
        manifest_path = _contained(root, "manifest.json", context="annotation manifest")
        manifest_payload = read_regular_file_bytes(manifest_path)
        if sha256_bytes(manifest_payload) != configured.manifest_sha256:
            raise DatasetAlignmentError(f"annotation manifest SHA-256 differs: {manifest_path}")
        manifest = _json(manifest_payload, context=f"annotation manifest {manifest_path}")
        files = manifest.get("files")
        if not isinstance(files, list):
            raise DatasetAlignmentError(f"annotation manifest files are invalid: {manifest_path}")
        for entry in files:
            if not isinstance(entry, dict):
                continue
            relative = entry.get("path")
            digest = entry.get("sha256")
            if not isinstance(relative, str) or not relative.startswith("validated/"):
                continue
            if not isinstance(digest, str):
                raise DatasetAlignmentError("validated annotation manifest hash is absent")
            key = (relative, digest)
            if key in index:
                raise DatasetAlignmentError(f"duplicate validated annotation artifact: {key}")
            path = _contained(root, relative, context="validated annotation")
            payload = read_regular_file_bytes(path)
            if sha256_bytes(payload) != digest:
                raise DatasetAlignmentError(f"validated annotation SHA-256 differs: {path}")
            index[key] = (configured.root_id, path)
        source_rows.append(
            {
                "rootId": configured.root_id,
                "root": str(root),
                "manifestSha256": configured.manifest_sha256,
            }
        )
    return index, tuple(source_rows)


def _validate_legacy_evidence_verbatim(
    annotation: BillOfLadingAnnotation, joined_raw_text: str
) -> tuple[str, ...]:
    pages = page_texts(joined_raw_text)
    strict_failures: list[str] = []
    for field in annotation.evidence:
        previous_page = 0
        for evidence in field.rawOcrEvidence:
            source = pages.get(evidence.pageNumber)
            if (
                source is None
                or evidence.pageNumber < previous_page
                or evidence.rawValue not in evidence.ocrExcerpt
                or evidence.ocrExcerpt not in source
            ):
                raise DatasetAlignmentError(
                    f"legacy evidence is not page-ordered verbatim OCR: {field.targetPath}"
                )
            previous_page = evidence.pageNumber
        try:
            validate_raw_ocr_evidence(pages, field.rawOcrEvidence)
        except WorkItemError:
            strict_failures.append(field.targetPath)
    return tuple(strict_failures)


def _verify_prior_corrections(
    *,
    annotation_target: dict[str, Any],
    source_target: dict[str, Any],
    joined_raw_text: str,
    corrections: Sequence[dict[str, Any]],
    document_id: str,
) -> None:
    corrected = json.loads(json.dumps(annotation_target))
    for row in corrections:
        pointer = row.get("target_path")
        before = row.get("before")
        after = row.get("after")
        evidence = row.get("raw_evidence")
        if not isinstance(pointer, str) or not isinstance(evidence, list):
            raise DatasetAlignmentError(f"prior correction is malformed: {document_id}")
        if _pointer_value(corrected, pointer) != before:
            raise DatasetAlignmentError(
                f"prior correction before-value differs: {document_id} {pointer}"
            )
        if any(not isinstance(value, str) or value not in joined_raw_text for value in evidence):
            raise DatasetAlignmentError(
                f"prior correction evidence is absent from raw OCR: {document_id} {pointer}"
            )
        _pointer_set(corrected, pointer, after)
    if corrected != source_target:
        raise DatasetAlignmentError(
            f"legacy annotation target differs beyond pinned prior corrections: {document_id}"
        )


def _normal_type_policy(target: dict[str, Any]) -> tuple[int, int]:
    patch = cast(dict[str, Any], target["documentPatch"])
    container_codes = 0
    package_codes = 0
    for container in patch.get("containers", []):
        if container.get("typeCode") is not None:
            container_codes += 1
            if container.get("typeDescription") is None:
                container["typeDescription"] = container["typeCode"]
            del container["typeCode"]
    for goods in patch.get("goodsItems", []):
        for package in goods.get("packages", []):
            if package.get("typeCode") is not None:
                package_codes += 1
                if package.get("type") is None:
                    package["type"] = package["typeCode"]
                del package["typeCode"]
    return container_codes, package_codes


def _apply_policy_corrections(
    normal_target: dict[str, Any],
    *,
    corrections: Sequence[PolicyCorrection],
    joined_raw_text: str,
) -> None:
    pages = page_texts(joined_raw_text)
    goods = cast(list[dict[str, Any]], normal_target["documentPatch"].get("goodsItems", []))
    for correction in corrections:
        if correction.goodsIndex >= len(goods):
            raise DatasetAlignmentError(
                f"correction goodsIndex is outside target: {correction.correctionId}"
            )
        target_goods = goods[correction.goodsIndex]
        if correction.kind == "cargo_description":
            current = target_goods.get("description")
            if current != correction.before:
                raise DatasetAlignmentError(
                    f"correction before-value differs: {correction.correctionId}"
                )
            target_goods["description"] = correction.after
        else:
            packages = cast(list[dict[str, Any]], target_goods.get("packages", []))
            package_index = cast(int, correction.packageIndex)
            if (
                package_index >= len(packages)
                or packages[package_index].get("type") != correction.before
            ):
                raise DatasetAlignmentError(
                    f"package correction before-value differs: {correction.correctionId}"
                )
            packages[package_index]["type"] = correction.after
        for evidence in correction.rawOcrEvidence:
            page = pages.get(evidence.pageNumber)
            if page is None or evidence.rawValue not in page:
                raise DatasetAlignmentError(
                    f"correction evidence is absent: {correction.correctionId}"
                )


def _align_relation_target(
    relation_target: dict[str, Any], normal_target: dict[str, Any]
) -> tuple[dict[str, Any], int, int]:
    aligned = json.loads(json.dumps(relation_target))
    relation_patch = cast(dict[str, Any], aligned["documentPatch"])
    normal_patch = cast(dict[str, Any], normal_target["documentPatch"])
    normal_containers = cast(list[dict[str, Any]], normal_patch.get("containers", []))
    relation_containers = cast(list[dict[str, Any]], relation_patch.get("containers", []))
    if len(normal_containers) != len(relation_containers):
        raise DatasetAlignmentError("legacy normal/relation container counts differ")
    container_categories = 0
    for normal, relation in zip(normal_containers, relation_containers, strict=True):
        if relation.pop("typeCategory", None) is not None:
            container_categories += 1
        printed = normal.get("typeDescription")
        if printed is None:
            relation.pop("typeDescription", None)
        else:
            relation["typeDescription"] = printed

    normal_goods = cast(list[dict[str, Any]], normal_patch.get("goodsItems", []))
    relation_groups = cast(list[dict[str, Any]], relation_patch.get("cargoGroups", []))
    if len(normal_goods) != len(relation_groups):
        raise DatasetAlignmentError("legacy normal/relation goods counts differ")
    for normal, relation in zip(normal_goods, relation_groups, strict=True):
        if normal.get("description") is None:
            relation.pop("description", None)
        else:
            relation["description"] = normal["description"]

    normal_packages = [
        package
        for goods in normal_goods
        for package in cast(list[dict[str, Any]], goods.get("packages", []))
    ]
    relation_packages = cast(list[dict[str, Any]], relation_patch.get("cargoPackages", []))
    if len(normal_packages) != len(relation_packages):
        raise DatasetAlignmentError("legacy normal/relation package counts differ")
    package_categories = 0
    for normal, relation in zip(normal_packages, relation_packages, strict=True):
        if relation.pop("typeCategory", None) is not None:
            package_categories += 1
        printed = normal.get("type")
        if printed is None:
            relation.pop("typeDescription", None)
        else:
            relation["typeDescription"] = printed
    return aligned, package_categories, container_categories


def _canonical_pair(
    normal_target: dict[str, Any], relation_target: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        normal = BillOfLadingLabel.model_validate_json(
            canonical_json_bytes(normal_target), strict=True
        )
        relation = BillOfLadingRelationExplicitLabel.model_validate_json(
            canonical_json_bytes(relation_target), strict=True
        )
        validate_dual_cargo_consistency(normal, relation)
    except ValueError as error:
        raise DatasetAlignmentError(f"aligned normal/relation targets differ: {error}") from error
    return normal.canonical_target(), relation.canonical_target()


def _minimal_record(
    source: dict[str, Any], normal_target: dict[str, Any], relation_target: dict[str, Any]
) -> dict[str, Any]:
    return {
        "documentId": source["documentId"],
        "joinedRawText": source["joinedRawText"],
        "joinedRawTextSha256": source["joinedRawTextSha256"],
        "normalTarget": normal_target,
        "target": relation_target,
    }


def _no_mapped_transport_categories(target: dict[str, Any], *, document_id: str) -> None:
    patch = cast(dict[str, Any], target["documentPatch"])
    if any("typeCategory" in row for row in patch.get("containers", [])):
        raise DatasetAlignmentError(f"mapped container category remains: {document_id}")
    if any("typeCategory" in row for row in patch.get("cargoPackages", [])):
        raise DatasetAlignmentError(f"mapped package category remains: {document_id}")


def _load_current_records(
    configured: CurrentDatasetSource,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    root, manifest_payload, manifest = _dataset_manifest(configured.dataset)
    if manifest.get("task") != "bill_of_lading_relation_single_source_v4":
        raise DatasetAlignmentError("current source task differs from single-source v4")
    loaded = _manifest_rows(root, manifest, kinds=(("training_records", None),))
    source_lineage = _manifest_rows(root, manifest, kinds=(("lineage", None),))
    if len(loaded) != configured.expected_training_records:
        raise DatasetAlignmentError("current training-record count differs from config")
    _unique_rows(loaded, context="current training records")
    lineage_artifacts: dict[str, tuple[str, str]] = {}
    for row in source_lineage:
        document_id = _document_id(row, context="current source lineage")
        relative = row.value.get("consolidatedArtifactPath")
        digest = row.value.get("consolidatedArtifactSha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise DatasetAlignmentError(
                f"current source lineage artifact is malformed: {document_id}"
            )
        lineage_artifacts[document_id] = (relative, digest)
    records: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for row in loaded:
        value = row.value
        document_id = _document_id(row, context="current training record")
        relative = value.get("validatedAnnotationPath")
        digest = value.get("validatedAnnotationSha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise DatasetAlignmentError(f"current annotation reference is absent: {document_id}")
        annotation_path = _contained(root, relative, context="current validated annotation")
        annotation_payload = read_regular_file_bytes(annotation_path)
        if sha256_bytes(annotation_payload) != digest:
            raise DatasetAlignmentError(f"current annotation SHA-256 differs: {document_id}")
        if lineage_artifacts.get(document_id) != (relative, digest):
            raise DatasetAlignmentError(
                f"current annotation differs from source lineage: {document_id}"
            )
        try:
            annotation = BillOfLadingDualCargoAnnotation.model_validate_json(
                annotation_payload, strict=True
            )
            work_item = AgentWorkItem(
                source=annotation.source,
                joinedRawText=cast(str, value["joinedRawText"]),
            )
            validate_annotation_evidence(work_item, annotation)
            normal, relation = _canonical_pair(
                cast(dict[str, Any], value["normalTarget"]),
                cast(dict[str, Any], value["target"]),
            )
        except (ValueError, WorkItemError) as error:
            raise DatasetAlignmentError(
                f"current validated record failed evidence/schema checks: {document_id}: {error}"
            ) from error
        if normal != annotation.normalLabel.canonical_target():
            raise DatasetAlignmentError(
                f"current normal target differs from annotation: {document_id}"
            )
        if relation != annotation.relationExplicitLabel.canonical_target():
            raise DatasetAlignmentError(
                f"current relation target differs from annotation: {document_id}"
            )
        _no_mapped_transport_categories(relation, document_id=document_id)
        records.append(_minimal_record(value, normal, relation))
        lineage.append(
            {
                "documentId": document_id,
                "sourceCorpus": "current_main680",
                "sourceDatasetId": configured.dataset.dataset_id,
                "sourceDatasetManifestSha256": sha256_bytes(manifest_payload),
                "sourceRecordPath": relative,
                "sourceValidatedAnnotationSha256": digest,
                "normalTargetCanonicalSha256": sha256_bytes(canonical_json_bytes(normal)),
                "targetCanonicalSha256": sha256_bytes(canonical_json_bytes(relation)),
            }
        )
    return tuple(records), tuple(lineage), manifest


def _load_legacy_records(
    configured: LegacyAlignmentSource,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    normal_root, normal_manifest_payload, normal_manifest = _dataset_manifest(
        configured.normal_dataset
    )
    relation_root, relation_manifest_payload, relation_manifest = _dataset_manifest(
        configured.relation_dataset
    )
    normal_rows = _manifest_rows(
        normal_root,
        normal_manifest,
        kinds=(("train_records", "train"), ("validation_records", "validation")),
    )
    relation_rows = _manifest_rows(
        relation_root,
        relation_manifest,
        kinds=(("train_records", "train"), ("validation_records", "validation")),
    )
    if len(normal_rows) != configured.expected_source_records:
        raise DatasetAlignmentError("legacy normal source count differs from config")
    if len(relation_rows) != configured.expected_aligned_records:
        raise DatasetAlignmentError("legacy relation source count differs from config")
    normal_by_id = _unique_rows(normal_rows, context="legacy normal records")
    relation_by_id = _unique_rows(relation_rows, context="legacy relation records")
    excluded_loaded = _manifest_rows(
        relation_root, relation_manifest, kinds=(("declared_exclusions", None),)
    )
    excluded_ids = {
        _document_id(row, context="legacy relation exclusion") for row in excluded_loaded
    }
    if len(excluded_ids) != configured.expected_relation_exclusions:
        raise DatasetAlignmentError("legacy relation exclusion count differs from config")
    if set(normal_by_id) - set(relation_by_id) != excluded_ids:
        raise DatasetAlignmentError("legacy relation exclusions do not equal source-minus-output")

    _, prior_manifest = _pinned_json(
        configured.prior_corrections, context="legacy prior corrections"
    )
    prior_rows = prior_manifest.get("corrections")
    if not isinstance(prior_rows, list):
        raise DatasetAlignmentError("legacy prior corrections are absent")
    prior_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prior_rows:
        if not isinstance(row, dict) or not isinstance(row.get("document_id"), str):
            raise DatasetAlignmentError("legacy prior correction row is malformed")
        prior_by_document[cast(str, row["document_id"])].append(row)

    _, policy_payload = _pinned_json(
        configured.policy_corrections, context="legacy current-policy corrections"
    )
    try:
        policy = PolicyCorrections.model_validate(policy_payload, strict=True)
    except ValueError as error:
        raise DatasetAlignmentError(f"legacy policy corrections are invalid: {error}") from error
    policy_by_document: dict[str, list[PolicyCorrection]] = defaultdict(list)
    for correction in policy.corrections:
        if correction.documentId not in relation_by_id:
            raise DatasetAlignmentError(
                f"policy correction targets an absent aligned record: {correction.correctionId}"
            )
        policy_by_document[correction.documentId].append(correction)

    annotation_index, annotation_sources = _annotation_index(configured.annotation_roots)
    annotation_details: dict[str, tuple[str, Path, BillOfLadingAnnotation, tuple[str, ...]]] = {}
    strict_failure_documents: dict[str, tuple[str, ...]] = {}
    for document_id, source_row in normal_by_id.items():
        value = source_row.value
        relative = value.get("validatedAnnotationPath")
        digest = value.get("validatedAnnotationSha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise DatasetAlignmentError(f"legacy annotation reference is absent: {document_id}")
        indexed = annotation_index.get((relative, digest))
        if indexed is None:
            raise DatasetAlignmentError(f"legacy annotation is not hash-pinned: {document_id}")
        root_id, annotation_path = indexed
        payload = read_regular_file_bytes(annotation_path)
        try:
            annotation = BillOfLadingAnnotation.model_validate_json(payload, strict=True)
            AgentWorkItem(
                source=annotation.source,
                joinedRawText=cast(str, value["joinedRawText"]),
            )
        except ValueError as error:
            raise DatasetAlignmentError(
                f"legacy annotation/source validation failed: {document_id}: {error}"
            ) from error
        strict_failures = _validate_legacy_evidence_verbatim(
            annotation, cast(str, value["joinedRawText"])
        )
        if strict_failures:
            strict_failure_documents[document_id] = strict_failures
        source_target = cast(dict[str, Any], value["target"])
        annotation_target = annotation.label.canonical_target()
        if annotation_target != source_target:
            corrections = prior_by_document.get(document_id, [])
            if not corrections:
                raise DatasetAlignmentError(
                    f"legacy annotation differs without a prior correction: {document_id}"
                )
            _verify_prior_corrections(
                annotation_target=annotation_target,
                source_target=source_target,
                joined_raw_text=cast(str, value["joinedRawText"]),
                corrections=corrections,
                document_id=document_id,
            )
        elif document_id in prior_by_document:
            raise DatasetAlignmentError(
                f"prior correction declared for an unchanged annotation: {document_id}"
            )
        annotation_details[document_id] = (
            root_id,
            annotation_path,
            annotation,
            strict_failures,
        )
    if set(prior_by_document) != {
        document_id
        for document_id, source_row in normal_by_id.items()
        if annotation_details[document_id][2].label.canonical_target() != source_row.value["target"]
    }:
        raise DatasetAlignmentError(
            "prior-correction document coverage differs from annotation drift"
        )

    records: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    correction_audit: list[dict[str, Any]] = []
    package_categories_removed = 0
    container_categories_removed = 0
    container_codes_normalized = 0
    package_codes_normalized = 0
    for relation_source in relation_rows:
        document_id = _document_id(relation_source, context="legacy relation record")
        normal_source = normal_by_id[document_id]
        normal_value = normal_source.value
        relation_value = relation_source.value
        if (
            normal_value["joinedRawText"] != relation_value["joinedRawText"]
            or normal_value["joinedRawTextSha256"] != relation_value["joinedRawTextSha256"]
        ):
            raise DatasetAlignmentError(f"legacy normal/relation raw OCR differs: {document_id}")
        transform = relation_value.get("semanticV3Transform")
        if not isinstance(transform, dict):
            raise DatasetAlignmentError(
                f"legacy relation transform provenance is absent: {document_id}"
            )
        source_target = cast(dict[str, Any], normal_value["target"])
        relation_target = cast(dict[str, Any], relation_value["target"])
        if transform.get("sourceTargetCanonicalSha256") != sha256_bytes(
            canonical_json_bytes(source_target)
        ):
            raise DatasetAlignmentError(f"legacy source-target lineage differs: {document_id}")
        if transform.get("targetCanonicalSha256") != sha256_bytes(
            canonical_json_bytes(relation_target)
        ):
            raise DatasetAlignmentError(f"legacy relation-target lineage differs: {document_id}")

        normal_aligned = json.loads(json.dumps(source_target))
        normalized_containers, normalized_packages = _normal_type_policy(normal_aligned)
        container_codes_normalized += normalized_containers
        package_codes_normalized += normalized_packages
        document_corrections = policy_by_document.get(document_id, [])
        _apply_policy_corrections(
            normal_aligned,
            corrections=document_corrections,
            joined_raw_text=cast(str, normal_value["joinedRawText"]),
        )
        relation_aligned, removed_packages, removed_containers = _align_relation_target(
            relation_target, normal_aligned
        )
        package_categories_removed += removed_packages
        container_categories_removed += removed_containers
        normal_canonical, relation_canonical = _canonical_pair(normal_aligned, relation_aligned)
        _no_mapped_transport_categories(relation_canonical, document_id=document_id)
        records.append(_minimal_record(normal_value, normal_canonical, relation_canonical))
        root_id, annotation_path, _, strict_failures = annotation_details[document_id]
        correction_ids = tuple(row.correctionId for row in document_corrections)
        lineage.append(
            {
                "documentId": document_id,
                "sourceCorpus": "legacy_combined487",
                "sourceNormalDatasetId": configured.normal_dataset.dataset_id,
                "sourceNormalManifestSha256": sha256_bytes(normal_manifest_payload),
                "sourceNormalPath": str(normal_source.source_path),
                "sourceNormalRowNumber": normal_source.row_number,
                "sourceNormalSplit": normal_source.split,
                "sourceNormalTargetCanonicalSha256": sha256_bytes(
                    canonical_json_bytes(source_target)
                ),
                "sourceRelationDatasetId": configured.relation_dataset.dataset_id,
                "sourceRelationManifestSha256": sha256_bytes(relation_manifest_payload),
                "sourceRelationPath": str(relation_source.source_path),
                "sourceRelationRowNumber": relation_source.row_number,
                "sourceRelationSplit": relation_source.split,
                "sourceRelationTargetCanonicalSha256": sha256_bytes(
                    canonical_json_bytes(relation_target)
                ),
                "sourceValidatedAnnotationRootId": root_id,
                "sourceValidatedAnnotationPath": str(annotation_path),
                "sourceValidatedAnnotationSha256": normal_value["validatedAnnotationSha256"],
                "legacyCurrentStrictEvidenceFailurePaths": list(strict_failures),
                "policyCorrectionIds": list(correction_ids),
                "normalTargetCanonicalSha256": sha256_bytes(canonical_json_bytes(normal_canonical)),
                "targetCanonicalSha256": sha256_bytes(canonical_json_bytes(relation_canonical)),
            }
        )
        for correction in document_corrections:
            correction_audit.append(
                {
                    **correction.model_dump(mode="json"),
                    "joinedRawTextSha256": normal_value["joinedRawTextSha256"],
                    "normalTargetCanonicalSha256": sha256_bytes(
                        canonical_json_bytes(normal_canonical)
                    ),
                    "targetCanonicalSha256": sha256_bytes(canonical_json_bytes(relation_canonical)),
                }
            )
    if len(correction_audit) != len(policy.corrections):
        raise DatasetAlignmentError("not every policy correction was applied exactly once")
    audit = {
        "schemaVersion": 1,
        "sourceRecords": len(normal_rows),
        "relationRecords": len(relation_rows),
        "relationExclusions": len(excluded_ids),
        "alignedRecords": len(records),
        "normalRelationConsistencyPass": len(records),
        "legacyAnnotationsHashAndSchemaValid": len(annotation_details),
        "legacyEvidenceVerbatimPageOrderedValid": len(annotation_details),
        "legacyEvidenceCurrentNonoverlapValid": len(annotation_details)
        - len(strict_failure_documents),
        "legacyEvidenceOverlapOnlyDocuments": len(strict_failure_documents),
        "legacyEvidenceOverlapOnlyFields": sum(
            len(value) for value in strict_failure_documents.values()
        ),
        "legacyEvidenceOverlapOnlyDetails": strict_failure_documents,
        "priorRawLatinCorrectionDocuments": len(prior_by_document),
        "priorRawLatinCorrections": len(prior_rows),
        "currentPolicyCorrectionDocuments": len(policy_by_document),
        "currentPolicyCorrections": len(policy.corrections),
        "packageCategoriesRemoved": package_categories_removed,
        "containerCategoriesRemoved": container_categories_removed,
        "containerTypeCodesCanonicalizedToDescription": container_codes_normalized,
        "packageTypeCodesCanonicalizedToDescription": package_codes_normalized,
        "annotationSources": list(annotation_sources),
    }
    return tuple(records), tuple(lineage), tuple(correction_audit), audit


def _flatten_paths(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            next_path = f"{path}.{key}" if path else key
            yield from _flatten_paths(item, next_path)
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_paths(item, f"{path}[]")
    else:
        yield path, value


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _profile(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paths: Counter[str] = Counter()
    cargo_groups = 0
    cargo_packages = 0
    allocation_groups = 0
    containers = 0
    country_values = 0
    country_values_absent_from_raw = 0
    source_characters: list[int] = []
    target_bytes: list[int] = []
    for record in records:
        target = cast(dict[str, Any], record["target"])
        patch = cast(dict[str, Any], target["documentPatch"])
        cargo_groups += len(patch.get("cargoGroups", []))
        cargo_packages += len(patch.get("cargoPackages", []))
        allocation_groups += len(patch.get("cargoAllocationGroups", []))
        containers += len(patch.get("containers", []))
        source_characters.append(len(cast(str, record["joinedRawText"])))
        target_bytes.append(len(canonical_json_bytes(target)))
        normalized_raw = _normalized_text(cast(str, record["joinedRawText"]))
        for path, value in _flatten_paths(target["documentPatch"]):
            paths[path] += 1
            if path.endswith(".country") and isinstance(value, str):
                country_values += 1
                if _normalized_text(value) not in normalized_raw:
                    country_values_absent_from_raw += 1
    return {
        "records": len(records),
        "cargoGroups": cargo_groups,
        "cargoPackages": cargo_packages,
        "cargoAllocationGroups": allocation_groups,
        "containers": containers,
        "countryValues": country_values,
        "countryValuesAbsentFromRawOcr": country_values_absent_from_raw,
        "sourceCharacters": {
            "min": min(source_characters),
            "max": max(source_characters),
            "mean": round(sum(source_characters) / len(source_characters), 3),
        },
        "targetCanonicalBytes": {
            "min": min(target_bytes),
            "max": max(target_bytes),
            "mean": round(sum(target_bytes) / len(target_bytes), 3),
        },
        "leafPathCounts": dict(sorted(paths.items())),
    }


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _publish_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        existing = read_regular_file_bytes(_file(path, context="existing publication"))
        if existing != payload:
            raise DatasetAlignmentError(
                f"immutable publication already exists with different bytes: {path}"
            )
        return
    atomic_publish_bytes(path, payload)


def _file_entry(path: Path, root: Path, payload: bytes, rows: int, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": path.relative_to(root).as_posix(),
        "rows": rows,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _resolve_cross_source_duplicates(
    *,
    legacy_records: Sequence[dict[str, Any]],
    legacy_lineage: Sequence[dict[str, Any]],
    current_records: Sequence[dict[str, Any]],
    current_lineage: Sequence[dict[str, Any]],
    resolutions: Sequence[CrossSourceDuplicateResolution],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    """Apply explicit, exhaustive duplicate decisions across two source datasets."""

    legacy_by_raw = {
        cast(str, row["joinedRawTextSha256"]): row for row in legacy_records
    }
    current_by_raw = {
        cast(str, row["joinedRawTextSha256"]): row for row in current_records
    }
    overlap = set(legacy_by_raw) & set(current_by_raw)
    configured = {row.joined_raw_text_sha256: row for row in resolutions}
    if missing := sorted(overlap - set(configured)):
        raise DatasetAlignmentError(
            "legacy/current raw OCR hashes lack explicit duplicate resolutions: "
            + ", ".join(missing)
        )
    if unexpected := sorted(set(configured) - overlap):
        raise DatasetAlignmentError(
            "configured duplicate resolutions are not cross-source overlaps: "
            + ", ".join(unexpected)
        )

    drop_legacy_ids: set[str] = set()
    drop_current_ids: set[str] = set()
    audit: list[dict[str, Any]] = []
    for joined_hash in sorted(overlap):
        resolution = configured[joined_hash]
        legacy_row = legacy_by_raw[joined_hash]
        current_row = current_by_raw[joined_hash]
        legacy_document_id = cast(str, legacy_row["documentId"])
        current_document_id = cast(str, current_row["documentId"])
        if legacy_document_id != resolution.legacy_document_id:
            raise DatasetAlignmentError(
                "duplicate-resolution legacy document ID differs for raw OCR hash: "
                f"{joined_hash}"
            )
        if current_document_id != resolution.current_document_id:
            raise DatasetAlignmentError(
                "duplicate-resolution current document ID differs for raw OCR hash: "
                f"{joined_hash}"
            )
        if resolution.retain == "current":
            drop_legacy_ids.add(legacy_document_id)
            retained_source = "current"
            retained_document_id = current_document_id
            discarded_source = "legacy"
            discarded_document_id = legacy_document_id
        else:
            drop_current_ids.add(current_document_id)
            retained_source = "legacy"
            retained_document_id = legacy_document_id
            discarded_source = "current"
            discarded_document_id = current_document_id
        audit.append(
            {
                "joinedRawTextSha256": joined_hash,
                "retainedSource": retained_source,
                "retainedDocumentId": retained_document_id,
                "discardedSource": discarded_source,
                "discardedDocumentId": discarded_document_id,
                "legacyTargetCanonicalSha256": sha256_bytes(
                    canonical_json_bytes(legacy_row["target"])
                ),
                "currentTargetCanonicalSha256": sha256_bytes(
                    canonical_json_bytes(current_row["target"])
                ),
                "rationale": resolution.rationale,
            }
        )

    retained_legacy_records = tuple(
        row for row in legacy_records if row["documentId"] not in drop_legacy_ids
    )
    retained_current_records = tuple(
        row for row in current_records if row["documentId"] not in drop_current_ids
    )
    retained_legacy_lineage = tuple(
        row for row in legacy_lineage if row["documentId"] not in drop_legacy_ids
    )
    retained_current_lineage = tuple(
        row for row in current_lineage if row["documentId"] not in drop_current_ids
    )
    retained_raw = {
        cast(str, row["joinedRawTextSha256"])
        for row in (*retained_legacy_records, *retained_current_records)
    }
    retained_count = len(retained_legacy_records) + len(retained_current_records)
    if len(retained_raw) != retained_count:
        raise DatasetAlignmentError("cross-source duplicate resolution left duplicate raw OCR")
    return (
        retained_legacy_records,
        retained_legacy_lineage,
        retained_current_records,
        retained_current_lineage,
        tuple(audit),
    )


def align_and_combine_datasets(config: DatasetAlignmentConfig) -> Path:
    """Publish aligned legacy records and a disjoint combined training corpus."""

    legacy_records, legacy_lineage, correction_audit, legacy_audit = _load_legacy_records(
        config.legacy
    )
    current_records, current_lineage, current_manifest = _load_current_records(config.current)
    legacy_ids = {cast(str, row["documentId"]) for row in legacy_records}
    current_ids = {cast(str, row["documentId"]) for row in current_records}
    if overlap := sorted(legacy_ids & current_ids):
        raise DatasetAlignmentError("legacy/current document IDs overlap: " + ", ".join(overlap))
    (
        combined_legacy_records,
        combined_legacy_lineage,
        combined_current_records,
        combined_current_lineage,
        duplicate_audit,
    ) = _resolve_cross_source_duplicates(
        legacy_records=legacy_records,
        legacy_lineage=legacy_lineage,
        current_records=current_records,
        current_lineage=current_lineage,
        resolutions=config.cross_source_duplicate_resolutions,
    )

    aligned_root = Path(config.output.aligned_legacy_root).resolve(strict=False)
    combined_root = Path(config.output.combined_root).resolve(strict=False)
    for root in (aligned_root, combined_root):
        if root == Path(root.anchor):
            raise DatasetAlignmentError("refusing to publish at filesystem root")
        root.parent.mkdir(parents=True, exist_ok=True)

    legacy_record_payload = _jsonl_bytes(legacy_records)
    legacy_lineage_payload = _jsonl_bytes(legacy_lineage)
    correction_payload = _jsonl_bytes(correction_audit)
    legacy_quality = {
        **legacy_audit,
        "profile": _profile(legacy_records),
    }
    legacy_quality_payload = canonical_json_bytes(legacy_quality) + b"\n"
    legacy_record_path = aligned_root / "records.jsonl"
    legacy_lineage_path = aligned_root / "lineage.jsonl"
    correction_path = aligned_root / "correction-audit.jsonl"
    legacy_quality_path = aligned_root / "quality-audit.json"
    for path, payload in (
        (legacy_record_path, legacy_record_payload),
        (legacy_lineage_path, legacy_lineage_payload),
        (correction_path, correction_payload),
        (legacy_quality_path, legacy_quality_payload),
    ):
        _publish_immutable(path, payload)
    legacy_manifest = {
        "schemaVersion": 1,
        "datasetId": config.output.aligned_legacy_dataset_id,
        "task": "bill_of_lading_relation_single_source_v4",
        "records": len(legacy_records),
        "sourceRecords": config.legacy.expected_source_records,
        "declaredRelationExclusions": config.legacy.expected_relation_exclusions,
        "qualityAligned": True,
        "policies": {
            "dates": "ambiguous_numeric_dates_day_first",
            "localitiesAndCountries": "printed_text_only",
            "packageAndContainerTypes": "printed_text_only_downstream_mapping",
            "goodsDescriptions": "product_wording_without_separately_modeled_packaging",
        },
        "sourceDatasets": [
            config.legacy.normal_dataset.model_dump(mode="json"),
            config.legacy.relation_dataset.model_dump(mode="json"),
        ],
        "files": [
            _file_entry(
                legacy_record_path,
                aligned_root,
                legacy_record_payload,
                len(legacy_records),
                "training_records",
            ),
            _file_entry(
                legacy_lineage_path,
                aligned_root,
                legacy_lineage_payload,
                len(legacy_lineage),
                "lineage",
            ),
            _file_entry(
                correction_path,
                aligned_root,
                correction_payload,
                len(correction_audit),
                "correction_audit",
            ),
            _file_entry(
                legacy_quality_path, aligned_root, legacy_quality_payload, 1, "quality_audit"
            ),
        ],
    }
    legacy_manifest_payload = canonical_json_bytes(legacy_manifest) + b"\n"
    _publish_immutable(aligned_root / "manifest.json", legacy_manifest_payload)

    combined_records = (*combined_legacy_records, *combined_current_records)
    combined_lineage = (
        *(
            {
                **row,
                "immediateSourceDatasetId": config.output.aligned_legacy_dataset_id,
                "immediateSourceDatasetManifestSha256": sha256_bytes(legacy_manifest_payload),
            }
            for row in combined_legacy_lineage
        ),
        *(
            {
                **row,
                "immediateSourceDatasetId": config.current.dataset.dataset_id,
                "immediateSourceDatasetManifestSha256": config.current.dataset.manifest_sha256,
            }
            for row in combined_current_lineage
        ),
    )
    combined_record_payload = _jsonl_bytes(combined_records)
    combined_lineage_payload = _jsonl_bytes(combined_lineage)
    duplicate_audit_payload = _jsonl_bytes(duplicate_audit)
    comparison = {
        "schemaVersion": 1,
        "legacy": _profile(legacy_records),
        "current": _profile(current_records),
        "combined": _profile(combined_records),
        "overlap": {
            "documentIds": 0,
            "joinedRawTextSha256": 0,
            "resolvedJoinedRawTextSha256": len(duplicate_audit),
        },
        "validation": {
            "legacyNormalRelationConsistencyPass": len(legacy_records),
            "currentNormalRelationConsistencyPass": len(current_records),
            "combinedSchemaValid": len(combined_records),
            "mappedPackageOrContainerCategories": 0,
        },
    }
    comparison_payload = canonical_json_bytes(comparison) + b"\n"
    combined_record_path = combined_root / "records.jsonl"
    combined_lineage_path = combined_root / "lineage.jsonl"
    duplicate_audit_path = combined_root / "cross-source-duplicate-resolutions.jsonl"
    comparison_path = combined_root / "quality-comparison.json"
    for path, payload in (
        (combined_record_path, combined_record_payload),
        (combined_lineage_path, combined_lineage_payload),
        (duplicate_audit_path, duplicate_audit_payload),
        (comparison_path, comparison_payload),
    ):
        _publish_immutable(path, payload)
    combined_manifest = {
        "schemaVersion": 1,
        "datasetId": config.output.combined_dataset_id,
        "task": "bill_of_lading_relation_single_source_v4",
        "records": len(combined_records),
        "trainingReady": True,
        "qualityAligned": True,
        "sourceDatasets": [
            {
                "datasetId": config.output.aligned_legacy_dataset_id,
                "manifestSha256": sha256_bytes(legacy_manifest_payload),
                "availableRecords": len(legacy_records),
                "records": len(combined_legacy_records),
            },
            {
                "datasetId": config.current.dataset.dataset_id,
                "manifestSha256": config.current.dataset.manifest_sha256,
                "availableRecords": len(current_records),
                "records": len(combined_current_records),
                "sourceOutcomes": current_manifest.get("outcomes"),
            },
        ],
        "crossSourceDuplicateResolutions": len(duplicate_audit),
        "files": [
            _file_entry(
                combined_record_path,
                combined_root,
                combined_record_payload,
                len(combined_records),
                "training_records",
            ),
            _file_entry(
                combined_lineage_path,
                combined_root,
                combined_lineage_payload,
                len(combined_lineage),
                "lineage",
            ),
            _file_entry(
                duplicate_audit_path,
                combined_root,
                duplicate_audit_payload,
                len(duplicate_audit),
                "cross_source_duplicate_resolutions",
            ),
            _file_entry(
                comparison_path, combined_root, comparison_payload, 1, "quality_comparison"
            ),
        ],
    }
    manifest_path = combined_root / "manifest.json"
    _publish_immutable(manifest_path, canonical_json_bytes(combined_manifest) + b"\n")
    return manifest_path


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise DatasetAlignmentError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _ArgumentParser(
        prog="document-kie-align-datasets",
        description="Align legacy B/L labels to the current contract and merge datasets.",
    )
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    config = load_dataset_alignment_config(arguments.config)
    manifest = align_and_combine_datasets(config)
    print(
        json.dumps(
            {"command": "align-and-combine", "manifest": str(manifest), "status": "complete"},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
