"""Readiness audit and immutable semantic-v2 -> relation-explicit-v3 transform."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from math import ceil
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from pydantic import Field, StringConstraints, model_validator

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading import BillOfLadingLabel
from document_ocr.label_schemas.bill_of_lading_v3 import BillOfLadingRelationExplicitLabel
from document_ocr.label_schemas.common import LabelSchemaModel
from document_ocr.semantic_v3.config import (
    FrozenArtifactInput,
    SemanticV3SourceFile,
    SemanticV3TransformConfig,
)
from document_ocr.training.config import TrainingConfig, load_training_config
from document_ocr.training.tasks import (
    RelationExplicitTaskConstraints,
    get_training_task,
)

CategoryToken = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$", max_length=96),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_RELATION_POLICY = "allocation_total_unique_match_v1"
_CATEGORY_POLICY = "exact_source_key_to_frozen_one_to_one_token_v1"
_CONTAINER_DESCRIPTION_POLICY = "preserve_printed_description_replace_wire_code_v1"
_DANGEROUS_GOODS_CATEGORY_POLICY = "fixed_imdg_class_and_packing_group_semantics_v1"

_HAZARD_CLASS_TO_CATEGORY = {
    "1": "EXPLOSIVES",
    "2": "GASES",
    "3": "FLAMMABLE_LIQUIDS",
    "4": "FLAMMABLE_SOLIDS",
    "5": "OXIDIZING_SUBSTANCES_AND_ORGANIC_PEROXIDES",
    "6": "TOXIC_AND_INFECTIOUS_SUBSTANCES",
    "7": "RADIOACTIVE_MATERIAL",
    "8": "CORROSIVE_SUBSTANCES",
    "9": "MISCELLANEOUS_DANGEROUS_SUBSTANCES_AND_ARTICLES",
}
_PACKING_GROUP_TO_CATEGORY = {
    "I": "HIGH_DANGER",
    "II": "MEDIUM_DANGER",
    "III": "LOW_DANGER",
}


class SemanticV3TransformError(RuntimeError):
    """The v3 audit or transform cannot preserve its declared contract."""


def _target_schema_sha256() -> str:
    return sha256_bytes(
        canonical_json_bytes(
            BillOfLadingRelationExplicitLabel.model_json_schema(mode="serialization")
        )
    )


def _transform_contract_sha256(config: SemanticV3TransformConfig) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "categoryPolicy": _CATEGORY_POLICY,
                "containerDescriptionPolicy": _CONTAINER_DESCRIPTION_POLICY,
                "dangerousGoodsCategoryPolicy": _DANGEROUS_GOODS_CATEGORY_POLICY,
                "relationPolicy": _RELATION_POLICY,
                "targetSchemaSha256": _target_schema_sha256(),
                "transform": config.transform,
            }
        )
    )


class CategoryRegistryEntry(LabelSchemaModel):
    categoryToken: CategoryToken
    applicationCode: Annotated[str, StringConstraints(pattern=r"^[0-9A-Z]{2,4}$")]
    displayName: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CategoryRegistry(LabelSchemaModel):
    schemaVersion: Literal[1]
    registryKind: Literal["package", "container"]
    sourceAuthority: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    sourceRevision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    sourcePath: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    sourceSha256: Sha256
    entries: tuple[CategoryRegistryEntry, ...]

    @model_validator(mode="after")
    def entries_are_unique_and_codes_match_kind(self) -> CategoryRegistry:
        if self.registryKind == "package" and not self.entries:
            raise ValueError("package category registry must contain at least one entry")
        tokens = tuple(row.categoryToken for row in self.entries)
        codes = tuple(row.applicationCode for row in self.entries)
        if len(tokens) != len(set(tokens)) or len(codes) != len(set(codes)):
            raise ValueError("category registry tokens and application codes must be unique")
        expected_length = 2 if self.registryKind == "package" else 4
        if any(len(code) != expected_length for code in codes):
            raise ValueError(
                f"{self.registryKind} application codes must be {expected_length} chars"
            )
        return self


class PackageCategoryAssignment(LabelSchemaModel):
    sourceTypeText: str | None
    sourceTypeCode: str | None
    occurrences: Annotated[int, Field(gt=0)]
    categoryToken: CategoryToken | None
    reviewBasis: Literal[
        "exact_printed_code",
        "manual_semantic_review",
        "insufficient_source_specificity",
    ]
    reviewedBy: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    reviewedOn: date

    @model_validator(mode="after")
    def source_is_present(self) -> PackageCategoryAssignment:
        if self.sourceTypeText is None and self.sourceTypeCode is None:
            raise ValueError("package assignment requires source text or a printed code")
        if self.categoryToken is None:
            if self.reviewBasis != "insufficient_source_specificity":
                raise ValueError(
                    "an unresolved package category requires insufficient_source_specificity"
                )
            if self.sourceTypeText is None:
                raise ValueError(
                    "an unresolved package category requires printed text for the v3 fallback"
                )
        elif self.reviewBasis == "insufficient_source_specificity":
            raise ValueError(
                "a resolved package category cannot use insufficient_source_specificity"
            )
        return self


class ContainerCategoryAssignment(LabelSchemaModel):
    sourceTypeDescription: str | None
    sourceTypeCode: str | None
    occurrences: Annotated[int, Field(gt=0)]
    categoryToken: CategoryToken | None
    reviewBasis: Literal[
        "exact_printed_code",
        "manual_semantic_review",
        "insufficient_source_specificity",
    ]
    reviewedBy: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    reviewedOn: date

    @model_validator(mode="after")
    def source_is_present(self) -> ContainerCategoryAssignment:
        if self.sourceTypeDescription is None and self.sourceTypeCode is None:
            raise ValueError("container assignment requires a description or printed code")
        if self.categoryToken is None:
            if self.reviewBasis != "insufficient_source_specificity":
                raise ValueError(
                    "an unresolved container category requires insufficient_source_specificity"
                )
        elif self.reviewBasis == "insufficient_source_specificity":
            raise ValueError(
                "a resolved container category cannot use insufficient_source_specificity"
            )
        return self


class CategoryAssignments(LabelSchemaModel):
    schemaVersion: Literal[2]
    reviewConfigSha256: Sha256
    sourceInventorySha256: Sha256
    packageRegistrySha256: Sha256
    containerRegistrySha256: Sha256
    packageAssignments: tuple[PackageCategoryAssignment, ...]
    containerAssignments: tuple[ContainerCategoryAssignment, ...]

    @model_validator(mode="after")
    def source_keys_are_unique(self) -> CategoryAssignments:
        package_keys = tuple(
            (row.sourceTypeText, row.sourceTypeCode) for row in self.packageAssignments
        )
        container_keys = tuple(
            (row.sourceTypeDescription, row.sourceTypeCode) for row in self.containerAssignments
        )
        if len(package_keys) != len(set(package_keys)):
            raise ValueError("package category assignments contain duplicate source keys")
        if len(container_keys) != len(set(container_keys)):
            raise ValueError("container category assignments contain duplicate source keys")
        return self


class CargoRelationAssignment(LabelSchemaModel):
    documentId: Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
    goodsIndex: Annotated[int, Field(ge=0)]
    sourceRelationSha256: Sha256
    classification: Literal[
        "one_to_one_package_allocations",
        "single_package_level",
        "all_package_levels_combined",
    ]
    reviewBasis: Literal["raw_ocr_and_semantic_v2_review"]
    reviewedBy: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    reviewedOn: date


class CargoRelationAssignments(LabelSchemaModel):
    schemaVersion: Literal[1]
    sourceInventorySha256: Sha256
    assignments: tuple[CargoRelationAssignment, ...]

    @model_validator(mode="after")
    def relation_keys_are_unique(self) -> CargoRelationAssignments:
        keys = tuple((row.documentId, row.goodsIndex) for row in self.assignments)
        if len(keys) != len(set(keys)):
            raise ValueError("cargo relation assignments contain duplicate document/goods keys")
        return self


@dataclass(frozen=True, slots=True)
class _SourceRow:
    source: SemanticV3SourceFile
    row_number: int
    value: dict[str, Any]
    label: BillOfLadingLabel


@dataclass(frozen=True, slots=True)
class _RelationAudit:
    document_id: str
    goods_index: int
    classification: str
    package_quantities: tuple[int, ...]
    allocation_quantities: tuple[int | None, ...]
    matched_package_indexes: tuple[int, ...]
    blocking_reason: str | None

    def row(self) -> dict[str, Any]:
        return {
            "allocationQuantities": self.allocation_quantities,
            "blockingReason": self.blocking_reason,
            "classification": self.classification,
            "documentId": self.document_id,
            "goodsIndex": self.goods_index,
            "matchedPackageIndexes": self.matched_package_indexes,
            "packageQuantities": self.package_quantities,
        }


def _canonical_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    absolute = Path(os.path.abspath(path))
    resolved = path.resolve(strict=True)
    if resolved != absolute or resolved.is_symlink():
        raise SemanticV3TransformError(f"output directory must be canonical: {path}")
    return resolved


def _canonical_file(path: Path, *, context: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SemanticV3TransformError(f"missing {context}: {path}") from error
    if resolved != absolute or not resolved.is_file() or resolved.is_symlink():
        raise SemanticV3TransformError(f"{context} must be a canonical regular file: {path}")
    return resolved


def _strict_json(payload: bytes, *, context: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite number {value!r}")

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (TypeError, UnicodeError, ValueError) as error:
        raise SemanticV3TransformError(f"invalid JSON: {context}") from error
    if not isinstance(decoded, dict):
        raise SemanticV3TransformError(f"JSON root must be an object: {context}")
    return cast(dict[str, Any], decoded)


def _source_inventory_sha256(config: SemanticV3TransformConfig) -> str:
    rows = [
        {
            "id": row.id,
            "path": row.path,
            "records": row.records,
            "sha256": row.sha256,
            "split": row.split,
        }
        for row in config.source.files
    ]
    return sha256_bytes(canonical_json_bytes(rows))


def _load_source_rows(config: SemanticV3TransformConfig) -> tuple[_SourceRow, ...]:
    rows: list[_SourceRow] = []
    document_ids: set[str] = set()
    for source in config.source.files:
        path = _canonical_file(Path(source.path), context="semantic-v3 source")
        payload = read_regular_file_bytes(path)
        if sha256_bytes(payload) != source.sha256:
            raise SemanticV3TransformError(f"source SHA-256 mismatch: {path}")
        source_count = 0
        for row_number, line in enumerate(payload.splitlines(), start=1):
            if not line:
                raise SemanticV3TransformError(f"blank source line: {path}:{row_number}")
            value = _strict_json(line, context=f"{path}:{row_number}")
            document_id = value.get("documentId")
            joined_text = value.get("joinedRawText")
            joined_digest = value.get("joinedRawTextSha256")
            target = value.get("target")
            if (
                not isinstance(document_id, str)
                or not document_id.startswith("doc_")
                or len(document_id) != 68
                or any(character not in "0123456789abcdef" for character in document_id[4:])
            ):
                raise SemanticV3TransformError(f"invalid documentId: {path}:{row_number}")
            if document_id in document_ids:
                raise SemanticV3TransformError(
                    f"duplicate documentId across sources: {document_id}"
                )
            if (
                not isinstance(joined_text, str)
                or not isinstance(joined_digest, str)
                or sha256_bytes(joined_text.encode("utf-8")) != joined_digest
            ):
                raise SemanticV3TransformError(f"joinedRawText digest mismatch: {document_id}")
            if not isinstance(target, dict):
                raise SemanticV3TransformError(f"target must be an object: {document_id}")
            try:
                label = BillOfLadingLabel.model_validate_json(
                    canonical_json_bytes(target), strict=True
                )
            except ValueError as error:
                raise SemanticV3TransformError(
                    f"semantic-v2 target validation failed: {document_id}: {error}"
                ) from error
            if label.canonical_target() != target:
                raise SemanticV3TransformError(f"semantic-v2 target is noncanonical: {document_id}")
            document_ids.add(document_id)
            source_count += 1
            rows.append(_SourceRow(source, row_number, value, label))
        if source_count != source.records:
            raise SemanticV3TransformError(
                f"source row count differs for {path}: expected {source.records}, "
                f"found {source_count}"
            )
    if len(rows) != config.source.expected_documents:
        raise SemanticV3TransformError("semantic-v3 source document count differs from config")
    return tuple(rows)


def _relation_audit(document_id: str, goods_index: int, goods: dict[str, Any]) -> _RelationAudit:
    packages = goods.get("packages", [])
    allocations = goods.get("containerAllocations", [])
    package_quantities = tuple(
        row["quantity"]
        for row in packages
        if isinstance(row, dict) and isinstance(row.get("quantity"), int)
    )
    allocation_quantities = tuple(
        row.get("packageQuantity") if isinstance(row, dict) else None for row in allocations
    )
    matched: tuple[int, ...] = ()
    blocker: str | None = None
    if not allocations:
        classification = "no_container_allocations"
    elif (
        packages
        and len(package_quantities) == len(packages) == len(allocation_quantities)
        and all(value is not None for value in allocation_quantities)
        and package_quantities == allocation_quantities
    ):
        classification = "one_to_one_package_allocations"
        matched = tuple(range(len(package_quantities)))
    elif package_quantities and all(value is not None for value in allocation_quantities):
        total = sum(cast(int, value) for value in allocation_quantities)
        matched = tuple(
            index for index, quantity in enumerate(package_quantities) if quantity == total
        )
        if len(matched) == 1:
            classification = "single_package_level"
        elif len(matched) > 1:
            classification = "ambiguous_duplicate_package_quantity"
            blocker = classification
        elif len(package_quantities) > 1 and total == sum(package_quantities):
            classification = "all_package_levels_combined"
        else:
            classification = "allocation_total_mismatch"
            blocker = classification
    elif package_quantities:
        classification = "incomplete_container_allocation_quantities"
        blocker = classification
    elif all(value is not None for value in allocation_quantities):
        classification = "unlinked_package_quantities"
    elif all(value is None for value in allocation_quantities):
        classification = "container_membership_only"
    else:
        classification = "mixed_unlinked_allocation_quantities"
        blocker = classification
    return _RelationAudit(
        document_id,
        goods_index,
        classification,
        package_quantities,
        allocation_quantities,
        matched,
        blocker,
    )


def _relation_source_sha256(goods: dict[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "containerAllocations": goods.get("containerAllocations", []),
                "packages": goods.get("packages", []),
            }
        )
    )


def _requires_relation_review(goods: dict[str, Any], audit: _RelationAudit) -> bool:
    return (
        audit.blocking_reason is None
        and len(goods.get("packages", [])) > 1
        and bool(goods.get("containerAllocations", []))
    )


def _valid_reviewed_relation_choices(goods: dict[str, Any]) -> set[str]:
    packages = goods.get("packages", [])
    allocations = goods.get("containerAllocations", [])
    package_quantities = [row.get("quantity") for row in packages]
    allocation_quantities = [row.get("packageQuantity") for row in allocations]
    if (
        not packages
        or not allocations
        or any(not isinstance(value, int) for value in package_quantities)
        or any(not isinstance(value, int) for value in allocation_quantities)
    ):
        return set()
    typed_packages = cast(list[int], package_quantities)
    typed_allocations = cast(list[int], allocation_quantities)
    total = sum(typed_allocations)
    choices: set[str] = set()
    if len(typed_packages) == len(typed_allocations) and typed_packages == typed_allocations:
        choices.add("one_to_one_package_allocations")
    if sum(quantity == total for quantity in typed_packages) == 1:
        choices.add("single_package_level")
    if len(typed_packages) > 1 and sum(typed_packages) == total:
        choices.add("all_package_levels_combined")
    return choices


def _inventories(
    rows: tuple[_SourceRow, ...],
) -> tuple[
    Counter[tuple[str | None, str | None]],
    Counter[tuple[str | None, str | None]],
    tuple[_RelationAudit, ...],
    dict[str, Counter[str]],
    dict[tuple[str | None, str | None], set[str]],
    dict[tuple[str | None, str | None], set[str]],
]:
    packages: Counter[tuple[str | None, str | None]] = Counter()
    containers: Counter[tuple[str | None, str | None]] = Counter()
    package_documents: dict[tuple[str | None, str | None], set[str]] = defaultdict(set)
    container_documents: dict[tuple[str | None, str | None], set[str]] = defaultdict(set)
    relations: list[_RelationAudit] = []
    categoricals: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        document_id = cast(str, row.value["documentId"])
        patch = cast(dict[str, Any], row.value["target"]["documentPatch"])
        for field_name in ("negotiability",):
            value = patch.get(field_name)
            if isinstance(value, str):
                categoricals[field_name][value] += 1
        freight = patch.get("freight")
        if isinstance(freight, dict) and isinstance(freight.get("paymentArrangement"), str):
            categoricals["freight.paymentArrangement"][freight["paymentArrangement"]] += 1
        parties = patch.get("parties")
        if isinstance(parties, dict):
            for notify_party in parties.get("notifyParties", []):
                if isinstance(notify_party, dict) and isinstance(
                    notify_party.get("sameAs"), str
                ):
                    categoricals["parties.notifyParties.sameAs"][notify_party["sameAs"]] += 1
        for container in patch.get("containers", []):
            description = container.get("typeDescription")
            code = container.get("typeCode")
            if description is not None or code is not None:
                key = (description, code)
                containers[key] += 1
                container_documents[key].add(document_id)
            for measurement_name in ("verifiedGrossMass", "temperatureSetpoint"):
                measurement = container.get(measurement_name)
                if isinstance(measurement, dict) and isinstance(measurement.get("unit"), str):
                    categoricals[f"containers.{measurement_name}.unit"][measurement["unit"]] += 1
        for goods_index, goods in enumerate(patch.get("goodsItems", [])):
            relations.append(_relation_audit(document_id, goods_index, goods))
            for package in goods.get("packages", []):
                type_text = package.get("type")
                type_code = package.get("typeCode")
                if type_text is not None or type_code is not None:
                    key = (type_text, type_code)
                    packages[key] += 1
                    package_documents[key].add(document_id)
            for measurement_name in ("grossWeight", "netWeight", "volume"):
                measurement = goods.get(measurement_name)
                if isinstance(measurement, dict) and isinstance(measurement.get("unit"), str):
                    categoricals[f"goodsItems.{measurement_name}.unit"][measurement["unit"]] += 1
            for dangerous in goods.get("dangerousGoods", []):
                hazard = dangerous.get("hazardClass")
                if isinstance(hazard, str):
                    categoricals["dangerousGoods.hazardClass"][hazard] += 1
                subsidiary = dangerous.get("subsidiaryHazard")
                if isinstance(subsidiary, str):
                    categoricals["dangerousGoods.subsidiaryHazard"][subsidiary] += 1
                flash = dangerous.get("flashPoint")
                if isinstance(flash, dict) and isinstance(flash.get("packingGroup"), str):
                    categoricals["dangerousGoods.packingGroup"][flash["packingGroup"]] += 1
    return (
        packages,
        containers,
        tuple(relations),
        categoricals,
        package_documents,
        container_documents,
    )


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def audit_semantic_v3_readiness(config: SemanticV3TransformConfig) -> Path:
    """Publish a manifest-last audit without altering or cloning source records."""

    source_rows = _load_source_rows(config)
    inventory_sha256 = _source_inventory_sha256(config)
    (
        package_counts,
        container_counts,
        relations,
        categoricals,
        package_documents,
        container_documents,
    ) = _inventories(source_rows)
    output_root = _canonical_directory(Path(config.output.audit_root))
    package_rows = [
        {
            "affectedDocuments": sorted(package_documents[key]),
            "assignmentStatus": "unassigned",
            "candidateCategoryToken": None,
            "occurrences": count,
            "sourceTypeCode": key[1],
            "sourceTypeText": key[0],
        }
        for key, count in sorted(package_counts.items(), key=lambda item: (-item[1], str(item[0])))
    ]
    container_rows = [
        {
            "affectedDocuments": sorted(container_documents[key]),
            "assignmentStatus": "unassigned",
            "candidateCategoryToken": None,
            "occurrences": count,
            "sourceTypeCode": key[1],
            "sourceTypeDescription": key[0],
        }
        for key, count in sorted(
            container_counts.items(), key=lambda item: (-item[1], str(item[0]))
        )
    ]
    relation_rows = [row.row() for row in relations]
    relation_by_key = {(row.document_id, row.goods_index): row for row in relations}
    relation_review_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        document_id = cast(str, source_row.value["documentId"])
        patch = cast(dict[str, Any], source_row.value["target"]["documentPatch"])
        for goods_index, goods in enumerate(patch.get("goodsItems", [])):
            relation = relation_by_key[(document_id, goods_index)]
            if not _requires_relation_review(goods, relation):
                continue
            relation_review_rows.append(
                {
                    "candidateClassification": relation.classification,
                    "containerAllocations": goods.get("containerAllocations", []),
                    "documentId": document_id,
                    "goodsIndex": goods_index,
                    "joinedRawTextSha256": source_row.value["joinedRawTextSha256"],
                    "packages": goods.get("packages", []),
                    "reviewStatus": "unreviewed",
                    "sourceId": source_row.source.id,
                    "sourceRelationSha256": _relation_source_sha256(goods),
                    "sourceRowNumber": source_row.row_number,
                }
            )
    blockers_by_document: dict[str, list[str]] = defaultdict(list)
    for relation in relations:
        if relation.blocking_reason is not None:
            blockers_by_document[relation.document_id].append(
                f"goodsItems[{relation.goods_index}]:{relation.blocking_reason}"
            )
    document_rows = [
        {
            "documentId": row.value["documentId"],
            "relationBlockers": blockers_by_document.get(row.value["documentId"], []),
            "split": row.source.split,
            "sourceId": row.source.id,
            "sourceRowNumber": row.row_number,
            "transformableAfterCategoryReview": not blockers_by_document.get(
                row.value["documentId"]
            ),
        }
        for row in source_rows
    ]
    categorical_payload = {
        "schemaVersion": 1,
        "ownership": {
            "container.typeCategory": (
                "omitted until the platform supplies a stable semantic registry for exact "
                "four-character codes"
            ),
            "container.typeDescription": (
                "printed source wording, or the printed code when no wording exists"
            ),
            "dangerousGoods.hazardCategory": (
                "model-readable IMDG class meaning; deterministic meaning-to-class projection"
            ),
            "dangerousGoods.packingGroupCategory": (
                "model-readable danger level; deterministic level-to-packing-group projection"
            ),
            "package.typeCategory": (
                "model-readable registry token when source-specific; deterministic "
                "token-to-UNECE-Rec21-code projection; otherwise omitted"
            ),
            "package.typeDescription": (
                "printed fallback only when no exact registry category is source-supported"
            ),
            "country_and_locality": "printed source text; ISO-2 and UN/LOCODE remain downstream",
            "small_semantic_enums": (
                "model-readable values retained; fixed application wire codes remain downstream"
            ),
            "identifiers": "source-supported identifiers, not categoricals",
        },
        "observedSmallCategoricals": {
            key: dict(sorted(counter.items())) for key, counter in sorted(categoricals.items())
        },
    }
    published: list[dict[str, Any]] = []
    for name, kind, rows in (
        ("package-category-review.jsonl", "package_category_review", package_rows),
        ("container-category-review.jsonl", "container_category_review", container_rows),
        ("cargo-relation-audit.jsonl", "cargo_relation_audit", relation_rows),
        ("cargo-relation-review.jsonl", "cargo_relation_review", relation_review_rows),
        ("document-readiness.jsonl", "document_readiness", document_rows),
    ):
        payload = _jsonl(rows)
        atomic_publish_bytes(output_root / name, payload)
        published.append(
            {
                "bytes": len(payload),
                "kind": kind,
                "path": name,
                "rows": len(rows),
                "sha256": sha256_bytes(payload),
            }
        )
    categorical_bytes = canonical_json_bytes(categorical_payload) + b"\n"
    atomic_publish_bytes(output_root / "categorical-ownership.json", categorical_bytes)
    published.append(
        {
            "bytes": len(categorical_bytes),
            "kind": "categorical_ownership",
            "path": "categorical-ownership.json",
            "rows": 1,
            "sha256": sha256_bytes(categorical_bytes),
        }
    )
    for name, kind, model in (
        ("category-registry.schema.json", "category_registry_schema", CategoryRegistry),
        (
            "category-assignments.schema.json",
            "category_assignments_schema",
            CategoryAssignments,
        ),
        (
            "cargo-relation-assignments.schema.json",
            "cargo_relation_assignments_schema",
            CargoRelationAssignments,
        ),
    ):
        schema_payload = canonical_json_bytes(model.model_json_schema(mode="serialization")) + b"\n"
        atomic_publish_bytes(output_root / name, schema_payload)
        published.append(
            {
                "bytes": len(schema_payload),
                "kind": kind,
                "path": name,
                "rows": 1,
                "sha256": sha256_bytes(schema_payload),
            }
        )
    relation_counter = Counter(row.classification for row in relations)
    actual_relation_exclusions = {
        row.document_id: row.blocking_reason for row in relations if row.blocking_reason is not None
    }
    allowed_relation_exclusions = {
        row.document_id: row.reason for row in config.allowed_relation_exclusions
    }
    registry_status = {
        "package_categories": config.registries.package_categories is not None,
        "container_categories": config.registries.container_categories is not None,
        "category_assignments": config.registries.category_assignments is not None,
        "relation_assignments": config.registries.relation_assignments is not None,
    }
    summary = {
        "schema_version": 1,
        "dataset_id": config.output.dataset_id,
        "source_documents": len(source_rows),
        "source_inventory_sha256": inventory_sha256,
        "target_schema_sha256": _target_schema_sha256(),
        "transform_contract_sha256": _transform_contract_sha256(config),
        "splits": dict(sorted(Counter(row.source.split for row in source_rows).items())),
        "package_category_occurrences": sum(package_counts.values()),
        "package_category_source_variants": len(package_counts),
        "container_category_occurrences": sum(container_counts.values()),
        "container_category_source_variants": len(container_counts),
        "cargo_relation_classifications": dict(sorted(relation_counter.items())),
        "cargo_relation_review_items": len(relation_review_rows),
        "relation_blocked_documents": len(actual_relation_exclusions),
        "actual_relation_exclusions": actual_relation_exclusions,
        "allowed_relation_exclusions": allowed_relation_exclusions,
        "relation_exclusions_match_config": actual_relation_exclusions
        == allowed_relation_exclusions,
        "registry_inputs_configured": registry_status,
        "transform_ready": (
            actual_relation_exclusions == allowed_relation_exclusions
            and all(registry_status.values())
        ),
        "next_action": (
            "Freeze the CargoX package/container registries at revision "
            "72f9aff735c2581f2ccff74e7fde34c2aea9b265 and review every exact source-key "
            "assignment before publication."
        ),
    }
    atomic_publish_json(output_root / "summary.json", summary)
    summary_bytes = read_regular_file_bytes(output_root / "summary.json")
    published.append(
        {
            "bytes": len(summary_bytes),
            "kind": "summary",
            "path": "summary.json",
            "rows": 1,
            "sha256": sha256_bytes(summary_bytes),
        }
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": config.output.dataset_id,
        "publication_state": "readiness_audit_only",
        "source_inventory_sha256": inventory_sha256,
        "target_schema_sha256": _target_schema_sha256(),
        "transform_contract_sha256": _transform_contract_sha256(config),
        "files": published,
    }
    manifest_path = output_root / "manifest.json"
    atomic_publish_json(manifest_path, manifest)
    return manifest_path


def _load_frozen_model[ModelT: LabelSchemaModel](
    configured: FrozenArtifactInput,
    model: type[ModelT],
) -> tuple[ModelT, str]:
    path = _canonical_file(Path(configured.path), context="frozen semantic-v3 artifact")
    payload = read_regular_file_bytes(path)
    digest = sha256_bytes(payload)
    if digest != configured.sha256:
        raise SemanticV3TransformError(f"frozen artifact SHA-256 mismatch: {path}")
    try:
        value = model.model_validate_json(payload, strict=True)
    except ValueError as error:
        raise SemanticV3TransformError(f"frozen artifact failed validation: {path}") from error
    return value, digest


def _category_inputs(
    config: SemanticV3TransformConfig,
    package_counts: Counter[tuple[str | None, str | None]],
    container_counts: Counter[tuple[str | None, str | None]],
) -> tuple[
    dict[tuple[str | None, str | None], str | None],
    dict[tuple[str | None, str | None], str | None],
    dict[str, str],
    dict[str, str],
]:
    registry_config = config.registries
    if (
        registry_config.package_categories is None
        or registry_config.container_categories is None
        or registry_config.category_assignments is None
    ):
        raise SemanticV3TransformError(
            "semantic-v3 publication requires frozen package/container registries and "
            "reviewed assignments"
        )
    package_registry, package_sha = _load_frozen_model(
        registry_config.package_categories, CategoryRegistry
    )
    container_registry, container_sha = _load_frozen_model(
        registry_config.container_categories, CategoryRegistry
    )
    assignments, _ = _load_frozen_model(registry_config.category_assignments, CategoryAssignments)
    if package_registry.registryKind != "package" or container_registry.registryKind != "container":
        raise SemanticV3TransformError("category registry kind differs from its configured role")
    if assignments.sourceInventorySha256 != _source_inventory_sha256(config):
        raise SemanticV3TransformError("category assignments target a different source inventory")
    if (
        assignments.packageRegistrySha256 != package_sha
        or assignments.containerRegistrySha256 != container_sha
    ):
        raise SemanticV3TransformError("category assignments target different registry snapshots")
    package_tokens = {row.categoryToken: row.applicationCode for row in package_registry.entries}
    container_tokens = {
        row.categoryToken: row.applicationCode for row in container_registry.entries
    }
    package_map = {
        (row.sourceTypeText, row.sourceTypeCode): row.categoryToken
        for row in assignments.packageAssignments
    }
    container_map = {
        (row.sourceTypeDescription, row.sourceTypeCode): row.categoryToken
        for row in assignments.containerAssignments
    }
    package_assignment_counts = {
        (row.sourceTypeText, row.sourceTypeCode): row.occurrences
        for row in assignments.packageAssignments
    }
    container_assignment_counts = {
        (row.sourceTypeDescription, row.sourceTypeCode): row.occurrences
        for row in assignments.containerAssignments
    }
    if package_assignment_counts != dict(package_counts):
        raise SemanticV3TransformError(
            "package assignments do not exactly cover observed source keys"
        )
    if container_assignment_counts != dict(container_counts):
        raise SemanticV3TransformError(
            "container assignments do not exactly cover observed source keys"
        )
    assigned_package_tokens = {value for value in package_map.values() if value is not None}
    assigned_container_tokens = {value for value in container_map.values() if value is not None}
    if not assigned_package_tokens.issubset(package_tokens):
        raise SemanticV3TransformError("package assignment references an absent category token")
    if not assigned_container_tokens.issubset(container_tokens):
        raise SemanticV3TransformError("container assignment references an absent category token")
    return package_map, container_map, package_tokens, container_tokens


def _validate_relation_assignments(
    config: SemanticV3TransformConfig,
    source_rows: tuple[_SourceRow, ...],
    relations: tuple[_RelationAudit, ...],
) -> dict[tuple[str, int], str]:
    configured = config.registries.relation_assignments
    if configured is None:
        raise SemanticV3TransformError(
            "semantic-v3 publication requires frozen reviewed cargo relation assignments"
        )
    assignments, _ = _load_frozen_model(configured, CargoRelationAssignments)
    if assignments.sourceInventorySha256 != _source_inventory_sha256(config):
        raise SemanticV3TransformError(
            "cargo relation assignments target a different source inventory"
        )
    relation_by_key = {(row.document_id, row.goods_index): row for row in relations}
    expected: dict[tuple[str, int], str] = {}
    goods_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for source_row in source_rows:
        document_id = cast(str, source_row.value["documentId"])
        patch = cast(dict[str, Any], source_row.value["target"]["documentPatch"])
        for goods_index, goods in enumerate(patch.get("goodsItems", [])):
            relation = relation_by_key[(document_id, goods_index)]
            if _requires_relation_review(goods, relation):
                key = (document_id, goods_index)
                expected[key] = _relation_source_sha256(goods)
                goods_by_key[key] = goods
    actual = {
        (row.documentId, row.goodsIndex): row.sourceRelationSha256
        for row in assignments.assignments
    }
    if actual != expected:
        raise SemanticV3TransformError(
            "cargo relation assignments do not exactly cover reviewed nontrivial relations"
        )
    choices: dict[tuple[str, int], str] = {
        (row.documentId, row.goodsIndex): row.classification for row in assignments.assignments
    }
    for key, classification in choices.items():
        if classification not in _valid_reviewed_relation_choices(goods_by_key[key]):
            raise SemanticV3TransformError(
                f"reviewed cargo relation classification is arithmetically invalid: {key}"
            )
    return choices


def _frozen_input_manifest(config: SemanticV3TransformConfig) -> dict[str, dict[str, Any]]:
    configured = {
        "package_categories": config.registries.package_categories,
        "container_categories": config.registries.container_categories,
        "category_assignments": config.registries.category_assignments,
        "relation_assignments": config.registries.relation_assignments,
    }
    missing = sorted(name for name, value in configured.items() if value is None)
    if missing:
        raise SemanticV3TransformError(
            f"semantic-v3 frozen transform inputs are incomplete: {missing!r}"
        )
    return {
        name: cast(FrozenArtifactInput, value).model_dump(mode="json")
        for name, value in configured.items()
    }


def _project_relative_path(path: Path, project_root: Path, *, context: str) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError as error:
        raise SemanticV3TransformError(
            f"{context} must be contained by the configured training project root"
        ) from error


def _length_distribution(values: list[int]) -> dict[str, int]:
    if not values:
        raise SemanticV3TransformError("token-length audit received no records")
    ordered = sorted(values)

    def percentile(numerator: int, denominator: int) -> int:
        index = ceil((len(ordered) * numerator) / denominator) - 1
        return ordered[max(0, min(index, len(ordered) - 1))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(50, 100),
        "p95": percentile(95, 100),
        "p99": percentile(99, 100),
        "max": ordered[-1],
    }


def _audit_training_token_lengths(
    publication: Any,
    *,
    project_root: Path,
    generated: TrainingConfig,
) -> dict[str, Any] | None:
    audit_source = publication.tokenizer_audit
    if audit_source is None:
        return None
    configured_path = Path(audit_source.path)
    absolute_path = Path(os.path.abspath(configured_path))
    try:
        tokenizer_path = configured_path.resolve(strict=True)
    except OSError as error:
        raise SemanticV3TransformError("tokenizer audit directory does not exist") from error
    if (
        tokenizer_path != absolute_path
        or not tokenizer_path.is_dir()
        or tokenizer_path.is_symlink()
    ):
        raise SemanticV3TransformError("tokenizer audit path must be a canonical directory")
    tokenizer_json = _canonical_file(
        tokenizer_path / "tokenizer.json", context="audit tokenizer.json"
    )
    tokenizer_config = _canonical_file(
        tokenizer_path / "tokenizer_config.json", context="audit tokenizer_config.json"
    )
    if sha256_bytes(read_regular_file_bytes(tokenizer_json)) != (
        audit_source.tokenizer_json_sha256
    ):
        raise SemanticV3TransformError("audit tokenizer.json SHA-256 mismatch")
    if sha256_bytes(read_regular_file_bytes(tokenizer_config)) != (
        audit_source.tokenizer_config_sha256
    ):
        raise SemanticV3TransformError("audit tokenizer_config.json SHA-256 mismatch")

    from transformers import AutoTokenizer

    from document_ocr.training.data import inspect_dataset
    from document_ocr.training.prompting import load_prompt
    from document_ocr.training.tasks import load_training_task

    task = load_training_task(project_root, generated)
    prompt = load_prompt(project_root, generated.prompt, task)
    records_by_split, inspection = inspect_dataset(
        project_root=project_root,
        config=generated,
        prompt=prompt,
        task=task,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=generated.model.tokenizer_use_fast,
    )
    if tokenizer.eos_token_id is None:
        raise SemanticV3TransformError("audit tokenizer has no EOS token")
    source_lengths: list[int] = []
    target_lengths: list[int] = []
    longest_source: tuple[int, str] = (0, "")
    longest_target: tuple[int, str] = (0, "")
    for records in records_by_split.values():
        for offset in range(0, len(records), 16):
            batch = records[offset : offset + 16]
            source_ids = tokenizer(
                [row.input_text for row in batch],
                add_special_tokens=generated.dataset.preprocessing.source_add_special_tokens,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            target_ids = tokenizer(
                text_target=[row.target_text for row in batch],
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            for record, source_tokens, target_tokens in zip(
                batch, source_ids, target_ids, strict=True
            ):
                source_length = len(source_tokens)
                target_length = len(target_tokens) + 1
                source_lengths.append(source_length)
                target_lengths.append(target_length)
                longest_source = max(longest_source, (source_length, record.document_id))
                longest_target = max(longest_target, (target_length, record.document_id))
    source_distribution = _length_distribution(source_lengths)
    target_distribution = _length_distribution(target_lengths)
    if source_distribution["max"] > generated.dataset.preprocessing.max_source_length:
        raise SemanticV3TransformError(
            "exact relation-explicit prompt exceeds configured max_source_length: "
            f"{source_distribution['max']} > "
            f"{generated.dataset.preprocessing.max_source_length}"
        )
    if target_distribution["max"] > generated.dataset.preprocessing.max_target_length:
        raise SemanticV3TransformError(
            "exact relation-explicit target exceeds configured max_target_length: "
            f"{target_distribution['max']} > "
            f"{generated.dataset.preprocessing.max_target_length}"
        )
    return {
        "schema_version": 1,
        "dataset_records": inspection.total_records,
        "split_records": inspection.split_records,
        "prompt_sha256": prompt.sha256,
        "source_tokens": source_distribution,
        "target_tokens_including_terminal_eos": target_distribution,
        "longest_source_document_id": longest_source[1],
        "longest_target_document_id": longest_target[1],
        "configured_max_source_length": generated.dataset.preprocessing.max_source_length,
        "configured_max_target_length": generated.dataset.preprocessing.max_target_length,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_json_sha256": audit_source.tokenizer_json_sha256,
        "tokenizer_config_sha256": audit_source.tokenizer_config_sha256,
    }


def _publish_training_config(
    config: SemanticV3TransformConfig,
    *,
    output_root: Path,
    split_files: list[dict[str, Any]],
    task_constraints_path: Path,
    task_constraints_sha256: str,
    train_records: int,
) -> tuple[Path, bytes, dict[str, Any] | None]:
    publication = config.training_config
    if publication is None:
        raise SemanticV3TransformError("training config publication is not configured")
    configured_root = Path(publication.source.project_root)
    absolute_root = Path(os.path.abspath(configured_root))
    try:
        project_root = configured_root.resolve(strict=True)
    except OSError as error:
        raise SemanticV3TransformError("training project root does not exist") from error
    if (
        project_root != absolute_root
        or not project_root.is_dir()
        or project_root.is_symlink()
    ):
        raise SemanticV3TransformError("training project root must be a canonical directory")
    baseline_path = _canonical_file(
        Path(publication.source.baseline_path), context="training baseline config"
    )
    prompt_path = _canonical_file(
        Path(publication.source.prompt_path), context="relation-explicit training prompt"
    )
    baseline_payload = read_regular_file_bytes(baseline_path)
    prompt_payload = read_regular_file_bytes(prompt_path)
    if sha256_bytes(baseline_payload) != publication.source.baseline_sha256:
        raise SemanticV3TransformError("training baseline SHA-256 mismatch")
    if sha256_bytes(prompt_payload) != publication.source.prompt_sha256:
        raise SemanticV3TransformError("relation-explicit training prompt SHA-256 mismatch")
    baseline = load_training_config(baseline_path)
    if baseline.task != "bill_of_lading_semantic_v2" or baseline.task_constraints is not None:
        raise SemanticV3TransformError(
            "training baseline must be the unconstrained semantic-v2 task"
        )

    split_entries: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for row in split_files:
        kind = row["kind"]
        if not isinstance(kind, str) or not kind.endswith("_records"):
            continue
        split = kind.removesuffix("_records")
        if split not in split_entries:
            raise SemanticV3TransformError(f"unsupported generated dataset split: {split}")
        path = output_root / cast(str, row["path"])
        payload = read_regular_file_bytes(path)
        if sha256_bytes(payload) != row["sha256"]:
            raise SemanticV3TransformError(
                f"generated {split} dataset SHA-256 changed before config publication"
            )
        if len(payload.splitlines()) != row["records"] or any(
            not line for line in payload.splitlines()
        ):
            raise SemanticV3TransformError(
                f"generated {split} dataset record count/content is invalid"
            )
        split_entries[split].append(
            {
                "path": _project_relative_path(
                    path, project_root, context="generated dataset file"
                ),
                "sha256": row["sha256"],
                "records": row["records"],
            }
        )
    if not split_entries["train"] or not split_entries["validation"]:
        raise SemanticV3TransformError(
            "generated training config requires train and validation splits"
        )
    if sum(row["records"] for row in split_entries["train"]) != train_records:
        raise SemanticV3TransformError(
            "generated training split count differs from optimizer-step input"
        )

    value = baseline.model_dump(mode="python")
    value["task"] = "bill_of_lading_relation_explicit_v3"
    value["task_constraints"] = {
        "path": _project_relative_path(
            task_constraints_path,
            project_root,
            context="training task constraints",
        ),
        "sha256": task_constraints_sha256,
    }
    value["run"]["run_id"] = publication.run_id
    value["prompt"]["path"] = _project_relative_path(
        prompt_path, project_root, context="relation-explicit training prompt"
    )
    value["dataset"]["splits"] = split_entries
    value["peft"]["adapter_name"] = publication.adapter_name
    capacity = publication.capacity
    baseline_effective_batch = (
        baseline.optimization.per_device_train_batch_size
        * baseline.optimization.gradient_accumulation_steps
    )
    configured_effective_batch = (
        capacity.per_device_train_batch_size * capacity.gradient_accumulation_steps
    )
    if configured_effective_batch != baseline_effective_batch:
        raise SemanticV3TransformError(
            "relation-explicit capacity must preserve the baseline effective batch size"
        )
    value["dataset"]["preprocessing"]["max_source_length"] = capacity.max_source_length
    value["dataset"]["preprocessing"]["max_target_length"] = capacity.max_target_length
    value["optimization"]["per_device_train_batch_size"] = (
        capacity.per_device_train_batch_size
    )
    value["optimization"]["gradient_accumulation_steps"] = (
        capacity.gradient_accumulation_steps
    )
    value["evaluation"]["per_device_batch_size"] = capacity.per_device_eval_batch_size
    value["evaluation"]["generation_max_length"] = capacity.max_target_length
    optimizer_steps_per_epoch = ceil(
        ceil(train_records / value["optimization"]["per_device_train_batch_size"])
        / value["optimization"]["gradient_accumulation_steps"]
    )
    evaluation_steps = optimizer_steps_per_epoch * publication.evaluation_every_epochs
    value["evaluation"]["strategy"] = "steps"
    value["evaluation"]["steps"] = evaluation_steps
    value["checkpoint"]["strategy"] = "steps"
    value["checkpoint"]["steps"] = evaluation_steps
    tags = value["logging"]["mlflow"]["tags"]
    tags["task"] = "bill-of-lading-relation-explicit-v3"
    tags["dataset"] = config.output.dataset_id
    tags["semantic_instructor"] = "relation-explicit-v3"
    tags["table_view_in_prompt"] = "false"

    try:
        generated = TrainingConfig.model_validate(value, strict=True)
    except ValueError as error:
        raise SemanticV3TransformError(
            f"generated relation-explicit training config is invalid: {error}"
        ) from error
    token_length_audit = _audit_training_token_lengths(
        publication,
        project_root=project_root,
        generated=generated,
    )
    payload = yaml.safe_dump(
        generated.model_dump(mode="json"),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).encode("utf-8")
    output_path = output_root / publication.output_filename
    atomic_publish_bytes(output_path, payload)
    return output_path, payload, token_length_audit


def _transform_dangerous_goods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    for source in rows:
        value: dict[str, Any] = {}
        if "unNumber" in source:
            value["unNumber"] = source["unNumber"]
        hazard_class = source.get("hazardClass")
        if hazard_class is not None:
            try:
                value["hazardCategory"] = _HAZARD_CLASS_TO_CATEGORY[hazard_class]
            except KeyError as error:
                raise SemanticV3TransformError(
                    f"unsupported dangerous-goods hazard class: {hazard_class!r}"
                ) from error
        subsidiary_hazard = source.get("subsidiaryHazard")
        if subsidiary_hazard is not None:
            try:
                value["subsidiaryHazardCategory"] = _HAZARD_CLASS_TO_CATEGORY[
                    subsidiary_hazard
                ]
            except KeyError as error:
                raise SemanticV3TransformError(
                    "subsidiary hazard is not an exact supported IMDG class: "
                    f"{subsidiary_hazard!r}"
                ) from error
        source_flash_point = source.get("flashPoint")
        if source_flash_point is not None:
            flash_point = {"temperature": source_flash_point["temperature"]}
            packing_group = source_flash_point.get("packingGroup")
            if packing_group is not None:
                try:
                    flash_point["packingGroupCategory"] = _PACKING_GROUP_TO_CATEGORY[
                        packing_group
                    ]
                except KeyError as error:
                    raise SemanticV3TransformError(
                        f"unsupported dangerous-goods packing group: {packing_group!r}"
                    ) from error
            value["flashPoint"] = flash_point
        if not value:
            raise SemanticV3TransformError(
                "dangerous-goods transform produced an empty semantic record"
            )
        transformed.append(value)
    return transformed


def _relation_explicit_task_constraints(
    config: SemanticV3TransformConfig,
    *,
    package_tokens: set[str],
    container_tokens: set[str],
) -> RelationExplicitTaskConstraints:
    package_registry = config.registries.package_categories
    container_registry = config.registries.container_categories
    if package_registry is None or container_registry is None:
        raise SemanticV3TransformError(
            "task constraints require frozen package and container registries"
        )
    task = get_training_task("bill_of_lading_relation_explicit_v3")
    return RelationExplicitTaskConstraints.model_validate(
        {
            "schemaVersion": 1,
            "task": task.name,
            "basePromptSchemaSha256": task.base_prompt_schema_sha256(),
            "targetSchemaSha256": _target_schema_sha256(),
            "packageRegistrySha256": package_registry.sha256,
            "containerRegistrySha256": container_registry.sha256,
            "packageCategoryTokens": tuple(sorted(package_tokens)),
            "containerCategoryTokens": tuple(sorted(container_tokens)),
        },
        strict=True,
    )


def _transform_target(
    target: dict[str, Any],
    *,
    package_map: dict[tuple[str | None, str | None], str | None],
    container_map: dict[tuple[str | None, str | None], str | None],
    relation_choices: dict[int, str],
) -> dict[str, Any]:
    source_patch = cast(dict[str, Any], target["documentPatch"])
    patch = {
        key: value for key, value in source_patch.items() if key not in {"containers", "goodsItems"}
    }
    transformed_containers: list[dict[str, Any]] = []
    for container in source_patch.get("containers", []):
        value = {key: item for key, item in container.items() if key != "typeCode"}
        source_key = (container.get("typeDescription"), container.get("typeCode"))
        if source_key != (None, None):
            category = container_map[source_key]
            if category is not None:
                value["typeCategory"] = category
            elif value.get("typeDescription") is None and source_key[1] is not None:
                value["typeDescription"] = source_key[1]
        transformed_containers.append(value)
    if transformed_containers:
        patch["containers"] = transformed_containers

    cargo_groups: list[dict[str, Any]] = []
    cargo_packages: list[dict[str, Any]] = []
    allocation_groups: list[dict[str, Any]] = []
    package_counter = 0
    for goods_index, goods in enumerate(source_patch.get("goodsItems", [])):
        group_id = f"g{goods_index + 1}"
        group = {
            "groupId": group_id,
            **{
                key: value
                for key, value in goods.items()
                if key not in {"packages", "containerAllocations", "dangerousGoods"}
            },
        }
        source_dangerous_goods = goods.get("dangerousGoods", [])
        if source_dangerous_goods:
            group["dangerousGoods"] = _transform_dangerous_goods(source_dangerous_goods)
        cargo_groups.append(group)
        group_package_ids: list[str] = []
        package_index_by_source_index: dict[int, str] = {}
        for source_package_index, package in enumerate(goods.get("packages", [])):
            package_counter += 1
            package_id = f"p{package_counter}"
            group_package_ids.append(package_id)
            package_index_by_source_index[source_package_index] = package_id
            fact: dict[str, Any] = {"groupId": group_id, "packageId": package_id}
            if "quantity" in package:
                fact["quantity"] = package["quantity"]
            source_key = (package.get("type"), package.get("typeCode"))
            if source_key != (None, None):
                category = package_map[source_key]
                if category is not None:
                    fact["typeCategory"] = category
                elif package.get("type") is not None:
                    fact["typeDescription"] = package["type"]
            cargo_packages.append(fact)
        allocations = goods.get("containerAllocations", [])
        if allocations:
            audit = _relation_audit("doc_" + "0" * 64, goods_index, goods)
            if audit.blocking_reason is not None:
                raise SemanticV3TransformError(
                    f"blocking cargo relation reached transform: goodsItems[{goods_index}]"
                )
            classification = relation_choices.get(goods_index, audit.classification)
            transformed_allocations = allocations
            if classification == "one_to_one_package_allocations":
                coverage = "one_to_one_package_allocations"
                package_ids = group_package_ids
                transformed_allocations = [
                    {**allocation, "packageId": package_id}
                    for allocation, package_id in zip(allocations, group_package_ids, strict=True)
                ]
            elif classification == "single_package_level":
                coverage = "single_package_level"
                package_ids = [package_index_by_source_index[audit.matched_package_indexes[0]]]
            elif classification == "all_package_levels_combined":
                coverage = "all_package_levels_combined"
                package_ids = group_package_ids
            elif classification == "unlinked_package_quantities":
                coverage = "unlinked_package_quantities"
                package_ids = []
            elif classification == "container_membership_only":
                coverage = "container_membership_only"
                package_ids = []
            else:
                raise SemanticV3TransformError(
                    f"unsupported nonblocking relation classification: {classification}"
                )
            allocation_groups.append(
                {
                    "groupId": group_id,
                    "coverage": coverage,
                    "packageIds": package_ids,
                    "allocations": transformed_allocations,
                }
            )
    if cargo_groups:
        patch["cargoGroups"] = cargo_groups
    if cargo_packages:
        patch["cargoPackages"] = cargo_packages
    if allocation_groups:
        patch["cargoAllocationGroups"] = allocation_groups
    candidate = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": patch,
    }
    try:
        parsed = BillOfLadingRelationExplicitLabel.model_validate_json(
            canonical_json_bytes(candidate), strict=True
        )
    except ValueError as error:
        raise SemanticV3TransformError(f"semantic-v3 target validation failed: {error}") from error
    return parsed.canonical_target()


def publish_semantic_v3_dataset(config: SemanticV3TransformConfig) -> Path:
    """Clone source rows, transform targets, exclude only predeclared relation blockers."""

    source_rows = _load_source_rows(config)
    (
        package_counts,
        container_counts,
        relations,
        _,
        _,
        _,
    ) = _inventories(source_rows)
    actual_exclusions = {
        row.document_id: row.blocking_reason for row in relations if row.blocking_reason is not None
    }
    allowed_exclusions = {row.document_id: row.reason for row in config.allowed_relation_exclusions}
    if actual_exclusions != allowed_exclusions:
        raise SemanticV3TransformError(
            "actual relation blockers differ from allowed_relation_exclusions"
        )
    package_map, container_map, package_codes, container_codes = _category_inputs(
        config, package_counts, container_counts
    )
    relation_choices = _validate_relation_assignments(config, source_rows, relations)
    relation_choices_by_document: dict[str, dict[int, str]] = defaultdict(dict)
    for (document_id, goods_index), classification in relation_choices.items():
        relation_choices_by_document[document_id][goods_index] = classification
    output_root = _canonical_directory(Path(config.output.dataset_root))
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        document_id = cast(str, source_row.value["documentId"])
        if document_id in actual_exclusions:
            excluded_rows.append(
                {
                    "documentId": document_id,
                    "reason": actual_exclusions[document_id],
                    "sourceId": source_row.source.id,
                    "sourceRowNumber": source_row.row_number,
                    "split": source_row.source.split,
                }
            )
            continue
        if "semanticV3Transform" in source_row.value:
            raise SemanticV3TransformError("source row already contains semanticV3Transform")
        source_target = cast(dict[str, Any], source_row.value["target"])
        v3_target = _transform_target(
            source_target,
            package_map=package_map,
            container_map=container_map,
            relation_choices=relation_choices_by_document.get(document_id, {}),
        )
        cloned = dict(source_row.value)
        cloned["target"] = v3_target
        cloned["semanticV3Transform"] = {
            "sourceTargetCanonicalSha256": sha256_bytes(canonical_json_bytes(source_target)),
            "targetCanonicalSha256": sha256_bytes(canonical_json_bytes(v3_target)),
            "targetSchemaSha256": _target_schema_sha256(),
            "transform": config.transform,
            "transformContractSha256": _transform_contract_sha256(config),
        }
        by_split[source_row.source.split].append(cloned)
        lineage_rows.append(
            {
                "documentId": document_id,
                "sourceFileSha256": source_row.source.sha256,
                "sourceId": source_row.source.id,
                "sourceRowNumber": source_row.row_number,
                "sourceTargetCanonicalSha256": cloned["semanticV3Transform"][
                    "sourceTargetCanonicalSha256"
                ],
                "split": source_row.source.split,
                "targetCanonicalSha256": cloned["semanticV3Transform"]["targetCanonicalSha256"],
                "targetSchemaSha256": cloned["semanticV3Transform"]["targetSchemaSha256"],
                "transformContractSha256": cloned["semanticV3Transform"]["transformContractSha256"],
            }
        )
    files: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        rows = by_split.get(split, [])
        if not rows:
            continue
        payload = _jsonl(rows)
        path = output_root / f"{split}.jsonl"
        atomic_publish_bytes(path, payload)
        files.append(
            {
                "bytes": len(payload),
                "kind": f"{split}_records",
                "path": path.name,
                "records": len(rows),
                "sha256": sha256_bytes(payload),
            }
        )
    for name, kind, rows in (
        ("excluded.jsonl", "declared_exclusions", excluded_rows),
        ("lineage.jsonl", "lineage", lineage_rows),
    ):
        payload = _jsonl(rows)
        atomic_publish_bytes(output_root / name, payload)
        files.append(
            {
                "bytes": len(payload),
                "kind": kind,
                "path": name,
                "records": len(rows),
                "sha256": sha256_bytes(payload),
            }
        )
    task_constraints = _relation_explicit_task_constraints(
        config,
        package_tokens={value for value in package_map.values() if value is not None},
        container_tokens={value for value in container_map.values() if value is not None},
    )
    task_constraints_payload = (
        canonical_json_bytes(task_constraints.model_dump(mode="json")) + b"\n"
    )
    task_constraints_path = output_root / "task-constraints.json"
    atomic_publish_bytes(task_constraints_path, task_constraints_payload)
    task_constraints_sha256 = sha256_bytes(task_constraints_payload)
    files.append(
        {
            "bytes": len(task_constraints_payload),
            "kind": "training_task_constraints",
            "path": task_constraints_path.name,
            "records": 1,
            "sha256": task_constraints_sha256,
        }
    )
    training_config_manifest: dict[str, Any] | None = None
    if config.training_config is not None:
        (
            training_config_path,
            training_config_payload,
            token_length_audit,
        ) = _publish_training_config(
            config,
            output_root=output_root,
            split_files=files,
            task_constraints_path=task_constraints_path,
            task_constraints_sha256=task_constraints_sha256,
            train_records=len(by_split.get("train", [])),
        )
        training_config_sha256 = sha256_bytes(training_config_payload)
        files.append(
            {
                "bytes": len(training_config_payload),
                "kind": "training_config",
                "path": training_config_path.name,
                "records": 1,
                "sha256": training_config_sha256,
            }
        )
        token_length_manifest: dict[str, Any] | None = None
        if token_length_audit is not None:
            token_length_path = output_root / "training-token-length-audit.json"
            atomic_publish_json(token_length_path, token_length_audit)
            token_length_payload = read_regular_file_bytes(token_length_path)
            token_length_sha256 = sha256_bytes(token_length_payload)
            files.append(
                {
                    "bytes": len(token_length_payload),
                    "kind": "training_token_length_audit",
                    "path": token_length_path.name,
                    "records": 1,
                    "sha256": token_length_sha256,
                }
            )
            token_length_manifest = {
                "path": token_length_path.name,
                "sha256": token_length_sha256,
                "source_max": token_length_audit["source_tokens"]["max"],
                "target_max": token_length_audit[
                    "target_tokens_including_terminal_eos"
                ]["max"],
            }
        publication = config.training_config
        training_config_manifest = {
            "baseline_path": publication.source.baseline_path,
            "baseline_sha256": publication.source.baseline_sha256,
            "evaluation_every_epochs": publication.evaluation_every_epochs,
            "path": training_config_path.name,
            "prompt_path": publication.source.prompt_path,
            "prompt_sha256": publication.source.prompt_sha256,
            "sha256": training_config_sha256,
            "token_length_audit": token_length_manifest,
        }
    manifest = {
        "schema_version": 1,
        "dataset_id": config.output.dataset_id,
        "target_schema_version": "3.0.0-experimental",
        "source_inventory_sha256": _source_inventory_sha256(config),
        "target_schema_sha256": _target_schema_sha256(),
        "transform_contract_sha256": _transform_contract_sha256(config),
        "source_documents": len(source_rows),
        "published_documents": sum(len(rows) for rows in by_split.values()),
        "declared_exclusions": len(excluded_rows),
        "split_records": {key: len(value) for key, value in sorted(by_split.items())},
        "training_task_constraints": {
            "path": task_constraints_path.name,
            "sha256": task_constraints_sha256,
        },
        "training_config": training_config_manifest,
        "category_projection": {
            "package_token_to_application_code": dict(sorted(package_codes.items())),
            "container_token_to_application_code": dict(sorted(container_codes.items())),
            "unresolved_package_source_keys": sum(
                value is None for value in package_map.values()
            ),
            "unresolved_container_source_keys": sum(
                value is None for value in container_map.values()
            ),
            "hazard_category_to_imdg_class": {
                category: code for code, category in _HAZARD_CLASS_TO_CATEGORY.items()
            },
            "packing_group_category_to_code": {
                category: code for code, category in _PACKING_GROUP_TO_CATEGORY.items()
            },
            "countries_and_localities": "printed_text_preserved_for_downstream_resolution",
        },
        "frozen_transform_inputs": _frozen_input_manifest(config),
        "source_records_immutable": True,
        "files": files,
    }
    manifest_path = output_root / "manifest.json"
    atomic_publish_json(manifest_path, manifest)
    return manifest_path
