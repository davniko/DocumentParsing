"""Experimental relation-explicit Bill-of-Lading supervision contract.

This target is intentionally model-facing rather than form-facing.  Readable,
registry-backed category tokens replace application codes.  Deterministic
document-local identifiers make cargo membership explicit; they are not source
facts and are removed by the downstream projector.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator

from document_ocr.label_schemas.bill_of_lading import (
    ApplicationText,
    BillOfLadingAnnotation,
    BillOfLadingLabel,
    BillOfLadingParties,
    BillOfLadingRoute,
    BillOfLadingTransport,
    CargoMass,
    CargoText,
    FreightTerms,
    GoodsOrigin,
    Mass,
    PackageTypeText,
    SemanticLocation,
    Temperature,
    Volume,
)
from document_ocr.label_schemas.common import (
    ExtractionSourceReference,
    FieldEvidence,
    LabelSchemaModel,
    LabelWarning,
    NonEmptyString,
    RawOcrValueEvidence,
)
from document_ocr.label_schemas.mpci_bill_of_lading import ContainerIdentifier, HsCode

CategoryToken = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$", max_length=96),
]
GroupId = Annotated[str, StringConstraints(pattern=r"^g[1-9][0-9]*$")]
PackageId = Annotated[str, StringConstraints(pattern=r"^p[1-9][0-9]*$")]
NonNegativeQuantity = Annotated[int, Field(ge=0)]
HazardCategory = Literal[
    "EXPLOSIVES",
    "GASES",
    "FLAMMABLE_LIQUIDS",
    "FLAMMABLE_SOLIDS",
    "OXIDIZING_SUBSTANCES_AND_ORGANIC_PEROXIDES",
    "TOXIC_AND_INFECTIOUS_SUBSTANCES",
    "RADIOACTIVE_MATERIAL",
    "CORROSIVE_SUBSTANCES",
    "MISCELLANEOUS_DANGEROUS_SUBSTANCES_AND_ARTICLES",
]
PackingGroupCategory = Literal["HIGH_DANGER", "MEDIUM_DANGER", "LOW_DANGER"]
AllocationCoverage = Literal[
    "one_to_one_package_allocations",
    "single_package_level",
    "all_package_levels_combined",
    "unlinked_package_quantities",
    "container_membership_only",
]

_GROUP_INDEX = re.compile(r"^g([1-9][0-9]*)$")
_PACKAGE_INDEX = re.compile(r"^p([1-9][0-9]*)$")


def _sequential(values: tuple[str, ...], pattern: re.Pattern[str], field_name: str) -> None:
    expected = tuple(range(1, len(values) + 1))
    actual = tuple(int(pattern.fullmatch(value).group(1)) for value in values)  # type: ignore[union-attr]
    if actual != expected:
        raise ValueError(f"{field_name} must be contiguous and source ordered")


class RelationExplicitContainer(LabelSchemaModel):
    containerNumber: ContainerIdentifier
    typeDescription: ApplicationText | None = None
    typeCategory: CategoryToken | None = None
    verifiedGrossMass: Mass | None = None
    sealNumbers: tuple[ApplicationText, ...] | None = Field(default=None, min_length=1)
    temperatureSetpoint: Temperature | None = None

    @model_validator(mode="after")
    def seals_are_unique(self) -> RelationExplicitContainer:
        if self.sealNumbers is not None and len(self.sealNumbers) != len(set(self.sealNumbers)):
            raise ValueError("sealNumbers values must be unique and source ordered")
        return self


class RelationExplicitDangerousGoodsFlashPoint(LabelSchemaModel):
    temperature: Temperature
    packingGroupCategory: PackingGroupCategory | None = None


class RelationExplicitDangerousGoods(LabelSchemaModel):
    unNumber: Annotated[str, StringConstraints(pattern=r"^[0-9]{4}$")] | None = None
    hazardCategory: HazardCategory | None = None
    subsidiaryHazardCategory: HazardCategory | None = None
    flashPoint: RelationExplicitDangerousGoodsFlashPoint | None = None

    @model_validator(mode="after")
    def contains_dangerous_goods(self) -> RelationExplicitDangerousGoods:
        if not self.model_dump(mode="python", exclude_none=True):
            raise ValueError("dangerous goods record must contain supported evidence")
        return self


class CargoGroup(LabelSchemaModel):
    groupId: GroupId
    description: CargoText | None = None
    additionalInformation: tuple[CargoText, ...] | None = Field(default=None, min_length=1)
    grossWeight: CargoMass | None = None
    netWeight: CargoMass | None = None
    volume: Volume | None = None
    marksAndNumbers: tuple[CargoText, ...] | None = Field(default=None, min_length=1)
    hsCodes: tuple[HsCode, ...] | None = Field(default=None, min_length=1)
    handlingInstructions: tuple[CargoText, ...] | None = Field(default=None, min_length=1)
    dangerousGoods: tuple[RelationExplicitDangerousGoods, ...] | None = Field(
        default=None, min_length=1
    )
    origin: GoodsOrigin | None = None

    @model_validator(mode="after")
    def repeated_values_are_unique(self) -> CargoGroup:
        for name in ("additionalInformation", "marksAndNumbers", "hsCodes", "handlingInstructions"):
            sequence = getattr(self, name)
            if sequence is not None and len(sequence) != len(set(sequence)):
                raise ValueError(f"{name} values must be unique and source ordered")
        return self


class CargoPackageFact(LabelSchemaModel):
    packageId: PackageId
    groupId: GroupId
    quantity: NonNegativeQuantity | None = None
    typeCategory: CategoryToken | None = None
    typeDescription: PackageTypeText | None = None

    @model_validator(mode="after")
    def contains_a_package_fact(self) -> CargoPackageFact:
        if self.typeCategory is not None and self.typeDescription is not None:
            raise ValueError(
                "cargo package typeDescription is a fallback and cannot accompany typeCategory"
            )
        if (
            self.quantity is None
            and self.typeCategory is None
            and self.typeDescription is None
        ):
            raise ValueError(
                "cargo package must contain a quantity, readable category, or printed fallback"
            )
        return self


class CargoContainerAllocation(LabelSchemaModel):
    containerNumber: ContainerIdentifier
    packageQuantity: NonNegativeQuantity | None = None
    packageId: PackageId | None = None


class CargoAllocationGroup(LabelSchemaModel):
    groupId: GroupId
    coverage: AllocationCoverage
    packageIds: tuple[PackageId, ...] = ()
    allocations: tuple[CargoContainerAllocation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def coverage_shape_is_consistent(self) -> CargoAllocationGroup:
        if len(self.packageIds) != len(set(self.packageIds)):
            raise ValueError("allocation packageIds must be unique and source ordered")
        package_indexes = tuple(
            int(_PACKAGE_INDEX.fullmatch(package_id).group(1))  # type: ignore[union-attr]
            for package_id in self.packageIds
        )
        if package_indexes != tuple(sorted(package_indexes)):
            raise ValueError("allocation packageIds must be source ordered")
        identifiers = tuple(row.containerNumber for row in self.allocations)
        if (
            self.coverage != "one_to_one_package_allocations"
            and len(identifiers) != len(set(identifiers))
        ):
            raise ValueError("allocation container numbers must be unique and source ordered")
        quantities = tuple(row.packageQuantity for row in self.allocations)
        allocation_package_ids = tuple(row.packageId for row in self.allocations)
        if self.coverage == "one_to_one_package_allocations":
            if (
                len(self.packageIds) != len(self.allocations)
                or tuple(self.packageIds) != allocation_package_ids
                or any(value is None for value in quantities)
            ):
                raise ValueError(
                    "one-to-one coverage requires one ordered package reference per allocation"
                )
        elif any(value is not None for value in allocation_package_ids):
            raise ValueError("allocation-level packageId is valid only for one-to-one coverage")
        elif self.coverage == "single_package_level":
            if len(self.packageIds) != 1 or any(value is None for value in quantities):
                raise ValueError("single-package coverage requires one package and all quantities")
        elif self.coverage == "all_package_levels_combined":
            if len(self.packageIds) < 2 or any(value is None for value in quantities):
                raise ValueError("combined coverage requires multiple packages and all quantities")
        elif self.coverage == "unlinked_package_quantities":
            if self.packageIds or any(value is None for value in quantities):
                raise ValueError("unlinked quantities require no packages and all quantities")
        elif self.packageIds or any(value is not None for value in quantities):
            raise ValueError("container-only membership cannot carry packages or quantities")
        return self


class RelationExplicitDocumentPatch(LabelSchemaModel):
    billOfLadingNumber: ApplicationText | None = None
    originalBillOfLadingNumber: ApplicationText | None = None
    masterBillOfLadingNumber: ApplicationText | None = None
    issueDate: date | None = None
    shippedOnBoardDate: date | None = None
    negotiability: Literal["negotiable", "non_negotiable"] | None = None
    placeOfIssue: SemanticLocation | None = None
    route: BillOfLadingRoute | None = None
    transport: BillOfLadingTransport | None = None
    freight: FreightTerms | None = None
    parties: BillOfLadingParties | None = None
    containers: tuple[RelationExplicitContainer, ...] | None = Field(default=None, min_length=1)
    forwardingAndExportReferences: tuple[ApplicationText, ...] | None = Field(
        default=None, min_length=1
    )
    cargoGroups: tuple[CargoGroup, ...] | None = Field(default=None, min_length=1)
    cargoPackages: tuple[CargoPackageFact, ...] | None = Field(default=None, min_length=1)
    cargoAllocationGroups: tuple[CargoAllocationGroup, ...] | None = Field(
        default=None, min_length=1
    )

    @model_validator(mode="after")
    def relations_are_complete_and_source_ordered(self) -> RelationExplicitDocumentPatch:
        if not self.model_dump(mode="python", exclude_none=True):
            raise ValueError("documentPatch must contain at least one supported fact")
        if self.forwardingAndExportReferences is not None and len(
            self.forwardingAndExportReferences
        ) != len(set(self.forwardingAndExportReferences)):
            raise ValueError("forwarding/export references must be unique and source ordered")
        containers = self.containers or ()
        container_ids = tuple(row.containerNumber for row in containers)
        if len(container_ids) != len(set(container_ids)):
            raise ValueError("container numbers must be unique and source ordered")
        groups = self.cargoGroups or ()
        group_ids = tuple(row.groupId for row in groups)
        _sequential(group_ids, _GROUP_INDEX, "cargo group IDs")
        packages = self.cargoPackages or ()
        package_ids = tuple(row.packageId for row in packages)
        _sequential(package_ids, _PACKAGE_INDEX, "cargo package IDs")
        if any(row.groupId not in set(group_ids) for row in packages):
            raise ValueError("cargo package references an absent group")
        allocation_groups = self.cargoAllocationGroups or ()
        allocation_group_ids = tuple(row.groupId for row in allocation_groups)
        if len(allocation_group_ids) != len(set(allocation_group_ids)):
            raise ValueError("at most one allocation group is allowed per cargo group")
        allocation_group_indexes = tuple(
            int(_GROUP_INDEX.fullmatch(group_id).group(1))  # type: ignore[union-attr]
            for group_id in allocation_group_ids
        )
        if allocation_group_indexes != tuple(sorted(allocation_group_indexes)):
            raise ValueError("cargo allocation groups must be source ordered")
        package_by_id = {row.packageId: row for row in packages}
        related_group_ids = {row.groupId for row in packages} | {
            row.groupId for row in allocation_groups
        }
        for group in groups:
            direct = group.model_dump(mode="python", exclude_none=True, exclude={"groupId"})
            if not direct and group.groupId not in related_group_ids:
                raise ValueError("cargo group must contain a fact or a package/allocation relation")
        for allocation in allocation_groups:
            if allocation.groupId not in set(group_ids):
                raise ValueError("cargo allocation references an absent group")
            for package_id in allocation.packageIds:
                package = package_by_id.get(package_id)
                if package is None or package.groupId != allocation.groupId:
                    raise ValueError("cargo allocation references an absent or foreign package")
            if any(row.containerNumber not in set(container_ids) for row in allocation.allocations):
                raise ValueError("cargo allocation references an absent container")
            quantities = tuple(
                row.packageQuantity
                for row in allocation.allocations
                if row.packageQuantity is not None
            )
            if allocation.coverage in {
                "single_package_level",
                "all_package_levels_combined",
            }:
                package_quantities = tuple(
                    package_by_id[package_id].quantity for package_id in allocation.packageIds
                )
                if any(value is None for value in package_quantities):
                    raise ValueError("covered package levels must carry quantities")
                if sum(quantities) != sum(
                    value for value in package_quantities if value is not None
                ):
                    raise ValueError("allocation total differs from the covered package levels")
            elif allocation.coverage == "one_to_one_package_allocations":
                for placement in allocation.allocations:
                    if placement.packageId is None:
                        raise ValueError("one-to-one allocation is missing packageId")
                    package_quantity = package_by_id[placement.packageId].quantity
                    if package_quantity is None or package_quantity != placement.packageQuantity:
                        raise ValueError(
                            "one-to-one allocation quantity differs from its package fact"
                        )
        return self


class BillOfLadingRelationExplicitLabel(LabelSchemaModel):
    schemaVersion: Literal["3.0.0-experimental"]
    documentPatch: RelationExplicitDocumentPatch

    def canonical_target(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class CargoRelationEvidence(LabelSchemaModel):
    """Raw-OCR anchors supporting one emitted cargo-allocation relationship."""

    groupId: GroupId
    coverage: AllocationCoverage
    relationshipBasis: Literal[
        "explicit_linkage",
        "row_alignment",
        "column_alignment",
        "quantity_reconciliation",
        "container_membership",
    ]
    rawOcrEvidence: tuple[RawOcrValueEvidence, ...] = Field(min_length=1)
    pdfUse: Literal["not_used", "grouping_only"] = "not_used"
    note: NonEmptyString

    @model_validator(mode="after")
    def evidence_is_source_ordered(self) -> CargoRelationEvidence:
        keys = tuple(
            (row.pageNumber, row.rawValue, row.ocrExcerpt) for row in self.rawOcrEvidence
        )
        if len(keys) != len(set(keys)):
            raise ValueError("cargo relation evidence entries must be unique")
        if tuple(row.pageNumber for row in self.rawOcrEvidence) != tuple(
            sorted(row.pageNumber for row in self.rawOcrEvidence)
        ):
            raise ValueError("cargo relation evidence must be source ordered")
        return self


_HAZARD_CLASS_TO_CATEGORY: dict[str, HazardCategory] = {
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
_PACKING_GROUP_TO_CATEGORY: dict[str, PackingGroupCategory] = {
    "I": "HIGH_DANGER",
    "II": "MEDIUM_DANGER",
    "III": "LOW_DANGER",
}
_CATEGORY_TO_HAZARD_CLASS = {
    category: hazard_class for hazard_class, category in _HAZARD_CLASS_TO_CATEGORY.items()
}
_CATEGORY_TO_PACKING_GROUP = {
    category: packing_group
    for packing_group, category in _PACKING_GROUP_TO_CATEGORY.items()
}


def _normal_dangerous_goods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normal_rows: list[dict[str, Any]] = []
    for row in rows:
        normal: dict[str, Any] = {}
        if "unNumber" in row:
            normal["unNumber"] = row["unNumber"]
        hazard = row.get("hazardCategory")
        if hazard is not None:
            normal["hazardClass"] = _CATEGORY_TO_HAZARD_CLASS[hazard]
        subsidiary = row.get("subsidiaryHazardCategory")
        if subsidiary is not None:
            normal["subsidiaryHazard"] = _CATEGORY_TO_HAZARD_CLASS[subsidiary]
        flash = row.get("flashPoint")
        if flash is not None:
            normal_flash: dict[str, Any] = {"temperature": flash["temperature"]}
            packing_group = flash.get("packingGroupCategory")
            if packing_group is not None:
                normal_flash["packingGroup"] = _CATEGORY_TO_PACKING_GROUP[packing_group]
            normal["flashPoint"] = normal_flash
        normal_rows.append(normal)
    return normal_rows


def project_relation_to_normal(
    relation_label: BillOfLadingRelationExplicitLabel,
) -> BillOfLadingLabel:
    """Project the single relation-explicit source of truth into semantic-v2.

    The model-facing relation contract carries deterministic ``gN``/``pN``
    identifiers and readable cargo relationships.  The normal view contains no
    information that is not already present there, so generating both views is
    unnecessary and creates a consistency failure surface.  Registry categories
    remain downstream and are intentionally rejected here until the frozen
    category projection is applied.
    """

    relation_patch = relation_label.model_dump(mode="json", exclude_none=True)[
        "documentPatch"
    ]
    normal_patch = {
        key: value
        for key, value in relation_patch.items()
        if key
        not in {
            "containers",
            "cargoGroups",
            "cargoPackages",
            "cargoAllocationGroups",
        }
    }

    relation_containers = relation_patch.get("containers", [])
    if relation_containers:
        normal_containers: list[dict[str, Any]] = []
        for row in relation_containers:
            if row.get("typeCategory") is not None:
                raise ValueError(
                    "relation-to-normal projection requires printed container typeDescription; "
                    "category mapping is downstream"
                )
            normal_containers.append(
                {key: value for key, value in row.items() if key != "typeCategory"}
            )
        normal_patch["containers"] = normal_containers

    packages_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in relation_patch.get("cargoPackages", []):
        if row.get("typeCategory") is not None:
            raise ValueError(
                "relation-to-normal projection requires printed package typeDescription; "
                "category mapping is downstream"
            )
        package: dict[str, Any] = {}
        if "quantity" in row:
            package["quantity"] = row["quantity"]
        if "typeDescription" in row:
            package["type"] = row["typeDescription"]
        packages_by_group.setdefault(row["groupId"], []).append(package)

    allocations_by_group = {
        row["groupId"]: row
        for row in relation_patch.get("cargoAllocationGroups", [])
    }
    relation_groups = relation_patch.get("cargoGroups", [])
    if relation_groups:
        normal_goods: list[dict[str, Any]] = []
        for group in relation_groups:
            group_id = group["groupId"]
            goods = {
                key: value
                for key, value in group.items()
                if key not in {"groupId", "dangerousGoods"}
            }
            if group.get("dangerousGoods"):
                goods["dangerousGoods"] = _normal_dangerous_goods(
                    group["dangerousGoods"]
                )
            packages = packages_by_group.get(group_id)
            if packages:
                goods["packages"] = packages
            allocation_group = allocations_by_group.get(group_id)
            if allocation_group is not None:
                goods["containerAllocations"] = [
                    {
                        key: value
                        for key, value in allocation.items()
                        if key != "packageId"
                    }
                    for allocation in allocation_group["allocations"]
                ]
            normal_goods.append(goods)
        normal_patch["goodsItems"] = normal_goods

    normal_label = BillOfLadingLabel.model_validate_json(
        json.dumps(
            {"schemaVersion": "2.0.0", "documentPatch": normal_patch},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        strict=True,
    )
    validate_dual_cargo_consistency(normal_label, relation_label)
    return normal_label


def _expected_relation_dangerous_goods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for row in rows:
        transformed: dict[str, Any] = {}
        if "unNumber" in row:
            transformed["unNumber"] = row["unNumber"]
        hazard = row.get("hazardClass")
        if hazard is not None:
            try:
                transformed["hazardCategory"] = _HAZARD_CLASS_TO_CATEGORY[hazard]
            except KeyError as error:
                raise ValueError(
                    f"normal label contains an unsupported hazard class: {hazard!r}"
                ) from error
        subsidiary = row.get("subsidiaryHazard")
        if subsidiary is not None:
            try:
                transformed["subsidiaryHazardCategory"] = _HAZARD_CLASS_TO_CATEGORY[
                    subsidiary
                ]
            except KeyError as error:
                raise ValueError(
                    "normal label contains an unsupported subsidiary hazard: "
                    f"{subsidiary!r}"
                ) from error
        flash = row.get("flashPoint")
        if flash is not None:
            transformed_flash: dict[str, Any] = {"temperature": flash["temperature"]}
            packing_group = flash.get("packingGroup")
            if packing_group is not None:
                transformed_flash["packingGroupCategory"] = _PACKING_GROUP_TO_CATEGORY[
                    packing_group
                ]
            transformed["flashPoint"] = transformed_flash
        expected.append(transformed)
    return expected


def validate_dual_cargo_consistency(
    normal_label: BillOfLadingLabel,
    relation_label: BillOfLadingRelationExplicitLabel,
) -> None:
    """Require the normal and relational views to describe the same OCR facts."""

    normal_patch = normal_label.canonical_target()["documentPatch"]
    relation_patch = relation_label.canonical_target()["documentPatch"]
    normal_header = {
        key: value
        for key, value in normal_patch.items()
        if key not in {"containers", "goodsItems"}
    }
    relation_header = {
        key: value
        for key, value in relation_patch.items()
        if key
        not in {
            "containers",
            "cargoGroups",
            "cargoPackages",
            "cargoAllocationGroups",
        }
    }
    if normal_header != relation_header:
        raise ValueError("normal and relation-explicit non-cargo facts differ")

    normal_containers = normal_patch.get("containers", [])
    relation_containers = relation_patch.get("containers", [])
    if len(normal_containers) != len(relation_containers):
        raise ValueError("normal and relation-explicit container counts differ")
    for normal, relation in zip(normal_containers, relation_containers, strict=True):
        expected = {key: value for key, value in normal.items() if key != "typeCode"}
        actual = {key: value for key, value in relation.items() if key != "typeCategory"}
        if expected != actual:
            raise ValueError("normal and relation-explicit container facts differ")
        if relation.get("typeCategory") is not None and not (
            normal.get("typeDescription") or normal.get("typeCode")
        ):
            raise ValueError("container category has no normal-view source fact")

    normal_goods = normal_patch.get("goodsItems", [])
    relation_groups = relation_patch.get("cargoGroups", [])
    if len(normal_goods) != len(relation_groups):
        raise ValueError("normal goods and relation-explicit cargo-group counts differ")
    expected_packages: list[tuple[str, dict[str, Any]]] = []
    for index, (goods, group) in enumerate(
        zip(normal_goods, relation_groups, strict=True), start=1
    ):
        group_id = f"g{index}"
        expected_group = {
            "groupId": group_id,
            **{
                key: value
                for key, value in goods.items()
                if key not in {"packages", "containerAllocations", "dangerousGoods"}
            },
        }
        dangerous = goods.get("dangerousGoods", [])
        if dangerous:
            expected_group["dangerousGoods"] = _expected_relation_dangerous_goods(
                dangerous
            )
        if group != expected_group:
            raise ValueError(f"normal goods item and relation cargo group {group_id} differ")
        expected_packages.extend((group_id, row) for row in goods.get("packages", []))

    relation_packages = relation_patch.get("cargoPackages", [])
    if len(expected_packages) != len(relation_packages):
        raise ValueError("normal and relation-explicit package counts differ")
    for index, ((group_id, normal), relation) in enumerate(
        zip(expected_packages, relation_packages, strict=True), start=1
    ):
        if relation["packageId"] != f"p{index}" or relation["groupId"] != group_id:
            raise ValueError("relation-explicit package identity/order differs from normal goods")
        if relation.get("quantity") != normal.get("quantity"):
            raise ValueError("normal and relation-explicit package quantities differ")
        raw_type_present = normal.get("type") is not None or normal.get("typeCode") is not None
        relation_type_present = (
            relation.get("typeDescription") is not None
            or relation.get("typeCategory") is not None
        )
        if raw_type_present != relation_type_present:
            raise ValueError("normal and relation-explicit package type presence differs")
        if relation.get("typeDescription") is not None and relation.get(
            "typeDescription"
        ) != normal.get("type"):
            raise ValueError("relation package fallback differs from printed normal type")

    allocations_by_group = {
        row["groupId"]: row for row in relation_patch.get("cargoAllocationGroups", [])
    }
    for index, goods in enumerate(normal_goods, start=1):
        group_id = f"g{index}"
        normal_allocations = goods.get("containerAllocations", [])
        relation_group = allocations_by_group.get(group_id)
        if not normal_allocations:
            if relation_group is not None:
                raise ValueError("relation allocation exists without a normal allocation")
            continue
        if relation_group is None:
            raise ValueError("normal allocation is absent from the relation-explicit view")
        relation_allocations = [
            {key: value for key, value in row.items() if key != "packageId"}
            for row in relation_group["allocations"]
        ]
        if relation_allocations != normal_allocations:
            raise ValueError("normal and relation-explicit container allocations differ")


class BillOfLadingDualCargoAnnotation(LabelSchemaModel):
    """One auditable label artifact carrying normal and relation-explicit views."""

    annotationSchemaVersion: Literal["3.0.0-experimental"]
    taskType: Literal["bill_of_lading_dual_cargo_kie"]
    documentType: Literal["bill_of_lading", "sea_waybill"]
    source: ExtractionSourceReference
    normalLabel: BillOfLadingLabel
    relationExplicitLabel: BillOfLadingRelationExplicitLabel
    evidence: tuple[FieldEvidence, ...] = Field(min_length=1)
    relationEvidence: tuple[CargoRelationEvidence, ...] = ()
    warnings: tuple[LabelWarning, ...] = ()
    reviewStatus: Literal["candidate", "validated", "needs_review", "rejected"]
    reviewNotes: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def views_and_evidence_are_consistent(self) -> BillOfLadingDualCargoAnnotation:
        BillOfLadingAnnotation.model_validate(
            {
                "annotationSchemaVersion": "2.0.0",
                "taskType": "bill_of_lading_kie",
                "documentType": self.documentType,
                "source": self.source,
                "label": self.normalLabel,
                "evidence": self.evidence,
                "warnings": self.warnings,
                "reviewStatus": self.reviewStatus,
                "reviewNotes": self.reviewNotes,
            },
            strict=True,
        )
        validate_dual_cargo_consistency(self.normalLabel, self.relationExplicitLabel)
        allocation_groups = self.relationExplicitLabel.documentPatch.cargoAllocationGroups or ()
        expected = tuple((row.groupId, row.coverage) for row in allocation_groups)
        actual = tuple((row.groupId, row.coverage) for row in self.relationEvidence)
        if actual != expected:
            raise ValueError(
                "relationEvidence must exactly cover source-ordered cargo allocation groups"
            )
        page_numbers = {row.pageNumber for row in self.source.pages}
        for evidence in self.relationEvidence:
            if not {row.pageNumber for row in evidence.rawOcrEvidence}.issubset(
                page_numbers
            ):
                raise ValueError("cargo relation evidence cites an absent source page")
        return self
