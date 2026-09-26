"""Sparse, MPCI-named Bill-of-Lading extraction target.

Version 6 nests package facts and container placements under their goods item.
The label is an extraction contract, not a fully coded CUSCAR submission.  In
particular, independent printed package and placement quantities are retained
even when a downstream form validator cannot yet accept their combination.
The audited version-5 source remains the lossless relation/provenance record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from typing import Any, Literal

from pydantic import Field, model_validator

from document_ocr.label_schemas.bill_of_lading import (
    ApplicationText,
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
from document_ocr.label_schemas.bill_of_lading_v3 import CategoryToken, NonNegativeQuantity
from document_ocr.label_schemas.bill_of_lading_v4 import RelationExplicitDangerousGoodsV4
from document_ocr.label_schemas.bill_of_lading_v5 import (
    TEMPERATURE_CAPABLE_CONTAINER_TYPES,
    ContainerSizeCategory,
    ContainerTypeCategory,
)
from document_ocr.label_schemas.common import LabelSchemaModel
from document_ocr.label_schemas.mpci_bill_of_lading import ContainerIdentifier, HsCode


class ContainerInformationV6(LabelSchemaModel):
    equipmentIdentifier: ContainerIdentifier
    typeDescription: ApplicationText | None = None
    sizeCategory: ContainerSizeCategory | None = None
    typeCategory: ContainerTypeCategory | None = None
    verifiedGrossMass: Mass | None = None
    sealNumbers: tuple[ApplicationText, ...] | None = Field(default=None, min_length=1)
    temperatureSetpoint: Temperature | None = None

    @model_validator(mode="after")
    def valid_equipment(self) -> ContainerInformationV6:
        if self.sealNumbers is not None and len(self.sealNumbers) != len(set(self.sealNumbers)):
            raise ValueError("sealNumbers must be unique and source ordered")
        semantic = self.sizeCategory is not None or self.typeCategory is not None
        if semantic and (self.sizeCategory is None or self.typeCategory is None):
            raise ValueError("container sizeCategory and typeCategory must be present together")
        if semantic and self.typeDescription is not None:
            raise ValueError("typeDescription is a fallback, not a second equipment category")
        if (
            self.temperatureSetpoint is not None
            and semantic
            and self.typeCategory not in TEMPERATURE_CAPABLE_CONTAINER_TYPES
        ):
            raise ValueError("temperature setpoint requires temperature-capable equipment")
        return self


class NumberAndTypeOfPackagesV6(LabelSchemaModel):
    packageQuantity: NonNegativeQuantity | None = None
    typeCategory: CategoryToken | None = None
    typeOfPackages: PackageTypeText | None = None

    @model_validator(mode="after")
    def valid_package_fact(self) -> NumberAndTypeOfPackagesV6:
        if self.typeCategory is not None and self.typeOfPackages is not None:
            raise ValueError("printed typeOfPackages is a fallback to typeCategory")
        if not self.model_dump(mode="python", exclude_none=True):
            raise ValueError("package must contain a printed or derived fact")
        return self


class SplitGoodsPlacementV6(LabelSchemaModel):
    equipmentIdentifier: ContainerIdentifier
    packageQuantity: NonNegativeQuantity | None = None


class GoodsItemDetailsV6(LabelSchemaModel):
    description: CargoText | None = None
    additionalInformation: tuple[CargoText, ...] | None = Field(default=None, min_length=1)
    grossWeight: CargoMass | None = None
    netWeight: CargoMass | None = None
    volume: Volume | None = None
    marksAndNumbers: tuple[CargoText, ...] | None = Field(default=None, min_length=1)
    hsCodes: tuple[HsCode, ...] | None = Field(default=None, min_length=1)
    handlingInstructions: tuple[CargoText, ...] | None = Field(default=None, min_length=1)
    dangerousGoods: tuple[RelationExplicitDangerousGoodsV4, ...] | None = Field(
        default=None, min_length=1
    )
    origin: GoodsOrigin | None = None
    numberAndTypeOfPackages: tuple[NumberAndTypeOfPackagesV6, ...] | None = Field(
        default=None, min_length=1
    )
    # Keep the relation last in the decoder's goods object: facts first, then placement.
    splitGoodsPlacement: tuple[SplitGoodsPlacementV6, ...] | None = Field(
        default=None, min_length=1
    )

    @model_validator(mode="after")
    def contains_source_supported_fact(self) -> GoodsItemDetailsV6:
        if not self.model_dump(mode="python", exclude_none=True):
            raise ValueError("goods item must contain a fact, package, or placement")
        for name in ("additionalInformation", "marksAndNumbers", "hsCodes", "handlingInstructions"):
            values = getattr(self, name)
            if values is not None and len(values) != len(set(values)):
                raise ValueError(f"{name} values must be unique and source ordered")
        return self


class BillOfLadingDocumentPatchV6(LabelSchemaModel):
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
    containerInformation: tuple[ContainerInformationV6, ...] | None = Field(
        default=None, min_length=1
    )
    forwardingAndExportReferences: tuple[ApplicationText, ...] | None = Field(
        default=None, min_length=1
    )
    goodsItemDetails: tuple[GoodsItemDetailsV6, ...] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def valid_patch(self) -> BillOfLadingDocumentPatchV6:
        if not self.model_dump(mode="python", exclude_none=True):
            raise ValueError("documentPatch must contain a supported fact")
        references = self.forwardingAndExportReferences
        if references is not None and len(references) != len(set(references)):
            raise ValueError("forwarding/export references must be unique and source ordered")
        identifiers = tuple(row.equipmentIdentifier for row in self.containerInformation or ())
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("container equipmentIdentifier values must be unique")
        known = set(identifiers)
        for goods in self.goodsItemDetails or ():
            for placement in goods.splitGoodsPlacement or ():
                if placement.equipmentIdentifier not in known:
                    raise ValueError("splitGoodsPlacement references absent containerInformation")
        return self


class BillOfLadingMPCIAlignedV6Label(LabelSchemaModel):
    schemaVersion: Literal["6.0.0-experimental"]
    documentPatch: BillOfLadingDocumentPatchV6

    def canonical_target(self) -> dict[str, Any]:
        target = self.model_dump(mode="json", exclude_none=True)
        for goods in target["documentPatch"].get("goodsItemDetails", []):
            if "splitGoodsPlacement" in goods:
                goods["splitGoodsPlacement"] = goods.pop("splitGoodsPlacement")
        return target


def project_relation_v5_target_to_v6(target: Mapping[str, Any]) -> dict[str, Any]:
    """Project a validated v5 graph to nested MPCI-named extraction facts.

    The model-facing target omits graph scaffolding, not printed facts.  The
    source v5 target must be kept for package-level relation reversibility.
    """

    from document_ocr.label_schemas.bill_of_lading_v5 import BillOfLadingRelationExplicitV5Label

    source = BillOfLadingRelationExplicitV5Label.model_validate_json(
        json.dumps(target, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        strict=True,
    ).canonical_target()
    patch = source["documentPatch"]
    result_patch = {
        key: value
        for key, value in patch.items()
        if key not in {"containers", "cargoGroups", "cargoPackages", "cargoAllocationGroups"}
    }
    if "containers" in patch:
        result_patch["containerInformation"] = [
            {
                "equipmentIdentifier": row["containerNumber"],
                **{key: value for key, value in row.items() if key != "containerNumber"},
            }
            for row in patch["containers"]
        ]
    packages_by_group: dict[str, list[dict[str, Any]]] = {}
    for package in patch.get("cargoPackages", []):
        packages_by_group.setdefault(package["groupId"], []).append(
            {
                **({"packageQuantity": package["quantity"]} if "quantity" in package else {}),
                **({"typeCategory": package["typeCategory"]} if "typeCategory" in package else {}),
                **(
                    {"typeOfPackages": package["typeDescription"]}
                    if "typeDescription" in package
                    else {}
                ),
            }
        )
    placement_by_group = {
        row["groupId"]: [
            {
                "equipmentIdentifier": placement["containerNumber"],
                **(
                    {"packageQuantity": placement["packageQuantity"]}
                    if "packageQuantity" in placement
                    else {}
                ),
            }
            for placement in row["allocations"]
        ]
        for row in patch.get("cargoAllocationGroups", [])
    }
    if "cargoGroups" in patch:
        result_patch["goodsItemDetails"] = []
        for group in patch["cargoGroups"]:
            group_id = group["groupId"]
            goods = {key: value for key, value in group.items() if key != "groupId"}
            if group_id in packages_by_group:
                goods["numberAndTypeOfPackages"] = packages_by_group[group_id]
            if group_id in placement_by_group:
                goods["splitGoodsPlacement"] = placement_by_group[group_id]
            result_patch["goodsItemDetails"].append(goods)
    output = {"schemaVersion": "6.0.0-experimental", "documentPatch": result_patch}
    return BillOfLadingMPCIAlignedV6Label.model_validate_json(
        json.dumps(output, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        strict=True,
    ).canonical_target()
