"""Provider-enforceable single-source relation schema for labeling agents.

The durable training target remains semantic-v3.  This wire contract removes
downstream category fields and expresses allocation coverage as a discriminated
union, so OpenAI Structured Outputs can enforce each shape before Pydantic's
cross-object canonical validation runs.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import Field

from document_ocr.label_schemas.bill_of_lading import (
    ApplicationText,
    BillOfLadingParties,
    BillOfLadingRoute,
    BillOfLadingTransport,
    FreightTerms,
    Mass,
    PackageTypeText,
    SemanticLocation,
    Temperature,
)
from document_ocr.label_schemas.bill_of_lading_v3 import (
    BillOfLadingRelationExplicitLabel,
    CargoGroup,
    GroupId,
    PackageId,
    RelationExplicitContainer,
)
from document_ocr.label_schemas.common import LabelSchemaModel
from document_ocr.label_schemas.mpci_bill_of_lading import ContainerIdentifier

NonNegativeQuantity = Annotated[int, Field(ge=0)]


def _identifier_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def _remove_redundant_container_marks(patch: dict[str, Any]) -> None:
    """Keep a container identifier in its dedicated field, never as cargo marks."""

    containers = patch.get("containers")
    if not isinstance(containers, list):
        return
    container_keys = {
        _identifier_key(number)
        for row in containers
        if isinstance(row, dict)
        and isinstance((number := row.get("containerNumber")), str)
    }
    cargo_groups = patch.get("cargoGroups")
    if not isinstance(cargo_groups, list):
        return
    for row in cargo_groups:
        if not isinstance(row, dict):
            continue
        marks = row.get("marksAndNumbers")
        if not isinstance(marks, list):
            continue
        retained = [
            value
            for value in marks
            if isinstance(value, str) and _identifier_key(value) not in container_keys
        ]
        if retained:
            row["marksAndNumbers"] = retained
        else:
            row.pop("marksAndNumbers", None)


def _separate_location_country(location: object) -> None:
    """Remove an exactly duplicated country suffix from a structured location name."""

    if not isinstance(location, dict):
        return
    name = location.get("name")
    country = location.get("country")
    if not isinstance(name, str) or not isinstance(country, str):
        return
    match = re.fullmatch(
        rf"(?P<name>.+?)(?:\s*,\s*|\s+-\s+|\s+){re.escape(country)}",
        name,
        flags=re.IGNORECASE,
    )
    if match is not None and (separated := match.group("name").strip(" ,-")):
        location["name"] = separated


def _separate_structured_location_countries(patch: dict[str, Any]) -> None:
    _separate_location_country(patch.get("placeOfIssue"))
    route = patch.get("route")
    if not isinstance(route, dict):
        return
    for key in (
        "placeOfReceipt",
        "portOfLoading",
        "transshipmentPort",
        "portOfDischarge",
        "placeOfDelivery",
        "finalDestination",
    ):
        _separate_location_country(route.get(key))


def _strip_address_separator_before_structured_locality(patch: dict[str, Any]) -> None:
    """Remove only a dangling delimiter left before a modeled city/country."""

    parties = patch.get("parties")
    if not isinstance(parties, dict):
        return
    rows: list[object] = [
        parties.get(key)
        for key in (
            "shipper",
            "consignee",
            "carrier",
            "forwardingAgent",
            "deliveryAgent",
            "consolidator",
        )
    ]
    notify_parties = parties.get("notifyParties")
    if isinstance(notify_parties, list):
        rows.extend(notify_parties)
    for row in rows:
        if not isinstance(row, dict) or not (row.get("city") or row.get("country")):
            continue
        address = row.get("address")
        if isinstance(address, str) and re.search(r"[-\u2013\u2014,;/|]\s*$", address):
            normalized = re.sub(r"\s*[-\u2013\u2014,;/|]\s*$", "", address)
            if normalized:
                row["address"] = normalized


class AgentRelationExplicitContainer(LabelSchemaModel):
    containerNumber: ContainerIdentifier
    typeDescription: ApplicationText | None = None
    verifiedGrossMass: Mass | None = None
    sealNumbers: tuple[ApplicationText, ...] | None = Field(default=None, min_length=1)
    temperatureSetpoint: Temperature | None = None


class AgentCargoPackageFact(LabelSchemaModel):
    packageId: PackageId
    groupId: GroupId
    quantity: NonNegativeQuantity | None = None
    typeDescription: PackageTypeText | None = None


class AgentContainerMembership(LabelSchemaModel):
    containerNumber: ContainerIdentifier


class AgentContainerQuantityAllocation(LabelSchemaModel):
    containerNumber: ContainerIdentifier
    packageQuantity: NonNegativeQuantity


class AgentOneToOneAllocation(LabelSchemaModel):
    containerNumber: ContainerIdentifier
    packageQuantity: NonNegativeQuantity
    packageId: PackageId


class AgentOneToOneAllocationGroup(LabelSchemaModel):
    groupId: GroupId
    coverage: Literal["one_to_one_package_allocations"]
    allocations: tuple[AgentOneToOneAllocation, ...] = Field(min_length=1)


class AgentSinglePackageAllocationGroup(LabelSchemaModel):
    groupId: GroupId
    coverage: Literal["single_package_level"]
    packageId: PackageId
    allocations: tuple[AgentContainerQuantityAllocation, ...] = Field(min_length=1)


class AgentCombinedPackageAllocationGroup(LabelSchemaModel):
    groupId: GroupId
    coverage: Literal["all_package_levels_combined"]
    packageIds: tuple[PackageId, ...] = Field(min_length=2)
    allocations: tuple[AgentContainerQuantityAllocation, ...] = Field(min_length=1)


class AgentUnlinkedQuantityAllocationGroup(LabelSchemaModel):
    groupId: GroupId
    coverage: Literal["unlinked_package_quantities"]
    allocations: tuple[AgentContainerQuantityAllocation, ...] = Field(min_length=1)


class AgentContainerMembershipAllocationGroup(LabelSchemaModel):
    groupId: GroupId
    coverage: Literal["container_membership_only"]
    allocations: tuple[AgentContainerMembership, ...] = Field(min_length=1)


AgentCargoAllocationGroup = Annotated[
    AgentOneToOneAllocationGroup
    | AgentSinglePackageAllocationGroup
    | AgentCombinedPackageAllocationGroup
    | AgentUnlinkedQuantityAllocationGroup
    | AgentContainerMembershipAllocationGroup,
    Field(discriminator="coverage"),
]


class AgentRelationExplicitDocumentPatch(LabelSchemaModel):
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
    containers: tuple[AgentRelationExplicitContainer, ...] | None = Field(
        default=None, min_length=1
    )
    forwardingAndExportReferences: tuple[ApplicationText, ...] | None = Field(
        default=None, min_length=1
    )
    cargoGroups: tuple[CargoGroup, ...] | None = Field(default=None, min_length=1)
    cargoPackages: tuple[AgentCargoPackageFact, ...] | None = Field(
        default=None, min_length=1
    )
    cargoAllocationGroups: tuple[AgentCargoAllocationGroup, ...] | None = Field(
        default=None, min_length=1
    )


class AgentBillOfLadingRelationExplicitLabel(LabelSchemaModel):
    schemaVersion: Literal["3.0.0-experimental"]
    documentPatch: AgentRelationExplicitDocumentPatch

    @classmethod
    def from_canonical(
        cls, label: BillOfLadingRelationExplicitLabel
    ) -> AgentBillOfLadingRelationExplicitLabel:
        """Recover the provider wire shape from an unmapped canonical label."""

        patch = label.documentPatch.model_dump(mode="json", exclude_none=True)
        for row in patch.get("containers", []):
            if row.get("typeCategory") is not None:
                raise ValueError("mapped container typeCategory cannot return to agent wire form")
            row.pop("typeCategory", None)
        for row in patch.get("cargoPackages", []):
            if row.get("typeCategory") is not None:
                raise ValueError("mapped package typeCategory cannot return to agent wire form")
            row.pop("typeCategory", None)
        for group in patch.get("cargoAllocationGroups", []):
            coverage = group["coverage"]
            package_ids = group.pop("packageIds")
            if coverage == "single_package_level":
                if len(package_ids) != 1:
                    raise ValueError("single_package_level requires one canonical package ID")
                group["packageId"] = package_ids[0]
            for allocation in group["allocations"]:
                if coverage != "one_to_one_package_allocations":
                    allocation.pop("packageId", None)
                if coverage == "container_membership_only":
                    allocation.pop("packageQuantity", None)
        return cls.model_validate_json(
            json.dumps(
                {"schemaVersion": label.schemaVersion, "documentPatch": patch},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            strict=True,
        )

    def to_canonical(self) -> BillOfLadingRelationExplicitLabel:
        patch = self.documentPatch.model_dump(mode="json", exclude_none=True)
        _remove_redundant_container_marks(patch)
        _separate_structured_location_countries(patch)
        _strip_address_separator_before_structured_locality(patch)
        if containers := patch.get("containers"):
            patch["containers"] = [
                RelationExplicitContainer.model_validate_json(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                    strict=True,
                ).model_dump(mode="json", exclude_none=True)
                for row in containers
            ]
        if packages := patch.get("cargoPackages"):
            patch["cargoPackages"] = [
                {
                    **row,
                    "typeCategory": None,
                }
                for row in packages
            ]
        if allocation_groups := self.documentPatch.cargoAllocationGroups:
            canonical_groups: list[dict[str, object]] = []
            for group in allocation_groups:
                if isinstance(group, AgentOneToOneAllocationGroup):
                    allocations = group.allocations
                    canonical_groups.append(
                        {
                            "groupId": group.groupId,
                            "coverage": group.coverage,
                            "packageIds": [row.packageId for row in allocations],
                            "allocations": [
                                row.model_dump(mode="json") for row in allocations
                            ],
                        }
                    )
                elif isinstance(group, AgentSinglePackageAllocationGroup):
                    canonical_groups.append(
                        {
                            "groupId": group.groupId,
                            "coverage": group.coverage,
                            "packageIds": [group.packageId],
                            "allocations": [
                                row.model_dump(mode="json") for row in group.allocations
                            ],
                        }
                    )
                elif isinstance(group, AgentCombinedPackageAllocationGroup):
                    canonical_groups.append(
                        {
                            "groupId": group.groupId,
                            "coverage": group.coverage,
                            "packageIds": list(group.packageIds),
                            "allocations": [
                                row.model_dump(mode="json") for row in group.allocations
                            ],
                        }
                    )
                else:
                    canonical_groups.append(
                        {
                            "groupId": group.groupId,
                            "coverage": group.coverage,
                            "packageIds": [],
                            "allocations": [
                                row.model_dump(mode="json") for row in group.allocations
                            ],
                        }
                    )
            patch["cargoAllocationGroups"] = canonical_groups
        return BillOfLadingRelationExplicitLabel.model_validate_json(
            json.dumps(
                {"schemaVersion": self.schemaVersion, "documentPatch": patch},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            strict=True,
        )
