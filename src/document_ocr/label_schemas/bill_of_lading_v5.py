"""Relation-explicit B/L target with semantic container size and type.

Version 5 keeps the independent dangerous-goods fields introduced in version
4 and replaces ambiguous printed equipment aliases with two readable concepts:

* physical size/height; and
* the BIC/MPCI equipment type group.

Those two values project deterministically to the four-character code accepted
by MPCI.  Printed spellings such as ``40HQ`` and ``40' HC`` belong to the OCR
renderer and evidence sidecar, not to the model-facing label.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

from document_ocr.label_schemas.bill_of_lading_v3 import RelationExplicitContainer
from document_ocr.label_schemas.bill_of_lading_v4 import (
    BillOfLadingRelationExplicitV4Label,
    CargoGroupV4,
    RelationExplicitDocumentPatchV4,
)
from document_ocr.label_schemas.common import LabelSchemaModel

ContainerSizeCategory = Literal[
    "TWENTY_FOOT_STANDARD_HEIGHT",
    "TWENTY_FOOT_HIGH_CUBE",
    "FORTY_FOOT_STANDARD_HEIGHT",
    "FORTY_FOOT_HIGH_CUBE",
    "FORTY_FIVE_FOOT_HIGH_CUBE",
]

ContainerTypeCategory = Literal[
    "GENERAL_PURPOSE",
    "VENTILATED_GENERAL_PURPOSE",
    "DRY_BULK",
    "NAMED_CARGO",
    "REFRIGERATED",
    "REFRIGERATED_AND_HEATED",
    "SELF_POWERED_REFRIGERATED",
    "REFRIGERATED_HEATED_REMOVABLE_EQUIPMENT",
    "INSULATED",
    "OPEN_TOP",
    "PLATFORM",
    "PLATFORM_FIXED",
    "PLATFORM_COLLAPSIBLE",
    "PLATFORM_COMPLETE_SUPERSTRUCTURE",
    "PLATFORM_NAMED_CARGO",
    "PRESSURIZED_TANK",
    "DRY_HOPPER_TANK",
    "DRY_REAR_DISCHARGE_TANK",
    "AIR_SURFACE",
]

_SIZE_CODE: dict[ContainerSizeCategory, str] = {
    "TWENTY_FOOT_STANDARD_HEIGHT": "22",
    "TWENTY_FOOT_HIGH_CUBE": "25",
    "FORTY_FOOT_STANDARD_HEIGHT": "42",
    "FORTY_FOOT_HIGH_CUBE": "45",
    "FORTY_FIVE_FOOT_HIGH_CUBE": "55",
}

_TYPE_CODE: dict[ContainerTypeCategory, str] = {
    "GENERAL_PURPOSE": "GP",
    "VENTILATED_GENERAL_PURPOSE": "VH",
    "DRY_BULK": "BU",
    "NAMED_CARGO": "SN",
    "REFRIGERATED": "RE",
    "REFRIGERATED_AND_HEATED": "RT",
    "SELF_POWERED_REFRIGERATED": "RS",
    "REFRIGERATED_HEATED_REMOVABLE_EQUIPMENT": "HR",
    "INSULATED": "HI",
    "OPEN_TOP": "UT",
    "PLATFORM": "PL",
    "PLATFORM_FIXED": "PF",
    "PLATFORM_COLLAPSIBLE": "PC",
    "PLATFORM_COMPLETE_SUPERSTRUCTURE": "PS",
    "PLATFORM_NAMED_CARGO": "PT",
    "PRESSURIZED_TANK": "KL",
    "DRY_HOPPER_TANK": "NH",
    "DRY_REAR_DISCHARGE_TANK": "NN",
    "AIR_SURFACE": "AS",
}

CONTAINER_SIZE_CATEGORIES: tuple[ContainerSizeCategory, ...] = tuple(_SIZE_CODE)
CONTAINER_TYPE_CATEGORIES: tuple[ContainerTypeCategory, ...] = tuple(_TYPE_CODE)

TEMPERATURE_CAPABLE_CONTAINER_TYPES: frozenset[ContainerTypeCategory] = frozenset(
    {
        "REFRIGERATED",
        "REFRIGERATED_AND_HEATED",
        "SELF_POWERED_REFRIGERATED",
        "REFRIGERATED_HEATED_REMOVABLE_EQUIPMENT",
    }
)


class RelationExplicitContainerV5(RelationExplicitContainer):
    sizeCategory: ContainerSizeCategory | None = Field(
        default=None,
        description=(
            "Physical container length and height class. Use the semantic class, not a "
            "printed carrier abbreviation such as 40HQ or 40RH."
        ),
    )
    typeCategory: ContainerTypeCategory | None = Field(
        default=None,
        description=(
            "Readable MPCI/BIC equipment family: general-purpose, ventilated, dry-bulk, "
            "named-cargo, refrigerated/thermal, insulated, open-top, platform, tank, or "
            "air/surface. This is the semantic class inferred from the printed equipment "
            "surface; downstream code projection is deterministic."
        ),
    )

    @model_validator(mode="after")
    def semantic_equipment_pair_is_complete(self) -> RelationExplicitContainerV5:
        semantic = self.sizeCategory is not None or self.typeCategory is not None
        if semantic and (self.sizeCategory is None or self.typeCategory is None):
            raise ValueError("container sizeCategory and typeCategory must be present together")
        if semantic and self.typeDescription is not None:
            raise ValueError(
                "container typeDescription is a printed fallback and cannot accompany "
                "semantic size/type categories"
            )
        if self.temperatureSetpoint is not None:
            if semantic and self.typeCategory not in TEMPERATURE_CAPABLE_CONTAINER_TYPES:
                raise ValueError("temperature setpoint requires temperature-capable equipment")
            if not semantic and self.typeDescription is None:
                raise ValueError(
                    "temperature setpoint requires semantic equipment or a printed fallback"
                )
        return self

    def application_size_type_code(self) -> str | None:
        """Return the exact MPCI/BIC code implied by the semantic pair."""

        if self.sizeCategory is None or self.typeCategory is None:
            return None
        return _SIZE_CODE[self.sizeCategory] + _TYPE_CODE[self.typeCategory]


# Pydantic field replacement on inherited models is intentional here; static
# assignment compatibility cannot express a separately versioned wire model.
class RelationExplicitDocumentPatchV5(RelationExplicitDocumentPatchV4):
    containers: tuple[RelationExplicitContainerV5, ...] | None = Field(
        default=None,
        min_length=1,
    )
    cargoGroups: tuple[CargoGroupV4, ...] | None = Field(default=None, min_length=1)


class BillOfLadingRelationExplicitV5Label(LabelSchemaModel):
    schemaVersion: Literal["5.0.0-experimental"]
    documentPatch: RelationExplicitDocumentPatchV5

    def canonical_target(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def migrate_relation_v4_target_to_v5(target: Mapping[str, Any]) -> dict[str, Any]:
    """Change only the schema version; enrichment is a separate audited step."""

    encoded = json.dumps(
        dict(target),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    source = BillOfLadingRelationExplicitV4Label.model_validate_json(encoded, strict=True)
    migrated = source.canonical_target()
    migrated["schemaVersion"] = "5.0.0-experimental"
    validated = BillOfLadingRelationExplicitV5Label.model_validate_json(
        json.dumps(migrated, allow_nan=False, ensure_ascii=False, separators=(",", ":")),
        strict=True,
    )
    return validated.canonical_target()


def semantic_container_code(
    size_category: ContainerSizeCategory,
    type_category: ContainerTypeCategory,
) -> str:
    """Project readable model values to MPCI's exact four-character code."""

    return _SIZE_CODE[size_category] + _TYPE_CODE[type_category]
