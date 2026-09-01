"""Relation-explicit B/L target with independent dangerous-goods semantics.

Version 4 is intentionally a new target contract.  Version 3 coupled packing
group to flashpoint and allowed only one subsidiary hazard, even though both
constraints are false for the regulatory source used by synthesis.  Existing
version-3 labels remain immutable; ``migrate_relation_v3_target_to_v4`` is the
only supported deterministic bridge.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

from document_ocr.label_schemas.bill_of_lading import Temperature
from document_ocr.label_schemas.bill_of_lading_v3 import (
    BillOfLadingRelationExplicitLabel,
    CargoGroup,
    HazardCategory,
    RelationExplicitDocumentPatch,
)
from document_ocr.label_schemas.common import LabelSchemaModel

PackingGroupCategoryV4 = Literal[
    "HIGH_DANGER",
    "MEDIUM_DANGER",
    "LOW_DANGER",
    "NOT_ASSIGNED",
]


class RelationExplicitDangerousGoodsFlashPointV4(LabelSchemaModel):
    """A printed flashpoint; packing group is deliberately not nested here."""

    temperature: Temperature


class RelationExplicitDangerousGoodsV4(LabelSchemaModel):
    """Sparse task-facing DG facts with readable regulatory categories."""

    unNumber: str | None = Field(default=None, pattern=r"^[0-9]{4}$")
    hazardCategory: HazardCategory | None = None
    subsidiaryHazardCategories: tuple[HazardCategory, ...] | None = Field(
        default=None,
        min_length=1,
    )
    packingGroupCategory: PackingGroupCategoryV4 | None = None
    flashPoint: RelationExplicitDangerousGoodsFlashPointV4 | None = None

    @model_validator(mode="after")
    def contains_unique_dangerous_goods(self) -> RelationExplicitDangerousGoodsV4:
        if not self.model_dump(mode="python", exclude_none=True):
            raise ValueError("dangerous goods record must contain supported evidence")
        subsidiaries = self.subsidiaryHazardCategories
        if subsidiaries is not None and len(subsidiaries) != len(set(subsidiaries)):
            raise ValueError(
                "subsidiaryHazardCategories must be unique and regulatory-source ordered"
            )
        return self


class CargoGroupV4(CargoGroup):
    # Pydantic intentionally replaces this v3 field with the breaking v4
    # contract. Static assignment compatibility cannot express model-field
    # replacement between two separately versioned schema types.
    dangerousGoods: tuple[RelationExplicitDangerousGoodsV4, ...] | None = Field(
        default=None,
        min_length=1,
    )  # type: ignore[assignment]


class RelationExplicitDocumentPatchV4(RelationExplicitDocumentPatch):
    cargoGroups: tuple[CargoGroupV4, ...] | None = Field(default=None, min_length=1)


class BillOfLadingRelationExplicitV4Label(LabelSchemaModel):
    schemaVersion: Literal["4.0.0-experimental"]
    documentPatch: RelationExplicitDocumentPatchV4

    def canonical_target(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def migrate_relation_v3_target_to_v4(target: Mapping[str, Any]) -> dict[str, Any]:
    """Move represented v3 DG facts without enriching or changing their values."""

    encoded = json.dumps(
        dict(target),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    source = BillOfLadingRelationExplicitLabel.model_validate_json(encoded, strict=True)
    migrated = source.canonical_target()
    migrated["schemaVersion"] = "4.0.0-experimental"
    for group in migrated.get("documentPatch", {}).get("cargoGroups", []):
        for dangerous in group.get("dangerousGoods", []):
            subsidiary = dangerous.pop("subsidiaryHazardCategory", None)
            if subsidiary is not None:
                dangerous["subsidiaryHazardCategories"] = [subsidiary]
            flashpoint = dangerous.get("flashPoint")
            if isinstance(flashpoint, dict):
                packing_group = flashpoint.pop("packingGroupCategory", None)
                if packing_group is not None:
                    dangerous["packingGroupCategory"] = packing_group
                if not flashpoint:
                    dangerous.pop("flashPoint")
    validated = BillOfLadingRelationExplicitV4Label.model_validate_json(
        json.dumps(migrated, allow_nan=False, ensure_ascii=False, separators=(",", ":")),
        strict=True,
    )
    return validated.canonical_target()
