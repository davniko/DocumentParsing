"""Construct the latest task target from one compiler-pinned source label.

Historical production templates pin relation-v3 labels; reviewed derivatives
may pin relation-v5 labels when source facts cannot be represented in v3.
Descendant synthesis publishes relation-v5 targets only. This module owns the
one-way boundary and replaces reviewed legacy equipment surfaces with their
readable v5 size/type pairs when the source supports that inference.

Unknown equipment surfaces remain the v5 ``typeDescription`` fallback. A printed
temperature with no printed equipment surface remains a temperature-only
observation: inventing a size/type would put an unobservable fact into the target.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast

from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes
from document_ocr.label_schemas.bill_of_lading_v4 import (
    migrate_relation_v3_target_to_v4,
)
from document_ocr.label_schemas.bill_of_lading_v5 import (
    BillOfLadingRelationExplicitV5Label,
)
from document_ocr.synthesis.container_semantics import review_source_equipment_surface


class LatestTargetConstructionError(ValueError):
    """A source label cannot become a faithful latest-schema target."""


def latest_target_from_source(source_target: Mapping[str, Any]) -> dict[str, Any]:
    """Return one canonical relation-v5 target without inventing source facts."""

    version = source_target.get("schemaVersion")
    if version == "3.0.0-experimental":
        migrated = deepcopy(migrate_relation_v3_target_to_v4(source_target))
        migrated["schemaVersion"] = "5.0.0-experimental"
    elif version == "5.0.0-experimental":
        migrated = deepcopy(dict(source_target))
    else:
        raise LatestTargetConstructionError(
            "production template source target must be relation-v3 or relation-v5"
        )
    patch = migrated.get("documentPatch")
    if not isinstance(patch, dict):
        raise LatestTargetConstructionError("source target lacks documentPatch")
    containers = patch.get("containers") or []
    if not isinstance(containers, list):
        raise LatestTargetConstructionError("source target containers are not a list")
    for index, container_value in enumerate(containers):
        if not isinstance(container_value, dict):
            raise LatestTargetConstructionError(f"source target container {index} is not an object")
        container = cast(dict[str, Any], container_value)
        description = container.get("typeDescription")
        if description is not None and not isinstance(description, str):
            raise LatestTargetConstructionError(
                f"source target container {index} typeDescription is not text"
            )
        reviewed = review_source_equipment_surface(
            description,
            temperature_present=container.get("temperatureSetpoint") is not None,
        )
        if reviewed.size_category is not None and reviewed.type_category is not None:
            container.pop("typeDescription", None)
            container["sizeCategory"] = reviewed.size_category
            container["typeCategory"] = reviewed.type_category

    try:
        validated = BillOfLadingRelationExplicitV5Label.model_validate_json(
            canonical_json_bytes(migrated), strict=True
        )
    except ValidationError as error:
        raise LatestTargetConstructionError(
            "source target cannot be represented faithfully in relation-v5"
        ) from error
    return validated.canonical_target()
