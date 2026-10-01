"""Generated cargo measurements must have same-group typed owners."""

from __future__ import annotations

import pytest

from document_ocr.synthesis.template_compiler.cargo_description_measurement_assertions import (
    validate,
)


def target(*goods: dict[str, object]) -> dict[str, object]:
    return {"documentPatch": {"cargoGroups": list(goods)}}


@pytest.mark.parametrize(
    ("description", "field"),
    [
        ("Fresh apples, gross weight 20237.000 kg", "grossWeight"),
        ("Display panels; gross mass 2749.678 kg", "grossWeight"),
        ("Vehicle, net weight 6217.600 kg", "netWeight"),
        ("Tools, total volume 6.492 m3", "volume"),
        ("Steel sheets, gross weight 4.2 metric tonnes", "grossWeight"),
    ],
)
def test_explicit_total_requires_its_same_group_fact(description: str, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        validate(target({"description": description}))
    unit = "cubic_metre" if field == "volume" else "kilogram"
    validate(target({"description": description, field: {"value": 1, "unit": unit}}))


def test_multiple_clauses_each_need_their_own_fact() -> None:
    description = "Display panels; gross mass 2749.678 kg; volume 21.5 cubic metres"
    with pytest.raises(ValueError, match="volume"):
        validate(
            target(
                {
                    "description": description,
                    "grossWeight": {"value": 2749.678, "unit": "kilogram"},
                }
            )
        )


def test_other_groups_fact_does_not_cover_missing_owner() -> None:
    with pytest.raises(ValueError, match="group 0"):
        validate(
            target(
                {"description": "Apples, gross weight 1000 kg"},
                {"description": "Pears", "grossWeight": {"value": 1000, "unit": "kilogram"}},
            )
        )


@pytest.mark.parametrize(
    "description",
    [
        "Garlic, gross weight 10 kg per carton",
        "Bottles, unit gross weight 12 kg",
        "Pumps, volume 0.25 m3 each",
        "Drums, gross weight 30 kg / package",
        "Garlic, gross weight 10 kg/CTN",
        "Fresh apples, 10 kg cartons",
    ],
)
def test_package_capacity_is_not_a_shipment_total(description: str) -> None:
    validate(target({"description": description}))


def test_missing_description_does_not_claim_a_measurement() -> None:
    validate(target({}))
