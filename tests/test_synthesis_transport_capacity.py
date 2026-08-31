from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from document_ocr.synthesis.config import load_synthesis_structured_baseline_config
from document_ocr.synthesis.transport_capacity import (
    capacity_limits,
    classify_equipment,
    document_capacity_receipt,
    group_capacity_budgets,
    numeric_fit_envelope_violations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/synthesis/mpci_bl_combined1157_structured_baseline50.yaml"


def _limits():
    config = load_synthesis_structured_baseline_config(CONFIG)
    return capacity_limits(config.generation.transport_capacity)


def _target(
    *,
    containers: list[dict[str, object]],
    groups: list[dict[str, object]],
    allocations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schemaVersion": 3,
        "documentPatch": {
            "containers": containers,
            "cargoGroups": groups,
            "cargoAllocationGroups": allocations or [],
        },
    }


def test_printed_equipment_aliases_resolve_to_supported_capacity_families() -> None:
    assert classify_equipment({"typeDescription": "20ST"}) == "twenty_standard"
    assert classify_equipment({"typeDescription": "DC 20"}) == "twenty_standard"
    assert classify_equipment({"typeDescription": "40'"}) == "forty_standard"
    assert classify_equipment({"typeDescription": "40 DRY 9'6"}) == "forty_high_cube"
    assert classify_equipment({"typeDescription": "40HQ"}) == "forty_high_cube"
    assert classify_equipment({"typeCode": "45G1"}) == "forty_high_cube"
    assert classify_equipment({"typeDescription": "1 X 45HC"}) == "forty_five_high_cube"
    assert classify_equipment({"typeDescription": "40 FLAT RACK"}) == "out_of_gauge"
    assert classify_equipment({"typeDescription": "CTNR"}) == "unclassified"


def test_document_capacity_is_decimal_exact_at_boundary_and_rejects_excess() -> None:
    target = _target(
        containers=[{"containerNumber": "MSCU0000000", "typeDescription": "20ST"}],
        groups=[
            {
                "groupId": "g1",
                "grossWeight": {"value": 29.715, "unit": "metric_tonne"},
                "netWeight": {"value": 29715, "unit": "kilogram"},
                "volume": {"value": 34.86, "unit": "cubic_metre"},
            }
        ],
    )
    receipt = document_capacity_receipt(target, _limits())
    assert receipt.valid
    assert receipt.gross_payload_utilization == Decimal("1")
    assert receipt.volume_utilization == Decimal("1")

    invalid = deepcopy(target)
    invalid["documentPatch"]["cargoGroups"][0]["grossWeight"]["value"] = 29.7151
    receipt = document_capacity_receipt(invalid, _limits())
    assert receipt.violations == ("document_gross_weight_exceeds_container_payload",)


def test_five_twenty_foot_regression_is_rejected_for_mass_and_volume() -> None:
    target = _target(
        containers=[
            {"containerNumber": f"MSCU00000{index}0", "typeDescription": "20ST"}
            for index in range(5)
        ],
        groups=[
            {
                "groupId": "g1",
                "grossWeight": {"value": 534804.6, "unit": "kilogram"},
                "volume": {"value": 2014.7, "unit": "cubic_metre"},
            }
        ],
    )

    receipt = document_capacity_receipt(target, _limits())

    assert receipt.violations == (
        "document_gross_weight_exceeds_container_payload",
        "document_volume_exceeds_container_capacity",
    )
    assert receipt.gross_payload_utilization is not None
    assert receipt.gross_payload_utilization > 3
    assert numeric_fit_envelope_violations(target, _limits()) == (
        "gross_weight_exceeds_absolute_equipment_envelope",
        "volume_exceeds_absolute_equipment_envelope",
    )


def test_minor_type_specific_excess_is_not_a_severe_numeric_fit_anomaly() -> None:
    target = _target(
        containers=[{"containerNumber": "MSCU0000000", "typeDescription": "40HC"}],
        groups=[
            {
                "groupId": "g1",
                "grossWeight": {"value": 32000, "unit": "kilogram"},
                "volume": {"value": 80, "unit": "cubic_metre"},
            }
        ],
    )

    assert not document_capacity_receipt(target, _limits()).valid
    assert numeric_fit_envelope_violations(target, _limits()) == ()


def test_non_containerized_breakbulk_is_explicitly_not_capacity_constrained() -> None:
    target = _target(
        containers=[],
        groups=[
            {
                "groupId": "g1",
                "grossWeight": {"value": 250000, "unit": "kilogram"},
                "volume": {"value": 1000, "unit": "cubic_metre"},
            }
        ],
    )

    receipt = document_capacity_receipt(target, _limits())

    assert receipt.valid
    assert receipt.payload_capacity_kg is None
    assert receipt.volume_capacity_m3 is None
    assert receipt.gross_payload_utilization is None


def test_group_budgets_preserve_document_capacity_and_tighten_explicit_links() -> None:
    target = _target(
        containers=[
            {"containerNumber": "MSCU0000000", "typeDescription": "20ST"},
            {"containerNumber": "MSCU0000018", "typeDescription": "40HC"},
        ],
        groups=[
            {"groupId": "g1", "grossWeight": {"value": 10000, "unit": "kilogram"}},
            {"groupId": "g2", "grossWeight": {"value": 30000, "unit": "kilogram"}},
        ],
        allocations=[
            {
                "groupId": "g1",
                "coverage": "membership_only",
                "packageIds": [],
                "allocations": [{"containerNumber": "MSCU0000000"}],
            }
        ],
    )

    budgets = group_capacity_budgets(target, _limits())

    total_capacity = (Decimal("28300") + Decimal("28690")) * Decimal("1.05")
    assert budgets["g1"]["gross"] == total_capacity / 4
    assert budgets["g2"]["gross"] == total_capacity * 3 / 4
    assert budgets["g1"]["gross"] <= Decimal("28300") * Decimal("1.05")
    assert budgets["g1"]["gross"] + budgets["g2"]["gross"] == total_capacity


def test_vgm_is_not_mistaken_for_cargo_mass_and_oog_volume_is_receipted_unbounded() -> None:
    target = _target(
        containers=[
            {
                "containerNumber": "MSCU0000000",
                "typeDescription": "40 FLAT RACK",
                "verifiedGrossMass": {"value": 999999, "unit": "kilogram"},
            }
        ],
        groups=[
            {
                "groupId": "g1",
                "grossWeight": {"value": 47000, "unit": "kilogram"},
                "volume": {"value": 500, "unit": "cubic_metre"},
            }
        ],
    )

    receipt = document_capacity_receipt(target, _limits())

    assert receipt.valid
    assert receipt.gross_weight_kg == Decimal("47000")
    assert receipt.volume_capacity_m3 is None
    assert receipt.volume_utilization is None
