"""Vehicle cargo must not enter the general commodity sampler by accident."""

from __future__ import annotations

import pytest

from document_ocr.synthesis.template_compiler.vehicle_source_evidence import (
    vehicle_source_evidence,
)


def _target(description: str, *, mark: str = "", package: str = "PACKAGE_UNIT") -> dict:
    return {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": description, "marksAndNumbers": [mark]}
            ],
            "cargoPackages": [{"groupId": "g1", "typeCategory": package}],
        }
    }


@pytest.mark.parametrize(
    ("raw", "description", "mark", "package", "reason"),
    [
        (
            "VIN Number(s): WP0ZZZY1ZRSA31140 1 NEW CAR TYPE: PORSCHE TAYCAN 4S",
            "NEW CAR TYPE: PORSCHE TAYCAN 4S",
            "WP0ZZZY1ZRSA31140",
            "PACKAGE_UNIT",
            "printed_vehicle_identity_caption",
        ),
        (
            "Marks and Numbers: CHASSIS NUMBER\nSALYA2BX4RA381177\n",
            "",
            "SALYA2BX4RA381177",
            "",
            "printed_vehicle_identity_caption",
        ),
        (
            "1 Container Said to Contain 2 VEHICLES",
            "Excavators",
            "",
            "PACKAGE_VEHICLE",
            "vehicle_package_category",
        ),
        (
            "14 used cars, marks listed in attachment",
            "KOREAN USED CARS",
            "",
            "PACKAGE_UNIT",
            "whole_vehicle_cargo_description",
        ),
        (
            "WVWZZZ1KZ7M133112",
            "VOLKSWAGEN JETTA (personal effects) 2007 1400 CC",
            "WVWZZZ1KZ7M133112",
            "PACKAGE_PACKAGE",
            "vehicle_context_for_printed_seventeen_character_mark",
        ),
        (
            "4 NEW Units MERCEDES-BENZ 4048 K 6x4\n"
            "CAR(S), VEHICLE(S), UNPACKED AND UNPROTECTED",
            "MERCEDES-BENZ 4048 K 6x4",
            "",
            "PACKAGE_UNIT",
            "printed_new_or_used_units_under_vehicle_clause",
        ),
    ],
)
def test_real_vehicle_evidence_requires_joint_sampling(
    raw: str, description: str, mark: str, package: str, reason: str
) -> None:
    evidence = vehicle_source_evidence(raw, _target(description, mark=mark, package=package))
    assert evidence is not None
    assert reason in evidence.reasons


@pytest.mark.parametrize(
    ("description", "mark", "raw"),
    [
        (
            "COMPONENT PARTS FOR HARNESS",
            "WKESD000000844299",
            "WKESD000000844299\nHS CODE 39173200",
        ),
        ("USED CAR SPARE PARTS", "", "SECONDHAND VEHICLE(S) carrier conditions"),
        ("USED TRUCK PARTS", "", "TRUCK FOGGER and parts"),
        ("PALM NUT", "", "CHASSIS / VIN Number is stated in the carrier's general terms"),
    ],
)
def test_unrelated_long_reference_or_generic_vehicle_words_do_not_trigger(
    description: str, mark: str, raw: str
) -> None:
    assert vehicle_source_evidence(raw, _target(description, mark=mark)) is None
