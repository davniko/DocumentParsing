from datetime import date
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.observed_thermal_goods import (
    extend_support,
    shared_setpoints,
    validate_setpoints,
)
from document_ocr.synthesis.thermal_goods import ThermalGoodsSupport


def source(code="300431", value=4):
    return {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "hsCodes": [code]}],
            "containers": [
                {
                    "containerNumber": "ONE",
                    "temperatureSetpoint": {"value": value, "unit": "celsius"},
                }
            ],
        }
    }


def registry():
    return NS(
        global_codes=("300431",),
        receipt=NS(snapshot_date=date(2026, 1, 1)),
        require_global=lambda code, on_date: NS(
            chapter_description="Medicaments",
            heading_description="Medicaments containing insulin",
            description="Insulin",
        ),
    )


def extend(rows):
    return extend_support(
        ThermalGoodsSupport((), (), (), ("39",)),
        registry=registry(),
        fit_targets=rows,
        bounds={"FROZEN": (-24, -18), "CHILLED": (-3, 5.5)},
    )


def test_explicit_fit_identity_keeps_exact_observed_temperature_and_provenance():
    result = extend({"fit-a": source(), "fit-b": source(value=2)})
    assert len(result.chilled) == 1
    identity = result.chilled[0]
    assert identity.hs6 == "300431"
    assert identity.observed_setpoints_celsius == (2, 4)
    assert identity.fit_document_ids == ("fit-a", "fit-b")
    assert not result.observation_review


def test_unknown_registry_codes_and_conflicting_profiles_remain_explicit_review():
    result = extend(
        {"bad-code": source(code="002023"), "warm": source(), "cold": source(value=-20)}
    )
    assert not result.chilled and not result.frozen
    assert {c for c, _ in result.observation_review} == {"002023", "300431"}


def test_partial_or_absent_equipment_temperature_cannot_establish_support():
    target = source()
    target["documentPatch"]["containers"].append({"containerNumber": "TWO"})
    assert not extend({"fit": target}).chilled
    target["documentPatch"]["containers"][0].pop("temperatureSetpoint")
    assert not extend({"fit": target}).chilled


def test_shared_box_intersects_observations_and_rejects_disjoint_ones():
    assert shared_setpoints(
        [{}, {"observedSetpointsCelsius": [2, 4]}, {"observedSetpointsCelsius": [4]}]
    ) == (4,)
    with pytest.raises(ValueError, match="common fit-observed"):
        shared_setpoints([{"observedSetpointsCelsius": [2]}, {"observedSetpointsCelsius": [4]}])


def test_publication_checks_actual_temperature_even_if_scenario_target_was_bad():
    facts = {"g1": [{"observedSetpointsCelsius": [4]}]}
    validate_setpoints(source(), facts)
    with pytest.raises(ValueError, match="outside the exact"):
        validate_setpoints(source(value=3), facts)
    target = source()
    target["documentPatch"]["containers"][0].pop("temperatureSetpoint")
    with pytest.raises(ValueError, match="lost their setpoint"):
        validate_setpoints(target, facts)
