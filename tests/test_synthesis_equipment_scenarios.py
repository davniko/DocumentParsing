from __future__ import annotations

from pathlib import Path

import pytest

from document_ocr.synthesis.equipment_registry import load_bic_equipment_registry
from document_ocr.synthesis.equipment_scenarios import (
    EquipmentFitObservation,
    build_equipment_scenario_support,
    sample_equipment_type,
)
from document_ocr.synthesis.generators import DeterministicStream

MANIFEST = Path(
    "data/registries/equipment/bic-iso6346-2022-web-snapshot-20260831/source-manifest.json"
)


def _observations() -> tuple[EquipmentFitObservation, ...]:
    return (
        EquipmentFitObservation("d1", "c1", "40GP", "40GP", False),
        EquipmentFitObservation("d2", "c2", "40 REEFER", "40RE", True),
        EquipmentFitObservation("d3", "c3", "40HQ", None, False),
        EquipmentFitObservation("d4", "c4", None, None, False),
    )


def test_support_is_exact_only_and_accounts_for_unclassified_surfaces() -> None:
    registry = load_bic_equipment_registry(MANIFEST)
    support = build_equipment_scenario_support(
        observations=_observations(),
        fit_document_ids=("d1", "d2", "d3", "d4"),
        registry=registry,
    )

    assert support.audit.exact_registry_observations == 2
    assert support.audit.unclassified_surface_observations == 1
    assert support.audit.absent_surface_observations == 1
    assert support.audit.temperature_observations == 1
    assert {row.identity.size_type_code for row in support.observed_codes} == {"40GP", "40RE"}


def test_sampling_is_deterministic_registry_backed_and_never_emits_aliases() -> None:
    registry = load_bic_equipment_registry(MANIFEST)
    observations = _observations()
    support = build_equipment_scenario_support(
        observations=observations,
        fit_document_ids=("d1", "d2", "d3", "d4"),
        registry=registry,
    )
    stream = DeterministicStream(17, "equipment-scenario-test", "d3")
    first = sample_equipment_type(
        observations[2],
        support=support,
        registry=registry,
        registry_exploration_permyriad=10_000,
        stream=stream,
    )
    second = sample_equipment_type(
        observations[2],
        support=support,
        registry=registry,
        registry_exploration_permyriad=10_000,
        stream=stream,
    )

    assert first == second
    assert first.sampling_component == "registry_exploration"
    assert first.source_resolution == "unclassified_printed_surface"
    assert first.printed_surface is None
    assert first.printed_surface_status == "pending_text_realization"
    assert first.form_projection == "exact_container_code"
    assert registry.classify(first.size_type_code).size_type_code == first.size_type_code
    assert first.size_type_code not in {"40HQ", "40HC", "20DV"}

    untyped = sample_equipment_type(
        observations[3],
        support=support,
        registry=registry,
        registry_exploration_permyriad=0,
        stream=DeterministicStream(19, "equipment-scenario-test", "d4"),
    )
    assert untyped.source_resolution == "untyped"


def test_temperature_rows_can_only_sample_setpoint_capable_equipment() -> None:
    registry = load_bic_equipment_registry(MANIFEST)
    observations = _observations()
    support = build_equipment_scenario_support(
        observations=observations,
        fit_document_ids=("d1", "d2", "d3", "d4"),
        registry=registry,
    )
    sampled = sample_equipment_type(
        observations[1],
        support=support,
        registry=registry,
        registry_exploration_permyriad=10_000,
        stream=DeterministicStream(23, "equipment-scenario-test", "d2"),
    )

    assert sampled.type_family == "R"
    assert sampled.supports_temperature_setpoint
    assert sampled.thermal_capability in {"refrigerated", "heated"}


def test_exact_temperature_contradiction_and_registry_mismatch_fail_closed() -> None:
    registry = load_bic_equipment_registry(MANIFEST)
    contradictory = (EquipmentFitObservation("d1", "c1", "40GP", "40GP", True),)
    with pytest.raises(ValueError, match="non-thermal"):
        build_equipment_scenario_support(
            observations=contradictory,
            fit_document_ids=("d1",),
            registry=registry,
        )

    support = build_equipment_scenario_support(
        observations=_observations(),
        fit_document_ids=("d1", "d2", "d3", "d4"),
        registry=registry,
    )
    changed = support.__class__(
        registry_manifest_sha256="0" * 64,
        fit_document_ids=support.fit_document_ids,
        observed_codes=support.observed_codes,
        audit=support.audit,
    )
    with pytest.raises(ValueError, match="differs from the support"):
        sample_equipment_type(
            _observations()[0],
            support=changed,
            registry=registry,
            registry_exploration_permyriad=0,
            stream=DeterministicStream(29, "equipment-scenario-test", "d1"),
        )
