from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.config import (
    ControlledPilotGenerationConfig,
    ControlledTransportGenerationConfig,
    SynthesisControlledPilotConfig,
    load_synthesis_controlled_pilot_config,
)
from document_ocr.synthesis.controlled_generation_pipeline import (
    ControlledGenerationError,
    _equipment_scenario_payload,
    _package_scenario_payload,
    _patch_vessel_target,
    _temperature_scenario_payload,
)
from document_ocr.synthesis.equipment_scenarios import EquipmentTypeScenario
from document_ocr.synthesis.package_scenarios import PackageTypeScenario
from document_ocr.synthesis.reefer_scenarios import (
    ReeferEquipmentIdentity,
    TemperatureScenario,
)


def _transport_config(**overrides: object) -> ControlledTransportGenerationConfig:
    values: dict[str, object] = {
        "vessel_name_method": "deferred_by_explicit_scope_v1",
        "voyage_number_method": "observed_character_class_shape_v1",
        "minimum_normalized_edit_distance": 0.2,
        "maximum_realization_attempts": 512,
        "imo_policy": "absent_without_authoritative_assigned_number_registry_v1",
    }
    values.update(overrides)
    return ControlledTransportGenerationConfig.model_validate(values, strict=True)


def test_controlled_transport_config_is_explicitly_voyage_only() -> None:
    config = _transport_config()
    assert config.voyage_number_method == "observed_character_class_shape_v1"
    assert config.vessel_name_method == "deferred_by_explicit_scope_v1"


def test_controlled_transport_can_defer_voyage_ownership_to_structured_stage() -> None:
    config = _transport_config(
        voyage_number_method="preserve_for_upstream_structured_identifier_v1"
    )

    assert config.voyage_number_method == "preserve_for_upstream_structured_identifier_v1"


def test_public_vessel_method_requires_both_pinned_registry_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    current = load_synthesis_controlled_pilot_config(
        root / "configs/synthesis/mpci_bl_combined1157_controlled_semantic_pilot50.yaml"
    ).model_dump(mode="python")
    current["generation"]["transport"]["vessel_name_method"] = (
        "public_cargo_vessel_registry_uniform_v1"
    )
    with pytest.raises(ValidationError, match="requires both pinned registry and receipt"):
        SynthesisControlledPilotConfig.model_validate(current, strict=True)

    current["inputs"]["vessel_name_registry"] = {
        "path": "data/vessel-names.jsonl",
        "sha256": "a" * 64,
        "records": 10,
    }
    current["inputs"]["vessel_name_registry_receipt"] = {
        "path": "data/registry-receipt.json",
        "sha256": "b" * 64,
    }
    parsed = SynthesisControlledPilotConfig.model_validate(current, strict=True)
    assert parsed.inputs.vessel_name_registry is not None


def test_vessel_patch_preserves_presence_and_replaces_only_the_value() -> None:
    target = {"documentPatch": {"transport": {"vesselName": "SOURCE", "voyageNumber": "1"}}}
    _patch_vessel_target(target, {"vesselName": "PUBLIC SAMPLE"})
    assert target == {
        "documentPatch": {"transport": {"vesselName": "PUBLIC SAMPLE", "voyageNumber": "1"}}
    }
    with pytest.raises(ControlledGenerationError, match="presence changed"):
        _patch_vessel_target(target, {"vesselName": None})


def test_cargo_origin_name_only_policy_matches_generation_contract() -> None:
    fields = ControlledPilotGenerationConfig.model_fields
    annotation = str(fields["cargo_origin_name_only_policy"].annotation)

    assert "replace_without_resolving_or_copying_source_name_v2" in annotation
    assert "cargo_origin_unresolved_policy" not in fields


def test_controlled_serializers_never_publish_source_surfaces() -> None:
    package = PackageTypeScenario(
        source_document_id="doc-source",
        group_id="g1",
        source_package_id="source-p1",
        projected_package_id="p1",
        source_package_position=0,
        source_package_level_count=1,
        metadata_only_package_count=0,
        role="direct_goods",
        disposition="task_facing",
        quantity=17,
        resolution="sampled_category",
        sampling_component="fit_empirical",
        category_token="PACKAGE_CARTON",
        application_code="CT",
        type_description=None,
        source_printed_surface="PRIVATE SOURCE CARTON WORDING",
        printed_surface_status="pending_text_realization",
    )
    equipment = EquipmentTypeScenario(
        source_document_id="doc-source",
        equipment_row_id="container:private-source-row",
        source_resolution="unclassified_printed_surface",
        sampling_component="fit_empirical",
        size_type_code="45R1",
        size_code="45",
        type_code="R1",
        type_family="R",
        thermal_capability="refrigerated",
        supports_temperature_setpoint=True,
        printed_surface=None,
        printed_surface_status="pending_text_realization",
        form_projection="exact_container_code",
        training_type_category_status="not_in_current_empty_task_vocabulary",
    )
    temperature = TemperatureScenario(
        identity=ReeferEquipmentIdentity(
            value="45R1",
            authority="authoritative_registry",
            is_reefer=True,
        ),
        resolution="sampled_fit_empirical_reefer_pool",
        value=-18.0,
        unit="celsius",
    )

    package_payload = _package_scenario_payload(package)
    equipment_payload = _equipment_scenario_payload(container_order=0, row=equipment)
    temperature_payload = _temperature_scenario_payload(container_order=0, row=temperature)

    assert "PRIVATE SOURCE CARTON WORDING" not in repr(package_payload)
    assert "doc-source" not in repr(package_payload)
    assert "doc-source" not in repr(equipment_payload)
    assert "private-source-row" not in repr(equipment_payload)
    assert temperature_payload["equipmentIdentity"] == "45R1"
