from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading_v5 import (
    CONTAINER_SIZE_CATEGORIES,
    CONTAINER_TYPE_CATEGORIES,
)
from document_ocr.synthesis.container_semantics import (
    SourceEquipmentObservation,
    build_equipment_semantic_support,
    canonical_equipment_surface,
    review_source_equipment_surface,
    sample_equipment_semantic,
)
from document_ocr.synthesis.country_registry import load_iso_country_registry
from document_ocr.synthesis.flashpoint_scenarios import sample_flashpoint
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.hs_registry import HsGlobalSubheading
from document_ocr.synthesis.semantic_completion_pipeline import (
    SemanticCompletionPlanRow,
    _apply_equipment_semantics,
    _ranked_quota,
)
from document_ocr.synthesis.thermal_goods import classify_thermal_hs
from document_ocr.synthesis.transport_auxiliary import (
    imo_check_digit,
    maritime_flag_country_codes,
    sample_imo_number,
    sample_vessel_flag,
)
from document_ocr.synthesis.world_port_registry import load_pinned_world_port_records


def _stream(identity: str) -> DeterministicStream:
    return DeterministicStream(
        seed=20260901, namespace="semantic-completion-test", identity=identity
    )


@pytest.mark.parametrize(
    ("surface", "temperature", "size", "type_category", "operation"),
    [
        ("40HR", True, "FORTY_FOOT_HIGH_CUBE", "REFRIGERATED", "active"),
        ("40 RH", True, "FORTY_FOOT_HIGH_CUBE", "REFRIGERATED", "active"),
        ("40 RF", True, "FORTY_FOOT_STANDARD_HEIGHT", "REFRIGERATED", "active"),
        ("40NOR", False, "FORTY_FOOT_HIGH_CUBE", "REFRIGERATED", "non_operating"),
        ("20' REEFER", True, "TWENTY_FOOT_STANDARD_HEIGHT", "REFRIGERATED", "active"),
        ("40HQ", False, "FORTY_FOOT_HIGH_CUBE", "GENERAL_PURPOSE", "not_indicated"),
        (
            "CONTAINER: 1 X 20FT GENERAL PURPOSE",
            False,
            "TWENTY_FOOT_STANDARD_HEIGHT",
            "GENERAL_PURPOSE",
            "not_indicated",
        ),
    ],
)
def test_reviewed_equipment_grammar_resolves_proven_source_families(
    surface: str,
    temperature: bool,
    size: str,
    type_category: str,
    operation: str,
) -> None:
    row = review_source_equipment_surface(surface, temperature_present=temperature)
    assert row.resolution == "reviewed_source_grammar"
    assert row.size_category == size
    assert row.type_category == type_category
    assert row.thermal_operation == operation


def test_every_canonical_equipment_surface_round_trips_to_its_semantic_categories() -> None:
    for size in CONTAINER_SIZE_CATEGORIES:
        for type_category in CONTAINER_TYPE_CATEGORIES:
            surface = canonical_equipment_surface(size, type_category)
            reviewed = review_source_equipment_surface(
                surface,
                temperature_present=type_category == "REFRIGERATED",
            )
            assert reviewed.resolution == "reviewed_source_grammar"
            assert reviewed.size_category == size
            assert reviewed.type_category == type_category


@pytest.mark.parametrize("surface", ["40RA", "40RK", "40RQ", "40RO"])
def test_unverified_carrier_reefer_height_codes_fail_closed(surface: str) -> None:
    row = review_source_equipment_surface(surface, temperature_present=True)
    assert row.resolution == "unresolved_source_surface"
    assert row.type_category == "REFRIGERATED"
    assert row.size_category is None
    assert row.review_rule == "reviewed_reefer_type_with_unresolved_carrier_height_code"


def test_equipment_sampling_preserves_source_type_marginal_and_supports_override() -> None:
    support = build_equipment_semantic_support(
        (
            SourceEquipmentObservation("d1", "0", "40HQ", False),
            SourceEquipmentObservation("d2", "0", "40HQ", False),
            SourceEquipmentObservation("d3", "0", "20GP", False),
            SourceEquipmentObservation("d4", "0", "40RH", True),
        )
    )
    empirical = sample_equipment_semantic(
        support=support,
        active_temperature=True,
        stream=_stream("active"),
    )
    assert empirical.size_category == "FORTY_FOOT_HIGH_CUBE"
    assert empirical.type_category == "REFRIGERATED"
    assert empirical.application_code == "45RE"
    configured = sample_equipment_semantic(
        support=support,
        active_temperature=False,
        stream=_stream("configured"),
        configured_joint_weights={"FORTY_FIVE_FOOT_HIGH_CUBE|OPEN_TOP": 1},
    )
    assert configured.application_code == "55UT"
    assert configured.sampling_method == "configured_joint_override"


def _hs(*, code: str, heading: str, leaf: str) -> HsGlobalSubheading:
    return HsGlobalSubheading.model_validate(
        {
            "edition": "HS2022",
            "valid_from": date(2022, 1, 1),
            "chapter_code": code[:2],
            "chapter_description": "Meat and edible meat offal",
            "heading_code": code[:4],
            "heading_description": heading,
            "code": code,
            "source_description": leaf,
            "description": leaf,
            "description_authority": (
                "UK Global Tariff implementation text; not asserted as canonical WCO nomenclature"
            ),
        },
        strict=True,
    )


def test_nearest_hs_level_prevents_fresh_goods_from_inheriting_frozen_parent() -> None:
    fresh_lamb = _hs(
        code="020410",
        heading="Meat of sheep or goats, fresh, chilled or frozen",
        leaf="Carcases and half-carcases of lamb, fresh or chilled",
    )
    frozen_lamb = _hs(
        code="020430",
        heading="Meat of sheep or goats, fresh, chilled or frozen",
        leaf="Carcases and half-carcases of lamb, frozen",
    )
    assert classify_thermal_hs(fresh_lamb) == "CHILLED"
    assert classify_thermal_hs(frozen_lamb) == "FROZEN"


def test_imo_generation_is_checksum_valid_unique_and_records_attempts() -> None:
    first = sample_imo_number(
        stream=_stream("imo"),
        excluded_values=("9305714",),
        used_values=set(),
        maximum_attempts=32,
        leading_digit_weights={str(value): 1 for value in range(1, 10)},
    )
    assert len(first.value) == 7
    assert int(first.value[-1]) == imo_check_digit(first.value[:6])
    assert first.value != "9305714"
    assert first.attempts >= 1


def test_sparse_document_quota_uses_full_selected_population() -> None:
    selected = _ranked_quota(
        document_ids=tuple(f"doc_{index}" for index in range(900)),
        permyriad=95,
        seed=1,
        namespace="imo",
        population_size=1157,
    )
    assert len(selected) == 11
    with pytest.raises(ValueError, match="exceeds the eligible"):
        _ranked_quota(
            document_ids=("only-eligible",),
            permyriad=5000,
            seed=1,
            namespace="invalid",
            population_size=10,
        )


def _thermal_completion_plan(*, second_setpoint: float) -> dict[str, object]:
    target: dict[str, object] = {"documentPatch": {}}
    return {
        "schema_version": 1,
        "scenario_id": "syn_1",
        "base_document_id": "doc_1",
        "template_id": "template_1",
        "upstream_target_sha256": "0" * 64,
        "target_task": "bill_of_lading_relation_explicit_v5",
        "target": target,
        "target_sha256": sha256_bytes(canonical_json_bytes(target)),
        "changes": (),
        "cargo_realizations": (
            {
                "cargo_group_id": "g1",
                "identity_order": 0,
                "semantic_source": "thermal_hs_registry",
                "hs6": "030368",
                "output_hs_code": "030368",
                "chapter_description": "Fish",
                "heading_description": "Frozen fish",
                "description": "Frozen blue whiting",
                "thermal_profile": "FROZEN",
                "associated_container_numbers": ("C1", "C2"),
            },
        ),
        "package_goods_realizations": (),
        "equipment_realizations": (
            {
                "container_number": "C1",
                "size_category": "FORTY_FOOT_HIGH_CUBE",
                "type_category": "REFRIGERATED",
                "application_code": "45RE",
                "active_temperature": True,
                "temperature_value_celsius": -18.0,
                "sampling_method": "source_type_marginal_size_conditional",
                "linked_thermal_cargo_groups": ("g1",),
            },
            {
                "container_number": "C2",
                "size_category": "FORTY_FOOT_HIGH_CUBE",
                "type_category": "REFRIGERATED",
                "application_code": "45RE",
                "active_temperature": True,
                "temperature_value_celsius": second_setpoint,
                "sampling_method": "source_type_marginal_size_conditional",
                "linked_thermal_cargo_groups": ("g1",),
            },
        ),
        "transport_auxiliary": {},
        "flashpoint_realizations": (),
        "upstream_dangerous_goods_realizations": (),
        "remaining_blockers": ("raw_ocr_patch_planning_and_execution",),
        "status": "resolved_non_linguistic_v5_pending_text_realization",
        "training_eligible": False,
    }


def test_thermal_cargo_group_requires_one_shared_setpoint_across_containers() -> None:
    valid = SemanticCompletionPlanRow.model_validate(
        _thermal_completion_plan(second_setpoint=-18.0), strict=True
    )
    assert len(valid.equipment_realizations) == 2
    with pytest.raises(ValidationError, match="one shared setpoint"):
        SemanticCompletionPlanRow.model_validate(
            _thermal_completion_plan(second_setpoint=-20.0), strict=True
        )


def test_equipment_application_propagates_group_setpoint_to_every_container() -> None:
    support = build_equipment_semantic_support(
        (SourceEquipmentObservation("source", "0", "40RH", True),)
    )

    class _EquipmentRegistry:
        @staticmethod
        def classify(code: str) -> SimpleNamespace:
            return SimpleNamespace(type=SimpleNamespace(type_code=code[2:]))

    config = SimpleNamespace(
        generation=SimpleNamespace(
            equipment=SimpleNamespace(
                configured_active_joint_weights={},
                configured_inactive_joint_weights={},
            ),
            thermal=SimpleNamespace(
                frozen_minimum_celsius=-24.0,
                frozen_maximum_celsius=-18.0,
                chilled_minimum_celsius=-3.0,
                chilled_maximum_celsius=5.5,
                step_celsius=0.5,
            ),
        )
    )
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "C1"},
                {"containerNumber": "C2"},
            ]
        }
    }
    rows = _apply_equipment_semantics(
        target=target,
        profile_by_group={"g1": "FROZEN"},
        allocations={"g1": ("C1", "C2")},
        support=support,
        equipment_registry=_EquipmentRegistry(),
        stream=_stream("shared-setpoint"),
        config=config,
    )
    assert len(rows) == 2
    assert {row.temperature_value_celsius for row in rows} == {
        target["documentPatch"]["containers"][0]["temperatureSetpoint"]["value"]
    }
    assert (
        target["documentPatch"]["containers"][0]["temperatureSetpoint"]
        == target["documentPatch"]["containers"][1]["temperatureSetpoint"]
    )


@pytest.mark.parametrize(
    ("packing_group", "minimum", "maximum"),
    [("MEDIUM_DANGER", -40.0, 22.5), ("LOW_DANGER", 23.0, 60.0)],
)
def test_primary_class3_flashpoint_respects_packing_group_bounds(
    packing_group: str, minimum: float, maximum: float
) -> None:
    sampled = sample_flashpoint(
        proper_shipping_name="FLAMMABLE LIQUID, N.O.S.",
        hazard_category="FLAMMABLE_LIQUIDS",
        subsidiary_hazard_categories=(),
        packing_group_category=packing_group,
        stream=_stream(packing_group),
        presence_permyriad={
            "CLASS3_FLAMMABLE_LIQUID": 10_000,
            "NON_CLASS3_EXPLICIT_LIQUID": 0,
            "DESENSITIZED_FLAMMABLE_SOLID": 0,
        },
        class3_minimum_celsius=-40.0,
        class3_maximum_celsius=60.0,
        non_class3_liquid_minimum_celsius=60.5,
        non_class3_liquid_maximum_celsius=160.0,
        desensitized_solid_minimum_celsius=-20.0,
        desensitized_solid_maximum_celsius=60.0,
        step_celsius=0.5,
    )
    assert sampled is not None
    assert minimum <= sampled.value <= maximum


@pytest.mark.parametrize(
    ("proper_name", "hazard", "expected_eligibility", "minimum", "maximum"),
    [
        (
            "CORROSIVE LIQUID, ACIDIC, ORGANIC, N.O.S.",
            "CORROSIVE_SUBSTANCES",
            "NON_CLASS3_EXPLICIT_LIQUID",
            60.5,
            160.0,
        ),
        (
            "NITROCELLULOSE, DAMPED WITH WATER",
            "FLAMMABLE_SOLIDS",
            "DESENSITIZED_FLAMMABLE_SOLID",
            -20.0,
            60.0,
        ),
    ],
)
def test_non_class3_flashpoints_require_explicit_physical_form(
    proper_name: str,
    hazard: str,
    expected_eligibility: str,
    minimum: float,
    maximum: float,
) -> None:
    sampled = sample_flashpoint(
        proper_shipping_name=proper_name,
        hazard_category=hazard,
        subsidiary_hazard_categories=(),
        packing_group_category="MEDIUM_DANGER",
        stream=_stream(expected_eligibility),
        presence_permyriad={
            "CLASS3_FLAMMABLE_LIQUID": 10_000,
            "NON_CLASS3_EXPLICIT_LIQUID": 10_000,
            "DESENSITIZED_FLAMMABLE_SOLID": 10_000,
        },
        class3_minimum_celsius=-40.0,
        class3_maximum_celsius=60.0,
        non_class3_liquid_minimum_celsius=60.5,
        non_class3_liquid_maximum_celsius=160.0,
        desensitized_solid_minimum_celsius=-20.0,
        desensitized_solid_maximum_celsius=60.0,
        step_celsius=0.5,
    )
    assert sampled is not None
    assert sampled.eligibility == expected_eligibility
    assert minimum <= sampled.value <= maximum


@pytest.mark.parametrize(
    ("proper_name", "hazard"),
    [
        ("ZINC NITRATE", "OXIDIZING_SUBSTANCES_AND_ORGANIC_PEROXIDES"),
        ("CORROSIVE MIXTURE, N.O.S.", "CORROSIVE_SUBSTANCES"),
        ("NITROCELLULOSE", "FLAMMABLE_SOLIDS"),
    ],
)
def test_unqualified_physical_form_never_receives_flashpoint(proper_name: str, hazard: str) -> None:
    sampled = sample_flashpoint(
        proper_shipping_name=proper_name,
        hazard_category=hazard,
        subsidiary_hazard_categories=(),
        packing_group_category="MEDIUM_DANGER",
        stream=_stream("not-eligible"),
        presence_permyriad={
            "CLASS3_FLAMMABLE_LIQUID": 10_000,
            "NON_CLASS3_EXPLICIT_LIQUID": 10_000,
            "DESENSITIZED_FLAMMABLE_SOLID": 10_000,
        },
        class3_minimum_celsius=-40.0,
        class3_maximum_celsius=60.0,
        non_class3_liquid_minimum_celsius=60.5,
        non_class3_liquid_maximum_celsius=160.0,
        desensitized_solid_minimum_celsius=-20.0,
        desensitized_solid_maximum_celsius=60.0,
        step_celsius=0.5,
    )
    assert sampled is None


def test_real_equipment_audit_has_reviewed_temperature_distribution() -> None:
    corpus = Path(
        "artifacts/kie-training/datasets/"
        "mpci-bl-combined1157-task-facing-package-categories-v2/records.jsonl"
    )
    if not corpus.is_file():
        pytest.skip("the immutable 1,157-record corpus is not present")
    observations: list[SourceEquipmentObservation] = []
    documents_with_temperature: set[str] = set()
    with corpus.open("rb") as stream:
        for raw in stream:
            row = json.loads(raw)
            document_id = row["documentId"]
            for order, container in enumerate(
                row["target"]["documentPatch"].get("containers") or []
            ):
                temperature = container.get("temperatureSetpoint") is not None
                if temperature:
                    documents_with_temperature.add(document_id)
                observations.append(
                    SourceEquipmentObservation(
                        document_id=document_id,
                        row_id=str(order),
                        printed_surface=container.get("typeDescription"),
                        temperature_present=temperature,
                    )
                )
    support = build_equipment_semantic_support(observations)
    assert len(documents_with_temperature) == 45
    assert support.audit == support.audit.__class__(
        input_rows=2115,
        type_resolved_rows=1937,
        resolved_rows=1926,
        unresolved_rows=189,
        temperature_rows=53,
        type_resolved_temperature_rows=52,
        resolved_temperature_rows=44,
        non_operating_reefer_rows=7,
        type_support_rows=8,
        joint_support_rows=16,
    )


def test_real_maritime_flag_support_uses_only_pinned_iso_countries() -> None:
    iso = Path("data/registries/countries/iso-codes-4.9.0-1/iso_3166-1.json")
    ports = Path(
        "artifacts/kie-synthesis/registries/nga-world-port-index-current-v1/port-whitelist.jsonl"
    )
    if not iso.is_file() or not ports.is_file():
        pytest.skip("the pinned ISO and maritime-port registries are not present")
    countries = load_iso_country_registry(
        iso_path=iso,
        iso_sha256="f7dc5542a692ad8e23b9b85a6a1800a63f7a05e5246065fac1ede04ed209ce00",
    )
    world_ports = load_pinned_world_port_records(
        ports,
        expected_sha256="b7d9d33450992bf13ee6fe5d137de4bc768a2df9dcb7d9840f7232a1974563fb",
        expected_records=3032,
    )
    codes = maritime_flag_country_codes(world_ports)
    assert len(codes) == 186
    sampled = sample_vessel_flag(
        country_codes=codes,
        countries=countries,
        stream=_stream("flag"),
    )
    assert sampled.country_code in codes
    assert sampled.printed_country == countries.printable_name(sampled.country_code)
