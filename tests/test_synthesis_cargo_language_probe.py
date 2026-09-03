from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import yaml
from pydantic import ValidationError

from document_ocr.synthesis.cargo_language_probe import (
    CargoLanguageGenerationOutput,
    GeneratedCargoLanguageGroup,
    build_cargo_language_seed,
    validate_cargo_language,
)
from document_ocr.synthesis.config import (
    SynthesisCargoLanguageProbeConfig,
    load_synthesis_cargo_language_probe_config,
    load_synthesis_linguistic_probe_analysis_config,
)
from document_ocr.synthesis.semantic_completion_pipeline import SemanticCompletionPlanRow


def _plan() -> SemanticCompletionPlanRow:
    return cast(
        SemanticCompletionPlanRow,
        SimpleNamespace(
            base_document_id="doc_" + "a" * 64,
            scenario_id="syn_test_cargo",
            target={
                "documentPatch": {
                    "route": {
                        "portOfLoading": {"name": "Cebu"},
                        "portOfDischarge": {"name": "Alexandria"},
                    },
                    "cargoGroups": [
                        {
                            "groupId": "g1",
                            "description": "SOURCE FURNITURE PARTS",
                            "additionalInformation": ["SOURCE GRADE 22"],
                            "marksAndNumbers": ["N/M", "SOURCE-100"],
                            "handlingInstructions": ["SOURCE COLD INSTRUCTION"],
                            "grossWeight": {"value": 1200.5, "unit": "kilogram"},
                        }
                    ],
                    "cargoPackages": [
                        {
                            "groupId": "g1",
                            "packageId": "p1",
                            "quantity": 40,
                            "typeCategory": "PACKAGE_CARTON",
                        }
                    ],
                    "cargoAllocationGroups": [
                        {
                            "groupId": "g1",
                            "coverage": "container_membership_only",
                            "packageIds": [],
                            "allocations": [{"containerNumber": "ABCU1234560"}],
                        }
                    ],
                }
            },
            cargo_realizations=(
                SimpleNamespace(
                    cargo_group_id="g1",
                    description="Frozen skipjack tuna",
                    output_hs_code="03034310",
                    hs6="030343",
                    chapter_description="FISH AND CRUSTACEANS",
                    heading_description="Fish, frozen",
                    thermal_profile="FROZEN",
                ),
            ),
            equipment_realizations=(
                SimpleNamespace(
                    container_number="ABCU1234560",
                    size_category="FORTY_FOOT_HIGH_CUBE",
                    type_category="REFRIGERATED",
                    temperature_value_celsius=-18.0,
                ),
            ),
            flashpoint_realizations=(),
            upstream_dangerous_goods_realizations=(),
        ),
    )


def test_probe_config_is_strict_and_pins_five_single_request_cases() -> None:
    path = Path("configs/synthesis/mpci_bl_cargo_language_luna_high_probe5.yaml")
    config = load_synthesis_cargo_language_probe_config(path)
    assert len(config.cases) == 5
    assert config.workflow.requests_per_case == 1
    assert config.workflow.structured_output_retries == 0
    assert config.workflow.provider_native_strict_json_schema is True
    assert config.workflow.exclude_categorical_printed_surfaces is True
    assert config.provider.store_responses is True

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["workflow"]["structured_output_retries"] = 1
    with pytest.raises(ValidationError, match="structured_output_retries"):
        SynthesisCargoLanguageProbeConfig.model_validate(value, strict=True)


def test_probe50_config_has_fifty_unique_single_request_cases() -> None:
    config = load_synthesis_cargo_language_probe_config(
        Path("configs/synthesis/mpci_bl_cargo_language_luna_high_probe50.yaml")
    )

    assert len(config.cases) == 50
    assert len({case.document_id for case in config.cases}) == 50
    assert config.workflow.requests_per_case == 1
    assert config.workflow.structured_output_retries == 0
    assert config.provider.max_output_tokens == 8192


def test_output_schema_covers_observed_forty_mark_slots() -> None:
    group = GeneratedCargoLanguageGroup(
        groupId="g1",
        description="STUD-LINK CHAIN",
        additionalInformation=(),
        marksAndNumbers=tuple(f"MARK-{index:02d}" for index in range(40)),
        handlingInstructions=(),
    )

    assert len(group.marksAndNumbers) == 40
    schema = GeneratedCargoLanguageGroup.model_json_schema(mode="validation")
    assert schema["properties"]["marksAndNumbers"]["maxItems"] == 64


def test_linguistic_probe_analysis_pins_both_fifty_case_runs() -> None:
    config = load_synthesis_linguistic_probe_analysis_config(
        Path("configs/synthesis/mpci_bl_linguistic_probe50_eda.yaml")
    )

    assert config.party.results.records == 50
    assert config.cargo.results.records == 50
    assert config.cargo_contract_probe.results.records == 1


def test_seed_preserves_text_topology_and_excludes_categorical_printed_surfaces() -> None:
    seed = build_cargo_language_seed(case_index=0, plan=_plan())
    group = seed.cargoGroups[0]
    assert group.goodsIdentities[0].description == "Frozen skipjack tuna"
    assert group.structuredFacts.packages[0].typeCategory == "PACKAGE_CARTON"
    assert group.structuredFacts.equipment[0].typeCategory == "REFRIGERATED"
    assert group.fieldContract.descriptionPresent is True
    assert [row.action for row in group.fieldContract.marksAndNumbersSlots] == [
        "preserve_literal",
        "generate",
    ]
    assert seed.excludedFinalPatchingFields == (
        "packagePrintedSurfaces",
        "containerPrintedSurfaces",
        "hsCodePrintedSurfaces",
    )
    assert set(CargoLanguageGenerationOutput.model_json_schema()["properties"]) == {
        "cargoGroups"
    }


def test_validation_enforces_semantics_topology_and_generic_marks() -> None:
    seed = build_cargo_language_seed(case_index=0, plan=_plan())
    output = CargoLanguageGenerationOutput(
        cargoGroups=(
            GeneratedCargoLanguageGroup(
                groupId="g1",
                description="FROZEN SKIPJACK TUNA LOINS",
                additionalInformation=("FOOD GRADE, LOT K72",),
                marksAndNumbers=("N/M", "COLDSEA-742"),
                handlingInstructions=("KEEP FROZEN AT -18°C",),
            ),
        )
    )
    validation = validate_cargo_language(seed=seed, output=output)
    assert validation.passed is True
    assert all(validation.checks.values())

    invalid = output.model_copy(
        update={
            "cargoGroups": (
                output.cargoGroups[0].model_copy(
                    update={
                        "description": "OFFICE CHAIRS",
                        "marksAndNumbers": ("NO MARKS", "SOURCE-100"),
                    }
                ),
            )
        }
    )
    failed = validate_cargo_language(seed=seed, output=invalid)
    assert failed.passed is False
    assert failed.checks["group_1_literal_marks_preserved"] is False
    assert failed.checks["group_1_substantive_marks_anonymized"] is False
    assert failed.checks["group_1_description_semantic_coverage"] is False


def test_short_goods_name_is_not_lost_behind_long_taxonomic_qualifiers() -> None:
    seed = build_cargo_language_seed(case_index=0, plan=_plan())
    output = CargoLanguageGenerationOutput(
        cargoGroups=(
            GeneratedCargoLanguageGroup(
                groupId="g1",
                description="FROZEN COD",
                additionalInformation=("FOOD GRADE, LOT K72",),
                marksAndNumbers=("N/M", "COLDSEA-742"),
                handlingInstructions=("KEEP FROZEN AT -18°C",),
            ),
        )
    )

    validation = validate_cargo_language(seed=seed, output=output)

    assert validation.checks["group_1_description_semantic_coverage"] is True


def test_hierarchical_heading_nouns_validate_complete_fragment_realization() -> None:
    seed = build_cargo_language_seed(case_index=0, plan=_plan())
    identity = seed.cargoGroups[0].goodsIdentities[0].model_copy(
        update={
            "description": "Of polyamides",
            "headingDescription": (
                "Other plates, sheets, film, foil and strip, of plastics, non-cellular"
            ),
            "thermalProfile": None,
        }
    )
    seed = seed.model_copy(
        update={
            "cargoGroups": (
                seed.cargoGroups[0].model_copy(
                    update={"goodsIdentities": (identity,)}
                ),
            )
        }
    )
    output = CargoLanguageGenerationOutput(
        cargoGroups=(
            GeneratedCargoLanguageGroup(
                groupId="g1",
                description="Polyamide plastic film and sheets",
                additionalInformation=("FOOD GRADE, LOT K72",),
                marksAndNumbers=("N/M", "COLDSEA-742"),
                handlingInstructions=("HANDLE WITH CARE",),
            ),
        )
    )

    validation = validate_cargo_language(seed=seed, output=output)

    assert validation.checks["group_1_description_semantic_coverage"] is True


def test_ungrounded_auxiliary_slots_can_be_explicitly_removed_without_filler() -> None:
    group = GeneratedCargoLanguageGroup(
        groupId="g1",
        description="POLYURETHANE SHEETS",
        additionalInformation=(None, None),
        marksAndNumbers=(),
        handlingInstructions=(),
    )
    assert group.additionalInformation == (None, None)
    schema = GeneratedCargoLanguageGroup.model_json_schema(mode="validation")
    item_schema = schema["properties"]["additionalInformation"]["items"]
    assert {row.get("type") for row in item_schema["anyOf"]} == {"string", "null"}
