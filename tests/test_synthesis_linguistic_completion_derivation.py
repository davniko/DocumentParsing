from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from document_ocr.synthesis.cargo_language_probe import (
    CargoLanguageGenerationOutput,
    GeneratedCargoLanguageGroup,
    build_cargo_language_seed,
)
from document_ocr.synthesis.linguistic_completion_derivation import (
    apply_cargo_language_correction,
)
from document_ocr.synthesis.semantic_completion_pipeline import SemanticCompletionPlanRow


def _case() -> tuple[SemanticCompletionPlanRow, dict[str, Any], CargoLanguageGenerationOutput]:
    document_id = "doc_" + "a" * 64
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "route": {"portOfLoading": {"name": "Cebu"}},
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "SOURCE STEEL",
                    "additionalInformation": ["SOURCE GRADE"],
                }
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 10,
                    "typeCategory": "PACKAGE_COIL",
                }
            ],
        },
    }
    plan = cast(
        SemanticCompletionPlanRow,
        SimpleNamespace(
            base_document_id=document_id,
            scenario_id="syn_test",
            target=target,
            cargo_realizations=(
                SimpleNamespace(
                    cargo_group_id="g1",
                    description="Steel coils",
                    output_hs_code="720839",
                    hs6="720839",
                    chapter_description="IRON AND STEEL",
                    heading_description="Flat-rolled products",
                    thermal_profile=None,
                ),
            ),
            equipment_realizations=(),
            flashpoint_realizations=(),
            upstream_dangerous_goods_realizations=(),
        ),
    )
    output = CargoLanguageGenerationOutput(
        cargoGroups=(
            GeneratedCargoLanguageGroup(
                groupId="g1",
                description="STEEL COILS",
                additionalInformation=("EXPORT GRADE",),
                marksAndNumbers=(),
                handlingInstructions=(),
            ),
        )
    )
    return plan, target, output


def test_cargo_correction_changes_only_linguistic_leaves() -> None:
    plan, target, output = _case()
    seed = build_cargo_language_seed(case_index=0, plan=plan)

    corrected = apply_cargo_language_correction(
        base_document_id=plan.base_document_id,
        base_target=target,
        base_output=CargoLanguageGenerationOutput(
            cargoGroups=(
                GeneratedCargoLanguageGroup(
                    groupId="g1",
                    description="SOURCE STEEL",
                    additionalInformation=("SOURCE GRADE",),
                    marksAndNumbers=(),
                    handlingInstructions=(),
                ),
            )
        ).model_dump(mode="json"),
        seed=seed,
        output=output,
    )

    group = corrected["documentPatch"]["cargoGroups"][0]
    assert group["description"] == "STEEL COILS"
    assert group["additionalInformation"] == ["EXPORT GRADE"]
    assert corrected["documentPatch"]["cargoPackages"] == target["documentPatch"][
        "cargoPackages"
    ]
    assert target["documentPatch"]["cargoGroups"][0]["description"] == "SOURCE STEEL"


def test_cargo_correction_rejects_a_style_reference_not_in_the_base_target() -> None:
    plan, target, output = _case()
    seed = build_cargo_language_seed(case_index=0, plan=plan)
    target["documentPatch"]["cargoGroups"][0]["additionalInformation"] = [
        "DIFFERENT SOURCE SLOT"
    ]

    with pytest.raises(ValueError, match="additionalInformation differs"):
        apply_cargo_language_correction(
            base_document_id=plan.base_document_id,
            base_target=target,
            base_output=CargoLanguageGenerationOutput(
                cargoGroups=(
                    GeneratedCargoLanguageGroup(
                        groupId="g1",
                        description="SOURCE STEEL",
                        additionalInformation=("SOURCE GRADE",),
                        marksAndNumbers=(),
                        handlingInstructions=(),
                    ),
                )
            ).model_dump(mode="json"),
            seed=seed,
            output=output,
        )
