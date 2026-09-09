from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest

from document_ocr.synthesis.cargo_language_probe import (
    CargoLanguageCaseRecord,
    CargoLanguageGenerationOutput,
    GeneratedCargoLanguageGroup,
    build_cargo_language_seed,
)
from document_ocr.synthesis.linguistic_completion_derivation import (
    _base_incurred_usage,
    _correction_usage_already_retained,
    apply_cargo_language_correction,
)
from document_ocr.synthesis.linguistic_completion_pipeline import (
    LinguisticUnitArtifact,
    LinguisticUsageTotals,
    linguistic_usage_summary_fields,
)
from document_ocr.synthesis.linguistic_probe_runtime import LinguisticUsageReceipt
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


def _usage(*response_ids: str) -> LinguisticUsageReceipt:
    requests = len(response_ids)
    return LinguisticUsageReceipt(
        requests=requests,
        providerResponseIds=response_ids,
        finishReasons=tuple("stop" for _ in response_ids),
        inputTokens=100 * requests,
        cacheReadTokens=40 * requests,
        cacheWriteTokens=10 * requests,
        outputTokens=30 * requests,
        reasoningTokens=20 * requests,
        visibleOutputTokens=10 * requests,
        estimatedCostUsd=Decimal("0.001") * requests,
    )


def test_reapplied_correction_usage_is_not_incurred_twice() -> None:
    correction_usage = _usage("resp_correction")
    correction = cast(
        CargoLanguageCaseRecord,
        SimpleNamespace(usage=correction_usage),
    )
    retained_unit = cast(
        LinguisticUnitArtifact,
        SimpleNamespace(attempts=(SimpleNamespace(usage=correction_usage),)),
    )
    base_incurred = LinguisticUsageTotals(
        requests=259,
        input_tokens=398027,
        cache_read_tokens=265037,
        cache_write_tokens=119000,
        output_tokens=73359,
        reasoning_tokens=51050,
        visible_output_tokens=22309,
        estimated_cost_usd=Decimal("0.126540190000"),
    )

    already_incurred = _correction_usage_already_retained(
        document_id="doc_" + "a" * 64,
        correction_commit_sha256="b" * 64,
        correction=correction,
        base_unit=retained_unit,
        prior_receipts=frozenset(),
    )
    incurred = base_incurred + (
        LinguisticUsageTotals()
        if already_incurred
        else LinguisticUsageTotals.from_receipt(correction_usage)
    )

    assert already_incurred is True
    assert incurred.requests == 259
    assert incurred.estimated_cost_usd == Decimal("0.126540190000")


def test_correction_usage_identity_is_exact_and_partial_overlap_fails_closed() -> None:
    base_unit = cast(
        LinguisticUnitArtifact,
        SimpleNamespace(attempts=(SimpleNamespace(usage=_usage("resp_a")),)),
    )
    correction = cast(
        CargoLanguageCaseRecord,
        SimpleNamespace(usage=_usage("resp_a", "resp_b")),
    )

    with pytest.raises(ValueError, match="partially overlap"):
        _correction_usage_already_retained(
            document_id="doc_" + "a" * 64,
            correction_commit_sha256="b" * 64,
            correction=correction,
            base_unit=base_unit,
            prior_receipts=frozenset(),
        )

    no_identifier_correction = cast(
        CargoLanguageCaseRecord,
        SimpleNamespace(
            usage=LinguisticUsageReceipt(
                requests=1,
                providerResponseIds=(),
                finishReasons=("stop",),
                inputTokens=10,
                cacheReadTokens=0,
                cacheWriteTokens=0,
                outputTokens=5,
                reasoningTokens=0,
                visibleOutputTokens=5,
                estimatedCostUsd=Decimal("0.001"),
            )
        ),
    )
    assert _correction_usage_already_retained(
        document_id="doc_" + "a" * 64,
        correction_commit_sha256="b" * 64,
        correction=no_identifier_correction,
        base_unit=base_unit,
        prior_receipts=frozenset({("doc_" + "a" * 64, "b" * 64)}),
    )


def test_legacy_lineage_without_complete_incurred_usage_fails_closed() -> None:
    retained = LinguisticUsageTotals(requests=2)
    complete = linguistic_usage_summary_fields(
        retained=retained,
        incremental=LinguisticUsageTotals(),
        incurred=LinguisticUsageTotals(requests=4),
    )
    assert _base_incurred_usage(base_summary=complete, base_retained=retained).requests == 4

    with pytest.raises(ValueError, match="predates complete incurred-usage"):
        _base_incurred_usage(
            base_summary={"requests": 4, "derivedFromRun": "legacy-base"},
            base_retained=retained,
        )
