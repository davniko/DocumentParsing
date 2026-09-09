from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.usage import RequestUsage

from document_ocr.synthesis.config import (
    SynthesisLinguisticCompletionConfig,
    load_synthesis_linguistic_completion_config,
)
from document_ocr.synthesis.linguistic_completion_pipeline import (
    GeneratedPartyContactsV2,
    GeneratedPartyIdentityV2,
    LinguisticAttemptRecord,
    LinguisticUnitArtifact,
    LinguisticUsageTotals,
    PartyCompletionOutput,
    SourceSensitiveInventory,
    _dynamic_cargo_output_model,
    _dynamic_party_output_model,
    _prior_incurred_usage,
    _resume_unit_passes_current_contract,
    _run_unit,
    _validate_persisted_output,
    build_document_linguistic_plan,
    linguistic_summary_incurred_usage,
    linguistic_unit_usage_totals,
    linguistic_usage_summary_fields,
    validate_party_completion,
)
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    openai_responses_settings,
)
from document_ocr.synthesis.semantic_completion_pipeline import SemanticCompletionPlanRow


def test_persisted_json_arrays_rehydrate_as_strict_tuple_fields() -> None:
    payload = {
        "party": {
            "partyRole": "shipper",
            "name": "FICTIONAL EXPORTS LTD",
            "address": "12 HARBOUR ROAD",
            "city": "CEBU CITY",
            "country": "PHILIPPINES",
            "contactDetails": {
                "contactName": None,
                "phoneNumbers": ["+63 32 555 0194"],
                "emailAddresses": [],
                "websiteUrls": [],
            },
        }
    }

    with pytest.raises(ValidationError, match="valid tuple"):
        PartyCompletionOutput.model_validate(payload, strict=True)
    hydrated = _validate_persisted_output(PartyCompletionOutput, payload)

    assert hydrated.party.contactDetails.phoneNumbers == ("+63 32 555 0194",)


def _usage_receipt(
    *,
    response_id: str,
    requests: int = 1,
    input_tokens: int = 100,
    cache_read_tokens: int = 40,
    cache_write_tokens: int = 10,
    output_tokens: int = 30,
    reasoning_tokens: int = 20,
    cost: str = "0.001",
) -> LinguisticUsageReceipt:
    return LinguisticUsageReceipt(
        requests=requests,
        providerResponseIds=(response_id,),
        finishReasons=tuple("stop" for _ in range(requests)),
        inputTokens=input_tokens,
        cacheReadTokens=cache_read_tokens,
        cacheWriteTokens=cache_write_tokens,
        outputTokens=output_tokens,
        reasoningTokens=reasoning_tokens,
        visibleOutputTokens=output_tokens - reasoning_tokens,
        estimatedCostUsd=Decimal(cost),
    )


def _usage_unit(*receipts: LinguisticUsageReceipt) -> LinguisticUnitArtifact:
    now = datetime(2026, 9, 8, tzinfo=UTC)
    attempts = tuple(
        LinguisticAttemptRecord(
            attempt=index,
            schemaMode="static",
            outputSchemaSha256="a" * 64,
            userPromptSha256="b" * 64,
            startedAt=now,
            completedAt=now,
            durationMs=0,
            status="success" if index == len(receipts) else "validation_failed",
            output={"value": index},
            checks={"accepted": index == len(receipts)},
            usage=receipt,
            errorType=None,
            errorMessage=None,
        )
        for index, receipt in enumerate(receipts, start=1)
    )
    return LinguisticUnitArtifact(
        schemaVersion=1,
        stage="party_identity",
        unitId="party-shipper-1",
        attempts=attempts,
        transcripts=tuple({"attempt": row.attempt} for row in attempts),
        selectedAttempt=len(attempts),
        selectedOutputSha256="c" * 64,
        status="success",
    )


def test_usage_accounting_keeps_retained_incremental_and_incurred_totals_distinct() -> None:
    retained = linguistic_unit_usage_totals(
        (
            _usage_unit(
                _usage_receipt(response_id="resp_old_failed", cost="0.002"),
                _usage_receipt(response_id="resp_retained", cost="0.003"),
            ),
        )
    )
    reused = LinguisticUsageTotals.from_receipt(
        _usage_receipt(response_id="resp_old_failed", cost="0.002")
    )
    incremental = retained - reused
    prior_incurred = LinguisticUsageTotals(
        requests=7,
        input_tokens=700,
        cache_read_tokens=280,
        cache_write_tokens=70,
        output_tokens=210,
        reasoning_tokens=140,
        visible_output_tokens=70,
        estimated_cost_usd=Decimal("0.014"),
    )
    fields = linguistic_usage_summary_fields(
        retained=retained,
        incremental=incremental,
        incurred=prior_incurred + incremental,
    )

    assert fields["requests"] == 2
    assert fields["incrementalRequests"] == 1
    assert fields["incurredRequests"] == 8
    assert fields["cacheWriteTokens"] == 20
    assert fields["incrementalCacheWriteTokens"] == 10
    assert fields["incurredCacheWriteTokens"] == 80
    assert fields["estimatedCostUsd"] == "0.005"
    assert fields["incrementalEstimatedCostUsd"] == "0.003"
    assert fields["incurredEstimatedCostUsd"] == "0.017"
    assert linguistic_summary_incurred_usage(fields) == prior_incurred + incremental


def test_usage_accounting_rejects_partial_or_negative_contracts() -> None:
    with pytest.raises(ValueError, match="partial incurred-usage"):
        linguistic_summary_incurred_usage({"incurredRequests": 1})
    with pytest.raises(ValueError, match="negative"):
        LinguisticUsageTotals(requests=1) - LinguisticUsageTotals(requests=2)


def test_resume_rejects_legacy_lineage_with_unprovable_discarded_attempts(
    tmp_path: Path,
) -> None:
    generation = tmp_path / "generation"
    generation.mkdir()
    (generation / "summary.json").write_text(
        json.dumps({"derivedFromRun": "legacy-parent", "requests": 10}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="predates complete incurred-usage"):
        _prior_incurred_usage(resume_root=tmp_path, document_plans=())


def _plan() -> SemanticCompletionPlanRow:
    consignee = {
        "name": "SOURCE IMPORTS LTD",
        "address": "OLD QUAY 12",
        "city": "Alexandria",
        "country": "Egypt",
    }
    return cast(
        SemanticCompletionPlanRow,
        SimpleNamespace(
            base_document_id="doc_" + "a" * 64,
            scenario_id="syn_test_linguistic",
            target_sha256="b" * 64,
            remaining_blockers=(
                "party_identity_and_contact_anonymization",
                "cargo_and_auxiliary_text_realization",
                "raw_ocr_patch_planning_and_execution",
            ),
            target={
                "documentPatch": {
                    "parties": {
                        "shipper": {
                            "name": "SOURCE EXPORTS LTD",
                            "city": "Cebu City",
                            "country": "Philippines",
                        },
                        "consignee": consignee,
                        "notifyParties": [
                            {"sameAs": "consignee"},
                            dict(consignee),
                            {
                                "sameAs": "consignee",
                                "contactDetails": {
                                    "phoneNumbers": ["+20 3 555 0194"],
                                },
                            },
                            {"city": "Giza", "country": "Egypt"},
                        ],
                    },
                    "route": {
                        "portOfLoading": {"name": "Cebu", "country": "Philippines"},
                        "portOfDischarge": {
                            "name": "Alexandria",
                            "country": "Egypt",
                        },
                        "placeOfDelivery": {"name": "Giza", "country": "Egypt"},
                    },
                    "cargoGroups": [
                        {
                            "groupId": "g1",
                            "description": "SOURCE FROZEN FISH",
                            "marksAndNumbers": ["N/M", "SOURCE-100"],
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
                    "cargoAllocationGroups": [],
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
            equipment_realizations=(),
            flashpoint_realizations=(),
            upstream_dangerous_goods_realizations=(),
        ),
    )


def test_integrated_config_preserves_provider_defaults_when_sampling_is_absent() -> None:
    path = Path(
        "configs/synthesis/mpci_bl_linguistic_completion_audit100_luna_high_v1.yaml"
    )
    config = load_synthesis_linguistic_completion_config(path)
    settings = openai_responses_settings(config.provider)

    assert config.workflow.max_concurrent_requests == 16
    assert config.workflow.max_concurrent_documents == 16
    assert config.provider.generation_settings is None
    assert "temperature" not in settings
    assert "top_p" not in settings
    assert "openai_text_verbosity" not in settings
    assert "openai_service_tier" not in settings


def test_provider_generation_settings_are_optional_strict_and_forwarded_verbatim() -> None:
    path = Path(
        "configs/synthesis/mpci_bl_linguistic_completion_audit100_luna_high_v1.yaml"
    )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["provider"]["service_tier"] = "priority"
    value["provider"]["generation_settings"] = {
        "temperature": 1.25,
        "top_p": 0.9,
        "text_verbosity": "medium",
    }
    config = SynthesisLinguisticCompletionConfig.model_validate(value, strict=True)
    settings = openai_responses_settings(config.provider)

    assert settings["temperature"] == 1.25
    assert settings["top_p"] == 0.9
    assert settings["openai_text_verbosity"] == "medium"
    assert settings["openai_service_tier"] == "priority"

    value["provider"]["generation_settings"]["temperature"] = 2.01
    with pytest.raises(ValidationError, match="less than or equal to 2"):
        SynthesisLinguisticCompletionConfig.model_validate(value, strict=True)


def test_linguistic_resume_source_is_an_explicit_immutable_pin() -> None:
    path = Path(
        "configs/synthesis/mpci_bl_linguistic_completion_audit100_luna_high_v1.yaml"
    )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["inputs"]["resume_run"] = {
        "path": "artifacts/kie-synthesis/prior-incomplete-run",
        "commit_sha256": "a" * 64,
        "transaction_sha256": "b" * 64,
    }

    config = SynthesisLinguisticCompletionConfig.model_validate(value, strict=True)

    assert config.inputs.resume_run is not None
    assert config.inputs.resume_run.commit_sha256 == "a" * 64


def test_party_plan_reuses_same_as_and_duplicate_notify_without_wasted_calls() -> None:
    plan = build_document_linguistic_plan(index=0, plan=_plan())

    assert len(plan.partyUnits) == 3
    by_role = {unit.seed.partyRole: unit for unit in plan.partyUnits}
    assert {unit.seed.partyRole for unit in plan.partyUnits} == {
        "shipper",
        "consignee",
        "notifyParties",
    }
    projections = plan.partyProjections
    assert projections[2].unitId is None
    assert projections[2].sameAsReference == "consignee"
    assert projections[3].unitId == by_role["consignee"].unitId
    assert projections[4].sameAsReference == "consignee"
    assert projections[5].unitId is None
    assert projections[5].sameAsReference is None

    contact_override = by_role["notifyParties"].seed
    assert contact_override.fieldPresence.namePresent is False
    assert contact_override.fieldPresence.addressPresent is False
    assert contact_override.fieldPresence.contacts.phoneNumberCount == 1


def test_topology_constrained_models_enforce_party_shape_and_cargo_cardinality() -> None:
    plan = build_document_linguistic_plan(index=0, plan=_plan())
    contact_seed = next(
        unit.seed for unit in plan.partyUnits if unit.seed.sameAsReference is not None
    )
    party_model = _dynamic_party_output_model(contact_seed)
    party_schema = OpenAIJsonSchemaTransformer(
        party_model.model_json_schema(mode="validation"), strict=True
    ).walk()
    assert party_schema["additionalProperties"] is False
    valid_party = {
        "party": {
            "partyRole": "notifyParties",
            "name": None,
            "address": None,
            "city": None,
            "country": None,
            "contactDetails": {
                "contactName": None,
                "phoneNumbers": ["+20 3 555 8124"],
                "emailAddresses": [],
                "websiteUrls": [],
            },
        }
    }
    assert party_model.model_validate_json(json.dumps(valid_party), strict=True)
    invalid_party = {**valid_party, "party": {**valid_party["party"], "name": "EXTRA"}}
    with pytest.raises(ValidationError):
        party_model.model_validate_json(json.dumps(invalid_party), strict=True)

    cargo_model = _dynamic_cargo_output_model(plan.cargoSeed)
    cargo_schema = OpenAIJsonSchemaTransformer(
        cargo_model.model_json_schema(mode="validation"), strict=True
    ).walk()
    assert cargo_schema["additionalProperties"] is False
    valid_cargo = {
        "cargoGroups": [
            {
                "groupId": "g1",
                "description": "FROZEN SKIPJACK TUNA",
                "additionalInformation": [],
                "marksAndNumbers": ["N/M", "BLUEFIN-204"],
                "handlingInstructions": [],
            }
        ]
    }
    assert cargo_model.model_validate_json(json.dumps(valid_cargo), strict=True)
    invalid_cargo = {
        "cargoGroups": [
            {**valid_cargo["cargoGroups"][0], "marksAndNumbers": ["N/M"]}
        ]
    }
    with pytest.raises(ValidationError):
        cargo_model.model_validate_json(json.dumps(invalid_cargo), strict=True)


def test_party_validation_supports_name_only_shape_and_blocks_source_reuse() -> None:
    plan = build_document_linguistic_plan(index=0, plan=_plan())
    seed = next(unit.seed for unit in plan.partyUnits if unit.seed.partyRole == "shipper")
    inventory = SourceSensitiveInventory(
        names=frozenset({"SOURCE EXPORTS LTD"}),
        addresses=frozenset(),
        contactNames=frozenset(),
        phoneNumbers=frozenset(),
        emailAddresses=frozenset(),
        websiteUrls=frozenset(),
        digest="c" * 64,
    )
    valid = PartyCompletionOutput(
        party=GeneratedPartyIdentityV2(
            partyRole="shipper",
            name="VISAYAN COLDCHAIN FOODS CORPORATION",
            address=None,
            city="Cebu City",
            country="Philippines",
            contactDetails=GeneratedPartyContactsV2(
                contactName=None,
                phoneNumbers=(),
                emailAddresses=(),
                websiteUrls=(),
            ),
        )
    )
    assert all(
        validate_party_completion(
            seed=seed, output=valid, source_inventory=inventory
        ).values()
    )

    copied = valid.model_copy(
        update={"party": valid.party.model_copy(update={"name": "SOURCE EXPORTS LTD"})}
    )
    checks = validate_party_completion(
        seed=seed, output=copied, source_inventory=inventory
    )
    assert checks["names_absent_from_source_corpus"] is False


@pytest.mark.asyncio
async def test_unit_runner_retains_failed_attempt_and_uses_fresh_constrained_retry() -> None:
    payload = {
        "party": {
            "partyRole": "carrier",
            "name": "FICTIONAL MARINE LTD",
            "address": None,
            "city": None,
            "country": None,
            "contactDetails": {
                "contactName": None,
                "phoneNumbers": [],
                "emailAddresses": [],
                "websiteUrls": [],
            },
        }
    }

    def respond(_: Any, __: Any) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(json.dumps(payload))],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
            model_name="offline-function-model",
            provider_response_id="resp_test_linguistic",
        )

    model = FunctionModel(
        respond,
        profile=ModelProfile(
            supports_json_schema_output=True,
            default_structured_output_mode="native",
        ),
    )
    validations = 0

    def validate(_: Any) -> dict[str, bool]:
        nonlocal validations
        validations += 1
        return {"accepted": validations > 1}

    config = load_synthesis_linguistic_completion_config(
        Path("configs/synthesis/mpci_bl_linguistic_completion_audit100_luna_high_v1.yaml")
    )
    seed = PartyCompletionOutput.model_validate_json(json.dumps(payload), strict=True)
    artifact = await _run_unit(
        stage="party_identity",
        unit_id="party-carrier-1",
        seed=seed,
        static_output_type=PartyCompletionOutput,
        constrained_output_type=PartyCompletionOutput,
        canonical_output_type=PartyCompletionOutput,
        validator=validate,
        model=cast(Any, model),
        settings={},
        limiter=ConcurrencyLimiter(1),
        system_prompt="offline test",
        max_output_tokens=4096,
        attempts=2,
        pricing=config.provider.pricing,
    )

    assert artifact.status == "success"
    assert artifact.selectedAttempt == 2
    assert [row.status for row in artifact.attempts] == [
        "validation_failed",
        "success",
    ]
    assert len(artifact.transcripts) == 2
    assert sum(row.usage.requests for row in artifact.attempts) == 2
    assert all(
        row.usage.providerResponseIds == ("resp_test_linguistic",)
        for row in artifact.attempts
    )
    assert _resume_unit_passes_current_contract(
        artifact,
        stage="party_identity",
        unit_id="party-carrier-1",
        output_type=PartyCompletionOutput,
        validator=lambda _: {"accepted": True},
    )
    assert not _resume_unit_passes_current_contract(
        artifact,
        stage="cargo_language",
        unit_id="party-carrier-1",
        output_type=PartyCompletionOutput,
        validator=lambda _: {"accepted": True},
    )
