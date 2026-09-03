from __future__ import annotations

import json
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
    PartyCompletionOutput,
    SourceSensitiveInventory,
    _run_unit,
    _dynamic_cargo_output_model,
    _dynamic_party_output_model,
    build_document_linguistic_plan,
    validate_party_completion,
)
from document_ocr.synthesis.linguistic_probe_runtime import openai_responses_settings
from document_ocr.synthesis.semantic_completion_pipeline import SemanticCompletionPlanRow


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
