from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import yaml
from pydantic import ValidationError
from pydantic_ai.usage import RequestUsage

from document_ocr.synthesis.config import (
    PartyIdentityProbeCaseConfig,
    SynthesisPartyIdentityProbeConfig,
    load_synthesis_party_identity_probe_config,
)
from document_ocr.synthesis.linguistic_probe_runtime import price_usage
from document_ocr.synthesis.party_identity_probe import (
    GeneratedPartyContacts,
    GeneratedPartyIdentity,
    PartyIdentityGenerationOutput,
    PartyIdentityGenerationSeed,
    build_party_generation_seed,
    validate_generated_party,
)
from document_ocr.synthesis.semantic_completion_pipeline import SemanticCompletionPlanRow


def _plan() -> SemanticCompletionPlanRow:
    return cast(
        SemanticCompletionPlanRow,
        SimpleNamespace(
            target={
                "documentPatch": {
                    "parties": {
                        "shipper": {
                            "name": "SOURCE EXPORTS LTD",
                            "address": "OLD ROAD 10",
                            "city": "Cebu City",
                            "country": "Philippines",
                            "contactDetails": {
                                "phoneNumbers": ["+63 32 555 0000"],
                                "emailAddresses": ["ops@source.invalid"],
                            },
                        }
                    }
                }
            },
            scenario_id="syn_0123456789abcdef",
            cargo_realizations=(
                SimpleNamespace(
                    cargo_group_id="g1",
                    description="Frozen skipjack tuna",
                    output_hs_code="03034310",
                    hs6="030343",
                    chapter_description="FISH AND CRUSTACEANS",
                    thermal_profile="FROZEN",
                ),
            ),
            flashpoint_realizations=(),
            upstream_dangerous_goods_realizations=(),
        ),
    )


def _seed() -> PartyIdentityGenerationSeed:
    case = PartyIdentityProbeCaseConfig(
        document_id="doc_" + "a" * 64,
        party_role="shipper",
        occurrence=0,
    )
    return build_party_generation_seed(case_index=0, case=case, plan=_plan())


def test_probe_config_is_strict_and_pins_one_request_per_case() -> None:
    path = Path("configs/synthesis/mpci_bl_party_identity_luna_high_probe5.yaml")
    config = load_synthesis_party_identity_probe_config(path)
    assert len(config.cases) == 5
    assert config.workflow.requests_per_case == 1
    assert config.workflow.structured_output_retries == 0
    assert config.workflow.provider_native_strict_json_schema is True
    assert config.provider.store_responses is True

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["workflow"]["structured_output_retries"] = 1
    with pytest.raises(ValidationError, match="structured_output_retries"):
        SynthesisPartyIdentityProbeConfig.model_validate(value, strict=True)


def test_probe50_config_has_unique_documents_and_role_coverage() -> None:
    config = load_synthesis_party_identity_probe_config(
        Path("configs/synthesis/mpci_bl_party_identity_luna_high_probe50.yaml")
    )

    assert len(config.cases) == 50
    assert len({case.document_id for case in config.cases}) == 50
    assert {case.party_role for case in config.cases} == {
        "shipper",
        "consignee",
        "notifyParties",
        "carrier",
        "forwardingAgent",
        "deliveryAgent",
    }
    assert config.workflow.requests_per_case == 1
    assert config.workflow.structured_output_retries == 0


def test_seed_preserves_target_locality_field_shape_and_goods() -> None:
    seed = _seed()
    assert seed.partyRole == "shipper"
    assert seed.targetLocality == {"city": "Cebu City", "country": "Philippines"}
    assert seed.sourcePartyNameStyleReference == "SOURCE EXPORTS LTD"
    assert seed.fieldPresence.addressPresent is True
    assert seed.fieldPresence.contacts.phoneNumberCount == 1
    assert seed.fieldPresence.contacts.emailAddressCount == 1
    assert seed.goods[0].description == "Frozen skipjack tuna"
    assert seed.goods[0].thermalProfile == "FROZEN"


def test_generated_party_validation_checks_semantics_beyond_json_schema() -> None:
    seed = _seed()
    output = PartyIdentityGenerationOutput(
        party=GeneratedPartyIdentity(
            partyRole="shipper",
            name="PACIFIC COLDWATER FOODS CORPORATION",
            address="Unit 8, North Reclamation Industrial Estate, 6000",
            city="Cebu City",
            country="Philippines",
            contactDetails=GeneratedPartyContacts(
                contactName=None,
                phoneNumbers=("+63 32 401 7286",),
                emailAddresses=("shipping@pacificcoldwater.ph",),
                websiteUrls=(),
            ),
        )
    )
    validation = validate_generated_party(
        seed=seed,
        output=output,
        source_names=frozenset({"SOURCE EXPORTS LTD"}),
    )
    assert validation.passed is True
    assert all(validation.checks.values())

    invalid = output.model_copy(
        update={
            "party": output.party.model_copy(
                update={"city": "Manila", "name": "SOURCE EXPORTS LTD"}
            )
        }
    )
    failed = validate_generated_party(
        seed=seed,
        output=invalid,
        source_names=frozenset({"SOURCE EXPORTS LTD"}),
    )
    assert failed.passed is False
    assert failed.checks["city_exact"] is False
    assert failed.checks["name_differs_from_source"] is False
    assert failed.checks["name_not_in_source_corpus"] is False


def test_pricing_counts_reasoning_as_output() -> None:
    config = load_synthesis_party_identity_probe_config(
        Path("configs/synthesis/mpci_bl_party_identity_luna_high_probe5.yaml")
    )
    usage = RequestUsage(input_tokens=1_000, output_tokens=500)
    assert price_usage(usage, config.provider.pricing) == Decimal("0.000800000000")
