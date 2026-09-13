from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from raw_text_template_experiment.agents import (
    _apply_compiler_repair,
    _compact_inventory_occurrence_lookup,
    _critic_output_with_inventory_ids,
    _critic_provider_payload,
    _openrouter_profile,
    _restore_reference_compact_critic,
    _restore_reference_compiler,
    _scoped_compact_critic_output_type,
    _scoped_compiler_repair_output_type,
    _scoped_critic_output_type,
    _scoped_reference_compiler_output_type,
    _settings,
)
from raw_text_template_experiment.host import inventory_binding_id
from raw_text_template_experiment.models import CompilerAgentOutput, OpenRouterProviderConfig


def _compiler_candidate() -> CompilerAgentOutput:
    occurrence = {
        "line_start": "L00002",
        "line_end": "L00002",
        "source_text": "OLD1",
        "occurrence_index": 0,
    }
    return CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": "Example Carrier Ltd",
                "aliases": (),
                "evidence_occurrences": (occurrence,),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Exact carrier occurrence.",
            },
            "anchor_overrides": (),
            "bindings": (
                {
                    "logical_key": "document_number",
                    "render_mode": "target_binding",
                    "value_kind": "identifier",
                    "group_kind": "document",
                    "group_key": "document",
                    "target_paths": ("documentPatch.billOfLadingNumber",),
                    "occurrences": (occurrence,),
                    "rationale": "Initial binding.",
                },
            ),
            "unresolved": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
            "semantic_only_target_facts": (),
        }
    )


def test_compiler_repair_is_scoped_and_applies_only_local_replacements() -> None:
    prior = _compiler_candidate()
    output_type = _scoped_compiler_repair_output_type(prior)
    replacement = {
        "logical_key": "document_number",
        "value_kind": "identifier",
        "group_kind": "document",
        "group_key": "document",
        "rendering": {
            "render_mode": "target_binding",
            "target_paths": ("documentPatch.billOfLadingNumber",),
        },
        "occurrences": (
            {
                "line_start": "L00003",
                "line_end": "L00003",
                "source_text": "NEW2",
                "occurrence_index": 0,
            },
        ),
        "rationale": "Move the binding to the correct occurrence.",
    }
    repair = output_type.model_validate(
        {
            "remove_binding_logical_keys": ("document_number",),
            "replacement_bindings": (replacement,),
            "rationale": "Apply only the cited local correction.",
        }
    )

    updated = _apply_compiler_repair(prior, repair)

    assert len(updated.bindings) == 1
    assert updated.bindings[0].occurrences[0].source_text == "NEW2"
    schema = output_type.model_json_schema(mode="validation")
    assert schema["properties"]["remove_binding_logical_keys"]["items"]["const"] == (
        "document_number"
    )
    assert schema["properties"]["remove_binding_logical_keys"]["uniqueItems"] is True
    with pytest.raises(ValidationError, match="remove_binding_logical_keys"):
        output_type.model_validate(
            {
                "remove_binding_logical_keys": ("invented",),
                "rationale": "Invalid removal.",
            }
        )
    with pytest.raises(ValidationError, match="must remove every existing key"):
        output_type.model_validate(
            {
                "replacement_bindings": (replacement,),
                "rationale": "Invalidly replace an existing key without removing it first.",
            }
        )


def test_compiler_repair_derives_completion_from_unresolved_items() -> None:
    prior = _compiler_candidate().model_copy(
        update={
            "unresolved": ("One concrete unresolved surface.",),
            "all_shipment_dependent_surfaces_accounted_for": False,
        }
    )
    output_type = _scoped_compiler_repair_output_type(prior)
    schema = output_type.model_json_schema(mode="validation")
    assert "completion_replacement" not in schema["properties"]
    repair = output_type.model_validate(
        {
            "unresolved_replacement": (),
            "rationale": "The cited surface is now completely resolved.",
        }
    )

    updated = _apply_compiler_repair(prior, repair)

    assert updated.unresolved == ()
    assert updated.all_shipment_dependent_surfaces_accounted_for is True
    with pytest.raises(ValidationError, match="completion_replacement"):
        output_type.model_validate(
            {
                "completion_replacement": False,
                "rationale": "The provider cannot control a derived host invariant.",
            }
        )


def _revision(removal: str) -> dict[str, object]:
    return {
        "verdict": "revise",
        "findings": (
            {
                "finding_kind": "incorrect_semantic_owner",
                "line_ids": ("L00001",),
                "evidence": "The current binding owns the wrong value.",
                "explanation": "Replace the cited current binding.",
            },
        ),
        "remove_binding_logical_keys": (removal,),
        "additional_bindings": (),
        "semantic_only_target_facts": (),
        "rationale": "Apply the cited replacement transaction.",
    }


def test_scoped_critic_schema_enumerates_only_current_logical_keys() -> None:
    current = (
        "anchor:documentPatch.billOfLadingNumber",
        "agent:customs:declaration_reference",
    )
    output_type = _scoped_critic_output_type(current)

    schema = output_type.model_json_schema(mode="validation")
    assert schema["properties"]["remove_binding_logical_keys"]["items"]["enum"] == list(current)
    parsed = output_type.model_validate(_revision(current[0]))
    assert parsed.remove_binding_logical_keys == (current[0],)
    translated = _critic_output_with_inventory_ids(parsed)
    assert translated.remove_inventory_binding_ids == (inventory_binding_id(current[0]),)
    with pytest.raises(ValidationError, match="remove_binding_logical_keys"):
        output_type.model_validate(_revision("agent:invented"))


def test_scoped_critic_schema_rejects_empty_or_duplicate_logical_keys() -> None:
    with pytest.raises(ValueError, match="no allowed removal logical keys"):
        _scoped_critic_output_type(())
    with pytest.raises(ValueError, match="duplicate removal logical keys"):
        _scoped_critic_output_type(
            (
                "anchor:documentPatch.billOfLadingNumber",
                "anchor:documentPatch.billOfLadingNumber",
            )
        )


def test_compact_critic_schema_discriminates_pass_from_revision() -> None:
    output_type = _scoped_compact_critic_output_type(
        ("anchor:documentPatch.billOfLadingNumber",),
        ("review_candidate_0001",),
        (),
        ("L00001",),
    )
    coverage = {
        "literal_completeness_checked": True,
        "target_ownership_checked": True,
        "topology_and_grouping_checked": True,
        "derivations_checked": True,
        "carrier_boundary_checked": True,
        "identifier_relationships_checked": True,
        "candidate_receipt": {
            "review_candidate_0001": {
                "conclusion": "valid_existing_contract",
                "rationale": "The current owner is correct.",
            }
        },
        "literal_line_receipt": {"L00001": True},
    }

    parsed = output_type.model_validate(
        {
            "decision": {
                "verdict": "pass",
                "coverage": coverage,
                "rationale": "Every required facet is clear.",
            }
        }
    )
    assert parsed.decision.verdict == "pass"
    with pytest.raises(ValidationError, match="review_candidate_0001"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "pass",
                    "coverage": {**coverage, "candidate_receipt": {}},
                    "rationale": "The candidate receipt is incomplete.",
                }
            }
        )
    with pytest.raises(ValidationError, match="Field required"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "pass",
                    "coverage": {**coverage, "literal_line_receipt": {}},
                    "rationale": "The line-level receipt is incomplete.",
                }
            }
        )
    with pytest.raises(ValidationError, match="pass cannot retain"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "pass",
                    "coverage": {
                        **coverage,
                        "candidate_receipt": {
                            "review_candidate_0001": {
                                "conclusion": "defect_requires_revision",
                                "rationale": "This candidate requires revision.",
                            }
                        },
                    },
                    "rationale": "An invalid pass with a candidate defect.",
                }
            }
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "pass",
                    "coverage": coverage,
                    "findings": [],
                    "rationale": "An illegal patch surface was attached.",
                }
            }
        )


def test_compact_critic_revision_requires_true_receipt_for_every_literal_line() -> None:
    output_type = _scoped_compact_critic_output_type(
        ("anchor:documentPatch.billOfLadingNumber",),
        (),
        (),
        ("L00001", "L00002"),
    )
    coverage = {
        "literal_completeness_checked": True,
        "target_ownership_checked": True,
        "topology_and_grouping_checked": True,
        "derivations_checked": True,
        "carrier_boundary_checked": True,
        "identifier_relationships_checked": True,
        "candidate_receipt": {},
        "literal_line_receipt": {
            "L00001": True,
            "L00002": True,
        },
    }
    finding = {
        "finding_kind": "unowned_private_or_auxiliary_fact",
        "line_ids": ("L00001",),
        "evidence": "REFERENCE 12345",
        "explanation": "The private reference must be regenerated.",
    }

    parsed = output_type.model_validate(
        {
            "decision": {
                "verdict": "revise",
                "coverage": coverage,
                "findings": (finding,),
                "rationale": "Own the one defective literal line.",
            }
        }
    )
    assert parsed.decision.verdict == "revise"
    with pytest.raises(ValidationError, match="Input should be True"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "revise",
                    "coverage": {
                        **coverage,
                        "literal_line_receipt": {
                            "L00001": False,
                            "L00002": True,
                        },
                    },
                    "findings": (finding,),
                    "rationale": "The finding and receipt disagree.",
                }
            }
        )


def test_critic_provider_view_uses_one_lossless_annotated_source() -> None:
    compact = {
        "annotatedSource": "L00001 | A ⟦binding_0000⟧VALUE⟦/binding⟧ B",
        "otherField": "unrelated field remains untouched",
        "maskedTemplate": "L00001 | A ⟦binding_0000⟧ B",
    }

    provider_view = _critic_provider_payload(compact)

    assert provider_view["annotatedSource"] == compact["annotatedSource"]
    assert "maskedTemplate" not in provider_view
    assert provider_view["otherField"] == "unrelated field remains untouched"
    assert compact["maskedTemplate"] == "L00001 | A ⟦binding_0000⟧ B"
    with pytest.raises(ValueError, match="requires annotated and masked source views"):
        _critic_provider_payload({"maskedTemplate": "L00001 | literal"})


def test_compact_critic_requires_every_remaining_risk_to_be_revised() -> None:
    output_type = _scoped_compact_critic_output_type(
        ("anchor:documentPatch.billOfLadingNumber",),
        (),
        ("risk_0001",),
        ("L00001",),
    )
    base_coverage = {
        "literal_completeness_checked": True,
        "target_ownership_checked": True,
        "topology_and_grouping_checked": True,
        "derivations_checked": True,
        "carrier_boundary_checked": True,
        "identifier_relationships_checked": True,
        "literal_line_receipt": {"L00001": True},
    }
    with pytest.raises(ValidationError, match="defect_requires_revision"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "pass",
                    "coverage": {
                        **base_coverage,
                        "candidate_receipt": {
                            "risk_0001": {
                                "conclusion": "valid_existing_contract",
                                "rationale": "Incorrectly ignored risk.",
                            }
                        },
                    },
                    "rationale": "Incorrect pass.",
                }
            }
        )
    with pytest.raises(ValidationError, match="pass cannot retain"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "pass",
                    "coverage": {
                        **base_coverage,
                        "candidate_receipt": {
                            "risk_0001": {
                                "conclusion": "defect_requires_revision",
                                "rationale": "The unowned risk requires a binding.",
                            }
                        },
                    },
                    "rationale": "Incorrect pass with an acknowledged defect.",
                }
            }
        )


def test_compact_critic_materializes_only_enumerated_occurrence_handles() -> None:
    occurrence_id = "compiler_occurrence_00001"
    output_type = _scoped_compact_critic_output_type(
        ("anchor:documentPatch.billOfLadingNumber",),
        (),
        (),
        ("L00001",),
        (occurrence_id,),
    )
    payload = {
        "occurrenceCandidates": {
            "columns": (
                "occurrenceId",
                "lineStart",
                "lineEnd",
                "sourceText",
                "occurrenceIndex",
            ),
            "rows": ((occurrence_id, "L00001", "L00001", "NEW123", 0),),
        }
    }
    source_payload = {
        "bindingInventory": (
            {
                "logicalKey": "anchor:documentPatch.billOfLadingNumber",
                "renderMode": "target_binding",
                "valueKind": "identifier",
                "groupKind": "document",
                "groupKey": "document",
                "targetPaths": ("documentPatch.billOfLadingNumber",),
                "derivation": None,
                "dependencyPaths": (),
                "dependencyBindings": (),
            },
        )
    }
    candidate = {
        "decision": {
            "verdict": "revise",
            "coverage": {
                "literal_completeness_checked": True,
                "target_ownership_checked": True,
                "topology_and_grouping_checked": True,
                "derivations_checked": True,
                "carrier_boundary_checked": True,
                "identifier_relationships_checked": True,
                "candidate_receipt": {},
                "literal_line_receipt": {"L00001": True},
            },
            "findings": (
                {
                    "finding_kind": "unowned_shipment_fact",
                    "line_ids": ("L00001",),
                    "evidence": "NEW123",
                    "explanation": "The identifier requires deterministic ownership.",
                },
            ),
            "occurrence_appends": (
                {
                    "logical_key": "anchor:documentPatch.billOfLadingNumber",
                    "occurrences": ({"occurrence_id": occurrence_id},),
                    "rationale": "Append the exact host-resolved occurrence.",
                },
            ),
            "rationale": "Add the unowned identifier.",
        }
    }

    parsed = output_type.model_validate(candidate)
    restored = _restore_reference_compact_critic(parsed, payload, source_payload)

    assert restored.additional_bindings[0].occurrences[0].source_text == "NEW123"
    assert restored.additional_bindings[0].occurrences[0].line_start == "L00001"
    candidate["decision"]["occurrence_appends"][0]["occurrences"] = (
        {"occurrence_id": "compiler_occurrence_99999"},
    )
    with pytest.raises(ValidationError, match="occurrence_id"):
        output_type.model_validate(candidate)


def test_compact_critic_scopes_and_materializes_partial_occurrence_removals() -> None:
    logical_key = "anchor:documentPatch.billOfLadingNumber"
    other_key = "agent:booking_reference"
    compact_payload = {
        "compactContract": {
            "bindingColumns": (
                "bindingId",
                "sourceBindingIds",
                "logicalKey",
                "renderMode",
                "valueKind",
                "groupKind",
                "groupKey",
                "targetPathIds",
                "targetRelationship",
                "independentTargetFactComponentPathIds",
                "derivation",
                "dependencyPathIds",
                "dependencyBindings",
                "occurrenceIds",
            ),
            "occurrenceColumns": (
                "occurrenceId",
                "sourceBindingId",
                "lineStartNumber",
                "lineEndNumber",
                "sourceText",
                "occurrenceIndex",
                "exactMatchCount",
                "exactMatchCandidates",
            ),
        },
        "bindingRows": (
            (
                "binding_0000",
                (),
                logical_key,
                "target_binding",
                "identifier",
                "document",
                "document",
                (),
                "single_target",
                (),
                None,
                (),
                (),
                ("occurrence_00000", "occurrence_00001"),
            ),
            (
                "binding_0001",
                (),
                other_key,
                "deterministic_auxiliary",
                "identifier",
                "document",
                "document",
                (),
                "source_only",
                (),
                None,
                (),
                (),
                ("occurrence_00002",),
            ),
        ),
        "occurrenceRows": (
            ("occurrence_00000", 0, 2, 2, "BOL123", 0, 2, ()),
            ("occurrence_00001", 0, 3, 3, "BOL123", 0, 2, ()),
            ("occurrence_00002", 0, 4, 4, "BOOK456", 0, 1, ()),
        ),
        "occurrenceCandidates": {
            "columns": (
                "occurrenceId",
                "lineStart",
                "lineEnd",
                "sourceText",
                "occurrenceIndex",
            ),
            "rows": (),
        },
    }
    inventory = _compact_inventory_occurrence_lookup(compact_payload)
    owners = {occurrence_id: owner for occurrence_id, (owner, _row) in inventory.items()}
    output_type = _scoped_compact_critic_output_type(
        (logical_key, other_key),
        (),
        (),
        ("L00001",),
        (),
        owners,
    )
    candidate = {
        "decision": {
            "verdict": "revise",
            "coverage": {
                "literal_completeness_checked": True,
                "target_ownership_checked": True,
                "topology_and_grouping_checked": True,
                "derivations_checked": True,
                "carrier_boundary_checked": True,
                "identifier_relationships_checked": True,
                "candidate_receipt": {},
                "literal_line_receipt": {"L00001": True},
            },
            "findings": (
                {
                    "finding_kind": "incorrect_semantic_owner",
                    "line_ids": ("L00003",),
                    "evidence": "The second copy is a caption substring.",
                    "explanation": "Drop only the bad physical occurrence.",
                },
            ),
            "occurrence_removals": (
                {
                    "logical_key": logical_key,
                    "occurrence_ids": ("occurrence_00001",),
                    "rationale": "Retain the binding and its first valid occurrence.",
                },
            ),
            "rationale": "Apply the exact partial removal.",
        }
    }

    parsed = output_type.model_validate(candidate)
    restored = _restore_reference_compact_critic(
        parsed,
        compact_payload,
        {"bindingInventory": ({"logicalKey": logical_key}, {"logicalKey": other_key})},
    )

    assert restored.occurrence_removals[0].logical_key == logical_key
    assert restored.occurrence_removals[0].occurrences[0].line_start == "L00003"
    candidate["decision"]["occurrence_removals"][0]["occurrence_ids"] = ("occurrence_00002",)
    with pytest.raises(ValidationError, match="declared logical key"):
        output_type.model_validate(candidate)
    candidate["decision"]["occurrence_removals"][0]["occurrence_ids"] = (
        "occurrence_00000",
        "occurrence_00001",
    )
    with pytest.raises(ValidationError, match="cannot empty a binding"):
        output_type.model_validate(candidate)


def test_reference_compiler_schema_materializes_only_enumerated_spans() -> None:
    occurrence_id = "compiler_occurrence_00001"
    output_type = _scoped_reference_compiler_output_type((occurrence_id,))
    payload = {
        "occurrenceCandidates": {
            "columns": (
                "occurrenceId",
                "lineStart",
                "lineEnd",
                "sourceText",
                "occurrenceIndex",
            ),
            "rows": (
                (
                    occurrence_id,
                    "L00002",
                    "L00002",
                    "Example Carrier Ltd",
                    0,
                ),
            ),
        }
    }
    candidate = {
        "carrier": {
            "canonical_name": "Example Carrier Ltd",
            "aliases": (),
            "evidence_occurrences": ({"occurrence_id": occurrence_id},),
            "source": "source_label_confirmed_by_ocr",
            "rationale": "Exact carrier occurrence supplied by the host.",
        },
        "anchor_overrides": (),
        "bindings": (
            {
                "logical_key": "agent:document:number",
                "value_kind": "identifier",
                "group_kind": "document",
                "group_key": "document",
                "rendering": {
                    "render_mode": "target_binding",
                    "target_paths": ("documentPatch.billOfLadingNumber",),
                },
                "occurrences": ({"occurrence_id": occurrence_id},),
                "rationale": "Fixture binding.",
            },
        ),
        "unresolved": (),
        "semantic_only_target_facts": (),
    }

    parsed = output_type.model_validate(candidate)
    restored = _restore_reference_compiler(parsed, payload)

    assert restored.carrier.evidence_occurrences[0].source_text == "Example Carrier Ltd"
    assert restored.bindings[0].occurrences[0].line_start == "L00002"
    invalid = dict(candidate)
    invalid["carrier"] = {
        **candidate["carrier"],
        "evidence_occurrences": ({"occurrence_id": "compiler_occurrence_99999"},),
    }
    with pytest.raises(ValidationError, match="occurrence_id"):
        output_type.model_validate(invalid)


def test_reference_compiler_schema_retains_exact_occurrence_escape_hatch() -> None:
    output_type = _scoped_reference_compiler_output_type(())
    occurrence = {
        "line_start": "L00003",
        "line_end": "L00003",
        "source_text": "UNLISTED",
        "occurrence_index": 0,
    }
    candidate = {
        "carrier": {
            "canonical_name": "Example Carrier Ltd",
            "aliases": (),
            "evidence_occurrences": (occurrence,),
            "source": "source_label_confirmed_by_ocr",
            "rationale": "Unlisted exact occurrence.",
        },
        "anchor_overrides": (),
        "bindings": (),
        "unresolved": (),
        "semantic_only_target_facts": (),
    }

    parsed = output_type.model_validate(candidate)
    restored = _restore_reference_compiler(
        parsed,
        {
            "occurrenceCandidates": {
                "columns": (
                    "occurrenceId",
                    "lineStart",
                    "lineEnd",
                    "sourceText",
                    "occurrenceIndex",
                ),
                "rows": (),
            }
        },
    )

    assert restored.carrier.evidence_occurrences[0].source_text == "UNLISTED"


def test_openrouter_settings_pin_exact_routing_and_price_ceiling() -> None:
    provider = OpenRouterProviderConfig.model_validate(
        {
            "kind": "openrouter",
            "model": "openai/gpt-5.6-luna",
            "api_key_env": "OPENROUTER_API_KEY",
            "reasoning_effort": "high",
            "request_timeout_seconds": 900.0,
            "transport_max_retries": 2,
            "max_output_tokens": 65536,
            "require_parameters": True,
            "data_collection": "deny",
            "allow_fallbacks": False,
            "provider_only": ("OpenAI",),
            "max_prompt_price_usd_per_million": Decimal("0.20"),
            "max_completion_price_usd_per_million": Decimal("1.20"),
            "pricing": {
                "currency": "USD",
                "effective_date": "2026-09-12",
                "source_url": "https://openrouter.ai/api/v1/models",
                "input_usd_per_million": Decimal("0.20"),
                "cached_input_usd_per_million": Decimal("0.02"),
                "cache_write_multiplier": Decimal("1.25"),
                "output_usd_per_million": Decimal("1.20"),
            },
        }
    )

    settings = _settings(provider)

    assert settings["openrouter_reasoning"] == {"effort": "high"}
    assert settings["openrouter_provider"] == {
        "only": ["OpenAI"],
        "order": ["OpenAI"],
        "require_parameters": True,
        "data_collection": "deny",
        "allow_fallbacks": False,
        "max_price": {"prompt": 0.2, "completion": 1.2},
    }
    assert _openrouter_profile(provider) is None

    verified_payload = provider.model_dump(mode="python")
    verified_payload.update(
        {
            "model": "z-ai/glm-5.3-flash",
            "provider_only": ("DeepInfra",),
            "native_structured_output_profile": "provider_verified",
            "native_structured_output_source_url": ("https://openrouter.ai/z-ai/glm-5.3-flash"),
        }
    )
    verified = OpenRouterProviderConfig.model_validate(verified_payload)
    assert _openrouter_profile(verified) == {"supports_json_schema_output": True}


def test_provider_verified_structured_output_requires_source_url() -> None:
    with pytest.raises(
        ValidationError,
        match="provider-verified native structured output requires exactly one source URL",
    ):
        OpenRouterProviderConfig.model_validate(
            {
                "kind": "openrouter",
                "model": "z-ai/glm-5.3-flash",
                "api_key_env": "OPENROUTER_API_KEY",
                "reasoning_effort": "high",
                "request_timeout_seconds": 900.0,
                "transport_max_retries": 2,
                "max_output_tokens": 65536,
                "require_parameters": True,
                "data_collection": "deny",
                "allow_fallbacks": False,
                "provider_only": ("DeepInfra",),
                "native_structured_output_profile": "provider_verified",
                "max_prompt_price_usd_per_million": Decimal("0.075"),
                "max_completion_price_usd_per_million": Decimal("0.25"),
                "pricing": {
                    "currency": "USD",
                    "effective_date": "2026-09-12",
                    "source_url": "https://openrouter.ai/z-ai/glm-5.3-flash",
                    "input_usd_per_million": Decimal("0.075"),
                    "cached_input_usd_per_million": Decimal("0.015"),
                    "cache_write_multiplier": Decimal(0),
                    "output_usd_per_million": Decimal("0.25"),
                },
            }
        )
