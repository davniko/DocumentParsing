from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

import raw_text_template_experiment.agents as agents_module
from raw_text_template_experiment.agents import (
    AgentRuntime,
    _apply_compiler_repair,
    _compact_inventory_occurrence_lookup,
    _compiler_repair_addable_anchor_ids,
    _compiler_repair_allowed_anchor_ids,
    _compiler_repair_allowed_paths,
    _compiler_repair_error_intersects_scope,
    _critic_output_with_inventory_ids,
    _critic_provider_payload,
    _critic_provider_request_bytes,
    _fixed_reference_compiler_output_type,
    _openrouter_profile,
    _restore_reference_compact_critic,
    _restore_reference_compiler,
    _scoped_compact_critic_output_type,
    _scoped_compiler_repair_output_type,
    _scoped_critic_output_type,
    _scoped_initial_compiler_output_type,
    _scoped_reference_compiler_output_type,
    _scoped_staged_plan_output_type,
    _settings,
    _structured_output_retry_message,
    _uses_partitioned_critic,
    _validate_compiler_repair_scope,
)
from raw_text_template_experiment.host import inventory_binding_id
from raw_text_template_experiment.models import (
    AgentStageArtifact,
    AnchorOverride,
    CompilerAgentOutput,
    CriticAgentOutput,
    OpenRouterProviderConfig,
)
from raw_text_template_experiment.staged_contract import (
    StagedAuditOutput,
    StagedCriticPlanOutput,
    StagedFacetAuditOutput,
)


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


@pytest.mark.asyncio
async def test_unprojectable_local_repair_is_explicit_and_provider_free() -> None:
    runtime = object.__new__(AgentRuntime)
    runtime._agent_contract_protocol = "staged_local_v4"
    prior = _compiler_candidate()

    output, stage = await runtime.compiler(
        pass_number=2,
        system_prompt="root compiler",
        repair_prompt="local repair",
        payload={
            "requiredRevision": "Compiler candidate is incomplete without grounded selectors",
            "previousCandidateOutput": prior.model_dump(mode="json"),
            "numberedSource": "L00001 | HEADER\nL00002 | OLD1",
            "allowedTargetPaths": ("documentPatch.billOfLadingNumber",),
            "anchorBindings": (),
            "previousCandidateBindingInventory": (),
            "sourceLabel": {"documentPatch": {"billOfLadingNumber": "OLD1"}},
        },
        retries=1,
    )

    assert output is None
    assert stage.status == "host_rejected"
    assert stage.usage.requests == 0
    assert stage.error_message is not None
    assert "cannot be safely projected" in stage.error_message
    assert stage.messages[0]["requiredRevision"] == (
        "Compiler candidate is incomplete without grounded selectors"
    )


@pytest.mark.asyncio
async def test_relational_protocol_projects_compiler_repair_to_local_transaction() -> None:
    runtime = object.__new__(AgentRuntime)
    runtime._agent_contract_protocol = "relational_reference_compact_v9"
    prior = _compiler_candidate()
    target_path = "documentPatch.billOfLadingNumber"
    payload = {
        "documentId": "doc_local_relational_repair",
        "expectedCarrierName": "Example Carrier Ltd",
        "requiredRevision": f"invalid occurrence at L00002 for {target_path}",
        "numberedSource": "L00001 | HEADER\nL00002 | OLD1\nL00003 | NEW2",
        "allowedTargetPaths": (target_path,),
        "sourceLabel": {"documentPatch": {"billOfLadingNumber": "NEW2"}},
        "anchorBindings": (),
        "requiredTargetCoBindings": (),
        "occurrenceCandidates": {
            "columns": (
                "occurrenceId",
                "lineStart",
                "lineEnd",
                "sourceText",
                "occurrenceIndex",
            ),
            "rows": (("compiler_occurrence_00001", "L00003", "L00003", "NEW2", 0),),
        },
        "previousCandidateBindingInventory": (
            {
                "logicalKey": "anchor:" + target_path,
                "targetPaths": (target_path,),
                "occurrences": (
                    {
                        "lineStart": "L00002",
                        "lineEnd": "L00002",
                        "sourceText": "OLD1",
                        "occurrenceIndex": 0,
                    },
                ),
            },
        ),
        "previousCandidateOutput": prior.model_dump(mode="json"),
    }
    captured: dict[str, object] = {}

    class Stage:
        def model_copy(self, *, update: dict[str, object]) -> Stage:
            captured["artifact_update"] = update
            return self

    async def fake_call(**kwargs: object) -> tuple[object, Stage]:
        captured.update(kwargs)
        output_type = kwargs["output_type"]
        repair = output_type.model_validate(
            {
                "replacement_bindings": (
                    {
                        "logical_key": "document_number",
                        "value_kind": "identifier",
                        "group_kind": "document",
                        "group_key": "document",
                        "rendering": {
                            "render_mode": "target_binding",
                            "target_paths": (target_path,),
                        },
                        "occurrences": (
                            {
                                "line_start": "L00003",
                                "line_end": "L00003",
                                "source_text": "NEW2",
                                "occurrence_index": 0,
                            },
                        ),
                        "rationale": "Move the binding to the exact corrected occurrence.",
                    },
                ),
                "rationale": "Repair only the host-cited document-number occurrence.",
            }
        )
        validator = kwargs["output_validator"]
        return validator(repair), Stage()

    runtime._call = fake_call
    output, _stage = await runtime.compiler(
        pass_number=2,
        system_prompt="root compiler",
        payload=payload,
        retries=1,
    )

    provider_payload = captured["payload"]
    assert provider_payload["contract"]["operation"] == "local_compiler_repair"
    assert "numberedSource" not in provider_payload
    assert provider_payload["candidateSlice"]["removableBindingKeys"] == ["document_number"]
    assert [row["targetPath"] for row in provider_payload["targetFacts"]] == [target_path]
    assert output is not None
    assert output.bindings[0].occurrences[0].source_text == "NEW2"


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
    implicit_atomic_replacement = output_type.model_validate(
        {
            "replacement_bindings": (replacement,),
            "rationale": "Atomically replace the existing binding under its stable key.",
        }
    )
    implicitly_updated = _apply_compiler_repair(prior, implicit_atomic_replacement)
    assert len(implicitly_updated.bindings) == 1
    assert implicitly_updated.bindings[0].occurrences[0].source_text == "NEW2"

    narrowed_type = _scoped_compiler_repair_output_type(prior, removable_binding_keys=())
    with pytest.raises(ValidationError, match="outside the local repair scope"):
        narrowed_type.model_validate(
            {
                "replacement_bindings": (replacement,),
                "rationale": "Cannot replace a retained key outside the local repair slice.",
            }
        )


def test_compiler_repair_rejects_rationale_only_anchor_override_replacement() -> None:
    anchor_id = "anchor_binding_0001"
    prior = _compiler_candidate().model_copy(
        update={
            "anchor_overrides": (
                AnchorOverride(
                    anchor_binding_id=anchor_id,
                    rationale="The accepted anchor must not be materialized.",
                ),
            )
        }
    )
    output_type = _scoped_compiler_repair_output_type(prior)

    with pytest.raises(ValidationError, match="rationale-only replacements"):
        output_type.model_validate(
            {
                "remove_anchor_override_ids": (anchor_id,),
                "additional_anchor_overrides": (
                    {
                        "anchor_binding_id": anchor_id,
                        "rationale": "A different explanation cannot alter ownership.",
                    },
                ),
                "rationale": "Attempt an operationally empty replacement.",
            }
        )


def test_local_compiler_repair_schema_enumerates_only_exposed_anchor_ids() -> None:
    allowed = ("anchor_binding_0006", "anchor_binding_0029")
    output_type = _scoped_compiler_repair_output_type(
        _compiler_candidate(),
        addable_anchor_ids=allowed,
    )
    schema_text = json.dumps(output_type.model_json_schema(mode="validation"))
    assert all(anchor_id in schema_text for anchor_id in allowed)
    assert "anchor_binding_3821" not in schema_text

    parsed = output_type.model_validate(
        {
            "additional_anchor_overrides": (
                {
                    "anchor_binding_id": allowed[0],
                    "rationale": "Relocate the exact accepted anchor supplied by the host.",
                },
            ),
            "rationale": "Use only the exposed accepted-anchor edit handle.",
        }
    )
    assert parsed.additional_anchor_overrides[0].anchor_binding_id == allowed[0]
    with pytest.raises(ValidationError, match="anchor_binding_0006"):
        output_type.model_validate(
            {
                "additional_anchor_overrides": (
                    {
                        "anchor_binding_id": "anchor_binding_3821",
                        "rationale": "A diagnostic binding ID is not an anchor edit handle.",
                    },
                ),
                "rationale": "Invalid fabricated anchor handle.",
            }
        )

    payload = {
        "anchorBindings": (
            {
                "occurrences": (
                    {"anchorBindingId": allowed[0]},
                    {"anchorBindingId": allowed[1]},
                )
            },
        )
    }
    assert _compiler_repair_allowed_anchor_ids(payload) == allowed


def test_local_compiler_repair_does_not_offer_existing_override_as_an_addition() -> None:
    existing, addable = "anchor_binding_0006", "anchor_binding_0029"
    prior = _compiler_candidate().model_copy(
        update={
            "anchor_overrides": (
                AnchorOverride(
                    anchor_binding_id=existing,
                    rationale="The accepted anchor is already suppressed.",
                ),
            )
        }
    )
    payload = {
        "anchorBindings": (
            {
                "occurrences": (
                    {"anchorBindingId": existing},
                    {"anchorBindingId": addable},
                )
            },
        )
    }

    authorized = _compiler_repair_addable_anchor_ids(payload, prior)
    assert authorized == (addable,)
    output_type = _scoped_compiler_repair_output_type(
        prior,
        addable_anchor_ids=authorized,
    )
    with pytest.raises(ValidationError, match=addable):
        output_type.model_validate(
            {
                "additional_anchor_overrides": (
                    {
                        "anchor_binding_id": existing,
                        "rationale": "Re-adding an existing override cannot change ownership.",
                    },
                ),
                "rationale": "Attempt an operationally empty addition.",
            }
        )


def test_compiler_repair_preview_retries_only_defects_inside_current_transaction() -> None:
    payload = {
        "candidateSlice": {
            "removableBindingKeys": ("hs_code_8_0",),
            "removableAnchorOverrideIds": ("anchor_binding_0040",),
            "removableSemanticOnlyTargetPaths": (),
            "bindings": (
                {
                    "logical_key": "hs_code_8_0",
                    "target_paths": ("documentPatch.cargoGroups[4].hsCodes[1]",),
                    "dependency_paths": (),
                },
            ),
        },
        "targetFacts": (
            {"targetPath": "documentPatch.cargoGroups[4].hsCodes[1]"},
            {"targetPath": "documentPatch.cargoGroups[8].hsCodes[0]"},
        ),
    }

    assert _compiler_repair_error_intersects_scope(
        ValueError("still missing documentPatch.cargoGroups[4].hsCodes[1]"), payload
    )
    assert not _compiler_repair_error_intersects_scope(
        ValueError("latent overlap for documentPatch.cargoPackages[9].quantity"), payload
    )
    assert not _compiler_repair_error_intersects_scope(
        ValueError(
            "selected path documentPatch.cargoGroups[4].hsCodes[1] now exposes latent "
            "documentPatch.cargoPackages[9].quantity"
        ),
        payload,
    )
    assert not _compiler_repair_error_intersects_scope(
        ValueError(
            "selected path documentPatch.cargoGroups[4].hsCodes[1] overlaps "
            "agent_binding_deadbeefdeadbeef"
        ),
        payload,
    )


def test_compiler_repair_preview_does_not_retry_incidental_target_context() -> None:
    selected_path = "documentPatch.cargoGroups[4].hsCodes[1]"
    latent_path = "documentPatch.containers[0].sealNumbers[0]"
    payload = {
        "candidateSlice": {
            "removableBindingKeys": ("hs_code_4_1",),
            "removableAnchorOverrideIds": (),
            "removableSemanticOnlyTargetPaths": (),
            "bindings": (
                {
                    "logical_key": "hs_code_4_1",
                    "target_paths": (selected_path,),
                    "dependency_paths": (),
                },
            ),
        },
        # Context is readable but does not grant authority to replace its existing owner.
        "targetFacts": (
            {"targetPath": selected_path},
            {"targetPath": latent_path},
        ),
    }
    repair_type = _scoped_compiler_repair_output_type(
        _compiler_candidate(), removable_binding_keys=()
    )
    repair = repair_type.model_validate(
        {
            "additional_semantic_only_target_facts": (
                {
                    "target_path": selected_path,
                    "rationale": "The selected fact has no source surface.",
                },
            ),
            "rationale": "Repair only the selected transaction.",
        }
    )

    assert _compiler_repair_error_intersects_scope(
        ValueError(f"selected repair remains invalid: {selected_path}"), payload, repair
    )
    assert not _compiler_repair_error_intersects_scope(
        ValueError(f"unrelated latent binding is invalid: {latent_path}"), payload, repair
    )


def test_compiler_repair_scope_does_not_expand_object_path_to_descendants() -> None:
    object_path = "documentPatch.containers[0]"
    payload = {
        "candidateSlice": {
            "removableBindingKeys": ("container_count:0",),
            "removableAnchorOverrideIds": (),
            "removableSemanticOnlyTargetPaths": (),
            "bindings": (
                {
                    "logical_key": "container_count:0",
                    "target_paths": (object_path,),
                    "dependency_paths": (object_path,),
                },
            ),
        },
        "targetFacts": ({"targetPath": object_path},),
    }

    assert _compiler_repair_error_intersects_scope(
        ValueError(f"invalid exact structural path: {object_path}"), payload
    )
    assert not _compiler_repair_error_intersects_scope(
        ValueError("latent leaf is invalid: documentPatch.containers[0].containerNumber"),
        payload,
    )


def test_compiler_repair_scope_rejects_target_fact_outside_local_vocabulary() -> None:
    output_type = _scoped_compiler_repair_output_type(_compiler_candidate())
    repair = output_type.model_validate(
        {
            "additional_semantic_only_target_facts": (
                {
                    "target_path": "documentPatch.cargoPackages[9].quantity",
                    "rationale": "This path is outside the current repair slice.",
                },
            ),
            "rationale": "Attempt an out-of-scope edit.",
        }
    )
    payload = {
        "targetFacts": ({"targetPath": "documentPatch.cargoGroups[4].hsCodes[1]"},),
        "anchorBindings": (),
        "candidateSlice": {"carrier": None},
    }

    with pytest.raises(ValueError, match="outside its local transaction"):
        _validate_compiler_repair_scope(repair, payload)


def test_compiler_repair_scope_distinguishes_vocabulary_from_actual_edit() -> None:
    prior = _compiler_candidate()
    output_type = _scoped_compiler_repair_output_type(prior)
    target_path = "documentPatch.cargoGroups[6].hsCodes[0]"
    repair = output_type.model_validate(
        {
            "replacement_bindings": (
                {
                    "logical_key": "cargo:6:hsCode:0",
                    "value_kind": "identifier",
                    "group_kind": "cargo",
                    "group_key": "cargo:6",
                    "rendering": {
                        "render_mode": "target_binding",
                        "target_paths": (target_path,),
                    },
                    "occurrences": (
                        {
                            "line_start": "L00223",
                            "line_end": "L00223",
                            "source_text": "2401108590",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Assign the duplicate value to its role-specific cargo row.",
                },
            ),
            "rationale": "Repair the host-cited cross-row ownership collision.",
        }
    )
    payload = {
        "allowedTargetPaths": (target_path,),
        "anchorBindings": (),
    }

    _validate_compiler_repair_scope(repair, payload)

    assert _compiler_repair_allowed_paths(payload) == frozenset((target_path,))
    assert not _compiler_repair_error_intersects_scope(
        ValueError(f"overlap still involves {target_path}"), payload
    )
    assert _compiler_repair_error_intersects_scope(
        ValueError(f"overlap still involves {target_path}"), payload, repair
    )


def test_compiler_repair_scope_rejects_missing_declared_vocabulary() -> None:
    repair = _scoped_compiler_repair_output_type(_compiler_candidate()).model_validate(
        {
            "unresolved_replacement": ("A cited source-only surface remains unresolved.",),
            "rationale": "Record the unresolved surface explicitly.",
        }
    )

    with pytest.raises(ValueError, match="lacks a declared target-path vocabulary"):
        _validate_compiler_repair_scope(repair, {"anchorBindings": ()})


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
    schema_text = json.dumps(output_type.model_json_schema(mode="validation"))
    assert "anchor:documentPatch.billOfLadingNumber" not in schema_text
    assert "review_candidate_0001" not in schema_text
    assert "L00001" not in schema_text
    schema = output_type.model_json_schema(mode="validation")
    coverage_schema = next(
        value
        for key, value in schema["$defs"].items()
        if key.startswith("ScopedCriticAuditCoverage_")
    )
    assert coverage_schema["properties"]["candidate_receipt"]["minItems"] == 1
    assert coverage_schema["properties"]["candidate_receipt"]["maxItems"] == 1
    assert coverage_schema["properties"]["literal_line_count"]["minimum"] == 1
    assert coverage_schema["properties"]["literal_line_count"]["maximum"] == 1
    coverage = {
        "literal_completeness_checked": True,
        "target_ownership_checked": True,
        "topology_and_grouping_checked": True,
        "derivations_checked": True,
        "carrier_boundary_checked": True,
        "identifier_relationships_checked": True,
        "candidate_receipt": ("valid_existing_contract",),
        "literal_line_count": 1,
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
    with pytest.raises(ValidationError, match="at least 1 item"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "pass",
                    "coverage": {**coverage, "candidate_receipt": ()},
                    "rationale": "The candidate receipt is incomplete.",
                }
            }
        )
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "pass",
                    "coverage": {**coverage, "literal_line_count": 0},
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
                        "candidate_receipt": ("defect_requires_revision",),
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


def test_compact_critic_pass_requires_exact_coherence_dispositions() -> None:
    candidate_id = "review_candidate_0001"
    output_type = _scoped_compact_critic_output_type(
        ("anchor:documentPatch.cargoGroups[0].marksAndNumbers[0]",),
        (candidate_id,),
        (),
        ("L00001",),
        coherence_candidate_ids=(candidate_id,),
    )
    coverage = {
        "literal_completeness_checked": True,
        "target_ownership_checked": True,
        "topology_and_grouping_checked": True,
        "derivations_checked": True,
        "carrier_boundary_checked": True,
        "identifier_relationships_checked": True,
        "candidate_receipt": ("valid_existing_contract",),
        "literal_line_count": 1,
    }
    decision = {
        "candidate_id": candidate_id,
        "disposition": "apply_suggestion",
        "suggestion_index": 0,
        "rationale": "The inclusive interval is the package quantity for this cargo group.",
    }

    parsed = output_type.model_validate(
        {
            "decision": {
                "verdict": "pass",
                "coverage": coverage,
                "coherence_decisions": (decision,),
                "rationale": "The template and arithmetic relationship are complete.",
            }
        }
    )
    assert parsed.decision.coherence_decisions[0].candidate_id == candidate_id

    with pytest.raises(ValidationError, match="Field required"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "pass",
                    "coverage": coverage,
                    "rationale": "The required semantic decision was omitted.",
                }
            }
        )
    revised = output_type.model_validate(
        {
            "decision": {
                "verdict": "revise",
                "coverage": {
                    **coverage,
                    "candidate_receipt": ("defect_requires_revision",),
                },
                "findings": (
                    {
                        "finding_kind": "incorrect_semantic_owner",
                        "line_ids": ("L00001",),
                        "evidence": "PACKAGE 1-7",
                        "explanation": "Fixture structural finding.",
                    },
                ),
                "coherence_decisions": (decision,),
                "rationale": "The receipt is an audit judgment; the decision remains explicit.",
            }
        }
    )
    assert revised.decision.coverage.candidate_receipt == ("defect_requires_revision",)

    wrong_id = {**decision, "candidate_id": "cross_field_0001"}
    with pytest.raises(ValidationError, match="review_candidate_0001"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "pass",
                    "coverage": coverage,
                    "coherence_decisions": (wrong_id,),
                    "rationale": "A renamed coherence handle is invalid at the schema boundary.",
                }
            }
        )


def test_compact_critic_revision_requires_exact_literal_line_count() -> None:
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
        "candidate_receipt": (),
        "literal_line_count": 2,
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
    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "revise",
                    "coverage": {
                        **coverage,
                        "literal_line_count": 1,
                    },
                    "findings": (finding,),
                    "rationale": "The finding and receipt disagree.",
                }
            }
        )


def test_critic_provider_view_uses_one_annotated_source_and_omits_host_provenance() -> None:
    compact = {
        "annotatedSource": "L00001 | A ⟦binding_0000⟧VALUE⟦/binding⟧ B",
        "otherField": "unrelated field remains untouched",
        "maskedTemplate": "L00001 | A ⟦binding_0000⟧ B",
        "allowedRemovalLogicalKeys": ("agent:value",),
        "sourceBindingIdTable": ("source_binding_verbose_id",),
        "compactContract": {
            "bindingColumns": ("sourceBindingIds", "logicalKey"),
            "occurrenceColumns": ("sourceBindingId", "sourceText"),
            "sourceBindingIdEncoding": "zero-based index into sourceBindingIdTable",
        },
        "bindingRows": (((0,), "agent:value"),),
        "occurrenceRows": ((0, "VALUE"),),
    }

    provider_view = _critic_provider_payload(compact)

    assert provider_view["annotatedSource"] == compact["annotatedSource"]
    assert "maskedTemplate" not in provider_view
    assert "allowedRemovalLogicalKeys" not in provider_view
    assert "sourceBindingIdTable" not in provider_view
    assert provider_view["compactContract"]["bindingColumns"] == ("logicalKey",)
    assert provider_view["compactContract"]["occurrenceColumns"] == ("sourceText",)
    assert provider_view["bindingRows"] == (("agent:value",),)
    assert provider_view["occurrenceRows"] == (("VALUE",),)
    assert provider_view["otherField"] == "unrelated field remains untouched"
    assert compact["maskedTemplate"] == "L00001 | A ⟦binding_0000⟧ B"
    with pytest.raises(ValueError, match="requires annotated and masked source views"):
        _critic_provider_payload({"maskedTemplate": "L00001 | literal"})


def test_hybrid_critic_routes_only_requests_strictly_above_measured_threshold() -> None:
    compact = {
        "annotatedSource": "L00001 | A ⟦binding_0000⟧VALUE⟦/binding⟧ B",
        "maskedTemplate": "L00001 | A ⟦binding_0000⟧ B",
        "allowedRemovalLogicalKeys": ("agent:value",),
        "sourceBindingIdTable": ("source_binding_verbose_id",),
        "compactContract": {
            "bindingColumns": ("sourceBindingIds", "logicalKey"),
            "occurrenceColumns": ("sourceBindingId", "sourceText"),
            "sourceBindingIdEncoding": "zero-based index into sourceBindingIdTable",
        },
        "bindingRows": (((0,), "agent:value"),),
        "occurrenceRows": ((0, "VALUE"),),
    }
    measured = _critic_provider_request_bytes(compact)

    assert not _uses_partitioned_critic(
        protocol="hybrid_reference_partitioned_v10",
        compact_payload=compact,
        threshold_bytes=measured,
    )
    assert _uses_partitioned_critic(
        protocol="hybrid_reference_partitioned_v10",
        compact_payload=compact,
        threshold_bytes=measured - 1,
    )
    assert not _uses_partitioned_critic(
        protocol="relational_reference_compact_v9",
        compact_payload=compact,
        threshold_bytes=None,
    )
    with pytest.raises(ValueError, match="non-hybrid"):
        _uses_partitioned_critic(
            protocol="relational_reference_compact_v9",
            compact_payload=compact,
            threshold_bytes=measured,
        )


@pytest.mark.asyncio
async def test_hybrid_compact_critic_previews_revision_before_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixtureCompactOutput(BaseModel):
        value: str

    runtime = object.__new__(AgentRuntime)
    runtime._agent_contract_protocol = "hybrid_reference_partitioned_v10"
    runtime._partition_critic_above_request_bytes = 1_000_000
    runtime._providers = {}
    runtime._critic_substage_cache = {}
    compact_payload = {
        "reviewCandidates": (),
        "remainingRiskCandidates": (),
        "literalLineReviewIds": ("L00001",),
    }
    translated = CriticAgentOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (
                {
                    "finding_kind": "unowned_shipment_fact",
                    "line_ids": ("L00001",),
                    "evidence": "VALUE",
                    "explanation": "The fixture requires host preview.",
                },
            ),
            "additional_bindings": (),
            "rationale": "Preview the compact transaction.",
        }
    )
    previewed: list[CriticAgentOutput] = []
    validator_was_present = False

    monkeypatch.setattr(agents_module, "compact_critic_payload", lambda _payload: compact_payload)
    monkeypatch.setattr(agents_module, "_critic_provider_payload", lambda _payload: {})
    monkeypatch.setattr(agents_module, "_reference_occurrence_rows", lambda _payload: ())
    monkeypatch.setattr(
        agents_module,
        "_compact_inventory_occurrence_lookup",
        lambda _payload: {},
    )
    monkeypatch.setattr(
        agents_module,
        "_scoped_compact_critic_output_type",
        lambda *_args, **_kwargs: FixtureCompactOutput,
    )
    monkeypatch.setattr(
        agents_module,
        "_restore_reference_compact_critic",
        lambda *_args, **_kwargs: translated,
    )

    async def fake_call(**kwargs: object) -> tuple[BaseModel, AgentStageArtifact]:
        nonlocal validator_was_present
        output = FixtureCompactOutput(value="fixture")
        validator = kwargs.get("output_validator")
        validator_was_present = validator is not None
        assert callable(validator)
        assert validator(output) == output
        timestamp = datetime.now(UTC)
        stage = AgentStageArtifact.model_validate(
            {
                "role": "critic",
                "pass_number": 1,
                "started_at": timestamp,
                "completed_at": timestamp,
                "duration_seconds": 0.0,
                "system_prompt_sha256": "a" * 64,
                "user_prompt_sha256": "b" * 64,
                "output_schema_sha256": "c" * 64,
                "status": "success",
                "output": output.model_dump(mode="json"),
                "error_type": None,
                "error_message": None,
                "messages": [],
                "usage": agents_module.empty_usage(),
            }
        )
        return output, stage

    monkeypatch.setattr(runtime, "_call", fake_call)
    output, _stage = await runtime.critic(
        pass_number=1,
        system_prompt="critic",
        payload={"allowedRemovalLogicalKeys": ("agent:value",)},
        retries=1,
        preview_review=previewed.append,
    )

    assert validator_was_present is True
    assert previewed == [translated]
    assert output == translated


@pytest.mark.asyncio
async def test_partitioned_critic_reuses_successful_facets_only_for_exact_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = object.__new__(AgentRuntime)
    runtime._agent_contract_protocol = "hybrid_reference_partitioned_v10"
    runtime._partition_critic_above_request_bytes = 1
    runtime._providers = {
        "critic": SimpleNamespace(
            model_dump=lambda **_kwargs: {"kind": "fixture", "model": "fixture-model"}
        )
    }
    runtime._critic_substage_cache = {}
    calls: list[str] = []
    cargo_failures = 0

    def usage(requests: int):
        return agents_module.empty_usage().model_copy(
            update={"requests": requests, "finishReasons": ("stop",) * requests}
        )

    def stage(*, output: object | None, status: str = "success") -> AgentStageArtifact:
        now = datetime.now(UTC)
        return AgentStageArtifact.model_validate(
            {
                "role": "critic",
                "pass_number": 1,
                "started_at": now,
                "completed_at": now,
                "duration_seconds": 0.0,
                "system_prompt_sha256": "a" * 64,
                "user_prompt_sha256": "b" * 64,
                "output_schema_sha256": "c" * 64,
                "status": status,
                "output": output.model_dump(mode="json") if output is not None else None,
                "error_type": "FixtureFailure" if output is None else None,
                "error_message": "fixture cargo failure" if output is None else None,
                "messages": [],
                "usage": usage(1),
            }
        )

    facet_coverage = {
        "assigned_bindings_checked": True,
        "assigned_candidates_checked": True,
        "assigned_literal_lines_checked": True,
        "assigned_target_relationships_checked": True,
    }

    async def fake_call(**kwargs: object):
        nonlocal cargo_failures
        output_type = kwargs["output_type"]
        payload = kwargs["payload"]
        validator = kwargs.get("output_validator")
        if output_type is StagedFacetAuditOutput:
            name = payload["auditFacet"]["name"]
            calls.append(name)
            if name == "cargo_topology" and cargo_failures == 0:
                cargo_failures += 1
                return None, stage(output=None, status="provider_error")
            output = StagedFacetAuditOutput.model_validate(
                {
                    "facet": name,
                    "verdict": "pass",
                    "findings": (),
                    "candidate_dispositions": (),
                    "coverage": facet_coverage,
                    "rationale": f"{name} passed.",
                }
            )
        else:
            calls.append("plan")
            output = StagedCriticPlanOutput.model_validate(
                {
                    "remove_binding_logical_keys": ("fixture_owner",),
                    "rationale": "Apply the fixture audit transaction.",
                }
            )
        if validator is not None:
            output = validator(output)
        return output, stage(output=output)

    monkeypatch.setattr(agents_module, "compact_critic_payload", lambda payload: payload)
    monkeypatch.setattr(agents_module, "_uses_partitioned_critic", lambda **_kwargs: True)
    monkeypatch.setattr(
        agents_module,
        "build_staged_audit_payload",
        lambda payload: {"stateRevision": payload["stateRevision"]},
    )
    monkeypatch.setattr(
        agents_module,
        "build_staged_audit_facet_payloads",
        lambda payload, **_kwargs: tuple(
            {
                "stateRevision": payload["stateRevision"],
                "auditFacet": {"name": name},
            }
            for name in (
                "surface_completeness",
                "cargo_topology",
                "document_topology",
            )
        ),
    )
    monkeypatch.setattr(agents_module, "compact_staged_audit_request", lambda payload: payload)
    monkeypatch.setattr(
        agents_module,
        "normalize_staged_facet_audit",
        lambda output, _payload: output,
    )
    audit = StagedAuditOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (
                {
                    "finding_kind": "unowned_shipment_fact",
                    "line_ids": ("L00001",),
                    "evidence": "fixture",
                    "explanation": "The fixture requires one transaction.",
                },
            ),
            "coverage": {
                "literal_completeness_checked": True,
                "target_ownership_checked": True,
                "topology_and_grouping_checked": True,
                "derivations_checked": True,
                "carrier_boundary_checked": True,
                "identifier_relationships_checked": True,
            },
            "rationale": "Merged fixture audit.",
        }
    )
    monkeypatch.setattr(agents_module, "merge_staged_facet_audits", lambda **_kwargs: audit)
    monkeypatch.setattr(
        agents_module,
        "build_staged_plan_payload",
        lambda **_kwargs: {
            "stateRevision": _kwargs["audit_payload"]["stateRevision"],
            "allowedRemovalLogicalKeys": ("fixture_owner",),
        },
    )
    monkeypatch.setattr(agents_module, "compact_staged_plan_request", lambda payload: payload)
    monkeypatch.setattr(
        agents_module,
        "_scoped_staged_plan_output_type",
        lambda _payload: StagedCriticPlanOutput,
    )
    translated = CriticAgentOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (
                {
                    "finding_kind": "unowned_shipment_fact",
                    "line_ids": ("L00001",),
                    "evidence": "fixture",
                    "explanation": "The fixture requires one transaction.",
                },
            ),
            "additional_bindings": (),
            "rationale": "Translated fixture transaction.",
        }
    )
    monkeypatch.setattr(agents_module, "restore_staged_plan", lambda **_kwargs: translated)
    runtime._call = fake_call

    common = {
        "system_prompt": "critic",
        "retries": 1,
        "audit_prompt": "audit",
        "plan_prompt": "plan",
        "preview_review": lambda _review: None,
    }
    first, first_stage = await runtime.critic(
        pass_number=1,
        payload={"allowedRemovalLogicalKeys": (), "stateRevision": "state-a"},
        **common,
    )
    second, second_stage = await runtime.critic(
        pass_number=2,
        payload={"allowedRemovalLogicalKeys": (), "stateRevision": "state-a"},
        **common,
    )
    third, third_stage = await runtime.critic(
        pass_number=3,
        payload={"allowedRemovalLogicalKeys": (), "stateRevision": "state-b"},
        **common,
    )

    assert first is None
    assert first_stage.usage.requests == 3
    assert second == translated
    assert second_stage.usage.requests == 2
    assert third == translated
    assert third_stage.usage.requests == 4
    assert calls == [
        "surface_completeness",
        "cargo_topology",
        "document_topology",
        "cargo_topology",
        "plan",
        "surface_completeness",
        "cargo_topology",
        "document_topology",
        "plan",
    ]


def test_staged_plan_schema_uses_exact_request_scoped_candidate_handles() -> None:
    occurrence_id = "cand_L00123_00456"
    output_type = _scoped_staged_plan_output_type(
        {
            "occurrenceCandidates": ({"occurrenceId": occurrence_id},),
            "candidateRows": (),
        }
    )
    candidate = {
        "occurrence_appends": (
            {
                "logical_key": "agent:reference",
                "occurrences": ({"occurrence_id": occurrence_id},),
                "rationale": "Append the exact host-enumerated repeated surface.",
            },
        ),
        "rationale": "Apply the cited occurrence repair.",
    }

    parsed = output_type.model_validate(candidate)

    assert parsed.occurrence_appends[0].occurrences[0].occurrence_id == occurrence_id
    schema = json.dumps(output_type.model_json_schema(mode="validation"), sort_keys=True)
    assert occurrence_id in schema
    assert "compiler_occurrence_" not in schema
    candidate["occurrence_appends"][0]["occurrences"] = (
        {"occurrence_id": "compiler_occurrence_00456"},
    )
    with pytest.raises(ValidationError, match="cand_L00123_00456"):
        output_type.model_validate(candidate)


def test_staged_plan_schema_requires_exact_host_coherence_candidate_ids() -> None:
    candidate_id = "review_candidate_0042"
    output_type = _scoped_staged_plan_output_type(
        {
            "occurrenceCandidates": (),
            "candidateRows": (
                {
                    "candidateId": candidate_id,
                    "kind": "cross_field_semantic_relation",
                    "requiredRevision": True,
                },
                {
                    "candidateId": "review_candidate_0043",
                    "kind": "unowned_exact_repeat",
                    "requiredRevision": True,
                },
            ),
        }
    )
    decision = {
        "candidate_id": candidate_id,
        "disposition": "reviewed_independent",
        "rationale": "The two values are independently mutable facts.",
    }
    parsed = output_type.model_validate(
        {"coherence_decisions": (decision,), "rationale": "Resolve the cited relation."}
    )

    assert parsed.coherence_decisions[0].candidate_id == candidate_id
    schema = json.dumps(output_type.model_json_schema(mode="validation"), sort_keys=True)
    assert candidate_id in schema
    assert "candidate_N" not in schema
    with pytest.raises(ValidationError, match=candidate_id):
        output_type.model_validate(
            {
                "coherence_decisions": ({**decision, "candidate_id": "candidate_42"},),
                "rationale": "An invented shorthand is invalid.",
            }
        )
    with pytest.raises(ValidationError, match="Field required"):
        output_type.model_validate({"rationale": "The required decision was omitted."})


def test_staged_plan_schema_materializes_unambiguous_audited_coherence() -> None:
    candidate_id = "review_candidate_0014"
    candidate_row = {
        "candidateIndex": 0,
        "candidateId": candidate_id,
        "kind": "cross_field_semantic_relation",
        "requiredRevision": True,
        "details": {
            "suggestedContracts": (
                {
                    "kind": "summed_numeric_value",
                    "memberLogicalKeys": ("agent:tare_weight:3",),
                    "dependencyPaths": (
                        "documentPatch.cargoAllocationGroups[3].allocations[0]"
                        ".packageQuantity",
                    ),
                },
            ),
            "ambiguousAlternatives": False,
        },
    }
    output_type = _scoped_staged_plan_output_type(
        {
            "occurrenceCandidates": (),
            "candidateRows": (candidate_row,),
            "findings": (
                {
                    "finding_kind": "missing_coherence_dependency",
                    "candidate_indexes": (0,),
                },
            ),
        }
    )

    parsed = output_type.model_validate(
        {
            "coherence_decisions": (),
            "rationale": "The host already proved the only valid repair.",
        }
    )

    assert len(parsed.coherence_decisions) == 1
    assert parsed.coherence_decisions[0].candidate_id == candidate_id
    assert parsed.coherence_decisions[0].disposition == "apply_suggestion"
    assert parsed.coherence_decisions[0].suggestion_index == 0
    parsed_wire_json = output_type.model_validate_json(
        json.dumps(
            {
                "coherence_decisions": [
                    {
                        "candidate_id": candidate_id,
                        "disposition": "apply_suggestion",
                        "suggestion_index": 0,
                        "rationale": "The provider independently selected the same repair.",
                    }
                ],
                "rationale": "Accept the exact JSON wire representation.",
            }
        )
    )
    assert parsed_wire_json.coherence_decisions[0].candidate_id == candidate_id
    with pytest.raises(ValidationError, match="contradicts audit-proved"):
        output_type.model_validate(
            {
                "coherence_decisions": (
                    {
                        "candidate_id": candidate_id,
                        "disposition": "reviewed_independent",
                        "rationale": "Contradict the unique host-proved repair.",
                    },
                ),
                "rationale": "This transaction must be rejected.",
            }
        )


def test_staged_plan_schema_keeps_ambiguous_audited_coherence_provider_required() -> None:
    candidate_id = "review_candidate_0015"
    output_type = _scoped_staged_plan_output_type(
        {
            "occurrenceCandidates": (),
            "candidateRows": (
                {
                    "candidateIndex": 0,
                    "candidateId": candidate_id,
                    "kind": "cross_field_semantic_relation",
                    "requiredRevision": True,
                    "details": {
                        "suggestedContracts": (
                            {"kind": "numeric_values", "memberLogicalKeys": ("first",)},
                            {"kind": "numeric_values", "memberLogicalKeys": ("second",)},
                        ),
                        "ambiguousAlternatives": True,
                    },
                },
            ),
            "findings": (
                {
                    "finding_kind": "missing_coherence_dependency",
                    "candidate_indexes": (0,),
                },
            ),
        }
    )

    with pytest.raises(ValidationError, match="Field required"):
        output_type.model_validate({"rationale": "No decision was supplied."})


def test_compact_critic_defers_required_risk_truth_to_the_host_gate() -> None:
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
        "literal_line_count": 1,
    }
    parsed = output_type.model_validate(
        {
            "decision": {
                "verdict": "revise",
                "coverage": {
                    **base_coverage,
                    "candidate_receipt": ("valid_existing_contract",),
                },
                "findings": (
                    {
                        "finding_kind": "unowned_shipment_fact",
                        "line_ids": ("L00001",),
                        "evidence": "REFERENCE 12345",
                        "explanation": "The repair can now reach deterministic host validation.",
                    },
                ),
                "rationale": "The host, not a repeated model token, proves residual risk state.",
            }
        }
    )
    assert parsed.decision.coverage.candidate_receipt == ("valid_existing_contract",)
    with pytest.raises(ValidationError, match="pass cannot retain"):
        output_type.model_validate(
            {
                "decision": {
                    "verdict": "pass",
                    "coverage": {
                        **base_coverage,
                        "candidate_receipt": ("defect_requires_revision",),
                    },
                    "rationale": "Incorrect pass with an acknowledged defect.",
                }
            }
        )


def test_compact_critic_materializes_only_enumerated_occurrence_handles() -> None:
    occurrence_id = "cand_L00001_00001"
    output_type = _scoped_compact_critic_output_type(
        ("anchor:documentPatch.billOfLadingNumber",),
        (),
        (),
        ("L00001",),
        (occurrence_id,),
    )
    schema_text = json.dumps(output_type.model_json_schema(mode="validation"))
    assert occurrence_id in schema_text
    assert "compiler_occurrence_" not in schema_text
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
                "candidate_receipt": (),
                "literal_line_count": 1,
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
        {"occurrence_id": "cand_L99999_99999"},
    )
    with pytest.raises(ValidationError, match="cand_L00001_00001"):
        output_type.model_validate(candidate)


def test_compact_critic_scopes_and_materializes_partial_occurrence_removals() -> None:
    logical_key = "anchor:documentPatch.billOfLadingNumber"
    other_key = "agent:booking_reference"
    append_occurrence_id = "cand_L00005_00003"
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
            "rows": ((append_occurrence_id, "L00005", "L00005", "BOL123", 0),),
        },
    }
    inventory = _compact_inventory_occurrence_lookup(compact_payload)
    owners = {occurrence_id: owner for occurrence_id, (owner, _row) in inventory.items()}
    output_type = _scoped_compact_critic_output_type(
        (logical_key, other_key),
        (),
        (),
        ("L00001",),
        (append_occurrence_id,),
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
                "candidate_receipt": (),
                "literal_line_count": 1,
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
            "occurrence_appends": (
                {
                    "logical_key": logical_key,
                    "occurrences": ({"occurrence_id": append_occurrence_id},),
                    "rationale": "Add the correct replacement occurrence to the retained binding.",
                },
            ),
            "rationale": "Apply the exact partial removal.",
        }
    }

    parsed = output_type.model_validate(candidate)
    restored = _restore_reference_compact_critic(
        parsed,
        compact_payload,
        {
            "bindingInventory": (
                {
                    "logicalKey": logical_key,
                    "renderMode": "target_binding",
                    "valueKind": "identifier",
                    "groupKind": "document",
                    "groupKey": "document",
                    "targetPaths": ("documentPatch.billOfLadingNumber",),
                    "derivation": None,
                    "dependencyPaths": (),
                    "dependencyBindings": (),
                },
                {"logicalKey": other_key},
            )
        },
    )

    assert restored.occurrence_removals[0].logical_key == logical_key
    assert restored.occurrence_removals[0].occurrences[0].line_start == "L00003"
    assert restored.additional_bindings[0].logical_key == logical_key
    assert restored.additional_bindings[0].occurrences[0].line_start == "L00005"
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


def test_fixed_reference_compiler_schema_defers_exact_id_membership_to_host() -> None:
    output_type = _fixed_reference_compiler_output_type()
    candidate_id = "compiler_occurrence_00001"
    candidate = {
        "carrier": {
            "canonical_name": "Example Carrier Ltd",
            "aliases": (),
            "evidence_occurrences": ({"occurrence_id": candidate_id},),
            "source": "source_label_confirmed_by_ocr",
            "rationale": "Exact carrier occurrence supplied by the host.",
        },
        "anchor_overrides": (),
        "bindings": (),
        "unresolved": (),
        "semantic_only_target_facts": (),
    }
    parsed = output_type.model_validate(candidate)
    payload = {
        "occurrenceCandidates": {
            "columns": (
                "occurrenceId",
                "lineStart",
                "lineEnd",
                "sourceText",
                "occurrenceIndex",
            ),
            "rows": ((candidate_id, "L00002", "L00002", "Example Carrier Ltd", 0),),
        }
    }

    assert _restore_reference_compiler(parsed, payload).carrier.canonical_name == (
        "Example Carrier Ltd"
    )
    invented = output_type.model_validate(
        {
            **candidate,
            "carrier": {
                **candidate["carrier"],
                "evidence_occurrences": ({"occurrence_id": "compiler_occurrence_99999"},),
            },
        }
    )
    with pytest.raises(ValueError, match="unknown occurrence ID"):
        _restore_reference_compiler(invented, payload)

    dynamic_schema = _scoped_reference_compiler_output_type(
        tuple(f"compiler_occurrence_{index:05d}" for index in range(200))
    ).model_json_schema(mode="validation")
    fixed_schema = output_type.model_json_schema(mode="validation")
    assert len(str(fixed_schema)) < len(str(dynamic_schema)) * 0.7


def test_initial_compiler_schema_limits_anchor_overrides_to_host_review_set() -> None:
    allowed_id = "anchor_binding_0007"
    candidate = {
        "carrier": {
            "canonical_name": "Example Carrier Ltd",
            "aliases": (),
            "evidence_occurrences": ({"occurrence_id": "compiler_occurrence_00001"},),
            "source": "source_label_confirmed_by_ocr",
            "rationale": "Exact carrier occurrence supplied by the host.",
        },
        "anchor_overrides": ({"anchor_binding_id": allowed_id, "rationale": "Composite review."},),
        "bindings": (),
        "unresolved": (),
        "semantic_only_target_facts": (),
    }

    scoped = _scoped_initial_compiler_output_type((allowed_id,))
    assert scoped.model_validate(candidate).anchor_overrides[0].anchor_binding_id == allowed_id

    candidate["anchor_overrides"] = (
        {"anchor_binding_id": "anchor_binding_0008", "rationale": "Unauthorized."},
    )
    with pytest.raises(ValidationError):
        scoped.model_validate(candidate)

    with pytest.raises(ValidationError):
        _scoped_initial_compiler_output_type(()).model_validate(candidate)


def test_v8_initial_compiler_schema_requires_exhaustive_anchor_topology_dispositions() -> None:
    composite_id = "anchor_binding_0007"
    cross_fact_override_id = "anchor_binding_0008"
    cross_fact_retained_id = "anchor_binding_0009"
    scoped = _scoped_initial_compiler_output_type(
        (composite_id, cross_fact_override_id, cross_fact_retained_id),
        required_anchor_override_ids=(composite_id,),
        cross_fact_review_ids=(cross_fact_override_id, cross_fact_retained_id),
    )
    candidate = {
        "carrier": {
            "canonical_name": "Example Carrier Ltd",
            "aliases": (),
            "evidence_occurrences": ({"occurrence_id": "compiler_occurrence_00001"},),
            "source": "source_label_confirmed_by_ocr",
            "rationale": "Exact carrier occurrence supplied by the host.",
        },
        "anchor_overrides": (
            {"anchor_binding_id": composite_id, "rationale": "Composite replacement."},
            {
                "anchor_binding_id": cross_fact_override_id,
                "rationale": "Separate role-specific occurrences exist.",
            },
        ),
        "retained_cross_fact_anchor_ids": (cross_fact_retained_id,),
        "bindings": (),
        "unresolved": (),
        "semantic_only_target_facts": (),
    }

    assert scoped.model_validate(candidate).retained_cross_fact_anchor_ids == (
        cross_fact_retained_id,
    )

    missing = dict(candidate)
    missing["retained_cross_fact_anchor_ids"] = ()
    with pytest.raises(ValidationError, match="omitted cross-fact anchor dispositions"):
        scoped.model_validate(missing)

    conflicting = dict(candidate)
    conflicting["retained_cross_fact_anchor_ids"] = (
        cross_fact_override_id,
        cross_fact_retained_id,
    )
    with pytest.raises(ValidationError, match="both overrides and retains"):
        scoped.model_validate(conflicting)


def test_structured_output_retry_requires_a_complete_replacement_response() -> None:
    message = _structured_output_retry_message(ValueError("fixture rejection"))

    assert "fixture rejection" in message
    assert "replaces the rejected response in full" in message
    assert "do not return only an incremental correction" in message


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
