from __future__ import annotations

import json

import pytest

from raw_text_template_experiment.agents import _scoped_compiler_repair_output_type
from raw_text_template_experiment.compact_contract import compact_critic_payload
from raw_text_template_experiment.models import (
    AgentBindingProposal,
    AgentOccurrence,
    AnchorOverride,
    CompilerAgentOutput,
    SemanticOnlyTargetFactProposal,
)
from raw_text_template_experiment.staged_contract import (
    StagedAuditOutput,
    StagedCriticPlanOutput,
    StagedFacetAuditOutput,
    _augment_proven_repeat_appends,
    _augment_unambiguous_coherence_decisions,
    build_local_compiler_repair_payload,
    build_staged_audit_facet_payloads,
    build_staged_audit_payload,
    build_staged_plan_payload,
    compact_staged_audit_request,
    compact_staged_plan_request,
    expand_staged_audit_request,
    expand_staged_plan_request,
    merge_staged_facet_audits,
    normalize_staged_facet_audit,
    restore_staged_plan,
    staged_schema_sizes,
    validate_staged_audit,
    validate_staged_facet_audit,
)


def _source_payload() -> dict[str, object]:
    key = "anchor:documentPatch.billOfLadingNumber"
    return {
        "documentId": "doc_staged",
        "expectedCarrierName": "Example Carrier Ltd",
        "documentMetadata": {"documentType": "bill_of_lading"},
        "sourceLabel": {
            "documentPatch": {
                "billOfLadingNumber": "ABC123",
                "parties": {"carrier": {"name": "Example Carrier Ltd"}},
            }
        },
        "allowedTargetPaths": [
            "documentPatch.billOfLadingNumber",
            "documentPatch.parties.carrier.name",
        ],
        "allowedRemovalLogicalKeys": [key],
        "requiredTargetCoBindings": [],
        "bindingInventory": [
            {
                "sourceBindingIds": ["anchor_binding_0001"],
                "logicalKey": key,
                "renderMode": "target_binding",
                "valueKind": "identifier",
                "groupKind": "document",
                "groupKey": "document",
                "targetPaths": ["documentPatch.billOfLadingNumber"],
                "targetRelationship": "single_target",
                "independentTargetFactComponents": [["documentPatch.billOfLadingNumber"]],
                "derivation": None,
                "dependencyPaths": [],
                "dependencyBindings": [],
                "occurrences": [
                    {
                        "sourceBindingId": "anchor_binding_0001",
                        "lineStart": "L00002",
                        "lineEnd": "L00002",
                        "sourceText": "ABC123",
                        "occurrenceIndex": 0,
                        "exactMatchCount": 1,
                        "exactMatchCandidates": [],
                    }
                ],
            }
        ],
        "maskedTemplate": f"L00002 | ⟦{key}:target_binding⟧\nL00003 | VAT 987654",
        "annotatedSource": (
            f"L00002 | ⟦{key}:target_binding⟧ABC123⟦/binding⟧\nL00003 | VAT 987654"
        ),
        "literalLineReviewIds": ["L00002", "L00003"],
        "reviewCandidates": [],
        "remainingRiskCandidates": [
            {
                "risk_id": "risk_0001",
                "line_id": "L00003",
                "source_text": "987654",
                "kind": "private_identifier",
                "context": "VAT 987654",
            }
        ],
        "semanticOnlyTargetFacts": [],
        "occurrenceCandidates": {
            "columns": (
                "occurrenceId",
                "lineStart",
                "lineEnd",
                "sourceText",
                "occurrenceIndex",
            ),
            "rows": (("compiler_occurrence_00001", "L00003", "L00003", "987654", 0),),
        },
    }


def _coverage(
    payload: dict[str, object], findings: tuple[dict[str, object], ...]
) -> dict[str, object]:
    return {
        "literal_completeness_checked": True,
        "target_ownership_checked": True,
        "topology_and_grouping_checked": True,
        "derivations_checked": True,
        "carrier_boundary_checked": True,
        "identifier_relationships_checked": True,
    }


def _revise_audit(payload: dict[str, object]) -> StagedAuditOutput:
    findings = (
        {
            "finding_kind": "unowned_private_or_auxiliary_fact",
            "line_ids": ("L00003",),
            "evidence": "987654",
            "explanation": "The private VAT identifier is still literal.",
            "candidate_indexes": (0,),
        },
    )
    return StagedAuditOutput.model_validate(
        {
            "verdict": "revise",
            "findings": findings,
            "coverage": _coverage(payload, findings),
            "rationale": "One private identifier lacks ownership.",
        }
    )


def test_staged_audit_is_fixed_small_and_validates_exact_receipts() -> None:
    compact = compact_critic_payload(_source_payload())
    payload = build_staged_audit_payload(compact)
    audit = _revise_audit(payload)

    assert validate_staged_audit(audit, payload) == audit
    assert payload["annotatedSource"] == "L00002 | ⟦B0⟧ABC123⟦/B⟧\nL00003 | VAT 987654"
    assert payload["targetFacts"][0]["targetPathIndex"] == 0
    assert payload["targetFacts"][0]["targetPath"] == "documentPatch.billOfLadingNumber"
    assert payload["bindings"][0]["bindingIndex"] == 0
    assert payload["bindings"][0]["targetPaths"] == ["documentPatch.billOfLadingNumber"]
    assert payload["bindings"][0]["occurrenceIndexes"] == [0]
    assert payload["occurrences"][0]["occurrenceRowIndex"] == 0
    assert payload["occurrences"][0]["sourceText"] == "ABC123"
    assert "exactMatchCandidates" not in payload["occurrences"][0]
    assert staged_schema_sizes()["audit"] < 4_000
    assert staged_schema_sizes()["facet_audit"] < 5_000
    assert staged_schema_sizes()["plan"] < 9_000

    invalid = audit.model_copy(
        update={"findings": (audit.findings[0].model_copy(update={"candidate_indexes": ()}),)}
    )
    with pytest.raises(ValueError, match="omits proven unowned candidate"):
        validate_staged_audit(invalid, payload)


def test_staged_audit_preserves_compact_candidate_guidance() -> None:
    source = _source_payload()
    source["reviewCandidates"] = [
        {
            "candidateId": "review_candidate_0001",
            "kind": "potential_country_code_derivation",
            "logicalKeys": ["anchor:documentPatch.billOfLadingNumber"],
            "lineIds": ["L00002"],
            "sourceTexts": ["ABC123"],
            "details": {
                "suggestedDerivations": ["country_code"],
                "relationshipToVerify": "Fixture guidance.",
            },
        }
    ]

    payload = build_staged_audit_payload(compact_critic_payload(source))

    assert payload["candidateRows"][0]["requiredRevision"] is True
    optional = next(
        row
        for row in payload["candidateRows"]
        if row["kind"] == "potential_country_code_derivation"
    )
    assert optional == {
        "candidateIndex": 1,
        "candidateId": "review_candidate_0001",
        "kind": "potential_country_code_derivation",
        "requiredRevision": False,
        "bindingIndexes": (0,),
        "lineIds": ["L00002"],
        "sourceTexts": ["ABC123"],
        "context": None,
        "details": {
            "suggestedDerivations": ["country_code"],
            "relationshipToVerify": "Fixture guidance.",
        },
    }


def test_staged_audit_merges_exact_required_risk_into_repeat_candidate() -> None:
    source = _source_payload()
    source["reviewCandidates"] = [
        {
            "candidateId": "review_candidate_0001",
            "kind": "unowned_exact_repeat",
            "logicalKeys": ["anchor:documentPatch.billOfLadingNumber"],
            "lineIds": ["L00003"],
            "sourceTexts": ["987654"],
            "context": ["VAT 987654"],
            "details": {"relationshipToVerify": "Fixture repeat relationship."},
        }
    ]

    payload = build_staged_audit_payload(compact_critic_payload(source))

    assert len(payload["candidateRows"]) == 1
    candidate = payload["candidateRows"][0]
    assert candidate["kind"] == "unowned_exact_repeat"
    assert candidate["requiredRevision"] is True
    assert candidate["bindingIndexes"] == (0,)
    assert candidate["candidateIndex"] == 0
    assert candidate["candidateId"] == "review_candidate_0001"
    assert candidate["details"]["mergedRequiredRiskEvidence"][0]["kind"] == ("private_identifier")


def test_partitioned_facets_cover_literal_lines_and_crop_context() -> None:
    source = _source_payload()
    source["reviewCandidates"] = [
        {
            "candidateId": "review_candidate_0001",
            "kind": "potential_country_code_derivation",
            "logicalKeys": ["anchor:documentPatch.billOfLadingNumber"],
            "lineIds": ["L00002"],
            "sourceTexts": ["ABC123"],
            "details": {"relationshipToVerify": "Fixture relationship."},
        }
    ]
    full_payload = build_staged_audit_payload(compact_critic_payload(source))

    facets = build_staged_audit_facet_payloads(full_payload, partition_literal_review=True)
    (document,) = facets

    assert document["auditFacet"]["name"] == "document_topology"
    assert document["literalLineRanges"] == ("L00002-L00003",)
    assigned = [
        line_id
        for facet in facets
        for value in facet["literalLineRanges"]
        for line_id in (
            [value]
            if "-" not in value
            else [f"L{line:05d}" for line in range(int(value[1:6]), int(value[-5:]) + 1)]
        )
    ]
    assert assigned == ["L00002", "L00003"]
    assert all("L000" in str(facet["annotatedSource"]) for facet in facets)


def test_partitioned_facets_authorize_shared_candidate_evidence_in_each_owner() -> None:
    source = _source_payload()
    source["reviewCandidates"] = [
        {
            "candidateId": "surface_candidate",
            "kind": "unowned_standalone_document_status",
            "logicalKeys": [],
            "lineIds": ["L00003"],
            "sourceTexts": ["987654"],
            "details": {"relationshipToVerify": "Surface fixture."},
        },
        {
            "candidateId": "document_candidate",
            "kind": "potential_country_code_derivation",
            "logicalKeys": ["anchor:documentPatch.billOfLadingNumber"],
            "lineIds": ["L00003"],
            "sourceTexts": ["987654"],
            "details": {"relationshipToVerify": "Document fixture."},
        },
    ]
    full_payload = build_staged_audit_payload(compact_critic_payload(source))

    surface, document = build_staged_audit_facet_payloads(
        full_payload, partition_literal_review=True
    )

    assert surface["literalLineRanges"] == ("L00003",)
    assert document["literalLineRanges"] == ("L00002-L00003",)


def test_repeated_finding_may_select_non_exact_local_semantic_owner() -> None:
    source = _source_payload()
    source["reviewCandidates"] = [
        {
            "candidateId": "repeat_candidate",
            "kind": "unowned_exact_repeat",
            "logicalKeys": ["anchor:documentPatch.billOfLadingNumber"],
            "lineIds": ["L00003"],
            "sourceTexts": ["987654"],
            "details": {"relationshipToVerify": "Repeat fixture."},
        }
    ]
    full_payload = build_staged_audit_payload(compact_critic_payload(source))
    document = next(
        facet
        for facet in build_staged_audit_facet_payloads(full_payload, partition_literal_review=True)
        if facet["auditFacet"]["name"] == "document_topology"
    )
    candidate_index = next(
        row["candidateIndex"]
        for row in document["candidateRows"]
        if row["kind"] == "unowned_exact_repeat"
    )
    finding = {
        "finding_kind": "unowned_repeated_fact",
        "line_ids": ("L00002", "L00003"),
        "evidence": "The selected local owner is cited with the unowned repeat.",
        "explanation": "Local semantic context selects the supplied binding.",
        "binding_indexes": (0,),
        "candidate_indexes": (candidate_index,),
    }
    output = StagedFacetAuditOutput.model_validate(
        {
            "facet": "document_topology",
            "verdict": "revise",
            "findings": (finding,),
            "candidate_dispositions": tuple(
                {
                    "candidate_index": row["candidateIndex"],
                    "conclusion": (
                        "defect_requires_revision"
                        if row["candidateIndex"] == candidate_index
                        else "valid_current_state"
                    ),
                }
                for row in document["candidateRows"]
            ),
            "coverage": {
                "assigned_bindings_checked": True,
                "assigned_candidates_checked": True,
                "assigned_literal_lines_checked": True,
                "assigned_target_relationships_checked": True,
            },
            "rationale": "One repeated projection remains unowned.",
        }
    )

    assert validate_staged_facet_audit(output, document) == output


def test_faceted_audit_partitions_candidates_and_requires_explicit_dispositions() -> None:
    source = _source_payload()
    source["reviewCandidates"] = [
        {
            "candidateId": "review_candidate_0001",
            "kind": "potential_country_code_derivation",
            "logicalKeys": ["anchor:documentPatch.billOfLadingNumber"],
            "lineIds": ["L00002"],
            "sourceTexts": ["ABC123"],
            "details": {"relationshipToVerify": "Fixture relationship."},
        }
    ]
    full_payload = build_staged_audit_payload(compact_critic_payload(source))
    facets = build_staged_audit_facet_payloads(full_payload)

    assert [facet["auditFacet"]["name"] for facet in facets] == [
        "surface_completeness",
        "document_topology",
    ]
    assert [row["candidateIndex"] for row in facets[0]["candidateRows"]] == [0]
    assert [row["candidateIndex"] for row in facets[1]["candidateRows"]] == [1]
    assert facets[0]["bindings"] == []
    assert [row["bindingIndex"] for row in facets[1]["bindings"]] == [0]

    surface_finding = _revise_audit(full_payload).findings[0]
    surface = StagedFacetAuditOutput.model_validate(
        {
            "facet": "surface_completeness",
            "verdict": "revise",
            "findings": (surface_finding,),
            "candidate_dispositions": (
                {"candidate_index": 0, "conclusion": "defect_requires_revision"},
            ),
            "coverage": {
                "assigned_bindings_checked": True,
                "assigned_candidates_checked": True,
                "assigned_literal_lines_checked": True,
                "assigned_target_relationships_checked": True,
            },
            "rationale": "The required surface is unowned.",
        }
    )
    topology = StagedFacetAuditOutput.model_validate(
        {
            "facet": "document_topology",
            "verdict": "pass",
            "findings": (),
            "candidate_dispositions": (
                {"candidate_index": 1, "conclusion": "valid_current_state"},
            ),
            "coverage": {
                "assigned_bindings_checked": True,
                "assigned_candidates_checked": True,
                "assigned_literal_lines_checked": True,
                "assigned_target_relationships_checked": True,
            },
            "rationale": "The document binding topology is valid.",
        }
    )

    assert validate_staged_facet_audit(surface, facets[0]) == surface
    assert validate_staged_facet_audit(topology, facets[1]) == topology
    merged = merge_staged_facet_audits(
        outputs=(surface, topology),
        facet_payloads=facets,
        full_payload=full_payload,
    )
    assert merged.verdict == "revise"
    assert merged.findings == (surface_finding,)

    missing = surface.model_copy(update={"candidate_dispositions": ()})
    with pytest.raises(ValueError, match="candidate disposition mismatch"):
        validate_staged_facet_audit(missing, facets[0])


def test_facet_normalization_derives_redundant_dispositions_from_detailed_findings() -> None:
    source = _source_payload()
    source["reviewCandidates"] = [
        {
            "candidateId": "review_candidate_0001",
            "kind": "potential_country_code_derivation",
            "logicalKeys": ["anchor:documentPatch.billOfLadingNumber"],
            "lineIds": ["L00002"],
            "sourceTexts": ["ABC123"],
            "details": {"relationshipToVerify": "Fixture relationship."},
        }
    ]
    full_payload = build_staged_audit_payload(compact_critic_payload(source))
    document = next(
        facet
        for facet in build_staged_audit_facet_payloads(full_payload)
        if facet["auditFacet"]["name"] == "document_topology"
    )
    candidate_index = document["candidateRows"][0]["candidateIndex"]
    finding = {
        "finding_kind": "topology_or_grouping_error",
        "line_ids": ("L00002",),
        "evidence": "ABC123",
        "explanation": "The supplied candidate needs revision.",
        "binding_indexes": (0,),
        "candidate_indexes": (candidate_index,),
    }
    inconsistent = StagedFacetAuditOutput.model_validate(
        {
            "facet": "document_topology",
            "verdict": "revise",
            "findings": (finding,),
            "candidate_dispositions": (
                {"candidate_index": candidate_index, "conclusion": "valid_current_state"},
            ),
            "coverage": {
                "assigned_bindings_checked": True,
                "assigned_candidates_checked": True,
                "assigned_literal_lines_checked": True,
                "assigned_target_relationships_checked": True,
            },
            "rationale": "The detailed finding is the auditable decision.",
        }
    )

    normalized = normalize_staged_facet_audit(inconsistent, document)

    assert normalized.findings == inconsistent.findings
    assert normalized.candidate_dispositions[0].conclusion == "defect_requires_revision"


def test_facet_normalization_materializes_host_required_candidate_finding() -> None:
    full_payload = build_staged_audit_payload(compact_critic_payload(_source_payload()))
    surface = next(
        facet
        for facet in build_staged_audit_facet_payloads(full_payload)
        if facet["auditFacet"]["name"] == "surface_completeness"
    )
    candidate_index = surface["candidateRows"][0]["candidateIndex"]
    omitted = StagedFacetAuditOutput.model_validate(
        {
            "facet": "surface_completeness",
            "verdict": "pass",
            "findings": (),
            "candidate_dispositions": (
                {"candidate_index": candidate_index, "conclusion": "valid_current_state"},
            ),
            "coverage": {
                "assigned_bindings_checked": True,
                "assigned_candidates_checked": True,
                "assigned_literal_lines_checked": True,
                "assigned_target_relationships_checked": True,
            },
            "rationale": "The model omitted the deterministic candidate.",
        }
    )

    normalized = normalize_staged_facet_audit(omitted, surface)

    assert normalized.verdict == "revise"
    assert normalized.findings[0].candidate_indexes == (candidate_index,)
    assert normalized.candidate_dispositions[0].conclusion == "defect_requires_revision"


def test_facet_normalization_materializes_required_repeat_with_remote_owner_evidence() -> None:
    source = _source_payload()
    source["reviewCandidates"] = [
        {
            "candidateId": "review_candidate_0001",
            "kind": "unowned_exact_repeat",
            "logicalKeys": ["anchor:documentPatch.billOfLadingNumber"],
            "lineIds": ["L00003"],
            "sourceTexts": ["987654"],
            "context": ["VAT 987654"],
            "details": {"relationshipToVerify": "Fixture repeat relationship."},
        }
    ]
    full_payload = build_staged_audit_payload(compact_critic_payload(source))
    document = next(
        facet
        for facet in build_staged_audit_facet_payloads(full_payload)
        if facet["auditFacet"]["name"] == "document_topology"
    )
    candidate_index = document["candidateRows"][0]["candidateIndex"]
    omitted = StagedFacetAuditOutput.model_validate(
        {
            "facet": "document_topology",
            "verdict": "pass",
            "findings": (),
            "candidate_dispositions": (
                {"candidate_index": candidate_index, "conclusion": "valid_current_state"},
            ),
            "coverage": {
                "assigned_bindings_checked": True,
                "assigned_candidates_checked": True,
                "assigned_literal_lines_checked": True,
                "assigned_target_relationships_checked": True,
            },
            "rationale": "The required repeated candidate was omitted.",
        }
    )

    normalized = normalize_staged_facet_audit(omitted, document)

    finding = normalized.findings[0]
    assert finding.finding_kind == "unowned_repeated_fact"
    assert finding.binding_indexes == (0,)
    assert finding.line_ids == ("L00002", "L00003")


def test_candidate_prepass_exposes_only_candidate_closure_and_cannot_claim_full_scope() -> None:
    source = _source_payload()
    full_payload = build_staged_audit_payload(compact_critic_payload(source))

    facets = build_staged_audit_facet_payloads(
        full_payload,
        partition_literal_review=True,
        candidate_only=True,
    )

    assert sum(len(facet["candidateRows"]) for facet in facets) == len(
        full_payload["candidateRows"]
    )
    assert all(facet["auditFacet"]["reviewScope"] == "candidate_prepass" for facet in facets)
    assert all(not facet["auditFacet"]["assignedBindingIndexes"] for facet in facets)
    assert {
        line_id
        for facet in facets
        for line_range in facet["literalLineRanges"]
        for line_id in (
            (line_range,)
            if "-" not in line_range
            else tuple(
                f"L{number:05d}" for number in range(int(line_range[1:6]), int(line_range[-5:]) + 1)
            )
        )
    } == {"L00003"}
    document = next(facet for facet in facets if facet["auditFacet"]["name"] == "document_topology")
    assert document["auditFacet"]["contextBindingIndexes"] == ()
    assert document["candidateRows"][0]["requiredRevision"] is True
    assert document["contract"]["schemaVersion"] == 6

    peer_source = json.loads(json.dumps(source))
    peer_key = "agent:document_peer"
    peer_source["allowedRemovalLogicalKeys"].append(peer_key)
    peer_source["bindingInventory"].append(
        {
            "sourceBindingIds": ["agent_binding_peer"],
            "logicalKey": peer_key,
            "renderMode": "deterministic_auxiliary",
            "valueKind": "identifier",
            "groupKind": "document",
            "groupKey": "document",
            "targetPaths": [],
            "targetRelationship": "source_only",
            "independentTargetFactComponents": [],
            "derivation": None,
            "dependencyPaths": [],
            "dependencyBindings": [],
            "occurrences": [
                {
                    "sourceBindingId": "agent_binding_peer",
                    "lineStart": "L00004",
                    "lineEnd": "L00004",
                    "sourceText": "XYZ789",
                    "occurrenceIndex": 0,
                    "exactMatchCount": 1,
                    "exactMatchCandidates": [],
                }
            ],
        }
    )
    peer_source["maskedTemplate"] += f"\nL00004 | ⟦{peer_key}:deterministic_auxiliary⟧"
    peer_source["annotatedSource"] += (
        f"\nL00004 | ⟦{peer_key}:deterministic_auxiliary⟧XYZ789⟦/binding⟧"
    )
    peer_source["remainingRiskCandidates"] = []
    peer_source["reviewCandidates"] = [
        {
            "candidateId": "review_candidate_0001",
            "kind": "repeated_binding_context_review",
            "logicalKeys": ["anchor:documentPatch.billOfLadingNumber"],
            "lineIds": ["L00002"],
            "sourceTexts": ["ABC123"],
            "details": {"relationshipToVerify": "Fixture candidate."},
        }
    ]
    peer_payload = build_staged_audit_payload(compact_critic_payload(peer_source))
    peer_facets = build_staged_audit_facet_payloads(
        peer_payload,
        partition_literal_review=True,
        candidate_only=True,
    )
    peer_document = next(
        facet for facet in peer_facets if facet["auditFacet"]["name"] == "document_topology"
    )
    assert peer_document["auditFacet"]["contextBindingIndexes"] == (0, 1)

    candidate = peer_payload["candidateRows"][0]
    candidate["details"]["derivationInputBindingIndexes"] = (1,)
    incomplete_finding = {
        "finding_kind": "missing_derivation",
        "line_ids": ("L00002",),
        "evidence": "The fixture projection lacks its declared input.",
        "explanation": "The supplied dependency binding must be cited.",
        "binding_indexes": (0,),
        "candidate_indexes": (0,),
    }
    incomplete = StagedAuditOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (incomplete_finding,),
            "coverage": _coverage(peer_payload, (incomplete_finding,)),
            "rationale": "Incomplete dependency context.",
        }
    )
    with pytest.raises(ValueError, match="omits host-provided derivation input"):
        validate_staged_audit(incomplete, peer_payload)


def test_staged_requests_use_compact_record_tables() -> None:
    source = _source_payload()
    compact_source = compact_critic_payload(source)
    audit_payload = build_staged_audit_payload(compact_source)
    audit_request = compact_staged_audit_request(audit_payload)

    expanded_audit = expand_staged_audit_request(audit_request)
    expected_audit = dict(audit_payload)
    expected_audit["bindings"] = [
        {key: value for key, value in row.items() if key != "independentTargetFactComponentPaths"}
        for row in audit_payload["bindings"]
    ]
    assert expanded_audit == expected_audit
    assert "targetFacts" not in audit_request
    assert audit_request["targetFactTable"]["columns"] == (
        "targetPathIndex",
        "targetPath",
        "sourceValue",
    )
    audit = validate_staged_audit(_revise_audit(audit_payload), audit_payload)
    plan_payload = build_staged_plan_payload(
        audit_payload=audit_payload,
        compact_payload=compact_source,
        audit=audit,
    )
    plan_request = compact_staged_plan_request(plan_payload)

    assert expand_staged_plan_request(plan_request) == plan_payload
    assert "allowedRemovalLogicalKeys" not in plan_request
    assert "occurrenceCandidates" not in plan_request
    assert plan_request["occurrenceCandidateTable"]["rows"]


def test_staged_plan_is_cited_slice_and_restores_occurrence_handle() -> None:
    source = _source_payload()
    compact = compact_critic_payload(source)
    audit_payload = build_staged_audit_payload(compact)
    audit = validate_staged_audit(_revise_audit(audit_payload), audit_payload)
    plan_payload = build_staged_plan_payload(
        audit_payload=audit_payload,
        compact_payload=compact,
        audit=audit,
    )
    plan = StagedCriticPlanOutput.model_validate(
        {
            "additional_bindings": (
                {
                    "logical_key": "agent:customs:vat",
                    "value_kind": "identifier",
                    "group_kind": "customs",
                    "group_key": "customs:vat",
                    "rendering": {"render_mode": "deterministic_auxiliary"},
                    "occurrences": ({"occurrence_id": "compiler_occurrence_00001"},),
                    "rationale": "The source-only VAT is a private generated identifier.",
                },
            ),
            "rationale": "Own the one audited literal identifier.",
        }
    )

    restored = restore_staged_plan(
        plan=plan,
        audit=audit,
        plan_payload=plan_payload,
        compact_payload=compact,
        source_payload=source,
    )

    assert restored.verdict == "revise"
    assert restored.additional_bindings[0].occurrences[0].source_text == "987654"
    assert restored.additional_bindings[0].occurrences[0].line_start == "L00003"
    assert "L00003 | VAT 987654" in plan_payload["sourceWindow"]
    assert plan_payload["targetFacts"][0]["targetPathIndex"] == 0
    assert plan_payload["bindings"][0]["logicalKey"] == ("anchor:documentPatch.billOfLadingNumber")
    assert plan_payload["bindings"][0]["targetPaths"] == ["documentPatch.billOfLadingNumber"]
    assert plan_payload["occurrences"][0]["occurrenceRowIndex"] == 0
    assert plan_payload["candidateRows"][0]["candidateIndex"] == 0
    assert plan_payload["occurrenceCandidates"][0] == {
        "occurrenceId": "compiler_occurrence_00001",
        "lineStart": "L00003",
        "lineEnd": "L00003",
        "sourceText": "987654",
        "occurrenceIndex": 0,
        "currentExactOwnerLogicalKeys": (),
    }
    assert json.dumps(plan_payload, ensure_ascii=False).count("ABC123") <= 3


def test_staged_plan_rejects_auxiliary_regression_for_missing_derivation() -> None:
    source = _source_payload()
    source["remainingRiskCandidates"] = []
    compact = compact_critic_payload(source)
    audit_payload = build_staged_audit_payload(compact)
    findings = (
        {
            "finding_kind": "missing_derivation",
            "line_ids": ("L00002",),
            "evidence": "The cited surface is calculated from the existing input.",
            "explanation": "The output must retain an explicit derivation contract.",
            "binding_indexes": (0,),
            "target_path_indexes": (0,),
        },
    )
    audit = validate_staged_audit(
        StagedAuditOutput.model_validate(
            {
                "verdict": "revise",
                "findings": findings,
                "coverage": _coverage(audit_payload, findings),
                "rationale": "One calculated surface lacks a derivation.",
            }
        ),
        audit_payload,
    )
    plan_payload = build_staged_plan_payload(
        audit_payload=audit_payload,
        compact_payload=compact,
        audit=audit,
    )
    logical_key = "anchor:documentPatch.billOfLadingNumber"
    plan = StagedCriticPlanOutput.model_validate(
        {
            "remove_binding_logical_keys": (logical_key,),
            "additional_bindings": (
                {
                    "logical_key": logical_key,
                    "value_kind": "identifier",
                    "group_kind": "document",
                    "group_key": "document",
                    "rendering": {"render_mode": "deterministic_auxiliary"},
                    "occurrences": (
                        {
                            "line_start": "L00002",
                            "line_end": "L00002",
                            "source_text": "ABC123",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Invalidly drops the required derivation.",
                },
            ),
            "rationale": "Regression fixture.",
        }
    )

    with pytest.raises(ValueError, match="no deterministic_derived replacement"):
        restore_staged_plan(
            plan=plan,
            audit=audit,
            plan_payload=plan_payload,
            compact_payload=compact,
            source_payload=source,
        )


def test_staged_plan_coalesces_decomposed_appends_for_one_owner() -> None:
    source = _source_payload()
    source["annotatedSource"] = str(source["annotatedSource"]) + "\nL00004 | VAT 654321"
    source["literalLineReviewIds"] = ["L00002", "L00003", "L00004"]
    source["remainingRiskCandidates"] = [
        *source["remainingRiskCandidates"],  # type: ignore[misc]
        {
            "risk_id": "risk_0002",
            "line_id": "L00004",
            "source_text": "654321",
            "kind": "private_identifier",
            "context": "VAT 654321",
        },
    ]
    source["occurrenceCandidates"] = {
        "columns": (
            "occurrenceId",
            "lineStart",
            "lineEnd",
            "sourceText",
            "occurrenceIndex",
        ),
        "rows": (
            ("compiler_occurrence_00001", "L00003", "L00003", "987654", 0),
            ("compiler_occurrence_00002", "L00004", "L00004", "654321", 0),
        ),
    }
    compact = compact_critic_payload(source)
    audit_payload = build_staged_audit_payload(compact)
    findings = (
        {
            "finding_kind": "unowned_private_or_auxiliary_fact",
            "line_ids": ("L00003", "L00004"),
            "evidence": "987654 and 654321",
            "explanation": "Both private VAT identifiers are still literal.",
            "candidate_indexes": (0, 1),
        },
    )
    audit = StagedAuditOutput.model_validate(
        {
            "verdict": "revise",
            "findings": findings,
            "coverage": _coverage(audit_payload, findings),
            "rationale": "Two occurrences need the same existing owner.",
        }
    )
    plan_payload = build_staged_plan_payload(
        audit_payload=audit_payload,
        compact_payload=compact,
        audit=audit,
    )
    logical_key = "anchor:documentPatch.billOfLadingNumber"
    plan = StagedCriticPlanOutput.model_validate(
        {
            "occurrence_appends": (
                {
                    "logical_key": logical_key,
                    "occurrences": ({"occurrence_id": "compiler_occurrence_00001"},),
                    "rationale": "Append the first cited occurrence.",
                },
                {
                    "logical_key": logical_key,
                    "occurrences": ({"occurrence_id": "compiler_occurrence_00002"},),
                    "rationale": "Append the second cited occurrence.",
                },
            ),
            "rationale": "Coalesce both physical additions under one owner.",
        }
    )

    restored = restore_staged_plan(
        plan=plan,
        audit=audit,
        plan_payload=plan_payload,
        compact_payload=compact,
        source_payload=source,
    )

    assert len(restored.additional_bindings) == 1
    assert restored.additional_bindings[0].logical_key == logical_key
    assert tuple(row.source_text for row in restored.additional_bindings[0].occurrences) == (
        "987654",
        "654321",
    )


def test_staged_plan_maps_unique_anchor_path_alias_to_cobound_owner() -> None:
    source = _source_payload()
    first_path = "documentPatch.billOfLadingNumber"
    second_path = "documentPatch.parties.carrier.name"
    combined_key = f"anchor:{first_path}|{second_path}"
    source["sourceLabel"] = {
        "documentPatch": {
            "billOfLadingNumber": "ABC123",
            "parties": {"carrier": {"name": "ABC123"}},
        }
    }
    source["allowedRemovalLogicalKeys"] = [combined_key]
    inventory = source["bindingInventory"]
    assert isinstance(inventory, list)
    original_key = inventory[0]["logicalKey"]
    inventory[0]["logicalKey"] = combined_key
    inventory[0]["targetPaths"] = [first_path, second_path]
    inventory[0]["targetRelationship"] = "shared_value_equality"
    inventory[0]["independentTargetFactComponents"] = [[first_path], [second_path]]
    source["maskedTemplate"] = str(source["maskedTemplate"]).replace(
        str(original_key), combined_key
    )
    source["annotatedSource"] = str(source["annotatedSource"]).replace(
        str(original_key), combined_key
    )
    compact = compact_critic_payload(source)
    audit_payload = build_staged_audit_payload(compact)
    audit = validate_staged_audit(_revise_audit(audit_payload), audit_payload)
    plan_payload = build_staged_plan_payload(
        audit_payload=audit_payload,
        compact_payload=compact,
        audit=audit,
    )
    plan = StagedCriticPlanOutput.model_validate(
        {
            "occurrence_appends": (
                {
                    "logical_key": f"anchor:{first_path}",
                    "occurrences": ({"occurrence_id": "compiler_occurrence_00001"},),
                    "rationale": "Append to the uniquely matching existing target owner.",
                },
            ),
            "rationale": "Use the existing co-bound owner.",
        }
    )

    restored = restore_staged_plan(
        plan=plan,
        audit=audit,
        plan_payload=plan_payload,
        compact_payload=compact,
        source_payload=source,
    )

    assert restored.additional_bindings[0].logical_key == combined_key
    assert restored.additional_bindings[0].target_paths == (first_path, second_path)


def test_staged_plan_exposes_only_cited_new_or_carried_occurrences() -> None:
    type_path = "documentPatch.containers[0].typeDescription"
    owner_key = f"anchor:{type_path}"
    auxiliary_key = "agent:host_localized_container_1_type_token"
    source: dict[str, object] = {
        "documentId": "doc_equipment_retarget",
        "expectedCarrierName": "Example Carrier Ltd",
        "documentMetadata": {"documentType": "bill_of_lading"},
        "sourceLabel": {
            "documentPatch": {
                "containers": [{"typeDescription": "1X40'HQ CONTAINER"}],
                "parties": {"carrier": {"name": "Example Carrier Ltd"}},
            }
        },
        "allowedTargetPaths": [type_path],
        "allowedRemovalLogicalKeys": [owner_key, auxiliary_key],
        "requiredTargetCoBindings": [],
        "bindingInventory": [
            {
                "sourceBindingIds": ["agent_binding_0001"],
                "logicalKey": owner_key,
                "renderMode": "target_binding",
                "valueKind": "equipment",
                "groupKind": "equipment",
                "groupKey": "container:0",
                "targetPaths": [type_path],
                "targetRelationship": "single_target",
                "independentTargetFactComponents": [[type_path]],
                "derivation": None,
                "dependencyPaths": [],
                "dependencyBindings": [],
                "occurrences": [
                    {
                        "sourceBindingId": "agent_binding_0001",
                        "lineStart": "L00002",
                        "lineEnd": "L00002",
                        "sourceText": "1X40'HQ CONTAINER",
                        "occurrenceIndex": 0,
                        "exactMatchCount": 1,
                        "exactMatchCandidates": [],
                    }
                ],
            },
            {
                "sourceBindingIds": ["agent_binding_0002"],
                "logicalKey": auxiliary_key,
                "renderMode": "deterministic_auxiliary",
                "valueKind": "equipment",
                "groupKind": "equipment",
                "groupKey": "container:1",
                "targetPaths": [],
                "targetRelationship": "source_only",
                "independentTargetFactComponents": [],
                "derivation": None,
                "dependencyPaths": [],
                "dependencyBindings": [],
                "occurrences": [
                    {
                        "sourceBindingId": "agent_binding_0002",
                        "lineStart": "L00010",
                        "lineEnd": "L00010",
                        "sourceText": "40HQ",
                        "occurrenceIndex": 0,
                        "exactMatchCount": 1,
                        "exactMatchCandidates": [],
                    }
                ],
            },
        ],
        "maskedTemplate": (
            "L00001 | 40HQ\n"
            f"L00002 | ⟦{owner_key}:target_binding⟧\n"
            f"L00010 | ⟦{auxiliary_key}:deterministic_auxiliary⟧"
        ),
        "annotatedSource": (
            "L00001 | 40HQ\n"
            f"L00002 | ⟦{owner_key}:target_binding⟧1X40'HQ CONTAINER⟦/binding⟧\n"
            f"L00010 | ⟦{auxiliary_key}:deterministic_auxiliary⟧40HQ⟦/binding⟧"
        ),
        "literalLineReviewIds": ["L00001"],
        "reviewCandidates": [],
        "remainingRiskCandidates": [
            {
                "risk_id": "risk_0001",
                "line_id": "L00001",
                "source_text": "40HQ",
                "kind": "equipment_type_token",
                "context": "40HQ",
            }
        ],
        "semanticOnlyTargetFacts": [],
        "occurrenceCandidates": {
            "columns": (
                "occurrenceId",
                "lineStart",
                "lineEnd",
                "sourceText",
                "occurrenceIndex",
            ),
            "rows": (
                ("compiler_occurrence_00001", "L00001", "L00001", "40HQ", 0),
                ("compiler_occurrence_00002", "L00010", "L00010", "40HQ", 0),
            ),
        },
    }
    compact = compact_critic_payload(source)
    audit_payload = build_staged_audit_payload(compact)
    findings = (
        {
            "finding_kind": "unowned_repeated_fact",
            "line_ids": ("L00001", "L00002"),
            "evidence": "40HQ",
            "explanation": "The adjacent compact type projection is unowned.",
            "binding_indexes": (0,),
            "candidate_indexes": (0,),
        },
    )
    audit = StagedAuditOutput.model_validate(
        {
            "verdict": "revise",
            "findings": findings,
            "coverage": _coverage(audit_payload, findings),
            "rationale": "The first equipment token needs the existing owner.",
        }
    )
    plan_payload = build_staged_plan_payload(
        audit_payload=audit_payload,
        compact_payload=compact,
        audit=audit,
    )
    assert [row["occurrenceId"] for row in plan_payload["occurrenceCandidates"]] == [
        "compiler_occurrence_00001"
    ]
    plan = StagedCriticPlanOutput.model_validate(
        {
            "occurrence_appends": (
                {
                    "logical_key": owner_key,
                    "occurrences": ({"occurrence_id": "compiler_occurrence_00001"},),
                    "rationale": "The intended token is the adjacent unowned projection.",
                },
            ),
            "rationale": "Correct the equipment projection.",
        }
    )

    restored = restore_staged_plan(
        plan=plan,
        audit=audit,
        plan_payload=plan_payload,
        compact_payload=compact,
        source_payload=source,
    )

    assert restored.additional_bindings[0].occurrences[0].line_start == "L00001"
    assert "Host retargeted an already owned equipment token" not in (
        restored.additional_bindings[0].rationale
    )


def test_staged_plan_normalizes_complete_occurrence_removal_to_full_binding() -> None:
    source = _source_payload()
    compact = compact_critic_payload(source)
    audit_payload = build_staged_audit_payload(compact)
    findings = (
        {
            "finding_kind": "incorrect_semantic_owner",
            "line_ids": ("L00002",),
            "evidence": "ABC123",
            "explanation": "Fixture requests complete replacement.",
            "binding_indexes": (0,),
            "target_path_indexes": (0,),
        },
    )
    audit = StagedAuditOutput.model_validate(
        {
            "verdict": "revise",
            "findings": findings,
            "coverage": _coverage(audit_payload, findings),
            "rationale": "Fixture owner is invalid.",
        }
    )
    plan_payload = build_staged_plan_payload(
        audit_payload=audit_payload,
        compact_payload=compact,
        audit=audit,
    )
    plan = StagedCriticPlanOutput.model_validate(
        {
            "occurrence_removals": (
                {
                    "logical_key": "anchor:documentPatch.billOfLadingNumber",
                    "occurrence_ids": ("occ_L00002_00000",),
                    "rationale": "Remove the complete one-occurrence owner.",
                },
            ),
            "rationale": "The complete occurrence set is the full binding.",
        }
    )

    restored = restore_staged_plan(
        plan=plan,
        audit=audit,
        plan_payload=plan_payload,
        compact_payload=compact,
        source_payload=source,
    )

    assert restored.occurrence_removals == ()
    assert len(restored.remove_inventory_binding_ids) == 1


def test_staged_plan_normalizes_complete_same_key_proposal_to_full_replacement() -> None:
    source = _source_payload()
    compact = compact_critic_payload(source)
    audit_payload = build_staged_audit_payload(compact)
    findings = (
        {
            "finding_kind": "incorrect_semantic_owner",
            "line_ids": ("L00002",),
            "evidence": "ABC123",
            "explanation": "Fixture requests a complete same-key replacement.",
            "binding_indexes": (0,),
            "target_path_indexes": (0,),
        },
    )
    audit = StagedAuditOutput.model_validate(
        {
            "verdict": "revise",
            "findings": findings,
            "coverage": _coverage(audit_payload, findings),
            "rationale": "The current owner requires a complete replacement.",
        }
    )
    plan_payload = build_staged_plan_payload(
        audit_payload=audit_payload,
        compact_payload=compact,
        audit=audit,
    )
    logical_key = "anchor:documentPatch.billOfLadingNumber"
    plan = StagedCriticPlanOutput.model_validate(
        {
            "additional_bindings": (
                {
                    "logical_key": logical_key,
                    "value_kind": "identifier",
                    "group_kind": "document",
                    "group_key": "document",
                    "rendering": {
                        "render_mode": "target_binding",
                        "target_paths": ("documentPatch.billOfLadingNumber",),
                    },
                    "occurrences": (
                        {
                            "line_start": "L00002",
                            "line_end": "L00002",
                            "source_text": "ABC123",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Carry the complete current owner into its replacement.",
                },
            ),
            "rationale": "The full same-key proposal is an atomic replacement.",
        }
    )

    restored = restore_staged_plan(
        plan=plan,
        audit=audit,
        plan_payload=plan_payload,
        compact_payload=compact,
        source_payload=source,
    )

    assert len(restored.remove_inventory_binding_ids) == 1
    assert "Host normalized the complete same-key proposal" in (
        restored.additional_bindings[0].rationale
    )


def test_staged_plan_retargets_wrong_handle_to_unique_cited_owner_occurrence() -> None:
    source = _source_payload()
    target_key = "anchor:documentPatch.billOfLadingNumber"
    other_key = "agent:customs:vat"
    source["allowedRemovalLogicalKeys"] = [target_key, other_key]
    source["bindingInventory"].append(
        {
            "sourceBindingIds": ["agent_binding_0002"],
            "logicalKey": other_key,
            "renderMode": "deterministic_auxiliary",
            "valueKind": "identifier",
            "groupKind": "customs",
            "groupKey": "customs:vat",
            "targetPaths": [],
            "targetRelationship": "none",
            "independentTargetFactComponents": [],
            "derivation": None,
            "dependencyPaths": [],
            "dependencyBindings": [],
            "occurrences": [
                {
                    "sourceBindingId": "agent_binding_0002",
                    "lineStart": "L00003",
                    "lineEnd": "L00003",
                    "sourceText": "987654",
                    "occurrenceIndex": 0,
                    "exactMatchCount": 1,
                    "exactMatchCandidates": [],
                }
            ],
        }
    )
    source["maskedTemplate"] = (
        f"L00002 | ⟦{target_key}:target_binding⟧\n"
        f"L00003 | VAT ⟦{other_key}:deterministic_auxiliary⟧"
    )
    source["annotatedSource"] = (
        f"L00002 | ⟦{target_key}:target_binding⟧ABC123⟦/binding⟧\n"
        f"L00003 | VAT ⟦{other_key}:deterministic_auxiliary⟧987654⟦/binding⟧"
    )
    source["remainingRiskCandidates"] = []
    compact = compact_critic_payload(source)
    audit_payload = build_staged_audit_payload(compact)
    findings = (
        {
            "finding_kind": "incorrect_semantic_owner",
            "line_ids": ("L00002",),
            "evidence": "ABC123",
            "explanation": "Fixture requests removal of the first binding.",
            "binding_indexes": (0,),
            "target_path_indexes": (0,),
        },
    )
    audit = validate_staged_audit(
        StagedAuditOutput.model_validate(
            {
                "verdict": "revise",
                "findings": findings,
                "coverage": _coverage(audit_payload, findings),
                "rationale": "The first binding is invalid.",
            }
        ),
        audit_payload,
    )
    plan_payload = build_staged_plan_payload(
        audit_payload=audit_payload,
        compact_payload=compact,
        audit=audit,
    )
    plan = StagedCriticPlanOutput.model_validate(
        {
            "occurrence_removals": (
                {
                    "logical_key": target_key,
                    "occurrence_ids": ("occ_L00003_00001",),
                    "rationale": "The model selected the adjacent row by mistake.",
                },
            ),
            "rationale": "Remove the cited invalid owner.",
        }
    )

    restored = restore_staged_plan(
        plan=plan,
        audit=audit,
        plan_payload=plan_payload,
        compact_payload=compact,
        source_payload=source,
    )

    assert restored.occurrence_removals == ()
    assert len(restored.remove_inventory_binding_ids) == 1


def test_staged_audit_rejects_existing_binding_selector_outside_cited_lines() -> None:
    payload = build_staged_audit_payload(compact_critic_payload(_source_payload()))
    finding = {
        "finding_kind": "topology_or_grouping_error",
        "line_ids": ("L00003",),
        "evidence": "VAT 987654",
        "explanation": "Fixture deliberately cites a binding from another line.",
        "binding_indexes": (0,),
        "candidate_indexes": (0,),
    }
    audit = StagedAuditOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (finding,),
            "coverage": _coverage(payload, (finding,)),
            "rationale": "Fixture invalid selector.",
        }
    )

    with pytest.raises(ValueError, match="binding 0 outside all finding lines"):
        validate_staged_audit(audit, payload)


def test_staged_plan_rejects_occurrence_outside_repair_slice() -> None:
    source = _source_payload()
    compact = compact_critic_payload(source)
    audit_payload = build_staged_audit_payload(compact)
    audit = validate_staged_audit(_revise_audit(audit_payload), audit_payload)
    plan_payload = build_staged_plan_payload(
        audit_payload=audit_payload,
        compact_payload=compact,
        audit=audit,
    )
    plan = StagedCriticPlanOutput.model_validate(
        {
            "additional_bindings": (
                {
                    "logical_key": "agent:customs:vat",
                    "value_kind": "identifier",
                    "group_kind": "customs",
                    "group_key": "customs:vat",
                    "rendering": {"render_mode": "deterministic_auxiliary"},
                    "occurrences": ({"occurrence_id": "compiler_occurrence_99999"},),
                    "rationale": "Invalid fixture reference.",
                },
            ),
            "rationale": "Invalid fixture plan.",
        }
    )

    with pytest.raises(ValueError, match="outside its repair slice"):
        restore_staged_plan(
            plan=plan,
            audit=audit,
            plan_payload=plan_payload,
            compact_payload=compact,
            source_payload=source,
        )


def _compiler_candidate() -> CompilerAgentOutput:
    return CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": "Example Carrier Ltd",
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00002",
                        "line_end": "L00002",
                        "source_text": "Example Carrier Ltd",
                        "occurrence_index": 0,
                    },
                ),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Exact printed carrier.",
            },
            "anchor_overrides": (),
            "bindings": (
                {
                    "logical_key": "agent:booking",
                    "render_mode": "target_binding",
                    "value_kind": "identifier",
                    "group_kind": "document",
                    "group_key": "document",
                    "target_paths": ("documentPatch.billOfLadingNumber",),
                    "occurrences": (
                        {
                            "line_start": "L00003",
                            "line_end": "L00003",
                            "source_text": "ABC123",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Printed document identifier.",
                },
            ),
            "unresolved": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
        }
    )


def _compiler_occurrence_candidates(
    *rows: tuple[str, str, str, str, int],
) -> dict[str, object]:
    return {
        "columns": ("occurrenceId", "lineStart", "lineEnd", "sourceText", "occurrenceIndex"),
        "rows": rows,
    }


def test_local_compiler_repair_excludes_global_candidate_and_closes_removal_scope() -> None:
    prior = _compiler_candidate()
    source = "\n".join(
        (
            "L00001 | HEADER",
            "L00002 | Example Carrier Ltd",
            "L00003 | ABC123",
            "L00004 | FOOTER",
            "L00005 | UNRELATED",
        )
    )
    payload = {
        "documentId": "doc_repair",
        "expectedCarrierName": "Example Carrier Ltd",
        "numberedSource": source,
        "sourceLabel": {"documentPatch": {"billOfLadingNumber": "ABC123"}},
        "allowedTargetPaths": ("documentPatch.billOfLadingNumber",),
        "requiredRevision": (
            "agent:booking L00003-L00003 selected the wrong exact occurrence for "
            "documentPatch.billOfLadingNumber; "
            'documentMatchRanges=["L00005-L00005"]'
        ),
        "anchorBindings": (),
        "occurrenceCandidates": _compiler_occurrence_candidates(
            ("compiler_occurrence_00001", "L00003", "L00003", "ABC123", 0)
        ),
        "previousCandidateBindingInventory": (),
        "previousCandidateOutput": prior.model_dump(mode="json"),
    }

    local = build_local_compiler_repair_payload(payload, prior, halo_lines=1)

    assert "previousCandidateOutput" not in local
    assert "L00005" not in local["sourceWindow"]
    assert local["occurrenceCandidates"] == [
        {
            "occurrenceId": "compiler_occurrence_00001",
            "lineStart": "L00003",
            "lineEnd": "L00003",
            "sourceText": "ABC123",
            "occurrenceIndex": 0,
            "candidateTargetPaths": ("documentPatch.billOfLadingNumber",),
            "currentExactOwnerLogicalKeys": (),
            "outsideOriginalRepairWindow": False,
        }
    ]
    assert local["candidateSlice"]["removableBindingKeys"] == ["agent:booking"]
    schema = _scoped_compiler_repair_output_type(
        prior,
        removable_binding_keys=("agent:booking",),
        removable_anchor_ids=(),
        removable_semantic_paths=(),
    ).model_json_schema()
    assert schema["properties"]["remove_anchor_override_ids"]["maxItems"] == 0
    assert schema["properties"]["remove_semantic_only_target_paths"]["maxItems"] == 0


def test_local_compiler_repair_includes_distinctive_global_target_alternatives() -> None:
    path_0 = "documentPatch.cargoGroups[0].additionalInformation[0]"
    path_1 = "documentPatch.cargoGroups[1].additionalInformation[0]"
    source_value = "MATERIAL 13672297"
    first_binding = (
        _compiler_candidate()
        .bindings[0]
        .model_copy(
            update={
                "logical_key": "cargo:0:material",
                "target_paths": (path_0,),
                "occurrences": (
                    AgentOccurrence(
                        line_start="L00003",
                        line_end="L00003",
                        source_text=source_value,
                        occurrence_index=0,
                    ),
                ),
            }
        )
    )
    second_binding = first_binding.model_copy(
        update={"logical_key": "cargo:1:material", "target_paths": (path_1,)}
    )
    prior = _compiler_candidate().model_copy(update={"bindings": (first_binding, second_binding)})
    payload = {
        "documentId": "doc_global_alternative",
        "expectedCarrierName": "Example Carrier Ltd",
        "numberedSource": "\n".join(
            (
                "L00001 | HEADER",
                "L00002 | CARGO ROW ZERO",
                f"L00003 | {source_value}",
                "L00004 | SEPARATOR",
                "L00005 | CARGO ROW ONE",
                f"L00006 | {source_value}",
                "L00007 | FOOTER",
            )
        ),
        "sourceLabel": {
            "documentPatch": {
                "cargoGroups": [
                    {
                        "groupId": "g0",
                        "description": "CARGO ROW ZERO",
                        "additionalInformation": [source_value],
                    },
                    {
                        "groupId": "g1",
                        "description": "CARGO ROW ONE",
                        "additionalInformation": [source_value],
                    },
                ]
            }
        },
        "allowedTargetPaths": (path_0, path_1),
        "requiredRevision": (
            f"template spans overlap: cargo:0:material {path_0} L00003 and "
            f"cargo:1:material {path_1} L00003"
        ),
        "anchorBindings": (),
        "occurrenceCandidates": _compiler_occurrence_candidates(
            ("compiler_occurrence_00001", "L00003", "L00003", source_value, 0),
            ("compiler_occurrence_00002", "L00006", "L00006", source_value, 0),
        ),
        "previousCandidateBindingInventory": (
            {
                "logicalKey": "cargo:0:material",
                "targetPaths": (path_0,),
                "occurrences": (
                    {
                        "lineStart": "L00003",
                        "lineEnd": "L00003",
                        "sourceText": source_value,
                        "occurrenceIndex": 0,
                    },
                ),
            },
            {
                "logicalKey": "cargo:1:material",
                "targetPaths": (path_1,),
                "occurrences": (
                    {
                        "lineStart": "L00003",
                        "lineEnd": "L00003",
                        "sourceText": source_value,
                        "occurrenceIndex": 0,
                    },
                ),
            },
        ),
    }

    local = build_local_compiler_repair_payload(payload, prior, halo_lines=1)

    alternatives = {row["occurrenceId"]: row for row in local["occurrenceCandidates"]}
    assert alternatives["compiler_occurrence_00001"]["outsideOriginalRepairWindow"] is False
    assert alternatives["compiler_occurrence_00001"]["currentExactOwnerLogicalKeys"] == (
        "cargo:0:material",
        "cargo:1:material",
    )
    assert alternatives["compiler_occurrence_00002"]["outsideOriginalRepairWindow"] is True
    assert alternatives["compiler_occurrence_00002"]["currentExactOwnerLogicalKeys"] == ()
    assert alternatives["compiler_occurrence_00002"]["candidateTargetPaths"] == (path_0, path_1)
    assert "L00005 | CARGO ROW ONE" in local["sourceWindow"]
    assert "L00006 | MATERIAL 13672297" in local["sourceWindow"]
    assert local["targetContexts"] == [
        {
            "contextId": "target_context_0001",
            "contextPath": "documentPatch.cargoGroups[0]",
            "sourceValue": {
                "groupId": "g0",
                "description": "CARGO ROW ZERO",
                "additionalInformation": [source_value],
            },
        },
        {
            "contextId": "target_context_0002",
            "contextPath": "documentPatch.cargoGroups[1]",
            "sourceValue": {
                "groupId": "g1",
                "description": "CARGO ROW ONE",
                "additionalInformation": [source_value],
            },
        },
    ]
    assert [row["contextId"] for row in local["targetFacts"]] == [
        "target_context_0001",
        "target_context_0002",
    ]


def test_local_compiler_repair_closes_over_canonical_inventory_owner_in_same_entity_block() -> None:
    path_0 = "documentPatch.cargoGroups[0].description"
    path_1 = "documentPatch.cargoGroups[1].description"
    context_path_1 = "documentPatch.cargoGroups[1].hsCodes[0]"
    shared_value = "STEEL COILS HOT ROLLED"
    owner = (
        _compiler_candidate()
        .bindings[0]
        .model_copy(
            update={
                "logical_key": "cargo:0:description",
                "value_kind": "natural_text",
                "group_kind": "cargo",
                "group_key": "cargo:0",
                "target_paths": (path_0,),
                "occurrences": (
                    AgentOccurrence(
                        line_start="L00003",
                        line_end="L00003",
                        source_text=shared_value,
                        occurrence_index=0,
                    ),
                ),
            }
        )
    )
    entity_context = owner.model_copy(
        update={
            "logical_key": "cargo:1:hs",
            "value_kind": "identifier",
            "group_key": "cargo:1",
            "target_paths": (context_path_1,),
            "occurrences": (
                AgentOccurrence(
                    line_start="L00004",
                    line_end="L00004",
                    source_text="7208",
                    occurrence_index=0,
                ),
            ),
        }
    )
    prior = _compiler_candidate().model_copy(update={"bindings": (owner, entity_context)})
    payload = {
        "documentId": "doc_owner_closure",
        "expectedCarrierName": "Example Carrier Ltd",
        "numberedSource": "\n".join(
            (
                "L00001 | HEADER",
                "L00002 | CARGO ROW",
                f"L00003 | {shared_value}",
                "L00004 | 7208",
                "L00005 | FOOTER",
            )
        ),
        "sourceLabel": {
            "documentPatch": {
                "cargoGroups": [
                    {"description": shared_value},
                    {"description": shared_value, "hsCodes": ["7208"]},
                ]
            }
        },
        "allowedTargetPaths": (path_0, path_1, context_path_1),
        "requiredRevision": f"missing replacement ownership near L00004: {path_1}",
        "anchorBindings": (),
        "occurrenceCandidates": _compiler_occurrence_candidates(
            ("compiler_occurrence_00001", "L00003", "L00003", shared_value, 0)
        ),
        "previousCandidateBindingInventory": (
            {
                "logicalKey": "anchor:" + path_0,
                "targetPaths": (path_0,),
                "occurrences": (
                    {
                        "lineStart": "L00003",
                        "lineEnd": "L00003",
                        "sourceText": shared_value,
                        "occurrenceIndex": 0,
                    },
                ),
            },
            {
                "logicalKey": "anchor:" + context_path_1,
                "targetPaths": (context_path_1,),
                "occurrences": (
                    {
                        "lineStart": "L00004",
                        "lineEnd": "L00004",
                        "sourceText": "7208",
                        "occurrenceIndex": 0,
                    },
                ),
            },
        ),
    }

    local = build_local_compiler_repair_payload(payload, prior, halo_lines=1)

    assert local["candidateSlice"]["removableBindingKeys"] == [
        "cargo:0:description",
        "cargo:1:hs",
    ]
    assert local["occurrenceCandidates"][0]["currentExactOwnerLogicalKeys"] == (
        "cargo:0:description",
    )
    assert {row["targetPath"] for row in local["targetFacts"]} == {
        path_0,
        path_1,
        context_path_1,
    }


def test_local_compiler_repair_closes_over_source_only_owner_in_target_group() -> None:
    country_path = "documentPatch.parties.notifyParties[0].country"
    owner = (
        _compiler_candidate()
        .bindings[0]
        .model_copy(
            update={
                "logical_key": "notify_country",
                "render_mode": "deterministic_auxiliary",
                "value_kind": "location",
                "group_kind": "party",
                "group_key": "party:notify:0",
                "target_paths": (),
                "occurrences": (
                    AgentOccurrence(
                        line_start="L00004",
                        line_end="L00004",
                        source_text="EGYPT",
                        occurrence_index=0,
                    ),
                ),
            }
        )
    )
    prior = _compiler_candidate().model_copy(update={"bindings": (owner,)})
    payload = {
        "documentId": "doc_source_only_owner_closure",
        "expectedCarrierName": "Example Carrier Ltd",
        "numberedSource": "\n".join(
            (
                "L00001 | HEADER",
                "L00002 | CONSIGNEE EGYPT",
                "L00003 | NOTIFY PARTY",
                "L00004 | *EGYPT",
                "L00005 | FOOTER",
            )
        ),
        "sourceLabel": {"documentPatch": {"parties": {"notifyParties": [{"country": "EGYPT"}]}}},
        "allowedTargetPaths": (country_path,),
        "requiredRevision": (
            f"anchor override lacks replacement target ownership: {country_path}; "
            "unowned exact candidates: none"
        ),
        "anchorBindings": (
            {
                "logicalKey": f"anchor:{country_path}",
                "renderMode": "target_binding",
                "valueKind": "location",
                "groupKind": "party",
                "groupKey": "party:notify:0",
                "targetPaths": (country_path,),
                "targetRelationship": "single_target",
                "independentTargetFactComponents": ((country_path,),),
                "renderPolicy": "natural_text",
                "occurrences": (
                    {
                        "anchorBindingId": "anchor_binding_0001",
                        "lineStart": "L00002",
                        "lineEnd": "L00002",
                        "sourceText": "EGYPT",
                    },
                ),
            },
        ),
        "requiredTargetCoBindings": (),
        "occurrenceCandidates": _compiler_occurrence_candidates(
            ("compiler_occurrence_00001", "L00004", "L00004", "EGYPT", 0)
        ),
        "previousCandidateBindingInventory": (
            {
                "logicalKey": "agent:notify_country",
                "targetPaths": (),
                "occurrences": (
                    {
                        "lineStart": "L00004",
                        "lineEnd": "L00004",
                        "sourceText": "EGYPT",
                        "occurrenceIndex": 0,
                    },
                ),
            },
        ),
    }

    local = build_local_compiler_repair_payload(payload, prior, halo_lines=1)

    assert local["candidateSlice"]["removableBindingKeys"] == ["notify_country"]
    assert local["occurrenceCandidates"][0]["currentExactOwnerLogicalKeys"] == ("notify_country",)


def test_local_compiler_repair_does_not_select_parent_collection_from_leaf_path_text() -> None:
    leaf_path = "documentPatch.cargoPackages[0].quantity"
    collection_path = "documentPatch.cargoPackages"
    base = _compiler_candidate().bindings[0]
    leaf = base.model_copy(
        update={
            "logical_key": "package_quantity_0",
            "value_kind": "integer",
            "group_kind": "package",
            "group_key": "package:0",
            "target_paths": (leaf_path,),
            "occurrences": (
                AgentOccurrence(
                    line_start="L00003",
                    line_end="L00003",
                    source_text="96",
                    occurrence_index=0,
                ),
            ),
        }
    )
    collection = base.model_copy(
        update={
            "logical_key": "package_total",
            "render_mode": "deterministic_derived",
            "value_kind": "integer",
            "group_kind": "document",
            "group_key": "document:totals",
            "target_paths": (collection_path,),
            "derivation": "package_count",
            "dependency_paths": (collection_path,),
            "occurrences": (
                AgentOccurrence(
                    line_start="L00002",
                    line_end="L00002",
                    source_text="10 PACKAGES",
                    occurrence_index=0,
                ),
            ),
        }
    )
    prior = _compiler_candidate().model_copy(update={"bindings": (leaf, collection)})
    payload = {
        "documentId": "doc_leaf_scope",
        "expectedCarrierName": "Example Carrier Ltd",
        "numberedSource": "\n".join(
            ("L00001 | HEADER", "L00002 | 10 PACKAGES", "L00003 | TGBU9219649")
        ),
        "sourceLabel": {"documentPatch": {"cargoPackages": [{"quantity": 96}]}},
        "allowedTargetPaths": (collection_path, leaf_path),
        "requiredRevision": f"overlap at L00003 for {leaf_path}",
        "anchorBindings": (),
        "occurrenceCandidates": _compiler_occurrence_candidates(
            ("compiler_occurrence_00001", "L00003", "L00003", "96", 0)
        ),
        "previousCandidateBindingInventory": (),
    }

    local = build_local_compiler_repair_payload(payload, prior, halo_lines=1)

    assert local["candidateSlice"]["removableBindingKeys"] == ["package_quantity_0"]
    assert collection_path not in {row["targetPath"] for row in local["targetFacts"]}


def test_local_compiler_repair_selects_prior_binding_with_invalid_target_path() -> None:
    invalid_path = "documentPatch.cargoGroups[5].hsCodes[1]"
    prior = _compiler_candidate().model_copy(
        update={
            "bindings": (
                _compiler_candidate()
                .bindings[0]
                .model_copy(
                    update={
                        "logical_key": "hs_code_5_1",
                        "target_paths": (invalid_path,),
                    }
                ),
            )
        }
    )
    payload = {
        "documentId": "doc_repair",
        "expectedCarrierName": "Example Carrier Ltd",
        "numberedSource": "L00001 | HEADER\nL00002 | ABC123\nL00003 | FOOTER",
        "sourceLabel": {"documentPatch": {"billOfLadingNumber": "ABC123"}},
        "allowedTargetPaths": ("documentPatch.billOfLadingNumber",),
        "requiredRevision": f"target path index does not exist in source label: {invalid_path}",
        "anchorBindings": (),
        "occurrenceCandidates": _compiler_occurrence_candidates(),
        "requiredTargetCoBindings": (),
        "previousCandidateBindingInventory": (),
    }

    local = build_local_compiler_repair_payload(payload, prior)

    assert local["invalidPriorTargetPaths"] == (invalid_path,)
    assert local["targetFacts"] == ()
    assert local["candidateSlice"]["removableBindingKeys"] == ["hs_code_5_1"]
    assert local["candidateSlice"]["bindings"][0]["target_paths"] == [invalid_path]
    assert "L00002 | ABC123" in local["sourceWindow"]


def test_local_compiler_repair_exposes_invalid_semantic_only_path_for_removal() -> None:
    invalid_path = "documentPatch.cargoGroups[1].grossWeight.value"
    second_invalid_path = "documentPatch.cargoGroups[1].volume.value"
    prior = _compiler_candidate().model_copy(
        update={
            "bindings": (),
            "semantic_only_target_facts": (
                SemanticOnlyTargetFactProposal(
                    target_path=invalid_path,
                    rationale="The compiler incorrectly declared an absent indexed fact.",
                ),
                SemanticOnlyTargetFactProposal(
                    target_path=second_invalid_path,
                    rationale="A second absent fact must be repaired in the same transaction.",
                ),
            ),
        }
    )
    payload = {
        "documentId": "doc_invalid_semantic_only",
        "expectedCarrierName": "Example Carrier Ltd",
        "numberedSource": "L00001 | HEADER\nL00002 | FOOTER",
        "sourceLabel": {"documentPatch": {"cargoGroups": []}},
        "allowedTargetPaths": ("documentPatch.cargoGroups",),
        "requiredRevision": f"target path index does not exist in source label: {invalid_path}",
        "anchorBindings": (),
        "occurrenceCandidates": _compiler_occurrence_candidates(),
        "requiredTargetCoBindings": (),
        "previousCandidateBindingInventory": (),
    }

    local = build_local_compiler_repair_payload(payload, prior)

    assert local["invalidPriorTargetPaths"] == (invalid_path, second_invalid_path)
    assert local["targetFacts"] == ()
    assert local["sourceWindow"] == ""
    assert local["candidateSlice"]["semanticOnlyTargetFacts"] == [
        {
            "target_path": invalid_path,
            "rationale": "The compiler incorrectly declared an absent indexed fact.",
        },
        {
            "target_path": second_invalid_path,
            "rationale": "A second absent fact must be repaired in the same transaction.",
        },
    ]
    assert local["candidateSlice"]["removableSemanticOnlyTargetPaths"] == [
        invalid_path,
        second_invalid_path,
    ]


def test_local_compiler_repair_selects_existing_override_without_prefix_path_bloat() -> None:
    prior = _compiler_candidate().model_copy(
        update={
            "anchor_overrides": (
                AnchorOverride(
                    anchor_binding_id="anchor_binding_0023",
                    rationale="The equal-value anchor needs occurrence-specific ownership.",
                ),
            )
        }
    )
    leaf_path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    payload = {
        "documentId": "doc_repair",
        "expectedCarrierName": "Example Carrier Ltd",
        "numberedSource": "L00001 | HEADER\nL00002 | MATERIAL 13672294\nL00003 | FOOTER",
        "sourceLabel": {
            "documentPatch": {"cargoGroups": [{"additionalInformation": ["MATERIAL 13672294"]}]}
        },
        "allowedTargetPaths": ("documentPatch.cargoGroups", leaf_path),
        "requiredRevision": (
            f"anchor override lacks replacement target ownership: {leaf_path}; "
            "evidence L00002='MATERIAL 13672294'"
        ),
        "anchorBindings": (
            {
                "logicalKey": f"anchor:{leaf_path}",
                "renderMode": "target_binding",
                "valueKind": "cargo_text",
                "groupKind": "cargo",
                "groupKey": "cargo:0",
                "targetPaths": (leaf_path,),
                "targetRelationship": "single_target",
                "independentTargetFactComponents": ((leaf_path,),),
                "renderPolicy": "natural_text",
                "occurrences": (
                    {
                        "anchorBindingId": "anchor_binding_0023",
                        "lineStart": "L00002",
                        "lineEnd": "L00002",
                        "sourceText": "MATERIAL 13672294",
                    },
                ),
            },
        ),
        "occurrenceCandidates": _compiler_occurrence_candidates(),
        "previousCandidateBindingInventory": (),
        "previousCandidateOutput": prior.model_dump(mode="json"),
    }

    local = build_local_compiler_repair_payload(payload, prior)

    assert [row["targetPath"] for row in local["targetFacts"]] == [leaf_path]
    assert local["candidateSlice"]["removableAnchorOverrideIds"] == ["anchor_binding_0023"]


def test_local_compiler_repair_closes_slice_over_required_target_cobinding() -> None:
    allocation_path = "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity"
    package_path = "documentPatch.cargoPackages[0].quantity"
    key = f"anchor:{allocation_path}"
    binding = AgentBindingProposal.model_validate(
        {
            "logical_key": key,
            "render_mode": "target_binding",
            "value_kind": "integer",
            "group_kind": "package",
            "group_key": "package:0",
            "target_paths": (package_path,),
            "occurrences": (
                {
                    "line_start": "L00002",
                    "line_end": "L00002",
                    "source_text": "96",
                    "occurrence_index": 0,
                },
            ),
            "rationale": "Printed package quantity.",
        }
    )
    prior = _compiler_candidate().model_copy(update={"bindings": (binding,)})
    unrelated_allocation_path = (
        "documentPatch.cargoAllocationGroups[1].allocations[0].packageQuantity"
    )
    unrelated_package_path = "documentPatch.cargoPackages[1].quantity"
    unrelated_binding = binding.model_copy(
        update={
            "logical_key": f"anchor:{unrelated_allocation_path}",
            "target_paths": (unrelated_package_path,),
        }
    )
    prior = prior.model_copy(update={"bindings": (binding, unrelated_binding)})
    payload = {
        "documentId": "doc_repair",
        "expectedCarrierName": "Example Carrier Ltd",
        "numberedSource": "L00001 | QUANTITY\nL00002 | 96\nL00003 | FOOTER",
        "sourceLabel": {
            "documentPatch": {
                "cargoAllocationGroups": [{"allocations": [{"packageQuantity": 96}]}],
                "cargoPackages": [{"quantity": 96}],
            }
        },
        "allowedTargetPaths": (
            allocation_path,
            package_path,
            unrelated_allocation_path,
            unrelated_package_path,
        ),
        "requiredTargetCoBindings": (
            {
                "relationship": "one_to_one_package_quantity",
                "targetPaths": (allocation_path, package_path),
            },
            {
                "relationship": "one_to_one_package_quantity",
                "targetPaths": (unrelated_allocation_path, unrelated_package_path),
            },
        ),
        "requiredRevision": (
            "required target co-binding violations: one_to_one_package_quantity missing "
            + allocation_path
        ),
        "anchorBindings": (),
        "occurrenceCandidates": _compiler_occurrence_candidates(),
        "previousCandidateBindingInventory": (),
        "previousCandidateOutput": prior.model_dump(mode="json"),
    }

    local = build_local_compiler_repair_payload(payload, prior)

    assert local["candidateSlice"]["removableBindingKeys"] == [key]
    assert local["requiredTargetCoBindings"] == [payload["requiredTargetCoBindings"][0]]
    assert {row["targetPath"] for row in local["targetFacts"]} == {
        allocation_path,
        package_path,
    }


def test_local_compiler_repair_does_not_expand_through_equal_value_anchor() -> None:
    allocation_path = "documentPatch.cargoAllocationGroups[8].allocations[0].packageQuantity"
    package_path = "documentPatch.cargoPackages[8].quantity"
    unrelated_package_path = "documentPatch.cargoPackages[0].quantity"
    prior = _compiler_candidate().model_copy(
        update={
            "bindings": (
                _compiler_candidate()
                .bindings[0]
                .model_copy(
                    update={
                        "logical_key": f"anchor:{allocation_path}",
                        "target_paths": (package_path,),
                    }
                ),
            )
        }
    )
    payload = {
        "documentId": "doc_repair",
        "expectedCarrierName": "Example Carrier Ltd",
        "numberedSource": "L00001 | 96\nL00002 | 96",
        "sourceLabel": {
            "documentPatch": {
                "cargoAllocationGroups": [
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    {},
                    {"allocations": [{"packageQuantity": 96}]},
                ],
                "cargoPackages": [{"quantity": 96}] * 9,
            }
        },
        "allowedTargetPaths": (
            allocation_path,
            package_path,
            unrelated_package_path,
        ),
        "requiredTargetCoBindings": (
            {
                "relationship": "one_to_one_package_quantity",
                "targetPaths": (allocation_path, package_path),
            },
        ),
        "requiredRevision": f"required target co-binding missing {allocation_path}",
        "anchorBindings": (
            {
                "logicalKey": f"anchor:{package_path}|{unrelated_package_path}",
                "targetPaths": (package_path, unrelated_package_path),
                "occurrences": (
                    {
                        "anchorBindingId": "anchor_binding_0001",
                        "lineStart": "L00001",
                        "lineEnd": "L00001",
                        "sourceText": "96",
                    },
                ),
            },
            {
                "logicalKey": f"anchor:{allocation_path}",
                "targetPaths": (allocation_path,),
                "occurrences": (
                    {
                        "anchorBindingId": "anchor_binding_0002",
                        "lineStart": "L00002",
                        "lineEnd": "L00002",
                        "sourceText": "96",
                    },
                ),
            },
        ),
        "occurrenceCandidates": _compiler_occurrence_candidates(),
        "previousCandidateBindingInventory": (),
        "previousCandidateOutput": prior.model_dump(mode="json"),
    }

    local = build_local_compiler_repair_payload(payload, prior)

    assert [row["logicalKey"] for row in local["anchorBindings"]] == [f"anchor:{allocation_path}"]
    assert {row["targetPath"] for row in local["targetFacts"]} == {
        allocation_path,
        package_path,
    }


def test_local_compiler_repair_selected_anchor_does_not_authorize_unrelated_paths() -> None:
    package_path = "documentPatch.cargoPackages[1].quantity"
    unrelated_package_path = "documentPatch.cargoPackages[0].quantity"
    prior = _compiler_candidate().model_copy(update={"bindings": ()})
    payload = {
        "documentId": "doc_repair",
        "expectedCarrierName": "Example Carrier Ltd",
        "numberedSource": "L00001 | 96\nL00002 | 96",
        "sourceLabel": {
            "documentPatch": {
                "cargoPackages": [
                    {"quantity": 96, "typeCategory": "PACKAGE_CARTON"},
                    {"quantity": 96, "typeCategory": "PACKAGE_CARTON"},
                ]
            }
        },
        "allowedTargetPaths": (unrelated_package_path, package_path),
        "requiredTargetCoBindings": (),
        "requiredRevision": (
            f"binding token boundaries: {package_path} uses a substring occurrence at L00001; "
            f"evidence hints: {package_path} unowned exact candidates: none"
        ),
        "anchorBindings": (
            {
                "logicalKey": f"anchor:{unrelated_package_path}|{package_path}",
                "targetPaths": (unrelated_package_path, package_path),
                "occurrences": (
                    {
                        "anchorBindingId": "anchor_binding_0001",
                        "lineStart": "L00001",
                        "lineEnd": "L00001",
                        "sourceText": "96",
                    },
                ),
            },
        ),
        "occurrenceCandidates": _compiler_occurrence_candidates(),
        "previousCandidateBindingInventory": (),
        "previousCandidateOutput": prior.model_dump(mode="json"),
    }

    local = build_local_compiler_repair_payload(payload, prior)

    assert [row["logicalKey"] for row in local["anchorBindings"]] == [
        f"anchor:{unrelated_package_path}|{package_path}"
    ]
    assert [row["targetPath"] for row in local["targetFacts"]] == [package_path]
    assert [row["contextPath"] for row in local["targetContexts"]] == [
        "documentPatch.cargoPackages[1]"
    ]
    assert "unowned exact candidates: none" in local["requiredRevision"]


def test_plan_host_completes_only_proven_single_owner_exact_repeat() -> None:
    owner = "anchor:documentPatch.cargoGroups[8].additionalInformation[3]"
    occurrence_id = "compiler_occurrence_00206"
    audit = StagedAuditOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (
                {
                    "finding_kind": "unowned_repeated_fact",
                    "line_ids": ("L00249", "L00250"),
                    "evidence": "The exact material value repeats on the next line.",
                    "explanation": "The second value belongs to the one cited owner.",
                    "binding_indexes": (123,),
                    "target_path_indexes": (119,),
                    "candidate_indexes": (30,),
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
            "rationale": "The exhaustive audit found one unowned exact repeat.",
        }
    )
    plan = StagedCriticPlanOutput.model_validate(
        {
            "remove_binding_logical_keys": ("unrelated_binding",),
            "rationale": "A separate finding requires an unrelated removal.",
        }
    )
    payload = {
        "bindings": ({"bindingIndex": 123, "logicalKey": owner},),
        "candidateRows": (
            {
                "candidateIndex": 30,
                "kind": "unowned_exact_repeat",
                "requiredRevision": True,
                "bindingIndexes": (123,),
                "lineIds": ("L00250",),
                "sourceTexts": ("13672297",),
            },
        ),
        "occurrenceCandidates": (
            {
                "occurrenceId": occurrence_id,
                "lineStart": "L00250",
                "lineEnd": "L00250",
                "sourceText": "13672297",
                "occurrenceIndex": 0,
                "currentExactOwnerLogicalKeys": (),
            },
        ),
    }

    augmented = _augment_proven_repeat_appends(
        plan=plan,
        audit=audit,
        plan_payload=payload,
    )

    assert len(augmented.occurrence_appends) == 1
    assert augmented.occurrence_appends[0].logical_key == owner
    assert augmented.occurrence_appends[0].occurrences[0].occurrence_id == occurrence_id


def test_staged_audit_preserves_host_required_coherence_candidate() -> None:
    source = _source_payload()
    source["remainingRiskCandidates"] = []
    source["reviewCandidates"] = [
        {
            "candidateId": "review_candidate_0014",
            "kind": "cross_field_semantic_relation",
            "requiredRevision": True,
            "logicalKeys": ("anchor:documentPatch.billOfLadingNumber",),
            "lineIds": ("L00002",),
            "sourceTexts": ("ABC123",),
            "details": {
                "suggestedContracts": (
                    {
                        "kind": "numeric_values",
                        "memberLogicalKeys": (
                            "anchor:documentPatch.billOfLadingNumber",
                        ),
                        "dependencyPaths": ("documentPatch.billOfLadingNumber",),
                    },
                ),
                "ambiguousAlternatives": False,
            },
        }
    ]

    payload = build_staged_audit_payload(compact_critic_payload(source))

    assert len(payload["candidateRows"]) == 1
    assert payload["candidateRows"][0]["requiredRevision"] is True


def test_plan_host_completes_audited_unambiguous_coherence_decision() -> None:
    candidate_id = "review_candidate_0014"
    audit = StagedAuditOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (
                {
                    "finding_kind": "missing_coherence_dependency",
                    "line_ids": ("L00112",),
                    "evidence": "15440.000",
                    "explanation": "The printed total lacks its structured dependency.",
                    "binding_indexes": (0,),
                    "candidate_indexes": (0,),
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
            "rationale": "The relation is semantically required.",
        }
    )
    plan = StagedCriticPlanOutput.model_validate(
        {
            "remove_binding_logical_keys": ("unrelated_binding",),
            "rationale": "A separate finding requires a removal.",
        }
    )
    payload = {
        "candidateRows": (
            {
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
            },
        )
    }

    augmented = _augment_unambiguous_coherence_decisions(
        plan=plan,
        audit=audit,
        plan_payload=payload,
    )

    assert len(augmented.coherence_decisions) == 1
    assert augmented.coherence_decisions[0].candidate_id == candidate_id
    assert augmented.coherence_decisions[0].disposition == "apply_suggestion"
    assert augmented.coherence_decisions[0].suggestion_index == 0
