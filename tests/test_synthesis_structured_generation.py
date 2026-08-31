from __future__ import annotations

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.selection import SelectionCandidate
from document_ocr.synthesis.structured_generation import (
    StructuredBaselineError,
    _filter_rejected_profile_cohorts,
    _numeric_summary,
    _validate_cross_artifact_joins,
)


def _artifacts() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    target = {"schemaVersion": 3, "documentPatch": {}}
    target_sha = sha256_bytes(canonical_json_bytes(target))
    proposal_id = "proposal-1"
    plan = {
        "synthetic_document_id": "syn-1",
        "base_document_id": "doc-1",
        "template_id": "template-1",
        "training_eligible": False,
        "proposed_target_sha256": target_sha,
        "proposal_receipts": [{"proposal_id": proposal_id}],
        "changes": [{"coupling_group": "cargo_numeric:g1"}],
    }
    target_row = {
        "syntheticDocumentId": "syn-1",
        "baseDocumentId": "doc-1",
        "templateId": "template-1",
        "trainingEligible": False,
        "targetSha256": target_sha,
        "target": target,
    }
    proposal = {
        "proposal_id": proposal_id,
        "base_document_id": "doc-1",
        "group_id": "g1",
    }
    return [plan], [target_row], [proposal]


def test_cross_artifact_join_requires_exact_receipt_group_and_target_hashes() -> None:
    plans, targets, proposals = _artifacts()

    assert _validate_cross_artifact_joins(
        plans=plans,
        targets=targets,
        proposal_rows=proposals,
    ) == {
        "validated_documents": 1,
        "validated_group_proposals": 1,
        "failures": 0,
    }

    proposals[0]["proposal_id"] = "wrong"
    try:
        _validate_cross_artifact_joins(
            plans=plans,
            targets=targets,
            proposal_rows=proposals,
        )
    except StructuredBaselineError as error:
        assert "receipts" in str(error)
    else:
        raise AssertionError("mismatched proposal receipt was accepted")


def test_numeric_summary_is_complete_and_interpolates_percentiles() -> None:
    assert _numeric_summary(()) == {"count": 0}
    assert _numeric_summary((1, 2, 3, 4, 5)) == {
        "count": 5,
        "minimum": 1.0,
        "p10": 1.4,
        "median": 3.0,
        "mean": 3.0,
        "p90": 4.6,
        "maximum": 5.0,
    }


def _candidate(document_id: str, source_row_index: int) -> SelectionCandidate:
    return SelectionCandidate(
        document_id=document_id,
        source_row_index=source_row_index,
        template_id=f"template-{source_row_index}",
        template_size=2,
        carrier_family=f"carrier-{source_row_index}",
        strata={"kind": "bill_of_lading"},
        contexts=frozenset({"container_allocations"}),
        eligible_families=frozenset({"structured"}),
        risk_score=0,
    )


def test_rejected_statistical_cohort_removes_every_dependent_candidate() -> None:
    candidates = [_candidate("doc-1", 1), _candidate("doc-2", 2), _candidate("doc-3", 3)]
    rejected = ("g0n0v0", "exact_identity_role", ("row-1", "row-2"))
    accepted = ("g1n1v0", "semantic_family_role", ("row-3", "row-4"))

    retained, removed = _filter_rejected_profile_cohorts(
        candidates=candidates,
        cohort_keys_by_document={
            "doc-1": frozenset({rejected}),
            "doc-2": frozenset({rejected, accepted}),
            "doc-3": frozenset({accepted}),
        },
        rejected_cohort_keys=frozenset({rejected}),
    )

    assert [row.document_id for row in retained] == ["doc-3"]
    assert removed == ("doc-1", "doc-2")
