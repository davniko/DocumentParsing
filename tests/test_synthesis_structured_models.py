from __future__ import annotations

import pytest
from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.generation_models import PendingRealization, SemanticChange
from document_ocr.synthesis.structured_models import (
    ResolvedNonLinguisticScenario,
    SourceScopeReceipt,
    StatisticalProposalReceipt,
    StructuredBaselinePlan,
)


def _scope() -> SourceScopeReceipt:
    return SourceScopeReceipt.model_validate(
        {
            "split": "train",
            "allowed_document_count": 10,
            "allowed_document_ids_sha256": "1" * 64,
            "allowed_template_ids_sha256": "2" * 64,
            "excluded_document_ids_sha256": "3" * 64,
            "partition_report_sha256": "4" * 64,
        },
        strict=True,
    )


def _proposal() -> StatisticalProposalReceipt:
    return StatisticalProposalReceipt.model_validate(
        {
            "proposal_id": "proposal-1",
            "view_name": "cargo_package",
            "method": "gaussian_copula_default",
            "model_receipt_sha256": "5" * 64,
            "fit_document_ids_sha256": "1" * 64,
            "fit_rows_sha256": "6" * 64,
            "sample_seed": 17,
            "sampled_row_index": 0,
            "source_numeric_equivalence_id": "9" * 64,
            "contextual_support_tier": "exact_identity_role",
            "contextual_support_rows": 5,
            "contextual_support_templates": 4,
            "contextual_support_sha256": "a" * 64,
            "contextual_distance": 0.25,
            "contextual_maximum_distance": 0.5,
            "raw_proposals_attempted": 2,
            "rejected_before_acceptance": 1,
        },
        strict=True,
    )


def _change(path: str = "documentPatch.cargoPackages[0].quantity") -> SemanticChange:
    return SemanticChange.model_validate(
        {
            "target_path": path,
            "role_path": "documentPatch.cargoPackages[].quantity",
            "family": "package_quantity",
            "old_value": 10,
            "new_value": 11,
            "method": "gaussian_copula_joint_cargo_v1",
            "coupling_group": "cargo_graph",
        },
        strict=True,
    )


def _pending(path: str, kind: str) -> PendingRealization:
    return PendingRealization.model_validate(
        {
            "target_path": path,
            "role_path": path,
            "kind": kind,
            "reason": "not part of the structured baseline",
        },
        strict=True,
    )


def test_structured_baseline_is_explicitly_non_publishable() -> None:
    plan = StructuredBaselinePlan.model_validate(
        {
            "schema_version": 1,
            "status": "structured_baseline_pending_realization",
            "synthetic_document_id": "syn_1",
            "base_document_id": "doc_1",
            "template_id": "template_1",
            "source_scope": _scope(),
            "variant_index": 0,
            "seed": 17,
            "source_target_sha256": "7" * 64,
            "proposed_target_sha256": "8" * 64,
            "changes": (_change(),),
            "proposal_receipts": (_proposal(),),
            "pending_realizations": (_pending("documentPatch.parties.shipper.name", "registry"),),
            "training_eligible": False,
        },
        strict=True,
    )

    assert plan.training_eligible is False
    assert plan.proposal_receipts[0].rejected_before_acceptance == 1


def test_structured_baseline_rejects_changed_and_pending_path_overlap() -> None:
    path = "documentPatch.cargoPackages[0].quantity"
    with pytest.raises(ValidationError, match="changed path cannot remain pending"):
        StructuredBaselinePlan.model_validate(
            {
                "schema_version": 1,
                "status": "structured_baseline_pending_realization",
                "synthetic_document_id": "syn_1",
                "base_document_id": "doc_1",
                "template_id": "template_1",
                "source_scope": _scope(),
                "variant_index": 0,
                "seed": 17,
                "source_target_sha256": "7" * 64,
                "proposed_target_sha256": "8" * 64,
                "changes": (_change(path),),
                "proposal_receipts": (_proposal(),),
                "pending_realizations": (_pending(path, "deterministic_unsupported"),),
                "training_eligible": False,
            },
            strict=True,
        )


def test_resolved_non_linguistic_scenario_rejects_registry_work() -> None:
    target = {"schemaVersion": "3.0.0-experimental"}
    with pytest.raises(ValidationError, match="non-linguistic work"):
        ResolvedNonLinguisticScenario.model_validate(
            {
                "schema_version": 1,
                "status": "resolved_non_linguistic_pending_linguistic",
                "synthetic_document_id": "syn_1",
                "base_document_id": "doc_1",
                "template_id": "template_1",
                "proposed_target": target,
                "proposed_target_sha256": sha256_bytes(canonical_json_bytes(target)),
                "changes": (_change(),),
                "proposal_receipts": (_proposal(),),
                "pending_linguistic": (
                    _pending("documentPatch.route.portOfLoading.name", "registry"),
                ),
                "training_eligible": False,
            },
            strict=True,
        )


def test_proposal_receipt_rejects_all_attempts_failed() -> None:
    with pytest.raises(ValidationError, match="at least one non-rejected"):
        StatisticalProposalReceipt.model_validate(
            {
                **_proposal().model_dump(mode="python"),
                "raw_proposals_attempted": 2,
                "rejected_before_acceptance": 2,
            },
            strict=True,
        )


def test_proposal_receipt_accepts_domain_mixed_candidate() -> None:
    receipt = StatisticalProposalReceipt.model_validate(
        {
            **_proposal().model_dump(mode="python"),
            "method": "gaussian_copula_domain_mixed",
        },
        strict=True,
    )

    assert receipt.method == "gaussian_copula_domain_mixed"


def test_proposal_receipt_accepts_physical_factor_candidate() -> None:
    receipt = StatisticalProposalReceipt.model_validate(
        {
            **_proposal().model_dump(mode="python"),
            "method": "gaussian_copula_physical_factors",
        },
        strict=True,
    )

    assert receipt.method == "gaussian_copula_physical_factors"


def test_proposal_receipt_rejects_value_outside_contextual_support() -> None:
    with pytest.raises(ValidationError, match="outside contextual support"):
        StatisticalProposalReceipt.model_validate(
            {
                **_proposal().model_dump(mode="python"),
                "contextual_distance": 0.51,
                "contextual_maximum_distance": 0.5,
            },
            strict=True,
        )
