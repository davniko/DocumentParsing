"""Typed publication states for statistically proposed synthesis scenarios.

The structured baseline is deliberately separated from final synthetic training
records.  Statistical and deterministic work may be inspected while registry
or linguistic realization is outstanding, but no partial target can cross the
training publication boundary.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.generation_models import PendingRealization, SemanticChange

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class StatisticalProposalReceipt(_StrictFrozenModel):
    """Immutable lineage for one accepted statistical proposal."""

    proposal_id: NonEmptyString
    view_name: NonEmptyString
    method: Literal[
        "gaussian_copula_default",
        "gaussian_copula_gaussian_kde",
        "gaussian_copula_domain_mixed",
        "gaussian_copula_physical_factors",
        "ctgan",
        "tvae",
    ]
    model_receipt_sha256: Sha256
    fit_document_ids_sha256: Sha256
    fit_rows_sha256: Sha256
    sample_seed: int
    sampled_row_index: Annotated[int, Field(ge=0)]
    source_numeric_equivalence_id: Sha256
    contextual_support_tier: Literal["exact_identity_role", "semantic_family_role"]
    contextual_support_rows: Annotated[int, Field(ge=2)]
    contextual_support_templates: Annotated[int, Field(ge=2)]
    contextual_support_sha256: Sha256
    contextual_distance: Annotated[float, Field(ge=0)]
    contextual_maximum_distance: Annotated[float, Field(ge=0)]
    raw_proposals_attempted: Annotated[int, Field(gt=0)]
    rejected_before_acceptance: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def attempts_balance(self) -> StatisticalProposalReceipt:
        if self.rejected_before_acceptance >= self.raw_proposals_attempted:
            raise ValueError("an accepted proposal requires at least one non-rejected attempt")
        tolerance = 1e-12 * max(1.0, self.contextual_maximum_distance)
        if self.contextual_distance > self.contextual_maximum_distance + tolerance:
            raise ValueError("accepted proposal lies outside contextual support")
        return self


class SourceScopeReceipt(_StrictFrozenModel):
    """Exact source boundary used by deterministic donors and statistical fits."""

    split: NonEmptyString
    allowed_document_count: Annotated[int, Field(gt=0)]
    allowed_document_ids_sha256: Sha256
    allowed_template_ids_sha256: Sha256
    excluded_document_ids_sha256: Sha256
    partition_report_sha256: Sha256


class CargoGroupNumericProposal(_StrictFrozenModel):
    """One coherent cargo-group proposal with package-level derived quantities."""

    group_id: Annotated[str, StringConstraints(pattern=r"^g[1-9][0-9]*$")]
    driver_package_id: Annotated[str, StringConstraints(pattern=r"^p[1-9][0-9]*$")]
    quantity_by_package_id: dict[
        Annotated[str, StringConstraints(pattern=r"^p[1-9][0-9]*$")],
        Annotated[int, Field(gt=0)],
    ] = Field(min_length=1)
    gross_weight_value: int | float | None
    net_weight_value: int | float | None
    volume_value: int | float | None

    @model_validator(mode="after")
    def driver_is_covered(self) -> CargoGroupNumericProposal:
        if self.driver_package_id not in self.quantity_by_package_id:
            raise ValueError("cargo-group driver package is absent from quantity proposals")
        return self


class StructuredBaselinePlan(_StrictFrozenModel):
    """Auditable structured output that remains outside the training boundary."""

    schema_version: Literal[1]
    status: Literal["structured_baseline_pending_realization"]
    synthetic_document_id: NonEmptyString
    base_document_id: NonEmptyString
    template_id: NonEmptyString
    source_scope: SourceScopeReceipt
    variant_index: Annotated[int, Field(ge=0)]
    seed: int
    source_target_sha256: Sha256
    proposed_target_sha256: Sha256
    changes: tuple[SemanticChange, ...]
    proposal_receipts: tuple[StatisticalProposalReceipt, ...]
    pending_realizations: tuple[PendingRealization, ...]
    training_eligible: Literal[False]

    @model_validator(mode="after")
    def paths_and_receipts_are_unique(self) -> StructuredBaselinePlan:
        changed_paths = tuple(row.target_path for row in self.changes)
        if len(changed_paths) != len(set(changed_paths)):
            raise ValueError("structured baseline contains duplicate semantic-change paths")
        pending_paths = tuple(row.target_path for row in self.pending_realizations)
        if len(pending_paths) != len(set(pending_paths)):
            raise ValueError("structured baseline contains duplicate pending-realization paths")
        proposal_ids = tuple(row.proposal_id for row in self.proposal_receipts)
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("structured baseline contains duplicate proposal receipts")
        if set(changed_paths) & set(pending_paths):
            raise ValueError("a changed path cannot remain pending")
        return self


class ResolvedNonLinguisticScenario(_StrictFrozenModel):
    """Gate between structured generation and later linguistic realization."""

    schema_version: Literal[1]
    status: Literal["resolved_non_linguistic_pending_linguistic"]
    synthetic_document_id: NonEmptyString
    base_document_id: NonEmptyString
    template_id: NonEmptyString
    proposed_target: dict[str, Any]
    proposed_target_sha256: Sha256
    changes: tuple[SemanticChange, ...]
    proposal_receipts: tuple[StatisticalProposalReceipt, ...]
    pending_linguistic: tuple[PendingRealization, ...]
    training_eligible: Literal[False]

    @model_validator(mode="after")
    def only_linguistic_work_remains(self) -> ResolvedNonLinguisticScenario:
        if sha256_bytes(canonical_json_bytes(self.proposed_target)) != self.proposed_target_sha256:
            raise ValueError("proposed target SHA-256 does not match its payload")
        if any(row.kind != "linguistic" for row in self.pending_linguistic):
            raise ValueError("resolved non-linguistic scenario contains non-linguistic work")
        if len(self.pending_linguistic) != len(
            {row.target_path for row in self.pending_linguistic}
        ):
            raise ValueError("resolved non-linguistic scenario contains duplicate pending paths")
        return self


class ResolvedSemanticPlan(_StrictFrozenModel):
    """Complete semantic target; text realization is still required for training."""

    schema_version: Literal[1]
    status: Literal["resolved_semantic_pending_text"]
    synthetic_document_id: NonEmptyString
    base_document_id: NonEmptyString
    template_id: NonEmptyString
    target: dict[str, Any]
    target_sha256: Sha256
    changes: tuple[SemanticChange, ...]
    proposal_receipts: tuple[StatisticalProposalReceipt, ...]
    training_eligible: Literal[False]

    @model_validator(mode="after")
    def target_hash_and_change_paths_are_valid(self) -> ResolvedSemanticPlan:
        if sha256_bytes(canonical_json_bytes(self.target)) != self.target_sha256:
            raise ValueError("resolved semantic target SHA-256 does not match its payload")
        paths = tuple(row.target_path for row in self.changes)
        if len(paths) != len(set(paths)):
            raise ValueError("resolved semantic plan contains duplicate change paths")
        return self


class PublishedSyntheticRecord(_StrictFrozenModel):
    """Final text/target pair; construction is intentionally deferred."""

    schema_version: Literal[1]
    status: Literal["published_synthetic_training_record"]
    synthetic_document_id: NonEmptyString
    base_document_id: NonEmptyString
    joined_raw_text: NonEmptyString
    target: dict[str, Any]
    joined_raw_text_sha256: Sha256
    target_sha256: Sha256
    manifest_sha256: Sha256
    training_eligible: Literal[True]

    @model_validator(mode="after")
    def payload_hashes_are_valid(self) -> PublishedSyntheticRecord:
        if sha256_bytes(self.joined_raw_text.encode("utf-8")) != self.joined_raw_text_sha256:
            raise ValueError("published raw-text SHA-256 does not match its payload")
        if sha256_bytes(canonical_json_bytes(self.target)) != self.target_sha256:
            raise ValueError("published target SHA-256 does not match its payload")
        return self
