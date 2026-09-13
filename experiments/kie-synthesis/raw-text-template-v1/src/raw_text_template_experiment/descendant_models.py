from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .models import NonEmptyText, PinnedFile, PinnedJsonl, ProviderConfig, Sha256

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class PinnedCommittedRun(BaseModel):
    model_config = _STRICT

    path: NonEmptyText
    commit_sha256: Sha256
    transaction_sha256: Sha256


class DescendantInputs(BaseModel):
    model_config = _STRICT

    template_run: PinnedCommittedRun
    synthetic_target_run: PinnedCommittedRun
    synthetic_targets: Annotated[PinnedJsonl, Field()]
    iso3166_snapshot: PinnedFile
    residual_replay_run: PinnedCommittedRun | None = None


class DescendantPrompts(BaseModel):
    model_config = _STRICT

    residual_renderer: PinnedFile


class DescendantWorkflow(BaseModel):
    model_config = _STRICT

    documents: Literal[30]
    controlled_target_seed: int
    max_concurrent_requests: Annotated[int, Field(gt=0, le=16)]
    max_requests_per_document: Literal[1]
    provider_launch_authorized: bool
    require_exact_topology: Literal[True]
    require_source_carrier: Literal[True]
    require_every_slot_bound_once: Literal[True]
    require_exact_literal_regions: Literal[True]
    require_page_markers_unchanged: Literal[True]
    require_line_endings_preserved: Literal[True]
    publish_training_records: Literal[False]


class DescendantConfig(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    task: Literal["carrier_bound_template_descendant_rendering"]
    run_name: NonEmptyText
    output_dir: NonEmptyText
    environment_file: NonEmptyText
    inputs: DescendantInputs
    prompts: DescendantPrompts
    provider: ProviderConfig
    workflow: DescendantWorkflow


class TargetAdaptation(BaseModel):
    model_config = _STRICT

    target_path: NonEmptyText
    reason: NonEmptyText
    source_value: JsonValue
    proposed_value: JsonValue
    adapted_value: JsonValue


class PreparedTargetReceipt(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    document_id: NonEmptyText
    target_origin: Literal[
        "existing_linguistic_target_carrier_restored",
        "controlled_source_variant",
    ]
    source_schema_version: NonEmptyText
    target_schema_version: NonEmptyText
    fixed_carrier_name: NonEmptyText
    source_target_sha256: Sha256
    proposed_target_sha256: Sha256
    prepared_target_sha256: Sha256
    synthetic_document_id: NonEmptyText
    topology_mismatch_count: Literal[0]
    target_leaf_count: Annotated[int, Field(ge=0)]
    changed_target_leaf_count: Annotated[int, Field(ge=0)]
    carrier_leaf_count: Annotated[int, Field(ge=1)]
    carrier_changed_leaf_count: Literal[0]
    compatibility_adaptations: tuple[TargetAdaptation, ...]
    training_eligible: Literal[False]


class BindingRoute(BaseModel):
    model_config = _STRICT

    binding_id: NonEmptyText
    logical_key: NonEmptyText
    compiled_mode: NonEmptyText
    runtime_route: Literal["deterministic", "agent"]
    route_reason: NonEmptyText
    slot_ids: tuple[NonEmptyText, ...]


class ResidualStageReceipt(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    document_id: NonEmptyText
    status: Literal["not_required", "success", "provider_error"]
    started_at: str | None
    completed_at: str | None
    duration_seconds: Annotated[float, Field(ge=0)]
    system_prompt_sha256: Sha256
    user_prompt_sha256: Sha256 | None
    output_schema_sha256: Sha256 | None
    output: JsonValue | None
    error_type: str | None
    error_message: str | None
    messages: JsonValue
    usage: JsonValue


class ResidualReplayReceipt(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    document_id: NonEmptyText
    source_run_commit_sha256: Sha256
    source_run_transaction_sha256: Sha256
    source_agent_stage_sha256: Sha256
    source_status: Literal["success", "not_required"]
    source_output_slot_count: Annotated[int, Field(ge=0)]
    replayed_output_slot_count: Annotated[int, Field(ge=0)]
    dropped_output_slot_count: Annotated[int, Field(ge=0)]
    dropped_slot_ids: tuple[NonEmptyText, ...]
    new_provider_requests: Literal[0]

    @model_validator(mode="after")
    def slot_counts_are_consistent(self) -> ResidualReplayReceipt:
        if self.replayed_output_slot_count + self.dropped_output_slot_count != (
            self.source_output_slot_count
        ):
            raise ValueError("replayed and dropped slots do not cover the source output")
        if self.dropped_output_slot_count != len(self.dropped_slot_ids):
            raise ValueError("dropped slot count differs from dropped slot identifiers")
        if tuple(sorted(self.dropped_slot_ids)) != self.dropped_slot_ids:
            raise ValueError("dropped slot identifiers are not sorted")
        if self.source_status == "success" and self.source_output_slot_count == 0:
            raise ValueError("successful source stage has no output slots")
        if self.source_status == "not_required" and self.source_output_slot_count != 0:
            raise ValueError("not-required source stage has output slots")
        return self


class DescendantCaseResult(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    document_id: NonEmptyText
    synthetic_document_id: NonEmptyText
    status: Literal["passed", "provider_error", "host_rejected"]
    error_type: str | None
    error_message: str | None
    target_origin: NonEmptyText
    template_binding_count: Annotated[int, Field(gt=0)]
    template_slot_count: Annotated[int, Field(gt=0)]
    deterministic_binding_count: Annotated[int, Field(ge=0)]
    agent_binding_count: Annotated[int, Field(ge=0)]
    deterministic_slot_count: Annotated[int, Field(ge=0)]
    agent_slot_count: Annotated[int, Field(ge=0)]
    changed_target_leaf_count: Annotated[int, Field(ge=0)]
    changed_slot_count: Annotated[int, Field(ge=0)]
    unchanged_static_slot_count: Annotated[int, Field(ge=0)]
    carrier_unchanged: bool
    exact_topology: bool
    every_slot_bound_once: bool
    exact_literal_regions: bool
    page_markers_unchanged: bool
    line_endings_preserved: bool
    format_envelopes_valid: bool
    target_binding_semantics_valid: bool
    source_relationships_valid: bool
    output_sha256: Sha256 | None
    source_bytes: Annotated[int, Field(gt=0)]
    output_bytes: Annotated[int, Field(ge=0)]
    render_duration_seconds: Annotated[float, Field(ge=0)]
    provider_requests: Annotated[int, Field(ge=0, le=1)]
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    reasoning_tokens: Annotated[int, Field(ge=0)]
    estimated_cost_usd: Annotated[Decimal, Field(ge=0, decimal_places=12)]
    provider_reported_cost_usd: Annotated[Decimal, Field(ge=0, decimal_places=12)] | None

    @model_validator(mode="after")
    def counts_and_status_are_consistent(self) -> DescendantCaseResult:
        if (
            self.deterministic_binding_count + self.agent_binding_count
            != self.template_binding_count
        ):
            raise ValueError("runtime binding routes do not cover the template")
        if self.deterministic_slot_count + self.agent_slot_count != self.template_slot_count:
            raise ValueError("runtime slot routes do not cover the template")
        if self.status == "passed" and not all(
            (
                self.carrier_unchanged,
                self.exact_topology,
                self.every_slot_bound_once,
                self.exact_literal_regions,
                self.page_markers_unchanged,
                self.line_endings_preserved,
                self.format_envelopes_valid,
                self.target_binding_semantics_valid,
                self.source_relationships_valid,
            )
        ):
            raise ValueError("passed descendant case has a false acceptance invariant")
        if self.status == "provider_error" and self.provider_requests > 1:
            raise ValueError("provider failure exceeded the one-request contract")
        return self
