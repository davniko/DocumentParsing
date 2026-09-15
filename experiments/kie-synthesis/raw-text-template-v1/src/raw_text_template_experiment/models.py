from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from document_ocr.synthesis.linguistic_probe_runtime import LinguisticUsageReceipt
from document_ocr.synthesis.raw_text_template import (
    CompiledRawTextTemplate,
    SlotId,
    TemplateSlot,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ExactText = Annotated[str, StringConstraints(min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
LineId = Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
BindingId = Annotated[str, StringConstraints(pattern=r"^binding_[0-9]{4}$")]
InventoryBindingId = Annotated[
    str,
    StringConstraints(pattern=r"^inventory_binding_[0-9a-f]{16}$"),
]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

RenderMode = Literal[
    "target_binding",
    "deterministic_auxiliary",
    "deterministic_derived",
    "agent_residual",
    "carrier_static",
    "literal_static",
]
AgentContractProtocol = Literal[
    "legacy_v1",
    "compact_discriminated_v2",
    "reference_compact_v3",
    "staged_local_v4",
    "faceted_staged_local_v5",
    "partitioned_staged_local_v6",
    "candidate_first_staged_local_v7",
    "candidate_first_staged_local_v8",
    "relational_reference_compact_v9",
]
ValueKind = Literal[
    "organization",
    "person",
    "address",
    "contact_name",
    "email",
    "phone",
    "url_or_domain",
    "identifier",
    "date",
    "integer",
    "decimal_measurement",
    "location",
    "equipment",
    "package",
    "cargo_text",
    "dangerous_goods",
    "temperature",
    "commercial_text",
    "legal_text",
    "operational_text",
    "other_text",
]
GroupKind = Literal[
    "document",
    "carrier",
    "party",
    "route",
    "transport",
    "equipment",
    "cargo",
    "package",
    "dangerous_goods",
    "temperature",
    "customs",
    "commercial",
    "legal",
    "other",
]
TargetRelationship = Literal[
    "none",
    "single_target",
    "shared_value_equality",
    "composite_target_surface",
]
RealizationMode = Literal[
    "static",
    "single_surface",
    "repeated_surface",
    "segmented_surface",
    "token_projected_surface",
    "normalized_projected_surface",
    "generated_auxiliary",
    "deterministic_derivation",
    "agent_required",
]
SurfaceAdapter = Literal[
    "static",
    "natural_text",
    "opaque_identifier",
    "date",
    "numeric",
    "categorical",
    "package_category",
    "measurement_unit",
    "equipment_type",
    "generated_auxiliary",
    "deterministic_derivation",
    "agent",
]
SlotValueRole = Literal[
    "static",
    "whole",
    "repeat",
    "segment",
    "token_projection",
    "normalized_projection",
    "generated",
    "derived",
    "agent",
]
Derivation = Literal[
    "sum_package_quantity",
    "sum_gross_weight",
    "sum_net_weight",
    "sum_tare_weight",
    "sum_volume",
    "container_count",
    "package_count",
    "number_to_words",
    "equipment_receipt",
    "container_package_count",
    "country_code",
    "temperature_setpoint",
    "sum_monetary_amounts",
    "sum_decimal_values",
    "same_as_binding",
]


class PinnedFile(BaseModel):
    model_config = _STRICT

    path: NonEmptyText
    sha256: Sha256


class PinnedJsonl(PinnedFile):
    records: Annotated[int, Field(gt=0)]


class PinnedRun(BaseModel):
    model_config = _STRICT

    path: NonEmptyText
    commit_sha256: Sha256
    allow_prompt_change: bool = False


class PricingConfig(BaseModel):
    model_config = _STRICT

    currency: Literal["USD"]
    effective_date: NonEmptyText
    source_url: NonEmptyText
    input_usd_per_million: Annotated[Decimal, Field(ge=0)]
    cached_input_usd_per_million: Annotated[Decimal, Field(ge=0)]
    cache_write_multiplier: Annotated[Decimal, Field(ge=0)]
    output_usd_per_million: Annotated[Decimal, Field(ge=0)]


class OpenAIResponsesProviderConfig(BaseModel):
    model_config = _STRICT

    kind: Literal["openai_responses"]
    model: NonEmptyText
    api_key_env: Literal["OPENAI_API_KEY"]
    reasoning_effort: Literal["high", "max"]
    request_timeout_seconds: Annotated[float, Field(gt=0)]
    transport_max_retries: Annotated[int, Field(ge=0, le=3)]
    max_output_tokens: Annotated[int, Field(gt=0)]
    store_responses: bool
    pricing: PricingConfig


class OpenRouterProviderConfig(BaseModel):
    model_config = _STRICT

    kind: Literal["openrouter"]
    model: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._~-]*/[A-Za-z0-9][A-Za-z0-9._~:/-]*$",
        ),
    ]
    api_key_env: Literal["OPENROUTER_API_KEY"]
    reasoning_effort: Literal["high"]
    request_timeout_seconds: Annotated[float, Field(gt=0)]
    transport_max_retries: Annotated[int, Field(ge=0, le=3)]
    max_output_tokens: Annotated[int, Field(gt=0)]
    require_parameters: Literal[True]
    data_collection: Literal["deny"]
    allow_fallbacks: Literal[False]
    provider_only: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    native_structured_output_profile: Literal[
        "provider_default",
        "provider_verified",
    ] = "provider_default"
    native_structured_output_source_url: NonEmptyText | None = None
    max_prompt_price_usd_per_million: Annotated[Decimal, Field(gt=0)]
    max_completion_price_usd_per_million: Annotated[Decimal, Field(gt=0)]
    pricing: PricingConfig

    @model_validator(mode="after")
    def routing_and_prices_are_pinned(self) -> OpenRouterProviderConfig:
        if len(set(self.provider_only)) != len(self.provider_only):
            raise ValueError("OpenRouter provider_only entries must be unique")
        if self.pricing.input_usd_per_million > self.max_prompt_price_usd_per_million:
            raise ValueError("pinned input price exceeds the OpenRouter prompt-price ceiling")
        if self.pricing.output_usd_per_million > self.max_completion_price_usd_per_million:
            raise ValueError("pinned output price exceeds the OpenRouter completion-price ceiling")
        verified = self.native_structured_output_profile == "provider_verified"
        if verified != (self.native_structured_output_source_url is not None):
            raise ValueError(
                "provider-verified native structured output requires exactly one source URL"
            )
        return self


ProviderConfig = Annotated[
    OpenAIResponsesProviderConfig | OpenRouterProviderConfig,
    Field(discriminator="kind"),
]


class ExtractionInputs(BaseModel):
    model_config = _STRICT

    source_corpus: PinnedJsonl
    document_features: PinnedJsonl
    anchors: PinnedFile
    current_targets: PinnedJsonl
    outcome_findings: PinnedFile
    outcome_invariants: PinnedFile
    manual_audit: PinnedFile


class ExtractionPrompts(BaseModel):
    model_config = _STRICT

    compiler: PinnedFile
    critic: PinnedFile
    compiler_repair: PinnedFile | None = None
    critic_audit: PinnedFile | None = None
    critic_plan: PinnedFile | None = None


class ExtractionWorkflow(BaseModel):
    model_config = _STRICT

    documents: Annotated[int, Field(gt=0)]
    agent_contract_protocol: AgentContractProtocol
    max_concurrent_documents: Annotated[int, Field(gt=0, le=16)]
    max_concurrent_requests: Annotated[int, Field(gt=0, le=16)]
    max_compiler_passes: Annotated[int, Field(ge=1, le=16)]
    max_critic_passes: Annotated[int, Field(ge=1, le=64)]
    compiler_output_retries: Annotated[int, Field(ge=0, le=3)]
    critic_output_retries: Annotated[int, Field(ge=0, le=3)]
    additional_call_launch_threshold_usd_per_document: Annotated[Decimal, Field(gt=0)]
    provider_launch_authorized: bool
    certified_resume_policy: Literal[
        "require_exact_contract",
        "current_host_recertify",
    ]
    require_all_documents_certified: Literal[True]
    require_carrier_resolution: Literal[True]
    require_source_round_trip: Literal[True]
    require_all_risk_candidates_owned: Literal[True]
    require_final_critic_pass: Literal[True]
    require_sentinel_isolation: Literal[True]
    publish_training_records: Literal[False]


class ExtractionConfig(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    task: Literal["carrier_bound_raw_text_template_extraction"]
    phase: Literal[
        "canary1",
        "efficiency_targeted3",
        "efficiency_probe5",
        "development30",
        "transfer200",
    ]
    run_name: NonEmptyText
    output_dir: NonEmptyText
    environment_file: NonEmptyText
    inputs: ExtractionInputs
    prompts: ExtractionPrompts
    compiler_provider: ProviderConfig
    critic_provider: ProviderConfig
    workflow: ExtractionWorkflow
    selection_seed: int
    resume_from: PinnedRun | None = None
    excluded_document_ids: tuple[NonEmptyText, ...] = ()
    pinned_document_ids: tuple[NonEmptyText, ...] = ()

    @model_validator(mode="after")
    def phase_count_is_exact(self) -> ExtractionConfig:
        expected = {
            "canary1": 1,
            "efficiency_targeted3": 3,
            "efficiency_probe5": 5,
            "development30": 30,
            "transfer200": 200,
        }[self.phase]
        if self.workflow.documents != expected:
            raise ValueError(f"{self.phase} requires exactly {expected} documents")
        if len(set(self.excluded_document_ids)) != len(self.excluded_document_ids):
            raise ValueError("excluded document IDs must be unique")
        if len(set(self.pinned_document_ids)) != len(self.pinned_document_ids):
            raise ValueError("pinned document IDs must be unique")
        if self.pinned_document_ids and len(self.pinned_document_ids) != self.workflow.documents:
            raise ValueError("pinned document IDs must exactly match the configured document count")
        if set(self.pinned_document_ids) & set(self.excluded_document_ids):
            raise ValueError("a document cannot be both pinned and excluded")
        staged_prompts = (
            self.prompts.compiler_repair,
            self.prompts.critic_audit,
            self.prompts.critic_plan,
        )
        if self.workflow.agent_contract_protocol in {
            "staged_local_v4",
            "faceted_staged_local_v5",
            "partitioned_staged_local_v6",
            "candidate_first_staged_local_v7",
            "candidate_first_staged_local_v8",
        }:
            if any(prompt is None for prompt in staged_prompts):
                raise ValueError(
                    "staged protocols require compiler_repair, critic_audit, and critic_plan "
                    "prompts"
                )
        elif any(prompt is not None for prompt in staged_prompts):
            raise ValueError("staged prompts are valid only for staged protocols")
        if (
            self.workflow.agent_contract_protocol
            in {"candidate_first_staged_local_v7", "candidate_first_staged_local_v8"}
            and self.workflow.max_critic_passes < 2
        ):
            raise ValueError(
                "candidate-first protocol requires a candidate pre-pass and a distinct final "
                "critic pass"
            )
        return self


class SelectionRow(BaseModel):
    model_config = _STRICT

    ordinal: Annotated[int, Field(gt=0)]
    document_id: NonEmptyText
    selection_basis: tuple[NonEmptyText, ...]
    source_sha256: Sha256
    carrier_name: NonEmptyText | None
    carrier_family: NonEmptyText | None
    template_proxy_id: NonEmptyText
    document_type: Literal["bill_of_lading", "sea_waybill"]
    page_count: Annotated[int, Field(gt=0)]
    ocr_lines: Annotated[int, Field(gt=0)]
    ocr_characters: Annotated[int, Field(gt=0)]
    container_count: Annotated[int, Field(ge=0)]
    cargo_group_count: Annotated[int, Field(ge=0)]
    package_count: Annotated[int, Field(ge=0)]
    dangerous_goods_count: Annotated[int, Field(ge=0)]
    temperature_count: Annotated[int, Field(ge=0)]


class SelectionManifest(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    phase: Literal[
        "canary1",
        "efficiency_targeted3",
        "efficiency_probe5",
        "development30",
        "transfer200",
    ]
    selection_seed: int
    source_corpus_sha256: Sha256
    document_features_sha256: Sha256
    rows: tuple[SelectionRow, ...]

    @model_validator(mode="after")
    def ordinals_and_ids_are_unique(self) -> SelectionManifest:
        if tuple(row.ordinal for row in self.rows) != tuple(range(1, len(self.rows) + 1)):
            raise ValueError("selection ordinals must be contiguous from one")
        ids = tuple(row.document_id for row in self.rows)
        if len(set(ids)) != len(ids):
            raise ValueError("selection document IDs must be unique")
        return self


class AgentOccurrence(BaseModel):
    model_config = _STRICT

    line_start: LineId
    line_end: LineId
    source_text: ExactText
    occurrence_index: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def line_order_is_valid(self) -> AgentOccurrence:
        if int(self.line_end[1:]) < int(self.line_start[1:]):
            raise ValueError("occurrence line_end precedes line_start")
        return self


class AgentBindingProposal(BaseModel):
    model_config = _STRICT

    logical_key: NonEmptyText
    render_mode: RenderMode
    value_kind: ValueKind
    group_kind: GroupKind
    group_key: NonEmptyText
    target_paths: tuple[NonEmptyText, ...] = ()
    derivation: Derivation | None = None
    dependency_paths: tuple[NonEmptyText, ...] = ()
    dependency_bindings: tuple[NonEmptyText, ...] = ()
    occurrences: Annotated[tuple[AgentOccurrence, ...], Field(min_length=1)]
    rationale: NonEmptyText

    @model_validator(mode="after")
    def semantics_match_render_mode(self) -> AgentBindingProposal:
        if self.render_mode == "target_binding" and not self.target_paths:
            raise ValueError("target_binding requires target_paths")
        if self.render_mode == "deterministic_derived":
            if self.derivation is None or not (self.dependency_paths or self.dependency_bindings):
                raise ValueError("deterministic_derived requires derivation and dependencies")
        elif self.derivation is not None or self.dependency_paths or self.dependency_bindings:
            raise ValueError("only deterministic_derived may declare derivation inputs")
        if self.render_mode == "literal_static" and self.target_paths:
            raise ValueError("literal_static bindings cannot declare target paths")
        if self.render_mode == "deterministic_auxiliary" and self.target_paths:
            raise ValueError("deterministic_auxiliary bindings cannot declare target paths")
        return self


class AnchorOverride(BaseModel):
    model_config = _STRICT

    anchor_binding_id: Annotated[str, StringConstraints(pattern=r"^anchor_binding_[0-9]{4}$")]
    rationale: NonEmptyText


class SemanticOnlyTargetFactProposal(BaseModel):
    model_config = _STRICT

    target_path: NonEmptyText
    rationale: NonEmptyText


class SemanticOnlyTargetFact(BaseModel):
    model_config = _STRICT

    target_path: NonEmptyText
    source_value: JsonValue
    provenance: Literal[
        "host_document_semantic",
        "compiler_audited_unprinted",
        "critic_audited_unprinted",
    ]
    rationale: NonEmptyText


class CarrierAssessment(BaseModel):
    model_config = _STRICT

    canonical_name: NonEmptyText
    aliases: tuple[NonEmptyText, ...]
    evidence_occurrences: Annotated[tuple[AgentOccurrence, ...], Field(min_length=1)]
    source: Literal[
        "source_label_confirmed_by_ocr",
        "ocr_resolved_missing_source_label",
    ]
    rationale: NonEmptyText


class CompilerAgentOutput(BaseModel):
    model_config = _STRICT

    carrier: CarrierAssessment
    anchor_overrides: tuple[AnchorOverride, ...]
    bindings: tuple[AgentBindingProposal, ...]
    unresolved: tuple[NonEmptyText, ...]
    all_shipment_dependent_surfaces_accounted_for: bool
    semantic_only_target_facts: tuple[SemanticOnlyTargetFactProposal, ...] = ()

    @model_validator(mode="after")
    def declared_completion_is_consistent(self) -> CompilerAgentOutput:
        if self.all_shipment_dependent_surfaces_accounted_for and self.unresolved:
            raise ValueError("completed compiler output cannot retain unresolved items")
        return self


class CriticFinding(BaseModel):
    model_config = _STRICT

    finding_kind: Literal[
        "unowned_shipment_fact",
        "unowned_private_or_auxiliary_fact",
        "unowned_repeated_fact",
        "incorrect_static_classification",
        "incorrect_semantic_owner",
        "missing_derivation",
        "carrier_binding_error",
        "topology_or_grouping_error",
    ]
    line_ids: Annotated[tuple[LineId, ...], Field(min_length=1)]
    evidence: NonEmptyText
    explanation: NonEmptyText


class CriticOccurrenceRemoval(BaseModel):
    """Remove exact bad occurrences while retaining a binding's semantic contract."""

    model_config = _STRICT

    logical_key: NonEmptyText
    occurrences: Annotated[tuple[AgentOccurrence, ...], Field(min_length=1)]
    rationale: NonEmptyText


class CriticAgentOutput(BaseModel):
    model_config = _STRICT

    verdict: Literal["pass", "revise"]
    findings: tuple[CriticFinding, ...]
    remove_inventory_binding_ids: tuple[InventoryBindingId, ...] = ()
    occurrence_removals: tuple[CriticOccurrenceRemoval, ...] = ()
    additional_bindings: tuple[AgentBindingProposal, ...]
    semantic_only_target_facts: tuple[SemanticOnlyTargetFactProposal, ...] = ()
    rationale: NonEmptyText

    @model_validator(mode="after")
    def verdict_is_consistent(self) -> CriticAgentOutput:
        if self.verdict == "pass" and (
            self.findings
            or self.remove_inventory_binding_ids
            or self.occurrence_removals
            or self.additional_bindings
            or self.semantic_only_target_facts
        ):
            raise ValueError("pass requires zero findings and zero patch operations")
        if self.verdict == "revise" and not self.findings:
            raise ValueError("revise requires at least one finding")
        if len(set(self.remove_inventory_binding_ids)) != len(self.remove_inventory_binding_ids):
            raise ValueError("critic inventory removal IDs must be unique")
        removal_keys = tuple(row.logical_key for row in self.occurrence_removals)
        if len(set(removal_keys)) != len(removal_keys):
            raise ValueError("critic occurrence-removal logical keys must be unique")
        return self


class TargetValueSnapshot(BaseModel):
    model_config = _STRICT

    target_path: NonEmptyText
    source_value: JsonValue


class SlotRealization(BaseModel):
    model_config = _STRICT

    slot_id: SlotId
    value_role: SlotValueRole
    segment_index: Annotated[int, Field(ge=0)] | None
    source_token_count: Annotated[int, Field(ge=0)]
    literal_prefix: str
    literal_suffix: str
    required_target_prefix_tokens: tuple[NonEmptyText, ...]
    required_target_suffix_tokens: tuple[NonEmptyText, ...]
    required_target_prefix_normalized: str
    required_target_suffix_normalized: str

    @model_validator(mode="after")
    def segment_metadata_is_consistent(self) -> SlotRealization:
        if self.value_role == "segment":
            if self.segment_index is None or self.source_token_count == 0:
                raise ValueError("segment slots require an index and positive token count")
        elif self.segment_index is not None:
            raise ValueError("only segment slots may declare segment_index")
        has_projection_frame = bool(
            self.required_target_prefix_tokens or self.required_target_suffix_tokens
        )
        if self.value_role == "token_projection":
            if self.source_token_count == 0:
                raise ValueError("token-projection slots require source tokens")
        elif has_projection_frame:
            raise ValueError("only token-projection slots may omit target tokens")
        has_normalized_projection_frame = bool(
            self.required_target_prefix_normalized or self.required_target_suffix_normalized
        )
        if self.value_role == "normalized_projection":
            if self.source_token_count == 0:
                raise ValueError("normalized-projection slots require source tokens")
        elif has_normalized_projection_frame:
            raise ValueError(
                "only normalized-projection slots may omit normalized target fragments"
            )
        return self


class BindingRealization(BaseModel):
    model_config = _STRICT

    mode: RealizationMode
    adapter: SurfaceAdapter
    deterministic: bool
    requires_agent: bool
    target_values: tuple[TargetValueSnapshot, ...]
    slots: Annotated[tuple[SlotRealization, ...], Field(min_length=1)]
    rationale: NonEmptyText

    @model_validator(mode="after")
    def mode_flags_are_consistent(self) -> BindingRealization:
        if self.deterministic == self.requires_agent:
            raise ValueError("exactly one of deterministic and requires_agent must be true")
        if self.requires_agent != (self.mode == "agent_required"):
            raise ValueError("requires_agent must exactly identify agent_required mode")
        if (self.adapter == "agent") != self.requires_agent:
            raise ValueError("agent adapter must exactly identify agent-required realization")
        slot_ids = tuple(slot.slot_id for slot in self.slots)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("realization slot IDs must be unique")
        target_paths = tuple(row.target_path for row in self.target_values)
        if len(set(target_paths)) != len(target_paths):
            raise ValueError("realization target paths must be unique")
        expected_roles: dict[str, set[str]] = {
            "static": {"static"},
            "single_surface": {"whole"},
            "repeated_surface": {"repeat"},
            "segmented_surface": {"segment"},
            "token_projected_surface": {"token_projection"},
            "normalized_projected_surface": {"normalized_projection"},
            "generated_auxiliary": {"generated"},
            "deterministic_derivation": {"derived"},
            "agent_required": {"agent"},
        }
        if {slot.value_role for slot in self.slots} != expected_roles[self.mode]:
            raise ValueError("slot value roles do not match realization mode")
        if self.mode == "single_surface" and len(self.slots) != 1:
            raise ValueError("single-surface realization requires exactly one slot")
        if (
            self.mode
            in {
                "single_surface",
                "repeated_surface",
                "segmented_surface",
                "token_projected_surface",
                "normalized_projected_surface",
            }
            and not self.target_values
        ):
            raise ValueError("target-surface realization requires target values")
        if self.mode == "generated_auxiliary" and self.target_values:
            raise ValueError("generated auxiliary realization cannot declare target values")
        if self.mode == "segmented_surface":
            indexes = tuple(slot.segment_index for slot in self.slots)
            if indexes != tuple(range(len(self.slots))):
                raise ValueError("segmented slots must have contiguous ordered indexes")
        if self.mode == "token_projected_surface" and not any(
            slot.required_target_prefix_tokens or slot.required_target_suffix_tokens
            for slot in self.slots
        ):
            raise ValueError("token-projected realization must omit tokens in at least one slot")
        if self.mode == "normalized_projected_surface" and not any(
            slot.required_target_prefix_normalized or slot.required_target_suffix_normalized
            for slot in self.slots
        ):
            raise ValueError(
                "normalized-projected realization must omit target fragments in at least one slot"
            )
        return self


class SourceBindingRelationship(BaseModel):
    model_config = _STRICT

    dependency_binding: NonEmptyText
    relationship: Literal[
        "embeds_exact_source_identifier",
        "is_embedded_in_exact_source_identifier",
    ]
    source_prefix: str
    source_suffix: str
    source_prefix_pattern: str | None
    source_suffix_pattern: str | None

    @model_validator(mode="after")
    def fragments_and_patterns_are_consistent(self) -> SourceBindingRelationship:
        if not self.source_prefix and not self.source_suffix:
            raise ValueError("an embedded identifier relationship requires an outer fragment")
        if (self.source_prefix_pattern is None) == bool(self.source_prefix):
            raise ValueError("source prefix pattern must exactly accompany a non-empty prefix")
        if (self.source_suffix_pattern is None) == bool(self.source_suffix):
            raise ValueError("source suffix pattern must exactly accompany a non-empty suffix")
        return self


class SemanticBinding(BaseModel):
    model_config = _STRICT

    binding_id: BindingId
    logical_key: NonEmptyText
    render_mode: RenderMode
    value_kind: ValueKind
    group_kind: GroupKind
    group_key: NonEmptyText
    target_paths: tuple[NonEmptyText, ...]
    target_relationship: TargetRelationship
    derivation: Derivation | None
    dependency_paths: tuple[NonEmptyText, ...]
    dependency_bindings: tuple[NonEmptyText, ...]
    occurrences: Annotated[tuple[TemplateSlot, ...], Field(min_length=1)]
    realization: BindingRealization
    source_relationships: tuple[SourceBindingRelationship, ...]
    rationale: NonEmptyText


class CarrierBinding(BaseModel):
    model_config = _STRICT

    canonical_name: NonEmptyText
    family: NonEmptyText
    aliases: tuple[NonEmptyText, ...]
    evidence_occurrences: Annotated[tuple[AgentOccurrence, ...], Field(min_length=1)]
    source: Literal[
        "source_label_confirmed_by_ocr",
        "ocr_resolved_missing_source_label",
    ]


class CapabilityContract(BaseModel):
    model_config = _STRICT

    document_type: Literal["bill_of_lading", "sea_waybill"]
    template_proxy_id: NonEmptyText
    page_count: Annotated[int, Field(gt=0)]
    line_count: Annotated[int, Field(gt=0)]
    character_count: Annotated[int, Field(gt=0)]
    container_count: Annotated[int, Field(ge=0)]
    seal_count: Annotated[int, Field(ge=0)]
    cargo_group_count: Annotated[int, Field(ge=0)]
    package_count: Annotated[int, Field(ge=0)]
    allocation_group_count: Annotated[int, Field(ge=0)]
    allocation_row_count: Annotated[int, Field(ge=0)]
    dangerous_goods_count: Annotated[int, Field(ge=0)]
    temperature_count: Annotated[int, Field(ge=0)]
    additional_information_count: Annotated[int, Field(ge=0)]
    marks_count: Annotated[int, Field(ge=0)]
    party_roles: tuple[NonEmptyText, ...]
    target_leaf_paths: tuple[NonEmptyText, ...]


class RiskCandidate(BaseModel):
    model_config = _STRICT

    risk_id: Annotated[str, StringConstraints(pattern=r"^risk_[0-9]{4}$")]
    kind: Literal[
        "email",
        "url_or_domain",
        "phone",
        "date",
        "measurement",
        "long_numeric_identifier",
        "alphanumeric_identifier",
        "equipment_identifier",
        "selected_text",
    ]
    line_id: LineId
    byte_start: Annotated[int, Field(ge=0)]
    byte_end: Annotated[int, Field(gt=0)]
    source_text: ExactText


class LiteralCertification(BaseModel):
    model_config = _STRICT

    masked_literal_sha256: Sha256
    final_critic_pass: Literal[True]
    critic_passes: Annotated[int, Field(ge=1)]
    remaining_unowned_risk_candidates: Literal[0]


class TemplateCertification(BaseModel):
    model_config = _STRICT

    source_hash_valid: Literal[True]
    source_round_trip: Literal[True]
    disjoint_utf8_spans: Literal[True]
    exact_literal_regions: Literal[True]
    page_markers_unchanged: Literal[True]
    line_endings_preserved: Literal[True]
    sentinel_isolation: Literal[True]
    carrier_resolution_valid: Literal[True]
    carrier_matches_source_label: bool | None
    all_risk_candidates_owned: Literal[True]
    final_critic_pass: Literal[True]
    all_bindings_realization_planned: Literal[True]
    all_unprinted_target_facts_classified: Literal[True]


class CertifiedSemanticTemplate(BaseModel):
    model_config = _STRICT

    schema_version: Literal[4]
    compiler: Literal["carrier_bound_semantic_template_v4"]
    document_id: NonEmptyText
    source_sha256: Sha256
    source_size_bytes: Annotated[int, Field(gt=0)]
    carrier: CarrierBinding
    capability: CapabilityContract
    semantic_only_target_facts: tuple[SemanticOnlyTargetFact, ...]
    bindings: Annotated[tuple[SemanticBinding, ...], Field(min_length=1)]
    byte_template: CompiledRawTextTemplate
    literal_certification: LiteralCertification
    certification: TemplateCertification

    @model_validator(mode="after")
    def nested_contracts_are_consistent(self) -> CertifiedSemanticTemplate:
        if self.document_id != self.byte_template.document_id:
            raise ValueError("byte template document differs from semantic template")
        if self.source_sha256 != self.byte_template.source_sha256:
            raise ValueError("byte template source hash differs")
        if self.source_size_bytes != self.byte_template.source_size_bytes:
            raise ValueError("byte template source size differs")
        slots = tuple(slot for binding in self.bindings for slot in binding.occurrences)
        if tuple(sorted(slots, key=lambda slot: slot.byte_start)) != self.byte_template.slots:
            raise ValueError("semantic binding occurrences differ from byte template slots")
        binding_ids = tuple(binding.binding_id for binding in self.bindings)
        if len(set(binding_ids)) != len(binding_ids):
            raise ValueError("semantic binding IDs must be unique")
        slot_ids = tuple(slot.slot_id for slot in slots)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("template slot IDs must be unique")
        semantic_only_paths = tuple(fact.target_path for fact in self.semantic_only_target_facts)
        if len(set(semantic_only_paths)) != len(semantic_only_paths):
            raise ValueError("semantic-only target paths must be unique")
        owned_paths = {path for binding in self.bindings for path in binding.target_paths}
        overlap = sorted(set(semantic_only_paths) & owned_paths)
        if overlap:
            raise ValueError(
                "semantic-only target paths also have source bindings: " + ", ".join(overlap)
            )
        for binding in self.bindings:
            if tuple(slot.slot_id for slot in binding.occurrences) != tuple(
                slot.slot_id for slot in binding.realization.slots
            ):
                raise ValueError("binding realization slots differ from binding occurrences")
            if tuple(row.target_path for row in binding.realization.target_values) != (
                binding.target_paths
            ):
                raise ValueError("binding realization target values differ from target paths")
        return self


class AgentStageArtifact(BaseModel):
    model_config = _STRICT

    role: Literal["compiler", "critic"]
    pass_number: Annotated[int, Field(gt=0)]
    started_at: datetime
    completed_at: datetime
    duration_seconds: Annotated[float, Field(ge=0)]
    system_prompt_sha256: Sha256
    user_prompt_sha256: Sha256
    output_schema_sha256: Sha256
    status: Literal["success", "provider_error", "host_rejected"]
    output: JsonValue | None
    error_type: str | None
    error_message: str | None
    messages: JsonValue
    usage: LinguisticUsageReceipt


class DraftCheckpointRow(BaseModel):
    model_config = _STRICT

    draft_id: NonEmptyText
    logical_key: NonEmptyText
    render_mode: RenderMode
    value_kind: ValueKind
    group_kind: GroupKind
    group_key: NonEmptyText
    target_paths: tuple[NonEmptyText, ...]
    derivation: Derivation | None
    dependency_paths: tuple[NonEmptyText, ...]
    dependency_bindings: tuple[NonEmptyText, ...]
    char_start: Annotated[int, Field(ge=0)]
    char_end: Annotated[int, Field(gt=0)]
    source_text: ExactText
    evidence_origin: NonEmptyText
    render_policy: NonEmptyText
    rationale: NonEmptyText

    @model_validator(mode="after")
    def span_is_valid(self) -> DraftCheckpointRow:
        if self.char_end <= self.char_start:
            raise ValueError("checkpoint draft span must have positive width")
        return self


class ExtractionStateCheckpoint(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    document_id: NonEmptyText
    source_sha256: Sha256
    source_label_sha256: Sha256
    drafts: Annotated[tuple[DraftCheckpointRow, ...], Field(min_length=1)]
    carrier_assessment: CarrierAssessment
    semantic_only_target_facts: tuple[SemanticOnlyTargetFact, ...]
    applied_critic_revisions: tuple[CriticAgentOutput, ...]

    @model_validator(mode="after")
    def revisions_are_applied_transactions(self) -> ExtractionStateCheckpoint:
        if any(review.verdict != "revise" for review in self.applied_critic_revisions):
            raise ValueError("checkpoint may retain only applied critic revisions")
        return self


class ExtractionCaseResult(BaseModel):
    model_config = _STRICT

    document_id: NonEmptyText
    status: Literal["certified", "rejected"]
    rejection_reasons: tuple[NonEmptyText, ...]
    template_sha256: Sha256 | None
    compiler_stages: tuple[AgentStageArtifact, ...]
    critic_stages: tuple[AgentStageArtifact, ...]
    risk_candidates: tuple[RiskCandidate, ...]
    elapsed_seconds: Annotated[float, Field(ge=0)]
    resumed_from_run: NonEmptyText | None = None
    resumed_prior_elapsed_seconds: Annotated[float, Field(ge=0)] = 0.0
    resumed_prior_compiler_stages: Annotated[int, Field(ge=0)] = 0
    resumed_prior_critic_stages: Annotated[int, Field(ge=0)] = 0
    resumed_prior_estimated_cost_usd: Annotated[Decimal, Field(ge=0)] = Decimal(0)
    resume_contract_match: bool | None = None
    resume_replay_warning: str | None = None
    resume_mode: Literal[
        "none",
        "exact_contract_reuse",
        "current_host_recertification",
        "continued_with_provider",
    ] = "none"
