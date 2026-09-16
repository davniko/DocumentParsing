from __future__ import annotations

from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, model_validator

from .models import (
    AgentBindingProposal,
    AgentOccurrence,
    AnchorOverride,
    CarrierAssessment,
    CoherenceCandidateDecision,
    CompilerAgentOutput,
    CriticAgentOutput,
    CriticFinding,
    CriticOccurrenceRemoval,
    Derivation,
    GroupKind,
    InventoryBindingId,
    NonEmptyText,
    SemanticOnlyTargetFactProposal,
    ValueKind,
)

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
OccurrenceId = Annotated[str, StringConstraints(pattern=r"^occurrence_[0-9]{5}$")]
CompilerOccurrenceId = Annotated[str, StringConstraints(pattern=r"^compiler_occurrence_[0-9]{5}$")]


class PathDependency(BaseModel):
    model_config = _STRICT

    kind: Literal["target_path"]
    value: NonEmptyText


class BindingDependency(BaseModel):
    model_config = _STRICT

    kind: Literal["binding"]
    value: NonEmptyText


Dependency = Annotated[
    PathDependency | BindingDependency,
    Field(discriminator="kind"),
]


class TargetBindingRendering(BaseModel):
    model_config = _STRICT

    render_mode: Literal["target_binding"]
    target_paths: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]


class DeterministicAuxiliaryRendering(BaseModel):
    model_config = _STRICT

    render_mode: Literal["deterministic_auxiliary"]


class DeterministicDerivedRendering(BaseModel):
    model_config = _STRICT

    render_mode: Literal["deterministic_derived"]
    target_paths: tuple[NonEmptyText, ...] = ()
    derivation: Derivation
    dependencies: Annotated[tuple[Dependency, ...], Field(min_length=1)]


class AgentResidualRendering(BaseModel):
    model_config = _STRICT

    render_mode: Literal["agent_residual"]
    target_paths: tuple[NonEmptyText, ...] = ()


class CarrierStaticRendering(BaseModel):
    model_config = _STRICT

    render_mode: Literal["carrier_static"]
    target_paths: tuple[NonEmptyText, ...] = ()


class LiteralStaticRendering(BaseModel):
    model_config = _STRICT

    render_mode: Literal["literal_static"]


BindingRendering = Annotated[
    TargetBindingRendering
    | DeterministicAuxiliaryRendering
    | DeterministicDerivedRendering
    | AgentResidualRendering
    | CarrierStaticRendering
    | LiteralStaticRendering,
    Field(discriminator="render_mode"),
]


class DiscriminatedBindingProposal(BaseModel):
    """Common binding identity plus a schema-discriminated rendering contract."""

    model_config = _STRICT

    logical_key: NonEmptyText
    value_kind: ValueKind
    group_kind: GroupKind
    group_key: NonEmptyText
    rendering: BindingRendering
    occurrences: Annotated[tuple[AgentOccurrence, ...], Field(min_length=1)]
    rationale: NonEmptyText


def restore_legacy_binding(
    proposal: DiscriminatedBindingProposal,
) -> AgentBindingProposal:
    """Translate the provider-facing discriminated contract to the host contract."""

    rendering = proposal.rendering
    target_paths: tuple[str, ...] = ()
    derivation: Derivation | None = None
    dependency_paths: tuple[str, ...] = ()
    dependency_bindings: tuple[str, ...] = ()
    if isinstance(
        rendering,
        (
            TargetBindingRendering,
            DeterministicDerivedRendering,
            AgentResidualRendering,
            CarrierStaticRendering,
        ),
    ):
        target_paths = rendering.target_paths
    if isinstance(rendering, DeterministicDerivedRendering):
        derivation = rendering.derivation
        dependency_paths = tuple(
            dependency.value
            for dependency in rendering.dependencies
            if isinstance(dependency, PathDependency)
        )
        dependency_bindings = tuple(
            dependency.value
            for dependency in rendering.dependencies
            if isinstance(dependency, BindingDependency)
        )
    return AgentBindingProposal(
        logical_key=proposal.logical_key,
        render_mode=rendering.render_mode,
        value_kind=proposal.value_kind,
        group_kind=proposal.group_kind,
        group_key=proposal.group_key,
        target_paths=target_paths,
        derivation=derivation,
        dependency_paths=dependency_paths,
        dependency_bindings=dependency_bindings,
        occurrences=proposal.occurrences,
        rationale=proposal.rationale,
    )


class DiscriminatedCompilerAgentOutput(BaseModel):
    """Compiler result whose binding modes cannot express illegal field combinations.

    Cross-binding semantic ownership is deliberately validated by the independent host after
    exact occurrence resolution. Keeping that rule out of the provider schema lets a retained
    candidate take the bounded local-repair path instead of regenerating the complete response.
    """

    model_config = _STRICT

    carrier: CarrierAssessment
    anchor_overrides: tuple[AnchorOverride, ...]
    bindings: tuple[DiscriminatedBindingProposal, ...]
    unresolved: tuple[NonEmptyText, ...]
    semantic_only_target_facts: tuple[SemanticOnlyTargetFactProposal, ...] = ()


def restore_legacy_compiler(
    output: DiscriminatedCompilerAgentOutput,
) -> CompilerAgentOutput:
    """Translate a discriminated compiler response without changing host semantics."""

    return CompilerAgentOutput(
        carrier=output.carrier,
        anchor_overrides=output.anchor_overrides,
        bindings=tuple(restore_legacy_binding(binding) for binding in output.bindings),
        unresolved=output.unresolved,
        all_shipment_dependent_surfaces_accounted_for=not output.unresolved,
        semantic_only_target_facts=output.semantic_only_target_facts,
    )


DISCRIMINATED_BINDING_ADAPTER: TypeAdapter[DiscriminatedBindingProposal] = TypeAdapter(
    DiscriminatedBindingProposal
)


def discriminate_binding(
    legacy: AgentBindingProposal,
) -> DiscriminatedBindingProposal:
    """Translate one accepted legacy proposal without changing its semantics."""

    if legacy.render_mode == "target_binding":
        rendering: BindingRendering = TargetBindingRendering(
            render_mode=legacy.render_mode,
            target_paths=legacy.target_paths,
        )
    elif legacy.render_mode == "deterministic_auxiliary":
        rendering = DeterministicAuxiliaryRendering(render_mode=legacy.render_mode)
    elif legacy.render_mode == "deterministic_derived":
        if legacy.derivation is None:
            raise ValueError("validated deterministic-derived binding lacks its derivation")
        dependencies: tuple[PathDependency | BindingDependency, ...] = (
            *(PathDependency(kind="target_path", value=value) for value in legacy.dependency_paths),
            *(
                BindingDependency(kind="binding", value=value)
                for value in legacy.dependency_bindings
            ),
        )
        rendering = DeterministicDerivedRendering(
            render_mode=legacy.render_mode,
            target_paths=legacy.target_paths,
            derivation=legacy.derivation,
            dependencies=dependencies,
        )
    elif legacy.render_mode == "agent_residual":
        rendering = AgentResidualRendering(
            render_mode=legacy.render_mode,
            target_paths=legacy.target_paths,
        )
    elif legacy.render_mode == "carrier_static":
        rendering = CarrierStaticRendering(
            render_mode=legacy.render_mode,
            target_paths=legacy.target_paths,
        )
    else:
        rendering = LiteralStaticRendering(render_mode=legacy.render_mode)
    return DiscriminatedBindingProposal(
        logical_key=legacy.logical_key,
        value_kind=legacy.value_kind,
        group_kind=legacy.group_kind,
        group_key=legacy.group_key,
        rendering=rendering,
        occurrences=legacy.occurrences,
        rationale=legacy.rationale,
    )


def project_legacy_candidate(candidate: dict[str, object]) -> dict[str, object]:
    """Project even-invalid legacy JSON into the new shape for schema-boundary probes."""

    render_mode = candidate.get("render_mode")
    rendering: dict[str, object] = {"render_mode": render_mode}
    if render_mode in {
        "target_binding",
        "deterministic_derived",
        "agent_residual",
        "carrier_static",
    }:
        rendering["target_paths"] = candidate.get("target_paths", [])
    dependencies = [
        *(
            {"kind": "target_path", "value": value}
            for value in cast(list[object], candidate.get("dependency_paths", []))
        ),
        *(
            {"kind": "binding", "value": value}
            for value in cast(list[object], candidate.get("dependency_bindings", []))
        ),
    ]
    if render_mode == "deterministic_derived" or dependencies:
        rendering["derivation"] = candidate.get("derivation")
        rendering["dependencies"] = dependencies
    return {
        key: candidate[key]
        for key in (
            "logical_key",
            "value_kind",
            "group_kind",
            "group_key",
            "occurrences",
            "rationale",
        )
        if key in candidate
    } | {"rendering": rendering}


class OccurrenceReference(BaseModel):
    """Provider output reference to a host-resolved immutable source occurrence."""

    model_config = _STRICT

    occurrence_id: OccurrenceId


class CompilerOccurrenceReference(BaseModel):
    """Reference to a host-resolved exact source span supplied with the compiler request."""

    model_config = _STRICT

    occurrence_id: CompilerOccurrenceId


class CriticCandidateReview(BaseModel):
    """One explicit disposition for a host-supplied critic lead."""

    model_config = _STRICT

    candidate_id: NonEmptyText
    conclusion: Literal["valid_existing_contract", "defect_requires_revision"]
    rationale: NonEmptyText


class CriticAuditCoverage(BaseModel):
    """Machine-checkable receipt that every independent critic facet was audited."""

    model_config = _STRICT

    literal_completeness_checked: Literal[True]
    target_ownership_checked: Literal[True]
    topology_and_grouping_checked: Literal[True]
    derivations_checked: Literal[True]
    carrier_boundary_checked: Literal[True]
    identifier_relationships_checked: Literal[True]
    candidate_reviews: tuple[CriticCandidateReview, ...]


class CompactCriticPassOutput(BaseModel):
    """A compact-contract pass has no mutation surface by construction."""

    model_config = _STRICT

    verdict: Literal["pass"]
    coverage: CriticAuditCoverage
    coherence_decisions: tuple[CoherenceCandidateDecision, ...] = ()
    rationale: NonEmptyText


class CompactCriticRevisionOutput(BaseModel):
    """A compact-contract revision contains a complete, schema-valid local transaction."""

    model_config = _STRICT

    verdict: Literal["revise"]
    coverage: CriticAuditCoverage
    findings: Annotated[tuple[CriticFinding, ...], Field(min_length=1)]
    remove_binding_logical_keys: tuple[NonEmptyText, ...] = ()
    additional_bindings: tuple[DiscriminatedBindingProposal, ...] = ()
    semantic_only_target_facts: tuple[SemanticOnlyTargetFactProposal, ...] = ()
    coherence_decisions: tuple[CoherenceCandidateDecision, ...] = ()
    rationale: NonEmptyText

    @model_validator(mode="after")
    def removal_keys_are_unique(self) -> CompactCriticRevisionOutput:
        if len(set(self.remove_binding_logical_keys)) != len(self.remove_binding_logical_keys):
            raise ValueError("critic removal logical keys must be unique")
        return self


def restore_legacy_compact_critic(
    output: CompactCriticPassOutput | CompactCriticRevisionOutput,
) -> CriticAgentOutput:
    """Translate a compact discriminated critic decision to the stable host artifact."""

    from .host import inventory_binding_id

    if isinstance(output, CompactCriticPassOutput):
        return CriticAgentOutput(
            verdict="pass",
            findings=(),
            remove_inventory_binding_ids=(),
            additional_bindings=(),
            semantic_only_target_facts=(),
            coherence_decisions=output.coherence_decisions,
            exhaustive_audit_receipt=True,
            rationale=output.rationale,
        )
    return CriticAgentOutput(
        verdict="revise",
        findings=output.findings,
        remove_inventory_binding_ids=tuple(
            inventory_binding_id(key) for key in output.remove_binding_logical_keys
        ),
        additional_bindings=tuple(
            restore_legacy_binding(binding) for binding in output.additional_bindings
        ),
        semantic_only_target_facts=output.semantic_only_target_facts,
        coherence_decisions=output.coherence_decisions,
        exhaustive_audit_receipt=True,
        rationale=output.rationale,
    )


class CriticPassOutput(BaseModel):
    """A pass cannot carry fields that mutate or reclassify the template."""

    model_config = _STRICT

    verdict: Literal["pass"]
    coherence_decisions: tuple[CoherenceCandidateDecision, ...] = ()
    rationale: NonEmptyText


class CriticRevisionOutput(BaseModel):
    """A revision makes its non-empty finding requirement structural."""

    model_config = _STRICT

    verdict: Literal["revise"]
    findings: Annotated[tuple[CriticFinding, ...], Field(min_length=1)]
    remove_inventory_binding_ids: tuple[InventoryBindingId, ...] = ()
    occurrence_removals: tuple[CriticOccurrenceRemoval, ...] = ()
    additional_bindings: tuple[DiscriminatedBindingProposal, ...] = ()
    semantic_only_target_facts: tuple[SemanticOnlyTargetFactProposal, ...] = ()
    coherence_decisions: tuple[CoherenceCandidateDecision, ...] = ()
    rationale: NonEmptyText

    @model_validator(mode="after")
    def removal_ids_are_unique(self) -> CriticRevisionOutput:
        if len(set(self.remove_inventory_binding_ids)) != len(self.remove_inventory_binding_ids):
            raise ValueError("critic inventory removal IDs must be unique")
        return self


DiscriminatedCriticOutput = Annotated[
    CriticPassOutput | CriticRevisionOutput,
    Field(discriminator="verdict"),
]
DISCRIMINATED_CRITIC_ADAPTER: TypeAdapter[DiscriminatedCriticOutput] = TypeAdapter(
    DiscriminatedCriticOutput
)


def discriminate_critic(
    legacy: CriticAgentOutput,
) -> CriticPassOutput | CriticRevisionOutput:
    """Translate one accepted critic result without changing its host semantics."""

    if legacy.verdict == "pass":
        return CriticPassOutput(
            verdict="pass",
            coherence_decisions=legacy.coherence_decisions,
            rationale=legacy.rationale,
        )
    return CriticRevisionOutput(
        verdict="revise",
        findings=legacy.findings,
        remove_inventory_binding_ids=legacy.remove_inventory_binding_ids,
        occurrence_removals=legacy.occurrence_removals,
        additional_bindings=tuple(
            discriminate_binding(binding) for binding in legacy.additional_bindings
        ),
        semantic_only_target_facts=legacy.semantic_only_target_facts,
        coherence_decisions=legacy.coherence_decisions,
        rationale=legacy.rationale,
    )


def project_legacy_critic_candidate(candidate: dict[str, object]) -> dict[str, object]:
    """Project a legacy candidate while retaining illegal non-empty pass operations.

    Empty legacy patch arrays are omitted for a pass because they carry no semantics. Any non-empty
    operation remains present and is therefore rejected as an extra field by the pass variant.
    """

    verdict = candidate.get("verdict")
    projected: dict[str, object] = {
        key: candidate[key] for key in ("verdict", "rationale") if key in candidate
    }
    operation_fields = (
        "findings",
        "remove_inventory_binding_ids",
        "occurrence_removals",
        "remove_binding_logical_keys",
        "additional_bindings",
        "semantic_only_target_facts",
        "coherence_decisions",
    )
    for key in operation_fields:
        value = candidate.get(key)
        if verdict == "revise" or value:
            projected[key] = value
    additions = projected.get("additional_bindings")
    if isinstance(additions, list):
        projected["additional_bindings"] = [
            project_legacy_candidate(cast(dict[str, object], binding)) for binding in additions
        ]
    return projected
