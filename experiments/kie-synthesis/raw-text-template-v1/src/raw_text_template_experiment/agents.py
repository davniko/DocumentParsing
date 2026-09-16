from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar, cast

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.linguistic_probe_runtime import (
    load_provider_key,
    model_messages,
    usage_receipt,
)
from document_ocr.synthesis.usage_receipt import LinguisticUsageReceipt
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator
from pydantic_ai import Agent, ModelProfile, ModelRetry, NativeOutput, capture_run_messages
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from .compact_contract import BINDING_COLUMNS, OCCURRENCE_COLUMNS, compact_critic_payload
from .models import (
    AgentBindingProposal,
    AgentContractProtocol,
    AgentOccurrence,
    AgentStageArtifact,
    AnchorOverride,
    CarrierAssessment,
    CoherenceCandidateDecision,
    CompilerAgentOutput,
    CriticAgentOutput,
    CriticFinding,
    CriticOccurrenceRemoval,
    NonEmptyText,
    ProviderConfig,
    SemanticOnlyTargetFactProposal,
)
from .optimization_contract import (
    CompactCriticPassOutput,
    CompactCriticRevisionOutput,
    CompilerOccurrenceReference,
    DiscriminatedBindingProposal,
    DiscriminatedCompilerAgentOutput,
    restore_legacy_binding,
    restore_legacy_compiler,
)
from .staged_contract import (
    StagedAuditOutput,
    StagedCriticPlanOutput,
    StagedFacetAuditOutput,
    StagedPlanBindingProposal,
    StagedPlanOccurrenceAppend,
    build_local_compiler_repair_payload,
    build_staged_audit_facet_payloads,
    build_staged_audit_payload,
    build_staged_plan_payload,
    compact_staged_audit_request,
    compact_staged_plan_request,
    host_completable_coherence_decisions,
    merge_staged_facet_audits,
    normalize_staged_facet_audit,
    restore_staged_plan,
    staged_audit_pass,
    validate_staged_audit,
)

OutputT = TypeVar("OutputT", bound=BaseModel)
_STAGED_PROTOCOLS = {
    "staged_local_v4",
    "faceted_staged_local_v5",
    "partitioned_staged_local_v6",
    "candidate_first_staged_local_v7",
    "candidate_first_staged_local_v8",
}
_PARTITIONED_PROTOCOLS = {
    "partitioned_staged_local_v6",
    "candidate_first_staged_local_v7",
    "candidate_first_staged_local_v8",
}
_CANDIDATE_FIRST_PROTOCOLS = {
    "candidate_first_staged_local_v7",
    "candidate_first_staged_local_v8",
}
_HYBRID_PROTOCOLS = {"hybrid_reference_partitioned_v10"}
_REFERENCE_PROTOCOLS = {
    "reference_compact_v3",
    "relational_reference_compact_v9",
    *_HYBRID_PROTOCOLS,
}
_LOCAL_COMPILER_REPAIR_PROTOCOLS = {
    *_STAGED_PROTOCOLS,
    "relational_reference_compact_v9",
    *_HYBRID_PROTOCOLS,
}
_COMPILER_ACTIONABLE_REFERENCE = re.compile(
    r"documentPatch(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[0-9]+\])+"
    r"|agent_binding_[0-9a-f]{16}|anchor_binding_[0-9]{4}"
)

_DISCRIMINATED_BINDING_ADAPTER = """
# Provider-facing delta-binding schema adapter (v3)

The response JSON schema is authoritative. For every proposed binding, common identity,
classification, occurrences, and rationale fields remain on the binding object. Render-mode fields
are now structurally discriminated under `rendering`:

- `target_binding`: `rendering={render_mode,target_paths}`.
- `deterministic_auxiliary`: `rendering={render_mode}`.
- `deterministic_derived`: `rendering={render_mode,target_paths,derivation,dependencies}`; every
  dependency is `{kind:"target_path"|"binding",value:"..."}`.
- `agent_residual` and `carrier_static`: `rendering={render_mode,target_paths}`.
- `literal_static`: `rendering={render_mode}`.

Do not emit the legacy flat `render_mode`, `target_paths`, `derivation`, `dependency_paths`, or
`dependency_bindings` fields. Before returning, verify that each target path has only one proposed
logical owner. Consolidate repeated occurrences into that owner; a distinct contextual variant
must not claim the same target path a second time. The provider schema uses `unresolved` as the
single completion signal and intentionally omits `all_shipment_dependent_surfaces_accounted_for`;
an empty list submits the candidate to independent host and critic validation. The semantic rules
below remain unchanged.

`bindings` is a delta against `anchorBindings`, not a repetition of the accepted anchor inventory.
Never copy an accepted anchor occurrence into `bindings` unless its `anchorBindingId` is present in
`anchor_overrides`. For an unanchored repeated appearance of a retained fact, propose only the new
physical occurrence under the retained anchor's logical key. Every overridden anchor must still
receive complete replacement ownership or an explicit semantic-only disposition.
""".strip()

_COMPACT_CRITIC_ADAPTER = """
# Compact exhaustive critic protocol (v8)

The payload is a semantic table encoding of the objects named by the rules below. Read
`bindingRows` using `compactContract.bindingColumns`, `occurrenceRows` using
`compactContract.occurrenceColumns`, and target path IDs through `targetPathTable`. Opaque source
provenance IDs are retained by the host but omitted from this provider view because they carry no
audit semantics. A marker `⟦binding_NNNN⟧` in `annotatedSource` refers to the row with that binding
ID. The `logicalKey` column of `bindingRows` is the exact removal and append vocabulary.
`annotatedSource` preserves every original source character between binding boundary markers and
is the authoritative provider view: unmarked text is currently unowned. Use it to audit both the
semantics and exact extent of every owned and unowned span without a duplicated source view.

The response has one top-level `decision`. Before returning either verdict, inspect every source
line and every inventory row for all six required coverage facets. Set each coverage field to true
only after completing that audit. After inspecting all entries in `literalLineReviewIds`, return
their exact count in `literal_line_count`. The schema enforces the expected count; report defects
separately in `findings`.
This is a required exhaustive receipt, not a sample. `candidate_receipt` is positionally aligned
with the concatenation of `reviewCandidates` followed by `remainingRiskCandidates`: return exactly
one conclusion string for every candidate, in request order. Do not repeat candidate IDs or
candidate rationales in the response; the host reconstructs IDs by position and requires grounded
explanations only for defects in `findings`. The receipt records your audit judgment. The host is
the authority for mechanically proven required revisions and will independently reject a pass or
patch that leaves one unresolved; do not spend a response trying to satisfy a receipt instead of
returning the complete repair.
Review candidates are high-recall host leads and may be valid; remaining risks are proven unowned
and therefore require a revision, even when their correct owner is `literal_static`. A `revise`
decision must aggregate every
independently visible defect into one complete local transaction; do not stop after finding the
first defect. Binding proposals use the v2 discriminated `rendering` object described by the
response schema. For every new occurrence available in `occurrenceCandidates`, return only its
exact `occurrence_id` handle copied from the row. New-occurrence handles have format
`cand_Lxxxxx_xxxxx`; they are not `compiler_occurrence_NNNNN` ordinals. Each handle embeds its
source start-line ID; that line must agree with the finding you intend to repair, and you must
verify the complete table row rather than guessing or constructing an ID. Use a literal occurrence
object solely when the exact intended span is genuinely
absent from that table. When an uncovered surface is merely another physical occurrence
of an existing binding with unchanged semantics, use `occurrence_appends`; do not remove or restate
that binding. When only one or more existing occurrences are semantically wrong, use
`occurrence_removals`; do not remove and restate the otherwise unchanged binding. Use full removal
plus `additional_bindings` only when the binding's semantic contract actually changes. Output full
target-path strings from the second column of `targetPathTable`, never the short path IDs.

For every `cross_field_semantic_relation` row whose `requiredRevision` is true, return exactly one
`coherence_decisions` entry. Use `apply_suggestion` with the zero-based suggestion index only when
the source context confirms that host-supplied arithmetic relationship; use
`reviewed_independent` for an incidental numeric resemblance and `review_required` only when the
relationship is genuinely unresolved.
These decisions are metadata, not structural patches: a complete, defect-free audit may return
`pass` with them. Copy the enclosing row's exact `candidateId` into each decision; do not rename it
after the candidate kind. Mark their positional candidate receipts `valid_existing_contract`
because the same decision completes the contract. Never invent a path, member, or arithmetic
relationship.

When the semantic rules below say `bindingInventory`, consult `bindingRows` plus `occurrenceRows`.
When they say `allowedTargetPaths`, consult `targetPathTable`. Exact-match candidate data remains in
the occurrence table and is unchanged.
""".strip()

_REFERENCE_COMPILER_ADAPTER = """
# Host-resolved occurrence-reference protocol (v4)

The payload contains `occurrenceCandidates`, an immutable table of exact source spans already
resolved by the host. Read each row using the table's `columns`; numbered source supplies its full
context. In `carrier.evidence_occurrences` and every binding's `occurrences`, emit
`{"occurrence_id":"cand_Lxxxxx_xxxxx"}` with the row's exact `occurrenceId` whenever a listed row
is the intended complete surface. The first numeric component is the source start-line number; the
second is a stable table identity, not a row number to guess. The response schema enumerates the
only legal IDs. Do not copy that row's line or text
fields into the response. Use a full `{line_start,line_end,source_text,occurrence_index}` occurrence
only when the exact intended span is genuinely absent from the table; copy such text byte-for-byte.

The binding `rendering` object follows the v2 discriminated protocol. Before returning, verify that
each target path has exactly one logical owner and that repeated physical appearances are grouped
as occurrences of that one owner. The semantic compiler rules below remain authoritative.
""".strip()

_COMPACT_COMPILER_INPUT_ADAPTER = """
# Indexed initial compiler input (v6)

The first-pass payload replaces repeated verbose inventories with lossless indexed tables. For
each table, zip `columns` with every row; a column position is never a semantic index.

- `targetPathTable` is the exhaustive `allowedTargetPaths` vocabulary. Copy full paths from its
  `targetPath` column into the response; `targetPathIndex` is only a compact input handle.
- `anchorBindingTable` is the complete accepted `anchorBindings` inventory. Resolve its path and
  component indexes through `targetPathTable`, and its occurrence indexes through
  `anchorOccurrenceTable`.
- `coBindingTable` is the complete `requiredTargetCoBindings` inventory and uses target-path
  indexes.
- `riskCandidateTable` is the complete risk inventory. A `currentlyOwnedByAcceptedAnchor=false`
  row requires ownership; byte offsets are omitted because its exact line and source text plus the
  numbered source are the authoritative evidence.

`sourceLabel` remains the complete nested semantic value tree. No table is a sample, and no row may
be skipped. `anchorBindingsAuthorizedForInitialReview`, when present, is the exhaustive
initial-pass authority for `anchor_overrides`; older requests use
`anchorBindingsRequiringSemanticReview`. The response schema permits only those IDs. Retain every
other accepted anchor unchanged. Do not copy, widen, replace, or overlap an unauthorized anchor
through a binding proposal.

For a v8 request, every ID in `anchorBindingsRequiringSemanticReview` must be overridden. Resolve
every ID in `anchorBindingsWithCrossFactEqualityAmbiguity` either by overriding it or by returning
it once in `retained_cross_fact_anchor_ids` after verifying that its one printed surface genuinely
summarizes all independently mutable components. The two dispositions are disjoint and exhaustive;
this receipt prevents known topology ambiguity from spilling into a later repair call. All compiler
rules referring to the verbose names above apply to their table forms.
""".strip()

_REFERENCE_OCCURRENCE_COLUMNS = (
    "occurrenceId",
    "lineStart",
    "lineEnd",
    "sourceText",
    "occurrenceIndex",
)

_COMPILER_REPAIR_ADAPTER = """
# Local compiler repair protocol (v3)

The payload contains a lossless `candidateSlice` of the current candidate, an exact
`requiredRevision` from the host, and only the source/target context authorized for this repair.
The host retains every omitted binding. Return only the smallest complete patch that makes the
current candidate satisfy every listed defect. Do not repeat unaffected bindings. A replacement
binding with an existing logical key atomically replaces that binding; use
`remove_binding_logical_keys` only to delete a binding without a replacement. An
`additional_anchor_overrides` entry may copy only an exact `anchorBindingId` nested under the
supplied `anchorBindings`. Diagnostic `agent_binding_*` identifiers in `requiredRevision` are not
edit handles: repair their corresponding candidate-slice binding and never transform or guess one
into an `anchor_binding_*` ID. Every entry in `invalidPriorTargetPaths` is an existing declaration
that cannot resolve against the source label: remove it through the exact
`remove_semantic_only_target_paths` handle (or replace its owning binding), and never return that
path in an addition or replacement. When one
printed surface was incorrectly assigned to independently mutable target facts, retain only the
fact the source topology supports and declare a genuinely unprinted fact through
`additional_semantic_only_target_facts`; never merge independent facts merely because their
current values are equal. Replacement bindings use the discriminated `rendering` schema.
""".strip()


class _CompilerRepairOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    remove_binding_logical_keys: Annotated[
        tuple[str, ...],
        Field(
            description=(
                "Existing bindings to delete without replacement. A replacement_bindings row "
                "with an existing logical_key replaces that binding atomically, so its key need "
                "not also appear here."
            ),
            json_schema_extra={"uniqueItems": True},
        ),
    ] = ()
    replacement_bindings: tuple[DiscriminatedBindingProposal, ...] = ()
    remove_anchor_override_ids: Annotated[
        tuple[str, ...], Field(json_schema_extra={"uniqueItems": True})
    ] = ()
    additional_anchor_overrides: tuple[AnchorOverride, ...] = ()
    remove_semantic_only_target_paths: Annotated[
        tuple[str, ...], Field(json_schema_extra={"uniqueItems": True})
    ] = ()
    additional_semantic_only_target_facts: tuple[SemanticOnlyTargetFactProposal, ...] = ()
    carrier_replacement: CarrierAssessment | None = None
    unresolved_replacement: tuple[NonEmptyText, ...] | None = None
    rationale: NonEmptyText

    @model_validator(mode="after")
    def patch_is_nonempty_and_unique(self) -> _CompilerRepairOutput:
        operations = (
            self.remove_binding_logical_keys,
            self.replacement_bindings,
            self.remove_anchor_override_ids,
            self.additional_anchor_overrides,
            self.remove_semantic_only_target_paths,
            self.additional_semantic_only_target_facts,
        )
        if (
            not any(operations)
            and self.carrier_replacement is None
            and (self.unresolved_replacement is None)
        ):
            raise ValueError("compiler repair must contain at least one operation")
        for name, values in (
            ("binding removals", self.remove_binding_logical_keys),
            ("anchor-override removals", self.remove_anchor_override_ids),
            ("semantic-only removals", self.remove_semantic_only_target_paths),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"compiler repair {name} must be unique")
        replacement_keys = tuple(row.logical_key for row in self.replacement_bindings)
        if len(set(replacement_keys)) != len(replacement_keys):
            raise ValueError("compiler replacement binding logical keys must be unique")
        return self


class _CriticOccurrenceAppend(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    logical_key: NonEmptyText
    occurrences: Annotated[tuple[AgentOccurrence, ...], Field(min_length=1)]
    rationale: NonEmptyText


class _CriticOccurrenceRemovalReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    logical_key: NonEmptyText
    occurrence_ids: Annotated[tuple[str, ...], Field(min_length=1)]
    rationale: NonEmptyText


def _scoped_compiler_repair_output_type(
    prior: CompilerAgentOutput,
    *,
    removable_binding_keys: Sequence[str] | None = None,
    removable_anchor_ids: Sequence[str] | None = None,
    removable_semantic_paths: Sequence[str] | None = None,
    addable_anchor_ids: Sequence[str] | None = None,
) -> type[_CompilerRepairOutput]:
    all_binding_keys = tuple(dict.fromkeys(row.logical_key for row in prior.bindings))
    all_anchor_ids = tuple(dict.fromkeys(row.anchor_binding_id for row in prior.anchor_overrides))
    all_semantic_paths = tuple(
        dict.fromkeys(row.target_path for row in prior.semantic_only_target_facts)
    )
    binding_keys = (
        tuple(removable_binding_keys) if removable_binding_keys is not None else all_binding_keys
    )
    anchor_ids = tuple(removable_anchor_ids) if removable_anchor_ids is not None else all_anchor_ids
    semantic_paths = (
        tuple(removable_semantic_paths)
        if removable_semantic_paths is not None
        else all_semantic_paths
    )
    authorized_additional_anchor_ids = (
        tuple(addable_anchor_ids) if addable_anchor_ids is not None else None
    )
    for name, selected, available in (
        ("binding", binding_keys, all_binding_keys),
        ("anchor override", anchor_ids, all_anchor_ids),
        ("semantic-only target", semantic_paths, all_semantic_paths),
    ):
        if len(set(selected)) != len(selected) or not set(selected) <= set(available):
            raise ValueError(f"compiler repair {name} scope is invalid")
    if authorized_additional_anchor_ids is not None and len(
        set(authorized_additional_anchor_ids)
    ) != len(authorized_additional_anchor_ids):
        raise ValueError("compiler repair addable anchor scope is invalid")
    fields: dict[str, Any] = {}
    if binding_keys:
        fields["remove_binding_logical_keys"] = (
            Annotated[
                tuple[Literal.__getitem__(binding_keys), ...],  # type: ignore[misc]
                Field(json_schema_extra={"uniqueItems": True}),
            ],
            (),
        )
    else:
        fields["remove_binding_logical_keys"] = (
            Annotated[
                tuple[str, ...],
                Field(max_length=0, json_schema_extra={"uniqueItems": True}),
            ],
            (),
        )
    if anchor_ids:
        fields["remove_anchor_override_ids"] = (
            Annotated[
                tuple[Literal.__getitem__(anchor_ids), ...],  # type: ignore[misc]
                Field(json_schema_extra={"uniqueItems": True}),
            ],
            (),
        )
    else:
        fields["remove_anchor_override_ids"] = (
            Annotated[
                tuple[str, ...],
                Field(max_length=0, json_schema_extra={"uniqueItems": True}),
            ],
            (),
        )
    if semantic_paths:
        fields["remove_semantic_only_target_paths"] = (
            Annotated[
                tuple[Literal.__getitem__(semantic_paths), ...],  # type: ignore[misc]
                Field(json_schema_extra={"uniqueItems": True}),
            ],
            (),
        )
    else:
        fields["remove_semantic_only_target_paths"] = (
            Annotated[
                tuple[str, ...],
                Field(max_length=0, json_schema_extra={"uniqueItems": True}),
            ],
            (),
        )
    if authorized_additional_anchor_ids:
        additional_anchor_type = create_model(
            "ScopedCompilerRepairAdditionalAnchor_"
            + sha256_bytes(canonical_json_bytes(authorized_additional_anchor_ids))[:16],
            __base__=AnchorOverride,
            __module__=__name__,
            anchor_binding_id=(Literal.__getitem__(authorized_additional_anchor_ids), ...),
        )
        fields["additional_anchor_overrides"] = (
            Annotated[
                tuple[additional_anchor_type, ...],  # type: ignore[valid-type]
                Field(json_schema_extra={"uniqueItems": True}),
            ],
            (),
        )
    elif authorized_additional_anchor_ids is not None:
        fields["additional_anchor_overrides"] = (
            Annotated[tuple[AnchorOverride, ...], Field(max_length=0)],
            (),
        )

    @model_validator(mode="after")
    def replacement_operations_are_transactional(
        value: _CompilerRepairOutput,
    ) -> _CompilerRepairOutput:
        out_of_scope_binding_replacements = sorted(
            {row.logical_key for row in value.replacement_bindings}
            & (set(all_binding_keys) - set(binding_keys))
        )
        missing_override_removals = sorted(
            {row.anchor_binding_id for row in value.additional_anchor_overrides}
            & set(all_anchor_ids) - set(value.remove_anchor_override_ids)
        )
        missing_semantic_removals = sorted(
            {row.target_path for row in value.additional_semantic_only_target_facts}
            & set(all_semantic_paths) - set(value.remove_semantic_only_target_paths)
        )
        rationale_only_override_replacements = sorted(
            {row.anchor_binding_id for row in value.additional_anchor_overrides}
            & set(value.remove_anchor_override_ids)
        )
        rationale_only_semantic_replacements = sorted(
            {row.target_path for row in value.additional_semantic_only_target_facts}
            & set(value.remove_semantic_only_target_paths)
        )
        errors = (
            *(f"binding:{key}" for key in out_of_scope_binding_replacements),
            *(f"anchor_override:{key}" for key in missing_override_removals),
            *(f"semantic_only:{key}" for key in missing_semantic_removals),
        )
        if errors:
            raise ValueError(
                "replacement operations reference existing state outside the local repair scope: "
                + ", ".join(errors)
            )
        rationale_only = (
            *(f"anchor_override:{key}" for key in rationale_only_override_replacements),
            *(f"semantic_only:{key}" for key in rationale_only_semantic_replacements),
        )
        if rationale_only:
            raise ValueError(
                "rationale-only replacements cannot change compiler behavior: "
                + ", ".join(rationale_only)
            )
        return value

    digest = sha256_bytes(
        canonical_json_bytes(
            (
                all_binding_keys,
                all_anchor_ids,
                all_semantic_paths,
                binding_keys,
                anchor_ids,
                semantic_paths,
                authorized_additional_anchor_ids,
            )
        )
    )[:16]
    return cast(
        type[_CompilerRepairOutput],
        create_model(
            f"ScopedCompilerRepairOutput_{digest}",
            __base__=_CompilerRepairOutput,
            __module__=__name__,
            __validators__=cast(
                dict[str, Callable[..., Any]],
                {
                    "replacement_operations_are_transactional": (
                        replacement_operations_are_transactional
                    )
                },
            ),
            **fields,
        ),
    )


def _apply_compiler_repair(
    prior: CompilerAgentOutput, repair: _CompilerRepairOutput
) -> CompilerAgentOutput:
    prior_binding_keys = {row.logical_key for row in prior.bindings}
    explicit_binding_removals = set(repair.remove_binding_logical_keys)
    unknown_bindings = explicit_binding_removals - prior_binding_keys
    if unknown_bindings:
        raise ValueError("compiler repair removes unknown bindings")
    replacements = tuple(restore_legacy_binding(row) for row in repair.replacement_bindings)
    # Same-key replacement is one atomic operation. Requiring the provider to duplicate the key
    # in a second removal array conveys no extra intent and caused valid local repairs to consume
    # another structured-output retry. The scoped output type above still rejects replacement of
    # any retained key outside the host-authorized local slice.
    remove_bindings = explicit_binding_removals | (
        {row.logical_key for row in replacements} & prior_binding_keys
    )
    retained_bindings = tuple(
        row for row in prior.bindings if row.logical_key not in remove_bindings
    )

    remove_overrides = set(repair.remove_anchor_override_ids)
    retained_overrides = tuple(
        row for row in prior.anchor_overrides if row.anchor_binding_id not in remove_overrides
    )
    added_override_ids = tuple(row.anchor_binding_id for row in repair.additional_anchor_overrides)
    if len(set(added_override_ids)) != len(added_override_ids) or set(added_override_ids) & {
        row.anchor_binding_id for row in retained_overrides
    }:
        raise ValueError("compiler repair produces duplicate anchor overrides")

    remove_semantic = set(repair.remove_semantic_only_target_paths)
    retained_semantic = tuple(
        row for row in prior.semantic_only_target_facts if row.target_path not in remove_semantic
    )
    added_semantic_paths = tuple(
        row.target_path for row in repair.additional_semantic_only_target_facts
    )
    if len(set(added_semantic_paths)) != len(added_semantic_paths) or set(added_semantic_paths) & {
        row.target_path for row in retained_semantic
    }:
        raise ValueError("compiler repair produces duplicate semantic-only target facts")

    payload = prior.model_dump(mode="python")
    payload.update(
        {
            "carrier": repair.carrier_replacement or prior.carrier,
            "anchor_overrides": (
                *retained_overrides,
                *repair.additional_anchor_overrides,
            ),
            "bindings": (*retained_bindings, *replacements),
            "semantic_only_target_facts": (
                *retained_semantic,
                *repair.additional_semantic_only_target_facts,
            ),
            "unresolved": (
                repair.unresolved_replacement
                if repair.unresolved_replacement is not None
                else prior.unresolved
            ),
            "all_shipment_dependent_surfaces_accounted_for": not (
                repair.unresolved_replacement
                if repair.unresolved_replacement is not None
                else prior.unresolved
            ),
        }
    )
    return CompilerAgentOutput.model_validate(payload)


def _compiler_operational_state(output: CompilerAgentOutput) -> bytes:
    """Hash compiler behavior while excluding explanations that cannot repair a contract."""

    def without_rationales(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: without_rationales(item) for key, item in value.items() if key != "rationale"
            }
        if isinstance(value, (tuple, list)):
            return tuple(without_rationales(item) for item in value)
        return value

    return canonical_json_bytes(without_rationales(output.model_dump(mode="json")))


def _compiler_repair_allowed_paths(payload: Mapping[str, Any]) -> frozenset[str]:
    """Read the declared target vocabulary for either compiler-repair protocol."""

    if "targetFacts" in payload:
        target_facts = payload["targetFacts"]
        if not isinstance(target_facts, (tuple, list)) or any(
            not isinstance(row, Mapping) or not isinstance(row.get("targetPath"), str)
            for row in target_facts
        ):
            raise ValueError("compiler repair targetFacts vocabulary is invalid")
        return frozenset(str(row["targetPath"]) for row in target_facts)

    allowed_paths = payload.get("allowedTargetPaths")
    if not isinstance(allowed_paths, (tuple, list)) or any(
        not isinstance(path, str) for path in allowed_paths
    ):
        raise ValueError("compiler repair lacks a declared target-path vocabulary")
    return frozenset(allowed_paths)


def _compiler_repair_allowed_anchor_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact accepted-anchor edit handles exposed to a local repair."""

    rows = payload.get("anchorBindings")
    if not isinstance(rows, (tuple, list)):
        raise ValueError("compiler repair anchorBindings vocabulary is invalid")
    identifiers: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("compiler repair anchor binding is invalid")
        occurrences = row.get("occurrences")
        if not isinstance(occurrences, (tuple, list)):
            raise ValueError("compiler repair anchor occurrence vocabulary is invalid")
        for occurrence in occurrences:
            if not isinstance(occurrence, Mapping) or not isinstance(
                occurrence.get("anchorBindingId"), str
            ):
                raise ValueError("compiler repair anchor occurrence is invalid")
            identifiers.append(cast(str, occurrence["anchorBindingId"]))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("compiler repair anchor IDs are not unique")
    return tuple(identifiers)


def _compiler_repair_addable_anchor_ids(
    payload: Mapping[str, Any], prior: CompilerAgentOutput
) -> tuple[str, ...]:
    """Expose only accepted anchors whose current state can actually change."""

    existing = {row.anchor_binding_id for row in prior.anchor_overrides}
    return tuple(
        anchor_id
        for anchor_id in _compiler_repair_allowed_anchor_ids(payload)
        if anchor_id not in existing
    )


def _compiler_repair_scope_references(
    payload: Mapping[str, Any], repair: _CompilerRepairOutput | None = None
) -> frozenset[str]:
    """Return exact identifiers the declared compiler transaction is authorized to change."""

    candidate_slice = payload.get("candidateSlice")
    references: set[str] = set()
    if isinstance(candidate_slice, Mapping):
        references.update(
            value
            for field in (
                "removableBindingKeys",
                "removableAnchorOverrideIds",
                "removableSemanticOnlyTargetPaths",
            )
            for value in candidate_slice.get(field, ())
            if isinstance(value, str)
        )
        for binding in candidate_slice.get("bindings", ()):
            if not isinstance(binding, Mapping):
                continue
            for field in ("logical_key", "target_paths", "dependency_paths"):
                values = (binding.get(field),) if field == "logical_key" else binding.get(field, ())
                references.update(value for value in values if isinstance(value, str))
            references.update(_compiler_proposal_draft_ids(binding))
    if repair is not None:
        references.update(repair.remove_binding_logical_keys)
        references.update(repair.remove_anchor_override_ids)
        references.update(repair.remove_semantic_only_target_paths)
        for binding in repair.replacement_bindings:
            references.add(binding.logical_key)
            restored = restore_legacy_binding(binding)
            references.update(restored.target_paths)
            references.update(restored.dependency_paths)
            references.update(_compiler_proposal_draft_ids(restored))
        references.update(row.anchor_binding_id for row in repair.additional_anchor_overrides)
        references.update(row.target_path for row in repair.additional_semantic_only_target_facts)
    return frozenset(references)


def _compiler_proposal_draft_ids(
    proposal: AgentBindingProposal | Mapping[str, Any],
) -> tuple[str, ...]:
    """Reproduce the stable draft handles emitted by ``resolve_agent_proposals``."""

    if isinstance(proposal, Mapping):
        logical_key = proposal.get("logical_key")
        occurrences = proposal.get("occurrences", ())
    else:
        logical_key = proposal.logical_key
        occurrences = proposal.occurrences
    if not isinstance(logical_key, str) or not isinstance(occurrences, (tuple, list)):
        raise ValueError("compiler repair binding has an invalid occurrence vocabulary")
    identities: list[tuple[str, str, str, int]] = []
    for occurrence in occurrences:
        if isinstance(occurrence, Mapping):
            values = (
                occurrence.get("line_start"),
                occurrence.get("line_end"),
                occurrence.get("source_text"),
                occurrence.get("occurrence_index"),
            )
        else:
            values = (
                occurrence.line_start,
                occurrence.line_end,
                occurrence.source_text,
                occurrence.occurrence_index,
            )
        line_start, line_end, source_text, occurrence_index = values
        if not (
            isinstance(line_start, str)
            and isinstance(line_end, str)
            and isinstance(source_text, str)
            and isinstance(occurrence_index, int)
            and not isinstance(occurrence_index, bool)
        ):
            raise ValueError("compiler repair binding occurrence is invalid")
        identities.append((line_start, line_end, source_text, occurrence_index))
    return tuple(
        "agent_binding_"
        + sha256_bytes(
            (
                logical_key
                + "\0"
                + line_start
                + "\0"
                + line_end
                + "\0"
                + source_text
                + "\0"
                + str(occurrence_index)
            ).encode("utf-8")
        )[:16]
        for line_start, line_end, source_text, occurrence_index in identities
    )


def _compiler_repair_error_intersects_scope(
    error: ValueError,
    payload: Mapping[str, Any],
    repair: _CompilerRepairOutput | None = None,
) -> bool:
    rendered = str(error)
    scope_references = _compiler_repair_scope_references(payload, repair)
    actionable_references = frozenset(_COMPILER_ACTIONABLE_REFERENCE.findall(rendered))
    if any(reference not in scope_references for reference in actionable_references):
        # The local transaction cannot edit every state element implicated by this rejection.
        # Let the outer host construct a fresh authoritative slice instead of paying for an
        # impossible retry against stale hidden state.
        return False
    for reference in scope_references:
        if reference.startswith("documentPatch."):
            # Structured paths are hierarchical. Editing an object or collection does not grant
            # authority over every descendant path merely because its spelling is a prefix.
            if re.search(re.escape(reference) + r"(?![.\[])", rendered) is not None:
                return True
        elif reference in rendered:
            return True
    return False


def _validate_compiler_repair_scope(
    value: _CompilerRepairOutput, payload: Mapping[str, Any]
) -> None:
    """Reject delta operations that reference facts absent from the local transaction."""

    allowed_paths = _compiler_repair_allowed_paths(payload)
    restored_replacements = tuple(restore_legacy_binding(row) for row in value.replacement_bindings)
    replacement_paths = {
        path for row in restored_replacements for path in (*row.target_paths, *row.dependency_paths)
    }
    replacement_paths.update(row.target_path for row in value.additional_semantic_only_target_facts)
    unknown_paths = sorted(replacement_paths - allowed_paths)
    if unknown_paths:
        raise ValueError(
            "compiler repair references target paths outside its local transaction: "
            + ", ".join(unknown_paths)
        )
    allowed_anchor_ids = set(_compiler_repair_allowed_anchor_ids(payload))
    unknown_anchor_ids = sorted(
        {row.anchor_binding_id for row in value.additional_anchor_overrides} - allowed_anchor_ids
    )
    if unknown_anchor_ids:
        raise ValueError(
            "compiler repair references anchor IDs outside its local transaction: "
            + ", ".join(unknown_anchor_ids)
        )
    candidate_slice = payload.get("candidateSlice")
    if (
        value.carrier_replacement is not None
        and isinstance(candidate_slice, Mapping)
        and candidate_slice.get("carrier") is None
    ):
        raise ValueError("compiler repair cannot replace a carrier outside its local transaction")


def _reference_occurrence_rows(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    table = payload.get("occurrenceCandidates")
    if not isinstance(table, Mapping):
        raise ValueError("reference payload lacks occurrenceCandidates")
    columns = table.get("columns")
    rows = table.get("rows")
    if not isinstance(columns, (tuple, list)) or tuple(columns) != _REFERENCE_OCCURRENCE_COLUMNS:
        raise ValueError("reference occurrence columns are invalid")
    if not isinstance(rows, (tuple, list)):
        raise ValueError("reference occurrence rows are invalid")
    output: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != len(columns):
            raise ValueError("reference occurrence row is invalid")
        materialized = dict(zip(columns, row, strict=True))
        if (
            not isinstance(materialized["occurrenceId"], str)
            or not isinstance(materialized["lineStart"], str)
            or not isinstance(materialized["lineEnd"], str)
            or not isinstance(materialized["sourceText"], str)
            or not isinstance(materialized["occurrenceIndex"], int)
            or isinstance(materialized["occurrenceIndex"], bool)
        ):
            raise ValueError("reference occurrence row has invalid field types")
        output.append(materialized)
    return tuple(output)


def _reference_occurrence_lookup(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in _reference_occurrence_rows(payload):
        occurrence_id = row.get("occurrenceId")
        if not isinstance(occurrence_id, str) or occurrence_id in lookup:
            raise ValueError("reference occurrence candidate IDs are invalid")
        lookup[occurrence_id] = {
            "line_start": row.get("lineStart"),
            "line_end": row.get("lineEnd"),
            "source_text": row.get("sourceText"),
            "occurrence_index": row.get("occurrenceIndex"),
        }
    return lookup


def _compact_inventory_occurrence_lookup(
    payload: Mapping[str, Any],
) -> dict[str, tuple[str, AgentOccurrence]]:
    """Materialize exact current-inventory occurrence handles and their logical owners."""

    contract = payload.get("compactContract")
    occurrence_rows = payload.get("occurrenceRows")
    binding_rows = payload.get("bindingRows")
    if not isinstance(contract, Mapping):
        raise ValueError("compact critic payload lacks its contract")
    if tuple(cast(Sequence[Any], contract.get("occurrenceColumns"))) != OCCURRENCE_COLUMNS:
        raise ValueError("compact critic occurrence columns are invalid")
    if tuple(cast(Sequence[Any], contract.get("bindingColumns"))) != BINDING_COLUMNS:
        raise ValueError("compact critic binding columns are invalid")
    if not isinstance(occurrence_rows, (tuple, list)) or not isinstance(
        binding_rows, (tuple, list)
    ):
        raise ValueError("compact critic inventory rows are invalid")

    occurrences: dict[str, AgentOccurrence] = {}
    for row in occurrence_rows:
        if not isinstance(row, (tuple, list)) or len(row) != len(OCCURRENCE_COLUMNS):
            raise ValueError("compact critic occurrence row is invalid")
        materialized = dict(zip(OCCURRENCE_COLUMNS, row, strict=True))
        occurrence_id = materialized["occurrenceId"]
        if not isinstance(occurrence_id, str) or occurrence_id in occurrences:
            raise ValueError("compact critic occurrence IDs are invalid")
        occurrences[occurrence_id] = AgentOccurrence(
            line_start=f"L{materialized['lineStartNumber']:05d}",
            line_end=f"L{materialized['lineEndNumber']:05d}",
            source_text=materialized["sourceText"],
            occurrence_index=materialized["occurrenceIndex"],
        )

    owners: dict[str, str] = {}
    for row in binding_rows:
        if not isinstance(row, (tuple, list)) or len(row) != len(BINDING_COLUMNS):
            raise ValueError("compact critic binding row is invalid")
        materialized = dict(zip(BINDING_COLUMNS, row, strict=True))
        logical_key = materialized["logicalKey"]
        occurrence_ids = materialized["occurrenceIds"]
        if not isinstance(logical_key, str) or not isinstance(occurrence_ids, (tuple, list)):
            raise ValueError("compact critic binding occurrence ownership is invalid")
        for occurrence_id in occurrence_ids:
            if occurrence_id not in occurrences or occurrence_id in owners:
                raise ValueError("compact critic occurrence ownership is invalid")
            owners[occurrence_id] = logical_key
    if set(owners) != set(occurrences):
        raise ValueError("compact critic inventory contains unowned occurrences")
    return {
        occurrence_id: (owners[occurrence_id], occurrence)
        for occurrence_id, occurrence in occurrences.items()
    }


def _critic_provider_payload(compact_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the semantic provider view while retaining host-only provenance separately."""

    if not isinstance(compact_payload.get("annotatedSource"), str) or not isinstance(
        compact_payload.get("maskedTemplate"), str
    ):
        raise ValueError("compact critic payload requires annotated and masked source views")
    contract = compact_payload.get("compactContract")
    binding_rows = compact_payload.get("bindingRows")
    occurrence_rows = compact_payload.get("occurrenceRows")
    if (
        not isinstance(contract, Mapping)
        or not isinstance(binding_rows, (tuple, list))
        or not isinstance(occurrence_rows, (tuple, list))
    ):
        raise ValueError("compact critic payload lacks provider table contracts")

    def project_rows(
        *, columns_value: Any, rows: Sequence[Any], omitted: frozenset[str]
    ) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...]]:
        if not isinstance(columns_value, (tuple, list)) or any(
            not isinstance(column, str) for column in columns_value
        ):
            raise ValueError("compact critic provider columns are invalid")
        columns = tuple(columns_value)
        kept_indexes = tuple(index for index, column in enumerate(columns) if column not in omitted)
        projected_rows: list[tuple[Any, ...]] = []
        for row in rows:
            if not isinstance(row, (tuple, list)) or len(row) != len(columns):
                raise ValueError("compact critic provider row does not match its columns")
            projected_rows.append(tuple(row[index] for index in kept_indexes))
        return tuple(columns[index] for index in kept_indexes), tuple(projected_rows)

    binding_columns, projected_bindings = project_rows(
        columns_value=contract.get("bindingColumns"),
        rows=cast(Sequence[Any], binding_rows),
        omitted=frozenset({"sourceBindingIds"}),
    )
    occurrence_columns, projected_occurrences = project_rows(
        columns_value=contract.get("occurrenceColumns"),
        rows=cast(Sequence[Any], occurrence_rows),
        omitted=frozenset({"sourceBindingId"}),
    )
    provider_contract = dict(contract)
    provider_contract["bindingColumns"] = binding_columns
    provider_contract["occurrenceColumns"] = occurrence_columns
    provider_contract.pop("sourceBindingIdEncoding", None)
    provider_payload = {
        key: value
        for key, value in compact_payload.items()
        if key
        not in {
            "allowedRemovalLogicalKeys",
            "maskedTemplate",
            "sourceBindingIdTable",
        }
    }
    provider_payload["compactContract"] = provider_contract
    provider_payload["bindingRows"] = projected_bindings
    provider_payload["occurrenceRows"] = projected_occurrences
    return provider_payload


def _critic_provider_request_bytes(compact_payload: Mapping[str, Any]) -> int:
    """Measure the exact UTF-8 request body used by the compact critic call."""

    return len(
        json.dumps(
            _critic_provider_payload(compact_payload),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _uses_partitioned_critic(
    *,
    protocol: AgentContractProtocol,
    compact_payload: Mapping[str, Any],
    threshold_bytes: int | None,
) -> bool:
    """Route only measured oversized hybrid requests through concurrent audit facets."""

    if protocol in _STAGED_PROTOCOLS:
        return True
    if protocol not in _HYBRID_PROTOCOLS:
        if threshold_bytes is not None:
            raise ValueError("critic partition threshold is invalid for a non-hybrid protocol")
        return False
    if threshold_bytes is None or threshold_bytes <= 0:
        raise ValueError("hybrid critic routing requires a positive request-byte threshold")
    return _critic_provider_request_bytes(compact_payload) > threshold_bytes


def _materialize_reference_occurrences(
    rows: Any, lookup: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    if not isinstance(rows, (tuple, list)):
        raise ValueError("reference occurrence list is invalid")
    restored: list[dict[str, Any]] = []
    for occurrence in rows:
        if isinstance(occurrence, BaseModel):
            occurrence = occurrence.model_dump(mode="python")
        if not isinstance(occurrence, Mapping):
            raise ValueError("reference occurrence is not an object")
        if set(occurrence) == {"occurrence_id"}:
            occurrence_id = occurrence["occurrence_id"]
            if occurrence_id not in lookup:
                raise ValueError("provider returned an unknown occurrence ID")
            restored.append(dict(lookup[cast(str, occurrence_id)]))
        else:
            restored.append(dict(occurrence))
    return tuple(restored)


class _CriticLogicalKeyOutput(BaseModel):
    """Provider-facing critic result; host artifacts retain canonical edit-handle receipts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    verdict: Literal["pass", "revise"]
    findings: tuple[CriticFinding, ...]
    remove_binding_logical_keys: tuple[str, ...] = ()
    additional_bindings: tuple[AgentBindingProposal, ...]
    semantic_only_target_facts: tuple[SemanticOnlyTargetFactProposal, ...] = ()
    coherence_decisions: tuple[CoherenceCandidateDecision, ...] = ()
    rationale: NonEmptyText

    @model_validator(mode="after")
    def verdict_is_consistent(self) -> _CriticLogicalKeyOutput:
        if self.verdict == "pass" and (
            self.findings
            or self.remove_binding_logical_keys
            or self.additional_bindings
            or self.semantic_only_target_facts
        ):
            raise ValueError("pass requires zero findings and zero structural patch operations")
        if self.verdict == "revise" and not self.findings:
            raise ValueError("revise requires at least one finding")
        if len(set(self.remove_binding_logical_keys)) != len(self.remove_binding_logical_keys):
            raise ValueError("critic removal logical keys must be unique")
        decision_ids = tuple(row.candidate_id for row in self.coherence_decisions)
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("critic coherence decision candidate IDs must be unique")
        return self


def _scoped_critic_output_type(
    allowed_removal_logical_keys: Sequence[str],
) -> type[_CriticLogicalKeyOutput]:
    """Constrain removals to exact semantic keys from the current host inventory.

    Opaque hash handles can be valid yet semantically unrelated to the cited finding. Logical keys
    make the selected binding self-describing, while an exact JSON Schema enum still prevents
    invented or mistyped edits at the provider boundary.
    """

    allowed = tuple(allowed_removal_logical_keys)
    if not allowed:
        raise ValueError("critic request has no allowed removal logical keys")
    if len(set(allowed)) != len(allowed):
        raise ValueError("critic request contains duplicate removal logical keys")
    logical_key = Literal.__getitem__(allowed)
    name = "ScopedCriticAgentOutput_" + sha256_bytes(canonical_json_bytes(allowed))[:16]
    model = create_model(
        name,
        __base__=_CriticLogicalKeyOutput,
        __module__=__name__,
        remove_binding_logical_keys=(tuple[logical_key, ...], ()),  # type: ignore[valid-type]
    )
    return model


def _critic_output_with_inventory_ids(
    output: _CriticLogicalKeyOutput,
) -> CriticAgentOutput:
    from .host import inventory_binding_id

    return CriticAgentOutput.model_validate(
        {
            "verdict": output.verdict,
            "findings": output.findings,
            "remove_inventory_binding_ids": tuple(
                inventory_binding_id(logical_key)
                for logical_key in output.remove_binding_logical_keys
            ),
            "additional_bindings": output.additional_bindings,
            "semantic_only_target_facts": output.semantic_only_target_facts,
            "coherence_decisions": output.coherence_decisions,
            "rationale": output.rationale,
        }
    )


def _scoped_compact_critic_output_type(
    allowed_removal_logical_keys: Sequence[str],
    review_candidate_ids: Sequence[str],
    remaining_risk_ids: Sequence[str],
    literal_line_ids: Sequence[str],
    occurrence_ids: Sequence[str] = (),
    inventory_occurrence_owners: Mapping[str, str] | None = None,
    *,
    required_review_candidate_ids: Sequence[str] = (),
    coherence_candidate_ids: Sequence[str] = (),
) -> type[BaseModel]:
    """Build a compact verdict whose validator enforces the document-specific vocabulary.

    Encoding every allowed line, candidate, logical key, and occurrence as JSON-Schema enums made
    the response schema document-specific and duplicated data already present in the request.
    The stable field types below retain strict structured output; model validators enforce the
    same exact-set, membership, ownership, and transaction rules and therefore participate in the
    configured PydanticAI retry contract.
    """

    allowed = tuple(allowed_removal_logical_keys)
    if not allowed:
        raise ValueError("critic request has no allowed removal logical keys")
    if len(set(allowed)) != len(allowed):
        raise ValueError("critic request contains duplicate removal logical keys")
    expected_candidates = tuple(review_candidate_ids)
    required_review_candidates = tuple(required_review_candidate_ids)
    expected_coherence_decisions = tuple(coherence_candidate_ids)
    required_risks = tuple(remaining_risk_ids)
    if len(set(expected_candidates)) != len(expected_candidates):
        raise ValueError("critic request contains duplicate review candidate IDs")
    if len(set(required_risks)) != len(required_risks):
        raise ValueError("critic request contains duplicate remaining risk IDs")
    if len(set(required_review_candidates)) != len(required_review_candidates) or not set(
        required_review_candidates
    ) <= set(expected_candidates):
        raise ValueError("critic required-review candidates are not a unique request subset")
    if len(set(expected_coherence_decisions)) != len(expected_coherence_decisions) or not set(
        expected_coherence_decisions
    ) <= set(expected_candidates):
        raise ValueError("critic coherence candidates are not a unique request subset")
    if set(required_review_candidates) & set(expected_coherence_decisions):
        raise ValueError("critic structural and coherence candidate sets overlap")
    if set(expected_candidates).intersection(required_risks):
        raise ValueError("critic review candidate IDs overlap remaining risk IDs")
    allowed_occurrences = tuple(occurrence_ids)
    if len(set(allowed_occurrences)) != len(allowed_occurrences):
        raise ValueError("critic request contains duplicate occurrence IDs")
    removable_occurrence_owners = dict(inventory_occurrence_owners or {})
    if any(owner not in allowed for owner in removable_occurrence_owners.values()):
        raise ValueError("critic removable occurrence has an unknown logical owner")
    expected_lines = tuple(literal_line_ids)
    if not expected_lines:
        raise ValueError("critic request contains no literal line IDs")
    if len(set(expected_lines)) != len(expected_lines):
        raise ValueError("critic request contains duplicate literal line IDs")
    all_candidates = (*expected_candidates, *required_risks)
    digest = sha256_bytes(
        canonical_json_bytes(
            (
                allowed,
                expected_candidates,
                required_review_candidates,
                expected_coherence_decisions,
                required_risks,
                expected_lines,
                allowed_occurrences,
                removable_occurrence_owners,
            )
        )
    )[:16]
    candidate_receipt_type = Annotated[
        tuple[Literal["valid_existing_contract", "defect_requires_revision"], ...],
        Field(
            min_length=len(all_candidates),
            max_length=len(all_candidates),
            description=(
                "One disposition per reviewCandidates item followed by every "
                "remainingRiskCandidates item, in exact request order."
            ),
        ),
    ]
    literal_line_count_type = Annotated[
        int,
        Field(
            ge=len(expected_lines),
            le=len(expected_lines),
            description=("Exact number of literalLineReviewIds inspected before this decision."),
        ),
    ]
    coverage_type = create_model(
        f"ScopedCriticAuditCoverage_{digest}",
        __module__=__name__,
        __config__=ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False),
        literal_completeness_checked=(Literal[True], ...),
        target_ownership_checked=(Literal[True], ...),
        topology_and_grouping_checked=(Literal[True], ...),
        derivations_checked=(Literal[True], ...),
        carrier_boundary_checked=(Literal[True], ...),
        identifier_relationships_checked=(Literal[True], ...),
        candidate_receipt=(candidate_receipt_type, ...),
        literal_line_count=(literal_line_count_type, ...),
    )

    @model_validator(mode="after")
    def coverage_is_exhaustive(value: Any) -> Any:
        candidate_receipt = value.coverage.candidate_receipt
        if len(candidate_receipt) != len(all_candidates):
            raise ValueError(
                "critic candidate receipt length differs from the exact request vocabulary; "
                f"expected={len(all_candidates)}; actual={len(candidate_receipt)}"
            )
        decision_ids = tuple(row.candidate_id for row in value.coherence_decisions)
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("critic coherence decision candidate IDs must be unique")
        if set(decision_ids) != set(expected_coherence_decisions):
            missing = sorted(set(expected_coherence_decisions) - set(decision_ids))
            unexpected = sorted(set(decision_ids) - set(expected_coherence_decisions))
            raise ValueError(
                "critic coherence decisions differ from the exact required vocabulary; "
                f"missing={missing}; unexpected={unexpected}"
            )
        literal_line_count = value.coverage.literal_line_count
        if literal_line_count != len(expected_lines):
            raise ValueError(
                "critic literal-line count differs from the exact request vocabulary; "
                f"expected={len(expected_lines)}; actual={literal_line_count}"
            )
        if value.verdict == "pass" and any(
            disposition == "defect_requires_revision" for disposition in candidate_receipt
        ):
            raise ValueError("pass cannot retain a candidate defect")
        return value

    coherence_fields: dict[str, Any]
    if expected_coherence_decisions:
        coherence_candidate_id = Literal.__getitem__(expected_coherence_decisions)
        coherence_decision_type = create_model(
            f"ScopedCoherenceCandidateDecision_{digest}",
            __base__=CoherenceCandidateDecision,
            __module__=__name__,
            candidate_id=(coherence_candidate_id, ...),
        )
        coherence_fields = {
            "coherence_decisions": (
                Annotated[
                    tuple[coherence_decision_type, ...],  # type: ignore[valid-type]
                    Field(
                        min_length=len(expected_coherence_decisions),
                        max_length=len(expected_coherence_decisions),
                    ),
                ],
                ...,
            )
        }
    else:
        coherence_fields = {
            "coherence_decisions": (
                Annotated[tuple[CoherenceCandidateDecision, ...], Field(max_length=0)],
                (),
            )
        }

    pass_type = create_model(
        f"ScopedCompactCriticPass_{digest}",
        __base__=CompactCriticPassOutput,
        __module__=__name__,
        __validators__=cast(
            dict[str, Callable[..., Any]],
            {"coverage_is_exhaustive": coverage_is_exhaustive},
        ),
        coverage=(coverage_type, ...),
        **coherence_fields,
    )
    binding_type: Any = DiscriminatedBindingProposal
    append_fields: dict[str, Any] = {}
    if allowed_occurrences:
        allowed_occurrence_id = Literal.__getitem__(allowed_occurrences)
        occurrence_reference_type = create_model(
            f"ScopedCriticOccurrenceReference_{digest}",
            __base__=CompilerOccurrenceReference,
            __module__=__name__,
            occurrence_id=(allowed_occurrence_id, ...),
        )
        binding_type = create_model(
            f"ScopedCriticBindingProposal_{digest}",
            __base__=DiscriminatedBindingProposal,
            __module__=__name__,
            occurrences=(
                Annotated[
                    tuple[occurrence_reference_type | AgentOccurrence, ...],  # type: ignore[valid-type]
                    Field(min_length=1),
                ],
                ...,
            ),
        )
        append_fields["occurrences"] = (
            Annotated[
                tuple[occurrence_reference_type | AgentOccurrence, ...],  # type: ignore[valid-type]
                Field(min_length=1),
            ],
            ...,
        )
    append_type = create_model(
        f"ScopedCriticOccurrenceAppend_{digest}",
        __base__=_CriticOccurrenceAppend,
        __module__=__name__,
        **append_fields,
    )
    removal_type: Any = _CriticOccurrenceRemovalReference

    @model_validator(mode="after")
    def critic_operations_are_unambiguous(value: Any) -> Any:
        append_keys = tuple(row.logical_key for row in value.occurrence_appends)
        occurrence_removal_keys = tuple(row.logical_key for row in value.occurrence_removals)
        replacement_keys = tuple(row.logical_key for row in value.additional_bindings)
        referenced_keys = (
            *value.remove_binding_logical_keys,
            *append_keys,
            *occurrence_removal_keys,
        )
        unknown_keys = sorted(set(referenced_keys) - set(allowed))
        if unknown_keys:
            raise ValueError(
                "critic mutation references unknown logical keys: " + ", ".join(unknown_keys)
            )
        if len(set(append_keys)) != len(append_keys):
            raise ValueError("critic occurrence-append logical keys must be unique")
        if len(set(occurrence_removal_keys)) != len(occurrence_removal_keys):
            raise ValueError("critic occurrence-removal logical keys must be unique")
        mutation_keys = set(value.remove_binding_logical_keys) | set(replacement_keys)
        collisions = (set(append_keys) | set(occurrence_removal_keys)) & mutation_keys
        if collisions:
            raise ValueError(
                "critic occurrence appends cannot also remove or replace the same binding: "
                + ", ".join(sorted(collisions))
            )
        for binding in value.additional_bindings:
            for occurrence in binding.occurrences:
                if (
                    isinstance(occurrence, CompilerOccurrenceReference)
                    and occurrence.occurrence_id not in allowed_occurrences
                ):
                    raise ValueError(
                        "critic binding references unknown occurrence ID: "
                        + occurrence.occurrence_id
                    )
        for append in value.occurrence_appends:
            for occurrence in append.occurrences:
                if (
                    isinstance(occurrence, CompilerOccurrenceReference)
                    and occurrence.occurrence_id not in allowed_occurrences
                ):
                    raise ValueError(
                        "critic append references unknown occurrence ID: "
                        + occurrence.occurrence_id
                    )
        for removal in value.occurrence_removals:
            occurrence_ids_value = tuple(removal.occurrence_ids)
            if len(set(occurrence_ids_value)) != len(occurrence_ids_value):
                raise ValueError("critic occurrence-removal IDs must be unique")
            unknown_occurrences = sorted(
                set(occurrence_ids_value) - set(removable_occurrence_owners)
            )
            if unknown_occurrences:
                raise ValueError(
                    "critic occurrence removal references unknown IDs: "
                    + ", ".join(unknown_occurrences)
                )
            wrong_owner = tuple(
                occurrence_id
                for occurrence_id in occurrence_ids_value
                if removable_occurrence_owners[occurrence_id] != removal.logical_key
            )
            if wrong_owner:
                raise ValueError(
                    "critic occurrence removals must belong to the declared logical key: "
                    + ", ".join(wrong_owner)
                )
            owned = {
                occurrence_id
                for occurrence_id, owner in removable_occurrence_owners.items()
                if owner == removal.logical_key
            }
            if set(occurrence_ids_value) == owned:
                raise ValueError(
                    "critic occurrence removal cannot empty a binding; use full removal"
                )
        return value

    revision_type = create_model(
        f"ScopedCompactCriticRevision_{digest}",
        __base__=CompactCriticRevisionOutput,
        __module__=__name__,
        __validators__=cast(
            dict[str, Callable[..., Any]],
            {
                "coverage_is_exhaustive": coverage_is_exhaustive,
                "critic_operations_are_unambiguous": critic_operations_are_unambiguous,
            },
        ),
        coverage=(coverage_type, ...),
        remove_binding_logical_keys=(tuple[NonEmptyText, ...], ()),
        additional_bindings=(tuple[binding_type, ...], ()),
        occurrence_appends=(tuple[append_type, ...], ()),  # type: ignore[valid-type]
        occurrence_removals=(tuple[removal_type, ...], ()),
        **coherence_fields,
    )
    decision_type = Annotated[
        pass_type | revision_type,  # type: ignore[valid-type]
        Field(discriminator="verdict"),
    ]
    envelope = create_model(
        f"ScopedCompactCriticEnvelope_{digest}",
        __module__=__name__,
        __config__=ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False),
        decision=(decision_type, ...),
    )
    return envelope


def _scoped_reference_compiler_output_type(
    occurrence_ids: Sequence[str],
) -> type[BaseModel]:
    """Build a compiler schema whose span references are limited to the request table."""

    allowed = tuple(occurrence_ids)
    if len(set(allowed)) != len(allowed):
        raise ValueError("compiler request contains duplicate occurrence IDs")
    digest = sha256_bytes(canonical_json_bytes(allowed))[:16]
    if allowed:
        occurrence_id = Literal.__getitem__(allowed)
        reference_type = create_model(
            f"ScopedCompilerOccurrenceReference_{digest}",
            __base__=CompilerOccurrenceReference,
            __module__=__name__,
            occurrence_id=(occurrence_id, ...),
        )
        occurrence_type: Any = reference_type | AgentOccurrence
    else:
        occurrence_type = AgentOccurrence
    occurrences_type = Annotated[
        tuple[occurrence_type, ...],
        Field(min_length=1),
    ]
    binding_type = create_model(
        f"ScopedReferenceBindingProposal_{digest}",
        __base__=DiscriminatedBindingProposal,
        __module__=__name__,
        occurrences=(occurrences_type, ...),
    )
    carrier_type = create_model(
        f"ScopedReferenceCarrierAssessment_{digest}",
        __base__=CarrierAssessment,
        __module__=__name__,
        evidence_occurrences=(occurrences_type, ...),
    )
    output_type = create_model(
        f"ScopedReferenceCompilerOutput_{digest}",
        __base__=DiscriminatedCompilerAgentOutput,
        __module__=__name__,
        carrier=(carrier_type, ...),
        bindings=(tuple[binding_type, ...], ...),  # type: ignore[valid-type]
    )
    return output_type


def _scoped_staged_plan_output_type(
    plan_payload: Mapping[str, Any],
) -> type[StagedCriticPlanOutput]:
    """Restrict every staged-plan reference to exact handles in this repair slice."""

    rows = plan_payload.get("occurrenceCandidates")
    if not isinstance(rows, (tuple, list)) or any(
        not isinstance(row, Mapping) or not isinstance(row.get("occurrenceId"), str) for row in rows
    ):
        raise ValueError("staged plan lacks a valid occurrence-candidate vocabulary")
    allowed = tuple(cast(str, row["occurrenceId"]) for row in rows)
    if len(set(allowed)) != len(allowed):
        raise ValueError("staged plan contains duplicate occurrence-candidate IDs")
    candidate_rows = plan_payload.get("candidateRows")
    if not isinstance(candidate_rows, (tuple, list)) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("candidateId"), str)
        or not isinstance(row.get("kind"), str)
        for row in candidate_rows
    ):
        raise ValueError("staged plan lacks a valid review-candidate vocabulary")
    all_required_coherence_ids = tuple(
        cast(str, row["candidateId"])
        for row in candidate_rows
        if row.get("kind") == "cross_field_semantic_relation"
        and row.get("requiredRevision") is True
    )
    if len(set(all_required_coherence_ids)) != len(all_required_coherence_ids):
        raise ValueError("staged plan contains duplicate coherence-candidate IDs")
    raw_findings = plan_payload.get("findings", ())
    if not isinstance(raw_findings, (tuple, list)) or any(
        not isinstance(row, Mapping) for row in raw_findings
    ):
        raise ValueError("staged plan lacks a valid audit-finding sequence")
    host_decisions = (
        host_completable_coherence_decisions(
            findings=cast(Sequence[Mapping[str, Any]], raw_findings),
            candidate_rows=cast(Sequence[Mapping[str, Any]], candidate_rows),
        )
        if raw_findings
        else ()
    )
    host_decisions_by_id = {row.candidate_id: row for row in host_decisions}
    provider_required_coherence_ids = tuple(
        candidate_id
        for candidate_id in all_required_coherence_ids
        if candidate_id not in host_decisions_by_id
    )
    digest = sha256_bytes(
        canonical_json_bytes((allowed, all_required_coherence_ids, tuple(host_decisions_by_id)))
    )[:16]
    if allowed:
        occurrence_id = Literal.__getitem__(allowed)
        reference_type = create_model(
            f"ScopedStagedPlanOccurrenceReference_{digest}",
            __base__=CompilerOccurrenceReference,
            __module__=__name__,
            occurrence_id=(occurrence_id, ...),
        )
        occurrence_type: Any = reference_type | AgentOccurrence
    else:
        occurrence_type = AgentOccurrence
    occurrences_type = Annotated[
        tuple[occurrence_type, ...],
        Field(min_length=1),
    ]
    binding_type = create_model(
        f"ScopedStagedPlanBindingProposal_{digest}",
        __base__=StagedPlanBindingProposal,
        __module__=__name__,
        occurrences=(occurrences_type, ...),
    )
    append_type = create_model(
        f"ScopedStagedPlanOccurrenceAppend_{digest}",
        __base__=StagedPlanOccurrenceAppend,
        __module__=__name__,
        occurrences=(occurrences_type, ...),
    )
    validators: dict[str, Callable[..., Any]] = {}
    if all_required_coherence_ids:
        coherence_candidate_id = Literal.__getitem__(all_required_coherence_ids)
        coherence_decision_type = create_model(
            f"ScopedStagedCoherenceCandidateDecision_{digest}",
            __base__=CoherenceCandidateDecision,
            __module__=__name__,
            candidate_id=(coherence_candidate_id, ...),
        )
        coherence_field: Any = (
            Annotated[
                tuple[coherence_decision_type, ...],  # type: ignore[valid-type]
                Field(
                    min_length=len(provider_required_coherence_ids),
                    max_length=len(all_required_coherence_ids),
                ),
            ],
            ... if provider_required_coherence_ids else host_decisions,
        )

        @field_validator("coherence_decisions", mode="after")  # type: ignore[misc]
        @classmethod
        def materialize_host_coherence_decisions(
            cls: type[BaseModel], value: tuple[CoherenceCandidateDecision, ...]
        ) -> tuple[CoherenceCandidateDecision, ...]:
            del cls
            normalized: list[CoherenceCandidateDecision] = []
            supplied_ids: set[str] = set()
            for row in value:
                supplied_ids.add(row.candidate_id)
                expected = host_decisions_by_id.get(row.candidate_id)
                if expected is None:
                    normalized.append(row)
                    continue
                if (
                    row.disposition,
                    row.suggestion_index,
                ) != (
                    expected.disposition,
                    expected.suggestion_index,
                ):
                    raise ValueError(
                        "critic plan contradicts audit-proved coherence decision: "
                        + row.candidate_id
                    )
                # Rationale is explanatory, not part of the deterministic host decision. Store
                # the canonical host receipt so semantically identical provider prose cannot
                # make an otherwise valid transaction non-reproducible.
                normalized.append(expected)
            additions = tuple(
                decision
                for candidate_id, decision in host_decisions_by_id.items()
                if candidate_id not in supplied_ids
            )
            if not additions:
                return tuple(normalized)
            return (*normalized, *additions)

        @model_validator(mode="after")
        def coherence_decisions_cover_provider_work(
            value: StagedCriticPlanOutput,
        ) -> StagedCriticPlanOutput:
            decisions = {row.candidate_id: row for row in value.coherence_decisions}
            missing = set(provider_required_coherence_ids) - set(decisions)
            if missing:
                raise ValueError(
                    "critic plan omits ambiguous coherence decisions: " + ", ".join(sorted(missing))
                )
            invalid_host_choices = tuple(
                candidate_id
                for candidate_id, expected in host_decisions_by_id.items()
                if candidate_id in decisions
                and decisions[candidate_id].model_dump(mode="json")
                != expected.model_dump(mode="json")
            )
            if invalid_host_choices:
                raise ValueError(
                    "critic plan contradicts audit-proved coherence decisions: "
                    + ", ".join(invalid_host_choices)
                )
            return value

        validators["materialize_host_coherence_decisions"] = materialize_host_coherence_decisions
        validators["coherence_decisions_cover_provider_work"] = cast(
            Callable[..., Any], coherence_decisions_cover_provider_work
        )
    else:
        coherence_field = (
            Annotated[tuple[CoherenceCandidateDecision, ...], Field(max_length=0)],
            (),
        )
    return cast(
        type[StagedCriticPlanOutput],
        create_model(
            f"ScopedStagedCriticPlanOutput_{digest}",
            __base__=StagedCriticPlanOutput,
            __module__=__name__,
            __validators__=validators,
            additional_bindings=(tuple[binding_type, ...], ()),  # type: ignore[valid-type]
            occurrence_appends=(tuple[append_type, ...], ()),  # type: ignore[valid-type]
            coherence_decisions=coherence_field,
        ),
    )


@lru_cache(maxsize=1)
def _fixed_reference_compiler_output_type() -> type[BaseModel]:
    """Return a provider-stable compiler schema with host-validated occurrence handles.

    The candidate table in the request is already the authoritative vocabulary. Repeating every
    document-specific handle as a JSON-Schema enum enlarges the schema and changes its prefix for
    every document without adding host integrity: `_restore_reference_compiler` performs exact
    membership validation before any candidate is accepted.
    """

    occurrence_type: Any = CompilerOccurrenceReference | AgentOccurrence
    occurrences_type = Annotated[
        tuple[occurrence_type, ...],
        Field(min_length=1),
    ]
    binding_type = create_model(
        "FixedReferenceBindingProposal",
        __base__=DiscriminatedBindingProposal,
        __module__=__name__,
        occurrences=(occurrences_type, ...),
    )
    carrier_type = create_model(
        "FixedReferenceCarrierAssessment",
        __base__=CarrierAssessment,
        __module__=__name__,
        evidence_occurrences=(occurrences_type, ...),
    )
    output_type = create_model(
        "FixedReferenceCompilerOutput",
        __base__=DiscriminatedCompilerAgentOutput,
        __module__=__name__,
        carrier=(carrier_type, ...),
        bindings=(tuple[binding_type, ...], ...),  # type: ignore[valid-type]
    )
    return output_type


@lru_cache(maxsize=128)
def _scoped_initial_compiler_output_type(
    allowed_anchor_override_ids: tuple[str, ...],
    *,
    required_anchor_override_ids: tuple[str, ...] = (),
    cross_fact_review_ids: tuple[str, ...] = (),
) -> type[BaseModel]:
    """Constrain first-pass anchor edits and require exhaustive topology dispositions."""

    if len(set(allowed_anchor_override_ids)) != len(allowed_anchor_override_ids):
        raise ValueError("initial compiler override authority contains duplicate IDs")
    if any(
        re.fullmatch(r"anchor_binding_[0-9]{4}", value) is None
        for value in allowed_anchor_override_ids
    ):
        raise ValueError("initial compiler override authority contains an invalid ID")
    for name, values in (
        ("required override", required_anchor_override_ids),
        ("cross-fact review", cross_fact_review_ids),
    ):
        if len(set(values)) != len(values) or not set(values) <= set(allowed_anchor_override_ids):
            raise ValueError(f"initial compiler {name} scope is invalid")
    if set(required_anchor_override_ids) & set(cross_fact_review_ids):
        raise ValueError("initial compiler topology scopes overlap")
    digest = sha256_bytes(
        canonical_json_bytes(
            (
                allowed_anchor_override_ids,
                required_anchor_override_ids,
                cross_fact_review_ids,
            )
        )
    )[:16]
    if allowed_anchor_override_ids:
        override_id = Literal.__getitem__(allowed_anchor_override_ids)
        override_type = create_model(
            f"ScopedInitialAnchorOverride_{digest}",
            __base__=AnchorOverride,
            __module__=__name__,
            anchor_binding_id=(override_id, ...),
        )
        overrides_annotation: Any = tuple[override_type, ...]  # type: ignore[valid-type]
    else:
        overrides_annotation = tuple[()]

    fields: dict[str, Any] = {"anchor_overrides": (overrides_annotation, ())}
    validators: dict[str, Callable[..., Any]] = {}
    if required_anchor_override_ids or cross_fact_review_ids:
        retained_annotation: Any
        if cross_fact_review_ids:
            retained_id = Literal.__getitem__(cross_fact_review_ids)
            retained_annotation = Annotated[
                tuple[retained_id, ...],  # type: ignore[valid-type]
                Field(json_schema_extra={"uniqueItems": True}),
            ]
        else:
            retained_annotation = Annotated[tuple[str, ...], Field(max_length=0)]
        fields["retained_cross_fact_anchor_ids"] = (retained_annotation, ...)

        @model_validator(mode="after")
        def topology_review_is_exhaustive(value: BaseModel) -> BaseModel:
            dynamic_value = cast(Any, value)
            override_ids = {
                row.anchor_binding_id
                for row in cast(Sequence[AnchorOverride], dynamic_value.anchor_overrides)
            }
            retained_ids = set(cast(Sequence[str], dynamic_value.retained_cross_fact_anchor_ids))
            missing_required = sorted(set(required_anchor_override_ids) - override_ids)
            if missing_required:
                raise ValueError(
                    "initial compiler omitted required anchor overrides: "
                    + ", ".join(missing_required)
                )
            reviewed_cross_fact = (override_ids & set(cross_fact_review_ids)) | retained_ids
            if reviewed_cross_fact != set(cross_fact_review_ids):
                missing = sorted(set(cross_fact_review_ids) - reviewed_cross_fact)
                raise ValueError(
                    "initial compiler omitted cross-fact anchor dispositions: " + ", ".join(missing)
                )
            overlap = sorted((override_ids & set(cross_fact_review_ids)) & retained_ids)
            if overlap:
                raise ValueError(
                    "initial compiler both overrides and retains cross-fact anchors: "
                    + ", ".join(overlap)
                )
            return value

        validators["topology_review_is_exhaustive"] = cast(
            Callable[..., Any], topology_review_is_exhaustive
        )
    return cast(
        type[BaseModel],
        create_model(
            f"ScopedInitialCompilerOutput_{digest}",
            __base__=_fixed_reference_compiler_output_type(),
            __module__=__name__,
            __validators__=validators,
            **fields,
        ),
    )


def _restore_reference_compiler(
    output: BaseModel,
    payload: Mapping[str, Any],
) -> CompilerAgentOutput:
    """Materialize provider span handles and restore the stable compiler contract."""

    lookup = _reference_occurrence_lookup(payload)
    candidate = output.model_dump(mode="python")
    candidate.pop("retained_cross_fact_anchor_ids", None)
    carrier = candidate.get("carrier")
    bindings = candidate.get("bindings")
    if not isinstance(carrier, dict) or not isinstance(bindings, (tuple, list)):
        raise ValueError("reference compiler output has an invalid root shape")
    carrier["evidence_occurrences"] = _materialize_reference_occurrences(
        carrier.get("evidence_occurrences"), lookup
    )
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("reference compiler binding is not an object")
        binding["occurrences"] = _materialize_reference_occurrences(
            binding.get("occurrences"), lookup
        )
    discriminated = DiscriminatedCompilerAgentOutput.model_validate(candidate)
    return restore_legacy_compiler(discriminated)


def _restore_reference_compact_critic(
    output: BaseModel,
    compact_payload: Mapping[str, Any],
    source_payload: Mapping[str, Any],
) -> CriticAgentOutput:
    """Materialize scoped critic span handles into the stable host artifact."""

    from .host import inventory_binding_id

    decision = cast(Any, output).decision
    if decision.verdict == "pass":
        return CriticAgentOutput(
            verdict="pass",
            findings=(),
            remove_inventory_binding_ids=(),
            additional_bindings=(),
            semantic_only_target_facts=(),
            coherence_decisions=decision.coherence_decisions,
            exhaustive_audit_receipt=True,
            rationale=decision.rationale,
        )
    lookup = _reference_occurrence_lookup(compact_payload)
    restored_bindings: list[AgentBindingProposal] = []
    for binding in decision.additional_bindings:
        candidate = binding.model_dump(mode="python")
        candidate["occurrences"] = _materialize_reference_occurrences(
            candidate.get("occurrences"), lookup
        )
        restored_bindings.append(
            restore_legacy_binding(DiscriminatedBindingProposal.model_validate(candidate))
        )
    inventory = source_payload.get("bindingInventory")
    if not isinstance(inventory, (tuple, list)):
        raise ValueError("critic source payload lacks bindingInventory")
    inventory_by_key: dict[str, Mapping[str, Any]] = {}
    for row in inventory:
        if not isinstance(row, Mapping) or not isinstance(row.get("logicalKey"), str):
            raise ValueError("critic source binding inventory is invalid")
        logical_key = cast(str, row["logicalKey"])
        if logical_key in inventory_by_key:
            raise ValueError("critic source binding inventory has duplicate logical keys")
        inventory_by_key[logical_key] = row
    for append in decision.occurrence_appends:
        row = inventory_by_key.get(append.logical_key)
        if row is None:
            raise ValueError("critic occurrence append references an unknown logical key")
        restored_bindings.append(
            AgentBindingProposal.model_validate(
                {
                    "logical_key": append.logical_key,
                    "render_mode": row.get("renderMode"),
                    "value_kind": row.get("valueKind"),
                    "group_kind": row.get("groupKind"),
                    "group_key": row.get("groupKey"),
                    "target_paths": tuple(cast(Sequence[str], row.get("targetPaths"))),
                    "derivation": row.get("derivation"),
                    "dependency_paths": tuple(cast(Sequence[str], row.get("dependencyPaths"))),
                    "dependency_bindings": tuple(
                        cast(Sequence[str], row.get("dependencyBindings"))
                    ),
                    "occurrences": _materialize_reference_occurrences(append.occurrences, lookup),
                    "rationale": append.rationale,
                }
            )
        )
    occurrence_removals: list[CriticOccurrenceRemoval] = []
    if decision.occurrence_removals:
        inventory_occurrences = _compact_inventory_occurrence_lookup(compact_payload)
        for removal in decision.occurrence_removals:
            materialized: list[AgentOccurrence] = []
            for occurrence_id in removal.occurrence_ids:
                owner, occurrence = inventory_occurrences[occurrence_id]
                if owner != removal.logical_key:
                    raise ValueError("critic occurrence removal has a mismatched logical owner")
                materialized.append(occurrence)
            occurrence_removals.append(
                CriticOccurrenceRemoval(
                    logical_key=removal.logical_key,
                    occurrences=tuple(materialized),
                    rationale=removal.rationale,
                )
            )
    return CriticAgentOutput(
        verdict="revise",
        findings=decision.findings,
        remove_inventory_binding_ids=tuple(
            inventory_binding_id(key) for key in decision.remove_binding_logical_keys
        ),
        occurrence_removals=tuple(occurrence_removals),
        additional_bindings=tuple(restored_bindings),
        semantic_only_target_facts=decision.semantic_only_target_facts,
        coherence_decisions=decision.coherence_decisions,
        exhaustive_audit_receipt=True,
        rationale=decision.rationale,
    )


def empty_usage() -> LinguisticUsageReceipt:
    return LinguisticUsageReceipt.model_validate(
        {
            "requests": 0,
            "providerResponseIds": (),
            "finishReasons": (),
            "inputTokens": 0,
            "cacheReadTokens": 0,
            "cacheWriteTokens": 0,
            "outputTokens": 0,
            "reasoningTokens": 0,
            "visibleOutputTokens": 0,
            "providerTokenAccountingAnomaly": False,
            "estimatedCostUsd": Decimal(0),
            "providerReportedCostUsd": None,
            "downstreamProviders": (),
        }
    )


def _structured_output_retry_message(error: ValueError) -> str:
    """Tell the provider that a validator retry replaces, rather than amends, its output."""

    return (
        "Host rejected structured output: "
        f"{error}\nThe next structured response replaces the rejected response in full. "
        "Repeat every still-valid field and operation from it, apply the named correction, and "
        "return one complete response; do not return only an incremental correction."
    )


def _combined_usage(stages: Sequence[AgentStageArtifact]) -> LinguisticUsageReceipt:
    receipts = tuple(stage.usage for stage in stages)
    reported = tuple(receipt.providerReportedCostUsd for receipt in receipts)
    if any(value is None for value in reported) and any(value is not None for value in reported):
        raise ValueError("substage provider-cost receipts have inconsistent coverage")
    return LinguisticUsageReceipt(
        requests=sum(receipt.requests for receipt in receipts),
        providerResponseIds=tuple(
            value for receipt in receipts for value in receipt.providerResponseIds
        ),
        finishReasons=tuple(value for receipt in receipts for value in receipt.finishReasons),
        inputTokens=sum(receipt.inputTokens for receipt in receipts),
        cacheReadTokens=sum(receipt.cacheReadTokens for receipt in receipts),
        cacheWriteTokens=sum(receipt.cacheWriteTokens for receipt in receipts),
        outputTokens=sum(receipt.outputTokens for receipt in receipts),
        reasoningTokens=sum(receipt.reasoningTokens for receipt in receipts),
        visibleOutputTokens=sum(receipt.visibleOutputTokens for receipt in receipts),
        providerTokenAccountingAnomaly=any(
            receipt.providerTokenAccountingAnomaly for receipt in receipts
        ),
        estimatedCostUsd=sum((receipt.estimatedCostUsd for receipt in receipts), Decimal(0)),
        providerReportedCostUsd=(
            sum((cast(Decimal, value) for value in reported), Decimal(0))
            if reported and reported[0] is not None
            else None
        ),
        downstreamProviders=tuple(
            value for receipt in receipts for value in receipt.downstreamProviders
        ),
    )


def _combined_stage(
    *,
    role: Literal["compiler", "critic"],
    pass_number: int,
    stages: Sequence[AgentStageArtifact],
    output: BaseModel | None,
) -> AgentStageArtifact:
    if not stages:
        raise ValueError("cannot combine an empty stage sequence")
    failed = next((stage for stage in reversed(stages) if stage.status != "success"), None)
    started_at = min(stage.started_at for stage in stages)
    completed_at = max(stage.completed_at for stage in stages)
    return AgentStageArtifact(
        role=role,
        pass_number=pass_number,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=(completed_at - started_at).total_seconds(),
        system_prompt_sha256=sha256_bytes(
            canonical_json_bytes(tuple(stage.system_prompt_sha256 for stage in stages))
        ),
        user_prompt_sha256=sha256_bytes(
            canonical_json_bytes(tuple(stage.user_prompt_sha256 for stage in stages))
        ),
        output_schema_sha256=sha256_bytes(
            canonical_json_bytes(tuple(stage.output_schema_sha256 for stage in stages))
        ),
        status=failed.status if failed is not None else "success",
        output=output.model_dump(mode="json") if output is not None else None,
        error_type=failed.error_type if failed is not None else None,
        error_message=failed.error_message if failed is not None else None,
        messages=[message for stage in stages for message in cast(list[Any], stage.messages)],
        usage=_combined_usage(stages),
    )


def _local_contract_error_stage(
    *,
    role: Literal["compiler", "critic"],
    pass_number: int,
    system_prompt: str,
    payload: Mapping[str, Any],
    operation: str,
    error: ValueError,
) -> AgentStageArtifact:
    """Record a provider-free local contract rejection as an explicit case stage."""

    timestamp = datetime.now(UTC)
    user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    message = f"{operation} rejected locally before provider launch: {error}"
    return AgentStageArtifact(
        role=role,
        pass_number=pass_number,
        started_at=timestamp,
        completed_at=timestamp,
        duration_seconds=0.0,
        system_prompt_sha256=sha256_bytes(system_prompt.encode("utf-8")),
        user_prompt_sha256=sha256_bytes(user_prompt.encode("utf-8")),
        output_schema_sha256=sha256_bytes(
            canonical_json_bytes({"localContractOperation": operation})
        ),
        status="host_rejected",
        output=None,
        error_type=type(error).__name__,
        error_message=message,
        messages=[
            {
                "part_kind": "local-contract-error",
                "operation": operation,
                "requiredRevision": payload.get("requiredRevision"),
                "message": message,
            }
        ],
        usage=empty_usage(),
    )


def _cached_success_stage(
    *,
    role: Literal["compiler", "critic"],
    pass_number: int,
    system_prompt: str,
    payload: Mapping[str, Any],
    output_type: type[BaseModel],
    output: BaseModel,
    cache_key: str,
) -> AgentStageArtifact:
    """Record an exact immutable-state reuse without duplicating provider usage."""

    timestamp = datetime.now(UTC)
    user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return AgentStageArtifact(
        role=role,
        pass_number=pass_number,
        started_at=timestamp,
        completed_at=timestamp,
        duration_seconds=0.0,
        system_prompt_sha256=sha256_bytes(system_prompt.encode("utf-8")),
        user_prompt_sha256=sha256_bytes(user_prompt.encode("utf-8")),
        output_schema_sha256=sha256_bytes(
            canonical_json_bytes(output_type.model_json_schema(mode="validation"))
        ),
        status="success",
        output=output.model_dump(mode="json"),
        error_type=None,
        error_message=None,
        messages=[
            {
                "part_kind": "local-cache-hit",
                "cache_key": cache_key,
                "contract": "exact_immutable_critic_substage_v1",
            }
        ],
        usage=empty_usage(),
    )


def _settings(provider: ProviderConfig) -> ModelSettings:
    if provider.kind == "openrouter":
        openrouter_settings: OpenRouterModelSettings = {
            "max_tokens": provider.max_output_tokens,
            "timeout": provider.request_timeout_seconds,
            "openrouter_reasoning": {"effort": provider.reasoning_effort},
            "openrouter_provider": cast(
                Any,
                {
                    "only": list(provider.provider_only),
                    "order": list(provider.provider_only),
                    "require_parameters": provider.require_parameters,
                    "data_collection": provider.data_collection,
                    "allow_fallbacks": provider.allow_fallbacks,
                    "max_price": {
                        "prompt": float(provider.max_prompt_price_usd_per_million),
                        "completion": float(provider.max_completion_price_usd_per_million),
                    },
                },
            ),
        }
        return cast(ModelSettings, openrouter_settings)
    openai_settings: OpenAIResponsesModelSettings = {
        "max_tokens": provider.max_output_tokens,
        "timeout": provider.request_timeout_seconds,
        "openai_reasoning_effort": provider.reasoning_effort,
        "openai_reasoning_mode": "standard",
        "openai_reasoning_context": "current_turn",
        "openai_store": provider.store_responses,
    }
    return cast(ModelSettings, openai_settings)


def _openrouter_profile(provider: ProviderConfig) -> ModelProfile | None:
    if provider.kind != "openrouter":
        return None
    if provider.native_structured_output_profile == "provider_default":
        return None
    return ModelProfile(supports_json_schema_output=True)


class AgentRuntime:
    def __init__(
        self,
        *,
        project_root: Path,
        environment_file: str,
        compiler_provider: ProviderConfig,
        critic_provider: ProviderConfig,
        agent_contract_protocol: AgentContractProtocol,
        max_concurrent_requests: int,
        partition_critic_above_request_bytes: int | None = None,
    ) -> None:
        self._agent_contract_protocol = agent_contract_protocol
        if agent_contract_protocol in _HYBRID_PROTOCOLS:
            if (
                partition_critic_above_request_bytes is None
                or partition_critic_above_request_bytes <= 0
            ):
                raise ValueError("hybrid critic routing requires a positive byte threshold")
        elif partition_critic_above_request_bytes is not None:
            raise ValueError("critic partition threshold requires the hybrid protocol")
        self._partition_critic_above_request_bytes = partition_critic_above_request_bytes
        self._providers = {
            "compiler": compiler_provider,
            "critic": critic_provider,
        }
        self._models: dict[str, Model] = {}
        for role, provider in self._providers.items():
            key = load_provider_key(project_root, environment_file, provider.api_key_env)
            client = AsyncOpenAI(
                api_key=key,
                base_url=(
                    "https://openrouter.ai/api/v1" if provider.kind == "openrouter" else None
                ),
                max_retries=provider.transport_max_retries,
                timeout=provider.request_timeout_seconds,
                default_headers=(
                    {"X-Title": "DocumentParsing"} if provider.kind == "openrouter" else None
                ),
            )
            if provider.kind == "openrouter":
                self._models[role] = OpenRouterModel(
                    provider.model,
                    provider=OpenRouterProvider(openai_client=client),
                    profile=_openrouter_profile(provider),
                )
            else:
                self._models[role] = OpenAIResponsesModel(
                    provider.model,
                    provider=OpenAIProvider(openai_client=client),
                )
        self._limiter = asyncio.Semaphore(max_concurrent_requests)
        self._critic_substage_cache: dict[str, StagedAuditOutput | StagedFacetAuditOutput] = {}

    def _critic_cache_key(
        self,
        *,
        system_prompt: str,
        payload: Mapping[str, Any],
        output_type: type[BaseModel],
        output_name: str,
        output_description: str,
    ) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "provider": self._providers["critic"].model_dump(mode="json"),
                    "protocol": self._agent_contract_protocol,
                    "systemPrompt": system_prompt,
                    "payload": payload,
                    "outputSchema": output_type.model_json_schema(mode="validation"),
                    "outputName": output_name,
                    "outputDescription": output_description,
                }
            )
        )

    async def compiler(
        self,
        *,
        pass_number: int,
        system_prompt: str,
        payload: Mapping[str, Any],
        retries: int,
        repair_prompt: str | None = None,
        preview_output: Callable[[CompilerAgentOutput], None] | None = None,
    ) -> tuple[CompilerAgentOutput | None, AgentStageArtifact]:
        if self._agent_contract_protocol in _STAGED_PROTOCOLS and repair_prompt is None:
            raise ValueError("staged protocols require a compiler repair prompt")
        prior_candidate = payload.get("previousCandidateOutput")
        if (
            self._agent_contract_protocol
            in {
                "compact_discriminated_v2",
                *_REFERENCE_PROTOCOLS,
                *_STAGED_PROTOCOLS,
            }
            and isinstance(prior_candidate, Mapping)
            and isinstance(payload.get("requiredRevision"), str)
        ):
            prior = CompilerAgentOutput.model_validate_json(
                json.dumps(prior_candidate, ensure_ascii=False, separators=(",", ":"))
            )
            if self._agent_contract_protocol in _LOCAL_COMPILER_REPAIR_PROTOCOLS:
                try:
                    repair_payload = build_local_compiler_repair_payload(payload, prior)
                except ValueError as error:
                    return None, _local_contract_error_stage(
                        role="compiler",
                        pass_number=pass_number,
                        system_prompt=cast(str, repair_prompt),
                        payload=payload,
                        operation="local_compiler_repair_projection",
                        error=error,
                    )
                candidate_slice = cast(Mapping[str, Any], repair_payload["candidateSlice"])
                repair_output_type = _scoped_compiler_repair_output_type(
                    prior,
                    removable_binding_keys=cast(
                        Sequence[str], candidate_slice["removableBindingKeys"]
                    ),
                    removable_anchor_ids=cast(
                        Sequence[str], candidate_slice["removableAnchorOverrideIds"]
                    ),
                    removable_semantic_paths=cast(
                        Sequence[str], candidate_slice["removableSemanticOnlyTargetPaths"]
                    ),
                    addable_anchor_ids=_compiler_repair_addable_anchor_ids(repair_payload, prior),
                )
            else:
                repair_output_type = _scoped_compiler_repair_output_type(prior)
                repair_payload = dict(payload)
            repair_payload["repairInstruction"] = (
                "Return only the local patch required by the repair response schema. Do not "
                "repeat the carrier, anchor inventory, unaffected bindings, or root compiler "
                "output. Audit and fix every defect listed in requiredRevision in one patch."
            )

            def validate_repair(value: _CompilerRepairOutput) -> _CompilerRepairOutput:
                _validate_compiler_repair_scope(value, repair_payload)
                translated_repair = _apply_compiler_repair(prior, value)
                if _compiler_operational_state(translated_repair) == _compiler_operational_state(
                    prior
                ):
                    raise ValueError(
                        "compiler repair changes explanations only; rationale and re-adding the "
                        "same anchor override cannot change ownership"
                    )
                if preview_output is not None:
                    try:
                        preview_output(translated_repair)
                    except ValueError as error:
                        if _compiler_repair_error_intersects_scope(error, repair_payload, value):
                            raise
                        # A successful local transaction can expose an unrelated latent defect.
                        # Return that candidate to the outer host gate, which will construct a
                        # fresh authoritative slice; retrying with the old schema cannot edit it.
                return value

            repair, artifact = await self._call(
                role="compiler",
                pass_number=pass_number,
                system_prompt=(
                    cast(str, repair_prompt)
                    if self._agent_contract_protocol in _STAGED_PROTOCOLS
                    else (
                        f"{system_prompt}\n\n{_DISCRIMINATED_BINDING_ADAPTER}\n\n"
                        f"{_COMPILER_REPAIR_ADAPTER}"
                    )
                ),
                payload=repair_payload,
                output_type=repair_output_type,
                output_name="carrier_bound_template_compiler_repair_v2",
                output_description=(
                    "Return only a local patch against previousCandidateOutput. Do not return "
                    "the complete compiler result."
                ),
                retries=retries,
                output_validator=validate_repair,
            )
            if repair is None:
                return None, artifact
            if not isinstance(repair, _CompilerRepairOutput):
                raise TypeError("compiler repair returned an unknown output type")
            translated = _apply_compiler_repair(prior, repair)
            return translated, artifact.model_copy(
                update={"output": translated.model_dump(mode="json")}
            )
        if self._agent_contract_protocol == "legacy_v1":
            return await self._call(
                role="compiler",
                pass_number=pass_number,
                system_prompt=system_prompt,
                payload=payload,
                output_type=CompilerAgentOutput,
                output_name="carrier_bound_template_compiler",
                retries=retries,
            )
        if self._agent_contract_protocol in {*_REFERENCE_PROTOCOLS, *_STAGED_PROTOCOLS}:
            raw_candidates = _reference_occurrence_rows(payload)
            reference_output_type = (
                _scoped_initial_compiler_output_type(
                    tuple(
                        cast(
                            Sequence[str],
                            payload.get(
                                "anchorBindingsAuthorizedForInitialReview",
                                payload.get("anchorBindingsRequiringSemanticReview", ()),
                            ),
                        )
                    ),
                    required_anchor_override_ids=(
                        tuple(
                            cast(
                                Sequence[str],
                                payload.get("anchorBindingsRequiringSemanticReview", ()),
                            )
                        )
                        if self._agent_contract_protocol == "candidate_first_staged_local_v8"
                        else ()
                    ),
                    cross_fact_review_ids=(
                        tuple(
                            cast(
                                Sequence[str],
                                payload.get("anchorBindingsWithCrossFactEqualityAmbiguity", ()),
                            )
                        )
                        if self._agent_contract_protocol == "candidate_first_staged_local_v8"
                        else ()
                    ),
                )
                if self._agent_contract_protocol in _PARTITIONED_PROTOCOLS
                else _fixed_reference_compiler_output_type()
                if self._agent_contract_protocol in _STAGED_PROTOCOLS
                else _scoped_reference_compiler_output_type(
                    tuple(cast(str, row["occurrenceId"]) for row in raw_candidates)
                )
            )

            def validate_reference_output(output: BaseModel) -> BaseModel:
                _restore_reference_compiler(output, payload)
                return output

            reference_output, reference_artifact = await self._call(
                role="compiler",
                pass_number=pass_number,
                system_prompt=(
                    f"{system_prompt}\n\n{_REFERENCE_COMPILER_ADAPTER}\n\n"
                    f"{_COMPACT_COMPILER_INPUT_ADAPTER}\n\n"
                    f"{_DISCRIMINATED_BINDING_ADAPTER}"
                    if self._agent_contract_protocol in _PARTITIONED_PROTOCOLS
                    else (
                        f"{system_prompt}\n\n{_REFERENCE_COMPILER_ADAPTER}\n\n"
                        f"{_DISCRIMINATED_BINDING_ADAPTER}"
                    )
                ),
                payload=payload,
                output_type=reference_output_type,
                output_name="carrier_bound_template_compiler_v4",
                retries=retries,
                output_validator=(
                    validate_reference_output
                    if self._agent_contract_protocol in _STAGED_PROTOCOLS
                    else None
                ),
            )
            if reference_output is None:
                return None, reference_artifact
            translated = _restore_reference_compiler(reference_output, payload)
            return translated, reference_artifact.model_copy(
                update={"output": translated.model_dump(mode="json")}
            )
        discriminated_output, discriminated_artifact = await self._call(
            role="compiler",
            pass_number=pass_number,
            system_prompt=f"{system_prompt}\n\n{_DISCRIMINATED_BINDING_ADAPTER}",
            payload=payload,
            output_type=DiscriminatedCompilerAgentOutput,
            output_name="carrier_bound_template_compiler_v2",
            retries=retries,
        )
        if discriminated_output is None:
            return None, discriminated_artifact
        translated = restore_legacy_compiler(discriminated_output)
        return translated, discriminated_artifact.model_copy(
            update={"output": translated.model_dump(mode="json")}
        )

    async def critic(
        self,
        *,
        pass_number: int,
        system_prompt: str,
        payload: Mapping[str, Any],
        retries: int,
        audit_prompt: str | None = None,
        plan_prompt: str | None = None,
        preview_review: Callable[[CriticAgentOutput], None] | None = None,
        candidate_only: bool = False,
    ) -> tuple[CriticAgentOutput | None, AgentStageArtifact]:
        if candidate_only and self._agent_contract_protocol not in _CANDIDATE_FIRST_PROTOCOLS:
            raise ValueError("candidate-only critic pass requires a candidate-first protocol")
        raw_allowed = payload.get("allowedRemovalLogicalKeys")
        if not isinstance(raw_allowed, (tuple, list)) or any(
            not isinstance(value, str) for value in raw_allowed
        ):
            raise ValueError("critic payload lacks a valid allowed-removal logical-key sequence")
        allowed = cast(Sequence[str], raw_allowed)
        compact_payload = compact_critic_payload(payload)
        uses_partitioned_critic = _uses_partitioned_critic(
            protocol=self._agent_contract_protocol,
            compact_payload=compact_payload,
            threshold_bytes=self._partition_critic_above_request_bytes,
        )
        if uses_partitioned_critic:
            if audit_prompt is None or plan_prompt is None or preview_review is None:
                raise ValueError(
                    "partitioned critic route requires audit and plan prompts plus a host preview"
                )
            audit_payload = build_staged_audit_payload(compact_payload)
            uses_facets = (
                self._agent_contract_protocol
                in {
                    "faceted_staged_local_v5",
                    "partitioned_staged_local_v6",
                    "candidate_first_staged_local_v7",
                    "candidate_first_staged_local_v8",
                }
                or self._agent_contract_protocol in _HYBRID_PROTOCOLS
            )
            partitions_literal_review = (
                self._agent_contract_protocol in _PARTITIONED_PROTOCOLS
                or self._agent_contract_protocol in _HYBRID_PROTOCOLS
            )
            audit: StagedAuditOutput | None
            if uses_facets:
                facet_payloads = build_staged_audit_facet_payloads(
                    audit_payload,
                    partition_literal_review=partitions_literal_review,
                    candidate_only=candidate_only,
                )

                async def run_facet(
                    facet_payload: Mapping[str, Any],
                ) -> tuple[StagedFacetAuditOutput | None, AgentStageArtifact]:
                    facet_contract = cast(Mapping[str, Any], facet_payload["auditFacet"])
                    facet_name = cast(str, facet_contract["name"])
                    request = compact_staged_audit_request(facet_payload)
                    output_name = f"carrier_bound_template_{facet_name}_audit_v9"
                    output_description = (
                        "Return only the assigned host-candidate review facet. Every finding "
                        "must resolve an assigned candidate; this pre-pass cannot certify the "
                        "template."
                        if candidate_only
                        else "Return only the assigned immutable semantic audit facet. "
                        "Do not plan or emit edits."
                    )
                    cache_key = self._critic_cache_key(
                        system_prompt=audit_prompt,
                        payload=request,
                        output_type=StagedFacetAuditOutput,
                        output_name=output_name,
                        output_description=output_description,
                    )
                    cached = self._critic_substage_cache.get(cache_key)
                    if isinstance(cached, StagedFacetAuditOutput):
                        return cached, _cached_success_stage(
                            role="critic",
                            pass_number=pass_number,
                            system_prompt=audit_prompt,
                            payload=request,
                            output_type=StagedFacetAuditOutput,
                            output=cached,
                            cache_key=cache_key,
                        )
                    output, stage = await self._call(
                        role="critic",
                        pass_number=pass_number,
                        system_prompt=audit_prompt,
                        payload=request,
                        output_type=StagedFacetAuditOutput,
                        output_name=output_name,
                        output_description=output_description,
                        retries=retries,
                        output_validator=lambda output: normalize_staged_facet_audit(
                            output, facet_payload
                        ),
                    )
                    if output is not None:
                        self._critic_substage_cache[cache_key] = output
                    return output, stage

                facet_results = await asyncio.gather(
                    *(run_facet(facet_payload) for facet_payload in facet_payloads)
                )
                facet_outputs = tuple(output for output, _stage in facet_results)
                facet_stages = tuple(stage for _output, stage in facet_results)
                if any(output is None for output in facet_outputs):
                    return None, _combined_stage(
                        role="critic",
                        pass_number=pass_number,
                        stages=facet_stages,
                        output=None,
                    )
                audit = merge_staged_facet_audits(
                    outputs=cast(Sequence[StagedFacetAuditOutput], facet_outputs),
                    facet_payloads=facet_payloads,
                    full_payload=audit_payload,
                )
                audit_stage = _combined_stage(
                    role="critic",
                    pass_number=pass_number,
                    stages=facet_stages,
                    output=audit,
                )
            else:
                audit_request = compact_staged_audit_request(audit_payload)
                audit_output_name = "carrier_bound_template_audit_v8"
                audit_output_description = (
                    "Return only the immutable semantic audit. Do not plan or emit edits."
                )
                audit_cache_key = self._critic_cache_key(
                    system_prompt=audit_prompt,
                    payload=audit_request,
                    output_type=StagedAuditOutput,
                    output_name=audit_output_name,
                    output_description=audit_output_description,
                )
                cached_audit = self._critic_substage_cache.get(audit_cache_key)
                if isinstance(cached_audit, StagedAuditOutput):
                    audit = cached_audit
                    audit_stage = _cached_success_stage(
                        role="critic",
                        pass_number=pass_number,
                        system_prompt=audit_prompt,
                        payload=audit_request,
                        output_type=StagedAuditOutput,
                        output=cached_audit,
                        cache_key=audit_cache_key,
                    )
                else:
                    audit, audit_stage = await self._call(
                        role="critic",
                        pass_number=pass_number,
                        system_prompt=audit_prompt,
                        payload=audit_request,
                        output_type=StagedAuditOutput,
                        output_name=audit_output_name,
                        output_description=audit_output_description,
                        retries=retries,
                        output_validator=lambda output: validate_staged_audit(
                            output, audit_payload
                        ),
                    )
                    if audit is not None:
                        self._critic_substage_cache[audit_cache_key] = audit
            if audit is None:
                return None, audit_stage
            if audit.verdict == "pass":
                translated = staged_audit_pass(audit)
                return translated, audit_stage.model_copy(
                    update={"output": translated.model_dump(mode="json")}
                )

            plan_payload = build_staged_plan_payload(
                audit_payload=audit_payload,
                compact_payload=compact_payload,
                audit=audit,
            )
            plan_request = compact_staged_plan_request(plan_payload)

            def validate_plan(plan: StagedCriticPlanOutput) -> StagedCriticPlanOutput:
                translated_plan = restore_staged_plan(
                    plan=plan,
                    audit=audit,
                    plan_payload=plan_payload,
                    compact_payload=compact_payload,
                    source_payload=payload,
                )
                preview_review(translated_plan)
                return plan

            plan, plan_stage = await self._call(
                role="critic",
                pass_number=pass_number,
                system_prompt=plan_prompt,
                payload=plan_request,
                output_type=_scoped_staged_plan_output_type(plan_payload),
                output_name=(
                    "carrier_bound_template_transaction_plan_v9"
                    if self._agent_contract_protocol
                    in {
                        "faceted_staged_local_v5",
                        "partitioned_staged_local_v6",
                        "candidate_first_staged_local_v7",
                        "candidate_first_staged_local_v8",
                        *_HYBRID_PROTOCOLS,
                    }
                    else "carrier_bound_template_transaction_plan_v8"
                ),
                output_description=(
                    "Return only one host-previewable transaction for the supplied audit findings."
                ),
                retries=retries,
                output_validator=validate_plan,
            )
            if plan is None:
                return None, _combined_stage(
                    role="critic",
                    pass_number=pass_number,
                    stages=(audit_stage, plan_stage),
                    output=None,
                )
            translated = restore_staged_plan(
                plan=plan,
                audit=audit,
                plan_payload=plan_payload,
                compact_payload=compact_payload,
                source_payload=payload,
            )
            return translated, _combined_stage(
                role="critic",
                pass_number=pass_number,
                stages=(audit_stage, plan_stage),
                output=translated,
            )
        if self._agent_contract_protocol == "legacy_v1":
            legacy_output_type = _scoped_critic_output_type(allowed)
            legacy_output, legacy_artifact = await self._call(
                role="critic",
                pass_number=pass_number,
                system_prompt=system_prompt,
                payload=payload,
                output_type=legacy_output_type,
                output_name="carrier_bound_template_literal_critic",
                retries=retries,
            )
            if legacy_output is None:
                return None, legacy_artifact
            translated = _critic_output_with_inventory_ids(legacy_output)
            return translated, legacy_artifact.model_copy(
                update={"output": translated.model_dump(mode="json")}
            )

        raw_candidates = compact_payload.get("reviewCandidates")
        if not isinstance(raw_candidates, (tuple, list)) or any(
            not isinstance(row, Mapping) or not isinstance(row.get("candidateId"), str)
            for row in raw_candidates
        ):
            raise ValueError("critic payload lacks a valid review-candidate sequence")
        raw_risks = compact_payload.get("remainingRiskCandidates")
        if not isinstance(raw_risks, (tuple, list)) or any(
            not isinstance(row, Mapping) or not isinstance(row.get("risk_id"), str)
            for row in raw_risks
        ):
            raise ValueError("critic payload lacks a valid remaining-risk sequence")
        raw_literal_lines = compact_payload.get("literalLineReviewIds")
        if not isinstance(raw_literal_lines, (tuple, list)) or any(
            not isinstance(line_id, str) for line_id in raw_literal_lines
        ):
            raise ValueError("critic payload lacks a valid literal-line review sequence")
        raw_occurrences = _reference_occurrence_rows(compact_payload)
        inventory_occurrences = _compact_inventory_occurrence_lookup(compact_payload)
        required_coherence_ids = tuple(
            cast(str, row["candidateId"])
            for row in raw_candidates
            if row.get("requiredRevision") is True
            and row.get("kind") == "cross_field_semantic_relation"
        )
        compact_output_type = _scoped_compact_critic_output_type(
            allowed,
            tuple(cast(str, row["candidateId"]) for row in raw_candidates),
            tuple(cast(str, row["risk_id"]) for row in raw_risks),
            tuple(raw_literal_lines),
            required_review_candidate_ids=tuple(
                cast(str, row["candidateId"])
                for row in raw_candidates
                if row.get("requiredRevision") is True
                and row.get("kind") != "cross_field_semantic_relation"
            ),
            coherence_candidate_ids=required_coherence_ids,
            occurrence_ids=tuple(cast(str, row["occurrenceId"]) for row in raw_occurrences),
            inventory_occurrence_owners={
                occurrence_id: owner
                for occurrence_id, (owner, _occurrence) in inventory_occurrences.items()
            },
        )
        provider_payload = _critic_provider_payload(compact_payload)

        def validate_compact_output(output: BaseModel) -> BaseModel:
            translated_output = _restore_reference_compact_critic(
                output,
                compact_payload,
                payload,
            )
            if preview_review is not None and translated_output.verdict == "revise":
                preview_review(translated_output)
            return output

        compact_output, compact_artifact = await self._call(
            role="critic",
            pass_number=pass_number,
            system_prompt=f"{_COMPACT_CRITIC_ADAPTER}\n\n{system_prompt}",
            payload=provider_payload,
            output_type=compact_output_type,
            output_name="carrier_bound_template_literal_critic_v8",
            retries=retries,
            output_validator=(validate_compact_output if preview_review is not None else None),
        )
        if compact_output is None:
            return None, compact_artifact
        translated = _restore_reference_compact_critic(compact_output, compact_payload, payload)
        return translated, compact_artifact.model_copy(
            update={"output": translated.model_dump(mode="json")}
        )

    async def _call(
        self,
        *,
        role: str,
        pass_number: int,
        system_prompt: str,
        payload: Mapping[str, Any],
        output_type: type[OutputT],
        output_name: str,
        retries: int,
        output_description: str | None = None,
        output_validator: Callable[[OutputT], OutputT] | None = None,
    ) -> tuple[OutputT | None, AgentStageArtifact]:
        provider = self._providers[role]
        user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        schema = output_type.model_json_schema(mode="validation")
        agent = Agent[Any, OutputT](
            self._models[role],
            output_type=NativeOutput(
                output_type,
                name=output_name,
                description=output_description
                or (
                    "Return the complete typed result for the pinned source document. "
                    "Do not return prose outside the schema."
                ),
                strict=True,
            ),
            system_prompt=system_prompt,
            model_settings=_settings(provider),
            retries=retries,
            name=f"raw-text-template-{role}",
        )
        if output_validator is not None:

            @agent.output_validator
            def validate_output(output: OutputT) -> OutputT:
                try:
                    return output_validator(output)
                except ValueError as error:
                    raise ModelRetry(_structured_output_retry_message(error)) from error

        started_at = datetime.now(UTC)
        started = time.perf_counter()
        captured: list[Any]
        output: OutputT | None = None
        error_type: str | None = None
        error_message: str | None = None
        status = "success"
        async with self._limiter:
            with capture_run_messages() as messages:
                try:
                    result = await agent.run(
                        user_prompt,
                        usage_limits=UsageLimits(
                            request_limit=1 + retries,
                            output_tokens_limit=provider.max_output_tokens * (1 + retries),
                        ),
                    )
                    output = result.output
                except Exception as error:  # retained verbatim in the immutable case receipt
                    status = "provider_error"
                    error_type = type(error).__name__
                    error_message = str(error)
                captured = list(messages)
        completed_at = datetime.now(UTC)
        responses = tuple(message for message in captured if isinstance(message, ModelResponse))
        usage = (
            usage_receipt(
                responses,
                cast(Any, provider.pricing),
                require_provider_cost=provider.kind == "openrouter",
            )
            if responses
            else empty_usage()
        )
        artifact = AgentStageArtifact.model_validate(
            {
                "role": role,
                "pass_number": pass_number,
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": time.perf_counter() - started,
                "system_prompt_sha256": sha256_bytes(system_prompt.encode("utf-8")),
                "user_prompt_sha256": sha256_bytes(user_prompt.encode("utf-8")),
                "output_schema_sha256": sha256_bytes(canonical_json_bytes(schema)),
                "status": status,
                "output": output.model_dump(mode="json") if output is not None else None,
                "error_type": error_type,
                "error_message": error_message,
                "messages": model_messages(cast(Sequence[Any], captured)),
                "usage": usage,
            }
        )
        return output, artifact
