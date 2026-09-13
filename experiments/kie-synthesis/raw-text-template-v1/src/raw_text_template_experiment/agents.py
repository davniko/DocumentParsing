from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar, cast

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    load_provider_key,
    model_messages,
    usage_receipt,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from pydantic_ai import Agent, ModelProfile, NativeOutput, capture_run_messages
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model, ModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.usage import UsageLimits

from .compact_contract import BINDING_COLUMNS, OCCURRENCE_COLUMNS, compact_critic_payload
from .models import (
    AgentBindingProposal,
    AgentContractProtocol,
    AgentOccurrence,
    AgentStageArtifact,
    AnchorOverride,
    CarrierAssessment,
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

OutputT = TypeVar("OutputT", bound=BaseModel)

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
# Compact exhaustive critic protocol (v7)

The payload is a lossless table encoding of the semantic objects named by the rules below. Read
`bindingRows` using `compactContract.bindingColumns`, `occurrenceRows` using
`compactContract.occurrenceColumns`, target path IDs through `targetPathTable`, and source-binding
indexes through `sourceBindingIdTable`. A marker `⟦binding_NNNN⟧` in `annotatedSource` refers to
the row with that binding ID. `allowedRemovalLogicalKeys` remains the exact removal vocabulary.
`annotatedSource` preserves every original source character between binding boundary markers and
is the authoritative provider view: unmarked text is currently unowned. Use it to audit both the
semantics and exact extent of every owned and unowned span without a duplicated source view.

The response has one top-level `decision`. Before returning either verdict, inspect every source
line and every inventory row for all six required coverage facets. Set each coverage field to true
only after completing that audit. In `literal_line_receipt`, return `true` under every exact
property named by `literalLineReviewIds` after inspecting that line. The receipt records exhaustive
coverage; report any defects separately in `findings`.
This is a required exhaustive receipt, not a sample. In `candidate_receipt`, return one concise
disposition under every exact property named by `reviewCandidates[].candidateId` and
`remainingRiskCandidates[].risk_id`. The property name is the candidate ID; do not repeat it in
the disposition.
Review candidates are high-recall host leads and may be valid; remaining risks are proven unowned
and therefore require a revision, even when their correct owner is `literal_static`. A `revise`
decision must aggregate every
independently visible defect into one complete local transaction; do not stop after finding the
first defect. Binding proposals use the v2 discriminated `rendering` object described by the
response schema. For every new occurrence available in `occurrenceCandidates`, return only its
`occurrence_id` handle; use a literal occurrence object solely when the exact intended span is
genuinely absent from that table. When an uncovered surface is merely another physical occurrence
of an existing binding with unchanged semantics, use `occurrence_appends`; do not remove or restate
that binding. When only one or more existing occurrences are semantically wrong, use
`occurrence_removals`; do not remove and restate the otherwise unchanged binding. Use full removal
plus `additional_bindings` only when the binding's semantic contract actually changes. Output full
target-path strings from the second column of `targetPathTable`, never the short path IDs.

When the semantic rules below say `bindingInventory`, consult `bindingRows` plus `occurrenceRows`.
When they say `allowedTargetPaths`, consult `targetPathTable`. Exact-match candidate data remains in
the occurrence table and is unchanged.
""".strip()

_REFERENCE_COMPILER_ADAPTER = """
# Host-resolved occurrence-reference protocol (v3)

The payload contains `occurrenceCandidates`, an immutable table of exact source spans already
resolved by the host. Read each row using the table's `columns`; numbered source supplies its full
context. In `carrier.evidence_occurrences` and every binding's `occurrences`, emit
`{"occurrence_id":"compiler_occurrence_NNNNN"}` whenever a listed row is the intended complete
surface. The response schema enumerates the only legal IDs. Do not copy that row's line or text
fields into the response. Use a full `{line_start,line_end,source_text,occurrence_index}` occurrence
only when the exact intended span is genuinely absent from the table; copy such text byte-for-byte.

The binding `rendering` object follows the v2 discriminated protocol. Before returning, verify that
each target path has exactly one logical owner and that repeated physical appearances are grouped
as occurrences of that one owner. The semantic compiler rules below remain authoritative.
""".strip()

_REFERENCE_OCCURRENCE_COLUMNS = (
    "occurrenceId",
    "lineStart",
    "lineEnd",
    "sourceText",
    "occurrenceIndex",
)

_COMPILER_REPAIR_ADAPTER = """
# Local compiler repair protocol (v2)

The payload contains `previousCandidateOutput` and an exact `requiredRevision` from the host.
Return only the smallest complete patch that makes that candidate satisfy every listed defect.
Do not repeat unaffected bindings. Remove a binding logical key before replacing it. When one
printed surface was incorrectly assigned to independently mutable target facts, retain only the
fact the source topology supports and declare a genuinely unprinted fact through
`additional_semantic_only_target_facts`; never merge independent facts merely because their
current values are equal. Replacement bindings use the discriminated `rendering` schema.
""".strip()


class _CompilerRepairOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    remove_binding_logical_keys: Annotated[
        tuple[str, ...], Field(json_schema_extra={"uniqueItems": True})
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


class _CriticCandidateDisposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    conclusion: Literal["valid_existing_contract", "defect_requires_revision"]
    rationale: NonEmptyText


class _CriticRequiredRevisionDisposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    conclusion: Literal["defect_requires_revision"]
    rationale: NonEmptyText


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
) -> type[_CompilerRepairOutput]:
    binding_keys = tuple(dict.fromkeys(row.logical_key for row in prior.bindings))
    anchor_ids = tuple(dict.fromkeys(row.anchor_binding_id for row in prior.anchor_overrides))
    semantic_paths = tuple(
        dict.fromkeys(row.target_path for row in prior.semantic_only_target_facts)
    )
    fields: dict[str, Any] = {}
    if binding_keys:
        fields["remove_binding_logical_keys"] = (
            Annotated[
                tuple[Literal.__getitem__(binding_keys), ...],
                Field(json_schema_extra={"uniqueItems": True}),
            ],
            (),
        )
    if anchor_ids:
        fields["remove_anchor_override_ids"] = (
            Annotated[
                tuple[Literal.__getitem__(anchor_ids), ...],
                Field(json_schema_extra={"uniqueItems": True}),
            ],
            (),
        )
    if semantic_paths:
        fields["remove_semantic_only_target_paths"] = (
            Annotated[
                tuple[Literal.__getitem__(semantic_paths), ...],
                Field(json_schema_extra={"uniqueItems": True}),
            ],
            (),
        )

    @model_validator(mode="after")
    def replacement_operations_are_transactional(
        value: _CompilerRepairOutput,
    ) -> _CompilerRepairOutput:
        missing_binding_removals = sorted(
            {row.logical_key for row in value.replacement_bindings}
            & set(binding_keys) - set(value.remove_binding_logical_keys)
        )
        missing_override_removals = sorted(
            {row.anchor_binding_id for row in value.additional_anchor_overrides}
            & set(anchor_ids) - set(value.remove_anchor_override_ids)
        )
        missing_semantic_removals = sorted(
            {row.target_path for row in value.additional_semantic_only_target_facts}
            & set(semantic_paths) - set(value.remove_semantic_only_target_paths)
        )
        errors = (
            *(f"binding:{key}" for key in missing_binding_removals),
            *(f"anchor_override:{key}" for key in missing_override_removals),
            *(f"semantic_only:{key}" for key in missing_semantic_removals),
        )
        if errors:
            raise ValueError(
                "replacement operations must remove every existing key before replacing it: "
                + ", ".join(errors)
            )
        return value

    digest = sha256_bytes(canonical_json_bytes((binding_keys, anchor_ids, semantic_paths)))[:16]
    return cast(
        type[_CompilerRepairOutput],
        create_model(
            f"ScopedCompilerRepairOutput_{digest}",
            __base__=_CompilerRepairOutput,
            __module__=__name__,
            __validators__={
                "replacement_operations_are_transactional": replacement_operations_are_transactional
            },
            **fields,
        ),
    )


def _apply_compiler_repair(
    prior: CompilerAgentOutput, repair: _CompilerRepairOutput
) -> CompilerAgentOutput:
    remove_bindings = set(repair.remove_binding_logical_keys)
    unknown_bindings = remove_bindings - {row.logical_key for row in prior.bindings}
    if unknown_bindings:
        raise ValueError("compiler repair removes unknown bindings")
    replacements = tuple(restore_legacy_binding(row) for row in repair.replacement_bindings)
    retained_bindings = tuple(
        row for row in prior.bindings if row.logical_key not in remove_bindings
    )
    retained_keys = {row.logical_key for row in retained_bindings}
    collisions = retained_keys & {row.logical_key for row in replacements}
    if collisions:
        raise ValueError(
            "compiler replacements collide with bindings not removed: "
            + ", ".join(sorted(collisions))
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
    """Return the single-source provider view while retaining the full host contract separately."""

    if not isinstance(compact_payload.get("annotatedSource"), str) or not isinstance(
        compact_payload.get("maskedTemplate"), str
    ):
        raise ValueError("compact critic payload requires annotated and masked source views")
    provider_payload = dict(compact_payload)
    del provider_payload["maskedTemplate"]
    return provider_payload


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
    rationale: NonEmptyText

    @model_validator(mode="after")
    def verdict_is_consistent(self) -> _CriticLogicalKeyOutput:
        if self.verdict == "pass" and (
            self.findings
            or self.remove_binding_logical_keys
            or self.additional_bindings
            or self.semantic_only_target_facts
        ):
            raise ValueError("pass requires zero findings and zero patch operations")
        if self.verdict == "revise" and not self.findings:
            raise ValueError("revise requires at least one finding")
        if len(set(self.remove_binding_logical_keys)) != len(self.remove_binding_logical_keys):
            raise ValueError("critic removal logical keys must be unique")
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
        remove_binding_logical_keys=(tuple[logical_key, ...], ()),
    )
    return cast(type[_CriticLogicalKeyOutput], model)


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
) -> type[BaseModel]:
    """Build a root-object discriminated verdict with an exact removal vocabulary."""

    allowed = tuple(allowed_removal_logical_keys)
    if not allowed:
        raise ValueError("critic request has no allowed removal logical keys")
    if len(set(allowed)) != len(allowed):
        raise ValueError("critic request contains duplicate removal logical keys")
    expected_candidates = tuple(review_candidate_ids)
    required_risks = tuple(remaining_risk_ids)
    if len(set(expected_candidates)) != len(expected_candidates):
        raise ValueError("critic request contains duplicate review candidate IDs")
    if len(set(required_risks)) != len(required_risks):
        raise ValueError("critic request contains duplicate remaining risk IDs")
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
    logical_key = Literal.__getitem__(allowed)
    candidate_receipt_type = create_model(
        "ScopedCriticCandidateReceipt_" + sha256_bytes(canonical_json_bytes(all_candidates))[:16],
        __module__=__name__,
        __config__=ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False),
        **{
            candidate_id: (
                _CriticRequiredRevisionDisposition
                if candidate_id in required_risks
                else _CriticCandidateDisposition,
                ...,
            )
            for candidate_id in all_candidates
        },
    )
    literal_line_receipt_type = create_model(
        "ScopedCriticLiteralLineReceipt_" + sha256_bytes(canonical_json_bytes(expected_lines))[:16],
        __module__=__name__,
        __config__=ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False),
        **{
            line_id: (
                Literal[True],
                ...,
            )
            for line_id in expected_lines
        },
    )

    @model_validator(mode="after")
    def pass_has_no_candidate_defects(value: BaseModel) -> BaseModel:
        if any(
            disposition["conclusion"] == "defect_requires_revision"
            for disposition in value.coverage.candidate_receipt.model_dump().values()
        ):
            raise ValueError("pass cannot retain a candidate defect")
        return value

    digest = sha256_bytes(
        canonical_json_bytes(
            (
                allowed,
                expected_candidates,
                required_risks,
                expected_lines,
                allowed_occurrences,
                removable_occurrence_owners,
            )
        )
    )[:16]
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
        literal_line_receipt=(literal_line_receipt_type, ...),
    )
    pass_type = create_model(
        f"ScopedCompactCriticPass_{digest}",
        __base__=CompactCriticPassOutput,
        __module__=__name__,
        __validators__={"pass_has_no_candidate_defects": pass_has_no_candidate_defects},
        coverage=(coverage_type, ...),
    )
    binding_type: Any = DiscriminatedBindingProposal
    append_fields: dict[str, Any] = {"logical_key": (logical_key, ...)}
    if allowed_occurrences:
        occurrence_id = Literal.__getitem__(allowed_occurrences)
        reference_type = create_model(
            f"ScopedCriticOccurrenceReference_{digest}",
            __base__=CompilerOccurrenceReference,
            __module__=__name__,
            occurrence_id=(occurrence_id, ...),
        )
        binding_type = create_model(
            f"ScopedCriticBindingProposal_{digest}",
            __base__=DiscriminatedBindingProposal,
            __module__=__name__,
            occurrences=(
                Annotated[
                    tuple[reference_type | AgentOccurrence, ...],
                    Field(min_length=1),
                ],
                ...,
            ),
        )
        append_fields["occurrences"] = (
            Annotated[
                tuple[reference_type | AgentOccurrence, ...],
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
    if removable_occurrence_owners:
        inventory_occurrence_id = Literal.__getitem__(tuple(removable_occurrence_owners))

        @model_validator(mode="after")
        def removal_occurrences_belong_to_binding(
            value: _CriticOccurrenceRemovalReference,
        ) -> _CriticOccurrenceRemovalReference:
            occurrence_ids_value = tuple(value.occurrence_ids)
            if len(set(occurrence_ids_value)) != len(occurrence_ids_value):
                raise ValueError("critic occurrence-removal IDs must be unique")
            wrong_owner = tuple(
                occurrence_id
                for occurrence_id in occurrence_ids_value
                if removable_occurrence_owners[occurrence_id] != value.logical_key
            )
            if wrong_owner:
                raise ValueError(
                    "critic occurrence removals must belong to the declared logical key: "
                    + ", ".join(wrong_owner)
                )
            owned = {
                occurrence_id
                for occurrence_id, owner in removable_occurrence_owners.items()
                if owner == value.logical_key
            }
            if set(occurrence_ids_value) == owned:
                raise ValueError(
                    "critic occurrence removal cannot empty a binding; use full removal"
                )
            return value

        removal_type = create_model(
            f"ScopedCriticOccurrenceRemoval_{digest}",
            __base__=_CriticOccurrenceRemovalReference,
            __module__=__name__,
            __validators__={
                "removal_occurrences_belong_to_binding": removal_occurrences_belong_to_binding
            },
            logical_key=(logical_key, ...),
            occurrence_ids=(
                Annotated[
                    tuple[inventory_occurrence_id, ...],
                    Field(min_length=1, json_schema_extra={"uniqueItems": True}),
                ],
                ...,
            ),
        )

    @model_validator(mode="after")
    def critic_operations_are_unambiguous(value: BaseModel) -> BaseModel:
        append_keys = tuple(row.logical_key for row in value.occurrence_appends)
        occurrence_removal_keys = tuple(row.logical_key for row in value.occurrence_removals)
        replacement_keys = tuple(row.logical_key for row in value.additional_bindings)
        if len(set(append_keys)) != len(append_keys):
            raise ValueError("critic occurrence-append logical keys must be unique")
        if len(set(occurrence_removal_keys)) != len(occurrence_removal_keys):
            raise ValueError("critic occurrence-removal logical keys must be unique")
        mutation_keys = set(value.remove_binding_logical_keys) | set(replacement_keys)
        collisions = ((set(append_keys) | set(occurrence_removal_keys)) & mutation_keys) | (
            set(append_keys) & set(occurrence_removal_keys)
        )
        if collisions:
            raise ValueError(
                "critic occurrence appends cannot also remove or replace the same binding: "
                + ", ".join(sorted(collisions))
            )
        return value

    revision_type = create_model(
        f"ScopedCompactCriticRevision_{digest}",
        __base__=CompactCriticRevisionOutput,
        __module__=__name__,
        __validators__={
            "critic_operations_are_unambiguous": critic_operations_are_unambiguous,
        },
        coverage=(coverage_type, ...),
        remove_binding_logical_keys=(tuple[logical_key, ...], ()),
        additional_bindings=(tuple[binding_type, ...], ()),
        occurrence_appends=(tuple[append_type, ...], ()),
        occurrence_removals=(tuple[removal_type, ...], ()),
    )
    decision_type = Annotated[
        pass_type | revision_type,
        Field(discriminator="verdict"),
    ]
    envelope = create_model(
        f"ScopedCompactCriticEnvelope_{digest}",
        __module__=__name__,
        __config__=ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False),
        decision=(decision_type, ...),
    )
    return cast(type[BaseModel], envelope)


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
        bindings=(tuple[binding_type, ...], ...),
    )
    return cast(type[BaseModel], output_type)


def _restore_reference_compiler(
    output: BaseModel,
    payload: Mapping[str, Any],
) -> CompilerAgentOutput:
    """Materialize provider span handles and restore the stable compiler contract."""

    lookup = _reference_occurrence_lookup(payload)
    candidate = output.model_dump(mode="python")
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

    decision = output.decision
    if decision.verdict == "pass":
        return CriticAgentOutput(
            verdict="pass",
            findings=(),
            remove_inventory_binding_ids=(),
            additional_bindings=(),
            semantic_only_target_facts=(),
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


def _settings(provider: ProviderConfig) -> ModelSettings:
    if provider.kind == "openrouter":
        settings: OpenRouterModelSettings = {
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
        return cast(ModelSettings, settings)
    settings: OpenAIResponsesModelSettings = {
        "max_tokens": provider.max_output_tokens,
        "timeout": provider.request_timeout_seconds,
        "openai_reasoning_effort": provider.reasoning_effort,
        "openai_reasoning_mode": "standard",
        "openai_reasoning_context": "current_turn",
        "openai_store": provider.store_responses,
    }
    return cast(ModelSettings, settings)


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
    ) -> None:
        self._agent_contract_protocol = agent_contract_protocol
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

    async def compiler(
        self,
        *,
        pass_number: int,
        system_prompt: str,
        payload: Mapping[str, Any],
        retries: int,
    ) -> tuple[CompilerAgentOutput | None, AgentStageArtifact]:
        prior_candidate = payload.get("previousCandidateOutput")
        if (
            self._agent_contract_protocol in {"compact_discriminated_v2", "reference_compact_v3"}
            and isinstance(prior_candidate, Mapping)
            and isinstance(payload.get("requiredRevision"), str)
        ):
            prior = CompilerAgentOutput.model_validate_json(
                json.dumps(prior_candidate, ensure_ascii=False, separators=(",", ":"))
            )
            output_type = _scoped_compiler_repair_output_type(prior)
            repair_payload = dict(payload)
            repair_payload["repairInstruction"] = (
                "Return only the local patch required by the repair response schema. Do not "
                "repeat the carrier, anchor inventory, unaffected bindings, or root compiler "
                "output. Audit and fix every defect listed in requiredRevision in one patch."
            )
            repair, artifact = await self._call(
                role="compiler",
                pass_number=pass_number,
                system_prompt=(
                    f"{system_prompt}\n\n{_DISCRIMINATED_BINDING_ADAPTER}\n\n"
                    f"{_COMPILER_REPAIR_ADAPTER}"
                ),
                payload=repair_payload,
                output_type=output_type,
                output_name="carrier_bound_template_compiler_repair_v2",
                output_description=(
                    "Return only a local patch against previousCandidateOutput. Do not return "
                    "the complete compiler result."
                ),
                retries=retries,
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
        if self._agent_contract_protocol == "reference_compact_v3":
            raw_candidates = _reference_occurrence_rows(payload)
            output_type = _scoped_reference_compiler_output_type(
                tuple(cast(str, row["occurrenceId"]) for row in raw_candidates)
            )
            output, artifact = await self._call(
                role="compiler",
                pass_number=pass_number,
                system_prompt=(
                    f"{system_prompt}\n\n{_REFERENCE_COMPILER_ADAPTER}\n\n"
                    f"{_DISCRIMINATED_BINDING_ADAPTER}"
                ),
                payload=payload,
                output_type=output_type,
                output_name="carrier_bound_template_compiler_v3",
                retries=retries,
            )
            if output is None:
                return None, artifact
            translated = _restore_reference_compiler(output, payload)
            return translated, artifact.model_copy(
                update={"output": translated.model_dump(mode="json")}
            )
        output, artifact = await self._call(
            role="compiler",
            pass_number=pass_number,
            system_prompt=f"{system_prompt}\n\n{_DISCRIMINATED_BINDING_ADAPTER}",
            payload=payload,
            output_type=DiscriminatedCompilerAgentOutput,
            output_name="carrier_bound_template_compiler_v2",
            retries=retries,
        )
        if output is None:
            return None, artifact
        translated = restore_legacy_compiler(output)
        return translated, artifact.model_copy(
            update={"output": translated.model_dump(mode="json")}
        )

    async def critic(
        self,
        *,
        pass_number: int,
        system_prompt: str,
        payload: Mapping[str, Any],
        retries: int,
    ) -> tuple[CriticAgentOutput | None, AgentStageArtifact]:
        raw_allowed = payload.get("allowedRemovalLogicalKeys")
        if not isinstance(raw_allowed, (tuple, list)) or any(
            not isinstance(value, str) for value in raw_allowed
        ):
            raise ValueError("critic payload lacks a valid allowed-removal logical-key sequence")
        allowed = cast(Sequence[str], raw_allowed)
        if self._agent_contract_protocol == "legacy_v1":
            output_type = _scoped_critic_output_type(allowed)
            output, artifact = await self._call(
                role="critic",
                pass_number=pass_number,
                system_prompt=system_prompt,
                payload=payload,
                output_type=output_type,
                output_name="carrier_bound_template_literal_critic",
                retries=retries,
            )
            if output is None:
                return None, artifact
            translated = _critic_output_with_inventory_ids(output)
            return translated, artifact.model_copy(
                update={"output": translated.model_dump(mode="json")}
            )

        compact_payload = compact_critic_payload(payload)
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
        output_type = _scoped_compact_critic_output_type(
            allowed,
            tuple(cast(str, row["candidateId"]) for row in raw_candidates),
            tuple(cast(str, row["risk_id"]) for row in raw_risks),
            tuple(raw_literal_lines),
            tuple(cast(str, row["occurrenceId"]) for row in raw_occurrences),
            {
                occurrence_id: owner
                for occurrence_id, (owner, _occurrence) in inventory_occurrences.items()
            },
        )
        provider_payload = _critic_provider_payload(compact_payload)
        output, artifact = await self._call(
            role="critic",
            pass_number=pass_number,
            system_prompt=f"{_COMPACT_CRITIC_ADAPTER}\n\n{system_prompt}",
            payload=provider_payload,
            output_type=output_type,
            output_name="carrier_bound_template_literal_critic_v7",
            retries=retries,
        )
        if output is None:
            return None, artifact
        translated = _restore_reference_compact_critic(output, compact_payload, payload)
        return translated, artifact.model_copy(
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
