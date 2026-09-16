from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes

from .compact_contract import BINDING_COLUMNS, OCCURRENCE_COLUMNS
from .models import (
    AgentBindingProposal,
    AgentOccurrence,
    CoherenceCandidateDecision,
    CompilerAgentOutput,
    CriticAgentOutput,
    CriticFinding,
    CriticOccurrenceRemoval,
    GroupKind,
    LineId,
    NonEmptyText,
    SemanticOnlyTargetFactProposal,
    ValueKind,
)
from .optimization_contract import (
    BindingRendering,
    CompilerOccurrenceReference,
    DiscriminatedBindingProposal,
    restore_legacy_binding,
)

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_BINDING_ID = re.compile(r"^binding_(?P<index>[0-9]{4})$")
_PATH_ID = re.compile(r"^path_(?P<index>[0-9]{4})$")
_OCCURRENCE_ID = re.compile(r"^occ_L(?P<line>[0-9]{5})_(?P<index>[0-9]{5})$")
_NUMBERED_LINE = re.compile(r"^L(?P<number>[0-9]{5}) \| ")
_CurrentOccurrenceId = Annotated[
    str,
    StringConstraints(pattern=r"^occ_L[0-9]{5}_[0-9]{5}$"),
]

_TARGET_FACT_COLUMNS = ("targetPathIndex", "targetPath", "sourceValue")
_BINDING_COLUMNS = (
    "bindingIndex",
    "logicalKey",
    "renderMode",
    "valueKind",
    "groupKind",
    "groupKey",
    "targetPathIndexes",
    "targetPaths",
    "targetRelationship",
    "independentTargetFactComponents",
    "independentTargetFactComponentPaths",
    "derivation",
    "dependencyPathIndexes",
    "dependencyPaths",
    "dependencyBindings",
    "occurrenceIndexes",
)
_OCCURRENCE_COLUMNS = (
    "occurrenceRowIndex",
    "lineStart",
    "lineEnd",
    "sourceText",
    "occurrenceIndex",
)
_CANDIDATE_COLUMNS = (
    "candidateIndex",
    "candidateId",
    "kind",
    "requiredRevision",
    "bindingIndexes",
    "lineIds",
    "sourceTexts",
    "context",
    "details",
)
_COBINDING_COLUMNS = (
    "coBindingIndex",
    "relationship",
    "targetPathIndexes",
    "targetPaths",
)
_OCCURRENCE_CANDIDATE_COLUMNS = (
    "occurrenceId",
    "lineStart",
    "lineEnd",
    "sourceText",
    "occurrenceIndex",
    "currentExactOwnerLogicalKeys",
)
_AUDIT_TABLES = {
    "targetFacts": ("targetFactTable", _TARGET_FACT_COLUMNS),
    "requiredTargetCoBindings": ("coBindingTable", _COBINDING_COLUMNS),
    "bindings": ("bindingTable", _BINDING_COLUMNS),
    "occurrences": ("occurrenceTable", _OCCURRENCE_COLUMNS),
    "candidateRows": ("candidateTable", _CANDIDATE_COLUMNS),
}
_AUDIT_REQUEST_BINDING_COLUMNS = tuple(
    column for column in _BINDING_COLUMNS if column != "independentTargetFactComponentPaths"
)
_AUDIT_REQUEST_TABLES = {
    **_AUDIT_TABLES,
    "bindings": ("bindingTable", _AUDIT_REQUEST_BINDING_COLUMNS),
}
_PLAN_TABLES = {
    **_AUDIT_TABLES,
    "occurrenceCandidates": (
        "occurrenceCandidateTable",
        _OCCURRENCE_CANDIDATE_COLUMNS,
    ),
}


class StagedAuditFinding(BaseModel):
    """Grounded semantic defect plus compact host-issued selectors for repair slicing."""

    model_config = _STRICT

    finding_kind: Literal[
        "unowned_shipment_fact",
        "unowned_private_or_auxiliary_fact",
        "unowned_repeated_fact",
        "incorrect_static_classification",
        "incorrect_semantic_owner",
        "missing_derivation",
        "missing_coherence_dependency",
        "carrier_binding_error",
        "topology_or_grouping_error",
    ]
    line_ids: Annotated[tuple[LineId, ...], Field(min_length=1)]
    evidence: NonEmptyText
    explanation: NonEmptyText
    binding_indexes: tuple[Annotated[int, Field(ge=0)], ...] = ()
    target_path_indexes: tuple[Annotated[int, Field(ge=0)], ...] = ()
    candidate_indexes: tuple[Annotated[int, Field(ge=0)], ...] = ()

    @model_validator(mode="after")
    def selectors_are_unique(self) -> StagedAuditFinding:
        for name, values in (
            ("line IDs", self.line_ids),
            ("binding indexes", self.binding_indexes),
            ("target-path indexes", self.target_path_indexes),
            ("candidate indexes", self.candidate_indexes),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"audit finding {name} must be unique")
        if (
            self.finding_kind
            in {
                "incorrect_static_classification",
                "incorrect_semantic_owner",
                "missing_derivation",
                "missing_coherence_dependency",
                "carrier_binding_error",
                "topology_or_grouping_error",
            }
            and not self.binding_indexes
        ):
            raise ValueError("an existing-binding defect requires at least one binding index")
        return self


class StagedAuditCoverage(BaseModel):
    model_config = _STRICT

    literal_completeness_checked: Literal[True]
    target_ownership_checked: Literal[True]
    topology_and_grouping_checked: Literal[True]
    derivations_checked: Literal[True]
    carrier_boundary_checked: Literal[True]
    identifier_relationships_checked: Literal[True]


class StagedAuditOutput(BaseModel):
    model_config = _STRICT

    verdict: Literal["pass", "revise"]
    findings: tuple[StagedAuditFinding, ...]
    coverage: StagedAuditCoverage
    rationale: NonEmptyText

    @model_validator(mode="after")
    def verdict_matches_findings(self) -> StagedAuditOutput:
        if self.verdict == "pass" and self.findings:
            raise ValueError("audit pass cannot contain findings")
        if self.verdict == "revise" and not self.findings:
            raise ValueError("audit revise requires at least one finding")
        return self


AuditFacetName = Literal[
    "surface_completeness",
    "cargo_topology",
    "document_topology",
]


class StagedFacetCandidateDisposition(BaseModel):
    """One explicit semantic decision for one host-issued audit candidate."""

    model_config = _STRICT

    candidate_index: Annotated[int, Field(ge=0)]
    conclusion: Literal["valid_current_state", "defect_requires_revision"]


class StagedFacetAuditCoverage(BaseModel):
    """Facet-local completion receipt; the host validates its exact assigned inventory."""

    model_config = _STRICT

    assigned_bindings_checked: Literal[True]
    assigned_candidates_checked: Literal[True]
    assigned_literal_lines_checked: Literal[True]
    assigned_target_relationships_checked: Literal[True]


class StagedFacetAuditOutput(BaseModel):
    """A bounded audit result that cannot silently skip any assigned candidate."""

    model_config = _STRICT

    facet: AuditFacetName
    verdict: Literal["pass", "revise"]
    findings: tuple[StagedAuditFinding, ...]
    candidate_dispositions: tuple[StagedFacetCandidateDisposition, ...]
    coverage: StagedFacetAuditCoverage
    rationale: NonEmptyText

    @model_validator(mode="after")
    def verdict_matches_findings(self) -> StagedFacetAuditOutput:
        if self.verdict == "pass" and self.findings:
            raise ValueError("facet audit pass cannot contain findings")
        if self.verdict == "revise" and not self.findings:
            raise ValueError("facet audit revise requires at least one finding")
        indexes = tuple(row.candidate_index for row in self.candidate_dispositions)
        if len(set(indexes)) != len(indexes):
            raise ValueError("facet candidate dispositions must be unique")
        return self


class StagedPlanBindingProposal(BaseModel):
    model_config = _STRICT

    logical_key: NonEmptyText
    value_kind: ValueKind
    group_kind: GroupKind
    group_key: NonEmptyText
    rendering: BindingRendering
    occurrences: Annotated[
        tuple[CompilerOccurrenceReference | AgentOccurrence, ...],
        Field(min_length=1),
    ]
    rationale: NonEmptyText


class StagedPlanOccurrenceAppend(BaseModel):
    model_config = _STRICT

    logical_key: NonEmptyText
    occurrences: Annotated[
        tuple[CompilerOccurrenceReference | AgentOccurrence, ...],
        Field(min_length=1),
    ]
    rationale: NonEmptyText


class StagedPlanOccurrenceRemoval(BaseModel):
    model_config = _STRICT

    logical_key: NonEmptyText
    occurrence_ids: Annotated[
        tuple[_CurrentOccurrenceId, ...],
        Field(min_length=1, json_schema_extra={"uniqueItems": True}),
    ]
    rationale: NonEmptyText


class StagedCriticPlanOutput(BaseModel):
    model_config = _STRICT

    remove_binding_logical_keys: Annotated[
        tuple[NonEmptyText, ...], Field(json_schema_extra={"uniqueItems": True})
    ] = ()
    additional_bindings: tuple[StagedPlanBindingProposal, ...] = ()
    occurrence_appends: tuple[StagedPlanOccurrenceAppend, ...] = ()
    occurrence_removals: tuple[StagedPlanOccurrenceRemoval, ...] = ()
    semantic_only_target_facts: tuple[SemanticOnlyTargetFactProposal, ...] = ()
    coherence_decisions: tuple[CoherenceCandidateDecision, ...] = ()
    rationale: NonEmptyText

    @model_validator(mode="after")
    def transaction_is_nonempty_and_unambiguous(self) -> StagedCriticPlanOutput:
        if not any(
            (
                self.remove_binding_logical_keys,
                self.additional_bindings,
                self.occurrence_appends,
                self.occurrence_removals,
                self.semantic_only_target_facts,
                self.coherence_decisions,
            )
        ):
            raise ValueError("critic plan must contain at least one operation")
        for name, values in (
            ("binding removals", self.remove_binding_logical_keys),
            (
                "occurrence-removal owners",
                tuple(row.logical_key for row in self.occurrence_removals),
            ),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"critic plan {name} must be unique")
        replacement_keys = tuple(row.logical_key for row in self.additional_bindings)
        if len(set(replacement_keys)) != len(replacement_keys):
            raise ValueError("critic plan replacement keys must be unique")
        decision_ids = tuple(row.candidate_id for row in self.coherence_decisions)
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("critic plan coherence decision candidate IDs must be unique")
        mutated = set(self.remove_binding_logical_keys) | set(replacement_keys)
        local_edits = {
            *(row.logical_key for row in self.occurrence_appends),
            *(row.logical_key for row in self.occurrence_removals),
        }
        collisions = mutated & local_edits
        if collisions:
            raise ValueError(
                "full binding mutations cannot also use occurrence edits: "
                + ", ".join(sorted(collisions))
            )
        return self


def _id_index(value: Any, *, pattern: re.Pattern[str], kind: str) -> int:
    if not isinstance(value, str) or (match := pattern.fullmatch(value)) is None:
        raise ValueError(f"invalid compact {kind} ID: {value!r}")
    return int(match.group("index"))


def _target_value_summary(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {"kind": "object", "keys": tuple(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {"kind": "list", "length": len(value)}
    return value


def _line_ranges(line_ids: Sequence[str]) -> tuple[str, ...]:
    numbers = tuple(int(line_id[1:]) for line_id in line_ids)
    if numbers != tuple(sorted(set(numbers))):
        raise ValueError("literal review line IDs must be sorted and unique")
    if not numbers:
        raise ValueError("staged critic requires literal review lines")
    ranges: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(f"L{start:05d}" if start == previous else f"L{start:05d}-L{previous:05d}")
        start = previous = number
    ranges.append(f"L{start:05d}" if start == previous else f"L{start:05d}-L{previous:05d}")
    return tuple(ranges)


def _slim_annotated_source(value: str) -> str:
    value = re.sub(r"⟦binding_0*([0-9]+)⟧", lambda match: f"⟦B{int(match.group(1))}⟧", value)
    return value.replace("⟦/binding⟧", "⟦/B⟧")


def _compact_record_tables(
    payload: Mapping[str, Any],
    tables: Mapping[str, tuple[str, tuple[str, ...]]],
) -> dict[str, Any]:
    """Replace repeated object keys with a lossless explicit column/row representation."""

    output: dict[str, Any] = {}
    for key, value in payload.items():
        specification = tables.get(key)
        if specification is None:
            output[key] = value
            continue
        table_key, columns = specification
        if not isinstance(value, (tuple, list)):
            raise ValueError(f"staged request table {key} is not a sequence")
        rows: list[tuple[Any, ...]] = []
        for record in value:
            if not isinstance(record, Mapping) or set(record) != set(columns):
                raise ValueError(f"staged request table {key} has an invalid record shape")
            rows.append(tuple(record[column] for column in columns))
        output[table_key] = {"columns": columns, "rows": tuple(rows)}
    return output


def _expand_record_tables(
    payload: Mapping[str, Any],
    tables: Mapping[str, tuple[str, tuple[str, ...]]],
) -> dict[str, Any]:
    """Testable inverse of `_compact_record_tables`; the live host retains the verbose state."""

    by_table_key = {
        table_key: (record_key, columns) for record_key, (table_key, columns) in tables.items()
    }
    output: dict[str, Any] = {}
    for key, value in payload.items():
        specification = by_table_key.get(key)
        if specification is None:
            output[key] = value
            continue
        record_key, expected_columns = specification
        if not isinstance(value, Mapping):
            raise ValueError(f"compact staged table {key} is not an object")
        columns = value.get("columns")
        rows = value.get("rows")
        if tuple(cast(Sequence[Any], columns)) != expected_columns or not isinstance(
            rows, (tuple, list)
        ):
            raise ValueError(f"compact staged table {key} has an invalid contract")
        materialized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, (tuple, list)) or len(row) != len(expected_columns):
                raise ValueError(f"compact staged table {key} has an invalid row")
            materialized.append(dict(zip(expected_columns, row, strict=True)))
        output[record_key] = materialized
    return output


def compact_staged_audit_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize the semantic audit view without redundant component-path copies."""

    bindings = payload.get("bindings")
    if not isinstance(bindings, (tuple, list)):
        raise ValueError("staged audit lacks bindings")
    projected = dict(payload)
    projected["bindings"] = tuple(
        {column: row[column] for column in _AUDIT_REQUEST_BINDING_COLUMNS}
        for row in bindings
        if isinstance(row, Mapping)
    )
    if len(cast(Sequence[Any], projected["bindings"])) != len(bindings):
        raise ValueError("staged audit binding row is invalid")
    return _compact_record_tables(projected, _AUDIT_REQUEST_TABLES)


def expand_staged_audit_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the provider-visible projected audit view for contract tests."""

    return _expand_record_tables(payload, _AUDIT_REQUEST_TABLES)


def compact_staged_plan_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize a lossless transaction slice and remove one provably redundant vocabulary."""

    allowed = payload.get("allowedRemovalLogicalKeys")
    bindings = payload.get("bindings")
    if not isinstance(allowed, (tuple, list)) or not isinstance(bindings, (tuple, list)):
        raise ValueError("staged plan lacks its removal vocabulary or binding slice")
    binding_keys = [row.get("logicalKey") for row in bindings if isinstance(row, Mapping)]
    if len(binding_keys) != len(bindings) or tuple(allowed) != tuple(binding_keys):
        raise ValueError("staged plan removal vocabulary differs from its binding slice")
    return _compact_record_tables(
        {key: value for key, value in payload.items() if key != "allowedRemovalLogicalKeys"},
        _PLAN_TABLES,
    )


def expand_staged_plan_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Recover a host transaction slice and its derived removal vocabulary for tests."""

    expanded = _expand_record_tables(payload, _PLAN_TABLES)
    bindings = expanded.get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("expanded staged plan lacks bindings")
    expanded["allowedRemovalLogicalKeys"] = [row["logicalKey"] for row in bindings]
    return expanded


def build_staged_audit_payload(compact_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a fixed-schema semantic audit view without provider-irrelevant edit machinery."""

    from .host import _resolve_target_path

    target_table = compact_payload.get("targetPathTable")
    source_label = compact_payload.get("sourceLabel")
    binding_rows = compact_payload.get("bindingRows")
    occurrence_rows = compact_payload.get("occurrenceRows")
    annotated_source = compact_payload.get("annotatedSource")
    literal_ids = compact_payload.get("literalLineReviewIds")
    review_candidates = compact_payload.get("reviewCandidates")
    remaining_risks = compact_payload.get("remainingRiskCandidates")
    if not isinstance(target_table, (tuple, list)) or not isinstance(source_label, Mapping):
        raise ValueError("staged audit lacks target facts")
    if not isinstance(binding_rows, (tuple, list)) or not isinstance(
        occurrence_rows, (tuple, list)
    ):
        raise ValueError("staged audit lacks binding inventory")
    if not isinstance(annotated_source, str) or not isinstance(literal_ids, (tuple, list)):
        raise ValueError("staged audit lacks its annotated source or literal line ledger")
    if not isinstance(review_candidates, (tuple, list)) or not isinstance(
        remaining_risks, (tuple, list)
    ):
        raise ValueError("staged audit lacks candidate ledgers")

    path_indexes: dict[str, int] = {}
    target_facts: list[dict[str, Any]] = []
    for expected_index, row in enumerate(target_table):
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("staged target-path row is invalid")
        path_index = _id_index(row[0], pattern=_PATH_ID, kind="target path")
        if path_index != expected_index or not isinstance(row[1], str):
            raise ValueError("staged target-path table is not contiguous")
        path = row[1]
        path_indexes[path] = path_index
        target_facts.append(
            {
                "targetPathIndex": path_index,
                "targetPath": path,
                "sourceValue": _target_value_summary(_resolve_target_path(source_label, path)),
            }
        )

    slim_occurrences: list[dict[str, Any]] = []
    occurrence_binding: dict[str, int] = {}
    for expected_index, row in enumerate(occurrence_rows):
        if not isinstance(row, (tuple, list)) or len(row) != len(OCCURRENCE_COLUMNS):
            raise ValueError("staged occurrence row is invalid")
        materialized = dict(zip(OCCURRENCE_COLUMNS, row, strict=True))
        occurrence_id = materialized["occurrenceId"]
        if not isinstance(occurrence_id, str):
            raise ValueError("staged occurrence ID is invalid")
        occurrence_match = _OCCURRENCE_ID.fullmatch(occurrence_id)
        if (
            occurrence_match is None
            or int(occurrence_match.group("index")) != expected_index
            or int(occurrence_match.group("line")) != materialized["lineStartNumber"]
        ):
            raise ValueError("staged occurrence table is not contiguous")
        slim_occurrences.append(
            {
                "occurrenceRowIndex": expected_index,
                "lineStart": materialized["lineStartNumber"],
                "lineEnd": materialized["lineEndNumber"],
                "sourceText": materialized["sourceText"],
                "occurrenceIndex": materialized["occurrenceIndex"],
            }
        )

    slim_bindings: list[dict[str, Any]] = []
    logical_key_indexes: dict[str, int] = {}
    for expected_index, row in enumerate(binding_rows):
        if not isinstance(row, (tuple, list)) or len(row) != len(BINDING_COLUMNS):
            raise ValueError("staged binding row is invalid")
        materialized = dict(zip(BINDING_COLUMNS, row, strict=True))
        binding_index = _id_index(materialized["bindingId"], pattern=_BINDING_ID, kind="binding")
        if binding_index != expected_index:
            raise ValueError("staged binding table is not contiguous")
        logical_key = materialized["logicalKey"]
        if not isinstance(logical_key, str) or logical_key in logical_key_indexes:
            raise ValueError("staged binding logical keys are invalid")
        logical_key_indexes[logical_key] = binding_index
        path_ids = cast(Sequence[str], materialized["targetPathIds"])
        dependency_path_ids = cast(Sequence[str], materialized["dependencyPathIds"])
        occurrence_ids = cast(Sequence[str], materialized["occurrenceIds"])
        occurrence_indexes: list[int] = []
        for occurrence_id in occurrence_ids:
            occurrence_index = _id_index(
                occurrence_id,
                pattern=_OCCURRENCE_ID,
                kind="occurrence",
            )
            prior = occurrence_binding.setdefault(occurrence_id, binding_index)
            if prior != binding_index:
                raise ValueError("staged occurrence has multiple owners")
            occurrence_indexes.append(occurrence_index)
        slim_bindings.append(
            {
                "bindingIndex": binding_index,
                "logicalKey": logical_key,
                "renderMode": materialized["renderMode"],
                "valueKind": materialized["valueKind"],
                "groupKind": materialized["groupKind"],
                "groupKey": materialized["groupKey"],
                "targetPathIndexes": [
                    _id_index(value, pattern=_PATH_ID, kind="target path") for value in path_ids
                ],
                "targetPaths": [
                    cast(str, target_table[index][1])
                    for index in (
                        _id_index(value, pattern=_PATH_ID, kind="target path") for value in path_ids
                    )
                ],
                "targetRelationship": materialized["targetRelationship"],
                "independentTargetFactComponents": [
                    [_id_index(value, pattern=_PATH_ID, kind="target path") for value in component]
                    for component in materialized["independentTargetFactComponentPathIds"]
                ],
                "independentTargetFactComponentPaths": [
                    [
                        cast(
                            str,
                            target_table[_id_index(value, pattern=_PATH_ID, kind="target path")][1],
                        )
                        for value in component
                    ]
                    for component in materialized["independentTargetFactComponentPathIds"]
                ],
                "derivation": materialized["derivation"],
                "dependencyPathIndexes": [
                    _id_index(value, pattern=_PATH_ID, kind="target path")
                    for value in dependency_path_ids
                ],
                "dependencyPaths": [
                    cast(str, target_table[index][1])
                    for index in (
                        _id_index(value, pattern=_PATH_ID, kind="target path")
                        for value in dependency_path_ids
                    )
                ],
                "dependencyBindings": materialized["dependencyBindings"],
                "occurrenceIndexes": occurrence_indexes,
            }
        )
    if len(occurrence_binding) != len(slim_occurrences):
        raise ValueError("staged occurrence inventory is incomplete")

    candidate_rows: list[dict[str, Any]] = []
    # Put host-proven mandatory revisions before optional semantic leads. This is both a stable
    # priority contract and a guard against losing required evidence late in a large document.
    for required, rows in ((True, remaining_risks), (False, review_candidates)):
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("staged candidate row is invalid")
            candidate_id = row.get("risk_id") if required else row.get("candidateId")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("staged candidate row lacks its exact host-issued ID")
            logical_keys = row.get("logicalKeys", ())
            compacted_fields = {
                "candidateId",
                "kind",
                "logicalKeys",
                "lineIds",
                "sourceTexts",
                "context",
                "line_id",
                "source_text",
                "details",
            }
            supplied_details = row.get("details", {})
            if not isinstance(supplied_details, Mapping):
                raise ValueError("staged candidate details must be an object")
            details = {key: value for key, value in row.items() if key not in compacted_fields}
            details.update(supplied_details)
            binding_indexes = tuple(
                logical_key_indexes[key]
                for key in logical_keys
                if isinstance(key, str) and key in logical_key_indexes
            )
            possible_dependencies = supplied_details.get("possibleDependencyLogicalKeys", ())
            if (
                row.get("kind") == "potential_country_code_derivation"
                and isinstance(possible_dependencies, (tuple, list))
                and possible_dependencies
            ):
                details["derivationInputBindingIndexes"] = tuple(
                    logical_key_indexes[key]
                    for key in possible_dependencies
                    if isinstance(key, str) and key in logical_key_indexes
                )
            effective_required = required or row.get("requiredRevision") is True
            candidate_rows.append(
                {
                    "candidateIndex": len(candidate_rows),
                    "candidateId": candidate_id,
                    "kind": row.get("kind"),
                    "requiredRevision": effective_required,
                    "bindingIndexes": binding_indexes,
                    "lineIds": row.get("lineIds", (row.get("line_id"),)),
                    "sourceTexts": row.get("sourceTexts", (row.get("source_text"),)),
                    "context": row.get("context"),
                    "details": details,
                }
            )

    # The deterministic risk detector and the semantic review inventory intentionally have
    # different recall goals.  They can therefore describe the same physical unowned repeat.
    # Presenting both rows to separate facets previously caused two independent findings and a
    # larger repair plan for one defect.  Merge only an exact, uniquely matched risk into the
    # richer repeat candidate; ambiguity remains as two independently auditable rows.
    removed_candidate_indexes: set[int] = set()
    merged_required_evidence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    optional_repeat_rows = tuple(
        row
        for row in candidate_rows
        if row["requiredRevision"] is False and row["kind"] == "unowned_exact_repeat"
    )
    for required_row in tuple(row for row in candidate_rows if row["requiredRevision"] is True):
        required_lines = set(cast(Sequence[str], required_row["lineIds"]))
        required_surfaces = set(cast(Sequence[str], required_row["sourceTexts"]))
        matches = tuple(
            row
            for row in optional_repeat_rows
            if required_lines <= set(cast(Sequence[str], row["lineIds"]))
            and required_surfaces <= set(cast(Sequence[str], row["sourceTexts"]))
        )
        if len(matches) != 1:
            continue
        matched = matches[0]
        removed_candidate_indexes.add(cast(int, required_row["candidateIndex"]))
        merged_required_evidence[cast(int, matched["candidateIndex"])].append(
            {
                "kind": required_row["kind"],
                "lineIds": required_row["lineIds"],
                "sourceTexts": required_row["sourceTexts"],
                "details": required_row["details"],
            }
        )
    merged_candidates: list[dict[str, Any]] = []
    for row in candidate_rows:
        old_index = cast(int, row["candidateIndex"])
        if old_index in removed_candidate_indexes:
            continue
        candidate = dict(row)
        evidence = merged_required_evidence.get(old_index)
        if evidence:
            candidate["requiredRevision"] = True
            details = dict(cast(Mapping[str, Any], candidate["details"]))
            details["mergedRequiredRiskEvidence"] = tuple(evidence)
            candidate["details"] = details
        candidate["candidateIndex"] = len(merged_candidates)
        merged_candidates.append(candidate)
    candidate_rows = merged_candidates
    candidate_ids = tuple(cast(str, row["candidateId"]) for row in candidate_rows)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("staged candidate IDs are not unique")

    required_cobindings: list[dict[str, Any]] = []
    for co_binding_index, row in enumerate(
        cast(Sequence[Mapping[str, Any]], compact_payload.get("requiredTargetCoBindings", ()))
    ):
        required_cobindings.append(
            {
                "coBindingIndex": co_binding_index,
                "relationship": row["relationship"],
                "targetPathIndexes": tuple(path_indexes[path] for path in row["targetPaths"]),
                "targetPaths": tuple(row["targetPaths"]),
            }
        )
    semantic_only = []
    for semantic_only_index, row in enumerate(
        cast(Sequence[Mapping[str, Any]], compact_payload.get("semanticOnlyTargetFacts", ()))
    ):
        semantic_only.append(
            {
                "semanticOnlyIndex": semantic_only_index,
                "targetPathIndex": path_indexes[row["targetPath"]],
                "sourceValue": row["sourceValue"],
                "provenance": row["provenance"],
                "reason": row["reason"],
            }
        )
    payload: dict[str, Any] = {
        "contract": {
            "schemaVersion": 4,
            "allExplicitIndexesAreZeroBased": True,
            "annotatedMarkerFormat": "B{bindingIndex}",
        },
        "documentId": compact_payload.get("documentId"),
        "expectedCarrierName": compact_payload.get("expectedCarrierName"),
        "documentMetadata": compact_payload.get("documentMetadata"),
        "targetFacts": target_facts,
        "requiredTargetCoBindings": required_cobindings,
        "bindings": slim_bindings,
        "occurrences": slim_occurrences,
        "semanticOnlyTargetFacts": semantic_only,
        "candidateRows": candidate_rows,
        "literalLineRanges": _line_ranges(cast(Sequence[str], literal_ids)),
        "annotatedSource": _slim_annotated_source(annotated_source),
    }
    payload["stateRevision"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


_CARGO_GROUP_KINDS = {
    "cargo",
    "dangerous_goods",
    "equipment",
    "package",
    "temperature",
}
_LEGACY_SURFACE_CANDIDATE_KINDS = {
    "semantic_only_evidence_review",
    "domain_vocabulary_surface_review",
    "unowned_repeated_literal_surface",
    "unowned_standalone_document_status",
}
_ISOLATED_SURFACE_CANDIDATE_KINDS = {"unowned_standalone_document_status"}


def _indexed_rows(rows: Any, *, index_key: str, kind: str) -> dict[int, Mapping[str, Any]]:
    if not isinstance(rows, (tuple, list)):
        raise ValueError(f"staged {kind} rows are not a sequence")
    output: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"staged {kind} row is not an object")
        index = row.get(index_key)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index in output:
            raise ValueError(f"staged {kind} indexes are invalid")
        output[index] = row
    return output


def _nested_target_paths(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"targetPath", "targetPaths", "dependencyPaths"}:
                if isinstance(child, str):
                    output.add(child)
                elif isinstance(child, (tuple, list)):
                    output.update(item for item in child if isinstance(item, str))
            output.update(_nested_target_paths(child))
    elif isinstance(value, (tuple, list)):
        for child in value:
            output.update(_nested_target_paths(child))
    return output


def _expanded_line_ranges(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError("staged literal line ranges are not a sequence")
    output: list[str] = []
    for value in values:
        if (
            not isinstance(value, str)
            or (match := re.fullmatch(r"L(?P<start>[0-9]{5})(?:-L(?P<end>[0-9]{5}))?", value))
            is None
        ):
            raise ValueError("staged literal line range is invalid")
        start = int(match.group("start"))
        end = int(match.group("end") or match.group("start"))
        if end < start:
            raise ValueError("staged literal line range is reversed")
        output.extend(f"L{number:05d}" for number in range(start, end + 1))
    if len(set(output)) != len(output):
        raise ValueError("staged literal line ranges overlap")
    return tuple(output)


def _audit_facet_slice(
    *,
    payload: Mapping[str, Any],
    facet: AuditFacetName,
    assigned_binding_indexes: set[int],
    candidate_rows: Sequence[Mapping[str, Any]],
    include_complete_state: bool,
    assigned_literal_line_ids: Sequence[str] | None = None,
    crop_annotated_source: bool = False,
    candidate_only: bool = False,
) -> dict[str, Any]:
    bindings_by_index = _indexed_rows(
        payload.get("bindings"), index_key="bindingIndex", kind="binding"
    )
    occurrences_by_index = _indexed_rows(
        payload.get("occurrences"), index_key="occurrenceRowIndex", kind="occurrence"
    )
    targets_by_index = _indexed_rows(
        payload.get("targetFacts"), index_key="targetPathIndex", kind="target fact"
    )
    target_index_by_path = {
        cast(str, row["targetPath"]): index for index, row in targets_by_index.items()
    }
    candidate_binding_indexes = {
        index
        for row in candidate_rows
        for index in cast(Sequence[int], row.get("bindingIndexes", ()))
    }
    selected_binding_indexes = (
        set(bindings_by_index)
        if include_complete_state
        else assigned_binding_indexes | candidate_binding_indexes
    )
    if candidate_only:
        candidate_groups = {
            (
                cast(str, bindings_by_index[index]["groupKind"]),
                cast(str, bindings_by_index[index]["groupKey"]),
            )
            for index in candidate_binding_indexes
        }
        selected_binding_indexes.update(
            index
            for index, row in bindings_by_index.items()
            if (cast(str, row["groupKind"]), cast(str, row["groupKey"])) in candidate_groups
        )
        if any(
            row.get("kind")
            in {
                "potential_calculated_total_derivation",
                "potential_equipment_receipt_derivation",
            }
            for row in candidate_rows
        ):
            selected_binding_indexes.update(
                index
                for index, row in bindings_by_index.items()
                if row.get("groupKind") in _CARGO_GROUP_KINDS
            )
        logical_key_indexes = {
            cast(str, row["logicalKey"]): index for index, row in bindings_by_index.items()
        }
        pending = list(selected_binding_indexes)
        while pending:
            index = pending.pop()
            for logical_key in cast(
                Sequence[str], bindings_by_index[index].get("dependencyBindings", ())
            ):
                dependency_index = logical_key_indexes.get(logical_key)
                if (
                    dependency_index is not None
                    and dependency_index not in selected_binding_indexes
                ):
                    selected_binding_indexes.add(dependency_index)
                    pending.append(dependency_index)
    selected_bindings = [bindings_by_index[index] for index in sorted(selected_binding_indexes)]
    selected_occurrence_indexes = {
        index
        for row in selected_bindings
        for index in cast(Sequence[int], row.get("occurrenceIndexes", ()))
    }
    selected_target_indexes = {
        index
        for row in selected_bindings
        for key in ("targetPathIndexes", "dependencyPathIndexes")
        for index in cast(Sequence[int], row.get(key, ()))
    }
    selected_target_indexes.update(
        index
        for row in selected_bindings
        for component in cast(
            Sequence[Sequence[int]], row.get("independentTargetFactComponents", ())
        )
        for index in component
    )
    selected_target_indexes.update(
        target_index_by_path[path]
        for row in candidate_rows
        for path in _nested_target_paths(row)
        if path in target_index_by_path
    )
    if include_complete_state:
        selected_target_indexes = set(targets_by_index)

    co_bindings = cast(Sequence[Mapping[str, Any]], payload.get("requiredTargetCoBindings", ()))
    changed = True
    while changed:
        changed = False
        for row in co_bindings:
            component = set(cast(Sequence[int], row.get("targetPathIndexes", ())))
            if component & selected_target_indexes and not component <= selected_target_indexes:
                selected_target_indexes.update(component)
                changed = True

    selected_semantic_only = [
        row
        for row in cast(Sequence[Mapping[str, Any]], payload.get("semanticOnlyTargetFacts", ()))
        if include_complete_state or cast(int, row["targetPathIndex"]) in selected_target_indexes
    ]
    group_contexts: dict[tuple[str, str], dict[str, set[int]]] = {}
    for row in selected_bindings:
        key = (cast(str, row["groupKind"]), cast(str, row["groupKey"]))
        context = group_contexts.setdefault(
            key,
            {"bindings": set(), "targets": set(), "lines": set()},
        )
        context["bindings"].add(cast(int, row["bindingIndex"]))
        context["targets"].update(cast(Sequence[int], row.get("targetPathIndexes", ())))
        for occurrence_index in cast(Sequence[int], row.get("occurrenceIndexes", ())):
            occurrence = occurrences_by_index[occurrence_index]
            context["lines"].update(
                range(
                    cast(int, occurrence["lineStart"]),
                    cast(int, occurrence["lineEnd"]) + 1,
                )
            )
    semantic_group_contexts = [
        {
            "groupKind": group_kind,
            "groupKey": group_key,
            "bindingIndexes": tuple(sorted(context["bindings"])),
            "targetPathIndexes": tuple(sorted(context["targets"])),
            "ownedLineRanges": _line_ranges(
                tuple(f"L{line:05d}" for line in sorted(context["lines"]))
            ),
            "lineEnvelope": (f"L{min(context['lines']):05d}-L{max(context['lines']):05d}"),
        }
        for (group_kind, group_key), context in sorted(group_contexts.items())
        if context["lines"]
    ]
    all_literal_line_ids = _expanded_line_ranges(payload.get("literalLineRanges"))
    if assigned_literal_line_ids is not None:
        selected_literal_line_ids = tuple(assigned_literal_line_ids)
    elif facet == "surface_completeness":
        selected_literal_line_ids = all_literal_line_ids
    else:
        scope_lines = {
            line
            for occurrence_index in selected_occurrence_indexes
            for line in range(
                cast(int, occurrences_by_index[occurrence_index]["lineStart"]) - 1,
                cast(int, occurrences_by_index[occurrence_index]["lineEnd"]) + 2,
            )
            if line > 0
        }
        scope_lines.update(
            line
            for row in candidate_rows
            for line_id in cast(Sequence[str], row.get("lineIds", ()))
            for line in range(int(line_id[1:]) - 1, int(line_id[1:]) + 2)
            if line > 0
        )
        selected_literal_line_ids = tuple(
            line_id for line_id in all_literal_line_ids if int(line_id[1:]) in scope_lines
        )
    if any(line_id not in set(all_literal_line_ids) for line_id in selected_literal_line_ids):
        raise ValueError("facet literal assignment is outside the full review ledger")
    source_context_line_ids: set[str] = set(selected_literal_line_ids)
    raw_annotated_source = payload.get("annotatedSource")
    if not isinstance(raw_annotated_source, str):
        raise ValueError("critic payload annotated source must be text")
    if crop_annotated_source:
        source_lines = _annotated_lines(raw_annotated_source)
        context_numbers = {
            line
            for occurrence_index in selected_occurrence_indexes
            for line in range(
                cast(int, occurrences_by_index[occurrence_index]["lineStart"]) - 2,
                cast(int, occurrences_by_index[occurrence_index]["lineEnd"]) + 3,
            )
            if line in source_lines
        }
        context_numbers.update(
            line
            for row in candidate_rows
            for line_id in cast(Sequence[str], row.get("lineIds", ()))
            for line in range(int(line_id[1:]) - 2, int(line_id[1:]) + 3)
            if line in source_lines
        )
        context_numbers.update(
            line
            for line_id in selected_literal_line_ids
            for line in range(int(line_id[1:]) - 1, int(line_id[1:]) + 2)
            if line in source_lines
        )
        source_context_line_ids.update(f"L{line:05d}" for line in context_numbers)
        annotated_source = "\n".join(
            source_lines[int(line_id[1:])] for line_id in sorted(source_context_line_ids)
        )
    else:
        annotated_source = raw_annotated_source
    contract = dict(cast(Mapping[str, Any], payload.get("contract", {})))
    contract["schemaVersion"] = 6 if candidate_only else 5
    output = {
        "contract": contract,
        "documentId": payload.get("documentId"),
        "expectedCarrierName": payload.get("expectedCarrierName"),
        "documentMetadata": payload.get("documentMetadata"),
        "auditFacet": {
            "name": facet,
            "assignedBindingIndexes": tuple(sorted(assigned_binding_indexes)),
            "contextBindingIndexes": tuple(
                sorted(selected_binding_indexes - assigned_binding_indexes)
            ),
            "assignedCandidateIndexes": tuple(
                cast(int, row["candidateIndex"]) for row in candidate_rows
            ),
            "assignedTargetPathIndexes": tuple(sorted(selected_target_indexes)),
            "completeStateTablesIncluded": include_complete_state,
            "reviewScope": "candidate_prepass" if candidate_only else "full_certification",
        },
        "targetFacts": [targets_by_index[index] for index in sorted(selected_target_indexes)],
        "requiredTargetCoBindings": [
            row
            for row in co_bindings
            if set(cast(Sequence[int], row.get("targetPathIndexes", ()))) & selected_target_indexes
        ],
        "bindings": selected_bindings,
        "occurrences": [
            occurrences_by_index[index] for index in sorted(selected_occurrence_indexes)
        ],
        "semanticOnlyTargetFacts": selected_semantic_only,
        "candidateRows": list(candidate_rows),
        "semanticGroupContexts": semantic_group_contexts,
        "literalLineRanges": (
            _line_ranges(selected_literal_line_ids) if selected_literal_line_ids else ()
        ),
        "annotatedSource": annotated_source,
        "stateRevision": payload.get("stateRevision"),
    }
    output["facetRevision"] = sha256_bytes(canonical_json_bytes(output))
    return output


def build_staged_audit_facet_payloads(
    payload: Mapping[str, Any],
    *,
    partition_literal_review: bool = False,
    candidate_only: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Partition one immutable state into concurrent semantic audit facets.

    Candidate-only facets are a pre-certification remediation pass. They expose every host-issued
    candidate and its directly referenced context, but no unrelated binding or literal inventory.
    They can propose a repair and can never stand in for the later full-certification facets.
    """

    if candidate_only and not partition_literal_review:
        raise ValueError("candidate-only audit requires partitioned literal review")

    bindings_by_index = _indexed_rows(
        payload.get("bindings"), index_key="bindingIndex", kind="binding"
    )
    candidate_rows = tuple(
        _indexed_rows(
            payload.get("candidateRows"), index_key="candidateIndex", kind="candidate"
        ).values()
    )
    cargo_binding_indexes = {
        index
        for index, row in bindings_by_index.items()
        if row.get("groupKind") in _CARGO_GROUP_KINDS
    }
    document_binding_indexes = set(bindings_by_index) - cargo_binding_indexes
    occurrences_by_index = _indexed_rows(
        payload.get("occurrences"), index_key="occurrenceRowIndex", kind="occurrence"
    )

    def owned_lines(binding_indexes: set[int]) -> set[int]:
        return {
            line
            for binding_index in binding_indexes
            for occurrence_index in cast(
                Sequence[int], bindings_by_index[binding_index].get("occurrenceIndexes", ())
            )
            for line in range(
                cast(int, occurrences_by_index[occurrence_index]["lineStart"]),
                cast(int, occurrences_by_index[occurrence_index]["lineEnd"]) + 1,
            )
        }

    cargo_owned_lines = owned_lines(cargo_binding_indexes)
    document_owned_lines = owned_lines(document_binding_indexes)

    def nearby_topology(candidate: Mapping[str, Any]) -> AuditFacetName | None:
        candidate_lines = {
            int(line_id[1:]) for line_id in cast(Sequence[str], candidate.get("lineIds", ()))
        }
        if not candidate_lines:
            return None

        def proximity(owned: set[int]) -> tuple[int, int | None]:
            if not owned:
                return 0, None
            distances = tuple(
                min(abs(candidate_line - owned_line) for owned_line in owned)
                for candidate_line in candidate_lines
            )
            return sum(distance <= 1 for distance in distances), min(distances)

        cargo_score, cargo_distance = proximity(cargo_owned_lines)
        document_score, document_distance = proximity(document_owned_lines)
        if cargo_score > document_score:
            return "cargo_topology"
        if document_score > cargo_score:
            return "document_topology"
        if cargo_score and cargo_distance is not None and document_distance is not None:
            if cargo_distance < document_distance:
                return "cargo_topology"
            if document_distance < cargo_distance:
                return "document_topology"
        return None

    surface_candidates: list[Mapping[str, Any]] = []
    cargo_candidates: list[Mapping[str, Any]] = []
    document_candidates: list[Mapping[str, Any]] = []
    for row in candidate_rows:
        if (
            not partition_literal_review and row.get("kind") in _LEGACY_SURFACE_CANDIDATE_KINDS
        ) or (partition_literal_review and row.get("kind") in _ISOLATED_SURFACE_CANDIDATE_KINDS):
            surface_candidates.append(row)
            continue
        binding_indexes = set(cast(Sequence[int], row.get("bindingIndexes", ())))
        target_paths = _nested_target_paths(row)
        cargo_target = any(
            path.startswith(
                (
                    "documentPatch.cargo",
                    "documentPatch.containers",
                    "documentPatch.dangerousGoods",
                    "documentPatch.temperature",
                )
            )
            for path in target_paths
        )
        if binding_indexes & cargo_binding_indexes or cargo_target:
            cargo_candidates.append(row)
        elif binding_indexes or target_paths:
            document_candidates.append(row)
        elif not partition_literal_review and row.get("requiredRevision") is True:
            surface_candidates.append(row)
        elif (topology := nearby_topology(row)) == "cargo_topology":
            cargo_candidates.append(row)
        elif topology == "document_topology":
            document_candidates.append(row)
        else:
            surface_candidates.append(row)

    all_literal_line_ids = _expanded_line_ranges(payload.get("literalLineRanges"))
    surface_candidate_lines = {
        line_id
        for row in surface_candidates
        for line_id in cast(Sequence[str], row.get("lineIds", ()))
        if line_id in set(all_literal_line_ids)
    }
    cargo_candidate_lines = {
        line_id
        for row in cargo_candidates
        for line_id in cast(Sequence[str], row.get("lineIds", ()))
        if line_id in set(all_literal_line_ids)
    }
    document_candidate_lines = {
        line_id
        for row in document_candidates
        for line_id in cast(Sequence[str], row.get("lineIds", ()))
        if line_id in set(all_literal_line_ids)
    }
    cargo_context_lines: set[str] = set(cargo_candidate_lines)
    document_context_lines: set[str] = set(document_candidate_lines)
    if partition_literal_review and not candidate_only:
        for line_id in all_literal_line_ids:
            if line_id in cargo_candidate_lines or line_id in document_candidate_lines:
                continue
            candidate = {"lineIds": (line_id,)}
            topology = nearby_topology(candidate)
            if topology == "cargo_topology":
                cargo_context_lines.add(line_id)
            elif topology == "document_topology":
                document_context_lines.add(line_id)
    topology_literal_lines = cargo_context_lines | document_context_lines
    surface_literal_lines = (
        tuple(sorted(surface_candidate_lines))
        if candidate_only
        else tuple(
            line
            for line in all_literal_line_ids
            if line not in topology_literal_lines or line in surface_candidate_lines
        )
        if partition_literal_review
        else None
    )

    facets: list[dict[str, Any]] = []
    if (
        (not candidate_only and not partition_literal_review)
        or surface_candidates
        or (surface_literal_lines and not candidate_only)
    ):
        facets.append(
            _audit_facet_slice(
                payload=payload,
                facet="surface_completeness",
                assigned_binding_indexes=set(),
                candidate_rows=surface_candidates,
                include_complete_state=False,
                assigned_literal_line_ids=surface_literal_lines,
                crop_annotated_source=partition_literal_review,
                candidate_only=candidate_only,
            )
        )
    if (cargo_binding_indexes and not candidate_only) or cargo_candidates:
        facets.append(
            _audit_facet_slice(
                payload=payload,
                facet="cargo_topology",
                assigned_binding_indexes=set() if candidate_only else cargo_binding_indexes,
                candidate_rows=cargo_candidates,
                include_complete_state=False,
                assigned_literal_line_ids=(
                    tuple(sorted(cargo_context_lines)) if partition_literal_review else None
                ),
                crop_annotated_source=partition_literal_review,
                candidate_only=candidate_only,
            )
        )
    if (document_binding_indexes and not candidate_only) or document_candidates:
        facets.append(
            _audit_facet_slice(
                payload=payload,
                facet="document_topology",
                assigned_binding_indexes=set() if candidate_only else document_binding_indexes,
                candidate_rows=document_candidates,
                include_complete_state=False,
                assigned_literal_line_ids=(
                    tuple(sorted(document_context_lines)) if partition_literal_review else None
                ),
                crop_annotated_source=partition_literal_review,
                candidate_only=candidate_only,
            )
        )

    if not facets:
        raise ValueError("candidate-only audit has no host-issued candidates")

    candidate_partitions = [
        cast(int, row["candidateIndex"])
        for facet_payload in facets
        for row in cast(Sequence[Mapping[str, Any]], facet_payload["candidateRows"])
    ]
    expected_candidate_indexes = {cast(int, row["candidateIndex"]) for row in candidate_rows}
    if (
        len(candidate_partitions) != len(set(candidate_partitions))
        or set(candidate_partitions) != expected_candidate_indexes
    ):
        raise ValueError("staged audit candidate facets are not an exact partition")
    assigned_topology_bindings = {
        index
        for facet_payload in facets
        if cast(Mapping[str, Any], facet_payload["auditFacet"])["name"]
        in {"cargo_topology", "document_topology"}
        for index in cast(
            Sequence[int],
            cast(Mapping[str, Any], facet_payload["auditFacet"])["assignedBindingIndexes"],
        )
    }
    if not candidate_only and assigned_topology_bindings != set(bindings_by_index):
        raise ValueError("staged topology facets do not cover every binding exactly once")
    if partition_literal_review:
        literal_partitions = [
            line_id
            for facet_payload in facets
            for line_id in _expanded_line_ranges(facet_payload["literalLineRanges"])
        ]
        expected_literal_lines = (
            {
                line_id
                for row in candidate_rows
                for line_id in cast(Sequence[str], row.get("lineIds", ()))
                if line_id in set(all_literal_line_ids)
            }
            if candidate_only
            else set(all_literal_line_ids)
        )
        if set(literal_partitions) != expected_literal_lines:
            raise ValueError("staged audit literal facets do not cover their required ledger")
    return tuple(facets)


def validate_staged_audit(
    output: StagedAuditOutput, payload: Mapping[str, Any]
) -> StagedAuditOutput:
    candidates_by_index = _indexed_rows(
        payload.get("candidateRows"), index_key="candidateIndex", kind="candidate"
    )
    finding_candidates = {
        index for finding in output.findings for index in finding.candidate_indexes
    }
    required_candidates = {
        index for index, row in candidates_by_index.items() if row.get("requiredRevision")
    }
    missing_required_candidates = required_candidates - finding_candidates
    if missing_required_candidates:
        raise ValueError(
            "audit omits proven unowned candidate findings: "
            + ", ".join(str(index) for index in sorted(missing_required_candidates))
        )

    literal_lines: list[str] = []
    for value in cast(Sequence[str], payload.get("literalLineRanges")):
        match = re.fullmatch(r"L(?P<start>[0-9]{5})(?:-L(?P<end>[0-9]{5}))?", value)
        if match is None:
            raise ValueError("audit payload contains an invalid literal line range")
        start = int(match.group("start"))
        end = int(match.group("end") or match.group("start"))
        if end < start:
            raise ValueError("audit payload literal line range is reversed")
        literal_lines.extend(f"L{number:05d}" for number in range(start, end + 1))
    if len(set(literal_lines)) != len(literal_lines):
        raise ValueError("audit payload literal line ranges overlap")

    bindings_by_index = _indexed_rows(
        payload.get("bindings"), index_key="bindingIndex", kind="binding"
    )
    occurrences_by_index = _indexed_rows(
        payload.get("occurrences"), index_key="occurrenceRowIndex", kind="occurrence"
    )
    targets_by_index = _indexed_rows(
        payload.get("targetFacts"), index_key="targetPathIndex", kind="target fact"
    )
    valid_lines = {
        f"L{int(match.group('number')):05d}"
        for row in str(payload.get("annotatedSource", "")).splitlines()
        if (match := _NUMBERED_LINE.match(row)) is not None
    }
    assigned_literal_lines = set(literal_lines)
    for finding_index, finding in enumerate(output.findings):
        outside_bindings = sorted(
            index for index in finding.binding_indexes if index not in bindings_by_index
        )
        if outside_bindings:
            raise ValueError(
                f"finding {finding_index} references bindings outside its facet: "
                f"{outside_bindings}; allowed={sorted(bindings_by_index)}"
            )
        outside_targets = sorted(
            index for index in finding.target_path_indexes if index not in targets_by_index
        )
        if outside_targets:
            raise ValueError(
                f"finding {finding_index} references target paths outside its facet: "
                f"{outside_targets}; allowed={sorted(targets_by_index)}"
            )
        outside_candidates = sorted(
            index for index in finding.candidate_indexes if index not in candidates_by_index
        )
        if outside_candidates:
            raise ValueError(
                f"finding {finding_index} references candidates outside its facet: "
                f"{outside_candidates}; allowed={sorted(candidates_by_index)}"
            )
        outside_lines = sorted(
            line_id for line_id in finding.line_ids if line_id not in valid_lines
        )
        if outside_lines:
            raise ValueError(
                f"finding {finding_index} cites lines outside its supplied source: {outside_lines}"
            )
        if finding.finding_kind.startswith("unowned_") and not (
            set(finding.line_ids) & assigned_literal_lines
        ):
            raise ValueError(
                f"finding {finding_index} ({finding.finding_kind}) cites no authorized literal "
                f"line; cited={list(finding.line_ids)}, "
                f"authorized_ranges={list(payload.get('literalLineRanges', ()))}. "
                "Remove the finding if those lines are context-only, or cite its assigned "
                "candidate evidence line when the same defect has an authorized surface."
            )
        for candidate_index in finding.candidate_indexes:
            candidate_lines = set(
                cast(Sequence[str], candidates_by_index[candidate_index].get("lineIds", ()))
            )
            if candidate_lines and not candidate_lines.intersection(finding.line_ids):
                raise ValueError(
                    f"finding {finding_index} for candidate {candidate_index} omits every "
                    f"candidate evidence line; cited={list(finding.line_ids)}, "
                    f"required_one_of={sorted(candidate_lines)}"
                )
        candidate_binding_indexes = {
            index
            for candidate_index in finding.candidate_indexes
            for index in cast(
                Sequence[int],
                candidates_by_index[candidate_index].get("bindingIndexes", ()),
            )
        }
        if finding.finding_kind == "missing_derivation":
            required_derivation_inputs = {
                index
                for candidate_index in finding.candidate_indexes
                for index in cast(
                    Sequence[int],
                    cast(
                        Mapping[str, Any],
                        candidates_by_index[candidate_index].get("details", {}),
                    ).get("derivationInputBindingIndexes", ()),
                )
            }
            missing_derivation_inputs = required_derivation_inputs - set(finding.binding_indexes)
            if missing_derivation_inputs:
                raise ValueError(
                    f"finding {finding_index} omits host-provided derivation input bindings: "
                    f"{sorted(missing_derivation_inputs)}"
                )
        if (
            finding.finding_kind == "unowned_repeated_fact"
            and candidate_binding_indexes
            and not finding.binding_indexes
        ):
            raise ValueError(
                f"finding {finding_index} for repeated candidate(s) "
                f"{list(finding.candidate_indexes)} omits an existing semantic owner; cite the "
                f"correct supplied binding. Candidate exact-match owner hints are "
                f"{sorted(candidate_binding_indexes)}, but another supplied facet binding is "
                "valid when local semantic context proves it is the owner."
            )
        if finding.binding_indexes:
            cited_numbers = {int(line_id[1:]) for line_id in finding.line_ids}
            for binding_index in finding.binding_indexes:
                occurrence_indexes = cast(
                    Sequence[int], bindings_by_index[binding_index].get("occurrenceIndexes", ())
                )
                binding_lines = {
                    line
                    for occurrence_index in occurrence_indexes
                    for line in range(
                        cast(int, occurrences_by_index[occurrence_index]["lineStart"]),
                        cast(int, occurrences_by_index[occurrence_index]["lineEnd"]) + 1,
                    )
                }
                if not binding_lines & cited_numbers:
                    raise ValueError(
                        f"finding {finding_index} cites binding {binding_index} outside all "
                        f"finding lines; cited={list(finding.line_ids)}, "
                        f"binding_lines={sorted(f'L{line:05d}' for line in binding_lines)}"
                    )
    return output


def validate_staged_facet_audit(
    output: StagedFacetAuditOutput, payload: Mapping[str, Any]
) -> StagedFacetAuditOutput:
    facet = payload.get("auditFacet")
    if not isinstance(facet, Mapping) or output.facet != facet.get("name"):
        raise ValueError("facet audit response names the wrong assigned facet")
    candidates_by_index = _indexed_rows(
        payload.get("candidateRows"), index_key="candidateIndex", kind="candidate"
    )
    dispositions = {row.candidate_index: row.conclusion for row in output.candidate_dispositions}
    if set(dispositions) != set(candidates_by_index):
        missing = sorted(set(candidates_by_index) - set(dispositions))
        extra = sorted(set(dispositions) - set(candidates_by_index))
        raise ValueError(f"facet candidate disposition mismatch; missing={missing}, extra={extra}")
    finding_candidates = {
        index for finding in output.findings for index in finding.candidate_indexes
    }
    defect_candidates = {
        index
        for index, conclusion in dispositions.items()
        if conclusion == "defect_requires_revision"
    }
    if finding_candidates != defect_candidates:
        raise ValueError(
            "facet candidate defects and finding selectors differ: "
            f"findings={sorted(finding_candidates)}, dispositions={sorted(defect_candidates)}"
        )
    required_candidates = {
        index for index, row in candidates_by_index.items() if row.get("requiredRevision") is True
    }
    if not required_candidates <= defect_candidates:
        raise ValueError("facet marks a host-proven unowned candidate as valid")
    translated = StagedAuditOutput(
        verdict=output.verdict,
        findings=output.findings,
        coverage=StagedAuditCoverage(
            literal_completeness_checked=True,
            target_ownership_checked=True,
            topology_and_grouping_checked=True,
            derivations_checked=True,
            carrier_boundary_checked=True,
            identifier_relationships_checked=True,
        ),
        rationale=output.rationale,
    )
    validate_staged_audit(translated, payload)
    return output


def normalize_staged_facet_audit(
    output: StagedFacetAuditOutput, payload: Mapping[str, Any]
) -> StagedFacetAuditOutput:
    """Canonicalize the facet's redundant candidate receipt without losing defects.

    Detailed findings are the auditable semantic decision. ``candidate_dispositions`` is a compact
    completeness receipt for the same decision and models occasionally transcribe the two lists
    inconsistently. The host therefore derives that redundant receipt from findings, while
    materializing every deterministic ``requiredRevision`` candidate as a finding if the model
    omitted it. Optional disposition-only flags cannot invent an unsupported defect.
    """

    candidates_by_index = _indexed_rows(
        payload.get("candidateRows"), index_key="candidateIndex", kind="candidate"
    )
    supplied_disposition_indexes = {row.candidate_index for row in output.candidate_dispositions}
    extra_dispositions = supplied_disposition_indexes - set(candidates_by_index)
    if extra_dispositions:
        raise ValueError(
            "facet candidate dispositions reference candidates outside the facet: "
            + ", ".join(str(index) for index in sorted(extra_dispositions))
        )
    finding_candidates = {
        index for finding in output.findings for index in finding.candidate_indexes
    }
    extra_finding_candidates = finding_candidates - set(candidates_by_index)
    if extra_finding_candidates:
        raise ValueError(
            "facet findings reference candidates outside the facet: "
            + ", ".join(str(index) for index in sorted(extra_finding_candidates))
        )

    findings = list(output.findings)
    required_candidates = {
        index for index, row in candidates_by_index.items() if row.get("requiredRevision") is True
    }
    for candidate_index in sorted(required_candidates - finding_candidates):
        candidate = candidates_by_index[candidate_index]
        line_ids = tuple(
            line_id
            for line_id in cast(Sequence[Any], candidate.get("lineIds", ()))
            if isinstance(line_id, str)
        )
        if not line_ids:
            raise ValueError(f"host-required candidate {candidate_index} has no evidence line IDs")
        source_texts = tuple(
            source_text
            for source_text in cast(Sequence[Any], candidate.get("sourceTexts", ()))
            if isinstance(source_text, str) and source_text.strip()
        )
        binding_indexes = tuple(
            index
            for index in cast(Sequence[Any], candidate.get("bindingIndexes", ()))
            if isinstance(index, int) and not isinstance(index, bool)
        )
        repeated = candidate.get("kind") == "unowned_exact_repeat" and bool(binding_indexes)
        if repeated:
            bindings_by_index = _indexed_rows(
                payload.get("bindings"), index_key="bindingIndex", kind="binding"
            )
            occurrences_by_index = _indexed_rows(
                payload.get("occurrences"),
                index_key="occurrenceRowIndex",
                kind="occurrence",
            )
            owner_line_ids = {
                f"L{line:05d}"
                for binding_index in binding_indexes
                for occurrence_index in cast(
                    Sequence[int],
                    bindings_by_index[binding_index].get("occurrenceIndexes", ()),
                )
                for line in range(
                    cast(int, occurrences_by_index[occurrence_index]["lineStart"]),
                    cast(int, occurrences_by_index[occurrence_index]["lineEnd"]) + 1,
                )
            }
            line_ids = tuple(
                sorted(set(line_ids) | owner_line_ids, key=lambda value: int(value[1:]))
            )
        candidate_id = candidate.get("candidateId")
        findings.append(
            StagedAuditFinding(
                finding_kind=("unowned_repeated_fact" if repeated else "unowned_shipment_fact"),
                line_ids=line_ids,
                evidence=" | ".join(source_texts) if source_texts else str(candidate_id),
                explanation=(
                    f"Host-proven required candidate {candidate_id} remains unowned; the repair "
                    "plan must resolve its supplied source evidence."
                ),
                binding_indexes=binding_indexes if repeated else (),
                candidate_indexes=(candidate_index,),
            )
        )
    normalized_finding_candidates = {
        index for finding in findings for index in finding.candidate_indexes
    }
    normalized_dispositions = tuple(
        StagedFacetCandidateDisposition(
            candidate_index=index,
            conclusion=(
                "defect_requires_revision"
                if index in normalized_finding_candidates
                else "valid_current_state"
            ),
        )
        for index in candidates_by_index
    )
    normalized_verdict = "revise" if findings else "pass"
    normalized = output.model_copy(
        update={
            "verdict": normalized_verdict,
            "findings": tuple(findings),
            "candidate_dispositions": normalized_dispositions,
        }
    )
    return validate_staged_facet_audit(normalized, payload)


def merge_staged_facet_audits(
    *,
    outputs: Sequence[StagedFacetAuditOutput],
    facet_payloads: Sequence[Mapping[str, Any]],
    full_payload: Mapping[str, Any],
) -> StagedAuditOutput:
    """Merge independently validated facet findings into one atomic planning audit."""

    if len(outputs) != len(facet_payloads) or not outputs:
        raise ValueError("staged facet audit outputs do not match their requests")
    expected_facets = {
        cast(str, cast(Mapping[str, Any], payload["auditFacet"])["name"])
        for payload in facet_payloads
    }
    if {output.facet for output in outputs} != expected_facets:
        raise ValueError("staged facet audit output names are incomplete or duplicated")
    for output, payload in zip(outputs, facet_payloads, strict=True):
        validate_staged_facet_audit(output, payload)

    unique_findings: dict[bytes, StagedAuditFinding] = {}
    for output in outputs:
        for finding in output.findings:
            unique_findings.setdefault(
                canonical_json_bytes(finding.model_dump(mode="json")), finding
            )
    findings = tuple(unique_findings.values())
    merged = StagedAuditOutput(
        verdict="revise" if findings else "pass",
        findings=findings,
        coverage=StagedAuditCoverage(
            literal_completeness_checked=True,
            target_ownership_checked=True,
            topology_and_grouping_checked=True,
            derivations_checked=True,
            carrier_boundary_checked=True,
            identifier_relationships_checked=True,
        ),
        rationale=" | ".join(f"{output.facet}: {output.rationale}" for output in outputs),
    )
    return validate_staged_audit(merged, full_payload)


def _base_finding(finding: StagedAuditFinding) -> CriticFinding:
    return CriticFinding(
        finding_kind=finding.finding_kind,
        line_ids=finding.line_ids,
        evidence=finding.evidence,
        explanation=finding.explanation,
    )


def staged_audit_pass(output: StagedAuditOutput) -> CriticAgentOutput:
    if output.verdict != "pass":
        raise ValueError("cannot materialize a revise audit as a pass")
    return CriticAgentOutput(
        verdict="pass",
        findings=(),
        remove_inventory_binding_ids=(),
        occurrence_removals=(),
        additional_bindings=(),
        semantic_only_target_facts=(),
        coherence_decisions=(),
        exhaustive_audit_receipt=True,
        rationale=output.rationale,
    )


def _annotated_lines(value: str) -> dict[int, str]:
    output: dict[int, str] = {}
    for row in value.splitlines():
        match = _NUMBERED_LINE.match(row)
        if match is not None:
            output[int(match.group("number"))] = row
    if not output:
        raise ValueError("annotated source has no numbered lines")
    return output


def build_staged_plan_payload(
    *,
    audit_payload: Mapping[str, Any],
    compact_payload: Mapping[str, Any],
    audit: StagedAuditOutput,
    halo_lines: int = 2,
) -> dict[str, Any]:
    """Slice a critic transaction to cited lines and explicitly implicated semantic owners."""

    if audit.verdict != "revise":
        raise ValueError("critic plan requires a revise audit")
    if halo_lines < 0:
        raise ValueError("critic plan halo cannot be negative")
    binding_rows = cast(Sequence[Sequence[Any]], compact_payload.get("bindingRows"))
    occurrence_rows = cast(Sequence[Sequence[Any]], compact_payload.get("occurrenceRows"))
    cited = {int(line_id[1:]) for finding in audit.findings for line_id in finding.line_ids}
    source_lines = _annotated_lines(cast(str, compact_payload.get("annotatedSource")))
    scoped_lines = {
        line
        for cited_line in cited
        for line in range(cited_line - halo_lines, cited_line + halo_lines + 1)
        if line in source_lines
    }
    selected_bindings = {index for finding in audit.findings for index in finding.binding_indexes}
    occurrence_materialized = [
        dict(zip(OCCURRENCE_COLUMNS, row, strict=True)) for row in occurrence_rows
    ]
    binding_materialized = [dict(zip(BINDING_COLUMNS, row, strict=True)) for row in binding_rows]
    candidate_rows = cast(Sequence[Mapping[str, Any]], audit_payload.get("candidateRows"))
    selected_candidate_indexes = {
        index for finding in audit.findings for index in finding.candidate_indexes
    }
    for candidate_index in selected_candidate_indexes:
        if candidate_index >= len(candidate_rows):
            raise ValueError("critic plan finding references an out-of-range candidate")
        selected_bindings.update(
            cast(Sequence[int], candidate_rows[candidate_index].get("bindingIndexes", ()))
        )
    for binding_index, row in enumerate(binding_materialized):
        if any(
            any(
                line in scoped_lines
                for line in range(
                    occurrence_materialized[
                        _id_index(
                            occurrence_id,
                            pattern=_OCCURRENCE_ID,
                            kind="occurrence",
                        )
                    ]["lineStartNumber"],
                    occurrence_materialized[
                        _id_index(
                            occurrence_id,
                            pattern=_OCCURRENCE_ID,
                            kind="occurrence",
                        )
                    ]["lineEndNumber"]
                    + 1,
                )
            )
            for occurrence_id in row["occurrenceIds"]
        ):
            selected_bindings.add(binding_index)

    selected_occurrences = {
        _id_index(
            occurrence_id,
            pattern=_OCCURRENCE_ID,
            kind="occurrence",
        )
        for binding_index in selected_bindings
        for occurrence_id in binding_materialized[binding_index]["occurrenceIds"]
    }
    occurrence_candidates = compact_payload.get("occurrenceCandidates")
    if not isinstance(occurrence_candidates, Mapping):
        raise ValueError("critic plan lacks occurrence candidates")
    columns = occurrence_candidates.get("columns")
    rows = occurrence_candidates.get("rows")
    if not isinstance(columns, (tuple, list)) or not isinstance(rows, (tuple, list)):
        raise ValueError("critic plan occurrence-candidate table is invalid")
    candidate_occurrences: list[Sequence[Any]] = []
    line_start_index = tuple(columns).index("lineStart")
    line_end_index = tuple(columns).index("lineEnd")
    source_text_index = tuple(columns).index("sourceText")
    occurrence_index_index = tuple(columns).index("occurrenceIndex")
    selected_current_occurrences = {
        (
            cast(int, occurrence_materialized[index]["lineStartNumber"]),
            cast(int, occurrence_materialized[index]["lineEndNumber"]),
            cast(str, occurrence_materialized[index]["sourceText"]),
            cast(int, occurrence_materialized[index]["occurrenceIndex"]),
        )
        for index in selected_occurrences
    }
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != len(columns):
            raise ValueError("critic plan occurrence-candidate row is invalid")
        start = int(str(row[line_start_index])[1:])
        end = int(str(row[line_end_index])[1:])
        identity = (
            start,
            end,
            cast(str, row[source_text_index]),
            cast(int, row[occurrence_index_index]),
        )
        # The source window carries a halo for interpretation, but a new span is authorized only
        # by an exact finding line.  Existing selected occurrences remain available so a complete
        # replacement can carry them forward.  Exposing other same-text halo/global matches made
        # invalid occurrence choices possible and caused avoidable output retries.
        if any(line in cited for line in range(start, end + 1)) or (
            identity in selected_current_occurrences
        ):
            candidate_occurrences.append(row)
            scoped_lines.update(line for line in range(start, end + 1) if line in source_lines)

    target_indexes = {index for finding in audit.findings for index in finding.target_path_indexes}
    for binding_index in selected_bindings:
        target_indexes.update(
            _id_index(path_id, pattern=_PATH_ID, kind="target path")
            for path_id in binding_materialized[binding_index]["targetPathIds"]
        )
        target_indexes.update(
            _id_index(path_id, pattern=_PATH_ID, kind="target path")
            for path_id in binding_materialized[binding_index]["dependencyPathIds"]
        )
    required = cast(Sequence[Mapping[str, Any]], audit_payload.get("requiredTargetCoBindings"))
    changed = True
    while changed:
        changed = False
        for required_row in required:
            component = set(cast(Sequence[int], required_row["targetPathIndexes"]))
            if component & target_indexes and not component <= target_indexes:
                target_indexes.update(component)
                changed = True

    target_facts = cast(Sequence[Mapping[str, Any]], audit_payload.get("targetFacts"))
    audit_bindings = cast(Sequence[Mapping[str, Any]], audit_payload.get("bindings"))
    audit_occurrences = cast(Sequence[Mapping[str, Any]], audit_payload.get("occurrences"))
    exact_owners: dict[tuple[int, int, str, int], set[str]] = defaultdict(set)
    for binding in binding_materialized:
        logical_key = cast(str, binding["logicalKey"])
        for occurrence_id in cast(Sequence[str], binding["occurrenceIds"]):
            occurrence = occurrence_materialized[
                _id_index(occurrence_id, pattern=_OCCURRENCE_ID, kind="occurrence")
            ]
            identity = (
                cast(int, occurrence["lineStartNumber"]),
                cast(int, occurrence["lineEndNumber"]),
                cast(str, occurrence["sourceText"]),
                cast(int, occurrence["occurrenceIndex"]),
            )
            exact_owners[identity].add(logical_key)
    named_occurrence_candidates = []
    for candidate_row in candidate_occurrences:
        candidate = dict(zip(columns, candidate_row, strict=True))
        identity = (
            int(cast(str, candidate["lineStart"])[1:]),
            int(cast(str, candidate["lineEnd"])[1:]),
            cast(str, candidate["sourceText"]),
            cast(int, candidate["occurrenceIndex"]),
        )
        candidate["currentExactOwnerLogicalKeys"] = tuple(sorted(exact_owners.get(identity, ())))
        named_occurrence_candidates.append(candidate)
    payload = {
        "contract": {
            "schemaVersion": 4,
            "allExplicitIndexesAreZeroBased": True,
            "bindingRecordsUseOriginalBindingIndex": True,
            "occurrenceRecordsUseOriginalOccurrenceRowIndex": True,
            "targetFactsUseOriginalTargetPathIndex": True,
        },
        "documentId": audit_payload.get("documentId"),
        "expectedCarrierName": audit_payload.get("expectedCarrierName"),
        "stateRevision": audit_payload.get("stateRevision"),
        "findings": [finding.model_dump(mode="json") for finding in audit.findings],
        "sourceWindow": "\n".join(source_lines[line] for line in sorted(scoped_lines)),
        "bindings": [audit_bindings[index] for index in sorted(selected_bindings)],
        "occurrences": [audit_occurrences[index] for index in sorted(selected_occurrences)],
        "targetFacts": [target_facts[index] for index in sorted(target_indexes)],
        "requiredTargetCoBindings": [
            row
            for row in required
            if set(cast(Sequence[int], row["targetPathIndexes"])) & target_indexes
        ],
        "candidateRows": [candidate_rows[index] for index in sorted(selected_candidate_indexes)],
        "occurrenceCandidates": named_occurrence_candidates,
        "allowedRemovalLogicalKeys": [
            binding_materialized[index]["logicalKey"] for index in sorted(selected_bindings)
        ],
    }
    return payload


def _candidate_occurrence_lookup(plan_payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = plan_payload.get("occurrenceCandidates")
    if not isinstance(rows, (tuple, list)):
        raise ValueError("critic plan lacks occurrence candidates")
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("critic plan occurrence candidate is invalid")
        occurrence_id = row.get("occurrenceId")
        if not isinstance(occurrence_id, str) or occurrence_id in output:
            raise ValueError("critic plan occurrence candidate IDs are invalid")
        current_owners = row.get("currentExactOwnerLogicalKeys", ())
        if not isinstance(current_owners, (tuple, list)) or any(
            not isinstance(owner, str) for owner in current_owners
        ):
            raise ValueError("critic plan occurrence candidate ownership is invalid")
        output[occurrence_id] = {
            "line_start": row["lineStart"],
            "line_end": row["lineEnd"],
            "source_text": row["sourceText"],
            "occurrence_index": row["occurrenceIndex"],
            "current_exact_owner_logical_keys": tuple(current_owners),
        }
    return output


def _materialize_plan_occurrences(
    rows: Sequence[CompilerOccurrenceReference | AgentOccurrence],
    lookup: Mapping[str, Mapping[str, Any]],
) -> tuple[AgentOccurrence, ...]:
    output: list[AgentOccurrence] = []
    for row in rows:
        if isinstance(row, CompilerOccurrenceReference):
            if row.occurrence_id not in lookup:
                raise ValueError("critic plan references an occurrence outside its repair slice")
            candidate = lookup[row.occurrence_id]
            output.append(
                AgentOccurrence.model_validate(
                    {
                        "line_start": candidate["line_start"],
                        "line_end": candidate["line_end"],
                        "source_text": candidate["source_text"],
                        "occurrence_index": candidate["occurrence_index"],
                    }
                )
            )
        else:
            output.append(row)
    return tuple(output)


def _augment_proven_repeat_appends(
    *,
    plan: StagedCriticPlanOutput,
    audit: StagedAuditOutput,
    plan_payload: Mapping[str, Any],
) -> StagedCriticPlanOutput:
    """Complete clerical appends already proved by one exhaustive audit finding.

    This is intentionally narrower than semantic planning. The audit must identify one existing
    owner and one required ``unowned_exact_repeat`` candidate, and the host must expose exactly one
    unowned physical occurrence matching that candidate. Any mutation, ambiguity, or reassignment
    leaves the finding to the model-authored plan and subsequent host preview.
    """

    bindings = plan_payload.get("bindings")
    candidate_rows = plan_payload.get("candidateRows")
    occurrence_candidates = plan_payload.get("occurrenceCandidates")
    if not all(
        isinstance(rows, (tuple, list))
        for rows in (bindings, candidate_rows, occurrence_candidates)
    ):
        raise ValueError("critic plan payload lacks repeat-append proof tables")
    bindings_by_index = {
        cast(int, row["bindingIndex"]): cast(str, row["logicalKey"])
        for row in cast(Sequence[Mapping[str, Any]], bindings)
        if isinstance(row, Mapping)
        and isinstance(row.get("bindingIndex"), int)
        and isinstance(row.get("logicalKey"), str)
    }
    candidates_by_index = {
        cast(int, row["candidateIndex"]): row
        for row in cast(Sequence[Mapping[str, Any]], candidate_rows)
        if isinstance(row, Mapping) and isinstance(row.get("candidateIndex"), int)
    }
    mutated_keys = {
        *plan.remove_binding_logical_keys,
        *(row.logical_key for row in plan.additional_bindings),
        *(row.logical_key for row in plan.occurrence_removals),
    }
    existing_append_ids = {
        occurrence.occurrence_id
        for append in plan.occurrence_appends
        for occurrence in append.occurrences
        if isinstance(occurrence, CompilerOccurrenceReference)
    }
    additions: list[StagedPlanOccurrenceAppend] = []
    for finding_index, finding in enumerate(audit.findings):
        if (
            finding.finding_kind != "unowned_repeated_fact"
            or len(finding.binding_indexes) != 1
            or len(finding.candidate_indexes) != 1
        ):
            continue
        binding_index = finding.binding_indexes[0]
        candidate_index = finding.candidate_indexes[0]
        owner = bindings_by_index.get(binding_index)
        candidate = candidates_by_index.get(candidate_index)
        if owner is None or candidate is None or owner in mutated_keys:
            continue
        if (
            candidate.get("kind") != "unowned_exact_repeat"
            or candidate.get("requiredRevision") is not True
            or tuple(candidate.get("bindingIndexes", ())) != (binding_index,)
        ):
            continue
        line_ids = tuple(candidate.get("lineIds", ()))
        source_texts = tuple(candidate.get("sourceTexts", ()))
        if len(line_ids) != 1 or len(source_texts) != 1:
            continue
        matching = tuple(
            row
            for row in cast(Sequence[Mapping[str, Any]], occurrence_candidates)
            if isinstance(row, Mapping)
            and row.get("lineStart") == line_ids[0]
            and row.get("lineEnd") == line_ids[0]
            and row.get("sourceText") == source_texts[0]
            and tuple(row.get("currentExactOwnerLogicalKeys", ())) == ()
            and isinstance(row.get("occurrenceId"), str)
        )
        if len(matching) != 1:
            continue
        occurrence_id = cast(str, matching[0]["occurrenceId"])
        if occurrence_id in existing_append_ids:
            continue
        additions.append(
            StagedPlanOccurrenceAppend(
                logical_key=owner,
                occurrences=(CompilerOccurrenceReference(occurrence_id=occurrence_id),),
                rationale=(
                    f"Host completed audit finding {finding_index}: the exhaustive audit and "
                    "candidate tables prove one unowned exact repeat for this existing owner."
                ),
            )
        )
        existing_append_ids.add(occurrence_id)
    if not additions:
        return plan
    return plan.model_copy(update={"occurrence_appends": (*plan.occurrence_appends, *additions)})


def _augment_unambiguous_coherence_decisions(
    *,
    plan: StagedCriticPlanOutput,
    audit: StagedAuditOutput,
    plan_payload: Mapping[str, Any],
) -> StagedCriticPlanOutput:
    """Complete a coherence operation already decided by the semantic audit.

    The host still requires the critic to decide whether a numeric coincidence is meaningful.
    Once a ``missing_coherence_dependency`` finding makes that semantic decision and its cited
    candidate has exactly one host-proved contract, asking the transaction planner to transcribe
    the only possible suggestion is redundant and caused valid audits to fail closed.
    """

    existing_ids = {decision.candidate_id for decision in plan.coherence_decisions}
    additions = tuple(
        decision
        for decision in host_completable_coherence_decisions(
            findings=audit.findings,
            candidate_rows=cast(Sequence[Mapping[str, Any]], plan_payload.get("candidateRows", ())),
        )
        if decision.candidate_id not in existing_ids
    )
    if not additions:
        return plan
    return plan.model_copy(update={"coherence_decisions": (*plan.coherence_decisions, *additions)})


def host_completable_coherence_decisions(
    *,
    findings: Sequence[StagedAuditFinding | Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> tuple[CoherenceCandidateDecision, ...]:
    """Return unique host operations for relations the semantic audit already proved."""

    candidates_by_index = _indexed_rows(
        candidate_rows, index_key="candidateIndex", kind="candidate"
    )
    decisions: dict[str, CoherenceCandidateDecision] = {}
    for finding_index, finding in enumerate(findings):
        if isinstance(finding, StagedAuditFinding):
            normalized_finding_kind: str | None = finding.finding_kind
            candidate_indexes = finding.candidate_indexes
        else:
            raw_finding_kind = finding.get("finding_kind")
            normalized_finding_kind = (
                raw_finding_kind if isinstance(raw_finding_kind, str) else None
            )
            raw_indexes = finding.get("candidate_indexes", ())
            candidate_indexes = tuple(raw_indexes) if isinstance(raw_indexes, (tuple, list)) else ()
        if normalized_finding_kind != "missing_coherence_dependency":
            continue
        for candidate_index in candidate_indexes:
            if not isinstance(candidate_index, int) or isinstance(candidate_index, bool):
                continue
            candidate = candidates_by_index.get(candidate_index)
            if (
                candidate is None
                or candidate.get("kind") != "cross_field_semantic_relation"
                or candidate.get("requiredRevision") is not True
            ):
                continue
            candidate_id = candidate.get("candidateId")
            details = candidate.get("details")
            suggestions = (
                details.get("suggestedContracts", ()) if isinstance(details, Mapping) else ()
            )
            ambiguous = (
                details.get("ambiguousAlternatives") if isinstance(details, Mapping) else None
            )
            if (
                not isinstance(candidate_id, str)
                or ambiguous is not False
                or not isinstance(suggestions, (tuple, list))
                or len(suggestions) != 1
                or not isinstance(suggestions[0], Mapping)
            ):
                continue
            decisions.setdefault(
                candidate_id,
                CoherenceCandidateDecision(
                    candidate_id=candidate_id,
                    disposition="apply_suggestion",
                    suggestion_index=0,
                    rationale=(
                        f"Host completed audit finding {finding_index}: the critic identified a "
                        "missing coherence dependency and the evidence exposes exactly one "
                        "unambiguous host-proved contract."
                    ),
                ),
            )
    return tuple(decisions.values())


def restore_staged_plan(
    *,
    plan: StagedCriticPlanOutput,
    audit: StagedAuditOutput,
    plan_payload: Mapping[str, Any],
    compact_payload: Mapping[str, Any],
    source_payload: Mapping[str, Any],
) -> CriticAgentOutput:
    from .host import inventory_binding_id

    plan = _augment_unambiguous_coherence_decisions(
        plan=plan,
        audit=audit,
        plan_payload=plan_payload,
    )
    plan = _augment_proven_repeat_appends(plan=plan, audit=audit, plan_payload=plan_payload)
    allowed = set(cast(Sequence[str], plan_payload.get("allowedRemovalLogicalKeys")))
    plan_bindings = plan_payload.get("bindings")
    if not isinstance(plan_bindings, (tuple, list)):
        raise ValueError("critic plan lacks its binding slice")
    target_paths_by_key: dict[str, set[str]] = {}
    for row in plan_bindings:
        if not isinstance(row, Mapping) or not isinstance(row.get("logicalKey"), str):
            raise ValueError("critic plan binding slice is invalid")
        target_paths_by_key[cast(str, row["logicalKey"])] = set(
            cast(Sequence[str], row.get("targetPaths", ()))
        )

    def canonical_existing_key(logical_key: str) -> str:
        if logical_key in allowed or not logical_key.startswith("anchor:"):
            return logical_key
        claimed_paths = set(logical_key.removeprefix("anchor:").split("|"))
        candidates = [
            candidate
            for candidate, target_paths in target_paths_by_key.items()
            if claimed_paths and claimed_paths <= target_paths
        ]
        return candidates[0] if len(candidates) == 1 else logical_key

    normalized_remove_keys = list(
        dict.fromkeys(canonical_existing_key(key) for key in plan.remove_binding_logical_keys)
    )
    edit_keys = {
        *normalized_remove_keys,
        *(canonical_existing_key(row.logical_key) for row in plan.occurrence_appends),
        *(canonical_existing_key(row.logical_key) for row in plan.occurrence_removals),
    }
    if not edit_keys <= allowed:
        raise ValueError("critic plan edits a binding outside its cited repair slice")
    occurrence_lookup = _candidate_occurrence_lookup(plan_payload)
    restored_bindings: list[AgentBindingProposal] = []
    for binding in plan.additional_bindings:
        candidate = binding.model_dump(mode="python")
        candidate["occurrences"] = _materialize_plan_occurrences(
            binding.occurrences, occurrence_lookup
        )
        restored_bindings.append(
            restore_legacy_binding(DiscriminatedBindingProposal.model_validate(candidate))
        )

    inventory = source_payload.get("bindingInventory")
    if not isinstance(inventory, (tuple, list)):
        raise ValueError("critic plan source inventory is invalid")
    inventory_by_key: dict[str, Mapping[str, Any]] = {}
    for row in inventory:
        if not isinstance(row, Mapping) or not isinstance(row.get("logicalKey"), str):
            raise ValueError("critic plan source inventory row is invalid")
        logical_key = cast(str, row["logicalKey"])
        if logical_key in inventory_by_key:
            raise ValueError("critic plan source inventory has duplicate logical keys")
        inventory_by_key[logical_key] = row

    occurrence_edit_keys = {
        *(canonical_existing_key(row.logical_key) for row in plan.occurrence_appends),
        *(canonical_existing_key(row.logical_key) for row in plan.occurrence_removals),
    }
    for binding_index, restored_binding in enumerate(restored_bindings):
        logical_key = canonical_existing_key(restored_binding.logical_key)
        current = inventory_by_key.get(logical_key)
        if current is None or logical_key in normalized_remove_keys:
            continue
        if logical_key in occurrence_edit_keys:
            raise ValueError(
                "critic plan cannot combine a same-key replacement with occurrence edits: "
                + logical_key
            )
        current_occurrences = current.get("occurrences")
        if not isinstance(current_occurrences, (tuple, list)) or any(
            not isinstance(occurrence, Mapping) for occurrence in current_occurrences
        ):
            raise ValueError("critic plan source inventory has invalid occurrences")
        current_identities = {
            (
                occurrence.get("lineStart"),
                occurrence.get("lineEnd"),
                occurrence.get("sourceText"),
                occurrence.get("occurrenceIndex"),
            )
            for occurrence in cast(Sequence[Mapping[str, Any]], current_occurrences)
        }
        proposed_identities = {
            (
                occurrence.line_start,
                occurrence.line_end,
                occurrence.source_text,
                occurrence.occurrence_index,
            )
            for occurrence in restored_binding.occurrences
        }
        missing = current_identities - proposed_identities
        if missing:
            raise ValueError(
                "critic same-key replacement omits current occurrences; use an occurrence append "
                f"or carry the complete binding: {logical_key}; missing={len(missing)}"
            )
        normalized_remove_keys.append(logical_key)
        restored_bindings[binding_index] = restored_binding.model_copy(
            update={
                "logical_key": logical_key,
                "rationale": (
                    restored_binding.rationale
                    + " Host normalized the complete same-key proposal to an atomic full "
                    "replacement."
                ),
            }
        )

    binding_index_by_key = {
        cast(str, row["logicalKey"]): cast(int, row["bindingIndex"]) for row in plan_bindings
    }
    binding_key_by_index = {index: key for key, index in binding_index_by_key.items()}
    for finding_index, finding in enumerate(audit.findings):
        if finding.finding_kind != "missing_derivation":
            continue
        cited_lines = {int(line_id[1:]) for line_id in finding.line_ids}
        cited_keys = {
            binding_key_by_index[index]
            for index in finding.binding_indexes
            if index in binding_key_by_index
        }
        cited_paths = {path for key in cited_keys for path in target_paths_by_key.get(key, ())}
        local_derived = tuple(
            binding
            for binding in restored_bindings
            if binding.render_mode == "deterministic_derived"
            and any(
                line in cited_lines
                for occurrence in binding.occurrences
                for line in range(int(occurrence.line_start[1:]), int(occurrence.line_end[1:]) + 1)
            )
        )
        if not local_derived:
            raise ValueError(
                f"missing-derivation finding {finding_index} has no deterministic_derived "
                "replacement on its cited lines; do not replace a calculated fact with a "
                "deterministic_auxiliary or direct binding"
            )
        if cited_keys or cited_paths:
            linked = any(
                set(binding.dependency_bindings) & cited_keys
                or set(binding.dependency_paths) & cited_paths
                for binding in local_derived
            )
            if not linked:
                raise ValueError(
                    f"missing-derivation finding {finding_index} has no replacement linked to "
                    f"a cited input; binding_inputs={sorted(cited_keys)}, "
                    f"target_inputs={sorted(cited_paths)}"
                )
    finding_lines_by_key: dict[str, set[int]] = defaultdict(set)
    for finding in audit.findings:
        cited_lines = {int(line_id[1:]) for line_id in finding.line_ids}
        for logical_key, binding_index in binding_index_by_key.items():
            if binding_index in finding.binding_indexes:
                finding_lines_by_key[logical_key].update(cited_lines)

    def materialize_append_occurrences(
        logical_key: str, appends: Sequence[StagedPlanOccurrenceAppend]
    ) -> tuple[tuple[AgentOccurrence, ...], bool]:
        inventory_row = inventory_by_key[logical_key]
        equipment_type_owner = (
            inventory_row.get("renderMode") == "target_binding"
            and inventory_row.get("groupKind") == "equipment"
            and any(
                str(path).endswith(".typeDescription")
                for path in cast(Sequence[str], inventory_row.get("targetPaths", ()))
            )
        )
        owner_lines = tuple(
            int(str(occurrence["lineStart"])[1:])
            for occurrence in cast(
                Sequence[Mapping[str, Any]], inventory_row.get("occurrences", ())
            )
        )
        cited_lines = finding_lines_by_key.get(logical_key, set())
        materialized: list[AgentOccurrence] = []
        retargeted = False
        for append in appends:
            for occurrence in append.occurrences:
                selected = occurrence
                if isinstance(occurrence, CompilerOccurrenceReference):
                    candidate = occurrence_lookup.get(occurrence.occurrence_id)
                    if candidate is None:
                        raise ValueError(
                            "critic plan references an occurrence outside its repair slice"
                        )
                    current_owners = set(
                        cast(
                            Sequence[str],
                            candidate.get("current_exact_owner_logical_keys", ()),
                        )
                    )
                    retained_other_owner = bool(
                        current_owners - {logical_key} - set(normalized_remove_keys)
                    )
                    if (
                        equipment_type_owner
                        and retained_other_owner
                        and owner_lines
                        and cited_lines
                    ):
                        alternatives: list[tuple[int, str]] = []
                        for candidate_id, alternative in occurrence_lookup.items():
                            alternative_line = int(str(alternative["line_start"])[1:])
                            if (
                                alternative["source_text"] == candidate["source_text"]
                                and not alternative.get("current_exact_owner_logical_keys")
                                and alternative_line in cited_lines
                            ):
                                distance = min(
                                    abs(alternative_line - owner_line) for owner_line in owner_lines
                                )
                                alternatives.append((distance, candidate_id))
                        alternatives.sort()
                        if len(alternatives) == 1 or (
                            alternatives and alternatives[0][0] < alternatives[1][0]
                        ):
                            selected = CompilerOccurrenceReference(occurrence_id=alternatives[0][1])
                            retargeted = selected.occurrence_id != occurrence.occurrence_id
                materialized.extend(_materialize_plan_occurrences((selected,), occurrence_lookup))
        return tuple(materialized), retargeted

    appends_by_key: dict[str, list[StagedPlanOccurrenceAppend]] = {}
    for append in plan.occurrence_appends:
        appends_by_key.setdefault(canonical_existing_key(append.logical_key), []).append(append)
    for logical_key, appends in appends_by_key.items():
        row = inventory_by_key[logical_key]
        occurrences, retargeted = materialize_append_occurrences(logical_key, appends)
        occurrence_identities = tuple(
            (
                occurrence.line_start,
                occurrence.line_end,
                occurrence.source_text,
                occurrence.occurrence_index,
            )
            for occurrence in occurrences
        )
        if len(set(occurrence_identities)) != len(occurrence_identities):
            raise ValueError("critic plan appends a duplicate exact occurrence")
        restored_bindings.append(
            AgentBindingProposal.model_validate(
                {
                    "logical_key": logical_key,
                    "render_mode": row["renderMode"],
                    "value_kind": row["valueKind"],
                    "group_kind": row["groupKind"],
                    "group_key": row["groupKey"],
                    "target_paths": tuple(cast(Sequence[str], row["targetPaths"])),
                    "derivation": row["derivation"],
                    "dependency_paths": tuple(cast(Sequence[str], row["dependencyPaths"])),
                    "dependency_bindings": tuple(cast(Sequence[str], row["dependencyBindings"])),
                    "occurrences": occurrences,
                    "rationale": " ".join(dict.fromkeys(append.rationale for append in appends))
                    + (
                        " Host retargeted an already owned equipment token to the unique nearest "
                        "unowned equal surface on an audit-cited line for this owner."
                        if retargeted
                        else ""
                    ),
                }
            )
        )

    current_occurrences_by_id: dict[str, tuple[str, AgentOccurrence]] = {}
    occurrence_rows = cast(Sequence[Sequence[Any]], compact_payload.get("occurrenceRows"))
    binding_rows = cast(Sequence[Sequence[Any]], compact_payload.get("bindingRows"))
    occurrence_models: dict[str, AgentOccurrence] = {}
    for occurrence_row in occurrence_rows:
        occurrence_data = dict(zip(OCCURRENCE_COLUMNS, occurrence_row, strict=True))
        occurrence_models[occurrence_data["occurrenceId"]] = AgentOccurrence(
            line_start=f"L{occurrence_data['lineStartNumber']:05d}",
            line_end=f"L{occurrence_data['lineEndNumber']:05d}",
            source_text=occurrence_data["sourceText"],
            occurrence_index=occurrence_data["occurrenceIndex"],
        )
    for binding_row in binding_rows:
        binding_data = dict(zip(BINDING_COLUMNS, binding_row, strict=True))
        for occurrence_id in binding_data["occurrenceIds"]:
            current_occurrences_by_id[occurrence_id] = (
                binding_data["logicalKey"],
                occurrence_models[occurrence_id],
            )
    occurrence_ids_by_key: dict[str, set[str]] = {}
    for occurrence_id, (logical_key, _occurrence) in current_occurrences_by_id.items():
        occurrence_ids_by_key.setdefault(logical_key, set()).add(occurrence_id)
    normalized_full_removal_keys = list(normalized_remove_keys)
    removals: list[CriticOccurrenceRemoval] = []
    for removal in plan.occurrence_removals:
        removal_logical_key = canonical_existing_key(removal.logical_key)
        removed_occurrences: list[AgentOccurrence] = []
        selected_occurrence_ids: list[str] = []
        retargeted = False
        for occurrence_id in removal.occurrence_ids:
            if occurrence_id not in current_occurrences_by_id:
                raise ValueError("critic plan removes an unknown current occurrence")
            owner, occurrence = current_occurrences_by_id[occurrence_id]
            if owner != removal_logical_key:
                cited_lines = finding_lines_by_key.get(removal_logical_key, set())
                alternatives = tuple(
                    candidate_id
                    for candidate_id in sorted(occurrence_ids_by_key.get(removal_logical_key, ()))
                    if any(
                        line in cited_lines
                        for line in range(
                            int(current_occurrences_by_id[candidate_id][1].line_start[1:]),
                            int(current_occurrences_by_id[candidate_id][1].line_end[1:]) + 1,
                        )
                    )
                )
                if len(alternatives) != 1:
                    raise ValueError("critic plan occurrence removal has the wrong owner")
                occurrence_id = alternatives[0]
                owner, occurrence = current_occurrences_by_id[occurrence_id]
                retargeted = True
            selected_occurrence_ids.append(occurrence_id)
            removed_occurrences.append(occurrence)
        selected_ids = set(selected_occurrence_ids)
        if len(selected_ids) != len(selected_occurrence_ids):
            raise ValueError("critic plan occurrence removal resolves to a duplicate occurrence")
        if selected_ids == occurrence_ids_by_key.get(removal_logical_key):
            normalized_full_removal_keys.append(removal_logical_key)
        else:
            removals.append(
                CriticOccurrenceRemoval(
                    logical_key=removal_logical_key,
                    occurrences=tuple(removed_occurrences),
                    rationale=(
                        removal.rationale
                        + (
                            " Host retargeted a mismatched occurrence handle to the unique "
                            "occurrence of the declared owner on an audit-cited line."
                            if retargeted
                            else ""
                        )
                    ),
                )
            )

    return CriticAgentOutput(
        verdict="revise",
        findings=tuple(_base_finding(finding) for finding in audit.findings),
        remove_inventory_binding_ids=tuple(
            inventory_binding_id(key) for key in dict.fromkeys(normalized_full_removal_keys)
        ),
        occurrence_removals=tuple(removals),
        additional_bindings=tuple(restored_bindings),
        semantic_only_target_facts=plan.semantic_only_target_facts,
        coherence_decisions=plan.coherence_decisions,
        exhaustive_audit_receipt=True,
        rationale=plan.rationale,
    )


def staged_schema_sizes() -> dict[str, int]:
    """Expose stable schema byte sizes for the experiment's offline benchmark."""

    return {
        "audit": len(canonical_json_bytes(StagedAuditOutput.model_json_schema(mode="validation"))),
        "facet_audit": len(
            canonical_json_bytes(StagedFacetAuditOutput.model_json_schema(mode="validation"))
        ),
        "plan": len(
            canonical_json_bytes(StagedCriticPlanOutput.model_json_schema(mode="validation"))
        ),
    }


def build_local_compiler_repair_payload(
    payload: Mapping[str, Any], prior: CompilerAgentOutput, *, halo_lines: int = 2
) -> dict[str, Any]:
    """Project a host rejection onto the exact candidate rows and source lines it names."""

    error = payload.get("requiredRevision")
    numbered_source = payload.get("numberedSource")
    allowed_paths = payload.get("allowedTargetPaths")
    if not isinstance(error, str) or not error.strip():
        raise ValueError("local compiler repair requires an explicit host rejection")
    if not isinstance(numbered_source, str) or not isinstance(allowed_paths, (tuple, list)):
        raise ValueError("local compiler repair lacks source or target vocabulary")
    source_lines = _annotated_lines(numbered_source)
    local_error = re.sub(r"documentMatchRanges=\[[^\]]*\]", "", error)
    referenced_lines = {int(value[1:]) for value in re.findall(r"L[0-9]{5}", local_error)}
    referenced_paths = {
        path
        for path in allowed_paths
        if isinstance(path, str) and re.search(rf"{re.escape(path)}(?![.\[])", error) is not None
    }
    error_referenced_paths = frozenset(referenced_paths)
    referenced_anchor_ids = set(re.findall(r"anchor_binding_[0-9]{4}", error))

    carrier_lines = {
        line
        for occurrence in prior.carrier.evidence_occurrences
        for line in range(int(occurrence.line_start[1:]), int(occurrence.line_end[1:]) + 1)
    }
    carrier_selected = "carrier" in error.casefold() or bool(carrier_lines & referenced_lines)
    if carrier_selected:
        referenced_lines.update(carrier_lines)
    selection_lines = frozenset(referenced_lines)

    anchor_rows = payload.get("anchorBindings")
    if not isinstance(anchor_rows, (tuple, list)):
        raise ValueError("local compiler repair lacks anchor inventory")
    anchor_scopes: list[tuple[Mapping[str, Any], set[int], set[str], set[str]]] = []
    for row in anchor_rows:
        if not isinstance(row, Mapping):
            raise ValueError("local compiler repair anchor row is invalid")
        occurrence_lines = {
            line
            for occurrence in cast(Sequence[Mapping[str, Any]], row.get("occurrences", ()))
            for line in range(
                int(str(occurrence["lineStart"])[1:]),
                int(str(occurrence["lineEnd"])[1:]) + 1,
            )
        }
        target_paths = set(cast(Sequence[str], row.get("targetPaths", ())))
        anchor_ids = {
            str(occurrence.get("anchorBindingId"))
            for occurrence in cast(Sequence[Mapping[str, Any]], row.get("occurrences", ()))
        }
        anchor_scopes.append((row, occurrence_lines, target_paths, anchor_ids))

    co_binding_rows = payload.get("requiredTargetCoBindings", ())
    if not isinstance(co_binding_rows, (tuple, list)):
        raise ValueError("local compiler repair co-binding inventory is invalid")
    co_binding_scopes: list[tuple[Mapping[str, Any], set[str]]] = []
    for row in co_binding_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("targetPaths"), (tuple, list)):
            raise ValueError("local compiler repair co-binding row is invalid")
        paths = set(cast(Sequence[str], row["targetPaths"]))
        if not paths:
            raise ValueError("local compiler repair co-binding row is empty")
        co_binding_scopes.append((row, paths))

    binding_scopes: list[tuple[AgentBindingProposal, set[int], set[str]]] = []
    for binding in prior.bindings:
        occurrence_lines = {
            line
            for occurrence in binding.occurrences
            for line in range(int(occurrence.line_start[1:]), int(occurrence.line_end[1:]) + 1)
        }
        binding_scopes.append(
            (
                binding,
                occurrence_lines,
                {*binding.target_paths, *binding.dependency_paths},
            )
        )
    prior_declared_paths = {
        path for binding, _occurrence_lines, paths in binding_scopes for path in paths
    } | {row.target_path for row in prior.semantic_only_target_facts}
    allowed_path_set = {path for path in allowed_paths if isinstance(path, str)}
    invalid_prior_semantic_paths = {
        row.target_path
        for row in prior.semantic_only_target_facts
        if row.target_path not in allowed_path_set
    }
    explicit_prior_paths = {
        path
        for path in prior_declared_paths
        if re.search(rf"{re.escape(path)}(?![.\[])", error) is not None
    }

    # Co-binding requirements are the only relationships allowed to expand a rejected
    # target-path set. Candidate owners can span unrelated equal-valued facts, so feeding
    # every selected owner's paths back into selection would turn a local repair into a
    # document-wide equality closure (for example, every package whose quantity is 96).
    selected_co_binding_indexes: set[int] = set()
    changed = True
    while changed:
        prior_paths = frozenset(referenced_paths)
        for index, (_row, paths) in enumerate(co_binding_scopes):
            if paths & referenced_paths:
                selected_co_binding_indexes.add(index)
                referenced_paths.update(paths)
        changed = frozenset(referenced_paths) != prior_paths

    selected_binding_indexes = {
        index
        for index, (binding, occurrence_lines, paths) in enumerate(binding_scopes)
        if (
            binding.logical_key in error
            or any(path in binding.logical_key for path in referenced_paths)
            or bool(paths & referenced_paths)
            or bool(paths & explicit_prior_paths)
            or bool(occurrence_lines & selection_lines)
        )
    }
    selected_anchor_indexes = {
        index
        for index, (_row, occurrence_lines, paths, anchor_ids) in enumerate(anchor_scopes)
        if (
            bool(paths & error_referenced_paths)
            or bool(occurrence_lines & selection_lines)
            or bool(anchor_ids & referenced_anchor_ids)
        )
    }

    selected_bindings = [prior.bindings[index] for index in sorted(selected_binding_indexes)]
    selected_anchors = [anchor_scopes[index][0] for index in sorted(selected_anchor_indexes)]
    selected_co_bindings = [
        co_binding_scopes[index][0] for index in sorted(selected_co_binding_indexes)
    ]
    context_paths = set(referenced_paths)
    for binding in selected_bindings:
        context_paths.update(binding.target_paths)
        context_paths.update(binding.dependency_paths)
    # A selected deterministic anchor remains visible with its complete impact surface, but it
    # does not authorize edits to every equal-valued target it happens to own.  Only an explicit
    # host diagnostic, a current mutable binding, or a declared co-binding relationship can widen
    # the repair transaction.  Otherwise one rejected repeated scalar (for example package 9's
    # quantity 96) pulls every other package quantity and its object context into the payload.
    for index in selected_binding_indexes:
        referenced_lines.update(binding_scopes[index][1])
    for index in selected_anchor_indexes:
        referenced_lines.update(anchor_scopes[index][1])
    # Invalid prior declarations are absent from the valid target vocabulary by definition, but
    # they must remain inside the transaction's removal vocabulary. Omitting them gave the model
    # a host error it had no schema-valid way to repair.
    selected_semantic = tuple(
        row
        for row in prior.semantic_only_target_facts
        if row.target_path in context_paths
        or row.target_path in explicit_prior_paths
        or row.target_path in invalid_prior_semantic_paths
    )
    selected_anchor_ids = {
        str(occurrence.get("anchorBindingId"))
        for row in selected_anchors
        for occurrence in cast(Sequence[Mapping[str, Any]], row.get("occurrences", ()))
    }
    selected_overrides = tuple(
        row
        for row in prior.anchor_overrides
        if row.anchor_binding_id in referenced_anchor_ids | selected_anchor_ids
    )

    prior_inventory = payload.get("previousCandidateBindingInventory", ())
    if not isinstance(prior_inventory, (tuple, list)):
        raise ValueError("local compiler repair candidate inventory is invalid")
    selected_keys = {binding.logical_key for binding in selected_bindings}
    selected_inventory = tuple(
        row
        for row in prior_inventory
        if isinstance(row, Mapping)
        and (
            row.get("logicalKey") in selected_keys
            or set(cast(Sequence[str], row.get("targetPaths", ()))) & context_paths
        )
    )
    if not (
        referenced_lines
        or referenced_paths
        or referenced_anchor_ids
        or selected_keys
        or selected_semantic
    ):
        raise ValueError("host rejection cannot be safely projected onto a local repair slice")
    initially_scoped_lines = {
        line
        for referenced in referenced_lines
        for line in range(referenced - halo_lines, referenced + halo_lines + 1)
        if line in source_lines
    }
    invalid_prior_paths = (explicit_prior_paths - allowed_path_set) | invalid_prior_semantic_paths
    if not initially_scoped_lines and not any(
        row.target_path in invalid_prior_paths for row in selected_semantic
    ):
        raise ValueError("local compiler repair has no source window")

    source_label = payload.get("sourceLabel")
    if not isinstance(source_label, Mapping):
        raise ValueError("local compiler repair lacks its target label")
    from .host import _canonical_target_group, _resolve_target_path

    repair_group_keys = {_canonical_target_group((path,))[1] for path in error_referenced_paths}

    target_context_ids: dict[str, str] = {}
    target_contexts: list[dict[str, Any]] = []

    def target_context_id(path: str) -> str | None:
        segments = path.split(".")
        indexed_segments = [
            index for index, segment in enumerate(segments) if re.search(r"\[[0-9]+\]", segment)
        ]
        context_path: str | None = None
        context_value: Any = None
        for indexed_segment in reversed(indexed_segments):
            candidate_path = ".".join(segments[: indexed_segment + 1])
            candidate_value = _resolve_target_path(source_label, candidate_path)
            if isinstance(candidate_value, Mapping):
                context_path = candidate_path
                context_value = candidate_value
                break
        if context_path is None and len(segments) > 2:
            candidate_path = ".".join(segments[:-1])
            candidate_value = _resolve_target_path(source_label, candidate_path)
            if isinstance(candidate_value, Mapping):
                context_path = candidate_path
                context_value = candidate_value
        if context_path is None or not isinstance(context_value, Mapping):
            return None
        existing = target_context_ids.get(context_path)
        if existing is not None:
            return existing
        identifier = f"target_context_{len(target_contexts) + 1:04d}"
        target_context_ids[context_path] = identifier
        target_contexts.append(
            {
                "contextId": identifier,
                "contextPath": context_path,
                "sourceValue": context_value,
            }
        )
        return identifier

    target_facts = tuple(
        {
            "targetPath": path,
            "sourceValue": _resolve_target_path(source_label, path),
            "contextId": target_context_id(path),
        }
        for path in allowed_paths
        if path in context_paths
    )
    raw_occurrence_candidates = payload.get("occurrenceCandidates")
    if not isinstance(raw_occurrence_candidates, Mapping):
        raise ValueError("local compiler repair lacks occurrence candidates")
    occurrence_columns = raw_occurrence_candidates.get("columns")
    occurrence_rows = raw_occurrence_candidates.get("rows")
    if not isinstance(occurrence_columns, (tuple, list)) or not isinstance(
        occurrence_rows, (tuple, list)
    ):
        raise ValueError("local compiler repair occurrence-candidate table is invalid")
    line_start_index = tuple(occurrence_columns).index("lineStart")
    line_end_index = tuple(occurrence_columns).index("lineEnd")
    source_text_index = tuple(occurrence_columns).index("sourceText")
    occurrence_index_index = tuple(occurrence_columns).index("occurrenceIndex")
    target_surfaces = {
        fact["targetPath"]: "".join(
            character.casefold() for character in str(fact["sourceValue"]) if character.isalnum()
        )
        for fact in target_facts
        if isinstance(fact["sourceValue"], (str, int, float))
        and not isinstance(fact["sourceValue"], bool)
    }
    # Five normalized characters is the same distinctiveness floor used by the document-wide
    # repeat review. Short counts and codes remain available inside the exact rejection window,
    # but do not fan a local repair out across every equal digit in a document.
    distinctive_surface_length = 5
    materialized_occurrences: list[tuple[dict[str, Any], set[int], tuple[str, ...]]] = []
    target_candidate_lines: set[int] = set()
    for row in occurrence_rows:
        if not isinstance(row, (tuple, list)) or len(row) != len(occurrence_columns):
            raise ValueError("local compiler repair occurrence-candidate row is invalid")
        start = int(str(row[line_start_index])[1:])
        end = int(str(row[line_end_index])[1:])
        row_lines = set(range(start, end + 1))
        normalized_source = "".join(
            character.casefold() for character in str(row[source_text_index]) if character.isalnum()
        )
        matching_paths = tuple(
            path
            for path, target_surface in target_surfaces.items()
            if min(len(normalized_source), len(target_surface)) >= distinctive_surface_length
            and (normalized_source in target_surface or target_surface in normalized_source)
        )
        if matching_paths:
            target_candidate_lines.update(row_lines)
        materialized_occurrences.append(
            (dict(zip(occurrence_columns, row, strict=True)), row_lines, matching_paths)
        )

    scoped_lines = initially_scoped_lines | {
        line
        for referenced in target_candidate_lines
        for line in range(referenced - halo_lines, referenced + halo_lines + 1)
        if line in source_lines
    }
    resolved_to_proposal_keys: dict[str, set[str]] = defaultdict(set)
    for binding in prior.bindings:
        resolved_key = (
            "anchor:" + "|".join(sorted(binding.target_paths))
            if binding.target_paths and binding.render_mode in {"target_binding", "carrier_static"}
            else binding.logical_key
            if binding.logical_key.startswith(("anchor:", "agent:"))
            else "agent:" + binding.logical_key
        )
        resolved_to_proposal_keys[resolved_key].add(binding.logical_key)

    exact_owners: dict[tuple[str, str, str, int], set[str]] = {}
    for inventory_row in prior_inventory:
        if not isinstance(inventory_row, Mapping):
            continue
        resolved_logical_key = inventory_row.get("logicalKey")
        if not isinstance(resolved_logical_key, str):
            continue
        proposal_keys = resolved_to_proposal_keys.get(resolved_logical_key, {resolved_logical_key})
        for occurrence in cast(Sequence[Mapping[str, Any]], inventory_row.get("occurrences", ())):
            identity = (
                str(occurrence.get("lineStart")),
                str(occurrence.get("lineEnd")),
                str(occurrence.get("sourceText")),
                int(occurrence.get("occurrenceIndex", 0)),
            )
            exact_owners.setdefault(identity, set()).update(proposal_keys)
    prior_bindings_by_key = {binding.logical_key: binding for binding in prior.bindings}

    # If an exact candidate capable of repairing a cited path is already owned by another
    # candidate binding, that owner is part of the atomic repair closure. Without it, an
    # equal-valued row can be reassigned only by overlapping the retained owner, and the model is
    # given no legal operation that could succeed. This is source-span ownership closure, not an
    # inference that the equal target facts share semantics.
    block_by_line: dict[int, int] = {}
    block_index = 0
    for line_number in sorted(source_lines):
        content = source_lines[line_number].split(" | ", 1)[1]
        if not content.strip():
            block_index += 1
            continue
        block_by_line[line_number] = block_index

    def semantic_entity(path: str) -> str | None:
        segments = path.split(".")
        for segment_index in reversed(
            [index for index, segment in enumerate(segments) if "[" in segment]
        ):
            candidate = ".".join(segments[: segment_index + 1])
            try:
                value = _resolve_target_path(source_label, candidate)
            except ValueError:
                continue
            if isinstance(value, Mapping):
                return candidate
        return None

    repair_entities = {
        entity for path in error_referenced_paths if (entity := semantic_entity(path)) is not None
    }
    repair_entities_by_block: dict[int, set[str]] = defaultdict(set)
    for inventory_row in prior_inventory:
        if not isinstance(inventory_row, Mapping):
            continue
        row_entities = {
            entity
            for path in cast(Sequence[str], inventory_row.get("targetPaths", ()))
            if (entity := semantic_entity(path)) in repair_entities
        }
        if not row_entities:
            continue
        for occurrence in cast(Sequence[Mapping[str, Any]], inventory_row.get("occurrences", ())):
            start = int(str(occurrence.get("lineStart"))[1:])
            end = int(str(occurrence.get("lineEnd"))[1:])
            for line_number in range(start, end + 1):
                block = block_by_line.get(line_number)
                if block is not None:
                    repair_entities_by_block[block].update(row_entities)

    repair_candidate_owner_keys: set[str] = set()
    for occurrence, _row_lines, matching_paths in materialized_occurrences:
        if not set(matching_paths).intersection(error_referenced_paths):
            continue
        start = int(str(occurrence[occurrence_columns[line_start_index]])[1:])
        end = int(str(occurrence[occurrence_columns[line_end_index]])[1:])
        identity = (
            str(occurrence[occurrence_columns[line_start_index]]),
            str(occurrence[occurrence_columns[line_end_index]]),
            str(occurrence[occurrence_columns[source_text_index]]),
            int(occurrence[occurrence_columns[occurrence_index_index]]),
        )
        owner_keys = exact_owners.get(identity, set())
        block_has_repair_entity = any(
            repair_entities_by_block.get(block_by_line.get(line_number, -1), set())
            for line_number in range(start, end + 1)
        )
        owner_has_repair_group = any(
            (owner := prior_bindings_by_key.get(key)) is not None
            and owner.group_key in repair_group_keys
            for key in owner_keys
        )
        if repair_entities and not (block_has_repair_entity or owner_has_repair_group):
            continue
        repair_candidate_owner_keys.update(owner_keys)
    selected_binding_indexes.update(
        index
        for index, (binding, _occurrence_lines, _paths) in enumerate(binding_scopes)
        if binding.logical_key in repair_candidate_owner_keys
    )
    if repair_candidate_owner_keys:
        selected_bindings = [prior.bindings[index] for index in sorted(selected_binding_indexes)]
        selected_keys = {binding.logical_key for binding in selected_bindings}
        for index in selected_binding_indexes:
            referenced_lines.update(binding_scopes[index][1])
            context_paths.update(binding_scopes[index][2])
        scoped_lines.update(
            line
            for referenced in referenced_lines
            for line in range(referenced - halo_lines, referenced + halo_lines + 1)
            if line in source_lines
        )
        target_facts = tuple(
            {
                "targetPath": path,
                "sourceValue": _resolve_target_path(source_label, path),
                "contextId": target_context_id(path),
            }
            for path in allowed_paths
            if path in context_paths
        )
        target_surfaces = {
            fact["targetPath"]: "".join(
                character.casefold()
                for character in str(fact["sourceValue"])
                if character.isalnum()
            )
            for fact in target_facts
            if isinstance(fact["sourceValue"], (str, int, float))
            and not isinstance(fact["sourceValue"], bool)
        }
        selected_inventory = tuple(
            row
            for row in prior_inventory
            if isinstance(row, Mapping)
            and (
                row.get("logicalKey") in selected_keys
                or set(cast(Sequence[str], row.get("targetPaths", ()))) & context_paths
            )
        )

    scoped_occurrence_candidates: list[dict[str, Any]] = []
    for occurrence, row_lines, _initial_matching_paths in materialized_occurrences:
        normalized_source = "".join(
            character.casefold()
            for character in str(occurrence[occurrence_columns[source_text_index]])
            if character.isalnum()
        )
        matching_paths = tuple(
            path
            for path, target_surface in target_surfaces.items()
            if min(len(normalized_source), len(target_surface)) >= distinctive_surface_length
            and (normalized_source in target_surface or target_surface in normalized_source)
        )
        if not (row_lines & initially_scoped_lines or matching_paths):
            continue
        identity = (
            str(occurrence[occurrence_columns[line_start_index]]),
            str(occurrence[occurrence_columns[line_end_index]]),
            str(occurrence[occurrence_columns[source_text_index]]),
            int(occurrence[occurrence_columns[occurrence_index_index]]),
        )
        occurrence["candidateTargetPaths"] = matching_paths
        occurrence["currentExactOwnerLogicalKeys"] = tuple(sorted(exact_owners.get(identity, ())))
        occurrence["outsideOriginalRepairWindow"] = not bool(row_lines & initially_scoped_lines)
        scoped_occurrence_candidates.append(occurrence)
    return {
        "contract": {
            "schemaVersion": 4,
            "operation": "local_compiler_repair",
            "completeCandidateRetainedByHost": True,
            "responseMustContainOnlyDeltaOperations": True,
            "priorPathsAbsentFromTargetFactsAreInvalid": True,
        },
        "documentId": payload.get("documentId"),
        "expectedCarrierName": payload.get("expectedCarrierName"),
        "requiredRevision": local_error,
        "sourceWindow": "\n".join(source_lines[line] for line in sorted(scoped_lines)),
        "occurrenceCandidates": scoped_occurrence_candidates,
        "targetFacts": target_facts,
        "targetContexts": target_contexts,
        "invalidPriorTargetPaths": tuple(sorted(invalid_prior_paths)),
        "requiredTargetCoBindings": selected_co_bindings,
        "candidateSlice": {
            "carrier": prior.carrier.model_dump(mode="json") if carrier_selected else None,
            "bindings": [row.model_dump(mode="json") for row in selected_bindings],
            "anchorOverrides": [row.model_dump(mode="json") for row in selected_overrides],
            "semanticOnlyTargetFacts": [row.model_dump(mode="json") for row in selected_semantic],
            "unresolved": prior.unresolved,
            "removableBindingKeys": [row.logical_key for row in selected_bindings],
            "removableAnchorOverrideIds": [row.anchor_binding_id for row in selected_overrides],
            "removableSemanticOnlyTargetPaths": [row.target_path for row in selected_semantic],
        },
        "anchorBindings": selected_anchors,
        "candidateBindingInventory": selected_inventory,
    }
