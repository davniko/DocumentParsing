"""Strict state contracts for deterministic synthesis generation stages."""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


ValuePolicy = Literal[
    "regenerate",
    "resample",
    "derive",
    "preserve_nonidentifying",
    "pending_linguistic",
    "legitimately_absent",
    "forbidden",
]
MissingnessPolicy = Literal["preserve_source", "must_be_absent"]
ImplementationStatus = Literal[
    "implemented",
    "pending_registry",
    "pending_linguistic",
    "structural",
    "disabled",
]


class FieldPolicy(BaseModel):
    """One exhaustive task-leaf policy, independent of source presence."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    role_path: NonEmptyString
    value_policy: ValuePolicy
    missingness_policy: MissingnessPolicy
    implementation_status: ImplementationStatus
    method: NonEmptyString
    coupling_group: NonEmptyString


class PendingRealization(_StrictModel):
    """Explicit work that prevents a draft from becoming a training target."""

    target_path: NonEmptyString
    role_path: NonEmptyString
    kind: Literal[
        "registry",
        "linguistic",
        "deterministic_unsupported",
        "unrenderable_evidence",
    ]
    reason: NonEmptyString


class SemanticChange(_StrictModel):
    """One deterministic semantic mutation or derived consequence."""

    target_path: NonEmptyString
    role_path: NonEmptyString
    family: Literal[
        "document_identifier",
        "voyage_identifier",
        "container_identifier",
        "seal_identifier",
        "document_date",
        "package_quantity",
        "cargo_measure",
        "allocation_reference",
        "allocation_quantity",
    ]
    old_value: JsonValue
    new_value: JsonValue
    method: NonEmptyString
    coupling_group: NonEmptyString

    @model_validator(mode="after")
    def value_changes(self) -> SemanticChange:
        if self.old_value == self.new_value:
            raise ValueError("semantic change must alter the value")
        return self


class DraftScenarioPlan(_StrictModel):
    """Non-publishable deterministic work with unresolved tasks made explicit."""

    schema_version: Literal[1]
    status: Literal["draft_pending_realization"]
    synthetic_document_id: NonEmptyString
    base_document_id: NonEmptyString
    template_id: NonEmptyString
    variant_index: Annotated[int, Field(ge=0)]
    seed: int
    source_raw_text_sha256: Sha256
    source_target_sha256: Sha256
    deterministic_target_sha256: Sha256
    changes: tuple[SemanticChange, ...]
    pending_realizations: tuple[PendingRealization, ...]
    policy_counts: dict[str, int]
    training_eligible: Literal[False]

    @model_validator(mode="after")
    def unique_paths(self) -> DraftScenarioPlan:
        paths = [row.target_path for row in self.changes]
        if len(paths) != len(set(paths)):
            raise ValueError("draft semantic change paths must be unique")
        pending = [row.target_path for row in self.pending_realizations]
        if len(pending) != len(set(pending)):
            raise ValueError("draft pending-realization paths must be unique")
        return self


class DeterministicTextEdit(_StrictModel):
    """An exact, locally resolved edit; offsets are never model-produced."""

    edit_id: NonEmptyString
    page_number: Annotated[int, Field(gt=0)]
    page_start: Annotated[int, Field(ge=0)]
    page_end: Annotated[int, Field(gt=0)]
    exact_old_text: NonEmptyString
    replacement_text: NonEmptyString
    target_paths: tuple[NonEmptyString, ...]
    family: NonEmptyString
    coupling_group: NonEmptyString

    @model_validator(mode="after")
    def valid_span(self) -> DeterministicTextEdit:
        if self.page_end - self.page_start != len(self.exact_old_text):
            raise ValueError("text-edit span length differs from exact old text")
        if not self.target_paths or len(self.target_paths) != len(set(self.target_paths)):
            raise ValueError("text edit target paths must be non-empty and unique")
        return self


class DeterministicTextPatchPlan(_StrictModel):
    """A fully local patch plan for the deterministic subset of one draft."""

    schema_version: Literal[1]
    synthetic_document_id: NonEmptyString
    source_raw_text_sha256: Sha256
    deterministic_target_sha256: Sha256
    edits: tuple[DeterministicTextEdit, ...]
    unrendered_change_paths: tuple[NonEmptyString, ...]
    planner: Literal["deterministic_anchor_context_v1"]

    @model_validator(mode="after")
    def disjoint_edits(self) -> DeterministicTextPatchPlan:
        by_page: dict[int, list[DeterministicTextEdit]] = {}
        for edit in self.edits:
            by_page.setdefault(edit.page_number, []).append(edit)
        for rows in by_page.values():
            ordered = sorted(rows, key=lambda row: (row.page_start, row.page_end))
            for left, right in pairwise(ordered):
                if left.page_end > right.page_start:
                    raise ValueError("deterministic text edits overlap")
        return self


class RenderedDraftReceipt(_StrictModel):
    """Proof that deterministic edits were applied without publishing a final label."""

    schema_version: Literal[1]
    synthetic_document_id: NonEmptyString
    rendered_raw_text_sha256: Sha256
    changed_source_characters: Annotated[int, Field(ge=0)]
    unchanged_source_characters: Annotated[int, Field(ge=0)]
    edit_count: Annotated[int, Field(ge=0)]
    page_count: Annotated[int, Field(gt=0)]
    exact_unchanged_regions: Literal[True]
    target_schema_valid: Literal[True]
    relational_inverse_valid: Literal[True]
    all_deterministic_changes_rendered: Literal[True]
    training_eligible: Literal[False]


def json_value(value: Any) -> JsonValue:
    """Validate a Python value at the explicit JSON boundary."""

    from pydantic import TypeAdapter

    return TypeAdapter(JsonValue).validate_python(value, strict=True)
