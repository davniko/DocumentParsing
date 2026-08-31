"""Composable, hash-chained state contracts for synthesis scenario stages.

The existing route and structured runners each own a complete target mutation
today.  These contracts provide the neutral boundary needed to compose those
mutations later without allowing a stage to overwrite an earlier stage's
facts, lose provenance, or claim a target change that is absent from its
ledger.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from document_ocr.hashing import canonical_json_bytes, identity_sha256, sha256_bytes
from document_ocr.synthesis.anchors import leaf_items, normalized_role_path

if TYPE_CHECKING:
    from document_ocr.synthesis.run_context import SynthesisRunContext

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ScenarioId = Annotated[str, StringConstraints(pattern=r"^syn_[0-9a-f]{64}$")]
StageId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[a-z][a-z0-9._-]{0,127}$"),
]
ScenarioNamespace = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[a-z][a-z0-9._-]{0,127}$"),
]
Seed = Annotated[int, Field(ge=0, lt=2**64)]

_TARGET_PATH = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])*)*$"
)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ScenarioIdentity(_StrictFrozenModel):
    """Stage-independent identity for one synthetic descendant."""

    schema_version: Literal[1]
    scenario_id: ScenarioId
    task: NonEmptyString
    scenario_namespace: ScenarioNamespace
    base_document_id: NonEmptyString
    template_id: NonEmptyString
    seed: Seed
    variant_index: Annotated[int, Field(ge=0, lt=2**64)]

    @model_validator(mode="after")
    def identifier_matches_inputs(self) -> ScenarioIdentity:
        expected = deterministic_scenario_id(
            task=self.task,
            scenario_namespace=self.scenario_namespace,
            base_document_id=self.base_document_id,
            template_id=self.template_id,
            seed=self.seed,
            variant_index=self.variant_index,
        )
        if self.scenario_id != expected:
            raise ValueError("scenario ID does not match its immutable identity inputs")
        return self


def deterministic_scenario_id(
    *,
    task: str,
    scenario_namespace: str,
    base_document_id: str,
    template_id: str,
    seed: int,
    variant_index: int,
) -> str:
    """Create one identity that is independent of stage order and target bytes."""

    values = {
        "task": task,
        "scenario_namespace": scenario_namespace,
        "base_document_id": base_document_id,
        "template_id": template_id,
    }
    if any(not value.strip() for value in values.values()):
        raise ValueError("scenario identity strings must be non-empty")
    if re.fullmatch(r"[a-z][a-z0-9._-]{0,127}", scenario_namespace) is None:
        raise ValueError("scenario namespace contains unsupported characters")
    if not 0 <= seed < 2**64 or not 0 <= variant_index < 2**64:
        raise ValueError("scenario seed and variant index must be non-negative uint64 values")
    digest = identity_sha256(
        "document-ocr-synthesis-scenario-v1",
        task,
        scenario_namespace,
        base_document_id,
        template_id,
        seed,
        variant_index,
    )
    return f"syn_{digest}"


def build_scenario_identity(
    *,
    task: str,
    scenario_namespace: str,
    base_document_id: str,
    template_id: str,
    seed: int,
    variant_index: int,
) -> ScenarioIdentity:
    """Validate and freeze the complete scenario identity tuple."""

    return ScenarioIdentity.model_validate(
        {
            "schema_version": 1,
            "scenario_id": deterministic_scenario_id(
                task=task,
                scenario_namespace=scenario_namespace,
                base_document_id=base_document_id,
                template_id=template_id,
                seed=seed,
                variant_index=variant_index,
            ),
            "task": task,
            "scenario_namespace": scenario_namespace,
            "base_document_id": base_document_id,
            "template_id": template_id,
            "seed": seed,
            "variant_index": variant_index,
        },
        strict=True,
    )


ScenarioChangeKind = Literal[
    "identifier",
    "temporal",
    "geography",
    "categorical",
    "numeric",
    "relationship",
    "linguistic",
    "structural",
]
RenderingKind = Literal[
    "exact_scalar",
    "formatted_date",
    "formatted_number",
    "linguistic",
    "structural_only",
]


class ScenarioChange(_StrictFrozenModel):
    """One exact leaf change owned by exactly one scenario stage."""

    target_path: NonEmptyString
    role_path: NonEmptyString
    stage_id: StageId
    change_kind: ScenarioChangeKind
    rendering_kind: RenderingKind
    old_present: bool
    old_value: JsonValue = None
    new_present: bool
    new_value: JsonValue = None
    method: NonEmptyString
    coupling_group: NonEmptyString

    @model_validator(mode="after")
    def path_and_values_are_exact(self) -> ScenarioChange:
        if _TARGET_PATH.fullmatch(self.target_path) is None:
            raise ValueError("scenario change target path is malformed")
        if self.role_path != normalized_role_path(self.target_path):
            raise ValueError("scenario change role path differs from its normalized target path")
        if not self.old_present and self.old_value is not None:
            raise ValueError("absent old value must be null")
        if not self.new_present and self.new_value is not None:
            raise ValueError("absent new value must be null")
        if not self.old_present and not self.new_present:
            raise ValueError("scenario change cannot be absent on both sides")
        if self.old_present == self.new_present and self.old_value == self.new_value:
            raise ValueError("scenario change must alter value presence or content")
        return self


class StageProvenanceReceipt(_StrictFrozenModel):
    """One immutable input/model/registry receipt used by a stage."""

    name: StageId
    sha256: Sha256


class AppliedStageReceipt(_StrictFrozenModel):
    """Hash-chain link retained in the cumulative scenario state."""

    schema_version: Literal[1]
    stage_id: StageId
    contract_id: NonEmptyString
    input_state_sha256: Sha256
    input_target_sha256: Sha256
    output_target_sha256: Sha256
    change_paths: tuple[NonEmptyString, ...]
    provenance: tuple[StageProvenanceReceipt, ...]

    @model_validator(mode="after")
    def collections_are_canonical(self) -> AppliedStageReceipt:
        if self.change_paths != tuple(sorted(set(self.change_paths))):
            raise ValueError("applied-stage change paths must be unique and sorted")
        names = tuple(row.name for row in self.provenance)
        if names != tuple(sorted(set(names))):
            raise ValueError("stage provenance names must be unique and sorted")
        return self


class ScenarioState(_StrictFrozenModel):
    """Complete semantic target plus an immutable, ordered stage history."""

    schema_version: Literal[1]
    identity: ScenarioIdentity
    source_target_sha256: Sha256
    target: dict[str, JsonValue]
    target_sha256: Sha256
    changes: tuple[ScenarioChange, ...]
    stage_receipts: tuple[AppliedStageReceipt, ...]

    @model_validator(mode="after")
    def target_and_history_are_consistent(self) -> ScenarioState:
        self.assert_integrity()
        stage_ids = tuple(row.stage_id for row in self.stage_receipts)
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("scenario state contains a duplicate applied stage")
        change_paths = tuple(row.target_path for row in self.changes)
        if len(change_paths) != len(set(change_paths)):
            raise ValueError("scenario state contains a duplicate changed target path")
        expected_changes = tuple(
            change
            for receipt in self.stage_receipts
            for change in self.changes
            if change.stage_id == receipt.stage_id
        )
        if expected_changes != self.changes:
            raise ValueError("scenario changes are not ordered by applied stage")
        if any(change.stage_id not in set(stage_ids) for change in self.changes):
            raise ValueError("scenario change references an unapplied stage")
        for receipt in self.stage_receipts:
            paths = tuple(
                change.target_path for change in self.changes if change.stage_id == receipt.stage_id
            )
            if paths != receipt.change_paths:
                raise ValueError("applied-stage receipt differs from its scenario changes")
        if not self.stage_receipts and (
            self.changes or self.target_sha256 != self.source_target_sha256
        ):
            raise ValueError("initial scenario state must equal its source target")
        return self

    def assert_integrity(self) -> None:
        """Detect mutation of the otherwise frozen model's nested target value."""

        if sha256_bytes(canonical_json_bytes(self.target)) != self.target_sha256:
            raise ValueError("scenario target SHA-256 does not match its payload")

    def content_sha256(self) -> str:
        """Hash the full state after rechecking nested target integrity."""

        self.assert_integrity()
        return sha256_bytes(canonical_json_bytes(self.model_dump(mode="json")))


class ScenarioStageProposal(_StrictFrozenModel):
    """A stage-authored proposal that cannot advance state without host checks."""

    schema_version: Literal[1]
    stage_id: StageId
    contract_id: NonEmptyString
    input_state_sha256: Sha256
    target: dict[str, JsonValue]
    target_sha256: Sha256
    changes: tuple[ScenarioChange, ...]
    provenance: tuple[StageProvenanceReceipt, ...] = ()

    @model_validator(mode="after")
    def proposal_is_canonical(self) -> ScenarioStageProposal:
        self.assert_integrity()
        paths = tuple(row.target_path for row in self.changes)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("stage proposal change paths must be unique and sorted")
        if any(row.stage_id != self.stage_id for row in self.changes):
            raise ValueError("stage proposal contains a change owned by another stage")
        names = tuple(row.name for row in self.provenance)
        if names != tuple(sorted(set(names))):
            raise ValueError("stage proposal provenance names must be unique and sorted")
        return self

    def assert_integrity(self) -> None:
        if sha256_bytes(canonical_json_bytes(self.target)) != self.target_sha256:
            raise ValueError("stage proposal target SHA-256 does not match its payload")


class ScenarioStageResult(_StrictFrozenModel):
    """Host-validated transition with explicit input and output state hashes."""

    schema_version: Literal[1]
    input_state_sha256: Sha256
    output_state_sha256: Sha256
    stage_receipt: AppliedStageReceipt
    output_state: ScenarioState

    @model_validator(mode="after")
    def hashes_and_receipt_match(self) -> ScenarioStageResult:
        if self.stage_receipt.input_state_sha256 != self.input_state_sha256:
            raise ValueError("stage result input hash differs from its receipt")
        if self.output_state.content_sha256() != self.output_state_sha256:
            raise ValueError("stage result output hash differs from its state")
        if (
            not self.output_state.stage_receipts
            or self.output_state.stage_receipts[-1] != self.stage_receipt
        ):
            raise ValueError("stage result receipt is not the final output-state receipt")
        return self


class ScenarioStage(Protocol):
    """Pure stage boundary; the run context validates and commits proposals."""

    stage_id: str
    contract_id: str

    def propose(
        self, *, context: SynthesisRunContext, state: ScenarioState
    ) -> ScenarioStageProposal: ...


def _leaf_map(target: Mapping[str, Any]) -> dict[str, Any]:
    return dict(leaf_items(target))


def _validate_exact_change_ledger(
    *,
    source_target: Mapping[str, Any],
    proposed_target: Mapping[str, Any],
    changes: Sequence[ScenarioChange],
) -> None:
    source_leaves = _leaf_map(source_target)
    proposed_leaves = _leaf_map(proposed_target)
    changed_paths = tuple(
        sorted(
            path
            for path in source_leaves.keys() | proposed_leaves.keys()
            if (path in source_leaves) != (path in proposed_leaves)
            or source_leaves.get(path) != proposed_leaves.get(path)
        )
    )
    ledger_paths = tuple(row.target_path for row in changes)
    if changed_paths != ledger_paths:
        missing = sorted(set(changed_paths) - set(ledger_paths))
        extra = sorted(set(ledger_paths) - set(changed_paths))
        raise ValueError(f"scenario change ledger mismatch: missing={missing}, extra={extra}")
    if source_target != proposed_target and not changed_paths:
        raise ValueError("scenario structural change cannot be represented as a leaf ledger")
    for change in changes:
        old_present = change.target_path in source_leaves
        new_present = change.target_path in proposed_leaves
        if (
            change.old_present != old_present
            or change.new_present != new_present
            or (old_present and change.old_value != source_leaves[change.target_path])
            or (new_present and change.new_value != proposed_leaves[change.target_path])
        ):
            raise ValueError(f"scenario change values differ at {change.target_path}")


def advance_scenario_state(
    *, state: ScenarioState, proposal: ScenarioStageProposal
) -> ScenarioStageResult:
    """Validate and commit one exact, non-overwriting stage transition."""

    input_state_sha256 = state.content_sha256()
    proposal.assert_integrity()
    if proposal.input_state_sha256 != input_state_sha256:
        raise ValueError("stage proposal was built from a different input state")
    if proposal.stage_id in {row.stage_id for row in state.stage_receipts}:
        raise ValueError(f"scenario stage was already applied: {proposal.stage_id}")
    previous_paths = {row.target_path for row in state.changes}
    repeated_paths = sorted(previous_paths & {row.target_path for row in proposal.changes})
    if repeated_paths:
        raise ValueError(f"scenario stage attempts to overwrite changed paths: {repeated_paths}")
    _validate_exact_change_ledger(
        source_target=state.target,
        proposed_target=proposal.target,
        changes=proposal.changes,
    )
    receipt = AppliedStageReceipt.model_validate(
        {
            "schema_version": 1,
            "stage_id": proposal.stage_id,
            "contract_id": proposal.contract_id,
            "input_state_sha256": input_state_sha256,
            "input_target_sha256": state.target_sha256,
            "output_target_sha256": proposal.target_sha256,
            "change_paths": tuple(row.target_path for row in proposal.changes),
            "provenance": tuple(row.model_dump(mode="python") for row in proposal.provenance),
        },
        strict=True,
    )
    output_target = json.loads(canonical_json_bytes(proposal.target))
    output = ScenarioState.model_validate(
        {
            "schema_version": 1,
            "identity": state.identity.model_dump(mode="python"),
            "source_target_sha256": state.source_target_sha256,
            "target": output_target,
            "target_sha256": proposal.target_sha256,
            "changes": tuple(row.model_dump(mode="python") for row in state.changes)
            + tuple(row.model_dump(mode="python") for row in proposal.changes),
            "stage_receipts": tuple(
                row.model_dump(mode="python") for row in (*state.stage_receipts, receipt)
            ),
        },
        strict=True,
    )
    return ScenarioStageResult.model_validate(
        {
            "schema_version": 1,
            "input_state_sha256": input_state_sha256,
            "output_state_sha256": output.content_sha256(),
            "stage_receipt": receipt.model_dump(mode="python"),
            "output_state": output.model_dump(mode="python"),
        },
        strict=True,
    )
