"""Immutable source/task boundary shared by composable synthesis stages."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.scenario_state import (
    ScenarioNamespace,
    ScenarioStage,
    ScenarioStageProposal,
    ScenarioStageResult,
    ScenarioState,
    advance_scenario_state,
    build_scenario_identity,
)
from document_ocr.synthesis.task_adapter import SynthesisTaskAdapter, TaskAdapterReceipt

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_SCENARIO_NAMESPACE_ADAPTER = TypeAdapter(ScenarioNamespace)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ScenarioSourceRecord(_StrictFrozenModel):
    """One canonical source target and its template/raw-text provenance."""

    document_id: NonEmptyString
    template_id: NonEmptyString
    source_raw_text_sha256: Sha256 | None = None
    target: dict[str, JsonValue]
    target_sha256: Sha256

    @model_validator(mode="after")
    def target_hash_matches(self) -> ScenarioSourceRecord:
        if sha256_bytes(canonical_json_bytes(self.target)) != self.target_sha256:
            raise ValueError("scenario source target SHA-256 does not match its payload")
        return self

    @classmethod
    def from_target(
        cls,
        *,
        document_id: str,
        template_id: str,
        target: dict[str, Any],
        source_raw_text_sha256: str | None = None,
    ) -> ScenarioSourceRecord:
        payload = canonical_json_bytes(target)
        return cls.model_validate(
            {
                "document_id": document_id,
                "template_id": template_id,
                "source_raw_text_sha256": source_raw_text_sha256,
                "target": json.loads(payload),
                "target_sha256": sha256_bytes(payload),
            },
            strict=True,
        )


class SynthesisRunContextReceipt(_StrictFrozenModel):
    """Compact immutable receipt for the validated run input boundary."""

    schema_version: Literal[1]
    scenario_namespace: ScenarioNamespace
    task_adapter: TaskAdapterReceipt
    source_document_count: Annotated[int, Field(gt=0)]
    source_inventory_sha256: Sha256
    context_sha256: Sha256

    @model_validator(mode="after")
    def context_hash_matches(self) -> SynthesisRunContextReceipt:
        body = {
            "schemaVersion": self.schema_version,
            "scenarioNamespace": self.scenario_namespace,
            "taskAdapter": self.task_adapter.model_dump(mode="json"),
            "sourceDocumentCount": self.source_document_count,
            "sourceInventorySha256": self.source_inventory_sha256,
        }
        if sha256_bytes(canonical_json_bytes(body)) != self.context_sha256:
            raise ValueError("synthesis run-context SHA-256 does not match its receipt")
        return self


@dataclass(frozen=True, slots=True)
class _SourceMetadata:
    document_id: str
    template_id: str
    source_raw_text_sha256: str | None
    target_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "documentId": self.document_id,
            "templateId": self.template_id,
            "sourceRawTextSha256": self.source_raw_text_sha256,
            "targetSha256": self.target_sha256,
        }


class SynthesisRunContext:
    """Validated, read-only source inventory used by every scenario stage."""

    __slots__ = (
        "_adapter",
        "_metadata",
        "_receipt",
        "_scenario_namespace",
        "_target_bytes",
    )

    def __init__(
        self,
        *,
        task_adapter: SynthesisTaskAdapter,
        scenario_namespace: str,
        source_records: Sequence[ScenarioSourceRecord],
    ) -> None:
        if not source_records:
            raise ValueError("synthesis run context requires at least one source record")
        if len({row.document_id for row in source_records}) != len(source_records):
            raise ValueError("synthesis run context contains duplicate source document IDs")
        validated_namespace = _SCENARIO_NAMESPACE_ADAPTER.validate_python(
            scenario_namespace, strict=True
        )
        metadata: dict[str, _SourceMetadata] = {}
        target_bytes: dict[str, bytes] = {}
        for record in sorted(source_records, key=lambda row: row.document_id):
            canonical = task_adapter.validate_target(
                document_id=record.document_id,
                target=record.target,
            )
            payload = canonical_json_bytes(canonical)
            if sha256_bytes(payload) != record.target_sha256:
                raise ValueError(
                    f"task adapter changed the pinned source target: {record.document_id}"
                )
            metadata[record.document_id] = _SourceMetadata(
                document_id=record.document_id,
                template_id=record.template_id,
                source_raw_text_sha256=record.source_raw_text_sha256,
                target_sha256=record.target_sha256,
            )
            target_bytes[record.document_id] = payload
        inventory = [row.to_dict() for row in metadata.values()]
        inventory_sha256 = sha256_bytes(canonical_json_bytes(inventory))
        context_body = {
            "schemaVersion": 1,
            "scenarioNamespace": validated_namespace,
            "taskAdapter": task_adapter.receipt.model_dump(mode="json"),
            "sourceDocumentCount": len(metadata),
            "sourceInventorySha256": inventory_sha256,
        }
        receipt = SynthesisRunContextReceipt.model_validate(
            {
                "schema_version": 1,
                "scenario_namespace": validated_namespace,
                "task_adapter": task_adapter.receipt.model_dump(mode="python"),
                "source_document_count": len(metadata),
                "source_inventory_sha256": inventory_sha256,
                "context_sha256": sha256_bytes(canonical_json_bytes(context_body)),
            },
            strict=True,
        )
        self._adapter = task_adapter
        self._scenario_namespace = receipt.scenario_namespace
        self._metadata = MappingProxyType(metadata)
        self._target_bytes = MappingProxyType(target_bytes)
        self._receipt = receipt

    @property
    def task_adapter(self) -> SynthesisTaskAdapter:
        return self._adapter

    @property
    def scenario_namespace(self) -> str:
        return self._scenario_namespace

    @property
    def receipt(self) -> SynthesisRunContextReceipt:
        return self._receipt

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(self._metadata)

    def source_record(self, document_id: str) -> ScenarioSourceRecord:
        try:
            metadata = self._metadata[document_id]
            payload = self._target_bytes[document_id]
        except KeyError as error:
            raise KeyError(f"unknown synthesis source document: {document_id}") from error
        return ScenarioSourceRecord.model_validate(
            {
                "document_id": metadata.document_id,
                "template_id": metadata.template_id,
                "source_raw_text_sha256": metadata.source_raw_text_sha256,
                "target": json.loads(payload),
                "target_sha256": metadata.target_sha256,
            },
            strict=True,
        )

    def initial_state(
        self, *, base_document_id: str, seed: int, variant_index: int
    ) -> ScenarioState:
        source = self.source_record(base_document_id)
        identity = build_scenario_identity(
            task=self._adapter.task,
            scenario_namespace=self._scenario_namespace,
            base_document_id=source.document_id,
            template_id=source.template_id,
            seed=seed,
            variant_index=variant_index,
        )
        return ScenarioState.model_validate(
            {
                "schema_version": 1,
                "identity": identity.model_dump(mode="python"),
                "source_target_sha256": source.target_sha256,
                "target": source.target,
                "target_sha256": source.target_sha256,
                "changes": (),
                "stage_receipts": (),
            },
            strict=True,
        )

    def validate_state(self, state: ScenarioState) -> None:
        state.assert_integrity()
        source = self.source_record(state.identity.base_document_id)
        identity = state.identity
        if (
            identity.task != self._adapter.task
            or identity.scenario_namespace != self._scenario_namespace
            or identity.template_id != source.template_id
            or state.source_target_sha256 != source.target_sha256
        ):
            raise ValueError("scenario state belongs to a different run context")
        self._adapter.validate_target(
            document_id=identity.scenario_id,
            target=state.target,
        )

    def advance(
        self, *, state: ScenarioState, proposal: ScenarioStageProposal
    ) -> ScenarioStageResult:
        self.validate_state(state)
        canonical = self._adapter.validate_target(
            document_id=state.identity.scenario_id,
            target=proposal.target,
        )
        if canonical != proposal.target:
            raise ValueError("stage proposal target is not canonical for the task adapter")
        result = advance_scenario_state(state=state, proposal=proposal)
        self.validate_state(result.output_state)
        return result

    def execute_stage(self, *, stage: ScenarioStage, state: ScenarioState) -> ScenarioStageResult:
        proposal = stage.propose(context=self, state=state)
        if proposal.stage_id != stage.stage_id or proposal.contract_id != stage.contract_id:
            raise ValueError("scenario stage proposal identity differs from its implementation")
        return self.advance(state=state, proposal=proposal)
