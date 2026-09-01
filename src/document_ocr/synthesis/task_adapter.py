"""Task-level schema, policy, and relational projection boundary for synthesis."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading_v3 import BillOfLadingRelationExplicitLabel
from document_ocr.label_schemas.bill_of_lading_v4 import BillOfLadingRelationExplicitV4Label
from document_ocr.label_schemas.bill_of_lading_v5 import BillOfLadingRelationExplicitV5Label
from document_ocr.synthesis.bill_of_lading_domain import ADAPTER, V4_ADAPTER, V5_ADAPTER
from document_ocr.synthesis.domain import DomainAdapter, RelationalTables
from document_ocr.synthesis.generation_models import FieldPolicy
from document_ocr.synthesis.policies import FIELD_POLICIES, V4_FIELD_POLICIES, V5_FIELD_POLICIES

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_DEFAULT_DOMAIN_ADAPTER: DomainAdapter = cast(DomainAdapter, ADAPTER)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class TaskAdapterReceipt(_StrictFrozenModel):
    """Serializable identity of one task adapter's executable contracts."""

    schema_version: Literal[1]
    task: NonEmptyString
    contract_id: NonEmptyString
    schema_sha256: Sha256
    field_policies_sha256: Sha256
    table_order: tuple[NonEmptyString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def tables_are_unique(self) -> TaskAdapterReceipt:
        if len(self.table_order) != len(set(self.table_order)):
            raise ValueError("task-adapter table order contains duplicates")
        return self


class SynthesisTaskAdapter(Protocol):
    """Minimal task contract required by the neutral scenario orchestrator."""

    task: str
    contract_id: str
    domain_adapter: DomainAdapter
    field_policies: Mapping[str, FieldPolicy]

    @property
    def receipt(self) -> TaskAdapterReceipt: ...

    def validate_target(self, *, document_id: str, target: Mapping[str, Any]) -> dict[str, Any]: ...

    def policy_for_target_path(self, path: str) -> FieldPolicy: ...


class BillOfLadingSynthesisTaskAdapter:
    """Relation-v3 B/L task wrapper around the existing lossless ADAPTER."""

    def __init__(
        self,
        *,
        task: str = ADAPTER.task,
        contract_id: str = "bill-of-lading-relation-explicit-v3-synthesis-adapter-v1",
        target_model: type[BaseModel] = BillOfLadingRelationExplicitLabel,
        domain_adapter: DomainAdapter = _DEFAULT_DOMAIN_ADAPTER,
        field_policies: Mapping[str, FieldPolicy] = FIELD_POLICIES,
    ) -> None:
        self.task = task
        self.contract_id = contract_id
        self.target_model = target_model
        self.domain_adapter = domain_adapter
        self.field_policies: Mapping[str, FieldPolicy] = MappingProxyType(dict(field_policies))
        schema = target_model.model_json_schema(mode="serialization")
        policies = {
            path: policy.model_dump(mode="json")
            for path, policy in sorted(self.field_policies.items())
        }
        self._receipt = TaskAdapterReceipt.model_validate(
            {
                "schema_version": 1,
                "task": self.task,
                "contract_id": self.contract_id,
                "schema_sha256": sha256_bytes(canonical_json_bytes(schema)),
                "field_policies_sha256": sha256_bytes(canonical_json_bytes(policies)),
                "table_order": self.domain_adapter.table_order,
            },
            strict=True,
        )

    @property
    def receipt(self) -> TaskAdapterReceipt:
        return self._receipt

    def validate_target(self, *, document_id: str, target: Mapping[str, Any]) -> dict[str, Any]:
        """Require canonical relation-v3 schema and exact relational inverse."""

        if not document_id.strip():
            raise ValueError("task-adapter document ID must be non-empty")
        validated = self.target_model.model_validate_json(
            canonical_json_bytes(dict(target)), strict=True
        )
        canonical = cast(dict[str, Any], cast(Any, validated).canonical_target())
        if canonical != target:
            raise ValueError("B/L synthesis target is not canonical")
        tables: RelationalTables = self.domain_adapter.project(
            document_id=document_id,
            source_row_index=0,
            target=canonical,
        )
        reconstructed = self.domain_adapter.reconstruct(
            document_id=document_id,
            tables=tables.rows,
        )
        if reconstructed != canonical:
            raise ValueError("B/L synthesis target fails relational inverse projection")
        return canonical

    def policy_for_target_path(self, path: str) -> FieldPolicy:
        from document_ocr.synthesis.anchors import normalized_role_path

        role_path = normalized_role_path(path)
        try:
            return self.field_policies[role_path]
        except KeyError as error:
            raise KeyError(f"no synthesis policy for target path: {path}") from error


BILL_OF_LADING_TASK_ADAPTER: SynthesisTaskAdapter = BillOfLadingSynthesisTaskAdapter()
BILL_OF_LADING_V4_TASK_ADAPTER: SynthesisTaskAdapter = BillOfLadingSynthesisTaskAdapter(
    task=V4_ADAPTER.task,
    contract_id="bill-of-lading-relation-explicit-v4-synthesis-adapter-v1",
    target_model=BillOfLadingRelationExplicitV4Label,
    domain_adapter=cast(DomainAdapter, V4_ADAPTER),
    field_policies=V4_FIELD_POLICIES,
)
BILL_OF_LADING_V5_TASK_ADAPTER: SynthesisTaskAdapter = BillOfLadingSynthesisTaskAdapter(
    task=V5_ADAPTER.task,
    contract_id="bill-of-lading-relation-explicit-v5-synthesis-adapter-v1",
    target_model=BillOfLadingRelationExplicitV5Label,
    domain_adapter=cast(DomainAdapter, V5_ADAPTER),
    field_policies=V5_FIELD_POLICIES,
)
