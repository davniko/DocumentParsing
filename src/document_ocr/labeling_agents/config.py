"""Strict configuration for concurrent PydanticAI labeling runs."""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from document_ocr.config import load_strict_yaml_mapping
from document_ocr.label_schemas.common import LabelSchemaModel

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class _ConfigModel(LabelSchemaModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def _absolute_safe_directory(value: str, field_name: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    normalized = Path(os.path.abspath(value))
    if normalized == Path(normalized.anchor):
        raise ValueError(f"{field_name} must not be the filesystem root")
    return value


def _safe_relative_file(value: str, suffix: str, field_name: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a safe relative POSIX path")
    if path.suffix != suffix:
        raise ValueError(f"{field_name} must end in {suffix}")
    return value


class WorkItemRootConfig(_ConfigModel):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]*$")]
    root: NonEmptyString
    include_glob: Literal["*.json"]
    selection_records_path: NonEmptyString
    selection_records_sha256: Sha256
    documents: PositiveInteger

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_safe_directory(value, "work-item root")

    @field_validator("selection_records_path")
    @classmethod
    def records_path_is_absolute_jsonl(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or path.suffix != ".jsonl":
            raise ValueError("selection_records_path must be an absolute .jsonl path")
        return value


class ExtractionRunRootConfig(_ConfigModel):
    run_id: NonEmptyString
    root: NonEmptyString

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_safe_directory(value, "extraction-run root")


class LabelingSourceConfig(_ConfigModel):
    reference_target_mode: Literal["required", "absent"]
    expected_documents: PositiveInteger
    expected_pages: PositiveInteger
    expected_inventory_sha256: Sha256
    work_item_roots: list[WorkItemRootConfig] = Field(min_length=1)
    extraction_runs: list[ExtractionRunRootConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def roots_are_unique_and_counts_match(self) -> LabelingSourceConfig:
        root_ids = [row.id for row in self.work_item_roots]
        roots = [row.root for row in self.work_item_roots]
        run_ids = [row.run_id for row in self.extraction_runs]
        if len(root_ids) != len(set(root_ids)) or len(roots) != len(set(roots)):
            raise ValueError("work-item root ids and paths must be unique")
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("extraction run ids must be unique")
        if sum(row.documents for row in self.work_item_roots) != self.expected_documents:
            raise ValueError("work-item root counts must sum to expected_documents")
        return self


class SelectionDocumentIdsFileConfig(_ConfigModel):
    path: NonEmptyString
    sha256: Sha256
    records: PositiveInteger

    @field_validator("path")
    @classmethod
    def path_is_absolute_jsonl(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or path.suffix != ".jsonl":
            raise ValueError("selection document_ids_file must be an absolute .jsonl path")
        return value


class SelectionConfig(_ConfigModel):
    count: PositiveInteger
    seed: Annotated[int, Field(ge=0)]
    namespace: NonEmptyString
    document_ids: list[
        Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
    ] | None = None
    document_ids_file: SelectionDocumentIdsFileConfig | None = None

    @model_validator(mode="after")
    def explicit_ids_match_count(self) -> SelectionConfig:
        if self.document_ids is not None and self.document_ids_file is not None:
            raise ValueError("configure document_ids or document_ids_file, not both")
        if self.document_ids is not None:
            if len(self.document_ids) != self.count:
                raise ValueError("explicit document_ids length must equal selection count")
            if len(set(self.document_ids)) != len(self.document_ids):
                raise ValueError("explicit document_ids must be unique")
        if (
            self.document_ids_file is not None
            and self.document_ids_file.records != self.count
        ):
            raise ValueError("document_ids_file records must equal selection count")
        return self


class PromptArtifactConfig(_ConfigModel):
    path: NonEmptyString
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def path_is_safe_markdown(cls, value: str) -> str:
        return _safe_relative_file(value, ".md", "prompt path")


class PromptSetConfig(_ConfigModel):
    extractor: PromptArtifactConfig
    reviewer: PromptArtifactConfig
    document_layout: PromptArtifactConfig


class OpenAIPricingConfig(_ConfigModel):
    currency: Literal["USD"]
    effective_date: date
    source_url: NonEmptyString
    input_usd_per_million: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    cached_input_usd_per_million: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    cache_write_multiplier: Annotated[float, Field(ge=1, allow_inf_nan=False)]
    output_usd_per_million: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    long_context_input_threshold_tokens: PositiveInteger
    long_context_input_multiplier: Annotated[float, Field(ge=1, allow_inf_nan=False)]
    long_context_output_multiplier: Annotated[float, Field(ge=1, allow_inf_nan=False)]

    @field_validator("source_url")
    @classmethod
    def source_is_https(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("pricing source_url must be an absolute HTTPS URL")
        return value


class OpenAIResponsesProviderConfig(_ConfigModel):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]*$")]
    kind: Literal["openai_responses"]
    model: Literal["gpt-5.6-luna", "gpt-5.6-terra"]
    api_key_env: Literal["OPENAI_API_KEY"]
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"]
    supports_pdf_documents: Literal[True]
    request_timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    transport_max_retries: Annotated[int, Field(ge=0, le=5)]
    max_output_tokens: PositiveInteger
    pricing: OpenAIPricingConfig


class OllamaProviderConfig(_ConfigModel):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]*$")]
    kind: Literal["ollama_openai_chat"]
    model: NonEmptyString
    base_url: NonEmptyString
    api_key: Literal["ollama"]
    native_json_schema: Literal[True]
    supports_pdf_documents: Literal[False]
    # The OpenAI-compatible Ollama endpoint does not expose a portable thinking
    # control across arbitrary model families. Keep provider/model defaults rather
    # than claiming to disable reasoning when no protocol-level guarantee exists.
    reasoning_mode: Literal["provider_default"]
    request_timeout_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    transport_max_retries: Annotated[int, Field(ge=0, le=5)]
    max_output_tokens: PositiveInteger

    @field_validator("base_url")
    @classmethod
    def base_url_is_openai_compatible(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Ollama base_url must be absolute HTTP(S)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Ollama base_url must not contain credentials, query, or fragment")
        if parsed.path.rstrip("/") != "/v1":
            raise ValueError("Ollama base_url must end in /v1")
        return value


ProviderConfig = Annotated[
    OpenAIResponsesProviderConfig | OllamaProviderConfig,
    Field(discriminator="kind"),
]


class ProviderAssignmentsConfig(_ConfigModel):
    labeler: NonEmptyString
    reviewer: NonEmptyString
    document_layout: NonEmptyString


class LabelingWorkflowConfig(_ConfigModel):
    max_concurrent_documents: PositiveInteger
    max_concurrent_model_requests: PositiveInteger
    max_candidate_attempts: PositiveInteger
    max_document_escalations_per_document: Annotated[int, Field(ge=0)]
    max_pdf_bytes: Annotated[int, Field(gt=0, le=50_000_000)]
    structured_output_retries: Annotated[int, Field(ge=0, le=3)]
    max_requests_per_agent_run: PositiveInteger
    input_tokens_limit_per_agent_run: PositiveInteger
    output_tokens_limit_per_agent_run: PositiveInteger
    document_mode: Literal["on_explicit_request"]
    pdf_page_scope: Literal["requested_pages"]
    require_independent_review: Literal[True]
    fail_fast: bool


class LabelingRunConfig(_ConfigModel):
    run_id: NonEmptyString
    output_root: NonEmptyString
    resume: bool

    @field_validator("run_id")
    @classmethod
    def run_id_is_safe(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError("run_id contains unsupported characters")
        return value

    @field_validator("output_root")
    @classmethod
    def output_root_is_absolute(cls, value: str) -> str:
        return _absolute_safe_directory(value, "labeling output root")


class AgentLabelingConfig(_ConfigModel):
    schema_version: Literal[3, 4]
    task: Literal[
        "bill_of_lading_dual_cargo_v3",
        "bill_of_lading_relation_single_source_v4",
    ]
    environment_file: Literal[".env"]
    run: LabelingRunConfig
    source: LabelingSourceConfig
    selection: SelectionConfig
    prompts: PromptSetConfig
    providers: list[ProviderConfig] = Field(min_length=1)
    assignments: ProviderAssignmentsConfig
    workflow: LabelingWorkflowConfig

    @model_validator(mode="after")
    def providers_and_selection_are_consistent(self) -> AgentLabelingConfig:
        expected_task = {
            3: "bill_of_lading_dual_cargo_v3",
            4: "bill_of_lading_relation_single_source_v4",
        }[self.schema_version]
        if self.task != expected_task:
            raise ValueError(
                f"schema_version {self.schema_version} requires task {expected_task!r}"
            )
        if self.selection.count > self.source.expected_documents:
            raise ValueError("selection.count must not exceed source.expected_documents")
        provider_ids = [row.id for row in self.providers]
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("provider ids must be unique")
        configured = set(provider_ids)
        assigned = {
            self.assignments.labeler,
            self.assignments.reviewer,
            self.assignments.document_layout,
        }
        if not assigned.issubset(configured):
            raise ValueError("every provider assignment must reference a configured provider")
        by_id = {row.id: row for row in self.providers}
        document_provider = by_id[self.assignments.document_layout]
        if (
            self.workflow.max_document_escalations_per_document > 0
            and not document_provider.supports_pdf_documents
        ):
            raise ValueError(
                "document-layout provider must support PDF documents when escalation is enabled"
            )
        return self


def load_agent_labeling_config(path: str | Path) -> AgentLabelingConfig:
    return AgentLabelingConfig.model_validate(load_strict_yaml_mapping(path), strict=True)
