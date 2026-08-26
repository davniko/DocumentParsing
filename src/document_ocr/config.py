"""Strict configuration contracts for the document OCR extraction pipeline.

The pipeline configuration is intentionally explicit at reproducibility
boundaries.  In particular, a source snapshot, output location, run identity,
and immutable model revision must all be supplied by the caller.
"""

from __future__ import annotations

import os
import re
from collections.abc import Hashable
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]
NonNegativeInteger = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

_SHA_40_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_S3_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""

    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> dict[Any, Any]:
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                )
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


class _StrictConfigModel(BaseModel):
    """Base for YAML-facing models: no coercion and no unknown settings."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class LocalSourceConfig(_StrictConfigModel):
    """A content-addressed local PDF source snapshot."""

    type: Literal["local"]
    root: NonEmptyString
    include_glob: NonEmptyString
    dataset_version: NonEmptyString
    require_content_sha256: Literal[True] = True

    @field_validator("root")
    @classmethod
    def root_must_be_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("local source root must be an absolute path")
        return value


class S3SourceConfig(_StrictConfigModel):
    """An S3 source whose objects must be read by exact VersionId."""

    type: Literal["s3"]
    bucket: NonEmptyString
    prefix: NonEmptyString
    include_glob: NonEmptyString
    region: NonEmptyString
    dataset_version: NonEmptyString
    require_object_version_ids: Literal[True] = True

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, value: str) -> str:
        if not _S3_BUCKET_PATTERN.fullmatch(value):
            raise ValueError("bucket must be a valid lowercase S3 bucket name")
        if ".." in value or ".-" in value or "-." in value:
            raise ValueError("bucket must be a valid S3 bucket name")
        return value

    @field_validator("prefix")
    @classmethod
    def prefix_is_not_an_absolute_uri(cls, value: str) -> str:
        if value.startswith("s3://"):
            raise ValueError("prefix must be a key prefix, not an s3:// URI")
        return value


SourceConfig = Annotated[
    LocalSourceConfig | S3SourceConfig,
    Field(discriminator="type"),
]


class ClassificationManifestConfig(_StrictConfigModel):
    """One content-pinned classification manifest used for local staging."""

    key: NonEmptyString
    expected_sha256: str
    local_name: NonEmptyString

    @field_validator("key")
    @classmethod
    def key_is_not_an_absolute_uri(cls, value: str) -> str:
        if value.startswith("s3://"):
            raise ValueError("classification manifest key must not be an s3:// URI")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("classification manifest key must be a safe S3 object key")
        return value

    @field_validator("expected_sha256")
    @classmethod
    def expected_hash_is_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("expected_sha256 must be a lowercase 64-character SHA-256")
        return value

    @field_validator("local_name")
    @classmethod
    def local_name_is_a_single_jsonl_filename(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.name != value or path.suffix != ".jsonl":
            raise ValueError("local_name must be a single .jsonl filename")
        return value


class SnapshotDownloadConfig(_StrictConfigModel):
    """Bounded S3 download and CPU inspection concurrency."""

    workers: PositiveInteger
    chunk_size_bytes: Annotated[int, Field(ge=64 * 1024, le=64 * 1024 * 1024)]
    page_inspection_processes: PositiveInteger
    max_attempts: PositiveInteger
    max_pdf_bytes: PositiveInteger


class PageCountQuarantineConfig(_StrictConfigModel):
    """An explicitly pinned malformed PDF excluded from extraction."""

    source_key: NonEmptyString
    source_sha256: str
    classified_page_count: PositiveInteger
    actual_page_count: PositiveInteger
    reason: NonEmptyString

    @field_validator("source_key")
    @classmethod
    def source_key_is_safe(cls, value: str) -> str:
        if value.startswith("s3://"):
            raise ValueError("quarantine source_key must not be an s3:// URI")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("quarantine source_key must be a safe S3 object key")
        return value

    @field_validator("source_sha256")
    @classmethod
    def source_hash_is_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("source_sha256 must be a lowercase 64-character SHA-256")
        return value


class S3LocalSnapshotConfig(_StrictConfigModel):
    """Immutable local mirror of a classifier-selected, unversioned S3 corpus."""

    schema_version: Literal[1]
    bucket: NonEmptyString
    raw_prefix: NonEmptyString
    region: NonEmptyString
    destination_root: NonEmptyString
    document_types: list[NonEmptyString] = Field(min_length=1)
    classification_label_mapping: dict[NonEmptyString, NonEmptyString] = Field(min_length=1)
    classification_manifests: list[ClassificationManifestConfig] = Field(min_length=1)
    expected_selection_sha256: str
    page_count_quarantine: list[PageCountQuarantineConfig]
    download: SnapshotDownloadConfig

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, value: str) -> str:
        if not _S3_BUCKET_PATTERN.fullmatch(value):
            raise ValueError("bucket must be a valid lowercase S3 bucket name")
        if ".." in value or ".-" in value or "-." in value:
            raise ValueError("bucket must be a valid S3 bucket name")
        return value

    @field_validator("raw_prefix")
    @classmethod
    def raw_prefix_is_a_safe_key_prefix(cls, value: str) -> str:
        if value.startswith("s3://"):
            raise ValueError("raw_prefix must be a key prefix, not an s3:// URI")
        if not value.endswith("/"):
            raise ValueError("raw_prefix must end with '/'")
        path = PurePosixPath(value.removesuffix("/"))
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("raw_prefix must be a safe S3 key prefix")
        return value

    @field_validator("destination_root")
    @classmethod
    def destination_root_is_safe_and_absolute(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("destination_root must be an absolute local path")
        normalized = Path(os.path.abspath(value))
        if normalized == Path(normalized.anchor):
            raise ValueError("destination_root must not be the filesystem root")
        return value

    @field_validator("expected_selection_sha256")
    @classmethod
    def selection_hash_is_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("expected_selection_sha256 must be a lowercase 64-character SHA-256")
        return value

    @field_validator("document_types")
    @classmethod
    def document_types_are_safe_slugs(cls, value: list[str]) -> list[str]:
        invalid = [item for item in value if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", item)]
        if invalid:
            raise ValueError("document_types must be lowercase filesystem-safe slugs")
        return value

    @field_validator("classification_label_mapping")
    @classmethod
    def classification_mapping_is_safe(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        invalid = [
            item
            for item in (*value.keys(), *value.values())
            if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", item)
        ]
        if invalid:
            raise ValueError(
                "classification_label_mapping keys and values must be lowercase "
                "filesystem-safe slugs"
            )
        return value

    @model_validator(mode="after")
    def selected_types_and_manifests_are_unique(self) -> S3LocalSnapshotConfig:
        if len(self.document_types) != len(set(self.document_types)):
            raise ValueError("document_types must be unique")
        if set(self.classification_label_mapping.values()) != set(self.document_types):
            raise ValueError(
                "classification_label_mapping values must exactly cover document_types"
            )
        keys = [item.key for item in self.classification_manifests]
        local_names = [item.local_name for item in self.classification_manifests]
        if len(keys) != len(set(keys)):
            raise ValueError("classification manifest keys must be unique")
        if len(local_names) != len(set(local_names)):
            raise ValueError("classification manifest local names must be unique")
        quarantine_keys = [item.source_key for item in self.page_count_quarantine]
        if len(quarantine_keys) != len(set(quarantine_keys)):
            raise ValueError("page-count quarantine source keys must be unique")
        outside_prefix = [key for key in quarantine_keys if not key.startswith(self.raw_prefix)]
        if outside_prefix:
            raise ValueError("page-count quarantine keys must be under raw_prefix")
        return self


class CorpusSnapshotSourceConfig(_StrictConfigModel):
    """One already-committed snapshot used to assemble a local corpus."""

    snapshot_config: NonEmptyString

    @field_validator("snapshot_config")
    @classmethod
    def snapshot_config_is_absolute_yaml(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or path.suffix not in {".yaml", ".yml"}:
            raise ValueError("snapshot_config must be an absolute YAML path")
        return value


class LocalCorpusConfig(_StrictConfigModel):
    """Immutable, deduplicated local corpus assembled from verified snapshots."""

    schema_version: Literal[1]
    document_type: NonEmptyString
    destination_root: NonEmptyString
    sources: list[CorpusSnapshotSourceConfig] = Field(min_length=2)
    expected_manifest_sha256: str
    verification_workers: PositiveInteger
    max_pdf_bytes: PositiveInteger

    @field_validator("document_type")
    @classmethod
    def document_type_is_a_safe_slug(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value):
            raise ValueError("document_type must be a lowercase filesystem-safe slug")
        return value

    @field_validator("destination_root")
    @classmethod
    def destination_root_is_safe_and_absolute(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("destination_root must be an absolute local path")
        normalized = Path(os.path.abspath(value))
        if normalized == Path(normalized.anchor):
            raise ValueError("destination_root must not be the filesystem root")
        return value

    @field_validator("expected_manifest_sha256")
    @classmethod
    def manifest_hash_is_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("expected_manifest_sha256 must be a lowercase 64-character SHA-256")
        return value

    @model_validator(mode="after")
    def source_configs_are_unique(self) -> LocalCorpusConfig:
        paths = [item.snapshot_config for item in self.sources]
        if len(paths) != len(set(paths)):
            raise ValueError("corpus snapshot_config paths must be unique")
        return self


class CatalogArtifactConfig(_StrictConfigModel):
    """One content-pinned S3 artifact used to build a classification catalog."""

    key: NonEmptyString
    expected_sha256: str
    local_name: NonEmptyString

    @field_validator("key")
    @classmethod
    def key_is_safe(cls, value: str) -> str:
        if value.startswith("s3://"):
            raise ValueError("catalog artifact key must not be an s3:// URI")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("catalog artifact key must be a safe S3 object key")
        return value

    @field_validator("expected_sha256")
    @classmethod
    def expected_hash_is_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("expected_sha256 must be a lowercase 64-character SHA-256")
        return value

    @field_validator("local_name")
    @classmethod
    def local_name_is_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.name != value or path.suffix not in {".csv", ".jsonl"}:
            raise ValueError("local_name must be a single .csv or .jsonl filename")
        return value


class CatalogLineageConfig(_StrictConfigModel):
    """The complete initial-to-final artifact chain for one classifier run."""

    name: NonEmptyString
    initial_manifest: CatalogArtifactConfig
    final_manifest: CatalogArtifactConfig
    dropped_report: CatalogArtifactConfig
    relabeled_report: CatalogArtifactConfig
    review_report: CatalogArtifactConfig | None

    @field_validator("name")
    @classmethod
    def name_is_a_safe_slug(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value):
            raise ValueError("catalog lineage name must be a lowercase filesystem-safe slug")
        return value

    @model_validator(mode="after")
    def artifact_paths_are_unique_and_typed(self) -> CatalogLineageConfig:
        artifacts = [
            self.initial_manifest,
            self.final_manifest,
            self.dropped_report,
            self.relabeled_report,
        ]
        if self.review_report is not None:
            artifacts.append(self.review_report)
        keys = [item.key for item in artifacts]
        names = [item.local_name for item in artifacts]
        if len(keys) != len(set(keys)):
            raise ValueError("catalog lineage artifact keys must be unique")
        if len(names) != len(set(names)):
            raise ValueError("catalog lineage local artifact names must be unique")
        if self.initial_manifest.local_name.endswith(".jsonl") is False:
            raise ValueError("initial_manifest local_name must end in .jsonl")
        if self.final_manifest.local_name.endswith(".jsonl") is False:
            raise ValueError("final_manifest local_name must end in .jsonl")
        for report in (self.dropped_report, self.relabeled_report, self.review_report):
            if report is not None and report.local_name.endswith(".csv") is False:
                raise ValueError("catalog report local_name must end in .csv")
        return self


class CatalogRawSourceConfig(_StrictConfigModel):
    """One meaningful-PDF S3 prefix represented in the catalog inventory."""

    lineage: NonEmptyString
    role: Literal["primary", "mirror"]
    prefix: NonEmptyString

    @field_validator("lineage")
    @classmethod
    def lineage_is_a_safe_slug(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value):
            raise ValueError("raw source lineage must be a lowercase filesystem-safe slug")
        return value

    @field_validator("prefix")
    @classmethod
    def prefix_is_safe(cls, value: str) -> str:
        if value.startswith("s3://") or not value.endswith("/"):
            raise ValueError("raw source prefix must be a key prefix ending in '/'")
        path = PurePosixPath(value.removesuffix("/"))
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("raw source prefix must be a safe S3 key prefix")
        return value


class ClassificationCatalogConfig(_StrictConfigModel):
    """Immutable document-level union of classifier evidence and source aliases."""

    schema_version: Literal[1]
    bucket: NonEmptyString
    region: NonEmptyString
    destination_root: NonEmptyString
    lineages: list[CatalogLineageConfig] = Field(min_length=1)
    raw_sources: list[CatalogRawSourceConfig] = Field(min_length=1)
    local_snapshots: list[CorpusSnapshotSourceConfig] = Field(min_length=1)
    expected_catalog_sha256: str
    expected_raw_inventory_sha256: str
    network_workers: PositiveInteger
    max_attempts: PositiveInteger
    download_chunk_size_bytes: Annotated[int, Field(ge=64 * 1024, le=64 * 1024 * 1024)]
    max_source_artifact_bytes: PositiveInteger

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, value: str) -> str:
        if not _S3_BUCKET_PATTERN.fullmatch(value):
            raise ValueError("bucket must be a valid lowercase S3 bucket name")
        if ".." in value or ".-" in value or "-." in value:
            raise ValueError("bucket must be a valid S3 bucket name")
        return value

    @field_validator("destination_root")
    @classmethod
    def destination_root_is_safe_and_absolute(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("destination_root must be an absolute local path")
        normalized = Path(os.path.abspath(value))
        if normalized == Path(normalized.anchor):
            raise ValueError("destination_root must not be the filesystem root")
        return value

    @field_validator("expected_catalog_sha256", "expected_raw_inventory_sha256")
    @classmethod
    def expected_hashes_are_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("expected catalog hashes must be lowercase 64-character SHA-256")
        return value

    @model_validator(mode="after")
    def sources_are_disjoint_and_complete(self) -> ClassificationCatalogConfig:
        lineage_names = [item.name for item in self.lineages]
        if len(lineage_names) != len(set(lineage_names)):
            raise ValueError("catalog lineage names must be unique")
        artifact_keys = [
            artifact.key
            for lineage in self.lineages
            for artifact in (
                lineage.initial_manifest,
                lineage.final_manifest,
                lineage.dropped_report,
                lineage.relabeled_report,
                lineage.review_report,
            )
            if artifact is not None
        ]
        if len(artifact_keys) != len(set(artifact_keys)):
            raise ValueError("catalog artifact keys must be globally unique")
        lineage_set = set(lineage_names)
        if any(item.lineage not in lineage_set for item in self.raw_sources):
            raise ValueError("every raw source must reference a configured lineage")
        if {item.lineage for item in self.raw_sources if item.role == "primary"} != lineage_set:
            raise ValueError("every lineage must have at least one primary raw source")
        raw_keys = [(item.lineage, item.role, item.prefix) for item in self.raw_sources]
        if len(raw_keys) != len(set(raw_keys)):
            raise ValueError("catalog raw sources must be unique")
        prefixes = [item.prefix for item in self.raw_sources]
        for index, prefix in enumerate(prefixes):
            for other in prefixes[index + 1 :]:
                if prefix.startswith(other) or other.startswith(prefix):
                    raise ValueError("catalog raw source prefixes must not overlap")
        snapshot_paths = [item.snapshot_config for item in self.local_snapshots]
        if len(snapshot_paths) != len(set(snapshot_paths)):
            raise ValueError("catalog local snapshot configurations must be unique")
        return self


class PilotQualityConfig(_StrictConfigModel):
    """Exact classifier evidence required for the bounded OCR pilot."""

    dummy_is_dummy: Literal[False]
    triage_requires_augmentation: Literal[False]
    readability: Literal["fully_readable"]
    augmentation_need: Literal["use_as_is"]


class PilotStratumConfig(_StrictConfigModel):
    """One exact source-lineage and visual-category quota."""

    lineage: NonEmptyString
    triage_category: Literal["photo", "rendered", "scanned"]
    documents: PositiveInteger

    @field_validator("lineage")
    @classmethod
    def lineage_is_a_safe_slug(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value):
            raise ValueError("pilot stratum lineage must be a lowercase filesystem-safe slug")
        return value


class ExcludedPilotConfig(_StrictConfigModel):
    """One completed pilot whose documents cannot be selected again."""

    root: NonEmptyString
    expected_manifest_sha256: str

    @field_validator("root")
    @classmethod
    def root_is_safe_and_absolute(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("excluded pilot root must be an absolute local path")
        normalized = Path(os.path.abspath(value))
        if normalized == Path(normalized.anchor):
            raise ValueError("excluded pilot root must not be the filesystem root")
        return value

    @field_validator("expected_manifest_sha256")
    @classmethod
    def manifest_hash_is_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(
                "excluded pilot manifest hash must be a lowercase 64-character SHA-256"
            )
        return value


class CatalogPilotConfig(_StrictConfigModel):
    """Immutable, quality-filtered pilot corpus selected from a verified catalog."""

    schema_version: Literal[1, 2]
    catalog_config: NonEmptyString
    expected_catalog_sha256: str
    destination_root: NonEmptyString
    final_label: NonEmptyString
    document_count: PositiveInteger
    max_pages_per_document: PositiveInteger
    selection_namespace: NonEmptyString
    quality: PilotQualityConfig
    deduplicate_by_content_sha256: Literal[True]
    excluded_pilots: list[ExcludedPilotConfig] = Field(default_factory=list)
    strata: list[PilotStratumConfig] = Field(min_length=1)
    expected_manifest_sha256: str
    verification_workers: PositiveInteger
    max_pdf_bytes: PositiveInteger

    @field_validator("catalog_config")
    @classmethod
    def catalog_config_is_absolute_yaml(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or path.suffix not in {".yaml", ".yml"}:
            raise ValueError("catalog_config must be an absolute YAML path")
        return value

    @field_validator("destination_root")
    @classmethod
    def destination_root_is_safe_and_absolute(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("destination_root must be an absolute local path")
        normalized = Path(os.path.abspath(value))
        if normalized == Path(normalized.anchor):
            raise ValueError("destination_root must not be the filesystem root")
        return value

    @field_validator("final_label")
    @classmethod
    def final_label_is_a_safe_slug(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", value):
            raise ValueError("final_label must be a lowercase filesystem-safe slug")
        return value

    @field_validator("expected_catalog_sha256", "expected_manifest_sha256")
    @classmethod
    def expected_hashes_are_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("pilot expected hashes must be lowercase 64-character SHA-256")
        return value

    @model_validator(mode="after")
    def strata_are_canonical_and_complete(self) -> CatalogPilotConfig:
        if self.schema_version == 1 and self.excluded_pilots:
            raise ValueError("pilot schema version 1 does not support excluded_pilots")
        if self.schema_version == 2 and not self.excluded_pilots:
            raise ValueError("pilot schema version 2 requires at least one excluded pilot")
        excluded_roots = [Path(os.path.abspath(item.root)) for item in self.excluded_pilots]
        if len(excluded_roots) != len(set(excluded_roots)):
            raise ValueError("excluded pilot roots must be unique")
        destination = Path(os.path.abspath(self.destination_root))
        for excluded_root in excluded_roots:
            if (
                destination == excluded_root
                or destination in excluded_root.parents
                or excluded_root in destination.parents
            ):
                raise ValueError("pilot destination must not overlap an excluded pilot root")
        keys = [(item.lineage, item.triage_category) for item in self.strata]
        if len(keys) != len(set(keys)):
            raise ValueError("pilot strata must be unique")
        if keys != sorted(keys):
            raise ValueError("pilot strata must be sorted by lineage and triage_category")
        if sum(item.documents for item in self.strata) != self.document_count:
            raise ValueError("pilot stratum quotas must sum to document_count")
        return self


class CatalogSelectionExclusionConfig(_StrictConfigModel):
    """One immutable catalog-pilot manifest excluded from a later surface."""

    root: NonEmptyString
    expected_manifest_sha256: str

    @field_validator("root")
    @classmethod
    def root_is_safe_and_absolute(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("catalog selection exclusion root must be absolute")
        normalized = Path(os.path.abspath(value))
        if normalized == Path(normalized.anchor):
            raise ValueError("catalog selection exclusion root must not be the filesystem root")
        return value

    @field_validator("expected_manifest_sha256")
    @classmethod
    def manifest_hash_is_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("excluded manifest hash must be a lowercase SHA-256")
        return value


class CatalogExtractionSelectionConfig(_StrictConfigModel):
    """All locally available, non-dummy final labels not already selected."""

    schema_version: Literal[2]
    catalog_config: NonEmptyString
    expected_catalog_sha256: str
    destination_root: NonEmptyString
    final_labels: list[NonEmptyString] = Field(min_length=1)
    dummy_is_dummy: Literal[False]
    excluded_pilots: list[CatalogSelectionExclusionConfig]
    deduplicate_by_content_sha256: Literal[True]
    max_pages_per_document: PositiveInteger
    expected_documents: PositiveInteger
    expected_pages: PositiveInteger
    expected_manifest_sha256: str
    verification_workers: PositiveInteger
    max_pdf_bytes: PositiveInteger

    @field_validator("catalog_config")
    @classmethod
    def catalog_config_is_absolute_yaml(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or path.suffix not in {".yaml", ".yml"}:
            raise ValueError("catalog_config must be an absolute YAML path")
        return value

    @field_validator("destination_root")
    @classmethod
    def destination_root_is_safe_and_absolute(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("destination_root must be absolute")
        normalized = Path(os.path.abspath(value))
        if normalized == Path(normalized.anchor):
            raise ValueError("destination_root must not be the filesystem root")
        return value

    @field_validator("expected_catalog_sha256", "expected_manifest_sha256")
    @classmethod
    def expected_hashes_are_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("catalog selection hashes must be lowercase SHA-256 values")
        return value

    @field_validator("final_labels")
    @classmethod
    def labels_are_canonical(cls, value: list[str]) -> list[str]:
        if any(not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", item) for item in value):
            raise ValueError("final_labels must be lowercase filesystem-safe slugs")
        if value != sorted(set(value)):
            raise ValueError("final_labels must be unique and sorted")
        return value

    @model_validator(mode="after")
    def paths_and_exclusions_are_disjoint(self) -> CatalogExtractionSelectionConfig:
        destination = Path(os.path.abspath(self.destination_root))
        roots = [Path(os.path.abspath(item.root)) for item in self.excluded_pilots]
        if len(roots) != len(set(roots)):
            raise ValueError("excluded_pilots roots must be unique")
        for root in roots:
            if destination == root or destination in root.parents or root in destination.parents:
                raise ValueError("selection destination must not overlap an excluded pilot")
        return self


class OutputConfig(_StrictConfigModel):
    """Destination and batching policy for the completed raw OCR dataset."""

    root: NonEmptyString
    retain_page_images: bool
    parquet_compression: Literal["zstd", "snappy", "none"]
    write_batch_rows: PositiveInteger

    @field_validator("root")
    @classmethod
    def output_root_is_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("output root must be an absolute local path")
        normalized = Path(os.path.abspath(value))
        if normalized == Path(normalized.anchor):
            raise ValueError("output root must not be the filesystem root")
        return value


class RunConfig(_StrictConfigModel):
    """Identity and failure semantics for one immutable extraction run."""

    run_id: NonEmptyString
    fail_fast: bool
    resume: bool
    expected_pages: PositiveInteger | None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not _RUN_ID_PATTERN.fullmatch(value):
            raise ValueError("run_id may contain only letters, digits, '.', '_', and '-'")
        return value


class RasterConfig(_StrictConfigModel):
    """Bounded pypdfium2 page rasterization settings."""

    dpi: PositiveInteger
    max_side_pixels: PositiveInteger
    max_pixels: PositiveInteger
    max_pages_per_document: PositiveInteger
    max_pdf_bytes: PositiveInteger
    max_cached_documents_per_process: PositiveInteger
    image_format: Literal["png", "jpeg"]
    jpeg_quality: Annotated[int, Field(ge=1, le=100)] | None = None
    draw_annotations: bool
    reject_xfa: bool

    @model_validator(mode="after")
    def validate_format_specific_quality(self) -> RasterConfig:
        if self.image_format == "jpeg" and self.jpeg_quality is None:
            raise ValueError("jpeg_quality is required when image_format is 'jpeg'")
        if self.image_format == "png" and self.jpeg_quality is not None:
            raise ValueError("jpeg_quality is only valid when image_format is 'jpeg'")
        return self


class SamplingConfig(_StrictConfigModel):
    """Deterministic generation parameters for raw-corpus extraction."""

    temperature: float
    top_p: Annotated[float, Field(gt=0.0, le=1.0)]
    top_k: PositiveInteger
    repetition_penalty: PositiveFloat
    max_tokens: PositiveInteger
    seed: NonNegativeInteger

    @field_validator("temperature")
    @classmethod
    def temperature_must_be_greedy(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("temperature must be 0 for deterministic raw OCR")
        return value


class RetryConfig(_StrictConfigModel):
    """Explicit bounded exponential retry policy for inference requests."""

    max_attempts: PositiveInteger
    initial_backoff_seconds: PositiveFloat
    max_backoff_seconds: PositiveFloat
    backoff_multiplier: Annotated[float, Field(ge=1.0)]
    jitter_fraction: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def maximum_backoff_covers_initial_backoff(self) -> RetryConfig:
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be at least initial_backoff_seconds")
        return self


class SpeculativeDecodingConfig(_StrictConfigModel):
    """GLM-OCR's vLLM multi-token-prediction decoding settings."""

    method: Literal["mtp"]
    num_speculative_tokens: Literal[1, 3]


class RepetitionDetectionConfig(_StrictConfigModel):
    """Deterministic vLLM token-pattern termination settings."""

    min_pattern_size: PositiveInteger
    max_pattern_size: PositiveInteger
    min_count: Annotated[int, Field(ge=2)]

    @model_validator(mode="after")
    def pattern_range_is_ordered(self) -> RepetitionDetectionConfig:
        if self.min_pattern_size > self.max_pattern_size:
            raise ValueError("min_pattern_size must not exceed max_pattern_size")
        return self


class _VllmConfigBase(_StrictConfigModel):
    """Shared vLLM server identity and request policy for one fixed task prompt."""

    endpoint: NonEmptyString
    model: NonEmptyString
    served_model_name: NonEmptyString
    revision: NonEmptyString
    engine_version: NonEmptyString
    container_base_image: NonEmptyString
    container_build_manifest_sha256: str
    dtype: Literal["bfloat16"]
    quantization: Literal["none", "fp8"]
    max_model_len: PositiveInteger
    max_num_batched_tokens: PositiveInteger
    max_num_seqs: PositiveInteger
    gpu_memory_utilization: Annotated[float, Field(gt=0.0, le=1.0)]
    request_timeout_seconds: PositiveFloat
    api_key_env: NonEmptyString
    sampling: SamplingConfig
    retry: RetryConfig
    speculative_decoding: SpeculativeDecodingConfig
    repetition_detection: RepetitionDetectionConfig

    @field_validator("endpoint")
    @classmethod
    def endpoint_must_be_http(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute HTTP(S) URL")
        try:
            hostname = parsed.hostname
            _ = parsed.port
        except ValueError as error:
            raise ValueError("endpoint has an invalid host or port") from error
        if not hostname:
            raise ValueError("endpoint must contain a valid hostname")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("endpoint must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("endpoint must not contain a query string or fragment")
        if parsed.path not in {"", "/"}:
            raise ValueError(
                "endpoint must be the base server URL; the client appends the API path"
            )
        return value

    @field_validator("revision")
    @classmethod
    def revision_must_be_an_immutable_commit(cls, value: str) -> str:
        if not _SHA_40_PATTERN.fullmatch(value):
            raise ValueError(
                "revision must be an immutable 40-character Git commit SHA; "
                "branches such as 'main' are not reproducible"
            )
        return value.lower()

    @field_validator("container_base_image")
    @classmethod
    def container_base_image_must_be_digest_pinned(cls, value: str) -> str:
        if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", value):
            raise ValueError(
                "container_base_image must include an immutable lowercase @sha256 digest"
            )
        return value

    @field_validator("container_build_manifest_sha256")
    @classmethod
    def container_build_manifest_sha256_must_be_valid(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(
                "container_build_manifest_sha256 must be a lowercase 64-character SHA-256"
            )
        return value

    @model_validator(mode="after")
    def model_context_covers_maximum_output(self) -> _VllmConfigBase:
        if self.max_model_len <= self.sampling.max_tokens:
            raise ValueError(
                "max_model_len must exceed sampling.max_tokens to leave room for "
                "image/prompt tokens"
            )
        return self


class VllmConfig(_VllmConfigBase):
    """Text-recognition-only vLLM contract used by the raw OCR pipeline."""

    prompt: Literal["Text Recognition:"]


class TableVllmConfig(_VllmConfigBase):
    """Table-recognition-only vLLM contract used by the auxiliary-view pipeline."""

    prompt: Literal["Table Recognition:"]


VllmClientConfig = VllmConfig | TableVllmConfig


class ConcurrencyConfig(_StrictConfigModel):
    """Independent document, rendering, and inference concurrency limits."""

    max_active_documents: PositiveInteger
    renderer_processes: PositiveInteger
    max_inflight_pages_global: PositiveInteger
    max_inflight_pages_per_document: PositiveInteger

    @model_validator(mode="after")
    def per_document_limit_fits_global_limit(self) -> ConcurrencyConfig:
        if self.max_inflight_pages_per_document > self.max_inflight_pages_global:
            raise ValueError(
                "max_inflight_pages_per_document must not exceed max_inflight_pages_global"
            )
        return self


class BenchmarkConcurrencyPoint(_StrictConfigModel):
    """One valid inference-concurrency point in a benchmark sweep."""

    max_inflight_pages_global: PositiveInteger
    max_inflight_pages_per_document: PositiveInteger

    @model_validator(mode="after")
    def per_document_limit_fits_global_limit(self) -> BenchmarkConcurrencyPoint:
        if self.max_inflight_pages_per_document > self.max_inflight_pages_global:
            raise ValueError(
                "benchmark per-document concurrency must not exceed global concurrency"
            )
        return self


class BenchmarkSweepConfig(_StrictConfigModel):
    """Optional, finite set of concurrency points to benchmark later."""

    points: list[BenchmarkConcurrencyPoint] = Field(min_length=1)
    warmup_pages: NonNegativeInteger
    measured_pages: PositiveInteger
    repetitions: PositiveInteger

    @field_validator("points")
    @classmethod
    def points_must_be_unique(
        cls, value: list[BenchmarkConcurrencyPoint]
    ) -> list[BenchmarkConcurrencyPoint]:
        keys = [
            (
                point.max_inflight_pages_global,
                point.max_inflight_pages_per_document,
            )
            for point in value
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("benchmark concurrency points must be unique")
        return value


class PipelineConfig(_StrictConfigModel):
    """Complete versioned contract for one raw page-OCR extraction run."""

    schema_version: Literal[2]
    source: SourceConfig
    output: OutputConfig
    run: RunConfig
    raster: RasterConfig
    vllm: VllmConfig
    concurrency: ConcurrencyConfig
    benchmark: BenchmarkSweepConfig | None = None

    @model_validator(mode="after")
    def local_source_and_output_must_not_overlap(self) -> PipelineConfig:
        if isinstance(self.source, LocalSourceConfig):
            source_root = Path(self.source.root).resolve(strict=False)
            output_root = Path(self.output.root).resolve(strict=False)
            if (
                source_root == output_root
                or source_root in output_root.parents
                or output_root in source_root.parents
            ):
                raise ValueError("local source and output roots must not overlap")
        return self


def load_strict_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load one UTF-8 YAML mapping while rejecting duplicate keys."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a YAML mapping")
    return raw


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML file and validate it without type coercion or extra fields."""

    raw = load_strict_yaml_mapping(path)
    return PipelineConfig.model_validate(raw, strict=True)


def load_snapshot_config(path: str | Path) -> S3LocalSnapshotConfig:
    """Load a strict YAML contract for a classifier-selected local S3 snapshot."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    if not isinstance(raw, dict):
        raise ValueError("snapshot configuration root must be a YAML mapping")
    return S3LocalSnapshotConfig.model_validate(raw, strict=True)


def load_corpus_config(path: str | Path) -> LocalCorpusConfig:
    """Load a strict YAML contract for an immutable combined local corpus."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    if not isinstance(raw, dict):
        raise ValueError("corpus configuration root must be a YAML mapping")
    return LocalCorpusConfig.model_validate(raw, strict=True)


def load_catalog_config(path: str | Path) -> ClassificationCatalogConfig:
    """Load a strict YAML contract for an immutable classification catalog."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    if not isinstance(raw, dict):
        raise ValueError("classification catalog configuration root must be a YAML mapping")
    return ClassificationCatalogConfig.model_validate(raw, strict=True)


def load_pilot_config(path: str | Path) -> CatalogPilotConfig:
    """Load a strict YAML contract for a classification-catalog pilot corpus."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    if not isinstance(raw, dict):
        raise ValueError("pilot configuration root must be a YAML mapping")
    return CatalogPilotConfig.model_validate(raw, strict=True)


def load_catalog_selection_config(path: str | Path) -> CatalogExtractionSelectionConfig:
    """Load a strict YAML contract for a catalog-derived extraction surface."""

    raw = load_strict_yaml_mapping(path)
    return CatalogExtractionSelectionConfig.model_validate(raw, strict=True)
