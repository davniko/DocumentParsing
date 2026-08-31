"""Pinned real-corpus runner for the transport identity SDV comparison."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.transport_identity import (
    TransportPrivacyPolicy,
    build_transport_identity_bundle,
)
from document_ocr.synthesis.transport_identity_benchmark import (
    TransportBenchmarkSettings,
    run_transport_identity_benchmark,
    transport_identity_candidate_specs,
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    output: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise ValueError(f"duplicate YAML key: {key!r}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _PinnedFile(_StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _DatasetFile(_PinnedFile):
    records: int = Field(gt=0)


class _RunConfig(_StrictModel):
    run_id: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)


class _InputsConfig(_StrictModel):
    preparation_root: str = Field(min_length=1)
    preparation_manifest: _PinnedFile
    template_groups: _DatasetFile
    partition_report: _PinnedFile


class _SelectionConfig(_StrictModel):
    split: str = Field(min_length=1)
    require_template_wholly_in_split: Literal[True]


class _ModelingConfig(_StrictModel):
    candidates: list[Literal["empirical", "gaussian_copula", "ctgan", "tvae"]] = Field(
        min_length=4, max_length=4
    )
    fold_count: int = Field(ge=2)
    fold_seed: int = Field(ge=0, lt=2**32)
    seeds: list[int] = Field(min_length=1)
    neural_epochs: int = Field(gt=0)
    neural_batch_size: int = Field(ge=10)
    proposal_multiplier: int = Field(gt=0)
    proposal_batch_rows: int | None = Field(default=None, gt=0)
    quality_margin: float = Field(ge=0)
    stability_penalty: float = Field(ge=0)
    example_rows_per_candidate: int = Field(gt=0)
    production_pilot_rows: int = Field(default=50, gt=0)
    lexical_ngram_orders: list[int] = Field(default=[2, 3, 4], min_length=1)
    lexical_quality_margin: float = Field(default=0.01, ge=0)
    lexical_stability_penalty: float = Field(default=0.25, ge=0)

    @model_validator(mode="after")
    def complete_comparison(self) -> _ModelingConfig:
        expected = ("empirical", "gaussian_copula", "ctgan", "tvae")
        if tuple(self.candidates) != expected:
            raise ValueError(f"transport benchmark requires candidates {expected}")
        if len(self.seeds) != len(set(self.seeds)) or any(
            not 0 <= seed < 2**32 for seed in self.seeds
        ):
            raise ValueError("modeling seeds must be unique uint32 values")
        if self.neural_batch_size % 10:
            raise ValueError("neural_batch_size must be divisible by CTGAN pac=10")
        if len(self.lexical_ngram_orders) != len(set(self.lexical_ngram_orders)) or any(
            not 2 <= order <= 5 for order in self.lexical_ngram_orders
        ):
            raise ValueError("lexical_ngram_orders must be unique integers in [2, 5]")
        return self


class _PrivacyConfig(_StrictModel):
    minimum_normalized_edit_distance: float = Field(ge=0, lt=1)
    maximum_attempts: int = Field(gt=0)
    minimum_absolute_edit_distance: int = Field(default=2, gt=0)
    maximum_source_substring_fraction: float = Field(default=0.70, gt=0, le=1)
    minimum_source_substring_characters: int = Field(default=5, ge=2)


class TransportIdentityBenchmarkConfig(_StrictModel):
    schema_version: Literal[1]
    task: Literal["bill_of_lading_relation_explicit_v3"]
    run: _RunConfig
    inputs: _InputsConfig
    selection: _SelectionConfig
    modeling: _ModelingConfig
    privacy: _PrivacyConfig


def load_transport_identity_benchmark_config(path: Path) -> TransportIdentityBenchmarkConfig:
    try:
        raw = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("transport benchmark config is not valid UTF-8") from error
    if not isinstance(raw, dict):
        raise ValueError("transport benchmark config root must be a mapping")
    return TransportIdentityBenchmarkConfig.model_validate(raw, strict=True)


def _resolve_file(project_root: Path, value: str, *, label: str) -> Path:
    unresolved = Path(value)
    path = unresolved if unresolved.is_absolute() else project_root / unresolved
    if path.is_symlink() or not path.resolve(strict=True).is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _resolve_directory(project_root: Path, value: str, *, label: str) -> Path:
    unresolved = Path(value)
    path = unresolved if unresolved.is_absolute() else project_root / unresolved
    if path.is_symlink() or not path.resolve(strict=True).is_dir():
        raise ValueError(f"{label} must be a real directory")
    return path.resolve(strict=True)


def _read_json(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    try:
        raw = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{label} root must be an object")
    return raw


def _read_jsonl(
    path: Path, *, expected_sha256: str, expected_records: int, label: str
) -> tuple[dict[str, Any], ...]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} path or SHA-256 differs")
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{label}:{line_number}: blank row")
            try:
                raw = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{label}:{line_number}: invalid JSON") from error
            if not isinstance(raw, dict):
                raise ValueError(f"{label}:{line_number}: row must be an object")
            rows.append(raw)
    if len(rows) != expected_records:
        raise ValueError(f"{label} expected {expected_records} rows, found {len(rows)}")
    return tuple(rows)


def _template_maps(
    rows: Sequence[Mapping[str, Any]], corpus_ids: frozenset[str]
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    by_document: dict[str, str] = {}
    members_by_template: dict[str, tuple[str, ...]] = {}
    for row in rows:
        template_id = row.get("template_id")
        members = row.get("member_document_ids")
        if not isinstance(template_id, str) or not isinstance(members, list):
            raise ValueError("template row has invalid identity fields")
        frozen = tuple(members)
        if (
            not template_id
            or template_id in members_by_template
            or any(not isinstance(value, str) or not value for value in frozen)
            or len(frozen) != len(set(frozen))
        ):
            raise ValueError(f"invalid template group: {template_id!r}")
        members_by_template[template_id] = cast(tuple[str, ...], frozen)
        for document_id in frozen:
            if document_id in by_document:
                raise ValueError(f"document occurs in multiple templates: {document_id}")
            by_document[document_id] = template_id
    if set(by_document) != corpus_ids:
        raise ValueError("template groups do not cover the corpus exactly")
    return by_document, members_by_template


def _partition_map(report: Mapping[str, Any], corpus_ids: frozenset[str]) -> dict[str, str]:
    try:
        outputs = report["inspection"]["partition"]["outputs"]
    except (KeyError, TypeError) as error:
        raise ValueError("partition report has unexpected structure") from error
    if not isinstance(outputs, Mapping):
        raise ValueError("partition outputs must be an object")
    by_document: dict[str, str] = {}
    for split, raw in outputs.items():
        if not isinstance(split, str) or not isinstance(raw, Mapping):
            raise ValueError("partition split is malformed")
        values = raw.get("document_ids")
        if not isinstance(values, list) or raw.get("records") != len(values):
            raise ValueError(f"partition split count differs: {split}")
        for document_id in values:
            if not isinstance(document_id, str) or document_id in by_document:
                raise ValueError(f"invalid partition document: {document_id!r}")
            by_document[document_id] = split
    if set(by_document) != corpus_ids:
        raise ValueError("partition report does not cover the corpus exactly")
    return by_document


def run_transport_identity_benchmark_pipeline(
    *, project_root: Path, config_path: Path, config: TransportIdentityBenchmarkConfig
) -> dict[str, Any]:
    preparation_root = _resolve_directory(
        project_root, config.inputs.preparation_root, label="preparation root"
    )
    manifest_path = _resolve_file(
        project_root,
        config.inputs.preparation_manifest.path,
        label="preparation manifest",
    )
    if manifest_path.parent != preparation_root:
        raise ValueError("preparation manifest is outside its pinned root")
    manifest = _read_json(
        manifest_path,
        expected_sha256=config.inputs.preparation_manifest.sha256,
        label="preparation manifest",
    )
    files = manifest.get("files")
    counts = (manifest.get("domainProjection") or {}).get("tableRows")
    if not isinstance(files, Mapping) or not isinstance(counts, Mapping):
        raise ValueError("preparation manifest lacks file/count receipts")
    document_receipt = files.get("tables/documents.jsonl")
    document_count = counts.get("documents")
    if (
        not isinstance(document_receipt, Mapping)
        or not isinstance(document_receipt.get("sha256"), str)
        or type(document_count) is not int
    ):
        raise ValueError("preparation manifest lacks documents receipt")
    documents_path = (preparation_root / "tables/documents.jsonl").resolve(strict=True)
    documents = _read_jsonl(
        documents_path,
        expected_sha256=cast(str, document_receipt["sha256"]),
        expected_records=document_count,
        label="prepared documents",
    )
    corpus_ids = frozenset(cast(str, row["document_id"]) for row in documents)
    if len(corpus_ids) != len(documents):
        raise ValueError("prepared documents contain duplicate document_id values")

    template_path = _resolve_file(
        project_root, config.inputs.template_groups.path, label="template groups"
    )
    template_rows = _read_jsonl(
        template_path,
        expected_sha256=config.inputs.template_groups.sha256,
        expected_records=config.inputs.template_groups.records,
        label="template groups",
    )
    template_by_document, members_by_template = _template_maps(template_rows, corpus_ids)
    partition_path = _resolve_file(
        project_root, config.inputs.partition_report.path, label="partition report"
    )
    partition_report = _read_json(
        partition_path,
        expected_sha256=config.inputs.partition_report.sha256,
        label="partition report",
    )
    partition_by_document = _partition_map(partition_report, corpus_ids)
    fit_ids = tuple(
        sorted(
            document_id
            for document_id, split in partition_by_document.items()
            if split == config.selection.split
            and all(
                partition_by_document[member] == config.selection.split
                for member in members_by_template[template_by_document[document_id]]
            )
        )
    )
    bundle = build_transport_identity_bundle(
        source_documents=documents,
        fit_document_ids=fit_ids,
        template_by_document=template_by_document,
        partition_by_document=partition_by_document,
        allowed_partition=config.selection.split,
    )
    settings = TransportBenchmarkSettings(
        candidates=transport_identity_candidate_specs(
            neural_epochs=config.modeling.neural_epochs,
            neural_batch_size=config.modeling.neural_batch_size,
        ),
        fold_count=config.modeling.fold_count,
        fold_seed=config.modeling.fold_seed,
        seeds=tuple(config.modeling.seeds),
        proposal_multiplier=config.modeling.proposal_multiplier,
        proposal_batch_rows=config.modeling.proposal_batch_rows,
        quality_margin=config.modeling.quality_margin,
        stability_penalty=config.modeling.stability_penalty,
        privacy_policy=TransportPrivacyPolicy(
            minimum_normalized_edit_distance=config.privacy.minimum_normalized_edit_distance,
            maximum_attempts=config.privacy.maximum_attempts,
            minimum_absolute_edit_distance=config.privacy.minimum_absolute_edit_distance,
            maximum_source_substring_fraction=(config.privacy.maximum_source_substring_fraction),
            minimum_source_substring_characters=(
                config.privacy.minimum_source_substring_characters
            ),
        ),
        example_rows_per_candidate=config.modeling.example_rows_per_candidate,
        production_pilot_rows=config.modeling.production_pilot_rows,
        lexical_ngram_orders=tuple(config.modeling.lexical_ngram_orders),
        lexical_quality_margin=config.modeling.lexical_quality_margin,
        lexical_stability_penalty=config.modeling.lexical_stability_penalty,
    )
    output_parent = Path(config.run.output_dir)
    if not output_parent.is_absolute():
        output_parent = project_root / output_parent
    output_root = output_parent / config.run.run_id
    return run_transport_identity_benchmark(
        bundle=bundle,
        settings=settings,
        artifact_dir=output_root,
        input_provenance={
            "configSha256": sha256_file(config_path),
            "preparationManifestSha256": config.inputs.preparation_manifest.sha256,
            "documentsSha256": document_receipt["sha256"],
            "sourceDocumentCount": len(documents),
            "fitDocumentCount": len(fit_ids),
            "templateGroupsSha256": config.inputs.template_groups.sha256,
            "partitionReportSha256": config.inputs.partition_report.sha256,
            "split": config.selection.split,
            "requireTemplateWhollyInSplit": True,
        },
    )
