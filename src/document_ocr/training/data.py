"""Immutable JSONL inspection and cached sequence-to-sequence tokenization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from document_ocr.training.config import DatasetFileConfig, TrainingConfig, resolve_config_path
from document_ocr.training.prompting import PromptTemplate
from document_ocr.training.splitting import (
    PartitionCandidate,
    select_runtime_partition,
    target_leaf_paths,
)
from document_ocr.training.tasks import TrainingTask, canonical_json


class Tokenizer(Protocol):
    """The narrow tokenizer surface used during preprocessing."""

    name_or_path: str
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None

    def __len__(self) -> int: ...

    def __call__(self, text: Sequence[str] | None = None, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    document_id: str
    input_sha256: str
    input_text: str
    target_text: str
    target_leaf_paths: frozenset[str]


@dataclass(frozen=True, slots=True)
class SourceFileReport:
    split: str
    path: str
    sha256: str
    records: int
    bytes: int


@dataclass(frozen=True, slots=True)
class DatasetInspection:
    task: str
    dataset_mode: str
    prompt_path: str
    prompt_sha256: str
    split_records: dict[str, int]
    total_records: int
    input_characters: dict[str, float | int]
    target_characters: dict[str, float | int]
    source_files: tuple[SourceFileReport, ...]
    partition: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreparedDatasets:
    datasets: Any
    inspection: DatasetInspection
    tokenizer_identity: dict[str, Any]
    decoder_target_contract: dict[str, Any]
    cache_identity: str
    token_lengths: dict[str, dict[str, dict[str, float | int] | int]]
    target_length_filter: dict[str, dict[str, Any]]

    def report(self) -> dict[str, Any]:
        return {
            "inspection": self.inspection.to_dict(),
            "tokenizer": self.tokenizer_identity,
            "decoder_target_contract": self.decoder_target_contract,
            "cache_identity": self.cache_identity,
            "token_lengths": self.token_lengths,
            "target_length_filter": self.target_length_filter,
        }


_SPLITS = ("train", "validation", "test")


def _contains_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_null(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_null(child) for child in value)
    return False


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0, "mean": 0.0}

    ordered = sorted(values)

    def nearest_rank(fraction: float) -> int:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _regular_source_file(project_root: Path, configured: DatasetFileConfig) -> Path:
    unresolved = resolve_config_path(project_root, configured.path)
    if unresolved.is_symlink():
        raise ValueError(f"dataset source must not be a symbolic link: {unresolved}")
    try:
        path = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"dataset source does not exist: {unresolved}") from error
    if not path.is_file():
        raise ValueError(f"dataset source is not a regular file: {path}")
    return path


def _require_string(record: dict[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: field {field!r} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{context}: field {field!r} must not contain a NUL character")
    return value


def _read_source_file(
    *,
    project_root: Path,
    configured: DatasetFileConfig,
    split: str,
    config: TrainingConfig,
    prompt: PromptTemplate,
    task: TrainingTask,
    seen_document_ids: set[str],
    collect_partition_metadata: bool,
) -> tuple[list[NormalizedRecord], SourceFileReport, list[int], list[int]]:
    path = _regular_source_file(project_root, configured)
    digest = hashlib.sha256()
    normalized: list[NormalizedRecord] = []
    input_lengths: list[int] = []
    target_lengths: list[int] = []
    fields = config.dataset.fields

    with path.open("rb") as source:
        for line_number, encoded_line in enumerate(source, start=1):
            digest.update(encoded_line)
            context = f"{path}:{line_number}"
            if not encoded_line.strip():
                raise ValueError(f"{context}: blank JSONL lines are forbidden")
            try:
                decoded_line = encoded_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{context}: line is not valid UTF-8") from error
            try:
                value = json.loads(decoded_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{context}: invalid JSON: {error.msg}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{context}: each JSONL record must be an object")
            record = cast(dict[str, Any], value)

            document_id = _require_string(record, fields.document_id, context)
            if document_id in seen_document_ids:
                raise ValueError(f"{context}: duplicate document ID {document_id!r} across splits")
            seen_document_ids.add(document_id)
            raw_input = _require_string(record, fields.input_text, context)
            actual_input_digest = (
                hashlib.sha256(raw_input.encode("utf-8")).hexdigest()
                if fields.input_sha256 is not None or collect_partition_metadata
                else ""
            )
            if fields.input_sha256 is not None:
                declared_input_digest = _require_string(record, fields.input_sha256, context)
                if declared_input_digest != actual_input_digest:
                    raise ValueError(
                        f"{context}: input text SHA-256 mismatch: expected "
                        f"{declared_input_digest}, found {actual_input_digest}"
                    )

            raw_target = record.get(fields.target)
            if not isinstance(raw_target, dict) or not raw_target:
                raise ValueError(f"{context}: field {fields.target!r} must be a non-empty object")
            target_object = cast(dict[str, Any], raw_target)
            if _contains_null(target_object):
                raise ValueError(f"{context}: sparse targets must not contain null values")
            try:
                canonical_target = task.canonicalize(target_object)
            except ValueError as error:
                raise ValueError(f"{context}: target schema validation failed: {error}") from error

            rendered_input = prompt.render(raw_input)
            target_text = canonical_json(canonical_target)
            normalized.append(
                NormalizedRecord(
                    document_id=document_id,
                    input_sha256=actual_input_digest,
                    input_text=rendered_input,
                    target_text=target_text,
                    target_leaf_paths=(
                        frozenset(target_leaf_paths(canonical_target))
                        if collect_partition_metadata
                        else frozenset()
                    ),
                )
            )
            input_lengths.append(len(rendered_input))
            target_lengths.append(len(target_text))

    actual_digest = digest.hexdigest()
    if actual_digest != configured.sha256:
        raise ValueError(
            f"dataset SHA-256 mismatch for {path}: expected {configured.sha256}, "
            f"found {actual_digest}"
        )
    if len(normalized) != configured.records:
        raise ValueError(
            f"dataset record-count mismatch for {path}: expected {configured.records}, "
            f"found {len(normalized)}"
        )
    return (
        normalized,
        SourceFileReport(
            split=split,
            path=str(path),
            sha256=actual_digest,
            records=len(normalized),
            bytes=path.stat().st_size,
        ),
        input_lengths,
        target_lengths,
    )


def inspect_dataset(
    *,
    project_root: Path,
    config: TrainingConfig,
    prompt: PromptTemplate,
    task: TrainingTask,
) -> tuple[dict[str, list[NormalizedRecord]], DatasetInspection]:
    """Verify all immutable split sources and normalize their training strings."""

    split_records: dict[str, list[NormalizedRecord]] = {
        split: [] for split in _SPLITS
    }
    reports: list[SourceFileReport] = []
    all_input_lengths: list[int] = []
    all_target_lengths: list[int] = []
    seen_document_ids: set[str] = set()
    partition_report: dict[str, Any] | None = None

    if config.dataset.splits is not None:
        for split in _SPLITS:
            records: list[NormalizedRecord] = []
            for source_config in getattr(config.dataset.splits, split):
                source_records, report, input_lengths, target_lengths = _read_source_file(
                    project_root=project_root,
                    configured=source_config,
                    split=split,
                    config=config,
                    prompt=prompt,
                    task=task,
                    seen_document_ids=seen_document_ids,
                    collect_partition_metadata=False,
                )
                records.extend(source_records)
                reports.append(report)
                all_input_lengths.extend(input_lengths)
                all_target_lengths.extend(target_lengths)
            split_records[split] = records
    else:
        source_config = config.dataset.source
        partition_config = config.dataset.partition
        if source_config is None or partition_config is None:
            raise AssertionError("validated runtime partition config is incomplete")
        records, report, input_lengths, target_lengths = _read_source_file(
            project_root=project_root,
            configured=source_config,
            split="source",
            config=config,
            prompt=prompt,
            task=task,
            seen_document_ids=seen_document_ids,
            collect_partition_metadata=True,
        )
        selection = select_runtime_partition(
            [
                PartitionCandidate(
                    document_id=record.document_id,
                    input_sha256=record.input_sha256,
                    target_leaf_paths=record.target_leaf_paths,
                )
                for record in records
            ],
            partition_config,
        )
        validation_ids = set(selection.validation_document_ids)
        split_records["train"] = [
            record for record in records if record.document_id not in validation_ids
        ]
        split_records["validation"] = [
            record for record in records if record.document_id in validation_ids
        ]
        reports.append(report)
        all_input_lengths.extend(input_lengths)
        all_target_lengths.extend(target_lengths)
        partition_report = selection.to_dict()

    inspection = DatasetInspection(
        task=task.name,
        dataset_mode=config.dataset.input_mode,
        prompt_path=str(prompt.path),
        prompt_sha256=prompt.sha256,
        split_records={split: len(split_records[split]) for split in _SPLITS},
        total_records=sum(len(split_records[split]) for split in _SPLITS),
        input_characters=_distribution(all_input_lengths),
        target_characters=_distribution(all_target_lengths),
        source_files=tuple(reports),
        partition=partition_report,
    )
    return split_records, inspection


def _tokenizer_identity(tokenizer: Tokenizer, config: TrainingConfig) -> dict[str, Any]:
    return {
        "name_or_path": config.model.tokenizer_name_or_path,
        "revision": config.model.tokenizer_revision,
        "class": type(tokenizer).__qualname__,
        "reported_name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocabulary_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "transformers_version": importlib.metadata.version("transformers"),
    }


def _decoder_target_contract(tokenizer: Tokenizer) -> dict[str, Any]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("sequence-to-sequence target tokenization requires an EOS token")
    return {
        "name": "content_tokens_then_terminal_eos_v1",
        "tokenizer_add_special_tokens": False,
        "append_eos_token_id": eos_token_id,
        "bos_token_id_forbidden_in_labels": tokenizer.bos_token_id,
    }


def _tokenize_batch(
    batch: Mapping[str, Sequence[str]],
    *,
    tokenizer: Tokenizer,
    config: TrainingConfig,
) -> dict[str, Any]:
    preprocessing = config.dataset.preprocessing
    inputs = list(batch["input_text"])
    targets = list(batch["target_text"])
    source_tokens = tokenizer(
        inputs,
        add_special_tokens=preprocessing.source_add_special_tokens,
        padding=False,
        truncation=False,
        return_attention_mask=True,
    )
    source_ids = cast(list[list[int]], source_tokens["input_ids"])
    source_lengths = [len(item) for item in source_ids]
    source_original_lengths = list(source_lengths)
    overflowing_sources = [
        length for length in source_lengths if length > preprocessing.max_source_length
    ]
    if overflowing_sources:
        if preprocessing.source_overflow == "error":
            raise ValueError(
                "source token length exceeds max_source_length; "
                f"maximum observed in batch={max(overflowing_sources)}, "
                f"configured={preprocessing.max_source_length}"
            )
        source_tokens = tokenizer(
            inputs,
            add_special_tokens=preprocessing.source_add_special_tokens,
            max_length=preprocessing.max_source_length,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )
        source_ids = cast(list[list[int]], source_tokens["input_ids"])
        source_lengths = [len(item) for item in source_ids]

    target_tokens = tokenizer(
        text_target=targets,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    target_content_ids = cast(list[list[int]], target_tokens["input_ids"])
    target_contract = _decoder_target_contract(tokenizer)
    eos_token_id = cast(int, target_contract["append_eos_token_id"])
    forbidden_control_ids = {
        token_id
        for token_id in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    label_ids = []
    for document_id, content_ids in zip(
        batch["document_id"], target_content_ids, strict=True
    ):
        unexpected = forbidden_control_ids.intersection(content_ids)
        if unexpected:
            raise ValueError(
                f"target for document {document_id!r} contains reserved control token IDs "
                f"before EOS insertion: {sorted(unexpected)}"
            )
        label_ids.append([*content_ids, eos_token_id])
    target_lengths = [len(item) for item in label_ids]
    return {
        "document_id": list(batch["document_id"]),
        "input_ids": source_ids,
        "attention_mask": source_tokens["attention_mask"],
        "labels": label_ids,
        "input_length": source_lengths,
        "target_length": target_lengths,
        "input_original_length": source_original_lengths,
        "source_truncated": [
            original > final
            for original, final in zip(source_original_lengths, source_lengths, strict=True)
        ],
    }


def _apply_target_length_limit(
    tokenized: Any, *, split: str, max_target_length: int
) -> tuple[Any, dict[str, Any]]:
    """Exclude complete over-limit training records; never alter held-out membership."""
    lengths = list(tokenized["target_length"])
    kept = [index for index, length in enumerate(lengths) if length <= max_target_length]
    rejected = [index for index, length in enumerate(lengths) if length > max_target_length]
    excluded = [
        {"document_id": row["document_id"], "target_length": row["target_length"]}
        for row in tokenized.select_columns(["document_id", "target_length"]).select(rejected)
    ]
    if rejected and split != "train":
        raise ValueError(
            f"{split} target token length exceeds max_target_length: "
            f"{len(rejected)} records, maximum={max(lengths)}, configured={max_target_length}; "
            "held-out samples must not be filtered or truncated"
        )
    print(
        f"[training-data] {split}: target limit {max_target_length} tokens (including EOS); "
        f"skipping {len(rejected):,} of {len(lengths):,} samples; retaining {len(kept):,}. "
        "Targets are not truncated.",
        file=sys.stderr,
        flush=True,
    )
    if not kept:
        raise ValueError(f"target-length filtering removed all {split} samples")
    report = {
        "policy": "exclude_training_only_v1",
        "max_target_length": max_target_length,
        "includes_terminal_eos": True,
        "original_records": len(lengths),
        "retained_records": len(kept),
        "excluded_records": len(rejected),
        "excluded": excluded,
    }
    return (tokenized.select(kept) if rejected else tokenized), report


def prepare_datasets(
    *,
    project_root: Path,
    config: TrainingConfig,
    prompt: PromptTemplate,
    task: TrainingTask,
    tokenizer: Tokenizer,
) -> PreparedDatasets:
    """Tokenize complete targets, then exclude over-limit training rows before model loading."""

    try:
        from datasets import Dataset, DatasetDict
    except ImportError as error:
        raise RuntimeError("dataset preparation requires the 'train' dependency group") from error

    records_by_split, inspection = inspect_dataset(
        project_root=project_root,
        config=config,
        prompt=prompt,
        task=task,
    )
    tokenizer_identity = _tokenizer_identity(tokenizer, config)
    decoder_target_contract = _decoder_target_contract(tokenizer)
    identity_payload = {
        "schema_version": config.schema_version,
        "task": config.task,
        "prompt_sha256": prompt.sha256,
        "fields": config.dataset.fields.model_dump(mode="json"),
        "preprocessing": config.dataset.preprocessing.model_dump(mode="json"),
        "tokenizer": {
            key: value
            for key, value in tokenizer_identity.items()
            if key != "reported_name_or_path"
        },
        "decoder_target_contract": decoder_target_contract,
        "target_length_policy": "exclude_training_only_v1",
    }
    if config.dataset.splits is not None:
        # Preserve the existing pre-split cache identity exactly; adding the runtime
        # mode must not invalidate immutable pre-split caches.
        identity_payload["sources"] = config.dataset.splits.model_dump(mode="json")
    else:
        source_config = config.dataset.source
        partition_config = config.dataset.partition
        if source_config is None or partition_config is None or inspection.partition is None:
            raise AssertionError("validated runtime partition inspection is incomplete")
        identity_payload["source"] = source_config.model_dump(mode="json")
        identity_payload["partition"] = partition_config.model_dump(mode="json")
        identity_payload["resolved_partition"] = inspection.partition
    cache_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_dir = resolve_config_path(
        project_root, config.dataset.preprocessing.cache_dir
    ).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not cache_dir.is_dir() or cache_dir.is_symlink():
        raise ValueError(f"tokenized-dataset cache must be a real directory: {cache_dir}")

    prepared: dict[str, Any] = {}
    token_lengths: dict[str, dict[str, dict[str, float | int] | int]] = {}
    target_length_filter: dict[str, dict[str, Any]] = {}
    for split in _SPLITS:
        records = records_by_split[split]
        if not records:
            continue
        dataset = Dataset.from_list(
            [
                {
                    "document_id": record.document_id,
                    "input_text": record.input_text,
                    "target_text": record.target_text,
                }
                for record in records
            ]
        )
        cache_file = cache_dir / f"{split}-{cache_identity}.arrow"
        tokenized = dataset.map(
            _tokenize_batch,
            batched=True,
            batch_size=config.dataset.preprocessing.batch_size,
            num_proc=config.dataset.preprocessing.num_proc,
            remove_columns=dataset.column_names,
            load_from_cache_file=config.dataset.preprocessing.load_from_cache_file,
            cache_file_name=str(cache_file),
            fn_kwargs={"tokenizer": tokenizer, "config": config},
            desc=f"Tokenizing {split}",
        )
        tokenized, target_length_filter[split] = _apply_target_length_limit(
            tokenized, split=split,
            max_target_length=config.dataset.preprocessing.max_target_length,
        )
        prepared[split] = tokenized
        token_lengths[split] = {
            "input": _distribution(cast(list[int], tokenized["input_length"])),
            "input_original": _distribution(
                cast(list[int], tokenized["input_original_length"])
            ),
            "target": _distribution(cast(list[int], tokenized["target_length"])),
            "source_truncated_records": sum(
                cast(list[bool], tokenized["source_truncated"])
            ),
        }

    evaluation_lengths = token_lengths.get(config.evaluation.split)
    if evaluation_lengths is not None:
        target_distribution = evaluation_lengths["target"]
        if not isinstance(target_distribution, dict):
            raise AssertionError("evaluation target-length distribution is incomplete")
        maximum_reference_length = target_distribution["max"]
        if not isinstance(maximum_reference_length, int):
            raise AssertionError("evaluation maximum target length is not an integer")
        if maximum_reference_length > config.evaluation.generation_max_length:
            raise ValueError(
                "evaluation generation_max_length cannot reproduce the longest reference "
                f"in split {config.evaluation.split!r}: observed={maximum_reference_length}, "
                f"configured={config.evaluation.generation_max_length}"
            )

    return PreparedDatasets(
        datasets=DatasetDict(prepared),
        inspection=inspection,
        tokenizer_identity=tokenizer_identity,
        decoder_target_contract=decoder_target_contract,
        cache_identity=cache_identity,
        token_lengths=token_lengths,
        target_length_filter=target_length_filter,
    )
