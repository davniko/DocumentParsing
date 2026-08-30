"""Immutable source inspection and exact decoder prompt/completion projection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from document_ocr.decoder_training.completions import (
    THINKING_CLOSE_TAG,
    THINKING_PROMPT_SUFFIX,
)
from document_ocr.decoder_training.config import DecoderTrainingConfig
from document_ocr.hashing import sha256_file
from document_ocr.training.prompting import PromptTemplate, load_prompt
from document_ocr.training.splitting import (
    PartitionCandidate,
    select_runtime_partition,
    target_leaf_paths,
)
from document_ocr.training.tasks import (
    RelationExplicitTaskConstraints,
    TrainingTask,
    canonical_json,
    get_training_task,
)


class ChatTokenizer(Protocol):
    eos_token: str | None
    eos_token_id: int | None

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str: ...
    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class DecoderRecord:
    document_id: str
    input_sha256: str
    prompt_message: str
    reference_target: str
    target_leaf_paths: frozenset[str]


@dataclass(frozen=True, slots=True)
class DatasetInspection:
    source_path: str
    source_sha256: str
    source_records: int
    split_records: dict[str, int]
    prompt_sha256: str
    partition: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TokenInspection:
    split_records: dict[str, int]
    prompt_tokens: dict[str, int]
    completion_tokens: dict[str, int]
    sequence_tokens: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_regular_file(project_root: Path, configured: str, label: str) -> Path:
    unresolved = Path(configured)
    path = unresolved if unresolved.is_absolute() else project_root / unresolved
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def load_task_and_prompt(
    project_root: Path, config: DecoderTrainingConfig
) -> tuple[TrainingTask, PromptTemplate]:
    task = get_training_task(config.task)
    if config.task_constraints is not None:
        path = _resolve_regular_file(
            project_root, config.task_constraints.path, "task-constraints file"
        )
        actual = sha256_file(path)
        if actual != config.task_constraints.sha256:
            raise ValueError(
                f"task-constraints SHA-256 mismatch: expected {config.task_constraints.sha256}, "
                f"found {actual}"
            )
        constraints = RelationExplicitTaskConstraints.model_validate_json(
            path.read_bytes(), strict=True
        )
        task = task.bind_constraints(constraints)
    return task, load_prompt(project_root, config.prompt, task)


def inspect_decoder_dataset(
    *, project_root: Path, config: DecoderTrainingConfig, task: TrainingTask, prompt: PromptTemplate
) -> tuple[dict[str, list[DecoderRecord]], DatasetInspection]:
    source = _resolve_regular_file(project_root, config.dataset.source.path, "dataset source")
    digest = hashlib.sha256()
    records: list[DecoderRecord] = []
    seen: set[str] = set()
    fields = config.dataset.fields
    assert fields.input_sha256 is not None
    with source.open("rb") as stream:
        for line_number, encoded in enumerate(stream, start=1):
            digest.update(encoded)
            try:
                row = json.loads(encoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{source}:{line_number}: invalid UTF-8 JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: record must be an object")
            document_id = row.get(fields.document_id)
            input_text = row.get(fields.input_text)
            input_sha256 = row.get(fields.input_sha256)
            target_value = row.get(fields.target)
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError(f"{source}:{line_number}: invalid document ID")
            if document_id in seen:
                raise ValueError(f"duplicate document ID: {document_id}")
            seen.add(document_id)
            if not isinstance(input_text, str) or not input_text.strip():
                raise ValueError(f"{source}:{line_number}: input text is empty")
            actual_input_sha = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
            if input_sha256 != actual_input_sha:
                raise ValueError(f"{source}:{line_number}: input SHA-256 mismatch")
            if not isinstance(target_value, dict):
                raise ValueError(f"{source}:{line_number}: target must be an object")
            canonical = task.canonicalize(cast(dict[str, Any], target_value))
            records.append(
                DecoderRecord(
                    document_id=document_id,
                    input_sha256=actual_input_sha,
                    prompt_message=prompt.render(input_text),
                    reference_target=canonical_json(canonical),
                    target_leaf_paths=frozenset(target_leaf_paths(canonical)),
                )
            )
    actual_digest = digest.hexdigest()
    expected = config.dataset.source
    if actual_digest != expected.sha256:
        raise ValueError(
            f"dataset SHA-256 mismatch: expected {expected.sha256}, found {actual_digest}"
        )
    if len(records) != expected.records:
        raise ValueError(
            f"dataset record count mismatch: expected {expected.records}, found {len(records)}"
        )
    selection = select_runtime_partition(
        [
            PartitionCandidate(row.document_id, row.input_sha256, row.target_leaf_paths)
            for row in records
        ],
        config.dataset.partition,
    )
    validation_ids = set(selection.validation_document_ids)
    split = {
        "train": [row for row in records if row.document_id not in validation_ids],
        "validation": [row for row in records if row.document_id in validation_ids],
    }
    inspection = DatasetInspection(
        source_path=str(source),
        source_sha256=actual_digest,
        source_records=len(records),
        split_records={name: len(rows) for name, rows in split.items()},
        prompt_sha256=prompt.sha256,
        partition=selection.to_dict(),
    )
    return split, inspection


def _token_ids(tokenizer: ChatTokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, padding=False)
    ids = encoded.get("input_ids")
    if not isinstance(ids, list) or any(not isinstance(value, int) for value in ids):
        raise ValueError("tokenizer did not return one flat input_ids list")
    return cast(list[int], ids)


def project_for_trl(
    records: dict[str, list[DecoderRecord]],
    *,
    tokenizer: ChatTokenizer,
    config: DecoderTrainingConfig,
) -> tuple[Any, TokenInspection]:
    """Render exact chat prompts and enforce zero truncation before TRL sees data."""

    try:
        from datasets import Dataset, DatasetDict
    except ImportError as error:
        raise RuntimeError("decoder dataset projection requires the decoder environment") from error
    if tokenizer.eos_token is None or tokenizer.eos_token_id is None:
        raise ValueError("decoder tokenizer must define an EOS token")
    projected: dict[str, Any] = {}
    maxima = {"prompt": 0, "completion": 0, "sequence": 0}
    thinking_enabled = config.sequence.thinking == "enabled"
    for split_name, split_rows in records.items():
        output_rows: list[dict[str, Any]] = []
        for row in split_rows:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row.prompt_message}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking_enabled,
            )
            if not isinstance(rendered, str) or not rendered:
                raise ValueError("chat template produced an empty prompt")
            if thinking_enabled and not rendered.endswith(THINKING_PROMPT_SUFFIX):
                raise ValueError(
                    "thinking-enabled chat template does not expose Qwen's native <think> boundary"
                )
            # GRPO does not train on this reference completion, but retaining a
            # minimally valid native completion makes the zero-truncation audit
            # account for the closing thinking boundary and complete final JSON.
            completion_prefix = f"{THINKING_CLOSE_TAG}\n\n" if thinking_enabled else ""
            completion = completion_prefix + row.reference_target + tokenizer.eos_token
            prompt_length = len(_token_ids(tokenizer, rendered))
            completion_length = len(_token_ids(tokenizer, completion))
            sequence_length = len(_token_ids(tokenizer, rendered + completion))
            limits = config.sequence
            if prompt_length > limits.max_prompt_length:
                raise ValueError(
                    f"document {row.document_id} prompt has {prompt_length} tokens, exceeding "
                    f"{limits.max_prompt_length}"
                )
            if completion_length > limits.max_completion_length:
                raise ValueError(
                    f"document {row.document_id} completion has {completion_length} tokens, "
                    f"exceeding {limits.max_completion_length}"
                )
            if sequence_length > limits.max_sequence_length:
                raise ValueError(
                    f"document {row.document_id} sequence has {sequence_length} tokens, "
                    f"exceeding {limits.max_sequence_length}"
                )
            maxima["prompt"] = max(maxima["prompt"], prompt_length)
            maxima["completion"] = max(maxima["completion"], completion_length)
            maxima["sequence"] = max(maxima["sequence"], sequence_length)
            output_rows.append(
                {
                    "document_id": row.document_id,
                    "input_sha256": row.input_sha256,
                    "prompt": rendered,
                    "completion": completion,
                    "reference_target": row.reference_target,
                    "prompt_length": prompt_length,
                    "completion_length": completion_length,
                    "sequence_length": sequence_length,
                }
            )
        projected[split_name] = Dataset.from_list(output_rows)
    return DatasetDict(projected), TokenInspection(
        split_records={name: len(rows) for name, rows in records.items()},
        prompt_tokens={"max": maxima["prompt"], "limit": config.sequence.max_prompt_length},
        completion_tokens={
            "max": maxima["completion"],
            "limit": config.sequence.max_completion_length,
        },
        sequence_tokens={"max": maxima["sequence"], "limit": config.sequence.max_sequence_length},
    )
