"""Strict readers for synthesis fit-partition artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def partition_document_ids(report: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Normalize an immutable split manifest or a training dataset report."""

    if "inspection" in report:
        try:
            partition = report["inspection"]["partition"]
        except (KeyError, TypeError) as error:
            raise ValueError("training report has no inspection partition") from error
    else:
        partition = report
        if partition.get("schema_version") != 1:
            raise ValueError("split manifest has an unsupported schema version")
        if partition.get("publication_status") != "complete":
            raise ValueError("split manifest is not complete")

    if not isinstance(partition, Mapping):
        raise ValueError("partition contract must be an object")
    if partition.get("algorithm") != "seeded_sha256_rank_v1":
        raise ValueError("partition uses an unsupported algorithm")
    outputs = partition.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise ValueError("partition has no outputs")

    normalized: dict[str, tuple[str, ...]] = {}
    assigned: set[str] = set()
    for split, raw in sorted(outputs.items()):
        if not isinstance(split, str) or not split or not isinstance(raw, Mapping):
            raise ValueError("partition split contract is invalid")
        ids = raw.get("document_ids")
        records = raw.get("records")
        if not isinstance(ids, list) or any(
            not isinstance(document_id, str) or not document_id for document_id in ids
        ):
            raise ValueError(f"partition {split!r} document IDs are invalid")
        frozen = tuple(ids)
        if records != len(frozen) or len(frozen) != len(set(frozen)):
            raise ValueError(f"partition {split!r} count or uniqueness differs")
        if assigned & set(frozen):
            raise ValueError("documents occur in multiple partition outputs")
        assigned.update(frozen)
        normalized[split] = frozen
    return normalized


def fit_document_ids(
    report: Mapping[str, Any], *, expected_split: str = "train"
) -> tuple[str, ...]:
    """Return one non-empty fit split after validating every output."""

    outputs = partition_document_ids(report)
    try:
        result = outputs[expected_split]
    except KeyError as error:
        raise ValueError(f"partition does not contain split {expected_split!r}") from error
    if not result:
        raise ValueError(f"partition split {expected_split!r} is empty")
    return result
