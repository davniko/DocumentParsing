from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from document_ocr.atomic import (
    ArtifactReadError,
    AtomicConflictError,
    atomic_publish_bytes,
    atomic_publish_json,
    atomic_write_json,
    read_regular_file_bytes,
)


def test_atomic_json_is_canonical_fsynced_publication_payload(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "record.json"

    atomic_write_json(target, {"z": "café", "a": [2, 1]})

    assert target.read_bytes() == b'{\n  "a": [\n    2,\n    1\n  ],\n  "z": "caf\xc3\xa9"\n}\n'
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_immutable_publication_is_idempotent_and_preserves_conflicting_bytes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.bin"
    original = b"immutable payload\x00"

    assert atomic_publish_bytes(target, original) is True
    assert atomic_publish_bytes(target, original) is False

    with pytest.raises(AtomicConflictError, match="immutable artifact conflicts"):
        atomic_publish_bytes(target, b"different payload")

    assert target.read_bytes() == original
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_immutable_json_publication_compares_canonical_bytes(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"

    assert atomic_publish_json(target, {"b": 2, "a": 1}) is True
    assert atomic_publish_json(target, {"a": 1, "b": 2}) is False

    with pytest.raises(AtomicConflictError):
        atomic_publish_json(target, {"a": 1, "b": 3})

    assert target.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'


def test_concurrent_different_publications_never_overwrite_the_winner(
    tmp_path: Path,
) -> None:
    target = tmp_path / "raced.bin"
    payloads = (b"first immutable value", b"second immutable value")
    start = threading.Barrier(2)

    def publish(payload: bytes) -> bool | AtomicConflictError:
        start.wait(timeout=5)
        try:
            return atomic_publish_bytes(target, payload)
        except AtomicConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, payloads))

    assert sum(outcome is True for outcome in outcomes) == 1
    assert sum(isinstance(outcome, AtomicConflictError) for outcome in outcomes) == 1
    assert target.read_bytes() in payloads
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_concurrent_identical_publications_are_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "same.bin"
    payload = b"same immutable value"
    start = threading.Barrier(2)

    def publish() -> bool:
        start.wait(timeout=5)
        return atomic_publish_bytes(target, payload)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish) for _ in range(2)]
        outcomes = [future.result() for future in futures]

    assert sorted(outcomes) == [False, True]
    assert target.read_bytes() == payload
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_immutable_publication_rejects_directory_target(tmp_path: Path) -> None:
    target = tmp_path / "occupied"
    target.mkdir()

    with pytest.raises(AtomicConflictError, match="not a regular file"):
        atomic_publish_bytes(target, b"payload")

    assert target.is_dir()


def test_immutable_publication_never_follows_existing_symlink(tmp_path: Path) -> None:
    destination = tmp_path / "destination.bin"
    destination.write_bytes(b"protected")
    target = tmp_path / "artifact.bin"
    target.symlink_to(destination)

    with pytest.raises(AtomicConflictError, match="not a readable regular file"):
        atomic_publish_bytes(target, b"replacement")

    assert target.is_symlink()
    assert destination.read_bytes() == b"protected"


def test_regular_file_reader_never_follows_symlink(tmp_path: Path) -> None:
    destination = tmp_path / "destination.json"
    destination.write_bytes(b'{"protected": true}')
    target = tmp_path / "artifact.json"
    target.symlink_to(destination)

    with pytest.raises(ArtifactReadError, match="not a readable regular file"):
        read_regular_file_bytes(target)

    assert destination.read_bytes() == b'{"protected": true}'


def test_regular_file_reader_rejects_directory(tmp_path: Path) -> None:
    target = tmp_path / "directory"
    target.mkdir()

    with pytest.raises(ArtifactReadError, match="not a regular file"):
        read_regular_file_bytes(target)
