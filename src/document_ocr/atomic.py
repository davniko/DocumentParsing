"""Small atomic-publication helpers for local run artifacts."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


class AtomicConflictError(RuntimeError):
    """An immutable artifact path already contains different bytes."""


class ArtifactReadError(RuntimeError):
    """An artifact path cannot be read without following symbolic links."""


def _read_only_no_follow_flags() -> int:
    missing = [name for name in ("O_CLOEXEC", "O_NOFOLLOW") if not hasattr(os, name)]
    if missing:
        raise ArtifactReadError(
            "safe artifact reads require operating-system flags: " + ", ".join(missing)
        )
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def read_regular_file_bytes(path: Path) -> bytes:
    """Read a regular file while refusing a symbolic-link final component."""

    try:
        descriptor = os.open(path, _read_only_no_follow_flags())
    except OSError as error:
        raise ArtifactReadError(f"artifact is not a readable regular file: {path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ArtifactReadError(f"artifact is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    except OSError as error:
        raise ArtifactReadError(f"cannot read artifact: {path}") from error
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write and fsync a file before atomically replacing its target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json_artifact_bytes(value))


def json_artifact_bytes(value: Any) -> bytes:
    """Serialize JSON exactly as the atomic JSON publishers persist it."""

    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    return payload + b"\n"


def atomic_publish_bytes(path: Path, payload: bytes) -> bool:
    """Publish immutable bytes once, accepting only an identical existing file.

    Returns ``True`` when this call created the target and ``False`` when an
    identical artifact was already present.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
            created = True
        except FileExistsError:
            try:
                existing_payload = read_regular_file_bytes(path)
            except ArtifactReadError as error:
                raise AtomicConflictError(f"immutable {error}") from error
            if existing_payload != payload:
                raise AtomicConflictError(
                    f"immutable artifact conflicts with existing path: {path}"
                ) from None
            created = False
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return created
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_publish_json(path: Path, value: Any) -> bool:
    return atomic_publish_bytes(path, json_artifact_bytes(value))
