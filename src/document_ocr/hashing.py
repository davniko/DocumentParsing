"""Stable hashes used for provenance and idempotency."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase hexadecimal SHA-256 digest."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without materializing it in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically and reject non-standard floats."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _identity_part(value: str | bytes | int | None) -> bytes:
    if value is None:
        return b"n"
    if isinstance(value, bytes):
        return b"b" + value
    if isinstance(value, int):
        if value < 0:
            raise ValueError("identity integers must be non-negative")
        return b"i" + struct.pack(">Q", value)
    return b"s" + value.encode("utf-8")


def identity_sha256(namespace: str, *parts: str | bytes | int | None) -> str:
    """Hash a typed tuple using unambiguous length-prefixed encoding."""

    digest = hashlib.sha256()
    all_parts: Iterable[str | bytes | int | None] = (namespace, *parts)
    for value in all_parts:
        encoded = _identity_part(value)
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def stable_id(prefix: str, namespace: str, *parts: str | bytes | int | None) -> str:
    return f"{prefix}_{identity_sha256(namespace, *parts)}"


def redact_mapping(mapping: Mapping[str, Any], secret_keys: set[str]) -> dict[str, Any]:
    """Recursively redact explicitly named secret-bearing keys."""

    result: dict[str, Any] = {}
    for key, value in mapping.items():
        if key in secret_keys:
            result[key] = "[REDACTED]"
        elif isinstance(value, Mapping):
            result[key] = redact_mapping(value, secret_keys)
        elif isinstance(value, list):
            result[key] = [
                redact_mapping(item, secret_keys) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            result[key] = value
    return result
