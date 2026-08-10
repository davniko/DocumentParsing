"""Deterministic, immutable source discovery and bounded materialization.

Local files are never trusted by pathname alone: discovery and materialization
open every path without following symbolic links, stream the PDF while hashing,
and compare the file identity before and after the read.  S3 objects are frozen
by exact ``VersionId`` and are hashed locally during an exact-version download;
an ETag is retained as provenance but is never interpreted as a content hash.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import os
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote

import boto3

from document_ocr.config import LocalSourceConfig, S3SourceConfig
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, stable_id
from document_ocr.models import SourceObject

_CHUNK_SIZE = 1024 * 1024
_PDF_HEADER_SCAN_BYTES = 1024
_DOCUMENT_ID_PATTERN = "doc_" + "[0-9a-f]" * 64
_S3_CHECKSUM_FIELDS = {
    "s3_checksum_crc32": "ChecksumCRC32",
    "s3_checksum_crc32c": "ChecksumCRC32C",
    "s3_checksum_sha1": "ChecksumSHA1",
    "s3_checksum_sha256": "ChecksumSHA256",
    "s3_checksum_crc64nvme": "ChecksumCRC64NVME",
    "s3_checksum_type": "ChecksumType",
}


class _BinaryWriter(Protocol):
    def write(self, data: bytes, /) -> int: ...


class SourceError(RuntimeError):
    """Base class for explicit source-contract failures."""


class SourceDiscoveryError(SourceError):
    """The configured snapshot could not be enumerated deterministically."""


class SourceSafetyError(SourceError):
    """A path or object violates a source safety invariant."""


class SourceBoundsError(SourceError):
    """A source exceeds a configured size bound or is empty."""


class SourceChangedError(SourceError):
    """A source no longer matches its frozen discovery identity."""


class InventoryConflictError(SourceError):
    """An immutable inventory path already contains different bytes."""


@dataclass(frozen=True, slots=True)
class MaterializedSource:
    """A verified scratch PDF and its now content-addressed provenance."""

    path: Path
    source: SourceObject


@dataclass(frozen=True, slots=True)
class FrozenSourceInventory:
    """Canonical source rows and the digest of their exact JSONL bytes."""

    path: Path
    digest_path: Path
    sha256: str
    objects: tuple[SourceObject, ...]


def _validate_max_pdf_bytes(max_pdf_bytes: int) -> None:
    if isinstance(max_pdf_bytes, bool) or not isinstance(max_pdf_bytes, int):
        raise TypeError("max_pdf_bytes must be an integer")
    if max_pdf_bytes <= 0:
        raise ValueError("max_pdf_bytes must be greater than zero")


def _open_flags(*, directory: bool = False) -> int:
    missing = [name for name in ("O_CLOEXEC", "O_NOFOLLOW") if not hasattr(os, name)]
    if directory and not hasattr(os, "O_DIRECTORY"):
        missing.append("O_DIRECTORY")
    if missing:
        raise SourceSafetyError(
            "safe source traversal requires operating-system flags: " + ", ".join(missing)
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if directory:
        flags |= os.O_DIRECTORY
    return flags


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_pdf_size(size_bytes: int, max_pdf_bytes: int, locator: str) -> None:
    if size_bytes <= 0:
        raise SourceBoundsError(f"PDF is empty: {locator}")
    if size_bytes > max_pdf_bytes:
        raise SourceBoundsError(
            f"PDF exceeds max_pdf_bytes ({size_bytes} > {max_pdf_bytes}): {locator}"
        )


def _stream_pdf(
    read: Callable[[int], bytes],
    *,
    max_pdf_bytes: int,
    locator: str,
    output: _BinaryWriter | None = None,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    header = bytearray()
    size_bytes = 0
    while True:
        chunk = read(_CHUNK_SIZE)
        if not isinstance(chunk, bytes):
            raise SourceSafetyError(f"source stream returned non-bytes data: {locator}")
        if not chunk:
            break
        size_bytes += len(chunk)
        _validate_pdf_size(size_bytes, max_pdf_bytes, locator)
        digest.update(chunk)
        if len(header) < _PDF_HEADER_SCAN_BYTES:
            remaining = _PDF_HEADER_SCAN_BYTES - len(header)
            header.extend(chunk[:remaining])
        if output is not None:
            written = output.write(chunk)
            if written != len(chunk):
                raise OSError(f"short scratch write for source: {locator}")

    _validate_pdf_size(size_bytes, max_pdf_bytes, locator)
    if b"%PDF-" not in header:
        raise SourceSafetyError(
            f"file does not contain a PDF header in its first {_PDF_HEADER_SCAN_BYTES} "
            f"bytes: {locator}"
        )
    return digest.hexdigest(), size_bytes


def _compile_source_glob(pattern: str) -> tuple[str, ...]:
    if pattern.startswith("/") or "\\" in pattern:
        raise SourceDiscoveryError("include_glob must be a relative POSIX glob")
    parts = tuple(pattern.split("/"))
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise SourceDiscoveryError(
            "include_glob must not contain empty, '.' or '..' path components"
        )
    return parts


def _matches_glob(path: PurePosixPath, pattern_parts: tuple[str, ...]) -> bool:
    path_parts = path.parts

    @cache
    def match(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        component = pattern_parts[pattern_index]
        if component == "**":
            return match(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and match(path_index + 1, pattern_index)
            )
        return (
            path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], component)
            and match(path_index + 1, pattern_index + 1)
        )

    return match(0, 0)


def _is_selected_pdf(relative_key: str, pattern_parts: tuple[str, ...]) -> bool:
    path = PurePosixPath(relative_key)
    return path.suffix.lower() == ".pdf" and _matches_glob(path, pattern_parts)


def _document_id(
    source_type: str,
    dataset_version: str,
    source_key: str,
    object_version: str,
) -> str:
    return stable_id(
        "doc",
        "source-document-v1",
        source_type,
        dataset_version,
        source_key,
        object_version,
    )


def _aware_utc(value: Any, field: str, locator: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SourceDiscoveryError(f"{field} must be a timezone-aware datetime: {locator}")
    return value.astimezone(UTC)


def _local_modified_at(mtime_ns: int) -> datetime:
    seconds, nanoseconds = divmod(mtime_ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=nanoseconds // 1_000)


def _hash_open_local_pdf(
    directory_fd: int,
    name: str,
    expected_stat: os.stat_result,
    *,
    max_pdf_bytes: int,
    locator: str,
) -> tuple[str, os.stat_result]:
    file_fd = os.open(name, _open_flags(), dir_fd=directory_fd)
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise SourceSafetyError(f"source is not a regular file: {locator}")
        if _stat_signature(before) != _stat_signature(expected_stat):
            raise SourceChangedError(f"source changed before hashing: {locator}")
        _validate_pdf_size(before.st_size, max_pdf_bytes, locator)
        digest, size_bytes = _stream_pdf(
            lambda amount: os.read(file_fd, amount),
            max_pdf_bytes=max_pdf_bytes,
            locator=locator,
        )
        after = os.fstat(file_fd)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        expected_signature = _stat_signature(before)
        if (
            size_bytes != before.st_size
            or _stat_signature(after) != expected_signature
            or _stat_signature(current) != expected_signature
        ):
            raise SourceChangedError(f"source changed while hashing: {locator}")
        return digest, before
    finally:
        os.close(file_fd)


def _discover_local(
    config: LocalSourceConfig,
    *,
    max_pdf_bytes: int,
) -> tuple[SourceObject, ...]:
    root_input = Path(config.root)
    if not root_input.is_absolute():
        raise SourceSafetyError("local source root must be absolute")
    try:
        root = root_input.resolve(strict=True)
    except OSError as error:
        raise SourceDiscoveryError(f"local source root cannot be resolved: {root_input}") from error
    if root != Path(os.path.abspath(root_input)):
        raise SourceSafetyError(f"local source root must not traverse symbolic links: {root_input}")
    pattern_parts = _compile_source_glob(config.include_glob)

    try:
        root_fd = os.open(root, _open_flags(directory=True))
    except OSError as error:
        raise SourceDiscoveryError(f"cannot open local source root: {root}") from error

    discovered: list[SourceObject] = []

    def walk(directory_fd: int, relative_parts: tuple[str, ...]) -> None:
        try:
            with os.scandir(directory_fd) as entries:
                children = sorted(
                    ((entry.name, entry.stat(follow_symlinks=False)) for entry in entries),
                    key=lambda item: item[0],
                )
        except OSError as error:
            relative_directory = PurePosixPath(*relative_parts).as_posix()
            raise SourceDiscoveryError(
                f"cannot enumerate local source directory: {relative_directory}"
            ) from error

        for name, entry_stat in children:
            relative_path = PurePosixPath(*relative_parts, name)
            locator = (root / Path(*relative_path.parts)).as_posix()
            if stat.S_ISLNK(entry_stat.st_mode):
                raise SourceSafetyError(
                    f"symbolic links are forbidden in the local source tree: {locator}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                try:
                    child_fd = os.open(
                        name,
                        _open_flags(directory=True),
                        dir_fd=directory_fd,
                    )
                except OSError as error:
                    raise SourceChangedError(
                        f"source directory changed during discovery: {locator}"
                    ) from error
                try:
                    if _stat_signature(os.fstat(child_fd)) != _stat_signature(entry_stat):
                        raise SourceChangedError(
                            f"source directory changed during discovery: {locator}"
                        )
                    walk(child_fd, (*relative_parts, name))
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise SourceSafetyError(
                    f"non-regular entries are forbidden in the local source tree: {locator}"
                )
            relative_key = relative_path.as_posix()
            if not _is_selected_pdf(relative_key, pattern_parts):
                continue

            digest, stable_stat = _hash_open_local_pdf(
                directory_fd,
                name,
                entry_stat,
                max_pdf_bytes=max_pdf_bytes,
                locator=locator,
            )
            canonical_path = root / Path(*relative_path.parts)
            discovered.append(
                SourceObject(
                    document_id=_document_id(
                        "local",
                        config.dataset_version,
                        relative_key,
                        digest,
                    ),
                    source_type="local",
                    source_uri=canonical_path.as_uri(),
                    source_dataset_version=config.dataset_version,
                    source_object_version=digest,
                    source_size_bytes=stable_stat.st_size,
                    source_last_modified=_local_modified_at(stable_stat.st_mtime_ns),
                    source_sha256=digest,
                    local_canonical_path=str(canonical_path),
                    local_relative_key=relative_key,
                    local_device=stable_stat.st_dev,
                    local_inode=stable_stat.st_ino,
                    local_mtime_ns=stable_stat.st_mtime_ns,
                )
            )

    try:
        root_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise SourceSafetyError(f"local source root is not a directory: {root}")
        walk(root_fd, ())
    finally:
        os.close(root_fd)

    if not discovered:
        raise SourceDiscoveryError(f"no PDF files matched {config.include_glob!r} under {root}")
    return tuple(sorted(discovered, key=lambda item: item.local_relative_key or ""))


def _required_mapping_value(
    value: Mapping[str, Any],
    key: str,
    expected_type: type[Any],
    locator: str,
) -> Any:
    result = value.get(key)
    if not isinstance(result, expected_type) or (expected_type is int and isinstance(result, bool)):
        raise SourceDiscoveryError(f"S3 response field {key} has an invalid type for {locator}")
    if isinstance(result, str) and not result:
        raise SourceDiscoveryError(f"S3 response field {key} is empty for {locator}")
    return result


def _s3_client(config: S3SourceConfig, supplied: Any | None) -> Any:
    if supplied is not None:
        return supplied
    # No credential arguments are accepted here. boto3 therefore uses only its
    # standard environment, shared-config, web-identity, container, or role chain.
    return boto3.client("s3", region_name=config.region)


def _latest_s3_versions(client: Any, config: S3SourceConfig) -> dict[str, str]:
    try:
        pages = client.get_paginator("list_object_versions").paginate(
            Bucket=config.bucket,
            Prefix=config.prefix,
        )
        latest_versions: dict[str, str] = {}
        latest_deletions: set[str] = set()
        for page in pages:
            if not isinstance(page, Mapping):
                raise SourceDiscoveryError("S3 version-list page is not a mapping")
            raw_versions = page.get("Versions", [])
            raw_deletions = page.get("DeleteMarkers", [])
            if not isinstance(raw_versions, list) or not isinstance(raw_deletions, list):
                raise SourceDiscoveryError("S3 version-list collections must be lists")
            for raw in raw_versions:
                if not isinstance(raw, Mapping):
                    raise SourceDiscoveryError("S3 version entry is not a mapping")
                if raw.get("IsLatest") is not True:
                    continue
                key = _required_mapping_value(raw, "Key", str, config.bucket)
                version_id = _required_mapping_value(raw, "VersionId", str, key)
                previous = latest_versions.get(key)
                if previous is not None and previous != version_id:
                    raise SourceDiscoveryError(
                        f"S3 returned multiple latest versions for key: {key}"
                    )
                latest_versions[key] = version_id
            for raw in raw_deletions:
                if not isinstance(raw, Mapping):
                    raise SourceDiscoveryError("S3 deletion marker is not a mapping")
                if raw.get("IsLatest") is True:
                    key = _required_mapping_value(raw, "Key", str, config.bucket)
                    latest_deletions.add(key)
    except SourceError:
        raise
    except Exception as error:
        raise SourceDiscoveryError(
            f"failed to list exact S3 versions under s3://{config.bucket}/{config.prefix}"
        ) from error

    for key in latest_deletions:
        latest_versions.pop(key, None)
    return latest_versions


def _discover_s3(
    config: S3SourceConfig,
    *,
    max_pdf_bytes: int,
    s3_client: Any | None,
) -> tuple[SourceObject, ...]:
    pattern_parts = _compile_source_glob(config.include_glob)
    client = _s3_client(config, s3_client)
    latest_versions = _latest_s3_versions(client, config)
    selected: list[tuple[str, str, str]] = []
    for key, version_id in latest_versions.items():
        if not key.startswith(config.prefix):
            raise SourceDiscoveryError(f"S3 returned a key outside the requested prefix: {key}")
        relative_key = key[len(config.prefix) :]
        if relative_key and _is_selected_pdf(relative_key, pattern_parts):
            if version_id == "null":
                raise SourceDiscoveryError(
                    f"selected S3 object has mutable null VersionId: s3://{config.bucket}/{key}"
                )
            selected.append((key, relative_key, version_id))

    discovered: list[SourceObject] = []
    for key, _relative_key, version_id in sorted(selected, key=lambda item: item[0]):
        locator = f"s3://{config.bucket}/{key}?versionId={version_id}"
        try:
            response = client.head_object(
                Bucket=config.bucket,
                Key=key,
                VersionId=version_id,
                ChecksumMode="ENABLED",
            )
        except Exception as error:
            raise SourceDiscoveryError(f"failed to HEAD exact S3 version: {locator}") from error
        if not isinstance(response, Mapping):
            raise SourceDiscoveryError(f"S3 HEAD response is not a mapping: {locator}")
        response_version = _required_mapping_value(response, "VersionId", str, locator)
        if response_version != version_id:
            raise SourceChangedError(
                f"S3 HEAD returned VersionId {response_version!r}, expected {version_id!r}: "
                f"{locator}"
            )
        if response.get("DeleteMarker") is True:
            raise SourceChangedError(f"exact S3 version is a delete marker: {locator}")
        size_bytes = _required_mapping_value(response, "ContentLength", int, locator)
        _validate_pdf_size(size_bytes, max_pdf_bytes, locator)
        etag = _required_mapping_value(response, "ETag", str, locator)
        last_modified = _aware_utc(response.get("LastModified"), "LastModified", locator)

        checksum_values: dict[str, Any] = {}
        for model_field, response_field in _S3_CHECKSUM_FIELDS.items():
            checksum = response.get(response_field)
            if checksum is not None and not isinstance(checksum, str):
                raise SourceDiscoveryError(
                    f"S3 response field {response_field} has an invalid type for {locator}"
                )
            checksum_values[model_field] = checksum

        source_uri = (
            f"s3://{config.bucket}/{quote(key, safe='/')}?versionId={quote(version_id, safe='')}"
        )
        discovered.append(
            SourceObject(
                document_id=_document_id(
                    "s3",
                    config.dataset_version,
                    f"{config.bucket}/{key}",
                    version_id,
                ),
                source_type="s3",
                source_uri=source_uri,
                source_dataset_version=config.dataset_version,
                source_object_version=version_id,
                source_size_bytes=size_bytes,
                source_last_modified=last_modified,
                source_sha256=None,
                s3_bucket=config.bucket,
                s3_key=key,
                s3_version_id=version_id,
                s3_etag=etag,
                **checksum_values,
            )
        )

    if not discovered:
        raise SourceDiscoveryError(
            f"no version-pinned PDF objects matched {config.include_glob!r} under "
            f"s3://{config.bucket}/{config.prefix}"
        )
    return tuple(discovered)


def discover_sources(
    config: LocalSourceConfig | S3SourceConfig,
    *,
    max_pdf_bytes: int,
    s3_client: Any | None = None,
) -> tuple[SourceObject, ...]:
    """Discover a sorted source snapshot, hashing every local PDF immediately."""

    _validate_max_pdf_bytes(max_pdf_bytes)
    if isinstance(config, LocalSourceConfig):
        if s3_client is not None:
            raise TypeError("s3_client is only valid for an S3 source")
        return _discover_local(config, max_pdf_bytes=max_pdf_bytes)
    if isinstance(config, S3SourceConfig):
        return _discover_s3(
            config,
            max_pdf_bytes=max_pdf_bytes,
            s3_client=s3_client,
        )
    raise TypeError(f"unsupported source configuration type: {type(config).__name__}")


def _ordered_inventory_objects(
    objects: Iterable[SourceObject],
) -> tuple[SourceObject, ...]:
    ordered = tuple(
        sorted(
            objects,
            key=lambda item: (
                item.source_type,
                item.source_uri,
                item.source_object_version,
                item.document_id,
            ),
        )
    )
    if not ordered:
        raise SourceDiscoveryError("source inventory must contain at least one object")
    document_ids = [item.document_id for item in ordered]
    identities = [
        (item.source_type, item.source_uri, item.source_object_version) for item in ordered
    ]
    if len(document_ids) != len(set(document_ids)):
        raise SourceDiscoveryError("source inventory contains duplicate document_id values")
    if len(identities) != len(set(identities)):
        raise SourceDiscoveryError("source inventory contains duplicate source identities")
    return ordered


def _inventory_payload(objects: tuple[SourceObject, ...]) -> bytes:
    return b"".join(canonical_json_bytes(item.model_dump(mode="json")) + b"\n" for item in objects)


def _canonical_directory(path: Path, *, create: bool) -> Path:
    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    if not parts or parts[0] != "/" or any(part in {"", ".", ".."} for part in parts[1:]):
        raise SourceSafetyError(f"directory path is not canonical: {path}")
    directory_fd = os.open("/", _open_flags(directory=True))
    try:
        for component in parts[1:]:
            try:
                next_fd = os.open(
                    component,
                    _open_flags(directory=True),
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                if not create:
                    raise SourceSafetyError(f"directory cannot be resolved: {path}") from None
                try:
                    os.mkdir(component, dir_fd=directory_fd)
                    next_fd = os.open(
                        component,
                        _open_flags(directory=True),
                        dir_fd=directory_fd,
                    )
                except OSError as error:
                    raise SourceSafetyError(
                        f"directory cannot be created safely: {path}"
                    ) from error
            except OSError as error:
                raise SourceSafetyError(
                    f"directory must not traverse symbolic links: {path}"
                ) from error
            os.close(directory_fd)
            directory_fd = next_fd
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise SourceSafetyError(f"path is not a directory: {path}")
    finally:
        os.close(directory_fd)
    return absolute


def _read_regular_file_no_follow(path: Path) -> bytes:
    file_fd = _open_absolute_file_no_symlinks(Path(os.path.abspath(path)))
    try:
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise SourceSafetyError(f"immutable path must be a regular non-symlink file: {path}")
        with os.fdopen(file_fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(file_fd)


def _publish_immutable(path: Path, payload: bytes) -> None:
    parent = _canonical_directory(path.parent, create=True)
    target = parent / path.name
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            written = stream.write(payload)
            if written != len(payload):
                raise OSError(f"short immutable-file write: {target}")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, target, follow_symlinks=False)
        except FileExistsError:
            if _read_regular_file_no_follow(target) != payload:
                raise InventoryConflictError(
                    f"immutable path already contains different bytes: {target}"
                ) from None
        directory_fd = os.open(parent, _open_flags(directory=True))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def freeze_source_inventory(
    path: str | Path,
    objects: Iterable[SourceObject],
) -> FrozenSourceInventory:
    """Publish canonical JSONL and its SHA-256 sidecar without overwriting either."""

    inventory_path = Path(path)
    if inventory_path.name in {"", ".", ".."}:
        raise SourceSafetyError("inventory path must name a file")
    ordered = _ordered_inventory_objects(objects)
    payload = _inventory_payload(ordered)
    digest = sha256_bytes(payload)
    digest_path = inventory_path.with_name(inventory_path.name + ".sha256")
    digest_payload = f"{digest}  {inventory_path.name}\n".encode("ascii")
    _publish_immutable(inventory_path, payload)
    _publish_immutable(digest_path, digest_payload)
    return FrozenSourceInventory(
        path=inventory_path,
        digest_path=digest_path,
        sha256=digest,
        objects=ordered,
    )


def load_source_inventory(path: str | Path) -> FrozenSourceInventory:
    """Load an inventory only after proving its digest and canonical encoding."""

    inventory_path = Path(path)
    digest_path = inventory_path.with_name(inventory_path.name + ".sha256")
    payload = _read_regular_file_no_follow(inventory_path)
    if not payload or not payload.endswith(b"\n"):
        raise SourceDiscoveryError("source inventory must be non-empty newline-delimited JSON")
    digest = sha256_bytes(payload)
    expected_digest_payload = f"{digest}  {inventory_path.name}\n".encode("ascii")
    if _read_regular_file_no_follow(digest_path) != expected_digest_payload:
        raise SourceChangedError(f"source inventory digest does not match: {inventory_path}")

    objects: list[SourceObject] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            objects.append(SourceObject.model_validate_json(line, strict=True))
        except Exception as error:
            raise SourceDiscoveryError(
                f"invalid source inventory row {line_number}: {inventory_path}"
            ) from error
    ordered = _ordered_inventory_objects(objects)
    if payload != _inventory_payload(ordered):
        raise SourceChangedError(
            f"source inventory is not in canonical deterministic form: {inventory_path}"
        )
    return FrozenSourceInventory(
        path=inventory_path,
        digest_path=digest_path,
        sha256=digest,
        objects=ordered,
    )


def _open_absolute_file_no_symlinks(path: Path) -> int:
    if not path.is_absolute():
        raise SourceSafetyError(f"source path must be absolute: {path}")
    parts = path.parts
    if not parts or parts[0] != "/" or any(part in {"", ".", ".."} for part in parts[1:]):
        raise SourceSafetyError(f"source path is not canonical: {path}")
    directory_fd = os.open("/", _open_flags(directory=True))
    try:
        for component in parts[1:-1]:
            next_fd = os.open(
                component,
                _open_flags(directory=True),
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(parts[-1], _open_flags(), dir_fd=directory_fd)
    except OSError as error:
        raise SourceSafetyError(
            f"source path cannot be opened without symbolic-link traversal: {path}"
        ) from error
    finally:
        os.close(directory_fd)


def _copy_local_source(
    source: SourceObject,
    output: _BinaryWriter,
    *,
    max_pdf_bytes: int,
) -> str:
    if (
        source.local_canonical_path is None
        or source.local_device is None
        or source.local_inode is None
        or source.local_mtime_ns is None
        or source.source_sha256 is None
    ):
        raise SourceSafetyError("local source provenance is incomplete")
    path = Path(source.local_canonical_path)
    file_fd = _open_absolute_file_no_symlinks(path)
    try:
        before = os.fstat(file_fd)
        expected = (
            source.local_device,
            source.local_inode,
            source.source_size_bytes,
            source.local_mtime_ns,
        )
        actual = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if expected != actual:
            raise SourceChangedError(f"local source changed after discovery: {path}")
        digest, size_bytes = _stream_pdf(
            lambda amount: os.read(file_fd, amount),
            max_pdf_bytes=max_pdf_bytes,
            locator=str(path),
            output=output,
        )
        after = os.fstat(file_fd)
        if _stat_signature(before) != _stat_signature(after) or size_bytes != before.st_size:
            raise SourceChangedError(f"local source changed during materialization: {path}")
        if digest != source.source_sha256 or digest != source.source_object_version:
            raise SourceChangedError(f"local source content hash changed: {path}")
        return digest
    finally:
        os.close(file_fd)


def _copy_s3_source(
    source: SourceObject,
    output: _BinaryWriter,
    *,
    max_pdf_bytes: int,
    s3_client: Any,
) -> str:
    if (
        source.s3_bucket is None
        or source.s3_key is None
        or source.s3_version_id is None
        or source.s3_etag is None
    ):
        raise SourceSafetyError("S3 source provenance is incomplete")
    locator = source.source_uri
    try:
        response = s3_client.get_object(
            Bucket=source.s3_bucket,
            Key=source.s3_key,
            VersionId=source.s3_version_id,
            ChecksumMode="ENABLED",
        )
    except Exception as error:
        raise SourceDiscoveryError(f"failed to GET exact S3 version: {locator}") from error
    if not isinstance(response, Mapping):
        raise SourceDiscoveryError(f"S3 GET response is not a mapping: {locator}")
    version_id = _required_mapping_value(response, "VersionId", str, locator)
    if version_id != source.s3_version_id:
        raise SourceChangedError(
            f"S3 GET returned VersionId {version_id!r}, expected "
            f"{source.s3_version_id!r}: {locator}"
        )
    if response.get("DeleteMarker") is True:
        raise SourceChangedError(f"exact S3 version is a delete marker: {locator}")
    content_length = _required_mapping_value(response, "ContentLength", int, locator)
    etag = _required_mapping_value(response, "ETag", str, locator)
    if content_length != source.source_size_bytes or etag != source.s3_etag:
        raise SourceChangedError(f"S3 exact-version metadata changed: {locator}")
    _validate_pdf_size(content_length, max_pdf_bytes, locator)
    for model_field, response_field in _S3_CHECKSUM_FIELDS.items():
        expected_checksum = getattr(source, model_field)
        actual_checksum = response.get(response_field)
        if expected_checksum is not None and actual_checksum != expected_checksum:
            raise SourceChangedError(f"S3 exact-version {response_field} changed: {locator}")

    body = response.get("Body")
    if body is None or not callable(getattr(body, "read", None)):
        raise SourceDiscoveryError(f"S3 GET response has no readable Body: {locator}")
    try:
        digest, size_bytes = _stream_pdf(
            body.read,
            max_pdf_bytes=max_pdf_bytes,
            locator=locator,
            output=output,
        )
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if size_bytes != source.source_size_bytes:
        raise SourceChangedError(f"S3 body length differs from exact-version metadata: {locator}")
    if source.s3_checksum_type == "FULL_OBJECT" and source.s3_checksum_sha256 is not None:
        encoded_digest = base64.b64encode(bytes.fromhex(digest)).decode("ascii")
        if encoded_digest != source.s3_checksum_sha256:
            raise SourceChangedError(f"S3 full-object SHA-256 checksum mismatch: {locator}")
    if source.source_sha256 is not None and digest != source.source_sha256:
        raise SourceChangedError(f"materialized S3 source SHA-256 changed: {locator}")
    return digest


def _validate_existing_materialization(
    destination: Path,
    source: SourceObject,
    *,
    max_pdf_bytes: int,
) -> str:
    if source.source_sha256 is None:
        raise SourceChangedError(
            f"scratch destination exists but source has no proven SHA-256: {destination}"
        )
    file_fd = _open_absolute_file_no_symlinks(destination)
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise SourceSafetyError(f"scratch destination is not regular: {destination}")
        digest, size_bytes = _stream_pdf(
            lambda amount: os.read(file_fd, amount),
            max_pdf_bytes=max_pdf_bytes,
            locator=str(destination),
        )
        after = os.fstat(file_fd)
    finally:
        os.close(file_fd)
    if (
        _stat_signature(before) != _stat_signature(after)
        or size_bytes != source.source_size_bytes
        or digest != source.source_sha256
    ):
        raise SourceChangedError(
            f"scratch destination conflicts with source identity: {destination}"
        )
    return digest


def materialize_source(
    source: SourceObject,
    scratch_dir: str | Path,
    *,
    max_pdf_bytes: int,
    s3_client: Any | None = None,
) -> MaterializedSource:
    """Copy one verified source into scratch without overwriting an existing file."""

    _validate_max_pdf_bytes(max_pdf_bytes)
    _validate_pdf_size(source.source_size_bytes, max_pdf_bytes, source.source_uri)
    if not fnmatch.fnmatchcase(source.document_id, _DOCUMENT_ID_PATTERN):
        raise SourceSafetyError("document_id is not a pipeline-generated safe identifier")
    scratch = _canonical_directory(Path(scratch_dir), create=False)
    destination = scratch / f"{source.document_id}.pdf"
    if destination.exists() or destination.is_symlink():
        digest = _validate_existing_materialization(
            destination,
            source,
            max_pdf_bytes=max_pdf_bytes,
        )
        enriched = SourceObject.model_validate(
            {**source.model_dump(), "source_sha256": digest},
            strict=True,
        )
        return MaterializedSource(path=destination, source=enriched)

    client: Any | None = None
    if source.source_type == "s3":
        if source.s3_bucket is None:
            raise SourceSafetyError("S3 source has no bucket")
        # Region is not persisted in SourceObject, so automatic construction
        # intentionally defers to boto3's standard region/config resolution.
        client = boto3.client("s3") if s3_client is None else s3_client
    elif s3_client is not None:
        raise TypeError("s3_client is only valid for an S3 source")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=scratch,
            prefix=f".{source.document_id}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            if source.source_type == "local":
                digest = _copy_local_source(
                    source,
                    output,
                    max_pdf_bytes=max_pdf_bytes,
                )
            elif source.source_type == "s3":
                digest = _copy_s3_source(
                    source,
                    output,
                    max_pdf_bytes=max_pdf_bytes,
                    s3_client=client,
                )
            else:
                raise TypeError(f"unsupported source type: {source.source_type}")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary_path, destination, follow_symlinks=False)
        except FileExistsError:
            if source.source_sha256 is None:
                raise SourceChangedError(
                    f"concurrent scratch publication cannot be verified: {destination}"
                ) from None
            _validate_existing_materialization(
                destination,
                source,
                max_pdf_bytes=max_pdf_bytes,
            )
        directory_fd = os.open(scratch, _open_flags(directory=True))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    enriched = SourceObject.model_validate(
        {**source.model_dump(), "source_sha256": digest},
        strict=True,
    )
    return MaterializedSource(path=destination, source=enriched)
