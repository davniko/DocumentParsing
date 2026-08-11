from __future__ import annotations

import base64
import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pytest

from document_ocr.config import LocalSourceConfig, S3SourceConfig
from document_ocr.sources import (
    InventoryConflictError,
    SourceBoundsError,
    SourceChangedError,
    SourceDiscoveryError,
    SourceSafetyError,
    discover_sources,
    freeze_source_inventory,
    load_source_inventory,
    materialize_source,
)

PDF_A = b"%PDF-1.7\n" + b"a" * 4096
PDF_B = b"%PDF-1.7\n" + b"b" * 4096
NOW = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)


def local_config(root: Path, include_glob: str = "**/*.pdf") -> LocalSourceConfig:
    return LocalSourceConfig(
        type="local",
        root=str(root),
        include_glob=include_glob,
        dataset_version="contracts-v1",
        require_content_sha256=True,
    )


def s3_config(include_glob: str = "**/*.pdf") -> S3SourceConfig:
    return S3SourceConfig(
        type="s3",
        bucket="document-training-data",
        prefix="incoming/pdfs/",
        include_glob=include_glob,
        region="eu-central-1",
        dataset_version="contracts-v1",
        require_object_version_ids=True,
    )


class FakePaginator:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return self.pages


class FakeS3:
    def __init__(
        self,
        pages: list[dict[str, Any]],
        heads: dict[tuple[str, str], dict[str, Any]],
        bodies: dict[tuple[str, str], bytes] | None = None,
    ) -> None:
        self.paginator = FakePaginator(pages)
        self.heads = heads
        self.bodies = bodies or {}
        self.paginator_names: list[str] = []
        self.head_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.returned_bodies: list[io.BytesIO] = []

    def get_paginator(self, name: str) -> FakePaginator:
        self.paginator_names.append(name)
        return self.paginator

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self.head_calls.append(kwargs)
        return self.heads[(kwargs["Key"], kwargs["VersionId"])]

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        key = (kwargs["Key"], kwargs["VersionId"])
        head = self.heads[key]
        body = io.BytesIO(self.bodies[key])
        self.returned_bodies.append(body)
        return {**head, "Body": body}


def s3_head(payload: bytes, version_id: str, etag: str = '"not-a-hash"') -> dict[str, Any]:
    digest = hashlib.sha256(payload).digest()
    return {
        "VersionId": version_id,
        "ContentLength": len(payload),
        "ETag": etag,
        "LastModified": NOW,
        "ChecksumSHA256": base64.b64encode(digest).decode("ascii"),
        "ChecksumType": "FULL_OBJECT",
    }


def test_local_discovery_is_sorted_content_addressed_and_recursive(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "z").mkdir(parents=True)
    (root / "a").mkdir()
    (root / "z" / "last.pdf").write_bytes(PDF_B)
    (root / "first.pdf").write_bytes(PDF_A)
    (root / "a" / "middle.PDF").write_bytes(PDF_A)
    (root / "ignored.txt").write_text("not a PDF", encoding="utf-8")

    first = discover_sources(local_config(root), max_pdf_bytes=1_000_000)
    second = discover_sources(local_config(root), max_pdf_bytes=1_000_000)

    # The glob is case-sensitive, while the final PDF type guard is case-insensitive.
    assert [item.local_relative_key for item in first] == ["first.pdf", "z/last.pdf"]
    assert first == second
    assert first[0].source_sha256 == hashlib.sha256(PDF_A).hexdigest()
    assert first[0].source_object_version == first[0].source_sha256
    assert first[0].source_uri == (root / "first.pdf").resolve().as_uri()
    assert first[0].local_device is not None
    assert first[0].local_inode is not None
    assert first[0].source_last_modified.tzinfo is not None
    assert first[0].document_id.startswith("doc_")


def test_double_star_glob_matches_root_and_arbitrary_depth(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "a" / "b").mkdir(parents=True)
    (root / "root.pdf").write_bytes(PDF_A)
    (root / "a" / "b" / "deep.pdf").write_bytes(PDF_B)

    objects = discover_sources(local_config(root), max_pdf_bytes=1_000_000)

    assert [item.local_relative_key for item in objects] == ["a/b/deep.pdf", "root.pdf"]


def test_local_discovery_rejects_empty_matches_and_invalid_pdf(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    with pytest.raises(SourceDiscoveryError, match="no PDF files matched"):
        discover_sources(local_config(root), max_pdf_bytes=1_000_000)

    (root / "fake.pdf").write_bytes(b"plain text")
    with pytest.raises(SourceSafetyError, match="does not contain a PDF header"):
        discover_sources(local_config(root), max_pdf_bytes=1_000_000)


def test_local_discovery_enforces_bound_before_hashing(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "large.pdf").write_bytes(PDF_A)

    with pytest.raises(SourceBoundsError, match="exceeds max_pdf_bytes"):
        discover_sources(local_config(root), max_pdf_bytes=len(PDF_A) - 1)


def test_local_discovery_never_traverses_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "source"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "safe.pdf").write_bytes(PDF_A)
    (outside / "hidden.pdf").write_bytes(PDF_B)
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SourceSafetyError, match="symbolic links are forbidden"):
        discover_sources(local_config(root), max_pdf_bytes=1_000_000)


def test_local_root_itself_must_not_be_a_symlink(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "one.pdf").write_bytes(PDF_A)
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)

    with pytest.raises(SourceSafetyError, match="must not traverse symbolic links"):
        discover_sources(local_config(linked), max_pdf_bytes=1_000_000)


def test_local_discovery_detects_a_mutation_during_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import document_ocr.sources as sources_module

    root = tmp_path / "source"
    root.mkdir()
    pdf = root / "one.pdf"
    pdf.write_bytes(PDF_A)
    original = sources_module._stream_pdf

    def mutate_after_read(*args: Any, **kwargs: Any) -> tuple[str, int]:
        result = original(*args, **kwargs)
        pdf.write_bytes(PDF_B)
        return result

    monkeypatch.setattr(sources_module, "_stream_pdf", mutate_after_read)

    with pytest.raises(SourceChangedError, match="changed while hashing"):
        discover_sources(local_config(root), max_pdf_bytes=1_000_000)


def test_local_materialization_copies_and_revalidates_the_frozen_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    scratch = tmp_path / "scratch"
    root.mkdir()
    scratch.mkdir()
    original = root / "one.pdf"
    original.write_bytes(PDF_A)
    source = discover_sources(local_config(root), max_pdf_bytes=1_000_000)[0]

    materialized = materialize_source(
        source,
        scratch,
        max_pdf_bytes=1_000_000,
    )

    assert materialized.path.parent == scratch
    assert materialized.path.name == f"{source.document_id}.pdf"
    assert materialized.path.read_bytes() == PDF_A
    assert materialized.source == source
    assert materialized.path.resolve() != original.resolve()
    # Publishing the same proven source is idempotent and revalidates the bytes.
    assert (
        materialize_source(
            source,
            scratch,
            max_pdf_bytes=1_000_000,
        )
        == materialized
    )


def test_local_materialization_rejects_post_discovery_change(tmp_path: Path) -> None:
    root = tmp_path / "source"
    scratch = tmp_path / "scratch"
    root.mkdir()
    scratch.mkdir()
    pdf = root / "one.pdf"
    pdf.write_bytes(PDF_A)
    source = discover_sources(local_config(root), max_pdf_bytes=1_000_000)[0]
    pdf.write_bytes(PDF_B)

    with pytest.raises(SourceChangedError, match="changed after discovery"):
        materialize_source(source, scratch, max_pdf_bytes=1_000_000)

    assert list(scratch.iterdir()) == []


def test_materialization_requires_existing_canonical_scratch_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "one.pdf").write_bytes(PDF_A)
    source = discover_sources(local_config(root), max_pdf_bytes=1_000_000)[0]

    with pytest.raises(SourceSafetyError, match="cannot be resolved"):
        materialize_source(source, tmp_path / "missing", max_pdf_bytes=1_000_000)


def test_s3_discovery_freezes_only_latest_live_exact_versions() -> None:
    config = s3_config()
    pages = [
        {
            "Versions": [
                {
                    "Key": "incoming/pdfs/z.pdf",
                    "VersionId": "version-z",
                    "IsLatest": True,
                },
                {
                    "Key": "incoming/pdfs/a.pdf",
                    "VersionId": "old-a",
                    "IsLatest": False,
                },
                {
                    "Key": "incoming/pdfs/a.pdf",
                    "VersionId": "version-a",
                    "IsLatest": True,
                },
                {
                    "Key": "incoming/pdfs/notes.txt",
                    "VersionId": "version-notes",
                    "IsLatest": True,
                },
                {
                    "Key": "incoming/pdfs/deleted.pdf",
                    "VersionId": "old-deleted",
                    "IsLatest": False,
                },
            ],
            "DeleteMarkers": [
                {
                    "Key": "incoming/pdfs/deleted.pdf",
                    "VersionId": "delete-1",
                    "IsLatest": True,
                }
            ],
        }
    ]
    heads = {
        ("incoming/pdfs/a.pdf", "version-a"): s3_head(PDF_A, "version-a", '"etag-a"'),
        ("incoming/pdfs/z.pdf", "version-z"): s3_head(PDF_B, "version-z", '"etag-z"'),
    }
    client = FakeS3(pages, heads)

    objects = discover_sources(
        config,
        max_pdf_bytes=1_000_000,
        s3_client=client,
    )

    assert [item.s3_key for item in objects] == [
        "incoming/pdfs/a.pdf",
        "incoming/pdfs/z.pdf",
    ]
    assert [item.s3_version_id for item in objects] == ["version-a", "version-z"]
    assert all(item.source_sha256 is None for item in objects)
    assert objects[0].s3_etag == '"etag-a"'
    assert objects[0].source_object_version == "version-a"
    assert objects[0].source_uri.endswith("a.pdf?versionId=version-a")
    assert client.paginator_names == ["list_object_versions"]
    assert client.paginator.calls == [{"Bucket": config.bucket, "Prefix": config.prefix}]
    assert [call["VersionId"] for call in client.head_calls] == [
        "version-a",
        "version-z",
    ]
    assert all(call["ChecksumMode"] == "ENABLED" for call in client.head_calls)


def test_s3_etag_is_never_used_as_content_sha256() -> None:
    etag_that_looks_like_sha256 = "a" * 64
    pages = [
        {
            "Versions": [
                {
                    "Key": "incoming/pdfs/a.pdf",
                    "VersionId": "version-a",
                    "IsLatest": True,
                }
            ]
        }
    ]
    client = FakeS3(
        pages,
        {
            ("incoming/pdfs/a.pdf", "version-a"): s3_head(
                PDF_A,
                "version-a",
                etag_that_looks_like_sha256,
            )
        },
    )

    source = discover_sources(
        s3_config(),
        max_pdf_bytes=1_000_000,
        s3_client=client,
    )[0]

    assert source.s3_etag == etag_that_looks_like_sha256
    assert source.source_sha256 is None


def test_s3_discovery_rejects_null_or_mismatched_version_ids() -> None:
    null_pages = [
        {
            "Versions": [
                {
                    "Key": "incoming/pdfs/a.pdf",
                    "VersionId": "null",
                    "IsLatest": True,
                }
            ]
        }
    ]
    with pytest.raises(SourceDiscoveryError, match="mutable null VersionId"):
        discover_sources(
            s3_config(),
            max_pdf_bytes=1_000_000,
            s3_client=FakeS3(null_pages, {}),
        )

    pinned_pages = [
        {
            "Versions": [
                {
                    "Key": "incoming/pdfs/a.pdf",
                    "VersionId": "pinned",
                    "IsLatest": True,
                }
            ]
        }
    ]
    client = FakeS3(
        pinned_pages,
        {("incoming/pdfs/a.pdf", "pinned"): s3_head(PDF_A, "different")},
    )
    with pytest.raises(SourceChangedError, match="expected 'pinned'"):
        discover_sources(
            s3_config(),
            max_pdf_bytes=1_000_000,
            s3_client=client,
        )


def test_s3_client_uses_only_standard_boto3_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = [
        {
            "Versions": [
                {
                    "Key": "incoming/pdfs/a.pdf",
                    "VersionId": "version-a",
                    "IsLatest": True,
                }
            ]
        }
    ]
    fake = FakeS3(
        pages,
        {("incoming/pdfs/a.pdf", "version-a"): s3_head(PDF_A, "version-a")},
    )
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def client(*args: Any, **kwargs: Any) -> FakeS3:
        calls.append((args, kwargs))
        return fake

    monkeypatch.setattr(boto3, "client", client)

    discover_sources(s3_config(), max_pdf_bytes=1_000_000)

    assert calls == [(("s3",), {"region_name": "eu-central-1"})]


def test_s3_materialization_hashes_exact_version_and_closes_body(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pages = [
        {
            "Versions": [
                {
                    "Key": "incoming/pdfs/a.pdf",
                    "VersionId": "version-a",
                    "IsLatest": True,
                }
            ]
        }
    ]
    head = s3_head(PDF_A, "version-a", '"multipart-etag-2"')
    client = FakeS3(
        pages,
        {("incoming/pdfs/a.pdf", "version-a"): head},
        {("incoming/pdfs/a.pdf", "version-a"): PDF_A},
    )
    source = discover_sources(
        s3_config(),
        max_pdf_bytes=1_000_000,
        s3_client=client,
    )[0]

    materialized = materialize_source(
        source,
        scratch,
        max_pdf_bytes=1_000_000,
        s3_client=client,
    )

    assert materialized.path.read_bytes() == PDF_A
    assert materialized.source.source_sha256 == hashlib.sha256(PDF_A).hexdigest()
    assert materialized.source.s3_etag == '"multipart-etag-2"'
    assert client.get_calls == [
        {
            "Bucket": "document-training-data",
            "Key": "incoming/pdfs/a.pdf",
            "VersionId": "version-a",
            "ChecksumMode": "ENABLED",
        }
    ]
    assert client.returned_bodies[0].closed

    # A hard crash can leave the fsynced scratch link while the caller still
    # holds only the inventory row (whose content hash is intentionally absent).
    # Re-downloading the exact S3 version proves and safely adopts that link.
    resumed = materialize_source(
        source,
        scratch,
        max_pdf_bytes=1_000_000,
        s3_client=client,
    )
    assert resumed == materialized
    assert len(client.get_calls) == 2
    assert client.returned_bodies[1].closed

    materialized.path.write_bytes(PDF_B)
    with pytest.raises(SourceChangedError, match="scratch destination conflicts"):
        materialize_source(
            source,
            scratch,
            max_pdf_bytes=1_000_000,
            s3_client=client,
        )
    assert materialized.path.read_bytes() == PDF_B
    assert len(client.get_calls) == 3
    assert client.returned_bodies[2].closed


def test_s3_materialization_rejects_body_length_or_checksum_change(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pages = [
        {
            "Versions": [
                {
                    "Key": "incoming/pdfs/a.pdf",
                    "VersionId": "version-a",
                    "IsLatest": True,
                }
            ]
        }
    ]
    head = s3_head(PDF_A, "version-a")
    client = FakeS3(
        pages,
        {("incoming/pdfs/a.pdf", "version-a"): head},
        {("incoming/pdfs/a.pdf", "version-a"): PDF_B[:-1]},
    )
    source = discover_sources(
        s3_config(),
        max_pdf_bytes=1_000_000,
        s3_client=client,
    )[0]

    with pytest.raises(SourceChangedError, match="body length differs"):
        materialize_source(
            source,
            scratch,
            max_pdf_bytes=1_000_000,
            s3_client=client,
        )

    assert list(scratch.iterdir()) == []


def test_inventory_is_canonical_hashed_immutable_and_reloadable(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "b.pdf").write_bytes(PDF_B)
    (root / "a.pdf").write_bytes(PDF_A)
    objects = discover_sources(local_config(root), max_pdf_bytes=1_000_000)
    inventory_path = tmp_path / "run" / "source_inventory.jsonl"

    frozen = freeze_source_inventory(inventory_path, reversed(objects))

    payload = inventory_path.read_bytes()
    assert frozen.sha256 == hashlib.sha256(payload).hexdigest()
    assert payload.endswith(b"\n")
    assert frozen.digest_path.read_text(encoding="ascii") == (
        f"{frozen.sha256}  source_inventory.jsonl\n"
    )
    assert freeze_source_inventory(inventory_path, objects) == frozen
    assert load_source_inventory(inventory_path) == frozen
    assert [item.source_uri for item in frozen.objects] == sorted(
        item.source_uri for item in objects
    )


def test_inventory_load_recovers_valid_inventory_only_crash_state(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "a.pdf").write_bytes(PDF_A)
    objects = discover_sources(local_config(root), max_pdf_bytes=1_000_000)
    inventory_path = tmp_path / "run" / "inventory.jsonl"
    frozen = freeze_source_inventory(inventory_path, objects)
    frozen.digest_path.unlink()

    recovered = load_source_inventory(inventory_path)

    assert recovered == frozen
    assert recovered.digest_path.read_text(encoding="ascii") == (
        f"{recovered.sha256}  inventory.jsonl\n"
    )


def test_inventory_recovery_never_commits_noncanonical_bytes(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "a.pdf").write_bytes(PDF_A)
    objects = discover_sources(local_config(root), max_pdf_bytes=1_000_000)
    inventory_path = tmp_path / "inventory.jsonl"
    frozen = freeze_source_inventory(inventory_path, objects)
    frozen.digest_path.unlink()
    inventory_path.write_bytes(
        inventory_path.read_bytes().replace(b'"document_id":', b' "document_id":')
    )

    with pytest.raises(SourceChangedError, match="not in canonical deterministic form"):
        load_source_inventory(inventory_path)
    assert not frozen.digest_path.exists()


def test_inventory_refuses_overwrite_and_detects_digest_tampering(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "a.pdf").write_bytes(PDF_A)
    (second_root / "b.pdf").write_bytes(PDF_B)
    first = discover_sources(local_config(first_root), max_pdf_bytes=1_000_000)
    second = discover_sources(local_config(second_root), max_pdf_bytes=1_000_000)
    inventory_path = tmp_path / "inventory.jsonl"
    frozen = freeze_source_inventory(inventory_path, first)

    with pytest.raises(InventoryConflictError, match="different bytes"):
        freeze_source_inventory(inventory_path, second)

    frozen.digest_path.write_text("0" * 64 + "  inventory.jsonl\n", encoding="ascii")
    with pytest.raises(SourceChangedError, match="digest does not match"):
        load_source_inventory(inventory_path)


@pytest.mark.parametrize("max_pdf_bytes", [0, -1, True, 1.5])
def test_invalid_size_bounds_are_rejected_explicitly(
    tmp_path: Path,
    max_pdf_bytes: Any,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "a.pdf").write_bytes(PDF_A)
    expected_error = TypeError if isinstance(max_pdf_bytes, (bool, float)) else ValueError

    with pytest.raises(expected_error):
        discover_sources(local_config(root), max_pdf_bytes=max_pdf_bytes)
