from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import stat
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.route_registry import (
    UNLOCODE_LICENSE_NAME,
    UNLOCODE_LICENSE_URL,
    CompiledUnlocodeRegistry,
    UnlocodeArchiveMemberPin,
    UnlocodeNormalizationAudit,
    UnlocodeRegistryError,
    UnlocodeRegistryPolicy,
    UnlocodeRegistryReceipt,
    UnlocodeSourcePin,
    compile_unlocode_registry,
    load_pinned_unlocode_locations,
    project_route_locations,
)
from document_ocr.synthesis.routes import maritime_locations
from document_ocr.synthesis.run_safety import StagedRunError

_PARTS = (
    "release/csv/UNLOCODE CodeListPart1.csv",
    "release/csv/UNLOCODE CodeListPart2.csv",
    "release/csv/UNLOCODE CodeListPart3.csv",
)
_SOURCE_URL = (
    "https://opensource.unicc.org/un/unece/uncefact/vocab-locode/-/jobs/"
    "artifacts/2025-1/download?job=package-release"
)


def _row(
    *,
    change: str = "",
    country: str,
    location: str,
    name: str,
    name_without_diacritics: str | None = None,
    subdivision: str = "",
    functions: str = "1-------",
    status: str = "AA",
    coordinates: str = "",
) -> list[str]:
    return [
        change,
        country,
        location,
        name,
        name if name_without_diacritics is None else name_without_diacritics,
        subdivision,
        functions,
        status,
        "2501",
        "",
        coordinates,
        "",
    ]


def _csv(rows: list[list[str]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def _fixture_parts() -> dict[str, bytes]:
    return {
        _PARTS[0]: _csv(
            [
                ["", "EG", "", ".EGYPT", "", "", "", "", "", "", "", ""],
                _row(
                    country="EG",
                    location="ALY",
                    name="Al Iskandarīyah",
                    name_without_diacritics="Al Iskandariyah",
                    subdivision="-",
                    functions="1-34----",
                    coordinates="3112N 02955E",
                ),
                _row(
                    country="EG",
                    location="PSD",
                    name="Port Said",
                    status="QQ",
                ),
                _row(
                    country="EG",
                    location="CAI",
                    name="Cairo",
                    functions="--3-----",
                ),
            ]
        ),
        _PARTS[1]: _csv(
            [
                _row(
                    change="X",
                    country="BR",
                    location="DEL",
                    name="Deleted Port",
                ),
                _row(country="B1", location="BAD", name="Malformed Country"),
                _row(
                    change="¦",
                    country="GT",
                    location="PAL",
                    name="Palin",
                    subdivision=":05",
                    functions="0-3-----",
                    status="AS",
                    coordinates="1424N 09042W",
                ),
                _row(
                    country="ZA",
                    location="CPT",
                    name="Cape Town",
                    status="RL",
                ),
            ]
        ),
        _PARTS[2]: _csv(
            [
                _row(
                    change="#",
                    country="ZA",
                    location="CPT",
                    name="Kaapstad (Cape Town)",
                    status="RL",
                ),
                _row(
                    country="DE",
                    location="HAM",
                    name="Hamburg",
                    subdivision="HH",
                    functions="1-3-----",
                    status="RL",
                    coordinates="5333N 00959E",
                ),
            ]
        ),
    }


def _zip_bytes(parts: dict[str, bytes], *, extra_members: dict[str, bytes] | None = None) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in [*parts.items(), *(extra_members or {}).items()]:
            info = zipfile.ZipInfo(name, date_time=(2025, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, content)
    return payload.getvalue()


def _member_pins(parts: dict[str, bytes]) -> tuple[UnlocodeArchiveMemberPin, ...]:
    return tuple(
        UnlocodeArchiveMemberPin(
            path=name,
            bytes=len(parts[name]),
            sha256=sha256_bytes(parts[name]),
        )
        for name in _PARTS
    )


def _source(
    path: Path,
    payload: bytes,
    *,
    parts: dict[str, bytes] | None = None,
) -> UnlocodeSourcePin:
    path.write_bytes(payload)
    effective_parts = _fixture_parts() if parts is None else parts
    return UnlocodeSourcePin(
        release="2025-1",
        source_url=_SOURCE_URL,
        bytes=len(payload),
        sha256=sha256_bytes(payload),
        csv_members=_member_pins(effective_parts),
    )


def _compile(tmp_path: Path, *, run_name: str = "registry") -> CompiledUnlocodeRegistry:
    source_path = tmp_path / f"{run_name}.zip"
    payload = _zip_bytes(
        _fixture_parts(), extra_members={"../../must-not-be-extracted.txt": b"unsafe"}
    )
    source = _source(source_path, payload)
    return compile_unlocode_registry(
        source_zip=source_path,
        source=source,
        output_parent=tmp_path / f"{run_name}-output",
        run_name=run_name,
    )


def test_compiler_publishes_canonical_jsonl_balanced_receipt_and_no_archive_paths(
    tmp_path: Path,
) -> None:
    result = _compile(tmp_path)

    assert result.created is True
    assert result.receipt.audit.model_dump() == {
        "source_rows": 10,
        "accepted_rows": 4,
        "excluded_deletion_rows": 1,
        "excluded_blank_locode_rows": 1,
        "excluded_malformed_locode_rows": 1,
        "excluded_status_rows": 1,
        "excluded_function_rows": 0,
        "excluded_duplicate_locodes": 1,
        "excluded_duplicate_locode_rows": 2,
    }
    payload = (result.root / "locations.jsonl").read_bytes()
    assert payload.endswith(b"\n")
    lines = payload.splitlines()
    assert [json.loads(line)["locode"] for line in lines] == [
        "DEHAM",
        "EGALY",
        "EGCAI",
        "GTPAL",
    ]
    assert all(line == canonical_json_bytes(json.loads(line)) for line in lines)
    alexandria = json.loads(lines[1])
    assert alexandria == {
        "coordinates": "3112N 02955E",
        "country_code": "EG",
        "function_codes": ["1", "3", "4"],
        "location_code": "ALY",
        "locode": "EGALY",
        "name": "Al Iskandarīyah",
        "name_without_diacritics": "Al Iskandariyah",
        "schema_version": 1,
        "status": "AA",
        "subdivision_code": "-",
    }
    assert not (tmp_path / "must-not-be-extracted.txt").exists()

    receipt_payload = (result.root / "registry-receipt.json").read_bytes()
    assert receipt_payload == canonical_json_bytes(json.loads(receipt_payload)) + b"\n"
    receipt = UnlocodeRegistryReceipt.model_validate_json(receipt_payload, strict=True)
    assert receipt.source.license_name == UNLOCODE_LICENSE_NAME
    assert receipt.source.license_url == UNLOCODE_LICENSE_URL
    assert tuple(row.path for row in receipt.source.csv_members) == _PARTS
    assert tuple(row.relative_path for row in result.commit_receipt.artifacts) == (
        "locations.jsonl",
        "registry-receipt.json",
        "registry.sqlite3",
    )


def test_sqlite_is_query_equivalent_indexed_integral_and_deterministic(
    tmp_path: Path,
) -> None:
    first = _compile(tmp_path, run_name="first")
    second = _compile(tmp_path, run_name="second")
    first_database = first.root / "registry.sqlite3"
    second_database = second.root / "registry.sqlite3"

    assert first_database.read_bytes() == second_database.read_bytes()
    uri = f"file:{first_database}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert connection.execute("PRAGMA application_id").fetchone() == (0x554E4C43,)
        assert connection.execute(
            "SELECT locode, name, subdivision_code, coordinates FROM locations ORDER BY locode"
        ).fetchall() == [
            ("DEHAM", "Hamburg", "HH", "5333N 00959E"),
            ("EGALY", "Al Iskandarīyah", "-", "3112N 02955E"),
            ("EGCAI", "Cairo", None, None),
            ("GTPAL", "Palin", ":05", "1424N 09042W"),
        ]
        assert connection.execute(
            "SELECT function_code FROM location_functions "
            "WHERE locode = 'EGALY' ORDER BY function_code"
        ).fetchall() == [("1",), ("3",), ("4",)]
        indexes = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        assert {
            "locations_country_status_locode_idx",
            "locations_country_name_locode_idx",
            "location_functions_code_locode_idx",
        } <= indexes
    assert not list(first.root.glob("registry.sqlite3-*"))


def test_identical_rerun_is_idempotent_and_tampering_fails_closed(tmp_path: Path) -> None:
    first = _compile(tmp_path)
    source = UnlocodeSourcePin(
        release="2025-1",
        source_url=_SOURCE_URL,
        bytes=first.receipt.source.source_bytes,
        sha256=first.receipt.source.source_sha256,
        csv_members=tuple(
            UnlocodeArchiveMemberPin(
                path=row.path,
                bytes=row.uncompressed_bytes,
                sha256=row.sha256,
            )
            for row in first.receipt.source.csv_members
        ),
    )
    second = compile_unlocode_registry(
        source_zip=tmp_path / "registry.zip",
        source=source,
        output_parent=tmp_path / "registry-output",
        run_name="registry",
    )
    assert second.created is False
    assert second.commit_receipt == first.commit_receipt

    (first.root / "locations.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(StagedRunError, match="artifact inventory differs"):
        compile_unlocode_registry(
            source_zip=tmp_path / "registry.zip",
            source=source,
            output_parent=tmp_path / "registry-output",
            run_name="registry",
        )


def test_source_size_hash_symlink_and_archive_inventory_are_strict(tmp_path: Path) -> None:
    payload = _zip_bytes(_fixture_parts())
    source_path = tmp_path / "source.zip"
    source = _source(source_path, payload)

    with pytest.raises(UnlocodeRegistryError, match="size mismatch"):
        compile_unlocode_registry(
            source_zip=source_path,
            source=source.model_copy(update={"bytes": len(payload) + 1}),
            output_parent=tmp_path / "bad-size",
        )
    with pytest.raises(UnlocodeRegistryError, match="SHA-256 mismatch"):
        compile_unlocode_registry(
            source_zip=source_path,
            source=source.model_copy(update={"sha256": "0" * 64}),
            output_parent=tmp_path / "bad-hash",
        )

    symlink = tmp_path / "source-link.zip"
    symlink.symlink_to(source_path)
    with pytest.raises(UnlocodeRegistryError, match="symbolic link"):
        compile_unlocode_registry(
            source_zip=symlink,
            source=source,
            output_parent=tmp_path / "bad-link",
        )

    missing_payload = _zip_bytes(
        {key: value for key, value in _fixture_parts().items() if key != _PARTS[2]}
    )
    missing_path = tmp_path / "missing.zip"
    missing_source = _source(missing_path, missing_payload)
    with pytest.raises(UnlocodeRegistryError, match=r"missing=.*CodeListPart3"):
        compile_unlocode_registry(
            source_zip=missing_path,
            source=missing_source,
            output_parent=tmp_path / "missing-output",
        )

    repeated_buffer = io.BytesIO()
    with zipfile.ZipFile(repeated_buffer, mode="w") as archive:
        for name, content in _fixture_parts().items():
            archive.writestr(name, content)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr(_PARTS[0], _fixture_parts()[_PARTS[0]])
    repeated_payload = repeated_buffer.getvalue()
    repeated_path = tmp_path / "repeated.zip"
    repeated_source = _source(repeated_path, repeated_payload)
    with pytest.raises(UnlocodeRegistryError, match=r"repeated=.*CodeListPart1"):
        compile_unlocode_registry(
            source_zip=repeated_path,
            source=repeated_source,
            output_parent=tmp_path / "repeated-output",
        )


def test_configurable_status_and_function_policy_is_strict(tmp_path: Path) -> None:
    assert UnlocodeRegistryPolicy().accepted_function_codes == (
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "B",
    )
    source_path = tmp_path / "source.zip"
    payload = _zip_bytes(_fixture_parts())
    source = _source(source_path, payload)
    result = compile_unlocode_registry(
        source_zip=source_path,
        source=source,
        output_parent=tmp_path / "custom",
        policy=UnlocodeRegistryPolicy(
            accepted_statuses=("AA", "AS", "QQ", "RL"),
            accepted_function_codes=("3",),
        ),
    )
    rows = [
        json.loads(line) for line in (result.root / "locations.jsonl").read_bytes().splitlines()
    ]
    assert [row["locode"] for row in rows] == ["DEHAM", "EGALY", "EGCAI", "GTPAL"]

    with pytest.raises(ValidationError, match="unique and sorted"):
        UnlocodeRegistryPolicy(
            accepted_statuses=("RL", "AA"),
            accepted_function_codes=("1",),
        )
    with pytest.raises(ValidationError, match="String should match pattern"):
        UnlocodeSourcePin(
            release="latest",
            source_url=_SOURCE_URL,
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            csv_members=_member_pins(_fixture_parts()),
        )


def test_compiled_loader_and_route_projection_are_strict_and_audited(tmp_path: Path) -> None:
    result = _compile(tmp_path)
    artifact = next(
        row for row in result.receipt.artifacts if row.role == "canonical_locations_jsonl"
    )
    path = result.root / artifact.path
    locations = load_pinned_unlocode_locations(
        path,
        expected_sha256=artifact.sha256,
        expected_records=artifact.records,
    )
    assert [row.locode for row in locations] == ["DEHAM", "EGALY", "EGCAI", "GTPAL"]

    projection = project_route_locations(locations)
    assert projection.audit.model_dump() == {
        "source_records": 4,
        "projected_records": 4,
        "removed_unverified_function_code_records": 1,
        "normalized_placeholder_subdivision_records": 1,
        "omitted_nonconforming_subdivision_records": 1,
    }
    palin = next(row for row in projection.locations if row.locode == "GTPAL")
    assert palin.function_codes == ("3",)
    assert palin.subdivision_code is None
    assert [row.locode for row in maritime_locations(projection.locations)] == ["DEHAM", "EGALY"]

    with pytest.raises(UnlocodeRegistryError, match="SHA-256 mismatch"):
        load_pinned_unlocode_locations(
            path,
            expected_sha256="0" * 64,
            expected_records=artifact.records,
        )
    with pytest.raises(UnlocodeRegistryError, match="count mismatch"):
        load_pinned_unlocode_locations(
            path,
            expected_sha256=artifact.sha256,
            expected_records=artifact.records + 1,
        )

    noncanonical_path = tmp_path / "noncanonical.jsonl"
    row = locations[0].model_dump(mode="json")
    noncanonical_payload = json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
    noncanonical_path.write_bytes(noncanonical_payload)
    with pytest.raises(UnlocodeRegistryError, match="not canonical JSON"):
        load_pinned_unlocode_locations(
            noncanonical_path,
            expected_sha256=sha256_bytes(noncanonical_payload),
            expected_records=1,
        )


def test_member_identity_file_type_and_decompression_bounds_fail_closed(
    tmp_path: Path,
) -> None:
    parts = _fixture_parts()
    payload = _zip_bytes(parts)
    source_path = tmp_path / "member-hash.zip"
    source = _source(source_path, payload, parts=parts)
    wrong_members = list(source.csv_members)
    wrong_members[0] = wrong_members[0].model_copy(update={"sha256": "0" * 64})
    with pytest.raises(UnlocodeRegistryError, match="member SHA-256 mismatch"):
        compile_unlocode_registry(
            source_zip=source_path,
            source=source.model_copy(update={"csv_members": tuple(wrong_members)}),
            output_parent=tmp_path / "member-hash-output",
        )

    symlink_buffer = io.BytesIO()
    with zipfile.ZipFile(symlink_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            info = zipfile.ZipInfo(name, date_time=(2025, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = stat.S_IFLNK | 0o777 if name == _PARTS[0] else stat.S_IFREG | 0o644
            info.external_attr = mode << 16
            archive.writestr(info, content)
    symlink_payload = symlink_buffer.getvalue()
    symlink_path = tmp_path / "member-symlink.zip"
    symlink_source = _source(symlink_path, symlink_payload, parts=parts)
    with pytest.raises(UnlocodeRegistryError, match="not a plain file"):
        compile_unlocode_registry(
            source_zip=symlink_path,
            source=symlink_source,
            output_parent=tmp_path / "member-symlink-output",
        )

    compressed_parts = {**parts, _PARTS[0]: b"A" * 1_000_000}
    compressed_payload = _zip_bytes(compressed_parts)
    compressed_path = tmp_path / "compression-bound.zip"
    compressed_source = _source(
        compressed_path,
        compressed_payload,
        parts=compressed_parts,
    )
    with pytest.raises(UnlocodeRegistryError, match="exceeds decompression bounds"):
        compile_unlocode_registry(
            source_zip=compressed_path,
            source=compressed_source,
            output_parent=tmp_path / "compression-bound-output",
        )


def test_duplicate_audit_rejects_impossible_group_accounting() -> None:
    with pytest.raises(ValidationError, match="duplicate-location accounting"):
        UnlocodeNormalizationAudit(
            source_rows=2,
            accepted_rows=0,
            excluded_deletion_rows=0,
            excluded_blank_locode_rows=0,
            excluded_malformed_locode_rows=0,
            excluded_status_rows=0,
            excluded_function_rows=0,
            excluded_duplicate_locodes=2,
            excluded_duplicate_locode_rows=2,
        )


def test_malformed_function_and_accepted_row_payload_fail_closed(tmp_path: Path) -> None:
    parts = _fixture_parts()
    parts[_PARTS[2]] = _csv(
        [_row(country="DE", location="HAM", name="Hamburg", functions="1--x----", status="RL")]
    )
    payload = _zip_bytes(parts)
    source_path = tmp_path / "malformed-function.zip"
    source = _source(source_path, payload, parts=parts)
    with pytest.raises(UnlocodeRegistryError, match="8-position Function"):
        compile_unlocode_registry(
            source_zip=source_path,
            source=source,
            output_parent=tmp_path / "malformed-function",
        )

    parts = _fixture_parts()
    parts[_PARTS[2]] = _csv(
        [_row(country="DE", location="HAM", name="", functions="1-------", status="RL")]
    )
    payload = _zip_bytes(parts)
    source_path = tmp_path / "blank-name.zip"
    source = _source(source_path, payload, parts=parts)
    with pytest.raises(UnlocodeRegistryError, match="invalid accepted UN/LOCODE row"):
        compile_unlocode_registry(
            source_zip=source_path,
            source=source,
            output_parent=tmp_path / "blank-name",
        )
