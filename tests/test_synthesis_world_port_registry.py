from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.route_registry import UnlocodeLocation
from document_ocr.synthesis.run_safety import StagedRunError
from document_ocr.synthesis.world_port_registry import (
    CURRENT_WPI_SOURCE,
    CURRENT_WPI_SOURCE_BYTES,
    CURRENT_WPI_SOURCE_RECORDS,
    CURRENT_WPI_SOURCE_SHA256,
    NGA_WPI_SOURCE_URL,
    PINNED_UNLOCODE_ARTIFACT,
    WPI_CSV_COLUMNS,
    CompiledWorldPortRegistry,
    UnlocodeArtifactPin,
    WorldPortIntersectionAudit,
    WorldPortRecord,
    WorldPortRegistryError,
    WorldPortRegistryReceipt,
    WpiSourcePin,
    compile_world_port_registry,
    load_pinned_world_port_records,
)


def _wpi_row(
    *,
    oid: int,
    wpi_number: int,
    locode: str,
    name: str,
    alternate_name: str = " ",
    country_name: str = "Never used for identity",
    latitude: str = "53.550000000000000",
    longitude: str = "9.983333000000000",
) -> list[str]:
    row = [" "] * len(WPI_CSV_COLUMNS)
    values = {
        "OID_": f"{oid}.0",
        "World Port Index Number": f"{wpi_number}.0",
        "Main Port Name": name,
        "Alternate Port Name": alternate_name,
        "UN/LOCODE": locode,
        "Country Code": country_name,
        "Latitude": latitude,
        "Longitude": longitude,
    }
    for column, value in values.items():
        row[WPI_CSV_COLUMNS.index(column)] = value
    return row


def _wpi_csv(rows: list[list[str]], *, header: tuple[str, ...] = WPI_CSV_COLUMNS) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return text.getvalue().encode("utf-8-sig")


def _fixture_rows() -> list[list[str]]:
    return [
        _wpi_row(oid=0, wpi_number=10, locode=" ", name="Blank LOCODE"),
        _wpi_row(oid=1, wpi_number=11, locode=" DE HAM", name="Malformed LOCODE"),
        _wpi_row(oid=2, wpi_number=12, locode="US XXX", name="Absent LOCODE"),
        _wpi_row(
            oid=3,
            wpi_number=100,
            locode="DE HAM",
            name="Hamburg West",
            alternate_name="Hamburg-Waltershof",
            country_name="Alias that must be ignored",
        ),
        _wpi_row(
            oid=4,
            wpi_number=200,
            locode="FRMRS",
            name="Marseille",
            country_name="Not France",
            latitude="43.300000000000000",
            longitude="5.366667000000000",
        ),
        _wpi_row(oid=5, wpi_number=101, locode="DE HAM", name="Hamburg East"),
        _wpi_row(oid=6, wpi_number=300, locode="FR MRS", name="Marseilles old spelling"),
        _wpi_row(oid=7, wpi_number=300, locode="FR MRS", name="Marseille new spelling"),
    ]


def _unlocode_payload() -> bytes:
    rows = (
        UnlocodeLocation(
            locode="DEHAM",
            country_code="DE",
            location_code="HAM",
            name="Hamburg",
            name_without_diacritics="Hamburg",
            subdivision_code="HH",
            function_codes=("1",),
            status="RL",
            coordinates="5333N 00959E",
        ),
        UnlocodeLocation(
            locode="FRMRS",
            country_code="FR",
            location_code="MRS",
            name="Marseille",
            name_without_diacritics="Marseille",
            subdivision_code="13",
            function_codes=("1",),
            status="RL",
            coordinates="4318N 00522E",
        ),
    )
    return b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in rows)


def _write_inputs(
    root: Path,
    rows: list[list[str]] | None = None,
) -> tuple[Path, WpiSourcePin, Path, UnlocodeArtifactPin]:
    source_payload = _wpi_csv(_fixture_rows() if rows is None else rows)
    source_path = root / "UpdatedPub150.csv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_payload)
    source = WpiSourcePin(
        source_url=NGA_WPI_SOURCE_URL,
        bytes=len(source_payload),
        sha256=sha256_bytes(source_payload),
        records=len(_fixture_rows() if rows is None else rows),
    )
    unlocode_payload = _unlocode_payload()
    unlocode_path = root / "locations.jsonl"
    unlocode_path.write_bytes(unlocode_payload)
    unlocode = UnlocodeArtifactPin(
        run_name="fixture-unlocode",
        release="2025-1",
        relative_path="fixture-unlocode/locations.jsonl",
        bytes=len(unlocode_payload),
        sha256=sha256_bytes(unlocode_payload),
        records=2,
    )
    return source_path, source, unlocode_path, unlocode


def _compile(
    tmp_path: Path,
    *,
    rows: list[list[str]] | None = None,
    run_name: str = "world-ports",
) -> CompiledWorldPortRegistry:
    source_path, source, unlocode_path, unlocode = _write_inputs(tmp_path / run_name, rows)
    return compile_world_port_registry(
        source_csv=source_path,
        source=source,
        unlocode_locations_jsonl=unlocode_path,
        unlocode_artifact=unlocode,
        output_parent=tmp_path / "artifacts",
        run_name=run_name,
    )


def test_current_source_and_unlocode_pins_are_exact() -> None:
    assert CURRENT_WPI_SOURCE.model_dump() == {
        "source_url": NGA_WPI_SOURCE_URL,
        "bytes": CURRENT_WPI_SOURCE_BYTES,
        "sha256": CURRENT_WPI_SOURCE_SHA256,
        "records": CURRENT_WPI_SOURCE_RECORDS,
    }
    assert PINNED_UNLOCODE_ARTIFACT.model_dump() == {
        "run_name": "unlocode-2025-1-all-functions-v1",
        "release": "2025-1",
        "relative_path": "unlocode-2025-1-all-functions-v1/locations.jsonl",
        "bytes": 25_475_870,
        "sha256": "c5feda51b7a8e39bb97bdd107fc5d96eed4e973461c27adf053a134c7090c278",
        "records": 111_561,
    }
    with pytest.raises(ValidationError, match="official NGA"):
        WpiSourcePin(
            source_url="https://example.com/UpdatedPub150.csv",
            bytes=1,
            sha256="0" * 64,
            records=1,
        )


def test_compiler_intersects_exact_locodes_ignores_country_names_and_excludes_duplicate_ids(
    tmp_path: Path,
) -> None:
    result = _compile(tmp_path)

    assert result.receipt.audit.model_dump() == {
        "source_rows": 8,
        "excluded_blank_locode_rows": 1,
        "excluded_malformed_locode_rows": 1,
        "excluded_locode_absent_from_pinned_unlocode_rows": 1,
        "intersected_source_rows": 5,
        "spaced_locode_rows": 4,
        "compact_locode_rows": 1,
        "accepted_port_records": 3,
        "excluded_duplicate_wpi_number_groups": 1,
        "excluded_duplicate_wpi_number_rows": 2,
        "multiple_facility_locodes": 1,
        "distinct_country_codes": 2,
    }
    payload = (result.root / "port-whitelist.jsonl").read_bytes()
    assert payload.endswith(b"\n")
    lines = payload.splitlines()
    assert all(line == canonical_json_bytes(json.loads(line)) for line in lines)
    records = [json.loads(line) for line in lines]
    assert [(row["locode"], row["world_port_index_number"]) for row in records] == [
        ("DEHAM", 100),
        ("DEHAM", 101),
        ("FRMRS", 200),
    ]
    assert records[0] == {
        "alternate_port_name": "Hamburg-Waltershof",
        "country_code": "DE",
        "latitude": "53.550000000000000",
        "locode": "DEHAM",
        "longitude": "9.983333000000000",
        "port_name": "Hamburg West",
        "schema_version": 1,
        "source_oid": 3,
        "world_port_index_number": 100,
    }
    assert records[2]["country_code"] == "FR"
    assert 300 not in {row["world_port_index_number"] for row in records}
    assert result.receipt.policy.country_identity == (
        "first_two_characters_of_intersected_unlocode_v1"
    )
    assert "country" not in result.receipt.policy.intersection


def test_receipt_is_canonical_self_hashed_and_commit_is_idempotent(tmp_path: Path) -> None:
    first = _compile(tmp_path)
    receipt_payload = (first.root / "registry-receipt.json").read_bytes()
    assert receipt_payload == canonical_json_bytes(json.loads(receipt_payload)) + b"\n"
    assert (
        WorldPortRegistryReceipt.model_validate_json(receipt_payload, strict=True) == first.receipt
    )
    assert tuple(row.relative_path for row in first.commit_receipt.artifacts) == (
        "port-whitelist.jsonl",
        "registry-receipt.json",
    )
    second = _compile(tmp_path)
    assert second.created is False
    assert second.commit_receipt == first.commit_receipt

    (first.root / "port-whitelist.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(StagedRunError, match="artifact inventory differs"):
        _compile(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("size", "size mismatch"),
        ("hash", "SHA-256 mismatch"),
        ("records", "record count mismatch"),
    ],
)
def test_source_pin_size_hash_and_count_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    source_path, source, unlocode_path, unlocode = _write_inputs(tmp_path / mutation)
    if mutation == "size":
        source = source.model_copy(update={"bytes": source.bytes + 1})
    elif mutation == "hash":
        source = source.model_copy(update={"sha256": "0" * 64})
    else:
        source = source.model_copy(update={"records": source.records + 1})
    with pytest.raises(WorldPortRegistryError, match=message):
        compile_world_port_registry(
            source_csv=source_path,
            source=source,
            unlocode_locations_jsonl=unlocode_path,
            unlocode_artifact=unlocode,
            output_parent=tmp_path / "artifacts",
            run_name=mutation,
        )


def test_source_symlink_header_shape_duplicate_oid_and_ambiguous_identity_fail_closed(
    tmp_path: Path,
) -> None:
    source_path, source, unlocode_path, unlocode = _write_inputs(tmp_path / "strict")
    symlink = tmp_path / "source-link.csv"
    symlink.symlink_to(source_path)
    with pytest.raises(WorldPortRegistryError, match="symbolic link"):
        compile_world_port_registry(
            source_csv=symlink,
            source=source,
            unlocode_locations_jsonl=unlocode_path,
            unlocode_artifact=unlocode,
            output_parent=tmp_path / "symlink-out",
        )

    bad_header = list(WPI_CSV_COLUMNS)
    bad_header[0] = "OID"
    payload = _wpi_csv(_fixture_rows(), header=tuple(bad_header))
    source_path.write_bytes(payload)
    changed = source.model_copy(update={"bytes": len(payload), "sha256": sha256_bytes(payload)})
    with pytest.raises(WorldPortRegistryError, match="header differs"):
        compile_world_port_registry(
            source_csv=source_path,
            source=changed,
            unlocode_locations_jsonl=unlocode_path,
            unlocode_artifact=unlocode,
            output_parent=tmp_path / "header-out",
        )

    duplicate_oid_rows = _fixture_rows()
    duplicate_oid_rows[-1][WPI_CSV_COLUMNS.index("OID_")] = "6.0"
    with pytest.raises(WorldPortRegistryError, match="duplicate source OID"):
        _compile(tmp_path, rows=duplicate_oid_rows, run_name="duplicate-oid")

    ambiguous_rows = _fixture_rows()
    ambiguous_rows[-1][WPI_CSV_COLUMNS.index("UN/LOCODE")] = "DE HAM"
    with pytest.raises(WorldPortRegistryError, match="multiple UN/LOCODE identities"):
        _compile(tmp_path, rows=ambiguous_rows, run_name="ambiguous-wpi")


def test_selected_metadata_and_unlocode_pin_fail_closed(tmp_path: Path) -> None:
    invalid_coordinate = _fixture_rows()
    invalid_coordinate[3][WPI_CSV_COLUMNS.index("Latitude")] = "91.000"
    with pytest.raises(WorldPortRegistryError, match=r"latitude is outside \[-90, 90\]"):
        _compile(tmp_path, rows=invalid_coordinate, run_name="coordinate")

    source_path, source, unlocode_path, unlocode = _write_inputs(tmp_path / "unlocode")
    with pytest.raises(WorldPortRegistryError, match="size mismatch"):
        compile_world_port_registry(
            source_csv=source_path,
            source=source,
            unlocode_locations_jsonl=unlocode_path,
            unlocode_artifact=unlocode.model_copy(update={"bytes": unlocode.bytes + 1}),
            output_parent=tmp_path / "unlocode-out",
        )


def test_loader_enforces_hash_count_canonical_json_order_and_unique_wpi_number(
    tmp_path: Path,
) -> None:
    result = _compile(tmp_path)
    artifact = result.receipt.artifacts[0]
    path = result.root / artifact.path
    loaded = load_pinned_world_port_records(
        path,
        expected_sha256=artifact.sha256,
        expected_records=artifact.records,
    )
    assert [(row.locode, row.world_port_index_number) for row in loaded] == [
        ("DEHAM", 100),
        ("DEHAM", 101),
        ("FRMRS", 200),
    ]
    with pytest.raises(WorldPortRegistryError, match="SHA-256 mismatch"):
        load_pinned_world_port_records(
            path,
            expected_sha256="0" * 64,
            expected_records=artifact.records,
        )
    with pytest.raises(WorldPortRegistryError, match="count mismatch"):
        load_pinned_world_port_records(
            path,
            expected_sha256=artifact.sha256,
            expected_records=artifact.records + 1,
        )

    noncanonical = tmp_path / "noncanonical.jsonl"
    payload = json.dumps(loaded[0].model_dump(mode="json"), ensure_ascii=False).encode() + b"\n"
    noncanonical.write_bytes(payload)
    with pytest.raises(WorldPortRegistryError, match="not canonical JSON"):
        load_pinned_world_port_records(
            noncanonical,
            expected_sha256=sha256_bytes(payload),
            expected_records=1,
        )

    repeated = tmp_path / "repeated.jsonl"
    second = loaded[1].model_copy(update={"world_port_index_number": 100})
    repeated_payload = b"".join(
        canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in (loaded[0], second)
    )
    repeated.write_bytes(repeated_payload)
    with pytest.raises(WorldPortRegistryError, match=r"duplicated|repeated WPI"):
        load_pinned_world_port_records(
            repeated,
            expected_sha256=sha256_bytes(repeated_payload),
            expected_records=2,
        )


def test_models_reject_nonprefix_country_invalid_coordinate_and_unbalanced_audit() -> None:
    body = {
        "world_port_index_number": 1,
        "source_oid": 0,
        "locode": "DEHAM",
        "country_code": "FR",
        "port_name": "Hamburg",
        "alternate_port_name": None,
        "latitude": "53.5",
        "longitude": "9.9",
    }
    with pytest.raises(ValidationError, match="UN/LOCODE prefix"):
        WorldPortRecord.model_validate(body, strict=True)
    with pytest.raises(ValidationError, match="outside"):
        WorldPortRecord.model_validate(
            {**body, "country_code": "DE", "longitude": "181.0"}, strict=True
        )
    with pytest.raises(ValidationError, match="does not balance"):
        WorldPortIntersectionAudit(
            source_rows=1,
            excluded_blank_locode_rows=0,
            excluded_malformed_locode_rows=0,
            excluded_locode_absent_from_pinned_unlocode_rows=0,
            intersected_source_rows=0,
            spaced_locode_rows=0,
            compact_locode_rows=0,
            accepted_port_records=0,
            excluded_duplicate_wpi_number_groups=0,
            excluded_duplicate_wpi_number_rows=0,
            multiple_facility_locodes=0,
            distinct_country_codes=0,
        )
