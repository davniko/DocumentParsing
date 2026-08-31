from __future__ import annotations

import json
import sqlite3
import stat
import zipfile
from pathlib import Path

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.locality_registry import (
    GEONAMES_LICENSE,
    GEONAMES_SOURCE_URL,
    GeoNamesLocalityError,
    GeoNamesLocalityReceipt,
    GeoNamesSourceReceipt,
    build_geonames_source_receipt,
    compile_geonames_locality_registry,
    load_geonames_locality_registry,
)
from document_ocr.synthesis.run_safety import StagedRunError


def _iso(path: Path) -> str:
    payload = canonical_json_bytes(
        {
            "3166-1": [
                {"alpha_2": "SI", "alpha_3": "SVN", "name": "Slovenia", "numeric": "705"},
                {
                    "alpha_2": "US",
                    "alpha_3": "USA",
                    "name": "United States",
                    "numeric": "840",
                },
            ]
        }
    )
    path.write_bytes(payload)
    return sha256_bytes(payload)


def _row(
    geoname_id: str,
    name: str,
    ascii_name: str,
    country: str,
    population: str,
    *,
    feature_class: str = "P",
    feature_code: str = "PPL",
    latitude: str = "46.05695",
    longitude: str = "14.50513",
    timezone: str = "Europe/Ljubljana",
    aliases: str = "ignored,alias",
) -> str:
    fields = [
        geoname_id,
        name,
        ascii_name,
        aliases,
        latitude,
        longitude,
        feature_class,
        feature_code,
        country,
        "",
        "00",
        "",
        "",
        "",
        population,
        "",
        "300",
        timezone,
        "2026-08-31",
    ]
    assert len(fields) == 19
    return "\t".join(fields)


def _archive(path: Path, rows: tuple[str, ...], *, member: str = "cities15000.txt") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member, date_time=(2026, 8, 31, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(info, ("\n".join(rows) + "\n").encode())


def _source(root: Path, rows: tuple[str, ...]) -> tuple[Path, Path]:
    archive_path = root / "raw" / "cities15000.zip"
    receipt_path = root / "source-receipt.json"
    _archive(archive_path, rows)
    build_geonames_source_receipt(archive_path=archive_path, output_path=receipt_path)
    return archive_path, receipt_path


def _compile(root: Path, rows: tuple[str, ...], *, run_name: str = "registry"):
    _archive_path, source_receipt_path = _source(root / "source", rows)
    iso_path = root / "iso.json"
    iso_sha256 = _iso(iso_path)
    return compile_geonames_locality_registry(
        source_receipt_path=source_receipt_path,
        expected_source_receipt_sha256=sha256_file(source_receipt_path),
        iso_path=iso_path,
        expected_iso_sha256=iso_sha256,
        output_parent=root / "artifacts",
        run_name=run_name,
    )


def test_source_receipt_pins_official_identity_archive_and_member(tmp_path: Path) -> None:
    archive_path, receipt_path = _source(
        tmp_path,
        (_row("3196359", "Ljubljana", "Ljubljana", "SI", "272220"),),
    )
    payload = receipt_path.read_bytes()
    receipt = GeoNamesSourceReceipt.model_validate_json(payload, strict=True)

    assert payload == canonical_json_bytes(json.loads(payload)) + b"\n"
    assert receipt.source_url == GEONAMES_SOURCE_URL
    assert receipt.license == GEONAMES_LICENSE
    assert receipt.archive_path == "raw/cities15000.zip"
    assert receipt.archive_bytes == archive_path.stat().st_size
    assert receipt.archive_sha256 == sha256_file(archive_path)
    assert len(receipt.members) == 1
    assert receipt.members[0].path == "cities15000.txt"
    assert receipt.members[0].uncompressed_bytes > 0
    assert receipt.members[0].sha256 == sha256_bytes(
        (_row("3196359", "Ljubljana", "Ljubljana", "SI", "272220") + "\n").encode()
    )


def test_compiler_keeps_canonical_fields_and_audits_non_iso_exclusion(
    tmp_path: Path,
) -> None:
    rows = (
        _row(
            "3196359",
            "Ljubljana",
            "Ljubljana",
            "SI",
            "272220",
            feature_code="PPLC",
            aliases="Laibach,Emona",
        ),
        _row(
            "5128581",
            "New York City",
            "New York City",
            "US",
            "0",
            latitude="40.71427",
            longitude="-74.00597",
            timezone="America/New_York",
        ),
        _row("786714", "Prishtina", "Prishtina", "XK", "550000"),
    )
    result = _compile(tmp_path, rows)

    assert result.created is True
    audit = result.receipt.locality_audit
    assert audit.source_records == 3
    assert audit.accepted_records == 2
    assert audit.excluded_non_iso_country_records == 1
    assert dict(audit.excluded_non_iso_country_counts) == {"XK": 1}
    assert dict(audit.country_code_counts) == {"SI": 1, "US": 1}
    assert dict(audit.feature_code_counts) == {"PPL": 1, "PPLC": 1}
    assert audit.zero_population_records == 1
    assert audit.minimum_population == 0
    assert audit.maximum_population == 272220

    payload = (result.root / "localities.jsonl").read_bytes()
    lines = payload.splitlines()
    assert [json.loads(line) for line in lines] == [
        {
            "ascii_name": "Ljubljana",
            "canonical_name": "Ljubljana",
            "country_code": "SI",
            "feature_code": "PPLC",
            "geoname_id": 3196359,
            "latitude": "46.05695",
            "longitude": "14.50513",
            "population": 272220,
            "timezone": "Europe/Ljubljana",
        },
        {
            "ascii_name": "New York City",
            "canonical_name": "New York City",
            "country_code": "US",
            "feature_code": "PPL",
            "geoname_id": 5128581,
            "latitude": "40.71427",
            "longitude": "-74.00597",
            "population": 0,
            "timezone": "America/New_York",
        },
    ]
    assert all(line == canonical_json_bytes(json.loads(line)) for line in lines)
    assert all("alias" not in json.loads(line) for line in lines)
    assert (
        GeoNamesLocalityReceipt.model_validate_json(
            (result.root / "registry-receipt.json").read_bytes(), strict=True
        )
        == result.receipt
    )
    registry = load_geonames_locality_registry(
        root=result.root,
        expected_receipt_sha256=sha256_file(result.root / "registry-receipt.json"),
    )
    assert registry.country_codes == ("SI", "US")
    assert registry.entry(3196359).canonical_name == "Ljubljana"
    assert registry.rows_for_country("US")[0].population == 0
    with pytest.raises(KeyError, match="no GeoNames localities"):
        registry.rows_for_country("XK")


def test_sqlite_is_deterministic_query_equivalent_and_population_indexed(
    tmp_path: Path,
) -> None:
    rows = (
        _row("3196359", "Ljubljana", "Ljubljana", "SI", "272220"),
        _row(
            "5128581",
            "New York City",
            "New York City",
            "US",
            "8804190",
            latitude="40.71427",
            longitude="-74.00597",
            timezone="America/New_York",
        ),
    )
    first = _compile(tmp_path / "first", rows, run_name="first")
    second = _compile(tmp_path / "second", rows, run_name="second")
    first_path = first.root / "localities.sqlite3"
    second_path = second.root / "localities.sqlite3"
    assert first_path.read_bytes() == second_path.read_bytes()

    with sqlite3.connect(f"file:{first_path}?mode=ro&immutable=1", uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert connection.execute("PRAGMA application_id").fetchone() == (0x474E4C43,)
        assert connection.execute(
            "SELECT geoname_id, canonical_name, ascii_name, country_code, feature_code, "
            "population, latitude, longitude, timezone FROM localities ORDER BY geoname_id"
        ).fetchall() == [
            (
                3196359,
                "Ljubljana",
                "Ljubljana",
                "SI",
                "PPL",
                272220,
                "46.05695",
                "14.50513",
                "Europe/Ljubljana",
            ),
            (
                5128581,
                "New York City",
                "New York City",
                "US",
                "PPL",
                8804190,
                "40.71427",
                "-74.00597",
                "America/New_York",
            ),
        ]
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT geoname_id, population FROM localities "
            "WHERE country_code = 'US' ORDER BY population DESC, geoname_id"
        ).fetchall()
        assert any("localities_country_population_id_idx" in row[-1] for row in plan)
    assert not tuple(first.root.glob("localities.sqlite3-*"))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_row("1", "Place", "Place", "SI", "1", feature_class="A"), "feature class"),
        (_row("1", "Place", "Place", "SI", "-1"), "invalid population"),
        (_row("0", "Place", "Place", "SI", "1"), "non-positive geoname id"),
        (_row("1", "Place", "Pláce", "SI", "1"), "locality contract"),
        (_row("1", "Place", "Place", "SI", "1", latitude="91"), "locality contract"),
    ],
)
def test_parser_rejects_invalid_geonames_values(tmp_path: Path, row: str, message: str) -> None:
    with pytest.raises(GeoNamesLocalityError, match=message):
        _compile(tmp_path, (row,))


def test_parser_rejects_wrong_column_count_and_duplicate_geoname_id(tmp_path: Path) -> None:
    with pytest.raises(GeoNamesLocalityError, match="18 columns, expected 19"):
        _compile(tmp_path / "columns", ("\t".join(["x"] * 18),))
    duplicate = (
        _row("1", "One", "One", "SI", "1"),
        _row("1", "Two", "Two", "US", "2"),
    )
    with pytest.raises(GeoNamesLocalityError, match="duplicate GeoNames geoname id"):
        _compile(tmp_path / "duplicate", duplicate)


def test_source_receipt_rejects_extra_traversal_and_symlink_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "raw" / "cities15000.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("cities15000.txt", b"data")
        archive.writestr("extra.txt", b"data")
    with pytest.raises(GeoNamesLocalityError, match="must contain only"):
        build_geonames_source_receipt(
            archive_path=archive_path, output_path=tmp_path / "receipt.json"
        )

    _archive(archive_path, ("data",), member="../cities15000.txt")
    with pytest.raises(GeoNamesLocalityError, match="must contain only"):
        build_geonames_source_receipt(
            archive_path=archive_path, output_path=tmp_path / "receipt.json"
        )

    info = zipfile.ZipInfo("cities15000.txt")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(info, b"target")
    with pytest.raises(GeoNamesLocalityError, match="symbolic link"):
        build_geonames_source_receipt(
            archive_path=archive_path, output_path=tmp_path / "receipt.json"
        )


def test_compiler_fails_on_source_iso_and_committed_artifact_tampering(
    tmp_path: Path,
) -> None:
    rows = (_row("3196359", "Ljubljana", "Ljubljana", "SI", "272220"),)
    source_root = tmp_path / "source"
    archive_path, receipt_path = _source(source_root, rows)
    iso_path = tmp_path / "iso.json"
    iso_sha256 = _iso(iso_path)
    kwargs = {
        "source_receipt_path": receipt_path,
        "expected_source_receipt_sha256": sha256_file(receipt_path),
        "iso_path": iso_path,
        "expected_iso_sha256": iso_sha256,
        "output_parent": tmp_path / "artifacts",
        "run_name": "registry",
    }
    with pytest.raises(GeoNamesLocalityError, match="source receipt SHA-256 mismatch"):
        compile_geonames_locality_registry(**{**kwargs, "expected_source_receipt_sha256": "0" * 64})
    with pytest.raises(GeoNamesLocalityError, match="ISO-3166 snapshot SHA-256 mismatch"):
        compile_geonames_locality_registry(**{**kwargs, "expected_iso_sha256": "0" * 64})

    first = compile_geonames_locality_registry(**kwargs)
    second = compile_geonames_locality_registry(**kwargs)
    assert second.created is False
    assert second.commit_receipt == first.commit_receipt

    archive_path.write_bytes(archive_path.read_bytes() + b"tampered")
    with pytest.raises(GeoNamesLocalityError, match="archive bytes differ"):
        compile_geonames_locality_registry(**kwargs)
    archive_path.write_bytes(archive_path.read_bytes()[: -len(b"tampered")])

    (first.root / "localities.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(StagedRunError, match="artifact inventory differs"):
        compile_geonames_locality_registry(**kwargs)
