from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from document_ocr.hashing import canonical_json_bytes
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.hs_registry import (
    HsCodeSurface,
    HsRegistryError,
    UkGlobalTariffSourcePin,
    compile_uk_global_tariff_registry,
    load_ukgt_source_pin,
    render_hs_code_surface,
    sample_global_hs6,
    sample_uk_tariff_code,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_SOURCE_ROOT = PROJECT_ROOT / "data/registries/hs/ukgt-v4.0.1590"
SOURCE_COLUMNS = (
    "id",
    "commodity__sid",
    "commodity__code",
    "commodity__suffix",
    "commodity__description",
    "commodity__validity_start",
    "commodity__validity_end",
    "parent__sid",
    "parent__code",
    "parent__suffix",
)


def _source_row(
    row_id: int,
    sid: str,
    code: str,
    suffix: str,
    description: str,
    parent: tuple[str, str, str] | None,
    *,
    valid_from: str = "2022-01-01",
) -> dict[str, str]:
    parent_values = parent or ("#NA", "#NA", "#NA")
    return dict(
        zip(
            SOURCE_COLUMNS,
            (
                str(row_id),
                sid,
                code,
                suffix,
                description,
                valid_from,
                "#NA",
                *parent_values,
            ),
            strict=True,
        )
    )


def _fixture_rows() -> list[dict[str, str]]:
    chapter_01 = ("1", "0100000000", "80")
    heading_0101 = ("2", "0101000000", "80")
    group_010121 = ("3", "0101210000", "10")
    group_010129 = ("5", "0101290000", "80")
    chapter_05 = ("8", "0500000000", "80")
    chapter_98 = ("13", "9800000000", "80")
    heading_9801 = ("14", "9801000000", "80")
    return [
        _source_row(1, "1", "0100000000", "80", "Live animals", None),
        _source_row(2, "2", "0101000000", "80", "Live horses", chapter_01),
        _source_row(3, "3", "0101210000", "10", "Pure-bred horses", heading_0101),
        _source_row(
            4,
            "4",
            "0101210000",
            "80",
            "Pure-bred breeding\nanimals",
            group_010121,
        ),
        _source_row(5, "5", "0101290000", "80", "Other horses", heading_0101),
        _source_row(6, "6", "0101291000", "80", "For slaughter", group_010129),
        _source_row(7, "7", "0101299000", "80", "Other", group_010129),
        _source_row(8, "8", "0500000000", "80", "Animal products", None),
        _source_row(9, "9", "0501000000", "80", "Human hair", chapter_05),
        _source_row(
            10,
            "10",
            "0101210011",
            "10",
            "Non-declarable grouping",
            group_010121,
        ),
        _source_row(11, "11", "0101295000", "80", "Duplicate", group_010129),
        _source_row(
            12,
            "12",
            "0101295000",
            "80",
            "Duplicate",
            group_010129,
            valid_from="2024-01-01",
        ),
        _source_row(13, "13", "9800000000", "80", "Special use", None),
        _source_row(14, "14", "9801000000", "80", "Special heading", chapter_98),
        _source_row(15, "15", "9801000000", "80", "Special leaf", heading_9801),
    ]


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=SOURCE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _metadata_bytes(*, title: str = "Tariffs to trade with the UK from 1 January 2021") -> bytes:
    payload: dict[str, Any] = {
        "dc:title": title,
        "dc:creator": "Department for International Trade",
        "dc:license": (
            "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
        ),
        "tables": [
            {
                "url": "tables/commodities-report/data?format=csv&download",
                "tableSchema": {"columns": [{"name": name} for name in SOURCE_COLUMNS]},
            }
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def _source_pin(metadata: bytes, report: bytes) -> UkGlobalTariffSourcePin:
    version = "v4.0.1"
    base = f"https://data.api.trade.gov.uk/v1/datasets/uk-tariff-2021-01-01/versions/{version}/"
    return UkGlobalTariffSourcePin(
        version=version,
        snapshot_date=date(2026, 8, 31),
        metadata_url=base + "metadata?format=csvw&download",
        metadata_bytes=len(metadata),
        metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        report_url=base + "tables/commodities-report/data?format=csv&download",
        report_bytes=len(report),
        report_sha256=hashlib.sha256(report).hexdigest(),
        expected_source_rows=15,
        expected_global_hs6_codes_sha256=hashlib.sha256(b"010121\n010129\n050100\n").hexdigest(),
        expected_global_hs6_records=3,
        expected_uk_tariff_records=4,
        hs_edition="HS2022",
        hs_effective_from=date(2022, 1, 1),
        license_name="Open Government Licence v3.0",
        license_url=("https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"),
        license_caveat=(
            "The Open Government Licence excludes third-party rights; UKGT descriptions "
            "are not redistributed as independently licensed WCO nomenclature."
        ),
        attribution=(
            "Contains public sector information from the UK Global Tariff, licensed under "
            "the Open Government Licence v3.0."
        ),
    )


def _compile_fixture(tmp_path: Path):
    metadata = _metadata_bytes()
    report = _csv_bytes(_fixture_rows())
    metadata_path = tmp_path / "metadata.json"
    report_path = tmp_path / "commodities-report.csv"
    metadata_path.write_bytes(metadata)
    report_path.write_bytes(report)
    return compile_uk_global_tariff_registry(
        metadata_path=metadata_path,
        report_path=report_path,
        source=_source_pin(metadata, report),
    )


def test_ukgt_compiler_separates_global_hs6_from_gb_extensions(tmp_path: Path) -> None:
    registry = _compile_fixture(tmp_path)

    assert registry.global_codes == ("010121", "010129", "050100")
    assert registry.uk_codes == ("0101210000", "0101291000", "0101299000", "0501000000")
    assert registry.require_global("010121", on_date=date(2025, 1, 1)).description == (
        "Pure-bred breeding animals"
    )
    assert registry.require_global("010121", on_date=date(2025, 1, 1)).source_description == (
        "Pure-bred breeding\nanimals"
    )
    gb_row = registry.require_uk("0101291000", on_date=date(2025, 1, 1))
    assert gb_row.jurisdiction == "GB"
    assert gb_row.hs6 == "010129"
    assert tuple(row.description for row in gb_row.description_path[-2:]) == (
        "Other horses",
        "For slaughter",
    )
    assert tuple(
        row.code for row in registry.uk_candidates("010129", on_date=date(2025, 1, 1))
    ) == (
        "0101291000",
        "0101299000",
    )
    assert registry.receipt.audit.model_dump() == {
        "source_rows": 15,
        "normalized_description_rows": 1,
        "global_chapters": 2,
        "global_headings": 2,
        "global_hs6_records": 3,
        "hierarchy_leaf_rows": 8,
        "excluded_nondeclarable_suffix_leaf_rows": 1,
        "excluded_ambiguous_code_groups": 1,
        "excluded_ambiguous_code_rows": 2,
        "excluded_special_chapter_leaf_rows": 1,
        "uk_tariff_records": 4,
    }

    with pytest.raises(HsRegistryError, match="unknown or ambiguous"):
        registry.require_uk("0101295000", on_date=date(2025, 1, 1))
    with pytest.raises(HsRegistryError, match="exactly ten digits"):
        registry.require_uk("0101290", on_date=date(2025, 1, 1))
    with pytest.raises(HsRegistryError, match="not proven"):
        registry.require_uk("0101291000", on_date=date(2020, 1, 1))
    with pytest.raises(HsRegistryError, match="not proven"):
        registry.require_global("010129", on_date=date(2027, 1, 1))


def test_sampling_requires_explicit_exact_registry_backed_distributions(tmp_path: Path) -> None:
    registry = _compile_fixture(tmp_path)
    global_path = registry.global_description_path("010129")
    assert (
        global_path[-1] == registry.require_global("010129", on_date=date(2025, 1, 1)).description
    )
    assert (
        global_path[0]
        == registry.require_global("010129", on_date=date(2025, 1, 1)).heading_description
    )
    assert all("slaughter" not in part.lower() for part in global_path)
    stream = DeterministicStream(seed=21, namespace="hs-test", identity="cargo-1")

    global_row = sample_global_hs6(
        registry=registry,
        candidate_weights={"010129": 3, "010121": 1},
        on_date=date(2025, 1, 1),
        stream=stream,
    )
    assert global_row.code in {"010121", "010129"}
    assert global_row == sample_global_hs6(
        registry=registry,
        candidate_weights={"010121": 1, "010129": 3},
        on_date=date(2025, 1, 1),
        stream=stream,
    )

    gb_row = sample_uk_tariff_code(
        registry=registry,
        hs6="010129",
        candidate_weights={"0101291000": 2, "0101299000": 1},
        on_date=date(2025, 1, 1),
        stream=stream,
    )
    assert gb_row.code in {"0101291000", "0101299000"}

    for invalid in ({}, {"0101290": 1}, {"010129": 0}, {"010129": True}):
        with pytest.raises(HsRegistryError):
            sample_global_hs6(
                registry=registry,
                candidate_weights=invalid,
                on_date=date(2025, 1, 1),
                stream=stream,
            )
    with pytest.raises(HsRegistryError, match="does not extend"):
        sample_uk_tariff_code(
            registry=registry,
            hs6="010121",
            candidate_weights={"0101291000": 1},
            on_date=date(2025, 1, 1),
            stream=stream,
        )


def test_hs_surface_formatting_preserves_every_digit_without_inference() -> None:
    assert render_hs_code_surface(
        "0101299000", groups=(2, 2, 2, 4), separator="."
    ) == HsCodeSurface(semantic_code="0101299000", surface="01.01.29.9000")
    assert render_hs_code_surface("0101299000", groups=(10,), separator="") == HsCodeSurface(
        semantic_code="0101299000", surface="0101299000"
    )

    with pytest.raises(ValueError, match="consume every semantic digit"):
        render_hs_code_surface("0101299000", groups=(2, 2, 2), separator=".")
    with pytest.raises(ValueError, match="6-18 digits"):
        render_hs_code_surface("10129", groups=(2, 3), separator=".")
    with pytest.raises(ValueError, match="empty HS surface separator"):
        render_hs_code_surface("010129", groups=(2, 4), separator="")
    with pytest.raises(ValueError, match="unsupported characters"):
        HsCodeSurface(semantic_code="010129", surface="01\t01\t29")
    with pytest.raises(ValueError, match="changes, pads, truncates"):
        HsCodeSurface(semantic_code="010129", surface="01.01.28")


def test_exact_source_pin_and_csvw_semantics_fail_closed(tmp_path: Path) -> None:
    metadata = _metadata_bytes(title="Wrong dataset")
    report = _csv_bytes(_fixture_rows())
    metadata_path = tmp_path / "metadata.json"
    report_path = tmp_path / "report.csv"
    metadata_path.write_bytes(metadata)
    report_path.write_bytes(report)

    with pytest.raises(HsRegistryError, match="unexpected dataset title"):
        compile_uk_global_tariff_registry(
            metadata_path=metadata_path,
            report_path=report_path,
            source=_source_pin(metadata, report),
        )

    valid_metadata = _metadata_bytes()
    metadata_path.write_bytes(valid_metadata)
    source = _source_pin(valid_metadata, report)
    report_path.write_bytes(report + b"tampered")
    with pytest.raises(HsRegistryError, match="size mismatch"):
        compile_uk_global_tariff_registry(
            metadata_path=metadata_path,
            report_path=report_path,
            source=source,
        )


def test_hs6_set_hash_and_exclusive_validity_end_fail_closed(tmp_path: Path) -> None:
    metadata = _metadata_bytes()
    rows = _fixture_rows()
    report = _csv_bytes(rows)
    metadata_path = tmp_path / "metadata.json"
    report_path = tmp_path / "report.csv"
    metadata_path.write_bytes(metadata)
    report_path.write_bytes(report)
    source = _source_pin(metadata, report)

    with pytest.raises(HsRegistryError, match="code-set SHA-256 mismatch"):
        compile_uk_global_tariff_registry(
            metadata_path=metadata_path,
            report_path=report_path,
            source=source.model_copy(update={"expected_global_hs6_codes_sha256": "0" * 64}),
        )

    rows[5]["commodity__validity_end"] = source.snapshot_date.isoformat()
    report = _csv_bytes(rows)
    report_path.write_bytes(report)
    with pytest.raises(HsRegistryError, match="not active on pinned snapshot date"):
        compile_uk_global_tariff_registry(
            metadata_path=metadata_path,
            report_path=report_path,
            source=_source_pin(metadata, report),
        )


def test_hierarchy_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    rows = _fixture_rows()
    rows[3]["parent__code"] = "0101290000"
    metadata = _metadata_bytes()
    report = _csv_bytes(rows)
    metadata_path = tmp_path / "metadata.json"
    report_path = tmp_path / "report.csv"
    metadata_path.write_bytes(metadata)
    report_path.write_bytes(report)

    with pytest.raises(HsRegistryError, match="parent identity mismatch"):
        compile_uk_global_tariff_registry(
            metadata_path=metadata_path,
            report_path=report_path,
            source=_source_pin(metadata, report),
        )


def test_ambiguous_structural_heading_identity_fails_closed(tmp_path: Path) -> None:
    rows = _fixture_rows()
    rows[9].update(
        {
            "commodity__code": "0101000000",
            "commodity__suffix": "80",
            "parent__sid": "1",
            "parent__code": "0100000000",
            "parent__suffix": "80",
        }
    )
    metadata = _metadata_bytes()
    report = _csv_bytes(rows)
    metadata_path = tmp_path / "metadata.json"
    report_path = tmp_path / "report.csv"
    metadata_path.write_bytes(metadata)
    report_path.write_bytes(report)

    with pytest.raises(HsRegistryError, match="heading identities are ambiguous"):
        compile_uk_global_tariff_registry(
            metadata_path=metadata_path,
            report_path=report_path,
            source=_source_pin(metadata, report),
        )


def test_real_ukgt_source_manifest_is_canonical_and_pinned() -> None:
    pin_path = REAL_SOURCE_ROOT / "source-manifest.json"
    pin = load_ukgt_source_pin(pin_path)

    assert pin.version == "v4.0.1590"
    assert pin.license_name == "Open Government Licence v3.0"
    assert pin_path.read_bytes() == canonical_json_bytes(pin.model_dump(mode="json")) + b"\n"
