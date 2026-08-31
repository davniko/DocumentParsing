from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.country_registry import CountryEntry, CountryRegistry
from document_ocr.synthesis.routes import TradeFlowRecord
from document_ocr.synthesis.run_safety import StagedRunError
from document_ocr.synthesis.trade_flow_registry import (
    CompiledWitsTradeFlows,
    WitsCountryMetadataAudit,
    WitsReporterPin,
    WitsSourceManifest,
    WitsTradeFlowError,
    WitsTradeFlowReceipt,
    _parse_reporter,
    _parse_wits_country_metadata,
    build_wits_source_manifest,
    compile_wits_trade_flows,
)

_HASH = "1" * 64
_FLOW_URL = (
    "https://wits.worldbank.org/API/V1/SDMX/V21/datasource/tradestats-trade/"
    "reporter/{reporter}/year/all/partner/all/product/Total/indicator/XPRT-TRD-VL"
)
_METADATA_URL = "https://wits.worldbank.org/API/V1/wits/datasource/tradestats-trade/country/ALL"


def _countries(*, iso_sha256: str = _HASH) -> CountryRegistry:
    entries = (
        CountryEntry(
            alpha2="CD",
            alpha3="COD",
            numeric="180",
            name="Congo, The Democratic Republic of the",
        ),
        CountryEntry(alpha2="DE", alpha3="DEU", numeric="276", name="Germany"),
        CountryEntry(alpha2="FR", alpha3="FRA", numeric="250", name="France"),
        CountryEntry(alpha2="RO", alpha3="ROU", numeric="642", name="Romania"),
        CountryEntry(alpha2="US", alpha3="USA", numeric="840", name="United States"),
    )
    return CountryRegistry(
        entries=entries,
        observed_aliases={},
        iso_sha256=iso_sha256,
        observed_aliases_sha256="2" * 64,
    )


def _metadata_country(
    code: str,
    numeric: str,
    name: str,
    *,
    reporter: str = "1",
    partner: str = "1",
    group: str = "No",
    group_type: str = "N/A",
) -> str:
    return (
        f'<wits:country countrycode="{numeric}" isreporter="{reporter}" '
        f'ispartner="{partner}" isgroup="{group}" grouptype="{group_type}">'
        f"<wits:iso3Code>{code}</wits:iso3Code><wits:name>{name}</wits:name>"
        "<wits:notes /></wits:country>"
    )


def _metadata(*, romania_name: str = "Romania") -> bytes:
    rows = (
        _metadata_country("DEU", "276", "Germany"),
        _metadata_country("FRA", "250", "France"),
        _metadata_country("OAS", "490", "Other Asia, nes"),
        _metadata_country("ROM", "642", romania_name),
        _metadata_country("SDN", "736", "Fm Sudan"),
        _metadata_country("USA", "840", "United States"),
        _metadata_country("WLD", "000", "World", group="Yes", group_type="Region"),
        _metadata_country("ZAR", "180", "Congo, Dem. Rep."),
    )
    return (
        '<wits:datasource datasourcecode="tradestats-trade" '
        'datasourcename="WITS TradeStats - Trade" language="en" total="8" '
        'xmlns:wits="http://wits.worldbank.org"><wits:countries>'
        + "".join(rows)
        + "</wits:countries></wits:datasource>"
    ).encode()


def _series(
    reporter: str,
    partner: str,
    observations: tuple[tuple[str, str], ...],
    *,
    datasource: str = "WITS-CMT",
) -> str:
    children = "".join(
        f'<Obs TIME_PERIOD="{year}" OBS_VALUE="{value}" DATASOURCE="{datasource}"/>'
        for year, value in observations
    )
    return (
        f'<Series FREQ="A" REPORTER="{reporter}" PARTNER="{partner}" '
        f'PRODUCTCODE="Total" INDICATOR="XPRT-TRD-VL">{children}</Series>'
    )


def _xml(series: tuple[str, ...]) -> bytes:
    return ("<StructureSpecificData>" + "".join(series) + "</StructureSpecificData>").encode()


def _source_fixture(root: Path) -> Path:
    raw = root / "raw"
    metadata = root / "metadata"
    raw.mkdir(parents=True)
    metadata.mkdir()
    (metadata / "countries.xml").write_bytes(_metadata())
    (raw / "DEU.xml").write_bytes(
        _xml(
            (
                _series("DEU", "FRA", (("2022", "1"), ("2023", "2.123456789"))),
                _series("DEU", "ROM", (("2023", "3"),)),
                _series("DEU", "ZAR", (("2023", "6.12345678901"),)),
                _series("DEU", "OAS", (("2023", "999"),)),
                _series("DEU", "WLD", (("2023", "1000"),)),
                _series("DEU", "DEU", (("2023", "5"),)),
                _series("DEU", "USA", (("2023", "0"),)),
            )
        )
    )
    (raw / "ROM.xml").write_bytes(_xml((_series("ROM", "DEU", (("2023", "7.25"),)),)))
    (raw / "SDN.xml").write_bytes(_xml((_series("SDN", "DEU", (("2011", "123"),)),)))
    return root


def _build(root: Path, *, countries: CountryRegistry | None = None) -> Path:
    manifest_path = root / "source-manifest-v2.json"
    build_wits_source_manifest(
        raw_dir=root / "raw",
        country_metadata_path=root / "metadata" / "countries.xml",
        countries=_countries() if countries is None else countries,
        output_path=manifest_path,
    )
    return manifest_path


def _compile(root: Path, *, run_name: str = "registry-v2") -> CompiledWitsTradeFlows:
    manifest_path = _build(root)
    return compile_wits_trade_flows(
        source_manifest_path=manifest_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        countries=_countries(),
        output_parent=root / "artifacts",
        run_name=run_name,
    )


def test_country_metadata_resolves_only_by_numeric_and_audits_names() -> None:
    resolutions, audit = _parse_wits_country_metadata(_metadata(), countries=_countries())

    assert resolutions["ROM"].iso_alpha2 == "RO"
    assert resolutions["ROM"].iso_alpha3 == "ROU"
    assert resolutions["ROM"].provider_code_matches_iso_alpha3 is False
    assert resolutions["OAS"].classification == "non_country"
    assert resolutions["OAS"].iso_alpha2 is None
    assert resolutions["WLD"].classification == "group"
    assert resolutions["ZAR"].iso_alpha2 == "CD"
    assert resolutions["ZAR"].normalized_name_matches is False
    assert audit.model_dump(exclude={"classification_sha256", "name_audit_sha256"}) == {
        "records": 8,
        "iso_country_records": 5,
        "group_records": 1,
        "non_country_records": 2,
        "normalized_name_match_records": 4,
        "normalized_name_difference_records": 1,
        "provider_code_match_records": 3,
        "provider_code_difference_records": 2,
    }


def test_name_differences_never_change_numeric_identity() -> None:
    resolutions, audit = _parse_wits_country_metadata(
        _metadata(romania_name="A deliberately different audit name"),
        countries=_countries(),
    )
    assert resolutions["ROM"].iso_alpha2 == "RO"
    assert resolutions["ROM"].normalized_name_matches is False
    assert audit.normalized_name_difference_records == 2


def test_manifest_is_canonical_metadata_pinned_and_classifies_every_source(
    tmp_path: Path,
) -> None:
    root = _source_fixture(tmp_path / "source")
    manifest_path = _build(root)
    payload = manifest_path.read_bytes()
    assert payload == canonical_json_bytes(json.loads(payload)) + b"\n"
    manifest = WitsSourceManifest.model_validate_json(payload, strict=True)
    assert manifest.schema_version == 2
    assert manifest.iso3166_sha256 == _HASH
    assert manifest.country_identity_policy == "exact_wits_to_iso_numeric_code_v1"
    assert manifest.country_metadata.source_url == _METADATA_URL
    assert manifest.country_metadata.sha256 == sha256_file(root / "metadata/countries.xml")
    pins = {row.wits_reporter_code: row for row in manifest.reporters}
    assert (pins["ROM"].origin_country_code, pins["ROM"].origin_alpha3) == ("RO", "ROU")
    assert pins["ROM"].wits_numeric_code == "642"
    assert pins["SDN"].classification == "non_country"
    assert pins["SDN"].origin_country_code is None
    assert tuple(pins) == ("DEU", "ROM", "SDN")
    assert all(row.sha256 == sha256_file(root / row.path) for row in manifest.reporters)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            b"<!DOCTYPE x><wits:datasource xmlns:wits='http://wits.worldbank.org'/>",
            "forbidden",
        ),
        (b"<wrong/>", "root contract differs"),
        (_metadata().replace(b'total="8"', b'total="9"'), "total differs"),
        (_metadata().replace(b'countrycode="642"', b'countrycode="276"'), "duplicated"),
        (_metadata().replace(b'isgroup="No"', b'isgroup="Maybe"', 1), "group flags"),
    ],
)
def test_country_metadata_contract_drift_fails_closed(payload: bytes, message: str) -> None:
    with pytest.raises(WitsTradeFlowError, match=message):
        _parse_wits_country_metadata(payload, countries=_countries())


def test_manifest_rejects_unclassified_reporter_extra_file_and_symlink(tmp_path: Path) -> None:
    root = _source_fixture(tmp_path / "source")
    (root / "raw" / "ZZZ.xml").write_bytes(_xml((_series("ZZZ", "DEU", (("2023", "1"),)),)))
    with pytest.raises(WitsTradeFlowError, match="absent or not reporter-eligible"):
        _build(root)

    (root / "raw" / "ZZZ.xml").unlink()
    (root / "raw" / "README.txt").write_text("unexpected")
    with pytest.raises(WitsTradeFlowError, match="invalid entries"):
        _build(root)

    (root / "raw" / "README.txt").unlink()
    (root / "raw" / "DEU.xml").unlink()
    (root / "raw" / "DEU.xml").symlink_to(root / "raw" / "ROM.xml")
    with pytest.raises(WitsTradeFlowError, match="invalid entries"):
        _build(root)


def test_compiler_preserves_exact_decimals_and_balances_all_dispositions(
    tmp_path: Path,
) -> None:
    result = _compile(_source_fixture(tmp_path / "source"))
    receipt = result.receipt
    assert result.created is True
    assert receipt.source_reporter_files == 3
    assert receipt.compiled_reporter_files == 2
    assert receipt.excluded_reporter_files == 1
    assert receipt.trade_flow_records == 4
    assert receipt.distinct_origin_countries == 2
    assert receipt.distinct_destination_countries == 4
    assert receipt.trade_value_representation == "exact_decimal_max_11_places_v1"
    assert receipt.excluded_reporters[0].model_dump() == {
        "wits_reporter_code": "SDN",
        "wits_numeric_code": "736",
        "wits_name": "Fm Sudan",
        "classification": "non_country",
    }
    assert receipt.reporter_audits[0].model_dump() == {
        "origin_country_code": "DE",
        "wits_reporter_code": "DEU",
        "selected_year": 2023,
        "selected_year_observations": 7,
        "accepted_positive_country_observations": 3,
        "excluded_group_partner_observations": 1,
        "excluded_non_country_partner_observations": 1,
        "excluded_domestic_observations": 1,
        "excluded_nonpositive_observations": 1,
        "duplicate_partner_observations": 0,
        "max_trade_value_decimal_places": 11,
    }
    payload = (result.root / "trade-flows.jsonl").read_bytes()
    lines = payload.splitlines()
    assert [json.loads(line) for line in lines] == [
        {
            "destination_country_code": "CD",
            "net_mass": None,
            "origin_country_code": "DE",
            "trade_value": "6.12345678901",
            "year": 2023,
        },
        {
            "destination_country_code": "FR",
            "net_mass": None,
            "origin_country_code": "DE",
            "trade_value": "2.123456789",
            "year": 2023,
        },
        {
            "destination_country_code": "RO",
            "net_mass": None,
            "origin_country_code": "DE",
            "trade_value": "3",
            "year": 2023,
        },
        {
            "destination_country_code": "DE",
            "net_mass": None,
            "origin_country_code": "RO",
            "trade_value": "7.25",
            "year": 2023,
        },
    ]
    assert all(line == canonical_json_bytes(json.loads(line)) for line in lines)
    assert (
        WitsTradeFlowReceipt.model_validate_json(
            (result.root / "registry-receipt.json").read_bytes(), strict=True
        )
        == receipt
    )


def test_sqlite_is_deterministic_query_equivalent_and_integral(tmp_path: Path) -> None:
    first = _compile(_source_fixture(tmp_path / "first"), run_name="first-v2")
    second = _compile(_source_fixture(tmp_path / "second"), run_name="second-v2")
    first_path = first.root / "trade-flows.sqlite3"
    second_path = second.root / "trade-flows.sqlite3"
    assert first_path.read_bytes() == second_path.read_bytes()
    with sqlite3.connect(f"file:{first_path}?mode=ro&immutable=1", uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        assert connection.execute("PRAGMA application_id").fetchone() == (0x57495453,)
        assert connection.execute(
            "SELECT origin_country_code, destination_country_code, year, trade_value "
            "FROM trade_flows ORDER BY origin_country_code, destination_country_code"
        ).fetchall() == [
            ("DE", "CD", 2023, "6.12345678901"),
            ("DE", "FR", 2023, "2.123456789"),
            ("DE", "RO", 2023, "3"),
            ("RO", "DE", 2023, "7.25"),
        ]


def test_unknown_or_non_partner_code_never_falls_back() -> None:
    metadata = _metadata().replace(b'ispartner="1"', b'ispartner="0"', 1)
    resolutions, _audit = _parse_wits_country_metadata(metadata, countries=_countries())
    payload = _xml((_series("ROM", "DEU", (("2023", "1"),)),))
    pin = WitsReporterPin(
        wits_reporter_code="ROM",
        wits_numeric_code="642",
        wits_name="Romania",
        classification="iso_country",
        origin_country_code="RO",
        origin_alpha3="ROU",
        iso_numeric_code="642",
        normalized_name_matches=True,
        provider_code_matches_iso_alpha3=False,
        path="raw/ROM.xml",
        source_url=_FLOW_URL.format(reporter="ROM"),
        bytes=len(payload),
        sha256=sha256_bytes(payload),
    )
    with pytest.raises(WitsTradeFlowError, match="absent or not partner-eligible"):
        _parse_reporter(payload, pin=pin, wits_countries=resolutions)


def test_trade_flow_domain_preserves_measured_11_places_and_rejects_more() -> None:
    exact = TradeFlowRecord.model_validate(
        {
            "origin_country_code": "DE",
            "destination_country_code": "FR",
            "year": 2023,
            "trade_value": Decimal("6664.18249317348"),
            "net_mass": None,
        },
        strict=True,
    )
    assert exact.trade_value == Decimal("6664.18249317348")
    with pytest.raises(ValidationError, match="11 decimal places"):
        TradeFlowRecord.model_validate(
            {
                "origin_country_code": "DE",
                "destination_country_code": "FR",
                "year": 2023,
                "trade_value": Decimal("1.123456789012"),
                "net_mass": None,
            },
            strict=True,
        )


def test_compile_fails_on_manifest_metadata_source_and_commit_tampering(tmp_path: Path) -> None:
    root = _source_fixture(tmp_path / "source")
    manifest_path = _build(root)
    manifest_sha = sha256_file(manifest_path)
    with pytest.raises(WitsTradeFlowError, match="manifest SHA-256 mismatch"):
        compile_wits_trade_flows(
            source_manifest_path=manifest_path,
            expected_manifest_sha256="0" * 64,
            countries=_countries(),
            output_parent=root / "bad-manifest",
            run_name="registry-v2",
        )
    with pytest.raises(WitsTradeFlowError, match="different ISO-3166 snapshot"):
        compile_wits_trade_flows(
            source_manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            countries=_countries(iso_sha256="3" * 64),
            output_parent=root / "bad-iso",
            run_name="registry-v2",
        )
    (root / "metadata" / "countries.xml").write_bytes(b"tampered")
    with pytest.raises(WitsTradeFlowError, match=r"size mismatch|SHA-256 mismatch"):
        compile_wits_trade_flows(
            source_manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            countries=_countries(),
            output_parent=root / "bad-metadata",
            run_name="registry-v2",
        )

    clean_root = _source_fixture(tmp_path / "clean")
    first = _compile(clean_root)
    second = compile_wits_trade_flows(
        source_manifest_path=clean_root / "source-manifest-v2.json",
        expected_manifest_sha256=sha256_file(clean_root / "source-manifest-v2.json"),
        countries=_countries(),
        output_parent=clean_root / "artifacts",
        run_name="registry-v2",
    )
    assert second.created is False
    (first.root / "trade-flows.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(StagedRunError, match="artifact inventory differs"):
        compile_wits_trade_flows(
            source_manifest_path=clean_root / "source-manifest-v2.json",
            expected_manifest_sha256=sha256_file(clean_root / "source-manifest-v2.json"),
            countries=_countries(),
            output_parent=clean_root / "artifacts",
            run_name="registry-v2",
        )


def test_metadata_audit_model_rejects_unbalanced_counts() -> None:
    with pytest.raises(ValidationError, match="classifications do not balance"):
        WitsCountryMetadataAudit(
            records=2,
            iso_country_records=1,
            group_records=0,
            non_country_records=0,
            normalized_name_match_records=1,
            normalized_name_difference_records=0,
            provider_code_match_records=1,
            provider_code_difference_records=0,
            classification_sha256=_HASH,
            name_audit_sha256=_HASH,
        )
