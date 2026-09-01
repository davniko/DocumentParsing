from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.vessel_name_registry import (
    LoadedVesselNameRegistry,
    VesselNameRegistryError,
    VesselNameRegistryPolicy,
    VesselNameRegistrySourceManifest,
    VesselNameSourceFilePin,
    VesselNameSourceLicense,
    compile_vessel_name_registry,
    load_vessel_name_registry,
)


def _csv(columns: list[str], rows: list[list[str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _pin(
    root: Path,
    *,
    source_id: str,
    family: str,
    snapshot: str,
    relative: str,
    source_url: str,
    payload: bytes,
    records: int,
) -> VesselNameSourceFilePin:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return VesselNameSourceFilePin.model_validate(
        {
            "source_id": source_id,
            "family": family,
            "snapshot": snapshot,
            "relative_path": relative,
            "source_url": source_url,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "records": records,
        },
        strict=True,
    )


def _sources(root: Path) -> tuple[Path, str]:
    imo_columns = ["imo", "vessel_name", "gross_tonnage", "type", "year_built", "flag"]
    pins = [
        _pin(
            root,
            source_id="imo-fixture",
            family="imo_vessel_names",
            snapshot="2020.1.0",
            relative="raw/imo.csv",
            source_url=(
                "https://raw.githubusercontent.com/interreg-speed/"
                "IMO-VESSEL-NAMES/master/data/3-vessels.csv"
            ),
            payload=_csv(
                imo_columns,
                [
                    ["1", "Ocean Grace", "10", "Container Ship", "2020", "XX"],
                    ["2", "Vessel 12", "10", "Container Ship", "2020", "XX"],
                ],
            ),
            records=2,
        )
    ]
    by_year = {
        2020: [("Sea Star", "70"), ("One Year", "70"), ("Tugboat", "69")],
        2021: [("Sea Star", "70"), ("Twin Year", "79")],
        2022: [("Twin Year", "70")],
    }
    for year, rows in by_year.items():
        pins.append(
            _pin(
                root,
                source_id=f"noaa-{year}",
                family="noaa_pmel_ais",
                snapshot=str(year),
                relative=f"raw/noaa-{year}.csv",
                source_url=(
                    "https://data.pmel.noaa.gov/pmel/erddap/tabledap/"
                    f"AIS{year}_AIS.csv?VesselName,VesselType&distinct()"
                ),
                payload=_csv(
                    ["VesselName", "VesselType"],
                    [[name, vessel_type] for name, vessel_type in rows],
                ),
                records=len(rows),
            )
        )
    manifest = VesselNameRegistrySourceManifest(
        schema_version=1,
        snapshot_date="2026-09-01",
        files=tuple(pins),
        licenses=(
            VesselNameSourceLicense(
                family="imo_vessel_names",
                authority="Fixture IMO source",
                dataset="Fixture IMO names",
                license_name="PDDL 1.0",
                license_url="https://opendatacommons.org/licenses/pddl/1-0/",
                attribution="Fixture IMO source",
            ),
            VesselNameSourceLicense(
                family="noaa_pmel_ais",
                authority="Fixture NOAA source",
                dataset="Fixture NOAA AIS",
                license_name="CC0 1.0",
                license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                attribution="Fixture NOAA source",
            ),
        ),
        policy=VesselNameRegistryPolicy(),
    )
    path = root / "source-manifest.json"
    path.write_bytes(canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n")
    return path, sha256_file(path)


def _compile(tmp_path: Path) -> tuple[Path, LoadedVesselNameRegistry]:
    source_root = tmp_path / "sources"
    manifest, digest = _sources(source_root)
    result = compile_vessel_name_registry(
        source_root=source_root,
        source_manifest_path=manifest,
        expected_manifest_sha256=digest,
        output_parent=tmp_path / "output",
        run_name="fixture-registry",
    )
    registry_path = result.root / "vessel-names.jsonl"
    loaded = load_vessel_name_registry(
        registry_path,
        expected_sha256=result.receipt.registry_sha256,
        expected_records=result.receipt.registry_records,
    )
    return registry_path, loaded


def test_registry_uses_whole_curated_names_and_cross_year_noaa_evidence(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    manifest, digest = _sources(source_root)
    result = compile_vessel_name_registry(
        source_root=source_root,
        source_manifest_path=manifest,
        expected_manifest_sha256=digest,
        output_parent=tmp_path / "output",
        run_name="fixture-registry",
    )

    assert result.receipt.audit.model_dump() == {
        "source_rows": 8,
        "syntax_eligible_rows": 6,
        "source_rejection_counts": {
            "imo_vessel_names:contains_numeric_or_voyage_suffix": 1,
            "noaa_pmel_ais:outside_cargo_type": 1,
        },
        "distinct_syntax_eligible_names": 4,
        "accepted_registry_names": 3,
        "accepted_imo_names": 1,
        "accepted_noaa_names": 2,
        "accepted_noaa_only_names": 2,
        "quarantined_single_year_noaa_only_names": 1,
    }
    loaded = load_vessel_name_registry(
        result.root / "vessel-names.jsonl",
        expected_sha256=result.receipt.registry_sha256,
        expected_records=3,
    )
    assert [row.name for row in loaded.records] == ["OCEAN GRACE", "SEA STAR", "TWIN YEAR"]


def test_sampling_is_deterministic_unique_and_fail_closed(tmp_path: Path) -> None:
    registry_path, registry = _compile(tmp_path)
    stream = DeterministicStream(29, "vessel-registry-test", "sample")
    first = registry.sample(stream=stream)
    assert first == registry.sample(stream=stream)

    second = registry.sample(
        stream=stream.derive("second"),
        used_identity_keys={first.identity_key},
    )
    assert second.identity_key != first.identity_key
    with pytest.raises(VesselNameRegistryError, match="no unblocked identity"):
        registry.sample(
            stream=stream,
            excluded_identity_keys={row.identity_key for row in registry.records},
        )

    registry_path.write_bytes(registry_path.read_bytes() + b"\n")
    with pytest.raises(VesselNameRegistryError, match="size or SHA-256"):
        load_vessel_name_registry(
            registry_path,
            expected_sha256=registry.sha256,
            expected_records=3,
        )
