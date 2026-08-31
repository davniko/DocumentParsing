from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from document_ocr.synthesis.locality_registry import GeoNamesLocalityReceipt
from document_ocr.synthesis.route_registry import UnlocodeRegistryReceipt
from document_ocr.synthesis.route_scenario_pipeline import (
    RouteScenarioPipelineError,
    _publish_runtime_once,
    _validate_locality_dependency,
    _validate_world_port_dependency,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.world_port_registry import WorldPortRegistryReceipt


def _world_port_dependency_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], SimpleNamespace]:
    unlocode_root = tmp_path / "unlocode-test-v1"
    unlocode_root.mkdir()
    unlocode_locations = unlocode_root / "locations.jsonl"
    unlocode_locations.write_bytes(b"abc")
    unlocode_receipt_path = unlocode_root / "registry-receipt.json"
    unlocode_receipt_path.write_bytes(b"{}")

    world_port_root = tmp_path / "world-ports"
    world_port_root.mkdir()
    whitelist_path = world_port_root / "port-whitelist.jsonl"
    whitelist_path.write_bytes(b"{}\n")
    receipt_path = world_port_root / "registry-receipt.json"
    receipt_path.write_bytes(b"{}")

    receipt = SimpleNamespace(
        artifacts=(
            SimpleNamespace(
                path="port-whitelist.jsonl",
                sha256="a" * 64,
                records=4,
                bytes=3,
            ),
        ),
        unlocode_artifact=SimpleNamespace(
            sha256="b" * 64,
            records=3,
            bytes=3,
            release="2025-1",
            run_name="unlocode-test-v1",
            relative_path="unlocode-test-v1/locations.jsonl",
        ),
    )
    values: dict[str, object] = {
        "receipt": cast(WorldPortRegistryReceipt, receipt),
        "receipt_path": receipt_path,
        "whitelist_path": whitelist_path,
        "expected_whitelist_sha256": "a" * 64,
        "expected_whitelist_records": 4,
        "unlocode_receipt": cast(
            UnlocodeRegistryReceipt,
            SimpleNamespace(source=SimpleNamespace(release="2025-1")),
        ),
        "unlocode_receipt_path": unlocode_receipt_path,
        "unlocode_locations_path": unlocode_locations,
        "expected_unlocode_sha256": "b" * 64,
        "expected_unlocode_records": 3,
    }
    return values, receipt


def test_world_port_dependency_accepts_only_the_exact_unlocode_build(tmp_path: Path) -> None:
    values, _receipt = _world_port_dependency_fixture(tmp_path)

    _validate_world_port_dependency(**values)  # type: ignore[arg-type]


def test_world_port_dependency_rejects_whitelist_byte_size_drift(tmp_path: Path) -> None:
    values, receipt = _world_port_dependency_fixture(tmp_path)
    receipt.artifacts[0].bytes = 99

    with pytest.raises(ValueError, match="differs from its receipt"):
        _validate_world_port_dependency(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sha256", "c" * 64),
        ("records", 2),
        ("bytes", 4),
        ("release", "2024-2"),
        ("run_name", "different-run"),
        ("relative_path", "different-run/locations.jsonl"),
    ),
)
def test_world_port_dependency_rejects_every_unlocode_or_artifact_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    values, receipt = _world_port_dependency_fixture(tmp_path)
    dependency = receipt.unlocode_artifact
    if field in {"sha256", "records", "bytes", "release", "run_name", "relative_path"}:
        setattr(dependency, field, value)

    with pytest.raises(ValueError, match="differs from its receipt"):
        _validate_world_port_dependency(**values)  # type: ignore[arg-type]


def test_locality_dependency_requires_the_same_iso_snapshot() -> None:
    receipt = cast(
        GeoNamesLocalityReceipt,
        SimpleNamespace(iso3166_sha256="d" * 64),
    )
    _validate_locality_dependency(
        receipt=receipt,
        expected_iso3166_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="ISO-3166 pin differs"):
        _validate_locality_dependency(
            receipt=receipt,
            expected_iso3166_sha256="e" * 64,
        )


def test_runtime_receipt_is_preserved_on_exact_committed_resume(tmp_path: Path) -> None:
    transaction = "f" * 64
    stage = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="route-runtime",
        transaction_sha256=transaction,
    )
    first = _publish_runtime_once(
        stage,
        {"elapsedSeconds": 4.5, "peakRssMiB": 443.0},
    )
    stage.commit(expected_artifacts=("runtime.json",), metadata={})

    resumed = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="route-runtime",
        transaction_sha256=transaction,
    )
    second = _publish_runtime_once(
        resumed,
        {"elapsedSeconds": 0.1, "peakRssMiB": 1.0},
    )
    commit = resumed.commit(expected_artifacts=("runtime.json",), metadata={})

    assert first == {"elapsedSeconds": 4.5, "peakRssMiB": 443.0}
    assert second == first
    assert commit.created is False


def test_runtime_receipt_rejects_a_malformed_interrupted_observation(tmp_path: Path) -> None:
    stage = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="route-runtime",
        transaction_sha256="e" * 64,
    )
    stage.publish_json("runtime.json", {"elapsedSeconds": "fast", "peakRssMiB": 1.0})

    with pytest.raises(RouteScenarioPipelineError, match="elapsedSeconds is not numeric"):
        _publish_runtime_once(
            stage,
            {"elapsedSeconds": 0.1, "peakRssMiB": 1.0},
        )
