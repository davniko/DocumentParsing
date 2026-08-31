from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from document_ocr.synthesis import cli
from document_ocr.synthesis.config import SynthesisRouteScenarioPilotConfig


def _route_config_value(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "task": "bill_of_lading_relation_explicit_v3",
        "run": {
            "run_id": "route-scenario-pilot-test-v1",
            "output_dir": str(tmp_path / "output"),
        },
        "source": {
            "format": "jsonl",
            "file": {
                "path": str(tmp_path / "source.jsonl"),
                "sha256": "a" * 64,
                "records": 10,
            },
            "fields": {
                "document_id": "documentId",
                "input_text": "joinedRawText",
                "target": "target",
                "input_sha256": "joinedRawTextSha256",
            },
        },
        "task_constraints": {
            "path": str(tmp_path / "task-constraints.json"),
            "sha256": "b" * 64,
        },
        "inputs": {
            "upstream_selection": {
                "path": str(tmp_path / "structured-selection.jsonl"),
                "sha256": "9" * 64,
                "records": 4,
            },
            "template_groups": {
                "path": str(tmp_path / "template-groups.jsonl"),
                "sha256": "c" * 64,
                "records": 8,
            },
            "partition_report": {
                "path": str(tmp_path / "partition.json"),
                "sha256": "d" * 64,
            },
            "iso3166_snapshot": {
                "path": str(tmp_path / "iso3166.json"),
                "sha256": "e" * 64,
            },
            "route_locations": {
                "path": str(tmp_path / "route-locations.jsonl"),
                "sha256": "1" * 64,
                "records": 100,
            },
            "route_registry_manifest": {
                "path": str(tmp_path / "route-registry-manifest.json"),
                "sha256": "2" * 64,
            },
            "world_ports": {
                "path": str(tmp_path / "world-ports.jsonl"),
                "sha256": "5" * 64,
                "records": 80,
            },
            "world_port_registry_manifest": {
                "path": str(tmp_path / "world-port-registry-manifest.json"),
                "sha256": "6" * 64,
            },
            "locality_registry": {
                "path": str(tmp_path / "locality-registry"),
                "manifest_sha256": "7" * 64,
            },
            "trade_flows": {
                "path": str(tmp_path / "trade-flows.jsonl"),
                "sha256": "3" * 64,
                "records": 500,
            },
            "trade_flow_registry_manifest": {
                "path": str(tmp_path / "trade-flow-registry-manifest.json"),
                "sha256": "4" * 64,
            },
        },
        "selection": {
            "split": "train",
            "requested_documents": 4,
            "require_template_wholly_in_split": True,
            "maximum_per_template": 1,
        },
        "generation": {
            "random_stream": "hmac_sha256_counter_v1",
            "seed": 20260831,
            "commercial_origin_prior": {
                "method": "observed_exporter_plus_maritime_registry_mixture_v1",
                "observed_exporter": {
                    "mixture_permyriad": 7_500,
                    "weighting": "train_isolated_shipper_country_document_count_v1",
                },
                "maritime_registry": {
                    "mixture_permyriad": 2_500,
                    "weighting": "uniform_route_feasible_iso_country_v1",
                },
            },
            "commercial_destination_prior": (
                "wits_latest_bilateral_numeric_iso_identity_conditioned_on_origin_v2"
            ),
            "physical_endpoint_relation_method": (
                "train_empirical_loading_by_origin_discharge_by_destination_else_"
                "commercial_destination_v1"
            ),
            "party_locality_relation_method": (
                "train_empirical_role_relation_and_locality_mode_v1"
            ),
            "freight_method": "train_empirical_arrangement_and_payment_side_v1",
            "port_method": "observed_empirical_plus_nga_wpi_whitelist_mixture_v1",
            "locality_method": "geonames_cities15000_population_weighted_v1",
            "registry_exploration_permyriad": 1_000,
            "preserve_source_leaf_presence": True,
            "preserve_source_party_cardinality": True,
            "direct_routes_only": True,
            "publish_training_records": False,
        },
    }


def _write_config(tmp_path: Path, value: dict[str, Any]) -> Path:
    path = tmp_path / "route-scenario.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "input_name",
    ("world_ports", "world_port_registry_manifest", "locality_registry"),
)
def test_route_scenario_config_requires_every_pinned_geography_input(
    tmp_path: Path, input_name: str
) -> None:
    value = _route_config_value(tmp_path)
    del value["inputs"][input_name]

    with pytest.raises(ValueError, match=input_name):
        SynthesisRouteScenarioPilotConfig.model_validate(value, strict=True)


def test_route_scenario_config_rejects_the_old_all_unlocode_port_method(
    tmp_path: Path,
) -> None:
    value = _route_config_value(tmp_path)
    value["generation"]["port_method"] = "observed_empirical_plus_unlocode_registry_mixture_v1"

    with pytest.raises(ValueError, match="port_method"):
        SynthesisRouteScenarioPilotConfig.model_validate(value, strict=True)


def test_validate_route_scenario_pilot_config_emits_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path, _route_config_value(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "document-kie-synthesis",
            "validate-route-scenario-pilot-config",
            "--config",
            str(config_path),
            "--project-root",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "command": "validate-route-scenario-pilot-config",
        "direct_routes_only": True,
        "requested_documents": 4,
        "run_id": "route-scenario-pilot-test-v1",
        "status": "valid",
    }


def test_validate_controlled_pilot_config_reports_separate_transport_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "controlled.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    controlled = SimpleNamespace(
        run=SimpleNamespace(run_id="controlled-test-v1"),
        selection=SimpleNamespace(requested_documents=4),
        generation=SimpleNamespace(
            transport=SimpleNamespace(
                vessel_name_method="deferred_by_explicit_scope_v1",
                voyage_number_method="observed_character_class_shape_v1",
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "load_synthesis_controlled_pilot_config",
        lambda _path: controlled,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "document-kie-synthesis",
            "validate-controlled-pilot-config",
            "--config",
            str(config_path),
            "--project-root",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "command": "validate-controlled-pilot-config",
        "requested_documents": 4,
        "run_id": "controlled-test-v1",
        "status": "valid",
        "vessel_name_method": "deferred_by_explicit_scope_v1",
        "voyage_number_method": "observed_character_class_shape_v1",
    }


def test_run_route_scenario_pilot_delegates_resolved_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = _route_config_value(tmp_path)
    expected_config = SynthesisRouteScenarioPilotConfig.model_validate(value, strict=True)
    config_path = _write_config(tmp_path, value)
    project_root = tmp_path / "project"
    project_root.mkdir()
    calls: list[tuple[Path, Path, SynthesisRouteScenarioPilotConfig]] = []

    def fake_run_route_scenario_pilot(
        *,
        project_root: Path,
        config_path: Path,
        config: SynthesisRouteScenarioPilotConfig,
    ) -> dict[str, Any]:
        calls.append((project_root, config_path, config))
        return {"runId": config.run.run_id, "scenarios": 4}

    from document_ocr.synthesis import route_scenario_pipeline

    monkeypatch.setattr(
        route_scenario_pipeline,
        "run_route_scenario_pilot",
        fake_run_route_scenario_pilot,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "document-kie-synthesis",
            "run-route-scenario-pilot",
            "--config",
            str(config_path),
            "--project-root",
            str(project_root),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert calls == [(project_root.resolve(), config_path.resolve(), expected_config)]
    assert json.loads(captured.out) == {
        "command": "run-route-scenario-pilot",
        "result": {"runId": "route-scenario-pilot-test-v1", "scenarios": 4},
        "status": "complete",
    }


def test_route_scenario_config_error_is_reported_without_running_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = _route_config_value(tmp_path)
    value["generation"]["direct_routes_only"] = False
    config_path = _write_config(tmp_path, value)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "document-kie-synthesis",
            "run-route-scenario-pilot",
            "--config",
            str(config_path),
            "--project-root",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 4
    captured = capsys.readouterr()
    assert captured.err == ""
    error = json.loads(captured.out)
    assert error["status"] == "error"
    assert error["error_type"] == "ValidationError"
    assert "direct_routes_only" in error["diagnostic"]
