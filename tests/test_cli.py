from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

from document_ocr import cli
from document_ocr.benchmark import PublishedBenchmarkReport
from document_ocr.client import ServerInfo
from document_ocr.config import PipelineConfig
from document_ocr.hashing import canonical_json_bytes, canonical_json_sha256
from document_ocr.ledger import IncompleteRunError
from document_ocr.pipeline import PipelineResult
from document_ocr.sources import FrozenSourceInventory, SourceDiscoveryError
from document_ocr.vllm_contract import runtime_contract_payload, runtime_contract_sha256

MODEL_REVISION = "ca5d8b3e287e52589e37c28385d9655ee4372f9d"
CONTAINER_IMAGE = (
    "vllm/vllm-openai:v0.26.0@sha256:"
    "ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
)
BUILD_MANIFEST_SHA256 = "d" * 64


def _server_info() -> ServerInfo:
    payload = runtime_contract_payload(
        model="zai-org/GLM-OCR",
        served_model_name="glm-ocr",
        model_revision=MODEL_REVISION,
        max_model_len=32768,
        max_num_seqs=16,
        gpu_memory_utilization=0.9,
        generation_config="vllm",
        speculative_method="mtp",
        num_speculative_tokens=1,
        image_limit_per_prompt=1,
        container_base_image=CONTAINER_IMAGE,
        container_build_manifest_sha256=BUILD_MANIFEST_SHA256,
    )
    return ServerInfo(
        version="0.26.0",
        models=("glm-ocr",),
        model_repository="zai-org/GLM-OCR",
        served_model_name="glm-ocr",
        model_revision=MODEL_REVISION,
        max_model_len=32768,
        max_num_seqs=16,
        gpu_memory_utilization=0.9,
        generation_config="vllm",
        speculative_method="mtp",
        num_speculative_tokens=1,
        image_limit_per_prompt=1,
        container_base_image=CONTAINER_IMAGE,
        container_build_manifest_sha256=BUILD_MANIFEST_SHA256,
        contract_sha256=runtime_contract_sha256(payload),
    )


def _config(tmp_path: Path) -> PipelineConfig:
    example = Path(__file__).parents[1] / "configs" / "glm_ocr.local.example.yaml"
    values: Any = yaml.safe_load(example.read_text(encoding="utf-8"))
    assert isinstance(values, dict)
    values["source"]["root"] = str(tmp_path / "source")
    values["output"]["root"] = str(tmp_path / "output")
    return PipelineConfig.model_validate(values, strict=True)


def _install_config(
    monkeypatch: pytest.MonkeyPatch,
    config: PipelineConfig,
) -> None:
    def fake_load_config(path: str | Path) -> PipelineConfig:
        assert Path(path) == Path("pipeline.yaml")
        return config

    monkeypatch.setattr(cli, "load_config", fake_load_config)


def _expected_line(value: dict[str, Any]) -> str:
    return (canonical_json_bytes(value) + b"\n").decode("utf-8")


def test_validate_config_emits_only_non_secret_canonical_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    _install_config(monkeypatch, config)

    exit_code = cli.main(["validate-config", "--config", "pipeline.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == _expected_line(
        {
            "command": "validate-config",
            "config_sha256": canonical_json_sha256(config.model_dump(mode="json")),
            "run_id": config.run.run_id,
            "schema_version": 1,
            "source_type": "local",
            "status": "valid",
        }
    )


@pytest.mark.parametrize("command", ["inventory", "prepare-inventory"])
def test_inventory_aliases_prepare_the_same_frozen_inventory(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    _install_config(monkeypatch, config)
    inventory_path = tmp_path / "output" / "runs" / config.run.run_id / "inventory.jsonl"
    inventory = cast(
        FrozenSourceInventory,
        SimpleNamespace(path=inventory_path, sha256="a" * 64, objects=(object(), object())),
    )
    calls: list[PipelineConfig] = []

    async def fake_prepare_inventory(actual: PipelineConfig) -> FrozenSourceInventory:
        calls.append(actual)
        return inventory

    monkeypatch.setattr(cli, "prepare_inventory", fake_prepare_inventory)

    exit_code = cli.main([command, "--config", "pipeline.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [config]
    assert captured.err == ""
    assert captured.out == _expected_line(
        {
            "command": "prepare-inventory",
            "document_count": 2,
            "inventory_path": str(inventory_path),
            "inventory_sha256": "a" * 64,
            "run_id": config.run.run_id,
            "status": "ready",
        }
    )


def test_run_passes_resolved_project_root_and_reports_server_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    _install_config(monkeypatch, config)
    project_root = tmp_path / "project"
    project_root.mkdir()
    manifest = tmp_path / "manifest.json"
    summary = {"successful_pages": 7, "expected_pages": 7}
    result = PipelineResult(
        run_id=config.run.run_id,
        resumed_complete_run=False,
        summary=summary,
        dataset_manifest=manifest,
        server_info=_server_info(),
    )
    calls: list[tuple[Path, PipelineConfig]] = []

    async def fake_run_pipeline(*, project_root: Path, config: PipelineConfig) -> PipelineResult:
        calls.append((project_root, config))
        return result

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    exit_code = cli.main(
        [
            "run",
            "--config",
            "pipeline.yaml",
            "--project-root",
            str(project_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [(project_root.resolve(), config)]
    assert captured.err == ""
    assert captured.out == _expected_line(
        {
            "command": "run",
            "dataset_manifest": str(manifest),
            "resumed_complete_run": False,
            "run_id": config.run.run_id,
            "server": {
                "container_base_image": CONTAINER_IMAGE,
                "container_build_manifest_sha256": BUILD_MANIFEST_SHA256,
                "contract_sha256": _server_info().contract_sha256,
                "generation_config": "vllm",
                "gpu_memory_utilization": 0.9,
                "image_limit_per_prompt": 1,
                "max_model_len": 32768,
                "max_num_seqs": 16,
                "model_repository": "zai-org/GLM-OCR",
                "model_revision": MODEL_REVISION,
                "models": ["glm-ocr"],
                "num_speculative_tokens": 1,
                "served_model_name": "glm-ocr",
                "speculative_method": "mtp",
                "version": "0.26.0",
            },
            "status": "complete",
            "summary": summary,
        }
    )


def test_status_uses_only_the_read_only_completion_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    _install_config(monkeypatch, config)
    summary = {"successful_pages": 3, "expected_pages": 3}
    calls: list[PipelineConfig] = []

    async def fake_require_run_complete(actual: PipelineConfig) -> dict[str, int]:
        calls.append(actual)
        return summary

    async def forbidden_run_pipeline(**_: Any) -> PipelineResult:
        raise AssertionError("status must not run OCR")

    monkeypatch.setattr(cli, "require_run_complete", fake_require_run_complete)
    monkeypatch.setattr(cli, "run_pipeline", forbidden_run_pipeline)

    exit_code = cli.main(["status", "--config", "pipeline.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [config]
    assert captured.err == ""
    assert captured.out == _expected_line(
        {
            "command": "status",
            "complete": True,
            "run_id": config.run.run_id,
            "status": "complete",
            "summary": summary,
        }
    )


def test_benchmark_command_delegates_without_starting_a_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    _install_config(monkeypatch, config)
    raster_manifest = tmp_path / "rasters.jsonl"
    report_path = tmp_path / "benchmark.json"
    published = PublishedBenchmarkReport(
        path=report_path,
        sha256="a" * 64,
        created=True,
        report={"benchmark_identity_sha256": "b" * 64},
    )
    calls: list[tuple[PipelineConfig, Path, Path]] = []

    async def fake_run_benchmark(
        *,
        config: PipelineConfig,
        raster_manifest_path: Path,
        report_path: Path,
    ) -> PublishedBenchmarkReport:
        calls.append((config, raster_manifest_path, report_path))
        return published

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    exit_code = cli.main(
        [
            "benchmark",
            "--config",
            "pipeline.yaml",
            "--raster-manifest",
            str(raster_manifest),
            "--report",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [(config, raster_manifest, report_path)]
    assert captured.err == ""
    assert captured.out == _expected_line(
        {
            "benchmark_identity_sha256": "b" * 64,
            "command": "benchmark",
            "created": True,
            "report_path": str(report_path),
            "report_sha256": "a" * 64,
            "status": "complete",
        }
    )


def test_usage_error_has_stable_exit_and_diagnostic_stream(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(["status"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == _expected_line(
        {
            "command": None,
            "error_type": "_UsageError",
            "message": "the following arguments are required: --config",
            "status": "error",
        }
    )


def test_config_failure_has_distinct_exit_without_echoing_invalid_contents(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sensitive-config-token")

    def failed_load_config(_: str | Path) -> PipelineConfig:
        raise ValueError("invalid configuration sensitive-config-token")

    monkeypatch.setattr(cli, "load_config", failed_load_config)

    exit_code = cli.main(["validate-config", "--config", "pipeline.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert captured.out == ""
    assert "sensitive-config-token" not in captured.err
    assert captured.err == _expected_line(
        {
            "command": "validate-config",
            "error_type": "ValueError",
            "message": "configuration could not be loaded or validated: pipeline.yaml",
            "status": "error",
        }
    )


def test_inventory_operation_failure_has_stable_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    _install_config(monkeypatch, config)
    monkeypatch.setenv(config.vllm.api_key_env, "sensitive-operation-token")

    async def failed_prepare_inventory(_: PipelineConfig) -> FrozenSourceInventory:
        raise SourceDiscoveryError("no matching PDFs sensitive-operation-token")

    monkeypatch.setattr(cli, "prepare_inventory", failed_prepare_inventory)

    exit_code = cli.main(["inventory", "--config", "pipeline.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 4
    assert captured.out == ""
    assert "sensitive-operation-token" not in captured.err
    assert captured.err == _expected_line(
        {
            "command": "inventory",
            "error_type": "SourceDiscoveryError",
            "message": "no matching PDFs [REDACTED]",
            "status": "error",
        }
    )


def test_incomplete_status_has_distinct_exit_without_running_ocr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _config(tmp_path)
    _install_config(monkeypatch, config)

    async def incomplete(_: PipelineConfig) -> dict[str, int]:
        raise IncompleteRunError("run is incomplete")

    monkeypatch.setattr(cli, "require_run_complete", incomplete)

    exit_code = cli.main(["status", "--config", "pipeline.yaml"])

    captured = capsys.readouterr()
    assert exit_code == 5
    assert captured.out == ""
    assert captured.err == _expected_line(
        {
            "command": "status",
            "error_type": "IncompleteRunError",
            "message": "run is incomplete",
            "status": "error",
        }
    )
