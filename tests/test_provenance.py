from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from document_ocr.config import PipelineConfig
from document_ocr.hashing import canonical_json_sha256
from document_ocr.provenance import collect_runtime_provenance

MODEL_REVISION = "c" * 40


def _config_data() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": {
            "type": "local",
            "root": "/data/pdfs",
            "include_glob": "**/*.pdf",
            "dataset_version": "snapshot-v1",
            "require_content_sha256": True,
        },
        "output": {
            "root": "/data/output",
            "retain_page_images": False,
            "parquet_compression": "zstd",
            "write_batch_rows": 64,
        },
        "run": {"run_id": "run-1", "fail_fast": False, "resume": True},
        "raster": {
            "dpi": 200,
            "max_side_pixels": 4096,
            "max_pixels": 12_000_000,
            "max_pages_per_document": 500,
            "max_pdf_bytes": 1_000_000_000,
            "max_cached_documents_per_process": 2,
            "image_format": "png",
            "jpeg_quality": None,
            "draw_annotations": True,
            "reject_xfa": True,
        },
        "vllm": {
            "endpoint": "http://127.0.0.1:8000",
            "model": "zai-org/GLM-OCR",
            "served_model_name": "glm-ocr",
            "revision": MODEL_REVISION,
            "engine_version": "0.26.0",
            "container_base_image": (
                "vllm/vllm-openai:v0.26.0@sha256:"
                "ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
            ),
            "container_build_manifest_sha256": "d" * 64,
            "max_model_len": 32768,
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.9,
            "prompt": "Text Recognition:",
            "request_timeout_seconds": 120.0,
            "api_key_env": "VLLM_API_KEY",
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "repetition_penalty": 1.0,
                "max_tokens": 8192,
                "seed": 0,
            },
            "retry": {
                "max_attempts": 4,
                "initial_backoff_seconds": 0.5,
                "max_backoff_seconds": 8.0,
                "backoff_multiplier": 2.0,
                "jitter_fraction": 0.0,
            },
            "speculative_decoding": {"method": "mtp", "num_speculative_tokens": 1},
        },
        "concurrency": {
            "max_active_documents": 2,
            "renderer_processes": 2,
            "max_inflight_pages_global": 8,
            "max_inflight_pages_per_document": 2,
        },
    }


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    source = root / "src" / "document_ocr"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    (source / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'pipeline'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    return root


@pytest.fixture
def deterministic_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "document_ocr.provenance._installed_versions",
        lambda: {"document-ocr-pipeline": "0.1.0", "pyarrow": "25.0.0"},
    )
    monkeypatch.setattr("document_ocr.provenance.platform.platform", lambda: "test-platform")
    monkeypatch.setattr("document_ocr.provenance.sys.version", "3.12.test")


def test_runtime_provenance_and_fingerprint_are_stable_and_self_verifying(
    tmp_path: Path, deterministic_runtime: None
) -> None:
    root = _project(tmp_path)
    config = PipelineConfig.model_validate(_config_data(), strict=True)

    first, first_fingerprint = collect_runtime_provenance(root, config)
    second, second_fingerprint = collect_runtime_provenance(root, config)

    assert first == second
    assert first_fingerprint == second_fingerprint
    assert len(first_fingerprint) == 64
    assert first["resolved_config_sha256"] == canonical_json_sha256(first["resolved_config"])
    assert first["extraction_contract"]["model"]["revision"] == MODEL_REVISION
    assert first["extraction_contract"]["model"]["container_base_image"].endswith(
        "@sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
    )
    assert first["extraction_contract"]["model"]["container_build_manifest_sha256"] == ("d" * 64)
    assert first["extraction_contract"]["model"]["speculative_decoding"] == {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }
    assert first["git"] == {
        "state": "not-a-git-worktree",
        "commit": None,
        "dirty": None,
        "status_sha256": None,
    }


def test_operational_config_drift_is_recorded_but_does_not_change_extraction_identity(
    tmp_path: Path, deterministic_runtime: None
) -> None:
    root = _project(tmp_path)
    first_data = _config_data()
    second_data = _config_data()
    second_data["concurrency"]["max_inflight_pages_global"] = 16
    second_data["run"]["run_id"] = "run-2"

    first, first_fingerprint = collect_runtime_provenance(
        root, PipelineConfig.model_validate(first_data, strict=True)
    )
    second, second_fingerprint = collect_runtime_provenance(
        root, PipelineConfig.model_validate(second_data, strict=True)
    )

    assert first_fingerprint == second_fingerprint
    assert first["resolved_config_sha256"] != second["resolved_config_sha256"]
    assert first["extraction_contract"] == second["extraction_contract"]


@pytest.mark.parametrize(
    "drift", ["raster", "model_revision", "source_tree", "pyproject", "lock", "compose"]
)
def test_extraction_or_executable_drift_changes_pipeline_fingerprint(
    tmp_path: Path, deterministic_runtime: None, drift: str
) -> None:
    root = _project(tmp_path)
    data = _config_data()
    config = PipelineConfig.model_validate(data, strict=True)
    _, baseline = collect_runtime_provenance(root, config)

    if drift == "raster":
        data["raster"]["dpi"] = 201
        config = PipelineConfig.model_validate(data, strict=True)
    elif drift == "model_revision":
        data["vllm"]["revision"] = "d" * 40
        config = PipelineConfig.model_validate(data, strict=True)
    elif drift == "source_tree":
        (root / "src" / "document_ocr" / "worker.py").write_text("VALUE = 2\n", encoding="utf-8")
    elif drift == "pyproject":
        (root / "pyproject.toml").write_text(
            "[project]\nname = 'changed-pipeline'\n", encoding="utf-8"
        )
    elif drift == "lock":
        (root / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    else:
        (root / "compose.yaml").write_text("services: {ocr: {}}\n", encoding="utf-8")

    _, changed = collect_runtime_provenance(root, config)

    assert changed != baseline


def test_installed_runtime_package_drift_changes_pipeline_fingerprint(
    tmp_path: Path,
    deterministic_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    config = PipelineConfig.model_validate(_config_data(), strict=True)
    _, baseline = collect_runtime_provenance(root, config)

    monkeypatch.setattr(
        "document_ocr.provenance._installed_versions",
        lambda: {"document-ocr-pipeline": "0.1.0", "pyarrow": "25.0.1"},
    )
    _, changed = collect_runtime_provenance(root, config)

    assert changed != baseline


def test_source_tree_hash_is_path_sensitive_and_ignores_non_python_files(
    tmp_path: Path, deterministic_runtime: None
) -> None:
    first_root = _project(tmp_path / "first")
    second_root = _project(tmp_path / "second")
    original = second_root / "src" / "document_ocr" / "worker.py"
    original.rename(original.with_name("renamed.py"))
    config = PipelineConfig.model_validate(_config_data(), strict=True)

    _, first = collect_runtime_provenance(first_root, config)
    _, renamed = collect_runtime_provenance(second_root, config)
    (first_root / "src" / "document_ocr" / "README.txt").write_text(
        "not executable", encoding="utf-8"
    )
    _, non_python_added = collect_runtime_provenance(first_root, config)

    assert renamed != first
    assert non_python_added == first


@pytest.mark.parametrize("required_name", ["pyproject.toml", "uv.lock", "compose.yaml"])
def test_missing_reproducibility_artifact_fails_closed(
    tmp_path: Path, deterministic_runtime: None, required_name: str
) -> None:
    root = _project(tmp_path)
    (root / required_name).unlink()
    config = PipelineConfig.model_validate(_config_data(), strict=True)

    with pytest.raises(FileNotFoundError, match="required reproducibility artifact"):
        collect_runtime_provenance(root, config)
