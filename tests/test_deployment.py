from __future__ import annotations

import hashlib
import json
import re
import tomllib
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import yaml

from document_ocr.config import (
    PipelineConfig,
    load_catalog_config,
    load_config,
    load_corpus_config,
    load_pilot_config,
    load_snapshot_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = PROJECT_ROOT / "compose.yaml"
VLLM_DOCKERFILE_PATH = PROJECT_ROOT / "docker" / "vllm" / "Dockerfile"
VLLM_BUILD_MANIFEST_PATH = PROJECT_ROOT / "docker" / "vllm" / "build-manifest.json"
VLLM_PATCH_PATH = (
    PROJECT_ROOT / "docker" / "vllm" / "patches" / "49869-glm-ocr-mtp-weight-prefix.patch"
)
EXAMPLE_CONFIG_PATHS = (
    PROJECT_ROOT / "configs" / "glm_ocr.local.example.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.s3.example.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.blc.local.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.blc150.local.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.swb.local.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.awbc.local.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.coo.local.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.inv.local.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.pl.local.yaml",
)


def test_runtime_dependencies_enable_aws_login_credentials() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert any(dependency.startswith("boto3[crt]") for dependency in dependencies)


def test_operator_entrypoints_and_snapshot_config_are_installed() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"] == {
        "document-ocr": "document_ocr.cli:main",
        "document-ocr-classification-catalog": ("document_ocr.classification_catalog_cli:main"),
        "document-ocr-corpus": "document_ocr.corpus_cli:main",
        "document-ocr-pilot": "document_ocr.pilot_cli:main",
        "document-ocr-snapshot": "document_ocr.snapshot_cli:main",
    }
    snapshot = load_snapshot_config(PROJECT_ROOT / "configs" / "s3_snapshot.blc_swb.yaml")
    assert snapshot.document_types == ["blc", "swb"]
    assert snapshot.classification_label_mapping == {"blc": "blc", "swb": "swb"}
    assert len(snapshot.page_count_quarantine) == 1
    corpus = load_corpus_config(PROJECT_ROOT / "configs" / "corpus.blc.local.yaml")
    assert corpus.document_type == "blc"
    assert len(corpus.sources) == 3
    catalog = load_catalog_config(PROJECT_ROOT / "configs" / "classification_catalog.yaml")
    assert [item.name for item in catalog.lineages] == ["old", "new_existing", "new_aci"]
    assert len(catalog.raw_sources) == 4
    assert len(catalog.local_snapshots) == 3
    pilot = load_pilot_config(PROJECT_ROOT / "configs" / "pilot.blc150.yaml")
    assert pilot.document_count == 150
    assert sum(item.documents for item in pilot.strata) == 150


def _load_compose_service() -> dict[str, Any]:
    document: Any = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    services = document.get("services")
    assert isinstance(services, dict)
    assert set(services) == {"glm-ocr-vllm"}
    service = services["glm-ocr-vllm"]
    assert isinstance(service, dict)
    return service


def _command(service: dict[str, Any]) -> list[str]:
    command = service.get("command")
    assert isinstance(command, list)
    assert all(isinstance(item, str) for item in command)
    return command


def _single_option(command: list[str], option: str) -> str:
    positions = [index for index, item in enumerate(command) if item == option]
    assert len(positions) == 1, f"expected one {option} option"
    position = positions[0]
    assert position + 1 < len(command), f"{option} has no value"
    value = command[position + 1]
    assert not value.startswith("--"), f"{option} has no value"
    return value


@pytest.mark.parametrize(
    "config_path",
    EXAMPLE_CONFIG_PATHS,
    ids=(
        "local-example",
        "s3-example",
        "blc-local",
        "blc150-local",
        "swb-local",
        "awbc-local",
        "coo-local",
        "inv-local",
        "pl-local",
    ),
)
def test_example_config_matches_pinned_vllm_compose_contract(config_path: Path) -> None:
    config: PipelineConfig = load_config(config_path)
    service = _load_compose_service()
    command = _command(service)

    manifest_bytes = VLLM_BUILD_MANIFEST_PATH.read_bytes()
    manifest: Any = json.loads(manifest_bytes)
    assert isinstance(manifest, dict)
    assert hashlib.sha256(manifest_bytes).hexdigest() == (
        config.vllm.container_build_manifest_sha256
    )
    assert manifest["base_image"] == config.vllm.container_base_image
    assert service["image"] == manifest["image"]
    assert service["build"] == {
        "context": ".",
        "dockerfile": "docker/vllm/Dockerfile",
    }
    assert command[0] == config.vllm.model
    assert "--model" not in command
    assert _single_option(command, "--revision") == config.vllm.revision
    assert _single_option(command, "--served-model-name") == config.vllm.served_model_name
    assert _single_option(command, "--generation-config") == "vllm"
    assert _single_option(command, "--middleware") == (
        "document_ocr.vllm_contract.runtime_contract_middleware"
    )

    speculative = config.vllm.speculative_decoding
    assert _single_option(command, "--speculative-config.method") == speculative.method
    assert int(_single_option(command, "--speculative-config.num_speculative_tokens")) == (
        speculative.num_speculative_tokens
    )
    assert json.loads(_single_option(command, "--limit-mm-per-prompt")) == {"image": 1}

    assert int(_single_option(command, "--max-model-len")) == config.vllm.max_model_len
    assert int(_single_option(command, "--max-num-seqs")) == config.vllm.max_num_seqs
    assert (
        float(_single_option(command, "--gpu-memory-utilization"))
        == config.vllm.gpu_memory_utilization
    )

    endpoint = urlsplit(config.vllm.endpoint)
    assert endpoint.hostname is not None
    assert ip_address(endpoint.hostname).is_loopback
    assert endpoint.port is not None
    assert service["ports"] == [
        f"{endpoint.hostname}:${{VLLM_PORT:-{endpoint.port}}}:{endpoint.port}"
    ]

    environment = service.get("environment")
    assert isinstance(environment, dict)
    assert environment["PYTHONPATH"] == "/opt/document-ocr-src"
    assert environment["VLLM_WSL2_ENABLE_PIN_MEMORY"] == "1"
    assert environment["DOCUMENT_OCR_VLLM_BUILD_MANIFEST"] == (
        "/opt/document-ocr/vllm-build-manifest.json"
    )
    assert "DOCUMENT_OCR_VLLM_IMAGE" not in environment
    assert "VLLM_SERVER_DEV_MODE" not in environment
    volumes = service.get("volumes")
    assert isinstance(volumes, list)
    assert "./src:/opt/document-ocr-src:ro" in volumes

    api_key_reference = _single_option(command, "--api-key")
    required_interpolation = re.fullmatch(
        r"\$\{([A-Za-z_][A-Za-z0-9_]*):\?(.+)\}", api_key_reference
    )
    assert required_interpolation is not None
    assert required_interpolation.group(1) == config.vllm.api_key_env
    assert required_interpolation.group(2).strip()


def test_vllm_patch_build_inputs_are_content_addressed() -> None:
    manifest_bytes = VLLM_BUILD_MANIFEST_PATH.read_bytes()
    manifest: Any = json.loads(manifest_bytes)
    assert isinstance(manifest, dict)
    patch_bytes = VLLM_PATCH_PATH.read_bytes()
    dockerfile = VLLM_DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert manifest == {
        "schema_version": 1,
        "image": "document-ocr/vllm-openai:v0.26.0-glm-ocr-mtp-89e3c3f",
        "base_image": (
            "vllm/vllm-openai:v0.26.0@sha256:"
            "ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
        ),
        "vllm_version": "0.26.0",
        "patch_source": (
            "https://github.com/vllm-project/vllm/commit/89e3c3f5b41d0f678d19a138dba59b3757a1a16f"
        ),
        "patch_commit": "89e3c3f5b41d0f678d19a138dba59b3757a1a16f",
        "patch_sha256": "0923d3e1975634a7e2b639119a8d78a190a991c5635fba7b8e2df7d17b5e3993",
        "target_path": (
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py"
        ),
        "target_before_sha256": (
            "f06d1a1a8d92e6ab39ebf7931d3eb3d6688d212606b1c409d7bed8eed5ba1fc2"
        ),
        "target_after_sha256": ("f8d69922ac178e4e8dc8083bef8714234005254785c2a3645ed74b3dfe521733"),
    }
    assert hashlib.sha256(manifest_bytes).hexdigest() == (
        "02f79af13ea66b4c63cf1609abc9287a7757efe9b7dd03ddeeb0d73ed041352d"
    )
    assert hashlib.sha256(patch_bytes).hexdigest() == manifest["patch_sha256"]
    assert dockerfile.startswith(
        "# syntax=docker/dockerfile:1.7@sha256:"
        "a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e\n"
    )
    assert f"FROM {manifest['base_image']}" in dockerfile
    assert manifest["patch_commit"] in dockerfile
    assert manifest["patch_sha256"] in dockerfile
    assert "RUN PYTHONDONTWRITEBYTECODE=1 python3" in dockerfile
    assert "model.language_model.layers.{base + i}." in patch_bytes.decode("utf-8")
