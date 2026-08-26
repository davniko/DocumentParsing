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
TRAINING_DOCKERFILE_PATH = PROJECT_ROOT / "docker" / "training" / "Dockerfile"
TRAINING_VERIFY_PATH = PROJECT_ROOT / "docker" / "training" / "verify_environment.py"
VLLM_BUILD_MANIFEST_PATH = PROJECT_ROOT / "docker" / "vllm" / "build-manifest.json"
FOLLOWUP_WATCHER_PATH = PROJECT_ROOT / "tools" / "run_blc500_after_training.sh"
VLLM_PATCH_PATHS = (
    PROJECT_ROOT / "docker" / "vllm" / "patches" / "49869-glm-ocr-mtp-weight-prefix.patch",
    PROJECT_ROOT / "docker" / "vllm" / "patches" / "51966-glm-ocr-mtp-cudagraph.patch",
)
EXAMPLE_CONFIG_PATHS = (
    PROJECT_ROOT / "configs" / "glm_ocr.local.example.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.s3.example.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.blc.local.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.blc150.local.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.blc500-followup.local.yaml",
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
        "document-ocr-catalog-selection": "document_ocr.catalog_selection_cli:main",
        "document-ocr-corpus": "document_ocr.corpus_cli:main",
        "document-ocr-pilot": "document_ocr.pilot_cli:main",
        "document-ocr-quality-filter": "document_ocr.quality_cli:main",
        "document-ocr-snapshot": "document_ocr.snapshot_cli:main",
        "document-ocr-table-view": "document_ocr.table_views.cli:main",
        "document-kie-label-agents": "document_ocr.labeling_agents.cli:main",
        "document-kie-label-source": "document_ocr.labeling_agents.source_cli:main",
        "document-kie-semantic-v3": "document_ocr.semantic_v3.cli:main",
        "document-kie-train": "document_ocr.training.cli:main",
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
    followup = load_pilot_config(PROJECT_ROOT / "configs" / "pilot.blc500-followup.yaml")
    assert followup.schema_version == 2
    assert followup.document_count == 500
    assert len(followup.excluded_pilots) == 1
    assert sum(item.documents for item in followup.strata) == 500


def test_followup_watcher_reacquires_cwd_and_persists_redacted_vllm_logs() -> None:
    watcher = FOLLOWUP_WATCHER_PATH.read_text(encoding="utf-8")

    assert FOLLOWUP_WATCHER_PATH.stat().st_mode & 0o111
    assert 'uv run document-ocr run' not in watcher
    assert '"${ocr_entrypoint}" run' in watcher
    assert 'cd /\ncd -- "${project_root}"' in watcher
    assert "docker compose logs" in watcher
    assert '--follow \\\n' in watcher
    assert 'line.replace(secret, "[REDACTED]")' in watcher
    assert 'followup500-${timestamp}.vllm.log' in watcher


def _load_compose_service() -> dict[str, Any]:
    document: Any = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    services = document.get("services")
    assert isinstance(services, dict)
    assert set(services) == {
        "glm-ocr-vllm",
        "kie-tools",
        "kie-trainer",
        "mlflow-server",
    }
    service = services["glm-ocr-vllm"]
    assert isinstance(service, dict)
    return service


def _load_training_compose_service() -> dict[str, Any]:
    document: Any = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    service = document["services"]["kie-trainer"]
    assert isinstance(service, dict)
    return service


def _load_training_tools_compose_service() -> dict[str, Any]:
    document: Any = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    service = document["services"]["kie-tools"]
    assert isinstance(service, dict)
    return service


def _load_mlflow_compose_service() -> dict[str, Any]:
    document: Any = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    service = document["services"]["mlflow-server"]
    assert isinstance(service, dict)
    return service


def test_training_container_is_explicit_safe_and_content_pinned() -> None:
    service = _load_training_compose_service()
    tools_service = _load_training_tools_compose_service()
    mlflow_service = _load_mlflow_compose_service()
    dockerfile = TRAINING_DOCKERFILE_PATH.read_text(encoding="utf-8")
    verifier = TRAINING_VERIFY_PATH.read_text(encoding="utf-8")

    assert service["profiles"] == ["training"]
    assert service["build"] == {
        "context": ".",
        "dockerfile": "docker/training/Dockerfile",
    }
    assert service["image"] == (
        "document-ocr/kie-trainer:torch2.13.0-cuda13.0-transformers5.15.0"
    )
    assert service["working_dir"] == "/workspace"
    assert service["command"] == ["--help"]
    assert tools_service["command"] == [
        "validate-config",
        "--config",
        "configs/training/t5gemma2_270m_lora.pilot106.yaml",
        "--project-root",
        "/workspace",
    ]
    assert service["environment"] == {
        "HF_TOKEN": "${HF_TOKEN:-}",
        "HF_HOME": "/cache/huggingface",
        "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_CACHE_DIR": "/cache/torch-inductor",
        "TOKENIZERS_PARALLELISM": "false",
    }
    assert ".:/workspace" in service["volumes"]
    assert "deploy" not in tools_service
    assert service["depends_on"] == {
        "mlflow-server": {"condition": "service_healthy"}
    }
    devices = service["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]

    assert dockerfile.startswith(
        "# syntax=docker/dockerfile:1.7@sha256:"
        "a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e\n"
    )
    assert (
        "ghcr.io/astral-sh/uv:0.11.8@sha256:"
        "3b7b60a81d3c57ef471703e5c83fd4aaa33abcd403596fb22ab07db85ae91347"
        in dockerfile
    )
    assert (
        "pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:"
        "db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407"
        in dockerfile
    )
    assert "--prune torch" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "uv venv --system-site-packages" in dockerfile
    assert "uv pip uninstall --system --break-system-packages spin" in dockerfile
    assert "python -m pip check" in dockerfile
    assert 'ENTRYPOINT ["python", "-m", "document_ocr.training.cli"]' in dockerfile
    assert "python docker/training/verify_environment.py" in dockerfile
    assert '"torch": "2.13.0"' not in verifier
    assert 'torch.__version__.split("+", maxsplit=1)[0] != "2.13.0"' in verifier

    assert mlflow_service["profiles"] == ["training"]
    assert mlflow_service["image"] == (
        "ghcr.io/mlflow/mlflow:v3.15.1@sha256:"
        "ea84a0b879f08b35a6f22f22b294024413e780b8fc978eecf5f760ac16cc9ce5"
    )
    assert mlflow_service["ports"] == ["127.0.0.1:5000:5000"]
    assert "--backend-store-uri" in mlflow_service["command"]
    assert "--artifacts-destination" in mlflow_service["command"]
    assert "--allowed-hosts" in mlflow_service["command"]
    assert mlflow_service["volumes"] == ["mlflow-data:/mlflow"]


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
        "blc500-followup-local",
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
    assert _single_option(command, "--dtype") == config.vllm.dtype
    if config.vllm.quantization == "none":
        assert "--quantization" not in command
    else:
        assert _single_option(command, "--quantization") == config.vllm.quantization

    assert int(_single_option(command, "--max-model-len")) == config.vllm.max_model_len
    assert int(_single_option(command, "--max-num-batched-tokens")) == (
        config.vllm.max_num_batched_tokens
    )
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
    patch_bytes = tuple(path.read_bytes() for path in VLLM_PATCH_PATHS)
    dockerfile = VLLM_DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert manifest == {
        "schema_version": 2,
        "image": "document-ocr/vllm-openai:v0.26.0-glm-ocr-mtp-89e3c3f-df63cb9",
        "base_image": (
            "vllm/vllm-openai:v0.26.0@sha256:"
            "ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
        ),
        "vllm_version": "0.26.0",
        "patches": [
            {
                "source": (
                    "https://github.com/vllm-project/vllm/commit/"
                    "89e3c3f5b41d0f678d19a138dba59b3757a1a16f"
                ),
                "commit": "89e3c3f5b41d0f678d19a138dba59b3757a1a16f",
                "sha256": "0923d3e1975634a7e2b639119a8d78a190a991c5635fba7b8e2df7d17b5e3993",
                "targets": [
                    {
                        "path": (
                            "/usr/local/lib/python3.12/dist-packages/vllm/"
                            "model_executor/models/utils.py"
                        ),
                        "before_sha256": (
                            "f06d1a1a8d92e6ab39ebf7931d3eb3d6688d212606b1c409d7bed8eed5ba1fc2"
                        ),
                        "after_sha256": (
                            "f8d69922ac178e4e8dc8083bef8714234005254785c2a3645ed74b3dfe521733"
                        ),
                    }
                ],
            },
            {
                "source": (
                    "https://github.com/vllm-project/vllm/commit/"
                    "df63cb9492e85d3df71284a5d9f234fc39ae0b74"
                ),
                "commit": "df63cb9492e85d3df71284a5d9f234fc39ae0b74",
                "sha256": "fbc1648c49f56ae4d3ee664969672afdfe1230da5f4878f285bbd99ad6469588",
                "targets": [
                    {
                        "path": (
                            "/usr/local/lib/python3.12/dist-packages/vllm/"
                            "model_executor/models/glm_ocr_mtp.py"
                        ),
                        "before_sha256": (
                            "f7e5c90fdefdad05f05cad17cc74c7f360a349aeecfb3885212b4f07dd8d1fc1"
                        ),
                        "after_sha256": (
                            "034d815af1c797bd2d7437d799efe0bc8010ab1cf8cfe8a3f39b3c89158f6101"
                        ),
                    },
                    {
                        "path": (
                            "/usr/local/lib/python3.12/dist-packages/vllm/"
                            "model_executor/models/glm4_moe_lite_mtp.py"
                        ),
                        "before_sha256": (
                            "a4c0fab8092b04fc0c54b4c8936ba4712a69d734f82bd6018572238e828cc444"
                        ),
                        "after_sha256": (
                            "499d3f66c7a939991bad6e760ba5a396fd49f32bab054fec1b440a4cc8408529"
                        ),
                    },
                ],
            },
        ],
    }
    assert hashlib.sha256(manifest_bytes).hexdigest() == (
        "c1ffd97e7a6fa3fdb02b5acd84966e6afc10c096a5eec5e7bd47dad7a88197b4"
    )
    assert [hashlib.sha256(value).hexdigest() for value in patch_bytes] == [
        patch["sha256"] for patch in manifest["patches"]
    ]
    assert dockerfile.startswith(
        "# syntax=docker/dockerfile:1.7@sha256:"
        "a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e\n"
    )
    assert f"FROM {manifest['base_image']}" in dockerfile
    for patch in manifest["patches"]:
        assert patch["commit"] in dockerfile
        assert patch["sha256"] in dockerfile
    assert "RUN PYTHONDONTWRITEBYTECODE=1 python3" in dockerfile
    assert "model.language_model.layers.{base + i}." in patch_bytes[0].decode("utf-8")
    assert "inputs_embeds = torch.where" in patch_bytes[1].decode("utf-8")
