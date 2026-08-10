from __future__ import annotations

import json
import re
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import yaml

from document_ocr.config import PipelineConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = PROJECT_ROOT / "compose.yaml"
EXAMPLE_CONFIG_PATHS = (
    PROJECT_ROOT / "configs" / "glm_ocr.local.example.yaml",
    PROJECT_ROOT / "configs" / "glm_ocr.s3.example.yaml",
)


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


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIG_PATHS, ids=("local", "s3"))
def test_example_config_matches_pinned_vllm_compose_contract(config_path: Path) -> None:
    config: PipelineConfig = load_config(config_path)
    service = _load_compose_service()
    command = _command(service)

    assert service["image"] == config.vllm.container_image
    assert _single_option(command, "--model") == config.vllm.model
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
    assert environment["DOCUMENT_OCR_VLLM_IMAGE"] == service["image"]
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
