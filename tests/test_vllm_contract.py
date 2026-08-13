from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import document_ocr.vllm_contract as contract_module
from document_ocr.vllm_contract import (
    RUNTIME_CONTRACT_PATH,
    RuntimeContractError,
    build_runtime_contract,
    runtime_contract_middleware,
)

CONTAINER_IMAGE = (
    "vllm/vllm-openai:v0.26.0@sha256:"
    "ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
)
PATCH_COMMITS = (
    "89e3c3f5b41d0f678d19a138dba59b3757a1a16f",
    "df63cb9492e85d3df71284a5d9f234fc39ae0b74",
)


def _runtime_state() -> SimpleNamespace:
    def get_limit_per_prompt(modality: str) -> int:
        assert modality == "image"
        return 1

    multimodal_config = SimpleNamespace(get_limit_per_prompt=get_limit_per_prompt)
    model_config = SimpleNamespace(
        model="zai-org/GLM-OCR",
        served_model_name="glm-ocr",
        revision="ca5d8b3e287e52589e37c28385d9655ee4372f9d",
        max_model_len=32768,
        generation_config="vllm",
        multimodal_config=multimodal_config,
    )
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=model_config,
            scheduler_config=SimpleNamespace(max_num_seqs=16),
            cache_config=SimpleNamespace(gpu_memory_utilization=0.9),
            speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=1),
        )
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _install_build_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_after_sha256: str | None = None,
    mutate_manifest: Callable[[dict[str, object]], None] | None = None,
    **changes: object,
) -> str:
    targets = (tmp_path / "utils.py", tmp_path / "glm_ocr_mtp.py")
    targets[0].write_bytes(b"patched vllm utils\n")
    targets[1].write_bytes(b"patched GLM-OCR MTP\n")
    target_sha256s = tuple(hashlib.sha256(target.read_bytes()).hexdigest() for target in targets)
    manifest: dict[str, object] = {
        "schema_version": 2,
        "image": "document-ocr/vllm-openai:v0.26.0-glm-ocr-mtp-89e3c3f-df63cb9",
        "base_image": CONTAINER_IMAGE,
        "vllm_version": "0.26.0",
        "patches": [
            {
                "source": f"https://github.com/vllm-project/vllm/commit/{PATCH_COMMITS[0]}",
                "commit": PATCH_COMMITS[0],
                "sha256": "a" * 64,
                "targets": [
                    {
                        "path": str(targets[0]),
                        "before_sha256": "b" * 64,
                        "after_sha256": target_after_sha256 or target_sha256s[0],
                    }
                ],
            },
            {
                "source": f"https://github.com/vllm-project/vllm/commit/{PATCH_COMMITS[1]}",
                "commit": PATCH_COMMITS[1],
                "sha256": "c" * 64,
                "targets": [
                    {
                        "path": str(targets[1]),
                        "before_sha256": "d" * 64,
                        "after_sha256": target_sha256s[1],
                    }
                ],
            },
        ],
    }
    manifest.update(changes)
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    raw_manifest = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    manifest_path = tmp_path / "build-manifest.json"
    manifest_path.write_bytes(raw_manifest)
    monkeypatch.setenv("DOCUMENT_OCR_VLLM_BUILD_MANIFEST", str(manifest_path))
    monkeypatch.setattr(contract_module.importlib.metadata, "version", lambda _: "0.26.0")
    return hashlib.sha256(raw_manifest).hexdigest()


def test_build_runtime_contract_returns_exact_resolved_claim_and_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_sha256 = _install_build_manifest(tmp_path, monkeypatch)
    expected_payload: dict[str, object] = {
        "schema_version": 2,
        "model": "zai-org/GLM-OCR",
        "served_model_name": "glm-ocr",
        "model_revision": "ca5d8b3e287e52589e37c28385d9655ee4372f9d",
        "max_model_len": 32768,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.9,
        "generation_config": "vllm",
        "speculative_method": "mtp",
        "num_speculative_tokens": 1,
        "image_limit_per_prompt": 1,
        "container_base_image": CONTAINER_IMAGE,
        "container_build_manifest_sha256": manifest_sha256,
    }

    actual = build_runtime_contract(_runtime_state())

    assert actual == {
        **expected_payload,
        "contract_sha256": _canonical_sha256(expected_payload),
    }


@pytest.mark.parametrize(
    ("mutate", "error_match"),
    [
        (
            lambda state: delattr(state, "vllm_config"),
            "vLLM runtime has no required 'vllm_config' field",
        ),
        (
            lambda state: delattr(state.vllm_config.model_config, "revision"),
            "vLLM runtime has no required 'revision' field",
        ),
        (
            lambda state: setattr(state.vllm_config.model_config, "model", ""),
            "vLLM runtime field 'model' must be a non-empty string",
        ),
        (
            lambda state: setattr(state.vllm_config.model_config, "max_model_len", True),
            "vLLM runtime field 'max_model_len' must be a positive integer",
        ),
        (
            lambda state: setattr(state.vllm_config.scheduler_config, "max_num_seqs", 0),
            "vLLM runtime field 'max_num_seqs' must be a positive integer",
        ),
        (
            lambda state: setattr(state.vllm_config.cache_config, "gpu_memory_utilization", "0.9"),
            "vLLM runtime field 'gpu_memory_utilization' must be numeric",
        ),
        (
            lambda state: setattr(state.vllm_config.cache_config, "gpu_memory_utilization", 1.01),
            "vLLM runtime field 'gpu_memory_utilization' must be in \\(0, 1]",
        ),
        (
            lambda state: setattr(state.vllm_config.model_config, "multimodal_config", None),
            "vLLM runtime has no multimodal configuration",
        ),
        (
            lambda state: setattr(
                state.vllm_config.model_config.multimodal_config,
                "get_limit_per_prompt",
                1,
            ),
            "vLLM multimodal image-limit accessor is not callable",
        ),
        (
            lambda state: setattr(
                state.vllm_config.model_config.multimodal_config,
                "get_limit_per_prompt",
                lambda modality: 0,
            ),
            "vLLM runtime field 'image_limit_per_prompt' must be a positive integer",
        ),
    ],
    ids=(
        "missing-vllm-config",
        "missing-model-revision",
        "empty-model",
        "boolean-model-length",
        "zero-sequence-limit",
        "string-gpu-utilization",
        "out-of-range-gpu-utilization",
        "missing-multimodal-config",
        "non-callable-image-limit",
        "invalid-image-limit",
    ),
)
def test_build_runtime_contract_rejects_missing_or_malformed_runtime_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[Any], object],
    error_match: str,
) -> None:
    _install_build_manifest(tmp_path, monkeypatch)
    state = _runtime_state()
    mutate(state)

    with pytest.raises(RuntimeContractError, match=error_match):
        build_runtime_contract(state)


@pytest.mark.parametrize(
    "image_claim",
    [
        "vllm/vllm-openai:v0.26.0",
        "vllm/vllm-openai:v0.26.0@sha256:abc",
        f"vllm/vllm-openai:v0.26.0@sha256:{'A' * 64}",
        f"vllm/vllm-openai:v0.26.0 @sha256:{'a' * 64}",
    ],
    ids=("tag-only", "short-digest", "uppercase-digest", "whitespace"),
)
def test_build_runtime_contract_rejects_malformed_container_base_image_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_claim: str,
) -> None:
    _install_build_manifest(tmp_path, monkeypatch, base_image=image_claim)

    with pytest.raises(
        RuntimeContractError,
        match="vLLM build manifest base image is not digest-pinned",
    ):
        build_runtime_contract(_runtime_state())


def test_build_runtime_contract_requires_container_base_image_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCUMENT_OCR_VLLM_BUILD_MANIFEST", raising=False)

    with pytest.raises(
        RuntimeContractError,
        match="DOCUMENT_OCR_VLLM_BUILD_MANIFEST must name the baked build manifest",
    ):
        build_runtime_contract(_runtime_state())


def test_build_runtime_contract_rejects_tampered_patched_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_build_manifest(tmp_path, monkeypatch, target_after_sha256="c" * 64)

    with pytest.raises(
        RuntimeContractError,
        match="installed vLLM target does not match the build manifest",
    ):
        build_runtime_contract(_runtime_state())


def test_build_runtime_contract_rejects_duplicate_patched_target_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def duplicate_target(manifest: dict[str, object]) -> None:
        patches = manifest["patches"]
        assert isinstance(patches, list)
        first_patch, second_patch = patches
        assert isinstance(first_patch, dict) and isinstance(second_patch, dict)
        first_targets = first_patch["targets"]
        second_targets = second_patch["targets"]
        assert isinstance(first_targets, list) and isinstance(second_targets, list)
        first_target = first_targets[0]
        second_target = second_targets[0]
        assert isinstance(first_target, dict) and isinstance(second_target, dict)
        second_target["path"] = first_target["path"]

    _install_build_manifest(tmp_path, monkeypatch, mutate_manifest=duplicate_target)

    with pytest.raises(RuntimeContractError, match="duplicate target paths"):
        build_runtime_contract(_runtime_state())


def test_build_runtime_contract_rejects_unpinned_patch_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def change_source(manifest: dict[str, object]) -> None:
        patches = manifest["patches"]
        assert isinstance(patches, list)
        first_patch = patches[0]
        assert isinstance(first_patch, dict)
        first_patch["source"] = "https://github.com/vllm-project/vllm/pull/49869"

    _install_build_manifest(tmp_path, monkeypatch, mutate_manifest=change_source)

    with pytest.raises(RuntimeContractError, match="source does not match its commit"):
        build_runtime_contract(_runtime_state())


def test_build_runtime_contract_rejects_mismatched_vllm_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_build_manifest(tmp_path, monkeypatch, vllm_version="0.25.1")

    with pytest.raises(
        RuntimeContractError,
        match="installed vLLM version does not match the build manifest",
    ):
        build_runtime_contract(_runtime_state())


def test_build_runtime_contract_rejects_disabled_speculative_decoding() -> None:
    state = _runtime_state()
    state.vllm_config.speculative_config = None

    with pytest.raises(
        RuntimeContractError,
        match="vLLM runtime has speculative decoding disabled",
    ):
        build_runtime_contract(state)


@pytest.mark.asyncio
async def test_runtime_contract_middleware_passes_through_other_paths() -> None:
    request = SimpleNamespace(url=SimpleNamespace(path="/health"))
    inner_response = SimpleNamespace(status_code=404, source="inner")
    calls: list[object] = []

    async def call_next(actual_request: object) -> object:
        calls.append(actual_request)
        return inner_response

    actual = await runtime_contract_middleware(request, call_next)

    assert actual is inner_response
    assert calls == [request]


@pytest.mark.asyncio
async def test_runtime_contract_endpoint_preserves_inner_authentication_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SimpleNamespace(
        url=SimpleNamespace(path=RUNTIME_CONTRACT_PATH),
        app=SimpleNamespace(state=object()),
    )
    unauthorized = SimpleNamespace(status_code=401, source="vllm-api-key-middleware")
    calls: list[object] = []

    async def call_next(actual_request: object) -> object:
        calls.append(actual_request)
        return unauthorized

    monkeypatch.setitem(sys.modules, "starlette.responses", None)

    actual = await runtime_contract_middleware(request, call_next)

    assert actual is unauthorized
    assert calls == [request]


@pytest.mark.asyncio
async def test_runtime_contract_endpoint_replaces_inner_404_with_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class JSONResponse:
        def __init__(self, content: object) -> None:
            self.content = content
            self.status_code = 200

    starlette_module = ModuleType("starlette")
    starlette_module.__path__ = []
    responses_module = ModuleType("starlette.responses")
    responses_module.JSONResponse = JSONResponse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "starlette", starlette_module)
    monkeypatch.setitem(sys.modules, "starlette.responses", responses_module)
    _install_build_manifest(tmp_path, monkeypatch)

    request = SimpleNamespace(
        url=SimpleNamespace(path=RUNTIME_CONTRACT_PATH),
        app=SimpleNamespace(state=_runtime_state()),
    )
    not_found = SimpleNamespace(status_code=404, source="inner")
    calls: list[object] = []

    async def call_next(actual_request: object) -> object:
        calls.append(actual_request)
        return not_found

    response = await runtime_contract_middleware(request, call_next)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert response.content == build_runtime_contract(request.app.state)
    assert calls == [request]
