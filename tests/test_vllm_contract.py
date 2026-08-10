from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

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


def test_build_runtime_contract_returns_exact_resolved_claim_and_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCUMENT_OCR_VLLM_IMAGE", CONTAINER_IMAGE)
    expected_payload: dict[str, object] = {
        "schema_version": 1,
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
        "container_image": CONTAINER_IMAGE,
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
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[Any], object],
    error_match: str,
) -> None:
    monkeypatch.setenv("DOCUMENT_OCR_VLLM_IMAGE", CONTAINER_IMAGE)
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
def test_build_runtime_contract_rejects_malformed_container_image_claim(
    monkeypatch: pytest.MonkeyPatch,
    image_claim: str,
) -> None:
    monkeypatch.setenv("DOCUMENT_OCR_VLLM_IMAGE", image_claim)

    with pytest.raises(
        RuntimeContractError,
        match="DOCUMENT_OCR_VLLM_IMAGE must contain the digest-pinned running image claim",
    ):
        build_runtime_contract(_runtime_state())


def test_build_runtime_contract_requires_container_image_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCUMENT_OCR_VLLM_IMAGE", raising=False)

    with pytest.raises(
        RuntimeContractError,
        match="DOCUMENT_OCR_VLLM_IMAGE must contain the digest-pinned running image claim",
    ):
        build_runtime_contract(_runtime_state())


def test_build_runtime_contract_rejects_disabled_speculative_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCUMENT_OCR_VLLM_IMAGE", CONTAINER_IMAGE)
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
    monkeypatch.setenv("DOCUMENT_OCR_VLLM_IMAGE", CONTAINER_IMAGE)

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
