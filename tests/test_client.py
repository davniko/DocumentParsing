from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

import document_ocr.client as client_module
from document_ocr.client import (
    OcrResponse,
    ServerInfo,
    VllmClientError,
    VllmOcrClient,
    VllmRasterError,
    VllmReadinessError,
)
from document_ocr.config import TableVllmConfig, VllmConfig
from document_ocr.models import PageProvenance
from document_ocr.vllm_contract import runtime_contract_payload, runtime_contract_sha256

MODEL_REVISION = "c" * 40
SHA256 = "a" * 64
NOW = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
CONTAINER_IMAGE = (
    "vllm/vllm-openai:v0.26.0@sha256:"
    "ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
)
BUILD_MANIFEST_SHA256 = "d" * 64


def make_config(*, max_attempts: int = 3) -> VllmConfig:
    return VllmConfig.model_validate(
        {
            "endpoint": "http://127.0.0.1:8000",
            "model": "zai-org/GLM-OCR",
            "served_model_name": "glm-ocr",
            "revision": MODEL_REVISION,
            "engine_version": "0.26.0",
            "container_base_image": CONTAINER_IMAGE,
            "container_build_manifest_sha256": BUILD_MANIFEST_SHA256,
            "dtype": "bfloat16",
            "quantization": "none",
            "max_model_len": 32768,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.9,
            "prompt": "Text Recognition:",
            "request_timeout_seconds": 120.0,
            "api_key_env": "VLLM_API_KEY",
            "sampling": {
                "temperature": 0.0,
                "top_p": 0.00001,
                "top_k": 1,
                "repetition_penalty": 1.1,
                "max_tokens": 8192,
                "seed": 0,
            },
            "retry": {
                "max_attempts": max_attempts,
                "initial_backoff_seconds": 0.001,
                "max_backoff_seconds": 1.0,
                "backoff_multiplier": 2.0,
                "jitter_fraction": 0.0,
            },
            "speculative_decoding": {
                "method": "mtp",
                "num_speculative_tokens": 1,
            },
            "repetition_detection": {
                "min_pattern_size": 5,
                "max_pattern_size": 64,
                "min_count": 5,
            },
        },
        strict=True,
    )


def make_table_config(*, max_attempts: int = 3) -> TableVllmConfig:
    value = make_config(max_attempts=max_attempts).model_dump(mode="python")
    value["prompt"] = "Table Recognition:"
    return TableVllmConfig.model_validate(value, strict=True)


def runtime_contract_body(**changes: object) -> dict[str, object]:
    payload = runtime_contract_payload(
        model="zai-org/GLM-OCR",
        served_model_name="glm-ocr",
        model_revision=MODEL_REVISION,
        dtype="bfloat16",
        quantization="none",
        max_model_len=32768,
        max_num_batched_tokens=16384,
        max_num_seqs=16,
        gpu_memory_utilization=0.9,
        generation_config="vllm",
        speculative_method="mtp",
        num_speculative_tokens=1,
        image_limit_per_prompt=1,
        container_base_image=CONTAINER_IMAGE,
        container_build_manifest_sha256=BUILD_MANIFEST_SHA256,
    )
    payload.update(changes)
    return {**payload, "contract_sha256": runtime_contract_sha256(payload)}


def expected_server_info() -> ServerInfo:
    contract = runtime_contract_body()
    return ServerInfo(
        version="0.26.0",
        models=("glm-ocr",),
        model_repository="zai-org/GLM-OCR",
        served_model_name="glm-ocr",
        model_revision=MODEL_REVISION,
        dtype="bfloat16",
        quantization="none",
        max_model_len=32768,
        max_num_batched_tokens=16384,
        max_num_seqs=16,
        gpu_memory_utilization=0.9,
        generation_config="vllm",
        speculative_method="mtp",
        num_speculative_tokens=1,
        image_limit_per_prompt=1,
        container_base_image=CONTAINER_IMAGE,
        container_build_manifest_sha256=BUILD_MANIFEST_SHA256,
        contract_sha256=str(contract["contract_sha256"]),
    )


def raster_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion_body(
    text: str = "  Heading\n\nbody text\n",
    *,
    finish_reason: str | None = "stop",
    stop_reason: str | int | None = None,
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-response-1",
        "object": "chat.completion",
        "created": 1_754_389_200,
        "model": "glm-ocr",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 128,
            "completion_tokens": 256,
            "total_tokens": 384,
        },
    }


def page_provenance() -> PageProvenance:
    return PageProvenance.model_validate(
        {
            "document_id": "document-001",
            "source_type": "local",
            "source_uri": "file:///data/pdfs/a.pdf",
            "source_dataset_version": "contracts-2026-08-05",
            "source_object_version": SHA256,
            "source_size_bytes": 1024,
            "source_last_modified": NOW,
            "source_sha256": SHA256,
            "local_canonical_path": "/data/pdfs/a.pdf",
            "local_relative_key": "a.pdf",
            "local_device": 42,
            "local_inode": 1234,
            "local_mtime_ns": 1_754_389_200_000_000_000,
            "schema_version": 2,
            "run_id": "glm-ocr-20260805T100000Z",
            "extraction_id": "extract-document-001-page-000002",
            "page_id": "document-001:2",
            "config_sha256": "b" * 64,
            "pipeline_fingerprint": "f" * 64,
            "document_page_count": 3,
            "page_index": 1,
            "page_number": 2,
        },
        strict=True,
    )


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "test-key-never-log")


def make_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_client_requires_api_key_and_positive_connection_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_API_KEY")
    with pytest.raises(VllmReadinessError, match="VLLM_API_KEY") as missing:
        VllmOcrClient(make_config(), max_connections=1)
    assert "test-key-never-log" not in str(missing.value)

    monkeypatch.setenv("VLLM_API_KEY", "present")
    with pytest.raises(ValueError, match="max_connections must be positive"):
        VllmOcrClient(make_config(), max_connections=0)


@pytest.mark.asyncio
async def test_readiness_validates_health_version_alias_and_repository() -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer test-key-never-log"
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.26.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "glm-ocr",
                            "root": "zai-org/GLM-OCR",
                            "max_model_len": 32768,
                            "object": "model",
                        }
                    ],
                },
            )
        if request.url.path == "/document-ocr/server-contract":
            return httpx.Response(200, json=runtime_contract_body())
        raise AssertionError(f"unexpected path {request.url.path}")

    async with VllmOcrClient(
        make_config(), max_connections=4, transport=make_transport(handler)
    ) as client:
        info = await client.check_readiness()

    assert info == expected_server_info()
    assert seen_paths[0] == "/health"
    assert set(seen_paths[1:]) == {
        "/version",
        "/v1/models",
        "/document-ocr/server-contract",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("version", "model_data", "message"),
    [
        (
            "0.25.0",
            [{"id": "glm-ocr", "root": "zai-org/GLM-OCR"}],
            "version mismatch",
        ),
        (
            "0.26.0",
            [{"id": "another-model", "root": "zai-org/GLM-OCR"}],
            "exactly once",
        ),
        (
            "0.26.0",
            [{"id": "glm-ocr", "root": "wrong/repository"}],
            "repository does not match",
        ),
        (
            "0.26.0",
            [{"id": "glm-ocr", "root": 123}],
            "invalid model root",
        ),
        (
            "0.26.0",
            [{"id": "glm-ocr", "root": "zai-org/GLM-OCR"}],
            "valid max_model_len",
        ),
        (
            "0.26.0",
            [
                {
                    "id": "glm-ocr",
                    "root": "zai-org/GLM-OCR",
                    "max_model_len": 16384,
                }
            ],
            "max_model_len does not match",
        ),
    ],
)
async def test_readiness_rejects_server_identity_mismatches(
    version: str,
    model_data: list[dict[str, object]],
    message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": version})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": model_data})
        return httpx.Response(200, json=runtime_contract_body())

    async with VllmOcrClient(
        make_config(), max_connections=2, transport=make_transport(handler)
    ) as client:
        with pytest.raises(VllmReadinessError, match=message):
            await client.check_readiness()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1),
        ("model", "wrong/repository"),
        ("served_model_name", "wrong-alias"),
        ("model_revision", "d" * 40),
        ("dtype", "float16"),
        ("quantization", "fp8"),
        ("max_model_len", 16384),
        ("max_num_batched_tokens", 8192),
        ("max_num_seqs", 8),
        ("gpu_memory_utilization", 0.8),
        ("generation_config", "auto"),
        ("speculative_method", "draft_model"),
        ("num_speculative_tokens", 2),
        ("image_limit_per_prompt", 2),
        (
            "container_base_image",
            "vllm/vllm-openai:v0.26.0@sha256:" + "d" * 64,
        ),
        ("container_build_manifest_sha256", "e" * 64),
    ],
)
async def test_readiness_rejects_resolved_runtime_contract_mismatch(
    field: str,
    value: object,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.26.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "glm-ocr",
                            "root": "zai-org/GLM-OCR",
                            "max_model_len": 32768,
                        }
                    ]
                },
            )
        return httpx.Response(200, json=runtime_contract_body(**{field: value}))

    async with VllmOcrClient(
        make_config(), max_connections=2, transport=make_transport(handler)
    ) as client:
        with pytest.raises(VllmReadinessError, match=f"mismatch for: {field}"):
            await client.check_readiness()


@pytest.mark.asyncio
async def test_readiness_rejects_invalid_runtime_contract_digest() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.26.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "glm-ocr",
                            "root": "zai-org/GLM-OCR",
                            "max_model_len": 32768,
                        }
                    ]
                },
            )
        return httpx.Response(200, json={**runtime_contract_body(), "contract_sha256": "0" * 64})

    async with VllmOcrClient(
        make_config(), max_connections=2, transport=make_transport(handler)
    ) as client:
        with pytest.raises(VllmReadinessError, match="contract SHA-256 is invalid"):
            await client.check_readiness()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_body",
    [
        b'{"version":"0.26.0","version":"0.26.0"}',
        b'{"version":"0.26.0","extension":NaN}',
    ],
)
async def test_readiness_rejects_ambiguous_or_nonstandard_json(
    invalid_body: bytes,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/version":
            return httpx.Response(200, content=invalid_body)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json=runtime_contract_body())

    async with VllmOcrClient(
        make_config(), max_connections=2, transport=make_transport(handler)
    ) as client:
        with pytest.raises(VllmReadinessError, match="not valid JSON"):
            await client.check_readiness()


@pytest.mark.asyncio
async def test_recognize_page_sends_exact_glm_request_and_preserves_raw_response(
    tmp_path: Path,
) -> None:
    raster = tmp_path / "page.png"
    raster_bytes = b"\x89PNG\r\n\x1a\npage-raster"
    raster.write_bytes(raster_bytes)
    raw_response = json.dumps(completion_body(), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    captured_payload: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key-never-log"
        assert request.headers["X-Request-Id"] == "page-request-a1"
        captured_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=raw_response,
            headers={"X-Request-Id": "page-request-a1"},
        )

    async with VllmOcrClient(
        make_config(), max_connections=2, transport=make_transport(handler)
    ) as client:
        response = await client.recognize_page(
            raster,
            mime_type="image/png",
            raster_sha256=raster_sha256(raster),
            request_id="page-request",
        )

    image_url = captured_payload["messages"][0]["content"][0]["image_url"]["url"]
    assert image_url == "data:image/png;base64,iVBORw0KGgpwYWdlLXJhc3Rlcg=="
    assert captured_payload == {
        "model": "glm-ocr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ],
        "n": 1,
        "max_tokens": 8192,
        "temperature": 0.0,
        "top_p": 0.00001,
        "top_k": 1,
        "repetition_penalty": 1.1,
        "seed": 0,
        "stream": False,
        "repetition_detection": {
            "min_pattern_size": 5,
            "max_pattern_size": 64,
            "min_count": 5,
        },
    }
    assert response.text == "  Heading\n\nbody text\n"
    assert response.raw_response == raw_response
    assert response.raw_response_sha256 == hashlib.sha256(raw_response).hexdigest()
    assert response.request_id == "page-request-a1"
    assert response.server_request_id == "page-request-a1"
    assert response.prompt_tokens == 128
    assert response.completion_tokens == 256
    assert response.total_tokens == 384
    assert "Heading" not in repr(response)


@pytest.mark.asyncio
async def test_recognize_page_sends_exact_glm_table_recognition_request(
    tmp_path: Path,
) -> None:
    raster = tmp_path / "page.png"
    raster.write_bytes(b"\x89PNG\r\n\x1a\ntable-page")
    captured_payload: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(200, json=completion_body("<table><tr><td>A</td></tr></table>"))

    async with VllmOcrClient(
        make_table_config(), max_connections=1, transport=make_transport(handler)
    ) as client:
        await client.recognize_page(
            raster,
            mime_type="image/png",
            raster_sha256=raster_sha256(raster),
            request_id="table-page",
        )

    content = captured_payload["messages"][0]["content"]
    assert [item["type"] for item in content] == ["image_url", "text"]
    assert content[1] == {"type": "text", "text": "Table Recognition:"}
    assert captured_payload["temperature"] == 0.0
    assert captured_payload["top_p"] == 0.00001
    assert captured_payload["top_k"] == 1
    assert captured_payload["repetition_penalty"] == 1.1


@pytest.mark.asyncio
async def test_repetition_stopped_response_is_preserved_for_quality_filtering(
    tmp_path: Path,
) -> None:
    raster = tmp_path / "page.png"
    raster.write_bytes(b"png")
    repeated_text = "carrier terms " * 25
    raw_response = json.dumps(
        completion_body(
            repeated_text,
            finish_reason="repetition",
            stop_reason="repetition_detected",
        ),
        separators=(",", ":"),
    ).encode("utf-8")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw_response)

    async with VllmOcrClient(
        make_config(), max_connections=1, transport=make_transport(handler)
    ) as client:
        response = await client.recognize_page(
            raster,
            mime_type="image/png",
            raster_sha256=raster_sha256(raster),
            request_id="repetition-page",
        )

    assert response.finish_reason == "repetition"
    assert response.text == repeated_text
    assert response.raw_response == raw_response
    assert response.attempts[-1].outcome == "success"


@pytest.mark.asyncio
async def test_length_stopped_response_remains_a_terminal_failure(tmp_path: Path) -> None:
    raster = tmp_path / "page.png"
    raster.write_bytes(b"png")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body("partial", finish_reason="length"))

    async with VllmOcrClient(
        make_config(), max_connections=1, transport=make_transport(handler)
    ) as client:
        with pytest.raises(VllmClientError) as captured:
            await client.recognize_page(
                raster,
                mime_type="image/png",
                raster_sha256=raster_sha256(raster),
                request_id="length-page",
            )

    assert captured.value.attempts[-1].outcome == "invalid_response"
    assert captured.value.attempts[-1].error_message == "OCR output was truncated at max_tokens"


@pytest.mark.asyncio
async def test_raster_encoding_runs_off_the_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = tmp_path / "page.jpeg"
    raster.write_bytes(b"jpeg")
    event_loop_thread = threading.get_ident()
    encoder_threads: list[int] = []

    expected_raster_sha256 = raster_sha256(raster)

    def instrumented_encoder(path: Path, mime_type: str, expected_sha256: str) -> str:
        assert path == raster
        assert mime_type == "image/jpeg"
        assert expected_sha256 == expected_raster_sha256
        encoder_threads.append(threading.get_ident())
        return "data:image/jpeg;base64,anBlZw=="

    monkeypatch.setattr(client_module, "_image_data_uri", instrumented_encoder)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion_body(""))

    async with VllmOcrClient(
        make_config(), max_connections=1, transport=make_transport(handler)
    ) as client:
        response = await client.recognize_page(
            raster,
            mime_type="image/jpeg",
            raster_sha256=expected_raster_sha256,
            request_id="blank-page",
        )

    assert encoder_threads and encoder_threads[0] != event_loop_thread
    assert response.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe_kind", ["hash_drift", "symbolic_link"])
async def test_raster_submission_rejects_bytes_not_bound_to_renderer_identity(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    target = tmp_path / "target.png"
    target.write_bytes(b"original raster bytes")
    expected_sha256 = raster_sha256(target)
    raster = target
    if unsafe_kind == "hash_drift":
        target.write_bytes(b"changed raster bytes")
        expected_message = "SHA-256 changed"
    else:
        raster = tmp_path / "page.png"
        raster.symlink_to(target)
        expected_message = "safe regular file"

    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json=completion_body())

    async with VllmOcrClient(
        make_config(), max_connections=1, transport=make_transport(handler)
    ) as client:
        with pytest.raises(VllmRasterError, match=expected_message):
            await client.recognize_page(
                raster,
                mime_type="image/png",
                raster_sha256=expected_sha256,
                request_id="unsafe-raster",
            )

    assert requests == 0


@pytest.mark.asyncio
async def test_retryable_http_error_records_retry_after_and_maps_to_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = tmp_path / "page.png"
    raster.write_bytes(b"png")
    requests = 0
    sleeps: list[tuple[int, float | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(
                503,
                content=b'{"error":{"message":"sensitive OCR page text"}}',
                headers={"Retry-After": "0.25", "X-Request-Id": "server-a7"},
            )
        return httpx.Response(
            200,
            json=completion_body("ok"),
            headers={"X-Request-Id": "server-a8"},
        )

    async def record_sleep(
        _self: VllmOcrClient,
        offset: int,
        retry_after: float | None,
    ) -> None:
        sleeps.append((offset, retry_after))

    monkeypatch.setattr(VllmOcrClient, "_sleep_before_retry", record_sleep)
    async with VllmOcrClient(
        make_config(), max_connections=1, transport=make_transport(handler)
    ) as client:
        response = await client.recognize_page(
            raster,
            mime_type="image/png",
            raster_sha256=raster_sha256(raster),
            request_id="retry-page",
            first_attempt_number=7,
        )

    assert requests == 2
    assert sleeps == [(0, 0.25)]
    assert [attempt.inference_request_id for attempt in response.attempts] == [
        "retry-page-a7",
        "retry-page-a8",
    ]
    assert response.attempts[0].retry_after_seconds == 0.25
    assert response.attempts[0].error_message == "vLLM returned HTTP 503"
    assert "sensitive OCR page text" not in repr(response.attempts)

    first_record = response.attempts[0].to_inference_attempt(page_provenance())
    success_record = response.attempts[1].to_inference_attempt(page_provenance())
    assert first_record.attempt_outcome == "retryable_error"
    assert first_record.inference_server_request_id == "server-a7"
    assert first_record.retry_after_seconds == 0.25
    assert success_record.attempt_outcome == "success"
    assert success_record.inference_request_id == "retry-page-a8"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["http", "transport"])
async def test_per_request_attempt_limit_terminates_retryable_error(
    tmp_path: Path, failure_kind: str
) -> None:
    raster = tmp_path / "page.png"
    raster.write_bytes(b"png")
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if failure_kind == "transport":
            raise httpx.ReadTimeout("mock timeout")
        return httpx.Response(503, json={"error": {"message": "retry later"}})

    async with VllmOcrClient(
        make_config(max_attempts=4), max_connections=1, transport=make_transport(handler)
    ) as client:
        error_message = "transport request failed" if failure_kind == "transport" else "HTTP 503"
        with pytest.raises(VllmClientError, match=error_message) as failure:
            await client.recognize_page(
                raster,
                mime_type="image/png",
                raster_sha256=raster_sha256(raster),
                request_id="attempt-limit-page",
                max_attempts=1,
            )

    assert requests == 1
    assert len(failure.value.attempts) == 1


@pytest.mark.asyncio
async def test_terminal_http_error_does_not_retry_or_leak_response_body(
    tmp_path: Path,
) -> None:
    raster = tmp_path / "page.png"
    raster.write_bytes(b"png")
    calls = 0
    secret = "customer passport number 123456789"

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, content=secret.encode())

    async with VllmOcrClient(
        make_config(), max_connections=1, transport=make_transport(handler)
    ) as client:
        with pytest.raises(VllmClientError) as captured:
            await client.recognize_page(
                raster,
                mime_type="image/png",
                raster_sha256=raster_sha256(raster),
                request_id="bad-request",
            )

    assert calls == 1
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value.attempts)
    assert captured.value.attempts[0].retryable is False
    assert (
        captured.value.attempts[0].to_inference_attempt(page_provenance()).attempt_outcome
        == "terminal_error"
    )


@pytest.mark.asyncio
async def test_only_classified_transport_errors_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster = tmp_path / "page.png"
    raster.write_bytes(b"png")
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json=completion_body("recovered"))

    async def no_sleep(
        _self: VllmOcrClient,
        _offset: int,
        _retry_after: float | None,
    ) -> None:
        return None

    monkeypatch.setattr(VllmOcrClient, "_sleep_before_retry", no_sleep)
    async with VllmOcrClient(
        make_config(), max_connections=1, transport=make_transport(handler)
    ) as client:
        response = await client.recognize_page(
            raster,
            mime_type="image/png",
            raster_sha256=raster_sha256(raster),
            request_id="transport-retry",
        )

    assert requests == 2
    assert response.attempts[0].outcome == "transport_error"
    assert response.attempts[0].retryable is True
    assert response.attempts[0].error_message == "transient vLLM transport failure"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "expected_message"),
    [
        (
            lambda body: body.update({"choices": []}),
            "exactly one choice",
        ),
        (
            lambda body: body["choices"][0].update({"finish_reason": "content_filter"}),
            "finish_reason must be 'stop' or 'repetition'",
        ),
        (
            lambda body: body.update({"usage": None}),
            "usage must be an object",
        ),
        (
            lambda body: body["usage"].update({"completion_tokens": True}),
            "completion_tokens must be a non-negative integer",
        ),
        (
            lambda body: body["usage"].update({"total_tokens": 999}),
            "total_tokens is inconsistent",
        ),
        (
            lambda body: body.update({"model": "wrong-model"}),
            "model does not match",
        ),
        (
            lambda body: body["choices"][0]["message"].update({"role": "user"}),
            "message role must be 'assistant'",
        ),
    ],
)
async def test_invalid_success_responses_are_terminal_and_do_not_leak_text(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    expected_message: str,
) -> None:
    raster = tmp_path / "page.png"
    raster.write_bytes(b"png")
    sensitive_text = "private bill of lading content"
    body = completion_body(sensitive_text)
    mutate(body)
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=body)

    async with VllmOcrClient(
        make_config(), max_connections=1, transport=make_transport(handler)
    ) as client:
        with pytest.raises(VllmClientError) as captured:
            await client.recognize_page(
                raster,
                mime_type="image/png",
                raster_sha256=raster_sha256(raster),
                request_id="invalid-response",
            )

    assert calls == 1
    assert sensitive_text not in str(captured.value)
    assert sensitive_text not in repr(captured.value.attempts)
    assert captured.value.attempts[0].outcome == "invalid_response"
    assert captured.value.attempts[0].error_message is not None
    assert expected_message in captured.value.attempts[0].error_message


@pytest.mark.asyncio
async def test_invalid_json_is_terminal_and_body_is_not_exposed(tmp_path: Path) -> None:
    raster = tmp_path / "page.png"
    raster.write_bytes(b"png")
    response_body = b"not-json private invoice contents"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=response_body)

    async with VllmOcrClient(
        make_config(), max_connections=1, transport=make_transport(handler)
    ) as client:
        with pytest.raises(VllmClientError) as captured:
            await client.recognize_page(
                raster,
                mime_type="image/png",
                raster_sha256=raster_sha256(raster),
                request_id="invalid-json",
            )

    assert response_body.decode() not in str(captured.value)
    assert response_body.decode() not in repr(captured.value.attempts)
    assert captured.value.attempts[0].error_message == ("vLLM response body is not valid JSON")


@pytest.mark.asyncio
async def test_ocr_text_must_be_utf8_encodable(tmp_path: Path) -> None:
    raster = tmp_path / "page.png"
    raster.write_bytes(b"png")
    raw_response = json.dumps(completion_body("\ud800"), ensure_ascii=True).encode("utf-8")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw_response)

    async with VllmOcrClient(
        make_config(), max_connections=1, transport=make_transport(handler)
    ) as client:
        with pytest.raises(VllmClientError) as captured:
            await client.recognize_page(
                raster,
                mime_type="image/png",
                raster_sha256=raster_sha256(raster),
                request_id="invalid-unicode",
            )

    assert captured.value.attempts[0].outcome == "invalid_response"
    assert captured.value.attempts[0].error_message == ("OCR response content must be valid UTF-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_kind", ["duplicate_key", "nonfinite_number"])
async def test_ambiguous_or_nonstandard_json_is_terminal(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    raster = tmp_path / "page.png"
    raster.write_bytes(b"png")
    response_text = "private customs declaration"
    raw_response = json.dumps(completion_body(response_text), separators=(",", ":"))
    if invalid_kind == "duplicate_key":
        raw_response = raw_response.replace(
            f'"content":"{response_text}"',
            f'"content":"{response_text}","content":"replacement"',
        )
    else:
        raw_response = raw_response.replace('"created":1754389200', '"created":NaN')

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=raw_response.encode("utf-8"))

    async with VllmOcrClient(
        make_config(), max_connections=1, transport=make_transport(handler)
    ) as client:
        with pytest.raises(VllmClientError) as captured:
            await client.recognize_page(
                raster,
                mime_type="image/png",
                raster_sha256=raster_sha256(raster),
                request_id="strict-json",
            )

    assert response_text not in str(captured.value)
    assert response_text not in repr(captured.value.attempts)
    assert captured.value.attempts[0].outcome == "invalid_response"
    assert captured.value.attempts[0].error_message == "vLLM response body is not valid JSON"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0.0),
        ("1.25", 1.25),
        ("-1", None),
        ("nan", None),
        ("inf", None),
        ("not a date", None),
        ("", None),
        (None, None),
    ],
)
def test_retry_after_delta_seconds_are_parsed_strictly(
    value: str | None,
    expected: float | None,
) -> None:
    assert VllmOcrClient._retry_after_seconds(value, now=NOW) == expected


def test_retry_after_http_date_is_calculated_from_supplied_clock() -> None:
    future = NOW + timedelta(seconds=45)
    past = NOW - timedelta(seconds=45)

    assert (
        VllmOcrClient._retry_after_seconds(future.strftime("%a, %d %b %Y %H:%M:%S GMT"), now=NOW)
        == 45.0
    )
    assert (
        VllmOcrClient._retry_after_seconds(past.strftime("%a, %d %b %Y %H:%M:%S GMT"), now=NOW)
        == 0.0
    )


@pytest.mark.asyncio
async def test_retry_after_sleep_is_capped_and_not_jittered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    async with VllmOcrClient(
        make_config(),
        max_connections=1,
        transport=make_transport(lambda _: httpx.Response(200)),
    ) as client:
        await client._sleep_before_retry(0, 30.0)

    assert delays == [1.0]


def test_request_id_rejects_empty_non_ascii_and_control_characters() -> None:
    for value in ("", "réquest", "bad request", "bad\nrequest"):
        with pytest.raises(ValueError, match="request_id"):
            VllmOcrClient._validate_request_id(value)


def test_type_surface_uses_required_usage_counts() -> None:
    annotations = OcrResponse.__annotations__
    assert annotations["prompt_tokens"] == "int"
    assert annotations["completion_tokens"] == "int"
    assert annotations["total_tokens"] == "int"
