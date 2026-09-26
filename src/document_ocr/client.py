"""Async, bounded client for GLM-OCR on vLLM's OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import base64
import email.utils
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from document_ocr.atomic import ArtifactReadError, read_regular_file_bytes
from document_ocr.config import VllmClientConfig
from document_ocr.hashing import sha256_bytes
from document_ocr.models import InferenceAttempt, PageProvenance
from document_ocr.vllm_contract import (
    RUNTIME_CONTRACT_PATH,
    runtime_contract_payload,
    runtime_contract_sha256,
)

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _image_data_uri(path: Path, mime_type: str, expected_sha256: str) -> str:
    """Read, verify, and encode the exact raster bytes submitted to vLLM."""

    try:
        raster = read_regular_file_bytes(path)
    except ArtifactReadError as error:
        raise VllmRasterError("raster cannot be opened as a safe regular file") from error
    if sha256_bytes(raster) != expected_sha256:
        raise VllmRasterError("raster SHA-256 changed before inference submission")
    encoded = base64.b64encode(raster).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@dataclass(frozen=True, slots=True)
class RequestAttempt:
    """One HTTP attempt, shaped for lossless conversion to ``InferenceAttempt``."""

    attempt_number: int
    attempt_started_at: datetime
    attempt_completed_at: datetime
    attempt_duration_ms: float
    outcome: Literal["success", "http_error", "transport_error", "invalid_response"]
    retryable: bool
    inference_request_id: str
    inference_server_request_id: str | None = None
    http_status_code: int | None = None
    retry_after_seconds: float | None = None
    error_type: str | None = None
    error_message: str | None = None

    def as_inference_attempt_fields(self) -> dict[str, object]:
        """Return fields accepted by the durable ``InferenceAttempt`` model."""

        if self.outcome == "success":
            model_outcome = "success"
        elif self.retryable:
            model_outcome = "retryable_error"
        else:
            model_outcome = "terminal_error"
        return {
            "attempt_number": self.attempt_number,
            "attempt_started_at": self.attempt_started_at,
            "attempt_completed_at": self.attempt_completed_at,
            "attempt_duration_ms": self.attempt_duration_ms,
            "attempt_outcome": model_outcome,
            "inference_request_id": self.inference_request_id,
            "inference_server_request_id": self.inference_server_request_id,
            "http_status_code": self.http_status_code,
            "retry_after_seconds": self.retry_after_seconds,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    def to_inference_attempt(self, provenance: PageProvenance) -> InferenceAttempt:
        """Bind this attempt to page provenance for durable persistence."""

        values = provenance.model_dump(mode="python")
        values.update(self.as_inference_attempt_fields())
        return InferenceAttempt.model_validate(values, strict=True)


@dataclass(frozen=True, slots=True)
class ServerInfo:
    version: str
    models: tuple[str, ...]
    model_repository: str
    served_model_name: str
    model_revision: str
    dtype: Literal["bfloat16"]
    quantization: Literal["none", "fp8"]
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    gpu_memory_utilization: float
    generation_config: Literal["vllm"]
    speculative_method: Literal["mtp"]
    num_speculative_tokens: Literal[1, 3]
    image_limit_per_prompt: Literal[1]
    container_base_image: str
    container_build_manifest_sha256: str
    contract_sha256: str


@dataclass(frozen=True, slots=True)
class OcrResponse:
    """Validated non-streaming OCR response with exact raw bytes and text."""

    text: str = field(repr=False)
    finish_reason: Literal["stop", "repetition"]
    request_id: str
    response_id: str
    response_model: str
    server_request_id: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_response: bytes = field(repr=False)
    raw_response_sha256: str
    attempts: tuple[RequestAttempt, ...]


class VllmClientError(RuntimeError):
    """Terminal vLLM request failure with all safely classified attempts."""

    def __init__(self, message: str, attempts: tuple[RequestAttempt, ...]) -> None:
        if not attempts:
            raise ValueError("VllmClientError requires at least one classified attempt")
        super().__init__(message)
        self.attempts = attempts


class VllmReadinessError(RuntimeError):
    """The configured server is unavailable or does not match its contract."""


class VllmRasterError(RuntimeError):
    """The raster bytes no longer match the renderer or benchmark identity."""


class _InvalidResponseError(ValueError):
    """A safe response-contract error that never embeds response content."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object contains a duplicate key")
        result[key] = value
    return result


def _reject_nonfinite_json(_: str) -> None:
    raise ValueError("JSON contains a non-finite number")


def _strict_json_loads(payload: bytes) -> Any:
    """Decode RFC-compliant UTF-8 JSON without ambiguous object keys."""

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json,
    )


class VllmOcrClient:
    """Persistent async HTTP client for deterministic, page-level GLM-OCR."""

    def __init__(
        self,
        config: VllmClientConfig,
        *,
        max_connections: int,
        transport: httpx.AsyncBaseTransport | None = None,
        random_source: random.Random | None = None,
    ) -> None:
        if max_connections <= 0:
            raise ValueError("max_connections must be positive")
        self.config = config
        self._random = random_source or random.SystemRandom()
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise VllmReadinessError(
                f"required vLLM API key environment variable {config.api_key_env!r} is unset"
            )
        timeout = httpx.Timeout(
            timeout=config.request_timeout_seconds,
            connect=min(30.0, config.request_timeout_seconds),
            pool=min(30.0, config.request_timeout_seconds),
        )
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        self._client = httpx.AsyncClient(
            base_url=config.endpoint.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            limits=limits,
            transport=transport,
        )

    async def __aenter__(self) -> VllmOcrClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def check_readiness(self) -> ServerInfo:
        """Verify public readiness plus the authenticated resolved runtime contract."""

        try:
            health = await self._client.get("health")
            health.raise_for_status()
            version_response, models_response, contract_response = await asyncio.gather(
                self._client.get("version"),
                self._client.get("v1/models"),
                self._client.get(RUNTIME_CONTRACT_PATH.lstrip("/")),
            )
            version_response.raise_for_status()
            models_response.raise_for_status()
            contract_response.raise_for_status()
            version_data: Any = _strict_json_loads(version_response.content)
            models_data: Any = _strict_json_loads(models_response.content)
            contract_data: Any = _strict_json_loads(contract_response.content)
        except httpx.HTTPStatusError as error:
            raise VllmReadinessError(
                f"vLLM readiness endpoint returned HTTP {error.response.status_code}"
            ) from error
        except httpx.HTTPError as error:
            raise VllmReadinessError(
                f"vLLM readiness transport failure ({type(error).__name__})"
            ) from error
        except (ValueError, TypeError) as error:
            raise VllmReadinessError("vLLM readiness response is not valid JSON") from error

        if not isinstance(version_data, dict):
            raise VllmReadinessError("vLLM /version response root must be an object")
        version = version_data.get("version")
        if not isinstance(version, str) or not version:
            raise VllmReadinessError("vLLM /version response has no version string")
        expected_version = self.config.engine_version.removeprefix("v")
        if version.removeprefix("v") != expected_version:
            raise VllmReadinessError(
                f"vLLM version mismatch: expected {expected_version}, received {version}"
            )

        if not isinstance(models_data, dict):
            raise VllmReadinessError("vLLM /v1/models response root must be an object")
        model_entries = models_data.get("data")
        if not isinstance(model_entries, list):
            raise VllmReadinessError("vLLM /v1/models response has no data list")

        models: list[str] = []
        configured_models: list[tuple[str, int]] = []
        for entry in model_entries:
            if not isinstance(entry, dict):
                raise VllmReadinessError("vLLM model list contains a non-object entry")
            model_id = entry.get("id")
            if not isinstance(model_id, str) or not model_id:
                raise VllmReadinessError("vLLM model list entry has no model id")
            root = entry.get("root")
            if root is not None and (not isinstance(root, str) or not root):
                raise VllmReadinessError("vLLM model list entry has an invalid model root")
            models.append(model_id)
            if model_id == self.config.served_model_name:
                if root != self.config.model:
                    raise VllmReadinessError(
                        "vLLM model repository does not match the configured repository"
                    )
                max_model_len = entry.get("max_model_len")
                if (
                    isinstance(max_model_len, bool)
                    or not isinstance(max_model_len, int)
                    or max_model_len <= 0
                ):
                    raise VllmReadinessError("configured vLLM model has no valid max_model_len")
                configured_models.append((root, max_model_len))

        if len(configured_models) != 1:
            raise VllmReadinessError(
                "configured served model must appear exactly once in /v1/models"
            )
        model_repository, max_model_len = configured_models[0]
        if max_model_len != self.config.max_model_len:
            raise VllmReadinessError(
                "vLLM max_model_len does not match the configured server contract"
            )
        runtime_contract = self._validate_runtime_contract(contract_data)
        return ServerInfo(
            version=version,
            models=tuple(models),
            model_repository=model_repository,
            max_model_len=max_model_len,
            served_model_name=runtime_contract["served_model_name"],
            model_revision=runtime_contract["model_revision"],
            dtype=runtime_contract["dtype"],
            quantization=runtime_contract["quantization"],
            max_num_batched_tokens=runtime_contract["max_num_batched_tokens"],
            max_num_seqs=runtime_contract["max_num_seqs"],
            gpu_memory_utilization=runtime_contract["gpu_memory_utilization"],
            generation_config=runtime_contract["generation_config"],
            speculative_method=runtime_contract["speculative_method"],
            num_speculative_tokens=runtime_contract["num_speculative_tokens"],
            image_limit_per_prompt=runtime_contract["image_limit_per_prompt"],
            container_base_image=runtime_contract["container_base_image"],
            container_build_manifest_sha256=runtime_contract["container_build_manifest_sha256"],
            contract_sha256=runtime_contract["contract_sha256"],
        )

    def _validate_runtime_contract(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise VllmReadinessError("vLLM runtime contract root must be an object")
        expected = runtime_contract_payload(
            model=self.config.model,
            served_model_name=self.config.served_model_name,
            model_revision=self.config.revision,
            dtype=self.config.dtype,
            quantization=self.config.quantization,
            max_model_len=self.config.max_model_len,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_num_seqs=self.config.max_num_seqs,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            generation_config="vllm",
            speculative_method=self.config.speculative_decoding.method,
            num_speculative_tokens=(self.config.speculative_decoding.num_speculative_tokens),
            image_limit_per_prompt=1,
            container_base_image=self.config.container_base_image,
            container_build_manifest_sha256=(self.config.container_build_manifest_sha256),
        )
        expected_keys = set(expected) | {"contract_sha256"}
        if set(value) != expected_keys:
            raise VllmReadinessError("vLLM runtime contract has unexpected fields")

        string_fields = {
            "model",
            "served_model_name",
            "model_revision",
            "dtype",
            "quantization",
            "generation_config",
            "speculative_method",
            "container_base_image",
            "container_build_manifest_sha256",
            "contract_sha256",
        }
        if any(not isinstance(value[field], str) or not value[field] for field in string_fields):
            raise VllmReadinessError("vLLM runtime contract has an invalid string field")
        integer_fields = {
            "schema_version",
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
            "num_speculative_tokens",
            "image_limit_per_prompt",
        }
        if any(
            isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] <= 0
            for field in integer_fields
        ):
            raise VllmReadinessError("vLLM runtime contract has an invalid integer field")
        gpu_memory_utilization = value["gpu_memory_utilization"]
        if (
            isinstance(gpu_memory_utilization, bool)
            or not isinstance(gpu_memory_utilization, (int, float))
            or not 0.0 < float(gpu_memory_utilization) <= 1.0
        ):
            raise VllmReadinessError("vLLM runtime contract has invalid gpu_memory_utilization")

        payload = {key: item for key, item in value.items() if key != "contract_sha256"}
        actual_sha256 = runtime_contract_sha256(payload)
        if value["contract_sha256"] != actual_sha256:
            raise VllmReadinessError("vLLM runtime contract SHA-256 is invalid")

        mismatches = sorted(
            key for key, expected_value in expected.items() if payload[key] != expected_value
        )
        if mismatches:
            raise VllmReadinessError(
                "vLLM resolved runtime contract mismatch for: " + ", ".join(mismatches)
            )
        return value

    async def recognize_page(
        self,
        raster_path: Path,
        *,
        mime_type: Literal["image/png", "image/jpeg"],
        raster_sha256: str,
        request_id: str,
        first_attempt_number: int = 1,
        max_attempts: int | None = None,
    ) -> OcrResponse:
        """Recognize one raster, preserving the model's text and raw JSON exactly."""

        if first_attempt_number <= 0:
            raise ValueError("first_attempt_number must be positive")
        attempt_limit = self.config.retry.max_attempts if max_attempts is None else max_attempts
        if attempt_limit <= 0 or attempt_limit > self.config.retry.max_attempts:
            raise ValueError("max_attempts must be within the configured retry limit")
        self._validate_request_id(request_id)
        image_url = await asyncio.to_thread(
            _image_data_uri,
            raster_path,
            mime_type,
            raster_sha256,
        )
        sampling = self.config.sampling
        payload = {
            "model": self.config.served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": self.config.prompt},
                    ],
                }
            ],
            "n": 1,
            "max_tokens": sampling.max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "repetition_penalty": sampling.repetition_penalty,
            "seed": sampling.seed,
            "stream": False,
            "repetition_detection": self.config.repetition_detection.model_dump(mode="json"),
        }
        attempts: list[RequestAttempt] = []
        for offset in range(attempt_limit):
            attempt_number = first_attempt_number + offset
            attempt_request_id = f"{request_id}-a{attempt_number}"
            self._validate_request_id(attempt_request_id)
            started_at = _utc_now()
            start = time.perf_counter()
            try:
                response = await self._client.post(
                    "v1/chat/completions",
                    json=payload,
                    headers={"X-Request-Id": attempt_request_id},
                )
            except httpx.TransportError as error:
                retryable = isinstance(error, _RETRYABLE_TRANSPORT_ERRORS)
                attempt = self._failed_attempt(
                    attempt_number=attempt_number,
                    attempt_request_id=attempt_request_id,
                    started_at=started_at,
                    start=start,
                    outcome="transport_error",
                    retryable=retryable,
                    error_type=type(error).__name__,
                    error_message=(
                        "transient vLLM transport failure"
                        if retryable
                        else "terminal vLLM transport failure"
                    ),
                )
                attempts.append(attempt)
                if retryable and offset + 1 < attempt_limit:
                    await self._sleep_before_retry(offset, None)
                    continue
                raise VllmClientError(
                    "vLLM OCR transport request failed", tuple(attempts)
                ) from error

            server_request_id = response.headers.get("X-Request-Id") or None
            if response.status_code != 200:
                retryable = response.status_code in _RETRYABLE_STATUS_CODES
                retry_after = (
                    self._retry_after_seconds(response.headers.get("Retry-After"))
                    if retryable
                    else None
                )
                attempt = self._failed_attempt(
                    attempt_number=attempt_number,
                    attempt_request_id=attempt_request_id,
                    started_at=started_at,
                    start=start,
                    outcome="http_error",
                    retryable=retryable,
                    server_request_id=server_request_id,
                    http_status_code=response.status_code,
                    retry_after_seconds=retry_after,
                    error_type="HTTPStatusError",
                    error_message=f"vLLM returned HTTP {response.status_code}",
                )
                attempts.append(attempt)
                if retryable and offset + 1 < attempt_limit:
                    await self._sleep_before_retry(offset, retry_after)
                    continue
                raise VllmClientError(
                    f"vLLM OCR request failed with HTTP {response.status_code}",
                    tuple(attempts),
                )

            raw_response = response.content
            try:
                parsed: Any = _strict_json_loads(raw_response)
            except (ValueError, TypeError) as error:
                attempt = self._failed_attempt(
                    attempt_number=attempt_number,
                    attempt_request_id=attempt_request_id,
                    started_at=started_at,
                    start=start,
                    outcome="invalid_response",
                    retryable=False,
                    server_request_id=server_request_id,
                    http_status_code=200,
                    error_type=type(error).__name__,
                    error_message="vLLM response body is not valid JSON",
                )
                attempts.append(attempt)
                raise VllmClientError(
                    "vLLM returned an invalid OCR response", tuple(attempts)
                ) from error

            try:
                return self._parse_success(
                    parsed,
                    raw_response=raw_response,
                    request_id=attempt_request_id,
                    server_request_id=server_request_id,
                    attempts=attempts,
                    attempt_number=attempt_number,
                    started_at=started_at,
                    start=start,
                )
            except _InvalidResponseError as error:
                attempt = self._failed_attempt(
                    attempt_number=attempt_number,
                    attempt_request_id=attempt_request_id,
                    started_at=started_at,
                    start=start,
                    outcome="invalid_response",
                    retryable=False,
                    server_request_id=server_request_id,
                    http_status_code=200,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                attempts.append(attempt)
                raise VllmClientError(
                    "vLLM returned an invalid OCR response", tuple(attempts)
                ) from error
        raise AssertionError("retry loop exited without a result")

    def _parse_success(
        self,
        parsed: Any,
        *,
        raw_response: bytes,
        request_id: str,
        server_request_id: str | None,
        attempts: list[RequestAttempt],
        attempt_number: int,
        started_at: datetime,
        start: float,
    ) -> OcrResponse:
        if not isinstance(parsed, dict):
            raise _InvalidResponseError("response root must be an object")
        if parsed.get("object") != "chat.completion":
            raise _InvalidResponseError("response object must be 'chat.completion'")
        response_id = parsed.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise _InvalidResponseError("response id must be a non-empty string")
        response_model = parsed.get("model")
        if response_model != self.config.served_model_name:
            raise _InvalidResponseError("response model does not match the requested model")

        choices = parsed.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise _InvalidResponseError("response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise _InvalidResponseError("response choice must be an object")
        choice_index = choice.get("index")
        if isinstance(choice_index, bool) or choice_index != 0:
            raise _InvalidResponseError("response choice index must be zero")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise _InvalidResponseError("choice message must be an object")
        if message.get("role") != "assistant":
            raise _InvalidResponseError("choice message role must be 'assistant'")
        text = message.get("content")
        if not isinstance(text, str):
            raise _InvalidResponseError("OCR response content must be a string")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise _InvalidResponseError("OCR response content must be valid UTF-8") from error

        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise _InvalidResponseError("OCR output was truncated at max_tokens")
        if finish_reason not in {"stop", "repetition"}:
            raise _InvalidResponseError("OCR response finish_reason must be 'stop' or 'repetition'")
        stop_reason = choice.get("stop_reason")
        if isinstance(stop_reason, bool) or not isinstance(
            stop_reason, (str, int, type(None))
        ):
            raise _InvalidResponseError("choice stop_reason must be a string, integer, or null")
        if finish_reason == "repetition" and stop_reason != "repetition_detected":
            raise _InvalidResponseError(
                "repetition finish_reason requires stop_reason='repetition_detected'"
            )

        usage = parsed.get("usage")
        if not isinstance(usage, dict):
            raise _InvalidResponseError("response usage must be an object")
        prompt_tokens = self._required_nonnegative_int(usage, "prompt_tokens")
        completion_tokens = self._required_nonnegative_int(usage, "completion_tokens")
        total_tokens = self._required_nonnegative_int(usage, "total_tokens")
        if total_tokens != prompt_tokens + completion_tokens:
            raise _InvalidResponseError("usage total_tokens is inconsistent")

        completed_at = _utc_now()
        attempt = RequestAttempt(
            attempt_number=attempt_number,
            attempt_started_at=started_at,
            attempt_completed_at=completed_at,
            attempt_duration_ms=(time.perf_counter() - start) * 1000,
            outcome="success",
            retryable=False,
            inference_request_id=request_id,
            inference_server_request_id=server_request_id,
            http_status_code=200,
        )
        return OcrResponse(
            text=text,
            finish_reason=finish_reason,
            request_id=request_id,
            response_id=response_id,
            response_model=response_model,
            server_request_id=server_request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=raw_response,
            raw_response_sha256=sha256_bytes(raw_response),
            attempts=(*attempts, attempt),
        )

    @staticmethod
    def _required_nonnegative_int(mapping: dict[str, Any], key: str) -> int:
        value = mapping.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _InvalidResponseError(f"usage.{key} must be a non-negative integer")
        return value

    @staticmethod
    def _failed_attempt(
        *,
        attempt_number: int,
        attempt_request_id: str,
        started_at: datetime,
        start: float,
        outcome: Literal["http_error", "transport_error", "invalid_response"],
        retryable: bool,
        error_type: str,
        error_message: str,
        server_request_id: str | None = None,
        http_status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> RequestAttempt:
        return RequestAttempt(
            attempt_number=attempt_number,
            attempt_started_at=started_at,
            attempt_completed_at=_utc_now(),
            attempt_duration_ms=(time.perf_counter() - start) * 1000,
            outcome=outcome,
            retryable=retryable,
            inference_request_id=attempt_request_id,
            inference_server_request_id=server_request_id,
            http_status_code=http_status_code,
            retry_after_seconds=retry_after_seconds,
            error_type=error_type,
            error_message=error_message,
        )

    @staticmethod
    def _validate_request_id(value: str) -> None:
        if not value:
            raise ValueError("request_id must be non-empty")
        try:
            encoded = value.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("request_id must contain only visible ASCII characters") from error
        if any(byte < 0x21 or byte > 0x7E for byte in encoded):
            raise ValueError("request_id must contain only visible ASCII characters")

    @staticmethod
    def _retry_after_seconds(
        value: str | None,
        *,
        now: datetime | None = None,
    ) -> float | None:
        if value is None or not value.strip():
            return None
        stripped = value.strip()
        try:
            seconds = float(stripped)
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(stripped)
            except (TypeError, ValueError, OverflowError):
                return None
            if parsed is None:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            current = now or datetime.now(UTC)
            return max(0.0, (parsed - current).total_seconds())
        if not math.isfinite(seconds) or seconds < 0.0:
            return None
        return seconds

    async def _sleep_before_retry(self, offset: int, retry_after: float | None) -> None:
        retry = self.config.retry
        if retry_after is not None:
            delay = min(retry.max_backoff_seconds, retry_after)
        else:
            base = min(
                retry.max_backoff_seconds,
                retry.initial_backoff_seconds * (retry.backoff_multiplier**offset),
            )
            jitter = base * retry.jitter_fraction
            delay = max(0.0, base + self._random.uniform(-jitter, jitter))
        await asyncio.sleep(delay)
