"""Authenticated runtime-contract endpoint for the pinned vLLM service.

vLLM's public readiness endpoints do not expose the resolved speculative,
scheduler, model-revision, or cache configuration.  The Compose service mounts
this module and registers :func:`runtime_contract_middleware` through vLLM's
documented ``--middleware`` hook.  The endpoint is intentionally narrow and
calls the inner application first, so vLLM's API-key middleware still owns
authentication.
"""

from __future__ import annotations

import os
import re
from typing import Any

from document_ocr.hashing import canonical_json_bytes, sha256_bytes

RUNTIME_CONTRACT_PATH = "/document-ocr/server-contract"
RUNTIME_CONTRACT_SCHEMA_VERSION = 1
_IMAGE_PATTERN = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")


class RuntimeContractError(RuntimeError):
    """The running vLLM process cannot prove the required server contract."""


def runtime_contract_payload(
    *,
    model: str,
    served_model_name: str,
    model_revision: str,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    generation_config: str,
    speculative_method: str,
    num_speculative_tokens: int,
    image_limit_per_prompt: int,
    container_image: str,
) -> dict[str, object]:
    """Build the single canonical field set hashed by server and clients."""

    return {
        "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "model": model,
        "served_model_name": served_model_name,
        "model_revision": model_revision,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "generation_config": generation_config,
        "speculative_method": speculative_method,
        "num_speculative_tokens": num_speculative_tokens,
        "image_limit_per_prompt": image_limit_per_prompt,
        "container_image": container_image,
    }


def runtime_contract_sha256(payload: dict[str, object]) -> str:
    """Hash a canonical runtime payload for provenance and attestation."""

    return sha256_bytes(canonical_json_bytes(payload))


def _attribute(owner: object, name: str) -> Any:
    try:
        return getattr(owner, name)
    except AttributeError as error:
        raise RuntimeContractError(f"vLLM runtime has no required {name!r} field") from error


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeContractError(f"vLLM runtime field {name!r} must be a non-empty string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeContractError(f"vLLM runtime field {name!r} must be a positive integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeContractError(f"vLLM runtime field {name!r} must be numeric")
    result = float(value)
    if not 0.0 < result <= 1.0:
        raise RuntimeContractError(f"vLLM runtime field {name!r} must be in (0, 1]")
    return result


def build_runtime_contract(state: object) -> dict[str, object]:
    """Read resolved settings from vLLM application state and hash the claim."""

    vllm_config = _attribute(state, "vllm_config")
    model = _attribute(vllm_config, "model_config")
    scheduler = _attribute(vllm_config, "scheduler_config")
    cache = _attribute(vllm_config, "cache_config")
    speculative = _attribute(vllm_config, "speculative_config")
    if speculative is None:
        raise RuntimeContractError("vLLM runtime has speculative decoding disabled")
    multimodal = _attribute(model, "multimodal_config")
    if multimodal is None:
        raise RuntimeContractError("vLLM runtime has no multimodal configuration")
    get_image_limit = _attribute(multimodal, "get_limit_per_prompt")
    if not callable(get_image_limit):
        raise RuntimeContractError("vLLM multimodal image-limit accessor is not callable")

    container_image = os.environ.get("DOCUMENT_OCR_VLLM_IMAGE")
    if container_image is None or not _IMAGE_PATTERN.fullmatch(container_image):
        raise RuntimeContractError(
            "DOCUMENT_OCR_VLLM_IMAGE must contain the digest-pinned running image claim"
        )

    payload = runtime_contract_payload(
        model=_string(_attribute(model, "model"), "model"),
        served_model_name=_string(_attribute(model, "served_model_name"), "served_model_name"),
        model_revision=_string(_attribute(model, "revision"), "model_revision"),
        max_model_len=_positive_integer(_attribute(model, "max_model_len"), "max_model_len"),
        max_num_seqs=_positive_integer(_attribute(scheduler, "max_num_seqs"), "max_num_seqs"),
        gpu_memory_utilization=_positive_float(
            _attribute(cache, "gpu_memory_utilization"), "gpu_memory_utilization"
        ),
        generation_config=_string(_attribute(model, "generation_config"), "generation_config"),
        speculative_method=_string(_attribute(speculative, "method"), "speculative_method"),
        num_speculative_tokens=_positive_integer(
            _attribute(speculative, "num_speculative_tokens"), "num_speculative_tokens"
        ),
        image_limit_per_prompt=_positive_integer(
            get_image_limit("image"), "image_limit_per_prompt"
        ),
        container_image=container_image,
    )
    return {
        **payload,
        "contract_sha256": runtime_contract_sha256(payload),
    }


async def runtime_contract_middleware(request: Any, call_next: Any) -> Any:
    """Serve the runtime claim only after vLLM has authenticated the request."""

    if request.url.path != RUNTIME_CONTRACT_PATH:
        return await call_next(request)

    response = await call_next(request)
    if response.status_code != 404:
        return response

    # Starlette is supplied by the vLLM image; keeping this import lazy avoids
    # adding the GPU server's web stack to the CPU dataset environment.
    from starlette.responses import JSONResponse  # type: ignore[import-not-found]

    return JSONResponse(build_runtime_contract(request.app.state))
