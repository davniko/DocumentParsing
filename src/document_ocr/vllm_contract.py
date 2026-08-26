"""Authenticated runtime-contract endpoint for the pinned vLLM service.

vLLM's public readiness endpoints do not expose the resolved speculative,
scheduler, model-revision, or cache configuration.  The Compose service mounts
this module and registers :func:`runtime_contract_middleware` through vLLM's
documented ``--middleware`` hook.  The endpoint is intentionally narrow and
calls the inner application first, so vLLM's API-key middleware still owns
authentication.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
from pathlib import Path
from typing import Any

from document_ocr.atomic import ArtifactReadError, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes

RUNTIME_CONTRACT_PATH = "/document-ocr/server-contract"
RUNTIME_CONTRACT_SCHEMA_VERSION = 4
_BUILD_MANIFEST_ENV = "DOCUMENT_OCR_VLLM_BUILD_MANIFEST"
_IMAGE_PATTERN = re.compile(r"^[^\s]+@sha256:[0-9a-f]{64}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_BUILD_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "image",
        "base_image",
        "vllm_version",
        "patches",
    }
)
_PATCH_KEYS = frozenset({"source", "commit", "sha256", "targets"})
_PATCH_TARGET_KEYS = frozenset({"path", "before_sha256", "after_sha256"})


class RuntimeContractError(RuntimeError):
    """The running vLLM process cannot prove the required server contract."""


def runtime_contract_payload(
    *,
    model: str,
    served_model_name: str,
    model_revision: str,
    dtype: str,
    quantization: str,
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    generation_config: str,
    speculative_method: str,
    num_speculative_tokens: int,
    image_limit_per_prompt: int,
    container_base_image: str,
    container_build_manifest_sha256: str,
) -> dict[str, object]:
    """Build the single canonical field set hashed by server and clients."""

    return {
        "schema_version": RUNTIME_CONTRACT_SCHEMA_VERSION,
        "model": model,
        "served_model_name": served_model_name,
        "model_revision": model_revision,
        "dtype": dtype,
        "quantization": quantization,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "generation_config": generation_config,
        "speculative_method": speculative_method,
        "num_speculative_tokens": num_speculative_tokens,
        "image_limit_per_prompt": image_limit_per_prompt,
        "container_base_image": container_base_image,
        "container_build_manifest_sha256": container_build_manifest_sha256,
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


def _model_dtype(value: object) -> str:
    if value == "bfloat16":
        return "bfloat16"
    if type(value).__module__ == "torch" and str(value) == "torch.bfloat16":
        return "bfloat16"
    raise RuntimeContractError("vLLM runtime model dtype must be torch.bfloat16")


def _model_quantization(value: object) -> str:
    if value is None:
        return "none"
    if value == "fp8":
        return "fp8"
    raise RuntimeContractError("vLLM runtime quantization must be disabled or 'fp8'")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _load_build_manifest() -> tuple[dict[str, object], str]:
    manifest_path_value = os.environ.get(_BUILD_MANIFEST_ENV)
    if manifest_path_value is None:
        raise RuntimeContractError(f"{_BUILD_MANIFEST_ENV} must name the baked build manifest")
    manifest_path = Path(manifest_path_value)
    if not manifest_path.is_absolute():
        raise RuntimeContractError(f"{_BUILD_MANIFEST_ENV} must be an absolute path")
    try:
        raw_manifest = read_regular_file_bytes(manifest_path)
    except ArtifactReadError as error:
        raise RuntimeContractError("vLLM build manifest is not a safe regular file") from error
    try:
        manifest = json.loads(
            raw_manifest.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeContractError("vLLM build manifest is not strict UTF-8 JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != _BUILD_MANIFEST_KEYS:
        raise RuntimeContractError("vLLM build manifest has an invalid field set")
    schema_version = manifest["schema_version"]
    if isinstance(schema_version, bool) or schema_version != 2:
        raise RuntimeContractError("vLLM build manifest has an unsupported schema version")

    string_fields = _BUILD_MANIFEST_KEYS - {"schema_version", "patches"}
    if any(not isinstance(manifest[field], str) or not manifest[field] for field in string_fields):
        raise RuntimeContractError("vLLM build manifest has an invalid string field")
    base_image = str(manifest["base_image"])
    if not _IMAGE_PATTERN.fullmatch(base_image):
        raise RuntimeContractError("vLLM build manifest base image is not digest-pinned")

    patches = manifest["patches"]
    if not isinstance(patches, list) or not patches:
        raise RuntimeContractError("vLLM build manifest patches must be a non-empty list")
    patch_commits: set[str] = set()
    target_paths: set[Path] = set()
    for patch_index, patch in enumerate(patches):
        if not isinstance(patch, dict) or set(patch) != _PATCH_KEYS:
            raise RuntimeContractError(
                f"vLLM build manifest patch {patch_index} has an invalid field set"
            )
        if any(
            not isinstance(patch[field], str) or not patch[field]
            for field in _PATCH_KEYS - {"targets"}
        ):
            raise RuntimeContractError(
                f"vLLM build manifest patch {patch_index} has an invalid string field"
            )
        patch_commit = str(patch["commit"])
        if not _GIT_SHA_PATTERN.fullmatch(patch_commit):
            raise RuntimeContractError(
                f"vLLM build manifest patch {patch_index} commit is not immutable"
            )
        if patch_commit in patch_commits:
            raise RuntimeContractError("vLLM build manifest has duplicate patch commits")
        patch_commits.add(patch_commit)
        expected_patch_source = "https://github.com/vllm-project/vllm/commit/" + patch_commit
        if patch["source"] != expected_patch_source:
            raise RuntimeContractError(
                f"vLLM build manifest patch {patch_index} source does not match its commit"
            )
        if not _SHA256_PATTERN.fullmatch(str(patch["sha256"])):
            raise RuntimeContractError(
                f"vLLM build manifest patch {patch_index} sha256 is not a SHA-256"
            )

        targets = patch["targets"]
        if not isinstance(targets, list) or not targets:
            raise RuntimeContractError(
                f"vLLM build manifest patch {patch_index} targets must be a non-empty list"
            )
        for target_index, target in enumerate(targets):
            if not isinstance(target, dict) or set(target) != _PATCH_TARGET_KEYS:
                raise RuntimeContractError(
                    "vLLM build manifest patch "
                    f"{patch_index} target {target_index} has an invalid field set"
                )
            if any(
                not isinstance(target[field], str) or not target[field]
                for field in _PATCH_TARGET_KEYS
            ):
                raise RuntimeContractError(
                    "vLLM build manifest patch "
                    f"{patch_index} target {target_index} has an invalid string field"
                )
            for field in ("before_sha256", "after_sha256"):
                if not _SHA256_PATTERN.fullmatch(str(target[field])):
                    raise RuntimeContractError(
                        "vLLM build manifest patch "
                        f"{patch_index} target {target_index} {field} is not a SHA-256"
                    )
            target_path = Path(str(target["path"]))
            if not target_path.is_absolute():
                raise RuntimeContractError(
                    "vLLM build manifest patch "
                    f"{patch_index} target {target_index} path is not absolute"
                )
            if target_path in target_paths:
                raise RuntimeContractError("vLLM build manifest has duplicate target paths")
            target_paths.add(target_path)
            try:
                target_bytes = read_regular_file_bytes(target_path)
            except ArtifactReadError as error:
                raise RuntimeContractError(
                    "patched vLLM target is not a safe regular file"
                ) from error
            if sha256_bytes(target_bytes) != target["after_sha256"]:
                raise RuntimeContractError(
                    "installed vLLM target does not match the build manifest"
                )
    try:
        installed_vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeContractError("vLLM distribution is not installed") from error
    if installed_vllm_version != manifest["vllm_version"]:
        raise RuntimeContractError("installed vLLM version does not match the build manifest")
    return manifest, sha256_bytes(raw_manifest)


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

    build_manifest, build_manifest_sha256 = _load_build_manifest()

    payload = runtime_contract_payload(
        model=_string(_attribute(model, "model"), "model"),
        served_model_name=_string(_attribute(model, "served_model_name"), "served_model_name"),
        model_revision=_string(_attribute(model, "revision"), "model_revision"),
        dtype=_model_dtype(_attribute(model, "dtype")),
        quantization=_model_quantization(_attribute(model, "quantization")),
        max_model_len=_positive_integer(_attribute(model, "max_model_len"), "max_model_len"),
        max_num_batched_tokens=_positive_integer(
            _attribute(scheduler, "max_num_batched_tokens"), "max_num_batched_tokens"
        ),
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
        container_base_image=str(build_manifest["base_image"]),
        container_build_manifest_sha256=build_manifest_sha256,
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
