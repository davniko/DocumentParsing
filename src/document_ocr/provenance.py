"""Resolved code, configuration, dependency, and runtime provenance."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from document_ocr.config import PipelineConfig
from document_ocr.hashing import canonical_json_sha256, identity_sha256, sha256_file

_RUNTIME_PACKAGES = (
    "aiosqlite",
    "boto3",
    "botocore",
    "document-ocr-pipeline",
    "httpx",
    "pillow",
    "pyarrow",
    "pydantic",
    "pypdfium2",
    "pyyaml",
)


def _required_file_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required reproducibility artifact does not exist: {path}")
    return sha256_file(path)


def _source_tree_hash(project_root: Path) -> str:
    source_root = project_root / "src" / "document_ocr"
    files = sorted(source_root.rglob("*.py"))
    if not files:
        raise RuntimeError(f"no Python pipeline source found under {source_root}")
    parts: list[str] = []
    for path in files:
        parts.extend((path.relative_to(project_root).as_posix(), sha256_file(path)))
    return identity_sha256("document-ocr-source-tree-v1", *parts)


def _run_git(project_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _git_provenance(project_root: Path) -> dict[str, Any]:
    inside = _run_git(project_root, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return {
            "state": "not-a-git-worktree",
            "commit": None,
            "dirty": None,
            "status_sha256": None,
        }

    head = _run_git(project_root, "rev-parse", "HEAD")
    commit = head.stdout.strip() if head.returncode == 0 else None
    status = _run_git(project_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0:
        raise RuntimeError(f"git status failed with exit code {status.returncode}")
    status_text = status.stdout
    return {
        "state": "worktree",
        "commit": commit,
        "dirty": bool(status_text),
        "status_sha256": canonical_json_sha256(status_text),
    }


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in _RUNTIME_PACKAGES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"locked runtime distribution is not installed: {distribution}"
            ) from error
    return versions


def _extraction_contract(config: PipelineConfig) -> dict[str, Any]:
    """Select settings that can change the bytes emitted for one page.

    Source/run/output/retry/concurrency settings are recorded in the resolved
    configuration but do not define an extraction algorithm.
    """

    return {
        "schema_version": config.schema_version,
        "raster": config.raster.model_dump(mode="json"),
        "model": {
            "repository": config.vllm.model,
            "served_model_name": config.vllm.served_model_name,
            "revision": config.vllm.revision,
            "engine": "vllm",
            "engine_version": config.vllm.engine_version,
            "container_base_image": config.vllm.container_base_image,
            "container_build_manifest_sha256": (config.vllm.container_build_manifest_sha256),
            "dtype": config.vllm.dtype,
            "quantization": config.vllm.quantization,
            "max_model_len": config.vllm.max_model_len,
            "max_num_batched_tokens": config.vllm.max_num_batched_tokens,
            "max_num_seqs": config.vllm.max_num_seqs,
            "gpu_memory_utilization": config.vllm.gpu_memory_utilization,
            "prompt": config.vllm.prompt,
            "sampling": config.vllm.sampling.model_dump(mode="json"),
            "speculative_decoding": config.vllm.speculative_decoding.model_dump(mode="json"),
            "repetition_detection": config.vllm.repetition_detection.model_dump(mode="json"),
        },
    }


def collect_runtime_provenance(
    project_root: Path, config: PipelineConfig
) -> tuple[dict[str, Any], str]:
    """Return a manifest-ready provenance record and 64-hex pipeline fingerprint."""

    resolved_config = config.model_dump(mode="json")
    package_versions = _installed_versions()
    provenance: dict[str, Any] = {
        "schema_version": 2,
        "resolved_config": resolved_config,
        "resolved_config_sha256": canonical_json_sha256(resolved_config),
        "extraction_contract": _extraction_contract(config),
        "source_tree_sha256": _source_tree_hash(project_root),
        "pyproject_sha256": _required_file_hash(project_root / "pyproject.toml"),
        "uv_lock_sha256": _required_file_hash(project_root / "uv.lock"),
        "compose_sha256": _required_file_hash(project_root / "compose.yaml"),
        "package_versions": package_versions,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git": _git_provenance(project_root),
    }
    fingerprint_inputs = {
        "schema": "document-ocr-pipeline-fingerprint-v2",
        "extraction_contract": provenance["extraction_contract"],
        "source_tree_sha256": provenance["source_tree_sha256"],
        "pyproject_sha256": provenance["pyproject_sha256"],
        "uv_lock_sha256": provenance["uv_lock_sha256"],
        "compose_sha256": provenance["compose_sha256"],
        "package_versions": package_versions,
        "python_version": provenance["python_version"],
        "platform": provenance["platform"],
    }
    return provenance, canonical_json_sha256(fingerprint_inputs)
