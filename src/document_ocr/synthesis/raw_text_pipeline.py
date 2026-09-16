"""Production orchestration for carrier-bound compiled-template synthesis."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.template_compiler.descendant import (
    load_descendant_config,
    preflight_descendants,
    run_descendants,
)
from document_ocr.synthesis.template_compiler.descendant_models import DescendantConfig
from document_ocr.synthesis.template_compiler.pipeline import project_root_from_config


def _validate_project_root(*, project_root: Path, config_path: Path) -> Path:
    resolved = project_root.resolve(strict=True)
    detected = project_root_from_config(config_path)
    if resolved != detected:
        raise ValueError(
            f"configured project root differs from the config repository: {resolved} != {detected}"
        )
    return resolved


def _validate_pipeline_config_file(
    *, config_path: Path, config: DescendantConfig
) -> DescendantConfig:
    """Prove that the supplied immutable config object matches the real config file."""

    loaded = load_descendant_config(config_path)
    if loaded != config:
        raise ValueError("raw-text pipeline config object differs from its source file")
    return loaded


def preflight_raw_text_pipeline(
    *,
    project_root: Path,
    config_path: Path,
    config: DescendantConfig,
) -> dict[str, Any]:
    """Validate the committed template/target lineage and deterministic render plan."""

    _validate_project_root(project_root=project_root, config_path=config_path)
    _validate_pipeline_config_file(config_path=config_path, config=config)
    return preflight_descendants(config_path)


def run_raw_text_pipeline(
    *,
    project_root: Path,
    config_path: Path,
    config: DescendantConfig,
) -> dict[str, Any]:
    """Render and publish one complete carrier-bound compiled-template cohort."""

    _validate_project_root(project_root=project_root, config_path=config_path)
    _validate_pipeline_config_file(config_path=config_path, config=config)
    artifact_root = asyncio.run(run_descendants(config_path))
    summary_path = artifact_root / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    if not isinstance(summary, dict):
        raise ValueError("compiled raw-text pipeline summary is not an object")
    summary["artifactRoot"] = str(artifact_root)
    summary["commitSha256"] = sha256_file(artifact_root / "_COMMIT.json")
    return summary
