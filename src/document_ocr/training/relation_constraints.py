"""Publish one hash-pinned vocabulary contract from canonical training targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.semantic_v3.transform import CategoryRegistry
from document_ocr.training.config import (
    DatasetFileConfig,
    RelationConstraintsBuildConfig,
    resolve_config_path,
)
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task


def _reject_json_constant(constant: str) -> Any:
    raise ValueError(f"non-standard JSON constant {constant!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _pinned_file(project_root: Path, configured_path: str, expected_sha256: str) -> Path:
    unresolved = resolve_config_path(project_root, configured_path)
    if unresolved.is_symlink():
        raise ValueError(f"pinned input must not be a symbolic link: {unresolved}")
    try:
        path = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"pinned input does not exist: {unresolved}") from error
    if not path.is_file():
        raise ValueError(f"pinned input is not a regular file: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"pinned input SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )
    return path


def _output_directory(project_root: Path, configured: str) -> Path:
    unresolved = resolve_config_path(project_root, configured)
    if unresolved == Path(unresolved.anchor):
        raise ValueError("constraints output directory must not be a filesystem root")
    current = Path(unresolved.anchor) if unresolved.is_absolute() else project_root
    parts = unresolved.parts[1:] if unresolved.is_absolute() else Path(configured).parts
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"constraints output path must not traverse a symlink: {current}")
    path = unresolved.resolve()
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"constraints output must be a real directory: {path}")
    return path


def _source_tokens(
    *,
    project_root: Path,
    source: DatasetFileConfig,
    target_field: str,
    task_name: str,
) -> tuple[set[str], set[str]]:
    path = _pinned_file(project_root, source.path, source.sha256)
    payload = read_regular_file_bytes(path)
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"constraints source JSONL must end with a newline: {path}")
    task = get_training_task(task_name)
    package_tokens: set[str] = set()
    container_tokens: set[str] = set()
    records = 0
    for line_number, encoded_line in enumerate(payload.splitlines(), start=1):
        context = f"{path}:{line_number}"
        if not encoded_line.strip():
            raise ValueError(f"{context}: blank JSONL lines are forbidden")
        try:
            value = json.loads(
                encoded_line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"{context}: invalid strict JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{context}: record must be an object")
        target = value.get(target_field)
        if not isinstance(target, dict):
            raise ValueError(f"{context}: {target_field} must be an object")
        try:
            canonical = task.canonicalize(cast(dict[str, Any], target))
        except ValueError as error:
            raise ValueError(f"{context}: target is not canonical {task_name}: {error}") from error
        patch = canonical["documentPatch"]
        for package in patch.get("cargoPackages", []):
            token = package.get("typeCategory")
            if token is not None:
                package_tokens.add(cast(str, token))
        for container in patch.get("containers", []):
            token = container.get("typeCategory")
            if token is not None:
                container_tokens.add(cast(str, token))
        records += 1
    if records != source.records:
        raise ValueError(
            f"constraints source count mismatch for {path}: "
            f"expected {source.records}, found {records}"
        )
    return package_tokens, container_tokens


def build_relation_constraints(
    *,
    project_root: Path,
    config: RelationConstraintsBuildConfig,
    config_sha256: str,
) -> dict[str, Any]:
    """Validate every target and publish its exact union vocabulary."""

    package_registry_path = _pinned_file(
        project_root,
        config.package_registry.path,
        config.package_registry.sha256,
    )
    _pinned_file(
        project_root,
        config.container_registry.path,
        config.container_registry.sha256,
    )
    package_tokens: set[str] = set()
    container_tokens: set[str] = set()
    for source in config.sources:
        source_package, source_container = _source_tokens(
            project_root=project_root,
            source=source,
            target_field=config.target_field,
            task_name=config.task,
        )
        package_tokens.update(source_package)
        container_tokens.update(source_container)

    try:
        package_registry = CategoryRegistry.model_validate_json(
            read_regular_file_bytes(package_registry_path), strict=True
        )
    except ValueError as error:
        raise ValueError("package registry failed schema validation") from error
    if package_registry.registryKind != "package":
        raise ValueError("package registry has the wrong registryKind")
    registered_package_tokens = {entry.categoryToken for entry in package_registry.entries}
    unregistered_package_tokens = sorted(package_tokens - registered_package_tokens)
    if unregistered_package_tokens:
        raise ValueError(
            "training targets contain package categories outside the pinned registry: "
            f"{unregistered_package_tokens}"
        )

    task = get_training_task(config.task)
    constraints = RelationExplicitTaskConstraints.model_validate(
        {
            "schemaVersion": 1,
            "task": config.task,
            "basePromptSchemaSha256": task.base_prompt_schema_sha256(),
            "targetSchemaSha256": sha256_bytes(
                canonical_json_bytes(task.target_model.model_json_schema(mode="serialization"))
            ),
            "packageRegistrySha256": config.package_registry.sha256,
            "containerRegistrySha256": config.container_registry.sha256,
            "packageCategoryTokens": tuple(sorted(package_tokens)),
            "containerCategoryTokens": tuple(sorted(container_tokens)),
        },
        strict=True,
    )
    constraints_payload = canonical_json_bytes(constraints.model_dump(mode="json")) + b"\n"
    output_dir = _output_directory(project_root, config.output_dir)
    constraints_path = output_dir / "task-constraints.json"
    atomic_publish_bytes(constraints_path, constraints_payload)
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "publicationStatus": "complete",
        "task": config.task,
        "configSha256": config_sha256,
        "sources": [source.model_dump(mode="json") for source in config.sources],
        "registries": {
            "package": config.package_registry.model_dump(mode="json"),
            "container": config.container_registry.model_dump(mode="json"),
        },
        "output": {
            "path": constraints_path.name,
            "sha256": sha256_bytes(constraints_payload),
            "bytes": len(constraints_payload),
            "packageCategoryTokens": len(package_tokens),
            "containerCategoryTokens": len(container_tokens),
        },
    }
    atomic_publish_json(output_dir / "manifest.json", manifest)
    return manifest
