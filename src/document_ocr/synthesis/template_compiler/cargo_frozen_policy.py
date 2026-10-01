"""An explicit, source-pinned exception for rare cargo that cannot be resampled.

This is a capability contract, not a source-copy fallback. Only the cargo graph
is fixed. The ordinary party, document-identity and equipment-identity variation
requirements remain in force, and every published target must reprove the graph.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from document_ocr.hashing import canonical_json_bytes, sha256_bytes


class CargoFrozenSourcePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    mode: Literal["cargo_frozen_source"]
    document_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cargo_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixed_source_binding_keys: tuple[str, ...] = ()


class CargoFrozenReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    document_id: str = Field(min_length=1)
    decision: Literal["pass"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cargo_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rationale: str = Field(min_length=32)
    source_evidence: tuple[str, ...] = Field(min_length=2)
    fixed_source_binding_keys: tuple[str, ...] = ()


_CARGO_KEYS = ("cargoGroups", "cargoPackages", "cargoAllocationGroups", "containers")


def _cargo_graph(target: Mapping[str, Any]) -> dict[str, Any]:
    patch = target["documentPatch"]
    if not isinstance(patch, dict):
        raise ValueError("frozen cargo target lacks a document patch")
    graph = {
        key: json.loads(canonical_json_bytes(patch[key]))
        for key in _CARGO_KEYS
        if key in patch
    }
    if not graph.get("cargoGroups"):
        raise ValueError("frozen cargo target lacks goods to protect")
    containers = graph.get("containers", [])
    if not isinstance(containers, list):
        raise ValueError("frozen cargo containers are not a list")
    identifiers = [row.get("containerNumber") for row in containers]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("frozen cargo container lacks its identity")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("frozen cargo has duplicate container identities")
    by_identity = {value: index for index, value in enumerate(identifiers)}
    for index, row in enumerate(containers):
        row["containerNumber"] = f"container-index:{index}"
        # Seal values are independently generated; their printed cardinality is not.
        if "sealNumbers" in row:
            seals = row["sealNumbers"]
            if not isinstance(seals, list) or any(
                not isinstance(value, str) or not value for value in seals
            ):
                raise ValueError("frozen cargo has invalid seal inventory")
            row["sealNumbers"] = [f"seal-index:{position}" for position in range(len(seals))]
    for group in graph.get("cargoAllocationGroups", []):
        for allocation in group.get("allocations", []):
            value = allocation.get("containerNumber")
            if value is not None:
                if value not in by_identity:
                    raise ValueError("cargo allocation refers to an unknown container")
                allocation["containerNumber"] = f"container-index:{by_identity[value]}"
    return graph


def cargo_graph_sha256(target: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_cargo_graph(target)))


def policy_sha256(policy: CargoFrozenSourcePolicy | None) -> str | None:
    return (
        sha256_bytes(canonical_json_bytes(policy.model_dump(mode="json")))
        if policy is not None
        else None
    )


def require_plan_mode(
    *, source_document_id: str, scenario_mode: str | None, policy: CargoFrozenSourcePolicy | None
) -> None:
    if (scenario_mode == "cargo_frozen_source") != (policy is not None):
        raise ValueError(
            "sample plan cargo_frozen_source mode differs from certified source policy: "
            + source_document_id
        )


def locked_description_paths(
    source: Mapping[str, Any], policy: CargoFrozenSourcePolicy | None
) -> frozenset[str]:
    if policy is None:
        return frozenset()
    return frozenset(
        f"documentPatch.cargoGroups[{index}].description"
        for index, _ in enumerate(source["documentPatch"].get("cargoGroups", []))
    )


def require_frozen_cargo(
    source: Mapping[str, Any], target: Mapping[str, Any], policy: CargoFrozenSourcePolicy
) -> None:
    expected = cargo_graph_sha256(source)
    if expected != policy.cargo_graph_sha256:
        raise ValueError("frozen cargo policy is not pinned to the source graph")
    if cargo_graph_sha256(target) != expected:
        raise ValueError("cargo_frozen_source changed a source-proven cargo fact or topology")


def load_case_policy(
    case_dir: Path,
    *,
    document_id: str,
    source: bytes,
    source_target: Mapping[str, Any],
    latest_target: Mapping[str, Any],
    template_bytes: bytes,
) -> CargoFrozenSourcePolicy | None:
    path = case_dir / "cargo-frozen-source-policy.json"
    if not path.exists():
        return None
    review_path = case_dir / "cargo-frozen-source-review.json"
    if any(p.is_symlink() or not p.is_file() for p in (path, review_path)):
        raise ValueError("frozen cargo policy or its review is not a regular case file")
    policy = CargoFrozenSourcePolicy.model_validate_json(path.read_bytes(), strict=True)
    review_bytes = review_path.read_bytes()
    review = CargoFrozenReview.model_validate_json(review_bytes, strict=True)
    evidence = {
        "document_id": document_id,
        "source_sha256": sha256_bytes(source),
        "source_target_sha256": sha256_bytes(canonical_json_bytes(source_target)),
        "template_sha256": sha256_bytes(template_bytes),
        "cargo_graph_sha256": cargo_graph_sha256(latest_target),
    }
    if any(
        getattr(policy, key) != value or getattr(review, key) != value
        for key, value in evidence.items()
    ):
        raise ValueError("frozen cargo policy/review differs from the exact source case")
    if policy.review_sha256 != sha256_bytes(review_bytes):
        raise ValueError("frozen cargo review receipt hash differs")
    source_text = source.decode("utf-8")
    if any(
        not fragment.strip() or fragment not in source_text
        for fragment in review.source_evidence
    ):
        raise ValueError("frozen cargo review cites absent source evidence")
    if policy.fixed_source_binding_keys != review.fixed_source_binding_keys:
        raise ValueError("frozen cargo fixed source binding list differs from review")
    if len(set(policy.fixed_source_binding_keys)) != len(policy.fixed_source_binding_keys):
        raise ValueError("frozen cargo fixed source bindings are duplicated")
    if policy.fixed_source_binding_keys:
        from .models import CertifiedSemanticTemplate

        template = CertifiedSemanticTemplate.model_validate_json(template_bytes, strict=True)
        bindings = {binding.logical_key: binding for binding in template.bindings}
        for key in policy.fixed_source_binding_keys:
            binding = bindings.get(key)
            if (
                binding is None
                or binding.group_kind != "cargo"
                or binding.value_kind != "operational_text"
                or binding.target_paths
                or binding.dependency_paths
                or binding.dependency_bindings
                or binding.derivation is not None
                or not binding.occurrences
                or any(
                    slot.source_text not in review.source_evidence
                    for slot in binding.occurrences
                )
            ):
                raise ValueError("frozen source-only cargo binding lacks exact review: " + key)
    return policy
