"""Fail-closed source-reviewed package projection for compiled B/L descendants.

The compiled source target retains all printed package levels for rendering.
This contract removes only reviewed outer/aggregate levels from the task-facing
training target, after rendering; it never changes the source or rendered text.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.training.package_projection import GroupDiagnosis, _project_relation_target

_SHA = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ReviewedPackageDecision:
    source_document_id: str
    source_label_sha256: str
    group_id: str
    source_package_ids: tuple[str, ...]
    allowed_target_package_id_sequences: frozenset[tuple[str, ...]]
    status: Literal["ready", "review"]
    retained_package_ids: tuple[str, ...] | None
    rationale: str


@dataclass(frozen=True, slots=True)
class ReviewedPackageContract:
    catalog_commit_sha256: str
    decisions: Mapping[tuple[str, str], ReviewedPackageDecision]


def load_reviewed_package_contract(
    path: Path, *, expected_sha256: str, catalog_commit_sha256: str
) -> ReviewedPackageContract:
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != expected_sha256:
        raise ValueError("reviewed package contract hash differs")
    document = json.loads(payload)
    if document["schemaVersion"] != 1 or document["catalogCommitSha256"] != catalog_commit_sha256:
        raise ValueError("reviewed package contract is not pinned to this catalog")
    rows = document["decisions"]
    if document["sourceGroupCount"] != len(rows):
        raise ValueError("reviewed package contract count differs")
    decisions = {}
    for row in rows:
        source_id, group_id = row["sourceDocumentId"], row["groupId"]
        key = (source_id, group_id)
        source_ids = tuple(row["sourcePackageIds"])
        allowed = frozenset(tuple(value) for value in row["allowedTargetPackageIdSequences"])
        status = row["status"]
        retained = (
            tuple(row["retainedPackageIds"])
            if row["retainedPackageIds"] is not None
            else None
        )
        if (
            key in decisions
            or _SHA.fullmatch(row["sourceLabelCanonicalSha256"]) is None
            or not source_ids
            or len(source_ids) != len(set(source_ids))
            or tuple(source_ids) not in allowed
            or status not in {"ready", "review"}
            or (status == "ready" and not retained)
            or (status == "review" and retained is not None)
            or (
                retained is not None
                and (
                    len(retained) != len(set(retained))
                    or any(not set(retained) <= set(sequence) for sequence in allowed)
                )
            )
            or not row["rationale"].strip()
        ):
            raise ValueError(f"invalid reviewed package decision: {source_id}/{group_id}")
        decisions[key] = ReviewedPackageDecision(
            source_document_id=source_id,
            source_label_sha256=row["sourceLabelCanonicalSha256"],
            group_id=group_id,
            source_package_ids=source_ids,
            allowed_target_package_id_sequences=allowed,
            status=status,
            retained_package_ids=retained,
            rationale=row["rationale"],
        )
    return ReviewedPackageContract(catalog_commit_sha256, decisions)


def project_reviewed_package_target(
    *,
    source_document_id: str,
    source_target: Mapping[str, Any],
    target: dict[str, Any],
    contract: ReviewedPackageContract | None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Project only proven package levels; reject missing or changed source roles."""

    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_target["documentPatch"].get("cargoPackages", []):
        source_groups[row["groupId"]].append(row)
    target_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target["documentPatch"].get("cargoPackages", []):
        target_groups[row["groupId"]].append(row)
    source_hash = sha256_bytes(canonical_json_bytes(source_target))
    diagnoses = []
    decisions_used = []
    for group in target["documentPatch"].get("cargoGroups", []):
        group_id = group["groupId"]
        source_rows = source_groups[group_id]
        target_rows = target_groups[group_id]
        target_ids = tuple(row["packageId"] for row in target_rows)
        if len(source_rows) > 1:
            decision = (
                contract.decisions.get((source_document_id, group_id))
                if contract is not None
                else None
            )
            if decision is None:
                raise ValueError(
                    f"multi-level package source lacks review: {source_document_id}/{group_id}"
                )
            if (
                source_hash != decision.source_label_sha256
                or tuple(row["packageId"] for row in source_rows) != decision.source_package_ids
            ):
                raise ValueError(
                    f"reviewed package source changed: {source_document_id}/{group_id}"
                )
            if decision.status == "review" or decision.retained_package_ids is None:
                raise ValueError(
                    f"package source requires review: {source_document_id}/{group_id}: "
                    f"{decision.rationale}"
                )
            if target_ids not in decision.allowed_target_package_id_sequences:
                raise ValueError(
                    f"unreviewed generated package topology: {source_document_id}/{group_id}"
                )
            retained = decision.retained_package_ids
            decisions_used.append(
                {
                    "groupId": group_id,
                    "retainedPackageIds": list(retained),
                    "metadataPackageIds": [value for value in target_ids if value not in retained],
                    "rationale": decision.rationale,
                }
            )
        else:
            if target_ids != tuple(row["packageId"] for row in source_rows):
                raise ValueError(
                    f"single-level package topology changed: {source_document_id}/{group_id}"
                )
            retained = target_ids
        diagnoses.append(
            GroupDiagnosis(
                group_id=group_id,
                status="projected" if retained != target_ids else "retained_non_hierarchical",
                reason="Source-reviewed package-level projection.",
                packages=(),
                retained_package_ids=retained,
                metadata_package_ids=tuple(value for value in target_ids if value not in retained),
            )
        )
    cargo_group_ids = {row.group_id for row in diagnoses}
    if set(target_groups) - cargo_group_ids or set(source_groups) - cargo_group_ids:
        raise ValueError(f"package references unknown cargo group: {source_document_id}")
    projected, allocation_audit = _project_relation_target(target, diagnoses)
    decisions_used.extend(
        {"allocationProjection": row} for row in allocation_audit if row["decision"] != "preserved"
    )
    return projected, tuple(decisions_used)
