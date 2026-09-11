"""Receipt-bound deterministic inputs and replay for raw-text certification v3.

The semantic model has quarantine authority only.  This module owns the executable checks that
can be proved from compiler artifacts and pinned public references, and keeps loading/replay
identical across certification, correction, publication, and offline evaluation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.config import RawTextCertificationInvariantInputsConfig
from document_ocr.synthesis.raw_text_certification_invariants import (
    CertificationInvariantAudit,
    CertificationInvariantEnvelope,
    DeterministicLineRepair,
    compile_and_audit_certification_invariants,
)
from document_ocr.synthesis.raw_text_certification_references import (
    CertificationReferenceIndex,
    load_certification_reference_index,
)
from document_ocr.synthesis.raw_text_inventory import (
    DeterministicInventoryEdit,
    InventoryCandidate,
)
from document_ocr.synthesis.raw_text_inventory_probe import _validate_reference_run
from document_ocr.synthesis.transport_capacity import TransportCapacityLimits, capacity_limits
from document_ocr.training.config import resolve_config_path


@dataclass(frozen=True, slots=True)
class CertificationInvariantContext:
    """One run-scoped, fully validated reference and capacity context."""

    references: CertificationReferenceIndex
    capacity_limits: TransportCapacityLimits


@dataclass(frozen=True, slots=True)
class CertificationInvariantCase:
    """Parsed compiler inputs plus their deterministic decision for one candidate."""

    inventory: tuple[InventoryCandidate, ...]
    deterministic_edits: tuple[DeterministicInventoryEdit, ...]
    envelope: CertificationInvariantEnvelope
    audit: CertificationInvariantAudit


def complete_deterministic_repair_set(
    audit: CertificationInvariantAudit,
) -> tuple[DeterministicLineRepair, ...] | None:
    """Return one complete repair set, or ``None`` when any defect lacks write authority.

    A repair is accepted only when the finding that carries it also cites the physical line.
    Duplicate identical repairs are collapsed, while conflicting repair claims remain a hard
    contract error instead of being resolved by order.
    """

    if audit.passed:
        return ()
    if any(not finding.repairs for finding in audit.findings):
        return None
    unique: dict[tuple[str, str, str, str, str], DeterministicLineRepair] = {}
    claimed_fragments: dict[tuple[str, str], tuple[str, str]] = {}
    for finding in audit.findings:
        evidence_lines = {row.lineId for row in finding.evidence}
        for repair in finding.repairs:
            if repair.lineId not in evidence_lines:
                raise ValueError(
                    "deterministic repair line is not cited by its certification finding"
                )
            identity = (
                repair.lineId,
                repair.expectedCurrentLineSha256,
                repair.oldFragment,
                repair.newFragment,
                repair.method,
            )
            unique[identity] = repair
            fragment_key = (repair.lineId, repair.oldFragment)
            replacement_identity = (
                repair.expectedCurrentLineSha256,
                repair.newFragment,
            )
            prior = claimed_fragments.setdefault(fragment_key, replacement_identity)
            if prior != replacement_identity:
                raise ValueError("deterministic findings claim conflicting line repairs")
    return tuple(
        unique[key]
        for key in sorted(
            unique,
            key=lambda row: (int(row[0][1:]), row[2], row[3], row[1], row[4]),
        )
    )


def apply_deterministic_line_repairs(
    candidate: str,
    repairs: Sequence[DeterministicLineRepair],
) -> str:
    """Apply a complete host-authored repair set atomically to exact physical lines."""

    physical_lines = candidate.splitlines(keepends=True)
    if not physical_lines and repairs:
        raise ValueError("deterministic repairs cannot address an empty candidate")
    grouped: dict[int, list[DeterministicLineRepair]] = defaultdict(list)
    for repair in repairs:
        number = int(repair.lineId[1:])
        if not 1 <= number <= len(physical_lines):
            raise ValueError(f"deterministic repair line is outside the candidate: {repair.lineId}")
        grouped[number].append(repair)

    output = list(physical_lines)
    for number, line_repairs in sorted(grouped.items()):
        physical = physical_lines[number - 1]
        ending = ""
        body = physical
        for candidate_ending in ("\r\n", "\n", "\r"):
            if physical.endswith(candidate_ending):
                body = physical[: -len(candidate_ending)]
                ending = candidate_ending
                break
        expected_hashes = {row.expectedCurrentLineSha256 for row in line_repairs}
        actual_hash = sha256_bytes(body.encode("utf-8"))
        if expected_hashes != {actual_hash}:
            raise ValueError(
                f"deterministic repair line identity differs from the candidate: L{number:05d}"
            )
        spans: list[tuple[int, int, DeterministicLineRepair]] = []
        for repair in line_repairs:
            if body.count(repair.oldFragment) != 1:
                raise ValueError(f"deterministic repair fragment is not unique on {repair.lineId}")
            start = body.index(repair.oldFragment)
            spans.append((start, start + len(repair.oldFragment), repair))
        ordered_spans = sorted(spans, key=lambda row: (row[0], row[1]))
        if any(left[1] > right[0] for left, right in pairwise(ordered_spans)):
            raise ValueError(f"deterministic repairs overlap on L{number:05d}")
        repaired = body
        for start, stop, repair in reversed(ordered_spans):
            repaired = repaired[:start] + repair.newFragment + repaired[stop:]
        output[number - 1] = repaired + ending
    return "".join(output)


def _read_json_array(path: Path) -> list[Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"certification invariant input is not a regular file: {path}")
    try:
        value = json.loads(read_regular_file_bytes(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"certification invariant input is not valid JSON: {path}") from error
    if not isinstance(value, list):
        raise ValueError(f"certification invariant input is not a JSON array: {path}")
    return value


def load_certification_invariant_context(
    *,
    project_root: Path,
    config: RawTextCertificationInvariantInputsConfig,
) -> CertificationInvariantContext:
    """Load every pinned external fact once, before processing any candidate."""

    geonames_root = _validate_reference_run(
        project_root,
        config.geonames_registry.path,
        config.geonames_registry.commit_sha256,
        config.geonames_registry.transaction_sha256,
    )
    references = load_certification_reference_index(
        iso3166_path=resolve_config_path(project_root, config.iso3166_snapshot.path),
        expected_iso3166_sha256=config.iso3166_snapshot.sha256,
        geonames_registry_root=geonames_root,
        expected_geonames_receipt_sha256=config.geonames_registry_receipt_sha256,
        expected_phonenumberslite_version=config.phonenumberslite_version,
        package_registry_path=resolve_config_path(project_root, config.package_registry.path),
        expected_package_registry_sha256=config.package_registry.sha256,
        expected_package_registry_entries=config.package_registry_entries,
    )
    return CertificationInvariantContext(
        references=references,
        capacity_limits=capacity_limits(config.transport_capacity),
    )


def load_certification_invariant_case(
    *,
    case_root: Path,
    document_id: str,
    source: str,
    candidate: str,
    contract: Mapping[str, Any],
    context: CertificationInvariantContext,
) -> CertificationInvariantCase:
    """Parse compiler-owned inputs and execute the exact deterministic audit."""

    inventory = tuple(
        InventoryCandidate.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in _read_json_array(case_root / "inventory.json")
    )
    deterministic_edits = tuple(
        DeterministicInventoryEdit.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in _read_json_array(case_root / "deterministic-edits.json")
    )
    return evaluate_certification_invariant_case(
        document_id=document_id,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=deterministic_edits,
        context=context,
    )


def evaluate_certification_invariant_case(
    *,
    document_id: str,
    source: str,
    candidate: str,
    contract: Mapping[str, Any],
    inventory: tuple[InventoryCandidate, ...],
    deterministic_edits: tuple[DeterministicInventoryEdit, ...],
    context: CertificationInvariantContext,
) -> CertificationInvariantCase:
    """Compile and audit one in-memory case without provider access."""

    envelope, audit = compile_and_audit_certification_invariants(
        document_id=document_id,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=deterministic_edits,
        references=context.references,
        capacity_limits=context.capacity_limits,
    )
    return CertificationInvariantCase(
        inventory=inventory,
        deterministic_edits=deterministic_edits,
        envelope=envelope,
        audit=audit,
    )


def replay_certification_invariant_case(
    *,
    document_id: str,
    source: str,
    candidate: str,
    contract: Mapping[str, Any],
    inventory_value: object,
    deterministic_edits_value: object,
    envelope_value: object,
    audit_value: object,
    context: CertificationInvariantContext,
) -> CertificationInvariantCase:
    """Recompute and compare every persisted v3 deterministic artifact."""

    if not isinstance(inventory_value, list) or not isinstance(deterministic_edits_value, list):
        raise ValueError("certification v3 compiler artifacts are not JSON arrays")
    inventory = tuple(
        InventoryCandidate.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in inventory_value
    )
    deterministic_edits = tuple(
        DeterministicInventoryEdit.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in deterministic_edits_value
    )
    expected = evaluate_certification_invariant_case(
        document_id=document_id,
        source=source,
        candidate=candidate,
        contract=contract,
        inventory=inventory,
        deterministic_edits=deterministic_edits,
        context=context,
    )
    envelope = CertificationInvariantEnvelope.model_validate_json(
        canonical_json_bytes(envelope_value), strict=True
    )
    audit = CertificationInvariantAudit.model_validate_json(
        canonical_json_bytes(audit_value), strict=True
    )
    if envelope != expected.envelope or audit != expected.audit:
        raise ValueError("certification v3 invariant artifacts fail deterministic replay")
    if audit.envelopeSha256 != sha256_bytes(canonical_json_bytes(envelope.model_dump(mode="json"))):
        raise ValueError("certification v3 invariant audit is not bound to its envelope")
    return CertificationInvariantCase(
        inventory=inventory,
        deterministic_edits=deterministic_edits,
        envelope=envelope,
        audit=audit,
    )
