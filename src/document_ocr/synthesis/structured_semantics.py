"""Complete non-linguistic semantic realization for structured B/L baselines.

This module owns deterministic facts and graph reconciliation only.  Statistical
models may propose package quantities and empirical train-only rows may propose
dates, but neither is allowed to write task targets directly.  Every accepted
proposal is projected through these functions and revalidated against the exact
leaf diff and allocation arithmetic.
"""

from __future__ import annotations

import hashlib
import re
import string
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, cast

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.anchors import leaf_items
from document_ocr.synthesis.drafts import (
    set_target_value,
    validate_allocation_arithmetic,
    validate_change_ledger,
)
from document_ocr.synthesis.generation_models import PendingRealization, SemanticChange, json_value
from document_ocr.synthesis.generators import (
    iso6346_check_digit,
    largest_remainder_allocation,
    reconcile_allocation_group,
    surface_pattern,
    validate_mass_order,
)
from document_ocr.synthesis.policies import FIELD_POLICIES, policy_for_target_path
from document_ocr.synthesis.run_safety import (
    IdentifierCandidateContext,
    IdentifierReservation,
    IdentifierReservationBundle,
    IdentifierReservationRequest,
    identifier_corpus_sha256,
    reserve_global_identifiers,
)
from document_ocr.synthesis.structured_models import CargoGroupNumericProposal

_ALPHABETS = {"A": string.ascii_uppercase, "a": string.ascii_lowercase, "9": string.digits}
_EMBEDDED_NUMERIC_DATE = re.compile(
    r"(?<![0-9])(?P<a>[0-9]{1,2})(?P<separator>[-./])(?P<b>[0-9]{1,2})"
    r"(?P=separator)(?P<year>[0-9]{2}|[0-9]{4})(?![0-9])"
)


def canonical_formal_identifier(value: str) -> str:
    """Conservative collision key across punctuation and Unicode presentation variants."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    canonical = "".join(character for character in normalized if character.isalnum())
    if not canonical:
        raise ValueError("formal identifier has no alphanumeric canonical form")
    return canonical


@dataclass(frozen=True, slots=True)
class IdentifierRequestInventory:
    """All run-wide formal identifier requests and their original surfaces."""

    container_requests: tuple[IdentifierReservationRequest, ...]
    container_prefix_by_request: Mapping[str, str]
    generic_requests: tuple[IdentifierReservationRequest, ...]
    generic_source_by_request: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class IdentifierAllocationPlan:
    """Globally collision-free values for every selected formal identifier."""

    containers: IdentifierReservationBundle | None
    generic: IdentifierReservationBundle | None

    def values(self) -> dict[str, str]:
        rows: list[IdentifierReservation] = []
        if self.containers is not None:
            rows.extend(self.containers.reservations)
        if self.generic is not None:
            rows.extend(self.generic.reservations)
        return {row.request_key: row.identifier for row in rows}


def _iter_formal_identifiers(
    target: Mapping[str, Any], *, request_prefix: str
) -> tuple[tuple[str, str, str], ...]:
    """Return ``(request_key, kind, value)`` in target order."""

    patch = cast(Mapping[str, Any], target["documentPatch"])
    rows: list[tuple[str, str, str]] = []
    for field in (
        "billOfLadingNumber",
        "originalBillOfLadingNumber",
        "masterBillOfLadingNumber",
    ):
        value = patch.get(field)
        if isinstance(value, str):
            rows.append((f"{request_prefix}/documentPatch.{field}", "document", value))
    transport = patch.get("transport")
    if isinstance(transport, Mapping) and isinstance(transport.get("voyageNumber"), str):
        rows.append(
            (
                f"{request_prefix}/documentPatch.transport.voyageNumber",
                "voyage",
                cast(str, transport["voyageNumber"]),
            )
        )
    for index, value in enumerate(patch.get("forwardingAndExportReferences") or []):
        if not isinstance(value, str):
            raise ValueError("forwarding/export reference must be a string")
        rows.append(
            (
                f"{request_prefix}/documentPatch.forwardingAndExportReferences[{index}]",
                "reference",
                value,
            )
        )
    for container_index, container in enumerate(patch.get("containers") or []):
        if not isinstance(container, Mapping):
            raise ValueError("container row must be an object")
        container_number = container.get("containerNumber")
        if not isinstance(container_number, str):
            raise ValueError("container number must be a string")
        rows.append(
            (
                f"{request_prefix}/documentPatch.containers[{container_index}].containerNumber",
                "container",
                container_number,
            )
        )
        for seal_index, seal in enumerate(container.get("sealNumbers") or []):
            if not isinstance(seal, str):
                raise ValueError("seal identifier must be a string")
            rows.append(
                (
                    f"{request_prefix}/documentPatch.containers[{container_index}]"
                    f".sealNumbers[{seal_index}]",
                    "seal",
                    seal,
                )
            )
    return tuple(rows)


def build_identifier_request_inventory(
    *, source_targets: Mapping[str, Mapping[str, Any]], selected_document_ids: Sequence[str]
) -> IdentifierRequestInventory:
    """Build one complete request set and reject unsupported identifier surfaces."""

    if not selected_document_ids or len(selected_document_ids) != len(set(selected_document_ids)):
        raise ValueError("selected document IDs must be non-empty and unique")
    missing = sorted(set(selected_document_ids) - set(source_targets))
    if missing:
        raise ValueError(f"selected documents are absent from source targets: {missing}")
    container_requests: list[IdentifierReservationRequest] = []
    container_prefixes: dict[str, str] = {}
    generic_requests: list[IdentifierReservationRequest] = []
    generic_sources: dict[str, str] = {}
    for document_id in selected_document_ids:
        for key, kind, value in _iter_formal_identifiers(
            source_targets[document_id], request_prefix=document_id
        ):
            request = IdentifierReservationRequest(request_key=key, identifier_kind=kind)
            if kind == "container":
                prefix = value[:4]
                # The normative checker validates owner/category syntax as well.
                iso6346_check_digit(prefix + "000000")
                container_requests.append(request)
                container_prefixes[key] = prefix
                continue
            pattern = surface_pattern(value)
            if not any(token in pattern for token in ("A", "a", "9")):
                raise ValueError(f"formal identifier has no replaceable characters: {key}")
            generic_requests.append(request)
            generic_sources[key] = value
    return IdentifierRequestInventory(
        container_requests=tuple(container_requests),
        container_prefix_by_request=container_prefixes,
        generic_requests=tuple(generic_requests),
        generic_source_by_request=generic_sources,
    )


def _entropy_bytes(entropy: bytes, *, minimum: int) -> bytes:
    output = bytearray(entropy)
    counter = 0
    while len(output) < minimum:
        output.extend(hashlib.sha256(entropy + counter.to_bytes(8, "big")).digest())
        counter += 1
    return bytes(output)


def reserve_structured_identifiers(
    *,
    all_source_targets: Mapping[str, Mapping[str, Any]],
    inventory: IdentifierRequestInventory,
    seed: int,
) -> IdentifierAllocationPlan:
    """Allocate selected identifiers against every real identifier in the corpus."""

    real_containers: list[str] = []
    real_generic: list[str] = []
    for document_id in sorted(all_source_targets):
        for _key, kind, value in _iter_formal_identifiers(
            all_source_targets[document_id], request_prefix=document_id
        ):
            (real_containers if kind == "container" else real_generic).append(value)

    containers = None
    if inventory.container_requests:
        count, _unique, digest = identifier_corpus_sha256(
            real_containers, canonicalize=canonical_formal_identifier
        )

        def container_factory(context: IdentifierCandidateContext) -> str:
            prefix = inventory.container_prefix_by_request[context.request_key]
            serial = int.from_bytes(context.entropy[:8], "big") % 1_000_000
            body = prefix + f"{serial:06d}"
            return body + iso6346_check_digit(body)

        containers = reserve_global_identifiers(
            real_identifiers=real_containers,
            requests=inventory.container_requests,
            seed=str(seed),
            namespace="mpci-bl-structured-container-identifiers-v1",
            canonicalize=canonical_formal_identifier,
            canonicalizer_name="unicode_nfkc_alphanumeric_casefold_v1",
            candidate_factory=container_factory,
            expected_real_identifier_count=count,
            expected_real_corpus_sha256=digest,
        )

    generic = None
    if inventory.generic_requests:
        count, _unique, digest = identifier_corpus_sha256(
            real_generic, canonicalize=canonical_formal_identifier
        )

        def generic_factory(context: IdentifierCandidateContext) -> str:
            pattern = surface_pattern(inventory.generic_source_by_request[context.request_key])
            tokens: list[tuple[str, str]] = []
            index = 0
            while index < len(pattern):
                if pattern[index] == "\\":
                    tokens.append(("literal", pattern[index + 1]))
                    index += 2
                else:
                    tokens.append((pattern[index], pattern[index]))
                    index += 1
            entropy = _entropy_bytes(
                context.entropy,
                minimum=sum(kind != "literal" for kind, _ in tokens),
            )
            variable_index = 0
            output: list[str] = []
            for kind, literal in tokens:
                if kind == "literal":
                    output.append(literal)
                    continue
                alphabet = _ALPHABETS[kind]
                output.append(alphabet[entropy[variable_index] % len(alphabet)])
                variable_index += 1
            return "".join(output)

        generic = reserve_global_identifiers(
            real_identifiers=real_generic,
            requests=inventory.generic_requests,
            seed=str(seed),
            namespace="mpci-bl-structured-generic-identifiers-v1",
            canonicalize=canonical_formal_identifier,
            canonicalizer_name="unicode_nfkc_alphanumeric_casefold_v1",
            candidate_factory=generic_factory,
            expected_real_identifier_count=count,
            expected_real_corpus_sha256=digest,
        )
    return IdentifierAllocationPlan(containers=containers, generic=generic)


def _change(
    *, path: str, family: str, old: Any, new: Any, method: str, coupling_group: str
) -> SemanticChange:
    return SemanticChange.model_validate(
        {
            "target_path": path,
            "role_path": policy_for_target_path(path).role_path,
            "family": family,
            "old_value": json_value(old),
            "new_value": json_value(new),
            "method": method,
            "coupling_group": coupling_group,
        },
        strict=True,
    )


def apply_identifier_plan(
    *, document_id: str, target: dict[str, Any], allocations: Mapping[str, str]
) -> tuple[SemanticChange, ...]:
    """Apply all reserved identifiers and derive allocation references."""

    changes: list[SemanticChange] = []
    container_mapping: dict[str, str] = {}
    for request_key, kind, old in _iter_formal_identifiers(target, request_prefix=document_id):
        try:
            new = allocations[request_key]
        except KeyError as error:
            raise ValueError(
                f"identifier request was not globally reserved: {request_key}"
            ) from error
        path = request_key.removeprefix(document_id + "/")
        set_target_value(target, path, new)
        family = {
            "container": "container_identifier",
            "seal": "seal_identifier",
            "voyage": "voyage_identifier",
            "document": "document_identifier",
            "reference": "document_identifier",
        }[kind]
        changes.append(
            _change(
                path=path,
                family=family,
                old=old,
                new=new,
                method=(
                    "global_iso6346_reservation_v1"
                    if kind == "container"
                    else "global_shape_preserving_reservation_v1"
                ),
                coupling_group="container_graph" if kind in {"container", "seal"} else kind,
            )
        )
        if kind == "container":
            container_mapping[old] = new
    patch = target["documentPatch"]
    for group_index, group in enumerate(patch.get("cargoAllocationGroups") or []):
        for allocation_index, allocation in enumerate(group["allocations"]):
            old = allocation["containerNumber"]
            try:
                new = container_mapping[old]
            except KeyError as error:
                raise ValueError("allocation references a non-reserved container") from error
            allocation["containerNumber"] = new
            changes.append(
                _change(
                    path=(
                        f"documentPatch.cargoAllocationGroups[{group_index}]"
                        f".allocations[{allocation_index}].containerNumber"
                    ),
                    family="allocation_reference",
                    old=old,
                    new=new,
                    method="derive_globally_reserved_container_reference_v1",
                    coupling_group="container_graph",
                )
            )
    return tuple(changes)


def apply_date_proposal(
    *, target: dict[str, Any], issue_date: date | None, shipped_on_board_date: date | None
) -> tuple[SemanticChange, ...]:
    """Apply a caller-proposed, already bounded date pair without changing missingness."""

    patch = target["documentPatch"]
    proposed = {
        "issueDate": issue_date.isoformat() if issue_date is not None else None,
        "shippedOnBoardDate": (
            shipped_on_board_date.isoformat() if shipped_on_board_date is not None else None
        ),
    }
    changes: list[SemanticChange] = []
    for field, new in proposed.items():
        old = patch.get(field)
        if (old is None) != (new is None):
            raise ValueError("date proposal changes source missingness")
        if old == new:
            continue
        patch[field] = new
        changes.append(
            _change(
                path=f"documentPatch.{field}",
                family="document_date",
                old=old,
                new=new,
                method="train_empirical_anchor_plus_bounded_jitter_v1",
                coupling_group="document_dates",
            )
        )
    if not changes and any(value is not None for value in proposed.values()):
        raise ValueError("date proposal did not change any present date")
    return tuple(changes)


def _parse_embedded_numeric_date(match: re.Match[str]) -> tuple[date, bool] | None:
    """Parse an OCR numeric date with the project's frozen day-first ambiguity policy."""

    first = int(match.group("a"))
    second = int(match.group("b"))
    year_surface = match.group("year")
    year = int(year_surface)
    if len(year_surface) == 2:
        year += 2000 if year <= 68 else 1900
    month_first = second > 12 and first <= 12
    day, month = (second, first) if month_first else (first, second)
    try:
        return date(year, month, day), month_first
    except ValueError:
        # Reference identifiers can themselves contain three numeric dot/dash groups. They are
        # not dates unless the source surface forms a valid calendar date.
        return None


def _render_embedded_numeric_date(
    value: date, *, match: re.Match[str], month_first: bool
) -> str:
    first = value.month if month_first else value.day
    second = value.day if month_first else value.month
    separator = match.group("separator")
    year = f"{value.year % 100:02d}" if len(match.group("year")) == 2 else f"{value.year:04d}"
    return (
        f"{first:0{len(match.group('a'))}d}{separator}"
        f"{second:0{len(match.group('b'))}d}{separator}{year}"
    )


def apply_embedded_reference_date_shift(
    *,
    source_target: Mapping[str, Any],
    target: dict[str, Any],
    changes: list[SemanticChange],
    fallback_shift_days: int,
) -> tuple[dict[str, Any], ...]:
    """Make dates embedded in generated forwarding references valid and temporally coherent.

    Generic identifier reservation intentionally preserves character shape, but digit-by-digit
    synthesis can turn a source date into an impossible value.  Shift every valid source
    reference date by the same document-date displacement used by the synthetic scenario.  When
    the template has no labeled document date, use the caller's non-zero deterministic bounded
    displacement.  The reference remains one atomic change-ledger leaf.
    """

    if fallback_shift_days == 0:
        raise ValueError("embedded reference-date fallback shift must be non-zero")
    source_patch = cast(Mapping[str, Any], source_target["documentPatch"])
    target_patch = cast(dict[str, Any], target["documentPatch"])
    source_anchor: date | None = None
    target_anchor: date | None = None
    for field in ("issueDate", "shippedOnBoardDate"):
        source_value = source_patch.get(field)
        target_value = target_patch.get(field)
        if isinstance(source_value, str) and isinstance(target_value, str):
            source_anchor = date.fromisoformat(source_value)
            target_anchor = date.fromisoformat(target_value)
            break
    shift_days = (
        (target_anchor - source_anchor).days
        if source_anchor is not None and target_anchor is not None
        else fallback_shift_days
    )
    if shift_days == 0:
        shift_days = fallback_shift_days
    source_references = source_patch.get("forwardingAndExportReferences") or []
    target_references = target_patch.get("forwardingAndExportReferences") or []
    if len(source_references) != len(target_references):
        raise ValueError("reference-date projection requires stable reference cardinality")
    receipts: list[dict[str, Any]] = []
    for index, (source_reference, generated_reference) in enumerate(
        zip(source_references, target_references, strict=True)
    ):
        if not isinstance(source_reference, str) or not isinstance(generated_reference, str):
            raise ValueError("reference-date projection requires string references")
        matches = tuple(_EMBEDDED_NUMERIC_DATE.finditer(source_reference))
        if not matches:
            continue
        if len(source_reference) != len(generated_reference):
            raise ValueError("shape-preserving reference allocation changed string length")
        rendered = generated_reference
        projected: list[dict[str, Any]] = []
        for match in reversed(matches):
            parsed = _parse_embedded_numeric_date(match)
            if parsed is None:
                continue
            source_date, month_first = parsed
            target_date = source_date + timedelta(days=shift_days)
            target_surface = _render_embedded_numeric_date(
                target_date, match=match, month_first=month_first
            )
            rendered = rendered[: match.start()] + target_surface + rendered[match.end() :]
            projected.append(
                {
                    "sourceSurface": match.group(0),
                    "targetSurface": target_surface,
                    "shiftDays": shift_days,
                }
            )
        if not projected:
            continue
        path = f"documentPatch.forwardingAndExportReferences[{index}]"
        target_references[index] = rendered
        matching_changes = [
            position for position, row in enumerate(changes) if row.target_path == path
        ]
        if len(matching_changes) != 1:
            raise ValueError(f"reference-date projection lacks one identifier change: {path}")
        changes[matching_changes[0]] = _change(
            path=path,
            family="document_identifier",
            old=source_reference,
            new=rendered,
            method="global_shape_reservation_plus_joint_embedded_date_shift_v1",
            coupling_group="reference",
        )
        receipts.append(
            {
                "targetPath": path,
                "sourceReference": source_reference,
                "targetReference": rendered,
                "dateProjections": list(reversed(projected)),
            }
        )
    return tuple(receipts)


def _decimal_places(value: int | float) -> int:
    decimal = Decimal(str(value))
    exponent = decimal.as_tuple().exponent
    if not decimal.is_finite() or not isinstance(exponent, int):
        raise ValueError("cargo measure is not finite")
    return max(0, -exponent)


def _coerce_proposed_measure(*, source: Mapping[str, Any], proposed: int | float) -> int | float:
    value = Decimal(str(proposed))
    if not value.is_finite() or value <= 0:
        raise ValueError("proposed cargo measure must be finite and positive")
    decimal_places = _decimal_places(cast(int | float, source["value"]))
    quantum = Decimal(1).scaleb(-decimal_places)
    rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    if rounded <= 0:
        raise ValueError("proposed cargo measure rounds to a non-positive value")
    if isinstance(source["value"], int) and not isinstance(source["value"], bool):
        return int(rounded)
    return float(rounded)


def derive_scaled_group_quantities(
    *,
    packages: Sequence[Mapping[str, Any]],
    driver_package_id: str,
    generated_driver_quantity: int,
) -> dict[str, int] | None:
    """Scale every present package level by one driver ratio without mixing units."""

    driver = next(
        (package for package in packages if package["packageId"] == driver_package_id),
        None,
    )
    if driver is None:
        raise ValueError("cargo-group driver package is absent")
    source_driver_quantity = driver.get("quantity")
    if (
        not isinstance(source_driver_quantity, int)
        or isinstance(source_driver_quantity, bool)
        or source_driver_quantity <= 0
        or not isinstance(generated_driver_quantity, int)
        or isinstance(generated_driver_quantity, bool)
        or generated_driver_quantity <= 0
    ):
        raise ValueError("cargo-group driver quantities must be positive integers")
    ratio = Decimal(generated_driver_quantity) / Decimal(source_driver_quantity)
    output: dict[str, int] = {}
    for package in packages:
        source_quantity = package.get("quantity")
        if source_quantity is None:
            continue
        if not isinstance(source_quantity, int) or isinstance(source_quantity, bool):
            raise ValueError("package quantity is not an integer")
        package_id = cast(str, package["packageId"])
        if package_id == driver_package_id:
            generated = generated_driver_quantity
        else:
            generated = int(
                (Decimal(source_quantity) * ratio).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            )
        if generated <= 0 or generated == source_quantity:
            return None
        output[package_id] = generated
    return output


def apply_cargo_group_numeric_proposals(
    *, target: dict[str, Any], proposals: Sequence[CargoGroupNumericProposal]
) -> tuple[SemanticChange, ...]:
    """Apply one joint numeric proposal per modeled cargo group and reconcile relations."""

    patch = target["documentPatch"]
    packages = cast(list[dict[str, Any]], patch.get("cargoPackages") or [])
    package_indices = {
        cast(str, package["packageId"]): index for index, package in enumerate(packages)
    }
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for package in packages:
        by_group[cast(str, package["groupId"])].append(package)
    expected_groups = {
        group_id
        for group_id, rows in by_group.items()
        if any(row.get("quantity") is not None for row in rows)
    }
    proposal_by_group = {proposal.group_id: proposal for proposal in proposals}
    if len(proposal_by_group) != len(proposals):
        raise ValueError("cargo-group proposals contain duplicate group IDs")
    if set(proposal_by_group) != expected_groups:
        missing = sorted(expected_groups - set(proposal_by_group))
        extra = sorted(set(proposal_by_group) - expected_groups)
        raise ValueError(
            f"cargo-group proposal coverage mismatch: missing={missing}, extra={extra}"
        )

    changes: list[SemanticChange] = []
    driver_scale_by_group: dict[str, tuple[int, int]] = {}
    groups = cast(list[dict[str, Any]], patch.get("cargoGroups") or [])
    group_indices = {cast(str, group["groupId"]): index for index, group in enumerate(groups)}
    if set(expected_groups) - set(group_indices):
        raise ValueError("package proposal references a missing cargo group")
    measure_fields = {
        "grossWeight": "gross_weight_value",
        "netWeight": "net_weight_value",
        "volume": "volume_value",
    }
    for group_id in sorted(expected_groups, key=lambda value: int(value[1:])):
        proposal = proposal_by_group[group_id]
        group_packages = by_group[group_id]
        present_package_ids = {
            cast(str, row["packageId"]) for row in group_packages if row.get("quantity") is not None
        }
        if set(proposal.quantity_by_package_id) != present_package_ids:
            raise ValueError(f"package proposal coverage differs for cargo group {group_id}")
        if proposal.driver_package_id not in present_package_ids:
            raise ValueError(f"cargo-group driver is not a quantified package: {group_id}")
        expected_quantities = derive_scaled_group_quantities(
            packages=group_packages,
            driver_package_id=proposal.driver_package_id,
            generated_driver_quantity=proposal.quantity_by_package_id[proposal.driver_package_id],
        )
        if expected_quantities is None or expected_quantities != proposal.quantity_by_package_id:
            raise ValueError(f"cargo-group quantities do not share one driver scale: {group_id}")
        source_driver_quantity = next(
            cast(int, package["quantity"])
            for package in group_packages
            if package["packageId"] == proposal.driver_package_id
        )
        driver_scale_by_group[group_id] = (
            proposal.quantity_by_package_id[proposal.driver_package_id],
            source_driver_quantity,
        )
        for package in group_packages:
            if package.get("quantity") is None:
                continue
            package_id = cast(str, package["packageId"])
            old = cast(int, package["quantity"])
            new = proposal.quantity_by_package_id[package_id]
            if new == old:
                raise ValueError(f"package proposal must change its source value: {package_id}")
            package["quantity"] = new
            changes.append(
                _change(
                    path=(f"documentPatch.cargoPackages[{package_indices[package_id]}].quantity"),
                    family="package_quantity",
                    old=old,
                    new=new,
                    method="profile_routed_group_driver_then_reconcile_v1",
                    coupling_group=f"cargo_numeric:{group_id}",
                )
            )

        group_index = group_indices[group_id]
        group = groups[group_index]
        for field, proposal_field in measure_fields.items():
            source_measure = group.get(field)
            proposed_value = cast(int | float | None, getattr(proposal, proposal_field))
            if (source_measure is None) != (proposed_value is None):
                raise ValueError(f"cargo measure proposal changes missingness: {group_id}/{field}")
            if source_measure is None:
                continue
            if not isinstance(source_measure, Mapping) or proposed_value is None:
                raise ValueError(f"cargo measure is malformed: {group_id}/{field}")
            new_value = _coerce_proposed_measure(
                source=source_measure,
                proposed=proposed_value,
            )
            old_value = source_measure["value"]
            if new_value == old_value:
                continue
            group[field] = {"value": new_value, "unit": source_measure["unit"]}
            changes.append(
                _change(
                    path=f"documentPatch.cargoGroups[{group_index}].{field}.value",
                    family="cargo_measure",
                    old=old_value,
                    new=new_value,
                    method=("profile_routed_per_driver_measure_then_exact_unit_inverse_v1"),
                    coupling_group=f"cargo_numeric:{group_id}",
                )
            )
        validate_mass_order(
            gross_weight=group.get("grossWeight"),
            net_weight=group.get("netWeight"),
        )

    for group_index, allocation_group in enumerate(
        cast(list[dict[str, Any]], patch.get("cargoAllocationGroups") or [])
    ):
        old_allocations = deepcopy(allocation_group["allocations"])
        reconciled = reconcile_allocation_group(
            packages=packages,
            allocation_group=allocation_group,
        )
        if allocation_group["coverage"] == "unlinked_package_quantities":
            allocation_group_id = cast(str, allocation_group["groupId"])
            if allocation_group_id not in driver_scale_by_group:
                raise ValueError("unlinked allocation quantities have no quantified package driver")
            numerator, denominator = driver_scale_by_group[allocation_group_id]
            old_quantities = [cast(int, row["packageQuantity"]) for row in old_allocations]
            source_total = sum(old_quantities)
            generated_total = int(
                (Decimal(source_total) * Decimal(numerator) / Decimal(denominator)).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
            )
            generated_quantities = largest_remainder_allocation(
                generated_total,
                old_quantities,
            )
            if any(
                generated <= 0 or generated == source
                for source, generated in zip(old_quantities, generated_quantities, strict=True)
            ):
                raise ValueError(
                    "unlinked allocation scaling must preserve positive membership "
                    "and change every derived quantity"
                )
            for row, quantity in zip(reconciled["allocations"], generated_quantities, strict=True):
                row["packageQuantity"] = quantity
        patch["cargoAllocationGroups"][group_index] = reconciled
        for allocation_index, (old, new) in enumerate(
            zip(old_allocations, reconciled["allocations"], strict=True)
        ):
            if old.get("packageQuantity") == new.get("packageQuantity"):
                continue
            changes.append(
                _change(
                    path=(
                        f"documentPatch.cargoAllocationGroups[{group_index}]"
                        f".allocations[{allocation_index}].packageQuantity"
                    ),
                    family="allocation_quantity",
                    old=old["packageQuantity"],
                    new=new["packageQuantity"],
                    method="derive_reconciled_allocation_quantity_v1",
                    coupling_group=f"cargo_numeric:{allocation_group['groupId']}",
                )
            )
    validate_allocation_arithmetic(target)
    return tuple(changes)


def pending_realizations(
    target: Mapping[str, Any], *, changed_paths: Sequence[str]
) -> tuple[PendingRealization, ...]:
    """List every remaining registry/linguistic task and reject deterministic gaps."""

    changed = frozenset(changed_paths)
    output: list[PendingRealization] = []
    for path, _value in leaf_items(target):
        if path in changed:
            continue
        policy = policy_for_target_path(path)
        if policy.implementation_status in {"pending_registry", "pending_linguistic"}:
            if policy.implementation_status == "pending_registry":
                output.append(
                    PendingRealization(
                        target_path=path,
                        role_path=policy.role_path,
                        kind="registry",
                        reason=(f"{policy.method} remains for the later semantic realization pass"),
                    )
                )
            else:
                output.append(
                    PendingRealization(
                        target_path=path,
                        role_path=policy.role_path,
                        kind="linguistic",
                        reason=(f"{policy.method} remains for the later semantic realization pass"),
                    )
                )
        elif (
            _value is not None
            and policy.implementation_status == "implemented"
            and policy.value_policy
            in {
                "regenerate",
                "resample",
            }
        ):
            raise ValueError(f"implemented source fact was not realized: {path}")
    return tuple(sorted(output, key=lambda row: row.target_path))


def policy_counts() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for policy in FIELD_POLICIES.values():
        counts[policy.value_policy] += 1
    return dict(sorted(counts.items()))


def finalize_non_linguistic_target(
    *, source_target: Mapping[str, Any], target: dict[str, Any], changes: Sequence[SemanticChange]
) -> str:
    """Prove the target diff, graph arithmetic, and return its canonical hash."""

    ordered = tuple(sorted(changes, key=lambda row: row.target_path))
    validate_change_ledger(source_target, target, ordered)
    validate_allocation_arithmetic(target)
    return sha256_bytes(canonical_json_bytes(target))
