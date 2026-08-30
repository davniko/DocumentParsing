"""Deterministic, cardinality-preserving B/L draft construction."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.anchors import leaf_items
from document_ocr.synthesis.generation_models import (
    DraftScenarioPlan,
    PendingRealization,
    SemanticChange,
    json_value,
)
from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_unique_container_numbers,
    generate_unique_seal_identifiers,
    reconcile_allocation_group,
    shift_document_dates,
    validate_mass_order,
)
from document_ocr.synthesis.policies import FIELD_POLICIES, policy_for_target_path

_SEGMENT = re.compile(r"([^.\[]+)(?:\[([0-9]+)\])?")


def _segments(path: str) -> tuple[tuple[str, int | None], ...]:
    output = []
    for part in path.split("."):
        match = _SEGMENT.fullmatch(part)
        if match is None:
            raise ValueError(f"invalid target path: {path}")
        output.append((match.group(1), int(match.group(2)) if match.group(2) else None))
    return tuple(output)


def target_value(target: Mapping[str, Any], path: str) -> Any:
    current: Any = target
    for name, index in _segments(path):
        current = current[name]
        if index is not None:
            current = current[index]
    return current


def set_target_value(target: dict[str, Any], path: str, value: Any) -> None:
    current: Any = target
    parts = _segments(path)
    for name, index in parts[:-1]:
        current = current[name]
        if index is not None:
            current = current[index]
    name, index = parts[-1]
    if index is None:
        current[name] = value
    else:
        current[name][index] = value


def validate_change_ledger(
    source_target: Mapping[str, Any],
    deterministic_target: Mapping[str, Any],
    changes: Sequence[SemanticChange],
) -> None:
    """Prove the semantic ledger is the exact leaf-level diff in both directions."""

    source_leaves = dict(leaf_items(source_target))
    target_leaves = dict(leaf_items(deterministic_target))
    changed_paths = {
        path
        for path in source_leaves.keys() | target_leaves.keys()
        if source_leaves.get(path) != target_leaves.get(path)
    }
    ledger_paths = {change.target_path for change in changes}
    if changed_paths != ledger_paths:
        missing = sorted(changed_paths - ledger_paths)
        extra = sorted(ledger_paths - changed_paths)
        raise ValueError(f"semantic change ledger mismatch: missing={missing}, extra={extra}")
    for change in changes:
        if source_leaves[change.target_path] != change.old_value:
            raise ValueError(f"semantic change old value mismatch: {change.target_path}")
        if target_leaves[change.target_path] != change.new_value:
            raise ValueError(f"semantic change new value mismatch: {change.target_path}")


def validate_allocation_arithmetic(target: Mapping[str, Any]) -> None:
    """Prove task-facing package quantities and allocation relations reconcile exactly."""

    patch = cast(Mapping[str, Any], target["documentPatch"])
    packages = {
        cast(str, package["packageId"]): package
        for package in cast(Sequence[Mapping[str, Any]], patch.get("cargoPackages") or [])
    }
    for group in cast(
        Sequence[Mapping[str, Any]], patch.get("cargoAllocationGroups") or []
    ):
        coverage = group["coverage"]
        package_ids = cast(Sequence[str], group.get("packageIds") or [])
        allocations = cast(Sequence[Mapping[str, Any]], group["allocations"])
        if coverage == "one_to_one_package_allocations":
            for allocation in allocations:
                package = packages[cast(str, allocation["packageId"])]
                if allocation["packageQuantity"] != package["quantity"]:
                    raise ValueError("one-to-one allocation quantity differs from its package")
        elif coverage in {"single_package_level", "all_package_levels_combined"}:
            package_total = sum(
                cast(int, packages[package_id]["quantity"])
                for package_id in package_ids
            )
            allocation_total = sum(
                cast(int, allocation["packageQuantity"]) for allocation in allocations
            )
            if package_total != allocation_total:
                raise ValueError("allocation quantity sum differs from covered package quantities")


@dataclass(frozen=True, slots=True)
class NumericCargoTuple:
    source_document_id: str
    group_id: str
    package_quantity: int
    package_identity: str
    gross_weight: tuple[int | float, str] | None
    net_weight: tuple[int | float, str] | None
    volume: tuple[int | float, str] | None

    def signature(self) -> tuple[str, tuple[str | None, str | None, str | None]]:
        gross, net, volume = self.measures()
        return (
            self.package_identity,
            (
                gross[1] if gross is not None else None,
                net[1] if net is not None else None,
                volume[1] if volume is not None else None,
            ),
        )

    def measures(
        self,
    ) -> tuple[
        tuple[int | float, str] | None,
        tuple[int | float, str] | None,
        tuple[int | float, str] | None,
    ]:
        return self.gross_weight, self.net_weight, self.volume

    def values(self) -> tuple[Any, ...]:
        return self.package_quantity, *self.measures()


def _package_identity(package: Mapping[str, Any]) -> str:
    category = package.get("typeCategory")
    description = package.get("typeDescription")
    if isinstance(category, str):
        return "category:" + category
    if isinstance(description, str):
        return "description:" + description.strip().upper()
    return "missing"


def _measure_tuple(value: Any) -> tuple[int | float, str] | None:
    if not isinstance(value, dict):
        return None
    scalar = value.get("value")
    unit = value.get("unit")
    if (
        not isinstance(scalar, (int, float))
        or isinstance(scalar, bool)
        or not isinstance(unit, str)
    ):
        raise ValueError("cargo measure is outside the task-facing numeric contract")
    return scalar, unit


def numeric_cargo_tuples(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[NumericCargoTuple, ...]:
    output = []
    for row in rows:
        document_id = cast(str, row["documentId"])
        patch = cast(Mapping[str, Any], cast(Mapping[str, Any], row["target"])["documentPatch"])
        packages_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for package in cast(Sequence[Mapping[str, Any]], patch.get("cargoPackages") or []):
            packages_by_group[cast(str, package["groupId"])].append(package)
        for group in cast(Sequence[Mapping[str, Any]], patch.get("cargoGroups") or []):
            group_id = cast(str, group["groupId"])
            packages = packages_by_group[group_id]
            if len(packages) != 1:
                continue
            quantity = packages[0].get("quantity")
            if not isinstance(quantity, int) or isinstance(quantity, bool):
                continue
            output.append(
                NumericCargoTuple(
                    source_document_id=document_id,
                    group_id=group_id,
                    package_quantity=quantity,
                    package_identity=_package_identity(packages[0]),
                    gross_weight=_measure_tuple(group.get("grossWeight")),
                    net_weight=_measure_tuple(group.get("netWeight")),
                    volume=_measure_tuple(group.get("volume")),
                )
            )
    return tuple(output)


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


def _container_changes(
    target: dict[str, Any], stream: DeterministicStream
) -> tuple[SemanticChange, ...]:
    patch = target["documentPatch"]
    containers = patch.get("containers") or []
    if not containers:
        raise ValueError("document has no container identifier")
    old_values = [row["containerNumber"] for row in containers]
    owners = {str(index): value[:4] for index, value in enumerate(old_values)}
    replacements = generate_unique_container_numbers(
        owner_by_identity=owners,
        stream=stream.derive("containers"),
        excluded=frozenset(old_values),
    )
    mapping: dict[str, str] = {}
    changes = []
    for index, row in enumerate(containers):
        old = row["containerNumber"]
        new = replacements[str(index)]
        mapping[old] = new
        row["containerNumber"] = new
        changes.append(
            _change(
                path=f"documentPatch.containers[{index}].containerNumber",
                family="container_identifier",
                old=old,
                new=new,
                method="preserve_owner_prefix_iso6346_v1",
                coupling_group="container_graph",
            )
        )
    for group_index, group in enumerate(patch.get("cargoAllocationGroups") or []):
        for allocation_index, allocation in enumerate(group["allocations"]):
            old = allocation["containerNumber"]
            new_container = mapping.get(old)
            if new_container is None:
                raise ValueError("allocation references a container absent from the document")
            allocation["containerNumber"] = new_container
            changes.append(
                _change(
                    path=(
                        f"documentPatch.cargoAllocationGroups[{group_index}]"
                        f".allocations[{allocation_index}].containerNumber"
                    ),
                    family="allocation_reference",
                    old=old,
                    new=new_container,
                    method="derive_regenerated_container_reference_v1",
                    coupling_group="container_graph",
                )
            )
    return tuple(changes)


def _seal_changes(
    target: dict[str, Any], stream: DeterministicStream
) -> tuple[SemanticChange, ...]:
    containers = target["documentPatch"].get("containers") or []
    sources: dict[str, str] = {}
    for container_index, container in enumerate(containers):
        for seal_index, value in enumerate(container.get("sealNumbers") or []):
            sources[f"{container_index}:{seal_index}"] = value
    if not sources:
        raise ValueError("document has no scalar seal identifier")
    replacements = generate_unique_seal_identifiers(
        source_by_identity=sources,
        stream=stream.derive("seals"),
        excluded=frozenset(sources.values()),
    )
    changes = []
    for identity, old in sources.items():
        container_index, seal_index = (int(value) for value in identity.split(":"))
        new = replacements[identity]
        containers[container_index]["sealNumbers"][seal_index] = new
        changes.append(
            _change(
                path=(
                    f"documentPatch.containers[{container_index}]"
                    f".sealNumbers[{seal_index}]"
                ),
                family="seal_identifier",
                old=old,
                new=new,
                method="simple_scalar_surface_pattern_v1",
                coupling_group="container_graph",
            )
        )
    return tuple(changes)


def _date_changes(
    target: dict[str, Any], stream: DeterministicStream, minimum: date, maximum: date
) -> tuple[SemanticChange, ...]:
    patch = target["documentPatch"]
    old_issue = date.fromisoformat(patch["issueDate"]) if patch.get("issueDate") else None
    old_shipped = (
        date.fromisoformat(patch["shippedOnBoardDate"])
        if patch.get("shippedOnBoardDate")
        else None
    )
    if old_issue is None and old_shipped is None:
        raise ValueError("document has no date")
    shifted = (old_issue, old_shipped)
    for attempt in range(256):
        shifted = shift_document_dates(
            issue_date=old_issue,
            shipped_on_board_date=old_shipped,
            minimum=minimum,
            maximum=maximum,
            stream=stream.derive(f"date-offset:{attempt}"),
        )
        if shifted != (old_issue, old_shipped):
            break
    if shifted == (old_issue, old_shipped):
        raise ValueError("date window did not produce a non-zero bounded shift")
    changes = []
    for field, old, new in (
        ("issueDate", old_issue, shifted[0]),
        ("shippedOnBoardDate", old_shipped, shifted[1]),
    ):
        if old is None or new is None:
            continue
        patch[field] = new.isoformat()
        changes.append(
            _change(
                path=f"documentPatch.{field}",
                family="document_date",
                old=old.isoformat(),
                new=new.isoformat(),
                method="joint_bounded_day_shift_v1",
                coupling_group="document_dates",
            )
        )
    return tuple(changes)


def _choose_numeric_donor(
    *,
    document_id: str,
    source: NumericCargoTuple,
    donors: Sequence[NumericCargoTuple],
    require_measures: bool,
    stream: DeterministicStream,
) -> NumericCargoTuple:
    candidates = [
        row
        for row in donors
        if row.source_document_id != document_id
        and row.signature() == source.signature()
        and bool(any(row.measures())) == require_measures
        and row.values() != source.values()
    ]
    if not candidates:
        raise ValueError("no distinct empirical cargo tuple matches the source structure")
    candidates.sort(key=lambda row: (row.source_document_id, row.group_id, row.values()))
    return candidates[stream.randbelow(len(candidates))]


def _numeric_changes(
    *,
    document_id: str,
    target: dict[str, Any],
    donors: Sequence[NumericCargoTuple],
    require_measures: bool,
    stream: DeterministicStream,
) -> tuple[SemanticChange, ...]:
    patch = target["documentPatch"]
    packages = patch.get("cargoPackages") or []
    groups = patch.get("cargoGroups") or []
    package_indexes: dict[str, list[int]] = defaultdict(list)
    for index, package in enumerate(packages):
        package_indexes[package["groupId"]].append(index)
    sources = []
    for group_index, group in enumerate(groups):
        indexes = package_indexes[group["groupId"]]
        if len(indexes) != 1:
            continue
        package = packages[indexes[0]]
        quantity = package.get("quantity")
        if not isinstance(quantity, int) or isinstance(quantity, bool):
            continue
        source = NumericCargoTuple(
            source_document_id=document_id,
            group_id=group["groupId"],
            package_quantity=quantity,
            package_identity=_package_identity(package),
            gross_weight=_measure_tuple(group.get("grossWeight")),
            net_weight=_measure_tuple(group.get("netWeight")),
            volume=_measure_tuple(group.get("volume")),
        )
        if bool(any(source.measures())) == require_measures:
            sources.append((group_index, indexes[0], source))
    if not sources:
        raise ValueError("document has no compatible single-package cargo tuple")
    ordered = sorted(sources, key=lambda row: row[2].group_id)
    source_index = stream.derive("source-group").randbelow(len(ordered))
    group_index, package_index, source = ordered[source_index]
    donor = _choose_numeric_donor(
        document_id=document_id,
        source=source,
        donors=donors,
        require_measures=require_measures,
        stream=stream.derive("donor"),
    )
    changes = []
    package_path = f"documentPatch.cargoPackages[{package_index}].quantity"
    packages[package_index]["quantity"] = donor.package_quantity
    changes.append(
        _change(
            path=package_path,
            family="package_quantity",
            old=source.package_quantity,
            new=donor.package_quantity,
            method="empirical_same_category_numeric_tuple_v1",
            coupling_group="cargo_numeric_tuple",
        )
    )
    if require_measures:
        for field, old, new in zip(
            ("grossWeight", "netWeight", "volume"),
            source.measures(),
            donor.measures(),
            strict=True,
        ):
            if old is None or new is None:
                continue
            groups[group_index][field]["value"] = new[0]
            changes.append(
                _change(
                    path=f"documentPatch.cargoGroups[{group_index}].{field}.value",
                    family="cargo_measure",
                    old=old[0],
                    new=new[0],
                    method="empirical_same_category_numeric_tuple_v1",
                    coupling_group="cargo_numeric_tuple",
                )
            )
        validate_mass_order(
            gross_weight=groups[group_index].get("grossWeight"),
            net_weight=groups[group_index].get("netWeight"),
        )
    group_id = source.group_id
    for allocation_group_index, allocation_group in enumerate(
        patch.get("cargoAllocationGroups") or []
    ):
        if allocation_group["groupId"] != group_id:
            continue
        old_allocations = deepcopy(allocation_group["allocations"])
        reconciled = reconcile_allocation_group(
            packages=packages,
            allocation_group=allocation_group,
        )
        patch["cargoAllocationGroups"][allocation_group_index] = reconciled
        for allocation_index, (old, new) in enumerate(
            zip(old_allocations, reconciled["allocations"], strict=True)
        ):
            if old.get("packageQuantity") == new.get("packageQuantity"):
                continue
            changes.append(
                _change(
                    path=(
                        f"documentPatch.cargoAllocationGroups[{allocation_group_index}]"
                        f".allocations[{allocation_index}].packageQuantity"
                    ),
                    family="allocation_quantity",
                    old=old["packageQuantity"],
                    new=new["packageQuantity"],
                    method="derive_reconciled_allocation_quantity_v1",
                    coupling_group="cargo_numeric_tuple",
                )
            )
    return tuple(changes)


def _pending_realizations(
    target: Mapping[str, Any], changed_paths: frozenset[str]
) -> tuple[PendingRealization, ...]:
    output = []
    for path, _value in leaf_items(target):
        if path in changed_paths:
            continue
        policy = policy_for_target_path(path)
        if policy.implementation_status in {"pending_registry", "pending_linguistic"}:
            kind = (
                "registry"
                if policy.implementation_status == "pending_registry"
                else "linguistic"
            )
        elif policy.implementation_status == "implemented" and policy.value_policy in {
            "regenerate",
            "resample",
        }:
            kind = "deterministic_unsupported"
        else:
            continue
        output.append(
            PendingRealization.model_validate(
                {
                    "target_path": path,
                    "role_path": policy.role_path,
                    "kind": kind,
                    "reason": f"{policy.method} is not realized by this single-family smoke draft",
                },
                strict=True,
            )
        )
    return tuple(sorted(output, key=lambda row: row.target_path))


def build_draft(
    *,
    row: Mapping[str, Any],
    template_id: str,
    family: str,
    seed: int,
    variant_index: int,
    date_minimum: date,
    date_maximum: date,
    donors: Sequence[NumericCargoTuple],
) -> tuple[dict[str, Any], DraftScenarioPlan]:
    """Build one schema-valid deterministic draft and expose all remaining work."""

    document_id = cast(str, row["documentId"])
    source_target = cast(dict[str, Any], row["target"])
    target = deepcopy(source_target)
    stream = DeterministicStream(seed, "mpci-bl-synthesis-v1", f"{document_id}:{variant_index}")
    if family == "container_identifier":
        changes = _container_changes(target, stream)
    elif family == "seal_identifier":
        changes = _seal_changes(target, stream)
    elif family == "document_dates":
        changes = _date_changes(target, stream, date_minimum, date_maximum)
    elif family == "package_quantity":
        changes = _numeric_changes(
            document_id=document_id,
            target=target,
            donors=donors,
            require_measures=False,
            stream=stream.derive("quantity"),
        )
    elif family == "cargo_mass":
        changes = _numeric_changes(
            document_id=document_id,
            target=target,
            donors=donors,
            require_measures=True,
            stream=stream.derive("mass"),
        )
    else:
        raise ValueError(f"unsupported deterministic mutation family: {family}")
    target_sha = sha256_bytes(canonical_json_bytes(target))
    validate_change_ledger(source_target, target, changes)
    if family in {"package_quantity", "cargo_mass"}:
        validate_allocation_arithmetic(target)
    synthetic_document_id = "syn_" + sha256_bytes(
        canonical_json_bytes([document_id, template_id, family, variant_index, seed, target_sha])
    )[:40]
    counts: dict[str, int] = defaultdict(int)
    for policy in FIELD_POLICIES.values():
        counts[policy.value_policy] += 1
    changed_paths = frozenset(change.target_path for change in changes)
    plan = DraftScenarioPlan.model_validate(
        {
            "schema_version": 1,
            "status": "draft_pending_realization",
            "synthetic_document_id": synthetic_document_id,
            "base_document_id": document_id,
            "template_id": template_id,
            "variant_index": variant_index,
            "seed": seed,
            "source_raw_text_sha256": cast(str, row["joinedRawTextSha256"]),
            "source_target_sha256": sha256_bytes(canonical_json_bytes(source_target)),
            "deterministic_target_sha256": target_sha,
            "changes": changes,
            "pending_realizations": _pending_realizations(target, changed_paths),
            "policy_counts": dict(sorted(counts.items())),
            "training_eligible": False,
        },
        strict=True,
    )
    return target, plan
