from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Protocol, cast

from pydantic import JsonValue, TypeAdapter

from document_ocr.hashing import canonical_json_bytes

from .models import (
    AggregateRangeConstraint,
    CoherenceCandidateDecision,
    CoherenceConstraint,
    InclusiveRangeConstraint,
    NumericSumConstraint,
    NumericValuesConstraint,
    ReviewedIndependentConstraint,
    ReviewRequiredConstraint,
)

_CARGO_TEXT_PATH = re.compile(
    r"^documentPatch\.cargoGroups\[([0-9]+)\]\."
    r"(?:description|additionalInformation\[[0-9]+\]|marksAndNumbers\[[0-9]+\]|"
    r"handlingInstructions\[[0-9]+\])$"
)
_CARGO_GROUP_PATH = re.compile(r"^documentPatch\.cargoGroups\[([0-9]+)\](?:\.|$)")
_PACKAGE_PATH = re.compile(r"^documentPatch\.cargoPackages\[([0-9]+)\](?:\.|$)")
_PACKAGE_QUANTITY_PATH = re.compile(r"^documentPatch\.cargoPackages\[([0-9]+)\]\.quantity$")
_ALLOCATION_PATH = re.compile(r"^documentPatch\.cargoAllocationGroups\[([0-9]+)\](?:\.|$)")
_ALLOCATION_QUANTITY_PATH = re.compile(
    r"^documentPatch\.cargoAllocationGroups\[([0-9]+)\]\."
    r"allocations\[([0-9]+)\]\.packageQuantity$"
)
_TARGET_PATH = re.compile(r"^documentPatch(?:\.[A-Za-z][A-Za-z0-9]*|\[[0-9]+\])+$")
_INTEGER_TOKEN = re.compile(r"(?<![A-Za-z0-9.])([0-9]+(?:,[0-9]{3})*)(?![A-Za-z0-9.]|,[0-9])")
_RANGE_SEPARATOR = r"(?:[-\u2013\u2014~]|\bTO\b)"
_REPEATED_PREFIX_RANGE = re.compile(
    rf"(?<![A-Za-z0-9])"
    rf"(?P<prefix>[A-Za-z][A-Za-z0-9._/]*?)"
    rf"(?P<start>[0-9]+(?:,[0-9]{{3}})*)"
    rf"[ \t\r\n]*{_RANGE_SEPARATOR}[ \t\r\n]*"
    rf"(?P=prefix)(?P<end>[0-9]+(?:,[0-9]{{3}})*)"
    rf"(?![A-Za-z0-9.]|,[0-9])",
    re.IGNORECASE,
)
_INTEGER_RANGE = re.compile(
    rf"(?<![A-Za-z0-9])"
    rf"(?P<start>[0-9]+(?:,[0-9]{{3}})*)"
    rf"[ \t\r\n]*{_RANGE_SEPARATOR}[ \t\r\n]*"
    rf"(?P<end>[0-9]+(?:,[0-9]{{3}})*)"
    rf"(?![A-Za-z0-9.]|,[0-9])",
    re.IGNORECASE,
)
_CONSTRAINT_ADAPTER: TypeAdapter[CoherenceConstraint] = TypeAdapter(CoherenceConstraint)


class BindingLike(Protocol):
    @property
    def logical_key(self) -> str: ...

    @property
    def render_mode(self) -> str: ...

    @property
    def value_kind(self) -> str: ...

    @property
    def group_kind(self) -> str: ...

    @property
    def group_key(self) -> str: ...

    @property
    def target_paths(self) -> tuple[str, ...]: ...

    @property
    def dependency_paths(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class _GroupedBinding:
    logical_key: str
    render_mode: str
    value_kind: str
    group_kind: str
    group_key: str
    target_paths: tuple[str, ...]
    dependency_paths: tuple[str, ...]
    occurrences: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class InclusiveRangeSurface:
    """One parsed inclusive integer interval with exact source offsets.

    ``repeated_prefix`` distinguishes identifier ranges such as ``SDW/1 TO SDW/80`` from
    ordinary numeric ranges. Rendering replaces only the final digits, preserving every prefix,
    separator, whitespace character, and OCR line break.
    """

    char_start: int
    char_end: int
    start_start: int
    start_end: int
    end_start: int
    end_end: int
    start: int
    end: int
    repeated_prefix: bool

    @property
    def cardinality(self) -> int:
        return abs(self.end - self.start) + 1

    def replace_end(self, source: str, replacement: str) -> str:
        if not (0 <= self.end_start < self.end_end <= len(source)):
            raise ValueError("inclusive-range endpoint offsets are outside the source")
        return source[: self.end_start] + replacement + source[self.end_end :]


@dataclass(frozen=True, slots=True)
class _PackageFact:
    index: int
    package_id: str
    group_id: str
    path: str
    value: int | None


@dataclass(frozen=True, slots=True)
class _QuantityRelation:
    relation_id: str
    group_id: str
    value: int
    dependency_paths: tuple[str, ...]
    topology: Literal[
        "allocation_leaf",
        "package_leaf",
        "single_package_total",
        "combined_package_total",
    ]


@dataclass(frozen=True, slots=True)
class _QuantityTopology:
    cargo_group_ids: Mapping[int, str]
    package_group_ids: Mapping[int, str]
    allocation_group_ids: Mapping[int, str]
    relations: tuple[_QuantityRelation, ...]


@dataclass(frozen=True, slots=True)
class _RangeAnalysis:
    binding: _GroupedBinding
    source_texts: tuple[str, ...]
    cardinalities: tuple[int, ...]
    contribution: int
    order: int


class CoherenceContractError(ValueError):
    """A candidate relationship lacks a complete, mechanically valid disposition."""


class CoherenceReviewRequired(ValueError):
    """A critic deliberately quarantined an ambiguous relationship."""


def _as_rows(value: Any, *, name: str) -> Sequence[Any]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"coherence {name} is not a row sequence")
    return value


def _required_group_id(row: Mapping[str, Any], *, name: str) -> str:
    value = row.get("groupId")
    if not isinstance(value, str) or not value:
        raise ValueError(f"coherence {name} lacks a valid groupId")
    return value


def _optional_integer(value: Any, *, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"coherence dependency is not an integer: {path}")
    return value


def _integer(value: Any, *, path: str) -> int:
    result = _optional_integer(value, path=path)
    if result is None:
        raise ValueError(f"coherence dependency is absent: {path}")
    return result


def _group_bindings(bindings: Sequence[BindingLike]) -> tuple[_GroupedBinding, ...]:
    grouped: dict[str, _GroupedBinding] = {}
    for binding in bindings:
        own_occurrences = getattr(binding, "occurrences", None)
        occurrences = (
            tuple(own_occurrences)
            if isinstance(own_occurrences, Sequence)
            and not isinstance(own_occurrences, (str, bytes))
            else (binding,)
        )
        row = _GroupedBinding(
            logical_key=binding.logical_key,
            render_mode=binding.render_mode,
            value_kind=binding.value_kind,
            group_kind=binding.group_kind,
            group_key=binding.group_key,
            target_paths=tuple(binding.target_paths),
            dependency_paths=tuple(binding.dependency_paths),
            occurrences=occurrences,
        )
        prior = grouped.get(row.logical_key)
        if prior is None:
            grouped[row.logical_key] = row
            continue
        if (
            row.render_mode,
            row.value_kind,
            row.group_kind,
            row.group_key,
            row.target_paths,
            row.dependency_paths,
        ) != (
            prior.render_mode,
            prior.value_kind,
            prior.group_kind,
            prior.group_key,
            prior.target_paths,
            prior.dependency_paths,
        ):
            raise CoherenceContractError(
                f"logical binding has inconsistent coherence topology: {row.logical_key}"
            )
        grouped[row.logical_key] = _GroupedBinding(
            logical_key=prior.logical_key,
            render_mode=prior.render_mode,
            value_kind=prior.value_kind,
            group_kind=prior.group_kind,
            group_key=prior.group_key,
            target_paths=prior.target_paths,
            dependency_paths=prior.dependency_paths,
            occurrences=(*prior.occurrences, *row.occurrences),
        )
    return tuple(grouped[key] for key in sorted(grouped))


def _resolve_path(target: Mapping[str, Any], path: str) -> JsonValue:
    if _TARGET_PATH.fullmatch(path) is None:
        raise ValueError(f"invalid coherence target path: {path}")
    current: Any = target
    tokens = re.findall(r"([A-Za-z][A-Za-z0-9]*)(?:\[([0-9]+)\])?", path)
    for name, index_text in tokens:
        if not isinstance(current, Mapping) or name not in current:
            raise ValueError(f"coherence path is absent: {path}")
        current = current[name]
        if index_text:
            index = int(index_text)
            if (
                not isinstance(current, Sequence)
                or isinstance(current, (str, bytes))
                or index >= len(current)
            ):
                raise ValueError(f"coherence path index is absent: {path}")
            current = current[index]
    return cast(JsonValue, current)


def integer_tokens(value: str) -> tuple[int, ...]:
    return tuple(int(match.group(1).replace(",", "")) for match in _INTEGER_TOKEN.finditer(value))


def inclusive_range_surfaces(value: str) -> tuple[InclusiveRangeSurface, ...]:
    """Parse plain and repeated-prefix integer intervals without double-counting overlaps."""

    parsed: list[InclusiveRangeSurface] = []
    for pattern, repeated_prefix in (
        (_REPEATED_PREFIX_RANGE, True),
        (_INTEGER_RANGE, False),
    ):
        for match in pattern.finditer(value):
            char_start, char_end = match.span()
            if any(
                char_start < existing.char_end and existing.char_start < char_end
                for existing in parsed
            ):
                continue
            start_text = match.group("start")
            end_text = match.group("end")
            parsed.append(
                InclusiveRangeSurface(
                    char_start=char_start,
                    char_end=char_end,
                    start_start=match.start("start"),
                    start_end=match.end("start"),
                    end_start=match.start("end"),
                    end_end=match.end("end"),
                    start=int(start_text.replace(",", "")),
                    end=int(end_text.replace(",", "")),
                    repeated_prefix=repeated_prefix,
                )
            )
    return tuple(sorted(parsed, key=lambda row: (row.char_start, row.char_end)))


def whole_inclusive_range_surface(value: str) -> InclusiveRangeSurface | None:
    """Return the sole complete formal range surface, including supported count captions."""

    parsed = inclusive_range_surfaces(value)
    if len(parsed) != 1:
        return None
    surface = parsed[0]
    leading = value[: surface.char_start]
    trailing = value[surface.char_end :]
    if trailing.strip():
        return None
    if surface.repeated_prefix:
        return surface if not leading.strip() else None
    if (
        re.fullmatch(
            r"(?i)[ \t\r\n]*(?:(?:NO(?:S)?|NUM(?:BER)?S?)\.?[ \t\r\n]*|#[ \t\r\n]*)?",
            leading,
        )
        is None
    ):
        return None
    return surface


def inclusive_range_cardinalities(value: str) -> tuple[int, ...]:
    return tuple(surface.cardinality for surface in inclusive_range_surfaces(value))


def _quantity_topology(source_target: Mapping[str, Any]) -> _QuantityTopology:
    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        return _QuantityTopology({}, {}, {}, ())

    cargo_group_ids: dict[int, str] = {}
    for index, raw_row in enumerate(_as_rows(patch.get("cargoGroups"), name="cargoGroups")):
        if not isinstance(raw_row, Mapping):
            raise ValueError("coherence cargo group is not an object")
        cargo_group_ids[index] = _required_group_id(raw_row, name="cargo group")

    packages: list[_PackageFact] = []
    package_by_id: dict[str, _PackageFact] = {}
    package_group_ids: dict[int, str] = {}
    for index, raw_row in enumerate(_as_rows(patch.get("cargoPackages"), name="cargoPackages")):
        if not isinstance(raw_row, Mapping):
            raise ValueError("coherence cargo package is not an object")
        group_id = _required_group_id(raw_row, name="cargo package")
        if cargo_group_ids and group_id not in set(cargo_group_ids.values()):
            raise ValueError("coherence cargo package references an absent group")
        package_id = raw_row.get("packageId")
        if not isinstance(package_id, str) or not package_id:
            raise ValueError("coherence cargo package lacks a valid packageId")
        path = f"documentPatch.cargoPackages[{index}].quantity"
        fact = _PackageFact(
            index=index,
            package_id=package_id,
            group_id=group_id,
            path=path,
            value=_optional_integer(raw_row.get("quantity"), path=path),
        )
        if package_id in package_by_id:
            raise ValueError("coherence cargo package IDs are not unique")
        packages.append(fact)
        package_by_id[package_id] = fact
        package_group_ids[index] = group_id

    allocation_group_ids: dict[int, str] = {}
    seen_allocation_group_ids: set[str] = set()
    relations: list[_QuantityRelation] = []
    covered_packages: set[str] = set()
    for group_index, raw_group in enumerate(
        _as_rows(patch.get("cargoAllocationGroups"), name="cargoAllocationGroups")
    ):
        if not isinstance(raw_group, Mapping):
            raise ValueError("coherence cargo allocation group is not an object")
        group_id = _required_group_id(raw_group, name="cargo allocation group")
        if cargo_group_ids and group_id not in set(cargo_group_ids.values()):
            raise ValueError("coherence cargo allocation references an absent group")
        if group_id in seen_allocation_group_ids:
            raise ValueError("coherence cargo group has multiple allocation groups")
        seen_allocation_group_ids.add(group_id)
        allocation_group_ids[group_index] = group_id
        coverage = raw_group.get("coverage")
        if coverage not in {
            "one_to_one_package_allocations",
            "single_package_level",
            "all_package_levels_combined",
            "unlinked_package_quantities",
            "container_membership_only",
        }:
            raise ValueError("coherence cargo allocation group has invalid coverage")
        package_ids = tuple(
            value
            for value in _as_rows(raw_group.get("packageIds"), name="allocation packageIds")
            if isinstance(value, str)
        )
        if len(package_ids) != len(
            _as_rows(raw_group.get("packageIds"), name="allocation packageIds")
        ):
            raise ValueError("coherence allocation packageId is not a string")
        if len(set(package_ids)) != len(package_ids):
            raise ValueError("coherence allocation packageIds are not unique")
        referenced: list[_PackageFact] = []
        for package_id in package_ids:
            package = package_by_id.get(package_id)
            if package is None or package.group_id != group_id:
                raise ValueError("coherence allocation references an absent or foreign package")
            referenced.append(package)

        allocations = _as_rows(raw_group.get("allocations"), name="allocations")
        allocation_values: list[int] = []
        allocation_value_rows: list[int | None] = []
        allocation_package_ids: list[str | None] = []
        for allocation_index, raw_allocation in enumerate(allocations):
            if not isinstance(raw_allocation, Mapping):
                raise ValueError("coherence allocation is not an object")
            path = (
                f"documentPatch.cargoAllocationGroups[{group_index}]."
                f"allocations[{allocation_index}].packageQuantity"
            )
            value = _optional_integer(raw_allocation.get("packageQuantity"), path=path)
            allocation_value_rows.append(value)
            allocation_package_id = raw_allocation.get("packageId")
            if allocation_package_id is not None and not isinstance(allocation_package_id, str):
                raise ValueError("coherence allocation packageId is not a string")
            allocation_package_ids.append(allocation_package_id)
            if value is None:
                continue
            allocation_values.append(value)
            relations.append(
                _QuantityRelation(
                    relation_id=f"allocation:{group_index}:{allocation_index}",
                    group_id=group_id,
                    value=value,
                    dependency_paths=(path,),
                    topology="allocation_leaf",
                )
            )

        if coverage == "one_to_one_package_allocations":
            if (
                len(package_ids) != len(allocations)
                or tuple(allocation_package_ids) != package_ids
                or any(value is None for value in allocation_value_rows)
            ):
                raise ValueError("coherence one-to-one allocation shape is invalid")
            for package_id, value in zip(package_ids, allocation_value_rows, strict=True):
                if package_by_id[package_id].value != value:
                    raise ValueError("coherence one-to-one allocation quantity differs")
        elif coverage == "single_package_level":
            if len(package_ids) != 1 or any(value is None for value in allocation_value_rows):
                raise ValueError("coherence single-package allocation shape is invalid")
            if referenced[0].value is None or sum(allocation_values) != referenced[0].value:
                raise ValueError("coherence single-package allocation total differs")
        elif coverage == "all_package_levels_combined":
            if len(package_ids) < 2 or any(value is None for value in allocation_value_rows):
                raise ValueError("coherence combined allocation shape is invalid")
            if any(package.value is None for package in referenced) or sum(
                allocation_values
            ) != sum(cast(int, package.value) for package in referenced):
                raise ValueError("coherence combined allocation total differs")
        elif coverage == "unlinked_package_quantities":
            if package_ids or any(value is None for value in allocation_value_rows):
                raise ValueError("coherence unlinked allocation shape is invalid")
        elif package_ids or any(value is not None for value in allocation_value_rows):
            raise ValueError("coherence membership-only allocation shape is invalid")

        if coverage in {
            "single_package_level",
            "one_to_one_package_allocations",
            "all_package_levels_combined",
        }:
            covered_packages.update(package_ids)
        if coverage == "single_package_level" and len(allocation_values) > 1:
            package = referenced[0]
            if package.value is None:
                raise ValueError("coherence covered package quantity is absent")
            relations.append(
                _QuantityRelation(
                    relation_id=f"single-package-total:{package.package_id}",
                    group_id=group_id,
                    value=package.value,
                    dependency_paths=(package.path,),
                    topology="single_package_total",
                )
            )
        elif coverage == "all_package_levels_combined":
            if any(package.value is None for package in referenced):
                raise ValueError("coherence covered package quantity is absent")
            relations.append(
                _QuantityRelation(
                    relation_id=f"combined-package-total:{group_index}",
                    group_id=group_id,
                    value=sum(cast(int, package.value) for package in referenced),
                    dependency_paths=tuple(package.path for package in referenced),
                    topology="combined_package_total",
                )
            )

    for package in packages:
        if package.value is None or package.package_id in covered_packages:
            continue
        relations.append(
            _QuantityRelation(
                relation_id=f"package:{package.package_id}",
                group_id=package.group_id,
                value=package.value,
                dependency_paths=(package.path,),
                topology="package_leaf",
            )
        )

    identities = tuple(row.relation_id for row in relations)
    if len(set(identities)) != len(identities):
        raise ValueError("coherence quantity relation identities are not unique")
    return _QuantityTopology(
        cargo_group_ids=cargo_group_ids,
        package_group_ids=package_group_ids,
        allocation_group_ids=allocation_group_ids,
        relations=tuple(relations),
    )


def _physical_source_texts(binding: _GroupedBinding) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            source_text
            for occurrence in binding.occurrences
            if isinstance((source_text := getattr(occurrence, "source_text", None)), str)
        )
    )


def _binding_source_texts(
    binding: _GroupedBinding, source_target: Mapping[str, Any]
) -> tuple[str, ...]:
    target_texts: list[str] = []
    for path in binding.target_paths:
        if _CARGO_TEXT_PATH.fullmatch(path) is None:
            continue
        value = _resolve_path(source_target, path)
        if isinstance(value, str):
            target_texts.append(value)
    return tuple(target_texts) if target_texts else _physical_source_texts(binding)


def _binding_group_id(binding: _GroupedBinding, *, topology: _QuantityTopology) -> str | None:
    candidates: set[str] = set()
    for path in (*binding.target_paths, *binding.dependency_paths):
        if (match := _CARGO_GROUP_PATH.match(path)) is not None:
            resolved_group_id = topology.cargo_group_ids.get(int(match.group(1)))
        elif (match := _PACKAGE_PATH.match(path)) is not None:
            resolved_group_id = topology.package_group_ids.get(int(match.group(1)))
        elif (match := _ALLOCATION_PATH.match(path)) is not None:
            resolved_group_id = topology.allocation_group_ids.get(int(match.group(1)))
        else:
            resolved_group_id = None
        if resolved_group_id is not None:
            candidates.add(resolved_group_id)

    prefix, separator, suffix = binding.group_key.partition(":")
    token = suffix.split(":", 1)[0] if separator else ""
    known_group_ids = set(topology.cargo_group_ids.values())
    key_group_id: str | None = None
    if prefix == "cargo":
        if token in known_group_ids:
            key_group_id = token
        elif token.isdigit():
            key_group_id = topology.cargo_group_ids.get(int(token))
    elif prefix == "package" and token.isdigit():
        key_group_id = topology.package_group_ids.get(int(token))
    elif prefix == "allocation" and token.isdigit():
        key_group_id = topology.allocation_group_ids.get(int(token))
    if key_group_id is not None:
        candidates.add(key_group_id)
    if not candidates and binding.group_kind in {"cargo", "package"}:
        relation_groups = {row.group_id for row in topology.relations}
        if len(relation_groups) == 1:
            candidates.update(relation_groups)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _has_cargo_text_topology(binding: _GroupedBinding) -> bool:
    if any(_CARGO_TEXT_PATH.fullmatch(path) for path in binding.target_paths):
        return True
    return (
        not binding.target_paths
        and binding.group_kind in {"cargo", "package"}
        and binding.value_kind
        in {
            "cargo_text",
            "package",
            "integer",
            "other_text",
            "operational_text",
            "commercial_text",
            "identifier",
        }
    )


def _line_ids(raw: str, binding: _GroupedBinding) -> tuple[str, ...]:
    output: set[str] = set()
    for occurrence in binding.occurrences:
        char_start = getattr(occurrence, "char_start", None)
        byte_start = getattr(occurrence, "byte_start", None)
        if isinstance(char_start, int):
            output.add(f"L{raw.count(chr(10), 0, char_start) + 1:05d}")
        elif isinstance(byte_start, int):
            try:
                prefix = raw.encode("utf-8")[:byte_start].decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("coherence slot byte boundary is not UTF-8 aligned") from error
            output.add(f"L{prefix.count(chr(10)) + 1:05d}")
    return tuple(sorted(output))


def _binding_order(binding: _GroupedBinding) -> int:
    offsets = tuple(
        value
        for occurrence in binding.occurrences
        if isinstance((value := getattr(occurrence, "char_start", None)), int)
    )
    if offsets:
        return min(offsets)
    byte_offsets = tuple(
        value
        for occurrence in binding.occurrences
        if isinstance((value := getattr(occurrence, "byte_start", None)), int)
    )
    return min(byte_offsets) if byte_offsets else 2**63 - 1


def _candidate_fingerprint(
    *, kind: str, logical_keys: Sequence[str], suggestions: Sequence[Mapping[str, Any]]
) -> str:
    return sha256(
        canonical_json_bytes(
            {
                "kind": kind,
                "logicalKeys": tuple(logical_keys),
                "suggestedContracts": tuple(suggestions),
            }
        )
    ).hexdigest()


def _candidate(
    *,
    raw: str,
    kind: str,
    bindings: Sequence[_GroupedBinding],
    source_target: Mapping[str, Any],
    suggested_contracts: Sequence[Mapping[str, Any]],
    relation_rows: Sequence[_QuantityRelation],
    relationship: str,
    constraints_by_fingerprint: Mapping[str, CoherenceConstraint],
) -> dict[str, Any]:
    logical_keys = tuple(binding.logical_key for binding in bindings)
    suggestions = tuple(suggested_contracts)
    fingerprint = _candidate_fingerprint(
        kind=kind,
        logical_keys=logical_keys,
        suggestions=suggestions,
    )
    existing = constraints_by_fingerprint.get(fingerprint)
    source_texts = tuple(
        text for binding in bindings for text in _binding_source_texts(binding, source_target)
    )
    return {
        "kind": "cross_field_semantic_relation",
        "relationshipKind": kind,
        "coherenceFingerprint": fingerprint,
        "logicalKeys": logical_keys,
        "lineIds": tuple(
            sorted({line for binding in bindings for line in _line_ids(raw, binding)})
        ),
        "sourceTexts": source_texts,
        "requiredRevision": existing is None,
        "details": {
            "suggestedContracts": suggestions,
            "ambiguousAlternatives": len(suggestions) > 1,
            "structuredQuantities": tuple(
                {
                    "suggestionIndex": index,
                    "relationId": row.relation_id,
                    "topology": row.topology,
                    "value": row.value,
                    "dependencyPaths": row.dependency_paths,
                }
                for index, row in enumerate(relation_rows)
            ),
            "members": tuple(
                {
                    "logicalKey": binding.logical_key,
                    "lineIds": _line_ids(raw, binding),
                    "sourceTexts": _binding_source_texts(binding, source_target),
                    "rangeCardinalities": tuple(
                        value
                        for text in _binding_source_texts(binding, source_target)
                        for value in inclusive_range_cardinalities(text)
                    ),
                }
                for binding in bindings
            ),
            "currentConstraint": (
                existing.model_dump(mode="json") if existing is not None else None
            ),
            "relationshipToVerify": relationship,
        },
    }


def _suggestion(
    *, kind: str, bindings: Sequence[_GroupedBinding], relation: _QuantityRelation
) -> dict[str, Any]:
    return {
        "kind": kind,
        "memberLogicalKeys": tuple(binding.logical_key for binding in bindings),
        "dependencyPaths": relation.dependency_paths,
    }


def coherence_review_candidates(
    *,
    raw: str,
    bindings: Sequence[BindingLike],
    source_target: Mapping[str, Any],
    constraints: Sequence[CoherenceConstraint] = (),
) -> tuple[dict[str, Any], ...]:
    """Build bounded, topology-scoped arithmetic leads without vocabulary special cases."""

    logical_bindings = _group_bindings(bindings)
    topology = _quantity_topology(source_target)
    if not topology.relations:
        return ()
    relations_by_group: dict[str, list[_QuantityRelation]] = defaultdict(list)
    for relation in topology.relations:
        relations_by_group[relation.group_id].append(relation)
    constraints_by_fingerprint = {row.candidate_fingerprint: row for row in constraints}
    analyses_by_group: dict[str, list[_RangeAnalysis]] = defaultdict(list)
    numeric_by_group: dict[str, list[tuple[_GroupedBinding, tuple[str, ...], tuple[int, ...]]]] = (
        defaultdict(list)
    )

    for binding in logical_bindings:
        if binding.render_mode in {
            "carrier_static",
            "literal_static",
            "deterministic_derived",
        } or not _has_cargo_text_topology(binding):
            continue
        group_id = _binding_group_id(binding, topology=topology)
        if group_id is None or group_id not in relations_by_group:
            continue
        source_texts = _binding_source_texts(binding, source_target)
        if not source_texts:
            continue
        cardinalities = tuple(
            value for text in source_texts for value in inclusive_range_cardinalities(text)
        )
        if cardinalities:
            analyses_by_group[group_id].append(
                _RangeAnalysis(
                    binding=binding,
                    source_texts=source_texts,
                    cardinalities=cardinalities,
                    contribution=sum(cardinalities),
                    order=_binding_order(binding),
                )
            )
            continue
        if binding.value_kind == "identifier" and not any(
            _CARGO_TEXT_PATH.fullmatch(path) for path in binding.target_paths
        ):
            continue
        numbers = tuple(value for text in source_texts for value in integer_tokens(text))
        if numbers:
            numeric_by_group[group_id].append((binding, source_texts, numbers))

    rows: list[dict[str, Any]] = []
    for group_id, raw_analyses in sorted(analyses_by_group.items()):
        analyses = sorted(raw_analyses, key=lambda row: (row.order, row.binding.logical_key))
        relations = tuple(relations_by_group[group_id])
        relation_values = {row.value for row in relations}
        for start in range(len(analyses)):
            subtotal = 0
            cardinality_count = 0
            for end in range(start, len(analyses)):
                subtotal += analyses[end].contribution
                cardinality_count += len(analyses[end].cardinalities)
                if subtotal not in relation_values:
                    continue
                members = tuple(row.binding for row in analyses[start : end + 1])
                kind = (
                    "inclusive_range_cardinality"
                    if len(members) == 1 and cardinality_count == 1
                    else "aggregate_inclusive_range_cardinality"
                )
                matching = tuple(row for row in relations if row.value == subtotal)
                suggestions = tuple(
                    _suggestion(kind=kind, bindings=members, relation=relation)
                    for relation in matching
                )
                rows.append(
                    _candidate(
                        raw=raw,
                        kind=kind,
                        bindings=members,
                        source_target=source_target,
                        suggested_contracts=suggestions,
                        relation_rows=matching,
                        relationship=(
                            "Decide whether these source-ordered inclusive intervals print the "
                            "host-listed structured quantity. The intervals may participate in "
                            "other overlapping subtotal or total relationships."
                        ),
                        constraints_by_fingerprint=constraints_by_fingerprint,
                    )
                )

    for group_id, bindings_with_numbers in sorted(numeric_by_group.items()):
        relations = tuple(relations_by_group[group_id])
        for binding, _source_texts, numbers in bindings_with_numbers:
            matching = tuple(row for row in relations if row.value in set(numbers))
            if not matching:
                continue
            suggestions = tuple(
                _suggestion(
                    kind=(
                        "numeric_values"
                        if len(relation.dependency_paths) == 1
                        else "summed_numeric_value"
                    ),
                    bindings=(binding,),
                    relation=relation,
                )
                for relation in matching
            )
            rows.append(
                _candidate(
                    raw=raw,
                    kind="structured_numeric_value",
                    bindings=(binding,),
                    source_target=source_target,
                    suggested_contracts=suggestions,
                    relation_rows=matching,
                    relationship=(
                        "Decide whether this cargo text deliberately prints one host-listed "
                        "structured quantity or merely contains coincident digits."
                    ),
                    constraints_by_fingerprint=constraints_by_fingerprint,
                )
            )

    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(cast(str, row["coherenceFingerprint"]), row)
    return tuple(
        sorted(
            unique.values(),
            key=lambda row: (
                row["relationshipKind"],
                tuple(row["logicalKeys"]),
                row["coherenceFingerprint"],
            ),
        )
    )


def _constraint_id(payload: Mapping[str, Any]) -> str:
    return "coherence_constraint_" + sha256(canonical_json_bytes(payload)).hexdigest()[:16]


def materialize_coherence_decisions(
    *,
    candidates: Sequence[Mapping[str, Any]],
    constraints: Sequence[CoherenceConstraint],
    decisions: Sequence[CoherenceCandidateDecision],
) -> tuple[CoherenceConstraint, ...]:
    coherence_candidates = {
        cast(str, row["candidateId"]): row
        for row in candidates
        if row.get("kind") == "cross_field_semantic_relation"
    }
    decision_ids = tuple(row.candidate_id for row in decisions)
    if len(set(decision_ids)) != len(decision_ids):
        raise CoherenceContractError("coherence decision candidate IDs must be unique")
    unknown = sorted(set(decision_ids) - set(coherence_candidates))
    if unknown:
        raise CoherenceContractError(
            "coherence decisions reference unknown candidates: " + ", ".join(unknown)
        )
    required = {
        candidate_id
        for candidate_id, row in coherence_candidates.items()
        if row.get("requiredRevision") is True
    }
    missing = sorted(required - set(decision_ids))
    if missing:
        raise CoherenceContractError(
            "critic omitted required coherence decisions: " + ", ".join(missing)
        )

    by_fingerprint = {row.candidate_fingerprint: row for row in constraints}
    for decision in decisions:
        candidate = coherence_candidates[decision.candidate_id]
        fingerprint = cast(str, candidate["coherenceFingerprint"])
        candidate_members = tuple(cast(Sequence[str], candidate["logicalKeys"]))
        by_fingerprint.pop(fingerprint, None)
        common: dict[str, Any] = {
            "candidate_fingerprint": fingerprint,
            "member_logical_keys": candidate_members,
            "rationale": decision.rationale,
        }
        if decision.disposition == "reviewed_independent":
            payload: dict[str, Any] = {**common, "kind": "reviewed_independent"}
        elif decision.disposition == "review_required":
            payload = {**common, "kind": "review_required"}
        else:
            details = candidate.get("details")
            suggestions = (
                details.get("suggestedContracts", ()) if isinstance(details, Mapping) else ()
            )
            if not isinstance(suggestions, (tuple, list)):
                raise CoherenceContractError("coherence candidate suggestions are invalid")
            index = cast(int, decision.suggestion_index)
            if index >= len(suggestions) or not isinstance(suggestions[index], Mapping):
                raise CoherenceContractError(
                    f"coherence suggestion index is invalid: {decision.candidate_id}={index}"
                )
            selected = dict(cast(Mapping[str, Any], suggestions[index]))
            selected_members = tuple(cast(Sequence[str], selected.pop("memberLogicalKeys")))
            selected_dependencies = tuple(cast(Sequence[str], selected.pop("dependencyPaths")))
            if selected_members != candidate_members:
                raise CoherenceContractError(
                    "coherence suggestion members differ from candidate evidence"
                )
            payload = {
                **common,
                **selected,
                "member_logical_keys": selected_members,
                "dependency_paths": selected_dependencies,
            }
        payload["constraint_id"] = _constraint_id(payload)
        by_fingerprint[fingerprint] = _CONSTRAINT_ADAPTER.validate_python(payload, strict=True)
    return tuple(sorted(by_fingerprint.values(), key=lambda row: row.constraint_id))


def reconcile_coherence_constraints_after_binding_revision(
    *,
    raw: str,
    bindings: Sequence[BindingLike],
    source_target: Mapping[str, Any],
    constraints: Sequence[CoherenceConstraint],
) -> tuple[CoherenceConstraint, ...]:
    """Retain only decisions whose exact evidence survives a structural binding revision.

    A critic transaction is evaluated against the pre-transaction inventory. If that transaction
    splits, removes, or regroups a relationship member, its old fingerprint no longer describes
    the revised template. The next independent critic pass must classify the newly exposed
    relationship; carrying the stale decision would be unsound, while rejecting the otherwise
    valid structural transaction would require the model to decide evidence that did not yet
    exist. Unrelated decisions retain their exact fingerprints and remain in force.
    """

    current_fingerprints = {
        cast(str, row["coherenceFingerprint"])
        for row in coherence_review_candidates(
            raw=raw,
            bindings=bindings,
            source_target=source_target,
            constraints=constraints,
        )
    }
    return tuple(
        constraint
        for constraint in constraints
        if constraint.candidate_fingerprint in current_fingerprints
    )


def coherence_dependency_paths(constraint: CoherenceConstraint) -> tuple[str, ...]:
    if isinstance(
        constraint,
        (
            NumericValuesConstraint,
            NumericSumConstraint,
            InclusiveRangeConstraint,
            AggregateRangeConstraint,
        ),
    ):
        return constraint.dependency_paths
    return ()


def constraint_dependency_paths_for_binding(
    constraints: Sequence[CoherenceConstraint], logical_key: str
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            path
            for constraint in constraints
            if logical_key in constraint.member_logical_keys
            for path in coherence_dependency_paths(constraint)
        )
    )


def _suggestion_signature(constraint: CoherenceConstraint) -> dict[str, Any]:
    return {
        "kind": constraint.kind,
        "memberLogicalKeys": constraint.member_logical_keys,
        "dependencyPaths": coherence_dependency_paths(constraint),
    }


def _constraint_member_texts(
    constraint: CoherenceConstraint,
    by_key: Mapping[str, _GroupedBinding],
    source_target: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        _binding_source_texts(by_key[key], source_target) for key in constraint.member_logical_keys
    )


def _validate_typed_constraint(
    *,
    constraint: CoherenceConstraint,
    member_texts: Sequence[Sequence[str]],
    target: Mapping[str, Any],
) -> None:
    texts = tuple(text for member in member_texts for text in member)
    if isinstance(constraint, NumericValuesConstraint):
        observed_values = {number for text in texts for number in integer_tokens(text)}
        expected_values = {
            _integer(_resolve_path(target, path), path=path) for path in constraint.dependency_paths
        }
        if not expected_values <= observed_values:
            raise ValueError(
                "coherent text omits structured numeric values: "
                + ", ".join(str(value) for value in sorted(expected_values - observed_values))
            )
    elif isinstance(constraint, NumericSumConstraint):
        expected_sum = sum(
            _integer(_resolve_path(target, path), path=path) for path in constraint.dependency_paths
        )
        observed_values = {number for text in texts for number in integer_tokens(text)}
        if expected_sum not in observed_values:
            raise ValueError(f"coherent text omits structured numeric sum {expected_sum}")
    elif isinstance(constraint, InclusiveRangeConstraint):
        path = constraint.dependency_paths[0]
        expected_cardinality = _integer(_resolve_path(target, path), path=path)
        observed_cardinalities = {
            value for text in texts for value in inclusive_range_cardinalities(text)
        }
        if expected_cardinality not in observed_cardinalities:
            raise ValueError(
                f"coherent interval cardinality does not equal {path}={expected_cardinality}"
            )
    elif isinstance(constraint, AggregateRangeConstraint):
        expected_total = sum(
            _integer(_resolve_path(target, path), path=path) for path in constraint.dependency_paths
        )
        observed_total = sum(
            value
            for member in member_texts
            for text in member
            for value in inclusive_range_cardinalities(text)
        )
        if observed_total != expected_total:
            raise ValueError(
                f"aggregate interval cardinality {observed_total} does not equal structured total "
                f"{expected_total}"
            )


def validate_coherence_contracts(
    *,
    raw: str,
    bindings: Sequence[BindingLike],
    source_target: Mapping[str, Any],
    constraints: Sequence[CoherenceConstraint],
    require_complete: bool = True,
) -> None:
    logical_bindings = _group_bindings(bindings)
    by_key = {binding.logical_key: binding for binding in logical_bindings}
    candidates = coherence_review_candidates(
        raw=raw,
        bindings=logical_bindings,
        source_target=source_target,
        constraints=constraints,
    )
    candidate_by_fingerprint = {cast(str, row["coherenceFingerprint"]): row for row in candidates}
    constraint_by_fingerprint: dict[str, CoherenceConstraint] = {}
    errors: list[str] = []
    reviews: list[str] = []
    for constraint in constraints:
        if constraint.candidate_fingerprint in constraint_by_fingerprint:
            errors.append(
                "duplicate coherence constraint fingerprint: " + constraint.candidate_fingerprint
            )
            continue
        constraint_by_fingerprint[constraint.candidate_fingerprint] = constraint
        candidate = candidate_by_fingerprint.get(constraint.candidate_fingerprint)
        if candidate is None:
            errors.append(
                "coherence constraint has no current arithmetic evidence: "
                + constraint.constraint_id
            )
            continue
        candidate_members = tuple(cast(Sequence[str], candidate["logicalKeys"]))
        if constraint.member_logical_keys != candidate_members:
            errors.append(f"{constraint.constraint_id} member bindings differ from evidence")
        unknown_members = sorted(set(constraint.member_logical_keys) - set(by_key))
        if unknown_members:
            errors.append(
                f"{constraint.constraint_id} references unknown bindings: "
                + ", ".join(unknown_members)
            )
            continue
        if isinstance(constraint, ReviewRequiredConstraint):
            reviews.append(f"{constraint.constraint_id}: {constraint.rationale}")
            continue
        if isinstance(constraint, ReviewedIndependentConstraint):
            continue
        source_only_non_residual = tuple(
            key
            for key in constraint.member_logical_keys
            if not by_key[key].target_paths and by_key[key].render_mode != "agent_residual"
        )
        if source_only_non_residual:
            errors.append(
                f"{constraint.constraint_id} source-only coherence members require "
                "agent_residual rendering: " + ", ".join(source_only_non_residual)
            )
            continue
        details = candidate.get("details")
        suggestions = (
            cast(Sequence[Mapping[str, Any]], details.get("suggestedContracts", ()))
            if isinstance(details, Mapping)
            else ()
        )
        if not any(_suggestion_signature(constraint) == dict(row) for row in suggestions):
            errors.append(
                f"{constraint.constraint_id} does not match its host-supplied arithmetic evidence"
            )
            continue
        try:
            for path in coherence_dependency_paths(constraint):
                _integer(_resolve_path(source_target, path), path=path)
            _validate_typed_constraint(
                constraint=constraint,
                member_texts=_constraint_member_texts(
                    constraint, by_key=by_key, source_target=source_target
                ),
                target=source_target,
            )
        except ValueError as error:
            errors.append(f"{constraint.constraint_id}: {error}")

    missing = sorted(set(candidate_by_fingerprint) - set(constraint_by_fingerprint))
    if require_complete and missing:
        errors.append("coherence candidates lack dispositions: " + ", ".join(missing))
    if errors:
        raise CoherenceContractError("; ".join(errors))
    if reviews:
        raise CoherenceReviewRequired("; ".join(reviews))


def _texts_from_target(binding: _GroupedBinding, target: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        value
        for path in binding.target_paths
        if _CARGO_TEXT_PATH.fullmatch(path)
        and isinstance((value := _resolve_path(target, path)), str)
    )


def _texts_from_output(binding: _GroupedBinding, output: Any) -> tuple[str, ...]:
    canonical = getattr(output, "canonical_value", None)
    if binding.target_paths:
        if isinstance(canonical, str) and any(
            _CARGO_TEXT_PATH.fullmatch(p) for p in binding.target_paths
        ):
            return (canonical,)
        if isinstance(canonical, Sequence) and not isinstance(canonical, (str, bytes)):
            canonical_texts = tuple(
                value
                for path, value in zip(binding.target_paths, canonical, strict=True)
                if _CARGO_TEXT_PATH.fullmatch(path) and isinstance(value, str)
            )
            if canonical_texts:
                return tuple(dict.fromkeys(canonical_texts))
    replacements = getattr(output, "replacements", None)
    if not isinstance(replacements, Mapping):
        raise ValueError(f"binding output lacks replacements: {binding.logical_key}")
    values = tuple(value for value in replacements.values() if isinstance(value, str))
    if not values:
        raise ValueError(f"binding output has no text surfaces: {binding.logical_key}")
    return tuple(dict.fromkeys(values))


def validate_render_coherence(
    *,
    bindings: Sequence[BindingLike],
    constraints: Sequence[CoherenceConstraint],
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    outputs: Mapping[str, Any] | None,
) -> None:
    by_key = {binding.logical_key: binding for binding in _group_bindings(bindings)}
    for constraint in constraints:
        if isinstance(constraint, ReviewRequiredConstraint):
            raise CoherenceReviewRequired(f"{constraint.constraint_id}: {constraint.rationale}")
        if isinstance(constraint, ReviewedIndependentConstraint):
            continue
        member_texts: list[tuple[str, ...]] = []
        includes_unrendered_source_only = False
        dependency_changed = any(
            _resolve_path(source_target, path) != _resolve_path(target, path)
            for path in coherence_dependency_paths(constraint)
        )
        for key in constraint.member_logical_keys:
            binding = by_key[key]
            if outputs is not None and key in outputs:
                texts = _texts_from_output(binding, outputs[key])
            elif binding.target_paths:
                texts = _texts_from_target(binding, target)
                if not texts:
                    includes_unrendered_source_only = True
                    texts = _physical_source_texts(binding)
            else:
                includes_unrendered_source_only = True
                texts = _physical_source_texts(binding)
            if not texts:
                raise ValueError(f"coherent binding has no text surface: {key}")
            member_texts.append(texts)
            if (
                binding.target_paths
                and dependency_changed
                and not isinstance(constraint, AggregateRangeConstraint)
            ):
                source_texts = _texts_from_target(binding, source_target)
                if source_texts and texts == source_texts:
                    raise ValueError(
                        f"coherence dependency changed while target text stayed unchanged: {key}"
                    )
        if outputs is None and includes_unrendered_source_only:
            continue
        _validate_typed_constraint(
            constraint=constraint,
            member_texts=member_texts,
            target=target,
        )
