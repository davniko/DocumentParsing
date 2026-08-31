"""Restricted, auditable synthesis of country-form cargo origins.

Cargo origin is a goods-level fact, not a synonym for either the shipper's
commercial country or a physical route endpoint.  This module therefore fits
only the observed relation between an explicitly printed, ISO-resolvable
country-form origin and a resolved shipper country. Any name-only source form
is nevertheless safe to replace with a generated country: the source name is
not copied, interpreted, or emitted. Identifiers remain excluded because a
country name cannot preserve their distinct semantic contract.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.country_registry import CountryRegistry
from document_ocr.synthesis.generators import DeterministicStream

CargoOriginRelation = Literal["same_country", "different_country"]
CargoOriginExclusionReason = Literal[
    "identifier_present",
    "missing_origin_name",
    "unresolved_origin_name",
    "unresolved_shipper_country",
]


class CargoOriginScenarioError(RuntimeError):
    """A cargo-origin source surface cannot meet the restricted contract."""


@dataclass(frozen=True, slots=True)
class WeightedCargoOriginValue:
    """A positive empirical weight with deterministic ordering semantics."""

    value: str
    count: int

    def __post_init__(self) -> None:
        if not self.value or self.count <= 0:
            raise ValueError("cargo-origin weighted values require content and positive count")


@dataclass(frozen=True, slots=True)
class CargoOriginExclusion:
    """One source group unavailable to fitting under this contract.

    ``identifier_present`` and ``missing_origin_name`` are non-generatable
    source forms. ``unresolved_origin_name`` and
    ``unresolved_shipper_country`` are excluded only from fitting the relation;
    their name-only structure remains generatable. ``origin`` is retained
    verbatim for a reviewer-facing audit and never copied into generated output.
    """

    document_id: str
    group_index: int
    reason: CargoOriginExclusionReason
    origin: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.document_id or self.group_index < 0:
            raise ValueError("cargo-origin exclusion identity is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "documentId": self.document_id,
            "groupIndex": self.group_index,
            "reason": self.reason,
            "origin": dict(self.origin),
        }


@dataclass(frozen=True, slots=True)
class CargoOriginFitAudit:
    """Balanced provenance for the isolated empirical fitting scope."""

    fit_documents: int
    fit_scope_sha256: str
    cargo_groups: int
    origin_present_groups: int
    origin_absent_groups: int
    generatable_country_name_groups: int
    fit_eligible_groups: int
    same_country_groups: int
    different_country_groups: int
    excluded_identifier_present_groups: int
    excluded_missing_origin_name_groups: int
    excluded_unresolved_origin_name_groups: int
    excluded_unresolved_shipper_country_groups: int

    def __post_init__(self) -> None:
        if self.fit_documents <= 0 or len(self.fit_scope_sha256) != 64:
            raise ValueError("cargo-origin fit scope is invalid")
        if (
            min(
                self.cargo_groups,
                self.origin_present_groups,
                self.origin_absent_groups,
                self.generatable_country_name_groups,
                self.fit_eligible_groups,
                self.same_country_groups,
                self.different_country_groups,
                self.excluded_identifier_present_groups,
                self.excluded_missing_origin_name_groups,
                self.excluded_unresolved_origin_name_groups,
                self.excluded_unresolved_shipper_country_groups,
            )
            < 0
        ):
            raise ValueError("cargo-origin audit counts cannot be negative")
        if self.origin_present_groups + self.origin_absent_groups != self.cargo_groups:
            raise ValueError("cargo-origin presence audit does not balance")
        if (
            self.generatable_country_name_groups
            + self.excluded_identifier_present_groups
            + self.excluded_missing_origin_name_groups
            != self.origin_present_groups
        ):
            raise ValueError("cargo-origin generatability audit does not balance")
        if (
            self.fit_eligible_groups
            + self.excluded_unresolved_origin_name_groups
            + self.excluded_unresolved_shipper_country_groups
            != self.generatable_country_name_groups
        ):
            raise ValueError("cargo-origin fit eligibility audit does not balance")
        if self.same_country_groups + self.different_country_groups != self.fit_eligible_groups:
            raise ValueError("cargo-origin relation audit does not balance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitDocuments": self.fit_documents,
            "fitScopeSha256": self.fit_scope_sha256,
            "cargoGroups": self.cargo_groups,
            "originPresentGroups": self.origin_present_groups,
            "originAbsentGroups": self.origin_absent_groups,
            "generatableCountryNameGroups": self.generatable_country_name_groups,
            "fitEligibleGroups": self.fit_eligible_groups,
            "sameCountryGroups": self.same_country_groups,
            "differentCountryGroups": self.different_country_groups,
            "excludedIdentifierPresentGroups": self.excluded_identifier_present_groups,
            "excludedMissingOriginNameGroups": self.excluded_missing_origin_name_groups,
            "excludedUnresolvedOriginNameGroups": self.excluded_unresolved_origin_name_groups,
            "excludedUnresolvedShipperCountryGroups": (
                self.excluded_unresolved_shipper_country_groups
            ),
            "countryResolution": "alias_free_iso3166_fit_only_name_projection_v2",
            "relationMethod": "train_isolated_goods_origin_to_shipper_country_v1",
            "differentCountryMethod": ("train_isolated_different_goods_origin_country_count_v1"),
        }


@dataclass(frozen=True, slots=True)
class CargoOriginSupport:
    """The complete isolated fit needed for deterministic sampling."""

    relation_weights: tuple[WeightedCargoOriginValue, ...]
    different_country_weights: tuple[WeightedCargoOriginValue, ...]
    exclusions: tuple[CargoOriginExclusion, ...]
    audit: CargoOriginFitAudit

    def __post_init__(self) -> None:
        relations = tuple(row.value for row in self.relation_weights)
        if relations != tuple(sorted(set(relations))) or set(relations) != {
            "same_country",
            "different_country",
        }:
            raise ValueError("cargo-origin relation support must contain both ordered relations")
        countries = tuple(row.value for row in self.different_country_weights)
        if countries != tuple(sorted(set(countries))):
            raise ValueError("cargo-origin different-country support must be unique and ordered")
        if not countries:
            raise ValueError("cargo-origin support has no observed different-country values")
        if len(self.exclusions) != (
            self.audit.excluded_identifier_present_groups
            + self.audit.excluded_missing_origin_name_groups
            + self.audit.excluded_unresolved_origin_name_groups
            + self.audit.excluded_unresolved_shipper_country_groups
        ):
            raise ValueError("cargo-origin exclusion detail differs from audit totals")


@dataclass(frozen=True, slots=True)
class CargoOriginScenario:
    """One generated country-form goods origin for one source cargo group."""

    group_index: int
    relation_to_commercial_origin: CargoOriginRelation
    country_code: str
    country_name: str

    def __post_init__(self) -> None:
        if self.group_index < 0 or len(self.country_code) != 2 or not self.country_name:
            raise ValueError("cargo-origin scenario is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "groupIndex": self.group_index,
            "relationToCommercialOrigin": self.relation_to_commercial_origin,
            "countryCode": self.country_code,
            "countryName": self.country_name,
            "nameProvider": "pinned_iso3166_printable_country_v1",
        }


@dataclass(frozen=True, slots=True)
class CargoOriginProjection:
    """An all-or-nothing replacement for a template's origin-present groups."""

    origins: tuple[CargoOriginScenario, ...]

    def __post_init__(self) -> None:
        indices = tuple(row.group_index for row in self.origins)
        if indices != tuple(sorted(set(indices))):
            raise ValueError("cargo-origin projection group indices must be unique and ordered")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "name_only_origin_to_country_projection_v2",
            "origins": [row.to_dict() for row in self.origins],
        }


def _require_alias_free_registry(country_registry: CountryRegistry) -> None:
    if country_registry.audit.observed_alias_records != 0:
        raise ValueError("cargo-origin sampling requires an alias-free ISO country registry")


def _patch(target: Mapping[str, Any]) -> Mapping[str, Any]:
    patch = target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("cargo-origin source target has no documentPatch object")
    return patch


def _groups(patch: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    raw = patch.get("cargoGroups")
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(not isinstance(row, Mapping) for row in raw):
        raise ValueError("cargo-origin source cargoGroups must be an array of objects")
    return cast(Sequence[Mapping[str, Any]], raw)


def _origin(group: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw = group.get("origin")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("cargo-origin source origin must be an object")
    return cast(Mapping[str, Any], raw)


def _country(value: Any, country_registry: CountryRegistry) -> str | None:
    return country_registry.resolve(value) if isinstance(value, str) else None


def _generation_exclusion(origin: Mapping[str, Any]) -> CargoOriginExclusionReason | None:
    if "identifier" in origin:
        return "identifier_present"
    name = origin.get("name")
    if not isinstance(name, str) or not name:
        return "missing_origin_name"
    return None


def _weighted_rows(
    counter: Mapping[str, int], *, label: str
) -> tuple[WeightedCargoOriginValue, ...]:
    rows = tuple(
        WeightedCargoOriginValue(value=value, count=count)
        for value, count in sorted(counter.items())
        if count > 0
    )
    if not rows:
        raise ValueError(f"cargo-origin {label} support cannot be empty")
    return rows


def _weighted_choice(
    rows: Sequence[WeightedCargoOriginValue], *, stream: DeterministicStream
) -> str:
    total = sum(row.count for row in rows)
    draw = stream.randbelow(total)
    offset = 0
    for row in rows:
        offset += row.count
        if draw < offset:
            return row.value
    raise RuntimeError("cargo-origin weighted selection did not resolve")


def build_cargo_origin_support(
    *,
    source_targets: Mapping[str, Mapping[str, Any]],
    fit_document_ids: Sequence[str],
    country_registry: CountryRegistry,
) -> CargoOriginSupport:
    """Fit only explicit ISO country-name origin/shipper relationships.

    The supplied document IDs are the complete fit scope.  Non-generatable
    source forms are recorded with their exact source payload, not normalized
    or retained as generated output.
    """

    _require_alias_free_registry(country_registry)
    if not fit_document_ids or len(fit_document_ids) != len(set(fit_document_ids)):
        raise ValueError("cargo-origin fit_document_ids must be non-empty and unique")
    missing = sorted(set(fit_document_ids) - set(source_targets))
    if missing:
        raise ValueError(f"cargo-origin fit documents are absent from source targets: {missing}")

    origin_present = origin_absent = cargo_groups = generatable = eligible = 0
    exclusions_by_reason = Counter[str]()
    relation = Counter[str]()
    different_countries = Counter[str]()
    exclusions: list[CargoOriginExclusion] = []

    for document_id in sorted(fit_document_ids):
        patch = _patch(source_targets[document_id])
        shipper = patch.get("parties")
        if shipper is not None and not isinstance(shipper, Mapping):
            raise ValueError("cargo-origin source parties must be an object when present")
        shipper_value = (
            cast(Mapping[str, Any], shipper).get("shipper")
            if isinstance(shipper, Mapping)
            else None
        )
        if shipper_value is not None and not isinstance(shipper_value, Mapping):
            raise ValueError("cargo-origin source shipper must be an object when present")
        shipper_country = _country(
            cast(Mapping[str, Any], shipper_value).get("country")
            if isinstance(shipper_value, Mapping)
            else None,
            country_registry,
        )
        for group_index, group in enumerate(_groups(patch)):
            cargo_groups += 1
            origin = _origin(group)
            if origin is None:
                origin_absent += 1
                continue
            origin_present += 1
            reason = _generation_exclusion(origin)
            if reason is not None:
                exclusions_by_reason[reason] += 1
                exclusions.append(
                    CargoOriginExclusion(
                        document_id=document_id,
                        group_index=group_index,
                        reason=reason,
                        origin=dict(origin),
                    )
                )
                continue
            generatable += 1
            origin_country = _country(origin["name"], country_registry)
            if origin_country is None:
                exclusions_by_reason["unresolved_origin_name"] += 1
                exclusions.append(
                    CargoOriginExclusion(
                        document_id=document_id,
                        group_index=group_index,
                        reason="unresolved_origin_name",
                        origin=dict(origin),
                    )
                )
                continue
            if shipper_country is None:
                exclusions_by_reason["unresolved_shipper_country"] += 1
                exclusions.append(
                    CargoOriginExclusion(
                        document_id=document_id,
                        group_index=group_index,
                        reason="unresolved_shipper_country",
                        origin=dict(origin),
                    )
                )
                continue
            eligible += 1
            relation_value: CargoOriginRelation = (
                "same_country" if origin_country == shipper_country else "different_country"
            )
            relation[relation_value] += 1
            if relation_value == "different_country":
                different_countries[origin_country] += 1

    if not eligible:
        raise CargoOriginScenarioError("no cargo origins meet the restricted fitting contract")
    # A relation sampler must model both outcomes rather than conceal an absent
    # different-origin mode as an implicit identity fallback.
    if not relation["same_country"] or not relation["different_country"]:
        raise CargoOriginScenarioError(
            "cargo-origin fitting requires observed same_country and different_country support"
        )
    audit = CargoOriginFitAudit(
        fit_documents=len(fit_document_ids),
        fit_scope_sha256=sha256_bytes(canonical_json_bytes(sorted(fit_document_ids))),
        cargo_groups=cargo_groups,
        origin_present_groups=origin_present,
        origin_absent_groups=origin_absent,
        generatable_country_name_groups=generatable,
        fit_eligible_groups=eligible,
        same_country_groups=relation["same_country"],
        different_country_groups=relation["different_country"],
        excluded_identifier_present_groups=exclusions_by_reason["identifier_present"],
        excluded_missing_origin_name_groups=exclusions_by_reason["missing_origin_name"],
        excluded_unresolved_origin_name_groups=exclusions_by_reason["unresolved_origin_name"],
        excluded_unresolved_shipper_country_groups=(
            exclusions_by_reason["unresolved_shipper_country"]
        ),
    )
    return CargoOriginSupport(
        relation_weights=_weighted_rows(relation, label="relation"),
        different_country_weights=_weighted_rows(different_countries, label="different country"),
        exclusions=tuple(
            sorted(exclusions, key=lambda row: (row.document_id, row.group_index, row.reason))
        ),
        audit=audit,
    )


def sample_cargo_origin_projection(
    *,
    source_target: Mapping[str, Any],
    support: CargoOriginSupport,
    country_registry: CountryRegistry,
    commercial_origin_country_code: str,
    stream: DeterministicStream,
) -> CargoOriginProjection:
    """Generate one country-name origin per source-present name-only group.

    The source name need not itself be a country. It is never copied or
    interpreted; only its name-only structural contract is preserved. A source
    identifier or missing name raises instead of being silently coerced.
    """

    _require_alias_free_registry(country_registry)
    country_registry.entry(commercial_origin_country_code)
    generated: list[CargoOriginScenario] = []
    for group_index, group in enumerate(_groups(_patch(source_target))):
        origin = _origin(group)
        if origin is None:
            continue
        reason = _generation_exclusion(origin)
        if reason is not None:
            raise CargoOriginScenarioError(
                f"cargoGroups[{group_index}].origin is non-generatable: {reason}"
            )
        relation = cast(
            CargoOriginRelation,
            _weighted_choice(
                support.relation_weights,
                stream=stream.derive(f"cargo-group-{group_index}-relation"),
            ),
        )
        if relation == "same_country":
            country_code = commercial_origin_country_code
        else:
            choices = tuple(
                row
                for row in support.different_country_weights
                if row.value != commercial_origin_country_code
            )
            if not choices:
                raise CargoOriginScenarioError(
                    "different_country has no observed fitted country distinct from "
                    "commercial origin"
                )
            country_code = _weighted_choice(
                choices,
                stream=stream.derive(f"cargo-group-{group_index}-different-country"),
            )
            if country_code == commercial_origin_country_code:
                raise RuntimeError(
                    "different_country selection did not differ from commercial origin"
                )
        generated.append(
            CargoOriginScenario(
                group_index=group_index,
                relation_to_commercial_origin=relation,
                country_code=country_code,
                country_name=country_registry.printable_name(country_code),
            )
        )
    return CargoOriginProjection(origins=tuple(generated))


def project_cargo_origin_target(
    *, source_target: Mapping[str, Any], projection: CargoOriginProjection
) -> dict[str, Any]:
    """Apply a complete projection while preserving group origin presence/cardinality."""

    target = cast(dict[str, Any], deepcopy(source_target))
    patch = cast(dict[str, Any], target["documentPatch"])
    groups = cast(list[dict[str, Any]], patch.get("cargoGroups") or [])
    source_indices = tuple(
        index for index, group in enumerate(groups) if group.get("origin") is not None
    )
    projected_indices = tuple(row.group_index for row in projection.origins)
    if source_indices != projected_indices:
        raise CargoOriginScenarioError(
            "cargo-origin projection must cover exactly the source origin-present groups"
        )
    for row in projection.origins:
        groups[row.group_index]["origin"] = {"name": row.country_name}
    return target
