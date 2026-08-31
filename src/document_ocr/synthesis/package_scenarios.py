"""Fit-isolated package-category scenarios with explicit hierarchy preservation.

Package category identity and printed evidence are separate concerns.  This
module combines role-conditioned empirical sampling with an explicitly bounded
registry-exploration arm. Exploration is limited to the same authoritative
display-name family and to the task's frozen vocabulary; it never guesses a
cross-family package meaning. The corresponding printed surface remains
pending for the later text-realizer. Fallback and quantity-only facts are never
reclassified.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Literal

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.modeling_views import (
    CargoPackageNumericObservation,
    PackageHierarchyRecord,
    PackageRole,
)
from document_ocr.synthesis.package_registry import LoadedPackageRegistry

PackageDisposition = Literal["task_facing", "metadata_only"]
PackageResolution = Literal[
    "sampled_category",
    "preserved_untyped",
    "preserved_metadata",
]
PackageSamplingComponent = Literal[
    "fit_empirical",
    "registry_same_family_exploration",
    "not_applicable",
]
PrintedSurfaceStatus = Literal[
    "pending_text_realization",
    "preserved_source_surface",
    "absent",
]
PackageExclusionReason = Literal[
    "generic_category_conflicts_with_direct_multi_package_role",
    "resolved_category_has_unknown_role",
    "generic_role_has_non_generic_category",
]

_GENERIC_PACKAGE_CATEGORY = "PACKAGE_PACKAGE"


@dataclass(frozen=True, slots=True)
class PackageFitObservation:
    """One fit-scoped package fact plus its non-modeled hierarchy identity."""

    source_document_id: str
    group_id: str
    source_package_id: str
    projected_package_id: str | None
    source_package_position: int
    source_package_level_count: int
    metadata_only_package_count: int
    role: PackageRole
    disposition: PackageDisposition
    quantity: int | None
    category_token: str | None
    printed_surface: str | None

    def __post_init__(self) -> None:
        identifiers = (self.source_document_id, self.group_id, self.source_package_id)
        if any(not value for value in identifiers):
            raise ValueError("package fit identities must be non-empty")
        if self.source_package_position < 0 or self.source_package_level_count <= 0:
            raise ValueError("package hierarchy positions and counts are invalid")
        if self.source_package_position >= self.source_package_level_count:
            raise ValueError("package position exceeds the source package-level count")
        if not 0 <= self.metadata_only_package_count <= self.source_package_level_count:
            raise ValueError("metadata-only package count is outside the source hierarchy")
        if (self.disposition == "task_facing") != (self.projected_package_id is not None):
            raise ValueError("only task-facing package facts have a projected package ID")
        if self.quantity is not None and self.quantity < 0:
            raise ValueError("package quantity must be non-negative")
        if self.category_token is not None and not self.printed_surface:
            raise ValueError("a resolved package category requires reviewed printed evidence")
        if self.disposition == "metadata_only" and self.category_token is not None:
            raise ValueError("metadata-only package facts are preserved, not reclassified")

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.source_document_id, self.group_id, self.source_package_id)


@dataclass(frozen=True, slots=True)
class PackageCategorySupport:
    category_token: str
    application_code: str
    semantic_family: str
    occurrences: int
    document_count: int


@dataclass(frozen=True, slots=True)
class PackageRegistryCategory:
    category_token: str
    application_code: str
    semantic_family: str


@dataclass(frozen=True, slots=True)
class PackageRoleSupport:
    role: PackageRole
    categories: tuple[PackageCategorySupport, ...]


@dataclass(frozen=True, slots=True)
class PackageSupportExclusion:
    source_document_id: str
    group_id: str
    source_package_id: str
    role: PackageRole
    category_token: str
    printed_surface: str
    reason: PackageExclusionReason

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.source_document_id, self.group_id, self.source_package_id)


@dataclass(frozen=True, slots=True)
class PackageScenarioAudit:
    fit_document_count: int
    input_package_facts: int
    task_facing_facts: int
    metadata_only_facts: int
    resolved_facts: int
    fallback_facts: int
    untyped_facts: int
    excluded_resolved_facts: int
    registry_category_count: int
    allowed_category_count: int
    observed_category_count: int
    registry_exploration_candidate_count: int


@dataclass(frozen=True, slots=True)
class PackageScenarioSupport:
    registry_sha256: str
    fit_document_ids: tuple[str, ...]
    allowed_category_tokens: tuple[str, ...]
    registry_categories: tuple[PackageRegistryCategory, ...]
    roles: tuple[PackageRoleSupport, ...]
    exclusions: tuple[PackageSupportExclusion, ...]
    audit: PackageScenarioAudit
    _fit_document_id_set: frozenset[str] = field(init=False, repr=False)
    _excluded_identity_set: frozenset[tuple[str, str, str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        fit_set = frozenset(self.fit_document_ids)
        if not fit_set or len(fit_set) != len(self.fit_document_ids):
            raise ValueError("package support fit document IDs must be non-empty and unique")
        excluded_set = frozenset(row.identity for row in self.exclusions)
        if len(excluded_set) != len(self.exclusions):
            raise ValueError("package support exclusions must have unique identities")
        object.__setattr__(self, "_fit_document_id_set", fit_set)
        object.__setattr__(self, "_excluded_identity_set", excluded_set)

    def role_support(self, role: PackageRole) -> PackageRoleSupport:
        for row in self.roles:
            if row.role == role:
                return row
        raise ValueError(
            f"fit partition has no resolved package-category support for role {role!r}"
        )

    @property
    def excluded_identities(self) -> frozenset[tuple[str, str, str]]:
        return self._excluded_identity_set

    def includes_fit_document(self, document_id: str) -> bool:
        return document_id in self._fit_document_id_set

    def registry_family(self, category_token: str) -> tuple[PackageRegistryCategory, ...]:
        source = next(
            (row for row in self.registry_categories if row.category_token == category_token),
            None,
        )
        if source is None:
            raise ValueError(
                f"package category is outside the frozen task vocabulary: {category_token!r}"
            )
        return tuple(
            row for row in self.registry_categories if row.semantic_family == source.semantic_family
        )


@dataclass(frozen=True, slots=True)
class PackageTypeScenario:
    source_document_id: str
    group_id: str
    source_package_id: str
    projected_package_id: str | None
    source_package_position: int
    source_package_level_count: int
    metadata_only_package_count: int
    role: PackageRole
    disposition: PackageDisposition
    quantity: int | None
    resolution: PackageResolution
    sampling_component: PackageSamplingComponent
    category_token: str | None
    application_code: str | None
    type_description: str | None
    source_printed_surface: str | None
    printed_surface_status: PrintedSurfaceStatus


def _projected_package_id(observation: CargoPackageNumericObservation) -> str:
    document_id = observation.projection.source_document_id
    prefix = f"cargo-package:{document_id}:package:"
    key = observation.projection.view_row_key
    if not key.startswith(prefix):
        raise ValueError(f"cargo-package view row has an unsupported identity: {key!r}")
    package_id = key.removeprefix(prefix)
    if not package_id or ":" in package_id:
        raise ValueError(f"cargo-package view row has an invalid package ID: {key!r}")
    return package_id


def package_fit_observations(
    *,
    fit_document_ids: Sequence[str],
    hierarchy: Sequence[PackageHierarchyRecord],
    packages: Sequence[CargoPackageNumericObservation],
) -> tuple[PackageFitObservation, ...]:
    """Join train-only model rows to the lossless hierarchy ledger."""

    fit_ids = tuple(fit_document_ids)
    if not fit_ids or len(fit_ids) != len(set(fit_ids)):
        raise ValueError("fit document IDs must be non-empty and unique")
    fit_set = frozenset(fit_ids)
    package_rows: dict[tuple[str, str], CargoPackageNumericObservation] = {}
    for row in packages:
        document_id = row.projection.source_document_id
        if document_id not in fit_set:
            raise ValueError("cargo-package observation lies outside the fit partition")
        key = (document_id, _projected_package_id(row))
        if key in package_rows:
            raise ValueError(f"duplicate cargo-package observation: {key!r}")
        package_rows[key] = row

    grouped_hierarchy: dict[tuple[str, str], list[PackageHierarchyRecord]] = defaultdict(list)
    hierarchy_identities: set[tuple[str, str, str]] = set()
    for hierarchy_row in hierarchy:
        if hierarchy_row.source_document_id not in fit_set:
            raise ValueError("package hierarchy row lies outside the fit partition")
        identity = (
            hierarchy_row.source_document_id,
            hierarchy_row.group_id,
            hierarchy_row.source_package_id,
        )
        if identity in hierarchy_identities:
            raise ValueError(f"duplicate package hierarchy identity: {identity!r}")
        hierarchy_identities.add(identity)
        grouped_hierarchy[(hierarchy_row.source_document_id, hierarchy_row.group_id)].append(
            hierarchy_row
        )

    output: list[PackageFitObservation] = []
    used_package_rows: set[tuple[str, str]] = set()
    for group_key in sorted(grouped_hierarchy):
        group = sorted(grouped_hierarchy[group_key], key=lambda row: row.source_package_position)
        if tuple(row.source_package_position for row in group) != tuple(range(len(group))):
            raise ValueError(f"source package positions are not contiguous: {group_key!r}")
        metadata_count = sum(row.disposition == "metadata_only" for row in group)
        for hierarchy_row in group:
            category: str | None = None
            surface = hierarchy_row.type_description
            if hierarchy_row.disposition == "task_facing":
                assert hierarchy_row.projected_package_id is not None
                package_key = (
                    hierarchy_row.source_document_id,
                    hierarchy_row.projected_package_id,
                )
                observation = package_rows.get(package_key)
                if observation is None:
                    raise ValueError(
                        f"task-facing hierarchy row has no package view: {package_key!r}"
                    )
                used_package_rows.add(package_key)
                features = observation.features
                if features.package_role != hierarchy_row.role:
                    raise ValueError("package role differs between hierarchy and package view")
                if features.source_package_position != hierarchy_row.source_package_position:
                    raise ValueError("package position differs between hierarchy and package view")
                if features.source_package_level_count != len(group):
                    raise ValueError(
                        "package-level count differs between hierarchy and package view"
                    )
                if features.metadata_only_package_count != metadata_count:
                    raise ValueError(
                        "metadata-only count differs between hierarchy and package view"
                    )
                category = features.type_category
                surface = features.printed_surface
            output.append(
                PackageFitObservation(
                    source_document_id=hierarchy_row.source_document_id,
                    group_id=hierarchy_row.group_id,
                    source_package_id=hierarchy_row.source_package_id,
                    projected_package_id=hierarchy_row.projected_package_id,
                    source_package_position=hierarchy_row.source_package_position,
                    source_package_level_count=len(group),
                    metadata_only_package_count=metadata_count,
                    role=hierarchy_row.role,
                    disposition=hierarchy_row.disposition,
                    quantity=hierarchy_row.quantity,
                    category_token=category,
                    printed_surface=surface,
                )
            )
    extra = sorted(set(package_rows) - used_package_rows)
    if extra:
        raise ValueError(f"cargo-package views have no hierarchy rows: {extra[:5]!r}")
    return tuple(output)


def _exclusion_reason(
    observation: PackageFitObservation,
    group: Sequence[PackageFitObservation],
) -> PackageExclusionReason | None:
    if observation.category_token is None or observation.disposition != "task_facing":
        return None
    if observation.role == "unknown":
        return "resolved_category_has_unknown_role"
    if observation.role == "generic_aggregate" and (
        observation.category_token != _GENERIC_PACKAGE_CATEGORY
    ):
        return "generic_role_has_non_generic_category"
    if (
        observation.role == "direct_goods"
        and observation.category_token == _GENERIC_PACKAGE_CATEGORY
        and any(
            row.disposition == "task_facing"
            and row.category_token not in {None, _GENERIC_PACKAGE_CATEGORY}
            for row in group
        )
    ):
        return "generic_category_conflicts_with_direct_multi_package_role"
    return None


def build_package_scenario_support(
    *,
    observations: Sequence[PackageFitObservation],
    fit_document_ids: Sequence[str],
    allowed_category_tokens: Collection[str],
    registry: LoadedPackageRegistry,
    expected_registry_sha256: str,
) -> PackageScenarioSupport:
    """Build role-conditioned category support exclusively from fit observations."""

    fit_ids = tuple(fit_document_ids)
    if not fit_ids or len(fit_ids) != len(set(fit_ids)):
        raise ValueError("fit document IDs must be non-empty and unique")
    if registry.sha256 != expected_registry_sha256:
        raise ValueError("package registry differs from the fit-scope receipt")
    allowed = tuple(sorted(allowed_category_tokens))
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError("allowed package categories must be non-empty and unique")
    registry_tokens = registry.category_tokens
    unknown_allowed = sorted(set(allowed) - registry_tokens)
    if unknown_allowed:
        raise ValueError(
            f"allowed package categories are absent from registry: {unknown_allowed!r}"
        )

    allowed_set = frozenset(allowed)
    registry_categories = tuple(
        sorted(
            (
                PackageRegistryCategory(
                    category_token=entry.categoryToken,
                    application_code=entry.applicationCode,
                    semantic_family=_semantic_family(entry.displayName),
                )
                for entry in registry.payload.entries
                if entry.categoryToken in allowed_set
            ),
            key=lambda row: row.category_token,
        )
    )
    if len(registry_categories) != len(allowed):
        raise RuntimeError("package task vocabulary did not resolve one-to-one to the registry")

    fit_set = frozenset(fit_ids)
    identities: set[tuple[str, str, str]] = set()
    grouped: dict[tuple[str, str], list[PackageFitObservation]] = defaultdict(list)
    for row in observations:
        if row.source_document_id not in fit_set:
            raise ValueError("package support observation lies outside the fit partition")
        if row.identity in identities:
            raise ValueError(f"duplicate package support observation: {row.identity!r}")
        identities.add(row.identity)
        grouped[(row.source_document_id, row.group_id)].append(row)

    exclusions: list[PackageSupportExclusion] = []
    excluded: set[tuple[str, str, str]] = set()
    for group_key in sorted(grouped):
        group = sorted(grouped[group_key], key=lambda row: row.source_package_position)
        for row in group:
            reason = _exclusion_reason(row, group)
            if reason is None:
                continue
            assert row.category_token is not None
            assert row.printed_surface is not None
            excluded.add(row.identity)
            exclusions.append(
                PackageSupportExclusion(
                    source_document_id=row.source_document_id,
                    group_id=row.group_id,
                    source_package_id=row.source_package_id,
                    role=row.role,
                    category_token=row.category_token,
                    printed_surface=row.printed_surface,
                    reason=reason,
                )
            )

    counts: Counter[tuple[PackageRole, str]] = Counter()
    documents: dict[tuple[PackageRole, str], set[str]] = defaultdict(set)
    resolved = fallback = untyped = metadata_only = task_facing = 0
    for row in observations:
        if row.disposition == "metadata_only":
            metadata_only += 1
            continue
        task_facing += 1
        if row.category_token is None:
            if row.printed_surface is None:
                untyped += 1
            else:
                fallback += 1
            continue
        resolved += 1
        if row.category_token not in allowed:
            raise ValueError(
                f"fit-observed package category is outside frozen task vocabulary: "
                f"{row.category_token!r}"
            )
        registry.entry(row.category_token)
        if row.identity in excluded:
            continue
        key = (row.role, row.category_token)
        counts[key] += 1
        documents[key].add(row.source_document_id)

    role_rows: list[PackageRoleSupport] = []
    for role in ("direct_goods", "generic_aggregate", "outer_transport", "unknown"):
        category_rows = []
        for candidate_role, token in sorted(counts):
            if candidate_role != role:
                continue
            entry = registry.entry(token)
            category_rows.append(
                PackageCategorySupport(
                    category_token=token,
                    application_code=entry.applicationCode,
                    semantic_family=_semantic_family(entry.displayName),
                    occurrences=counts[(role, token)],
                    document_count=len(documents[(role, token)]),
                )
            )
        if category_rows:
            role_rows.append(PackageRoleSupport(role=role, categories=tuple(category_rows)))

    return PackageScenarioSupport(
        registry_sha256=registry.sha256,
        fit_document_ids=fit_ids,
        allowed_category_tokens=allowed,
        registry_categories=registry_categories,
        roles=tuple(role_rows),
        exclusions=tuple(exclusions),
        audit=PackageScenarioAudit(
            fit_document_count=len(fit_ids),
            input_package_facts=len(observations),
            task_facing_facts=task_facing,
            metadata_only_facts=metadata_only,
            resolved_facts=resolved,
            fallback_facts=fallback,
            untyped_facts=untyped,
            excluded_resolved_facts=len(exclusions),
            registry_category_count=len(registry.payload.entries),
            allowed_category_count=len(allowed),
            observed_category_count=len({token for _, token in counts}),
            registry_exploration_candidate_count=sum(
                any(
                    candidate.semantic_family == row.semantic_family
                    and candidate.category_token != row.category_token
                    for candidate in registry_categories
                )
                for row in registry_categories
            ),
        ),
    )


def sample_package_type(
    observation: PackageFitObservation,
    *,
    support: PackageScenarioSupport,
    registry_exploration_permyriad: int,
    stream: DeterministicStream,
) -> PackageTypeScenario:
    """Sample one identity while preserving every package hierarchy coordinate."""

    if not support.includes_fit_document(observation.source_document_id):
        raise ValueError("package sampling source lies outside the fit partition")
    if not 0 <= registry_exploration_permyriad <= 10_000:
        raise ValueError("package registry exploration weight must be within [0, 10000]")
    if observation.disposition == "metadata_only":
        return _package_scenario(
            observation,
            resolution="preserved_metadata",
            sampling_component="not_applicable",
            category_token=None,
            application_code=None,
            type_description=observation.printed_surface,
            printed_surface_status=(
                "preserved_source_surface" if observation.printed_surface is not None else "absent"
            ),
        )
    if observation.category_token is None and observation.printed_surface is None:
        return _package_scenario(
            observation,
            resolution="preserved_untyped",
            sampling_component="not_applicable",
            category_token=None,
            application_code=None,
            type_description=None,
            printed_surface_status="absent",
        )
    if observation.identity in support.excluded_identities:
        raise ValueError("package fact is excluded by the audited role/category conflict policy")
    family_candidates = (
        tuple(
            row
            for row in support.registry_family(observation.category_token)
            if row.category_token != observation.category_token
        )
        if observation.category_token is not None
        else ()
    )
    explore = bool(family_candidates) and (
        stream.derive("component").randbelow(10_000) < registry_exploration_permyriad
    )
    if explore:
        registry_row = family_candidates[
            stream.derive("registry-family").randbelow(len(family_candidates))
        ]
        selected_token = registry_row.category_token
        selected_code = registry_row.application_code
        component: PackageSamplingComponent = "registry_same_family_exploration"
    else:
        selected = _sample_role_category(
            support.role_support(observation.role),
            stream=stream.derive("category"),
        )
        selected_token = selected.category_token
        selected_code = selected.application_code
        component = "fit_empirical"
    return _package_scenario(
        observation,
        resolution="sampled_category",
        sampling_component=component,
        category_token=selected_token,
        application_code=selected_code,
        type_description=None,
        printed_surface_status="pending_text_realization",
    )


def _sample_role_category(
    role_support: PackageRoleSupport, *, stream: DeterministicStream
) -> PackageCategorySupport:
    total = sum(row.occurrences for row in role_support.categories)
    draw = stream.randbelow(total)
    cumulative = 0
    for row in role_support.categories:
        cumulative += row.occurrences
        if draw < cumulative:
            return row
    raise RuntimeError("package-category cumulative support is inconsistent")


def _package_scenario(
    observation: PackageFitObservation,
    *,
    resolution: PackageResolution,
    sampling_component: PackageSamplingComponent,
    category_token: str | None,
    application_code: str | None,
    type_description: str | None,
    printed_surface_status: PrintedSurfaceStatus,
) -> PackageTypeScenario:
    return PackageTypeScenario(
        source_document_id=observation.source_document_id,
        group_id=observation.group_id,
        source_package_id=observation.source_package_id,
        projected_package_id=observation.projected_package_id,
        source_package_position=observation.source_package_position,
        source_package_level_count=observation.source_package_level_count,
        metadata_only_package_count=observation.metadata_only_package_count,
        role=observation.role,
        disposition=observation.disposition,
        quantity=observation.quantity,
        resolution=resolution,
        sampling_component=sampling_component,
        category_token=category_token,
        application_code=application_code,
        type_description=type_description,
        source_printed_surface=observation.printed_surface,
        printed_surface_status=printed_surface_status,
    )


def _semantic_family(display_name: str) -> str:
    """Derive a registry-owned noun family without maintaining an alias map."""

    normalized = unicodedata.normalize("NFKD", display_name).encode("ascii", "ignore").decode()
    head = normalized.partition(",")[0]
    words = tuple(re.findall(r"[A-Za-z0-9]+", head.casefold()))
    if not words:
        raise ValueError(f"package display name has no semantic family: {display_name!r}")
    return "_".join(words)
