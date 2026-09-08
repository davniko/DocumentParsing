"""Fit-isolated joint goods/package support for synthetic B/L cargo.

Package categories are not independent cargo decorations.  This module learns
whole task-facing package signatures from the training partition and couples
them to HS headings, explicit thermal operation, or dangerous-goods hazard
families.  A generated cargo identity is therefore selected together with a
source-observed package signature instead of being combined with an unrelated
role-marginal draw.

The HS heading is the narrowest stable international semantic unit used here:
the first four digits identify one Harmonized System heading, while national
digits beyond the six-digit subheading are deliberately irrelevant.  No text
regex, hand-written goods alias, or package compatibility table is used.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.fit_partition import fit_document_ids
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.thermal_goods import (
    AmbientGoodsIdentity,
    ThermalGoodsIdentity,
    ThermalGoodsSupport,
    ThermalProfile,
)

CompatibilityBasis = Literal[
    "fit_hs_heading_joint",
    "fit_hs_signature_conditioned_heading_pool",
    "fit_thermal_profile_joint",
    "fit_dangerous_goods_hazard_joint",
]
PackageSignature = tuple[str, ...]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


@dataclass(frozen=True, slots=True)
class SignatureSupportRow:
    """One complete ordered package signature and its fit-only support."""

    key: str
    package_count: int
    categories: PackageSignature
    occurrences: int
    document_count: int

    def __post_init__(self) -> None:
        if not self.key or self.package_count <= 0:
            raise ValueError("package signature support key and count must be positive")
        if len(self.categories) != self.package_count or not all(self.categories):
            raise ValueError("package signature does not match its declared cardinality")
        if self.occurrences <= 0 or not 0 < self.document_count <= self.occurrences:
            raise ValueError("package signature support counts are invalid")


@dataclass(frozen=True, slots=True)
class PackageSignaturePoolSupportRow:
    """Fit-observed package signature with its conditional HS-heading distribution."""

    heading_occurrences: tuple[tuple[str, int], ...]
    package_count: int
    categories: PackageSignature
    occurrences: int
    document_count: int

    def __post_init__(self) -> None:
        if not self.heading_occurrences or any(
            len(heading) != 4 or not heading.isdigit() or occurrences <= 0
            for heading, occurrences in self.heading_occurrences
        ):
            raise ValueError("package signature pool has invalid HS-heading support")
        if tuple(sorted(self.heading_occurrences)) != self.heading_occurrences:
            raise ValueError("package signature pool headings must be unique and sorted")
        if self.package_count <= 0 or len(self.categories) != self.package_count:
            raise ValueError("package signature pool has an invalid cardinality")
        if not all(self.categories):
            raise ValueError("package signature pool contains an empty category")
        if self.occurrences <= 0 or not 0 < self.document_count <= self.occurrences:
            raise ValueError("package signature pool support counts are invalid")


@dataclass(frozen=True, slots=True)
class PackageGoodsFitAudit:
    fit_document_count: int
    cargo_group_count: int
    typed_package_group_count: int
    hs_heading_group_count: int
    thermal_group_count: int
    dangerous_goods_group_count: int
    excluded_incomplete_package_groups: int


@dataclass(frozen=True, slots=True)
class PackageGoodsFitSupport:
    """Immutable train-only joint support indexed by explicit semantic context."""

    fit_document_ids: tuple[str, ...]
    allowed_category_tokens: tuple[str, ...]
    hs_heading_rows: tuple[SignatureSupportRow, ...]
    hs_signature_pool_rows: tuple[PackageSignaturePoolSupportRow, ...]
    thermal_profile_rows: tuple[SignatureSupportRow, ...]
    dangerous_goods_rows: tuple[SignatureSupportRow, ...]
    audit: PackageGoodsFitAudit

    def __post_init__(self) -> None:
        if not self.fit_document_ids or len(self.fit_document_ids) != len(
            set(self.fit_document_ids)
        ):
            raise ValueError("fit document IDs must be non-empty and unique")
        allowed = frozenset(self.allowed_category_tokens)
        if not allowed or len(allowed) != len(self.allowed_category_tokens):
            raise ValueError("allowed package categories must be non-empty and unique")
        for row in (
            *self.hs_heading_rows,
            *self.hs_signature_pool_rows,
            *self.thermal_profile_rows,
            *self.dangerous_goods_rows,
        ):
            if not set(row.categories) <= allowed:
                raise ValueError("joint support contains a category outside the task vocabulary")

    def rows(
        self, *, basis: CompatibilityBasis, key: str, package_count: int
    ) -> tuple[SignatureSupportRow, ...]:
        source = {
            "fit_hs_heading_joint": self.hs_heading_rows,
            "fit_thermal_profile_joint": self.thermal_profile_rows,
            "fit_dangerous_goods_hazard_joint": self.dangerous_goods_rows,
        }[basis]
        return tuple(
            row for row in source if row.key == key and row.package_count == package_count
        )


@dataclass(frozen=True, slots=True)
class CompatibleCargoSelection:
    identities: tuple[AmbientGoodsIdentity | ThermalGoodsIdentity, ...]
    package_signature: PackageSignature
    basis: Literal[
        "fit_hs_signature_conditioned_heading_pool",
        "fit_thermal_profile_joint",
    ]
    support_key: str
    support_occurrences: int
    support_document_count: int


@dataclass(frozen=True, slots=True)
class DangerousGoodsPackageSelection:
    package_signature: PackageSignature
    basis: Literal["fit_dangerous_goods_hazard_joint"]
    support_key: str
    support_occurrences: int
    support_document_count: int


def supported_thermal_profiles(
    *,
    support: PackageGoodsFitSupport,
    goods_support: ThermalGoodsSupport,
    package_count: int,
    identity_count: int,
) -> tuple[ThermalProfile, ...]:
    """Return profiles with enough goods identities and a compatible package signature.

    A cargo group with no package rows does not require a package signature, but it
    still requires enough distinct registry identities for the target HS topology.
    Packaged cargo must additionally have a complete fit-observed signature at the
    exact package cardinality.  This makes thermal selection constraint-aware before
    the document/profile quota is sampled.
    """

    if package_count < 0 or identity_count <= 0:
        raise ValueError("thermal compatibility cardinalities are invalid")
    output: list[ThermalProfile] = []
    for profile in cast(tuple[ThermalProfile, ...], ("FROZEN", "CHILLED")):
        if len(goods_support.thermal_candidates(profile)) < identity_count:
            continue
        if package_count and not support.rows(
            basis="fit_thermal_profile_joint",
            key=profile,
            package_count=package_count,
        ):
            continue
        output.append(profile)
    return tuple(output)


def sample_supported_thermal_profile(
    *,
    profiles: Sequence[ThermalProfile],
    weights_permyriad: Mapping[ThermalProfile, int],
    stream: DeterministicStream,
) -> ThermalProfile:
    """Draw from configured profile weights renormalized over proven support."""

    expected = {"FROZEN", "CHILLED"}
    if set(weights_permyriad) != expected or sum(weights_permyriad.values()) != 10_000:
        raise ValueError("thermal profile weights must name both profiles and sum to 10000")
    available = tuple(dict.fromkeys(profiles))
    if not available or not set(available) <= expected:
        raise ValueError("supported thermal profiles must be a non-empty valid subset")
    total = sum(weights_permyriad[profile] for profile in available)
    if total <= 0:
        raise ValueError("supported thermal profiles have zero configured probability")
    draw = stream.derive("supported-thermal-profile").randbelow(total)
    cumulative = 0
    for profile in available:
        cumulative += weights_permyriad[profile]
        if draw < cumulative:
            return profile
    raise RuntimeError("supported thermal profile draw did not resolve")


class PackageDangerousGoodsFact(BaseModel):
    model_config = _STRICT

    properShippingName: NonEmptyText
    hs6: Annotated[str, StringConstraints(pattern=r"^[0-9]{6}$")]
    hazardCategory: NonEmptyText
    exactHazardClass: NonEmptyText
    subsidiaryHazardCategories: tuple[NonEmptyText, ...]
    packingGroupCategory: NonEmptyText | None


class PackageCompatibilityContext(BaseModel):
    """All facts relevant to one unsupported cargo-group DG package decision."""

    model_config = _STRICT

    dangerousGoods: tuple[PackageDangerousGoodsFact, ...] = Field(min_length=1, max_length=16)
    packageCount: Annotated[int, Field(gt=0, le=32)]

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.model_dump(mode="json")))


class PackageCompatibilityCandidate(BaseModel):
    model_config = _STRICT

    categories: tuple[NonEmptyText, ...] = Field(min_length=1, max_length=32)
    rationale: NonEmptyText


class PackageCompatibilityResolution(BaseModel):
    """Provider-constrained plausible signatures for one unsupported context."""

    model_config = _STRICT

    schemaVersion: Literal[1]
    context: PackageCompatibilityContext
    contextSha256: Sha256
    allowedCategoryTokensSha256: Sha256
    candidates: tuple[PackageCompatibilityCandidate, ...] = Field(min_length=1, max_length=8)
    method: Literal["provider_native_constrained_package_compatibility_v1"]

    @model_validator(mode="after")
    def digests_and_cardinality_match(self) -> PackageCompatibilityResolution:
        if self.context.sha256 != self.contextSha256:
            raise ValueError("package compatibility context digest differs from payload")
        signatures = tuple(row.categories for row in self.candidates)
        if len(signatures) != len(set(signatures)):
            raise ValueError("package compatibility candidates repeat a signature")
        if any(len(row) != self.context.packageCount for row in signatures):
            raise ValueError("package compatibility candidate cardinality differs from context")
        return self


def load_fit_partition_document_ids(
    path: Path, *, expected_split: str = "train"
) -> tuple[str, ...]:
    """Read the exact immutable runtime partition used to fit compatibility support."""

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"fit partition report is not a regular file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fit partition report has an unsupported structure") from error
    if not isinstance(value, Mapping):
        raise ValueError("fit partition report root must be an object")
    return fit_document_ids(value, expected_split=expected_split)


def _package_groups(patch: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for value in patch.get("cargoPackages") or ():
        if not isinstance(value, Mapping) or not isinstance(value.get("groupId"), str):
            raise ValueError("source cargo package is malformed")
        output[cast(str, value["groupId"])].append(value)
    return output


def _temperature_groups(
    patch: Mapping[str, Any],
    *,
    frozen_minimum_celsius: float,
    frozen_maximum_celsius: float,
    chilled_minimum_celsius: float,
    chilled_maximum_celsius: float,
) -> dict[str, ThermalProfile]:
    temperature_by_container: dict[str, float] = {}
    for value in patch.get("containers") or ():
        if not isinstance(value, Mapping) or not isinstance(value.get("containerNumber"), str):
            raise ValueError("source container is malformed")
        setpoint = value.get("temperatureSetpoint")
        if setpoint is None:
            continue
        if not isinstance(setpoint, Mapping) or not isinstance(setpoint.get("value"), (int, float)):
            raise ValueError("source temperature setpoint is malformed")
        if setpoint.get("unit") != "celsius":
            continue
        temperature_by_container[cast(str, value["containerNumber"])] = float(setpoint["value"])
    profiles: dict[str, ThermalProfile] = {}
    for allocation_group in patch.get("cargoAllocationGroups") or ():
        if not isinstance(allocation_group, Mapping) or not isinstance(
            allocation_group.get("groupId"), str
        ):
            raise ValueError("source allocation group is malformed")
        values = {
            temperature_by_container[number]
            for allocation in allocation_group.get("allocations") or ()
            if isinstance(allocation, Mapping)
            and isinstance((number := allocation.get("containerNumber")), str)
            and number in temperature_by_container
        }
        if not values:
            continue
        candidates: set[ThermalProfile] = set()
        if all(frozen_minimum_celsius <= value <= frozen_maximum_celsius for value in values):
            candidates.add("FROZEN")
        if all(chilled_minimum_celsius <= value <= chilled_maximum_celsius for value in values):
            candidates.add("CHILLED")
        if len(candidates) == 1:
            profiles[cast(str, allocation_group["groupId"])] = candidates.pop()
    return profiles


def _rows_from_counts(
    counts: Mapping[tuple[str, int, PackageSignature], int],
    documents: Mapping[tuple[str, int, PackageSignature], set[str]],
) -> tuple[SignatureSupportRow, ...]:
    return tuple(
        SignatureSupportRow(
            key=key,
            package_count=count,
            categories=signature,
            occurrences=occurrences,
            document_count=len(documents[(key, count, signature)]),
        )
        for (key, count, signature), occurrences in sorted(counts.items())
    )


def _signature_pool_rows_from_counts(
    counts: Mapping[tuple[int, PackageSignature], int],
    documents: Mapping[tuple[int, PackageSignature], set[str]],
    headings: Mapping[tuple[int, PackageSignature], Counter[str]],
) -> tuple[PackageSignaturePoolSupportRow, ...]:
    return tuple(
        PackageSignaturePoolSupportRow(
            heading_occurrences=tuple(sorted(headings[(count, signature)].items())),
            package_count=count,
            categories=signature,
            occurrences=occurrences,
            document_count=len(documents[(count, signature)]),
        )
        for (count, signature), occurrences in sorted(counts.items())
    )


def build_package_goods_fit_support(
    *,
    source_targets: Mapping[str, Mapping[str, Any]],
    fit_document_ids: Sequence[str],
    allowed_category_tokens: Sequence[str],
    frozen_minimum_celsius: float,
    frozen_maximum_celsius: float,
    chilled_minimum_celsius: float,
    chilled_maximum_celsius: float,
) -> PackageGoodsFitSupport:
    """Compile complete package signatures from the pinned fit partition only."""

    fit_ids = tuple(fit_document_ids)
    fit_set = frozenset(fit_ids)
    if not fit_set or len(fit_set) != len(fit_ids):
        raise ValueError("fit partition document IDs must be non-empty and unique")
    if not fit_set <= set(source_targets):
        raise ValueError("fit partition references a document absent from the source corpus")
    allowed = frozenset(allowed_category_tokens)
    if not allowed or len(allowed) != len(tuple(allowed_category_tokens)):
        raise ValueError("allowed package category tokens must be non-empty and unique")

    heading_counts: Counter[tuple[str, int, PackageSignature]] = Counter()
    signature_pool_counts: Counter[tuple[int, PackageSignature]] = Counter()
    thermal_counts: Counter[tuple[str, int, PackageSignature]] = Counter()
    dangerous_counts: Counter[tuple[str, int, PackageSignature]] = Counter()
    heading_documents: dict[tuple[str, int, PackageSignature], set[str]] = defaultdict(set)
    signature_pool_documents: dict[tuple[int, PackageSignature], set[str]] = defaultdict(set)
    signature_pool_headings: dict[tuple[int, PackageSignature], Counter[str]] = defaultdict(Counter)
    thermal_documents: dict[tuple[str, int, PackageSignature], set[str]] = defaultdict(set)
    dangerous_documents: dict[tuple[str, int, PackageSignature], set[str]] = defaultdict(set)
    cargo_groups = typed_groups = hs_groups = thermal_groups = dangerous_groups = excluded = 0

    for document_id in fit_ids:
        target = source_targets[document_id]
        patch = target.get("documentPatch")
        if not isinstance(patch, Mapping):
            raise ValueError("source target lacks a documentPatch object")
        packages_by_group = _package_groups(patch)
        thermal_by_group = _temperature_groups(
            patch,
            frozen_minimum_celsius=frozen_minimum_celsius,
            frozen_maximum_celsius=frozen_maximum_celsius,
            chilled_minimum_celsius=chilled_minimum_celsius,
            chilled_maximum_celsius=chilled_maximum_celsius,
        )
        for group in patch.get("cargoGroups") or ():
            cargo_groups += 1
            if not isinstance(group, Mapping) or not isinstance(group.get("groupId"), str):
                raise ValueError("source cargo group is malformed")
            group_id = cast(str, group["groupId"])
            package_values = packages_by_group.get(group_id, ())
            if not package_values:
                continue
            categories = tuple(value.get("typeCategory") for value in package_values)
            if any(not isinstance(value, str) or value not in allowed for value in categories):
                excluded += 1
                continue
            signature = cast(PackageSignature, categories)
            package_count = len(signature)
            typed_groups += 1

            hs6_values = tuple(
                value[:6]
                for value in group.get("hsCodes") or ()
                if isinstance(value, str) and len(value) >= 6 and value[:6].isdigit()
            )
            headings = tuple(sorted({value[:4] for value in hs6_values}))
            if headings:
                hs_groups += 1
                signature_key = (package_count, signature)
                signature_pool_counts[signature_key] += 1
                signature_pool_documents[signature_key].add(document_id)
                signature_pool_headings[signature_key].update(
                    value[:4] for value in hs6_values
                )
                for heading in headings:
                    key = (heading, package_count, signature)
                    heading_counts[key] += 1
                    heading_documents[key].add(document_id)

            profile = thermal_by_group.get(group_id)
            if profile is not None:
                thermal_groups += 1
                key = (profile, package_count, signature)
                thermal_counts[key] += 1
                thermal_documents[key].add(document_id)

            hazards = tuple(
                value.get("hazardCategory")
                for value in group.get("dangerousGoods") or ()
                if isinstance(value, Mapping) and isinstance(value.get("hazardCategory"), str)
            )
            if hazards:
                dangerous_groups += 1
                for hazard in sorted(set(cast(tuple[str, ...], hazards))):
                    key = (hazard, package_count, signature)
                    dangerous_counts[key] += 1
                    dangerous_documents[key].add(document_id)

    return PackageGoodsFitSupport(
        fit_document_ids=fit_ids,
        allowed_category_tokens=tuple(sorted(allowed)),
        hs_heading_rows=_rows_from_counts(heading_counts, heading_documents),
        hs_signature_pool_rows=_signature_pool_rows_from_counts(
            signature_pool_counts,
            signature_pool_documents,
            signature_pool_headings,
        ),
        thermal_profile_rows=_rows_from_counts(thermal_counts, thermal_documents),
        dangerous_goods_rows=_rows_from_counts(dangerous_counts, dangerous_documents),
        audit=PackageGoodsFitAudit(
            fit_document_count=len(fit_ids),
            cargo_group_count=cargo_groups,
            typed_package_group_count=typed_groups,
            hs_heading_group_count=hs_groups,
            thermal_group_count=thermal_groups,
            dangerous_goods_group_count=dangerous_groups,
            excluded_incomplete_package_groups=excluded,
        ),
    )


class _WeightedSupportRow(Protocol):
    occurrences: int


def _weighted_row[WeightedRow: _WeightedSupportRow](
    rows: Sequence[WeightedRow], *, stream: DeterministicStream
) -> WeightedRow:
    if not rows:
        raise ValueError("package compatibility has no fit-supported candidate")
    total = sum(row.occurrences for row in rows)
    draw = stream.randbelow(total)
    cumulative = 0
    for row in rows:
        cumulative += row.occurrences
        if draw < cumulative:
            return row
    raise RuntimeError("weighted package compatibility draw did not resolve")


def sample_compatible_cargo(
    *,
    support: PackageGoodsFitSupport,
    goods_support: ThermalGoodsSupport,
    profile: ThermalProfile | None,
    package_count: int,
    identity_count: int,
    stream: DeterministicStream,
    excluded_hs6: set[str],
) -> CompatibleCargoSelection:
    """Co-sample registry goods and a complete fit-observed package signature."""

    if package_count <= 0 or identity_count <= 0:
        raise ValueError("compatible cargo sampling requires positive cardinalities")
    support_key: str
    if profile is not None:
        rows = support.rows(
            basis="fit_thermal_profile_joint", key=profile, package_count=package_count
        )
        selected = _weighted_row(rows, stream=stream.derive("package-signature"))
        identity_pool: tuple[AmbientGoodsIdentity | ThermalGoodsIdentity, ...] = (
            goods_support.thermal_candidates(profile)
        )
        basis: Literal[
            "fit_hs_signature_conditioned_heading_pool",
            "fit_thermal_profile_joint",
        ] = "fit_thermal_profile_joint"
        support_key = profile
    else:
        available_by_heading: dict[
            str, list[AmbientGoodsIdentity | ThermalGoodsIdentity]
        ] = defaultdict(list)
        for value in goods_support.ambient:
            if value.hs6 not in excluded_hs6:
                available_by_heading[value.hs6[:4]].append(value)
        eligible_rows = tuple(
            row
            for row in support.hs_signature_pool_rows
            if row.package_count == package_count
            and sum(
                len(available_by_heading.get(heading, ()))
                for heading, _occurrences in row.heading_occurrences
            )
            >= identity_count
        )
        selected = _weighted_row(eligible_rows, stream=stream.derive("package-signature"))
        basis = "fit_hs_signature_conditioned_heading_pool"
        support_key = sha256_bytes(
            canonical_json_bytes(
                {
                    "packageSignature": selected.categories,
                    "headingOccurrences": selected.heading_occurrences,
                }
            )
        )
        identities = []
        remaining_by_heading = {
            heading: list(values) for heading, values in available_by_heading.items()
        }
        for order in range(identity_count):
            active = tuple(
                (heading, weight)
                for heading, weight in selected.heading_occurrences
                if remaining_by_heading.get(heading)
            )
            total_weight = sum(weight for _heading, weight in active)
            draw = stream.derive(f"heading-{order}").randbelow(total_weight)
            cumulative = 0
            heading = active[-1][0]
            for candidate_heading, weight in active:
                cumulative += weight
                if draw < cumulative:
                    heading = candidate_heading
                    break
            remaining = remaining_by_heading[heading]
            index = stream.derive(f"identity-{order}").randbelow(len(remaining))
            identities.append(remaining.pop(index))
    if profile is not None:
        available = tuple(value for value in identity_pool if value.hs6 not in excluded_hs6)
        if len(available) < identity_count:
            raise ValueError("compatible HS identity pool cannot satisfy distinct cardinality")
        identities = []
        remaining = list(available)
        for order in range(identity_count):
            index = stream.derive(f"identity-{order}").randbelow(len(remaining))
            identities.append(remaining.pop(index))
    excluded_hs6.update(value.hs6 for value in identities)
    return CompatibleCargoSelection(
        identities=tuple(identities),
        package_signature=selected.categories,
        basis=basis,
        support_key=support_key,
        support_occurrences=selected.occurrences,
        support_document_count=selected.document_count,
    )


def sample_dangerous_goods_package(
    *,
    support: PackageGoodsFitSupport,
    hazard_categories: Sequence[str],
    package_count: int,
    stream: DeterministicStream,
) -> DangerousGoodsPackageSelection | None:
    """Resolve DG packages only when every hazard supports one common signature."""

    hazards = tuple(sorted(set(hazard_categories)))
    if not hazards:
        raise ValueError("dangerous-goods package sampling requires at least one hazard")
    rows_by_hazard = {
        hazard: support.rows(
            basis="fit_dangerous_goods_hazard_joint",
            key=hazard,
            package_count=package_count,
        )
        for hazard in hazards
    }
    if any(not rows for rows in rows_by_hazard.values()):
        return None
    common = set.intersection(
        *({row.categories for row in rows} for rows in rows_by_hazard.values())
    )
    if not common:
        return None
    combined = tuple(
        SignatureSupportRow(
            key="|".join(hazards),
            package_count=package_count,
            categories=signature,
            occurrences=sum(
                next(
                    row.occurrences
                    for row in rows_by_hazard[hazard]
                    if row.categories == signature
                )
                for hazard in hazards
            ),
            document_count=sum(
                next(
                    row.document_count
                    for row in rows_by_hazard[hazard]
                    if row.categories == signature
                )
                for hazard in hazards
            ),
        )
        for signature in sorted(common)
    )
    selected = _weighted_row(combined, stream=stream.derive("package-signature"))
    return DangerousGoodsPackageSelection(
        package_signature=selected.categories,
        basis="fit_dangerous_goods_hazard_joint",
        support_key="|".join(hazards),
        support_occurrences=selected.occurrences,
        support_document_count=selected.document_count,
    )


def apply_package_signature(
    *, target: dict[str, Any], group_id: str, signature: PackageSignature
) -> None:
    """Apply one complete ordered task-facing signature without changing topology."""

    patch = cast(dict[str, Any], target["documentPatch"])
    packages = [
        value
        for value in patch.get("cargoPackages") or ()
        if isinstance(value, dict) and value.get("groupId") == group_id
    ]
    if len(packages) != len(signature):
        raise ValueError("package signature cardinality differs from target topology")
    for value, category in zip(packages, signature, strict=True):
        value["typeCategory"] = category
