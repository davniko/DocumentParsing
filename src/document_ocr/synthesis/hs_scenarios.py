"""Train-isolated, registry-backed Harmonized System synthesis scenarios.

Source labels contribute only a coarse chapter prior after their first six
digits have been validated against the pinned HS edition.  Every generated
global identity is sampled from the compiled registry.  A source field longer
than six digits retains its exact digit length: an exact UK tariff leaf is used
when available and selected for a GB route; otherwise a fresh, explicitly
unregistered national suffix is generated.  Source suffixes are never copied or
claimed to be authoritative tariff identities.

Dangerous-goods status and printed OCR formatting are deliberately outside
this module.  Neither can be inferred from an HS identity.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.hs_registry import (
    HsGlobalSubheading,
    HsRegistryError,
    UkGlobalTariffRegistry,
    UkTariffCommodity,
)

IsoCountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

HsScenarioComponent = Literal[
    "fit_observed_chapter_registry_hs6",
    "registry_wide_hs6_exploration",
]
HsOutputScope = Literal[
    "global_hs6",
    "gb_tariff_10",
    "synthetic_national_extension",
]
HsExtensionStatus = Literal[
    "not_applicable_global_hs6",
    "exact_gb_tariff_registry_leaf",
    "synthetic_unregistered_national_suffix",
]


class HsScenarioPolicy(BaseModel):
    """All statistical and national-extension choices are explicit policy."""

    model_config = _STRICT

    schema_version: Literal[1]
    observed_chapter_mixture_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    registry_wide_mixture_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    gb_tariff_extension_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    observed_chapter_weighting: Literal["fit_document_count_v1"]
    observed_chapter_hs6_weighting: Literal["uniform_registry_hs6_within_selected_chapter_v1"]
    registry_hs6_weighting: Literal["uniform_registry_hs6_v1"]
    gb_tariff_leaf_weighting: Literal["uniform_registry_leaves_v1"]
    output_length_method: Literal["preserve_source_length_exact_registry_else_random_suffix_v1"]
    minimum_output_digits: Literal[6]
    maximum_output_digits: Annotated[int, Field(ge=6, le=18)]
    maximum_extension_attempts: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def mixture_is_complete(self) -> HsScenarioPolicy:
        if self.observed_chapter_mixture_permyriad + self.registry_wide_mixture_permyriad != 10_000:
            raise ValueError("HS chapter mixture weights must sum exactly to 10000")
        if self.maximum_output_digits < self.minimum_output_digits:
            raise ValueError("HS output digit bounds are reversed")
        return self


@dataclass(frozen=True, slots=True)
class HsFitObservation:
    source_document_id: str
    source_row_id: str
    semantic_code: str

    def __post_init__(self) -> None:
        if not self.source_document_id or not self.source_row_id:
            raise ValueError("HS fit observations require non-empty source identities")
        if re.fullmatch(r"[0-9]{6,18}", self.semantic_code) is None:
            raise ValueError("source HS codes must contain 6-18 digits")

    @property
    def global_hs6(self) -> str:
        return self.semantic_code[:6]


@dataclass(frozen=True, slots=True)
class HsSupportExclusion:
    source_document_id: str
    source_row_id: str
    semantic_code: str
    global_hs6: str
    reason: Literal["global_hs6_absent_from_pinned_edition"]


@dataclass(frozen=True, slots=True)
class HsCodeSupport:
    code: str
    fit_document_count: int

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9]{6}", self.code) is None or self.fit_document_count <= 0:
            raise ValueError("HS6 support requires an exact code and positive document count")


@dataclass(frozen=True, slots=True)
class HsChapterSupport:
    chapter_code: str
    fit_document_count: int
    observed_hs6: tuple[HsCodeSupport, ...]
    registry_hs6_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9]{2}", self.chapter_code) is None:
            raise ValueError("HS chapter support requires exactly two digits")
        if self.fit_document_count <= 0 or not self.observed_hs6 or not self.registry_hs6_codes:
            raise ValueError("HS chapter support must be non-empty")
        observed = tuple(row.code for row in self.observed_hs6)
        if observed != tuple(sorted(set(observed))):
            raise ValueError("observed HS6 support must be unique and sorted")
        if self.registry_hs6_codes != tuple(sorted(set(self.registry_hs6_codes))):
            raise ValueError("registry HS6 support must be unique and sorted")
        if any(code[:2] != self.chapter_code for code in observed + self.registry_hs6_codes):
            raise ValueError("HS6 support lies outside its chapter")


@dataclass(frozen=True, slots=True)
class HsScenarioAudit:
    fit_document_count: int
    input_source_rows: int
    supported_source_rows: int
    excluded_source_rows: int
    documents_with_supported_hs: int
    observed_chapter_count: int
    observed_hs6_count: int
    registry_chapter_count: int
    registry_hs6_count: int
    source_length_counts: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class HsScenarioSupport:
    fit_document_ids: tuple[str, ...]
    registry_version: str
    registry_content_sha256: str
    on_date: date
    chapters: tuple[HsChapterSupport, ...]
    all_registry_hs6_codes: tuple[str, ...]
    exclusions: tuple[HsSupportExclusion, ...]
    audit: HsScenarioAudit

    def __post_init__(self) -> None:
        if self.fit_document_ids != tuple(sorted(set(self.fit_document_ids))):
            raise ValueError("HS support fit document IDs must be unique and sorted")
        chapter_codes = tuple(row.chapter_code for row in self.chapters)
        if chapter_codes != tuple(sorted(set(chapter_codes))):
            raise ValueError("HS chapter support must be unique and sorted")
        if self.all_registry_hs6_codes != tuple(sorted(set(self.all_registry_hs6_codes))):
            raise ValueError("global registry HS6 support must be unique and sorted")


class HsScenario(BaseModel):
    """One semantic output identity; printed formatting remains unresolved."""

    model_config = _STRICT

    component: HsScenarioComponent
    customs_jurisdiction: IsoCountryCode
    global_identity: HsGlobalSubheading
    output_scope: HsOutputScope
    output_code: Annotated[str, StringConstraints(pattern=r"^[0-9]{6,18}$")]
    output_digits: Annotated[int, Field(ge=6, le=18)]
    extension_status: HsExtensionStatus
    gb_tariff_identity: UkTariffCommodity | None
    printed_surface_status: Literal["pending_text_realization"]
    dangerous_goods_status: Literal["independent_not_inferred_from_hs"]

    @model_validator(mode="after")
    def identity_scope_is_consistent(self) -> HsScenario:
        if len(self.output_code) != self.output_digits:
            raise ValueError("HS output digit count differs from its code")
        if self.output_scope == "global_hs6":
            if (
                self.gb_tariff_identity is not None
                or self.output_code != self.global_identity.code
                or self.extension_status != "not_applicable_global_hs6"
            ):
                raise ValueError("global HS scenario carries a national extension")
        elif self.output_scope == "gb_tariff_10":
            if self.customs_jurisdiction != "GB" or self.gb_tariff_identity is None:
                raise ValueError("GB tariff output requires GB customs jurisdiction")
            if (
                self.gb_tariff_identity.hs6 != self.global_identity.code
                or self.output_code != self.gb_tariff_identity.code
                or self.extension_status != "exact_gb_tariff_registry_leaf"
            ):
                raise ValueError("GB tariff output is not an exact child of its global HS6")
        elif (
            self.gb_tariff_identity is not None
            or self.output_digits <= 6
            or not self.output_code.startswith(self.global_identity.code)
            or self.extension_status != "synthetic_unregistered_national_suffix"
        ):
            raise ValueError("synthetic national HS extension is internally inconsistent")
        return self


def hs_fit_observations(
    *,
    fit_document_ids: Sequence[str],
    cargo_hs_code_rows: Sequence[Mapping[str, Any]],
) -> tuple[HsFitObservation, ...]:
    """Project relation-table HS rows without admitting validation documents."""

    fit_ids = tuple(sorted(fit_document_ids))
    if not fit_ids or fit_ids != tuple(sorted(set(fit_ids))):
        raise ValueError("fit document IDs must be non-empty and unique")
    fit_set = frozenset(fit_ids)
    observations: list[HsFitObservation] = []
    row_ids: set[str] = set()
    for row in cargo_hs_code_rows:
        document_id = row.get("document_id")
        if document_id not in fit_set:
            continue
        source_row_id = row.get("cargo_group_value_id")
        value = row.get("value")
        if not isinstance(source_row_id, str) or not isinstance(value, str):
            raise ValueError("cargo HS rows require string identity and value fields")
        if source_row_id in row_ids:
            raise ValueError(f"duplicate cargo HS row identity: {source_row_id}")
        row_ids.add(source_row_id)
        observations.append(
            HsFitObservation(
                source_document_id=document_id,
                source_row_id=source_row_id,
                semantic_code=value,
            )
        )
    return tuple(
        sorted(
            observations,
            key=lambda row: (row.source_document_id, row.source_row_id),
        )
    )


def build_hs_scenario_support(
    *,
    fit_document_ids: Sequence[str],
    observations: Sequence[HsFitObservation],
    registry: UkGlobalTariffRegistry,
    on_date: date,
) -> HsScenarioSupport:
    """Build chapter priors solely from the declared fit partition."""

    fit_ids = tuple(sorted(fit_document_ids))
    if not fit_ids or fit_ids != tuple(sorted(set(fit_ids))):
        raise ValueError("fit document IDs must be non-empty and unique")
    registry.require_global(registry.global_codes[0], on_date=on_date)
    fit_set = frozenset(fit_ids)
    identities: set[tuple[str, str]] = set()
    supported: list[HsFitObservation] = []
    exclusions: list[HsSupportExclusion] = []
    for row in observations:
        if row.source_document_id not in fit_set:
            raise ValueError("HS fit observation lies outside the declared fit partition")
        identity = (row.source_document_id, row.source_row_id)
        if identity in identities:
            raise ValueError(f"duplicate HS fit observation identity: {identity!r}")
        identities.add(identity)
        try:
            registry.require_global(row.global_hs6, on_date=on_date)
        except HsRegistryError:
            exclusions.append(
                HsSupportExclusion(
                    source_document_id=row.source_document_id,
                    source_row_id=row.source_row_id,
                    semantic_code=row.semantic_code,
                    global_hs6=row.global_hs6,
                    reason="global_hs6_absent_from_pinned_edition",
                )
            )
        else:
            supported.append(row)

    chapter_documents: dict[str, set[str]] = defaultdict(set)
    code_documents: dict[str, set[str]] = defaultdict(set)
    for row in supported:
        chapter_documents[row.global_hs6[:2]].add(row.source_document_id)
        code_documents[row.global_hs6].add(row.source_document_id)
    registry_by_chapter: dict[str, list[str]] = defaultdict(list)
    for code in registry.global_codes:
        registry_by_chapter[code[:2]].append(code)
    chapters: list[HsChapterSupport] = []
    for chapter in sorted(chapter_documents):
        observed_codes = tuple(
            HsCodeSupport(code=code, fit_document_count=len(documents))
            for code, documents in sorted(code_documents.items())
            if code[:2] == chapter
        )
        chapters.append(
            HsChapterSupport(
                chapter_code=chapter,
                fit_document_count=len(chapter_documents[chapter]),
                observed_hs6=observed_codes,
                registry_hs6_codes=tuple(registry_by_chapter[chapter]),
            )
        )
    length_counts = Counter(len(row.semantic_code) for row in observations)
    return HsScenarioSupport(
        fit_document_ids=fit_ids,
        registry_version=registry.receipt.version,
        registry_content_sha256=registry.receipt.registry_content_sha256,
        on_date=on_date,
        chapters=tuple(chapters),
        all_registry_hs6_codes=registry.global_codes,
        exclusions=tuple(
            sorted(exclusions, key=lambda row: (row.source_document_id, row.source_row_id))
        ),
        audit=HsScenarioAudit(
            fit_document_count=len(fit_ids),
            input_source_rows=len(observations),
            supported_source_rows=len(supported),
            excluded_source_rows=len(exclusions),
            documents_with_supported_hs=len({row.source_document_id for row in supported}),
            observed_chapter_count=len(chapters),
            observed_hs6_count=len(code_documents),
            registry_chapter_count=len(registry_by_chapter),
            registry_hs6_count=len(registry.global_codes),
            source_length_counts=tuple(sorted(length_counts.items())),
        ),
    )


def sample_hs_scenario(
    *,
    support: HsScenarioSupport,
    registry: UkGlobalTariffRegistry,
    policy: HsScenarioPolicy,
    customs_jurisdiction: str,
    source_output_digits: int,
    stream: DeterministicStream,
    excluded_output_codes: Collection[str] = (),
    excluded_global_hs6: Collection[str] = (),
) -> HsScenario:
    """Sample one unique global identity and preserve the source digit length.

    GB extension is decided before the global subheading is selected.  When an
    extension is requested, the eligible HS6 pool is restricted to subheadings
    that actually have a pinned UKGT leaf.  This avoids a late, data-dependent
    failure and makes the configured extension probability conditional on a
    valid national projection.
    """

    if registry.receipt.version != support.registry_version or (
        registry.receipt.registry_content_sha256 != support.registry_content_sha256
    ):
        raise ValueError("HS scenario support and registry receipts differ")
    if re.fullmatch(r"[A-Z]{2}", customs_jurisdiction) is None:
        raise ValueError("customs jurisdiction must be an ISO alpha-2 code")
    if not policy.minimum_output_digits <= source_output_digits <= policy.maximum_output_digits:
        raise ValueError("source HS output length lies outside configured bounds")
    output_exclusions = frozenset(excluded_output_codes)
    global_exclusions = frozenset(excluded_global_hs6)
    if any(re.fullmatch(r"[0-9]{6,18}", value) is None for value in output_exclusions):
        raise ValueError("excluded HS output codes must contain 6-18 digits")
    if any(re.fullmatch(r"[0-9]{6}", value) is None for value in global_exclusions):
        raise ValueError("excluded global HS identities must contain exactly 6 digits")

    request_gb_extension = (
        source_output_digits == 10
        and customs_jurisdiction == "GB"
        and (stream.derive("gb-extension").randbelow(10_000) < policy.gb_tariff_extension_permyriad)
    )
    component_draw = stream.derive("component").randbelow(10_000)
    if component_draw < policy.observed_chapter_mixture_permyriad:
        if not support.chapters:
            raise ValueError("observed-chapter sampling requested without fit support")
        total = sum(row.fit_document_count for row in support.chapters)
        draw = stream.derive("observed-chapter").randbelow(total)
        cumulative = 0
        selected_chapter: HsChapterSupport | None = None
        for row in support.chapters:
            cumulative += row.fit_document_count
            if draw < cumulative:
                selected_chapter = row
                break
        if selected_chapter is None:
            raise AssertionError("HS chapter cumulative support is inconsistent")
        codes = selected_chapter.registry_hs6_codes
        component: HsScenarioComponent = "fit_observed_chapter_registry_hs6"
    else:
        codes = support.all_registry_hs6_codes
        component = "registry_wide_hs6_exploration"

    eligible_codes = tuple(
        code
        for code in codes
        if code not in global_exclusions
        and code not in output_exclusions
        and (
            not request_gb_extension
            or any(
                row.code not in output_exclusions
                for row in registry.uk_candidates(code, on_date=support.on_date)
            )
        )
    )
    if not eligible_codes:
        raise HsRegistryError(
            "HS scenario has no eligible unique identity for its sampled component and scope"
        )
    global_code = eligible_codes[stream.derive("global-hs6").randbelow(len(eligible_codes))]
    global_identity = registry.require_global(global_code, on_date=support.on_date)

    gb_identity: UkTariffCommodity | None = None
    if request_gb_extension:
        candidates = tuple(
            row
            for row in registry.uk_candidates(global_code, on_date=support.on_date)
            if row.code not in output_exclusions
        )
        if not candidates:
            raise AssertionError("eligible GB extension code has no eligible tariff leaf")
        gb_identity = candidates[stream.derive("gb-tariff-leaf").randbelow(len(candidates))]

    output_scope: HsOutputScope = "global_hs6"
    output_code = global_identity.code
    extension_status: HsExtensionStatus = "not_applicable_global_hs6"
    if gb_identity is not None:
        output_scope = "gb_tariff_10"
        output_code = gb_identity.code
        extension_status = "exact_gb_tariff_registry_leaf"
    elif source_output_digits > 6:
        suffix_digits = source_output_digits - 6
        suffix_space = 10**suffix_digits
        for attempt in range(policy.maximum_extension_attempts):
            suffix = stream.derive(f"national-suffix:{attempt}").randbelow(suffix_space)
            candidate = f"{global_identity.code}{suffix:0{suffix_digits}d}"
            if candidate not in output_exclusions:
                output_code = candidate
                break
        else:
            raise HsRegistryError(
                "HS national extension exhausted its configured unique proposal attempts"
            )
        output_scope = "synthetic_national_extension"
        extension_status = "synthetic_unregistered_national_suffix"

    return HsScenario(
        component=component,
        customs_jurisdiction=customs_jurisdiction,
        global_identity=global_identity,
        output_scope=output_scope,
        output_code=output_code,
        output_digits=source_output_digits,
        extension_status=extension_status,
        gb_tariff_identity=gb_identity,
        printed_surface_status="pending_text_realization",
        dangerous_goods_status="independent_not_inferred_from_hs",
    )
