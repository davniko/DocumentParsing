"""Pinned Harmonized System and jurisdictional tariff identities.

The UK Global Tariff (UKGT) is used here as an openly licensed, official
implementation of the current HS hierarchy.  The compiler deliberately keeps
the globally portable six-digit HS identity separate from UK-specific
ten-digit declarable commodity codes.  Formatting found in source documents
is a third, independent concern represented by :class:`HsCodeSurface`.

Nothing in this module guesses, pads, truncates, or manufactures a code.  A
registry is available only after both the CSVW metadata and commodities report
match an exact source pin and the complete hierarchy passes validation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, Literal, cast
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from document_ocr.hashing import canonical_json_bytes
from document_ocr.synthesis.generators import DeterministicStream

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Version = Annotated[str, StringConstraints(pattern=r"^v[0-9]+\.[0-9]+\.[0-9]+$")]
HttpsUrl = Annotated[str, StringConstraints(pattern=r"^https://[^\s]+$")]
Code2 = Annotated[str, StringConstraints(pattern=r"^[0-9]{2}$")]
Code4 = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}$")]
Code6 = Annotated[str, StringConstraints(pattern=r"^[0-9]{6}$")]
Code10 = Annotated[str, StringConstraints(pattern=r"^[0-9]{10}$")]
CanonicalHsCode = Annotated[str, StringConstraints(pattern=r"^[0-9]{6,18}$")]
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_SOURCE_COLUMNS = (
    "id",
    "commodity__sid",
    "commodity__code",
    "commodity__suffix",
    "commodity__description",
    "commodity__validity_start",
    "commodity__validity_end",
    "parent__sid",
    "parent__code",
    "parent__suffix",
)
_DATASET_ID: Final = "uk-tariff-2021-01-01"
_TABLE_PATH: Final = "tables/commodities-report/data?format=csv&download"
_METADATA_QUERY: Final = "metadata?format=csvw&download"
_LICENSE_URL: Final = "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
_DATASET_TITLE: Final = "Tariffs to trade with the UK from 1 January 2021"
_DATASET_CREATOR: Final = "Department for International Trade"
_PARSER_CONTRACT: Final = "ukgt_commodities_report_csvw_v1"
_ATTRIBUTION: Final = (
    "Contains public sector information from the UK Global Tariff, licensed under the "
    "Open Government Licence v3.0."
)
_LICENSE_CAVEAT: Final = (
    "The Open Government Licence excludes third-party rights; UKGT descriptions are not "
    "redistributed as independently licensed WCO nomenclature."
)
_DESCRIPTION_AUTHORITY: Final = (
    "UK Global Tariff implementation text; not asserted as canonical WCO nomenclature"
)
_DESCRIPTION_NORMALIZATION: Final = "unicode_preserving_whitespace_collapse_v1"
_HS2022_EFFECTIVE_FROM = date(2022, 1, 1)
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_SPECIAL_NATIONAL_CHAPTERS = frozenset({"98", "99"})
_SURFACE_PATTERN = re.compile(r"^[0-9 .,:/-]+$")


class HsRegistryError(RuntimeError):
    """A pinned tariff registry cannot be compiled without ambiguity."""


class UkGlobalTariffSourcePin(BaseModel):
    """Exact identity and expected shape of one official UKGT snapshot."""

    model_config = _STRICT

    version: Version
    snapshot_date: date
    metadata_url: HttpsUrl
    metadata_bytes: Annotated[int, Field(gt=0, le=_MAX_METADATA_BYTES)]
    metadata_sha256: Sha256
    report_url: HttpsUrl
    report_bytes: Annotated[int, Field(gt=0, le=_MAX_REPORT_BYTES)]
    report_sha256: Sha256
    expected_source_rows: Annotated[int, Field(gt=0)]
    expected_global_hs6_codes_sha256: Sha256
    expected_global_hs6_records: Annotated[int, Field(gt=0)]
    expected_uk_tariff_records: Annotated[int, Field(gt=0)]
    hs_edition: Literal["HS2022"]
    hs_effective_from: date
    license_name: Literal["Open Government Licence v3.0"]
    license_url: Literal[
        "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
    ]
    attribution: Literal[
        "Contains public sector information from the UK Global Tariff, licensed under the "
        "Open Government Licence v3.0."
    ]
    license_caveat: Literal[
        "The Open Government Licence excludes third-party rights; UKGT descriptions are not "
        "redistributed as independently licensed WCO nomenclature."
    ]

    @model_validator(mode="after")
    def source_identity_is_exact(self) -> UkGlobalTariffSourcePin:
        base = f"https://data.api.trade.gov.uk/v1/datasets/{_DATASET_ID}/versions/{self.version}/"
        if self.metadata_url != base + _METADATA_QUERY:
            raise ValueError("metadata_url must pin the exact-version official CSVW metadata")
        if self.report_url != base + _TABLE_PATH:
            raise ValueError("report_url must pin the exact-version commodities-report CSV")
        for value in (self.metadata_url, self.report_url):
            parsed = urlsplit(value)
            if parsed.scheme != "https" or parsed.netloc != "data.api.trade.gov.uk":
                raise ValueError("UKGT sources must use the official data.api.trade.gov.uk host")
        if self.snapshot_date < self.hs_effective_from:
            raise ValueError("snapshot date predates the asserted HS edition")
        if self.hs_effective_from != _HS2022_EFFECTIVE_FROM:
            raise ValueError("HS2022 effective date must be 2022-01-01")
        return self


class HsGlobalSubheading(BaseModel):
    """One globally portable six-digit HS 2022 subheading."""

    model_config = _STRICT

    edition: Literal["HS2022"]
    valid_from: date
    chapter_code: Code2
    chapter_description: NonEmptyText
    heading_code: Code4
    heading_description: NonEmptyText
    code: Code6
    source_description: NonEmptyText
    description: NonEmptyText
    description_authority: Literal[
        "UK Global Tariff implementation text; not asserted as canonical WCO nomenclature"
    ]

    @model_validator(mode="after")
    def hierarchy_is_prefix_consistent(self) -> HsGlobalSubheading:
        if self.valid_from != _HS2022_EFFECTIVE_FROM:
            raise ValueError("global HS subheading has an unexpected edition start")
        if self.code[:2] != self.chapter_code or self.code[:4] != self.heading_code:
            raise ValueError("global HS hierarchy is not prefix-consistent")
        if self.chapter_code in _SPECIAL_NATIONAL_CHAPTERS:
            raise ValueError("national special-use chapters are not global HS subheadings")
        return self


class UkTariffDescriptionNode(BaseModel):
    """One exact SID-addressed UKGT ancestry node and its display projection."""

    model_config = _STRICT

    source_sid: Annotated[str, StringConstraints(pattern=r"^[0-9]+$")]
    code: Code10
    suffix: Code2
    source_description: NonEmptyText
    description: NonEmptyText


class UkTariffCommodity(BaseModel):
    """One unambiguous declarable UK ten-digit commodity code."""

    model_config = _STRICT

    jurisdiction: Literal["GB"]
    source_sid: Annotated[str, StringConstraints(pattern=r"^[0-9]+$")]
    code: Code10
    hs6: Code6
    source_description: NonEmptyText
    description: NonEmptyText
    description_path: tuple[UkTariffDescriptionNode, ...] = Field(min_length=1)
    description_authority: Literal[
        "UK Global Tariff implementation text; not asserted as canonical WCO nomenclature"
    ]
    valid_from: date
    valid_to: date | None

    @model_validator(mode="after")
    def identity_and_validity_are_consistent(self) -> UkTariffCommodity:
        if self.code[:6] != self.hs6:
            raise ValueError("UK tariff code does not extend its global HS6 identity")
        if self.code[:2] in _SPECIAL_NATIONAL_CHAPTERS:
            raise ValueError("special-use UK codes cannot be projected to a global HS6 identity")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("UK tariff validity interval is inverted")
        leaf = self.description_path[-1]
        if (
            leaf.source_sid != self.source_sid
            or leaf.code != self.code
            or leaf.source_description != self.source_description
            or leaf.description != self.description
        ):
            raise ValueError("UK tariff description path must end at the leaf description")
        return self


class HsCodeSurface(BaseModel):
    """Exact printed formatting paired with, but separate from, semantic digits."""

    model_config = _STRICT

    semantic_code: CanonicalHsCode
    surface: Annotated[str, StringConstraints(min_length=6, max_length=96)]

    @model_validator(mode="after")
    def surface_preserves_exact_digits(self) -> HsCodeSurface:
        if self.surface != self.surface.strip():
            raise ValueError("HS code surface must not contain outer whitespace")
        if _SURFACE_PATTERN.fullmatch(self.surface) is None:
            raise ValueError("HS code surface contains unsupported characters")
        digits = "".join(character for character in self.surface if character.isdigit())
        if digits != self.semantic_code:
            raise ValueError(
                "HS code surface changes, pads, truncates, or reorders semantic digits"
            )
        return self


class UkGlobalTariffAudit(BaseModel):
    """Complete row and exclusion accounting for the compiled snapshot."""

    model_config = _STRICT

    source_rows: Annotated[int, Field(gt=0)]
    normalized_description_rows: Annotated[int, Field(ge=0)]
    global_chapters: Annotated[int, Field(gt=0)]
    global_headings: Annotated[int, Field(gt=0)]
    global_hs6_records: Annotated[int, Field(gt=0)]
    hierarchy_leaf_rows: Annotated[int, Field(gt=0)]
    excluded_nondeclarable_suffix_leaf_rows: Annotated[int, Field(ge=0)]
    excluded_ambiguous_code_groups: Annotated[int, Field(ge=0)]
    excluded_ambiguous_code_rows: Annotated[int, Field(ge=0)]
    excluded_special_chapter_leaf_rows: Annotated[int, Field(ge=0)]
    uk_tariff_records: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def leaf_accounting_balances(self) -> UkGlobalTariffAudit:
        classified = (
            self.excluded_nondeclarable_suffix_leaf_rows
            + self.excluded_ambiguous_code_rows
            + self.excluded_special_chapter_leaf_rows
            + self.uk_tariff_records
        )
        if classified != self.hierarchy_leaf_rows:
            raise ValueError("UKGT hierarchy-leaf accounting does not balance")
        if (self.excluded_ambiguous_code_groups == 0) != (self.excluded_ambiguous_code_rows == 0):
            raise ValueError("UKGT ambiguous-code accounting is inconsistent")
        if self.normalized_description_rows > self.source_rows:
            raise ValueError("UKGT description-normalization count is impossible")
        return self


class UkGlobalTariffReceipt(BaseModel):
    """Reproducible source and compiled-content proof."""

    model_config = _STRICT

    schema_version: Literal[1]
    parser_contract: Literal["ukgt_commodities_report_csvw_v1"]
    version: Version
    snapshot_date: date
    metadata_url: HttpsUrl
    metadata_bytes: Annotated[int, Field(gt=0)]
    metadata_sha256: Sha256
    report_url: HttpsUrl
    report_bytes: Annotated[int, Field(gt=0)]
    report_sha256: Sha256
    license_name: Literal["Open Government Licence v3.0"]
    license_url: Literal[
        "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
    ]
    attribution: Literal[
        "Contains public sector information from the UK Global Tariff, licensed under the "
        "Open Government Licence v3.0."
    ]
    license_caveat: Literal[
        "The Open Government Licence excludes third-party rights; UKGT descriptions are not "
        "redistributed as independently licensed WCO nomenclature."
    ]
    description_authority: Literal[
        "UK Global Tariff implementation text; not asserted as canonical WCO nomenclature"
    ]
    description_normalization: Literal["unicode_preserving_whitespace_collapse_v1"]
    audit: UkGlobalTariffAudit
    global_hs6_codes_sha256: Sha256
    registry_content_hash_contract: Literal["canonical_jsonl_global_then_gb_v1"]
    registry_content_sha256: Sha256


@dataclass(frozen=True, slots=True)
class _CommodityRow:
    row_id: int
    sid: str
    code: str
    suffix: str
    source_description: str
    description: str
    valid_from: date
    valid_to: date | None
    parent_sid: str | None
    parent_code: str | None
    parent_suffix: str | None


class UkGlobalTariffRegistry:
    """Immutable, queryable global-HS and UK-extension registry."""

    def __init__(
        self,
        *,
        global_subheadings: Sequence[HsGlobalSubheading],
        uk_commodities: Sequence[UkTariffCommodity],
        receipt: UkGlobalTariffReceipt,
    ) -> None:
        global_by_code = {row.code: row for row in global_subheadings}
        uk_by_code = {row.code: row for row in uk_commodities}
        if len(global_by_code) != len(global_subheadings):
            raise HsRegistryError("compiled global HS6 identities are duplicated")
        if len(uk_by_code) != len(uk_commodities):
            raise HsRegistryError("compiled UK tariff identities are duplicated")
        if tuple(global_by_code) != tuple(sorted(global_by_code)):
            raise HsRegistryError("compiled global HS6 identities are not sorted")
        if tuple(uk_by_code) != tuple(sorted(uk_by_code)):
            raise HsRegistryError("compiled UK tariff identities are not sorted")
        by_hs6: dict[str, list[UkTariffCommodity]] = defaultdict(list)
        for row in uk_commodities:
            if row.hs6 not in global_by_code:
                raise HsRegistryError(f"UK tariff code has no global HS6 identity: {row.code}")
            by_hs6[row.hs6].append(row)
        self._global = MappingProxyType(global_by_code)
        self._uk = MappingProxyType(uk_by_code)
        self._uk_by_hs6 = MappingProxyType(
            {code: tuple(rows) for code, rows in sorted(by_hs6.items())}
        )
        self._receipt = receipt

    @property
    def receipt(self) -> UkGlobalTariffReceipt:
        return self._receipt

    @property
    def global_codes(self) -> tuple[str, ...]:
        return tuple(self._global)

    @property
    def uk_codes(self) -> tuple[str, ...]:
        return tuple(self._uk)

    def require_global(self, code: str, *, on_date: date) -> HsGlobalSubheading:
        if re.fullmatch(r"[0-9]{6}", code) is None:
            raise HsRegistryError("global HS identity must contain exactly six digits")
        try:
            row = self._global[code]
        except KeyError as error:
            raise HsRegistryError(f"unknown global HS6 identity: {code}") from error
        if not row.valid_from <= on_date <= self._receipt.snapshot_date:
            raise HsRegistryError(
                f"global HS6 {code} is not proven for date {on_date.isoformat()} by this snapshot"
            )
        return row

    def require_uk(self, code: str, *, on_date: date) -> UkTariffCommodity:
        if re.fullmatch(r"[0-9]{10}", code) is None:
            raise HsRegistryError("UK tariff identity must contain exactly ten digits")
        self.require_global(code[:6], on_date=on_date)
        try:
            row = self._uk[code]
        except KeyError as error:
            raise HsRegistryError(f"unknown or ambiguous UK tariff identity: {code}") from error
        if (
            on_date > self._receipt.snapshot_date
            or on_date < row.valid_from
            or (row.valid_to is not None and on_date >= row.valid_to)
        ):
            raise HsRegistryError(
                f"UK tariff code {code} is not proven for date "
                f"{on_date.isoformat()} by this snapshot"
            )
        return row

    def uk_candidates(self, hs6: str, *, on_date: date) -> tuple[UkTariffCommodity, ...]:
        self.require_global(hs6, on_date=on_date)
        return tuple(
            row
            for row in self._uk_by_hs6.get(hs6, ())
            if row.valid_from <= on_date
            and (row.valid_to is None or on_date < row.valid_to)
            and on_date <= self._receipt.snapshot_date
        )

    def global_description_path(self, hs6: str) -> tuple[str, ...]:
        """Prove the heading-to-HS6 wording, including intermediate qualifiers.

        A leaf such as 'Other' is not a standalone commodity definition. National
        descendants carry the SID-proven ancestry; stop at the global node so
        no national restriction is accidentally promoted into the HS identity.
        """
        self.require_global(hs6, on_date=self.receipt.snapshot_date)
        paths = set()
        for row in self._uk_by_hs6.get(hs6, ()):
            nodes = row.description_path
            starts = [
                i for i, n in enumerate(nodes) if n.code == hs6[:4] + "000000" and n.suffix == "80"
            ]
            ends = [i for i, n in enumerate(nodes) if n.code == hs6 + "0000" and n.suffix == "80"]
            if len(starts) != 1 or len(ends) != 1 or starts[0] > ends[0]:
                raise HsRegistryError("HS description ancestry is not uniquely resolved: " + hs6)
            paths.add(tuple(n.description for n in nodes[starts[0] : ends[0] + 1]))
        if len(paths) != 1:
            raise HsRegistryError("HS description lacks a unique complete ancestry: " + hs6)
        return paths.pop()


def render_hs_code_surface(
    semantic_code: str,
    *,
    groups: Sequence[int],
    separator: str,
) -> HsCodeSurface:
    """Format exact digits with an explicit grouping contract.

    There is intentionally no inferred or default grouping.  The caller must
    supply a grouping whose sum equals the code length.
    """

    if re.fullmatch(r"[0-9]{6,18}", semantic_code) is None:
        raise ValueError("semantic HS code must contain 6-18 digits")
    if not groups or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in groups
    ):
        raise ValueError("HS surface groups must be positive integers")
    if sum(groups) != len(semantic_code):
        raise ValueError("HS surface groups must consume every semantic digit exactly once")
    if separator == "":
        if len(groups) != 1:
            raise ValueError("an empty HS surface separator requires exactly one group")
    elif any(character.isdigit() for character in separator) or (
        _SURFACE_PATTERN.fullmatch(separator) is None
    ):
        raise ValueError("HS surface separator must contain only supported non-digit characters")
    offset = 0
    parts: list[str] = []
    for width in groups:
        parts.append(semantic_code[offset : offset + width])
        offset += width
    return HsCodeSurface(semantic_code=semantic_code, surface=separator.join(parts))


def sample_global_hs6(
    *,
    registry: UkGlobalTariffRegistry,
    candidate_weights: Mapping[str, int],
    on_date: date,
    stream: DeterministicStream,
) -> HsGlobalSubheading:
    """Sample one global HS6 from an explicit, registry-validated distribution.

    The registry deliberately does not invent a default empirical or uniform
    distribution.  A fit-partition builder must provide the candidate weights,
    making distribution provenance a required input rather than hidden policy.
    """

    ordered = _validated_weighted_candidates(
        candidate_weights,
        expected_digits=6,
        label="global HS6",
    )
    identities = tuple(registry.require_global(code, on_date=on_date) for code, _ in ordered)
    selected = _weighted_index(
        tuple(weight for _, weight in ordered),
        stream=stream.derive("global-hs6"),
    )
    return identities[selected]


def sample_uk_tariff_code(
    *,
    registry: UkGlobalTariffRegistry,
    hs6: str,
    candidate_weights: Mapping[str, int],
    on_date: date,
    stream: DeterministicStream,
) -> UkTariffCommodity:
    """Sample one exact GB tariff leaf under an already selected global HS6.

    National extensions are jurisdiction-specific.  This function therefore
    exposes ``GB`` in its name and result type, requires the parent HS6, and
    rejects every candidate that is not an exact ten-digit UKGT leaf under that
    parent.  It never derives a suffix arithmetically.
    """

    global_identity = registry.require_global(hs6, on_date=on_date)
    ordered = _validated_weighted_candidates(
        candidate_weights,
        expected_digits=10,
        label="GB tariff",
    )
    identities: list[UkTariffCommodity] = []
    for code, _ in ordered:
        identity = registry.require_uk(code, on_date=on_date)
        if identity.hs6 != global_identity.code:
            raise HsRegistryError(f"GB tariff candidate {code} does not extend selected HS6 {hs6}")
        identities.append(identity)
    selected = _weighted_index(
        tuple(weight for _, weight in ordered),
        stream=stream.derive("gb-tariff"),
    )
    return identities[selected]


def _validated_weighted_candidates(
    candidate_weights: Mapping[str, int],
    *,
    expected_digits: int,
    label: str,
) -> tuple[tuple[str, int], ...]:
    if not candidate_weights:
        raise HsRegistryError(f"{label} sampling requires at least one candidate")
    ordered = tuple(sorted(candidate_weights.items()))
    for code, weight in ordered:
        if re.fullmatch(rf"[0-9]{{{expected_digits}}}", code) is None:
            raise HsRegistryError(
                f"{label} candidate must contain exactly {expected_digits} digits: {code!r}"
            )
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            raise HsRegistryError(f"{label} weights must be positive integers")
    return ordered


def _weighted_index(weights: Sequence[int], *, stream: DeterministicStream) -> int:
    total = sum(weights)
    draw = stream.randbelow(total)
    cumulative = 0
    for index, weight in enumerate(weights):
        cumulative += weight
        if draw < cumulative:
            return index
    raise AssertionError("validated weighted sampling failed to select a candidate")


def _code_set_sha256(codes: Sequence[str]) -> str:
    ordered = tuple(codes)
    if ordered != tuple(sorted(set(ordered))) or any(
        re.fullmatch(r"[0-9]{6}", code) is None for code in ordered
    ):
        raise HsRegistryError("global HS6 code set must be unique, sorted, and six-digit")
    return hashlib.sha256("".join(f"{code}\n" for code in ordered).encode()).hexdigest()


def _registry_content_sha256(
    global_rows: Sequence[HsGlobalSubheading],
    uk_rows: Sequence[UkTariffCommodity],
) -> str:
    digest = hashlib.sha256()
    for kind, rows in (("globalHs6", global_rows), ("ukTariffCode", uk_rows)):
        for row in rows:
            digest.update(
                canonical_json_bytes({"kind": kind, "record": row.model_dump(mode="json")})
            )
            digest.update(b"\n")
    return digest.hexdigest()


def load_ukgt_source_pin(path: Path) -> UkGlobalTariffSourcePin:
    """Load a strict source manifest; source bytes remain independently pinned."""

    payload = _read_regular_file(path, max_bytes=_MAX_METADATA_BYTES, label="UKGT source pin")
    try:
        pin = UkGlobalTariffSourcePin.model_validate_json(payload, strict=True)
    except ValueError as error:
        raise HsRegistryError(f"UKGT source pin is invalid: {error}") from error
    canonical = canonical_json_bytes(pin.model_dump(mode="json")) + b"\n"
    if payload != canonical:
        raise HsRegistryError("UKGT source pin must be canonical JSON with one trailing newline")
    return pin


def compile_uk_global_tariff_registry(
    *,
    metadata_path: Path,
    report_path: Path,
    source: UkGlobalTariffSourcePin,
) -> UkGlobalTariffRegistry:
    """Compile an exact UKGT snapshot into global and national identities."""

    metadata_payload = _read_pinned_file(
        metadata_path,
        expected_bytes=source.metadata_bytes,
        expected_sha256=source.metadata_sha256,
        max_bytes=_MAX_METADATA_BYTES,
        label="UKGT CSVW metadata",
    )
    _validate_csvw_metadata(metadata_payload)
    report_payload = _read_pinned_file(
        report_path,
        expected_bytes=source.report_bytes,
        expected_sha256=source.report_sha256,
        max_bytes=_MAX_REPORT_BYTES,
        label="UKGT commodities report",
    )
    rows, normalized_descriptions = _parse_report(
        report_payload, snapshot_date=source.snapshot_date
    )
    if len(rows) != source.expected_source_rows:
        raise HsRegistryError(
            "UKGT source row count mismatch: "
            f"expected {source.expected_source_rows}, found {len(rows)}"
        )
    by_sid = _validate_hierarchy(rows)
    global_rows = _compile_global_hs6(rows, by_sid=by_sid)
    if len(global_rows) != source.expected_global_hs6_records:
        raise HsRegistryError(
            "global HS6 count mismatch: "
            f"expected {source.expected_global_hs6_records}, found {len(global_rows)}"
        )
    global_codes_sha256 = _code_set_sha256(tuple(row.code for row in global_rows))
    if global_codes_sha256 != source.expected_global_hs6_codes_sha256:
        raise HsRegistryError(
            "global HS6 code-set SHA-256 mismatch: "
            f"expected {source.expected_global_hs6_codes_sha256}, "
            f"found {global_codes_sha256}"
        )
    uk_rows, leaf_counts = _compile_uk_codes(rows, by_sid=by_sid, global_rows=global_rows)
    if len(uk_rows) != source.expected_uk_tariff_records:
        raise HsRegistryError(
            "UK tariff code count mismatch: "
            f"expected {source.expected_uk_tariff_records}, found {len(uk_rows)}"
        )

    chapters = {row.chapter_code for row in global_rows}
    headings = {row.heading_code for row in global_rows}
    audit = UkGlobalTariffAudit(
        source_rows=len(rows),
        normalized_description_rows=normalized_descriptions,
        global_chapters=len(chapters),
        global_headings=len(headings),
        global_hs6_records=len(global_rows),
        hierarchy_leaf_rows=leaf_counts["leaf_rows"],
        excluded_nondeclarable_suffix_leaf_rows=leaf_counts["nondeclarable"],
        excluded_ambiguous_code_groups=leaf_counts["ambiguous_groups"],
        excluded_ambiguous_code_rows=leaf_counts["ambiguous_rows"],
        excluded_special_chapter_leaf_rows=leaf_counts["special_chapter"],
        uk_tariff_records=len(uk_rows),
    )
    registry_content_sha256 = _registry_content_sha256(global_rows, uk_rows)
    receipt = UkGlobalTariffReceipt(
        schema_version=1,
        parser_contract=_PARSER_CONTRACT,
        version=source.version,
        snapshot_date=source.snapshot_date,
        metadata_url=source.metadata_url,
        metadata_bytes=source.metadata_bytes,
        metadata_sha256=source.metadata_sha256,
        report_url=source.report_url,
        report_bytes=source.report_bytes,
        report_sha256=source.report_sha256,
        license_name=source.license_name,
        license_url=source.license_url,
        attribution=source.attribution,
        license_caveat=source.license_caveat,
        description_authority=_DESCRIPTION_AUTHORITY,
        description_normalization=_DESCRIPTION_NORMALIZATION,
        audit=audit,
        global_hs6_codes_sha256=global_codes_sha256,
        registry_content_hash_contract="canonical_jsonl_global_then_gb_v1",
        registry_content_sha256=registry_content_sha256,
    )
    return UkGlobalTariffRegistry(
        global_subheadings=global_rows,
        uk_commodities=uk_rows,
        receipt=receipt,
    )


def _read_regular_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    if path.is_symlink():
        raise HsRegistryError(f"{label} must not be a symbolic link: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise HsRegistryError(f"{label} is not readable: {path}") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise HsRegistryError(f"{label} is not a regular file: {path}")
        if info.st_size > max_bytes:
            raise HsRegistryError(f"{label} exceeds the bounded input size: {info.st_size}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _read_pinned_file(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    max_bytes: int,
    label: str,
) -> bytes:
    payload = _read_regular_file(path, max_bytes=max_bytes, label=label)
    if len(payload) != expected_bytes:
        raise HsRegistryError(
            f"{label} size mismatch: expected {expected_bytes}, found {len(payload)}"
        )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise HsRegistryError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )
    return payload


def _validate_csvw_metadata(payload: bytes) -> None:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HsRegistryError("UKGT CSVW metadata is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise HsRegistryError("UKGT CSVW metadata root must be an object")
    if value.get("dc:title") != _DATASET_TITLE:
        raise HsRegistryError("UKGT CSVW metadata has an unexpected dataset title")
    if value.get("dc:creator") != _DATASET_CREATOR:
        raise HsRegistryError("UKGT CSVW metadata has an unexpected dataset creator")
    if value.get("dc:license") != _LICENSE_URL:
        raise HsRegistryError("UKGT CSVW metadata does not declare OGL v3.0")
    tables = value.get("tables")
    if not isinstance(tables, list):
        raise HsRegistryError("UKGT CSVW metadata tables must be an array")
    matches = [row for row in tables if isinstance(row, dict) and row.get("url") == _TABLE_PATH]
    if len(matches) != 1:
        raise HsRegistryError("UKGT CSVW metadata does not identify one commodities report")
    schema = cast(dict[str, object], matches[0]).get("tableSchema")
    columns = schema.get("columns") if isinstance(schema, dict) else None
    names = tuple(
        row.get("name") if isinstance(row, dict) else None
        for row in (columns if isinstance(columns, list) else [])
    )
    if names != _SOURCE_COLUMNS:
        raise HsRegistryError("UKGT CSVW commodities-report column contract changed")


def _normalized_description(value: str, *, row_number: int) -> tuple[str, bool]:
    unsafe_controls = {
        character
        for character in value
        if unicodedata.category(character) == "Cc" and character not in {"\t", "\n", "\r"}
    }
    if unsafe_controls:
        raise HsRegistryError(f"UKGT row {row_number} has an unsafe commodity description")
    normalized = " ".join(value.split())
    if not normalized:
        raise HsRegistryError(f"UKGT row {row_number} has an empty commodity description")
    return normalized, normalized != value


def _parse_date(value: str, *, field: str, row_number: int) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise HsRegistryError(f"UKGT row {row_number} has invalid {field}: {value!r}") from error


def _parse_report(payload: bytes, *, snapshot_date: date) -> tuple[tuple[_CommodityRow, ...], int]:
    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise HsRegistryError("UKGT commodities report is not valid UTF-8 CSV") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if tuple(reader.fieldnames or ()) != _SOURCE_COLUMNS:
        raise HsRegistryError("UKGT commodities report header contract changed")
    rows: list[_CommodityRow] = []
    normalized_descriptions = 0
    try:
        for row_number, raw in enumerate(reader, start=2):
            if set(raw) != set(_SOURCE_COLUMNS) or any(value is None for value in raw.values()):
                raise HsRegistryError(f"UKGT row {row_number} does not match the CSV contract")
            row_id = raw["id"]
            sid = raw["commodity__sid"]
            code = raw["commodity__code"]
            suffix = raw["commodity__suffix"]
            if re.fullmatch(r"[1-9][0-9]*", row_id) is None:
                raise HsRegistryError(f"UKGT row {row_number} has invalid row identity")
            if re.fullmatch(r"[0-9]+", sid) is None:
                raise HsRegistryError(f"UKGT row {row_number} has invalid commodity SID")
            if re.fullmatch(r"[0-9]{10}", code) is None:
                raise HsRegistryError(f"UKGT row {row_number} has invalid ten-digit code")
            if re.fullmatch(r"[0-9]{2}", suffix) is None:
                raise HsRegistryError(f"UKGT row {row_number} has invalid commodity suffix")
            source_description = raw["commodity__description"]
            description, changed = _normalized_description(
                source_description, row_number=row_number
            )
            normalized_descriptions += int(changed)
            valid_from = _parse_date(
                raw["commodity__validity_start"], field="validity start", row_number=row_number
            )
            end_value = raw["commodity__validity_end"]
            valid_to = (
                None
                if end_value == "#NA"
                else _parse_date(end_value, field="validity end", row_number=row_number)
            )
            if valid_to is not None and valid_to <= valid_from:
                raise HsRegistryError(
                    f"UKGT row {row_number} has an empty or inverted validity interval"
                )
            if valid_from > snapshot_date or (valid_to is not None and valid_to <= snapshot_date):
                raise HsRegistryError(
                    f"UKGT row {row_number} is not active on pinned snapshot date"
                )
            parent_values = (
                raw["parent__sid"],
                raw["parent__code"],
                raw["parent__suffix"],
            )
            if parent_values == ("#NA", "#NA", "#NA"):
                parent_sid = parent_code = parent_suffix = None
            elif any(value == "#NA" for value in parent_values):
                raise HsRegistryError(f"UKGT row {row_number} has a partial parent identity")
            else:
                parent_sid, parent_code, parent_suffix = parent_values
                if re.fullmatch(r"[0-9]+", parent_sid) is None:
                    raise HsRegistryError(f"UKGT row {row_number} has invalid parent SID")
                if re.fullmatch(r"[0-9]{10}", parent_code) is None:
                    raise HsRegistryError(f"UKGT row {row_number} has invalid parent code")
                if re.fullmatch(r"[0-9]{2}", parent_suffix) is None:
                    raise HsRegistryError(f"UKGT row {row_number} has invalid parent suffix")
            rows.append(
                _CommodityRow(
                    row_id=int(row_id),
                    sid=sid,
                    code=code,
                    suffix=suffix,
                    source_description=source_description,
                    description=description,
                    valid_from=valid_from,
                    valid_to=valid_to,
                    parent_sid=parent_sid,
                    parent_code=parent_code,
                    parent_suffix=parent_suffix,
                )
            )
    except csv.Error as error:
        raise HsRegistryError("UKGT commodities report contains malformed CSV") from error
    if not rows:
        raise HsRegistryError("UKGT commodities report contains no data rows")
    return tuple(rows), normalized_descriptions


def _validate_hierarchy(rows: Sequence[_CommodityRow]) -> Mapping[str, _CommodityRow]:
    by_sid: dict[str, _CommodityRow] = {}
    row_ids: set[int] = set()
    for row in rows:
        if row.sid in by_sid:
            raise HsRegistryError(f"UKGT commodity SID is duplicated: {row.sid}")
        if row.row_id in row_ids:
            raise HsRegistryError(f"UKGT source row identity is duplicated: {row.row_id}")
        by_sid[row.sid] = row
        row_ids.add(row.row_id)
    for row in rows:
        if row.parent_sid is None:
            continue
        parent = by_sid.get(row.parent_sid)
        if parent is None:
            raise HsRegistryError(f"UKGT parent SID is absent: {row.sid} -> {row.parent_sid}")
        if (parent.code, parent.suffix) != (row.parent_code, row.parent_suffix):
            raise HsRegistryError(f"UKGT parent identity mismatch for SID {row.sid}")

    states: dict[str, int] = {}

    def visit(sid: str) -> None:
        state = states.get(sid, 0)
        if state == 1:
            raise HsRegistryError(f"UKGT hierarchy contains a cycle at SID {sid}")
        if state == 2:
            return
        states[sid] = 1
        parent_sid = by_sid[sid].parent_sid
        if parent_sid is not None:
            visit(parent_sid)
        states[sid] = 2

    for sid in by_sid:
        visit(sid)
    return MappingProxyType(by_sid)


def _unique_suffix80(rows: Sequence[_CommodityRow], *, code: str, label: str) -> _CommodityRow:
    matches = [row for row in rows if row.code == code and row.suffix == "80"]
    if len(matches) != 1:
        raise HsRegistryError(
            f"UKGT {label} {code} must have exactly one suffix-80 description row"
        )
    return matches[0]


def _unique_structural_rows(
    rows: Sequence[_CommodityRow],
    *,
    by_sid: Mapping[str, _CommodityRow],
    identity_length: int,
    label: str,
) -> dict[str, _CommodityRow]:
    grouped: dict[str, list[_CommodityRow]] = defaultdict(list)
    for row in rows:
        if row.suffix != "80" or row.code[identity_length:] != "0" * (10 - identity_length):
            continue
        if identity_length == 4 and row.code[2:] == "0" * 8:
            continue
        grouped[row.code[:identity_length]].append(row)
    selected: dict[str, _CommodityRow] = {}
    duplicated: dict[str, list[_CommodityRow]] = {}
    for code, values in grouped.items():
        depths = {row.sid: len(_ancestor_sids(row, by_sid)) for row in values}
        minimum = min(depths.values())
        shallowest = [row for row in values if depths[row.sid] == minimum]
        if len(shallowest) != 1:
            duplicated[code] = shallowest
        else:
            selected[code] = shallowest[0]
    if duplicated:
        summary = ", ".join(
            f"{code} ({len(values)} rows)" for code, values in sorted(duplicated.items())
        )
        raise HsRegistryError(f"UKGT {label} identities are ambiguous: {summary}")
    return dict(sorted(selected.items()))


def _ancestor_sids(row: _CommodityRow, by_sid: Mapping[str, _CommodityRow]) -> tuple[str, ...]:
    output: list[str] = []
    current: _CommodityRow | None = row
    while current is not None:
        output.append(current.sid)
        current = by_sid[current.parent_sid] if current.parent_sid is not None else None
    output.reverse()
    return tuple(output)


def _compile_global_hs6(
    rows: Sequence[_CommodityRow],
    *,
    by_sid: Mapping[str, _CommodityRow],
) -> tuple[HsGlobalSubheading, ...]:
    chapter_rows = _unique_structural_rows(
        rows,
        by_sid=by_sid,
        identity_length=2,
        label="chapter",
    )
    heading_rows = _unique_structural_rows(
        rows,
        by_sid=by_sid,
        identity_length=4,
        label="heading",
    )
    direct_codes = {
        row.code[:6]
        for row in rows
        if row.code[4:] != "0" * 6 and row.code[:2] not in _SPECIAL_NATIONAL_CHAPTERS
    }
    direct_by_heading: dict[str, set[str]] = defaultdict(set)
    for code in direct_codes:
        direct_by_heading[code[:4]].add(code)
    unsplit = {
        heading + "00"
        for heading in heading_rows
        if heading[:2] not in _SPECIAL_NATIONAL_CHAPTERS and not direct_by_heading[heading]
    }
    codes = tuple(sorted(direct_codes | unsplit))
    records: list[HsGlobalSubheading] = []
    for code in codes:
        chapter = chapter_rows.get(code[:2])
        heading = heading_rows.get(code[:4])
        if chapter is None or heading is None:
            raise HsRegistryError(f"global HS6 {code} lacks chapter or heading identity")
        subheading = _unique_suffix80(rows, code=code + "0000", label="HS6")
        ancestry = _ancestor_sids(subheading, by_sid)
        if chapter.sid not in ancestry or heading.sid not in ancestry:
            raise HsRegistryError(
                f"global HS6 {code} does not descend from its code-prefix chapter and heading"
            )
        if ancestry.index(chapter.sid) > ancestry.index(heading.sid):
            raise HsRegistryError(f"global HS6 {code} has an inverted chapter/heading ancestry")
        records.append(
            HsGlobalSubheading(
                edition="HS2022",
                valid_from=_HS2022_EFFECTIVE_FROM,
                chapter_code=code[:2],
                chapter_description=chapter.description,
                heading_code=code[:4],
                heading_description=heading.description,
                code=code,
                source_description=subheading.source_description,
                description=subheading.description,
                description_authority=_DESCRIPTION_AUTHORITY,
            )
        )
    return tuple(records)


def _description_path(
    row: _CommodityRow,
    by_sid: Mapping[str, _CommodityRow],
    description_nodes: Mapping[str, UkTariffDescriptionNode],
) -> tuple[UkTariffDescriptionNode, ...]:
    path: list[UkTariffDescriptionNode] = []
    current: _CommodityRow | None = row
    while current is not None:
        path.append(description_nodes[current.sid])
        current = by_sid[current.parent_sid] if current.parent_sid is not None else None
    path.reverse()
    return tuple(path)


def _compile_uk_codes(
    rows: Sequence[_CommodityRow],
    *,
    by_sid: Mapping[str, _CommodityRow],
    global_rows: Sequence[HsGlobalSubheading],
) -> tuple[tuple[UkTariffCommodity, ...], Counter[str]]:
    parent_sids = {row.parent_sid for row in rows if row.parent_sid is not None}
    leaves = [row for row in rows if row.sid not in parent_sids]
    counts: Counter[str] = Counter(leaf_rows=len(leaves))
    declarable: list[_CommodityRow] = []
    for row in leaves:
        if row.suffix != "80":
            counts["nondeclarable"] += 1
        else:
            declarable.append(row)
    by_code: dict[str, list[_CommodityRow]] = defaultdict(list)
    for row in declarable:
        by_code[row.code].append(row)
    ambiguous = {code: values for code, values in by_code.items() if len(values) != 1}
    counts["ambiguous_groups"] = len(ambiguous)
    counts["ambiguous_rows"] = sum(len(values) for values in ambiguous.values())
    global_codes = {row.code for row in global_rows}
    description_nodes = {
        row.sid: UkTariffDescriptionNode(
            source_sid=row.sid,
            code=row.code,
            suffix=row.suffix,
            source_description=row.source_description,
            description=row.description,
        )
        for row in rows
    }
    accepted: list[UkTariffCommodity] = []
    for code, values in sorted(by_code.items()):
        if code in ambiguous:
            continue
        row = values[0]
        if code[:2] in _SPECIAL_NATIONAL_CHAPTERS:
            counts["special_chapter"] += 1
            continue
        hs6 = code[:6]
        if hs6 not in global_codes:
            raise HsRegistryError(f"UK tariff leaf {code} lacks a global HS6 identity")
        accepted.append(
            UkTariffCommodity(
                jurisdiction="GB",
                source_sid=row.sid,
                code=code,
                hs6=hs6,
                source_description=row.source_description,
                description=row.description,
                description_path=_description_path(row, by_sid, description_nodes),
                description_authority=_DESCRIPTION_AUTHORITY,
                valid_from=row.valid_from,
                valid_to=row.valid_to,
            )
        )
    return tuple(accepted), counts
