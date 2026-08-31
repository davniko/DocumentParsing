"""Pinned ISO-3166 country identities and conservative corpus aliases.

Synthetic scenarios use ISO alpha-2 internally, while task labels retain a
printable country name.  This module is the only boundary between those two
representations.  It never guesses from substrings or place names: an input is
resolved only when its normalized alias has exactly one configured country.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file

CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_TOKEN = re.compile(r"[^A-Z0-9]+")
_Strict = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def normalize_country_alias(value: str) -> str:
    """Return the frozen NFKD/ASCII/alphanumeric country lookup key."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("country alias must be a non-empty string")
    ascii_value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").upper()
    )
    tokens = _TOKEN.sub(" ", ascii_value).split()
    normalized_tokens: list[str] = []
    single_letters: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            single_letters.append(token)
            continue
        if single_letters:
            normalized_tokens.append("".join(single_letters))
            single_letters.clear()
        normalized_tokens.append(token)
    if single_letters:
        normalized_tokens.append("".join(single_letters))
    normalized = " ".join(normalized_tokens)
    if not normalized:
        raise ValueError("country alias has no alphanumeric content")
    return normalized


class CountryEntry(BaseModel):
    model_config = _Strict

    alpha2: CountryCode
    alpha3: Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
    numeric: Annotated[str, StringConstraints(pattern=r"^[0-9]{3}$")]
    name: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    official_name: Annotated[str, StringConstraints(min_length=1, max_length=192)] | None = None
    common_name: Annotated[str, StringConstraints(min_length=1, max_length=192)] | None = None

    @model_validator(mode="after")
    def text_has_canonical_edges(self) -> CountryEntry:
        for field in ("name", "official_name", "common_name"):
            value = getattr(self, field)
            if value is not None and value != value.strip():
                raise ValueError(f"country {field} contains outer whitespace")
        return self


class CountryRegistryAudit(BaseModel):
    model_config = _Strict

    iso_records: Annotated[int, Field(gt=0)]
    observed_alias_records: Annotated[int, Field(ge=0)]
    distinct_aliases: Annotated[int, Field(gt=0)]
    iso_sha256: Sha256
    observed_aliases_sha256: Sha256
    normalization: str


class CountryRegistry:
    """Immutable country lookup with an explicit ambiguity rejection policy."""

    def __init__(
        self,
        *,
        entries: Iterable[CountryEntry],
        observed_aliases: Mapping[str, str],
        iso_sha256: str,
        observed_aliases_sha256: str,
    ) -> None:
        by_code: dict[str, CountryEntry] = {}
        aliases: dict[str, set[str]] = defaultdict(set)
        for entry in entries:
            if entry.alpha2 in by_code:
                raise ValueError(f"duplicate ISO alpha-2 country: {entry.alpha2}")
            by_code[entry.alpha2] = entry
            for value in (
                entry.alpha2,
                entry.alpha3,
                entry.name,
                entry.official_name,
                entry.common_name,
            ):
                if value is not None:
                    aliases[normalize_country_alias(value)].add(entry.alpha2)
        if not by_code:
            raise ValueError("country registry cannot be empty")
        for raw_alias, raw_code in observed_aliases.items():
            if raw_code not in by_code:
                raise ValueError(
                    f"observed country alias points to absent ISO code: {raw_alias!r} -> {raw_code}"
                )
            aliases[normalize_country_alias(raw_alias)].add(raw_code)
        ambiguous = {
            alias: tuple(sorted(codes)) for alias, codes in aliases.items() if len(codes) != 1
        }
        if ambiguous:
            preview = ", ".join(
                f"{alias}={codes}" for alias, codes in sorted(ambiguous.items())[:10]
            )
            raise ValueError(f"country aliases are ambiguous: {preview}")
        self._entries = MappingProxyType(dict(sorted(by_code.items())))
        self._aliases = MappingProxyType(
            {alias: next(iter(codes)) for alias, codes in sorted(aliases.items())}
        )
        self._audit = CountryRegistryAudit.model_validate(
            {
                "iso_records": len(by_code),
                "observed_alias_records": len(observed_aliases),
                "distinct_aliases": len(self._aliases),
                "iso_sha256": iso_sha256,
                "observed_aliases_sha256": observed_aliases_sha256,
                "normalization": "NFKD_ASCII_UPPER_ALPHANUMERIC_WORDS_V1",
            },
            strict=True,
        )

    @property
    def audit(self) -> CountryRegistryAudit:
        return self._audit

    @property
    def country_codes(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def entry(self, country_code: str) -> CountryEntry:
        try:
            return self._entries[country_code]
        except KeyError as error:
            raise KeyError(f"unknown ISO alpha-2 country: {country_code}") from error

    def resolve(self, value: str | None) -> str | None:
        if value is None:
            return None
        return self._aliases.get(normalize_country_alias(value))

    def require(self, value: str, *, field: str) -> str:
        resolved = self.resolve(value)
        if resolved is None:
            raise ValueError(f"unresolved country at {field}: {value!r}")
        return resolved

    def printable_name(self, country_code: str) -> str:
        return self.entry(country_code).name


def _regular_pinned_json(path: Path, *, expected_sha256: str, label: str) -> Any:
    if path.is_symlink() or not path.resolve(strict=True).is_file():
        raise ValueError(f"{label} must be a regular file")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual}")
    try:
        return json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error


def _load_iso_entries(*, iso_path: Path, iso_sha256: str) -> list[CountryEntry]:
    iso_value = _regular_pinned_json(
        iso_path, expected_sha256=iso_sha256, label="ISO-3166 snapshot"
    )
    if (
        not isinstance(iso_value, dict)
        or set(iso_value) != {"3166-1"}
        or not isinstance(iso_value["3166-1"], list)
    ):
        raise ValueError("ISO-3166 snapshot has an unexpected root contract")
    entries: list[CountryEntry] = []
    for index, raw in enumerate(iso_value["3166-1"]):
        if not isinstance(raw, dict):
            raise ValueError(f"ISO-3166 record {index} is not an object")
        allowed = {"alpha_2", "alpha_3", "numeric", "name", "official_name", "common_name", "flag"}
        unexpected = set(raw) - allowed
        if unexpected:
            raise ValueError(
                f"ISO-3166 record {index} contains unexpected fields: {sorted(unexpected)}"
            )
        entries.append(
            CountryEntry.model_validate(
                {
                    "alpha2": raw.get("alpha_2"),
                    "alpha3": raw.get("alpha_3"),
                    "numeric": raw.get("numeric"),
                    "name": raw.get("name"),
                    "official_name": raw.get("official_name"),
                    "common_name": raw.get("common_name"),
                },
                strict=True,
            )
        )

    return entries


def load_iso_country_registry(*, iso_path: Path, iso_sha256: str) -> CountryRegistry:
    """Load only official ISO identities; unresolved corpus surfaces stay unresolved."""

    empty_alias_sha256 = sha256_bytes(canonical_json_bytes({"observedMappings": {}}))
    return CountryRegistry(
        entries=_load_iso_entries(iso_path=iso_path, iso_sha256=iso_sha256),
        observed_aliases={},
        iso_sha256=iso_sha256,
        observed_aliases_sha256=empty_alias_sha256,
    )


def load_country_registry(
    *,
    iso_path: Path,
    iso_sha256: str,
    observed_aliases_path: Path,
    observed_aliases_sha256: str,
) -> CountryRegistry:
    """Load ISO identities plus a reviewed source-audit alias artifact.

    This loader remains available for corpus auditing. Synthetic country
    sampling uses :func:`load_iso_country_registry` and never samples aliases.
    """

    alias_value = _regular_pinned_json(
        observed_aliases_path,
        expected_sha256=observed_aliases_sha256,
        label="observed country aliases",
    )
    base_keys = {
        "iso3166SnapshotPath",
        "iso3166SnapshotSha256",
        "normalization",
        "observedMappings",
    }
    provenance_keys = {
        "sourceArtifact",
        "sourceArtifactSha256",
        "reviewedAliasPolicy",
    }
    if (
        not isinstance(alias_value, dict)
        or set(alias_value) not in {frozenset(base_keys), frozenset(base_keys | provenance_keys)}
        or not isinstance(alias_value.get("observedMappings"), dict)
    ):
        raise ValueError("observed country alias artifact has an unexpected contract")
    if alias_value.get("iso3166SnapshotSha256") != iso_sha256:
        raise ValueError("observed country aliases reference a different ISO-3166 snapshot")
    if set(alias_value) == base_keys | provenance_keys and (
        not isinstance(alias_value["sourceArtifact"], str)
        or not alias_value["sourceArtifact"]
        or not isinstance(alias_value["sourceArtifactSha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", alias_value["sourceArtifactSha256"]) is None
        or alias_value["reviewedAliasPolicy"] != "exact_observed_surface_to_iso_alpha2_v1"
    ):
        raise ValueError("reviewed country alias provenance is invalid")
    mappings = alias_value["observedMappings"]
    if any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in mappings.items()
    ):
        raise ValueError("observed country mappings must contain string keys and values")
    return CountryRegistry(
        entries=_load_iso_entries(iso_path=iso_path, iso_sha256=iso_sha256),
        observed_aliases=mappings,
        iso_sha256=iso_sha256,
        observed_aliases_sha256=observed_aliases_sha256,
    )
