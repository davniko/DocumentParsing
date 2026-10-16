"""Pinned country identities, input aliases and explicit rendering distributions.

Country identity recognition preserves punctuation. Input-only corpus aliases
never become sampling options; document-level alias opportunities and physical
source-frame constraints are supplied by the rendering owner.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_REGISTRY_MANIFEST = (
    "artifacts/registries/compiled/country-presentations-iso-cldr-48.2.3-v1/"
    "production-manifest.json"
)
DEFAULT_MANIFEST_SHA256 = "aa8f579b7a5855a47b11dca35b27c38198f07d090d801b6c1fe86917c4b854b4"
LANGUAGES = ("en", "fr", "de", "es", "it", "pt", "tr")
FAMILIES = ("name", "iso2", "iso3", "abbreviation", "dotted_code", "localized_name")
_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_Weight = Annotated[float, Field(ge=0)]
_CODE = r"[A-Za-z](?:\.?[A-Za-z]){1,2}\.?"
_COMPOUNDS = (
    re.compile(rf"(.+?)\s*(?:-\s*)?\(\s*({_CODE})\s*\)", re.DOTALL),
    re.compile(rf"(.+?)\s*-\s*({_CODE})", re.DOTALL),
)


class CountryFamilyWeights(BaseModel):
    model_config = _CONFIG

    name: _Weight = 1.0
    iso2: _Weight = 1.0
    iso3: _Weight = 1.0
    abbreviation: _Weight = 1.0
    dotted_code: _Weight = 1.0
    localized_name: _Weight = 1.0

    @model_validator(mode="after")
    def positive_mass(self) -> CountryFamilyWeights:
        if not any(self.model_dump().values()):
            raise ValueError("country family weights require positive mass")
        return self


class CountryLanguageContext(BaseModel):
    """Reviewed literal issuer evidence; the renderer must prove it stays fixed."""

    model_config = _CONFIG

    source_id: Annotated[str, Field(pattern=r"^doc_[0-9a-f]{64}$")]
    literal: Annotated[str, Field(min_length=1)]
    country_code: Annotated[str, Field(pattern=r"^[A-Z]{2}$")]

    @model_validator(mode="after")
    def literal_has_exact_edges(self) -> CountryLanguageContext:
        if self.literal != self.literal.strip():
            raise ValueError("reviewed country context literal must have nonblank exact edges")
        return self


class CountryPresentationConfig(BaseModel):
    model_config = _CONFIG

    alias_probability: Annotated[float, Field(ge=0, le=1)] = 0.30
    family_weights: CountryFamilyWeights = Field(default_factory=CountryFamilyWeights)
    language_allowlist: tuple[str, ...] = ("en", "it", "fr", "es")
    unavailable_family_policy: Literal["error", "renormalize_eligible"] = "renormalize_eligible"
    reviewed_contexts: tuple[CountryLanguageContext, ...] = ()

    @model_validator(mode="after")
    def supported_languages(self) -> CountryPresentationConfig:
        if (
            not self.language_allowlist
            or "en" not in self.language_allowlist
            or len(set(self.language_allowlist)) != len(self.language_allowlist)
            or set(self.language_allowlist) - set(LANGUAGES)
        ):
            raise ValueError(
                "country language allowlist must contain unique supported languages and en"
            )
        owners = [(context.source_id, context.literal) for context in self.reviewed_contexts]
        if len(set(owners)) != len(owners):
            raise ValueError("reviewed country context literals require unique source ownership")
        return self


def _key(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("country presentation must be nonempty text")
    return " ".join(unicodedata.normalize("NFKC", value).upper().split())


def _atomic_key(value: str) -> str:
    return re.sub(r"\s*([()])\s*", r"\1", _key(value))


def _literal_pattern(surfaces: Iterable[str]) -> re.Pattern[str]:
    return re.compile(
        r"(?<!\w)(?:"
        + "|".join(
            re.escape(surface).replace(r"\ ", r"[ \t]+")
            for surface in sorted(surfaces, key=lambda s: (-len(s), s))
        )
        + r")(?!\w)",
        re.IGNORECASE,
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class CountryPresentation:
    code: str
    surface: str
    family: str
    languages: tuple[str, ...]
    provenance: tuple[Mapping[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "surface": self.surface,
            "family": self.family,
            "languages": list(self.languages),
            "provenance": _thaw(self.provenance),
        }


@dataclass(frozen=True, slots=True, init=False)
class Registry:
    """Immutable, cached snapshot; changed assets require a new explicit pin/load."""

    path: Path
    codes: tuple[str, ...]
    manifest_sha256: str
    _names: Mapping[str, str]
    _lookup: Mapping[str, str]
    _atomic_names: Mapping[str, str]
    _ambiguous: frozenset[str]
    _options: Mapping[tuple[str, str], tuple[CountryPresentation, ...]]
    _patterns: Mapping[str, re.Pattern[str]]
    _blocked: re.Pattern[str]
    _terminal_named: re.Pattern[str]
    _territories: Mapping[str, Any]

    def __init__(
        self,
        directory: str | Path,
        *,
        expected_manifest_sha256: str = DEFAULT_MANIFEST_SHA256,
        manifest_name: str = "production-manifest.json",
    ) -> None:
        path = Path(directory).resolve()
        manifest_bytes = (path / manifest_name).read_bytes()
        actual = hashlib.sha256(manifest_bytes).hexdigest()
        if actual != expected_manifest_sha256:
            raise ValueError("country registry manifest hash mismatch")
        manifest = json.loads(manifest_bytes)
        expected_files = {
            "presentations.json",
            "preferred-names.json",
            "territory-languages.json",
            "input-aliases.json",
            "source-provenance.json",
            "LICENSE",
        }
        if (
            manifest.get("version") != 1
            or manifest.get("contract") != "country-presentations-v1"
            or set(manifest["files"]) != expected_files
        ):
            raise ValueError("unsupported country registry manifest contract")
        data = {}
        for filename, pin in manifest["files"].items():
            raw = (path / filename).read_bytes()
            if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
                raise ValueError(f"country registry asset hash mismatch: {filename}")
            if filename.endswith(".json"):
                data[filename] = json.loads(raw)
        vocabulary = data["presentations.json"]
        codes = tuple(sorted(row["code"] for row in vocabulary["countries"]))
        names = data["preferred-names.json"]["names"]
        territories = data["territory-languages.json"]["territories"]
        if (
            len(codes) != 249
            or len(set(codes)) != 249
            or set(names) != set(codes)
            or set(territories) != set(codes)
            or len(set(names.values())) != 249
            or len(vocabulary["presentations"]) != manifest["presentationCount"]
            or len(data["input-aliases.json"]["aliases"]) != manifest["inputAliasCount"]
        ):
            raise ValueError("country registry coverage/count mismatch")
        ambiguous = frozenset(
            _key(row["surface"])
            for row in vocabulary["ambiguous"] + data["input-aliases.json"]["ambiguous"]
        )
        lookup: dict[str, str] = {}
        atomic_names: dict[str, str] = {}
        named_surfaces: set[str] = set()
        surfaces: dict[str, set[str]] = defaultdict(set)

        def recognize(surface: str, code: str) -> None:
            normalized = _key(surface)
            if code not in names or normalized in ambiguous:
                raise ValueError(f"invalid country alias authority: {surface}")
            if normalized in lookup and lookup[normalized] != code:
                raise ValueError(f"country alias collision: {surface}")
            lookup[normalized] = code
            surfaces[code].add(normalized)

        by_code: dict[str, list[CountryPresentation]] = defaultdict(list)
        for row in vocabulary["presentations"]:
            recognize(row["surface"], row["code"])
            family = row["family"]
            if family == "name":
                atomic_names[_atomic_key(row["surface"])] = row["code"]
                named_surfaces.add(row["surface"])
            if family == "name" and not {"en", "und"}.intersection(row["languages"]):
                family = "localized_name"
            by_code[row["code"]].append(
                CountryPresentation(
                    row["code"],
                    row["surface"],
                    family,
                    tuple(row["languages"]),
                    tuple(_freeze(p) for p in row["provenance"]),
                )
            )
        for code, name in names.items():
            recognize(name, code)
            atomic_names[_atomic_key(name)] = code
            named_surfaces.add(name)
        for row in data["input-aliases.json"]["aliases"]:
            recognize(row["surface"], row["code"])
            if row["family"] == "name":
                named_surfaces.add(row["surface"])
            elif row["family"] not in {"code", "abbreviation", "composite"}:
                raise ValueError(f"unsupported input country alias family: {row['family']}")
        options = {}
        for code in codes:
            for language in LANGUAGES:
                options[code, language] = tuple(
                    row
                    for row in by_code[code]
                    if {"en", "und", language}.intersection(row.languages)
                )
        patterns = {code: _literal_pattern(values) for code, values in surfaces.items()}
        # Protect longer foreign identities, e.g. CH inside C.H.N and CONGO
        # inside DEMOCRATIC REPUBLIC OF CONGO, without a global document scan.
        foreign_surfaces: dict[str, set[str]] = defaultdict(set)
        for surface, identity in lookup.items():
            tokens = tuple(re.finditer(r"\w+", surface))
            for index, first in enumerate(tokens):
                for last in tokens[index:]:
                    fragment = surface[first.start() : last.end()]
                    foreign = lookup.get(fragment)
                    if foreign is not None and foreign != identity:
                        foreign_surfaces[foreign].add(surface)
        for code, values in foreign_surfaces.items():
            foreign = _literal_pattern(values)
            patterns[code] = re.compile(
                rf"(?P<foreign>{foreign.pattern})|{patterns[code].pattern}",
                re.IGNORECASE,
            )
        blocked = re.compile(
            r"(?<!\w)(?:"
            + "|".join(re.escape(s) for s in sorted(ambiguous, key=len, reverse=True))
            + r")(?!\w)"
            if ambiguous
            else r"(?!)",
            re.IGNORECASE,
        )
        terminal_named = re.compile(
            r"(?<!\w)(?P<name>"
            + "|".join(
                re.escape(surface).replace(r"\ ", r"[ \t]+").replace(r"[ \t]+\(", r"[ \t]*\(")
                for surface in sorted(named_surfaces, key=lambda s: (-len(s), s))
            )
            + r")(?!\w)[ \t,.;]*\s*\Z",
            re.IGNORECASE,
        )
        for name, value in {
            "path": path,
            "codes": codes,
            "manifest_sha256": actual,
            "_names": MappingProxyType(names),
            "_lookup": MappingProxyType(lookup),
            "_atomic_names": MappingProxyType(atomic_names),
            "_ambiguous": ambiguous,
            "_options": MappingProxyType(options),
            "_patterns": MappingProxyType(patterns),
            "_blocked": blocked,
            "_terminal_named": terminal_named,
            "_territories": _freeze(territories),
        }.items():
            object.__setattr__(self, name, value)

    @classmethod
    def load(
        cls,
        project_root: str | Path,
        *,
        manifest_path: str | Path = DEFAULT_REGISTRY_MANIFEST,
        expected_sha256: str = DEFAULT_MANIFEST_SHA256,
    ) -> Registry:
        manifest = (Path(project_root) / manifest_path).resolve()
        return _load(manifest, expected_sha256)

    def _check(self, code: str, language: str = "en") -> None:
        if code not in self._names:
            raise ValueError(f"unknown ISO alpha-2 country: {code}")
        if language not in LANGUAGES:
            raise ValueError(f"unsupported country presentation language: {language}")

    def preferred_name(self, code: str) -> str:
        self._check(code)
        return self._names[code]

    def resolve(self, surface: str) -> str:
        normalized = _key(surface)
        if normalized in self._ambiguous:
            raise ValueError(f"ambiguous country presentation: {surface}")
        if code := self._lookup.get(normalized):
            return code
        if code := self._atomic_names.get(_atomic_key(normalized)):
            return code
        for pattern in _COMPOUNDS:
            if match := pattern.fullmatch(surface.strip()):
                name_code, suffix_code = self.resolve(match[1]), self.resolve(match[2])
                if name_code != suffix_code:
                    raise ValueError(f"contradictory country name/code: {surface}")
                return name_code
        raise ValueError(f"unknown country presentation: {surface}")

    def require(self, surface: str, *, field: str = "country") -> str:
        try:
            return self.resolve(surface)
        except ValueError as error:
            raise ValueError(f"{field}: {error}") from error

    def normalized(self, surface: str) -> str:
        return self.preferred_name(self.resolve(surface))

    def is_atomic_name(self, surface: str, code: str | None = None) -> bool:
        if code is not None:
            self._check(code)
        identity = self._atomic_names.get(_atomic_key(surface))
        return identity is not None and (code is None or code == identity)

    def terminal_named_country(self, text: str) -> tuple[int, int, str, str] | None:
        """Return a named terminal country clause; the caller owns role/locality proof."""
        if match := self._terminal_named.search(text):
            surface = match["name"]
            return match.start("name"), match.end("name"), surface, self.resolve(surface)
        return None

    def options(
        self,
        code: str,
        family: str | None = None,
        language: str = "en",
        *,
        exclude_preferred: bool = False,
    ) -> tuple[CountryPresentation, ...]:
        self._check(code, language)
        if family is not None and family not in (*FAMILIES, "composite"):
            raise ValueError(f"unsupported country presentation family: {family}")
        return tuple(
            row
            for row in self._options[code, language]
            if (family is None or row.family == family)
            and (not exclude_preferred or row.surface != self._names[code])
        )

    def eligible_languages(
        self,
        context_country_codes: Iterable[str],
        *,
        enabled: tuple[str, ...] = ("en", "it", "fr", "es"),
    ) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
        if (
            not enabled
            or "en" not in enabled
            or len(set(enabled)) != len(enabled)
            or set(enabled) - set(LANGUAGES)
        ):
            raise ValueError(
                "enabled country languages must be unique supported languages including en"
            )
        eligible = {"en"}
        evidence = []
        for code in sorted(set(context_country_codes)):
            self._check(code)
            for language, metadata in self._territories[code].items():
                status = metadata.get("_officialStatus")
                if language in enabled and status in {"official", "de_facto_official"}:
                    eligible.add(language)
                    evidence.append({"country": code, "language": language, "status": status})
        return tuple(language for language in enabled if language in eligible), tuple(evidence)

    def choose(
        self,
        code: str,
        rng: random.Random,
        policy: CountryPresentationConfig,
        *,
        language: str = "en",
        allowed_families: tuple[str, ...] | None = None,
        alias_opportunity: bool,
        min_words: int = 1,
        code_width: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        self._check(code, language)
        if language not in policy.language_allowlist:
            raise ValueError(f"country language outside configured allowlist: {language}")
        if type(min_words) is not int or min_words < 1 or code_width not in (None, 2, 3):
            raise ValueError("invalid country span word count or ISO width")
        families = FAMILIES if allowed_families is None else allowed_families
        if not families or len(set(families)) != len(families) or set(families) - set(FAMILIES):
            raise ValueError("country frame must allow unique supported presentation families")
        if not alias_opportunity and "name" in families:
            surface = self.preferred_name(code)
            if len(surface.split()) < min_words:
                raise ValueError("preferred country name cannot preserve required physical lines")
            return surface, {
                "family": "base_name",
                "option": surface,
                "reason": "document_base_name",
                "eligibleFamilies": ["name"],
                "ineligibleFamilies": [],
                "nonVaryingSlot": False,
            }
        if not alias_opportunity:
            for family in ("iso2", "iso3"):
                if family not in families:
                    continue
                if code_width is not None and code_width != int(family[-1]):
                    continue
                if min_words != 1:
                    raise ValueError("an opaque ISO code cannot span multiple physical lines")
                surface = self.options(code, family=family)[0].surface
                return surface, {
                    "family": family,
                    "option": surface,
                    "reason": "source_frame_requires_code",
                    "eligibleFamilies": [family],
                    "ineligibleFamilies": [],
                    "nonVaryingSlot": True,
                }
            raise ValueError("base country presentation requires a name or explicit ISO frame")
        candidates = defaultdict(list)
        for row in self.options(code, language=language, exclude_preferred=alias_opportunity):
            if row.family not in families or len(row.surface.split()) < min_words:
                continue
            if code_width is not None:
                if row.family not in {"iso2", "iso3", "dotted_code"}:
                    continue
                if len(row.surface.replace(".", "")) != code_width:
                    continue
            candidates[row.family].append(row)
        weights = policy.family_weights.model_dump()
        requested = [family for family in families if weights[family] > 0]
        eligible = [family for family in requested if candidates[family]]
        unavailable = [family for family in requested if not candidates[family]]
        if unavailable and policy.unavailable_family_policy == "error":
            raise ValueError(f"unavailable country presentation families: {unavailable}")
        if (
            not eligible
            and alias_opportunity
            and "name" in requested
            and len(self.preferred_name(code).split()) >= min_words
            and code_width is None
        ):
            surface = self.preferred_name(code)
            return surface, {
                "family": "base_name",
                "option": surface,
                "reason": "no_alias_within_frame",
                "eligibleFamilies": [],
                "ineligibleFamilies": unavailable,
                "nonVaryingSlot": True,
            }
        if not eligible:
            raise ValueError(f"no eligible country presentation for {code}, {language}, {families}")
        family = rng.choices(eligible, weights=[weights[item] for item in eligible], k=1)[0]
        option = rng.choice(candidates[family])
        return option.surface, {
            "family": family,
            "option": option.as_dict(),
            "reason": (
                "document_alias_opportunity" if alias_opportunity else "source_frame_requires_code"
            ),
            "eligibleFamilies": eligible,
            "ineligibleFamilies": unavailable,
            "nonVaryingSlot": (
                len({row.surface for name in eligible for row in candidates[name]}) == 1
            ),
        }

    def find_spans(
        self,
        text: str,
        code: str,
        *,
        postal: bool = False,
    ) -> list[tuple[int, int, str]]:
        """Find literal country mentions within an already-owned region, not a document."""
        self._check(code)
        blocked = (
            [match.span() for match in self._blocked.finditer(text)] if self._ambiguous else []
        )
        matches = []
        for match in self._patterns[code].finditer(text):
            if match.lastgroup == "foreign":
                continue
            start, end = match.span()
            if blocked and any(start < right and end > left for left, right in blocked):
                continue
            # Brazilian road designators are not postal country occurrences.
            if postal and code == "BR" and re.match(r"[ \t]*-[ \t]*\d", text[end:]):
                continue
            matches.append((start, end, match.group()))
        return matches


@lru_cache(maxsize=8)
def _load(manifest: Path, expected_sha256: str) -> Registry:
    return Registry(
        manifest.parent, expected_manifest_sha256=expected_sha256, manifest_name=manifest.name
    )
