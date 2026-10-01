"""Check explicit address locality components against the pinned route registry.

Street names are not geocoded. Only components after a comma, semicolon, or
line break, optionally split by a spaced dash or prefixed by a postal code,
establish a city claim.
Canonical, ASCII, and source-pinned alternate names sharing a GeoNames
identity are equivalent.
"""

from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.locality_registry import (
    GeoNamesLocalityRegistry,
    load_geonames_locality_registry,
)

LocalityIdentities = int | frozenset[int]


def _add_name(names: dict[str, LocalityIdentities], name: str, identifier: int) -> None:
    key = normalized(name)
    if not key:
        return
    previous = names.get(key)
    if previous is None:
        names[key] = identifier
    elif isinstance(previous, int):
        if previous != identifier:
            names[key] = frozenset((previous, identifier))
    elif identifier not in previous:
        names[key] = previous | {identifier}


def _overlap(left: LocalityIdentities, right: LocalityIdentities) -> bool:
    if isinstance(left, int):
        return left == right if isinstance(right, int) else left in right
    return right in left if isinstance(right, int) else not left.isdisjoint(right)


def normalized(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", value.casefold()) if c.isalnum())


def explicit_components(address: str, *, include_first: bool = False) -> tuple[str, ...]:
    values = []
    components = re.split(r"[,;\n]", address)
    for index, component in enumerate(components):
        subcomponents = re.split(r"\s+[-\u2013\u2014]\s+", component)
        # The first street/site phrase is not a locality claim. A spaced-dash
        # suffix, however, is its own printed component even on the first line.
        if index == 0 and not include_first:
            subcomponents = subcomponents[1:]
        for subcomponent in subcomponents:
            words = subcomponent.strip().split()
            while words and any(c.isdigit() for c in words[0]):
                words.pop(0)
            values.append(" ".join(words))
    return tuple(values)


@dataclass(frozen=True)
class AddressLocalities:
    names: Mapping[str, Mapping[str, LocalityIdentities]]
    ambiguous_names: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def from_registry(
        cls,
        registry: GeoNamesLocalityRegistry,
        regions: Mapping[str, frozenset[str]] | None = None,
        alternate_names: Iterable[tuple[int, str]] | None = None,
    ) -> AddressLocalities:
        country_names: dict[str, dict[str, LocalityIdentities]] = {}
        area_names: dict[str, set[str]] = {}
        for country in registry.country_codes:
            names: dict[str, LocalityIdentities] = {}
            areas = set((regions or {}).get(country, ()))
            for row in registry.rows_for_country(country):
                for name in (row.canonical_name, row.ascii_name):
                    _add_name(names, name, row.geoname_id)
                    if row.feature_code == "PPLX":
                        areas.add(normalized(name))
            country_names[country] = names
            area_names[country] = areas
        for identifier, name in alternate_names or ():
            row = registry.entry(identifier)
            key = normalized(name)
            if not key:
                continue
            _add_name(country_names[row.country_code], name, identifier)
            if row.feature_code == "PPLX":
                area_names[row.country_code].add(key)
        return cls(
            country_names,
            {country: frozenset(values) for country, values in area_names.items()},
        )

    def conflicts(
        self,
        address: str,
        *,
        country: str,
        city: str,
        country_name: str = "",
        include_first: bool = False,
    ) -> tuple[str, ...]:
        if country not in self.names:
            raise ValueError("sampled address country lacks pinned locality support: " + country)
        names = self.names[country]
        expected = normalized(city)
        expected_ids = names.get(expected)
        ambiguous = self.ambiguous_names.get(country, frozenset())
        # These fields can also name a region, neighbourhood or port alias.
        # Without a municipal identity, a second place name is not proof of a
        # contradiction. Country checks still apply independently.
        if expected_ids is None or expected in ambiguous:
            return ()
        components = explicit_components(address, include_first=include_first)
        conflicts = []
        for name in components:
            key = normalized(name)
            ids = names.get(key)
            if (
                ids is not None
                and (isinstance(ids, int) or len(ids) == 1)
                and key not in ambiguous
                and key != normalized(country_name)
                and not _overlap(ids, expected_ids)
            ):
                conflicts.append(name)
        return tuple(conflicts)


def _pinned_alternate_names(
    *, root: Path, pin: Mapping[str, str], registry: GeoNamesLocalityRegistry
) -> Iterator[tuple[int, str]]:
    candidate = root / pin["path"]
    if candidate.is_symlink():
        raise ValueError("GeoNames alternate names require a regular pinned archive")
    path = candidate.resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError("GeoNames alternate names require a regular pinned archive")
    if path.stat().st_size > 16 * 1024 * 1024 or sha256_file(path) != pin["sha256"]:
        raise ValueError("GeoNames alternate-name archive differs from its pin")
    if pin["sha256"] != registry.receipt.archive_sha256:
        raise ValueError("GeoNames alternate names differ from the compiled locality source")
    seen: set[int] = set()
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["cities15000.txt"]:
            raise ValueError("GeoNames alternate-name archive has an unexpected member")
        member = archive.getinfo("cities15000.txt")
        if member.file_size > 64 * 1024 * 1024:
            raise ValueError("GeoNames alternate-name member exceeds safety bounds")
        with archive.open(member) as stream:
            for raw in stream:
                if len(raw) > 64 * 1024:
                    raise ValueError("GeoNames alternate-name row exceeds safety bounds")
                parts = raw.decode("utf-8").rstrip("\n").split("\t")
                if len(parts) != 19:
                    raise ValueError("GeoNames alternate-name row has an invalid shape")
                identifier = int(parts[0])
                try:
                    record = registry.entry(identifier)
                except KeyError:
                    continue  # The compiled ISO registry explicitly excludes this source row.
                if record.country_code != parts[8] or identifier in seen:
                    raise ValueError("GeoNames alternate-name identity differs from the registry")
                seen.add(identifier)
                for name in filter(None, parts[3].split(",")):
                    yield identifier, name
                    words = name.split()
                    if len(words) >= 3 and words[-1].casefold() == "city":
                        yield identifier, " ".join(words[:-1])
    if len(seen) != registry.receipt.locality_audit.accepted_records:
        raise ValueError("GeoNames alternate names do not cover the compiled registry")


def load_address_localities(root: Path, route_run: Path, admin1_path: Path) -> AddressLocalities:
    # Caller has already verified the immutable route-run commit, including this
    # transaction. Reuse its exact locality pin, never an unpinned new snapshot.
    config = json.loads((route_run / "transaction.json").read_bytes())["config"]
    pin = config["locality_registry"]
    path = (root / pin["path"]).resolve(strict=True)
    if root not in path.parents:
        raise ValueError("route locality registry escapes project")
    registry = load_geonames_locality_registry(
        root=path, expected_receipt_sha256=pin["manifest_sha256"]
    )
    regions: dict[str, set[str]] = defaultdict(set)
    for line in admin1_path.read_text(encoding="utf-8").splitlines():
        code, name, ascii_name, identity = line.split("\t")
        if not identity.isdigit() or len(code.split(".")[0]) != 2:
            raise ValueError("invalid pinned administrative-area record")
        regions[code[:2]].update((normalized(name), normalized(ascii_name)))
    raw_pin = config.get("geonames_raw_cities")
    alternate_names = (
        _pinned_alternate_names(root=root, pin=raw_pin, registry=registry)
        if raw_pin is not None
        else None
    )
    return AddressLocalities.from_registry(
        registry, {k: frozenset(v) for k, v in regions.items()}, alternate_names
    )
