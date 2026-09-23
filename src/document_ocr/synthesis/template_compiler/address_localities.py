"""Check explicit address locality components against the pinned route registry.

Street names are not geocoded. Only comma/semicolon-separated components after
the street line, optionally prefixed by a postal code, establish a city claim.
Canonical and ASCII names sharing a GeoNames identity are equivalent.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from document_ocr.synthesis.locality_registry import (
    GeoNamesLocalityRegistry,
    load_geonames_locality_registry,
)


def normalized(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", value.casefold()) if c.isalpha())


def explicit_components(address: str) -> tuple[str, ...]:
    values = []
    for component in re.split(r"[,;\n]", address)[1:]:
        words = component.strip().split()
        while words and any(c.isdigit() for c in words[0]):
            words.pop(0)
        values.append(" ".join(words))
    return tuple(values)


@dataclass(frozen=True)
class AddressLocalities:
    names: Mapping[str, Mapping[str, frozenset[int]]]
    ambiguous_names: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def from_registry(
        cls, registry: GeoNamesLocalityRegistry, regions: Mapping[str, frozenset[str]] | None = None
    ) -> AddressLocalities:
        countries = {}
        ambiguous = {}
        for country in registry.country_codes:
            names: dict[str, set[int]] = defaultdict(set)
            area_names = set((regions or {}).get(country, ()))
            for row in registry.rows_for_country(country):
                for name in (row.canonical_name, row.ascii_name):
                    names[normalized(name)].add(row.geoname_id)
                    if row.feature_code == "PPLX":
                        area_names.add(normalized(name))
            countries[country] = {key: frozenset(ids) for key, ids in names.items()}
            ambiguous[country] = frozenset(area_names)
        return cls(countries, ambiguous)

    def conflicts(
        self, address: str, *, country: str, city: str, country_name: str = ""
    ) -> tuple[str, ...]:
        if country not in self.names:
            raise ValueError("sampled address country lacks pinned locality support: " + country)
        names = self.names[country]
        expected = normalized(city)
        expected_ids = names.get(expected, frozenset())
        ambiguous = self.ambiguous_names.get(country, frozenset())
        # These fields can also name a region, neighbourhood or port alias.
        # Without a municipal identity, a second place name is not proof of a
        # contradiction. Country checks still apply independently.
        if not expected_ids or expected in ambiguous:
            return ()
        components = explicit_components(address)
        if any(names.get(normalized(name), frozenset()) & expected_ids for name in components):
            return ()  # The sampled locality is explicitly present; other names may be its parents.
        conflicts = []
        for name in components:
            key = normalized(name)
            ids = names.get(key)
            if (
                ids
                and key not in ambiguous
                and key != normalized(country_name)
                and not ids.intersection(expected_ids)
            ):
                conflicts.append(name)
        return tuple(conflicts)


def load_address_localities(root: Path, route_run: Path, admin1_path: Path) -> AddressLocalities:
    # Caller has already verified the immutable route-run commit, including this
    # transaction. Reuse its exact locality pin, never an unpinned new snapshot.
    pin = json.loads((route_run / "transaction.json").read_bytes())["config"]["locality_registry"]
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
    return AddressLocalities.from_registry(registry, {k: frozenset(v) for k, v in regions.items()})
