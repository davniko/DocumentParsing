"""Reviewed package wording, independent of observed shipment donors.

The pinned vocabulary distinguishes application categories from reviewed
extraction-only extensions. These surface forms preserve those semantics;
they do not decide which commodity can use a particular package.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random
from types import MappingProxyType
from typing import Literal

from document_ocr.synthesis.package_registry import package_category_surface_present

PackageAliasMode = Literal["source", "registry"]


@dataclass(frozen=True, slots=True)
class PackageAlias:
    """Explicit singular/plural forms retaining every material and subtype."""

    application_code: str | None
    singular: str
    plural: str
    alternate_singular: str | None = None
    alternate_plural: str | None = None

    def variants(self, quantity: int) -> tuple[str, ...]:
        """Return reviewed surfaces, not mechanically pluralized token names."""
        if type(quantity) is not int or quantity <= 0:
            raise ValueError("package alias requires a positive integer quantity")
        natural = self.singular if quantity == 1 else self.plural
        alternate = self.alternate_singular if quantity == 1 else self.alternate_plural
        return (natural,) if alternate is None else (natural, alternate)


PACKAGE_ALIASES = MappingProxyType(
    {
        "PACKAGE_BAG": PackageAlias("BG", "BAG", "BAGS"),
        "PACKAGE_BAG_JUMBO": PackageAlias(
            "JB", "JUMBO BAG", "JUMBO BAGS", "JUMBO BAG(S)", "JUMBO BAG(S)"
        ),
        "PACKAGE_BAG_PAPER": PackageAlias(
            "5M", "PAPER BAG", "PAPER BAGS", "PAPER BAG(S)", "PAPER BAG(S)"
        ),
        "PACKAGE_BAG_PAPER_MULTI_WALL": PackageAlias(
            "XJ",
            "MULTI-WALL PAPER BAG",
            "MULTI-WALL PAPER BAGS",
            "MULTI-WALL PAPER BAG(S)",
            "MULTI-WALL PAPER BAG(S)",
        ),
        "PACKAGE_BAG_PLASTIC": PackageAlias("EC", "PLASTIC BAG", "PLASTIC BAGS"),
        "PACKAGE_BAG_SUPER_BULK": PackageAlias("43", "SUPER BULK BAG", "SUPER BULK BAGS"),
        "PACKAGE_BALE_NON_COMPRESSED": PackageAlias(
            "BN", "NON-COMPRESSED BALE", "NON-COMPRESSED BALES"
        ),
        "PACKAGE_BALE": PackageAlias(None, "BALE", "BALES"),
        "PACKAGE_SET": PackageAlias(None, "SET", "SETS"),
        "PACKAGE_BIN": PackageAlias("BI", "BIN", "BINS"),
        "PACKAGE_BLOCK": PackageAlias("OK", "BLOCK", "BLOCKS"),
        "PACKAGE_BOX": PackageAlias("BX", "BOX", "BOXES"),
        "PACKAGE_BOX_FIBREBOARD": PackageAlias("4G", "FIBREBOARD BOX", "FIBREBOARD BOXES"),
        "PACKAGE_BOX_WOODEN_NATURAL_WOOD_ORDINARY": PackageAlias(
            "QP", "ORDINARY WOODEN BOX (NATURAL WOOD)", "ORDINARY WOODEN BOXES (NATURAL WOOD)"
        ),
        "PACKAGE_BULK_LIQUID": PackageAlias("VL", "BULK LIQUID", "BULK LIQUID"),
        "PACKAGE_BULK_SOLID_FINE_PARTICLES_POWDERS": PackageAlias(
            "VY", "SOLID BULK, FINE PARTICLES (POWDERS)", "SOLID BULK, FINE PARTICLES (POWDERS)"
        ),
        "PACKAGE_BULK_SOLID_GRANULAR_PARTICLES_GRAINS": PackageAlias(
            "VR",
            "SOLID BULK, GRANULAR PARTICLES (GRAINS)",
            "SOLID BULK, GRANULAR PARTICLES (GRAINS)",
        ),
        "PACKAGE_BUNDLE": PackageAlias("BE", "BUNDLE", "BUNDLES"),
        "PACKAGE_CARTON": PackageAlias("CT", "CARTON", "CARTONS"),
        "PACKAGE_CASE": PackageAlias("CS", "CASE", "CASES"),
        "PACKAGE_CASE_WOODEN": PackageAlias("7B", "WOODEN CASE", "WOODEN CASES"),
        "PACKAGE_COIL": PackageAlias("CL", "COIL", "COILS", "COIL(S)", "COIL(S)"),
        "PACKAGE_CRATE": PackageAlias("CR", "CRATE", "CRATES", "CRATE(S)", "CRATE(S)"),
        "PACKAGE_CRATE_WOODEN": PackageAlias("8B", "WOODEN CRATE", "WOODEN CRATES"),
        "PACKAGE_DRUM": PackageAlias("DR", "DRUM", "DRUMS"),
        "PACKAGE_DRUM_FIBRE": PackageAlias(
            "1G", "FIBRE DRUM", "FIBRE DRUMS", "FIBRE DRUM(S)", "FIBRE DRUM(S)"
        ),
        "PACKAGE_DRUM_PLASTIC": PackageAlias(
            "IH", "PLASTIC DRUM", "PLASTIC DRUMS", "PLASTIC DRUM(S)", "PLASTIC DRUM(S)"
        ),
        "PACKAGE_DRUM_STEEL": PackageAlias("1A", "STEEL DRUM", "STEEL DRUMS"),
        "PACKAGE_FLEXIBAG": PackageAlias("FB", "FLEXIBAG", "FLEXIBAGS"),
        "PACKAGE_FLEXITANK": PackageAlias("FE", "FLEXITANK", "FLEXITANKS"),
        "PACKAGE_INTERMEDIATE_BULK_CONTAINER": PackageAlias(
            "WA", "INTERMEDIATE BULK CONTAINER", "INTERMEDIATE BULK CONTAINERS", "IBC", "IBCS"
        ),
        "PACKAGE_LOT": PackageAlias("LT", "LOT", "LOTS"),
        "PACKAGE_OCTABIN": PackageAlias("OT", "OCTABIN", "OCTABINS"),
        "PACKAGE_PACKAGE": PackageAlias("PK", "PACKAGE", "PACKAGES"),
        "PACKAGE_PARCEL": PackageAlias("PC", "PARCEL", "PARCELS"),
        "PACKAGE_PALLET": PackageAlias("PX", "PALLET", "PALLETS"),
        "PACKAGE_PALLET_WOODEN": PackageAlias("8A", "WOODEN PALLET", "WOODEN PALLETS"),
        "PACKAGE_PIECE": PackageAlias("PP", "PIECE", "PIECES"),
        "PACKAGE_RACK": PackageAlias("RK", "RACK", "RACKS"),
        "PACKAGE_REEL": PackageAlias("RL", "REEL", "REELS"),
        "PACKAGE_ROLL": PackageAlias("RO", "ROLL", "ROLLS"),
        "PACKAGE_SACK": PackageAlias("SA", "SACK", "SACKS"),
        "PACKAGE_SKID": PackageAlias("SI", "SKID", "SKIDS", "SKID(S)", "SKID(S)"),
        "PACKAGE_SLAB": PackageAlias("SB", "SLAB", "SLABS"),
        "PACKAGE_UNIT": PackageAlias("UN", "UNIT", "UNITS"),
        "PACKAGE_UNPACKED_OR_UNPACKAGED": PackageAlias(
            "NE", "UNPACKED OR UNPACKAGED GOODS", "UNPACKED OR UNPACKAGED GOODS"
        ),
        "PACKAGE_UNPACKED_OR_UNPACKAGED_SINGLE_UNIT": PackageAlias(
            "NF", "UNPACKED OR UNPACKAGED SINGLE UNIT", "UNPACKED OR UNPACKAGED SINGLE UNITS"
        ),
        "PACKAGE_VEHICLE": PackageAlias("VN", "VEHICLE", "VEHICLES"),
    }
)

# Abbreviations survive only when the template already uses that category's
# abbreviation. New registry draws use the separately reviewed forms above.
_SOURCE_ABBREVIATIONS = MappingProxyType(
    {
        "PACKAGE_BOX": ("BX", "BXS"),
        "PACKAGE_CARTON": ("CTN", "CTNS"),
        "PACKAGE_CRATE": ("CRT", "CRT"),
        "PACKAGE_DRUM": ("DRM", "DRM"),
        "PACKAGE_INTERMEDIATE_BULK_CONTAINER": ("IBC", "IBCS"),
        "PACKAGE_PACKAGE": ("PKG", "PKGS"),
        "PACKAGE_PALLET": ("PLT", "PLTS"),
    }
)
_OPTIONAL_PLURAL = re.compile(r"\((?:S|ES)\)", re.I)


def _optional_form(entry: PackageAlias) -> str:
    """Use the explicit inflection difference, retaining trailing qualifiers."""
    if entry.singular == entry.plural:
        return entry.singular
    prefix = 0
    for singular, plural in zip(entry.singular, entry.plural, strict=False):
        if singular != plural:
            break
        prefix += 1
    suffix = 0
    while (
        suffix < len(entry.singular) - prefix
        and entry.singular[-1 - suffix] == entry.plural[-1 - suffix]
    ):
        suffix += 1
    old_end = len(entry.singular) - suffix
    new_end = len(entry.plural) - suffix
    if old_end != prefix:
        raise ValueError("package optional plural requires an explicit insertion inflection")
    return (
        entry.singular[:prefix]
        + "("
        + entry.plural[prefix:new_end]
        + ")"
        + entry.singular[old_end:]
    )


def select_package_alias(
    category: str,
    quantity: int,
    source: str,
    *,
    mode: PackageAliasMode,
    rng: Random,
) -> str:
    """Select exact package nouns; template casing is applied by the renderer.

    ``source`` preserves an existing compatible abbreviation or optional plural
    style, not the source commodity/category. ``registry`` draws from every
    reviewed form for the selected category, even if no observed donor used it.
    """
    try:
        entry = PACKAGE_ALIASES[category]
    except KeyError as error:
        raise ValueError(f"package category has no reviewed alias: {category}") from error
    forms = entry.variants(quantity)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("package alias requires a nonempty source surface")
    if mode == "registry":
        return rng.choice(forms)
    if mode != "source":
        raise ValueError(f"unknown package alias mode: {mode}")
    core = source.strip().upper()
    abbreviations = _SOURCE_ABBREVIATIONS.get(category, ())
    if core in abbreviations:
        return abbreviations[0 if quantity == 1 else 1]
    if _OPTIONAL_PLURAL.search(core):
        return _optional_form(entry)
    return forms[0]


def validate_package_alias_registry(categories: set[str], codes: dict[str, str | None]) -> None:
    """Fail before sampling if the deployed vocabulary has drifted."""
    # A task may admit a subset of the reviewed aliases. Both are bounded by
    # its pinned extraction registry; extending aliases must not redefine a
    # frozen task's categorical vocabulary.
    if not categories or not categories <= PACKAGE_ALIASES.keys() or not categories <= codes.keys():
        raise ValueError("reviewed package alias categories differ from the task registry")
    if any(
        PACKAGE_ALIASES[category].application_code != codes[category] for category in categories
    ):
        raise ValueError("reviewed package alias application codes differ from the task registry")
    for category in categories:
        entry = PACKAGE_ALIASES[category]
        for quantity in (1, 2):
            if any(
                not package_category_surface_present(surface, category)
                for surface in entry.variants(quantity)
            ):
                raise ValueError(f"reviewed alias omits a package subtype/material: {category}")
