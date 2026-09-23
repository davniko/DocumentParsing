"""Physical package support without turning unprinted categories into labels.

Reviewed package names/codes share physical fit support. Other free-form printed
descriptions remain separate observed signatures. All hidden categories are
projected away at the training boundary; no unprinted category is added.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

PRINTED_PREFIX = "printed-package:"
# Closed package codes agree with the existing deterministic renderer and
# carrier documentation (FedEx FSMS 2024, Appendix G; Kuehne+Nagel BOL PKGType).
# Ambiguous/compound wording is not expanded or matched by substring.
_PHYSICAL_CATEGORIES = {
    "bag": "PACKAGE_BAG",
    "bags": "PACKAGE_BAG",
    "bbl": "PACKAGE_BARREL",
    "barrel": "PACKAGE_BARREL",
    "barrels": "PACKAGE_BARREL",
    "box": "PACKAGE_BOX",
    "boxes": "PACKAGE_BOX",
    "cas": "PACKAGE_CASE",
    "case": "PACKAGE_CASE",
    "cases": "PACKAGE_CASE",
    "ctn": "PACKAGE_CARTON",
    "ctns": "PACKAGE_CARTON",
    "carton": "PACKAGE_CARTON",
    "cartons": "PACKAGE_CARTON",
    "crt": "PACKAGE_CRATE",
    "crate": "PACKAGE_CRATE",
    "crates": "PACKAGE_CRATE",
    "drm": "PACKAGE_DRUM",
    "drum": "PACKAGE_DRUM",
    "drums": "PACKAGE_DRUM",
    "pkg": "PACKAGE_PACKAGE",
    "pkgs": "PACKAGE_PACKAGE",
    "package": "PACKAGE_PACKAGE",
    "packages": "PACKAGE_PACKAGE",
    "plt": "PACKAGE_PALLET",
    "plts": "PACKAGE_PALLET",
    "pallet": "PACKAGE_PALLET",
    "pallets": "PACKAGE_PALLET",
    "pcs": "PACKAGE_PIECE",
    "piece": "PACKAGE_PIECE",
    "pieces": "PACKAGE_PIECE",
    "rol": "PACKAGE_ROLL",
    "roll": "PACKAGE_ROLL",
    "rolls": "PACKAGE_ROLL",
    "unt": "PACKAGE_UNIT",
    "unit": "PACKAGE_UNIT",
    "units": "PACKAGE_UNIT",
}
# Grammatical number also shares support for otherwise unclassified descriptions.
_COUNT_NOUNS = frozenset(
    {
        "bale",
        "bundle",
        "set",
        "pack",
        "crate",
        "drum",
        "roll",
        "bag",
        "carton",
        "pallet",
        "piece",
        "unit",
    }
)


def printed_signature(description: str) -> str:
    text = re.sub(r"\b([a-z]+)\(s\)", r"\1s", description.casefold())
    words = re.findall(r"[^\W_]+", text)
    if not words:
        raise ValueError("printed package description has no semantic surface")
    canonical = _PHYSICAL_CATEGORIES.get(" ".join(words))
    if canonical is not None:
        return canonical
    return PRINTED_PREFIX + " ".join(word + "s" if word in _COUNT_NOUNS else word for word in words)


def fit_target(target: Mapping[str, Any]) -> dict[str, Any]:
    """A private fit view; never published as a task-schema target."""
    result = deepcopy(dict(target))
    for package in result["documentPatch"].get("cargoPackages", []):
        if "typeCategory" not in package and "typeDescription" in package:
            package["typeCategory"] = printed_signature(package["typeDescription"])
    return result


def category_domains(
    packages: Sequence[Mapping[str, Any]], known_categories: frozenset[str]
) -> tuple[frozenset[str] | None, ...]:
    return tuple(
        frozenset(v for v in known_categories if not v.startswith(PRINTED_PREFIX))
        if "typeCategory" in package
        else frozenset({printed_signature(package["typeDescription"])})
        if "typeDescription" in package
        else None
        for package in packages
    )


def validate_signature(signature: Sequence[str], domains: Sequence[frozenset[str] | None]) -> bool:
    if len(signature) != len(domains):
        raise ValueError("observed package domains differ from sampled topology")
    return all(
        domain is None or value in domain for value, domain in zip(signature, domains, strict=True)
    )


def unowned_package_observation(binding: Any) -> bool:
    """Recognize closed package-kind observations, not counts or free-form text.

    These surfaces may describe an outer packing level rather than a labelled
    inner package. Without an explicit owner, never independently invent another
    noun or guess which package row to change. Preserve the complete observed
    packaging signature as a constraint *before* drawing a new scenario.
    """
    from .descendant import _PACKAGE_SURFACES

    if (
        binding.target_paths
        or binding.dependency_paths
        or binding.dependency_bindings
        or binding.derivation is not None
        or binding.value_kind not in {"package", "other_text"}
        or binding.group_kind not in {"cargo", "package", "equipment", "customs"}
    ):
        return False
    labels = {" ".join(v.upper().split()) for vs in _PACKAGE_SURFACES.values() for v in vs}
    labels.update({"PALLETS SLAC", "PIECE(S)", "BG", "WOODEN PALLETS", "WOODEN CRATES"})
    return bool(binding.occurrences) and all(
        " ".join(s.source_text.upper().split()) in labels for s in binding.occurrences
    )


def retained_category_constraints(template: Any, source: Mapping[str, Any]) -> dict[int, str]:
    """Explicit representability constraints; never undo an already sampled value."""
    if not any(unowned_package_observation(b) for b in template.bindings):
        return {}
    packages = source["documentPatch"].get("cargoPackages", [])
    if not packages:
        raise ValueError("source-only package kinds need a latent packaging contract")
    return {i: p["typeCategory"] for i, p in enumerate(packages) if "typeCategory" in p}


def observable_target(source: Mapping[str, Any], sampled: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(sampled))
    old = source["documentPatch"].get("cargoPackages", [])
    new = result["documentPatch"].get("cargoPackages", [])
    if len(old) != len(new):
        raise ValueError("sampled package topology changed")
    for original, generated in zip(old, new, strict=True):
        if original.get("typeDescription") != generated.get("typeDescription"):
            raise ValueError("retained printed package observation changed")
        if "typeCategory" not in original:
            physical = generated.pop("typeCategory", None)
            if physical is None:
                raise ValueError("latent package has no sampled physical signature")
            if "typeDescription" in original and physical != printed_signature(
                original["typeDescription"]
            ):
                raise ValueError("latent package signature contradicts printed description")
        elif generated["typeCategory"].startswith(PRINTED_PREFIX):
            raise ValueError("private package signature cannot enter extraction labels")
    return result
