"""Deterministically project printed equipment wording into supported target categories.

The input label already owns its equipment wording. This stage changes representation,
not ownership: it neither searches other containers for a type nor borrows dimensions
from cargo. Standard-height conventions apply only to reviewed equipment wording.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from document_ocr.synthesis.container_semantics import review_source_equipment_surface

_CMA_CARRIER = re.compile(
    r"\b(?:CARRIER\s*:|(?:SIGNED\s+FOR|AGENTS?\s+FOR)\s+(?:THE\s+)?CARRIER)"
    r"\s*CMA\s+CGM\b",
    re.I,
)
_TARROS_CARRIER = re.compile(
    r"\b(?:CARRIER\s*:?\s+TARROS\b|"
    r"TARROS(?:\s+S\.?\s*P\.?\s*A\.?)?\s+AS\s+CARRIER\b)",
    re.I,
)


def _source_contains(source: str, surface: str) -> bool:
    # Label whitespace/punctuation may have been joined across physical OCR lines.
    # This is only a presence gate; source field ownership stays with the extraction.
    tokens = re.findall(r"[A-Z0-9]+", surface.upper())
    if not tokens:
        return False
    pattern = (
        r"(?<![A-Z0-9])(?:[1-9][0-9]*\s*[X\u00d7]\s*)?"
        + r"[^A-Z0-9]*".join(map(re.escape, tokens))
        + r"(?![A-Z0-9])"
    )
    return re.search(pattern, source, re.I) is not None


def reconcile_equipment_categories(
    target: Mapping[str, Any],
    *,
    source_text: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Return a copied target and decisions for every equipment fallback encountered.

    Recognized categories replace ``typeDescription``; extraction versions 6/7
    can retain a known family without inventing missing dimensions. Every retained
    fallback has an explicit reason. Source text is never changed, and a carrier
    vessel name alone cannot activate carrier-specific equipment vocabulary.
    """
    result = deepcopy(dict(target))
    patch = result.get("documentPatch", {})
    if not isinstance(patch, dict):
        raise ValueError("equipment normalization requires an object documentPatch")
    if "containers" in patch and "containerInformation" in patch:
        raise ValueError("equipment normalization received two competing container arrays")
    key = "containerInformation" if "containerInformation" in patch else "containers"
    containers = patch.get(key, [])
    if not isinstance(containers, list):
        raise ValueError("equipment normalization requires a container array")
    allow_partial = result.get("schemaVersion") in {"6.0.0", "7.0.0"}
    identified_carriers = [
        name
        for name, pattern in (("CMA CGM", _CMA_CARRIER), ("TARROS", _TARROS_CARRIER))
        if pattern.search(source_text)
    ]
    # Competing carrier declarations do not authorize a carrier-specific alias.
    carrier = identified_carriers[0] if len(identified_carriers) == 1 else None
    decisions: list[dict[str, Any]] = []
    for index, container in enumerate(containers):
        if not isinstance(container, dict):
            raise ValueError("equipment normalization received a non-object container")
        surface = container.get("typeDescription")
        if surface is None:
            continue
        if not isinstance(surface, str):
            raise ValueError("equipment fallback must be text")
        if "sizeCategory" in container or "typeCategory" in container:
            raise ValueError("equipment fallback and canonical categories are exclusive")
        observation = review_source_equipment_surface(
            surface,
            temperature_present=container.get("temperatureSetpoint") is not None,
            carrier_name=carrier,
        )
        if observation.review_rule == "non_operating_reefer_conflicts_with_setpoint":
            raise ValueError("non-operating reefer conflicts with temperature setpoint")
        evidence = [surface] if _source_contains(source_text, surface) else []
        # A reviewed extraction may preserve corroborating printed code aliases as
        # A / B or A (description). Resolve each, requiring agreement on every
        # known dimension and explicit OCR support for every contributing phrase.
        conflicting = False
        if not evidence or observation.size_category is None or re.search(r"[/()]", surface):
            parts = [s.strip() for s in re.split(r"\s*/\s*|[()]", surface) if s.strip()]
            observations = [
                review_source_equipment_surface(
                    part,
                    temperature_present=container.get("temperatureSetpoint") is not None,
                    carrier_name=carrier,
                )
                for part in parts
            ]
            supported = [
                item
                for part, item in zip(parts, observations, strict=True)
                if item.size_category is not None
                and item.type_category is not None
                and _source_contains(source_text, part)
            ]
            if supported and all(_source_contains(source_text, part) for part in parts):
                # A conventionally defaulted height yields to an explicit,
                # corroborated height for the same length and equipment family.
                candidate = next(
                    (
                        item
                        for item in supported
                        if "default_standard_height" not in item.review_rule
                    ),
                    supported[0],
                )
                compatible = all(
                    (
                        item.size_category in {None, candidate.size_category}
                        or (
                            "default_standard_height" in item.review_rule
                            and item.size_category is not None
                            and candidate.size_category is not None
                            and item.size_category.removesuffix("_STANDARD_HEIGHT")
                            == candidate.size_category.removesuffix("_HIGH_CUBE")
                        )
                    )
                    and item.type_category in {None, candidate.type_category}
                    for item in observations
                )
                if compatible:
                    observation = candidate
                    evidence = parts
                else:
                    conflicting = True
        classifiable = (
            bool(evidence)
            and not conflicting
            and (
                (observation.size_category is not None and observation.type_category is not None)
                or (allow_partial and observation.type_category is not None)
            )
        )
        decision: dict[str, Any] = {
            "index": index,
            "path": f"documentPatch.{key}[{index}]",
            "equipmentIdentifier": container.get(
                "equipmentIdentifier", container.get("containerNumber")
            ),
            "surface": surface,
            "action": "canonicalize" if classifiable else "retain_fallback",
            "reason": (
                "conflicting_equipment_aliases"
                if conflicting
                else observation.review_rule
                if evidence
                else "printed_surface_not_recovered"
            ),
            "source_evidence": evidence,
            "carrier_context": carrier,
            "sizeCategory": observation.size_category,
            "typeCategory": observation.type_category,
        }
        if classifiable:
            del container["typeDescription"]
            if observation.size_category is not None:
                container["sizeCategory"] = observation.size_category
            if observation.type_category is not None:
                container["typeCategory"] = observation.type_category
        decisions.append(decision)
    return result, tuple(decisions)
