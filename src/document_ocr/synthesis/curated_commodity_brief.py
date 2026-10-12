"""Separate sampled product facts from contextual tariff nomenclature.

Official classification wording may describe alternatives or exclude a form;
it is not a second physical cargo specification. The sampled load profile owns
physical form, while reviewed commercial identities own their actual attributes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from document_ocr.synthesis.curated_scenarios import GoodsIdentity


def commodity_brief(
    identities: Sequence[GoodsIdentity], *, physical_form: str | None = None
) -> str:
    """Build wording/review context without conflating taxonomy and cargo facts.

Tariff numbers remain in host-owned sampled facts rather than product prose.
No registry wording is edited, narrowed, or interpreted here.
"""
    if not identities:
        raise ValueError("commodity brief requires at least one identity")
    if physical_form is not None and (
        not isinstance(physical_form, str) or not physical_form.strip()
    ):
        raise ValueError("physical form must be a nonempty string when supplied")
    contextual: list[str] = []
    commercial: list[str] = []
    for index, identity in enumerate(identities, 1):
        if not isinstance(identity.phrase, str) or not identity.phrase.strip():
            raise ValueError(f"commodity identity {index} requires nonempty wording")
        line = f"Item {index}: {identity.phrase}"
        if identity.phrase_role == "classification_context":
            contextual.append(line)
        elif identity.phrase_role == "commercial":
            commercial.append(line)
        else:
            raise ValueError(f"unknown commodity phrase role: {identity.phrase_role}")
    if contextual and physical_form is None:
        raise ValueError("classification context requires authoritative physical form")
    lines = ["COMMODITY BRIEF"]
    if contextual:
        lines.extend(
            [
                "CLASSIFICATION REFERENCES — contextual commodity scope, not shipment requirements",
                *contextual,
            ]
        )
    lines.extend(["REQUIRED PRODUCT FACTS — authoritative for this shipment", *commercial])
    if physical_form is not None:
        lines.append(f"Physical form: {physical_form}")
    return "\n".join(lines)
