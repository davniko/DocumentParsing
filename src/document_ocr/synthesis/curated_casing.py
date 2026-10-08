"""Separate target normalization from reproducible, owned-text presentation."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.labeling_agents.target_normalization import UPPERCASE_FIELDS
from document_ocr.synthesis.generators import DeterministicStream

TargetCasing = Literal["uppercase", "preserve"]
RenderCasing = Literal["preserve", "uppercase", "title"]

# Product specifications, equipment codes and package units are not ordinary
# proper names. Leave their generated presentation untouched, as with endpoints
# and identifiers. Only complete, explicitly owned name/address regions qualify.
_RENDER_FIELDS = UPPERCASE_FIELDS - {
    "goodsItemDetails[].description",
    "goodsItemDetails[].handlingInstructions[]",
    "containerInformation[].typeDescription",
    "goodsItemDetails[].numberAndTypeOfPackages[].typeOfPackages",
}


class CasingPolicy(BaseModel):
    """Target convention and equally sampled document-level presentation styles.

    One style is shared by eligible party, locality and vessel regions, including
    repeats. Unowned source text, goods wording and technical strings are kept.
    ``preserve`` targets are an opt-out of normalization, not an uppercase dataset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    target: TargetCasing = "uppercase"
    render_styles: tuple[RenderCasing, ...] = Field(default=("preserve",), min_length=1)

    @model_validator(mode="after")
    def unique_styles(self) -> CasingPolicy:
        if len(set(self.render_styles)) != len(self.render_styles):
            raise ValueError("render casing styles must be distinct")
        return self

    def select(self, seed: int, sample_id: str) -> RenderCasing:
        stream = DeterministicStream(seed, "curated-render-casing-v1", sample_id)
        return self.render_styles[stream.randbelow(len(self.render_styles))]


def case_owned_text(text: str, paths: tuple[str, ...], style: RenderCasing) -> str:
    """Change casing only, and only when every owner is eligible human text."""
    if style not in {"preserve", "uppercase", "title"}:
        raise ValueError(f"unknown rendered casing: {style}")
    if (
        style == "preserve"
        or not paths
        or any(
            re.sub(r"\[\d+\]", "[]", path.removeprefix("documentPatch.")) not in _RENDER_FIELDS
            for path in paths
        )
    ):
        return text

    def token(match: re.Match) -> str:
        value = match[0]
        # Protect embedded postcode/plot identifiers and contact endpoints too,
        # even though those should ordinarily have separate owned regions.
        if (
            any(c.isdigit() for c in value)
            or "@" in value
            or "://" in value
            or value.lower().startswith("www.")
        ):
            return value
        return value.upper() if style == "uppercase" else value.title()

    result = re.sub(r"\S+", token, text)
    # Unicode titlecasing can decompose a letter (İ -> i + combining dot).
    # Compare canonical forms so the fidelity guard accepts the same letter,
    # without stripping accents or changing the emitted text.
    if unicodedata.normalize("NFC", result.upper()) != unicodedata.normalize("NFC", text.upper()):
        raise ValueError("render casing changed more than letter case")
    return result
