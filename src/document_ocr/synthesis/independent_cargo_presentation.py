"""Explicit reviewed cargo domains for semantically constrained presentations.

These are source-format constraints, not observed commodity donors. Ordinary
sources have no constraint; no domain is inferred from their original HS code.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from document_ocr.synthesis.package_aliases import PACKAGE_ALIASES


class CargoPresentation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    hsHeadings: list[str] = Field(min_length=1)
    excludedHs6: list[str]
    packageTypes: list[str] = Field(min_length=1)

    @field_validator("hsHeadings", "excludedHs6", "packageTypes")
    @classmethod
    def unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("cargo presentation domains must be distinct")
        return values

    @model_validator(mode="after")
    def valid_domains(self) -> "CargoPresentation":
        if any(
            len(code) != 4 or not code.isascii() or not code.isdigit() for code in self.hsHeadings
        ):
            raise ValueError("cargo presentation needs four-digit HS headings")
        if any(
            len(code) != 6
            or not code.isascii()
            or not code.isdigit()
            or code[:4] not in self.hsHeadings
            for code in self.excludedHs6
        ):
            raise ValueError("excluded HS6 must belong to the presentation headings")
        if not set(self.packageTypes) <= PACKAGE_ALIASES.keys():
            raise ValueError("cargo presentation has an unknown flat package type")
        return self

    def permits(self, hs6: str) -> bool:
        return hs6[:4] in self.hsHeadings and hs6 not in self.excludedHs6


def validate_cargo_presentation(
    expected: Mapping[str, Any] | None, request: Mapping[str, Any]
) -> None:
    """Recheck certificate ancestry, identities and final package selection."""
    if request.get("cargoPresentation") != expected:
        raise ValueError("frozen cargo presentation differs from reviewed source authority")
    if expected is None:
        return
    domain = CargoPresentation.model_validate(expected)
    if any(not domain.permits(goods["hs6"]) for goods in request["goods"]):
        raise ValueError("frozen goods violate reviewed cargo presentation")
    if "selectedCategory" in request and request["selectedCategory"] not in domain.packageTypes:
        raise ValueError("frozen package violates reviewed cargo presentation")
