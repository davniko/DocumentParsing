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
    dgUnNumbers: list[str] = Field(default_factory=list, exclude_if=lambda value: not value)
    articleGroups: list[list[str]] | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )

    @field_validator("hsHeadings", "excludedHs6", "packageTypes", "dgUnNumbers")
    @classmethod
    def unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("cargo presentation domains must be distinct")
        return values

    @model_validator(mode="after")
    def valid_domains(self) -> "CargoPresentation":
        if self.articleGroups is not None:
            if not self.dgUnNumbers or any(
                not group or any(role not in {"passenger car", "motorbike"} for role in group)
                for group in self.articleGroups
            ):
                raise ValueError("vehicle groups require explicit DG identities and article roles")
            if not set(self.packageTypes) <= {"PACKAGE_SET", "PACKAGE_UNPACKED_OR_UNPACKAGED"}:
                raise ValueError("bare vehicle groups require counted sets or unpackaged articles")
        if any(
            len(code) != 4 or not code.isascii() or not code.isdigit() for code in self.dgUnNumbers
        ):
            raise ValueError("cargo presentation needs four-digit UN numbers")
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
    if (
        domain.dgUnNumbers
        and request.get("dangerousGoods", {}).get("unNumber") not in domain.dgUnNumbers
    ):
        raise ValueError("frozen DG identity violates the reviewed article presentation")
    if any(not domain.permits(goods["hs6"]) for goods in request["goods"]):
        raise ValueError("frozen goods violate reviewed cargo presentation")
    if "selectedCategory" in request and request["selectedCategory"] not in domain.packageTypes:
        raise ValueError("frozen package violates reviewed cargo presentation")
