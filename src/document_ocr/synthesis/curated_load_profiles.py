"""Reviewed synthetic commodity forms, independent of observed shipment donors.

These envelopes supply extraction-training facts, not shipping approvals or
empirical shipment frequencies. Package accounting is always in declared units.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.package_registry import CategoryToken


class RegistryLoadProfile(BaseModel):
    """One non-DG ambient form with explicit per-declared-unit physical facts."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    id: str = Field(min_length=1)
    hs_headings: tuple[str, ...] = Field(min_length=1)
    form: str = Field(min_length=1)
    package_category: CategoryToken
    unit_gross_kg: Decimal = Field(gt=0)
    unit_net_kg: Decimal = Field(gt=0)
    unit_volume_m3: Decimal = Field(gt=0)
    quantity_min: int = Field(strict=True, gt=0)
    quantity_max: int = Field(strict=True, gt=0)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def coherent(self):
        if any(len(h) != 4 or not h.isascii() or not h.isdigit() for h in self.hs_headings):
            raise ValueError("load profile requires exact HS4 headings")
        if len(set(self.hs_headings)) != len(self.hs_headings):
            raise ValueError("duplicate load-profile HS4 headings")
        if self.unit_net_kg > self.unit_gross_kg:
            raise ValueError("load profile net mass exceeds gross mass")
        if self.quantity_min > self.quantity_max:
            raise ValueError("load profile quantity interval is empty")
        return self


class RegistryLoadProfiles(BaseModel):
    """Versioned, reviewed form catalog; category coverage is explicit data."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Annotated[int, Field(strict=True, ge=1, le=1)]
    profiles: tuple[RegistryLoadProfile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct(self):
        if len({p.id for p in self.profiles}) != len(self.profiles):
            raise ValueError("duplicate commodity/load profile ID")
        return self


def weighted_choice(values, weights: dict[str, int], key, stream: DeterministicStream):
    """Draw from the explicit eligible intersection; never replace a missing draw."""
    selected = [(value, weights.get(key(value), 0) if weights else 1) for value in values]
    selected = [(value, weight) for value, weight in selected if weight > 0]
    if not selected:
        raise ValueError("requested sampling weights have no eligible support")
    ticket = stream.randbelow(sum(weight for _, weight in selected))
    for value, weight in selected:
        if ticket < weight:
            return value
        ticket -= weight
    raise AssertionError("weighted draw escaped its finite support")
