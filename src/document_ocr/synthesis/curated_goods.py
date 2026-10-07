"""Registry commodity domains and explicit cold-chain synthesis policies.

Commodity identity is independent of the document furnishing its physical load
profile. These are training-data policies, not operational shipping instructions.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.synthesis.generators import DeterministicStream


class TemperatureDomain(BaseModel):
    """Finite Celsius lattice used consistently across a shared cargo group."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    minimum: Decimal
    maximum: Decimal
    step: Decimal = Field(gt=0)

    @model_validator(mode="after")
    def lattice(self) -> TemperatureDomain:
        if self.maximum < self.minimum or (self.maximum - self.minimum) % self.step:
            raise ValueError("temperature bounds require an ordered, exact step lattice")
        return self

    def sample(self, stream: DeterministicStream) -> Decimal:
        return self.minimum + self.step * stream.randbelow(
            int((self.maximum - self.minimum) / self.step) + 1
        )


class RegistryThermalPolicy(BaseModel):
    """Food-only registry profiles; observed produce keeps its own settings.

    Frozen uses the earlier synthesis envelope. Zero Celsius is the common
    chilled-meat/fish baseline; different domains must be explicitly configured.
    Fresh-air ventilation is closed for these non-respiring commodity profiles.
    Produce not explicitly classified by HS is admitted separately with exact
    observed commodity/settings pairs, not assigned these generic settings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    frozen: TemperatureDomain = TemperatureDomain(minimum=-24, maximum=-18, step="0.5")
    chilled: TemperatureDomain = TemperatureDomain(minimum=0, maximum=0, step="0.5")
    frozen_chapters: tuple[str, ...] = ("02", "03", "07", "08", "16", "20")
    chilled_chapters: tuple[str, ...] = ("02", "03")
    package_categories: tuple[str, ...] = ("PACKAGE_CARTON", "PACKAGE_BOX", "PACKAGE_PACKAGE")

    @model_validator(mode="after")
    def valid_policy(self) -> RegistryThermalPolicy:
        if self.frozen.maximum > -18 or self.chilled.minimum < -2 or self.chilled.maximum > 2:
            raise ValueError("thermal synthesis domain exceeds its food profile envelope")
        for values in (self.frozen_chapters, self.chilled_chapters, self.package_categories):
            if not values or len(set(values)) != len(values):
                raise ValueError("thermal policy domains must be nonempty and distinct")
        return self
