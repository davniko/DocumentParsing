"""Goods-conditioned thermal settings with native evidence and source-owned presence.

Ranges are synthetic plausibility profiles, not commercial storage approval. The
conventional reefer envelope follows the carrier references pinned in the R28
thermal audit; product ranges must be narrower where their state requires it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from document_ocr.synthesis.curated_thermal import ventilation_meaning
from document_ocr.synthesis.generators import DeterministicStream

STAGE = "independent-thermal-profile"
SYSTEM = """Propose a small PLAUSIBLE carrying-settings range for synthetic maritime
extraction training, not an operational recommendation. Supplied registry wording
is classification context: select its explicitly requested frozen/chilled/fresh
state, not all tariff alternatives simultaneously. Keep all supplied products.
For freshProduceCodes, those HS identities select their FRESH produce
branch: 'fresh or dried' does NOT supply a second dried product. Determine
respiration from the selected products, not their HS chapter or hazard category.
Roots outside chapters 07/08 can also respire. UNKNOWN requires
supportedRange=null; a known result requires unknown=false and ordered bounds.
Frozen food stays at -18 C or colder. Chilled animal/fish products are non-live,
non-respiring. Fresh edible fruit/vegetables can respire; frozen produce does not.
Zero fresh-air exchange does not disable internal circulation. Numeric RATE slots
allow zero; CLOSED requires zero; ABSENT requires null, never an invented field.
ABSENT means the document does not print a ventilation setting, not that the
equipment lacks ventilation. For ABSENT, still propose the goods-appropriate
temperature range, leave both airflow fields null, and do not return UNKNOWN
solely because ventilation is unprinted.
Non-respiring foods require closed fresh air. Fresh produce needs a plausible
commodity-dependent exchange, not a universal zero. Select narrow compatible
ranges for EVERY product sharing one compartment. Do not average disjoint storage
requirements; return unknown instead. All combinations within the proposed small
ranges must be plausible. Propose variation where justified, not merely one ideal
setpoint, but never widen just for diversity. Use integer or half-degree Celsius
bounds and integer cubic-metres/hour airflow bounds. Equipment bounds are fences,
not product recommendations. Do not invent humidity, controlled-atmosphere settings,
product names or medical/formulation facts. Explain in one short sentence.
"""


def fresh_produce_eligible(code: str, nearest_first: Sequence[str]) -> bool:
    """Recognize a selectable fresh edible produce branch, without rerouting ambient.

    A fresh-or-dried tariff identity may select fresh. A nearer exclusively dried,
    frozen or processed description cannot inherit freshness from a broad parent.
    """
    if code[:2] not in {"07", "08"}:
        return False
    for value in nearest_first:
        text = value.casefold()
        if re.search(r"\b(?:not|excluding|other than)\s+fresh\b", text):
            return False
        if re.search(r"\bfresh\b", text):
            return True
        if re.search(
            r"\b(?:dried|desiccated|frozen|preserved|prepared|cooked|roasted|powder|flour)\b",
            text,
        ):
            return False
    return False


class ThermalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    goods: list[dict[str, str]] = Field(min_length=1)
    regime: Literal["frozen", "chilled"]
    freshProduceCodes: list[str]
    ventilation: Literal["ABSENT", "CLOSED", "RATE"]
    equipment: list[dict[str, str]]


class ThermalRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    temperatureMinC: float = Field(ge=-35, le=30)
    temperatureMaxC: float = Field(ge=-35, le=30)
    freshAirMinM3H: int | None = Field(ge=0, le=250)
    freshAirMaxM3H: int | None = Field(ge=0, le=250)
    respiration: Literal["non_respiring", "respiring_produce", "mixed"]

    @model_validator(mode="after")
    def coherent(self):
        if self.temperatureMinC > self.temperatureMaxC:
            raise ValueError("thermal temperature range must be ordered")
        if any(v * 2 % 1 for v in (self.temperatureMinC, self.temperatureMaxC)):
            raise ValueError("thermal profile requires whole or half Celsius degrees")
        a, b = self.freshAirMinM3H, self.freshAirMaxM3H
        if (a is None) != (b is None) or (a is not None and a > b):
            raise ValueError("thermal airflow range must be jointly present and ordered")
        if self.respiration == "non_respiring" and a is not None and (a != 0 or b != 0):
            raise ValueError("non-respiring cargo requires closed fresh air")
        return self


class ThermalProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    unknown: bool
    supportedRange: ThermalRange | None
    basis: str = Field(min_length=1, max_length=550)

    @model_validator(mode="after")
    def known(self):
        if self.unknown != (self.supportedRange is None):
            raise ValueError("known thermal profile requires a range; UNKNOWN must omit it")
        return self


def make_request(frozen: Mapping[str, Any], target: Mapping[str, Any]) -> ThermalRequest:
    instructions = [
        value
        for goods in target.get("goodsItemDetails", [])
        for value in goods.get("handlingInstructions", [])
    ]
    meanings = [meaning for text in instructions if (meaning := ventilation_meaning(text))]
    mode = "CLOSED" if any(m.closed for m in meanings) else "RATE" if meanings else "ABSENT"
    return ThermalRequest(
        goods=frozen["goods"],
        regime=frozen["regime"],
        freshProduceCodes=frozen["freshProduceCodes"],
        ventilation=mode,
        equipment=frozen["equipmentSlots"],
    )


@lru_cache(maxsize=6)
def _output_type(regime: str, ventilation: str) -> type[ThermalProposal]:
    fields = {}
    low, high = (-35, -18) if regime == "frozen" else (-3, 30)
    for name in ("temperatureMinC", "temperatureMaxC"):
        fields[name] = (float, Field(ge=low, le=high))
    for name in ("freshAirMinM3H", "freshAirMaxM3H"):
        if ventilation == "ABSENT":
            fields[name] = (type(None), ...)
        elif ventilation == "CLOSED" or regime == "frozen":
            fields[name] = (Literal[0], ...)
    if regime == "frozen":
        fields["respiration"] = (Literal["non_respiring"], ...)
    bounded = create_model("SourceBoundThermalRange", __base__=ThermalRange, **fields)
    return create_model(
        "SourceBoundThermalProposal",
        __base__=ThermalProposal,
        supportedRange=(bounded | None, ...),
    )


def output_type(request: ThermalRequest) -> type[ThermalProposal]:
    return _output_type(request.regime, request.ventilation)


def prompt(request: ThermalRequest) -> str:
    return json.dumps(request.model_dump(mode="json"), sort_keys=True)


def validate_profile(frozen: Mapping[str, Any]) -> ThermalProposal:
    evidence = frozen.get("nativeThermalEvidence")
    if evidence is None:
        raise ValueError("thermal draw requires an explicitly prepared native thermal profile")
    native = ThermalRequest.model_validate(evidence["request"])
    if (
        native.goods != frozen["goods"]
        or native.regime != frozen["regime"]
        or native.freshProduceCodes != frozen["freshProduceCodes"]
        or native.equipment != frozen["equipmentSlots"]
    ):
        raise ValueError("thermal profile differs from sampled goods/state/equipment")
    result = output_type(native).model_validate(evidence["output"])
    if result.unknown:
        raise ValueError("UNKNOWN thermal context has no sampling fallback")
    return result


def product_context(frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Pass the same selected state to packaging, physical planning and wording."""
    profile = validate_profile(frozen).supportedRange
    assert profile is not None
    return {
        "regime": frozen["regime"],
        "freshProduceCodes": frozen["freshProduceCodes"],
        "respiration": profile.respiration,
        "instruction": "Select the supplied frozen/chilled state, not every tariff alternative. "
        "Fresh-produce codes mean FRESH, never dried. Chilled animal/fish products "
        "are NON-LIVE; do not introduce live cargo. Keep the same product state "
        "in all descriptions, packing and numerical estimates. Settings are context, "
        "not extra goods-description content unless that source passage owns them.",
    }


def verify_evidence(root: Path, frozen, target, *, provider) -> None:
    from document_ocr.synthesis.independent_native_evidence import verify_native_evidence

    request = make_request(frozen, target)
    verify_native_evidence(
        root,
        frozen["nativeThermalEvidence"],
        stage=STAGE,
        identity=f"{frozen['sourceDocumentId']}:{frozen['variant']}",
        request=request.model_dump(mode="json"),
        model=output_type(request),
        system=SYSTEM,
        prompt=prompt(request),
        provider=provider,
    )
    validate_profile(frozen)


def sample_context(frozen, seed: int) -> dict[str, Any]:
    proposal = validate_profile(frozen)
    profile = proposal.supportedRange
    assert profile is not None
    # All disjoint package groups share a compartment: group-specific draw keys
    # must not accidentally select different carrying temperatures.
    stream = DeterministicStream(
        seed, "independent-thermal-v1", f"{frozen['sourceDocumentId']}:{frozen['variant']}"
    )
    lo, hi = int(profile.temperatureMinC * 2), int(profile.temperatureMaxC * 2)
    temperature = Decimal(lo + stream.derive("temperature").randbelow(hi - lo + 1)) / 2
    airflow = None
    if profile.freshAirMinM3H is not None:
        lo, hi = profile.freshAirMinM3H, profile.freshAirMaxM3H
        airflow = lo + stream.derive("fresh-air").randbelow(hi - lo + 1)
    return {
        "temperatureCelsius": str(temperature),
        "ventilationCbmPerHour": airflow,
        "respiration": profile.respiration,
        "basis": proposal.basis,
        "authority": STAGE,
        "supportedRange": profile.model_dump(mode="json"),
    }
