"""Native, goods-conditioned unit facts for flat independent shipments.

The model estimates plausible examples; the host owns interpolation, precision,
counts and capacity. Gross mass and a tare fraction make net <= gross true by
construction, without asking a decoder to enforce an arithmetic expression.
These are synthetic planning facts, not packing approvals or extracted labels.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import ROUND_FLOOR, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.independent_dg_context import PhysicalForm, physical_form
from document_ocr.synthesis.independent_envelopes import package_family
from document_ocr.synthesis.independent_packages import equipment_choice_brief
from document_ocr.synthesis.package_aliases import PACKAGE_ALIASES
from document_ocr.synthesis.tank_equipment import TANK_TYPES

_STRICT = ConfigDict(extra="forbid", strict=True, frozen=True)


def source_unit_profile_precision(blueprint, numeric_regions=()) -> dict[str, str]:
    """Use the coarsest owned cargo-measure precision, not equipment tare.

    With no printed owner, 0.01 kg/0.0001 m3 are explicit PRIVATE arithmetic
    precision, not an invented source presentation. Owned tonne values are
    converted to kg before constructing the lattice. Converted aliases keep
    their existing independently rounded rendering contract.
    """
    from document_ocr.synthesis.curated import flat
    from document_ocr.synthesis.curated_measurements import printed_quantum

    values = flat(blueprint.target)
    quanta: dict[str, list[Decimal]] = {"massKg": [], "volumeM3": []}
    variables = {v.key: v for v in blueprint.contract.variables}
    seen = set()
    for binding in blueprint.contract.targets:
        path = binding.path
        if not path.endswith(".value") or not any(
            part in path for part in (".grossWeight.", ".netWeight.", ".volume.")
        ):
            continue
        key = binding.expression.strip("{}")
        variable = variables.get(key)
        if variable is None or variable.kind not in {"mass", "volume"}:
            continue
        unit = values[path[:-5] + "unit"]
        factor = {
            "kilogram": Decimal(1),
            "metric_tonne": Decimal(1000),
            "pound": Decimal("0.45359237"),
            "cubic_metre": Decimal(1),
            "cubic_foot": Decimal("0.028316846592"),
        }[unit]
        kind = "volumeM3" if variable.kind == "volume" else "massKg"
        for occurrence in variable.occurrences:
            quantum = printed_quantum(occurrence.text, Decimal(variable.value)) * factor
            # Decimal planning has a power-of-ten lattice. Use a finer lattice
            # for exact unit conversions, not a coarser rounded conversion.
            quanta[kind].append(Decimal(1).scaleb(quantum.normalize().as_tuple().exponent))
            seen.add((path, occurrence.text))
    for region in blueprint.regions:
        for path in region.target_paths:
            if (
                (path, region.source) in seen
                or not path.endswith(".value")
                or not any(part in path for part in (".grossWeight.", ".netWeight.", ".volume."))
            ):
                continue
            baseline = region.baseline if region.baseline is not None else values.get(path)
            if baseline is None or region.presentation != "number":
                continue
            kind = "volumeM3" if ".volume." in path else "massKg"
            unit = values[path[:-5] + "unit"]
            factor = {
                "kilogram": Decimal(1),
                "metric_tonne": Decimal(1000),
                "pound": Decimal("0.45359237"),
                "cubic_metre": Decimal(1),
                "cubic_foot": Decimal("0.028316846592"),
            }[unit]
            quantum = printed_quantum(region.source, Decimal(str(baseline))) * factor
            quanta[kind].append(Decimal(1).scaleb(quantum.normalize().as_tuple().exponent))
    for region in numeric_regions:
        if region["role"] not in {
            "private_measurement",
            "component_measurement",
            "container_net_mass",
        }:
            continue
        baseline = region.get("baseline", region.get("baselineKg"))
        if baseline is None:
            raise ValueError("owned numeric precision lacks its certified baseline")
        kind = (
            "volumeM3"
            if "Volume" in region.get("fact", "") or region.get("fact") == "volumeM3"
            else "massKg"
        )
        quanta[kind].append(printed_quantum(region["source_text"], Decimal(str(baseline))))
    return {
        key: str(max(items) if items else Decimal("0.01" if key == "massKg" else "0.0001"))
        for key, items in quanta.items()
    }


STAGE = "independent-flat-unit-profile-v2"
SYSTEM = """Estimate three plausible coherent examples for ONE declared counted unit
of the supplied goods in maritime transport. Keep the selected package category,
counting context, thermal regime and DG identity unchanged. Include all packing
in grossKg, but EXCLUDE carrying-container hardware tare. packagingTareFraction
is packing mass divided by that gross mass, not a percent. occupiedVolumeM3 is
the occupied transport volume of the complete counted unit, not just its contents.
Order examples by increasing gross mass. Estimate real plausible scales, not
artificially miniaturized machines to fit. UNIT/SET may mean a complete machine.
Unpackaged articles, true loose bulk and installed-tank contents lots have zero
separate packing tare. A flexitank/bladder itself DOES have packing tare.
Private assumptions do not authorize printing a new package material or label.
The supplied capacity bounds are ceilings, not suggested unit weights or volumes.
For an ordinary package, all three examples describe ONE physical package, NEVER
the combined contents of a container. A pallet-sized bin is not a 20-tonne load
of many bins. One bale is not all the bales in a container. Use the supplied
equipment explanation and authoritative DG physical state; do not silently
substitute powder for liquid, or a dry-bulk tank for liquid-tank hardware.
All listed goods belong to the shared accounting group. With an assortment,
estimate a representative counted package across that assortment, not a shipment
total or a profile of only the final HS entry. This does not claim every package
physically contains every listed product.
If no plausible unit fits, return unknown=true and an empty examples list. Never
change the goods to make them fit. Otherwise return exactly three examples.
Give a single short basis sentence, at most 300 characters. No packing approval,
shipment quantity, allocation, new product wording or new extraction field.
"""


class UnitProfileRequest(BaseModel):
    model_config = _STRICT

    goods: list[dict[str, str]] = Field(min_length=1)
    regime: str
    selectedCategory: str
    countedUnitContext: str | None
    equipment: list[dict[str, str]]
    payloadKg: float = Field(gt=0, allow_inf_nan=False)
    volumeBudgetM3: float = Field(gt=0, allow_inf_nan=False)
    volumeBasis: str
    reviewFeedback: str | None = Field(
        default=None, min_length=1, max_length=2000, exclude_if=lambda value: value is None
    )
    dangerousGoods: dict[str, Any] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    physicalState: PhysicalForm | None = Field(default=None, exclude_if=lambda value: value is None)
    articleGroups: list[list[str]] | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def context(self) -> UnitProfileRequest:
        if self.articleGroups is not None and (
            self.physicalState != "article"
            or len(self.equipment) != 1
            or any(
                not group or any(role not in {"passenger car", "motorbike"} for role in group)
                for group in self.articleGroups
            )
        ):
            raise ValueError(
                "bare vehicle manifest requires explicit article roles in one carrying unit"
            )
        package_family(self.selectedCategory)
        if self.regime not in {"ambient", "frozen", "chilled", "dg"}:
            raise ValueError("unknown operational regime")
        if self.regime == "dg":
            if self.dangerousGoods is None or self.physicalState is None:
                raise ValueError(
                    "DG unit planning requires its authoritative tuple and physical state"
                )
            if self.physicalState == "gas" and self.countedUnitContext in {
                "vehicle",
                "discrete_article",
                "bulk_loading_lot",
            }:
                raise ValueError(
                    "gas cargo needs contained receptacles or installed-tank contents, "
                    "not unpackaged gas"
                )
        elif self.dangerousGoods is not None or self.physicalState is not None:
            raise ValueError("non-DG unit planning cannot invent DG physical authority")
        return self


class UnitExample(BaseModel):
    model_config = _STRICT

    grossKg: float = Field(gt=0, allow_inf_nan=False)
    packagingTareFraction: float = Field(ge=0, lt=1, allow_inf_nan=False)
    occupiedVolumeM3: float = Field(gt=0, allow_inf_nan=False)


class UnitProfile(BaseModel):
    model_config = _STRICT

    unknown: bool
    examples: list[UnitExample] = Field(max_length=3)
    basis: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def coherent(self) -> UnitProfile:
        if len(self.examples) != (0 if self.unknown else 3):
            raise ValueError("known profiles need three examples; unknown needs none")
        return self


def zero_packing_tare(request: UnitProfileRequest) -> bool:
    if request.articleGroups is not None:
        return True
    if request.countedUnitContext == "bulk_loading_lot":
        # A dry box does not itself contain loose liquid. Its unprinted liner
        # or receptacles are packing, not carrying-container hardware.
        return not packed_liquid_lot(request)
    return request.countedUnitContext in {
        "tank_loading_lot",
        "vehicle",
        "discrete_article",
    }


def packed_liquid_lot(request: UnitProfileRequest) -> bool:
    return (
        request.selectedCategory == "PACKAGE_BULK_LIQUID"
        and request.countedUnitContext == "bulk_loading_lot"
        and bool(request.equipment)
        and all(row["typeCategory"] not in TANK_TYPES for row in request.equipment)
    )


@lru_cache(maxsize=128)
def _bounded_type(
    payload: float, volume: float, zero_tare: bool, positive_tare: bool
) -> type[UnitProfile]:
    example = create_model(
        "CapacityBoundUnitExample",
        __base__=UnitExample,
        grossKg=(float, Field(gt=0, le=payload, allow_inf_nan=False)),
        occupiedVolumeM3=(float, Field(gt=0, le=volume, allow_inf_nan=False)),
        packagingTareFraction=(
            float,
            Field(ge=0, le=0, allow_inf_nan=False)
            if zero_tare
            else Field(gt=0, lt=1, allow_inf_nan=False)
            if positive_tare
            else Field(ge=0, lt=1, allow_inf_nan=False),
        ),
    )
    return create_model(
        "CapacityBoundUnitProfile",
        __base__=UnitProfile,
        examples=(list[example], Field(max_length=3)),
    )


def unit_profile_output_type(request: UnitProfileRequest) -> type[UnitProfile]:
    return _bounded_type(
        request.payloadKg,
        request.volumeBudgetM3,
        zero_packing_tare(request),
        packed_liquid_lot(request)
        or (request.physicalState == "gas" and request.countedUnitContext != "tank_loading_lot"),
    )


def unit_profile_prompt(request: UnitProfileRequest) -> str:
    if request.articleGroups is not None:
        raise ValueError("an article manifest requires the per-component native profile interface")
    context = request.model_dump(mode="json")
    context["equipmentContext"] = equipment_choice_brief(request.equipment)
    context["unitMeaning"] = (
        "ONE declared liquid contents lot INSIDE compatible self-contained receptacles or "
        "a liner/bladder within the supplied dry-box hardware. Include ALL such packing "
        "mass in gross and packing fraction; it is NOT carrying-container hardware. "
        "This does not change the declared counted category or authorize extra labels."
        if packed_liquid_lot(request)
        else "ONE individual piece of the sampled product, NEVER a pallet stack, bundle "
        "or carton containing many pieces."
        if request.selectedCategory == "PACKAGE_PIECE"
        else "ONE declared contents lot; carrying-tank hardware is not packing."
        if request.countedUnitContext in {"tank_loading_lot", "bulk_loading_lot"}
        else "ONE complete counted article; not a carrying container or shipment."
        if request.countedUnitContext in {"vehicle", "discrete_article"}
        else "ONE "
        + PACKAGE_ALIASES[request.selectedCategory].singular
        + "; not a container load of these packages."
    )
    return json.dumps(context, ensure_ascii=False, sort_keys=True)


def validate_profile(request: UnitProfileRequest, value: Mapping[str, Any]) -> UnitProfile:
    return unit_profile_output_type(request).model_validate(value)


def validate_profile_context(
    request: UnitProfileRequest, frozen: Mapping[str, Any], physical_states: Mapping[str, str]
) -> None:
    if request.dangerousGoods != frozen.get("dangerousGoods"):
        raise ValueError("native unit profile changed the authoritative DG tuple")
    if request.articleGroups != frozen.get("cargoPresentation", {}).get("articleGroups"):
        raise ValueError("native vehicle manifest changed source-owned article groups")
    state = physical_form(frozen, physical_states)
    if request.physicalState != state:
        raise ValueError("native unit profile changed the authoritative DG physical state")
    for key in ("goods", "regime", "selectedCategory", "countedUnitContext", "equipment"):
        expected = frozen["equipmentSlots"] if key == "equipment" else frozen[key]
        if getattr(request, key) != expected:
            raise ValueError("native unit profile differs from frozen " + key)


def verify_profile_evidence(
    root: Path,
    frozen: Mapping[str, Any],
    *,
    provider: Mapping[str, Any],
    physical_states: Mapping[str, str],
) -> UnitProfileRequest:
    from document_ocr.synthesis.independent_native_evidence import verify_native_evidence

    evidence = frozen["nativeUnitProfileEvidence"]
    request = UnitProfileRequest.model_validate(evidence["request"])
    if request.articleGroups is not None:
        raise ValueError("vehicle manifests require per-component native evidence")
    validate_profile_context(request, frozen, physical_states)
    answer = verify_native_evidence(
        root,
        evidence,
        stage=STAGE,
        identity=f"{frozen['sourceDocumentId']}:{frozen['variant']}",
        request=request.model_dump(mode="json"),
        model=unit_profile_output_type(request),
        system=SYSTEM,
        prompt=unit_profile_prompt(request),
        provider=provider,
    )
    if validate_profile(request, answer).unknown:
        raise ValueError("UNKNOWN numeric profile has no physical fallback")
    return request


def sample_unit_profile(
    request: UnitProfileRequest,
    profile: UnitProfile,
    capacities: Sequence[tuple[Decimal, Decimal]],
    stream: DeterministicStream,
    *,
    mass_quantum_kg: Decimal = Decimal("0.01"),
    volume_quantum_m3: Decimal = Decimal("0.0001"),
) -> tuple[Decimal, Decimal, Decimal, dict[str, Any]]:
    """Interpolate complete examples, preserving capacity and derived mass sums."""
    if profile.unknown or not capacities:
        raise ValueError("known numeric profile and explicit planning capacities required")
    checked = validate_profile(request, profile.model_dump(mode="json"))
    for quantum in (mass_quantum_kg, volume_quantum_m3):
        if not quantum.is_finite() or quantum <= 0 or quantum.normalize().as_tuple().digits != (1,):
            raise ValueError("source numeric quantum must be a positive power of ten")
    # Decimal.quantize uses the exponent, not the value (10 != 1E+1 here).
    mass_quantum_kg = mass_quantum_kg.normalize()
    volume_quantum_m3 = volume_quantum_m3.normalize()
    if any(not p.is_finite() or not v.is_finite() or p <= 0 or v <= 0 for p, v in capacities):
        raise ValueError("positive finite planning capacities required")
    if any(
        Decimal(str(request.payloadKg)) > payload or Decimal(str(request.volumeBudgetM3)) > volume
        for payload, volume in capacities
    ):
        raise ValueError("native unit bounds exceed selected hardware planning capacity")
    segment = stream.derive("unit-profile-segment").randbelow(2)
    alpha = Decimal(stream.derive("unit-profile-alpha").randbelow(1_000_001)) / 1_000_000
    # Ordering is presentation, not a physical constraint. Retain the received
    # examples unchanged in evidence and sort their indices only for traversal.
    order = sorted(range(3), key=lambda index: checked.examples[index].grossKg)
    left, right = (checked.examples[index] for index in order[segment : segment + 2])

    def mass(example: UnitExample) -> tuple[Decimal, Decimal]:
        gross = Decimal(str(example.grossKg))
        tare = gross * Decimal(str(example.packagingTareFraction))
        return gross - tare, tare

    left_net, left_tare = mass(left)
    right_net, right_tare = mass(right)
    raw_net = left_net + (right_net - left_net) * alpha
    raw_tare = left_tare + (right_tare - left_tare) * alpha
    a, b = Decimal(str(left.occupiedVolumeM3)), Decimal(str(right.occupiedVolumeM3))
    raw_cube = a + (b - a) * alpha
    if request.articleGroups is not None:
        group_count = Decimal(len(request.articleGroups))
        raw_net /= group_count
        raw_tare /= group_count
        raw_cube /= group_count
    # An integer-CBM source can still describe small packages. Use the smallest
    # decimal count group that represents a positive unit, then require every
    # row to contain whole such groups. Do not turn 0.04 m3 into zero or 1 m3.
    step = 1
    while True:
        mass_quantum = mass_quantum_kg / step
        cube_quantum = volume_quantum_m3 / step
        net = raw_net.quantize(mass_quantum, rounding=ROUND_FLOOR)
        tare = raw_tare.quantize(mass_quantum, rounding=ROUND_FLOOR)
        cube = raw_cube.quantize(cube_quantum, rounding=ROUND_FLOOR)
        if net > 0 and cube > 0 and (raw_tare == 0 or tare > 0):
            break
        step *= 10
        if any(raw_net * step > p or raw_cube * step > v for p, v in capacities):
            raise ValueError("source precision needs a count group that cannot fit the hardware")
    gross = net + tare
    if any(gross > p or cube > v for p, v in capacities):
        raise ValueError("interpolated unit exceeds explicit capacity")
    return (
        net,
        gross,
        cube,
        {
            "authority": "native_goods_conditioned_unit_profile_v1",
            "request": request.model_dump(mode="json"),
            "segment": segment,
            "exampleOrder": order,
            "massQuantumKg": str(mass_quantum_kg),
            "volumeQuantumM3": str(volume_quantum_m3),
            "quantityMultiple": step,
            "alpha": str(alpha),
            "unitPackagingTareKg": str(tare),
            "sourceCargoAccess": False,
            "packingApproval": False,
        },
    )
