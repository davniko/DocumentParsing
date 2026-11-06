"""Composition-first physical authority for nested and mixed counted packages.

The native model proposes a complete group and its topology before any shipment
total is frozen. The host samples coherent physical examples and allocates whole
groups. Support mass is already part of gross mass: it is never added afterward.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import ROUND_FLOOR, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.independent_unit_profiles import (
    UnitExample,
    UnitProfile,
    UnitProfileRequest,
    sample_unit_profile,
)
from document_ocr.synthesis.package_aliases import PACKAGE_ALIASES

STAGE = "independent-composed-unit-profile-v5"
TOPOLOGY_STAGE = "independent-composition-topology-v2"
TOPOLOGY_SYSTEM = """Define ONE plausible complete packing group for the supplied
goods and frozen package roles, for maritime transport. Pick only positive integer
coefficients. Do not propose weights, dimensions, shipment counts or source ratios.
CONTAINS: one counted outer package enclosing innerPerOuter inner packages.
SUPPORTS: one counted support platform with innerPerSupport packages on it.
ON_SUPPORT: countedPerSupport counted packages/articles on ONE support.
MIXED: supportsPerGroup platforms, EACH carrying packedPerSupport inner packages,
plus loosePerGroup loose inner packages. The host derives the public package total
as supports + loose, and total inner packages as supports * packed + loose.
Never reverse the relation or reinterpret pallets as small inner retail packs.
Return the relation's coefficient object in topology. For ON_SUPPORT, for example,
eight counted bags on one pallet means topology={"countedPerSupport":8}.
Only an impossible/unknown proposal has topology=null. There is no separate unknown
flag. Keep basis to one short sentence, at most 180 chars.
"""
SYSTEM = """Estimate THREE plausible physical examples of the EXACT FROZEN GROUP
defined in the request. DO NOT CHOOSE OR CHANGE ANY COUNTS. This is not a whole
shipment unless the frozen group itself says so. If the group is ONE PALLET with
four drums, estimate one pallet with exactly four drums, never 20 pallets, never
two/four/six drums in different examples. Every example has IDENTICAL frozen
component counts. Vary only plausible fills/sizes/weights/occupied volume.
grossKg includes all cargo and ALL packing/supports in that exact group, excluding
carrying-container hardware tare. occupiedVolumeM3 covers the complete group.
packagingTareFraction = all packing mass / group gross mass. supportShareOfPacking
= support mass / all packing mass. No support means supportShareOfPacking=0.
Supports must have positive mass included in gross, not added after planning.
Preserve goods, physical form, thermal regime and DG identity. UNKNOWN has no
examples; known has exactly three examples. Do not miniaturize goods to fit.
Hardware capacity constraints are ceilings, NOT unit-size suggestions. These are
private synthetic estimates, not extraction labels or transport approval.
Give one short basis sentence describing THAT SAME frozen group, at most 180 chars.
"""


def packing_roles(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Only semantic roles enter the native brief, never receipts or source pins."""
    keys = ("privatePackingRelation", "relatedPackageType")
    if plan["privatePackingRelation"] == "MIXED":
        keys += ("supportPackageType",)
    return {key: plan[key] for key in keys}


class TopologyRequest(UnitProfileRequest):
    privatePacking: dict[str, Any]
    packingLevels: Literal[2, 3] = Field(default=2, exclude_if=lambda value: value == 2)
    innerContentsKind: Literal["packages", "goods_pieces"] = Field(
        default="packages", exclude_if=lambda value: value == "packages"
    )

    @model_validator(mode="after")
    def composed(self) -> TopologyRequest:
        if self.innerContentsKind == "goods_pieces" and self.packingLevels != 3:
            raise ValueError("goods-piece contents require three explicit counting levels")
        if self.privatePacking.get("privatePackingRelation") not in {
            "CONTAINS",
            "SUPPORTS",
            "ON_SUPPORT",
            "MIXED",
        }:
            raise ValueError("composed physical authority requires an explicit relation")
        if self.packingLevels == 3 and self.privatePacking["privatePackingRelation"] not in {
            "CONTAINS",
            "SUPPORTS",
        }:
            raise ValueError("three-level packing requires counted outer units and inner packages")
        return self


class GroupExample(UnitExample):
    supportShareOfPacking: float = Field(ge=0, le=1, allow_inf_nan=False)


class ContainsTopology(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    innerPerOuter: int = Field(ge=1)


class SupportsTopology(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    innerPerSupport: int = Field(ge=1)


class OnSupportTopology(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    countedPerSupport: int = Field(ge=1)


class MixedTopology(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    supportsPerGroup: int = Field(ge=1)
    packedPerSupport: int = Field(ge=1)
    loosePerGroup: int = Field(ge=1)


class InnerContents(BaseModel):
    """Private innermost packages inside EACH related/intermediate package."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    # Private receptacles need not be categories in the public Flat48 target.
    # Their nouns are linguistic authority, not another public package row.
    packageSingular: str = Field(min_length=1, max_length=80, pattern=r"^[A-Z]+(?:[ -][A-Z]+)*$")
    packagePlural: str = Field(min_length=1, max_length=80, pattern=r"^[A-Z]+(?:[ -][A-Z]+)*$")
    unitsPerRelated: int = Field(ge=1)

    def noun(self, count: int) -> str:
        return self.packageSingular if count == 1 else self.packagePlural


class ThreeLevelContainsTopology(ContainsTopology):
    contents: InnerContents


class ThreeLevelSupportsTopology(SupportsTopology):
    contents: InnerContents


Topology = (
    ThreeLevelContainsTopology
    | ThreeLevelSupportsTopology
    | ContainsTopology
    | SupportsTopology
    | OnSupportTopology
    | MixedTopology
)
TOPOLOGY_TYPES = {
    "CONTAINS": ContainsTopology,
    "SUPPORTS": SupportsTopology,
    "ON_SUPPORT": OnSupportTopology,
    "MIXED": MixedTopology,
}
THREE_LEVEL_TYPES = {
    "CONTAINS": ThreeLevelContainsTopology,
    "SUPPORTS": ThreeLevelSupportsTopology,
}


class TopologyProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    topology: Topology | None
    basis: str = Field(min_length=1, max_length=180)

    @property
    def unknown(self) -> bool:
        return self.topology is None


@lru_cache(maxsize=8)
def topology_output_type(
    relation: str, packing_levels: int = 2, contents_kind: str = "packages"
) -> type[TopologyProposal]:
    if packing_levels not in (2, 3) or (packing_levels == 3 and relation not in THREE_LEVEL_TYPES):
        raise ValueError("unsupported packing depth/relation")
    topology = (THREE_LEVEL_TYPES if packing_levels == 3 else TOPOLOGY_TYPES)[relation]
    if contents_kind == "goods_pieces":
        if packing_levels != 3:
            raise ValueError("goods-piece contents require three counting levels")
        contents = create_model(
            "GoodsPieceContents",
            __base__=InnerContents,
            packageSingular=(Literal["PIECE"], ...),
            packagePlural=(Literal["PIECES"], ...),
        )
        topology = create_model("GoodsPieceTopology", __base__=topology, contents=(contents, ...))
    elif contents_kind != "packages":
        raise ValueError("unknown inner contents kind")
    return create_model(
        "FrozenCompositionTopology",
        __base__=TopologyProposal,
        topology=(
            topology | None,
            ...,
        ),
    )


def topology_prompt(request: TopologyRequest) -> str:
    context = {
        key: value
        for key, value in request.model_dump(mode="json").items()
        if key
        in {
            "goods",
            "regime",
            "selectedCategory",
            "countedUnitContext",
            "dangerousGoods",
            "physicalState",
            "privatePacking",
            "reviewFeedback",
            "componentPhysicalStates",
        }
    }
    if request.packingLevels == 3:
        context["requiredPackingLevels"] = 3
        context["innerContentsInstruction"] = (
            "Preserve THREE levels: one counted outer/support unit, its related/intermediate "
            "packages, and innermost packages inside EACH intermediate package. Choose a "
            "plausible innermost package/receptacle and positive unitsPerRelated. "
            "Give its uppercase packageSingular and packagePlural nouns (for example "
            "GLASS BOTTLE / GLASS BOTTLES), not a product name, quantity or instruction. "
            "The frozen relatedPackageType is the middle level, not the innermost level. "
            "Each level must be a DISTINCT physical package, not the same receptacle "
            "restated with its material (DRUM containing one STEEL DRUM does not establish "
            "a third level). If a drum is the intermediate overpack, its contents must "
            "be separate smaller inner receptacles; do not nest full-sized drums inside it. "
            "Do not drop a level or replace package identities with cargo pieces. "
            "If these frozen package roles cannot plausibly form this nesting, "
            "return topology=null."
        )
        if request.innerContentsKind == "goods_pieces":
            context["innerContentsInstruction"] = (
                "Preserve THREE counting levels: counted outer/support units, related "
                "INTERMEDIATE packages, and individual GOODS PIECES inside each intermediate "
                "package. The last level counts products, not additional wrappers/receptacles. "
                "Use PIECE/PIECES and a plausible positive unitsPerRelated. The selected "
                "goods must be countable as pieces; do not invent bottles or bags for loose "
                "liquid/powder or miniaturize large goods to force a fit. Return topology=null "
                "if this complete arrangement is not plausible. Source ratios are not supplied."
            )
    return json.dumps(context, ensure_ascii=False, sort_keys=True)


class ComposedRequest(TopologyRequest):
    topology: Topology

    @model_validator(mode="after")
    def exact_topology(self):
        expected = (THREE_LEVEL_TYPES if self.packingLevels == 3 else TOPOLOGY_TYPES)[
            self.privatePacking["privatePackingRelation"]
        ]
        if type(self.topology) is not expected:
            raise ValueError("frozen topology belongs to a different packing relation")
        if self.innerContentsKind == "goods_pieces" and (
            self.topology.contents.packageSingular != "PIECE"
            or self.topology.contents.packagePlural != "PIECES"
        ):
            raise ValueError("goods-piece contents cannot become inner packaging")
        return self


class PhysicalGroupProfile(UnitProfile):
    examples: list[GroupExample] = Field(max_length=3)


class ComposedProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    unknown: bool
    topology: Topology | None
    examples: list[GroupExample] = Field(max_length=3)
    basis: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def cardinality(self):
        if self.unknown:
            if self.topology is not None or self.examples:
                raise ValueError("UNKNOWN composed profile has no topology or examples")
        elif len(self.examples) != 3 or self.topology is None:
            raise ValueError("known complete group needs a topology and three examples")
        return self

    @property
    def countedPerGroup(self) -> int:
        if isinstance(self.topology, OnSupportTopology):
            return self.topology.countedPerSupport
        if isinstance(self.topology, MixedTopology):
            return self.topology.supportsPerGroup + self.topology.loosePerGroup
        return 0 if self.topology is None else 1

    @property
    def relatedPerGroup(self) -> int:
        if isinstance(self.topology, ContainsTopology):
            return self.topology.innerPerOuter
        if isinstance(self.topology, SupportsTopology):
            return self.topology.innerPerSupport
        if isinstance(self.topology, OnSupportTopology):
            return 1
        if isinstance(self.topology, MixedTopology):
            return (
                self.topology.supportsPerGroup * self.topology.packedPerSupport
                + self.topology.loosePerGroup
            )
        return 0

    @property
    def supportPerGroup(self) -> int:
        if isinstance(self.topology, MixedTopology):
            return self.topology.supportsPerGroup
        return int(isinstance(self.topology, (SupportsTopology, OnSupportTopology)))

    @property
    def loosePerGroup(self) -> int:
        return self.topology.loosePerGroup if isinstance(self.topology, MixedTopology) else 0


@lru_cache(maxsize=128)
def _output_type(payload: float, volume: float, relation: str) -> type[PhysicalGroupProfile]:
    example = create_model(
        "CapacityBoundCompleteGroupExample",
        __base__=GroupExample,
        grossKg=(float, Field(gt=0, le=payload, allow_inf_nan=False)),
        occupiedVolumeM3=(float, Field(gt=0, le=volume, allow_inf_nan=False)),
        packagingTareFraction=(
            float,
            Field(
                ge=0 if relation == "CONTAINS" else None,
                gt=0 if relation != "CONTAINS" else None,
                lt=1,
                allow_inf_nan=False,
            ),
        ),
        supportShareOfPacking=(
            float,
            Field(
                ge=0 if relation == "CONTAINS" else None,
                gt=0 if relation != "CONTAINS" else None,
                le=0 if relation == "CONTAINS" else 1,
                allow_inf_nan=False,
            ),
        ),
    )

    return create_model(
        "CapacityBoundPhysicalGroupProfile",
        __base__=PhysicalGroupProfile,
        examples=(list[example], Field(max_length=3)),
    )


def composed_output_type(request: ComposedRequest) -> type[PhysicalGroupProfile]:
    return _output_type(
        request.payloadKg, request.volumeBudgetM3, request.privatePacking["privatePackingRelation"]
    )


def validate_composed(request: ComposedRequest, value: Mapping[str, Any]) -> ComposedProfile:
    physical = composed_output_type(request).model_validate(value)
    return ComposedProfile(
        **physical.model_dump(mode="json"),
        topology=None if physical.unknown else request.topology,
    )


def composed_prompt(request: ComposedRequest) -> str:
    context = json.loads(topology_prompt(request))
    context["frozenTopology"] = request.topology.model_dump(mode="json")
    counted = PACKAGE_ALIASES[request.selectedCategory].singular
    related = PACKAGE_ALIASES[request.privatePacking["relatedPackageType"]].plural
    t = request.topology
    if isinstance(t, ContainsTopology):
        definition = f"ONE {counted} enclosing EXACTLY {t.innerPerOuter} {related}."
    elif isinstance(t, SupportsTopology):
        definition = f"ONE {counted} carrying EXACTLY {t.innerPerSupport} {related}."
    elif isinstance(t, OnSupportTopology):
        support = PACKAGE_ALIASES[request.privatePacking["relatedPackageType"]].singular
        definition = f"EXACTLY {t.countedPerSupport} counted {counted} units on ONE {support}."
    else:
        support = PACKAGE_ALIASES[request.privatePacking["supportPackageType"]].plural
        definition = (
            f"EXACTLY {t.supportsPerGroup} {support}, EACH carrying {t.packedPerSupport} "
            f"{related}, PLUS {t.loosePerGroup} loose {related}. The entire group has "
            f"{t.supportsPerGroup + t.loosePerGroup} public packages and "
            f"{t.supportsPerGroup * t.packedPerSupport + t.loosePerGroup} inner packages."
        )
    if isinstance(t, (ThreeLevelContainsTopology, ThreeLevelSupportsTopology)):
        contents = t.contents.packagePlural
        definition += (
            f" EACH of those intermediate {related} encloses EXACTLY "
            f"{t.contents.unitsPerRelated} {contents}. Include packing at ALL THREE levels."
        )
        if request.innerContentsKind == "goods_pieces":
            definition = definition.replace(
                "Include packing at ALL THREE levels.",
                "The innermost units are individual goods pieces, not more packages. "
                "Include mass/volume of every piece and all actual outer/intermediate packing.",
            )
    context["exactFrozenGroup"] = (
        definition + " These coefficients count the named PACKAGES, not product pieces, "
        "unless the supplied counted category itself is an unpackaged article. "
        "Do not replace inner crates with glass panes or change any unit identity."
    )
    if request.innerContentsKind == "goods_pieces":
        context["exactFrozenGroup"] = definition + " Preserve these distinct counting roles."
    return json.dumps(context, ensure_ascii=False, sort_keys=True)


def verify_composed_evidence(
    root: Path,
    frozen: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    provider: Mapping[str, Any],
    physical_states: Mapping[str, str],
) -> None:
    from document_ocr.synthesis.independent_native_evidence import verify_native_evidence

    evidence = frozen["nativeComposedProfileEvidence"]
    request = ComposedRequest.model_validate(evidence["request"])
    if request.privatePacking != packing_roles(plan):
        raise ValueError("composed profile changed the frozen private packing plan")
    for key in ("goods", "regime", "selectedCategory", "countedUnitContext", "dangerousGoods"):
        if getattr(request, key) != frozen.get(key):
            raise ValueError("composed profile changed frozen " + key)
    if request.equipment != frozen["equipmentSlots"]:
        raise ValueError("composed profile changed frozen equipment")
    from document_ocr.synthesis.independent_dg_context import physical_form, physical_forms

    expected_state = physical_form(frozen, physical_states)
    if request.physicalState != expected_state:
        raise ValueError("composed profile changed DG physical authority")
    expected_forms = physical_forms(frozen, physical_states) if "dgComponents" in frozen else None
    if request.componentPhysicalStates != expected_forms:
        raise ValueError("composed profile changed component physical authority")
    topology_request = TopologyRequest.model_validate(
        {k: v for k, v in request.model_dump(mode="json").items() if k != "topology"}
    )
    topology_answer = verify_native_evidence(
        root,
        frozen["nativeCompositionTopologyEvidence"],
        stage=TOPOLOGY_STAGE,
        identity=f"{frozen['sourceDocumentId']}:{frozen['variant']}",
        request=topology_request.model_dump(mode="json"),
        model=topology_output_type(
            request.privatePacking["privatePackingRelation"],
            request.packingLevels,
            request.innerContentsKind,
        ),
        system=TOPOLOGY_SYSTEM,
        prompt=topology_prompt(topology_request),
        provider=provider,
    )
    if topology_answer["topology"] != request.topology.model_dump(mode="json"):
        raise ValueError("physical profile does not use its received frozen topology")
    answer = verify_native_evidence(
        root,
        evidence,
        stage=STAGE,
        identity=f"{frozen['sourceDocumentId']}:{frozen['variant']}",
        request=request.model_dump(mode="json"),
        model=composed_output_type(request),
        system=SYSTEM,
        prompt=composed_prompt(request),
        provider=provider,
    )
    if validate_composed(request, answer).unknown:
        raise ValueError("UNKNOWN complete group has no physical fallback")


def sample_composed_profile(
    request: ComposedRequest,
    value: Mapping[str, Any],
    capacities: Sequence[tuple[Decimal, Decimal]],
    stream: DeterministicStream,
    *,
    mass_quantum_kg: Decimal,
    volume_quantum_m3: Decimal,
):
    profile = validate_composed(request, value)
    if profile.unknown:
        raise ValueError("UNKNOWN complete group cannot be sampled")
    mass_quantum_kg = mass_quantum_kg.normalize()
    volume_quantum_m3 = volume_quantum_m3.normalize()
    count = Decimal(profile.countedPerGroup)
    # Sampling per-count averages on the source lattice avoids nonterminating
    # decimal division (e.g. three public packages per group). Whole groups are
    # enforced below; no partial support or mixed accounting group is loaded.
    unit = UnitProfile(
        unknown=False,
        basis=profile.basis,
        examples=[
            UnitExample(
                grossKg=e.grossKg / float(count),
                packagingTareFraction=e.packagingTareFraction,
                occupiedVolumeM3=e.occupiedVolumeM3 / float(count),
            )
            for e in profile.examples
        ],
    )
    base = UnitProfileRequest.model_validate(
        {
            k: v
            for k, v in request.model_dump(mode="json").items()
            if k not in {"privatePacking", "topology", "packingLevels", "innerContentsKind"}
        }
    )
    # Complete supported articles are packed groups even if the counted
    # article itself is unpackaged. Do not apply the flat zero-tare rule.
    base = base.model_copy(update={"countedUnitContext": None})
    net, gross, cube, authority = sample_unit_profile(
        base,
        unit,
        capacities,
        stream,
        mass_quantum_kg=mass_quantum_kg,
        volume_quantum_m3=volume_quantum_m3,
    )
    step = authority["quantityMultiple"]
    order = authority["exampleOrder"]
    left, right = [
        profile.examples[i] for i in order[authority["segment"] : authority["segment"] + 2]
    ]
    alpha = Decimal(authority["alpha"])

    def support(e):
        return (
            Decimal(str(e.grossKg))
            * Decimal(str(e.packagingTareFraction))
            * Decimal(str(e.supportShareOfPacking))
        )

    raw_support = min(
        support(left) + (support(right) - support(left)) * alpha, (gross - net) * count
    )
    support_group = raw_support.quantize(mass_quantum_kg / step, rounding=ROUND_FLOOR)
    # A source printing tens of kg can still use a 3 kg pallet. Just as for
    # small unit cube, require enough complete groups to represent the mass;
    # never erase the support or substitute a family default.
    while profile.supportPerGroup and support_group <= 0:
        step *= 10
        if any(gross * count * step > p or cube * count * step > v for p, v in capacities):
            raise ValueError("source support precision needs complete groups that cannot fit")
        support_group = raw_support.quantize(mass_quantum_kg / step, rounding=ROUND_FLOOR)
    authority["quantityMultiple"] = profile.countedPerGroup * step
    authority.update(
        authority="native_goods_conditioned_composed_profile_v1",
        composition={
            "relation": request.privatePacking["privatePackingRelation"],
            "countedPerGroup": profile.countedPerGroup,
            "relatedPerGroup": profile.relatedPerGroup,
            "supportPerGroup": profile.supportPerGroup,
            "loosePerGroup": profile.loosePerGroup,
            "supportTareKgPerGroup": str(support_group),
        },
        request=request.model_dump(mode="json"),
    )
    if request.packingLevels == 3:
        authority["composition"]["contents"] = request.topology.contents.model_dump(mode="json")
    return net, gross, cube, authority


def scope_composition(
    draw: Mapping[str, Any], plan: Mapping[str, Any], quantity: int
) -> dict[str, Any] | None:
    numeric = draw["provenance"].get("numericEnvelope", {})
    group = numeric.get("composition")
    if group is None:
        return None
    if group["relation"] != plan["privatePackingRelation"] or quantity % group["countedPerGroup"]:
        raise ValueError("scoped quantity does not own whole frozen composition groups")
    groups = quantity // group["countedPerGroup"]
    support = None
    count = groups * group["supportPerGroup"]
    if count:
        category = (
            plan["supportPackageType"]
            if group["relation"] == "MIXED"
            else draw["internalCategory"]
            if group["relation"] == "SUPPORTS"
            else plan["relatedPackageType"]
        )
        total = groups * Decimal(group["supportTareKgPerGroup"])
        support = {
            "category": category,
            "quantity": count,
            "unitTareKg": str(total / count),
            "totalTareKg": str(total),
            "packingBudgetKg": str(
                quantity
                * (
                    Decimal(draw["physical"]["unitGrossKg"])
                    - Decimal(draw["physical"]["unitNetKg"])
                )
            ),
            "authority": {"authority": numeric["authority"], "sourceCargoAccess": False},
        }
    result = {
        "relatedQuantity": groups * group["relatedPerGroup"],
        "supportEnvelope": support,
        "mixed": {
            "supportQuantity": count,
            "packedQuantity": groups * (group["relatedPerGroup"] - group["loosePerGroup"]),
            "looseQuantity": groups * group["loosePerGroup"],
            "totalRelatedQuantity": groups * group["relatedPerGroup"],
            "declaredQuantity": quantity,
            "supportEnvelope": support,
            "sourceCountRatioUsed": False,
        },
    }
    if "contents" in group:
        contents = InnerContents.model_validate(group["contents"])
        result["contents"] = {
            **contents.model_dump(mode="json"),
            "quantity": result["relatedQuantity"] * contents.unitsPerRelated,
        }
    return result
