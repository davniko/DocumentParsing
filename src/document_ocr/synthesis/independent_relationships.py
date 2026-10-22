"""Source-capability-bounded private package roles, never public extraction labels."""

from __future__ import annotations

import json
from collections.abc import Container, Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from document_ocr.synthesis.independent_envelopes import (
    PACKAGE_CATEGORY_FAMILIES,
    PACKAGE_COUNTING_CONTEXTS,
)
from document_ocr.synthesis.independent_packages import (
    PACKAGE_TYPES,
    PackageGoodsContext,
    PackageType,
)
from document_ocr.synthesis.package_aliases import PACKAGE_ALIASES

PrivatePackingRelation = Literal["CONTAINS", "SUPPORTS", "ON_SUPPORT", "NONE", "UNKNOWN"]
_RESOLVED_RELATIONS = frozenset({"CONTAINS", "SUPPORTS", "ON_SUPPORT", "NONE"})
_OPEN_SUPPORT_FAMILIES = frozenset({"PALLET", "SKID", "RACK"})
RELATED_PACKAGE_TYPES = tuple(
    category for category in PACKAGE_TYPES if not PACKAGE_COUNTING_CONTEXTS[category]
)
if TYPE_CHECKING:
    RelatedPackageType = str
else:
    RelatedPackageType = Literal.__getitem__(RELATED_PACKAGE_TYPES)

PRIVATE_PACKING_SYSTEM = """Propose a private packing relationship for a maritime bill of lading.
The declared counted package type is FIXED. Never change it. Consider all goods
together and preserve their supplied form. The plan is private generation context,
not extraction labels or legal dangerous-goods packing approval. Unusual but workable
arrangements are acceptable; grossly impossible pairings require UNKNOWN.
Relations are relative to the counted unit:
CONTAINS: counted units enclose related packages.
SUPPORTS: counted units support related packages, e.g. counted pallets supporting cartons.
ON_SUPPORT: counted units rest on related support packages, e.g. counted drums on pallets.
CONTAINS requires physical enclosure, not merely an outer accounting level.
An open load platform supports its load; it does not enclose it.
NONE: a positive decision that no separate inner/support package is needed or supplied.
UNKNOWN: no workable plan can be proposed without changing the fixed counted unit/form.
NONE and UNKNOWN must have relatedPackageType=null; the other relations need one type.
Choose only a presentation-supported relation, or UNKNOWN if none works.
Prefer NONE when goods directly fill the counted receptacle and no separate packing
layer is supplied. Integral parts of a package are not separate package units.
Do not invent counts, weights, dimensions, capacities, identifiers or regulated facts.
Related packages are countable packing units. Bulk-liquid/grain loading lots,
unpacked goods, vehicles and equipment-sized flexitank systems are not inner
packages here. Directly filled liquid belongs to the contents: choose NONE rather
than inventing a numbered BULK LIQUID inner layer inside drums.
Do not infer related-package material from cargo material or the counted package's
material. Use a material-specific related type only if the supplied form establishes it.
Keep the reason short and about the relationship; the host supplies all quantities
and renders every occurrence from the accepted plan. The reason remains audit-only.
"""


class PrivatePackingRequest(PackageGoodsContext):
    """Reviewed source presentation capabilities are explicit, not donor constraints."""

    countedCategory: PackageType
    allowedRelations: list[Literal["CONTAINS", "SUPPORTS", "ON_SUPPORT", "NONE"]] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def distinct_relations(self) -> PrivatePackingRequest:
        if len(self.allowedRelations) != len(set(self.allowedRelations)):
            raise ValueError("presentation-supported relations must be distinct")
        return self


class PrivatePackingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    privatePackingRelation: PrivatePackingRelation
    relatedPackageType: RelatedPackageType | None
    reason: str = Field(
        min_length=1, pattern=r"\S", description="Audit only; no numerical authority."
    )

    @property
    def unknown(self) -> bool:
        return self.privatePackingRelation == "UNKNOWN"

    @model_validator(mode="after")
    def related_type_presence(self) -> PrivatePackingPlan:
        if (self.privatePackingRelation not in {"NONE", "UNKNOWN"}) != (
            self.relatedPackageType is not None
        ):
            raise ValueError("related type presence disagrees with private packing relation")
        return self


def private_packing_prompt(
    request: PrivatePackingRequest | Mapping[str, Any], *, hs_codes: Container[str]
) -> str:
    value = (
        request
        if isinstance(request, PrivatePackingRequest)
        else PrivatePackingRequest.model_validate(request)
    )
    if any(goods.hs6 not in hs_codes for goods in value.goods):
        raise ValueError("private-packing goods must belong to the supplied HS registry")
    lines = ["Complete shared goods group:"]
    lines.extend(f"- HS {goods.hs6}: {goods.description}" for goods in value.goods)
    lines.extend(
        [
            "Declared counted package (unchangeable): " + value.countedCategory,
            "Transport: maritime",
            "Regime: " + value.regime,
            "Equipment: " + value.equipment,
        ]
    )
    for name in ("form", "dangerousGoods", "packingContext"):
        context = getattr(value, name)
        if context is not None:
            lines.append(name + ": " + json.dumps(context, ensure_ascii=False, sort_keys=True))
    lines.append("Available related package types:")
    lines.extend(
        category + ": " + PACKAGE_ALIASES[category].singular for category in RELATED_PACKAGE_TYPES
    )
    lines.extend(
        [
            "Presentation-supported and counted-type-compatible private relations: "
            + ", ".join(_compatible_relations(value)),
            "UNKNOWN is explicit if no supported relationship is workable.",
            "Return only privatePackingRelation, relatedPackageType and short audit-only reason.",
        ]
    )
    return "\n".join(lines)


def _compatible_relations(request: PrivatePackingRequest) -> tuple[str, ...]:
    counted_open = PACKAGE_CATEGORY_FAMILIES[request.countedCategory] in _OPEN_SUPPORT_FAMILIES
    return tuple(
        sorted(
            relation
            for relation in request.allowedRelations
            if not (counted_open and relation == "CONTAINS")
        )
    )


@lru_cache(maxsize=16)
def _output_type(relations: tuple[str, ...]) -> type[PrivatePackingPlan]:
    return create_model(
        "PrivatePackingPlanFor" + "Or".join(relations),
        __base__=PrivatePackingPlan,
        privatePackingRelation=(Literal.__getitem__((*relations, "UNKNOWN")), ...),
    )


def private_packing_output_type(
    request: PrivatePackingRequest | Mapping[str, Any],
) -> type[PrivatePackingPlan]:
    value = (
        request
        if isinstance(request, PrivatePackingRequest)
        else PrivatePackingRequest.model_validate(request)
    )
    return _output_type(_compatible_relations(value))


def validate_packing_direction(
    counted_category: str, relation: str, related_category: str | None
) -> None:
    """Reject demonstrated support/enclosure reversals without a goods classifier."""
    if counted_category not in PACKAGE_CATEGORY_FAMILIES:
        raise ValueError("counted package is outside the complete flat vocabulary")
    if relation not in _RESOLVED_RELATIONS:
        raise ValueError("private relation is unresolved or unknown")
    if (relation == "NONE") != (related_category is None):
        raise ValueError("private relation and related type presence disagree")
    if relation == "NONE":
        return
    if related_category is None or related_category not in PACKAGE_CATEGORY_FAMILIES:
        raise ValueError("related package is outside the complete flat vocabulary")
    if related_category not in RELATED_PACKAGE_TYPES:
        raise ValueError("related unit requires a counting context, not a private package layer")
    counted_open = PACKAGE_CATEGORY_FAMILIES[counted_category] in _OPEN_SUPPORT_FAMILIES
    related_open = PACKAGE_CATEGORY_FAMILIES[related_category] in _OPEN_SUPPORT_FAMILIES
    if relation == "CONTAINS" and counted_open:
        raise ValueError("an open counted support platform cannot enclose its related packages")
    if relation == "SUPPORTS" and related_open and not counted_open:
        raise ValueError("SUPPORTS reverses a counted nonplatform and related support platform")
    if relation == "ON_SUPPORT" and counted_open and not related_open:
        raise ValueError("ON_SUPPORT reverses a counted platform and nonplatform related package")


def validate_private_packing_plan(
    request: PrivatePackingRequest | Mapping[str, Any],
    value: PrivatePackingPlan | Mapping[str, Any],
) -> PrivatePackingPlan:
    """An explicit UNKNOWN is returned for the caller to hold, never converted to NONE."""
    request = (
        request
        if isinstance(request, PrivatePackingRequest)
        else PrivatePackingRequest.model_validate(request)
    )
    raw = value.model_dump(mode="python") if isinstance(value, PrivatePackingPlan) else value
    plan = private_packing_output_type(request).model_validate(raw)
    if plan.unknown:
        return plan
    validate_packing_direction(
        request.countedCategory, plan.privatePackingRelation, plan.relatedPackageType
    )
    return plan


def private_packing_facts(plan: PrivatePackingPlan) -> dict[str, Any]:
    """Never propagate a model's free-text explanation into host counts or wording."""
    if plan.unknown:
        raise ValueError("unresolved private relationship cannot become operative facts")
    return plan.model_dump(mode="json", exclude={"reason"})


def verify_native_plan_evidence(
    root: Path,
    frozen_request: Mapping[str, Any],
    frozen_plan: Mapping[str, Any],
    *,
    provider: Mapping[str, Any],
    hs_codes: Container[str],
    allowed_relations: tuple[str, ...],
    source_certificate_sha256: str,
) -> dict[str, Any]:
    """Return operative roles only after exact native or reviewed-NONE provenance."""
    from document_ocr.synthesis.independent_native_evidence import verify_native_evidence

    if allowed_relations == ("MIXED",):
        from document_ocr.synthesis.independent_accounting import verify_mixed_plan_evidence

        return verify_mixed_plan_evidence(
            root, frozen_request, frozen_plan, provider=provider, hs_codes=hs_codes
        )

    if frozen_plan.get("origin") == "reviewed_no_private_relation":
        if (
            allowed_relations != ("NONE",)
            or frozen_plan.get("sourceCertificateSha256") != source_certificate_sha256
            or frozen_plan.get("privatePackingRelation") != "NONE"
            or frozen_plan.get("relatedPackageType") is not None
        ):
            raise ValueError("reviewed NONE plan differs from its source certificate")
        return {"privatePackingRelation": "NONE", "relatedPackageType": None}
    if frozen_plan.get("origin") != "native_typed_private_plan":
        raise ValueError("private plan lacks a validated native or reviewed-NONE origin")
    evidence = frozen_plan.get("nativePlanEvidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("private plan lacks immutable native evidence")
    request = PrivatePackingRequest.model_validate(evidence["request"])
    if request.countedCategory != frozen_request[
        "selectedCategory"
    ] or request.allowedRelations != list(allowed_relations):
        raise ValueError("native private plan differs from current counted unit/source capability")
    data = request.model_dump(mode="json")
    if any(
        data[key] != value
        for key, value in {
            "goods": frozen_request["goods"],
            "regime": frozen_request["regime"],
            "form": frozen_request["goodsForm"],
            "dangerousGoods": frozen_request["dangerousGoods"],
        }.items()
    ):
        raise ValueError("native private-plan goods/context differs from frozen request")
    output = verify_native_evidence(
        root,
        evidence,
        stage="independent-private-packing",
        identity=f"{frozen_request['sourceDocumentId']}:{frozen_request['variant']}",
        request=data,
        model=private_packing_output_type(request),
        system=PRIVATE_PACKING_SYSTEM,
        prompt=private_packing_prompt(request, hs_codes=hs_codes),
        provider=provider,
    )
    result = private_packing_facts(validate_private_packing_plan(request, output))
    if any(frozen_plan.get(key) != value for key, value in result.items()):
        raise ValueError("frozen private relationship differs from received native decision")
    return result
