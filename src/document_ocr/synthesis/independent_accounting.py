"""Typed semantic choices for a source-owned mixed support/loose-unit equation.

The provider chooses only compatible package types. Quantities and conservation
remain exclusively host-owned; the public counted type remains generic PACKAGE.
"""

from __future__ import annotations

from collections.abc import Container, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from document_ocr.synthesis.independent_packages import PackageGoodsContext
from document_ocr.synthesis.independent_relationships import RELATED_PACKAGE_TYPES
from document_ocr.synthesis.package_aliases import PACKAGE_ALIASES

MIXED_PACKING_SYSTEM = """Choose package types for a maritime mixed-accounting presentation.
The document counts support units PLUS loose packages as generic PACKAGES.
Some packages are carried ON supports; additional packages of the SAME kind are
loose. The host supplies all counts and renders the source-supported equation.
Choose a plausible supportPackageType and relatedPackageType for ALL supplied
goods together. A support is a pallet, skid or rack, not an enclosing receptacle.
The related type must be a loose countable package, not another support, a bulk
loading mode, a vehicle or unpacked cargo. Do not change the goods or their form,
invent liquids/inner receptacles to rescue an impossible pairing, supply numeric
facts, or infer packing material from cargo material. Generic types are allowed.
This is extraction-training plausibility, not regulatory DG packing approval.
If no workable arrangement exists, return unknown=true and both types null.
Keep the reason short and audit-only. Return only the requested structured object.
"""

_SUPPORT_TYPES = ("PACKAGE_PALLET", "PACKAGE_PALLET_WOODEN", "PACKAGE_SKID", "PACKAGE_RACK")
_LOOSE_TYPES = tuple(
    category for category in RELATED_PACKAGE_TYPES if category not in _SUPPORT_TYPES
)
if TYPE_CHECKING:
    MixedLoosePackageType = str
else:
    MixedLoosePackageType = Literal.__getitem__(_LOOSE_TYPES)


class MixedPackingRequest(PackageGoodsContext):
    countedCategory: Literal["PACKAGE_PACKAGE"]


class MixedPackingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    unknown: bool
    supportPackageType: (
        Literal["PACKAGE_PALLET", "PACKAGE_PALLET_WOODEN", "PACKAGE_SKID", "PACKAGE_RACK"] | None
    )
    relatedPackageType: MixedLoosePackageType | None
    reason: str

    @model_validator(mode="after")
    def coherent_types(self) -> MixedPackingPlan:
        if not self.reason.strip():
            raise ValueError("mixed accounting requires an audit reason")
        if self.unknown:
            if self.supportPackageType is not None or self.relatedPackageType is not None:
                raise ValueError("UNKNOWN mixed plan must not supply package types")
        elif self.supportPackageType is None or self.relatedPackageType is None:
            raise ValueError("resolved mixed accounting needs both package types")
        return self


def mixed_packing_prompt(request: MixedPackingRequest, *, hs_codes: Container[str]) -> str:
    if any(goods.hs6 not in hs_codes for goods in request.goods):
        raise ValueError("mixed accounting goods must belong to the pinned HS registry")
    return (
        "Source-supported equation: declared PACKAGES = support units + loose packages; "
        "all related packages = packages on supports + loose packages.\n"
        + request.model_dump_json()
        + "\nAvailable package vocabulary:\n"
        + "\n".join(key + ": " + PACKAGE_ALIASES[key].singular for key in RELATED_PACKAGE_TYPES)
    )


def mixed_packing_facts(plan: MixedPackingPlan) -> dict[str, Any]:
    if plan.unknown:
        raise ValueError("UNKNOWN mixed accounting cannot become operative facts")
    return {
        "privatePackingRelation": "MIXED",
        "supportPackageType": plan.supportPackageType,
        "relatedPackageType": plan.relatedPackageType,
    }


def verify_mixed_plan_evidence(
    root: Path,
    frozen_request: Mapping[str, Any],
    frozen_plan: Mapping[str, Any],
    *,
    provider: Mapping[str, Any],
    hs_codes: Container[str],
) -> dict[str, Any]:
    from document_ocr.synthesis.independent_native_evidence import verify_native_evidence

    if frozen_plan.get("origin") != "native_mixed_accounting":
        raise ValueError("mixed accounting requires native semantic evidence")
    evidence = frozen_plan["nativePlanEvidence"]
    request = MixedPackingRequest.model_validate(evidence["request"])
    data = request.model_dump(mode="json")
    for key, value in {
        "goods": frozen_request["goods"],
        "regime": frozen_request["regime"],
        "form": frozen_request["goodsForm"],
        "dangerousGoods": frozen_request["dangerousGoods"],
        "countedCategory": frozen_request["selectedCategory"],
    }.items():
        if data[key] != value:
            raise ValueError("mixed native plan differs from frozen goods/package context")
    output = verify_native_evidence(
        root,
        evidence,
        stage="independent-mixed-accounting",
        identity=f"{frozen_request['sourceDocumentId']}:{frozen_request['variant']}",
        request=data,
        model=MixedPackingPlan,
        system=MIXED_PACKING_SYSTEM,
        prompt=mixed_packing_prompt(request, hs_codes=hs_codes),
        provider=provider,
    )
    facts = mixed_packing_facts(MixedPackingPlan.model_validate(output))
    if any(frozen_plan.get(key) != value for key, value in facts.items()):
        raise ValueError("mixed accounting differs from received native types")
    return facts
