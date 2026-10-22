from __future__ import annotations

import pytest

from document_ocr.synthesis.independent_accounting import (
    MixedPackingPlan,
    MixedPackingRequest,
    mixed_packing_facts,
    mixed_packing_prompt,
)
from document_ocr.synthesis.independent_envelopes import PACKAGE_COUNTING_CONTEXTS
from document_ocr.synthesis.independent_relationships import (
    RELATED_PACKAGE_TYPES,
    PrivatePackingPlan,
    validate_packing_direction,
)


def test_native_mixed_choice_owns_types_only_not_counts_or_reasons():
    plan = MixedPackingPlan(
        unknown=False,
        supportPackageType="PACKAGE_PALLET",
        relatedPackageType="PACKAGE_BALE",
        reason="Plausible maritime packing",
    )
    assert mixed_packing_facts(plan) == {
        "privatePackingRelation": "MIXED",
        "supportPackageType": "PACKAGE_PALLET",
        "relatedPackageType": "PACKAGE_BALE",
    }
    request = MixedPackingRequest(
        goods=[{"hs6": "520100", "description": "Cotton, not carded"}],
        regime="ambient",
        equipment="Maritime carrying container",
        countedCategory="PACKAGE_PACKAGE",
    )
    assert "520100" in mixed_packing_prompt(request, hs_codes={"520100"})
    with pytest.raises(ValueError, match="pinned HS registry"):
        mixed_packing_prompt(request, hs_codes={"520200"})
    with pytest.raises(ValueError):
        MixedPackingRequest.model_validate(
            {**request.model_dump(), "countedCategory": "PACKAGE_BALE"}
        )


@pytest.mark.parametrize(
    "related",
    ["PACKAGE_PALLET", "PACKAGE_BULK_LIQUID", "PACKAGE_VEHICLE", "PACKAGE_UNPACKED_OR_UNPACKAGED"],
)
def test_mixed_equation_rejects_nonpackage_loose_component(related):
    with pytest.raises(ValueError):
        MixedPackingPlan(
            unknown=False,
            supportPackageType="PACKAGE_PALLET",
            relatedPackageType=related,
            reason="Not a valid mixed package plan",
        )


def test_unknown_is_not_coerced_to_resolved_and_host_counts_are_forbidden():
    plan = MixedPackingPlan(
        unknown=True,
        supportPackageType=None,
        relatedPackageType=None,
        reason="Goods cannot be carried in this presentation",
    )
    with pytest.raises(ValueError, match="UNKNOWN"):
        mixed_packing_facts(plan)
    with pytest.raises(ValueError):
        MixedPackingPlan.model_validate({**plan.model_dump(), "packageQuantity": 12})
    with pytest.raises(ValueError):
        MixedPackingPlan(
            unknown=True,
            supportPackageType="PACKAGE_PALLET",
            relatedPackageType=None,
            reason="Unresolved",
        )


@pytest.mark.parametrize(
    "category", [c for c, contexts in PACKAGE_COUNTING_CONTEXTS.items() if contexts]
)
def test_related_package_schema_cannot_number_bulk_or_unpacked_contents(category):
    assert category not in RELATED_PACKAGE_TYPES
    with pytest.raises(ValueError):
        PrivatePackingPlan(
            privatePackingRelation="CONTAINS",
            relatedPackageType=category,
            reason="Not an actual inner package",
        )
    with pytest.raises(ValueError, match="counting context"):
        validate_packing_direction("PACKAGE_DRUM", "CONTAINS", category)
    direct = PrivatePackingPlan(
        privatePackingRelation="NONE",
        relatedPackageType=None,
        reason="The drum directly contains the liquid",
    )
    assert direct.relatedPackageType is None
