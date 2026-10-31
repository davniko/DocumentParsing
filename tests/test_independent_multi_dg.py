"""Multi-DG components retain separate identities, requirements and evidence."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_independent_sampling import sampler as sampler

from document_ocr.synthesis.curated import Variable
from document_ocr.synthesis.curated_dg_wording import validate_product_ownership
from document_ocr.synthesis.independent_dg_group import components, refresh
from document_ocr.synthesis.independent_preparation import _dg_requirements


def group(sampler, count=3, extras=0):
    return sampler.prepare_request(
        source_document_id="multi-dg-test",
        variant=1,
        seed=20261029,
        regime="dg",
        identity_count=count + extras,
        dg_requirements=[{"packingGroup": True, "subsidiaryCount": 0}] * count,
    )


@pytest.mark.parametrize("count", [2, 3, 7])
def test_ordered_components_and_independent_native_scope(sampler, count):
    request = group(sampler, count)
    sampler.validate_request(request, require_context=False)
    assert len(components(request)) == len(request["dangerousGoods"]) == count
    assert len({r["recordId"] for r in request["dangerousGoods"]}) == count
    assert request == group(sampler, count)
    assert [c["sourceDocumentId"] for c in components(request)] == [
        f"multi-dg-test/dg/{i}" for i in range(count)
    ]


def test_optional_fields_are_requirements_per_declaration():
    fields = _dg_requirements(
        {
            "dangerousGoods": [
                {"unNumber": "1111"},
                {"unNumber": "2222", "packingGroupCategory": "LOW_DANGER"},
                {"unNumber": "3333", "subsidiaryHazardCategories": ["FLAMMABLE_LIQUIDS"]},
            ]
        },
        "dg",
    )
    assert fields == [
        {"packingGroup": False, "flashPoint": False},
        {"packingGroup": True, "flashPoint": False},
        {"packingGroup": False, "flashPoint": False, "subsidiaryCount": 1},
    ]


@pytest.mark.parametrize("field", ["unNumber", "properShippingName", "packingGroup"])
def test_parent_fact_tampering_is_rejected(sampler, field):
    request = deepcopy(group(sampler))
    request["dangerousGoods"][1][field] = "WRONG"
    with pytest.raises(ValueError, match="component projection"):
        sampler.validate_request(request, require_context=False)


def test_spare_hs_slots_do_not_create_unrelated_goods(sampler):
    with pytest.raises(ValueError, match="spare HS slots"):
        group(sampler, 2, 13)


def test_native_context_projection_does_not_change_other_components(sampler):
    request = group(sampler)
    original = deepcopy(request)
    request["dgComponents"][1]["goods"][0]["hs6"] = "123456"
    refresh(request)
    assert request["goods"][1]["hs6"] == "123456"
    assert request["goods"][0] == original["goods"][0]
    assert request["goods"][2] == original["goods"][2]
    assert request["dangerousGoods"] == original["dangerousGoods"]
    with pytest.raises((ValueError, KeyError)):
        sampler.validate_request(request)


@pytest.mark.parametrize("indices", [[True], [-1], [0, 0], ["0"]])
def test_product_indices_reject_ambiguous_contract_values(indices):
    with pytest.raises(ValueError):
        Variable(
            key="product",
            kind="product",
            value="GOODS",
            meaning="goods",
            required_literals=[],
            occurrences=[{"text": "GOODS", "occurrence": 1, "presentation": "text"}],
            dg_indices=indices,
        )


@pytest.mark.parametrize("indices", [None, [0], [0, 1, 2]])
def test_missing_or_out_of_range_product_ownership_fails_before_call(indices):
    bp = SimpleNamespace(
        target={"documentPatch": {"goodsItemDetails": [{"dangerousGoods": [{}, {}]}]}},
        contract=SimpleNamespace(variables=[SimpleNamespace(kind="product", dg_indices=indices)]),
    )
    with pytest.raises(ValueError, match="multi-DG product"):
        validate_product_ownership(bp)


def test_repeated_product_passages_can_own_the_same_dg():
    bp = SimpleNamespace(
        target={"documentPatch": {"goodsItemDetails": [{"dangerousGoods": [{}, {}]}]}},
        contract=SimpleNamespace(
            variables=[SimpleNamespace(kind="product", dg_indices=ids) for ids in ([0], [0], [1])]
        ),
    )
    validate_product_ownership(bp)


def test_mixed_group_gas_keeps_positive_receptacle_mass():
    from test_independent_unit_profiles import answer, request

    from document_ocr.synthesis.independent_unit_profiles import validate_profile

    req = request(
        regime="dg",
        selectedCategory="PACKAGE_PACKAGE",
        dangerousGoods=[{"recordId": "gas"}, {"recordId": "liquid"}],
        physicalState="mixed",
        componentPhysicalStates=["gas", "liquid"],
    )
    assert req.contains_gas
    value = answer()
    value["examples"][0]["packagingTareFraction"] = 0.0
    with pytest.raises(ValueError):
        validate_profile(req, value)


@pytest.mark.parametrize("forms,summary", [(["gas"], "gas"), (["gas", "liquid"], "liquid")])
def test_group_cannot_lose_or_misrepresent_component_physical_state(forms, summary):
    from test_independent_unit_profiles import request

    with pytest.raises(ValueError, match=r"physical|componentPhysicalStates"):
        request(
            regime="dg",
            dangerousGoods=[{}, {}],
            physicalState=summary,
            componentPhysicalStates=forms,
        )


def test_group_cannot_plan_unpacked_gas():
    from test_independent_unit_profiles import request

    with pytest.raises(ValueError, match="gas cargo needs contained"):
        request(
            regime="dg",
            dangerousGoods=[{}, {}],
            physicalState="mixed",
            componentPhysicalStates=["gas", "solid"],
            countedUnitContext="bulk_loading_lot",
        )
