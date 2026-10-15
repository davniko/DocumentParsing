"""DG vehicle propulsion follows the source UN tuple and authoritative HS domain."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_curated_scenarios import catalog, source_row

from document_ocr.synthesis.curated_scenarios import SourceCapabilities

HAZARD = "MISCELLANEOUS_DANGEROUS_SUBSTANCES_AND_ARTICLES"


def hmt(number, phrase, hazard=HAZARD):
    return SimpleNamespace(
        un_number=number,
        proper_shipping_name=phrase,
        maritime_eligible=True,
        technical_name_required=False,
        record_id="un" + number + phrase,
        hazard_category=hazard,
        packing_group_category=None,
        packing_group_code=None,
        exact_hazard_class="9",
        exact_subsidiary_hazards=(),
        subsidiary_hazard_categories=(),
    )


def source_vehicle(number="3171", code="8703800010"):
    row = source_row(code=code, quantity=3)
    patch = row["target"]["documentPatch"]
    patch["containerInformation"] = patch["containerInformation"][:1]
    goods = patch["goodsItemDetails"][0]
    goods.update(
        description="BATTERY-ELECTRIC PERSONAL CARS"
        if number == "3171"
        else "PETROL PERSONAL CARS",
        dangerousGoods=[{"unNumber": number, "hazardCategory": HAZARD}],
        grossWeight={"value": 6305.0, "unit": "kilogram"},
        volume={"value": 68.0, "unit": "cubic_metre"},
        splitGoodsPlacement=[{"equipmentIdentifier": "MSCU1234567", "packageQuantity": 3}],
    )
    goods.pop("netWeight")
    goods["numberAndTypeOfPackages"][0]["typeCategory"] = "PACKAGE_UNIT"
    return row


def authority(row):
    support = catalog(
        [row],
        maximum_candidates=4,
        phrases={
            "870323": "Passenger motor cars, spark-ignition, engine 1,500-3,000 cc",
            "870380": "Battery-electric passenger vehicles",
            "870340": "Hybrid electric passenger vehicles with petrol engines",
        },
    )
    support.dg.hmt_records = (
        hmt("3166", "Vehicle, flammable gas powered"),
        hmt("3166", "Vehicle, flammable liquid powered"),
        hmt("3171", "Battery-powered equipment"),
        hmt("3171", "Battery-powered vehicle or Battery-powered equipment"),
    )
    return support


def capability(code="870380"):
    return SourceCapabilities(
        family="dg_vehicle",
        allowed_hs_headings=("8703",),
        allowed_hs_codes=(code,),
        identity_count=1,
        physical_profile="source_whole_units",
        fixed_package_quantity=3,
    )


def test_three_battery_vehicles_use_source_un_and_electric_hs_without_changing_load():
    row = source_vehicle()
    original = deepcopy(row)
    support = authority(row)
    cap = capability()
    result = support.sample(row, seed=20261014, variant=1, capabilities=cap)
    assert result == support.sample(row, seed=20261014, variant=1, capabilities=cap)
    goods = result.cargo["goodsItemDetails"][0]
    assert goods["dangerousGoods"] == [{"unNumber": "3171", "hazardCategory": HAZARD}]
    assert result.goods_identities[0].hs6 == "870380"
    assert len(goods["hsCodes"][0]) == 10 and goods["hsCodes"][0].startswith("870380")
    assert (
        goods["grossWeight"]
        == original["target"]["documentPatch"]["goodsItemDetails"][0]["grossWeight"]
    )
    assert goods["volume"]["value"] == 68
    assert goods["numberAndTypeOfPackages"] == [
        {"packageQuantity": 3, "typeCategory": "PACKAGE_UNIT"}
    ]
    assert goods["splitGoodsPlacement"][0]["packageQuantity"] == 3
    assert row == original
    assert result.provenance["dgPrintedFacts"]["class"] == "9"
    assert "vehicle or" in result.provenance["dgPrintedFacts"]["properShippingName"]


def test_liquid_fuel_vehicle_keeps_its_existing_domain_and_replay():
    row = source_vehicle("3166", "8703231900")
    support = authority(row)
    result = support.sample(row, seed=42, variant=1, capabilities=capability("870323"))
    assert result.goods_identities[0].hs6 == "870323"
    assert (
        result.provenance["dgPrintedFacts"]["properShippingName"]
        == "Vehicle, flammable liquid powered"
    )
    assert result.cargo["goodsItemDetails"][0]["dangerousGoods"][0]["unNumber"] == "3166"


@pytest.mark.parametrize(
    ("number", "code"), [("3171", "870323"), ("3166", "870380"), ("3171", "870340")]
)
def test_un_and_hs_propulsion_mismatch_cannot_escape_through_another_donor(number, code):
    row = source_vehicle(number)
    with pytest.raises(ValueError, match="no physically compatible scenario"):
        authority(row).sample(row, seed=19, variant=1, capabilities=capability(code))


@pytest.mark.parametrize(
    "description", ["FLAMMABLE GAS POWERED CAR", "LPG CAR", "CNG CAR", "HYBRID CAR"]
)
def test_explicit_unsupported_propulsion_is_not_reinterpreted_as_petrol(description):
    row = source_vehicle("3166", "8703231900")
    row["target"]["documentPatch"]["goodsItemDetails"][0]["description"] = description
    with pytest.raises(ValueError, match="propulsion is outside"):
        authority(row).sample(row, seed=19, variant=1, capabilities=capability("870323"))


def test_battery_equipment_only_registry_row_is_not_a_vehicle_fact():
    row = source_vehicle()
    support = authority(row)
    support.dg.hmt_records = (hmt("3171", "Battery-powered equipment"),)
    with pytest.raises(ValueError, match="no exact supported HMT"):
        support.sample(row, seed=19, variant=1, capabilities=capability())


def test_source_class_conflict_is_rejected_before_candidate_retries():
    row = source_vehicle()
    row["target"]["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"][0]["hazardCategory"] = (
        "CORROSIVE_SUBSTANCES"
    )
    with pytest.raises(ValueError, match="UN/class/propulsion tuple"):
        authority(row).sample(row, seed=19, variant=1, capabilities=capability())
