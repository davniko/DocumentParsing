"""Reviewed source package physics is distinct from generic DG packaging."""

from copy import deepcopy
from decimal import Decimal

import pytest
from test_curated_dg_presentations import record
from test_curated_scenarios import catalog, source_row

from document_ocr.synthesis.curated_scenarios import (
    CandidateIncompatibility,
    SourceCapabilities,
    _observation,
)
from document_ocr.synthesis.generators import DeterministicStream


def tote_source():
    """Exact five-bin accounting from the newly reviewed 45386289 source."""
    row = source_row(quantity=5)
    patch = row["target"]["documentPatch"]
    goods = patch["goodsItemDetails"][0]
    goods.update(
        description="PROPIONIC ACID, FOOD GRADE, 1016 KG (2240 LB), TOTE BIN",
        grossWeight={"value": 5365.765, "unit": "kilogram"},
        netWeight={"value": 5080.0, "unit": "kilogram"},
        volume={"value": 7.236, "unit": "cubic_metre"},
        dangerousGoods=[{"unNumber": "3463", "hazardCategory": "CORROSIVE_SUBSTANCES"}],
    )
    goods["numberAndTypeOfPackages"][0]["typeCategory"] = "PACKAGE_BIN"
    goods["splitGoodsPlacement"] = [{"equipmentIdentifier": "MSCU1234567", "packageQuantity": 5}]
    patch["containerInformation"] = [
        {
            "equipmentIdentifier": "MSCU1234567",
            "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
            "typeCategory": "GENERAL_PURPOSE",
        }
    ]
    return row


def dg_support(row, **overrides):
    support = catalog([row], **overrides)
    support.chemical_candidates = ((record(), "520512"),)
    return support


def test_reviewed_dg_source_bundle_preserves_tote_accounting_and_printed_precision():
    row = tote_source()
    before = deepcopy(row)
    support = dg_support(row)
    cap = SourceCapabilities(family="dg_chemical", physical_profile="source_bundle")
    quantities = set()
    for variant in range(1, 13):
        scenario = support.sample(row, seed=13, variant=variant, capabilities=cap)
        goods = scenario.cargo["goodsItemDetails"][0]
        quantity = goods["numberAndTypeOfPackages"][0]["packageQuantity"]
        quantities.add(quantity)
        assert goods["numberAndTypeOfPackages"][0]["typeCategory"] == "PACKAGE_BIN"
        assert goods["splitGoodsPlacement"][0]["packageQuantity"] == quantity
        assert Decimal(str(goods["netWeight"]["value"])) == Decimal(1016) * quantity
        assert Decimal(str(goods["grossWeight"]["value"])) == Decimal("1073.153") * quantity
        expected_cube = (Decimal("1.4472") * quantity).quantize(Decimal(".001"))
        assert Decimal(str(goods["volume"]["value"])) == expected_cube
        assert scenario.provenance["donorDocumentId"] == row["documentId"]
        assert scenario.provenance["physicalProfile"] == "source_bundle"
        assert not scenario.provenance["rejectedCandidates"]
    assert len(quantities) > 1
    assert row == before


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("grossWeight", 2500.0, "package mass"),
        ("volume", 3.0, "package cube"),
    ],
)
def test_generic_dg_packaging_keeps_existing_non_bulk_envelope(field, value, expected):
    row = tote_source()
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    goods.update(
        grossWeight={"value": 1000.0, "unit": "kilogram"},
        netWeight={"value": 900.0, "unit": "kilogram"},
        volume={"value": 1.0, "unit": "cubic_metre"},
    )
    goods[field]["value"] = value
    support = dg_support(row)
    cap = SourceCapabilities(family="dg_chemical", fixed_package_quantity=5)
    source = _observation(row, "dg_chemical")
    identities, dg_record = support._identities(
        source, source, cap, DeterministicStream(13, "test", "source")
    )
    with pytest.raises(CandidateIncompatibility, match=expected):
        support._cargo(
            source, source, cap, identities, dg_record, DeterministicStream(13, "test", "source")
        )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("grossWeight", 30000.0, "payload capacity"),
        ("volume", 34.0, "enclosed volume"),
    ],
)
def test_reviewed_dg_source_bundle_still_enforces_equipment_capacity(field, value, expected):
    row = tote_source()
    row["target"]["documentPatch"]["goodsItemDetails"][0][field]["value"] = value
    support = dg_support(row)
    cap = SourceCapabilities(
        family="dg_chemical", physical_profile="source_bundle", fixed_package_quantity=5
    )
    source = _observation(row, "dg_chemical")
    identities, dg_record = support._identities(
        source, source, cap, DeterministicStream(13, "test", "source")
    )
    with pytest.raises(CandidateIncompatibility, match=expected):
        support._cargo(
            source, source, cap, identities, dg_record, DeterministicStream(13, "test", "source")
        )


def test_dg_source_bundle_cannot_substitute_a_package_outside_its_source_domain():
    row = tote_source()
    support = dg_support(row)
    cap = SourceCapabilities(
        family="dg_chemical",
        physical_profile="source_bundle",
        allowed_package_categories=("PACKAGE_DRUM",),
    )
    with pytest.raises(ValueError, match="no complete train-only cargo donors"):
        support.sample(row, seed=13, variant=1, capabilities=cap)


def test_one_tote_net_only_cargo_preserves_source_fields_without_inventing_equipment():
    """0b7f3731's net-only physics and exact source equipment visibility."""
    row = tote_source()
    patch = row["target"]["documentPatch"]
    goods = patch["goodsItemDetails"][0]
    goods["netWeight"]["value"] = 1016.0
    del goods["grossWeight"], goods["volume"], goods["splitGoodsPlacement"]
    goods["numberAndTypeOfPackages"][0]["packageQuantity"] = 1
    patch["containerInformation"] = [{"equipmentIdentifier": "TCKU6384218"}]
    support = dg_support(row)
    source = _observation(row, "dg_chemical")
    cap = SourceCapabilities(
        family="dg_chemical", physical_profile="source_bundle", fixed_package_quantity=1
    )
    stream = DeterministicStream(13, "test", "source")
    identities, dg_record = support._identities(source, source, cap, stream)
    cargo, receipt = support._cargo(source, source, cap, identities, dg_record, stream)
    assert cargo["goodsItemDetails"][0]["netWeight"]["value"] == 1016.0
    assert cargo["goodsItemDetails"][0]["numberAndTypeOfPackages"][0] == {
        "packageQuantity": 1,
        "typeCategory": "PACKAGE_BIN",
    }
    assert set(cargo["goodsItemDetails"][0]) == set(goods)
    assert cargo["containerInformation"] == patch["containerInformation"]
    assert receipt["physicalProfile"] == "source_bundle"
    sampled = support.sample(row, seed=13, variant=1, capabilities=cap)
    assert sampled.cargo["containerInformation"] == patch["containerInformation"]
    assert sampled.cargo["goodsItemDetails"][0]["netWeight"] == goods["netWeight"]
    assert set(sampled.cargo["goodsItemDetails"][0]) == set(goods)
    assert sampled.provenance["physicalRows"][0]["capacity"] == {
        "payloadKg": "47300.0",
        "volumeM3": "86.0",
    }


@pytest.mark.parametrize("profile", ["observed_train_bundle", "source_bundle"])
def test_incomplete_equipment_donor_never_claims_an_explicit_allowed_pair(profile):
    row = tote_source()
    del row["target"]["documentPatch"]["containerInformation"][0]["typeCategory"]
    support = dg_support(row)
    cap = SourceCapabilities(
        family="dg_chemical",
        physical_profile=profile,
        allowed_equipment_pairs=("TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE",),
    )
    with pytest.raises(ValueError, match="no complete train-only cargo donors"):
        support.sample(row, seed=13, variant=1, capabilities=cap)


def test_unknown_equipment_never_enters_generic_dg_transferable_donor_pool():
    row = tote_source()
    del row["target"]["documentPatch"]["containerInformation"][0]["typeCategory"]
    support = dg_support(row)
    with pytest.raises(ValueError, match="no complete train-only cargo donors"):
        support.sample(
            row, seed=13, variant=1, capabilities=SourceCapabilities(family="dg_chemical")
        )


def test_generic_dg_package_and_random_stream_are_unchanged():
    row = source_row()
    row["target"]["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"] = [
        {"unNumber": "1824", "hazardCategory": "CORROSIVE_SUBSTANCES"}
    ]
    support = dg_support(row)
    cap = SourceCapabilities(family="dg_chemical")
    for variant in range(1, 13):
        result = support.sample(row, seed=13, variant=variant, capabilities=cap)
        attempt = result.provenance["candidateAttempts"] - 1
        stream = (
            DeterministicStream(13, "curated-v7-scenarios-v1", f"{row['documentId']}:{variant}")
            .derive(f"cargo-{attempt}")
            .derive("chemical-packaging")
        )
        categories = ("PACKAGE_DRUM", "PACKAGE_CARTON", "PACKAGE_BOX")
        assert (
            result.cargo["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["typeCategory"]
            == categories[stream.randbelow(len(categories))]
        )
