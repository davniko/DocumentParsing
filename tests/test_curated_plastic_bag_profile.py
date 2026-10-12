"""Reviewed plastic-bag form data and its unchanged material/measure boundaries."""

from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest
from test_curated_scenarios import catalog, config, source_row

from document_ocr.synthesis.curated_load_profiles import RegistryLoadProfiles
from document_ocr.synthesis.curated_scenarios import ScenarioCatalog, SourceCapabilities
from document_ocr.synthesis.package_aliases import PACKAGE_ALIASES


def profiles():
    path = Path(__file__).resolve().parents[1] / "artifacts/sampling-registries/load-profiles.json"
    return RegistryLoadProfiles.model_validate_json(path.read_bytes())


def plastic_catalog(row):
    base = catalog(
        [row],
        phrases={"520511": "Cotton yarn", "390110": "Polyethylene", "390120": "Polyethylene"},
    )
    selected = profiles().model_copy(
        update={"profiles": tuple(p for p in profiles().profiles if p.id == "plastic_bag_granules")}
    )
    return ScenarioCatalog(
        config=config(load_profiles={"path": "fixture", "sha256": "0" * 64}),
        countries=base.countries,
        hs=base.hs,
        phrases=base.phrases,
        ports=base.ports,
        localities=base.localities,
        dg=base.dg,
        train_rows=[row],
        validation_ids=frozenset(),
        load_profiles=selected,
    )


def test_plastic_bag_profile_preserves_reviewed_form_and_exact_envelope():
    loaded = profiles()
    plastic = next(p for p in loaded.profiles if p.id == "plastic_bag_granules")
    jumbo = next(p for p in loaded.profiles if p.id == "jumbo_bag_granules")
    assert plastic.package_category == "PACKAGE_BAG_PLASTIC"
    assert PACKAGE_ALIASES[plastic.package_category].application_code == "EC"
    assert plastic.hs_headings == jumbo.hs_headings == ("3901",)
    assert plastic.form == jumbo.form
    assert plastic.unit_net_kg == Decimal(198000) / 7920 == Decimal(140000) / 5600 == 25
    assert plastic.unit_gross_kg == 26
    assert plastic.unit_volume_m3 == jumbo.unit_volume_m3 * plastic.unit_net_kg / jumbo.unit_net_kg
    assert (plastic.quantity_min, plastic.quantity_max) == (1, 1000)


def test_plastic_form_sampling_is_donor_independent_and_conserves_declared_units():
    row = source_row()
    original = deepcopy(row)
    support = plastic_catalog(row)
    cap = SourceCapabilities(
        family="ambient",
        physical_profile="registry_form",
        allowed_package_categories=("PACKAGE_BAG_PLASTIC",),
    )
    for variant in range(1, 31):
        scenario = support.sample(row, seed=31, variant=variant, capabilities=cap)
        goods = scenario.cargo["goodsItemDetails"][0]
        package = goods["numberAndTypeOfPackages"][0]
        quantity = package["packageQuantity"]
        assert 2 <= quantity <= 1000
        assert package["typeCategory"] == "PACKAGE_BAG_PLASTIC"
        assert sum(a["packageQuantity"] for a in goods["splitGoodsPlacement"]) == quantity
        assert goods["grossWeight"]["value"] == 26 * quantity
        assert goods["netWeight"]["value"] == 25 * quantity
        assert scenario.provenance["registryLoadProfile"]["id"] == "plastic_bag_granules"
        assert "donorDocumentId" not in scenario.provenance
    assert row == original


def test_plastic_form_does_not_relax_a_different_material_cap():
    row = source_row()
    with pytest.raises(ValueError, match="no reviewed registry-form support"):
        plastic_catalog(row).sample(
            row,
            seed=31,
            variant=1,
            capabilities=SourceCapabilities(
                family="ambient",
                physical_profile="registry_form",
                allowed_package_categories=("PACKAGE_BAG_PAPER",),
            ),
        )


def test_private_plastic_form_measures_do_not_add_public_labels():
    row = source_row()
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    del goods["netWeight"], goods["volume"]
    scenario = plastic_catalog(row).sample(
        row,
        seed=31,
        variant=1,
        capabilities=SourceCapabilities(
            family="ambient",
            physical_profile="registry_form",
            allowed_package_categories=("PACKAGE_BAG_PLASTIC",),
        ),
    )
    goods = scenario.cargo["goodsItemDetails"][0]
    assert "netWeight" not in goods and "volume" not in goods
    assert set(scenario.provenance["explicitPhysicalFacts"]) == {
        "grossWeight",
        "netWeight",
        "volume",
    }
