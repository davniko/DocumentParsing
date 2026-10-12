"""Donor-independent form/load sampling and its physical authority boundary."""

from copy import deepcopy
from decimal import Decimal

import pytest
from test_curated_scenarios import catalog, config, source_row

from document_ocr.synthesis.curated_load_profiles import RegistryLoadProfiles
from document_ocr.synthesis.curated_scenarios import ScenarioCatalog, SourceCapabilities


def profile_data():
    return {
        "version": 1,
        "profiles": [
            {
                "id": "yarn",
                "hs_headings": ["5205"],
                "form": "DRY COTTON YARN",
                "package_category": "PACKAGE_CARTON",
                "unit_gross_kg": "30",
                "unit_net_kg": "28",
                "unit_volume_m3": "0.1",
                "quantity_min": 1,
                "quantity_max": 100,
                "rationale": "Test synthetic per-declared-unit envelope.",
            }
        ],
    }


def registry_catalog(row, *, profiles_data=None, **kwargs):
    base = catalog([row])
    return ScenarioCatalog(
        config=config(load_profiles={"path": "fixture", "sha256": "0" * 64}, **kwargs),
        countries=base.countries,
        hs=base.hs,
        phrases=base.phrases,
        ports=base.ports,
        localities=base.localities,
        dg=base.dg,
        train_rows=[row],
        validation_ids=frozenset(),
        load_profiles=RegistryLoadProfiles.model_validate(
            profile_data() if profiles_data is None else profiles_data
        ),
    )


def test_registry_draw_has_no_observed_donor_and_preserves_allocations():
    row = source_row()
    before = deepcopy(row)
    support = registry_catalog(row, equipment_weights={"TWENTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE": 1})
    cap = SourceCapabilities(family="ambient", physical_profile="registry_form")
    for variant in range(1, 50):
        scenario = support.sample(row, seed=31, variant=variant, capabilities=cap)
        assert scenario == support.sample(row, seed=31, variant=variant, capabilities=cap)
        goods = scenario.cargo["goodsItemDetails"][0]
        q = goods["numberAndTypeOfPackages"][0]["packageQuantity"]
        assert q >= 2
        assert sum(a["packageQuantity"] for a in goods["splitGoodsPlacement"]) == q
        assert goods["grossWeight"]["value"] == 30 * q
        assert goods["netWeight"]["value"] == 28 * q
        assert "donorDocumentId" not in scenario.provenance
        assert scenario.provenance["registryLoadProfile"]["id"] == "yarn"
        assert all(
            c["sizeCategory"] == "TWENTY_FOOT_HIGH_CUBE"
            for c in scenario.cargo["containerInformation"]
        )
    assert row == before


def test_no_public_measure_is_added_by_a_private_profile_fact():
    row = source_row()
    row["target"]["documentPatch"]["goodsItemDetails"][0].pop("netWeight")
    scenario = registry_catalog(row).sample(
        row,
        seed=7,
        variant=1,
        capabilities=SourceCapabilities(family="ambient", physical_profile="registry_form"),
    )
    assert "netWeight" not in scenario.cargo["goodsItemDetails"][0]
    assert "netWeight" in scenario.provenance["explicitPhysicalFacts"]


def test_registry_quantity_interval_accounts_for_selected_equipment_capacity():
    row = source_row()
    data = profile_data()
    data["profiles"][0]["quantity_max"] = 100000
    support = registry_catalog(
        row,
        profiles_data=data,
        equipment_weights={"TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 1},
    )
    cap = SourceCapabilities(family="ambient", physical_profile="registry_form")
    for variant in range(1, 20):
        scenario = support.sample(row, seed=31, variant=variant, capabilities=cap)
        q = scenario.cargo["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["packageQuantity"]
        assert 2 <= q <= int(Decimal("33.2") / (Decimal("0.1") * Decimal("0.6")))


def test_alias_policy_requires_an_explicit_package_registry_count():
    from document_ocr.synthesis.curated_campaign import AliasSamplingPolicy

    with pytest.raises(ValueError, match="explicit entry count"):
        AliasSamplingPolicy(package_registry={"path": "fixture", "sha256": "0" * 64})


@pytest.mark.parametrize(
    "weights",
    [
        {"package_weights": {"PACKAGE_DRUM": 1}},
        {"profile_weights": {"missing": 1}},
        {"equipment_weights": {"FORTY_FOOT_STANDARD_HEIGHT|REFRIGERATED": 1}},
    ],
)
def test_unknown_distribution_support_is_not_a_fallback(weights):
    with pytest.raises(ValueError, match=r"support|GP pairs"):
        registry_catalog(source_row(), **weights)


@pytest.mark.parametrize(
    "changes",
    [
        {"family": "frozen"},
        {"family": "dg_chemical"},
    ],
)
def test_thermal_and_dg_do_not_enter_ambient_form_sampler(changes):
    with pytest.raises(ValueError, match="registry-form"):
        SourceCapabilities(physical_profile="registry_form", **changes)


@pytest.mark.parametrize("hazard", ["dg", "thermal"])
def test_source_facts_cannot_be_bypassed_by_an_ambient_capability(hazard):
    row = source_row()
    if hazard == "dg":
        row["target"]["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"] = [
            {"unNumber": "1133"}
        ]
    else:
        row["target"]["documentPatch"]["containerInformation"][0]["temperatureSetpoint"] = {
            "value": -20,
            "unit": "celsius",
        }
    with pytest.raises(ValueError, match="registry-form source has non-ambient"):
        registry_catalog(row).sample(
            row,
            seed=7,
            variant=1,
            capabilities=SourceCapabilities(family="ambient", physical_profile="registry_form"),
        )


def test_source_category_domain_is_enforced_without_resampling_into_another_domain():
    row = source_row()
    with pytest.raises(ValueError, match="no reviewed registry-form support"):
        registry_catalog(row).sample(
            row,
            seed=7,
            variant=1,
            capabilities=SourceCapabilities(
                family="ambient",
                physical_profile="registry_form",
                allowed_package_categories=("PACKAGE_DRUM",),
            ),
        )


def test_profile_validator_rejects_bad_physics_and_duplicate_ids():
    data = profile_data()
    data["profiles"][0]["unit_net_kg"] = "31"
    with pytest.raises(ValueError, match="net mass"):
        RegistryLoadProfiles.model_validate(data)
    data = profile_data()
    data["profiles"].append(deepcopy(data["profiles"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        RegistryLoadProfiles.model_validate(data)


def test_each_profile_heading_must_exist_in_authoritative_registry():
    data = profile_data()
    data["profiles"][0]["hs_headings"].append("9999")
    with pytest.raises(ValueError, match="absent authoritative HS registry headings: 9999"):
        registry_catalog(source_row(), profiles_data=data)


def test_duplicate_profile_headings_are_rejected_before_identity_sampling():
    data = profile_data()
    data["profiles"][0]["hs_headings"].append("5205")
    with pytest.raises(ValueError, match="duplicate load-profile HS4 headings"):
        RegistryLoadProfiles.model_validate(data)


def test_physical_compiler_rejects_tampered_registry_ledger():
    from test_curated_physical import fixture

    from document_ocr.synthesis.curated_physical import prepare_physical_render

    row, bp, scenario, physical_support = fixture()
    profile = profile_data()["profiles"][0]
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    goods["numberAndTypeOfPackages"][0]["typeCategory"] = "PACKAGE_CARTON"
    q = goods["numberAndTypeOfPackages"][0]["packageQuantity"]
    scenario.provenance.pop("donorDocumentId")
    for physical_row in scenario.provenance["physicalRows"]:
        physical_row["capacity"]["payloadKg"] = "10000"
    scenario.provenance.update(
        registryLoadProfile=profile,
        explicitPhysicalFacts={
            "grossWeight": {"value": str(30 * q), "unit": "kilogram"},
            "netWeight": {"value": str(28 * q), "unit": "kilogram"},
            "volume": {"value": str(q / 10), "unit": "cubic_metre"},
        },
    )
    # Decimal textual identity belongs to the receipt contract too.
    scenario.provenance["explicitPhysicalFacts"]["volume"]["value"] = str(Decimal("0.1") * q)
    prepare_physical_render(bp, scenario, row["target"], support=physical_support)
    scenario.provenance["explicitPhysicalFacts"]["netWeight"]["value"] = "999999"
    with pytest.raises(ValueError, match="explicit physical facts"):
        prepare_physical_render(bp, scenario, row["target"], support=physical_support)
