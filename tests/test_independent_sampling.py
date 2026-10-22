"""Full-registry, source-cargo-independent shipment fact sampling."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from document_ocr.synthesis.independent_sampling import IndependentSampler

ROOT = Path(__file__).resolve().parents[1]
HS = "artifacts/registries/sources/hs/ukgt-v4.0.1590"
DG = "artifacts/registries/compiled/dangerous-goods-phmsa-ecics-20260831-v4"
PINS = {
    "hsManifest": (
        HS + "/source-manifest.json",
        "7dec14e2d9c3979ad5f40f75a490a3a68e4a8382bad8691784f3facafb2f8a4e",
    ),
    "hsMetadata": (
        HS + "/raw/metadata.csvw.json",
        "4e59b782504796c4b027d242e4f3136ee10e7f21f7425f21359552cb12646bb7",
    ),
    "hsReport": (
        HS + "/raw/commodities-report.csv",
        "13fa169f822ec9e985ffe0e18af2557e3e09ffb4c58de9499a4c8b3df52c8513",
    ),
    "phrases": (
        "artifacts/registries/phrases/hs6-commercial-phrases-ukgt-v4.0.1590-v1/phrases.jsonl",
        "05340e29eabf21e9cae56d28287834e6bfa1a5791e9311cb889031affb0f3078",
    ),
    "dgHmt": (
        DG + "/hmt-records.jsonl",
        "bb581d3f1c60f46aa82d13f171f78e61124bbf7d7e83b7295bb970cd69305cb2",
    ),
    "dgEcics": (
        DG + "/ecics-links.jsonl",
        "51e23d0de3437276a8c1eaa62835705212f0154259de955b2139287fc60ac15f",
    ),
    "flash": (
        "artifacts/sampling-registries/niosh-flashpoints-20261014/registry.json",
        "ddc4f1bbe438d7ad58825188d91790d0e63c31891d0f4f90c33de8c9a20329f3",
    ),
}


def registry_config():
    return {key: {"path": path, "sha256": digest} for key, (path, digest) in PINS.items()}


@pytest.fixture(scope="module")
def sampler():
    return IndependentSampler.from_config(ROOT, registry_config())


def ambient_request(sampler, *, variant=1, mode="uniform"):
    return sampler.prepare_request(
        source_document_id="presentation-only-test-source",
        variant=variant,
        regime="ambient",
        identity_count=1,
        seed=20261021,
        equipment_mode=mode,
    )


def frozen_package(request, category, context=None):
    selected = deepcopy(request)
    selected["selectedCategory"] = category
    selected["countedUnitContext"] = context
    return selected


def test_requires_explicit_exact_registry_file_pins():
    config = registry_config()
    config["phrases"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from configured pin"):
        IndependentSampler.from_config(ROOT, config)
    config = registry_config()
    config.pop("flash")
    with pytest.raises(ValueError, match="seven exact file pins"):
        IndependentSampler.from_config(ROOT, config)


def test_entire_hs_registry_and_exact_joined_dg_are_eligible(sampler):
    support = sampler.support()
    assert support["registryHs6"] == 5612
    assert sum(support["hs6ByRegime"].values()) == 5612
    assert set().union(*(set(codes) for codes in sampler.pools.values())) == set(sampler.phrases)
    assert support["joinedDgRecords"] == 384
    assert support["joinedDgLinkRows"] == 386
    assert support["joinedDgPairs"] == 384
    assert support["joinedDgHs6"] == 138
    assert support["sourceCargoAccess"] is False


def test_explicit_vehicle_presentation_reaches_registry_not_original_source(sampler):
    from document_ocr.synthesis.independent_cargo_presentation import CargoPresentation

    domain = {
        "hsHeadings": ["8701", "8702", "8703", "8704", "8705", "8711", "8716"],
        "excludedHs6": ["871690"],
        "packageTypes": ["PACKAGE_UNPACKED_OR_UNPACKAGED"],
    }
    seen = set()
    for variant in range(1, 31):
        request = sampler.prepare_request(
            source_document_id="vehicle-presentation",
            variant=variant,
            regime="ambient",
            identity_count=1,
            seed=20261021,
            cargo_presentation=domain,
        )
        sampler.validate_request(request)
        code = request["goods"][0]["hs6"]
        assert CargoPresentation.model_validate(domain).permits(code)
        assert request["registrySelection"]["eligibleIdentities"] > 20
        seen.add(code)
    assert len(seen) > 15
    request["goods"][0] = {"hs6": "853921", "description": sampler.phrases["853921"]}
    with pytest.raises(ValueError, match="goods violate"):
        sampler.validate_request(request)


def test_empty_presentation_registry_domain_does_not_fallback(sampler):
    with pytest.raises(ValueError):
        sampler.prepare_request(
            source_document_id="impossible-presentation",
            variant=1,
            regime="ambient",
            identity_count=1,
            seed=1,
            cargo_presentation={
                "hsHeadings": ["0000"],
                "excludedHs6": [],
                "packageTypes": ["PACKAGE_CARTON"],
            },
        )


def test_reviewed_dg_domain_keeps_exact_join_and_printed_field_requirements(sampler):
    initial = sampler.prepare_request(
        source_document_id="dg-presentation",
        variant=1,
        regime="dg",
        identity_count=1,
        seed=20261021,
    )
    domain = {
        "hsHeadings": [initial["goods"][0]["hs6"][:4]],
        "excludedHs6": [],
        "packageTypes": ["PACKAGE_DRUM"],
    }
    for variant in range(1, 5):
        request = sampler.prepare_request(
            source_document_id="dg-presentation",
            variant=variant,
            regime="dg",
            identity_count=1,
            seed=20261021,
            cargo_presentation=domain,
        )
        sampler.validate_request(request)
        assert request["goods"][0]["hs6"][:4] in domain["hsHeadings"]
        record = request["dangerousGoods"]["recordId"]
        assert request["goods"][0]["hs6"] in sampler.hs_links[record]


def test_goods_first_request_is_deterministic_and_has_no_source_cargo_input(sampler):
    first = ambient_request(sampler)
    assert first == ambient_request(sampler)
    assert first["goods"][0]["hs6"] in sampler.pools["ambient"]
    assert first["sourceCargoUsedForSelection"] is False
    second = sampler.prepare_request(
        source_document_id="presentation-only-test-source",
        variant=2,
        regime="ambient",
        identity_count=1,
        seed=20261021,
        excluded_identifiers=[first["goods"][0]["hs6"]],
    )
    assert second["goods"][0]["hs6"] != first["goods"][0]["hs6"]
    assert first["registrySelection"]["fullHs6"] == 5612
    assert second["registrySelection"]["reusedAfterExhaustion"] is False
    with pytest.raises(ValueError, match="outside the eligible registry pool"):
        sampler.prepare_request(
            source_document_id="presentation-only-test-source",
            variant=3,
            regime="ambient",
            identity_count=1,
            seed=20261021,
            excluded_identifiers=["999999"],
        )


def test_exhausted_dg_reuse_is_explicit_in_request_receipt(sampler):
    eligible = sampler._dg_pool({})
    request = sampler.prepare_request(
        source_document_id="exhausted-dg-presentation",
        variant=1,
        regime="dg",
        identity_count=1,
        seed=20261021,
        excluded_identifiers=eligible,
    )
    assert request["registrySelection"]["reusedAfterExhaustion"] is True
    assert request["registrySelection"]["excludedIdentities"] == len(eligible)


def test_draw_preserves_flat_shape_and_count_mass_volume_conservation(sampler):
    request = frozen_package(ambient_request(sampler), "PACKAGE_BOX_FIBREBOARD")
    draw = sampler.draw(
        request=request,
        seed=41,
        draw_key="presentation-only-test-source:1",
        container_count=2,
        selected_category="PACKAGE_BOX_FIBREBOARD",
        fixed_quantity=6,
    )
    assert draw == sampler.draw(
        request=request,
        seed=41,
        draw_key="presentation-only-test-source:1",
        container_count=2,
        selected_category="PACKAGE_BOX_FIBREBOARD",
        fixed_quantity=6,
    )
    assert draw["package"] == {
        "packageQuantity": 6,
        "packageTypeFamily": "BOX",
        "packageMaterial": "FIBREBOARD",
    }
    assert draw["internalCategory"] == "PACKAGE_BOX_FIBREBOARD"
    assert sum(row["packageQuantity"] for row in draw["allocations"]) == 6
    assert sum(Decimal(row["grossKg"]) for row in draw["allocations"]) == Decimal(
        draw["physical"]["grossWeight"]["value"]
    )
    assert sum(Decimal(row["volumeM3"]) for row in draw["allocations"]) == Decimal(
        draw["physical"]["volume"]["value"]
    )
    assert all(Decimal(row["grossKg"]) <= 16000 for row in draw["allocations"])
    assert all(Decimal(row["volumeM3"]) <= 24 for row in draw["allocations"])
    assert draw["provenance"]["sourceShipmentDependence"] is False


def test_explicit_row_capacities_and_mixed_equipment_fail_closed(sampler):
    request = frozen_package(ambient_request(sampler, mode="mixed"), "PACKAGE_BOX")
    with pytest.raises(ValueError, match="two distinct printed rows"):
        sampler.draw(
            request=request,
            seed=7,
            draw_key="mixed-one",
            container_count=1,
            selected_category="PACKAGE_BOX",
        )
    domain = [
        {"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE"},
        {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"},
    ]
    draw = sampler.draw(
        request=request,
        seed=7,
        draw_key="mixed-two",
        container_count=2,
        selected_category="PACKAGE_BOX",
        fixed_quantity=2,
        equipment_domain=domain,
        row_capacities=[
            {"payloadKg": "100", "volumeM3": "0.5"},
            {"payloadKg": "100", "volumeM3": "0.5"},
        ],
    )
    assert draw["equipment"][0] != draw["equipment"][1]
    assert all(Decimal(row["grossKg"]) <= 100 for row in draw["allocations"])
    with pytest.raises(ValueError, match="fixed quantity exceeds"):
        sampler.draw(
            request=request,
            seed=7,
            draw_key="overfull",
            container_count=2,
            selected_category="PACKAGE_BOX",
            fixed_quantity=500,
            equipment_domain=domain,
            row_capacities=[
                {"payloadKg": "100", "volumeM3": "0.5"},
                {"payloadKg": "100", "volumeM3": "0.5"},
            ],
        )


def test_certified_fixed_equipment_mix_preserves_exact_slot_counts(sampler):
    request = frozen_package(ambient_request(sampler, mode="mixed"), "PACKAGE_BOX")
    domain = [
        {"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE"},
        {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"},
    ]
    fixed = [domain[0], domain[1], domain[1], domain[1]]
    draw = sampler.draw(
        request=request,
        seed=7,
        draw_key="fixed-mix",
        container_count=4,
        selected_category="PACKAGE_BOX",
        fixed_quantity=4,
        equipment_domain=domain,
        equipment_slots=fixed,
    )
    assert draw["equipment"] == fixed
    assert draw["provenance"]["equipmentSlotsFixed"] is True
    with pytest.raises(ValueError, match="fixed equipment slots differ"):
        sampler.draw(
            request=request,
            seed=7,
            draw_key="missing-fixed-row",
            container_count=4,
            selected_category="PACKAGE_BOX",
            equipment_domain=domain,
            equipment_slots=fixed[:3],
        )


def test_context_sensitive_counting_unit_is_never_inferred(sampler):
    request = frozen_package(ambient_request(sampler), "PACKAGE_VEHICLE")
    with pytest.raises(ValueError, match="requires counted_unit_context"):
        sampler.draw(
            request=request,
            seed=3,
            draw_key="vehicle-context",
            container_count=1,
            selected_category="PACKAGE_VEHICLE",
        )


def test_draw_requires_frozen_package_choice_and_context(sampler):
    request = ambient_request(sampler)
    with pytest.raises(ValueError, match="chosen package category differs"):
        sampler.draw(
            request=request,
            seed=3,
            draw_key="unfrozen-choice",
            container_count=1,
            selected_category="PACKAGE_BOX",
        )
    request = frozen_package(request, "PACKAGE_BOX")
    with pytest.raises(ValueError, match="chosen package category differs"):
        sampler.draw(
            request=request,
            seed=3,
            draw_key="different-choice",
            container_count=1,
            selected_category="PACKAGE_BAG",
        )
    del request["countedUnitContext"]
    with pytest.raises(ValueError, match="counted-unit context differs"):
        sampler.draw(
            request=request,
            seed=3,
            draw_key="missing-context",
            container_count=1,
            selected_category="PACKAGE_BOX",
        )


def test_dg_request_is_exact_registry_join_and_retains_required_flash(sampler):
    request = frozen_package(
        sampler.prepare_request(
            source_document_id="dg-presentation-only",
            variant=1,
            regime="dg",
            identity_count=1,
            seed=9,
            dg_requirements={"flashPoint": True},
        ),
        "PACKAGE_DRUM_STEEL",
    )
    assert request["dangerousGoods"]["recordId"] in sampler.hs_links
    assert request["goods"][0]["hs6"] in sampler.hs_links[request["dangerousGoods"]["recordId"]]
    draw = sampler.draw(
        request=request,
        seed=9,
        draw_key="dg-presentation-only:1",
        container_count=1,
        selected_category="PACKAGE_DRUM_STEEL",
        fixed_quantity=1,
    )
    assert draw["dangerousGoods"]["flashPoint"]["casNumber"]
    assert draw["identities"][0]["hs6"] == request["goods"][0]["hs6"]
    tampered = deepcopy(request)
    tampered["goods"][0]["hs6"] = "999999"
    with pytest.raises(ValueError, match="exact eligible HMT/HS6 join"):
        sampler.draw(
            request=tampered,
            seed=9,
            draw_key="tampered",
            container_count=1,
            selected_category="PACKAGE_DRUM_STEEL",
        )


def test_non_dg_phrase_or_thermal_ancestry_tamper_is_rejected(sampler):
    request = frozen_package(ambient_request(sampler), "PACKAGE_BOX")
    tampered = deepcopy(request)
    tampered["goods"][0]["description"] = "source cargo copy"
    tampered["goodsForm"] = "source cargo copy"
    with pytest.raises(ValueError, match="full HS6 registry ancestry"):
        sampler.draw(
            request=tampered,
            seed=1,
            draw_key="tampered-phrase",
            container_count=1,
            selected_category="PACKAGE_BOX",
        )
