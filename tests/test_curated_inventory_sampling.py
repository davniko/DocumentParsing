"""Explicit inventory distributions and their certified numeric sampling path."""

import json
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_curated_registry_sampling import profile_data, registry_catalog
from test_curated_scenarios import config, source_row

from document_ocr.synthesis.curated import assign
from document_ocr.synthesis.curated_equipment_inventory import compile_equipment_inventory
from document_ocr.synthesis.curated_scenarios import (
    CandidateIncompatibility,
    EquipmentInventorySampling,
    ScenarioCatalog,
    SourceCapabilities,
    _inventory_shares,
)
from document_ocr.synthesis.curated_templates import RenderRegion
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.transport_capacity import equipment_capacity

GP = "GENERAL_PURPOSE"
S20 = "TWENTY_FOOT_STANDARD_HEIGHT"
H20 = "TWENTY_FOOT_HIGH_CUBE"
S40 = "FORTY_FOOT_STANDARD_HEIGHT"
H40 = "FORTY_FOOT_HIGH_CUBE"
H45 = "FORTY_FIVE_FOOT_HIGH_CUBE"
CAP = SourceCapabilities(family="ambient", physical_profile="registry_form")


def inventory_policy(**values):
    return EquipmentInventorySampling.model_validate({"source_ids": ("source",), **values})


@pytest.mark.parametrize("probability,expected", [("0", "uniform"), ("1", "mixed")])
def test_inventory_probability_endpoints_are_exact(probability, expected):
    policy = inventory_policy(mixed_probability=probability)
    assert {policy.mode(i, DeterministicStream(17, "policy", str(i))) for i in range(1, 101)} == {
        expected
    }


def test_inventory_schedule_cycles_by_draw_not_source():
    policy = inventory_policy(mixed_probability=0, modes=("uniform", "mixed"))
    assert [
        policy.mode(i, DeterministicStream(17, "policy", "one-source")) for i in range(1, 7)
    ] == ["uniform", "mixed", "uniform", "mixed", "uniform", "mixed"]
    assert policy == EquipmentInventorySampling.model_validate(json.loads(policy.model_dump_json()))


def test_inventory_bernoulli_mode_replays_and_varies_per_draw():
    policy = inventory_policy(mixed_probability=Decimal("0.5"))
    drawn = [policy.mode(i, DeterministicStream(17, "policy", str(i))) for i in range(1, 101)]
    assert set(drawn) == {"uniform", "mixed"}
    assert drawn == [
        policy.mode(i, DeterministicStream(17, "policy", str(i))) for i in range(1, 101)
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"mixed_probability": -1},
        {"mixed_probability": 2},
        {"mixed_probability": "NaN"},
        {"mixed_probability": "Infinity"},
        {"category_count_weights": {}},
        {"category_count_weights": {1: 1}},
        {"category_count_weights": {6: 1}},
        {"category_count_weights": {2: 0}},
        {"category_count_weights": {2: True}},
        {"category_count_weights": {2: 1.5}},
        {"category_count_weights": {True: 1}},
        {"category_count_weights": {"02": 1}},
        {"category_count_weights": {"2.0": 1}},
        {"category_count_weights": {"+2": 1}},
        {"category_count_weights": {2: 1, "2": 2}},
        {"modes": ()},
        {"modes": ("automatic",)},
        {"unknown": True},
        {"source_ids": ()},
        {"source_ids": ("source", "source")},
        {"source_ids": (" source",)},
    ],
)
def test_inventory_policy_rejects_invalid_or_implicit_support(changes):
    with pytest.raises(ValueError):
        EquipmentInventorySampling.model_validate(
            {"source_ids": ("source",), "mixed_probability": "0.5", **changes}
        )


def test_inventory_count_distribution_json_replays_exactly():
    policy = inventory_policy(mixed_probability="0.3", category_count_weights={2: 3, 3: 1})
    assert policy == EquipmentInventorySampling.model_validate_json(policy.model_dump_json())
    assert policy == EquipmentInventorySampling.model_validate(policy.model_dump(mode="json"))
    assert "equipment_inventory" not in config().model_dump(mode="json")
    configured = config(equipment_inventory=policy)
    assert configured == type(configured).model_validate(configured.model_dump(mode="json"))


def inventory_blueprint(row, groups=None):
    rows = row["target"]["documentPatch"]["containerInformation"]
    groups = groups or tuple((index,) for index in range(len(rows)))
    aliases = {S20: "20GP", H20: "20HQ", S40: "40GP", H40: "40HC", H45: "45HC"}
    text, bindings, regions = "", {}, []

    def append(key, surface, paths):
        nonlocal text
        start = len(text.encode())
        text += surface
        end = len(text.encode())
        text += "\n"
        bindings[key] = {
            "target_paths": list(paths),
            "occurrences": [{"byte_start": start, "byte_end": end, "source_text": surface}],
        }
        regions.append(RenderRegion(key, 0, start, end, surface, tuple(paths), "historical"))

    for index, group in enumerate(groups):
        append(
            f"type{index}",
            aliases[rows[group[0]]["sizeCategory"]],
            [
                f"documentPatch.containerInformation[{row_index}].{field}"
                for row_index in group
                for field in ("sizeCategory", "typeCategory")
            ],
        )
        if len(group) > 1:
            bindings[f"type{index}"]["target_relationship"] = "shared_value_equality"
    for index, container in enumerate(rows):
        append(
            f"id{index}",
            container["equipmentIdentifier"],
            [f"documentPatch.containerInformation[{index}].equipmentIdentifier"],
        )
    row["joinedRawText"] = text
    return SimpleNamespace(
        document_id=row["documentId"],
        source=text,
        target=row["target"],
        historical_bindings=bindings,
        ownership_data={},
        regions=tuple(regions),
    )


def certified_catalog(row=None, *, policy=None, groups=None, bind=True, **weights):
    row = source_row() if row is None else row
    bp = inventory_blueprint(row, groups)
    policy = policy or inventory_policy(mixed_probability=1)
    support = registry_catalog(row, equipment_inventory=policy, **weights)
    contract = compile_equipment_inventory(bp)
    if bind:
        support.bind_equipment_inventory({row["documentId"]: contract})
    return row, support, contract


def applied_cargo(row, scenario):
    target = deepcopy(row["target"])
    for path, value in scenario.replacements.items():
        assign(target, path, deepcopy(value))
    return {
        key: target["documentPatch"][key] for key in ("goodsItemDetails", "containerInformation")
    }


def test_same_certified_source_supports_uniform_and_mixed_replay_without_donor(monkeypatch):
    row, support, _ = certified_catalog(
        policy=inventory_policy(mixed_probability="0.5", modes=("uniform", "mixed"))
    )
    before = deepcopy(row)

    def no_donor(*args):
        raise AssertionError("registry inventory must not use an observed donor")

    monkeypatch.setattr(support, "_donors", no_donor)
    seen = set()
    for variant in range(1, 101):
        scenario = support.sample(row, seed=17, variant=variant, capabilities=CAP)
        assert scenario == support.sample(row, seed=17, variant=variant, capabilities=CAP)
        rows = scenario.cargo["containerInformation"]
        sizes = [container["sizeCategory"] for container in rows]
        seen.update(sizes)
        assert len(set(sizes)) == (1 if variant % 2 else 2)
        assert scenario.provenance["equipmentInventory"]["sizes"] == sizes
        assert scenario.provenance["equipmentInventory"]["mode"] == (
            "uniform" if variant % 2 else "mixed"
        )
        assert "donorDocumentId" not in scenario.provenance
        assert applied_cargo(row, scenario) == scenario.cargo
        assert [container["equipmentIdentifier"] for container in rows] == [
            container["equipmentIdentifier"]
            for container in row["target"]["documentPatch"]["containerInformation"]
        ]
        for container, physical in zip(rows, scenario.provenance["physicalRows"], strict=True):
            capacity = equipment_capacity(container, support.limits)
            assert physical["equipmentIdentifier"] == container["equipmentIdentifier"]
            assert physical["capacity"] == {
                "payloadKg": str(capacity.payload_kg),
                "volumeM3": str(capacity.volume_m3),
            }
            assert Decimal(physical["measures"]["grossWeight"]["value"]) <= capacity.payload_kg
            assert Decimal(physical["measures"]["volume"]["value"]) <= capacity.volume_m3
    assert seen == {S20, H20, S40, H40, H45}
    assert row == before


def test_inventory_readiness_validates_domains_without_drawing(monkeypatch):
    row, support, _ = certified_catalog()

    def no_draw(*args, **kwargs):
        raise AssertionError("readiness must not sample cargo")

    monkeypatch.setattr(support, "_registry_cargo", no_draw)
    support.validate_equipment_inventory_support(row["documentId"], CAP)
    with pytest.raises(ValueError, match="no reviewed registry-form support"):
        support.validate_equipment_inventory_support(
            row["documentId"],
            CAP.model_copy(update={"allowed_package_categories": ("PACKAGE_BAG_PLASTIC",)}),
        )
    with pytest.raises(ValueError, match="no feasible source support"):
        support.validate_equipment_inventory_support(
            row["documentId"], CAP.model_copy(update={"allowed_equipment_pairs": (H45 + "|" + GP,)})
        )
    with pytest.raises(ValueError, match="registry-form domain"):
        support.validate_equipment_inventory_support(
            row["documentId"], SourceCapabilities(family="frozen")
        )


def test_inventory_receipt_roundtrips_and_train_label_mutation_cannot_bypass_pin():
    row, support, _ = certified_catalog()
    scenario = support.sample(row, seed=17, variant=1, capabilities=CAP)
    assert scenario == type(scenario).model_validate(json.loads(scenario.model_dump_json()))
    # The train map retains this same source object. Comparing caller vs train
    # alone would approve a shared-object mutation; the certificate must not.
    row["target"]["documentPatch"]["goodsItemDetails"][0]["grossWeight"]["value"] += 1
    with pytest.raises(ValueError, match="bound ownership certificate"):
        support.sample(row, seed=17, variant=1, capabilities=CAP)
    with pytest.raises(ValueError, match="bound exact source certificate"):
        support.validate_equipment_inventory_support(row["documentId"], CAP)


def test_existing_uniform_draw_is_unchanged_when_inventory_mode_is_uniform():
    row, support, _ = certified_catalog(policy=inventory_policy(mixed_probability=0))
    legacy = registry_catalog(row)
    for variant in range(1, 20):
        new = support.sample(row, seed=11, variant=variant, capabilities=CAP)
        old = legacy.sample(row, seed=11, variant=variant, capabilities=CAP)
        assert new.cargo == old.cargo
        assert new.goods_identities == old.goods_identities
        assert new.replacements == old.replacements


def test_mixed_source_requires_certificate_even_to_reclassify_uniform():
    row = source_row()
    row["target"]["documentPatch"]["containerInformation"][0]["sizeCategory"] = S20
    with pytest.raises(ValueError, match="separate inventory contract"):
        registry_catalog(row).sample(row, seed=11, variant=1, capabilities=CAP)
    row, support, _ = certified_catalog(row, policy=inventory_policy(mixed_probability=0))
    scenario = support.sample(row, seed=11, variant=1, capabilities=CAP)
    assert len({c["sizeCategory"] for c in scenario.cargo["containerInformation"]}) == 1


def test_category_count_draw_uses_explicit_feasible_intersection_not_downgrade():
    row, support, _ = certified_catalog(
        policy=inventory_policy(mixed_probability=1, category_count_weights={2: 1, 3: 100})
    )
    scenario = support.sample(row, seed=11, variant=1, capabilities=CAP)
    assert scenario.provenance["equipmentInventory"]["eligibleCategoryCounts"] == [2]
    assert len(set(scenario.provenance["equipmentInventory"]["sizes"])) == 2
    row, support, _ = certified_catalog(
        policy=inventory_policy(mixed_probability=1, category_count_weights={3: 1})
    )
    with pytest.raises(ValueError, match="no feasible source support"):
        support.sample(row, seed=11, variant=1, capabilities=CAP)


def test_pair_weights_constrain_every_sampled_category_and_cannot_homogenize():
    row, support, _ = certified_catalog(equipment_weights={S20 + "|" + GP: 1, H45 + "|" + GP: 1})
    for variant in range(1, 10):
        scenario = support.sample(row, seed=17, variant=variant, capabilities=CAP)
        assert set(scenario.provenance["equipmentInventory"]["sizes"]) == {S20, H45}
    row, support, _ = certified_catalog(equipment_weights={H45 + "|" + GP: 1})
    with pytest.raises(ValueError, match="no feasible source support"):
        support.sample(row, seed=17, variant=1, capabilities=CAP)


def test_mixed_mode_does_not_change_during_candidate_retries():
    row, support, _ = certified_catalog()
    checked = []

    def reject_first(candidate):
        checked.append(candidate)
        if len(checked) == 1:
            raise CandidateIncompatibility("controlled physical-grid rejection")

    result = support.sample(
        row, seed=17, variant=1, capabilities=CAP, validate_candidate=reject_first
    )
    assert len(checked) == result.provenance["candidateAttempts"] == 2
    assert all(
        candidate.provenance["equipmentInventory"]["mode"] == "mixed" for candidate in checked
    )
    assert applied_cargo(row, result) == result.cargo


@pytest.mark.parametrize(
    "failure", ["unbound", "source_ocr", "non_registry", "target_pin", "groups"]
)
def test_invalid_inventory_contracts_fail_before_admission(failure):
    row, support, contract = certified_catalog(bind=False)
    if failure == "target_pin":
        with pytest.raises(ValueError, match="exact train-fit source"):
            support.bind_equipment_inventory(
                {"source": replace(contract, source_target_sha256="0" * 64)}
            )
        return
    if failure == "groups":
        with pytest.raises(ValueError, match="independent GP groups"):
            support.bind_equipment_inventory({"source": replace(contract, groups=((0,), (0, 1)))})
        return
    if failure != "unbound":
        support.bind_equipment_inventory({"source": contract})
    if failure == "source_ocr":
        row = deepcopy(row)
        row["joinedRawText"] += "changed"
    cap = CAP if failure != "non_registry" else SourceCapabilities(family="ambient")
    with pytest.raises(ValueError, match=r"certificate|registry-form"):
        support.sample(row, seed=17, variant=1, capabilities=cap)


def test_inventory_share_ownership_is_id_based_not_placement_order():
    row = source_row()
    patch = row["target"]["documentPatch"]
    assert _inventory_shares(patch["containerInformation"], patch["goodsItemDetails"][0], 100) == (
        Decimal("0.4"),
        Decimal("0.6"),
    )
    patch["goodsItemDetails"][0]["splitGoodsPlacement"].reverse()
    assert _inventory_shares(patch["containerInformation"], patch["goodsItemDetails"][0], 100) == (
        Decimal("0.4"),
        Decimal("0.6"),
    )


@pytest.mark.parametrize(
    "defect", ["duplicate_id", "unknown_id", "partial_counts", "sum", "missing_row"]
)
def test_inventory_share_contract_rejects_invalid_source_allocations(defect):
    row = source_row()
    patch = row["target"]["documentPatch"]
    placements = patch["goodsItemDetails"][0]["splitGoodsPlacement"]
    if defect == "duplicate_id":
        placements[1]["equipmentIdentifier"] = placements[0]["equipmentIdentifier"]
    elif defect == "unknown_id":
        placements[1]["equipmentIdentifier"] = "UNKNOWN"
    elif defect == "partial_counts":
        placements[1].pop("packageQuantity")
    elif defect == "sum":
        placements[1]["packageQuantity"] += 1
    else:
        placements.pop()
    with pytest.raises(ValueError, match="inventory"):
        _inventory_shares(patch["containerInformation"], patch["goodsItemDetails"][0], 100)


def multi_row_source(count):
    row = source_row()
    patch = row["target"]["documentPatch"]
    patch["containerInformation"] = [
        {"equipmentIdentifier": f"MSCU{index:07d}", "sizeCategory": H40, "typeCategory": GP}
        for index in range(count)
    ]
    patch["goodsItemDetails"][0]["splitGoodsPlacement"] = [
        {"equipmentIdentifier": container["equipmentIdentifier"], "packageQuantity": 100 // count}
        for container in patch["containerInformation"]
    ]
    return row


def test_complete_mixed_inventory_supports_three_to_five_categories():
    for requested in range(3, 6):
        row, support, _ = certified_catalog(
            multi_row_source(5),
            policy=inventory_policy(mixed_probability=1, category_count_weights={requested: 1}),
        )
        for variant in range(1, 11):
            scenario = support.sample(row, seed=23, variant=variant, capabilities=CAP)
            assert len(set(scenario.provenance["equipmentInventory"]["sizes"])) == requested
            assert scenario.provenance["equipmentInventory"]["eligibleCategoryCounts"] == [
                requested
            ]
            assert applied_cargo(row, scenario) == scenario.cargo


def test_shared_scalar_groups_are_never_split_by_composition():
    row = multi_row_source(4)
    row, support, contract = certified_catalog(row, groups=((0, 1), (2, 3)))
    assert contract.groups == ((0, 1), (2, 3))
    for variant in range(1, 11):
        scenario = support.sample(row, seed=23, variant=variant, capabilities=CAP)
        sizes = scenario.provenance["equipmentInventory"]["sizes"]
        assert sizes[0] == sizes[1] != sizes[2] == sizes[3]


def test_mixed_quantity_bound_accounts_for_every_assigned_row_and_profile_interval():
    data = profile_data()
    data["profiles"][0].update(quantity_max=100000, unit_volume_m3="1")
    row, support, _ = certified_catalog(
        profiles_data=data,
        equipment_weights={S20 + "|" + GP: 1, H45 + "|" + GP: 1},
    )
    for variant in range(1, 31):
        scenario = support.sample(row, seed=23, variant=variant, capabilities=CAP)
        goods = scenario.cargo["goodsItemDetails"][0]
        quantity = goods["numberAndTypeOfPackages"][0]["packageQuantity"]
        assert quantity <= scenario.provenance["equipmentInventory"]["quantityUpperBound"]
        assert goods["volume"]["value"] == quantity
        for container, physical in zip(
            scenario.cargo["containerInformation"], scenario.provenance["physicalRows"], strict=True
        ):
            capacity = equipment_capacity(container, support.limits)
            assert Decimal(physical["measures"]["volume"]["value"]) <= capacity.volume_m3


def test_infeasible_fixed_quantity_is_rejected_not_changed_or_homogenized():
    row, support, _ = certified_catalog(maximum_candidates=2)
    with pytest.raises(ValueError, match=r"after 2 draws.*fixed quantity"):
        support.sample(
            row,
            seed=23,
            variant=1,
            capabilities=CAP.model_copy(update={"fixed_package_quantity": 1000}),
        )


def test_explicit_scope_leaves_unscoped_registry_draw_unchanged():
    row, base, contract = certified_catalog()
    unscoped = source_row(identifier="other")
    values = dict(
        countries=base.countries,
        hs=base.hs,
        phrases=base.phrases,
        ports=base.ports,
        localities=base.localities,
        dg=base.dg,
        train_rows=[row, unscoped],
        validation_ids=frozenset(),
        load_profiles=base.load_profiles,
    )
    support = ScenarioCatalog(config=base.config, **values)
    support.bind_equipment_inventory({"source": contract})
    plain = ScenarioCatalog(
        config=base.config.model_copy(update={"equipment_inventory": None}), **values
    )
    for variant in range(1, 10):
        scoped_catalog = support.sample(unscoped, seed=23, variant=variant, capabilities=CAP)
        ordinary = plain.sample(unscoped, seed=23, variant=variant, capabilities=CAP)
        assert scoped_catalog.cargo == ordinary.cargo
        assert scoped_catalog.replacements == ordinary.replacements
        assert "equipmentInventory" not in scoped_catalog.provenance


def test_unknown_source_scope_cannot_be_fitted_or_silently_ignored():
    with pytest.raises(ValueError, match="entirely train-fit"):
        registry_catalog(
            source_row(),
            equipment_inventory=inventory_policy(source_ids=("not-training",), mixed_probability=1),
        )


@pytest.fixture(scope="session")
def actual_inventory_campaign():
    from document_ocr.synthesis.curated_campaign import Campaign

    root = Path(__file__).resolve().parents[1]
    inventory = json.loads(
        (
            root
            / "docs/analysis/mixed-equipment-expanded-pilot-20261012"
            / "registered-source-inventory.json"
        ).read_text()
    )
    selected = tuple(
        row for row in inventory["sources"] if row["reason"] == "independently_owned_ambient_gp"
    )
    ids = tuple(row["sourceDocumentId"] for row in selected)
    assert len(ids) == len(set(ids)) == 45
    base = yaml.safe_load(
        (root / "configs/synthesis/mpci_bl_curated_v7_mixed_inventory_validation.yaml").read_text()
    )
    base.update(
        source_ids=list(ids),
        variants_per_source=2,
        output="docs/analysis/production-equipment-inventory-20261012/offline-test-campaign",
        capabilities={sid: base["capabilities"][sid] for sid in ids},
    )
    base["sampling"]["equipment_inventory"] = inventory_policy(
        source_ids=ids, mixed_probability="0.5", modes=("uniform", "mixed")
    ).model_dump(mode="json")
    base.pop("template_sampling", None)
    campaign = Campaign(root, base)

    async def no_paid_calls(*args, **kwargs):
        raise AssertionError("offline sampler regression must never contact a model")

    campaign.calls.call = no_paid_calls
    campaign.validate_source_readiness()
    return campaign


@pytest.mark.parametrize("source_index", range(45))
def test_all_45_previous_admitted_sources_have_real_uniform_and_mixed_physical_plans(
    actual_inventory_campaign, source_index
):
    campaign = actual_inventory_campaign
    sid = campaign.source_ids[source_index]
    source = deepcopy(campaign.rows[sid])
    for variant, expected in ((1, "uniform"), (2, "mixed")):
        bp, scenario, _sample_id, target = campaign.plan(sid, variant)
        physical = campaign.physical_render(bp, scenario, target)
        rows = physical.target["documentPatch"]["containerInformation"]
        sizes = [row["sizeCategory"] for row in rows]
        assert len(set(sizes)) == (1 if expected == "uniform" else 2)
        assert scenario.provenance["equipmentInventory"]["mode"] == expected
        assert scenario.provenance["equipmentInventory"]["sizes"] == sizes
        assert len(sizes) == len(source["target"]["documentPatch"]["containerInformation"])
        assert len(physical.receipt["equipmentInventoryAliases"]["aliases"]) == len(set(sizes))
        assert (
            scenario.cargo["containerInformation"]
            == applied_cargo(campaign.rows[sid], scenario)["containerInformation"]
        )
    assert campaign.rows[sid] == source
