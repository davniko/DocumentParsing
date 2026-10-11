"""Printed equipment mixes: matching, source authority and sampled admission."""

from copy import deepcopy
from decimal import Decimal
from itertools import permutations
from random import Random

import pytest
from pydantic import ValidationError
from test_curated_scenarios import catalog, source_row

from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.curated_equipment import PrintedEquipmentMix, capacity_assignment
from document_ocr.synthesis.curated_scenarios import SourceCapabilities


def mix_source(*, volume=144):
    source = source_row()
    source["joinedRawText"] = "NO. OF CONTAINERS\n1X20 DR\n3X40 HC\n"
    patch = source["target"]["documentPatch"]
    patch["containerInformation"] = [{"equipmentIdentifier": f"MSCU123456{i}"} for i in range(4)]
    goods = patch["goodsItemDetails"][0]
    goods["volume"]["value"] = volume
    goods["splitGoodsPlacement"] = [
        {"equipmentIdentifier": c["equipmentIdentifier"], "packageQuantity": q}
        for c, q in zip(patch["containerInformation"], [10, 30, 30, 30], strict=True)
    ]
    certificate = PrintedEquipmentMix(
        source_sha256=sha256_bytes(source["joinedRawText"].encode()),
        source_text_literals=("1X20 DR\n3X40 HC",),
        members=[
            dict(
                count=1, sizeCategory="TWENTY_FOOT_STANDARD_HEIGHT", typeCategory="GENERAL_PURPOSE"
            ),
            dict(count=3, sizeCategory="FORTY_FOOT_HIGH_CUBE", typeCategory="GENERAL_PURPOSE"),
        ],
    )
    return source, certificate


@pytest.mark.parametrize(
    ("volumes", "weights", "feasible"),
    [
        ([3, 7, 7, 7], [2, 6, 6, 6], True),
        ([7, 3, 7, 7], [6, 2, 6, 6], True),
        ([3, 3, 3, 3], [2, 2, 2, 2], True),
        ([4, 4, 4, 4], [2, 2, 2, 2], False),
        ([3, 7, 7, 9], [2, 6, 6, 6], False),
        ([3, 7, 7, 7], [4, 6, 6, 6], False),
        # Small volume and small mass are in different rows: no joint witness.
        ([3, 7, 7, 7], [6, 2, 6, 6], False),
    ],
)
def test_joint_capacity_assignment(volumes, weights, feasible):
    capacities = [{"payloadKg": "3", "volumeM3": "3"}] + [
        {"payloadKg": "8", "volumeM3": "8"} for _ in range(3)
    ]
    loads = [
        {"volume": Decimal(v), "grossWeight": Decimal(w)}
        for v, w in zip(volumes, weights, strict=True)
    ]
    result = capacity_assignment(loads, capacities)
    assert (result is not None) == feasible
    if result is not None:
        assert len(set(result)) == 4


def test_matching_agrees_with_exhaustive_oracle_and_handles_reassignment():
    rng = Random(20261010)
    for _ in range(500):
        capacities = [
            {"payloadKg": rng.randint(1, 9), "volumeM3": rng.randint(1, 9)} for _ in range(4)
        ]
        loads = [
            {"grossWeight": Decimal(rng.randint(0, 9)), "volume": Decimal(rng.randint(0, 9))}
            for _ in range(4)
        ]
        oracle = any(
            all(
                loads[i]["grossWeight"] <= capacities[j]["payloadKg"]
                and loads[i]["volume"] <= capacities[j]["volumeM3"]
                for i, j in enumerate(order)
            )
            for order in permutations(range(4))
        )
        assert (capacity_assignment(loads, capacities) is not None) == oracle


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1"])
def test_nonfinite_or_negative_load_is_a_contract_error(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        capacity_assignment([{"volume": Decimal(value)}], [{"payloadKg": "10", "volumeM3": "10"}])


def test_unknown_dimensions_and_incomplete_mix_fail_closed():
    with pytest.raises(ValueError, match="unknown physical dimension"):
        capacity_assignment([{"quantity": Decimal(1)}], [{"payloadKg": "10", "volumeM3": None}])
    with pytest.raises(ValueError, match="one load per capacity"):
        capacity_assignment([], [{"payloadKg": "10", "volumeM3": "10"}])


@pytest.mark.parametrize("payload", ["0", "-1", "NaN", "Infinity"])
def test_invalid_capacity_is_a_contract_error(payload):
    with pytest.raises(ValueError, match="finite and positive"):
        capacity_assignment([{}], [{"payloadKg": payload, "volumeM3": None}])


def test_equipment_certificate_requires_exact_source_and_unassigned_identifiers():
    source, mix = mix_source()
    containers = source["target"]["documentPatch"]["containerInformation"]
    mix.validate_source(source["joinedRawText"], containers)
    with pytest.raises(ValueError, match="source hash"):
        mix.validate_source(source["joinedRawText"] + "CHANGED", containers)
    with pytest.raises(ValueError, match="count differs"):
        mix.validate_source(source["joinedRawText"], containers[:3])
    with pytest.raises(ValueError, match="public equipment categories"):
        mix.validate_source(
            source["joinedRawText"], [{**c, "typeCategory": "GENERAL_PURPOSE"} for c in containers]
        )
    bad = mix.model_copy(update={"source_text_literals": ("NOT PRESENT",)})
    with pytest.raises(ValueError, match="evidence"):
        bad.validate_source(source["joinedRawText"], containers)


@pytest.mark.parametrize(
    "field,value",
    [("count", True), ("count", 0), ("sizeCategory", "UNKNOWN"), ("typeCategory", "UNKNOWN")],
)
def test_invalid_canonical_mix_is_rejected(field, value):
    _, mix = mix_source()
    data = mix.model_dump()
    data["members"][0][field] = value
    with pytest.raises(ValidationError):
        PrintedEquipmentMix.model_validate(data)


def test_sampler_preserves_unknown_public_types_and_carries_a_private_witness():
    source, mix = mix_source()
    before = deepcopy(source)
    donor = deepcopy(source)
    donor["documentId"] = "measured-donor"
    for container in donor["target"]["documentPatch"]["containerInformation"]:
        container.update(sizeCategory="FORTY_FOOT_HIGH_CUBE", typeCategory="GENERAL_PURPOSE")
    support = catalog([source, donor])
    scenario = support.sample(
        source,
        seed=33,
        variant=1,
        capabilities=SourceCapabilities(
            family="ambient",
            fixed_package_quantity=100,
            equipment_mix=mix,
        ),
    )
    assert source == before
    assert all(set(c) == {"equipmentIdentifier"} for c in scenario.cargo["containerInformation"])
    assert Decimal(scenario.provenance["physicalRows"][0]["capacity"]["volumeM3"]) == Decimal(
        "33.2"
    )
    assert scenario.provenance["equipmentMix"]["certificate"] == mix.model_dump(mode="json")


def test_sampler_rejects_four_40_cube_loads_even_when_aggregate_capacity_is_sufficient():
    source, mix = mix_source(volume=160)
    for placement in source["target"]["documentPatch"]["goodsItemDetails"][0][
        "splitGoodsPlacement"
    ]:
        placement["packageQuantity"] = 25
    donor = deepcopy(source)
    donor["documentId"] = "measured-donor"
    for container in donor["target"]["documentPatch"]["containerInformation"]:
        container.update(sizeCategory="FORTY_FOOT_HIGH_CUBE", typeCategory="GENERAL_PURPOSE")
    with pytest.raises(ValueError, match="cannot fit the printed equipment mix"):
        catalog([source, donor], maximum_candidates=2).sample(
            source,
            seed=33,
            variant=1,
            capabilities=SourceCapabilities(
                family="ambient",
                fixed_package_quantity=100,
                equipment_mix=mix,
            ),
        )
