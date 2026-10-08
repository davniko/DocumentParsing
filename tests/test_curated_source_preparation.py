"""Source-contract preparation must preserve shared printed fact ownership."""

import importlib.util
from pathlib import Path

import pytest

from document_ocr.synthesis.curated import SourceContract, compile_contract


def test_rebase_shared_numeric_targets_never_reference_a_removed_variable():
    path = Path(__file__).resolve().parents[1] / "scripts/synthesis/rebase_curated_v7.py"
    spec = importlib.util.spec_from_file_location("rebase_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity"
    second = "documentPatch.goodsItemDetails[0].splitGoodsPlacement[0].packageQuantity"
    row = {
        "documentId": "source",
        "joinedRawText": "COUNT 12\nCOPY 12\nMSKU8231362\n",
        "target": {
            "schemaVersion": "7.0.0",
            "documentPatch": {
                "negotiability": "non_negotiable",
                "containerInformation": [{"equipmentIdentifier": "MSKU8231362"}],
                "goodsItemDetails": [
                    {
                        "numberAndTypeOfPackages": [{"packageQuantity": 12}],
                        "splitGoodsPlacement": [
                            {"equipmentIdentifier": "MSKU8231362", "packageQuantity": 12}
                        ],
                    }
                ],
            },
        },
    }
    historical = {
        "bindings": [
            {
                "logical_key": "first",
                "target_paths": [first],
                "derivation": None,
                "occurrences": [{"byte_start": 6, "source_text": "12"}],
            },
            {
                "logical_key": "copy",
                "target_paths": [second, first],
                "derivation": None,
                "occurrences": [{"byte_start": 14, "source_text": "12"}],
            },
        ]
    }
    draft, missing = module.draft(row, historical, {}, {})
    assert not missing
    contract = SourceContract.model_validate(draft)
    compile_contract(row, contract)
    numeric = [v for v in contract.variables if v.kind == "count"]
    assert len(numeric) == 1
    assert len(numeric[0].occurrences) == 2
    assert len({t.expression for t in contract.targets if t.path in (first, second)}) == 1


@pytest.fixture
def rebase():
    path = Path(__file__).resolve().parents[1] / "scripts/synthesis/rebase_curated_v7.py"
    spec = importlib.util.spec_from_file_location("rebase_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binding(raw, key, quote, paths=(), **attributes):
    start = raw.index(quote)
    return {
        "logical_key": key,
        "target_paths": list(paths),
        "derivation": None,
        "occurrences": [{"byte_start": len(raw[:start].encode()), "source_text": quote}],
        **attributes,
    }


def row(raw, goods, containers=()):
    patch = {"negotiability": "non_negotiable", "goodsItemDetails": [goods]}
    if containers:
        patch["containerInformation"] = [{"equipmentIdentifier": c} for c in containers]
    return {
        "documentId": "source",
        "joinedRawText": raw,
        "target": {"schemaVersion": "7.0.0", "documentPatch": patch},
    }


def test_extra_count_occurrences_certify_both_digits_and_words(rebase):
    raw = "12 PACKAGES\nTWELVE (12) PACKAGES ONLY\nTHIRTEEN\n"
    source = row(raw, {"numberAndTypeOfPackages": [{"packageQuantity": 12}]})
    historical = {
        "bindings": [binding(raw, "total", "12", ["documentPatch.cargoPackages[0].quantity"])]
    }
    draft, missing = rebase.draft(source, historical, {}, {})
    assert not missing
    key = next(v["key"] for v in draft["variables"] if v["kind"] == "count")
    result, _ = rebase.draft(source, historical, {}, {"extra_occurrences": {key: ["TWELVE", "12"]}})
    variable = next(v for v in result["variables"] if v["key"] == key)
    assert [(o["text"], o["presentation"]) for o in variable["occurrences"]] == [
        ("12", "number"),
        ("TWELVE", "words"),
        ("12", "number"),
    ]
    compile_contract(source, SourceContract.model_validate(result))
    with pytest.raises(ValueError, match="disagree"):
        rebase.draft(source, historical, {}, {"extra_occurrences": {key: ["THIRTEEN"]}})
    with pytest.raises(ValueError, match="absent"):
        rebase.draft(source, historical, {}, {"extra_occurrences": {key: ["FOURTEEN"]}})


def test_carrier_receipt_selects_proven_package_count_not_container_count(rebase):
    raw = "MSKU8231362\n4 PALLETS\nTOTAL CONTAINERS OR PACKAGES4\n"
    source = row(raw, {"numberAndTypeOfPackages": [{"packageQuantity": 4}]}, ["MSKU8231362"])
    receipt = binding(
        raw,
        "carrier_receipt",
        "4",
        ["documentPatch.containers"],
        derivation="container_package_count",
    )
    receipt["occurrences"][0]["byte_start"] = len(raw[: raw.rfind("4")].encode())
    historical = {
        "bindings": [
            binding(raw, "package_count", "4", ["documentPatch.cargoPackages[0].quantity"]),
            receipt,
        ]
    }
    result, missing = rebase.draft(source, historical, {}, {})
    assert not missing
    counts = [v for v in result["variables"] if v["kind"] == "count"]
    assert len(counts) == 1 and len(counts[0]["occurrences"]) == 2
    compile_contract(source, SourceContract.model_validate(result))


def test_allocation_measure_role_requires_proven_current_equipment_map(rebase):
    item = {
        "logical_key": "agent:cargo:allocation:2:gross_weight",
        "group_key": "allocation:0:2",
        "target_paths": [],
    }
    assert rebase.measure_role(item, {}, {(0, 2): 1}) == ("grossWeight", 1)
    with pytest.raises(ValueError, match="source-proven"):
        rebase.measure_role(item, {})


def test_word_counts_follow_declared_package_dependencies_only(rebase):
    raw = "PACKAGES 6\nSIX PACKAGES\nTHREE ORIGINAL BILLS\nONE CONTAINER\n"
    source = row(raw, {"numberAndTypeOfPackages": [{"packageQuantity": 6}]})
    quantity = "documentPatch.cargoPackages[0].quantity"
    historical = {
        "bindings": [
            binding(raw, "packages", "6", [quantity]),
            binding(
                raw,
                "package_words",
                "SIX PACKAGES",
                derivation="number_to_words",
                dependency_paths=[quantity],
            ),
            binding(
                raw,
                "original_bill_count",
                "THREE ORIGINAL BILLS",
                derivation="number_to_words",
                group_kind="document",
            ),
            binding(
                raw,
                "container_count",
                "ONE CONTAINER",
                derivation="number_to_words",
                dependency_paths=["documentPatch.containers"],
            ),
        ]
    }
    contract, missing = rebase.draft(source, historical, {}, {})
    assert not missing
    numeric = [v for v in contract["variables"] if v["kind"] == "count"]
    assert len(numeric) == 1
    assert {v["text"] for v in numeric[0]["occurrences"]} == {"6", "SIX"}
    compile_contract(source, SourceContract.model_validate(contract))


def test_collapsed_package_rows_derive_total_without_false_shared_alias(rebase):
    raw = "MSKU8231362 4 COILS\nTCLU5036160 5 COILS\n"
    source = row(
        raw,
        {
            "numberAndTypeOfPackages": [{"packageQuantity": 9}],
            "splitGoodsPlacement": [
                {"equipmentIdentifier": "MSKU8231362", "packageQuantity": 4},
                {"equipmentIdentifier": "TCLU5036160", "packageQuantity": 5},
            ],
        },
        ["MSKU8231362", "TCLU5036160"],
    )
    historical = {
        "bindings": [
            binding(
                raw,
                f"row{i}",
                str(value),
                [
                    f"documentPatch.cargoAllocationGroups[0].allocations[{i}].packageQuantity",
                    f"documentPatch.cargoPackages[{i}].quantity",
                ],
            )
            for i, value in enumerate([4, 5])
        ]
    }
    for b, value in zip(historical["bindings"], [4, 5], strict=True):
        b["occurrences"][0]["byte_start"] = raw.index(f" {value} COILS") + 1
    contract, missing = rebase.draft(source, historical, {}, {})
    assert not missing
    total = next(
        t["expression"] for t in contract["targets"] if ".numberAndTypeOfPackages" in t["path"]
    )
    assert " + " in total
    compile_contract(source, SourceContract.model_validate(contract))


def test_single_container_current_placement_inherits_the_proven_total_owner(rebase):
    raw = "MSKU8231362\n6 CASES\n"
    goods = {
        "numberAndTypeOfPackages": [{"packageQuantity": 6}],
        "splitGoodsPlacement": [{"equipmentIdentifier": "MSKU8231362", "packageQuantity": 6}],
    }
    source = row(raw, goods, ["MSKU8231362"])
    historical = {
        "bindings": [binding(raw, "total", "6 CASES", ["documentPatch.cargoPackages[0].quantity"])]
    }
    contract, missing = rebase.draft(source, historical, {}, {})
    assert not missing
    targets = {t["path"]: t["expression"] for t in contract["targets"]}
    assert (
        targets["documentPatch.goodsItemDetails[0].splitGoodsPlacement[0].packageQuantity"]
        == targets["documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity"]
    )
    compile_contract(source, SourceContract.model_validate(contract))
    goods["splitGoodsPlacement"][0]["equipmentIdentifier"] = "UNRELATED"
    other, _ = rebase.draft(source, historical, {}, {})
    assert not any(
        t["path"].endswith("splitGoodsPlacement[0].packageQuantity") for t in other["targets"]
    )


def test_reordered_allocations_follow_printed_equipment_not_historical_index(rebase):
    raw = "MSKU8231362\nTCLU5036160 5 PACKAGES\nTOTAL 5\n"
    source = row(
        raw,
        {
            "numberAndTypeOfPackages": [{"packageQuantity": 5}],
            "splitGoodsPlacement": [
                {"equipmentIdentifier": "MSKU8231362"},
                {"equipmentIdentifier": "TCLU5036160", "packageQuantity": 5},
            ],
        },
        ["MSKU8231362", "TCLU5036160"],
    )
    identity_path = "documentPatch.cargoAllocationGroups[0].allocations[0].containerNumber"
    historical = {
        "bindings": [
            binding(
                raw,
                "equipment",
                "TCLU5036160",
                [identity_path],
                realization={
                    "target_values": [{"target_path": identity_path, "source_value": "TCLU5036160"}]
                },
            ),
            binding(
                raw,
                "count",
                "5 PACKAGES",
                ["documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity"],
            ),
            binding(raw, "total", "TOTAL 5", ["documentPatch.cargoPackages[0].quantity"]),
        ]
    }
    contract, missing = rebase.draft(source, historical, {}, {})
    assert not missing
    assert any(
        t["path"].endswith("splitGoodsPlacement[1].packageQuantity") for t in contract["targets"]
    )
    assert not any(
        t["path"].endswith("splitGoodsPlacement[0].packageQuantity") for t in contract["targets"]
    )
    compile_contract(source, SourceContract.model_validate(contract))


@pytest.mark.parametrize(
    "unit,second_kind,complete,expected",
    [
        ("KGS", "weight", True, True),
        ("LBS", "weight", True, False),
        ("KGS", "net_weight", True, False),
        ("KGS", "weight", False, False),
    ],
)
def test_only_complete_same_unit_gross_rows_can_prove_total(
    rebase, unit, second_kind, complete, expected
):
    raw = f"MSKU8231362 10.000 {unit}\nTCLU5036160 20.000 {unit}\nPACKAGES 2\n"
    source = row(
        raw,
        {
            "grossWeight": {"value": 30.0, "unit": "kilogram"},
            "numberAndTypeOfPackages": [{"packageQuantity": 2}],
        },
        ["MSKU8231362", "TCLU5036160"],
    )
    first, second = "agent:weight:container:0", f"agent:{second_kind}:container:1"
    historical = {
        "bindings": [binding(raw, first, "10.000")]
        + ([binding(raw, second, "20.000")] if complete else [])
        + [binding(raw, "packages", "PACKAGES 2", ["documentPatch.cargoPackages[0].quantity"])]
    }
    side = {
        key: {"role": "cargo_mass", "mode": "source_scaled", "source_value": value}
        for key, value in [(first, "10.000"), (second, "20.000")]
    }
    contract, missing = rebase.draft(source, historical, side, {})
    gross = "documentPatch.goodsItemDetails[0].grossWeight.value"
    assert any(t["path"] == gross for t in contract["targets"]) is expected
    assert any(m.get("path") == gross for m in missing) is not expected
    if expected:
        compile_contract(source, SourceContract.model_validate(contract))
        assert all(
            "gross_weight container:" in v["meaning"]
            for v in contract["variables"]
            if v["kind"] == "mass"
        )


def test_stale_numeric_byte_hint_is_rejected_before_contract_creation(rebase):
    raw = "PACKAGES 6\nWRONG 7\n"
    source = row(raw, {"numberAndTypeOfPackages": [{"packageQuantity": 6}]})
    item = binding(raw, "count", "6", ["documentPatch.cargoPackages[0].quantity"])
    item["occurrences"][0]["byte_start"] = raw.index("7")
    with pytest.raises(ValueError, match="numeric source bytes differ"):
        rebase.draft(source, {"bindings": [item]}, {}, {})


@pytest.mark.parametrize(
    "key", ["agent:container:0:weight", "agent:cargo:container_measurement:gross_0"]
)
def test_private_row_roles_are_canonical_even_when_a_printed_total_already_exists(rebase, key):
    raw = "ROW 10.000 KGS\nTOTAL 30.000 KGS\n"
    source = row(raw, {"grossWeight": {"value": 30.0, "unit": "kilogram"}})
    historical = {
        "bindings": [
            binding(raw, key, "10.000"),
            binding(raw, "total", "30.000", ["documentPatch.cargoGroups[0].grossWeight.value"]),
        ]
    }
    side = {key: {"role": "cargo_mass", "mode": "source_scaled", "source_value": "10"}}
    contract, missing = rebase.draft(source, historical, side, {})
    assert not missing
    assert next(v for v in contract["variables"] if v["value"] == "10")["meaning"] == (
        "Printed cargo gross_weight container:0"
    )
