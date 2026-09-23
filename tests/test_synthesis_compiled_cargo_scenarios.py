from copy import deepcopy
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.template_compiler import cargo_scenarios as cargo

SMALL_DRY = "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE"
LARGE_DRY = "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE"


def _physical_sampling_fixture(
    target, *, text=b"", bindings=(), equipment_surfaces=("20GP", "40HC")
):
    """Small real sampling supports; no model, registry download or pipeline bypass."""
    from document_ocr.synthesis.template_compiler.complete_targets import SourceTemplate
    from document_ocr.synthesis.thermal_goods import AmbientGoodsIdentity, ThermalGoodsSupport
    from document_ocr.synthesis.transport_capacity import TransportCapacityLimits

    template = SimpleNamespace(
        bindings=bindings,
        coherence_constraints=(),
        auxiliary_semantic_plan=SimpleNamespace(dispositions=()),
    )
    source = SourceTemplate("test", text, deepcopy(target), deepcopy(target), template, "test-pin")
    limits = TransportCapacityLimits(
        policy="source_type_aware_maersk_upper_bounds_v1",
        **{
            field: Decimal("0.05")
            if field == "published_reference_margin_fraction"
            else Decimal(1000)
            for field in TransportCapacityLimits.__dataclass_fields__
            if field != "policy"
        },
    )
    goods = ThermalGoodsSupport(
        frozen=(),
        chilled=(),
        ambient=(AmbientGoodsIdentity("090111", "Coffee", "Coffee", "Green coffee"),),
        ambient_chapters=("09",),
    )
    support = SimpleNamespace(
        config=SimpleNamespace(transport_capacity=limits),
        goods=goods,
        equipment_types_by_heading={"0901": frozenset({"GENERAL_PURPOSE"})},
        equipment=cargo.build_equipment_semantic_support(
            [
                cargo.SourceEquipmentObservation("fit", str(i), surface, False)
                for i, surface in enumerate(equipment_surfaces)
            ]
        ),
        hs=SimpleNamespace(
            receipt=SimpleNamespace(snapshot_date=date(2026, 1, 1)),
            require_global=lambda *_args, **_kwargs: SimpleNamespace(valid_from=date(2020, 1, 1)),
        ),
        descriptions={"090111": "Green coffee"},
    )
    return source, support


def _source_only_binding(key, text, start, *, value_kind="equipment", owners=()):
    return SimpleNamespace(
        logical_key=key,
        value_kind=value_kind,
        target_paths=(),
        dependency_paths=owners,
        dependency_bindings=(),
        derivation=None,
        target_relationship="independent",
        render_mode="deterministic_auxiliary",
        group_kind="equipment",
        group_key="equipment:all",
        occurrences=(
            SimpleNamespace(
                slot_id=key,
                source_text=text,
                byte_start=start,
                byte_end=start + len(text.encode()),
                render_policy="numeric_surface"
                if value_kind == "decimal_measurement"
                else "natural_text",
            ),
        ),
    )


@pytest.mark.parametrize("printed_type", [False, True])
def test_container_only_document_does_not_invent_cargo_or_unprinted_equipment(printed_type):
    from document_ocr.synthesis.generators import DeterministicStream
    from document_ocr.synthesis.transport_capacity import TransportCapacityLimits

    container = {"containerNumber": "ABCU0000001"}
    if printed_type:
        container.update(sizeCategory="TWENTY_FOOT_STANDARD_HEIGHT", typeCategory="GENERAL_PURPOSE")
    target = {"documentPatch": {"containers": [container]}}
    template = SimpleNamespace(bindings=(), coherence_constraints=())
    source = SimpleNamespace(source=b"Container ABCU0000001", target=target, template=template)
    weights = {"TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 1}
    limits = TransportCapacityLimits(
        policy="source_type_aware_maersk_upper_bounds_v1",
        **{
            field: Decimal("0.05")
            if field == "published_reference_margin_fraction"
            else Decimal(1000)
            for field in TransportCapacityLimits.__dataclass_fields__
            if field != "policy"
        },
    )
    support = SimpleNamespace(
        config=SimpleNamespace(transport_capacity=limits),
        goods=SimpleNamespace(ambient=()),
        equipment_types_by_heading={},
        equipment=cargo.build_equipment_semantic_support(
            [cargo.SourceEquipmentObservation("fit", "row", "20GP", False)]
        ),
    )
    equipment = cargo.EquipmentConstraints(template, weights, ((0,),), (weights,), (), {}, ())
    cargo.require_contract(source)
    sampled = cargo._sample_candidate(
        source, target, support, DeterministicStream(1, "test", "s"), equipment, {}, {}, {}, {}
    )
    assert sampled.target == target
    assert not sampled.identities
    assert sampled.receipt["equipment"][container["containerNumber"]]["basis"] == (
        "configured_equipment_without_observed_cargo"
    )
    assert sampled.receipt["capacity"]["valid"]


def test_actual_candidate_samples_tare_jointly_and_keeps_independent_receipt():
    from document_ocr.synthesis.generators import DeterministicStream
    from document_ocr.synthesis.template_compiler.equipment_tares import TareSupport
    from document_ocr.synthesis.template_compiler.numeric_auxiliary import (
        NumericContract,
        numeric_bindings,
        prepare,
        render_prepared,
    )

    raw = b"ABCU0000001\nTARE 3700 KGS\nCOPY TARE WEIGHT 3700 KGS\n"
    binding = _source_only_binding(
        "tare",
        "3700",
        raw.index(b"3700"),
        value_kind="decimal_measurement",
        owners=("documentPatch.containers[0]",),
    )
    binding.occurrences += (
        SimpleNamespace(
            slot_id="tare-repeat",
            source_text="3700",
            byte_start=raw.rindex(b"3700"),
            byte_end=raw.rindex(b"3700") + 4,
            render_policy="numeric_surface",
        ),
    )
    target = {"documentPatch": {"containers": [{"containerNumber": "ABCU0000001"}]}}
    source, support = _physical_sampling_fixture(target, text=raw, bindings=(binding,))
    original = deepcopy(source.target)
    contract = NumericContract(
        mode="sampled_equipment_tare",
        role="tare",
        source_value="3700",
        target_paths=[],
        multiplier="1",
        divisor=1,
        reason="Sample a train-supported row tare, preserving ownership",
    )
    weights = {SMALL_DRY: 1, LARGE_DRY: 1}
    tares = TareSupport(
        {Decimal(2200): frozenset({SMALL_DRY}), Decimal(3800): frozenset({LARGE_DRY})},
        {(Decimal(2200), SMALL_DRY): ("fit-small",), (Decimal(3800), LARGE_DRY): ("fit-large",)},
    )
    equipment = cargo.equipment_constraints(
        source,
        SimpleNamespace(equipment_joint_weights=weights),
        numeric_contracts={"tare": contract},
        tare_support=tares,
    )
    choices = set()
    for seed in range(12):
        sampled = cargo._sample_candidate(
            source,
            target,
            support,
            DeterministicStream(seed, "test", "s"),
            equipment,
            {},
            {},
            {},
            {},
        )
        receipt = sampled.receipt
        physical = receipt["equipment"]["ABCU0000001"]
        pair = physical["size"] + "|" + physical["type"]
        value = {SMALL_DRY: "2200", LARGE_DRY: "3800"}[pair]
        assert receipt["sampledEquipmentTares"] == {"tare": value}
        assert receipt["privateEquipmentTaresKg"] == {"0": value}
        assert sampled.target == target
        choices.add(pair)
        numbers = {key: Decimal(v) for key, v in receipt["sampledEquipmentTares"].items()}
        prepared = prepare(
            numeric_bindings(source.template),
            {"tare": contract},
            source_target=source.target,
            target=sampled.target,
            scale=Decimal("0.8"),
            source_template=source.template,
            equipment_tare_values=numbers,
        )
        rendered = render_prepared(
            numeric_bindings(source.template),
            prepared,
            source_target=source.target,
            target=sampled.target,
            source_template=source.template,
            equipment_tare_values=numbers,
        )
        assert rendered["tare"] == {"tare": value, "tare-repeat": value}
        with pytest.raises(ValueError, match="differs from its dependency contract"):
            render_prepared(
                numeric_bindings(source.template),
                prepared,
                source_target=source.target,
                target=sampled.target,
                source_template=source.template,
                equipment_tare_values={"tare": Decimal(value) + 1},
            )
        with pytest.raises(ValueError, match="required for a generated scenario"):
            prepare(
                numeric_bindings(source.template),
                {"tare": contract},
                source_target=source.target,
                target=sampled.target,
                scale=Decimal("0.8"),
                source_template=source.template,
            )
    assert choices == {SMALL_DRY, LARGE_DRY}
    assert source.target == original and source.source == raw


def test_actual_candidate_retains_fixed_tare_and_observed_equipment_pair():
    from document_ocr.synthesis.generators import DeterministicStream
    from document_ocr.synthesis.template_compiler.numeric_auxiliary import (
        NumericContract,
        numeric_bindings,
        prepare,
        render_prepared,
    )

    raw = b"ABCU0000001\nTARE 3800 KGS\n"
    binding = _source_only_binding(
        "tare",
        "3800",
        raw.index(b"3800"),
        value_kind="decimal_measurement",
        owners=("documentPatch.containers[0]",),
    )
    target = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "ABCU0000001",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        }
    }
    source, support = _physical_sampling_fixture(target, text=raw, bindings=(binding,))
    contract = NumericContract(
        mode="source_fixed",
        role="tare",
        source_value="3800",
        target_paths=[],
        multiplier="1",
        divisor=1,
        reason="Retain a printed fixed tare and its observed equipment",
    )
    equipment = cargo.equipment_constraints(
        source,
        SimpleNamespace(equipment_joint_weights={SMALL_DRY: 100, LARGE_DRY: 1}),
        numeric_contracts={"tare": contract},
    )
    assert equipment.weights == ({LARGE_DRY: 1},)
    for seed in range(4):
        sampled = cargo._sample_candidate(
            source,
            target,
            support,
            DeterministicStream(seed, "test", "fixed-tare"),
            equipment,
            {},
            {},
            {},
            {},
        )
        assert sampled.target == target
        assert sampled.receipt["sampledEquipmentTares"] == {}
        assert sampled.receipt["equipment"]["ABCU0000001"]["retainedTareBindings"] == ["tare"]
        prepared = prepare(
            numeric_bindings(source.template),
            {"tare": contract},
            source_target=source.target,
            target=sampled.target,
            scale=Decimal("1.2"),
            source_template=source.template,
            equipment_tare_values={},
        )
        assert render_prepared(
            numeric_bindings(source.template),
            prepared,
            source_target=source.target,
            target=sampled.target,
            source_template=source.template,
            equipment_tare_values={},
        ) == {"tare": {"tare": "3800"}}


def _mixed_sampling_fixture(*, second="1X40HC", cargo_group=False):
    first = "1X20ST"
    raw = (first + "\n" + second).encode()
    bindings = tuple(
        _source_only_binding(key, value, start, owners=("documentPatch.containers",))
        for key, value, start in (("small", first, 0), ("large", second, len(first) + 1))
    )
    target = {
        "documentPatch": {
            "containers": [{"containerNumber": "ABCU0000001"}, {"containerNumber": "ABCU0000002"}],
        }
    }
    if cargo_group:
        target["documentPatch"]["cargoGroups"] = [{"groupId": "g1", "hsCodes": ["090111"]}]
    source, support = _physical_sampling_fixture(
        target,
        text=raw,
        bindings=bindings,
        equipment_surfaces=("20GP", "40OT" if second == "1X40OT" else "40HC"),
    )
    second_pair = "FORTY_FOOT_STANDARD_HEIGHT|OPEN_TOP" if second == "1X40OT" else LARGE_DRY
    equipment = cargo.equipment_constraints(
        source, SimpleNamespace(equipment_joint_weights={SMALL_DRY: 1, second_pair: 1})
    )
    return source, support, equipment


def test_actual_candidate_respects_mixed_multiset_without_inventing_type_to_id_labels():
    from document_ocr.synthesis.generators import DeterministicStream

    source, support, equipment = _mixed_sampling_fixture()
    before = deepcopy(source.target)
    assignments = set()
    for seed in range(12):
        result = cargo._sample_candidate(
            source,
            source.target,
            support,
            DeterministicStream(seed, "test", "mixed"),
            equipment,
            {},
            {},
            {},
            {},
        )
        audit = result.receipt["mixedEquipmentInventory"]
        assert audit["observedAggregate"] == {SMALL_DRY: 1, LARGE_DRY: 1}
        assert audit["originalTypeToIdentifierMappingClaimed"] is False
        assert audit["labelVisibilityUnchanged"] is True
        assignments.add(tuple(audit["privateSyntheticAssignment"][i] for i in range(2)))
        assert result.target == before
        assert result.receipt["sampledEquipmentTares"] == {}
    assert assignments == {(SMALL_DRY, LARGE_DRY), (LARGE_DRY, SMALL_DRY)}
    assert source.target == before


def test_actual_candidate_rejects_mixed_inventory_without_common_goods_support():
    from document_ocr.synthesis.generators import DeterministicStream
    from document_ocr.synthesis.thermal_goods import AmbientGoodsIdentity

    source, support, equipment = _mixed_sampling_fixture(second="1X40OT", cargo_group=True)
    support.goods = replace(
        support.goods,
        ambient=(
            *support.goods.ambient,
            AmbientGoodsIdentity("100630", "Cereals", "Rice", "Rice"),
        ),
    )
    support.equipment_types_by_heading = {
        "0901": frozenset({"GENERAL_PURPOSE"}),
        "1006": frozenset({"OPEN_TOP"}),
    }
    domains = {
        "g1": (
            1,
            {
                "GENERAL_PURPOSE": frozenset({((), "090111")}),
                "OPEN_TOP": frozenset({((), "100630")}),
            },
        )
    }
    for seed in range(4):
        with pytest.raises(ValueError, match="equipment envelope lacks joint goods support"):
            cargo._sample_candidate(
                source,
                source.target,
                support,
                DeterministicStream(seed, "test", "mixed"),
                equipment,
                {},
                {},
                domains,
                {},
            )


def test_actual_candidate_enforces_package_mass_floor_without_inventing_weight_labels():
    from document_ocr.synthesis.generators import DeterministicStream
    from document_ocr.synthesis.package_goods_compatibility import build_package_goods_fit_support
    from document_ocr.synthesis.template_compiler.numeric_auxiliary import NumericContract

    raw = b"40 X 25 KG BAGS"
    count = _source_only_binding("quantity", "40", 0, value_kind="integer")
    count.target_paths = ("documentPatch.cargoPackages[0].quantity",)
    unit = _source_only_binding(
        "unit-mass",
        "25",
        5,
        value_kind="decimal_measurement",
        owners=("documentPatch.cargoPackages[0]",),
    )
    target = {
        "documentPatch": {
            "containers": [{"containerNumber": "ABCU0000001"}],
            "cargoGroups": [{"groupId": "g1", "hsCodes": ["090111"]}],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 40,
                    "typeCategory": "PACKAGE_BAG",
                }
            ],
        }
    }
    source, support = _physical_sampling_fixture(target, text=raw, bindings=(count, unit))
    support.packages = build_package_goods_fit_support(
        source_targets={"fit": target},
        fit_document_ids=("fit",),
        allowed_category_tokens=("PACKAGE_BAG",),
        frozen_minimum_celsius=-24,
        frozen_maximum_celsius=-18,
        chilled_minimum_celsius=-3,
        chilled_maximum_celsius=5.5,
    )
    support.measurements = cargo.cargo_measurements.build_support(
        {"fit": target},
        lower_multiplier=0.5,
        upper_multiplier=2,
    )
    contract = NumericContract(
        mode="source_fixed",
        role="per_unit_measurement",
        source_value="25",
        target_paths=[],
        multiplier="1",
        divisor=1,
        reason="Each declared bag contains 25 printed kilograms",
    )
    equipment = cargo.equipment_constraints(
        source,
        SimpleNamespace(equipment_joint_weights={SMALL_DRY: 1}),
        numeric_contracts={"unit-mass": contract},
    )
    assert len(equipment.package_unit_loads) == 1
    scenario_args = (
        equipment,
        {"g1": None},
        {"g1": (frozenset({"PACKAGE_BAG"}),)},
        {"g1": (1, {"GENERAL_PURPOSE": frozenset({(("PACKAGE_BAG",), "090111")})})},
        {"g1": {}},
    )
    sampled = cargo._sample_candidate(
        source,
        target,
        support,
        DeterministicStream(1, "test", "package"),
        *scenario_args,
    )
    assert sampled.receipt["privatePackageMassFloorKg"] == {"g1": "1000"}
    assert sampled.receipt["capacity"]["valid"] is True
    assert sampled.receipt["capacity"]["gross_weight_kg"] == 1000
    assert sampled.target == target
    assert "grossWeight" not in sampled.target["documentPatch"]["cargoGroups"][0]
    assert "netWeight" not in sampled.target["documentPatch"]["cargoGroups"][0]
    too_heavy = deepcopy(target)
    too_heavy["documentPatch"]["cargoPackages"][0]["quantity"] = 43
    with pytest.raises(ValueError, match="equipment capacity is insufficient"):
        cargo._sample_candidate(
            source,
            too_heavy,
            support,
            DeterministicStream(1, "test", "package"),
            *scenario_args,
        )
    assert source.target == target and source.source == raw


def test_absent_cargo_labels_do_not_hide_a_source_only_cargo_statement():
    source = SimpleNamespace(
        target={"documentPatch": {}},
        template=SimpleNamespace(
            bindings=(SimpleNamespace(group_kind="cargo", render_mode="agent_required"),)
        ),
    )
    with pytest.raises(ValueError, match="source-only cargo requires a physical owner"):
        cargo.require_contract(source)


def test_summary_tariff_cannot_be_independent_of_bare_product_names():
    source = SimpleNamespace(
        target={
            "documentPatch": {
                "cargoGroups": [
                    {
                        "groupId": "summary",
                        "hsCodes": ["080410"],
                        "netWeight": {"unit": "kilogram", "value": 120},
                    },
                    {"groupId": "item1", "description": "AJWA DATES"},
                    {"groupId": "item2", "description": "SAFAWI DATES"},
                ],
                "cargoPackages": [{"groupId": "summary", "packageId": "p", "quantity": 10}],
            }
        }
    )
    with pytest.raises(ValueError, match="summary and bare item groups"):
        cargo.require_contract(source)


@pytest.mark.parametrize("owns_package", [True, False])
def test_independent_item_groups_with_their_own_observations_remain_supported(owns_package):
    target = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "hsCodes": ["080410"]},
                {"groupId": "g2", "description": "COFFEE BEANS"},
            ],
            "cargoPackages": [],
        }
    }
    if owns_package:
        target["documentPatch"]["cargoPackages"].append(
            {"groupId": "g2", "packageId": "p", "quantity": 10}
        )
    else:
        target["documentPatch"]["cargoGroups"][1]["hsCodes"] = ["090111"]
    source = SimpleNamespace(
        source=b"COFFEE BEANS",
        target=target,
        template=SimpleNamespace(bindings=(), coherence_constraints=()),
    )
    cargo.require_contract(source)


def test_derived_measurements_preflight_with_the_final_unit_aware_adapter(monkeypatch):
    path = "documentPatch.cargoGroups[0].grossWeight.value"
    original = {"documentPatch": {"cargoGroups": [{"grossWeight": {"value": 100}}]}}
    target = {"documentPatch": {"cargoGroups": [{"grossWeight": {"value": 80}}]}}
    binding = SimpleNamespace(
        target_paths=(path,),
        derivation=None,
        realization=SimpleNamespace(mode="deterministic_derivation", requires_agent=False),
    )
    source = SimpleNamespace(
        source=b"100 KGS",
        target=original,
        source_target=original,
        template=SimpleNamespace(bindings=(binding,), byte_template="byte"),
    )
    output = object()
    calls = []
    monkeypatch.setattr(cargo.render, "_has_semantic_equipment_values", lambda *args: False)

    def measured(owner, case):
        assert owner is binding and case.source_target is original and case.target is target
        calls.append("typed measurement")
        return output

    monkeypatch.setattr(cargo.render, "_render_target_measurement", measured)

    def wrong_adapter(*args, **kwargs):
        pytest.fail("A derived measurement cannot use the generic text-surface adapter")

    monkeypatch.setattr(cargo.render, "_render_target_binding", wrong_adapter)

    def checked(**kwargs):
        assert kwargs["output"] is output
        calls.append("exact byte envelope")

    monkeypatch.setattr(cargo.render, "_validate_binding_format", checked)
    cargo._validate_sampled_bindings(source, target)
    assert calls == ["typed measurement", "exact byte envelope"]


def test_measurement_freedom_is_decided_before_sampling_from_exact_dependency_contracts(
    monkeypatch,
):
    from document_ocr.synthesis.template_compiler.numeric_auxiliary import NumericContract

    source = SimpleNamespace(
        template=SimpleNamespace(bindings=()),
        target={
            "documentPatch": {
                "cargoGroups": [
                    {
                        "groupId": "g1",
                        "grossWeight": {"value": 100, "unit": "kilogram"},
                        "netWeight": {"value": 90, "unit": "kilogram"},
                    }
                ]
            }
        },
    )
    equipment = SimpleNamespace(row_measurements=())
    monkeypatch.setattr(cargo.targets, "_numeric_quantum", lambda *a: Decimal("0.1"))
    assert cargo._measurement_quanta(source, {}, equipment) == {
        "g1": {"grossWeight": Decimal("0.1"), "netWeight": Decimal("0.1")}
    }
    equation = NumericContract(
        mode="unit_product",
        role="per_unit_measurement",
        source_value="9",
        target_paths=[
            "documentPatch.cargoPackages[0].quantity",
            "documentPatch.cargoGroups[0].netWeight.value",
        ],
        multiplier="1",
        divisor=1,
        reason="Explicit per-package net equation",
    )
    assert cargo._measurement_quanta(source, {"per-unit": equation}, equipment) == {
        "g1": {"grossWeight": Decimal("0.1")}
    }
    fixed = equation.model_copy(update={"mode": "source_fixed", "target_paths": []})
    assert cargo._measurement_quanta(source, {"per-unit": fixed}, equipment) == {"g1": {}}
    assert cargo._measurement_quanta(source, {}, SimpleNamespace(row_measurements=("row",))) == {
        "g1": {}
    }


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("GENERAL PURPOSE CONT.", (None, "GENERAL_PURPOSE", None)),
        ("refrigerated container", (None, "REFRIGERATED", None)),
        ("20' FLATRACK COLLAPSIBLE", ("20", "PLATFORM_COLLAPSIBLE", None)),
        ("40HIGH CUBE", ("40", None, "FORTY_FOOT_HIGH_CUBE")),
        ("CTNR", (None, None, None)),
        ("LCL.", (None, None, None)),
        ("1 x 40RA", ("40", "REFRIGERATED", None)),
        ("01 X 40RQ", ("40", "REFRIGERATED", None)),
        ("20DRX1", (None, "GENERAL_PURPOSE", "TWENTY_FOOT_STANDARD_HEIGHT")),
    ],
)
def test_private_equipment_constraints_preserve_partial_visibility(description, expected):
    assert cargo._partial_equipment_constraint({"typeDescription": description}) == expected


@pytest.mark.parametrize("description", ["2 x 40RA", "20DRX2", "1 x 40UNKNOWN", "1x40RAx2"])
def test_single_container_description_does_not_absorb_multiple_or_unknown_equipment(description):
    with pytest.raises(ValueError, match="reviewed physical constraint"):
        cargo._partial_equipment_constraint({"typeDescription": description})


def test_printed_per_package_equation_locks_its_measurement(monkeypatch):
    source = SimpleNamespace(
        template=SimpleNamespace(bindings=()),
        target={
            "documentPatch": {
                "cargoGroups": [
                    {
                        "groupId": "g1",
                        "description": "25 KG NET EACH BAG",
                        "netWeight": {"value": 250, "unit": "kilogram"},
                        "grossWeight": {"value": 270, "unit": "kilogram"},
                    }
                ],
                "cargoPackages": [{"groupId": "g1", "quantity": 10}],
            }
        },
    )
    monkeypatch.setattr(cargo.targets, "_numeric_quantum", lambda *a: Decimal("1"))
    assert cargo._measurement_quanta(source, {}, SimpleNamespace(row_measurements=())) == {
        "g1": {"grossWeight": Decimal("1")}
    }


@pytest.mark.parametrize("owner", ["typeCategory", "typeDescription"])
def test_package_surface_mass_equation_is_locked_before_joint_measurement_sampling(
    monkeypatch, owner
):
    # Real failure: the proposal proved 960 boxes * 25 kg, but joint sampling
    # later treated that same net mass as free because its prose lived in a
    # package-type binding instead of the cargo description.
    source = SimpleNamespace(
        template=SimpleNamespace(
            bindings=(
                SimpleNamespace(
                    target_paths=(f"documentPatch.cargoPackages[0].{owner}",),
                    occurrences=(SimpleNamespace(source_text="BOXES OF 25KG EACH"),),
                ),
            )
        ),
        target={
            "documentPatch": {
                "cargoGroups": [
                    {
                        "groupId": "g1",
                        "netWeight": {"value": 24000, "unit": "kilogram"},
                        "grossWeight": {"value": 24720, "unit": "kilogram"},
                        "volume": {"value": 50, "unit": "cubic_metre"},
                    }
                ],
                "cargoPackages": [{"groupId": "g1", "quantity": 960}],
            }
        },
    )
    monkeypatch.setattr(cargo.targets, "_numeric_quantum", lambda *a: Decimal("0.1"))
    assert cargo._measurement_quanta(source, {}, SimpleNamespace(row_measurements=())) == {
        "g1": {"grossWeight": Decimal("0.1"), "volume": Decimal("0.1")}
    }
    source.target["documentPatch"]["cargoGroups"][0]["netWeight"]["value"] = 23999
    with pytest.raises(ValueError, match="exact source group-total proof"):
        cargo._measurement_quanta(source, {}, SimpleNamespace(row_measurements=()))


def test_equipment_weight_contract_rejects_unknown_semantics():
    import json

    value = json.loads(
        Path("configs/synthesis/production/mpci_bl_diversified_cargo_sampling_v1.json").read_bytes()
    )
    value["equipment_joint_weights"] = {"FORTY_FOOT_HIGH_CUBE|MADE_UP_CONTAINER": 1}
    with pytest.raises(ValidationError):
        cargo.CargoSamplingConfig.model_validate_json(json.dumps(value))


def test_sampling_exhaustion_never_returns_the_source(monkeypatch):
    source = SimpleNamespace(
        document_id="source", target={"documentPatch": {}}, template=SimpleNamespace(bindings=())
    )
    support = SimpleNamespace(
        validation_ids=frozenset(), config=SimpleNamespace(maximum_candidates=3), tares=None
    )
    monkeypatch.setattr(cargo, "require_contract", lambda _: None)
    monkeypatch.setattr(
        cargo, "equipment_constraints", lambda *a, **kw: SimpleNamespace(row_measurements=())
    )
    attempts = []

    def reject(*args):
        attempts.append(args)
        raise ValueError("incompatible cargo")

    monkeypatch.setattr(cargo, "_sample_candidate", reject)
    with pytest.raises(ValueError, match="no representable joint cargo scenario"):
        cargo.sample_scenario(source, {"documentPatch": {}}, support=support, sample_id="s", seed=1)
    assert len(attempts) == 3


def test_validation_documents_are_rejected_before_sampling(monkeypatch):
    source = SimpleNamespace(document_id="held-out")
    support = SimpleNamespace(validation_ids=frozenset({"held-out"}))
    with pytest.raises(ValueError, match="validation source"):
        cargo.sample_scenario(source, {}, support=support, sample_id="s", seed=1)


def test_lexical_representability_is_checked_inside_local_sampling(monkeypatch):
    source = SimpleNamespace(
        document_id="source", target={"documentPatch": {}}, template=SimpleNamespace(bindings=())
    )
    support = SimpleNamespace(
        validation_ids=frozenset(), config=SimpleNamespace(maximum_candidates=2), tares=None
    )
    monkeypatch.setattr(cargo, "require_contract", lambda _: None)
    monkeypatch.setattr(
        cargo, "equipment_constraints", lambda *a, **kw: SimpleNamespace(row_measurements=())
    )
    monkeypatch.setattr(cargo, "_sample_candidate", lambda *a: cargo.CargoScenario({}, {}, (), {}))
    checked = []

    def validate(candidate):
        checked.append(candidate)
        if len(checked) == 1:
            raise ValueError("wording cannot fill source slots")

    result = cargo.sample_scenario(
        source,
        {"documentPatch": {}},
        support=support,
        sample_id="s",
        seed=1,
        validate_lexical=validate,
    )
    assert len(checked) == 2
    assert result.receipt["candidateAttempts"] == 2
    assert result.receipt["rejectedCandidates"] == {"wording cannot fill source slots": 1}


def scenario():
    target = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "NEW HYDROCHLORIC ACID INDUSTRIAL GRADE",
                    "hsCodes": ["280610"],
                    "dangerousGoods": [
                        {"unNumber": "1789", "hazardCategory": "CORROSIVE_SUBSTANCES"}
                    ],
                }
            ],
            "containers": [
                {"sizeCategory": "FORTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE"}
            ],
        }
    }
    return cargo.CargoScenario(
        target, {"g1": [{"properShippingName": "Hydrochloric acid"}]}, (), {}
    )


@pytest.mark.parametrize("field", ["unNumber", "hazardCategory", "hsCode", "equipment"])
def test_linguistic_generation_cannot_overwrite_sampled_semantic_facts(field):
    sampled = scenario()
    target = deepcopy(sampled.target)
    group = target["documentPatch"]["cargoGroups"][0]
    if field == "hsCode":
        group["hsCodes"][0] = "999999"
    elif field == "equipment":
        target["documentPatch"]["containers"][0]["typeCategory"] = "OPEN_TOP"
    else:
        group["dangerousGoods"][0][field] = "WRONG"
    with pytest.raises(ValueError, match="sampled cargo fact changed"):
        cargo.validate_final(sampled, target)


def test_dg_description_must_express_the_new_chemical_not_an_old_commercial_variant():
    sampled = scenario()
    cargo.validate_final(sampled, sampled.target)
    target = deepcopy(sampled.target)
    target["documentPatch"]["cargoGroups"][0]["description"] = "New electronic arc lighters"
    with pytest.raises(ValueError, match="proper shipping name"):
        cargo.validate_final(sampled, target)
    # Structured preflight deliberately precedes creation of linguistic content.
    cargo.validate_structured_facts(sampled, target)


def test_cargo_prose_cannot_reintroduce_a_stale_un_number():
    sampled = scenario()
    target = deepcopy(sampled.target)
    target["documentPatch"]["cargoGroups"][0]["description"] += " UN 1057"
    with pytest.raises(ValueError, match="UN number"):
        cargo.validate_final(sampled, target)


def test_normal_goods_cannot_keep_the_previous_product_under_a_new_hs_code():
    sampled = scenario()
    group = sampled.target["documentPatch"]["cargoGroups"][0]
    group.pop("dangerousGoods")
    group["hsCodes"] = ["821410"]
    required = "Paperknives, letter openers, erasing knives, pencil sharpeners and blades therefor"
    sampled.identities["g1"] = [{"requiredDescription": required}]
    group["description"] = "Stainless steel precision craft scissors with protective sleeves"
    with pytest.raises(ValueError, match="authoritative goods description"):
        cargo.validate_final(sampled, sampled.target)
    group["description"] = required + ", stainless steel assortment"
    cargo.validate_final(sampled, sampled.target)


def test_missingness_is_not_filled_with_unrequested_training_label_fields():
    source = SimpleNamespace(
        source=b"12 packages",
        target={"documentPatch": {"cargoPackages": [{"quantity": 12}]}},
        template=SimpleNamespace(bindings=[], coherence_constraints=[]),
    )
    cargo.require_contract(source)
    physical = deepcopy(source.target)
    physical["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_CARTON"
    assert cargo.package_observations.observable_target(source.target, physical) == source.target


def test_unprinted_equipment_can_have_a_latent_physical_envelope_without_new_labels():
    source = SimpleNamespace(
        target={"documentPatch": {"containers": [{"containerNumber": "ABCU1234567"}]}},
        template=SimpleNamespace(bindings=[]),
    )
    target = deepcopy(source.target)
    target["documentPatch"]["containers"][0].update(
        containerNumber="NEWU7654321",
        sizeCategory="FORTY_FOOT_HIGH_CUBE",
        typeCategory="GENERAL_PURPOSE",
    )
    assert cargo.latent_equipment_indices(source) == frozenset({0})
    projected = cargo._observable_equipment_target(source, target)
    assert projected["documentPatch"]["containers"] == [{"containerNumber": "NEWU7654321"}]
    assert source.target["documentPatch"]["containers"] == [{"containerNumber": "ABCU1234567"}]


@pytest.mark.parametrize(
    "field,value",
    [
        ("typeDescription", "40 UNKNOWN"),
        ("typeCategory", "GENERAL_PURPOSE"),
    ],
)
def test_unreviewed_or_invalid_partial_equipment_is_not_silently_inferred(field, value):
    source = SimpleNamespace(
        target={
            "documentPatch": {"containers": [{"containerNumber": "ABCU1234567", field: value}]}
        },
        template=SimpleNamespace(bindings=[]),
    )
    with pytest.raises(ValueError):
        cargo.latent_equipment_indices(source)


def test_printed_temperature_without_equipment_type_constrains_private_choices_only():
    container = {
        "containerNumber": "ABCU1234567",
        "temperatureSetpoint": {"value": -18.0, "unit": "celsius"},
    }
    source = SimpleNamespace(
        source=b"ABCU1234567 TEMP -18C",
        target={"documentPatch": {"containers": [container]}},
        template=SimpleNamespace(bindings=[]),
    )
    config = SimpleNamespace(
        equipment_joint_weights={
            "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 1,
            "FORTY_FOOT_HIGH_CUBE|REFRIGERATED": 2,
        }
    )
    assert cargo.latent_equipment_indices(source) == frozenset({0})
    constraints = cargo.equipment_constraints(source, config)
    assert constraints.weights == ({"FORTY_FOOT_HIGH_CUBE|REFRIGERATED": 2},)
    physical = deepcopy(source.target)
    physical["documentPatch"]["containers"][0].update(
        sizeCategory="FORTY_FOOT_HIGH_CUBE", typeCategory="REFRIGERATED"
    )
    assert cargo._observable_equipment_target(source, physical) == source.target


def test_source_only_equipment_requires_a_real_owner_before_latent_sampling():
    source = SimpleNamespace(
        target={"documentPatch": {"containers": [{"containerNumber": "ABCU1234567"}]}},
        template=SimpleNamespace(
            bindings=[
                SimpleNamespace(
                    logical_key="unowned_equipment",
                    render_mode="deterministic_auxiliary",
                    value_kind="equipment",
                    target_paths=(),
                    dependency_paths=(),
                    derivation=None,
                )
            ]
        ),
    )
    with pytest.raises(ValueError, match="explicit latent owner"):
        cargo.latent_equipment_indices(source)


@pytest.mark.parametrize("surface,description", [("1 CONTAINER", None), ("1 x 20'", "20'")])
def test_owned_count_or_partial_receipt_does_not_require_unseen_equipment_labels(
    monkeypatch, surface, description
):
    container = {"containerNumber": "ABCU1234567"}
    if description is not None:
        container["typeDescription"] = description
    source = SimpleNamespace(
        source=surface.encode(),
        target={"documentPatch": {"containers": [container]}},
        template=SimpleNamespace(
            bindings=[
                SimpleNamespace(
                    logical_key="equipment_receipt",
                    value_kind="equipment",
                    derivation="equipment_receipt",
                    target_paths=(),
                    dependency_paths=("documentPatch.containers[0]",),
                    occurrences=(SimpleNamespace(slot_id="count", source_text=surface),),
                )
            ]
        ),
    )
    monkeypatch.setattr(cargo.numeric_auxiliary, "numeric_bindings", lambda _: ())
    assert cargo.latent_equipment_indices(source) == frozenset({0})
    assert "sizeCategory" not in container and "typeCategory" not in container


def test_owned_receipt_constrains_private_equipment_without_inventing_labels(monkeypatch):
    source = SimpleNamespace(
        source=b"1X40HC",
        target={"documentPatch": {"containers": [{"containerNumber": "ABCU1234567"}]}},
        template=SimpleNamespace(
            byte_template=None,
            bindings=[
                SimpleNamespace(
                    logical_key="equipment_receipt",
                    value_kind="equipment",
                    derivation="equipment_receipt",
                    target_paths=(),
                    dependency_paths=("documentPatch.containers[0]",),
                    occurrences=(SimpleNamespace(slot_id="count", source_text="1X40HC"),),
                )
            ],
        ),
    )
    monkeypatch.setattr(cargo.numeric_auxiliary, "numeric_bindings", lambda _: ())
    # This test isolates the physical domain; receipt grammar and byte-format
    # validation are exercised independently by the receipt integration tests.
    monkeypatch.setattr(cargo.render, "_validate_binding_format", lambda **_: None)
    config = SimpleNamespace(
        equipment_joint_weights={
            "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 1,
            "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE": 2,
            "FORTY_FOOT_HIGH_CUBE|REFRIGERATED": 1,
        }
    )
    assert cargo.latent_equipment_indices(source) == frozenset({0})
    constraints = cargo.equipment_constraints(source, config)
    assert constraints.groups == ((0,),)
    assert constraints.weights == ({"FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE": 2},)
    assert source.target == {"documentPatch": {"containers": [{"containerNumber": "ABCU1234567"}]}}


def test_dimensions_only_receipt_and_printed_type_jointly_constrain_private_equipment(monkeypatch):
    surface = "1 CONT. 20'X8'6\""
    source = SimpleNamespace(
        source=surface.encode(),
        target={
            "documentPatch": {
                "containers": [
                    {"containerNumber": "ABCU1234567", "typeDescription": "GENERAL PURPOSE CONT."}
                ]
            }
        },
        template=SimpleNamespace(
            byte_template=None,
            bindings=[
                SimpleNamespace(
                    logical_key="equipment_receipt",
                    value_kind="equipment",
                    derivation="equipment_receipt",
                    target_paths=(),
                    dependency_paths=("documentPatch.containers[0]",),
                    occurrences=(SimpleNamespace(slot_id="count", source_text=surface),),
                )
            ],
        ),
    )
    original = deepcopy(source.target)
    monkeypatch.setattr(cargo.numeric_auxiliary, "numeric_bindings", lambda _: ())
    monkeypatch.setattr(cargo.render, "_validate_binding_format", lambda **_: None)
    config = SimpleNamespace(
        equipment_joint_weights={
            "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 1,
            "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE": 1,
            "TWENTY_FOOT_STANDARD_HEIGHT|REFRIGERATED": 1,
        }
    )
    constraints = cargo.equipment_constraints(source, config)
    assert constraints.weights == ({"TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 1},)
    assert source.target == original


def test_linguistic_generation_cannot_publish_a_latent_equipment_category():
    target = {"documentPatch": {"containers": [{"containerNumber": "NEWU7654321"}]}}
    sampled = cargo.CargoScenario(target, {}, (), {})
    invented = deepcopy(target)
    invented["documentPatch"]["containers"][0]["typeCategory"] = "GENERAL_PURPOSE"
    with pytest.raises(ValueError, match="equipment label visibility changed"):
        cargo.validate_structured_facts(sampled, invented)


@pytest.mark.parametrize(
    "mode,role,allowed",
    [
        ("source_scaled", "cargo_mass", False),
        ("source_fixed", "operational", True),
        ("target_share", "cargo_mass", True),
        ("source_scaled", "cargo_volume", False),
    ],
)
def test_unprinted_equipment_requires_proved_physical_numeric_dependencies(
    monkeypatch, mode, role, allowed
):
    binding = SimpleNamespace(
        logical_key="measurement",
        render_mode="deterministic_auxiliary",
        value_kind="decimal_measurement",
        target_paths=(),
        derivation=None,
        occurrences=(),
    )
    source = SimpleNamespace(
        source=b"",
        target={"documentPatch": {"containers": [{"containerNumber": "ABCU1234567"}]}},
        template=SimpleNamespace(bindings=[binding]),
    )
    monkeypatch.setattr(cargo.numeric_auxiliary, "numeric_bindings", lambda _: (binding,))
    config = SimpleNamespace(
        equipment_joint_weights={"TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 1}
    )
    contracts = {
        "measurement": cargo.numeric_auxiliary.NumericContract(
            mode=mode,
            role=role,
            source_value="1",
            target_paths=[],
            multiplier="1",
            divisor=1,
            reason="Exercise physical dependency admission without printed occurrences.",
        )
    }
    if allowed:
        assert cargo.equipment_constraints(source, config, numeric_contracts=contracts).groups == (
            (0,),
        )
    else:
        with pytest.raises(ValueError, match="numeric facts in the physical scenario"):
            cargo.equipment_constraints(source, config, numeric_contracts=contracts)


@pytest.mark.parametrize(
    "description,size",
    [
        ("20'", "TWENTY_FOOT_STANDARD_HEIGHT"),
        ("40", "FORTY_FOOT_HIGH_CUBE"),
        ("45FT", "FORTY_FIVE_FOOT_HIGH_CUBE"),
        ("40'RQ", "FORTY_FOOT_HIGH_CUBE"),
        ("40RO", "FORTY_FOOT_HIGH_CUBE"),
        ("1 x 40RA", "FORTY_FOOT_HIGH_CUBE"),
    ],
)
def test_partial_equipment_retains_only_observed_labels(description, size):
    source = SimpleNamespace(
        target={
            "documentPatch": {
                "containers": [{"containerNumber": "OLDU1234567", "typeDescription": description}]
            }
        },
        template=SimpleNamespace(bindings=[]),
    )
    target = deepcopy(source.target)
    target["documentPatch"]["containers"][0].update(
        containerNumber="NEWU7654321",
        sizeCategory=size,
        typeCategory="REFRIGERATED",
    )
    assert cargo._observable_equipment_target(source, target) == {
        "documentPatch": {
            "containers": [{"containerNumber": "NEWU7654321", "typeDescription": description}]
        }
    }


@pytest.mark.parametrize(
    "description,expected",
    [
        ("20'", {"TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE"}),
        (
            "40'",
            {"FORTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE", "FORTY_FOOT_HIGH_CUBE|REFRIGERATED"},
        ),
        ("40RQ", {"FORTY_FOOT_HIGH_CUBE|REFRIGERATED"}),
        ("1 x 40RA", {"FORTY_FOOT_HIGH_CUBE|REFRIGERATED"}),
        ("20DRX1", {"TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE"}),
    ],
)
def test_partial_equipment_constrains_candidates_before_sampling(description, expected):
    source = SimpleNamespace(
        target={"documentPatch": {"containers": [{"typeDescription": description}]}},
        template=SimpleNamespace(bindings=[]),
    )
    config = SimpleNamespace(
        equipment_joint_weights={
            "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 1,
            "FORTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 1,
            "FORTY_FOOT_HIGH_CUBE|REFRIGERATED": 1,
            "FORTY_FIVE_FOOT_HIGH_CUBE|GENERAL_PURPOSE": 1,
        }
    )
    constraints = cargo.equipment_constraints(source, config)
    assert set(constraints.weights[0]) == expected


def test_composite_equipment_receipt_uses_its_inventory_renderer(monkeypatch):
    binding = SimpleNamespace(
        logical_key="equipment_receipt",
        value_kind="equipment",
        derivation="equipment_receipt",
        target_paths=("documentPatch.containers[0]", "documentPatch.containers[0].typeDescription"),
        dependency_paths=("documentPatch.containers[0].typeDescription",),
        occurrences=(SimpleNamespace(slot_id="s", source_text="01 X 40'HC"),),
    )
    original = {
        "documentPatch": {
            "containers": [
                {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}
            ]
        }
    }
    source = SimpleNamespace(
        source=b"01 X 40'HC",
        target=original,
        template=SimpleNamespace(bindings=(binding,), byte_template=object()),
    )
    monkeypatch.setattr(cargo.numeric_auxiliary, "numeric_bindings", lambda _: ())
    monkeypatch.setattr(cargo.equipment_row_constraints, "compile_rows", lambda *a: ())
    outputs = []
    monkeypatch.setattr(
        cargo.render,
        "_validate_binding_format",
        lambda **kwargs: outputs.append(kwargs["output"].replacements["s"]),
    )
    weights = {
        "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 1,
        "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE": 1,
    }
    result = cargo.equipment_constraints(source, SimpleNamespace(equipment_joint_weights=weights))
    assert result.weights == (weights,)
    assert any("20" in text for text in outputs)
    assert original == source.target
    target = deepcopy(original)
    target["documentPatch"]["containers"][0]["sizeCategory"] = "TWENTY_FOOT_STANDARD_HEIGHT"
    cargo._validate_sampled_bindings(source, target)
    assert "20" in outputs[-1]
    binding.occurrences = (SimpleNamespace(slot_id="s", source_text="02 X 40'HC"),)
    with pytest.raises(ValueError, match="source count/type"):
        cargo.equipment_constraints(source, SimpleNamespace(equipment_joint_weights=weights))


def test_fixed_tare_constrains_equipment_before_sampling(monkeypatch):
    binding = SimpleNamespace(
        logical_key="tare",
        target_paths=(),
        dependency_paths=(),
        derivation=None,
        value_kind="decimal_measurement",
        occurrences=(),
    )
    source = SimpleNamespace(
        source=b"TARE 3770 KGS",
        target={
            "documentPatch": {
                "containers": [
                    {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}
                ]
            }
        },
        template=SimpleNamespace(bindings=[binding]),
    )
    monkeypatch.setattr(cargo.numeric_auxiliary, "numeric_bindings", lambda _: (binding,))
    monkeypatch.setattr(cargo, "latent_equipment_indices", lambda *_, **__: frozenset())
    config = SimpleNamespace(
        equipment_joint_weights={
            "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 100,
            "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE": 1,
            "FORTY_FOOT_HIGH_CUBE|REFRIGERATED": 100,
        }
    )
    contract = cargo.numeric_auxiliary.NumericContract(
        mode="source_fixed",
        role="tare",
        source_value="3770",
        target_paths=[],
        multiplier="1",
        divisor=1,
        reason="Printed fixed equipment tare",
    )
    with pytest.raises(ValueError, match="complete source numeric contracts"):
        cargo.equipment_constraints(source, config)
    constrained = cargo.equipment_constraints(source, config, numeric_contracts={"tare": contract})
    assert constrained.weights == ({"FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE": 1},)
    assert constrained.retained_tare_bindings == ("tare",)
    source.target["documentPatch"]["containers"][0].pop("sizeCategory")
    source.target["documentPatch"]["containers"][0].pop("typeCategory")
    monkeypatch.setattr(cargo, "latent_equipment_indices", lambda *_, **__: frozenset({0}))
    with pytest.raises(ValueError, match="observed equipment envelope"):
        cargo.equipment_constraints(source, config, numeric_contracts={"tare": contract})


def test_unconstrained_equipment_remains_varied():
    source = SimpleNamespace(
        target={
            "documentPatch": {
                "containers": [
                    {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}
                ]
            }
        },
        template=SimpleNamespace(bindings=[]),
    )
    weights = {
        "TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE": 1,
        "FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE": 1,
    }
    result = cargo.equipment_constraints(source, SimpleNamespace(equipment_joint_weights=weights))
    assert result.weights == (weights,)
    assert not result.retained_tare_bindings


def test_only_explicit_shared_hs_owners_merge_identity_components():
    def binding(a, b, relationship="shared_value_equality"):
        return SimpleNamespace(
            target_paths=(
                f"documentPatch.cargoGroups[{a}].hsCodes[0]",
                f"documentPatch.cargoGroups[{b}].hsCodes[0]",
            ),
            target_relationship=relationship,
        )

    template = SimpleNamespace(
        bindings=[binding(0, 1), binding(1, 2), binding(3, 4, "independent")]
    )
    components = cargo._shared_hs_components(template)
    assert components == {(0, 0): "shared-hs-0-0", (1, 0): "shared-hs-0-0", (2, 0): "shared-hs-0-0"}


@pytest.mark.parametrize(
    ("codes", "indices"),
    [
        (["52010020", "5201000090"], (0, 0)),
        (["0202300000", "0202.30.00", "020230"], (0, 0, 0)),
        (["52010020", "520210", "5201000090"], (0, 1, 0)),
        ([], (0,)),
    ],
)
def test_national_tariff_variants_share_a_goods_class_not_a_printed_code(codes, indices):
    assert cargo._tariff_identity_indices(codes) == indices


def test_short_tariff_classification_is_not_silently_completed():
    with pytest.raises(ValueError, match="fewer than six"):
        cargo._tariff_identity_indices(["5201"])


def test_goods_support_requires_distinct_codes_within_one_package_signature():
    bales = ("printed-package:bales",)
    cartons = ("PACKAGE_CARTON",)
    assert cargo._has_identity_support(frozenset({(bales, "520100")}), 1)
    assert not cargo._has_identity_support(frozenset({(bales, "520100")}), 2)
    assert not cargo._has_identity_support(frozenset({(bales, "520100"), (cartons, "520210")}), 2)
    assert cargo._has_identity_support(frozenset({(bales, "520100"), (bales, "520210")}), 2)


def test_equipment_conditioning_keeps_heterogeneous_types_only_with_common_goods():
    bales = ("printed-package:bales",)
    dry = frozenset({(bales, "520100"), (bales, "520210")})
    open_top = frozenset({(bales, "520210"), (bales, "530500")})
    # Choosing either type first still allows mixed types for a shared supported
    # commodity; unrelated goods in the union must never make a false match.
    assert cargo._has_identity_support(dry & open_top, 1)
    assert not cargo._has_identity_support(dry & open_top, 2)
    assert not cargo._has_identity_support(dry & frozenset({(bales, "530500")}), 1)


def test_linguistic_generation_cannot_change_a_retained_partial_equipment_observation():
    target = {"documentPatch": {"containers": [{"typeDescription": "20'"}]}}
    sampled = cargo.CargoScenario(target, {}, (), {})
    changed = deepcopy(target)
    changed["documentPatch"]["containers"][0]["typeDescription"] = "40'"
    with pytest.raises(ValueError, match="sampled cargo fact changed"):
        cargo.validate_structured_facts(sampled, changed)


@pytest.mark.parametrize("change_description", [False, True])
def test_latent_projection_rejects_contradictory_physical_or_printed_equipment(change_description):
    source = SimpleNamespace(
        target={"documentPatch": {"containers": [{"typeDescription": "20'"}]}},
        template=SimpleNamespace(bindings=[]),
    )
    target = deepcopy(source.target)
    target["documentPatch"]["containers"][0].update(
        sizeCategory="FORTY_FOOT_HIGH_CUBE", typeCategory="GENERAL_PURPOSE"
    )
    if change_description:
        target["documentPatch"]["containers"][0]["typeDescription"] = "40'"
    with pytest.raises(ValueError, match=r"equipment.*(?:contract|observation)"):
        cargo._observable_equipment_target(source, target)


@pytest.mark.parametrize("surface", [b"1 INTERMEDIATE BULK CONTAINERS", b"24 CARTONS", b"12DRUMS"])
def test_unlabelled_printed_packaging_requires_an_explicit_sampling_contract(surface):
    source = SimpleNamespace(
        source=surface,
        target={"documentPatch": {}},
        template=SimpleNamespace(bindings=[], coherence_constraints=[]),
    )
    with pytest.raises(ValueError, match="printed packaging"):
        cargo.require_contract(source)


@pytest.mark.parametrize(
    "surface",
    [
        b"Total No. of Containers received by the Carrier: 1\nPackages received by the Carrier:",
        b"Where a container is loaded with more than one package or unit, the packages...",
    ],
)
def test_package_guard_does_not_treat_headings_or_conditional_terms_as_declarations(surface):
    source = SimpleNamespace(
        source=surface,
        target={"documentPatch": {}},
        template=SimpleNamespace(bindings=[], coherence_constraints=[]),
    )
    cargo.require_contract(source)


def test_every_equipment_category_has_a_semantically_complete_surface():
    from document_ocr.label_schemas.bill_of_lading_v5 import (
        CONTAINER_SIZE_CATEGORIES,
        CONTAINER_TYPE_CATEGORIES,
    )
    from document_ocr.synthesis.template_compiler.descendant import (
        _equipment_semantics_match,
        _equipment_surface_candidates,
    )

    for size in CONTAINER_SIZE_CATEGORIES:
        for kind in CONTAINER_TYPE_CATEGORIES:
            value = {"sizeCategory": size, "typeCategory": kind}
            assert any(
                _equipment_semantics_match(value, text)
                for text in _equipment_surface_candidates(value, "40 FOOT GENERAL PURPOSE")
            )


@pytest.mark.parametrize("length", [20, 40])
@pytest.mark.parametrize("separator", ["'X", "X", " X "])
def test_explicit_equipment_height_cannot_be_silently_classified_as_standard(length, separator):
    from document_ocr.synthesis.container_semantics import review_source_equipment_surface
    from document_ocr.synthesis.template_compiler.descendant import _equipment_semantics_match

    surface = f"{length}{separator}9'6\" REEFER CONTAINER"
    prefix = "TWENTY" if length == 20 else "FORTY"
    reviewed = review_source_equipment_surface(surface, temperature_present=True)
    assert reviewed.size_category == f"{prefix}_FOOT_HIGH_CUBE"
    assert not _equipment_semantics_match(
        {"sizeCategory": f"{prefix}_FOOT_STANDARD_HEIGHT", "typeCategory": "REFRIGERATED"},
        surface,
    )


@pytest.mark.parametrize(
    "category",
    [
        "PACKAGE_DRUM_FIBRE",
        "PACKAGE_BOX_PLYWOOD",
        "PACKAGE_INTERMEDIATE_BULK_CONTAINER",
        "PACKAGE_PIECE",
        "PACKAGE_UNPACKED_OR_UNPACKAGED",
    ],
)
def test_package_rendering_is_not_restricted_to_the_ten_abbreviation_categories(category):
    from document_ocr.synthesis.package_registry import package_category_surface_present
    from document_ocr.synthesis.template_compiler.descendant import _package_candidate

    rendered = _package_candidate("CARTONS", category)
    assert "PACKAGE_" not in rendered
    assert package_category_surface_present(rendered, category)
