from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler import anonymous_equipment as anonymous
from document_ocr.synthesis.template_compiler import cargo_scenarios as cargo
from document_ocr.synthesis.template_compiler import complete_targets as targets
from document_ocr.synthesis.template_compiler.numeric_auxiliary import NumericContract


class Node(SimpleNamespace):
    def model_copy(self, *, update):
        return Node(**(vars(self) | update))


def fixture(monkeypatch, *, receipt="3X20'DC", heading="Gross Weight", unit="kilogram"):
    raw = f"Inventory\n{receipt}\n{heading}\n80,578.000\n".encode()

    def slot(text):
        start = raw.index(text.encode())
        return Node(source_text=text, byte_start=start, byte_end=start + len(text.encode()))

    equipment = Node(
        logical_key="fleet",
        value_kind="equipment",
        target_paths=(),
        render_mode="deterministic_auxiliary",
        group_key="equipment:shipment",
        occurrences=(slot(receipt),),
    )
    mass = Node(
        logical_key="mass",
        value_kind="decimal",
        target_paths=(),
        render_mode="deterministic_auxiliary",
        occurrences=(slot("80,578.000"),),
        dependency_paths=(),
    )
    template = Node(bindings=(equipment, mass))
    target = {"documentPatch": {"cargoGroups": [{"groupId": "g1", "description": "STEEL"}]}}
    source = targets.SourceTemplate("source", raw, deepcopy(target), target, template, "source-pin")
    contract = NumericContract(
        mode="source_scaled",
        role="cargo_mass",
        source_value="80578.000",
        target_paths=[],
        multiplier="1",
        divisor=1,
        reason="Gross column",
        synthetic_unit=unit,
    )
    monkeypatch.setattr(anonymous.numeric, "numeric_bindings", lambda _: (mass,))
    return source, {"mass": contract}


def test_private_inventory_and_units_never_become_extraction_labels(monkeypatch):
    source, contracts = fixture(monkeypatch)
    original = deepcopy(source.target)
    inventory = anonymous.compile_inventory(source, contracts)
    assert inventory is not None
    assert inventory.equipment_pairs == ("TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE",)
    physical = inventory.proposed(source.target, sample_id="s", seed=3)
    assert len(physical["documentPatch"]["containers"]) == 3
    assert physical["documentPatch"]["cargoGroups"][0]["grossWeight"]["unit"] == "kilogram"
    observed = inventory.observable(physical)
    assert physical == original == source.target
    assert "kilogram" not in source.source.decode()
    assert observed["labelVisibilityUnchanged"]
    assert len(observed["privateEquipment"]) == 3
    assert source.template.bindings[1].dependency_paths == ()


@pytest.mark.parametrize("receipt", ["40HC", "ONE CONTAINER", "0X20'DC"])
def test_inventory_requires_explicit_positive_count(monkeypatch, receipt):
    source, contracts = fixture(monkeypatch, receipt=receipt)
    with pytest.raises(ValueError):
        anonymous.compile_inventory(source, contracts)


@pytest.mark.parametrize("heading", ["Weight", "Gross Weight\nNet Weight", "Invoice Amount"])
def test_private_units_cannot_supply_missing_cargo_ownership(monkeypatch, heading):
    source, contracts = fixture(monkeypatch, heading=heading)
    with pytest.raises(ValueError, match="unique printed cargo heading"):
        anonymous.compile_inventory(source, contracts)


def test_private_measurement_cannot_override_existing_label(monkeypatch):
    source, contracts = fixture(monkeypatch)
    source.target["documentPatch"]["cargoGroups"][0]["grossWeight"] = dict(
        value=80578, unit="kilogram"
    )
    with pytest.raises(ValueError, match="contradicts its physical cargo owner"):
        anonymous.compile_inventory(source, contracts)


def test_multiple_groups_or_receipts_require_explicit_scopes(monkeypatch):
    source, contracts = fixture(monkeypatch)
    source.target["documentPatch"]["cargoGroups"].append({"groupId": "g2"})
    with pytest.raises(ValueError, match="one cargo owner"):
        anonymous.compile_inventory(source, contracts)


def test_observed_inventory_uses_existing_path(monkeypatch):
    source, contracts = fixture(monkeypatch)
    source.target["documentPatch"]["containers"] = [{"containerNumber": "ABCU1234567"}]
    assert anonymous.compile_inventory(source, contracts) is None


@pytest.mark.parametrize("receipt", ["3X20'DC FCL", "3X20'DC CONTAINERS"])
def test_complete_inventory_accepts_printed_carriage_tail(monkeypatch, receipt):
    source, contracts = fixture(monkeypatch, receipt=receipt)
    inventory = anonymous.compile_inventory(source, contracts)
    assert inventory is not None
    assert len(inventory.physical_source.target["documentPatch"]["containers"]) == 3
    assert source.source.decode().splitlines()[1] == receipt


def test_anonymous_inventory_cannot_hide_unowned_physical_facts(monkeypatch):
    source, contracts = fixture(monkeypatch, unit=None)
    with pytest.raises(ValueError, match="unowned source-only physical measurement"):
        anonymous.compile_inventory(source, contracts)


def test_public_sampler_validates_only_observable_fields_and_retains_physical_audit(monkeypatch):
    source, contracts = fixture(monkeypatch)
    inventory = anonymous.compile_inventory(source, contracts)
    weights = {inventory.equipment_pairs[0]: 1}
    support = cargo.CargoSupport.__new__(cargo.CargoSupport)
    # Only the orchestration boundary is under test; real cargo compatibility
    # is exercised by the separate catalog replay, not mocked as a quality pass.
    for field in cargo.CargoSupport.__dataclass_fields__:
        object.__setattr__(support, field, None)
    object.__setattr__(support, "validation_ids", frozenset())
    object.__setattr__(support, "config", Node(equipment_joint_weights=weights))
    equipment = cargo.EquipmentConstraints(
        source.template, weights, (), (), (), {}, (), anonymous_inventory=inventory
    )
    checks = []

    def internal(physical_source, proposed, **kwargs):
        assert physical_source is inventory.physical_source
        assert len(proposed["documentPatch"]["containers"]) == 3
        candidate = cargo.CargoScenario(proposed, {}, (), {"physicalCheck": "test-double"})
        kwargs["validate_lexical"](candidate)
        assert "containers" in candidate.target["documentPatch"]
        return candidate

    monkeypatch.setattr(cargo, "_sample_scenario", internal)
    sampled = cargo.sample_scenario(
        source,
        source.target,
        support=support,
        sample_id="s",
        seed=4,
        prepared_equipment=equipment,
        numeric_contracts=contracts,
        validate_lexical=lambda s: checks.append(deepcopy(s.target)),
    )
    assert checks == [source.target]
    assert sampled.target == source.target
    assert (
        sampled.receipt["anonymousEquipment"]["privateCargoMeasurements"]["grossWeight"]["value"]
        > 0
    )
    with pytest.raises(ValueError, match="differs from the pinned source"):
        cargo.sample_scenario(
            replace(source),
            source.target,
            support=support,
            sample_id="s",
            seed=4,
            prepared_equipment=equipment,
            numeric_contracts=contracts,
        )


@pytest.mark.parametrize("receipt,count,length", [("3X40FT", 3, "40 FT"), ("1 CTNR", 1, None)])
def test_private_inventory_keeps_unprinted_type_unlabelled(monkeypatch, receipt, count, length):
    source, contracts = fixture(monkeypatch, receipt=receipt)
    inventory = anonymous.compile_inventory(source, contracts)
    assert inventory is not None
    assert inventory.row_pairs == (None,) * count
    rows = inventory.physical_source.target["documentPatch"]["containers"]
    assert len(rows) == count
    assert all(row.get("typeDescription") == length for row in rows)
    assert all("typeCategory" not in row and "sizeCategory" not in row for row in rows)
    projected = inventory.proposed(source.target, sample_id="x", seed=2)
    inventory.observable(projected)
    assert projected == source.target


@pytest.mark.parametrize("receipt", ["20GP*1", "20GPX1", "1 X 20GP", "1X20GP FCL"])
def test_receipt_orientation_preserves_same_observed_inventory(monkeypatch, receipt):
    source, contracts = fixture(monkeypatch, receipt=receipt)
    inventory = anonymous.compile_inventory(source, contracts)
    assert inventory is not None
    assert inventory.row_pairs == ("TWENTY_FOOT_STANDARD_HEIGHT|GENERAL_PURPOSE",)


def split_fixture(monkeypatch, *, count="1 CTNR", equipment="40'HQ", same_scope=True):
    source, contracts = fixture(monkeypatch, receipt=count)
    raw = source.source + (equipment + "\n").encode()
    first = source.template.bindings[0]
    second = first.model_copy(
        update={
            "logical_key": "type",
            "group_key": first.group_key if same_scope else "another-shipment",
            "occurrences": (
                Node(
                    source_text=equipment,
                    byte_start=len(source.source),
                    byte_end=len(source.source) + len(equipment),
                ),
            ),
        }
    )
    return replace(
        source,
        source=raw,
        template=source.template.model_copy(
            update={
                "bindings": (*source.template.bindings, second),
            }
        ),
    ), contracts


def test_split_count_type_same_reviewed_scope_has_complete_private_inventory(monkeypatch):
    source, contracts = split_fixture(monkeypatch)
    inventory = anonymous.compile_inventory(source, contracts)
    assert inventory is not None
    assert inventory.binding_keys == ("fleet", "type")
    assert inventory.row_pairs == ("FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE",)
    assert source.template.bindings[0].render_mode == "deterministic_auxiliary"
    assert inventory.physical_source.template.bindings[0].dependency_paths == (
        "documentPatch.containers",
    )


def test_separate_inventory_scope_cannot_be_joined_by_proximity(monkeypatch):
    source, contracts = split_fixture(monkeypatch, same_scope=False)
    with pytest.raises(ValueError, match="reviewed inventory scope"):
        anonymous.compile_inventory(source, contracts)


@pytest.mark.parametrize("equipment", ["2X40HC", "1X20GP"])
def test_contradictory_repeated_counts_or_types_fail(monkeypatch, equipment):
    source, contracts = split_fixture(monkeypatch, count="1X40HC", equipment=equipment)
    with pytest.raises(ValueError):
        anonymous.compile_inventory(source, contracts)


def test_changed_source_span_is_rejected_before_private_inventory(monkeypatch):
    source, contracts = fixture(monkeypatch)
    source = replace(source, source=source.source.replace(b"3X20", b"4X20"))
    with pytest.raises(ValueError, match="source evidence is stale"):
        anonymous.compile_inventory(source, contracts)


@pytest.mark.parametrize("receipt", ["20 GP", "40 RH", "40 GP", "45 HC"])
def test_dimensional_prefix_is_not_an_inventory_count(monkeypatch, receipt):
    source, contracts = fixture(monkeypatch, receipt=receipt)
    with pytest.raises(ValueError, match="explicit positive inventory count"):
        anonymous.compile_inventory(source, contracts)


def test_invalid_existing_inventory_dependency_is_not_silently_replaced(monkeypatch):
    source, contracts = fixture(monkeypatch)
    binding = source.template.bindings[0].model_copy(
        update={
            "dependency_paths": ("documentPatch.cargoPackages",),
        }
    )
    source = replace(
        source,
        template=source.template.model_copy(
            update={
                "bindings": (binding, source.template.bindings[1]),
            }
        ),
    )
    with pytest.raises(ValueError, match="contradictory inventory dependency"):
        anonymous.compile_inventory(source, contracts)


def test_reviewed_numeric_equipment_component_can_supply_nonadjacent_count(monkeypatch):
    source, contracts = split_fixture(monkeypatch, count="2")
    first = source.template.bindings[0]
    first = first.model_copy(
        update={
            "group_kind": "equipment",
            "occurrences": (
                first.occurrences[0].model_copy(update={"render_policy": "numeric_surface"}),
            ),
        }
    )
    source = replace(
        source,
        template=source.template.model_copy(
            update={
                "bindings": (first, *source.template.bindings[1:]),
            }
        ),
    )
    inventory = anonymous.compile_inventory(source, contracts)
    assert inventory.row_pairs == ("FORTY_FOOT_HIGH_CUBE|GENERAL_PURPOSE",) * 2


def test_bare_untyped_number_cannot_become_an_equipment_count(monkeypatch):
    source, contracts = split_fixture(monkeypatch, count="2")
    with pytest.raises(ValueError):
        anonymous.compile_inventory(source, contracts)
