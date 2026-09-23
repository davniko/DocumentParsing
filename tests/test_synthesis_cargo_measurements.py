from copy import deepcopy
from decimal import Decimal

import pytest

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.template_compiler.cargo_measurements import (
    MeasurementSupport,
    build_support,
    measures_per_package,
)


def test_missing_quantity_uses_observed_mass_per_container_without_inventing_a_count():
    group = {
        "groupId": "g1",
        "hsCodes": ["392190"],
        "grossWeight": {"value": 10000, "unit": "kilogram"},
    }
    source = {
        "documentPatch": {
            "cargoGroups": [group],
            "cargoPackages": [{"groupId": "g1", "quantity": 100, "typeCategory": "PACKAGE_ROLL"}],
            "containers": [{"containerNumber": "ONE"}],
        }
    }
    support = build_support({"fit": source}, lower_multiplier=0.5, upper_multiplier=2)
    packages = [{"typeCategory": "PACKAGE_ROLL"}]
    assert measures_per_package(group, packages, container_count=1) == {
        "grossWeightPerContainer": 10000
    }
    assert support.compatible("392190", group, packages, container_count=1)
    assert not support.compatible("392190", group, packages, container_count=10)
    assert "quantity" not in packages[0]


def test_vehicle_units_cannot_inherit_light_merchandise_numeric_scenario():
    support = MeasurementSupport(
        {("8703", ("PACKAGE_UNIT",)): ({"grossWeight": 1200, "volume": 8},)}, 0.5, 2
    )
    packages = [{"typeCategory": "PACKAGE_UNIT", "quantity": 126}]
    assert not support.compatible(
        "870331",
        {
            "grossWeight": {"value": 1751.974, "unit": "kilogram"},
            "volume": {"value": 7.762, "unit": "cubic_metre"},
        },
        packages,
    )
    assert support.compatible(
        "870331",
        {
            "grossWeight": {"value": 150000, "unit": "kilogram"},
            "volume": {"value": 1000, "unit": "cubic_metre"},
        },
        packages,
    )


def test_missing_support_is_not_a_permissive_fallback():
    support = MeasurementSupport({}, 0.5, 2)
    assert not support.compatible(
        "870331",
        {"grossWeight": {"value": 100, "unit": "kilogram"}},
        [{"typeCategory": "PACKAGE_UNIT", "quantity": 1}],
    )


def test_mass_and_volume_must_share_one_observed_cargo_group():
    support = MeasurementSupport(
        {
            ("1234", ("PACKAGE_BOX",)): (
                {"grossWeight": 1000, "volume": 1},
                {"grossWeight": 1, "volume": 1000},
            )
        },
        0.5,
        2,
    )
    assert not support.compatible(
        "123456",
        {
            "grossWeight": {"value": 1000, "unit": "kilogram"},
            "volume": {"value": 1000, "unit": "cubic_metre"},
        },
        [{"typeCategory": "PACKAGE_BOX", "quantity": 1}],
    )


def test_unmeasured_piece_counts_cannot_be_promoted_to_thousands_of_handling_units():
    support = MeasurementSupport(
        {
            ("9405", ("PACKAGE_PALLET",)): ({"packagesPerContainer": 20},),
            ("3907", ("PACKAGE_INTERMEDIATE_BULK_CONTAINER",)): ({"packagesPerContainer": 24},),
            ("4411", ("PACKAGE_PIECE",)): ({"packagesPerContainer": 1500},),
        },
        0.5,
        2,
    )
    packages = [{"typeCategory": "PACKAGE_PIECE", "quantity": 1090}]
    assert support.allowed_heading_signatures({}, packages, container_count=1) == frozenset(
        {("4411", ("PACKAGE_PIECE",))}
    )
    for hs, category in [
        ("940591", "PACKAGE_PALLET"),
        ("390721", "PACKAGE_INTERMEDIATE_BULK_CONTAINER"),
    ]:
        assert not support.compatible(
            hs, {}, [{"typeCategory": category, "quantity": 1090}], container_count=1
        )
    assert support.compatible(
        "940591", {}, [{"typeCategory": "PACKAGE_PALLET", "quantity": 40}], container_count=2
    )


def test_free_measurements_are_generated_jointly_not_rejected_for_source_ratios():
    support = MeasurementSupport(
        {
            ("9405", ("PACKAGE_PALLET",)): (
                {"grossWeight": 200, "netWeight": 180, "volume": 2, "packagesPerContainer": 10},
                {"grossWeight": 500, "netWeight": 470, "volume": 3, "packagesPerContainer": 10},
            )
        },
        0.5,
        2,
    )
    packages = [{"typeCategory": "PACKAGE_PALLET", "quantity": 10}]
    original = {
        "grossWeight": {"value": 100, "unit": "kilogram"},
        "netWeight": {"value": 90, "unit": "kilogram"},
        "volume": {"value": 1, "unit": "cubic_metre"},
    }
    assert support.allowed_heading_signatures(original, packages, container_count=1) == frozenset()
    assert support.allowed_heading_signatures(
        original, packages, container_count=1, mutable_fields=frozenset(original)
    ) == frozenset({("9405", ("PACKAGE_PALLET",))})
    values = set()
    for seed in range(20):
        group = deepcopy(original)
        receipt = support.draw_measures(
            "940591",
            group,
            packages,
            quanta={k: Decimal("0.1") for k in original},
            container_count=1,
            stream=DeterministicStream(seed, "test", "sample"),
        )
        values.add(tuple(group[k]["value"] for k in original))
        assert group.keys() == original.keys()
        assert support.compatible("940591", group, packages, container_count=1)
        assert receipt["fields"] == sorted(original)
    assert values == {(2000, 1800, 20), (5000, 4700, 30)}


def test_joint_measurement_respects_locked_equations_and_does_not_add_missing_measures():
    support = MeasurementSupport(
        {("1234", ("PACKAGE_BOX",)): ({"grossWeight": 100, "netWeight": 90, "volume": 2},)}, 0.5, 2
    )
    packages = [{"typeCategory": "PACKAGE_BOX", "quantity": 2}]
    group = {
        "grossWeight": {"value": 190, "unit": "kilogram"},
        "netWeight": {"value": 180, "unit": "kilogram"},
    }
    support.draw_measures(
        "123456",
        group,
        packages,
        quanta={"grossWeight": Decimal(1)},
        container_count=None,
        stream=DeterministicStream(1, "test", "s"),
    )
    assert group == {
        "grossWeight": {"value": 200, "unit": "kilogram"},
        "netWeight": {"value": 180, "unit": "kilogram"},
    }
    bad = {**group, "netWeight": {"value": 1000, "unit": "kilogram"}}
    with pytest.raises(ValueError, match="no conditioned fit point"):
        support.draw_measures(
            "123456",
            bad,
            packages,
            quanta={"grossWeight": Decimal(1)},
            container_count=None,
            stream=DeterministicStream(1, "test", "s"),
        )
    assert bad["netWeight"]["value"] == 1000


def test_missing_quantity_uses_fit_mass_per_container_without_inventing_quantity():
    support = MeasurementSupport(
        {("1234", ("PACKAGE_BOX",)): ({"grossWeight": 100, "grossWeightPerContainer": 10000},)},
        0.5,
        2,
    )
    group = {"grossWeight": {"value": 1, "unit": "metric_tonne"}}
    packages = [{"typeCategory": "PACKAGE_BOX"}]
    support.draw_measures(
        "123456",
        group,
        packages,
        quanta={"grossWeight": Decimal("0.001")},
        container_count=2,
        stream=DeterministicStream(1, "test", "s"),
    )
    assert group["grossWeight"] == {"value": 20, "unit": "metric_tonne"}
    assert "quantity" not in packages[0]


def test_joint_fit_exploration_preserves_density_quantities_visibility_and_bounds():
    support = MeasurementSupport(
        {
            ("9405", ("PACKAGE_PALLET",)): (
                {"grossWeight": 200, "netWeight": 180, "volume": 2, "packagesPerContainer": 10},
            )
        },
        0.5,
        2,
    )
    packages = [{"typeCategory": "PACKAGE_PALLET", "quantity": 10}]
    original = {
        "grossWeight": {"value": 100, "unit": "kilogram"},
        "netWeight": {"value": 90, "unit": "kilogram"},
        "volume": {"value": 1, "unit": "cubic_metre"},
    }
    values = set()
    for seed in range(40):
        group = deepcopy(original)
        receipt = support.draw_measures(
            "940591",
            group,
            packages,
            quanta={k: Decimal("0.001") for k in original},
            container_count=1,
            stream=DeterministicStream(seed, "test", "s"),
            vary_fit_scale=True,
        )
        factor = Decimal(receipt["commonFitScale"])
        assert Decimal("0.5") <= factor <= Decimal(2)
        for field, base in (("grossWeight", 2000), ("netWeight", 1800), ("volume", 20)):
            assert abs(Decimal(str(group[field]["value"])) - Decimal(base) * factor) <= Decimal(
                "0.0005"
            )
        assert group.keys() == original.keys()
        assert support.compatible("940591", group, packages, container_count=1)
        assert packages == [{"typeCategory": "PACKAGE_PALLET", "quantity": 10}]
        values.add(group["grossWeight"]["value"])
    assert len(values) > 30


def test_locked_measurements_do_not_get_a_hidden_independent_fit_multiplier():
    support = MeasurementSupport(
        {("1234", ("PACKAGE_BOX",)): ({"grossWeight": 100, "netWeight": 90},)},
        0.5,
        2,
    )
    group = {
        "grossWeight": {"value": 200, "unit": "kilogram"},
        "netWeight": {"value": 180, "unit": "kilogram"},
    }
    receipt = support.draw_measures(
        "123456",
        group,
        [{"typeCategory": "PACKAGE_BOX", "quantity": 2}],
        quanta={"grossWeight": Decimal(1)},
        container_count=None,
        stream=DeterministicStream(1, "test", "s"),
        vary_fit_scale=True,
    )
    assert group["grossWeight"]["value"] == 200
    assert group["netWeight"]["value"] == 180
    assert receipt["commonFitScale"] == "1"
    assert receipt["fitScalePolicy"] == "locked_measurements_preserve_exact_fit_point"
