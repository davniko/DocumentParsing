from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.independent_envelopes import (
    PACKAGE_CATEGORY_FAMILIES,
    PACKAGE_COUNTING_CONTEXTS,
    configure_sampler,
    maximum_units_per_row,
    package_categories,
    package_envelope,
    package_family,
)
from document_ocr.synthesis.package_aliases import PACKAGE_ALIASES


def test_exact_flat48_inventory_and_all_envelopes_are_valid() -> None:
    assert len(package_categories()) == 48
    assert set(package_categories()) == set(PACKAGE_ALIASES)
    assert set(PACKAGE_CATEGORY_FAMILIES) == set(PACKAGE_ALIASES)
    assert set(PACKAGE_COUNTING_CONTEXTS) == set(PACKAGE_ALIASES)
    for category in package_categories():
        contexts = PACKAGE_COUNTING_CONTEXTS[category] or frozenset({None})
        for context in contexts:
            envelope = package_envelope(category, counted_unit_context=context)
            assert envelope.category == category
            assert envelope.family == package_family(category)
            assert 0 < envelope.net_kg[0] <= envelope.net_kg[1]
            assert envelope.tare_kg >= 0
            assert Decimal(0) < envelope.cube_m3 <= 24
            assert envelope.receipt()["sourceCargoAccess"] is False


@pytest.mark.parametrize(
    "category",
    sorted(category for category, contexts in PACKAGE_COUNTING_CONTEXTS.items() if contexts),
)
def test_context_sensitive_units_are_never_inferred(category: str) -> None:
    with pytest.raises(ValueError, match="requires counted_unit_context"):
        package_envelope(category)
    with pytest.raises(ValueError, match="requires counted_unit_context"):
        package_envelope(category, counted_unit_context="source_donor")


def test_vehicle_and_flexi_domains_are_narrow_and_explicit() -> None:
    vehicle = package_envelope("PACKAGE_VEHICLE", counted_unit_context="vehicle")
    assert vehicle.counted_unit_context == "vehicle"
    assert vehicle.net_kg == (500, 1800)
    flexi = package_envelope(
        "PACKAGE_FLEXITANK", counted_unit_context="non_dg_liquid_system"
    )
    assert flexi.counted_unit_context == "non_dg_liquid_system"
    assert flexi.net_kg == (11000, 15000)
    with pytest.raises(ValueError, match="requires counted_unit_context"):
        package_envelope("PACKAGE_FLEXITANK", counted_unit_context="dg_liquid")


def test_unknown_and_ordinary_contexts_never_fall_back() -> None:
    with pytest.raises(ValueError, match="outside the reviewed flat48"):
        package_envelope("PACKAGE_UNKNOWN")
    with pytest.raises(ValueError, match="does not accept"):
        package_envelope("PACKAGE_DRUM_STEEL", counted_unit_context="steel_drum")


def test_capacity_arithmetic_uses_explicit_limits() -> None:
    envelope = package_envelope("PACKAGE_DRUM_STEEL")
    assert maximum_units_per_row(
        envelope,
        unit_net_kg=Decimal("188"),
        payload_kg=Decimal("2000"),
        volume_m3=Decimal("2.8"),
    ) == 10
    with pytest.raises(ValueError, match="outside"):
        maximum_units_per_row(envelope, unit_net_kg=Decimal("201"))
    with pytest.raises(ValueError, match="does not fit"):
        maximum_units_per_row(
            envelope,
            unit_net_kg=Decimal("188"),
            payload_kg=Decimal("100"),
        )


def test_sampler_configuration_is_atomic_and_instance_local() -> None:
    original = {"packageEnvelopes": {"EXISTING": {"netKg": [1, 2]}}}
    selected = SimpleNamespace(catalog=original)
    peer = SimpleNamespace(catalog=original)
    receipts = configure_sampler(
        selected,
        categories=["PACKAGE_OCTABIN", "PACKAGE_VEHICLE"],
        category_contexts={"PACKAGE_VEHICLE": "vehicle"},
    )
    assert {item["category"] for item in receipts} == {
        "PACKAGE_OCTABIN",
        "PACKAGE_VEHICLE",
    }
    assert "OCTABIN" in selected.catalog["packageEnvelopes"]
    assert "OCTABIN" not in peer.catalog["packageEnvelopes"]
    before = selected.catalog.copy()
    with pytest.raises(ValueError, match="requires counted_unit_context"):
        configure_sampler(selected, categories=["PACKAGE_VEHICLE"])
    assert selected.catalog == before
    with pytest.raises(ValueError, match="cannot encode multiple flat categories"):
        configure_sampler(selected, categories=["PACKAGE_DRUM", "PACKAGE_DRUM_STEEL"])
    assert selected.catalog == before

