"""The chooser and numeric sampler must use identical carrying hardware."""

import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.independent_campaign import IndependentCampaign
from document_ocr.synthesis.independent_packages import (
    equipment_choice_brief,
    verify_native_choice_evidence,
)
from document_ocr.synthesis.independent_sampling import sample_equipment

GP20 = {"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT", "typeCategory": "GENERAL_PURPOSE"}
GP40 = {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}
OT40 = {"sizeCategory": "FORTY_FOOT_STANDARD_HEIGHT", "typeCategory": "OPEN_TOP"}


@pytest.mark.parametrize("size", [
    "TWENTY_FOOT_STANDARD_HEIGHT", "FORTY_FOOT_STANDARD_HEIGHT", "FORTY_FOOT_HIGH_CUBE",
])
def test_platform_hardware_is_not_briefed_as_closed_box(size):
    row = {"sizeCategory": size, "typeCategory": "PLATFORM"}
    kwargs = dict(regime="ambient", equipment_mode="uniform", seed=91,
                  draw_key="platform", container_count=1, equipment_domain=[row])
    assert sample_equipment(**kwargs) == [row]
    assert sample_equipment(**kwargs, equipment_slots=[row]) == [row]
    brief = equipment_choice_brief([row])
    assert "platform" in brief and "no enclosed-box" in brief
    assert "liquids need self-contained discrete" in brief
    assert "These are closed-box" not in brief


@pytest.mark.parametrize("regime,domain", [
    ("chilled", [{**OT40, "typeCategory": "PLATFORM"}]),
    ("frozen", [{**OT40, "typeCategory": "PLATFORM"}]),
    ("dg", [{**OT40, "typeCategory": "PLATFORM"}]),
    ("ambient", [{"sizeCategory": "FORTY_FIVE_FOOT_HIGH_CUBE", "typeCategory": "PLATFORM"}]),
    ("ambient", [{**OT40, "typeCategory": "PLATFORM"}, GP40]),
])
def test_platform_domain_cannot_evade_regime_or_mix_with_closed_box(regime, domain):
    with pytest.raises(ValueError):
        sample_equipment(regime=regime, equipment_mode="uniform", seed=91,
                         draw_key="invalid-platform", container_count=1, equipment_domain=domain)


def test_explicit_open_top_preserves_source_hardware_and_numeric_freeze():
    kwargs = dict(regime="ambient", equipment_mode="uniform", seed=91,
                  draw_key="open-top", container_count=2, equipment_domain=[OT40])
    assert sample_equipment(**kwargs) == [OT40, OT40]
    assert sample_equipment(**kwargs, equipment_slots=[OT40, OT40]) == [OT40, OT40]
    brief = equipment_choice_brief([OT40])
    assert "open-top containers with side walls" in brief
    assert "normal length, width and height" in brief
    assert "liquids need self-contained discrete" in brief
    assert "closed-box" not in brief


def test_explicit_ambient_reefer_is_hardware_not_a_thermal_goods_draw():
    row = {"sizeCategory": "FORTY_FOOT_STANDARD_HEIGHT", "typeCategory": "REFRIGERATED"}
    assert sample_equipment(regime="ambient", equipment_mode="uniform", seed=91,
                            draw_key="nor", container_count=2,
                            equipment_domain=[row], equipment_slots=[row, row]) == [row, row]
    default = sample_equipment(regime="ambient", equipment_mode="uniform", seed=91,
                               draw_key="still-dry", container_count=2)
    assert all(r["typeCategory"] == "GENERAL_PURPOSE" for r in default)


@pytest.mark.parametrize("regime,domain", [
    ("chilled", [OT40]), ("frozen", [OT40]), ("dg", [OT40]),
    ("ambient", [{**OT40, "sizeCategory": "FORTY_FIVE_FOOT_HIGH_CUBE"}]),
    ("ambient", [OT40, GP40]),
])
def test_open_top_cannot_evade_cargo_or_hardware_family_domain(regime, domain):
    with pytest.raises(ValueError):
        sample_equipment(regime=regime, equipment_mode="uniform", seed=91,
                         draw_key="invalid-ot", container_count=1, equipment_domain=domain)


@pytest.mark.parametrize("regime", ["ambient", "chilled", "frozen", "dg"])
@pytest.mark.parametrize("mode,count", [("uniform", 0), ("uniform", 1), ("mixed", 5)])
def test_independent_equipment_freeze_survives_numeric_retries(regime, mode, count):
    slots = sample_equipment(regime=regime, equipment_mode=mode, seed=91,
                            draw_key="before-package", container_count=count)
    original = deepcopy(slots)
    for attempt in range(3):
        assert sample_equipment(
            regime=regime, equipment_mode=mode, seed=91,
            draw_key=f"numeric-attempt-{attempt}", container_count=count,
            equipment_slots=slots,
        ) == original
    assert slots == original


@pytest.mark.parametrize("changes,error", [
    ({"container_count": 1}, "container count"),
    ({"equipment_domain": [GP40]}, "outside the certified"),
    ({"equipment_mode": "uniform", "equipment_slots": [GP20, GP40]}, "uniform equipment"),
    ({"regime": "chilled"}, "outside the certified"),
    ({"equipment_mode": "mixed"}, "two distinct"),
])
def test_frozen_equipment_cannot_evade_source_or_mode_constraints(changes, error):
    kwargs = dict(regime="ambient", equipment_mode="uniform", seed=91,
                  draw_key="frozen", container_count=2, equipment_slots=[GP20, GP20])
    kwargs.update(changes)
    with pytest.raises(ValueError, match=error):
        sample_equipment(**kwargs)


@pytest.mark.parametrize("slots", [None, "20GP", [{"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT"}]])
def test_equipment_brief_rejects_missing_or_partial_hardware(slots):
    with pytest.raises(ValueError, match="hardware"):
        equipment_choice_brief(slots)


def test_equipment_brief_is_explicit_and_does_not_invent_unprinted_hardware():
    assert "no printed carrying containers" in equipment_choice_brief([])
    text = equipment_choice_brief([GP20, GP40])
    assert "TWENTY_FOOT_STANDARD_HEIGHT" in text
    assert "FORTY_FOOT_HIGH_CUBE" in text
    assert "closed-box" in text and "not flat racks" in text


@pytest.mark.parametrize("slots,error", [(None, "explicit list"), ([GP40], "differs from frozen")])
def test_native_evidence_cannot_be_reused_for_other_hardware(slots, error):
    native = dict(goods=[{"hs6": "520100", "description": "Cotton"}], regime="ambient",
                  equipment=equipment_choice_brief([GP20]), form="Cotton")
    frozen = dict(goods=native["goods"], regime="ambient", goodsForm="Cotton",
                  dangerousGoods=None, equipmentSlots=slots,
                  nativeChoiceEvidence={"request": native})
    with pytest.raises(ValueError, match=error):
        verify_native_choice_evidence(Path("."), frozen, hs_codes={}, provider={})


@pytest.mark.parametrize("conflict", [False, True])
def test_campaign_passes_frozen_hardware_to_numeric_sampler(monkeypatch, conflict):
    campaign = IndependentCampaign.__new__(IndependentCampaign)
    campaign._plan_overrides = {}
    campaign._planned = {}
    campaign._plan_lock_guard = threading.Lock()
    campaign._plan_locks = {}
    campaign.variant_counts = {"source": 1}
    campaign.config = {"seed": 47}
    campaign.requests = {"source:1": {
        "regime": "ambient", "equipmentSlots": [GP20],
        "selectedCategory": "PACKAGE_BALE", "countedUnitContext": None,
    }}
    campaign.private_plans = {"source:1": {}}
    campaign.physical_support = object()
    campaign.blueprint = lambda _: SimpleNamespace(target={
        "documentPatch": {"containerInformation": [{}]},
    })
    campaign._planning_capability = lambda *_: SimpleNamespace(
        family="ambient", quantity_multiple=1,
    )
    campaign.rows = {"source": {}}
    campaign.catalog = SimpleNamespace(
        _geography=lambda *_: (None, None, None, None, None),
        config=SimpleNamespace(maximum_candidates=1),
    )
    monkeypatch.setattr(
        "document_ocr.synthesis.independent_campaign.source_equipment_readiness",
        lambda *_: (None, [], [GP40] if conflict else None),
    )

    class ObservedDraw(Exception):
        pass

    def draw(**kwargs):
        assert kwargs["equipment_slots"] == [GP20]
        raise ObservedDraw

    campaign.sampler = SimpleNamespace(draw=draw)
    if conflict:
        with pytest.raises(ValueError, match="source-certified mix"):
            campaign.plan("source", 1)
    else:
        with pytest.raises(ObservedDraw):
            campaign.plan("source", 1)
