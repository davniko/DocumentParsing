"""Native unit preparation retains explicit scope and source-group atomicity."""

import asyncio
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest

from document_ocr.synthesis import independent_unit_preparation as preparation


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    frozen = {
        f"a:{variant}": {
            "sourceDocumentId": "a",
            "variant": variant,
            "regime": "ambient",
            "goods": [{"hs6": "842951", "description": "Wheel loaders"}],
            "selectedCategory": "PACKAGE_UNIT",
            "countedUnitContext": None,
            "equipmentSlots": [
                {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "GENERAL_PURPOSE"}
            ],
        }
        for variant in (1, 2, 3)
    }
    plans = {key: {"privatePackingRelation": "NONE"} for key in frozen}
    state = SimpleNamespace(requests=frozen, plans=plans, unknown=set(), calls=0)
    config = {"independent": {"requests": "requests", "private_plans": "plans"}}
    source = SimpleNamespace(
        source_ids=["a"],
        variant_counts={"a": 3},
        catalog=SimpleNamespace(limits=None),
        output=tmp_path,
        calls=SimpleNamespace(spent=Decimal(0)),
    )
    monkeypatch.setattr(
        preparation,
        "_load_source_context",
        lambda *args: SimpleNamespace(
            source=source,
            certificates={"a": {"unitProfilePrecision": {"massKg": "1", "volumeM3": "0.001"}}},
        ),
    )
    monkeypatch.setattr(preparation, "_load_pin", lambda root, key: getattr(state, key))
    monkeypatch.setattr(
        preparation,
        "unit_profile_capacities",
        lambda *args: [{"payloadKg": "28000", "volumeM3": "76"}],
    )

    async def native(runner, jobs, directory):
        state.calls += 1
        return {
            key: {
                "output": (
                    {
                        "topology": None if key in state.unknown else {"countedPerSupport": 2},
                        "basis": "Test",
                    }
                    if job[0] == preparation.composed.TOPOLOGY_STAGE
                    else {"unknown": key in state.unknown, "examples": [], "basis": "Test"}
                ),
                "error": None,
                "packet": {},
                "packetSha256": "test",
                "callReceipts": [],
            }
            for key, job in jobs.items()
        }

    monkeypatch.setattr(preparation, "_native_batch", native)

    def runnable(root, config, groups, requests, private_plans, directory):
        state.published = deepcopy(requests)
        return {"ready": True} if groups else None

    monkeypatch.setattr(preparation, "_runnable_config", runnable)
    monkeypatch.setattr(
        preparation, "IndependentCampaign", lambda *args: SimpleNamespace(plan=lambda *args: None)
    )
    state.run = lambda **kwargs: asyncio.run(
        preparation.prepare_unit_profiles(tmp_path, config, **kwargs)
    )
    return state


def test_source_compiler_is_not_assumed_to_own_frozen_requests(prepared):
    report = prepared.run()
    assert report["readySlots"] == 3 and prepared.calls == 1
    assert all("nativeUnitProfileEvidence" in row for row in prepared.published.values())
    assert all("nativeUnitProfileEvidence" not in row for row in prepared.requests.values())


def test_unknown_preserves_sibling_group_hold(prepared):
    prepared.unknown.add("a:2")
    report = prepared.run()
    assert report["readySlots"] == 0 and report["config"] is None
    assert report["inventory"]["a:2"]["status"] == "held_unknown_profile"
    assert report["inventory"]["a:1"]["status"] == "blocked_by_source_group"


def test_repeated_invocations_have_separate_journals_not_overwritten_timings(prepared):
    first, second = prepared.run(), prepared.run()
    assert first["readySlots"] == second["readySlots"] == 3
    assert first["preparedConfig"]["path"] != second["preparedConfig"]["path"]


def test_composed_profile_uses_its_own_authority_not_the_flat_profile(prepared):
    prepared.plans["a:3"].update(
        privatePackingRelation="ON_SUPPORT", relatedPackageType="PACKAGE_SKID"
    )
    report = prepared.run()
    assert report["readySlots"] == 3 and prepared.calls == 2
    assert "nativeComposedProfileEvidence" in prepared.published["a:3"]
    assert "nativeCompositionTopologyEvidence" in prepared.published["a:3"]
    assert "nativeUnitProfileEvidence" not in prepared.published["a:3"]


def test_unknown_topology_stops_numeric_call_for_that_slot(prepared):
    prepared.plans["a:3"].update(
        privatePackingRelation="ON_SUPPORT", relatedPackageType="PACKAGE_SKID"
    )
    prepared.unknown.add("a:3")
    report = prepared.run()
    assert report["readySlots"] == 0
    assert report["inventory"]["a:3"]["status"] == "held_unknown_topology"
    assert "nativeComposedProfileEvidence" not in prepared.published["a:3"]


def test_review_feedback_is_explicit_scoped_input_not_silent_output_edit(prepared):
    prepared.run(review_feedback={"a:2": "Previous estimate described a stack, not one sheet."})
    assert prepared.published["a:2"]["nativeUnitProfileEvidence"]["request"]["reviewFeedback"]
    assert "reviewFeedback" not in prepared.published["a:1"]["nativeUnitProfileEvidence"]["request"]
    with pytest.raises(ValueError, match="selected request scope"):
        prepared.run(review_feedback={"other:1": "Not selected"})


@pytest.mark.parametrize("scope", [[], ["not-selected"]])
def test_invalid_scope_cannot_silently_select_another_source(prepared, scope):
    with pytest.raises(ValueError, match="nonempty and selected"):
        prepared.run(source_ids=scope)
    assert prepared.calls == 0


def test_positive_component_rows_bound_the_profile_before_totals_are_frozen():
    cert = {
        "sourceContainerCount": 1,
        "regions": [
            {"role": "component_package", "componentCount": 2},
            {"role": "component_measurement", "componentCount": 2},
        ],
    }
    assert preparation.minimum_groups_per_load(cert, "NONE") == 2
    assert preparation.minimum_groups_per_load(cert, "CONTAINS") == 2
    assert preparation.minimum_groups_per_load(cert, "ON_SUPPORT") == 1
    cert["sourceContainerCount"] = 3
    assert preparation.minimum_groups_per_load(cert, "NONE") == 1
    cert["regions"][0]["scope"] = {"kind": "container", "index": 0}
    assert preparation.minimum_groups_per_load(cert, "SUPPORTS") == 2
