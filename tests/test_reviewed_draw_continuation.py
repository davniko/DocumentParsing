"""Inspection continuation must retain already adjudicated registry draws."""

import importlib.util
import json
from pathlib import Path

_PATH = (
    Path(__file__).parents[1]
    / "docs/analysis/r28-donor-free-full-registry-20261021/complete_reviewed_draws.py"
)
_SPEC = importlib.util.spec_from_file_location("reviewed_draw_continuation", _PATH)
completion = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(completion)


def state(tmp_path, name, requests):
    (tmp_path / (name + "-requests.json")).write_text(json.dumps(requests))
    (tmp_path / (name + "-plans.json")).write_text(
        json.dumps({k: {"stage": name} for k in requests})
    )
    return {
        "independent": {
            "requests": {"path": name + "-requests.json"},
            "private_plans": {"path": name + "-plans.json"},
        }
    }


def test_runnable_inspection_retains_previous_redraw_instead_of_initial_unknown(tmp_path):
    initial = state(tmp_path, "initial", {"s:1": {"goods": "old-one"}})
    current = state(
        tmp_path, "current", {"s:1": {"goods": "new-one"}, "s:2": {"goods": "resolved"}}
    )
    bare = {"s:1": {"goods": "old-one"}, "s:2": {"goods": "UNKNOWN"}}
    requests, plans, complete = completion.prepared_request_state(
        tmp_path, current, {"config": initial}, bare
    )
    assert requests == complete == {"s:1": {"goods": "new-one"}, "s:2": {"goods": "resolved"}}
    assert plans["s:2"] == {"stage": "current"}
    assert bare["s:2"]["goods"] == "UNKNOWN"


def test_initial_preparation_retains_unresolved_slots_for_authored_completion(tmp_path):
    prepared = state(tmp_path, "initial", {"s:1": {"goods": "ready"}})
    requests, plans, bare = completion.prepared_request_state(
        tmp_path, {"independent": {}}, {"config": prepared}, {"s:2": {"goods": "unresolved"}}
    )
    assert set(requests) == set(plans) == {"s:1"}
    assert bare == {"s:1": {"goods": "ready"}, "s:2": {"goods": "unresolved"}}


def test_dg_redraw_replaces_tuple_and_physical_state_without_stale_chemistry():
    packet = {"goods": ["old"], "dangerousGoods": {"recordId": "old"},
              "packingContext": "Registry physical state: L", "regime": "dg"}
    replacement = {"goods": [{"hs6": "282520", "description": "Lithium hydroxide"}],
                   "goodsForm": "Lithium hydroxide", "regime": "dg",
                   "dangerousGoods": {"recordId": "new", "unNumber": "2680"},
                   "equipmentSlots": [{"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                                       "typeCategory": "GENERAL_PURPOSE"}]}
    result = completion.redraw_choice_context(packet, replacement, {"new": "S"})
    assert result["goods"] == replacement["goods"]
    assert result["dangerousGoods"] == replacement["dangerousGoods"]
    assert result["packingContext"] == "Registry physical state: S"
    assert packet["dangerousGoods"]["recordId"] == "old"
    assert result["requiredCategory"] is None
