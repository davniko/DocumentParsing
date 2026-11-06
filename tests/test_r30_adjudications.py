import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from apply_r30_adjudications import adjudicate


def test_explicit_decisions_preserve_other_fields_and_source():
    old = {"goods": [{"description": "OLD", "mass": 3}], "country": "A"}
    result = adjudicate(
        old,
        {"goods": [{"description": "NEW"}]},
        {
            "acceptCandidatePaths": ["/goods/0/description"],
            "set": {"/goods/0/quantity": 4},
            "remove": ["/country"],
            "notes": "reviewed",
        },
    )
    assert result == {"goods": [{"description": "NEW", "mass": 3, "quantity": 4}]}
    assert old["goods"][0]["description"] == "OLD"


def test_missing_parent_or_removal_is_not_silently_ignored():
    for set_values, removals in [({"/missing/field": 3}, []), ({}, ["/missing"])]:
        with pytest.raises(KeyError):
            adjudicate(
                {},
                {},
                {
                    "acceptCandidatePaths": [],
                    "set": set_values,
                    "remove": removals,
                    "notes": "reviewed",
                },
            )


def test_no_implicit_acceptance_of_proposal():
    assert adjudicate(
        {"value": 1},
        {"value": 99},
        {"acceptCandidatePaths": [], "set": {}, "remove": [], "notes": "rejected"},
    ) == {"value": 1}
