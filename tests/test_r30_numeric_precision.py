import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from r30_missing_numeric import complete_numeric, numeric_context, reconcile_numeric_precision
from test_r30_missing_numeric import candidate


def fixture():
    c = candidate()
    c["physical"]["measures"]["volume"] = {
        "unit": "cubic_metre",
        "total": "10.000",
        "quantum": "0.001",
        "beforeRounding": "10.0001760339342523860021209",
    }
    q = numeric_context(c, ["volume"])
    v = {
        "unknown": False,
        "basis": "Two equal synthetic loads.",
        "row0_volume": 5,
        "row1_volume": 5,
    }
    return c, q, v


def test_explicit_precision_remainder_preserves_all_retained_facts():
    c, q, v = fixture()
    before = copy.deepcopy((c, q, v))
    with pytest.raises(ValueError, match="conserve"):
        complete_numeric(c, q, v)
    values, adjustments = reconcile_numeric_precision(c, q, v)
    assert (c, q, v) == before
    assert values["row0_volume"] == 5
    assert values["row1_volume"] == "5.0001760339342523860021209"
    assert len(adjustments) == 1 and adjustments[0]["field"] == "row1_volume"
    completed = complete_numeric(c, q, values)
    assert completed["target"] == c["target"] and completed["physical"] == c["physical"]


@pytest.mark.parametrize("value", [4.99, 4.9999, True, "NaN", "Infinity"])
def test_nonrounding_errors_are_not_repaired(value):
    c, q, v = fixture()
    with pytest.raises(ValueError):
        reconcile_numeric_precision(c, q, {**v, "row1_volume": value})


def test_exact_proposal_needs_no_adjustment():
    c, q, v = fixture()
    v["row1_volume"] = "5.0001760339342523860021209"
    assert reconcile_numeric_precision(c, q, v) == (v, [])


def test_remainder_cannot_exceed_capacity_or_resolve_unknown():
    c, q, v = fixture()
    c["scenario"]["provenance"]["physicalRows"][1]["capacity"]["volumeM3"] = "5"
    q = numeric_context(c, ["volume"])
    with pytest.raises(ValueError, match="valid bounds"):
        reconcile_numeric_precision(c, q, v)
    with pytest.raises(ValueError, match="complete known"):
        reconcile_numeric_precision(c, q, {**v, "unknown": True})


def test_known_row_is_never_adjusted():
    c, _, _ = fixture()
    c["scenario"]["provenance"]["physicalRows"][1]["measures"]["volume"] = {
        "value": "5",
        "total": c["physical"]["measures"]["volume"]["beforeRounding"],
        "unit": "cubic_metre",
    }
    q = numeric_context(c, ["volume"])
    values, changes = reconcile_numeric_precision(
        c,
        q,
        {
            "unknown": False,
            "basis": "Exact remainder.",
            "row0_volume": 5,
        },
    )
    assert changes[0]["field"] == "row0_volume"
    result = complete_numeric(c, q, values)
    assert (
        result["scenario"]["provenance"]["physicalRows"][1]
        == c["scenario"]["provenance"]["physicalRows"][1]
    )
