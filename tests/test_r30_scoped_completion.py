import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from complete_r30_retained_inner_packing import complete_ledger
from stage_r30_catalog_corrections import apply_edit


def fixture():
    candidate = {
        "sourceDocumentId": "source",
        "physical": {
            "measures": {
                "grossWeight": {"total": "2880.000", "unit": "kilogram"},
                "volume": {"total": "13.500", "unit": "cubic_metre"},
            }
        },
        "scenario": {"provenance": {}},
    }
    public = [{"packageQuantity": 18, "typeCategory": "PACKAGE_SET"}]
    decision = {
        "expectedPackage": public[0],
        "cartonsPerUnit": 4,
        "piecesPerCarton": 8,
        "relation": "CONTAINS",
        "basis": "Reviewed pump components",
    }
    native = {"owner": "inner", "nestedPacking": {"nativeFingerprint": "pinned"}}
    return candidate, public, decision, native


def test_count_completion_preserves_absent_net_and_inputs():
    args = fixture()
    before = copy.deepcopy(args)
    ledger = complete_ledger(*args, {"path": "review", "sha256": "pin"})
    assert args == before
    assert ledger["relatedQuantity"] == 72
    assert ledger["facts"]["externalPackingPresentation"]["contents"]["quantity"] == 576
    assert "netKg" not in ledger
    assert ledger["grossKg"] == "2880.000"


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_count_completion_rejects_nonpositive_noninteger_counts(count):
    c, p, d, n = fixture()
    d["cartonsPerUnit"] = count
    with pytest.raises(ValueError, match="positive integers"):
        complete_ledger(c, p, d, n, {})


def test_count_completion_does_not_overwrite_existing_packing():
    c, p, d, n = fixture()
    c["scenario"]["provenance"]["freshPrivatePacking"] = {"relation": "CONTAINS"}
    with pytest.raises(ValueError, match="overwrite"):
        complete_ledger(c, p, d, n, {})


def test_template_edits_fail_closed_and_keep_review_values_independent():
    t = {"overlay": {"extras": [], "owner": "old"}}
    append = {
        "action": "append",
        "path": "/overlay/extras",
        "beforeLength": 0,
        "after": [{"owner": "new"}],
    }
    apply_edit(t, append)
    t["overlay"]["extras"][0]["owner"] = "changed"
    assert append["after"][0]["owner"] == "new"
    with pytest.raises(ValueError, match="position changed"):
        apply_edit(t, append)
    with pytest.raises(ValueError, match="member changed"):
        apply_edit(
            t, {"action": "replace", "path": "/overlay/owner", "before": "wrong", "after": "new"}
        )
    with pytest.raises(ValueError, match="absent"):
        apply_edit(t, {"action": "add", "path": "/overlay/owner", "after": "new"})


def test_disjoint_completion_uses_per_unit_facts_and_conserves_both_groups():
    c, _, d, n = fixture()
    public = [
        {"packageQuantity": 2, "typeCategory": "PACKAGE_OCTABIN"},
        {"packageQuantity": 3, "typeCategory": "PACKAGE_PALLET"},
    ]
    c["physical"]["measures"] = {"grossWeight": {"total": "360", "unit": "kilogram"}, "volume": {"total": "5", "unit": "cubic_metre"}}
    c["scenario"]["provenance"] = {
        "freshPrivatePacking": {
            "relation": "NONE",
            "facts": {},
            "publicQuantity": 5,
            "netKg": "320",
            "grossKg": "360",
        },
        "declaredPackages": {
            "rows": public,
            "groups": [
                {"category": "PACKAGE_OCTABIN", "netKg": "100", "grossKg": "105", "volumeM3": "1"},
                {"category": "PACKAGE_PALLET", "netKg": "40", "grossKg": "50", "volumeM3": "1"},
            ],
            "totals": {"netKg": "320", "grossKg": "360", "volumeM3": "5"},
        },
    }
    decision = {"groups": [{**d, "expectedPackage": row} for row in public]}
    before = copy.deepcopy(c)
    result = complete_ledger(c, public, decision, n, {})
    assert c == before
    first, second = result["facts"]["declaredGroupLedgers"]
    assert (first["relatedQuantity"], second["relatedQuantity"]) == (8, 12)
    assert (first["netKg"], second["netKg"]) == ("200", "120")
    assert result["grossKg"] == "360"
    c["scenario"]["provenance"]["declaredPackages"]["groups"][0]["netKg"] = "99"
    with pytest.raises(ValueError, match="conserve"):
        complete_ledger(c, public, decision, n, {})
    c = before
    c["scenario"]["provenance"]["freshPrivatePacking"]["facts"] = {"existing": True}
    with pytest.raises(ValueError, match="existing hierarchy"):
        complete_ledger(c, public, decision, n, {})


def test_inner_completion_converts_retained_units():
    c, p, d, n = fixture()
    c["physical"]["measures"]["grossWeight"] = {"total": "2.880", "unit": "metric_tonne"}
    result = complete_ledger(c, p, d, n, {})
    assert result["grossKg"] == "2880.000"
    assert result["reviewedRetainedPacking"]["averageGrossKgPerCarton"] == "40.000"


def test_inner_completion_rejects_missing_or_wrong_units():
    c, p, d, n = fixture()
    del c["physical"]["measures"]["grossWeight"]["unit"]
    with pytest.raises(KeyError): complete_ledger(c, p, d, n, {})
    c["physical"]["measures"]["grossWeight"]["unit"] = "cubic_metre"
    with pytest.raises(ValueError): complete_ledger(c, p, d, n, {})
