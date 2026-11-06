import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from audit_r30_synthetic_batch import require_review_ready


def test_complete_batch_and_explicit_scoped_diagnostics_are_supported():
    rows = [{"id": str(i)} for i in range(100)]
    require_review_ready({"selected": 100, "results": rows, "failures": []}, ())
    require_review_ready({"selected": 100, "results": rows[:1], "failures": ["held"]}, ["0"])


def test_no_automatic_paid_review_ahead_of_mechanical_holds():
    staged = {"selected": 100, "results": [{"id": "0"}], "failures": ["held"]}
    with pytest.raises(ValueError, match="mechanical readiness"):
        require_review_ready(staged, ())
    for ids in (["missing"], ["0", "0"]):
        with pytest.raises(ValueError, match="distinct rendered"):
            require_review_ready(staged, ids)
