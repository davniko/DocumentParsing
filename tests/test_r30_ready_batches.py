import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from certify_r30_ready_batches import require_ready


def fixtures(tmp_path):
    rows = [{"id": str(i), "source": "source", "rendered": {"sha256": str(i)}} for i in range(100)]
    staged = {"selected": 100, "failures": [], "results": rows}
    inventory = {}
    for r in rows:
        path = tmp_path / (r["id"] + ".json")
        path.write_text(json.dumps({**r, "status": "PROPOSED_REQUIRES_MAIN_AGENT_ADJUDICATION",
                                    "review": {"findings": []}, "invalidEvidence": []}))
        pin = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        inventory[r["id"]] = {**r, "exact": [{"pin": pin, "findings": 0, "invalidEvidence": 0}]}
    return staged, inventory


def test_exact_reviewed_cohort_is_eligible_without_paid_calls(tmp_path):
    staged, inventory = fixtures(tmp_path)
    require_ready(staged, inventory)


@pytest.mark.parametrize("case", ["held", "short", "stale", "missing", "finding", "invalid"])
def test_no_weakened_review_or_membership_gate(case, tmp_path):
    staged, inventory = fixtures(tmp_path)
    if case == "held": staged["failures"] = ["held"]
    elif case == "short": staged["results"].pop()
    elif case == "stale": inventory["0"]["rendered"] = {"sha256": "old"}
    elif case == "missing": inventory["0"]["exact"] = []
    elif case == "finding": inventory["0"]["exact"][0]["findings"] = 1
    elif case == "invalid": inventory["0"]["exact"][0]["invalidEvidence"] = 1
    with pytest.raises(ValueError): require_ready(staged, inventory)
