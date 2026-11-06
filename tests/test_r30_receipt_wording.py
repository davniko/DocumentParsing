import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "receipt_wording", Path(__file__).parents[1] / "scripts/migrations/r30_receipt_wording.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    request = {"documentId": "doc", "sourceFingerprint": "source", "candidateSha256": "new",
               "requested": {"agency": "Fictional agency"}, "context": {"destination": "A"}}
    wording = {"documentId": "doc", "sourceFingerprint": "source", "candidateSha256": "old",
               "status": "REVIEWED", "values": {"agency": "Harbour Agency"},
               "prompt": json.dumps({"requested": request["requested"], **request["context"]})}
    return request, wording


def test_receipt_change_keeps_reviewed_values_without_mutating_old_receipt():
    request, wording = fixture()
    before = copy.deepcopy(wording)
    result = module.rebind_wording(wording, request, {"path": "old", "sha256": "hash"})
    assert result["candidateSha256"] == "new"
    assert result["values"] == wording["values"]
    assert wording == before


@pytest.mark.parametrize("change", ["status", "identity", "source", "context", "requested", "values"])
def test_reuse_rejects_changed_authority(change):
    request, wording = fixture()
    if change == "status": wording["status"] = "PROPOSED"
    if change == "identity": request["documentId"] = "other"
    if change == "source": request["sourceFingerprint"] = "other"
    if change == "context": request["context"]["destination"] = "B"
    if change == "requested": request["requested"]["agency"] = "Different owner"
    if change == "values": wording["values"]["extra"] = "Not requested"
    with pytest.raises(ValueError):
        module.rebind_wording(wording, request, {})
