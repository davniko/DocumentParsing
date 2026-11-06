import importlib.util
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
spec = importlib.util.spec_from_file_location(
    "amend_r30_published_synthetic",
    Path(__file__).resolve().parents[1] / "scripts/migrations/amend_r30_published_synthetic.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unaffected_wording_is_preserved():
    old = {"path": "old", "sha256": "old"}
    assert module.amendment_wording({"wording": old}, "one", {}) == old


@pytest.mark.parametrize(
    "change",
    [
        {"status": "PROPOSED_REQUIRES_REVIEW"},
        {"documentId": "different"},
        {"candidateSha256": "different"},
    ],
)
def test_new_wording_requires_reviewed_same_shipment(monkeypatch, change):
    value = {"status": "REVIEWED", "documentId": "one", "candidateSha256": "candidate", **change}
    monkeypatch.setattr(module, "verify_pin", lambda p: p)
    monkeypatch.setattr(module, "read", lambda p: value)
    with pytest.raises(ValueError):
        module.amendment_wording(
            {"wording": None, "candidate": {"sha256": "candidate"}}, "one", {"one": {"path": "new"}}
        )


def test_existing_wording_cannot_be_replaced():
    with pytest.raises(ValueError, match="overwrite"):
        module.amendment_wording({"wording": {"path": "old"}}, "one", {"one": {"path": "new"}})


def test_reviewed_new_wording_accepted(monkeypatch):
    new = {"path": "new"}
    monkeypatch.setattr(module, "verify_pin", lambda p: p)
    monkeypatch.setattr(
        module,
        "read",
        lambda p: {"status": "REVIEWED", "documentId": "one", "candidateSha256": "candidate"},
    )
    assert (
        module.amendment_wording(
            {"wording": None, "candidate": {"sha256": "candidate"}}, "one", {"one": new}
        )
        == new
    )


def extension_fixture(monkeypatch):
    old_pin = {"path": "old", "sha256": "old"}
    new_pin = {"path": "new", "sha256": "new"}
    common = {"status": "REVIEWED", "documentId": "one", "candidateSha256": "candidate",
              "sourceFingerprint": "source"}
    old = {**common, "values": {"signature": "Halen"},
           "prompt": json.dumps({"requested": {"signature": "Five letters"}, "destination": "UAE"})}
    new = {**common, "values": {"signature": "Halen", "agent": "Fictional Agency LLC"},
           "prompt": json.dumps({"requested": {"signature": "Five letters", "agent": "Company name"}, "destination": "UAE"})}
    receipts = {"old": old, "new": new}
    monkeypatch.setattr(module, "verify_pin", lambda p: p)
    monkeypatch.setattr(module, "read", lambda p: receipts[p["path"]])
    evidence = {"wording": old_pin, "candidate": {"sha256": "candidate"}}
    additions = {"one": {"receipt": new_pin, "wordingExtension": old_pin}}
    return evidence, additions, receipts


def test_additive_owners_preserve_existing_values(monkeypatch):
    evidence, additions, _ = extension_fixture(monkeypatch)
    assert module.amendment_wording(evidence, "one", additions) == additions["one"]["receipt"]


def test_additional_context_preserves_every_prior_context_value(monkeypatch):
    evidence, additions, receipts = extension_fixture(monkeypatch)
    prompt = json.loads(receipts["new"]["prompt"])
    prompt["equipment"] = [{"equipmentIdentifier": "TEST0000001"}]
    receipts["new"]["prompt"] = json.dumps(prompt)
    assert module.amendment_wording(evidence, "one", additions) == additions["one"]["receipt"]


@pytest.mark.parametrize("change", ["value", "request", "context", "remove", "no_addition", "source", "prior"])
def test_owner_extension_rejects_any_changed_prior_contract(monkeypatch, change):
    evidence, additions, receipts = extension_fixture(monkeypatch)
    new = receipts["new"]
    prompt = json.loads(new["prompt"])
    if change == "value":
        new["values"]["signature"] = "Different"
    elif change == "request":
        prompt["requested"]["signature"] = "Different format"
    elif change == "context":
        prompt["destination"] = "Egypt"
    elif change == "remove":
        del new["values"]["signature"]
    elif change == "no_addition":
        new["values"] = copy.deepcopy(receipts["old"]["values"])
        prompt = json.loads(receipts["old"]["prompt"])
    elif change == "source":
        new["sourceFingerprint"] = "another source"
    else:
        additions["one"]["wordingExtension"] = {"path": "wrong"}
    new["prompt"] = json.dumps(prompt)
    with pytest.raises(ValueError):
        module.amendment_wording(evidence, "one", additions)
