import asyncio
import copy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis import paddle_templates as native
from document_ocr.synthesis.paddle_wording import propose, wording_model


@pytest.mark.parametrize(
    "overrides",
    [
        [{"key": "unused", "byteStart": 4}] * 2,
        [{"key": "unused"}] * 2,
        [{"key": "unused"}, {"key": "unused", "byteStart": 4}],
        [{"key": "unused", "byteStart": 4}, {"key": "unused"}],
        [{"key": "unused", "byteStart": 4}, {"key": "unused", "byteStart": 4, "byteEnd": 9}],
        [{"key": "unused", "byteStart": 4, "byteEnd": 9}] * 2,
    ],
)
def test_override_selectors_reject_ambiguity_before_shipments(overrides):
    with pytest.raises(ValueError, match="ambiguous native override selector"):
        native.validate_override_selectors({"overrides": overrides})


def test_override_selectors_allow_distinct_occurrences_and_keys():
    native.validate_override_selectors({})
    native.validate_override_selectors(
        {"overrides": [{"key": "a", "byteStart": 4}, {"key": "a", "byteStart": 8}, {"key": "b"}]}
    )
    native.validate_override_selectors(
        {
            "overrides": [
                {"key": "a", "byteStart": 4, "byteEnd": 9},
                {"key": "a", "byteStart": 4, "byteEnd": 19},
            ]
        }
    )


@pytest.mark.parametrize(
    "bounds", [{"byteEnd": 9}, {"byteStart": 9, "byteEnd": 9}, {"byteStart": -1, "byteEnd": 9}]
)
def test_bounded_override_requires_exact_valid_interval(bounds):
    with pytest.raises(ValueError, match="valid byte interval"):
        native.validate_override_selectors({"overrides": [{"key": "a", **bounds}]})


def test_evidence_pin_rejects_changes_and_path_escape(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text('{"a":1}')
    pin = {"path": path.name, "sha256": sha256_file(path)}
    assert native.read_pin(tmp_path, pin) == {"a": 1}
    for value in (str(path), "../evidence.json"):
        with pytest.raises(ValueError, match="project-relative"):
            native.read_pin(tmp_path, {**pin, "path": value})
    path.write_text('{"a":2}')
    with pytest.raises(ValueError, match="changed"):
        native.read_pin(tmp_path, pin)


@pytest.fixture
def catalog(tmp_path):
    authority = b"OLD BILL"
    candidate = {
        "sourceDocumentId": "source",
        "documentId": "new-shipment",
        "joinedRawText": "NEW BILL",
        "target": {"documentPatch": {"negotiability": "non_negotiable"}},
        "proof": {
            "sourceSha256": hashlib.sha256(authority).hexdigest(),
            "edits": [{"byteStart": 0, "byteEnd": 8, "before": "OLD BILL"}],
        },
    }
    path = tmp_path / "candidate.json"
    atomic_publish_json(path, candidate)
    cat = object.__new__(native.PaddleTemplateCatalog)
    cat.root = tmp_path
    cat.package_modes = None
    cat.target_casing = None
    cat.entries = {
        "source": {
            "samplingSource": {"sha256": candidate["proof"]["sourceSha256"]},
            "privacy": {"kind": "reviewed_native_owners", "decision": {}},
        }
    }
    cat._loaded = {
        "source": (
            {"overlay": {}},
            {"documentId": "source"},
            {"descendants": [{"id": "certified-control"}]},
            authority,
        )
    }
    cat.engine = cat.equipment = cat.packages = None
    cat.validator = Draft202012Validator({"type": "object"})
    return cat, path, candidate


def test_fresh_receipt_selection_is_local_and_does_not_need_old_dataset(catalog):
    cat, path, _ = catalog
    _, _, transfer, candidate = cat.shipment(path)
    assert transfer["descendants"][0]["id"] == candidate["documentId"]
    assert transfer["descendants"][0]["receiptSha256"] == sha256_file(path)
    assert cat._loaded["source"][2]["descendants"] == [{"id": "certified-control"}]


def test_single_row_admission_rejects_multirow_before_wording(catalog):
    cat, path, candidate = catalog
    cat.package_modes = {"singleRowOnly": ["source"]}
    rows = [{"packageQuantity": 1, "typeCategory": "PACKAGE_PACKAGE"}]
    candidate["target"]["documentPatch"]["goodsItemDetails"] = [{"numberAndTypeOfPackages": rows}]
    path.write_text(json.dumps(candidate))
    cat.shipment(path)
    rows.append({"packageQuantity": 2, "typeCategory": "PACKAGE_BAG"})
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="single-row"):
        cat.wording_specification(path)


@pytest.mark.parametrize("invalid", ["source", "hash", "text", "range", "negative"])
def test_fresh_receipt_fail_closed_on_owner_authority(catalog, invalid):
    cat, path, candidate = catalog
    if invalid == "source":
        candidate["sourceDocumentId"] = "other"
    elif invalid == "hash":
        candidate["proof"]["sourceSha256"] = "bad"
    elif invalid == "text":
        candidate["proof"]["edits"][0]["before"] = "BAD BILL"
    else:
        candidate["proof"]["edits"][0]["byteEnd"] = 99 if invalid == "range" else -1
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError):
        cat.shipment(path)


def test_fresh_render_runs_all_validation_and_preserves_retained_target(catalog, monkeypatch):
    cat, path, _ = catalog
    called, captured = [], []

    def render(*args, **kwargs):
        captured.append(copy.deepcopy(args[3]))
        return {"target": args[3]["target"]}

    monkeypatch.setattr(native, "render_one", render)
    monkeypatch.setattr(native, "pages_from_record", lambda _: ())
    for name in (
        "check_views",
        "check_conservation",
        "check_synthetic_relations",
        "check_private_lexemes",
        "check_new_private_ownership",
    ):
        monkeypatch.setattr(native, name, lambda *a, _name=name: called.append(_name))
    result = cat.render(path)
    assert result["target"] == {"negotiability": "non_negotiable"}
    assert len(called) == 5
    current = {"documentId": "new-shipment", "target": {"negotiability": "negotiable"}}
    cat.render(path, current=current)
    assert captured[-1] == current
    assert current == {"documentId": "new-shipment", "target": {"negotiability": "negotiable"}}
    with pytest.raises(ValueError, match="different shipment"):
        cat.render(path, current={**current, "documentId": "another"})
    cat._loaded["source"][2]["descendants"] = [{"id": "retained", "path": str(path)}]
    retained = {**current, "documentId": "retained"}
    cat.render(path, current=retained)
    assert captured[-1] == retained
    cat.entries["source"]["privacy"]["kind"] = "unknown"
    with pytest.raises(ValueError, match="privacy authority"):
        cat.render(path)


def test_wording_schema_and_proposal_do_not_implicitly_certify():
    model = wording_model({"name": "Fictional signature"}, {"name": "^[A-Z]{3}$"})
    assert model(name="ABC").name == "ABC"
    for value in ({"name": "longer"}, {"name": "ABC", "extra": "bad"}):
        with pytest.raises(ValidationError):
            model(**value)
    with pytest.raises(ValueError, match="unrequested"):
        wording_model({"name": "signature"}, {"other": ".*"})

    class Runner:
        async def call(self, stage, identity, response, system, prompt, **kwargs):
            assert identity == "new" and "shipment facts" in system.lower()
            assert json.loads(prompt)["goods"] == ["chilled pears"]
            return response(name="ABC")

    spec = {
        "documentId": "new",
        "sourceFingerprint": "source-sha",
        "candidateSha256": "sha",
        "requested": {"name": "signature"},
        "patterns": {"name": "^[A-Z]{3}$"},
        "context": {"goods": ["chilled pears"]},
    }
    result = asyncio.run(propose(Runner(), spec))
    assert result["status"] == "PROPOSED_REQUIRES_REVIEW"
    assert result["candidateSha256"] == "sha" and result["values"] == {"name": "ABC"}
