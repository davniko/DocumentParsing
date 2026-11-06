import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
import publish_r30_sources as publication
from repair_r30_publication_evidence import PAYLOAD, require_same_payload

from document_ocr.atomic import AtomicConflictError, atomic_publish_bytes, atomic_write_json
from document_ocr.spatial_inputs.paddle import PaddlePage, PaddleRegion, serialize


def test_amendment_chain_uses_pinned_predecessors_not_filenames(tmp_path, monkeypatch):
    monkeypatch.setattr(publication, "ROOT", tmp_path)
    monkeypatch.setattr(publication, "DATA", tmp_path / "data")
    root = publication.DATA / "batches/one/certificate.json"
    other = publication.DATA / "batches/two/certificate.json"
    atomic_write_json(root, {})
    atomic_write_json(other, {})
    evidence = tmp_path / "evidence.json"
    atomic_write_json(evidence, {"reviewed": True})

    def amendment(name, previous, batch="one"):
        path = publication.DATA / "amendments" / name / "certificate.json"
        atomic_write_json(
            path,
            {
                "batch": batch,
                "previous": previous,
                "review": publication.pin(evidence),
                "records": {},
            },
        )
        return path

    first = amendment("z-first", publication.pin(root))
    second = amendment("a-second", publication.pin(first))
    assert publication.active_source_certificates() == {"one": second, "two": other}
    branch = amendment("branch", publication.pin(first))
    with pytest.raises(ValueError, match="without branching"):
        publication.active_source_certificates()
    branch.unlink()
    orphan = amendment("orphan", {"path": "missing.json", "sha256": "0" * 64})
    with pytest.raises(ValueError, match="unreachable or changed predecessor"):
        publication.active_source_certificates()
    orphan.unlink()
    mismatch = amendment("wrong-batch", publication.pin(second), "two")
    with pytest.raises(ValueError, match="another source batch"):
        publication.active_source_certificates()
    mismatch.unlink()
    atomic_write_json(first, {**publication.read(first), "changed": True})
    with pytest.raises(ValueError, match="unreachable or changed predecessor"):
        publication.active_source_certificates()


def test_render_evidence_repair_cannot_change_any_payload_field():
    record = {key: [key] for key in PAYLOAD}
    require_same_payload(record, copy.deepcopy(record))
    for key in PAYLOAD:
        changed = copy.deepcopy(record)
        changed[key] = []
        with pytest.raises(ValueError, match="cannot change published payload"):
            require_same_payload(record, changed)


def test_order_amendment_is_only_a_lossless_permutation(tmp_path, monkeypatch):
    for name, suffix in (("ROOT", ""), ("DATA", "data")):
        monkeypatch.setattr(publication, name, tmp_path / suffix)

    def record(texts):
        pages = (
            PaddlePage(
                0,
                100,
                100,
                tuple(
                    PaddleRegion(i, text, (1, 10 + 20 * (text == "A"), 30, 20 + 20 * (text == "A")))
                    for i, text in enumerate(texts)
                ),
            ),
        )
        return {
            "documentId": "doc_a",
            "target": {"negotiability": "non_negotiable"},
            "pages": [asdict(p) for p in pages],
            "joinedRawText": serialize(pages, geometry="none"),
            "positionedText": serialize(pages),
            "bboxText": serialize(pages, geometry="bbox"),
        }

    before, after = record(["A", "B"]), record(["B", "A"])
    publication.order_only_record(before, after)
    for field in ("text", "box"):
        bad = copy.deepcopy(after)
        bad["pages"][0]["regions"][0][field] = "C" if field == "text" else (1, 11, 30, 20)
        with pytest.raises(ValueError, match="text or geometry"):
            publication.order_only_record(before, bad)
    bad = copy.deepcopy(after)
    bad["target"]["negotiability"] = "negotiable"
    with pytest.raises(ValueError, match="targets"):
        publication.order_only_record(before, bad)
    bad = copy.deepcopy(after)
    bad["positionedText"] = "wrong"
    with pytest.raises(ValueError, match="inconsistent input view"):
        publication.order_only_record(before, bad)
    original = publication.DATA / "batches/one/real-train.jsonl"
    atomic_publish_bytes(original, publication.jsonl([before]))
    cert = original.parent / "certificate.json"
    atomic_write_json(cert, {"records": {"real-train": publication.pin(original)}})
    candidate = tmp_path / "candidate.jsonl"
    atomic_publish_bytes(candidate, publication.jsonl([after]))
    evidence = tmp_path / "native-proof.json"
    atomic_write_json(evidence, {"reviewed": True})
    review = tmp_path / "order.json"
    atomic_write_json(
        review,
        {
            "previous": publication.pin(cert),
            "rationale": "Native replay",
            "sourceEvidence": [publication.pin(evidence)],
            "records": {"real-train": publication.pin(candidate)},
            "changedDocuments": ["doc_a"],
        },
    )
    publication.amend_input_order("one", review)
    assert list(publication.rows(original)) == json.loads(json.dumps([before]))
    active = publication.read(publication.active_source_certificates()["one"])
    assert list(
        publication.rows(publication.verify_pin(active["records"]["real-train"]))
    ) == json.loads(json.dumps([after]))
    with pytest.raises(ValueError, match="current evidence"):
        publication.amend_input_order("one", review)


def test_batch_readback_must_be_exact_not_a_subset(monkeypatch):
    monkeypatch.setattr(
        publication,
        "batch_records",
        lambda _: [{"documentId": "doc_12345678"}, {"documentId": "doc_87654321"}],
    )
    for reviewed in (["12345678"], ["12345678", "12345678"], ["12345678", "missing"]):
        with pytest.raises(ValueError, match="exactly this batch"):
            publication.publish_batch("batch", reviewed)


def test_publication_is_idempotent_but_rejects_changed_bytes(tmp_path):
    path = tmp_path / "records.jsonl"
    payload = publication.jsonl([{"documentId": "b"}, {"documentId": "a"}])
    assert atomic_publish_bytes(path, payload)
    assert not atomic_publish_bytes(path, payload)
    assert [json.loads(s)["documentId"] for s in path.read_text().splitlines()] == ["a", "b"]
    with pytest.raises(AtomicConflictError):
        atomic_publish_bytes(path, payload + b"\n")


def test_wider_source_reference_cannot_enter_training_membership(tmp_path, monkeypatch):
    for name, suffix in (
        ("ROOT", ""),
        ("DATA", "data"),
        ("ART", "artifacts"),
        ("OUT", "audit"),
        ("TEMPLATES", "templates"),
    ):
        monkeypatch.setattr(publication, name, tmp_path / suffix)
    source = {"documentId": "doc_12345678", "split": "template-only"}
    monkeypatch.setattr(publication, "batch_records", lambda _: [source])
    decisions = publication.OUT / "batches/wider/main-agent-decisions.json"
    atomic_write_json(decisions, {"12345678": {"notes": ["Reviewed literal native reference"]}})
    staged = {k: [] for k in ("pages",)} | {
        k: "" for k in ("joinedRawText", "positionedText", "bboxText")
    }
    staged.update(
        documentId=source["documentId"], target={}, reviewSha256=publication.sha256_file(decisions)
    )
    atomic_write_json(publication.ART / "staged-real/doc_12345678.json", staged)
    monkeypatch.setattr(publication, "audit_staged_record", lambda *args: {"audited": True})
    publication.publish_batch("wider", ["12345678"])
    assert not (publication.DATA / "batches").exists()
    cert = publication.read(publication.TEMPLATES / "source-batches/wider/certificate.json")
    assert cert["status"] == "REVIEWED_TEMPLATE_REFERENCE_NOT_TRAINING_DATA"
    assert set(cert["records"]) == {"template-only"}


def test_manifest_keeps_partial_dataset_untrainable_and_rejects_duplicate_members(
    tmp_path, monkeypatch
):
    for name, suffix in (
        ("ROOT", ""),
        ("OUT", "audit"),
        ("DATA", "data"),
        ("BASE", "base"),
        ("TEMPLATES", "templates"),
    ):
        monkeypatch.setattr(publication, name, tmp_path / suffix)
    atomic_write_json(
        publication.OUT / "plan.json",
        {
            "sources": [
                {"id": "a", "split": "real-train", "activeTemplate": True},
                {"id": "b", "split": "real-train", "activeTemplate": False},
                {"id": "v", "split": "validation", "activeTemplate": False},
            ]
        },
    )
    atomic_write_json(publication.BASE / "schema.json", {})
    data_path = publication.DATA / "batches/one/real-train.jsonl"
    atomic_publish_bytes(data_path, publication.jsonl([{"documentId": "a"}]))
    cert = {"records": {"real-train": publication.pin(data_path)}}
    atomic_write_json(data_path.parent / "certificate.json", cert)
    result = publication.update_manifest()
    assert result["status"] == "PARTIAL_CERTIFIED_BATCHES_NOT_TRAINABLE"
    assert result["publishedCounts"] == {"real-train": 1, "validation": 0}
    assert result["requiredCounts"] == {"real-train": 2, "validation": 1}
    atomic_write_json(publication.DATA / "batches/two/certificate.json", cert)
    with pytest.raises(ValueError, match="duplicate published"):
        publication.update_manifest()


def test_changed_evidence_fails_before_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(publication, "ROOT", tmp_path)
    path = tmp_path / "evidence.json"
    atomic_write_json(path, {"a": 1})
    pin = publication.pin(path)
    assert publication.verify_pin(pin) == path
    atomic_write_json(path, {"a": 2})
    with pytest.raises(ValueError, match="evidence changed"):
        publication.verify_pin(pin)


def test_template_amendment_is_pinned_validated_and_non_branching(tmp_path, monkeypatch):
    for name, suffix in (("ROOT", ""), ("TEMPLATES", "templates"), ("ART", "artifacts")):
        monkeypatch.setattr(publication, name, tmp_path / suffix)

    def evidence(name, value):
        path = tmp_path / (name + ".json")
        atomic_write_json(path, value)
        return publication.pin(path)

    source = evidence("source", {})
    shipments = [evidence(f"shipment-{i}", {}) for i in range(3)]
    transfer = evidence(
        "transfer",
        {
            "descendants": [
                {"id": f"syn_{i}", "path": item["path"]} for i, item in enumerate(shipments)
            ]
        },
    )
    old_case = {
        "sourceDocumentId": "doc_12345678",
        "source": source,
        "transfer": transfer,
        "overlay": {"extras": []},
        "testDescendants": [f"syn_{i}" for i in range(3)],
    }
    old = {"template": evidence("old", old_case), "controls": []}
    atomic_write_json(
        publication.TEMPLATES / "batches/one.json",
        {
            "sourceCertificate": evidence("source-cert", {}),
            "templates": [old],
        },
    )
    new_case = copy.deepcopy(old_case)
    new_case["overlay"]["extras"] = [{"owner": "new-mass"}]
    new_pin = evidence("new", new_case)
    controls = [
        evidence(
            f"control-{i}",
            {
                "documentId": f"syn_{i}",
                "target": {},
                "migrationPins": {
                    "template": new_pin["sha256"],
                    "transfer": transfer["sha256"],
                    "shipment": shipments[i]["sha256"],
                },
            },
        )
        for i in range(3)
    ]
    checks = []
    for name in (
        "check_training_target",
        "check_views",
        "check_conservation",
        "check_synthetic_relations",
        "check_private_lexemes",
        "check_new_private_ownership",
    ):
        monkeypatch.setattr(publication, name, lambda *args, name=name: checks.append(name))
    monkeypatch.setattr(publication, "load_pages", lambda _: [])
    replacement = {"template": new_pin, "controls": controls}
    review = {
        "rationale": "Reviewed missing mass repetition",
        "mainAgentReadback": ["doc_12345678"],
        "templates": [
            {
                "sourceDocumentId": "doc_12345678",
                "previous": old,
                "replacement": replacement,
                "decisions": evidence("decisions", {"12345678": {}}),
            }
        ],
    }
    review_path = tmp_path / "review.json"
    atomic_write_json(review_path, review)
    publication.amend_templates(review_path)
    assert len(checks) == 18
    assert publication.active_template_entries()["doc_12345678"] == replacement
    assert publication.read(publication.verify_pin(old["template"])) == old_case
    with pytest.raises(ValueError, match="current evidence"):
        publication.amend_templates(review_path)
    # An independently published stale branch cannot silently replace the winner.
    cert = publication.TEMPLATES / "amendments/review/certificate.json"
    atomic_write_json(
        publication.TEMPLATES / "amendments/z-branch/certificate.json", publication.read(cert)
    )
    with pytest.raises(ValueError, match="without branching"):
        publication.active_template_entries()


def test_target_amendment_preserves_inputs_and_pinned_history(tmp_path, monkeypatch):
    for name, suffix in (("ROOT", ""), ("DATA", "data"), ("BASE", "base")):
        monkeypatch.setattr(publication, name, tmp_path / suffix)
    atomic_write_json(publication.BASE / "schema.json", {})
    record = {
        "documentId": "doc_a",
        "positionedText": "BUSAN PORT || 5,9",
        "target": {
            "negotiability": "non_negotiable",
            "route": {"portOfLoading": {"name": "BUSAN PORT"}},
        },
    }
    path = publication.DATA / "batches/one/real-train.jsonl"
    atomic_publish_bytes(path, publication.jsonl([record]))
    certificate = path.parent / "certificate.json"
    atomic_write_json(certificate, {"records": {"real-train": publication.pin(path)}})
    previous_bytes = path.read_bytes()
    review_path = tmp_path / "001-locality.json"
    review = {
        "previous": publication.pin(certificate),
        "rationale": "Reviewed generic descriptor",
        "changes": {
            "doc_a": [
                {"path": "/route/portOfLoading/name", "before": "BUSAN PORT", "after": "BUSAN"}
            ]
        },
    }
    atomic_write_json(review_path, review)
    publication.amend_targets("one", review_path)
    active = publication.active_source_certificates()["one"]
    amended = list(
        publication.rows(publication.verify_pin(publication.read(active)["records"]["real-train"]))
    )
    assert amended[0]["positionedText"] == record["positionedText"]
    assert amended[0]["target"]["route"]["portOfLoading"]["name"] == "BUSAN"
    assert path.read_bytes() == previous_bytes
    with pytest.raises(ValueError, match="current evidence"):
        publication.amend_targets("one", review_path)
    review["previous"] = publication.pin(active)
    review["changes"]["doc_a"][0]["before"] = "WRONG"
    second = tmp_path / "002-locality.json"
    atomic_write_json(second, review)
    with pytest.raises(ValueError, match="before-value"):
        publication.amend_targets("one", second)
    assert not (publication.DATA / "amendments/002-locality").exists()
    # An already certified review is immutable, just like source/overlay evidence.
    atomic_write_json(review_path, {"changed": True})
    with pytest.raises(ValueError, match="evidence changed"):
        publication.active_source_certificates()


@pytest.mark.parametrize("splits", [("real-train",), ("validation",), ("real-train", "validation")])
def test_measurement_addition_preserves_every_split_and_input(tmp_path, monkeypatch, splits):
    for name, suffix in (("ROOT", ""), ("DATA", "data"), ("BASE", "base")):
        monkeypatch.setattr(publication, name, tmp_path / suffix)
    atomic_write_json(publication.BASE / "schema.json", {})
    records = {}
    for split in splits:
        record = {
            "documentId": "doc_" + split,
            "positionedText": "GW 7,737.00 KOS || 40,51\nTARE 2030 KOS || 4,52",
            "target": {
                "negotiability": "non_negotiable",
                "goodsItemDetails": [{"description": "GOODS"}],
            },
        }
        path = publication.DATA / ("batches/pilot/" + split + ".jsonl")
        atomic_publish_bytes(path, publication.jsonl([record]))
        records[split] = (record, path, path.read_bytes())
    cert = publication.DATA / "batches/pilot/certificate.json"
    atomic_write_json(cert, {"records": {s: publication.pin(r[1]) for s, r in records.items()}})
    review = {
        "previous": publication.pin(cert),
        "rationale": (
            "Explicit GW and accepted KOS kilogram abbreviation; tare is not cargo weight."
        ),
        "changes": {
            "doc_" + splits[-1]: [
                {
                    "operation": "addMeasurement",
                    "beforeAbsent": True,
                    "path": "/goodsItemDetails/0/grossWeight",
                    "after": {"unit": "kilogram", "value": 7737},
                    "rationale": "Native gross role, amount and KOS unit reviewed.",
                }
            ]
        },
    }
    path = tmp_path / "001-mass.json"
    atomic_write_json(path, review)
    publication.amend_targets("pilot", path)
    active = publication.read(publication.active_source_certificates()["pilot"])
    assert set(active["records"]) == set(splits)
    for split, (original, old_path, old_bytes) in records.items():
        actual = next(publication.rows(publication.verify_pin(active["records"][split])))
        expected = copy.deepcopy(original)
        if split == splits[-1]:
            expected["target"]["goodsItemDetails"][0]["grossWeight"] = {
                "unit": "kilogram",
                "value": 7737,
            }
        assert actual == expected
        assert old_path.read_bytes() == old_bytes
    review["previous"] = publication.pin(publication.active_source_certificates()["pilot"])
    atomic_write_json(tmp_path / "002-overwrite.json", review)
    with pytest.raises(ValueError, match="overwrite"):
        publication.amend_targets("pilot", tmp_path / "002-overwrite.json")
    assert not (publication.DATA / "amendments/002-overwrite").exists()
    edit = next(iter(review["changes"].values()))[0]
    for bad in (
        {**edit, "path": "/goodsItemDetails/0"},
        {**edit, "beforeAbsent": False},
        {**edit, "before": None},
        {**edit, "after": {"unit": "kilogram", "value": 7737, "unsupported": True}},
    ):
        review["changes"] = {"doc_" + splits[-1]: [bad]}
        atomic_write_json(tmp_path / "003-invalid.json", review)
        with pytest.raises(ValueError, match="explicit absent field"):
            publication.amend_targets("pilot", tmp_path / "003-invalid.json")
    assert not (publication.DATA / "amendments/003-invalid").exists()


def test_reviewed_phone_list_amendment_removes_fax_without_rewriting_input(tmp_path, monkeypatch):
    for name, suffix in (("ROOT", ""), ("DATA", "data"), ("BASE", "base")):
        monkeypatch.setattr(publication, name, tmp_path / suffix)
    atomic_write_json(publication.BASE / "schema.json", {})
    record = {
        "documentId": "doc_a",
        "positionedText": "TEL: +82 31 111 2222\nFAX: +82 31 111 3333",
        "target": {
            "negotiability": "non_negotiable",
            "parties": {
                "shipper": {
                    "contactDetails": {"phoneNumbers": ["+82 31 111 2222", "+82 31 111 3333"]}
                }
            },
        },
    }
    path = publication.DATA / "batches/one/real-train.jsonl"
    atomic_publish_bytes(path, publication.jsonl([record]))
    cert = path.parent / "certificate.json"
    atomic_write_json(cert, {"records": {"real-train": publication.pin(path)}})
    review = {
        "previous": publication.pin(cert),
        "rationale": "Separately captioned fax is input-only under the extraction policy.",
        "changes": {
            "doc_a": [
                {
                    "path": "/parties/shipper/contactDetails/phoneNumbers",
                    "before": ["+82 31 111 2222", "+82 31 111 3333"],
                    "after": ["+82 31 111 2222"],
                }
            ]
        },
    }
    review_path = tmp_path / "001-fax.json"
    atomic_write_json(review_path, review)
    publication.amend_targets("one", review_path)
    active = publication.read(publication.active_source_certificates()["one"])
    amended = list(publication.rows(publication.verify_pin(active["records"]["real-train"])))
    assert amended[0]["positionedText"] == record["positionedText"]
    assert amended[0]["target"]["parties"]["shipper"]["contactDetails"]["phoneNumbers"] == [
        "+82 31 111 2222"
    ]
    assert list(publication.rows(path)) == [record]
    review["previous"] = publication.pin(publication.active_source_certificates()["one"])
    review["changes"]["doc_a"][0]["path"] = "/parties"
    review["changes"]["doc_a"][0]["before"] = record["target"]["parties"]
    second = tmp_path / "002-invalid.json"
    atomic_write_json(second, review)
    with pytest.raises(ValueError, match="before-value"):
        publication.amend_targets("one", second)
    assert not (publication.DATA / "amendments/002-invalid").exists()
