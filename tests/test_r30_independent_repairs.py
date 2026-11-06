import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
import stage_r30_independent_repairs as repairs
from stage_r30_template_repairs import apply_overlay_review

from document_ocr.spatial_inputs.paddle import PaddlePage, PaddleRegion, serialize


def test_overlay_add_and_replace_items_require_exact_unique_review():
    original = {"extras": [], "items": [{"value": 1}]}
    review = {"reason": "reviewed", "add": {"flag": True},
              "replaceItems": {"items": [{"before": {"value": 1}, "after": {"value": 2}}]}}
    assert apply_overlay_review(original, review) == {"extras": [], "items": [{"value": 2}], "flag": True}
    assert original == {"extras": [], "items": [{"value": 1}]}
    with pytest.raises(ValueError, match="already exists"):
        apply_overlay_review({**original, "flag": False}, review)
    with pytest.raises(ValueError, match="unique"):
        apply_overlay_review({**original, "items": [{"value": 1}, {"value": 1}]}, review)


def pages():
    return (
        PaddlePage(0, 500, 500, (
            PaddleRegion(0, "MSCU 1234567", (10, 10, 100, 30)),
            PaddleRegion(1, "TGHU9876543", (10, 40, 100, 60)),
        )),
    )


def test_target_edits_are_explicit_and_do_not_mutate_input():
    original = {"parties": {"shipper": {"name": "A", "contactDetails": {"contactName": "B"}}}}
    before = copy.deepcopy(original)
    result = repairs.amended_target(original, {
        "set": {"/parties/shipper/name": "C"},
        "remove": ["/parties/shipper/contactDetails/contactName"],
    }, pages())
    assert original == before
    assert result == {"parties": {"shipper": {"name": "C"}}}
    for review, error in (
        ({"set": {"/parties/shipper/name": "A"}}, "changes nothing"),
        ({"set": {"/parties/shipper": {}, "/parties/shipper/name": "C"}}, "overlap"),
    ):
        with pytest.raises(ValueError, match=error):
            repairs.amended_target(original, review, pages())
    with pytest.raises(KeyError):
        repairs.amended_target(original, {"remove": ["/parties/shipper/country"]}, pages())
    with pytest.raises(KeyError):
        repairs.amended_target(original, {"set": {"/route/placeOfDelivery/name": "A"}}, pages())


def test_contact_cleanup_preserves_remaining_contact_facts():
    original = {"parties": {"shipper": {"contactDetails": {"contactName": "B", "phoneNumbers": ["123456"]}}}}
    result = repairs.amended_target(original, {"remove": ["/parties/shipper/contactDetails/contactName"]}, pages())
    assert result["parties"]["shipper"]["contactDetails"] == {"phoneNumbers": ["123456"]}


def test_container_order_moves_whole_objects_and_preserves_allocations():
    target = {
        "containerInformation": [
            {"equipmentIdentifier": "TGHU9876543", "sealNumbers": ["SECOND"]},
            {"equipmentIdentifier": "MSCU1234567", "sealNumbers": ["FIRST"]},
        ],
        "goodsItemDetails": [{"splitGoodsPlacement": [{"equipmentIdentifier": "TGHU9876543", "packageQuantity": 10}]}],
    }
    result = repairs.amended_target(target, {"orderContainersByFirstOccurrence": True}, pages())
    assert [r["sealNumbers"] for r in result["containerInformation"]] == [["FIRST"], ["SECOND"]]
    assert result["goodsItemDetails"] == target["goodsItemDetails"]
    with pytest.raises(ValueError, match="already native"):
        repairs.amended_target(result, {"orderContainersByFirstOccurrence": True}, pages())
    target["containerInformation"][0]["equipmentIdentifier"] = "UNKNOWN"
    with pytest.raises(ValueError, match="unprinted"):
        repairs.amended_target(target, {"orderContainersByFirstOccurrence": True}, pages())


def test_source_stage_does_not_touch_views_or_identity(monkeypatch):
    monkeypatch.setattr(repairs, "check_training_target", lambda _: None)
    pp = pages()
    record = {
        "documentId": "test", "sourceDocumentId": "test", "evidence": {"path": "previous"},
        "pages": json.loads(json.dumps([asdict(p) for p in pp])),
        "joinedRawText": serialize(pp, geometry="none"),
        "positionedText": serialize(pp), "bboxText": serialize(pp, geometry="bbox"),
        "target": {"route": {"finalDestination": {"name": "UNSUPPORTED"}}},
    }
    before = copy.deepcopy(record)
    result, evidence = repairs.repair_record(record, {"remove": ["/route/finalDestination"]}, None)
    assert record == before
    assert result["target"] == {"route": {}}
    assert all(result[k] == record[k] for k in repairs.VIEW_KEYS)
    assert evidence["previousEvidence"] == record["evidence"]


def test_review_loader_rejects_model_results_duplicates_and_unknown_operations(tmp_path):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"status": "MODEL_PROPOSAL", "sources": {}}))
    with pytest.raises(ValueError, match="explicit reviewed"):
        repairs.merged_reviews([path])
    status = "REVIEWED_CORRECTIONS_AWAITING_STAGING_AND_TEMPLATE_REPLAY"
    path.write_text(json.dumps({"status": status, "sources": {"abc": {"mystery": True}}}))
    with pytest.raises(ValueError, match="unknown"):
        repairs.merged_reviews([path])
    path.write_text(json.dumps({"status": status, "sources": {"abc": {"set": {"/a": 1}}}}))
    with pytest.raises(ValueError, match="duplicate"):
        repairs.merged_reviews([path, path])
