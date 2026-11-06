import copy
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/migrations"))
from rebase_r30_sources import (
    FieldReview,
    apply_proposal,
    audit_staged_record,
    check_subsumed_privacy,
    source_target,
)

from document_ocr.spatial_inputs.paddle import PaddlePage, PaddleRegion, fingerprint, serialize
from document_ocr.synthesis.paddle_validation import (
    check_new_private_ownership,
    check_private_lexemes,
    source_privacy_edits,
)


def review(path="/placeOfIssue", value='{"name":"SEOUL"}', quote="SEOUL", operation="set"):
    return FieldReview.model_validate(
        {
            "edits": [
                {
                    "operation": operation,
                    "path": path,
                    "value_json": value,
                    "reason": "explicit issue box",
                    "evidence": [{"page": 0, "region": 1, "quote": quote}],
                }
            ],
            "unresolved": [],
        }
    )


RECORD = {"pages": [{"index": 0, "regions": [{"index": 1, "text": "SEOUL"}]}]}


def test_wider_source_target_requires_exact_v7_envelope():
    target = {"issueDate": "2025-01-01"}
    training = {"split": "real-train", "current": {"target": target}}
    assert source_target(training) is target
    wider = {
        "split": "template-only",
        "current": {"target": {"schemaVersion": "7.0.0", "documentPatch": target}},
    }
    assert source_target(wider) is target
    for invalid in (target, {"schemaVersion": "6.0.0", "documentPatch": target}):
        wider["current"]["target"] = invalid
        with pytest.raises(ValueError, match="exact V7 envelope"):
            source_target(wider)


def test_wider_reference_has_no_training_anonymization_fallback():
    assert source_privacy_edits({"split": "template-only", "privacyPlan": None}) == []
    with pytest.raises(ValueError, match="requires its reviewed anonymization"):
        source_privacy_edits({"split": "real-train", "privacyPlan": None})
    page = PaddlePage(0, 100, 100, (PaddleRegion(0, "ALAN", (0, 0, 40, 10)),))
    source = {"split": "template-only", "pages": [asdict(page)]}
    decision = {"privacy": {"extras": []}, "templatePrivateOwners": [{"regions": [[0, 0]]}]}
    with pytest.raises(ValueError, match="lacks a synthetic binding"):
        check_new_private_ownership(source, {"proof": []}, decision)
    proof = [
        {
            "owner": "private",
            "after": "BETH",
            "spans": [{"page": 0, "region": 0, "start": 0, "end": 4, "expected": "ALAN"}],
        }
    ]
    check_new_private_ownership(source, {"proof": proof}, decision)
    source["split"] = "real-train"
    with pytest.raises(ValueError, match="also require anonymization"):
        check_new_private_ownership(source, {"proof": proof}, decision)


def test_new_private_owner_cannot_survive_template_transfer():
    page = PaddlePage(0, 100, 100, (PaddleRegion(0, "ALAN/BETH", (0, 0, 80, 10)),))
    source = {"pages": [asdict(page)]}
    decision = {"privacy": {"extras": [{"regions": [[0, 0]]}]}}
    proof = [
        {
            "owner": "name-a",
            "after": "CARL",
            "spans": [{"page": 0, "region": 0, "start": 0, "end": 4, "expected": "ALAN"}],
        }
    ]
    with pytest.raises(ValueError, match="lacks a synthetic binding"):
        check_new_private_ownership(source, {"proof": proof}, decision)
    proof.append(
        {
            "owner": "name-b",
            "after": "DORA",
            "spans": [{"page": 0, "region": 0, "start": 5, "end": 9, "expected": "BETH"}],
        }
    )
    check_new_private_ownership(source, {"proof": proof}, decision)
    proof[1]["spans"][0]["expected"] = "FAKE"
    with pytest.raises(ValueError, match="differs from source"):
        check_new_private_ownership(source, {"proof": proof}, decision)


def test_subsumed_privacy_requires_every_native_occurrence_to_be_owned():
    prefix = {"page": 0, "region": 1, "start": 0, "end": 8, "expected": "12345678"}
    full = dict(prefix, end=9, expected="123456789")
    check_subsumed_privacy({"spans": [prefix]}, [full])
    with pytest.raises(ValueError, match="completely covered"):
        check_subsumed_privacy({"matches": [[prefix], [dict(prefix, region=2)]]}, [full])
    with pytest.raises(ValueError, match="lacks native"):
        check_subsumed_privacy({"matches": []}, [full])


def test_new_role_with_native_evidence_preserves_current_target():
    old = {"issueDate": "2024-06-04"}
    result = apply_proposal(RECORD, old, review())
    assert result == {"issueDate": "2024-06-04", "placeOfIssue": {"name": "SEOUL"}}
    assert old == {"issueDate": "2024-06-04"}


def test_root_loop_and_duplicate_edits_fail():
    with pytest.raises(ValidationError):
        review(path="/")
    payload = review().model_dump()
    payload["edits"] *= 2
    with pytest.raises(ValidationError, match="distinct"):
        FieldReview.model_validate(payload)
    payload["edits"][1] = review(path="/placeOfIssue/name", value='"SEOUL"').model_dump()["edits"][
        0
    ]
    with pytest.raises(ValidationError, match="non-overlapping"):
        FieldReview.model_validate(payload)


def test_no_invented_evidence_no_noop_and_no_missing_parent():
    with pytest.raises(ValueError, match="evidence"):
        apply_proposal(RECORD, {}, review(quote="BUSAN"))
    with pytest.raises(ValueError, match="unchanged"):
        apply_proposal(RECORD, {"placeOfIssue": {"name": "SEOUL"}}, review())
    with pytest.raises(KeyError):
        apply_proposal(RECORD, {}, review(path="/placeOfIssue/name", value='"SEOUL"'))


def test_absence_removal_is_explicit_and_does_not_shift_arrays():
    assert (
        apply_proposal(
            RECORD, {"placeOfIssue": "incorrect"}, review(value="null", operation="remove")
        )
        == {}
    )
    with pytest.raises(ValueError, match="replace the array"):
        apply_proposal(
            RECORD, {"values": [1]}, review(path="/values/0", value="null", operation="remove")
        )


def test_json_value_is_validated_before_a_provider_response_is_accepted():
    with pytest.raises(ValidationError):
        review(value="SEOUL")
    with pytest.raises(ValidationError, match="null"):
        review(value='"SEOUL"', operation="remove")


def audit_fixture():
    pages = (PaddlePage(0, 1000, 1000, (PaddleRegion(0, "PLSTONE TYRES", (10, 10, 200, 30)),)),)
    source = {
        "pages": [asdict(p) for p in pages],
        "current": {"target": {}},
        "privacyPlan": {"rawEdits": [{"before": "PLSTONE", "after": "NVCEDAR"}]},
    }
    record = {
        "documentId": "test",
        "pages": source["pages"],
        "sourceFingerprint": fingerprint(pages),
        "fingerprint": fingerprint(pages),
        "joinedRawText": serialize(pages, geometry="none"),
        "positionedText": serialize(pages),
        "bboxText": serialize(pages, geometry="bbox"),
        "proof": [],
        "geometry": {"pages": []},
        "target": {"negotiability": None, "goodsItemDetails": [{"description": "PLSTONE TYRES"}]},
        "reviewedConflicts": [],
    }
    decision = {
        "physical": {"physicalGroups": []},
        "retainedLexemes": [
            {
                "planIndex": 0,
                "page": 0,
                "text": "PLSTONE TYRES",
                "reason": "Goods brand, not the Marks party.",
            }
        ],
    }
    return source, record, decision


def test_retained_brand_requires_exact_role_specific_witness():
    source, record, decision = audit_fixture()
    assert audit_staged_record(source, record, decision)["privacyReplacements"] == 0
    with pytest.raises(ValueError, match="without role review"):
        audit_staged_record(source, record, {"physical": decision["physical"]})
    stale = copy.deepcopy(decision)
    stale["retainedLexemes"].append({"planIndex": 0, "page": 0, "text": "OLD", "reason": "Stale"})
    with pytest.raises(ValueError, match="stale or unused"):
        audit_staged_record(source, record, stale)


def test_replay_privacy_guard_checks_input_only_repetitions_without_targets():
    source, record, _ = audit_fixture()
    # Synthetic replay has no real-stage metadata and no target for this owner.
    source["privacyPlan"]["rawEdits"] = [{"before": "PLSTONE", "after": "NVCEDAR"}]
    record.pop("target")
    with pytest.raises(ValueError, match="private-plan lexeme survives"):
        check_private_lexemes(source, record, [])
    source["privacyPlan"]["rawEdits"][0]["before"] = "ABSENT ORIGINAL"
    check_private_lexemes(source, record, [])


def test_private_standalone_mark_does_not_match_unrelated_word_substrings():
    source, record, _ = audit_fixture()
    source["privacyPlan"]["rawEdits"] = [{"before": "\nMP\n", "after": "\nNP\n"}]
    region = record["pages"][0]["regions"][0]
    for text in ("PUMP SPRAYER", "SHIPPING COMPANY", "IMPORTERVAT NO:123", "NP"):
        region["text"] = text
        check_private_lexemes(source, record, [])
    for text in ("MP", " mp "):
        region["text"] = text
        with pytest.raises(ValueError, match="private-plan lexeme survives"):
            check_private_lexemes(source, record, [])
    # Ordinary unguarded identifiers must still be found inside a longer B/L.
    source["privacyPlan"]["rawEdits"] = [{"before": "GOAE1234", "after": "GOAE5678"}]
    region["text"] = "ONEYGOAE1234"
    with pytest.raises(ValueError, match="private-plan lexeme survives"):
        check_private_lexemes(source, record, [])


def test_new_number_collision_requires_exact_generated_span_and_native_proof():
    import copy

    source, record, _ = audit_fixture()
    source = copy.deepcopy(source)
    source["privacyPlan"]["rawEdits"] = [{"before": "1234", "after": "5678"}]
    source["pages"][0]["regions"][0]["text"] = "+20263302047"
    record["pages"][0]["regions"][0]["text"] = "TEL:+971501234141"
    record["proof"] = [
        {
            "after": "+971501234141",
            "spans": [{"page": 0, "region": 0, "start": 0, "end": 12, "expected": "+20263302047"}],
        }
    ]
    check_private_lexemes(source, record, [])
    record["pages"][0]["regions"][0]["text"] += " ADDRESS 1234"
    with pytest.raises(ValueError, match="private-plan lexeme survives"):
        check_private_lexemes(source, record, [])
    record["pages"][0]["regions"][0]["text"] = "TEL:+971501234141"
    record["proof"][0]["spans"][0]["expected"] = "wrong"
    with pytest.raises(ValueError, match="proof differs"):
        check_private_lexemes(source, record, [])
    record["proof"][0]["spans"][0]["expected"] = "OLD 1234"
    with pytest.raises(ValueError, match="private-plan lexeme survives"):
        check_private_lexemes(source, record, [])


def test_real_conflict_does_not_trigger_balancing_but_missing_equipment_fails():
    source, record, decision = audit_fixture()
    record["target"] = {
        "negotiability": None,
        "containerInformation": [{"equipmentIdentifier": "MSCU6639870"}],
        "goodsItemDetails": [
            {
                "numberAndTypeOfPackages": [
                    {"packageQuantity": 58, "typeCategory": "PACKAGE_PACKAGE"}
                ],
                "splitGoodsPlacement": [
                    {"equipmentIdentifier": "MSCU6639870", "packageQuantity": 510}
                ],
            }
        ],
    }
    assert audit_staged_record(source, record, decision)["status"].startswith("STRUCTURAL")
    del record["target"]["containerInformation"]
    with pytest.raises(ValueError, match="placement references absent containerInformation"):
        audit_staged_record(source, record, decision)


def test_stale_rendered_text_or_geometry_is_rejected():
    source, record, decision = audit_fixture()
    record["positionedText"] += " EXTRA"
    with pytest.raises(ValueError, match="centroid view"):
        audit_staged_record(source, record, decision)
    record["fingerprint"] = "stale"
    with pytest.raises(ValueError, match="fingerprint"):
        audit_staged_record(source, record, decision)
