from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/bill_of_lading_v2_full_run.py"
_SPEC = importlib.util.spec_from_file_location("bill_of_lading_v2_full_run", _TOOL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to import semantic-v2 full-run tool")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _work_item(document_id: str, raw_hash: str, raster_hash: str) -> dict[str, object]:
    return {
        "source": {
            "documentId": document_id,
            "joinedRawTextSha256": raw_hash,
            "pages": [{"rasterSha256": raster_hash}],
        }
    }


def test_page_splitter_preserves_order() -> None:
    joined = "--- PAGE 1 ---\nFIRST\n\n--- PAGE 2 ---\nSECOND"

    assert _MODULE._page_texts(joined) == {1: "FIRST", 2: "SECOND"}


def test_country_resolver_uses_frozen_iso_snapshot_and_explicit_aliases() -> None:
    index, digest = _MODULE._country_index()

    assert len(digest) == 64
    assert index[_MODULE._normalize_country("Netherlands")] == "NL"
    assert index[_MODULE._normalize_country("U.K") ] == "GB"
    assert index[_MODULE._normalize_country("Türkiye")] == "TR"


def test_duplicate_groups_join_exact_ocr_and_first_page_raster_components() -> None:
    items = [
        _work_item("doc_a", "1" * 64, "a" * 64),
        _work_item("doc_b", "2" * 64, "a" * 64),
        _work_item("doc_c", "2" * 64, "c" * 64),
        _work_item("doc_d", "4" * 64, "d" * 64),
    ]

    groups = _MODULE._duplicate_groups(items)

    assert len(groups) == 1
    assert groups[0]["documentIds"] == ["doc_a", "doc_b", "doc_c"]
    assert {row["kind"] for row in groups[0]["identityEvidence"]} == {
        "first_page_raster",
        "joined_raw_text",
    }


def test_semantic_gate_checks_flavor_text_and_address_redundancy() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "name": "TAX ID: 123",
                    "address": "11 MAIN STREET, CAIRO",
                    "city": "CAIRO",
                }
            },
            "goodsItems": [{"marksAndNumbers": ["SHIPPER'S LOAD & COUNT"]}],
        }
    }

    findings = _MODULE._semantic_findings(target)

    assert any("prohibited tax/boilerplate" in finding for finding in findings)
    assert any("address repeats separately emitted city" in finding for finding in findings)


def test_semantic_gate_rejects_explicit_null_in_sparse_target() -> None:
    target = {
        "documentPatch": {
            "masterBillOfLadingNumber": None,
            "parties": {"shipper": {"name": "EXAMPLE LTD", "city": None}},
        }
    }

    assert _MODULE._semantic_findings(target) == [
        "documentPatch.masterBillOfLadingNumber: explicit null must be omitted from sparse target",
        "documentPatch.parties.shipper.city: explicit null must be omitted from sparse target",
    ]


def test_semantic_gate_rejects_postal_label_but_keeps_postal_value() -> None:
    contaminated = {
        "documentPatch": {
            "parties": {"shipper": {"address": "P.O. BOX 4 POSTAL CODE 107"}}
        }
    }
    clean = {
        "documentPatch": {"parties": {"shipper": {"address": "P.O. BOX 4 107"}}}
    }

    assert _MODULE._semantic_findings(contaminated) == [
        "documentPatch.parties.shipper.address: postal value retains its field label "
        "'P.O. BOX 4 POSTAL CODE 107'"
    ]
    assert _MODULE._semantic_findings(clean) == []


@pytest.mark.parametrize(
    "heading",
    [
        "Delivery Agent at place of delivery",
        "9. Destination Agent",
        "AGENT AT PORT OF DISCHARGE",
        "PORT OF DISCHARGE AGENT: ARKAS EGYPT",
        "Agent's Address at Destination:",
        "FOR DELIVERY OF GOODS PLEASE APPLY TO",
        "TO OBTAIN DELIVERY CONTACT:",
        "(AS FRT FWDRS DELIVERY AGENT ONLY)",
    ],
)
def test_delivery_agent_role_gate_accepts_explicit_role_heading(heading: str) -> None:
    target = {"documentPatch": {"parties": {"deliveryAgent": {"name": "AGENT"}}}}

    assert _MODULE._source_role_findings(target, heading) == []


@pytest.mark.parametrize(
    "heading",
    [
        "SHIPPING AGENT DETAILS",
        "DESTINATION OFFICE ADDRESS",
        "NAME AND FULL ADDRESS OF SHIPPING AGENT IN EGYPT",
        "AGENT ADDRESS",
    ],
)
def test_delivery_agent_role_gate_rejects_generic_agent_heading(heading: str) -> None:
    target = {"documentPatch": {"parties": {"deliveryAgent": {"name": "AGENT"}}}}

    assert _MODULE._source_role_findings(target, heading) == [
        "deliveryAgent requires an explicit destination/discharge/delivery-agent source "
        "heading"
    ]


def test_conditional_non_negotiable_gate_requires_named_consignee_evidence() -> None:
    annotation = SimpleNamespace(
        evidence=(
            SimpleNamespace(
                targetPath="documentPatch.negotiability",
                rawOcrEvidence=(
                    SimpleNamespace(
                        rawValue="NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER"
                    ),
                ),
            ),
        )
    )
    target = {
        "documentPatch": {
            "negotiability": "non_negotiable",
            "parties": {"consignee": {"name": "NAMED IMPORTER LTD"}},
        }
    }

    assert _MODULE._negotiability_evidence_findings(
        annotation,
        target,
        "NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER",
    ) == ["conditional non-negotiable target must cite the emitted named consignee"]


def test_conditional_non_negotiable_gate_accepts_named_consignee_evidence() -> None:
    annotation = SimpleNamespace(
        evidence=(
            SimpleNamespace(
                targetPath="documentPatch.negotiability",
                rawOcrEvidence=(
                    SimpleNamespace(
                        rawValue="NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER"
                    ),
                    SimpleNamespace(rawValue="NAMED IMPORTER LTD"),
                ),
            ),
        )
    )
    target = {
        "documentPatch": {
            "negotiability": "non_negotiable",
            "parties": {"consignee": {"name": "NAMED IMPORTER LTD"}},
        }
    }

    assert _MODULE._negotiability_evidence_findings(
        annotation,
        target,
        "NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER",
    ) == []


def test_semantic_gate_distinguishes_chemical_acid_from_customs_acid_id() -> None:
    chemical = {
        "documentPatch": {
            "goodsItems": [
                {"description": "CORROSIVE LIQUID (CAPRYLIC ACID)"}
            ]
        }
    }
    customs_identifier = {
        "documentPatch": {
            "goodsItems": [{"description": "ACID: 1004977722024020145"}]
        }
    }

    assert _MODULE._semantic_findings(chemical) == []
    assert any(
        "prohibited tax/boilerplate" in finding
        for finding in _MODULE._semantic_findings(customs_identifier)
    )


def test_address_redundancy_gate_does_not_match_city_inside_facility_name() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "address": (
                        "ROOM 1702, INTERNATIONAL TRADE CENTER OF YUYAO PLASTIC CITY, "
                        "ZHEJIANG"
                    ),
                    "city": "YUYAO",
                    "country": "CHINA",
                }
            }
        }
    }

    assert _MODULE._semantic_findings(target) == []


def test_address_redundancy_gate_matches_city_before_state_and_postal_code() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "address": "37 EDISON AVE. WEST BABYLON NY 11704",
                    "city": "WEST BABYLON",
                    "country": "USA",
                }
            }
        }
    }

    assert _MODULE._semantic_findings(target) == [
        "party address repeats separately emitted city 'WEST BABYLON'"
    ]


def test_address_redundancy_gate_allows_city_name_inside_distinct_locality() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "notifyParties": [
                    {
                        "address": (
                            "BLOCK NO. 4, ABDEL RAHMAN BISAR ST., "
                            "FIRST SETTLEMENT, NEW CAIRO"
                        ),
                        "city": "CAIRO",
                        "country": "EGYPT",
                    }
                ]
            }
        }
    }

    assert _MODULE._semantic_findings(target) == []


def test_address_redundancy_gate_matches_terminal_city_after_postal_code() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "address": "BLEICHERSTR. 28 DE-73066 UHINGEN",
                    "city": "UHINGEN",
                    "country": "GERMANY",
                }
            }
        }
    }

    assert _MODULE._semantic_findings(target) == [
        "party address repeats separately emitted city 'UHINGEN'"
    ]


def test_semantic_gate_rejects_generic_freight_payment_place() -> None:
    target = {
        "documentPatch": {
            "freight": {
                "paymentArrangement": "collect",
                "paymentPlace": {"name": "DESTINATION"},
            }
        }
    }

    assert _MODULE._semantic_findings(target) == [
        "freight payment place is a generic direction 'DESTINATION'"
    ]


def test_warning_gate_rejects_valid_container_claimed_as_invalid() -> None:
    annotation = SimpleNamespace(
        warnings=(
            SimpleNamespace(
                code="invalid_identifier_omitted",
                message="Printed GCXU2131234 fails ISO 6346 validation.",
            ),
        )
    )

    assert _MODULE._warning_findings(annotation) == [
        "invalid_identifier_omitted warning names valid ISO 6346 identifier "
        "'GCXU2131234'"
    ]


def test_phone_evidence_gate_distinguishes_tel_and_fax_on_same_line() -> None:
    path = "documentPatch.parties.deliveryAgent.contactDetails.phoneNumbers[0]"
    excerpt = "Tel.+20 3 4833755 Fax.+20 3 4836774"

    assert not _MODULE._phone_evidence_is_fax(path, "+20 3 4833755", excerpt)
    assert _MODULE._phone_evidence_is_fax(path, "+20 3 4836774", excerpt)


def test_phone_evidence_gate_allows_joint_tel_fax_value_as_phone() -> None:
    path = "documentPatch.parties.consignee.contactDetails.phoneNumbers[0]"
    excerpt = "Tel / Fax :+201003300211 TAX ID NO : 599640898"

    assert not _MODULE._phone_evidence_is_fax(path, "+201003300211", excerpt)


def test_semantic_gate_rejects_contact_value_with_joint_phone_fax_label() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "contactDetails": {
                        "phoneNumbers": ["PHONE & FAX 002 02 2575 2002"]
                    }
                }
            }
        }
    }

    assert any(
        "communication value retains its field label" in finding
        for finding in _MODULE._semantic_findings(target)
    )


def test_semantic_gate_rejects_forwarding_reference_with_field_label() -> None:
    target = {
        "documentPatch": {
            "forwardingAndExportReferences": ["Svc Contract 299232432"]
        }
    }

    assert any(
        "reference value retains its field label" in finding
        for finding in _MODULE._semantic_findings(target)
    )


def test_semantic_gate_allows_invoice_prefix_that_is_part_of_identifier() -> None:
    target = {
        "documentPatch": {
            "forwardingAndExportReferences": ["INV/2023/00186"]
        }
    }

    assert _MODULE._semantic_findings(target) == []


def test_semantic_gate_rejects_contact_footnote_and_mark_field_label() -> None:
    target = {
        "documentPatch": {
            "parties": {
                "notifyParties": [
                    {
                        "contactDetails": {
                            "emailAddresses": ["OPS@EXAMPLE.COM***"]
                        }
                    }
                ]
            },
            "goodsItems": [{"marksAndNumbers": ["C/NO. 30 /3,5,8"]}],
        }
    }

    findings = _MODULE._semantic_findings(target)

    assert any("communication value retains a footnote marker" in row for row in findings)
    assert any("mark value retains its field label" in row for row in findings)


@pytest.mark.parametrize(
    ("target_path", "target", "raw_value"),
    [
        ("documentPatch.issueDate", "2023-09-30", "SEP/30/2023"),
        ("documentPatch.issueDate", "2024-03-29", "MARCH 29TH, 2024"),
        ("documentPatch.issueDate", "2024-06-28", "JUN.28,2024"),
        ("documentPatch.goodsItems[0].grossWeight.value", 60960, "60,960KG"),
        ("documentPatch.goodsItems[0].grossWeight.value", 7740, "7.740 KG"),
        ("documentPatch.goodsItems[0].grossWeight.value", 5370.153, "5370,153"),
        ("documentPatch.goodsItems[0].packages[0].quantity", 6, "SIX"),
        (
            "documentPatch.goodsItems[0].packages[0].quantity",
            52,
            "TOTAL: FIFTY-TWO IBC ONLY",
        ),
        (
            "documentPatch.goodsItems[0].packages[0].quantity",
            5272,
            "FIVE THOUSAND TWO HUNDRED AND SEVENTY TWO ONLY",
        ),
        ("documentPatch.goodsItems[0].grossWeight.unit", "kilogram", "KGS"),
        ("documentPatch.goodsItems[0].volume.unit", "cubic_metre", "cu. m."),
        ("documentPatch.negotiability", "non_negotiable", "NON-NEGOTIABLE"),
        (
            "documentPatch.negotiability",
            "non_negotiable",
            "Bill of Lading Type EXPRESS / EXPRESS BILL OF LADING",
        ),
        (
            "documentPatch.negotiability",
            "non_negotiable",
            "In accepting this Waybill the Merchant expressly accepts its terms",
        ),
        (
            "documentPatch.negotiability",
            "non_negotiable",
            (
                "Consignee (negotiable only if consigned to order) "
                "AL MAJIC FOR IMPORT EXPORT"
            ),
        ),
        ("documentPatch.parties.shipper.name", "Solucoes", "Soluções"),
        ("documentPatch.containers[0].containerNumber", "YMLU3540304", "YMLU 3540304"),
    ],
)
def test_target_leaf_support_accepts_authorized_normalizations(
    target_path: str, target: object, raw_value: str
) -> None:
    assert _MODULE._target_leaf_is_supported(target_path, target, raw_value)


@pytest.mark.parametrize(
    ("target_path", "target", "raw_value"),
    [
        ("documentPatch.billOfLadingNumber", "VASHYDSOK001653", "BL Barcode"),
        (
            "documentPatch.parties.deliveryAgent.name",
            "EGYPTIAN INTERNATIONAL SHIPPING AGENCIES AND SERVICES",
            "Delivery",
        ),
        ("documentPatch.goodsItems[0].grossWeight.value", 13884.09, "weight"),
        ("documentPatch.goodsItems[0].grossWeight.unit", "kilogram", "weight"),
        (
            "documentPatch.negotiability",
            "non_negotiable",
            "EXPRESS RELEASE REQUEST",
        ),
    ],
)
def test_target_leaf_support_rejects_unrelated_raw_ocr(
    target_path: str, target: object, raw_value: str
) -> None:
    assert not _MODULE._target_leaf_is_supported(target_path, target, raw_value)


def test_duplicate_comparison_surfaces_shared_path_contradictions() -> None:
    groups = [{"groupId": "duplicate-001", "documentIds": ["doc_a", "doc_b"]}]
    targets = {
        "doc_a": {"documentPatch": {"billOfLadingNumber": "A"}},
        "doc_b": {"documentPatch": {"billOfLadingNumber": "B"}},
    }

    comparisons = _MODULE._duplicate_comparisons(groups, targets)

    assert comparisons[0]["pairwise"][0]["contradictions"] == [
        {"path": "documentPatch.billOfLadingNumber", "first": "A", "second": "B"}
    ]


def test_manifest_digest_recovers_after_two_artifact_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    monkeypatch.setattr(_MODULE, "RUN_ID", "test-run")
    (tmp_path / "payload.json").write_text('{"ok":true}\n', encoding="utf-8")

    _MODULE._publish_manifest()
    (tmp_path / "manifest.json.sha256").unlink()
    _MODULE._recover_manifest_digest()

    digest_line = (tmp_path / "manifest.json.sha256").read_text(encoding="utf-8")
    assert digest_line.endswith("  manifest.json\n")


def test_worker_start_rejects_unknown_document_without_writing_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)

    with pytest.raises(RuntimeError, match="work item does not exist"):
        _MODULE.record_worker_start("doc_missing", 1, "/root/worker")

    assert not (tmp_path / "worker-logs").exists()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_omit_explicit_nulls_preserves_sparse_values_and_list_order() -> None:
    value = {
        "present": "value",
        "absent": None,
        "nested": {"keep": 0, "drop": None},
        "items": [{"keep": False, "drop": None}, None, "tail"],
    }

    assert _MODULE._omit_explicit_nulls(value) == {
        "present": "value",
        "nested": {"keep": 0},
        "items": [{"keep": False}, "tail"],
    }


def test_check_attempt_records_exact_candidate_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_digest"
    attempt_path = (
        tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    )
    _write_json(attempt_path, {"candidate": "exact bytes"})
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-1.json",
        {
            "status": "completed",
            "candidateSha256": _MODULE._file_digest(attempt_path),
        },
    )
    monkeypatch.setattr(
        _MODULE,
        "_validate_attempt",
        lambda _document_id, _path: ("excluded", {"rawOcrEvidence": []}, None, {}),
    )

    _MODULE.check_attempt(document_id, 1)

    check = json.loads(
        (
            tmp_path
            / "validation/attempt-checks"
            / f"{document_id}.attempt-1.json"
        ).read_text(encoding="utf-8")
    )
    assert check["attemptSha256"] == _MODULE._file_digest(attempt_path)


def test_worker_finish_seals_completed_candidate_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_sealed"
    attempt_path = (
        tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    )
    _write_json(attempt_path, {"candidate": "published"})
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-1.json",
        {"status": "running", "completedAt": None},
    )

    _MODULE.record_worker_finish(document_id, 1, "completed")

    worker = json.loads(
        (
            tmp_path / "worker-logs" / f"{document_id}.attempt-1.json"
        ).read_text(encoding="utf-8")
    )
    assert worker["candidatePath"] == (
        f"candidate-attempts/{document_id}.attempt-1.json"
    )
    assert worker["candidateSha256"] == _MODULE._file_digest(attempt_path)
    assert worker["status"] == "completed"
    assert worker["completedAt"] is not None


def test_check_attempt_seals_failure_and_rejects_mutated_completed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_mutated_before_check"
    attempt_path = (
        tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    )
    _write_json(attempt_path, {"candidate": "sealed"})
    sealed_digest = _MODULE._file_digest(attempt_path)
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-1.json",
        {"status": "completed", "candidateSha256": sealed_digest},
    )
    _write_json(attempt_path, {"candidate": "mutated"})

    with pytest.raises(RuntimeError, match="changed after worker completion"):
        _MODULE.check_attempt(document_id, 1)

    check_path = (
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-1.json"
    )
    failed = json.loads(check_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["attemptSha256"] == _MODULE._file_digest(attempt_path)

    with pytest.raises(RuntimeError, match="attempt check is immutable"):
        _MODULE.check_attempt(document_id, 1)


def test_review_rejects_candidate_mutated_after_passing_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_mutated"
    attempt_path = (
        tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json"
    )
    _write_json(attempt_path, {"candidate": "checked"})
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-1.json",
        {
            "status": "completed",
            "taskName": "/root/worker",
            "model": "gpt-5.6-luna",
            "reasoningEffort": "high",
            "startedAt": "2026-08-17T00:00:00Z",
            "completedAt": "2026-08-17T00:01:00Z",
            "usage": "unavailable_from_collaboration_runtime",
        },
    )
    _write_json(
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-1.json",
        {
            "status": "passed",
            "kind": "approved",
            "attemptSha256": _MODULE._file_digest(attempt_path),
        },
    )
    _write_json(attempt_path, {"candidate": "changed after check"})

    with pytest.raises(RuntimeError, match="changed after automated validation"):
        _MODULE.record_review(
            document_id, 1, "accepted", "include", ["looks correct"]
        )

    assert not (
        tmp_path
        / "validation/attempt-reviews"
        / f"{document_id}.attempt-1.json"
    ).exists()


def test_review_rejects_implicit_decision_overwrite_before_writing_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_a"
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-2.json",
        {
            "status": "completed",
            "taskName": "/root/worker",
            "model": "gpt-5.6-luna",
            "reasoningEffort": "high",
            "startedAt": "2026-08-17T00:00:00Z",
            "completedAt": "2026-08-17T00:01:00Z",
            "usage": "unavailable_from_collaboration_runtime",
        },
    )
    _write_json(
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-2.json",
        {"status": "passed", "kind": "annotation"},
    )
    _write_json(
        tmp_path / "validation/decisions" / f"{document_id}.json",
        {"documentId": document_id, "attempt": 1, "decision": "approve"},
    )

    with pytest.raises(RuntimeError, match="already has an accepted decision"):
        _MODULE.record_review(
            document_id, 2, "accepted", "include", ["corrected candidate"]
        )

    assert not (
        tmp_path
        / "validation/attempt-reviews"
        / f"{document_id}.attempt-2.json"
    ).exists()


def test_review_can_reject_failed_check_without_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_failed"
    _write_json(
        tmp_path / "candidate-attempts" / f"{document_id}.attempt-1.json",
        {"candidate": "semantically contaminated"},
    )
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-1.json",
        {
            "status": "completed",
            "taskName": "/root/worker",
            "model": "gpt-5.6-luna",
            "reasoningEffort": "high",
            "startedAt": "2026-08-17T00:00:00Z",
            "completedAt": "2026-08-17T00:01:00Z",
            "usage": "unavailable_from_collaboration_runtime",
        },
    )
    _write_json(
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-1.json",
        {"status": "failed", "error": "semantic contamination"},
    )

    _MODULE.record_review(
        document_id, 1, "rejected", None, ["remove duplicated country"]
    )

    review = json.loads(
        (
            tmp_path
            / "validation/attempt-reviews"
            / f"{document_id}.attempt-1.json"
        ).read_text(encoding="utf-8")
    )
    assert review["result"] == "rejected"
    assert not (tmp_path / "validation/decisions" / f"{document_id}.json").exists()


def test_review_explicitly_supersedes_and_preserves_decision_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)
    document_id = "doc_a"
    worker = {
        "status": "completed",
        "taskName": "/root/worker",
        "model": "gpt-5.6-luna",
        "reasoningEffort": "high",
        "startedAt": "2026-08-17T00:00:00Z",
        "completedAt": "2026-08-17T00:01:00Z",
        "usage": "unavailable_from_collaboration_runtime",
    }
    _write_json(
        tmp_path / "worker-logs" / f"{document_id}.attempt-2.json", worker
    )
    attempt_path = (
        tmp_path / "candidate-attempts" / f"{document_id}.attempt-2.json"
    )
    _write_json(attempt_path, {"candidate": "corrected"})
    _write_json(
        tmp_path / "validation/attempt-checks" / f"{document_id}.attempt-2.json",
        {
            "status": "passed",
            "kind": "approved",
            "attemptSha256": _MODULE._file_digest(attempt_path),
        },
    )
    previous = {
        "documentId": document_id,
        "attempt": 1,
        "attemptPath": f"candidate-attempts/{document_id}.attempt-1.json",
        "decision": "approve",
        "trainingDisposition": "include",
        "reviewPath": f"validation/attempt-reviews/{document_id}.attempt-1.json",
        "decidedAt": "2026-08-17T00:02:00Z",
    }
    decision_path = tmp_path / "validation/decisions" / f"{document_id}.json"
    _write_json(decision_path, previous)

    _MODULE.record_review(
        document_id,
        2,
        "accepted",
        "include",
        ["corrected candidate"],
        supersede_accepted=True,
    )

    history_path = (
        tmp_path
        / "validation/decision-history"
        / f"{document_id}.attempt-1.json"
    )
    assert json.loads(history_path.read_text(encoding="utf-8")) == previous
    current = json.loads(decision_path.read_text(encoding="utf-8"))
    assert current["attempt"] == 2
    assert current["supersedesDecisionPath"] == (
        f"validation/decision-history/{document_id}.attempt-1.json"
    )


def test_size_comparison_summary_reports_exact_totals_and_delta() -> None:
    rows = [
        {
            "v1Leaves": 10,
            "v2Leaves": 7,
            "v1CanonicalBytes": 100,
            "v2CanonicalBytes": 80,
            "v1Tokens": 20,
            "v2Tokens": 15,
        },
        {
            "v1Leaves": 6,
            "v2Leaves": 5,
            "v1CanonicalBytes": 60,
            "v2CanonicalBytes": 50,
            "v1Tokens": 12,
            "v2Tokens": 10,
        },
    ]

    assert _MODULE._size_comparison_summary(rows) == {
        "documents": 2,
        "leaves": {
            "v1Total": 16,
            "v2Total": 12,
            "absoluteDelta": -4,
            "percentDelta": -25.0,
        },
        "canonicalBytes": {
            "v1Total": 160,
            "v2Total": 130,
            "absoluteDelta": -30,
            "percentDelta": -18.75,
        },
        "tokens": {
            "v1Total": 32,
            "v2Total": 25,
            "absoluteDelta": -7,
            "percentDelta": -21.875,
        },
    }


def test_size_comparison_summary_requires_overlap() -> None:
    with pytest.raises(RuntimeError, match="overlapping validated documents"):
        _MODULE._size_comparison_summary([])


def test_eda_counts_leaf_paths_instead_of_treating_values_as_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "DEST", tmp_path)

    def annotation(target: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            label=SimpleNamespace(canonical_target=lambda: {"documentPatch": target}),
            warnings=(SimpleNamespace(code="schema_cannot_represent"),),
            source=SimpleNamespace(documentPageCount=1),
            documentType="bill_of_lading",
        )

    tokenizer = SimpleNamespace(
        encode=lambda value, add_special_tokens: value.split()
    )
    result = _MODULE._eda(
        {
            "doc_a": annotation(
                {"billOfLadingNumber": "A", "issueDate": "2024-01-01"}
            ),
            "doc_b": annotation({"billOfLadingNumber": "B"}),
        },
        {},
        tokenizer,
    )

    field_rows = (
        tmp_path / "eda/tables/field_presence.csv"
    ).read_text(encoding="utf-8").splitlines()
    assert "documentPatch.billOfLadingNumber,2" in field_rows
    assert "documentPatch.issueDate,1" in field_rows
    assert result["validatedDocuments"] == 2
