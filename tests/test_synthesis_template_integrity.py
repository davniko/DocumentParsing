from __future__ import annotations

from document_ocr.synthesis.template_integrity import (
    explicit_carrier_receipt_container_counts,
    source_template_integrity_issues,
)


def test_carrier_receipt_container_count_accepts_same_and_following_line_grammars() -> None:
    raw = (
        "CARRIER'S RECEIPT (No. of Cntrs or Pkgs rcvd by Carrier)\n"
        "5 CNTRS\n"
        "CARRIER\N{RIGHT SINGLE QUOTATION MARK}S RECEIPT: 3 CONTAINER(S)\n"
        "Cargo contains 99 containers\n"
    )

    assert explicit_carrier_receipt_container_counts(raw) == (5, 3)


def test_source_template_integrity_rejects_only_a_proven_count_contradiction() -> None:
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "AAAA0000000"},
                {"containerNumber": "BBBB0000000"},
                {"containerNumber": "CCCC0000000"},
            ]
        }
    }

    assert source_template_integrity_issues(
        "CARRIER'S RECEIPT\n5 CNTRS\n", target
    ) == (
        "explicit_carrier_receipt_container_count_differs_from_labeled_containers:5_vs_3",
    )
    assert source_template_integrity_issues(
        "CARRIER'S RECEIPT\n3 CNTRS\n", target
    ) == ()
    assert source_template_integrity_issues("Cargo contains 5 containers\n", target) == ()
