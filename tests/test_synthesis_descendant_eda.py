from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.template_compiler.descendant_eda import (
    DescendantEdaConfig,
    ManualReview,
    _change_family,
    _exporter_country_surfaces,
    _number_contradictions,
)


def _pin(prefix: str) -> dict[str, str]:
    return {
        "path": f"artifacts/{prefix}",
        "commit_sha256": "a" * 64,
        "transaction_sha256": "b" * 64,
    }


def _review_entry(document_id: str = "doc_a") -> dict[str, object]:
    return {
        "document_id": document_id,
        "target_origin": "controlled_source_variant",
        "compilation_lineage": "compilation-200",
        "selection_reason": "stress case",
        "mechanical_fidelity": "pass",
        "target_binding_fidelity": "pass",
        "carrier_preservation": "pass",
        "layout_preservation": "pass",
        "whole_document_coherence": "concern",
        "quality_decision": "review_required",
        "primary_surface": "customs identity",
        "evidence_lines": [1, 4],
        "observation": "The related surfaces disagree.",
    }


def test_eda_contract_is_strict_and_accepts_pinned_inputs() -> None:
    config = DescendantEdaConfig.model_validate_json(
        json.dumps(
            {
                "schema_version": 1,
                "task": "bill_of_lading_compiled_descendant_eda_v1",
                "run_name": "eda-150",
                "output_dir": "artifacts/kie-synthesis",
                "descendant_run": _pin("descendant"),
                "template_catalog": _pin("catalog"),
                "iso3166_snapshot": {"path": "iso.json", "sha256": "c" * 64},
                "manual_review": {"path": "review.yaml", "sha256": "d" * 64},
                "expected_documents": 150,
                "expected_manual_reviews": 10,
            }
        )
    )

    assert config.expected_documents == 150


def test_manual_review_rejects_duplicate_documents() -> None:
    payload = {
        "schema_version": 1,
        "run": _pin("descendant"),
        "template_catalog": _pin("catalog"),
        "selection_method": "purposive stress sample",
        "reviews": [_review_entry(), _review_entry()],
    }

    with pytest.raises(ValidationError, match="repeats a document"):
        ManualReview.model_validate_json(json.dumps(payload))


def test_manual_review_requires_review_decision_for_coherence_concern() -> None:
    entry = _review_entry()
    entry["quality_decision"] = "pass"
    payload = {
        "schema_version": 1,
        "run": _pin("descendant"),
        "template_catalog": _pin("catalog"),
        "selection_method": "purposive stress sample",
        "reviews": [entry],
    }

    with pytest.raises(ValidationError, match="must be review_required"):
        ManualReview.model_validate_json(json.dumps(payload))


def test_number_contradiction_screen_handles_words_and_sequences() -> None:
    text = "\n".join(
        (
            "SAY THIRTY NINE (39) PACKAGES ONLY",
            "ZERO (2)",
            "1 Of Three",
            "8 Of Three",
        )
    )

    rows = _number_contradictions("doc_a", text)

    assert [(row["line_number"], row["kind"]) for row in rows] == [
        (2, "words_parenthetical_digits_disagree"),
        (4, "sequence_exceeds_total"),
    ]


def test_exporter_country_surface_supports_inline_and_split_layouts() -> None:
    text = "\n".join(
        (
            "FOREIGN EXPORTER COUNTRY: BRAZIL",
            "FOREIGN EXPORTER COUNTRY:",
            "NEW ZEALAND",
            "FOREIGN EXPORTER",
            "COUNTRY: CANADA",
            "FOREIGN EXPORTER COUNTRY CODE: CA",
        )
    )

    assert _exporter_country_surfaces(text) == ("BRAZIL", "NEW ZEALAND", "CANADA")


def test_exporter_country_surface_keeps_country_when_code_shares_ocr_line() -> None:
    text = (
        "FOREIGN EXPORTER COUNTRY: CANADA Foreign Exporter Country Code: CA "
        "FREIGHT PREPAID"
    )

    assert _exporter_country_surfaces(text) == (
        "CANADA Foreign Exporter Country Code: CA FREIGHT PREPAID",
    )


@pytest.mark.parametrize(
    ("path", "family"),
    (
        ("documentPatch.parties.shipper.country", "parties"),
        ("documentPatch.cargoPackages[0].quantity", "packages"),
        ("documentPatch.containers[0].containerNumber", "equipment"),
        ("documentPatch.billOfLadingNumber", "document"),
    ),
)
def test_change_family_is_stable(path: str, family: str) -> None:
    assert _change_family(cast(str, path)) == family
