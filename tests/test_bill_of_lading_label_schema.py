from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from document_ocr.label_schemas.bill_of_lading import (
    BillOfLadingAnnotation,
    BillOfLadingExclusion,
    BillOfLadingLabel,
)


def _source_payload(*, pages: int = 1) -> dict[str, Any]:
    return {
        "documentId": f"doc_{'a' * 64}",
        "extractionRunId": "glm-ocr-test",
        "sourceUri": "file:///data/example.pdf",
        "localCanonicalPath": "/data/example.pdf",
        "sourceSha256": "b" * 64,
        "documentPageCount": pages,
        "joinedRawTextSha256": "c" * 64,
        "pages": [
            {
                "pageIndex": index,
                "pageNumber": index + 1,
                "pageId": f"page-{index + 1}",
                "extractionId": f"extract-{index + 1}",
                "rawOcrTextSha256": "d" * 64,
                "rawResponsePath": f"raw-responses/page-{index + 1}.json",
                "rawResponseSha256": "e" * 64,
                "rasterPath": f"page-images/page-{index + 1}.png",
                "rasterSha256": "f" * 64,
            }
            for index in range(pages)
        ],
    }


def _minimal_label() -> dict[str, Any]:
    return {
        "schemaVersion": "2.0.0",
        "documentPatch": {"billOfLadingNumber": "HBL-2026-001"},
    }


def test_canonical_target_is_sparse_and_uses_semantic_names() -> None:
    payload = _minimal_label()
    payload["documentPatch"]["route"] = {
        "portOfLoading": {"name": "BUSAN"},
        "portOfDischarge": None,
    }

    label = BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)

    assert label.canonical_target() == {
        "schemaVersion": "2.0.0",
        "documentPatch": {
            "billOfLadingNumber": "HBL-2026-001",
            "route": {"portOfLoading": {"name": "BUSAN"}},
        },
    }
    target_text = json.dumps(label.canonical_target(), sort_keys=True)
    for structural_name in (
        "partyFunction",
        "locationOfIdentificationQualifier",
        "textSubjectCodeQualifier",
        "measuredAttributeCode",
        "chargeCategory",
    ):
        assert structural_name not in target_text


def test_one_party_address_is_one_logical_scalar_not_ocr_line_rows() -> None:
    payload = _minimal_label()
    payload["documentPatch"]["parties"] = {
        "shipper": {
            "name": "H2 CORPORATION",
            "address": "#501 UH BLDG., 89-4, 48 STREET SEOCHODAERO, SEOCHO-GU, SEOUL, 06650",
        }
    }

    label = BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)

    shipper = label.canonical_target()["documentPatch"]["parties"]["shipper"]
    assert isinstance(shipper["address"], str)
    assert "addresses" not in shipper


@pytest.mark.parametrize(
    "address",
    [
        "11 MAIN STREET TAX ID: 754-375-706",
        "11 MAIN STREET VAT: BE0406804736",
        "11 MAIN STREET VD: 600 042 8579",
        "11 MAIN STREET CIF: B12345678",
        "11 MAIN STREET PH: (48) 3255-1391",
        "11 MAIN STREET user@example.com",
    ],
)
def test_address_rejects_tax_and_contact_contamination(address: str) -> None:
    payload = _minimal_label()
    payload["documentPatch"]["parties"] = {
        "consignee": {"name": "EXAMPLE LTD", "address": address}
    }

    with pytest.raises(ValidationError, match="address contains"):
        BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("description", "SHIPPER'S LOAD & COUNT"),
        ("description", "SAID TO CONTAIN"),
        ("marksAndNumbers", ["SHIPPER'S LOAD,COUNT,SEALED & WEIGHT"]),
    ],
)
def test_goods_fields_reject_carrier_boilerplate(field: str, value: object) -> None:
    payload = _minimal_label()
    payload["documentPatch"]["goodsItems"] = [{field: value}]

    with pytest.raises(ValidationError, match="carrier boilerplate"):
        BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


def test_same_as_relation_replaces_repeated_party_payload() -> None:
    payload = _minimal_label()
    payload["documentPatch"]["parties"] = {
        "consignee": {"name": "EXAMPLE LTD", "address": "11 MAIN STREET"},
        "notifyParties": [{"sameAs": "consignee"}],
    }

    label = BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)
    notify = label.canonical_target()["documentPatch"]["parties"]["notifyParties"][0]
    assert notify == {"sameAs": "consignee"}

    payload["documentPatch"]["parties"]["notifyParties"][0]["name"] = "EXAMPLE LTD"
    with pytest.raises(ValidationError, match="sameAs replaces"):
        BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


def test_contact_details_accepts_a_grounded_name_without_a_communication_channel() -> None:
    payload = _minimal_label()
    payload["documentPatch"]["parties"] = {
        "notifyParties": [{"contactDetails": {"contactName": "WALID MOHAMED"}}]
    }

    label = BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)

    notify = label.canonical_target()["documentPatch"]["parties"]["notifyParties"][0]
    assert notify["contactDetails"] == {"contactName": "WALID MOHAMED"}


def test_contact_details_rejects_an_empty_object() -> None:
    payload = _minimal_label()
    payload["documentPatch"]["parties"] = {
        "notifyParties": [{"contactDetails": {}}]
    }

    with pytest.raises(ValidationError, match="requires a contact name"):
        BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


@pytest.mark.parametrize("forbidden_field", ["countryCode", "unLocode"])
def test_location_rejects_geographic_code_fields(forbidden_field: str) -> None:
    payload = _minimal_label()
    payload["documentPatch"]["route"] = {
        "portOfLoading": {"name": "BUSAN", "country": "KOREA", forbidden_field: "KR"}
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


def test_party_rejects_country_code_and_preserves_printed_abbreviation() -> None:
    payload = _minimal_label()
    payload["documentPatch"]["parties"] = {
        "shipper": {"name": "EXAMPLE LTD", "country": "EG"}
    }

    label = BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)
    assert label.canonical_target()["documentPatch"]["parties"]["shipper"]["country"] == "EG"

    payload["documentPatch"]["parties"]["shipper"]["countryCode"] = "EG"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


@pytest.mark.parametrize(
    "value",
    [
        "MOTA II Soluções Cerâmicas, S. A.",
        "Oiã",
        "ATATÜRK MAH.",
        "Oia\u0303",
        "41 Ø POTS \u2013 2 m³",
    ],
)
def test_application_text_preserves_printable_latin_unicode(value: str) -> None:
    payload = _minimal_label()
    payload["documentPatch"]["parties"] = {"shipper": {"name": value}}

    label = BillOfLadingLabel.model_validate_json(
        json.dumps(payload, ensure_ascii=False), strict=True
    )

    assert label.documentPatch.parties is not None
    assert label.documentPatch.parties.shipper is not None
    assert label.documentPatch.parties.shipper.name == value


@pytest.mark.parametrize("value", ["القاهرة", "上海", "Москва"])
def test_application_text_rejects_non_latin_scripts(value: str) -> None:
    payload = _minimal_label()
    payload["documentPatch"]["parties"] = {"shipper": {"name": value}}

    with pytest.raises(ValidationError, match="Latin script"):
        BillOfLadingLabel.model_validate_json(
            json.dumps(payload, ensure_ascii=False), strict=True
        )


def test_annotation_keeps_raw_value_and_requires_exact_target_evidence() -> None:
    payload = {
        "annotationSchemaVersion": "2.0.0",
        "taskType": "bill_of_lading_kie",
        "documentType": "bill_of_lading",
        "source": _source_payload(),
        "label": {
            "schemaVersion": "2.0.0",
            "documentPatch": {"negotiability": "non_negotiable"},
        },
        "evidence": [
            {
                "targetPath": "documentPatch.negotiability",
                "evidenceKind": "contextual_code",
                "rawOcrEvidence": [
                    {
                        "pageNumber": 1,
                        "rawValue": "NON-NEGOTIABLE COPY",
                        "ocrExcerpt": "NON-NEGOTIABLE COPY\n\nBILL OF LADING",
                    }
                ],
                "normalizationRule": (
                    "Map explicit NON-NEGOTIABLE heading to semantic non_negotiable enum"
                ),
            }
        ],
        "warnings": [],
        "reviewStatus": "candidate",
        "reviewNotes": [],
    }

    annotation = BillOfLadingAnnotation.model_validate_json(json.dumps(payload), strict=True)

    assert annotation.evidence[0].rawOcrEvidence[0].rawValue == "NON-NEGOTIABLE COPY"
    assert annotation.label.documentPatch.negotiability == "non_negotiable"

    payload["evidence"][0]["targetPath"] = "documentPatch.billOfLadingNumber"
    with pytest.raises(ValidationError, match="evidence paths differ"):
        BillOfLadingAnnotation.model_validate_json(json.dumps(payload), strict=True)


def test_multiple_transport_documents_have_a_schema_valid_fail_closed_record() -> None:
    payload = {
        "exclusionSchemaVersion": "2.0.0",
        "taskType": "bill_of_lading_kie",
        "source": _source_payload(pages=2),
        "reason": "multiple_transport_documents",
        "rawOcrEvidence": [
            {
                "pageNumber": 1,
                "rawValue": "PO VIL001450",
                "ocrExcerpt": "Shipper's description of goods\nPO VIL001450",
            },
            {
                "pageNumber": 2,
                "rawValue": "PO VIL001451",
                "ocrExcerpt": "Shipper's description of goods\nPO VIL001451",
            },
        ],
        "reviewStatus": "rejected",
        "reviewNotes": ["Two distinct B/L faces are concatenated in one source PDF."],
    }

    exclusion = BillOfLadingExclusion.model_validate_json(json.dumps(payload), strict=True)

    assert exclusion.reason == "multiple_transport_documents"
