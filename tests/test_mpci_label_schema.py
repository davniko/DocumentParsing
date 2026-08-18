from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from document_ocr.label_schemas.mpci_bill_of_lading import (
    MpciBillOfLadingAnnotation,
    MpciBillOfLadingLabel,
)


def _label_payload() -> dict[str, Any]:
    return {
        "schemaVersion": "1.0.0",
        "documentPatch": {
            "billOfLadingIssueDate": None,
            "blIdentifiers": {
                "houseBLNumber": "HBL-2026-001",
                "originalBLNumber": None,
            },
        },
    }


def _source_payload() -> dict[str, Any]:
    return {
        "documentId": f"doc_{'a' * 64}",
        "extractionRunId": "glm-ocr-blc-pilot150-faa3c717dbc4",
        "sourceUri": "file:///data/2026-01-01_example.pdf",
        "localCanonicalPath": "/data/2026-01-01_example.pdf",
        "sourceSha256": "b" * 64,
        "documentPageCount": 1,
        "joinedRawTextSha256": "c" * 64,
        "pages": [
            {
                "pageIndex": 0,
                "pageNumber": 1,
                "pageId": "page-1",
                "extractionId": "extract-1",
                "rawOcrTextSha256": "d" * 64,
                "rawResponsePath": "raw-responses/page-1.json",
                "rawResponseSha256": "e" * 64,
                "rasterPath": "page-images/page-1.png",
                "rasterSha256": "f" * 64,
            }
        ],
    }


def _annotation_payload() -> dict[str, Any]:
    return {
        "annotationSchemaVersion": "1.0.0",
        "taskType": "mpci_cuscar_kie",
        "documentType": "bill_of_lading",
        "source": _source_payload(),
        "label": _label_payload(),
        "evidence": [
            {
                "targetPath": "documentPatch.blIdentifiers.houseBLNumber",
                "evidenceKind": "verbatim",
                "rawOcrEvidence": [
                    {
                        "pageNumber": 1,
                        "rawValue": "HBL-2026-001",
                        "ocrExcerpt": "BILL OF LADING NO: HBL-2026-001",
                    }
                ],
                "imageUse": "not_used",
            }
        ],
        "warnings": [],
        "reviewStatus": "candidate",
        "reviewNotes": [],
    }


def _schema_leaf_paths(schema: dict[str, Any], root: dict[str, Any], prefix: str = "") -> set[str]:
    definitions = schema.get("$defs", {})

    def walk(node: dict[str, Any], path: str) -> set[str]:
        if "$ref" in node:
            definition_name = node["$ref"].rsplit("/", maxsplit=1)[-1]
            return walk(definitions[definition_name], path)

        variants = node.get("anyOf") or node.get("oneOf")
        if variants is not None:
            paths: set[str] = set()
            for variant in variants:
                if variant.get("type") != "null":
                    paths.update(walk(variant, path))
            return paths

        node_type = node.get("type")
        if isinstance(node_type, list):
            node_type = next((item for item in node_type if item != "null"), None)
        if node_type == "object" or "properties" in node:
            paths = set()
            for name, child in node.get("properties", {}).items():
                child_path = f"{path}.{name}" if path else name
                paths.update(walk(child, child_path))
            return paths
        if node_type == "array":
            return walk(node["items"], f"{path}[]")
        return {path}

    return walk(root, prefix)


def test_optional_nulls_validate_but_canonical_training_target_is_sparse() -> None:
    label = MpciBillOfLadingLabel.model_validate_json(json.dumps(_label_payload()), strict=True)

    assert label.canonical_target() == {
        "schemaVersion": "1.0.0",
        "documentPatch": {"blIdentifiers": {"houseBLNumber": "HBL-2026-001"}},
    }


def test_strict_annotation_validation_uses_json_for_json_only_representations() -> None:
    payload = _annotation_payload()
    payload["label"]["documentPatch"]["billOfLadingIssueDate"] = "2026-01-02"
    payload["evidence"].append(
        {
            "targetPath": "documentPatch.billOfLadingIssueDate",
            "evidenceKind": "verbatim",
            "rawOcrEvidence": [
                {
                    "pageNumber": 1,
                    "rawValue": "2026-01-02",
                    "ocrExcerpt": "ISSUE DATE: 2026-01-02",
                }
            ],
            "imageUse": "not_used",
        }
    )

    annotation = MpciBillOfLadingAnnotation.model_validate_json(json.dumps(payload), strict=True)

    assert annotation.label.documentPatch.billOfLadingIssueDate == date(2026, 1, 2)
    with pytest.raises(ValidationError, match="Input should be a valid tuple"):
        MpciBillOfLadingAnnotation.model_validate(payload)


def test_schema_uses_exact_application_paths_and_excludes_platform_scaffolding() -> None:
    schema = MpciBillOfLadingLabel.model_json_schema()
    patch = schema["$defs"]["MpciBillOfLadingDocumentPatch"]["properties"]

    assert "blIdentifiers" in patch
    assert "containerInformation" in patch
    assert "consignmentInformation" in patch
    assert "beginningOfMessage" not in patch
    assert "testIndicator" not in patch
    assert "updateFiling" not in patch
    assert "placeOfBillIssueFreeTextToggle" not in patch


def test_every_label_leaf_is_an_exact_discovered_application_path() -> None:
    label_schema = MpciBillOfLadingLabel.model_json_schema()
    label_paths = _schema_leaf_paths(
        label_schema,
        label_schema["properties"]["documentPatch"],
    )
    canonical_schema_path = (
        Path(__file__).parents[1]
        / "artifacts"
        / "mpci-ai-schema"
        / "mpci-cuscar-canonical.schema.json"
    )
    canonical_schema = json.loads(canonical_schema_path.read_text(encoding="utf-8"))
    canonical_paths = _schema_leaf_paths(
        canonical_schema,
        canonical_schema["$defs"]["CuscarFormData"],
    )

    assert label_paths
    assert label_paths <= canonical_paths


def test_unknown_label_field_is_rejected() -> None:
    payload = _label_payload()
    payload["documentPatch"]["testIndicator"] = False

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MpciBillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


def test_container_and_goods_relationship_validate() -> None:
    payload = {
        "schemaVersion": "1.0.0",
        "documentPatch": {
            "containerInformation": [
                {
                    "equipmentIdentification": {
                        "equipmentIdentifier": "TGHU1234567",
                    },
                    "equipmentSizeAndType": {
                        "containerSizeAndType": {"containerCode": "40GP"},
                    },
                }
            ],
            "consignmentInformation": {
                "consignmentDetails": {
                    "goodsItemDetails": [
                        {
                            "numberAndTypeOfPackages": [
                                {
                                    "packageQuantity": 20,
                                    "packageTypeDescriptionCode": "CT",
                                }
                            ],
                            "splitGoodsPlacement": [
                                {
                                    "equipmentIdentification": {
                                        "equipmentIdentifier": "TGHU1234567",
                                        "packageQuantity": 20,
                                    }
                                }
                            ],
                        }
                    ]
                }
            },
        },
    }

    label = MpciBillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)
    placement = label.documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[
        0
    ].splitGoodsPlacement[0]
    assert placement.equipmentIdentification.equipmentIdentifier == "TGHU1234567"


def test_container_identity_cannot_carry_goods_placement_quantity() -> None:
    payload = {
        "schemaVersion": "1.0.0",
        "documentPatch": {
            "containerInformation": [
                {
                    "equipmentIdentification": {
                        "equipmentIdentifier": "TGHU1234567",
                        "packageQuantity": 20,
                    }
                }
            ]
        },
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MpciBillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


@pytest.mark.parametrize(
    "identifier",
    [
        "MSCU6639871",  # wrong check digit
        "TGHU123456A",  # nonnumeric serial/check position
        "SYNX1234567",  # invalid equipment category identifier
    ],
)
def test_invalid_iso6346_container_identifier_is_rejected(identifier: str) -> None:
    payload = {
        "schemaVersion": "1.0.0",
        "documentPatch": {
            "containerInformation": [
                {"equipmentIdentification": {"equipmentIdentifier": identifier}}
            ]
        },
    }

    with pytest.raises(ValidationError, match="equipmentIdentifier"):
        MpciBillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


def test_invalid_imo_check_digit_is_rejected() -> None:
    payload = {
        "schemaVersion": "1.0.0",
        "documentPatch": {
            "voyageDetails": {
                "transportInformation": {"transportMeansIdentificationNameIdentifier": "9074728"}
            }
        },
    }

    with pytest.raises(ValidationError, match="IMO check digit"):
        MpciBillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


def test_bill_of_lading_issue_date_can_follow_shipped_on_board_date() -> None:
    payload = {
        "schemaVersion": "1.0.0",
        "documentPatch": {
            "billOfLadingIssueDate": "2025-12-30",
            "shippedOnBoardDate": "2025-12-15",
        },
    }

    label = MpciBillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)

    assert label.canonical_target()["documentPatch"] == {
        "billOfLadingIssueDate": "2025-12-30",
        "shippedOnBoardDate": "2025-12-15",
    }


def test_voyage_departure_cannot_follow_estimated_arrival() -> None:
    payload = {
        "schemaVersion": "1.0.0",
        "documentPatch": {
            "voyageDetails": {
                "timeOfArrivalAndDepartures": {
                    "estimatedTimeOfArrival": "2026-01-15T10:00:00Z",
                    "estimatedTimeOfDeparture": "2026-01-15T11:00:00Z",
                }
            }
        },
    }

    with pytest.raises(ValidationError, match="estimatedTimeOfDeparture"):
        MpciBillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


def test_monetary_type_code_is_bounded() -> None:
    payload = {
        "schemaVersion": "1.0.0",
        "documentPatch": {
            "consignmentInformation": {
                "consignmentDetails": {
                    "monetaryAmount": [
                        {
                            "typeCodeQualifier": "551",
                            "amount": 1.0,
                            "currencyIdentificationCode": "USD",
                        }
                    ]
                }
            }
        },
    }

    with pytest.raises(ValidationError, match="typeCodeQualifier"):
        MpciBillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


def test_orphan_goods_placement_is_rejected() -> None:
    payload = {
        "schemaVersion": "1.0.0",
        "documentPatch": {
            "blIdentifiers": {"houseBLNumber": "HBL-1"},
            "consignmentInformation": {
                "consignmentDetails": {
                    "goodsItemDetails": [
                        {
                            "splitGoodsPlacement": [
                                {"equipmentIdentification": {"equipmentIdentifier": "TGHU1234567"}}
                            ]
                        }
                    ]
                }
            },
        },
    }

    with pytest.raises(ValidationError, match="absent from containerInformation"):
        MpciBillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


def test_consignment_locations_require_canonical_qualifier_order() -> None:
    payload = {
        "schemaVersion": "1.0.0",
        "documentPatch": {
            "consignmentInformation": {
                "consignmentDetails": {
                    "locationOfIdentification": [
                        {
                            "locationOfIdentificationQualifier": "88",
                            "locationName": "JEBEL ALI",
                        },
                        {
                            "locationOfIdentificationQualifier": "9",
                            "locationName": "SHANGHAI",
                        },
                    ]
                }
            }
        },
    }

    with pytest.raises(ValidationError, match="canonical qualifier order"):
        MpciBillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


def test_payment_arrangement_requires_charge_category() -> None:
    payload = {
        "schemaVersion": "1.0.0",
        "documentPatch": {
            "consignmentInformation": {
                "consignmentDetails": {"chargePaymentInstructions": [{"paymentArrangement": "P"}]}
            }
        },
    }

    with pytest.raises(ValidationError, match="chargeCategory"):
        MpciBillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)


def test_annotation_requires_one_ocr_evidence_record_per_emitted_leaf() -> None:
    annotation = MpciBillOfLadingAnnotation.model_validate_json(
        json.dumps(_annotation_payload()), strict=True
    )
    assert annotation.reviewStatus == "candidate"

    missing = _annotation_payload()
    missing["evidence"] = []
    with pytest.raises(ValidationError, match="evidence"):
        MpciBillOfLadingAnnotation.model_validate_json(json.dumps(missing), strict=True)

    wrong = _annotation_payload()
    wrong["evidence"][0]["targetPath"] = "documentPatch.blIdentifiers.parentBLNumber"
    with pytest.raises(ValidationError, match="evidence paths differ"):
        MpciBillOfLadingAnnotation.model_validate_json(json.dumps(wrong), strict=True)


def test_annotation_keeps_exact_pre_mapping_value_with_mapped_label() -> None:
    payload = _annotation_payload()
    payload["label"]["documentPatch"] = {
        "processingInformation": {"processingIndicatorDescriptionCode": "NEG"}
    }
    payload["evidence"] = [
        {
            "targetPath": (
                "documentPatch.processingInformation.processingIndicatorDescriptionCode"
            ),
            "evidenceKind": "contextual_code",
            "rawOcrEvidence": [
                {
                    "pageNumber": 1,
                    "rawValue": "TO ORDER",
                    "ocrExcerpt": "CONSIGNEE: TO ORDER",
                }
            ],
            "imageUse": "not_used",
            "normalizationRule": "Mapped explicit TO ORDER text to negotiability code NEG.",
        }
    ]

    annotation = MpciBillOfLadingAnnotation.model_validate_json(json.dumps(payload), strict=True)

    assert (
        annotation.label.documentPatch.processingInformation.processingIndicatorDescriptionCode
        == "NEG"
    )
    assert annotation.evidence[0].rawOcrEvidence[0].rawValue == "TO ORDER"


def test_raw_ocr_value_must_occur_verbatim_in_its_excerpt() -> None:
    payload = _annotation_payload()
    payload["evidence"][0]["rawOcrEvidence"][0]["rawValue"] = "HBL-OTHER"

    with pytest.raises(ValidationError, match="rawValue must occur verbatim"):
        MpciBillOfLadingAnnotation.model_validate_json(json.dumps(payload), strict=True)


def test_raw_ocr_evidence_page_must_belong_to_source_document() -> None:
    payload = _annotation_payload()
    payload["evidence"][0]["rawOcrEvidence"][0]["pageNumber"] = 2

    with pytest.raises(ValidationError, match="outside the source document"):
        MpciBillOfLadingAnnotation.model_validate_json(json.dumps(payload), strict=True)


def test_annotation_source_pages_must_be_complete_and_ordered() -> None:
    payload = _annotation_payload()
    payload["source"]["documentPageCount"] = 2

    with pytest.raises(ValidationError, match="pages must be complete and ordered"):
        MpciBillOfLadingAnnotation.model_validate_json(json.dumps(payload), strict=True)


def test_annotation_source_extraction_ids_must_be_unique() -> None:
    payload = _annotation_payload()
    payload["source"]["documentPageCount"] = 2
    second_page = dict(payload["source"]["pages"][0])
    second_page.update(
        {
            "pageIndex": 1,
            "pageNumber": 2,
            "pageId": "page-2",
            "rawOcrTextSha256": "1" * 64,
            "rawResponsePath": "raw-responses/page-2.json",
            "rawResponseSha256": "2" * 64,
            "rasterPath": "page-images/page-2.png",
            "rasterSha256": "3" * 64,
        }
    )
    payload["source"]["pages"].append(second_page)

    with pytest.raises(ValidationError, match="extractionId values must be unique"):
        MpciBillOfLadingAnnotation.model_validate_json(json.dumps(payload), strict=True)


def test_non_verbatim_evidence_requires_an_explicit_normalization_rule() -> None:
    payload = _annotation_payload()
    payload["evidence"][0]["evidenceKind"] = "contextual_code"

    with pytest.raises(ValidationError, match="normalizationRule"):
        MpciBillOfLadingAnnotation.model_validate_json(json.dumps(payload), strict=True)
