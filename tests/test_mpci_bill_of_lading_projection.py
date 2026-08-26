from __future__ import annotations

import json

import pytest

from document_ocr.label_schemas.bill_of_lading import BillOfLadingLabel
from document_ocr.label_schemas.mpci_projection import (
    MpciProjectionError,
    project_bill_of_lading_to_mpci,
)


def _semantic_payload() -> dict[str, object]:
    return {
        "schemaVersion": "2.0.0",
        "documentPatch": {
            "billOfLadingNumber": "HBL-001",
            "issueDate": "2026-01-23",
            "shippedOnBoardDate": "2026-01-23",
            "negotiability": "non_negotiable",
            "placeOfIssue": {"name": "SEOUL", "country": "KOREA"},
            "route": {
                "placeOfReceipt": {"name": "BUSAN", "country": "KOREA"},
                "portOfLoading": {"name": "BUSAN", "country": "KOREA"},
                "portOfDischarge": {"name": "ROTTERDAM", "country": "NETHERLANDS"},
                "placeOfDelivery": {"name": "ROTTERDAM", "country": "NETHERLANDS"},
            },
            "transport": {"vesselName": "HMM GARAM", "voyageNumber": "0007W"},
            "freight": {
                "paymentArrangement": "prepaid",
                "paymentPlace": {"name": "SEOUL", "country": "KOREA"},
            },
            "parties": {
                "shipper": {
                    "name": "H2 CORPORATION",
                    "address": "#501 UH BLDG., 89-4, 48 STREET SEOUL, 06650",
                    "country": "KOREA",
                },
                "consignee": {
                    "name": "HSBC BANK PLC",
                    "address": "8 CANADA SQUARE, LONDON E14 5HQ, U.K",
                    "country": "U.K",
                    "contactDetails": {
                        "contactName": "CHIUDO OJIKE",
                        "emailAddresses": ["CHIUDO.OJIKE@HSBC.COM"],
                    },
                },
                "notifyParties": [{"sameAs": "consignee"}],
                "carrier": {"name": "HMM CO., LTD."},
            },
            "containers": [
                {
                    "containerNumber": "TGHU1234567",
                    "typeDescription": "20' DC",
                    "sealNumbers": ["212776446"],
                }
            ],
            "goodsItems": [
                {
                    "description": "FERRO MOLYBDENUM",
                    "packages": [
                        {"quantity": 60, "type": "PALLETS"},
                        {"quantity": 60, "type": "BAGS"},
                    ],
                    "grossWeight": {"value": 60960.0, "unit": "kilogram"},
                    "netWeight": {"value": 60000.0, "unit": "kilogram"},
                    "containerAllocations": [
                        {"containerNumber": "TGHU1234567", "packageQuantity": 20}
                    ],
                    "marksAndNumbers": ["N/M"],
                    "hsCodes": ["72027000"],
                }
            ],
        },
    }


def _country_resolver(value: str) -> str | None:
    return {"KOREA": "KR", "NETHERLANDS": "NL", "U.K": "GB"}.get(value)


def test_projection_preserves_a_contact_name_without_a_communication_channel() -> None:
    payload = _semantic_payload()
    parties = payload["documentPatch"]["parties"]
    parties["notifyParties"] = [
        {"name": "NOTIFY LIMITED", "contactDetails": {"contactName": "WALID MOHAMED"}}
    ]
    semantic = BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)

    projected = project_bill_of_lading_to_mpci(
        semantic, country_resolver=_country_resolver
    ).canonical_target()["documentPatch"]

    notify = projected["consignmentInformation"]["consignmentDetails"]["partiesInformation"][2]
    assert notify["contactInformations"] == [
        {
            "contactInformation": {
                "contactIdentifier": "COM",
                "contactName": "WALID MOHAMED",
            },
            "communicationContact": [],
        }
    ]


def test_projection_merges_same_as_identity_with_notify_contact_override() -> None:
    payload = _semantic_payload()
    consignee = payload["documentPatch"]["parties"]["consignee"]
    consignee["city"] = "LONDON"
    payload["documentPatch"]["parties"]["notifyParties"] = [
        {
            "sameAs": "consignee",
            "contactDetails": {
                "contactName": "NOTIFY DESK",
                "phoneNumbers": ["+44 20 7123 4567"],
            },
        }
    ]
    semantic = BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)

    projected = project_bill_of_lading_to_mpci(
        semantic, country_resolver=_country_resolver
    ).canonical_target()["documentPatch"]

    notify = projected["consignmentInformation"]["consignmentDetails"][
        "partiesInformation"
    ][2]
    assert notify == {
        "partyFunction": "NI",
        "partyNames": [{"name": "HSBC BANK PLC"}],
        "addresses": [{"address": "8 CANADA SQUARE, LONDON E14 5HQ, U.K"}],
        "city": "LONDON",
        "country": "GB",
        "contactInformations": [
            {
                "contactInformation": {
                    "contactIdentifier": "COM",
                    "contactName": "NOTIFY DESK",
                },
                "communicationContact": [
                    {"communicationMeans": "TE", "identifier": "+44 20 7123 4567"}
                ],
            }
        ],
    }


def test_projection_adds_only_deterministic_mpci_structure_and_codes() -> None:
    semantic = BillOfLadingLabel.model_validate_json(json.dumps(_semantic_payload()), strict=True)

    projected = project_bill_of_lading_to_mpci(
        semantic, country_resolver=_country_resolver
    ).canonical_target()["documentPatch"]

    assert projected["processingInformation"] == {"processingIndicatorDescriptionCode": "NON"}
    details = projected["consignmentInformation"]["consignmentDetails"]
    assert details["chargePaymentInstructions"] == [
        {"chargeCategory": "4", "paymentArrangement": "P"}
    ]
    assert [party["partyFunction"] for party in details["partiesInformation"]] == [
        "CZ",
        "CN",
        "NI",
        "CG",
    ]
    assert details["partiesInformation"][0]["addresses"] == [
        {"address": "#501 UH BLDG., 89-4, 48 STREET SEOUL, 06650"}
    ]
    assert details["partiesInformation"][2]["partyNames"] == [{"name": "HSBC BANK PLC"}]
    assert projected["voyageDetails"]["departureAndArrivalPorts"]["portOfDeparture"] == {
        "name": "BUSAN",
        "country_code": "KR",
    }
    loading_rows = [
        row
        for row in details["locationOfIdentification"]
        if row["locationOfIdentificationQualifier"] == "9"
    ]
    assert loading_rows == [
        {
            "locationOfIdentificationQualifier": "9",
            "locationIdentifier": {"name": "BUSAN", "country_code": "KR"},
        }
    ]
    goods = details["goodsItemDetails"][0]
    assert goods["freeText"] == [
        {"textSubjectCodeQualifier": "AAA", "freeText": "FERRO MOLYBDENUM"}
    ]
    assert goods["measurements"] == [
        {"measuredAttributeCode": "AAB", "measurementUnitCode": "KGM", "measure": 60960.0},
        {"measuredAttributeCode": "AAA", "measurementUnitCode": "KGM", "measure": 60000.0},
    ]


def test_projection_converts_metric_tonne_cargo_mass_to_kilograms() -> None:
    payload = _semantic_payload()
    goods = payload["documentPatch"]["goodsItems"][0]
    goods["grossWeight"] = {"value": 7.1234, "unit": "metric_tonne"}
    goods["netWeight"] = {"value": 60.0, "unit": "metric_tonne"}
    semantic = BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)

    projected = project_bill_of_lading_to_mpci(
        semantic, country_resolver=_country_resolver
    ).canonical_target()["documentPatch"]

    assert semantic.canonical_target()["documentPatch"]["goodsItems"][0][
        "grossWeight"
    ] == {"value": 7.1234, "unit": "metric_tonne"}
    projected_goods = projected["consignmentInformation"]["consignmentDetails"][
        "goodsItemDetails"
    ][0]
    assert projected_goods["measurements"] == [
        {"measuredAttributeCode": "AAB", "measurementUnitCode": "KGM", "measure": 7123.4},
        {"measuredAttributeCode": "AAA", "measurementUnitCode": "KGM", "measure": 60000.0},
    ]


def test_projection_never_guesses_unresolved_country_text() -> None:
    semantic = BillOfLadingLabel.model_validate_json(json.dumps(_semantic_payload()), strict=True)

    with pytest.raises(MpciProjectionError, match="requires the configured country resolver"):
        project_bill_of_lading_to_mpci(semantic)

    with pytest.raises(MpciProjectionError, match="did not resolve unambiguously"):
        project_bill_of_lading_to_mpci(semantic, country_resolver=lambda _value: None)


def test_projection_transliterates_latin_semantic_text_without_changing_label() -> None:
    payload = _semantic_payload()
    shipper = payload["documentPatch"]["parties"]["shipper"]
    shipper["name"] = "YOUNÉXA ESPAÑA"
    shipper["address"] = "VALL D'ALBA, N.º 2447, CASTELLÓN \u2013 ESPAÑA"
    shipper["country"] = "ESPAÑA"
    semantic = BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)

    projected = project_bill_of_lading_to_mpci(
        semantic,
        country_resolver=lambda value: "ES" if value == "ESPAÑA" else _country_resolver(value),
    ).canonical_target()["documentPatch"]

    projected_shipper = projected["consignmentInformation"]["consignmentDetails"][
        "partiesInformation"
    ][0]
    assert projected_shipper["partyNames"] == [{"name": "YOUNEXA ESPANA"}]
    assert projected_shipper["addresses"] == [
        {"address": "VALL D'ALBA, N.o 2447, CASTELLON - ESPANA"}
    ]
    assert semantic.documentPatch.parties.shipper.name == "YOUNÉXA ESPAÑA"
    assert semantic.documentPatch.parties.shipper.address == (
        "VALL D'ALBA, N.º 2447, CASTELLÓN \u2013 ESPAÑA"
    )


def test_projection_maps_registered_sign_without_changing_label() -> None:
    payload = _semantic_payload()
    payload["documentPatch"]["goodsItems"][0]["description"] = "SABIC® MDI 2031"
    semantic = BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)

    projected = project_bill_of_lading_to_mpci(
        semantic, country_resolver=_country_resolver
    ).canonical_target()["documentPatch"]

    goods = projected["consignmentInformation"]["consignmentDetails"]["goodsItemDetails"][0]
    assert goods["freeText"] == [
        {"textSubjectCodeQualifier": "AAA", "freeText": "SABIC(R) MDI 2031"}
    ]
    assert semantic.documentPatch.goodsItems[0].description == "SABIC® MDI 2031"


def test_projection_rejects_unmapped_text_instead_of_dropping_it() -> None:
    payload = _semantic_payload()
    payload["documentPatch"]["parties"]["shipper"]["name"] = "LATIN ƿ"
    semantic = BillOfLadingLabel.model_validate_json(json.dumps(payload), strict=True)

    with pytest.raises(MpciProjectionError, match="no deterministic CUSCAR ASCII mapping"):
        project_bill_of_lading_to_mpci(semantic, country_resolver=_country_resolver)
