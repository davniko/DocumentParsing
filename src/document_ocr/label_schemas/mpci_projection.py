"""Deterministic semantic-B/L to sparse MPCI/CUSCAR projection.

The projection is intentionally boring: named semantic fields select fixed
MPCI paths and constants.  It never guesses country codes, container codes, or
relationships.  A caller-owned country resolver must resolve readable country
text, and unresolved values fail the projection.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from document_ocr.label_schemas.bill_of_lading import (
    BillOfLadingDocumentPatch,
    BillOfLadingGoodsItem,
    BillOfLadingLabel,
    BillOfLadingParties,
    DangerousGoods,
    Mass,
    SemanticLocation,
    SemanticParty,
)
from document_ocr.label_schemas.mpci_bill_of_lading import MpciBillOfLadingLabel

CountryResolver = Callable[[str], str | None]

_PAYMENT_CODES = {
    "prepaid": "P",
    "collect": "C",
    "third_party": "B",
    "payable_elsewhere": "A",
}
_MASS_UNIT_CODES = {"kilogram": "KGM", "pound": "LBR"}
_TEMPERATURE_UNIT_CODES = {"celsius": "CEL", "fahrenheit": "FAH"}
_PACKING_GROUP_CODES = {"I": "1", "II": "2", "III": "3"}


class MpciProjectionError(ValueError):
    """A semantic fact cannot be projected without guessing or data loss."""


def _resolved_country(
    *,
    country: str | None,
    path: str,
    resolver: CountryResolver | None,
) -> str | None:
    if country is None:
        return None
    if resolver is None:
        raise MpciProjectionError(f"{path} requires the configured country resolver")
    resolved = resolver(country)
    if resolved is None:
        raise MpciProjectionError(f"{path} country {country!r} did not resolve unambiguously")
    if len(resolved) != 2 or not resolved.isascii() or not resolved.isalpha():
        raise MpciProjectionError(f"{path} resolver returned invalid ISO-2 code {resolved!r}")
    return resolved.upper()


def _location(
    location: SemanticLocation,
    *,
    path: str,
    country_resolver: CountryResolver | None,
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    if location.name is not None:
        value["name"] = location.name
    country_code = _resolved_country(
        country=location.country,
        path=path,
        resolver=country_resolver,
    )
    if country_code is not None:
        value["country_code"] = country_code
    return value


def _contact_information(party: SemanticParty) -> list[dict[str, Any]] | None:
    contact = party.contactDetails
    if contact is None:
        return None
    communication: list[dict[str, str]] = []
    communication.extend(
        {"communicationMeans": "TE", "identifier": value}
        for value in contact.phoneNumbers or ()
    )
    communication.extend(
        {"communicationMeans": "EM", "identifier": value}
        for value in contact.emailAddresses or ()
    )
    communication.extend(
        {"communicationMeans": "AO", "identifier": value}
        for value in contact.websiteUrls or ()
    )
    return [
        {
            "contactInformation": {
                "contactIdentifier": "COM",
                **(
                    {"contactName": contact.contactName}
                    if contact.contactName is not None
                    else {}
                ),
            },
            "communicationContact": communication,
        }
    ]


def _party(
    party: SemanticParty,
    *,
    role: str,
    path: str,
    country_resolver: CountryResolver | None,
) -> dict[str, Any]:
    if party.sameAs is not None:
        raise MpciProjectionError(f"{path} must be resolved before party projection")
    result: dict[str, Any] = {"partyFunction": role}
    if party.name is not None:
        result["partyNames"] = [{"name": party.name}]
    if party.address is not None:
        result["addresses"] = [{"address": party.address}]
    if party.city is not None:
        result["city"] = party.city
    country_code = _resolved_country(
        country=party.country,
        path=f"{path}.country",
        resolver=country_resolver,
    )
    if country_code is not None:
        result["country"] = country_code
    contacts = _contact_information(party)
    if contacts is not None:
        result["contactInformations"] = contacts
    return result


def _parties(
    parties: BillOfLadingParties,
    *,
    country_resolver: CountryResolver | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    references = {"shipper": parties.shipper, "consignee": parties.consignee}

    for name, role in (("shipper", "CZ"), ("consignee", "CN")):
        party = getattr(parties, name)
        if party is not None:
            result.append(
                _party(
                    party,
                    role=role,
                    path=f"documentPatch.parties.{name}",
                    country_resolver=country_resolver,
                )
            )

    for index, party in enumerate(parties.notifyParties or ()):
        resolved = references[party.sameAs] if party.sameAs is not None else party
        if resolved is None:
            raise MpciProjectionError(f"notify party {index} has an unresolved sameAs relation")
        result.append(
            _party(
                resolved,
                role="NI" if index == 0 else "N2",
                path=f"documentPatch.parties.notifyParties[{index}]",
                country_resolver=country_resolver,
            )
        )

    for name, role in (
        ("carrier", "CG"),
        ("forwardingAgent", "DDR"),
        ("deliveryAgent", "DP"),
        ("consolidator", "COX"),
    ):
        party = getattr(parties, name)
        if party is not None:
            result.append(
                _party(
                    party,
                    role=role,
                    path=f"documentPatch.parties.{name}",
                    country_resolver=country_resolver,
                )
            )
    return result


def _mass(mass: Mass, attribute: str) -> dict[str, Any]:
    return {
        "measuredAttributeCode": attribute,
        "measurementUnitCode": _MASS_UNIT_CODES[mass.unit],
        "measure": mass.value,
    }


def _dangerous_goods(value: DangerousGoods) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if value.hazardClass is not None:
        hazard: dict[str, Any] = {"hazardIdentificationCode": value.hazardClass}
        if value.subsidiaryHazard is not None:
            hazard["additionalHazardClassificationIdentifier"] = value.subsidiaryHazard
        result["hazardCode"] = [hazard]
    if value.unNumber is not None:
        result["undgInformation"] = {"identifier": value.unNumber}
    if value.flashPoint is not None:
        flashpoint: dict[str, Any] = {
            "shipmentFlashpointDegree": value.flashPoint.temperature.value,
            "measurementUnitCode": _TEMPERATURE_UNIT_CODES[value.flashPoint.temperature.unit],
        }
        if value.flashPoint.packingGroup is not None:
            flashpoint["packagingDangerLevelCode"] = _PACKING_GROUP_CODES[
                value.flashPoint.packingGroup
            ]
        result["dangerousGoodsShipmentFlashpoint"] = [flashpoint]
    return result


def _goods_item(item: BillOfLadingGoodsItem) -> dict[str, Any]:
    result: dict[str, Any] = {}
    free_text: list[dict[str, str]] = []
    if item.description is not None:
        free_text.append({"textSubjectCodeQualifier": "AAA", "freeText": item.description})
    free_text.extend(
        {"textSubjectCodeQualifier": "AAI", "freeText": value}
        for value in item.additionalInformation or ()
    )
    if free_text:
        result["freeText"] = free_text

    if item.packages is not None:
        result["numberAndTypeOfPackages"] = [
            {
                **({"packageQuantity": package.quantity} if package.quantity is not None else {}),
                **({"typeOfPackages": package.type} if package.type is not None else {}),
                **(
                    {"packageTypeDescriptionCode": package.typeCode}
                    if package.typeCode is not None
                    else {}
                ),
            }
            for package in item.packages
        ]

    measurements: list[dict[str, Any]] = []
    if item.grossWeight is not None:
        measurements.append(_mass(item.grossWeight, "AAB"))
    if item.netWeight is not None:
        measurements.append(_mass(item.netWeight, "AAA"))
    if item.volume is not None:
        measurements.append(
            {
                "measuredAttributeCode": "ABJ",
                "measurementUnitCode": "MTQ",
                "measure": item.volume.value,
            }
        )
    if measurements:
        result["measurements"] = measurements

    if item.containerAllocations is not None:
        result["splitGoodsPlacement"] = [
            {
                "equipmentIdentification": {
                    "equipmentIdentifier": allocation.containerNumber,
                    **(
                        {"packageQuantity": allocation.packageQuantity}
                        if allocation.packageQuantity is not None
                        else {}
                    ),
                }
            }
            for allocation in item.containerAllocations
        ]

    if item.marksAndNumbers is not None:
        result["packageIdentification"] = [
            {
                "marksAndLabels": [
                    {"shippingMarksDescription": value} for value in item.marksAndNumbers
                ]
            }
        ]
    if item.hsCodes is not None:
        result["customsStatusOfGoods"] = {
            "customsIdentityCodes": {
                "customsGoodsIdentifier": [{"value": value} for value in item.hsCodes]
            }
        }
    if item.handlingInstructions is not None:
        result["handlingInstructions"] = [
            {"handlingInstructionDescription": value} for value in item.handlingInstructions
        ]
    if item.dangerousGoods is not None:
        result["dangerousGoods"] = [_dangerous_goods(value) for value in item.dangerousGoods]
    if item.origin is not None:
        identifier = item.origin.identifier or item.origin.name
        if identifier is None:
            raise MpciProjectionError("goods origin cannot be projected without an identifier/name")
        result["locationOfIdentification"] = [
            {
                "locationOfIdentificationQualifier": "27",
                "locationIdentifier": identifier,
                **({"locationName": item.origin.name} if item.origin.name is not None else {}),
            }
        ]
    return result


def _document_patch(
    patch: BillOfLadingDocumentPatch,
    *,
    country_resolver: CountryResolver | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    identifiers = {
        key: value
        for key, value in (
            ("houseBLNumber", patch.billOfLadingNumber),
            ("originalBLNumber", patch.originalBillOfLadingNumber),
            ("parentBLNumber", patch.masterBillOfLadingNumber),
        )
        if value is not None
    }
    if identifiers:
        result["blIdentifiers"] = identifiers
    if patch.issueDate is not None:
        result["billOfLadingIssueDate"] = patch.issueDate.isoformat()
    if patch.shippedOnBoardDate is not None:
        result["shippedOnBoardDate"] = patch.shippedOnBoardDate.isoformat()
    if patch.negotiability is not None:
        result["processingInformation"] = {
            "processingIndicatorDescriptionCode": (
                "NEG" if patch.negotiability == "negotiable" else "NON"
            )
        }
    if patch.placeOfIssue is not None:
        result["placeOfBillIssue"] = _location(
            patch.placeOfIssue,
            path="documentPatch.placeOfIssue",
            country_resolver=country_resolver,
        )
    if patch.freight is not None and patch.freight.paymentPlace is not None:
        result["placeOfFreightPayment"] = _location(
            patch.freight.paymentPlace,
            path="documentPatch.freight.paymentPlace",
            country_resolver=country_resolver,
        )

    transport_information: dict[str, Any] = {}
    if patch.transport is not None:
        if patch.transport.voyageNumber is not None:
            transport_information["meansOfTransportJourneyIdentifier"] = (
                patch.transport.voyageNumber
            )
        if patch.transport.vesselImoNumber is not None:
            transport_information["transportMeansIdentificationNameIdentifier"] = (
                patch.transport.vesselImoNumber
            )
        if patch.transport.vesselName is not None:
            transport_information["transportMeansIdentificationName"] = patch.transport.vesselName
        if patch.transport.vesselFlagCountry is not None:
            transport_information["transportMeansNationalityCode"] = (
                _resolved_country(
                    country=patch.transport.vesselFlagCountry,
                    path="documentPatch.transport.vesselFlagCountry",
                    resolver=country_resolver,
                )
            )
    if patch.parties is not None and patch.parties.carrier is not None:
        carrier = patch.parties.carrier
        if carrier.name is not None:
            transport_information["carrierIdentifierFreeText"] = carrier.name

    voyage: dict[str, Any] = {}
    if transport_information:
        voyage["transportInformation"] = transport_information
    if patch.route is not None:
        ports: dict[str, Any] = {}
        if patch.route.portOfLoading is not None:
            ports["portOfDeparture"] = _location(
                patch.route.portOfLoading,
                path="documentPatch.route.portOfLoading",
                country_resolver=country_resolver,
            )
        if patch.route.portOfDischarge is not None:
            ports["portOfArrival"] = _location(
                patch.route.portOfDischarge,
                path="documentPatch.route.portOfDischarge",
                country_resolver=country_resolver,
            )
        if ports:
            voyage["departureAndArrivalPorts"] = ports
    if voyage:
        result["voyageDetails"] = voyage

    if patch.containers is not None:
        containers: list[dict[str, Any]] = []
        for container in patch.containers:
            target: dict[str, Any] = {
                "equipmentIdentification": {
                    "equipmentIdentifier": container.containerNumber
                }
            }
            equipment: dict[str, Any] = {}
            if container.typeCode is not None:
                equipment["containerSizeAndType"] = {"containerCode": container.typeCode}
            if container.typeDescription is not None:
                equipment["equipmentDescription"] = container.typeDescription
            if equipment:
                target["equipmentSizeAndType"] = equipment
            if container.verifiedGrossMass is not None:
                target["containerVerifiedGrossMass"] = {
                    "measure": container.verifiedGrossMass.value,
                    "measurementUnitCode": _MASS_UNIT_CODES[
                        container.verifiedGrossMass.unit
                    ],
                }
            if container.sealNumbers is not None:
                target["sealNumbers"] = [
                    {"transportUnitSealIdentifier": seal} for seal in container.sealNumbers
                ]
            if container.temperatureSetpoint is not None:
                target["temperatureSettings"] = [
                    {
                        "temperatureTypeCodeQualifier": "2",
                        "temperatureDegree": container.temperatureSetpoint.value,
                        "unitCode": _TEMPERATURE_UNIT_CODES[
                            container.temperatureSetpoint.unit
                        ],
                    }
                ]
            containers.append(target)
        result["containerInformation"] = containers

    details: dict[str, Any] = {}
    if patch.route is not None:
        locations: list[dict[str, Any]] = []
        for name, qualifier in (
            ("portOfLoading", "9"),
            ("portOfDischarge", "12"),
            ("transshipmentPort", "13"),
            ("placeOfReceipt", "88"),
            ("placeOfDelivery", "7"),
            ("finalDestination", "96"),
        ):
            location = getattr(patch.route, name)
            if location is not None:
                locations.append(
                    {
                        "locationOfIdentificationQualifier": qualifier,
                        "locationIdentifier": _location(
                            location,
                            path=f"documentPatch.route.{name}",
                            country_resolver=country_resolver,
                        ),
                    }
                )
        if locations:
            details["locationOfIdentification"] = locations
    if patch.freight is not None and patch.freight.paymentArrangement is not None:
        details["chargePaymentInstructions"] = [
            {
                "chargeCategory": "4",
                "paymentArrangement": _PAYMENT_CODES[patch.freight.paymentArrangement],
            }
        ]
    if patch.parties is not None:
        details["partiesInformation"] = _parties(
            patch.parties, country_resolver=country_resolver
        )
    if patch.forwardingAndExportReferences is not None:
        details["forwardingAndExportReferences"] = [
            {"references": value} for value in patch.forwardingAndExportReferences
        ]
    if patch.goodsItems is not None:
        details["goodsItemDetails"] = [_goods_item(item) for item in patch.goodsItems]
    if details:
        result["consignmentInformation"] = {"consignmentDetails": details}
    return result


def project_bill_of_lading_to_mpci(
    label: BillOfLadingLabel,
    *,
    country_resolver: CountryResolver | None = None,
) -> MpciBillOfLadingLabel:
    """Project every semantic v2 target field into the frozen sparse MPCI v1 shape.

    Every OCR-copied country string deliberately requires an injected,
    versioned downstream resolver.
    """

    payload = {
        "schemaVersion": "1.0.0",
        "documentPatch": _document_patch(
            label.documentPatch, country_resolver=country_resolver
        ),
    }
    return MpciBillOfLadingLabel.model_validate_json(
        json.dumps(payload, ensure_ascii=False), strict=True
    )
