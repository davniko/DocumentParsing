"""OCR-conditioned MPCI/CUSCAR KIE label schema for Bills of Lading.

The wire field names intentionally match the persisted MPCI form vocabulary.
Only document-owned paths are present: system state, form toggles, mutable
lookup results, UI helpers, and submission scaffolding are not model targets.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, Field, StringConstraints, field_validator, model_validator

from document_ocr.label_schemas.common import (
    ExtractionSourceReference,
    FieldEvidence,
    LabelSchemaModel,
    LabelWarning,
    NonEmptyString,
    TrimmedString,
)


def _ascii_text(value: str) -> str:
    if value != value.strip():
        raise ValueError("CUSCAR text must not contain leading or trailing whitespace")
    if not value:
        raise ValueError("CUSCAR text must not be empty")
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        raise ValueError("CUSCAR text must contain printable ASCII only")
    return value


def _finite_positive(value: float) -> float:
    if value <= 0.0:
        raise ValueError("measurement must be positive")
    return value


AsciiText = Annotated[str, AfterValidator(_ascii_text)]
CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
CurrencyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]
Locode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}[A-Z0-9]{3}$")]
ContainerCode = Annotated[str, StringConstraints(pattern=r"^[0-9A-Z]{4}$")]
PackageTypeCode = Annotated[str, StringConstraints(pattern=r"^[0-9A-Z]{2}$")]
HsCode = Annotated[str, StringConstraints(pattern=r"^[0-9]{6,18}$")]
UndgIdentifier = Annotated[str, StringConstraints(pattern=r"^[0-9]{4}$")]
PositiveMeasure = Annotated[float, Field(allow_inf_nan=False), AfterValidator(_finite_positive)]
NonNegativeQuantity = Annotated[int, Field(ge=0)]

PartyFunction = Literal["CZ", "CN", "NI", "N2", "CG", "DDR", "DP", "COX"]
PaymentArrangement = Literal["A", "B", "C", "P"]
MeasurementAttribute = Literal["AAB", "AAA", "ABJ"]
MeasurementUnit = Literal["KGM", "LBR", "MTQ"]
TemperatureUnit = Literal["CEL", "FAH"]
CommunicationMeans = Literal["TE", "EM", "AO"]
ContactIdentifier = Literal["COM", "IND"]


def iso6346_expected_check_digit(identifier_without_check_digit: str) -> int:
    """Return the ISO 6346 check digit for a validated ten-character body."""

    if len(identifier_without_check_digit) != 10:
        raise ValueError("ISO 6346 body must contain exactly ten characters")
    if (
        not identifier_without_check_digit[:4].isalpha()
        or not identifier_without_check_digit[:4].isascii()
        or not identifier_without_check_digit[:4].isupper()
        or not identifier_without_check_digit[4:].isascii()
        or not identifier_without_check_digit[4:].isdigit()
    ):
        raise ValueError("ISO 6346 body must contain four uppercase letters and six digits")
    letter_values = {
        letter: value
        for letter, value in zip(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            (
                10,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                34,
                35,
                36,
                37,
                38,
            ),
            strict=True,
        )
    }
    total = 0
    for position, character in enumerate(identifier_without_check_digit):
        value = int(character) if character.isdigit() else letter_values[character]
        total += value * (2**position)
    remainder = total % 11
    return 0 if remainder == 10 else remainder


def _validate_container_identifier(value: str) -> str:
    if len(value) != 11 or not value[:3].isalpha() or value[3] not in "UJZ":
        raise ValueError("equipmentIdentifier must be an ISO 6346 identifier")
    if not value[:4].isupper() or not value[4:].isdigit():
        raise ValueError("equipmentIdentifier must be uppercase letters followed by digits")
    if iso6346_expected_check_digit(value[:10]) != int(value[-1]):
        raise ValueError("equipmentIdentifier has an invalid ISO 6346 check digit")
    return value


ContainerIdentifier = Annotated[str, AfterValidator(_validate_container_identifier)]


def _validate_imo_number(value: str) -> str:
    if len(value) != 7 or not value.isdigit():
        raise ValueError(
            "transportMeansIdentificationNameIdentifier must be a seven-digit IMO number"
        )
    checksum = sum(
        int(digit) * weight for digit, weight in zip(value[:6], range(7, 1, -1), strict=True)
    )
    if checksum % 10 != int(value[-1]):
        raise ValueError(
            "transportMeansIdentificationNameIdentifier has an invalid IMO check digit"
        )
    return value


ImoNumber = Annotated[str, AfterValidator(_validate_imo_number)]


def _has_content(model: LabelSchemaModel, *, ignore: frozenset[str] = frozenset()) -> bool:
    values = model.model_dump(mode="python", exclude_none=True)
    return any(name not in ignore for name in values)


class LocationCandidate(LabelSchemaModel):
    """Document evidence for a location; mutable lookup results stay external."""

    locode: Locode | None = None
    name: AsciiText | None = None
    country_code: CountryCode | None = None

    @model_validator(mode="after")
    def contains_location_evidence(self) -> LocationCandidate:
        if not _has_content(self):
            raise ValueError("location candidate must contain at least one supported value")
        return self


class BlIdentifiers(LabelSchemaModel):
    houseBLNumber: AsciiText | None = None
    originalBLNumber: AsciiText | None = None
    parentBLNumber: AsciiText | None = None

    @model_validator(mode="after")
    def contains_identifier(self) -> BlIdentifiers:
        if not _has_content(self):
            raise ValueError("blIdentifiers must contain at least one identifier")
        return self


class NegotiabilityInformation(LabelSchemaModel):
    processingIndicatorDescriptionCode: Literal["NON", "NEG"] | None = None

    @model_validator(mode="after")
    def contains_negotiability(self) -> NegotiabilityInformation:
        if not _has_content(self):
            raise ValueError("processingInformation must contain supported evidence")
        return self


class TransportInformation(LabelSchemaModel):
    meansOfTransportJourneyIdentifier: AsciiText | None = None
    carrierIdentifierFreeText: AsciiText | None = None
    transportMeansIdentificationNameIdentifier: ImoNumber | None = None
    transportMeansIdentificationName: AsciiText | None = None
    transportMeansNationalityCode: CountryCode | None = None
    transportMeansOwnershipIndicatorCode: Literal["1", "2", "3"] | None = None
    powerTypeCode: Literal["1", "2", "3", "4", "5", "6", "7", "8", "9"] | None = None
    powerTypeDescription: AsciiText | None = None

    @model_validator(mode="after")
    def contains_transport_evidence(self) -> TransportInformation:
        if not _has_content(self):
            raise ValueError("transportInformation must contain supported evidence")
        return self


class DepartureAndArrivalPorts(LabelSchemaModel):
    portOfDeparture: LocationCandidate | None = None
    portOfArrival: LocationCandidate | None = None

    @model_validator(mode="after")
    def contains_port_evidence(self) -> DepartureAndArrivalPorts:
        if not _has_content(self):
            raise ValueError("departureAndArrivalPorts must contain a port")
        return self


class ArrivalAndDepartureTimes(LabelSchemaModel):
    estimatedTimeOfArrival: datetime | None = None
    estimatedTimeOfDeparture: datetime | None = None
    actualTimeOfDeparture: datetime | None = None

    @field_validator(
        "estimatedTimeOfArrival",
        "estimatedTimeOfDeparture",
        "actualTimeOfDeparture",
    )
    @classmethod
    def timestamps_are_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("voyage timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def contains_time_evidence(self) -> ArrivalAndDepartureTimes:
        if not _has_content(self):
            raise ValueError("timeOfArrivalAndDepartures must contain a timestamp")
        for field_name, departure in (
            ("estimatedTimeOfDeparture", self.estimatedTimeOfDeparture),
            ("actualTimeOfDeparture", self.actualTimeOfDeparture),
        ):
            if (
                departure is not None
                and self.estimatedTimeOfArrival is not None
                and departure > self.estimatedTimeOfArrival
            ):
                raise ValueError(f"{field_name} must not be after estimatedTimeOfArrival")
        return self


class VoyageDetails(LabelSchemaModel):
    transportInformation: TransportInformation | None = None
    departureAndArrivalPorts: DepartureAndArrivalPorts | None = None
    timeOfArrivalAndDepartures: ArrivalAndDepartureTimes | None = None

    @model_validator(mode="after")
    def contains_voyage_evidence(self) -> VoyageDetails:
        if not _has_content(self):
            raise ValueError("voyageDetails must contain supported evidence")
        return self


class ContainerSizeAndType(LabelSchemaModel):
    containerCode: ContainerCode | None = None

    @model_validator(mode="after")
    def contains_container_code(self) -> ContainerSizeAndType:
        if not _has_content(self):
            raise ValueError("containerSizeAndType must contain a containerCode")
        return self


class EquipmentSizeAndType(LabelSchemaModel):
    containerSizeAndType: ContainerSizeAndType | None = None
    equipmentDescription: AsciiText | None = None
    fullOrEmptyIndicatorCodes: (
        Literal["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"] | None
    ) = None

    @model_validator(mode="after")
    def contains_equipment_evidence(self) -> EquipmentSizeAndType:
        if not _has_content(self):
            raise ValueError("equipmentSizeAndType must contain supported evidence")
        return self


class NatureOfCargo(LabelSchemaModel):
    cargoTypeClassificationCode: Literal["1", "2", "3", "4", "9", "13", "19", "20", "21"]


class TransportServiceRequirement(LabelSchemaModel):
    serviceRequirementCode: Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]?$")]
    natureOfCargo: NatureOfCargo

    @field_validator("serviceRequirementCode")
    @classmethod
    def service_requirement_is_in_domain(cls, value: str) -> str:
        if int(value) > 66:
            raise ValueError("serviceRequirementCode must be between 1 and 66")
        return value


class VerifiedGrossMass(LabelSchemaModel):
    measure: PositiveMeasure
    measurementUnitCode: Literal["KGM", "LBR"]


class SealNumber(LabelSchemaModel):
    transportUnitSealIdentifier: AsciiText
    sealingPartyNameCode: TrimmedString | None = None
    sealingPartyName: AsciiText | None = None
    sealType: Literal["1", "2", "3"] | None = None


class TemperatureSetting(LabelSchemaModel):
    temperatureTypeCodeQualifier: Literal["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    temperatureDegree: Annotated[float, Field(allow_inf_nan=False)]
    unitCode: TemperatureUnit


class ContainerEquipmentIdentification(LabelSchemaModel):
    equipmentIdentifier: ContainerIdentifier


class GoodsPlacementEquipmentIdentification(LabelSchemaModel):
    equipmentIdentifier: ContainerIdentifier
    packageQuantity: NonNegativeQuantity | None = None


class ContainerInformation(LabelSchemaModel):
    equipmentIdentification: ContainerEquipmentIdentification
    equipmentSizeAndType: EquipmentSizeAndType | None = None
    transportServiceRequirements: tuple[TransportServiceRequirement, ...] | None = Field(
        default=None, min_length=1
    )
    containerVerifiedGrossMass: VerifiedGrossMass | None = None
    sealNumbers: tuple[SealNumber, ...] | None = Field(default=None, min_length=1)
    temperatureSettings: tuple[TemperatureSetting, ...] | None = Field(default=None, min_length=1)


class ConsignmentLocation(LabelSchemaModel):
    locationOfIdentificationQualifier: Literal["7", "9", "12", "13", "88", "96"]
    locationIdentifier: LocationCandidate | None = None
    locationIdentifierCountry: CountryCode | None = None
    locationName: AsciiText | None = None
    firstRelatedLocationName: AsciiText | None = None

    @model_validator(mode="after")
    def contains_location_value(self) -> ConsignmentLocation:
        if not _has_content(self, ignore=frozenset({"locationOfIdentificationQualifier"})):
            raise ValueError("consignment location must contain document-supported location data")
        return self


class PartyName(LabelSchemaModel):
    name: AsciiText


class PartyAddress(LabelSchemaModel):
    address: AsciiText


class ContactInformationValue(LabelSchemaModel):
    contactIdentifier: ContactIdentifier
    contactName: AsciiText | None = None


class CommunicationContact(LabelSchemaModel):
    communicationMeans: CommunicationMeans
    identifier: AsciiText


class PartyContactInformation(LabelSchemaModel):
    contactInformation: ContactInformationValue
    communicationContact: tuple[CommunicationContact, ...]


class PartyInformation(LabelSchemaModel):
    partyFunction: PartyFunction
    partyIdentifier: AsciiText | None = None
    codeListIdentificationCode: Literal["1", "2"] | None = None
    partyNames: tuple[PartyName, ...] | None = Field(default=None, min_length=1)
    addresses: tuple[PartyAddress, ...] | None = Field(default=None, min_length=1)
    city: AsciiText | None = None
    country: CountryCode | None = None
    contactInformations: tuple[PartyContactInformation, ...] | None = Field(
        default=None, min_length=1
    )

    @model_validator(mode="after")
    def contains_party_value(self) -> PartyInformation:
        if not _has_content(self, ignore=frozenset({"partyFunction"})):
            raise ValueError("party must contain document-supported data in addition to its role")
        return self


class MonetaryAmount(LabelSchemaModel):
    typeCodeQualifier: Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]{0,2}$")]
    amount: PositiveMeasure
    currencyIdentificationCode: CurrencyCode

    @field_validator("typeCodeQualifier")
    @classmethod
    def type_code_is_in_domain(cls, value: str) -> str:
        if int(value) > 550:
            raise ValueError("typeCodeQualifier must be between 1 and 550")
        return value


class ChargePaymentInstruction(LabelSchemaModel):
    chargeCategory: Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]?$")] | None = None
    paymentArrangement: PaymentArrangement | None = None

    @field_validator("chargeCategory")
    @classmethod
    def charge_category_is_in_domain(cls, value: str | None) -> str | None:
        if value is not None and int(value) > 24:
            raise ValueError("chargeCategory must be between 1 and 24")
        return value

    @model_validator(mode="after")
    def contains_charge_evidence(self) -> ChargePaymentInstruction:
        if not _has_content(self):
            raise ValueError("chargePaymentInstruction must contain supported evidence")
        if self.paymentArrangement is not None and self.chargeCategory is None:
            raise ValueError("paymentArrangement requires its document-supported chargeCategory")
        return self


class ForwardingAndExportReference(LabelSchemaModel):
    references: AsciiText


class NumberAndTypeOfPackages(LabelSchemaModel):
    packageQuantity: NonNegativeQuantity | None = None
    packageTypeDescriptionCode: PackageTypeCode | None = None
    typeOfPackages: Annotated[str, AfterValidator(_ascii_text), Field(max_length=35)] | None = None
    packagingRelatedDescriptionCode: TrimmedString | None = None

    @model_validator(mode="after")
    def contains_package_evidence(self) -> NumberAndTypeOfPackages:
        if not _has_content(self):
            raise ValueError("package record must contain supported evidence")
        return self


class HandlingInstruction(LabelSchemaModel):
    descriptionCode: TrimmedString | None = None
    handlingInstructionDescription: AsciiText | None = None

    @model_validator(mode="after")
    def contains_handling_evidence(self) -> HandlingInstruction:
        if not _has_content(self):
            raise ValueError("handling instruction must contain supported evidence")
        return self


class GoodsFreeText(LabelSchemaModel):
    textSubjectCodeQualifier: Literal["AAA", "AAI"]
    freeText: AsciiText


class GoodsMeasurement(LabelSchemaModel):
    measuredAttributeCode: MeasurementAttribute
    measurementUnitCode: MeasurementUnit
    measure: PositiveMeasure


class SplitGoodsPlacement(LabelSchemaModel):
    equipmentIdentification: GoodsPlacementEquipmentIdentification


class HazardCode(LabelSchemaModel):
    hazardIdentificationCode: Literal["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    additionalHazardClassificationIdentifier: AsciiText | None = None


class UndgInformation(LabelSchemaModel):
    identifier: UndgIdentifier
    flashpointDescription: AsciiText | None = None


class DangerousGoodsShipmentFlashpoint(LabelSchemaModel):
    shipmentFlashpointDegree: Annotated[float, Field(allow_inf_nan=False)] | None = None
    measurementUnitCode: TemperatureUnit | None = None
    packagingDangerLevelCode: Literal["1", "2", "3", "4"] | None = None

    @model_validator(mode="after")
    def contains_paired_temperature_or_packing_group(
        self,
    ) -> DangerousGoodsShipmentFlashpoint:
        if (self.shipmentFlashpointDegree is None) != (self.measurementUnitCode is None):
            raise ValueError("shipment flashpoint degree and unit must be present together")
        if self.shipmentFlashpointDegree is None and self.packagingDangerLevelCode is None:
            raise ValueError("dangerous-goods flashpoint row must contain supported evidence")
        return self


class DangerousGoods(LabelSchemaModel):
    hazardCode: tuple[HazardCode, ...] | None = Field(default=None, min_length=1)
    undgInformation: UndgInformation | None = None
    dangerousGoodsShipmentFlashpoint: tuple[DangerousGoodsShipmentFlashpoint, ...] | None = Field(
        default=None, min_length=1
    )

    @model_validator(mode="after")
    def contains_dangerous_goods_evidence(self) -> DangerousGoods:
        if not _has_content(self):
            raise ValueError("dangerousGoods must contain supported evidence")
        return self


class MarkAndLabel(LabelSchemaModel):
    shippingMarksDescription: AsciiText


class PackageIdentification(LabelSchemaModel):
    markingInstructionCode: TrimmedString | None = None
    marksAndLabels: tuple[MarkAndLabel, ...] = Field(min_length=1)


class CustomsGoodsIdentifier(LabelSchemaModel):
    value: HsCode


class CustomsIdentityCodes(LabelSchemaModel):
    customsGoodsIdentifier: tuple[CustomsGoodsIdentifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def customs_identifiers_are_unique(self) -> CustomsIdentityCodes:
        values = [item.value for item in self.customsGoodsIdentifier]
        if len(values) != len(set(values)):
            raise ValueError("customsGoodsIdentifier values must be unique within a goods item")
        return self


class CustomsStatusOfGoods(LabelSchemaModel):
    customsIdentityCodes: CustomsIdentityCodes


class GoodsLocation(LabelSchemaModel):
    locationOfIdentificationQualifier: Literal["27"]
    locationIdentifier: AsciiText
    locationName: AsciiText | None = None


class GoodsItemDetails(LabelSchemaModel):
    numberAndTypeOfPackages: tuple[NumberAndTypeOfPackages, ...] | None = Field(
        default=None, min_length=1
    )
    handlingInstructions: tuple[HandlingInstruction, ...] | None = Field(default=None, min_length=1)
    freeText: tuple[GoodsFreeText, ...] | None = Field(default=None, min_length=1)
    measurements: tuple[GoodsMeasurement, ...] | None = Field(default=None, min_length=1)
    splitGoodsPlacement: tuple[SplitGoodsPlacement, ...] | None = Field(default=None, min_length=1)
    dangerousGoods: tuple[DangerousGoods, ...] | None = Field(default=None, min_length=1)
    packageIdentification: tuple[PackageIdentification, ...] | None = Field(
        default=None, min_length=1
    )
    customsStatusOfGoods: CustomsStatusOfGoods | None = None
    locationOfIdentification: tuple[GoodsLocation, ...] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def contains_goods_evidence(self) -> GoodsItemDetails:
        if not _has_content(self):
            raise ValueError("goods item must contain at least one document-supported field")
        if self.splitGoodsPlacement is not None:
            identifiers = [
                placement.equipmentIdentification.equipmentIdentifier
                for placement in self.splitGoodsPlacement
            ]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("splitGoodsPlacement container references must be unique")
        return self


class ConsignmentDetails(LabelSchemaModel):
    monetaryAmount: tuple[MonetaryAmount, ...] | None = Field(default=None, min_length=1)
    locationOfIdentification: tuple[ConsignmentLocation, ...] | None = Field(
        default=None, min_length=1
    )
    chargePaymentInstructions: tuple[ChargePaymentInstruction, ...] | None = Field(
        default=None, min_length=1
    )
    partiesInformation: tuple[PartyInformation, ...] | None = Field(default=None, min_length=1)
    forwardingAndExportReferences: tuple[ForwardingAndExportReference, ...] | None = Field(
        default=None, min_length=1
    )
    goodsItemDetails: tuple[GoodsItemDetails, ...] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def entities_are_canonical(self) -> ConsignmentDetails:
        if not _has_content(self):
            raise ValueError("consignmentDetails must contain document-supported evidence")
        if self.locationOfIdentification is not None:
            qualifiers = [
                item.locationOfIdentificationQualifier for item in self.locationOfIdentification
            ]
            if len(qualifiers) != len(set(qualifiers)):
                raise ValueError("consignment location qualifiers must be unique")
            canonical_location_order = {
                "9": 0,
                "12": 1,
                "13": 2,
                "88": 3,
                "7": 4,
                "96": 5,
            }
            if qualifiers != sorted(qualifiers, key=canonical_location_order.__getitem__):
                raise ValueError(
                    "consignment locations must follow canonical qualifier order 9,12,13,88,7,96"
                )
        if self.partiesInformation is not None:
            roles = [item.partyFunction for item in self.partiesInformation]
            if len(roles) != len(set(roles)):
                raise ValueError("partyFunction values must be unique")
        return self


class ConsignmentInformation(LabelSchemaModel):
    consignmentDetails: ConsignmentDetails


class MpciBillOfLadingDocumentPatch(LabelSchemaModel):
    """Sparse, model-owned MPCI patch for one complete Bill of Lading."""

    billOfLadingIssueDate: date | None = None
    shippedOnBoardDate: date | None = None
    processingInformation: NegotiabilityInformation | None = None
    blIdentifiers: BlIdentifiers | None = None
    placeOfBillIssue: LocationCandidate | None = None
    placeOfFreightPayment: LocationCandidate | None = None
    voyageDetails: VoyageDetails | None = None
    containerInformation: tuple[ContainerInformation, ...] | None = Field(
        default=None, min_length=1
    )
    consignmentInformation: ConsignmentInformation | None = None

    @model_validator(mode="after")
    def patch_is_nonempty_and_relations_are_valid(self) -> MpciBillOfLadingDocumentPatch:
        if not _has_content(self):
            raise ValueError("documentPatch must contain at least one supported value")
        containers = self.containerInformation or ()
        container_ids = [item.equipmentIdentification.equipmentIdentifier for item in containers]
        if len(container_ids) != len(set(container_ids)):
            raise ValueError("container equipmentIdentifier values must be unique")

        details = (
            self.consignmentInformation.consignmentDetails if self.consignmentInformation else None
        )
        goods = details.goodsItemDetails if details and details.goodsItemDetails else ()
        known_containers = set(container_ids)
        for goods_index, item in enumerate(goods):
            placements = item.splitGoodsPlacement or ()
            for placement in placements:
                identifier = placement.equipmentIdentification.equipmentIdentifier
                if identifier not in known_containers:
                    raise ValueError(
                        "goods item "
                        f"{goods_index} references container {identifier!r} absent from "
                        "containerInformation"
                    )
            packages = item.numberAndTypeOfPackages or ()
            if len(packages) == 1 and placements:
                placement_quantities = [
                    placement.equipmentIdentification.packageQuantity for placement in placements
                ]
                package_quantity = packages[0].packageQuantity
                if package_quantity is not None and all(
                    quantity is not None for quantity in placement_quantities
                ):
                    allocated = sum(
                        quantity for quantity in placement_quantities if quantity is not None
                    )
                    if allocated != package_quantity:
                        raise ValueError(
                            f"goods item {goods_index} placement quantities do not equal "
                            "packageQuantity"
                        )
        return self


class MpciBillOfLadingLabel(LabelSchemaModel):
    """One training target for one multi-page Bill of Lading."""

    schemaVersion: Literal["1.0.0"]
    documentPatch: MpciBillOfLadingDocumentPatch

    def canonical_target(self) -> dict[str, Any]:
        """Return the stable sparse JSON target used for model training."""

        return self.model_dump(mode="json", exclude_none=True)


def _leaf_paths(value: Any, prefix: str) -> set[str]:
    if isinstance(value, dict):
        paths: set[str] = set()
        for key, child in value.items():
            paths.update(_leaf_paths(child, f"{prefix}.{key}"))
        return paths
    if isinstance(value, list):
        paths = set()
        for index, child in enumerate(value):
            paths.update(_leaf_paths(child, f"{prefix}[{index}]"))
        return paths
    return {prefix}


class MpciBillOfLadingAnnotation(LabelSchemaModel):
    """Auditable candidate/final annotation; not the model decoder target itself."""

    annotationSchemaVersion: Literal["1.0.0"]
    taskType: Literal["mpci_cuscar_kie"]
    documentType: Literal["bill_of_lading"]
    source: ExtractionSourceReference
    label: MpciBillOfLadingLabel
    evidence: tuple[FieldEvidence, ...] = Field(min_length=1)
    warnings: tuple[LabelWarning, ...] = ()
    reviewStatus: Literal["candidate", "validated", "needs_review", "rejected"]
    reviewNotes: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def every_target_leaf_has_exactly_one_ocr_evidence(self) -> MpciBillOfLadingAnnotation:
        target = self.label.canonical_target()["documentPatch"]
        expected_paths = _leaf_paths(target, "documentPatch")
        evidence_paths = [item.targetPath for item in self.evidence]
        if len(evidence_paths) != len(set(evidence_paths)):
            raise ValueError("each targetPath may have only one evidence record")
        supplied_paths = set(evidence_paths)
        if supplied_paths != expected_paths:
            missing = sorted(expected_paths - supplied_paths)
            unexpected = sorted(supplied_paths - expected_paths)
            raise ValueError(
                f"evidence paths differ from emitted target leaves; missing={missing!r}, "
                f"unexpected={unexpected!r}"
            )
        page_numbers = {page.pageNumber for page in self.source.pages}
        for evidence_item in self.evidence:
            evidence_page_numbers = {item.pageNumber for item in evidence_item.rawOcrEvidence}
            if not evidence_page_numbers.issubset(page_numbers):
                raise ValueError("evidence references a page outside the source document")
        for warning in self.warnings:
            if not set(warning.pageNumbers).issubset(page_numbers):
                raise ValueError("warning references a page outside the source document")
        return self
