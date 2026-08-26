"""Learnable, OCR-conditioned semantic KIE contract for Bills of Lading.

The decoder target in this module describes document facts with readable field
names.  MPCI/CUSCAR structural codes, repeated form locations, and other
application scaffolding belong to the deterministic projection layer.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Annotated, Any, Literal

import regex
from pydantic import AfterValidator, Field, StringConstraints, model_validator

from document_ocr.label_schemas.common import (
    ExtractionSourceReference,
    FieldEvidence,
    LabelSchemaModel,
    LabelWarning,
    NonEmptyString,
    RawOcrValueEvidence,
)
from document_ocr.label_schemas.mpci_bill_of_lading import (
    ContainerIdentifier,
    HsCode,
    ImoNumber,
)

_LATIN_SCRIPT_LETTER = regex.compile(r"(?V1)\A[\p{L}&&\p{scx=Latin}]\Z")


def _application_text(value: str) -> str:
    if value != value.strip():
        raise ValueError("text must not contain leading or trailing whitespace")
    if not value:
        raise ValueError("text must not be empty")
    if "\n" in value or "\r" in value:
        raise ValueError("one semantic value must not contain physical OCR line breaks")
    preceding_base_is_latin = False
    for character in value:
        codepoint = ord(character)
        if 0x20 <= codepoint <= 0x7E:
            preceding_base_is_latin = character.isalpha()
            continue
        category = unicodedata.category(character)
        name = unicodedata.name(character, "")
        if not character.isprintable():
            raise ValueError("application text must contain printable characters only")
        if category.startswith("L"):
            if "LATIN" not in name and _LATIN_SCRIPT_LETTER.fullmatch(character) is None:
                raise ValueError("application text letters must use the Latin script")
            preceding_base_is_latin = True
            continue
        if category.startswith("M"):
            if not preceding_base_is_latin:
                raise ValueError("combining marks must follow a Latin-script letter")
            continue
        if category[0] in {"N", "P", "S"} or category == "Zs":
            preceding_base_is_latin = False
            continue
        raise ValueError(
            "application text may contain only Latin-script letters and printable "
            "numbers, punctuation, symbols, or spaces"
        )
    return value


ApplicationText = Annotated[str, AfterValidator(_application_text)]
CountryText = Annotated[ApplicationText, Field(max_length=128)]
# The MPCI form currently limits its supplementary package description to 35
# characters.  That is a downstream submission constraint, not a document fact:
# supervision must retain the complete printed package wording for later mapping.
PackageTypeText = ApplicationText
PositiveMeasure = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeQuantity = Annotated[int, Field(ge=0)]
ContainerTypeCode = Annotated[str, StringConstraints(pattern=r"^[0-9A-Z]{4}$")]


_ADDRESS_CONTAMINATION = re.compile(
    r"(?ix)"
    r"(?:\b(?:tax(?:\s*(?:id|no|number))?|vat|cif|v\.?d\.?|c\.?n\.?p\.?j|"
    r"acid(?:\s*code)?|"
    r"e-?mail|tel(?:ephone)?|phone|ph|fax)\b\s*[:#])|(?:\S+@\S+)"
)
_CARGO_BOILERPLATE = re.compile(
    r"(?ix)\b(?:"
    r"shipper['\u2019]?s\s+load(?:\s*[,/&]\s*|\s+and\s+)count|"
    r"said\s+to\s+contain|s\.?t\.?c\.?|"
    r"weight(?:,?\s+(?:contents?|measure|quality|quantity|condition|value))+\s+unknown|"
    r"particulars\s+furnished\s+by\s+shipper"
    r")\b"
)


def _pure_address(value: str) -> str:
    if _ADDRESS_CONTAMINATION.search(value):
        raise ValueError("address contains a tax identifier or communication field")
    return value


def _pure_cargo_text(value: str) -> str:
    if _CARGO_BOILERPLATE.search(value):
        raise ValueError("cargo value contains carrier boilerplate rather than a cargo fact")
    return value


AddressText = Annotated[ApplicationText, AfterValidator(_pure_address)]
CargoText = Annotated[ApplicationText, AfterValidator(_pure_cargo_text)]


def _has_content(model: LabelSchemaModel, *, ignore: frozenset[str] = frozenset()) -> bool:
    values = model.model_dump(mode="python", exclude_none=True)
    return any(name not in ignore for name in values)


def _unique(values: tuple[Any, ...] | None, field_name: str) -> None:
    if values is not None and len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique and source ordered")


class SemanticLocation(LabelSchemaModel):
    """Printed place data; all geographic coding remains downstream."""

    name: ApplicationText | None = None
    country: CountryText | None = None

    @model_validator(mode="after")
    def contains_location(self) -> SemanticLocation:
        if not _has_content(self):
            raise ValueError("location must contain at least one OCR-supported value")
        return self


class BillOfLadingRoute(LabelSchemaModel):
    placeOfReceipt: SemanticLocation | None = None
    portOfLoading: SemanticLocation | None = None
    transshipmentPort: SemanticLocation | None = None
    portOfDischarge: SemanticLocation | None = None
    placeOfDelivery: SemanticLocation | None = None
    finalDestination: SemanticLocation | None = None

    @model_validator(mode="after")
    def contains_route(self) -> BillOfLadingRoute:
        if not _has_content(self):
            raise ValueError("route must contain at least one named location")
        return self


class ContactDetails(LabelSchemaModel):
    contactName: ApplicationText | None = None
    phoneNumbers: tuple[ApplicationText, ...] | None = Field(default=None, min_length=1)
    emailAddresses: tuple[ApplicationText, ...] | None = Field(default=None, min_length=1)
    websiteUrls: tuple[ApplicationText, ...] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def contains_contact_value(self) -> ContactDetails:
        if not any(
            (self.contactName, self.phoneNumbers, self.emailAddresses, self.websiteUrls)
        ):
            raise ValueError(
                "contactDetails requires a contact name, phone, email, or website value"
            )
        _unique(self.phoneNumbers, "phoneNumbers")
        _unique(self.emailAddresses, "emailAddresses")
        _unique(self.websiteUrls, "websiteUrls")
        return self


PartyReference = Literal["shipper", "consignee"]


class SemanticParty(LabelSchemaModel):
    """One logical party; address is one joined semantic value, never OCR rows."""

    sameAs: PartyReference | None = None
    name: ApplicationText | None = None
    address: AddressText | None = None
    city: ApplicationText | None = None
    country: CountryText | None = None
    contactDetails: ContactDetails | None = None

    @model_validator(mode="after")
    def is_reference_or_party_value(self) -> SemanticParty:
        identity_fields = (self.name, self.address, self.city, self.country)
        if self.sameAs is not None and any(value is not None for value in identity_fields):
            raise ValueError(
                "sameAs replaces repeated identity, address, and location details; "
                "it may accompany only contactDetails as an explicit override"
            )
        if self.sameAs is None and not any(
            value is not None for value in (*identity_fields, self.contactDetails)
        ):
            raise ValueError("party must contain details or an explicit sameAs relation")
        return self


class BillOfLadingParties(LabelSchemaModel):
    shipper: SemanticParty | None = None
    consignee: SemanticParty | None = None
    notifyParties: tuple[SemanticParty, ...] | None = Field(
        default=None, min_length=1
    )
    carrier: SemanticParty | None = None
    forwardingAgent: SemanticParty | None = None
    deliveryAgent: SemanticParty | None = None
    consolidator: SemanticParty | None = None

    @model_validator(mode="after")
    def contains_parties_and_valid_references(self) -> BillOfLadingParties:
        if not _has_content(self):
            raise ValueError("parties must contain at least one named role")
        for role in (
            "shipper",
            "consignee",
            "carrier",
            "forwardingAgent",
            "deliveryAgent",
            "consolidator",
        ):
            party = getattr(self, role)
            if party is not None and party.sameAs is not None:
                raise ValueError("sameAs is authorized only within notifyParties")
        for party in self.notifyParties or ():
            if party.sameAs == "shipper" and self.shipper is None:
                raise ValueError("notify party references absent shipper")
            if party.sameAs == "consignee" and self.consignee is None:
                raise ValueError("notify party references absent consignee")
        return self


class BillOfLadingTransport(LabelSchemaModel):
    vesselName: ApplicationText | None = None
    vesselImoNumber: ImoNumber | None = None
    voyageNumber: ApplicationText | None = None
    vesselFlagCountry: CountryText | None = None

    @model_validator(mode="after")
    def contains_transport(self) -> BillOfLadingTransport:
        if not _has_content(self):
            raise ValueError("transport must contain at least one OCR-supported value")
        return self


MassUnit = Literal["kilogram", "pound"]
CargoMassUnit = Literal["kilogram", "pound", "metric_tonne"]


class Mass(LabelSchemaModel):
    """Container verified gross mass in an MPCI-supported source unit."""

    value: PositiveMeasure
    unit: MassUnit


class CargoMass(LabelSchemaModel):
    """Cargo mass in the explicit unit printed by the source document."""

    value: PositiveMeasure
    unit: CargoMassUnit


class Volume(LabelSchemaModel):
    value: PositiveMeasure
    unit: Literal["cubic_metre"]


class Temperature(LabelSchemaModel):
    value: Annotated[float, Field(allow_inf_nan=False)]
    unit: Literal["celsius", "fahrenheit"]


class BillOfLadingContainer(LabelSchemaModel):
    containerNumber: ContainerIdentifier
    typeDescription: ApplicationText | None = None
    typeCode: ContainerTypeCode | None = None
    verifiedGrossMass: Mass | None = None
    sealNumbers: tuple[ApplicationText, ...] | None = Field(default=None, min_length=1)
    temperatureSetpoint: Temperature | None = None

    @model_validator(mode="after")
    def seals_are_unique(self) -> BillOfLadingContainer:
        _unique(self.sealNumbers, "sealNumbers")
        return self


class PackageRecord(LabelSchemaModel):
    quantity: NonNegativeQuantity | None = None
    type: PackageTypeText | None = None
    typeCode: Annotated[str, StringConstraints(pattern=r"^[0-9A-Z]{2}$")] | None = None

    @model_validator(mode="after")
    def contains_package(self) -> PackageRecord:
        if not _has_content(self):
            raise ValueError("package record must contain quantity, type, or exact printed code")
        return self


class ContainerAllocation(LabelSchemaModel):
    containerNumber: ContainerIdentifier
    packageQuantity: NonNegativeQuantity | None = None


class DangerousGoodsFlashPoint(LabelSchemaModel):
    temperature: Temperature
    packingGroup: Literal["I", "II", "III"] | None = None


class DangerousGoods(LabelSchemaModel):
    unNumber: Annotated[str, StringConstraints(pattern=r"^[0-9]{4}$")] | None = None
    hazardClass: Literal["1", "2", "3", "4", "5", "6", "7", "8", "9"] | None = None
    subsidiaryHazard: ApplicationText | None = None
    flashPoint: DangerousGoodsFlashPoint | None = None

    @model_validator(mode="after")
    def contains_dangerous_goods(self) -> DangerousGoods:
        if not _has_content(self):
            raise ValueError("dangerous goods record must contain supported evidence")
        return self


class GoodsOrigin(LabelSchemaModel):
    name: ApplicationText | None = None
    identifier: ApplicationText | None = None

    @model_validator(mode="after")
    def contains_origin(self) -> GoodsOrigin:
        if not _has_content(self):
            raise ValueError("goods origin must contain a name or printed identifier")
        return self


class BillOfLadingGoodsItem(LabelSchemaModel):
    description: CargoText | None = None
    additionalInformation: tuple[CargoText, ...] | None = Field(default=None, min_length=1)
    packages: tuple[PackageRecord, ...] | None = Field(default=None, min_length=1)
    grossWeight: CargoMass | None = None
    netWeight: CargoMass | None = None
    volume: Volume | None = None
    containerAllocations: tuple[ContainerAllocation, ...] | None = Field(
        default=None, min_length=1
    )
    marksAndNumbers: tuple[CargoText, ...] | None = Field(default=None, min_length=1)
    hsCodes: tuple[HsCode, ...] | None = Field(default=None, min_length=1)
    handlingInstructions: tuple[CargoText, ...] | None = Field(default=None, min_length=1)
    dangerousGoods: tuple[DangerousGoods, ...] | None = Field(default=None, min_length=1)
    origin: GoodsOrigin | None = None

    @model_validator(mode="after")
    def contains_goods_and_unique_lists(self) -> BillOfLadingGoodsItem:
        if not _has_content(self):
            raise ValueError("goods item must contain at least one OCR-supported fact")
        _unique(self.additionalInformation, "additionalInformation")
        _unique(self.marksAndNumbers, "marksAndNumbers")
        _unique(self.hsCodes, "hsCodes")
        _unique(self.handlingInstructions, "handlingInstructions")
        if self.containerAllocations is not None:
            allocation_facts = tuple(
                (item.containerNumber, item.packageQuantity)
                for item in self.containerAllocations
            )
            _unique(allocation_facts, "containerAllocations")
        return self


class FreightTerms(LabelSchemaModel):
    paymentArrangement: Literal[
        "prepaid", "collect", "third_party", "payable_elsewhere"
    ] | None = None
    paymentPlace: SemanticLocation | None = None

    @model_validator(mode="after")
    def contains_freight_term(self) -> FreightTerms:
        if not _has_content(self):
            raise ValueError("freight must contain an arrangement or payment place")
        return self


class BillOfLadingDocumentPatch(LabelSchemaModel):
    """Sparse semantic facts for one and only one transport document."""

    billOfLadingNumber: ApplicationText | None = None
    originalBillOfLadingNumber: ApplicationText | None = None
    masterBillOfLadingNumber: ApplicationText | None = None
    issueDate: date | None = None
    shippedOnBoardDate: date | None = None
    negotiability: Literal["negotiable", "non_negotiable"] | None = None
    placeOfIssue: SemanticLocation | None = None
    route: BillOfLadingRoute | None = None
    transport: BillOfLadingTransport | None = None
    freight: FreightTerms | None = None
    parties: BillOfLadingParties | None = None
    containers: tuple[BillOfLadingContainer, ...] | None = Field(default=None, min_length=1)
    forwardingAndExportReferences: tuple[ApplicationText, ...] | None = Field(
        default=None, min_length=1
    )
    goodsItems: tuple[BillOfLadingGoodsItem, ...] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def patch_is_nonempty_and_relations_are_valid(self) -> BillOfLadingDocumentPatch:
        if not _has_content(self):
            raise ValueError("documentPatch must contain at least one OCR-supported fact")
        _unique(self.forwardingAndExportReferences, "forwardingAndExportReferences")
        containers = self.containers or ()
        identifiers = tuple(container.containerNumber for container in containers)
        _unique(identifiers, "containers.containerNumber")
        known_containers = set(identifiers)
        for goods_index, goods in enumerate(self.goodsItems or ()):
            for allocation in goods.containerAllocations or ():
                if allocation.containerNumber not in known_containers:
                    raise ValueError(
                        f"goods item {goods_index} references absent container "
                        f"{allocation.containerNumber!r}"
                    )
        return self


class BillOfLadingLabel(LabelSchemaModel):
    schemaVersion: Literal["2.0.0"]
    documentPatch: BillOfLadingDocumentPatch

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


class BillOfLadingAnnotation(LabelSchemaModel):
    """Auditable semantic annotation; evidence is excluded from decoder targets."""

    annotationSchemaVersion: Literal["2.0.0"]
    taskType: Literal["bill_of_lading_kie"]
    documentType: Literal["bill_of_lading", "sea_waybill"]
    source: ExtractionSourceReference
    label: BillOfLadingLabel
    evidence: tuple[FieldEvidence, ...] = Field(min_length=1)
    warnings: tuple[LabelWarning, ...] = ()
    reviewStatus: Literal["candidate", "validated", "needs_review", "rejected"]
    reviewNotes: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def evidence_exactly_covers_target(self) -> BillOfLadingAnnotation:
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
                f"evidence paths differ from target leaves; missing={missing!r}, "
                f"unexpected={unexpected!r}"
            )
        page_numbers = {page.pageNumber for page in self.source.pages}
        for evidence_item in self.evidence:
            cited_pages = {item.pageNumber for item in evidence_item.rawOcrEvidence}
            if not cited_pages.issubset(page_numbers):
                raise ValueError("evidence references a page outside the source document")
        return self


class BillOfLadingExclusion(LabelSchemaModel):
    """Fail-closed record for a work item that cannot yield one truthful label."""

    exclusionSchemaVersion: Literal["2.0.0"]
    taskType: Literal["bill_of_lading_kie"]
    source: ExtractionSourceReference
    reason: Literal[
        "multiple_transport_documents",
        "not_bill_of_lading_or_sea_waybill",
        "insufficient_ocr",
        "non_latin_text",
    ]
    rawOcrEvidence: tuple[RawOcrValueEvidence, ...] = Field(min_length=1)
    reviewStatus: Literal["rejected"]
    reviewNotes: tuple[NonEmptyString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def exclusion_evidence_uses_source_pages(self) -> BillOfLadingExclusion:
        page_numbers = {page.pageNumber for page in self.source.pages}
        if not {item.pageNumber for item in self.rawOcrEvidence}.issubset(page_numbers):
            raise ValueError("exclusion evidence references a page outside the source document")
        return self
