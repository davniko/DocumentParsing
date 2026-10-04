"""OCR-grounded real-data target: complete postal address and goods-local facts.

Historical target models stay immutable for old-run reproduction. This explicit
new contract removes the separate party city and cargo overflow field; no implicit
conversion can reconstruct postal text or decide cargo ownership.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, Field, model_validator

from document_ocr.label_schemas.bill_of_lading import (
    AddressText,
    ApplicationText,
    BillOfLadingRoute,
    BillOfLadingTransport,
    CargoMass,
    CargoText,
    ContactDetails,
    CountryText,
    FreightTerms,
    GoodsOrigin,
    Mass,
    NonNegativeQuantity,
    PackageTypeText,
    PartyReference,
    SemanticLocation,
    Temperature,
    Volume,
)
from document_ocr.label_schemas.bill_of_lading_v3 import CategoryToken, HazardCategory
from document_ocr.label_schemas.bill_of_lading_v4 import (
    PackingGroupCategoryV4,
    RelationExplicitDangerousGoodsFlashPointV4,
    RelationExplicitDangerousGoodsV4,
)
from document_ocr.label_schemas.bill_of_lading_v5 import (
    ContainerSizeCategory,
    ContainerTypeCategory,
)
from document_ocr.label_schemas.bill_of_lading_v6 import (
    ContainerInformationV6,
    NumberAndTypeOfPackagesV6,
    SplitGoodsPlacementV6,
)
from document_ocr.label_schemas.common import LabelSchemaModel
from document_ocr.label_schemas.mpci_bill_of_lading import ContainerIdentifier, HsCode, ImoNumber


# Describe the active contract here without changing historical schema hashes.
# Inherited validators preserve the existing wire types and validation behavior.
class LocationV7(SemanticLocation):
    """One document-owned locality; registry codes and geographic enrichment are downstream."""

    name: ApplicationText | None = Field(
        default=None,
        description=(
            "Locality name supported by this location's OCR phrase. Preserve its "
            "spelling and distinguishing locality words; omit the country name or "
            "adjective and generic facility descriptors such as seaport, airport or "
            "terminal. Retain words that are part of the locality's actual name, "
            "rather than stripping matching words mechanically. No registry enrichment."
        ),
    )
    country: CountryText | None = Field(
        default=None,
        description=(
            "Country explicitly supported by this location's OCR wording or country "
            "code, including an unambiguous national adjective. Normalize to the "
            "conventional English short country name in uppercase. A city alone or "
            "another party's country does not establish this location's country."
        ),
    )


class ContactsV7(ContactDetails):
    """Contact people and communication values explicitly owned by the named party."""

    contactName: ApplicationText | None = Field(
        default=None,
        description=(
            "Explicitly identified contact person names, in source order, separated by "
            "'; ' when several are printed. Uncaptioned identity names belong in the "
            "party name."
        ),
    )
    phoneNumbers: tuple[ApplicationText, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "All distinct party-owned telephone numbers, including uncaptioned "
            "continuation lines, one per entry. Preserve spelling and printed "
            "prefixes; never supply omitted prefixes. A number explicitly labeled for "
            "both telephone and fax qualifies. A number labeled only FAX is excluded, "
            "even beside a separate TEL number on the same physical line."
        ),
    )
    emailAddresses: tuple[ApplicationText, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Distinct party-owned email addresses as printed. Do not repair missing "
            "characters or turn a bare domain into email."
        ),
    )
    websiteUrls: tuple[ApplicationText, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Distinct explicitly attributed company/contact websites, including an "
            "owned homepage in company terms or a footer. A homepage remains eligible "
            "when mentioned in legal text; exclude links to specific articles/help/legal "
            "clauses and websites not owned by this party."
        ),
    )


class RouteV7(BillOfLadingRoute):
    """Shipment locations in supported transport roles, not inferred itinerary order."""

    placeOfReceipt: LocationV7 | None = Field(
        default=None,
        description=(
            "Place where the carrier receives the shipment, identified by "
            "receipt/pre-carriage context; distinct from the loading port."
        ),
    )
    portOfLoading: LocationV7 | None = Field(
        default=None,
        description=(
            "Sea port where the main ocean carriage loads the shipment; distinct from "
            "receipt place or pre-carriage origin."
        ),
    )
    transshipmentPort: LocationV7 | None = Field(
        default=None,
        description=(
            "Explicit intermediate port where the shipment changes vessel. Do not "
            "classify an arbitrary extra port as transshipment."
        ),
    )
    portOfDischarge: LocationV7 | None = Field(
        default=None,
        description=(
            "Sea port where the main ocean carriage discharges the shipment; distinct "
            "from inland delivery or final destination."
        ),
    )
    placeOfDelivery: LocationV7 | None = Field(
        default=None,
        description=(
            "Place where the carrier delivers the goods, supported by shipment delivery "
            "context. An office where the customer applies for release documents is a "
            "delivery-agent address, not the goods' delivery place. A consignee address "
            "or repeated discharge location alone does not establish this role."
        ),
    )
    finalDestination: LocationV7 | None = Field(
        default=None,
        description=(
            "Explicit final-destination place when identified separately. Do not derive"
            " it from another route field."
        ),
    )


class TransportV7(BillOfLadingTransport):
    """Main-carriage vessel and voyage, separate from pre-carriage and transshipment legs."""

    vesselName: ApplicationText | None = Field(
        default=None,
        description=(
            "Printed main-carriage vessel name, excluding a separate voyage number and "
            "field caption."
        ),
    )
    vesselImoNumber: ImoNumber | None = Field(
        default=None,
        description=(
            "Explicit seven-digit vessel IMO identifier, without the IMO caption. A "
            "voyage, booking or company identifier is not an IMO number; never repair "
            "its digits to pass validation."
        ),
    )
    voyageNumber: ApplicationText | None = Field(
        default=None,
        description=(
            "Printed main-carriage voyage identifier, keeping letters and leading "
            "zeros. Exclude vessel name, bill number and a separately headed leg/direction code."
        ),
    )
    vesselFlagCountry: CountryText | None = Field(
        default=None,
        description=(
            "Explicitly stated vessel flag/nationality country. Carrier nationality, "
            "ports and party countries do not establish the flag."
        ),
    )


class MassV7(Mass):
    """Explicit container verified gross mass, retaining the printed unit."""

    value: float = Field(
        gt=0,
        allow_inf_nan=False,
        description=(
            "Printed positive VGM amount, parsed numerically without changing magnitude"
            " or converting units."
        ),
    )
    unit: Literal["kilogram", "pound"] = Field(
        description=(
            "Printed unit: KG/KGS/KGM means kilogram; LB/LBS/LBR means pound. Leave the"
            " enclosing mass absent if no unit can be established."
        )
    )


class CargoMassV7(CargoMass):
    """Goods mass in its source unit, distinct from container tare and package capacity."""

    value: float = Field(
        gt=0,
        allow_inf_nan=False,
        description=(
            "Goods mass, parsing source decimal/thousands separators. Exact totals may "
            "sum complete, nonduplicated portions at the same goods/package level and "
            "unit; do not multiply a package capacity by an assumed count."
        ),
    )
    unit: Literal["kilogram", "pound", "metric_tonne"] = Field(
        description=(
            "Canonical printed unit: KG/KGS/KGM→kilogram, LB/LBS/LBR→pound, MT/metric "
            "tonnes→metric_tonne. Preserve magnitude and unit; never guess an absent "
            "unit or convert tonnes to kilograms. A unit in a shared column heading "
            "can apply to its rows; a unit on a different measure does not transfer."
        )
    )


class VolumeV7(Volume):
    """Goods volume, not container capacity or dimensions."""

    value: float = Field(
        gt=0,
        allow_inf_nan=False,
        description=(
            "Printed goods volume or exact sum of complete nonduplicated same-goods "
            "portions in the same unit."
        ),
    )
    unit: Literal["cubic_metre"] = Field(
        description=(
            "Cubic metres, explicitly printed as CBM, m3, m³ or equivalent. Do not "
            "assume this unit for an unqualified measurement."
        )
    )


class TemperatureV7(Temperature):
    """A signed temperature with its explicit scale; neither a range nor an inferred value."""

    value: float = Field(
        allow_inf_nan=False,
        description=(
            "Printed signed temperature value, preserving minus signs and zero. Do not "
            "replace a range by a guessed point."
        ),
    )
    unit: Literal["celsius", "fahrenheit"] = Field(
        description=(
            "Normalize printed C/°C/CELSIUS (including clear spelling variants) to "
            "celsius, and F/°F/FAHRENHEIT to fahrenheit. Do not infer a missing scale."
        )
    )


class OriginV7(GoodsOrigin):
    """Explicit origin of this goods item, not the shipper address or loading port."""

    name: ApplicationText | None = Field(
        default=None,
        description=(
            "Printed goods-origin place/country name, typically an origin or "
            "manufacture declaration; retain source wording."
        ),
    )
    identifier: ApplicationText | None = Field(
        default=None,
        description=(
            "Recognizable geographic code explicitly identifying this goods origin. "
            "A detached origin heading does not make a nearby commercial/B/L identifier "
            "an origin code. Do not generate codes or reinterpret shipment references."
        ),
    )


class FreightV7(FreightTerms):
    """Shipment-specific freight payment instructions, excluding unselected form alternatives."""

    paymentArrangement: Literal["prepaid", "collect", "third_party", "payable_elsewhere"] | None = (
        Field(
            default=None,
            description=(
                "Selected freight terms: prepaid=paid at origin; collect=payable at "
                "destination; third_party=explicit third-party payer; "
                "payable_elsewhere=explicit other payment place/arrangement. Empty "
                "Empty payment captions, including a freight-advance receipt box, "
                "do not select a value. A reference to charter-party terms alone "
                "does not state the payment arrangement."
            ),
        )
    )
    paymentPlace: LocationV7 | None = Field(
        default=None,
        description=(
            "Explicit place where freight is payable. Do not substitute a "
            "loading/discharge port without payment context."
        ),
    )


class FlashPointV7(RelationExplicitDangerousGoodsFlashPointV4):
    """Explicit flash point of hazardous cargo, separate from carrying temperature."""

    temperature: TemperatureV7 = Field(
        description=(
            "Printed DG flash-point temperature and scale, not container setpoint, "
            "melting point or ambient temperature."
        )
    )


class DangerousGoodsV7(RelationExplicitDangerousGoodsV4):
    """Printed dangerous-goods declaration for this goods item; no enrichment from UN lookup."""

    unNumber: str | None = Field(
        default=None,
        pattern=r"^[0-9]{4}$",
        description=(
            "Four printed UN-number digits, retaining leading zeros and removing the UN"
            " caption. Do not infer a UN number from product name."
        ),
    )
    hazardCategory: HazardCategory | None = Field(
        default=None,
        description=(
            "Primary printed IMDG/IMO class family: 1 explosives; 2 gases; 3 flammable "
            "liquids; 4 flammable solids; 5 oxidizers/peroxides; 6 toxic/infectious; 7 "
            "radioactive; 8 corrosive; 9 miscellaneous. Map divisions to their family; "
            "no lookup-inferred class."
        ),
    )
    subsidiaryHazardCategories: tuple[HazardCategory, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Distinct explicitly printed subsidiary hazard class families using the "
            "primary-class mapping. A primary class is not automatically subsidiary."
        ),
    )
    packingGroupCategory: PackingGroupCategoryV4 | None = Field(
        default=None,
        description=(
            "Printed packing group I/1→HIGH_DANGER, II/2→MEDIUM_DANGER, "
            "III/3→LOW_DANGER. NOT_ASSIGNED requires an explicit unassigned statement; "
            "missing group stays absent."
        ),
    )
    flashPoint: FlashPointV7 | None = Field(
        default=None,
        description=(
            "Flash point explicitly declared for this hazardous goods entry. "
            "Independent of whether a packing group is printed."
        ),
    )


class ContainerInformationV7(ContainerInformationV6):
    """One printed container and its owned facts; repeated page copies add no containers."""

    equipmentIdentifier: ContainerIdentifier = Field(
        pattern=r"^[A-Z]{3}[UJZ][0-9]{7}$",
        description=(
            "Printed container ID, letters and digits including check digit; remove "
            "presentation spaces/separators only. Do not include adjacent seal/type "
            "text or invent digits to satisfy ISO validation."
        ),
    )
    typeDescription: ApplicationText | None = Field(
        default=None,
        description=(
            "Printed equipment wording when a complete supported size/type category "
            "pair cannot be established. Exclusive with the canonical pair; an ID alone"
            " implies no type."
        ),
    )
    sizeCategory: ContainerSizeCategory | None = Field(
        default=None,
        description=(
            "Printed length/height class. Standard=8ft6; high cube=9ft6. Select "
            "supported 20/40/45-foot class only with typeCategory; otherwise retain "
            "printed wording as typeDescription."
        ),
    )
    typeCategory: ContainerTypeCategory | None = Field(
        default=None,
        description=(
            "Printed equipment family; emit only together with sizeCategory, otherwise "
            "use typeDescription. GP/DRY=general purpose; complete 40HC/HQ=40ft "
            "high-cube GP unless specialized type is explicit. RE=reefer, "
            "RT=reefer/heated, RS=self-powered reefer, HR=removable thermal equipment, "
            "HI=insulated, UT=open top; VH=ventilated, BU=dry bulk, SN=named cargo, "
            "PL/PF/PC/PS/PT=platform variants, KL=pressurized tank, NH/NN=dry tank "
            "discharge variants, AS=air/surface. Unknown tokens, including GEN, do not "
            "establish a family."
        ),
    )
    verifiedGrossMass: MassV7 | None = Field(
        default=None,
        description=(
            "Explicit VGM for this container, including cargo, packing and tare. Cargo "
            "gross mass alone is not VGM."
        ),
    )
    sealNumbers: tuple[ApplicationText, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Distinct seals belonging to this container, retaining alphanumeric "
            "prefixes and leading zeros. A container-detail row can identify a seal "
            "without a separate caption. Carrier and shipper seals are distinct; a "
            "no-seal declaration for one does not cancel the other. Exclude absence "
            "declarations, seal captions, equipment dimensions and adjacent separators."
        ),
    )
    temperatureSetpoint: TemperatureV7 | None = Field(
        default=None,
        description=(
            "Carrying temperature explicitly applicable to this container. A clearly "
            "shared shipment/cargo declaration may apply to several containers; "
            "position or reefer type alone is insufficient."
        ),
    )


class PackagesV7(NumberAndTypeOfPackagesV6):
    """Inner-level package facts, not outer pallet/container counts."""

    packageQuantity: NonNegativeQuantity | None = Field(
        default=None,
        description=(
            "Printed inner-level package count, or exact sum of complete same-level "
            "portions for this goods item. Distinguish package count from "
            "pieces/product capacity and avoid double-counting aggregate and portions. "
            "Serial-number ranges and mass divided by capacity do not establish a target count."
        ),
    )
    typeCategory: CategoryToken | None = Field(
        default=None,
        description=(
            "Canonical package category from the supplied package vocabulary, selected "
            "by printed meaning. Never invent category tokens; use typeOfPackages when "
            "the vocabulary does not cover the printed type."
        ),
    )
    typeOfPackages: PackageTypeText | None = Field(
        default=None,
        description=(
            "Complete printed package-type wording when no supplied category applies. "
            "Exclusive with typeCategory; not the goods description."
        ),
    )


class PlacementV7(SplitGoodsPlacementV6):
    """Goods-to-container association, optionally with its inner-package count."""

    equipmentIdentifier: ContainerIdentifier = Field(
        pattern=r"^[A-Z]{3}[UJZ][0-9]{7}$",
        description=(
            "Exact canonical ID of an emitted container carrying this goods item: four "
            "uppercase letters and seven digits, with presentation spaces/separators removed. "
            "Association "
            "must follow cargo rows, attachment structure or an explicit shared "
            "declaration. One shared goods entry explicitly covering a container list "
            "supports each listed membership even when some allocation counts are absent."
        ),
    )
    packageQuantity: NonNegativeQuantity | None = Field(
        default=None,
        description=(
            "Count of this goods item's packages in this container, at its labeled "
            "package level. Unknown count stays absent while supported membership "
            "remains; do not distribute totals evenly or force partial allocations to "
            "balance."
        ),
    )


class ExtractionPartyV7(LabelSchemaModel):
    """One named party with a complete postal address and separate contacts."""

    sameAs: PartyReference | None = Field(
        default=None,
        description=(
            "Notify-party reference to an emitted shipper or consignee only when OCR "
            "explicitly says same as that role. Replaces name/address/country; separate"
            " contact overrides may remain. Similar text alone does not establish a "
            "reference."
        ),
    )
    name: ApplicationText | None = Field(
        default=None,
        description=(
            "Printed party identity, preserving spelling and uncaptioned personal names"
            " under the company. Follow linked identity continuations even after a page "
            "break or tax/contact lines. Preserve A ON BEHALF OF B in a shared named-party "
            "postal block, not in a carrier signature's agent/principal clause; separate "
            "competing identities/addresses require "
            "review. Keep legal-name suffixes; exclude share-capital amounts, company "
            "registration boilerplate, bare TO ORDER and role captions. Postal zones, "
            "building/block designations and localities belong to addressLine even "
            "when they share a physical line with the company name."
        ),
    )
    addressLine: AddressText | None = Field(
        default=None,
        description=(
            "Complete party-owned postal address including buildings, districts, "
            "localities, postcode and country wherever printed. Write one line with "
            "comma-space separators between distinct postal components. Rejoin wrapped "
            "words/identifiers; ordinary word boundaries retain a space. Preserve wording, numbers "
            "and order, and existing internal punctuation. Include "
            "owned continuations after contacts/customs text or page headers, and postal "
            "department/landmark text; exclude names, contacts, tax/registration data, "
            "captions and formatting markers. Keep postcode/address values but omit "
            "their POSTAL CODE/ZIP/ADD captions. Include each repeated postal component "
            "once, even when one occurrence touches a postcode; retain all distinct "
            "postcode digits and building/district qualifiers. Repeated copies add no text. For "
            "alternative addresses select only an unambiguous primary address; if "
            "undecidable leave addressLine null for review, never concatenate them."
        ),
    )
    country: CountryText | None = Field(
        default=None,
        description=(
            "Recognizable postal country name or code printed for this party, "
            "preserving source form. Include country in addressLine too. Clear "
            "aliases/demonyms may support the country; no geocoding from city, "
            "telephone or company name, and no guessed repair of malformed country "
            "text."
        ),
    )
    contactDetails: ContactsV7 | None = Field(
        default=None,
        description=(
            "Explicitly owned contact people, telephone, email and websites, including "
            "separated continuations. Exclude another party's contacts and unowned "
            "footer details."
        ),
    )

    @model_validator(mode="after")
    def valid_party(self) -> ExtractionPartyV7:
        identity = (self.name, self.addressLine, self.country)
        if self.sameAs is not None and any(value is not None for value in identity):
            raise ValueError("sameAs replaces identity/address/country, not contact overrides")
        if self.sameAs is None and not any(
            value is not None for value in (*identity, self.contactDetails)
        ):
            raise ValueError("party requires details or an explicit sameAs relation")
        return self


class ExtractionPartiesV7(LabelSchemaModel):
    """Parties by document-supported function; missing form-required parties remain absent."""

    shipper: ExtractionPartyV7 | None = Field(
        default=None,
        description=(
            "Named shipper/consignor in shipment context. An exporter customs ID alone "
            "does not identify or supply a shipper address."
        ),
    )
    consignee: ExtractionPartyV7 | None = Field(
        default=None,
        description=(
            "Named consignee or named bank/entity in a TO ORDER OF clause; omit the "
            "order preamble from its identity. Bare TO ORDER supplies no named "
            "consignee. Negotiability is separate; never invent a consignee to meet "
            "form submission rules."
        ),
    )
    notifyParties: tuple[ExtractionPartyV7, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "All distinct notify parties in printed order, or an explicit sameAs "
            "reference to a present shipper/consignee. Repeated page copies are not "
            "extra notify parties."
        ),
    )
    carrier: ExtractionPartyV7 | None = Field(
        default=None,
        description=(
            "Named contractual carrier/issuer, supported by carrier wording, full "
            "issuer imprint or an explicit signature reference to the named carrier. "
            "In an agent signing for a carrier, extract the named carrier, not the "
            "agent/principal signature phrase. A bare logo/SCAC is insufficient."
        ),
    )
    forwardingAgent: ExtractionPartyV7 | None = Field(
        default=None,
        description=(
            "Party explicitly acting as forwarding agent for this shipment. Generic "
            "shipping/signing agent or ON BEHALF OF text alone does not establish this "
            "role."
        ),
    )
    deliveryAgent: ExtractionPartyV7 | None = Field(
        default=None,
        description=(
            "Named delivery/release agent or carrier destination office handling "
            "arrival/release. Do not infer delivery role merely from a company's "
            "location."
        ),
    )
    consolidator: ExtractionPartyV7 | None = Field(
        default=None,
        description=(
            "Explicit consolidator of this shipment. Carrier, forwarder, exporter and "
            "groupage cargo wording do not alone identify a consolidator."
        ),
    )

    @model_validator(mode="after")
    def valid_references(self) -> ExtractionPartiesV7:
        if not self.model_dump(exclude_none=True):
            raise ValueError("parties must contain a role")
        for role in type(self).model_fields:
            if role == "notifyParties":
                continue
            party = getattr(self, role)
            if party is not None and party.sameAs is not None:
                raise ValueError("sameAs is allowed only for notifyParties")
        for party in self.notifyParties or ():
            if party.sameAs is not None and getattr(self, party.sameAs) is None:
                raise ValueError("notify party references absent party")
        return self


class GoodsItemDetailsV7(LabelSchemaModel):
    """One goods identity/accounting entry, potentially placed in several containers.

    Separate independently quantified different products/specifications/lots. Repeated portions
    of one shared product are one goods item with placements, even with per-container weights.
    Multiple product names or HS codes alone do not force a split; repeated copies add no goods.
    Establish identities across the complete document before assigning portions. A product
    separately accounted anywhere retains that identity in mixed portions; shared amounts
    remain unallocated. Combine product names into one joint entry only when they are
    accounted together throughout, without separately established product accounting.
    """

    description: CargoText | None = Field(
        default=None,
        description=(
            "Complete product-owned wording: identity, brand, model/product/article codes, "
            "composition, "
            "specifications, condition, lot qualifiers, proper shipping name and "
            "printed package capacity. Join lines with spaces in source order. A "
            "description column may also contain shipment quantities/masses, destinations "
            "and shipment/administrative references: these are separate facts, not product "
            "wording. Exclude "
            "extracted HS/DG codes and generic disclaimers."
        ),
    )
    grossWeight: CargoMassV7 | None = Field(
        default=None,
        description=(
            "Gross mass of this goods item including its packaging, not container "
            "tare/VGM or weight per package. A shipment total belongs to this entry "
            "when it is the sole goods identity covering the entire shipment. "
            "Conflicting same-scope totals or complete portion sums require review."
        ),
    )
    netWeight: CargoMassV7 | None = Field(
        default=None,
        description=(
            "Explicit net mass of this goods item excluding packaging. Do not infer net"
            " from an unlabeled weight or subtract guessed tare. A shipment net total "
            "belongs to the sole goods identity covering the entire shipment."
        ),
    )
    volume: VolumeV7 | None = Field(
        default=None,
        description=(
            "Volume attributed to this goods item; not an unrelated consignment total "
            "or container capacity. A shipment volume belongs to the sole goods "
            "identity covering the entire shipment."
        ),
    )
    marksAndNumbers: tuple[CargoText, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Printed package/cargo identification markings for this goods item, in source "
            "order, including explicit N/M. A package-label block can contain order/invoice "
            "numbers and origin wording. Consignee/project shorthand can be a cargo mark "
            "in a combined equipment/marks panel; establish its identifying function "
            "rather than requiring a separate marks caption. Standalone customs/"
            "administrative references, "
            "package-type captions, container IDs and seals have other meanings. A "
            "combined marks/container heading does not turn customs identifiers into "
            "marks. For doubtful column ownership inspect the PDF layout and complete "
            "block before assigning. Product wording also printed "
            "on package labels still belongs in description."
        ),
    )
    hsCodes: tuple[HsCode, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "All distinct printed HS/customs commodity codes for this goods item, "
            "including multiple codes. Remove presentation dots/spaces only; retain all"
            " 6-18 digits and leading zeros. Do not pad, truncate or infer a code from "
            "product identity."
        ),
    )
    handlingInstructions: tuple[CargoText, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Explicit goods/shipment-specific carriage, storage, handling or delivery "
            "instructions in source order. Generic "
            "carrier responsibility/disclaimer text, including shipper's load/stow/count/"
            "seal declarations, is not a goods-specific handling instruction."
        ),
    )
    dangerousGoods: tuple[DangerousGoodsV7, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Distinct DG declarations explicitly tied to this goods item, without "
            "duplication of repeated statements or enrichment from product/UN "
            "registries."
        ),
    )
    origin: OriginV7 | None = Field(
        default=None,
        description=(
            "Explicit goods origin/manufacture declaration for this item, not the "
            "origin of its shipper or transport."
        ),
    )
    numberAndTypeOfPackages: tuple[PackagesV7, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Goods-owned package facts at the inner level (bags on pallets→bags). "
            "Containment must be established by the source; the mere presence of two "
            "package types does not rank them as inner and outer. Keep "
            "source-distinct package rows; outer packing is not a second target level. "
            "Do not emit aggregate and component counts twice."
        ),
    )
    splitGoodsPlacement: tuple[PlacementV7, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Source-supported placements of this goods item in emitted containers, "
            "after the goods facts. Shared goods may span several containers; unknown "
            "allocations remain absent rather than guessed."
        ),
    )

    @model_validator(mode="after")
    def valid_facts(self) -> GoodsItemDetailsV7:
        if not self.model_dump(exclude_none=True):
            raise ValueError("goods item must contain a supported fact")
        for name in ("marksAndNumbers", "hsCodes", "handlingInstructions"):
            values = getattr(self, name)
            if values is not None and len(values) != len(set(values)):
                raise ValueError(f"{name} values must be unique and source ordered")
        return self


def _unique_references(values: tuple[str, ...]) -> tuple[str, ...]:
    """Keep the reference constraint on its field so section models inherit it."""
    if len(values) != len(set(values)):
        raise ValueError("references must be unique and source ordered")
    return values


class BillOfLadingDocumentPatchV7(LabelSchemaModel):
    """OCR-supported facts of one bill of lading/sea waybill, not a completed customs submission."""

    billOfLadingNumber: ApplicationText | None = Field(
        default=None,
        description=(
            "This document's B/L or sea-waybill identifier, retaining letters and "
            "leading zeros. Use the carrier-issued document identifier, not a booking, "
            "master-reference, uploaded-file name or electronic-platform receipt/reference."
        ),
    )
    originalBillOfLadingNumber: ApplicationText | None = Field(
        default=None,
        description=(
            "Explicit original-B/L reference identifier. ORIGINAL stamps, copy counts "
            "and repetitions of the current bill number do not establish this "
            "reference."
        ),
    )
    masterBillOfLadingNumber: ApplicationText | None = Field(
        default=None,
        description=(
            "Explicit master/parent B/L reference identifier. Do not promote the "
            "current bill number or infer one from carrier identity."
        ),
    )
    issueDate: date | None = Field(
        default=None,
        description=(
            "B/L issue date normalized to YYYY-MM-DD, not departure/ETD or on-board "
            "date. Establish day/month order from an explicit format or unambiguous "
            "dates in the same source convention, never from country, issuer or language. "
            "If both orders remain possible, absence is the correct target."
        ),
    )
    shippedOnBoardDate: date | None = Field(
        default=None,
        description=(
            "Explicit shipped/on-board date normalized to YYYY-MM-DD, not a generic "
            "issue/sailing date. Establish day/month order from source format evidence, "
            "not country, issuer or language. If both orders remain possible, absence "
            "is the correct target."
        ),
    )
    negotiability: Literal["negotiable", "non_negotiable"] | None = Field(
        default=None,
        description=(
            "Read the actual consignee instruction: TO ORDER, TO THE ORDER OF or "
            "equivalent order-consignment wording means negotiable; a named consignee "
            "without order wording means non_negotiable. An explicit sea waybill/"
            "non-negotiable issuance also supports non_negotiable. If neither consignee "
            "instruction nor issuance declaration is available, leave absent. Generic "
            "form captions, contract wording about order/assigns and copy stamps do not "
            "supply the instruction. Conflicting shipment declarations require review."
        ),
    )
    placeOfIssue: LocationV7 | None = Field(
        default=None,
        description=(
            "Printed place where this B/L was issued, not automatically a loading port "
            "or agent address."
        ),
    )
    route: RouteV7 | None = Field(
        default=None,
        description=(
            "Supported transport locations in their actual roles, including attachment "
            "continuations."
        ),
    )
    transport: TransportV7 | None = Field(
        default=None,
        description=(
            "Main-carriage vessel/voyage information; separate identifiers from "
            "neighboring route or pre-carriage text."
        ),
    )
    freight: FreightV7 | None = Field(
        default=None,
        description=(
            "Selected shipment freight payment terms and payment place, not empty form "
            "alternatives."
        ),
    )
    parties: ExtractionPartiesV7 | None = Field(
        default=None,
        description=(
            "Supported party identities, full postal addresses and contacts by role; no"
            " platform scaffolding or geographic enrichment."
        ),
    )
    containerInformation: tuple[ContainerInformationV7, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Distinct identified shipment containers in first source order, including "
            "attachments. Missing type/size/seal facts remain absent; repeated page "
            "copies do not add containers."
        ),
    )
    forwardingAndExportReferences: (
        Annotated[tuple[ApplicationText, ...], AfterValidator(_unique_references)] | None
    ) = Field(
        default=None,
        min_length=1,
        description=(
            "Distinct explicit shipment/commercial/customs references (booking, "
            "invoice, order, vendor, ACID, party tax/import/export IDs including those "
            "inside party blocks) in source order, "
            "retaining captions with values. This is not an inventory of every identifier: "
            "exclude B/L/vessel IMO or Lloyds/voyage/container/seal IDs, HS/DG codes, "
            "shipping marks and product "
            "lot/model identifiers belonging to their dedicated facts/description. Exclude "
            "bare numbers, country/type metadata and unrelated company registrations. "
            "An identifier invalid for its dedicated field is not reassigned here."
        ),
    )
    goodsItemDetails: tuple[GoodsItemDetailsV7, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Complete source-ordered goods entries, including attachments. Different "
            "quantified products remain separate; shared product container portions "
            "become placements, not duplicate goods. Multiple HS codes alone do not "
            "split a goods item."
        ),
    )

    @model_validator(mode="after")
    def valid_patch(self) -> BillOfLadingDocumentPatchV7:
        if not self.model_dump(exclude_none=True):
            raise ValueError("documentPatch requires a supported fact")
        identifiers = [row.equipmentIdentifier for row in self.containerInformation or ()]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("equipment identifiers must be unique")
        for goods in self.goodsItemDetails or ():
            placements = goods.splitGoodsPlacement or ()
            for placement in placements:
                if placement.equipmentIdentifier not in identifiers:
                    raise ValueError("placement references absent containerInformation")
        return self


class BillOfLadingExtractionV7Label(LabelSchemaModel):
    """OCR-grounded MPCI/CUSCAR training labels without audit metadata."""

    schemaVersion: Literal["7.0.0"] = Field(
        description="Fixed extraction contract version, not text to find in the document."
    )
    documentPatch: BillOfLadingDocumentPatchV7 = Field(
        description=(
            "Only supported document facts; absent optional values are null in "
            "structured responses and omitted from serialized training labels."
        )
    )

    def canonical_target(self) -> dict[str, Any]:
        target = self.model_dump(mode="json", exclude_none=True)
        for goods in target["documentPatch"].get("goodsItemDetails", []):
            if "splitGoodsPlacement" in goods:
                goods["splitGoodsPlacement"] = goods.pop("splitGoodsPlacement")
        return target
