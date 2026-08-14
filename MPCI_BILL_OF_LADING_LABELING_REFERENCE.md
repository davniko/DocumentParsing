# MPCI Bill-of-Lading raw-OCR-to-label reference

Version: `1.0.0`  
Applies to: `MpciBillOfLadingAnnotation` / `MpciBillOfLadingLabel` schema version `1.0.0`  
Semantic snapshot date: 2026-08-13

## Purpose

This is the semantic companion to the strict Pydantic models in
`src/document_ocr/label_schemas/`. The models define legal JSON shape, types, lexical patterns,
cross-field constraints, and evidence mechanics. They deliberately do not carry every field
definition, display meaning, categorical label, or source-to-code decision needed by a labeling
worker. This reference supplies that missing operational meaning.

The target is a sparse MPCI/CUSCAR `documentPatch` for one complete, ordered, multi-page Bill of
Lading. It is not a transcription of the full form and it is not a submission-ready filing. The
training input is the joined raw GLM-OCR text. The PDF and retained page images may clarify layout
and grouping, but they are never an additional source of target characters or facts.

## Authority and conflict rules

Apply these authorities together:

1. The work item's ordered raw OCR text is the complete truth boundary for target content.
2. `src/document_ocr/label_schemas/common.py` and
   `src/document_ocr/label_schemas/mpci_bill_of_lading.py` are the executable authority for shape,
   types, constraints, and relationships.
3. This reference is the authority for field meaning and which semantic conversions are authorized
   for annotation schema `1.0.0`.
4. The discovery bundle under `artifacts/mpci-ai-schema/` explains the application contract and
   exclusions, but does not broaden the Pydantic target.
5. External standards linked in this reference explain stable standardized codes. Workers must use
   this frozen explanation, not make live lookups or select a newer code-list meaning ad hoc.

When authorities appear to disagree, apply the most restrictive truthful result. The Pydantic
schema may accept a lexical value or numeric range whose business meaning has not been frozen here.
Such a value is **schema-valid but not annotation-authorized**. Omit it and add an
`unsupported_or_unclear_code` warning. Never infer a meaning merely because Pydantic accepts the
token.

This reference cannot authorize a field absent from the Pydantic model, weaken a Pydantic
constraint, or permit a value absent from raw OCR. If a real OCR-present fact has no matching target
field, omit it and use `schema_cannot_represent` with `targetPath = null`.

## Source references used for this snapshot

Local application discovery sources:

- [`field-catalog.md`](artifacts/mpci-ai-schema/field-catalog.md)
- [`selects-and-code-lists.md`](artifacts/mpci-ai-schema/selects-and-code-lists.md)
- [`relationships.md`](artifacts/mpci-ai-schema/relationships.md)
- [`schema-layer-matrix.md`](artifacts/mpci-ai-schema/schema-layer-matrix.md)
- [`ai-genie-normalization-and-mapping.md`](artifacts/mpci-ai-schema/ai-genie-normalization-and-mapping.md)
- [`dataset-contract.md`](artifacts/mpci-ai-schema/dataset-contract.md)

Standards sources:

- [UN/EDIFACT service requirement code 7273, D.12B](https://service.unece.org/trade/untdid/d12b/tred/tred7273.htm)
- [UN/EDIFACT full-or-empty code 8169, D.19A](https://service.unece.org/trade/untdid/d19a/tred/tred8169.htm)
- [UN/EDIFACT cargo classification code 7085, D.01B](https://service.unece.org/trade/untdid/d01b/tred/tred7085.htm)
- [UN/EDIFACT temperature qualifier 6245, D.19B](https://service.unece.org/trade/untdid/d19b/tred/tred6245.htm)
- [UN/EDIFACT power type code 7041, D.13A](https://service.unece.org/trade/untdid/d13a/tred/tred7041.htm)
- [UN/EDIFACT transport ownership code 8281, D.00A](https://service.unece.org/trade/untdid/d00a/tred/tred8281.htm)
- [UN/EDIFACT sealing-party code 9303, D.03B](https://service.unece.org/trade/untdid/d03b/tred/tred9303.htm)
- [UN/EDIFACT seal type code 4525, D.23A](https://service.unece.org/trade/untdid/d23a/tred/tred4525.htm)
- [UN/EDIFACT packaging-danger code 8339, D.09B](https://service.unece.org/trade/untdid/d09b/tred/tred8339.htm)
- [UN/EDIFACT text-subject code 4451, D.05A](https://service.unece.org/trade/untdid/d05a/tred/tred4451.htm)
- [UN/EDIFACT communication means code 3155, D.03A](https://service.unece.org/trade/untdid/d03a/tred/tred3155.htm)
- [UNECE Recommendation 21 package-code catalog](https://unece.org/code-list-recommendations)
- [BIC ISO 6346 size/type explanation](https://www.bic-code.org/size-type-code/)
- [BIC ISO 6346 container identification explanation](https://www.bic-code.org/identification-number/)
- [IMO IMDG dangerous-goods overview](https://www.imo.org/en/ourwork/safety/pages/dangerousgoods-default.aspx)

## The labeling boundary

### What may become a target

A target value may be emitted only when its characters or all facts needed for an authorized
deterministic conversion occur in the raw OCR text. Each non-null target leaf must cite verbatim OCR
through exactly one `FieldEvidence` record. That record keeps each exact pre-mapping raw value, its
page number, and its surrounding raw excerpt; `targetPath` links those originals to the converted
value in `label.documentPatch`.

Permitted uses of the PDF or raster are limited to:

- locating columns, headings, table rows, and page continuations;
- grouping OCR-present values into OCR-present entities;
- associating an OCR-present value with an OCR-present heading;
- selecting one OCR-present candidate over another repeated candidate; and
- checking a discrepancy before omitting an unsupported value.

The image cannot supply a missing character, check digit, decimal, unit, date component, field, or
value. Native PDF text and a second OCR engine are also outside the truth boundary.

### What must remain outside the target

Do not emit platform state, UI state, submission defaults, mutable lookup results, user decisions,
or filing workflow fields. Examples include message function, document-name code, update/cancel
reason, test indicator, delegator and shipping-line platform IDs, toggles, error helpers, remote
database IDs, sequential goods item numbers, and default placeholder rows.

An ACID number, booking number, VAT number, tax number, or generic customer reference is not an HS
code. A carrier name is not a platform carrier identifier. A place name is not a UN/LOCODE unless
the code itself appears in OCR.

## Worker conversion procedure

For one work item and one document:

1. Read every page in ascending `pageIndex`; do not label pages independently.
2. Build a private fact ledger from raw OCR only: literal excerpt, page, nearby heading, candidate
   entity, and whether another page repeats it.
3. Use layout only to group those OCR facts into document, voyage, container, party, goods, and
   dangerous-goods entities.
4. Select a Pydantic field only when its meaning below matches the fact exactly.
5. Apply only a normalization or categorical conversion explicitly authorized here.
6. Omit unsupported, ambiguous, invalid, or unrepresentable facts and add the appropriate warning.
7. Preserve source order for repeated entities. Use the canonical location order where required.
8. Create the sparse `documentPatch`; do not create empty objects or placeholder rows.
9. Add exactly one evidence item for every emitted leaf, using concrete array indexes. Retain every
   contributing pre-mapping string under `rawOcrEvidence` in page/source order.
10. Strictly validate the complete annotation with Pydantic before publishing the candidate.

Repeated page content does not create repeated entities. If page 1 contains a damaged value and page
2 contains an intact repetition, the intact page-2 value is permitted because it occurs in the raw
input. Cite it with `cross_page_resolution`. If two intact candidates conflict and context cannot
resolve them, omit the field and warn.

## Common lexical and structural rules

All models are strict, immutable, and reject unknown keys.

| Concept | Required representation | Annotation rule |
|---|---|---|
| Absence | Omit key or use `null`; canonical target removes `None` | Never use `""`, `0`, `[]`, or a placeholder object to stand for unknown unless that value is genuinely observed and schema-valid. |
| Text | Nonempty, trimmed, printable ASCII | Diacritic/transliteration normalization is allowed only when it deterministically represents OCR-present characters and is documented in evidence. Never invent missing letters. |
| Dates | `YYYY-MM-DD` | Convert only an unambiguous printed date. Do not choose day/month order from guesswork. |
| Voyage times | Timezone-aware ISO 8601 datetime | A printed local time without an explicit offset/zone cannot be assigned a timezone; omit it rather than inventing UTC. |
| Numbers | JSON number; finite | Remove OCR-present grouping separators and unit text only when the number and unit association are unambiguous. |
| Positive measures | Greater than zero | Zero or negative values are invalid for measures and monetary amounts. |
| Quantities | Non-negative integer | Do not round or coerce a decimal package count. |
| Country | Uppercase ISO 3166-1 alpha-2 | Map an explicit, unambiguous country name to ISO-2; never infer country from a city or address. |
| Currency | Uppercase ISO 4217 alpha-3 | Emit only with an OCR-supported currency and an authorized monetary qualifier. |
| UN/LOCODE | Two letters plus three letters/digits | Emit only when that exact code is printed. A place name remains `name`/manual location text. |
| Arrays | Source order | Do not sort repeated facts except page-number evidence and the required consignment-location qualifier order. |
| Nested objects | At least one meaningful child | Do not instantiate an optional object solely to carry nulls. |

## Root document fields

| Target path | Meaning | Raw-OCR conversion rule |
|---|---|---|
| `documentPatch.billOfLadingIssueDate` | Date on which the B/L was issued | Require an issue-date heading or unmistakable issue context. Normalize an unambiguous date to ISO. |
| `documentPatch.shippedOnBoardDate` | Date cargo was shipped/on board | Require `SHIPPED ON BOARD`, `ON BOARD DATE`, or equivalent context. It is independent of issue date and may be earlier. |
| `documentPatch.processingInformation.processingIndicatorDescriptionCode` | B/L negotiability | Use the `NON`/`NEG` table below. It is not a declaration-processing code. |
| `documentPatch.blIdentifiers.houseBLNumber` | House/current B/L number used by the application target | Use the principal `B/L NO`, `BILL OF LADING NO`, or explicitly `HOUSE B/L` value. Do not use booking/container/reference numbers. |
| `documentPatch.blIdentifiers.originalBLNumber` | Explicit original B/L reference | Emit only when the document labels a distinct number as original B/L. `ORIGINAL` copy-count boilerplate is not a number. |
| `documentPatch.blIdentifiers.parentBLNumber` | Parent/master B/L reference | Emit an explicitly labeled parent/master B/L number; do not infer from carrier/forwarder layout. |
| `documentPatch.placeOfBillIssue.*` | Place at which the B/L was issued | Use the location rules below and issue-place context only. |
| `documentPatch.placeOfFreightPayment.*` | Place where freight is payable | Use a location explicitly linked to freight payment, not the port of discharge by default. |

### Negotiability codes

| Code | Meaning | Authorized evidence |
|---|---|---|
| `NEG` | Negotiable / to-order B/L | Explicit `TO ORDER`, `ORDER OF ...`, `NEGOTIABLE`, or equivalent consignee/negotiability language. |
| `NON` | Non-negotiable / straight B/L | Explicit `NON-NEGOTIABLE`, `STRAIGHT BILL`, a clearly named fixed consignee in the negotiability context, or equivalent direct evidence. |

Do not infer `NON` merely because no `TO ORDER` phrase was found. Generic `NON-NEGOTIABLE COPY`
watermarks can describe a document copy rather than the negotiability of the underlying B/L; use
only when the surrounding document context makes the status unambiguous.

## Location representation

A `LocationCandidate` can carry printed/candidate location facts without performing the application's
mutable UN/LOCODE lookup:

| Child path | Meaning | Rule |
|---|---|---|
| `locode` | Printed UN/LOCODE | Exact uppercase five-character code only. Do not geocode or look it up from a name. |
| `name` | Printed locality/port/place name | Preserve the OCR-supported name after permitted ASCII/whitespace normalization. |
| `country_code` | Explicit country as ISO-2 | Convert an unambiguous printed country name/code; do not infer from locality. |

At least one child must be present. A location can contain both its printed code and printed name.
Do not add a name or country obtained from a code lookup; lookup hydration belongs to the application.

## Voyage details

| Target path | Meaning | Raw-OCR conversion rule |
|---|---|---|
| `documentPatch.voyageDetails.transportInformation.meansOfTransportJourneyIdentifier` | Voyage number/reference | Require a voyage heading or vessel/voyage pairing. |
| `documentPatch.voyageDetails.transportInformation.carrierIdentifierFreeText` | Carrier/shipping-line name as free text | Use the printed carrier name. Do not put a platform carrier ID here. |
| `documentPatch.voyageDetails.transportInformation.transportMeansIdentificationNameIdentifier` | Vessel IMO number | Strip an explicit `IMO` label/prefix only; resulting seven digits must pass the IMO checksum. |
| `documentPatch.voyageDetails.transportInformation.transportMeansIdentificationName` | Vessel name | Require vessel/ship context. Do not use carrier name. |
| `documentPatch.voyageDetails.transportInformation.transportMeansNationalityCode` | Vessel flag/nationality | Convert an explicit vessel nationality/flag country to ISO-2. Do not infer from vessel or carrier. |
| `documentPatch.voyageDetails.transportInformation.transportMeansOwnershipIndicatorCode` | Ownership arrangement | Use only the frozen ownership table below and explicit evidence. |
| `documentPatch.voyageDetails.transportInformation.powerTypeCode` | Propulsion/power type | Use only verified codes `1`-`6` below. |
| `documentPatch.voyageDetails.transportInformation.powerTypeDescription` | Printed propulsion description | Preserve explicit power text when a stable code is unavailable. |
| `documentPatch.voyageDetails.departureAndArrivalPorts.portOfDeparture.*` | B/L port of loading/departure | Map the document's port-of-loading/departure field. Use `LocationCandidate` rules. |
| `documentPatch.voyageDetails.departureAndArrivalPorts.portOfArrival.*` | B/L port of discharge/arrival | Map the document's port-of-discharge/arrival field. Use `LocationCandidate` rules. |
| `documentPatch.voyageDetails.timeOfArrivalAndDepartures.estimatedTimeOfArrival` | ETA | Emit only an explicit ETA with an explicit timezone/offset. |
| `documentPatch.voyageDetails.timeOfArrivalAndDepartures.estimatedTimeOfDeparture` | ETD | Emit only an explicit ETD with timezone/offset; it must not be after ETA. |
| `documentPatch.voyageDetails.timeOfArrivalAndDepartures.actualTimeOfDeparture` | Actual departure | Emit only an explicit actual-departure time with timezone/offset; it must not be after ETA. |

### Transport ownership

| Code | Meaning |
|---|---|
| `1` | Transport for the goods owner's account; the goods owner owns or rented the means. |
| `2` | Transport for another account; the goods owner neither owns nor rented the means. |
| `3` | Privately owned transport. |

### Power type

| Code | Meaning | Status |
|---|---|---|
| `1` | Diesel | Authorized |
| `2` | Diesel and electric | Authorized |
| `3` | Electric | Authorized |
| `4` | Liquefied petroleum/propane gas | Authorized |
| `5` | Petrol/gasoline | Authorized |
| `6` | Petrol and electric | Authorized |
| `7`, `8`, `9` | No meaning frozen in the cited UN/EDIFACT list | Pydantic-accepted, **not annotation-authorized**; use free text and warn. |

## Containers and equipment

Every `containerInformation[]` row requires a valid `equipmentIdentifier`. If OCR contains a seal,
temperature, type, service, or weight for a container whose identifier is missing or remains invalid,
the schema cannot represent that container row truthfully. Do not invent or repair the identifier to
retain the child fact; omit the row or unsupported children and warn.

| Target path | Meaning | Raw-OCR conversion rule |
|---|---|---|
| `documentPatch.containerInformation[].equipmentIdentification.equipmentIdentifier` | ISO 6346 equipment ID | After removing only OCR-present spaces/hyphens, require three owner letters, `U`/`J`/`Z`, six serial digits, and a valid check digit. Preserve source order and uniqueness. |
| `documentPatch.containerInformation[].equipmentSizeAndType.containerSizeAndType.containerCode` | Four-character ISO size/type code | Emit an exact printed code or a mapping specifically authorized below. Do not derive a code from dimensions by intuition. |
| `documentPatch.containerInformation[].equipmentSizeAndType.equipmentDescription` | Printed size/type description | Preserve source text such as `40 HC` or `REEFER`; this is the safe fallback when a canonical code is unclear. |
| `documentPatch.containerInformation[].equipmentSizeAndType.fullOrEmptyIndicatorCodes` | Container load/fullness state | Use only explicit source context and the frozen table below. |
| `documentPatch.containerInformation[].transportServiceRequirements[].serviceRequirementCode` | Requested transport service | A row is allowed only when both this service and its nested cargo classification are independently supported. |
| `documentPatch.containerInformation[].transportServiceRequirements[].natureOfCargo.cargoTypeClassificationCode` | Cargo classification paired with the service | Do not create `9` merely because a container row exists; require actual cargo-class context. |
| `documentPatch.containerInformation[].containerVerifiedGrossMass.measure` | Verified gross mass (VGM) value | Require explicit `VGM`/`VERIFIED GROSS MASS` context. Ordinary cargo gross weight is a goods measurement. |
| `documentPatch.containerInformation[].containerVerifiedGrossMass.measurementUnitCode` | VGM unit | `KGM` for kg/kgs/kilograms; `LBR` for lb/lbs/pounds. |
| `documentPatch.containerInformation[].sealNumbers[].transportUnitSealIdentifier` | Seal number | Require seal context and associate only to the OCR-supported container. |
| `documentPatch.containerInformation[].sealNumbers[].sealingPartyNameCode` | Party that applied the seal | Use only the sealing-party table below. The field is lexically broad but annotation policy is closed. |
| `documentPatch.containerInformation[].sealNumbers[].sealingPartyName` | Printed sealing-party name | Preserve an explicit name; do not infer it from a seal prefix. |
| `documentPatch.containerInformation[].sealNumbers[].sealType` | Physical seal type | Use the seal-type table only when explicit. |
| `documentPatch.containerInformation[].temperatureSettings[].temperatureTypeCodeQualifier` | Purpose/category of the setting | Use the temperature table. Generic reefer set temperature maps to `2` only when transport-setting context is clear. |
| `documentPatch.containerInformation[].temperatureSettings[].temperatureDegree` | Numeric temperature | Preserve sign and decimal. Do not convert Celsius/Fahrenheit unless both source and conversion are recorded; normally retain source unit. |
| `documentPatch.containerInformation[].temperatureSettings[].unitCode` | Temperature unit | `CEL` for C/Celsius; `FAH` for F/Fahrenheit. |

### Container size/type codes

An exact four-character code printed in OCR may be copied when it matches the Pydantic lexical form.
The following high-frequency phrase mappings are frozen for this annotation version from the BIC ISO
6346 size/type explanation:

| Explicit OCR phrase | Code | Required disambiguation |
|---|---|---|
| 20-foot, 8-foot-6 general-purpose/dry | `22G1` | Both size and general-purpose/dry type must be clear. |
| 40-foot, 8-foot-6 general-purpose/dry | `42G1` | Must not be described as high cube. |
| 40-foot, 9-foot-6 high-cube general-purpose/dry | `45G1` | `HC`/`HIGH CUBE` and 40-foot size must both be clear. |

Labels such as `RF`, `OT`, `FR`, `TK`, `BU`, a bare `GP`, or a dimension without a complete stable
mapping are not enough for a code under this snapshot. Preserve `equipmentDescription` and add
`unsupported_or_unclear_code`. A printed four-character ISO code remains directly usable.

### Full-or-empty indicator

| Code | Meaning |
|---|---|
| `1` | More than one-quarter of volume remains available. |
| `2` | More than one-half remains available. |
| `3` | More than three-quarters remains available. |
| `4` | Empty. |
| `5` | Full. |
| `6` | No volume remains available. |
| `7` | Full container holding mixed LCL consignments. |
| `8` | Full container holding one FCL consignment. |
| `9` | Part load; one customs declaration spans multiple containers. |
| `10` | Part load mixed with consignments from other declarations. |
| `11` | Load covered by one invoice. |
| `12` | Load covered by multiple invoices. |
| `13` | Full load for one consignee under multiple B/L numbers. |

`FCL` and `LCL` are not interchangeable with every code in this table. Use `7` or `8` only when the
OCR context supports the corresponding mixed/single-consignment meaning, not merely because a B/L
contains `FCL/FCL` or `LCL/LCL` boilerplate.

### Cargo classification allowed by this schema

| Code | Meaning |
|---|---|
| `1` | Documents not subject to duties/restrictions. |
| `2` | Low-value non-dutiable consignment. |
| `3` | Low-value dutiable consignment. |
| `4` | High-value consignment. |
| `9` | Containerized cargo. |
| `13` | Liquid cargo. |
| `19` | Obnoxious cargo, objectionable to human senses. |
| `20` | Out-of-gauge cargo with at least one non-standard dimension. |
| `21` | Household goods and personal effects. |

These codes do **not** mean generic categories such as dangerous, perishable, general, or frozen.
Although broader UN/EDIFACT code 7085 contains those concepts, this Pydantic version accepts only the
listed subset. Never force a fact into the nearest accepted category.

### Service requirement codes

The Pydantic field accepts `1` through `66`, but the cited authoritative directory freezes meanings
only for `1` through `64`. Codes `65` and `66` are therefore not annotation-authorized in this run.
The concise meanings below are semantic selectors, not phrases to match loosely.

| Code | Meaning | Code | Meaning |
|---|---|---|---|
| `1` | Carrier loads cargo | `33` | Report CSC safety-plate information |
| `2` | Full-load service | `34` | Check seals |
| `3` | Less-than-full-load service | `35` | Container must be clean |
| `4` | Shipper loads cargo | `36` | Provide proof of delivery |
| `5` | Deliver as instructed | `37` | Perform customs procedure |
| `6` | Hold pending instructions | `38` | Perform administrative services |
| `7` | Transshipment allowed | `39` | Insulated INTERFRIGO transport |
| `8` | Transshipment prohibited | `40` | Mechanically refrigerated INTERFRIGO transport |
| `9` | Partial shipment allowed | `41` | Cool/freeze service outside INTERFRIGO |
| `10` | Partial shipment prohibited | `42` | Overseas transshipment |
| `11` | Partial shipment/drawing allowed | `43` | Station delivery |
| `12` | Partial shipment/drawing prohibited | `44` | Non-station delivery |
| `13` | Carrier unloads cargo | `45` | Cleaning or disinfection |
| `14` | Shipper unloads cargo | `46` | Close ventilation valve |
| `15` | Consignee unloads cargo | `47` | Hold consignment for pickup |
| `16` | Consignee loads cargo | `48` | Check refrigeration unit |
| `17` | Exclusive equipment use | `49` | Carrier clears customs in arrival country |
| `18` | Non-exclusive equipment use | `50` | Carrier clears customs in departure country |
| `19` | Direct delivery | `51` | Provide heating for live animals |
| `20` | Direct pickup | `52` | Humidify goods |
| `21` | Delivery-advice service | `53` | Ensure load is secure |
| `22` | Do not arrange customs clearance | `54` | Open ventilation valve |
| `23` | Arrange customs clearance | `55` | Perform phytosanitary control |
| `24` | Check container condition | `56` | Carrier checks equipment tare |
| `25` | Damaged containers accepted | `57` | Check temperature |
| `26` | Dirty containers accepted | `58` | Weigh goods |
| `27` | Forklift pockets not required | `59` | Escort required |
| `28` | Forklift pockets required | `60` | No escort required |
| `29` | Insure goods during transport | `61` | Request berthing service |
| `30` | Arrange main carriage | `62` | Consider planned berth |
| `31` | Arrange on-carriage | `63` | Inbound passage through port area |
| `32` | Arrange pre-carriage | `64` | Outbound passage through port area |

Do not map shipping terms such as `CY/CY`, `FCL/FCL`, or `DOOR/PORT` to this list unless the exact
service meaning is unambiguous and its required cargo-classification partner is also supported.

### Sealing party and seal type

| Sealing-party code | Meaning |
|---|---|
| `AA` | Consolidator |
| `AB` | Unknown sealing party; use only if the document explicitly says unknown |
| `AC` | Quarantine agency |
| `CA` | Carrier |
| `CU` | Customs |
| `SH` | Shipper |
| `TO` | Terminal operator |

| Seal-type code | Meaning |
|---|---|
| `1` | Mechanical seal |
| `2` | Electronic seal |
| `3` | Air-fresh-vent seal |

A bare seal number does not establish either sealing party or seal type.

### Temperature qualifier

| Code | Meaning |
|---|---|
| `1` | Storage temperature |
| `2` | Transport temperature |
| `3` | Cargo operating/handling temperature |
| `4` | Transport emergency temperature |
| `5` | Transport control temperature |
| `6` | Boiling point |
| `7` | Recorded temperature |
| `8` | Self-accelerating decomposition temperature (SADT) |
| `9` | Self-accelerating polymerization temperature (SAPT) |

## Consignment details

`consignmentInformation.consignmentDetails` contains document-level money, locations, freight
payment, parties, references, and goods. It is sparse: do not create empty arrays or application
default rows.

### Monetary amounts

| Target path | Meaning | Annotation policy |
|---|---|---|
| `documentPatch.consignmentInformation.consignmentDetails.monetaryAmount[].typeCodeQualifier` | Kind/purpose of monetary amount | Pydantic accepts numeric strings `1`-`550`, but the MPCI option snapshot and meanings are not present in this repository. Not annotation-authorized without a future frozen list. |
| `documentPatch.consignmentInformation.consignmentDetails.monetaryAmount[].amount` | Positive monetary value | Cannot be emitted alone; the row also requires an authorized qualifier and currency. |
| `documentPatch.consignmentInformation.consignmentDetails.monetaryAmount[].currencyIdentificationCode` | ISO-4217 currency | Convert an explicit currency name/symbol only when unambiguous, but omit the entire row while its type qualifier is unresolved. |

An OCR phrase such as `FREIGHT USD 500` does not establish the required monetary type code by itself.
Preserve the issue as `unsupported_or_unclear_code`; do not select a number from the allowed range.

### Consignment locations

Each row is keyed by its semantic qualifier. Qualifiers must be unique and serialized in this exact
canonical order: `9`, `12`, `13`, `88`, `7`, `96`.

| Qualifier | Meaning | Typical OCR heading |
|---|---|---|
| `9` | Port/place of loading | `PORT OF LOADING`, `POL` |
| `12` | Port/place of discharge | `PORT OF DISCHARGE`, `POD` |
| `13` | Transshipment place | `TRANSSHIPMENT PORT/PLACE` |
| `88` | Place of receipt | `PLACE OF RECEIPT`, `RECEIVED AT` |
| `7` | Place of delivery | `PLACE OF DELIVERY`, `FINAL DESTINATION` only when used as delivery place |
| `96` | Emirates location | Explicit Emirates/UAE-location field in the MPCI document context |

| Target path within each row | Meaning | Rule |
|---|---|---|
| `documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].locationOfIdentificationQualifier` | Role key from the table | Contextual code; cite the heading and value. |
| `documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].locationIdentifier.locode` | Printed UN/LOCODE | Exact code only. |
| `documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].locationIdentifier.name` | Coded/candidate location name | OCR-supported place name. |
| `documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].locationIdentifier.country_code` | Country attached to candidate location | Explicit country only, normalized to ISO-2. |
| `documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].locationIdentifierCountry` | Country for a manually expressed place | Use only when the document presents a manual/free-text place and its country explicitly. Do not duplicate a nested country automatically. |
| `documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].locationName` | Manual/free-text place name | Use for an OCR-supported place that is not represented as a printed locode. Do not duplicate merely to fill both modes. |
| `documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].firstRelatedLocationName` | Explicit first related location text | Emit only when the document supplies a separately identified related location. Do not treat an inferred city, country, terminal, or alternate spelling as this field. |

The nested candidate form and manual name/country form reflect two application representation modes.
Do not populate both with duplicated values simply because both are optional. Prefer a printed locode
candidate when present; otherwise use the source-supported manual name/country representation that
matches the document evidence.

### Charge/payment instructions

| Target path | Meaning | Rule |
|---|---|---|
| `documentPatch.consignmentInformation.consignmentDetails.chargePaymentInstructions[].chargeCategory` | Category of transport charge | Only basic freight category `4` is frozen for source conversion in this version. Codes `1`-`24` are schema-valid but all other meanings are unresolved. |
| `documentPatch.consignmentInformation.consignmentDetails.chargePaymentInstructions[].paymentArrangement` | Who/where pays the charge | Requires the same row's supported charge category. Use the table below. |

| Code | Meaning | Authorized phrase/context |
|---|---|---|
| `P` | Prepaid | Explicit `FREIGHT PREPAID` or equivalent basic-freight statement |
| `C` | Collect | Explicit `FREIGHT COLLECT` or equivalent basic-freight statement |
| `B` | Third party pays | Explicit third-party payment statement tied to basic freight |
| `A` | Payable elsewhere | Explicit payable-elsewhere statement tied to basic freight |

For these mappings emit `chargeCategory = "4"`. A generic `PREPAID` printed next to another charge,
or an Incoterm such as `CIF`, does not establish basic-freight payment.

### Party roles and fields

At most one row of each party function is allowed. Preserve the document's party-block order unless
doing so would violate role uniqueness. Do not copy a consignee into notify merely because the notify
block is blank. Copying is allowed only when OCR explicitly states `SAME AS CONSIGNEE` or equivalent;
each copied leaf still cites that relation and the consignee's OCR-present value.

| Code | Meaning | Typical source heading |
|---|---|---|
| `CZ` | Shipper / consignor | `SHIPPER`, `CONSIGNOR` |
| `CN` | Consignee | `CONSIGNEE` |
| `NI` | Notify party | `NOTIFY PARTY` |
| `N2` | Second/additional notify party | `SECOND NOTIFY`, `NOTIFY PARTY 2` |
| `CG` | Carrier | `CARRIER` when identifying the transporting party |
| `DDR` | Forwarding agent / freight forwarder | `FORWARDING AGENT`, `FREIGHT FORWARDER` |
| `DP` | Delivery agent | `DELIVERY AGENT` |
| `COX` | Consolidator | `CONSOLIDATOR` |

| Target path within a party | Meaning | Raw-OCR conversion rule |
|---|---|---|
| `documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].partyFunction` | Structural role | Map only an explicit heading or unambiguous block context using the table. |
| `documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].partyIdentifier` | Printed identifier for that party | Require an explicit party-ID context. Names, tax IDs, VAT IDs, and address numbers are not interchangeable. |
| `documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].codeListIdentificationCode` | Code list governing `partyIdentifier` | Meanings for `1`/`2` are not frozen locally; omit even when an identifier is emitted unless a future application snapshot authorizes the code. |
| `documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].partyNames[].name` | Party legal/display name | Preserve source order when multiple names are genuinely printed. Do not split one wrapped name into several parties. |
| `documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].addresses[].address` | Address line | Preserve ordered OCR address lines; do not geocode or append image-only text. |
| `documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].city` | Explicit city | Do not infer from postal code, port, or external geocoder. |
| `documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].country` | Explicit country | Normalize an unambiguous printed country to ISO-2; do not derive from city/phone domain. |
| `documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].contactInformations[].contactInformation.contactIdentifier` | Contact group/category | `COM` is authorized for an explicit general party phone/email/URL group because that is the frozen application mapping. `IND` meaning is unresolved and is not annotation-authorized. |
| `documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].contactInformations[].contactInformation.contactName` | Named contact person/department | Require explicit contact context; do not repeat party name by default. |
| `documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].contactInformations[].communicationContact[].communicationMeans` | Communication channel | Use `TE`/`EM`/`AO` below. |
| `documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].contactInformations[].communicationContact[].identifier` | Phone, email, or URL value | Preserve the OCR-supported value; never repair an address from the image. |

Communication channels:

| Code | Meaning |
|---|---|
| `TE` | Telephone |
| `EM` | Electronic mail |
| `AO` | URL / web address under the selected UN/EDIFACT vocabulary |

A contact group requires both `contactInformation` and at least one `communicationContact`. Do not
emit a contact name alone. Fax is not accepted by this Pydantic version; warn with
`schema_cannot_represent` rather than putting a fax number under `TE`.

### Forwarding and export references

`documentPatch.consignmentInformation.consignmentDetails.forwardingAndExportReferences[].references`
holds ordered OCR text explicitly identified as a forwarding/export reference. Preserve separate
source references as separate rows. Do not newline-join all references and do not place booking,
ACID, tax, invoice, or unrelated tracking identifiers here without the required forwarding/export
context.

## Goods-item construction

A goods item is a source-supported commodity/line grouping, not one item per page or one item per
container. Use headings, table rows, continuation markers, package/weight alignment, and repeated
values to group OCR facts. One commodity spread over multiple containers remains one goods item with
multiple `splitGoodsPlacement` rows when the relationship is supported. Preserve first-appearance
order across pages.

Do not duplicate document-level aggregate package/weight totals into each goods item. If a total is
visible but allocation is not, omit the per-item value and use `aggregate_not_allocated`.

### Packages

| Target path | Meaning | Rule |
|---|---|---|
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].numberAndTypeOfPackages[].packageQuantity` | Count of packages in this package row | Non-negative integer; require row/item association. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].numberAndTypeOfPackages[].packageTypeDescriptionCode` | UNECE Recommendation 21 package code | Emit an exact printed code or one of the frozen phrase mappings below. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].numberAndTypeOfPackages[].typeOfPackages` | Package type as source free text | Safe fallback when code conversion is unclear; maximum 35 printable-ASCII characters. Do not silently truncate a longer source phrase. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].numberAndTypeOfPackages[].packagingRelatedDescriptionCode` | Additional coded packaging characteristic | Lexically open in Pydantic but no application meaning snapshot is present; not annotation-authorized. |

High-frequency package mappings frozen from UNECE Recommendation 21 and the current application
normalizer surface:

| Explicit singular/plural OCR label | Code |
|---|---|
| Pallet | `PX` |
| Package | `PK` |
| Bag | `BG` |
| Fibre/fiber drum | `1G` |
| Drum, material unspecified | `DR` |
| Carton | `CT` |
| Box | `BX` |
| Case | `CS` |
| Crate | `CR` |

Material-specific or qualified labels must not be collapsed to a generic code when that loses a
meaning needed by the source. A bare `BALE` is ambiguous between compressed and non-compressed Rec.
21 values; keep free text unless the qualifier is explicit. A printed valid two-character code may
be copied with contextual evidence.

### Handling instructions

| Target path | Meaning | Rule |
|---|---|---|
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].handlingInstructions[].descriptionCode` | Coded handling instruction | No frozen MPCI list is present; omit unless the exact code and meaning are later supplied in a versioned snapshot. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].handlingInstructions[].handlingInstructionDescription` | Handling text | Preserve an explicit goods-handling instruction such as source-visible handling requirements; do not treat ordinary B/L legal boilerplate as a goods instruction. |

### Goods descriptions and additional information

| Target path | Meaning | Rule |
|---|---|---|
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].freeText[].textSubjectCodeQualifier` | Semantic kind of the text row | `AAA` is goods description; `AAI` is additional/general information about that goods item. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].freeText[].freeText` | OCR-supported text | Preserve the goods text after permitted line joining/ASCII normalization; do not summarize, translate, or correct it. |

Use `AAA` for the actual cargo/nature-of-goods description. Use `AAI` only for additional general
information that is clearly tied to the same goods item and is not a mark, handling instruction,
reference, party, or dangerous-goods field. The fact that `AAA` is also a measurement code for net
weight is not a conflict: code meaning is determined by its schema path.

When joining wrapped OCR lines, keep all source words in order and use a single space. Record the
exact operation as `normalizationRule`; the evidence excerpt itself remains verbatim raw OCR.

### Goods measurements

| Target path | Meaning | Rule |
|---|---|---|
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].measurements[].measuredAttributeCode` | What is measured | Use `AAB` gross weight, `AAA` net weight, or `ABJ` volume only with explicit attribute context. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].measurements[].measurementUnitCode` | Unit | `KGM` kg/kgs/kilogram; `LBR` lb/lbs/pound; `MTQ` cubic metre/meter, `M3`, or `CBM`. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].measurements[].measure` | Positive numeric measure | Remove clear thousands separators and unit text; preserve decimal magnitude. |

| Attribute code | Meaning | Allowed units in this schema |
|---|---|---|
| `AAB` | Gross weight | `KGM`, `LBR` |
| `AAA` | Net weight | `KGM`, `LBR` |
| `ABJ` | Volume | `MTQ` |

The Pydantic type technically permits any cross-product of the three attribute and unit enums, but
semantically invalid pairs such as gross weight in `MTQ` or volume in `KGM` are not authorized.
Container tare and VGM are not ordinary goods measurements. A document-level total must not be
allocated to individual goods without OCR support.

### Goods-to-container placement

| Target path | Meaning | Rule |
|---|---|---|
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].splitGoodsPlacement[].equipmentIdentification.equipmentIdentifier` | Container holding this goods item | Must exactly match one valid emitted `containerInformation` identifier. Cite raw association/grouping evidence. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].splitGoodsPlacement[].equipmentIdentification.packageQuantity` | Package count for this goods item in that container | Emit only when the allocation is explicit. |

Container references within one goods item must be unique. If there is one package-total row and all
placement quantities are present, their sum must equal the total. Do not create an association merely
because a container and goods description occur somewhere in the same document; require table,
heading, continuation, or explicit reference evidence.

### Dangerous goods

Dangerous-goods rows are emitted only for OCR-explicit dangerous-goods facts. A chemical-sounding
description is not enough. Do not use an external UNDG database to complete or correct a record.

| Target path | Meaning | Rule |
|---|---|---|
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].hazardCode[].hazardIdentificationCode` | Main IMDG hazard class | Map an explicitly printed class/division to its parent class `1`-`9`; preserve division/subsidiary details only where separately supported. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].hazardCode[].additionalHazardClassificationIdentifier` | Explicit additional/subsidiary hazard identifier | Do not use as a dumping ground for proper shipping name or UN number. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].undgInformation.identifier` | Four-digit UN dangerous-goods number | Remove an explicit `UN` prefix and separators only; exactly four digits must remain. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].undgInformation.flashpointDescription` | Source flashpoint text associated with the UNDG record | Preserve only an explicitly printed flashpoint description. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].dangerousGoodsShipmentFlashpoint[].shipmentFlashpointDegree` | Numeric flashpoint | Preserve sign and decimal; require explicit flashpoint context. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].dangerousGoodsShipmentFlashpoint[].measurementUnitCode` | Flashpoint unit | `CEL` Celsius or `FAH` Fahrenheit. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].dangerousGoodsShipmentFlashpoint[].packagingDangerLevelCode` | UN/IMDG packing group | Map explicit group `I`/`II`/`III` or `NOT ASSIGNED` with the table below. |

Main IMDG classes:

| Code | Meaning |
|---|---|
| `1` | Explosives |
| `2` | Gases |
| `3` | Flammable liquids |
| `4` | Flammable solids, spontaneously combustible material, or material dangerous when wet |
| `5` | Oxidizing substances and organic peroxides |
| `6` | Toxic and infectious substances |
| `7` | Radioactive material |
| `8` | Corrosive substances |
| `9` | Miscellaneous dangerous substances and articles |

An OCR class such as `4.1` can support parent class `4` with a documented contextual normalization.
Do not manufacture a decimal division when OCR gives only `4`. A separately printed subsidiary risk
can populate `additionalHazardClassificationIdentifier`; an implicit risk cannot.

Packing groups:

| Source | Code | Meaning |
|---|---|---|
| `I` or `1` in explicit packing-group context | `1` | Great danger / Packing Group I |
| `II` or `2` in explicit packing-group context | `2` | Medium danger / Packing Group II |
| `III` or `3` in explicit packing-group context | `3` | Minor danger / Packing Group III |
| Explicit `NOT ASSIGNED` | `4` | No packaging danger level assigned |

Do not map a bare Roman numeral that is not tied to `PACKING GROUP`/`PG`. A proper shipping name has
no target field in schema `1.0.0`; if useful OCR text cannot truthfully be represented as `AAA`/`AAI`,
surface `schema_cannot_represent` rather than putting it into `flashpointDescription`.

### Marks and labels

| Target path | Meaning | Rule |
|---|---|---|
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].packageIdentification[].markingInstructionCode` | Coded marking instruction | No frozen application code list is present; not annotation-authorized. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].packageIdentification[].marksAndLabels[].shippingMarksDescription` | Shipping marks and numbers | Preserve OCR text explicitly under marks/numbers context. Do not use container, seal, reference, or goods-description text merely to fill the field. |

`marksAndLabels` is required when a `packageIdentification` row exists. Therefore, do not create a
row containing only an unresolved marking code.

### HS/customs-goods identifiers

`documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].customsStatusOfGoods.customsIdentityCodes.customsGoodsIdentifier[].value`
is an HS/customs commodity identifier for the containing goods item.

Rules:

- Require explicit `HS`, `H.S. CODE`, `HARMONIZED CODE`, `TARIFF CODE`, or equivalent commodity-code
  context.
- Remove only OCR-present dots and spaces; the normalized value must contain 6-18 digits.
- Preserve source order and uniqueness within the goods item.
- Do not correct or pad digits using the image or external tariff tables.
- Do not place ACID, VAT, booking, B/L, container, seal, invoice, tax, product, or customer reference
  numbers here.
- If a candidate remains malformed, omit it and use `invalid_identifier_omitted`.

### Goods location

| Target path | Meaning | Rule |
|---|---|---|
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].locationOfIdentification[].locationOfIdentificationQualifier` | Goods-item location role | Always literal `27`, but emit it only as part of an OCR-supported goods-location row. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].locationOfIdentification[].locationIdentifier` | Printed goods-location identifier/text | Require an explicit goods-origin/location association. It is not automatically a UN/LOCODE. |
| `documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].locationOfIdentification[].locationName` | Printed goods-location name | Optional source-supported display name; do not infer from identifier. |

Qualifier `27` means the location associated with the goods item in this application contract. It is
not permission to treat a party country, loading port, `MADE IN` phrase, or generic origin claim as
the field without matching goods-location context.

## Annotation envelope, provenance, and evidence

The annotation envelope is an audit artifact. Only `label.canonical_target()` becomes the training
target. Source provenance, evidence, warnings, and review notes remain outside model input/output.

### Fixed envelope values

| Field | Required value/meaning |
|---|---|
| `annotationSchemaVersion` | Literal `1.0.0` |
| `taskType` | Literal `mpci_cuscar_kie` |
| `documentType` | Literal `bill_of_lading` |
| `label.schemaVersion` | Literal `1.0.0` |
| `reviewStatus` | Worker uses `candidate`; only overseer uses `validated`; `needs_review`/`rejected` are review outcomes. |

### Source provenance

`source` is not a labeling decision. Copy it mechanically from the immutable work item and assert
exact structural equality before publication. Never retype or recompute its identifiers, paths, page
count, or hashes in a worker.

The source contract guarantees:

- `documentId` is `doc_` plus the lowercase source SHA-256;
- `documentPageCount` is positive;
- `pages` is complete and ordered by contiguous zero-based `pageIndex`;
- `pageNumber = pageIndex + 1`;
- page and extraction IDs are unique inside a document;
- hashes are lowercase 64-character SHA-256 strings; and
- every retained raw response, page raster, source, and joined OCR input is traceable.

### Raw-to-mapped representation

Keep originals and conversions in the same per-document annotation:

| Location | Contents |
|---|---|
| `label.documentPatch` at `evidence[].targetPath` | The mapped/normalized value used as the training label. |
| `evidence[].rawOcrEvidence[].rawValue` | Exact source value as written in raw OCR before mapping or normalization. |
| `evidence[].rawOcrEvidence[].pageNumber` | Source page containing that exact raw value. |
| `evidence[].rawOcrEvidence[].ocrExcerpt` | Verbatim surrounding OCR context containing `rawValue`, used to prove field meaning and grouping. |
| `evidence[].normalizationRule` | Exact deterministic rule connecting the original value to the mapped value. |

`rawValue` is mandatory even for verbatim fields. It must occur byte-for-byte as a substring of its
paired `ocrExcerpt`; the Pydantic model enforces this. If a target is assembled from several raw
strings, include one `rawOcrEvidence` record per contributing string. If a cross-page selection uses
both a damaged candidate and an intact repetition, retain every materially used candidate and state
which OCR-present value was selected in `normalizationRule`/`note`.

Do not add a duplicated `mappedValue` inside evidence. The mapped value already exists at
`targetPath`, and the annotation validator guarantees one evidence record for exactly that leaf.
This single-source representation prevents the evidence copy of a mapped value from drifting away
from the actual training label.

### Evidence kinds

Every non-null leaf beneath `documentPatch` has exactly one evidence record. An object or array does
not get its own evidence; its scalar children do. In evidence paths, replace documentation `[]`
wildcards with actual zero-based indexes, for example
`documentPatch.containerInformation[0].equipmentIdentification.equipmentIdentifier`.

| `evidenceKind` | Use when | `normalizationRule` |
|---|---|---|
| `verbatim` | Emitted scalar exactly equals the cited OCR value | Must be `null`/omitted. |
| `normalized` | Deterministic lexical conversion: date format, separator removal, line join, number/unit stripping, permitted ASCII normalization | Required; state the exact transformation. |
| `contextual_code` | Heading/phrase is converted to a structural or categorical code | Required; name the source phrase and frozen mapping. |
| `cross_page_resolution` | A repeated intact raw value on another page resolves a damaged/conflicting occurrence | Required; state which OCR-present candidate was selected and why. |

`rawOcrEvidence` records must be ordered by page number and must be unique. Both `rawValue` and
`ocrExcerpt` are raw strings: never put a cleaned target value in either field. The excerpt must
contain its raw value verbatim and enough surrounding text to establish context, especially for
codes. The overseer additionally verifies both strings against the declared page's raw OCR.

Parsing OCR characters into a JSON number is `normalized` even when the visible digits are unchanged.
Date formatting, case conversion, separator removal, unit conversion, line joining, and every
phrase-to-code mapping are likewise non-verbatim. Reserve `verbatim` for string leaves whose emitted
string exactly matches the cited raw OCR string without lexical or semantic conversion.

### Image-use values

| `imageUse` | Meaning |
|---|---|
| `not_used` | Worker did not use PDF/raster to make this field decision. |
| `grouping_only` | Image clarified row/block/entity grouping; all target characters remain in OCR. |
| `candidate_disambiguation` | Image helped choose between candidates that both occur in OCR; chosen target still occurs in OCR. |
| `discrepancy_check` | Image was checked because OCR looked suspect; it did not supply a correction. Usually accompanies omission/warning rather than an emitted corrected value. |

Using an image never changes what values are permitted. Set `imageUse` honestly per leaf; do not
mark the whole annotation `not_used` if visual grouping affected a field.

### Precise normalization-rule examples

Good rules are deterministic and reproducible:

- `Parsed unambiguous OCR date "12 AUG 2026" as YYYY-MM-DD.`
- `Removed OCR-present "UN" prefix from UNDG identifier; retained four OCR digits.`
- `Removed spaces from OCR container candidate; validated unchanged characters with ISO 6346 check digit.`
- `Mapped explicit party heading "SHIPPER" to frozen partyFunction CZ.`
- `Mapped explicit "FREIGHT PREPAID" to basic-freight category 4 and arrangement P.`
- `Joined two OCR-wrapped description lines with one ASCII space; no characters added or corrected.`

Bad rules are vague or conceal inference: `cleaned value`, `standardized`, `fixed OCR`, `looked up
port`, `obvious country`, or `corrected from image`.

## Warnings and omission policy

Warnings document an ambiguity or omitted fact. They never insert, repair, or override a target.
Use concrete page numbers and a target path when there is a semantically matching target. Use
`targetPath = null` when the schema has no matching field.

| Warning code | Use when | Typical action |
|---|---|---|
| `aggregate_not_allocated` | A document total cannot be truthfully assigned to goods/containers | Omit per-entity value; cite aggregate pages. |
| `ambiguous_ocr_candidates` | Multiple OCR-present candidates cannot be resolved | Omit contested field or entity. |
| `image_only_value_omitted` | Image shows a value absent from raw OCR | Omit value; never transcribe it. |
| `invalid_identifier_omitted` | Container/IMO/HS/UNDG or other candidate fails its exact constraint after permitted OCR-only normalization | Omit invalid value and dependent relation/row as needed. |
| `schema_cannot_represent` | OCR contains a genuine fact with no semantically matching Pydantic field | Omit; use `targetPath = null`, e.g. ACID or unsupported fax. |
| `unsupported_or_unclear_code` | Field exists but categorical mapping is missing, unresolved, or ambiguous | Prefer source free-text sibling when one exists; otherwise omit code/row. |
| `other` | A material issue fits none of the above | Explain precisely; do not use as a generic substitute. |

Do not add warnings for every absent optional field. Absence is normal. Warn when there was a
material OCR candidate or document fact that was deliberately omitted, an ambiguity affects label
quality, or review is needed.

## End-to-end examples

### Explicit basic freight

Raw OCR:

```text
FREIGHT PREPAID
```

Target fragment:

```json
{
  "chargePaymentInstructions": [
    {"chargeCategory": "4", "paymentArrangement": "P"}
  ]
}
```

Both leaves need `contextual_code` evidence citing the raw phrase. Category `4` means basic freight;
`P` means prepaid.

The same annotation therefore contains both sides of the conversion:

```json
{
  "label": {
    "schemaVersion": "1.0.0",
    "documentPatch": {
      "consignmentInformation": {
        "consignmentDetails": {
          "chargePaymentInstructions": [
            {"chargeCategory": "4", "paymentArrangement": "P"}
          ]
        }
      }
    }
  },
  "evidence": [
    {
      "targetPath": "documentPatch.consignmentInformation.consignmentDetails.chargePaymentInstructions[0].chargeCategory",
      "evidenceKind": "contextual_code",
      "rawOcrEvidence": [
        {
          "pageNumber": 1,
          "rawValue": "FREIGHT PREPAID",
          "ocrExcerpt": "FREIGHT PREPAID"
        }
      ],
      "imageUse": "not_used",
      "normalizationRule": "Mapped explicit FREIGHT PREPAID to basic-freight charge category 4."
    },
    {
      "targetPath": "documentPatch.consignmentInformation.consignmentDetails.chargePaymentInstructions[0].paymentArrangement",
      "evidenceKind": "contextual_code",
      "rawOcrEvidence": [
        {
          "pageNumber": 1,
          "rawValue": "FREIGHT PREPAID",
          "ocrExcerpt": "FREIGHT PREPAID"
        }
      ],
      "imageUse": "not_used",
      "normalizationRule": "Mapped explicit FREIGHT PREPAID to payment arrangement P."
    }
  ]
}
```

### Package and measurement

Raw OCR:

```text
120 CARTONS     GROSS WEIGHT 1,250.50 KGS
```

Authorized target facts are quantity `120`, package code `CT`, measurement attribute `AAB`, unit
`KGM`, and measure `1250.5`, provided table grouping ties them to the goods item. Each leaf receives
its own evidence. The normalized number rule removes the OCR grouping comma and unit text; it does
not change magnitude.

### Invalid container with image-only check digit

Raw OCR contains `MSCU 123456`; the image appears to show a final digit. The schema requires a valid
11-character ISO 6346 identifier. Do not append the image digit, do not emit the container, and do
not emit child seals/placements that would require that invented key. Add
`image_only_value_omitted` and/or `invalid_identifier_omitted` as applicable.

### Cross-page repetition

Page 1 OCR has `B/L NO ABCI23`; page 2 OCR repeats `B/L NO ABC123`. If the second value is clearly the
same document field, `ABC123` may be selected because it is present in raw OCR. Use
`cross_page_resolution`, cite both pages if both drove the decision, and state that the intact
page-2 repetition was chosen. Never replace it with an image-only third spelling.

### Unsupported identifier

Raw OCR contains `ACID: 1234567890123456789`. ACID is not an HS code and has no target in this
Pydantic schema. Omit it and add `schema_cannot_represent` with `targetPath = null`. Do not place it
under customs goods identifiers or forwarding references merely to retain it.

## Schema-valid but annotation-prohibited summary

The following are deliberately closed until an application-owned, versioned list is supplied:

| Field/code surface | Current policy |
|---|---|
| `powerTypeCode` values `7`-`9` | Do not emit; verified standard snapshot defines only `1`-`6`. |
| `serviceRequirementCode` values `65`-`66` | Do not emit; cited directory defines only `1`-`64`. |
| Monetary `typeCodeQualifier` `1`-`550` | Do not select; meanings absent from local snapshot. |
| Charge categories other than `4` | Do not select; only basic freight is frozen. |
| Party `codeListIdentificationCode` `1`/`2` | Do not select; meanings absent. |
| Contact identifier `IND` | Do not select; application meaning absent. Use frozen `COM` mapping only for general party contact. |
| Packaging-related, handling, and marking codes | Do not select; Pydantic is lexically open and local option values are absent. |
| Unlisted package-label mappings | Keep `typeOfPackages` free text or warn; do not guess a Rec. 21 code. |
| Unlisted container shorthand mappings | Keep `equipmentDescription` or exact printed four-character code. |
| Mutable UN/LOCODE, carrier, party, and UNDG lookups | Never query or hydrate inside a worker. |

This table is not a defect workaround. It is the explicit boundary between syntactic acceptance and
supervised semantic truth. Expanding it requires a versioned schema/reference update before a
labeling run begins, not a worker-level decision.

## Appendix A: complete target-leaf inventory

This inventory is a drift guard and navigation aid. `[]` denotes a repeated row in documentation;
evidence paths must use a concrete numeric index. Every leaf currently accepted under
`documentPatch` appears below, and every leaf is explained in the sections above.

```text
documentPatch.billOfLadingIssueDate
documentPatch.shippedOnBoardDate
documentPatch.processingInformation.processingIndicatorDescriptionCode
documentPatch.blIdentifiers.houseBLNumber
documentPatch.blIdentifiers.originalBLNumber
documentPatch.blIdentifiers.parentBLNumber
documentPatch.placeOfBillIssue.locode
documentPatch.placeOfBillIssue.name
documentPatch.placeOfBillIssue.country_code
documentPatch.placeOfFreightPayment.locode
documentPatch.placeOfFreightPayment.name
documentPatch.placeOfFreightPayment.country_code
documentPatch.voyageDetails.transportInformation.meansOfTransportJourneyIdentifier
documentPatch.voyageDetails.transportInformation.carrierIdentifierFreeText
documentPatch.voyageDetails.transportInformation.transportMeansIdentificationNameIdentifier
documentPatch.voyageDetails.transportInformation.transportMeansIdentificationName
documentPatch.voyageDetails.transportInformation.transportMeansNationalityCode
documentPatch.voyageDetails.transportInformation.transportMeansOwnershipIndicatorCode
documentPatch.voyageDetails.transportInformation.powerTypeCode
documentPatch.voyageDetails.transportInformation.powerTypeDescription
documentPatch.voyageDetails.departureAndArrivalPorts.portOfDeparture.locode
documentPatch.voyageDetails.departureAndArrivalPorts.portOfDeparture.name
documentPatch.voyageDetails.departureAndArrivalPorts.portOfDeparture.country_code
documentPatch.voyageDetails.departureAndArrivalPorts.portOfArrival.locode
documentPatch.voyageDetails.departureAndArrivalPorts.portOfArrival.name
documentPatch.voyageDetails.departureAndArrivalPorts.portOfArrival.country_code
documentPatch.voyageDetails.timeOfArrivalAndDepartures.estimatedTimeOfArrival
documentPatch.voyageDetails.timeOfArrivalAndDepartures.estimatedTimeOfDeparture
documentPatch.voyageDetails.timeOfArrivalAndDepartures.actualTimeOfDeparture
documentPatch.containerInformation[].equipmentIdentification.equipmentIdentifier
documentPatch.containerInformation[].equipmentSizeAndType.containerSizeAndType.containerCode
documentPatch.containerInformation[].equipmentSizeAndType.equipmentDescription
documentPatch.containerInformation[].equipmentSizeAndType.fullOrEmptyIndicatorCodes
documentPatch.containerInformation[].transportServiceRequirements[].serviceRequirementCode
documentPatch.containerInformation[].transportServiceRequirements[].natureOfCargo.cargoTypeClassificationCode
documentPatch.containerInformation[].containerVerifiedGrossMass.measure
documentPatch.containerInformation[].containerVerifiedGrossMass.measurementUnitCode
documentPatch.containerInformation[].sealNumbers[].transportUnitSealIdentifier
documentPatch.containerInformation[].sealNumbers[].sealingPartyNameCode
documentPatch.containerInformation[].sealNumbers[].sealingPartyName
documentPatch.containerInformation[].sealNumbers[].sealType
documentPatch.containerInformation[].temperatureSettings[].temperatureTypeCodeQualifier
documentPatch.containerInformation[].temperatureSettings[].temperatureDegree
documentPatch.containerInformation[].temperatureSettings[].unitCode
documentPatch.consignmentInformation.consignmentDetails.monetaryAmount[].typeCodeQualifier
documentPatch.consignmentInformation.consignmentDetails.monetaryAmount[].amount
documentPatch.consignmentInformation.consignmentDetails.monetaryAmount[].currencyIdentificationCode
documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].locationOfIdentificationQualifier
documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].locationIdentifier.locode
documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].locationIdentifier.name
documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].locationIdentifier.country_code
documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].locationIdentifierCountry
documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].locationName
documentPatch.consignmentInformation.consignmentDetails.locationOfIdentification[].firstRelatedLocationName
documentPatch.consignmentInformation.consignmentDetails.chargePaymentInstructions[].chargeCategory
documentPatch.consignmentInformation.consignmentDetails.chargePaymentInstructions[].paymentArrangement
documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].partyFunction
documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].partyIdentifier
documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].codeListIdentificationCode
documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].partyNames[].name
documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].addresses[].address
documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].city
documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].country
documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].contactInformations[].contactInformation.contactIdentifier
documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].contactInformations[].contactInformation.contactName
documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].contactInformations[].communicationContact[].communicationMeans
documentPatch.consignmentInformation.consignmentDetails.partiesInformation[].contactInformations[].communicationContact[].identifier
documentPatch.consignmentInformation.consignmentDetails.forwardingAndExportReferences[].references
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].numberAndTypeOfPackages[].packageQuantity
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].numberAndTypeOfPackages[].packageTypeDescriptionCode
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].numberAndTypeOfPackages[].typeOfPackages
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].numberAndTypeOfPackages[].packagingRelatedDescriptionCode
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].handlingInstructions[].descriptionCode
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].handlingInstructions[].handlingInstructionDescription
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].freeText[].textSubjectCodeQualifier
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].freeText[].freeText
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].measurements[].measuredAttributeCode
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].measurements[].measurementUnitCode
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].measurements[].measure
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].splitGoodsPlacement[].equipmentIdentification.equipmentIdentifier
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].splitGoodsPlacement[].equipmentIdentification.packageQuantity
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].hazardCode[].hazardIdentificationCode
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].hazardCode[].additionalHazardClassificationIdentifier
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].undgInformation.identifier
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].undgInformation.flashpointDescription
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].dangerousGoodsShipmentFlashpoint[].shipmentFlashpointDegree
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].dangerousGoodsShipmentFlashpoint[].measurementUnitCode
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].dangerousGoods[].dangerousGoodsShipmentFlashpoint[].packagingDangerLevelCode
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].packageIdentification[].markingInstructionCode
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].packageIdentification[].marksAndLabels[].shippingMarksDescription
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].customsStatusOfGoods.customsIdentityCodes.customsGoodsIdentifier[].value
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].locationOfIdentification[].locationOfIdentificationQualifier
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].locationOfIdentification[].locationIdentifier
documentPatch.consignmentInformation.consignmentDetails.goodsItemDetails[].locationOfIdentification[].locationName
```

## Appendix B: pre-publication worker checklist

- Exactly one complete multi-page work item was labeled.
- Raw OCR, not PDF/image text, is the target boundary.
- No image-only correction or field was introduced.
- Every emitted field has the exact semantic meaning documented here.
- Every categorical value is explicitly authorized here and supported by context.
- No schema-valid-but-unresolved code was guessed.
- Arrays preserve source order; consignment locations use `9,12,13,88,7,96` order.
- Container identifiers and IMO numbers pass their checksums.
- Every placement references an emitted container exactly.
- Aggregate values were not duplicated or allocated without evidence.
- The sparse patch contains no empty object, empty placeholder row, or system/UI field.
- Every emitted leaf has exactly one concrete-index evidence path.
- Every `rawValue` and paired `ocrExcerpt` is verbatim OCR from the declared page; original values
  remain unconverted, evidence stays in page/source order, and normalization rules are precise.
- Source provenance is mechanically identical to the work item.
- Strict Pydantic validation succeeds with `reviewStatus = "candidate"`.
