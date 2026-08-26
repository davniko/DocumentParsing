# Bill-of-Lading semantic KIE labeling reference v2

This is the frozen semantic authority for `schemaVersion = "2.0.0"`. The executable schema in
`src/document_ocr/label_schemas/bill_of_lading.py` is authoritative for shape and types; this file
defines meaning, admissible mappings, exclusions, and evidence behavior that types cannot express.

## 1. Truth boundary

The page-ordered raw GLM-OCR text is the sole source of target facts. Retained images/PDFs may help
identify headings, columns, row associations, and page continuation, but cannot add or correct any
target character or fact.

For every non-null target leaf:

1. cite one `FieldEvidence` with the exact concrete `targetPath`;
2. retain every contributing exact OCR substring in `rawOcrEvidence[].rawValue`;
3. put each raw value inside a verbatim `ocrExcerpt` from the same page;
4. record a precise `normalizationRule` unless the emitted scalar equals the cited raw value; and
5. keep entries inside each `FieldEvidence.rawOcrEvidence` sequence in page/source order.

The outer `evidence[]` array follows canonical target traversal and has no global OCR-order
contract. Do not reorder independent `FieldEvidence` records merely because their cited values
occur elsewhere on the page in a different order.

An `ocrExcerpt` is one exact contiguous substring of the cited raw-OCR page. Never reconstruct an
excerpt by skipping intervening headers/columns, and never place image-visible text in evidence.

If the image is clearer than OCR, do not correct OCR. If another OCR page contains the intact value,
the intact OCR value is admissible with cross-page evidence.

## 2. Semantic atomicity: values, not physical lines

A target scalar contains only the semantic value of its field. OCR rows are layout artifacts, not
target boundaries.

If OCR says:

```text
11 extension of abdelhamid badawy st, cairo, egypt TAX ID : 754-375-706
```

the admissible address is:

```json
{"address":"11 extension of abdelhamid badawy st, cairo, egypt"}
```

The exact raw address fragment and, if useful, the full mixed row stay in evidence. The tax value is
not part of any current target field.

If OCR says:

```text
IMBITUBA - SC - CEP 88780-000 - PH: (48) 3255-1391
```

the address ends before `- PH:` and the phone belongs only in `contactDetails.phoneNumbers`.

Apply the same rule to every field. Exclude headings, labels, neighboring column values, tax text,
contact text, legal clauses, signatures, portal metadata, and explanatory flavor text. Do not
silently discard an uncertain *meaningful transport fact*: omit its target and add a warning.
Routine tax/regulatory, portal/audit, administrative/security, upload, filename, and blockchain
metadata is excluded silently unless it creates a genuine ambiguity for a supported target.

## 3. Text and normalization

Application text must be one non-empty printable scalar with no leading/trailing whitespace or
embedded newline. Letters must use the Latin script; preserve printed Latin diacritics exactly.
Printable numbers, punctuation, symbols, and spaces remain allowed. This does not mean copying an
entire OCR row.

Allowed deterministic transformations, with exact raw evidence retained:

- join wrapped semantic fragments in page order with one space;
- collapse OCR whitespace introduced by wrapping;
- preserve OCR-printed Latin Unicode, including diacritics, without transliteration or correction;
- normalize every explicitly headed issue/on-board date to `YYYY-MM-DD`; whenever both leading
  numeric components are 12 or below, use the frozen dataset convention directly: first component
  is day and second is month (`01/04/2024` -> `2024-04-01`; `12.07.2024` -> `2024-07-12`). This rule
  applies to the ambiguous token itself even if another date in the same document establishes a
  different numeric order. Never infer order from country, port, language, party nationality, or
  another geographic locality. Preserve the exact printed form in evidence and never omit or hold
  a document solely for this resolvable numeric-date ambiguity;
- normalize named-month dates directly, including ordinal forms such as `MAR 4TH 2025` to
  `2025-03-04`; these are unambiguous and do not require locale inference;
- preserve a date-like value under the semantic role stated by its OCR heading; for example,
  `INV.NO: 29/03/2024` yields forwarding/export reference `29/03/2024`, not an issue/on-board date;
- remove OCR-present separators from a supported identifier/number;
- parse an explicit numeric measure while preserving sign/decimal and citing its unit;
- map an explicit semantic phrase to a frozen readable enum; and
- preserve printed country/locality wording without lookup, expansion, or code mapping.

Forbidden transformations include image correction, spell correction, geocoding, inventing dates
that are absent from raw OCR, reclassifying an invoice/rate/manufacture/expiry date as an issue or
on-board date,
inventing missing identifier digits/check digits, inferring unstated units, or resolving ambiguous
countries/locations.

## 4. Root paths

### Identifiers and dates

| Target | Meaning | Rule |
|---|---|---|
| `documentPatch.billOfLadingNumber` | Primary B/L or sea-waybill number | Require an explicit B/L-number heading or unmistakable transport-document identifier context. Do not use portal EBL references, upload IDs, hashes, filenames, or generic reference numbers. |
| `originalBillOfLadingNumber` | Explicit original B/L identifier | Do not confuse number-of-originals with an identifier. |
| `masterBillOfLadingNumber` | Explicit master/parent B/L identifier | Require the relationship label; do not infer from number format. |
| `issueDate` | Date of issue | Require issue-date context; split combined place/date evidence correctly. Do not omit an explicitly printed date solely because numeric day/month order needs a documented interpretation. |
| `shippedOnBoardDate` | On-board/shipped date | Require explicit on-board context. Do not omit an explicitly printed date solely because numeric day/month order needs a documented interpretation. |

Number of originals and generic reference numbers have no v2 target. Warn only when their omission
matters to coverage; never place them in a B/L-number field.

### Negotiability

`documentPatch.negotiability` uses readable values:

| Value | Required evidence |
|---|---|
| `non_negotiable` | Explicit `NON-NEGOTIABLE`, `SEA WAYBILL`, equivalent document wording, or a completed named non-order consignee that makes the B/L a straight bill. |
| `negotiable` | Explicit negotiability language or an unambiguous `TO ORDER` consignee construction authorized by this reference. |

Conditional wording such as `NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER` must be resolved against
the actual consignee construction. A `non_negotiable` target based on that condition requires a
named consignee that is not a `TO ORDER` construction, and the leaf evidence must cite both the
conditional wording and the named consignee. If the consignee is explicitly `TO ORDER`, emit
`negotiable` instead. If the consignee construction is absent from raw OCR, omit the category.

A completed named consignee that is not a `TO ORDER` construction establishes a straight,
non-negotiable B/L. Generic preprinted positive-original, surrender, one-accomplished/others-void,
or `delivered unto order or assigns` boilerplate does not turn that completed straight consignment
into an order bill. A `NON-NEGOTIABLE COPY` stamp remains copy status rather than the deciding
evidence: cite the completed named consignee (and any applicable conditional term) for the target.
An explicitly completed `TO ORDER` consignee remains `negotiable`.

`EXPRESS BILL OF LADING` is authorized equivalent document wording for `non_negotiable`.
Do not map a bare use of `EXPRESS` in unrelated release, service, or shipping text. This
interpretation follows Maersk's shipping glossary, which defines an Express B/L as a sea waybill
that cannot be negotiated or transferred to a third party.

Do not default every B/L to `non_negotiable`. Absence is null. The projector owns `NON`/`NEG`.

### Place of issue

`documentPatch.placeOfIssue` uses `SemanticLocation`. If one line contains place and date, emit the
place only and cite the exact place fragment.

## 5. Locations and route

Each semantic location supports:

- `name`: printed locality/port/place name;
- `country`: the OCR-printed country wording or abbreviation, copied without lookup or expansion.

Do not infer country from city, vessel route, address, or general knowledge. If OCR prints
`BUSAN, KOREA`, `name = "BUSAN"` and `country = "KOREA"` are learnable copy targets. The country
resolver later maps `KOREA` to an application code. If OCR itself prints only `KR`, preserve `KR`
under `country`; do not expand it. `countryCode` and `unLocode` are not semantic-v2 fields.

Route paths are named and appear once:

| Path | Source role |
|---|---|
| `route.placeOfReceipt` | Place of receipt |
| `route.portOfLoading` | Port of loading |
| `route.transshipmentPort` | Explicit transshipment port |
| `route.portOfDischarge` | Port of discharge |
| `route.placeOfDelivery` | Place of delivery |
| `route.finalDestination` | Explicit final destination |

Do not duplicate loading/discharge under another target branch. The projector creates the MPCI
voyage and qualifier views.

## 6. Transport and carrier

| Path | Meaning | Rule |
|---|---|---|
| `transport.vesselName` | Vessel name only | Remove a clear vessel-type prefix such as `M/V` or `MV`; retain an attached voyage token only if OCR/layout cannot separate it, and warn. |
| `transport.vesselImoNumber` | Seven-digit IMO number | Require explicit IMO context and a valid checksum. |
| `transport.voyageNumber` | Voyage identifier | Require a voyage heading/marker; separate from vessel name when OCR supports it. |
| `transport.vesselFlagCountry` | Explicit printed vessel flag/nationality country text | Copy the source wording; do not infer from carrier/route or map to a code. |

Carrier is represented once as `parties.carrier`, not repeated in `transport`. A name under
`SIGNED AS AGENT FOR THE CARRIER` is not automatically the carrier; distinguish carrier from agent
using the OCR headings.

## 7. Parties

Named fields imply the final role:

| Semantic field | MPCI role owned by projector |
|---|---|
| `shipper` | CZ |
| `consignee` | CN |
| first/second `notifyParties[]` | NI/N2 |
| `carrier` | CG |
| `forwardingAgent` | DDR |
| `deliveryAgent` | DP |
| `consolidator` | COX |

Role headings control assignment. Emit `forwardingAgent` only from an explicit forwarding-agent
heading or equally unambiguous wording. A party named as signing/issuing agent for the carrier,
agent at origin, or agent on behalf of the carrier is not a forwarding agent; when no v2 role fits,
omit it with a warning. Emit `deliveryAgent` only for an explicit destination/discharge/delivery
agent, never merely because an agent address is present.

### Party value rules

- `name`: legal/trading name only; exclude role heading, identifier label, address, contact, and
  descriptive sentence.
- `address`: one logical address, with source-ordered wrapped fragments joined by one space.
  Retain printed postal/ZIP values because v2 has no separate postal field, but remove field labels
  such as `POSTAL CODE`, `POST CODE`, `POSTCODE`, and `ZIP CODE`.
- `city`: only an explicitly separable city value; do not geocode or infer.
- `country`: OCR-printed country wording or abbreviation; do not infer, expand, or code-map it.
- `contactDetails.contactName`: named person, without `CONTACT:`. A grounded contact name is valid
  even when the OCR contains no phone, email, or website for that person.
- phone/email/website arrays: values only, without `TEL:`, `EMAIL:`, `FAX:`, or similar labels.

A standalone `FAX` value has no semantic-v2 target: omit it and add a page-bound
`schema_cannot_represent` warning. A single value explicitly shared by a `TEL/FAX` heading may be
kept as a phone number because the OCR itself also identifies it as telephone contact.

Never include tax ID, VAT ID, CNPJ, ACID code, customs number, phone, email, URL, or fax in an
address. Never include an address or tax text in a party name. A tax identifier currently has no
target path and must be omitted.

Avoid duplicating city/country in two target fields. When a city or country is confidently
separable and emitted separately, remove that exact trailing component from `address`. If it cannot
be separated safely from the logical address, keep it in `address` and omit the separate field.
All party geography uses `country`; `countryCode` is not a semantic-v2 field.

### Explicit same-as relation

When OCR explicitly states `SAME AS CONSIGNEE` or `SAME AS SHIPPER`, emit only:

```json
{"sameAs":"consignee"}
```

or:

```json
{"sameAs":"shipper"}
```

Do not copy the referenced party fields. Do not infer `sameAs` merely because two blocks happen to
contain equal text.

## 8. Freight

`freight.paymentArrangement` uses readable values:

| Value | Explicit source concept | MPCI projection |
|---|---|---|
| `prepaid` | freight prepaid | category 4 + P |
| `collect` | freight collect | category 4 + C |
| `third_party` | third-party payment | category 4 + B |
| `payable_elsewhere` | explicitly payable elsewhere | category 4 + A |

`freight.paymentPlace` is an explicitly headed prepaid/payment place. Do not treat any issue or
route place as payment place without source context.

`AS ARRANGED`, Incoterms, rate dates, and arbitrary total-freight amounts are not payment enums.
The current v2 target intentionally omits Incoterms and monetary type qualifiers.

## 9. Containers

Every container item requires a valid ISO 6346 `containerNumber`. Normalize only OCR-present
spaces/hyphens and require a valid check digit. Never repair from the image.

| Path | Meaning | Rule |
|---|---|---|
| `typeDescription` | Source type wording | Preserve useful text such as `20'DC`; omit service boilerplate such as `CY/CY`. |
| `typeCode` | Four-character canonical code | Copy an exact printed code; a separately frozen mapping is required for phrase conversion. |
| `verifiedGrossMass` | Explicit VGM | Ordinary cargo gross weight belongs to goods. |
| `sealNumbers[]` | Seal identifiers | Value only, explicitly associated with this container. |
| `temperatureSetpoint` | Explicit transport setpoint | Require set-temperature/reefer context and explicit unit. |

Container-service and cargo-class codes are not v2 targets. In particular, `SHIPPER'S LOAD &
COUNT` is normally boilerplate and does not authorize a service-requirement label.

## 10. Goods

One `goodsItems[]` item is one source-supported cargo grouping. Do not create separate items merely
because description wraps across lines. Do not merge independent B/Ls.

### Description and additional information

`description` is the core goods/product description, with separately modeled values removed. Do
not prefix it with gross weight, package count, `S.T.C.`, or load/count boilerplate. For example:

```text
60,000KG FERRO MOLYBDENUM
NET WEIGHT: 60,000KG
```

maps to description `FERRO MOLYBDENUM` and net weight 60000 kg. It does not map to description
`60,000KG FERRO MOLYBDENUM`.

`additionalInformation[]` is for explicit cargo-specific description that is useful to the MPCI
goods record and is not already represented elsewhere. It is not a dumping ground for legal text,
trade terms, document metadata, totals, or headings.

### Packages

Each `packages[]` row is one semantic package level/type:

```json
[
  {"quantity":60,"type":"PALLETS"},
  {"quantity":60,"type":"BAGS"}
]
```

This is preferred to the combined string `60 PALLETS (60 BAGS)`. Do not allocate nested package
counts across containers unless OCR states the allocation. `typeCode` is allowed only for an exact
printed or frozen deterministic UNECE code; `type` is the safe source-text form.

An explicitly partitioned outer-pallet schedule is a supported normalization, not an either/or
package choice. When source-ordered headings such as `Pallet No. 1 - 4`, `Pallet No. 5 - 6`, ...
form one contiguous, non-overlapping range beginning at pallet 1 and reconcile exactly to the
printed pallet total for one identified container:

- each headed pallet-range/product row is one cargo group;
- emit the inclusive range cardinality as that group's outer `PALLET` package quantity;
- emit the row's drums, tinplate containers, cartons, or other receptacles as a second, inner
  package level in the same group;
- attach the row's explicitly printed net and gross weights to that group;
- link only the outer pallet level to the identified container with `single_package_level`; and
- do not also emit the document-wide pallet total as another package fact, because it is the exact
  sum of the same outer pallets and would duplicate them.

For example, `Pallet No. 5 - 6: 8 Drums ...` yields outer quantity `2` `PALLET` plus inner quantity
`8` `Drums`. The evidence sidecar retains the full printed range and states the inclusive-range
normalization. Do not synthesize a containment sentence in `additionalInformation`, and do not
mistake the row's per-receptacle net wording for the cargo group's aggregate `netWeight`. This
narrow rule is invalid if ranges overlap, leave gaps, reset numbering, fail to reconcile to the
printed total, or span multiple containers without explicit container linkage.

### Named measures

- `grossWeight` and `netWeight` accept `kilogram` or `pound`;
- `volume` accepts `cubic_metre`;
- a value must be positive, explicitly labeled, and retain its source unit in evidence; and
- do not treat container VGM as goods gross weight or vice versa.

Do not arithmetically sum per-row or per-container measures into an absent cargo-group aggregate.
Each emitted scalar must normalize from one printed scalar; a directly printed total remains valid.

When the same value occurs both in a table total and description, use the clearest explicit
occurrence and cite all occurrences only when they contribute to cross-page resolution.

### Container allocations

`containerAllocations[]` is emitted only when OCR explicitly links a goods item to a known emitted
container. It repeats the container identifier solely as a relational foreign key. If one goods
item and all containers are globally listed but no allocation is stated, omit allocations; the
application may apply a deterministic single-item relation policy separately.

When a goods item contains a package quantity and also contains package quantities in its container
allocations, every allocation must carry a quantity and their sum must exactly cover either one
emitted package level or the total of multiple emitted package levels. This supports both nested
package levels (allocate the source-linked level only) and explicitly additive package categories.
Never emit a partial allocation, borrow a total from another cargo level, or guess a remainder to
make the projection validate. Split source-supported per-container goods rows into separate items
when that is the actual document grouping; otherwise omit the unsupported allocation and warn when
the omission matters. If OCR flattens a cargo table, use the retained raster only to identify which
OCR values share a row/column; a raster-only value remains absent from the target.

### Marks and numbers

Keep actual shipping marks such as `N/M`, lot numbers, or marked reference strings when headed and
clearly associated. Exclude `SHIPPER'S LOAD & COUNT`, `SAID TO CONTAIN`, generic `S.T.C.`, table
headers, and liability disclaimers. Absence of marks is null, not a fabricated `N/M`.

### HS codes

Require explicit HS/customs-tariff context. Remove only printed punctuation/spaces and require 6–18
digits. Deduplicate while preserving source order. Do not treat VAT, tax, ACID, CNPJ, PO, or portal
hash values as HS codes.

### Handling, dangerous goods, and origin

- handling instructions require explicit cargo-handling context, not generic legal conditions;
- UN number requires explicit UN/DG context and four digits;
- hazard class requires explicit class context;
- flashpoint requires explicit degree and unit; packing group is retained only inside the current
  flashpoint structure because that is the form surface this v2 projector can map; and
- goods origin uses an explicit printed origin name/identifier, never an inferred country.

## 11. Forwarding and export references

`forwardingAndExportReferences[]` contains value-only strings under an explicit forwarding/export
reference heading. A company name can be a valid reference if the document actually places it in
that block. An explicit inline cargo label such as `SHIPPING BILL`, `S/BILL`, `EXPORT REF`, or
`INV.NO` is also qualifying semantic context; a blank generic form block elsewhere does not cancel
the later labeled value. Exclude the heading/label itself and explicit empty placeholders such as
`NO REF`, `N/A`, `NONE`, or `NIL`. Do not repurpose unlabeled booking,
control, portal-audit, or blockchain identifiers.

## 12. Portal/audit and legal pages

An eBL export may append portal metadata, document hashes, upload details, possession-transfer
logs, blockchain URLs, timestamps, and platform actors. These are not transport-document facts and
must not enter the label. A page can still provide a legitimate carrier contact block or trade term
if it is visibly part of the B/L, but the value must map to a supported target path.

Carrier conditions, package-limitation clauses, copyright text, Hague/Hague-Visby clauses, and
similar boilerplate are not goods, marks, parties, references, or handling instructions.

## 13. One PDF versus one transport document

Before labeling, count independent transport-document faces using repeated titles, identifiers,
parties/routes, and cargo blocks. Repeated legal-condition pages do not create another document.

If one PDF contains two independent B/Ls, do not select one silently and do not merge them. Produce
`BillOfLadingExclusion` with `reason = "multiple_transport_documents"` and exact evidence showing
the independent faces. The PDF must be split upstream before either face can become training data.

## 14. Duplicate sources

Workers label each assigned source independently, but the overseer checks duplicate clusters. If
the B/L face is the same while portal/audit pages differ, shared OCR-supported semantic facts must
agree. The targets may differ when the frozen raw OCR differs: an image-visible value omitted from
one OCR input must remain absent from that input's label even if a duplicate OCR run captured it.
Record such asymmetric OCR support explicitly; never harmonize labels from the image. Differences
caused only by treating portal metadata as document facts are errors. Preserve both source
annotations for audit, then select one representative during dataset deduplication.

## 15. Warning decisions

Use existing warning codes narrowly:

- `ambiguous_ocr_candidates`: multiple OCR-supported candidates cannot be resolved;
- `image_only_value_omitted`: image shows a fact absent/corrupt in OCR;
- `invalid_identifier_omitted`: printed/normalized identifier fails validation;
- `schema_cannot_represent`: legitimate OCR fact is outside the frozen v2 target;
- `aggregate_not_allocated`: only an aggregate is known and per-item allocation is absent;
- `unsupported_or_unclear_code`: source phrase does not justify a frozen code; and
- `other`: only when none of the above applies, with a precise message.

Warnings never add target data. Do not warn for every ordinary absent field or every ignored legal
clause.

## 16. Pre-submission checklist for a worker

Before atomically writing a candidate:

1. Confirm the PDF contains exactly one eligible maritime B/L, Sea Waybill, or equivalent maritime
   multimodal/combined transport document with a sea leg; otherwise write an exclusion. Air, road,
   truck, CMR, rail, invoices, packing lists, unknowns, and generic non-maritime multimodal documents
   remain outside scope.
2. Confirm every target character/fact is supported by page-ordered raw OCR.
3. Confirm every address is one logical scalar and contains no tax/contact data.
4. Confirm all contacts occur only in contact fields and all tax identifiers are omitted.
5. Confirm cargo descriptions/marks contain no load-count, S.T.C., said-to-contain, legal, or portal
   boilerplate.
6. Confirm roles, location qualifiers, contact codes, text qualifiers, measurement codes, and fixed
   charge/temperature codes do not appear in the semantic target.
7. Confirm explicit same-as is a relation rather than a copied party.
8. Confirm repeated B/L facts across pages were resolved from OCR, not image correction.
9. Validate JSON with `BillOfLadingAnnotation.model_validate_json(..., strict=True)` or validate the
   exclusion with `BillOfLadingExclusion`.
10. Confirm evidence paths equal all and only canonical target leaves and every excerpt occurs
    verbatim on the cited page.
