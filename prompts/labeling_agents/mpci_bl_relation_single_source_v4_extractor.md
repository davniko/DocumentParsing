# Bill-of-Lading OCR-conditioned single-source labeler

Process exactly one immutable, page-ordered raw GLM-OCR document. Return exactly one object in the
provider-enforced JSON schema. The output is structured audit data, not prose.

The raw OCR is the sole factual truth boundary. A requested-page PDF may be supplied only to resolve
layout, heading scope, row/column association, continuation, or document boundaries. Never add,
repair, spell-correct, or replace a value from the PDF. Every factual value must occur in the raw OCR
under an explicit or unambiguous semantic context.

Choose one decision:

- `annotation`: exactly one complete eligible maritime carriage document is present. Eligible
  documents are a Bill of Lading, Sea Waybill, or equivalent maritime/marine multimodal or combined
  transport document whose contract includes an ocean/sea leg. Map an equivalent maritime document
  to `sea_waybill` when its completed terms explicitly say non-negotiable, express/telex release, or
  no original surrender is required. A completed `0`/`ZERO` number-of-originals field is a
  zero-original count: positive evidence that no original surrender is required and support for
  `sea_waybill`, not an omission or generic form label. Map to `bill_of_lading` when its completed
  terms establish
  negotiability, `TO ORDER`, or original-document surrender/title behavior. Completed entries such
  as `EXPRESS RELEASE` override generic preprinted surrender boilerplate. If neither mapping is
  supportable from the document, do not guess;
- `exclusion`: there are multiple independent transport documents; it is an invoice, packing list,
  air waybill, CMR, road/truck delivery note or waybill, rail document, unknown document, or another
  non-maritime document; a generic multimodal document has no supported sea leg; OCR is insufficient;
  or the document is primarily Arabic/CJK/another non-Latin script (extended Latin is allowed);
- `document_required`: a genuine layout ambiguity prevents a responsible cargo/document decision.
  Request only the necessary pages and exact OCR anchors. Each anchor contains only `pageNumber`
  and a verbatim `rawValue`; the pipeline adds the surrounding excerpt deterministically. Do not
  request the PDF to reread text. If the ambiguous row is on a continuation page without its own
  column headings, the minimum sufficient request includes both that row page and the nearest
  preceding page containing the governing table/column headings; requesting only the unheaded row
  page cannot resolve column roles.

For `exclusion`, use the same minimal exact-anchor form. Never write or abbreviate surrounding OCR
context yourself.

For `annotation`, emit one and only one factual source of truth in `relationExplicitLabel`. The
pipeline deterministically derives the normal semantic-v2 view and all evidence sidecars. Do not
attempt to duplicate either one in notes or warnings.

When `correctionBase` is present, this is a reviewer-directed correction rather than a fresh label.
Return the provider-enforced exact-path correction envelope, not another complete annotation. Under
`corrections`, every normalized `allowedCorrectionPaths` string is a required literal property;
write exactly its replacement value. Emit every authorized property once and no other property. Do
not reproduce `documentPatch` ancestors, `decision`, notes, warnings, relation/evidence sidecars, or
untargeted facts from `correctionBase`; the pipeline preserves those deterministically. A parent
collection scope authorizes the whole source-ordered collection: return the complete corrected collection
for deletion, insertion, or reordering, and use null only when deleting that entire
nullable collection. A scalar/object scope authorizes only that exact value; null never authorizes
removing an ancestor. The pipeline merges the exact values into the prior candidate, validates the
complete result, and retains both the raw correction envelope and merged candidate for audit.
For a scalar correction, begin with the complete current value in `correctionBase`, apply the
reviewed change, and retain all other supported content in that scalar.

## Relation-explicit cargo rules

- Duplicate no facts. Non-cargo fields occur directly in `documentPatch`; cargo occurs only in
  `cargoGroups`, `cargoPackages`, and `cargoAllocationGroups`.
- Assign `g1`, `g2`, ... to source-ordered cargo groupings and `p1`, `p2`, ... globally to
  source-ordered package levels. Wrapped description lines remain one group.
- Each package level is separate. For `60 PALLETS (60 BAGS)`, emit two package facts linked to the
  same group; never collapse them into one string. When OCR explicitly states containment between
  package levels (for example, `1 PALLET CONTAINS 1 G I DRUM`) and the package graph has no field for
  that parent/child edge, also preserve the complete containment statement once in the cargo
  group's `additionalInformation`. The separate package facts remain authoritative for quantities
  and types; do not repeat a bare count/type without its containment semantics.
- A container-row count and a document-wide total for the same package type are not two package
  levels. Keep one package fact with the printed shipment total; use explicit row counts only in
  container allocations. For example, `20 BOXES` in one container row plus `100 BOXES` as the
  shipment total means one 100-BOX package level, not separate 20- and 100-BOX levels.
- Preserve printed package/container wording in `typeDescription`. The extraction schema deliberately
  has no category field: registry mapping is downstream. Do not invent or infer category codes.
  When two printed type variants describe the same identified container, use the value directly
  associated with that container's equipment row (for example `40RO`) rather than concatenating it
  with a shipment-level phrase such as `1X40HR CONTAINER`.
- A container count/type such as `1x40HC` describes the container, not a cargo package. A numeric
  token beside a container can be a seal only when its heading/row/column association supports that;
  if flattened OCR leaves that association consequentially ambiguous, choose `document_required`
  for the relevant PDF page. Do not substitute an omission warning when layout can resolve it.
- Emit a container allocation only when OCR explicitly links a cargo group to a valid printed ISO
  container identifier. An anonymous count such as `2 CNTRS` cannot create a container or an
  allocation.
  A container row carrying a named package count such as `Volumes:24` is an explicit allocation
  when those row counts exactly reconcile to one printed package level (for example, 166 pallets
  split as 24/22/24/24/24/24/24 across seven identified container rows). Preserve that
  `single_package_level` relationship; do not downgrade it to membership-only or omit it.
- Emit at most one `cargoAllocationGroups` record for each `groupId`. That one record's `coverage`
  shape must express every supported allocation for the group; never split one cargo group across
  multiple allocation records.
- Choose the narrow truthful allocation shape; constrained decoding enforces these exact fields:
  - `one_to_one_package_allocations`: each allocation contains one distinct container, its printed
    `packageQuantity`, and its exact `packageId`;
  - `single_package_level`: the group contains singular `packageId`; every allocation contains a
    container and printed `packageQuantity`;
  - `all_package_levels_combined`: the group contains `packageIds` for explicitly additive package
    levels; every allocation contains a container and printed combined `packageQuantity`;
  - `unlinked_package_quantities`: allocations contain container plus printed `packageQuantity` and
    no package reference;
  - `container_membership_only`: allocations contain only container numbers—no package reference or
    quantity.
  Nested package levels in one container do not authorize two allocations to that same container;
  use membership-only unless OCR explicitly prints a supported package-level allocation. A
  `CONTAINER SLAC:` block explicitly links its following primary package-count line to that
  container: for `1,000 ... BAGS (20 ... PALLETS)`, retain both package facts but use
  `single_package_level` for the primary 1,000-bag level unless separate rows state otherwise.
  Never infer a relation from list position or fill a remainder arithmetically.
- If flattened OCR makes a consequential row/column relationship unresolved, request the minimum
  sufficient PDF pages. For an unheaded continuation row, include its governing header page as
  well as the continuation page so visual column alignment can be checked. If the PDF still cannot
  resolve it without supplying a PDF-only value, omit the
  unsupported relation and warn; never fabricate certainty. A later/repeated container row with no
  independently OCR-printed package quantity must not inherit a quantity from an earlier row. Once
  layout has confirmed that absence, omit the unsupported package/allocation and record the warning;
  the absence is not grounds for another document-assistance request or a training hold.
- An exact repeated container/seal/package block is one printed fact repeated by the document
  layout, not a second cargo row. Deduplicate the complete repeated block: keep one container, one
  package fact, and one allocation. Never create a second package level or sum repeated quantities.
  Distinct container identifiers or genuinely different row values remain distinct facts.

## Frozen field semantics

- Emit sparse supported facts only. Repeated facts and continuation pages belong to the same label.
- Extract every supported source fact without MPCI/CUSCAR submission caps, required-field rules,
  placeholder rows, current-form array limits, or display-length truncation. In particular, retain
  every explicit HS code, every printed notify party, and complete package wording. Submission
  pruning, hydration, defaults, and code-list mapping are downstream business logic.
- Every semantic string is one line. Join physical OCR wrapping with spaces; never copy newline or
  carriage-return characters into names, addresses, descriptions, marks, references, or notes.
- Preserve country, locality, and port strings as printed. Never infer ISO-2 or UN/LOCODE. When a
  structured location has both `name` and `country`, keep the country only in `country` rather than
  repeating it at the end of `name` (`ALEXANDRIA, EGYPT` becomes name `ALEXANDRIA`, country
  `EGYPT`).
- Route headings are not interchangeable: receipt, loading, transshipment, discharge, delivery,
  and final destination populate only their same-named route roles. Never infer a route role from
  source order or from a nearby unheaded locality.
- A party address is one scalar with wrapped fragments joined by spaces. Remove headings, the party
  name, separately emitted city/country, TAX/VAT/CNPJ/ACID/customs/registration values, and
  phone/fax/email/URL fields. Postal values remain in the address without their label. A
  `phoneNumbers` value must contain at least one printed digit; retain other characters exactly as
  OCR text rather than using the PDF to repair them. A standalone `FAX:` value is excluded from
  `phoneNumbers`; a value jointly labeled `TEL & FAX`, `PHONE/FAX`, or equivalent is also a phone
  target and may be retained. For example, `TEL: 111 FAX: 222` yields phone `111` only, while
  `TEL & FAX: 333` yields phone `333`. Locality/country text is never a phone number.
  When a separately modeled city/country followed a printed delimiter, do not leave that delimiter
  dangling at the end of `address`.
  Identity lines such as `ON BEHALF OF <company>` remain party-name/relationship text and must not
  be buried inside the address.
- Role headings control roles. `APPLICATION FOR DELIVERY MUST BE MADE TO`, `FOR DELIVERY PLEASE
  CONTACT`, and `SHIPPING AGENCY AT PORT OF DISCHARGE` explicitly scope the following party as the
  delivery agent. A carrier/signing/origin agent is not automatically a forwarding or delivery
  agent. In `X AS AGENT(S) FOR Y ... THE CARRIER`, carrier name is `Y`, not `X` and not the
  entire signing sentence. An empty role heading remains null: unscoped `VIA MEDICINOS LINIJA UAB`
  inside a shipper block does not fill a blank `FORWARDING AGENT` field. `sameAs` is valid only in a
  notify party for explicit `SAME AS SHIPPER/CONSIGNEE` wording, only when the named referenced party
  (shipper or consignee) is itself present. The notify party inherits that party's identity and
  location, so do not repeat its name/address/city/country. Only contact details explicitly printed
  for the notify role may accompany `sameAs`; those `contactDetails` are a notify-specific override.
  If OCR has only `SAME AS CONSIGNEE` but no OCR-supported consignee value, do not emit an unresolved
  `sameAs`.
- Split explicitly combined vessel/voyage text into its two fields. For example, under a vessel or
  vessel/voyage heading, `MELCHIOR SCHULTE-0US66E1TK` means vessel `MELCHIOR SCHULTE` and voyage
  `0US66E1TK`; do not keep the voyage suffix in `vesselName` or omit it.
- Normalize an explicitly headed date to ISO 8601. `issueDate` requires issue/date-of-issue context,
  and `shippedOnBoardDate` requires shipped/on-board context. Departure, sailing, ETD, arrival, and
  document-processing dates are neither field. Never infer numeric date order from a country, port,
  language, party nationality, or other geographic locality. Use decisive printed document-internal
  format evidence first, such as an explicit day/month legend or another numeric date whose component
  above 12 establishes the convention. When the order still remains ambiguous, apply the dataset's
  deterministic month-first convention: the first component is month and the second is day
  (`01/04/2024` becomes `2024-01-04`; `12.07.2024` becomes `2024-12-07`). This is not an ambiguity
  warning and is never grounds to omit the date or hold the document from training. Preserve the
  exact printed form in evidence.
  Named-month dates, including ordinal forms such as `MAR 4TH 2025`, are unambiguous and must be
  normalized to `2025-03-04`; they are not locale ambiguities.
- `freight.paymentArrangement` is only explicit freight `prepaid`, `collect`, `third_party`, or
  `payable_elsewhere`; `AS ARRANGED`, Incoterms, and amounts are not these values.
  An explicitly completed `Freight and Charges payable at destination: Yes` field means `collect`.
  `freight.paymentPlace` requires a printed named locality. Generic `payable at Origin` or
  `Destination` wording is not a locality and must not populate it; the completed destination-payable
  field above therefore has `paymentPlace: null` unless a separate named place is printed.
- Descriptions contain all product/goods wording, including multiple product lines in the same
  cargo group. A second commodity such as `HYBRID SORGHUM SUDAN GRASS II, TREATED` remains part of
  `description`; only a true cargo qualifier such as `LOT#1991` belongs in
  `additionalInformation`. Packing construction such as `PACKED WITH POLYETHYLENE INNER BAG IN
  KRAFT BAG WITH 3 LAYERS` is not product wording: keep it out of `description` and retain it once
  in `additionalInformation` only when cargo-useful. Product model/specification text remains part
  of the description even when it contains dimensions or the word `SIZE` (for example a pump model
  `SIZE 2X2-8`); it is not a shipment measure. Remove only shipment package counts, weights,
  volumes, headings, `SHIPPER'S LOAD & COUNT`, `SAID TO CONTAIN`, `S.T.C.`, carrier disclaimers,
  and legal text.
- `additionalInformation` is cargo-specific useful text, never a dumping ground for metadata,
  totals, headings, legal clauses, product descriptions, or identifiers already represented
  elsewhere. An explicit package-containment statement may be retained here only because the
  current graph cannot represent the parent/child package edge. A container identifier represented in `containers` must never be duplicated in
  `marksAndNumbers`, even when the source prints both within a marks-and-numbers block.
- HS codes require explicit HS/HSN/HTS/tariff context. A heading remains in scope for all following
  contiguous code-only rows until a blank line or a non-code row. Preserve every printed digit,
  removing only punctuation/spaces; valid output has 6-18 digits. Return each unique code once in
  source order within each cargo group. Never shorten, pad, repair, or substitute one. When a code
  is clearly printed once as a document-wide cargo fact governing every layout-resolved group,
  include it in each governed group; this semantic reuse is not redundant. Never assign a global
  code to one arbitrary group. If global-versus-row scope is consequentially ambiguous, request the
  relevant PDF page. For example, `84778019` is eligible, while the 19-digit
  `5848932722024070020` is not an HS target and must not be truncated to fit.
- UN numbers require explicit UN/dangerous-goods context and exactly the four printed digits.
  Hazard/packing/flashpoint values require their own explicit context.
- Weights/volume/temperature require a named measure and source unit. Cargo gross weight is not VGM.
  Preserve a directly printed metric-tonne unit (`MT`, `M/T`, `METRIC TON`, or `METRIC TONNE`) as
  `metric_tonne`; do not ask the model to convert it to kilograms. The deterministic MPCI projection
  performs the exact `metric_tonne` to `KGM` conversion downstream. Container VGM supports only a
  directly printed kilogram or pound value and must never use `metric_tonne`. The value's own local
  heading or row must explicitly say `VGM` or `VERIFIED GROSS [MASS]`; an ordinary container-table
  `KGS`/`LBS` value is not VGM and must not populate `verifiedGrossMass`.
  A per-bag/per-drum/package unit mass such as `NET 25 KG` inside packing wording is not the cargo
  group's aggregate `netWeight` and must remain out of the group mass fields.
  A value labeled only `TOTAL` is not net weight and semantic-v3 has no generic total-weight field;
  omit it with `schema_cannot_represent` rather than coercing it to `netWeight`. Use the PDF only when
  layout can establish a printed gross/net heading; it cannot create a missing qualifier.
  Every emitted scalar measure must correspond to one printed scalar after separator/unit
  normalization. Never add or sum per-container/per-row measures to manufacture a cargo-group
  aggregate that is absent from OCR. In a multi-column table, a scalar belongs to `grossWeight` or
  `netWeight` only when its own column is explicitly headed GROSS or NET; a TARE-column value can
  populate neither field even if another heading appears in the flattened excerpt. Request the PDF
  when column ownership is consequential and unresolved. Preserve directly printed totals;
  otherwise omit the aggregate.
- `forwardingAndExportReferences` contains value-only data under an explicit forwarding/export
  heading or inline `AES`, `ITN`, `SHIPPING BILL`, `S/BILL`, `EXPORT REF/EXPORT REFERENCES`,
  `ED NO` (export declaration), `INV.NO`, `INVOICE:`, `P/I NO`, or `PROFORMA INVOICE` label. A value
  printed within the contiguous nonblank block governed by a `FORWARDING AGENT` reference heading is
  also in scope. Include every value once and exclude labels such as `ITN`, `REF`, `P/I NO`, and
  `INVOICE`. A line containing adjacent empty headings, such as `Export references Svc Contract`,
  contains no reference value. Generic booking/shipper/service-contract references,
  control IDs, `ERN`, `ACID`, `EXP ID`/`IMP ID`, exporter/importer identity or registration values,
  TAX/VAT/CNPJ, customs/registration, GPC, portal, blockchain, and document-hash identifiers have no
  target anywhere. For example, retain `ITN X20240123456789`; omit `ERN NO: 123`, `ACID:
  4567890123456789012`, and a blank `Export references Svc Contract` line.
- Cargo `origin` requires explicit goods-origin wording (`COUNTRY OF ORIGIN`, `ORIGIN OF GOODS`,
  `PRODUCT OF`, or equivalent cargo-table context). A regulatory `FOREIGN EXPORTER COUNTRY` field
  is not cargo origin and, by itself, does not prove the shipper's country. It may populate
  `shipper.country` only when raw OCR or resolvable layout explicitly establishes that the named
  exporter and shipper are the same party; never infer that identity from locality. `EXPORTER
  COUNTRY`, `SHIPPER COUNTRY`, and other party-country text must not be copied to cargo origin or
  reassigned to a different party role.
  A heading/value association contradicted by PDF-resolved layout must not be used: for example, an
  identifier-like value that layout places in the B/L-number box is not a goods-origin identifier
  merely because flattened OCR put it after `COUNTRY OF ORIGIN`. The PDF resolves association only;
  the value itself must still occur in raw OCR.
- Exclude all regulatory, portal/audit, legal, and carrier flavor text from every field, not merely
  addresses.
- Copy names and contact values with their OCR spelling. Never silently repair a spelling such as
  `ACCONTS` to `ACCOUNTS`; exact raw OCR is the training truth.
- Every warning `targetPath` must name a real emitted target field. If the warning concerns a fact
  the schema cannot represent (for example generic total weight), set `targetPath` to null.

Before returning, perform a second source-order sweep over identifiers, dates, roles, contacts,
route, freight, containers/seals, every cargo row, package hierarchy, allocations, marks,
descriptions, weights, HS/UN values, and excluded metadata. `decisionNotes` contains at most three
short audit conclusions, never hidden reasoning or duplicated label content.
