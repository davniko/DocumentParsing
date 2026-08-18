# Bill-of-Lading semantic KIE target v2

Status: implemented for a bounded four-document remediation test. The 15-document r3 pilot and
its v1 MPCI-wire annotations remain immutable historical evidence.

## Decision

The training target should be a sparse **semantic Bill-of-Lading patch**, not a sparse copy of the
MPCI/CUSCAR form tree. A deterministic, versioned projector converts the semantic patch into the
application form. This does not reintroduce the historical ai-genie mapping layer: the projector
only performs fixed structural expansion, fixed enum translation, explicit reference expansion,
sanitation, and caller-owned code-list lookup. It does not classify free text, infer missing facts,
geocode, allocate cargo, or silently drop unsupported values.

The executable contracts are:

- `src/document_ocr/label_schemas/bill_of_lading.py`: v2 decoder target, annotation sidecar, and
  fail-closed exclusion record;
- `src/document_ocr/label_schemas/mpci_projection.py`: semantic-v2 to frozen sparse-MPCI-v1
  projection;
- `src/document_ocr/label_schemas/mpci_bill_of_lading.py`: retained v1 MPCI-wire contract; and
- `BILL_OF_LADING_LABELING_REFERENCE_V2.md`: annotation semantics and content-purity rules.

## Confirmed problems in the r3 target

### Physical OCR lines were modeled as logical addresses

The v1 reference explicitly said to preserve ordered OCR address lines in `addresses[].address`.
The workers followed that instruction. In the 15 validated pilot records:

- 50 parties had an address;
- those parties produced 105 address rows, or 2.1 rows per addressed party;
- 36 of 50 addressed parties had more than one row; and
- no sampled record demonstrated two distinct logical addresses for one party.

The target was therefore teaching page layout, not address semantics. In v2, a party has one
nullable `address` scalar. Wrapped OCR rows are joined in source order with single spaces. If a
future real document contains multiple distinct addresses for one party, that is a concrete schema
extension; physical line wrapping is never a reason to create another item.

### Mixed lines leaked unrelated data

All 15 raw work items contained at least one TAX/VAT/ACID/CIF-like mention. Fourteen labels omitted
those values. One label copied `TAX ID : 754-375-706` into the consignee address and then copied the
same contaminated address to notify. Another label copied `PH: (48) 3255-1391` into two addresses
while also extracting it as a telephone contact.

The cause was semantic segmentation, not OCR quality: the related and unrelated text occupied one
OCR line. Physical line membership cannot define a target value. The annotator must split a mixed
line at OCR-visible labels/delimiters and retain the exact fragments in evidence. V2 rejects common
tax/contact markers in `address` at schema validation time and the reference applies the same
value-only rule to every field.

### Boilerplate entered marks/cargo fields

One label emitted `SHIPPER'S LOAD & COUNT` as a shipping mark. This is carrier/shipper-load
boilerplate, not a mark. The v2 cargo and marks types reject high-confidence boilerplate patterns.
The prompt also requires semantic review instead of a broad denylist: values such as `N/M`, lot
numbers, hold numbers, and explicitly headed export references may be legitimate facts even when
they look terse or unfamiliar.

### The target spent capacity on path-implied codes and duplicated facts

The 15 canonical v1 targets contained 974 scalar leaves. At least 239 leaves (24.5%) were opaque
structural/code fields. The largest groups were party functions (57), location qualifiers (42),
communication means (39), contact identifiers (26), goods-text qualifiers (25), and measurement
attribute codes (17). Every document also repeated loading/discharge places in both voyage and
consignment branches. Carrier names were duplicated in two branches in five documents.

A conservative count identifies 329 of 974 scalar emissions (33.8%) as avoidable structural,
route/carrier duplication, or address fragmentation. This is a scalar-count diagnosis, not a claim
of an equal token or accuracy gain. Tokenizer-specific size and held-out accuracy still require an
A/B evaluation.

## V2 semantic target

One multi-page source containing one transport document produces one sparse `documentPatch`.
Named object paths carry role semantics, so the decoder does not emit codes whose value follows
from the path.

| V2 group | Semantic content | Deliberately absent from target |
|---|---|---|
| identifiers/dates | B/L, original, master numbers; issue/on-board dates normalized from every explicitly headed OCR date, with ambiguous numeric order documented rather than omitted; date-like invoice/export references retained under their printed role | MPCI identifier wrappers and form defaults |
| negotiability | `negotiable` or `non_negotiable` when explicit | `NEG`/`NON` wire code |
| route | receipt, loading, transshipment, discharge, delivery, destination, each once | qualifier codes and voyage/consignment duplicates |
| transport | vessel, IMO, voyage, flag | form transport wrappers |
| parties | explicitly headed named roles, one logical address, contacts, explicit `sameAs`; signing/issuing agents never reclassified as forwarding agents | party-function and contact codes; repeated same-as payload |
| freight | readable payment arrangement and payment place | fixed charge category and letter code |
| containers | number, source type/code, VGM, seals, setpoint | equipment wrappers and fixed temperature qualifier |
| goods | description, additional information, packages, named measures, explicit allocations, marks, HS, handling, DG, origin | `AAA`/`AAI`, measurement codes, goods-location qualifier |

Null and absent values remain valid at optional positions. Empty child objects and empty arrays are
invalid. The canonical decoder target recursively omits `None`.

## Address and party contract

`parties.shipper`, `parties.consignee`, and the other named properties imply their roles. Up to two
notify parties remain an array because the form has distinct NI/N2 roles. Each concrete party can
contain:

- one name;
- one logical address string, formed by joining only address fragments and excluding a separately
  emitted city/country component;
- a separately supported city and country candidate; and
- one contact object with an optional named person and source-ordered phone, email, and website
  arrays; a grounded name does not require a communication channel to be present.

Tax/VAT/CNPJ/ACID values, phone/email/fax values, role headings, and other labels cannot be embedded
inside the address. The current target intentionally has no tax-ID field because the MPCI target
does not consume it and it created contamination risk.

When city/country cannot be split safely, the label keeps the full logical address and omits the
separate component. It never emits the same component twice. Country/locality values are copied as
printed; geographic codes are not part of the semantic target.

When raw OCR explicitly says `SAME AS CONSIGNEE` or `SAME AS SHIPPER`, a notify item emits only
`{"sameAs":"consignee"}` or `{"sameAs":"shipper"}`. Exact coincidental equality without that
phrase does not authorize `sameAs`. The projector expands the relation for the final form.

## Categorical ownership

Categoricals are divided by ownership rather than treated uniformly.

### Path-implied or fixed values: deterministic projector

The model does not emit these:

- party functions (`CZ`, `CN`, `NI`, `N2`, `CG`, `DDR`, `DP`, `COX`);
- route/location qualifiers (`9`, `12`, `13`, `88`, `7`, `96`, goods `27`);
- contact wrapper/means codes (`COM`, `TE`, `EM`, `AO`);
- goods text qualifiers (`AAA`, `AAI`);
- measurement attributes/units (`AAB`, `AAA`, `ABJ`, `KGM`, `LBR`, `MTQ`);
- fixed basic-freight category `4`; and
- generic set-temperature qualifier `2`.

### Small semantic decisions: readable model enums

The model emits readable values that correspond to explicit document language:

- `negotiable` / `non_negotiable`;
- `prepaid` / `collect` / `third_party` / `payable_elsewhere`;
- `kilogram` / `pound` / `cubic_metre`; and
- `celsius` / `fahrenheit`.

The projector maps them by closed dictionaries. Evidence retains the exact raw wording, so the
mapping remains auditable.

### Large or mutable code lists: source candidate plus resolver

The model copies printed country text or abbreviations, location names, and source type text. It
never emits `countryCode`, UN/LOCODE, or a looked-up geographic value. Even when the source prints
an ISO-like abbreviation, that literal is retained under `country`. A separate post-inference MPCI
projection uses an injected, versioned country resolver. Missing or ambiguous resolution raises
`MpciProjectionError`; there is no silent fallback and projection output is not a training target.

Container and HS identifiers are deterministically normalized only under the frozen lexical rules
and then schema validated. An invalid container check digit is omitted with a warning; it is never
repaired from the image.

## Exact projection map

| Semantic v2 | Sparse MPCI projection |
|---|---|
| `billOfLadingNumber` | `blIdentifiers.houseBLNumber` |
| `originalBillOfLadingNumber` | `blIdentifiers.originalBLNumber` |
| `masterBillOfLadingNumber` | `blIdentifiers.parentBLNumber` |
| `issueDate`, `shippedOnBoardDate` | root application dates |
| `negotiability` | `processingInformation.processingIndicatorDescriptionCode` |
| `placeOfIssue` | `placeOfBillIssue` |
| `freight.paymentPlace` | `placeOfFreightPayment` |
| transport fields | `voyageDetails.transportInformation` |
| route loading/discharge | voyage ports and qualifier rows `9`/`12` |
| remaining route fields | one named consignment qualifier row each |
| named parties | ordered `partiesInformation[]` with fixed function codes |
| notify `sameAs` | resolved referenced party, then NI/N2 row |
| freight arrangement | charge category `4` plus `P/C/B/A` |
| container fields | `containerInformation[]` and fixed nested wrappers |
| `description` / `additionalInformation` | `freeText[]` with `AAA` / `AAI` |
| gross/net/volume | `measurements[]` with `AAB` / `AAA` / `ABJ` |
| packages/allocations/marks/HS/handling/DG/origin | corresponding MPCI goods children |

Duplicated locations and carrier data exist only after projection because the final application
schema requires them. They are not duplicated in supervision.

## Content-purity rule for every scalar

Each target leaf contains only the value belonging to that field. Do not include its field label,
neighboring values, legal clauses, table headings, page metadata, signatures, or explanatory
flavor text. A mixed OCR row must be segmented by the visible label/delimiter even though the
resulting target substring is not the whole row.

High-confidence exclusions include:

- tax/VAT/CNPJ/ACID identifiers from party names and addresses;
- telephone/email/fax values from addresses;
- `SHIPPER'S LOAD & COUNT`, `SAID TO CONTAIN`, generic `S.T.C.`, carrier liability language, and
  package-limitation clauses from descriptions and marks;
- labels such as `AGENCY OFFICE`, `ADDRESS`, `DESCRIPTION OF GOODS`, and `GROSS WEIGHT`; and
- external eBL portal metadata, hashes, audit logs, upload records, and blockchain links.

Do not turn this into a global substring denylist. `N/M` under marks, an explicit lot number, or a
company name under a headed export-reference block can be valid. Unclear segmentation is omitted
and warned for review.

## Multi-document and duplicate policy

The unit of supervision is one transport document, not one PDF. A PDF containing two independent
B/L faces must be split upstream with provenance or rejected. It must never be merged into a single
label. `BillOfLadingExclusion` records the reason and exact raw evidence.

Byte/content duplicates remain separate source records for audit, but a duplicate-group decision
must be recorded before training publication. Labels for genuinely equivalent duplicate faces must
be consistent on shared OCR-supported facts. Identical rasters can still yield different frozen OCR;
an image-only omission remains absent from that record's OCR-conditioned label and is documented,
not copied across. Only one group representative should enter a deduplicated training split.
Portal/audit pages may differ without making the underlying B/L facts different.

## Known v2 scope boundaries

The first v2 contract intentionally omits fields that lack a clean, currently needed MPCI mapping:
number of originals, Incoterms, arbitrary monetary type qualifiers, stand-alone dangerous-goods
packing group without flashpoint, proper shipping name, and the 64-value service-requirement list.
These facts remain visible in OCR and may be noted as `schema_cannot_represent`; they must not be
forced into neighboring fields. A later extension requires real examples, an exact application
path, frozen semantics, adapter coverage, and tests.

## Four-document remediation acceptance criteria

The bounded rerun covers the four prior anomalies:

1. the two HMM eBL portal exports, which are a duplicate B/L face with different audit pages;
2. the Turkish/Egyptian B/L whose v1 address and marks contained tax/boilerplate leakage; and
3. the four-page CONGENBILL PDF containing two independent B/L faces.

Acceptance requires:

- schema-valid candidate or fail-closed exclusion for every source;
- exact raw-OCR evidence for every emitted target leaf;
- no tax/contact leakage, carrier boilerplate, physical-line address arrays, or explicit same-as
  payload duplication;
- semantically consistent core labels for the duplicate pair;
- rejection of the multi-B/L source rather than a merged label;
- successful deterministic projection for accepted candidates using explicit country resolution;
  and
- a measured comparison of canonical target leaves/bytes against v1 without claiming training
  quality from serialization size alone.
