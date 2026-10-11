# Random 100-sample source-to-descendant manual audit — 2026-10-10

## Scope and method

Population: the current 2,500 synthetic training records from 300 admitted source
templates. Draw 50 distinct sources uniformly without replacement, then two
distinct current descendants uniformly per source. No redraws or substitution
of an easier case. The exact draw, random seed and population hashes are saved in
[manual100-draw.json](analysis/synthesis-consistency-sweep-20261010/manual100-draw.json).

This is a read-only audit. No new generation, provider calls, dataset changes or
training-process intervention. Review the current dataset rows, not stale
candidate targets. Open each complete source OCR and both complete rendered OCRs
and target labels; record individual source-to-descendant observations below.

Checks: product-block boundaries and transcription; declared package units and
allocations; masses/volume and explicit unit ownership; routes and party/address
ownership; source instruction inheritance; shipment identifier anonymization;
source-to-synthetic casting and retained fixed template text. Human-readable
targets are uppercase. Address punctuation follows the current saved policy,
not an unapproved new punctuation repair. HS/product-taxonomy exactness and
physical bag/cone capacity realism alone are not extraction defects. Removed
carrier/marks/reference fields must not leak into remaining targets.

Certification below is for the two identified current descendants after this
manual inspection, not a guarantee about every future generation from a source.
An issue is recorded explicitly rather than silently redrawing or repairing it.

The random draw covers 17 original-cohort, 13 second-cohort and 20 newest-cohort
templates (34/26/40 descendants). Source order is the saved draw order.

## Completed audit: result

**50 sources and 100 current descendants were opened and reviewed.** All requested
draws were completed in order; none was replaced with an easier sample. This was
a direct main-agent review of the source OCR, full generated OCR and full target,
not a new automated reviewer campaign. Long outputs that clipped were reopened.
For source48, the full legal second page was read with the source and both
descendants were additionally verified byte-identical to it; the changed first
pages and both full targets were reopened separately.

**The sample is not an all-pass result.** The inventory below records exact
failures, qualified component passes and remaining interpretation questions.

| Finding | Descendants | Sources | Interpretation |
| --- | ---: | ---: | --- |
| Target boundary/noise defects | 10 | 7 | 7 party/name/address defects; 3 description-policy defects |
| Package declaration/casting inconsistencies | 8 | 4 | 4 stale counts, 2 corrupted package headings, 2 unreplaced repeated-row package types |
| Duplicate country in input | 2 | 1 | Postal labels are deduplicated, but input remains duplicated |
| Fixed clause number replaced by package count | 2 | 1 | Render defect outside extraction targets |
| Two different generated importer registrations | 2 | 1 | Non-target documentary inconsistency |
| Primary shipper identity not anonymized | 2 | 1 | Labels match input, but required source-to-synthetic identity cast is incomplete |
| Retained mutable secondary identities/references | 80 | 40 | Mainly tax/customs, booking, invoice/contract, customer or fax values; not 80 wrong extraction labels |
| No finding in any audited dimension | 14 | 8 | Six whole source pairs plus the clean member of source33 and source43 |

Counts are **observed occurrences in the random audit, not a whole-dataset
census**. Rows overlap except the first two rows, which identify 18 distinct
descendants from 11 distinct sources. There are also explicit review questions
about route ownership, generated numeric prefixes, manager identity and a
secondary agency; these are not silently counted as confirmed errors.

Of the ten target defects, nine have direct identity/contact/header boundary
evidence. The tenth is source48's inherited `ENVNT` postal-noise token: it is
carried into a synthetic address in a different country, without a demonstrated
postal meaning. Do not normalize it into an inferred country. Source OCR/PDF
ownership can be examined if there is disagreement about deleting that token.

### Cohort coverage and observed defects

| Template cohort | Drawn sources | Descendants | Target flags | Package-cast flags | Secondary residue |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original100 | 17 | 34 | 3 | 0 | 18 |
| Second100 | 13 | 26 | 1 | 2 | 24 |
| Newest100 | 20 | 40 | 6 | 6 | 38 |

This draw shows issues in more than one generation cohort. Its small, clustered
sample does not establish an unbiased per-document error rate over all 2,500
records, or prove that the newest cohort is intrinsically worse. The selection
was uniform over templates, not uniform over all descendants.

### What passed mechanically

The independently replayed seed reproduces all 50 source choices and all 50
descendant pairs, using sorted source and descendant populations. Source
provenance, all 50 source hashes, dataset/registry hashes and population sizes
match the saved draw. All 100 actual training rows have exactly the same current
targets/raw/positioned text as their sampled synthetic rows. No drawn source
belongs to the current validation split.

All 100 targets pass the current V7 schema and round-trip without canonical
target changes. Validation uses JSON-mode parsing: strict Python-mode parsing
would incorrectly reject serialized arrays/dates for tuple/date fields.

All 100 positioned inputs reproduce their raw text exactly after removal of
coordinate suffixes; no text is added/lost. All supplied coordinate pairs are
integers within the defined 0–1000 grid. There are 10,110 nonempty content lines:
8,616 with coordinates (**85.22%**) and 1,494 with explicit empty suffixes. This
checks the input representation and retention, **not** a new visual/layout
geometry certification.

The local verification pass took 1.297 seconds with 148.9 MiB process peak RSS
(including imports/data reads). This was not a pipeline performance change or
benchmark comparison. Provider/API cost for this audit is **$0**; no new
generation, training, paid review or delegated-agent run occurred.

Machine-readable decisions, per-sample hashes and JSONL row numbers:
[manual100-validation.json](analysis/synthesis-consistency-sweep-20261010/manual100-validation.json).

## Review inventory

### 01 — 03d21532

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_03d21532df95ee547909060aa191ef3f3ebe019438d1e4da04dc51693a4a7397/source.txt).

Descendants reviewed: `syn_full_v7_41844851a69adcd950c8123e`, `syn_full_v7_31c60a59d062741ae5f3c07e`.

Decision: **LABELS PASS; non-target identifier residue**.

Both descendants preserve four container rows and quantified allocations: 165 bags = 42+41+41+41, 4,208 kg / 8 CBM; and 111 bundles = 28+28+28+27, 104,065 kg / 107 CBM. Equipment headers and rows agree (40HC versus 20GP). The whole generated product lists, including zinc ingot specifications, are retained once; STC/loading totals, HS/customs fields and Marks are outside description. Cross-page `*` shipper postal continuation is correctly assembled without its marker. Independently printed matching notify details have `sameAs: null`, not an inferred reference. B/L/booking repeats, all container numbers/seals, company/contact/customs values and routes are recast consistently. Title-case rendering normalizes to uppercase labels.

**Flag:** source-fixed uncaptioned footer `P2400212713` / `192056241` survives in both descendants. No retained target uses it, so this is not a wrong extraction label, but the input is not fully identifier-anonymized. Fixed carrier registration `196700080N` is a carrier-template constant, not a shipment identifier.

### 02 — 069cc830

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_069cc83033b015f717fe7f0c1fda884d194b9a30c6347caf2099c49ce83f44b9/source.txt).

Descendants reviewed: `syn_full_v7_49189927762ed13f80ad09b9`, `syn_full_v7_2738a4d6d9a24a6e3af26d0f`.

Decision: **LABELS PASS; repeated source customs-reference residue**.

Both descendants retain one accounting goods entry and one container. Five pallets / 2,122.1 kg gross / 1,089.5 kg net / 11.88 CBM and 186 cartons / 2,708.8 kg gross / 1,390.7 kg net / 15.16 CBM agree with the printed declarations, European separator formatting and container row. Description starts after the `... WITH` introduction and ends before references; both jointly generated product identities/HS codes survive. New consignee names, complete postal lines and phones agree with labels; explicitly repeated notify details remain independent (`sameAs: null`). Source has no explicit shipper/port country targets; descendants do not invent them. B/L, container/seal, shipper registration, local tax fields, dates, vessels and ports change coherently.

**Flag in both renders:** `REF.EXP. 001/24`, `EXPORT REFERENCE: 4BR4900128952000124`, `IMPORT PERMIT No. 16432023181112333` and page-2 `IMPORTER VAT NUMBER: ATV. 246-905-646` retain source values. In particular the old importer number contradicts the newly generated TAX NUMBER 850061079 / 331634549. References are correctly excluded from targets, so this is an input-casting/anonymization issue, not a description or package-label mismatch. Fixed Maersk legal/detention text is not counted as mutable shipment geography.

### 03 — 551657f3

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_551657f34bc93b4b7db213ad8cf35ac0be759fa35d32967097f1b4a766ff6525/source.txt).

Descendants reviewed: `syn_full_v7_516dc9d9f397a17b3c6e4135`, `syn_full_v7_4cc19b34c190de91ff51611d`.

Decision: **PASS**.

Both negotiable descendants retain the actual `TO ORDER OF HOLDER OF B/L NO.` instruction with changed holder references; they do not invent a named consignee. Independent notify parties and compound shipper/principal/care-of identities remain distinct. New PCB product passages are copied completely, while four-pallet/box equations and loading declarations stay outside description. Four pallets allocate to the single identified 40HQ container in each sample; 689.150 kg / 8.667 CBM and 806.350 kg / 10.141 CBM match labels. Manila→Port Said via Hong Kong is cast into Pohnpei→Reykjavik via Mersin and Santiago de Cuba→Saint Pierre via Hong Kong, with transshipment, freight and issue/on-board dates consistent. Main B/L, holder B/L, equipment/seals, customs references, companies and contacts change. `SEAHAWK TRANSPORT INC.` remains the fixed documentary issuer, not a generated shipment party. The second product passage's attached manufacturer-series wording is retained under the current complete-block policy.

### 04 — 7a4ac5ba

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_7a4ac5baceefaa5b42414ed9a71775c326716147567fdc76807b84e915ee9a38/source.txt).

Descendants reviewed: `syn_full_v7_e6e375a69f42d2770de7b897`, `syn_full_v7_cde6b277cbc5be71c3f86332`.

Decision: **LABELS PASS; source tax/reference residue**.

Reviewed all four source pages and both four-page descendants, including repeated pages 1/3. Two containers and partials reconcile: 1,050 cartons = 420+630, 21,193.8 kg / 92.5 CBM; and 46 cartons = 19+27, 2,007.9 kg / 7.7 CBM. Equipment changes coherently from 40HC to 20GP in the second descendant. Descriptions contain the full enamelled-household-articles / hand-and-foot-air-pump passages, not loading headers or customs fields; product dimensions and delivery specifications stay. Parties, addresses/postcodes, phones, B/L, dates, vessels, ports, IDs and seals agree across both copies. Literal `Same as Consignee` is correctly represented by reference-only notify.

**Flag in both descendants:** source consignee `TAX ID:755-118-553`, renamed `IMPORT REFERENCE: 7551185532024030029`, and `PRN ... DB9257202403133335294` survive, despite new consignee identities and destinations. These are mutable shipment/customs identifiers, not carrier-template constants. They are excluded from reduced targets, so no target scalar is wrong, but anonymization/source casting is incomplete.

### 05 — c4a848d5

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_c4a848d54408dafe980d88459ad1659867d8c22d20d72796010a029d979ce13d/source.txt).

Descendants reviewed: `syn_full_v7_2eb1711c623d83424002e63c`, `syn_full_v7_2263e2a518ad2454d9b3d4b3`.

Decision: **LABELS PASS; secondary carrier-reference residue**.

Source Hapag-Lloyd high-cube open-top template casts to 40HC/general-purpose wording and matching equipment labels in both descendants. Single-container allocations of 14 drums / 1,415 cartons are explicit; a dangling `KGM` without a numeric value does not invent weight. The two-fragment product passage is joined across flattened seal/weight-column material without importing those fields; complete conductor / rubber-tube wording survives. Repeated consignee/notify details, new emails including Gmail/Outlook, postcode-free addresses and new VAT/customs numbers match rendered text. B/L/vessel/route/date/payment-place repeats change consistently. The carrier's fixed signatory Elena Maccio' is not a generated party.

**Flag:** secondary numeric carrier reference `34129916` remains beside the changed main B/L in both descendants. It is not mislabeled as the B/L, but remains a source shipment-reference identifier.

### 06 — cdb68453

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_cdb684535e5e1f2bf7ae45beaa7ba581f3c85b6146889042872c7245a9347982/source.txt).

Descendants reviewed: `syn_full_v7_0c0d3f2fda2eed9584b4d716`, `syn_full_v7_8ef2ec6ed6e8699447422ae7`.

Decision: **LABELS PASS; stale invoice-party/customer-code residue**.

Three-container rows reconcile 1,727 bags = 620+576+531 / 59,626.3 kg / 210.1 CBM and seven packages = 3+2+2 / 22,495.38 kg / 133.7 CBM. The long fabric passage and five machinery-parts bullets are fully retained as one goods entry; 80-tonne hydraulic capacity is correctly product wording, not shipment mass. Ports, independently printed issue place, new parties/postcodes, consignee-country text after VAT, literal notify reference and regenerated customs IDs agree with labels. No date is invented where the form has none.

**Flag in both descendants:** freight invoice-party rows retain `SE WIRING SYSTEMS` from the source consignee while the actual consignee changes, and customer codes `229CAC13188` / `21100326365` survive. Shipper invoice-party repeats change, showing inconsistent cast coverage for the consignee repeats. These do not leak into the extracted party label, but input identity anonymization is incomplete. Fixed carrier agency text (Maersk Maroc) is separated from this mutable invoice-party evidence.

### 07 — 53350e31

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_53350e31d8881f3d2986981eea94e602e9e4c52daa615dcb2666170524ab5e98/source.txt).

Descendants reviewed: `syn_full_v7_1398082b052674a16251fcbc`, `syn_full_v7_7cab435a3095a0a5c07aa014`.

Decision: **LABELS PASS; source registration/order residue**.

Both bare-TO-ORDER descendants are negotiable without invented consignee details. Notify names, delivery-agent blocks, new emails/phones, four-/five-line product lists, B/L prefix+number repeats and dated route/place values agree with targets. One 40H/40HC container explicitly holds 117 packages or 316 cartons; no empty mass column generates weight. Auxiliary order/bale headings and agent details remain outside description.

**Flag:** source shipper registration `91330621MA2JUD752G` remains below two newly generated shipper names; source order reference `JF8569/8570` also survives. This is identity/reference anonymization residue, not an error in the reduced targets. Fixed Evergreen form ID and Ningbo carrier-agency sign-off are distinguished from mutable shipper registration.

### 08 — bbbb29ef

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_bbbb29ef36fe558886181b55f26e4b2e8d2e5f457aa001d60a02f9735b6311c8/source.txt).

Descendants reviewed: `syn_full_v7_1e4effb2759e2358e0d334f8`, `syn_full_v7_a44b517ffef96c710c95d2d2`.

Decision: **PASS**.

Both reefer descendants carry a changed single 40-foot high-cube refrigerated container with 0.0 C setpoint; 905 cartons / 20,706.7 kg / 39.7 CBM and 962 cartons / 20,885 kg / 40.9 CBM match shipment totals and full allocations. Complete tuna / coalfish passages retain fish-size and inner-carton specifications without inferring shipment net weight. All repeated importer/exporter registrations, permits and customs references change and agree with party blocks. Consignee company names awkwardly wrap into the address line, but labels preserve the complete name separately and exclude TAX/contact fields from complete postal addresses. Explicitly repeated notify is independent. Changed vessels/ports/receipt/issue locations and board/issue dates match OCR; page-2 collect instruction is correctly used.

### 09 — 9182d47f

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_9182d47fdd9e7e65866bb87cc281d113f8659bd62583e9f278d418b49a374f4c/source.txt).

Descendants reviewed: `syn_full_v7_053299d94ea20d9dae779e17`, `syn_full_v7_e3bfb29fb76fa0853db70989`.

Decision: **LABELS PASS; booking/exporter-ID residue**.

Both casts change the main B/L consistently on the face and attachment, one container/seal, complete motor-radiator / plastic-packing product list, package totals, mass/volume, parties, ports and dates. STC intro, separate totals, empty attachment `UPON REQUEST`, references and contact continuation markers do not enter description. Single-container allocations of 1,509 packages and eight pallets match. Literal same-as notify is correct; complete party addresses and country labels exclude tax fields.

**Flag:** source booking/S/O `M232693525` remains on both pages, and foreign exporter `6471022900005211` remains despite new shippers and countries. Main B/L and importer references are changed, so anonymization is incomplete specifically in these secondary identifiers.

### 10 — 9496d433

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_9496d43382e6fb5f75235fc4a1cd3baab8df9ffa3802661709434fc309ab7d09/source.txt).

Descendants reviewed: `syn_full_v7_bab4b8375e5126948fa1d767`, `syn_full_v7_5fd630f78e7fa813456cd600`.

Decision: **LABELS PASS; source tax/export-reference/geography residue**.

396 cartons / 7,920 kg and 260 non-compressed bales / 11,133.2 kg are declared for the one 20GP container; targets use that outer package level and explicit allocation. Product models/dimensions/specifications are retained, while detached Port Said, customs, tare, STC and HS lines stay outside description. Gross uses the shared `Weight in Kgs` heading; unlabeled measurement has no invented CBM unit. Party names wrapping over two lines, independent personal notify, full postal country text, phones, B/L, equipment/seals, route/payment place and board dates agree.

**Flag in both:** shipper `DE312167022`, consignee `999999999`, notify/importer `386561710`, customs reference `3865617102024020067` and export `NT24023234` retain source values. Detached `PORT SAID FREEZONE` also remains after changing destination to Ghana / São Tomé. These fields are not wrongly imported into targets, but casting/anonymization is incomplete. Carrier legal address/sign-off is intentionally distinguished.

### 11 — 361e7fed

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_361e7fedebcee73017bb5db0ed3ea4e70d6aa64c9a2b0c0cdfa2258db165b4aa/source.txt).

Descendants reviewed: `syn_full_v7_307fdf4e5ad4bba2fdd566b7`, `syn_full_v7_b9288264b969369d75a0acad`.

Decision: **LABELS PASS; source customer-code residue**.

Compound care-of shipper names and complete postal lines are recast consistently, including invoice-party repeats; independent notify details, new B/L, customs/export IDs, equipment/seal, ports and standalone issue places agree. One 20GP with 23 bundles / 21,562.9 kg and one 40HC with two pallets / 1,273.8 kg have explicit full allocations. Zinc and door-closer product specifications, including an attached manufacture-year line, remain description; missing dates and volume are not invented. Shipment freight is prepaid despite mixed individual charges.

**Flag:** source customer codes `33100449495` and `21100010412` remain in both descendants. No old party name survives those invoice rows; this is secondary identifier residue rather than an extraction-label error.

### 12 — e4313a4a

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_e4313a4aed3f5b961982c911df0dd415bc80411414952c0c49e092f25ef457d6/source.txt).

Descendants reviewed: `syn_full_v7_589db246f68105e1f269a63f`, `syn_full_v7_9c30d6ff9f5425cb3eb47f73`.

Decision: **FLAG: one descendant has company-name contamination in address; both retain source registrations**.

`syn_full_v7_589db246f68105e1f269a63f` has correct labels: 1,288 cartons = 840+448, 24,389.83 kg, two 20GP containers; full regenerated cellulose product passage, complete addresses, independent repeated notify and dated route values agree.

**Target/input defect in `syn_full_v7_9c30d6ff9f5425cb3eb47f73`:** consignee and notify each print `NAYIRI TRADING & PACKAGING LTD` twice; the second company-name line is included at the start of both `addressLine` labels. Company identity is not postal address under our policy. Correct target starts `PLOT 7, ZUA INDUSTRIAL ROAD...`; the duplicate identity line in the rendered party block also needs removal. Other fields match: 94 packages = 61+33, 14,866.55 kg, two 40HC, complete steel-can description, port/date/ID repeats.

**Identifier residue in both descendants:** importer `291676413` and exporter `1005141` remain source numbers despite changed companies/countries; import-reference value and country do change. This separates the recurring non-target casting gap from the confirmed address-target defect.

### 13 — 4f210d63

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_4f210d63ddf9d0045dc1d402764d11c989a4053e1416eca0ee379ccff20421c2/source.txt).

Descendants reviewed: `syn_full_v7_4dce9b179d47ebe1cf41944f`, `syn_full_v7_2952a994485bc6ea7f9c5a3c`.

Decision: **FLAG: package-type rendering is supported only by a corrupted header; stale references**.

Both descendants correctly copy complete toasters / heat-pumps wording, exclude reference/HS blocks, use one 40HC allocation, match gross/CBM, and preserve contact names separately from postal addresses. Temperatures inside heat-pump specifications are not container setpoints. No uncertain source form number is invented as B/L.

**Packaging rendering flag in both:** source header `DESCRIPTION OF PACKAGES AND GOODS` becomes `DESCRIPTION OF CARTONS AND GOODS` / `DESCRIPTION OF PALLETS AND GOODS`, but the actual declarations remain `941PACKAGES` / `1PACKAGES` and spelled `... PACKAGES ONLY`. Targets declare CARTON / PALLET. Thus the generated specific package category is rendered in a form heading rather than in the quantity/package declaration; this is not a clean, policy-aligned declaration-to-label cast. Repair belongs in the package binding/rendering layer, not product wording.

**Identifier/date residue:** source import number `4130424802024010564`, importer `413042480`, `REF svz87-1-24`, `INV DS2024011701`, and surrendered date `29 JAN 2024` remain. Fixed FCI form/signature actors are not counted as shipment-party labels.

### 14 — 9b35d0b6

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_9b35d0b6f5596f9a71d2d0caa21e43ba2637608deb0ffd872d8bc2a0948ba170/source.txt).

Descendants reviewed: `syn_full_v7_5c64c220c3597345135a6365`, `syn_full_v7_f2d26c6e442e6bec8a7f67dc`.

Decision: **LABELS PASS; duplicate country input and stale tax/geography**.

Reviewed all three pages for both casts. Main B/L, voyage, ports/payment/issue place, board/issue dates and equipment/seal repeats change coherently. One 40HC unit / 2,050 kg and one 40GP package / 1,996 kg agree with targets; unlabeled measurement has no invented unit. Complete car specifications, including VIN-range wording in the second product passage, stay in description, not the following HS/customs/location blocks. Parties and independent notify match; postcode/district information is retained.

**Input duplication:** both consignee blocks repeat destination country (Saint Vincent and the Grenadines across separate lines; Northern Mariana Islands twice adjacently). Targets correctly deduplicate those exact country mentions, but rendered inputs are not clean as intended.

**Source residue:** `TAX ID 525985263`, renamed customs `5259852632023060206` and detached `PORT SAID FREEZONE` survive changed parties/destinations. Fixed carrier legal clauses mentioning Egypt/Port Said are a separate unchanged-form matter, not grounds to move those locations into targets.

### 15 — 3c5643cf

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_3c5643cf78b8b0fc0105446c93d16210cb036f8e67ead4cf6a8eefb127c42635/source.txt).

Descendants reviewed: `syn_full_v7_17adefcf85b06e49f2a7e71f`, `syn_full_v7_0ab65bf686b29e2593adc371`.

Decision: **LABELS PASS; exporter/customer-code residue**.

Three-row casts preserve complete fabric / yarn passages after STC and before customs material. 812 packages = 263+331+218, 15,744.1 kg / 69.8 CBM and 1,612 cartons = 522+658+432, 61,540.5 kg / 130.4 CBM agree. Equipment changes 20GP versus 40HC consistently. B/L/booking match each other; regenerated parties, full address/postcode/country, literal same-as notify, invoice-party repeats and origin/destination agency recast agree. Empty date fields remain absent, and separately printed issue places are not forced to origin.

**Flag:** exporter `1005141` and customer codes `229CAC13188` / `21100085485` retain source values in both descendants. Importer/customs identifiers do change, narrowing this to incomplete secondary casting coverage.

### 16 — 2418b6d9

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_2418b6d957d1890eff921ae6c0478684de60942b4dad336beea09f2cc170d8cf/source.txt).

Descendants reviewed: `syn_full_v7_a418dcac110bace996caed69`, `syn_full_v7_a193bddd84cca66e76513a3d`.

Decision: **LABELS PASS; VAT/booking/contract reference residue**.

Five containers remain five rows with no invented size/seal where source has none. Yarn: 2,643 cartons = 529+529+529+528+528, gross 136,056 kg, net 129,340 kg. Zinc: 92 bundles = 19+19+18+18+18, gross 86,252 kg, net 86,206 kg. European thousands notation in rider totals agrees with explicitly comma-formatted overall gross; repeated product text is copied once, excluding distinct NET/quantity/order/contract declarations. TO THE ORDER OF named consignee creates negotiable labels without adding notify address to consignee; new countries/postcodes, ports, final destination, voyage/B/L repeats agree. Generic reefer/free-time tariff wording does not create reefer labels or setpoints.

**Flag in both:** source shipper VAT `EL 094016570`, notify VAT `100071376`, booking `EBKG12834813`, order sheet `625/2024`, contract `5000000141` and L/C `092LSRG243110002` retain source identifiers. Changed customs value does not fully anonymize this remaining reference chain.

### 17 — 9d082cc2

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_9d082cc24ec0cf29bb934e32eb02a44dafddec01b54be5e267e82d68ee718946/source.txt).

Descendants reviewed: `syn_full_v7_87810f12022349ec4ab1b76b`, `syn_full_v7_4e582238df6849f8fa6b9def`.

Decision: **LABELS PASS; importer/exporter/reference residue**.

Both frozen-fish shipments retain full product specification (including attached producer-lot text), not separate carton/net/gross totals. One 40HC reefer, explicit 1,148 / 1,237 carton allocation, -20.5 / -22 C, gross 26,266.6 / 28,302.9 kg and explicitly declared net 25,209.9 / 27,164.4 kg agree with targets. Full delivery-agent phone pairs and independently printed shipper/consignee blocks, B/L/booking/date/port repeats and HS codes are correctly cast. No notify is invented where source has none. Physical load bounds are not treated as an extraction-label defect under the user's clarified objective.

**Flag:** export reference `3154594`, importer `607038519`, customs `6070385192023070059` and exporter `100000639300003` remain source identifiers despite changed companies/countries.

### 18 — 6c5e700d

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_6c5e700d90753d44af4b8dcd0df70bb24c1a3df767b7a66a23b2eb58c29ee829/source.txt).

Descendants reviewed: `syn_full_v7_c1f04ea4f95619bc86003c89`, `syn_full_v7_4173c75a29f67efffd561322`.

Decision: **FLAG: company-name contamination in both consignee addresses; stale invoice/export references**.

Both complete tyre product passages remain one goods entry with two separately labeled HS codes; STC, proforma boilerplate and customs text stay outside description. Single 40HC explicitly carries 6,264 / 2,488 packages, and quantity labels match. All generated primary/also-notify/delivery-agent identities, phones, postal country text, main B/L/date/vessel and route repeats otherwise agree; made-in origin is retained from separately printed Marks as its origin field, not product-description leakage.

**Confirmed target/input defect in both descendants:** repeated consignee identity starts the generated address (`KIANGA RUBBER TRADING HOUSE, SALALA ROAD...` / `MBOUANGA NIVEL TRADING, 12 RUE...`). That company name belongs only in `name`; address targets must start `SALALA ROAD...` / `12 RUE...`. Notify address is clean, showing the inconsistency within the same document. Rendered duplicate identity should also be cleaned.

**Residue:** proforma `STONE-09/24`, date `22.04.2024` and export reference `HMB-EXP-045949-29042024` survive. The form's `FOR TRANSSHIPMENT (14)` location is labeled `placeOfDelivery`; this inherited role mapping should receive layout/form adjudication before claiming complete route-policy certification.

### 19 — 7227df76

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_7227df7649223f8aca5afdb9b15484e90b7fff5addc86b6bd6cfd94c6a9cedc4/source.txt).

Descendants reviewed: `syn_full_v7_1c0793bfcdbe3491d4d4e0ce`, `syn_full_v7_1ac85b9a1363d7bdb875126b`.

Decision: **LABELS PASS; S/O residue**.

Flattened description/customs/container rows are correctly separated in both casts. Three 40HQ allocations match 3 pallets = 1+1+1 / 3,450 kg / 39.7 CBM and 2,059 cartons = 687+686+686 / 21,633.9 kg / 177.5 CBM. Machine capacity/temperature-controller specifications are product wording, not shipment quantities or container thermal settings. Complete generated product lists, independent matching consignee/notify, postal granularity, new registrations/customs, B/L/date/vessel repeats agree. Unprinted route ports and freight arrangement are not invented from issue address.

**Flag:** attachment S/O `R232162420` remains the source shipment reference beside the changed B/L in both descendants.

### 20 — a177de46

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_a177de46a81ab0e061b8bf391f74e741b70004f5b9f9dc0d6da8bde10ab853fb/source.txt).

Descendants reviewed: `syn_full_v7_10a6739a7cf9e578dc515c67`, `syn_full_v7_7cb41af923b17a6b81090d28`.

Decision: **LABELS PASS; invoice/customs-registration residue**.

One declared container holds 16 pallets / 13,609.6 kg / 25.6 CBM or 19 packages / 6,626.4 kg / 23.009 CBM; equipment wording and complete allocations match. Product passage starts after detached CIF declaration and ends before invoice/customs fields. Attached manufacturer model refs/dimensions remain product wording. Primary/notify/Marks repeats change coherently; full party addresses retain postcode/region/country without ATTN/contact contamination, and named contacts/phones are correct. B/L/booking, ports including explicit countries, vessels and shifted board/issue dates agree.

**Flag:** invoice `AMIMED-240318-01`, customs `6771614762024030039`, importer `677161476` and exporter `1018143015` remain source identifiers in both samples; changing country captions alone is not complete anonymization.

### 21 — 7383087e

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_7383087e48ef35f1cd5c7f01ce109e5fc7dfa7b61b38f455e01067d90b116c31/source.txt).

Descendants reviewed: `syn_full_v7_548dd2687e4ecd4fdd5aae69`, `syn_full_v7_808abbff58c87d16e827d6fe`.

Decision: **PASS**.

Inline `SAID TO CONTAIN: TOTAL ... OF` declaration is cleanly separated from product wording and following NET WT/HS fields. 13 pallets / gross 16,907 kg / net 16,647 kg and eight pallets / gross 17,307.1 kg / net 17,147 kg match; total gross+tare is not cargo mass. Single 20GP allocation is supported, and duplicate canonical/type wording adds no conflicting equipment label. Names, phones, ATTN contact names, full postal locations and all tax/customs identifiers change; literal same-as notify is reference-only. Vessel/voyage/port fields match even in flattened row. No blank date/B/L is invented. Fixed Lucy Borchard signatory and generic tariff equipment prices are not shipment identifiers.

### 22 — 73576135

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_73576135bddebf063b7d50b25a4683fdf6a738749240f9377630a4224bc7037f/source.txt).

Descendants reviewed: `syn_full_v7_06454e4d3e1f14aef6f905d2`, `syn_full_v7_659277a6ce1eb3cfe9ce6993`.

Decision: **LABELS PASS; invoice/importer-ID residue**.

13 pallets and 18 packages explicitly allocate to one 20GP container. Complete steel-strip / electrical-conductor product passages retain attached packing/batch-model specifications and `NOT RESTRICTED`, but not following invoice/customs fields or rider Marks. Generated shipper name spans, complete postcode/country addresses, literal same-as notify, receipt/ports and repeated board/issue dates agree. No mass/volume/HS code is invented from sampled internal facts when absent from OCR.

**Flag:** invoice list `4700 DL7A 029732, 029733, 029734, 030031` and importer tax `207-677-689` remain source values in both descendants. Exporter/customs and main B/L do change. Fixed Dachser issuer office remains separate from generated shipment parties.

### 23 — cc1f69ab

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_cc1f69abcacb3edfd96d11d938e03571e7770e3370de8ef451f2bc2cd79a7932/source.txt).

Descendants reviewed: `syn_full_v7_09d33f2a2b776c47ed721d69`, `syn_full_v7_73f6e5f20daf0ab1d15a599e`.

Decision: **LABELS PASS; exporter-registration residue**.

Long bumper / winter-boot descriptions expand cleanly into one product passage; carton/box declarations, standalone HS, registration continuations and empty source textile metre-count remnants do not contaminate description. One 40HC holds 1,690 boxes / 16,623.1 kg / 24.4 CBM or 258 cartons / 13,912.6 kg / 59.9 CBM with explicit full allocation. Uppercase postal/name targets and complete repeated notify details agree. Main B/L prefix+document number, ports/receipt/delivery/issue country, dates and prepaid place agree; redundant destination locality repeats are not duplicated in target names.

**Flag:** shipper continuation registration `9133062130764453XJ` remains source identity in both descendants. Importer tax/customs numbers change.

### 24 — b3aeb7ab

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_b3aeb7ab7b0a87281bf9391bbb5fcab1b3b7a867b60da1bfd6be91697b2ec91a/source.txt).

Descendants reviewed: `syn_full_v7_0035969670397c83219b464b`, `syn_full_v7_d2e4c12a8d559753c3b02e06`.

Decision: **LABELS PASS; booking/shipper-reference residue**.

Both two-container samples repeat the same complete generated product passage once per container while labeling one goods description once. Radiators: 234 boxes = 116+118, 17,999.348 kg; zinc: 31 bundles = 15+16, 29,063.096 kg. Container categories/seals, standalone HS, per-row quantities/mass, complete independently repeated consignee/notify, postal lines and phones agree. Zinc's 8 kg-per-ingot specification does not invent a shipment net label. Source export/footer numeric IDs and government registration values are changed; conditional USA legal text is not interpreted as origin.

**Flag:** shipper reference `EW5467.01/02` and booking `EBKG08063724` retain source identifiers in both descendants. Carrier SCAC/MSC sign-off remains fixed form text.

### 25 — 51d5f830

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_51d5f8302e5b4e1f70a50cf4eaed3fd51cc1d661d5953acc25a1e9e76983874e/source.txt).

Descendants reviewed: `syn_full_v7_28351f040978b83f2268e028`, `syn_full_v7_76ffdbe21d604f4c13f0eacf`.

Decision: **PASS**.

Reviewed all three pages per descendant, not just first-page output. Repeated shipper/consignee/notify, main B/L on page 3, container plus two seals, quantities, spelled totals, full product passage, HS/customs, route/payment/issue/date values agree throughout. 27 packages in 20GP and 37 pallets in 40HC both have explicit single-container STC allocation. Telecom / two-yarn-variant wording remains one goods entry with two HS codes; declaration and references stay separate. Complete addresses preserve split postcode/CP text. Delivery-agent block has two named contacts separated by `; ` and two matching numbers, without adding them to address. All shipment IDs/registrations/seals change. Fixed Vasco carrier registration `199804855M` is not treated as mutable shipper identity.

### 26 — ecb17253

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_ecb172534201c4018e786081e3f1048a7dd7608bfd2dc8aca14289eaafc557dd/source.txt).

Descendants reviewed: `syn_full_v7_d5354c97d9e3f235cfc8d24c`, `syn_full_v7_a1e3b3f2aeb9749aa3e14b48`.

Decision: **FLAG: generated tax/contact text in shipper name; source registrations remain**.

Both three-page casts have correct container/package/weight/HS and complete product passages: 15 bins / 16,317 kg and 15 pallets / 3,304.7 kg in one 40HC. Repeated vessel/voyage/waybill/route/payment/date values agree; unlabeled measurements have no invented volume unit. Consignee/notify postal addresses exclude printed tax numbers.

**Target defects:** `syn_full_v7_d5354c97d9e3f235cfc8d24c` puts generated `RIF J-40528716-3` into shipper `name`; it is an explicit tax identifier, not company identity. `syn_full_v7_a1e3b3f2aeb9749aa3e14b48` puts `ATN: LIC. MARLENY ORTEGA` into shipper `name`, rather than `contactDetails.contactName`. Correct names are `POLIMEROS DEL ORINOCO C.A.` and `COPAN DENTAL EXPORTACIONES S.A. DE C.V.`. This is generated name-field contamination, not an address or product-number issue.

**Residue in both:** source exporter/shipper `08301340967`, party `412061597`, importer `223017078` and customs `2230170782024050010` remain unchanged. Product serial numbers are newly generated and printed consistently, unlike these old documentary IDs.

### 27 — e3caae78

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_e3caae788c043bdd153db98cb10ab2d44b17975c94dc7db3da8018c74aed06aa/source.txt).

Descendants reviewed: `syn_full_v7_67f94f102825cd5edd778f7a`, `syn_full_v7_25343cfefc56c1fe5fa6e95b`.

Decision: **FLAG: unchanged package total contradicts both sampled declarations; source ID/part-load residue**.

Both full three-page descendants were inspected. `syn_full_v7_67f94f102825cd5edd778f7a` renders 73 packages/10,480.6 kg/40HC, and `syn_full_v7_25343cfefc56c1fe5fa6e95b` renders 468 packages/16,793.4 kg/20GP; both correctly label the primary declaration and allocation but still print the source's **`TOTAL PACKAGE 1`**. This is an actual conflicting quantity, not a merely cosmetic reference. The generated multi-model descriptions, party details, routes, B/L/voyage, gross weights and repeated main B/L references otherwise agree. The consignee/notify `ID NO.: 30607260300055` and other part-load B/L `MBE0448643A` survive unchanged. Fixed CMA legal clauses are not mutable identities. **Do not certify either complete rendered input** until the repeated package total and mutable secondary IDs are addressed.

### 28 — 92da45e4

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_92da45e47852cb1c6b8c7dedd3228be6b5a00293109a478929134e32452981c8/source.txt).

Descendants reviewed: `syn_full_v7_ad1b3ea297772fc397c56c64`, `syn_full_v7_a3aee7a2046419b36bade047`.

Decision: **LABELS PASS; booking/contract/PO/export-reference residue**.

Source and both complete three-page descendants inspected. New wool products (500 cartons, 26,500 kg, 63.559 m³, 40HC) and cherry wood products (7 bundles, 12,295.5 kg, 17.998 m³, 40HC) preserve their full product passage, including specifications; detached customs/HS/PO/STC lines stay out. Complete parties, contacts, coded-country addresses, route, seal, gross/volume and single-container allocations match. Blank dates are not invented; repeated primary reference values change consistently. Booking `EBKG05366692`, contract `23-420WW`, source PO list and standalone export reference `X20230522992188` remain. These are secondary input anonymization findings, not evidence that the new goods labels are wrong.

### 29 — a4cb81c1

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_a4cb81c178d0efc633a9eb6ede4245755240e921e17c538d8b7d405ab2eea94e/source.txt).

Descendants reviewed: `syn_full_v7_10cac6f9b074b49e279a11be`, `syn_full_v7_298f77cf3b839562607559ee`.

Decision: **FLAG: both retain a conflicting source package count; delivery-role grounding question**.

Both descendants change parties/routes/goods/equipment and seals, matching their principal STC declarations and amount-in-words: **21 pallets** and **3,553 packages**. However, each still prints the source's standalone **`1265`** immediately after the seals in the `No of Packgs` area. That is a quantity contradiction, not an excluded reference. Goods, HS, two seals, contacts, dates and negotiability otherwise align. `route.placeOfDelivery` equals the discharge/payment port although no separately filled final-delivery field is visible in this OCR; record this as a role-grounding review question, not proof of an explicit delivery occurrence. Fixed VASCO registration/signatories may remain.

### 30 — 370e140a

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_370e140ab048444ffa726b600122bb613edfd93a808e6db6da68e16f9301d565/source.txt).

Descendants reviewed: `syn_full_v7_c002f77ec83511ce01fb7767`, `syn_full_v7_3c4ed2a5e06d03dbbcfb7d8e`.

Decision: **LABELS PASS on established description boundary; source invoice/reference residue and prefix review**.

EVA pellets (1,056 bags/27,033.6 kg/42.667 m³) and electric works trucks (13 pallets/19,882.353 kg/22.941 m³) each retain their full long generated passage in one goods item, with correct inline allocation, HS, gross/volume, 40HQ seals, dates, parties and uppercase targets against variable-case OCR. Detached INV/customs lines stay out. Source reference pair `27690050146062 / OESA2400001989` and invoices `1390956633/1390956634` survive. The numeric prefix replacing source `TB 20836316` (`47864131` / `19557690`) is excluded from description under the inherited bookkeeping boundary; its rendered form has lost `TB`, so it no longer independently communicates that ownership. Flag that presentation for boundary review rather than silently call a bare generated prefix product text or prove it wrong.

### 31 — d3c71ee6

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_d3c71ee6f5a19b7d46c5bd6385f7d9f4dd8e715d78e85095f06b1fd70b6426e6/source.txt).

Descendants reviewed: `syn_full_v7_305750e34f61599e437afe32`, `syn_full_v7_90b3f5699650319ed6f1ff62`.

Decision: **PASS**.

Both complete two-page descendants inspected. Six 40DRY9′6 containers remain six high-cube GP entries with all generated seals and declared splits: 46 pallets (6+6+11+6+6+11) and 2,914 cartons (385+384+704+321+384+736). Full polymer-bag/drilling-parts wording is separate from customs lines; gross totals equal the printed per-container sums to rounding. All mutable party/B/L/registration references change and repeated primary parties agree. Conditional `to order` in the consignee heading correctly does not create negotiability. Freight-prepaid continuation, issue/onboard dates, route and casing align. Source MAD tariff boilerplate remains outside targets; this is not a contradictory cargo quantity or mutable person identifier.

### 32 — a3171859

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_a317185960d6b573694f1d7687e7ef2eb73758a05a9e5d70a6f66c22159b88f3/source.txt).

Descendants reviewed: `syn_full_v7_bc4e9a40e0811b888ba13f9a`, `syn_full_v7_1ecb06d01d86ed88f32c7dff`.

Decision: **PASS**.

Both complete two-page descendants inspected. Seven 40HC GP containers, seven seals and seven package placements all match. Package sums 270 (40+59+52+33+32+30+24) and 479 (72+106+92+58+56+53+42) agree with totals. Full plastic-bag and line-pipe passages preserve specifications and embedded product quantities without splitting one declared goods entry. Gross labels sum printed rows to rounding; no unsupported HS value appears. New references/companies/addresses/routes/dates are consistently rendered and targeted, explicit repeated notify parties retain `sameAs:null`, and headings/legal conditional order wording do not affect non-negotiability. `BONAIRE` as island plus its full country name is not automatically a duplicate city/country defect.

### 33 — 00388da5

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_00388da5efb560ae60c645831f6b34db7049b9c0e0772d76eecda0d7387de900/source.txt).

Descendants reviewed: `syn_full_v7_d77e308fa57d8ecb5979ba35`, `syn_full_v7_6c474378c591cfd3d383f195`.

Decision: **FLAG: delivery-agent contacts contaminate one generated description; other descendant passes**.

`syn_full_v7_d77e308fa57d8ecb5979ba35` passes this review: complete long yarn passage, 472 cartons, both seals, standalone count and words-total all agree; no fabricated weight. `syn_full_v7_6c474378c591cfd3d383f195` correctly renders/labels 319 packages but appends **`DELIVERY AGENT CONTACT: MR. DESMOND WONGSUPHASAWAT; MS. ROSITA PERSAUD-NAKUM.`** to the product passage and description label. Those names already belong to the delivery-agent block. This is generated cross-section contamination, not a product specification to preserve merely because it was placed in the description. Other party/route/date/HS/equipment fields agree; identifiers are changed. The source's standalone package count is successfully rebased here, unlike source29.

### 34 — 9ad49fcc

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_9ad49fcc846c53b1780f28a74c0cd1073d7bb2d812e5b5a76d2744b1ba4a7f02/source.txt).

Descendants reviewed: `syn_full_v7_b569c3d66ac99519ee436c94`, `syn_full_v7_22186a5909ec212e1e1fa4bd`.

Decision: **LABELS PASS; retained booking and awkward contact separators**.

Both complete two-page descendants inspected. Bolduc fabrics (10 pieces/20GP) and clutch parts (1,178 packages/40HC) preserve full generated descriptions and correct HS, seals, allocations, all three delivery-agent phones, explicit notify details, port/final-delivery roles and dates. Printed `G.W 3250.000` / `G.W 16084.796` lacks an owned unit; absence of a structured gross-weight value is consistent with the agreed no-unit-invention policy, not a new omission. Source booking `OSL-63658/25` remains. The inherited contact layout produces doubled `++` and joined phone lines; values are recoverable and targets agree, but note the cosmetic render artifact separately. Fixed Oceanic/Ben Line issuer wording is not a synthetic primary-party leak.

### 35 — 4e766b20

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_4e766b207e0913cbf720427ebbb3a37e8ef0508b2ed94876763de81224b882bb/source.txt).

Descendants reviewed: `syn_full_v7_7ed2cc9c023794d35169f3f2`, `syn_full_v7_325c32ebe7f6b89269e0aebc`.

Decision: **LABELS PASS; booking/contract residue**.

Both full two-page descendants inspected: 22 reels of coated paper/27,634.7 kg/40HC and 16 bundles of zinc/15,000.3 kg/20GP. Repeated declarations and container rows agree. Description removes the leading shipment count while preserving the continuous full product list and embedded nominal 25 kg ingot specification; that capacity is not a shipment net total. Generated parties, dates, seals, customs and B/L match targets; second-page prepaid declaration is correctly captured. Booking `236420435` and service contract `298812895` remain source values. No unsupported HS value is added where source has no mutable HS slot.

### 36 — 00a165a2

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_00a165a2adf7156bdecd557c885de531a2a69a98a920a4e029d491bd5d50b3ed/source.txt).

Descendants reviewed: `syn_full_v7_ed48b37968dc6188f7634daf`, `syn_full_v7_74491e99383593ebb61f98fb`.

Decision: **LABELS PASS; extensive secondary-identity/geography residue and heading corruption**.

Both full three-page descendants inspected. Two-container yarn and polymer shipments have complete four-product passages/four HS codes, correct net/gross, summed volumes and declared splits (512+366=878 cartons; 1,039+743=1,782 bags). Repeated B/L/booking and consignor name update correctly; freight-collect continuation and dates agree. Source invoice `SHIN-ELQ-240522_2415`, PO `4010002415`, registration values `205055745/6098184859`, warehouse `(BANHA- SOSDI -ODP)` and the uncategorized C C agency block with `Mohamed YASSIN`, telephone/fax and Port Said address remain unchanged despite new parties/routes. Record that agency block for ownership/anonymization review rather than assume it is carrier legal boilerplate. Both heading references **`see clause 1 and 14` become `1 and 512` / `1 and 1039`**, showing a sampled package count was also substituted into fixed text. Cargo targets themselves still match their correctly rendered rows; this is a render-cast defect plus secondary residue.

### 37 — c92e5900

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_c92e59000dc7f8a89e5822297b060d4e56d345b42ccc9d1987f64be1ce8688ef/source.txt).

Descendants reviewed: `syn_full_v7_60e4f3643211ea0b2fa6c7c8`, `syn_full_v7_03fa342f180459a0ace6997b`.

Decision: **LABELS PASS; source registration/contract residue**.

Both descendants correctly isolate new long goods passages from the leading loading declaration: 564 cartons of steel household articles and 759 rolls of knitted fabric. Gross/volume, single-container allocation, dates, receipt/load/discharge/delivery locations, freight-payment place and changed forwarding-agent name match. `SAME AS CONSIGNEE` stays reference-only; uncaptioned personal names within the company's identity are retained under the agreed rule, while `SH>`/`CN>` stay out of postal labels. Source `4454247022024030018`, `9144010MA5CN9R09Q`, `445424702` and contract `TNBON00401A` survive. Fixed ONE agent signature is permitted.

### 38 — 35136b22

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_35136b22922177501a794351a470cce57d8873a40338c10058abf1712f0459e8/source.txt).

Descendants reviewed: `syn_full_v7_b0c48076c801d501884bc4a3`, `syn_full_v7_27e3fecbf6566c925d14f5d1`.

Decision: **FLAG: shipment package prefix included in both descriptions; unchanged owner identity**.

Both complete three-page descendants inspected, including the separately reopened first descendant to recover a truncated display. Five 40HC containers and explicit placements are consistent: 1+1+1+1+2=6 pallets of moulding machinery; 11+9+11+10+12=53 pallets of MRI apparatus. All seals, gross values with Kgs heading, repeated B/L/voyage/date and primary parties agree. However, descriptions begin **`6 PALLET OTHER RUBBER…`** and **`53 PALLET MAGNETIC…`**. Those are leading shipment-count declarations, not embedded product specifications; current boundary policy excludes them (the comparable prefix is removed in source35). Both notify blocks also retain source **`Owner: Al Ghurair Centre LLC`** under new company names. Unitless measurement figures are not fabricated into volume labels. These are specific description-policy/anonymization findings, not failures of the five-container split itself.

### 39 — a53a1a9e

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_a53a1a9e2cc54246896e0fcd8651ecfe668e9d0f135feb4428ead766c18294cc/source.txt).

Descendants reviewed: `syn_full_v7_ec6ce8baa67af243d79d015a`, `syn_full_v7_76652c2aa810111e689e2b27`.

Decision: **LABELS PASS; non-target importer-registration consistency flag**.

Both full two-page descendants inspected. Three repeated container rows repeat the same long product passage; labels correctly keep one goods description/one HS code, totals/net/gross and placements (1,053+1,033+1,033=3,119 bags; 1,676+1,645+1,645=4,966 packages). All primary parties, represented-company wording, separate second notify and delivery agent, dates, seals/routes and shipment IDs change consistently. Excluded net/gross/SEAL lines do not leak into description. The generated consignee-block company registration differs from the generated `IMPORTER COMPANY REGISTRATION` in customs: **757836254 vs 435839111**, and **661462799 vs 414248908**. No target scalar uses those values, so do not call this a cargo-label defect; it is a duplicated importer-fact consistency flag for input casting (the source also had discrepant VAT digits). Fixed Atlantic carrier signature may remain.

### 40 — 3b839670

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_3b8396709a612d870263e6484bf91d5ee074c6f595818d2babd643ad1c70d691/source.txt).

Descendants reviewed: `syn_full_v7_889cc8b395524efb1951848c`, `syn_full_v7_7c1d5f04fef642cc8c6983a6`.

Decision: **LABELS PASS; source VAT/import-reference residue**.

Both full three-page descendants inspected; truncated combined output was reopened. Four containers/seals match and remain untyped because no length/type is printed. Repeated long fan/converter descriptions are deduplicated into one complete goods passage, with allocations 15+15+15+14=59 cartons and 742+742+742+741=2,967 cartons. Kgs total is correctly gross weight; unitless measurement/tare values stay out of unsupported labels. Country/contact boundaries, repeated parties, delivery agent, route/voyage/B/L and dates agree. Source shipper VAT `EL094469570`, customs `3279079672025120046`, importer `327907967` and destination-specific Egyptian customs boilerplate remain; these are separate cast/anonymization residues, not demonstrated container-label omissions.

### 41 — 926e04e2

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_926e04e23df588fc33da547ec1a11fe2c98fe5030c911d770651f56662f78d3f/source.txt).

Descendants reviewed: `syn_full_v7_b82ebf11a51da6dbb72e8ea0`, `syn_full_v7_cbd021aeaf451198fc1f39bb`.

Decision: **LABELS PASS; source invoice/order/PO/ITN residue**.

Both complete descendants inspected. New cable conductors (15 drums/8,249 kg) and cultivation machinery (5 packages/3,987.5 kg) have complete product passages, HS, 40HC seal, correctly repeated totals, gross labels, load/issue dates and forwarding/notify contact details. There is no consignee occurrence in this source format, so `negotiability:null` is consistent and no consignee is invented from the intermediate-notify block. New main document/footer IDs and primary companies/addresses/tax references change. Source invoice `469107`, order `22033`, PO `PO23/00009`, ITN `X202305416672482` remain. Fixed issuer OTI license `13247NF` and Seamates/Zim signature are not personal shipment identifiers to remove.

### 42 — f256749b

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_f256749bd9718a0bae20e7cdd8fe6942a7b1e3759f3089b86a23d7524f5638f0/source.txt).

Descendants reviewed: `syn_full_v7_5c74fc7de4558520b5fbd44e`, `syn_full_v7_0a0bcbff233535c12580ee5a`.

Decision: **LABELS PASS; primary shipper not anonymized and represented-party/reference residue**.

Both complete three-page descendants inspected. Two untyped containers have changed IDs/seals, complete deduplicated descriptions, unit-bearing per-container net values summed correctly (43,095.24 kg yarn; 33,758.91 kg airbags), owned gross totals and explicitly printed total packages (849 CTN; 2,760 PK). Placements correctly omit quantities because no per-container package split is printed; unitless measurement stays out. Changed consignee/notify/delivery addresses, phones, routes/dates agree with labels. **The primary shipper remains `VARDHMAN TEXTILES LIMITED` in both new countries**, and `A/C SONVIGO INTERNATIONAL LTD` survives in its target identity. The second-page SONVIGO company/address, its Mauritius registration/country code, subrefs `NV250000217/218`, SB numbers/dates, bolt IDs and booking-office identity also remain. This is an actual primary-company anonymization/casting failure, although the extraction labels match what is now printed.

### 43 — 2763b891

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_2763b891d894632d56df7eaa22ff6d81426e42a49b9f2816098120a63dc1d12a/source.txt).

Descendants reviewed: `syn_full_v7_385c69f6ec1beb3b820cdf99`, `syn_full_v7_6c139fd0a4cc637f48c28391`.

Decision: **LABELS PASS; generated manager-name boundary noted**.

Both complete two-page non-containerized descendants inspected (second reopened after a clipped legal-text display). New shipper/consignee/notify addresses, phones and tax IDs, chassis IDs, discharge port, vessel and voyage are rendered/targeted consistently. No container is invented; one vehicle, 10,800 kg/59.579 m³ retain the source's fixed RoRo accounting. Product passage excludes marks/chassis and extensive liability boilerplate. Source has swapped-looking OCR vessel/loading captions; the actual generated ship name/voyage is correctly typed and no loading port is invented. Second descendant's uncaptioned person remains in identity under agreed policy. First uses `GERENTE GENERAL:` within consignee name: retain as an explicit role-boundary review note, not silently reinterpret it as a contact when the agreed rule only moves identified contacts. All old primary mutable IDs are replaced; Grimaldi carrier address/privacy URL remain legitimately fixed.

### 44 — 257db44a

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_257db44ac681e2860a171f07a2dc6f23f552c7e3b39532eb9c417db84c15dc6c/source.txt).

Descendants reviewed: `syn_full_v7_12f043a061344b84f42c98f0`, `syn_full_v7_ecdf464c746a4872874e0616`.

Decision: **LABELS PASS; source invoice/customs registration residue**.

Both complete descendants inspected. New fabric shipment (525 bags, 18,126.1 kg, 63.9 m³) and door-closer shipment (2 pallets, 1,273.8 kg, 5.1 m³) correctly exclude initial `525BAG OF` / `2PLT OF` from full product passages, preserving all specifications. 40HC seal/single-container allocation, repeated B/L/booking, dates, consignee address/contact/emails/phones and reference-only same-as notify align. Shipper has no address in source, so no unsupported postal label is added. Source invoice `DM20230521`, import reference `2396619662023050024`, importer `239661966`, exporter `6261430000001237` survive with changed country/code. This is a secondary-ID anonymization finding, not a mislabeled product total.

### 45 — 4fc21678

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_4fc21678ae8c728ae726dd27fa4d3f94365132267f734c455cf880e87f6b0d5b/source.txt).

Descendants reviewed: `syn_full_v7_91b07a661861f57edb92ccbd`, `syn_full_v7_b32772ed342e1492b4d54b43`.

Decision: **LABELS PASS; shipping-agent fax residue**.

Both complete two-page descendants inspected. Eight 20GP containers have all eight new seals; 69 pallets of steel and 1,142 cartons of monofilament retain correct complete attachment descriptions, HS, matching repeated net/gross/total declarations and routes/dates/B/L. Placements identify all eight containers without inventing per-container quantities. Same-as notify remains reference-only. Party phones/emails and addresses change with correct targets, including case-sensitive email values and printed diacritics. Source shipping-agent fax **623190252** remains under both newly generated delivery agencies; record as non-target contact anonymization residue. Fixed Wan Hai legal issuer identity and exchange-rate boilerplate are not label failures.

### 46 — e70b7185

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_e70b7185df18aa28c376668c4f6404163c8c95ceb37adf9a5857b3215b1c735d/source.txt).

Descendants reviewed: `syn_full_v7_e771fa8d92246e84eb4122a8`, `syn_full_v7_39aeb508cf1661f6b3f8e754`.

Decision: **FLAG: one descendant has contaminated party identity/postal boundaries; source tax/reference residue**.

Both complete three-page descendants inspected, recovering the clipped first-descendant page2 separately. Thermal/container arithmetic and labels work: 2,610 cartons of swordfish, three 40RH reefers at −20.5°C, gross 56,662.92/net 54,236.99 kg/110.9 m³; 5,449 cartons of carp, three 40RH reefers at −21.5°C, gross 58,501.88/net 55,777.38 kg/80.4 m³. Reversed-order duplicate container references are updated, splits and full product passages agree. First descendant's extraction fields pass. **Second `syn_full_v7_39aeb508cf1661f6b3f8e754` has shipper name `KANDAHAR CO. PLOT 22, BLVD. HAMMERFEST` (postal text inside identity), repeats `KANDAHAR CO.` inside `addressLine`, and places `POINTE NORD SARL` in consignee/notify postal labels.** These are generated party-boundary errors, not acceptable casing normalization. Both retain source ABN `39006312616`, importer `202446026`, reference `2024460262024030231` and export refs `2411-2412-2413`. No separate HS scalar is invented when none is rendered.

### 47 — 53cf4d99

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_53cf4d9917f9c1736f3d09087d3d745ebeec296fb34f26b3c2a2fa09dfd73695/source.txt).

Descendants reviewed: `syn_full_v7_9513fb3afc9e40b2590f80a2`, `syn_full_v7_566b8e3c6cb9ff33450ac666`.

Decision: **LABELS PASS; source customs/registration residue**.

Both complete two-page descendants inspected. Long tyre specification list (5,495 packages/40HC) and four screw variants (1,067 bags/20GP) keep complete product passages but exclude leading declarations/customs/Marks. Single-container split, seal, gross/volume, route, dates/prepaid, complete party addresses and explicitly captioned contacts agree. Detached repeated consignee in the cargo area is correctly excluded from description and not invented as a separate notify party. Source import reference `6771614762025040011`, importer `677161476`, exporter `1018143015` survive although countries/companies change. No HS field is invented where the generated input has no HS declaration.

### 48 — 88deddad

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_88deddadf982fd5c5920c4f64487fc06a0db25f6e8a3e7cd76efd0168960a10b/source.txt).

Descendants reviewed: `syn_full_v7_3b68e516600b5897d89b19b1`, `syn_full_v7_4b5ab378f15f4cb16d5acc88`.

Decision: **FLAG: one postal label includes source OCR noise; secondary identifier residue**.

Both complete descendants inspected. The unchanged page-two legal terms were read with the source and verified byte-for-byte identical in both descendants; changed page-one text and full targets were reopened separately after a clipped combined display. New spectacle-lens shipment (7 packages/40HC, 3,813.3 kg, 9.1 m³) and charcoal shipment (519 cartons/20GP, 9,861 kg, 22.4 m³) have complete descriptions, explicit allocations, HS codes, seals, new routes/dates and correct non-negotiability. Consignee and notify are independently printed repeated parties, not reference-only same-as. **Charcoal descendant `syn_full_v7_3b68e516600b5897d89b19b1` includes the inherited nonsensical source token `ENVNT` at the end of `deliveryAgent.addressLine`.** It is not a generated postal component; the other descendant omits it correctly. This is a postal-boundary/noise-contamination defect under the agreed label policy, not an instruction to invent a country. Source consolidation reference `CDE1910153`, order `37460`, VAT `435-517-155` and import reference `4355171552023120033` survive; the linked main/reference B/L pair is changed consistently. Fixed DSV legal identity/VAT remains legitimately unchanged.

### 49 — 2c599a8b

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_2c599a8b65676a12af477b9155aca96c79e1a4305ced1c7a0ced59d461a98c8a/source.txt).

Descendants reviewed: `syn_full_v7_3f393a2702db52216f255c54`, `syn_full_v7_e68566dac0bd5f5b4123931a`.

Decision: **FLAG: two descendants retain conflicting package-type aliases across container rows; secondary booking/party residue**.

Both complete two-page descendants inspected. The source has three repeated single-piece container rows. New total quantities and all three per-container quantities/weights, equipment/seals, product passages, companies/tax IDs, ports and dates are correctly cast; the new goods total is 7,129 packages for tyres, or 219 packages for steel. **Only the first container's unit wording changes to `PKGS`; the other two still print `PIECE`/`Pieces`, while the target declares all three splits at `PACKAGE_PACKAGE`.** This is a surviving repeated-row package-type casting inconsistency, rather than a seal/quantity arithmetic error. It affects both `syn_full_v7_3f393a2702db52216f255c54` and `syn_full_v7_e68566dac0bd5f5b4123931a`. Uncaptioned consignee personal identity is retained consistently with the agreed name rule; `MANAGING PARTNER:` is not automatically an explicitly named contact. Source supplementary booking IDs `236766105`/`237138972`, marks `ALX-BR22627`/`ALX-BR22610`, forwarder `BR22610`, freight invoice party `BALDAN LTD`/customer code `130LDN5331` remain; they do not appear as newly extracted target references. Fixed carrier legal/routing boilerplate is not treated as a target failure.

### 50 — 97efe837

Source: [complete OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_97efe837d0a87d40b07f5f532477670ce38700e5c8e01d715fc443983242a8c5/source.txt).

Descendants reviewed: `syn_full_v7_74c0c83c510fcba3b194f144`, `syn_full_v7_301114018e98410b4b65de1f`.

Decision: **LABELS PASS; source importer/exporter and delivery-fax residue**.

Both complete descendants inspected. New primary names, postal lines, phone/email/contact pairs, reference-only same-as notify, routes/dates and container/seal agree with targets. Full chemical list (1 package) and three display-module variants (73 cartons) are preserved as one goods description; leading declaration, HS/customs and Marks are excluded appropriately. Package declarations and spelled-out totals agree; explicit single-container wording supports allocations. No equipment size/type, gross weight or bill number is invented where this source does not print them. Source importer `204990688`, exporter `1088150292`, delivery-agent fax `+2 (02) 22694915` survive alongside new primary tax IDs. Fixed product mark `MTX` is outside target and not automatically a contradiction, but the mutable secondary tax/contact residue fails full anonymization. This audit does not reject the generated product capacities or wording merely for lacking HS-taxonomy perfection.

## Direct links to the 18 target/package-cast failures

These links open the exact current JSONL row containing both rendered text and
target. Source links and full explanations are in the corresponding numbered
inventory entries above. The audit hash manifest pins this version.

| Source index | Current descendant text and target | Finding |
| --- | --- | --- |
| 12 | [syn_full_v7_9c30d6ff9f5425cb3eb47f73](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:1412) | LABEL_PARTY_ADDRESS_BOUNDARY |
| 13 | [syn_full_v7_4dce9b179d47ebe1cf41944f](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:2081) | RENDER_PACKAGE_HEADER_BINDING |
| 13 | [syn_full_v7_2952a994485bc6ea7f9c5a3c](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:2082) | RENDER_PACKAGE_HEADER_BINDING |
| 18 | [syn_full_v7_c1f04ea4f95619bc86003c89](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:2182) | LABEL_PARTY_ADDRESS_BOUNDARY |
| 18 | [syn_full_v7_4173c75a29f67efffd561322](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:2181) | LABEL_PARTY_ADDRESS_BOUNDARY |
| 26 | [syn_full_v7_d5354c97d9e3f235cfc8d24c](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:2450) | LABEL_PARTY_TAX_IN_NAME |
| 26 | [syn_full_v7_a1e3b3f2aeb9749aa3e14b48](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:2451) | LABEL_PARTY_CONTACT_IN_NAME |
| 27 | [syn_full_v7_67f94f102825cd5edd778f7a](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:1878) | INPUT_CONFLICTING_PACKAGE_TOTAL |
| 27 | [syn_full_v7_25343cfefc56c1fe5fa6e95b](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:1403) | INPUT_CONFLICTING_PACKAGE_TOTAL |
| 29 | [syn_full_v7_10cac6f9b074b49e279a11be](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:2309) | INPUT_CONFLICTING_PACKAGE_TOTAL |
| 29 | [syn_full_v7_298f77cf3b839562607559ee](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:2311) | INPUT_CONFLICTING_PACKAGE_TOTAL |
| 33 | [syn_full_v7_6c474378c591cfd3d383f195](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:262) | LABEL_PARTY_TEXT_IN_DESCRIPTION |
| 38 | [syn_full_v7_b0c48076c801d501884bc4a3](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:70) | LABEL_SHIPMENT_HEADER_IN_DESCRIPTION |
| 38 | [syn_full_v7_27e3fecbf6566c925d14f5d1](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:303) | LABEL_SHIPMENT_HEADER_IN_DESCRIPTION |
| 46 | [syn_full_v7_39aeb508cf1661f6b3f8e754](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:2430) | LABEL_PARTY_IDENTITY_POSTAL_BOUNDARY |
| 48 | [syn_full_v7_3b68e516600b5897d89b19b1](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:2245) | LABEL_POSTAL_SOURCE_NOISE |
| 49 | [syn_full_v7_3f393a2702db52216f255c54](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:1994) | INPUT_CONFLICTING_PACKAGE_TYPE |
| 49 | [syn_full_v7_e68566dac0bd5f5b4123931a](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-v1/synthetic.jsonl:1996) | INPUT_CONFLICTING_PACKAGE_TYPE |

## Resolution priorities suggested by the audit — not applied in this turn

1. **Party semantic boundaries (sources12,18,26,46,48).** Separate company identity,
   explicit tax/contact values and postal wording in the generated fields.
   Rendering and target construction currently preserve the same wrong field
   membership, so string presence/schema validity cannot detect these failures.
   Source48 additionally exposes inherited source-target noise. Review each
   affected family once, then search all descendants with the same field-ownership
   predicate. Do not introduce blanket deletion of personal names: the existing
   policy permits uncaptioned persons as identity.
2. **Goods-description boundaries (sources33,38).** Exclude explicit delivery-agent
   contact wording and leading shipment-count declarations. Preserve the full
   product passage, embedded capacities/quantities and specification text. These
   findings do not justify resurrecting the old aggressive numeric stripping.
   Check source/contract boundaries and wording validation, not just these three
   labels.
3. **Package occurrence ownership (sources13,27,29,49).** Ensure declared package
   quantities/types bind to actual data occurrences, including standalone values,
   repeated rows and amount-in-words statements. Keep form captions fixed. The
   source13 category must be expressed in its declaration, not manufactured by
   renaming a header; source49's second/third units must be cast along with the
   first. Source36's changed clause number supplies a related negative control:
   a package count must never replace a fixed legal clause number.
4. **Identity/anonymization coverage.** Primary shipper source42 is a direct
   uncompleted cast. Secondary tax/reference/contact leftovers occur in 40
   audited families. Separate mutable shipment/party values from fixed carrier
   legal registrations, signatures and privacy links before proposing repairs.
   Some leftovers are training-irrelevant flavor text under the user's earlier
   decision; this audit records retention without treating every retained prefix,
   contract or invoice as a mislabeled training target. Repeated registrations
   with different values (source02 old/new; source39 generated/generated) warrant
   consistency checks separately from simple anonymization.
5. **Adjudication, not guessing.** Inspect route ownership for sources18/29 and
   the source30 generated numeric prefix before extending certification to
   those particular roles. Source43's manager identity and source36's secondary
   agency remain narrowly named questions. A question about one field does not
   negate passed container/description/quantity components of the same sample.

### Why these defects could survive previous validation

A field can be schema-valid and exactly present in rendered OCR while still
belonging to the wrong semantic field: company text in an address, tax text in a
name, or a delivery contact in a product description. Conversely, principal
cargo rows can match their labels while a second quantity/type occurrence
remains stale elsewhere in the input. A checker limited to target-value presence
and principal mutable slots cannot certify those dimensions.

This audit supplies specific counterexamples, not a claim that every previous
review was ineffective. It also confirms working long-description copying,
repeated-container identity/allocations, explicit thermal settings, same-as and
negotiability inheritance, route casting, casing normalization and identifier
replacement in many inspected records. **There is no blanket clean bill for the
100 samples; there is a complete, field-specific reviewed inventory.**

### Completion and preservation

All six requested audit steps were completed for all 50 randomly drawn sources.
The source/descendant draw and full reviews are preserved; no sample was
redrawn, regenerated, silently edited or removed. No source template, pipeline
code, training configuration or dataset label/input was changed by this audit.
Its deliverables are this report, the pinned draw and the validation/decision
manifest. Future repairs should be separately applied and validated against
these exact counterexamples, including siblings outside the draw.
