# Description boundaries and shipment packaging — audit and bounded repair probes

Date: 2026-10-09. Scope: the current 600 real training records, 60 real validation records, 1,500 synthetic records, and 200 active synthesis templates.

Follow-up: the approved description-boundary repair is now applied and validated;
see [the implementation report](description-boundary-repair-2026-10-09.md).
The counts and dry-run status below describe this earlier audit snapshot.
The packaging-level policy remains a separate, unapplied audit question.

**Status: audit and dry-run only. No published label, OCR, positioned input, source template, production code, prompt, or training configuration was changed in this pass.** Existing uncommitted changes from the preceding description-policy repair were preserved.

This follows the [description-block repair](description-block-policy-repair-2026-10-09.md), and refines the [label policy](kie-real-baseline-label-policy-2026-10-05.md). It does not silently replace that policy or its published labels. The user explicitly approved reducing train363 to `T850QVN04.2 85" ASSY OPEN CELL`, and the bounded product passage for train368. The other decisions below test the consistent extension of those examples.

## 1. Conclusions

1. **Description boundaries still need refinement.** Some labels include generic package introductions, standalone packing equations, repeated copies, and detached declarations. These are not all the same failure as missing embedded capacities.
2. **Non-contiguous OCR selection is not itself an error.** Correct product lists are interrupted by HS fields, references, adjacent columns, or page boundaries. Forcing one continuous OCR substring would lose product information or reintroduce Marks.
3. **The proposed repair has a demonstrated cheap path.** Thirty reviewed real-document fixtures yield 19 proposed changes and 11 unchanged controls. Eleven corresponding template contracts were tested. Ten compile immediately; one needs a precise product-owner boundary correction. Across their 83 saved descendants, 76 can use direct source-to-render projection; the remaining seven were explicitly reviewed and resolved in dry-run without regeneration. Seventy-six descendant descriptions would change; seven control descendants remain unchanged.
4. **Source review alone is insufficient.** Seven additionally inspected synthetic records introduce package introductions, standalone counts or totals in generated product wording. Four saved total-bearing product regions pass the current wording validator. Their totals match sampled shipment facts; the confirmed issue here is field membership, not invented arithmetic.
5. **The packaging question is real, and separate.** The extraction schema and cargo mapper explicitly prefer the inner shipment-package level. Some documents declare and allocate pallets, while labels select bags/cartons and sometimes lose directly printed allocation quantities. This is a policy mismatch to discuss, not proof that every old inner-level label or every model prediction is wrong.

## 2. Exact inventory and meaning of the counts

Inputs:

- `data/curated/mpci-bl-real600-synthetic1500-v7-positions-v1/train.jsonl`
- `data/curated/mpci-bl-real600-synthetic1500-v7-positions-v1/validation.jsonl`
- `artifacts/synthesis-templates/mpci-bl-v7-reviewed/{manifest.json,ownership.yaml,cases/}`
- Source-boundary receipts and render edits from `docs/analysis/description-policy-repair-20261009/`.

All 200 template sources are among the 600 real training records; their OCR is byte-identical to those real inputs. They are not 200 additional real documents. No validation source is in this catalog.

### Description screening

| Observation | Real train /600 | Validation /60 | Synthetic /1,500 | Templates /200 |
|---|---:|---:|---:|---:|
| At least one review flag | 197 | 18 | 309 | 45 |
| Non-whitespace gap between selected passages | 109 | 11 | 127 | 18 |
| Leading counted-package phrase | 99 | 6 | 188 | 25 |
| Numeric-initial description, broad backstop | 122 | 9 | 203 | 32 |
| Declaration-wording candidate | 18 | 1 | 7 | 2 |
| Detached attribute/packing continuation | 37 | 3 | 48 | 7 |
| Standalone accounting-shaped line | 19 | 1 | 11 | 1 |
| Explicit `TOTAL:` / `TOTAL=` caption | 1 | 0 | 4 | 0 |
| Documentary-word candidate | 0 | 0 | 4 | 0 |

Rows overlap. **These are exact screening counts, not exact counts of erroneous labels.** For example, `100% POLYESTER`, a 1992 vehicle year, and the user-approved `1012322163DXH DEGREE 1` are deliberately caught by the broad numeric screen but should remain.

The 45 flagged templates have **334 descendants**. Fifteen further synthetic records are flagged although their source descriptions are not, across 13 other families. The combined source/descendant review envelope is therefore **349 synthetic records**, not just the 309 with a direct text flag. An unflagged descendant of a flagged source still belongs in its family review.

The focused adjudications establish **19 real-label changes, 76 corresponding synthetic-label changes, and seven further synthetic boundary issues** under the proposed clarification. That is 102 distinct current labels with concrete issue findings, not a claim that the remaining screen candidates are all correct or all wrong. Thirty real fixtures and 83 family descendants have exact proposed targets; the seven additional synthetic cases have diagnosed boundary issues but no published repairs.

An exact final defect total for the *entire* corpus requires disposition of the remaining candidates and a recall check outside the flags. The audit does not manufacture such a number from regex matches. The requested small repair probes are complete; whole-corpus publication was intentionally not performed.

### Evidence files

- [Summary](analysis/description-boundaries-packaging-audit-20261009/summary.json)
- [All description observations, exact fragments and intervening text](analysis/description-boundaries-packaging-audit-20261009/description-inventory.json)
- [Template-to-real and template-to-descendant mapping](analysis/description-boundaries-packaging-audit-20261009/template-inventory.json)
- [Package quantities/types/placements and OCR contexts](analysis/description-boundaries-packaging-audit-20261009/packaging-inventory.json)
- [Thirty readable before/after fixtures with OCR and PDF links](analysis/description-boundaries-packaging-audit-20261009/probe-review.md)

## 3. Simplest policy that survives the probes

Suggested concise extraction-field wording:

> Copy the product-description passage and its genuine continuations in printed order. Keep its models, specifications, capacities and embedded packing or quantity wording. Leave out separately owned Marks, references, loading introductions, standalone package/weight accounting and documentary declarations. Select one copy of an identical repeated description. Preserve printed punctuation; uppercase and replace physical line breaks with spaces.

The distinction is **what the passage describes**, not whether a token is numeric, contains a unit, or is on a separate physical line.

### Include

- Product identity, models, dimensions, composition, capacities, product-specific packing and qualifiers inside the product passage.
- An embedded expression such as `PACKED IN 3930X50 KG BAGS`, even where it repeats facts useful elsewhere.
- Product units or models: `01 UNIT NEW RANGE ROVER AUTOBIOGRAPHY`, a year, a numeric model code. Package-count stripping must not become number stripping.
- Genuine product continuations across separate OCR spans when an excluded field or adjacent column interrupts them.
- Origin/manufacturing wording when it forms part of the product passage. This does not authorize searching elsewhere to append a detached country-of-origin declaration.

### Exclude

- A generic shipment introduction such as `6 PALLETS STC`, `1 PACKAGE(S)`, or `25 PACKAGE(S) OF`, when it declares carriage packaging rather than product identity/specification.
- Independent packing ledgers such as `16PLTS = 128 CTNS`, `TOTAL:21 PLTS = 167 CTNS`, or a detached bare carton count.
- Packaging-material certification/declarations, separately printed gross/net totals, customs/ACID and invoice/reference fields.
- Marks-column content, even if commercially relevant to the product.
- Repeated copies of the same product passage printed against several containers. This is whole-passage deduplication, not removal of repeated words or distinct product entries.

### Important counterexamples

1. **Train032:** `1800 CARTON(S) X 13.61 KGS OF IQF WILD BLUEBERRIES FROZEN` stays intact in the probe. It describes the supplied product presentation, not merely a detached count. This defeats a universal leading-number/package removal rule.
2. **Train423:** `POWDER TOTAL 1000 BAGS , 25 KG EACH PACKED IN ... (LOW HEAT)` remains part of the product passage. Searching for the word `TOTAL` and truncating there would be wrong.
3. **Validation028:** `TITANIUM DIOXIDE KRONOS 2360 / 4000 Paper Bags (25 kg)` remains; the separate bare `100 Pallet` accounting line is not appended. A line break alone does not make the capacity phrase a separate field.
4. **Train266:** `MARK: L N L` inside the product's specification sequence is not automatically a Marks-and-Numbers column. Field ownership, not a keyword, controls this distinction.
5. **Train422:** retain `1012322163DXH DEGREE 1` with the following fittings description. No inference that the first token must be a purchase-order ID.

This makes extraction simpler without redefining every line in the broad Description column as product description. It also avoids turning the target into a semantically rewritten summary.

## 4. Reviewed boundary changes

All row numbers refer to the current 600/60 real dataset. Full exact passages and PDF/OCR links are in the [fixture review](analysis/description-boundaries-packaging-audit-20261009/probe-review.md).

| Real train row | Proposed treatment | Why |
|---|---|---|
| 055 | One `FROZEN OX OFFALS LIVER (HALAL)` occurrence | Remove two package introductions and detached origin; avoid duplicated product identity. |
| 095 | `BEVERAGE BASE ENERGY DRINK FOR VOLT YELLOW` | Omit `600 BOXES WITH:` shipment introduction. |
| 124 | `LUBRICATING OILS NON HAZARDOUS` | Omit `4 PALLETS STC`; keep cargo qualifier. |
| 236 | `ORAFTI GR SACO 25 KG(1T)` | Keep product capacity; omit shipment count and wood-treatment declaration. Composite synthesis owner needs narrowing. |
| 237 | `01 UNIT NEW RANGE ROVER AUTOBIOGRAPHY` | Omit `1 PACKAGE(S)`, preserve commodity unit wording. |
| 240 | `AC POWER CORD` | Exclude detached `15PLTS/540CTNS`; do not jump over invoice fields to append packing arithmetic. |
| 258 | Swedish redwood passage followed by Swedish whitewood passage | Remove `684 packages`, but preserve the genuine product continuation around intervening Marks. |
| 267 | `FROZEN OX OFFALS LIVER` | Omit package introduction and detached origin declaration. |
| 277 | `LUBRICATING OILS AND HEAVY OILS NON HAZARDOUS` | Remove `6 PALLETS STC`, retain the goods-owned qualifier. |
| 278 | `ABS INJECTION ... HG-0760AT/G62287(RDAWP6039) 16,000 KG` | Preserve the bounded main passage through its embedded mass; exclude the separate total and detached `STAREX`. This adds previously omitted wording rather than only deleting text. |
| 294 | Brush / sponge / bodycare tools / plastic take nails | Preserve all four product continuations across HS fields; omit detached origin. |
| 305 | `COMPRISING VEGETABLE OIL PROCESSING EQUIPMENT MACHINE PARTS` | Remove `1 PALLET STC`; preserve OCR-interleaved product continuation. |
| 319 | `PRINTED CIRCUIT BOARD ASSEMBLY (PCBA)` | Drop standalone `(146 BOXES/ 4 PALLETS)` accounting. |
| 339 | One `INSHELL WALNUTS` occurrence | Two container-specific quantities do not create two distinct descriptions. |
| 345 | `TRANSPARENT GLUE STICK` | Drop `2,880 CARTONS OF`; the package target is not changed here. |
| 348 | `PALM NUT` | Remove package introduction and wood-package declaration. |
| 363 | `T850QVN04.2 85" ASSY OPEN CELL` | User-approved: one product copy, no per-container equations, packaging-material declarations or aggregate equation. |
| 368 | `25,925 MTCHAMBRIL WHITE WOODFREE WRITING AND PRINTING PAPER COPIER HEINZEL 1077669` | Keep bounded product passage including its printed quantity; drop `38 ROLLS WITH` and detached origin/wood declarations. Do not silently repair joined OCR `MTCHAMBRIL`. |
| 423 | `BENNI BRAND, SKIMMED MILK` plus complete powder/25 kg/packing/low-heat passage | Remove generic leading count and detached serial/batch table; preserve internal packing wording. |

Unchanged controls: train026, 032, 234, 259, 266, 422, 485; validation028, 031, 052, 053. These test capacities, multiple HS-separated products, numeric model identity, specification labels, page continuation, and adjacent-column interference.

## 5. Why some split descriptions must remain

There are 120 real descriptions with non-whitespace gaps, and 127 synthetic descriptions with such gaps. These are not 247 confirmed mistakes.

Confirmed legitimate shapes include:

- Product A → separate HS field → Product B. Train259 has stearic acid, petroleum resin, curing bag and work clothes. Retaining only its first span drops actual goods.
- A page break/attachment interrupts a continuing product list; validation031's dyestuff list is a control.
- OCR interleaves neighboring columns. In train485 the PDF attachment shows `UCAT(TM) UG-150 CATALYST INTERMEDIATE BULK CONTAINER (IBC)` in the description column, while OCR inserts `MARK THE` from the left Marks column between `INTERMEDIATE` and `BULK`. The correct target needs two OCR spans. See [page-two preview](analysis/description-boundaries-packaging-audit-20261009/previews/train485-page2.png).
- A product modifier follows an excluded measurement field; validation053's `GOUDA CHEESE 48% F.I.D.M.` and `BLOCK 15KG` retain the complete product presentation.

Unnecessary gaps arise when prior annotation jumps out to a detached packing, origin, or tracking field. Reduce those by choosing the actual product passage. **Do not optimize for a single raw substring at the expense of correct column ownership.**

## 6. Source-to-descendant probe: actual implementation path

The probe uses production `compile_description_blocks`, `build_owned_blueprint`, `validate_description_regions`, and `project_rendered_descriptions`, not a substitute string replacer.

### Results

- 30 reviewed source fixtures: 19 changed, 11 preserved.
- Every proposed target passes the strict V7 JSON schema.
- All non-description target fields are equal before/after; no raw or positioned input edits.
- 11 source templates tested: 10 compile with revised membership alone.
- One template, train236 / `e9120c73`, originally bound product and wood-treatment declaration into `v009`.
- Its seven descendants fail direct boundary projection correctly: a source substring cannot be proportionally mapped into free-form generated text.
- The probe narrows that owner in memory and recompiles successfully.
- The seven saved generated products were then read individually: six have no wood-treatment wording in the generated product; the seventh ends with a distinct packing/certification line. All seven have exact proposed rendered-text selectors. No LLM or regeneration was needed.
- Final dry-run resolution: **83/83 descendants accounted for**, comprising 76 revised labels and seven unchanged controls. This is specifically a description-membership result, not a new whole-document certification.

### What this establishes

Most boundary repairs can reuse saved sampled values and rendered text. We do not need to regenerate parties, routes, goods wording, geometry or entire documents. Correcting a source policy and projecting it through saved edits is efficient where ownership regions already separate the relevant concepts.

Composite source owners require one deliberate source-level boundary correction plus review of their saved generated values. They do not justify guessed slicing or an indefinite review loop. The seven-instance exception above is fully resolved in the dry-run artifacts.

Evidence:

- [Source decisions](analysis/description-boundaries-packaging-audit-20261009/probe-source-decisions.json)
- [First-pass template compiler results](analysis/description-boundaries-packaging-audit-20261009/probe-template-results.json)
- [Direct descendant replay results](analysis/description-boundaries-packaging-audit-20261009/probe-descendant-results.json)
- [Composite-owner correction and seven explicit descendant resolutions](analysis/description-boundaries-packaging-audit-20261009/probe-composite-owner-recovery.json)

## 7. Additional synthetic-only findings

Seven concrete examples need description-membership attention independently of source flags:

| Synthetic ID | Finding |
|---|---|
| `syn_full_v7_e38dfb7e8203cb067138f3c6` | Product generator introduced `1,718 boxes` before a detailed vehicle-braking-products description; the source example did not have that wrapper. |
| `syn_full_v7_9085dc67d32b8afd92c4db6f` | Standalone trailing `688 CARTONS` in generated charcoal wording. |
| `syn_full_v7_2502e162de7b74cf777ade62` | Standalone trailing `410 CARTONS` in generated yarn wording. |
| `syn_full_v7_f7fe878069d2cdaedc49466f` | Yarn grades with an explicit `TOTAL: 491 CARTONS`. |
| `syn_full_v7_1d1e91449ea875bf4e88d0dc` | Resin grades with `TOTAL: 802 BAGS`. |
| `syn_full_v7_161a0841c40746f8fd2aa66f` | Lumber wording ending `TOTAL: 17565 KG, 25.7 CBM`. |
| `syn_full_v7_f3a79bbac236cdda53916fd8` | Automotive-parts wording ending `TOTAL: 1718 CARTONS`. |

The four `TOTAL:` examples were fed as their actual saved product edits into the current `validate_wording` function. **All four were accepted.** The existing accounting-pattern check catches forms such as `TOTAL CARTONS 491`, not `TOTAL: 491 CARTONS` or an uncaptioned mass/volume total. This is a demonstrated guard gap, not a hypothesis about model behavior. The probe isolates wording validation; it does not rerun the whole synthesis campaign or its external reviewer.

The counts/mass in these totals agree with host-selected values. The reason to change their ownership is the clarified field policy, not a claim that all the totals were fabricated or contradictory. Three variant-level carton/bag lists also deserve attention when designing future generation: these are detailed product assortments, not automatically separate goods entries.

A blanket documentary-word filter would overcorrect other samples:

- `ACID NUMBER 15 MG KOH/G` is a product property, not an ACID customs reference.
- `PACKED FOR SEA FREIGHT TRANSPORT` / `CRATED AND PALLETIZED FOR OCEAN FREIGHT` are product packing statements, not freight-payment fields.
- `1 UNIT` attached to a specific machine, `100% COTTON`, and `1,1-DICHLORO-1-NITROETHANE` are not generic manifest introductions.

See [saved inputs and validator outcomes](analysis/description-boundaries-packaging-audit-20261009/probe-generated-totals.json). Future fixes must distinguish independently owned accounting statements from embedded product capacities, not simply forbid numerals, `TOTAL`, `ACID`, or `FREIGHT`.

## 8. Separate packaging-level audit

### Current behavior is explicit

The current rule is not merely an accidental side effect:

- `src/document_ocr/label_schemas/bill_of_lading_v7.py`: `PackagesV7`, `PlacementV7`, and `GoodsItemDetailsV7.numberAndTypeOfPackages` describe inner-level target packages.
- `src/document_ocr/labeling_agents/direct_cargo.py`: cargo mapping distinguishes target-inner, outer and product-capacity facts.
- `prompts/labeling_agents/direct_cargo_mapper.md`: explicitly instructs innermost source-established shipment packaging.
- Synthesis samples and renders the inherited selected package level. Its physical/accounting code then uses that level. Changing labels alone would leave future synthesis inconsistent.

The distinction between product contents and shipment packages already exists, but “always inner” can conflict with the most directly declared/allocated shipment unit.

### Exact screening counts

| Observation | Train /600 | Validation /60 | Synthetic /1,500 |
|---|---:|---:|---:|
| More than one counted package-word family | 75 | 8 | 149 |
| Pallet plus another counted package family | 36 | 4 | 24 |
| Membership-only or partly quantified placements | 31 | 8 | 51 |
| All labeled placement counts present, but their sum differs from package total | 5 | 1 | 0 |

These are review candidates, not an error census. Product quantities, freight-rate “units,” model numbers and separate OCR columns can produce false flags. The scanner was tightened after observing model suffixes such as `ROD-45, DRUM` and address-like `Route 1, Box...`; it still never authorizes repairs by itself.

The synthetic pallet/other review found **22 actual shipment-hierarchy examples in three families**: train319's source (eight descendants, four pallets), train236 (seven, six pallets), and train018 (seven, five pallets). The other two flagged records refer to product items/contents rather than a second shipment packaging level. These 22 are hierarchy cases, not all declared defective.

### Concrete examples that establish the policy question

| Document | Printed shipment accounting | Current target | Consequence of a declared-accounting-level policy |
|---|---|---|---|
| Validation028, KRONOS2360 | 100 pallets; five container rows each say 20 pallets; product presentation includes 4,000 paper bags of 25 kg | 4,000 paper bags; five memberships with no counts | 100 pallets and five allocations of 20. Keep the 25 kg bag presentation in description. |
| Validation052, KRONOS2064 | 20 pallets; single container row 20 pallets; product says 800 paper bags of 25 kg on 20 pallets | 800 paper bags and allocation 800 | 20 pallets and allocation 20. |
| Train113, ASSY OPEN CELL | 88 pallets; container rows 36 and 52; packing equation 1,556 cartons /31,064 pieces | 1,556 cartons; membership-only placements | 88 pallets, with 36 and 52 directly supported allocations. |
| Train363 | Container rows 16 and five pallets; conversions to 128 and 39 cartons | 167 cartons; allocations128/39 | 21 pallets; allocations16/5, independently of the approved short description. |
| Train068, polycarbonate | Total1,280 bags; container rows640 bags each;32 pallets appear as packing context | 1,280 bags;640/640 | **No change.** The declared level is bags, despite pallets being physically outer. |
| Train180, power cords |22 packages =19 pallets holding728 cartons +3 loose cartons; container row22 packages |731 cartons; allocation731 | A genuine mixed outer composition. Do not relabel it22 pallets. Explicit22 generic packages is a candidate; decide representation before migrating. |

This is why “always pallets” would be just as wrong as “always inner.” The candidate rule is **the document's shipment accounting unit**, with totals and allocations kept at that same unit.

Mixed pallet/loose-carton examples also occur at train323,354,564. Other documents have genuinely distinct typed package rows, not just nested levels; those need preservation rather than forced conversion to one category.

### Relation-error interpretation

The previously reported gold-absent pallet allocations are not all arbitrary model guesses. In the KRONOS examples, pallet quantities are visibly printed against containers, while the gold uses bags. The model can therefore read a real relation but disagree with the target's selected unit. That is a policy/evaluation mismatch worth resolving before using such errors as evidence of poor relational understanding.

This finding does **not** vindicate copying container-size numbers, an unrelated shipment total, or uncertain `646` rows into placements. Those remain separate ownership checks. No scores were recomputed under a new package policy in this audit.

The six real sum mismatches are train016,201,297,325,345 and validation012. Some reflect only partial container information in OCR (e.g. train345's one visible720-carton row against the shipment2,880 total), not bad arithmetic. Do not force allocations to balance by inventing omitted rows.

### Standards and target-form cross-reference

The [MPCI Business Specification, table10 and AppendixD](https://naic.icp.gov.ae/portal/assets/mpci-guidelines/UAE%20MPCI%20Business%20Specification%20Document%20V1.0.pdf) separates product description, package type/count, weight and container placement. Its description guidance rejects vague/incomplete descriptions and STC/invoice-only substitutes; this is not a blanket instruction to delete every embedded packing phrase. The reviewed specification does not prescribe a universal innermost package target.

[CUSCAR](https://service.unece.org/trade/untdid/d96a/trmd/cuscar_d.htm) has distinct goods/package, text, measurement and placement segments. The [local MPCI field catalog](../artifacts/mpci-ai-schema/field-catalog.md) likewise separates package rows, AAA free text, measurements and split placement; it does not impose “always inner.” Choosing the declared accounting level is our proposed extraction policy, not a newly asserted statutory requirement.

## 9. Falsification and validation results

Executed:

```bash
.venv/bin/python docs/analysis/description-boundaries-packaging-audit-20261009/audit.py
.venv/bin/python docs/analysis/description-boundaries-packaging-audit-20261009/probe.py
TMPDIR=/tmp .venv/bin/python -m pytest -q \
  docs/analysis/description-boundaries-packaging-audit-20261009/test_probe.py \
  tests/test_curated_descriptions.py
```

Result: **31 tests passed** (8.09s on the measured run). The inventory took3.09s, with820.74MiB peak process RSS reported. The bounded repair probe took approximately2.6s excluding imports. These are new audit workloads, not a before/after production throughput comparison. **External model calls:0; API cost:$0.**

Checks and counterexamples:

| Test | Result / meaning |
|---|---|
| Source SHA changed | Rejected. |
| Unsupported target text | Rejected against reviewed source spans. |
| Reordered/overlapping selected occurrences | Rejected. |
| Changed rendered bytes | Rejected. |
| Source boundary slices a mutable generated region | Rejected at compilation and projection. |
| Product owner disappears entirely from approved description | Rejected. |
| Rebuild complete saved rendering from edits, then independently map selected endpoints | Matches production projection for all76 directly projected descendants. |
| Select only first product span | Falsified by real train259: loses other products. |
| Remove all initial quantities/numeric text | Falsified by train032 capacity presentation and train422's approved code. |
| Remove everything after `TOTAL` | Falsified by train423's embedded product packing. |
| Exact text grounding proves correct ownership | **Falsified:** a self-consistent selector can include Marks or omit a capacity and still pass mechanical projection. Reviewed membership is a necessary separate input. |
| Current wording accounting guard covers actual generated totals | **Falsified:** all four saved `TOTAL: number unit` examples pass. |

The proof boundary is explicit: reviewed semantic membership + exact replay + unchanged unrelated fields. Passing byte checks alone is not called semantic approval. The retained controls prevent the new policy from becoming a destructive numeric blacklist.

## 10. Recommended next repair, kept finite and layered

### Description pass first

1. Approve the short policy in section3, including its include/exclude counterexamples. Keep package type/count/allocation policy unchanged during this pass.
2. Finish a source-level disposition ledger for the215 real candidates: revise / retain / inspect-layout. Include the45 template sources in that same ledger, not a second independent annotation exercise. Reuse the19 finished revisions and11 controls.
3. Review gap *reasons*, not merely gap presence: same-column continuation, excluded HS/reference, neighboring-column OCR, duplicated block, or detached accounting/declaration. Only genuinely uncertain ownership needs a PDF look.
4. Check detector recall outside the flags, using descriptions with capacities, long paragraphs, OCR-flattened columns and attachments. A clean regex result alone cannot certify a document.
5. Publish each real description and its explicit source membership together. Preserve input text, structured package fields and coordinates. Validate full schema and exact non-description equality.
6. For each changed template, update `description_blocks`; if it cuts a product owner, narrow/split that owner deliberately. Compile and identity-replay the source before publishing. Existing generation and projection code already supplies most of this mechanism.
7. Replay saved descendants for affected families. Handle composite owners from their saved generated strings, as demonstrated for the seven train236 descendants. Use targeted wording regeneration only when saved wording itself is contradictory or cannot supply a clean product passage; do not regenerate entire documents.
8. Independently scan/review generated product regions, including the15 source-unflagged cases, so the source review does not miss lexical-generation defects.
9. Update the V7 description field and reviewer/corrector/cargo instructions together. Remove the current implication that every packing declaration inside the broad goods column belongs in description.
10. Align wording generation and its guard: keep product capacities/specifications, avoid independent manifest tables/totals in product regions, and test the saved failure examples plus the retained embedded-total counterexample. Reject or review only the affected generated region; never silently strip a guessed substring.
11. Publish with a closed ledger and exact affected-file hashes. Run the source/descendant regression fixtures and a no-unrelated-change check against the pre-pass dataset. Only then prepare the corrected-data training experiment.

### Packaging pass separately

1. Decide declared-accounting-level semantics, including mixed generic packages and cases where only one side of the hierarchy is explicitly allocated.
2. Inspect the83 real multi-type candidates (40 have pallet/other patterns), the149 synthetic multi-type candidates, and the source contracts that establish nesting. Do not mistake product counts/contents for shipment packages.
3. Make a per-goods decision recording selected level, printed total, printed per-container counts, other-level context, and unresolved source contradictions. All counts must use the selected level; no equal division or mass/capacity inference.
4. Update labels, placements, typed package contracts, host sampling/physical calculations, and extraction descriptions/mapping prompts together. Audit rendered counts against that same decision.
5. Recompute relation metrics after any gold-level change. Otherwise a semantic policy change could be mistaken for improved model accuracy.

The two passes share source evidence but have different acceptance conditions. Excluding a packing equation from a free-text description does not by itself choose which package level should populate structured targets.

## 11. Preserved-state receipts

SHA-256 before and after this audit/probe matched:

| File | SHA-256 |
|---|---|
| Active train.jsonl | `eb2cf00e3190dd5931507447b54df9101edd4f1933dd97d421bc491a06ee0fa1` |
| Active validation.jsonl | `8c4c740996590f8f01c7e1d2d9af8e8f32785766aeff2e53e475a67cb74303d7` |
| Template ownership.yaml | `6a726c638c1d0583ec038600ca8a0caa5ec9270adc28b0aa270c776732ccd384` |
| Template manifest.json | `bd649f49a06e9d8d105e919417073458e67157074846030a9a34a38849ced88a` |

Reproducible scripts and all outputs are in [the audit directory](analysis/description-boundaries-packaging-audit-20261009/). No paid review or fresh synthesis was required for these experiments.
