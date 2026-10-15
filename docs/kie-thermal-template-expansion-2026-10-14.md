# Thermal and non-operating reefer template expansion

Status: **complete and admitted**. All thirty new sources and ninety descendants
are content- and geometry-certified and included in the reviewed registry.
The cohort is **25 thermal sources (12 frozen, 13 chilled) plus five explicit
non-operating-reefer sources**. Each produced three complete descendants.
There are no unresolved sample or source findings. The current training dataset
is unchanged; these are validation descendants and new synthesis templates.

The verified starting registry had 383 templates, including nine frozen and
four chilled sources, two ventilation sources and six other reefer sources
without a supported setpoint. The registry now has 413 templates:
21 frozen, 17 chilled, six temperature-bearing ventilation sources and seven
explicit NOR sources. Four older reefer-without-setpoint sources remain
operation-unspecified; they are not reclassified as NOR.

## Deliverables and scope

- [Frozen source selection](analysis/thermal-template-expansion30-20261014/selection/frozen-selection.json): all thirty identities, original OCR/PDF paths and hashes, page counts and selection evidence.
- [Final source inventory](analysis/thermal-template-expansion30-20261014/final-source-inventory.json): source product, thermal family, equipment, HS presence and ventilation.
- [Source followed by all rendered descendants](../artifacts/kie-synthesis-production/curated-v7-thermal-expansion30-validation-v1/samples.md).
- [Coordinate-enriched rendered descendants](../artifacts/kie-synthesis-production/curated-v7-thermal-expansion30-validation-v1/positions-reflow-v1/samples.md).
- [Complete individual content certification](analysis/thermal-template-expansion30-20261014/manual-review.json), [independent validation](analysis/thermal-template-expansion30-20261014/validation.json), [visual geometry certification](analysis/thermal-template-expansion30-20261014/geometry/visual-certification.md) and [geometry gallery](analysis/thermal-template-expansion30-20261014/geometry/GALLERY.md).
- [Production validation configuration](../configs/synthesis/mpci_bl_curated_v7_thermal_expansion30_validation.yaml), installed with the registry admission.

Sources span sixty supplied PDF pages and eight thermal carrier presentations.
Every source is a filtered B/L or sea waybill with no more than five pages.
All selected source identities and PDFs are excluded from current validation.
The twenty-five thermal sources are outside the prior 677-record authority;
two NOR sources are also new, while three are reviewed but previously
uncompiled real-training sources. The new synthesis authority consequently has
644 training-source rows plus the same sixty validation rows, not 660 model
training documents. No training has been started.

The twenty-five thermal sources have 35 containers: twenty-one single-container
sources, three three-container sources and one five-container source. The five
NOR sources each have one container. The ninety descendants therefore contain
120 container records, including 105 OCR-grounded carrying setpoints.

## Selected repertoire

The prefix is a navigation key; the inventories above hold full IDs and PDF/OCR
links. One goods item means one jointly accounted consignment, not necessarily
one short product phrase.

| Source prefix | Family/product presentation | Distinguishing coverage |
| --- | --- | --- |
| `56e9e1cf` | Frozen squid | Complete carrying clause; delivery-agent postal continuation; untyped measurement excluded from targets |
| `8923530a` | Frozen buffalo offal | Two product portions, one package total; two eight-digit HS codes; word-total typography |
| `de71b67a` | Frozen shrimp | `40RF96`, separate aggregate owner, carrying −18°C versus recorded gate-in −6°C |
| `1444835f` | Frozen bovine liver | No printed HS; `40 REEF 9'6`; address-only delivery party; repeated booking/B/L identity |
| `9184af53` | Frozen poultry | Three explicit container allocations; row-summed gross/volume; attachment and seal repeats |
| `15329736` | Frozen beef | NCM eight-digit plus dotted HS six-digit presentation; legal/signatory repeats |
| `ea06d095` | Frozen bovine liver | Ten-digit HS; direct `DEG CEL` handling clause and setpoint; secondary pounds omission |
| `940e11db` | Frozen beef | Separate complete TEL subscribers and source-only fax; directly supported gross total |
| `e0129eda` | Frozen buffalo meat | Primary office versus separate factory; buyer/address repeats; delivery contacts on later page |
| `7102f18f` | Frozen beef | Paired eight-/six-digit HS; issue versus onboard dates; private signatory/CPF |
| `dd69725e` | Frozen liver/kidney | One jointly accounted goods item; two ten-digit HS codes; forwarding agent |
| `35afce36` | Frozen liver | Five reefer rows and five explicit allocations; no public HS; identified personal contact |
| `0f03e468` | Chilled garlic | Two airflow printouts, operational/min/max/carrying temperatures, marked postal continuation |
| `48f03224` | Chilled apples | Individual shipper; eight-digit HS; same-as notify; explicit high-cube reefer |
| `ec7035fc` | Chilled grapes | Individual consignee; no PDF-only fields; source malformed-email clause omitted in descendants |
| `fe1eea92` | Chilled apples | Three rows and repeated vents; rider totals; `PLUS 1 DEG' C` |
| `c483b542` | Chilled garlic | Three rows and three airflow printouts; no public HS; product-attached `10KG CARTON` retained |
| `2c30f72b` | Chilled plums | No public HS; numeric B/L distinct from public SCAC |
| `fd541ea7` | Chilled apricot/cherry/peach presentation | One shared box total; private commodity domain, no invented HS label |
| `9a9aeef8` | Chilled pomegranate | English/Spanish product passages; full dotted HS formats; two seals; explicit tare grouping |
| `6ea0c767` | Chilled apples | Declared pallets, no printed ventilation; bounded primary identity and documentary continuations |
| `91c343f7` | Chilled apples | Received-for-shipment date distinguished from onboard date; long generated wording |
| `639f54cd` | Chilled apples | Individual shipper; repeated invoice/payment identities and source-page ownership |
| `8fc5897f` | Chilled apples | Cross-page carrying instruction; 20 CBM/H; declared boxes versus contained pallets |
| `e6b6c931` | Chilled bananas | Empirical 13°C; complete per-box product qualifiers; no public HS |
| `7c103968` | Explicit NOR / mechanical goods | `40NOR`; corrected fused name/postal seam; complete repeated contacts |
| `e16b3787` | Explicit NOR / lighting goods | Separate `40'` and `NOR` lines; damaged same-as instruction; source-shaped goods passage |
| `9d3de4a7` | Explicit NOR / electrical goods | Postal-only shipper; ten-digit HS; transit instruction; no invented measures |
| `7484c804` | Explicit NOR / industrial parts | Two pages, repeated NOR/equipment; three product passages and two unique HS owners |
| `341fc437` | Explicit NOR / ceramic-industry spares | `40' / NOR`; shipment quantity/allocation; no guessed measurement unit; correctly owned office locality |

## Source selection and equipment operation

Select from the filtered B/L and sea-waybill pool with at most five pages,
excluding all current registry sources and all validation sources. Establish
thermal commodity, temperature, ventilation, packaging and allocation ownership
from the complete retained OCR. PDFs may clarify layout but cannot restore
omitted scalar values.

A document that prints reefer equipment but no temperature is not automatically
NOR. All five additions actually print NOR. Refrigerated remains the hardware
category; ordinary compatible registry goods are sampled without introducing
operating temperature or ventilation. Hapag-Lloyd describes NOR as insulated
reefer transport without active temperature control in its
[official NOR explanation](https://www.hapag-lloyd.com/en/services-information/news/2022/07/jul21-nor-non-operating-reefers-quick-quotes.html).

An initially narrow search missed `40' / NOR` and `40'HQ NOR`. Broadening the
scan found enough explicit sources; the superseded four-NOR/one-ordinary-reefer
proposal remains in the selection-revision audit. The selected five require no
inference of power-off state from absent temperature or from commodity alone.

## Thermal sampling and ventilation

Existing registry goods sampling remains in effect, rather than limiting all
descendants to the original product. Thermal identities, compatible refrigerated
equipment, declared packaging and quantities, physical loads and carrying
settings are sampled together. The language model generates names, postal text,
goods wording and relevant contacts around those facts, through text requests
and native structured Pydantic outputs. Provider: GLM-5.3-Flash, low reasoning,
concurrency eight; no switch to an expensive model was required.

Six thermal sources print no HS code. Reviewed registered commodity codes guide
their generation privately but are never inserted into OCR or labels. Private
frozen-liver context is resolved from the full authoritative hierarchy, not
guessed from the short leaf name. Global thermal draw pools remain 82 frozen
and 73 chilled identities. The existing source-bundle profile preserves the
source's package/accounting topology; this is not a new template-specific goods
generator.

Empirical fresh-produce tuples outside the generic thermal pool preserve their
supported carrying settings, including bananas at 13°C and pomegranate at
5.5°C. Registered chilled garlic uses the existing generic 0°C/zero-airflow
profile rather than the source −3°C/15-CBM tuple. Generated text and labels are
consistent with the sampled tuple; this campaign did not silently alter that
sampling policy.

Four new templates explicitly print ventilation: `0f03`, `fe1e`, `c483` and
`8fc`. Their twelve descendants preserve those handling-field presentations.
Ventilation is represented publicly in **handlingInstructions**, not a new
standalone schema field. A canonical m³/hour value remains in private sampling
provenance. No field is added to templates whose source does not print it.
All selected airflow presentations use CBM/hour. The newly shared parser also
has separately tested CFM, M3/M³, Celsius/Fahrenheit and explicit closed-state
branches; this does not falsely claim a new CFM or Fahrenheit source layout.

## Corrections and prevention at the owning layer

| Diagnosed issue | Corrected layer and validation |
| --- | --- |
| Standard-height 40-foot reefers were unsupported by the capacity classifier | Added an explicit published cold-box family, not a GP/HC fallback. SEA BOX's [manufacturer specification](https://www.seabox.com/products/detail/SB561.6-40ft-refrigerated-container1) supplies 26,130 kg payload and 58.8 m³. Existing configurable margins remain; these are bounded-synthesis reference values, not universal operational limits. Actual five-NOR/15-plan probes pass. |
| Ventilation punctuation, unit conversion and qualitative closed states were inconsistent between sampler and renderer | One explicit-unit/state reader/renderer; conflicting, negative, negated and unqualified rates fail rather than infer units. CFM values must fit the source numeric precision exactly. Existing CBM rendering stays unchanged. |
| Explicit no-HS thermal goods lacked private classification support; one exact pallet source was rejected solely for lacking ventilation | Reused reviewed source-bundle capabilities with registered private commodity context; no public HS field added. Exact source packaging may remain without invented airflow; broad donor restrictions remain. |
| `40RF96` was interpreted as a 96-container aggregate | Corrected source ownership: individual equipment row plus independently owned `01X40'HC` aggregate. The parser's honest NOR/FCL framing vocabulary was separately tested without allowing unrelated prose. |
| A formatted HS value restored to its original value left a transient changed-path proof | Fixed final HS projection membership. All three pomegranate text/targets are identical before/after; proof now matches independent replay. Four positive/negative regressions cover unchanged, mixed changed/unchanged and conflicting digits. |
| Three `ea06` handling labels retained −18 despite correctly rendered −19.5/−19.5/−23.5°C | Replaced the old handling-label gate with the shared explicit temperature reader, retaining UNKNOWN failure and AS-PER-ABOVE behavior. Exactly three labels change; all ninety planned rendered surfaces and other target facts are unchanged. Independent validator now rejects stale/absent/unit-stripped carrying labels. |
| New text inherited the source's zero-byte company/address seam in three NOR descendants | Added a hash-bound reviewed same-party name→postal separator recipe. Only confirmed adjacent owners qualify; exact source replay stays byte-identical. Source metadata now records the seam; regular edit receipts and reflow retain ownership. Invalid hash, party, occurrence, kind, adjacency and separator controls fail. |
| One office-location alias followed discharge port instead of its party locality | Corrected the source-only auxiliary binding using existing party_location capability; three descendants re-rendered and completely reopened. No new target field. |
| Two individual-party requests returned companies | Field-scoped LLM corrections produced personal names; contact dependencies were rechecked, full documents re-rendered and reopened. Source instructions already explicitly require individual identity. These linguistic proposals were not silently accepted. |
| One description introduced a conflicting explicit shipping temperature; one generated postal site duplicated its locality | Exact bounded sample corrections removed the unsupported shipping-temperature phrase and changed the fictional site name. All other values and failed proposals are preserved; final text and labels were re-reviewed. No fuzzy address cleanup or blanket regeneration. |
| A manual-review inventory emitted notes as a list rather than the admission string contract | The strict read-only admission gate rejected it. Metadata-only fix retains the lists as observations and renders notes as text; every content and provenance hash is unchanged. |

Source rebasing also corrects previously unsupported roles/units and repeat
ownership before generation. Examples: freight-payment locality is not receipt;
an empty delivery box is not a postal continuation; raw measurement `50.000`
does not gain units from ventilation `CBM/H`; 9'6 is not a seal; and a printed
gate-in snapshot is not the carrying setpoint. Full-source package row sums are
used only with proven quantity/container ownership, not duplicate text matches.

Bounded source-only omissions are explicitly recorded: conflicting secondary
pounds presentations in two sources, one genuinely malformed double-at email
clause, and the uncertain detached NINA activity fragment. Original OCR remains
unchanged. No broad arithmetic cleanup, guessed email reconstruction or global
ampersand splitting was introduced.

Twenty-seven new authority rows and three already-known NOR authority rows were
adjudicated for template use. For example, unsupported real-gold collect wording
is excluded where the OCR has only empty freight headers. These template-source
corrections do **not** silently edit the current 600-real training set; the
differences are documented in the per-source reviews.

## What was examined and what a pass means here

Every complete source OCR/current source target was opened. Every one of the
ninety complete rendered OCRs and complete candidate targets was opened for
individual review. The four review inventories contain 24, 24, 27 and 15
sample decisions. Changed documents were reopened after correction, and
unchanged documents' content digests were checked before refreshing file pins.

Separate provider review covers every final descendant. Nineteen remaining
review claims were individually adjudicated as non-defects under explicit user
policy: packing-fill arithmetic, exact HS-to-product classification, retained
country-style reference prefixes and a fictional brand spelling. Description
and separately declared structured values still agree with printed OCR. No
new unlabelled explicit shipment total is accepted by that reasoning. The true
temperature/identity defects instead received fresh post-fix review.

The final certification requires more than a model's agreement: exact source/edit
and target replay; complete schema/direct-target checks; printed optional-field
presence; unit-aware thermal/ventilation assertions; original private-identity
residual checks; source/Paddle coordinate ownership; independent geometry; and
hash-pinned whole-document review decisions. Missing/stale decisions, changed
publication bytes or a non-PASS component prevent admission.

## Coordinates and geometric evidence

Source coordinates were independently replayed against raw pinned Paddle
geometry: 30 sources, sixty pages, 2,866 content lines, 2,339 known coordinates
(81.61%). All 293 source-coordinate corruptions reject, including forged
self-consistent boxes and same-text occurrence swaps. Existing three real-NOR
alignments are byte-identical to their prior authority.

The ninety descendants have 8,757 content lines and 6,994 coordinate-bearing
lines (**79.87%**). On the same descendants, conservative anchor-only rendering
provided 6,833 coordinates; established reflow adds **161**, with **zero lost
coordinates or introduced collisions**. Unknowns remain explicit ` ||`; text is
not dropped and another occurrence's anchor is not borrowed.

| Field class | Known / lines | Coverage | Explicit unknown |
| --- | --- | --- | --- |
| Goods-description lines | 136 / 148 | 91.89% | 12 |
| Address lines | 439 / 795 | 55.22% | 356 |
| Other lines | 6,419 / 7,814 | 82.15% | 1,395 |

The twelve goods gaps are confined to `0f03`, `9184` and `c483` descendants:
flattened mixed-field text or incomplete repeated source anchors. Repeated
containers, quantities, countries and postal tables are checked for correct
source page/line/region rather than matching equal strings globally. Address
coverage is a real limitation of this cohort's measured source support, not a
claim that all addresses received safe positions.

All ninety receipts pass numerical geometry/ancestry checks. All 24 full-size
plots across twelve selected families were actually opened and inspected,
covering NOR, ventilation, five-container rows, long goods, continuation pages
and sparse anchors. No geometry defect requires a revision. The separate
geometry pass rejects 180 mutation controls; final domain validation's 307
controls include its own geometry and semantic mutations. Those overlapping
test groups are not added together as distinct discovered defects.

## Tests, measured impact and protected state

- Main relevant production suite: **1,770 passed in 70.89 seconds**; Ruff and diff whitespace checks pass.
- Admission/archive/domain-validation suite: **149 passed**. This overlaps parts of the main suite and is reported separately.
- Independent final ninety-sample validation: **13.57 seconds**, peak RSS **800.41 MiB**, all 307 corruption controls rejected.
- Current-code old-registry regression: **383 sources / 1,149 plans PASS**, **40.31 seconds**, peak RSS **800.41 MiB**, zero paid requests. All six assets and old metadata remain unchanged before promotion.
- Fresh installed-registry execution: **413 sources / 1,239 plans PASS**, **39.07 seconds**, peak RSS **488.44 MiB**, zero paid requests. This loads the promoted production configuration and installed authority, verifies every case's six assets and all old metadata, and rechecks the protected training hashes. It is a real post-admission CPU run, not a config-only check; [receipt](analysis/thermal-template-expansion30-20261014/regression-after-admission.json).
- Thermal warm planning/preflight benchmark: **2.785 → 2.735 ms/source** (−1.79%), traced peak **370,788 → 356,801 bytes** (−3.77%). The final paired handling-gate comparison is 2.820 → 2.745 ms/source; these are warm component measurements, not remote generation throughput.
- Standard-reefer capacity microbenchmark: **23.577 → 22.327 ms per 24,000 unchanged-family calls**, same 1,307-byte traced peak.
- HS receipt microbenchmark: **55.142 → 53.158 μs/render**, unchanged 9,646-byte peak; small timing variance is acknowledged.
- Separator rendering: no-recipe overhead **0.33%** (0.369 → 0.370 ms); active reviewed recipe **0.400 ms**. Active-source expansion has a small measured cost; ordinary sources do not get a new heuristic or output change.

All five protected current training files retain their initial hashes. Original
source OCR/PDFs and sixty validation records are unchanged. The new authority,
templates, validation descendants and audit evidence are separate. No GPU work,
training launch or paid OCR campaign occurred.

## Provider cost and accounting

All **96 recorded provider calls cost $0.08670546**, including rejected wording
proposals, bounded corrections and superseded reviews. This is OpenRouter API
cost, not the Codex session or CPU cost. Full ledger:
[cost-accounting.json](analysis/thermal-template-expansion30-20261014/cost-accounting.json).

| Stage | Calls | Provider cost USD |
| --- | ---: | ---: |
| Initial wording | 30 | 0.02969030 |
| Bounded wording-validation repair | 5 | 0.00271660 |
| Company contacts | 6 | 0.00067676 |
| Postal correction | 17 | 0.00272744 |
| Individual-name correction | 2 | 0.00035820 |
| Rendered review, including superseded reviews | 36 | 0.05053616 |

Provider accounting totals: 469,808 input tokens, including 50,752 cache-read
tokens; 44,649 output tokens, of which 13,382 are reasoning and 31,267 visible.
Reasoning is part of output, not an additional duplicated charge. Accepted-only
totals are not substituted for all-attempt cost.

## Evidence and reusable entry points

The source-specific audit inventories, adjudications and full-reading reports
are under `analysis/thermal-template-expansion30-20261014/batches/`. The
operational/capacity details are in [capability/REPORT.md](analysis/thermal-template-expansion30-20261014/capability/REPORT.md)
and [capacity-standard-reefer/README.md](analysis/thermal-template-expansion30-20261014/capacity-standard-reefer/README.md).
The reusable admission and source/geometry-gate design is documented in
[admission-design.md](analysis/thermal-template-expansion30-20261014/admission-design.md).

Admission uses `scripts/synthesis/reviewed_template_admission.py` with an explicit
cohort context. It does not copy or monkeypatch the old DG campaign's fixed
20/60 counts or privacy migration. Default operation is read-only; promotion
verifies the complete source/content/geometry gates, protected old files and
current schema, stages the new authority, validates the installed registry and
commits its manifest last with a scoped rollback backup.

Final [admission receipt](analysis/thermal-template-expansion30-20261014/admission.json):
383→413, all thirty additions installed, old case assets/metadata and training
files unchanged, zero paid admission requests. The immutable authority and
production configuration are installed. Final registry manifest SHA-256:
`25a38cc76b960c4506b2ad21135e6fa68f366c9f692231a02cdefdb14c32e020`.

