# Current training dataset: coverage and description-boundary audit

Date: 2026-10-11. This is a read-only audit of the dataset pinned by the running
[real600 + synthetic5000 UAE training configuration](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic5000_uae_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml).
No training inputs, labels, templates, production code, hyperparameters or running
processes were changed. External API cost: **$0**.

## 1. Scope and principal conclusions

The current dataset is
[`mpci-bl-real600-synthetic5000-uae-v7-positions-v1`](../data/curated/mpci-bl-real600-synthetic5000-uae-v7-positions-v1/):

| Cohort | Documents | Source families | Container objects |
| --- | ---: | ---: | ---: |
| Real training | 600 | 600 | 1,251 |
| Inherited synthetic | 2,500 | 300 | 4,475 |
| Newly added UAE synthetic | 2,500 | 358 | 5,877 |
| Training total | 5,600 | Not additive across overlapping cohorts | 11,603 |
| Real validation | 60 | 60 | 120 |

The reusable synthesis registry contains 360 source cases. Two are not represented
by the new campaign's accepted descendants. Cohorts are determined from document
IDs and publication manifests, not guessed from row positions.

Main findings:

1. Common equipment has substantial support, but nominal equipment diversity is
   narrow: only five of 19 supported type categories occur. General purpose is
   89.67% of training container objects. Twenty-foot high-cube has **zero training
   examples**, despite one validation example.
2. Packaging covers 43 of 44 task categories, but 17 observed categories have
   **no synthetic reinforcement** and only one to seven real examples. Paper bags
   have no examples. Generic packages, cartons and pallets account for 72.61% of
   training package rows.
3. New synthesis increases multi-container, open-top, refrigerated, 45ft high-cube,
   DG and membership-only allocation coverage. It does not add mixed declared
   package groups, unknown package counts, partly counted allocations or fallback
   package types.
4. **No target description contains a generic STC / said-to-contain / shipper
   loading declaration.** All descriptions are uppercase, single-line values.
   All 5,000 synthetic descriptions have exact rendered-text support through
   current source projections or their applicable later repair decisions.
5. Transshipment remains sparse: **116/5,600 training documents (2.07%)**, eight
   synthetic source families, and **zero validation positives**. Current aggregate
   validation metrics therefore cannot measure positive transshipment extraction.
6. One confirmed equipment annotation omission remains in a real training
   document. It has no synthetic descendants. Two apparent alias conflicts were
   resolved by PDF inspection and require no label repair.

Exact counts are exhaustive over the published labels. Printed-alias counts are
explicitly provenance-scoped lower bounds; incidental OCR keywords are not treated
as ownership evidence or semantic verdicts.

## 2. Container types, sizes and relationships

Cells below count container objects, not documents.

| Container type | Real | Inherited synthetic | New synthetic | All train | Validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| General purpose | 1,122 | 4,089 | 5,193 | 10,404 | 98 |
| Refrigerated | 45 | 178 | 297 | 520 | 7 |
| Open top | 11 | 13 | 68 | 92 | 1 |
| Platform | 1 | 0 | 0 | 1 | 0 |
| Liquid/gas tank target category | 1 | 1 | 1 | 3 | 0 |
| Type unassigned | 71 | 194 | 318 | 583 | 14 |

The liquid/gas tank bucket is named `PRESSURIZED_TANK` in the target vocabulary;
its enum spelling is not evidence of operating pressure or hazardous cargo.

Fourteen supported type categories are unrepresented: ventilated GP, dry bulk,
named cargo, refrigerated-and-heated, self-powered refrigerated, refrigerated/heated
removable equipment, insulated, fixed/collapsible/complete-superstructure/named-cargo
platform variants, dry hopper tank, dry rear-discharge tank, and air/surface.
These are coverage gaps, not a requirement to manufacture equal support for every
rare type regardless of actual deployment needs.

| Size | Real | Inherited synthetic | New synthetic | All train | Validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20ft standard | 326 | 1,138 | 1,856 | 3,320 | 33 |
| 20ft high cube | 0 | 0 | 0 | 0 | 1 |
| 40ft standard | 35 | 175 | 168 | 378 | 0 |
| 40ft high cube | 812 | 2,968 | 3,470 | 7,250 | 62 |
| 45ft high cube | 4 | 0 | 65 | 69 | 10 |
| Size unassigned | 74 | 194 | 318 | 586 | 14 |

There are 11 observed complete size/type combinations. This is not a claim that
all 95 combinations of the five sizes and 19 types are valid or required.
Of 520 refrigerated containers, 510 are 40ft high cube, nine 40ft standard and one
type-only. There is no labeled 20ft refrigerated equipment. All 69 labeled 45ft
containers are GP and arise from just four source families. Platform is real-only;
tank has only three examples across three families.

### Incomplete classification is not the same as a fallback label

There are **zero `typeDescription` fallbacks**. However, 583 training containers
have neither size nor type. Three further real containers are type-only. There
are no size-only objects.

The missing-category synthetic rows descend from source-incomplete families:
194 containers / 92 documents / 12 families in the inherited cohort, and 318 /
129 / 13 in the new cohort. Some sources print an aggregate mixed-equipment
receipt without assigning each size to an ID. The synthesis contract preserves
that uncertainty instead of distributing sizes arbitrarily.

For example, a four-ID source prints a 1x20 dry + 3x40 HC mixture. Its private
`equipment_mix` contract deliberately does not assign public categories to those
IDs. This is not an alias-normalization failure. Whether a common type can be
recovered separately from an unknown size assignment is a distinct policy question.

### Confirmed omission and rejected false alarms

Training row 393,
[`doc_8c6da3...` OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_8c6da3ddb7c7d91a71a36c0a181ce2158d206bd9bddc6e6471176cc0a56dc912/ocr.txt),
has the single container `CXRU1576038`. The OCR explicitly says that the cargo
is stowed in a **refrigerated container**, continuing with **−18 degrees Celsius**
on the following page. The target already retains the temperature and handling
instruction but omits the refrigerated type. Type can be corrected without
inventing length. The source has **zero descendants** in the current 5,000
synthetic records and is outside the admitted 360-template catalog.

The real112 and real355 `40HO` warnings were investigated, not automatically
patched. Both source PDFs visibly print **40HQ**, alongside **1X40HC**. OCR changed
Q to O; the existing GP/high-cube labels are supported by the separate OCR-retained
HC declaration and are correct. See the
[two PDF crops and verified source receipt](analysis/current-dataset-coverage-20261011/equipment/source-pdf-render-receipt.json).

### Thermal and DG-related coverage

- 389 temperature-setpoint containers in 303 training documents; all Celsius.
  388 are labeled refrigerated; the remaining one is the omission above.
- 132 refrigerated containers have no temperature setpoint. A reefer label does
  not authorize inventing a setpoint or treating a NOR alias as active cooling.
- No Fahrenheit setpoint targets. Heated/self-powered/insulated families have
  no target examples.
- 152 DG documents have placements linking to 334 GP containers and one
  unclassified container. No current labeled DG links to specialized equipment.
  This describes dataset exposure, not a claim that DG requires specialized equipment.

Full inventory:
[equipment report](analysis/current-dataset-coverage-20261011/equipment/REPORT.md),
[every incomplete object](analysis/current-dataset-coverage-20261011/equipment/incomplete-equipment-rows.jsonl).

## 3. Actual printed equipment aliases

The audit joined all 5,000 synthetic records to their original campaign candidates
and admitted only type/receipt binding surfaces present in final OCR. Real evidence
uses source bindings and previously accepted equipment wording. A type-looking
string in an address, tariff clause or goods specification is not counted as an
equipment alias merely because it matches a regex.

The admitted inventory covers 358 real-training, 2,350 inherited-synthetic,
2,297 new-synthetic and seven validation documents. Consequently the following
are **lower bounds on total printed coverage**, not a claim that an unobserved
alias is absent from every unindexed source.

| Printed group | Real documents | Inherited synthetic | New synthetic |
| --- | ---: | ---: | ---: |
| HC | 110 | 961 | 851 |
| HQ | 32 | 145 | 147 |
| GP | 24 | 616 | 587 |
| DC | 12 | 16 | 25 |
| ST | 13 | 15 | 15 |
| RH | 6 | 37 | 64 |
| HR | 5 | 10 | 10 |
| NOR | 5 | 0 | 0 |
| BX | 2 | 3 | 1 |
| DV | 1 | 0 | 1 |
| DY | 1 | 0 | 0 |
| RF | 2 | 0 | 0 |
| KL | 0 | 1 | 1 |
| UT | 0 | 8 | 21 |

Counts can overlap: one document can print several styles. There are 312 distinct
owned equipment/receipt surfaces, 265 of which resolve a type under the standalone
grammar. Count-only receipts are included in 312 but are not type failures.

Registry support is substantially broader than admitted printed exposure:

- Eight of the 24 compact type aliases are observed: BX, DC, DV, DY, GP, KL, RF,
  UT. The HC/HQ/RH/HR and carrier rules are additional alias families, not members
  of that particular 24-token compact list.
- The detailed ISO evidence consists of `22G1` in seven synthetic documents,
  `45G1` in one real and 27 synthetic documents, `45R0` in one new synthetic
  document, and `22GO` in one validation document.
- Only detailed code families G0/G1/R0 are admitted by this scan, versus 61
  supported detail codes. `45G1` correctly denotes 40ft high cube; it is not a
  literal 45-foot length token.
- NOR remains real-only in the admitted evidence; the known 20HO example is
  validation-only. Current sampling does not reinforce those observed styles.

No admitted single-owner synthetic equipment surface disagreed with its canonical
target. This does not independently classify unowned or mixed-owner strings.
See [surface CSV](analysis/current-dataset-coverage-20261011/equipment/printed-equipment-surfaces.csv)
and [alias evidence](analysis/current-dataset-coverage-20261011/equipment/equipment-alias-groups.json).

## 4. Package categories, wordings and allocations

The task vocabulary contains **44 approved package categories**, distinct from
the broader 405-entry application registry. Training covers 43/44; validation
covers 14/44. `PACKAGE_BAG_PAPER` is absent from both. Multiwall paper bag is a
different category and has one real training example.

| Package category | Real documents | Inherited synthetic | New synthetic | All train documents | Validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generic package | 200 | 721 | 720 | 1,641 | 18 |
| Carton | 91 | 668 | 553 | 1,312 | 8 |
| Pallet | 112 | 470 | 530 | 1,112 | 11 |
| Bag | 40 | 207 | 129 | 376 | 5 |
| Piece | 19 | 59 | 65 | 143 | 0 |
| Bundle | 10 | 65 | 49 | 124 | 2 |
| Box | 12 | 49 | 47 | 108 | 1 |
| Roll | 11 | 51 | 40 | 102 | 1 |
| Plastic bag | 1 | 24 | 58 | 83 | 0 |
| Ordinary natural-wood box | 1 | 0 | 43 | 44 | 0 |
| Steel drum | 2 | 10 | 21 | 33 | 1 |
| Intermediate bulk container | 7 | 0 | 0 | 7 | 1 |
| Paper bag | 0 | 0 | 0 | 0 | 0 |

Exact row total: 5,600 train package objects = 5,590 categorical, nine fallback
and one quantity-only untyped. Generic package has 1,642 rows in 1,641 documents;
row and document counts must not be conflated. Validation has 59 package objects.

Seventeen observed categories are **real-only**, with one to seven examples each:
jumbo bag, multiwall paper bag, block, fibreboard box, bulk liquid, powder bulk,
grain/granular bulk, coil, crate, fibre drum, plastic drum, flexibag, flexitank,
intermediate bulk container, octabin, skid and unpackaged single unit.

New synthesis adds 43 ordinary-wood-box examples, but from only two source
families; all 21 new steel-drum examples come from one family. More examples of a
category are not equivalent to more layouts or source contexts.

### Printed package aliases

For the new cohort, 1,669/2,500 package rows have type-only owned edit spans. Those
yield 68 normalized surfaces across 24 categories. The other 831 rows are outside
this particular mutable-span alias inventory, not incorrectly typed.

| Owned alias | New documents |
| --- | ---: |
| CTN / CTNS | 18 / 36 |
| PKG / PKGS | 26 / 37 |
| PLT / PLTS | 19 / 36 |
| CARTON / CARTON(S) / CARTONS | 70 / 32 / 221 |
| PALLET / PALLET(S) / PALLETS | 57 / 45 / 297 |
| BAG / BAG(S) / BAGS | 28 / 16 / 90 |

Two realism findings are distinct from incorrect extraction supervision:
`BALE NON COMPRESSEDS` occurs in ten new documents; `WOODEN NATURAL WOOD ORDINARY
BOXES` in 24. These mechanical wordings are not category/quantity contradictions,
but more natural printed aliases would improve realism in later generation.

### Allocation and nested-packing exposure

All 5,660 train/validation documents have exactly one goods entry. Eight real
training documents have multiple declared package rows; seven have different
declared package categories. No synthetic record has multiple declared package
rows. This is compatible with the present single-goods scope, but limits exposure
to mixed packaging within one goods accounting group.

| Document trait | Real | Inherited synthetic | New synthetic | All train | Validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Every emitted allocation counted | 543 | 2,385 | 2,260 | 5,188 | 49 |
| Membership only, no allocated counts | 21 | 77 | 175 | 273 | 4 |
| Partly counted allocations | 7 | 0 | 0 | 7 | 1 |
| No allocations | 29 | 38 | 65 | 132 | 6 |
| One container | 378 | 1,748 | 1,450 | 3,576 | 30 |
| Multiple containers | 193 | 714 | 985 | 1,892 | 24 |
| No containers | 29 | 38 | 65 | 132 | 6 |

Multi-container share improves from 28.56% in the inherited synthetic cohort to
39.4% in the new cohort. Synthetic counts reach 14 containers, but the three real
training cases with 39–40 containers have no comparable synthetic reinforcement.

There are 83 training documents across nine source families matching explicit
contained-count contracts. This is a conservative contract-backed subset of
nested-packing exposure, not an exhaustive count of every inner/outer mention.
The declared shipment package unit remains the accounting/placement unit;
an embedded bag capacity does not automatically change an outer-pallet label.

All 5,000 synthetic fully quantified package/allocation sums balance. Six real
train documents and one validation document have unequal emitted sums because
the source describes unallocated portions or the OCR-retained container roster
is incomplete. For example, 40 containers are declared but one ID is malformed,
so 39 recognized IDs account for 38,610 of 39,600 bags. These must not be repaired
by automatically distributing or balancing quantities. This is an extraction
task, not a requirement that incomplete OCR account for every physical package.

Full 44-category table, quantity ranges, alias evidence and the seven unequal-sum
cases are in the
[packaging report](analysis/current-dataset-coverage-20261011/packaging/REPORT.md)
and [category CSV](analysis/current-dataset-coverage-20261011/packaging/category-coverage.csv).

## 5. Goods-description boundaries and formatting

The applicable
[main-product-passage policy](kie-real-baseline-label-policy-2026-10-05.md#boundary-clarification-main-product-passage-2026-10-09)
excludes loading/package introductions, standalone accounting, separately owned
Marks and documentary fields. It preserves actual main product wording,
including embedded capacities, packing specifications, qualifiers and genuine
continuations. It is not a blacklist that deletes every count or package word.

### Corpus sweep and direct source checks

- All 5,660 descriptions are present, uppercase and single-line.
- **Zero** contain `SAID TO CONTAIN`, STC/S.T.C. or the tested shipper loading/count
  declarations.
- A broader count-led screen found **67 descriptions across nine source families**
  (seven real, 25 inherited synthetic, 35 new synthetic). The complete target
  values and their source-product contexts were inspected. They are product-led
  phrases, not standalone loading introductions.
- Packing-related wording occurs in 43 real, 485 inherited synthetic, 482 new
  synthetic and nine validation descriptions. Such mentions are not automatic
  policy violations; the relevant distinction is passage ownership.

Examples:

```text
OCR: 1 Container Said to Contain 260 BALES
     260 bales of sisal fiber ug grade 100kg each

Target: 260 BALES OF SISAL FIBER UG GRADE 100KG EACH
```

The first line is excluded. The second is the actual product/capacity sentence.

```text
OCR: 1 Package(s) 01 UNIT NEW RANGE ROVER AUTOBIOGRAPHY

Target: 01 UNIT NEW RANGE ROVER AUTOBIOGRAPHY
```

The generic package wrapper is excluded; the actual counted product wording stays.

```text
OCR: 1800 CARTON(S) X 13.61 KGS OF IQF WILD
     BLUEBERRIES FROZEN
     (NET WT: 24,498 KGS)

Target: 1800 CARTON(S) X 13.61 KGS OF IQF WILD BLUEBERRIES FROZEN
```

The attached capacity/product phrase is retained; the separate net-total line is
excluded from description. This source was explicitly retained during the earlier
boundary repair. It is not a reason to strip every count-led product phrase.

Source/target inspection packets are under
[description-candidates](analysis/current-dataset-coverage-20261011/description-candidates/).

### Exact projection and historical-repair reconciliation

All **360 current source-description contracts** compile against their exact OCR
and source targets. All **2,500 new descriptions** match current source-block
projections through saved render edits, including exact rendered byte positions.

The inherited cohort requires recognizing later authorized repairs, rather than
assuming its original candidate receipts remain current:

| Verification route, inherited samples | Count |
| --- | ---: |
| Current source-block projection through original render edits | 2,350 |
| Exact projection using later field-boundary repair proof | 90 |
| Previously adjudicated description fragments at exact current offsets | 35 |
| Previously adjudicated fragments relocated to their unique verbatim occurrence | 24 |
| Later explicitly adjudicated removal of a standalone package-total footer | 1 |
| Total verified | 2,500 |

Initial historical replay disagreed in seven targets and could not replay 143
old edit lists directly. Those are **not 150 newly defective labels**. They are
resolved by the named later repair evidence above. For example, a previously
reviewed `TOTAL: 491 CARTONS` tail remains excluded from the current yarn target,
and `6 PACKAGES TOTAL` remains excluded from the loader target as explicitly
adjudicated in the packaging follow-up. No new target changes were made.

Repeated fragments cannot be relocated by arbitrary string matching: exact
certified offsets must still match. Relocation is allowed only for a unique
verbatim already-adjudicated fragment, with order and complete target equality
checked. Tests reject changed product bytes, invented label words, shifted
duplicate fragments and labels that disagree with their saved approved selection.

The [replay inventory](analysis/current-dataset-coverage-20261011/description-replay.json)
records each route and retains the initial historical disagreements. These checks
establish exact text fidelity to reviewed membership; they are not a fresh manual
semantic recertification of every field in every document.

### Length and wording exposure

Nearest-rank character percentiles, after current label normalization:

| Cohort | Median | 90th percentile | 99th percentile | Maximum | Distinct descriptions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real training | 30 | 89 | 272 | 1,179 | 523 |
| Inherited synthetic | 367 | 678 | 1,047 | 1,638 | 2,450 |
| New synthetic | 176 | 336 | 750 | 1,410 | 2,388 |
| Validation | 38 | 108 | 563 | 563 | 59 |

Synthetic wording remains much longer than typical validation wording. The new
cohort moves toward shorter descriptions, but adds no very-long real-layout
container rosters. Repeated descriptions in some constrained vehicle/machinery
families are not duplicates of complete documents; they limit lexical variability
within those families.

## 6. Transshipment coverage and role checks

| Trait | Real train | Inherited synthetic | New synthetic | All train | Validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transshipment-port target present | 11 | 57 | 48 | 116 | 0 |
| Share of documents | 1.83% | 2.28% | 1.92% | 2.07% | 0% |
| Positive source families | 11 | 6 | 8 | 11, with eight supplying synthetic positives | 0 |
| Distinct literal port-name targets | 8 | 41 | 42 | 80 | 0 |
| Transshipment country target present | 0 | 0 | 0 | 0 | 0 |

Distinct literal names are not necessarily distinct physical ports: OCR spelling
and terminal/locality wording can differ. A source family with an intermediate
route slot produces variable port names; country labels remain absent where
their own OCR evidence/binding is absent. Do not fill countries from geographic
knowledge simply to improve target coverage.

All 11 positive real-source route contexts were opened. They include explicit
transshipment boxes and narrative multi-vessel continuations, not only keyword
presence. Examples include New York → Aliaga → Alexandria; Gothenburg → Hamburg
with different pre-carriage/main vessels; and Port Klang → Jebel Ali → Sokhna.

The Manila family legitimately has both loading and transshipment as Hong Kong:
the OCR explicitly says `T/S CARGO FROM MANILA ... TO PORT SAID ... VIA HONG KONG
BY SEA`. Sixteen current documents descend from that real source including the
source itself. A global rule forbidding transshipment from equaling another route
role would incorrectly reject these labels.

The keyword screen finds 438 training and four validation documents with cues,
but only 110 of the 116 positive targets have those cues. Conversely, 328 training
documents have cues without a transshipment target. Blank template headings,
carrier boilerplate and even an email domain account for many such hits; the
count is **not an omitted-label count**.

One real route annotation deserves a focused consistency decision:
[`doc_a67afe...`](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_a67afe13d38550e2bf86ed7218e312e2b1a087b9a55ecd72577b0704b287b9a6/ocr.txt)
labels EL DEKHEILA as both discharge and transshipment from `For Transshipment To
EL DEKHEILA`. The wording alone does not clearly establish an intermediate vessel
change, unlike the Manila narrative. Two other source contexts print repeated
destination wording under route/transshipment headings without such a target:
`a5265d59...` (16 current source/descendant records) and `6c5e700d...` (11).
These are a bounded role-boundary review, not authorization to infer an
intermediate port or delete every equal-location target. No route label was changed.
The potentially questionable positive `a67afe...` has no synthetic descendants.

Evidence: [all positive routes and cue contexts](analysis/current-dataset-coverage-20261011/transshipments.json),
[unlabeled cue families](analysis/current-dataset-coverage-20261011/unlabeled-transshipment-cues.json).

## 7. Other categorical/fact coverage relevant to the next experiment

| Trait | Real | Inherited synthetic | New synthetic | All train | Validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Negotiable | 48 | 133 | 200 | 381 | 7 |
| Non-negotiable | 544 | 2,324 | 2,280 | 5,148 | 51 |
| Negotiability unknown | 8 | 43 | 20 | 71 | 2 |
| Notify same-as-consignee | 136 | 619 | 675 | 1,430 | 19 |
| Dangerous-goods documents | 17 | 35 | 100 | 152 | 1 |
| Multiple HS codes in the one goods entry | 62 | 233 | 358 | 653 | 3 |
| Goods gross-weight target | 506 | 2,098 | 2,109 | 4,713 | 48 |
| Goods net-weight target | 135 | 339 | 489 | 963 | 15 |
| Goods volume target | 308 | 1,206 | 1,215 | 2,729 | 33 |

Negotiability unknown means the required field is present with null, not that
an arbitrary missing output should become non-negotiable. Multiple HS codes do
not imply multiple goods; all current records intentionally remain single-goods.

The training union contains 1,354 distinct HS-code strings across 66 chapters.
New synthesis covers 777 strings versus 690 in the inherited cohort, but both
synthetic cohorts cover only 38 chapters. Count strings rather than pretending
that different national code lengths are identical classifications.

DG targets contain 108 distinct UN-number strings and seven hazard categories:
corrosive 20, flammable liquids 39, flammable solids 12, gases one, miscellaneous
32, toxic/infectious 38, and oxidizing/organic peroxides ten. Explosives and
radioactive have no examples. These counts concern the target hazard taxonomy,
not distinct regulatory classes within each broad category.

Packing-group support is real-only: seven low-danger and two medium-danger
training entries; no high-danger target. Neither synthetic cohort provides
packing-group labels. Flashpoint is present in two real training documents and
subsidiary hazards in one, with no synthetic reinforcement. These are coverage
gaps; this audit did not establish that those values are printed but omitted in
every DG synthetic sample.

Freight is similarly uneven: 3,960 prepaid, 971 collect, two payable-elsewhere
and 667 unspecified training targets. The two payable-elsewhere cases are
real-only. A lack of synthetic examples is not evidence that an unsupported
value should be filled in a source document.

## 8. Recommended follow-up, separated by purpose

### Small data-quality follow-up

1. Correct the isolated OCR-supported refrigerated type, retaining unknown length.
   This is a one-label issue, not a regeneration campaign or propagated template defect.
2. Adjudicate the narrow repeated-destination/transshipment-role comparison before
   treating it as additional positive transshipment training support.
3. Improve the two mechanical package alias wordings for future synthesis if
   realism is prioritized. Existing category/quantity targets are not contradicted.

No repair was applied while the current training run uses this hash-pinned dataset.

### Coverage follow-up

1. Prioritize 20ft high-cube and the already relevant specialized types, not only
   more GP/40HC rows. Separate equipment-family coverage from printed-alias coverage.
2. Reinforce task package types that are real-only; include paper bags through
   genuinely capable templates. Spread rare types across layouts where possible.
3. Add transshipment-positive layouts/narratives. Create a separate held-out
   positive evaluation slice from unused real sources; do not silently alter the
   current validation set mid-comparison or train on its sources.
4. Consider mixed package rows within one goods group, unknown counts, partly
   counted allocations and longer container rosters as distinct capabilities.
   Do not force them onto templates that cannot print or supervise them coherently.
5. Add packing-group/flashpoint/subsidiary-hazard support only in capable DG
   templates with explicit printed values. More UN-number variety alone does not
   teach these optional outputs.
6. Preserve the current main-product-passage policy and its exact projection;
   do not globally delete embedded packing or count-led product identities.

## 9. Verification, reproducibility and audit boundaries

Input pins verified before/after read-only inventory:

```text
train.jsonl      08e44d14b371d517b711b5ef2cde0c99cdb8e7d9a44f0839e8492762db693075
synthetic.jsonl  1629c9274ed90f9f77bfb83569bf57b2c73cae2f163d308a6bf00e5961df93d9
validation.jsonl 2948e9001d2429c63e9e918597d9dd10e7ebec3b26d9f4c1758a1dfe168318be
```

The root inventory also verifies each raw and positioned text hash. Independent
package bookkeeping reproduces category totals; package/placement structure checks
find no vocabulary violations or broken target container references. All 360
source description contracts compile; all 5,000 synthetic descriptions resolve
through exact projection or named approved repair evidence. Four membership
controls plus changed-byte/invented-word mutations pass/reject as intended.

Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python docs/analysis/current-dataset-coverage-20261011/inventory.py
PYTHONPATH=src .venv/bin/python docs/analysis/current-dataset-coverage-20261011/check_descriptions.py
PYTHONPATH=src .venv/bin/python docs/analysis/current-dataset-coverage-20261011/equipment/audit_equipment.py
PYTHONPATH=src .venv/bin/python docs/analysis/current-dataset-coverage-20261011/equipment/summarize_aliases.py
.venv/bin/python docs/analysis/current-dataset-coverage-20261011/packaging/audit_packaging.py
```

The full description replay took 51.33 seconds; packaging audit 19.53 seconds with
45,608 KiB peak RSS. These are audit costs, not measured training speed changes.
No model was loaded, no GPU kernel/training configuration was altered and no paid
review/generation calls were made. All writes are report/analysis artifacts.

This audit establishes exact coverage and bounded description-fidelity findings;
it does not claim a fresh full-document semantic certification or measured model
performance on categories absent from validation. The concrete label defect and
narrow role question are explicitly separated from missing training exposure.

## 10. Follow-up: what can actually be resampled? (2026-10-11)

This follow-up answers the user's questions about the active sampler, not just
observed counts. It remains read-only: no dataset, policy, production code,
template or configuration was changed, and no generation API was called.

### Equipment is mutable, but not an independent registry draw

`ScenarioCatalog._donors` selects compatible real train-fit cargo observations.
`_cargo` transfers the chosen donor's homogeneous equipment pair into the source's
already-present category fields. The source supplies the layout and accounting
topology, not necessarily the sampled size/type. Goods identities are separately
sampled from registries; the physical/package/equipment support is observation-led.

Source capabilities constrain family, printed measures, package categories and
optional equipment pairs. The default `observed_train_bundle` profile can use
other donors; `source_bundle` and `source_whole_units` pin the physical donor to
the source. Missing source category fields are not fabricated by this transfer.

The accepted new cohort proves cross-source equipment variation:

| Source equipment | Sampled equipment | New containers | New documents |
| --- | --- | ---: | ---: |
| 40HC GP | 20ft standard GP | 1,212 | 433 |
| 40HC GP | 40ft standard GP | 107 | 48 |
| 40HC GP | 20ft open top | 44 | 15 |
| 40HC GP | 40HC refrigerated | 48 | 21 |

These are equipment-category transitions, not permission to add a thermal
setpoint to a source without that printed capability. The thermal-family
compatibility checks remain in force.

20ft standard is not missing: there are 3,320 training entries. The missing
20ft-high-cube category has no train-fit donor and cannot be introduced by merely
raising template quotas. Existing rendering and conservative capacity handling
support that category; the current donor-driven selection does not reach it.

45ft is similarly not a missing renderer. There are four real-training 45HC
documents: one is not an eligible typed-package observation (`SETS` fallback),
and three have no usable HS6 identity. The latter are admitted through explicit
self-bundle exceptions. Those three source templates produce the new cohort's
65 examples. No generic 40HC source has a 45HC donor in its current eligible pool.

Alias rendering is deterministic and source-styled, not quota-controlled. Offline
real-path examples: unchanged `40HQ` stays `40HQ`; changing its pair to 20ft GP
prints `20GP`; ISO `45G1` changed to 20ft GP prints `22G1`; changing `40HC` to a
40HC reefer prints `40' HIGH CUBE REFRIGERATED`. A recognition registry with many
aliases does not imply that the renderer samples all of them.

Evidence: [current equipment sampler/renderer probe](analysis/current-dataset-coverage-20261011/equipment/current-sampling-probe.json).
Code: [donor selection](../src/document_ocr/synthesis/curated_scenarios.py),
[equipment receipt realization](../src/document_ocr/synthesis/template_compiler/descendant.py).

### Packaging is mutable too, but reachability limits the distribution

In the new cohort, 1,670/2,500 documents change category from their own source;
310/358 source families emit multiple package categories. There are 28 explicit
single-category source domains. Thus this is not general source-type locking.

Ordinary sampling is uniform over eligible donor observations, not over package
categories. It transfers the donor's package category. Chemical DG templates use
a separate permitted-category draw; the whole-unit profile can preserve the
receiver's declared category. Family, measures, equipment, source domains and
available registry identities all affect admission. Exact template quotas exist;
package-category and equipment-alias quotas do not.

An offline check of every selected family finds:

| Stage | Package categories |
| --- | ---: |
| Present in eligible cargo-donor pools | 34 |
| Donors with compatible available goods identities | 27 |
| Reachable final category selection under the current configuration | 26 |
| Actually present in accepted new samples | 26 |

Therefore the 17 real-only categories are not simply unlucky random draws.
Seven have candidate donors but no compatible identity receiver: jumbo bags,
multiwall paper bags, powder bulk, granular bulk, flexibags, flexitanks and IBCs.
Fibre drums survive the identity stage for two chemical-DG receivers, but their
current output domains redraw DRUM/CARTON/BOX instead. The remaining real-only
categories have no eligible current donor/receiver combination. Ordinary paper
bags additionally have no real observation at all.

Increasing donor/category probabilities alone cannot reach the missing types
under these unchanged configurations. Resampling can balance currently reachable
types. Expanding reachability needs reviewed compatible cargo support, appropriate
identity domains and capable templates; do not silently remove admission checks.
Category quotas should then be applied to feasible combinations, with an error
for an impossible quota, rather than relying on accidental frequency.

Evidence: [package feasibility inventory](analysis/current-dataset-coverage-20261011/packaging/followup/feasibility.json).

### Description follow-up: three simple count prefixes need separate consideration

The previous zero-STC result remains correct. It does not settle every leading
count-prefix boundary. A refined numeric-prefix check, including parenthesized
plural and common abbreviated package forms, separates the 67 broad candidates:

| Description shape | Real train | Inherited synthetic | New synthetic | Validation | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple `N package-type OF product` prefix | 1 | 2 | 0 | 0 | 3 |
| Leading package count with attached fill/capacity expression | 1 | 2 | 6 | 0 | 9 |
| Other counted product wording, principally PCS/UNIT | 5 | 21 | 29 | 0 | 55 |

The simple-prefix cases are:

- Real `doc_391d1b5f...`: `260 BALES OF SISAL FIBER UG GRADE 100KG EACH`.
  Its separate STC introduction and container row already declare 260 bales.
  It has no current synthetic descendants and is outside the template registry.
- Inherited `syn_full_v7_6c6a724deec6fc1e0dfa8b97`:
  `1173 CARTON(S) OF FROZEN SEA URCHINS, WHOLE ROE-IN-SHELL, ...`.
- Inherited `syn_full_v7_93013b502f5b77969d5a19c4`:
  `896 CARTON(S) OF FROZEN SARDINES, ...`.
  Both descendants belong to the blueberry source family; no new descendant has
  the simple-OF prefix. The source itself has an attached `N CARTON(S) X kg`
  capacity expression, not the same simple-OF shape.

The earlier sisal adjudication deliberately retained the whole capacity-bearing
line. This was not a later STC reintroduction. That historical decision is exact
text-fidelity evidence, not proof that it is the simplest desired boundary.
The two inherited synthetic prefixes are generated wording retained by previous
membership decisions.

Recommended refinement for discussion: start these three targets at the product
identity, e.g. `SISAL FIBER UG GRADE 100KG EACH`, retaining the attached capacity.
Do not globally strip packing words, `01 UNIT` product counts, or source-owned
`CARTON(S) X kg` specifications. This distinguishes a leading declared-count
wrapper from a product specification without deleting information from the middle.
No labels were changed in this audit.

The current wording prompt already says to begin with product wording. A direct
probe of `has_shipment_accounting` shows a specific guard limitation: ordinary
`CARTONS OF ...` is recognized, but the parenthesized `CARTON(S) OF ...` forms
above and the BALES prefix are not. Thus a future refinement needs consistent
package-alias handling and the reviewed product-start rule, not another general
instruction dump. The guard is for newly generated product-only fields, not a
global deletion regex over real OCR or host-owned capacity phrases.

## 11. Donor-independent sampling and explicit alias variability

### Scope and conclusion

Follow-up requested on 2026-10-11: investigate a better sampler, not change the
published dataset or launch synthesis. All experiments below are offline.
Production code, campaign configs, registry snapshots and dataset rows remain
unchanged. External API requests and charges: **zero**.

**Document donors are not an intrinsic requirement for equipment or package
category selection.** They currently provide a convenient joint load example,
but have become the admission authority for several unrelated choices. That
coupling blocks category coverage even where a template can already render the
desired facts. Retain observations as optional load calibration; do not require
a category, alias or commodity/package pair to have appeared in a donor.

There are two separate replacements, not one instruction to delete safeguards:

1. Explicit coverage-aware choice of feasible equipment, package and commodity
   profiles, independent of observed document frequencies.
2. Explicit, versioned choice of compatible printed aliases, independent of
   whichever spelling the selected source happened to use.

We do not need a universal HS-to-product taxonomy, a stowage simulator or a
shipment-compliance engine to achieve our extraction-training goal. We do need
coherent printed facts, clear package accounting and no contradictory labels.
Specialized bulk, thermal and DG scenarios require their own supported scope.

### Why enlarging the donor pool alone is the wrong abstraction

The current path selects a donor in `curated_scenarios.py:_donors`, chooses
registry goods within the donor-conditioned identity domain in `_registry_pool`,
and obtains packages, equipment and scaled measurements in `_cargo`.
`curated_physical.py:_measure_totals` then **unconditionally looks up the recorded
training donor before processing public or private physical measures**. Thus
removing the donor-category filter would not make the full path donor-free.

The host shipment plan must instead carry its complete measured facts:
declared unit/count, totals, container shares, public versus private measures,
and relevant tare/operating facts. The physical compiler can then consume those
facts without having to recover a missing value through a mandatory donor ID.
An optional donor/provenance reference remains useful but is not a substitute
for those explicit facts. Preserve existing rounding, conservation, integer
allocation, net/gross ordering and render-ownership checks.

### Equipment mutation experiment: real physical compiler, not a mocked renderer

A fixed-seed selection of **50 ambient/general-purpose production baselines**
was joined to its exact source contracts and ownership. Each unmodified baseline
passed the current physical compiler. Each was then tested with all five GP
size categories, retaining its accepted load and replacing equipment capacities
consistently in the scenario. Printed mixed-equipment certificates were outside
this experiment. Existing explicit campaign pair whitelists are recorded in
the receipt; this research does not override them in production.

| Proposed GP size | Physical compiler accepts | Rejects |
| --- | ---: | ---: |
| 20ft standard | 37/50 | 13 |
| 20ft high cube | 31/50 | 19 |
| 40ft standard | 49/50 | 1 |
| 40ft high cube | 50/50 | 0 |
| 45ft high cube | 50/50 | 0 |

There were **217 accepted mutations / 250**, with **26 volume rejections** and
**7 missing 20HC tare-support rejections**. The smaller-equipment failures are
not evidence that those categories are unusable: this probe deliberately did
not reduce/re-sample the retained load. The tare failures identify a real
separate dependency when tare is printed; a reviewed equipment specification
can supply that attribute without requiring a matching cargo document donor.

This directly demonstrates that the physical compiler already admits 45HC and
20HC on many ordinary templates despite their current donor-pool limitations.
It does **not** certify new commodity profiles, new package loads, rewritten
goods descriptions or synthetic coordinates. Those stages were not run here.

The reproducible experiment and per-template receipts are in
[probe_physical_and_quotas.py](analysis/current-dataset-coverage-20261011/sampling-redesign/probe_physical_and_quotas.py)
and [physical-and-quota-probe.json](analysis/current-dataset-coverage-20261011/sampling-redesign/physical-and-quota-probe.json).

### Package compatibility: reusable selected-form profiles, not donor co-occurrences

All **44 task package categories** exist in the pinned **405-entry application
registry**. That registry defines names/codes, not a complete commodity/package
compatibility matrix. The older observed HS-heading/signature support and sparse
DG agent catalogs do not supply such a general matrix either.

A research inventory proposed **14 reusable commodity/form profiles across 13
registry HS6 identities**, providing examples for all 44 categories, including
all **18 missing from the new cohort** (17 real-only plus paper bag). **40/44
representative category/HS6 pairs are absent from real training observations.**
These are candidate data profiles to review, not automatic approvals and not a
recommendation to shrink our goods registry to thirteen codes.

Representative proposals include flour/powder in paper or multiwall paper bags,
non-DG liquid oil in suitable drums/IBCs, ceramic articles in protective crates,
and explicitly coiled steel as coils. Different forms or handling roles can
have different compatible packages without depending on their observed source.
Package material/containment is separate from whether a pallet/skid is a
handling unit or overpack; the declared shipment unit remains the public label.

Two attempted shortcuts were falsified:

- Legacy display-name grouping puts liquid bulk, powder bulk and grain bulk
  into one `bulk` family. That is vocabulary grouping, not valid compatibility.
- The existing narrow chemical registry has **18 HS6 identities with both
  liquid and solid DG records**. HS6 alone cannot determine physical state;
  select the actual identity/form, not a blanket chapter-based package rule.

The actual package text helper was exercised on **44 categories × two source
styles × two quantities = 176 cases**. All outputs passed its category-presence
check. It correctly rendered `PAPER BAGS`, `INTERMEDIATE BULK CONTAINERS` and
`CRATES` without a matching donor. It also produced `BALE NON COMPRESSEDS` and
`BULK SOLID FINE PARTICLES POWDERSES`, which the same mechanical check accepted.
Therefore natural singular/plural/abbreviation forms should be reviewed data in
the alias registry rather than blindly pluralized application display names.
These are wording-quality findings, not evidence that existing extraction
labels have the wrong canonical category.

Ordinary packages should not be rigidly assigned to one container size. Express
only real requirements: thermal operation where applicable, liquid versus
solid containment, handling/overpack role, and specialized bulk/tank/flexitank
conditions. Keep DG substance-specific support separate. Our goal is plausible
extraction examples, not certification of a real cargo shipment.

Detailed evidence, profile hypotheses, counterexamples and authoritative
references are in the
[package research report](analysis/current-dataset-coverage-20261011/packaging/registry_profile_research/REPORT.md).

### Alias variability: concrete owned-text probes and remaining grammar gaps

Canonical equipment and its printed spelling are different choices. The source
should contribute a preferred style—compact, ISO, full wording—not dictate one
fixed alias. Draw from compatible aliases with configurable style/frequency
targets, keeping the canonical target unchanged.

There were **64 successful owned-text alias replays** on **16 production
baselines / 15 source templates**. Four same-class forms were tried per baseline:

- 20ft standard GP: `20DV`, `20DC`, `22G1`, full canonical wording.
- 40HC GP: `40HQ`, `40' HC`, `45G1`, full canonical wording.

All target values, unrelated owners and tares remained unchanged. Three
printed-tare baselines exercised four tare receipts. All original baselines
replayed exactly; all variant line counts remained unchanged. A separate **200
seeded draws** used all eight forms and preserved the complete pair on every
draw. These probes test text ownership/classification, not coordinate fit;
longer text still requires the existing reflow/geometry admission stage.

Primary-source checks support carrier `40HR` and `40RH` as high-cube reefers:
[SITC's dimensions](https://sitc.co.jp/vessel/container),
[Hapag-Lloyd's explicit standard/HC distinction](https://www.hapag-lloyd.com/fr/services-information/news/2020/11/peak-season-surcharge--pss----valencia--spain-to-east-coast-sout.html).
`40RQ` is also supported by the pinned OOCL/terminal evidence. The explicitly
agreed annotation convention is decisive here: **if no owned equipment code,
wording or measurement establishes high cube, assume standard height**. Thus
bare `40RF` labels as 40ft standard refrigerated; `40RF 9'6` or `40RF HIGH CUBE`
labels as 40ft high-cube refrigerated. This is our extraction convention, not a
claim that every carrier uses RF identically. Synthetic high-cube descendants
must print their high-cube evidence; do not give indistinguishable bare wording
two different height targets. Alias selection alone never creates a temperature
or ventilation setting.

The alias registry needs an explicit **code system**: carrier `40HR` is not
parsed like ISO size code `40` plus ISO group `HR`. Likewise ISO `RH` is heated
equipment, not the carrier's generic high-cube reefer abbreviation. Neither
reefer wording nor alias selection should create an operating setpoint.

Confirmed prerequisites before broadening alias generation:

- Joined `20HC`/`20HQ` currently classify as standard, while spaced forms and
  `25G1` correctly classify as 20HC. Repair the parser and add regression tests.
- Documented `40OH` and `40RFHC` lack current grammar support; their verified
  semantics should not be lost through generic string normalization.
- The compiled BIC table omits assigned `A0` detail rows. Separately distinguish
  modern pinned-code support from legacy `L5...` forms; absence from the current
  snapshot does not make a historical code universally invalid.

The 95 canonical size/type phrases roundtrip, but this is not permission to
sample all 95 combinations: real equipment/capacity/cargo support is a separate
gate. The current classifier rejects twelve reefer combinations at capacity
admission. Full findings and exact rejection controls are in the
[alias research report](analysis/current-dataset-coverage-20261011/equipment/ALIAS-POLICY-RESEARCH.md).

### Coverage planning experiment and recommended configuration surface

Rather than independently drawing weights and hoping accepted frequencies are
right, first compile a feasible template/scenario inventory and assign coverage
targets over it. Declare the counting unit: documents, goods groups, containers
or alias occurrences. A ten-container document is not ten independent document
examples.

An offline integer allocator assigned **100 document plans** over the actual
passed equipment-mutation edges: **20 per GP size**, at most two per source.
Planning took approximately **1ms**. It rejected an absent-category request and
a shared-scarce-template conflict; **2,304 exhaustive small cases** agreed with
an independent brute-force feasibility reference. This proves that tested
single-category assignment problem, not arbitrary multi-marginal optimization.
Joint equipment/package/thermal targets must be planned over compatible cells;
they cannot be treated as independent guaranteed quotas.

The desired public configuration has three independent concerns:

1. **Scenario coverage:** equipment-pair and package-category weights/floors,
   thermal/DG/source-topology constraints and maximum concentration by source.
2. **Profile support:** pinned, reviewed commodity/form/load domains; observations
   optional for calibration; explicit unsupported combinations.
3. **Printed diversity:** same-pair and same-package aliases, style mix, natural
   singular/plural forms and scope/operating restrictions.

Source capabilities continue to describe actual render freedoms, missing versus
present fields, fixed counts, nested packaging ratios and printed equipment
mixtures. A category whitelist arising from an immutable printed constraint
must not be removed merely because a generic profile admits more options.

### Recommended implementation order and acceptance boundary

1. Add the pinned alias surface data and fix confirmed parser gaps. Validate
   exact categories and alias-owned substitutions; no per-sample LLM alias call.
2. Decouple ordinary GP size selection from cargo donors. Select compatible
   equipment first, then a fitting load; attach reviewed equipment attributes
   when printed. Keep thermal/DG template eligibility and immutable mixtures.
3. Add reviewed commodity/form/package profiles, starting with missing ordinary
   packages. Materialize public/private host facts explicitly rather than
   requiring a donor lookup. Preserve current goods-registry variability.
4. Plan coverage over feasible cells, not accidental observed frequencies.
   Preflight impossible requests before paid generation; record any unmet
   target instead of silently redistributing it to common categories.
5. Run a targeted end-to-end synthesis across supported capability classes,
   including negative controls. Check declarations, description boundaries,
   labels, repeated equipment/package aliases, tares, nested quantities,
   conservation, thermal/DG consistency, and coordinate reflow. Only then
   choose replacements or new samples for the published training dataset.

This does not require rewriting all templates or another open-ended relabeling
campaign. Most ordinary category/alias changes already have owned render spans.
Only sources whose immutable wording or absent fields cannot represent a
requested scenario need further template work. The categorical planning and
alias choices should add **no per-document API call**; existing wording receives
the resulting host facts. Quantitative generation-cost/performance claims await
an implementation benchmark, not extrapolation from these offline probes.

The final streamed-input physical/quota experiment completed in approximately
**3.35s**, with approximately **194MiB process peak RSS**; the package research including its
renderer exercise completed in approximately **15s**, with approximately
**437MiB peak RSS**. These are analysis-process measurements, not estimates of
production generation memory or throughput. Production dataset hashes were
verified unchanged before/after in the saved receipts.

## 12. Follow-up manual variability probes and height-default clarification

The user explicitly confirmed on 2026-10-11: **standard height is the default
when no owned code, wording or measurement establishes high cube**. Bare
`40RF` is therefore standard-height refrigerated; high-cube variants must print
their high-cube evidence. This is the labeling policy, not a request to infer
carrier-specific defaults. Section 11 above is corrected accordingly.

The follow-up experiments are documented exhaustively in
[the manual-probe report](analysis/sampling-variability-probes-20261011/REPORT.md),
with full source OCR, positioned variants and labels in
[the inspection gallery](analysis/sampling-variability-probes-20261011/SOURCE-AND-VARIANTS.md).

- **16 final samples across four source layouts**, including 14 new goods
  descriptions, ten package categories, all five GP sizes and two same-target
  refrigerated alias variants. New declared units/load facts do not come from
  observed document donors. The current compiler's mandatory support lookup is
  supplied by an isolated explicit-fact adapter, not removed in production.
- Two-container allocations preserve declared-unit accounting: **13 + 17 = 30
  paper bags** and **40 + 44 = 84 crates**, with consistent physical row totals.
- Existing reefer temperature **0°C** and ventilation **0 CBM/hour** remain
  unchanged. No alias creates or suppresses an operating setting.
- The joined-20/count-prefix parser experiment passes **845/845** complete
  pairs, versus **821/845** in production, with no original-pass regression.
  Framed ISO surfaces and explicit height conflicts are separate controls;
  full printed surfaces are checked, not only their bare aliases.
- Natural package aliases pass **212/212** owned-render probes across all 44
  categories; ten material/subtype/bare-code negative controls reject.
- Final geometry: **987/1,149 lines (85.90%)** positioned; **25/25 goods-edit
  lines** positioned. Unsupported source-anchor groups retain explicit unknown
  coordinates rather than fabricated positions.
- Eight deliberately corrupted final-sample controls reject. Final samples
  also undergo direct source/rendered/label inspection and coordinate plots.
- GLM-5.3-Flash low-reasoning goods-wording requests cost **$0.00401975 total**
  across both passes, including the first proposals. This is an isolated
  wording test, not a full-synthesis price estimate.

Three first-pass findings are important: dependency ownership alone falsely
treated a count-only container declaration as an alias slot; numeric shipment
context encouraged repeated totals in generated goods; and counted ISO codes
with retained movement suffixes exposed parser framing gaps. The experimental
adapter/request/grammar boundaries were refined and re-run. One dimensionless
specific-gravity wording was also manually corrected with a pinned receipt.
Initial failures are preserved in the report, not presented as first-pass
successes. None was applied to the published dataset.

The tested direction is viable: explicit shipment facts, compatible category
coverage and separately selected reviewed aliases. Before dataset resampling,
integrate those boundaries at the sampler/physical-plan/alias-owner layers and
extend reviewed profile/capacity data for requested categories. The experiment
does not require a source rewrite merely because the sampled equipment or
package differs from the source. Production code/configs/registries and the
published dataset remain unchanged in this investigation turn.

## 13. Fresh thirty-variant confirmation

The follow-up [confirmation report](analysis/sampling-variability-confirmation30-20261011/REPORT.md)
and [complete final inspection gallery](analysis/sampling-variability-confirmation30-20261011/FINAL-SOURCE-AND-VARIANTS.md)
record **30 fresh product variants on 15 new source layouts**, selected with
seed 2026101130 within explicit eligible ambient/thermal strata. Coverage is
five GP sizes plus 40ft-HC reefer, 19 equipment spellings, 11 package categories,
20 package spellings, four singular and four two-container examples. Unrelated
party/route/contact/identifier values are retained controls, not newly generated.

All final changed-field supervision passes complete-text review and actual
schema/replay/description/allocation checks. There are **42 checked owned
equipment surfaces, 76 package surfaces, 96 deliberately corrupted controls
rejected, and 269 relevant existing unit tests passed**. Final geometry is
**2,214/2,594 lines (85.35%)** and **54/54 goods-edit lines** positioned;
explicit geometry-group holds do not become invented coordinates.

Independent complete-text review caught **two orphaned historical cargo facts
in four raw variants**: the original inner `919 PIECES` tally and glove
manufacturer. Inspection-copy, source-hash-pinned auxiliary ownership removes
only those obsolete statements; all labels are exactly unchanged. Original
proposals remain archived, and the source-contract tests reject drift and
removal of active target owners. Production registry/dataset edits have not
yet been authorized for this follow-up and were not made.

The bounded driver also distinguishes repeated valid product wording from a
semantic label error and awaits concurrent outcomes before reporting validation
errors. Recorded generation spend is **$0.00373987** for 15 goods-only requests;
initial cancelled in-flight calls lack complete billing receipts, so that sum
is not represented as guaranteed all-attempt spend. Minor product-prose polish
observations are documented separately from actual supervision defects.

The result supports integration of the explicit load/category/alias approach,
including the newly identified auxiliary source decisions. It does not imply
that unsupported tare/cold-chain/DG profiles were silently broadened or that a
2,500-sample production resampling has already occurred.

## 14. Mixed containers in current and older real sources

This follow-up is an **audit, not a repair or synthesis run**. It inspects only
real sources: the current 600 training and 60 validation documents, the saved
historical real-v6 labels, and the wider filtered maritime PDF/Paddle inventory.
No training input, label, source PDF, template, registry or production code was
changed. Paid-call cost is **$0**. The next improved UAE cohort will **replace
the current newer 2,500**, retaining 5,000 synthetic plus 600 real in total;
the earlier additive 7,500-synthetic proposal has been corrected in the
[coverage plan](kie-synthesis-next2500-uae-coverage-plan-2026-10-11.md).

### 14.1 Current real set: exact label census and confirmed OCR findings

The authority is `mpci-bl-real-v7-reviewed-r16-paddle-positions-660`. Every one
of its 660 IDs, raw OCR texts and targets was compared with the real rows of
the latest `mpci-bl-real600-synthetic5000-uae-v7-positions-v1` dataset:
**all match exactly**. The audit does not accidentally count synthetic rows
or inspect an earlier, different version of the real labels.

| Measure | Real train | Real validation |
| --- | ---: | ---: |
| Documents | 600 | 60 |
| Documents with more than one labeled container | 193 | 24 |
| Different sizes/heights assigned to individual IDs | **9** | **0** |
| Additional genuine aggregate mixed-size declarations, ID mapping unavailable | **2** | **0** |
| Confirmed mixed-size shipments in OCR | **11** | **0** |
| Different canonical equipment families assigned to IDs | **0** | **0** |
| Multi-container documents with some unassigned equipment categories | 13 | 3 |

All eleven confirmed shipments mix **general-purpose** container sizes/heights,
not GP with reefer/open-top/tank. Nine have directly owned per-ID assignments,
covering **28 container objects**. Two have **six IDs in total** and a mixed
aggregate count but no printed ID-to-size correspondence. The presence of
unclassified equipment elsewhere is not proof of homogeneous underlying
hardware: the finding concerns what current OCR/labels establish.

The nine assigned mixtures consist of seven 20ft-standard/40ft-HC documents
and two 20ft-standard/40ft-standard documents. The two aggregate cases add
one 20ft-standard/40ft-HC and one 40ft-standard/40ft-HC shipment. Consequently,
11/600 training documents (1.83%), or 11/193 multi-container training documents
(5.70%), have a confirmed mixture. Current validation has **no positive mixed
equipment example** and therefore cannot measure mixed-equipment recovery.

### 14.2 Complete current positive inventory

Row numbers refer to the current real `train.jsonl`; the full IDs, equipment
objects, literal evidence lines and source hashes are preserved in
[current-adjudications.json](analysis/mixed-container-real-sources-20261011/current-adjudications.json).
`Registry` means that the real source is already in the shared 360-template
catalog; it does not assert that the newly integrated homogeneous registry
override can currently synthesize mixtures.

| Train row / ID prefix | Mixture | Presentation / assignment | Registry | Source OCR | Source PDF |
| --- | --- | --- | --- | --- | --- |
| 29 / `4799921c` | 4×20DC + 1×40HC | ID/tare/type/seal table, continued on rider; per-ID | Yes | [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_4799921c130f9f5971dec476b815a3454a6b59b5d1f08d60dc0e1bab61b002e9/ocr.txt:68) | [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/6bb35e19a49dfafcdcf15dbad63e202cb7e73ee58ad7777430230c24182ea854.pdf) |
| 51 / `98f0e0e0` | 1×20DR + 3×40HC | Separate ID list and count/type box; **aggregate only** | Yes | [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_98f0e0e0461007b238fa574b587e2dcc4614b2f72e3724b360bc3aa345049627/ocr.txt:67) | [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/4d4f0dbb22509eeee33dbc1dd8db173e80b8e117f2a3afd5fc2521bba6fc0af2.pdf) |
| 90 / `9ef31623` | 1×40 standard + 1×40HC | Mixed summary on page 1, untyped ID rows on page 2; **aggregate only** | No | [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_9ef31623373ffb60fff39465bc0d9b46ac7ef97eb4b4973f524c9d6597653079/ocr.txt:37) | [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/16a44ee09eb4e14207a51becc07b0ebd2250ac5c2a298cd4aa1b95c63871633a.pdf) |
| 146 / `b60ce9db` | 4×40HC + 1×20DC | Five explicit ID/type/seal/package/gross rows; per-ID | Yes | [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_b60ce9db5781b204ad6f4eb5555b12c7659a5fe46eae388fc42434932b35decc/ocr.txt:56) | [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/190220f88c6b8d6005e4993b9dd81fef85909e2a41bc0beb3ebe0eb7ec22e469.pdf) |
| 202 / `06744b8f` | 1×20 dry 8′6″ + 1×40 dry 9′6″ | Two adjacent full ID/seal/dimensions/package/gross rows; per-ID | Yes | [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_06744b8fb1ec13f5d717e09621238d401a2e33b00ec7c5a31bf4878873438137/ocr.txt:55) | [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/62b3a6234e299abc7e937ff90681ab1842ec51323878dcebecc519132d715a1b.pdf) |
| 329 / `0bd3ce7b` | 1×20GP + 2×40GP | Mixed front summary plus ID/type/quantity/weight/measure rider; per-ID | Yes | [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_0bd3ce7b5dd84812d58987e15385efa2e95506302e90d847a9b689495b9cd009/ocr.txt:123) | [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/38975d0def77c0a69bd478dc71764bf165ee9508512419fd9cb54a888849f9f3.pdf) |
| 378 / `baabf5a3` | 1×20GP + 1×40GP | Separate container/type/package blocks and detailed crate rows; per-ID | No | [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_baabf5a30d104143dd0b8fd74683cb6d597cdf8610fe613e1c99b77abec34dec/ocr.txt:81) | [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/777cca1e76ef65f958841e64ca8bad1bada0f0c5d7e9b37bf49b246d64c4c593.pdf) |
| 447 / `32ebac37` | 1×20 dry 8′6″ + 1×40 dry 9′6″ | One type-bearing equipment row per page, both with DG facts; per-ID | No | [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_32ebac37f60803fca4d6e46c05278045a5a0e8855cb30bd4ba75ae02fc5dbf18/ocr.txt:75) | [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/54bc1c46b781ebcca2308fe7eaa9e696e9144c906ce77cc14a185976b8a6c82a.pdf) |
| 488 / `2237126f` | 1×20DC + 4×40HC PW | ID/type slash records beside nested package/product blocks; per-ID | No | [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_2237126f936a2dd9c1bc65060c09e7f227286d1e397337f005469322b6f53939/ocr.txt:94) | [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/39589dc7fbdc3d2b06a5ff634826a2e5c461a7d1ebdc5e9589389fe6699c843d.pdf) |
| 551 / `e745f5ba` | 1×20ST + 1×40HC | ID/seal followed by count/type and package line within each block; per-ID | Yes | [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_e745f5ba986215849e936612f1c76767ae3dc4b270bf753e8595cf5048b262bb/ocr.txt:61) | [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/1d4e3d2b25e33edf8c0693a835b85ea2f1a7f71d1d739da243543c7155be2d38.pdf) |
| 562 / `50864bab` | 1×20 dry 8′6″ + 1×40 dry 9′6″ | Two inline equipment rows repeated on a document copy; **two IDs, not four** | No | [OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_50864bab956e163961b99c93daa614a07278e0f1c69f99f59548266168782077/ocr.txt:63) | [PDF](../data/corpora/paddle-bl-swb-filtered-max5/files/b19d910f73c3d8aa5de383a8f124280f98bdd3b94a7f1e66bc00e1e5c6d4ab41.pdf) |

Five of the nine assigned mixed sources and one aggregate source are already
in the template registry. Five further current training sources are not in
that registry. There is no need to use a validation document to obtain these
presentation patterns.

### 14.3 What the row/block relationships look like

**Explicit table or inline row.** The source directly pairs ID and equipment:

```text
MRSU8156948 ML-MA0417315 40 DRY 9'6 48 PACKAGE 16136.950 KGS
MRKU6883890 ML-MA0147841 20 DRY 8'6 10 PACKAGE 11403.460 KGS
```

Here the type/size, quantity and gross weight belong to the same row. A mixed
summary is not necessary: the two rows establish the mixture themselves.
Other sources use `ID / 40HQ / SEAL / QUANTITY / WEIGHT` instead. Different
ordering or delimiters do not alter the row ownership.

**Type in the same spatial block, but on a different OCR line.** Row 551 is:

```text
CMAU0885108
SEAL 15201
1 x 20ST 10 PACKAGE(S)
...
FFAU4445614
SEAL 15029
1 x 40HC 20 PACKAGE(S)
```

The PDF places each ID and its count/type/quantity on the same horizontal
equipment row; OCR expands the columns into lines. Thus a same-line-only
matcher would miss valid assignments. The available positions are useful
for connecting the block, not for inventing values absent from OCR.

**Front-page summary plus detailed rider.** Rows 29 and 329 give an aggregate
mixture on the first page and the remaining ID/type rows on the attachment.
Complete-page processing is essential. Row 447 instead prints one explicitly
typed row on page 1 and the other on page 2.

**Aggregate-only mixture.** Row 51 has four ID/seal/quantity/measurement
records in one column and `1X20 DR / 3X40 HC` in a separate box. The PDF
confirms that the box is a shipment total, not a type for each adjacent ID.
The 19.660 m³ measurement is not permission to identify a 20ft container by
capacity inference. Row 90 similarly has one standard and one HC receipt
but untyped ID rows. Their per-ID sizes must not be assigned by list order,
container-number prefix, weight, or guessed capacity. Common type-only
recovery is a separate annotation question and was not changed in this audit.

### 14.4 False positives and verification performed

The first complete label/OCR scan nominated **31 current documents** for mixed
categories. After source-context inspection:

| Main-screen outcome | Documents |
| --- | ---: |
| Confirmed assigned mixtures | 9 |
| Confirmed aggregate-only mixtures | 2 |
| Equipment lists in demurrage/rate tables or generic form headings, not shipment mixtures | 17 |
| Single-container OCR alias conflicts, not mixtures | 3 |

The 16 partial multi-container documents were also inspected; two are the
aggregate positives above. The other fourteen do not establish a mixture
in OCR. Validation row 42, for example, owns one `1 x 20ST 8 PALLETS` occurrence
but does not provide equipment categories for its six other IDs.

Two additional independent screens were used to challenge the main grammar:

- A wider bare-foot/count×length scan adds **seven** current candidates.
  Six are rate/form headings; one is `1,000x20KG BAGS`, a package fill weight,
  not a 20ft equipment declaration.
- A broad compact-code scan of the remaining multi-container documents adds
  **nine** cases with unresolved surface tokens. Six are rate tables or
  uniform full-dimension equipment with tariff `CTR` tokens; two are product/
  weight-unit fragments; one is row 317's `40EQ` versus `40HQ` OCR discrepancy.
  None adds a genuine mixture. The historical-label overlap independently
  nominated row 317 as well.

PDF inspection of row 317 shows that its first container actually prints
**40HQ**, matching the other six; GLM emits `40EQ`. Historical labels had
incorrectly treated the first as 40ft-standard. The current labels already
assign all seven to 40ft-HC; no new repair was applied. This illustrates why
historical label diversity is a nomination, not a count of true mixtures.

The other current OCR alias controls are rows 112/355 (`40HO`, PDF **40HQ**)
and row 302 (`40RO`, PDF **40RQ**, with an explicit `40HR` receipt). Neither
multiple aliases nor one noisy character establishes different equipment.
Likewise `45G1` is an ISO size/type code for 40ft-HC, not 45ft length.

All nine assigned positives and both aggregate positives had their source
PDF layouts viewed. Relevant current source rendering covers 18 pages across
13 documents, including the reefer OCR control and historical-overlap control;
the two earlier HQ/HO control crops were already preserved by the preceding
audit. Literal source proof replay checks **28/28 assigned container objects**.
Deliberately wrong-size and wrong-ID substitutions are rejected for every
one: **56 negative checks**, plus four grammar controls for aggregate count
prefixes, complete 9′6″ dimensions, ISO codes and bag-capacity distractors.

These are source-specific ownership checks, not a newly deployed universal
classifier or a re-certification of unrelated labels. The current census,
adjudications, OCR/PDF paths, hashes and reproduction scripts are in
[mixed-container-real-sources-20261011](analysis/mixed-container-real-sources-20261011/).
The full current read-only inventory runs in approximately 2.2 seconds;
source hashes are checked before and after. There is no production throughput
change to benchmark because no production component was edited.

### 14.5 Older real sources and additional presentation diversity

The filtered at-most-five-page B/L/sea-waybill pool has 1,918 PDF objects:
660 current plus **1,258 older/wider sources**. All saved Paddle outputs for
the latter were screened. Separately, 1,156 historical real-v6 records contain
496 outside the current set. These inventories overlap and must not be added.

The wider screen produces 52 multiple-code-surface candidates. Source-context
inspection confirms **20 additional GP mixed-size receipts** among them.
These are positive equipment findings, not twenty newly gold-labeled complete
documents or a claim of exhaustive negative coverage of all older sources.
Historical labels nominate twelve older candidates: eleven GP size mixtures
and one MAFI/closed-container combination. Six representative older sources
and three older OCR false-mixture controls had their PDFs visually inspected.

Strong additional examples are:

- **`6fe428c6` — two 45HC plus one 40HC**, each ID with its own type/count,
  gross, tare and measurement row and one common product block. This adds a
  45/40 contrast absent from current real mixed examples.
- **`bb332523` — one 20DC plus one 40HC**, explicit ID/tare/type/seal table.
  Its aggregate summary lists the sizes in the opposite order to the ID rows:
  direct proof that summary order cannot determine ID ownership.
- **`918488d0` — one 40HQ plus one 20GP**, slash-delimited complete equipment
  rows with quantity, gross and CBM; the aggregate receipt corroborates them.
- **`cc8eb26d` — one 40HC plus one 45HC PW**, two product/equipment blocks.
  PDF inspection reveals an omitted second gross weight in the archived GLM
  text and possible multi-goods scope concerns. Useful presentation evidence,
  but not an immediately approved donor for the current single-goods task.
- **`8555bbdd` — MAFI 40CC plus CNTR 40HC**, explicitly different transport
  unit kinds on separate rows. The historical platform enum is not newly
  certified here; this special-cargo case should not be treated as a routine
  GP/reefer/open-top mixed-container source.

One apparently promising reefer source, `73d810e8`, prints one `40RH` and
three `40RA`. Carrier-aware resolution for CMA CGM maps both to **40ft-HC
reefer**. It is a useful different-alias control, not a confirmed different-
height or different-family shipment. No ordinary closed-GP plus reefer/
open-top/tank combination was confirmed in the inspected older examples.

The [older-source review](analysis/mixed-container-real-sources-20261011/OLDER_SOURCE_REVIEW.md)
contains full OCR/PDF links and exact excerpts; the
[older inventory](analysis/mixed-container-real-sources-20261011/older-source-screen.json)
contains all nominated sources, the twenty positive receipts, page/line evidence
and input hashes. No older source was automatically admitted to the current
dataset or synthesis registry.

### 14.6 Consequences for the upcoming synthesis decision

1. **Mixed GP sizes have real precedent and sufficient reusable layouts.**
   They are not a hypothetical format requiring fabricated document structure.
   Begin with the five already-registered, directly owned current source cases.
   If more diversity is required, assess the other four current assigned cases
   and the clean older 45/40 and slash-row examples before compiling them.
2. **Sampling can be per-container only where the template owns that value.**
   All per-ID equipment aliases, package allocations, row measurements, tare
   and aggregate count/type receipts must come from the same container plan.
   No requirement follows that every descendant retain exactly its source mix,
   but independent new draws must be representable by those owned fields.
3. **An aggregate-only source remains an aggregate-only source.** Preserve its
   private mixed multiset and avoid public per-ID size inventions. Do not claim
   that changing the sampler resolves absent ownership evidence.
4. **Do not broaden arbitrary specialized family mixtures from this audit.**
   Current evidence supports mixed GP lengths/heights, not freely swapping
   thermal/DG/platform facts into a layout without the corresponding contract.
5. **Validation has no positive mixtures.** The next mixed synthesis feature
   needs a dedicated source-replay/holdout test for row/block ownership and
   mixed summaries; existing aggregate validation metrics cannot demonstrate
   its correctness by themselves. This audit does not change the validation
   split or authorize that implementation or generation run.
