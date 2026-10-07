# Source-conditioned stochastic coordinate synthesis: investigation and probes

**Status update:** the approved method is now integrated into production and has
regenerated all 200 published samples. The original investigation below is
preserved; see [production integration](#production-integration-and-200-sample-regeneration)
for the current output, commands, tests and 50-page visual gallery.

## Result

There is a useful, tested improvement over frozen-region interpolation:
**source-conditioned, typography-aware elastic page reflow**. It samples spacing,
typesets replacement regions virtually, uses existing whitespace, and moves
dependent source blocks when necessary. It preserves measured row/column
relationships instead of preserving every original absolute coordinate.

On the existing **100 templates / 200 synthetic documents**, the conservative
seed-0 experiment produces:

| Line group | Current published coordinates | Experimental coordinates |
| --- | ---: | ---: |
| All content lines | 15,594 / 20,269 = **76.94%** | 16,716 / 20,269 = **82.47%** |
| Goods descriptions | 98 / 1,288 = **7.61%** | 902 / 1,288 = **70.03%** |
| Party addresses | 827 / 1,925 = **42.96%** | 1,123 / 1,925 = **58.34%** |
| Other lines | 14,669 / 17,056 = **86.01%** | 14,691 / 17,056 = **86.13%** |

This adds **1,122 coordinates**: 804 goods lines, 296 address lines and 22 other
lines. Previously positioned lines remain positioned; their coordinates may
move as part of the new layout. Goods have some positional coverage in **165/200
documents**, versus 40/200 before; all goods lines are covered in **161/200**,
versus 35/200. Addresses have some coverage in **192/200**, versus 173/200.

**The prototype is separate from production.** No published dataset, source OCR,
label, template, campaign config or production geometry implementation was changed
by this investigation. The previous template relocation remains intact. There
were **no paid API calls, no model training, no new dependency installations and
no subagents**. Cost for these experiments: **USD 0**.

## Evidence and reproducibility

All probe artifacts are under
[`docs/analysis/synthesis-position-reflow-20261007/`](analysis/synthesis-position-reflow-20261007/):

- [`probe.py`](analysis/synthesis-position-reflow-20261007/probe.py): provenance
  verification, source grouping, source-derived priors, layout generation,
  proposal gates, separate JSONL and layout receipts.
- [`validate_probe.py`](analysis/synthesis-position-reflow-20261007/validate_probe.py):
  seeded stress tests, falsification, font sensitivity, masked-anchor experiments,
  comparative timing and geometric galleries.
- [`support.json`](analysis/synthesis-position-reflow-20261007/support.json):
  exact source partitions, measured prior statistics and font hash.
- [`validation.json`](analysis/synthesis-position-reflow-20261007/validation.json):
  complete tests, seeds, timing and failure inventories.
- [`elastic-s0-floor6.81487/summary.json`](analysis/synthesis-position-reflow-20261007/elastic-s0-floor6.81487/summary.json):
  the reported 200-document result.
- [`elastic-s0-floor6.81487/dataset.jsonl`](analysis/synthesis-position-reflow-20261007/elastic-s0-floor6.81487/dataset.jsonl):
  experimental inputs with unchanged labels and plain text.
- [`elastic-s0-floor6.81487/layout.json`](analysis/synthesis-position-reflow-20261007/elastic-s0-floor6.81487/layout.json):
  per-page proposals, source blocks, accepted geometry and explicit rejection reasons.
- `anchor-s0-floor0/`, `typography-s0-floor0/`,
  `typography-s0-floor6.81487/`: comparison policies, not approved alternatives.
- `initial-probe/`: earlier probe summaries, retained rather than replacing the
  experimental history with only its successful endpoint.

Reproduce from the repository root:

```bash
.venv/bin/python docs/analysis/synthesis-position-reflow-20261007/probe.py \
  --mode elastic --seed 0 --density-floor 6.814868512901933

.venv/bin/python docs/analysis/synthesis-position-reflow-20261007/validate_probe.py \
  --seeds 10
```

The numerical density floor is not an invented font constant: it is the 1st
percentile of page-median normalized glyph heights in the 80-source fitting
partition. The exact fitting sources and value are saved in `support.json`.
This explicit argument reproduces this probe; integration should expose a
named source-fitted density policy rather than hard-code this corpus value.

Final preservation check: all 200 experimental records equal the published plain
records after removing the added `positionedText` field. Published plain and
positioned files retain their pre-investigation SHA-256 hashes:

```text
published plain:      5268bc8367e06a04e94c2fcae6320f8801a9bb6f0456b10884f09caf0b276dae
published positions:  c35131d73070903cc38bc7c2f9865e772a0eb622ccb4fcf56780353b568d5fae
experimental seed 0:  20c31d6e3239b0fcf03373175240ad7480916f8e7a7504942918883f8683034f
```

## Diagnosis: why the current implementation abstains so often

The published implementation in `curated_positions.py` and
`curated_position_regions.py` operates one edit at a time:

1. Resolve exact source byte ownership and its measured line anchors.
2. Keep the original anchors if line count is unchanged.
3. Otherwise fit the replacement inside the source region plus half the adjacent
   vertical gaps, at source-derived line spacing.
4. Do not move neighbouring content. If the expansion cannot fit, emit ` ||`.
5. Apply coherent page scale/translation **after** this placement decision.

The existing page transform already is seeded/stochastic. It cannot restore
coordinates rejected before the transformation, and uniformly scaling text and
available space does not change whether the text fits. The main limitation is
**frozen independent regions**, not that the arithmetic is deterministic.

The old method also carries original horizontal centres into new wording. A
short word's centre is not a reliable horizontal centre for a long line. It can
still be a coarse region anchor, but it is not a text-length-aware centroid.

The source has fewer lines, not more: two copies of the 100 sources would have
19,062 content lines, whereas the generated set has 20,269. Most growth is goods
and address wording. The failure breakdown and historical controlled replay are
in [the expansion follow-up](kie-synthesis-expansion100-2026-10-07.md#follow-up-rendered-content-audit-coordinate-coverage-and-reusable-catalog).

## Alternatives considered and tested

### More independent random jitter — not a solution

Jitter does not create room or establish the owner of a missing source anchor.
It risks changing table-row alignment and field proximity. Existing whole-page
scale/translation remains useful **after** layout construction, but does not
replace layout construction.

### Copy/interpolate centres and stretch the page — useful control, incomplete

The anchor-only control positions 920 goods lines, but ignores the physical
width of new strings. In particular, a long description can inherit the narrow
footprint of a one-word source. Its coverage therefore is not sufficient
evidence of a realistic layout. It is not the recommended method.

### Typography-aware reflow preserving every original gap — too rigid

Estimating line width and wrapping removes the narrow-footprint problem. But
preserving all original blank gaps while inserting longer blocks unnecessarily
extends and compresses pages. Without a density gate, this control shrinks some
pages to **65.74%** of their original normalized vertical scale. With the
source-derived density gate, it admits 175 pages, rejects nine page proposals,
and places 844 goods lines / 1,106 address lines.

### Typography-aware elastic reflow — recommended

Keep source column positions, shared-row ties, above/below relationships and
clearance, while treating blank space as consumable. Independent columns need
not retain a global ordering of unrelated rows. The document does not need to
grow merely to preserve unused bottom margin.

This admits **183 page proposals**, leaves **186 pages with no eligible reflow**
on their existing anchors, and rejects **one page proposal** at seed 0 for
excessive density. The latter retains the baseline coordinate representation,
with an explicit rejected-proposal record—not a fabricated successful layout.
The 183 accepted pages include **173 with no vertical compression at all**.

## Recommended mechanism, step by step

1. **Establish layout ownership from the renderer's exact edit provenance.**
   Connect edits sharing a source line or measured region. This includes inline
   captions and numeric peers; it does not infer ownership from field-name
   keywords or rendered-word matching. Page boundaries remain hard boundaries.
2. **Start with all measured source regions.** Unmatched Paddle regions are still
   obstacles. A candidate replacement requires complete source anchors and a
   geometrically coherent region; an envelope containing unrelated source text
   is not silently claimed by the replacement.
3. **Estimate a coarse typesetting scale.** Measured source text widths and a
   real font's advance measurements estimate the local scale. Adjacent source
   regions bound usable width. New wording is virtually word-wrapped within that
   width, including character wrapping for a long indivisible code/URL. This
   does not rewrite the OCR string: one logical input line can represent a
   multi-line physical footprint, just as real GLM/Paddle alignments can.
4. **Sample block spacing from source observations.** The fitting partition
   supplies 303 consecutive-line spacing observations within owned regions.
   The interquartile pitch/glyph-height ratio is **1.063–1.286**. One seeded
   spacing choice applies coherently to a block, not independent x/y noise per
   line. Sampling all page-neighbour gaps instead was rejected: its upper
   quartile mixed paragraph gaps into line spacing and over-expanded text.
5. **Solve page clearance constraints jointly.** Same-source-row peers move
   together. Blocks in overlapping columns preserve their order and required
   separation. Original positions are preferred: downstream movement happens
   only when new content cannot fit. Existing gaps can be consumed down to a
   measured interline clearance. Other columns move when row ties or overlap
   constraints require it, not merely because their OCR lines occur later.
6. **Normalize to the page and check density.** Existing spare bottom margin can
   absorb growth, retaining a one-source-glyph guard or the original smaller
   margin. If growth still exceeds the page, normalize vertically, subject to
   the source-fitted density gate. Font/glyph metrics here are geometric proxies,
   not a claim that a new PDF has been rendered and measured.
7. **Compute logical line centres and validate them separately.** A box-level
   pass alone is insufficient: composed OCR lines can aggregate several boxes.
   Check line-centre placement, rounding, duplicate centres, unchanged-column
   order and interference with another generated block.
8. **Apply the existing coherent page augmentation afterward.** It now transforms
   the completed synthetic layout, not the old source layout with holes. The
   existing affine validator continues to apply to this new layout.

This is stochastic, source-conditioned generation with deterministic constraints.
Reproducibility is intentional; it is not a requirement to copy the same points
into every synthetic variant. An LLM is not needed to invent individual x/y values.

## Validation and attempts to falsify the method

### Entire current pilot, not selected successes

- All **200 samples, 100 templates and 370 pages** were processed.
- **10 seeds**, or **2,000 document-level layout trials / 3,700 page trials**.
- Overall known-line counts range **16,716–16,734**; goods coverage ranges
  **902–920 / 1,288 (70.03–71.43%)**.
- Every accepted proposal passes bounds, ordering, row alignment, column
  relationships, source-obstacle clearance and emitted-centroid checks.
- Zero previously known coordinates become unknown. Exact plain-text recovery
  and target equality are checked separately; no labels are regenerated.
- Across the ten seeds, **4,598 line identities** receive multiple distinct
  coordinate variants. About **23.3%** of comparable coordinate pairs differ
  from seed 0 before adding the global page transform.
- The first three seeds also passed the existing affine augmentation checks on
  **551 accepted page layouts**: 547 scale/translation modes and four explicitly
  recorded integer-translation-only modes.

The source support uses a fixed **80-family fitting / 20-family diagnostic split**.
The 20 families supply no spacing/density fitting data. Their goods coverage is
**205/243**, versus **8/243** previously; addresses **231/379**, versus **180/379**.
Both partitions were inspected during development. This is source-disjoint
diagnostic evidence, not an untouched model-performance benchmark.

### Faults found in the probes, rather than hidden by aggregate coverage

1. **Floating-point boundary touches** looked like overlaps at approximately
   `2e-13` normalized units. An explicit `1e-9` arithmetic tolerance handles
   numerical equality; it is not permission for perceptible overlaps.
2. **Moving a rounded y coordinate instead of its source region** could leave a
   following caption inside an expanded block. Movement now follows exact source
   region identity and its measured geometry. All three observed affected page
   proposals then passed this check, before density filtering.
3. **Logical-centre collision despite nonoverlapping physical boxes:** a stress
   seed placed two distinct composite OCR lines at `(544, 665)`. This exposed
   the need for the separate line-centre gate. Final ten-seed runs reject two
   page proposals for this collision/order issue and two for density; all four
   retain their baseline anchors explicitly. No invalid proposal is published
   as an accepted reflow.

### Adversarial and randomized tests

- **11/11 deliberate corruptions rejected:** off-page/nonfinite geometry,
  inverted rectangles, moved columns, footer above goods, row-peer drift,
  cross-column expansion and invalid scales.
- **1,000 randomized two-column layouts** with varying row heights, counts and
  expansions pass independent geometry/order checks. This tests solver arithmetic
  and topology, not the semantic ownership of an unknown future template.
- Corrupting target values, sampled scenario and target-path names in **ten
  documents** leaves coordinates identical. Placement does not look at gold
  values or categories to decide where a word should be.
- Three font surrogates were tested on all 200 documents. DejaVu Sans positions
  902 goods lines, Liberation Sans 914, and DejaVu Serif 920. Gates remain active:
  one Liberation Sans proposal is rejected for a logical-centre collision, and
  the seed-0 DejaVu Sans density rejection remains explicit. There is no large
  dependence on one chosen font; none of these establishes exact source font identity.

The existing position/spatial-input regression suites also passed: **63 tests
in 10.66 seconds** (`tests/test_curated_positions.py` and
`tests/test_spatial_inputs.py`). Ruff checks for both probe scripts and
`git diff --check` passed. These regression tests supplement the real-pilot
experiments; they do not replace them.

### Missing-anchor recovery: a separate, deliberately difficult question

I hid **153 known interior anchors from 67 sources**, predicted from visible
neighbours, then compared against the withheld measured coordinates.

Naive centre interpolation gets only **16/153** within half a line height.
Its median horizontal error is **20/1000 page-width units**, 95th percentile
**108.1**, maximum **160.5**. Vertical error is much smaller: median zero,
95th percentile one normalized unit. This is largely the effect of different
line lengths, not evidence that all local vertical ordering is unreliable.

A second predictor estimates the left margin and font scale from the two visible
neighbours and uses the hidden line's available OCR text. With an aligned-margin/
local-spacing gate, it covers 149 cases: **122/149** within half a line height,
**142/149** within one, maximum error **1.82 line heights**. This is promising
for coarse *inferred* block-local anchors. It is not permission to call inferred
coordinates measured, nor to fill unbracketed continuations or ambiguous columns.
It is not applied to the experimental dataset in this pass. The reflow results
above do not rely on accepting these missing-anchor guesses.

## Remaining coordinate gaps and their scope

Seed 0 leaves **3,553 lines** unpositioned:

| Original cause | Still unknown |
| --- | ---: |
| Missing measured source anchors | 3,230 |
| Unowned text intersects source region | 298 |
| Local-space failure on the page rejected by the new density gate | 18 |
| Competing/composed edit owners | 7 |

The proposed transfer avoids inferring the location of regions with incomplete
source anchors: **170 replacement-group occurrences across 74 samples / 42
families**. It also declines envelopes containing unrelated source text:
**79 group occurrences across 46 samples / 23 families**. These categories
overlap at document/family level and are not counts of faulty labels.

This separates the next work clearly:

- **Ordinary measured expansions:** the joint-reflow mechanism is demonstrated
  across this whole pilot, not just a handpicked few examples.
- **Missing source matches:** improve source alignment or introduce separately
  calibrated, explicitly inferred block-local anchors. Additional randomization
  cannot supply missing ownership evidence.
- **Interleaved/mixed regions:** subdivide the source layout into a reviewed
  compound block that retains the unrelated text and row dependencies. Do not
  merely ignore the obstacle or claim its space.
- **Occasional rejected stochastic proposal:** retain the complete baseline with
  an explicit rejection, or add a bounded, logged resampling policy during
  production integration. No unbounded retry loop is needed.

## Visual inspection

The gallery shows measured source boxes, current source-frame line anchors and
proposed layouts side by side. Red points are goods; green addresses; blue other
lines. Gray rectangles are measured or virtually reflowed envelopes, not synthetic
PDF glyph measurements. Page plots are normalized diagrams, not source scans.

- [One-word TYRE source to 14 generated description lines](analysis/synthesis-position-reflow-20261007/gallery/049ad12d-p1.png):
  all 14 now have distinct positions; following clauses move below the enlarged
  cargo block; no page compression is needed in the elastic variant.
- [Goods plus address expansion](analysis/synthesis-position-reflow-20261007/gallery/86ed2b9e-p1.png):
  separate word-length-dependent centres, preserved columns and downstream text.
- [Density-limited page](analysis/synthesis-position-reflow-20261007/gallery/19580ef7-p1.png):
  explicitly retains the baseline at seed 0. Its proposed median normalized
  glyph height was 6.717, below the fitted 6.815 floor.
- [Missing source-anchor family](analysis/synthesis-position-reflow-20261007/gallery/b5c11ec9-p2.png).
- [Multi-page source, first page](analysis/synthesis-position-reflow-20261007/gallery/3dc8551d-p1.png).
- [Continuation page](analysis/synthesis-position-reflow-20261007/gallery/9db60987-p2.png).

## Runtime, limitations and integration recommendation

In the paired 200-document CPU benchmark, existing transfer took **0.348 seconds**
and the experimental reflow stage **1.912 seconds**; median per-document times
were **1.63 ms and 10.04 ms**. The prototype currently consumes existing edit-line
provenance, so these are stage timings, not a claim that it completely replaces
the old work. It adds roughly two seconds per 200 documents, or a simple serial
projection of about **96 seconds per 10,000** for this stage. That projection
excludes geometry loading, wording generation and publication.

The complete stress/falsification/font-sensitivity run took **40.02 seconds**,
peak RSS **346.3 MiB**. Geometry/source loading in the generation probe peaked
around **380 MiB**. No network inference was needed. This is a feature-quality
expansion with measured CPU cost, not a speedup claim or a production performance
optimization pass.

**Recommendation:** integrate this as an explicit source-conditioned elastic
position policy, keep the current whole-page augmentation after it, and retain
per-page provenance/abstention receipts. Promote the tested controls—row ties,
column/order/clearance, emitted integer-centre checks and source-fitted density—
together. Expose the font surrogate, prior-fitting partition, density quantile,
row-lock tolerance and seed as policy rather than burying corpus values in code.
Use per-field coverage, not only overall coverage, as a release metric.

The tests establish reproducibility, text/label preservation and useful coarse
layout structure for the tested cohort and admissible regions. They do **not**
establish exact typeset PDF geometry, universally correct source alignment or an
increase in downstream extraction F1. The latter needs a controlled training
comparison; no training was started here. There is no need to regenerate goods
or parties merely to apply this coordinate policy.

## Relevant primary research

The design direction is consistent with constrained layout generation: generate
variable layouts while rejecting forbidden relationships rather than independently
jittering every coordinate. [LayoutFormer++](https://arxiv.org/abs/2208.08037)
uses explicit constraints and restricted generation; it does not validate our
particular B/L layouts. Google's [Variational Transformer Networks overview](https://research.google/blog/using-variational-transformer-networks-to-automate-document-layout-design/)
also evaluates layout overlap/alignment and diversity separately. Those works
motivate checking both properties here; this experiment does **not** train or use
their neural layout models. The recommendation above is based primarily on the
local full-pilot measurements and falsification results.

## Production integration and 200-sample regeneration

The approved elastic policy is implemented in the main synthesis pipeline, not
called through the experimental scripts. The `positions` command now performs
verified source-edit transfer → joint reflow → whole-page augmentation → immutable
publication when `positions.reflow` is configured. The active expansion100 config
enables it; historical pilot recipes are left unchanged.

### Published output and inspection links

Everything for the regenerated 200 samples is under
[`positions-reflow-v1/`](../artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-reflow-v1/):

- [Positioned dataset JSONL](../artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-reflow-v1/dataset.jsonl).
- [All 200 rendered inputs](../artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-reflow-v1/samples.md).
- [Visual gallery: 20 documents, all 50 of their pages](../artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-reflow-v1/audit/GALLERY.md).
- [Saved-output validation](../artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-reflow-v1/audit/validation.json).
- [Per-document coverage inventory](../artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-reflow-v1/audit/documents.json).
- [Manifest and receipt hashes](../artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-reflow-v1/manifest.json).

The gallery spans 14 source families and includes both variants where useful.
Selection includes the rejected page, large goods/address gains, high residual
missingness, and a hash-selected spread—not only successful examples. Every plot
shows measured source boxes, previous synthetic coordinates, reflow before page
augmentation, and the final published coordinates. Numbered points map to the
rendered input included below the plots. Rectangles are approximate envelopes,
not measured synthetic PDF glyphs. Representative inspected examples:

- [Sixteen newly positioned goods lines, constrained page reflow](../artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-reflow-v1/audit/gallery/syn_full_v7_3ac4bee8cfa3b7c030a12739-p1.png).
- [Goods and address expansion while preserving separate columns](../artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-reflow-v1/audit/gallery/syn_full_v7_8a87efaa393e8684fb218947-p1.png).
- [Rejected dense-page variant: full baseline preserved](../artifacts/kie-synthesis-production/curated-v7-expansion100-v1/positions-reflow-v1/audit/gallery/syn_full_v7_54cbeda5f70bdf232bb24132-p1.png).

### What changed in code

| Component | Responsibility |
| --- | --- |
| `curated_reflow.py` | Connected source ownership, text-width-aware placement, seeded spacing, independent emitted-centre checks, whole-page acceptance/rejection and replayable receipts. No target values enter this API. |
| `curated_reflow_geometry.py` | Shared-row/column clearance solver and independently evaluated rectangle/order constraints. Only the approved elastic policy was promoted; experimental control modes were not copied. |
| `curated_reflow_policy.py` | Strict calibration/settings models; hash-checked font and calibration loading; per-engine font advance cache bounded to 4,096 entries. No silent font substitution or calibration defaults. |
| `curated_layout.py` | Optional explicit `positions.reflow` configuration; existing affine policy remains applicable after reflow. |
| `curated_positions.py` | Integrates reflow before augmentation, passes the reflowed envelopes to affine validation, records held groups/rejections/coverage and serializes reproducible publication receipts. Source iteration and saved JSON are deterministic. |
| `position-calibration.json` in reusable template catalog | Carries the existing measured priors, 80 fitting-source alignment hashes, diagnostic partition and source dataset manifest hash. It is versioned and pinned by the campaign. |
| `mpci_bl_curated_v7_expansion100.yaml` | Enables the tested reflow seed 0 with the same measured density/spacing priors, explicit geometric guards and pinned DejaVu Sans file. Writes `positions-reflow-v1/`, not over `positions-v2/`. |
| `scripts/synthesis/audit_curated_positions.py` | Reusable saved-file audit and deterministic visual-subset generator. Category tags stratify the report only; they never affect layout. |

The configured font is `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`, SHA-256
`690243adfefe0ce154b547db6205794bd30ac4277275179517a90994f4980648`.
On another machine, provide that font at the configured path or explicitly point
to the same file. Missing/changed font or calibration is an error, not a fallback.
No new Python dependency, paid API call or GPU job was needed.

### Final measurements

Production matches the seed-0 experiment exactly before the global affine step:

| Group | Previous known / total | New known / total | Change |
| --- | ---: | ---: | ---: |
| Goods descriptions | 98 / 1,288 (7.61%) | 902 / 1,288 (70.03%) | +804 |
| Party addresses | 827 / 1,925 (42.96%) | 1,123 / 1,925 (58.34%) | +296 |
| Other lines | 14,669 / 17,056 | 14,691 / 17,056 | +22 |
| All lines | 15,594 / 20,269 (76.94%) | 16,716 / 20,269 (82.47%) | +1,122 |

All 200 records retain their exact plain text, labels and non-positional metadata.
No previously positioned line becomes unpositioned. The final 370 pages comprise
183 accepted reflows, 186 with no eligible reflow, and one explicitly rejected
density proposal. Affine augmentation then produces 367 scaled/translated pages,
one integer-translation-only page and two pages without known anchors.

The published plain dataset and old `positions-v2/dataset.jsonl` retain their
original hashes listed earlier. The new final positioned dataset hash is:

```text
f164265be119ed7b8712e25de28c19368d0e8d86f3092bc1599d7d1183502660
```

The 3,553 remaining unpositioned lines are not newly discovered label failures.
Their source-anchor/ownership/density limitations are unchanged from the probe
and remain visible in the receipts. This integration does not invent anchors to
force 100% coverage or regenerate any product/party text.

### Production validation, including faults found during integration

- **326 targeted synthesis/spatial tests passed in 17.37 seconds**, including
  campaign, template, publication, physical, scenario, contact and position suites.
  New reflow tests cover expansion, contraction, deletion, blank lines, seed
  repeatability/diversity, missing anchors, excessive density, font/calibration
  corruption, wrapping, geometric mutations and production publication replay.
- The real production engine was run through **10 seeds × 200 documents**,
  including the affine stage on **all 3,700 page trials**. Coverage and explicit
  rejection outcomes match the earlier experiments. No invalid proposal was
  accepted. The same 1,000 randomized layout checks and 11 deliberate corruptions
  passed against the production geometry functions.
- All **200 pre-affine positioned texts** exactly match the successful probe;
  all **200 final receipts** exactly match the saved production publication.
  [Integration stress evidence](analysis/synthesis-position-reflow-integration-20261007/validation.json).
- A second independent CLI process with `PYTHONHASHSEED=317` reproduced the
  complete dataset, all receipt hashes and the manifest without overwriting
  anything different.
- The independent saved-file audit checks record identity, unchanged labels/text,
  hashes, line/page coverage, generated-centre positions, rejection atomicity,
  reflow geometry and final affine correctness. All 200 passed. It generated the
  50 page plots in 4.47 seconds including validation/loading.
- Ruff and `git diff --check` passed.

The initial saved-file audit caught a serialization bug introduced during
integration: integer keys in `lineBoxes` sorted differently after JSON converted
them into strings, making a pre-serialization receipt digest differ on reload.
The engine now normalizes these keys before hashing. A multi-digit line-map JSON
round-trip regression test covers this. The initial attempt was moved to a
temporary directory; it is not the published dataset. No text/labels or coordinate
placement required repair. This illustrates why checking saved outputs—not only
in-memory layout checks—is part of the acceptance process.

### Runtime and memory

The complete 200-sample CLI, including campaign initialization, source validation,
candidate replay and writes, took **33.83 seconds before** and **36.43 seconds
after** (repeat: **36.28 seconds**). The timed positioning/publication section was
**20.01 seconds before**, **21.79 seconds after**, and **21.67 seconds on replay**.
Thus the new feature adds about **1.8 seconds per 200 samples** in that section;
it is not reported as a speedup. The extra CPU work performs font-aware layout and
the additional geometric checks. It incurs no inference cost and is negligible
beside the wording-generation stage. Font loading is once per campaign and advance
measurements use a bounded cache.

Peak process memory was about **477 MiB before**, **473 MiB after**, and **483 MiB
on replay**; these are observed process peaks, not a claim of memory savings.
The separate 10-seed production stress/falsification run took **30.22 seconds**
and peaked at **350.4 MiB**. No LLM generation or training was launched.

### Commands

Regenerate or exactly replay the new positioned variant:

```bash
.venv/bin/python -m document_ocr.synthesis.curated_campaign positions \
  --config configs/synthesis/mpci_bl_curated_v7_expansion100.yaml \
  --project-root .
```

Audit saved outputs and rebuild the visual gallery:

```bash
.venv/bin/python scripts/synthesis/audit_curated_positions.py \
  --config configs/synthesis/mpci_bl_curated_v7_expansion100.yaml \
  --baseline-subdirectory positions-v2 \
  --line-categories artifacts/kie-synthesis-production/curated-v7-expansion100-v1/audit/coverage-lines.csv \
  --plot-documents 20
```

`--line-categories` is optional for geometry validation/plots; omit it for a
campaign without a field-attribution inventory. Unknown/held positions remain
explicit in the rendered inputs. These measurements validate layout plausibility
and pipeline correctness on this cohort, not downstream extraction accuracy.
