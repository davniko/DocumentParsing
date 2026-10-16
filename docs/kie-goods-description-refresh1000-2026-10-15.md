# Goods description refresh:1,000 current synthetic samples

Date:2026-10-15. **Complete: regenerated, reviewed, adjudicated, published
in-place into R26, and independently read back. No training was launched.**

This is a separate work unit from the
[failed-model diagnostic](kie-failing-run-deep-audit-2026-10-15.md) and
[country audit/base-model experiment](kie-country-presentations-audit-2026-10-15.md).

## Selection and contract

The base is the latest corrected
[R26 dataset](../data/curated/mpci-bl-real600-synthetic7000-dg-llm-wording-r26-positions-v1/manifest.json),
containing the DG wording, phone-presentation and container-ID repairs.
Draw seed:`2026101501`. There are6,458 eligible model-owned ordinary/thermal
training examples. The452 current DG examples have already been regenerated in
the preceding repair and are deliberately preserved;90 non-DG whole-unit
host-owned examples are outside this wording-only refresh.

The1,000 examples span324 source templates. One hundred are drawn from737
eligible detail-bearing examples, then900 from the remaining eligible pool;
the combined draw is shuffled into ten blocks of100. Selection recognizes
source specifications such as purity, grades, dimensions and compact per-pack
wording **only for this audit draw**. No runtime classifier or hard word/line
quota was added.

The live second polished description prompt is used. DG already uses this
same shared prompt and field-description policy. Commercial wording and
plausible new intrinsic details are model-generated; existing HS presentations,
shipment quantities, package categories, allocations, DG/thermal declarations,
parties, dates, identifiers, contacts and routes remain unchanged.

The [frozen design](analysis/goods-description-refresh1000-20261015/design.json)
records every selected ID, full native wording request, original authority
hash and source-context example. The
[execution script](analysis/goods-description-refresh1000-20261015/refresh.py)
persists every paid attempt, output, validation repair, scoped semantic review
and primary adjudication. Initial pilot calls contain five shipments; later
blocks use ten to amortize shared request overhead. Failed groups are explicitly
resumed; an invalid shipment is repaired separately rather than rewriting its
accepted neighbours.

## Mutation and review boundaries

- Product occurrences and explicit commercial repeats are traced from the
  native source/edit proof to the actual current text, never by rerendering a
  stale historical candidate over current privacy/format fixes.
- The preflight found22 mapping exceptions. Sixteen preserve the complete
  prior-owned byte string uniquely, but older HS repairs joined it to the next
  caption and defeated whole-line alignment. Six documents were manually
  reviewed initially, followed by four narrower product-owner exceptions found
  by the per-product preflight. Eleven exact existing-target owner spans across
  those ten documents are pinned in the
  [adjudication artifact](analysis/goods-description-refresh1000-20261015/adjudicated-owner-overrides.json).
  Leading packing declarations, trailing accounting totals and intervening HS
  rows remain excluded. The complete1,000-row preflight subsequently passed.
- A complete block projection independently derives each final description
  label, including unchanged source-certified host framing. Every other target
  value and every raw byte outside approved edits is required to remain equal.
- Coordinates are transferred through exact current-input edit spans, without
  rematching or regenerating unrelated template blocks. Unedited line anchors
  remain exact; contractions stay inside the existing owned span. Expansion
  lacking proven current-frame clearance remains explicitly unpositioned.
- Each100-row block gets deterministic projection/geometry checks and a
  separate scoped model review. Every finding is adjudicated by the primary
  assistant. Deliberately generated brands, years, intrinsic capacities and
  harmless punctuation are not treated as defects just because absent from the
  source. Inferred per-pack arithmetic is not an extraction-label requirement.

The publication gate requires all ten blocks, complete review coverage, closed
findings, exact old/new byte replay, non-description target equality, consistent
sidecar mirrors, untouched real/validation data, and the actual CPU training
loader accepting7,600 training/60 validation records. The exact original R26
files are preserved in the
[backup](analysis/goods-description-refresh1000-20261015/backup-r26-before-refresh/manifest.json);
historical trained datasets stay untouched.

## Confirmed generation ambiguity and upstream clarification

Early blocks exposed two recurring ambiguities rather than a rendering defect:
the model sometimes copied the source assembly's fixed `ORIGIN: CANADA` into a
requested product fragment, or imitated a compact `3930X50 KG` expression with
an invented whole-shipment count. The actual current host already owns the
origin and declared counts. We retain analogous per-unit detail, but it must
not introduce a second contradictory count or country declaration.

The production field requirement now explicitly distinguishes per-unit fill
from whole-shipment counts; the source-assembly caption says its fixed text is
host-replaced presentation context, not requested wording. This is a short
instruction clarification, not a classifier, hard character/line quota or new
model stage. The prior second-polish requirement is preserved in
[the prompt archive](prompt-archives/goods-wording-20261015/second-polish-before-count-clarification.json).
DG and ordinary goods share this requirement.

Before integration, eight native proposals on four affected shipments, two
replicates each, retained plausible source-like detail without invented counts
or stale origins. Exact production request/schema/response equivalence is
recorded in
[guidance integration validation](analysis/goods-description-refresh1000-20261015/guidance-integration-validation.json).
The first400 requests retain their original frozen instructions; the remaining
600 use the validated clarification, as recorded in the design revision.
All ten blocks still undergo the same final projection/review gate.

Request construction was benchmarked with seven runs of1,000 identical
source/scenario requests: median211.39→219.77 microseconds, an8.39-microsecond
increase. This is negligible against native API calls lasting seconds and adds
no paid call or classification step. Before/after measurements are preserved
under `guidance-benchmark-before.json` and `guidance-benchmark-after.json` in
the analysis directory. Scope-matching word counts are soft guidance, not an
acceptance threshold; legitimate variation is retained.

## Final outcome and review scope

- **1,000 native regenerations;996 changed descriptions**, across324 templates.
  Four valid proposals reproduce the prior wording and remain unchanged.
- Ten separate100-sample scoped-model review blocks cover every initial
  regenerated passage and complete description projection. Final refinements
  receive primary review and deterministic revalidation, not a second blanket
  model-editing round. Of78 review findings,23 were resolved across
  **22 proposals**;55 were rejected as false positives or non-issues. These are
  fresh-generation refinements, not22 newly discovered corrupt original labels.
- Actual repairs remove contradictory source origins/invented shipment counts,
  reconcile contradictory processing states or explicit shipment physical form,
  and clarify a few malformed/dangling specifications. Exact HS subheading
  matching, inferred per-pack arithmetic, fictional specifications and harmless
  punctuation/casing are not requirements imposed after the fact.
- The selected sources include three metal-purity examples,58 composition-percent
  examples and71 examples with per-unit/dimensional detail. These overlap and
  supplement the deliberately detail-bearing100-sample stratum; they are not
  mutually exclusive shipment categories.
- The231 affected old-UAE descendants are identically updated in both mirrored
  sidecars. Every unselected row remains byte-identical. Real600, validation60,
  all non-description target values, and the recently regenerated452 training
  DG descriptions remain unchanged. The153 quarantined IDs remain excluded.

This is complete mechanical coverage plus a scoped model read of every candidate
description, with primary review of final corrections, adjudication of every finding and selected pair
inspection. It is **not** a claim that the primary assistant reopened1,000 full
PDFs, or that all documents underwent another unrelated-party audit.

Full [1,000-row inventory](analysis/goods-description-refresh1000-20261015/inventory.md),
[source/old/new examples](analysis/goods-description-refresh1000-20261015/examples.md),
[statistics and all-attempt accounting](analysis/goods-description-refresh1000-20261015/summary.json),
[publication](analysis/goods-description-refresh1000-20261015/publication.json), and
[independent readback](analysis/goods-description-refresh1000-20261015/independent-readback.json).

|Description words|Before|After|
|---|---:|---:|
|Selected1,000: median|32|7|
|Selected1,000: mean|39.75|9.08|
|Selected1,000: under10 words|40/1,000|680/1,000|
|Whole7,000 synthetic corpus: under10 words|704 (10.06%)|1,344 (19.20%)|
|Whole7,000 synthetic corpus: median|30|26|

The source product examples have median6 words. Thus this is principally a
source-scope correction with analogous intrinsic detail, not blanket expansion
or a forced fixed length. Refreshed descriptions still range1–113 words.
The corpus remains much less short-description-heavy than real training; this
refresh addresses part of the identified gap, not every description-boundary
failure or a demonstrated trained-model performance gain.

## Coordinate migration finding and correction

The first candidate passed text/schema preservation but a stricter comparison
found **1,010 previously known anchors lost on identical unedited lines**.
The cause was rebuilding a final synthetic document from the original template
using coarse whole-line differences: this discarded some earlier generated
party/address geometry. That coordinate candidate was rejected and archived in
`rejected-original-source-coordinate-candidate`; its text/labels were not the problem.
The interim coordinate publication was superseded before final certification
and this handoff; both its publication record and the original R26 backup remain.

The corrected reusable `transfer_existing_positions` path uses current enriched
text as its source of truth and exact UTF-8-owned edits, not substring matching.
It verifies all **97,857 unedited line anchors unchanged**, preserving duplicate
text by occurrence. Contraction interpolates inside a known, strictly ordered
owned anchor span; no extra space or new global transform is invented. Eighty-seven
expanded occurrences lack proven current-frame clearance and retain explicit
` ||`, with reasons in the receipts. This is deliberate uncertainty, not lost text.
Another128 line-changed owned regions have incomplete pre-existing anchors and
are also explicitly held. Same-line-count regions preserve each existing known
or unknown anchor. The full placement breakdown is969 same-count occurrences,
235 known single-line contractions,44 known multiline contractions,87
clearance-limited expansions and128 incomplete-anchor regions.
Full-production synthesis retains its measured-source elastic reflow; this is
the separate incremental-edit path.

Selected-input coordinate coverage is83,990/100,922 lines before (83.22%) and
82,790/99,797 after (82.96%). The0.26-point change reflects different product-line
counts and conservative abstention on changed passages; **no unedited anchor is
lost or moved**. The initial rejected rebuild had only82,113 known anchors.
The entire1,000-input current-frame probe, including artifact writes, took18.08s.
No real-input GLM/Paddle joining behavior was changed.

## Validation and measured bounds

- **1,723 synthesis tests pass** in75.36s; Ruff passes on touched production/tests.
- Eight native instruction-clarification probes precede integration; all1,000
  generation outputs, all1,000 reviews and every adjudication are retained.
- Exact byte replay proves every raw change is an approved product/commercial
  occurrence. All other target fields, mirrored memberships, hashes, order,
  quarantine exclusions and protected real/validation files pass readback.
- Actual CPU training loader:7,600 train/60 validation accepted. Final independent
  validation, including independent coordinate replay, takes16.85s and peaks at
  811MiB RSS; no model weights/training loaded.
- Native pinned tokenizer and actual `_tokenize_batch`:all1,000 pass, maximum
  source7,789/19,200 tokens and target1,639/5,500 tokens;7.14s,1,151MiB RSS,
  CPU only. Candidate-file hashes are checked before/after tokenization and
  before publication.

The clarification is not a guarantee that an LLM can never invent another count
or misleading qualifier; semantic review remains part of certification. No
end-to-end model-quality improvement or API-latency gain is claimed without a
subsequent controlled training/evaluation comparison.

## Cost and tokens

All generation, retries, review, refinement and the eight-proposal guidance call
total299 requests:245 successful,54 failed. Provider-confirmed cost is
**$0.25363886**. Three ambiguous failed calls retain a conservative additional
**$0.05925 upper-bound reservation**, not an asserted charge; known-plus-reserved
is$0.31288886. The other51 rejected attempts cost zero. Accounting preserves
1,416,978 input tokens,97,744 output tokens (32,316 reasoning,65,428 visible),
and64,832 cache-read tokens separately. Stage-level figures and every native
attempt are linked in the statistics. This is wording-refresh cost, not the
price of generating an entire shipment from scratch.
