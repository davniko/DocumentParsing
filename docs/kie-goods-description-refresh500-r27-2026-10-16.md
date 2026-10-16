# Additional short-description refresh: R27, 2026-10-16

Completed in place in **R27**. No R28 or other training-dataset version was created.
All 500 selected records are in the active 7,000-row synthetic training shard;
89 also occur in the two retained old-UAE mirrors, which were updated identically.

## Selection and outcome

Seed 2026101602; five sequential batches of 100; 261 source templates.
The draw excludes the previous1,000 refreshed records and the already refreshed
DG descriptions. Source examples:400 at most10words,75 at11–20words,25 broader.
These are selection strata, not a generation classifier or hard length quota.

| Measure | Before | After |
|---|---:|---:|
|Selected description mean words|40.116|9.454|
|Selected median words|32|6|
|Selected descriptions under10words|0|342|
|All7,000 synthetic training descriptions under10words|1,344 (19.20%)|1,686 (24.09%)|

484 descriptions became shorter;16 did not. The objective is source-like
scope and better short-description coverage, not shortening every record.
Intrinsic specifications, models, dimensions, purity and per-unit packing may
remain where the source's presentation supports that detail. The production
second polished prompt is unchanged; shipment facts were not resampled.

## Review and adjudication

Every batch was generated, projected into its existing owned product regions,
and reviewed before proceeding. A model reviewed all500 source examples,
sampled contexts, final owned passages, complete printed description blocks
and resulting goods labels. The primary agent inspected all final description
texts and every finding/correction; this is a scoped goods review, not a new
full-document semantic audit of unrelated historical fields.

|Batch|Reviewed|Model findings|Resolved findings|Rejected false positives|
|---|---:|---:|---:|---:|
|001|100|3|3|0|
|002|100|8|3|5|
|003|100|3|3|0|
|004|100|1|1|0|
|005|100|6|3|3|

Three additional primary-review findings were resolved. In total16 records
received small post-generation corrections;484 needed none. Examples:

- Removed invented40HC transport wording that contradicted the host's20HC/45HC
  inventory; retained the product and its per-unit25kg packing.
- Removed conflicting generated banana specifications, retaining the separately
  printed host specifications and their existing description projection.
- Restored a one-vehicle description after generation introduced a second unit;
  removed duplicated VIN wording from another two-region description.
- Replaced lentil-derived `CROP` terminology in motor/yarn descriptions;
  removed a model-invented Brazil origin conflicting with printed Bermuda.
- Removed an invented25MT shipment total; preserved the actual host weights.
- Removed literal `Genuine NewLine` drafting text from a product list.

No per-bag arithmetic, universal container payload limit or exact HS-to-product
classification was imposed. One review claim based only on multiplying200kg
bag fill was explicitly rejected under the agreed extraction-training objective.
Unchanged host framing, terminal commas, `NON STACKABLE` and `NOT RESTRICTED`
were not treated as newly generated wording or silently removed. This pass
does not reopen their previously certified target boundaries.

Four records required manually pinned current ownership evidence (12 product
occurrences), including nine repeated passages. Their exact source/current
quotes and hashes were checked; no first-match guess or broad replacement was
used. Source-private VIN/serial/batch literals were also checked for exact
carryover into the refreshed descriptions: none detected by that targeted test.

## Preservation and execution-path validation

- Independent replay of every edit and description projection; every non-description
  target and non-owned text span preserved.
- Current-anchor coordinate transfer replayed exactly for all500 records:
  40,460/48,536 content lines positioned (83.36%). Unknown positions remain explicit;
  all emitted coordinates are integer0–1000. This is the affected subset, not a
  claim that the whole dataset's geometry changed to this percentage.
- Actual frozen direct-target schema, casing and training loader passed:
  7,600 training /60 validation records.
- CPU-only pinned training tokenizer passed all500 final candidates:
  maximum7,792 source /1,619 target tokens, limits19,200 /5,500;
  zero source truncations.4.06s;1,176,616KiB process peak RSS, including imports.
- Real600 and validation60 byte-identical during this goods-only publication;
  quarantine, ordering, memberships and all mirrors verified.
- Publication+independent validation32.13s. No GPU/model weights/training launched.

Subsequent manual real-data anonymization is a separate authorized R27 mutation;
see [its report](kie-r27-real-training-anonymization-2026-10-16.md).

## Cost and evidence

OpenRouter GLM-5.3-Flash through Together; 55 generation requests (including five
scoped validation repairs), 50 semantic-review requests. All 105 saved attempts:
687,960 input /45,360 output tokens;15,991 reasoning tokens are a subset of output.
Provider-reported cost **$0.12582792**: generation$0.06455327,
review$0.06127465. Codex manual work is not included in these provider figures.

Evidence directory: `docs/analysis/goods-description-refresh500-r27-20261016/`:

- `design.json`, `ownership-preflight.json`, `adjudicated-owner-overrides.json`;
- five batch generation/repair/review/certification files;
- `corrections.json`, `adjudications.json`, all provider call receipts;
- per-record `repaired/` and `repair-receipts/`, `before/` backup;
- `token-limit-validation.json`, `staged-validation.json`, `publication.json`.

Published synthetic SHA is recorded in R27's manifest; the goods-only publication
manifest SHA was `2562e29097a04cf3ccadaa221b104d29360701c10f2ff11031cee79cc4ec1a2a`.
Later authorized R27 edits necessarily change its manifest; use the current
manifest and the independent receipts rather than treating this historical pin
as an immutable alias for R27 forever.
