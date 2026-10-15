# Synthetic-only training coverage and augmentation plan

Updated: 2026-10-14.

Repair follow-up: the HS6-only sampling/label defect and four old UAE composite
shipper labels described below have now been repaired; see
[the HS/DG repair report](kie-hs-dg-format-repair-2026-10-14.md). Missing DG
presentations are addressed through separately certified new templates and an
authoritative property registry, not by adding unprinted values to existing
labels. The original selection manifests below are historical proposals: their
row/file pins must be refreshed before assembling an augmentation from the
now-repaired sidecar. The original coverage tables describe the pre-repair audit,
not the latest code-width inventory. No synthesis/training is launched by this
planning document.

## Recommended next dataset

Retain the current **5,000 repaired synthetic records**, remove the **600 real
training records**, add the selected **1,000 old UAE records**, and generate
**300 Egypt-endpoint records with targeted format/field coverage**. This gives
**6,300 synthetic training records**, with the same **60 real validation
records** kept separate.

The existing 1,000-record selection is sufficient to recover every old UAE
long-roster, membership-only allocation and transshipment example. A nested
1,500-record alternative is also prepared, giving 6,800 training records after
the Egypt additions. Its extra 500 records add ordinary lexical variation and
more thermal/DG examples, but no additional source layouts or examples of those
three structural priorities. Start with 1,000 to keep the augmentation focused.

**The Egypt generation needs more than a destination override.** The current
sampler produces only six-digit HS labels, and the admitted source templates
lack DG packing-group/flash-point/subsidiary-hazard slots. Removing the real
records without addressing those gaps would remove all positive supervision for
several active target fields and formats. Sections below define the finite work
needed before that generation.

This is a completed inventory and selection plan. The audit did not change
dataset rows, templates, production code or training configuration, and did not
launch generation or training. API cost: **$0**.

## Available cohorts

The working pool contains **7,500 unique synthetic records**, not
7,500 additional records. Training JSONL mirrors and archived revisions are not
new examples. The retired, previously defective 30k synthesis is outside this
audited pool.

Four older UAE records have unresolved composite-company targets and are
excluded from the proposed additions. The eligible old UAE pool is therefore
**2,496**, before the exact 1,000/1,500 selection. None of those four is in the
current 5,000. See the source-authority findings below.

|Cohort|Documents|Source templates|Multiple containers|Five or more containers|Eight or more containers|Descriptions with 30+ words|Thermal setpoints|DG declarations|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Retained broad-country synthetic|2,500|300|714|202|48|2,027|99|35|
|Current replacement UAE synthetic|2,500|360|894|318|180|756|300|300|
|Previous UAE synthetic, currently excluded|2,500|358|985|336|180|1,050|180|100|
|Real training, proposed removal|600|600 source documents|193|61|26|11|24|17|
|Real validation, unchanged|60|60 source documents|24|6|3|2|4|1|

The retained cohort's 300-template count and all other counts are reproduced in
[coverage.json](analysis/synthetic-only-augmentation-audit-20261014/coverage.json).
Counts in this table are documents, not container rows. Description word bins
are diagnostic length bins, not annotation boundaries or rules for rewriting
descriptions. Thermal means an actual container temperature-setpoint target;
generic reefer equipment is counted separately.

Use the repaired R20 files:

- [Current synthetic 5,000](../data/curated/mpci-bl-real600-synthetic5000-uae-coverage-r20-party-masterbill-positions-v1/synthetic.jsonl).
- [Current UAE 2,500 subset](../data/curated/mpci-bl-real600-synthetic5000-uae-coverage-r20-party-masterbill-positions-v1/replacement-synthetic.jsonl).
- [Repaired old UAE 2,500](../data/curated/mpci-bl-real600-synthetic5000-uae-coverage-r20-party-masterbill-positions-v1/previous-uae-synthetic.jsonl).

The old UAE sidecar has the recent party, master-B/L, secondary-organization,
contact and private-reference repairs. Do not retrieve additions from the
earlier campaign's unrepaired outputs or frozen training file. All these R20
targets are direct extraction objects, without `schemaVersion`/`documentPatch`
envelopes.

## What the previous run supports targeting

The previous results are described in
[results-ledger section 16](mpci-training-results-and-next-experiments.md#16-uae-coverage-replacement-run-k).
J used the old UAE cohort; K replaced it with the current UAE cohort. Both used
5,000 synthetic plus 600 real records, rank 48 and the same 60 validation
references.

|Complete common target|Previous J|Current K|
|---|---:|---:|
|Field F1, all 60 documents|0.9017|0.8937|
|Field accuracy, all 60|0.8650|0.8516|
|Relation F1, all 60|0.9450|0.9336|
|Field F1, same 58 jointly parseable documents|0.9154|0.9073|
|Correct size-category values, same 58|89/104|96/104|
|Correct equipment identifiers, same 58|110/112|102/112|
|Correct placement identifiers, same 58|110/112|100/112|
|Correct placement quantities, same 58|94/98|90/98|

**Size classification improved; container identity and allocation recall did
not.** The old cohort is useful for recovering uniform multi-container lists,
repeated ID/seal/quantity ownership and the ordinary cargo/document structures
that were less prominent after the replacement. Its descriptors also give
additional moderate-length product wording. The current cohort should remain
intact to retain its new 20HC/45HC, mixed-equipment, rare-package and alias
contrasts.

These results identify capability targets, not causal attribution to individual
training examples. The J/K comparison did not perform sample-removal ablations,
and its paired uncertainty intervals include zero for aggregate F1 differences.
No validation identities or gold values enter the selection optimizer.

Descriptions need boundary/ownership coverage rather than indiscriminately
longer wording. The retained broad-country cohort already has 1,124 descriptions
with at least 60 words. Validation has only one such description. The matched
source examples illustrate preserved specifications, repeated product blocks,
distinct declarations and container ownership; simply increasing verbosity is
not the objective.

## Exact old UAE selections

|Overlapping coverage in additions|Selected 1,000|Selected 1,500|
|---|---:|---:|
|Distinct source templates|358|358|
|Multiple identified containers|619|688|
|Five or more containers|336|336|
|Eight or more containers|180|180|
|Membership-only placement documents|175|175|
|Explicit transshipment-port documents|48|48|
|Descriptions with 30+ words|500|750|
|Descriptions with 60+ words|100|120|
|Descriptions under 30 words|500|750|
|Container temperature setpoints|80|120|
|DG declarations|50|75|
|Negotiable consignee instructions|120|150|
|Explicit goods net-weight targets|286|362|
|Master B/L targets|25|25|

Each selection includes all 487 distinct old UAE records satisfying at least one
of the long-roster, membership-only or transshipment priorities, plus at least
one record from every available source. Overlapping quotas also preserve
description, thermal, DG, negotiability and net-weight coverage. A binary
optimizer enforces exact cardinalities and all quotas, then favors sources with
less exposure in the retained/current dataset. Stable document-ID hashes break
ties. Its optimum concerns this explicit coverage/exposure objective, not an
unmeasured model-performance objective.

The 1,500 option contains every member of the 1,000 option. Selection manifests
pin the repaired sidecar and each member's entire row, target, OCR and positioned
input. They are references to existing records, not regenerated samples:

- [Exact 1,000-member selection](analysis/synthetic-only-augmentation-audit-20261014/old-uae-1000-selection.json).
- [Nested 1,500-member selection](analysis/synthetic-only-augmentation-audit-20261014/old-uae-1500-selection.json).
- [Complete sample inventory](analysis/synthetic-only-augmentation-audit-20261014/sample-inventory.csv).

### Resulting coverage before the Egypt additions

|Training-only coverage|Current synthetic 5,000|With selected 1,000|With selected 1,500|
|---|---:|---:|---:|
|Total documents|5,000|6,000|6,500|
|Multiple containers|1,608 (32.16%)|2,227 (37.12%)|2,296 (35.32%)|
|Five or more containers|520|856|856|
|Fully assigned mixed equipment|577|577|577|
|Membership-only placements|252|427|427|
|Thermal setpoints|399|479|519|
|DG declarations|335|385|410|
|Negotiable|333|453|483|
|Transshipment port|107|155|155|
|Container rows|10,138|13,907|14,527|
|20-foot high-cube rows|635|635|635|
|45-foot high-cube rows|1,142|1,145|1,171|
|Located text-line coverage|84.07%|83.67%|83.97%|

The old selections add mostly familiar GP equipment. They preserve absolute
20HC/45HC exposure but reduce its relative frequency. For example, 20HC drops
from 6.26% to 4.57% of container rows with the 1,000 addition; 45HC drops from
11.26% to 8.23%. Egypt generation should retain those contrasts, including
source-supported mixed inventories, rather than only adding 40HC rows.

Alias-surface inventory finds 58 reviewed aliases in the current 5,000, 23 in
the full old cohort, and 20/22 in the selections. These are literal full-OCR
surface observations, not per-ID classifications. Native aliases outside the
registry are not counted. See
[alias coverage](analysis/synthetic-only-augmentation-audit-20261014/alias-coverage.json).
For example, a generic tariff line saying `USD 100/20rf ... USD 200/40rf`
occurs in GP-container documents. That is non-extraction tariff wording, not
a mislabeled reefer. Literal alias presence must not override actual inventory
ownership or be counted indiscriminately as positive category supervision.

The selected 1,000 alone has 81.70% coordinate coverage because it deliberately
concentrates long inventories and membership layouts with less complete source
anchoring. It does not introduce changed text, out-of-range points or new
synthetic reflow. Their coordinates and earlier geometry receipts are retained.

## Coverage removed with the real training records

### Source layouts and large inventories

All 360 admitted synthesis sources are in the current real training set. The
other **240 real sources have no admitted template**. The old UAE cohort adds
zero new layouts to the current synthetic set.

Real training has five ten-container sources, one eleven-container source and
three 39/40-container sources; neither synthetic cohort has those exact roster
lengths. Synthetic examples do include 13- and 14-container inventories, so
long-list supervision is not absent. New 10/11-container templates would add
different real roster presentations, not just a higher numeric maximum. These
are training-source candidates only; do not use validation sources to fill them.

### HS-code formats

|HS target digit count|Current synthetic 5,000|Old UAE 2,500|Real train 600|Validation 60|
|---|---:|---:|---:|---:|
|6|3,863|2,318|171|10|
|8|0|0|186|22|
|10|0|0|149|16|
|12|0|0|86|3|
|7 or 11|0|0|5|0|

Counts are code values; a document may carry several. All labels here are
already digits-only. This is a **code-width coverage gap**, not a casing or
punctuation-scoring problem. The field policy preserves every printed digit;
it does not truncate national codes to HS6.

The sampler explicitly assigns `goods["hsCodes"] = [i.hs6 for i in identities]`
in [curated_scenarios.py](../src/document_ocr/synthesis/curated_scenarios.py:1622).
It uses a six-digit classification identity for both goods context and the
rendered/extracted code. Adding old UAE records cannot introduce longer codes.

For Egypt generation, separate the HS6 classification context from the complete
printed/extracted code. Preserve source-supported code counts and lengths;
sample complete values using registry descendants or explicitly synthetic
source-format suffixes, then render and label the same full value. This is
generation of new document facts, not padding/truncating labels after OCR.
Exact HS/product taxonomy is not the agreed learning objective. Any claim that
a generated national code is legally valid needs a national registry; format
synthetics must not be described as that.

### DG and other disappearing fields

|Target absent from both synthetic pools|Real positive documents|Validation positive documents|Existing admitted source support|
|---|---:|---:|---|
|DG packing group|9|1|None|
|DG subsidiary hazards|1|0|None|
|DG flash point|2|0|None|
|Free-text package fallback `typeOfPackages`|9|4|None|
|Original B/L number|1|0|None|
|Freight `payable_elsewhere` value|2|0|None|
|DG `GASES` category|1|0|One source, but no sampled synthetic occurrence|

The generator updates packing-group/subsidiary fields only when the source
goods tuple already includes those fields:
[source-preserving DG assignment](../src/document_ocr/synthesis/curated_scenarios.py:1623).
No amount of donor weighting adds missing printed/compiled field slots. The
GASES source is present, but chemical eligibility and physical-profile selection
have generated other classes; a printed source category is not proof that its
generated descendants cover that category.

Useful uncompiled real sources are:

- `2e8c2a1f`: low-danger packing group and explicit 110°C flash point.
- `32ebac37`: low-danger packing group plus subsidiary flammable-liquid hazard.
- `a1087462`: medium-danger packing group, 12°C flash point and fibre drums.
- `3bc03951` or another inventory candidate: printed package fallback such as `SETS`.
- `6cc8c295` or another unpackaged-vehicle source: single-unit unpackaged cargo.

The full IDs, current targets, page/container counts and missing-field reasons
are in the
[Egypt source-gap candidate inventory](analysis/synthetic-only-augmentation-audit-20261014/egypt-source-gap-candidates.json).
Those source examples show what can be compiled; they are not yet admitted
templates. DG sampling must retain a coherent supported regulatory tuple and
correct printed settings/ownership. Fresh source/template probes are required
before counting these new fields as generated coverage.

### Ten named package categories disappear

Removing the real set eliminates these positive category targets from training:

- Block; fibreboard box; fibre drum; octabin.
- Flexibag; flexitank; bulk liquid.
- Bulk granular/grain solids; bulk fine/powder solids.
- Unpacked/unpackaged single unit.

They total **19 real package rows**. Only the single-unit unpackaged category
appears in current validation, once. The existing registry has a source for
bulk fine/powder solids, but none of the other nine. An existing source category
still does not prove a compatible sampled load profile or rendered owner.
These are genuine positive-supervision gaps, not evidence that the existing
synthetic package labels are incorrect. The rare categories already covered
by current UAE synthesis remain covered when the old examples are added.

There are also 26 HS chapters present only in real training. Because HS/product
classification is not the extraction objective, these are lexical/topic
diversity losses, secondary to code-width and field-presence coverage. Counts
and exact paths are in
[field coverage](analysis/synthetic-only-augmentation-audit-20261014/field-coverage.json) and
[real-removal gaps](analysis/synthetic-only-augmentation-audit-20261014/real-removal-gaps.json).

## Egypt 300 design

Egypt-specific exposure is sparse in the retained broad-country synthesis:
15 literal `EGYPT` consignee-country targets and eight explicit discharge-country
targets. Country absence is not absence of an Egyptian route: some templates
print only a port name or an order instruction, and those remain sparse labels.
The added Egyptian records should therefore support local route and party
wording without inventing country/party fields that are not printed.

Start with this exclusive family allocation, while assigning overlapping
field/format goals independently:

|Goods family|Documents|Purpose|
|---|---:|---|
|Ordinary ambient/vehicle/documentary cases|240|Single and uniform/mixed multi-container controls, ordinary descriptions, package-role boundaries, receipts/delivery and complete locations|
|Thermal cargo|40|Retain chilled/frozen, reefer aliases, signed settings and ownership|
|DG|20|Prioritize missing packing-group/subsidiary/flash-point slots and a supported gas declaration, rather than more instances of already dominant classes|
|Total|300|All terminal/import endpoints in Egypt; origins varied|

Suggested overlapping goals:

1. At least 100 source layouts, with 260 or more records exposing supported
   8/10/12-digit HS values and a small six-digit/absent-HS control group.
   Include source-supported repeated/multiple-code presentations. Do not create
   new goods identities solely to increase code counts.
2. Approximately 100 uniform multi-container and 40 mixed-container records;
   use a substantial five-plus-container portion for the list-recall target.
   Keep ordinary single-container and no-ID controls. These are overlap goals,
   not extra documents on top of the family allocation.
3. Preserve complete product specifications and legitimate continuations,
   with mostly short/moderate descriptions rather than forcing verbosity.
   Exclude separately owned marks, declarations and references under the current
   policy. Retain declared-package/allocation consistency.
4. Include the missing free-text-package fallback and single-unit unpackaged
   patterns after their source bindings and sampling paths are admitted. The
   three DG sources above form a small concrete starting group for the missing
   DG fields; they need fresh compilation/generation validation.
5. Retain negotiable/order and notify-reference contrasts, transshipment roles,
   complex locality wording and the current rare equipment/alias contrasts.
   Preserve genuine unavailable-role nulls and unassigned ownership.

These are generation-plan targets, not a realized 300-slot campaign. The next
implementation should prove the joint finite plan against source capabilities,
perform small probes for the full-code and new-field paths, then generate the
300. Destination constraints are currently unset in all 360 registry cases,
but route topology and repeated country/party owners still require the normal
compiled checks. Use the active registry/source authorities and shared auxiliary
file, not frozen experimental configurations from earlier UAE repairs.

## Source authority and validation

The live registry differs from the current real targets in four source families:

- `20cf0a0c` and `6c14020d` add OCR-supported gross masses in the isolated
  synthesis authority.
- `a3e3c1e3` removes consignee contacts supported only in the goods continuation.

Those three changes are documented in
[the source-preparation report](kie-synthesis-expansion360-results-2026-10-10.md#source-preparation-and-exclusions).
They were intentionally not applied to the protected real training set at that
time. Removing real training also removes those three old annotation problems
from the next training file; the corrected descendants remain usable.

The fourth, `ca74e1e4`, has a composite `A VIA B` shipper source authority while
the real target selects A alone. It has **16 current plus eight old synthetic
descendants**. All 16 current and four old descendants replace that complete
source name with one generated company and an OCR-grounded generated address.
The other **four old descendants still join two companies in `shipper.name`**,
with one address whose entity ownership is not independently established.
That is a policy/ownership defect, not a formatting or anonymization issue.

Those four are excluded from augmentation without modifying the protected
dataset. The initially selected defective record was replaced and the selections
rebalanced; all counts and coverage floors still hold. A whole-pool scan for
VIA/on-behalf/O-B/care-of name markers finds these four in old UAE and zero
in the current 5,000. Before generating this original compound-source case
again, explicitly reconcile primary/auxiliary entity ownership or adopt an
unambiguous, single-entity synthetic rewrite. Do not just truncate the company
name and retain a potentially borrowed address. The other 20 descendants'
single-company names and applicable addresses are present in their OCR.

The [four-record hold inventory](analysis/synthetic-only-augmentation-audit-20261014/review-exclusions.json)
and exact source differences are saved in
[source-authority differences](analysis/synthetic-only-augmentation-audit-20261014/source-authority-differences.json).

The all-row audit checked **8,160 unique records**: 7,500 synthetic, 600 real
training and 60 validation. It verifies published file hashes, 360 cases and
shared registry hashes; the exact training task/schema/constraints;
canonical direct targets; duplicate IDs/input hashes; synthetic equipment-ID
presence and placement membership; positioned-text replay and 0–1,000 coordinate
ranges; and validation separation by ID, normalized source text and B/L number.
All invariant checks pass; the four semantic-policy holds above remain excluded
from augmentation. No published dataset or registry byte changed.

These checks verify the stated invariants and reuse the existing semantic,
privacy and geometry certificates. They are not a new independent model review
of all 7,500 texts. Eight training-trait-selected source/old/current UAE pairs
are provided in full for inspection:
[matched examples](analysis/synthetic-only-augmentation-audit-20261014/MATCHED_EXAMPLES.md).
Prior full-cohort repair evidence remains linked in the
[current completion certificate](analysis/synthesis-private-identifiers-repair-20261013/completion-certificate.json).

The constrained selections were rerun with identical member hashes. Twelve
tests pass, including rejection of text/coordinate corruption, altered page or
line counts, duplicate IDs, old target envelopes, illegal values and impossible
selection quotas; an independent pass checks every selected row hash and every
mandatory coverage member. The final all-row audit took **25.44 seconds**, with
**601.15 MiB** peak RSS. No GPU or paid calls were used.

Before publishing the next training set, enforce the same whole-cohort semantic
and geometry gates on the 300 new descendants, verify that all original 5,000
rows and all selected old rows are preserved exactly, and measure token lengths
on the actual assembled inputs/targets. Keep validation unchanged. Retain
rank 48 and the previous optimizer/input settings for this composition
experiment unless a separate rank change is requested.

The new experiment also uses the already-approved R20 direct target and privacy
repairs, whereas completed J/K trained historical wrapped targets. Its result
will measure the new overall training recipe; it cannot isolate real-removal,
old-UAE addition, Egypt augmentation and direct-output formatting as separate
causes. The fixed validation set supports the same field/category/relation
comparisons and an explicit list-completeness breakdown.
