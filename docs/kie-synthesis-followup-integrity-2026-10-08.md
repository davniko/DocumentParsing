# Synthesis follow-up: old500 back-check and reusable fixes

## Outcome

The requested back-check and repair are complete. The old500 and combined1500
synthetic datasets were repaired **in place**, with exact pre-edit backups. The
training mixture remains **600 real + 1,500 synthetic**, with **60 unchanged real
validation records**. No training was started. No document was dropped or fully
regenerated to bypass a failing check.

This report supersedes the earlier campaign report's statement that the old500
remain byte-identical: that described the initial publication, before this
explicitly requested repair. The real600 and validation60 still are unchanged.

| Change | Old500 | New1000 | Target impact |
|---|---:|---:|---|
| Repeated exporter registration left at source value | 5 | 3 | None: source-only private text |
| Product operating masses exceed whole shipment gross | 1 | 0 | One goods description corrected with its input |
| Re-render existing length-only equipment receipts explicitly | 11 | 0 | None: fuller equipment support already existed elsewhere |
| **Distinct changed input records** | **17** | **3** | **One changed target in all1500** |

The 11 equipment changes are consistency improvements from replaying the current
renderer, **not 11 established old annotation defects**. All other targets,
shipment facts, sample IDs, split membership and unrelated sample metadata are
preserved. Only the corresponding 20 positioned inputs change.

The [exact preservation result](analysis/synthesis-followup-integrity-20261008/final-preservation.json)
lists every changed ID. The
[repair inventory](analysis/synthesis-followup-integrity-20261008/repair-changes.json)
binds before/after candidate hashes and source IDs. Original publications,
wording, candidates, reviews, adjudications, dataset files and shared catalog
declarations are recoverable under
`docs/analysis/synthesis-followup-integrity-20261008/before/`; its manifest pins
**5,076 original files**. Superseded publications and geometry galleries are
also retained under that investigation's `retired-publications/`.

## Back-check scope and adjudication

The [old500 inventory](analysis/synthesis-followup-integrity-20261008/old500-targeted-content-audit.json)
screens all500 from all100 original families. The production renderer then
replays all1500 against the current shared contracts, schema, physical facts,
party policies, auxiliary dependencies and wording checks. Complete rendered
OCR/target review receipts cover all1500 final candidates: unchanged exact
requests can reuse prior model responses; changed content is reviewed again.
Old500 reviews were refreshed, not assumed valid merely because a template had
passed an earlier pilot.

The targeted old500 content screen included:

- 77 mass/rating candidates, 16 packing/count candidates and 30 thermal
  candidates, inspected in context. These sets overlap and are not defect counts.
- 1,510 party address values checked against their owned rendered fragments:
  1,485 match the owned word/number sequence directly; 25 additionally include
  a separately printed country belonging to the same party.
- Searches for unresolved interpolation/scaffold text, stale changed party
  literals and zero-postcode labels found no remaining hits.
- Current review findings were adjudicated against the specific sample and,
  when necessary, the original PDF/layout. For example, `3d0d3748` prints receipt
  and freight-payment places in adjacent columns; their linear OCR order is not
  ownership. Inspection of PDF page1 confirmed the existing bindings.

The [16 current old500 review dispositions](analysis/synthesis-followup-integrity-20261008/old500-review-decisions.json)
include exact evidence quotations and finding hashes. They are rejected findings,
not additional defects: examples include product/classification preferences,
private facts intentionally absent from targets, and false reading-order claims.
Two other review findings belonged to archival candidates outside the configured
500 and were not counted as current dataset defects.

This is complete programmatic replay plus full-pair model review and targeted
manual adjudication. It is not a claim that a human read all1500 pairs line by
line or that street deliverability was independently certified.

## Defect 1: repeated exporter identity

Source `5242647c` has a tax registration, an `EXID` representation, and a repeated
foreign-exporter registration. The tax and foreign-exporter occurrences varied,
but the numeric core and country prefix in `EXID` were left at their source value.
It affected all **eight** published descendants of this source: five old and
three new.

Example, `syn_full_v7_2ca7b8a04845376412e3faf7`:

```text
Before:
TAX : 3164376654
EXID : TR-02-1800041427
FOREIGN EXPORTER ID: 3164376654

After:
TAX : 3164376654
EXID : PH-3164376654
FOREIGN EXPORTER ID: 3164376654
```

The repair is in the reusable **ownership and auxiliary declarations**, not a
document-wide substitution or a Python branch for this document. The exact
numeric source occurrence now uses the same existing registration sampler as
the related occurrences; its prefix uses the sampled origin-country code. The
source OCR snapshot stays unchanged. No public registration label was invented.
This provides internally coherent synthetic identifiers, not a certification of
each country's real registration syntax.

All eight descendants were re-rendered from those corrected declarations.
**100 further production-path draws**, spanning **81 countries and 100 distinct
registration values**, pass equality/prefix checks. See the
[stress receipt](analysis/synthesis-followup-integrity-20261008/exporter-binding-stress.json).
All1,204 catalog case/shared file hashes pass the final independent inventory.

## Defect 2: generated item masses contradict shipment accounting

Old sample `syn_full_v7_93f7ff8b7d313b514e844782`, source `fbfcab7a`, lists six
loaders. Two entries claimed operating weights **16,500kg and 12,800kg**, already
**29,300kg**, while the whole shipment gross is **20,500kg**. Those are actual
machine masses, not lifting ratings.

Only the two unsupported operating-weight clauses were removed, from both
generated wording/input and the matching description label. Models, serials,
bucket capacities, six packages, shipment gross, volume and container facts stay
unchanged. The original wording response remains preserved.

The upstream cause had two parts:

1. Generation needed the physical facts that would actually be printed, including
   source-only totals and final rounded precision—not just public donor labels.
2. Validation did not reject explicit item-mass lower bounds exceeding that gross.

`curated_campaign.py` now builds wording context from the same physical render
plan used by the renderer. It includes printed public/private measures; unasserted
zero placeholders are excluded. Generation, contact/postal resumption and render
replay all use the same request identity. The physical plan is reused inside
rendering rather than recomputed there.

`curated_wording.py` now validates explicit actual unit/operating/shipping masses
against known whole-cargo gross. It handles canonical kilograms, grams, pounds
and metric tonnes; unknown units and ambiguous newly generated number notation
are not silently coerced. Clearly separate named product-list entries contribute
a lower bound, while repeated entries and mass-only specification bullets are
not double-counted. Ratings, product dimensions, density and size grades are not
treated as shipment mass.

The check runs through the existing generation validation/bounded-correction
path and again when saved wording is validated for rendering. Exhausted
correction attempts remain visible failures; there is no auto-accept fallback.
Explicit generator-context captions are also rejected from goods wording.

This is a deliberately bounded arithmetic guard, not a claim to understand every
possible physical assertion in arbitrary prose. Full rendered-pair review and
adjudication remain part of publication.

## Scope discipline: what was deliberately not “fixed”

The task is extraction training, not perfect commercial prose or tariff advice.
We therefore did not add rules that would rewrite or reject:

- Related product variants merely because an exact HS classification might differ.
- Valid inner packages within a different outer package category.
- Processing/freezing temperatures distinct from the carriage setting.
- Equipment ratings, density/GSM, dimensions or approximate product specifications
  treated by an overly broad scan as shipment totals.
- Unusual but extractable product wording, distinct invoice/issue dates without
  a proven role conflict, private equipment tare, or source phone suffixes that
  correctly remain outside full-phone labels.
- A package average interpreted as a mandatory maximum or uniform fill. Without
  an explicit per-package allocation, total net/count does not prove that every
  package has that weight. Two fresh-probe findings of this kind were rejected.

An attempted broader package-fill matcher incorrectly rejected 17 coherent
existing samples. It was discarded; the existing package-fill guard is unchanged.
This negative result is important: expanding rejection coverage is not an
improvement if it destroys legitimate description variability.

## Fresh synthesis, regression tests and geometry

A separate fresh-seed probe generated **16 samples from eight affected families**
using the actual production generation, contacts, postal correction, rendering,
review, publication and coordinate paths. They are inspection probes, **not extra
training samples**.

The first probe exposed the private-gross context gap: a 21,000kg excavator was
generated for a template printing 13,143kg gross. Rendering correctly rejected
it. After supplying that actual printed total to generation, the fresh probe was
rerun and all16 completed. Two calls used the existing bounded wording repair
for package-fill/accounting intrusions; postal corrections were also exercised.
It is not claimed that every first model response was valid.

- [All16 source/rendered pairs](../artifacts/kie-synthesis-production/curated-v7-followup-integrity-probe-v1/samples.md).
- [Positioned pairs](../artifacts/kie-synthesis-production/curated-v7-followup-integrity-probe-v1/positions-reflow-v1/samples.md).
- [Four-document geometry gallery](../artifacts/kie-synthesis-production/curated-v7-followup-integrity-probe-v1/positions-reflow-v1/audit/GALLERY.md).

Final targeted tests: **447 passed in16.20s**. The suite covers generation and
rendering, source-only and rounded gross, compatible/incompatible item masses,
units, product/specification-list distinctions, coherent package details,
ownership, auxiliary interpolation, equipment, contacts, publication and geometry.
Ruff and whitespace checks pass.

The original1500 wording regression has exactly **one newly rejected sample**:
the known uncorrected loader example. There are no newly rejected coherent
samples in that corpus. All1500 repaired records replay successfully.

Independent geometry validation was rerun for **all1500**, with zero lost source
anchors. Coverage remains **128,977 / 154,424 lines (83.52%)**. Coordinate gaps
remain explicit; no text was dropped to improve coverage. The fresh16 have
1,328/1,617 positioned lines, including77/89 goods and112/181 address lines.
Four adversarial controls—changed text, off-page coordinates, duplicated lines
and inconsistent anchors—are rejected in each campaign/probe audit. Refreshed
galleries include28 existing documents and the four fresh probe documents.

## Measured overhead and spend

The [1,500-record wording benchmark](analysis/synthesis-followup-integrity-20261008/wording-regression-benchmark.json)
measured a median **0.1049s before versus0.1325s after**: approximately **18µs extra
per sample**, with no additional model-call stage. This is a validation-component
measurement, not an end-to-end throughput claim. Peak RSS of that harness was
692.5MiB, including three loaded campaigns. A narrower generated-description
harness measured about146MiB; these are different workload scopes.

Recorded extra API spend, including failed/retried attempts:

| Work | USD |
|---|---:|
| Refresh200 old reviews | 0.06791110 |
| Refresh300 old reviews | 0.09837105 |
| Changed new1000 review requests | 0.00154425 |
| Fresh16 probe, all attempts/corrections/reviews | 0.02335170 |
| **Total follow-up** | **0.19117810** |

Engineering/manual-review time and training compute are not included. No training
compute was used. Unchanged paid responses were reused where their request
identity matched; historical reviews with different contracts were not silently
treated as equivalent.

## Final published state and subsequent runs

The combined dataset passed schema/casing checks on all2,160 targets, full
synthetic review/adjudication coverage, exact rendering and geometric checks,
train/validation separation and input/hash checks. The preservation test verifies
the exact single-target edit and unchanged real/validation records.

| File | Current SHA-256 |
|---|---|
| old500 `synthetic.jsonl` | `408a8b31638f4c7d9560cc4339ce0baddb5b283a10d63d0492137ca832368fba` |
| old500 mixture `train.jsonl` | `97404211dc21bbab6103d0975574dc632d3fcfb1f5967f99bd4881f2dfbc51e8` |
| current1500 `synthetic.jsonl` | `8b6120752452bf6cf78d61e4ea0d05132dcd8c3d76c058d4ed1594c6eec5313b` |
| current1500 mixture `train.jsonl` | `b2dc5f3b2515a63268068bc294fb809827bdce2ef2835c77cb8b0e890ac2300b` |
| unchanged `validation.jsonl` | `14d306aff64e2903a8496d5809279ec42f4e9672b6d50432e72b8d170ec50f68` |

Both mixture training configs now pin the repaired train files. No hyperparameter,
prompt, target schema or split allocation was changed. The repaired current500
are copied exactly into the1500 set; the manifest explicitly distinguishes that
from preserving the pre-followup500. Source template ownership/auxiliary pins are
updated, and historical data remain recoverable.

The rebuilt Compose image passed its dependency verification and the actual
`kie-tools inspect-dataset` command for both configs: 2,160 records in1.377s for
the current mixture, and1,160 records in0.788s for the old500 mixture. These checks
load the image's installed code and verify current input hashes, prompt/schema
compatibility and targets; no model weights or training loop were started.
Complete combined-dataset assembly/validation took27.80s with549.4MiB peak RSS.

The prior equipment, nested auxiliary, Unicode and source-ownership fixes remain
in the reusable renderer/catalog; this pass verifies them against the old500 and
adds the mass-context/guard and repeated-EXID corrections. Future runs use these
paths and must still complete full-pair review and publication gates. The known
repaired failure paths are covered; unrestricted future LLM wording is not
promised to be infallible or exempt from review.
