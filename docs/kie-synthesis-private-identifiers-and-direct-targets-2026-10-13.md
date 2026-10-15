# Private fiscal/commercial identifiers and direct extraction targets

## Outcome and dataset scope

This pass completes the approved source-owned tax/registration/customs and
commercial-reference repairs in the R20 working copy. It also removes the
training-target envelope: the model emits extraction facts directly, without
schemaVersion or documentPatch. No model training was launched.

Working dataset:
[R20 README](../data/curated/mpci-bl-real600-synthetic5000-uae-coverage-r20-party-masterbill-positions-v1/README.md).
Exact pre-edit snapshots, decisions, compiled owners, edit receipts and
validation evidence:
[audit directory](analysis/synthesis-private-identifiers-repair-20261013/).
Publication is pinned by the
[completion certificate](analysis/synthesis-private-identifiers-repair-20261013/completion-certificate.json)
and independently read back in
[publication verification](analysis/synthesis-private-identifiers-repair-20261013/publication-verification.json).

| Synthetic cohort | Records checked | Documents repaired | Exact text edits |
| --- | ---: | ---: | ---: |
| Retained non-UAE | 2,500 | 1,746 | 6,251 |
| Current replacement UAE | 2,500 | 1,514 | 5,500 |
| Superseded previous UAE | 2,500 | 1,590 | 5,863 |
| Total | **7,500** | **4,850** | **17,614** |

The active training cohort remains 600 real + 5,000 synthetic; validation
remains 60 real. The superseded UAE records stay in the separate sidecar,
not in training. There are 3,260 repaired documents in active training.
Already anonymized, coherent identities are retained rather than regenerated.

**No extraction fact changed in this pass.** These fiscal/commercial jobs
proved input-only in the actual dataset. All 660 real inputs and factual
targets are preserved. The real targets' serialization changes only because
the outer envelope is removed. Every physical line, page boundary and
coordinate suffix is preserved, including records whose identifier lengths
were reconciled to their proper source envelope.

## Diagnosis and inventory

The preceding contact audit was not a fiscal/commercial repair certificate.
The new census starts from its reviewed private-identifier taxonomy, then
extends it with the actual typed source-owner inventory across all 360 cases.
That extension catches short references, multiline codes and continuation-page
owners missed by caption-only searches.

The original census captured 1,389 source-value candidates. These are candidates,
not 1,389 independent private identifiers: prefixes, nested importer IDs and
repeated presentations overlap. The completed decisions distinguish private
references from dates, masses, quantities, equipment, HS codes, contacts,
postal text, voyage identifiers and public carrier/legal constants. The
[source decisions](analysis/synthesis-private-identifiers-repair-20261013/source-decisions.json)
preserve the contexts and classifications.

The compiled repair covers **1,644 numeric/alphanumeric owners and 1,985 source
occurrences across 359 templates**. All 360 templates receive an explicit
identifier-privacy inventory, including the source with no numeric repair job.
The current generator classifies source-only identifier owners as private,
public or dependent. Missing ownership/classification is a pre-paid error.

### Categories replaced

- Private importer/exporter/party tax, VAT, registration and customs references.
- Shipment-specific import/advance-cargo declarations and export identifiers.
- Invoice/proforma, purchase/sales order, booking, customer/account, contract,
  payment, letter-of-credit and other reviewed private commercial references.
- Repeated and differently punctuated presentations of the same owned code,
  including continuation pages and source-only Marks/reference text.

No new target fields are introduced. The existing auxiliary target recipe
remains available if a future schema explicitly owns one of these references.

### Exclusions and scope corrections

Public carrier organizations, SCACs, corporate registrations, form codes,
legal/tariff constants and real source evidence are not anonymized by this
slice. Individual signatory CPF data belongs to the earlier contact repair,
not to a public-carrier exemption.

The typed-owner audit found false private candidates: carrier form/revision
codes, a legal Act year, clause identifiers, a pre-carriage voyage code and
public carrier registrations. These were excluded **before dataset publication**.
A printed statement ordinal is not a private customer ID. Identical digits
under a postal or quantity owner do not authorize a reference edit there.

A private forwarding agent's FMC/OTI registration is distinct from the fixed
public carrier's registration. The former receives a private generated owner.
There are no matching or separator-normalized collisions between the new
private-code jobs and active source identifier target values.

This is a certificate for the stated source-owned categories, not a claim
that every arbitrary product SKU or unexplained alphanumeric mark in the world
identifies a private person. The original taxonomy remains available for
separately adjudicating such text.

## How values are generated and repeated occurrences are bound

Production generation is deterministic and keyed to the sample and normalized
source identity. The source code is a formatting envelope, not a value to copy.
Every digit/letter class, letter case, separator and physical line break is
preserved. Leading zeros and multiline layouts are supported.

For example, an envelope AB-123/2024 produces another two-letter/five-part
code of the same shape; it does not change the adjacent caption or an
independently printed date. UUID-shaped references retain hexadecimal
character classes and their 32-character stem. No country-specific checksum,
national registry validity or real-world existence is claimed.

Punctuation/case aliases share one normalized identity and one generated stem.
The renderer maps that stem back into each occurrence's own presentation.
It rejects incompatible canonical alias declarations instead of drawing
unrelated identities. A narrowly declared source_value can identify the
code inside a mixed source envelope while preserving all other bytes.

One legacy owner combined an invoice number and discharge locality in Marks.
The fix splits the exact numeric owner from its existing sampled locality
owner. Two stages no longer compete for one binding; neither the locality
dependency nor the numeric identity is lost.

Original source.txt/source-positioned.txt, PDF, source gold, historical
template and alignment files remain evidence. Their private code literals are
now generation-owned in the compiled rendering contract; they are not copied
as immutable synthetic footer/context text.

## Existing-dataset repair and ownership validation

The repair does not perform whole-document digit replacement.

1. Recover each descendant's exact historical source from its saved disjoint
   byte-edit proof, and verify the source SHA-256.
2. Locate the reviewed private source span and replay the original render.
3. Map that precise owner through any later certified line changes. Ambiguous
   occurrence/shape mappings fail rather than select a convenient occurrence.
4. Reconcile aliases within a document, reusing a compatible existing
   anonymized identity where possible.
5. Edit only those proven ranges. Rebuild the corresponding positioned-text
   line with its original suffix; do not recompute or invent coordinates.
6. Independently replay the complete record, compare every other field,
   validate hashes and cross-file mirrors, and inspect every private owner.

The whole-population owner validation checks **41,508 rendered owned
occurrences across 7,500 synthetic documents**. All are anonymized under
their owners; repeated aliases agree. Source/byte proof ownership is checked
separately from the migration's declared receipts. All five JSONL files are
verified, including redundant train/synthetic/replacement mirrors.

A second scan, independent of the edited-range receipts, checks reviewed
private and already-generated source reference identities throughout every
synthetic OCR. It finds **zero retained-source-value candidates** for identities
of at least six normalized characters. Short references are covered by the
exact-owner check, rather than globally treating every matching small number
as a privacy leak.

Falsification probes demonstrate that an invoice number equal to a gross
weight does not authorize changing that gross weight. Corrupt native-source
hashes and historical wrappers submitted to the new direct contract are
rejected. Complete-record replay also detects unrelated label/text changes,
coordinate changes, stale hashes and missing mirror updates.

One duplicate synthesis-authority row lacked an already-certified 7,140 kg
gross-weight fact. Its duplicate target was reconciled to case gold, with
printed OCR evidence, before running the ordinary synthesis preflight. No
real input, real factual label or alignment changed. Its old file and exact
reconciliation receipt are retained.

## Upstream synthesis validation

Every affected template is exercised twice:

- **718 ordinary production sampling/auxiliary preflights** over 359 templates.
- **718 complete numeric-render controls** through render_sampling_blueprint.
- Complete target equality against the matched baseline render and physical
  line-count preservation.

These controls deliberately retain the source's linguistic wording to isolate
the deterministic change. They are inspection controls, not fully anonymous
new lexical descendants, and are not added to training. There is no reason to
pay an LLM to re-generate parties/goods to validate a numeric formatter.
The preflight still runs the real shipment sampler, package/equipment/alias
policies and contact/owner validation before any paid request.

The active mixed-equipment validation config now uses the promoted registry
source authority and current auxiliary contract. Historical campaign artifacts
remain frozen; an old incompatible auxiliary snapshot is rejected, not silently
merged with the current inventory.

## Direct targets, schemas and metrics

New task: **bill_of_lading_extraction_v7_reduced_direct**.

The direct object contains the former documentPatch facts: metadata fields,
parties, route, goods, containers and their relations at the root. There is
no required schemaVersion, no documentPatch key and no artificial parent.
Required negotiability/null and notify sameAs/null policies remain intact.
Reduced/dropped-field restrictions and the frozen category vocabulary remain.

The task has its own model, strict schema and constraint hashes. Dataset readers,
prompt-schema export, metrics and relation constraints use explicit document
access rather than inspecting the output shape and guessing a fallback.
Synthesis publication and position publication now export the direct target.
Native source/candidate/review artifacts retain their historical internal
annotation wire format for replay evidence; they are not model targets.

Logical per-field metric names retain their historical paths for cross-run
comparison. Controlled-error tests verify identical precision, recall, F1,
accuracy, category and relation scores between wrapped and direct contracts.
Order-insensitive matching remains unchanged.

The existing R20 records contain 15,660 target occurrences across the five
mirrored files, representing 8,160 unique real/synthetic documents. Every
occurrence is now direct. Its factual object equals the previous documentPatch
exactly. Historical R19 configs and datasets are not rewritten.

## Measurements, cost and launch boundary

Paid API cost for this pass: **$0.00**, including probes and render controls.
All identifier generation and reconciliation are deterministic.

- Data backfill: approximately 67 seconds for all 7,500 synthetic records.
- Complete row replay: 38.97 seconds in the measured repeat.
- Independent owner sweep: approximately 42 seconds.
- Combined validation including imports: 89.40 seconds, 191,712 KiB peak RSS,
  zero swaps.
- Production sampling plus 718 numeric render controls: 114.27 seconds.
- Median source-format numeric rendering: 5.29 microseconds; alphanumeric:
  33.36 microseconds.
- Canonicalization comparison: wrapped 128.36 vs direct 126.83 microseconds
  per representative target; traced peak 23,476 vs 23,578 bytes. This removes
  the wrapper without a meaningful processing/memory regression.

Exact test results and rebuilt-image config/dataset validation are recorded
in test-validation.json and runtime-validation.json in the audit directory.
The scoped current synthesis/privacy/publication and training contract/data/
metrics suites pass **1,438 tests**. An additional isolated DG realization/
packaging check passes 55 tests; its two modules were run separately after a
working-directory error during optional broad legacy collection.
The remaining broad synthesis suite passes **3,359 tests**, with nine explicit
environment-dependent skips (SDV/SDMetrics and dedicated benchmark/registry
environments). Its warnings are existing seaborn/Matplotlib deprecations;
the training-data suite's warnings concern existing fork/multithreading usage.
The training Docker image was rebuilt so imports use the changed source,
not a stale COPY src layer. No training, inference model or GPU workload
was launched.

The new direct-target config preserves the previous R48 hyperparameters:
[training config](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic5000_private_ids_direct_r20_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml).
It points to the current repaired R20 files and repinned direct constraints.
