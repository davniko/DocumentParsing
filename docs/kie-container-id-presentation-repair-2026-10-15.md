# Container-ID source-format rendering and dataset repair

Date: 2026-10-15. This pass follows the
[phone presentation repair](kie-phone-presentation-repair-2026-10-15.md).
Its base is the published R24 dataset, not the older pre-phone-repair inputs.

## Contract and diagnosed failure

Container targets remain canonical: uppercase letters/digits including the ISO
6346 check digit. The same identifiers in goods placements must stay unchanged.
Only OCR presentation changes, using the independently owned source occurrence's
format. No new container identity, shipment quantity, seal, equipment category,
phone, date, goods wording or party value is sampled in this repair.

The existing [presentation audit](kie-real-synthetic-format-audit-2026-10-15.md)
identified 52 templates with 76 formatted container fields in 101 source
occurrences. In the 7,000-row synthetic shard these correspond to 864 documents
and 1,457 target fields. For example:

```text
Source OCR:          MCLU 507449-8
Canonical source:    MCLU5074498
Old synthetic OCR:   MCLU5688412
Required OCR style:  MCLU 568841-2
Unchanged target:    MCLU5688412
```

The source formats remain in source text and historical byte-owned bindings.
They disappear because ordinary curated-variable replacement reflows a compact
canonical ID as one word. Explicit historical surface overrides supply a
canonical ID too, bypassing the old partial separator-preserving branch. Both
paths need the same source-conditioned identifier presentation operation.

This is distinct from character correction: the reviewed `ECDU`/`ESDU` OCR
alias is explicitly declared and must retain its established behavior. Separator
removal cannot justify inventing a letter or digit.

## Scope and preservation requirements

- Repair all 8,000 unique retained synthetics: the current 7,000 plus the complete
  older-UAE sidecar, including its 1,000 additional non-overlapping records.
- Keep the 1,500-row selected-addition mirror consistent with the repaired pool.
- Preserve all 153 sidecar quarantine exclusions; formatting is not readmission.
- Preserve every target exactly, including container identifiers, placements and
  the phone fixes. Real 600 and validation 60 are byte-identical R24 copies.
- Edit only verified container-ID pieces in actual current owners. Do not
  reconstruct entire source regions or restore old words, suffixes or identities.
- Preserve physical lines and their positional suffixes. All 101 previously
  found noncanonical source occurrences are single-line; any contrary case must
  be investigated rather than silently moved.
- Publish a new versioned dataset; protected R24, R23 and earlier files remain
  unchanged. Do not launch training or paid generation.

## Baseline measurement

Before implementation, seven timed batches of eight native cached-receipt
Egypt-single100 renderings, including independent validation after warmup, have
median **0.141179 s per batch** and **451,760 KiB** peak process RSS. No API calls,
model loading, GPU work or campaign writes. The
[baseline receipt](analysis/container-id-presentation-repair-20261015/native-baseline-benchmark.json)
preserves all batch times for the after-change comparison.

## Implementation and upstream validation

| Component | Change and rationale |
| --- | --- |
| `src/document_ocr/synthesis/curated_container_ids.py` | One source-mask formatter and typed render plan for current container and placement ownership. Eleven verified identifier characters are replaced; the source separators remain. Pure owners and already-rendered composite owners use the same contract. |
| `src/document_ocr/synthesis/curated_templates.py` | Automatically prepares and applies that plan in native rendering, before byte-edit/proof publication. Explicit historical surface overrides no longer bypass container formatting. Native validation independently regenerates the resulting text and receipt. |
| Current contract ownership | Use effective current expressions rather than stale historical array indices. An ID legitimately used inside a handling-instruction expression remains supported; formatting must not rewrite that expression's target. |
| Dataset migration | Derive current spans through hash-pinned native inverse/forward proofs, project their verified source frames, then independently restrict edits to complete ID subspans. Reject unowned suffix restoration, overlapping/ambiguous owners, identity changes and new physical lines. |
| Policy and registry metadata | Record canonical-target versus source-input presentation explicitly. No case assets, ownership bindings, auxiliary rules or target schema needed rebasing for this repair. |

The production [validation receipt](analysis/container-id-presentation-repair-20261015/production-validation.json)
records **413/413 native render/replay passes**, including 405 container-bearing
templates and eight no-container controls. Its 716 container fields and 716
placement links cover 809 owned source occurrences: **808 reversible masks and
one explicitly reviewed character alias**. All 101 noncanonical source
presentations pass. A separate source-only booked-container occurrence already
uses canonical formatting and independently generated identities; it remains an
unchanged control, not an invented target.

The 11 actual source masks are:

```text
AAAA9999999       AAAA 9999999       AAAA 999999-9
AAAA 999999.9     AAAA 999999/9      AAAA-999999-9
AAAA-999999/9     AAAA/999999/9      AAAA999999-9
AAAA 999 999-9    AAAA999 999-9
```

The final broad regression run passes **1,794 tests in 84.58 s**; the independent
migration/falsification suite passes **18 tests in 15.38 s**. Ruff and
`git diff --check` pass. The broad command includes curated synthesis, phone
presentations, contact values and the previous phone migration tests. The new
migration tests run separately because both audit folders intentionally contain
a `test_repair.py` module.

The complete production receipt contains exact source-probe reproduction
commands, all 413 per-source hashes, focused test evidence and benchmark arrays.
No model/API call, GPU work or training is required for these checks.

## Performance and memory

Profiling identified redundant full-target flattening and repeat formatting of
an already validated, plan-produced owner. ID extraction now walks only the
concrete container/placement arrays; exact plan-produced text is not parsed a
second time. This removes measured work without weakening mask validation.

The identical native cached-render command after the final changes measures
**0.127015 s/eight documents**, versus the recorded **0.141179 s** baseline;
peak RSS is **447,332 versus 451,760 KiB**. These short batches vary, so this is
not a claim that adding presentation validation itself creates a 10% speedup.
An alternating same-process feature-on/off counterfactual isolates a median
paired cost of **0.379 ms/document**, including independent native validation.
The pure formatter performs 80,800 validated calls in **0.323531 s**, or
**4.00 microseconds/call**. No new API calls or blocking I/O enter the render path.

See the [final native benchmark](analysis/container-id-presentation-repair-20261015/native-final-benchmark.json)
and [paired benchmark/profiler evidence](analysis/container-id-presentation-repair-20261015/production-validation.json).

## Completed dataset publication

Published [R25 dataset](../data/curated/mpci-bl-real600-synthetic7000-container-id-presentations-r25-positions-v1/README.md),
preserving the R24 phone corrections and all earlier retained labels. Its
[manifest](../data/curated/mpci-bl-real600-synthetic7000-container-id-presentations-r25-positions-v1/manifest.json)
SHA-256 is `d1c9010ff14ed1a13db1167a7c694ccfa10d0b97eac6a34d30dd4c74f158224c`.

| Published shard | Documents | Inputs changed | Targets changed |
| --- | ---: | ---: | ---: |
| Current synthetic training | 7,000 | 864 | 0 |
| Complete previous-UAE sidecar | 2,500 | 302 | 0 |
| Selected previous-UAE addition mirror | 1,500 | 149 | 0 |
| Real training | 600 | 0 | 0 |
| Validation | 60 | 0 | 0 |

Overlapping sidecar/addition records are not new examples: the repair covers
**8,000 unique synthetic documents**, with **2,801 ID occurrence edits across
1,017 unique documents**. All 19,236 inspected owner occurrences were compact
before repair. Afterwards 16,435 remain compact and 2,801 recover their source
format, covering all 11 source masks. This includes 25 unchanged source-only
booked-container controls; no target is invented for them.

Verified examples from the actual repair receipts:

| Synthetic sample | Before | After | Canonical target |
| --- | --- | --- | --- |
| `2ca7b8a04845376412e3faf7` | `MCLU5688412` | `MCLU 568841-2` | `MCLU5688412` |
| `ca0e1c62e90391ac8574370d` | `FSCU8313192` | `FSCU/831319/2` | `FSCU8313192` |
| `84c46447833e8df97f2c07fe` | `MCLU4804694` | `MCLU 480469.4` | `MCLU4804694` |
| `fce266658ff66e011d54762f`, first occurrence | `OOCU0501791` | `OOCU 050 179-1` | `OOCU0501791` |
| Same document, repeated occurrence | `OOCU0501791` | `OOCU050 179-1` | Same identity |

These are independently owned occurrences, not global substitutions. In
particular, a repeated ID may correctly have two different printed formats.

No coordinate synthesis/reflow is needed: the repair adds only in-line source
separators, preserving every physical row and its exact ` ||` coordinate suffix.
The raw and positioned texts and their hashes are updated together. Every
non-identifier byte, target, placement, seal, type, phone, date and goods value
is preserved. All 153 previously quarantined old-UAE records retain their exact
quarantine reasons and remain excluded from training.

The complete [8,000-record repair plan](analysis/container-id-presentation-repair-20261015/full-v1.json)
passes without errors in **139.60 s**, with **819,624 KiB** peak RSS. The separate
manifest-last publication/revalidation takes **77.96 s**. Its
[publication receipt](analysis/container-id-presentation-repair-20261015/publication.json)
binds the implementation, current registry, protected source files and the
[per-record span proofs](analysis/container-id-presentation-repair-20261015/full-v1-receipts.json).
Historical R24/R23 data and the currently running training inputs/config remain
untouched; no training was launched.

## Independent final certification

The separate [readback and training-loader check](analysis/container-id-presentation-repair-20261015/independent-readback.json)
passes in **29.30 s**, with **819,624 KiB** peak RSS. It independently reconstructs
the ID-only edits and source masks, compares every target, verifies every
physical-line coordinate suffix, checks all file/protected-input hashes and
checks overlapping mirrors and quarantine membership. It does not invoke the
migration's edit/replay functions. All 8,000 unique synthetic records pass;
real/validation bytes and all non-ID text remain exact.

As a concrete coverage check, the current synthetic training shard previously
had **0/14,472 container IDs** without a contiguous raw-text copy. It now has
**1,297/14,472 (8.96%)** whose extraction requires joining source-format pieces.
The unchanged real set has **132/1,251 (10.55%)**, and validation has
**11/120 (9.17%)**. Repeated aliases explain why this count is smaller than the
number of source-formatted occurrence edits: a formatted occurrence may also
have a compact repeat elsewhere in the document.

The real training startup's `load_training_task`, `load_prompt` and
`inspect_dataset` paths accept R25 as **7,600 training + 60 validation records**
with the direct reduced-V7 target and unchanged input-only prompt. This was an
in-memory config binding for inspection, not a saved config modification or a
training launch. No tokenizer/model download or GPU was used.

**Status: container-ID rendering and the complete retained synthetic repair are
finished and validated.** Future native synthesis automatically uses the source
masks. The corrected R25 inputs are ready for the next training configuration;
the current running job remains pinned to its original data.
