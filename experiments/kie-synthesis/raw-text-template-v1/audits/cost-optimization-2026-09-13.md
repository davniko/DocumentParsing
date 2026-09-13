# Template descendant and compiler cost audit — 2026-09-13

## Decision

The recurring descendant-render cost problem is resolved on the pinned 30-document cohort. The
one-time compilation cost is materially lower than the original lineage, but is not yet low or
stable enough to authorize the full carrier-bound corpus.

- Descendant rendering fell from **$0.12604420 / 30** to **$0.00024828 / 30**: a **99.8030%**
  reduction (**507.67x**), while exact validation improved from 1/30 to 30/30.
- On the same five compilation documents, the original lineage cost **$0.72328422 / 40
  requests**. The latest fully certified concurrent arm cost **$0.39620235 / 24 requests**:
  **45.22%** less cost and **40.00%** fewer requests.
- That compilation rate is still **$0.07924047/document**, or **$15.848094/200** and
  **$130.746776/1,650 eligible documents**. This is a bounded five-document projection, not an
  authorization to scale.
- The newest exact-reference canary was rejected after six calls and **$0.16195356**. It was not
  expanded to five documents.

All projections exclude target-label synthesis. Compilation is one-time per source template;
descendant rendering is recurring for every synthetic document.

## Why deterministic slots initially saved so little

The former descendant runner invoked one residual model request for every document, even though
the template contract contained mostly deterministic slots. A deterministic slot reduced only the
amount of text edited inside that already-mandatory request; it did not remove the request's fixed
prompt, schema, and reasoning cost. The earlier comparison therefore measured 30 still-mandatory
calls, not a deterministic execution path.

The typed descendant renderer now handles identifiers, parties, dates, measurements, package
categories, equipment, repeated values, derived values, and exact formatting locally. It batches
only the residual surfaces that actually require linguistic work. In this cohort that leaves one
residual slot in one document, so 29 documents make no provider request and the remaining residuals
use one request in total.

| Metric | Old residual run | Typed residual run | Change |
|---|---:|---:|---:|
| Documents passing all gates | 1 / 30 | 30 / 30 | +29 |
| Deterministic slots | 2,090 / 2,595 | 2,594 / 2,595 | 99.9615% deterministic |
| Provider requests | 30 | 1 | -96.67% |
| Input tokens | 126,574 | 1,347 | -98.94% |
| Output tokens | 78,671 | 184 | -99.77% |
| Cost | $0.12604420 | $0.00024828 | -99.8030% |
| Wall time | 160.218 s | 19.542 s | -87.80% |

Evidence:

- `artifacts/kie-synthesis/mpci-bl-carrier-bound-template-descendant30-v1-luna-high`
- `artifacts/kie-synthesis/mpci-bl-carrier-bound-template-descendant30-v10-typed-residual-offline-replay`
- `artifacts/kie-synthesis/mpci-bl-carrier-bound-template-descendant30-v10-typed-residual-luna-high`

At the measured recurring rate, 100 descendants cost **$0.0008276**, 1,000 cost **$0.008276**,
10,000 cost **$0.08276**, and 50,000 cost **$0.41380**. These are straight-line projections of
the pinned cohort's residual incidence; a larger validation cohort is still required before using
them as a budget commitment.

## Original compilation cost anatomy

The 230-document lineage cost **$33.40492748**, or **$0.14523882 per selected document**, and
certified 229/230. Its transfer-200 baseline alone cost **$25.84332216**.

- Host-rejected stages cost **$13.11364013**, or **50.74%** of the transfer-200 bill.
- Structured-output retry responses cost another **$1.85275953** across 155 retry responses.
- The full lineage used 2,027 requests, or **8.81 requests/document**.
- Spend is tail-heavy: the most expensive document cost **$2.48310584**; the top 20 documents
  contributed **34.01%**, and the top 50 contributed **54.07%**, of all 230-document spend.
- Median cost was **$0.09378456**, p90 **$0.28517644**, and p95 **$0.37475729**.

This proves that the main compiler problem is repeated failed semantic transactions and repeated
whole-document audits, not merely model pricing or JSON size.

## Paired compiler experiment

The five probe documents were all certified in both the original lineage and each accepted fresh
control. They include the expensive OOCL case and four materially different carriers.

| Carrier | Original cost / requests | Fresh v11 cost / requests | Concurrent v20 cost / requests |
|---|---:|---:|---:|
| OOCL | $0.30528717 / 11 | $0.27306059 / 14 | $0.21057863 / 10 |
| G.C.S. Container Line | $0.02360163 / 3 | $0.01905165 / 2 | $0.02318353 / 2 |
| Hapag-Lloyd | $0.15635337 / 11 | $0.02584815 / 2 | $0.04138628 / 3 |
| E Global Shipping Line | $0.15936426 / 9 | $0.04826525 / 3 | $0.08163577 / 5 |
| Tepmare | $0.07867779 / 6 | $0.02315975 / 2 | $0.03941814 / 4 |
| **Total** | **$0.72328422 / 40** | **$0.38938539 / 23** | **$0.39620235 / 24** |

The v11 control is 46.16% cheaper than the same-document original lineage. The v20 arm is 45.22%
cheaper and completes in 609.343 seconds versus v11's 1,003.188 seconds because five documents run
concurrently. Concurrency therefore cut cohort wall time by **39.26%**, but—as expected—did not
itself lower token spend.

Evidence:

- `artifacts/kie-synthesis/mpci-bl-template-compiler-efficiency-probe5-v11-compact-repair-fresh-luna-high`
- `artifacts/kie-synthesis/mpci-bl-template-compiler-efficiency-probe5-v20-occurrence-removal-luna-high`

The conservative v20 rate projects to:

| Scope | Projected one-time cost |
|---|---:|
| 30 documents | $2.377214 |
| 100 documents | $7.924047 |
| 200 documents | $15.848094 |
| 1,421 not-yet-compiled eligible documents | $112.600708 |
| all 1,650 carrier-bound eligible documents | $130.746776 |

The 2,174-document corpus projection is intentionally omitted as an operational estimate because
524 sources still require carrier resolution and cannot enter the carrier-bound compiler as-is.

## Accepted and rejected optimization findings

### Accepted

- The discriminated compiler/critic schemas structurally exclude invalid render-mode combinations.
- The critic receives one annotated source instead of duplicated annotated and masked source views.
- Host-derived completion removes a provider-controlled boolean that caused otherwise-complete
  compiler outputs to retry.
- Exact occurrence append/removal transactions avoid restating whole bindings.
- Deterministic host normalization now covers typed temperature surfaces, safe composite splits,
  repeated multiline target projections, and canonical identifier policy.
- Exact-span complementary package/allocation quantity owners are now joined before overlap
  validation. Replaying V21's second compiler candidate proves the seven overlap defects disappear;
  the genuinely contradictory semantic-only declaration remains rejected.

### Rejected as general cost solutions

- **Provider/model substitution:** GLM reduced one compiler call's price, but increased downstream
  calls/residuals and made the paired medium case 12.04% more expensive. This is not a
  quality-preserving general optimization.
- **Annotated-source-only v18:** certified 5/5 but cost $0.47989747, 23.24% above v11.
- **Occurrence-removal v20 as an incremental cost arm:** certified 5/5 and was much faster, but cost
  1.75% above v11. Its value is host correctness and lower latency, not a paid-cost win over the
  current control.
- **Reference plus delta repair v21:** rejected after three compiler and three critic calls,
  $0.16195356, and 670.974 seconds. The final critic repeatedly mapped an HS-code target to a cargo
  description outside its cited finding. The host correctly rejected it, so no five-document v22
  was launched.
- **Lossless table compaction alone:** on V21, generic columnar encoding reduced provider-payload
  bytes by 17.88% on the initial compiler request and 22.06–22.28% on repairs. Output/reasoning and
  repeated audits dominate the bill, so this can save only a few percent and is not the required
  step-change.

## Next provider-neutral compiler experiment

The next material arm should stop recompiling unchanged text. It should be a **carrier-bound
certified-template delta compiler**, not another prompt or provider tweak:

1. Select a previously certified seed only from the same carrier and template-proxy family.
2. Align the new OCR against the seed deterministically. Reuse a prior binding only when its exact
   literal/span neighborhood and semantic contract match; hashes make inherited regions explicit.
3. Build ordinary target anchors locally for the new source. Emit a bounded review ledger only for
   inserted, deleted, moved, ambiguous, or target-shape-changed regions.
4. Give the full-capability compiler that delta ledger plus local source windows and exact edit
   handles. It returns typed local transactions, not a whole-document binding inventory.
5. Re-run all host round-trip, topology, realization, carrier, risk, and sentinel gates over the
   complete materialized template. The agent independently audits only novel regions; already
   certified byte-identical regions retain their seed certification and provenance.
6. Escalate an alignment that cannot be proven safe to the current full compiler explicitly. It is
   a visible route, not a silent fallback.

This direction is supported by corpus structure but does not yet have a paid savings claim:

- 1,656 sources have a carrier; the official eligible set is 1,650.
- Exact carrier plus template-proxy grouping yields 806 groups, so 850 documents have a potential
  same-family predecessor.
- 58.32% of lines recur in at least two documents of the same carrier; 38.85% recur in at least
  five documents and at least 10% of that carrier's documents.
- Exact full-template reuse is not safe: including target-leaf shape raises the group count to
  1,533, and route/cargo lines can recur while representing mutable data. The delta must therefore
  be host-proven at span/contract level.

The first paid gate should use 30 documents from at least ten carrier/template families, with one
seed and two chronological held-out documents per family. Run held-out documents concurrently.
Promotion requires all 30 to certify, zero regression in deterministic/agent-assisted realization,
manual inspection of every changed-region class, and at least a 50% paired cost reduction versus
the current full compiler on the same held-out documents. Only then should it expand to 100.

## Validation and safety state

- Targeted exact-span co-binding tests: 3 passed.
- Full experiment suite: 144 passed.
- Ruff format and lint: passed.
- Real OOCL overlap-normalization microbenchmark (181 drafts, 3,000 iterations): 0.071 ms/call
  before and 0.356 ms/call after, an absolute +0.285 ms outside the model hot path.
- All checked-in provider launch gates are false.
- No 100-document or full-corpus compilation was launched.
