# MPCI bill-of-lading production template base

This directory contains the launch configuration for the production carrier-bound
template base. It is intentionally separate from the development and experiment
artifacts under `artifacts/kie-synthesis`.

## Prepared split runs

The remaining production compilation is split into two disjoint, balanced cohorts:

- Half A config: `mpci_bl_template_compilation_remaining710_a_v1_luna.yaml`
- Half B config: `mpci_bl_template_compilation_remaining710_b_v1_luna.yaml`
- Output parent: `artifacts/kie-synthesis-production/template-base/remaining1420-split`
- Each half contains exactly 710 explicitly pinned documents.
- Each half uses 16-document and 16-request concurrency.
- Provider launch is locked by default: `provider_launch_authorized: false`.
- Half A resumes a committed, exact 150-document prefix recovered from its interrupted
  production transaction. Half B was consumed through disjoint, explicitly pinned budget shards;
  every Half B ordinal has now been processed and every launch config is provider-locked.

The halves alternate odd and even ordinals from the frozen 1,420-document selection.
This distributes early coverage-oriented and difficult documents across both runs instead of
concentrating them in Half A.

The earlier 1,420-document staging transaction remains preserved but is superseded by this
split plan. Its completed cases are not copied into either half because an uncommitted staging
transaction is not a supported provenance source.

## Half A production result

Half A committed successfully at:

`artifacts/kie-synthesis-production/template-base/remaining1420-split/mpci-bl-template-compilation-remaining710-a-v1-luna-high`

Its immutable provenance is:

- Commit-file SHA-256:
  `2248783f564ff5000601e8cfa9a621f5a93e3f73d4e6d324c62dd86dc5fd4b2d`
- Transaction SHA-256:
  `4382e0f36e97678b6ba429e9e2d9f8885268a8e4fa33a62b314576816e46bd77`
- Content SHA-256:
  `975e5e37f91db6c50994c9f5b32313b6a8ba273163e7629189338c272785337d`
- `results.jsonl` SHA-256:
  `888e304dbefdf1c6d1eef77714a0da8b4973eaac1dc7ddfd86dfa6179e557886`
- `catalog.jsonl` SHA-256:
  `f84dfbd08414177a390b7f4973fa0b9eba013893e0e6458edb47f1e01e0b7fd2`

The run produced 710 explicit outcomes: 645 certified, 22 review-required, and 43
rejected. The production catalog contains exactly the 645 certified templates. The run's
no-rejection acceptance gate is therefore **false**: the commit is a durable audit and resume
source, not a claim that Half A is acceptance-complete. Its marginal provider cost was
`$32.749163040000`; its cumulative lineage cost, including the pinned 150-document recovery
prefix, was `$40.172558940000`. Both split configs are provider-locked after the run.

Half A's first corrected transaction later stopped on a second host orchestration defect after
completing an exact contiguous prefix of 150 documents. That prefix was sealed without provider
calls into the committed recovery run below; all case payloads were copied byte-for-byte, and the
original host contract remains pinned so the current host must revalidate it on resume:

- Recovery run:
  `artifacts/kie-synthesis-production/template-base/remaining1420-split/recovery/mpci-bl-template-compilation-remaining710-a-prefix150-recovery-v1`
- Recovery documents: 150 (140 certified, 9 rejected, 1 review-required).
- Commit-file SHA-256:
  `08a6ce45d59d4a2f82e714081a2ee118b535ff364f782e01741b9bcc5a57e22d`
- Provider calls made while sealing the recovery run: 0.

## Half B budget-shard progress

Half B ordinals 1-180 were compiled first at:

`artifacts/kie-synthesis-production/template-base/remaining1420-split/budgeted-batches/mpci-bl-template-compilation-remaining710-b-prefix180-v1-luna-high`

That immutable shard contains 167 certified, 4 review-required, and 9 rejected outcomes at a
receipt-backed provider cost of `$9.616039320000`. Its commit-file SHA-256 is
`3c2b4020938e7c5c7661f66ffbf79e75199e6421d2bd06d0f8d0dc501d6ba51a`, and its transaction
SHA-256 is `dd7b949d7a79c24df75459ea7f0beea11612e7b66bb4f1794c5e6c7edca178a4`. The four review
outcomes were closed by the pinned manual-resolution overlay used by the production catalog.

Half B ordinals 181-520 were compiled as one exact 340-document selection targeting about
`$20` of durable provider usage. A host-side draft/source alignment defect interrupted the first
transaction after 211 complete outcomes. Those exact outcomes were sealed without provider calls,
the alignment invariant was hardened at compiler, critic, inventory, and checkpoint boundaries,
and only the 129 missing documents were launched under the corrected host contract.

The two committed shards are:

- Recovered 211-document shard:
  `artifacts/kie-synthesis-production/template-base/remaining1420-split/recovery/mpci-bl-template-compilation-b-ordinals181-520-completed211-recovery-v1`
  - Outcomes: 193 certified, 5 review-required, 13 rejected.
  - Durable provider cost: `$12.176715890000`.
  - Commit-file SHA-256:
    `abaec60f4b6ae11a088fabb46e0d1d84978943778eabe5522ddb10ee21931f5f`.
  - Transaction SHA-256:
    `7cb71baa932c306f6787ddc373ad8bd40ef05bd50a0c7218b9380fad96f5c1a2`.
- Corrected 129-document continuation:
  `artifacts/kie-synthesis-production/template-base/remaining1420-split/budgeted-batches/mpci-bl-template-compilation-b-ordinals181-520-missing129-v1-luna-high`
  - Outcomes: 113 certified, 7 review-required, 9 rejected.
  - Durable provider cost: `$7.463006550000`.
  - Commit-file SHA-256:
    `bc917f88e58ae9c1db3b19094ad82302cdbb8942cbf6781c1e14b3cad29029ee`.
  - Transaction SHA-256:
    `295ee3bd3d1027025b77128df31785c0d7479d36ce37bd72c67febf7027da930`.

Combined, the shards cover the intended 340 documents exactly once and contain 306 certified,
12 review-required, and 22 rejected outcomes. Their receipt-backed provider cost is
`$19.639722440000`. The interrupted transaction may also have incurred a small amount of
provider-side usage for canceled in-flight requests whose usage receipts were never returned;
that unknown exposure is deliberately excluded from the durable ledger rather than estimated.
All 306 certified templates were strict-schema parsed and exact-source round-trip rendered during
post-run validation. Both launch configs are now provider-locked.

All 12 review-required outcomes were then inspected against their complete source text, structured
label, and compiler checkpoint. Each is a genuine incomplete-container-topology case: the source
declares or prints more containers than the target can represent. Promoting one would require
inventing identities or leaving shipment-dependent surfaces inconsistent. The two immutable manual
resolution overlays therefore exclude all 12, leave zero unresolved reviews, and make no provider
calls. The 22 compiler rejections were also reviewed; none contains an independently confirmed
template that can be safely promoted by manual adjudication. They remain explicit exclusions and
can only be salvaged by a fresh compiler remediation run.

## Current production catalog

All six adjudicated shards, including the final 190-document tail, are merged into:

`artifacts/kie-synthesis-production/template-base/catalogs/mpci-bl-production-template-catalog1510-v5`

The catalog is a closed, disjoint partition of all 1,650 carrier-eligible source outcomes:

- Usable current-schema templates: 1,510.
- Explicit exclusions: 140 (85 compiler rejections and 55 manual review exclusions).
- Unresolved reviews or duplicate document IDs: 0.
- Provider requests and incremental provider cost for adjudication and merge: 0.
- Commit-file SHA-256:
  `07c7ae3e1d87ac8b1de6e766a4013c748fad12ad493be5ab5f252229ac4d0b44`.
- Transaction SHA-256:
  `5441ee47baef025e9159df336e78e4ebb9424f8913aceecaf6fa373530bedfa7`.
- Catalog SHA-256:
  `46353767e830467842da95a5f72b8ab272449a56f6f63f2954ed9d92472d84bb`.

An independent population audit re-hashed all 7,557 committed artifacts, strict-schema parsed all
1,510 templates, reproduced every pinned source byte-for-byte, and found exact catalog/case/source
agreement. The catalog contains 100,959 bindings, of which 97,720 (96.7918%) are compiler-classified
deterministic and 3,239 are agent-assisted; only 2,008 are residual-renderer bindings. This catalog
is the current production template base. The predecessor 1,344-template v4 catalog remains immutable
for provenance but is superseded for new work.

## Final Half B shard result

Half B ordinals 521-710 were processed as the exact frozen 190-document tail at:

`artifacts/kie-synthesis-production/template-base/remaining1420-split/budgeted-batches/mpci-bl-template-compilation-b-ordinals521-710-v1-luna-high`

The committed run contains 166 certified, 13 review-required, and 11 rejected outcomes. Its
receipt-backed provider cost is `$10.806850700000` across 794 requests. It contains 11,082
bindings, of which 10,846 (97.8704%) are deterministic and 236 are agent-assisted. Immutable
provenance is:

- Ordered 190-document ID SHA-256:
  `6ae6c10ba4712c34ad906a15c03d5f3ce2a6c234a64f5b09b1d824c3fc7e9ef1`.
- Commit-file SHA-256:
  `229825d91868d99ec2527d74b2ca94cbf3fd50c61b10a908840abf19fb152e13`.
- Transaction SHA-256:
  `bf3eb4218d9ef70f22eaef717e60ecdf653f46158883de6057174f7d4b5f58c6`.
- `results.jsonl` SHA-256:
  `7ed09fdae8edf61e9957544ab0229b83f8b28aafd645888f9ceff78921c2e080`.
- `catalog.jsonl` SHA-256:
  `1c49fc1de506e4aad572b1f5e9f8dfd238401ef9a772d68a1431d3ed51111229`.

The provider run took 4,369.886 seconds with a measured peak RSS of 887.258 MiB and no swap. A
separate post-run audit re-hashed all 1,452 committed artifacts, strict-schema parsed all 166
certified templates, reproduced all 166 pinned sources byte-for-byte, and confirmed exact ordered
coverage with no duplicate or omitted document. All 13 review outcomes are fail-closed source/label
container-count contradictions. The 11 rejected outcomes are compiler or critic workflow failures;
no rejected template entered the catalog. The run config is provider-locked after completion.

The three Half B shards (180 + 340 + 190) cover all 710 Half B documents exactly once. The 13 final
review outcomes were manually inspected and excluded as genuine incomplete-container-topology
cases; the 11 workflow rejections remain explicit compiler exclusions. Their zero-cost immutable
resolution overlay is:

`artifacts/kie-synthesis-production/template-base/remaining1420-split/manual-review-resolutions/mpci-bl-template-compilation-b-ordinals521-710-manual-resolved-v1`

It closes all 190 outcomes as 166 usable and 24 excluded, with zero unresolved reviews. Its
commit-file SHA-256 is
`6667ddb5a5349cd73ef75ead284ab822e22ec2753069173ff43869dcbb962938`, and its transaction
SHA-256 is `86f82c5afb04150c897db952ca8cfc90d1903039a0cab3611889b9830383d8af`.

## Exact selection contract

The pinned corpus has 2,174 source documents. The carrier-bound compiler classifies
1,650 as eligible and 524 as requiring external carrier enrichment. The existing
230 outcomes are all in the eligible population, so this run selects the exact
remaining 1,420 eligible documents:

```text
1,650 eligible - 230 existing outcomes = 1,420 configured outcomes
```

The config embeds the 230 exclusions in sorted order and requests every remaining
eligible document. Consequently, selection ranking cannot omit or substitute a
remaining eligible document.

The split configs explicitly pin that same complement. Their identity contracts are:

- Half A ordered-ID SHA-256:
  `ca99f213a1d7405e38a93c1334ebb336cb85da776962e8799a7a53af1a8f739f`
- Half B ordered-ID SHA-256:
  `3e3596781251ee0c4ce252c5f65098fbff37fccbe558b5045a2196cceffdf31d`
- Intersection: 0 documents.
- Sorted union SHA-256:
  `3b2c03f2b372ecd91e45d4398a42f083fbb3d35c58a912c518791dc829cf027a`

Existing-catalog provenance:

- Path:
  `artifacts/kie-synthesis/mpci-bl-compiled-template-catalog230-semantic-v4`
- Commit-file SHA-256:
  `61c5e1ced535352c4395ca6d4eb6bc9d5e2a05ab7f211d6ca1a7d24a2e415e8e`
- Transaction SHA-256:
  `e92bd190a5ad449b98a95757a05f3b60afecd21dd2f0e6cbab74b49184e0649a`
- `outcomes.jsonl` SHA-256:
  `187a4a42bd6663f9b4e2f90759c12a924dc6672555f4ed95d8a8c004ba2e3657`
- Existing outcomes: 230 (226 certified, 4 review-required)
- Canonical SHA-256 of the sorted 230 document-ID array:
  `d85c0a60b127efece0974ccffc897357a43d0ab193b30a7f9c25a71dc10ba4a4`
- Canonical SHA-256 of the sorted 1,420-document complement:
  `3b2c03f2b372ecd91e45d4398a42f083fbb3d35c58a912c518791dc829cf027a`

The 524 carrier-ineligible documents are not silently discarded: they remain in
the source corpus, but the current carrier-bound contract cannot compile them
safely without a trustworthy carrier identity. They are outside this production
template line unless a separately validated carrier-enrichment process is added.

## Subsequent remediation rule

Do not re-authorize or rerun the committed Half A run name. Any remediation of its 43
rejected outcomes must use a new run name and pin the committed Half A receipt as its resume
source, after the repair-path defects have been addressed and validated.

Do not re-authorize any completed production config. Their committed shards are the authoritative
outcome and recovery sources. Remediation of a rejected outcome must use a new run name, an exact
pinned selection, and immutable source-run provenance.

The current catalog already closes all 1,650 carrier-eligible source outcomes. Any future attempt to
salvage one of the 85 compiler rejections must use a new run name, an exact pinned selection, and
immutable source-run provenance. A successor catalog must preserve the same closed 1,650-document
partition; rejected or unresolved documents must never be silently omitted.

## Prepared 10,000-document synthesis plans

Two provider-free, immutable synthesis plans are prepared from the final 1,510-template catalog.
Both target only `bill_of_lading_relation_explicit_v5` / `5.0.0-experimental`, assign 10,000
unique sample identities, and deliberately reuse carrier-bound templates. There is no one-sample-
per-template ceiling in the planned renderer contract.

The primary plan excludes only the 100 exact validation document IDs:

- Config: `mpci_bl_production_synthesis10000_exact_validation_id_v1.yaml`.
- Artifact:
  `artifacts/kie-synthesis-production/synthesis-plans/mpci-bl-production-synthesis10000-exact-validation-id-v1`.
- Eligible and selected templates: 1,441 / 1,441.
- Exact validation-source templates excluded: 66; three additional source templates are
  fail-closed because their legacy labels cannot be represented faithfully in v5.
- Exact validation IDs retained: 0.
- Shared layout proxies are intentionally allowed: 456 non-validation sibling templates across
  26 validation-associated proxy families remain eligible.
- Commit-file SHA-256:
  `352242474dc9aa2d5cadd4ea074fb5758fe56e80d3a806c9c5e03af1a820f1e3`.
- Transaction SHA-256:
  `ac540ff2b7cdd7a1202f227d4f5f10eb3d7700b3d20f4121db418d9b481bf7c1`.
- `plan.jsonl` SHA-256:
  `36704dcdb497ce17afb190a0e9bb23e73269ce99e0a6f3dc25b6958eb18a1a75`.

The secondary transfer plan excludes every catalog template in any proxy family represented by a
validation document:

- Config: `mpci_bl_production_synthesis10000_layout_proxy_holdout_v1.yaml`.
- Artifact:
  `artifacts/kie-synthesis-production/synthesis-plans/mpci-bl-production-synthesis10000-layout-proxy-holdout-v1`.
- Eligible and selected templates: 985 / 985.
- Validation-family templates excluded: 523; the union is 525 after the same three v5
  incompatibilities (one incompatibility is already in the validation-family exclusion).
- Exact validation IDs and validation-associated proxy families retained: 0 / 0.
- Commit-file SHA-256:
  `583e824632b6e3d5fa2bcad477049903884cbfdf7bc4bfa26dc90050e2c9da4f`.
- Transaction SHA-256:
  `2bbaa22af796520b324aaa17667ce24560c000a085301ec58c2d5de13d39b666`.
- `plan.jsonl` SHA-256:
  `d2eaab33c67404afcd7c119ddeedc921a01e2722edef1d3e8bfe53ce657d5401`.

Each plan contains exactly 7,500 standard, 1,000 dangerous-goods, and 1,500
temperature-controlled samples. Reuse is balanced within each capability cohort with deterministic
HMAC-ranked allocation, so counts differ by at most one among templates in the same cohort. The
primary plan is the intended first real-plus-10,000 training path. The proxy-family plan is a
separate later transfer experiment, not an automatic gate or staged promotion step.

The corresponding launch configurations are
`mpci_bl_production_compiled_synthesis10000_exact_validation_id_v1.yaml` and
`mpci_bl_production_compiled_synthesis10000_layout_proxy_holdout_v1.yaml`. Both are direct
10,000-document runs with 16-request concurrency, no staged schedule, and no automatic promotion
gate. They were initially provider-locked and were subsequently launched only after explicit
authorization.

### Initial measured renderer preflight

Both exact launch configurations passed a complete 10,000-sample provider-free preflight. The
preflight constructs every latest-schema target, resolves every runtime route, materializes every
case that needs no provider, and checks carrier binding and exact target topology. It never enters
the provider path.

| Path | Provider-free documents | Planned one-call documents | Deterministic slots | Agent slots | Compatibility adaptations | Wall time | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact validation ID | 6,077 | 3,923 | 980,142 / 990,702 (98.934%) | 10,560 | 1,404 | 161.87 s | 2,461,676 KiB |
| Validation proxy family | 6,098 | 3,902 | 901,311 / 910,351 (99.007%) | 9,040 | 1,480 | 142.69 s | 1,973,328 KiB |

The live provider executor uses a bounded worker pool of at most 16 workers; it does not create one
async task per planned sample. Every document is limited to zero or one residual request.

Using the receipt-backed 28-call, 150-document baseline in two deliberately conservative ways gives
a planning estimate of approximately $4.68-$5.06 for the primary 10,000-document residual-rendering
run (about $0.00047-$0.00051 per output document). A $5.50 budget leaves margin for normal token
variance. This is a projection, not incurred or provider-reported cost. There is no separate
label-generation model charge in this path because the configured controlled target generator is
deterministic.

### Completed 10,000-document synthesis datasets

Host hardening after the initial exact-ID attempt made substantially more bindings deterministic.
The accepted exact-ID dataset was therefore published from the already-paid response lineage using
an offline replay; the layout-proxy-holdout arm then ran against the provider with the hardened
renderer. Both immutable runs published exactly 10,000 training records with zero host rejections,
zero provider errors, and complete format, semantic, source-relationship, topology, carrier,
literal-region, page-marker, and line-ending validation.

| Path | Passed | Residual calls | Deterministic slots | Additional estimated cost | Wall time | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Exact validation ID offline replay | 10,000 / 10,000 | 1,740 restored, 0 new | 985,207 / 990,702 (99.445%) | $0 | 46m 46s | 3,324,396 KiB |
| Validation proxy family | 10,000 / 10,000 | 1,816 new | 905,918 / 910,351 (99.513%) | $1.54268679 | 43m 19s | 2,747,668 KiB |

The exact-ID replay artifact is
`artifacts/kie-synthesis-production/synthesis-runs/mpci-bl-production-compiled-synthesis10000-exact-validation-id-v2-offline-replay`:

- Commit-file SHA-256:
  `0225f05647afe546a236367a6ed7224cf4f68ade928c12958f172690e19e7a0c`.
- Transaction SHA-256:
  `dba3edd55c0c58a3a3cd91a3c14b99d140c50e750cb4354ed946515f7512b9d8`.
- `dataset/records.jsonl` SHA-256:
  `c4173c04aba4c32027e11afd03bccd1e30bf71c5c8374f5ea53bd3798cf793a9`.
- The restored responses used by the accepted renderer represent $1.96829930 of the upstream
  lineage's usage; the replay itself incurred no new provider cost.

The layout-proxy-holdout artifact is
`artifacts/kie-synthesis-production/synthesis-runs/mpci-bl-production-compiled-synthesis10000-layout-proxy-holdout-v1`:

- Commit-file SHA-256:
  `e4f3911d813a0ef2432cfd0bf47d0f7e49a27989d8aa9316e4b0a649f0b0b2ea`.
- Transaction SHA-256:
  `739f7bd6b9c4ab74ecbacb8efaeefba6798913fa0c51a1b3136f63e4ce7ae6a9`.
- `dataset/records.jsonl` SHA-256:
  `0cffa4faa5305b546ebf8669495fc5ec76349250bad9439285e134d452fbff6c`.

Each accepted dataset retains the planned 7,500 standard, 1,000 dangerous-goods, and 1,500
temperature-controlled documents. Every document ID and rendered-input hash is unique within and
across the two arms. Neither arm overlaps the 100 real validation document IDs, validation input
hashes, or validation source IDs; the layout-proxy-holdout arm additionally has zero overlap with
the 26 validation-associated template-proxy families retained by the exact-ID arm.

### Target-generation scope

These plans use `controlled_latest_schema_from_source_v1`. Every published target is
`bill_of_lading_relation_explicit_v5` / `5.0.0-experimental`; no earlier target version is emitted.
The controlled generator preserves template topology and carrier while changing supported document,
voyage, forwarding/export, and seal identifiers plus issue/on-board dates. Source-only auxiliary
fields can also be regenerated during rendering when the compiler contract permits it.

For the primary plan, the final prepared targets contain 31,935 changed leaves (mean 3.1935 per
sample): voyage number 9,180 times, bill-of-lading number 8,260, issue date 7,672, shipped-on-board
date 6,652, forwarding/export references 120, and seal numbers 51. There are 9,953 distinct final
target payloads among 10,000 unique sample identities; 47 samples share a final target payload with
another variant. Therefore this configuration is controlled extraction/layout augmentation, not
broad semantic resynthesis of parties, routes, cargo descriptions, quantities, or dangerous-goods
content. The capability quotas describe the retained template/label capabilities; they do not claim
10,000 independently regenerated cargo scenarios.
