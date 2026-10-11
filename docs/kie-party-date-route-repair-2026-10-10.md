# Primary-party, date and delivery-occurrence repair

Date: 2026-10-10. Status: **published and validated**.

This report closes the three requested repair components: related-company party
targets, numeric date policy, and delivery labels borrowed from other OCR roles.
It records applied changes, not just an audit proposal. Training results and
further synthetic scaling are separate work.

## 1. Result and current authorities

The current mixed dataset has been edited **in place**:

[`data/curated/mpci-bl-real600-synthetic2500-v7-positions-boundary-repaired-v1`](../data/curated/mpci-bl-real600-synthetic2500-v7-positions-boundary-repaired-v1).

It still contains 3,100 training records—600 real and 2,500 synthetic—and 60 real
validation records. No document was added, removed, moved between splits or
regenerated. **381 unique documents have corrected targets:** 77 real training,
eight validation and 296 synthetic. Component counts overlap; adding the rows
below would double-count nine documents.

| Applied component | Real train | Validation | Synthetic | Unique documents |
| --- | ---: | ---: | ---: | ---: |
| Primary A instead of A/O-B composite identity | 29 | 3 | 104 | 136 |
| Care-of, account and explicitly represented-agent boundaries | 15 | 0 | 72 | 87 |
| Supported issue/on-board dates added | 31 | 4 | 105 | 140 |
| Unsupported delivery role removed | 5 | 1 | 21 | 27 |

Combined party corrections affect **223 documents: 47 real and 176 synthetic**.
All related-party metadata is retained in the dataset's
[`party-date-route-repair.json`](../data/curated/mpci-bl-real600-synthetic2500-v7-positions-boundary-repaired-v1/party-date-route-repair.json),
outside the extraction target.

The corresponding real-source authority is
[`mpci-bl-real-v7-reviewed-r16-paddle-positions-660`](../data/curated/mpci-bl-real-v7-reviewed-r16-paddle-positions-660).
Its 600 training and 60 validation targets agree with the real records in the
mixed dataset. The active
[`300-source template registry`](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/README.md)
has **44 changed source cases**, including their necessary target, contract,
ownership, auxiliary and route-capability changes. Every current template's
source text and target replay exactly.

**All real OCR and positional inputs are unchanged.** A targeted follow-up
corrects one repeated secondary-company footer in six synthetic records, in
both plain and positioned text. Every coordinate value, line count and page
membership remains unchanged; all other input text is unchanged. The
synthetic-only mirror matches the 2,500 synthetic training records exactly.
The new in-place data does not rewrite snapshots already stored in completed
model runs.

Before publication, all existing files to be changed were copied to
[`artifacts/dataset-backups/party-date-route-20261010T183624Z`](../artifacts/dataset-backups/party-date-route-20261010T183624Z).
Older R14/per-sample source-label stores and historical synthesis campaigns remain
historical snapshots; they are not interchangeable with the current R16 JSONL
authority.
The six input corrections additionally have their own pre-edit archive, recorded
in [the input publication receipt](analysis/party-date-route-repair-20261010/related-input-publication.json).

## 2. Adopted party policy and what the research establishes

The user explicitly selected **primary printed company A and A's own address**
for an `A ON BEHALF OF B` / `A O/B B` relationship. B and the relationship remain
audit metadata, not a second organization concatenated into A's `name`.

“Primary printed” is our extraction selection rule. It does **not** mean that A
is the legal principal in an agency relationship: B is the represented actor.
The local MPCI schema separates shipper, consignee, notify, forwarding and
delivery roles. It contains no inspected represented-principal relationship
slot that justifies placing two different organizations in one singular KIE
name/address pair.

The official MPCI specification separately describes shipper and forwarding
agent fields; Appendix C concerns actual ultimate-party identity for filing.
It does not make `O/B` an automatic forwarding-agent classification, nor does
it supply a universal first-company/last-company extraction algorithm.
[MPCI business specification, table 10 and Appendix C](https://naic.icp.gov.ae/portal/assets/mpci-guidelines/UAE%20MPCI%20Business%20Specification%20Document%20V1.0.pdf).

The distinction matters: extracting the printed party selected by our target
policy is not independent verification of the ultimate exporter/importer for
a regulatory filing. If the application needs an ultimate party different from
that printed party, it needs a separately established role and address—not
silent substitution inside the extraction label.

A Maersk advisory uses O/B and C/O for freight-contract customer validation,
demonstrating that these expressions can concern commercial relationships rather
than proving a particular MPCI role.
[Maersk contractual-customer advisory](https://www.maersk.com/~/media_sc9/maersk/local-information/files/asia-pacific/korea/export/customer-advisory/notice-of-contractual-customer-validation-rule.pdf).
An NVOCC's function includes issuing its own house B/L and acting as shipper
toward the vessel carrier. That function cannot be inferred merely from O/B.
[FMC ocean transportation intermediary definitions](https://www.fmc.gov/about/bureaus-offices/bureau-of-enforcement-investigations-and-compliance-beic/office-of-compliance/ocean-transportation-intermediaries/).

Accordingly, **no new forwarding-agent role was invented** in this repair.
Independently printed roles, including B printed separately as notify, survive.

### 2.1 Before and after: the LKW example

The OCR prints:

```text
LKW-HANDEL M.NOCK
KOLUMBUSSTRASSE 49
53881 EUSKIRCHE
GERMANY
O/B LKW-HANDEL
GUNTHER STIETZEL GMBH & CO. KG
```

Before, `name` combined both companies:

```json
"name": "LKW-HANDEL M.NOCK O/B LKW-HANDEL GUNTHER STIETZEL GMBH & CO. KG"
```

After:

```json
{
  "name": "LKW-HANDEL M.NOCK",
  "addressLine": "KOLUMBUSSTRASSE 49 53881 EUSKIRCHE GERMANY",
  "country": "GERMANY"
}
```

The separate represented/notify company remains in its independently supported
notify role and in relationship metadata. Future synthesis generates a new A
as well as B; it no longer leaves `LKW-HANDEL M.NOCK` as a fixed literal. A and B
have separate lexical ownership, while B's independent notify occurrence shares
B's generated identity.

### 2.2 Address and contact ownership

Three cases are handled differently, deliberately:

- A's postal block precedes a separate B relationship: keep A's own block.
- A acts for B, followed by a clearly B-owned or nested third-party address:
  do not label that address as A's.
- A/O-B is followed by one undifferentiated postal/contact block: archive it
  with **owner unspecified** and omit that component from A's target. Keep the
  resolved A name. This does not claim that the block belongs to B.

The unspecified-ownership archive contains 13 real and 30 synthetic records.
These are not unresolved primary-company choices or whole-document holds. The
selected A identity is resolved; uncertain postal/contact ownership is explicitly
outside the target. Country, real-world headquarters, domain names and company
knowledge were not used to fabricate ownership.

The complete primary-company audit read 37 real source party blocks and all 137
current descendants of the relevant 16 active families. It retained five
already-correct real controls and 33 descendants whose actual generated topology
had become A-only. A blanket family-level postal deletion would have damaged
those valid descendants, so individual before/after decisions were used.
Three translated relationship names were included; the inventory was not limited
to an English `O/B` regex.

### 2.3 Genuine care-of is postal routing, not a combined name

For example:

```text
ACE PRIMA RESOURCES SDN BHD
C/O CHEMIDEX FZCO
23080-001 A2 BUILDING IFZA BUSINESS PARK DDP, DUBAI UNITED ARAB EMIRATES
```

The target name is only `ACE PRIMA RESOURCES SDN BHD`. The address begins
`C/O CHEMIDEX FZCO ...`, retaining the printed routing recipient. We do not strip
every organization-shaped string from postal text.

`A/C B` is account relationship context, not a legal-name suffix. `AS AGENTS OF`
and `NVOCC AS PRINCIPAL` are capacity descriptions, not automatically another
form role or part of the company's name. The care-of/account/agent pass corrected
15 real and 72 synthetic records and 11 active templates. Their exact approved
target changes and relationship metadata are in
[the care-of/account plan](analysis/party-date-route-repair-20261010/party/careof-account-plan.json).

## 3. Dates: exact effect and prevention

Both active full and reduced extraction schemas contain exactly two structured
date fields: `issueDate` and `shippedOnBoardDate`. Both were inventoried across
all 3,160 current records and all 300 templates.

The policy is now:

1. Establish the occurrence's semantic owner; issue, on-board, invoice and other
   dates are not interchangeable.
2. Honor an explicit printed format, written month or independently established
   same-source convention. Do not infer order from geography, issuer or language.
3. When both numeric orders remain possible without such evidence, use **DD-MM**.
   Thus `03/06/2025` becomes `2025-06-03`, not an absent target.
4. Impossible, incomplete or conflicting dates remain review issues; the default
   does not invent a year or resolve conflicting ownership.

The applied repair contains **213 date additions**: 139 issue dates and 74
on-board dates, across 140 documents. **No existing date label was replaced.**
Date-field presence changed as follows:

| Field | Before | After | Added |
| --- | ---: | ---: | ---: |
| `issueDate` | 2,441 | 2,580 | 139 |
| `shippedOnBoardDate` | 2,236 | 2,310 | 74 |

Source `0d102efa`, previously described as a deliberately omitted date in the
random audit, and all nine of its current descendants now have the supported
issue date. That earlier audit used the wrong field policy and is explicitly
superseded; its original observations are retained as history rather than
rewritten to make the prior decision appear correct.

Sixteen template cases were repaired. Fifteen source targets gain 15 issue and
eight on-board dates; the sixteenth preserves an explicit month-first printed
format in both repeated copies. Two hidden month-first auxiliary recipes are
now target-backed day-first owners where the source provides no month-first
evidence. Source `66dc4377`, which actually prints `Month/Day/Year`, retains its
explicit month-first rendering. This is evidence-based handling, not a global
rewrite of every slash date.

The new shared [date policy module](../src/document_ocr/date_policy.py) is used
by deterministic grounding and numeric synthesis rendering. The direct reviewer
also receives a non-mutating date-fidelity diagnostic. This diagnostic is narrowly
captioned; it cannot label every date-shaped token by itself.

Falsification found a letter-of-credit `DATE OF ISSUE` that a broader detector
would have mistaken for the B/L's date. The diagnostic was narrowed and a
regression test added. Telephone substrings, an administrative address fragment
and product serials resembling dates remain unchanged. A named-month value in
a freight Collect cell was also left out because its date **role** was not
established—not because numeric order was ambiguous.

See [the full date census, adjudications and probes](analysis/party-date-route-repair-20261010/DATE-REPORT.md).

## 4. Delivery labels: missing occurrence is not a missing heading

The delivery inventory covered all 660 real documents, all 300 templates and
all 2,500 synthetic descendants. Before repair, 229 real documents, 100 templates
and 880 synthetic descendants had a delivery target.

Six real labels were unsupported by a delivery-owned OCR occurrence. Three of
those real sources are active templates with 21 current descendants. Consequently
**27 `placeOfDelivery` objects were removed**: five real training, one validation,
21 synthetic. Discharge, payment, parties and all other route labels were preserved.

The affected real source prefixes are `291f67b3`, `a4cb81c1`, `147a2626`,
`5adf9d49`, `9c68fe94` and validation `2809b1ea`. Full IDs, OCR positions,
PDF links and dispositions are in
[the route report](analysis/party-date-route-repair-20261010/route/REPORT.md).

The root cause was declarative: a discharge/payment location owner had been
assigned additional delivery target paths, and `route_topology` then sampled a
delivery value despite the missing delivery occurrence. Identical strings
elsewhere in OCR were incorrectly treated as evidence for the omitted role.
This exact replay could pass mechanically while being semantically wrong.

The repair removes the unsupported target **and** its ownership paths and
route-topology/capability entry. It does not delete genuine discharge/payment
owners. Future descendants of those three templates therefore no longer acquire
this unsupported field.

Ten explicit positive controls protect genuine retained delivery occurrences
whose headings were lost, detached or misordered. They remain unchanged. Layout
may establish ownership of text that survives OCR; it cannot restore a value
occurrence omitted by OCR. No source-ID exception or literal-heading-only rule
was added to the runtime extraction prompt.

## 5. Layer-by-layer implementation

| Layer | Applied change and rationale |
| --- | --- |
| Current mixed train/validation targets | Apply only reviewed component deltas; preserve IDs, split order and all other labels |
| Six repeated secondary-company input occurrences | Replace only the stale footer identity with the already printed A/C company; preserve every coordinate and other input byte |
| Synthetic mirror | Publish the identical corrected synthetic records, not a competing revision |
| R16 real-source train/validation targets | Mirror the current real targets used for future compilation and sampling |
| Active source cases | Update target snapshots, source contracts and target hashes together |
| Shared ownership/auxiliary declarations | Separate A/B/postal roles, add missing date owners/visible format recipes, remove unsupported delivery paths/topology |
| Registry/config manifests | Repin every changed source, declaration, source-dataset and task-constraint dependency |
| V7 extraction schema | Specify one primary entity, genuine care-of postal routing, evidence-first DD-MM date fallback and OCR-retained route roles |
| Section reviewer priorities | Use the same entity/role policies; unresolved ownership is flagged rather than guessed |
| Deterministic/legacy date paths | Use the same numeric resolver and explicit printed-format evidence |
| Template compiler/critic | Treat compound party targets as a source-label issue; keep actors separate and certify actual role occurrences |
| Synthesis lexical request | Generate related actors as distinct, source-only context in the existing call; do not invent extraction roles |
| Synthesis rendered review | Provide related-actor context even when no identity-kind certificate exists |
| Admission/generation validation | Reject a related actor concatenated into A's name or an unowned repeated related identity; reject generated relationship framing in a name component |

The new `related_actors` declaration is deliberately small: a lexical variable,
its context party and its relationship. It does not add an agent chain or a new
public extraction field. Source-only names and postals are generated in the
existing lexical call, with concise field descriptions. There is no per-record
API relabeling campaign and no extra paid call required just to support B.

The active training config remains
[`t5gemma2_270m_lora.mpci_bl_real600_synthetic2500_boundaries_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml`](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic2500_boundaries_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml).
Its dataset hashes and frozen constraints were updated; hyperparameters were
not changed and no training was launched. Active coverage/expansion synthesis
configs likewise use the repaired source and current ownership/capabilities.

New frozen constraints live in
[`mpci_bl_real600_synthetic2500_party_date_route_v1`](../configs/training/production/contracts/mpci_bl_real600_synthetic2500_party_date_route_v1).
The actual training-task loader caught a missing canonical final newline during
publication verification. The file was corrected to the required canonical
encoding and every dependent hash was repinned. The actual config/task startup
loader now passes. Final task-constraints SHA-256 is
`be8ba7b284e1e16f8664bd8a3383e07dca89e20caadad254d2f26fe0c257338d`.
Old frozen contracts remain intact for reproducing old experiments.

## 6. Validation: what was proved and how

### 6.1 Exact publication reconciliation

[The post-publication checker](analysis/party-date-route-repair-20261010/validate_publication.py)
compares every current row with the backed-up input plus its approved exact
component edits. It does not accept substring presence as proof of semantic
ownership. The latest
[receipt](analysis/party-date-route-repair-20261010/postpublication-validation.json)
records:

- All **3,160 targets** validate against the actual configured schema.
- The **381 changed documents** equal their approved before/after edit replay;
  all other document targets are unchanged.
- All real inputs and unrelated synthetic input text are unchanged; six exact
  footer-name edits and their input hashes replay from the approved receipt.
  Coordinate values, line/page counts and other record metadata are unchanged.
- All **660 current real targets** agree between source authority and mixture.
- All **2,500 synthetic mirror records** equal their training counterparts.
- All **300 template source-text/source-target replays** pass after the repairs.
- All manifest, catalog dependency, data-file and active training-config hashes
  match; the actual frozen-schema/task loader passes.
- Zero remaining findings from the scoped corpus date diagnostic.
- All 27 delivery removals are verified and all ten retained positive controls
  remain intact.

This exact-delta check also prevents a resolved component from being held back
because a different semantic component needed a separate decision. Unassigned
postal ownership is explicitly excluded, not converted into a whole-document
failure or silently declared resolved.

### 6.2 Targeted renderer probes and falsification

Before publication, source/changed-value renderer probes included:

- 16 primary-party source replays and 32 changed-party production renders;
- 11 care-of/account/agent source replays and 11 changed-A/B renders;
- 16 date source replays and 32 changed-date renders, including explicit-MDY
  repeats and dates that distinguish day/month ordering;
- seven delivery/positive-control source and changed-route renders;
- seven route-presence negative controls and three stale-ownership negative
  controls, all rejected as expected.

Production tests additionally reject related actors that appear in the primary
name, unknown context roles, incomplete actor declarations, and repeated actors
whose second source occurrence is unowned. Genuine care-of postal text and
independently named roles remain positive controls.

The final combined relevant suite passed **488 tests in 28.62 seconds**.
Ruff passes for the touched production modules/tests and final validation scripts.
An earlier broader run recorded four unrelated failures: one existing equipment
alias expectation and three raw-text fixtures missing `cargo_policy`. They were
not silently fixed or counted as successes; the date subreport retains that
history. The 488-test statement is the specified relevant suite, not an assertion
that every repository test was run and passed.

### 6.3 Actual fresh synthesis, not cached-wording replay

Ten selected active templates ran the normal registry sampler, host shipment
facts, GLM-5.3-Flash constrained lexical generation, rendering and full rendered
review. Root read the complete source OCR and final rendered OCR/target for each.
[Final inspected inventory](analysis/party-date-route-repair-20261010/UPSTREAM-INVENTORY.md)
and [hash-bound validation receipt](analysis/party-date-route-repair-20261010/upstream-synthesis-validation.json)
identify the exact accepted versions.

The fresh test covered unknown A postal ownership, independent B notify,
genuine care-of routing, A/C, explicit agency, formerly omitted dates, explicit
month-first formatting, and all three removed-delivery template families.

One generated postal locality duplicated an existing fragment; the existing
bounded postal follow-up corrected it. The first `f256749b` candidate also
exposed a **new template occurrence defect**: A/C B changed, but a repeated
exporter-footer occurrence of B remained literal. All 27 related-name variables
were scanned; this was the one unowned full-name repeat. Both occurrences now
share the same mutable B owner. A general admission guard rejects any declared
related actor with an unowned repeated full-name occurrence, including wrapped
whitespace forms. No source-specific runtime branch was added.

That template was regenerated and reviewed again. The **final ten samples have
ten hash-matched rendered reviews with zero findings**, plus the completed root
inspection. The initial f256 candidate is explicitly replaced, not relabeled
as a pass. A separate initial review comment about private tare/measurement
plausibility did not establish a public-target error and did not authorize
inventing another target field.

The same repeat defect was then checked in all six existing descendants of
f256. Each printed its generated B after `A/C` but retained `SONVIGO INTERNATIONAL
LTD` in the exporter footer. Each footer now uses that same sample's already
printed B, in uppercase, with exact single-occurrence edits in plain and
positioned text. No LLM rewrite, label change, line insertion, coordinate change
or whole-sample regeneration was needed. A dry-run verified all six isolated
edits before publication; final corpus validation replays them against the
original inputs. The
[input repair receipt](../data/curated/mpci-bl-real600-synthetic2500-v7-positions-boundary-repaired-v1/related-actor-input-repair.json)
records exact before/after spans, line numbers and hashes. Historical geometry
receipts remain provenance for the inherited centroids; the explicit text
amendment records that the footer wording changed without resynthesizing those
coordinates.

Fresh provider-reported cost, including both runs, reviews and the postal
follow-up: **$0.01227670**, approximately 1.23 cents. Corpus target repair itself
used **zero paid requests**. None of these inspection samples was added to
training.

### 6.4 Coordinates

The ten final fresh samples also ran measured-source alignment, position
transfer, calibrated page reflow and coherent positional augmentation through
the production path. An independent check verifies accepted page geometry using
the actual configured minimum clearance—not a zero-gap substitute—plus page
ownership, text preservation and integer coordinate bounds.

[Geometry receipt](analysis/party-date-route-repair-20261010/upstream-geometry-validation.json):
**959 lines**, of which **808 have coordinates** and 151 remain explicitly
unpositioned; 11 pages accept reflow and five have no eligible reflow. All
accepted pages pass clearance/geometry checks, and all retained coordinates are
within 0–1000 on the correct page. Unknown coordinates remain explicit ` ||`
without dropping text or inventing measured anchors.

These are plausible source-conditioned synthetic positions, not a claim to have
measured a newly rendered PDF. Existing training coordinates were not resynthesized
by this target repair.

## 7. Runtime and memory measurements

| Measurement | Observed result |
| --- | --- |
| Complete publication check | 12.68 seconds in checker; 20.57 seconds including Python imports; peak RSS 269.9 MiB |
| 44 affected templates, compile + exact source replay, six alternating rounds | Median 358.91 ms before / 361.37 ms after; **+0.69%**, negligible parity-level difference |
| Current lexical/related-actor admission validation | Median 4.83 ms for 44 templates; **109.8 microseconds/template** |
| Numeric date renderer, committed before implementation vs current | 76.79 ms → 20.93 ms per 400 calls; **3.67× faster** |
| Corpus date diagnostic with tracemalloc | 0.664 seconds for 3,160 records; tracked peak allocation 51,508 bytes |
| Fresh ten-sample geometry pass | 10.33 seconds |

The compilation comparison deliberately measures the same build/replay work
on both versions. Old contracts fail the newly required repeated-actor guard
by design, so an old/current admission throughput comparison is not fabricated.
The new admission cost is measured separately. No new hot-path network call or
GPU workload was introduced.

The corpus checker streams the old/current mixed rows and retains only the
targets and hashes needed for cross-publication checks. A temporary earlier
validation implementation retained duplicated large rows; it was replaced before
the final verification to avoid that unnecessary memory usage.

## 8. Boundaries and remaining non-target context

There are **zero unresolved primary-entity selections, date-repair decisions or
delivery-occurrence decisions** in these enumerated repair components. Supported
components are published; unassignable postal/contact context is archived and
deliberately absent from A's target.

This is not blanket anonymization of every carrier clause, booking office,
customs caption, person or private reference in every source. The confirmed
A/C-versus-footer repeated-company mismatch is fixed in all six current
descendants and in future template generation. Remaining fixed carrier/legal
and private context is distinguishable from the completed extraction-target
repairs and is not silently described as fully anonymized. None of that context
justifies inserting B into A's target.

No extra source address was obtained from a company website, no missing PDF-only
delivery occurrence was restored, no non-date target was normalized as a side
effect, and no old model-run metrics were overwritten.

## 9. Evidence and repeatable verification

- [Applied publication summary](analysis/party-date-route-repair-20261010/publication.json).
- [Primary-company adjudication and exact metadata](analysis/party-date-route-repair-20261010/party/REPORT.md).
- [Primary template repair probes](analysis/party-date-route-repair-20261010/party/TEMPLATE-REPORT.md).
- [Date census and renderer tests](analysis/party-date-route-repair-20261010/DATE-REPORT.md).
- [Delivery occurrence decisions and PDF links](analysis/party-date-route-repair-20261010/route/REPORT.md).
- [Repeated related-actor before/after receipt](analysis/party-date-route-repair-20261010/related-repeat-repair.json).
- [Six existing descendant input corrections](analysis/party-date-route-repair-20261010/related-input-publication.json).
- [Final synthesis inspection inventory](analysis/party-date-route-repair-20261010/UPSTREAM-INVENTORY.md).
- [Performance receipt](analysis/party-date-route-repair-20261010/benchmark.json).

Read-only corpus validation:

```bash
.venv/bin/python docs/analysis/party-date-route-repair-20261010/validate_publication.py
```

The publisher's before-state plans must **not** be rerun against already repaired
files. They deliberately require the original values and hashes. New repairs
need their own reviewed before-state receipt, not a silent reapplication or a
relaxed precondition.

The central [label-policy document](kie-real-baseline-label-policy-2026-10-05.md)
now records these decisions. This visible report is the current status; linked
subreports describe their earlier proposal/adjudication stage and are preserved
as chronological evidence.
