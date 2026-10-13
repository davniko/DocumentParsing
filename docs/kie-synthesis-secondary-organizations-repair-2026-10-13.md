# Shipper and secondary organization anonymization repairs

## Completed repairs and dataset scope

The remaining audited shipper and secondary-company identities are repaired in
the existing R20 working copy. The repairs cover older non-UAE samples, current
UAE samples and the superseded UAE sidecar. Their source bindings and synthesis
paths now generate or reuse the corresponding identities consistently, including
cross-page invoice references and abbreviated company names.

| Cohort | Synthetic records checked | Records repaired |
| --- | ---: | ---: |
| Older retained non-UAE | 2,500 | 76 |
| Current UAE replacement | 2,500 | 86 |
| Previous superseded UAE | 2,500 | 66 |
| Total | 7,500 | 228 |

Only **six records change a target**, all `parties.shipper.name` anonymization
backfills. The other **222 records change input text only**. No cargo, route,
date, package allocation, equipment, address target or contact label changes.
All 600 real training and 60 real validation records remain byte-identical.
The active training set remains 600 real + 5,000 synthetic. The repaired earlier
2,500 UAE records remain a separate sidecar, not additional training records.

Working copy:
[R20 dataset](../data/curated/mpci-bl-real600-synthetic5000-uae-coverage-r20-party-masterbill-positions-v1/README.md).
Original authorities and the running training dataset remain unchanged. The
pre-edit R20 files are preserved in the audit directory's `before/dataset/`.
No training job was launched.

This closes the named-organization repair slice. It is not full anonymization
of every numeric identifier, signatory or personal mailbox. Those categories
remain separate items in the
[broader anonymization inventory](kie-synthetic-anonymization-audit-2026-10-13.md).

## Corrections by source family

The table counts distinct documents per source family, not occurrences of a
name. A document may contain several repaired actors or aliases.

| Source prefix | Non-UAE | Current UAE | Previous UAE | Correction |
| --- | ---: | ---: | ---: | --- |
| `e5a6a403` | 6 | 2 | 4 | LKW shipper backfill and two independent invoice customers |
| `fbfcab7a` | 10 | 3 | 4 | Repeated LANS invoice aliases reuse the generated shipper |
| `0bdef3ca` | 6 | 3 | 4 | EL GALLAD abbreviation under Marks reuses the generated consignee |
| `0ba3ad4f` | 6 | 0 | 0 | Represented MORRIS organization and postal block |
| `19580ef7` | 10 | 27 | 10 | WESTON, BRODIES, postal block and abbreviated freight payer |
| `c1b20a71` | 10 | 26 | 10 | BRODIES/postal backfill and correct represented-company freight payer |
| `27a344df` | 6 | 15 | 20 | Independent SONVIGO exporter/postal block and VARDHMAN Marks alias |
| `cf1c5acd` | 6 | 2 | 4 | Independent SONVIGO exporter and postal block |
| `5d7bd82a` | 6 | 6 | 6 | Independent SONVIGO LIMITED exporter |
| `76dcec12` | 10 | 2 | 4 | Independent OCEAN CROWN invoice customer |
| Total | 76 | 86 | 66 | 228 distinct records |

The old inventory's generic `COMPANY` registration-type wording is not a private
company identity and was left intact. The previously fixed delivery-agent
identity is not repaired again.

## LKW shipper and independent invoice customers

`LKW-HANDEL M.NOCK` is the primary printed shipper A. Its own printed address
belongs to its target. The independently printed O/B company is also printed
as the notify party; it retains its own notify identity and is not concatenated
into the shipper's legal name or address. O/B does not establish an additional
forwarder or NVOCC role by itself.

The existing source contract already had a mutable `ob_primary` name owner.
Six old descendants nevertheless retained the original shipper identity in
both OCR and label, even though the rest of the party had been synthesized.
They were privacy backfills, not six discoveries that OCR and labels disagreed.
GLM generated six new company names conditioned on the existing party's
address/country. Each now appears identically in OCR and the shipper target;
the existing synthetic address and all other party fields are preserved.

For example, the first affected record now uses
`TRANS-MARCHANDISES K. BARRY`, replacing `LKW-HANDEL M.NOCK`. Its independently
printed notify/O/B company is not renamed as part of that primary repair.

Full inspection also found `GLOBAL BENELUX NV` and
`ELALAMIYA FOR METAL WORKS` in this source's freight invoice table. These are
customer names, not the public carrier/business-unit name next to them.
They are not exact aliases of the named primary parties, so they now have
separate generated invoice-company owners. Five origin-customer occurrences
and three destination-customer occurrences share their respective new name
within each document, across both pages. All twelve descendants receive this
input-only update; no invoice-party target is invented.

## Repeated primary company aliases

### LANS invoice references

The source prints `LANS LOGISTICS LLC` once in the shipper block and three more
times under invoice-party context, including page two. Seventeen descendants
retained the old invoice occurrences while their shipper label/name had already
changed. These occurrences must use the existing generated shipper, not a
second independent logistics-company draw.

The source contract's `v001` now covers all four exact occurrences. Existing
Care Of equipment-company and postal ownership remain separate. Dataset repair
reuses each document's current `parties.shipper.name`; that target is unchanged.

### EL GALLAD Marks abbreviation

The full source consignee is longer than the abbreviated company name under
Marks. Thirteen descendants retained `EL GALLAD COMPANY FOR TRADING` in Marks.
The Marks binding now reuses the generated consignee name. This neither creates
a Marks target nor imports Marks content into the goods description.

### VARDHMAN brand alias

Full-text synthesis review identified `VARDHMAN MADE IN …` after the shipper
had changed. Forty-one descendants are affected. The existing Marks expression
now uses the generated shipper followed by `MADE IN` and the existing sampled
origin. The country and goods-origin target are unchanged.

## Independent represented and exporter organizations

MORRIS, WESTON, BRODIES and the detached invoice/exporter companies are not
automatically interchangeable with the primary shipper. Existing reviewed
on-behalf and care-of ownership was reused for the older MORRIS/BRODIES/WESTON
backfills. Their company names and still-original postal blocks are generated
separately; none is copied into A's label.

SONVIGO exporter records and OCEAN CROWN invoice records previously had source
text/historical bindings but no active generated lexical owner. New `name`
and, where present, `postal` variables cover their exact spans. Two different
documents' independent secondary companies receive independent draws. Repeated
occurrences within one document share one draw.

The exporter postal blocks retain their explicitly printed public geography:
Port Louis, Mauritius. This location is verified against the pinned GeoNames
registry, including entry `934154`; it is not inferred from the primary party,
issuer or language. Its address words/numbers are generated anew. The exporter
and its country declarations stay separate from the shipper's country/target.

## Freight payer ownership and upstream implementation

`BY WESTON` is an abbreviated occurrence of the represented company B. It must
reuse B, not a new third company and not primary shipper A. One older source
recipe mapped that payer to A; both affected source families now explicitly
reuse `ob_related_name_1`. Ninety-three documents receive this alias alignment,
some overlapping the longer-name/postal repairs above.

`ownership_surfaces()` now supports a small explicit recipe:

```yaml
agent:shipment:freight_payer:
  lexical_key: ob_related_name_1
```

It accepts only a known name variable with a supplied nonempty generated value.
Missing values, unknown keys and postal/numeric owners raise an error. During
preflight, the caller explicitly supplies baseline values to validate ownership;
during final rendering it supplies generated wording. No silent default name
or additional model call is introduced.

`curated_campaign.py` also supports source-only `invoice_party` and `exporter`
actor contexts. Optional independently certified geography uses the existing
`ScenarioLocation` type. It must have printed country evidence and match the
pinned registry. These actors participate in wording and final review without
adding extraction-party targets. Existing on-behalf, care-of, account-of and
agent-for behavior is retained.

Six source contracts were extended. Ownership declarations changed only where
needed; the auxiliary file changed only two source blocks. Unrelated source
blocks were preserved byte-for-byte after minimizing the formatting diff.
Source OCR, source positions and all real gold targets remain unchanged.

## Validation and rejected drafts

### Complete dataset comparison

[Dataset validation](analysis/synthesis-secondary-organizations-repair-20261013/dataset-validation.json)
checks all 7,500 synthetic records, including the earlier UAE sidecar. It
replays every exact edit independently, checks all target paths and input
hashes, preserves unchanged rows byte-for-byte, verifies mirrored splits,
and validates the canonical schema. All line counts, page markers, coordinate
suffixes and known/unknown position states remain unchanged. New company
wording uses the same line-owner anchors; it is not a claim of measured
synthetic glyph positions.

Only the six authorized shipper-name leaves change. Public carrier identity
occurrence counts remain unchanged. No audited old organization/postal block
remains in its affected family, including abbreviated freight and Marks aliases.

Eight deliberate corruption controls were rejected, including a mismatched
shipper target, a missed repeated invoice occurrence, invented source-only
party target, unrelated container mutation, lost coordinates, stale input hash,
and carrier/signatory replacement. The strongest control supplies a
self-consistent forged edit receipt attempting to replace a public carrier
as a customer; the independent public-identity check still rejects it.

The draft corpus alias join needed tightening: an unscoped `ON BEHALF OF` or
`BY` search could select carrier signing text instead of the continuation
company/freight payer. Draft outputs were not applied to the working dataset.
The final join is anchored to the unique continuation-shipper and freight-payable
blocks and checks the actor relationship independently. Production uses exact
compiled binding spans rather than that ad hoc corpus search. This failure and
its corruption control are retained in the audit, not hidden behind edit replay.

### Semantic review and adjudication

All 119 records with the initial generated company/postal replacements received
scoped full-OCR/target review. The reviewer returned 91 entries: two genuine
abbreviated WESTON observations and 89 non-actionable/incorrect observations.
Most were explicit passing notes or assumed that the old `before` identity
should still occur in the already-repaired OCR, or that source-only companies
needed invented target parties. All entries were adjudicated and retained;
good values were not rewritten to satisfy those mistaken claims.

The two real alias findings led to the source-wide freight-payer fix described
above. Current edit proofs check all 93 corresponding documents. The reusable
review instructions now explicitly identify the OCR as final repaired text and
require an empty findings list for passing checks. The supplemental twelve LKW
invoice-company pairs were also inspected directly, including all repeated
occurrences and distinction from the primary parties.

### Fresh synthesis and coordinates

All ten templates generated two fresh production descendants each. The twenty
samples passed current source replay, native review provenance, rendered/target
ownership checks, production publication and independent geometry validation.
The final inspection samples were not added to training.

Geometry validation covers **48 pages**, **2,156 in-bounds source/page boxes**,
**26 generated blocks** and **69 generated centroids**. The 275 inherited
source-rectangle overlaps are recorded as inherited, not claimed as newly
created reflow collisions. Exact candidate/source/position replay checks the
geometry actually emitted; missing source anchors remain explicitly unknown.

Five final pilot review entries were rejected: passing observations, a
per-carton-fill arithmetic complaint outside the agreed extraction objective,
and a cargo-net-versus-container-tare observation. No corresponding label/text
contradiction existed under the extraction policy, so good data was preserved.

One unrelated pilot postal proposal repeated an extra metropolitan city/country
qualifier. The normal postal gate held it. Bounded proposals and a documented
inspection-only adjudication retained the supplied locality, postcode, street
and one country. It then passed rendering, a fresh review and geometry. No
production city-specific rule was added, and that inspection sample is not
training data.

See [pilot certification](analysis/synthesis-secondary-organizations-repair-20261013/pilot-certification.json)
and each `upstream-pilot/<source-prefix>/samples.md` / `positions-reflow-v1/samples.md`
for the actual source/render pairs and coordinates.

### Registry and tests

All **360 source contracts**, **2,160 case files** and four shared files pass
identity replay and current file-pin validation. There are **25,889 compiled
regions** and no unresolved ownership overlaps.

The final relevant suite passes **1,275 tests** in 45.78 seconds:

```bash
TMPDIR=/tmp .venv/bin/python -m pytest tests/test_curated*.py tests/test_synthesis_curated.py -q
```

New tests cover independent invoice/exporter contexts, independently declared
exporter geography, invalid location evidence, and generated name-alias values
including missing/empty/wrong-kind owners. Ruff and diff checks pass.

## Cost and performance

All linguistic calls use GLM-5.3-Flash, low reasoning, native structured output
and the existing plain-text prompting path. Total provider-reported cost,
including initial reviews, follow-ups and fresh pilots, is **$0.10472412**.

| Work | Cost USD |
| --- | ---: |
| Initial company/postal generation | 0.00733743 |
| Initial scoped semantic review | 0.06911352 |
| Supplemental LKW invoice companies | 0.00050614 |
| Fresh production pilots and follow-ups | 0.02776703 |
| Total | 0.10472412 |

The review, not generation, dominates cost; its passing-note ambiguity is
documented above. Existing generated aliases require no new company draw.
Independent secondary actors are generated in the source's ordinary joint
wording call, not a separate call per occurrence.

The unchanged ownership-rendering microbenchmark measured **110.68 µs before**
and **110.84 µs after**, a **0.17 µs** difference, with identical output. It uses
seven repetitions of 2,000 calls against the same unchanged real blueprint.
No new blocking call was added to the rendering hot path. The measured complete
dataset audit used approximately **164 MiB peak RSS**; its final audit function
took about 28 seconds. No GPU or training model was used for this repair.

## Evidence and remaining work

The audit directory is
`docs/analysis/synthesis-secondary-organizations-repair-20261013/`.
It contains the pre-edit dataset/registry backups, proposed and installed
contracts, generated values and provider receipts, exact edit inventory,
adjudication records, corruption checks, fresh production samples, geometry
receipts, registry validation and final completion certificate.

The completed delivery-agent/master-B/L repairs are preserved. Numeric tax,
registration, customer/reference identifiers and personal names/mailboxes in
other non-target contexts remain outside this slice. Public carrier identities
are deliberately preserved. Original source material and audit backups remain
private provenance; they are not part of the anonymized synthetic training rows.
