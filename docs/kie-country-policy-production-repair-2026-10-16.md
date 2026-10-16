# Country policy, production rendering and dataset migration

Date: 2026-10-16. Completed production implementation and bounded country-only
dataset certification. Historical datasets and metrics remain unchanged.

## Decision and boundaries

Country fields contain one pinned, uppercase, preferred English country name,
for parties, routes, issue places and payment places alike. Examples include
`DEMOCRATIC REPUBLIC OF CONGO`, `SOUTH KOREA`, `UNITED STATES`, `TURKEY` and
`UNITED ARAB EMIRATES`. They are not ISO-code targets in this experiment.
The separately proposed ISO-target training experiment remains a future change.

Postal `addressLine` retains the actual rendered country presentation. Locality
`name` retains the printed locality spelling. Neither is geocoded or translated.
An explicitly country-qualified occurrence of the same printed place can supply
document-local, reviewed co-reference evidence; this is not city-name geocoding
or permission to borrow an unrelated party's country. Absent optional country fields are not invented. Real OCR and coordinates are
not rewritten by this country-label migration.

Punctuation matters: user-approved `KOREA` resolves to South Korea, while dotted
`A.R.E`/`A.R.E.` resolves to Egypt and undotted ISO3 `ARE` remains UAE. The Egyptian
initialism is documented in the [official EGPC model agreement, definition (j)](https://eug.petroleum.gov.eg/dp/jsp/assets/docs/2023%20EGPC%20Model%20Agreement%20%28English%29.pdf).
Aliases are not recognized by blindly stripping punctuation. Unknown identities
raise an explicit review condition; genuinely unresolved records remain unchanged
for user review rather than losing their country fields.

## Production architecture

1. The immutable, hash-pinned registry supplies 249 ISO identities, 3,401 sampled
   presentations and a separate input-only alias vocabulary. It includes ISO
   names/codes, CLDR localized names and reviewed initialisms; corpus-only typos
   and business aliases do not automatically become rendering choices.
2. Shipment geography and lexical generation stay canonical. Existing postal,
   ownership, packaging, DG and thermal validators run before alias projection.
3. The country render plan selects an owned country span within each native
   region, never by global document replacement. It protects actual sampled
   locality and company wording, preserves source name/code frames and code
   captions, and projects matching printed country text into address targets.
4. Country fields normalize to preferred English names. Native edit receipts
   include registry/configuration pins, source ownership, language evidence,
   input/output values and exact country edits. Independent replay reconstructs
   the same result before publication.
5. Existing synthetic reflow receives those native edits. Country-only migration
   of already-positioned records preserves each physical line and its exact
   coordinate suffix, including explicit unpositioned ` ||` lines.
6. Extraction and repair apply programmatic country normalization before final
   target validation, recording changes. Review instructions distinguish
   normalized country fields from literal address/locality text.

The same registry also resolves source-country metadata for phone rendering and
shared-party locality certificates. An exhaustive check found 17 preferred
English names (including `DEMOCRATIC REPUBLIC OF CONGO`, `IRAN` and `RUSSIA`)
that the old official-ISO-name-only resolver did not recognize. Leaving that
consumer unchanged would silently lose known source-country metadata after this
migration. ISO identities still own shipment sampling; presentation aliases do
not change sampled geography or phone digits.

No model loading or training is part of this work. Historical dataset snapshots
remain unchanged.

## Configuration

The default is a **30% document-level alias opportunity**, not a promise that
30% of all country tokens change. Source frames, code captions, available
languages and physical line counts constrain eligible choices. The remaining
documents use preferred English names except where a source frame requires a
code. Within alias opportunities, six families have independently configurable
weights; the default weights are equal.

```yaml
country_registry_manifest: artifacts/registries/compiled/country-presentations-iso-cldr-48.2.3-v1/production-manifest.json
country_registry_sha256: aa8f579b7a5855a47b11dca35b27c38198f07d090d801b6c1fe86917c4b854b4
country_presentations:
  alias_probability: 0.30
  family_weights:
    name: 1.0
    iso2: 1.0
    iso3: 1.0
    abbreviation: 1.0
    dotted_code: 1.0
    localized_name: 1.0
  language_allowlist: [en, it, fr, es]
  unavailable_family_policy: renormalize_eligible
```

`renormalize_eligible` is an explicit selection policy, with eligibility recorded
in receipts, not an unknown-country fallback. `error` is available for strict
cohort experiments. Nondefault registries require an explicit manifest hash.

Localized names require a language enabled by sampled document geography or a
reviewed, fixed issuer-country literal. An Italian-language name is not enabled
by guessing nationality from a generated company name. English remains available.
Two/three-letter sampling aliases are ISO codes, except explicitly reviewed
international initialisms and their punctuation variants. Atomic names such as
`CONGO (DRC)` and `ROYAUME-UNI` are recognized before splitting name/code frames.

## Source and dataset inventories

The previous isolated [end-to-end probe](kie-country-endtoend-probe-2026-10-16.md)
established the projection mechanisms. This implementation additionally scans
all retained synthetic datasets rather than assuming that probe coverage proves
full-dataset correctness.

Detailed machine-readable inventories, migration code, source decisions and
fresh synthesis configurations live in:

- [Production repair inventory](analysis/country-production-repair-20261016/).
- [Source decisions and review holds](analysis/country-policy-migration-20261016/README.md).
- [Policy decisions](kie-real-baseline-label-policy-2026-10-05.md).

The input dataset is R26, including current 7,000 synthetics and the retained
older UAE sidecar. Together they contain **8,000 distinct synthetic documents**;
overlapping records must receive exactly the same repair, not independent draws.
Real training and validation are audited separately. Current direct decoder
targets remain direct objects, without adding a `documentPatch`/version wrapper.

### Confirmed migration edge cases

- Repeated source occurrences retain their distinct ownership and coordinate
  suffixes. Already-retired redundant country anchors require independent
  current postal evidence; they do not authorize reintroducing deleted text.
- Country words inside locality names, e.g. `SANTIAGO DE CUBA`, are protected.
  A shared city/country word such as `DJIBOUTI` stays intact when one occurrence
  serves both roles; it is not replaced by a country code or duplicated.
- One actual generated address used the shipper's `FALKLAND ISLANDS (MALVINAS)`
  as the final country of a Djibouti consignee/notify address. Sampled geography
  and explicit current target identity establish the repair. Both printed text
  and corresponding address targets require the same country-span correction.
- `RÉUNION, FRANCE` in two addresses is legitimate territory/parent wording.
  Réunion is independently printed; France is preserved. This is not the
  preceding defect and must not trigger a broad “two countries” rejection.

## Validation results

The new dataset is
[`mpci-bl-real600-synthetic7000-country-presentations-r27-positions-v1`](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic7000-country-presentations-r27-positions-v1/README.md).
R26 is preserved as the backup. No records were removed: 7,000 current synthetic,
600 real training and 60 validation rows remain, with the 2,500 older-UAE and
1,500 overlapping reinforcement sidecars retained consistently.

| Measurement | Result |
| --- | ---: |
| Unique synthetic documents migrated, including sidecars | 8,000 |
| All unique records independently verified | 8,660 |
| Additional overlapping copies verified | 3,000 |
| Changed raw country spans | 11,560 |
| Country target values normalized | 1,436 |
| Linked postal targets checked | 21,278 |
| Changed postal targets | 6,550 |
| Document-level alias opportunities | 2,384 / 8,000 (29.8%) |
| Current synthetic rows changed | 2,518 / 7,000 |
| Real training rows changed, country fields only | 162 / 600 |
| Validation rows changed, country fields only | 12 / 60 |
| Final dataset country holds | 0 |

Every coordinate suffix and physical line count was preserved. Real/validation
OCR, postal strings and non-country labels are byte-for-byte unchanged. Earlier
phone/container-ID formats, DG wording, descriptions, identifiers, quantities and
other targets were protected. The independent reader verifies forward/inverse
edits and all sidecar copies. A second root implementation reopened all 11,660
stored rows and independently checked membership/order, normalized countries,
permitted target differences, identical coordinate suffixes and frozen R26 pins.

Inventories:

- [Full migration report](analysis/country-production-repair-20261016/migration-report.md).
- [87-chunk inventory](analysis/country-production-repair-20261016/migration-validation-inventory.md).
- [Verified presentation examples](analysis/country-production-repair-20261016/migration-presentation-examples.md).
- [Root readback/publication](analysis/country-production-repair-20261016/root-publication.json).

### Source and production-path validation

Exactly **127 country-label corrections across 100 templates** were promoted.
All 413 native source no-op renders, schema checks and dependency checks pass.
No source OCR, source coordinates, alignment bytes or contract-variable wording
was changed. All 29 active configurations pin the registry, policy and refreshed
source authority explicitly.

Two existing templates had route-country facts incorrectly coupled to unrelated
party postals. Their actual qualified route declarations now own those facts:
one CIP location and one repeated marks/route location. Two exact source no-ops
and four destination-versus-third-country controls pass; unrelated YAML bytes
were independently recovered by reversing only these approved changes.

The native adapter also handles split `NAME(CODE)` owners, source-required code
captions, atomic parenthesized names, city/company-name containment, repeated
occurrences, multiline country slots and explicit non-varying literal targets.
A country inside an unchanged, jointly owned freetime instruction stays printed
as before; other country occurrences remain variable. The canonical postal
validator now counts a valid host-owned `NAME(CODE)` as one declaration, while
still rejecting genuinely repeated independent names/codes.

Fresh validation generated **12 samples from six templates**: four ambient,
two DG, two thermal and four mixed-container examples. All underwent native
render/replay, full rendered-text/label model review, root adjudication, plain
publication and measured-anchor positional processing. Their 56 country targets
are canonical. All six presentation families occur. Coordinates are in bounds:
1,042 known lines and 144 explicitly unpositioned lines (**87.9% of suffixed
lines**). No coordinate-reflow rejection was recorded. Existing conservative
group holds remain explicit where source anchors do not support reflow; they
are not invented positions.

[Fresh source/rendered pairs and positional samples](analysis/country-production-repair-20261016/final-fresh-validation.md)
and [machine-readable validation](analysis/country-production-repair-20261016/final-fresh-validation.json)
include the final hashes. These inspection examples were not added to training.
Existing postal guards caught duplicate locality/country generation; exact
manual/correction receipts preserve their rejected forms. One incidental
per-drum fill/gross arithmetic finding was adjudicated as outside the user's
extraction-defect scope: the description and declared-value labels agree, and
there is no missing separately printed net total. It is not a claim of physical
shipping-arithmetic validity.

The source-authority promotion initially omitted inherited geometry dependency
pins. Native positional execution caught this before completion. The builder
now preserves **all inherited dependencies**, with omitted-pin and blind-repin
negative tests. Actual parquet hashes match the original snapshots exactly;
all 704 source alignments, 1,367 pages and 123,234 boxes match the parent geometry.
See [contract completion](analysis/country-production-repair-20261016/contracts-completion.md).

Validation totals: **831 project tests**, **46 additional migration/contract
tests**, Ruff and diff checks pass. The rebuilt training image imports installed
code from `/opt/document-ocr/src`, resolves the pinned registry, validates
normalization and binds the R27 direct task constraints. It accepts synthetic,
real and validation example targets. This was CPU-only, network-disabled and
did not load a model. Synthesis itself runs in the project environment; its
agent dependencies are intentionally not added to the training-only image.

### Performance and cost

- Registry load versus the isolated probe: **0.313 → 0.060 s**; 498,000 hot
  lookup/span operations: **0.486 → 0.477 s**; retained Python memory
  **6.40 → 4.65 MiB**.
- Actual production-document native renderer, seven batches of 100:
  **1.266 ms** without country projection, **1.656 ms** with a prepared plan,
  **1.878 ms** including preparation. The added **0.613 ms/document** is
  negligible beside measured multi-second lexical generation. This is feature
  overhead, not a claim that native rendering itself became faster. Peak traced
  allocation: **126,291 → 132,937 bytes**.
- Migration: **131.08 s / 660,472 KiB peak RSS**; separate full readback:
  **17.80 s / 396,800 KiB**. Both are streaming/chunked; migration has no paid
  requests or GPU work.
- Complete 704-source geometry load: prior **1.479 s**, promoted **1.415 s**;
  no slowdown observed (cache/noise applies).
- Fresh generation/review, including preserved first-pass/retry calls:
  **$0.0150105**, 76,910 input and 6,948 output tokens; 2,500 output tokens were
  reasoning tokens. Twenty paid call receipts are retained. Cached replays are
  not counted twice.

### Source edge cases and subsequent user adjudication

One **original source**, `c7060e22`, remains unchanged: its notify country is
printed as undotted `ARE` and the target says Egypt. **The user approved retaining
that target on 2026-10-16; its original-source review hold is now closed.** This
is a document-specific interpretation, not a global ARE-to-Egypt alias. It is not in real
training or validation. Its six synthetic descendants independently print their
sampled countries and all pass; they are not held. The original source is still
selected as a layout scaffold by two existing DG configurations; future descendants remain
subject to actual-render country ownership/replay checks.

Review [source OCR](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_c7060e220634333eba11aecef9d3a01cbad4a0d51a2a9ec6c7fc27db2b2bf170/source.txt:15)
and [PDF](/mnt/d/projects/documentparsing/data/corpora/paddle-bl-swb-filtered-max5/files/e4cde1eb8292da5bfebc06fb057ce897d984fd5525a52c9658cc21160e33a8ff.pdf).
No ambiguous country field was silently deleted.

The country sweep separately found five pre-existing `ebb856c7` descendants
with repeated package-declaration wording already present in the original OCR.
The user approved retaining that source-faithful presentation on 2026-10-16;
optional cleanup is not a required label repair. Their description targets remain
clean product wording; no target contradiction was established by this country
audit. They remain unchanged and explicitly inventoried in the migration report.
This bounded country certification does not claim a fresh semantic audit of
every non-country field in the entire dataset.

Both accepted cases, exact affected records and any future review items are
tracked in the [edge-case inventory](kie-dataset-edge-case-inventory.md).
