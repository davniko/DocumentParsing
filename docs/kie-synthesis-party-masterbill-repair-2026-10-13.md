# Delivery-agent identity and master-B/L repair — 2026-10-13

## Outcome and dataset authority

The two requested repairs are complete across the active synthetic dataset and
the previous UAE cohort. The source/template owners and production synthesis
paths are corrected and verified with fresh synthesis, not just existing-row
edits. No training was launched or running training data modified.

The new working copy is
[R20](../data/curated/mpci-bl-real600-synthetic5000-uae-coverage-r20-party-masterbill-positions-v1/README.md).
It contains:

| Cohort | Documents inspected | Master-B/L repairs | Delivery-company repairs | Training membership |
| --- | ---: | ---: | ---: | --- |
| Retained older non-UAE synthetic | 2,500 | 10 | 0 | Retained |
| Current replacement UAE synthetic | 2,500 | 24 | 1 | Retained |
| Previous superseded UAE synthetic | 2,500 | 25 | 0 | Separate repaired sidecar |
| Real training | 600 | 0 | 0 | Retained, byte-identical |
| Real validation | 60 | 0 | 0 | Retained, byte-identical |

There are **60 distinct repaired synthetic records**, not 60 additional records.
The active training set is still 600 real + 5,000 synthetic; validation is still
60 real. The repaired earlier UAE records are in
`previous-uae-synthetic.jsonl`; they are deliberately **not** appended to training.

The active parent R19 dataset and the earlier R17 authority remain unchanged.
R19's `supersededNewerUAECohort.orderedDocumentIds` identifies the exact earlier
2,500 UAE rows. Its source-file hash was verified before selecting them.
Unchanged JSONL records were copied byte-for-byte, not reserialized. Training,
synthetic and replacement mirrors agree on every shared document.

This closes these two repair items. It does **not** claim that unrelated tax IDs,
private footer contacts or other secondary organizations from the broader
[anonymization inventory](kie-synthetic-anonymization-audit-2026-10-13.md) are
already anonymized; those are separate remaining tasks.

## 1. Delivery agent: diagnosis, correction and recurrence prevention

### What was wrong

The affected family is
`doc_53350e31d8881f3d2986981eea94e602e9e4c52daa615dcb2666170524ab5e98`.
The single affected descendant is `syn_full_v7_f712c5877dbeabef9e5273ec`.

Its printed text and target agreed on:

```text
ARABIAN GULF MARINE TRADING COMPANY
(SHARJAH OFFICE)
```

Thus this was **not** a label/OCR mismatch or an uncompiled delivery-agent
block. The company-name owner was already mutable and correctly bound to
`documentPatch.parties.deliveryAgent.name`. The generator retained the source
company core and changed only the office qualifier. The old copied-name check
compared the entire string, so this variant escaped it.

### Current record repair

GLM-5.3-Flash generated a new company core:

```text
AL NOOMAN LOGISTICS SERVICES LLC
(SHARJAH OFFICE)
```

Its owned email was generated in the existing contact-generation path using
the new company context. Both the rendered raw OCR, position-enriched OCR and
corresponding name/email targets were updated. Postal address, country, telephone,
office qualifier and all other targets remain unchanged.

There is one printed company-core occurrence and one printed email occurrence
in this record. The physical line/page structure and every coordinate suffix
are unchanged. At the same measured font, the replacement company core is
87.65% of the old width, and the email is 97.72%; neither expands the previously
admitted line footprint. This is a scoped text replacement, not new geometry.

The family has 23 retained/archived descendants: nine older non-UAE, nine current
UAE and five previous UAE. Only one needed this repair; the other 22 were not
rewritten. All were checked for remaining occurrences of the original company
core, including text outside the label.

### Template and production changes

- The source contract's `v002` meaning now clearly specifies a genuinely new
  organization identity, not a changed branch/office of the original company.
- The active ownership declaration explicitly certifies `v002` as an
  `organization`. Existing name, postal and contact bindings stay intact.
- `curated_wording.py` communicates the same rule in the native field description
  and concise generation prompt. The rule applies to party-name generation,
  not only this source ID or the delivery-agent role.
- `identity_key()` compares the complete normalized identity core independently
  of parenthetical qualifiers. An unchanged core with a changed office now
  fails `validate_wording()` before rendering/publication.
- The check is an equality check, not a common-word substring blacklist.
  Genuinely different companies sharing words such as `FREIGHT SOLUTIONS` remain
  admissible. An entirely parenthesized name is still compared as a name rather
  than becoming an empty key.

The broader post-repair census checked **2,703 synthetic delivery-agent names**:
1,782 active and 921 previous UAE. None retains a source-equivalent core under
this check. This establishes coverage for the detected core-copy failure; it is
not a general guarantee that every possible partial identity resemblance is
detectable by string equality.

## 2. Master B/L: diagnosis, correction and recurrence prevention

### What was wrong

The family is
`doc_e827ecdd1cb1a45de52691113a4fc9d9a5e54ca9c4e60f16e35dbfca8a599671`.
The source prints the same identifier twice:

```text
--- PAGE 1 ---
BOOKING NUMBER ML250031
...
--- PAGE 2 ---
MASTERBILLNO: ML250031
```

The structured `masterBillOfLadingNumber` was present and matched the text.
However, `public_identity_updates()` sampled ordinary B/L and voyage values,
but omitted master B/L. The source contract also lacked an explicit shared
host variable for these two printed references. Consequently the original
`ML250031` survived in every descendant of this family.

This is identifier anonymization and binding coverage, not an instruction to
delete a supported master-B/L target or reinterpret the source reference.

### All existing affected descendants

All **59** descendants were repaired: ten retained non-UAE, 24 current UAE,
25 previous UAE. Each receives its own deterministic identifier with the same
letter/digit/separator pattern as the source. The booking occurrence, the
second-page master occurrence and the structured master target share that
single draw.

The independent checker confirms exactly two occurrences, one on each page,
that the old value is absent, and that all 59 new identifiers are distinct.
Other B/L numbers, container identifiers, seals, quantities and targets were
not changed. Every positional suffix and the line/page structure are preserved.

### Template and production changes

- The source contract now declares `master_bill`, kind `identifier`, with both
  source occurrences and target expression
  `documentPatch.masterBillOfLadingNumber = {master_bill}`.
- Compilation merges the old booking and master historical surfaces under this
  common owner; no second independent draw can disagree with it.
- `curated_campaign.py` uses one document-identifier suffix set in host target
  sampling and host-variable propagation: ordinary, master and original B/L,
  plus voyage number. The existing format-preserving deterministic generator
  is reused; no LLM call is needed for master identifiers.
- When several identifier fields genuinely share an old identity, the existing
  identity map reuses one new value. Separate old identities remain independent.

Three real training documents have master-B/L labels; none in validation does.
Their original real OCR/labels are correct and were left intact. Only this one
master-bearing source is in the active template catalog. Another real record
has an original-B/L label but is not an active template; it too is unchanged.
Source text and real targets remain provenance, not anonymized synthetic output.

## 3. Validation and falsification

### Complete dataset comparison

[Independent dataset validation](analysis/synthesis-party-masterbill-repair-20261013/dataset-validation.json)
compares every output with its original, reconstructing the permitted changes
without invoking the repair function. It checks:

1. Exactly the approved target paths change: master identifier, or delivery-agent
   name and owned email. All other values and metadata remain equal.
2. Full raw and positioned text equal the original plus those exact replacements;
   there are no unrelated deletions, additions or reorderings.
3. All emitted rows satisfy the current canonical reduced schema. Raw and
   positioned input hashes match their text.
4. Position suffixes, line counts, page markers and text preservation agree.
5. Training/synthetic/replacement mirrors agree. Previous UAE IDs are disjoint
   from active synthetic IDs and are not reintroduced into training.
6. Parent files and the previous UAE source file retain their original hashes.
7. There are no residual original master references in the repaired family,
   and no remaining source-core copies in the delivery-agent name census.

Seven deliberately corrupted real repaired records were all rejected:

- Leave one booking/master occurrence unchanged.
- Change only the master label, disagreeing with both printed references.
- Remove the supported master label.
- Change the company label but retain the original printed company.
- Change the email label but retain the original printed email.
- Alter a coordinate suffix.
- Mutate unrelated negotiability.

This matters: the checker is tested against the failures it claims to reject,
not merely observed returning success on the new dataset.

### All template contracts

[Registry validation](analysis/synthesis-party-masterbill-repair-20261013/registry-validation.json)
passes for all **360 sources**, **2,160 case files**, four shared declaration
files and **25,892 compiled regions**. Current source identity contracts replay
without unresolved ownership overlaps; case and shared-file pins agree with
the active manifest. Only the two specified source contracts and the delivery
identity-kind declaration were revised. Source OCR and real gold are unchanged.

### Fresh ordinary synthesis: four descendants

Both repaired templates generated **two new descendants each** through ordinary
production sampling, native structured wording, company contacts, rendering,
full rendered-text review, publication and coordinate processing. All four
had zero semantic-review findings. The root agent also opened and read every
complete rendered text and target; no Codex subagents were used.

The delivery-agent descendants contain genuinely new company cores, owned
postal/contact fields and matching targets. The master descendants print
`OY623886` and `ML907754`, respectively, identically in booking and master
positions and in their targets; neither prints the original `ML250031`.

[Pilot certification](analysis/synthesis-party-masterbill-repair-20261013/pilot-certification.json)
includes exact candidate replay, native-review provenance and independent
geometry checking. Six pages pass; 266 source/page rectangles are in bounds,
four generated blocks and ten generated centroids are checked. Six inherited
source rectangle overlaps are explicitly identified as source overlaps, not
new reflow collisions. There are **zero rejected reflow pages** and no unresolved
review findings. Coordinate availability is 132/166 lines for the delivery
family and 110/170 for the master family; unavailable anchors remain explicitly
unpositioned rather than invented. Position coverage was not changed as part
of this privacy task. These four inspection descendants were not added to training.

Inspection galleries:

- [Delivery-agent source and fresh descendants](analysis/synthesis-party-masterbill-repair-20261013/upstream-pilot/53350e31/samples.md).
- [Master-B/L source and fresh descendants](analysis/synthesis-party-masterbill-repair-20261013/upstream-pilot/e827ecdd/samples.md).

### Automated tests and historical caches

The relevant suite passes **1,271 tests** in 47.52 seconds:

```bash
TMPDIR=/tmp .venv/bin/python -m pytest tests/test_curated*.py tests/test_synthesis_curated.py -q
```

Twelve new targeted tests cover qualifier-only identity copies, genuine new
names/common words, native field descriptions, shared identifier draws,
format preservation and two-page master rendering.

The first broad test run exposed ten historical-replay fixture failures, all
at the expected stale-request-hash guard after changing native field
descriptions. The read-only historical fixture now binds its archived field
descriptions as well as its archived system/user prompt. It does **not** rewrite
old receipts or bypass production cache validation. A separate test confirms
that current instructions still reject the historical cache. The complete
rerun passes. Ruff checks and formatting checks pass; `git diff --check` passes.

Existing generation artifacts and their old certificates are retained as
historical provenance. They were not rewritten to claim generation under the
new rules. New campaigns use the current instruction/schema hashes; stale
wording caches are rejected and must follow ordinary regeneration/revalidation.

## 4. Cost, performance and side effects

Total provider-reported LLM cost for the replacement company/email and all
fresh generation/contact/review calls is **$0.00405525**:

| Operation | Cost USD |
| --- | ---: |
| Existing-record company and email | 0.00015180 |
| Two fresh delivery-template descendants, contacts and review | 0.00144205 |
| Two fresh master-template descendants, contacts and review | 0.00246140 |
| Rejected rate-limited calls | 0.00000000 |

The company result was reused rather than regenerated when providers were
rate-limited. Fireworks and DeepInfra rejected email calls with zero charge;
Together completed the email and fresh validation calls. These were explicit
provider choices, not silent model/provider fallback. All use the same
GLM-5.3-Flash model, low reasoning and native structured output. Provider support
and prices were checked against the official
[OpenRouter endpoint metadata](https://openrouter.ai/api/v1/models/z-ai/glm-5.3-flash/endpoints)
and saved in the audit directory. No stronger model or training job was launched.

[Microbenchmarks](analysis/synthesis-party-masterbill-repair-20261013/benchmark.json):

| Local operation | Before | After | Interpretation |
| --- | ---: | ---: | --- |
| Wording validation | 11.55 µs | 11.31 µs | No measured slowdown |
| Host identity updates | 27.98 µs | 30.40 µs | Adds the formerly omitted master update; +2.42 µs per call |

Medians use five repetitions, 3,000 wording checks or 1,000 host-update calls
per repetition. The extra host work is negligible relative to generation;
no new blocking call or expensive source scan was added to the production hot
path. The complete independent dataset audit used approximately 147 MiB peak
RSS in the measured invocation; its final expanded census took 30.46 seconds
inside the audit function. Fresh publication/geometry verification used
approximately 665 MiB peak RSS and 23.73 seconds including interpreter/import
startup. Neither uses GPU or loads a training model.

No source-country policy, routes, goods, package allocation, equipment alias,
negotiability, casing policy or field schema was changed. The only new LLM
repair is the one company/email pair. Format-preserving identifier repair is
deterministic. Current training input/target counts and validation membership
are unchanged. Other private footer/reference categories remain the next
anonymization work items, not hidden side effects of this patch.

## 5. Reproducible evidence

The scoped repair/audit directory is
`docs/analysis/synthesis-party-masterbill-repair-20261013/`:

- `before-registry/`: original manifest/declarations and both affected cases.
- `party-wording.json`: replacement company/email, preserved fields and cost.
- `dataset-revision.json`: exact input/target edit inventory and protected hashes.
- `dataset-validation.json`: independent all-row checks and negative controls.
- `registry-revision.json`, `registry-validation.json`: changed owners and all-case replay.
- `pilot-configs/`, `upstream-pilot/`: actual generation inputs, calls, candidates,
  native review receipts, publications and geometry receipts.
- `pilot-certification.json`: completed production-path validation.
- `completion-certificate.json`: final pinned working-copy and implementation evidence.

The repair-copy script refuses to overwrite an existing R20 directory. The
validation scripts are read-only with respect to dataset rows and source
authorities; they can be rerun against this working copy.
