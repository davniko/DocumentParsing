# Real versus synthetic presentation: audit, evidence and proposed corrections

Follow-up: the phone-format implementation and dataset repair are documented in
[Phone presentation repair](kie-phone-presentation-repair-2026-10-15.md).
The remaining format investigations below are not thereby marked implemented.

Date: 2026-10-15. This consolidates the successive real-data-addback and
presentation audits and the subsequent questions about phones, equipment,
countries and goods descriptions. It is an **investigation and decision record**,
not publication of repaired data. No training, dataset, template, label,
coordinate or production-code changes were made for this audit. Offline
prototypes described below are not integrated features.

Main conclusions:

- Phone formats are a manageable compositional vocabulary, not a single
  international format or hundreds of unrelated hardcoded masks. Preserve the
  current printed-value target policy; separate identity from presentation.
- Container labels are correctly canonical, but input rendering discards known
  source separators. A reversible per-occurrence prototype passes 808 occurrences;
  integration must cover both normal variables and explicit surface overrides.
- Seals mostly already retain formatting. Three repeated-alias homogenizations
  are not established extraction-label defects.
- Country targets intentionally differ by role. The deeper audit finds 18
  current party-country policy violations in addition to missing abbreviation
  coverage; these are separate from the 18 duplicated-plus phone samples.
- Goods verbosity is a genuine historical distribution difference, not proven
  harmless and not a newly introduced cause of this run's regression. The
  examples below distinguish free elaboration, many sampled identities and
  host-controlled chemical wording.
- Real-data benefit extends to unseen phone values and fragmented IDs. That
  supports preserving real OCR structure in future surgical anonymization,
  without claiming that every benefit or privacy question is already isolated.

## 1. Scope and reproducibility

The current R23 training run reads exactly:

| Input | Records | SHA-256 |
|---|---:|---|
| [Synthetic shard](../data/curated/mpci-bl-synthetic7000-old-uae-reinforcement-r22-positions-v1/train.jsonl) | 7,000 | `90b0be46d6b0387dd270b7f2d0a56264ad4515f596c3e08e6df1e2478e6f8939` |
| [Real training](../data/curated/mpci-bl-real600-synthetic7000-hs-preserved-r23-positions-v1/real-train.jsonl) | 600 | `c1e5251084c8ba03db7bfbc5a25e4c63c23523894bf7f47c77a3c96489596e05` |
| [Real validation](../data/curated/mpci-bl-real600-synthetic7000-hs-preserved-r23-positions-v1/validation.jsonl) | 60 | `5b0bdccba39158a97f75b5e478e297c8cc9dd58dd0018ee130d0c028dc6ae30f` |

These three live files were checked against the actual saved R23 resolved
configuration, not merely the intended launch YAML. The active
[template registry](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/manifest.json)
has 413 source cases. Source-format counts use compiled ownership, exact source
occurrences and auxiliary declarations; unrestricted matches elsewhere on a page
are not substituted for a field's owner.

“Real” means the retained GLM OCR and its reviewed labels, not a reconstruction
from PDF. “Synthetic” here means the current 7,000-row shard, not an obsolete
campaign export. Whole-document semantic certification is not inferred from a
format-counting screen.

Related records:

- [Training result ledger](mpci-training-results-and-next-experiments.md).
- [Real-data addback and HS preservation](kie-real-addback-and-hs-preservation-2026-10-15.md).
- [HS generation and full-width repair](kie-hs-dg-format-repair-2026-10-14.md).
- [Baseline and later label policies](kie-real-baseline-label-policy-2026-10-05.md).
- [Old-UAE reinforcement and the 7,000-row assembly](kie-old-uae-reinforcement-and-synthetic7000-2026-10-14.md).

## 2. Interpretation rules and correction to the historical comparison

### 2.1 Historical HS scores must retain their own contract

The user correctly rejected using today's separator-preserving HS targets to
grade older models as though their task had not changed. Older synthesis mainly
taught six-digit HS values and targets without separators. Current synthesis
preserves full source widths and printed punctuation. A replay against the new
contract is a **contract-compatibility diagnostic**, not a fair headline model
quality comparison. Earlier language suggesting that such rescoring explained
away part of the performance deficit is withdrawn.

Use historical native scores for the historical task. For a cross-contract
performance comparison, use verified unchanged components, explicitly excluding
changed HS expectations. Do not quietly normalize the new gold back to the old
contract either.

### 2.2 The observed recovery is real, but not an isolated causal experiment

First scheduled evaluations, approximately epoch 2.5:

| Run | Training composition | Eval loss | Field F1 | Field accuracy | Relation F1 | Category F1 | JSON / schema valid |
|---|---|---:|---:|---:|---:|---:|---:|
| J | 600 real + 5,000 older synthetic | .05362 | .8882 | .8421 | .8878 | .8658 | 59 / 48 |
| K | 600 real + 5,000 replacement-coverage synthetic | .05324 | .8878 | .8374 | .8768 | .9081 | 58 / 49 |
| R22 | 7,000 synthetic only | .13870 | .8135 | .7496 | .8889 | .8419 | 55 / 46 |
| R23 | Same 7,000 synthetic + 600 real, HS-preserved reference | .06124 | .8727 | .8147 | .9104 | .8797 | 55 / 46 |

R22 to R23 lowers loss by 55.85% and raises field F1 by 5.918 percentage points,
without increasing the number of parseable/schema-valid documents. Under each
run's native reference the correct-field counts are 2,009 to 2,163, with wrong or
extra values decreasing 354 to 218. The reference HS strings do differ; these
counts are not falsely presented as a perfectly controlled common-gold trial.

Saved runtime/source fingerprints match between R22 and R23. Rank 48, alpha 32,
rsLoRA, LR 1e-4, ScheduleFree AdamW, microbatch 1/accumulation 32, input-only
positioned OCR and ten epochs are unchanged. There is no observed truncation or
dataset-hash failure. However, data-dependent EVA initialization, batch order,
updates per epoch, real-example sampling share and HS reference expectations
change. They prevent a unique causal attribution from aggregate scores alone.

R23's first-evaluation generated strings were not persisted. The metric callback
discards individual assessments and the runtime publishes predictions after
training. Consequently the current run's first-eval phone/description errors
cannot be enumerated without another inference pass. The detailed model-error
audits below instead use **completed K and R22 final predictions**.

Loss below .06 at epoch 6 is not a justified hard acceptance gate: successful K
had approximately .0653 at epochs 5 and 7.5 and later improved field extraction.
Teacher-forced loss and extraction accuracy are different measurements.

### 2.3 Unchanged-fact comparison used below

The K and R22 references contain identical non-HS document facts for all 60
validation identities. We reproduce their native full-field scores from their
saved predictions and references, then exclude HS facts from comparisons where
necessary. We do not replace either model's expected HS strings with new ones.

On the same 57 documents where both final outputs parse, unchanged non-HS facts
give K 2,151 correct versus R22 2,035 out of 2,390 expected. Non-HS F1 is .905684
versus .859375. This is a conditional diagnostic, not the complete 60-document
score or a result for R23.

Phone totals below use the native list-aligned field projection. Earlier
role-set summaries counted 94 K phone matches; the native aligned projection
counts 96. These projections must not be mixed without identifying them.

## 3. Phones: existing policy, finite format families and lost variability

### 3.1 Target policy and form restrictions

[ContactsV7.phoneNumbers](../src/document_ocr/label_schemas/bill_of_lading_v7.py#L92)
asks for distinct party-owned numbers, preserving spelling and printed prefixes.
Standalone abbreviated suffixes/extensions are not independent complete numbers
and are not expanded by guessing missing digits. Physical OCR line breaks can
be joined as required for a single string; that does not authorize adding `+`,
dropping a printed `00`, or inventing a country calling code.

The saved [MPCI form schema](../artifacts/mpci-ai-schema/mpci-cuscar-canonical.schema.json#L120)
represents telephone communication as `communicationMeans: "TE"` and an ASCII
`identifier`. It does not require E.164. Common ASCII punctuation, spaces,
parentheses, slash and plus are not forbidden by that snapshot. The extraction
field is `ApplicationText`, not a telephone parser demanding international form.

**Recommendation for discussion:** keep extraction source-faithful rather than
change the task to E.164 normalization. If a normalized dialable number is useful
for the platform, derive it separately when enough country/dialing context exists.
That avoids turning an extraction field into a guessing task. Extension/range
policy needs explicit treatment; do not silently strip or expand the compounds
described below.

### 3.2 Are there a limited number of formats?

There are a small number of composable families, not a tiny list of fixed masks.
Counts below are over 1,110 real-training and 116 validation phone values:

| Family | Real training | Validation |
|---|---:|---:|
| Literal leading `+` | 469 | 42 |
| Parenthesized leading `+` | 36 | 5 |
| Leading `00` | 168 | 26 |
| National leading zero | 154 | 19 |
| Bare digits | 238 | 18 |
| Parenthesized bare prefix | 36 | 6 |
| Parenthesized national zero | 6 | 0 |
| Other, e.g. `{+}` | 3 | 0 |
| Total | 1,110 | 116 |

The “literal `+`” versus “parenthesized `+`” distinction explains previous counts
of 42 versus 47 validation phones: the latter combines those two families. It is
not a change in dataset membership.

After folding whitespace and replacing digit runs with their widths, real
training has 265 masks and validation 60. Replacing runs with a width-independent
`D` leaves 99 and 23 punctuation shapes. The ten most common width-independent
shapes cover 74.32% of real values and 83.62% of validation; the top twenty cover
86.76% and 97.41%; the top thirty cover 90.99% and 100%.

Those rankings are computed **within each cohort**. They do not mean a selected
twenty-mask training list automatically covers 97.41% of validation. A separate
train-ranked transfer check gives 86/116 validation values for the ten most
common training shapes, 109/116 for twenty, and 109/116 for thirty. All 99 training
shapes cover 110/116. Six validation values have previously unseen punctuation
combinations; 19 validation width-specific masks, affecting 22 values, are absent
from training. This supports a compositional grammar rather than an exhaustive
allowlist of observed strings. Sampling choices should be based on training
coverage; the holdout is used here to diagnose transfer, not select its targets.

For comparison, synthesis uses 61 exact-width masks but just 12
width-independent shapes, all with a literal leading plus. Some frequent real
shapes illustrate why mask count alone is not coverage:

| Shape (`D` = one digit run of variable length) | Real training | Validation | Synthetic |
|---|---:|---:|---:|
| `D` | 270 | 29 | 0 |
| `+D` | 154 | 19 | 0 |
| `D-D-D` | 67 | 13 | 0 |
| `+D D D D` | 69 | 6 | 8,296 |
| `+D D D` | 59 | 3 | 2,044 |
| `(D-D)D` | 17 | 6 | 0 |

Prefix family is an independent axis: a single `D` run may begin `00`, national
zero, or neither. These shape counts do not normalize those prefixes away.

Thus a reusable grammar of prefix, digit groups, separators and optional
parentheses is appropriate. Hardcoding all 265 width masks would be brittle;
calling all numbers one international format is inadequate.

Real values also contain 445 space-bearing, 261 hyphen-bearing, 111
parenthesized, 27 slash-bearing, six dotted and two comma-bearing presentations.
These features overlap. There are 18 extension-token values and other letter
compounds. In total 49 real values have letters, slash or comma requiring
deliberate handling, not an automatic “invalid label” verdict. Validation has two
slash compounds. A punctuation-shape screen does not decide whether several
printed fragments constitute one number, several complete numbers, or a suffix.

### 3.3 What is lost and where?

The actual active source ownership contains 735 phone target paths in 307
templates, 702 logical owners and 792 physical regions. All phone paths have
owners. Sixty-seven owners repeat across 45 templates; four repeat with different
presentations. Source-owned regions include `00`, national and bare forms,
parentheses, slash, dots and multiline values.

Current synthesis has **12,663 phone targets in 5,158 documents**. Every target
uses a literal `+` and spaces. There are no `00`, national or bare-prefix targets,
no parentheses or slash; 1,388 have country-specific hyphens.

Cause:

1. [public_identity_updates](../src/document_ocr/synthesis/curated_campaign.py#L392)
   asks the deterministic contact generator for a country-valid number.
2. [phone()](../src/document_ocr/synthesis/template_compiler/contact_values.py#L298)
   returns `PhoneNumberFormat.INTERNATIONAL` every time.
3. Rendering inserts that supplied presentation rather than reconstructing the
   source dialing/punctuation style.

Examples from existing descendants:

| Source-owned form | Generated form | Descendant |
|---|---|---|
| `010 62 777 800` | `+679 701 3291` | `syn_full_v7_e4496821430b6cff4ce9da36` |
| `00201270722232` | `+973 3600 3515` | `syn_full_v7_10035dd7e686581ca64f4e12` |
| `(+202) 33351271` | `+242 06 123 0923` | `syn_full_v7_b51657d299c6ff5fd2a2fdd8` |

These synthetic values generally agree with their rendered labels. The problem
is absent presentation/normalization supervision, not automatically wrong labels.

Multiline input has not disappeared. One source's `+2\n02 2269 3193` becomes
`+31\n6 12342444`. Do not infer input-line absence from joined phone targets.

### 3.4 What the models actually do

On the matched 57 final-output documents:

| Expected presentation | Gold phone fields | K correct | R22 correct |
|---|---:|---:|---:|
| Plain leading `+` | 42 | 39 | 36 |
| All other printed forms | 64 | 57 | 0 |
| Total | 106 | 96 | 36 |

All 84 R22 phone predictions start with `+`, versus 41 of K's 104 predictions.
Some R22 values start `+0020` or `+086`: this is not evidence of a valid E.164
normalizer. The model strongly reproduces a synthetic output prior.

After exact role-scoped matching of the native aligned fields, R22's 106 expected
and 84 predicted phone fields decompose as follows:

| Category | Expected fields | Predicted values |
|---|---:|---:|
| Exact matches | 36 | 36 |
| Same digits, changed presentation | 20 | 20 |
| Remaining wrong/partial numbers | 22 | 22 |
| Missing after those matches | 28 | 0 |
| Extra after those matches | 0 | 6 |

Pairing the remaining wrong numbers is an operational error taxonomy, not proof
of a unique generation mechanism. The initial categories are seven partial
strings, five one/two-digit changes, three extra-prefix/digit cases and seven
different numbers. Some repeat another phone rather than invent a completely new
number. One clear transfer puts a delivery-agent number under notify. Of the 28
otherwise missing numbers, one appears under another party and 27 are absent
from all predicted phone roles. Thus the regression is not mostly role movement.

Concrete OCR-conditioned examples:

- Source `d8363a4d`: consignee/notify OCR has `PHONE & FAX 002` followed by
  `02 2575 2002`; gold `002 02 2575 2002`, R22 `+92 02 2575 2002`. This is more than
  punctuation normalization: a leading digit changes too.
- Source `3b5b0c08`: consignee/notify prints `0020-2-24718492`; R22 gives
  `+96310`, not printed in the reviewed party blocks. It is a partial/wrong value,
  not an acceptable international rendering of the source number.
- Source `661601`: `00201223949046 / 1228802800` yields duplicated/truncated
  versions of the first number and loss of the second. Distinct printed complete
  numbers must remain distinct targets; slash is not permission to combine them.

Source links:
[d8363a4d OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_d8363a4deef9dea2b76d716b8621ed6f7a7e2694e9ce9b7ac0260f2db3a921fd/ocr.txt#L23),
[3b5b0c08 OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_3b5b0c08ec599fae9a9c8c2b380fa793e087ef6dc7d2f3e95a420ebc06d53784/ocr.txt#L29),
[661601 OCR](../data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_661601d9795403830ecbb829fd5689642414479847eb0f54ff094c0b33c8a503/ocr.txt#L28).
Prediction strings are in
[K](../artifacts/kie-training/t5gemma2-270m-mpci-bl-real600-synthetic5000-uae-coverage-r19-positions-inputonly-v7-e10-compact-eva-a32-r48-local-sfadamw-v1/predictions/validation.jsonl)
and [R22](../artifacts/kie-training/t5gemma2-270m-mpci-bl-syn7000-old-uae-reinforcement-r22-positions-inputonly-v7-e10-compact-eva-a32-r48-local-sfadamw-v1/predictions/validation.jsonl).

These are reasons not to “fix” evaluation by broadly removing phone punctuation
or prefixes. Some model failures are genuinely incomplete extraction.

### 3.5 Proposed clean design and offline feasibility

Separate a sampled telephone identity from its printed presentation. Prefer a
source-conditioned presentation record per owned occurrence; allow configurable
family sampling when compatible, so a source need not permanently enforce only
one style. Repeated numbers share one identity but can retain occurrence-specific
forms. Derive the target using the chosen extraction policy, not the generator's
preferred international string.

Do not force a source's fixed digit grouping onto a different country's
incompatible-length number. Represent prefix and grouping semantics, use
compatible country/format combinations, and surface an unsupported combination
instead of silently truncating, padding or changing the requested style.
Standalone suffixes, extensions and number ranges need their own reviewed
ownership/composition treatment rather than the simple-number formatter.

A pure offline probe generated 120 valid numbers across EG, AE, CN, GB, US, DE,
FR, IN, JP, KR, BR and IT, then rendered nine variants: compact `+`, metadata
international, `00`, bare international, parenthesized plus, parenthesized calling
code, metadata national, hyphens and dots. All **1,080 presentations** round-tripped
to the same country-valid E.164 identity. This proves simple style/core
separation is feasible. It is not a production integration test and does not
certify extension/range handling or all source masks.

Reproducibility: for each of the twelve country codes, the probe drew ten numbers
using `contact_values.phone(DeterministicStream(20261015,
"offline-phone-style-audit", country + str(index)), country_code=country)`, with
`index` from zero through nine. The harness knew which presentation it had
constructed: it converted the known `00` and bare-international variants back
to `+` for parsing, and supplied the known region for national forms. It required
both `is_valid_number_for_region()` and identical E.164 serialization. This is
not a claim that arbitrary unknown OCR numbers can be normalized without context.
The installed `phonenumbers` version was not captured in the original probe.

### 3.6 Confirmed boundary defect: 18 doubled-plus samples

One source phone slot excludes its leading `+`, which belongs to the unchanged
literal frame. The current curated renderer inserts another `+`. The source is
[9ad49fcc, line 28](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_9ad49fcc846c53b1780f28a74c0cd1073d7bb2d812e5b5a76d2744b1ba4a7f02/source.txt#L28),
`deliveryAgent.contactDetails.phoneNumbers[1]`.

The owned fragment `20 3 481 4950 / EXT\n112` starts at source bytes 543–566.
Replacing it with `+380 50 123\n3066` leaves the existing literal plus untouched.
Example descendant `syn_full_v7_7b1c687a2cfad092c825844d` prints `,++380...` while
the target contains one plus. The
[candidate proof](../artifacts/kie-synthesis-production/curated-v7-expansion1000-v1/candidates/syn_full_v7_7b1c687a2cfad092c825844d.json)
records those exact edits.

All **18 current descendants** of this family are affected. A complete adjacent
`+`/`00` check over all 792 owned phone regions found this single problematic
boundary and no external-`00` counterparts. An existing
`phone_inside_literal_prefix()` helper protects an older renderer, but the
curated path does not invoke that protection. This is a real rendering defect,
distinct from the broader format-distribution gap. No repair was applied here.

## 4. Container IDs: correct canonical targets, unused source presentations

### 4.1 Precise meaning of the scope numbers

Of 413 cases, 405 contain 716 container target fields in 809 owned occurrences.
Fifty-two templates have **76 fields** whose source spelling includes spaces or
separators; these fields appear in **101 source occurrences**. A field can appear
several times, which is why the counts differ.

Those 52 templates underpin **864 current synthetic documents**, which contain
**1,457 corresponding container fields** and 2,187 weighted noncanonical
source-occurrence opportunities. None of those source-style spellings survives.
Every one of the 14,472 synthetic container targets has a contiguous copy
somewhere in rendered OCR.

The original formats have not vanished from the template assets: source text,
byte ranges, printed occurrences and historical format metadata retain them.
However, that is not the same thing as an active typed presentation contract.
The current renderer mostly treats a canonical sampled ID as ordinary text.

[ContainerInformationV7.equipmentIdentifier](../src/document_ocr/label_schemas/bill_of_lading_v7.py#L403)
correctly requires letters/digits including the check digit, stripping only
presentation separators. **Keep that target normalization.** The missing layer
is source-shaped input rendering.

### 4.2 Example and root cause

Source [5242647c, line 36](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_5242647c15b8d9847aef3c4321fb5d93df660576d1aaab7e6e4e2f261124de04/source.txt#L36):

```text
Source OCR:                 MCLU 507449-8
Source target:              MCLU5074498
Current synthetic OCR:      MCLU5688412
Current synthetic target:   MCLU5688412
Source-style synthetic OCR: MCLU 568841-2
Target would stay:          MCLU5688412
```

The current synthetic is `syn_full_v7_2ca7b8a04845376412e3faf7`, row 91 in the
current synthetic shard. Another source's `VMLU 4211677` becomes
`VMLU7862728/HC40`, not `VMLU 7862728/HC40`.

The flow samples canonical IDs in
[curated_campaign](../src/document_ocr/synthesis/curated_campaign.py#L368), then
the [curated-variable branch](../src/document_ocr/synthesis/curated_templates.py#L704)
calls `wrap_owned_text`. Its general
[layout_surface](../src/document_ocr/synthesis/curated.py#L375) reflows replacement
words, so a compact one-word ID loses internal source punctuation.

There is also an explicit-surface path: source `c7060e22` has non-curated
`container_0/1` owners whose declared expressions supply canonical whole-ID
surfaces. Fixing only the curated-variable branch would miss those owners.

Therefore: most required information already exists and labels are normalized
correctly, but it would be inaccurate to claim every template is “ready” merely
by turning on a flag. Both rendering paths, explicit overrides and intentional
OCR aliases need deliberate integration and replay.

### 4.3 Validation of a proposed presentation operation

An in-memory prototype maps a verified source occurrence reversibly to its
canonical ID, then substitutes the new ID's 11 letters/digits while preserving
the per-occurrence source separators:

| Source occurrence | New canonical target | Prototype rendering |
|---|---|---|
| `VMLU 4211677` | `VMLU9482650` | `VMLU 9482650` |
| `MCLU 510204.9` | `MCLU1801198` | `MCLU 180119.8` |
| `MCLU 509083/7` | `MCLU0371702` | `MCLU 037170/2` |
| `OOCU 866 027-5` | `OOCU2449809` | `OOCU 244 980-9` |

Differently formatted repeats retain their own styles while sharing one target.
For example `TCKU 685625-3` and `TCKU6856253` become `TCKU 974980-8` and
`TCKU9749808`, both targeting `TCKU9749808`.

All **808 reversible owned occurrences** passed identity replay, canonical
round-trip, generated check-digit validity and unchanged surrounding bytes/span
length checks. This includes all 101 noncanonical occurrences. One deliberate
`ECDU`/`ESDU` OCR-character alias was correctly rejected as **not formatting**;
its explicit existing alias policy must remain separate. Seven contaminated,
truncated, wrong-category or unsupported negative controls were rejected.

The prototype processed 80,800 occurrences in .0933 seconds, approximately
1.16 microseconds each. That is a feasibility measurement, not an integrated
production benchmark. No production formatter or dataset was changed.

The deterministic draw used seed `20261015`, namespace
`identifier-presentation-offline-audit` and identity
`source_id + "|" + target_path`, keeping the old four-letter equipment prefix
and excluding the old number. Reversible presentations were verified by removing
only whitespace, `.`, `/` and `-` and requiring exact equality to the known old
canonical value. The prototype substituted the new eleven ASCII letters/digits
in place and retained each separator. It did not guess character substitutions.
Timing used 100 passes over 808 precomputed triples and excluded loading,
new-number generation, full document rendering and surrounding-byte assertions.

All 101 affected source occurrences are single-line: this observed bug does not
remove physical lines or position anchors. Integration must nevertheless replay
OCR/position hashes, surrounding seal/type text, allocations and uniform/mixed
equipment controls before publication.

### 4.4 Why this is relevant to the real-data benefit

The real 600 contain **132 of 1,251 IDs across 67 documents without a contiguous
copy anywhere**. Of these, 130 reconstruct with spaces/slashes/hyphens; two use a
dot before the check digit. Validation has 11 such IDs across six documents.
Synthetic has zero mandatory reconstruction cases despite having suitable
source templates.

In the matched final-output comparison, K gets 94/101 contiguous IDs versus
R22's 96/101, but K gets **11/11 fragmented IDs versus R22's 0/11**. This is a
specific missing extraction skill, not a broad loss of container recognition.

## 5. Seals: substantial source-format variability already survives

The registry contains 778 seal fields in 837 owned occurrences across 396
sources. All have owners. All **15,455 synthetic seal targets** are printed
literally. They include 3,114 hyphenated values, 131 with slashes, 80 with spaces,
13 with lowercase letters and 1,510 with leading-zero numeric components.

Only three repeated-alias cases are homogenized by explicit surface recipes:

| Source | Alternate spelling versus chosen target | Descendants |
|---|---|---:|
| `069cc830` | `ML-BR3273047` / `MLBR3273047` | 17 |
| `8c54d974` | `ML-BR2305139` / `MLBR2305139` | 9 |
| `c1b20a71` | `0015881 SIF2015` / `0015881/SIF2015` | 35 |

Every source also prints its chosen target elsewhere. The 61 descendants remain
supported and extraction-consistent. Retaining alternate spellings would add
fidelity/variability; this audit does not establish a required label repair.
In matched model outputs, exact correct seals improve **110 to 114 of 119** in
R22. Do not apply a global container-ID normalization rule to seals or remove
meaningful seal punctuation.

## 6. Countries: target semantics must be separated from printed presentation

### 6.1 Existing policy differs by role

| Field family | Established extraction expectation |
|---|---|
| Route, issue and payment locations | Normalize explicitly supported country wording/code/adjective to the conventional English short country name, uppercase. |
| Parties | Preserve recognizable country wording or code as printed, uppercase; no inference from city, telephone or company knowledge. |
| Final MPCI form fields | ISO alpha-2 country codes; deterministic mapping is downstream of this KIE representation. |

See [LocationV7.country](../src/document_ocr/label_schemas/bill_of_lading_v7.py#L68),
[ExtractionPartyV7.country](../src/document_ocr/label_schemas/bill_of_lading_v7.py#L596),
and [the saved form schema](../artifacts/mpci-ai-schema/mpci-cuscar-canonical.schema.json#L119).
The extraction country string is not interchangeable with the final form's
country-code field.

Thus location OCR `KE` should target `KENYA`; a party printing only `HK` should
target `HK` under the existing source-faithful party rule. This audit does not
unilaterally change those rules or normalize all countries to one new convention.

### 6.2 Established inventories and rendering issue

The real 600 contain 1,666 party-country facts, including 70 undotted
code/short-form values plus eight punctuated abbreviations: **78 total**.
Validation contains 151 party-country facts, including six undotted and two
punctuated abbreviations: **eight total**. Observed
short forms include `EG`, `US`, `UAE`, `TR`, `USA`, `ET`, `ES`, `IT`, `DE`, `HK`,
`CN`, `UK`, `U.A.E.`, `U.K.` and `A.R.E`. They are not all literal ISO
alpha-2/alpha-3 values: e.g. `UAE`
is a familiar abbreviation, not ISO alpha-3 `ARE`.

Real training's 898 location-country facts and validation's 105 contain no code
targets, consistent with normalized location-country policy. Synthetic has
18,932 party-country facts and 10,110 location-country facts, all full names.

The source registry has 28 owned two-letter location-country occurrences in 14
templates, underpinning 188 descendants. One opened example changes
`Place Issued: MOMBASA,KE` into
`Place Issued: BOMA,CONGO, THE DEMOCRATIC REPUBLIC OF THE`. The country target can
be semantically consistent while the abbreviation-to-name learning opportunity
has disappeared. These are source/exposure counts, not an assertion that every
descendant's every location slot was independently replayed.

For parties, 37 source families have 45 code-valued source targets and contribute
705 descendants / 852 corresponding country facts. In 36 families the country
is inside a whole generated postal region, which prints the newly generated
full name. Those current full-name labels remain grounded; the lost code
examples are not automatically 852 bad labels.

One confirmed exception is source `b54eac08`: the shipper has a standalone exact
country-code owner and no postal address. Eighteen descendants print a code
such as `COUNTRY CODE: Ar *` but target `ARGENTINA`. The source printed `HK` and
its source target is `HK`. This is semantic normalization, but it contradicts
the established **party source-wording** target policy. It must not be described
as merely a harmless change in input typography. This bounded counterexample
does not justify declaring all synthetic party-country labels wrong.

An all-413 native ownership compilation narrows the standalone two-/three-letter
party-country-owner class to six occurrences across four families:

| Source family | Current descendant country facts | Own-party full-name alternative printed? |
|---|---:|---|
| `01d86535` | 64 | Yes |
| `22fa3887` | 32 | Yes |
| `dd8cea1c` | 16 | Yes |
| `b54eac08` | 18 | No |

For 80 of the first 112 facts, the full name matches the owned postal text
directly. The other 32 have an explicit same-party postal continuation after
TEL/VAT, such as `MALDIVES(MV)`; those were inspected rather than rejected by a
simple address-substring test. For the final 18, the expanded target is absent
from the entire rendered document. The 18 count is exhaustive **for this
standalone two-/three-letter owner class**, not a claim to resolve every possible
country abbreviation or OCR ambiguity in natural language.

The cause crosses layers:

1. [Scenario assignment](../src/document_ocr/synthesis/curated_scenarios.py#L1378)
   puts the full registry country name in every sampled party target.
2. [Owned rendering](../src/document_ocr/synthesis/curated_ownership.py#L681)
   preserves a standalone source alpha-2 owner as a new alpha-2 code.
3. The country grounding guard accepts geographic equivalence to the certified
   code. That proves the **country identity**, not the party's required
   **printed target representation**.

The appropriate immediate recommendation is to make sampling/rendering honor the
existing role rules. A different, unified country-normalization rule would be a
separate explicit policy/contract decision, not something to introduce silently
while repairing input styles.

Country names from the sampling registry also need to be distinguished from the
label policy's conventional short English names; do not assume all ISO registry
labels automatically satisfy that wording convention.

### 6.3 Country performance on unchanged expected values

The following use each completed model's own frozen final reference; all country
facts are identical between those references. They include all 60 documents, so
JSON failures remain penalties:

| Country scope | K correct / gold / predicted | K F1 / accuracy | R22 correct / gold / predicted | R22 F1 / accuracy |
|---|---|---|---|---|
| Parties | 138 / 151 / 149 | .9200 / .8519 | 131 / 151 / 145 | .8851 / .7939 |
| Locations | 95 / 105 / 100 | .9268 / .8636 | 85 / 105 / 96 | .8458 / .7328 |

On the eight abbreviated party-country targets, K gets seven correct and R22 two.
K's missing one belongs to a malformed-JSON document. All eight R22 documents
parse: two correct, two missing and four wrong. Other party-country wording
targets are much closer: K 131/143, R22 129/143.

Examples of R22 errors: printed party `USA` becomes target `UNITED STATES`;
printed `IT` becomes `ITALY`; consignee `EG` becomes locality `BEHEIRA`; notify
`US` becomes another party's `EGYPT`. The first two would be appropriate
normalization for a location country, but not for the current party rule. The
last two are ownership/type errors, not acceptable country normalization.

Four manually verified location-code input facts are receipt/loading
`SAN VICENTE, CL`, discharge `DUMYAT (DAMIETTA), EG` in `aa9ed483`, and discharge
`ALEXANDRIA(EG)` in `75006aac`. K gets three of four country facts, R22 zero;
R22 omits all four. This small, inspected cohort supports investigating the
missing code-to-full-name task but cannot prove a causal percentage.

The subsequent all-105-fact review in section 12.3 expands this initial bounded
cohort to five code/abbreviation facts by finding `KHALIFA PORT, UAE`; use that
later exhaustive gold-supported inventory for the policy discussion.

A literal screen of real-country labels found one genuine minor typography
variance: source `94709757` prints `U.K`, while its country and address labels add
a final period, `U.K.`. Two other flags were boundary/comma-spacing differences,
not new countries. This narrow real-label consistency item is recorded for later
adjudication; no real or validation label was changed.

## 7. Goods descriptions: source, sampled facts and generated wording

### 7.1 Scope and interpretation

Across all 7,000 source/descendant pairs, median word-count expansion is 5.33x;
5,740 are at least twice their source and 3,684 at least five times. For 3,179
descendants whose source descriptions have 1–4 words, the source median is three
and descendant median 29. This is descriptive-depth drift, not automatically
an incorrect rendered label.

It is **not newly characteristic of R22/R23**. Whole-training description medians
in J/K/R23 are 33/29/27 words and means 39.99/38.03/35.02. Current data is not
becoming more verbose relative to those successful runs. Real-training and
validation medians are five and six words.

The current prompt already requests source-appropriate depth: concise identity,
specification passage or product list, with no character/line quotas. Adding the
same sentence again would not establish a solution. Some historical expansion
also follows sampled commodity multiplicity, not just free model elaboration.
Exact HS/product taxonomy matching is not the user's extraction objective.

The following are existing current samples, not newly generated probes. Sampled
HS6 is the genuine wording context. Full printed HS extensions are separately
synthesized formatting examples; where no HS is printed, no HS target is added.

### 7.2 Example A: short source becomes a short specification passage

Source `92da45e4`, descendant `syn_full_v7_af090812b0598113a95b25cc`:

```text
Source description: OILFIELD EQUIPMENT
Source printed/target HS: 8413.91
Sampled wording HS6: 390319
Sampled subject: OTHER POLYSTYRENE IN PRIMARY FORMS
Current printed/target HS: 3903.19

Generated product wording:
OTHER POLYSTYRENE IN PRIMARY FORMS
CRYSTALLINE POLYSTYRENE (GPPS) GRADE PS-118, MELT FLOW 8 G/10 MIN
HIGH IMPACT POLYSTYRENE (HIPS) GRADE PS-242, IMPACT MODIFIED, EXTRUSION USE
```

This changes a two-word commodity identity to a 28-word two-grade description.
The current target joins these generated lines with spaces. It is label/render
consistent; matching the brief source depth is a separate distribution question.
The [wording/scenario receipt](../artifacts/kie-synthesis-production/curated-v7-expansion1000-v1/wording/syn_full_v7_af090812b0598113a95b25cc.json)
records the genuine sampled identity and generated lines.

### 7.3 Example B: one identity becomes a long model catalogue

Source `b3aeb7ab`, descendant `syn_full_v7_0035969670397c83219b464b`:

```text
Source description: INSHELL WALNUTS
Source printed/target HS: 0802.31.0000
Sampled wording HS6: 851621
Sampled subject: ELECTRIC STORAGE HEATING RADIATORS
Current printed/target HS: 8516.21.4244
```

The generated wording is a 196-word catalogue: four models, their electrical and
thermal specifications, accessories and finish. For example, it starts:

```text
ELECTRIC STORAGE HEATING RADIATORS
MODEL SRH-1200 DENSE-CHARGE STORAGE RADIATOR — 1200 W RATED OUTPUT,
230 V / 50 HZ, CHAMOTTE MAGNESIA BRICK HEAT CORE, CHARGED DURING OFF-PEAK
PERIODS, FANLESS STATIC CONVECTION DISCHARGE, SEVEN-FIN STEEL CASING,
WALL-MOUNT BRACKETS INCLUDED, DIMENSIONS 780 X 560 X 220 MM
```

Subsequent models are SRH-1700, SRH-2500 and SRH-3200, followed by bracket kits,
timer controls, sensor probes, stands and finish details. These are model-created
specifications around one sampled identity, not 196 words demanded by the source.
The exact complete generated value is in the
[wording receipt](../artifacts/kie-synthesis-production/curated-v7-expansion1000-v1/wording/syn_full_v7_0035969670397c83219b464b.json).
Keeping this as a valid long-description example is different from letting this
depth dominate descendants of terse sources.

### 7.4 Example C: expansion partly required by sampled multiplicity

Source `1e66b2bc`, descendant `syn_full_v7_0f9195ee4450e78434f39281`:

```text
Source description: ACCESSORIES
Source HS presentations: 16 codes, representing 15 distinct HS6 prefixes
Sampled wording identities: 15 cotton-yarn subjects
Generated description: 232 words, headed COTTON YARN, NON-RETAIL PACKING
```

The sampled HS6 values are `520535`, `520532`, `520542`, `520521`, `520541`,
`520543`, `520512`, `520511`, `520522`, `520548`, `520524`, `520523`, `520526`,
`520544` and `520546`. Generated wording lists their yarn constructions, fibre
preparation and count ranges. The current 16-code target preserves distinct
national extensions, including two of `520522`; sampled wording context remains
six-digit HS identities.

The [scenario and complete generated value](../artifacts/kie-synthesis-production/curated-v7-uae2500-v1/authority/wording/syn_full_v7_0f9195ee4450e78434f39281.json)
show why it is inaccurate to attribute all this expansion to aimless LLM
verbosity: the model was asked to represent 15 sampled subjects. A generic
source heading can cover many codes without spelling all their technical
distinctions out. This is a sampling/description-depth interaction to discuss,
not a reason to enforce a precise HS-to-product ontology or create 15 goods rows.

### 7.5 Example D: already-detailed source stays a specification passage

Source `4e766b20`, descendant `syn_full_v7_0f73b5910e9517b42c6ebd06`:

```text
Source description:
USED AND NEW EQUIPMENT: COOLING COMPRESSOR BURNER & ACCESSORIES
ELECTROMOTOR VENTILATORS PUMPS TOOLS LATHE ELEKTRO TRANSMISSION
KOMPRESSOR COOLING COMPRESSOR AIR CYCLINDER AIR DRYER BLOWER BLOWER
PERSONALITY GOODS DRILLING MACHINE HYDRAULCIC PUMP WELDING MACHINE CAPACITOR

Sampled wording HS6: 841810
Sampled subject: COMBINED REFRIGERATOR-FREEZERS WITH SEPARATE EXTERNAL
DOORS OR DRAWERS

Generated wording:
HOUSEHOLD REFRIGERATION APPLIANCES - COMBINED REFRIGERATOR-FREEZERS WITH
SEPARATE EXTERNAL DOORS OR DRAWERS:
MODEL KF-460 IX - 460 LITER GROSS CAPACITY COMBINED REFRIGERATOR-FREEZER,
STAINLESS FINISH, SEPARATE EXTERNAL DOORS FOR REFRIGERATOR AND FREEZER
COMPARTMENTS, FROST-FREE, 230V/50HZ, CLASS A+
```

No HS code is printed/targeted in this template; HS6 is generation context only.
This is a plausible detailed-description transformation, unlike treating every
short commodity source as a long catalogue. The exact source/generated physical
line arrangement is in the
[receipt](../artifacts/kie-synthesis-production/curated-v7-expansion1000-v1/wording/syn_full_v7_0f73b5910e9517b42c6ebd06.json).

### 7.6 Example E: concise regulated wording is host-controlled

Source `7d44dcc4`, descendant `syn_full_v7_20a2c891c2a325ca5e245ddd`:

```text
Source description: POTASSIUM AMYL XANTHATE
Source HS: 3824990090
Sampled HS6 and registry subject: 293979, NICOTINE
Current description: NICOTINE
Current full HS target: 2939797170
```

This chemical profile selects exact product wording from its registry; the LLM
writes the other requested linguistic fields, not the chemical name. It is not
evidence that the model generated `NICOTINE` from a free product prompt. See the
[saved scenario](../artifacts/kie-synthesis-production/curated-v7-expansion100-v1/wording/syn_full_v7_20a2c891c2a325ca5e245ddd.json)
and `host_product_wording()` in the production campaign.

### 7.7 Relevance to observed model failures

On the 57 matched final-output documents, exact descriptions decline 39 to 34.
By gold length: K/R22 get 17/16 of 19 descriptions of 1–4 words; 13/12 of 22
descriptions of 5–9 words; 8/6 of 14 of 10–29 words; and 1/0 of two longer ones.
The model failures do not establish that short-description overexpansion is the
main cause. R22 did not display a pattern of inventing long paragraphs for short
gold descriptions in the inspected outputs.

Phones and container/placement IDs account for 76 of the 116 **net** fewer
correct unchanged non-HS facts; descriptions account for five. This measures
error concentration, not causal fractions or the number of errors a new formatter
would automatically repair. Prioritize the demonstrated presentation gaps;
consider descriptive-depth controls in a separate experiment without truncating
valid descriptions or adding arbitrary character/line limits.

## 8. What useful signal comes from real examples, rather than reused values?

There is some literal value overlap between real training and validation:
54/182 party-name values, 29/175 addresses and 26/116 phone strings after
case/whitespace folding. This is not proof of memorization or duplicate documents.

To test beyond contact reuse, exclude every phone whose last seven digits occur
anywhere in real training. On the matched parseable cohort, 70 conservative
unseen phone values remain: **K gets 60, R22 gets 19**. A digits-only comparison
with international `00` folding leaves 72 unseen values: K gets 62, R22 20.
The benefit demonstrably extends beyond copying known full contact values.

Container IDs are also predominantly new: among 111 matched validation IDs not
seen in real training, K gets 104 versus R22 95. By contrast seals improve in the
synthetic-only model and matched addresses have 132 exact matches in both.
Real-data benefit is not uniform across every field.

One train/validation pair shares container, seal, vessel/voyage and date, but has
different consignees, cars, VINs, weights and volumes, with explicit part-container
wording. This is correlated co-loading, not evidence that they are identical
documents. Physical-shipment grouping is worth considering for a future holdout;
it does not account for the unseen-phone result. No split was changed.

These observations identify useful presentation and reconstruction supervision.
They do not isolate the effects of all 600 examples, EVA calibration, sample
weighting or country distribution into exact causal percentages.

## 9. Other presentation findings and boundaries

- **Casing:** human-readable targets remain uppercase. Synthetic party/locality
  input regions are approximately 50/50 uppercase/title case by configuration;
  real parties are overwhelmingly uppercase. That is intentional input
  augmentation, not a target bug. Source-case preservation would instead be a
  requirement of a future surgical anonymization mode.
- **Dates:** source numeric and named-month variation exists. A generic audit
  cannot call every recipe mismatch a defect; an observed full-month recipe
  renders `October`, while a generic abbreviated-month probe expects `Oct`.
  No broad date-format collapse was established.
- **Measurements:** existing number/grouping and unit-surface adapters preserve
  source typography. No broad new numeric rendering loss was demonstrated.
  However, gross-weight supervision has zero pound targets in the templates,
  synthetic set and real training; validation has one pound target. This is an
  actual coverage hole, not a reason real-data addback would solve it.
- **Websites:** the current contact flow explicitly preserves `WWW.` versus
  scheme-bearing forms; no analogous prefix collapse was demonstrated.
- **Non-target fax realism:** 109 source-only fax-named recipes still select the
  phone/mobile generator. These are not extracted phone-label errors. Keep
  this separate from the main extraction repairs; do not inflate the task into
  unnecessary telecommunications realism.
- **Literal matching limits:** name/address concatenations and real OCR
  boundary irregularities can make word-boundary searches fail even when text
  follows its source. Such a screen does not authorize a label repair.

## 10. Implications for surgical anonymization, not implementation authorization

The user's proposed direction is to replace sensitive identifiers and party
names/addresses while retaining real dates, vessels, cargo, layout and OCR
presentation. The measured benefit on unseen values supports this direction:
the original private values do not need to be preserved to preserve the
observed reconstruction and presentation task.

Do not run the full current synthesis campaign unchanged for this purpose. It
resamples cargo/routes and changes public identity/date fields too. A surgical
path should reuse reviewed labels and exact ownership, edit only approved
sensitive spans and their repeats, update only dependent target fields, and
verify exact immutability outside that allowlist. Preserve punctuation, casing
patterns, physical line boundaries and coordinate suffixes. Do not relabel or
rewrite the entire document unnecessarily.

An explicit scope decision remains: eight real descriptions contain VINs.
Product wording can stay unchanged while an embedded sensitive identifier still
needs anonymization. Retaining dates/vessels/cargo also leaves shipment
linkability to assess; changing a party name is not by itself proof of complete
anonymity. This audit neither establishes legal compliance nor authorizes changing
validation during a running comparison.

## 11. Proposed next work and acceptance evidence

No item below is marked fixed by this investigation.

| Priority | Finding | Appropriate layer | Required evidence before publication |
|---|---|---|---|
| High | Phone family collapse | Sampled identity/presentation separation, owned rendering, target projection | Family coverage including non-`+`, compounds held explicitly, repeated occurrences, source-style and normalized-identity controls |
| High | 18 doubled-plus samples | Complete boundary ownership or shared boundary-safe renderer; affected data replay | All affected descendants, adjacent-literal negative controls, unchanged unrelated text/labels/coordinates |
| High | Container separator collapse | Typed per-occurrence presentation across curated and explicit-surface paths | All 808 reversible owners, explicit OCR-alias routing, 864 descendants, surrounding type/seal/allocation controls |
| High | Party country code versus full-name inconsistency | Respect or explicitly revise the role policy; synchronize scenario, rendering and targets | Owned code-only versus code+name cases, real/validation consistency and unchanged unrelated fields |
| Medium | Location code presentations expanded away | Country presentation policy separate from normalized target | Alpha-2/alpha-3/printed-name examples with correct locality ownership and target normalization |
| Separate experiment | Description-depth distribution | Commodity multiplicity/context and wording policy, not text clipping | Source/descendant examples, concise/detail/list coverage and held-out extraction outcomes |
| Optional | Three seal-alias homogenizations | Occurrence-specific alias presentation | Preserve actual seal identity/leading zeros and avoid treating meaningful punctuation as generic separators |

After a proposed change, validate source bytes/ownership, repeated values,
rendered text, target semantics and positions together. Passing a schema alone
does not establish presentation coverage or field ownership. Conversely, a
source-style difference is not automatically a wrong extraction label. Keep
these two questions separate to avoid another repair/re-repair cycle.

## Appendix A. Exact current rows for the two confirmed 18-sample defects

Rows refer to the pinned 7,000-row synthetic shard in section 1. The two lists
are disjoint. They identify future repair targets; none was changed here.

### A.1 Literal-frame plus duplicated by generated phone

Source `doc_9ad49fcc846c53b1780f28a74c0cd1073d7bb2d812e5b5a76d2744b1ba4a7f02`;
field `parties.deliveryAgent.contactDetails.phoneNumbers[1]`.

| Row | Synthetic document ID |
|---:|---|
| 1184 | `syn_full_v7_7b1c687a2cfad092c825844d` |
| 1185 | `syn_full_v7_3693ab9a7544aad2b37366e9` |
| 1186 | `syn_full_v7_9bf68e844d21d7ce6db5bb25` |
| 1187 | `syn_full_v7_22186a5909ec212e1e1fa4bd` |
| 1188 | `syn_full_v7_1c451beb521312c5a4ef403a` |
| 1189 | `syn_full_v7_759eeae65517b577a6c32dac` |
| 1190 | `syn_full_v7_b569c3d66ac99519ee436c94` |
| 1837 | `syn_full_v7_8e4b8291b32799b747122a4f` |
| 1838 | `syn_full_v7_9f195b2394b3d3d75a638888` |
| 3070 | `syn_full_v7_3ecf2132219e8a5d7dc0e925` |
| 4387 | `syn_full_v7_990d3ca132260886435f997f` |
| 4456 | `syn_full_v7_005b5a8f0e656f875e4498b4` |
| 4457 | `syn_full_v7_b5b95d766b44d0d2f026d0c0` |
| 4458 | `syn_full_v7_68286c180bd63b0fddd86121` |
| 4459 | `syn_full_v7_7570388508b91506056cf66c` |
| 4460 | `syn_full_v7_bd8522367106a9bf2d8e93f5` |
| 4733 | `syn_full_v7_0ab1c3de3a575db9eb610c86` |
| 4734 | `syn_full_v7_09152825bb3d674ef7e00c6a` |

### A.2 Party targets expanded despite code-only input

Source [b54eac08](../artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases/doc_b54eac08475183f42a2f4d8d60c2ad065714136cd816e6978dac1759c11236a6/source.txt);
field `parties.shipper.country`. Uppercasing the printed code, rather than
expanding it, follows the current party rule. The choice remains a policy
discussion here; the table records actual input and current target.

| Row | Synthetic document ID | Printed code | Current target |
|---:|---|---|---|
| 1260 | `syn_full_v7_fedff0426e859eb606523746` | `Ar` | `ARGENTINA` |
| 1261 | `syn_full_v7_be137471f0600a11218ea80f` | `DE` | `GERMANY` |
| 1262 | `syn_full_v7_a44075d86dbf5f83e358cb65` | `It` | `ITALY` |
| 1263 | `syn_full_v7_44602baffeb6c73536860bbb` | `It` | `ITALY` |
| 1264 | `syn_full_v7_3e52d6073dd1bf1ea27c6aae` | `Au` | `AUSTRALIA` |
| 1265 | `syn_full_v7_0d7956c8e4fe7e39fe20e1bf` | `MX` | `MEXICO` |
| 1266 | `syn_full_v7_4a29cf414afc2c61a7a8355c` | `Pk` | `PAKISTAN` |
| 1851 | `syn_full_v7_ee159f7d1651471dac36a136` | `Mx` | `MEXICO` |
| 1852 | `syn_full_v7_523c2ca5fa74d25779dd127a` | `BR` | `BRAZIL` |
| 3124 | `syn_full_v7_c16aef4bf1026730e9d1fab5` | `Cn` | `CHINA` |
| 4131 | `syn_full_v7_2b79d82ad9ddfd38e8d458ec` | `Fr` | `FRANCE` |
| 4132 | `syn_full_v7_bb531799f879a45c8b466110` | `Es` | `SPAIN` |
| 4133 | `syn_full_v7_ebf6fc5829e92bf28ebf215b` | `Mm` | `MYANMAR` |
| 4134 | `syn_full_v7_4d53d8d0dc585b90116b0e56` | `Pk` | `PAKISTAN` |
| 4135 | `syn_full_v7_c6037f7b5947e37c776c5f6f` | `De` | `GERMANY` |
| 4873 | `syn_full_v7_d0f01779af5e23bc18c931ba` | `CN` | `CHINA` |
| 4874 | `syn_full_v7_ff2163e5aa2751e14a68b103` | `Bd` | `BANGLADESH` |
| 6253 | `syn_full_v7_7fc088269f02110ab28de0a6` | `Pl` | `POLAND` |

## Appendix B. Validation boundary for this report

Compilation and presentation probes used existing data and CPU-only code. No
LLM calls, inference, synthesis campaign, model loading or training controls were
used. No new temporary scripts were added to the repository. The only delivered
workspace change for this task is this report.

All asserted counts distinguish target fields, physical owned occurrences,
distinct templates and descendant documents. Generic pattern searches are
explicitly screens unless reviewed ownership establishes their meaning. Numeric
identity equivalence in the phone feasibility probe is not confused with the
printed-value extraction policy; geographic equivalence is not confused with
party country source-form fidelity. These distinctions are part of the required
future validation, not merely caveats to append after a pass.

Final readback rechecked all three dataset SHA-256 pins, all 36 local evidence
links and all 36 explicitly inventoried defect rows against the current shard.
Both 18-row family memberships, row positions, doubled-plus occurrences and
code-only party targets agree with the inventory. A fresh streaming recount also
confirmed 12,663 plus-prefixed, space-bearing synthetic phone targets and 18,932
party-country targets. These checks took 2.70 seconds; they are validation of
this report's evidence, not a production repair or a model-quality benchmark.

## 12. Follow-up: repair scope, country mapping and description error anatomy

This follow-up answers the subsequent questions without editing datasets,
templates, schemas, renderer code or running training. The input pins in section
1 remain the audit authority. Counts refer to the actual 7,000 synthetic,
600 real-training and 60 validation records, not a proposed future selection.

### 12.1 Phone policy and which layers need work

The current task is **printed-value extraction, not E.164 normalization**.
Joining physical OCR lines or folding whitespace for an audit does not authorize
changing digits, deleting a printed dialing prefix or supplying a missing one.
Printed compounds/extensions still need explicit boundary treatment; a complete
number and an abbreviated suffix are not two independently complete numbers.

A fresh punctuation-/prefix-preserving literal screen, folding only whitespace,
finds 1,103/1,110 real-training values and all 116 validation values in their OCR.
Seven training values in five documents need closer treatment. None changes the
decimal digit sequence. Four are digit-group/physical-line joins; the remaining
three concern braces, duplicated hyphens and a multi-number parenthesis envelope.
Literal presence anywhere is not proof of party ownership: two additional
real-training contact assignments conflict with decisions already applied in
the active source authorities:

| Source | Current real assignment | Established source-authority decision |
|---|---|---|
| `a3e3c1e3` | Consignee phone `0020 1022896630` | Its occurrence is in the goods-area continuation, not the direct consignee block. The active authority already omits this consignee contact. |
| `2e8c2a1f` | Notify phone `203 3991070`, within a duplicated notify/delivery party assignment | The explicit `AS FRT FWDRS DELIVERY AGENT ONLY` wording establishes delivery ownership. The active authority omits notify. This is a party-role correction, not merely deleting its phone. |

The isolated-authority/protected-real split is explicitly recorded in
[expansion360 preparation](kie-synthesis-expansion360-results-2026-10-10.md#L112)
and [DG admission](kie-dg-template-expansion20-2026-10-14.md#L129).
The full source IDs are
`doc_a3e3c1e3c3d3a24d554f0e5ce647e106c7557768da461671d85e2f9757998929`
and `doc_2e8c2a1f893def8a2660ec987c6d99d52923bf989ac2abdcd75c66eaa512b559`.

Among 369 real-training sources with admitted templates, 662 of 664 current-real
phone paths agree with the active source authority; these are the two exceptions.
Another 446 real phone paths belong to sources without an admitted template.
This comparison establishes agreement with reviewed authorities, not a new
independent semantic certification of every agreeing contact.

The seven literal-screen flags are:

| Source / party phone index | Current label | Printed OCR | Disposition for repair planning |
|---|---|---|---|
| `ab1056f9`, delivery `[0]` | `+20 2 24801010` | `PHONE:{+}20 2 24801010` | Braces removed; inconsistent with three other real `{+}` labels. Reconcile punctuation policy. |
| `374deb27`, delivery `[5]` | `+20122738114` | `+201227381` then `14` | Physical digit-run wrap joined; no digit/punctuation alteration. |
| `24c091c8`, shipper `[0]` | `00.90.212.982.49.58` | `00.90.2` then `12.982.49.58` | Physical digit-run wrap joined. |
| `24c091c8`, consignee `[0]` | `+20 48 2659023-14` | `+20 48 2659023-` then `14` | Physical line wrap joined. |
| `24c091c8`, notify `[0]` | `+20 48 2659023-14` | Same printed split | Same whitespace-only case. |
| `a0f7d986`, delivery `[0]` | `(+203 4865088)` | `(+203 4865088/4865082/4865080/` then `4863088 )` | Parenthesis surrounds a list; label supplies it to the first individual number. Needs a list-envelope convention, not a guessed digit correction. |
| `2b88d69f`, delivery `[2]` | `(20-3)4873951` | `(20-` then `-3)4873951` | Deletes one printed hyphen; exceeds whitespace-only cleanup. Reconcile punctuation policy. |

These five full source IDs are:

```text
doc_ab1056f920fae5f2b728d3109a26c9a39e2c65ad6f550f488382f0135deb5348
doc_374deb27cc95bb1eb708437980656edfe03919ff7aaaf92b4c35d500fb14f301
doc_24c091c863feb5d61516da466cd436eae72f2710957a76a94b925537ee1d85bd
doc_a0f7d986fd766a3ba88d1d9206ba32db2d6a0eb480efff23c36fb33967c66010
doc_2b88d69f06717c214c31562ab8a6bf6fce245d11525aa7f492539172e2513498
```

Their OCR is under the real r14 `samples/<source-id>/ocr.txt` directory linked
earlier in this report. None of these seven flags establishes a wrong digit.
In addition, 407 of the literal-screen target values have repeated occurrences;
this is another reason an anywhere-in-document match does not certify a role.

A supplementary FAX-caption screen produced six flags, all rejected after
context review: four are explicit `MOB.` values after a separate FAX field and
two are joint `TEL-FAX`/`PHONE/FAX` values. The older
[_standalone_fax_governs](../src/document_ocr/labeling_agents/deterministic_annotation.py#L455)
guard recognizes fewer heading variants than the review-policy helper. That is
a guard limitation, not six bad real labels or evidence that it caused the
synthetic-only model's phone errors.

The historical `ca74e1e4` compiler source still has B's phone in a composite
`A VIA B` record while the current real target selects A alone. This was an
intentional provenance distinction already documented in the
[VIA repair](kie-hs-dg-format-repair-2026-10-14.md#L123), not a third real-label
conflict to copy back from a historical compiler fixture.

The significant `002 ...` to `+92 ...` example is an incorrect **model
prediction**, not evidence that its real gold number needs replacement.

| Layer | Required direction |
|---|---|
| Real targets | Preserve source-faithful targets; review the finite punctuation flags and correct established ownership inconsistencies, not globally normalize all phones. |
| Real OCR and positions | Leave unchanged. A wrongly copied model prediction does not justify changing the original input. |
| Template authority/bindings | Preserve source presentation per occurrence; fix incomplete boundary ownership such as the external literal `+`. Do not discard intentional historical compiler evidence or existing reviewed authority overrides. |
| Generation/rendering | Separate numeric identity from source-conditioned or explicitly sampled presentation, then handle repetitions and compounds deliberately. |
| Synthetic OCR | Re-render affected phone occurrences and all owned repeats in the intended presentation. |
| Synthetic targets | Update together with rendered phones when their printed presentation changes. Under the current policy the label is not a separate E.164 string. |
| Positioned synthetic inputs and manifests | Publish the corresponding positioned text and updated hashes together; preserve line/geometry invariants and unrelated content. |

This is therefore mostly a synthesis/presentation correction, but not exclusively
one. A boundary defect needs a source/binding or shared renderer correction, and
the real-label ownership deltas must not be lost merely because source admission
used an isolated corrected authority.

For the narrow doubled-plus defect, the intended target already has one `+`:
correct the accidental extra input prefix; do not change that target to `++`.
This differs from deliberately restoring `00`/national/parenthesized phone
presentations, where both input and target must change together.

### 12.2 Container repair scope is different from phone repair scope

Container targets remain canonical identifiers. Restoring input separators does
**not** require adding those separators to target IDs or placement references.
The renderer should preserve each owned occurrence's reversible source
presentation, covering ordinary and explicit-surface paths. Existing synthetic
raw/positioned inputs and publication hashes need synchronized updates.

Do not resample every container identity to achieve this. The existing canonical
draw can be rendered differently; seals, categories, goods allocations and all
other facts can remain unchanged. Intentional OCR-character aliases are separate
from punctuation and need their explicit current contract, not guessed repair.

### 12.3 Full validation audit of location-country mapping

The earlier four-fact code cohort was not exhaustive. This follow-up manually
inspected the OCR context for **all 105 labeled location-country facts across
40 validation documents**. The other twenty documents have no labeled
location-country fact. This is a gold-supported format/recall audit, not proof
that no country fact is missing from gold.

K and R22 are their completed final predictions, using their own frozen
references; every country expectation is identical between them. R23 has no
saved final prediction file at the time of this audit, so no current-run country
performance is invented and no new inference was launched.

| OCR presentation | Gold facts | K correct / missing / wrong | R22 correct / missing / wrong |
|---|---:|---|---|
| Explicit full country name | 87 | 85 / 2 / 0 | 83 / 3 / 1 |
| Full name with punctuation, `UNITED,STATES` | 1 | 1 / 0 / 0 | 1 / 0 / 0 |
| Country code/abbreviation requiring expansion | 5 | 3 / 1 / 1 | 0 / 4 / 1 |
| Country-name alias requiring normalization | 7 | 5 / 0 / 2 | 1 / 0 / 6 |
| City-state wording, `SINGAPORE` | 3 | 1 / 2 / 0 | 0 / 3 / 0 |
| Identical locality's explicit country elsewhere, not in issue phrase | 2 | 0 / 2 / 0 | 0 / 2 / 0 |
| Total | 105 | 95 / 7 / 3 | 85 / 12 / 8 |

No demonym-only location-country input exists in this validation inventory.
The exact code cohort is:

| Document / role | OCR | Expected country | K | R22 |
|---|---|---|---|---|
| `aa9ed483` receipt | `SAN VICENTE, CL` | CHILE | Entire receipt object absent | Country absent; locality retained |
| `aa9ed483` loading | `SAN VICENTE, CL` | CHILE | Correct | Country absent; locality retained |
| `aa9ed483` discharge | `DUMYAT (DAMIETTA), EG` | EGYPT | Correct | Country absent; locality retained |
| `75006aac` discharge | `ALEXANDRIA(EG)` | EGYPT | Correct | Country absent; `(EG)` incorrectly retained in locality name |
| `6a1350a5` loading | `KHALIFA PORT, UAE` | UNITED ARAB EMIRATES | `UNITED` | `UAE` |

The last case was missed by the earlier tight adjacency screen because `PORT`
separates the target locality and country abbreviation. Five facts are only
three documents and three distinct abbreviations; CL and EG each occur twice.
All code-cohort outputs parse. These results **do not establish robust mapping**.
They also do not reduce every error to country conversion: one K error omits
the whole receipt and R22 sometimes fails country/locality segmentation.

The seven alias facts are five `BUSAN, KOREA` occurrences, expected SOUTH KOREA,
and two Turkish issue-country spellings, expected TURKEY under this task. R22
copies `KOREA` on all five Korean facts, copies `TURKIYE` on one Turkish fact and
normalizes the other correctly. K normalizes all three Korean loading facts
but not the two receipt facts, and gets both Turkish issue countries right.
This is a broader normalization weakness than two-letter codes alone.

Two issue-country labels use an identical locality's country printed elsewhere:
`ff7f0853` has `MARINA DI CARRARA - ITALY` at loading and the same locality at
issue; `da651f2f` similarly has `OULU PORT FINLAND` and `Oulu` at issue. These are
not borrowing an unrelated party's country. Whether explicit same-locality
resolution across roles is allowed needs a consistent policy statement; they
were left unchanged rather than called proven label errors.

**Policy options:**

1. Keep normalized location-country names and restore code/alias input
   presentations during synthesis. This retains the current contract and can
   test whether the absent supervision closes the observed gap.
2. Extract the printed country surface and use a downstream resolver. This
   reduces model-side transformations, but is a deliberate real/validation/
   template/synthetic contract migration, not merely a rendering fix.
3. Emit form-ready alpha-2 directly. This minimizes output adaptation but still
   asks the model to recognize codes, aliases and names and map them consistently.
   The present evidence does not demonstrate that broader task is solved.

The saved physical form stores alpha-2, so English-name output does not remove
all downstream mapping. Literal extraction does not remove ambiguity such as
`KOREA`, nor fix omissions or incorrect ownership. The lowest-disruption
experiment is to repair the missing presentations under the existing contract
and measure the named/code/alias strata separately before a country-wide
policy migration. Do not silently change party-country source-form policy.

### 12.4 What the wording model is actually instructed to do

The current [WORDING_PROMPT](../src/document_ocr/synthesis/curated_wording.py#L225)
contains:

```text
Generate fragments jointly at the source's descriptive depth and
presentation: concise identity, specification passage, or product list. Cover each
sampled commodity subject. Use natural item boundaries without line or
character quotas.
```

The per-product field requirement in
[wording_request](../src/document_ocr/synthesis/curated_campaign.py#L613)
also says:

```text
Complete commercial product wording, matching the source's level of detail
and list structure. Express the required commodity facts and physical form.
Classification wording supplies context, not extra required attributes.
Product variants and intrinsic specifications may enrich an example that
has that detail.
```

The shipment context repeats that concise/specification/list distinction and
requires every sampled identity at the source's level of detail. Requested
product fields carry their source fragment as an example; split fragments are
generated jointly with description-assembly context. Thus this is not a missing
instruction to respect source scope. There is no instruction to maximize length,
nor a universal instruction to expand or contract every source. Context about
multiple sampled subjects and optional intrinsic variants can nevertheless
encourage elaboration. Proving a better prompt or request decomposition needs a
controlled wording probe, not another untested sentence or hard length cutoff.

The subsequent [manual prompt comparison](kie-goods-description-prompt-probe-2026-10-15.md)
tests this directly on six short/medium/long sources. Stating the measured source
word count in the field description substantially improves short/long scope in
that probe; generic word-count guidance alone is weaker. All raw comparisons,
including two rejected HS-caption proposals, are preserved. Production wording
and this audit's dataset measurements remain unchanged.

### 12.5 Description-length coverage, including short and contracted examples

Counts are whitespace-delimited words in the **final description target**, not
model reasoning tokens, characters or the full rendered goods section:

| Words | Real 600 | Validation 60 | Synthetic 7,000 |
|---|---:|---:|---:|
| 1–4 | 282 | 21 | 416 |
| 5–9 | 187 | 22 | 335 |
| 10–19 | 107 | 12 | 1,193 |
| 20–49 | 19 | 4 | 3,383 |
| 50+ | 5 | 1 | 1,673 |
| Median | 5 | 6 | 29 |
| Mean | 7.323 | 9.500 | 37.392 |

There are **751 synthetic descriptions below ten words, 10.73%**, compared
with 469/600 real descriptions, 78.17%. Short examples are not absent, but their
share and context are very different. Of the 416 synthetic 1–4-word descriptions,
356 are DG-chemical profiles; only 36 belong to ambient profiles. Among 5,802
ambient synthetic samples, 236 are below ten words, 4.07%.

Compared with the corresponding source description, 6,481 descendants are
longer, 399 shorter and 120 equal in word count: **92.59% / 5.70% / 1.71%**.
Equal word count is not identical semantic depth, and different commodities
naturally require different names. Of 3,179 descendants from 1–4-word sources,
334 remain in 1–4 words, 151 in 5–9 and 1,583 reach at least thirty words.
This establishes widespread expansion, not merely cherry-picked long examples.

There are also successful lexical-generation examples, not just host-controlled
chemical names:

- [Concise identity](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/wording/syn_full_v7_0258a588c6061fc1b697691a.json):
  source `WHEAT BULGUR FINE`; sampled HS6 `843131`; generated
  `PARTS OF LIFTS, SKIP HOISTS OR ESCALATORS`. It remains a concise commodity
  identity even though its word count is larger.
- [Contracted product list](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/wording/syn_full_v7_0271cc52c969d944c422f246.json):
  source is a 34-word list of glucose-test strips/meters; sampled `540419`, other
  synthetic monofilament; generated thirteen words:

  ```text
  SYNTHETIC NYLON 6 MONOFILAMENT, DIAMETER 0.30 MM
  SYNTHETIC POLYESTER MONOFILAMENT, DIAMETER 0.45 MM
  ```

- [Shorter detailed list](../artifacts/kie-synthesis-production/curated-v7-coverage1000-v1/wording/syn_full_v7_1f37f2a12190e176f95f5c2b.json):
  source is a 132-word used-compressor/dryer inventory; sampled `390230`,
  propylene copolymer resins; generated a 62-word four-grade resin list, rather
  than expanding the source again.

These saved receipts contain the actual lexical output fields. They are not
fresh calls or examples constructed for this report. The existing system can
produce concise, contracted and detailed outputs. Its **distribution** still
overweights elaboration, especially outside the short host-controlled DG names.
The earlier report's 27-word R23 median is for the combined 7,600-row training
set; 29 above is the 7,000-row synthetic shard alone. These must not be confused.

### 12.6 Saved description failures: what an incorrect scalar actually looks like

All 23 R22 description mismatches in the jointly parseable 57-document cohort
were opened, compared with their expected strings, and reviewed against OCR
context and the prior description-boundary adjudication. The outputs establish
observable behavior, not access to the model's internal reasoning.

K has 39 exact descriptions, R22 34 in that cohort. Both are correct on 32;
seven K-correct cases regress, two K-wrong cases improve, and sixteen fail in
both. A nonempty description is present in every one of those 23 R22 failures.

| Primary observed failure | Shared-cohort cases |
|---|---:|
| Omitted prefix, attached specification, continuation or product-list material | 14 |
| Added declaration/reference/heading material | 4 |
| Mixed boundary/duplicate/foreign-field contamination | 3 |
| Selected the wrong repeated/auxiliary passage | 1 |
| Corrupted product words | 1 |
| Formatting-only | 0 |

Primary categories are mutually exclusive for accounting; a case can contain
secondary symptoms. The exact scalar metric gives no partial credit to a
nearly complete description. It is nevertheless incorrect to dismiss these
23 as casing/comma-only misses.

| Source ID prefix | Expected versus actual R22 behavior | Primary category |
|---|---|---|
| `13ea92fa` | Drops `REFRACTORY MATERIAL` before `MOULD POWDER, GBB714/L-9`. | Omission |
| `d9be84a2` | Keeps pesticide/brand identity but loses both attached product-code/capacity lines after a separate HS line. | Omission |
| `51820556` | `COPPER MILBERRY SCRAP GRADE 99.7% AS PER ISRI` stops at `GRADE`. | Omission |
| `e5a37f5a` | Adds the leading `1` to otherwise complete `NEW CAR SKODA …`. | Added declaration |
| `ff146232` | OCR/gold `AIRTHROUGHBONDED ACQUISITION` becomes `AIRTHOUGHBONEED ACUCTION`. | Word corruption |
| `da651f2f` | `FINNISH WHITEWOOD` gains the following accounting caption `VOLUME`. | Added heading |
| `ff7f0853` | Drops `PLT` from `PLT RESIN`; prior PDF adjudication places `PLT RESIN` in the description column, with `4` in a separate quantity column. | Omission |
| `cda61263` | `PAPRIKA POWDER 60 ASTA HTST 1680X25KG.` loses the final capacity line. | Omission |
| `5222fa98` | Loses `5 RACKS OF` and appends the shipper's Luxembourg address to `STEEL CASING.`. | Mixed contamination |
| `3b5b0c08` | `RED OAK LUMBER 691 PIECES 10258 BF` becomes `RED OAK LUMBER`. | Omission |
| `cf7299a1` | Keeps `MICROCRYSTALLINE CELLULOSE MCC 102` but omits preceding `PHARMACEUTICAL GRADE`. | Omission |
| `9f97c8ed` | Fuses `F.I.D.M.BLOCK` and appends the separately printed HS `04069078`. | Mixed boundary |
| `53e77979` | Selects detached `VIN: … DIM: …` instead of the main `CAT LOADER 972H SERIAL # …` block; this auxiliary passage has differently rounded dimensions. | Wrong passage |
| `5e03e6b2` | Drops the attached `***NONSTACKABLE***` qualifier. | Omission |
| `5cabcd4b` | Adds `GUPCO PO LINE ITEM 1 / ITEM # 317800` before an otherwise complete 47-word product passage. | Added reference |
| `d87589f1` | Drops text within the embedded code/model introduction and loses the entire page-two product continuation. | Omission |
| `aa9ed483` | Repeats `FORGED STEEL GRINDING BALLS`, then adds volume and incoterm information after the valid product passage. | Mixed boundary/duplication |
| `4f981d62` | Drops attached `1000 KG IBC (STEEL PALLETS) X10`, retaining only the lecithin product identity. | Omission |
| `624696c0` | Keeps `TITANIUM DIOXIDE KRONOS 2064`, omits `800 PAPER BAGS (25 KG) ON 20 PALLETS`. | Omission |
| `75006aac` | Adds preceding `CFS-FO` to `SPARE PARTS`. | Added heading |
| `68b8d417` | Drops the product's `CASO FCC` prefix, retaining the rest of the long passage. | Omission |
| `66feaac3` | Stops after the reactive-dye/code introduction, omitting all three `SUNZOL …` product-list lines. | Omission |
| `c1d9e099` | Keeps `UNMANUFACTURED RAW TOBACCO`, loses crop year, grade and attached packing specification. | Omission |

The two improvements are `8625ddda` (R22 correctly stops before an extra
`100 PALLET`) and `6a1350a5` (R22 retains `BORSTAR HE6062` before `HD POLYETHYLENE`).

For completeness outside the paired cohort, R22 has one additional parseable
description error: `684f442c` omits `'AL HAMD 'BRAND'` from the frozen-buffalo
description. K's JSON is invalid there, so it is not added to the paired
comparison. Over **all sixty** documents, K has forty exact descriptions and
one malformed output; R22 has thirty-four exact, twenty-four parseable
description errors and two malformed outputs. No malformed document was silently
counted as an inspected nonempty description.

The principal training targets suggested by these errors are bounded passage
selection, retention of attached product specifications, correct page/list
continuation, and exclusion of neighboring accounting/reference/party text.
Simple short and detailed passages are both needed; merely lengthening or
shortening all synthetic descriptions cannot address those different behaviors.
Include matched variants where the product identity is the same but an attached
qualifier, a separate HS/reference row, a page continuation or a nearby total
changes the correct boundary. That is a proposed targeted experiment, not a
claim that generation-length changes have already repaired model behavior.

Follow-up readback verified all 45 local evidence links, the two current-real
ownership examples, all 23 shared-cohort description inventory entries, and the
unchanged hashes of the three protected input shards. Only this Markdown report
was added/extended; no production behavior or training data was changed.
