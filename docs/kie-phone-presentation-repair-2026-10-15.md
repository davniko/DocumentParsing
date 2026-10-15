# Phone presentation repair: diagnosis, implementation and validation

Date: 2026-10-15. Scope: source-conditioned synthetic telephone/fax presentation,
matching phone targets, and two already-adjudicated real-training ownership fixes.
No model inference, training or paid generation is required for this repair.
Surgical anonymization, container-ID formatting and country-policy changes are
separate work and are not silently bundled into it.

## 1. How many saved prediction errors?

The audit uses the **completed R22 synthetic-only run's final saved predictions**
against that run's own frozen reference. It does not invent per-document results
for R23's intermediate evaluations, whose generated strings were not saved.

| R22 phone diagnostic, all 60 validation documents | Expected values |
|---|---:|
| Exact native metric matches | 36 |
| Same complete digits, different presentation | 21 |
| Altered/incomplete digits or non-phone payload | 23 |
| Missing | 36 |
| Total expected | 116 |

There are also six extra predictions. The 80 wrong/missing expected values occur
across 34 documents. The quoted `002…` to `+92…` substitution occurs twice,
in consignee and notify of the same document. It is not 80 examples of that
particular substitution. The 23 altered/non-phone pairings comprise seven
different digit strings, seven removed leading zeros, five one/two-digit edits,
two added leading-digit cases, one annotation-contaminated value and one email
in a phone field.

On the 57 documents that parse in both compared runs, R22 has 36/106 exact,
20 presentation-only, 22 altered/non-phone, 28 missing and six extra. Successful
run K has 99/116 exact over all 60, 17 missing and nine extra; under the same
diagnostic it has no residual same-digit presentation changes or paired digit
corruptions. On the jointly parseable subset K has 96/106 exact.

Pairing removes native exact facts, then equal-digit residuals within the same
aligned party role, then minimum digit-edit-distance residual pairs. These
diagnostic categories do not change the metric implementation and do not prove
that every different-number pair has the same cause. Full paths, examples and
frozen prediction hashes are in the
[prediction inventory](analysis/phone-presentation-repair-20261015/r22-phone-error-inventory.json).

## 2. Confirmed defect and target policy

The old synthesis host sampled a valid identity but rendered it in one default
international style. Reviewed source separators, dialing envelopes and physical
wraps were not retained. An unchanged `+` outside a binding could combine with
the generated value's `+`, producing two prefixes. Several historical phone
bindings also no longer matched the current reviewed array order or field span.

The extraction target remains **the printed telephone value**, not normalized
E.164. Preserve dialing digits, punctuation, extensions and abbreviations;
physical line wraps may be joined, including reuniting a split digit run where
the reviewed source establishes that presentation. Do not invent an international prefix or
silently remove leading zeros. Standalone fax remains non-target auxiliary text;
joint TEL/FAX may be a telephone target. Existing OCR artifacts in real inputs
are not rewritten to make model predictions appear correct.

The repair separates:

1. **Identity sampling:** the existing country-aware sampler owns the new valid
   phone identity. This repair does not truncate a subscriber to fit an old mask.
2. **Presentation:** source groupings, dialing family and physical wraps control
   how that identity is printed. Extensions/abbreviated suffixes are deterministic
   auxiliaries, not extra independent phone-array entries.
3. **Ownership:** each exact source occurrence is rendered through its existing
   binding. Repeated occurrences share identity but retain their own styles.
4. **Projection:** printed text and the corresponding phone target change together.
   Independent replay reconstructs and checks the presentation receipt.

Unknown original geography is explicitly represented as an unknown source
country, not guessed from a company name or city. A bare unproven old number
supplies a prefixless national shape; the newly sampled identity still requires
a valid target country. A malformed old number supplies presentation only.

## 3. Upstream implementation

| Component | Responsibility |
|---|---|
| [phone_presentations.py](../src/document_ocr/synthesis/phone_presentations.py) | Source grammar, prefix/grouping transfer, valid identity preservation, suffix handling and exact newline preservation |
| [curated_phones.py](../src/document_ocr/synthesis/curated_phones.py) | Typed configurable policy, occurrence ownership, repeated-value coherence, target projection and pinned replay receipts |
| [curated_templates.py](../src/document_ocr/synthesis/curated_templates.py) | Explicit phone-plan requirement for changed phones; phone-only assembly and composite-owner overlays |
| [curated_campaign.py](../src/document_ocr/synthesis/curated_campaign.py) | Preflight before paid work, production render integration and independent candidate replay |
| [curated_auxiliary.py](../src/document_ocr/synthesis/curated_auxiliary.py) | Apply the same source-presentation model to non-target phones/faxes |
| [Label policy](kie-real-baseline-label-policy-2026-10-05.md) | Restate printed-value and party-ownership rules |

Default configuration preserves source presentation, which supplies the actual
corpus variation. Optional dialing-family weights are explicit:

```yaml
phones:
  family_weights:
    - family: source
      weight: 8
    - family: double_zero
      weight: 1
    - family: national
      weight: 1
```

Other available families are `plus`, `international_bare` and `national_bare`.
Source remains the default; this repair does not impose the illustrative weights
above. Immutable external `+`/`00` framing constrains eligible families. An
incompatible configured policy raises instead of silently erasing the frame.
Groupings and newline counts remain source-led. Presentation choices have their
own deterministic stream and do not consume draws from shipment/goods sampling.

Five active source declarations needed targeted ownership rebases:

| Source prefix | Confirmed mismatch | Repair |
|---|---|---|
| `28f4` | Notify mobile and joint phone/fax no longer matched the reviewed phone array | Corrected the mobile index and linked the auxiliary phone/fax to its actual target |
| `9ea52` | One owner contains two current phones; subsequent indices had shifted | Declared the actual two-field owner and corrected later indices |
| `5311` | A wrapped `002` prefix and subscriber were separate historical owners | Bound the complete two-line telephone occurrence |
| `0ba3` | Four notify-phone owners had rotated relative to reviewed array order | Rebound each exact source occurrence to the correct phone target |
| `5c5e` | A target phone included `EXT:117`, but the historical owner excluded it | Bound base plus extension together and consumed the superseded extension span |

Source OCR, reviewed source target bytes, position anchors and task schema are
unchanged by those rebases. The registry manifest pins the updated ownership and
auxiliary files. Source framing is fixed at the proper layer rather than by
global regex replacement in descendants.

One additional explicit owned-presentation declaration corrects `ab1056`:
its source actually prints `PHONE:{+}20 2 24801010`, while the historical
compilation target omitted the braces. New descendants target the literal
`{+}` presentation. This is a reviewed selector in ownership data, not a
template-ID branch or a generic punctuation-stripping fallback. Historical
source datasets and contract-envelope hashes remain valid; the current renderer
projects the corrected target and records its source-owned basis.

Strict grounding also accounts for an immutable opening parenthesis immediately
outside a phone binding, and for a source-attested phone digit run split over
physical lines. Only the verified external character and explicitly identified
newline boundaries may participate. Matching just the same digits elsewhere
does not pass.

## 4. Validation on the actual code path

- **1,765 tests passed in 87.11 seconds** across all 41 curated test modules,
  formatter, contact sampling and the final migration tests. The narrower initial
  integration suite passed 528 tests in 24.63 seconds.
  Ruff and `git diff --check` pass.
- All **413 templates compile**. All **735 public phone paths across 307
  phone-bearing templates**, covering **811 owned presentations**, pass the
  complete registry probe; zero unresolved owners. See the
  [strict grounding validation](analysis/phone-presentation-repair-20261015/all-template-phone-grounding-validation.json).
- Cross-country/source-style and explicit-family probes cover **5,400 cases**
  without identity truncation or changed physical line counts. Tests exercise
  wrapped prefixes, repeated aliases and roles, literal-plus ownership,
  extensions, abbreviated alternatives, ranges, composite owners, missing
  ownership, malformed inputs and deliberate replay tampering.
- Eight saved-receipt sources pass native production `Campaign.render` and
  `Campaign.replay_candidate`, without generation/API calls. Phone-free controls
  are included. Eleven additional DG, thermal and multi-container controls pass
  final native render/replay, including the delivery-only, external-parenthesis
  and split-digit-run sources. This is native integration, not just a formatter probe.
- Separate-subprocess CPU benchmarking, five batches of eight documents after
  warmup: old median **0.142893 s**, new **0.155983 s**, approximately **1.64 ms
  added per document including independent validation**. Process peak RSS was
  **451,076 to 451,384 KiB** (+308 KiB). This small absolute rendering overhead
  is not presented as a training-speed or whole-generation throughput result.
  See the [native benchmark](analysis/phone-presentation-repair-20261015/native-campaign-benchmark.json).
- Repeating that benchmark after the final strict-grounding additions gives
  **0.146806 s per eight documents**, about **0.49 ms/doc above baseline**,
  with **450,576 KiB** peak RSS (no measured increase). Both measurements are
  small CPU-stage probes with observable timing variation, not generation-rate
  guarantees. See the [final code-path validation](analysis/phone-presentation-repair-20261015/native-campaign-final-validation.json).

The tests establish correct rendering/projection and the intended new coverage.
They do not claim that formatting alone will eliminate every model omission or
digit error; that is measured by the next training experiment.

## 5. Full-data replay and gained presentation coverage

The final repair replay passes **8,000/8,000 unique synthetic documents**, with
zero unresolved records: the current 7,000 plus the full older UAE sidecar,
which adds 1,000 non-overlapping documents. It plans **19,195 exact owned edits
in 5,931 changed records** and validates **19,340 phone/auxiliary-owner
occurrences**. Runtime is **118.91 s**, process peak RSS **819,624 KiB**.
The [final replay](analysis/phone-presentation-repair-20261015/full-v4.json) and
[per-record edits/proofs](analysis/phone-presentation-repair-20261015/full-v4-receipts.json)
retain exact original/current ownership frames, native source proof hashes,
dependent target updates and preservation checks.

An extra preservation falsification rejected the preliminary v3 publication:
reconstructing whole source owners had restored the non-phone suffix `egy` in
17 records. The final migration overlays phone pieces into the **actual current
owner** whenever complete; reconstruction is reserved for verified missing
co-owned phone pieces. Every owner now has a typed-phone-only alphabetic
preservation check. In total 353 records / 516 owner bodies retain existing
non-phone separators/suffixes instead of reconstructing old source framing;
all target projections are unchanged from the earlier phone plan. All 17
`egy` insertions are eliminated, and unowned `/1` suffixes are not restored.
The rejected publication is preserved separately, explicitly nonadmitted.
No protected original or active training file was modified.

Coverage in the current 7,000-record training shard, over the same 12,663 phone
values, changes as follows:

| Printed prefix family | Before | Repaired |
|---|---:|---:|
| Plus, including parenthesized/braced plus | 12,663 | 5,663 |
| `00` | 0 | 2,189 |
| National leading zero | 0 | 1,501 |
| Bare | 0 | 3,310 |

Width-independent punctuation/grouping shapes expand from **12 to 72**.
Previously missing shapes now include contiguous `D` (3,522 values), contiguous
`+D` (1,924), `D-D-D` (727) and `(D-D)D` (192). This measures genuine presentation
coverage, not just different randomly sampled digits. Family classification is
a diagnostic only; it does not normalize any target. See the
[coverage inventory](analysis/phone-presentation-repair-20261015/phone-style-coverage.json).

Known quarantined old-UAE sidecar rows receive the phone-only repair but retain
their original nonadmission status and unresolved non-phone reasons. They are
not added to training. Existing phone-absent targets are not resurrected. In two
independent non-target fax families, the existing valid international identity
has a different country from its old nominal scenario; its actual identity is
preserved and the mismatch is recorded, not resampled or falsely relabeled.

## 6. Published dataset and final certification

The corrected [R24 dataset](../data/curated/mpci-bl-real600-synthetic7000-phone-presentations-r24-positions-v1/README.md)
is published, with the following exact inventory:

| File | Records | Rows changed |
|---|---:|---:|
| `synthetic-train.jsonl` | 7,000 | 5,200 |
| `real-train.jsonl` | 600 | 3, targets only |
| `validation.jsonl` | 60 | 0, byte-identical |
| `previous-uae-synthetic.jsonl` | 2,500 | 1,842 |
| `old-uae-addition.jsonl` | 1,500, overlapping mirror | 1,111 |

Within the current 7,000, 5,135 documents have changed phone targets; the
additional raw-only changes cover source-only phone/fax presentations. There
are 12,581 changed phone scalar values, without changing phone-array membership
or order. The complete repaired sidecar and selected-addition mirror agree
exactly with their overlapping training rows. All 153 sidecar quarantine IDs
remain excluded from training.

The three real-target corrections are:

- [A3, row 83](../data/curated/mpci-bl-real600-synthetic7000-phone-presentations-r24-positions-v1/real-train.jsonl#L83):
  remove consignee `contactDetails` containing the goods-area phone and email.
  Keep independently owned notify contacts. The active template authority already
  contained this adjudication.
- [2e8, row 256](../data/curated/mpci-bl-real600-synthetic7000-phone-presentations-r24-positions-v1/real-train.jsonl#L256):
  remove the duplicated notify party, preserving the explicit delivery-only
  GLOBELINK role and its contact information. The active authority already agreed.
- [ab1056, row 101](../data/curated/mpci-bl-real600-synthetic7000-phone-presentations-r24-positions-v1/real-train.jsonl#L101):
  change only delivery phone `+20 2 24801010` to printed `{+}20 2 24801010`.
  This reconciles the source's braces with the other real brace-bearing targets
  and the new template-owned projection. No digits change.

**All 18 doubled-plus descendants of source `9ad49fcc` now have no doubled
plus**, checked directly in the final shard against the protected original.

The [manifest](../data/curated/mpci-bl-real600-synthetic7000-phone-presentations-r24-positions-v1/manifest.json)
has SHA-256 `c8cbced38131f311388b2fc03d297eef10dc7286ee600b5a92218fee1923e3bb`.
Key published shard hashes:

```text
synthetic-train.jsonl
211cb1b24af0934b0945a433a716ffdc481ff8bb7c6ffa91fee271fc3e918139
real-train.jsonl
dc71297f20317957e508baf394277d4fc7da3be80e5b1dd14be489fcbe7a3485
validation.jsonl
5b0bdccba39158a97f75b5e478e297c8cc9dd58dd0018ee130d0c028dc6ae30f
```

Final publication took **53.53 s**, peak RSS **885,916 KiB**. Root then performed
an independent, non-mutating readback of every published row and every source,
output, registry and implementation pin. It reconstructed every changed raw
string from the exact recorded edits, checked every non-phone target and
unrelated metadata field, compared every physical line's positional suffix,
verified all overlaps/quarantine/memberships, and re-counted actual phone shapes.
This passed in **22.30 s**, peak RSS **253,984 KiB**. See the
[independent readback](analysis/phone-presentation-repair-20261015/independent-readback.json).

The real training startup path (`load_training_task`, `load_prompt`,
`inspect_dataset`) also passes for all **7,660 train/validation records**, using
the previous configuration rebound **in memory** to these exact R24 hashes.
This confirms the direct target, frozen schema/constraints, input hashes and
split inventory; no tokenizer/model is loaded. Runtime **5.84 s**, peak RSS
**146,524 KiB**. See the
[startup inspection](analysis/phone-presentation-repair-20261015/training-startup-inspection.json).

The phone-format repair and the three scoped real corrections are therefore
complete and validated. All protected R23/historical files remain byte-identical.
No current training configuration, dates, goods, container identities or other
unrelated extraction values were rewritten. No training was launched and no
paid requests were made. Surgical real-data anonymization remains the separate
next scope the user intends to define.
