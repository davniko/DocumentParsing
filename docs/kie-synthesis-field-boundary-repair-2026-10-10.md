# Synthesis field-boundary repair — 2026-10-10

## Published result and next-run entry point

**Published and validated:**
[repaired dataset](../data/curated/mpci-bl-real600-synthetic2500-v7-positions-boundary-repaired-v1/README.md).
The [matching training config](../configs/training/production/t5gemma2_270m_lora.mpci_bl_real600_synthetic2500_boundaries_positions_inputonly_v7_e10_compact_eva_a32_r48_local_schedulefree_v1.yaml)
uses the repaired train-file hash with the same rank 48, alpha 32, compact targets,
positioned input-only prompt, optimizer, warmup, evaluation and checkpoint settings.
Only dataset paths/hash and run-identifying metadata differ. **No training was
launched or stopped.** The previous running dataset/config and all 60 validation
rows remain byte-identical.

The actual final-path training startup inspector accepted **3,100 training and 60
validation rows**. All 155 staged data/geometry/provenance artifacts are preserved
byte-for-byte; only current publication metadata is added/replaced. Publication
took **38.60 seconds**, peak RSS **983.36 MiB** including training-stack imports.
Eight publication-gate tests passed, including rejection of missing/unresolved or
stale review adjudications and changes to unrelated training settings.

| Published split | SHA-256 |
| --- | --- |
| train.jsonl | `379b8a5fe39b8ef98391ad690b944fa2b236b4dd7bf710c6ba5b39d962508ac9` |
| synthetic.jsonl | `79bb23dfab11b5428ce5b23b792ad82581f670b9d6261a50f9e18c991391a78a` |
| validation.jsonl | `9f3b4c36fe5d6310b3fb79a6d380a06d24d2fe8583acf80748d83861b2fe94da` |

Exact final configuration, publication files, evidence and unchanged baseline pins
are in the [publication receipt](analysis/synthesis-field-boundary-repair-20261010/publication.json).

## Applied correction set

Follow-up on 2026-10-10: a [fresh combined production synthesis test and new
100-sample random audit](kie-synthesis-postrepair-random100-audit-2026-10-10.md)
now supplement the component/replay evidence below. They verify the repaired
mechanisms execute and document additional bounded casting/anonymization and
generated-accounting findings; those new findings are not silently included in
the completed correction set or counted as whole-document passes.

The identified field-boundary, primary-identity, package-declaration and certified
identifier-alias repairs are complete. **151 synthetic records across 41 source
families and two real training labels are corrected.** No identified boundary
repair remains held. This is the completed repair of the enumerated defects, not
a fresh whole-dataset annotation or a certificate of every private source residue.

| Scope | Applied result |
| --- | ---: |
| Training membership | 600 real + 2,500 synthetic = 3,100 |
| Protected validation membership | 60 real, byte-identical |
| Changed synthetic records | 151 |
| Changed real training labels | 2 |
| Changed inputs | 141 |
| Changed targets | 81 |
| Other training rows | 2,947, byte-identical |
| Unrelated input or target changes | 0 |
| Changed synthetic records semantically reviewed and adjudicated | 151 / 151 |
| Current source blueprints passing production preflight | 300 / 300 |

The repaired records span all four relevant synthesis cohorts: **104** from the
latest coverage-1,000 campaign, **31** from the earlier expansion-1,000 campaign,
**10** from expansion-300 and **6** from expansion-100. The latter two account for
affected records in the earlier 500-sample set; repairs were not limited to the
latest generated samples.

## Repair contract and authority

This implementation follows the [completed diagnosis and probes](kie-synthesis-field-boundary-probes-2026-10-10.md)
and the user's authorization to repair the dataset, affected templates and the
generation path. It addresses the identified extraction defects, not broader
physical-realism or HS-to-product classification objectives.

The source of truth for a descendant is its **currently published row**, not an
older generation candidate. Saved candidates provide sampled facts and exact
source-edit lineage. Later, explicitly recorded description/contact/equipment
repairs remain authoritative. Fresh renders are merged only at reviewed owner
keys; a wholesale replacement with a historical candidate is prohibited.

The current training job's mixed dataset and launch configuration remain
unchanged. Repairs are staged and published as a separate sibling dataset. The
canonical real source catalog receives two target-only corrections so future
synthesis cannot reintroduce those source-label mistakes. Validation examples,
raw real OCR, and measured real coordinates remain unchanged.

## Defects and root-level changes

### Generated field contamination

The confirmed name/postal/product mistakes existed in saved model wording before
rendering. The renderer subsequently copied those strings faithfully into both
OCR and labels. Text/label agreement alone therefore did not certify correct
field ownership.

The generation interface now uses semantic Pydantic aliases visible to the model,
such as `shipper_postal_component_1`, while retaining existing internal owner
keys. Each field carries concise expectations for the requested semantic value.
Names exclude tax/contact captions; postal values exclude company identity and
unrelated contact data while allowing actual named facilities and care-of
routing; goods wording excludes party-contact text. Host-provided product text
does not trigger an instruction to generate a nonexistent output field.

Source-certified identity kinds distinguish an individual, an organization, and
an organization with proprietor. The generic instruction about an example's
proprietor no longer contradicts an organization-only certificate. Final
rendered-text reviewers receive the certified meaning **and actual generated
component**, not just a flattened name string.

Examples of repaired behavior:

- A generated company prefix is removed from its postal value while the street,
  district, postcode, city, country and genuine facility are retained.
- `RIF J…` is removed from a generated company name; an uncaptioned personal name
  belonging to the source's company identity is retained without the invented
  `CONTACT:`/manager caption.
- A delivery-agent contact suffix is removed from goods wording without changing
  product specifications, quantities or the separately owned party contact.
- A street accidentally included in a company name moves into the corresponding
  address once, with its plot number preserved.
- Genuine `C/O BP EXPLORATION (DELTA) LIMITED` postal routing remains. Named
  estates remain in addresses; two synthetic company identities were changed
  instead of erasing their independently meaningful sites.

The semantic decisions and exact old/new values are in
[semantic-value-proposals.json](analysis/synthesis-field-boundary-repair-20261010/semantic-value-proposals.json).
It records 44 individually reviewed documents: 41 proposed repairs, three valid
care-of cases retained, 49 changed lexical values, and 250 unchanged values.

### Declarative package, declaration and country ownership

| Family prefix | Dataset defects identified | Root correction |
| --- | ---: | --- |
| `4f210d63` | 4 | Package noun changes declarations, not the description heading |
| `e3caae78` | 9 | Declared package total replaces stale row cardinality |
| `a4cb81c1` | 6 | Repeated standalone package total receives its sampled count |
| `00a165a2` | 6 | Legal-clause `14` stays fixed; only the actual allocation occurrence is mutable |
| `35136b22` | 10 | Description starts at the product, excluding the loading/package declaration |
| `2c599a8b` | 6 | Repeated package nouns bind the current declared package level |
| `ff351983` | 4 | Same heading-versus-declaration correction |
| `aaa232a2` | 5 | Eight repeated category owners are rebased consistently |
| `147a2626` | 6 | Repeated standalone total receives its sampled count |
| `53d4726a` | 9 | Same repeated-total correction |
| `5d9f7853` | 0 current | Latent second-row category mismatch fixed before future BAG draws |
| `9b35d0b6` | 6 | Delete only the certified redundant country occurrence |

These are existing declarative binding, integer, package-surface, retirement,
deletion and description-block mechanisms—not source-ID branches in Python.
Fixed headers, legal clauses and independent package levels remain unchanged.
No global removal of numbers, country strings or description prefixes is used.

The twelve-family repair changes `ownership.yaml`, one allocation occurrence in
`00a165a2`'s contract, and `35136b22`'s source target/hash. Its canonical real
description becomes `TIRES`, not `6070 PIECES TIRES`; the complete loading
declaration remains in OCR, outside that target.

See [template validation](analysis/synthesis-field-boundary-repair-20261010/template_validation/README.md)
for exact selectors, 82 descendant replays and 33 fresh coupled package draws.

### Formerly fixed composite names and source postal noise

Four source contracts gain eight independently owned name components, covering
five public name targets and 24 current descendants:

| Family | Identity assembly |
| --- | --- |
| `0bd3ce7b` | Company plus independently printed uncaptioned person; consignee and notify copies share the same values |
| `48973902` | Principal company `ON BEHALF OF` represented company; framing stays fixed |
| `600a1e89` | Notify company plus uncaptioned person, repeated on the second page |
| `f256749b` | Shipper company `A/C` account company; tax/address intervals remain separate |

These names were formerly omitted from generation requests. Clearer prompts
alone could not change an identity the model was never asked to generate.
Only those missing components were regenerated using each descendant's saved
locality and shipment context. Separately, six personal-party descendants use a
certified personal identity rather than a generated business.

Four organization-only API outputs acquired an unrequested personal first line.
Those original answers are preserved; explicit manual adjudication keeps their
organization line before rendering. This is not a silent runtime fallback.

For `88deddad`, `ENVNT` is excluded from the source postal owner/example and
canonical real address label. The actual printed street and postcode remain.
The source OCR is untouched; synthetic noise occurrences are repaired through
the certified owner. No global OCR-noise dictionary is introduced.

See [name/postal contracts and falsification](analysis/synthesis-field-boundary-repair-20261010/NAME_POSTAL_CONTRACT_REPAIR.md).

### Certified VAT aliases

Source `a53a1a9e` uses formatted consignee/notify VAT `643-668-997` and the
source-reviewed importer spelling `643668977`. Their intended common identity
is declared explicitly as `canonical_identity: '643668997'` in three auxiliary
owners. Generation derives one shared value from this certificate.

All ten current descendants have exact input-only importer-VAT proposals. Public
targets and unrelated identifiers remain unchanged. Other numeric IDs are not
merged by similarity, digit distance, or party-role guesses. Invalid alias
certificates fail rather than falling back to independent generation.

See [alias proposals](analysis/synthesis-field-boundary-repair-20261010/alias-proposals.json)
and `tests/test_curated_auxiliary_identity.py`.

## Validation, scope and regression prevention

Validation has separate responsibilities:

1. **Semantic decisions:** individually inspected old/new values, source field
   ownership and identity certificates; final read-only model review of changed
   rendered documents supplements this inspection.
2. **Exact application:** before-row hashes, declared JSON-pointer changes,
   character-offset input edit replay, source UTF-8 byte edit replay, and the
   complete unchanged-row/field census.
3. **Geometry:** changed synthetic inputs traverse the real transfer, reflow and
   augmentation path. Independent receipt validation checks text retention,
   page boundaries, source anchors, collisions, line order and transforms.
   Missing coordinates remain explicit, not invented merely to improve coverage.
4. **Dataset integrity:** train/synthetic mirrors, protected validation bytes,
   schema/casing/input hashes, and the actual training `inspect_dataset` path.
5. **Future execution:** actual synthesis `Campaign` construction and all 300
   production `blueprint` calls, plus hash-pinned source/target dependencies.

Historical candidates/reviews remain immutable provenance, not newly rerun
semantic approvals. The staging manifest explicitly identifies its predecessor
and new scoped repair receipts. Existing no-hit records do not need speculative
rewriting. Package capacities, retail/cone specifications and HS-neighbor product
wording that were agreed nonissues remain unchanged.

Completed execution evidence:

- **588 curated synthesis tests passed** in 19.68 seconds, including 40 new
  field-boundary tests.
- **18 transaction tests passed**, including a complete 3,160-row no-change
  stage and the actual training startup inspector; all 13 baseline dataset
  artifacts stayed byte-identical in that control.
- **300/300 actual production source blueprints passed**, with current source
  target pins and lexical ownership, in 13.638 seconds; peak RSS 439.56 MiB.
- **660/660 canonical real labels match their per-document mirrors**. Exactly
  two train target scalars changed; all real inputs/coordinates and validation
  rows stayed unchanged.
- Declarative fixes: **82/82** descendant renders and **33/33** fresh package
  counterfactuals passed. Deliberately selecting the wrong repeated `TIRES`
  occurrence is rejected.
- Missing-name contracts: **30/30** descendant replays pass; **24/24** omitted
  required-name negative controls are rejected.
- Wording: 18 fresh requests covered 104 full-request fields plus 12 certified
  identity components. No scoped name/postal/product contamination appeared.
  Existing later postal/accounting checks held four of the 14 full requests;
  these experimental responses were not published as accepted documents.

Final independent reruns after integration: **588/588 production synthesis tests
passed in 18.81 seconds**, and **24/24 repair-transaction tests passed in 22.53
seconds**, including six exact residual-adjudication controls. Ruff checks passed.

### Final changed-record semantic review and adjudication

The read-only GLM review received the full final rendered OCR and final targets
for **every one of the 151 changed synthetic records**, plus complete original
source OCR, old targets, approved edit context and certified identity components.
It returned **145 passes and six issues**. The six raw issue records are preserved;
they were not rerun to obtain desired answers or silently changed to model passes.

- Two `35136b22` findings requested restoring a declared pallet-count prefix to
  goods descriptions. The independently certified product region starts at
  `TIRES` occurrence 2; package counts/nouns have separate owners. Same-line
  flattened OCR does not merge those fields. Root source/contract inspection
  accepted the repaired descriptions without restoring the prefix.
- Four `9b35d0b6` findings identified unchanged private tax/reference/free-zone
  text. These passages are absent from the current description and primary
  route/party targets. Their exact references, `TAX ID : 525985263`,
  `IMPORT REFERENCE : 5259852632023060206`, and `PORT SAID FREEZONE`, were already
  recorded in the original random audit as a **separate secondary
  casting/anonymization follow-up**. They do not invalidate the applied
  duplicated-country correction. They remain explicitly tracked rather than
  being added to extraction fields, silently deleted or certified as fully
  anonymized. This report does not declare all secondary source residues fixed.

The combined final decisions accept **151/151 scoped boundary repairs**, with
**zero unresolved boundary repair findings**. The two real target changes are
separately source-certified and manually reviewed; their OCR/positions are frozen.
See [final review report](analysis/synthesis-field-boundary-repair-20261010/FINAL-SCOPED-REVIEW.md)
and [hash-bound final decisions](analysis/synthesis-field-boundary-repair-20261010/final-semantic-review/closure.json).

The final population census re-ran the original broad selectors over **2,500
synthetic records and 300 templates**. It found 49 residual selector documents:
45 individually cleared controls, three source-proven care-of cases and one
preserved named estate. Every residual field/value has individual adjudication;
there are **zero unadjudicated residual hits and zero unbound primary names**.
An old approved document ID cannot authorize a new field or changed value. A
named-site predicate-code transition is allowed only for the exact unchanged,
individually reviewed postal value, with negative controls for field/value drift.
See [population closure receipt](analysis/synthesis-field-boundary-repair-20261010/final-boundary-closure.json).

Source-publication evidence:
[registry-publication.json](analysis/synthesis-field-boundary-repair-20261010/registry-publication.json),
[source-label-validation.json](analysis/synthesis-field-boundary-repair-20261010/source-label-validation.json),
[future-campaign-preflight.json](analysis/synthesis-field-boundary-repair-20261010/future-campaign-preflight.json).

## Performance and API accounting

The same 82-input renderer workload measured 71.043 ms before versus 71.717 ms
after; the 8.2 µs/document difference is negligible. Local schema/prompt building
adds 0.049 ms/request, with no new network stage in ordinary generation. The
explicit-alias auxiliary stage measured 147.505 µs versus 150.027 µs before;
the ordinary non-alias path measured 149.503 µs. The full no-change staging
transaction took 9.21 seconds with peak RSS 983 MiB including training imports.
These are measured local-path results, not a claim about provider network speed.

Fresh wording validation cost **$0.00685080**; the two named-site identity calls
cost **$0.00018095**. Missing-name generation has **$0.00294390** in reported
receipts including its rejected first response. Four initially canceled
concurrent requests have no usage receipt, so their charge is unknown—not
asserted to be zero. Their enclosing generation budget was $0.50. Final review
has a separate $0.30 cap and preserves provider usage. The earlier diagnostic
pilot's $0.03739969 is historical and not charged again by deterministic replay.

Final semantic review recorded **$0.12306986** in provider-reported costs. Two
timed-out attempts have unknown actual usage; their conservative reservation is
**$0.02139750**, giving **$0.14446736 accounted** within the $0.30 review budget.
All 65 attempts are retained; only four failed batches were retried, and successful
reviews were reused with matching hashes. Across this repair's generation and
review receipts, the **known provider-reported amount is $0.13304551**, plus the
explicitly unknown canceled/timed-out attempts. The reservation is not presented
as a measured charge, and missing receipts are not treated as zero-cost calls.

## Evidence and backups

All implementation receipts and analysis live in
`docs/analysis/synthesis-field-boundary-repair-20261010/`. Before-images preserve
source labels, contracts, declarations and code; the dataset transaction snapshots
the complete mixed baseline. Original sampled shipment facts and API answers
are retained. The [wording implementation report](analysis/synthesis-field-boundary-repair-20261010/WORDING-IMPLEMENTATION.md)
details schema aliases, certificate behavior, tests, cost, runtime and controls.

The process closes enumerated defects by checking their final applied values and
source ownership. It does not equate strict JSON decoding or a substring match
with universal semantic correctness, and it does not discard completed repairs
merely because generation can still produce a reviewable future exception.
