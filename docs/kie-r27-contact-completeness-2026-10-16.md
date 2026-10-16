# R27 contact-list completeness audit — 2026-10-16

## Scope and status

**Latest closure:** the [legacy-reference and upstream repair](kie-r27-legacy-reference-and-upstream-closure-2026-10-16.md)
is now published. All four held contact families, source authorities, instruction
changes and their schema/configuration pins are live. Fifteen fresh descendants
from five repaired families pass complete source/render/label review and geometry.
The original service-contract hold and analogous reviewed owners have been
repaired in R27 and upstream. Current R27 manifest is
`39e13a0df3891ff149f20c64be00ac12843b29acf428fb9d1102039e49767865`;
the older phase descriptions and receipts below retain their historical meaning.

**Earlier source-boundary phase:** the later source-boundary review additionally corrected
eight real records and 34 unique synthetic records whose consignee contacts were
printed only in goods-area continuations. This is a contact-ownership repair,
not a missing-list repair. Direct consignee contacts, independently owned notify
contacts and supported postal continuations remain intact. The
[complete audit](analysis/template-expansion60-r27-20261016/consignee-goods-contact-audit.md)
records the nine PDF-reviewed source families and the 75 descendants deliberately
left unchanged because their generated contacts actually appear inside the party
blocks. The [guarded combined publication](analysis/template-expansion60-r27-20261016/batch02/r27-publication.json)
also includes the separately adjudicated batch02 source-policy corrections:
56 unique records / 72 physical copies in this phase, using current anonymized
values only. It preserves all coordinate suffixes and changes no validation rows.
The current manifest is
`aec461f198b3ca511b3e656c0f5a7b0cc35d347ae9f5211a9cd9a44be2e49460`.
Earlier counts and manifest hashes below describe their own completed phases.
The four associated admitted-family upstream changes are separately staged and
**held, not published**. Twelve fresh descendants pass contact/extraction review,
but complete text review found a distinct older private service-contract
residual. Its confirmed existing scope is 75 unique synthetic records / 135
physical copies. The [hold inventory](analysis/r27-consignee-continuation-20261016/FINAL-HOLD.md)
records the cause, a broader unadjudicated legacy-reference screen, exact staged
publication boundaries and all provider usage. This data-publication receipt
does not claim those additional upstream changes are live or the entire corpus
is privacy-complete.

A separate native regression also found three wrong fax-context declarations
across two older templates. The public Marseille carrier fax stays public/static;
two private shipper faxes now use shipper rather than delivery-agent context.
The [diagnosis and applied upstream repair](analysis/template-expansion60-r27-20261016/batch02/phone-owner-repair-diagnosis.md)
records all 192 screened auxiliary-phone recipes, manually adjudicated analogous
owners, the original failing seed and all 1,299 passing native plans. This fixes
template sampling ownership, not contact-list gold or the phone formatter. It
does not require re-generating correctly extracted existing non-target faxes,
and changes no R27 bytes.

This is a focused contact-list audit, not another full semantic certification of
every record. It follows the unresolved `a44bb232` validation telephone list in
the [failure follow-up](kie-r27-failure-followup-and-validation-overlap-2026-10-16.md).

The read-only inventory covers all 7,000 active synthetic training records, the
older UAE sidecar and its mirrored addition, 600 anonymized real-training records,
and 60 validation records: **8,660 unique records**, including **8,000 unique
synthetic records**. The 3,000 overlapping JSONL copies were checked for exact
agreement rather than counted again. Input hashes and all screening locations
are retained in the four [grouped candidate inventories](analysis/r27-contact-completeness-20261016/grouped-candidates-001.json).

**Three records received confident target-only phone-list corrections:** two real
training records and one validation record. Direct readback confirms all four
affected party arrays now match the reviewed patches. Two active template
families also received a certified split of their old merged phone ownership.
Existing synthetic descendants of
those families contain one generated phone where one is labeled; they are not
mislabeled and need no speculative phone addition.

Dataset and active-registry publication are now complete, coordinated by the root
publisher. The [first-phase readback](analysis/r27-pretrain-repairs-20261016/phase1-final-readback.json)
passes all **11,660 stored rows / 8,660 unique records**. Across the combined
contact/package/source-review follow-up, **23 unique records received target-only
repairs and 14 other records received privacy-only input repairs**: 37 unique
changed records affecting 53 physical row copies in total. The contact
contribution is the three records/four party arrays described below; these counts
are not additional to the 23.
R27's final manifest SHA-256 is
`2a5f00b6cb03b4859f2962ccfb39207a36065ce1a0cd2f3a029ab4a481eab07b`.

The explicit [row patches](analysis/r27-contact-completeness-20261016/reviewed-row-patches.json)
pin the pre-edit row/input/position hashes and preserve the current anonymized
digits. Original private source values are used only in their separate source
authorities, never copied into the anonymized training shard.

## Screening and its limits

The reproducible [audit](analysis/r27-contact-completeness-20261016/audit.py)
looks for captioned telephone regions whose complete values are absent from the
target, unrepresented email regions, and target scalars containing two long,
hyphen-separated telephone numbers. It masks known target literals before
screening. It inventories regions; **it does not infer party ownership** or
automatically add regex matches to targets.

| Screening result | Regions | Distinct records |
| --- | ---: | ---: |
| Potentially unrepresented captioned phone region | 1,492 | 1,340 |
| Potentially unrepresented email region | 193 | 191 |
| Merged full-number telephone target | 3 | 2 |

These candidates belong to 133 source families and 156 source/kind groups. They
are not counts of labeling defects. Large candidate classes are intentionally
excluded carrier/issuer contact blocks, separate unassigned organizations,
emergency-response contacts, goods-area continuations, and false lexical matches
such as `TEL AVIV` or telephone products. The TEL/FAX distinction, captions such
as VAT/CIF/RUT, and phone punctuation can also produce screen false positives.

The audit reviewed owned-list evidence and representative repeated source
families, not every complete 8,660-record document. This method cannot establish
exhaustive recall for arbitrary uncaptioned contacts, severely broken email OCR,
or missing complete party roles. No broad automatic “repair” was applied.

## Confirmed repairs

| Record | Existing target | Correct target under existing policy |
| --- | --- | --- |
| Validation row 54, `a44bb232` — delivery agent | `(+203 4865088` only | Keep that literal, add `4865082`, `4865080`, `4863088` as three separate entries. |
| Real row 156, `551f3a34` — consignee and explicitly repeated notify | `+2036471938-6472851` as one entry | `+2036471938`, `6472851`. |
| Real row 183, `a47ae6af` — consignee | `+(202) 74619825-74619826` as one entry | `+(202) 74619825`, `74619826`. |

All values are independently printed complete numbers. No missing country or area
prefix is invented. Source text and all coordinate suffixes remain unchanged.
The unmatched parenthesis in validation is printed OCR, not repaired by guessing.

Six analogous real-training MEDEGYPT-family sources already retain their four
printed phone entries. They need no new label edits. A validation shipper list
whose shorter `65360940` repeats an already labeled `0086-551-65360940` is also
not evidence for another distinct telephone identity.

### Upstream cause and targeted fix

For `551f3a34` and `a47ae6af`, historical accepted labels and compiled ownership
treated the hyphen-joined pair as one scalar. The current sampler correctly
samples one value for that one old field, so descendants agree with their
rendered text but lose the source's two-number presentation.

The repair uses the existing reviewed ownership mechanism: retire the merged
owner and add two exact, disjoint telephone owners with explicit target paths.
In `551f3a34`, each phone identity is shared between its two actual printed party
occurrences. The hyphen remains literal template text. No generic punctuation
splitter, country-prefix inference, source-specific runtime branch or new
schema rule is needed.

Staged assets, before/after targets, exact ownership changes, case pins and source
authority rows are in the [template rebase plan](analysis/r27-contact-completeness-20261016/template-rebase-plan.json).
Both templates passed exact source/target identity replay. Six new native
scenario plans passed preflight in **6.76 seconds**, with the expected two-phone
cardinality in the corrected roles. Final privacy-extended preflight also passed
all six plans in **7.01 seconds**. Fresh six-descendant wording/rendering, review
and geometry certification are recorded below.

### Source-only private residuals discovered during the fresh test

Opening the complete first-pass rendered texts found two pre-existing privacy
gaps outside extraction targets. These were not introduced by the phone repair:

- Seven current synthetic descendants of `551f3a34` retained the source's two SB
  references `5464632` and `5464651`.
- Seven current synthetic descendants of `a47ae6af` retained the source supplier,
  exporter and importer premises around already generated party names.

The [record inventory](analysis/r27-contact-completeness-20261016/two-family-private-residual-inventory.json)
also covers the older UAE mirrors: four records per family occur in those
sidecars, but they are copies of current records, not extra unique affected
samples. Both associated anonymized real-training records had already replaced
the original private values. Scope is **14 unique synthetic records**.

The upstream extension reuses existing ownership and native generation:

1. The two existing historical SB-reference owners receive source-format private
   identifier recipes, rather than new overlapping regex spans.
2. The existing exporter/importer postal aliases refer to their generated party
   address targets, without creating new extraction fields.
3. The supplier's genuinely separate Malaysian office gets a native source-only
   postal variable and an explicit Petaling Jaya locality context. It does not
   become another shipper address target or lose its independent office meaning.

Both extended templates again replayed source text and targets exactly. The
updated contracts, ownership and auxiliary declarations are in the staged case
plan and [privacy staging receipt](analysis/r27-contact-completeness-20261016/private-residual-staging.json).
The root publisher subsequently applied the [reviewed private-span plan](analysis/r27-contact-completeness-20261016/manual-private-residual-plan.json):
**49 unique literal substitutions / 105 across 30 physical row copies** repaired
the 14 unique records, including their older UAE mirrors. The final readback
proves that targets, coordinate suffixes and membership were preserved for every
privacy-only repair; no old private values were restored into the anonymized
real data. The plan and its original review retain their pre-publication status
as historical evidence; the final readback is the application receipt.

The [active-registry promotion readback](analysis/r27-package-vocabulary-20261016/current-source-promotion-readback.json)
also passes. Both contact/privacy families, their ownership and auxiliary
bindings, and the current `real704` source rows are promoted, including the
validation `a44bb232` phone list. All **413 exact source/target identity replays**
and [**1,239 native scenario plans**](analysis/r27-package-vocabulary-20261016/current-source-native-preflight.json)
pass; historical authorities and alignments remain unchanged. The
[coordinated regression run](analysis/r27-package-vocabulary-20261016/final-regression-validation.json)
passed **775 tests in 33.71 seconds**.
The active catalog manifest is
`365b452252dc0b5ee4db9f08cd91df2141b56150a4895638955667d5a1053b42`;
the current `real704` manifest is
`73b7dcfa0c72d30ce00c06c730b524055de27540df08042c9952a55e61737b50`.

Fixed public Marks country/product wording was deliberately not rewritten. It
does not supply route, country, description, shipment-net-weight or date labels.
The user-approved extraction scope distinguishes that non-target context from
the actual private references and physical premises that needed repair.

## Email review and one retained OCR edge case

No second clear omission analogous to the four-number validation list was
confirmed for an independently printed, party-owned email list. Missing regex
matches in carrier signatures, source-only employees and goods-area contacts
must not be silently promoted into private-party labels.

An additional target-value sweep found no standalone phone entries with fewer
than five digits among 15,671 phone leaves. Among 5,284 email leaves, one fails a
complete-domain syntax screen: real row 13 prints and labels
`SALMA.NABIL@VTRCA`. That is a retained truncated OCR value, not permission to
invent its missing domain or discard a source-supported contact. This syntax
screen is diagnostic, not a new general email/phone validator.

One real source required an explicit retained decision rather than speculative
editing: row 399, `a282ae30`. The same destination office prints
`Nader.rami@ibis-eg.com` on two pages and `Nader.raml@ibis-eg.com` on the third.
The current target uses the repeated complete `rami` occurrence. The i/l variation
was already present in source OCR and was preserved, not introduced, by manual
anonymization. Direct inspection of the original PDF's second and third
destination-office blocks confirms the same mailbox in both. Therefore this is
**adjudicated, no repair needed**, not an unresolved second-contact omission.
Keep the existing OCR-supported target and both input spellings unchanged. The
PDF resolved identity between retained occurrences; it did not supply a new
PDF-only extraction value. Do not “repair” the OCR, invent a second identity, or
undo anonymization. The source PDF is
[the original three-page B/L](../data/corpora/paddle-bl-swb-filtered-max5/files/2ebe5f9e1212cc2ab7bd2ae821c65fa505842d4bd3d0716614eecb13f2a717ce.pdf).

## Policy clarification

The active `ContactsV7` descriptions already define the relevant semantics. No
schema/hash change is required. The following review checklist makes their
application explicit:

1. Establish the exact printed party block before considering contacts.
2. Inspect the entire owned contact list, not only its first entry. Each complete,
   independently printed telephone number or email receives its own list entry.
3. Preserve each telephone's own printed prefix; do not expand shortened suffixes
   or extensions into guessed full numbers. A seven-digit local number can be
   complete even when it has no repeated international prefix.
4. FAX-only values remain excluded. An independently printed TEL value remains
   eligible even if the same identity also appears under FAX.
5. Deduplicate repeated identities. Do not treat source OCR variants in repeated
   copies as proven new contact identities without adjudication.
6. A carrier footer, issuer signature, separate organization or goods-area
   continuation does not supply contacts to another party merely because a name,
   domain or number resembles it.

Completeness and literal support are different checks: proving that every
existing target occurs in OCR does not prove that all eligible list members were
included. The corrected templates' new checks assert both cardinality and exact
owned occurrences, while ownership remains an adjudicated source contract.

## Validation and side effects

- The original screening was read-only. Subsequent guarded publication applied
  exactly the reviewed changes; the final full-set readback above establishes
  their scope and preservation guarantees.
- The contact rebase changes only the reviewed telephone arrays and ownership.
  Its subsequent source-only privacy extension adds the explicit SB-reference
  and postal bindings described above. Source identity replay preserves the
  complete source text and measured positions; descendant generation changes
  those newly owned private spans without adding extraction fields. Goods,
  packaging, routes and unrelated contacts remain unchanged by these repairs.
- No production runtime code was changed, so there is no new hot-path cost.
- The read-only inventory was changed from retaining decoded records to buffered
  streaming. Single-run observations were **5.20 → 5.00 seconds** and
  **472,556 → 27,852 KiB peak RSS**, with identical input hashes, candidate counts
  and mirror counts. These are bounded audit-tool measurements, not a claim
  about training or synthesis throughput.
- Historical training datasets, source authorities and campaigns remain
  immutable; new inspection descendants are not training additions.
- Four focused target/preservation tests passed (**10.74 seconds**); all four
  audit/test helpers pass repository lint rules.
- Final native synthesis, rendering, full-text review/adjudication and local
  inspection publication pass **6/6 descendants from two source templates**.
  Strict existing postal-spelling guards caught five generated variations in the
  final pass; the existing native postal-correction stage resolved them before
  certification. No new silent fallback or renderer exception was added.
- All final review findings were explicitly adjudicated: six concern permitted
  public non-target Marks context; three are the reviewer's own no-action notes;
  one incorrectly claimed a different preferred VI country name. Direct lookup
  in the pinned registry confirms `UNITED STATES VIRGIN ISLANDS` is correct.
- Native source-transfer/reflow/page-transform coordinates match the independent
  geometry probe exactly for all six records: **569/654 content lines (87.0%)**,
  **12 pages**, and **zero rejected reflow proposals**. Phone-list lines lacking
  source anchors retain explicit blank ` ||`; no coordinates were invented.
- Final geometry validation took **7.02 seconds** including source/provenance
  loading; the subsequent native positions stage took **1.44 seconds** with warm
  dependencies. Existing source rectangle overlaps are not new collisions.
- Two complete inspection passes cost **$0.01653965 total** across 14 accepted
  provider calls, including postal corrections and reviews. The
  [accounting](analysis/r27-contact-completeness-20261016/provider-accounting.json)
  includes all calls: 73,171 input tokens, 11,128 output tokens, including 3,229
  reasoning tokens. No provider-output retries were discarded from accounting.
- The final [certificate](analysis/r27-contact-completeness-20261016/final-certification.json)
  pins the exact staged files and six candidate hashes. The isolated native
  [inspection dataset](../artifacts/kie-synthesis-production/contact-completeness-r27-20261016-private-v2/dataset.jsonl)
  and [positioned renderings](../artifacts/kie-synthesis-production/contact-completeness-r27-20261016-private-v2/positions-reflow-v1/samples.md)
  are not training additions.
