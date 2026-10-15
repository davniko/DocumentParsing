# Private-contact anonymization: dataset and synthesis repairs

## Scope and completion contract

This pass implements the approved private phone/fax, personal mailbox and
source-only individual/CPF repairs on the existing R20 working dataset. It
covers the retained non-UAE 2,500, current replacement UAE 2,500 and previous
superseded UAE 2,500 synthetic records. The previous UAE set remains a sidecar,
not extra training data. The real 600 training and 60 validation records remain
unchanged; their OCR and labels are source evidence, not synthetic identities.

The active dataset remains **600 real + 5,000 synthetic**, with 60 real
validation records. No training is launched or configuration changed here.

The repair certificate is component-specific: it proves the authorized contact
changes, their source ownership, OCR/label agreement, exact unrelated-data
preservation, recurrence guards and fresh-render/geometry checks. It does not
reclassify fiscal, booking or invoice identifiers as contact data, or call a
fresh inspection descendant a gold training record merely because its contacts
pass. Remaining non-contact privacy categories stay in the existing
[anonymization inventory](kie-synthetic-anonymization-audit-2026-10-13.md).

Working dataset:
[R20](../data/curated/mpci-bl-real600-synthetic5000-uae-coverage-r20-party-masterbill-positions-v1/README.md).
Evidence and exact pre-edit backups:
[audit directory](analysis/synthesis-private-contacts-repair-20261013/).

**Completed and published in the R20 working copy.** The
[completion certificate](analysis/synthesis-private-contacts-repair-20261013/completion-certificate.json)
pins the applied data and validation evidence; the
[publication verification](analysis/synthesis-private-contacts-repair-20261013/publication-verification.json)
checks the destination manifest and all nine dataset-file hashes. The counts
below describe actual applied changes, not proposals waiting for approval.

## Changes to existing samples

| Cohort | Records checked | Records repaired |
| --- | ---: | ---: |
| Retained non-UAE | 2,500 | 161 |
| Current UAE replacement | 2,500 | 278 |
| Previous superseded UAE | 2,500 | 200 |
| Total synthetic | 7,500 | **639** |

There are **439 repairs in active training** and 200 in the superseded sidecar.
Of the 639 repaired records, **131 change email targets** and **508 change input
text only**. There are 209 corrected email scalar paths, including repeated
party occurrences. No party name/address target, contact-name target, phone
target, cargo, route, date, equipment, package or allocation target is changed.

The authorized jobs comprise 268 source-only personal-name replacements, 253
mailbox replacements and 510 numeric/alias replacements. These are job counts,
not unique document counts; an owner can occur several times. The edit receipts
contain **1,075 exact text edits**. Among numeric jobs, 144 replace individual
CPF declarations and 366 cover the reviewed phone/fax owners and aliases.

Original OCR line count, physical page boundaries and every coordinate suffix
are preserved. Each target-bearing mailbox is replaced in OCR and labels as
one fact; its repeated rendered occurrences share that value. Case-sensitive
email values are not uppercased as prose. Canonical input hashes and mirrored
dataset files are rebuilt and checked.

## Diagnosis and why the old paths missed these values

### A new domain did not necessarily anonymize a recipient

Some historical contacts had changed domains while retaining the original
personal name or account identifier. Changing `_` to `.`, case, or separator
style does not anonymize the same recipient. A copied numeric account identity
inside a longer new local part likewise remains the original identity.

The source inventory reviewed 212 target-email paths in 128 template families:
159 personal/private-account paths and 53 generic/public department-style paths.
All now have explicit recipient classification. Generic `info`, operations or
customer-service wording can remain when it is not a personal/account identity.

The initial refinement showed that exposing the original recipient encouraged
the model to remix that same identity. The repaired prompt omits private source
recipients, gives new company/contact-name context, and supplies only formatting
style. Where no current human name exists, it asks for a fictional recipient.
Native structured output still validates complete mailboxes, and semantic
validation rejects retained identities. The unsuccessful proposals and one
bounded recorded correction remain in the cost/provenance ledger.

### A changed recipient could still retain a private company domain

A further source-to-target scan found private corporate hosts retained under
new recipients. Those contacts now also change their business domain. The
registry explicitly separates 148 company-domain target paths from 64 public
mail-provider-domain paths. Shared provider domains can remain; they do not
identify the original private business. Source-only public carrier/issuer
domains are a separate, explicit context and can remain while their named
recipient changes. Email and website domains are still independent: this pass
does **not** restore the previously rejected shared-domain requirement.

Both recipient and domain classifications must cover every source email target
before a campaign starts paid work. A new source cannot silently inherit a
generic recipient or public-provider exemption because metadata is absent.

A fresh probe also exposed an unnamed party website copying the old private
company hostname. The source-style hint now exposes only its presentation
form (for example, `WWW` or the scheme), not that hostname, and the field
description asks for fictional company wording. The repaired ordinary synthesis
path passed the repeat probe. A scan of all 7,500 existing synthetic records
found no corresponding unchanged private website host requiring a dataset edit.

### Public carrier companies and private people were conflated

Named employees/signatories had sometimes been frozen with their public carrier
company. This pass adds eleven source-only human-name owners, covering explicit
signatories, attention contacts and the two HANA CHRISTINA occurrences. These
actors do not become invented shipper, consignee or carrier target parties.
The public company, signature caption and unrelated dates remain intact.

The full typed-owner sweep caught three additional named carrier signatories
beyond the original shortlist: source prefixes `0dbc75a0`, `27e75cc8` and
`9ea52ed1`. Each now has a generated individual owner. Future preflight rejects
a typed private person/contact-name binding that remains `carrier_static`.

One source grouped two human-name occurrences and a CPF number into a single
historical binding. Another CPF used a generic tax-identifier key. The repair
uses printed CPF evidence, not key-name guessing: the mixed owner is split,
names are lexical and the numeric declaration is a separate exact owner.
All four CPF-bearing families are covered. Generated CPF-like codes preserve
the numeric envelope; national validity/checksum is not an objective here.

### Fax and repeated phone aliases needed the correct owner and channel

Source-only fax numbers do not need a new label field. They need typed auxiliary
owners that follow the corresponding generated party's sampled country. Where
the source explicitly prints one subscriber as TEL and FAX, the alias reuses
the existing generated target phone; a second unrelated draw would break the
printed relationship. Distinct fax owners use fixed-line-capable number types,
not the mobile-oriented phone generator.

An issuer phone is explicitly source-country-owned rather than attached to a
new shipment destination. Local and international variants of the same source
subscriber use a canonical identity. Literal captions and `(8LINES)` remain
outside the numeric identity. Maximal matched spans are owned first, preventing
a short local-number match from being replaced again inside its full variant.

## Templates and production code

**139 of 360 source contracts have updated policy/ownership metadata.** Only
eleven require new lexical human-name variables; the numeric extension covers
32 reviewed owners across eighteen source families. Original case source OCR,
source-positioned OCR, target gold, historical template and alignment files are
preserved. Registry hashes pin the revised contracts and shared metadata.

| Component | Responsibility |
| --- | --- |
| `curated_private_contacts.py` | Validate exact private-human and source-only mailbox ownership; require recipient/domain classification; preserve separation from public companies and extraction roles |
| `curated_contacts.py` | New-person context, redacted private recipient styles, independent domain policy, native mailbox schema and retained-identity/domain rejection |
| `curated_campaign.py` | Pre-paid readiness, source-only human wording, contact generation and identical render/replay checks, source-only binding surfaces |
| `curated_auxiliary.py` | Source-envelope numeric identifiers, explicit phone/fax channel and country ownership, coherent canonical aliases |
| `template_compiler/contact_values.py` | Deterministic fixed-line-capable fax values with validity/type/region checks |
| `curated.py` | Source-contract field descriptions distinguish mutable private employees/contacts from public fixed companies |
| `curated_wording.py` | Reject known generation-template tokens leaked into generated text, without rejecting unrelated product brace notation |

Two stale duplicate source-authority rows were also reconciled to their already
certified registry gold. One lacked a previously corrected gross weight; the
other had old contact values. These are duplicate synthesis-authority updates,
not changes to real training/validation labels. The exact two-row receipt and
old authority file are retained separately.

The spatial-authority manifest was repinned to those reconciled duplicate
labels. Its source/Paddle/alignment geometry pins remain unchanged. The ordinary
geometry loader initially rejected the stale label pin; the explicit
[reconciliation receipt](analysis/synthesis-private-contacts-repair-20261013/source-authority-reconciliation.json)
records the correction rather than bypassing that check.

There are no source-ID branches in production generation. Source-specific
ownership belongs in reviewed metadata. The one-off repair inventory maps
explicit source owners to those policies and retains every authorized edit.

## Validation and falsification

### Existing dataset

1. Independently replay every edit from the pre-edit record and the separately
   authorized owner jobs. A self-consistent forged receipt is insufficient.
2. Permit only the declared email target paths to change; compare the complete
   reconstructed row, not a partial field allowlist.
3. Validate all 7,500 synthetic rows, schema, hashes, cross-file mirrors, page
   boundaries and coordinate suffix preservation. Untouched rows stay
   byte-identical, including all 600 real training and 60 validation rows.
4. Sweep source private recipient identities, source-only names, original CPF
   declarations and reviewed phone/fax source values independently of the edit
   receipts. Check private business-domain elimination as a separate criterion.
5. Falsification controls deliberately restore an old label/occurrence, remove
   coordinates, alter an unrelated company, stale an input hash and forge a
   public-company edit receipt. All six controls are rejected.

Latest measured whole-dataset replay takes 19.733 seconds; the independent
source-identity sweep takes 6.262 seconds. The registry compile/hash check covers
all 360 sources and 25,889 owned regions, not only repaired source families.
The complete replay process, including imports, took 27.81 seconds with
202,564 KiB maximum resident memory and no swaps.

### Fresh synthesis and coordinates

Two fresh ordinary descendants are generated per updated source. These test
source-only names/mailboxes, party contacts, numeric envelopes, repeated aliases
and the original generation/render/replay path. They use GLM-5.3-Flash with low
reasoning and native PydanticAI structured outputs; original private recipients
are not supplied as examples. Actual saved provider request/response bytes,
model, usage and current candidate hashes bind every full-text review.

Geometry uses the existing measured-source transfer, reflow and coherent page
transform implementations. The independent checker verifies integer normalized
centroids, source ownership, page bounds, transformations and collisions. An
unpositioned line remains explicit; rejected reflow proposals are not approved
as geometry. No inspection sample is added to training.

The full-text reviewer also sees non-contact fields. Its complete findings are
retained separately, rather than being silently erased or used to expand this
repair into cargo/route/registration relabeling. Contact-component certification
and whole-document gold approval are different recorded statuses.

The final [pilot certificate](analysis/synthesis-private-contacts-repair-20261013/pilot-certification.json)
covers **278 fresh descendants from all 139 updated templates**, with no
unresolved contact-component defects. Geometry validation checks 532 pages,
21,955 in-page rectangles, 358 generated blocks and 1,070 exactly reconstructed
generated centroids. The 2,586 inherited source-rectangle overlaps are recorded
as source layout, not silently reclassified as invented collisions.

The native full-text reviews produced 59 observations. Their complete
[adjudication inventory](analysis/synthesis-private-contacts-repair-20261013/fresh-render-review-inventory.md)
retains both rejected review claims and separate follow-up concerns:

- Numeric shipment arithmetic, exact HS-to-product classification, duplicated
  source TEL/FAX aliases and generic consignee-header wording were adjudicated
  against the existing extraction objective, not used to rewrite unrelated gold.
- Remaining registration/invoice/batch identifiers and non-contact Marks text
  belong to the next anonymization inventory, not to the completed mailbox,
  private-person, CPF and reviewed phone/fax slice.
- Inspection-only carrying/origin wording and caption/packing/identity concerns
  retain explicit separate dispositions. They do not receive blanket training
  approval and these 278 inspection descendants are not added to training.

The exact edit-replay checks confirm that none of those unrelated fields were
changed by this dataset repair. This is a completed, bounded repair with a
specific pass meaning, not a claim of full-dataset anonymization.

## Performance, costs and publication evidence

The fixed-line fax generator was checked across all 245 supported phone regions.
Median generation time in six measured countries remains tens of microseconds;
the largest measured phone-to-fax difference was 6.52 microseconds. The initial
private-recipient guard added 16.96 microseconds per request to native output
unpacking; traced retained memory decreased after GC. Final guard benchmarks,
test counts, all-attempt costs, source/sample inventories and publication hashes
are recorded in the audit directory and completion certificate.

The dataset repair itself uses small batches of names/mailboxes, not whole-
document LLM rewrites. Numeric jobs and dataset validation are local. Fresh
whole-render semantic reviews are accounted separately from repair wording;
rejected/corrective billed proposals are not excluded from totals.

The final regression run passed **4,661 tests**, with nine skips for the pinned
DG compilation, SDV/SDMetrics and dedicated benchmark environments. The three
warnings are existing seaborn/Matplotlib deprecations. The run took 130.05
seconds. The latest private-contact subset passed all 47 tests, including
invalid metadata, source-owner repetition, private recipient/domain retention
and unnamed-website context controls. Ruff and whitespace checks passed.

Final contact-unpacking benchmark medians were 422.837 microseconds before and
398.388 after. Building the contact prompt changed from 0.526 to 1.269
microseconds; the wording-template-token guard changed from 15.197 to 18.080
microseconds. These added local checks are negligible beside network generation.
Contact benchmark traced peak memory was 268,884 versus 292,615 bytes; the
wording guard peak was 12,796 versus 12,556 bytes. These are measured small
allocations, not a claim that every workload becomes faster or uses less memory.

**Recorded paid cost: $0.47910953 total**, approximately **$0.48**, including
repair wording, probes, all fresh synthesis/reviews and billed corrective or
rejected proposals. Names/mailbox repair wording itself cost **$0.01609345**;
numeric replacements and validation used no paid model. The
[cost ledger](analysis/synthesis-private-contacts-repair-20261013/cost-accounting.json)
keeps the separate operation totals and counts cache reuse only once.

Backups precede mutation. Staged data files are published only after scoped
validation succeeds, then destination hashes and the updated manifest are
verified. Earlier R19 data, real source PDFs, real OCR/labels and prior campaign
artifacts are preserved. No model training is started.
