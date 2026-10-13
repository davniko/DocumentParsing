# Synthetic-source anonymization audit

Date: 2026-10-13. Performed by root, without Codex sub-agents. **Audit only:
no dataset, template, source, training config or production code was changed.**
Provider API spend: **$0**. No synthesis, training or GPU inference was started.

## 1. Outcome and scope

The current synthetic set is not fully anonymized. Primary generated companies,
addresses, contacts and equipment identifiers mostly change correctly, but
secondary private references and actors are not comprehensively covered.
This is a privacy/source-casting finding, not a declaration that every affected
sample has an incorrect extraction label.

Population inspected programmatically:

- All **5,000** active synthetic records: **2,500 retained non-UAE** and
  **2,500 replacement UAE** records.
- All **360** active source templates; every source is represented in that set.
- Current dataset:
  `data/curated/mpci-bl-real600-synthetic5000-uae-coverage-r19-positions-v1`.
- Current registry: `artifacts/synthesis-templates/mpci-bl-v7-reviewed`.
- Real training/validation records were not anonymized or relabelled.
- Original source PDFs and source snapshots remain original real data. Historical
  campaign outputs are evidence, not included in a claim that the repository
  itself has been anonymized.

The main source-local inventory identifies **458 deduplicated source/value
replacement candidates across 202 families**, supported by explicit private-role
ownership, identifier captions or direct inspection. These occur in **2,853
distinct synthetic documents**: **1,455 retained** and **1,398 replacement UAE**.
The number counts documents containing at least one captured finding, not
independent defects or a universal privacy-recall measurement.

Within that population, **41 documents have directly confirmed original
identity/reference material in their targets**:

| Target-bearing residue | Documents | Location |
| --- | ---: | --- |
| `ML250031` | 34 | `documentPatch.masterBillOfLadingNumber` and printed booking/master-B/L references |
| `LKW-HANDEL M.NOCK` | 6 | `documentPatch.parties.shipper.name` and printed shipper |
| `ARABIAN GULF MARINE TRADING COMPANY` | 1 | Delivery-agent name; only the office suffix changed |

These three groups are disjoint. Their labels agree with their OCR: the problem
is retained source identity, not invented or ungrounded supervision. Privacy
repairs must change their labels and corresponding OCR together.

There are also **1,077 deduplicated source/value review candidates** after the
initial role dispositions. This is **not** 1,077 additional confirmed privacy
violations or a requirement to manually read 4,216 documents. It includes
generic legal paragraphs, public agency context, product/model codes, locations
and weakly typed footer strings. They require source-level disposition, and
their descendants can then be handled together. Broad matching alone cannot
authorize edits. The only unresolved target-bearing code in this screen is
`PC003211`, repeated in 13 goods descriptions; its commercial-reference versus
public product-code role has not been established.

## 2. Detailed inventory and reproducibility

The audit outputs are under
[`docs/analysis/synthetic-anonymization-audit-20261013`](analysis/synthetic-anonymization-audit-20261013/):

- [Ownership-supported replacement inventory](analysis/synthetic-anonymization-audit-20261013/CONFIRMED-INVENTORY.md):
  458 source/value tasks, source lines, current affected document counts,
  target-document counts and owner keys. The file name distinguishes this list
  from the broad screen; it is not certification that every candidate has a
  fully specified replacement boundary/alias policy.
- [Machine-readable replacement inventory](analysis/synthetic-anonymization-audit-20261013/confirmed-replacement-inventory.json):
  full source/document IDs and deduplicated evidence references.
- [Full contextual inventory](analysis/synthetic-anonymization-audit-20261013/INVENTORY.md):
  source values, matched descendant excerpts and source ownership.
- [All candidates](analysis/synthetic-anonymization-audit-20261013/inventory.json),
  [role dispositions](analysis/synthetic-anonymization-audit-20261013/role-dispositions.json),
  [disposition summary](analysis/synthetic-anonymization-audit-20261013/adjudication-summary.json),
  [source census](analysis/synthetic-anonymization-audit-20261013/sources.json),
  and [input hashes](analysis/synthetic-anonymization-audit-20261013/input-hashes.json).
- `inventory.py` and `adjudicate.py` are reproducible audit utilities in the
  gitignored analysis directory, not new production pipeline modules. Their
  role-screen rules are not runtime replacement regexes.

For each captured value the JSON records original source lines/context, the
current carrier, declared region keys and recipes (or unowned/static status),
all affected synthetic document IDs, cohort counts, any target paths and
representative descendant text. Multiple detector categories can describe one
source/value; the actionable table deduplicates those. Different punctuation
surfaces and partial name components still need an explicit shared-actor/alias
decision before being combined into a single repair owner.

**These reports contain original private source values.** Keep them and original
source artifacts access-controlled; do not distribute the audit as if it were a
sanitized synthetic dataset.

## 3. Taxonomy of remaining work

Counts below are overlapping document sets. Do not sum them. They describe
captured ownership/caption-supported candidates, not an extrapolation from the
earlier random 100-document audits.

| Category | Captured source values | Source families | Documents | Typical location / finding |
| --- | ---: | ---: | ---: | --- |
| Private registration/tax/customs identifiers, including CPF | 246 | 110 | 1,418 | Shipper/consignee captions, customs continuations, signer footers |
| Commercial/shipment references | 172 | 124 | 1,860 | Invoice, purchase order, booking, customer/account, service contract, Marks/footer references |
| Private phone/fax occurrences | 20 | 15 | 135 | Fax remains original while generated party/telephone changes |
| Individual names immediately above CPF | 4 | 4 | 108 | Carrier-signature footers; three distinct person-name spellings across four sources |
| Further directly inspected person-name/contact context | 3 | 3 | 43 | Includes an ATTN name and a CPF-free signature-name variant; overlaps preceding groups |
| Personal email mailboxes | 3 | 3 | 42 | Two distinct personal addresses in carrier/issuer contact areas |
| Secondary organization lines | 8 | 8 | 84 | Exporter/account/invoice-party context, represented companies, changed office of an unchanged company |
| Identity fragments in current name variables | 9 | 7 | 55 | Whole or partial private company names in continuation/Marks/footer text; overlaps organization/name findings |
| Directly inspected private postal fragments | 2 | 2 | 19 | Secondary Morris/Sonvigo address blocks, not primary generated addresses |
| Original master B/L reference | 1 | 1 | 34 | Structured master-B/L field plus printed aliases |

The broad uncaptained-code detector additionally captures private values already
listed above and unresolved codes. It is not a separate additive defect count.

### 3.1 Registration, tax and customs identifiers

Examples include importer VAT/tax numbers, exporter company registrations,
ACID/import references, CNPJ/CPF, EORI and secondary agency/customer IDs.
Many are in fields dropped from the reduced extraction schema, so the text can
remain faithfully rendered while exposing original values.

Example family `00a165a2`: six retained descendants still contain original
`205055745`, `2050557452024051307`, `6098184859`, invoice
`SHIN-ELQ-240522_2415` and order `4010002415`. The current auxiliary declarations
already cover these identifiers; the replacement UAE descendants did not retain
these particular original values. This is a concrete **old-sample backfill**
case, not evidence that those exact current recipes need another redesign.

Other families show incomplete coverage upstream. For example, `2f680d56`
declares an anonymizing recipe for one occurrence of `428498523`, but a separately
owned exporter occurrence retains the same original digits. `5c5e51a9` changes
the customs importer occurrence of `530578166`, while consignee/notify tax
occurrences can retain it. The inventories distinguish owners, not just digits.

Public carrier corporate registration/VAT constants are separately retained,
including `199804855M` and the inspected public carrier VAT/GCR declarations.
An individual's CPF is never exempt merely because it appears beside a carrier.

### 3.2 Commercial references and tracking values

Typical captions: invoice/proforma, purchase order, booking/BKG, customer code,
service contract, shipping order, consolidation reference, sub-reference,
export declaration, lot/batch or shipment-specific Marks identifiers.

Examples: `NV250000217`/`NV250000218` in an exporter continuation;
`33101411044` beside an original invoice party; source booking/customer codes
under new generated parties. A booking or customer reference assigned by a
public carrier is still shipment/customer-specific—it is not the same as the
carrier's public corporate identity.

Keep quantities, totals, dimensions, HS/UN classifications, equipment-type
aliases, legal clause numbers, tariffs and public voyage context out of an
indiscriminate identifier rewrite. The screen did catch tariff values such as
`12.00` in a long paragraph mentioning invoices; those were rejected as privacy
findings. A historical voyage in booking prose is separately flagged for owner
review, rather than classified automatically as a private booking code.

### 3.3 Original master B/L in targets

Family `e827ecdd` prints:

```text
BOOKING NUMBER ML250031
...
MASTERBILLNO: ML250031
```

All 34 current descendants retain `masterBillOfLadingNumber: ML250031`.
The actual `public_identity_updates` function randomizes `.billOfLadingNumber`,
`.voyageNumber`, equipment identifiers and seals, but does not select
`.masterBillOfLadingNumber`. A targeted executable probe confirms the omission:
given ordinary and master B/L fields containing the same source value, the
ordinary field changes and the master field stays source-fixed.

This requires a host identifier-field coverage fix and source-local alias
coverage, not just deleting a footer. The known linked booking/master surfaces
must follow the same new value. The corresponding target must change.

### 3.4 Secondary private companies and postal blocks

Examples directly checked include:

- `LANS LOGISTICS LLC`: original invoice-party occurrences survive in 13
  descendants even though the primary shipper is generated anew.
- `MORRIS PRODUCTS S.A. TOLLING` and its avenue/address fragment survive in six
  descendants.
- `WESTON IMPORTERS LTD` and `BRODIES LLP` remain in represented-company/C/O
  continuation context. The selected primary A target must not acquire B's
  name/address during anonymization.
- `EL GALLAD COMPANY FOR TRADING` survives in a Marks occurrence despite changed
  primary party wording.
- `SONVIGO INTERNATIONAL LTD` and the Mauritius postal continuation require
  source-only exporter ownership rather than a second unrelated company draw.
- `LKW-HANDEL M.NOCK` remains as the complete shipper target in six retained
  examples. This is a privacy repair separate from the previously resolved
  primary-A versus represented-B labeling policy.
- One descendant uses `ARABIAN GULF MARINE TRADING COMPANY (SHARJAH OFFICE)`;
  the source is the same company with `(ALEXANDRIA OFFICE)`. A changed branch
  and address do not anonymize the original company identity.

Use an already generated entity when the source proves an alias of that entity.
Generate a separate secondary actor only when it really is a different actor.
Do not infer that every repeated old spelling denotes the primary shipper.

### 3.5 Fax and contact residue

Five source families alone have **33 documents** where an original primary-party
telephone survives in an unlabelled fax occurrence. Examples:

```text
Fatima Khalid Almansoori TEL:+971 50 123 7043 FAX:20100546112
TEL: +212 6 50 12 15 26 FAX: (60) 3 7877 7890
```

The current labels correctly contain the new telephone numbers, not those old
fax values. The privacy repair belongs to the fax owner/alias, not to inventing
a new fax target. Including further explicit private-party phone/fax owners
expands the captured contact scope to 135 documents across 15 families.

Personal addresses `hana.christina@lxpantos.com` and
`nicolascastillo@trazalogistica.cl` remain in 42 descendants across three
families. The underlying public company/domain can remain; the identifiable
individual mailbox needs anonymization. Generic inspected carrier mailboxes,
such as organizational `info@...` accounts, are a separate keep category.

### 3.6 Signatories and personal registration numbers

Four source families carry original names immediately above CPF declarations:

- `Carolina de Oliveira Caligiuri` / `29787875897` in two families, 47 descendants.
- `FABIANO SILVEIRA` / `133.848.188-65` in one family, 25 descendants.
- `João Marcos Pauli` / `068.474.059-10` in one family, 36 descendants.

Those are **108 documents**, of which **83 are replacement UAE** examples.
The same person can appear more than once in a document. Another source has the
CPF-free `Cardinale de Oliveira Caligiuri` spelling in 15 descendants; an
`ATTN : Mohamed YASSIN` contact survives in 14.

Preserve the public carrier, signing-agent company and legal clause. Replace
the personal name and associated individual ID. A blanket exemption for every
`carrier:*` binding would incorrectly preserve these people.

### 3.7 Uncaptioned codes, brands and ambiguous context

The inventory retains candidates for uncaptioned footer codes, product/SKU
codes, Marks identities and secondary agency roles. Examples requiring a role
decision include `PC003211` in 13 descriptions and an issuer licence
`MTO/DGS/193/SEP/2025`. Product model identifiers and public licences are not
automatically private customer identifiers.

One useful falsification: source `23080` occurs in an exporter-ID declaration
and a postal block. The current exporter ID is changed; matching digits survive
in `IFZA BUSINESS PARK DDP, 23080-001-DUBAI`. This does not authorize a global
`23080` replacement or count as an unchanged exporter-ID occurrence. Its postal
owner is the unit to inspect. Similarly, generic corporate wording such as
`FREIGHT SOLUTIONS`, `CO.,LTD.` or `LIMITED SIRKETI` can appear in new company
names without preserving the original company identity.

## 4. Why earlier extraction audits allowed this

The earlier reports explicitly scoped complete anonymization of non-target
carrier/Marks/footer text out of extraction acceptance. They recorded secondary
privacy findings separately from label fidelity. This audit introduces the
requested full-private-source anonymization objective; it does not silently
reinterpret those prior extraction passes as privacy certifications.

The actual implementation exposes five different states:

1. **Contracted primary identities:** company/address/contact wording is
   generated; host identifiers are sampled. This mostly works.
2. **Historical binding exists, but current target no longer owns it:** the
   reduced schema dropped Marks, forward/export references and carrier targets.
   A historical binding without a supplied surface can retain original text.
3. **Auxiliary recipe exists for only some aliases:** one customs occurrence can
   change while an old tax/footer/fax occurrence under another key stays original.
4. **Literal has no active mutable region:** the byte renderer intentionally
   copies all untouched source bytes. Compilation's old literal-risk certificate
   is not a privacy certificate under the new requirement.
5. **Current upstream ownership is already fixed, but old accepted descendants
   predate it:** those require dataset backfill, not another source redesign.

Code evidence:

- `curated_auxiliary.py` module policy expressly retained unrelated third-party
  commercial actors. `auxiliary_surfaces` skips undeclared recipes.
- `curated_templates.py:569` documents unchanged auxiliary surfaces; at line709
  an unchanged region uses its source text, and untouched bytes are preserved.
- `curated_campaign.py:325` contains the explicit host identifier-field selector;
  the executable master-B/L omission probe targets this real function.
- `curated_auxiliary.py:306` currently renders many auxiliary identifier recipes
  as digits plus an optional prefix; it does not universally preserve arbitrary
  mixed letter/digit/separator formats.

Thus finding a binding is not enough to conclude it was anonymized, and finding
an anonymized occurrence is not enough to conclude its aliases were covered.

## 5. Recommended least-complex repair path

This is a proposed next pass, not implemented work in this audit.

### A. Inventory source-owned privacy spans once

Add a small privacy disposition for original values/regions:

- private party/individual;
- private party postal/contact occurrence;
- private shipment/commercial/registration identifier;
- public carrier/issuer/company constant;
- public vocabulary, product/model/classification value;
- unresolved owner/role.

Make the disposition occurrence-scoped, including source-only Marks, attachments,
signatures and footers. Reuse existing render regions and auxiliary declarations
where present. Resolve ambiguous values at source level; do not ask an agent to
reclassify the same footer independently for every descendant.

### B. Deterministic format-preserving identifier generation

The existing `generators.surface_pattern` / `generate_from_surface_pattern`
already preserve ASCII letter case, digit positions and literal separators,
exclude the original complete value, and support deterministic streams.
This is the appropriate starting point; no national tax/registration validator
or country-specific format engine is needed for the requested generic codes.

Examples from this audit's actual unchanged-code probe:

```text
SHIN-ELQ-240522_2415 -> QFUY-VYE-883771_5094
100/496/296         -> 513/503/557
36AABCA7371G1ZQ     -> 76PIEYT8578N8QU
068.474.059-10      -> 638.591.103-93
```

All six tested surfaces generated a different value, retained the same encoded
format and reproduced identically on a repeated call. The fixed prefix can be
kept when explicitly intended, with only the private suffix sampled. Do not
invent checksum/national-validity requirements for these auxiliary codes.

Generate once per source-local logical identity **and descendant**, then replay
all proven aliases. Do not use one global old-string replacement across unrelated
actors, numeric totals, postal values or carrier legal clauses. Keep existing
specialized equipment/check-digit behavior for labeled container identifiers.

### C. Reuse existing lexical generation only where needed

When the secondary occurrence aliases a generated party, reuse that party's
already generated name/contact/address. For independent people/companies,
include a compact source-only name/address/contact request in the existing
wording call. Do not add a full-document rewrite or per-reference LLM call.
The majority of this scope is deterministic identifier work.

### D. Backfill the active dataset without rebuilding unrelated data

Apply certified source/occurrence replacements to the current synthetic records.
Preserve all unrelated text and targets. Update targets for the 41 confirmed
target-bearing cases and any other subsequently confirmed private label field.
Keep label-free non-target repairs label-free. No change is required to real
training/validation samples under this synthetic-only request.

Same-format IDs keep character counts/separators and normally physical line
counts, so their coordinate suffixes can remain unchanged. Names/postal changes
must preserve or revalidate line ownership/geometry using the established
position pipeline; never recompute positions by globally matching a short code.

### E. Close the same upstream paths

Ensure all private regions receive a replacement, including dropped-target and
source-only fields. Add the master-B/L field to identifier sampling with its
proven aliases. Fill missing auxiliary alias declarations. Privacy acceptance
must fail for an unresolved private occurrence; extraction-schema validity alone
cannot act as the privacy gate. Preserve public carrier exceptions narrowly,
excluding individual names/CPF/personal mailboxes in their blocks.

Re-run source/descendant residue checks across all 5,000 records after repair,
then synthesize targeted descendants for the repaired source mechanisms. No
blanket regeneration of parties/goods, new geography sampling or GPU work is
needed to address the confirmed scope.

## 6. Validation, false positives and limits

Performed in this audit:

- Read all 5,000 synthetic JSONL records; validated original/positioned input
  hashes before using them.
- Built the current owned+auxiliary blueprint for all 360 source cases to inspect
  actual mutable-region ownership, rather than rely only on old template slots.
- Scanned all selected source values across each source's actual descendants;
  checked corresponding target paths; separated retained/replacement cohorts.
- Pinned and rechecked **1,448 dataset/registry/source files**. No audited input
  changed during the run. The current train SHA-256 remains
  `001fa2a63ac426109993547524d0cc5b5ca7e0408b7a8bdbc8a681486fdb5d49`.
- Directly inspected flagged names, person/CPF neighborhoods, fax aliases,
  personal mailbox ownership, the master-B/L field and representative old/current
  auxiliary-recipe occurrences.
- Ran six audit-helper positive/negative checks. A partial email suffix
  `CUSTOMS@GMAIL.COM` inside a new `coralbayfuels.customs@gmail.com` is rejected
  as evidence of complete-source-email survival. Near-identical identifier digits
  and larger containing identifiers are not exact source-value matches.
- Ran six deterministic format/replay/exclusion probes against the existing
  identifier generator; **6/6 passed**.
- Executed the actual ordinary/master-B/L host-sampling counterexample.
- Existing focused generator/auxiliary identity tests: **27 passed**.
- Audit utilities compile successfully. Final population scan: **54.641 seconds**,
  peak process RSS **413.8 MiB**. Production throughput is unchanged because no
  production code was edited.

The scan selected 13,852 category-specific source candidates; 2,839 survive in
some descendant. Category overlap and generic/public text account for a large
part of this number. The dispositions explicitly preserve carrier constants and
reject non-identifying wording; the raw screen's 4,770 document hits are **not**
4,770 confirmed privacy defects.

This was a whole-population source-derived scan plus targeted direct semantic
inspection, not a claim that a model/human manually read all 5,000 complete
documents in this turn. Non-detection is not a privacy certificate: uncaptained
personal names, OCR-mangled identifiers, partial retained addresses and unknown
secondary roles can evade any finite list of lexical heuristics. The inventory
therefore gives exact captured scope and a source-owned completion path, while
keeping unknown-role candidates visible rather than declaring them clean.

The original real/source artifacts and this audit must remain private even after
the synthetic export is repaired. Generated realistic identities can also
coincide with unrelated real businesses; source anonymization means breaking
the connection to original parties/identifiers, not proving a worldwide company
name/domain has never existed.

## 7. Next action

Proceed with a **source-owned anonymization/backfill pass**, not regeneration of
the complete dataset. The identified identifier/reference workload is bounded
and mostly deterministic. First resolve remaining role candidates by their
source owner, then enforce complete private-occurrence coverage upstream and
run the exact same full-population audit after repair. Keep public carrier
organizations, corporate websites/contacts and legal constants; anonymize their
individual signatories and personal mailboxes separately.
