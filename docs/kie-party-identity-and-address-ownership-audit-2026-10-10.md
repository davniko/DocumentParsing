# Party identity and address ownership audit — 2026-10-10

## Subsequent adopted policy and published repair

The user selected **primary printed company A with A-owned details**, with B
retained in audit metadata. This is now implemented in the current dataset,
active source targets/templates, extraction descriptions and synthesis flow.
Genuine care-of routing remains in addressLine. The
[completed repair report](kie-party-date-route-repair-2026-10-10.md)
contains the exact counts, fresh synthesis results and post-publication checks.
The audit-only statements and alternatives below describe the earlier diagnosis,
not the current publication status or current field instructions.

## Outcome and scope

This audit confirms a **target-definition inconsistency**, separate from the already identified anonymization defect. Some labels combine two or three organizations into one `name`; other labels omit a represented organization in materially similar source structures. A composite name is then paired with only one of the printed addresses. Exact agreement between a synthetic label and its rendered text does not resolve that semantic mismatch.

**Audit only:** no dataset, source OCR, label, template, schema, synthesis code or extraction instruction was changed. No paid inference, synthesis or training was run. The proposals below are not adopted policy.

The audit covers the current 600 real training records, 60 real validation records and 2,500 synthetic records in:

- [Current dataset](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real600-synthetic2500-v7-positions-boundary-repaired-v1)
- [300-template registry](/mnt/d/projects/documentparsing/artifacts/synthesis-templates/mpci-bl-v7-reviewed/manifest.json)
- [Complete source and family inventory](analysis/party-identity-20261010/INVENTORY.md)

Current mixed-dataset targets are authoritative for this audit. Older per-source `labels.json` files are not assumed to contain the same revisions.

| Inventory | Count | Meaning |
| --- | ---: | --- |
| Real documents with relationship wording in current party targets | 47 | 44 training + 3 validation; candidate cases, not 47 proven defects |
| Additional real documents found through OCR continuations | 5 | Their current targets omit the relationship/secondary entity |
| Combined real review inventory | 52 | 49 training + 3 validation |
| Synthetic documents with relationship wording in current targets | 181 | From 29 source-template families |
| Compiled families in the combined source/descendant review cohort | 32 | Includes families whose descendants omit a source relationship |
| All current synthetic descendants of those 32 families | 268 | Complete family-level follow-up cohort, not 268 confirmed defects |
| Synthetic-only introduction of care-of wording | 5 documents / 4 families | Requires review of whether it is a legitimate postal qualifier or an added actor |

### How the inventory was established

1. Scan every current real and synthetic party target for ON BEHALF/O/B, CARE OF/C/O, ACCOUNT OF/A/C and explicit agent-of/agent-for wording.
2. Search source OCR as well, so labels that omit a relationship do not disappear from the inventory.
3. Inspect contexts to distinguish actual shipper/consignee/notify identities from carrier signatures, freight-payment clauses, generic terms and printed template captions.
4. Resolve source PDF links through each curated alignment's `pdf_id` and the Paddle document inventory; curated OCR IDs and PDF IDs are not interchangeable.
5. Inspect selected PDFs for the difficult ownership shapes, including explicit two-address boxes and linked continuations; inspect nested relationships in the OCR and current targets.
6. Join source IDs to the compiled registry and all current descendants, not only descendants retaining the marker.

This is an exhaustive **current-target marker census**, supplemented by source-OCR context review. It is not proof that every unmarked multi-entity case in the corpus has been found. In particular, uncaptioned companies or people and OCR-corrupted relationship wording are outside the marker census. The 52-source inventory is not a claim that every source received a complete visual PDF or whole-document semantic audit.

The initial broad detector was tightened before reporting: matching dotted A.C indiscriminately can mistake a legal suffix such as S.A.C. for an account-of relationship. Carrier boilerplate mentioning care-of, freight-account text and numeric account references must also remain separate from actual party relationships.

## 1. What the words establish—and what they do not

### On behalf of / O/B

`A on behalf of B` names two actors and states that A is acting for B in some relevant capacity. It is not one company's legal name. In ordinary agency terminology, B is the represented party/principal; calling the first-listed A the principal simply because it is printed first was incorrect.

But the wording alone does **not** prove which actor is the ultimate physical exporter/importer, or whose street address follows. A Maersk advisory uses these relationships specifically to identify the freight-contract price owner, with different ordering for on-behalf-of and care-of forms. Its purpose is freight-contract validation, not a universal rule identifying the ultimate MPCI shipper. [Maersk contractual-customer advisory](https://www.maersk.com/~/media_sc9/maersk/local-information/files/asia-pacific/korea/export/customer-advisory/notice-of-contractual-customer-validation-rule.pdf).

### Care of / C/O

Care-of can be a postal routing instruction: A receives correspondence at B's location. In that use, A remains the named party while `C/O B` can be part of its postal address. Removing every organization name from `addressLine` would damage such addresses.

It can also appear inside a more complicated commercial or agency relationship. It must not automatically be treated as equivalent to on-behalf-of, nor should it automatically move B to `notifyParties` or `forwardingAgent`.

### Account of / A/C

A/C followed by an organization records an account/commercial association. It is not automatically a postal care-of or agency instruction. A/C followed by an account number is a different shape again: it is an identifier, not a second company.

### Explicit “as agents of”

This does identify an agent and represented actor. It still does not establish that the agent is specifically a forwarding agent, delivery agent or carrier agent. Those are different roles in the form. Do not infer a form party function merely from the word AGENT.

## 2. What our target and MPCI form actually support

### Current extraction policy

The [current party model](/mnt/d/projects/documentparsing/src/document_ocr/label_schemas/bill_of_lading_v7.py:550) calls this one named party, but explicitly instructs the extractor to preserve A ON BEHALF OF B in a shared named-party postal block. It separately says competing identities/addresses require review.

The [earlier reviewed-dataset decision](/mnt/d/projects/documentparsing/docs/kie-real-reviewed-dataset-build-2026-10-04.md:825) retained a shared A ON BEHALF OF B string when one postal block was provided and distinguished that from combining different companies with different addresses.

Consequently, many compound labels were **policy-directed**, not spontaneous LLM hallucinations. Asking a stronger model to follow the same field description would retain the ambiguity.

### Local form schema

The [MPCI form discovery](/mnt/d/projects/documentparsing/artifacts/mpci-ai-schema/README.md) was recorded at revision `72f9aff735c2581f2ccff74e7fde34c2aea9b265`. This audit inspected that checked-in snapshot, not the other application's current runtime.

A party-function record has arrays of names and addresses, plus country/location/contact fields. The [field catalog](/mnt/d/projects/documentparsing/artifacts/mpci-ai-schema/field-catalog.md:138) identifies shipper, consignee, notify, forwarding and delivery roles.

Those arrays do not, in the inspected contract, encode “A represents B,” identify which address belongs to which company, or establish that parallel array indexes are separate linked organizations. Putting two names in `partyNames[]` therefore does not by itself solve this issue.

There is no inspected, purpose-built represented-principal relationship field. Existing agent roles are not a license to assign every on-behalf-of actor to a particular agent function.

### Official MPCI expectations

Appendix C of the official business specification expects actual ultimate shipper/consignee identity and corresponding address information, rather than substituting logistics intermediaries. It does not provide a mechanical first-name/last-name selection algorithm for O/B or C/O text. [NAIC MPCI business specification, Appendix C](https://naic.icp.gov.ae/portal/assets/mpci-guidelines/UAE%20MPCI%20Business%20Specification%20Document%20V1.0.pdf).

**Implication, not an explicit quoted rule:** faithfully extracting a printed B/L party block and verifying the ultimate party for a regulatory filing are distinct operations. Some documents do not supply enough information to complete the second operation from OCR alone.

## 3. Concrete problem shapes

### A. One address printed before the represented company

Source `e5a6a403`: [OCR](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_e5a6a40300b6e50a58dd6db5e6c1c9da7f58d71b15ab3dc683561b711eb99325/ocr.txt) · [PDF](/mnt/d/Projects/DocumentParsing/data/corpora/paddle-bl-swb-filtered-max5/files/e93d0f0bd222cce2599b2370cdb607b3a3f3c268e0450831f2fadd9cbf3f835f.pdf).

```text
LKW-HANDEL M.NOCK
KOLUMBUSSTRASSE 49
53881 EUSKIRCHE
GERMANY
O/B LKW-HANDEL
GUNTHER STIETZEL GMBH & CO. KG
```

Current target:

```json
{
  "name": "LKW-HANDEL M.NOCK O/B LKW-HANDEL GUNTHER STIETZEL GMBH & CO. KG",
  "addressLine": "KOLUMBUSSTRASSE 49 53881 EUSKIRCHE GERMANY",
  "country": "GERMANY"
}
```

The printed arrangement associates the German postal block with the first-listed identity. It does not independently provide a postal address for the represented second identity. The represented company is also separately printed as notify party.

Under a **printed-party extraction** policy, the first name and its German address form a directly supported pair. Under an **ultimate-party filing** policy, selecting the represented company would require separately establishing that company's relevant address; the first company's address cannot simply be borrowed.

This distinction is the target issue. The anonymization issue is separate: the contract expression is `LKW-HANDEL M.NOCK O/B {v006}`, so the first name is a literal while the second is mutable. All six current descendants retain the original first company. Example `syn_full_v7_c78eac155066769a2f2f4875` renders:

```text
LKW-HANDEL M.NOCK
AKUNNERSIVIK 14
P.O. BOX 312
NUUK
GREENLAND
O/B MESURADO
IMPORT GROUP
```

Its target combines the fixed first name and generated second name, with the first postal block. Fixing the literal alone would not resolve the entity-selection policy.

### B. Two explicitly different names AND addresses in one role box

Source `e827ecdd`: [OCR](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_e827ecdd1cb1a45de52691113a4fc9d9a5e54ca9c4e60f16e35dbfca8a599671/ocr.txt) · [PDF](/mnt/d/Projects/DocumentParsing/data/corpora/paddle-bl-swb-filtered-max5/files/d5fe87ee837309d340f51185512364a71543414502c1915f5ba5ff1ee51828e1.pdf).

```text
NEVDUBSTROY LLC
KYRGYZ REPUBLIC, BISHKEK, SVERDLOVSK REGION, SHABDAN
BAATYR STR. , 55 K, 720001
VAT 0143202410223
FOR AND ON BEHALF OF TIMBER TREASURE TRADING L.L.C.
AL SAFIYA-BLDG/HOR AL ANZ/PLOT N. 1974-0 UNITED ARAB EMIRATES
FOREIGN IMPORTER TAX ID:1238883
```

Current `name` combines both companies; `addressLine` and `country` retain only the first company's Kyrgyz postal block. The PDF confirms two distinct postal blocks.

Ten descendants retain the compound structure. The contract generates both names but leaves the secondary UAE address fixed; inspected descendants additionally label that address as an independent correspondence office. That existing rebase changes the ownership cues and needs to be considered in a later family decision.

This is not equivalent to a genuinely shared address. A one-party target must choose a coherent entity/address pair or flag the selection as unresolved.

### C. Similar two-address structures with a different target treatment

These already use only the first company and first address:

- BSH Turkey on behalf of BSH Germany, `ef438af5`: [OCR](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_ef438af5a29d8d83ea221d4187ecba7d3fbf45f257be49ed8dffa11b940c1552/ocr.txt) · [PDF](/mnt/d/Projects/DocumentParsing/data/corpora/paddle-bl-swb-filtered-max5/files/1c5b7b52e513b9a4b5c23a3fbf22565bd154a5bca949e02865ef94d5adca5e01.pdf).
- SOGELAM France on behalf of ATN Turkey, `baf80643`: [OCR](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_baf806435410f5474705794f489b31e103a654b9ca3a5853524a1feb46955dd4/ocr.txt) · [PDF](/mnt/d/Projects/DocumentParsing/data/corpora/paddle-bl-swb-filtered-max5/files/f6ee758c70491a1b5035aa3839028356931b9bc314631806d41f657ec99d16c3.pdf).
- Philip Morris CR at Kutna Hora on behalf of Philip Morris Products in Lausanne, `0ba3ad4f`: [OCR](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_0ba3ad4fcd4e713df5520786bd64916559b17258db1ba013ef90ec0d38e45a9f/ocr.txt) · [PDF](/mnt/d/Projects/DocumentParsing/data/corpora/paddle-bl-swb-filtered-max5/files/f5adc1b9df9dda006cb49e17c3f27798c51132237b5e79b9dd608b7e20d886e4.pdf).

The contrast with case B is a confirmed cross-document policy inconsistency. It does not prove that selecting the first company is the correct ultimate-party filing decision.

The Philip Morris family has six descendants; their targets select the generated first party while a fixed represented-company/address continuation remains in the text.

### D. A linked shipper continuation on another page

- `c1b20a71`: [OCR](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_c1b20a71d7a0a83c4976dc87b33e0ac00505cb695f39f170a76d368f5b1549cc/ocr.txt) · [PDF](/mnt/d/Projects/DocumentParsing/data/corpora/paddle-bl-swb-filtered-max5/files/b5b153865ea8e7d2a1335fe244f91f53ec05f3e579012e81a9c2bf037d787497.pdf).
- `19580ef7`: [OCR](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_19580ef7f4598b71d42dd7cd98cf3e1eceb456bce7520d3bc50112a8c92ddc82/ocr.txt) · [PDF](/mnt/d/Projects/DocumentParsing/data/corpora/paddle-bl-swb-filtered-max5/files/7a2e6d0e406681d3faa73cbc8fd77b1d83645693732b3d97cfe593741df18e40.pdf).

Both print MARFRIG in the first-page shipper block and a linked continuation naming WESTON IMPORTERS, followed by a care-of Brodies postal address in the UK.

The first current target combines MARFRIG and WESTON but retains MARFRIG's Brazil address. The second retains only MARFRIG and its Brazil address. Each compiled family has ten current descendants.

This is another concrete inconsistent treatment, not a generic need to search unrelated goods text for party details. The explicit shipper-continuation caption establishes why the later block is relevant.

### E. A legitimate postal care-of qualifier

Source `36d2930f`: [OCR](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_36d2930f49cb3d8d79acd2968fea98a5b6e279ce3b37c7752660333cab609b47/ocr.txt) · [PDF](/mnt/d/Projects/DocumentParsing/data/corpora/paddle-bl-swb-filtered-max5/files/ce26199723e62bc63f6528610ce0efdd212365f7f0f635b2d5ead5438986b5d1.pdf).

```text
BRANCH OF PRIDE GLOBAL II LTD
AMERIYA GENERAL FREE ZONE AREA
PIECE 705 ALEXANDRIA
C/O BP EXPLORATION (DELTA) LIMITED
```

Current target already keeps the first entity as `name` and C/O BP in `addressLine`. This is a useful counterexample to “all company wording must be removed from the address.” The care-of routing can be address information rather than a second target identity.

Ten descendants belong to this family; five retain explicit care-of wording in current party targets.

### F. Three actors with nested relationships

Source `551657f3`: [OCR](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_551657f34bc93b4b7db213ad8cf35ac0be759fa35d32967097f1b4a766ff6525/ocr.txt) · [PDF](/mnt/d/Projects/DocumentParsing/data/corpora/paddle-bl-swb-filtered-max5/files/5b248dd402b19721d56966463590e209b1f064b1e5ade4c412cc5a1c3c4d00b4.pdf).

```text
CHARTER LINK LOGISTICS(HK)LTD O/B
SERIAL MICROELECTRONICS PTE LTD.
C/O WYNTRON,INC.
LOT 9 ... ROSARIO, CAVITE PHILIPPINES
```

The target flattens all three names into `name`. The wording expresses an on-behalf-of relationship and a care-of relationship, not three aliases of one organization. Ten descendants belong to this family. A two-way string split is not enough to decide postal ownership.

### G. Agent and account relationships

- KAIZEN as agents of EXPRESS AUTO PARTS EXPORT, `97e9b467`: [OCR](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_97e9b467b37f213e5f9566b74b2bb392fa0239b30a0681bf157d763740802eae/ocr.txt) · [PDF](/mnt/d/Projects/DocumentParsing/data/corpora/paddle-bl-swb-filtered-max5/files/a190006fc111151dd5e02169787a49f33316b0efdb047bde4d6e9824f9de2d79.pdf). Nine descendants; eight retain the explicit agent phrase.
- VARDHMAN A/C SONVIGO, `f256749b`: [OCR](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_f256749bd9718a0bae20e7cdd8fe6942a7b1e3759f3089b86a23d7524f5638f0/ocr.txt) · [PDF](/mnt/d/Projects/DocumentParsing/data/corpora/paddle-bl-swb-filtered-max5/files/2636c9c6caae5a3f9d9f57ca5937e7b191d136533b9b5cb18301d95e6e729a2e.pdf). Six descendants.

Both currently flatten two names into one. Their relationship meanings differ and must not inherit the same address/role-selection rule without review.

### H. Source structure dropped or newly introduced during synthesis

TURA on behalf of DAELIM, `5718f96f`: [OCR](/mnt/d/projects/documentparsing/data/curated/mpci-bl-real-v7-reviewed-r14-equipment-660/samples/doc_5718f96fce89f4f7ec8616548702042eba867fd4bb61f7b90a301e5e398f6a95/ocr.txt) · [PDF](/mnt/d/Projects/DocumentParsing/data/corpora/paddle-bl-swb-filtered-max5/files/ed48841f19ba223e65a330f06da64da48c8cae9b4660ba0e212bb7e161752ed9.pdf). Its real target is compound, but all six descendants use one generated identity and no on-behalf-of structure. Render and label may agree while the source relationship scenario has disappeared.

Conversely, four families introduced care-of wording not present in the source/current real label: `a7ede7d8`, `c92e5900`, `5c2fdc31`, `7046a05d`—five documents total. Some could be legitimate postal care-of qualifiers; one names two generated organizations and a person. These are review findings, not automatic defects.

## 4. Why previous agreement checks did not settle this

There are three separate questions:

1. **Grounding:** are the target words present in the rendered OCR?
2. **Ownership:** do this name, address, country and contacts belong to the same chosen entity?
3. **Business-role selection:** is that chosen entity the party intended by our extraction contract or by the final filing?

A synthetic compiler can satisfy question 1 perfectly while preserving a policy mistake in questions 2 or 3. The current name description explicitly encourages some compound names. A reviewer checking source presence alone therefore can accept them without proving a coherent entity/address pair.

The source/template layer also matters: fixed secondary addresses, fixed primary names and dropped/introduced relationship structures cannot be diagnosed from field-level exact matching alone.

## 5. Recommended decision framework — not applied

### Option 1: Printed B/L party extraction, with separate filing validation

For the training target, represent the primary explicitly declared party for the document role, paired with its own printed postal block. Preserve secondary relationship information in audit metadata rather than concatenate organizations into a singular name.

This aligns most directly with the OCR-to-structured-extraction objective. It does not certify that an intermediary is acceptable as the ultimate party in a later filing.

“Primary” must be established from the role block and ownership cues, not a universal “pick the first string” regex. Explicitly captioned alternatives, agents and nested relationships require a family/source decision.

### Option 2: Ultimate-party filing target

Require the ultimate shipper/consignee and its corresponding address. If the source establishes that entity but omits its address, leave the missing field absent or flag it; do not substitute an intermediary's address.

This is closer to strict filing semantics but some sources need information beyond OCR. No stronger model can extract an address that is not supplied.

### Option 3: Composite printed identity as a display string

Retain current A O/B B strings as faithful display text. This avoids discarding relationship wording but leaves a singular-entity field semantically overloaded and does not solve regulatory/address ownership.

**Recommendation for discussion:** option 1 for extraction training, while keeping ultimate-party compliance as a separate, explicit step. Do not adopt that recommendation silently: choosing the target's meaning is the key decision here.

### Invariants for any eventual repair

- Name and postal address must be a coherent pair for the chosen entity.
- Country and contacts must follow that same ownership decision.
- Do not concatenate two separate postal addresses.
- Do not attach A's address to B because B follows O/B.
- A real care-of routing qualifier may remain in the postal address.
- Do not invent an agent/notify role for a secondary company.
- PDF may establish layout/ownership, but may not introduce values absent from OCR.
- Retain source relationships in audit metadata if the extraction target does not represent them.

## 6. Efficient follow-up once the target meaning is agreed

1. Resolve the source relationship/ownership shape once per family/source; the present inventory bounds the explicit-marker work at 52 real sources and 32 compiled families.
2. Record the chosen entity and its own printed postal fragments, plus the distinct secondary actors. These are case decisions, not new inference calls for every descendant.
3. Align the relevant source target and compiler contract to that decision. Treat primary-name literals, fixed secondary addresses and relation-shape changes as separate template issues.
4. Apply the family decision to all 268 current descendants, including marker-free descendants. Repair only the fields actually affected.
5. Validate both render/target agreement and entity/address ownership. Check repeated parties and linked continuations rather than relying on marker presence alone.
6. Test future generation against these relationship shapes with explicit expected entity/address pairs, not merely a generic model “pass.”

No full-dataset relabelling, paid per-document repair or blanket removal of O/B/C/O is justified by this audit. Conversely, the other unmarked records are not certified clean merely because they did not match a relationship marker.

## 7. Preservation and evidence

The six current dataset/registry/schema hashes below were recorded before documentation and checked again after documentation: **all six are unchanged**. This check is scoped: it proves these files were not changed, not that the entire existing dirty worktree was clean.

An independent read-only join of the current JSONL files confirmed 600 real training, 2,500 synthetic training and 60 real validation records; all 52 inventoried real IDs and 181 listed synthetic IDs exist. The 32-family cohort contains exactly 268 descendants. All local links in both new reports resolved, Markdown code fences were balanced, and no trailing whitespace was found. The verification scan took 1.563 seconds. Repository `git diff --check` also passed. No runtime code was changed, so no training or synthesis performance benchmark was applicable.

- `data/curated/mpci-bl-real600-synthetic2500-v7-positions-boundary-repaired-v1/train.jsonl`: `379b8a5fe39b8ef98391ad690b944fa2b236b4dd7bf710c6ba5b39d962508ac9`
- `data/curated/mpci-bl-real600-synthetic2500-v7-positions-boundary-repaired-v1/validation.jsonl`: `9f3b4c36fe5d6310b3fb79a6d380a06d24d2fe8583acf80748d83861b2fe94da`
- `data/curated/mpci-bl-real600-synthetic2500-v7-positions-boundary-repaired-v1/synthetic.jsonl`: `79bb23dfab11b5428ce5b23b792ad82581f670b9d6261a50f9e18c991391a78a`
- `artifacts/synthesis-templates/mpci-bl-v7-reviewed/manifest.json`: `a86549ed252d53850f4e7c7a88821b61b9698a09e145a9d4ab05d7db42677870`
- `artifacts/synthesis-templates/mpci-bl-v7-reviewed/ownership.yaml`: `76e207b4337c354aaddb91fa6117beb05202b3a9e059afe7c6bfeebe5cca03f4`
- `src/document_ocr/label_schemas/bill_of_lading_v7.py`: `45fc89da02fb5ced77d35d9dae4adccb0b30e6d972cd8ea6284353b0ef75dc9f`

The inventory links every real source OCR/PDF, records the current target projection, lists all 32 compiled families, and gives the exact 181 marker-bearing synthetic IDs. No repair approval or semantic certification is implied by an inventory entry.
