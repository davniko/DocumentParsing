# Reviewer-flow improvements and paired reruns — 2026-10-03

## Result

Completed the requested investigation, changes, and rerun on the **same 20 original
silver candidates**, using **Luna / high**, full plain-text OCR, five scoped
reviewers, and eight globally concurrent requests. There were two bounded paired
experiments, followed by an explicit one-document replay for a discovered native
schema gap. No production labels, OCR, datasets, training settings or prior results
were changed. No Codex subagents were used.

The final version detects more known defects and avoids the six specific harmful
corrections from the first review experiment. It is **a better supervised review
baseline, not a reliable automatic gold publisher**. This is the completed result
of this experiment; the remaining semantic failures are identified below rather
than hidden behind the agents' pass votes.

| Source-reviewed measure | Previous review | Intermediate rerun | Final rerun |
|---|---:|---:|---:|
| Known semantic defects detected | 5/9 | 8/9 | 8/9 |
| Known defects fully repaired | 4/9 | 6/9 | 5/9 |
| Known defects appropriately held | 1/9 | 2/9 | 2/9 |
| Known defects partially repaired | 0/9 | 0/9 | 1/9 |
| Known defects missed | 4/9 | 1/9 | 1/9 |
| Previously correct controls preserved | 8/8 | 8/8 | 8/8 |
| Original six harmful corrections repeated | 6 | 0 | 0 |

The intermediate version repaired the full port name but deleted its supported
loading role. The final version retained/recovered the roles but still shortened
the names. Thus its lower exact-repair count is real, not concealed by changing the
test. The nine checks are a fixed diagnostic panel, **not exhaustive gold or
field-level F1**. This is a development panel used to clarify instructions, not an
untouched generalization test. One run per version cannot isolate stochastic model
variation from each individual instruction change.

Latest outcomes, substituting the explicit replay for the failed document:

- 20 completed review cycles: 11 agent-all-pass candidates, 9 requiring adjudication.
- 91 passing section votes, 5 corrections-needed, 4 unresolved.
- 18 application-valid targets. The two malformed source-identifier drafts remain
  blocked by Python validation; neither is silently made valid.
- These counts describe workflow states, not semantic certification. One all-pass
  candidate still loses a product code between sections.

## Diagnosed mechanisms and implemented changes

### 1. Scope-specific review responsibilities

The previous generic review instruction did not sufficiently distinguish the jobs.
`SECTION_PRIORITIES` in
[direct_models.py](../src/document_ocr/labeling_agents/direct_models.py) now gives
each reviewer a compact checklist:

- Parties: identity/postal boundaries, all owned contacts, remote continuations,
  alternative addresses, and company-homepage ownership.
- Route: role versus reading order, full proper names, location-owned countries,
  receipt/loading/delivery distinctions, and layout requests for detached headings.
- Metadata: carrier B/L versus platform/file references, selected freight terms,
  reference ownership, and duplicate mentions.
- Equipment: distinct containers and seals, joined-token boundaries, source
  ambiguity, supported size/type and temperature ownership.
- Cargo: goods groups versus container portions, allocations, complete totals,
  unit scope, marks versus continued party text, DG and handling.

The field descriptions remain the semantic authority. No sample IDs, expected
answers, known-defect list, historical labels or source-specific exceptions were
sent to the agents.

### 2. Clarified field ownership and boundaries

[bill_of_lading_v7.py](../src/document_ocr/label_schemas/bill_of_lading_v7.py) now
distinguishes:

- Owned homepages in carrier terms from links to individual legal/help articles.
- Carrier B/L identifiers from platform receipt/file identifiers.
- Postal zones/building designations from company names, even on the same line.
- Complete place names from shortened/geocoded labels.
- Locally scoped measure units from units borrowed from a different measure.
- Delivery-agent offices from delivery places for the goods.
- A named carrier from an agent's signature on its behalf.
- Geographic origin codes from adjacent commercial identifiers.
- Goods-specific instructions from generic load/stow/count/seal declarations.
- Forwarding/export references from HS/DG codes, voyage/container/seal identifiers,
  shipping marks and product lot/model identifiers with another target owner.

Target field names, types, ordering and optionality remain unchanged. These
description changes also affect future direct extraction, not just review.

### 3. Correction and re-review retain the relevant context

[direct.py](../src/document_ocr/labeling_agents/direct.py) now supplies Python
validation diagnostics for the assigned candidate section, without suggesting an
invented replacement. Re-review receives exact before/after changes within its
scope, alongside the complete current section and OCR. Changed lists are shown
whole so regrouping is not mistaken for positional entity correspondence.

The reviewer prompt asks it to check the actual candidate before calling something
missing, and to evaluate both additions and removals. The corrector is explicitly
allowed to reject an unsupported review suggestion. No extra agent stage, scalar
evidence inventory, hidden retry or correct-until-green loop was introduced.

The previous PDF-continuity fix was live-tested here: for document `10-b25a3616`,
route requests carried **0 → 1 → 1 → 1 images** across initial review, assisted
review, correction and re-review. The final reviewer still disputed supported
roles; the saved request proves this is now a **semantic decision failure**, not
missing image context.

### 4. Native schema gap found and repaired

In the final full batch, document `07-24370697`'s equipment corrector emitted
`typeCategory=REFRIGERATED` with no size. Python requires a complete size/type pair,
or printed `typeDescription` when the pair is unsupported. The original provider
schema did not express that Python-only constraint, so the response was native
schema-valid but application-invalid. The request failed explicitly, with its
response and charge retained.

The native annotation schema now expresses the existing contract as two nested
`anyOf` alternatives: complete canonical pair, or printed wording/unknown type
with both categories absent. This changes neither the target shape nor the Python
policy. It prevents this structural failure without guessing a size or dropping
the container. Other Python semantic validators remain necessary.

The official Structured Outputs documentation supports nested `anyOf` and does
not support `if/then` composition; the implementation uses the supported route.
[OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

An exhaustive **240-combination probe** over size/type/printed wording produced
**97 accepted and 143 rejected by both native-schema validation and Python**, with
zero disagreements. The explicit live document replay completed: it retained
`typeDescription="refrigerated container"` without inventing a size. The original
failed attempt is not overwritten or omitted from spending.

## What actually improved

- `0600699f`: the missing third shipper telephone is restored.
- `0701d457`: two yarn specifications become one shared goods group, with two
  680-carton placements, 1,360 cartons, 51,800 kg and 110 m³; no arbitrary
  product-to-container assignment. The carrier remains the principal, not its agent.
- `439f6102`: `NASR CITY F.Z. BL. J` moves from consignee/notify names into the
  complete address line, ahead of `11816 CAIRO EG`.
- `4d49ad47`: both the carrier seal and customs seal are retained.
- `b25a3616`: partial four-container mass/volume sums are no longer treated as
  totals for a five-container shipment.
- Both known Suez/Alexandria alternative-address cases are now held instead of one
  passing as a single postal address.
- The three previously removed carrier homepages survive. The eBL file reference
  does not become the B/L number. A shipper country is not borrowed into receipt,
  and a unitless net weight does not acquire kilograms.
- HS/product information is no longer indiscriminately *added* to export references.

## Remaining defects and decisions, with concrete scope

### A. Cross-section transfers are not yet coordinated

Document `13-ab905fd4` prints `ITEM CODE.: 300398-120180`. Metadata correctly removes
it from export references, but cargo does not add it to the product description.
The final candidate therefore loses the code, and all five reviewers pass.

This is a **confirmed new completeness defect**. Equipment/cargo and
parties/metadata are currently reviewed as dependency groups, but a product-fact
transfer from metadata to cargo is not represented as an explicit handoff. A next
change should coordinate such transfers and recheck both source and destination;
it should not simply delete a value from the wrong field or restore it there.

### B. Cross-page party continuation remains missed

The same document ends its discharge-agent address on page 2 with `CAIRO`; page 3
continues it with `EGYPT` beneath repeated cargo headers. OCR contains that word.
The source PDF confirms the continuation. The final party label still omits it.
The intermediate rerun incorrectly used it as a shipping mark; the final rerun
avoids that new error but still misses the address completion.

### C. Route roles and full names remain inconsistent

For `b25a3616`, the final target correctly recovers receipt/delivery roles from
layout but emits `BUSAN SEAPORT` and `SOKHNA SEAPORT`, dropping integral printed
`SOUTH KOREAN`/`EGYPTIAN` wording. Loading/discharge names are also only partially
restored. Re-review receives the PDF yet wrongly asks to delete loading/delivery
because the OCR lacks their headings. OCR-grounded values with PDF-established
ownership are allowed; importing PDF-only values is not.

### D. Unsettled ownership or normalization is kept distinct from proven errors

The reviewed candidate set includes these decisions requiring adjudication:

- `0600699f`: consignee/invoice text treated as shipping marks in a shared box.
- `99adb051`: OCR `Collect` versus the PDF's caption and selected freight term.
- `4d49ad47`: a deemed issuance-office clause used to populate carrier identity.
- `439f6102`: a transit-warehouse instruction used as carrier delivery place.
- `ab905fd4`: a hazard-emergency number assigned to the shipper without a clear
  owner caption.
- `24370697` replay: `India` as goods origin from printed `INDIAN`; the meaning is
  supported but the source-wording normalization needs a consistent ruling.

None of these is silently counted as a correct repair. There are also false
re-review flags: restoring a unitless volume as cubic metres; borrowing a country
from another same-named locality; and rejecting layout-supported route roles.
The malformed identifier in `298806fe` still gets agent passes despite diagnostics,
but Python continues to block the invalid final document.

## Validation, runtime and cost

| Experiment | API requests | Wall time | Usage-derived cost |
|---|---:|---:|---:|
| Previous review baseline | 172 | 237.14 s | $0.1911885 |
| Intermediate rerun | 181 | 298.64 s | $0.22230013 |
| Final full rerun, including rejected correction | 175 | 277.70 s | $0.19429384 |
| Explicit failed-document replay | 14 | 62.39 s | $0.01490143 |
| **This task, all attempts** | **370** | separate runs above | **$0.4314954** |

The latest complete 20-document cycle, including its failed attempt and explicit
replay, costs **$0.20919527**, approximately **1.05 cents/document**. These are
usage-derived estimates, not invoice reconciliation: USD/million ordinary input
0.10, cache read 0.01, cache write 0.125, output 0.50. Cache subsets are deducted
from input; reasoning is included in output, not charged twice. No request has
unknown billing. The failed application validation has a recorded provider response.

There is **no measured end-to-end speedup**: the final full batch took 40.57 seconds
longer than the old baseline, with three more calls and variable provider latency.
The local change is negligible: 100 context constructions measured 145.86 ms before
and 155.06 ms after (about 0.092 ms extra per context). Peak traced allocation was
604,060 versus 570,033 bytes; this is a local allocation probe, not GPU or total
process memory. Context characters rose about 8.9%, including the native pair
contract. Do not interpret the single-run allocation decrease as a proven memory
optimization.

- **521 tests pass** after the final code change; Ruff and targeted mypy pass.
- All **370 raw provider responses** conform to their actual native wire schemas.
  This includes the explicitly rejected Python-only pair violation before its fix.
- Exact complete OCR, section scope, model/reasoning settings and native strict mode
  were checked on captured request bodies. API credentials were not captured.
- Source OCR, PDFs, original candidates and historical responses retain their hashes.
- Every exported valid target passes the full Python schema. Invalid identifier
  drafts remain separate, with no `target.json` falsely representing validity.
- All 41 intermediate and 40 final-batch initial findings were independently
  adjudicated, plus all changed scalar paths and the replay changes. Relevant source
  PDF pages were visually inspected. This is not exhaustive human gold for all scalars.

The experiment follows the task-specific, human-calibrated evaluation principle in
[OpenAI's evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices):
measure concrete misses, harmful corrections and correct controls rather than
treating agreement or structural validity as accuracy.

## Artifacts and reproduction

- [Intermediate results and findings](../artifacts/kie-labeling/direct-review-20261003-luna-high20-r2/adjudication.json).
- [Final comparison and adjudication](../artifacts/kie-labeling/direct-review-20261003-luna-high20-r3/adjudication.json).
- [Final per-document OCR/before/after/review comparisons](../artifacts/kie-labeling/direct-review-20261003-luna-high20-r3/comparisons/).
- [Explicit replay and its receipts](../artifacts/kie-labeling/direct-review-20261003-luna-high20-r3/schema-replay/).
- [Native-schema falsification probe](../artifacts/kie-labeling/direct-review-20261003-luna-high20-r3/schema_probe.py).

Each experiment has its own immutable input manifest, implementation snapshot,
request/response receipts, actual wire schemas, usage, timing and analysis scripts.
The runners refuse overwrites and implicit retries. Offline analysis uses the
temporary `jsonschema` environment via
`PYTHONPATH=/tmp/documentparsing-jsonschema-probe .venv/bin/python <analysis-script>`.

The useful next intervention is **cross-section ownership/transfer verification
and a small source-adjudicated layout challenge set**, not bulk publication and not
more undirected instructions. The current changes are retained because their
runtime behavior is tested and their benefits and remaining failures are measured.
