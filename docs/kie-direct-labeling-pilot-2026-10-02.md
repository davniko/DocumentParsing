# Luna high: 20-document direct-extraction pilot

Run and inspected on 2026-10-02. This is the requested **first extraction step**,
not a repair campaign. No existing dataset, source OCR, template or training run
was changed. No paid review/correction calls were launched.

## Result and interpretation

All **20 documents received exactly one request** to `gpt-6-luna`, with **high
reasoning**. All 20 returned completed JSON responses. **17 passed the application
schema and package-category checks**, producing `target.json`; three were rejected
and their full responses/usage were preserved. No silent retries or replacements.

This experiment demonstrates a cheap, useful first-pass extractor. In particular,
it makes several correct address and cargo decisions that prior annotations got
wrong. It also exposes concrete remaining extraction mistakes and two avoidable
empty-object integration failures. It does **not** establish that 17 documents
are gold, or that the downstream training model will reach a particular F1.

The next useful experiment is bounded section review on these saved results, not
another complete extraction or a bulk relabeling launch. That next experiment has
not been run.

## Configuration, inputs and reproducibility

- Model: `gpt-6-luna`; reasoning: `high`; maximum output including reasoning:
  16,384 tokens; timeout: 180 seconds; maximum five concurrent document calls.
- One request per document, using the maintained direct extraction flow,
  PydanticAI native structured output, described V7 models and pinned package
  categories. Both provider and validation retries were disabled.
- The model received complete literal OCR text, not JSON-wrapped line records,
  old labels, repair decisions, PDFs, evidence-span requirements or a rationale
  schema. The prompt/schema were unchanged during the run.
- Panel: ten specified challenge cases plus five seeded train and five seeded
  validation selections (`20261002`). The resulting split is nine train and
  eleven validation documents. This is a diagnostic panel, not a representative
  random accuracy sample.
- Existing input/label datasets and the extraction implementation were hash-pinned;
  the final analysis confirmed they were unchanged. Every run's OCR matches its
  selected source hash.
- Prior labels: historical V3, trained V6 and current working address-repair
  labels are available for all 20; the older reviewed V7 snapshot covers four.
  All comparison records use identical OCR to their new extraction. That V7
  snapshot's cargo approvals for `0701d457` and `2aa1d48b` had already been
  withdrawn; it is not an unquestionable gold reference.

Artifacts: [pilot directory](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/),
[selection](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/selection.json),
[analysis and per-document costs/diffs](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/analysis.json),
[OCR/new/previous comparison files](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/).
Each `runs/` directory preserves config, OCR, schema, prompts and provider receipts.

Selection bookkeeping correction: document eight, `096bccea`, was initially
misdescribed as the DUMECTIN DG case; it is a frozen-beef case. An attempted
pre-admission replacement failed at immutable manifest publication. The original
20-document manifest was run, without changing any selected document. The unused
`inputs/08-32ebac37` and `selection-amendment.json` are proposal artifacts, not
another extraction or a billed call. The analysis corrects the challenge
description, and the selection script now names the actual challenge correctly.
The seeded panel nevertheless contains one real DG document, `ab905fd4`.

## Cost and time

| Measurement | All 20 attempts, including failures |
|---|---:|
| Input tokens | 229,874 |
| Cache-read tokens, included in input | 185,478 |
| Cache-write tokens, included in input | 44,336 |
| Other input tokens | 60 |
| Output tokens, including reasoning | 103,725 |
| Reasoning tokens, included in output | 76,638 |
| Usage-derived cost | **$0.05926528** |
| Mean cost per attempted document | **$0.002963264** |
| Cost attributable to the three rejected attempts | $0.008955135 |
| Median request time | 48.71 seconds |
| Request time range | 33.54–113.51 seconds |

The initial single-document live check took 40.13 seconds including setup; the
remaining 19-document batch took 275.46 seconds. Combined active batch time was
approximately 5 minutes 16 seconds, excluding investigation/manual review time.

The provider returned token usage but **no monetary charge**. These dollar
amounts are estimates calculated from the official standard short-context rates:
$0.10/M ordinary input, $0.01/M cache reads, $0.125/M cache writes, $0.50/M output.
Cached subsets are subtracted from ordinary input; reasoning is not charged a
second time. [Official model pricing](https://developers.openai.com/api/docs/models/gpt-6-luna).

At this panel's token lengths, first extraction of 1,157 documents projects to
**$3.43** with the observed caching, or **$4.66** if every input token incurred
the cache-write rate. These are first-pass projections only: not quotes for a
gold dataset, review, correction, PDFs, retries or human adjudication. High
reasoning consumed about 74% of output tokens. The panel does not compare reasoning
levels, so it cannot establish that high is better than medium.

## Improvements against previous output

### Full addresses and ownership

`0701d457`'s working address had:

```text
FUNCTIONAL INDUSTRIAL SOUTH ZONE, DONGLIN TOWN, WUXING DISTRICT,
HUZHOU, ZHEJIANG PROVINCE
```

Luna correctly included **CHINA** from the shipper's page-two continuation, and
kept its telephone separate. It also attached the consignee/notify continuation
emails to the appropriate roles. No per-word evidence output was required.

`14f66f23` retained **MERTER / GUNGOREN / ISTANBUL** after the intervening tax and
phone text, while excluding those nonpostal values from the address. This agrees
with the repaired V7 address and improves on the trained V6 stripped target.

`439f6102` moved **NASR CITY F.Z. BL. J** from the previous company-name label into
the address for consignee and notify. `096bccea` similarly put **EL - OBOUR** into
the address rather than the company name. These are semantic boundary improvements,
not merely higher string agreement with old labels.

`c9c96562` preserved **CONAIR ITALY S.R.L ON BEHALF OF RASMI KACHLAN** as the
shared-context identity and kept its single printed postal address. It did not
mistake the separately printed exporter-country metadata for that postal country.

### Goods identities and placements

| Document | Earlier output | Luna high first pass |
|---|---|---|
| `0701d457` yarn | V6 put specifications in AAI; the withdrawn V7 snapshot split the same product bundle into two goods | One complete description; 1,360 cartons; two placements of 680; exact gross/volume sums |
| `2aa1d48b` paper | Six duplicate PAPER REELS goods, one per container | One PAPER REELS goods; 71 reels; six placements 12/10/13/10/13/13; exact gross sum 169,111 kg |
| `096bccea` beef | One merged goods with five separately counted cuts | Five goods with respective carton counts 1,075/45/44/47/38 and container placements; total 1,249 |
| `c0f529a4` lubricants | One long description plus 16 package facts, with unclear product ownership | 16 independently quantified product entries; no invented mapping of these products to individual containers |
| `a202c173` glaze | Two identical GLAZE goods | One goods, 40 pallets, two placements of 20 |
| `66d5d7f6` oils, rejected document | Four container-centric goods with mixed products | Raw answer correctly grouped three products with counts 46/30/20 and corresponding placements; document rejected for an unrelated empty contact object |

The beef split legitimately cannot copy whole-shipment gross weight/volume into
every product. Omitting unsupported per-product values is not an extraction
failure. Likewise, shared-container net weights cannot be silently divided among
different oil products. The schema's lack of a separate consignment-total field is
a representational trade-off, distinct from a missed goods-level fact.

`ab905fd4` correctly extracted UN 1993, class 3, packing group III and **50°C flash
point**, as well as 28 inner drums rather than seven outer pallets. `e47eaf30`
retained complete ENOS-prefixed seals. These are useful positive controls against
previous issue families, not complete DG or equipment coverage.

## Failures and remaining defects

### Three application-validation failures

| Document | Cause | What the offline diagnosis established |
|---|---|---|
| `99adb051` | `freight` object with both children null | Changing only that empty object to null in memory makes the full answer structurally valid, with every non-null scalar unchanged |
| `298806fe` | Printed `INLLU4102226` does not match the required ISO identifier shape | The model copied the malformed OCR rather than inventing an ID. This is a source/target-contract conflict; do not manufacture a compliant ID |
| `66d5d7f6` | Shipper `contactDetails` object with all four children null | Changing only that empty object to null in memory makes the full answer structurally valid, with every non-null scalar unchanged |

These are not refusals, timeouts, truncated responses or JSON parsing failures.
The two all-null probes are saved in
[offline-integration-probes.json](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/offline-integration-probes.json).
Neither probe changes the stored response, publishes a target, or approves its
semantics. They establish that two failures need not require another paid call.
No normalization change has been shipped to the maintained flow in this pilot.

### Concrete semantic findings

- **Caption contamination and omission:** `0600699f` includes `POSTAL CODE:` in
  the consignee address, contrary to the field description; it also omits the
  third printed shipper telephone `(86-512)62589948`. Its party tax ID is not in
  the references either. The repaired working address already removed that caption.
- **Alternative addresses concatenated:** `c0f529a4` and the raw answer for
  `66d5d7f6` join the Suez postal context with `568 El Horreya Avenue ... Alexandria`
  and repeated locality/country text. That does not satisfy the no-alternative-
  address-concatenation policy. Selecting the right address requires adjudication;
  copying all words is not by itself a correct party target.
- **Cross-page postal continuation:** `ab905fd4` stops the delivery-agent address
  at CAIRO and omits the EGYPT continuation after the repeated page-three header.
  This is the same class of continuation the model handled correctly for CHINA
  in `0701d457`; it is not consistently solved by one call.
- **Identity boilerplate:** `096bccea` and `6fe428c6` include `au Capital de
  234 988 330 Euros` in the carrier name. Paid-in capital is not party identity.
- **Wrong field ownership:** `c9c96562` emits `TOTAL` as a shipping mark. A table
  total label is not a cargo mark. Its noncontiguous party blocks otherwise look
  substantially better than the old stripped-address targets.
- **Missed membership:** `b25a3616` lists five containers and one shared goods
  entry but emits no goods placements. Membership and known package counts must
  be judged separately: the OCR only shows four repeated 20-box rows against a
  five-container/100-box total, so blindly distributing totals would be wrong.
- **Voyage/leg boundary:** `e47eaf30` emits `02509/S` as the voyage despite the
  heading `vessel/voyage/leg` and value `RDO FAVOUR/02509/S`; the previous target
  separated the voyage as `02509`.
- **Reference completeness:** for example `a202c173` omits the explicitly printed
  `TAX ID: 100483143`, while retaining its ACID and exporter ID. Old-label agreement
  would miss this because the previous target was incomplete too.

### Decisions needing focused review rather than blind correction

- `24370697`: Dadri/Delhi values follow several flattened table headings. Luna
  uses DELHI as receipt place and omits payment place; previous reviewed labels
  use DADRI, UP for receipt and DELHI for payment. This is a good candidate for
  the planned PDF-layout-assisted reviewer, with values still restricted to OCR.
- `14f66f23`: the previous B/L identifier `MLTRLS2418021` is present, but directly
  follows an incongruous Country of Origin heading. Luna leaves B/L absent.
  Resolve role from layout, not identifier-shaped-string guessing.
- `99adb051`: the raw response reproduces seal `7CM08875356`; earlier human
  adjudication treated the `7` in the joined token as a mistyped delimiter and
  approved `CM08875356`. The agent did not receive that adjudication. The source
  also prints both 65 and 150 packages; the answer retains both as package facts.
  This is not safely resolved by an ordinary presence check or by summing them.
- Equipment normalization is uneven: literal `DC 20`, `20ST`, `40RH`, `40H
  (HI-CUBE)` and `45HC` remain descriptions where old labels used categories.
  Distinguish incomplete source types from genuinely supported aliases, then
  check consistent mapping. Do not call all literal fallback values hallucinations.
- `b25a3616` recognizes EGYPTIAN as country wording but shortens the port to
  SOKHNA, losing the agreed proper-name treatment of SOKHNA EGYPTIAN SEAPORT.
  Its product wording is also reordered around `CLNA-TR8142EC AS PER`.
- `2aa1d48b` adds country to discharge/issue fields from other document contexts;
  review role-specific support, not just whether EGYPT/SWEDEN occurs somewhere.
- `439f6102`: the issuer/footer and legal consolidator-versus-carrier language
  warrant role review; generic negotiability wording should not be accepted
  without checking whether it applies to this instrument.
- `096bccea`'s `SB-168/24` marks assignment and `ab905fd4`'s `SR-1690` product-
  versus-marks ownership need the cargo reviewer. Both are source-present strings;
  presence alone does not settle their field.

## Per-document review index

The comparisons contain full OCR, raw new answer and all available old labels.
I inspected the selected outputs against source text and investigated their
differences. This is an initial semantic audit, not a blinded exhaustive gold
annotation or a statistical confidence bound. In particular, a row with no
identified problem is not proof of zero latent errors.

| # / document | Result | Main review observation | Cost, USD |
|---|---|---|---:|
| [01 0600699f](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/01-0600699f.md) | Draft | Postal caption, missed phone/reference; cargo facts otherwise recovered | .003470 |
| [02 14f66f23](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/02-14f66f23.md) | Draft | Good interrupted address; missing B/L needs role/layout review | .003401 |
| [03 1c402851](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/03-1c402851.md) | Draft | Correct 60 inner bags and membership without invented split counts; equipment/lot ownership review | .002605 |
| [04 0701d457](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/04-0701d457.md) | Draft | Correct shared goods and address continuation; carrier homepage completeness check | .002450 |
| [05 2aa1d48b](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/05-2aa1d48b.md) | Draft | Correct one-goods/six-placement aggregation; route-country scope review | .002235 |
| [06 99adb051](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/06-99adb051.md) | Rejected | Empty freight; prior seal adjudication and conflicting package counts | .003127 |
| [07 24370697](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/07-24370697.md) | Draft | Good parties/seals/temperature; receipt/payment table alignment | .003148 |
| [08 096bccea](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/08-096bccea.md) | Draft | Correct five quantified products; carrier-name contamination; marks/type review | .003603 |
| [09 c9c96562](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/09-c9c96562.md) | Draft | Good on-behalf-of identity; TOTAL incorrectly used as mark | .002152 |
| [10 b25a3616](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/10-b25a3616.md) | Draft | Named order consignee; missing placements; route proper-name/description edits | .003458 |
| [11 4d49ad47](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/11-4d49ad47.md) | Draft | Strong address, cargo, marks and full-seal extraction; no concrete defect identified in this inspection | .002254 |
| [12 439f6102](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/12-439f6102.md) | Draft | Better company/address boundary; carrier and negotiability review | .002860 |
| [13 ab905fd4](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/13-ab905fd4.md) | Draft | Correct DG/inner drums; missed cross-page agent country; cargo-text ownership review | .003469 |
| [14 c0f529a4](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/14-c0f529a4.md) | Draft | Correct 16 independently quantified products; concatenated alternative addresses | .003270 |
| [15 a202c173](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/15-a202c173.md) | Draft | Correct shared glaze/no guessed weight unit; missed tax reference and unheaded-role review | .003556 |
| [16 298806fe](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/16-298806fe.md) | Rejected | Malformed source ID; useful raw shared-goods/metric-tonne extraction | .002815 |
| [17 e47eaf30](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/17-e47eaf30.md) | Draft | Good full ENOS seals, totals and allocation; voyage includes separate leg | .003335 |
| [18 b0da3b37](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/18-b0da3b37.md) | Draft | Keeps tax/contact outside address, does not invent country or unprinted net-weight unit; reference completeness review | .002088 |
| [19 6fe428c6](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/19-6fe428c6.md) | Draft | Correct shared goods/allocations; carrier capital text and 45HC mapping | .002956 |
| [20 66d5d7f6](../artifacts/kie-labeling/direct-pilot-20261002-luna-high20/comparisons/20-66d5d7f6.md) | Rejected | Empty contacts; good three-product graph in raw answer; alternative addresses concatenated | .003014 |

The 17 structurally valid targets contain 72 party records (58 addressLine
values), 36 containers, 36 goods entries, 31 placement entries and one DG
declaration. These are inventory counts, **not counts of approved annotations**.

## Comparison limits and recommended next step

An old-label F1 would reward mistakes such as the six PAPER REELS goods and
penalize intended changes such as full addresses, merged shared goods, split
quantified products, removal of AAI and retention of reference captions. The
analysis therefore records scalar/path differences as **agreement diagnostics**,
not accuracy. Array-index diffs are especially misleading after a topology change.
The comparison files expose the actual before/after content instead.

Recommended next experiment, pending discussion:

1. Address the two zero-information object failures explicitly, without removing
   any populated field or paying to re-extract an otherwise complete response.
   Decide the malformed-source-ID representation separately; never fix it by
   inventing a compliant value.
2. Run the planned bounded section reviewers on these same outputs. Give the
   relevant reviewer known source adjudications, and allow narrowly requested PDF
   layout inspection where OCR reading order is ambiguous. Keep OCR-only value
   grounding. Do not turn every scalar into an evidence-production task again.
3. Use the concrete findings above as a reviewer detection test, including
   omissions as well as false positives. Measure which defects are detected and
   corrected, whether sound cargo/address facts survive unchanged, new errors,
   unresolved decisions, and total incremental cost.
4. Only then decide whether to expand. No stronger model, full-schema rewrite or
   large processing campaign is justified by this pilot alone.

Validation performed in this turn: all saved valid targets replayed through V7;
all 20 response bodies parsed; exactly one receipt per selected document; source
and implementation hashes unchanged; two in-memory all-null recovery probes;
**15 existing direct-flow tests passed in 10.23 seconds**, without paid calls.
No prompt or field semantics were tuned after inspecting the panel.
