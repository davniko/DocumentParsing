# R27 real600: manual surgical anonymization

All 600 real-training documents have been manually reviewed and anonymized in
their R27 training copy. No new dataset version, synthesis campaign, relabeling
run or training run was created. The separate
[additional500 goods refresh](kie-goods-description-refresh500-r27-2026-10-16.md)
is already included in the same R27 directory.

Dataset: `data/curated/mpci-bl-real600-synthetic7000-country-presentations-r27-positions-v1`.
Audit evidence: `docs/analysis/r27-real-anonymization-20261016/`.

## Scope, method and provenance

The 600 real records were already present in R27. Before editing, their exact
JSONL and manifest/README were backed up locally; historical datasets and
original PDFs remain unchanged. Pre-edit real shard SHA:
`a11f784fa8ee96d57d73f584a4714fc4a38204455b0e7577212103ba704ccc0b`.

Eight Codex manual reviewers opened each assigned document's complete OCR and
existing target, including all 1,160 supplied OCR pages. They individually chose
literal replacements and explicitly listed every dependent target update.
Additional cross-review inspected10 complex peer documents. The primary agent
adjudicated boundaries, inspected all goods/handling target changes, checked
residual identities and ran the publication checks.

No script, dictionary, random generator, synthesis agent or relabeling model
invented the replacements. Audit tools only displayed, applied and validated
the explicitly authored edits. ISO6346 arithmetic checked manually chosen
container bodies. There were no separate paid generation calls for real-data
anonymization; Codex worker usage is not included in the goods API ledger.

Each document has a reviewed plan, resulting candidate, exact edit receipt and
inventory entry. Initial75-row shards were supplemented by explicitly assigned
10/19-row transfers; the inventories distinguish the actual authors. All600
plans are complete; no unresolved holds remain.

## What changed

11,315 manually authored edit choices cover 14,964 actual raw-text occurrences
and 9,216 existing target scalars. Repetitions on other pages and non-target
private text were included; this was not restricted to the label fields.

|Target component|Changed scalar values|
|---|---:|
|Party names|1,810|
|Address lines|1,533|
|Contact names|165|
|Phone/fax values|1,011|
|Email addresses|344|
|Website URLs|5|
|B/L numbers|477|
|Master B/L numbers|3|
|Original B/L number|1|
|Container IDs plus existing placement references|2,502|
|Seal values|1,328|
|Goods-description identifying substrings|30|
|Handling-instruction identifying substrings|7|

All 1,251 labeled containers and their 1,251 placement links are changed
consistently. The 30 description edits replace embedded private VIN/serial/lot
identifiers, not commodity wording or specifications. The seven handling edits
replace private references/party identities, not geographical destination names.
Unlabeled booking, invoice, tax/customs, contact, Marks and continuation identities
are covered by the raw-edit receipts and do not require adding target fields.

## What remains unchanged

- Every existing target key/list/null, ownership decision, category, quantity,
  mass/volume, HS/DG/thermal fact, freight fact and negotiability/sameAs value.
- Dates, vessel/voyage/shipping facts, ports/localities/countries and known
  geographic postcodes. No geographic correction or country resampling here.
- Cargo wording, public product brands/models, dimensions and qualifiers,
  except the explicitly identifying substrings described above.
- All raw whitespace, punctuation, physical line/page boundaries and casing
  style. Phone dialing conventions, separators, group widths and source ID masks
  remain; no E.164 conversion, comma insertion or container-format flattening.
- All original line-coordinate suffixes, attached to the same physical lines.
  These real inputs do not use synthetic reflow or newly generated coordinates.
- Public carrier/issuer identity and general corporate directory where its
  document role establishes the exemption; individual staff/signatories and
  personal mailboxes are anonymized independently.
- Synthetic shards after the500 goods refresh, validation60, schemas, constraints,
  prompts, quarantine provenance and original source PDFs.

### Bounded adjudications

- User decision: anonymize genuinely unresolved address numbers, retaining width
  and format. Known geographic postcodes and obvious all-zero placeholders stay.
- Geographic districts/industrial zones and named public terminal/dry-port/depot
  destinations stay. The same name in an actual private-party block can still be
  anonymized; the roles are reviewed separately. Final QA restored proposed
  over-masking of SOSDI/KabariEFS destinations, matching the treatment of the
  RAYA/TEDA transit-destination list. Private factory/consignee legal names do
  not become geographical exemptions merely because they appear in an address.
- Public safety-service constants, clearly identified governmental project
  context, and public carrier form/legal constants may remain with explicit
  evidence. A publicly searchable private trader alone is not exempt.
- Product identifiers explicitly adjudicated as cargo wording, including
  `1012322163DXH DEGREE1`, were not reclassified as private order references.
- Date-shaped text was not blanket-matched: explicit dates remain; a captioned
  invoice number or opaque Marks reference can be anonymized without treating
  it as an issue/on-board date.

## Problems caught before publication

The unchanged-target grounding check rejected an over-broad short exporter-ID
edit that would also have changed HS digits; its replacement was restricted to
the correct source occurrence. Additional checks caught and corrected lost
container zero-padding, coincident replacement B/L/container IDs, private mailbox
suffixes and the geographical over-masking described above. These were staged
proposals, not silently published regressions.

The final pass finds no old full party-name literal remaining for a changed
party, no unchanged structured contact under a renamed private party, and no
different original B/L/container identities collapsed into the same replacement.
These are targeted checks, not a claim that regular expressions prove privacy.

## Validation and measured execution

- All600 manual plans replay exactly; frozen direct-target schema, casing,
  copy-grounding, container checksums and placement links pass.
- Protected targets, list topology, raw punctuation/whitespace and each real
  coordinate suffix are preserved. Unlisted target changes are rejected.
- Actual training loader passes7,600 train /60 validation records with the
  existing input-only prompt and direct JSON target.
- Final CPU-only actual training tokenizer:600 records, maximum7,838 source /
  3,145 target tokens, against19,200 /5,500; zero source truncations.
  5.01s,1,174,984KiB peak RSS including framework imports; no GPU or model weights.
- Full600 literal replay measured4.75s /85,848KiB peak RSS in the pre-publication
  check. Final staging plus native loader11.62s /1,034,900KiB peak RSS.
- Ten targeted safeguard tests pass in1.25s; Ruff passes. Tests include protected
  geography, stale quotes/targets, whitespace/punctuation, missing grounding,
  unresolved holds, HS substring collisions, target casing and container padding.

No production training/generation hot path changed. These are validation timings,
not claimed GPU-training speedups; training throughput was not benchmarked.

## Publication and review inventory

Publication commits only `real-train.jsonl`, README and manifest, with manifest
last. All other current R27 files must match their pre-publication hashes.
The frozen plans, candidate digest and native tokenizer receipt are pinned before
commit. `publication.json` and the independent readback record final hashes.

Final independent readback passed all 600 published rows in 9.87 seconds,
209,072 KiB peak RSS, including another native training-loader inspection.
All 85,003 physical lines retained their original coordinate suffixes;
52,140 lines have coordinates. Every non-real R27 file is unchanged by this
publication. These are physical-line counts, not a new content-line coverage rate.

Final real-shard SHA:
`3ffa9718c65256c544a5ebe6c658ed38e5a5f2c766b8b1f0ff75523c8a3db20b`.
Final R27 manifest SHA:
`e58854b0c6e3db386380edc9bc5a8f38c1bee64fa1643e12cae705b3c5a6d220`.

Evidence includes:

- `SCOPE.md`, `before/real-train.jsonl`, initial and pre-publication manifests;
- `worker01/` through `worker08/`: plans, candidates, receipts, per-row inventories;
- transfer supplements under workers03/07/08;
- `token-limit-validation.json`, `staged-validation.json`, `publication.json`;
- `inventory.md` for all600 records and `independent-readback.json`.

An earlier uncommitted staging proposal was retained as rejected audit evidence
after the geographic QA correction; it is not a training source or a new dataset
version. Only the final R27 shard is the approved current copy.

## Limits

This is scoped anonymization of training inputs/labels, **not a guarantee of
irreversible de-identification**. Preserved vessels, dates, cargo and geography
can still link a shipment. Original PDFs, historical datasets and private audit
backups intentionally remain recoverable and must not be mistaken for anonymized
training inputs. Validation remains unchanged and unanonymized by this task.
The previously documented validation phone-completeness hold is unchanged.
No historical model was rescored, and no training was launched.
