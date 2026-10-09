# External-reference footer: narrow synthesis repair

Date: 2026-10-09. Follow-up to the [12-template validation pilot](package-accounting-extension12-2026-10-09.md).

## Decision and cause

The user permits the literal `CN-02-` prefix as untargeted flavor text. No country-prefix interpretation or resampling is required. The only correction is keeping the embedded exporter identifier consistent with its other printed occurrences.

Source `doc_9bd26a2be130f82c3ffff083efbf2b202a78a873bd8dc4d952beae32f3cd3aae` prints exporter ID `91320282050282384L` three times. Its historical composite footer owner covered the entire `CN-02-91320282050282384L` string, preventing the embedded ID from sharing the sampled shipper/exporter value. This affected seven existing synthetic training records and both latest pilot descendants. External references are not extracted targets.

## Applied changes

- In the active 200-template catalog's `ownership.yaml`, the composite owner now selects only the third exporter-ID occurrence. The prefix stays literal.
- In `auxiliary.yaml`, this suffix uses the same identifier recipe as the other exporter-ID occurrences. No production runtime code, prompt or model call was added.
- The catalog manifest records the changed shared-file hashes and this maintenance operation; all 1,200 per-template catalog files retain their stored hashes.
- Seven current synthetic records were repaired in both `synthetic.jsonl` and its `train.jsonl` mirror. Two pilot candidates were freshly rendered from their existing wording through the updated production path, then republished with both position variants and inspection galleries.
- The changed pilot candidates have explicit **manual exact-delta reviews**, preserving the previous full-text model reviews in the backup. They are not represented as newly LLM-reviewed texts.

For example, one training record's footer changes from `CN-02-91320282050282384L` to `CN-02-85021151569801569`, matching its already printed sampled exporter ID. Nothing else changes in that record's text or labels.

## Validation and measured impact

- Exact before/after comparisons: **nine input repairs; zero target changes, zero coordinate changes, zero prefix changes, zero unrelated text changes**. All unchanged dataset records retain their original serialization. The validation set, real-source data, original PDFs and historical synthesis campaigns are unchanged. No records were added or removed.
- Fresh production rendering reproduces the two changed pilot candidates exactly. A 100-seed probe yields 100 distinct sampled identifiers, with all three occurrences agreeing for every seed.
- **244 targeted tests pass**, covering auxiliary generation, ownership/templates, publication, positions and campaign behavior. Five new regression cases cover the generic composite-suffix path and the actual tracked catalog dependency. Ruff and whitespace checks pass.
- Independent geometry validation checks all **3,249 lines across 24 pilot records** and rejects **4/4** deliberately corrupted inputs. All previous coordinates are unchanged. Final coverage remains **2,734/3,249 lines (84.1%)**, including 143/192 goods-description lines. The 24-document/54-page gallery is refreshed.
- Nine alternating cached-render benchmark repetitions: median **18.624 ms before versus 18.791 ms after per two samples** (+0.167 ms, +0.90%; approximately +0.084 ms per sample). This is negligible overhead, not a claimed speedup. The complete repair/publication operation took 18.97 seconds, with 820.74 MiB peak process RSS; this is whole-process memory, not an isolated memory delta. Independent geometry probes took 5.50 seconds with 485.79 MiB peak RSS.
- **Zero API calls; $0 additional API cost.** No training was started and no training config changed.

## Receipts and recovery

[Exact repair receipt](analysis/external-reference-suffix-repair-20261009/receipt.json) contains changed document IDs, before/after strings, file hashes, seed results and timing. [Backups](analysis/external-reference-suffix-repair-20261009/backup/) preserve the previous live mixed dataset and pilot; [catalog backups](analysis/external-reference-suffix-repair-20261009/catalog-before/) preserve the three previous catalog files. Earlier pilot validation receipts remain historical; the new receipt records the authorized subsequent input changes.

The fix addresses repeated-identifier consistency without adding semantics to an excluded reference prefix or changing unrelated source-fixed boilerplate.
