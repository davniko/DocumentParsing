# Integrated B/L synthesis semantic plan and generator-method probe

Date: 2026-08-31

## Outcome

The deterministic/statistical part of the B/L synthesis pipeline now has one composed,
same-template semantic hand-off. It joins the structured generator, route scenario, and
controlled semantic generator without allowing overlapping field ownership. It remains
deliberately non-training output until party identity, cargo language, printed categorical
surfaces, DG semantics, and raw-OCR patching are complete.

Dangerous-goods generation is not implemented from the 38 observed corpus rows. Those rows
are evaluation evidence, not sufficient coverage for a broad generator. DG is deferred to a
goods-first stage that must select or synthesize the goods, determine dangerousness, and only
then produce one coherent set of UN number, proper shipping name, hazard class, subsidiary
risk, packing group, and conditional flashpoint. The current pipeline does not infer DG from
HS code or independently sample DG fields.

## Implemented execution chain

1. Structured generation preserves template topology while regenerating document/voyage,
   container, and seal identifiers; shifts dates jointly; samples quantities and measures;
   and reconciles allocations.
2. Route generation selects valid maritime countries and ports, constructs commercial
   origin/destination context, projects party localities, and samples freight semantics.
3. Controlled generation applies package/equipment categories, reefer settings, HS codes,
   cargo origin, and route/freight results.
4. The semantic-plan composer validates all input commits and hashes, proves exact source
   selection equality, rejects overlapping changed paths, applies both ledgers in order,
   and validates the final strict schema and relational inverse.

Key implementation and configuration:

- `src/document_ocr/synthesis/semantic_plan_pipeline.py`
- `src/document_ocr/synthesis/controlled_generation_pipeline.py`
- `src/document_ocr/synthesis/hs_scenarios.py`
- `configs/synthesis/mpci_bl_combined1157_semantic_plan50.yaml`
- `configs/synthesis/mpci_bl_combined1157_controlled_semantic_pilot50.yaml`
- `configs/synthesis/mpci_bl_combined1157_route_scenario_pilot50.yaml`

The route pilot contains 50 distinct templates: 35 B/L and 15 SWB. Its commercial-origin
prior used 41 observed-exporter draws and 9 maritime-registry exploration draws. It covered
16 origin countries and 25 destination countries; the country sampler does not emit observed
aliases as label values.

## Measured 50-document pilot

| Stage | Runtime | Peak RSS | Principal result |
|---|---:|---:|---|
| Structured v11 | 234.60 s | 834.21 MiB | 50/50 schema, inverse, capacity, arithmetic, and change-ledger valid |
| Route v13 | 4.54 s | 444.72 MiB | 50/50 route scenarios over the exact structured selection |
| Controlled v4 | 6.59 s | 467.37 MiB | 50/50 schema and relational-inverse valid |
| Semantic plan v1 | 1.46 s | 86.86 MiB | 50 exact two-stage chains; 0 overlapping changed paths |

The composed plans contain 1,560 changed leaves: 779 structured changes and 781 controlled
changes. No training record was published.

Controlled coverage included:

- 50 route/freight projections;
- 50 package projections across 21 observed output categories;
- 39 equipment-bearing documents across 23 generated size/type codes;
- 32 HS-bearing documents and 97 HS outputs;
- 5 cargo-origin projections;
- 2 DG documents explicitly deferred; and
- 3 handling-instruction documents explicitly deferred.

The immutable composed artifact is
`artifacts/kie-synthesis/mpci-bl-combined1157-semantic-plan50-v1/`.

## HS length handling

HS6 is selected from the pinned registry. Source output length is preserved from 6 through
18 digits. Where an exact jurisdictional leaf is available it is used; otherwise the national
suffix is generated deterministically, digit-only, collision-checked, and explicitly marked
as synthetic and non-authoritative.

The 50-document pilot emitted 97 values:

| Output length | Count |
|---:|---:|
| 6 | 8 |
| 7 | 1 |
| 8 | 47 |
| 9 | 1 |
| 10 | 19 |
| 11 | 1 |
| 12 | 20 |

Statuses were 8 global HS6 values, 1 exact GB tariff leaf, and 88 synthetic national suffixes.

## Isolated generator benchmark design

The method probe used only the training partition: 778 documents, 600 template groups, and
435 connected template-plus-vessel identity groups. Three grouped outer folds prevent the
same template or vessel identity from crossing fit and held-out sets. Recurrent temperature
selection uses an inner validation split. Every generated vessel name passes the same syntax,
full-source exact/near/substring, and within-batch uniqueness boundary.

The pinned local environment is `environments/synthesis-experiments/`. The final run used:

- NVIDIA RTX 4090, CUDA capability 8.9;
- PyTorch 2.11.0 / CUDA 13.0;
- MostlyAI Engine 2.6.2;
- NameMaker 1.2; and
- matplotlib 3.11.1 and seaborn 0.13.2.

The current upstream `makemore.py` implements RNN/GRU but explicitly does not implement
LSTM. The probe therefore uses compact makemore-style PyTorch GRU and LSTM models with
nested validation and early stopping instead of vendoring the educational CLI. NameMaker
uses its supported `AVG` candidate preference.

Official references:

- MOSTLY AI local/GPU SDK: https://mostly.ai/docs/python-sdk
- MOSTLY AI Engine LanguageModel and TabularARGN: https://github.com/mostly-ai/mostlyai-engine
- NameMaker: https://github.com/Rickmsd/namemaker
- makemore: https://github.com/karpathy/makemore
- SDV synthesizer guidance: https://docs.sdv.dev/SDV/single-table-data/modeling/synthesizers
- SDV CTGAN: https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/ctgansynthesizer
- SDV TVAE: https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/tvaesynthesizer

## Vessel-name results

Lower discriminator excess and BPC gap are better; the real-vs-real row is the attainable
reference. Lexical realism alone is not sufficient because a generator can match coarse
character statistics while still producing malformed names.

| Candidate | Lexical realism | Discriminator excess | BPC gap | Acceptance | Total time, 3 folds |
|---|---:|---:|---:|---:|---:|
| Real vs real | 0.7954 | 0.0395 | 0.0609 | n/a | n/a |
| NameMaker order 2 | 0.8195 | 0.0991 | 0.7681 | 0.9438 | 4.02 s |
| NameMaker order 3 | 0.8104 | 0.1410 | 0.6578 | 0.5154 | 5.42 s |
| NameMaker order 4 | 0.7994 | 0.3090 | 0.7141 | 0.2522 | 14.08 s |
| Character GRU | 0.8084 | 0.1226 | 0.2273 | 0.8969 | 41.32 s |
| Character LSTM | 0.8047 | 0.1624 | 0.1842 | 0.9011 | 45.18 s |
| MostlyAI LSTMFromScratch-3m | 0.7979 | 0.0881 | 0.3336 | 0.9122 | 252.12 s |

Qualitative examples still contain malformed or contaminated-looking constructions across
all candidates. NameMaker order 2 is fast but has a BPC gap more than twelve times the
real-vs-real baseline. Higher NameMaker orders increasingly replay long source fragments.
The recurrent models improve BPC but remain visibly weak, while MostlyAI is substantially
slower and does not close the quality gap. No vessel-name method is promoted.

Voyage identifiers are independent of vessel names and are already handled by the
shape-preserving identifier generator. The composed pilot changed 46 voyage-number leaves.

## MostlyAI party and HS probes

Route-conditioned party generation respected all 300 role/country/city seeds, fitted on
1,854 rows in 25.80 seconds, and sampled in 0.46 seconds. It is rejected for production:

- names were 96.0% unique but 4.67% were exact full-source replays;
- addresses were 99.57% unique but 0.85% were exact full-source replays; and
- length-distribution similarity was only 0.629 for names and 0.366 for addresses.

Raw party strings were not published because the failed replay gate makes them unsafe.

For national HS suffixes, MostlyAI fitted 389 rows but produced only 79/200 structurally
valid results. Of those valid results, 26.58% replayed a complete source code and only 78.48%
were unique. The deterministic collision-checked generator produced 200/200 valid, 200/200
unique outputs with zero source replay. The deterministic method remains the default.

The final immutable benchmark, detailed report, safe vessel examples, and eight plots are in
`artifacts/kie-synthesis/mpci-bl-combined1157-generator-methods-gpu-probe-v3/`.

## Earlier SDV comparison

The existing grouped party-structure benchmark remains relevant for non-linguistic shape:

| Candidate | Mean quality | Raw valid | Novel | Fit time |
|---|---:|---:|---:|---:|
| Empirical | 0.9108 | 1.0000 | 0.0000 | 0.002 s |
| Gaussian Copula | 0.7657 | 0.3884 | 0.9504 | 0.79 s |
| CTGAN | 0.8470 | 0.7997 | 0.8939 | 97.96 s |
| TVAE | 0.8928 | 0.9206 | 0.6705 | 32.04 s |

TVAE was the strongest neural structural candidate on this view, while CTGAN traded some
validity/quality for greater novelty. These results do not authorize either model to generate
party identity strings. SDV's own documentation recommends Gaussian Copula as the fast,
transparent baseline and neural CTGAN/TVAE when measured fidelity justifies their additional
cost; this pipeline therefore selects methods per component rather than declaring one global
synthesizer.

## Remaining boundary before text patching

The following still require a later semantic/linguistic stage before any final training row
can be emitted:

1. complete party names, addresses, and contact anonymization;
2. goods descriptions and auxiliary/flavor information;
3. goods-first dangerous-goods determination and coherent DG facts;
4. natural printed package, equipment, and HS surfaces;
5. handling-instruction language;
6. a higher-quality vessel-name method or agentic generation; and
7. one audited raw-OCR patch operation that receives source target, synthetic target, and
   full source text, preserves layout/formatting, anonymizes auxiliary PII, and proves the
   patched text supports the synthetic label.

These are explicit blockers, not silent fallbacks. The current semantic plans are useful as
the exact structured input to that later stage but are not training samples.

## Validation

- 366 synthesis tests passed; 8 environment-specific tests skipped.
- All changed Python files pass Ruff lint and format checks.
- The synthesis-experiment uv lock resolves 105 pinned packages and passes `uv lock --check`.
- The final method probe completed in 556.37 seconds with 1,819.84 MiB process peak RSS and
  98,958,848 bytes maximum independently reported component CUDA allocation.
- No model weights, unsafe party strings, API calls, or training-ready samples were published.
