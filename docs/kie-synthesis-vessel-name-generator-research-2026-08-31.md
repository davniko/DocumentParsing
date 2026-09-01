# Vessel-name synthesis research and production decision

Date: 2026-08-31
Task: `bill_of_lading_relation_explicit_v3`

## Decision

No vessel-name generator from this experiment is production-approved.

The strongest trainable candidate, MOSTLY AI `LSTMFromScratch-3m`, matches useful held-out aggregate statistics. An initial morphology-oriented review labeled many unfamiliar outputs as malformed, but the task owner subsequently judged the reviewed surfaces plausible for vessel-name extraction training. The model is therefore retained as an experimental future registry-expansion source, while the composed pilot uses whole-name sampling from a separately compiled, provenance-bearing public cargo-vessel registry as the cheaper and simpler default.

Voyage-number generation is unaffected. Its existing character-class-shape generator remains independently usable because it redraws every variable position and validates full-source and batch novelty.

## Public sources examined

### IMO-VESSEL-NAMES

- Source: <https://github.com/interreg-speed/IMO-VESSEL-NAMES>
- Declared license: ODC Public Domain Dedication and License 1.0 in `datapackage.json`.
- Downloaded rows: 13,055 across the repository's `3-vessels.csv`, `4-vessels.csv`, and `5-vessels.csv` files.
- Clean distinct names after the repository's existing contamination filter: 10,795.
- Temporary source hashes:
  - `3-vessels.csv`: `71fb43261599c41cfa9a807f627630112ebc462b8194937df0097ae7f11ba1cc`
  - `4-vessels.csv`: `0307dbf12234b3e4301a2754fa142d2815f97add996c6b18bfda3c3f283286b6`
  - `5-vessels.csv`: `4838e3745b40b25907fe47eb0bf40fe6f48fa69d559dfa41e58d71a9e75302fd`
  - `datapackage.json`: `f1d085e9ea34c535d385a0da618839eec2d2db3d1e45cbbc82a7da9f865f1a5f`

### NOAA PMEL AIS

- Source: <https://data.pmel.noaa.gov/pmel/erddap/tabledap/AIS2021_AIS.html>
- Declared license: CC0-1.0/public-domain dedication.
- Years queried: 2020 through 2024.
- Server-side fields: `VesselName,VesselType`; local acceptance required an integer AIS vessel-type code from 70 through 79 inclusive.
- Accepted rows by year: 6,759; 8,815; 9,167; 9,277; 9,520.
- Clean distinct NOAA names across the five years: 15,796.
- Clean distinct union after deduplicating NOAA and IMO-VESSEL-NAMES: 19,745.

The NOAA expansion is useful and legitimate, but the test proves that more lexical rows alone do not cure the learned lexical artifacts.

## Evaluation contract

All candidates used the same boundary:

1. Normalize to an ASCII uppercase surface and reject malformed punctuation, numeric contamination, and field delimiters with the existing project filter.
2. Reject exact names from the complete public source corpus.
3. Reject names within normalized Levenshtein distance `0.20`, absolute edit distance below `2`, or source-substring overlap at or above `0.70` with at least five characters.
4. Reject duplicates inside a generated batch.
5. Fit on a deterministic hash partition only; evaluate on untouched names.
6. Compare against a size-matched real-versus-real baseline using:
   - lexical realism over character 1- through 4-grams and independent shape features;
   - character TF-IDF discriminator excess over random chance;
   - held-out character-language-model bits-per-character (BPC) gap;
   - length and word-count distribution similarity;
   - direct inspection of generated values.

The first 10,795-name split contained 8,697 fit and 2,098 holdout names. The expanded split contained 15,908 fit and 3,837 holdout names. On the matched initial reference, real-versus-real discriminator excess was `0.03135` and BPC gap was `0.03589`.

## Methods and measured outcomes

| Method | Corpus | Representative/best operating point | Lexical realism | Discriminator excess | BPC gap | Decision |
|---|---:|---|---:|---:|---:|---|
| NameMaker order 4 | 10,795 | higher-order character Markov | not production-selected | not production-selected | not production-selected | Reject: malformed fragments remained |
| Custom 2-layer character LSTM | 10,795 | 256 hidden, temperature 0.95 | 0.8505 | 0.1035 | 0.1314 | Reject |
| MOSTLY AI LSTMFromScratch-3m | 10,795 | temperature 0.70 | 0.8410 | 0.1673 | 0.0406 | Reject: good BPC but malformed values |
| MOSTLY AI LSTMFromScratch-3m | 19,745 | temperature 0.75 | 0.8556 | 0.1310 | 0.0261 | Reject: doubled data did not fix morphology |
| MOSTLY AI LSTMFromScratch-3m | 19,745 | temperature 0.85 | 0.8546 | 0.0484 | 0.1766 | Reject: discriminator improved by trading away likelihood |
| DistilGPT2 full fine-tune | 10,795 | deterministic probe | 0.7663 | 0.2953 | 0.0922 | Reject |
| MOSTLY AI SmolLM2-135M LoRA | 19,745 | temperature 0.75 | 0.8366 | 0.1988 | 0.6675 | Reject: generated generic English phrases |
| Interpolated Witten-Bell token LM | 10,795 | order 2 | 0.8745 | 0.0849 | 0.2064 | Reject: awkward token order and shape skew |
| MOSTLY LSTM plus fit-token gate | 10,795 | temperature 0.75 | 0.8674 | 0.2221 | 0.6401 | Reject: severe common-token mode collapse |
| MOSTLY LSTM plus cross-fit rejection discriminator | 19,745 | 90% real calibration acceptance | 0.8570 | 0.0970 | 0.1917 | Reject: malformed outputs still passed |

The expanded MOSTLY LSTM fit completed 87 epochs, selected epoch 82 at validation loss `1.0492`, required 126.9 seconds end to end (49.5 engine-training seconds), and peaked at 215,054,336 CUDA-allocated bytes. The SmolLM2 arm completed six epochs at best validation loss `1.6028`, required 416.9 seconds end to end (258.5 engine-training seconds), and peaked at 1,523,544,576 CUDA-allocated bytes.

The standard NLTK Kneser-Ney implementation was also probed. Generation scores the large complete vocabulary repeatedly and was not operationally practical at this vocabulary size; its early outputs predominantly replayed common full names. It was stopped rather than misreported as a completed contender.

## Representative outputs

### Plausible values that show the approach's potential

- `VIKING DISCOVERY`
- `BRILLIANT DREAM`
- `OCEAN SERENITY`
- `STELLAR MERCHANT`
- `LADY VISION`
- `CAPE TOPIC`
- `OCEAN GRACE`
- `SEAMAX PEARL`

### Failures that prevent production use

- `TSWOKEELU`
- `CHRIMSONINA`
- `BUXKARANADA`
- `PRAFTK`
- `CHIPOLFINITY`
- `CMA CGM ROSTOVENELLA`
- `HOSPITAL WIFE`
- `THESE THINGS`

The critical observation is that aggregate scores can look convincing while individual names remain unsuitable. Direct-value review is therefore a mandatory selection gate for this field.

## Fixed-sample malformed-output audit

A follow-up audit regenerated 2,400 proposals from the strongest expanded MOSTLY AI `LSTMFromScratch-3m` checkpoint at temperature `0.75` and top-p `0.95`. It applied the complete 19,745-name exact, normalized-distance, absolute-distance, substring-overlap, syntax, and batch-uniqueness boundary used by the experiment. This left 946 accepted novel candidates. A fixed SHA-256-ranked sample of 100 was then inspected under an explicit three-way rubric:

- clearly plausible merchant-vessel name: 40;
- clearly malformed lexical or concatenation artifact: 38;
- unfamiliar or borderline surface requiring review: 22.

This was a conservative morphology-oriented audit rather than a claim about objective linguistic truth. It is not an extraction-suitability failure rate: unfamiliar vessel names can still be valid copy/extraction targets, and the task owner accepted the displayed examples under that operational criterion. The automated boundary remains useful for syntax, source collision, and uniqueness; subjective name familiarity is not an acceptance rule.

A fixed-seed random ten-name subset of that audited sample was:

| Name | Audit disposition |
|---|---|
| `DIAMOND PELICAN` | plausible |
| `KUMPOSA GENES` | borderline |
| `BUFU` | borderline |
| `CAPE VIRGO` | plausible |
| `MYKAL HAYANG` | plausible |
| `POLEDDA` | plausible |
| `JAL SATELIA` | borderline |
| `NATT KOYA` | borderline |
| `NORD TALENT` | plausible |
| `CHIPOL STAR` | malformed |

The same model under the less restrictive syntax/exact-source boundary had an obvious-malformation lower bound between 19% and 24% in two independent 100-name reviews. The production-style novelty boundary is therefore not a realism filter and must not be presented as one.

## Whole-name registry replacement

The source union contains 19,745 distinct syntax-clean names. The default sampler exposes 16,356 high-confidence whole names:

- all 10,795 syntax-clean names from the curated IMO-VESSEL-NAMES files;
- NOAA AIS-only cargo-vessel names observed in at least two different annual snapshots from 2020 through 2024.

The remaining 3,389 NOAA-only names seen in one year are retained in source provenance but quarantined from default sampling. This is a source-evidence rule, not a hand-written alias or exception list. The compiled registry loads once into an immutable tuple and is sampled by deterministic index; it requires no runtime CSV joins or model inference.

## Tokenization finding

The strongest MOSTLY AI checkpoint was not strictly character-by-character. Its persisted tokenizer is a 797-token BPE model with whitespace-aware pre-tokenization; it contains individual letters as fallbacks plus learned fragments such as `AN`, `ER`, `name`, and `vessel`. The custom GRU/LSTM arm was character-level. A larger domain tokenizer or a hybrid whole-token/subword model could reduce broken morphemes, but it would not by itself enforce plausible token combinations. The whole-word language-model probe already preserved tokens and still produced awkward compositions, while a generic pretrained tokenizer/model drifted into ordinary English phrases. Because whole-name registry sampling solves the operational requirement without inference, additional tokenizer research is not justified for the default path.

## Online model search result

No credible vessel-name-specific pretrained generator with a suitable open license and reproducible model artifact was located. Public search results were dominated by rule-based/fantasy boat-name websites or unrelated ship-image/hull generation. The general pretrained models tested here did not learn the narrow vessel-name distribution better than the small from-scratch model.

MOSTLY AI officially supports both its small from-scratch LSTM and pretrained Hugging Face causal language models. The probe used both supported paths, not an unsupported wrapper. See <https://github.com/mostly-ai/mostlyai-engine> and <https://mostly-ai.github.io/mostlyai-engine/api/>.

## Recommended production path

Use deterministic whole-name sampling from the compiled high-confidence public registry by default. Preserve source field presence, exclude names already present in the private training corpus, and enforce batch uniqueness. This gives realistic extraction targets without model inference.

For future runs large enough to need more unique surfaces, generate a bounded proposal set with the archived MOSTLY AI checkpoint, apply the same syntax and complete-registry collision boundary, and append accepted values to a new immutable registry version with explicit synthetic-model provenance. Generated values must never be attributed to the public IMO/NOAA source families.

Only if a later requirement demands genuinely fictional vessel identities should realization become a linguistic step in the PydanticAI/local-model layer:

1. Generate a batch of explicitly fictional vessel names, conditioned where useful on vessel class, route region, carrier surface policy, and requested name shape.
2. Require structured output and retain the generation receipt.
3. Apply the same complete-source exact/near/substring collision guard and batch uniqueness guard.
4. Apply deterministic syntax and shape validation.
5. Validate a held-out pilot before accepting a model/provider as the configured renderer.

A deliberately narrow deterministic fallback could recombine two or more observed vessel-domain tokens, but it should only be enabled as an explicit limited-coverage cohort. It must not masquerade as a complete renderer because it cannot safely reproduce the one-word-name distribution and its unconstrained combinations can be semantically awkward.

## Side effects and repository state

- Pinned source snapshots and a compiled whole-name registry are stored under the gitignored local `data/registries/vessels/` tree.
- The complete fitted MOSTLY AI workspace is preserved under `data/models/vessel-names/mostlyai-lstm-from-scratch-3m-public-corpus-v1/` with a per-file hash receipt; it is not selected by the runtime sampler.
- The controlled semantic pilot supports and validates `public_cargo_vessel_registry_uniform_v1`; no lexical model weights are deployed.
- Temporary lexical-model workspaces remain under `/tmp` and are not runtime dependencies.
- Existing voyage-number synthesis remains unchanged.
