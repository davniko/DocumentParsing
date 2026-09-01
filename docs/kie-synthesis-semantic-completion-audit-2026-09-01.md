# B/L semantic-completion audit and pilot

Date: 2026-09-01

## Outcome

The deterministic synthesis path now composes cargo semantics, thermal intent,
compatible container equipment, temperature setpoints, sparse IMO/flag fields,
and conditional dangerous-goods flashpoints into one relation-v5 plan.  The
stage deliberately publishes no training text: generated semantic targets
remain blocked until the later linguistic/raw-OCR realization stage inserts
matching printed values.

The production-like 50-document probe is pinned by
`configs/synthesis/mpci_bl_semantic_completion50_v5.yaml` and published under
`artifacts/kie-synthesis/mpci-bl-combined1157-semantic-completion50-v5-v9`.

## Container/equipment source audit

The 1,157-document corpus contains 2,115 container rows and 174 distinct
non-null printed type surfaces.  Temperature settings occur on 53 containers
in 45 documents, or 3.89% of documents.  The most frequent temperature-bearing
surfaces are:

| Printed source surface | Rows |
|---|---:|
| `40 REEF 9'6` | 14 |
| `40' HIGH CUBE REEFER` | 12 |
| `40HR` | 4 |
| `40' RH` | 3 |
| `40'X9'6" REEFER CONTAINER` | 2 |
| `REFRIGERATED CONTAINER` | 2 |
| `40RH` | 2 |
| `40RQ` | 2 |

The reviewed grammar resolves a semantic type for 1,937/2,115 source rows and
a complete size/type pair for 1,926/2,115 rows.  Of the 189 rows without a
complete pair, 145 have no printed type and 44 contain size-only, generic, or
carrier shorthand whose height is not established by reviewed evidence.  It
resolves the refrigerated type for 52/53 temperature rows and both size and
type for 44/53 without guessing height:

| Semantic size/type | Temperature rows |
|---|---:|
| Forty-foot high-cube refrigerated | 38 |
| Forty-foot standard-height refrigerated | 5 |
| Twenty-foot standard-height refrigerated | 1 |

The inactive-equipment type marginal contains 1,831 general-purpose, 29
refrigerated, 21 open-top, and one row each for platform,
platform-collapsible, platform-named-cargo, and pressurized-tank.  All 52
type-resolved active-temperature rows are refrigerated.  This is the baseline
distribution used by the generator; the rare source categories are retained
rather than collapsed into general purpose.

`40RH` and `40HR` are treated as carrier high-cube reefer shorthand; `40RF` is
standard-height reefer.  `40RA`, `40RK`, `40RO`, and `40RQ` establish a reefer
surface but do not establish height from the reviewed evidence.  They therefore
contribute to the source type marginal while size is drawn conditionally from
rows with defensible size evidence.  Non-operating reefer (`NOR`) is kept
distinct from an active temperature setting.

The generated target does not learn opaque carrier spellings.  Relation-v5
emits `sizeCategory` and `typeCategory`; the pair deterministically maps to the
four-character application code (for example,
`FORTY_FOOT_HIGH_CUBE + REFRIGERATED -> 45RE`).  The type vocabulary is checked
against all 19 categories in the pinned MPCI registry and every generated code
is checked against the pinned BIC snapshot.  By default, type follows the
reviewed source marginal separately for active and inactive temperature
operation, and size follows the source distribution conditioned on sampled
type and operation.  This retains unambiguous type evidence without inventing
height.  Explicit joint weights can replace either default distribution
through configuration.

BIC documents that the first two code characters describe length/height and
the last two describe type/characteristics:
<https://www.bic-code.org/size-type-code/>.  Its type designations include the
thermal families used by MPCI: <https://www.bic-code.org/type-code-designation/>.
Official carrier material also distinguishes `40RF` reefer from `40RH`
high-cube reefer:
<https://www.maersk.com/~/media_sc9/maersk/local-information/files/asia-pacific/korea/export/soc-container-acceptance-form-guideline-20250318.pdf>
and
<https://www.hapag-lloyd.com/content/dam/website/downloads/pdf/2026_Chinese_New_Year-Shipping_from_China_and_HongKong_The_export_empty_container_pickup_timeline_is_changing.pdf>.

## Stratified goods-to-thermal generation

The source setpoints are all Celsius.  The most frequent values are -18 (20
rows), -20 (6), 0 (5), -22 (4), +5 (4), +1 (3), and +4 (2).  At document level,
26 records are frozen-range, 17 are chilled/cool-range, and two are
positive controlled-ambient cases (+14 and +19).

The deterministic classifier uses the nearest unambiguous HS hierarchy level,
not concatenated parent text.  This prevents a leaf such as “fresh or chilled
lamb” from inheriting “frozen” from a broad parent heading.  The pinned HS
snapshot supplies:

- 93 unambiguous frozen HS6 identities;
- 108 unambiguous chilled HS6 identities; and
- 1,629 configured ambient HS6 identities from low-risk chapters.

Generation order is:

1. assign an exact, deterministic run-level thermal-goods cohort quota;
2. choose the frozen versus chilled/cool profile using the supported source
   document distribution (26:17);
3. sample an HS identity from the matching registry pool;
4. classify the sampled identity back to the same profile as an invariant;
5. select only temperature-capable equipment for every allocated container;
6. generate a profile-compatible setpoint.

This is stratified cargo sampling, not equipment-first inference.  The cohort
assignment supplies configurable prevalence, while the sampled and reclassified
goods identity is the source of truth from which equipment and temperature are
derived.

The four real allocation groups that span multiple containers provide a second
correlation rule: all three groups with complete setpoint evidence repeat one
value across every container; the fourth repeats +4 C on the two containers
where OCR captured a value and omits it on two others.  Synthetic generation
therefore samples one setpoint per thermal cargo group and propagates it to all
allocated containers.  Cross-object validation rejects mismatched allocation
sets or divergent setpoints.

The default run-level thermal prevalence is 389/10,000, matching 45/1,157
source documents.  Frozen values use -24 to -18 C and chilled/cool values use
-3 to +5.5 C in 0.5-degree increments; all are configurable.  FAO guidance
uses -18 C or below for frozen products and values close to 0 C / generally no
more than 4 C for fresh or chilled fish, which supports the separation while
the wider chilled/cool bounds retain observed source surfaces:
<https://www.fao.org/input/download/standards/10273/CXP_052e.pdf>.

The two source controlled-ambient cases are not generalized from HS chapter
alone: pharmaceutical and other controlled-ambient requirements are
product/formulation-specific.  They remain an explicit extension point for a
pinned commodity-profile source or the later agent layer; no current
synthetic record silently claims that relationship.

## IMO and vessel flags

The source contains 11 IMO values (0.95% of documents), all seven-digit values
with valid checksums.  Synthetic values are random six-digit bodies plus the
IMO checksum, collision-checked against source and run values.  The source
leading-digit support is configurable; the baseline uses `9`, matching all 11
observed values.  Presence is an exact HMAC-ranked quota, not a high-variance
per-document Bernoulli draw.

The source contains eight flags (0.69%) and provides no defensible relation to
route or party country.  Flags are therefore sampled independently from 186
ISO countries represented in the pinned maritime-port whitelist.  The target
stores the canonical country name; printed-code/name variation belongs to the
later renderer.  Presence is also an exact run-level quota.

## Flashpoints

The source contains seven flashpoint rows across five documents.  The same UN
number appears with 38, 40, and 50 C, confirming that flashpoint must not be a
fixed UN-number lookup.  The deterministic layer first requires coherent
hazard and explicit physical-form evidence, then samples a formulation-level
value with configurable probability and noise:

- primary class-3, packing group II: below 23 C;
- primary class-3, packing group III: 23 through 60 C;
- other explicitly named liquids/solutions: above 60 C in the configured
  range;
- explicitly wetted/damped/alcohol-containing flammable solids: at or below
  60 C in the configured range.

Generic words such as `mixture`, or `nitrocellulose` without explicit physical
form, do not trigger generation.  The IMO class-3 packing-group bounds are
documented in IMDG 2.3.2.6:
<https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MSCResolutions/MSC.328%2890%29.pdf>.

The selected 50-document upstream DG plan happened to contain two DG rows
that do not meet these flashpoint gates, so the integrated pilot correctly
generated none.  Targeted tests exercise all eligible branches, bounds,
omission behavior, and non-eligible physical forms.

## Pilot and validation results

The final pilot produced:

- 50/50 strict relation-v5 schema-valid targets;
- 50/50 exact relational projection/reconstruction checks;
- 89 container rows: 85 general-purpose, 3 refrigerated, and 1 open-top;
- sizes: 59 forty-foot high-cube, 21 twenty-foot standard-height, 5
  forty-five-foot high-cube, and 4 forty-foot standard-height;
- 2/50 thermal documents, the exact rounded 3.89% run-level quota;
- both active thermal pilot records happened to be chilled; distribution
  fidelity is assessed by the large-sample probe below, not this two-record
  draw;
- 0 IMO and 0 flag values, the exact rounded result of sub-1% rates in a
  50-document probe;
- no training records, by design; all 50 remain pending linguistic
  realization.

Runtime was 3.30 seconds with 137.0 MiB peak RSS.  A separate 100,000-draw
equipment benchmark completed in 0.727 seconds (137,607 draws/second).  Its
largest absolute type-marginal deviation was 2.628 permyriad (0.026 percentage
points).  The largest conditional-size deviation was 93.968 permyriad (0.940
percentage points), on the 1,125 generated open-top rows; common categories had
substantially more support.  The 100,000-draw thermal profile probe produced
60,589 frozen and 39,411 chilled/cool values versus configured weights of
60.47% and 39.53%.

Targeted component/schema tests: 28 passed.  The broader relevant regression
suite: 97 passed and one environment-dependent DG compilation test skipped.
Ruff and strict mypy pass on all newly added completion modules.  An exact
rerun returned `created=false`, proving byte-identical immutable publication
after volatile runtime/RSS telemetry was removed from artifact content.

Two unrelated pre-existing training-config expectation failures remain outside
this work: one test expects start-of-run evaluation to be enabled, and another
expects gradient accumulation 24 while the current user-edited config uses 32.
