# Dangerous-goods synthesis registry and semantic-plan integration

Date: 2026-09-01

## Outcome

The non-linguistic dangerous-goods stage is implemented as a deterministic,
source-pinned extension of the composed Bill-of-Lading semantic plan. It does
not generate PDFs and it does not yet rewrite raw OCR. It produces strict
relation-v4 semantic targets plus a provenance sidecar for the later text
realizer.

The implementation deliberately does not treat a small collection of observed
training labels as a regulatory generator. Instead, it compiles two official
public sources and exposes two fail-closed sampling branches:

1. `general_regulatory_tuple` samples one complete PHMSA tuple and emits no HS
   code.
2. `hs_linked_exact_chemical` samples an exact ECICS chemical/CUS record linked
   to one unambiguous maritime-eligible PHMSA tuple only when an official ECICS
   chemical name exactly matches the PHMSA proper shipping name after
   mechanical normalization, then emits the first six CN digits as portable
   HS6.

UN number, primary class, subsidiary classes, and packing group are never
sampled independently. HS is never inferred from UN alone.

## Authority and source boundary

- PHMSA oCFR HMT is the machine-readable regulatory-tuple source:
  <https://www.phmsa.dot.gov/standards-rulemaking/hazmat/phmsas-online-cfr-ocfr>
- ECICS is the exact chemical-to-CN bridge:
  <https://taxation-customs.ec.europa.eu/online-services/online-services-and-databases-customs/european-customs-inventory-chemical-substances-ecics-0_en>
- Only six CN digits are treated as globally portable HS. CN8 is EU-specific:
  <https://taxation-customs.ec.europa.eu/customs/common-customs-tariff-cct/tariff-classification-goods/combined-nomenclature_en>
- IMO IMDG Amendment 42-24 remains the maritime compliance authority:
  <https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MSCResolutions/MSC.556(108).pdf>
- UN/EDIFACT DGS confirms that packing danger level and flashpoint are distinct
  concepts:
  <https://service.unece.org/trade/untdid/d12b/trsd/trsddgs.htm>
- WCO's study explains why a universal HS-to-dangerous-goods crosswalk is not
  safe:
  <https://www.wcoomd.org/en/topics/facilitation/resources/permanent-technical-committee/~/media/8E263790AC234D018554E2C6BD99367E.ashx>

The compiled PHMSA registry must not be described or used as an IMDG compliance
engine. Exact classes and source provenance are retained so the generated
semantics are auditable.

BAM's official Dangerous Goods Dataservice was also evaluated because it
publishes downloadable IMDG datasets. It was not acquired or integrated: its
legal notice reserves automated text/data-mining use without prior written BAM
consent, despite the general attribution licence. The pipeline therefore does
not quietly rely on that source:
<https://tes.bam.de/en/dangerous-goods-database/legal-notice-for-the-use-of-dgg-info-and-dangerous-goods-dataservice>.

## Pinned source snapshot

The source manifest is:

`data/registries/dangerous-goods/source-manifest-20260831.json`

Its SHA-256 is
`839c08a3502e473d3136e3c86680510249cc8c2c5a7ff45b188d2255118402f9`.
It pins every acquired byte, including all 140 ECICS result pages, the PHMSA
BIFF workbook, the ECICS archive, and the XLSX member inside that archive.

The compiler validates workbook format, sheet names, column contracts, source
sizes, hashes, page coverage, exact CUS joins, identifier shapes, exact
normalized chemical identity, cardinality balances, and conservative maritime
eligibility before atomic publication.

The final compiled registry is:

`artifacts/kie-synthesis/registries/dangerous-goods-phmsa-ecics-20260831-v4`

It contains:

- 3,687 PHMSA source rows;
- 2,934 valid UN/class tuples;
- 2,909 conservative maritime-eligible tuples across 2,305 UN numbers;
- 3,480 ECICS UN/CN chemical records;
- 470 high-confidence chemical/HS links across 468 UN numbers and 152 HS6
  codes;
- 1,362 otherwise linkable ECICS records retained as audited chemical-name
  mismatches but excluded from sampling;
- all 80 PHMSA rows with two subsidiary hazards, retained without loss.

The immutable transaction and receipt include the compiler implementation hash
as well as source and output hashes. Repeating the same build returned
`created: false`, proving byte-identical deterministic publication.

## Relation-v4 task schema

The relation-v3 target incorrectly nested packing group under flashpoint and
allowed only one subsidiary hazard. Relation-v4 changes only that DG contract:

```json
{
  "unNumber": "1993",
  "hazardCategory": "FLAMMABLE_LIQUIDS",
  "subsidiaryHazardCategories": ["CORROSIVE_SUBSTANCES"],
  "packingGroupCategory": "MEDIUM_DANGER",
  "flashPoint": {
    "temperature": {"value": 16.0, "unit": "celsius"}
  }
}
```

The model-facing values remain readable semantic categories. Exact regulatory
class/division, proper shipping name, ECICS CUS/CAS/CN identifiers,
technical-name requirement, symbols, and vessel stowage remain in the
provenance sidecar.

The deterministic MPCI projection supports:

- broad hazard category to MPCI parent class `1` through `9`;
- `HIGH_DANGER`, `MEDIUM_DANGER`, `LOW_DANGER`, and explicit `NOT_ASSIGNED` to
  MPCI codes `1`, `2`, `3`, and `4`;
- packing-group-only sparse rows without inventing a flashpoint;
- paired Celsius/Fahrenheit flashpoint value and unit.

Projection deliberately fails when more than one subsidiary hazard is present
because the platform artifact establishes only one additional-hazard slot. The
registry retains every subsidiary hazard; the current generation configuration
limits sampled tuples to at most one until the application serialization rule
is confirmed.

## Sampling policy

Sampling is hierarchical and HMAC-keyed, so worker count and scheduling do not
change outputs:

1. Select a semantic hazard category from configured integer weights.
2. Renormalize only over categories explicitly supported by the selected
   branch. This is a named configuration policy, not a silent fallback.
3. Select a UN number within that category.
4. Select one complete PHMSA regulatory row for the UN.
5. For the exact branch only, select an ECICS chemical linked to that row whose
   official chemical name, IUPAC description, first nomenclature description,
   or synonym exactly matches the PHMSA proper shipping name after
   NFKD/ASCII/alphanumeric normalization; then emit its HS6.

The composed plan chooses the exact branch when the source template has an HS
slot on the DG cargo group; otherwise it uses the general branch and preserves
HS absence. DG cardinality, cargo-group membership, and all other template
topology remain unchanged.

Numeric flashpoint is always omitted when the tuple changes. Neither PHMSA HMT
nor ECICS supplies a formulation-specific numeric flashpoint, so retaining the
old template value or synthesizing one from class/packing group would create an
unsupported chemical relationship.

## Corpus evidence

The 1,157-document corpus contains 33 DG documents and 38 DG rows:

- 24 distinct UN numbers;
- 22 rows with HS values;
- HS lengths of 6, 8, and 10 digits;
- 7 numeric flashpoints;
- 6 represented packing groups;
- 1 represented subsidiary hazard.

All 24 observed UN numbers occur in the general compiled HMT branch. Only 1 of
the 24 occurs in the conservative exact-identity ECICS branch, and only 1 of
the 22 HS-bearing real rows matches one of those exact `(UN, HS6)` links. This
confirms that many real goods are mixtures or formulated commercial products
and that UN-to-HS inference is unsafe.

The complete analysis, CSV distributions, and ten matplotlib/seaborn plots are
in:

`artifacts/kie-synthesis/mpci-bl-dangerous-goods-registry-analysis-20260831-v5`

Plots labeled `top25` are views only; the paired CSV files contain complete
distributions.

## Integrated 50-document probe

The final composed output is:

`artifacts/kie-synthesis/mpci-bl-combined1157-semantic-plan50-dangerous-goods-v4-v6`

Results:

- 50/50 strict relation-v4 targets validated;
- 50/50 relational inverse projections matched exactly;
- 2 DG documents and 2 DG rows were preserved;
- both template rows had HS slots and therefore used the exact branch;
- 2 distinct UN numbers and 2 distinct HS6 codes were generated;
- zero numeric flashpoints were invented;
- zero training records were published, by policy.

The 10,000-draw-per-branch validation covered all nine general-branch semantic
categories and all eight exact-branch categories. The exact ECICS branch has no
supported radioactive link in this snapshot. Maximum deviation from the
renormalized configured category weights was 49 permyriad for the general
branch and 73 permyriad for the exact branch, below the configured 300 limit.

## Performance and validation

The original sampler rebuilt immutable joins and category indexes on every
draw. Moving those indexes to registry load time changed measured hot-path
throughput from 708.6 to 75,587 general samples/second and 78,773 exact samples/
second. Registry load took 0.244 seconds in the same final benchmark.

Final measured commands:

- registry compile: 7.87 seconds wall time, 180,812 KiB peak RSS;
- analysis and ten plots: 10.36 seconds, 352,832 KiB peak RSS;
- integrated 50-target plan plus 20,000 validation draws: 2.22 seconds,
  123,636 KiB peak RSS, including environment startup.

Validation results:

- full synthesis regression: 423 passed;
- final DG/schema/CLI/integration subset: 63 passed;
- strict Ruff checks: passed;
- strict mypy checks on every changed DG/schema integration module: passed;
- final registry, analysis, and composed-plan reruns all returned
  `created: false`.

## Deliberately remaining linguistic work

The following belongs to the later PydanticAI text-realization layer and is not
silently approximated here:

- rewrite cargo description/proper shipping name and any required technical
  name into a realistic source-template surface;
- patch all occurrences in raw OCR while preserving page order and formatting;
- anonymize auxiliary party/cargo flavor text not present in the target label;
- emit a numeric flashpoint only when a coherent formulation/property source
  is available;
- validate the patched OCR against the final v4 target before making a record
  training-eligible.

Until those checks pass, every current semantic-plan record remains explicitly
`training_eligible: false`.
