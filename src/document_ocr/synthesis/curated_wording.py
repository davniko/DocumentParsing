"""Small fact-conditioned wording requests for the curated synthesis pipeline.

The model writes language, not shipment arithmetic, equipment classes or routes.
Requests are plain text; native structured output carries only requested values.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, create_model

from document_ocr.synthesis.curated_casing import TargetCasing


@dataclass(frozen=True)
class WordingField:
    key: str
    role: str
    example: str
    requirement: str


@dataclass(frozen=True)
class WordingRequest:
    sample_id: str
    context: str
    fields: tuple[WordingField, ...]
    description_groups: tuple[tuple[str, ...], ...] = ()
    gross_weight_kg: Decimal | None = None


class WordingValue(BaseModel):
    """One complete replacement for a requested, source-owned text region."""

    model_config = ConfigDict(extra="forbid")
    key: str = Field(description="Requested field key, copied exactly.")
    text: str = Field(
        min_length=1,
        description=(
            "New text for this region. Addresses contain postal information only; "
            "names are fictional new identities. Postal fields follow the requested "
            "component granularity and approximate line span. Products use natural "
            "paragraph/list boundaries and express the supplied commodity facts."
        ),
    )


class ShipmentWording(BaseModel):
    """Joint wording for one sampled shipment; repeated uses share one value."""

    model_config = ConfigDict(extra="forbid")
    sample_id: str = Field(description="Requested shipment identity, copied exactly.")
    values: list[WordingValue] = Field(description="Exactly one value for every requested key.")


class WordingBatch(BaseModel):
    """A complete batch of independent sampled shipments."""

    model_config = ConfigDict(extra="forbid")
    shipments: list[ShipmentWording] = Field(description="One result per requested shipment.")


class ReviewFinding(BaseModel):
    """A defect or unresolved semantic boundary requiring adjudication before publication."""

    model_config = ConfigDict(extra="forbid")
    field: str = Field(description="Affected target path or printed shipment fact.")
    problem: str = Field(
        description=(
            "Concrete defect or unresolved competing interpretations; never a passing observation."
        )
    )
    evidence: str = Field(description="Short exact quotation from the rendered text.")
    correction: str = Field(
        description=(
            "Supported remedy, or recommended review action when the boundary remains uncertain."
        )
    )


class RenderedFinding(ReviewFinding):
    """A reviewed defect bound to its shipment by the host."""

    sample_id: str = Field(description="Affected supplied shipment identity.")


class ShipmentReview(BaseModel):
    """Actionable defects and unresolved boundaries; neither can silently pass publication."""

    model_config = ConfigDict(extra="forbid")
    findings: list[ReviewFinding] = Field(
        description=(
            "Concrete corrections or genuine unresolved boundaries; omit passing checks, "
            "cosmetic notes and no-change observations."
        )
    )


def review_output_type(count: int) -> type[BaseModel]:
    """The native schema owns shipment coverage; the model never copies hashes."""
    if count < 1:
        raise ValueError("review requires at least one shipment")
    return create_model(
        "ShipmentReviews",
        __config__=ConfigDict(extra="forbid"),
        **{f"s{i}": (ShipmentReview, ...) for i in range(count)},
    )


def unpack_review(output: BaseModel, identities: list[str]) -> RenderedReview:
    payload = output.model_dump()
    if len(set(identities)) != len(identities) or set(payload) != {
        f"s{i}" for i in range(len(identities))
    }:
        raise ValueError("review shipment coverage differs")
    return RenderedReview(
        reviewed_ids=identities,
        findings=[
            RenderedFinding(sample_id=identity, **finding)
            for index, identity in enumerate(identities)
            for finding in payload[f"s{index}"]["findings"]
        ],
    )


class RenderedReview(BaseModel):
    """Final-text semantic review for the requested independent shipments."""

    model_config = ConfigDict(extra="forbid")
    reviewed_ids: list[str] = Field(description="Every supplied shipment identity reviewed once.")
    findings: list[RenderedFinding] = Field(
        description="Defects and genuine unresolved boundaries; empty only when neither remains."
    )


RENDERED_REVIEW_PROMPT = """Audit final synthetic Bill of Lading training examples.
The host has sampled registry commodity identities, routes, equipment, packaging and loads;
the model has written fictional names and postal addresses. Compare the final text and labels
with those supplied facts. Check coherent product meaning, DG/thermal compatibility, repeated facts,
postal ownership and geography, stale source names/localities, and accounting quantities.
HS codes must agree between sampled facts, printed codes and labels; exact HS-to-product
classification is not this synthesis task. Plausible related products, fictional models,
serials, dimensions and capacities may enrich a description. Description continuations
belong to one accounting goods group; repeated printouts must remain consistent.
Description labels copy the main product block and genuine product continuations in source
order, including embedded packing/capacity qualifiers. Headings, loading declarations,
standalone accounting and detached auxiliary/tracking passages remain outside it, as do Marks.
Compare every packing or transport claim inside the description with the supplied shipment:
extra drum/bag/tank claims or unit fill weights need explicit supplied support. Also check
that descriptions contain final commercial wording, not drafting or correction commentary.
Do not reject descriptions for length, line count, or harmless classification differences.
Thermal wording must match the sampled commodity and cold-chain profile, rather than
the source example's product. Every labeled package category needs its own printed type support.
Labels normalize human-readable text to uppercase; rendered text may retain source casing.
Physical newlines become spaces. Countries and equipment aliases may be normalized.
Fictional street addresses need plausible hierarchy, not postal deliverability.
Email and website domains are independent; free-mail addresses are valid business contacts.
National customs captions may be neutral import/export references; do not require a particular
country's filing system. Carrier and separate third-party commercial actors may remain fixed.
The reduced target omits carrier, marks, export references, fax-only contacts and additional
information. Private host-rendered facts can appear in OCR without a public target field.
Package targets count the declared shipment units; container allocations use that same unit.
Contained packing and supporting handling units can differ from that declared level. A sole
identified container's STC declaration can establish its allocation without a separate table.
The supplied contained-packing facts certify private counts and their printed units.
Negotiability follows consignee order wording, not a preprinted document title.
Report concrete contradictions, missing supported target content, invented target content,
or genuinely unresolved boundaries, with a short quotation and recommended action.
State competing interpretations when uncertain; a finding holds publication for adjudication.
Cosmetic preferences and hypothetical shipping schedules
are not defects. Leave findings empty for a passing shipment; report no passing observations.
Review the rendered text, not just the proposed labels."""


WORDING_PROMPT = """Write fictional Bill of Lading training text from the supplied shipment facts.
Return the requested text regions only. Facts and quantities are already chosen by the host.
Generate genuinely new company/person identities. Generate each party's postal fragments
together, appropriate to its supplied locality and country, with roughly the example's
component hierarchy and line span. Examples show structure, not facts to copy. Punctuation
may change. A locality/country belongs once in the complete address, in its requested region.
Keep postal text separate from company names, contacts, tax IDs and headings.
Write natural commercial goods wording that retains every supplied product attribute.
Generate all description fragments jointly, using the complete source as an example of
descriptive depth and list presentation. Use natural item boundaries without line or
character quotas. Fictional product variants and technical qualifiers are welcome.
Host-owned shipment packing, unit fill weights, totals, transport instructions and customs
captions are rendered separately; generate only product identity and specifications here.
Choose product form compatible with supplied contained packing, as well as declared packaging.
Begin with the product wording; keep loading introductions and standalone accounting
statements outside these regions. Attached product qualifiers remain part of the passage.
Any item masses or package contents must agree with the supplied shipment accounting.
Keep product dimensions, size grades, material density and rated capacities distinct from mass.
Use decimal points and optional comma thousands separators in newly generated quantities;
state mass units explicitly as kilograms, grams, pounds or metric tonnes.
Each shipment is an independent variant: vary its product wording and fictional identifiers.
Use uppercase human-readable text. Separate physical lines with newlines when useful."""


def rendered_review_prompt(target_casing: TargetCasing) -> str:
    if target_casing == "uppercase":
        return RENDERED_REVIEW_PROMPT
    if target_casing != "preserve":
        raise ValueError(f"unknown target casing: {target_casing}")
    return RENDERED_REVIEW_PROMPT.replace(
        "Labels normalize human-readable text to uppercase; "
        "rendered text may retain source casing.",
        "Labels preserve generated casing; rendered text can vary casing without changing meaning.",
    )


def validate_postal_geography(
    address: str, *, locality: str, country: str, country_code: str, country_labelled: bool
) -> None:
    """Check the explicit generation contract, not real-address deliverability.

    The generator is asked to use the supplied locality spelling once. Reject
    departures for correction; do not delete apparently duplicate text or guess
    aliases. Country-only text cannot satisfy a distinct locality requirement.
    """

    def normalized(value: str) -> str:
        text = unicodedata.normalize("NFKD", value.casefold())
        text = "".join(char for char in text if not unicodedata.combining(char))
        return " ".join(re.sub(r"[^\w]+", " ", text).split())

    text, place, nation = map(normalized, (address, locality, country))
    country_pattern = rf"(?<!\w){re.escape(nation)}(?!\w)"
    all_country_hits = list(re.finditer(country_pattern, text))
    all_locality_hits = list(re.finditer(rf"(?<!\w){re.escape(place)}(?!\w)", text))

    # Proper-name containment is not a duplicated postal component: BELIZE CITY
    # / BELIZE and SAINT PIERRE / SAINT PIERRE AND MIQUELON work in both directions.
    def separate(hits, containers):
        return [
            hit
            for hit in hits
            if place == nation
            or not any(
                other.start() <= hit.start() and hit.end() <= other.end() for other in containers
            )
        ]

    country_hits = separate(all_country_hits, all_locality_hits)
    # An optional parenthetical qualifier is not needed to recognize an
    # unwanted country line. Keep positive country grounding strict; this
    # extra form only detects leakage into country-less source party shapes.
    short_country = normalized(re.sub(r"\([^)]*\)", "", country))
    short_country_hits = separate(
        list(re.finditer(rf"(?<!\w){re.escape(short_country)}(?!\w)", text))
        if short_country and short_country != nation
        else [],
        all_locality_hits,
    )
    code_at_end = re.search(rf"(?<!\w){re.escape(country_code.casefold())}$", text)
    if not country_labelled and (
        (country_hits and place != nation) or short_country_hits or code_at_end
    ):
        raise ValueError("generated country was not requested for this party")
    if country_labelled and len(country_hits) + bool(code_at_end) != 1:
        raise ValueError("generated postal country must occur once as its supplied name or code")
    # E.g. SAINT-PIERRE / SAINT PIERRE AND MIQUELON: the country's
    # overlapping words are not a duplicate of the separately printed city.
    hits = separate(all_locality_hits, all_country_hits)
    if len(hits) != 1:
        raise ValueError("generated postal locality must occur once with its supplied spelling")


def _has_tariff_declaration(text: str) -> bool:
    for match in re.finditer(
        r"\b(?:HS(?:[ -]*CODE)?|TARIFF\s+CODE)\s*[:.\-]?\s*\d{4,}", text, re.I
    ):
        # A compact identifier explicitly owned by a product-model caption is
        # not a tariff declaration. Bare HS-123456 remains host-controlled.
        if re.fullmatch(r"HS-\d+", match[0], re.I) and re.search(
            r"\b(?:MODEL|TYPE|SERIES)\s*[:#]?\s*$", text[: match.start()], re.I
        ):
            continue
        return True
    return False


_NUMBER = r"\d(?:[\d,.\u00a0 ]*\d)?"
_MASS_UNIT = r"(?:KG(?:S)?|KILOGRAMS?|G|GRAMS?|LB(?:S)?|POUNDS?|METRIC\s+TON(?:NE)?S?|TONNES?)"
_MASS_VALUE = rf"{_NUMBER}\s*{_MASS_UNIT}\b(?!\s*/)"
_ITEM_MASS_CAPTION = (
    r"(?:(?:OPERATING|SHIPPING|TRANSPORT|PACKED|UNIT|ITEM|INDIVIDUAL|MACHINE)"
    r"\s+(?:WEIGHT|MASS)|(?:WEIGHT|MASS)\s+(?:PER|OF\s+(?:EACH|THE))"
    r"\s+(?:UNIT|ITEM|MACHINE|PIECE))"
)
_MASS_QUALIFIER = r"(?:(?:IS|OF|APPROX(?:IMATELY)?\.?|ABOUT|NOMINAL(?:LY)?)\s+)?"
_HOST_OWNED_ITEM_MASS = re.compile(
    rf"\b{_ITEM_MASS_CAPTION}\s*[:=\-]?\s*{_MASS_QUALIFIER}{_MASS_VALUE}"
    rf"|\b{_MASS_VALUE}\s+{_ITEM_MASS_CAPTION}\b"
    rf"|\b(?:WEIGHS?|WEIGHING)\s+{_MASS_QUALIFIER}{_MASS_VALUE}",
    re.I,
)
_ITEM_MASS_VALUE = re.compile(
    rf"(?P<number>{_NUMBER})\s*(?P<unit>{_MASS_UNIT})\b(?!\s*/)",
    re.I,
)
_ENGLISH_NUMBER = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_PRODUCT_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", re.M)
_MASS_FACTORS = {
    "kilogram": Decimal(1),
    "gram": Decimal("0.001"),
    "pound": Decimal("0.45359237"),
    "metric_tonne": Decimal(1000),
}
_GENERATOR_CONTEXT_CAPTION = re.compile(
    r"^\s*(?:SAMPLED\s+(?:COMMODITY|SHIPMENT|GOODS)\s+(?:BRIEF|FACTS|CONTEXT)"
    r"|HOST(?:-OWNED)?\s+SHIPMENT\s+(?:FACTS|DATA|CONTEXT)|STRUCTURE\s+EXAMPLE)\b",
    re.I | re.M,
)


def mass_in_kilograms(value: Decimal | float | str, unit: str) -> Decimal:
    """Convert explicit canonical mass units; unknown units never imply kilograms."""
    if unit not in _MASS_FACTORS:
        raise ValueError(f"unsupported product accounting mass unit: {unit}")
    mass = Decimal(str(value))
    if not mass.is_finite() or mass < 0:
        raise ValueError("product accounting mass must be finite and nonnegative")
    return mass * _MASS_FACTORS[unit]


def _item_masses(text: str) -> list[Decimal]:
    masses = []
    for match in _HOST_OWNED_ITEM_MASS.finditer(text):
        value = _ITEM_MASS_VALUE.search(match[0])
        assert value is not None  # The caption pattern includes this same measured scalar.
        number, unit = value["number"], value["unit"].upper()
        if not _ENGLISH_NUMBER.fullmatch(number):
            raise ValueError(f"generated item mass uses ambiguous numeric notation: {number}")
        canonical = (
            "kilogram"
            if unit.startswith(("KG", "KILO"))
            else "gram"
            if unit.startswith("G")
            else "pound"
            if unit.startswith(("LB", "POUND"))
            else "metric_tonne"
        )
        masses.append(mass_in_kilograms(number.replace(",", ""), canonical))
    return masses


def validate_product_masses(texts: list[str], gross_weight_kg: Decimal | None) -> None:
    """Reject proven unit-mass lower bounds above a supplied whole-cargo gross.

    Each distinctly named bullet/numbered product entry contributes at least
    one unit; alternative masses for one entry contribute only their maximum.
    Plain prose and mass-only specification bullets do not establish independent
    units. Ratings, density and size grades are not actual item-mass declarations.
    This deliberately bounded
    check complements semantic review; it does not parse arbitrary arithmetic.
    """
    if gross_weight_kg is None:
        return
    bound = mass_in_kilograms(gross_weight_kg, "kilogram")
    text = "\n".join(texts)
    masses = _item_masses(text)
    if not masses:
        return
    markers = list(_PRODUCT_LIST_MARKER.finditer(text))
    entry_lower_bounds: dict[str, Decimal] = {}
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        entry = text[marker.end() : end]
        entry_masses = _item_masses(entry)
        if entry_masses:
            caption = _HOST_OWNED_ITEM_MASS.search(entry)
            assert caption is not None
            prefix = entry[: caption.start()]
            # A specification list (operating / shipping weight) is not a list
            # of independent products. Require a named product before its mass;
            # repeated names still contribute only their largest stated mass.
            if not any(char.isalpha() for char in prefix):
                continue
            identity = " ".join(unicodedata.normalize("NFC", prefix).casefold().split())
            entry_lower_bounds[identity] = max(
                entry_lower_bounds.get(identity, Decimal(0)), max(entry_masses)
            )
    lower_bound = max(max(masses), sum(entry_lower_bounds.values(), Decimal(0)))
    if lower_bound > bound:
        raise ValueError(
            "generated product item-mass lower bound "
            f"{lower_bound} kg exceeds whole-cargo gross weight {bound} kg"
        )


def wording_prompt(requests: list[WordingRequest]) -> str:
    if not requests or len({r.sample_id for r in requests}) != len(requests):
        raise ValueError("wording batch requires distinct shipment identities")
    blocks = []
    for request in requests:
        if not request.fields or len({f.key for f in request.fields}) != len(request.fields):
            raise ValueError("wording fields must be nonempty and uniquely keyed")
        fields = "\n\n".join(
            f"FIELD {f.key} — {f.role}\n{f.requirement}\nStructure example:\n{f.example}"
            for f in request.fields
        )
        blocks.append(f"SHIPMENT {request.sample_id}\n{request.context}\n\n{fields}")
    return "\n\n---\n\n".join(blocks)


def wording_output_type(requests: list[WordingRequest]) -> type[BaseModel]:
    """Native schema owns exact shipment/field coverage; the model writes values.

    Short schema keys avoid asking the model to copy hashes and field names into
    a free-form list. Field descriptions carry each region's precise semantics.
    """
    shipments = {}
    for index, request in enumerate(requests):
        fields = {
            field.key: (str, Field(min_length=1, description=field.role + ". " + field.requirement))
            for field in request.fields
        }
        shipment = create_model(f"Shipment{index}", __config__=ConfigDict(extra="forbid"), **fields)
        shipment.__doc__ = "All requested text regions for one independently sampled shipment."
        shipments[f"s{index}"] = (shipment, Field(description=f"Wording for shipment s{index}."))
    result = create_model(
        "ExactShipmentWording", __config__=ConfigDict(extra="forbid"), **shipments
    )
    result.__doc__ = (
        "Exact text fields for the supplied shipments; each value is newly generated text."
    )
    return result


def native_wording_prompt(requests: list[WordingRequest]) -> str:
    """Plain-text requirements also reach providers with external schema decoders."""
    return "\n\n---\n\n".join(
        f"SHIPMENT s{index}\n{request.context}\n\n"
        + "\n\n".join(
            f"{field.key} — {field.role}\n{field.requirement}\nStructure example:\n{field.example}"
            for field in request.fields
        )
        for index, request in enumerate(requests)
    )


def unpack_wording(output: BaseModel, requests: list[WordingRequest]) -> dict[str, dict[str, str]]:
    payload = output.model_dump()
    batch = WordingBatch.model_validate(
        {
            "shipments": [
                {
                    "sample_id": request.sample_id,
                    "values": [{"key": k, "text": v} for k, v in payload[f"s{index}"].items()],
                }
                for index, request in enumerate(requests)
            ]
        }
    )
    return validate_wording(batch, requests)


_SHIPMENT_ACCOUNTING = re.compile(
    r"\b(?:TOTAL\s+(?:PACKAGES?|CARTONS?|PALLETS?)|(?:TOTAL\s+)?(?:GROSS|NET)\s+WEIGHT)"
    r"\s*[:=\-]?\s*\d"
    r"|(?:^|[.!?]\s+)\s*TOTAL\s*[:=]\s*\d"
    r"|^\s*\d[\d,.]*\s+(?:CARTONS?|CTNS?|BOX(?:ES)?|BAGS?|DRUMS?|PALLETS?|PACKAGES?|PKGS?)\s*[.]?\s*$"
    r"|^\s*\d[\d,.]*\s+(?:CARTONS?|CTNS?|BOXES|PACKAGES?|PALLETS?)\s+(?:OF\s+)?(?=\w)",
    re.I | re.M,
)


def has_shipment_accounting(text: str) -> bool:
    """Reject generated shipment-ledger wording, not embedded product specifications.

    This is a generation guard, never a label cleanup regex. Source-owned capacity
    phrases are projected from approved blocks separately. Semantic boundary
    ambiguities still belong to the rendered reviewer and publication hold.
    """
    return _SHIPMENT_ACCOUNTING.search(text) is not None


def validate_wording(
    batch: WordingBatch, requests: list[WordingRequest]
) -> dict[str, dict[str, str]]:
    expected = {r.sample_id: r for r in requests}
    if len(batch.shipments) != len(expected) or {s.sample_id for s in batch.shipments} != set(
        expected
    ):
        raise ValueError("wording output shipment coverage differs")
    result = {}
    descriptions_seen: dict[str, str] = {}
    for shipment in batch.shipments:
        fields = {f.key: f for f in expected[shipment.sample_id].fields}
        if len(shipment.values) != len(fields) or {v.key for v in shipment.values} != set(fields):
            raise ValueError("wording output field coverage differs")
        values = {}
        for value in shipment.values:
            # Generated region whitespace is not source evidence. Keep every
            # nonempty physical line, but normalize padding/blank separators
            # before the deterministic source-style renderer lays it out.
            text = "\n".join(line.strip() for line in value.text.splitlines() if line.strip())
            if not text:
                raise ValueError(f"{value.key}: empty generated region")
            if re.search(r"\\[nr]", text):
                raise ValueError(f"{value.key}: literal escaped line break in generated text")
            if any(re.match(r"\s*[,.;:]", line) for line in text.splitlines()):
                raise ValueError(f"{value.key}: detached line-leading punctuation")
            if "postal" in fields[value.key].role.casefold() and re.search(
                r"(?<![\w.])0{5,6}(?![\w.])", text
            ):
                raise ValueError(f"{value.key}: generated placeholder postcode")
            if "name" in fields[value.key].role.casefold() and re.search(
                r"(?im)^\s*(?:ATTN[.:]?\s+|ATTENTION\s*:)\S", text
            ):
                raise ValueError(f"{value.key}: contact caption inside an identity region")
            if "name" in fields[value.key].role.casefold() and (
                re.sub(r"\W+", "", text).casefold()
                == re.sub(r"\W+", "", fields[value.key].example).casefold()
            ):
                raise ValueError(f"{value.key}: copied source identity")
            if fields[value.key].role == "goods description" and re.search(
                r"\b\d+(?:[.,]\d+)?\s*(?:KG|KGS|KILOGRAMS?|G|GRAMS?|LB|LBS|LITRES?|LITERS?|PCS|PIECES?)"
                r"\s*(?:PER|/)\s*(?:MASTER\s+)?(?:CARTONS?|BOX(?:ES)?|BAGS?|DRUMS?|PACKAGES?|CONES?|SPOOLS?|COILS?)\b"
                r"|\b(?:CONES?|SPOOLS?|COILS?|BAGS?|DRUMS?|CARTONS?)\s+\d+(?:[.,]\d+)?\s*(?:KG|KGS|LB|LBS)\b"
                r"|\bON\s*\d+(?:[.,]\d+)?\s*(?:KG|KGS|LB|LBS)\s*(?:CONES?|SPOOLS?|COILS?)\b",
                text,
                re.I,
            ):
                raise ValueError(
                    f"{value.key}: host-owned package fill inside generated description"
                )
            if fields[value.key].role == "goods description" and (
                _GENERATOR_CONTEXT_CAPTION.search(text)
            ):
                raise ValueError(
                    f"{value.key}: generator context caption inside generated description"
                )
            if fields[value.key].role == "goods description" and has_shipment_accounting(text):
                raise ValueError(f"{value.key}: shipment accounting inside generated description")
            if fields[value.key].role == "goods description" and re.search(
                r"\bPACKAGE_[A-Z_]+\b", text
            ):
                raise ValueError(
                    f"{value.key}: internal package category inside generated description"
                )
            if fields[value.key].role == "goods description" and _has_tariff_declaration(text):
                raise ValueError(
                    f"{value.key}: host-owned tariff caption inside generated description"
                )
            if fields[value.key].role == "goods description" and re.search(
                r"\bFREIGHT\s+(?:PREPAID|COLLECT|PAYABLE)\b|\bACCOUNTING\s+GOODS\s+GROUP\b"
                r"|\bCOUNTRY\s+OF\s+ORIGIN\s*:",
                text,
                re.I,
            ):
                raise ValueError(f"{value.key}: non-product policy inside generated description")
            values[value.key] = text
        validate_product_masses(
            [values[key] for key, field in fields.items() if field.role == "goods description"],
            expected[shipment.sample_id].gross_weight_kg,
        )
        groups = expected[shipment.sample_id].description_groups or (
            tuple(f.key for f in fields.values() if f.role == "goods description"),
        )
        for group in groups:
            if group and all(
                " ".join(values[key].upper().split())
                == " ".join(fields[key].example.upper().split())
                for key in group
            ):
                raise ValueError("copied source description: " + ", ".join(group))
        result[shipment.sample_id] = values
        description = " ".join(
            " ".join(values[key].upper().split())
            for key, field in fields.items()
            if field.role == "goods description"
        )
        if description:
            if description in descriptions_seen:
                raise ValueError(
                    "duplicate generated description across shipments: "
                    + descriptions_seen[description]
                    + ", "
                    + shipment.sample_id
                )
            descriptions_seen[description] = shipment.sample_id
    return result
