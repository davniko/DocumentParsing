"""Small fact-conditioned wording requests for the curated synthesis pipeline.

The model writes language, not shipment arithmetic, equipment classes or routes.
Requests are plain text; native structured output carries only requested values.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model

from document_ocr.label_schemas.bill_of_lading import ApplicationText
from document_ocr.synthesis.curated_casing import TargetCasing

_APPLICATION_TEXT = TypeAdapter(ApplicationText)
IdentityKind = Literal["individual", "organization", "organization_with_proprietor"]
IDENTITY_KIND_REQUIREMENTS = {
    "individual": "Individual party: generate only the person's full personal name.",
    "organization": "Organization: generate only its legal or trading organization name.",
    "organization_with_proprietor": (
        "Organization with an uncaptioned proprietor: generate the organization name "
        "followed by the proprietor's personal name, without a contact or job-title caption."
    ),
}


@dataclass(frozen=True)
class WordingField:
    key: str
    role: str
    example: str
    requirement: str
    identity_kind: IdentityKind | None = None

    def __post_init__(self) -> None:
        if self.identity_kind is not None and (
            not isinstance(self.identity_kind, str)
            or self.identity_kind not in IDENTITY_KIND_REQUIREMENTS
            or not self.role.endswith(" name")
            or self.role == "contact name"
        ):
            raise ValueError("identity kind requires a certified party-name field")


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


@dataclass(frozen=True)
class RenderedReviewRequest:
    """One exact native review request, shared by execution and offline audit."""

    sample_ids: tuple[str, ...]
    output_type: type[BaseModel]
    system: str
    prompt: str


RENDERED_REVIEW_PROMPT = """Audit final synthetic Bill of Lading training examples.
The host has sampled registry commodity identities, routes, equipment, packaging and loads;
the model has written fictional names, postal addresses and commercial product descriptions.
Source wording illustrates presentation; replacing its commodity with the sampled commodity
is intentional, not a source/identity inconsistency. Compare the final text and labels
with those supplied facts. Check coherent product meaning, DG/thermal compatibility, repeated facts,
postal ownership and geography, stale source names/localities, and accounting quantities.
Check field meaning even when a generated value matches its label exactly: party names contain
identities, postal fields contain address components, contacts contain contact information,
and goods descriptions contain product wording. Flag a company identity inserted into its
address or a tax/contact/manager caption inserted into its name. Genuine named sites and
uncaptioned proprietors are permitted; explain ambiguous ownership rather than deleting them.
Where source-certified identity components are supplied, enforce their declared meaning:
organization means organization identity only, individual means a person's identity only,
and organization_with_proprietor means both. An added proprietor in an organization-only
component is a mismatch even if its name is repeated correctly in the target.
HS codes must agree between sampled facts, printed codes and labels; exact HS-to-product
classification is not this synthesis task. Classification references supply commodity context;
their tariff alternatives and exclusions are not extra cargo requirements. Required product
facts and the sampled physical form define the actual shipment. Plausible related products,
fictional models, serials, dimensions and capacities may enrich a description.
DG commercial prose may differ from the proper shipping name while describing the same
sampled substance/formulation or vehicle propulsion. Separately owned regulatory names and
declarations retain the sampled facts; do not require them inside every product description.
Description continuations belong to one accounting goods group;
repeated printouts must remain consistent.
Description labels copy the main product block and genuine product continuations in source
order, including embedded packing/capacity qualifiers. Headings, loading declarations,
standalone accounting and detached auxiliary/tracking passages remain outside it, as do Marks.
Compare every packing or transport claim inside the description with the supplied shipment:
whole-shipment counts/totals, package units and carrying/equipment claims must agree.
Incidental per-unit fill or capacity may enrich compatible product wording; multiplying it
into an undeclared shipment total or imposing a universal payload bound is not the task. Check
that descriptions contain final commercial wording, not drafting or correction commentary.
Do not reject descriptions for length, line count, or harmless classification differences.
Thermal wording must match the sampled commodity and cold-chain profile, rather than
the source example's product. Every labeled package category needs its own printed type support.
Labels normalize human-readable text to uppercase; rendered text may retain source casing.
Physical newlines become spaces. Country fields use the pinned preferred English names;
addresses retain their printed country presentations, including eligible aliases and ISO codes.
Locality names retain their printed spelling. Name/code frames may print both presentations
of the same country; this is not a duplicated postal component. Equipment aliases may normalize.
Fictional street addresses need plausible hierarchy, not postal deliverability.
Judge consignee contacts from their actual final rendered occurrence and its role:
keep a contact printed directly in the consignee block even if it repeats in the
goods area. Contacts printed only in the goods area are excluded from consignee
targets even when marked as a consignee continuation. Explicitly linked, OCR-supported
postal/country continuations may complete its address/country there. This
consignee-specific contact restriction does not exclude valid notify or other-role
shared-footnote contacts.
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


WORDING_PROMPT = (
    """Write fictional Bill of Lading training text from the supplied shipment facts.
Return exactly the requested fields; other supplied facts are compatibility context.
Generate genuinely new company/person identities. Generate each party's postal fragments
together, appropriate to its supplied locality and country, with roughly the example's
component hierarchy and line span. Examples show structure, not facts to copy. Punctuation
may change. A locality/country belongs once in the complete address, in its requested region.
Keep postal text separate from company names, contacts, tax IDs and headings.
The name field owns the party identity: postal fields must neither repeat it nor
introduce another company identity. Preserve explicitly supplied care-of recipients and named sites.
Changing only an office/location qualifier does not create a new company identity.
When description fields are requested, write natural commercial product wording.
Required product facts and physical form describe the actual shipment. Classification
references suggest the commodity subject and vocabulary; tariff scope and alternatives
are context rather than extra shipment attributes. Use compatible product specifications.
Keep each product variant's physical state internally consistent. Distinct variants may
differ; identify them separately rather than attach competing states to one product.
Generate fragments jointly at the source's descriptive depth and
presentation: concise identity, specification passage, or product list. Cover each
sampled commodity subject. Use natural item boundaries without line or
character quotas.
"""
    "Use approximately the same total word count as the supplied source description "
    "fragments. Contract or distill the supplied goods wording, or expand it as needed, "
    "to reach a comparable overall length while retaining the sampled commodity subjects "
    "and required physical form."
    """
This request owns product identity and intrinsic specifications: material, form, grade,
model, dimensions and technical properties. Whole-shipment package counts, declared contents,
shipment masses, transport settings and references belong to separate host-rendered regions.
Those facts remain outside the requested descriptions, including when an example mentions them.
Keep the product physically compatible with the declared packaging and equipment.
Begin with the product wording; keep loading introductions and standalone accounting
statements outside these regions. Attached product qualifiers remain part of the passage.
Intrinsic product specifications may include dimensions, grades and operating ratings.
Incidental per-unit packing capacity/fill may enrich compatible product specifications; do not
turn it into a new whole-shipment count or weight declaration. Complete each product phrase.
Each shipment is an independent variant: vary its product wording and fictional identifiers.
Use uppercase human-readable text. Separate physical lines with newlines when useful."""
)


def goods_wording_requirement(source_word_count: int) -> str:
    """Apply the tested soft-length/detail policy to model-owned product regions."""
    if type(source_word_count) is not int or source_word_count <= 0:
        raise ValueError("source description word count must be a positive integer")
    return (
        "Complete commercial product wording, matching the source's level of detail "
        "and list structure. Express the required commodity facts and physical form. "
        "Classification wording supplies context, not extra required attributes. "
        "Product variants and intrinsic specifications may enrich an example that "
        "has that detail. Incidental per-unit capacity/fill can be product detail. "
        "Shipment counts, weights, transport settings and codes are separate host facts."
        f" The supplied source description contains {source_word_count} words in total. "
        f"Aim for approximately {source_word_count} words across the requested description fields, "
        "distilling or expanding the sampled goods wording as needed."
        " Follow the example's information pattern as well as its approximate word count: "
        "replace its product identity and carry through analogous attached specification "
        "or detail roles, with newly synthesized values suited to the sampled goods. "
        "Per-unit capacity or fill is part of that pattern. "
        "A source packing expression may also contain a whole-shipment package count: "
        "retain the per-unit detail, but omit that count from model-owned wording; "
        "the host prints declared shipment counts separately. For an example "
        "consisting only of a commodity name, output only a commodity name. Distill "
        "classification wording to leave room for source-like qualifiers."
        " Use surrounding shipment facts to check compatibility, not to expand the "
        "description: do not add HS/tariff declarations, container/equipment or routing "
        "details, or shipment totals. Follow the source's product information instead, "
        "retaining analogous attached specifications and per-unit packing where shown."
    )


POSTAL_CORRECTION_PROMPT = """Correct fictional postal components under joint address assemblies.
The supplied old values failed validation; they are not formatting examples.
Return exactly the requested corrected components. Components jointly form one address;
they are not independent complete addresses. Follow the assembly's locality, country,
certified country-owner and host-literal rules. Some parties omit country; preserve that omission.
Party names are fixed context, not postal components: do not repeat them or insert another
company identity. Preserve explicitly supplied care-of recipients and named sites.
When a care-of recipient is separately owned, keep its name and C/O framing outside
the postal component; do not invent an additional care-of company.
Choose plausible new street/site wording without repeating locality or country inside it.
Preserve approximate address hierarchy and use uppercase text.
Each output contains final component text only, without field headers,
instructions or commentary."""


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


class PostalGeographyCounts(NamedTuple):
    """Generation-contract mentions, excluding geographic name-containment overlaps."""

    country: int
    locality: int
    short_country: int
    terminal_code: bool
    same_name: bool


def postal_geography_counts(
    address: str, *, locality: str, country: str, country_code: str
) -> PostalGeographyCounts:
    """Share exact matching between validation and fixed-component accounting."""

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
    # A host-owned NAME(CODE) frame is one country declaration, not a
    # duplicated name plus an independent code. Inspect its punctuation before
    # normalization removes the brackets; plain "NAME CODE" still counts twice.
    original = unicodedata.normalize("NFKD", address.casefold())
    original = "".join(char for char in original if not unicodedata.combining(char))
    country_literal = unicodedata.normalize("NFKD", country.casefold())
    country_literal = "".join(char for char in country_literal if not unicodedata.combining(char))
    frame = re.search(
        r"(?<!\w)"
        + r"\s+".join(re.escape(token) for token in country_literal.split())
        + rf"\s*(?:-\s*)?\(\s*{re.escape(country_code.casefold())}\s*\)\s*[.,;]?\s*$",
        original,
    )
    hits = separate(all_locality_hits, all_country_hits)
    return PostalGeographyCounts(
        len(country_hits),
        len(hits),
        len(short_country_hits),
        bool(code_at_end) and frame is None,
        place == nation,
    )


def validate_postal_geography(
    address: str, *, locality: str, country: str, country_code: str, country_labelled: bool
) -> None:
    """Check the explicit generation contract, not real-address deliverability.

    The generator is asked to use the supplied locality spelling once. Reject
    departures for correction; do not delete apparently duplicate text or guess
    aliases. Country-only text cannot satisfy a distinct locality requirement.
    """
    # These are literal orchestration captions, not ordinary postal vocabulary.
    # Detect a copied instruction even when its geography happens to be correct.
    if re.search(
        r"\b(?:POSTAL\s+ASSEMBLY|(?:COUNTRY|LOCALITY)\s+OWNER\s+REGIONS|INVALID\s+OLD\s+VALUE|"
        r"PREVIOUS\s+COMPLETE\s+ADDRESS|POSTAL\s+CORRECTION)\b"
        r"|(?:\b|_)POSTAL_COMPONENT_[0-9]+\b\s*[\"']?\s*:"
        r"|\b[A-Z_][A-Z0-9_]*_POSTAL_COMPONENT_[0-9]+\b"
        r"|\bLEXICAL_[0-9]+(?:_[0-9]+)*\b"
        r"|\bREGION\s+[A-Z_][A-Z0-9_]*\s*\(\s*DOCUMENTPATCH\.PARTIES\."
        r"[A-Z_][A-Z0-9_]*(?:\[[0-9]+\])?\s+POSTAL\s*\)"
        r"|\bDOCUMENTPATCH\.PARTIES\.[A-Z_][A-Z0-9_]*(?:\[[0-9]+\])?"
        r"(?:\.ADDRESS(?:LINE|_LINE(?:_[0-9]+)?))?\s*[\"']?\s*:",
        address,
        re.I,
    ):
        raise ValueError("generated postal text contains a generation instruction caption")

    counts = postal_geography_counts(
        address, locality=locality, country=country, country_code=country_code
    )
    if not country_labelled and (
        (counts.country and not counts.same_name) or counts.short_country or counts.terminal_code
    ):
        raise ValueError("generated country was not requested for this party")
    if country_labelled and counts.country + counts.terminal_code != 1:
        raise ValueError("generated postal country must occur once as its supplied name or code")
    if counts.locality != 1:
        raise ValueError("generated postal locality must occur once with its supplied spelling")


def validate_postal_identity(address: str, *, name: str, locality: str, country: str) -> None:
    """Hold a separate leading copy of this party's name, without guessing organizations.

    Explicit identity/contact captions and a complete leading line equal to the
    generated name are evidence here.
    Care-of recipients, named buildings and inline name/site overlaps are not
    classified by this guard; unrelated identities still require semantic review.
    """

    def normalized(value: str) -> str:
        return " ".join(unicodedata.normalize("NFC", value).casefold().split())

    if _IDENTITY_CAPTION.search(address) is not None:
        raise ValueError("generated postal component contains an identity/contact caption")
    lines = [normalized(line) for line in address.splitlines() if line.strip()]
    identity = normalized(name)
    if (
        len(lines) > 1
        and identity
        and identity not in {normalized(locality), normalized(country)}
        and lines[0] == identity
    ):
        raise ValueError("generated postal component repeats its party identity as a leading line")


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
            f"FIELD {f.key} — {f.role}\n{wording_field_requirement(f)}\n"
            f"Structure example:\n{f.example}"
            for f in request.fields
        )
        blocks.append(f"SHIPMENT {request.sample_id}\n{request.context}\n\n{fields}")
    return "\n\n---\n\n".join(blocks)


def wording_output_type(requests: list[WordingRequest]) -> type[BaseModel]:
    """Native schema owns exact shipment/field coverage; the model writes values.

    Semantic aliases identify the model-facing fields; internal ownership keys
    remain the Pydantic field names and are restored by ordinary model_dump().
    """
    if not requests or len({request.sample_id for request in requests}) != len(requests):
        raise ValueError("native wording requires nonempty unique shipments")
    shipments = {}
    for index, request in enumerate(requests):
        aliases = wording_api_keys(request)
        fields = {
            field.key: (
                str,
                Field(
                    min_length=1,
                    alias=aliases[field.key],
                    description=field.role + ". " + wording_field_requirement(field),
                ),
            )
            for field in request.fields
        }
        shipment = create_model(
            f"Shipment{index}",
            __config__=ConfigDict(extra="forbid", validate_by_name=True),
            **fields,
        )
        shipment.__doc__ = "All requested text regions for one independently sampled shipment."
        shipments[f"s{index}"] = (shipment, Field(description=f"Wording for shipment s{index}."))
    result = create_model(
        "ExactShipmentWording", __config__=ConfigDict(extra="forbid"), **shipments
    )
    result.__doc__ = (
        "Exact text fields for the supplied shipments; each value is newly generated text."
    )
    return result


def wording_field_requirement(field: WordingField) -> str:
    """Expose an identity kind only when the source contract explicitly certifies it."""
    requirement = field.requirement
    if field.role.endswith(" name"):
        requirement += (
            " Generate a different identity, not the source company/person with "
            "a changed office or location qualifier."
        )
    if field.role.endswith(" postal"):
        requirement += (
            " The party name belongs only to its name field; do not repeat it or "
            "insert another company identity into this postal component. "
            "Preserve explicitly supplied care-of recipients and named sites."
        )
    if field.identity_kind is not None:
        requirement += " " + IDENTITY_KIND_REQUIREMENTS[field.identity_kind]
    return requirement


def wording_api_keys(request: WordingRequest) -> dict[str, str]:
    """Deterministic semantic schema aliases, independently numbered by owner/role."""
    if not request.fields or len({f.key for f in request.fields}) != len(request.fields):
        raise ValueError("wording fields must be nonempty and uniquely keyed")
    seen: Counter[str] = Counter()
    mapping = {}
    for field in request.fields:
        role = field.role.replace("documentPatch.parties.", "").replace(
            "notifyParties", "notify_party"
        )
        if role == "contact name":
            role = "personal contact name"
        stem = re.sub(r"[^a-zA-Z0-9]+", "_", role).strip("_").lower()
        if not stem:
            raise ValueError("wording field requires a semantic role")
        seen[stem] += 1
        mapping[field.key] = f"{stem}_component_{seen[stem]}"
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("wording semantic key collision")
    return mapping


def native_wording_prompt(requests: list[WordingRequest]) -> str:
    """Plain-text requirements also reach providers with external schema decoders."""
    blocks = []
    for index, request in enumerate(requests):
        mapping = wording_api_keys(request)
        context = re.sub(
            r"\{([a-z][a-z0-9_]*)\}",
            lambda match, aliases=mapping: "{" + aliases.get(match[1], match[1]) + "}",
            request.context,
        )
        fields = "\n\n".join(
            f"{mapping[field.key]} — {field.role}\n{wording_field_requirement(field)}\n"
            f"Structure example:\n{field.example}"
            for field in request.fields
        )
        blocks.append(f"SHIPMENT s{index}\n{context}\n\n{fields}")
    return "\n\n---\n\n".join(blocks)


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
    r"|\bTOTAL(?:\s+|\s*[:=]\s*)\d[\d,.]*\s*"
    r"(?:MT|MTS|KG|KGS|KILOGRAMS?|LBS?|POUNDS?|TONNES?|METRIC\s+TON(?:NE)?S?)\b(?!\s*/)"
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


_DG_DECLARATION = re.compile(
    r"\bPACKING\s+GROUP\s*[:=\-]?\s*(?:I{1,3}|[123])\b"
    r"|\b(?:IMO|IMDG|DG)\s+CLASS\s*[:=\-]?\s*\d"
    r"|\bUN\s+(?:NUMBER|NO\.?)\s*[:=\-]?\s*\d{4}\b"
    r"|\bFLASH\s*POINT\s*[:=\-]?\s*[+\-]?\d",
    re.I,
)


_IDENTITY_CAPTION = re.compile(
    r"\b(?:ATN|ATTN|ATTENTION|CONTACT|AGENTE|MANAGER|DIRECTOR)\s*[:.]"
    r"|(?:^|\n)\s*ATTN\s+\S"
    r"|\b(?:MANAGING\s+(?:DIRECTOR|PARTNER)|SALES\s+DIRECTOR|TRADING\s+MANAGER"
    r"|DIRECTOR\s+OF\s+OPERATIONS|DIRECTOR\s+DE\s+OPERACIONES|DIRECTOR-GERAL"
    r"|FIRM\s+OF\s+MANAGER|GERENTE(?:\s+GENERAL)?)\b"
    r"|\b(?:SHIPPER|CONSIGNEE|NOTIFY(?:\s+PARTY)?|DELIVERY\s+AGENT|FORWARDING\s+AGENT|CARRIER)\s*:"
    r"|\b(?:RIF|RFC|CIF|VAT|TIN|TAX\s*(?:ID|NUMBER)|REGISTRATION\s*(?:NO\.?|NUMBER))"
    r"\s*[:#]?\s*[A-Z]?[- ]?\d"
    r"|\bTHIS\s+(?:LINE|VALUE|TEXT|FIELD)\s+IS\s+(?:(?:AN?|THE)\s+)?"
    r"(?:NAME|IDENTITY|COMPANY\s+NAME)\s+FIELD\s+ONLY\s*:",
    re.I,
)
_ACTOR_RELATIONSHIP = re.compile(
    r"\b(?:ON\s+BEHALF\s+OF|O\s*/\s*B|C\s*/\s*O|(?:IN\s+)?CARE\s+OF"
    r"|A\s*/\s*C|AS\s+AGENTS?\s+OF|AS\s+PRINCIPAL)\b",
    re.I,
)
_PRODUCT_PARTY_CAPTION = re.compile(
    r"\b(?:DELIVERY\s+AGENT|CONSIGNEE|SHIPPER|NOTIFY\s+PARTY)"
    r"\s+(?:CONTACT|ADDRESS|TELEPHONE|EMAIL)\s*:",
    re.I,
)


def has_field_scope_caption(field: WordingField, text: str) -> bool:
    """Hold explicit cross-field captions; do not classify uncaptained identities.

    CONTACT TECHNOLOGIES may be a company and CO. KILDARE a postal component.
    Ambiguous vocabulary and identity-in-address cases need semantic review;
    this bounded gate never deletes or relocates generated text.
    """
    if field.role.endswith(" name"):
        return (
            _IDENTITY_CAPTION.search(text) is not None
            or _ACTOR_RELATIONSHIP.search(text) is not None
        )
    if field.role.endswith(" postal"):
        return _IDENTITY_CAPTION.search(text) is not None
    if field.role == "goods description":
        return _PRODUCT_PARTY_CAPTION.search(text) is not None
    return False


def identity_key(text: str) -> str:
    """Compare identity wording independently of parenthetical office qualifiers.

    Qualifier-only changes still disclose the source identity. This does not
    match common-word substrings or decide ownership of independent actors.
    """
    if "(" in text:
        core = re.sub(r"\([^)]*\)", "", text)
        key = re.sub(r"\W+", "", core).casefold()
        if key:
            return key
    return re.sub(r"\W+", "", text).casefold()


def validate_wording(
    batch: WordingBatch, requests: list[WordingRequest]
) -> dict[str, dict[str, str]]:
    """Validate every shipment independently; wording reuse is a diversity observation."""
    expected = {r.sample_id: r for r in requests}
    if len(batch.shipments) != len(expected) or {s.sample_id for s in batch.shipments} != set(
        expected
    ):
        raise ValueError("wording output shipment coverage differs")
    result = {}
    for shipment in batch.shipments:
        fields = {f.key: f for f in expected[shipment.sample_id].fields}
        reserved_field_lines = {
            key.casefold()
            for key in (*fields, *wording_api_keys(expected[shipment.sample_id]).values())
        }
        template_keys = reserved_field_lines | {
            k.casefold()
            for k in (
                *fields,
                *re.findall(r"\{([a-z][a-z0-9_]*)\}", expected[shipment.sample_id].context, re.I),
            )
        }
        if len(shipment.values) != len(fields) or {v.key for v in shipment.values} != set(fields):
            raise ValueError("wording output field coverage differs")
        values = {}
        for value in shipment.values:
            # Generated region whitespace is not source evidence. Keep every
            # nonempty physical line, but normalize padding/blank separators
            # before the deterministic source-style renderer lays it out.
            text = normalize_wording_text(value.text)
            if not text:
                raise ValueError(f"{value.key}: empty generated region")
            if text.casefold() == "tbd":
                raise ValueError(f"{value.key}: unfinished generated region")
            if any(
                key.casefold() in template_keys
                for key in re.findall(r"\{([a-z][a-z0-9_]*)\}", text, re.I)
            ):
                raise ValueError(f"{value.key}: unresolved generation template variable")
            for line in text.splitlines():
                _APPLICATION_TEXT.validate_python(line)
            if re.search(r"\\[nr]", text):
                raise ValueError(f"{value.key}: literal escaped line break in generated text")
            if any(re.match(r"\s*[,.;:]", line) for line in text.splitlines()):
                raise ValueError(f"{value.key}: detached line-leading punctuation")
            if "postal" in fields[value.key].role.casefold() and re.search(
                r"(?<![\w.])0{5,6}(?![\w.])", text
            ):
                raise ValueError(f"{value.key}: generated placeholder postcode")
            if has_field_scope_caption(fields[value.key], text):
                raise ValueError(f"{value.key}: identity/tax/contact caption outside its field")
            if "name" in fields[value.key].role.casefold() and (
                identity_key(text) == identity_key(fields[value.key].example)
            ):
                raise ValueError(f"{value.key}: copied source identity")
            # A per-package fill or cone capacity is product/packing wording,
            # not a shipment total. Keep it for semantic compatibility review;
            # do not reject equivalent descriptions based on phrase order.
            # Explicit shipment accounting remains separately guarded below.
            if fields[value.key].role == "goods description" and (
                _GENERATOR_CONTEXT_CAPTION.search(text)
            ):
                raise ValueError(
                    f"{value.key}: generator context caption inside generated description"
                )
            if fields[value.key].role == "goods description" and any(
                line.casefold() in reserved_field_lines for line in text.splitlines()
            ):
                raise ValueError(f"{value.key}: requested field key inside generated description")
            if fields[value.key].role == "goods description" and has_shipment_accounting(text):
                raise ValueError(f"{value.key}: shipment accounting inside generated description")
            if fields[value.key].role == "goods description" and _DG_DECLARATION.search(text):
                raise ValueError(
                    f"{value.key}: host-owned DG declaration inside generated description"
                )
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
    return result


def normalize_wording_text(text: str) -> str:
    """Normalize provider padding without changing any non-whitespace content."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def repeated_goods_descriptions(
    requests: list[WordingRequest], wording: dict[str, dict[str, str]]
) -> list[list[str]]:
    """Report shipments sharing complete product wording, without rejecting valid labels.

    Independent loads and equipment can legitimately accompany the same product
    wording. Run this after validation; it describes lexical diversity rather
    than certifying or invalidating any shipment's extraction supervision.
    """
    descriptions: dict[str, list[str]] = {}
    for request in requests:
        values = wording[request.sample_id]
        description = " ".join(
            " ".join(values[field.key].upper().split())
            for field in request.fields
            if field.role == "goods description"
        )
        if description:
            descriptions.setdefault(description, []).append(request.sample_id)
    return [identities for identities in descriptions.values() if len(identities) > 1]
