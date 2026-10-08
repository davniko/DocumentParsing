"""Small fact-conditioned wording requests for the curated synthesis pipeline.

The model writes language, not shipment arithmetic, equipment classes or routes.
Requests are plain text; native structured output carries only requested values.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

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
    """A concrete mismatch in the final rendered shipment, not a style preference."""

    model_config = ConfigDict(extra="forbid")
    field: str = Field(description="Affected target path or printed shipment fact.")
    problem: str = Field(
        description="Concrete defect requiring correction; never a passing observation."
    )
    evidence: str = Field(description="Short exact quotation from the rendered text.")
    correction: str = Field(description="What must change to reconcile the sampled facts and text.")


class RenderedFinding(ReviewFinding):
    """A reviewed defect bound to its shipment by the host."""

    sample_id: str = Field(description="Affected supplied shipment identity.")


class ShipmentReview(BaseModel):
    """Only actionable defects in this shipment; an empty list means no defects found."""

    model_config = ConfigDict(extra="forbid")
    findings: list[ReviewFinding] = Field(
        description=(
            "Concrete corrections only; omit passing checks, cosmetic notes "
            "and no-change observations."
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
        description="Concrete defects only; empty if none found."
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
Compare every packing or transport claim inside the description with the supplied shipment:
extra drum/bag/tank claims or unit fill weights need explicit supplied support. Also check
that descriptions contain final commercial wording, not drafting or correction commentary.
Do not reject descriptions for length, line count, or harmless classification differences.
Thermal wording must match the sampled commodity and cold-chain profile, rather than
the source example's product. Every labeled package category needs its own printed type support.
Labels normalize human-readable text to uppercase; rendered text may retain source casing.
Physical newlines become spaces. Countries and equipment aliases may be normalized.
Fictional street addresses need plausible hierarchy, not postal deliverability.
National customs captions may be neutral import/export references; do not require a particular
country's filing system. Carrier and separate third-party commercial actors may remain fixed.
The reduced target omits carrier, marks, export references, fax-only contacts and additional
information. Private host-rendered facts can appear in OCR without a public target field.
Negotiability follows consignee order wording, not a preprinted document title.
Report only concrete contradictions, missing supported target content, or invented target
content, with a short quotation. Cosmetic preferences and hypothetical shipping schedules
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
            if fields[value.key].role == "goods description" and re.search(
                r"\b(?:TOTAL\s+(?:PACKAGES?|CARTONS?|PALLETS?)|(?:TOTAL\s+)?(?:GROSS|NET)\s+WEIGHT)"
                r"\s*[:=\-]?\s*\d",
                text,
                re.I,
            ):
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
