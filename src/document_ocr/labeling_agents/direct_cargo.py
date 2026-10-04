"""Small source-first cargo accounting model, separate from the training target.

The map retains shared and outer quantities without projecting them onto products.
Literal checks establish presence only; the model still establishes ownership.
"""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, create_model, model_validator

from document_ocr.label_schemas.bill_of_lading_v7 import GoodsItemDetailsV7
from document_ocr.label_schemas.common import LabelSchemaModel
from document_ocr.labeling_agents.direct_models import ReviewFinding, SectionReview

CARGO_ACCOUNTING_FIELDS = (
    "description",
    "hsCodes",
    "numberAndTypeOfPackages",
    "splitGoodsPlacement",
    "grossWeight",
    "netWeight",
    "volume",
)
CargoAccountingValues = create_model(
    "CargoAccountingValues",
    __base__=LabelSchemaModel,
    __doc__=GoodsItemDetailsV7.__doc__,
    **{
        name: (field.annotation, deepcopy(field))
        for name in CARGO_ACCOUNTING_FIELDS
        for field in (GoodsItemDetailsV7.model_fields[name],)
    },
)


class CargoProduct(LabelSchemaModel):
    """One distinct goods identity; repeated container portions share this identity."""

    key: str = Field(min_length=1, description="Unique local key used by statements, e.g. g1.")
    description: str = Field(
        min_length=1,
        description=(
            "OCR product description identifying this group, including distinguishing "
            "models/specifications/lots and printed per-package capacity. Classification "
            "codes and shipping marks are "
            "separate facts, not added description wording."
        ),
    )


class CargoCount(LabelSchemaModel):
    """An explicitly printed count at one packing or product level."""

    quantity: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Printed count, not an inferred split or computed total; null when unstated.",
    )
    packageType: str = Field(
        min_length=1,
        description="Printed package/unit name, e.g. CARTONS, BAGS, PALLETS or PIECES.",
    )
    level: Literal["target", "outer", "product_capacity"] = Field(
        description=(
            "target=innermost identified shipment packaging; outer=containing packaging; "
            "product_capacity=retail/set contents. Pallets can be target when no inner "
            "shipment packaging is identified."
        ),
    )


class CargoMeasure(LabelSchemaModel):
    """A printed cargo measure used to distinguish and reconcile local portions."""

    printedValue: str = Field(
        min_length=1,
        description=(
            "Exact OCR numeric token, including decimal/grouping separators, without unit."
        ),
    )
    value: float = Field(
        gt=0,
        allow_inf_nan=False,
        description=(
            "Numeric interpretation of printedValue without unit conversion. Establish decimal "
            "versus thousands separators from the same table's notation and explicit totals."
        ),
    )
    unit: str = Field(
        min_length=1,
        description="Printed mass/volume unit; no unit inferred from a different measure.",
    )


class CargoStatement(LabelSchemaModel):
    """One source occurrence and its exact scope, not a Cartesian product of lists."""

    products: list[str] = Field(
        description=(
            "Product keys covered collectively by this occurrence; empty for a "
            "shipment-wide fact or genuinely unidentified cargo."
        )
    )
    containers: list[str] = Field(
        description=(
            "OCR-present container IDs only: omit presentation spaces and adjacent "
            "seal/type/tare text. Empty when unidentified. Multiple "
            "goods and containers do not imply every possible membership."
        )
    )
    scope: Literal["portion", "goods_total", "shared_total", "shipment_total"] = Field(
        description=(
            "portion=local cargo occurrence; goods_total=explicit total for one product; "
            "shared_total=unsplit measure covering several products/containers; "
            "shipment_total=whole consignment. Repeated print copies describe one occurrence."
        ),
    )
    packages: list[CargoCount] = Field(
        description=(
            "Printed counts/types by packing level; empty when none is stated. "
            "Do not derive counts here."
        )
    )
    grossWeight: CargoMeasure | None = Field(
        default=None, description="Printed gross cargo mass owned by this occurrence, or null."
    )
    netWeight: CargoMeasure | None = Field(
        default=None, description="Printed net cargo mass owned by this occurrence, or null."
    )
    volume: CargoMeasure | None = Field(
        default=None, description="Printed cargo volume owned by this occurrence, or null."
    )
    explanation: str = Field(
        min_length=1,
        description=(
            "Brief ownership explanation, including row/continuation direction where "
            "material. PDF may establish this ownership but cannot supply values "
            "absent from OCR."
        ),
    )


class CargoSourceMap(LabelSchemaModel):
    """Candidate-blind source accounting; no verdict about existing labels."""

    layoutInterpretation: str = Field(
        min_length=1,
        description=(
            "How product blocks, counts and container rows align, including any page "
            "continuation. Establish direction from the document; neither preceding "
            "nor following text is universally the owner."
        ),
    )
    products: list[CargoProduct] = Field(
        description=(
            "Distinct OCR-supported products/accounting entries; repeated portions "
            "are one product. Names or HS-code multiplicity alone do not force a split."
        )
    )
    statements: list[CargoStatement] = Field(
        description=(
            "All local cargo portions and printed totals, retaining unknowns and "
            "shared scopes without assigning invented splits."
        )
    )
    uncertainties: list[str] = Field(
        description=(
            "Unresolved source associations or conflicting OCR facts only. Correct "
            "omission of PDF-only values or out-of-schema information is not uncertainty."
        )
    )

    @model_validator(mode="after")
    def valid_references(self) -> CargoSourceMap:
        keys = [p.key for p in self.products]
        if len(keys) != len(set(keys)):
            raise ValueError("cargo product keys must be unique")
        for statement in self.statements:
            if not set(statement.products) <= set(keys):
                raise ValueError("cargo statement references absent product")
            if len(statement.products) != len(set(statement.products)):
                raise ValueError("duplicate products in cargo statement")
            if len(statement.containers) != len(set(statement.containers)):
                raise ValueError("duplicate containers in cargo statement")
            if statement.scope == "goods_total" and len(statement.products) != 1:
                raise ValueError("goods_total requires exactly one product")
        return self


class CargoRelationFinding(ReviewFinding):
    """Cargo accounting defects: owners, counts, measures and placements."""

    field: Literal[
        "goodsItemDetails.grouping",
        "goodsItemDetails.numberAndTypeOfPackages",
        "goodsItemDetails.splitGoodsPlacement",
        "goodsItemDetails.grossWeight",
        "goodsItemDetails.netWeight",
        "goodsItemDetails.volume",
        "cargoSourceMap",
    ] = Field(
        description=(
            "Exact responsibility affected. Identify the product/container in the "
            "explanation. All row-owned quantities/measures belong here; no markings, "
            "description wording, handling or unrelated field edits."
        )
    )


class CargoRelationReview(SectionReview):
    """Narrow comparison of candidate grouping/packages/placements to source accounting."""

    findings: list[CargoRelationFinding] = Field(
        description="Relationship/package defects only; no unrelated cargo findings."
    )


class CargoRelationResponse(LabelSchemaModel):
    """Relationship review with the complete PDF already supplied."""

    response: CargoRelationReview = Field(
        description="Scoped review; complete PDF is available, so no page request is needed."
    )
    mapCorrection: CargoSourceMap | None = Field(
        default=None,
        description=(
            "Complete corrected source map only when the supplied working map is wrong; "
            "otherwise null. Resolve source ownership/numeric interpretation independently "
            "of candidate values. Preserve all supported portions and printed totals."
        ),
    )


class CargoFactFinding(ReviewFinding):
    """Cargo fact review, excluding the relationship reviewer's responsibilities."""

    field: Literal[
        "description",
        "marksAndNumbers",
        "hsCodes",
        "dangerousGoods",
        "handlingInstructions",
        "origin",
    ] = Field(
        description=(
            "Cargo fact field; identify the goods item in the explanation. Grouping, "
            "package counts, masses, volume and container placements have a dedicated review."
        )
    )


class CargoFactReview(SectionReview):
    """Review descriptions and cargo facts without issuing relationship edits."""

    findings: list[CargoFactFinding] = Field(
        description="Defects in cargo facts only; never package/allocation edits."
    )


def numeric_values(text: str) -> set[Decimal]:
    """Decimal/thousands alternatives, including explicit three-digit space groups.

    This is an absence screen, not a claim that every numeric interpretation has
    the right meaning. Arbitrarily spaced counts are never concatenated.
    """
    values: set[Decimal] = set()
    tokens = re.findall(r"(?<!\d)\d+(?:[.,]\d+)*(?!\d)", text)
    tokens.extend(
        re.sub(r"[ \u00a0\u202f]", "", token)
        for token in re.findall(r"(?<!\d)\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?(?!\d)", text)
    )
    for token in tokens:
        values.add(Decimal(token.replace(",", "").replace(".", "")))
        last = max(token.rfind(","), token.rfind("."))
        if last >= 0:
            values.add(Decimal(re.sub(r"[.,]", "", token[:last]) + "." + token[last + 1 :]))
    return values


def identifier_present(value: str, ocr: str) -> bool:
    """Absence screen allowing OCR presentation separators and joined identifier rows."""
    chars = [re.escape(c) for c in value if c.isalnum()]
    return bool(chars) and re.search(r"[\W_]*".join(chars), ocr, re.IGNORECASE) is not None


def measure_interpretations(token: str) -> set[Decimal]:
    """Plausible numeric readings of one token; context must resolve real ambiguity."""

    def integer(part: str) -> str | None:
        if part.isascii() and part.isdigit():
            return part
        if re.fullmatch(r"\d{1,3}([., \u00a0\u202f])\d{3}(?:\1\d{3})*", part):
            return re.sub(r"\D", "", part)
        return None

    grouped = integer(token)
    values = {Decimal(grouped)} if grouped is not None else set()
    for separator in (".", ","):
        left, found, right = token.rpartition(separator)
        whole = integer(left)
        if found and separator not in left and whole is not None and right.isdigit():
            values.add(Decimal(whole + "." + right))
    return values


def map_grounding_errors(source: CargoSourceMap, ocr: str) -> list[str]:
    numbers = numeric_values(ocr)
    errors = []
    for i, statement in enumerate(source.statements):
        for identifier in statement.containers:
            if not identifier_present(identifier, ocr):
                errors.append(f"statement {i}: container {identifier!r} absent from OCR")
        amounts = [p.quantity for p in statement.packages if p.quantity is not None]
        for measure in (statement.grossWeight, statement.netWeight, statement.volume):
            if measure is None:
                continue
            if not re.fullmatch(r"\d+(?:[., \u00a0\u202f]\d+)*", measure.printedValue):
                errors.append(f"statement {i}: measure must quote one numeric token")
            elif (
                re.search(r"(?<![\d.,])" + re.escape(measure.printedValue) + r"(?![\d.,])", ocr)
                is None
            ):
                errors.append(
                    f"statement {i}: numeric token {measure.printedValue!r} absent from OCR"
                )
            elif Decimal(str(measure.value)) not in measure_interpretations(measure.printedValue):
                errors.append(f"statement {i}: value does not interpret its own printed token")
        for value in amounts:
            if Decimal(str(value)) not in numbers:
                errors.append(f"statement {i}: printed amount {value} absent from OCR")
    return errors


def candidate_cargo_view(target: dict[str, Any]) -> dict[str, Any]:
    patch = target["documentPatch"]
    return {
        "containers": [
            c.get("equipmentIdentifier") for c in patch.get("containerInformation") or []
        ],
        "goods": [
            {k: g[k] for k in CARGO_ACCOUNTING_FIELDS if k in g}
            for g in patch.get("goodsItemDetails") or []
        ],
    }


def derived_amounts(source: CargoSourceMap | None, description: str, field: str) -> set[Decimal]:
    """Allowed exact totals from candidate-blind, singly owned source portions.

    This is an arithmetic guard, not semantic certification. A matching reviewer
    must still establish completeness and identity; unmatched identities get no
    arithmetic exemption rather than borrowing another product's quantities.
    """
    if source is None:
        return set()

    def normalized(s: str) -> str:
        folded = unicodedata.normalize("NFKD", s.casefold())
        letters = "".join(c for c in folded if not unicodedata.combining(c))
        return " ".join(re.findall(r"\w+", letters))

    keys = {p.key for p in source.products if normalized(p.description) == normalized(description)}
    if len(keys) != 1:
        return set()
    # An unsplit shared portion makes a product-wide total incomplete even when
    # its other, singly owned portions are fully quantified.
    if any(
        (s.scope == "shared_total" and bool(set(s.products) & keys))
        or (
            s.scope == "portion"
            and (not s.products or (bool(set(s.products) & keys) and set(s.products) != keys))
        )
        for s in source.statements
    ):
        return set()
    portions = [s for s in source.statements if s.scope == "portion" and set(s.products) == keys]
    if not portions:
        return set()
    if field == "packageQuantity":
        counts = [p for s in portions for p in s.packages if p.level == "target"]
        if len(counts) != len(portions) or any(p.quantity is None for p in counts):
            return set()
        if len({normalized(p.packageType) for p in counts}) != 1:
            return set()
        return {sum((Decimal(str(p.quantity)) for p in counts), Decimal(0))}
    measures = [getattr(s, field) for s in portions]
    if any(m is None for m in measures) or len({m.unit.upper() for m in measures}) != 1:
        return set()
    return {sum((Decimal(str(m.value)) for m in measures), Decimal(0))}


def cargo_numeric_findings(
    target: dict[str, Any], ocr: str, source: CargoSourceMap | None
) -> list[ReviewFinding]:
    """Reject absent cargo amounts unless supported by a source-map exact sum."""
    numbers = numeric_values(ocr)
    findings = []
    for i, goods in enumerate(target["documentPatch"].get("goodsItemDetails") or []):
        amounts = []
        for field in ("grossWeight", "netWeight", "volume"):
            if goods.get(field):
                amounts.append((field, goods[field]["value"], True))
        amounts.extend(
            ("packageQuantity", p["packageQuantity"], True)
            for p in goods.get("numberAndTypeOfPackages") or []
            if "packageQuantity" in p and p["packageQuantity"] is not None
        )
        amounts.extend(
            ("packageQuantity", p["packageQuantity"], False)
            for p in goods.get("splitGoodsPlacement") or []
            if "packageQuantity" in p and p["packageQuantity"] is not None
        )
        for field, value, total in amounts:
            allowed = (
                derived_amounts(source, goods.get("description", ""), field) if total else set()
            )
            if Decimal(str(value)) not in numbers | allowed:
                findings.append(
                    ReviewFinding(
                        field=f"goodsItemDetails[{i}].{field}",
                        issue="unsupported",
                        explanation=(
                            f"Amount {value} is absent from OCR and lacks a matching "
                            "exact source-portion sum. PDF amounts cannot supply labels."
                        ),
                        suggestedCorrection=(
                            "Retain OCR-supported facts; omit this unsupported "
                            "amount or establish its complete same-level OCR "
                            "derivation. Do not infer a local allocation from "
                            "a shipment total."
                        ),
                    )
                )
    return findings
