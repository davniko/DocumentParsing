"""Provider-constrained probes for target-facing B/L cargo language.

Each case is one independent request.  The model generates only text leaves
whose task labels preserve printed wording: goods descriptions, additional
cargo information, substantive marks, and handling instructions.  Package,
container, and HS-code printed surfaces are deliberately deferred to the final
raw-text realization stage because their task targets are semantic values.
"""

from __future__ import annotations

import asyncio
import json
import re
import resource
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator
from pydantic_ai import Agent, NativeOutput, capture_run_messages
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import SynthesisCargoLanguageProbeConfig
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    load_openai_key,
    model_messages,
    openai_responses_settings,
    usage_receipt,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.semantic_completion_pipeline import SemanticCompletionPlanRow
from document_ocr.training.config import resolve_config_path

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_GENERIC_MARKS_NORMALIZED = frozenset(
    {"NM", "NIL", "NONE", "NOMARKS", "NOMARKSANDNUMBERS", "UNMARKED"}
)
_TEXTUAL_PLACEHOLDERS = frozenset(
    {
        "n/a",
        "na",
        "none",
        "not available",
        "null",
        "tba",
        "tbd",
        "unavailable",
        "unknown",
    }
)
_TOKEN = re.compile(r"[A-Z0-9]+")


def _is_textual_placeholder(value: str) -> bool:
    """Recognize a placeholder even when a model retains source punctuation around it."""

    return value.strip().strip(":;,.()[]{}<>").strip().casefold() in _TEXTUAL_PLACEHOLDERS


_SEMANTIC_STOPWORDS = frozenset(
    {
        "AND",
        "FOR",
        "FROM",
        "OTHER",
        "THAN",
        "THE",
        "THIS",
        "WITH",
        "WITHOUT",
    }
)
_AUXILIARY_FACT_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("gross_weight", re.compile(r"\bGROSS\s+WEIGHT\b", re.I)),
    ("net_weight", re.compile(r"\bNET\s+WEIGHT\b", re.I)),
    ("volume", re.compile(r"\b(?:VOLUME|CUBIC\s+MET(?:ER|RE)S?)\b", re.I)),
    (
        "temperature",
        re.compile(
            r"\b(?:TEMPERATURE|SET\s*POINT|DEGREES?\s+C(?:ELSIUS)?)\b"
            r"|[+-]?\d+(?:\.\d+)?\s*°\s*C\b",
            re.I,
        ),
    ),
    (
        "hs_code",
        re.compile(r"\b(?:H\.?S\.?|HARMONI[ZS]ED)\s*(?:CODE|NO\.?|NUMBER)?\b", re.I),
    ),
    (
        "container",
        re.compile(
            r"\b(?:CONTAINER(?:\s+(?:NO\.?|NUMBER|TYPE))?|"
            r"EQUIPMENT\s+(?:NO\.?|NUMBER|TYPE))\b",
            re.I,
        ),
    ),
    ("free_days", re.compile(r"\bFREE\s+DAYS?\b", re.I)),
    ("invoice", re.compile(r"\bINVOICE\s*(?:NO\.?|NUMBER)?\b", re.I)),
    (
        "booking",
        re.compile(r"\bBOOKING\s*(?:NO\.?|NUMBER|REF(?:ERENCE)?)?\b", re.I),
    ),
    (
        "order",
        re.compile(
            r"\b(?:PURCHASE\s+)?ORDER\s*(?:NO\.?|NUMBER|REF(?:ERENCE)?)?\b",
            re.I,
        ),
    ),
    ("lot", re.compile(r"\b(?:LOT|BATCH)\s*(?:NO\.?|NUMBER)?\b", re.I)),
    ("origin", re.compile(r"\b(?:COUNTRY\s+OF\s+)?ORIGIN\b", re.I)),
)
_PACKAGE_WORD = re.compile(
    r"\b(?:BAGS?|BALES?|BARRELS?|BASKETS?|BINS?|BOTTLES?|BOX(?:ES)?|BUNDLES?|"
    r"CANS?|CARTONS?|CASES?|COILS?|CRATES?|CYLINDERS?|DRUMS?|IBCS?|PACKAGES?|"
    r"PALLETS?|PIECES?|PLTS?|CTNS?|PCS?|ROLLS?|SACKS?|UNITS?)\b",
    re.I,
)
_PACKAGE_ABBREVIATION = re.compile(r"(?<![A-Z])(?:PLTS?|CTNS?|PCS?)(?![A-Z])", re.I)
_INTERNAL_PACKAGE_NAMESPACE_PROSE = re.compile(
    r"\bPACKAGE[ \t]+(?=(?:BAGS?|BALES?|BARRELS?|BASKETS?|BINS?|BOTTLES?|"
    r"BOX(?:ES)?|BUNDLES?|CANS?|CARTONS?|CASES?|COILS?|CRATES?|CYLINDERS?|"
    r"DRUMS?|IBCS?|PACKAGES?|PALLETS?|PIECES?|ROLLS?|SACKS?|SKIDS?|TANKS?|"
    r"UNITS?|VEHICLES?)\b)",
    re.I,
)
_PACKAGE_QUANTITY = re.compile(
    rf"(?<![A-Z0-9])(?:[0-9][0-9,.' ]*|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|"
    rf"EIGHT|NINE|TEN|HUNDRED|THOUSAND)(?:\s+|(?=[A-Z])){_PACKAGE_WORD.pattern}",
    re.I,
)
_PACKING_METHOD = re.compile(
    r"\b(?:PACK(?:ED|ING)?|PALLETI[ZS]ED|WRAPP?ED|STRAPPED|BANDING|HEAT\s+TREATED|"
    r"SECURED|WOOD(?:EN)?\s+PACKAG(?:E|ING)|SHRINK\s+WRAP)\b",
    re.I,
)
_TRANSIT_OR_WAREHOUSE = re.compile(r"\b(?:IN\s+TRANSIT|TRANSIT\s+TO|BONDED\s+WAREHOUSE)\b", re.I)
_CONSOLIDATION = re.compile(r"\bCONSOLIDAT(?:ED|ION)\b", re.I)
_DANGEROUS_STATUS = re.compile(
    r"\b(?:NON[ -]?HAZARDOUS|HAZARDOUS|DANGEROUS\s+GOODS?|IMDG|UNDG|UN\s*[0-9]{4})\b",
    re.I,
)
_PURPOSE_OR_END_USE = re.compile(r"\b(?:PURPOSE|END\s+USE|PROJECT)\s*:", re.I)
_COMMERCIAL_IDENTIFIER = re.compile(
    r"^(?:(?=[A-Z0-9._/#-]{5,}$)(?=[A-Z0-9._/#-]*[0-9])[A-Z0-9._/#-]+|"
    r"(?:MATERIAL|BATCH|LOT|RMS|GRADE|TYPE)\s*(?:NO\.?|NUMBER)?\s*[:#-]?\s*"
    r"[A-Z0-9._/#-]{2,})$",
    re.I,
)
_WRITTEN_PACKAGE_HIERARCHY = re.compile(r"\b(?:CONTAIN(?:ING|S)?|EACH|PER)\b|=|/", re.I)

type CargoAuxiliaryRole = Literal[
    "package_hierarchy_or_quantity",
    "packing_method_or_per_unit_measure",
    "measurement_statement",
    "consolidation_status",
    "transit_or_bonded_movement",
    "dangerous_goods_status",
    "purpose_or_end_use",
    "commercial_or_product_identifier",
    "product_attribute_or_condition",
]


def _cargo_auxiliary_role(value: str) -> CargoAuxiliaryRole:
    """Classify an occupied source slot by explicit printed syntax.

    This is deliberately a role grammar rather than a surface-to-value mapping.  It never
    selects a synthetic fact.  The classification only tells the generator and validator which
    *kind* of source slot must be preserved.  The broad final category keeps genuinely lexical
    product attributes inside the model-owned stage without pretending they are structured data.
    """

    if _TRANSIT_OR_WAREHOUSE.search(value):
        return "transit_or_bonded_movement"
    if _CONSOLIDATION.search(value):
        return "consolidation_status"
    if _DANGEROUS_STATUS.search(value):
        return "dangerous_goods_status"
    if _PURPOSE_OR_END_USE.search(value):
        return "purpose_or_end_use"
    package_terms = len(_PACKAGE_WORD.findall(value)) + len(_PACKAGE_ABBREVIATION.findall(value))
    if package_terms and (
        package_terms > 1
        or _WRITTEN_PACKAGE_HIERARCHY.search(value)
        or len(_PACKAGE_QUANTITY.findall(value)) > 1
    ):
        return "package_hierarchy_or_quantity"
    if package_terms or _PACKING_METHOD.search(value):
        return "packing_method_or_per_unit_measure"
    fact_kinds = _auxiliary_fact_kinds(value)
    if fact_kinds & {"gross_weight", "net_weight", "volume", "temperature"}:
        return "measurement_statement"
    if _COMMERCIAL_IDENTIFIER.fullmatch(value.strip()):
        return "commercial_or_product_identifier"
    return "product_attribute_or_condition"


class CargoGoodsIdentitySeed(BaseModel):
    model_config = _STRICT

    description: NonEmptyText
    hsCode: Annotated[str, StringConstraints(pattern=r"^[0-9]{6,18}$")] | None
    chapterDescription: NonEmptyText | None
    headingDescription: NonEmptyText | None
    thermalProfile: Literal["FROZEN", "CHILLED"] | None


class CargoDangerousGoodsSeed(BaseModel):
    model_config = _STRICT

    properShippingName: NonEmptyText
    unNumber: Annotated[str, StringConstraints(pattern=r"^[0-9]{4}$")]
    hazardCategory: NonEmptyText
    exactHazardClass: NonEmptyText
    subsidiaryHazardCategories: tuple[NonEmptyText, ...]
    packingGroupCategory: NonEmptyText | None
    flashpointCelsius: float | None


class CargoPackageFact(BaseModel):
    model_config = _STRICT

    quantity: Annotated[int, Field(ge=0)]
    typeCategory: NonEmptyText


class CargoEquipmentFact(BaseModel):
    model_config = _STRICT

    sizeCategory: NonEmptyText
    typeCategory: NonEmptyText
    temperatureSetpointCelsius: float | None


class CargoMeasurementFact(BaseModel):
    model_config = _STRICT

    value: float
    unit: NonEmptyText


class CargoStructuredFacts(BaseModel):
    model_config = _STRICT

    packages: tuple[CargoPackageFact, ...]
    equipment: tuple[CargoEquipmentFact, ...]
    cargoOrigin: NonEmptyText | None
    grossWeight: CargoMeasurementFact | None
    netWeight: CargoMeasurementFact | None
    volume: CargoMeasurementFact | None


class CargoTextSlot(BaseModel):
    model_config = _STRICT

    sourceStyleReference: NonEmptyText
    semanticRole: CargoAuxiliaryRole
    sourceFactKinds: tuple[NonEmptyText, ...]

    @model_validator(mode="after")
    def role_matches_source_syntax(self) -> CargoTextSlot:
        expected_role = _cargo_auxiliary_role(self.sourceStyleReference)
        if self.semanticRole != expected_role:
            raise ValueError("cargo auxiliary semantic role differs from source syntax")
        expected_kinds = tuple(sorted(_auxiliary_fact_kinds(self.sourceStyleReference)))
        if self.sourceFactKinds != expected_kinds:
            raise ValueError("cargo auxiliary fact kinds differ from source syntax")
        return self


class CargoStyleSlot(BaseModel):
    model_config = _STRICT

    sourceStyleReference: NonEmptyText


class CargoMarksSlot(BaseModel):
    model_config = _STRICT

    action: Literal["generate", "preserve_literal"]
    sourceStyleReference: NonEmptyText


class CargoLanguageFieldContract(BaseModel):
    model_config = _STRICT

    descriptionPresent: bool
    sourceDescriptionStyleReference: NonEmptyText | None
    additionalInformationSlots: tuple[CargoTextSlot, ...]
    marksAndNumbersSlots: tuple[CargoMarksSlot, ...]
    handlingInstructionSlots: tuple[CargoStyleSlot, ...]

    @model_validator(mode="after")
    def description_presence_matches_reference(self) -> CargoLanguageFieldContract:
        if self.descriptionPresent != (self.sourceDescriptionStyleReference is not None):
            raise ValueError("description presence differs from its source style reference")
        return self


class CargoLanguageGroupSeed(BaseModel):
    model_config = _STRICT

    groupId: NonEmptyText
    goodsIdentities: tuple[CargoGoodsIdentitySeed, ...]
    dangerousGoods: tuple[CargoDangerousGoodsSeed, ...]
    structuredFacts: CargoStructuredFacts
    fieldContract: CargoLanguageFieldContract

    @model_validator(mode="after")
    def semantic_identity_is_present(self) -> CargoLanguageGroupSeed:
        if not self.goodsIdentities and not self.dangerousGoods:
            raise ValueError("cargo-language groups require a goods or dangerous-goods identity")
        return self


class CargoLanguageRouteContext(BaseModel):
    model_config = _STRICT

    portOfLoading: NonEmptyText | None
    portOfDischarge: NonEmptyText | None
    placeOfDelivery: NonEmptyText | None


class CargoLanguageGenerationSeed(BaseModel):
    model_config = _STRICT

    caseId: NonEmptyText
    sourceDocumentId: Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
    scenarioId: NonEmptyText
    routeContext: CargoLanguageRouteContext
    cargoGroups: tuple[CargoLanguageGroupSeed, ...] = Field(min_length=1, max_length=24)
    excludedFinalPatchingFields: tuple[
        Literal[
            "packagePrintedSurfaces",
            "containerPrintedSurfaces",
            "hsCodePrintedSurfaces",
        ],
        ...,
    ]


class GeneratedCargoLanguageGroup(BaseModel):
    model_config = _STRICT

    groupId: Annotated[NonEmptyText, Field(description="Exact input cargo group identifier.")]
    description: Annotated[
        NonEmptyText | None,
        Field(
            description=(
                "Natural goods description that names the supplied synthetic goods identity, "
                "or null only when the input field contract says the leaf is absent."
            )
        ),
    ]
    additionalInformation: Annotated[
        tuple[NonEmptyText, ...],
        Field(
            max_length=32,
            description=(
                "One non-placeholder result per occupied auxiliary source slot, in order. Use "
                "supplied target facts when the slot states a task fact; otherwise generate "
                "compatible fictional auxiliary flavor with the same semantic role."
            ),
        ),
    ]
    marksAndNumbers: Annotated[
        tuple[NonEmptyText, ...],
        Field(
            max_length=64,
            description=(
                "Marks in input-slot order. Preserve literal slots verbatim and synthesize "
                "novel realistic values for generate slots."
            ),
        ),
    ]
    handlingInstructions: Annotated[
        tuple[NonEmptyText, ...],
        Field(
            max_length=16,
            description=(
                "Cargo-coherent handling statements in input-slot order and exact requested "
                "count; empty when no slots were supplied."
            ),
        ),
    ]

    @model_validator(mode="after")
    def values_are_single_line_and_unique(self) -> GeneratedCargoLanguageGroup:
        values = (
            *((self.description,) if self.description is not None else ()),
            *self.additionalInformation,
            *self.marksAndNumbers,
            *self.handlingInstructions,
        )
        if any("\n" in value or "\r" in value for value in values):
            raise ValueError("cargo-language values must be single-line semantic values")
        linguistic_values = (
            *((self.description,) if self.description is not None else ()),
            *self.additionalInformation,
            *self.handlingInstructions,
        )
        if any(_is_textual_placeholder(value) for value in linguistic_values):
            raise ValueError(
                "cargo-language fields must use grounded text or schema null, never a textual "
                "placeholder"
            )
        for label, rows in (
            ("additionalInformation", self.additionalInformation),
            ("marksAndNumbers", self.marksAndNumbers),
            ("handlingInstructions", self.handlingInstructions),
        ):
            if len(rows) != len(set(rows)):
                raise ValueError(f"{label} values must be distinct")
        return self


class CargoLanguageGenerationOutput(BaseModel):
    model_config = _STRICT

    cargoGroups: tuple[GeneratedCargoLanguageGroup, ...] = Field(min_length=1, max_length=24)


class CargoLanguageValidation(BaseModel):
    model_config = _STRICT

    passed: bool
    checks: dict[NonEmptyText, bool]

    @model_validator(mode="after")
    def aggregate_matches(self) -> CargoLanguageValidation:
        if not self.checks or self.passed != all(self.checks.values()):
            raise ValueError("cargo-language validation aggregate differs from its checks")
        return self


class CargoLanguageCaseRecord(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[1]
    caseIndex: Annotated[int, Field(ge=0)]
    seed: CargoLanguageGenerationSeed
    userPromptSha256: Sha256
    outputSchemaSha256: Sha256
    providerNativeStrictJsonSchema: Literal[True]
    automaticRepairRequests: Literal[0]
    startedAt: datetime
    completedAt: datetime
    durationMs: Annotated[float, Field(ge=0)]
    status: Literal["success", "validation_failed", "call_failed"]
    output: CargoLanguageGenerationOutput | None
    validation: CargoLanguageValidation | None
    usage: LinguisticUsageReceipt
    errorType: NonEmptyText | None
    errorMessage: NonEmptyText | None

    @model_validator(mode="after")
    def status_payloads_match(self) -> CargoLanguageCaseRecord:
        if self.completedAt < self.startedAt:
            raise ValueError("cargo-language case completion precedes start")
        successful_call = self.status != "call_failed"
        if successful_call != (self.output is not None and self.validation is not None):
            raise ValueError("cargo-language output/validation do not match call status")
        if self.status == "success" and not cast(CargoLanguageValidation, self.validation).passed:
            raise ValueError("successful cargo-language case did not pass validation")
        if (
            self.status == "validation_failed"
            and cast(CargoLanguageValidation, self.validation).passed
        ):
            raise ValueError("validation-failed cargo-language case has passing checks")
        if (self.status == "call_failed") != (self.errorType is not None):
            raise ValueError("cargo-language error fields do not match call status")
        return self


def _resolve_pinned_file(
    project_root: Path, configured_path: str, expected_sha256: str, *, label: str
) -> Path:
    path = resolve_config_path(project_root, configured_path)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} differs from its configured pin: {path}")
    return path.resolve(strict=True)


def _validate_completion_run(project_root: Path, config: SynthesisCargoLanguageProbeConfig) -> Path:
    configured = config.inputs.semantic_completion_run
    root = resolve_config_path(project_root, configured.path)
    commit = root / "_COMMIT.json"
    if root.is_symlink() or not root.is_dir() or sha256_file(commit) != configured.commit_sha256:
        raise ValueError(f"semantic completion run differs from its configured pin: {root}")
    StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=configured.transaction_sha256,
    ).validate_committed_run()
    return root.resolve(strict=True)


def _load_completion_plans(path: Path, *, records: int) -> dict[str, SemanticCompletionPlanRow]:
    output: dict[str, SemanticCompletionPlanRow] = {}
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip() or not raw.endswith(b"\n"):
                raise ValueError(f"completion plan row {line_number} is blank or unterminated")
            try:
                row = SemanticCompletionPlanRow.model_validate_json(raw, strict=True)
            except ValueError as error:
                raise ValueError(f"completion plan row {line_number} is invalid") from error
            if row.base_document_id in output:
                raise ValueError(f"completion plans repeat {row.base_document_id}")
            output[row.base_document_id] = row
    if len(output) != records:
        raise ValueError("completion plan count differs from configuration")
    return output


def _generic_mark(value: str) -> bool:
    normalized = "".join(_TOKEN.findall(value.upper()))
    return normalized in _GENERIC_MARKS_NORMALIZED


def _text_values(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(row, str) and row.strip() for row in value
    ):
        raise ValueError(f"{label} must be a list of non-empty text values")
    return tuple(value)


def _measurement(value: Any) -> CargoMeasurementFact | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("cargo measurement must be an object")
    number, unit = value.get("value"), value.get("unit")
    if (
        not isinstance(number, (int, float))
        or isinstance(number, bool)
        or not isinstance(unit, str)
    ):
        raise ValueError("cargo measurement has invalid value or unit")
    return CargoMeasurementFact(value=float(number), unit=unit)


def _route_name(patch: Mapping[str, Any], role: str) -> str | None:
    route = patch.get("route") or {}
    if not isinstance(route, Mapping):
        raise ValueError("cargo-language route must be an object")
    value = route.get(role)
    if value is None:
        return None
    if not isinstance(value, Mapping) or not isinstance(value.get("name"), str):
        raise ValueError(f"cargo-language route role has no name: {role}")
    return cast(str, value["name"])


def build_cargo_language_seed(
    *, case_index: int, plan: SemanticCompletionPlanRow
) -> CargoLanguageGenerationSeed:
    patch = cast(Mapping[str, Any], plan.target["documentPatch"])
    group_rows = patch.get("cargoGroups") or []
    package_rows = patch.get("cargoPackages") or []
    allocation_rows = patch.get("cargoAllocationGroups") or []
    if not isinstance(group_rows, list) or not isinstance(package_rows, list):
        raise ValueError("cargo-language target cargo collections are invalid")
    equipment_by_number = {row.container_number: row for row in plan.equipment_realizations}
    allocated_by_group: dict[str, tuple[str, ...]] = {}
    for row in allocation_rows:
        if not isinstance(row, Mapping):
            raise ValueError("cargo allocation row is not an object")
        allocated_by_group[cast(str, row["groupId"])] = tuple(
            cast(str, allocation["containerNumber"])
            for allocation in cast(Sequence[Mapping[str, Any]], row["allocations"])
        )
    flashpoints = {
        (row.cargo_group_id, row.dangerous_goods_order): row.value_celsius
        for row in plan.flashpoint_realizations
        if row.generated
    }
    dangerous_by_group: dict[str, list[CargoDangerousGoodsSeed]] = {}
    for dangerous_value in plan.upstream_dangerous_goods_realizations:
        group_id = cast(str, dangerous_value["cargo_group_id"])
        order = cast(int, dangerous_value["dangerous_goods_order"])
        dangerous_by_group.setdefault(group_id, []).append(
            CargoDangerousGoodsSeed(
                properShippingName=cast(str, dangerous_value["proper_shipping_name"]),
                unNumber=cast(str, dangerous_value["un_number"]),
                hazardCategory=cast(str, dangerous_value["semantic_hazard_category"]),
                exactHazardClass=cast(str, dangerous_value["exact_hazard_class"]),
                subsidiaryHazardCategories=tuple(
                    cast(
                        Sequence[str],
                        dangerous_value["semantic_subsidiary_hazard_categories"],
                    )
                ),
                packingGroupCategory=cast(
                    str | None, dangerous_value.get("packing_group_category")
                ),
                flashpointCelsius=flashpoints.get((group_id, order)),
            )
        )
    identities_by_group: dict[str, list[CargoGoodsIdentitySeed]] = {}
    for realization in plan.cargo_realizations:
        identities_by_group.setdefault(realization.cargo_group_id, []).append(
            CargoGoodsIdentitySeed(
                description=realization.description,
                hsCode=realization.output_hs_code or realization.hs6,
                chapterDescription=realization.chapter_description,
                headingDescription=realization.heading_description,
                thermalProfile=realization.thermal_profile,
            )
        )
    groups: list[CargoLanguageGroupSeed] = []
    for raw_group in group_rows:
        if not isinstance(raw_group, Mapping):
            raise ValueError("cargo-language group row is not an object")
        group_id = cast(str, raw_group["groupId"])
        source_description = raw_group.get("description")
        if source_description is not None and not isinstance(source_description, str):
            raise ValueError("cargo description style reference is not text")
        additional = _text_values(
            raw_group.get("additionalInformation"), label="additionalInformation"
        )
        marks = _text_values(raw_group.get("marksAndNumbers"), label="marksAndNumbers")
        handling = _text_values(raw_group.get("handlingInstructions"), label="handlingInstructions")
        packages = tuple(
            CargoPackageFact(
                quantity=cast(int, row["quantity"]),
                typeCategory=cast(str, row["typeCategory"]),
            )
            for row in package_rows
            if isinstance(row, Mapping) and row.get("groupId") == group_id
        )
        equipment = tuple(
            CargoEquipmentFact(
                sizeCategory=equipment_by_number[number].size_category,
                typeCategory=equipment_by_number[number].type_category,
                temperatureSetpointCelsius=equipment_by_number[number].temperature_value_celsius,
            )
            for number in allocated_by_group.get(group_id, ())
            if number in equipment_by_number
        )
        origin = raw_group.get("origin")
        origin_name = (
            cast(str, origin["name"])
            if isinstance(origin, Mapping) and isinstance(origin.get("name"), str)
            else None
        )
        groups.append(
            CargoLanguageGroupSeed(
                groupId=group_id,
                goodsIdentities=tuple(identities_by_group.get(group_id, ())),
                dangerousGoods=tuple(dangerous_by_group.get(group_id, ())),
                structuredFacts=CargoStructuredFacts(
                    packages=packages,
                    equipment=equipment,
                    cargoOrigin=origin_name,
                    grossWeight=_measurement(raw_group.get("grossWeight")),
                    netWeight=_measurement(raw_group.get("netWeight")),
                    volume=_measurement(raw_group.get("volume")),
                ),
                fieldContract=CargoLanguageFieldContract(
                    descriptionPresent=source_description is not None,
                    sourceDescriptionStyleReference=source_description,
                    additionalInformationSlots=tuple(
                        CargoTextSlot(
                            sourceStyleReference=value,
                            semanticRole=_cargo_auxiliary_role(value),
                            sourceFactKinds=tuple(sorted(_auxiliary_fact_kinds(value))),
                        )
                        for value in additional
                    ),
                    marksAndNumbersSlots=tuple(
                        CargoMarksSlot(
                            action="preserve_literal" if _generic_mark(value) else "generate",
                            sourceStyleReference=value,
                        )
                        for value in marks
                    ),
                    handlingInstructionSlots=tuple(
                        CargoStyleSlot(sourceStyleReference=value) for value in handling
                    ),
                ),
            )
        )
    return CargoLanguageGenerationSeed(
        caseId=f"cargo-{case_index + 1:02d}",
        sourceDocumentId=plan.base_document_id,
        scenarioId=plan.scenario_id,
        routeContext=CargoLanguageRouteContext(
            portOfLoading=_route_name(patch, "portOfLoading"),
            portOfDischarge=_route_name(patch, "portOfDischarge"),
            placeOfDelivery=_route_name(patch, "placeOfDelivery"),
        ),
        cargoGroups=tuple(groups),
        excludedFinalPatchingFields=(
            "packagePrintedSurfaces",
            "containerPrintedSurfaces",
            "hsCodePrintedSurfaces",
        ),
    )


def _semantic_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _TOKEN.findall(value.upper())
        if len(token) >= 3 and token not in _SEMANTIC_STOPWORDS and not token.isdigit()
    )


def _casing_style(value: str) -> Literal["upper", "lower", "mixed", "uncased"]:
    letters = "".join(character for character in value if character.isalpha())
    if not letters:
        return "uncased"
    if letters.isupper():
        return "upper"
    if letters.islower():
        return "lower"
    return "mixed"


def _casing_style_compatible(*, source: str, generated: str) -> bool:
    """Preserve unambiguous all-upper/all-lower template casing.

    Mixed-case source text is semantic style evidence rather than a mechanically
    enforceable casing pattern: proper nouns and sentence starts can legitimately
    move uppercase characters when the synthetic goods identity changes.
    """

    source_style = _casing_style(source)
    return (
        source_style == "mixed"
        or source_style == "uncased"
        or (_casing_style(generated) == source_style)
    )


def _casing_from_source(*, source: str, generated: str) -> str:
    style = _casing_style(source)
    if style == "upper":
        return generated.upper()
    if style == "lower":
        return generated.lower()
    return generated


def normalize_cargo_language_output(
    *, seed: CargoLanguageGenerationSeed, output: CargoLanguageGenerationOutput
) -> CargoLanguageGenerationOutput:
    """Apply mechanical source casing before semantic validation.

    Casing is template formatting, not a linguistic decision. Normalizing it locally avoids a
    paid retry for a transformation the host can perform exactly; the captured provider messages
    still retain the original response for audit.
    """

    if len(seed.cargoGroups) != len(output.cargoGroups):
        return output
    groups: list[GeneratedCargoLanguageGroup] = []
    for expected, generated in zip(seed.cargoGroups, output.cargoGroups, strict=True):
        contract = expected.fieldContract
        description = generated.description
        if description is not None and contract.sourceDescriptionStyleReference is not None:
            description = _casing_from_source(
                source=contract.sourceDescriptionStyleReference,
                generated=description,
            )

        def normalized_rows(
            values: Sequence[str],
            slots: Sequence[CargoTextSlot | CargoStyleSlot | CargoMarksSlot],
        ) -> tuple[str, ...]:
            if len(values) != len(slots):
                return tuple(values)
            return tuple(
                _INTERNAL_PACKAGE_NAMESPACE_PROSE.sub(
                    "",
                    _casing_from_source(source=slot.sourceStyleReference, generated=value),
                )
                for slot, value in zip(slots, values, strict=True)
            )

        groups.append(
            generated.model_copy(
                update={
                    "description": description,
                    "additionalInformation": normalized_rows(
                        generated.additionalInformation,
                        contract.additionalInformationSlots,
                    ),
                    "marksAndNumbers": normalized_rows(
                        generated.marksAndNumbers,
                        contract.marksAndNumbersSlots,
                    ),
                    "handlingInstructions": normalized_rows(
                        generated.handlingInstructions,
                        contract.handlingInstructionSlots,
                    ),
                }
            )
        )
    return output.model_copy(update={"cargoGroups": tuple(groups)})


def _slot_casing_styles_compatible(
    *,
    generated: Sequence[str],
    source: Sequence[CargoTextSlot | CargoStyleSlot | CargoMarksSlot],
) -> bool:
    return len(generated) == len(source) and all(
        _casing_style_compatible(
            source=slot.sourceStyleReference,
            generated=value,
        )
        for slot, value in zip(source, generated, strict=True)
    )


def _auxiliary_fact_kinds(value: str) -> frozenset[str]:
    return frozenset(name for name, pattern in _AUXILIARY_FACT_MARKERS if pattern.search(value))


def _auxiliary_slot_scope_compatible(*, source: str, generated: str) -> bool:
    """Reject newly introduced, unrelated structured-fact classes in one slot."""

    return _auxiliary_fact_kinds(generated) <= _auxiliary_fact_kinds(source)


def _contains_integer_surface(value: str, expected: int) -> bool:
    compact = value.replace(",", "").replace(" ", "")
    return re.search(rf"(?<![0-9]){expected}(?![0-9])", compact) is not None


def _package_category_surface_present(value: str, category: str) -> bool:
    if not category.startswith("PACKAGE_"):
        raise ValueError(f"unsupported package category token: {category!r}")
    # MPCI category tokens follow the registry's noun-plus-qualifier order (for example,
    # ``PACKAGE_BOX_FIBREBOARD``), while natural printed English commonly reverses it
    # (``FIBREBOARD BOXES``).  Match every semantic token independently so word order is not
    # mistaken for category identity, while still requiring the full category rather than a
    # generic package noun.
    words = category.removeprefix("PACKAGE_").split("_")

    def variants(word: str) -> tuple[str, ...]:
        values = {word, f"{word}S"}
        if word.endswith("Y") and len(word) > 1:
            values.add(f"{word[:-1]}IES")
        if word.endswith(("S", "X", "Z", "CH", "SH")):
            values.add(f"{word}ES")
        return tuple(sorted(values))

    return all(
        any(re.search(rf"\b{re.escape(candidate)}\b", value, re.I) for candidate in variants(word))
        for word in words
    )


def _package_fact_present(value: str, fact: CargoPackageFact) -> bool:
    return _contains_integer_surface(value, fact.quantity) and _package_category_surface_present(
        value, fact.typeCategory
    )


def _semantic_identity_covered(
    value: str,
    *,
    goods: Sequence[CargoGoodsIdentitySeed],
    dangerous: Sequence[CargoDangerousGoodsSeed],
) -> bool:
    generated = _semantic_tokens(value)
    return all(
        bool(
            generated
            & (_semantic_tokens(row.description) | _semantic_tokens(row.headingDescription or ""))
        )
        for row in goods
    ) and all(bool(generated & _semantic_tokens(row.properShippingName)) for row in dangerous)


def _semantic_stems(value: str) -> frozenset[str]:
    return frozenset(token[:5] for token in _semantic_tokens(value) if len(token) >= 5)


def _semantic_identity_has_stem_overlap(
    value: str,
    *,
    goods: Sequence[CargoGoodsIdentitySeed],
    dangerous: Sequence[CargoDangerousGoodsSeed],
) -> bool:
    generated = _semantic_stems(value)
    return all(
        bool(
            generated
            & (
                _semantic_stems(row.description)
                | _semantic_stems(row.headingDescription or "")
                | _semantic_stems(row.chapterDescription or "")
            )
        )
        for row in goods
    ) and all(bool(generated & _semantic_stems(row.properShippingName)) for row in dangerous)


def _route_context_present(value: str, route: CargoLanguageRouteContext) -> bool:
    candidates = tuple(
        candidate
        for candidate in (route.portOfLoading, route.portOfDischarge, route.placeOfDelivery)
        if candidate is not None
    )
    return bool(candidates) and any(
        re.search(rf"(?<![A-Z0-9]){re.escape(candidate)}(?![A-Z0-9])", value, re.I) is not None
        for candidate in candidates
    )


def _measurement_value_present(value: str, measurement: CargoMeasurementFact | None) -> bool:
    if measurement is None:
        return False
    expected = Decimal(str(measurement.value))
    for surface in re.findall(r"(?<![0-9])[0-9]+(?:[.,][0-9]+)*(?![0-9])", value):
        try:
            normalized = Decimal(surface.replace(",", ""))
        except ArithmeticError:
            continue
        if normalized == expected:
            return True
    return False


def _additional_role_compatible(
    *,
    slot: CargoTextSlot,
    generated: str,
    expected: CargoLanguageGroupSeed,
    route: CargoLanguageRouteContext,
) -> bool:
    role = slot.semanticRole
    facts = expected.structuredFacts
    if role == "package_hierarchy_or_quantity":
        return bool(facts.packages) and all(
            _package_fact_present(generated, fact) for fact in facts.packages
        )
    if role == "packing_method_or_per_unit_measure":
        source_has_package = bool(
            _PACKAGE_WORD.search(slot.sourceStyleReference)
            or _PACKAGE_ABBREVIATION.search(slot.sourceStyleReference)
        )
        if not source_has_package:
            return _PACKING_METHOD.search(generated) is not None
        return bool(facts.packages) and any(
            _package_fact_present(generated, fact) for fact in facts.packages
        )
    if role == "measurement_statement":
        checks: list[bool] = []
        for fact_kind, measurement in (
            ("gross_weight", facts.grossWeight),
            ("net_weight", facts.netWeight),
            ("volume", facts.volume),
        ):
            if fact_kind in slot.sourceFactKinds:
                checks.append(_measurement_value_present(generated, measurement))
        if "temperature" in slot.sourceFactKinds:
            setpoints = tuple(
                row.temperatureSetpointCelsius
                for row in facts.equipment
                if row.temperatureSetpointCelsius is not None
            )
            checks.append(
                bool(setpoints)
                and any(
                    _measurement_value_present(
                        generated, CargoMeasurementFact(value=v, unit="celsius")
                    )
                    for v in setpoints
                )
            )
        return bool(checks) and all(checks)
    if role == "consolidation_status":
        return _CONSOLIDATION.search(generated) is not None
    if role == "transit_or_bonded_movement":
        return _TRANSIT_OR_WAREHOUSE.search(generated) is not None and _route_context_present(
            generated, route
        )
    if role == "dangerous_goods_status":
        if expected.dangerousGoods:
            return all(row.unNumber in generated for row in expected.dangerousGoods) and (
                _DANGEROUS_STATUS.search(generated) is not None
            )
        return re.search(r"\bNON[ -]?HAZARDOUS\b", generated, re.I) is not None
    if role == "purpose_or_end_use":
        return _PURPOSE_OR_END_USE.search(
            generated
        ) is not None and _semantic_identity_has_stem_overlap(
            generated, goods=expected.goodsIdentities, dangerous=expected.dangerousGoods
        )
    if role == "commercial_or_product_identifier":
        return (
            generated.casefold() != slot.sourceStyleReference.casefold()
            and _COMMERCIAL_IDENTIFIER.fullmatch(generated.strip()) is not None
            and _auxiliary_fact_kinds(generated) <= set(slot.sourceFactKinds)
        )
    return not _auxiliary_fact_kinds(generated)


def validate_cargo_language(
    *, seed: CargoLanguageGenerationSeed, output: CargoLanguageGenerationOutput
) -> CargoLanguageValidation:
    checks: dict[str, bool] = {
        "group_count_exact": len(output.cargoGroups) == len(seed.cargoGroups),
        "group_order_and_ids_exact": tuple(row.groupId for row in output.cargoGroups)
        == tuple(row.groupId for row in seed.cargoGroups),
    }
    if not checks["group_count_exact"]:
        return CargoLanguageValidation(passed=False, checks=checks)
    for group_index, (expected, generated) in enumerate(
        zip(seed.cargoGroups, output.cargoGroups, strict=True), start=1
    ):
        prefix = f"group_{group_index}"
        contract = expected.fieldContract
        checks[f"{prefix}_description_presence"] = contract.descriptionPresent == (
            generated.description is not None
        )
        checks[f"{prefix}_additional_count"] = len(generated.additionalInformation) == len(
            contract.additionalInformationSlots
        )
        checks[f"{prefix}_additional_values_present"] = all(
            not _is_textual_placeholder(value) for value in generated.additionalInformation
        )
        checks[f"{prefix}_additional_slot_scope"] = len(generated.additionalInformation) == len(
            contract.additionalInformationSlots
        ) and all(
            _auxiliary_slot_scope_compatible(
                source=slot.sourceStyleReference,
                generated=value,
            )
            for slot, value in zip(
                contract.additionalInformationSlots,
                generated.additionalInformation,
                strict=True,
            )
        )
        checks[f"{prefix}_additional_semantic_role"] = len(generated.additionalInformation) == len(
            contract.additionalInformationSlots
        ) and all(
            _additional_role_compatible(
                slot=slot,
                generated=value,
                expected=expected,
                route=seed.routeContext,
            )
            for slot, value in zip(
                contract.additionalInformationSlots,
                generated.additionalInformation,
                strict=True,
            )
        )
        checks[f"{prefix}_additional_source_values_anonymized"] = len(
            generated.additionalInformation
        ) == len(contract.additionalInformationSlots) and all(
            value.casefold() != slot.sourceStyleReference.casefold()
            for slot, value in zip(
                contract.additionalInformationSlots,
                generated.additionalInformation,
                strict=True,
            )
        )
        checks[f"{prefix}_marks_count"] = len(generated.marksAndNumbers) == len(
            contract.marksAndNumbersSlots
        )
        checks[f"{prefix}_handling_count"] = len(generated.handlingInstructions) == len(
            contract.handlingInstructionSlots
        )
        checks[f"{prefix}_description_casing_style"] = (
            generated.description is None
            or contract.sourceDescriptionStyleReference is None
            or _casing_style_compatible(
                source=contract.sourceDescriptionStyleReference,
                generated=generated.description,
            )
        )
        for field_name, generated_rows, source_rows in (
            (
                "additional",
                generated.additionalInformation,
                contract.additionalInformationSlots,
            ),
            ("marks", generated.marksAndNumbers, contract.marksAndNumbersSlots),
            (
                "handling",
                generated.handlingInstructions,
                contract.handlingInstructionSlots,
            ),
        ):
            checks[f"{prefix}_{field_name}_casing_style"] = _slot_casing_styles_compatible(
                generated=generated_rows,
                source=source_rows,
            )
        if len(generated.marksAndNumbers) == len(contract.marksAndNumbersSlots):
            checks[f"{prefix}_literal_marks_preserved"] = all(
                generated.marksAndNumbers[index] == slot.sourceStyleReference
                for index, slot in enumerate(contract.marksAndNumbersSlots)
                if slot.action == "preserve_literal"
            )
            checks[f"{prefix}_substantive_marks_anonymized"] = all(
                generated.marksAndNumbers[index].casefold() != slot.sourceStyleReference.casefold()
                for index, slot in enumerate(contract.marksAndNumbersSlots)
                if slot.action == "generate"
            )
        else:
            checks[f"{prefix}_literal_marks_preserved"] = False
            checks[f"{prefix}_substantive_marks_anonymized"] = False
        generated_values = (
            *((generated.description,) if generated.description else ()),
            *generated.additionalInformation,
            *generated.marksAndNumbers,
            *generated.handlingInstructions,
        )
        checks[f"{prefix}_no_internal_package_namespace_prose"] = not any(
            _INTERNAL_PACKAGE_NAMESPACE_PROSE.search(value) for value in generated_values
        )
        hs_codes = tuple(row.hsCode for row in expected.goodsIdentities if row.hsCode is not None)
        checks[f"{prefix}_hs_printed_surface_deferred"] = not any(
            code in value for code in hs_codes for value in generated_values
        )
        if generated.description is not None:
            generated_tokens = _semantic_tokens(generated.description)
            goods_covered = all(
                bool(
                    generated_tokens
                    & (
                        _semantic_tokens(row.description)
                        | _semantic_tokens(row.headingDescription or "")
                    )
                )
                for row in expected.goodsIdentities
            )
            dangerous_goods_covered = all(
                bool(_semantic_tokens(row.properShippingName) & generated_tokens)
                for row in expected.dangerousGoods
            )
            checks[f"{prefix}_description_semantic_coverage"] = (
                goods_covered and dangerous_goods_covered
            )
            thermal_profiles = {
                row.thermalProfile for row in expected.goodsIdentities if row.thermalProfile
            }
            checks[f"{prefix}_thermal_description_coherent"] = not thermal_profiles or all(
                profile in generated.description.upper() for profile in thermal_profiles
            )
        else:
            checks[f"{prefix}_description_semantic_coverage"] = True
            checks[f"{prefix}_thermal_description_coherent"] = True
        if generated.handlingInstructions and any(
            row.thermalProfile for row in expected.goodsIdentities
        ):
            handling_text = " ".join(generated.handlingInstructions).upper()
            checks[f"{prefix}_thermal_handling_coherent"] = any(
                token in handling_text
                for token in (
                    "FROZEN",
                    "CHILLED",
                    "COLD",
                    "REEFER",
                    "REFRIGERATED",
                    "TEMPERATURE",
                    "°C",
                )
            )
        else:
            checks[f"{prefix}_thermal_handling_coherent"] = True
    return CargoLanguageValidation(passed=all(checks.values()), checks=checks)


async def _run_case(
    *,
    index: int,
    seed: CargoLanguageGenerationSeed,
    agent: Agent[object, CargoLanguageGenerationOutput],
    config: SynthesisCargoLanguageProbeConfig,
    output_schema_sha256: str,
) -> tuple[CargoLanguageCaseRecord, JsonValue]:
    user_prompt = json.dumps(
        seed.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
    )
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    with capture_run_messages() as captured:
        try:
            result = await agent.run(
                user_prompt,
                usage_limits=UsageLimits(
                    request_limit=config.workflow.requests_per_case,
                    output_tokens_limit=config.provider.max_output_tokens,
                ),
            )
            output = CargoLanguageGenerationOutput.model_validate(
                result.output.model_dump(mode="python"), strict=True
            )
            output = normalize_cargo_language_output(seed=seed, output=output)
            validation = validate_cargo_language(seed=seed, output=output)
            responses = tuple(
                message for message in result.new_messages() if isinstance(message, ModelResponse)
            )
            if len(responses) != 1 or result.usage.requests != 1:
                raise RuntimeError(
                    "standalone cargo-language probe produced other than one response"
                )
            completed_at = datetime.now(UTC)
            return (
                CargoLanguageCaseRecord(
                    schemaVersion=1,
                    caseIndex=index,
                    seed=seed,
                    userPromptSha256=sha256_bytes(user_prompt.encode()),
                    outputSchemaSha256=output_schema_sha256,
                    providerNativeStrictJsonSchema=True,
                    automaticRepairRequests=0,
                    startedAt=started_at,
                    completedAt=completed_at,
                    durationMs=(time.perf_counter() - started) * 1000.0,
                    status="success" if validation.passed else "validation_failed",
                    output=output,
                    validation=validation,
                    usage=usage_receipt(responses, config.provider.pricing),
                    errorType=None,
                    errorMessage=None,
                ),
                model_messages(captured),
            )
        except Exception as error:
            responses = tuple(message for message in captured if isinstance(message, ModelResponse))
            completed_at = datetime.now(UTC)
            return (
                CargoLanguageCaseRecord(
                    schemaVersion=1,
                    caseIndex=index,
                    seed=seed,
                    userPromptSha256=sha256_bytes(user_prompt.encode()),
                    outputSchemaSha256=output_schema_sha256,
                    providerNativeStrictJsonSchema=True,
                    automaticRepairRequests=0,
                    startedAt=started_at,
                    completedAt=completed_at,
                    durationMs=(time.perf_counter() - started) * 1000.0,
                    status="call_failed",
                    output=None,
                    validation=None,
                    usage=usage_receipt(responses, config.provider.pricing),
                    errorType=type(error).__name__,
                    errorMessage=str(error),
                ),
                model_messages(captured),
            )


def _case_path(index: int) -> str:
    return f"generation/cases/{index + 1:02d}.json"


def _request_path(index: int) -> str:
    return f"generation/requests/{index + 1:02d}.json"


def _transcript_path(index: int) -> str:
    return f"generation/transcripts/{index + 1:02d}.json"


def _report(records: Sequence[CargoLanguageCaseRecord], summary: Mapping[str, Any]) -> str:
    lines = [
        "# Synthetic cargo-language probe",
        "",
        (
            f"{len(records)} independent first-pass GPT-5.6 Luna calls using "
            "provider-native strict JSON Schema. Each call jointly generated only task-facing "
            "printed cargo text leaves; package, container, and HS printed surfaces were excluded "
            "for final patching."
        ),
        "",
        "## Aggregate",
        "",
        f"- Cases: {summary['cases']}",
        f"- Schema-valid outputs: {summary['schemaValidOutputs']}",
        f"- Deterministic validation passes: {summary['validatedOutputs']}",
        f"- Requests: {summary['requests']}",
        f"- Input tokens: {summary['inputTokens']}",
        f"- Output tokens: {summary['outputTokens']}",
        f"- Reasoning tokens: {summary['reasoningTokens']}",
        f"- Estimated API cost: ${summary['estimatedCostUsd']}",
        f"- Wall time: {summary['wallSeconds']:.3f} s",
        "",
        "## Cases",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"### {record.caseIndex + 1}. {record.seed.sourceDocumentId}",
                "",
                f"- Scenario: `{record.seed.scenarioId}`",
                f"- Cargo groups: {len(record.seed.cargoGroups)}",
                f"- Status: `{record.status}`",
                f"- Latency: {record.durationMs / 1000.0:.3f} s",
                (
                    f"- Usage: {record.usage.inputTokens} input, "
                    f"{record.usage.outputTokens} output "
                    f"({record.usage.reasoningTokens} reasoning), "
                    f"${record.usage.estimatedCostUsd} estimated"
                ),
                "",
                "Input seed:",
                "",
                "```json",
                json.dumps(record.seed.model_dump(mode="json"), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        if record.output is None:
            lines.extend([f"Error: `{record.errorType}: {record.errorMessage}`", ""])
        else:
            lines.extend(
                [
                    "Output:",
                    "",
                    "```json",
                    json.dumps(record.output.model_dump(mode="json"), ensure_ascii=False, indent=2),
                    "```",
                    "",
                    "Validation:",
                    "",
                    "```json",
                    json.dumps(cast(CargoLanguageValidation, record.validation).checks, indent=2),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


async def _run_probe_async(
    *, project_root: Path, config_path: Path, config: SynthesisCargoLanguageProbeConfig
) -> dict[str, JsonValue]:
    completion_root = _validate_completion_run(project_root, config)
    plans_path = _resolve_pinned_file(
        project_root,
        config.inputs.completion_plans.path,
        config.inputs.completion_plans.sha256,
        label="completion plans",
    )
    if not plans_path.is_relative_to(completion_root):
        raise ValueError("completion plans are outside the pinned completion run")
    prompt_path = _resolve_pinned_file(
        project_root, config.prompt.path, config.prompt.sha256, label="cargo-language prompt"
    )
    plans = _load_completion_plans(plans_path, records=config.inputs.completion_plans.records)
    seeds = tuple(
        build_cargo_language_seed(case_index=index, plan=plans[case.document_id])
        for index, case in enumerate(config.cases)
    )
    prompt_bytes = read_regular_file_bytes(prompt_path)
    try:
        system_prompt = prompt_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("cargo-language prompt is not valid UTF-8") from error
    output_schema = CargoLanguageGenerationOutput.model_json_schema(mode="validation")
    output_schema_sha256 = sha256_bytes(canonical_json_bytes(output_schema))
    transaction = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "completionPlansSha256": config.inputs.completion_plans.sha256,
        "promptSha256": config.prompt.sha256,
        "outputSchemaSha256": output_schema_sha256,
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "cases": [seed.model_dump(mode="json") for seed in seeds],
        "runtime": {
            "model": config.provider.model,
            "reasoningEffort": config.provider.reasoning_effort,
            "pydanticAiVersion": version("pydantic-ai-slim"),
            "openaiVersion": version("openai"),
        },
    }
    transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    staged = StagedArtifactRun(
        output_parent=resolve_config_path(project_root, config.run.output_dir),
        run_name=config.run.run_id,
        transaction_sha256=transaction_sha256,
    )
    expected = [
        "REPORT.md",
        "config.yaml",
        "generation/results.jsonl",
        "generation/summary.json",
        "prompt.md",
        "schema/output.schema.json",
    ]
    for index in range(len(seeds)):
        expected.extend((_case_path(index), _request_path(index), _transcript_path(index)))
    expected = sorted(expected)
    if staged.completed:
        staged.commit(
            expected_artifacts=expected,
            metadata={"cases": len(seeds), "schema_version": 1},
        )
        return cast(
            dict[str, JsonValue],
            json.loads(read_regular_file_bytes(staged.final_root / "generation/summary.json")),
        )
    staged.recover_interrupted_temporary_files()
    staged.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    staged.publish_bytes("prompt.md", prompt_bytes)
    staged.publish_bytes("schema/output.schema.json", json_artifact_bytes(output_schema))
    for index, seed in enumerate(seeds):
        user_prompt = json.dumps(
            seed.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
        )
        staged.publish_json(
            _request_path(index),
            {
                "systemPromptSha256": config.prompt.sha256,
                "systemPrompt": system_prompt,
                "userPrompt": user_prompt,
                "seed": seed.model_dump(mode="json"),
            },
        )
    existing: dict[int, tuple[CargoLanguageCaseRecord, JsonValue]] = {}
    for index in range(len(seeds)):
        result_path = staged.stage_root / _case_path(index)
        transcript_path = staged.stage_root / _transcript_path(index)
        if result_path.exists() or transcript_path.exists():
            if not result_path.is_file() or not transcript_path.is_file():
                raise RuntimeError("partial cargo-language case artifact pair cannot be resumed")
            existing[index] = (
                CargoLanguageCaseRecord.model_validate_json(
                    read_regular_file_bytes(result_path), strict=True
                ),
                cast(JsonValue, json.loads(read_regular_file_bytes(transcript_path))),
            )
    client = AsyncOpenAI(
        api_key=load_openai_key(project_root, config.environment_file),
        max_retries=config.provider.transport_max_retries,
        timeout=config.provider.request_timeout_seconds,
    )
    model = OpenAIResponsesModel(
        config.provider.model, provider=OpenAIProvider(openai_client=client)
    )
    settings = openai_responses_settings(config.provider)
    agent = Agent[object, CargoLanguageGenerationOutput](
        model,
        output_type=NativeOutput(
            CargoLanguageGenerationOutput,
            name="synthetic_bill_of_lading_cargo_language",
            description=(
                "Return target-facing printed cargo text for every supplied cargo group while "
                "preserving field topology and generic marks literals."
            ),
            strict=True,
        ),
        system_prompt=system_prompt,
        model_settings=settings,
        retries=config.workflow.structured_output_retries,
        max_concurrency=ConcurrencyLimiter(config.workflow.max_concurrent_requests),
        name="synthetic-bill-of-lading-cargo-language",
    )
    wall_started = time.perf_counter()
    pending = {
        index: asyncio.create_task(
            _run_case(
                index=index,
                seed=seed,
                agent=agent,
                config=config,
                output_schema_sha256=output_schema_sha256,
            )
        )
        for index, seed in enumerate(seeds)
        if index not in existing
    }
    for index, task in pending.items():
        record, transcript = await task
        staged.publish_json(_case_path(index), record.model_dump(mode="json"))
        staged.publish_json(_transcript_path(index), transcript)
        existing[index] = (record, transcript)
    await client.close()
    records = tuple(existing[index][0] for index in range(len(seeds)))
    wall_seconds = time.perf_counter() - wall_started
    total_cost = sum((row.usage.estimatedCostUsd for row in records), Decimal(0))
    summary: dict[str, JsonValue] = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "model": config.provider.model,
        "reasoningEffort": config.provider.reasoning_effort,
        "providerNativeStrictJsonSchema": True,
        "automaticRepairRequestsPerCase": 0,
        "providerResponseStorage": True,
        "cases": len(records),
        "schemaValidOutputs": sum(row.output is not None for row in records),
        "validatedOutputs": sum(row.status == "success" for row in records),
        "validationFailures": sum(row.status == "validation_failed" for row in records),
        "callFailures": sum(row.status == "call_failed" for row in records),
        "requests": sum(row.usage.requests for row in records),
        "inputTokens": sum(row.usage.inputTokens for row in records),
        "cacheReadTokens": sum(row.usage.cacheReadTokens for row in records),
        "cacheWriteTokens": sum(row.usage.cacheWriteTokens for row in records),
        "outputTokens": sum(row.usage.outputTokens for row in records),
        "reasoningTokens": sum(row.usage.reasoningTokens for row in records),
        "visibleOutputTokens": sum(row.usage.visibleOutputTokens for row in records),
        "estimatedCostUsd": str(total_cost),
        "wallSeconds": wall_seconds,
        "meanCaseLatencySeconds": sum(row.durationMs for row in records) / len(records) / 1000.0,
        "peakProcessRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "outputSchemaSha256": output_schema_sha256,
        "transactionSha256": transaction_sha256,
    }
    staged.publish_bytes(
        "generation/results.jsonl",
        b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in records),
    )
    staged.publish_json("generation/summary.json", summary)
    staged.publish_bytes("REPORT.md", _report(records, summary).encode("utf-8"))
    staged.commit(
        expected_artifacts=expected,
        metadata={"cases": len(seeds), "schema_version": 1},
    )
    return summary


def run_cargo_language_probe(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisCargoLanguageProbeConfig,
) -> dict[str, JsonValue]:
    return asyncio.run(
        _run_probe_async(project_root=project_root, config_path=config_path, config=config)
    )
