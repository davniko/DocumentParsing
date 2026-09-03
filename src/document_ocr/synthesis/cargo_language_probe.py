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
_TOKEN = re.compile(r"[A-Z0-9]+")
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
    handlingInstructionSlots: tuple[CargoTextSlot, ...]

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
        tuple[NonEmptyText | None, ...],
        Field(
            max_length=32,
            description=(
                "One result per auxiliary source slot, in order. Return grounded replacement "
                "text only when supplied target facts support it; return null to remove a source "
                "slot that has no target-grounded semantic replacement."
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
            *(value for value in self.additionalInformation if value is not None),
            *self.marksAndNumbers,
            *self.handlingInstructions,
        )
        if any("\n" in value or "\r" in value for value in values):
            raise ValueError("cargo-language values must be single-line semantic values")
        for label, rows in (
            ("additionalInformation", self.additionalInformation),
            ("marksAndNumbers", self.marksAndNumbers),
            ("handlingInstructions", self.handlingInstructions),
        ):
            substantive = tuple(value for value in rows if value is not None)
            if len(substantive) != len(set(substantive)):
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
        if self.status == "validation_failed" and cast(
            CargoLanguageValidation, self.validation
        ).passed:
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


def _validate_completion_run(
    project_root: Path, config: SynthesisCargoLanguageProbeConfig
) -> Path:
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
        handling = _text_values(
            raw_group.get("handlingInstructions"), label="handlingInstructions"
        )
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
                        CargoTextSlot(sourceStyleReference=value) for value in additional
                    ),
                    marksAndNumbersSlots=tuple(
                        CargoMarksSlot(
                            action="preserve_literal" if _generic_mark(value) else "generate",
                            sourceStyleReference=value,
                        )
                        for value in marks
                    ),
                    handlingInstructionSlots=tuple(
                        CargoTextSlot(sourceStyleReference=value) for value in handling
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
        checks[f"{prefix}_marks_count"] = len(generated.marksAndNumbers) == len(
            contract.marksAndNumbersSlots
        )
        checks[f"{prefix}_handling_count"] = len(generated.handlingInstructions) == len(
            contract.handlingInstructionSlots
        )
        if len(generated.marksAndNumbers) == len(contract.marksAndNumbersSlots):
            checks[f"{prefix}_literal_marks_preserved"] = all(
                generated.marksAndNumbers[index] == slot.sourceStyleReference
                for index, slot in enumerate(contract.marksAndNumbersSlots)
                if slot.action == "preserve_literal"
            )
            checks[f"{prefix}_substantive_marks_anonymized"] = all(
                generated.marksAndNumbers[index].casefold()
                != slot.sourceStyleReference.casefold()
                for index, slot in enumerate(contract.marksAndNumbersSlots)
                if slot.action == "generate"
            )
        else:
            checks[f"{prefix}_literal_marks_preserved"] = False
            checks[f"{prefix}_substantive_marks_anonymized"] = False
        generated_values = (
            *((generated.description,) if generated.description else ()),
            *(value for value in generated.additionalInformation if value is not None),
            *generated.marksAndNumbers,
            *generated.handlingInstructions,
        )
        hs_codes = tuple(
            row.hsCode for row in expected.goodsIdentities if row.hsCode is not None
        )
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
                for token in ("FROZEN", "CHILLED", "COLD", "REEFER", "TEMPERATURE", "°C")
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
    plans = _load_completion_plans(
        plans_path, records=config.inputs.completion_plans.records
    )
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
        b"".join(
            canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in records
        ),
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
