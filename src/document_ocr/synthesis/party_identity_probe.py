"""Provider-constrained, first-pass probes for synthetic B/L party identities.

This is the first linguistic synthesis boundary.  Each configured case is one
independent model request: there is no message history, repair turn, reviewer,
or hidden fallback.  The complete model-visible request, provider transcript,
structured result, usage, price estimate, and deterministic validation are
published so the experiment can be audited without repeating paid calls.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import resource
import time
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from difflib import SequenceMatcher
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from urllib.parse import urlsplit

from dotenv import dotenv_values
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator
from pydantic_ai import Agent, NativeOutput, capture_run_messages
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage, UsageLimits

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import (
    PartyIdentityProbeCaseConfig,
    PartyIdentityProbePricingConfig,
    PartyIdentityRole,
    SynthesisPartyIdentityProbeConfig,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.semantic_completion_pipeline import SemanticCompletionPlanRow
from document_ocr.training.config import resolve_config_path

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_USD_QUANTUM = Decimal("0.000000000001")
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)


class PartyIdentityContactShape(BaseModel):
    model_config = _STRICT

    contactNamePresent: bool
    phoneNumberCount: Annotated[int, Field(ge=0, le=8)]
    emailAddressCount: Annotated[int, Field(ge=0, le=8)]
    websiteUrlCount: Annotated[int, Field(ge=0, le=8)]


class PartyIdentityFieldPresence(BaseModel):
    model_config = _STRICT

    addressPresent: bool
    cityPresent: Literal[True]
    countryPresent: Literal[True]
    contacts: PartyIdentityContactShape


class PartyIdentityGoodsSeed(BaseModel):
    model_config = _STRICT

    cargoGroupId: NonEmptyText
    kind: Literal["ordinary_goods", "dangerous_goods"]
    description: NonEmptyText
    hsCode: Annotated[str, StringConstraints(pattern=r"^[0-9]{6,18}$")] | None
    chapterDescription: NonEmptyText | None
    thermalProfile: Literal["FROZEN", "CHILLED"] | None
    unNumber: Annotated[str, StringConstraints(pattern=r"^[0-9]{4}$")] | None
    hazardClass: NonEmptyText | None
    packingGroup: NonEmptyText | None
    flashpointCelsius: float | None

    @model_validator(mode="after")
    def semantic_branch_is_complete(self) -> PartyIdentityGoodsSeed:
        dangerous = self.kind == "dangerous_goods"
        if dangerous != (self.unNumber is not None and self.hazardClass is not None):
            raise ValueError("dangerous-goods seeds require UN number and hazard class")
        if not dangerous and any(
            value is not None
            for value in (
                self.unNumber,
                self.hazardClass,
                self.packingGroup,
                self.flashpointCelsius,
            )
        ):
            raise ValueError("ordinary-goods seed contains dangerous-goods fields")
        return self


class PartyIdentityGenerationSeed(BaseModel):
    model_config = _STRICT

    caseId: NonEmptyText
    sourceDocumentId: Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
    scenarioId: NonEmptyText
    partyRole: PartyIdentityRole
    partyOccurrence: Annotated[int, Field(ge=0)]
    targetLocality: dict[Literal["city", "country"], NonEmptyText]
    goods: tuple[PartyIdentityGoodsSeed, ...] = Field(min_length=1)
    sourcePartyNameStyleReference: NonEmptyText | None
    fieldPresence: PartyIdentityFieldPresence

    @model_validator(mode="after")
    def locality_has_both_values(self) -> PartyIdentityGenerationSeed:
        if set(self.targetLocality) != {"city", "country"}:
            raise ValueError("party target locality must contain exactly city and country")
        return self


class GeneratedPartyContacts(BaseModel):
    """Target-schema contact values, with empty collections for absent fields."""

    model_config = _STRICT

    contactName: Annotated[
        NonEmptyText | None,
        Field(
            description=(
                "Fictional human contact name for this party, or null when the input field "
                "presence says no contact name. Never emit a heading or company name."
            )
        ),
    ]
    phoneNumbers: Annotated[
        tuple[NonEmptyText, ...],
        Field(
            max_length=8,
            description=(
                "Distinct printable phone numbers plausible in the target country; exact count "
                "is specified by the input field-presence contract."
            ),
        ),
    ]
    emailAddresses: Annotated[
        tuple[NonEmptyText, ...],
        Field(
            max_length=8,
            description=(
                "Distinct syntactically valid email addresses belonging to the fictional entity; "
                "exact count is specified by the input field-presence contract."
            ),
        ),
    ]
    websiteUrls: Annotated[
        tuple[NonEmptyText, ...],
        Field(
            max_length=8,
            description=(
                "Distinct printable website host names or URLs belonging to the fictional entity; "
                "exact count is specified by the input field-presence contract."
            ),
        ),
    ]

    @model_validator(mode="after")
    def values_are_unique(self) -> GeneratedPartyContacts:
        for label, values in (
            ("phoneNumbers", self.phoneNumbers),
            ("emailAddresses", self.emailAddresses),
            ("websiteUrls", self.websiteUrls),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must contain distinct values")
        return self


class GeneratedPartyIdentity(BaseModel):
    """One fictional party projected directly onto the task-facing party fields."""

    model_config = _STRICT

    partyRole: Annotated[
        PartyIdentityRole,
        Field(description="Exact party role supplied by the request."),
    ]
    name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=2, max_length=180),
        Field(
            description=(
                "A novel fictional legal or trading name plausible for the role and locality; "
                "not the source name and not a trivial edit of it."
            )
        ),
    ]
    address: Annotated[
        NonEmptyText | None,
        Field(
            description=(
                "One joined single-line postal address excluding party name, city, country, "
                "contacts, identifiers, and headings; null only when not requested."
            )
        ),
    ]
    city: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
        Field(description="Exact target city/locality supplied by the request."),
    ]
    country: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
        Field(description="Exact target country name supplied by the request."),
    ]
    contactDetails: GeneratedPartyContacts

    @model_validator(mode="after")
    def address_is_one_semantic_value(self) -> GeneratedPartyIdentity:
        if self.address is not None and any(character in self.address for character in "\r\n"):
            raise ValueError("address must be a single-line semantic value")
        return self


class PartyIdentityGenerationOutput(BaseModel):
    model_config = _STRICT

    party: GeneratedPartyIdentity


class PartyIdentityValidation(BaseModel):
    model_config = _STRICT

    passed: bool
    checks: dict[NonEmptyText, bool]
    sourceNameSimilarity: Annotated[float, Field(ge=0, le=1)] | None
    generatedNameNormalized: NonEmptyText
    sourceNameNormalized: NonEmptyText | None

    @model_validator(mode="after")
    def aggregate_matches(self) -> PartyIdentityValidation:
        if not self.checks or self.passed != all(self.checks.values()):
            raise ValueError("party validation aggregate differs from its checks")
        return self


class PartyIdentityUsageReceipt(BaseModel):
    model_config = _STRICT

    requests: Annotated[int, Field(ge=0)]
    providerResponseIds: tuple[NonEmptyText, ...]
    finishReasons: tuple[NonEmptyText, ...]
    inputTokens: Annotated[int, Field(ge=0)]
    cacheReadTokens: Annotated[int, Field(ge=0)]
    cacheWriteTokens: Annotated[int, Field(ge=0)]
    outputTokens: Annotated[int, Field(ge=0)]
    reasoningTokens: Annotated[int, Field(ge=0)]
    visibleOutputTokens: Annotated[int, Field(ge=0)]
    estimatedCostUsd: Annotated[Decimal, Field(ge=0, decimal_places=12)]

    @model_validator(mode="after")
    def token_buckets_are_consistent(self) -> PartyIdentityUsageReceipt:
        if self.cacheReadTokens + self.cacheWriteTokens > self.inputTokens:
            raise ValueError("cache buckets exceed input token total")
        if self.reasoningTokens > self.outputTokens:
            raise ValueError("reasoning tokens exceed output token total")
        if self.visibleOutputTokens != self.outputTokens - self.reasoningTokens:
            raise ValueError("visible output tokens differ from output minus reasoning")
        if self.requests != len(self.finishReasons):
            raise ValueError("request count differs from response finish-reason count")
        return self


class PartyIdentityCaseRecord(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[1]
    caseIndex: Annotated[int, Field(ge=0)]
    seed: PartyIdentityGenerationSeed
    userPromptSha256: Sha256
    outputSchemaSha256: Sha256
    providerNativeStrictJsonSchema: Literal[True]
    automaticRepairRequests: Literal[0]
    startedAt: datetime
    completedAt: datetime
    durationMs: Annotated[float, Field(ge=0)]
    status: Literal["success", "validation_failed", "call_failed"]
    output: PartyIdentityGenerationOutput | None
    validation: PartyIdentityValidation | None
    usage: PartyIdentityUsageReceipt
    errorType: NonEmptyText | None
    errorMessage: NonEmptyText | None

    @model_validator(mode="after")
    def status_payloads_match(self) -> PartyIdentityCaseRecord:
        if self.completedAt < self.startedAt:
            raise ValueError("party case completion precedes start")
        successful_call = self.status != "call_failed"
        if successful_call != (self.output is not None and self.validation is not None):
            raise ValueError("party case output/validation do not match call status")
        if self.status == "success" and not cast(PartyIdentityValidation, self.validation).passed:
            raise ValueError("successful party case did not pass validation")
        if self.status == "validation_failed" and cast(
            PartyIdentityValidation, self.validation
        ).passed:
            raise ValueError("validation-failed case has passing checks")
        if (self.status == "call_failed") != (self.errorType is not None):
            raise ValueError("party case error fields do not match call status")
        return self


def _resolve_pinned_file(
    project_root: Path,
    configured_path: str,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    path = resolve_config_path(project_root, configured_path)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} differs from its configured pin: {path}")
    return path.resolve(strict=True)


def _validate_completion_run(project_root: Path, config: SynthesisPartyIdentityProbeConfig) -> Path:
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


def _party_at(target: Mapping[str, Any], case: PartyIdentityProbeCaseConfig) -> Mapping[str, Any]:
    try:
        patch = cast(Mapping[str, Any], target["documentPatch"])
        parties = cast(Mapping[str, Any], patch["parties"])
        value = parties[case.party_role]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"configured party is absent: {case.document_id} {case.party_role}"
        ) from error
    if isinstance(value, list):
        if case.occurrence >= len(value) or not isinstance(value[case.occurrence], dict):
            raise ValueError(f"configured party occurrence is absent: {case}")
        return cast(Mapping[str, Any], value[case.occurrence])
    if case.occurrence != 0 or not isinstance(value, dict):
        raise ValueError(f"configured singleton party has invalid occurrence: {case}")
    return cast(Mapping[str, Any], value)


def _goods_seeds(plan: SemanticCompletionPlanRow) -> tuple[PartyIdentityGoodsSeed, ...]:
    rows: list[PartyIdentityGoodsSeed] = []
    for cargo in plan.cargo_realizations:
        rows.append(
            PartyIdentityGoodsSeed(
                cargoGroupId=cargo.cargo_group_id,
                kind="ordinary_goods",
                description=cargo.description,
                hsCode=cargo.output_hs_code or cargo.hs6,
                chapterDescription=cargo.chapter_description,
                thermalProfile=cargo.thermal_profile,
                unNumber=None,
                hazardClass=None,
                packingGroup=None,
                flashpointCelsius=None,
            )
        )
    flashpoints = {
        (row.cargo_group_id, row.dangerous_goods_order): row.value_celsius
        for row in plan.flashpoint_realizations
        if row.generated
    }
    for value in plan.upstream_dangerous_goods_realizations:
        group_id = cast(str, value["cargo_group_id"])
        order = cast(int, value["dangerous_goods_order"])
        generated_codes = cast(Sequence[str], value.get("generated_hs_codes") or ())
        rows.append(
            PartyIdentityGoodsSeed(
                cargoGroupId=group_id,
                kind="dangerous_goods",
                description=cast(str, value["proper_shipping_name"]),
                hsCode=generated_codes[0] if generated_codes else None,
                chapterDescription=None,
                thermalProfile=None,
                unNumber=cast(str, value["un_number"]),
                hazardClass=cast(str, value["exact_hazard_class"]),
                packingGroup=cast(str | None, value.get("packing_group_category")),
                flashpointCelsius=flashpoints.get((group_id, order)),
            )
        )
    if not rows:
        raise ValueError(f"party probe source has no resolved cargo semantics: {plan.scenario_id}")
    return tuple(rows)


def build_party_generation_seed(
    *,
    case_index: int,
    case: PartyIdentityProbeCaseConfig,
    plan: SemanticCompletionPlanRow,
) -> PartyIdentityGenerationSeed:
    party = _party_at(plan.target, case)
    name = party.get("name")
    city = party.get("city")
    country = party.get("country")
    if not isinstance(name, str) or not isinstance(city, str) or not isinstance(country, str):
        raise ValueError("party probe requires source name and resolved target city/country")
    contact = party.get("contactDetails") or {}
    if not isinstance(contact, dict):
        raise ValueError("party contact details are not an object")
    return PartyIdentityGenerationSeed(
        caseId=f"party-{case_index + 1:02d}-{case.party_role}",
        sourceDocumentId=case.document_id,
        scenarioId=plan.scenario_id,
        partyRole=case.party_role,
        partyOccurrence=case.occurrence,
        targetLocality={"city": city, "country": country},
        goods=_goods_seeds(plan),
        sourcePartyNameStyleReference=name,
        fieldPresence=PartyIdentityFieldPresence(
            addressPresent=isinstance(party.get("address"), str),
            cityPresent=True,
            countryPresent=True,
            contacts=PartyIdentityContactShape(
                contactNamePresent=isinstance(contact.get("contactName"), str),
                phoneNumberCount=len(contact.get("phoneNumbers") or ()),
                emailAddressCount=len(contact.get("emailAddresses") or ()),
                websiteUrlCount=len(contact.get("websiteUrls") or ()),
            ),
        ),
    )


def _normalize_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[A-Z0-9]+", ascii_value.upper()))


def _source_name_inventory(path: Path, *, records: int, target_field: str) -> frozenset[str]:
    names: set[str] = set()
    observed = 0
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip() or not raw.endswith(b"\n"):
                raise ValueError(f"source corpus row {line_number} is blank or unterminated")
            try:
                target = json.loads(raw)[target_field]
                parties = target["documentPatch"].get("parties") or {}
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"source corpus row {line_number} is malformed") from error
            observed += 1
            for value in parties.values():
                rows = value if isinstance(value, list) else [value]
                for party in rows:
                    if isinstance(party, dict) and isinstance(party.get("name"), str):
                        names.add(_normalize_name(party["name"]))
    if observed != records:
        raise ValueError("source corpus count differs from configuration")
    if not names:
        raise ValueError("source party-name inventory is empty")
    return frozenset(names)


def _contains_normalized(haystack: str, needle: str) -> bool:
    normalized_haystack = " ".join(re.findall(r"[A-Z0-9]+", haystack.upper()))
    normalized_needle = " ".join(re.findall(r"[A-Z0-9]+", needle.upper()))
    return bool(normalized_needle and normalized_needle in normalized_haystack)


def validate_generated_party(
    *,
    seed: PartyIdentityGenerationSeed,
    output: PartyIdentityGenerationOutput,
    source_names: frozenset[str],
) -> PartyIdentityValidation:
    party = output.party
    source_name = seed.sourcePartyNameStyleReference
    generated_normalized = _normalize_name(party.name)
    source_normalized = _normalize_name(source_name) if source_name is not None else None
    shape = seed.fieldPresence.contacts
    address = party.address
    checks = {
        "party_role_exact": party.partyRole == seed.partyRole,
        "city_exact": party.city == seed.targetLocality["city"],
        "country_exact": party.country == seed.targetLocality["country"],
        "address_presence_exact": (address is not None) == seed.fieldPresence.addressPresent,
        "address_single_line": address is None
        or ("\n" not in address and "\r" not in address),
        "address_excludes_city": address is None
        or not _contains_normalized(address, seed.targetLocality["city"]),
        "address_excludes_country": address is None
        or not _contains_normalized(address, seed.targetLocality["country"]),
        "contact_name_presence_exact": (party.contactDetails.contactName is not None)
        == shape.contactNamePresent,
        "phone_count_exact": len(party.contactDetails.phoneNumbers) == shape.phoneNumberCount,
        "email_count_exact": len(party.contactDetails.emailAddresses) == shape.emailAddressCount,
        "website_count_exact": len(party.contactDetails.websiteUrls) == shape.websiteUrlCount,
        "phone_syntax": all(
            7 <= len(re.findall(r"[0-9]", value)) <= 16
            and not re.search(r"[*Xx]{2,}", value)
            for value in party.contactDetails.phoneNumbers
        ),
        "email_syntax": all(
            _EMAIL.fullmatch(value) is not None and value.lower().split("@", 1)[1] != "example.com"
            for value in party.contactDetails.emailAddresses
        ),
        "website_syntax": all(
            " " not in value
            and "." in urlsplit(value if "://" in value else f"https://{value}").netloc
            and "example.com" not in value.lower()
            for value in party.contactDetails.websiteUrls
        ),
        "name_differs_from_source": source_normalized is None
        or generated_normalized != source_normalized,
        "name_not_in_source_corpus": generated_normalized not in source_names,
    }
    similarity = (
        SequenceMatcher(None, generated_normalized, source_normalized).ratio()
        if source_normalized is not None
        else None
    )
    return PartyIdentityValidation(
        passed=all(checks.values()),
        checks=checks,
        sourceNameSimilarity=similarity,
        generatedNameNormalized=generated_normalized,
        sourceNameNormalized=source_normalized,
    )


def _load_openai_key(project_root: Path, environment_file: str) -> str:
    path = resolve_config_path(project_root, environment_file)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"environment file is not a regular file: {path}")
    key = os.environ.get("OPENAI_API_KEY") or dotenv_values(path).get("OPENAI_API_KEY")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("OPENAI_API_KEY is absent or empty")
    return key


def _price_usage(usage: RequestUsage, pricing: PartyIdentityProbePricingConfig) -> Decimal:
    uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
    if uncached < 0:
        raise ValueError("provider reports cache tokens above total input tokens")
    cost = (
        Decimal(uncached) * Decimal(str(pricing.input_usd_per_million))
        + Decimal(usage.cache_read_tokens)
        * Decimal(str(pricing.cached_input_usd_per_million))
        + Decimal(usage.cache_write_tokens)
        * Decimal(str(pricing.input_usd_per_million))
        * Decimal(str(pricing.cache_write_multiplier))
        + Decimal(usage.output_tokens) * Decimal(str(pricing.output_usd_per_million))
    ) / Decimal(1_000_000)
    return cost.quantize(_USD_QUANTUM, rounding=ROUND_HALF_UP)


def _usage_receipt(
    responses: Sequence[ModelResponse], pricing: PartyIdentityProbePricingConfig
) -> PartyIdentityUsageReceipt:
    usage = RequestUsage()
    for response in responses:
        usage.incr(response.usage)
    reasoning_tokens = usage.details.get("reasoning_tokens", 0)
    if not isinstance(reasoning_tokens, int) or reasoning_tokens < 0:
        raise ValueError("provider reports invalid reasoning token usage")
    response_ids = tuple(
        response.provider_response_id
        for response in responses
        if response.provider_response_id is not None
    )
    finish_reasons = tuple(response.finish_reason or "unknown" for response in responses)
    return PartyIdentityUsageReceipt(
        requests=len(responses),
        providerResponseIds=response_ids,
        finishReasons=finish_reasons,
        inputTokens=usage.input_tokens,
        cacheReadTokens=usage.cache_read_tokens,
        cacheWriteTokens=usage.cache_write_tokens,
        outputTokens=usage.output_tokens,
        reasoningTokens=reasoning_tokens,
        visibleOutputTokens=usage.output_tokens - reasoning_tokens,
        estimatedCostUsd=_price_usage(usage, pricing),
    )


def _model_messages(messages: Sequence[ModelRequest | ModelResponse]) -> JsonValue:
    return cast(JsonValue, ModelMessagesTypeAdapter.dump_python(list(messages), mode="json"))


async def _run_case(
    *,
    index: int,
    seed: PartyIdentityGenerationSeed,
    agent: Agent[object, PartyIdentityGenerationOutput],
    config: SynthesisPartyIdentityProbeConfig,
    source_names: frozenset[str],
    output_schema_sha256: str,
) -> tuple[PartyIdentityCaseRecord, JsonValue]:
    payload = seed.model_dump(mode="json")
    user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
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
            output = PartyIdentityGenerationOutput.model_validate(
                result.output.model_dump(mode="python"), strict=True
            )
            validation = validate_generated_party(
                seed=seed,
                output=output,
                source_names=source_names,
            )
            responses = tuple(
                message for message in result.new_messages() if isinstance(message, ModelResponse)
            )
            if len(responses) != 1 or result.usage.requests != 1:
                raise RuntimeError("standalone party probe produced other than one model response")
            completed_at = datetime.now(UTC)
            return (
                PartyIdentityCaseRecord(
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
                    usage=_usage_receipt(responses, config.provider.pricing),
                    errorType=None,
                    errorMessage=None,
                ),
                _model_messages(captured),
            )
        except Exception as error:
            responses = tuple(
                message for message in captured if isinstance(message, ModelResponse)
            )
            completed_at = datetime.now(UTC)
            return (
                PartyIdentityCaseRecord(
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
                    usage=_usage_receipt(responses, config.provider.pricing),
                    errorType=type(error).__name__,
                    errorMessage=str(error),
                ),
                _model_messages(captured),
            )


def _case_relative_path(index: int, seed: PartyIdentityGenerationSeed) -> str:
    return f"generation/cases/{index + 1:02d}-{seed.partyRole}.json"


def _transcript_relative_path(index: int, seed: PartyIdentityGenerationSeed) -> str:
    return f"generation/transcripts/{index + 1:02d}-{seed.partyRole}.json"


def _request_relative_path(index: int, seed: PartyIdentityGenerationSeed) -> str:
    return f"generation/requests/{index + 1:02d}-{seed.partyRole}.json"


def _report(records: Sequence[PartyIdentityCaseRecord], summary: Mapping[str, Any]) -> str:
    lines = [
        "# Synthetic party identity probe",
        "",
        (
            "Five independent first-pass GPT-5.6 Luna calls using provider-native strict JSON "
            "Schema. No conversation history, semantic repair request, reviewer, PDF, or raw OCR "
            "was used."
        ),
        "",
        "## Aggregate",
        "",
        f"- Cases: {summary['cases']}",
        f"- Schema-valid provider outputs: {summary['schemaValidOutputs']}",
        f"- Deterministic validation passes: {summary['validatedOutputs']}",
        f"- Model requests: {summary['requests']}",
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
        seed = record.seed
        lines.extend(
            [
                f"### {record.caseIndex + 1}. {seed.partyRole}",
                "",
                f"- Case: `{seed.caseId}`",
                f"- Source document: `{seed.sourceDocumentId}`",
                (
                    f"- Target locality: {seed.targetLocality['city']}, "
                    f"{seed.targetLocality['country']}"
                ),
                f"- Source-name style reference: `{seed.sourcePartyNameStyleReference}`",
                "- Goods supplied:",
            ]
        )
        for good in seed.goods:
            qualifiers = [good.kind, f"HS {good.hsCode}" if good.hsCode else "no HS"]
            if good.thermalProfile:
                qualifiers.append(good.thermalProfile)
            if good.unNumber:
                qualifiers.append(f"UN {good.unNumber} / class {good.hazardClass}")
            lines.append(f"  - {good.description} ({'; '.join(qualifiers)})")
        lines.extend(
            [
                (
                    "- Requested field presence: `"
                    + json.dumps(seed.fieldPresence.model_dump(mode="json"), sort_keys=True)
                    + "`"
                ),
                f"- Status: `{record.status}`",
                f"- Latency: {record.durationMs / 1000.0:.3f} s",
                (
                    f"- Usage: {record.usage.inputTokens} input, "
                    f"{record.usage.outputTokens} output "
                    f"({record.usage.reasoningTokens} reasoning), "
                    f"${record.usage.estimatedCostUsd} estimated"
                ),
                "",
            ]
        )
        if record.output is not None:
            validation = cast(PartyIdentityValidation, record.validation)
            lines.extend(
                [
                    "```json",
                    json.dumps(
                        record.output.model_dump(mode="json"),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    "```",
                    "",
                    f"Validation: `{json.dumps(validation.checks, sort_keys=True)}`",
                    "",
                ]
            )
        else:
            lines.extend([f"Error: `{record.errorType}: {record.errorMessage}`", ""])
    return "\n".join(lines).rstrip() + "\n"


async def _run_probe_async(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisPartyIdentityProbeConfig,
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
    source_path = _resolve_pinned_file(
        project_root,
        config.inputs.source_corpus.path,
        config.inputs.source_corpus.sha256,
        label="source corpus",
    )
    prompt_path = _resolve_pinned_file(
        project_root,
        config.prompt.path,
        config.prompt.sha256,
        label="party prompt",
    )
    plans = _load_completion_plans(
        plans_path,
        records=config.inputs.completion_plans.records,
    )
    seeds = tuple(
        build_party_generation_seed(
            case_index=index,
            case=case,
            plan=plans[case.document_id],
        )
        for index, case in enumerate(config.cases)
    )
    source_names = _source_name_inventory(
        source_path,
        records=config.inputs.source_corpus.records,
        target_field=config.inputs.source_target_field,
    )
    prompt_bytes = read_regular_file_bytes(prompt_path)
    try:
        system_prompt = prompt_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("party prompt is not valid UTF-8") from error
    output_schema = PartyIdentityGenerationOutput.model_json_schema(mode="validation")
    output_schema_bytes = canonical_json_bytes(output_schema)
    output_schema_sha256 = sha256_bytes(output_schema_bytes)
    implementation_sha256 = sha256_file(_IMPLEMENTATION_PATH)
    transaction = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "completionPlansSha256": config.inputs.completion_plans.sha256,
        "sourceCorpusSha256": config.inputs.source_corpus.sha256,
        "promptSha256": config.prompt.sha256,
        "outputSchemaSha256": output_schema_sha256,
        "implementationSha256": implementation_sha256,
        "cases": [seed.model_dump(mode="json") for seed in seeds],
        "runtime": {
            "model": config.provider.model,
            "reasoningEffort": config.provider.reasoning_effort,
            "pydanticAiVersion": version("pydantic-ai-slim"),
            "openaiVersion": version("openai"),
        },
    }
    transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    output_parent = resolve_config_path(project_root, config.run.output_dir)
    staged = StagedArtifactRun(
        output_parent=output_parent,
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
    for index, seed in enumerate(seeds):
        expected.extend(
            (
                _case_relative_path(index, seed),
                _request_relative_path(index, seed),
                _transcript_relative_path(index, seed),
            )
        )
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
        staged.publish_json(
            _request_relative_path(index, seed),
            {
                "systemPromptSha256": config.prompt.sha256,
                "systemPrompt": system_prompt,
                "userPrompt": json.dumps(
                    seed.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "seed": seed.model_dump(mode="json"),
            },
        )

    existing: dict[int, tuple[PartyIdentityCaseRecord, JsonValue]] = {}
    for index, seed in enumerate(seeds):
        result_path = staged.stage_root / _case_relative_path(index, seed)
        transcript_path = staged.stage_root / _transcript_relative_path(index, seed)
        if result_path.exists() or transcript_path.exists():
            if not result_path.is_file() or not transcript_path.is_file():
                raise RuntimeError("partial party case artifact pair cannot be resumed")
            existing[index] = (
                PartyIdentityCaseRecord.model_validate_json(
                    read_regular_file_bytes(result_path), strict=True
                ),
                cast(JsonValue, json.loads(read_regular_file_bytes(transcript_path))),
            )

    client = AsyncOpenAI(
        api_key=_load_openai_key(project_root, config.environment_file),
        max_retries=config.provider.transport_max_retries,
        timeout=config.provider.request_timeout_seconds,
    )
    provider = OpenAIProvider(openai_client=client)
    model = OpenAIResponsesModel(config.provider.model, provider=provider)
    settings = OpenAIResponsesModelSettings(
        max_tokens=config.provider.max_output_tokens,
        timeout=config.provider.request_timeout_seconds,
        openai_reasoning_effort=config.provider.reasoning_effort,
        openai_reasoning_mode="standard",
        openai_reasoning_context="current_turn",
        openai_store=config.provider.store_responses,
        openai_text_verbosity="low",
    )
    agent = Agent[object, PartyIdentityGenerationOutput](
        model,
        output_type=NativeOutput(
            PartyIdentityGenerationOutput,
            name="synthetic_maritime_party_identity",
            description=(
                "Return exactly one fictional, role-aware maritime party identity with the "
                "requested target locality and field-presence shape."
            ),
            strict=True,
        ),
        system_prompt=system_prompt,
        model_settings=settings,
        retries=config.workflow.structured_output_retries,
        max_concurrency=ConcurrencyLimiter(config.workflow.max_concurrent_requests),
        name="synthetic-maritime-party-identity",
    )
    wall_started = time.perf_counter()
    pending = {
        index: asyncio.create_task(
            _run_case(
                index=index,
                seed=seed,
                agent=agent,
                config=config,
                source_names=source_names,
                output_schema_sha256=output_schema_sha256,
            )
        )
        for index, seed in enumerate(seeds)
        if index not in existing
    }
    for index, task in pending.items():
        record, transcript = await task
        seed = seeds[index]
        staged.publish_json(_case_relative_path(index, seed), record.model_dump(mode="json"))
        staged.publish_json(_transcript_relative_path(index, seed), transcript)
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
    results_bytes = b"".join(
        canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in records
    )
    staged.publish_bytes("generation/results.jsonl", results_bytes)
    staged.publish_json("generation/summary.json", summary)
    staged.publish_bytes("REPORT.md", _report(records, summary).encode("utf-8"))
    staged.commit(
        expected_artifacts=expected,
        metadata={"cases": len(seeds), "schema_version": 1},
    )
    return summary


def run_party_identity_probe(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisPartyIdentityProbeConfig,
) -> dict[str, JsonValue]:
    return asyncio.run(
        _run_probe_async(project_root=project_root, config_path=config_path, config=config)
    )
