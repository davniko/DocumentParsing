"""Transactional party and cargo language completion for synthetic B/L plans.

The upstream semantic-completion run is the single source of structured truth.
This stage replaces every concrete party identity and every task-facing cargo
text leaf while preserving party/cargo topology.  It deliberately does not
patch raw OCR or realize categorical printed surfaces; consequently no output
from this stage is a training record yet.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import resource
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Literal, Union, cast
from urllib.parse import urlsplit

from openai import AsyncOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    create_model,
    model_validator,
)
from pydantic_ai import Agent, NativeOutput, capture_run_messages
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis import cargo_language_probe as cargo_language_module
from document_ocr.synthesis import party_identity_probe as party_identity_module
from document_ocr.synthesis.cargo_language_probe import (
    CargoLanguageGenerationOutput,
    CargoLanguageGenerationSeed,
    build_cargo_language_seed,
    normalize_cargo_language_output,
    validate_cargo_language,
)
from document_ocr.synthesis.config import (
    LinguisticProbePricingConfig,
    PartyIdentityRole,
    SynthesisLinguisticCompletionConfig,
)
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    load_openai_key,
    model_messages,
    openai_responses_settings,
    usage_receipt,
)
from document_ocr.synthesis.party_identity_probe import (
    PartyIdentityContactShape,
    PartyIdentityGoodsSeed,
    party_goods_seeds,
)
from document_ocr.synthesis.raw_text_template import printed_topology_mismatches
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.semantic_completion_pipeline import SemanticCompletionPlanRow
from document_ocr.synthesis.task_adapter import (
    BILL_OF_LADING_TASK_ADAPTER,
    BILL_OF_LADING_V5_TASK_ADAPTER,
)
from document_ocr.training.config import resolve_config_path

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_EMAIL_PATTERN = (
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)
EmailText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=6,
        max_length=254,
        pattern=_EMAIL_PATTERN,
    ),
]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_PARTY_ROLES: tuple[PartyIdentityRole, ...] = (
    "shipper",
    "consignee",
    "notifyParties",
    "carrier",
    "forwardingAgent",
    "deliveryAgent",
    "consolidator",
)
_SENSITIVE_PARTY_FIELDS = ("name", "address")
_CONTACT_FIELDS = ("contactName", "phoneNumbers", "emailAddresses", "websiteUrls")
_GENERIC_BLOCKERS_RESOLVED = frozenset(
    {
        "cargo_and_auxiliary_text_realization",
        "dangerous_goods_printed_text_realization",
        "handling_instructions_require_linguistic_realization",
        "party_identity_and_contact_anonymization",
    }
)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)


class IncompleteLinguisticCompletionError(RuntimeError):
    """At least one document exhausted its configured fresh attempts."""


class PartyTargetLocality(BaseModel):
    model_config = _STRICT

    city: NonEmptyText | None
    country: NonEmptyText | None


class PartyRouteContext(BaseModel):
    model_config = _STRICT

    portOfLoading: NonEmptyText | None
    portOfLoadingCountry: NonEmptyText | None
    portOfDischarge: NonEmptyText | None
    portOfDischargeCountry: NonEmptyText | None
    placeOfDelivery: NonEmptyText | None
    placeOfDeliveryCountry: NonEmptyText | None


class PartyCompletionFieldPresence(BaseModel):
    model_config = _STRICT

    namePresent: bool
    addressPresent: bool
    cityPresent: bool
    countryPresent: bool
    contactDetailsPresent: bool
    contacts: PartyIdentityContactShape

    @model_validator(mode="after")
    def contact_container_matches_fields(self) -> PartyCompletionFieldPresence:
        has_contact = self.contacts.contactNamePresent or any(
            (
                self.contacts.phoneNumberCount,
                self.contacts.emailAddressCount,
                self.contacts.websiteUrlCount,
            )
        )
        if self.contactDetailsPresent != has_contact:
            raise ValueError("party contactDetails presence differs from its children")
        return self


class PartyCompletionSeed(BaseModel):
    model_config = _STRICT

    unitId: NonEmptyText
    sourceDocumentId: Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
    scenarioId: NonEmptyText
    partyRole: PartyIdentityRole
    partyOccurrence: Annotated[int, Field(ge=0)]
    sameAsReference: Literal["shipper", "consignee"] | None
    targetLocality: PartyTargetLocality
    routeContext: PartyRouteContext
    goods: tuple[PartyIdentityGoodsSeed, ...] = Field(min_length=1)
    sourcePartyNameStyleReference: NonEmptyText | None
    fieldPresence: PartyCompletionFieldPresence

    @model_validator(mode="after")
    def target_values_match_presence(self) -> PartyCompletionSeed:
        if self.fieldPresence.namePresent != (self.sourcePartyNameStyleReference is not None):
            raise ValueError("party name presence differs from its style reference")
        if self.fieldPresence.cityPresent != (self.targetLocality.city is not None):
            raise ValueError("party city presence differs from target locality")
        if self.fieldPresence.countryPresent != (self.targetLocality.country is not None):
            raise ValueError("party country presence differs from target locality")
        if self.sameAsReference is not None and any(
            (
                self.fieldPresence.namePresent,
                self.fieldPresence.addressPresent,
                self.fieldPresence.cityPresent,
                self.fieldPresence.countryPresent,
            )
        ):
            raise ValueError("sameAs override unit may generate contacts only")
        return self


class GeneratedPartyContactsV2(BaseModel):
    model_config = _STRICT

    contactName: Annotated[
        NonEmptyText | None,
        Field(description="Fictional human contact name, or null when absent in the template."),
    ]
    phoneNumbers: Annotated[
        tuple[NonEmptyText, ...],
        Field(
            max_length=8,
            description="Fictional, locality-plausible phone numbers in template cardinality.",
        ),
    ]
    emailAddresses: Annotated[
        tuple[EmailText, ...],
        Field(
            max_length=8,
            description="Fictional syntactically valid emails coherent with the new party.",
        ),
    ]
    websiteUrls: Annotated[
        tuple[NonEmptyText, ...],
        Field(
            max_length=8,
            description="Fictional syntactically plausible websites coherent with the new party.",
        ),
    ]

    @model_validator(mode="after")
    def repeated_contacts_are_unique(self) -> GeneratedPartyContactsV2:
        for label, values in (
            ("phoneNumbers", self.phoneNumbers),
            ("emailAddresses", self.emailAddresses),
            ("websiteUrls", self.websiteUrls),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must contain distinct values")
        return self


class GeneratedPartyIdentityV2(BaseModel):
    model_config = _STRICT

    partyRole: Annotated[
        PartyIdentityRole,
        Field(description="Exact party role supplied by the generation seed."),
    ]
    name: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=2, max_length=180),
            Field(
                description=(
                    "Novel fictional legal or trading name plausible for the role and locality; "
                    "null exactly when the source field was absent."
                )
            ),
        ]
        | None
    )
    address: Annotated[
        NonEmptyText | None,
        Field(
            description=(
                "One fictional single-line postal address excluding party name, standalone city, "
                "country, contacts, identifiers, and headings; null exactly when absent."
            )
        ),
    ]
    city: Annotated[
        NonEmptyText | None,
        Field(description="Exact target city/locality from the seed, including null."),
    ]
    country: Annotated[
        NonEmptyText | None,
        Field(description="Exact target country from the seed, including null."),
    ]
    contactDetails: Annotated[
        GeneratedPartyContactsV2,
        Field(description="Contacts matching the source field-presence and cardinality contract."),
    ]

    @model_validator(mode="after")
    def semantic_values_are_single_line(self) -> GeneratedPartyIdentityV2:
        values = tuple(
            value
            for value in (
                self.name,
                self.address,
                self.city,
                self.country,
                self.contactDetails.contactName,
                *self.contactDetails.phoneNumbers,
                *self.contactDetails.emailAddresses,
                *self.contactDetails.websiteUrls,
            )
            if value is not None
        )
        if any("\n" in value or "\r" in value for value in values):
            raise ValueError("party values must be single-line semantic values")
        return self


class PartyCompletionOutput(BaseModel):
    model_config = _STRICT

    party: Annotated[
        GeneratedPartyIdentityV2,
        Field(description="One source-safe fictional replacement party."),
    ]


class PartyProjection(BaseModel):
    model_config = _STRICT

    role: PartyIdentityRole
    occurrence: Annotated[int, Field(ge=0)]
    unitId: NonEmptyText | None
    sameAsReference: Literal["shipper", "consignee"] | None


class PartyGenerationUnitPlan(BaseModel):
    model_config = _STRICT

    unitId: NonEmptyText
    seed: PartyCompletionSeed


class DocumentLinguisticPlan(BaseModel):
    model_config = _STRICT

    documentIndex: Annotated[int, Field(ge=0)]
    baseDocumentId: NonEmptyText
    scenarioId: NonEmptyText
    upstreamTargetSha256: Sha256
    partyUnits: tuple[PartyGenerationUnitPlan, ...]
    partyProjections: tuple[PartyProjection, ...]
    cargoSeed: CargoLanguageGenerationSeed


class LinguisticAttemptRecord(BaseModel):
    model_config = _STRICT

    attempt: Annotated[int, Field(ge=1, le=3)]
    schemaMode: Literal["static", "topology_constrained"]
    outputSchemaSha256: Sha256
    userPromptSha256: Sha256
    startedAt: datetime
    completedAt: datetime
    durationMs: Annotated[float, Field(ge=0)]
    status: Literal["success", "validation_failed", "call_failed"]
    output: dict[str, JsonValue] | None
    checks: dict[NonEmptyText, bool] | None
    usage: LinguisticUsageReceipt
    errorType: NonEmptyText | None
    errorMessage: NonEmptyText | None

    @model_validator(mode="after")
    def status_matches_payload(self) -> LinguisticAttemptRecord:
        if self.completedAt < self.startedAt:
            raise ValueError("linguistic attempt completion precedes start")
        if (self.errorType is None) != (self.errorMessage is None):
            raise ValueError("linguistic attempt error type/message presence differs")
        if self.status == "call_failed":
            if self.output is not None or self.checks is not None or self.errorType is None:
                raise ValueError("failed linguistic call carries an invalid payload")
        elif self.status == "validation_failed":
            if self.output is None:
                raise ValueError("linguistic validation failure lost the provider output")
            if self.checks is not None and all(self.checks.values()):
                raise ValueError("linguistic validation failure has no false check")
        elif (
            self.output is None
            or self.checks is None
            or not all(self.checks.values())
            or self.errorType is not None
        ):
            raise ValueError("successful linguistic attempt has an invalid payload")
        return self


class LinguisticUnitArtifact(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[1]
    stage: Literal["party_identity", "cargo_language"]
    unitId: NonEmptyText
    attempts: tuple[LinguisticAttemptRecord, ...] = Field(min_length=1, max_length=3)
    transcripts: tuple[JsonValue, ...] = Field(min_length=1, max_length=3)
    selectedAttempt: Annotated[int, Field(ge=1, le=3)] | None
    selectedOutputSha256: Sha256 | None
    status: Literal["success", "failed"]

    @model_validator(mode="after")
    def selection_matches_attempts(self) -> LinguisticUnitArtifact:
        if len(self.attempts) != len(self.transcripts):
            raise ValueError("linguistic unit attempt/transcript counts differ")
        successes = {row.attempt for row in self.attempts if row.status == "success"}
        success = self.status == "success"
        if success != (self.selectedAttempt in successes):
            raise ValueError("linguistic unit selection differs from successful attempts")
        if success != (self.selectedOutputSha256 is not None):
            raise ValueError("linguistic unit output digest differs from status")
        return self


class LinguisticCompletionDocumentRecord(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[1]
    baseDocumentId: NonEmptyText
    scenarioId: NonEmptyText
    upstreamTargetSha256: Sha256
    status: Literal["success", "failed"]
    partyUnitPaths: tuple[NonEmptyText, ...]
    cargoUnitPath: NonEmptyText
    target: dict[str, JsonValue] | None
    targetSha256: Sha256 | None
    remainingBlockers: tuple[NonEmptyText, ...]
    trainingEligible: Literal[False]

    @model_validator(mode="after")
    def target_matches_status(self) -> LinguisticCompletionDocumentRecord:
        success = self.status == "success"
        if success != (self.target is not None and self.targetSha256 is not None):
            raise ValueError("linguistic document target differs from status")
        if self.target is not None and (
            sha256_bytes(canonical_json_bytes(self.target)) != self.targetSha256
        ):
            raise ValueError("linguistic document target digest differs from payload")
        return self


class SourceSensitiveInventory(BaseModel):
    model_config = _STRICT

    names: frozenset[NonEmptyText]
    addresses: frozenset[NonEmptyText]
    contactNames: frozenset[NonEmptyText]
    phoneNumbers: frozenset[NonEmptyText]
    emailAddresses: frozenset[NonEmptyText]
    websiteUrls: frozenset[NonEmptyText]
    digest: Sha256


def _normalize_text(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[A-Z0-9]+", ascii_value.upper()))


def _normalize_phone(value: str) -> str:
    return "".join(re.findall(r"[0-9]+", value))


def _normalize_url(value: str) -> str:
    parsed = urlsplit(value if "://" in value else f"https://{value}")
    return (parsed.netloc + parsed.path.rstrip("/")).casefold()


def _normalizer(field: str) -> Callable[[str], str]:
    if field == "phoneNumbers":
        return _normalize_phone
    if field in {"emailAddresses", "websiteUrls"}:
        return (lambda value: value.casefold()) if field == "emailAddresses" else _normalize_url
    return _normalize_text


def _load_source_sensitive_inventory(
    path: Path,
    *,
    records: int,
    target_field: str,
    target_schema: Literal["bill_of_lading_relation_explicit_v3"],
) -> SourceSensitiveInventory:
    values: dict[str, set[str]] = {
        "names": set(),
        "addresses": set(),
        "contactNames": set(),
        "phoneNumbers": set(),
        "emailAddresses": set(),
        "websiteUrls": set(),
    }
    observed = 0
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip() or not raw.endswith(b"\n"):
                raise ValueError(f"source corpus row {line_number} is blank or unterminated")
            try:
                row = json.loads(raw)
                document_id = row["documentId"]
                target = row[target_field]
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"source corpus row {line_number} is malformed") from error
            if not isinstance(document_id, str) or not isinstance(target, dict):
                raise ValueError(f"source corpus row {line_number} has invalid identity/target")
            if target_schema != "bill_of_lading_relation_explicit_v3":
                raise ValueError(f"unsupported source target schema: {target_schema}")
            try:
                canonical = BILL_OF_LADING_TASK_ADAPTER.validate_target(
                    document_id=document_id,
                    target=target,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"source corpus row {line_number} fails its configured task schema"
                ) from error
            parties = canonical["documentPatch"].get("parties") or {}
            observed += 1
            for value in parties.values():
                rows = value if isinstance(value, list) else [value]
                for party in rows:
                    if not isinstance(party, dict):
                        continue
                    for field, inventory_field in (("name", "names"), ("address", "addresses")):
                        item = party.get(field)
                        if isinstance(item, str) and (normalized := _normalize_text(item)):
                            values[inventory_field].add(normalized)
                    contacts = party.get("contactDetails") or {}
                    if not isinstance(contacts, dict):
                        continue
                    contact_name = contacts.get("contactName")
                    if isinstance(contact_name, str) and (
                        normalized := _normalize_text(contact_name)
                    ):
                        values["contactNames"].add(normalized)
                    for field in ("phoneNumbers", "emailAddresses", "websiteUrls"):
                        for item in contacts.get(field) or ():
                            if isinstance(item, str) and (normalized := _normalizer(field)(item)):
                                values[field].add(normalized)
    if observed != records:
        raise ValueError("source corpus count differs from configuration")
    body = {key: sorted(rows) for key, rows in values.items()}
    return SourceSensitiveInventory(
        **{key: frozenset(rows) for key, rows in values.items()},
        digest=sha256_bytes(canonical_json_bytes(body)),
    )


def _route_context(target: Mapping[str, Any]) -> PartyRouteContext:
    patch = cast(Mapping[str, Any], target["documentPatch"])
    route = cast(Mapping[str, Any], patch.get("route") or {})

    def cell(role: str, field: str) -> str | None:
        value = route.get(role)
        if not isinstance(value, Mapping):
            return None
        item = value.get(field)
        return item if isinstance(item, str) else None

    return PartyRouteContext(
        portOfLoading=cell("portOfLoading", "name"),
        portOfLoadingCountry=cell("portOfLoading", "country"),
        portOfDischarge=cell("portOfDischarge", "name"),
        portOfDischargeCountry=cell("portOfDischarge", "country"),
        placeOfDelivery=cell("placeOfDelivery", "name"),
        placeOfDeliveryCountry=cell("placeOfDelivery", "country"),
    )


def _party_rows(
    target: Mapping[str, Any],
) -> tuple[tuple[PartyIdentityRole, int, dict[str, Any]], ...]:
    patch = cast(Mapping[str, Any], target["documentPatch"])
    parties = patch.get("parties")
    if not isinstance(parties, Mapping):
        return ()
    output: list[tuple[PartyIdentityRole, int, dict[str, Any]]] = []
    for role in _PARTY_ROLES:
        value = parties.get(role)
        rows = value if role == "notifyParties" and isinstance(value, list) else [value]
        for occurrence, party in enumerate(rows):
            if isinstance(party, dict):
                output.append((role, occurrence, party))
    return tuple(output)


def _party_signature(party: Mapping[str, Any]) -> str | None:
    if party.get("sameAs") is not None:
        return None
    body = {
        key: party.get(key)
        for key in ("name", "address", "city", "country", "contactDetails")
        if party.get(key) is not None
    }
    return sha256_bytes(canonical_json_bytes(body)) if body else None


def _contact_shape(party: Mapping[str, Any]) -> tuple[bool, PartyIdentityContactShape]:
    contacts = party.get("contactDetails")
    if contacts is None:
        return False, PartyIdentityContactShape(
            contactNamePresent=False,
            phoneNumberCount=0,
            emailAddressCount=0,
            websiteUrlCount=0,
        )
    if not isinstance(contacts, Mapping):
        raise ValueError("party contactDetails must be an object")
    return True, PartyIdentityContactShape(
        contactNamePresent=isinstance(contacts.get("contactName"), str),
        phoneNumberCount=len(contacts.get("phoneNumbers") or ()),
        emailAddressCount=len(contacts.get("emailAddresses") or ()),
        websiteUrlCount=len(contacts.get("websiteUrls") or ()),
    )


def _party_seed(
    *,
    plan: SemanticCompletionPlanRow,
    role: PartyIdentityRole,
    occurrence: int,
    party: Mapping[str, Any],
    unit_id: str,
) -> PartyCompletionSeed:
    contact_present, contacts = _contact_shape(party)
    name = party.get("name") if isinstance(party.get("name"), str) else None
    city = party.get("city") if isinstance(party.get("city"), str) else None
    country = party.get("country") if isinstance(party.get("country"), str) else None
    same_as = party.get("sameAs") if isinstance(party.get("sameAs"), str) else None
    return PartyCompletionSeed(
        unitId=unit_id,
        sourceDocumentId=plan.base_document_id,
        scenarioId=plan.scenario_id,
        partyRole=role,
        partyOccurrence=occurrence,
        sameAsReference=cast(Literal["shipper", "consignee"] | None, same_as),
        targetLocality=PartyTargetLocality(city=city, country=country),
        routeContext=_route_context(plan.target),
        goods=party_goods_seeds(plan),
        sourcePartyNameStyleReference=name,
        fieldPresence=PartyCompletionFieldPresence(
            namePresent=name is not None,
            addressPresent=isinstance(party.get("address"), str),
            cityPresent=city is not None,
            countryPresent=country is not None,
            contactDetailsPresent=contact_present,
            contacts=contacts,
        ),
    )


def build_document_linguistic_plan(
    *, index: int, plan: SemanticCompletionPlanRow
) -> DocumentLinguisticPlan:
    units: list[PartyGenerationUnitPlan] = []
    projections: list[PartyProjection] = []
    concrete_owner_by_signature: dict[str, str] = {}
    for role, occurrence, party in _party_rows(plan.target):
        same_as = party.get("sameAs") if isinstance(party.get("sameAs"), str) else None
        contact_present, _ = _contact_shape(party)
        if same_as is not None and not contact_present:
            projections.append(
                PartyProjection(
                    role=role,
                    occurrence=occurrence,
                    unitId=None,
                    sameAsReference=cast(Literal["shipper", "consignee"], same_as),
                )
            )
            continue
        has_sensitive_surface = (
            any(isinstance(party.get(field), str) for field in _SENSITIVE_PARTY_FIELDS)
            or contact_present
        )
        if not has_sensitive_surface:
            projections.append(
                PartyProjection(
                    role=role,
                    occurrence=occurrence,
                    unitId=None,
                    sameAsReference=None,
                )
            )
            continue
        signature = _party_signature(party)
        reused_unit = (
            concrete_owner_by_signature.get(signature)
            if role == "notifyParties" and signature is not None
            else None
        )
        if reused_unit is not None:
            projections.append(
                PartyProjection(
                    role=role,
                    occurrence=occurrence,
                    unitId=reused_unit,
                    sameAsReference=None,
                )
            )
            continue
        unit_id = f"party-{role}-{occurrence + 1}"
        unit = PartyGenerationUnitPlan(
            unitId=unit_id,
            seed=_party_seed(
                plan=plan,
                role=role,
                occurrence=occurrence,
                party=party,
                unit_id=unit_id,
            ),
        )
        units.append(unit)
        projections.append(
            PartyProjection(
                role=role,
                occurrence=occurrence,
                unitId=unit_id,
                sameAsReference=cast(Literal["shipper", "consignee"] | None, same_as),
            )
        )
        if role in {"shipper", "consignee"} and signature is not None:
            concrete_owner_by_signature.setdefault(signature, unit_id)
    if len({row.unitId for row in units}) != len(units):
        raise RuntimeError("party generation unit IDs are not unique")
    return DocumentLinguisticPlan(
        documentIndex=index,
        baseDocumentId=plan.base_document_id,
        scenarioId=plan.scenario_id,
        upstreamTargetSha256=plan.target_sha256,
        partyUnits=tuple(units),
        partyProjections=tuple(projections),
        cargoSeed=build_cargo_language_seed(case_index=index, plan=plan),
    )


def _location_component_present(address: str, locality: str | None) -> bool:
    if locality is None:
        return False
    expected = _normalize_text(locality)
    components = tuple(
        _normalize_text(value) for value in re.split(r"[,;|/]", address) if _normalize_text(value)
    )
    return expected in components


def _party_sensitive_values(party: GeneratedPartyIdentityV2) -> dict[str, tuple[str, ...]]:
    return {
        "names": tuple(value for value in (party.name,) if value is not None),
        "addresses": tuple(value for value in (party.address,) if value is not None),
        "contactNames": tuple(
            value for value in (party.contactDetails.contactName,) if value is not None
        ),
        "phoneNumbers": party.contactDetails.phoneNumbers,
        "emailAddresses": party.contactDetails.emailAddresses,
        "websiteUrls": party.contactDetails.websiteUrls,
    }


def validate_party_completion(
    *,
    seed: PartyCompletionSeed,
    output: PartyCompletionOutput,
    source_inventory: SourceSensitiveInventory,
) -> dict[str, bool]:
    party = output.party
    presence = seed.fieldPresence
    contacts = presence.contacts
    checks: dict[str, bool] = {
        "party_role_exact": party.partyRole == seed.partyRole,
        "name_presence_exact": (party.name is not None) == presence.namePresent,
        "address_presence_exact": (party.address is not None) == presence.addressPresent,
        "city_exact": party.city == seed.targetLocality.city,
        "country_exact": party.country == seed.targetLocality.country,
        "contact_name_presence_exact": (party.contactDetails.contactName is not None)
        == contacts.contactNamePresent,
        "phone_count_exact": len(party.contactDetails.phoneNumbers) == contacts.phoneNumberCount,
        "email_count_exact": len(party.contactDetails.emailAddresses) == contacts.emailAddressCount,
        "website_count_exact": len(party.contactDetails.websiteUrls) == contacts.websiteUrlCount,
        "address_excludes_standalone_city": party.address is None
        or not _location_component_present(party.address, seed.targetLocality.city),
        "address_excludes_standalone_country": party.address is None
        or not _location_component_present(party.address, seed.targetLocality.country),
        "phone_syntax": all(
            7 <= len(re.findall(r"[0-9]", value)) <= 16 and re.search(r"[*Xx]{2,}", value) is None
            for value in party.contactDetails.phoneNumbers
        ),
        "email_syntax": all(
            re.fullmatch(_EMAIL_PATTERN, value) is not None
            and value.rsplit("@", 1)[1].casefold() != "example.com"
            for value in party.contactDetails.emailAddresses
        ),
        "website_syntax": all(
            " " not in value
            and "." in urlsplit(value if "://" in value else f"https://{value}").netloc
            and "example.com" not in value.casefold()
            for value in party.contactDetails.websiteUrls
        ),
    }
    inventory_values = source_inventory.model_dump(mode="python", exclude={"digest"})
    for field, generated in _party_sensitive_values(party).items():
        normalizer = _normalizer(
            {
                "names": "name",
                "addresses": "address",
                "contactNames": "contactName",
            }.get(field, field)
        )
        source_values = cast(frozenset[str], inventory_values[field])
        checks[f"{field}_absent_from_source_corpus"] = all(
            bool(normalized := normalizer(value)) and normalized not in source_values
            for value in generated
        )
    return checks


def _party_validator(
    seed: PartyCompletionSeed,
    source_inventory: SourceSensitiveInventory,
) -> Callable[[BaseModel], dict[str, bool]]:
    def validate(output: BaseModel) -> dict[str, bool]:
        return validate_party_completion(
            seed=seed,
            output=cast(PartyCompletionOutput, output),
            source_inventory=source_inventory,
        )

    return validate


def _cargo_validator(
    seed: CargoLanguageGenerationSeed,
) -> Callable[[BaseModel], dict[str, bool]]:
    def validate(output: BaseModel) -> dict[str, bool]:
        return dict(
            validate_cargo_language(
                seed=seed,
                output=cast(CargoLanguageGenerationOutput, output),
            ).checks
        )

    return validate


def _dynamic_party_output_model(seed: PartyCompletionSeed) -> type[BaseModel]:
    shape = seed.fieldPresence

    def present_type(present: bool, populated: Any) -> Any:
        return populated if present else Literal[None]

    def exact_or_none(value: str | None) -> Any:
        return Literal.__getitem__(value) if value is not None else Literal[None]

    contacts = create_model(
        f"PartyContacts_{shape.contacts.phoneNumberCount}_{shape.contacts.emailAddressCount}_"
        f"{shape.contacts.websiteUrlCount}_{int(shape.contacts.contactNamePresent)}",
        __config__=_STRICT,
        contactName=(present_type(shape.contacts.contactNamePresent, NonEmptyText), ...),
        phoneNumbers=(
            Annotated[
                tuple[NonEmptyText, ...],
                Field(
                    min_length=shape.contacts.phoneNumberCount,
                    max_length=shape.contacts.phoneNumberCount,
                ),
            ],
            ...,
        ),
        emailAddresses=(
            Annotated[
                tuple[EmailText, ...],
                Field(
                    min_length=shape.contacts.emailAddressCount,
                    max_length=shape.contacts.emailAddressCount,
                ),
            ],
            ...,
        ),
        websiteUrls=(
            Annotated[
                tuple[NonEmptyText, ...],
                Field(
                    min_length=shape.contacts.websiteUrlCount,
                    max_length=shape.contacts.websiteUrlCount,
                ),
            ],
            ...,
        ),
    )
    party = create_model(
        f"Party_{seed.partyRole}_{int(shape.namePresent)}_{int(shape.addressPresent)}_"
        f"{int(shape.cityPresent)}_{int(shape.countryPresent)}",
        __config__=_STRICT,
        partyRole=(Literal.__getitem__(seed.partyRole), ...),
        name=(
            present_type(
                shape.namePresent,
                Annotated[
                    str,
                    StringConstraints(strip_whitespace=True, min_length=2, max_length=180),
                ],
            ),
            ...,
        ),
        address=(present_type(shape.addressPresent, NonEmptyText), ...),
        city=(exact_or_none(seed.targetLocality.city), ...),
        country=(exact_or_none(seed.targetLocality.country), ...),
        contactDetails=(contacts, ...),
    )
    return create_model("PartyCompletionOutput", __config__=_STRICT, party=(party, ...))


def _dynamic_cargo_output_model(seed: CargoLanguageGenerationSeed) -> type[BaseModel]:
    group_models: list[type[BaseModel]] = []
    for index, group in enumerate(seed.cargoGroups, start=1):
        contract = group.fieldContract
        description_type = NonEmptyText if contract.descriptionPresent else Literal[None]
        group_models.append(
            create_model(
                f"CargoLanguageGroup_{index}",
                __config__=_STRICT,
                groupId=(Literal.__getitem__(group.groupId), ...),
                description=(description_type, ...),
                additionalInformation=(
                    Annotated[
                        tuple[NonEmptyText, ...],
                        Field(
                            min_length=len(contract.additionalInformationSlots),
                            max_length=len(contract.additionalInformationSlots),
                        ),
                    ],
                    ...,
                ),
                marksAndNumbers=(
                    Annotated[
                        tuple[NonEmptyText, ...],
                        Field(
                            min_length=len(contract.marksAndNumbersSlots),
                            max_length=len(contract.marksAndNumbersSlots),
                        ),
                    ],
                    ...,
                ),
                handlingInstructions=(
                    Annotated[
                        tuple[NonEmptyText, ...],
                        Field(
                            min_length=len(contract.handlingInstructionSlots),
                            max_length=len(contract.handlingInstructionSlots),
                        ),
                    ],
                    ...,
                ),
            )
        )
    item_type = (
        group_models[0] if len(group_models) == 1 else Union.__getitem__(tuple(group_models))
    )
    cargo_groups_type = cast(Any, Annotated)[
        tuple.__class_getitem__((item_type, Ellipsis)),
        Field(min_length=len(group_models), max_length=len(group_models)),
    ]
    return create_model(
        "CargoLanguageCompletionOutput",
        __config__=_STRICT,
        cargoGroups=(cargo_groups_type, ...),
    )


async def _call_attempt(
    *,
    attempt: int,
    schema_mode: Literal["static", "topology_constrained"],
    stage: Literal["party_identity", "cargo_language"],
    seed_payload: dict[str, JsonValue],
    output_type: type[BaseModel],
    canonical_output_type: type[BaseModel],
    validator: Callable[[BaseModel], dict[str, bool]],
    model: OpenAIResponsesModel,
    settings: OpenAIResponsesModelSettings,
    limiter: ConcurrencyLimiter,
    system_prompt: str,
    max_output_tokens: int,
    pricing: LinguisticProbePricingConfig,
) -> tuple[LinguisticAttemptRecord, JsonValue]:
    output_schema = output_type.model_json_schema(mode="validation")
    output_schema_sha256 = sha256_bytes(canonical_json_bytes(output_schema))
    user_prompt = json.dumps(seed_payload, ensure_ascii=False, separators=(",", ":"))
    agent = Agent[object, BaseModel](
        model,
        output_type=NativeOutput(
            output_type,
            name=f"synthetic_bill_of_lading_{stage}",
            description=(
                "Return only the provider-constrained synthetic values requested by the input "
                "field and topology contract."
            ),
            strict=True,
        ),
        system_prompt=system_prompt,
        model_settings=settings,
        retries=0,
        max_concurrency=limiter,
        name=f"synthetic-bill-of-lading-{stage}",
    )
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    with capture_run_messages() as captured:
        try:
            result = await agent.run(
                user_prompt,
                usage_limits=UsageLimits(
                    request_limit=1,
                    output_tokens_limit=max_output_tokens,
                ),
            )
            responses = tuple(
                message for message in result.new_messages() if isinstance(message, ModelResponse)
            )
            if len(responses) != 1 or result.usage.requests != 1:
                raise RuntimeError("linguistic attempt produced other than one model response")
            raw_output = cast(dict[str, JsonValue], result.output.model_dump(mode="json"))
            try:
                canonical = canonical_output_type.model_validate(
                    result.output.model_dump(mode="python"), strict=True
                )
                if stage == "cargo_language":
                    cargo_seed = CargoLanguageGenerationSeed.model_validate_json(
                        canonical_json_bytes(seed_payload), strict=True
                    )
                    canonical = normalize_cargo_language_output(
                        seed=cargo_seed,
                        output=cast(CargoLanguageGenerationOutput, canonical),
                    )
                canonical_output = cast(dict[str, JsonValue], canonical.model_dump(mode="json"))
                checks = validator(canonical)
                validation_error: Exception | None = None
            except Exception as error:
                canonical_output = raw_output
                checks = None
                validation_error = error
            completed_at = datetime.now(UTC)
            successful = validation_error is None and checks is not None and all(checks.values())
            return (
                LinguisticAttemptRecord(
                    attempt=attempt,
                    schemaMode=schema_mode,
                    outputSchemaSha256=output_schema_sha256,
                    userPromptSha256=sha256_bytes(user_prompt.encode()),
                    startedAt=started_at,
                    completedAt=completed_at,
                    durationMs=(time.perf_counter() - started) * 1000.0,
                    status="success" if successful else "validation_failed",
                    output=canonical_output,
                    checks=checks,
                    usage=usage_receipt(responses, pricing),
                    errorType=(
                        type(validation_error).__name__ if validation_error is not None else None
                    ),
                    errorMessage=(str(validation_error) if validation_error is not None else None),
                ),
                model_messages(captured),
            )
        except Exception as error:
            responses = tuple(message for message in captured if isinstance(message, ModelResponse))
            completed_at = datetime.now(UTC)
            return (
                LinguisticAttemptRecord(
                    attempt=attempt,
                    schemaMode=schema_mode,
                    outputSchemaSha256=output_schema_sha256,
                    userPromptSha256=sha256_bytes(user_prompt.encode()),
                    startedAt=started_at,
                    completedAt=completed_at,
                    durationMs=(time.perf_counter() - started) * 1000.0,
                    status="call_failed",
                    output=None,
                    checks=None,
                    usage=usage_receipt(responses, pricing),
                    errorType=type(error).__name__,
                    errorMessage=str(error),
                ),
                model_messages(captured),
            )


async def _run_unit(
    *,
    stage: Literal["party_identity", "cargo_language"],
    unit_id: str,
    seed: BaseModel,
    static_output_type: type[BaseModel],
    constrained_output_type: type[BaseModel],
    canonical_output_type: type[BaseModel],
    validator: Callable[[BaseModel], dict[str, bool]],
    model: OpenAIResponsesModel,
    settings: OpenAIResponsesModelSettings,
    limiter: ConcurrencyLimiter,
    system_prompt: str,
    max_output_tokens: int,
    attempts: int,
    pricing: LinguisticProbePricingConfig,
) -> LinguisticUnitArtifact:
    records: list[LinguisticAttemptRecord] = []
    transcripts: list[JsonValue] = []
    selected: LinguisticAttemptRecord | None = None
    for attempt in range(1, attempts + 1):
        schema_mode: Literal["static", "topology_constrained"] = (
            "static" if attempt == 1 else "topology_constrained"
        )
        record, transcript = await _call_attempt(
            attempt=attempt,
            schema_mode=schema_mode,
            stage=stage,
            seed_payload=cast(dict[str, JsonValue], seed.model_dump(mode="json")),
            output_type=(static_output_type if attempt == 1 else constrained_output_type),
            canonical_output_type=canonical_output_type,
            validator=validator,
            model=model,
            settings=settings,
            limiter=limiter,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            pricing=pricing,
        )
        records.append(record)
        transcripts.append(transcript)
        if record.status == "success":
            selected = record
            break
    return LinguisticUnitArtifact(
        schemaVersion=1,
        stage=stage,
        unitId=unit_id,
        attempts=tuple(records),
        transcripts=tuple(transcripts),
        selectedAttempt=selected.attempt if selected is not None else None,
        selectedOutputSha256=(
            sha256_bytes(canonical_json_bytes(selected.output)) if selected is not None else None
        ),
        status="success" if selected is not None else "failed",
    )


def _selected_output(unit: LinguisticUnitArtifact) -> dict[str, JsonValue]:
    if unit.selectedAttempt is None:
        raise ValueError(f"linguistic unit has no successful output: {unit.unitId}")
    for attempt in unit.attempts:
        if attempt.attempt == unit.selectedAttempt:
            return cast(dict[str, JsonValue], attempt.output)
    raise RuntimeError("linguistic unit selected attempt is absent")


def _validate_persisted_output[OutputModel: BaseModel](
    model_type: type[OutputModel], payload: Mapping[str, JsonValue]
) -> OutputModel:
    """Strictly rehydrate a persisted JSON payload at the JSON boundary."""

    return model_type.model_validate_json(canonical_json_bytes(payload), strict=True)


def _project_party(output: PartyCompletionOutput, *, same_as: str | None) -> dict[str, Any]:
    party = output.party
    projected: dict[str, Any] = {}
    if same_as is not None:
        projected["sameAs"] = same_as
    for field in ("name", "address", "city", "country"):
        value = getattr(party, field)
        if value is not None:
            projected[field] = value
    contacts: dict[str, Any] = {}
    if party.contactDetails.contactName is not None:
        contacts["contactName"] = party.contactDetails.contactName
    for field in ("phoneNumbers", "emailAddresses", "websiteUrls"):
        values = getattr(party.contactDetails, field)
        if values:
            contacts[field] = list(values)
    if contacts:
        projected["contactDetails"] = contacts
    return projected


def _apply_linguistic_outputs(
    *,
    upstream: SemanticCompletionPlanRow,
    document_plan: DocumentLinguisticPlan,
    party_units: Mapping[str, LinguisticUnitArtifact],
    cargo_unit: LinguisticUnitArtifact,
) -> dict[str, Any]:
    target = copy.deepcopy(upstream.target)
    patch = cast(dict[str, Any], target["documentPatch"])
    parties = cast(dict[str, Any], patch.get("parties") or {})
    for projection in document_plan.partyProjections:
        current = parties[projection.role]
        if projection.unitId is None:
            continue
        unit = party_units[projection.unitId]
        output = _validate_persisted_output(PartyCompletionOutput, _selected_output(unit))
        replacement = _project_party(output, same_as=projection.sameAsReference)
        if projection.role == "notifyParties":
            cast(list[dict[str, Any]], current)[projection.occurrence] = replacement
        else:
            parties[projection.role] = replacement
    if parties:
        patch["parties"] = parties

    cargo_output = _validate_persisted_output(
        CargoLanguageGenerationOutput, _selected_output(cargo_unit)
    )
    groups = {row["groupId"]: row for row in cast(list[dict[str, Any]], patch["cargoGroups"])}
    for generated in cargo_output.cargoGroups:
        group = groups[generated.groupId]
        if generated.description is None:
            group.pop("description", None)
        else:
            group["description"] = generated.description
        for field, values in (
            (
                "additionalInformation",
                generated.additionalInformation,
            ),
            ("marksAndNumbers", generated.marksAndNumbers),
            ("handlingInstructions", generated.handlingInstructions),
        ):
            if values:
                group[field] = list(values)
            else:
                group.pop(field, None)
    canonical = BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
        document_id=upstream.scenario_id,
        target=target,
    )
    mismatches = printed_topology_mismatches(upstream.target, canonical)
    if mismatches:
        detail = ", ".join(row.path for row in mismatches[:8])
        raise RuntimeError(f"linguistic completion changed printed topology: {detail}")
    return canonical


def _resolve_pinned_file(
    project_root: Path, configured_path: str, expected_sha256: str, *, label: str
) -> Path:
    path = resolve_config_path(project_root, configured_path)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} differs from its configured pin: {path}")
    return path.resolve(strict=True)


def _validate_completion_run(
    project_root: Path, config: SynthesisLinguisticCompletionConfig
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


def _load_completion_plans(path: Path, *, records: int) -> tuple[SemanticCompletionPlanRow, ...]:
    rows: list[SemanticCompletionPlanRow] = []
    identifiers: set[str] = set()
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip() or not raw.endswith(b"\n"):
                raise ValueError(f"completion plan row {line_number} is blank or unterminated")
            try:
                row = SemanticCompletionPlanRow.model_validate_json(raw, strict=True)
            except ValueError as error:
                raise ValueError(f"completion plan row {line_number} is invalid") from error
            if row.base_document_id in identifiers:
                raise ValueError(f"completion plans repeat {row.base_document_id}")
            identifiers.add(row.base_document_id)
            rows.append(row)
    if len(rows) != records:
        raise ValueError("completion plan count differs from configuration")
    return tuple(rows)


def _unit_path(document_index: int, unit_id: str) -> str:
    return f"generation/units/{document_index + 1:05d}/{unit_id}.json"


def _document_plan_path(document_index: int) -> str:
    return f"planning/documents/{document_index + 1:05d}.json"


def _document_result_path(document_index: int) -> str:
    return f"generation/documents/{document_index + 1:05d}.json"


def _load_existing_unit(path: Path) -> LinguisticUnitArtifact | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"linguistic unit resume path is not a regular file: {path}")
    return LinguisticUnitArtifact.model_validate_json(read_regular_file_bytes(path), strict=True)


def _resume_unit_passes_current_contract(
    unit: LinguisticUnitArtifact,
    *,
    stage: Literal["party_identity", "cargo_language"],
    unit_id: str,
    output_type: type[BaseModel],
    validator: Callable[[BaseModel], dict[str, bool]],
) -> bool:
    """Return whether one immutable prior success is reusable under today's validator.

    A prior provider response is reused only when its identity and stage match and its selected
    payload passes the current strict Pydantic model and semantic contract.  Failed or stale
    successes are regenerated; they are never silently carried into the new run.
    """

    if unit.status != "success" or unit.stage != stage or unit.unitId != unit_id:
        return False
    try:
        output = _validate_persisted_output(output_type, _selected_output(unit))
        checks = validator(output)
    except (RuntimeError, ValueError):
        return False
    return bool(checks) and all(checks.values())


def _validate_resume_run(
    *,
    project_root: Path,
    config: SynthesisLinguisticCompletionConfig,
    document_plans: Sequence[DocumentLinguisticPlan],
    party_prompt_bytes: bytes,
    cargo_prompt_bytes: bytes,
    base_schemas: Mapping[str, Any],
    source_inventory: SourceSensitiveInventory,
    document_count: int,
) -> Path | None:
    configured = config.inputs.resume_run
    if configured is None:
        return None
    root = resolve_config_path(project_root, configured.path)
    commit = root / "_COMMIT.json"
    if root.is_symlink() or not root.is_dir() or sha256_file(commit) != configured.commit_sha256:
        raise ValueError(f"linguistic resume run differs from its configured pin: {root}")
    StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=configured.transaction_sha256,
    ).validate_committed_run()
    expected_payloads = {
        "prompts/party.md": party_prompt_bytes,
        "prompts/cargo.md": cargo_prompt_bytes,
        "schema/party-output.schema.json": json_artifact_bytes(base_schemas["party"]),
        "schema/cargo-output.schema.json": json_artifact_bytes(base_schemas["cargo"]),
    }
    for relative, expected in expected_payloads.items():
        if read_regular_file_bytes(root / relative) != expected:
            raise ValueError(
                f"linguistic resume run has a different immutable contract: {relative}"
            )
    summary = cast(
        dict[str, JsonValue],
        json.loads(read_regular_file_bytes(root / "generation/summary.json")),
    )
    if summary.get("status") != "incomplete" or summary.get("documents") != document_count:
        raise ValueError("linguistic resume source is not the matching incomplete run")
    if summary.get("sourceSensitiveInventorySha256") != source_inventory.digest:
        raise ValueError("linguistic resume source used a different sensitive-value inventory")
    prior_plans = tuple(
        cast(dict[str, JsonValue], json.loads(raw))
        for raw in read_regular_file_bytes(root / "planning/document-plans.jsonl").splitlines()
        if raw
    )
    prior_identities = tuple(
        (
            row.get("documentIndex"),
            row.get("baseDocumentId"),
            row.get("scenarioId"),
            row.get("upstreamTargetSha256"),
        )
        for row in prior_plans
    )
    current_identities = tuple(
        (row.documentIndex, row.baseDocumentId, row.scenarioId, row.upstreamTargetSha256)
        for row in document_plans
    )
    if prior_identities != current_identities:
        raise ValueError("linguistic resume source has a different document lineage or order")
    return root.resolve(strict=True)


def _seed_reusable_units(
    *,
    resume_root: Path | None,
    staged: StagedArtifactRun,
    document_plans: Sequence[DocumentLinguisticPlan],
    source_inventory: SourceSensitiveInventory,
) -> tuple[LinguisticUnitArtifact, ...]:
    if resume_root is None:
        return ()
    reused: list[LinguisticUnitArtifact] = []
    for index, plan in enumerate(document_plans):
        prior_plan = cast(
            dict[str, JsonValue],
            json.loads(read_regular_file_bytes(resume_root / _document_plan_path(index))),
        )
        raw_party_units = prior_plan.get("partyUnits")
        if not isinstance(raw_party_units, list) or any(
            not isinstance(row, dict) for row in raw_party_units
        ):
            raise ValueError("linguistic resume source has invalid party-unit plans")
        prior_party_units = {
            cast(str, row.get("unitId")): row
            for row in cast(list[dict[str, JsonValue]], raw_party_units)
        }
        for unit_plan in plan.partyUnits:
            prior_unit_plan = prior_party_units.get(unit_plan.unitId)
            if prior_unit_plan is None or prior_unit_plan.get("seed") != unit_plan.seed.model_dump(
                mode="json"
            ):
                continue
            relative = _unit_path(index, unit_plan.unitId)
            payload = read_regular_file_bytes(resume_root / relative)
            unit = LinguisticUnitArtifact.model_validate_json(payload, strict=True)
            if _resume_unit_passes_current_contract(
                unit,
                stage="party_identity",
                unit_id=unit_plan.unitId,
                output_type=PartyCompletionOutput,
                validator=_party_validator(unit_plan.seed, source_inventory),
            ):
                staged.publish_bytes(relative, payload)
                reused.append(unit)
        relative = _unit_path(index, "cargo")
        payload = read_regular_file_bytes(resume_root / relative)
        unit = LinguisticUnitArtifact.model_validate_json(payload, strict=True)
        if prior_plan.get("cargoSeed") == plan.cargoSeed.model_dump(
            mode="json"
        ) and _resume_unit_passes_current_contract(
            unit,
            stage="cargo_language",
            unit_id="cargo",
            output_type=CargoLanguageGenerationOutput,
            validator=_cargo_validator(plan.cargoSeed),
        ):
            staged.publish_bytes(relative, payload)
            reused.append(unit)
    return tuple(reused)


async def _run_completion_async(
    *,
    config: SynthesisLinguisticCompletionConfig,
    staged: StagedArtifactRun,
    plans: Sequence[SemanticCompletionPlanRow],
    document_plans: Sequence[DocumentLinguisticPlan],
    party_prompt: str,
    cargo_prompt: str,
    source_inventory: SourceSensitiveInventory,
    project_root: Path,
) -> tuple[LinguisticCompletionDocumentRecord, ...]:
    client = AsyncOpenAI(
        api_key=load_openai_key(project_root, config.environment_file),
        max_retries=config.provider.transport_max_retries,
        timeout=config.provider.request_timeout_seconds,
    )
    model = OpenAIResponsesModel(
        config.provider.model, provider=OpenAIProvider(openai_client=client)
    )
    limiter = ConcurrencyLimiter(
        config.workflow.max_concurrent_requests,
        name="synthetic-linguistic-provider-requests",
    )
    document_limiter = asyncio.Semaphore(config.workflow.max_concurrent_documents)
    progress_lock = asyncio.Lock()
    progress_started = time.perf_counter()
    processed_documents = 0
    successful_documents = 0
    party_settings = openai_responses_settings(
        config.provider,
        max_output_tokens=config.stage_limits.party_max_output_tokens,
    )
    cargo_settings = openai_responses_settings(
        config.provider,
        max_output_tokens=config.stage_limits.cargo_max_output_tokens,
    )

    async def process_document(index: int) -> LinguisticCompletionDocumentRecord:
        nonlocal processed_documents, successful_documents
        plan = plans[index]
        document_plan = document_plans[index]
        async with document_limiter:
            party_artifacts: dict[str, LinguisticUnitArtifact] = {}
            party_tasks: dict[str, asyncio.Task[LinguisticUnitArtifact]] = {}
            for unit in document_plan.partyUnits:
                relative = _unit_path(index, unit.unitId)
                existing = _load_existing_unit(staged.stage_root / relative)
                if existing is not None:
                    party_artifacts[unit.unitId] = existing
                    continue
                party_tasks[unit.unitId] = asyncio.create_task(
                    _run_unit(
                        stage="party_identity",
                        unit_id=unit.unitId,
                        seed=unit.seed,
                        static_output_type=PartyCompletionOutput,
                        constrained_output_type=_dynamic_party_output_model(unit.seed),
                        canonical_output_type=PartyCompletionOutput,
                        validator=_party_validator(unit.seed, source_inventory),
                        model=model,
                        settings=party_settings,
                        limiter=limiter,
                        system_prompt=party_prompt,
                        max_output_tokens=config.stage_limits.party_max_output_tokens,
                        attempts=config.workflow.semantic_validation_attempts,
                        pricing=config.provider.pricing,
                    )
                )
            cargo_relative = _unit_path(index, "cargo")
            cargo_artifact = _load_existing_unit(staged.stage_root / cargo_relative)
            cargo_task: asyncio.Task[LinguisticUnitArtifact] | None = None
            if cargo_artifact is None:
                cargo_task = asyncio.create_task(
                    _run_unit(
                        stage="cargo_language",
                        unit_id="cargo",
                        seed=document_plan.cargoSeed,
                        static_output_type=CargoLanguageGenerationOutput,
                        constrained_output_type=_dynamic_cargo_output_model(
                            document_plan.cargoSeed
                        ),
                        canonical_output_type=CargoLanguageGenerationOutput,
                        validator=_cargo_validator(document_plan.cargoSeed),
                        model=model,
                        settings=cargo_settings,
                        limiter=limiter,
                        system_prompt=cargo_prompt,
                        max_output_tokens=config.stage_limits.cargo_max_output_tokens,
                        attempts=config.workflow.semantic_validation_attempts,
                        pricing=config.provider.pricing,
                    )
                )
            for unit_id, task in party_tasks.items():
                artifact = await task
                staged.publish_json(_unit_path(index, unit_id), artifact.model_dump(mode="json"))
                party_artifacts[unit_id] = artifact
            if cargo_task is not None:
                cargo_artifact = await cargo_task
                staged.publish_json(cargo_relative, cargo_artifact.model_dump(mode="json"))
            assert cargo_artifact is not None
            success = all(row.status == "success" for row in party_artifacts.values()) and (
                cargo_artifact.status == "success"
            )
            if success:
                target = _apply_linguistic_outputs(
                    upstream=plan,
                    document_plan=document_plan,
                    party_units=party_artifacts,
                    cargo_unit=cargo_artifact,
                )
                blockers = tuple(sorted(set(plan.remaining_blockers) - _GENERIC_BLOCKERS_RESOLVED))
                record = LinguisticCompletionDocumentRecord(
                    schemaVersion=1,
                    baseDocumentId=plan.base_document_id,
                    scenarioId=plan.scenario_id,
                    upstreamTargetSha256=plan.target_sha256,
                    status="success",
                    partyUnitPaths=tuple(
                        _unit_path(index, unit.unitId) for unit in document_plan.partyUnits
                    ),
                    cargoUnitPath=cargo_relative,
                    target=target,
                    targetSha256=sha256_bytes(canonical_json_bytes(target)),
                    remainingBlockers=blockers,
                    trainingEligible=False,
                )
            else:
                record = LinguisticCompletionDocumentRecord(
                    schemaVersion=1,
                    baseDocumentId=plan.base_document_id,
                    scenarioId=plan.scenario_id,
                    upstreamTargetSha256=plan.target_sha256,
                    status="failed",
                    partyUnitPaths=tuple(
                        _unit_path(index, unit.unitId) for unit in document_plan.partyUnits
                    ),
                    cargoUnitPath=cargo_relative,
                    target=None,
                    targetSha256=None,
                    remainingBlockers=plan.remaining_blockers,
                    trainingEligible=False,
                )
            staged.publish_json(_document_result_path(index), record.model_dump(mode="json"))
            async with progress_lock:
                processed_documents += 1
                successful_documents += record.status == "success"
                elapsed = time.perf_counter() - progress_started
                rate = processed_documents / elapsed if elapsed else 0.0
                remaining = len(plans) - processed_documents
                print(
                    json.dumps(
                        {
                            "command": "run-linguistic-completion",
                            "phase": "party_and_cargo_generation",
                            "processed_documents": processed_documents,
                            "remaining_documents": remaining,
                            "successful_documents": successful_documents,
                            "failed_documents": (processed_documents - successful_documents),
                            "elapsed_seconds": round(elapsed, 3),
                            "throughput_documents_per_hour": round(rate * 3600.0, 3),
                            "eta_seconds": round(remaining / rate, 3) if rate else None,
                            "status": "progress",
                        },
                        allow_nan=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return record

    results: list[LinguisticCompletionDocumentRecord | None] = [None] * len(plans)
    next_index = 0
    index_lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal next_index
        while True:
            async with index_lock:
                if next_index >= len(plans):
                    return
                index = next_index
                next_index += 1
            results[index] = await process_document(index)

    try:
        await asyncio.gather(
            *(worker() for _ in range(min(config.workflow.max_concurrent_documents, len(plans))))
        )
    finally:
        await client.close()
    if any(row is None for row in results):
        raise RuntimeError("linguistic worker pool returned an incomplete result set")
    return tuple(cast(LinguisticCompletionDocumentRecord, row) for row in results)


def _all_units_from_root(
    root: Path, document_plans: Sequence[DocumentLinguisticPlan]
) -> tuple[LinguisticUnitArtifact, ...]:
    rows: list[LinguisticUnitArtifact] = []
    for index, plan in enumerate(document_plans):
        for unit in plan.partyUnits:
            path = root / _unit_path(index, unit.unitId)
            rows.append(
                LinguisticUnitArtifact.model_validate_json(
                    read_regular_file_bytes(path), strict=True
                )
            )
        rows.append(
            LinguisticUnitArtifact.model_validate_json(
                read_regular_file_bytes(root / _unit_path(index, "cargo")),
                strict=True,
            )
        )
    return tuple(rows)


def _all_units(
    staged: StagedArtifactRun, document_plans: Sequence[DocumentLinguisticPlan]
) -> tuple[LinguisticUnitArtifact, ...]:
    return _all_units_from_root(staged.stage_root, document_plans)


@dataclass(frozen=True, slots=True)
class LinguisticUsageTotals:
    requests: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    visible_output_tokens: int = 0
    estimated_cost_usd: Decimal = Decimal(0)

    @classmethod
    def from_receipt(cls, usage: LinguisticUsageReceipt) -> LinguisticUsageTotals:
        return cls(
            requests=usage.requests,
            input_tokens=usage.inputTokens,
            cache_read_tokens=usage.cacheReadTokens,
            cache_write_tokens=usage.cacheWriteTokens,
            output_tokens=usage.outputTokens,
            reasoning_tokens=usage.reasoningTokens,
            visible_output_tokens=usage.visibleOutputTokens,
            estimated_cost_usd=usage.estimatedCostUsd,
        )

    def __add__(self, other: LinguisticUsageTotals) -> LinguisticUsageTotals:
        return LinguisticUsageTotals(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            visible_output_tokens=self.visible_output_tokens + other.visible_output_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
        )

    def __sub__(self, other: LinguisticUsageTotals) -> LinguisticUsageTotals:
        result = LinguisticUsageTotals(
            requests=self.requests - other.requests,
            input_tokens=self.input_tokens - other.input_tokens,
            cache_read_tokens=self.cache_read_tokens - other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens - other.cache_write_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            reasoning_tokens=self.reasoning_tokens - other.reasoning_tokens,
            visible_output_tokens=self.visible_output_tokens - other.visible_output_tokens,
            estimated_cost_usd=self.estimated_cost_usd - other.estimated_cost_usd,
        )
        if any(
            value < 0
            for value in (
                result.requests,
                result.input_tokens,
                result.cache_read_tokens,
                result.cache_write_tokens,
                result.output_tokens,
                result.reasoning_tokens,
                result.visible_output_tokens,
                result.estimated_cost_usd,
            )
        ):
            raise ValueError("linguistic usage subtraction produced a negative value")
        return result


def linguistic_unit_usage_totals(
    units: Sequence[LinguisticUnitArtifact],
) -> LinguisticUsageTotals:
    total = LinguisticUsageTotals()
    for unit in units:
        for attempt in unit.attempts:
            total += LinguisticUsageTotals.from_receipt(attempt.usage)
    return total


def linguistic_summary_incurred_usage(
    summary: Mapping[str, Any],
) -> LinguisticUsageTotals | None:
    names = {
        "requests": "incurredRequests",
        "input_tokens": "incurredInputTokens",
        "cache_read_tokens": "incurredCacheReadTokens",
        "cache_write_tokens": "incurredCacheWriteTokens",
        "output_tokens": "incurredOutputTokens",
        "reasoning_tokens": "incurredReasoningTokens",
        "visible_output_tokens": "incurredVisibleOutputTokens",
    }
    present = {name for name in (*names.values(), "incurredEstimatedCostUsd") if name in summary}
    if not present:
        return None
    expected = {*names.values(), "incurredEstimatedCostUsd"}
    if present != expected:
        raise ValueError("resume summary has a partial incurred-usage accounting contract")
    values: dict[str, int] = {}
    for field, name in names.items():
        value = summary[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"resume summary has invalid {name}")
        values[field] = value
    cost_value = summary["incurredEstimatedCostUsd"]
    if not isinstance(cost_value, str):
        raise ValueError("resume summary has invalid incurredEstimatedCostUsd")
    cost = Decimal(cost_value)
    if not cost.is_finite() or cost < 0:
        raise ValueError("resume summary has invalid incurredEstimatedCostUsd")
    return LinguisticUsageTotals(**values, estimated_cost_usd=cost)


def linguistic_usage_summary_fields(
    *,
    retained: LinguisticUsageTotals,
    incremental: LinguisticUsageTotals,
    incurred: LinguisticUsageTotals,
) -> dict[str, JsonValue]:
    """Render the three non-interchangeable provider-usage views in one contract."""

    return {
        "usageAccounting": "retained_incremental_and_all_attempt_resume_lineage_v1",
        "requests": retained.requests,
        "inputTokens": retained.input_tokens,
        "cacheReadTokens": retained.cache_read_tokens,
        "cacheWriteTokens": retained.cache_write_tokens,
        "outputTokens": retained.output_tokens,
        "reasoningTokens": retained.reasoning_tokens,
        "visibleOutputTokens": retained.visible_output_tokens,
        "estimatedCostUsd": str(retained.estimated_cost_usd),
        "incrementalRequests": incremental.requests,
        "incrementalInputTokens": incremental.input_tokens,
        "incrementalCacheReadTokens": incremental.cache_read_tokens,
        "incrementalCacheWriteTokens": incremental.cache_write_tokens,
        "incrementalOutputTokens": incremental.output_tokens,
        "incrementalReasoningTokens": incremental.reasoning_tokens,
        "incrementalVisibleOutputTokens": incremental.visible_output_tokens,
        "incrementalEstimatedCostUsd": str(incremental.estimated_cost_usd),
        "incurredRequests": incurred.requests,
        "incurredInputTokens": incurred.input_tokens,
        "incurredCacheReadTokens": incurred.cache_read_tokens,
        "incurredCacheWriteTokens": incurred.cache_write_tokens,
        "incurredOutputTokens": incurred.output_tokens,
        "incurredReasoningTokens": incurred.reasoning_tokens,
        "incurredVisibleOutputTokens": incurred.visible_output_tokens,
        "incurredEstimatedCostUsd": str(incurred.estimated_cost_usd),
    }


def _prior_incurred_usage(
    *,
    resume_root: Path | None,
    document_plans: Sequence[DocumentLinguisticPlan],
) -> LinguisticUsageTotals:
    if resume_root is None:
        return LinguisticUsageTotals()
    summary = cast(
        dict[str, Any],
        json.loads(read_regular_file_bytes(resume_root / "generation/summary.json")),
    )
    accounted = linguistic_summary_incurred_usage(summary)
    if accounted is not None:
        return accounted
    if summary.get("resumedFromRun") is not None or summary.get("derivedFromRun") is not None:
        raise ValueError(
            "resume lineage predates complete incurred-usage accounting; its discarded ancestor "
            "attempts cannot be proven from the retained artifact"
        )
    return linguistic_unit_usage_totals(_all_units_from_root(resume_root, document_plans))


def _report(summary: Mapping[str, Any]) -> str:
    locality_only = summary["localityOnlyPartyProjections"]
    token_totals = (
        f"{summary['inputTokens']:,} / {summary['outputTokens']:,} / {summary['reasoningTokens']:,}"
    )
    return f"""# Integrated B/L linguistic completion

- Documents: **{summary["documents"]:,}**
- Successfully completed: **{summary["successfulDocuments"]:,}**
- Failed after configured fresh attempts: **{summary["failedDocuments"]:,}**
- Party generation units: **{summary["partyUnits"]:,}**
- Deterministic sameAs references: **{summary["sameAsReferences"]:,}**
- Locality-only party projections requiring no model call: **{locality_only:,}**
- Reused explicit notify identities: **{summary["reusedNotifyIdentities"]:,}**
- Cargo generation units: **{summary["cargoUnits"]:,}**
- Reused prior successful units: **{summary["reusedUnits"]:,}**
- Provider requests represented by retained unit artifacts: **{summary["requests"]:,}**
- New provider requests in this run: **{summary["incrementalRequests"]:,}**
- All provider requests incurred across the resume lineage: **{summary["incurredRequests"]:,}**
- First-attempt unit acceptance: **{summary["firstAttemptAcceptedUnits"]:,} / {summary["units"]:,}**
- Retried units: **{summary["retriedUnits"]:,}**
- Input / output / reasoning tokens: **{token_totals}**
- Estimated cost represented by retained unit artifacts: **${summary["estimatedCostUsd"]}**
- Incremental provider cost in this run: **${summary["incrementalEstimatedCostUsd"]}**
- All-attempt estimated cost across the resume lineage: **${summary["incurredEstimatedCostUsd"]}**
- Concurrent wall time: **{summary["wallSeconds"]:.3f} seconds**
- Throughput: **{summary["throughputDocumentsPerHour"]:.2f} documents/hour**

Every successful target passed the relation-v5 schema and relational inverse after party and cargo
projection.  Provider-native strict JSON Schema was used for every call; a failed first semantic
validation receives a fresh topology-constrained attempt with no conversational history.  All
attempts, provider-visible messages, token usage, and outputs are retained.

These outputs are **not training records**.  Categorical printed-surface realization and raw-OCR
patching remain explicit downstream work, and `trainingEligible` stays false.
"""


def run_linguistic_completion(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisLinguisticCompletionConfig,
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
    party_prompt_path = _resolve_pinned_file(
        project_root,
        config.prompts.party_identity.path,
        config.prompts.party_identity.sha256,
        label="party prompt",
    )
    cargo_prompt_path = _resolve_pinned_file(
        project_root,
        config.prompts.cargo_language.path,
        config.prompts.cargo_language.sha256,
        label="cargo prompt",
    )
    plans = _load_completion_plans(plans_path, records=config.inputs.completion_plans.records)
    document_plans = tuple(
        build_document_linguistic_plan(index=index, plan=plan) for index, plan in enumerate(plans)
    )
    source_inventory = _load_source_sensitive_inventory(
        source_path,
        records=config.inputs.source_corpus.records,
        target_field=config.inputs.source_target_field,
        target_schema=config.inputs.source_target_schema,
    )
    party_prompt_bytes = read_regular_file_bytes(party_prompt_path)
    cargo_prompt_bytes = read_regular_file_bytes(cargo_prompt_path)
    try:
        party_prompt = party_prompt_bytes.decode("utf-8")
        cargo_prompt = cargo_prompt_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("linguistic prompt is not valid UTF-8") from error
    base_schemas = {
        "party": PartyCompletionOutput.model_json_schema(mode="validation"),
        "cargo": CargoLanguageGenerationOutput.model_json_schema(mode="validation"),
    }
    plan_payload = b"".join(
        canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in document_plans
    )
    resume_root = _validate_resume_run(
        project_root=project_root,
        config=config,
        document_plans=document_plans,
        party_prompt_bytes=party_prompt_bytes,
        cargo_prompt_bytes=cargo_prompt_bytes,
        base_schemas=base_schemas,
        source_inventory=source_inventory,
        document_count=len(plans),
    )
    transaction = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "completionRunTransactionSha256": (
            config.inputs.semantic_completion_run.transaction_sha256
        ),
        "completionPlansSha256": config.inputs.completion_plans.sha256,
        "sourceCorpusSha256": config.inputs.source_corpus.sha256,
        "sourceSensitiveInventorySha256": source_inventory.digest,
        "partyPromptSha256": config.prompts.party_identity.sha256,
        "cargoPromptSha256": config.prompts.cargo_language.sha256,
        "baseSchemasSha256": sha256_bytes(canonical_json_bytes(base_schemas)),
        "documentPlansSha256": sha256_bytes(plan_payload),
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "partyImplementationSha256": sha256_file(
            Path(party_identity_module.__file__).resolve(strict=True)
        ),
        "cargoImplementationSha256": sha256_file(
            Path(cargo_language_module.__file__).resolve(strict=True)
        ),
        "runtime": {
            "model": config.provider.model,
            "reasoningEffort": config.provider.reasoning_effort,
            "pydanticAiVersion": version("pydantic-ai-slim"),
            "openaiVersion": version("openai"),
        },
        "resumeRun": (
            {
                "path": config.inputs.resume_run.path,
                "commitSha256": config.inputs.resume_run.commit_sha256,
                "transactionSha256": config.inputs.resume_run.transaction_sha256,
            }
            if config.inputs.resume_run is not None
            else None
        ),
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
        "generation/targets.jsonl",
        "planning/document-plans.jsonl",
        "prompts/cargo.md",
        "prompts/party.md",
        "schema/cargo-output.schema.json",
        "schema/party-output.schema.json",
    ]
    for index, plan in enumerate(document_plans):
        expected.extend((_document_plan_path(index), _document_result_path(index)))
        expected.extend(_unit_path(index, unit.unitId) for unit in plan.partyUnits)
        expected.append(_unit_path(index, "cargo"))
    expected = sorted(expected)
    if staged.completed:
        staged.commit(
            expected_artifacts=expected,
            metadata={"documents": len(plans), "schema_version": 1},
        )
        completed_summary = cast(
            dict[str, JsonValue],
            json.loads(read_regular_file_bytes(staged.final_root / "generation/summary.json")),
        )
        if completed_summary["failedDocuments"]:
            raise IncompleteLinguisticCompletionError(
                f"linguistic completion has {completed_summary['failedDocuments']} failed documents"
            )
        return completed_summary
    staged.recover_interrupted_temporary_files()
    staged.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    staged.publish_bytes("prompts/party.md", party_prompt_bytes)
    staged.publish_bytes("prompts/cargo.md", cargo_prompt_bytes)
    staged.publish_bytes(
        "schema/party-output.schema.json", json_artifact_bytes(base_schemas["party"])
    )
    staged.publish_bytes(
        "schema/cargo-output.schema.json", json_artifact_bytes(base_schemas["cargo"])
    )
    staged.publish_bytes("planning/document-plans.jsonl", plan_payload)
    for index, document_plan in enumerate(document_plans):
        staged.publish_json(_document_plan_path(index), document_plan.model_dump(mode="json"))
    reused_units = _seed_reusable_units(
        resume_root=resume_root,
        staged=staged,
        document_plans=document_plans,
        source_inventory=source_inventory,
    )
    wall_started = time.perf_counter()
    records = asyncio.run(
        _run_completion_async(
            config=config,
            staged=staged,
            plans=plans,
            document_plans=document_plans,
            party_prompt=party_prompt,
            cargo_prompt=cargo_prompt,
            source_inventory=source_inventory,
            project_root=project_root,
        )
    )
    units = _all_units(staged, document_plans)
    successful = tuple(row for row in records if row.status == "success")
    generated_names = tuple(
        _normalize_text(party.name)
        for unit in units
        if unit.stage == "party_identity" and unit.status == "success"
        if (
            party := _validate_persisted_output(PartyCompletionOutput, _selected_output(unit)).party
        ).name
    )
    generated_name_counts = Counter(generated_names)
    retained_usage = linguistic_unit_usage_totals(units)
    reused_usage = linguistic_unit_usage_totals(reused_units)
    incremental_usage = retained_usage - reused_usage
    incurred_usage = _prior_incurred_usage(
        resume_root=resume_root,
        document_plans=document_plans,
    ) + incremental_usage
    party_units = sum(len(row.partyUnits) for row in document_plans)
    same_as = sum(
        row.sameAsReference is not None for plan in document_plans for row in plan.partyProjections
    )
    locality_only = sum(
        row.unitId is None and row.sameAsReference is None
        for plan in document_plans
        for row in plan.partyProjections
    )
    reused_notify = sum(
        1
        for plan in document_plans
        for projection in plan.partyProjections
        if projection.role == "notifyParties"
        and projection.unitId is not None
        and projection.unitId
        not in {unit.unitId for unit in plan.partyUnits if unit.seed.partyRole == "notifyParties"}
    )
    wall_seconds = time.perf_counter() - wall_started
    summary: dict[str, JsonValue] = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "status": "complete" if len(successful) == len(records) else "incomplete",
        "documents": len(records),
        "successfulDocuments": len(successful),
        "failedDocuments": len(records) - len(successful),
        "partyUnits": party_units,
        "sameAsReferences": same_as,
        "localityOnlyPartyProjections": locality_only,
        "reusedNotifyIdentities": reused_notify,
        "cargoUnits": len(document_plans),
        "units": len(units),
        "reusedUnits": len(reused_units),
        "newUnits": len(units) - len(reused_units),
        "firstAttemptAcceptedUnits": sum(unit.attempts[0].status == "success" for unit in units),
        "retriedUnits": sum(len(unit.attempts) > 1 for unit in units),
        **linguistic_usage_summary_fields(
            retained=retained_usage,
            incremental=incremental_usage,
            incurred=incurred_usage,
        ),
        "resumedFromRun": resume_root.name if resume_root is not None else None,
        "wallSeconds": wall_seconds,
        "throughputDocumentsPerHour": (
            len(records) * 3600.0 / wall_seconds if wall_seconds else 0.0
        ),
        "maxConcurrentRequests": config.workflow.max_concurrent_requests,
        "maxConcurrentDocuments": config.workflow.max_concurrent_documents,
        "configuredGenerationSettings": (
            config.provider.generation_settings.model_dump(mode="json", exclude_none=True)
            if config.provider.generation_settings is not None
            else None
        ),
        "generatedPartyNames": len(generated_names),
        "uniqueGeneratedPartyNames": len(generated_name_counts),
        "repeatedGeneratedPartyNameValues": sum(
            count > 1 for count in generated_name_counts.values()
        ),
        "peakProcessRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "sourceSensitiveInventorySha256": source_inventory.digest,
        "transactionSha256": transaction_sha256,
        "trainingEligible": False,
    }
    staged.publish_bytes(
        "generation/results.jsonl",
        b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in records),
    )
    staged.publish_bytes(
        "generation/targets.jsonl",
        b"".join(
            canonical_json_bytes(
                {
                    "scenarioId": row.scenarioId,
                    "baseDocumentId": row.baseDocumentId,
                    "target": row.target,
                    "targetSha256": row.targetSha256,
                    "trainingEligible": False,
                    "remainingBlockers": row.remainingBlockers,
                }
            )
            + b"\n"
            for row in successful
        ),
    )
    staged.publish_json("generation/summary.json", summary)
    staged.publish_bytes("REPORT.md", _report(summary).encode("utf-8"))
    staged.commit(
        expected_artifacts=expected,
        metadata={"documents": len(records), "schema_version": 1},
    )
    if len(successful) != len(records):
        raise IncompleteLinguisticCompletionError(
            f"linguistic completion has {len(records) - len(successful)} failed documents"
        )
    return summary
