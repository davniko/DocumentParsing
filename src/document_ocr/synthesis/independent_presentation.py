"""Reviewed presentation constraints for independent flat synthesis.

The functions in this module strengthen wording requests and validate generated
values.  They do not classify identities, infer missing geography, select source
documents, call a provider, or modify public shipment facts.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from document_ocr.hashing import sha256_bytes
from document_ocr.synthesis.curated import digest, interpolate
from document_ocr.synthesis.curated_wording import WordingField, WordingRequest
from document_ocr.synthesis.independent_geography import IndependentGeographyGate

_COMPONENT = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_SHA256 = r"^[0-9a-f]{64}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("*", mode="after")
    @classmethod
    def nonblank_strings(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("review strings must not be blank")
        return value


@dataclass(frozen=True, slots=True)
class LocalityAssembly:
    """One reviewed generated owner of a locality within a postal assembly."""

    key: str
    component_keys: tuple[str, ...]
    owner_key: str
    locality: str
    source_evidence: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.component_keys, tuple)
            or not self.component_keys
            or any(not isinstance(key, str) or not key.strip() for key in self.component_keys)
        ):
            raise ValueError("locality assembly requires nonempty tuple component keys")
        if len(set(self.component_keys)) != len(self.component_keys):
            raise ValueError("locality assembly component keys must be distinct")
        if self.owner_key not in self.component_keys:
            raise ValueError("locality assembly owner must belong to its components")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.key, self.owner_key, self.locality, self.source_evidence)
        ):
            raise ValueError("locality assembly requires explicit source-evidenced ownership")


class SourceOccurrenceLocalityReview(_StrictModel):
    kind: Literal["source_occurrence"]
    source_document_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=_SHA256)
    role: str = Field(min_length=1)
    expression: str = Field(min_length=1)
    keys: tuple[str, ...] = Field(min_length=1)
    source_locality_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_keys(self) -> SourceOccurrenceLocalityReview:
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("reviewed locality keys must be distinct")
        if set(self.keys) != set(_COMPONENT.findall(self.expression)):
            raise ValueError("reviewed locality keys must exactly match its expression")
        return self


class SourceOnlyAssemblyLocalityReview(_StrictModel):
    kind: Literal["source_only_certificate"]
    source_document_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=_SHA256)
    assembly: str = Field(min_length=1)


LocalityReview = Annotated[
    SourceOccurrenceLocalityReview | SourceOnlyAssemblyLocalityReview,
    Field(discriminator="kind"),
]


@dataclass(frozen=True, slots=True)
class DistinctActor:
    related_key: str
    primary_expression: str
    source_evidence: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.related_key, self.primary_expression, self.source_evidence)
        ):
            raise ValueError("distinct actor requires explicit nonblank identity evidence")


class DistinctActorReview(_StrictModel):
    source_document_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=_SHA256)
    primary_role: str = Field(min_length=1)
    primary_expression: str = Field(min_length=1)
    related_key: str = Field(min_length=1)
    source_evidence: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class LocalityReviewIndex:
    file_sha256: str
    reviews: Mapping[str, LocalityReview]

    def for_document(self, source_document_id: str) -> LocalityReview | None:
        return self.reviews.get(source_document_id)


@dataclass(frozen=True, slots=True)
class DistinctActorReviewIndex:
    file_sha256: str
    reviews: Mapping[str, DistinctActorReview]

    def for_document(self, source_document_id: str) -> DistinctActorReview | None:
        return self.reviews.get(source_document_id)


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"review JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _pinned_review_object(payload: bytes, expected_sha256: str) -> tuple[str, dict[str, Any]]:
    if not re.fullmatch(_SHA256, expected_sha256):
        raise ValueError("expected review SHA-256 is invalid")
    actual = sha256_bytes(payload)
    if actual != expected_sha256:
        raise ValueError("review file SHA-256 differs")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("review file is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("review file must be a source-ID object")
    return actual, value


def load_locality_reviews(
    payload: bytes, *, expected_sha256: str
) -> LocalityReviewIndex:
    """Load the pinned reviewed locality mapping and bind each row to its source ID."""

    actual, raw = _pinned_review_object(payload, expected_sha256)
    reviews: dict[str, LocalityReview] = {}
    for source_id, row in raw.items():
        if not isinstance(source_id, str) or not source_id or not isinstance(row, dict):
            raise ValueError("locality reviews require nonempty source IDs and object rows")
        if row.get("source_document_id") != source_id:
            raise ValueError("locality review source ID disagrees with its mapping key")
        data = dict(row)
        if data.get("kind") == "source_occurrence":
            if not isinstance(data.get("keys"), list):
                raise ValueError("source-occurrence locality keys must be a JSON array")
            data["keys"] = tuple(data["keys"])
            review: LocalityReview = SourceOccurrenceLocalityReview.model_validate(data)
        elif data.get("kind") == "source_only_certificate":
            review = SourceOnlyAssemblyLocalityReview.model_validate(data)
        else:
            raise ValueError("unknown reviewed locality ownership authority")
        reviews[source_id] = review
    return LocalityReviewIndex(actual, MappingProxyType(reviews))


def load_distinct_actor_reviews(
    payload: bytes, *, expected_sha256: str
) -> DistinctActorReviewIndex:
    """Load pinned distinct-actor evidence and bind every row to its source ID."""

    actual, raw = _pinned_review_object(payload, expected_sha256)
    reviews: dict[str, DistinctActorReview] = {}
    for source_id, row in raw.items():
        if not isinstance(source_id, str) or not source_id or not isinstance(row, dict):
            raise ValueError("distinct-actor reviews require nonempty source IDs and object rows")
        if row.get("source_document_id") != source_id:
            raise ValueError("distinct-actor review source ID disagrees with its mapping key")
        reviews[source_id] = DistinctActorReview.model_validate(row)
    return DistinctActorReviewIndex(actual, MappingProxyType(reviews))


class LocalityOwnerFinding(_StrictModel):
    assembly: str
    owner: str
    problem: str
    locality: str
    component_counts: dict[str, int]


class DistinctActorFinding(_StrictModel):
    key: str
    problem: str
    primary_identity: str
    related_identity: str


def _reviewed_source(blueprint: Any, source_document_id: str, source_sha256: str) -> None:
    if blueprint.document_id != source_document_id:
        raise ValueError("presentation review belongs to another source document")
    if digest(blueprint.source.encode()) != source_sha256:
        raise ValueError("presentation review source bytes changed")


def source_locality_owner(
    component_examples: Mapping[str, str], source_locality: str
) -> str:
    """Return one explicit source occurrence; never guess a country or last slot."""

    literal = " ".join(source_locality.casefold().split())
    if not literal:
        raise ValueError("reviewed source locality spelling is required")
    pattern = re.compile(r"(?<!\w)" + re.escape(literal) + r"(?!\w)")
    hits = [
        key
        for key, text in component_examples.items()
        for _ in pattern.finditer(" ".join(text.casefold().split()))
    ]
    if len(hits) != 1:
        raise ValueError("source locality has no unique explicit component occurrence")
    return hits[0]


def apply_scoped_values(
    parent_values: Mapping[str, str],
    proposed_values: Mapping[str, str],
    allowed_keys: Set[str],
) -> dict[str, str]:
    """Apply exactly the authorized lexical leaves and preserve every other value."""

    if set(proposed_values) != set(allowed_keys) or not set(allowed_keys) <= parent_values.keys():
        raise ValueError("presentation correction differs from the exact authorized field set")
    if any(not isinstance(value, str) or not value.strip() for value in proposed_values.values()):
        raise ValueError("presentation correction must supply nonempty final text")
    result = copy.deepcopy(dict(parent_values))
    result.update(proposed_values)
    return result


def resolve_reviewed_locality_assemblies(
    blueprint: Any,
    scenario: Any,
    review: LocalityReview | None,
    *,
    geography: IndependentGeographyGate | None = None,
) -> tuple[LocalityAssembly, ...]:
    """Resolve only a caller-supplied review for this exact source snapshot."""

    if review is None:
        return ()
    if not isinstance(
        review, (SourceOccurrenceLocalityReview, SourceOnlyAssemblyLocalityReview)
    ):
        raise TypeError("locality resolver requires a typed reviewed authority")
    authority = review
    _reviewed_source(
        blueprint,
        authority.source_document_id,
        authority.source_sha256,
    )
    if isinstance(authority, SourceOccurrenceLocalityReview):
        bindings = [
            target
            for target in blueprint.contract.targets
            if target.path == authority.role + ".addressLine"
        ]
        if len(bindings) != 1 or bindings[0].expression != authority.expression:
            raise ValueError("reviewed locality assembly expression changed")
        variables = {variable.key: variable for variable in blueprint.contract.variables}
        examples: dict[str, str] = {}
        for key in authority.keys:
            if key not in variables or variables[key].kind != "postal":
                raise ValueError("reviewed locality component is not postal")
            if not variables[key].occurrences:
                raise ValueError("reviewed locality component has no source occurrence")
            examples[key] = variables[key].occurrences[0].text
        owner = source_locality_owner(examples, authority.source_locality_text)
        if authority.role not in scenario.party_localities:
            raise ValueError("reviewed locality role has no explicit sampled location")
        location = scenario.party_localities[authority.role]
        return (
            LocalityAssembly(
                key=authority.role,
                component_keys=authority.keys,
                owner_key=owner,
                locality=location.name,
                source_evidence=(
                    "Unique source locality occurrence: " + authority.source_locality_text
                ),
            ),
        )
    if geography is None:
        raise ValueError("source-only locality review requires explicit geography authority")
    certificate = geography.certificates.get(blueprint.document_id)
    if certificate is None:
        raise ValueError("source-only locality review has no source certificate")
    assemblies = [
        assembly for assembly in certificate.assemblies if assembly.key == authority.assembly
    ]
    if len(assemblies) != 1:
        raise ValueError("reviewed source-only locality assembly is absent or ambiguous")
    assembly = assemblies[0]
    if assembly.locality is None or assembly.locality.component is None:
        raise ValueError("reviewed source-only locality has no generated component owner")
    location = geography.location_for(blueprint, assembly, scenario)
    return (
        LocalityAssembly(
            key=assembly.key,
            component_keys=assembly.postal_keys,
            owner_key=assembly.locality.component,
            locality=location.name,
            source_evidence="Existing source-only postal assembly certificate: " + assembly.key,
        ),
    )


def validate_locality_owners(
    values: Mapping[str, str], assemblies: Sequence[LocalityAssembly]
) -> list[LocalityOwnerFinding]:
    findings: list[LocalityOwnerFinding] = []
    for assembly in assemblies:
        pattern = re.compile(
            r"(?<!\w)"
            + re.escape(" ".join(assembly.locality.casefold().split()))
            + r"(?!\w)"
        )
        try:
            counts = {
                key: len(pattern.findall(" ".join(values[key].casefold().split())))
                for key in assembly.component_keys
            }
        except KeyError as error:
            raise ValueError(f"locality validation is missing value: {error.args[0]}") from error
        if counts[assembly.owner_key] != 1 or sum(counts.values()) != 1:
            findings.append(
                LocalityOwnerFinding(
                    assembly=assembly.key,
                    owner=assembly.owner_key,
                    problem="Locality must occur once in its reviewed component only",
                    locality=assembly.locality,
                    component_counts=counts,
                )
            )
    return findings


def strengthen_locality_request(
    request: WordingRequest, assemblies: Sequence[LocalityAssembly]
) -> WordingRequest:
    by_key: dict[str, str] = {}
    context: list[str] = []
    keys = {field.key for field in request.fields}
    for assembly in assemblies:
        if not set(assembly.component_keys) <= keys:
            raise ValueError("locality assembly component is missing from the wording request")
        context.append(
            f"REVIEWED LOCALITY OWNER {assembly.key}: {{{assembly.owner_key}}} owns "
            f"{assembly.locality!r} exactly once as the locality component. "
            "Do not use that name inside street, building, site, estate or district names."
        )
        for key in assembly.component_keys:
            if key == assembly.owner_key:
                instruction = (
                    f" Locality owner: include {assembly.locality!r} exactly once as the "
                    "locality component, never inside a street, building, site, estate or "
                    "district name. Use different fictional names for those address parts."
                )
            else:
                instruction = (
                    f" Locality is owned by {{{assembly.owner_key}}}, not this component. "
                    f"Do not include {assembly.locality!r} here, including inside a street, "
                    "building, site, estate or district name. Return only this component."
                )
            if key in by_key and by_key[key] != instruction:
                raise ValueError("shared postal component has conflicting locality ownership")
            by_key[key] = instruction
    return replace(
        request,
        context=request.context + ("\n" + "\n".join(context) if context else ""),
        fields=tuple(
            replace(field, requirement=field.requirement + by_key[field.key])
            if field.key in by_key
            else field
            for field in request.fields
        ),
    )


def strengthen_individual_field(field: WordingField) -> WordingField:
    """Use the existing certified identity kind; never classify from spelling."""

    if field.identity_kind != "individual":
        return field
    return replace(
        field,
        requirement=field.requirement
        + " Return a natural person's full personal name only, not a company, "
        "organization or trading name. Do not append business/legal-form suffixes "
        "such as LTD, LLC, HF or SARL.",
    )


def resolve_reviewed_distinct_actors(
    blueprint: Any,
    review: DistinctActorReview | None,
) -> tuple[DistinctActor, ...]:
    if review is None:
        return ()
    if not isinstance(review, DistinctActorReview):
        raise TypeError("distinct-actor resolver requires a typed reviewed authority")
    authority = review
    _reviewed_source(
        blueprint,
        authority.source_document_id,
        authority.source_sha256,
    )
    bindings = [
        target
        for target in blueprint.contract.targets
        if target.path == authority.primary_role + ".name"
    ]
    if len(bindings) != 1 or bindings[0].expression != authority.primary_expression:
        raise ValueError("distinct-actor primary identity expression changed")
    variables = {variable.key: variable for variable in blueprint.contract.variables}
    if authority.related_key not in variables or variables[authority.related_key].kind != "name":
        raise ValueError("distinct actor no longer owns a reviewed name variable")
    actors = blueprint.ownership_data.get("related_actors", {})
    actor = actors.get(authority.related_key)
    if not isinstance(actor, Mapping) or actor.get("context_party") != authority.primary_role:
        raise ValueError("distinct actor no longer owns the reviewed related identity")
    return (
        DistinctActor(
            related_key=authority.related_key,
            primary_expression=bindings[0].expression,
            source_evidence=authority.source_evidence,
        ),
    )


def strengthen_distinct_actors(
    request: WordingRequest, constraints: Sequence[DistinctActor]
) -> WordingRequest:
    fields = {field.key: field for field in request.fields}
    by_key: dict[str, str] = {}
    contexts: list[str] = []
    for boundary in constraints:
        field = fields.get(boundary.related_key)
        if field is None or not field.role.endswith(" name"):
            raise ValueError("distinct-actor boundary does not own a requested identity field")
        primary_keys = set(_COMPONENT.findall(boundary.primary_expression))
        if not primary_keys <= fields.keys() or boundary.related_key in primary_keys:
            raise ValueError("distinct-actor primary expression has missing or circular owners")
        if boundary.related_key in by_key:
            raise ValueError("duplicate distinct-actor identity constraint")
        by_key[boundary.related_key] = (
            " Generate a separate actor's identity, not the complete primary party name "
            "plus a division, department, branch, office or location qualifier. "
            "Shared brand words are allowed; do not copy the complete primary legal name."
        )
        contexts.append(
            f"DISTINCT ACTOR {{{boundary.related_key}}}: the primary identity is "
            f"{boundary.primary_expression}. This separate actor requires a different "
            "complete identity, not that primary legal name with a department/division suffix."
        )
    return replace(
        request,
        context=request.context + ("\n" + "\n".join(contexts) if contexts else ""),
        fields=tuple(
            replace(field, requirement=field.requirement + by_key[field.key])
            if field.key in by_key
            else field
            for field in request.fields
        ),
    )


def validate_distinct_actors(
    values: Mapping[str, str], constraints: Sequence[DistinctActor]
) -> list[DistinctActorFinding]:
    findings: list[DistinctActorFinding] = []
    for boundary in constraints:
        try:
            primary_value = interpolate(
                boundary.primary_expression, cast(dict[str, str], values)
            )
            related_value = values[boundary.related_key]
        except KeyError as error:
            raise ValueError(
                f"distinct-actor validation is missing value: {error.args[0]}"
            ) from error
        primary = " ".join(primary_value.casefold().split())
        related = " ".join(related_value.casefold().split())
        if not primary or not related:
            raise ValueError("distinct-actor full identity cannot be empty")
        if re.search(r"(?<!\w)" + re.escape(primary) + r"(?!\w)", related):
            findings.append(
                DistinctActorFinding(
                    key=boundary.related_key,
                    problem="Separate actor contains the complete primary legal identity",
                    primary_identity=primary_value,
                    related_identity=related_value,
                )
            )
    return findings


_OPERATIONAL_STATES = {
    "ambient": (
        "Selected operational product state: AMBIENT, not frozen or chilled. "
        "Use a plausible non-frozen/non-chilled realization of the supplied commodity. "
        "A broad classification may enumerate frozen, fresh, cooked or preserved alternatives; "
        "those alternatives are contextual scope, not permission to choose a different "
        "operational state for this shipment."
    ),
    "frozen": (
        "Selected operational product state: FROZEN. Keep the supplied commodity in its "
        "frozen form, not fresh/chilled or ambient."
    ),
    "chilled": (
        "Selected operational product state: CHILLED. Keep the supplied fresh/chilled "
        "commodity form compatible with chilled carriage; do not invent frozen goods."
    ),
}
_COUNTED_UNIT_AUTHORITY = (
    "Host-owned unitNetKg/unitGrossKg describe the same COUNTED package or article. "
    "If product wording states an actual net fill or actual gross mass PER THAT COUNTED "
    "UNIT, it must equal the supplied corresponding unit fact, not a newly invented fill. "
    "Capacity/rating is not necessarily actual fill; do not silently equate them. "
    "An explicitly separate inner/support unit may have a different scope: do not multiply "
    "its fill by the outer counted quantity or invent its counts/totals. Where its actual "
    "numeric facts are not supplied, leave that fill/count unspecified. Preserve useful "
    "intrinsic product specifications. Do not add shipment net/gross, equipment, allocation, "
    "temperature or other extraction fields merely because private facts were supplied."
)


def apply_wording_authority(request: WordingRequest, *, regime: str) -> WordingRequest:
    if regime not in {*_OPERATIONAL_STATES, "dg"}:
        raise ValueError("unknown sampled operational regime")
    context = request.context
    if regime in _OPERATIONAL_STATES:
        context += "\nSAMPLED OPERATIONAL PRODUCT STATE\n" + _OPERATIONAL_STATES[regime]
        context += " Numeric carriage settings and printed-field presence remain host-owned."
    context += "\nCOUNTED UNIT VERSUS INNER/SUPPORT NUMERIC AUTHORITY\n"
    context += _COUNTED_UNIT_AUTHORITY
    return replace(request, context=context)


def strengthen_presentation_request(
    request: WordingRequest,
    *,
    locality_assemblies: Sequence[LocalityAssembly],
    distinct_actors: Sequence[DistinctActor],
    regime: Literal["ambient", "chilled", "frozen", "dg"],
) -> WordingRequest:
    """Compose all reviewed presentation constraints in one deterministic order."""

    result = strengthen_locality_request(request, locality_assemblies)
    result = replace(
        result,
        fields=tuple(strengthen_individual_field(field) for field in result.fields),
    )
    result = strengthen_distinct_actors(result, distinct_actors)
    return apply_wording_authority(result, regime=regime)


SEMANTIC_PRIVATE_PACKING_FIELDS = (
    "countedCategory",
    "publicQuantity",
    "relation",
    "relatedCategory",
    "relatedQuantity",
    "unitNetKg",
    "unitGrossKg",
    "netKg",
    "grossKg",
    "facts",
    "surfaces",
    "wordingFacts",
    "wordingAppends",
    "presentationDecisions",
)
PRIVATE_PACKING_AUDIT_FIELDS = frozenset(
    {
        "version",
        "sourceDocumentId",
        "certificateSha256",
        "certificateFileSha256",
        "publicAllocationVisibilityUnchanged",
        "invariants",
    }
)


def semantic_private_packing(ledger: dict[str, Any]) -> dict[str, Any]:
    """Project every semantic cargo fact and reject unclassified host fields."""

    if not isinstance(ledger, dict):
        raise ValueError("private packing ledger must be a mapping")
    required = set(SEMANTIC_PRIVATE_PACKING_FIELDS)
    if missing := required - ledger.keys():
        raise ValueError(f"private packing semantic fields missing: {sorted(missing)}")
    if unknown := ledger.keys() - required - PRIVATE_PACKING_AUDIT_FIELDS - {"scopeLedgers"}:
        raise ValueError(f"unclassified private packing fields: {sorted(unknown)}")
    projected = {key: copy.deepcopy(ledger[key]) for key in SEMANTIC_PRIVATE_PACKING_FIELDS}
    if "scopeLedgers" in ledger:
        scoped = ledger["scopeLedgers"]
        if not isinstance(scoped, dict):
            raise ValueError("scoped private ledgers must be a mapping")
        projected["scopeLedgers"] = {
            key: semantic_private_packing(value) for key, value in scoped.items()
        }
    return projected


def wording_scenario(scenario: Any) -> Any:
    """Create a prompt-only view while retaining the complete host scenario."""

    if "freshPrivatePacking" not in scenario.provenance:
        raise ValueError("wording scenario lacks its explicit private packing ledger")
    provenance = {
        **scenario.provenance,
        "freshPrivatePacking": semantic_private_packing(
            scenario.provenance["freshPrivatePacking"]
        ),
    }
    return scenario.model_copy(update={"provenance": provenance})
