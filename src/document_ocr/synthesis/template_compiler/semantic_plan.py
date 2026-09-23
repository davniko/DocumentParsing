"""Host-owned semantic plan for mutable facts absent from the training target.

The compiler agents find physical spans.  This module performs the cheaper and stricter job of
classifying every source-only mutable binding after those spans are final: document-copy numbers
and sequences become immutable facts, party attributes become members of canonical entities, and
everything else receives an explicit disposition.  The published model target is never extended.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any, cast

from document_ocr.hashing import canonical_json_bytes

from .coherence import CoherenceReviewRequired
from .models import (
    AuxiliaryBindingDisposition,
    AuxiliaryEntity,
    AuxiliaryEntityMember,
    AuxiliarySemanticPlan,
    CompositeNumberFact,
    DocumentSequenceFact,
    SemanticBinding,
)


class AuxiliarySemanticPlanReviewRequired(CoherenceReviewRequired):
    """A relational source fact cannot be certified without human evidence."""


_NUMBER_UNITS = {
    "none": 0,
    "zero": 0,
    "zeroes": 0,
    "one": 1,
    "first": 1,
    "two": 2,
    "second": 2,
    "three": 3,
    "third": 3,
    "four": 4,
    "fourth": 4,
    "five": 5,
    "fifth": 5,
    "six": 6,
    "sixth": 6,
    "seven": 7,
    "seventh": 7,
    "eight": 8,
    "eighth": 8,
    "nine": 9,
    "ninth": 9,
    "ten": 10,
    "tenth": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_NUMBER_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_TOKENS = frozenset((*_NUMBER_UNITS, *_NUMBER_TENS, "hundred", "thousand", "and"))
_TOKEN = re.compile(r"[A-Za-z]+|[0-9]+")
_SEQUENCE_CONNECTOR = re.compile(r"(?i)(?:\bof\b|/)")
_PARTY_VALUE_KINDS = frozenset(
    {
        "organization",
        "person",
        "address",
        "contact_name",
        "email",
        "phone",
        "location",
        "identifier",
        "legal_text",
        "commercial_text",
        "other_text",
    }
)
_TARGET_PARTY_PATHS = {
    "shipper": "documentPatch.parties.shipper",
    "consignee": "documentPatch.parties.consignee",
    "delivery_agent": "documentPatch.parties.deliveryAgent",
    "forwarding_agent": "documentPatch.parties.forwardingAgent",
    "consolidator": "documentPatch.parties.consolidator",
    "carrier": "documentPatch.parties.carrier",
}


def _semantic_id(kind: str, payload: Any) -> str:
    digest = sha256(canonical_json_bytes(payload)).hexdigest()[:16]
    return f"aux_{kind}_{digest}"


def _normalized(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _identity_tokens(value: str) -> tuple[str, ...]:
    split_camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return tuple(token for token in re.split(r"[^a-z0-9]+", split_camel.casefold()) if token)


def _word_number(tokens: Sequence[str]) -> int | None:
    cleaned = tuple(token.casefold() for token in tokens if token.casefold() != "and")
    if not cleaned or any(token not in _NUMBER_TOKENS for token in cleaned):
        return None
    total = 0
    current = 0
    for token in cleaned:
        if token in _NUMBER_UNITS:
            current += _NUMBER_UNITS[token]
        elif token in _NUMBER_TENS:
            current += _NUMBER_TENS[token]
        elif token == "hundred":
            current = max(current, 1) * 100
        elif token == "thousand":
            total += max(current, 1) * 1000
            current = 0
        else:  # pragma: no cover - guarded by the vocabulary check
            return None
    return total + current


def _word_values(value: str) -> tuple[int, ...]:
    tokens = tuple(token.casefold() for token in re.findall(r"[A-Za-z]+", value))
    rows: list[int] = []
    current: list[str] = []
    for token in (*tokens, "<END>"):
        if token in _NUMBER_TOKENS:
            current.append(token)
            continue
        if current:
            parsed = _word_number(current)
            if parsed is not None:
                rows.append(parsed)
            current = []
    return tuple(rows)


def _one_number_token(value: str) -> int | None:
    compact = value.strip()
    if re.fullmatch(r"[0-9]+", compact):
        return int(compact)
    words = re.findall(r"[A-Za-z]+", compact)
    if words and not re.search(r"[0-9]", compact):
        return _word_number(words)
    return None


def sequence_surface(value: str) -> tuple[int, int] | None:
    """Return an explicit ``index of total`` or ``index/total`` pair."""

    matches = tuple(_SEQUENCE_CONNECTOR.finditer(value))
    if len(matches) != 1:
        return None
    connector = matches[0]
    left = value[: connector.start()].strip(" \t\r\n[](){}:;.,")
    right = value[connector.end() :].strip(" \t\r\n[](){}:;.,")
    right = re.sub(r"(?i)\s+original(?:s|\s+bills?.*)?$", "", right).strip()
    left_value = _one_number_token(left)
    right_value = _one_number_token(right)
    if left_value is None or right_value is None:
        return None
    if connector.group(0).casefold() == "of":
        return left_value, right_value
    # Equal slash forms such as 3/THREE are two projections of one count unless the binding
    # explicitly calls itself a sequence.  The caller makes that final distinction.
    return left_value, right_value


def composite_number_surface(value: str) -> int | None:
    """Return the shared value of one digit and one word projection."""

    digits = tuple(int(token) for token in re.findall(r"[0-9]+", value))
    words = _word_values(value)
    if len(digits) != 1 or len(words) != 1 or digits[0] != words[0]:
        return None
    return digits[0]


def _source_texts(binding: SemanticBinding) -> tuple[str, ...]:
    return tuple(dict.fromkeys(slot.source_text.strip() for slot in binding.occurrences))


def _count_like(binding: SemanticBinding) -> bool:
    tokens = set(_identity_tokens(binding.logical_key + " " + binding.group_key))
    return bool(
        tokens
        & {
            "count",
            "original",
            "originals",
            "sequence",
            "rider",
            "copies",
            "copy",
        }
    )


def _role_and_index(binding: SemanticBinding) -> tuple[str, int] | None:
    if binding.value_kind not in _PARTY_VALUE_KINDS:
        return None
    logical_tokens = _identity_tokens(binding.logical_key)
    group_tokens = _identity_tokens(binding.group_key)

    def role_from(tokens: Sequence[str]) -> str | None:
        token_set = set(tokens)
        for candidate, required in (
            ("foreign_exporter", ("foreign", "exporter")),
            ("delivery_agent", ("delivery", "agent")),
            ("forwarding_agent", ("forwarding", "agent")),
            ("issuing_agent", ("issuing", "agent")),
        ):
            if set(required) <= token_set:
                return candidate
        for marker, candidate in (
            ("exporter", "exporter"),
            ("importer", "importer"),
            ("supplier", "supplier"),
            ("shipper", "shipper"),
            ("consignee", "consignee"),
            ("notify", "notify"),
            ("consolidator", "consolidator"),
            ("carrier", "carrier"),
            ("agent", "agent"),
        ):
            if marker in token_set:
                return candidate
        return None

    logical_role = role_from(logical_tokens)
    group_role = role_from(group_tokens)
    role: str | None
    if len(group_tokens) >= 3 and group_tokens[:2] == ("party", "source"):
        role = "source_" + "_".join(group_tokens[2:])
    elif logical_role is not None and logical_role != "agent":
        # An issuer/signature embedded in another party's layout block is still a distinct
        # entity.  The binding's explicit semantic role is stronger than its physical group.
        role = logical_role
    else:
        role = group_role or logical_role
    explicit_party = binding.group_kind == "party" or role in {
        "foreign_exporter",
        "exporter",
        "importer",
        "supplier",
    }
    if role is None or not explicit_party:
        return None
    if role == "agent" and binding.group_kind == "party":
        stable_group = tuple(
            token for token in group_tokens if token != "party" and not token.isdigit()
        )
        if stable_group:
            role = "_".join(stable_group)
    identity_tokens = (*group_tokens, *logical_tokens)
    role_tokens = tuple(role.split("_"))
    for offset in range(len(identity_tokens) - len(role_tokens) + 1):
        if identity_tokens[offset : offset + len(role_tokens)] != role_tokens:
            continue
        following = offset + len(role_tokens)
        if following < len(identity_tokens) and identity_tokens[following].isdigit():
            return role, int(identity_tokens[following])
    numeric_tokens = tuple(token for token in identity_tokens if token.isdigit())
    return role, int(numeric_tokens[0]) if len(numeric_tokens) == 1 else 0


def _sequence_scope(binding: SemanticBinding) -> str:
    """Return the semantic document-control series a sequence belongs to."""

    tokens = set(_identity_tokens(binding.logical_key + " " + binding.group_key))
    if "rider" in tokens:
        return "rider"
    if tokens & {"original", "originals", "copy", "copies"}:
        return "document_copy"
    stable_tokens = tuple(
        token
        for token in _identity_tokens(binding.group_key)
        if not token.isdigit() and token not in {"sequence", "count", "member"}
    )
    return "_".join(stable_tokens) or "document_sequence"


def _entity_field(binding: SemanticBinding) -> str:
    tokens = set(_identity_tokens(binding.logical_key + " " + binding.group_key))
    if binding.value_kind in {"identifier", "location"} and tokens & {
        "postal",
        "postcode",
        "zipcode",
    }:
        return "postal_code"
    if binding.value_kind == "phone" and tokens & {"extension", "ext"}:
        return "phone_extension"
    if "registration" in tokens and "type" in tokens:
        return "registration_type"
    if binding.value_kind == "organization":
        return "name"
    if binding.value_kind == "address":
        return "address"
    if binding.value_kind in {"person", "contact_name"}:
        return "contact_name"
    if binding.value_kind == "email":
        return "email"
    if binding.value_kind == "phone":
        return "phone"
    if binding.value_kind == "location":
        if "country" in tokens and "code" in tokens:
            return "country_code"
        if "country" in tokens or "nation" in tokens:
            return "country"
        if "city" in tokens or "locality" in tokens:
            return "city"
        if tokens & {"state", "province", "region"}:
            return "region"
        if "postal" in tokens:
            return "postal_code"
        return "other"
    if binding.value_kind == "identifier":
        if "country" in tokens and "code" in tokens:
            return "country_code"
        if tokens & {"tax", "vat", "gst"}:
            return "tax_identifier"
        if tokens & {"registration", "registry", "exporter", "importer"}:
            return "registration_identifier"
        return "other_identifier"
    return "other"


def _resolve_path(target: Mapping[str, Any], path: str) -> Any:
    current: Any = target
    for name, index_text in re.findall(r"([A-Za-z0-9_]+)(?:\[([0-9]+)\])?", path):
        if not isinstance(current, Mapping) or name not in current:
            raise KeyError(path)
        current = current[name]
        if index_text:
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
                raise KeyError(path)
            current = current[int(index_text)]
    return current


def _surface_equivalent(left: str, right: str) -> bool:
    left_normalized = _normalized(left)
    right_normalized = _normalized(right)
    if left_normalized == right_normalized:
        return True
    shorter, longer = sorted((left_normalized, right_normalized), key=len)
    if len(shorter) >= 8 and shorter in longer and len(shorter) / len(longer) >= 0.72:
        return True
    left_tokens = set(_identity_tokens(left))
    right_tokens = set(_identity_tokens(right))
    union = left_tokens | right_tokens
    return bool(union) and len(left_tokens & right_tokens) / len(union) >= 0.8


def _target_party_path(role: str, index: int) -> str | None:
    if role == "notify":
        return f"documentPatch.parties.notifyParties[{index}]"
    return _TARGET_PARTY_PATHS.get(role)


def _entity_relationship(
    *,
    role: str,
    index: int,
    members: Sequence[tuple[SemanticBinding, str]],
    raw: str,
    source_target: Mapping[str, Any],
) -> tuple[str, str | None, str]:
    direct_path = _target_party_path(role, index)
    if direct_path is not None:
        try:
            target_party = _resolve_path(source_target, direct_path)
        except (KeyError, IndexError):
            target_party = None
        if isinstance(target_party, Mapping):
            return (
                "same_as_target_party",
                direct_path,
                "The canonical party role is present in the structured source target.",
            )

    comparison_role = (
        "shipper"
        if role in {"exporter", "foreign_exporter"}
        else ("consignee" if role == "importer" else None)
    )
    comparison_path = _target_party_path(comparison_role, 0) if comparison_role else None
    comparison = None
    if comparison_path is not None:
        try:
            comparison = _resolve_path(source_target, comparison_path)
        except (KeyError, IndexError):
            comparison = None
    comparable: dict[str, list[bool]] = defaultdict(list)
    target_fields = {
        "name": "name",
        "address": "address",
        "city": "city",
        "country": "country",
    }
    if isinstance(comparison, Mapping):
        for binding, field in members:
            target_field = target_fields.get(field)
            target_value = comparison.get(target_field) if target_field is not None else None
            if not isinstance(target_value, str) or not target_value.strip():
                continue
            for source in _source_texts(binding):
                if source:
                    comparable[field].append(_surface_equivalent(source, target_value))
    combined_caption = re.search(r"(?i)\bSHIPPER\s*/\s*EXPORTER\b", raw) is not None
    name_matches = comparable.get("name", [])
    address_matches = comparable.get("address", [])
    geography_matches = [
        value for field in ("city", "country") for value in comparable.get(field, [])
    ]
    if combined_caption and role in {"exporter", "foreign_exporter"}:
        explicit_identity_evidence = [
            value
            for field in ("name", "address", "city", "country")
            for value in comparable.get(field, [])
        ]
        if not explicit_identity_evidence or all(explicit_identity_evidence):
            assert comparison_path is not None
            return (
                "same_as_target_party",
                comparison_path,
                "The combined SHIPPER/EXPORTER role has no conflicting exporter identity evidence.",
            )
    same_party = any(name_matches) or (
        not name_matches and any(address_matches) and any(geography_matches)
    )
    if same_party and (role not in {"exporter", "foreign_exporter"} or combined_caption):
        assert comparison_path is not None
        return (
            "same_as_target_party",
            comparison_path,
            f"Canonical identity evidence matches the structured {comparison_role}.",
        )
    return (
        "independent",
        None,
        "The source presents this role independently from any structured target party.",
    )


def _disposition(
    *, logical_key: str, disposition: str, semantic_id: str | None, rationale: str
) -> AuxiliaryBindingDisposition:
    return AuxiliaryBindingDisposition.model_validate(
        {
            "logical_key": logical_key,
            "disposition": disposition,
            "semantic_id": semantic_id,
            "rationale": rationale,
        }
    )


def build_auxiliary_semantic_plan(
    *, raw: str, bindings: Sequence[SemanticBinding], source_target: Mapping[str, Any]
) -> AuxiliarySemanticPlan:
    """Classify every mutable source-only binding or fail closed on contradictory evidence."""

    mutable = tuple(
        binding
        for binding in bindings
        if binding.render_mode in {"deterministic_auxiliary", "agent_residual"}
        and not binding.target_paths
        and not binding.dependency_paths
        and not binding.dependency_bindings
    )
    dispositions: dict[str, AuxiliaryBindingDisposition] = {}
    composite_rows: list[CompositeNumberFact] = []
    sequence_candidates: list[tuple[SemanticBinding, int, int]] = []
    referenced_dependency_keys = {
        dependency for binding in bindings for dependency in binding.dependency_bindings
    }

    for binding in mutable:
        texts = _source_texts(binding)
        count_like = _count_like(binding)
        sequences = tuple(sequence_surface(text) for text in texts)
        explicit_sequence = "sequence" in set(
            _identity_tokens(binding.logical_key + " " + binding.group_key)
        ) or any(re.search(r"(?i)\bof\b", text) for text in texts)
        if count_like and sequences and all(row is not None for row in sequences):
            typed_sequences = cast(tuple[tuple[int, int], ...], sequences)
            if len(set(typed_sequences)) != 1:
                raise AuxiliarySemanticPlanReviewRequired(
                    f"document sequence occurrences disagree: {binding.logical_key}"
                )
            index, total = typed_sequences[0]
            if explicit_sequence or index != total:
                if index > total or (total == 0 and index != 0):
                    raise AuxiliarySemanticPlanReviewRequired(
                        f"document sequence is impossible: {binding.logical_key}={index}/{total}"
                    )
                sequence_candidates.append((binding, index, total))
                continue
        composite_values = tuple(composite_number_surface(text) for text in texts)
        if count_like and composite_values and all(row is not None for row in composite_values):
            typed_values = cast(tuple[int, ...], composite_values)
            if len(set(typed_values)) != 1:
                raise AuxiliarySemanticPlanReviewRequired(
                    f"composite-number occurrences disagree: {binding.logical_key}"
                )
            semantic_id = _semantic_id("number", [binding.logical_key, typed_values[0]])
            composite_rows.append(
                CompositeNumberFact.model_validate(
                    {
                        "fact_id": semantic_id,
                        "member_logical_keys": (binding.logical_key,),
                        "value": typed_values[0],
                        "rationale": (
                            "The source prints one document fact in digit and word form; the "
                            "renderer must preserve both projections atomically."
                        ),
                    }
                )
            )
            dispositions[binding.logical_key] = _disposition(
                logical_key=binding.logical_key,
                disposition="composite_number",
                semantic_id=semantic_id,
                rationale="Bound to one parsed digit/word document fact.",
            )

    sequence_rows: list[DocumentSequenceFact] = []
    by_scope: dict[str, dict[int, list[tuple[SemanticBinding, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for binding, index, total in sequence_candidates:
        by_scope[_sequence_scope(binding)][total].append((binding, index))
    for scope, totals in sorted(by_scope.items()):
        if len(totals) > 1:
            details = ", ".join(str(total) for total in sorted(totals))
            raise AuxiliarySemanticPlanReviewRequired(
                f"{scope} sequence uses conflicting totals: {details}"
            )
    for scope, totals in sorted(by_scope.items()):
        total, rows = next(iter(sorted(totals.items())))
        semantic_id = _semantic_id(
            "sequence",
            [scope, [(binding.logical_key, index, total) for binding, index in rows]],
        )
        sequence_rows.append(
            DocumentSequenceFact.model_validate(
                {
                    "fact_id": semantic_id,
                    "total": total,
                    "members": tuple(
                        {"logical_key": binding.logical_key, "index": index}
                        for binding, index in rows
                    ),
                    "rationale": (
                        "The source explicitly prints bounded document-copy positions against "
                        "one shared total."
                    ),
                }
            )
        )
        for binding, _index in rows:
            dispositions[binding.logical_key] = _disposition(
                logical_key=binding.logical_key,
                disposition="document_sequence",
                semantic_id=semantic_id,
                rationale="Bound to one parsed and range-checked document sequence.",
            )

    entity_candidates: dict[tuple[str, int], list[tuple[SemanticBinding, str]]] = defaultdict(list)
    for binding in mutable:
        if binding.logical_key in dispositions:
            continue
        candidate_role = _role_and_index(binding)
        if candidate_role is not None:
            entity_candidates[candidate_role].append((binding, _entity_field(binding)))

    exporter_indexes = {
        index for role, index in entity_candidates if role in {"exporter", "foreign_exporter"}
    }
    for index in sorted(exporter_indexes):
        exporter_key = ("exporter", index)
        foreign_key = ("foreign_exporter", index)
        if exporter_key not in entity_candidates or foreign_key not in entity_candidates:
            continue
        exporter_members = entity_candidates[exporter_key]
        foreign_members = entity_candidates[foreign_key]
        conflicting_identity = False
        for field in ("name", "address", "country"):
            exporter_values = tuple(
                text
                for binding, member_field in exporter_members
                if member_field == field
                for text in _source_texts(binding)
            )
            foreign_values = tuple(
                text
                for binding, member_field in foreign_members
                if member_field == field
                for text in _source_texts(binding)
            )
            if (
                exporter_values
                and foreign_values
                and not any(
                    _surface_equivalent(left, right)
                    for left in exporter_values
                    for right in foreign_values
                )
            ):
                conflicting_identity = True
                break
        if not conflicting_identity:
            entity_candidates[foreign_key].extend(exporter_members)
            del entity_candidates[exporter_key]

    entities: list[AuxiliaryEntity] = []
    for (entity_role, entity_index), members in sorted(entity_candidates.items()):
        relationship, target_path, rationale = _entity_relationship(
            role=entity_role,
            index=entity_index,
            members=members,
            raw=raw,
            source_target=source_target,
        )
        entity_id = _semantic_id("entity", [entity_role, entity_index])
        entity = AuxiliaryEntity.model_validate(
            {
                "entity_id": entity_id,
                "role": f"{entity_role}:{entity_index}",
                "relationship": relationship,
                "target_party_path": target_path,
                "members": tuple(
                    AuxiliaryEntityMember.model_validate(
                        {"logical_key": binding.logical_key, "field": field}
                    )
                    for binding, field in sorted(members, key=lambda row: row[0].logical_key)
                ),
                "rationale": rationale,
            }
        )
        entities.append(entity)
        for binding, _field in members:
            dispositions[binding.logical_key] = _disposition(
                logical_key=binding.logical_key,
                disposition="entity_member",
                semantic_id=entity_id,
                rationale=f"Canonical member of auxiliary {entity_role}:{entity_index}.",
            )

    for binding in mutable:
        if binding.logical_key in dispositions:
            continue
        identity_tokens = set(_identity_tokens(binding.logical_key + " " + binding.group_key))
        if binding.source_relationships or binding.logical_key in referenced_dependency_keys:
            disposition = "declared_relationship"
            rationale = "The compiler already declares an exact source-binding relationship."
        elif _count_like(binding) or (
            {"registration", "type"} <= identity_tokens or {"original", "status"} <= identity_tokens
        ):
            disposition = "stable_vocabulary"
            rationale = (
                "This document-control vocabulary is template state, not an independently "
                "randomized shipment fact."
            )
        else:
            disposition = "independent_fact"
            rationale = (
                "No target, declared dependency, entity role, composite number, or sequence "
                "relationship is present after the complete host candidate sweep."
            )
        dispositions[binding.logical_key] = _disposition(
            logical_key=binding.logical_key,
            disposition=disposition,
            semantic_id=None,
            rationale=rationale,
        )

    plan = AuxiliarySemanticPlan.model_validate(
        {
            "schema_version": 1,
            "entities": tuple(entities),
            "composite_numbers": tuple(composite_rows),
            "document_sequences": tuple(sequence_rows),
            "dispositions": tuple(dispositions[key] for key in sorted(dispositions)),
        }
    )
    validate_auxiliary_semantic_plan(plan=plan, bindings=bindings, source_target=source_target)
    return plan


def validate_auxiliary_semantic_plan(
    *,
    plan: AuxiliarySemanticPlan,
    bindings: Sequence[SemanticBinding],
    source_target: Mapping[str, Any],
) -> None:
    """Validate plan closure and every source-side formal fact before certification."""

    by_key = {binding.logical_key: binding for binding in bindings}
    mutable = {
        binding.logical_key
        for binding in bindings
        if binding.render_mode in {"deterministic_auxiliary", "agent_residual"}
        and not binding.target_paths
        and not binding.dependency_paths
        and not binding.dependency_bindings
    }
    planned = {row.logical_key for row in plan.dispositions}
    if mutable != planned:
        raise ValueError(
            "auxiliary plan coverage differs from mutable source-only bindings: "
            f"missing={sorted(mutable - planned)} extra={sorted(planned - mutable)}"
        )
    for entity in plan.entities:
        if entity.target_party_path is not None:
            target = _resolve_path(source_target, entity.target_party_path)
            if not isinstance(target, Mapping):
                raise ValueError(
                    f"auxiliary entity target is not a party object: {entity.entity_id}"
                )
        for member in entity.members:
            if member.logical_key not in by_key:
                raise ValueError(f"unknown auxiliary entity member: {member.logical_key}")
    for number_fact in plan.composite_numbers:
        for key in number_fact.member_logical_keys:
            values = tuple(composite_number_surface(text) for text in _source_texts(by_key[key]))
            if not values or any(value != number_fact.value for value in values):
                raise ValueError(f"composite-number source no longer matches {number_fact.fact_id}")
    for sequence_fact in plan.document_sequences:
        expected_sequences = {
            row.logical_key: (row.index, sequence_fact.total) for row in sequence_fact.members
        }
        for key, expected_sequence in expected_sequences.items():
            sequence_values = tuple(sequence_surface(text) for text in _source_texts(by_key[key]))
            if not sequence_values or any(value != expected_sequence for value in sequence_values):
                raise ValueError(
                    f"document-sequence source no longer matches {sequence_fact.fact_id}"
                )


def disposition_by_binding(plan: AuxiliarySemanticPlan) -> dict[str, AuxiliaryBindingDisposition]:
    return {row.logical_key: row for row in plan.dispositions}


def entity_members_by_binding(
    plan: AuxiliarySemanticPlan,
) -> dict[str, tuple[AuxiliaryEntity, AuxiliaryEntityMember]]:
    return {
        member.logical_key: (entity, member)
        for entity in plan.entities
        for member in entity.members
    }


def resolve_geographic_members(
    plan: AuxiliarySemanticPlan,
    bindings: Sequence[SemanticBinding],
    country_codes: Mapping[str, str],
) -> AuxiliarySemanticPlan:
    """Resolve separately owned geography and source-proven ISO country codes.

    Tax codes and unpaired registration codes are never reinterpreted. A CODE
    must be a complete ISO alphabetic code matching every country occurrence in
    that same entity. Registration codes additionally require the matching
    registration-country owner, not merely a country elsewhere in the entity.
    """
    by_key = {binding.logical_key: binding for binding in bindings}
    entities = []
    for entity in plan.entities:
        postal_keys = {
            m.logical_key
            for m in entity.members
            if m.field == "postal_code"
            or (
                m.field in {"registration_identifier", "other_identifier"}
                and _entity_field(by_key[m.logical_key]) == "postal_code"
            )
        }
        region_keys = {
            m.logical_key
            for m in entity.members
            if m.field == "address"
            and set(_identity_tokens(m.logical_key)) & {"state", "province", "region"}
            and postal_keys - {m.logical_key}
            and all(
                any(c.isalpha() for c in s.source_text)
                and not any(c.isdigit() for c in s.source_text)
                for s in by_key[m.logical_key].occurrences
            )
        }
        # A compiler may label the region slice as address_region_postal even
        # though its postcode has a distinct binding. Preserve those owners;
        # generating another street for the region would corrupt the address.
        candidates = region_keys | {
            member.logical_key
            for member in entity.members
            if member.field in {"registration_identifier", "other_identifier"}
        }
        if not candidates:
            entities.append(entity)
            continue
        countries = {
            country_codes.get(_normalized(slot.source_text))
            for member in entity.members
            if member.field == "country"
            for slot in by_key[member.logical_key].occurrences
        }
        registration_countries = {
            tuple(_identity_tokens(member.logical_key))
            for member in entity.members
            if member.field == "country"
            and set(_identity_tokens(member.logical_key)) & {"registration", "registry"}
        }
        members = []
        for member in entity.members:
            if member.logical_key not in candidates:
                members.append(member)
                continue
            binding = by_key[member.logical_key]
            if member.logical_key in region_keys:
                members.append(member.model_copy(update={"field": "region"}))
                continue
            tokens = set(_identity_tokens(binding.logical_key))
            ordered_tokens = tuple(_identity_tokens(binding.logical_key))
            paired_registration_country = (
                bool(ordered_tokens)
                and ordered_tokens[-1] == "code"
                and any(
                    country_tokens[-1] == "country"
                    and ordered_tokens[:-1] in {country_tokens, country_tokens[:-1]}
                    for country_tokens in registration_countries
                )
            )
            surfaces = [slot.source_text.strip() for slot in binding.occurrences]
            if _entity_field(binding) == "postal_code":
                members.append(member.model_copy(update={"field": "postal_code"}))
                continue
            if (
                member.field in {"registration_identifier", "other_identifier"}
                and "code" in tokens
                and not tokens & {"tax", "vat", "gst"}
                and (not tokens & {"registration", "registry"} or paired_registration_country)
                and len(countries) == 1
                and None not in countries
                and bool(surfaces)
                and all(
                    re.fullmatch(r"[A-Za-z]{2,3}", value)
                    and country_codes.get(_normalized(value)) in countries
                    for value in surfaces
                )
            ):
                member = member.model_copy(update={"field": "country_code"})
            members.append(member)
        entities.append(
            entity
            if tuple(members) == entity.members
            else entity.model_copy(update={"members": tuple(members)})
        )
    return (
        plan
        if tuple(entities) == plan.entities
        else plan.model_copy(update={"entities": tuple(entities)})
    )


def validate_auxiliary_render(
    *,
    plan: AuxiliarySemanticPlan,
    bindings: Sequence[SemanticBinding],
    outputs: Mapping[str, Any],
    target: Mapping[str, Any],
    country_codes: Mapping[str, str],
    target_projection_values: Mapping[str, str] | None = None,
) -> None:
    """Parse formal auxiliary surfaces back after rendering and reject contradictions."""
    plan = resolve_geographic_members(plan, bindings, country_codes)
    by_key = {binding.logical_key: binding for binding in bindings}
    for entity in plan.entities:
        values_by_field: dict[str, list[str]] = defaultdict(list)
        for member in entity.members:
            output = outputs.get(member.logical_key)
            canonical = getattr(output, "canonical_value", None)
            if (
                entity.target_party_path is not None
                and target_projection_values is not None
                and member.logical_key in target_projection_values
            ):
                projected_expected = target_projection_values[member.logical_key]
                if not isinstance(canonical, str) or not _surface_equivalent(
                    canonical, projected_expected
                ):
                    raise ValueError("auxiliary target-projection facet differs from its contract")
                # A proven postcode, legal suffix or locality fragment is not the
                # entire address/name/country. Validate that facet, not equality
                # between the fragment and its whole parent field.
                continue
            if isinstance(canonical, str) and canonical.strip():
                values_by_field[member.field].append(canonical)
        # An address can be split across several independently owned physical lines (street,
        # postal locality, building, and so on).  Those canonical fragments are complementary,
        # not conflicting duplicate scalar values.  Target-linked full addresses are still
        # checked member-by-member against the structured party below.
        for field in ("name", "city", "region", "postal_code", "country"):
            normalized = {_normalized(value) for value in values_by_field.get(field, [])}
            if len(normalized) > 1:
                raise ValueError(
                    f"canonical auxiliary entity has conflicting {field} values: {entity.entity_id}"
                )
        countries = values_by_field.get("country", [])
        codes = values_by_field.get("country_code", [])
        if countries and codes:
            expected = country_codes.get(_normalized(countries[0]))
            if expected is None or any(code.casefold() != expected.casefold() for code in codes):
                raise ValueError(
                    f"canonical auxiliary entity country code conflicts with its country: "
                    f"{entity.entity_id}"
                )
        if entity.target_party_path is not None:
            party = _resolve_path(target, entity.target_party_path)
            if not isinstance(party, Mapping):
                raise ValueError(f"rendered auxiliary entity target is absent: {entity.entity_id}")
            for field, target_field in {
                "name": "name",
                "address": "address",
                "city": "city",
                "country": "country",
            }.items():
                target_value = party.get(target_field)
                if not isinstance(target_value, str) or not target_value.strip():
                    continue
                if any(
                    not _surface_equivalent(value, target_value)
                    for value in values_by_field.get(field, [])
                ):
                    raise ValueError(
                        f"auxiliary {field} differs from target party: {entity.entity_id}"
                    )
    for number_fact in plan.composite_numbers:
        for key in number_fact.member_logical_keys:
            output = outputs[key]
            replacements = cast(Mapping[str, str], output.replacements)
            binding = by_key[key]
            values = tuple(
                composite_number_surface(replacements[slot.slot_id]) for slot in binding.occurrences
            )
            if not values or any(value != number_fact.value for value in values):
                raise ValueError(f"rendered composite-number contradiction: {number_fact.fact_id}")
    for sequence_fact in plan.document_sequences:
        expected_sequences = {
            row.logical_key: (row.index, sequence_fact.total) for row in sequence_fact.members
        }
        for key, expected_sequence in expected_sequences.items():
            output = outputs[key]
            replacements = cast(Mapping[str, str], output.replacements)
            binding = by_key[key]
            sequence_values = tuple(
                sequence_surface(replacements[slot.slot_id]) for slot in binding.occurrences
            )
            if not sequence_values or any(value != expected_sequence for value in sequence_values):
                raise ValueError(
                    f"rendered document-sequence contradiction: {sequence_fact.fact_id}"
                )
