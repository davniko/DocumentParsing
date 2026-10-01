"""Source-pinned semantic roles for mutable party-address surfaces.

The compiled byte template says *where* text can change. This companion
certificate says what each party-address span is allowed to mean. It is
independent of the renderer mode: single, segmented, and repeated address
bindings all pass through the same coverage check.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .models import CertifiedSemanticTemplate, SemanticBinding

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
_SHA = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_TEXT = Annotated[str, StringConstraints(min_length=1)]
AddressComponentRole = Literal[
    "street_or_site",
    "district_or_neighborhood",
    "administrative_region",
    "city",
    "country",
    "postal_code",
    "floor_or_unit",
    "building",
]
_WORDS = re.compile(r"\w+", flags=re.UNICODE)
_SLOT_KEY_PREFIX = "party_address_slot:"
_NUMBER_WORDS = {
    name: str(index)
    for index, name in enumerate(
        (
            "zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        )
    )
}
_NUMBER_WORDS.update(
    {
        name: str(index)
        for index, name in enumerate(
            (
                "zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
                "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
                "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
                "eighteenth", "nineteenth",
            )
        )
    }
)


def slot_value_key(binding_id: str, slot_id: str) -> str:
    """Receipt key for one source-owned printed address component."""

    if not binding_id or not slot_id or ":" in binding_id or ":" in slot_id:
        raise ValueError("party-address slot key has an invalid binding or slot identity")
    return _SLOT_KEY_PREFIX + binding_id + ":" + slot_id


def known_slot_value_keys(template: CertifiedSemanticTemplate) -> frozenset[str]:
    return frozenset(
        slot_value_key(str(binding.binding_id), str(slot.slot_id))
        for binding in template.bindings
        if binding.group_kind == "party" and binding.value_kind == "address"
        for slot in binding.occurrences
    )


def binding_slot_values(
    binding: SemanticBinding, auxiliary: Mapping[str, str] | None
) -> dict[str, str] | None:
    """Read all typed address slots, never a partial or unowned component."""

    if not auxiliary:
        return None
    if binding.group_kind != "party" or binding.value_kind != "address":
        return None
    keys = {
        slot.slot_id: slot_value_key(binding.binding_id, slot.slot_id)
        for slot in binding.occurrences
    }
    present = {
        slot_id: auxiliary[key]
        for slot_id, key in keys.items()
        if key in auxiliary
    }
    if present and set(present) != set(keys):
        raise ValueError("typed party address lacks one or more source-owned slots")
    return present or None


def slot_request_fields(
    certificate: PartyAddressRoleCertificate,
    template: CertifiedSemanticTemplate,
    lexical_fields: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Replace one flat linguistic address request with exact source-slot requests.

    The original lexical request remains the target/auxiliary owner. These
    provider-only fields have no target paths, so the host must assemble and
    validate the target before accepting a candidate.
    """

    bindings = _address_bindings(template)
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in certificate.groups:
        binding = bindings[group.binding_id]
        field = address_lexical_owner(group, binding, lexical_fields)
        if field is None:
            continue
        party_path = (
            group.address_target_path.removesuffix(".address")
            if group.address_target_path is not None
            else field.get("partyPath")
        )
        if not isinstance(party_path, str) or not party_path.startswith("documentPatch.parties."):
            raise ValueError("certified party address lacks an exact target-party owner")
        for slot in group.slots:
            key = slot_value_key(group.binding_id, slot.slot_id)
            if key in seen:
                raise ValueError("party address provider slot request repeats")
            seen.add(key)
            requests.append(
                {
                    "key": key,
                    "paths": [],
                    "auxiliaryKey": key,
                    "partyPath": party_path,
                    "partyAddressSlot": True,
                    "partyAddressTargetPath": group.address_target_path,
                    "source": slot.source_text,
                    "constraints": [
                        {
                            "sourceAddressRoles": slot.roles,
                            "sourceSlotId": slot.slot_id,
                            "sourceBindingId": group.binding_id,
                            "sourceSlotText": slot.source_text,
                            "separatelyBoundLocalityFields": group.separately_bound_locality_fields,
                            "administrativeLevel": slot.administrative_level,
                        }
                    ],
                }
            )
    return tuple(requests)


def address_lexical_owner(
    group: PartyAddressGroupRole,
    binding: SemanticBinding,
    lexical_fields: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    owners = [
        field
        for field in lexical_fields
        if set(group.binding_target_paths).intersection(field["paths"])
        or field.get("auxiliaryKey") == binding.logical_key
    ]
    if len(owners) > 1:
        raise ValueError("certified party address lacks one lexical owner")
    return owners[0] if owners else None


def _words(value: str) -> tuple[str, ...]:
    return tuple(match.group().casefold() for match in _WORDS.finditer(value))


def _contains(surface: str, value: str) -> bool:
    observed, wanted = _words(surface), _words(value)
    return bool(wanted) and any(
        observed[offset : offset + len(wanted)] == wanted
        for offset in range(len(observed) - len(wanted) + 1)
    )


def _subsequences(words: tuple[str, ...], sought: tuple[str, ...]) -> tuple[range, ...]:
    if not sought:
        return ()
    return tuple(
        range(offset, offset + len(sought))
        for offset in range(len(words) - len(sought) + 1)
        if words[offset : offset + len(sought)] == sought
    )


def _place_tokens(value: str) -> tuple[tuple[int, str], ...]:
    result = []
    for index, token in enumerate(_words(value)):
        if token == "of":
            continue
        ordinal = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", token)
        normalized = ordinal.group(1) if ordinal else _NUMBER_WORDS.get(token, token)
        result.append((index, normalized))
    return tuple(result)


def _one_edit(left: str, right: str) -> bool:
    if len(left) < 6 or len(right) < 6 or abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) == 1
    short, long = (left, right) if len(left) < len(right) else (right, left)
    return any(short == long[:offset] + long[offset + 1 :] for offset in range(len(long)))


def _locality_ranges(surface: str, locality: str) -> tuple[range, ...]:
    """Find conservative orthographic aliases, retaining original token positions.

    This is a rejection screen, not a geocoder. One fuzzy token requires another
    exact token in a multiword locality; short single-word names remain exact.
    """

    observed, wanted = _place_tokens(surface), _place_tokens(locality)
    if not wanted or len(observed) < len(wanted):
        return ()
    matches = []
    for offset in range(len(observed) - len(wanted) + 1):
        window = observed[offset : offset + len(wanted)]
        fuzzy = sum(
            _one_edit(actual, expected)
            for (_, actual), (_, expected) in zip(window, wanted, strict=True)
            if actual != expected
        )
        unequal = sum(
            actual != expected
            for (_, actual), (_, expected) in zip(window, wanted, strict=True)
        )
        if unequal == 0 or (len(wanted) > 1 and unequal == 1 and fuzzy == 1):
            matches.append(range(window[0][0], window[-1][0] + 1))
    return tuple(matches)


class EmbeddedLocalityRole(BaseModel):
    """A source-owned named place that legitimately repeats a separate locality."""

    model_config = _STRICT

    field: Literal["city", "country"]
    source_locality: _TEXT
    source_phrase: _TEXT
    prefix: str
    suffix: str
    semantic_role: Literal["named_site", "district_or_neighborhood"]
    replacement_policy: Literal["same_locality_only", "sampled_locality"]
    evidence: _TEXT

    @model_validator(mode="after")
    def source_phrase_is_explicit(self) -> EmbeddedLocalityRole:
        if self.source_phrase != self.prefix + self.source_locality + self.suffix:
            raise ValueError("embedded locality must identify one exact source phrase")
        if not self.prefix.strip() and not self.suffix.strip():
            raise ValueError("bare locality is not an embedded named address role")
        return self


class AddressSlotRole(BaseModel):
    model_config = _STRICT

    slot_id: _TEXT
    source_text: _TEXT
    roles: Annotated[tuple[AddressComponentRole, ...], Field(min_length=1)]
    administrative_level: Annotated[int, Field(ge=1, le=3)] | None = None
    repeat_group_index: Annotated[int, Field(ge=0)] | None = None
    retire_on_geography_change: tuple[_TEXT, ...] = ()
    embedded_locality_roles: tuple[EmbeddedLocalityRole, ...] = ()
    evidence: _TEXT

    @model_validator(mode="after")
    def role_details_are_explicit(self) -> AddressSlotRole:
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("address source slot repeats a semantic role")
        if ("administrative_region" in self.roles) != (self.administrative_level is not None):
            raise ValueError("administrative region role requires its exact level")
        if len(set(self.retire_on_geography_change)) != len(self.retire_on_geography_change):
            raise ValueError("retired source tokens must be unique")
        if any(not _contains(self.source_text, term) for term in self.retire_on_geography_change):
            raise ValueError("retired source token is absent from its certified source slot")
        for embedded in self.embedded_locality_roles:
            if embedded.source_phrase not in self.source_text:
                raise ValueError("embedded locality phrase is absent from its source slot")
            if embedded.semantic_role == "named_site" and not any(
                role in self.roles for role in ("street_or_site", "building")
            ):
                raise ValueError("named-site locality is not owned by a site/building slot")
            if (
                embedded.semantic_role == "district_or_neighborhood"
                and "district_or_neighborhood" not in self.roles
            ):
                raise ValueError("district locality is not owned by a district slot")
        return self


class FixedAdmin1Frame(BaseModel):
    """A source-local state/province declaration fixed by the route contract."""

    model_config = _STRICT

    byte_start: Annotated[int, Field(ge=0)]
    byte_end: Annotated[int, Field(gt=0)]
    source_text: _TEXT
    source_only_binding_id: _TEXT | None = None
    source_only_slot_id: _TEXT | None = None
    country_code: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
    geonames_admin1_code: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}\.[A-Za-z0-9]+$")]
    evidence: _TEXT

    @model_validator(mode="after")
    def region_identity_is_consistent(self) -> FixedAdmin1Frame:
        if self.byte_end <= self.byte_start:
            raise ValueError("fixed admin1 frame has empty or reversed byte span")
        if not self.geonames_admin1_code.startswith(self.country_code + "."):
            raise ValueError("fixed admin1 GeoNames identity differs from its country")
        if (self.source_only_binding_id is None) != (self.source_only_slot_id is None):
            raise ValueError("fixed admin1 source-only binding and slot must be specified together")
        return self


class PartyAddressGroupRole(BaseModel):
    model_config = _STRICT

    group_key: _TEXT
    binding_id: _TEXT
    binding_target_paths: tuple[_TEXT, ...]
    address_target_path: _TEXT | None
    separately_bound_locality_fields: tuple[Literal["city", "country"], ...]
    slots: Annotated[tuple[AddressSlotRole, ...], Field(min_length=1)]
    fixed_admin1_frames: tuple[FixedAdmin1Frame, ...] = ()

    @model_validator(mode="after")
    def group_has_unique_slots_and_frames(self) -> PartyAddressGroupRole:
        if len({slot.slot_id for slot in self.slots}) != len(self.slots):
            raise ValueError("address role group repeats a slot ID")
        if len(set(self.separately_bound_locality_fields)) != len(
            self.separately_bound_locality_fields
        ):
            raise ValueError("address role group repeats a separate locality field")
        address_paths = tuple(
            path for path in self.binding_target_paths if path.endswith(".address")
        )
        if self.address_target_path is None:
            if address_paths:
                raise ValueError("source-only address cannot have a task-facing address target")
        elif not address_paths or self.address_target_path != address_paths[0]:
            raise ValueError("primary address target must be the binding's first address target")
        codes = {
            (frame.country_code, frame.geonames_admin1_code) for frame in self.fixed_admin1_frames
        }
        if len(codes) > 1:
            raise ValueError("one party has conflicting immutable admin1 frames")
        return self


class PartyAddressRoleCertificate(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    source_document_id: _TEXT
    source_sha256: _SHA
    template_sha256: _SHA
    groups: Annotated[tuple[PartyAddressGroupRole, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def groups_are_unique(self) -> PartyAddressRoleCertificate:
        if len({group.binding_id for group in self.groups}) != len(self.groups):
            raise ValueError("party address role certificate repeats a binding")
        return self


def _address_bindings(template: CertifiedSemanticTemplate) -> dict[str, SemanticBinding]:
    return {
        str(binding.binding_id): binding
        for binding in template.bindings
        if binding.group_kind == "party" and binding.value_kind == "address"
    }


def validate_role_certificate(
    certificate: PartyAddressRoleCertificate,
    *,
    template: CertifiedSemanticTemplate,
    source: bytes,
    template_bytes: bytes,
    require_coverage: bool = True,
) -> None:
    """Certify exact source, binding, slot, and fixed-frame ownership."""

    if certificate.source_document_id != template.document_id:
        raise ValueError("party address roles belong to a different source document")
    if certificate.source_sha256 != hashlib.sha256(source).hexdigest():
        raise ValueError("party address roles pin different source bytes")
    if certificate.template_sha256 != hashlib.sha256(template_bytes).hexdigest():
        raise ValueError("party address roles pin different compiled template bytes")
    if template.source_sha256 != certificate.source_sha256:
        raise ValueError("compiled template itself pins different source bytes")
    bindings = _address_bindings(template)
    if require_coverage and {group.binding_id for group in certificate.groups} != set(bindings):
        raise ValueError("party address roles do not cover every address binding")
    all_mutable_slots = template.byte_template.slots
    for group in certificate.groups:
        fixed_source_bindings: dict[str, set[str]] = defaultdict(set)
        binding = bindings.get(group.binding_id)
        if binding is None:
            raise ValueError("party address role points to an absent address binding")
        if group.group_key != binding.group_key or binding.target_paths != (
            group.binding_target_paths
        ):
            raise ValueError("party address role changed its binding owner or target path")
        direct_owner_paths = tuple(
            path.removesuffix(".address")
            for path in group.binding_target_paths
            if path.endswith(".address")
        )
        auxiliary_owner_paths = tuple(
            entity.target_party_path
            for entity in template.auxiliary_semantic_plan.entities
            if entity.target_party_path is not None
            and any(member.logical_key == binding.logical_key for member in entity.members)
        )
        owner_paths = tuple(dict.fromkeys((*direct_owner_paths, *auxiliary_owner_paths)))
        expected_locality = tuple(
            field
            for field in ("city", "country")
            if any(
                owner + "." + field in other.target_paths
                and other.binding_id != binding.binding_id
                for other in template.bindings
                for owner in owner_paths
            )
        )
        if group.separately_bound_locality_fields != expected_locality:
            raise ValueError("party address role misstated separately printed locality")
        if [slot.slot_id for slot in group.slots] != [slot.slot_id for slot in binding.occurrences]:
            raise ValueError("party address role does not cover exact source slots in order")
        for slot, source_slot, plan in zip(
            group.slots, binding.occurrences, binding.realization.slots, strict=True
        ):
            if slot.source_text != source_slot.source_text:
                raise ValueError("party address role changed a source slot's text")
            if slot.repeat_group_index != plan.repeat_group_index:
                raise ValueError("party address role changed repeat-group topology")
            for embedded in slot.embedded_locality_roles:
                if embedded.field not in group.separately_bound_locality_fields:
                    raise ValueError("embedded locality lacks a separate source binding")
                if not any(
                    owner + "." + embedded.field in other.target_paths
                    and other.binding_id != binding.binding_id
                    and any(
                        _contains(source_occurrence.source_text, embedded.source_locality)
                        for source_occurrence in other.occurrences
                    )
                    for other in template.bindings
                    for owner in owner_paths
                ):
                    raise ValueError("embedded locality differs from its separate source field")
        for frame in group.fixed_admin1_frames:
            if source[frame.byte_start : frame.byte_end] != frame.source_text.encode("utf-8"):
                raise ValueError("fixed admin1 frame does not match exact source bytes")
            overlapping = [
                slot
                for slot in all_mutable_slots
                if slot.byte_start < frame.byte_end and frame.byte_start < slot.byte_end
            ]
            if frame.source_only_slot_id is None and overlapping:
                raise ValueError("fixed admin1 frame overlaps a mutable template slot")
            if frame.source_only_slot_id is not None:
                source_binding = next(
                    (
                        other
                        for other in template.bindings
                        if other.binding_id == frame.source_only_binding_id
                    ),
                    None,
                )
                if source_binding is None or source_binding.group_key != group.group_key:
                    raise ValueError("fixed admin1 source-only binding has a different owner")
                if source_binding.target_paths or source_binding.value_kind != "location":
                    raise ValueError("fixed admin1 source-only binding is not a location auxiliary")
                if not any(
                    slot.slot_id == frame.source_only_slot_id
                    and slot.byte_start == frame.byte_start
                    and slot.byte_end == frame.byte_end
                    for slot in source_binding.occurrences
                ):
                    raise ValueError("fixed admin1 frame does not cover its certified source slot")
                assert frame.source_only_binding_id is not None
                fixed_source_bindings[frame.source_only_binding_id].add(frame.source_only_slot_id)
            if any(
                slot.byte_start < frame.byte_end
                and frame.byte_start < slot.byte_end
                and slot.slot_id != frame.source_only_slot_id
                for slot in all_mutable_slots
            ):
                raise ValueError("fixed admin1 frame overlaps another mutable template slot")
        for binding_id, covered_slots in fixed_source_bindings.items():
            source_binding = next(row for row in template.bindings if row.binding_id == binding_id)
            if covered_slots != {slot.slot_id for slot in source_binding.occurrences}:
                raise ValueError("fixed admin1 auxiliary binding is only partly source-pinned")


def required_immutable_admin1(
    certificate: PartyAddressRoleCertificate, group_key: str
) -> tuple[str, str] | None:
    """Return a source-proven route constraint, or no constraint if absent."""

    if not any(group.group_key == group_key for group in certificate.groups):
        raise ValueError("unknown party group in immutable-admin1 query")
    frames = [
        frame
        for group in certificate.groups
        if group.group_key == group_key
        for frame in group.fixed_admin1_frames
    ]
    if not frames:
        return None
    identities = {(frame.country_code, frame.geonames_admin1_code) for frame in frames}
    if len(identities) != 1:
        raise ValueError("party has conflicting immutable admin1 frames")
    return next(iter(identities))


def fixed_source_only_replacements(
    certificate: PartyAddressRoleCertificate,
) -> dict[str, dict[str, str]]:
    """Exact auxiliary binding/slot surfaces that route-certified render must retain."""

    result: dict[str, dict[str, str]] = defaultdict(dict)
    for group in certificate.groups:
        for frame in group.fixed_admin1_frames:
            if frame.source_only_binding_id is not None:
                assert frame.source_only_slot_id is not None
                previous = result[frame.source_only_binding_id].get(frame.source_only_slot_id)
                if previous is not None and previous != frame.source_text:
                    raise ValueError("fixed admin1 source-only slot has conflicting text")
                result[frame.source_only_binding_id][frame.source_only_slot_id] = frame.source_text
    return dict(result)


def propose_address_label_from_source_projection(
    group: PartyAddressGroupRole,
    *,
    source_target_address: str,
    values: Mapping[str, str],
) -> str:
    """Invert exact printed source slots while retaining certified fixed label context.

    The proposal is not itself a rendering proof: callers must run the compiled
    forward realization and compare every replacement with ``values``. Cases
    whose source label normalizes or omits slot content fail closed here.
    """

    if group.address_target_path is None:
        raise ValueError("source-only address has no task-facing label to propose")
    if not source_target_address.strip():
        raise ValueError("source address label is empty")
    if set(values) != {slot.slot_id for slot in group.slots}:
        raise ValueError("party address label proposal lacks exact source slots")
    first_repeat = min(slot.repeat_group_index or 0 for slot in group.slots)
    slots = tuple(
        slot
        for slot in group.slots
        if (slot.repeat_group_index or 0) == first_repeat
    )
    candidate = " ".join(source_target_address.split())
    for slot in slots:
        source_phrase = " ".join(slot.source_text.split())
        positions = tuple(
            match.span()
            for match in re.finditer(re.escape(source_phrase), candidate, flags=re.IGNORECASE)
        )
        if len(positions) != 1:
            raise ValueError("source address label does not uniquely expose a certified slot")
        start, end = positions[0]
        candidate = candidate[:start] + " ".join(values[slot.slot_id].split()) + candidate[end:]
    return candidate


def validate_address_label_context(
    group: PartyAddressGroupRole,
    *,
    source_target_address: str,
    candidate: str,
    values: Mapping[str, str],
) -> None:
    """Prove that non-slot label text is only a route-pinned fixed admin frame.

    The old generic proportional renderer is not an inverse for typed slots of
    new lengths. Instead, replace each exact first-repeat phrase in both labels
    and compare the remaining words with the fixed, source-pinned frames.
    """

    if set(values) != {slot.slot_id for slot in group.slots}:
        raise ValueError("address-label context check lacks exact source slots")
    first_repeat = min(slot.repeat_group_index or 0 for slot in group.slots)
    slots = tuple(
        slot for slot in group.slots if (slot.repeat_group_index or 0) == first_repeat
    )

    def remainder(text: str, phrases: Sequence[str]) -> tuple[str, ...]:
        residual = " ".join(text.split())
        for phrase in phrases:
            normalized = " ".join(phrase.split())
            matches = tuple(
                match.span()
                for match in re.finditer(re.escape(normalized), residual, flags=re.IGNORECASE)
            )
            if len(matches) != 1:
                raise ValueError("address label does not expose one exact source-role phrase")
            start, end = matches[0]
            residual = residual[:start] + " " + residual[end:]
        return _words(residual)

    source_remaining = remainder(source_target_address, [slot.source_text for slot in slots])
    generated_remaining = remainder(candidate, [values[slot.slot_id] for slot in slots])
    if generated_remaining != source_remaining:
        raise ValueError("generated address changed source-owned fixed label context")
    if not source_remaining:
        return
    framed_words = [
        _words(frame.source_text)
        for frame in group.fixed_admin1_frames
    ]
    if not framed_words:
        raise ValueError("address label retains unowned text outside typed slots")
    possible: set[tuple[str, ...]] = {()}
    for frame_words in framed_words:
        possible |= {prefix + frame_words for prefix in tuple(possible)}
    if source_remaining not in possible:
        raise ValueError("address label context is not a pinned fixed admin frame")


def role_values_cover_repetitions(
    group: PartyAddressGroupRole,
    values: Mapping[str, str],
    *,
    geography_changed: bool,
    sampled_city: str | None,
    sampled_country: str | None,
    administrative_names_by_level: Mapping[int, str],
) -> str:
    """Validate exact slot values and derive one task-facing address value.

    Actual byte-format realization is intentionally left to the descendant
    renderer, which owns case, punctuation, and line wrapping.
    """

    expected = {slot.slot_id for slot in group.slots}
    if set(values) != expected:
        raise ValueError("generated party components do not cover exact address slots")
    repeats: dict[int, list[str]] = defaultdict(list)
    for slot in group.slots:
        value = values[slot.slot_id]
        if not value or value != value.strip() or "\r" in value:
            raise ValueError("generated party address component has unsafe edge/control text")
        if ("\n" in value) != ("\n" in slot.source_text):
            raise ValueError("generated party component changed source multiline topology")
        if geography_changed and any(
            _contains(value, retired) for retired in slot.retire_on_geography_change
        ):
            raise ValueError("generated party component retained retired source locality")
        if slot.administrative_level is not None:
            expected_region = administrative_names_by_level.get(slot.administrative_level)
            if expected_region is None:
                raise ValueError("source administrative slot lacks sampled registry region")
            if not _contains(value, expected_region):
                raise ValueError("generated party administrative slot contradicts sampled region")
        if (
            "city" in group.separately_bound_locality_fields
            and sampled_city
            and _locality_ranges(value, sampled_city)
        ):
            same_admin1 = (
                slot.administrative_level == 1
                and administrative_names_by_level.get(1) is not None
                and _words(administrative_names_by_level[1]) == _words(sampled_city)
            )
            if (
                "city" not in slot.roles
                and not same_admin1
                and not _embedded_locality_covers(
                    slot, value, field="city", sampled_locality=sampled_city
                )
            ):
                raise ValueError("generated city occupies a different source address role")
        if (
            "country" in group.separately_bound_locality_fields
            and sampled_country
            and _locality_ranges(value, sampled_country)
            and "country" not in slot.roles
            and not _embedded_locality_covers(
                slot, value, field="country", sampled_locality=sampled_country
            )
        ):
            raise ValueError("generated country occupies a different source address role")
        repeats[slot.repeat_group_index or 0].append(value.replace("\n", " "))
    normalized = {tuple(_words(" ".join(parts))) for parts in repeats.values()}
    if len(normalized) != 1:
        raise ValueError("repeated printed party addresses encode different values")
    return " ".join(repeats[min(repeats)])


def _embedded_locality_covers(
    slot: AddressSlotRole,
    value: str,
    *,
    field: Literal["city", "country"],
    sampled_locality: str,
) -> bool:
    words = _words(value)
    locality_ranges = _locality_ranges(value, sampled_locality)
    if not locality_ranges:
        return False
    licensed: set[int] = set()
    for embedded in slot.embedded_locality_roles:
        if embedded.field != field:
            continue
        if embedded.replacement_policy == "same_locality_only" and _words(
            sampled_locality
        ) != _words(embedded.source_locality):
            continue
        phrase = embedded.prefix + sampled_locality + embedded.suffix
        for phrase_range in _subsequences(words, _words(phrase)):
            licensed.update(phrase_range)
    return all(set(locality_range).issubset(licensed) for locality_range in locality_ranges)
