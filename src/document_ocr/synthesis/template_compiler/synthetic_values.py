from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from document_ocr.synthesis.generators import DeterministicStream

from .models import AuxiliaryEntity, AuxiliaryEntityMember, SemanticBinding


@dataclass(frozen=True, slots=True)
class GeoProfile:
    city: str
    country: str
    country_code: str
    region: str
    region_code: str
    postal_code: str


_GEOGRAPHIES = (
    GeoProfile("Valencia", "Spain", "ES", "Valencian Community", "VC", "46011"),
    GeoProfile("Rotterdam", "Netherlands", "NL", "South Holland", "ZH", "3011"),
    GeoProfile("Hamburg", "Germany", "DE", "Hamburg", "HH", "20457"),
    GeoProfile("Gdansk", "Poland", "PL", "Pomeranian", "PM", "80-001"),
    GeoProfile("Izmir", "Turkey", "TR", "Izmir", "IZ", "35210"),
    GeoProfile("Busan", "South Korea", "KR", "Busan", "BS", "48940"),
    GeoProfile("Singapore", "Singapore", "SG", "Singapore", "SG", "089763"),
    GeoProfile("Antwerp", "Belgium", "BE", "Antwerp", "AN", "2000"),
    GeoProfile("Savannah", "United States", "US", "Georgia", "GA", "31401"),
    GeoProfile("Montreal", "Canada", "CA", "Quebec", "QC", "H3B 2Y5"),
    GeoProfile("Santos", "Brazil", "BR", "Sao Paulo", "SP", "11013-922"),
    GeoProfile("Auckland", "New Zealand", "NZ", "Auckland", "AUK", "1010"),
)

_ORGANIZATION_PREFIXES = (
    "Bluehaven",
    "Northstar",
    "Meridian",
    "Silver Quay",
    "Harborline",
    "Crestpoint",
    "Seabridge",
    "Atlas Shore",
    "Everfield",
    "Portstone",
)
_ORGANIZATION_NOUNS = (
    "Logistics",
    "Maritime Services",
    "Trading",
    "Freight Solutions",
    "Industrial Supply",
    "Cargo Management",
    "Export Services",
    "Shipping Agency",
)
_PERSON_FIRST_NAMES = (
    "Elena",
    "Marta",
    "Daniel",
    "Jonas",
    "Leila",
    "Mateo",
    "Nadia",
    "Victor",
)
_PERSON_LAST_NAMES = (
    "Marin",
    "Kovac",
    "Navarro",
    "Larsen",
    "Rahman",
    "Petrov",
    "Santos",
    "Weber",
)
_STREET_NAMES = (
    "Quayside Road",
    "Harbor Avenue",
    "Dockland Street",
    "Mariner Lane",
    "Commerce Boulevard",
    "Terminal Way",
    "Seaport Drive",
    "Anchor Court",
)

_TARGET_PARTY_ROLES = {
    "shipper": "shipper",
    "consignee": "consignee",
    "deliveryagent": "deliveryAgent",
    "forwardingagent": "forwardingAgent",
    "consolidator": "consolidator",
    "carrier": "carrier",
}


def _normalized(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _tokens(value: str) -> frozenset[str]:
    return frozenset(token for token in re.split(r"[^a-z0-9]+", value.casefold()) if token)


def _target_party(target: Mapping[str, Any], binding: SemanticBinding) -> Mapping[str, Any] | None:
    patch = target.get("documentPatch")
    parties = patch.get("parties") if isinstance(patch, Mapping) else None
    if not isinstance(parties, Mapping):
        return None
    identity = _normalized(binding.group_key + " " + binding.logical_key)
    if "notify" in identity:
        rows = parties.get("notifyParties")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and rows:
            first = rows[0]
            return first if isinstance(first, Mapping) else None
    for marker, role in _TARGET_PARTY_ROLES.items():
        if marker in identity:
            value = parties.get(role)
            return value if isinstance(value, Mapping) else None
    return None


def _resolve_mapping_path(target: Mapping[str, Any], path: str) -> Mapping[str, Any] | None:
    current: Any = target
    for name, index_text in re.findall(r"([A-Za-z0-9_]+)(?:\[([0-9]+)\])?", path):
        if not isinstance(current, Mapping) or name not in current:
            return None
        current = current[name]
        if index_text:
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes)):
                return None
            index = int(index_text)
            if index >= len(current):
                return None
            current = current[index]
    return current if isinstance(current, Mapping) else None


class DeterministicValueFactory:
    """Create typed, schedule-independent auxiliary values for one descendant.

    The compiler's ``generated_auxiliary`` contract promises that these values do not require
    semantic invention at render time.  This factory makes that promise executable while keeping
    related party fields on one deterministic entity record.  It deliberately returns ``None``
    for a value it cannot type safely; the caller then retains the explicit agent route.
    """

    def __init__(self, *, seed: int, document_id: str, target: Mapping[str, Any]) -> None:
        self._stream = DeterministicStream(
            seed, "carrier-bound-descendant-auxiliary-v2", document_id
        )
        self._target = target

    def _select(self, values: Sequence[str], identity: str, *, counter: int = 0) -> str:
        stream = self._stream.derive(identity)
        return values[stream.randbelow(len(values), counter=counter)]

    def geography(self, binding: SemanticBinding) -> GeoProfile:
        return self.geography_for_identity(binding.group_key)

    def geography_for_identity(self, identity: str) -> GeoProfile:
        identity = "geo:" + identity
        index = self._stream.derive(identity).randbelow(len(_GEOGRAPHIES))
        return _GEOGRAPHIES[index]

    def organization(self, binding: SemanticBinding) -> str:
        return self.organization_for_identity(binding.group_key)

    def organization_for_identity(self, identity: str) -> str:
        identity = "organization:" + identity
        prefix = self._select(_ORGANIZATION_PREFIXES, identity + ":prefix")
        noun = self._select(_ORGANIZATION_NOUNS, identity + ":noun")
        return f"{prefix} {noun} Ltd."

    def person(self, binding: SemanticBinding) -> str:
        return self.person_for_identity(binding.group_key)

    def person_for_identity(self, identity: str) -> str:
        identity = "person:" + identity
        first = self._select(_PERSON_FIRST_NAMES, identity + ":first")
        last = self._select(_PERSON_LAST_NAMES, identity + ":last")
        return f"{first} {last}"

    def address(self, binding: SemanticBinding) -> str:
        party = _target_party(self._target, binding)
        if party is not None:
            target_address = party.get("address")
            if isinstance(target_address, str) and target_address.strip():
                return target_address
        geo = self.geography(binding)
        number = 10 + self._stream.derive("address:" + binding.group_key).randbelow(890)
        street = self._select(_STREET_NAMES, "street:" + binding.group_key)
        return f"{number} {street}, {geo.postal_code} {geo.city}"

    def address_for_identity(self, identity: str) -> str:
        geo = self.geography_for_identity(identity)
        number = 10 + self._stream.derive("address:" + identity).randbelow(890)
        street = self._select(_STREET_NAMES, "street:" + identity)
        return f"{number} {street}, {geo.postal_code} {geo.city}"

    def location(self, binding: SemanticBinding) -> str:
        tokens = _tokens(binding.logical_key + " " + binding.group_key)
        party = _target_party(self._target, binding)
        if party is not None:
            if {"country", "nation"} & tokens:
                value = party.get("country")
                if isinstance(value, str) and value.strip():
                    return value
            if {"city", "locality", "place"} & tokens:
                value = party.get("city")
                if isinstance(value, str) and value.strip():
                    return value
        geo = self.geography(binding)
        source = binding.occurrences[0].source_text.strip()
        if "code" in tokens:
            return geo.country_code
        if {"state", "province", "region"} & tokens:
            return geo.region_code if len(_normalized(source)) <= 3 else geo.region
        if {"country", "nation"} & tokens:
            return geo.country_code if len(_normalized(source)) <= 3 else geo.country
        if any(separator in source for separator in (",", "/")):
            return f"{geo.city}, {geo.country}"
        return geo.city

    def textual(self, binding: SemanticBinding) -> str | None:
        """Return a typed semantic candidate, not a formatted slot surface."""

        if binding.value_kind == "organization":
            party = _target_party(self._target, binding)
            target_name = party.get("name") if party is not None else None
            return (
                target_name
                if isinstance(target_name, str) and target_name.strip()
                else self.organization(binding)
            )
        if binding.value_kind in {"person", "contact_name"}:
            return self.person(binding)
        if binding.value_kind == "address":
            return self.address(binding)
        if binding.value_kind == "location":
            return self.location(binding)
        return None

    def entity_textual(
        self,
        *,
        entity: AuxiliaryEntity,
        member: AuxiliaryEntityMember,
        country_codes: Mapping[str, str],
    ) -> str | None:
        """Return one field from a canonical auxiliary entity.

        A target-linked entity projects the already synthesized target party.  An independent
        entity draws every geographic and identity field from one entity-keyed record, so two
        bindings can no longer invent mutually inconsistent countries or organizations.
        """

        party = (
            _resolve_mapping_path(self._target, entity.target_party_path)
            if entity.target_party_path is not None
            else None
        )
        field = member.field
        direct_party_fields = {
            "name": "name",
            "address": "address",
            "city": "city",
            "country": "country",
        }
        if party is not None and field in direct_party_fields:
            value = party.get(direct_party_fields[field])
            if isinstance(value, str) and value.strip():
                return value
        contact = party.get("contactDetails") if party is not None else None
        if isinstance(contact, Mapping):
            if field == "contact_name":
                value = contact.get("contactName")
                if isinstance(value, str) and value.strip():
                    return value
            if field in {"phone", "email"}:
                collection_name = "phoneNumbers" if field == "phone" else "emailAddresses"
                values = contact.get(collection_name)
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    value = next(
                        (row for row in values if isinstance(row, str) and row.strip()),
                        None,
                    )
                    if value is not None:
                        return value
        if party is not None and field == "country_code":
            country = party.get("country")
            if isinstance(country, str):
                normalized = _normalized(country)
                code = country_codes.get(normalized)
                if code is not None:
                    return code
        if party is not None:
            return self.person_for_identity(entity.entity_id) if field == "contact_name" else None

        identity = entity.entity_id
        geo = self.geography_for_identity(identity)
        if field == "name":
            return self.organization_for_identity(identity)
        if field == "address":
            return self.address_for_identity(identity)
        if field == "city":
            return geo.city
        if field == "region":
            return geo.region
        if field == "postal_code":
            return geo.postal_code
        if field == "country":
            return geo.country
        if field == "country_code":
            return geo.country_code
        if field == "contact_name":
            return self.person_for_identity(identity)
        return None
