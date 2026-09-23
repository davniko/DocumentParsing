"""Route-first proposals and explicit context for compiled-template synthesis.

Sampling is offline. A pinned per-sample projection is applied before linguistic
generation and checked again afterwards; source restoration is never a repair.
Latent party localities and source-only facts do not extend the training schema.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.country_registry import CountryRegistry
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.shipment_scenarios import (
    ScenarioSupport,
    ShipmentScenario,
    _sample_locality,
    project_scenario_target,
    sample_shipment_scenario,
)

from . import complete_targets as targets
from . import contact_values, geographic_context, route_derivations, transport_derivations
from . import descendant as render
from .address_localities import AddressLocalities
from .models import NonEmptyText, Sha256
from .realization_contract import fixed_projection_ranges, projected_auxiliary_values


class PartyGeography(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    country_code: NonEmptyText
    country_name: NonEmptyText
    city: NonEmptyText | None


class AddressCountryConflict(ValueError):
    def __init__(self, party_path: str, country_name: str) -> None:
        self.party_path = party_path
        self.country_name = country_name
        super().__init__(
            f"generated address contradicts sampled party country: {party_path}; "
            f"end the address with the explicit country name {country_name!r} "
            "to disambiguate country-like region names or abbreviations"
        )


class AddressLocalityConflict(AddressCountryConflict):
    def __init__(
        self, party_path: str, country_name: str, city: str, conflicts: tuple[str, ...]
    ) -> None:
        super().__init__(party_path, country_name)
        self.city = city
        self.conflicts = conflicts
        self.args = (
            f"generated address names conflicting localities {conflicts}: {party_path}; "
            f"remove those locality components and use the sampled city {city!r}, "
            f"country {country_name!r}",
        )


class RouteProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal[1]
    sample_id: NonEmptyText
    source_document_id: NonEmptyText
    source_target_sha256: Sha256
    updates: dict[str, str]
    party_geography: dict[str, PartyGeography]
    auxiliary_values: dict[str, str]
    scenario: dict[str, Any]
    rejected_candidates: tuple[str, ...]

    def apply(self, source: targets.SourceTemplate, target: Mapping[str, Any]) -> dict[str, Any]:
        if (
            source.document_id != self.source_document_id
            or self.source_target_sha256 != sha256_bytes(canonical_json_bytes(source.target))
        ):
            raise ValueError("route projection differs from its pinned source")
        if any(path.startswith("documentPatch.parties.carrier.") for path in self.updates):
            raise ValueError("route projection cannot alter the template carrier")
        leaves = render._flatten_leaves(source.target)
        if self.updates.keys() - leaves.keys():
            raise ValueError("route projection cannot introduce training-label leaves")
        result = deepcopy(dict(target))
        for path, value in self.updates.items():
            targets._set(result, path, value)
        return result

    def validate_final(
        self,
        target: Mapping[str, Any],
        auxiliary: Mapping[str, str],
        *,
        country_codes: Mapping[str, str] | None = None,
        source_target: Mapping[str, Any] | None = None,
        localities: AddressLocalities | None = None,
    ) -> None:
        for path, value in self.updates.items():
            if render._resolve_path(target, path) != value:
                raise ValueError(f"sampled route fact changed after generation: {path}")
        if any(auxiliary.get(key) != value for key, value in self.auxiliary_values.items()):
            raise ValueError("sampled source-only geography changed after generation")
        for path, geography in self.party_geography.items():
            party = render._resolve_path(target, path)
            if not isinstance(party, Mapping):
                raise ValueError("sampled party geography points to a non-party value")
            address = party.get("address")
            if country_codes is not None and isinstance(address, str):
                country = geographic_context.generated_address_country(
                    address,
                    country_codes,
                    region_names=(
                        localities.ambiguous_names.get(geography.country_code, frozenset())
                        if localities is not None
                        else frozenset()
                    ),
                )
                if country is not None and country != geography.country_code:
                    raise AddressCountryConflict(path, geography.country_name)
            if localities is not None and isinstance(address, str) and geography.city is not None:
                conflicts = localities.conflicts(
                    address,
                    country=geography.country_code,
                    city=geography.city,
                    country_name=geography.country_name,
                )
                if conflicts:
                    raise AddressLocalityConflict(
                        path, geography.country_name, geography.city, conflicts
                    )
            phones = party.get("contactDetails", {}).get("phoneNumbers", [])
            original = (
                render._resolve_path(source_target, path) if source_target is not None else None
            )
            try:
                contact_values.validate_party_phones(
                    original.get("contactDetails", {}).get("phoneNumbers", [])
                    if original is not None
                    else None,
                    phones,
                    country_code=geography.country_code,
                )
            except ValueError as error:
                raise ValueError(f"{error}: {path}") from error


def _party_context(
    scenario: ShipmentScenario,
    *,
    support: ScenarioSupport,
    countries: CountryRegistry,
    stream: DeterministicStream,
) -> dict[str, PartyGeography]:
    result: dict[str, PartyGeography] = {}
    for party in scenario.party_localities:
        path = "documentPatch.parties." + party.role
        if party.role == "notifyParties":
            path += f"[{party.occurrence}]"
        owner = party.same_as
        if party.concrete_identity_source is not None:
            owner = party.concrete_identity_source.role
        if owner is not None:
            owner_path = "documentPatch.parties." + owner
            if owner == "notifyParties":
                if party.concrete_identity_source is None:
                    raise ValueError("notify-party geography owner requires an occurrence")
                owner_path += f"[{party.concrete_identity_source.occurrence}]"
            if owner_path not in result:
                raise ValueError("party geography owner has not been sampled")
            result[path] = result[owner_path]
            continue
        locality = party.locality
        if locality is None:
            # This is latent address context, not a new output-schema city field.
            locality = _sample_locality(
                country=party.conditioning_country_code,
                support=support,
                country_registry=countries,
                stream=stream.derive(path),
                endpoint=None,
            )
        result[path] = PartyGeography(
            country_code=party.conditioning_country_code,
            country_name=party.conditioning_country_name,
            city=locality.name,
        )
    return result


def _auxiliary_geography(
    source: targets.SourceTemplate,
    projected: Mapping[str, Any],
    context: Mapping[str, PartyGeography],
    country_codes: Mapping[str, str],
) -> dict[str, str]:
    result = {}
    projected_context = projected_auxiliary_values(source.template, projected)
    by_key = {b.logical_key: b for b in source.template.bindings}
    for entity in source.template.auxiliary_semantic_plan.entities:
        if entity.target_party_path is None or entity.target_party_path not in context:
            continue
        geo = context[entity.target_party_path]
        pinned = geographic_context.address_country_context(
            entity, source.template, source.source, country_codes
        )
        if pinned and pinned != {geo.country_code}:
            raise ValueError(f"party address has immutable country context: {entity.entity_id}")
        party = render._resolve_path(projected, entity.target_party_path)
        for member in entity.members:
            if member.field not in {"city", "country", "country_code"}:
                continue
            if render._party_exposes_auxiliary_field(party, member.field):
                continue
            value = (
                geo.city
                if member.field == "city"
                else geo.country_code
                if member.field == "country_code"
                else geo.country_name
            )
            if value is None:
                raise ValueError("source-only city has no sampled locality")
            if member.logical_key in projected_context:
                raise ValueError(
                    "geographic auxiliary is also a lexical projection; compilation review "
                    f"is required: {member.logical_key}"
                )
            binding = by_key[member.logical_key]
            fixed = geographic_context.pinned_country(
                binding, source.template, source.source, country_codes
            )
            if fixed and fixed != geo.country_code:
                raise ValueError(
                    f"party geography has an immutable country frame: {binding.logical_key}"
                )
            output = render._render_text_candidate(binding, value)
            render._validate_binding_format(
                source=source.source, template=source.template.byte_template, output=output
            )
            result[member.logical_key] = value
    return result


def require_route_contract(source: targets.SourceTemplate) -> None:
    """An untyped auxiliary route is not permission to sample an unrelated port."""
    contact_values.require_contact_name_coverage(source.source, source.template, source.target)
    for binding in source.template.bindings:
        if binding.derivation in route_derivations.DERIVATIONS:
            route_derivations.endpoint(binding)
        if binding.derivation in transport_derivations.DERIVATIONS:
            transport_derivations.validate(binding)
    if any(b.derivation in route_derivations.DERIVATIONS for b in source.template.bindings):
        route_derivations.validate_source(source.target, source.template.bindings)
    transport_derivations.validate_source(source.template.bindings, source.target)
    fixed_geography = [
        b.logical_key
        for b in source.template.bindings
        if b.realization.mode == "static"
        and any(
            p.startswith(("documentPatch.route.", "documentPatch.placeOfIssue."))
            or p.startswith("documentPatch.freight.paymentPlace.")
            for p in b.target_paths
        )
    ]
    if fixed_geography:
        raise ValueError(
            "fixed carrier-office or route geography needs a constrained route contract: "
            + ", ".join(fixed_geography)
        )
    owners = {
        tuple(t[0] for t in render._token_spans(slot.source_text)): binding.logical_key
        for binding in source.template.bindings
        if any(
            p.endswith((".city", ".country"))
            or re.fullmatch(r"documentPatch\.(?:route\.[^.]+|placeOfIssue)\.name", p)
            for p in binding.target_paths
        )
        for slot in binding.occurrences
    }
    for binding in source.template.bindings:
        from .projected_context import declared_edges

        mutable_context = {
            edge.tokens for edge in declared_edges(binding, source.template.bindings)
        }
        for _, _, fragment in fixed_projection_ranges(binding):
            if fragment in mutable_context:
                continue
            if fragment in owners and owners[fragment] != binding.logical_key:
                raise ValueError(
                    "mutable geography owns a fixed lexical frame; requires a cross-binding "
                    "dependency contract: " + binding.logical_key
                )
            if (
                fragment
                and any(
                    path.startswith("documentPatch.parties.")
                    and not path.startswith("documentPatch.parties.carrier.")
                    and path.endswith(".address")
                    for path in binding.target_paths
                )
                and fragment not in {("no",), ("rd", "no"), ("po",), ("of",), ("du",)}
            ):
                # An omitted region/postcode is still a geographic fact. A
                # country/city sampler cannot preserve arbitrary address words
                # simply because the compiler left them outside the value slot.
                # Only reviewed generic address grammar is location-independent;
                # other fixed fragments need an explicit typed contract/review.
                raise ValueError(
                    "fixed address fragment lacks a geography-independent contract: "
                    + binding.logical_key
                    + ": "
                    + " ".join(fragment)
                )
    for binding in source.template.bindings:
        if (
            any(p.endswith(".city") for p in binding.target_paths)
            and any(p.endswith(".country") for p in binding.target_paths)
            and binding.target_relationship == "shared_value_equality"
            and not (
                binding.realization.mode == "static"
                and all(
                    p.startswith("documentPatch.parties.carrier.") for p in binding.target_paths
                )
            )
        ):
            raise ValueError(
                "one printed value represents both city and country; "
                "requires a constrained geography contract, not independent locality sampling"
            )
    unresolved = [
        b.logical_key
        for b in source.template.bindings
        if b.group_kind == "route"
        and b.value_kind == "location"
        and not b.target_paths
        and not b.dependency_paths
        and not b.dependency_bindings
        and not render._explicit_unknown_placeholder(b)
        and b.realization.mode != "static"
    ]
    if unresolved:
        raise ValueError(
            "source-only route facts lack dependency contracts: " + ", ".join(unresolved)
        )
    independent_transport = [
        b.logical_key
        for b in source.template.bindings
        if b.group_kind in {"transport", "route"}
        and b.value_kind == "equipment"
        and b.derivation not in transport_derivations.DERIVATIONS
        and not b.target_paths
        and not b.dependency_paths
        and not b.dependency_bindings
        and not render._explicit_unknown_placeholder(b)
        and b.realization.mode != "static"
    ]
    if independent_transport:
        raise ValueError(
            "source-only transport facts lack route dependency contracts: "
            + ", ".join(independent_transport)
        )
    projected = projected_auxiliary_values(source.template, source.target)
    geographic_keys = {
        member.logical_key
        for entity in source.template.auxiliary_semantic_plan.entities
        if entity.target_party_path is not None
        for member in entity.members
        if member.field in {"city", "country", "country_code"}
    }
    if overlapping := geographic_keys.intersection(projected):
        raise ValueError(
            "geographic auxiliary is also a lexical projection; compilation review "
            "is required: " + ", ".join(sorted(overlapping))
        )
    phone_paths = {
        path: value
        for path, value in render._flatten_leaves(source.target).items()
        if re.fullmatch(r"documentPatch\.parties\..+\.contactDetails\.phoneNumbers\[\d+\]", path)
        and not path.startswith("documentPatch.parties.carrier.")
    }
    for path, value in phone_paths.items():
        if not isinstance(value, str):
            raise ValueError("source party phone is not a string: " + path)
        if any(path in b.target_paths for b in source.template.bindings):
            continue
        digits = re.sub(r"\D", "", value)
        for binding in source.template.bindings:
            if binding.value_kind != "identifier" or binding.target_paths:
                continue
            if any(
                re.fullmatch(r"(?i)(?:TEL|PHONE)" + re.escape(digits), s.source_text)
                for s in binding.occurrences
            ):
                raise ValueError(
                    "labeled telephone is an unbound fixed-width auxiliary identifier; "
                    "requires a phone realization contract before country sampling: " + path
                )


def _shared_route_locations(source: targets.SourceTemplate) -> tuple[tuple[str, ...], ...]:
    """Forward actual name-slot equality, never equality of country names alone."""
    groups = []
    for binding in source.template.bindings:
        if binding.target_relationship != "shared_value_equality":
            continue
        roles = tuple(
            match[1]
            for path in binding.target_paths
            if (match := re.fullmatch(r"documentPatch\.route\.([^.]+)\.name", path))
        )
        if len(roles) > 1:
            groups.append(roles)
    return tuple(groups)


def _validate_shared_phone_context(
    source: targets.SourceTemplate, context: Mapping[str, PartyGeography]
) -> None:
    """Reject incompatible private dial-code choices before publishing a route.

    A compiled shared phone is one generated value, even when no party name or
    address is printed. This constrains its geographic context; it does not assert
    that the parties are identical or add missing geography to extraction labels.
    """
    for binding in source.template.bindings:
        owners = {
            path.split(".contactDetails.phoneNumbers[", 1)[0]
            for path in binding.target_paths
            if ".contactDetails.phoneNumbers[" in path
            and not path.startswith("documentPatch.parties.carrier.")
        }
        if len(owners) < 2:
            continue
        if not owners <= context.keys():
            raise ValueError("shared phone owner lacks sampled geographic context")
        if len({context[owner].country_code for owner in owners}) != 1:
            raise ValueError("shared phone field has inconsistent sampled countries")


def sample_projection(
    source: targets.SourceTemplate,
    *,
    sample_id: str,
    seed: int,
    support: ScenarioSupport,
    countries: CountryRegistry,
    country_codes: Mapping[str, str],
    registry_exploration_permyriad: int,
    maximum_candidates: int,
) -> RouteProjection:
    require_route_contract(source)
    if maximum_candidates < 1:
        raise ValueError("route candidate limit must be positive")
    known_auxiliary = {
        member.logical_key
        for entity in source.template.auxiliary_semantic_plan.entities
        for member in entity.members
    }
    pending = frozenset(
        r["auxiliaryKey"]
        for r in targets.lexical_contract(source)
        if r.get("auxiliaryKey") in known_auxiliary
    )
    rejected = []
    physical_source = route_derivations.physical_source(source.target, source.template.bindings)
    route_derivations.validate_subdivision_evidence(
        physical_source, source.template.bindings, support=support, countries=countries
    )
    shared_locations = _shared_route_locations(source)
    for binding in source.template.bindings:
        if binding.derivation == "sampled_route_country_code":
            owner = physical_source["documentPatch"]["route"][route_derivations.endpoint(binding)]
            source_country = owner.get("country")
            if source_country is None:
                raise ValueError("source-only route country code lacks a physical country owner")
            code = country_codes.get(render._alphanumeric(source_country).casefold())
            if code is None or any(
                slot.source_text.upper() != code for slot in binding.occurrences
            ):
                raise ValueError("source-only route country code contradicts its physical owner")
    stream = DeterministicStream(seed, "compiled-route-first-v1", sample_id)
    for index in range(maximum_candidates):
        try:
            attempt = stream.derive(f"candidate-{index}")
            scenario = sample_shipment_scenario(
                base_document_id=source.document_id,
                source_target=physical_source,
                support=support,
                country_registry=countries,
                stream=attempt,
                registry_exploration_permyriad=registry_exploration_permyriad,
                shared_route_locations=shared_locations,
            )
            projected = project_scenario_target(source_target=physical_source, scenario=scenario)
            projected_route = projected["documentPatch"].get("route", {})
            for role in (
                projected_route.keys() - source.target["documentPatch"].get("route", {}).keys()
            ):
                del projected_route[role]
            for role, observed in source.target["documentPatch"].get("route", {}).items():
                projected_route[role] = {
                    key: value for key, value in projected_route[role].items() if key in observed
                }
            if "route" not in source.target["documentPatch"]:
                projected["documentPatch"].pop("route", None)
            context = _party_context(scenario, support=support, countries=countries, stream=attempt)
            _validate_shared_phone_context(source, context)
            auxiliary = _auxiliary_geography(source, projected, context, country_codes)
            route_values = route_derivations.values(source.template.bindings, scenario.to_dict())
            if auxiliary.keys() & route_values.keys():
                raise ValueError("party and route contracts own the same auxiliary surface")
            for binding in source.template.bindings:
                if binding.logical_key in route_values:
                    render._validate_binding_format(
                        source=source.source,
                        template=source.template.byte_template,
                        output=render._render_text_candidate(
                            binding, route_values[binding.logical_key]
                        ),
                    )
            auxiliary.update(route_values)
            render._validate_target_compatibility(
                source=source.source,
                source_target=source.source_target,
                target=projected,
                template=source.template,
                pending_auxiliary_keys=(
                    pending
                    | frozenset(auxiliary)
                    | (
                        frozenset(projected_auxiliary_values(source.template, projected))
                        & known_auxiliary
                    )
                ),
            )
        except ValueError as error:
            rejected.append(str(error))
            continue
        old, new = render._flatten_leaves(source.target), render._flatten_leaves(projected)
        if old.keys() != new.keys():
            raise ValueError("route sampler changed target leaf topology")
        updates = {p: value for p, value in new.items() if value != old[p]}
        if any(not isinstance(value, str) for value in updates.values()):
            raise ValueError("route sampler produced a non-string geographic fact")
        return RouteProjection(
            schema_version=1,
            sample_id=sample_id,
            source_document_id=source.document_id,
            source_target_sha256=sha256_bytes(canonical_json_bytes(source.target)),
            updates=cast(dict[str, str], updates),
            party_geography=context,
            auxiliary_values=auxiliary,
            scenario=scenario.to_dict(),
            rejected_candidates=tuple(rejected),
        )
    raise ValueError(f"no representable route in {maximum_candidates} candidates: {rejected[-1]}")
