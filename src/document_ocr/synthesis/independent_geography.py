"""Source-certified geography for independent flat synthesis.

This module owns no registry selection, provider call, source-ID policy, or
artifact path.  Callers supply the exact reviewed certificate bytes and the
expected file digest.  Geography that is absent from a certified source
assembly remains absent: neither country nor locality is inferred here.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from string import Formatter
from threading import RLock
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from document_ocr.hashing import canonical_json_sha256, sha256_bytes
from document_ocr.synthesis.curated import digest, flat, interpolate
from document_ocr.synthesis.curated_campaign import related_actor_contexts
from document_ocr.synthesis.curated_equipment_inventory import (
    EquipmentInventoryContract,
    compile_equipment_inventory,
)
from document_ocr.synthesis.curated_party_geography import (
    validate_postal_component_identities,
)
from document_ocr.synthesis.curated_scenarios import ScenarioLocation
from document_ocr.synthesis.curated_wording import (
    PostalGeographyCounts,
    postal_geography_counts,
)

_COMPONENT = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_SHA256 = r"^[0-9a-f]{64}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("*", mode="after")
    @classmethod
    def nonblank_strings(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("certificate strings must not be blank")
        return value


class PostalAssemblyOwnership(_StrictModel):
    """An explicit owner; ``component=None`` means immutable source framing."""

    component: str | None


class PostalAssemblySpan(_StrictModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self) -> PostalAssemblySpan:
        if self.end <= self.start:
            raise ValueError("postal source span end must follow start")
        if len(self.text.encode()) != self.end - self.start:
            raise ValueError("postal source span byte length differs from its text")
        return self


class PostalAssemblyEvidence(_StrictModel):
    locality: str | None
    country: str | None
    review: str = Field(min_length=1)


class PostalAssemblyCertificate(_StrictModel):
    key: str = Field(min_length=1)
    postal_keys: tuple[str, ...] = Field(min_length=1)
    identity_keys: tuple[str, ...]
    expression: str = Field(min_length=1)
    scope: Literal["address", "country_only", "locality_country"]
    locality: PostalAssemblyOwnership | None
    country: PostalAssemblyOwnership | None
    source_occurrences: tuple[PostalAssemblySpan, ...] = Field(min_length=1)
    evidence: PostalAssemblyEvidence

    @model_validator(mode="after")
    def coherent(self) -> PostalAssemblyCertificate:
        if set(_COMPONENT.findall(self.expression)) != set(self.postal_keys):
            raise ValueError("postal assembly expression must reference exactly its postal keys")
        if len(set(self.postal_keys)) != len(self.postal_keys):
            raise ValueError("postal assembly contains duplicate component keys")
        if len(set(self.identity_keys)) != len(self.identity_keys):
            raise ValueError("postal assembly contains duplicate identity keys")
        if set(self.postal_keys) & set(self.identity_keys):
            raise ValueError("postal and identity owners must be disjoint")
        if any(
            owner is not None
            and owner.component is not None
            and owner.component not in self.postal_keys
            for owner in (self.locality, self.country)
        ):
            raise ValueError("geography owner is outside this postal assembly")
        if self.scope == "country_only" and (self.locality is not None or self.country is None):
            raise ValueError("country-only assembly must own country only")
        if self.scope == "locality_country" and (self.locality is None or self.country is None):
            raise ValueError("locality-country assembly must own both components")
        for label, owner in (("locality", self.locality), ("country", self.country)):
            if (owner is None) != (getattr(self.evidence, label) is None):
                raise ValueError("postal geography ownership disagrees with reviewed evidence")
        return self


class SourceGeographyCertificate(_StrictModel):
    source_sha256: str = Field(pattern=_SHA256)
    effective_contract_sha256: str = Field(pattern=_SHA256)
    ownership_sha256: str = Field(pattern=_SHA256)
    assemblies: tuple[PostalAssemblyCertificate, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_assemblies(self) -> SourceGeographyCertificate:
        keys = [assembly.key for assembly in self.assemblies]
        if len(keys) != len(set(keys)):
            raise ValueError("source geography certificate contains duplicate assembly keys")
        return self


class _CertificateEnvelope(_StrictModel):
    schemaVersion: Literal[1]
    inventorySha256: str = Field(pattern=_SHA256)
    sources: dict[str, SourceGeographyCertificate] = Field(min_length=1)

    @model_validator(mode="after")
    def nonempty_source_ids(self) -> _CertificateEnvelope:
        if any(not source_id for source_id in self.sources):
            raise ValueError("geography certificate source IDs must be nonempty")
        return self


@dataclass(frozen=True, slots=True)
class IndependentGeographyCertificates:
    """Validated reviewed authority plus its caller-pinned byte identity."""

    file_sha256: str
    inventory_sha256: str
    sources: Mapping[str, SourceGeographyCertificate]


class GeographyFinding(_StrictModel):
    assembly_key: str
    variable_keys: tuple[str, ...]
    stage: Literal["wording", "rendered"]
    message: str
    repairable: bool = True


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"geography certificate JSON contains duplicate key: {key}")
        result[key] = value
    return result


def load_independent_geography_certificates(
    payload: bytes, *, expected_sha256: str
) -> IndependentGeographyCertificates:
    """Validate caller-owned certificate bytes without selecting a path or artifact."""

    if not re.fullmatch(_SHA256, expected_sha256):
        raise ValueError("expected geography certificate SHA-256 is invalid")
    actual = sha256_bytes(payload)
    if actual != expected_sha256:
        raise ValueError("geography certificate file SHA-256 differs")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("geography certificate file is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("geography certificate file must contain one object")
    envelope = _CertificateEnvelope.model_validate_json(payload, strict=True)
    return IndependentGeographyCertificates(
        file_sha256=actual,
        inventory_sha256=envelope.inventorySha256,
        sources=MappingProxyType(dict(envelope.sources)),
    )


def _normalized(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    folded = "".join(character for character in folded if not unicodedata.combining(character))
    return " ".join(re.sub(r"[^\w]+", " ", folded).split())


def _source_normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _counts(text: str, location: ScenarioLocation) -> PostalGeographyCounts:
    return postal_geography_counts(
        text,
        locality=location.name,
        country=location.country,
        country_code=location.country_code,
    )


def _geography_errors(
    text: str, assembly: PostalAssemblyCertificate, location: ScenarioLocation
) -> list[str]:
    count = _counts(text, location)
    errors: list[str] = []
    if assembly.country is not None:
        if count.country + count.terminal_code != 1:
            errors.append("country must occur once as the supplied name or ISO2 code")
    elif (
        count.short_country
        or count.terminal_code
        or (count.country and not count.same_name)
        or (count.same_name and assembly.locality is None and count.country)
    ):
        errors.append("country declaration is omitted by this source assembly")
    if assembly.locality is not None:
        if count.locality != 1:
            errors.append("locality must occur once with the supplied spelling")
    elif count.locality and (not count.same_name or assembly.country is None):
        errors.append("locality declaration is omitted by this source assembly")
    if assembly.scope == "country_only":
        if _normalized(text) not in {
            _normalized(location.country),
            location.country_code.casefold(),
        }:
            errors.append("country-only source cannot expand into a postal address")
    elif assembly.scope == "locality_country":
        locality = _normalized(location.name)
        countries = {_normalized(location.country), location.country_code.casefold()}
        allowed = {locality + " " + country for country in countries}
        if count.same_name:
            allowed.add(locality)
        if _normalized(text) not in allowed:
            errors.append("locality-country source cannot expand into street/premises content")
    return errors


class IndependentGeographyGate:
    """Gate one runtime instance against explicit immutable geography certificates."""

    def __init__(self, certificates: IndependentGeographyCertificates):
        if not isinstance(certificates, IndependentGeographyCertificates):
            raise TypeError("geography gate requires validated explicit certificates")
        self._certificates = certificates
        self._prepared: dict[str, tuple[str, str]] = {}
        self._prepared_lock = RLock()

    @property
    def certificates(self) -> Mapping[str, SourceGeographyCertificate]:
        return self._certificates.sources

    @property
    def certificate_file_sha256(self) -> str:
        return self._certificates.file_sha256

    @property
    def inventory_sha256(self) -> str:
        return self._certificates.inventory_sha256

    @staticmethod
    def _blueprint_pins(blueprint: Any) -> tuple[str, str]:
        return (
            digest(blueprint.contract.model_dump(mode="json")),
            digest(dict(blueprint.ownership_data)),
        )

    def bind_blueprint(self, ordinary_blueprint: Any, prepared_blueprint: Any) -> None:
        """Prove preparation preserves postal ownership, then pin its exact snapshot."""

        certificate = self._certificate(ordinary_blueprint, allow_prepared=False)
        if certificate is None:
            if related_actor_contexts(prepared_blueprint) != related_actor_contexts(
                ordinary_blueprint
            ):
                raise ValueError("preparation introduced unreviewed related postal context")
            self._certificate(prepared_blueprint, allow_prepared=False)
            return
        if (
            ordinary_blueprint.document_id != prepared_blueprint.document_id
            or ordinary_blueprint.source != prepared_blueprint.source
        ):
            raise ValueError("postal preparation changed source identity or OCR bytes")
        if self._postal_state(ordinary_blueprint, certificate) != self._postal_state(
            prepared_blueprint, certificate
        ):
            raise ValueError("preparation changed certified postal/name/component ownership")
        pins = self._blueprint_pins(prepared_blueprint)
        with self._prepared_lock:
            existing = self._prepared.get(prepared_blueprint.document_id)
            if existing is not None and existing != pins:
                raise ValueError("postal gate already binds a different prepared source snapshot")
            self._prepared[prepared_blueprint.document_id] = pins
            try:
                self._certificate(prepared_blueprint)
            except Exception:
                if existing is None:
                    del self._prepared[prepared_blueprint.document_id]
                else:
                    self._prepared[prepared_blueprint.document_id] = existing
                raise

    @staticmethod
    def _postal_state(blueprint: Any, certificate: SourceGeographyCertificate) -> tuple[Any, ...]:
        protected = {
            key
            for assembly in certificate.assemblies
            for key in (*assembly.postal_keys, *assembly.identity_keys)
        }
        variables = {
            variable.key: variable.model_dump(mode="json")
            for variable in blueprint.contract.variables
            if variable.key in protected
        }
        targets = [
            target.model_dump(mode="json")
            for target in blueprint.contract.targets
            if protected.intersection(_COMPONENT.findall(target.expression))
        ]
        spans = [
            span for assembly in certificate.assemblies for span in assembly.source_occurrences
        ]
        regions = [
            asdict(region)
            for region in blueprint.regions
            if region.curated_key in protected
            or any(region.start < span.end and region.end > span.start for span in spans)
        ]
        actors = {
            key: value
            for key, value in related_actor_contexts(blueprint).items()
            if key in protected
        }
        identities = {
            key: value
            for key, value in blueprint.ownership_data.get("identity_kinds", {}).items()
            if key in protected
        }
        aliases = {
            key: value
            for key, value in blueprint.ownership_data.get("surfaces", {}).items()
            if value.get("lexical_key") in protected
            or protected.intersection(_COMPONENT.findall(value.get("expression", "")))
        }
        return variables, targets, regions, actors, identities, aliases

    def locality_projection_view(
        self, original_blueprint: Any, projected_blueprint: Any
    ) -> IndependentGeographyGate:
        """Bind unchanged postal ownership in an isolated, per-render gate.

        Route-only projection changes the complete blueprint hash. It must not
        weaken certificate matching or replace the shared prepared-source pin.
        """
        certificate = self._certificate(original_blueprint)
        if (
            original_blueprint.document_id != projected_blueprint.document_id
            or original_blueprint.source != projected_blueprint.source
        ):
            raise ValueError("locality projection changed source identity or OCR bytes")
        before, after = flat(original_blueprint.target), flat(projected_blueprint.target)
        if any(path not in before or before[path] != value for path, value in after.items()) or any(
            not _is_locality_country_path(path) for path in before.keys() - after.keys()
        ):
            raise ValueError("locality projection changed non-country target values")
        if certificate is None:
            self._certificate(projected_blueprint, allow_prepared=False)
            return self
        if self._postal_state(original_blueprint, certificate) != self._postal_state(
            projected_blueprint, certificate
        ):
            raise ValueError("locality projection changed certified postal/name ownership")
        view = IndependentGeographyGate(self._certificates)
        view._prepared[projected_blueprint.document_id] = self._blueprint_pins(projected_blueprint)
        view._certificate(projected_blueprint)
        return view

    def _certificate(
        self, blueprint: Any, *, allow_prepared: bool = True
    ) -> SourceGeographyCertificate | None:
        actors = related_actor_contexts(blueprint)
        variables = {variable.key: variable for variable in blueprint.contract.variables}
        postal = {key for key in actors if variables[key].kind == "postal"}
        if not postal:
            if blueprint.document_id in self.certificates:
                raise ValueError("certified source-only postal actor context disappeared")
            return None
        if blueprint.document_id not in self.certificates:
            raise ValueError("source-only postal geography has no reviewed assembly certificate")
        certificate = self.certificates[blueprint.document_id]
        pins = self._blueprint_pins(blueprint)
        original = (
            certificate.effective_contract_sha256,
            certificate.ownership_sha256,
        )
        with self._prepared_lock:
            prepared = self._prepared.get(blueprint.document_id) if allow_prepared else None
        if certificate.source_sha256 != digest(blueprint.source.encode()) or (
            pins != original and pins != prepared
        ):
            raise ValueError(
                "source-only postal assembly certificate differs from active blueprint"
            )
        keys = [key for assembly in certificate.assemblies for key in assembly.postal_keys]
        if set(keys) != postal or len(keys) != len(postal):
            raise ValueError("source-only postal certificates must partition every component once")
        original_values = {
            variable.key: variable.value for variable in blueprint.contract.variables
        }
        source = blueprint.source.encode()
        for assembly in certificate.assemblies:
            declarations = [actors[key] for key in assembly.postal_keys]
            if not all(declaration == declarations[0] for declaration in declarations):
                raise ValueError("one postal assembly cannot join different actor contexts")
            if any(
                key not in variables or variables[key].kind != "name"
                for key in assembly.identity_keys
            ):
                raise ValueError("postal identity guard requires explicitly typed name keys")
            baseline = interpolate(assembly.expression, original_values)
            for span in assembly.source_occurrences:
                if source[span.start : span.end].decode() != span.text or _source_normalized(
                    baseline
                ) != _source_normalized(span.text):
                    raise ValueError(
                        "postal assembly source span differs from certified expression"
                    )
            literal = _COMPONENT.sub("", assembly.expression)
            for label, owner in (("locality", assembly.locality), ("country", assembly.country)):
                evidence = getattr(assembly.evidence, label)
                if evidence is not None:
                    owned = (
                        original_values[owner.component]
                        if owner is not None and owner.component is not None
                        else literal
                    )
                    if _source_normalized(evidence) not in _source_normalized(owned):
                        raise ValueError("postal geography evidence is outside its declared owner")
        return certificate

    @staticmethod
    def _location(
        blueprint: Any,
        assembly: PostalAssemblyCertificate,
        scenario: Any,
    ) -> ScenarioLocation:
        actor = blueprint.ownership_data["related_actors"][assembly.postal_keys[0]]
        return (
            ScenarioLocation.model_validate(actor["location"])
            if "location" in actor
            else scenario.party_localities[actor["context_party"]]
        )

    def location_for(
        self, blueprint: Any, assembly: PostalAssemblyCertificate, scenario: Any
    ) -> ScenarioLocation:
        """Expose only the explicit actor/scenario location used by the gate."""

        self._certificate(blueprint)
        return self._location(blueprint, assembly, scenario)

    def prepare_request(self, request: Any, blueprint: Any, scenario: Any) -> Any:
        certificate = self._certificate(blueprint)
        if certificate is None:
            return request
        by_key: dict[str, str] = {}
        contexts: list[str] = []
        request_keys = {field.key for field in request.fields}
        for assembly in certificate.assemblies:
            if not set(assembly.postal_keys) <= request_keys:
                raise ValueError("wording request omits a certified related postal component")
            location = self._location(blueprint, assembly, scenario)
            rules: list[str] = []
            for label, value, owner in (
                ("LOCALITY", location.name, assembly.locality),
                ("COUNTRY", location.country, assembly.country),
            ):
                if owner is None:
                    rules.append(f"{label}: omitted; do not introduce {value!r} as a declaration.")
                elif owner.component is None:
                    rules.append(
                        f"{label}: {value!r} is supplied once by unchanged literal framing; "
                        "do not repeat it in generated components."
                    )
                else:
                    rules.append(
                        f"{label}: {value!r}, exactly once, owned by {{{owner.component}}}."
                    )
            scope = {
                "address": "Preserve this assembly's postal component hierarchy.",
                "country_only": (
                    "Country name or ISO2 code only; no locality, street, postcode "
                    "or other postal content."
                ),
                "locality_country": (
                    "Locality and country only, in that order; no street, postcode "
                    "or other postal content."
                ),
            }[assembly.scope]
            contexts.append(
                f"SOURCE-ONLY POSTAL ASSEMBLY {assembly.key}: {assembly.expression}\n"
                + "\n".join(rules)
                + "\n"
                + scope
                + " This is not an extraction-party target. Keep separately generated "
                "identities out of postal components. Context geography describes the actor; "
                "only explicitly owned geography is printed. If locality and country coincide, "
                "one printed mention suffices. Choose new site/street/district names without "
                "repeating the supplied geography."
            )
            for key in assembly.postal_keys:
                ownership: list[str] = []
                for label, value, owner in (
                    ("locality", location.name, assembly.locality),
                    ("country", location.country, assembly.country),
                ):
                    if owner is not None and owner.component == key:
                        ownership.append(f"include {label} {value!r} once")
                    else:
                        ownership.append(f"do not add {label} {value!r} as a declaration")
                by_key[key] = (
                    f" Source-only assembly {assembly.key}: return only component {{{key}}}; "
                    + "; ".join(ownership)
                    + ". "
                    + scope
                )
        fields = tuple(
            replace(field, requirement=field.requirement + " " + by_key[field.key])
            if field.key in by_key
            else field
            for field in request.fields
        )
        return replace(
            request,
            fields=fields,
            context=request.context + "\n\n" + "\n\n".join(contexts),
        )

    def validate_values(
        self, blueprint: Any, values: Mapping[str, str], scenario: Any
    ) -> list[GeographyFinding]:
        certificate = self._certificate(blueprint)
        if certificate is None:
            return []
        findings: list[GeographyFinding] = []
        for assembly in certificate.assemblies:
            missing = (set(assembly.postal_keys) | set(assembly.identity_keys)) - values.keys()
            if missing:
                raise ValueError(
                    "source-only postal validation is missing generated values: "
                    + repr(sorted(missing))
                )
            location = self._location(blueprint, assembly, scenario)
            text = interpolate(assembly.expression, dict(values))
            errors = _geography_errors(text, assembly, location)
            literal = _COMPONENT.sub("", assembly.expression)
            for label, owner in (("locality", assembly.locality), ("country", assembly.country)):
                if owner is not None:
                    value = values[owner.component] if owner.component is not None else literal
                    counts = _counts(value, location)
                    number = (
                        counts.locality
                        if label == "locality"
                        else counts.country + counts.terminal_code
                    )
                    if number != 1:
                        errors.append(f"{label} is absent or repeated in its certified owner")
            expression = (
                " ".join("{" + key + "}" for key in assembly.identity_keys)
                + " "
                + assembly.expression
            )
            try:
                validate_postal_component_identities(
                    blueprint.contract,
                    expression,
                    dict(values),
                    locality=location.name,
                    country=location.country,
                )
            except ValueError as error:
                errors.append(str(error))
            if errors:
                findings.append(
                    GeographyFinding(
                        assembly_key=assembly.key,
                        variable_keys=assembly.postal_keys,
                        stage="wording",
                        message="; ".join(errors),
                    )
                )
        return findings

    def validate_rendered(
        self, blueprint: Any, candidate: Mapping[str, Any], scenario: Any
    ) -> list[GeographyFinding]:
        findings = self.validate_values(blueprint, candidate["renderValues"], scenario)
        certificate = self._certificate(blueprint)
        if certificate is None:
            return findings
        source = blueprint.source.encode()
        rendered = candidate["joinedRawText"].encode()
        proof = candidate["proof"]
        if proof["sourceSha256"] != digest(source) or proof["renderedSha256"] != digest(rendered):
            raise ValueError("related postal rendered proof identity mismatch")
        edits = sorted(proof["edits"], key=lambda edit: edit["byteStart"])
        replay: list[bytes] = []
        end = 0
        for edit in edits:
            start, stop = edit["byteStart"], edit["byteEnd"]
            if start < end or source[start:stop].decode() != edit["before"]:
                raise ValueError("related postal rendered proof has overlapping or invalid edits")
            replay.extend((source[end:start], edit["after"].encode()))
            end = stop
        replay.append(source[end:])
        if b"".join(replay) != rendered:
            raise ValueError("related postal rendered proof does not replay to actual text")
        for assembly in certificate.assemblies:
            location = self._location(blueprint, assembly, scenario)
            for span in assembly.source_occurrences:
                owned = [
                    edit
                    for edit in edits
                    if edit["byteStart"] < span.end and edit["byteEnd"] > span.start
                ]
                if any(
                    edit["byteStart"] < span.start or edit["byteEnd"] > span.end for edit in owned
                ):
                    findings.append(
                        GeographyFinding(
                            assembly_key=assembly.key,
                            variable_keys=assembly.postal_keys,
                            stage="rendered",
                            repairable=False,
                            message=(
                                "rendered edit crosses certified source-only postal "
                                "assembly boundary"
                            ),
                        )
                    )
                    continue
                after = source[span.start : span.end]
                for edit in reversed(owned):
                    start = edit["byteStart"] - span.start
                    stop = edit["byteEnd"] - span.start
                    after = after[:start] + edit["after"].encode() + after[stop:]
                errors = _geography_errors(after.decode(), assembly, location)
                if errors:
                    findings.append(
                        GeographyFinding(
                            assembly_key=assembly.key,
                            variable_keys=assembly.postal_keys,
                            stage="rendered",
                            message="; ".join(errors),
                        )
                    )
        return findings


def _is_locality_country_path(path: str) -> bool:
    if path in {
        "documentPatch.placeOfIssue.country",
        "documentPatch.freight.paymentPlace.country",
    }:
        return True
    return (
        path.startswith("documentPatch.route.")
        and path.endswith(".country")
        and path.count(".") == 3
        and "[" not in path
    )


def _pure_pair_expression(expression: str, name_path: str, country_path: str) -> bool:
    try:
        parts = list(Formatter().parse(expression))
    except ValueError:
        return False
    fields = [part for part in parts if part[1] is not None]
    if [part[1] for part in fields] != [name_path, country_path]:
        return False
    if any(part[2] or part[3] for part in fields):
        return False
    literals = [part[0] for part in parts]
    return bool(fields[1][0]) and all(
        character.isspace() or character in ",;()/:-"
        for literal in literals
        for character in literal
    )


def _remove_scalar_path(target: dict[str, Any], path: str) -> None:
    node: Any = target
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ValueError("locality projection path is absent from its target")
    del node[parts[-1]]


def project_composite_locality_homonyms(
    plan: tuple[Any, Any, Any, dict[str, Any]], candidate: Mapping[str, Any]
) -> tuple[tuple[Any, Any, Any, dict[str, Any]], dict[str, Any] | None]:
    """Collapse only complete printed name/country pairs with equal identities."""

    blueprint, scenario, sample, target = plan
    if candidate["proof"]["sourceSha256"] != digest(blueprint.source.encode()):
        raise ValueError("composite locality proof has a stale source hash")
    values = flat(candidate["target"])
    regions_by_id = {(region.key, region.occurrence): region for region in blueprint.regions}
    recipes = blueprint.ownership_data.get("surfaces", {})
    edits_by_path: dict[str, list[Mapping[str, Any]]] = {}
    for edit in candidate["proof"]["edits"]:
        for path in edit["targetPaths"]:
            if _is_locality_country_path(path):
                edits_by_path.setdefault(path, []).append(edit)
    projected_regions: set[tuple[str, int]] = set()
    omitted: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for country_path, edits in edits_by_path.items():
        name_path = country_path.removesuffix(".country") + ".name"
        country, name = values.get(country_path), values.get(name_path)
        if not isinstance(name, str) or not isinstance(country, str):
            continue
        if _normalized(name) != _normalized(country):
            continue
        # Split scalar owners need a reviewed complete inline frame before the
        # existing projection can remove punctuation and the redundant country
        # together. Never silently pass a known duplicate through another owner
        # representation, or delete independently captioned country evidence.
        for country_edit in edits:
            if set(country_edit["targetPaths"]) != {country_path}:
                continue
            for name_edit in candidate["proof"]["edits"]:
                if set(name_edit["targetPaths"]) != {name_path}:
                    continue
                gap = blueprint.source.encode()[name_edit["byteEnd"] : country_edit["byteStart"]]
                if (
                    name_edit["byteEnd"] < country_edit["byteStart"]
                    and re.fullmatch(rb"[ \t,;()/:-]+", gap)
                    and _normalized(name_edit["after"])
                    == _normalized(country_edit["after"])
                    == _normalized(name)
                ):
                    raise ValueError(
                        "repeated inline locality/country needs complete frame ownership: "
                        + country_path
                    )
        owners = [region for region in blueprint.regions if country_path in region.target_paths]
        if len(owners) != len(edits) or {(region.key, region.occurrence) for region in owners} != {
            (edit["key"], edit["occurrence"]) for edit in edits
        }:
            raise ValueError("homonymous country proof differs from its compiled owners")
        selected: list[Any] = []
        for edit in edits:
            region = regions_by_id[edit["key"], edit["occurrence"]]
            recipe = recipes.get(region.key, {})
            binding = blueprint.historical_bindings.get(region.key)
            if (
                region.curated_key is not None
                or set(region.target_paths) != {name_path, country_path}
                or set(recipe) != {"expression"}
                or not _pure_pair_expression(recipe["expression"], name_path, country_path)
                or binding is None
                or set(binding["target_paths"]) != {name_path, country_path}
            ):
                break
            expression = recipe["expression"]
            expected = expression.replace("{" + name_path + "}", name).replace(
                "{" + country_path + "}", country
            )
            if _normalized(edit["after"]) != _normalized(expected):
                break
            if (
                edit["byteStart"] != region.start
                or edit["byteEnd"] != region.end
                or edit["before"] != region.source
                or blueprint.source.encode()[region.start : region.end] != region.source.encode()
            ):
                raise ValueError("composite locality proof is not its exact source frame")
            selected.append(region)
        else:
            keys = {region.key for region in selected}
            all_key_regions = {region for region in blueprint.regions if region.key in keys}
            if all_key_regions != set(selected):
                raise ValueError(
                    "composite locality recipe is shared with an unreviewed occurrence"
                )
            projected_regions.update((region.key, region.occurrence) for region in selected)
            omitted.add(country_path)
            evidence.append(
                {
                    "countryPath": country_path,
                    "namePath": name_path,
                    "name": name,
                    "occurrences": len(selected),
                    "owners": [
                        {
                            "key": region.key,
                            "occurrence": region.occurrence,
                            "byteStart": region.start,
                            "byteEnd": region.end,
                            "source": region.source,
                            "expressionBefore": recipes[region.key]["expression"],
                            "expressionAfter": "{" + name_path + "}",
                            "printedBefore": edit["after"],
                        }
                        for region, edit in zip(selected, edits, strict=True)
                    ],
                }
            )
    if not omitted:
        return plan, None
    source_target = copy.deepcopy(blueprint.target)
    projected_target = copy.deepcopy(target)
    for value in (source_target, projected_target):
        for path in omitted:
            _remove_scalar_path(value, path)
    ownership = copy.deepcopy(dict(blueprint.ownership_data))
    bindings = copy.deepcopy(dict(blueprint.historical_bindings))
    for field in evidence:
        country_path, name_path = field["countryPath"], field["namePath"]
        for owner in field["owners"]:
            key = owner["key"]
            ownership["surfaces"][key] = {"expression": "{" + name_path + "}"}
            bindings[key]["target_paths"] = [name_path]
            for occurrence in bindings[key]["occurrences"]:
                if "target_paths" in occurrence:
                    occurrence["target_paths"] = [
                        path for path in occurrence["target_paths"] if path != country_path
                    ]
    projected = replace(
        blueprint,
        target=source_target,
        historical_bindings=bindings,
        ownership_data=ownership,
        contract=blueprint.contract.model_copy(
            update={
                "targets": [
                    binding for binding in blueprint.contract.targets if binding.path not in omitted
                ]
            }
        ),
        regions=tuple(
            replace(
                region,
                target_paths=tuple(path for path in region.target_paths if path not in omitted),
            )
            if (region.key, region.occurrence) in projected_regions
            else region
            for region in blueprint.regions
        ),
    )
    projected_scenario = scenario.model_copy(
        update={
            "replacements": {
                key: value for key, value in scenario.replacements.items() if key not in omitted
            }
        }
    )
    return (projected, projected_scenario, sample, projected_target), {
        "rule": (
            "Collapse only certified two-scalar composites whose locality/country identities "
            "and actual printed presentations are equal; retain independent country evidence."
        ),
        "beforeCandidateSha256": digest(candidate),
        "sourceSha256": digest(blueprint.source.encode()),
        "fields": evidence,
        "preservation": (
            "Other roles, short-name owners, lexical fields and source bytes unchanged."
        ),
    }


def recompile_projected_equipment_inventory(
    ordinary: Any,
    projected: Any,
    original_target: Mapping[str, Any],
    target: Mapping[str, Any],
    inventory: EquipmentInventoryContract,
) -> EquipmentInventoryContract:
    """Compose locality-only projection with unchanged certified equipment ownership."""

    if (
        ordinary.document_id != projected.document_id
        or ordinary.source != projected.source
        or inventory.source_id != ordinary.document_id
        or inventory.source_sha256 != digest(ordinary.source.encode())
        or inventory.source_target_sha256 != canonical_json_sha256(ordinary.target)
    ):
        raise ValueError("projected equipment inventory has a stale original source certificate")
    before, after = flat(ordinary.target), flat(projected.target)
    omitted = before.keys() - after.keys()
    if (
        not omitted
        or after.keys() - before.keys()
        or any(before[path] != value for path, value in after.items())
        or not all(_is_locality_country_path(path) for path in omitted)
    ):
        raise ValueError("inventory projection must remove only scalar locality countries")
    sampled_before, sampled_after = flat(original_target), flat(target)
    if (
        sampled_before.keys() - sampled_after.keys() != omitted
        or sampled_after.keys() - sampled_before.keys()
        or any(sampled_before[path] != value for path, value in sampled_after.items())
    ):
        raise ValueError("sampled target differs from the certified locality-only projection")
    for path in omitted:
        name_path = path.removesuffix(".country") + ".name"
        country, name = sampled_before.get(path), sampled_before.get(name_path)
        if (
            not isinstance(country, str)
            or not isinstance(name, str)
            or _normalized(country) != _normalized(name)
            or not any(path in region.target_paths for region in ordinary.regions)
        ):
            raise ValueError("removed country is not an owned sampled locality homonym")
    current = compile_equipment_inventory(projected, allow_uniform_aggregate=True)
    old_state, new_state = asdict(inventory), asdict(current)
    new_state["source_target_sha256"] = old_state["source_target_sha256"]
    if new_state != old_state:
        raise ValueError("locality projection changed certified equipment rows or owners")
    return current
