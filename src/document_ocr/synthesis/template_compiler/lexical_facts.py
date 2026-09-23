"""Host-owned commodity wording and contacts, before target acceptance.

The model owns organization/address language, not chemical identity, equipment
facts, telephone numbering, or item allocation. Split cargo descriptions require
a reviewed source-pinned recipe; a missing recipe is an error, never a fallback.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_from_surface_pattern,
    surface_pattern,
)

from . import (
    cargo_identifiers,
    contact_values,
    lexical_partitions,
    package_equations,
    package_prose,
    temperature_prose,
)
from . import complete_targets as targets
from . import descendant as render
from .cargo_scenarios import CargoScenario
from .route_projection import RouteProjection


class FragmentRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    role: Literal["item", "reference", "detail", "literal"]
    source: str
    identity_indices: tuple[int, ...] = ()


class ReferenceRecipe(BaseModel):
    """Reviewed semantic role for a source-pinned standalone mark/reference."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    role: Literal["source_shaped_code", "source_shaped_code_list", "shipment_mark"]
    source: str


class CargoLexicalContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    source_template_sha256: str
    lexical_contract_sha256: str
    fragments: dict[str, FragmentRecipe]
    references: dict[str, ReferenceRecipe] = Field(default_factory=dict)


def _validate_literal_fragment(field: Mapping[str, Any], recipe: FragmentRecipe) -> None:
    """Only a proved form caption may survive a change of commodity identity.

    A model-reviewed literal is not a compatibility contract for a retained
    product, grade, composition or technical specification. The currently
    supported split caption is PART followed by its printed number caption;
    every occurrence must prove that frame independently.
    """
    if recipe.role != "literal":
        return
    contexts = field.get("cargoFragment", {}).get("contexts", ())
    if (
        recipe.source.strip().casefold() == "part"
        and contexts
        and all(
            context.get("sourceSlot", "").strip().casefold() == "part"
            and re.match(r"\s+NO\s*\.:", context.get("after", ""), re.I)
            for context in contexts
        )
    ):
        return
    raise ValueError("retained cargo literal lacks a proved invariant caption: " + field["key"])


def require_cargo_recipes(
    fields: Sequence[Mapping[str, Any]], contract: CargoLexicalContract | None
) -> None:
    """Reject missing source annotations before stochastic cargo draws.

    Candidate retries can solve physical incompatibility, but cannot supply a
    missing fragment owner or change its source-pinned recipe. This check uses
    only source contracts; wording length and goods identity checks remain in
    ``prepare`` because those really do depend on the sampled candidate.
    """
    for field in fields:
        paths = field.get("cargoFragment", {}).get("targetPaths", field["paths"])
        if not any(
            re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.description", p) for p in paths
        ):
            continue
        if "hostAssembly" in field:
            raise ValueError("fixed commodity literal requires a goods-specific lexical contract")
        if "cargoFragment" not in field:
            continue
        key = field["key"]
        if contract is None or key not in contract.fragments:
            raise ValueError("split cargo description lacks a reviewed lexical recipe: " + key)
        if contract.fragments[key].source != field["source"]:
            raise ValueError("cargo fragment recipe source changed")
        _validate_literal_fragment(field, contract.fragments[key])


@dataclass(frozen=True)
class HostLexicalPlan:
    values: dict[str, str]
    evidence: dict[str, Any]

    def merge(self, generated: Mapping[str, str]) -> dict[str, str]:
        if self.values.keys() & generated.keys():
            raise ValueError("provider attempted to overwrite host-owned lexical facts")
        return {**generated, **self.values}

    def preview(
        self,
        source: targets.SourceTemplate,
        target: Mapping[str, Any],
        fields: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        result = deepcopy(dict(target))
        auxiliary = {}
        for field in fields:
            if field["key"] not in self.values:
                # The preview retains ungenerated prose, but declared context
                # is already sampled. Keep that context consistent for the
                # renderability check; this is never published linguistic output.
                for path in field["paths"] if field.get("sampledContext") else ():
                    value = render._resolve_path(result, path)
                    spans = render._token_spans(value)
                    tokens = tuple(token for token, _, _ in spans)
                    edits = []
                    for old, new in field["sampledContext"].items():
                        expected = tuple(token for token, _, _ in render._token_spans(old))
                        matches = [
                            i
                            for i in range(len(tokens) - len(expected) + 1)
                            if tokens[i : i + len(expected)] == expected
                        ]
                        if len(matches) != 1:
                            raise ValueError("preview context lacks unique declared occurrence")
                        i = matches[0]
                        edits.append((spans[i][1], spans[i + len(expected) - 1][2], new))
                    for start, end, text in sorted(edits, reverse=True):
                        value = value[:start] + text + value[end:]
                    targets._set(result, path, value)
                continue
            value = targets.assemble_lexical_value(field, self.values[field["key"]])
            for path in field["paths"]:
                targets._set(result, path, value)
            if "cargoFragment" in field:
                auxiliary[field["auxiliaryKey"]] = value
        if auxiliary:
            for path, value in lexical_partitions.assembled_targets(
                source.template, auxiliary
            ).items():
                targets._set(result, path, value)
        from . import labelled_context

        for path, value in labelled_context.generated_values(source, result).items():
            targets._set(result, path, value)
        return result


def _reference(sample_id: str, key: str, seed: int) -> str:
    return (
        DeterministicStream(seed, "cargo-commercial-reference-v1", sample_id)
        .derive(key)
        .bytes(counter=0, length=6)
        .hex()
        .upper()
    )


def _definitions(field: Mapping[str, Any], scenario: CargoScenario) -> list[str]:
    paths = field.get("cargoFragment", {}).get("targetPaths", field["paths"])
    indices = {
        int(match[1])
        for path in paths
        if (match := re.fullmatch(r"documentPatch\.cargoGroups\[(\d+)\]\.description", path))
    }
    groups = scenario.target["documentPatch"].get("cargoGroups", [])
    return [
        identity.get("properShippingName") or identity["requiredDescription"]
        for i in sorted(indices)
        for identity in scenario.identities[groups[i]["groupId"]]
    ]


def _package_mentions(source: targets.SourceTemplate, target: Mapping[str, Any], path: str) -> str:
    """Retain explicitly proved count/noun assertions, not arbitrary source prose."""
    old = render._resolve_path(source.target, path)
    clauses = []
    for mention in package_prose.mentions(source.target, path, old):
        quantities = [render._resolve_path(target, p) for p in mention.quantity_paths]
        if not mention.aggregate and len(set(quantities)) != 1:
            raise ValueError("cargo description has ambiguous package quantity ownership")
        quantity = sum(quantities) if mention.aggregate else quantities[0]
        kinds = {
            render._resolve_path(target, p.removesuffix("quantity") + "typeCategory")
            for p in mention.quantity_paths
        }
        if len(kinds) != 1:
            raise ValueError("one printed package count cannot express mixed package kinds")
        clauses.append(f"{quantity} {render._package_surface(kinds.pop())}")
    return "; ".join(dict.fromkeys(clauses))


def prepare(
    source: targets.SourceTemplate,
    fields: Sequence[Mapping[str, Any]],
    *,
    scenario: CargoScenario | None,
    projection: RouteProjection | None,
    contract: CargoLexicalContract | None,
    sample_id: str,
    seed: int,
) -> HostLexicalPlan:
    if contract is not None and (
        contract.source_template_sha256 != source.original_template_sha256
        or contract.lexical_contract_sha256
        != sha256_bytes(canonical_json_bytes(targets.lexical_contract(source)))
    ):
        raise ValueError("cargo lexical contract differs from its pinned source")
    if contract is not None and not set(contract.references).issubset(f["key"] for f in fields):
        raise ValueError("reviewed reference contract names an absent field")
    values: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    references = (
        cargo_identifiers.generate_mark_references(
            source.target,
            source.template,
            DeterministicStream(seed, "host-mark-references-v1", sample_id),
            reviewed_lists={
                path: contract.references[field["key"]].source
                for field in fields
                if contract is not None
                and field["key"] in contract.references
                and contract.references[field["key"]].role == "source_shaped_code_list"
                for path in field["paths"]
            },
        )
        if scenario is not None
        else {}
    )
    ranges = (
        package_equations.mark_ranges(source.target, scenario.target)
        if scenario is not None
        else {}
    )
    overpacks = (
        package_equations.overpack_surfaces(source.target, scenario.target)
        if scenario is not None
        else {}
    )
    origin_marks = {}
    temperatures = (
        temperature_prose.generate(source.template, source.target, scenario.target)
        if scenario is not None
        else {}
    )
    if scenario is not None:
        for binding in source.template.bindings:
            origins = [
                p
                for p in binding.target_paths
                if re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.origin\.name", p)
            ]
            marks = [
                p
                for p in binding.target_paths
                if re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.marksAndNumbers\[\d+\]", p)
            ]
            if len(origins) != 1 or not marks:
                continue
            old = render._resolve_path(source.target, origins[0])
            new = render._resolve_path(scenario.target, origins[0])
            for path in marks:
                text = render._resolve_path(source.target, path)
                match = re.fullmatch(r"(?P<prefix>MADE\s+IN\s+)" + re.escape(old), text, re.I)
                if match:
                    origin_marks[path] = match["prefix"] + render._case_like(old, new)
    identifier_dependencies = {
        key
        for binding in source.template.bindings
        if binding.derivation == "same_as_binding" and binding.value_kind == "identifier"
        for key in binding.dependency_bindings
    }
    anonymous = scenario.receipt.get("anonymousEquipment") if scenario is not None else None
    for field in fields:
        key = field["key"]
        if (
            anonymous is not None
            and field.get("auxiliaryKey") in anonymous["sourceEquipmentBindings"]
        ):
            values[key] = field["source"]
            evidence[key] = dict(kind="proved_retained_anonymous_inventory", **anonymous)
            continue
        if "cargoIdentityPath" in field:
            from .cargo_identity_derivations import description

            if scenario is None:
                raise ValueError("commodity alias generation requires the complete goods scenario")
            values[key] = description(field["cargoIdentityPath"], scenario)
            evidence[key] = dict(
                kind="registry_owned_commodity_alias", owner=field["cargoIdentityPath"]
            )
            continue
        if field["paths"] and all(p in temperatures for p in field["paths"]):
            generated = {temperatures[p] for p in field["paths"]}
            if len(generated) != 1 or "hostAssembly" in field:
                raise ValueError("temperature prose has incompatible lexical ownership")
            values[key] = generated.pop()
            evidence[key] = dict(kind="compiled_temperature_dependency", paths=field["paths"])
            continue
        if contract is not None and key in contract.references:
            reference_recipe = contract.references[key]
            if (
                field["source"] != reference_recipe.source
                or not field["paths"]
                or "hostAssembly" in field
                or "cargoFragment" in field
                or not all(
                    re.fullmatch(
                        r"documentPatch\.(?:cargoGroups\[\d+\]\.marksAndNumbers|forwardingAndExportReferences)\[\d+\]",
                        path,
                    )
                    for path in field["paths"]
                )
            ):
                raise ValueError("reviewed reference has incompatible source/field ownership")
            if reference_recipe.role == "source_shaped_code":
                if not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9/.-]*", reference_recipe.source
                ) or not re.search(r"\d", reference_recipe.source):
                    raise ValueError("reviewed reference is not an opaque source-shaped code")
                value = generate_from_surface_pattern(
                    pattern=surface_pattern(reference_recipe.source),
                    stream=DeterministicStream(seed, "reviewed-reference-v1", sample_id).derive(
                        reference_recipe.source
                    ),
                    additional_excluded=reference_recipe.source,
                )
            elif reference_recipe.role == "source_shaped_code_list":
                candidates = {references[path] for path in field["paths"]}
                if len(candidates) != 1:
                    raise ValueError("reviewed reference list has inconsistent shared ownership")
                value = candidates.pop()
            else:
                value = "MARK-" + _reference(sample_id, key, seed)
            if len(re.findall(r"[A-Za-z0-9]+", value)) < max(
                (c.get("minimumWords", 1) for c in field["constraints"]), default=1
            ):
                raise ValueError("reviewed reference cannot fill its certified text segments")
            values[key] = value
            evidence[key] = dict(
                kind="reviewed_reference",
                role=reference_recipe.role,
                source=reference_recipe.source,
            )
            continue
        if field["paths"] and all(p in origin_marks for p in field["paths"]):
            generated = {origin_marks[p] for p in field["paths"]}
            if len(generated) != 1 or "hostAssembly" in field:
                raise ValueError("manufacturing-origin mark has incompatible lexical ownership")
            values[key] = generated.pop()
            evidence[key] = dict(kind="compiled_manufacturing_origin", paths=field["paths"])
            continue
        if field["paths"] and all(p in overpacks for p in field["paths"]):
            generated = {overpacks[p] for p in field["paths"]}
            if len(generated) != 1 or "hostAssembly" in field:
                raise ValueError("overpack surfaces have incompatible shared lexical ownership")
            values[key] = generated.pop()
            evidence[key] = dict(kind="source_proven_overpack_levels", paths=field["paths"])
            continue
        if field.get("hostRangeIdentity"):
            if "hostAssembly" not in field or not field["paths"]:
                raise ValueError("host shipping-mark identity lacks its proven range frame")
            values[key] = "MARK-" + _reference(sample_id, key, seed)
            evidence[key] = dict(
                kind="host_shipping_mark_and_range",
                paths=field["paths"],
                frame=field["hostAssembly"],
            )
            continue
        if field["paths"] and all(p in references for p in field["paths"]):
            generated = {references[p] for p in field["paths"]}
            if len(generated) != 1 or "hostAssembly" in field:
                raise ValueError("mark references have incompatible shared lexical ownership")
            values[key] = generated.pop()
            evidence[key] = dict(kind="source_shaped_mark_reference", paths=field["paths"])
            continue
        if field["paths"] and all(p in ranges for p in field["paths"]):
            values_for_field = {ranges[p] for p in field["paths"]}
            if len(values_for_field) != 1 or "hostAssembly" in field:
                raise ValueError("package ranges have incompatible shared lexical ownership")
            values[key] = values_for_field.pop()
            evidence[key] = dict(kind="source_proven_carton_lot_partition", paths=field["paths"])
            continue
        if (
            field["paths"]
            and all(
                p.startswith("documentPatch.forwardingAndExportReferences[") for p in field["paths"]
            )
            and re.fullmatch(r"[A-Za-z0-9/.-]+", field["source"])
            and re.search(r"\d", field["source"])
            and "hostAssembly" not in field
            and not any(c.get("fixedLiteralTokens") for c in field["constraints"])
            and any(
                b.logical_key in identifier_dependencies
                and set(b.target_paths).intersection(field["paths"])
                for b in source.template.bindings
            )
        ):
            values[key] = generate_from_surface_pattern(
                pattern=surface_pattern(field["source"]),
                stream=DeterministicStream(seed, "dependent-reference-v1", sample_id).derive(key),
                additional_excluded=field["source"],
            )
            evidence[key] = dict(kind="source_shaped_dependent_reference", source=field["source"])
            continue
        phone_owners = [
            match[1]
            for path in field["paths"]
            if (
                match := re.fullmatch(
                    r"(documentPatch\.parties\.[^.]+)\.contactDetails\.phoneNumbers\[\d+\]", path
                )
            )
        ]
        if projection is not None and phone_owners:
            if len(phone_owners) != len(field["paths"]):
                raise ValueError("phone field shares ownership with a non-phone scalar")
            countries = {projection.party_geography[p].country_code for p in phone_owners}
            if len(countries) != 1:
                raise ValueError("shared phone field has inconsistent sampled countries")
            country = countries.pop()
            value = contact_values.phone(
                DeterministicStream(seed, "structured-party-phone-v1", sample_id).derive(key),
                country_code=country,
            )
            values[key] = value
            evidence[key] = dict(kind="country_valid_phone", country=country)
            continue
        if scenario is None or not (definitions := _definitions(field, scenario)):
            continue
        reference = _reference(sample_id, key, seed)
        if "hostAssembly" in field:
            raise ValueError("fixed commodity literal requires a goods-specific lexical contract")
        if "cargoFragment" in field:
            if contract is None or key not in contract.fragments:
                raise ValueError("split cargo description lacks a reviewed lexical recipe: " + key)
            recipe = contract.fragments[key]
            if recipe.source != field["source"]:
                raise ValueError("cargo fragment recipe source changed")
            _validate_literal_fragment(field, recipe)
            if recipe.role == "literal":
                value = recipe.source
            elif recipe.role == "reference":
                value = reference
            elif recipe.role == "detail":
                value = "Commercial product reference " + reference
            else:
                if not recipe.identity_indices or any(
                    i < 0 or i >= len(definitions) for i in recipe.identity_indices
                ):
                    raise ValueError("cargo item recipe does not address sampled identities")
                value = "; ".join(definitions[i] for i in recipe.identity_indices)
                value += "; product reference " + reference
        else:
            value = "; ".join(definitions) + "; product reference " + reference
            packages = list(
                dict.fromkeys(
                    phrase
                    for path in field["paths"]
                    if (phrase := _package_mentions(source, scenario.target, path))
                )
            )
            if packages:
                value += "; " + "; ".join(packages)
        minimum = max((c.get("minimumWords", 1) for c in field["constraints"]), default=1)
        if len(re.findall(r"[A-Za-z0-9]+", value)) < minimum:
            raise ValueError("controlled cargo wording cannot fill the certified text segments")
        if boundary := field.get("requiredBoundaryPunctuation"):
            value = boundary["prefix"] + value + boundary["suffix"]
        values[key] = value
        evidence[key] = dict(kind="registry_owned_cargo", definitions=definitions)
    return HostLexicalPlan(values, evidence)
