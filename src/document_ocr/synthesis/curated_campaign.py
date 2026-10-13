"""Full-scenario current-schema synthesis: plan, word, render and validate."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import random
import re
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from document_ocr.hashing import sha256_file
from document_ocr.labeling_agents.direct_grounding import order_consignment_candidates
from document_ocr.labeling_agents.target_normalization import normalize_target_casing
from document_ocr.synthesis.curated import (
    CuratedSynthesisConfig,
    Runner,
    SourceContract,
    assign,
    digest,
    flat,
    interpolate,
    locate,
    save,
)
from document_ocr.synthesis.curated_auxiliary import (
    augment_auxiliary_blueprint,
    auxiliary_surfaces,
    load_auxiliary_contract,
)
from document_ocr.synthesis.curated_casing import CasingPolicy, TargetCasing
from document_ocr.synthesis.curated_commodity_brief import commodity_brief
from document_ocr.synthesis.curated_contacts import (
    CONTACT_PROMPT,
    apply_contacts,
    contact_output_type,
    contact_parties,
    contact_prompt,
    unpack_contacts,
)
from document_ocr.synthesis.curated_contacts import (
    request_hash as contact_request_hash,
)
from document_ocr.synthesis.curated_descriptions import validate_description_regions
from document_ocr.synthesis.curated_equipment import PrintedEquipmentMix
from document_ocr.synthesis.curated_equipment_aliases import (
    EquipmentAliasRegistry,
    apply_equipment_aliases,
)
from document_ocr.synthesis.curated_equipment_inventory import (
    compile_equipment_inventory,
    project_equipment_inventory,
    rebase_equipment_inventory_headers,
)
from document_ocr.synthesis.curated_identifiers import declared_identifier_values
from document_ocr.synthesis.curated_ownership import (
    apply_dependent_text,
    lexical_expression,
    load_owned_blueprint,
    ownership_surfaces,
)
from document_ocr.synthesis.curated_packaging import (
    compile_package_count_context,
    contained_packing_context,
    project_package_aliases_with_context,
    validate_contained_quantity_sampling,
)
from document_ocr.synthesis.curated_physical import (
    PhysicalRenderPlan,
    PhysicalSupport,
    prepare_physical_render,
    printed_tare_equipment_domain,
    project_nested_equipment_handling,
    validate_source_physical_requirements,
)
from document_ocr.synthesis.curated_publication import publish_campaign, review_contract_hash
from document_ocr.synthesis.curated_routes import validate_route_values
from document_ocr.synthesis.curated_sampling import (
    source_variant_counts,
    validate_instruction_inheritance,
)
from document_ocr.synthesis.curated_scenarios import (
    FilePin,
    GoodsIdentity,
    ScenarioCatalog,
    ScenarioLocation,
    ScenarioSamplingConfig,
    ShipmentScenario,
    SourceCapabilities,
    _pin,
)
from document_ocr.synthesis.curated_templates import (
    SamplingBlueprint,
    render_sampling_blueprint,
    validate_postal_country_owner,
    wrap_owned_text,
)
from document_ocr.synthesis.curated_wording import (
    IDENTITY_KIND_REQUIREMENTS,
    WORDING_PROMPT,
    RenderedReview,
    RenderedReviewRequest,
    WordingBatch,
    WordingField,
    WordingRequest,
    mass_in_kilograms,
    native_wording_prompt,
    rendered_review_prompt,
    repeated_goods_descriptions,
    review_output_type,
    unpack_review,
    unpack_wording,
    validate_postal_geography,
    validate_wording,
    wording_output_type,
    wording_prompt,
)
from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_container_number,
    generate_from_surface_pattern,
    surface_pattern,
)
from document_ocr.synthesis.package_aliases import PACKAGE_ALIASES, validate_package_alias_registry
from document_ocr.synthesis.package_registry import load_package_registry
from document_ocr.synthesis.template_compiler import contact_values
from document_ocr.synthesis.template_compiler.descendant import _number_to_words
from document_ocr.synthesis.vessel_lexical import vessel_name_fit_exclusion_reason


class AliasSamplingPolicy(BaseModel):
    """Independent reviewed wordings; absent policy preserves source rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    equipment_registry: FilePin | None = None
    package_registry: FilePin
    package_mode: Literal["source", "registry"] = "source"

    @model_validator(mode="after")
    def counted_package_registry(self):
        if self.package_registry.records is None:
            raise ValueError("alias package registry requires an explicit entry count")
        return self


def party_path(path: str) -> str | None:
    match = re.match(r"documentPatch\.parties\.[^.]+", path)
    return match[0] if match else None


def related_actor_contexts(blueprint: SamplingBlueprint) -> dict[str, dict]:
    """Certify separately generated actors without inventing extraction roles."""
    declarations = blueprint.ownership_data.get("related_actors", {})
    if not isinstance(declarations, dict):
        raise ValueError("related actors require a variable-to-context mapping")
    if not declarations:
        return declarations
    variables = {v.key: v for v in blueprint.contract.variables}
    leaves = flat(blueprint.target)
    for key, declaration in declarations.items():
        if (
            key not in variables
            or variables[key].kind not in {"name", "postal"}
            or not isinstance(declaration, dict)
            or not {"context_party", "relationship"} <= declaration.keys()
            or set(declaration) - {"context_party", "relationship", "location"}
            or declaration["relationship"]
            not in {"on_behalf", "care_of", "account_of", "agent_for", "invoice_party", "exporter"}
        ):
            raise ValueError("related actor requires a known lexical key and explicit relationship")
        role = declaration["context_party"]
        if "location" in declaration:
            location = ScenarioLocation.model_validate(declaration["location"])
            if not all(value.strip() for value in location.model_dump().values()) or not re.search(
                r"(?<!\w)" + re.escape(location.country) + r"(?!\w)", blueprint.source, re.I
            ):
                raise ValueError("related actor location requires printed source country evidence")
        if (
            not isinstance(role, str)
            or party_path(role + ".name") != role
            or not any(path.startswith(role + ".") for path in leaves)
        ):
            raise ValueError("related actor context must name a present concrete party")
        if any(
            binding.path == role + ".name" and "{" + key + "}" in binding.expression
            for binding in blueprint.contract.targets
        ):
            raise ValueError("related actor cannot be concatenated into its context party name")
        variable = variables[key]
        if variable.kind == "name":
            pattern = (
                r"(?<!\w)"
                + r"\s+".join(re.escape(token) for token in variable.value.split())
                + r"(?!\w)"
            )
            owned = [locate(blueprint.source, occurrence) for occurrence in variable.occurrences]
            for match in re.finditer(pattern, blueprint.source, re.I):
                start = len(blueprint.source[: match.start()].encode())
                end = len(blueprint.source[: match.end()].encode())
                if not any(a <= start and end <= b for a, b in owned):
                    raise ValueError(
                        f"related actor has an unowned repeated source identity: {key}"
                    )
    return declarations


def validate_lexical_owners(blueprint: SamplingBlueprint) -> None:
    """Every lexical target needs a mutable owner of its semantic kind."""
    kinds = {variable.key: variable.kind for variable in blueprint.contract.variables}
    generated = {
        binding.path: {
            kinds[key]
            for key in re.findall(r"\{([a-z][a-z0-9_]*)\}", binding.expression)
            if key in kinds
        }
        for binding in blueprint.contract.targets
    }
    missing = []
    for path in flat(blueprint.target):
        expected = None
        if path.endswith(".addressLine"):
            expected = "postal"
        elif path.endswith(".description"):
            expected = "product"
        elif ".parties." in path and path.endswith(".name"):
            expected = "name"
        if expected is not None and expected not in generated.get(path, set()):
            missing.append(path)
    if missing:
        raise ValueError(f"lexical targets lack mutable source wording owners: {sorted(missing)}")
    related_actor_contexts(blueprint)


def certified_identity_kinds(blueprint: SamplingBlueprint) -> dict[str, str]:
    """Read explicit source declarations, never infer identity shape from wording."""
    declarations = blueprint.ownership_data.get("identity_kinds", {})
    name_keys = {v.key for v in blueprint.contract.variables if v.kind == "name"}
    if not isinstance(declarations, dict) or (
        set(declarations) - name_keys
        or any(
            not isinstance(kind, str) or kind not in IDENTITY_KIND_REQUIREMENTS
            for kind in declarations.values()
        )
    ):
        raise ValueError("identity kinds require known name keys and explicit supported meanings")
    return declarations


def identity_review_context(blueprint: SamplingBlueprint, values: dict[str, str]) -> str:
    """Expose certified component meaning and actual generated text to review."""
    declarations = certified_identity_kinds(blueprint)
    related = related_actor_contexts(blueprint)
    if not declarations and not related:
        return ""
    components = []
    for key, kind in declarations.items():
        paths = [
            binding.path
            for binding in blueprint.contract.targets
            if "{" + key + "}" in binding.expression
        ]
        if not paths and key in related:
            actor = related[key]
            paths = [f"related {actor['relationship']} actor for {actor['context_party']}"]
        if not paths or key not in values:
            raise ValueError("certified identity review requires owned generated components")
        components.append(
            f"{', '.join(paths)} ({kind}): {IDENTITY_KIND_REQUIREMENTS[kind]}\n"
            f"Generated component:\n{values[key]}"
        )
    context = (
        "\nSOURCE-CERTIFIED IDENTITY COMPONENTS\n" + "\n\n".join(components) if components else ""
    )
    variables = {v.key: v for v in blueprint.contract.variables}
    relationships = []
    for key, actor in related.items():
        if key not in values:
            raise ValueError("related actor review requires its generated component")
        paths = [
            binding.path
            for binding in blueprint.contract.targets
            if "{" + key + "}" in binding.expression
        ]
        ownership = ", ".join(paths) if paths else "source-only context; no extraction-party field"
        relationships.append(
            f"{actor['relationship']} with {actor['context_party']}: {ownership}\n"
            f"Reviewed source meaning: {variables[key].meaning}\n"
            f"Generated {variables[key].kind}:\n{values[key]}"
        )
    if relationships:
        context += (
            "\nSOURCE-CERTIFIED RELATED ACTOR OWNERSHIP\n"
            "The primary target represents A only. Independently named B is relationship "
            "context, not an omitted second party role. Source-only postal context must "
            "not be transferred into A's address merely because it follows B in the "
            "same printed box. Explicit care-of recipient routing stays in the address.\n"
            + "\n\n".join(relationships)
        )
    return context


def wording_request_hash(request: WordingRequest) -> str:
    """A changed instruction or structured contract invalidates saved wording."""
    return digest(
        {
            "system": WORDING_PROMPT,
            "request": native_wording_prompt([request]),
            "schema": wording_output_type([request]).model_json_schema(),
            "gross_weight_kg": (
                str(request.gross_weight_kg) if request.gross_weight_kg is not None else None
            ),
        }
    )


def apply_replacements(target: dict, replacements: dict) -> dict:
    result = deepcopy(target)
    for path, value in replacements.items():
        assign(result, path, deepcopy(value))
    return result


_DOCUMENT_IDENTIFIER_SUFFIXES = (
    ".billOfLadingNumber",
    ".masterBillOfLadingNumber",
    ".originalBillOfLadingNumber",
    ".voyageNumber",
)


def public_identity_updates(
    target: dict, scenario: ShipmentScenario, stream: DeterministicStream
) -> dict:
    """Generate identifiers and contacts once, preserving repeated identities."""
    result = deepcopy(target)
    leaves = flat(result)
    identities = {}
    date_paths = [p for p in leaves if p.endswith((".issueDate", ".shippedOnBoardDate"))]
    shift = timedelta(days=30 + stream.derive("date-shift").randbelow(730))
    for path, old in leaves.items():
        value = old
        if path in date_paths:
            value = (date.fromisoformat(old) + shift).isoformat()
        elif path.endswith(".equipmentIdentifier"):
            identity = ("container", old)
            if identity not in identities:
                identities[identity] = generate_container_number(
                    owner_and_category=old[:4], stream=stream.derive(old), excluded={old}
                )
            value = identities[identity]
        elif path.endswith(_DOCUMENT_IDENTIFIER_SUFFIXES) or ".sealNumbers[" in path:
            identity = ("identifier", old)
            if identity not in identities:
                identities[identity] = generate_from_surface_pattern(
                    pattern=surface_pattern(old),
                    stream=stream.derive(old),
                    additional_excluded=old,
                )
            value = identities[identity]
        elif ".contactDetails." in path:
            role = party_path(path)
            if role not in scenario.party_localities:
                raise ValueError(f"contact has no sampled party locality: {path}")
            geo = scenario.party_localities[role]
            identity = (path.rsplit(".", 1)[-1].split("[")[0], old, geo.country_code)
            if identity not in identities:
                channel = stream.derive(repr(identity))
                if ".phoneNumbers[" in path or path.endswith(".faxNumber"):
                    identities[identity] = contact_values.phone(
                        channel, country_code=geo.country_code
                    )
                elif ".emailAddresses[" in path or ".websiteUrls[" in path:
                    # Company-conditioned wording runs after company names exist.
                    continue
                elif path.endswith(".contactName"):
                    # Contact names are language; they join the wording request.
                    continue
                else:
                    raise ValueError(f"unsupported contact field: {path}")
            value = identities[identity]
        if value != old:
            assign(result, path, value)
    return result


def contact_wording_fields(
    blueprint: SamplingBlueprint, scenario: ShipmentScenario
) -> dict[str, tuple[WordingField, tuple[str, ...]]]:
    """Share a person's generated name across the same printed party identity."""
    grouped: dict[str, tuple[WordingField, list[str]]] = {}
    lexical_paths = {
        b.path
        for b in blueprint.contract.targets
        if any(
            v.kind == "name" and "{" + v.key + "}" in b.expression
            for v in blueprint.contract.variables
        )
    }
    for path, old in flat(blueprint.target).items():
        if not path.endswith(".contactName") or path in lexical_paths:
            continue
        geo = scenario.party_localities[party_path(path)]
        key = "contact_" + digest([old, geo.country_code])[:12]
        if key not in grouped:
            count = len(old.split(";"))
            field = WordingField(
                key,
                "contact name",
                old,
                f"Generate {count} new fictional contact name(s) appropriate to {geo.country}. "
                "Separate multiple people with '; '.",
            )
            grouped[key] = (field, [])
        grouped[key][1].append(path)
    return {key: (field, tuple(paths)) for key, (field, paths) in grouped.items()}


def host_product_wording(
    blueprint: SamplingBlueprint, scenario: ShipmentScenario
) -> dict[str, str]:
    """Regulated chemicals and whole-unit loads require exact product identities.

    Chemical variants can change the required UN tuple (e.g. solid vs solution).
    Use the selected registry phrase for these tightly coupled profiles.
    The remaining names and addresses still use the lexical generation path.
    """
    chemical = scenario.provenance.get("capabilities", {}).get("family") == "dg_chemical"
    if not chemical and scenario.provenance.get("physicalProfile") not in {
        "source_whole_units",
        "reviewed_whole_units",
    }:
        return {}
    keys = [v.key for v in blueprint.contract.variables if v.kind == "product"]
    if len(keys) != 1 or len(scenario.goods_identities) != 1:
        raise ValueError("exact product wording requires one certified commodity region")
    return {keys[0]: scenario.goods_identities[0].phrase.upper()}


def wording_request(
    blueprint: SamplingBlueprint,
    scenario: ShipmentScenario,
    sample_id: str,
    *,
    goods_only: bool = False,
    physical_plan: PhysicalRenderPlan | None = None,
) -> WordingRequest:
    bindings = blueprint.contract.targets
    identity_kinds = certified_identity_kinds(blueprint)
    related_actors = related_actor_contexts(blueprint)
    identities = list(scenario.goods_identities)
    product_keys = {v.key for v in blueprint.contract.variables if v.kind == "product"}
    product_groups = [
        binding
        for binding in bindings
        if binding.path.endswith(".description")
        and product_keys.intersection(re.findall(r"\{([a-z][a-z0-9_]*)\}", binding.expression))
    ]
    covered = {
        key
        for binding in product_groups
        for key in re.findall(r"\{([a-z][a-z0-9_]*)\}", binding.expression)
        if key in product_keys
    }
    if covered != product_keys:
        raise ValueError("product regions require explicit description target ownership")
    host_products = host_product_wording(blueprint, scenario)
    fields = []
    for variable in blueprint.contract.variables:
        if variable.kind not in {"name", "postal", "product"}:
            continue
        if goods_only and variable.kind != "product":
            continue
        if variable.key in host_products:
            continue
        owners = [b for b in bindings if "{" + variable.key + "}" in b.expression]
        roles = sorted({p for binding in owners if (p := party_path(binding.path)) is not None})
        if variable.key in related_actors:
            actor = related_actors[variable.key]
            geo = (
                ScenarioLocation.model_validate(actor["location"])
                if "location" in actor
                else scenario.party_localities[actor["context_party"]]
            )
            role = f"{actor['context_party']} related {actor['relationship']} {variable.kind}"
            requirement = (
                f"Separate fictional actor in the printed {actor['relationship']} relationship "
                f"with {actor['context_party']}. This is not an additional extraction-party "
                "function and must not be merged into that party's legal name. "
            )
            if variable.kind == "name":
                requirement += (
                    "Generate one distinct identity only, matching the example's legal/trading "
                    "or personal identity structure; exclude postal, contact and relationship "
                    "framing. Independently owned repeated occurrences reuse this identity."
                )
            else:
                requirement += (
                    f"Generate postal components only in {geo.name}, {geo.country} "
                    f"({geo.country_code}), matching the example's address hierarchy. "
                    "Exclude names, contacts, tax data and relationship framing."
                )
        elif roles:
            identity_roles = roles
            direct_roles = set()
            if variable.kind == "name":
                direct_roles = {
                    role
                    for binding in owners
                    if binding.path.endswith(".name")
                    and binding.expression.strip() == "{" + variable.key + "}"
                    and (role := party_path(binding.path)) is not None
                }
                # A repeated legal identity uses its directly owned party's
                # locality, even when another role references the same identity.
                if len(direct_roles) == 1:
                    identity_roles = sorted(direct_roles)
            locations = {scenario.party_localities[p].model_dump_json() for p in identity_roles}
            shared_identity = variable.kind == "name" and len(direct_roles) > 1
            if len(locations) != 1 and not shared_identity:
                raise ValueError(
                    f"shared party region has conflicting sampled geography: {variable.key}"
                )
            geo = scenario.party_localities[identity_roles[0]]
            requirement = f"New party in {geo.name}, {geo.country} ({geo.country_code})."
            if shared_identity and len(locations) > 1:
                offices = sorted(
                    {
                        f"{scenario.party_localities[p].name}, "
                        f"{scenario.party_localities[p].country}"
                        for p in identity_roles
                    }
                )
                requirement = (
                    "One new fictional identity shared by these party roles, "
                    "with offices in " + "; ".join(offices) + "."
                )
            if variable.kind == "postal":
                requirement += (
                    " Postal components only: exclude party identities, tax/registration "
                    "data, contacts and captions. Genuine named sites remain postal components. "
                    "Generate this role's components jointly, retaining the example's hierarchy "
                    "and approximate line span. POSTAL ASSEMBLY defines required geography. "
                    "Use the supplied locality spelling once. A literal locality/country "
                    "in the assembly is supplied by the host. Choose street/site names "
                    "without reusing that locality/country inside them."
                )
            else:
                requirement += (
                    " New identity only. Exclude postal, tax/registration and "
                    "contact/manager/agent statements."
                )
                if variable.key not in identity_kinds:
                    requirement += (
                        " Match the example's legal/trading or personal identity structure. "
                        "An uncaptioned proprietor stays uncaptioned."
                    )
                if identity_roles != roles:
                    requirement += (
                        f" Its complete identity belongs to {identity_roles[0]}; "
                        "reuse it unchanged in its other explicitly owned occurrences."
                    )
            role = ", ".join(roles) + " " + variable.kind
        elif variable.kind == "product":
            requirement = (
                "Complete commercial product wording, matching the source's level of detail "
                "and list structure. Express the required commodity facts and physical form. "
                "Classification wording supplies context, not extra required attributes. "
                "Product variants and intrinsic specifications may enrich an example that "
                "has that detail. Incidental per-unit capacity/fill can be product detail. "
                "Shipment counts, weights, transport settings and codes are separate host facts."
            )
            role = "goods description"
        else:
            raise ValueError(f"lexical region has no current owner: {variable.key}")
        fields.append(
            WordingField(
                variable.key,
                role,
                variable.occurrences[0].text,
                requirement,
                identity_kinds.get(variable.key),
            )
        )
    if not goods_only:
        fields.extend(field for field, _ in contact_wording_fields(blueprint, scenario).values())
    loading = scenario.route_locations.get("portOfLoading", scenario.origin)
    context = (
        f"Loading: {loading.name}, {loading.country}. "
        f"Discharge: {scenario.destination.name}, {scenario.destination.country}."
    )
    if via := scenario.route_locations.get("transshipmentPort"):
        context += f"\nPrinted transshipment location: {via.name}, {via.country}."
    generated_products = product_keys - host_products.keys()
    profile = scenario.provenance.get("registryLoadProfile")
    if generated_products:
        context += "\n" + commodity_brief(
            identities, physical_form=profile["form"] if profile else None
        )
        context += (
            "\nGenerate requested regions as one coherent commercial description: a "
            "concise commodity identity, a specification passage, or a product list, matching "
            "the source's descriptive depth. Represent every sampled identity at that level "
            "of detail. Classification references are contextual alternatives; required "
            "product facts and physical form govern the description. Exact HS-to-product "
            "classification is not the wording objective. "
            "Repeated uses share one value; output final commercial wording only. "
            "Schema instructions are context, never product text."
        )
        for binding in product_groups:
            context += f"\nDESCRIPTION ASSEMBLY {binding.path}: {binding.expression}"
    cargo = physical_plan.target["documentPatch"] if physical_plan else scenario.cargo
    physical = {
        "goods": [
            {
                "packageCategories": [
                    package["typeCategory"] for package in item.get("numberAndTypeOfPackages", [])
                ],
            }
            for item in cargo.get("goodsItemDetails", [])
        ],
        "equipmentCategories": [
            dict(zip(("sizeCategory", "typeCategory"), pair, strict=True))
            for pair in sorted(
                {
                    (item.get("sizeCategory"), item.get("typeCategory"))
                    for item in cargo.get("containerInformation", [])
                },
                key=lambda pair: tuple(value or "" for value in pair),
            )
        ],
    }
    if contents := contained_packing_context(blueprint, {"documentPatch": cargo}):
        physical["containedPackageWordings"] = sorted(
            {item["printed_contained_unit"] for item in contents}
        )
    context += (
        "\nTRANSPORT COMPATIBILITY (categories only; accounting is host-rendered)\n"
        + yaml.safe_dump(physical, sort_keys=False, allow_unicode=True)
    )
    if profile:
        context += (
            "\nPRESENTATION CONTEXT\nDeclared transport unit: "
            + profile["package_category"].removeprefix("PACKAGE_").replace("_", " ")
            + ". The host prints packing counts, package fills and shipment weights separately. "
            "Product wording contains commodity identity and specifications, without standalone "
            "shipment totals or a conflicting inner-package presentation."
        )
    if thermal := scenario.provenance.get("thermalCommodityContext"):
        # Product identities already own their cold-chain form. Numeric carriage
        # settings are host facts, not product prose to regenerate.
        context += "\nREQUIRED COLD-CHAIN PRODUCT FORM\n" + yaml.safe_dump(
            {"profile": thermal["profile"], "productForm": thermal["productForm"]},
            sort_keys=False,
            allow_unicode=True,
        )
        if generated_products:
            context += (
                "Describe the sampled commodity in its stated chilled/frozen/fresh form. "
                "The source description illustrates presentation, not the product to preserve. "
                "Where an observed produce extension supplies a description, retain that "
                "commodity's form. Temperature and ventilation are rendered separately."
            )
    for binding in bindings:
        if goods_only:
            break
        if not binding.path.endswith(".addressLine"):
            continue
        role = party_path(binding.path)
        if role is None or role not in scenario.party_localities:
            continue
        geo = scenario.party_localities[role]
        country_rule = (
            f"Country {geo.country} appears once in the complete postal address."
            if role + ".country" in flat(blueprint.target)
            else "This source omits the country from this party's postal address; "
            "keep that omission."
        )
        context += (
            f"\nPOSTAL ASSEMBLY {role}: {lexical_expression(blueprint, binding, scenario)}. "
            f"Locality {geo.name} appears once across these regions. {country_rule} "
            "Braced keys are generated regions; literal text is supplied by the host."
        )
    groups = tuple(
        group
        for b in product_groups
        if (
            group := tuple(
                key
                for key in re.findall(r"\{([a-z][a-z0-9_]*)\}", b.expression)
                if key in product_keys and key not in host_products
            )
        )
    )
    goods = cargo.get("goodsItemDetails", [])
    gross = goods[0].get("grossWeight") if len(goods) == 1 else None
    if physical_plan:
        measure = physical_plan.receipt["measures"].get("grossWeight")
        gross = (
            {"value": measure["total"], "unit": measure["unit"]}
            if measure and measure["method"] != "source_zero_unasserted"
            else None
        )
    return WordingRequest(
        sample_id,
        context,
        tuple(fields),
        groups,
        mass_in_kilograms(gross["value"], gross["unit"]) if gross is not None else None,
    )


def host_variable_values(
    blueprint: SamplingBlueprint, target: dict, stream: DeterministicStream
) -> dict[str, str]:
    """Share host-generated identifiers across public targets and auxiliary print."""
    before, after = flat(blueprint.target), flat(target)
    identities = {
        str(old): str(after[path])
        for path, old in before.items()
        if path in after
        and old != after[path]
        and (
            path.endswith((*_DOCUMENT_IDENTIFIER_SUFFIXES, ".equipmentIdentifier"))
            or ".sealNumbers[" in path
        )
    }
    result = {}
    declared, _ = declared_identifier_values(blueprint, stream)
    for variable in blueprint.contract.variables:
        if variable.kind not in {"identifier", "container"}:
            continue
        if variable.key in declared:
            result[variable.key] = declared[variable.key]
            continue
        old = variable.value
        if old not in identities:
            if variable.kind == "container":
                identities[old] = generate_container_number(
                    owner_and_category=old[:4], stream=stream.derive(old), excluded={old}
                )
            else:
                identities[old] = generate_from_surface_pattern(
                    pattern=surface_pattern(old),
                    stream=stream.derive(old),
                    additional_excluded=old,
                )
        result[variable.key] = identities[old]
    return result


def assemble_lexical_target(
    blueprint: SamplingBlueprint,
    target: dict,
    wording: dict[str, str],
    scenario: ShipmentScenario,
    host_values: dict[str, str] | None = None,
    *,
    target_casing: TargetCasing = "uppercase",
    goods_only: bool = False,
) -> dict:
    result = deepcopy(target)
    values = {v.key: v.value for v in blueprint.contract.variables}
    values.update(host_values or {})
    values.update({k: " ".join(v.split()) for k, v in wording.items()})
    leaves = flat(result)
    for binding in blueprint.contract.targets:
        if not isinstance(leaves[binding.path], str) or not any(
            "{" + key + "}" in binding.expression for key in {*wording, *(host_values or {})}
        ):
            continue
        expression = lexical_expression(blueprint, binding, scenario)
        text = interpolate(expression, values)
        assign(result, binding.path, " ".join(text.split()))
    for key, (_, paths) in (
        {} if goods_only else contact_wording_fields(blueprint, scenario)
    ).items():
        for path in paths:
            assign(result, path, " ".join(wording[key].split()))
    if target_casing == "uppercase":
        return normalize_target_casing(result)[0]
    if target_casing != "preserve":
        raise ValueError(f"unknown target casing: {target_casing}")
    return result


def combine_surfaces(*groups: dict[str, str | list[str]]) -> dict[str, str | list[str]]:
    """Conflicting semantic owners are errors, not precedence-dependent edits."""
    result = {}
    for group in groups:
        for key, value in group.items():
            if key in result and result[key] != value:
                raise ValueError(f"conflicting render ownership: {key}")
            result[key] = value
    return result


def postal_wording_errors(blueprint, scenario, target, values) -> dict[str, list[str]]:
    """Use both publication checks: complete geography and certified country owners."""
    assembled = flat(assemble_lexical_target(blueprint, target, values, scenario))
    errors: dict[str, list[str]] = {}
    for role, location in scenario.party_localities.items():
        address = assembled.get(role + ".addressLine")
        if address is None:
            continue
        try:
            validate_postal_geography(
                address,
                locality=location.name,
                country=location.country,
                country_code=location.country_code,
                country_labelled=role + ".country" in assembled,
            )
        except ValueError as error:
            errors.setdefault(role, []).append(str(error))
    for region in blueprint.regions:
        if region.kind != "postal" or region.curated_key not in values:
            continue
        for path in region.target_paths:
            if not path.endswith(".country") or (role := party_path(path)) is None:
                continue
            location = scenario.party_localities[role]
            try:
                validate_postal_country_owner(
                    values[region.curated_key],
                    str(assembled[path]),
                    location.country_code,
                    path,
                )
            except ValueError as error:
                message = f"Region {region.curated_key}: {error}"
                if message not in errors.setdefault(role, []):
                    errors[role].append(message)
    return errors


def postal_correction_context(blueprint, scenario, target, keys) -> str:
    """Expose exact per-party assemblies and country owners, without unrelated goods context."""
    leaves, lines = flat(target), []
    for binding in blueprint.contract.targets:
        if not binding.path.endswith(".addressLine"):
            continue
        if not keys.intersection(re.findall(r"\{([a-z][a-z0-9_]*)\}", binding.expression)):
            continue
        role = party_path(binding.path)
        location = scenario.party_localities[role]
        country_path = role + ".country"
        owners = sorted(
            {
                region.curated_key
                for region in blueprint.regions
                if region.kind == "postal"
                and region.curated_key
                and country_path in region.target_paths
            }
        )
        lines.extend(
            [
                f"POSTAL ASSEMBLY {role}: {lexical_expression(blueprint, binding, scenario)}",
                f"LOCALITY: {location.name!r}, exactly once in the complete address.",
            ]
        )
        if country_path not in leaves:
            lines.append("COUNTRY: omitted from this party's printed address; keep that omission.")
        elif owners:
            lines.append(
                f"COUNTRY: {location.country!r}, exactly once in the complete address. "
                "Copy the supplied spelling and accents. "
                "Use the name alone, without an added ISO code."
            )
            lines.append(
                "COUNTRY OWNER REGIONS: " + ", ".join(owners) + ". "
                "Keep the country inside its certified owner region, not in another component."
            )
        else:
            lines.append(
                f"COUNTRY: {location.country!r}. The country is already supplied by a "
                "host-owned literal in the assembly; retain that literal spelling/name/code "
                "and keep generated regions free of that country name and code."
            )
        lines.append(
            "Braced keys are generated; literal text is supplied by the host and already counts."
        )
    return "\n".join(lines)


def remaining_text_surfaces(blueprint: SamplingBlueprint, target: dict, supplied: dict) -> dict:
    """Render owned locality aliases and contact surfaces from their public facts.

    A historical location span sometimes spells an alias rather than the current
    normalized name. Its entire certified location region may change; no search
    for those words in unrelated source text is performed.
    """
    before, after = flat(blueprint.target), flat(target)
    grouped = {}
    for region in blueprint.regions:
        if region.curated_key or region.key in supplied or not region.target_paths:
            continue
        paths = region.target_paths
        if any(path not in after for path in paths):
            continue
        if all(before[path] == after[path] for path in paths):
            continue
        if region.kind == "location" or all(path.endswith(".country") for path in paths):
            values = list(
                dict.fromkeys(
                    str(after[path]) for path in sorted(paths, key=lambda p: p.endswith(".country"))
                )
            )
            text = ", ".join(values)
        elif all(".contactDetails." in path for path in paths):
            values = list(dict.fromkeys(str(after[path]) for path in paths))
            text = " / ".join(values)
        elif set(paths) == {
            "documentPatch.transport.vesselName",
            "documentPatch.transport.voyageNumber",
        }:
            text = (
                after["documentPatch.transport.vesselName"]
                + " / "
                + after["documentPatch.transport.voyageNumber"]
            )
        else:
            continue
        count = len(blueprint.historical_bindings[region.key]["occurrences"])
        grouped.setdefault(region.key, [None] * count)[region.occurrence] = wrap_owned_text(
            region.source, text
        )
    for key, values in grouped.items():
        if any(value is None for value in values):
            raise ValueError(f"partly overridden historical text key: {key}")
    return grouped


def _project_handling_labels(blueprint, text, target, proof, *, casing):
    target, receipt = project_nested_equipment_handling(
        blueprint, text, target, proof["edits"], casing=casing
    )
    if receipt is not None:
        proof = deepcopy(proof)
        proof["handlingProjection"] = receipt
        proof["coveredTargetPaths"] = sorted(
            set(proof["coveredTargetPaths"]) | {field["path"] for field in receipt["fields"]}
        )
        proof["changedTargetPaths"] = sorted(
            set(proof["changedTargetPaths"])
            | {field["path"] for field in receipt["fields"] if field["before"] != field["value"]}
        )
    return target, proof


def _handling_replay_target(blueprint, candidate, *, casing):
    """Restore derived parents only after independently verifying their receipt.

    The ordinary renderer requires unowned static parents to keep their source
    value. Their final value is derived from the owned child edits afterward.
    Recompute that derivation before resetting replay inputs, so a fabricated
    receipt cannot conceal unrelated handling changes. Complete render/proof
    equality is still checked by the caller after this temporary restoration.
    """
    if "handlingProjection" not in candidate["proof"]:
        return candidate["target"]
    recorded = candidate["proof"]["handlingProjection"]
    projected, receipt = project_nested_equipment_handling(
        blueprint,
        candidate["joinedRawText"],
        candidate["target"],
        candidate["proof"]["edits"],
        casing=casing,
    )
    if (
        receipt is None
        or recorded != receipt
        or projected != candidate["target"]
        or receipt["renderedSha256"] != candidate["joinedRawTextSha256"]
    ):
        raise ValueError(
            "handling projection receipt differs from independent source/edit authority"
        )
    return apply_replacements(
        candidate["target"], {field["path"]: field["before"] for field in receipt["fields"]}
    )


def validate_sample(
    blueprint: SamplingBlueprint,
    scenario: ShipmentScenario,
    candidate: dict,
    task,
    *,
    casing_policy: CasingPolicy | None = None,
) -> dict:
    """Replay supplied render inputs and check schema/accounting invariants.

    Campaign.replay_candidate separately enforces complete pinned source,
    scenario and wording authority; this component alone is not that gate.
    """
    casing_policy = casing_policy or CasingPolicy()
    country_codes = {
        role + ".country": geo.country_code
        for role, geo in scenario.party_localities.items()
        if role + ".country" in flat(candidate["target"])
    }
    if candidate["countryCodes"] != country_codes:
        raise ValueError("country-code authority differs from the sampled registry geography")
    text, target, proof = render_sampling_blueprint(
        blueprint,
        _handling_replay_target(blueprint, candidate, casing=casing_policy.target),
        candidate["renderValues"],
        surface_values=candidate["surfaceValues"],
        certified_country_codes=country_codes,
        render_casing=casing_policy.select(candidate["seed"], candidate["documentId"]),
        target_casing=casing_policy.target,
    )
    target, proof = _project_handling_labels(
        blueprint, text, target, proof, casing=casing_policy.target
    )
    if (candidate["joinedRawText"], candidate["target"], candidate["proof"]) != (
        text,
        target,
        proof,
    ):
        raise ValueError("candidate differs from independent source/edit replay")
    if candidate["joinedRawTextSha256"] != digest(text.encode()):
        raise ValueError("candidate OCR hash differs")
    if task.canonicalize(target) != target:
        raise ValueError("candidate is not canonical under the current reduced training contract")
    if casing_policy.target == "uppercase" and normalize_target_casing(target)[1]:
        raise ValueError("candidate target differs from configured uppercase policy")
    patch = target["documentPatch"]
    source = blueprint.target["documentPatch"]
    validate_instruction_inheritance(blueprint.target, target)
    if bool(order_consignment_candidates(blueprint.source)) != bool(
        order_consignment_candidates(text)
    ):
        raise ValueError("rendering changed the source order-consignment instruction")
    validate_route_values(source, patch, scenario.replacements)
    if source.get("route") and patch.get("route") == source["route"]:
        raise ValueError("source-copy route is not a sampled scenario")
    goods = patch["goodsItemDetails"]
    if len(goods) != 1:
        raise ValueError("pilot source policy requires one accounting goods group")
    expected_hs = [identity.hs6 for identity in scenario.goods_identities]
    if "hsCodes" in goods[0] and goods[0]["hsCodes"] != expected_hs:
        raise ValueError("rendered goods classifications differ from sampled identities")
    commodity_wording_changed = (
        goods[0]["description"] != source["goodsItemDetails"][0]["description"]
    )
    if (
        not commodity_wording_changed
        and source["goodsItemDetails"][0].get("hsCodes")
        and not scenario.provenance["sameHS6AsSource"]
    ):
        raise ValueError("unchanged commodity wording conflicts with changed sampled identity")
    containers = patch.get("containerInformation", [])
    identifiers = {c["equipmentIdentifier"] for c in containers}
    allocations = goods[0].get("splitGoodsPlacement", [])
    if any(row["equipmentIdentifier"] not in identifiers for row in allocations):
        raise ValueError("placement points to a nonexistent sampled container")
    old_allocations = source["goodsItemDetails"][0].get("splitGoodsPlacement", [])
    if ["packageQuantity" in row for row in allocations] != [
        "packageQuantity" in row for row in old_allocations
    ]:
        raise ValueError("sampled placement quantity presence differs from printed source topology")
    if allocations and all("packageQuantity" in row for row in allocations):
        total = sum(p["packageQuantity"] for p in goods[0]["numberAndTypeOfPackages"])
        if sum(row["packageQuantity"] for row in allocations) != total:
            raise ValueError("sampled allocation quantities do not sum to their package total")
    repeated = {}
    for role, location in scenario.party_localities.items():
        old = {
            p.removeprefix(role + "."): v
            for p, v in flat(blueprint.target).items()
            if p.startswith(role + ".")
        }
        new = {
            p.removeprefix(role + "."): v
            for p, v in flat(target).items()
            if p.startswith(role + ".")
        }
        signature = (old.get("name"), old.get("addressLine"))
        if all(signature):
            current = (new.get("name"), new.get("addressLine"))
            if signature in repeated and repeated[signature] != current:
                raise ValueError(f"repeated source party diverged: {role}")
            repeated[signature] = current
        if "country" in new and new["country"].upper() != location.country.upper():
            raise ValueError(f"party country differs from sampled geography: {role}")
        if "addressLine" in new and re.search(r"(?<!\d)0{5,6}(?!\d)", new["addressLine"]):
            raise ValueError(f"generated placeholder postcode: {role}")
        if "addressLine" in new:
            try:
                validate_postal_geography(
                    new["addressLine"],
                    locality=location.name,
                    country=location.country,
                    country_code=location.country_code,
                    country_labelled="country" in new,
                )
            except ValueError as error:
                raise ValueError(f"{role}: {error}") from error
    return {
        "exactReplay": True,
        "currentSchema": True,
        "sampledIdentity": True,
        "placementTopology": True,
        "repeatedParties": True,
        "changedFields": len(proof["changedTargetPaths"]),
        "commodityWordingChanged": commodity_wording_changed,
    }


class Campaign:
    def __init__(self, root: Path, config: dict):
        self.root, self.config = root, config
        self.casing = CasingPolicy.model_validate(config.get("casing", {}))
        base = {key: config[key] for key in CuratedSynthesisConfig.model_fields}
        self.calls = Runner(root, base)
        self.output = self.calls.output
        data = root / config["dataset"]
        self.rows = {
            r["documentId"]: r
            for r in map(json.loads, (data / "train.jsonl").read_text().splitlines())
        }
        self.variant_counts = source_variant_counts(config, self.rows)
        self.source_ids = list(self.variant_counts)
        validation = list(map(json.loads, (data / "validation.jsonl").read_text().splitlines()))
        self.catalog = ScenarioCatalog.load(
            root,
            ScenarioSamplingConfig.model_validate_json(json.dumps(config["sampling"])),
            list(self.rows.values()),
            frozenset(r["documentId"] for r in validation),
        )
        self.capabilities = {
            sid: SourceCapabilities.model_validate(v) for sid, v in config["capabilities"].items()
        }
        if (
            set(config["source_ids"]) != set(self.capabilities)
            or not set(self.capabilities) <= self.rows.keys()
        ):
            raise ValueError(
                "source capability coverage differs from selected current training sources"
            )
        validation_ocr = {digest(r["joinedRawText"].encode()) for r in validation}
        if any(
            digest(self.rows[sid]["joinedRawText"].encode()) in validation_ocr
            for sid in self.capabilities
        ):
            raise ValueError("pilot source shares validation OCR")
        self.vessels = tuple(
            sorted(
                {
                    name
                    for row in self.rows.values()
                    if (
                        name := row["target"]["documentPatch"]
                        .get("transport", {})
                        .get("vesselName")
                    )
                    and vessel_name_fit_exclusion_reason(name) is None
                }
            )
        )
        if len(self.vessels) < 2:
            raise ValueError("current training set has insufficient valid vessel-name support")
        self.physical_support = PhysicalSupport.fit(list(self.rows.values()))
        self.auxiliary = load_auxiliary_contract(root / config["auxiliary"])
        self.alias_policy = (
            AliasSamplingPolicy.model_validate(config["aliases"]) if "aliases" in config else None
        )
        self.equipment_alias_registry = (
            EquipmentAliasRegistry.from_data(
                json.loads(_pin(root, self.alias_policy.equipment_registry).read_text())
            )
            if self.alias_policy is not None and self.alias_policy.equipment_registry is not None
            else None
        )
        if self.alias_policy is not None:
            constraints = json.loads((root / config["task_constraints"]).read_text())
            pin = self.alias_policy.package_registry
            if pin.sha256 != constraints["packageRegistrySha256"]:
                raise ValueError("alias package registry differs from task vocabulary authority")
            packages = load_package_registry(
                _pin(root, pin), expected_sha256=pin.sha256, expected_entries=pin.records
            )
            validate_package_alias_registry(
                set(constraints["packageCategoryTokens"]),
                {entry.categoryToken: entry.applicationCode for entry in packages.payload.entries},
            )
            if self.catalog.load_profiles is not None and any(
                p.package_category not in PACKAGE_ALIASES
                for p in self.catalog.load_profiles.profiles
            ):
                raise ValueError("load-profile package category has no reviewed task alias")
        self._blueprints: dict[str, SamplingBlueprint] = {}
        self._blueprint_file_pins: dict[Path, str] = {
            path: digest(path.read_bytes())
            for path in (
                data / "train.jsonl",
                data / "validation.jsonl",
                root / config["auxiliary"],
            )
        }
        policy = self.catalog.config.equipment_inventory
        self.equipment_inventories = {}
        self.equipment_compatibility = {}
        if policy is not None:
            if not set(policy.source_ids) <= set(self.source_ids):
                raise ValueError("equipment inventory scope exceeds selected campaign sources")
            for sid in policy.source_ids:
                cap = self.capabilities[sid]
                if cap.family != "ambient" or cap.physical_profile != "registry_form":
                    raise ValueError("equipment inventory scope requires ambient registry forms")
                blueprint = self.blueprint(sid)
                self.equipment_inventories[sid] = compile_equipment_inventory(blueprint)
                domain = printed_tare_equipment_domain(blueprint, self.physical_support)
                if domain is not None:
                    supported = sorted(
                        domain.intersection(cap.allowed_equipment_pairs)
                        if cap.allowed_equipment_pairs
                        else domain
                    )
                    if not supported:
                        raise ValueError("inventory equipment has no supported printed-tare domain")
                    self.capabilities[sid] = self.capabilities[sid].model_copy(
                        update={"allowed_equipment_pairs": tuple(supported)}
                    )
                    self.equipment_compatibility[sid] = {
                        "basis": "source_owned_tares_and_train_observed_changed_pairs",
                        "eligiblePairs": supported,
                    }
            self.catalog.bind_equipment_inventory(self.equipment_inventories)
            for sid in policy.source_ids:
                self.catalog.validate_equipment_inventory_support(sid, self.capabilities[sid])

    def physical_render(self, blueprint, scenario, target):
        """Compile physical facts, then independently realize their owned aliases."""
        inventory = self.equipment_inventories.get(scenario.source_id)
        plan = prepare_physical_render(
            blueprint,
            scenario,
            target,
            support=self.physical_support,
            equipment_inventory=inventory,
        )
        if "aliases" not in self.config:
            return plan
        stream = DeterministicStream(
            self.config["seed"], "curated-aliases-v1", f"{scenario.source_id}:{scenario.variant}"
        )
        surfaces, receipt = plan.surface_values, deepcopy(plan.receipt)
        if scenario.source_id in self.equipment_compatibility:
            receipt["equipmentCompatibility"] = self.equipment_compatibility[scenario.source_id]
        if self.equipment_alias_registry is not None:
            containers = plan.target["documentPatch"].get("containerInformation", [])
            pairs = {(c.get("sizeCategory"), c.get("typeCategory")) for c in containers}
            if any(None in pair for pair in pairs) or (inventory is None and len(pairs) != 1):
                raise ValueError("registry alias sampling requires one complete equipment pair")
            if inventory is not None:
                aliases = {
                    (size, kind): self.equipment_alias_registry.select(
                        size, kind, stream.derive(f"equipment:{size}|{kind}")
                    )
                    for size, kind in sorted(pairs)
                }
                projected, owners = project_equipment_inventory(
                    inventory,
                    blueprint.target["documentPatch"]["containerInformation"],
                    containers,
                    format_equipment=lambda row, _source: aliases[
                        row["sizeCategory"], row["typeCategory"]
                    ],
                    number_words=_number_to_words,
                )
                surfaces = {**surfaces, **projected}
                receipt["equipmentInventoryAliases"] = {
                    "aliases": [
                        {"sizeCategory": size, "typeCategory": kind, "surface": value}
                        for (size, kind), value in sorted(aliases.items())
                    ],
                    "owners": owners,
                    "registrySha256": self.alias_policy.equipment_registry.sha256,
                }
            else:
                size, kind = pairs.pop()
                alias = self.equipment_alias_registry.select(size, kind, stream.derive("equipment"))
                surfaces, owners = apply_equipment_aliases(blueprint, plan.target, surfaces, alias)
                receipt["equipmentAlias"] = {
                    "surface": alias,
                    "owners": owners,
                    "registrySha256": self.alias_policy.equipment_registry.sha256,
                }
        alias_plan = replace(plan, surface_values=surfaces)
        count_context = compile_package_count_context(blueprint, alias_plan)
        surfaces = project_package_aliases_with_context(
            blueprint,
            alias_plan,
            count_context,
            mode=self.alias_policy.package_mode,
            rng=random.Random(
                f"{self.config['seed']}:{scenario.source_id}:{scenario.variant}:package-aliases"
            ),
        )
        receipt["packageAliasMode"] = self.alias_policy.package_mode
        receipt["packageCountContext"] = [fact.receipt() for fact in count_context]
        return replace(plan, surface_values=surfaces, receipt=receipt)

    def blueprint(self, sid: str) -> SamplingBlueprint:
        if sid in self._blueprints:
            return self._blueprints[sid]
        contract_path = self.root / self.config["source_contracts"] / sid / "contract.json"
        historical_path = self.root / self.config["historical_catalog"] / sid / "template.json"
        ownership_path = self.root / self.config["ownership"]
        for path in (contract_path, historical_path, ownership_path):
            observed = digest(path.read_bytes())
            if path in self._blueprint_file_pins and self._blueprint_file_pins[path] != observed:
                raise ValueError("campaign source files changed after snapshot certification")
            self._blueprint_file_pins[path] = observed
        envelope = json.loads(contract_path.read_text())
        row = self.rows[sid]
        if envelope["sourceSha256"] != digest(row["joinedRawText"].encode()) or envelope[
            "targetSha256"
        ] != digest(row["target"]):
            raise ValueError("source contract identity mismatch")
        historical = json.loads(historical_path.read_text())
        blueprint = load_owned_blueprint(
            row,
            historical,
            SourceContract.model_validate(envelope["contract"]),
            ownership_path,
        )
        blueprint = augment_auxiliary_blueprint(blueprint, self.auxiliary)
        policy = self.catalog.config.equipment_inventory
        if policy is not None and sid in policy.source_ids:
            blueprint = rebase_equipment_inventory_headers(blueprint)
        validate_lexical_owners(blueprint)
        certified_identity_kinds(blueprint)
        for actor in related_actor_contexts(blueprint).values():
            if "location" in actor:
                location = ScenarioLocation.model_validate(actor["location"])
                registry = (
                    self.catalog.localities
                    if location.registry == "geonames"
                    else self.catalog.ports
                )
                if location not in registry.get(location.country_code, ()):
                    raise ValueError("related actor location differs from its pinned registry")
        validate_description_regions(blueprint.description_blocks, blueprint.regions)
        cap = self.capabilities[sid]
        for group in cap.party_locality_groups:
            group.validate_source(row, self.catalog.countries)
            group.validate_blueprint(blueprint)
        if "equipment_mix" in blueprint.ownership_data:
            mix = PrintedEquipmentMix.model_validate(blueprint.ownership_data["equipment_mix"])
            if cap.equipment_mix is not None and cap.equipment_mix != mix:
                raise ValueError("campaign equipment mix conflicts with source ownership")
            # The fixed printed mix is an immutable template requirement, not
            # a campaign distribution knob. Every campaign inherits it.
            cap = cap.model_copy(update={"equipment_mix": mix})
            self.capabilities[sid] = cap
        validate_source_physical_requirements(blueprint, cap)
        if cap.equipment_mix is not None:
            cap.equipment_mix.validate_source(
                blueprint.source, blueprint.target["documentPatch"].get("containerInformation", [])
            )
        validate_contained_quantity_sampling(
            blueprint, multiple=cap.quantity_multiple, fixed=cap.fixed_package_quantity
        )
        self._blueprints[sid] = blueprint
        return blueprint

    def validate_source_readiness(self) -> None:
        """Check every selected source before any concurrent paid work starts."""
        for sid in self.source_ids:
            self.blueprint(sid)
        # Immutable snapshots eliminate repeated YAML compilation. Stage
        # boundaries still hash every dependency, including metadata-preserving
        # edits, so caching cannot silently approve changed source contracts.
        for path, expected in self._blueprint_file_pins.items():
            if sha256_file(path) != expected:
                raise ValueError("campaign source files changed after snapshot certification")
        for sid in self.equipment_inventories:
            self.catalog.validate_equipment_inventory_support(sid, self.capabilities[sid])

    def plan(self, sid: str, variant: int) -> tuple[SamplingBlueprint, ShipmentScenario, str, dict]:
        blueprint = self.blueprint(sid)
        sample_id = "syn_full_v7_" + digest([sid, self.config["seed"], variant])[:24]
        target = None

        def validate_candidate(candidate: ShipmentScenario) -> None:
            nonlocal target
            target = public_identity_updates(
                apply_replacements(blueprint.target, candidate.replacements),
                candidate,
                DeterministicStream(self.config["seed"], "full-current-v7", sample_id),
            )
            target = apply_dependent_text(blueprint, target)
            # The exact source grid belongs to candidate admission, not to a
            # late failure after selecting a shipment (or after a paid call).
            # Contract errors propagate; only typed draw incompatibilities retry.
            self.physical_render(blueprint, candidate, target)

        scenario = self.catalog.sample(
            self.rows[sid],
            seed=self.config["seed"],
            variant=variant,
            capabilities=self.capabilities[sid],
            validate_candidate=validate_candidate,
        )
        if target is None:
            raise ValueError("scenario acceptance did not validate its target")
        if old_vessel := target["documentPatch"].get("transport", {}).get("vesselName"):
            candidates = tuple(v for v in self.vessels if v != old_vessel)
            stream = DeterministicStream(self.config["seed"], "full-current-v7", sample_id)
            target["documentPatch"]["transport"]["vesselName"] = candidates[
                stream.derive("vessel-name").randbelow(len(candidates))
            ]
        return blueprint, scenario, sample_id, target

    def plans(self, sid: str) -> list[tuple]:
        return [self.plan(sid, variant) for variant in range(1, self.variant_counts[sid] + 1)]

    def preflight(self, plans: list[tuple]) -> None:
        """Reject broken deterministic source/physical contracts before paid wording."""
        for blueprint, scenario, sample_id, target in plans:
            validate_lexical_owners(blueprint)
            physical = self.physical_render(blueprint, scenario, target)
            before, after = flat(blueprint.target), flat(physical.target)
            changed = {path for path in after if before.get(path) != after[path]}
            owned = {path for region in blueprint.regions for path in region.target_paths}
            if missing := changed - owned:
                raise ValueError(
                    f"sampled target fields lack source render ownership: {sorted(missing)}"
                )
            surfaces = combine_surfaces(
                physical.surface_values,
                ownership_surfaces(
                    blueprint,
                    physical.target,
                    scenario,
                    lexical_values={v.key: v.value for v in blueprint.contract.variables},
                ),
            )
            auxiliary_surfaces(
                blueprint,
                scenario,
                physical.target,
                DeterministicStream(self.config["seed"], "full-current-v7", sample_id),
                existing=surfaces,
                vessels=self.vessels,
            )

    def wording_request(
        self, blueprint: SamplingBlueprint, scenario: ShipmentScenario, sample_id: str, target: dict
    ) -> WordingRequest:
        """Keep numeric loads private to validation; give wording categorical context."""
        physical = self.physical_render(blueprint, scenario, target)
        return wording_request(blueprint, scenario, sample_id, physical_plan=physical)

    async def generate(self, sid: str) -> dict:
        plans = self.plans(sid)
        self.preflight(plans)
        requests = [
            self.wording_request(bp, scenario, sample_id, target)
            for bp, scenario, sample_id, target in plans
        ]
        wording, pending = {}, []
        repair_error = None
        for request in requests:
            path = self.output / "wording" / f"{request.sample_id}.json"
            if path.exists():
                cached = json.loads(path.read_text())
                if cached["requestSha256"] == wording_request_hash(request):
                    batch = WordingBatch.model_validate(
                        {
                            "shipments": [
                                {
                                    "sample_id": request.sample_id,
                                    "values": [
                                        {"key": k, "text": v} for k, v in cached["values"].items()
                                    ],
                                }
                            ]
                        }
                    )
                    try:
                        accepted = validate_wording(batch, [request])
                    except ValueError:
                        # A new policy can invalidate earlier cached wording.
                        # Send it through the same recorded, bounded repair as
                        # a fresh invalid response; never silently reuse it.
                        pending.append(request)
                    else:
                        wording.update(accepted)
                    continue
            pending.append(request)
        batch_size = self.config["variants_per_source"]
        for start in range(0, len(pending), batch_size):
            batch_requests = pending[start : start + batch_size]
            output = await self.calls.call(
                "exact-wording",
                sid,
                wording_output_type(batch_requests),
                WORDING_PROMPT,
                native_wording_prompt(batch_requests),
            )
            try:
                accepted = unpack_wording(output, batch_requests)
            except ValueError as error:
                # One bounded, recorded correction; never publish invalid wording
                # or silently retry forever. Both billed outputs remain cached.
                repair_error = str(error)
                repaired = await self.calls.call(
                    "wording-validation-repair",
                    sid,
                    wording_output_type(batch_requests),
                    WORDING_PROMPT,
                    native_wording_prompt(batch_requests)
                    + "\nVALIDATION FAILURE\n"
                    + repair_error
                    + "\nPREVIOUS PROPOSAL\n"
                    + yaml.safe_dump(output.model_dump(), sort_keys=False, allow_unicode=True)
                    + "\nCorrect the failing wording. Keep other accepted fields unchanged. "
                    "Return the complete requested fields. Shipment accounting and transport "
                    "facts belong to host-rendered regions, not the product description.",
                )
                accepted = unpack_wording(repaired, batch_requests)
            wording.update(accepted)
        # Reused and new fragments share the same per-shipment correctness
        # checks. Repeated product wording is reported separately as diversity.
        validate_wording(
            WordingBatch.model_validate(
                {
                    "shipments": [
                        {
                            "sample_id": request.sample_id,
                            "values": [
                                {"key": key, "text": text}
                                for key, text in wording[request.sample_id].items()
                            ],
                        }
                        for request in requests
                    ]
                }
            ),
            requests,
        )
        for blueprint, scenario, sample_id, target in plans:
            path = self.output / "wording" / f"{sample_id}.json"
            request_hash = wording_request_hash(
                next(r for r in requests if r.sample_id == sample_id)
            )
            prior = json.loads(path.read_text()) if path.exists() else {}
            retained = prior if prior.get("requestSha256") == request_hash else {}
            save(
                path,
                {
                    **retained,
                    "sourceDocumentId": sid,
                    "documentId": sample_id,
                    "requestSha256": request_hash,
                    "scenario": scenario.model_dump(mode="json"),
                    "targetBeforeWording": target,
                    "values": wording[sample_id],
                    "blueprint": blueprint.inventory(),
                    **({"wordingValidationRepair": repair_error} if repair_error else {}),
                },
            )
        contacts = await self.generate_contacts(sid)
        return {
            "sourceDocumentId": sid,
            "generated": len(plans),
            "contacts": contacts,
            "wordingValidationRepair": repair_error,
            "repeatedGoodsDescriptions": repeated_goods_descriptions(requests, wording),
        }

    def contact_requests(self, sid: str) -> list:
        parties = []
        for bp, scenario, sample_id, target in self.plans(sid):
            receipt = json.loads((self.output / "wording" / f"{sample_id}.json").read_text())
            if receipt["requestSha256"] != wording_request_hash(
                self.wording_request(bp, scenario, sample_id, target)
            ):
                raise ValueError("contact generation requires current company wording")
            wording = combine_surfaces(receipt["values"], host_product_wording(bp, scenario))
            target = assemble_lexical_target(bp, target, wording, scenario)
            parties.extend(contact_parties(bp.target, target, sample_id))
        return parties

    async def generate_contacts(self, sid: str) -> dict:
        parties = self.contact_requests(sid)
        if not parties:
            return {"companies": 0}
        path = self.output / "contacts" / f"{sid}.json"
        cached = json.loads(path.read_text()) if path.exists() else None
        if cached is not None and cached["requestSha256"] == contact_request_hash(parties):
            output = contact_output_type(parties).model_validate(cached["output"])
        else:
            ids = list(dict.fromkeys(p.sample_id for p in parties))
            native = {}
            batch_size = self.config["variants_per_source"]
            for start in range(0, len(ids), batch_size):
                selected_ids = set(ids[start : start + batch_size])
                indices = [i for i, p in enumerate(parties) if p.sample_id in selected_ids]
                batch = [parties[i] for i in indices]
                answer = await self.calls.call(
                    "company-contacts",
                    sid,
                    contact_output_type(batch),
                    CONTACT_PROMPT,
                    contact_prompt(batch),
                )
                unpack_contacts(answer, batch)
                values = answer.model_dump(mode="json")
                native.update({f"p{i}": values[f"p{j}"] for j, i in enumerate(indices)})
            output = contact_output_type(parties).model_validate(native)
        values = unpack_contacts(output, parties)
        batch_receipt = {
            "requestSha256": contact_request_hash(parties),
            "output": output.model_dump(mode="json"),
            "values": values,
        }
        save(self.output / "contacts" / f"{sid}.json", batch_receipt)
        # Calls are batched by source; render/replay consumes only this shipment.
        # This avoids resampling every sibling to validate one document's contacts.
        for sample_id in values:
            indices = [i for i, party in enumerate(parties) if party.sample_id == sample_id]
            selected = [parties[i] for i in indices]
            native = {f"p{j}": batch_receipt["output"][f"p{i}"] for j, i in enumerate(indices)}
            save(
                self.output / "contacts" / "samples" / f"{sample_id}.json",
                {
                    "requestSha256": contact_request_hash(selected),
                    "output": native,
                    "values": values[sample_id],
                    "batchReceiptSha256": digest(batch_receipt),
                },
            )
        return {"companies": len(parties), "fields": sum(len(v) for v in values.values())}

    async def generate_contacts_all(self) -> dict:
        self.validate_source_readiness()
        results = await asyncio.gather(
            *(self.generate_contacts(sid) for sid in self.source_ids),
            return_exceptions=True,
        )
        report = {"sources": [], "errors": [], "costUsd": str(self.calls.spent)}
        for sid, result in zip(self.source_ids, results, strict=True):
            if isinstance(result, BaseException):
                report["errors"].append({"sourceDocumentId": sid, "error": str(result)})
            else:
                report["sources"].append({"sourceDocumentId": sid, **result})
        save(self.output / "contacts-summary.json", report)
        return report

    async def generate_all(self) -> dict:
        started = time.perf_counter()
        self.validate_source_readiness()
        results = await asyncio.gather(
            *(self.generate(sid) for sid in self.source_ids), return_exceptions=True
        )
        report = {
            "sources": [],
            "errors": [],
            "seconds": time.perf_counter() - started,
            "costUsd": str(self.calls.spent),
        }
        for sid, result in zip(self.source_ids, results, strict=True):
            if isinstance(result, BaseException):
                report["errors"].append(
                    {"sourceDocumentId": sid, "error": f"{type(result).__name__}: {result}"}
                )
            else:
                report["sources"].append(result)
        save(self.output / "generation-summary.json", report)
        return report

    async def correct_postal(self, sid: str) -> dict:
        """Reword only failed party postal regions; preserve every other value.

        This is an explicit bounded correction pass, not an automatic retry loop.
        The same locality contract must pass afterward; rejected proposals remain
        inspectable in the call ledger and cannot authorize publication.
        """
        requests, changes = [], {}
        for blueprint, scenario, sample_id, target in self.plans(sid):
            path = self.output / "wording" / f"{sample_id}.json"
            receipt = json.loads(path.read_text())
            original = self.wording_request(blueprint, scenario, sample_id, target)
            if receipt["requestSha256"] != wording_request_hash(original):
                raise ValueError("postal correction requires current generation inputs")
            values = combine_surfaces(receipt["values"], host_product_wording(blueprint, scenario))
            assembled = flat(assemble_lexical_target(blueprint, target, values, scenario))
            failed, keys = [], set()
            for role, errors in postal_wording_errors(blueprint, scenario, target, values).items():
                address_path = role + ".addressLine"
                failed.append(
                    f"{role}: {'; '.join(errors)}. "
                    f"Previous complete address: {assembled[address_path]}"
                )
                for binding in blueprint.contract.targets:
                    if binding.path == address_path:
                        keys.update(
                            v.key
                            for v in blueprint.contract.variables
                            if v.kind == "postal" and "{" + v.key + "}" in binding.expression
                        )
            if not failed:
                continue
            if not keys:
                raise ValueError("postal correction has no owned mutable regions")
            fields = tuple(
                WordingField(
                    f.key,
                    f.role,
                    values[f.key],
                    "Fresh synthetic postal rewrite under the supplied POSTAL ASSEMBLY rules. "
                    "Choose new street/site wording without the sampled locality or "
                    "country inside street, building, district or region names. "
                    "Preserve approximate hierarchy and line span. Place required geography "
                    "in its specified region; one mention suffices when locality and country "
                    "coincide.",
                )
                for f in original.fields
                if f.key in keys
            )
            request = WordingRequest(
                sample_id,
                postal_correction_context(blueprint, scenario, target, keys)
                + "\nPOSTAL CORRECTION\n"
                + "\n".join(failed)
                + "\nRewrite these fictional postal regions jointly. The old values show "
                "approximate hierarchy and line span, not facts to preserve. New street "
                "names and numbers are allowed. Supplied locality/country and host-owned "
                "literal components stay fixed. Return only the requested regions.",
                fields,
            )
            requests.append(request)
            changes[sample_id] = (path, receipt, failed, blueprint, scenario, target)
        if not requests:
            return {"sourceDocumentId": sid, "corrected": 0}
        proposed = {}
        batch_size = self.config["variants_per_source"]
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            output = await self.calls.call(
                "postal-correction",
                sid,
                wording_output_type(batch),
                "Correct fictional postal text. The supplied old values FAILED validation; "
                "they are not formatting examples. Return corrected regions only. Follow each "
                "party's explicit LOCALITY, COUNTRY and COUNTRY OWNER rules. Some parties omit "
                "country; preserve that omission. Host-owned literal components already count. "
                "If a "
                "fictional street/site name repeats the locality, choose a different street/site "
                "name. Preserve approximate address hierarchy and use uppercase text. "
                "Return final postal text only, without editing or validation commentary.",
                "\n\n---\n\n".join(
                    f"SHIPMENT s{index}\n"
                    + request.context
                    + "\n\n"
                    + "\n\n".join(
                        f"REGION {field.key} ({field.role})\n{field.requirement}\n"
                        f"INVALID OLD VALUE:\n{field.example}"
                        for field in request.fields
                    )
                    for index, request in enumerate(batch)
                ),
            )
            proposed.update(unpack_wording(output, batch))
        remaining = []
        for request in requests:
            path, receipt, failed, blueprint, scenario, target = changes[request.sample_id]
            before = deepcopy(receipt["values"])
            receipt["values"].update(proposed[request.sample_id])
            post_errors = [
                f"{role}: {error}"
                for role, errors in postal_wording_errors(
                    blueprint,
                    scenario,
                    target,
                    combine_surfaces(receipt["values"], host_product_wording(blueprint, scenario)),
                ).items()
                for error in errors
            ]
            remaining.extend(f"{request.sample_id}: {error}" for error in post_errors)
            receipt.setdefault("corrections", []).append(
                {
                    "requestSha256": digest(wording_prompt([request])),
                    "issues": failed,
                    "before": {key: before[key] for key in proposed[request.sample_id]},
                    "after": proposed[request.sample_id],
                    "postCheckErrors": post_errors,
                }
            )
            save(path, receipt)
        if remaining:
            raise ValueError("postal correction still requires review: " + "; ".join(remaining))
        return {"sourceDocumentId": sid, "corrected": len(requests)}

    async def correct_postal_all(self) -> dict:
        self.validate_source_readiness()
        results = await asyncio.gather(
            *(self.correct_postal(sid) for sid in self.source_ids),
            return_exceptions=True,
        )
        report = {"sources": [], "errors": [], "costUsd": str(self.calls.spent)}
        for sid, result in zip(self.source_ids, results, strict=True):
            if isinstance(result, BaseException):
                report["errors"].append({"sourceDocumentId": sid, "error": str(result)})
            else:
                report["sources"].append(result)
        save(self.output / "postal-correction-summary.json", report)
        return report

    def render(self, sid: str, variant: int, *, persist: bool = True) -> dict:
        blueprint, scenario, sample_id, target = self.plan(sid, variant)
        receipt = json.loads((self.output / "wording" / f"{sample_id}.json").read_text())
        physical = self.physical_render(blueprint, scenario, target)
        request = wording_request(blueprint, scenario, sample_id, physical_plan=physical)
        if receipt["requestSha256"] != wording_request_hash(request):
            raise ValueError(
                "wording request changed; regenerate this source through the request cache"
            )
        wording = validate_wording(
            WordingBatch.model_validate(
                {
                    "shipments": [
                        {
                            "sample_id": sample_id,
                            "values": [
                                {"key": key, "text": value}
                                for key, value in receipt["values"].items()
                            ],
                        }
                    ]
                }
            ),
            [request],
        )[sample_id]
        wording = combine_surfaces(wording, host_product_wording(blueprint, scenario))
        stream = DeterministicStream(self.config["seed"], "full-current-v7", sample_id)
        host_values = combine_surfaces(
            host_variable_values(blueprint, physical.target, stream),
            physical.variable_values,
        )
        target = assemble_lexical_target(
            blueprint,
            physical.target,
            wording,
            scenario,
            host_values,
            target_casing=self.casing.target,
        )
        parties = contact_parties(blueprint.target, target, sample_id)
        contact_receipt = None
        if parties:
            contact_receipt = json.loads(
                (self.output / "contacts" / "samples" / f"{sample_id}.json").read_text()
            )
            batch_receipt = json.loads((self.output / "contacts" / f"{sid}.json").read_text())
            if (
                contact_receipt["batchReceiptSha256"] != digest(batch_receipt)
                or contact_receipt["values"] != batch_receipt["values"][sample_id]
            ):
                raise ValueError("sample contacts differ from their generated batch receipt")
            if contact_receipt["requestSha256"] != contact_request_hash(parties):
                raise ValueError("company contact receipt is stale")
            checked = unpack_contacts(
                contact_output_type(parties).model_validate(contact_receipt["output"]), parties
            )[sample_id]
            if checked != contact_receipt["values"]:
                raise ValueError("company contacts differ from checked model output")
            target = apply_contacts(target, checked, parties)
        surfaces = combine_surfaces(
            physical.surface_values,
            ownership_surfaces(blueprint, target, scenario, lexical_values=wording),
        )
        surfaces, auxiliary_receipt = auxiliary_surfaces(
            blueprint, scenario, target, stream, existing=surfaces, vessels=self.vessels
        )
        surfaces = combine_surfaces(surfaces, remaining_text_surfaces(blueprint, target, surfaces))
        keys = {v.key for v in blueprint.contract.variables}
        values = combine_surfaces(
            host_values, {key: value for key, value in wording.items() if key in keys}
        )
        country_codes = {
            role + ".country": geo.country_code
            for role, geo in scenario.party_localities.items()
            if role + ".country" in flat(target)
        }
        text, target, proof = render_sampling_blueprint(
            blueprint,
            target,
            values,
            surface_values=surfaces,
            certified_country_codes=country_codes,
            render_casing=self.casing.select(self.config["seed"], sample_id),
            target_casing=self.casing.target,
        )
        target, proof = _project_handling_labels(
            blueprint, text, target, proof, casing=self.casing.target
        )
        candidate = {
            "documentId": sample_id,
            "sourceDocumentId": sid,
            "seed": self.config["seed"],
            "variant": variant,
            "joinedRawText": text,
            "joinedRawTextSha256": digest(text.encode()),
            "target": target,
            "scenario": scenario.model_dump(mode="json"),
            "renderValues": values,
            "surfaceValues": surfaces,
            "countryCodes": country_codes,
            "proof": proof,
            "physical": physical.receipt,
            "auxiliary": auxiliary_receipt,
            "identifierPolicies": declared_identifier_values(blueprint, stream)[1],
            "wordingReceiptSha256": digest(receipt),
            "blueprint": blueprint.inventory(),
        }
        candidate["validation"] = validate_sample(
            blueprint, scenario, candidate, self.calls.task, casing_policy=self.casing
        )
        if contact_receipt is not None:
            candidate["contactReceiptSha256"] = digest(contact_receipt)
        if persist:
            save(self.output / "candidates" / f"{sample_id}.json", candidate)
        return candidate

    def replay_candidate(self, candidate: dict) -> None:
        expected = self.render(candidate["sourceDocumentId"], candidate["variant"], persist=False)
        if expected != candidate:
            raise ValueError("candidate differs from source/scenario/wording authority replay")

    def review_requests(
        self, sid: str, candidates: Sequence[dict]
    ) -> tuple[RenderedReviewRequest, ...]:
        """Compose exact current requests without calls, replay or artifact writes.

        The caller supplies final candidates in variant order after validating
        their source/scenario authority. Execution and offline audit use this
        same assembly, including sampled product form and cold-chain context.
        """
        if sid not in self.variant_counts or len(candidates) != self.variant_counts[sid]:
            raise ValueError("review requests require the exact configured source candidate count")
        for variant, candidate in enumerate(candidates, 1):
            expected_id = "syn_full_v7_" + digest([sid, self.config["seed"], variant])[:24]
            if (
                candidate.get("sourceDocumentId") != sid
                or candidate.get("variant") != variant
                or candidate.get("documentId") != expected_id
            ):
                raise ValueError("review requests require exact source identities in variant order")
        blocks = []
        batch_size = self.config["variants_per_source"]
        for index, candidate in enumerate(candidates):
            facts = candidate["scenario"]
            blocks.append(
                "SHIPMENT "
                + f"s{index % batch_size}"
                + "\nPARTY LOCALITIES\n"
                + "\n".join(
                    f"{role}: {location['name']}, {location['country']} "
                    f"({location['country_code']})"
                    for role, location in facts["party_localities"].items()
                )
                + "\n"
                + commodity_brief(
                    [GoodsIdentity.model_validate(g) for g in facts["goods_identities"]],
                    physical_form=(
                        facts["provenance"]["registryLoadProfile"]["form"]
                        if "registryLoadProfile" in facts["provenance"]
                        else None
                    ),
                )
                + (
                    "\nSAMPLED COLD-CHAIN IDENTITY, PRODUCT FORM AND SETTINGS\n"
                    + yaml.safe_dump(
                        facts["provenance"]["thermalCommodityContext"],
                        sort_keys=False,
                        allow_unicode=True,
                    )
                    if "thermalCommodityContext" in facts["provenance"]
                    else ""
                )
                + "\nTARGET VALUES\n"
                + yaml.safe_dump(candidate["target"], sort_keys=False, allow_unicode=True)
                + identity_review_context(self.blueprint(sid), candidate["renderValues"])
                + "\nHOST-RENDERED CONTAINED PACKING (private facts, not extra target rows)\n"
                + yaml.safe_dump(
                    contained_packing_context(self.blueprint(sid), candidate["target"]),
                    sort_keys=False,
                    allow_unicode=True,
                )
                + "\nCOMPLETE RENDERED TEXT\n"
                + candidate["joinedRawText"]
            )
        requests = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            requests.append(
                RenderedReviewRequest(
                    sample_ids=tuple(c["documentId"] for c in batch),
                    output_type=review_output_type(len(batch)),
                    system=rendered_review_prompt(self.casing.target),
                    prompt="\n\n---\n\n".join(blocks[start : start + batch_size]),
                )
            )
        return tuple(requests)

    async def review(self, sid: str) -> dict:
        candidates = []
        for variant in range(1, self.variant_counts[sid] + 1):
            sample_id = "syn_full_v7_" + digest([sid, self.config["seed"], variant])[:24]
            candidate = json.loads((self.output / "candidates" / f"{sample_id}.json").read_text())
            self.replay_candidate(candidate)
            candidates.append(candidate)
        reviews = []
        for request in self.review_requests(sid, candidates):
            answer = await self.calls.call(
                "rendered-review",
                sid,
                request.output_type,
                request.system,
                request.prompt,
            )
            reviews.append(unpack_review(answer, list(request.sample_ids)))
        output = RenderedReview(
            reviewed_ids=[sid for r in reviews for sid in r.reviewed_ids],
            findings=[finding for r in reviews for finding in r.findings],
        )
        identities = {c["documentId"] for c in candidates}
        if len(output.reviewed_ids) != len(identities) or set(output.reviewed_ids) != identities:
            raise ValueError("rendered review did not cover every candidate exactly once")
        if any(f.sample_id not in identities for f in output.findings):
            raise ValueError("rendered review cites an unrequested candidate")
        receipt = {
            "sourceDocumentId": sid,
            "reviewContractSha256": review_contract_hash(len(candidates), self.casing.target),
            "candidateHashes": {c["documentId"]: digest(c) for c in candidates},
            "review": output.model_dump(mode="json"),
        }
        save(self.output / "reviews" / f"{sid}.json", receipt)
        return {"sourceDocumentId": sid, "findings": len(output.findings)}

    async def review_all(self) -> dict:
        self.validate_source_readiness()
        results = await asyncio.gather(
            *(self.review(sid) for sid in self.source_ids), return_exceptions=True
        )
        report = {"sources": [], "errors": [], "costUsd": str(self.calls.spent)}
        for sid, result in zip(self.source_ids, results, strict=True):
            if isinstance(result, BaseException):
                report["errors"].append(
                    {"sourceDocumentId": sid, "error": f"{type(result).__name__}: {result}"}
                )
            else:
                report["sources"].append(result)
        save(self.output / "review-summary.json", report)
        return report

    def render_all(self) -> dict:
        started = time.perf_counter()
        self.validate_source_readiness()
        report = {"accepted": [], "errors": []}
        for sid in self.source_ids:
            for variant in range(1, self.variant_counts[sid] + 1):
                try:
                    candidate = self.render(sid, variant)
                except Exception as error:
                    report["errors"].append(
                        {
                            "sourceDocumentId": sid,
                            "variant": variant,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                else:
                    report["accepted"].append(candidate["documentId"])
        report["seconds"] = time.perf_counter() - started
        save(self.output / "render-summary.json", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "generate",
            "contacts",
            "correct-postal",
            "render",
            "review",
            "validate",
            "publish",
            "positions",
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--retry-rate-limited",
        action="store_true",
        help="Retry cached HTTP 429 rejections, preserving each attempt receipt.",
    )
    parser.add_argument(
        "--retry-invalid-output",
        action="store_true",
        help="Allow one new call for a cached invalid native output, retaining billed failures.",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    campaign = Campaign(root, yaml.safe_load((root / args.config).read_text()))
    campaign.calls.retry_rate_limited = args.retry_rate_limited
    campaign.calls.retry_invalid_output = args.retry_invalid_output
    with (campaign.output / ".campaign.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "generate":
            report = asyncio.run(campaign.generate_all())
        elif args.stage == "contacts":
            report = asyncio.run(campaign.generate_contacts_all())
        elif args.stage == "review":
            report = asyncio.run(campaign.review_all())
        elif args.stage == "correct-postal":
            report = asyncio.run(campaign.correct_postal_all())
        elif args.stage == "positions":
            from document_ocr.synthesis.curated_positions import position_campaign

            report = position_campaign(campaign)
        elif args.stage in {"validate", "publish"}:
            report = publish_campaign(campaign, publish=args.stage == "publish")
        else:
            report = campaign.render_all()
    print(json.dumps(report, indent=2))
    if report.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
