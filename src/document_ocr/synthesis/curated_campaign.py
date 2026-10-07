"""Full-scenario current-schema synthesis: plan, word, render and validate."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import re
import time
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import yaml

from document_ocr.labeling_agents.target_normalization import normalize_target_casing
from document_ocr.synthesis.curated import (
    CuratedSynthesisConfig,
    Runner,
    SourceContract,
    assign,
    digest,
    flat,
    interpolate,
    save,
)
from document_ocr.synthesis.curated_auxiliary import (
    augment_auxiliary_blueprint,
    auxiliary_surfaces,
    load_auxiliary_contract,
)
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
from document_ocr.synthesis.curated_identifiers import declared_identifier_values
from document_ocr.synthesis.curated_ownership import (
    lexical_expression,
    load_owned_blueprint,
    ownership_surfaces,
)
from document_ocr.synthesis.curated_physical import PhysicalSupport, prepare_physical_render
from document_ocr.synthesis.curated_publication import publish_campaign, review_contract_hash
from document_ocr.synthesis.curated_scenarios import (
    ScenarioCatalog,
    ScenarioSamplingConfig,
    ShipmentScenario,
    SourceCapabilities,
)
from document_ocr.synthesis.curated_templates import (
    SamplingBlueprint,
    render_sampling_blueprint,
    wrap_owned_text,
)
from document_ocr.synthesis.curated_wording import (
    RENDERED_REVIEW_PROMPT,
    WORDING_PROMPT,
    WordingBatch,
    WordingField,
    WordingRequest,
    native_wording_prompt,
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
from document_ocr.synthesis.template_compiler import contact_values
from document_ocr.synthesis.vessel_lexical import vessel_name_fit_exclusion_reason


def party_path(path: str) -> str | None:
    match = re.match(r"documentPatch\.parties\.[^.]+", path)
    return match[0] if match else None


def wording_request_hash(request: WordingRequest) -> str:
    """A changed instruction or structured contract invalidates saved wording."""
    return digest(
        {
            "system": WORDING_PROMPT,
            "request": native_wording_prompt([request]),
            "schema": wording_output_type([request]).model_json_schema(),
        }
    )


def apply_replacements(target: dict, replacements: dict) -> dict:
    result = deepcopy(target)
    for path, value in replacements.items():
        assign(result, path, deepcopy(value))
    return result


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
        elif path.endswith((".billOfLadingNumber", ".voyageNumber")) or ".sealNumbers[" in path:
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
    """Whole-unit loads license a category, not invented models or body styles.

    Use the registry's commercial phrase for these tightly coupled profiles.
    The remaining names and addresses still use the lexical generation path.
    """
    if scenario.provenance.get("physicalProfile") not in {
        "source_whole_units",
        "reviewed_whole_units",
    }:
        return {}
    keys = [v.key for v in blueprint.contract.variables if v.kind == "product"]
    if len(keys) != 1 or len(scenario.goods_identities) != 1:
        raise ValueError("whole-unit product wording requires one certified commodity region")
    return {keys[0]: scenario.goods_identities[0].phrase.upper()}


def wording_request(
    blueprint: SamplingBlueprint, scenario: ShipmentScenario, sample_id: str
) -> WordingRequest:
    bindings = blueprint.contract.targets
    identities = list(scenario.goods_identities)
    product_keys = [v.key for v in blueprint.contract.variables if v.kind == "product"]
    host_products = host_product_wording(blueprint, scenario)
    fields = []
    for variable in blueprint.contract.variables:
        if variable.kind not in {"name", "postal", "product"}:
            continue
        if variable.key in host_products:
            continue
        owners = [b.path for b in bindings if "{" + variable.key + "}" in b.expression]
        roles = sorted({p for path in owners if (p := party_path(path)) is not None})
        if roles:
            locations = {scenario.party_localities[p].model_dump_json() for p in roles}
            if len(locations) != 1:
                raise ValueError(
                    f"shared party region has conflicting sampled geography: {variable.key}"
                )
            geo = scenario.party_localities[roles[0]]
            requirement = f"New party in {geo.name}, {geo.country} ({geo.country_code})."
            if variable.kind == "postal":
                requirement += (
                    " Generate this postal component jointly with the other components "
                    "of this role. Use the example's component kinds and approximate line span; "
                    "POSTAL ASSEMBLY defines the complete address and required geography. "
                    "Use the supplied locality spelling once. A literal locality/country "
                    "in the assembly is supplied by the host. Choose street/site names "
                    "without reusing that locality/country inside them."
                )
            else:
                requirement += " A new fictional identity, not a renamed version of the example."
            role = ", ".join(roles) + " " + variable.kind
        elif variable.kind == "product":
            if len(product_keys) == 1:
                selected = identities
            elif len(product_keys) == len(identities):
                selected = [identities[product_keys.index(variable.key)]]
            else:
                raise ValueError("product fragments need a complete commodity ownership contract")
            requirement = "A specific commercial product within each category: " + "; ".join(
                f"HS {i.hs6}: {i.phrase}" for i in selected
            )
            requirement += (
                ". Choose concrete product wording within the category's constraints, "
                "rather than copying tariff alternatives as a list. "
                "Retain every distinguishing qualifier. Use unbranded generic products; "
                "specific manufacturer/model names and unsupplied technical specifications "
                "are outside this request. "
                "Exclude shipment counts, packaging, weight/volume totals "
                "and tariff-code captions; "
                "those are rendered separately."
            )
            role = "goods description"
        else:
            raise ValueError(f"lexical region has no current owner: {variable.key}")
        fields.append(WordingField(variable.key, role, variable.occurrences[0].text, requirement))
    fields.extend(field for field, _ in contact_wording_fields(blueprint, scenario).values())
    context = (
        f"Loading: {scenario.origin.name}, {scenario.origin.country}. "
        f"Discharge: {scenario.destination.name}, {scenario.destination.country}.\n"
        "Goods: " + "; ".join(i.phrase for i in identities)
    )
    context += (
        "\nREGISTRY CLASSIFICATION\n"
        + yaml.safe_dump(
            [identity.model_dump(mode="json") for identity in identities],
            sort_keys=False,
            allow_unicode=True,
        )
        + "The selected HS6 and its description path define each product. "
        "Same-heading alternatives are different classifications, not product choices. "
        "National child descriptions give permitted narrower examples within the selected HS6."
    )
    if thermal := scenario.provenance.get("thermalCommodityContext"):
        context += (
            "\nEMPIRICAL COLD-CHAIN BUNDLE\n"
            + yaml.safe_dump(thermal, sort_keys=False, allow_unicode=True)
            + "Preserve this observed commodity's physical form and processing state. "
            "Reword its identity without introducing preparation or processing claims "
            "that are absent from the observed description."
        )
    for binding in bindings:
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
    return WordingRequest(sample_id, context, tuple(fields))


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
            path.endswith((".equipmentIdentifier", ".billOfLadingNumber", ".voyageNumber"))
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
    for key, (_, paths) in contact_wording_fields(blueprint, scenario).items():
        for path in paths:
            assign(result, path, " ".join(wording[key].split()))
    return normalize_target_casing(result)[0]


def combine_surfaces(*groups: dict[str, str | list[str]]) -> dict[str, str | list[str]]:
    """Conflicting semantic owners are errors, not precedence-dependent edits."""
    result = {}
    for group in groups:
        for key, value in group.items():
            if key in result and result[key] != value:
                raise ValueError(f"conflicting render ownership: {key}")
            result[key] = value
    return result


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


def validate_sample(
    blueprint: SamplingBlueprint, scenario: ShipmentScenario, candidate: dict, task
) -> dict:
    """Recompute every output; then check sampled facts and accounting invariants."""
    country_codes = {
        role + ".country": geo.country_code
        for role, geo in scenario.party_localities.items()
        if role + ".country" in flat(candidate["target"])
    }
    if candidate["countryCodes"] != country_codes:
        raise ValueError("country-code authority differs from the sampled registry geography")
    text, target, proof = render_sampling_blueprint(
        blueprint,
        candidate["target"],
        candidate["renderValues"],
        surface_values=candidate["surfaceValues"],
        certified_country_codes=country_codes,
    )
    if (candidate["joinedRawText"], candidate["proof"]) != (text, proof):
        raise ValueError("candidate differs from independent source/edit replay")
    if candidate["joinedRawTextSha256"] != digest(text.encode()):
        raise ValueError("candidate OCR hash differs")
    if task.canonicalize(target) != target:
        raise ValueError("candidate is not canonical under the current reduced training contract")
    patch = target["documentPatch"]
    source = blueprint.target["documentPatch"]
    if patch.get("route") == source.get("route"):
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
        if "country" in new and new["country"] != location.country:
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
        base = {key: config[key] for key in CuratedSynthesisConfig.model_fields}
        self.calls = Runner(root, base)
        self.output = self.calls.output
        data = root / config["dataset"]
        self.rows = {
            r["documentId"]: r
            for r in map(json.loads, (data / "train.jsonl").read_text().splitlines())
        }
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

    def blueprint(self, sid: str) -> SamplingBlueprint:
        envelope = json.loads(
            (self.root / self.config["source_contracts"] / sid / "contract.json").read_text()
        )
        row = self.rows[sid]
        if envelope["sourceSha256"] != digest(row["joinedRawText"].encode()) or envelope[
            "targetSha256"
        ] != digest(row["target"]):
            raise ValueError("source contract identity mismatch")
        historical = json.loads(
            (self.root / self.config["historical_catalog"] / sid / "template.json").read_text()
        )
        blueprint = load_owned_blueprint(
            row,
            historical,
            SourceContract.model_validate(envelope["contract"]),
            self.root / self.config["ownership"],
        )
        return augment_auxiliary_blueprint(blueprint, self.auxiliary)

    def plan(self, sid: str, variant: int) -> tuple[SamplingBlueprint, ShipmentScenario, str, dict]:
        blueprint = self.blueprint(sid)
        scenario = self.catalog.sample(
            self.rows[sid],
            seed=self.config["seed"],
            variant=variant,
            capabilities=self.capabilities[sid],
        )
        sample_id = "syn_full_v7_" + digest([sid, self.config["seed"], variant])[:24]
        target = public_identity_updates(
            apply_replacements(blueprint.target, scenario.replacements),
            scenario,
            DeterministicStream(self.config["seed"], "full-current-v7", sample_id),
        )
        if old_vessel := target["documentPatch"].get("transport", {}).get("vesselName"):
            candidates = tuple(v for v in self.vessels if v != old_vessel)
            stream = DeterministicStream(self.config["seed"], "full-current-v7", sample_id)
            target["documentPatch"]["transport"]["vesselName"] = candidates[
                stream.derive("vessel-name").randbelow(len(candidates))
            ]
        return blueprint, scenario, sample_id, target

    def plans(self, sid: str) -> list[tuple]:
        return [
            self.plan(sid, variant) for variant in range(1, self.config["variants_per_source"] + 1)
        ]

    async def generate(self, sid: str) -> dict:
        plans = self.plans(sid)
        requests = [
            wording_request(bp, scenario, sample_id) for bp, scenario, sample_id, _ in plans
        ]
        wording, pending = {}, []
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
                    wording.update(validate_wording(batch, [request]))
                    continue
            pending.append(request)
        if pending:
            output = await self.calls.call(
                "exact-wording",
                sid,
                wording_output_type(pending),
                WORDING_PROMPT,
                native_wording_prompt(pending),
            )
            wording.update(unpack_wording(output, pending))
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
                },
            )
        contacts = await self.generate_contacts(sid)
        return {"sourceDocumentId": sid, "generated": len(plans), "contacts": contacts}

    def contact_requests(self, sid: str) -> list:
        parties = []
        for bp, scenario, sample_id, target in self.plans(sid):
            receipt = json.loads((self.output / "wording" / f"{sample_id}.json").read_text())
            if receipt["requestSha256"] != wording_request_hash(
                wording_request(bp, scenario, sample_id)
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
            output = await self.calls.call(
                "company-contacts",
                sid,
                contact_output_type(parties),
                CONTACT_PROMPT,
                contact_prompt(parties),
            )
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
        results = await asyncio.gather(
            *(self.generate_contacts(sid) for sid in self.config["source_ids"]),
            return_exceptions=True,
        )
        report = {"sources": [], "errors": [], "costUsd": str(self.calls.spent)}
        for sid, result in zip(self.config["source_ids"], results, strict=True):
            if isinstance(result, BaseException):
                report["errors"].append({"sourceDocumentId": sid, "error": str(result)})
            else:
                report["sources"].append({"sourceDocumentId": sid, **result})
        save(self.output / "contacts-summary.json", report)
        return report

    async def generate_all(self) -> dict:
        started = time.perf_counter()
        results = await asyncio.gather(
            *(self.generate(sid) for sid in self.config["source_ids"]), return_exceptions=True
        )
        report = {
            "sources": [],
            "errors": [],
            "seconds": time.perf_counter() - started,
            "costUsd": str(self.calls.spent),
        }
        for sid, result in zip(self.config["source_ids"], results, strict=True):
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
            original = wording_request(blueprint, scenario, sample_id)
            if receipt["requestSha256"] != wording_request_hash(original):
                raise ValueError("postal correction requires current generation inputs")
            values = combine_surfaces(receipt["values"], host_product_wording(blueprint, scenario))
            assembled = flat(assemble_lexical_target(blueprint, target, values, scenario))
            failed, keys = [], set()
            for role, location in scenario.party_localities.items():
                address_path = role + ".addressLine"
                if address_path not in assembled:
                    continue
                try:
                    validate_postal_geography(
                        assembled[address_path],
                        locality=location.name,
                        country=location.country,
                        country_code=location.country_code,
                        country_labelled=role + ".country" in assembled,
                    )
                except ValueError as error:
                    failed.append(
                        f"{role}: {error}. Previous complete address: {assembled[address_path]}"
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
                    f.requirement + " This is a fresh synthetic postal rewrite, not transcription. "
                    "Choose new street/site wording without the sampled locality or "
                    "country inside street, building, district or region names. "
                    "Use the locality and requested country once across the complete "
                    "assembly; one mention suffices when their names are identical.",
                )
                for f in original.fields
                if f.key in keys
            )
            request = WordingRequest(
                sample_id,
                original.context
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
        output = await self.calls.call(
            "postal-correction",
            sid,
            wording_output_type(requests),
            "Correct fictional postal text. The supplied old values FAILED validation; "
            "they are not formatting examples. Return corrected regions only. Each complete "
            "address must contain its supplied locality exactly once and its requested "
            "country exactly once. Host-owned literal components already count. If a "
            "fictional street/site name repeats the locality, choose a different street/site "
            "name. Preserve approximate address hierarchy and use uppercase text.",
            "\n\n---\n\n".join(
                f"SHIPMENT s{index}\n"
                + "\n".join(
                    line
                    for line in request.context.splitlines()
                    if line.startswith(("POSTAL ASSEMBLY", "documentPatch.parties."))
                )
                + "\n\n"
                + "\n\n".join(
                    f"REGION {field.key} ({field.role})\nINVALID OLD VALUE:\n{field.example}"
                    for field in request.fields
                )
                for index, request in enumerate(requests)
            ),
        )
        proposed = unpack_wording(output, requests)
        remaining = []
        for request in requests:
            path, receipt, failed, blueprint, scenario, target = changes[request.sample_id]
            before = deepcopy(receipt["values"])
            receipt["values"].update(proposed[request.sample_id])
            assembled = flat(
                assemble_lexical_target(
                    blueprint,
                    target,
                    combine_surfaces(receipt["values"], host_product_wording(blueprint, scenario)),
                    scenario,
                )
            )
            post_errors = []
            for role, location in scenario.party_localities.items():
                if role + ".addressLine" not in assembled:
                    continue
                try:
                    validate_postal_geography(
                        assembled[role + ".addressLine"],
                        locality=location.name,
                        country=location.country,
                        country_code=location.country_code,
                        country_labelled=role + ".country" in assembled,
                    )
                except ValueError as error:
                    post_errors.append(f"{role}: {error}")
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
        results = await asyncio.gather(
            *(self.correct_postal(sid) for sid in self.config["source_ids"]),
            return_exceptions=True,
        )
        report = {"sources": [], "errors": [], "costUsd": str(self.calls.spent)}
        for sid, result in zip(self.config["source_ids"], results, strict=True):
            if isinstance(result, BaseException):
                report["errors"].append({"sourceDocumentId": sid, "error": str(result)})
            else:
                report["sources"].append(result)
        save(self.output / "postal-correction-summary.json", report)
        return report

    def render(self, sid: str, variant: int, *, persist: bool = True) -> dict:
        blueprint, scenario, sample_id, target = self.plan(sid, variant)
        receipt = json.loads((self.output / "wording" / f"{sample_id}.json").read_text())
        request = wording_request(blueprint, scenario, sample_id)
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
        physical = prepare_physical_render(
            blueprint, scenario, target, support=self.physical_support
        )
        host_values = combine_surfaces(
            host_variable_values(blueprint, physical.target, stream),
            physical.variable_values,
        )
        target = assemble_lexical_target(blueprint, physical.target, wording, scenario, host_values)
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
            physical.surface_values, ownership_surfaces(blueprint, target, scenario)
        )
        surfaces, auxiliary_receipt = auxiliary_surfaces(
            blueprint, scenario, target, stream, existing=surfaces
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
        candidate["validation"] = validate_sample(blueprint, scenario, candidate, self.calls.task)
        if contact_receipt is not None:
            candidate["contactReceiptSha256"] = digest(contact_receipt)
        if persist:
            save(self.output / "candidates" / f"{sample_id}.json", candidate)
        return candidate

    def replay_candidate(self, candidate: dict) -> None:
        expected = self.render(candidate["sourceDocumentId"], candidate["variant"], persist=False)
        if expected != candidate:
            raise ValueError("candidate differs from source/scenario/wording authority replay")

    async def review(self, sid: str) -> dict:
        candidates = []
        for variant in range(1, self.config["variants_per_source"] + 1):
            sample_id = "syn_full_v7_" + digest([sid, self.config["seed"], variant])[:24]
            candidate = json.loads((self.output / "candidates" / f"{sample_id}.json").read_text())
            self.replay_candidate(candidate)
            candidates.append(candidate)
        blocks = []
        for index, candidate in enumerate(candidates):
            facts = candidate["scenario"]
            blocks.append(
                "SHIPMENT "
                + f"s{index}"
                + "\nPARTY LOCALITIES\n"
                + "\n".join(
                    f"{role}: {location['name']}, {location['country']} "
                    f"({location['country_code']})"
                    for role, location in facts["party_localities"].items()
                )
                + "\nREGISTRY COMMODITIES\n"
                + yaml.safe_dump(facts["goods_identities"], sort_keys=False, allow_unicode=True)
                + (
                    "\nEMPIRICAL COLD-CHAIN BUNDLE (preserve observed form/processing)\n"
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
                + "\nCOMPLETE RENDERED TEXT\n"
                + candidate["joinedRawText"]
            )
        output = await self.calls.call(
            "rendered-review",
            sid,
            review_output_type(len(candidates)),
            RENDERED_REVIEW_PROMPT,
            "\n\n---\n\n".join(blocks),
        )
        output = unpack_review(output, [c["documentId"] for c in candidates])
        identities = {c["documentId"] for c in candidates}
        if len(output.reviewed_ids) != len(identities) or set(output.reviewed_ids) != identities:
            raise ValueError("rendered review did not cover every candidate exactly once")
        if any(f.sample_id not in identities for f in output.findings):
            raise ValueError("rendered review cites an unrequested candidate")
        receipt = {
            "sourceDocumentId": sid,
            "reviewContractSha256": review_contract_hash(len(candidates)),
            "candidateHashes": {c["documentId"]: digest(c) for c in candidates},
            "review": output.model_dump(mode="json"),
        }
        save(self.output / "reviews" / f"{sid}.json", receipt)
        return {"sourceDocumentId": sid, "findings": len(output.findings)}

    async def review_all(self) -> dict:
        results = await asyncio.gather(
            *(self.review(sid) for sid in self.config["source_ids"]), return_exceptions=True
        )
        report = {"sources": [], "errors": [], "costUsd": str(self.calls.spent)}
        for sid, result in zip(self.config["source_ids"], results, strict=True):
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
        report = {"accepted": [], "errors": []}
        for sid in self.config["source_ids"]:
            for variant in range(1, self.config["variants_per_source"] + 1):
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
