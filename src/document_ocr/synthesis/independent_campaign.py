"""Donor-independent flat V7 campaign using explicit, reviewed source authorities."""

from __future__ import annotations

import copy
import json
import re
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from document_ocr.hashing import canonical_json_sha256
from document_ocr.synthesis.curated import digest, save
from document_ocr.synthesis.curated_campaign import (
    Campaign,
    apply_replacements,
    postal_wording_errors,
    public_identity_updates,
)
from document_ocr.synthesis.curated_equipment_inventory import (
    compile_equipment_inventory,
    rebase_equipment_inventory_headers,
)
from document_ocr.synthesis.curated_hs import sample_source_hs_codes
from document_ocr.synthesis.curated_load_profiles import RegistryLoadProfile
from document_ocr.synthesis.curated_ownership import apply_dependent_text
from document_ocr.synthesis.curated_phones import prepare_phone_rendering
from document_ocr.synthesis.curated_physical import (
    PhysicalRenderPlan,
    PhysicalSupport,
    _active_tare_bindings,
    _tare_surfaces,
    printed_tare_equipment_domain,
)
from document_ocr.synthesis.curated_scenarios import (
    CandidateIncompatibility,
    FilePin,
    GoodsIdentity,
    ScenarioLocation,
    ShipmentScenario,
    SourceCapabilities,
    _pin,
)
from document_ocr.synthesis.curated_templates import SamplingBlueprint
from document_ocr.synthesis.curated_wording import WordingRequest
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.independent_cargo_presentation import validate_cargo_presentation
from document_ocr.synthesis.independent_envelopes import DEFAULT_ROW_VOLUME_M3
from document_ocr.synthesis.independent_geography import (
    IndependentGeographyGate,
    load_independent_geography_certificates,
    project_composite_locality_homonyms,
    recompile_projected_equipment_inventory,
)
from document_ocr.synthesis.independent_numeric import (
    IndependentNumericCertificates,
    project_handling_targets,
)
from document_ocr.synthesis.independent_packages import verify_native_choice_evidence
from document_ocr.synthesis.independent_presentation import (
    load_distinct_actor_reviews,
    load_locality_reviews,
    resolve_reviewed_distinct_actors,
    resolve_reviewed_locality_assemblies,
    semantic_private_packing,
    strengthen_presentation_request,
    validate_distinct_actors,
    validate_locality_owners,
    wording_scenario,
)
from document_ocr.synthesis.independent_relationships import (
    validate_packing_direction,
    verify_native_plan_evidence,
)
from document_ocr.synthesis.independent_sampling import IndependentSampler
from document_ocr.synthesis.independent_wording import (
    load_certificates as load_wording_certificates,
)
from document_ocr.synthesis.independent_wording import (
    phrase_surfaces,
)
from document_ocr.synthesis.independent_wording import (
    prepare_blueprint as prepare_wording_blueprint,
)
from document_ocr.synthesis.independent_wording import (
    prepare_request as prepare_wording_request,
)
from document_ocr.synthesis.transport_capacity import equipment_capacity

_MASS_FACTORS = {
    "kilogram": Decimal(1),
    "metric_tonne": Decimal(1000),
    "pound": Decimal("0.45359237"),
}
_INPUT_KEYS = frozenset(
    {
        "registry",
        "numeric_certificates",
        "wording_certificates",
        "geography_certificates",
        "requests",
        "private_plans",
        "locality_owner_reviews",
        "distinct_actor_reviews",
    }
)
Plan = tuple[SamplingBlueprint, ShipmentScenario, str, dict[str, Any]]


def _pinned_json(root: Path, value: Mapping[str, Any]) -> Any:
    pin = FilePin.model_validate(value)
    if pin.records is not None:
        raise ValueError("independent JSON input does not use a record-count pin")
    return json.loads(_pin(root, pin).read_bytes())


def _complete_optional_package_suffix(blueprint: SamplingBlueprint) -> SamplingBlueprint:
    """Attach an immediate optional plural to its exact scalar type owner."""
    raw = blueprint.source.encode()
    bindings = copy.deepcopy(dict(blueprint.historical_bindings))
    regions = list(blueprint.regions)
    extensions = []
    for index, region in enumerate(regions):
        if (
            region.curated_key is not None
            or not region.target_paths
            or not all(
                re.fullmatch(
                    r"documentPatch\.goodsItemDetails\[\d+\]\.numberAndTypeOfPackages"
                    r"\[\d+\]\.typeCategory",
                    path,
                )
                for path in region.target_paths
            )
        ):
            continue
        suffix = re.match(rb"\((?:S|ES)\)", raw[region.end :], re.I)
        if suffix is None:
            continue
        end = region.end + suffix.end()
        if any(
            other.start < end and region.end < other.end
            for other_index, other in enumerate(regions)
            if other_index != index
        ):
            raise ValueError("optional plural overlaps another certified owner")
        quote = raw[region.start : end].decode()
        occurrence = bindings[region.key]["occurrences"][region.occurrence]
        if any(
            occurrence.get(key) != expected
            for key, expected in (
                ("byte_start", region.start),
                ("byte_end", region.end),
                ("source_text", region.source),
            )
        ):
            raise ValueError("optional plural owner no longer matches its source occurrence")
        occurrence.update(byte_end=end, source_text=quote)
        if "source_text_sha256" in occurrence:
            if occurrence["source_text_sha256"] != digest(region.source.encode()):
                raise ValueError("optional plural source hash changed")
            occurrence["source_text_sha256"] = digest(quote.encode())
        regions[index] = replace(region, end=end, source=quote)
        extensions.append(
            {
                "owner": region.key,
                "occurrence": region.occurrence,
                "before": region.source,
                "after": quote,
            }
        )
    ownership = copy.deepcopy(dict(blueprint.ownership_data))
    ownership["independentCompleteOptionalPluralOwners"] = extensions
    return replace(
        blueprint,
        historical_bindings=bindings,
        regions=tuple(regions),
        ownership_data=ownership,
    )


def _independent_family(source_family: str) -> str:
    mapping = {
        "ambient": "ambient",
        "chilled": "chilled",
        "frozen": "frozen",
        "dg_chemical": "dg_chemical",
        "vehicle": "ambient",
        "dg_vehicle": "dg_chemical",
    }
    if source_family not in mapping:
        raise ValueError("unknown source presentation family: " + source_family)
    return mapping[source_family]


def _equipment_authority(
    blueprint: SamplingBlueprint,
    support: PhysicalSupport,
    allowed_pairs: tuple[str, ...],
    regime: str,
) -> tuple[list[dict[str, str]] | None, list[dict[str, Any]]]:
    """Constrain hardware only where source-printed tare/typing requires it."""
    rows = blueprint.target["documentPatch"].get("containerInformation", [])
    # A printed hardware type cannot silently turn into the independent
    # regime's GP or reefer envelope, even if the label has a text owner.
    required_type = "REFRIGERATED" if regime in {"chilled", "frozen"} else "GENERAL_PURPOSE"
    source_domain = None
    if (
        regime == "ambient"
        and rows
        and rows[0].get("typeCategory") in {"OPEN_TOP", "PLATFORM"}
        and all(row.get("typeCategory") == rows[0]["typeCategory"] for row in rows)
    ):
        # Keep the actual printed special hardware. This enables new cargo,
        # not an implicit source-label conversion into the ordinary GP domain.
        required_type = rows[0]["typeCategory"]
        if any("sizeCategory" not in row for row in rows):
            raise ValueError("unenclosed source requires explicit equipment size ownership")
        source_domain = {row["sizeCategory"] + "|" + required_type for row in rows}
    elif (
        regime == "ambient"
        and rows
        and all(row.get("typeCategory") == "REFRIGERATED" for row in rows)
    ):
        if any("temperatureSetpoint" in row for row in rows):
            raise ValueError("ambient reefer source cannot retain a thermal setpoint")
        if any("sizeCategory" not in row for row in rows):
            raise ValueError("ambient reefer source requires explicit equipment size ownership")
        # NOR and other unconditioned reefer presentations retain their actual
        # insulated hardware/capacity without inventing a cold-chain setting.
        required_type = "REFRIGERATED"
        source_domain = {row["sizeCategory"] + "|REFRIGERATED" for row in rows}
    for row in rows:
        category = row.get("typeCategory")
        if category is not None and category != required_type:
            raise ValueError(
                "source equipment type lacks the independent regime envelope: " + category
            )
    untyped = bool(rows) and all(
        "sizeCategory" not in row and "typeCategory" not in row for row in rows
    )
    retained_tares = []
    if untyped and _active_tare_bindings(blueprint):  # type: ignore[no-untyped-call]
        _, retained_tares = _tare_surfaces(  # type: ignore[no-untyped-call]
            blueprint, blueprint.target, support, None, {}
        )
        owners = set(range(len(rows)))
        # Untyped rows can still have explicit tare cells. The physical parser
        # has already proved each row owner and every aggregate's exact source
        # sum. Retain those private observations without inferring size/type.
        # An aggregate-only presentation must cover the complete inventory;
        # individual presentations must jointly cover it, never a partial guess.
        individual = {r["owner"] for r in retained_tares if "owner" in r}
        aggregates = [r for r in retained_tares if "owner" not in r]
        if not retained_tares or (
            individual and individual != owners
        ) or any(
            set(r.get("owners", ())) != owners
            or r.get("ownershipBasis") != "explicit_aggregate_or_declared_sum"
            for r in aggregates
        ):
            raise ValueError("untyped tare lacks complete certified hardware ownership")
        tare_domain = None
    else:
        tare_domain = printed_tare_equipment_domain(blueprint, support)
    domain = set(allowed_pairs) if allowed_pairs else None
    if source_domain is not None:
        domain = source_domain if domain is None else domain & source_domain
    if tare_domain is not None:
        domain = set(tare_domain) if domain is None else domain & set(tare_domain)
    if domain is not None:
        domain = {pair for pair in domain if pair.endswith("|" + required_type)}
    if domain is not None and not domain:
        raise ValueError("source hardware presentation has no eligible equipment pair")
    equipment_domain = (
        [
            {"sizeCategory": pair.split("|", 1)[0], "typeCategory": pair.split("|", 1)[1]}
            for pair in sorted(domain)
        ]
        if domain is not None
        else None
    )
    return equipment_domain, retained_tares


def _fixed_equipment_slots(
    blueprint: SamplingBlueprint, capability: SourceCapabilities
) -> list[dict[str, str]] | None:
    mix = capability.equipment_mix
    if mix is None:
        return None
    rows = blueprint.target["documentPatch"].get("containerInformation", [])
    mix.validate_source(blueprint.source, rows)
    return [
        {"sizeCategory": member.sizeCategory, "typeCategory": member.typeCategory}
        for member in mix.members
        for _ in range(member.count)
    ]


def source_equipment_readiness(
    blueprint: SamplingBlueprint,
    support: PhysicalSupport,
    capability: SourceCapabilities,
    regime: str,
) -> tuple[list[dict[str, str]] | None, list[dict[str, Any]], list[dict[str, str]] | None]:
    """Preflight source-owned hardware before any native cargo-choice call."""
    domain, retained_tares = _equipment_authority(
        blueprint, support, capability.allowed_equipment_pairs, regime
    )
    slots = _fixed_equipment_slots(blueprint, capability)
    if slots is not None:
        required_type = (
            domain[0]["typeCategory"]
            if domain is not None
            else "REFRIGERATED"
            if regime in {"chilled", "frozen"}
            else "GENERAL_PURPOSE"
        )
        eligible = (
            {(item["sizeCategory"], item["typeCategory"]) for item in domain}
            if domain is not None
            else None
        )
        if any(
            slot["typeCategory"] != required_type
            or (
                eligible is not None
                and (slot["sizeCategory"], slot["typeCategory"]) not in eligible
            )
            for slot in slots
        ):
            raise ValueError("source-certified equipment mix lacks a regime-compatible domain")
    return domain, retained_tares, slots


def _validate_row_capacity(
    draw: Mapping[str, Any], physical_rows: Sequence[Mapping[str, Any]]
) -> None:
    """Recheck every realized row after equal-load and support-mass composition."""
    allocations = draw["allocations"]
    if len(allocations) != len(physical_rows):
        raise ValueError("independent row capacity count differs from allocations")
    if not allocations:
        return
    if sum(row["packageQuantity"] for row in allocations) != draw["package"]["packageQuantity"]:
        raise ValueError("independent row quantities do not conserve the counted total")
    unit_net = Decimal(draw["physical"]["unitNetKg"])
    unit_gross = Decimal(draw["physical"]["unitGrossKg"])
    unit_cube = Decimal(draw["physical"]["unitVolumeM3"])
    if not all(value.is_finite() and value > 0 for value in (unit_net, unit_gross, unit_cube)):
        raise ValueError("independent unit loads require positive finite numeric facts")
    if unit_net > unit_gross:
        raise ValueError("independent net mass exceeds gross mass")
    for allocation, physical in zip(allocations, physical_rows, strict=True):
        quantity = allocation["packageQuantity"]
        if type(quantity) is not int or quantity < 1 or quantity != physical["quantity"]:
            raise ValueError("independent row quantity differs from capacity witness")
        gross = Decimal(allocation["grossKg"])
        net = Decimal(allocation["netKg"])
        volume = Decimal(allocation["volumeM3"])
        if (gross, net, volume) != (
            unit_gross * quantity,
            unit_net * quantity,
            unit_cube * quantity,
        ):
            raise ValueError("independent row loads differ from selected unit facts")
        capacity = physical["capacity"]
        payload = Decimal(capacity["payloadKg"])
        cube_text = capacity["volumeM3"]
        if not payload.is_finite() or payload <= 0:
            raise ValueError("independent enclosed equipment lacks positive capacity")
        if cube_text is None and capacity.get("family") in {
            "twenty_open_top",
            "forty_open_top",
            "out_of_gauge",
        }:
            # The carrier contract deliberately has no enclosed-volume ceiling
            # for OT/platforms. Keep the independent 24 m3 planning budget,
            # rather than inventing a GP capacity or an out-of-gauge load.
            if gross > payload or volume > DEFAULT_ROW_VOLUME_M3:
                raise CandidateIncompatibility(
                    "unenclosed load exceeds payload or synthetic planning budget"
                )
            continue
        if cube_text is None:
            raise ValueError("independent enclosed equipment lacks positive capacity")
        cube = Decimal(cube_text)
        if not cube.is_finite() or cube <= 0:
            raise ValueError("independent enclosed equipment lacks positive cube")
        if gross > payload or volume > cube:
            raise CandidateIncompatibility(
                "independent realized row exceeds selected equipment capacity"
            )


class IndependentCampaign(Campaign):
    """Selected-source donor-independent cargo; ordinary V7 output/replay contract."""

    sampling_mode = "independent"

    def __init__(self, root: Path, config: dict[str, Any]) -> None:
        independent = config["independent"]
        if set(independent) != _INPUT_KEYS:
            raise ValueError(
                "independent config requires exactly its registry and seven pinned inputs: "
                + str(sorted(_INPUT_KEYS ^ set(independent)))
            )
        self.sampler = IndependentSampler.from_config(root, independent["registry"])
        numeric_pin = FilePin.model_validate(independent["numeric_certificates"])
        self.numeric = IndependentNumericCertificates.from_path(
            _pin(root, numeric_pin), expected_sha256=numeric_pin.sha256
        )
        wording_pin = FilePin.model_validate(independent["wording_certificates"])
        self.wording_certificates = load_wording_certificates(
            _pin(root, wording_pin), expected_sha256=wording_pin.sha256
        )
        geography_pin = FilePin.model_validate(independent["geography_certificates"])
        self.geography = IndependentGeographyGate(
            load_independent_geography_certificates(
                _pin(root, geography_pin).read_bytes(), expected_sha256=geography_pin.sha256
            )
        )
        self.requests = _pinned_json(root, independent["requests"])
        self.private_plans = _pinned_json(root, independent["private_plans"])
        locality_pin = FilePin.model_validate(independent["locality_owner_reviews"])
        self.locality_owner_reviews = load_locality_reviews(
            _pin(root, locality_pin).read_bytes(), expected_sha256=locality_pin.sha256
        )
        actor_pin = FilePin.model_validate(independent["distinct_actor_reviews"])
        self.distinct_actor_reviews = load_distinct_actor_reviews(
            _pin(root, actor_pin).read_bytes(), expected_sha256=actor_pin.sha256
        )
        self._prepared: dict[str, SamplingBlueprint] = {}
        self._planned: dict[tuple[str, int], Plan] = {}
        self._plans_by_sample_id: dict[str, tuple[str, int]] = {}
        self._plan_overrides: dict[tuple[str, int], Plan] = {}
        self._projection_originals: dict[tuple[str, int], Plan] = {}
        self._blueprint_lock = threading.RLock()
        self._plan_lock_guard = threading.Lock()
        self._plan_locks: dict[tuple[str, int], threading.Lock] = {}
        super().__init__(root, config)
        slots = {
            f"{sid}:{variant}"
            for sid in self.source_ids
            for variant in range(1, self.variant_counts[sid] + 1)
        }
        if not self.source_ids or set(self.requests) != slots or set(self.private_plans) != slots:
            raise ValueError("independent request/private-plan keys differ from selected slots")
        for sid in self.source_ids:
            self.numeric.has_certificate(sid)
            holds = self.numeric.source_readiness_holds(sid)
            for hold in holds:
                _pin(root, hold.evidence)
            if holds:
                raise ValueError(
                    "source_readiness_hold: "
                    + sid
                    + ": "
                    + "; ".join(f"{hold.category}: {hold.reason}" for hold in holds)
                )
        for key, request in self.requests.items():
            sid, variant = key.rsplit(":", 1)
            if request["sourceDocumentId"] != sid or request["variant"] != int(variant):
                raise ValueError("independent request identity differs from selected slot")
            if request.get("choiceHold"):
                raise ValueError("independent package choice remains held: " + key)
            if request.get("planningStage") != "native_package_choice_accepted":
                raise ValueError("independent package choice lacks accepted native stage: " + key)
            self.sampler.validate_request(request)
            validate_cargo_presentation(
                self.numeric.certificate(sid).get("cargoPresentation"), request
            )
            verify_native_choice_evidence(
                root,
                request,
                provider=self.calls.config["provider"],
                hs_codes=self.sampler.phrases,
            )
            private = self.private_plans[key]
            operative = verify_native_plan_evidence(
                root,
                request,
                private,
                provider=self.calls.config["provider"],
                hs_codes=self.sampler.phrases,
                allowed_relations=self.numeric.allowed_private_relations(sid),
                source_certificate_sha256=digest(self.numeric.certificate(sid)),
            )
            if operative["privatePackingRelation"] != "MIXED":
                validate_packing_direction(
                    request["selectedCategory"],
                    operative["privatePackingRelation"],
                    operative["relatedPackageType"],
                )
            if operative["privatePackingRelation"] not in self.numeric.allowed_private_relations(
                sid
            ):
                raise ValueError("private relationship lacks source-certified presentation: " + key)

    def blueprint(self, sid: str) -> SamplingBlueprint:
        with self._blueprint_lock:
            if sid in self._prepared:
                return self._prepared[sid]
            ordinary = Campaign.blueprint(self, sid)
            goods = ordinary.target["documentPatch"]["goodsItemDetails"]
            if len(goods) != 1 or len(goods[0]["numberAndTypeOfPackages"]) != 1:
                raise ValueError("independent flat campaign requires one goods and one package row")
            prepared = self.numeric.apply_blueprint(ordinary, sid)
            if sid in self.wording_certificates:
                prepared = prepare_wording_blueprint(prepared, self.wording_certificates[sid])
            prepared = _complete_optional_package_suffix(prepared)
            rows = prepared.target["documentPatch"].get("containerInformation", [])
            if len(rows) > 1 and all(
                {"sizeCategory", "typeCategory"} <= row.keys() for row in rows
            ):
                prepared = rebase_equipment_inventory_headers(prepared)
                self.equipment_inventories[sid] = compile_equipment_inventory(
                    prepared, allow_uniform_aggregate=True
                )
            self.geography.bind_blueprint(ordinary, prepared)
            self._prepared[sid] = prepared
            return prepared

    def validate_source_readiness(self) -> None:
        """Recheck pinned source ownership and the independent hardware domain."""
        Campaign.validate_source_readiness(self)
        policy = self.catalog.config.equipment_inventory
        expected_inventories = set()
        for sid in self.source_ids:
            blueprint = self.blueprint(sid)
            rows = blueprint.target["documentPatch"].get("containerInformation", [])
            policy_source = policy is not None and sid in policy.source_ids
            if policy_source or (
                len(rows) > 1
                and all({"sizeCategory", "typeCategory"} <= row.keys() for row in rows)
            ):
                expected_inventories.add(sid)
                current = self.equipment_inventories.get(sid)
                compiled = compile_equipment_inventory(
                    blueprint, allow_uniform_aggregate=not policy_source
                )
                if current != compiled:
                    raise ValueError("independent equipment inventory differs from source owners")
                compiled.validate_rows(rows)
            capability = self._planning_capability(sid, blueprint)
            regime = "dg" if capability.family == "dg_chemical" else capability.family
            for variant in range(1, self.variant_counts[sid] + 1):
                if self.requests[f"{sid}:{variant}"]["regime"] != regime:
                    raise ValueError(
                        "independent request regime conflicts with source presentation"
                    )
            source_equipment_readiness(blueprint, self.physical_support, capability, regime)
        if set(self.equipment_inventories) != expected_inventories:
            raise ValueError("independent equipment inventory scope differs from source owners")

    def _planning_capability(self, sid: str, blueprint: SamplingBlueprint) -> SourceCapabilities:
        original = self.capabilities[sid]
        family = _independent_family(original.family)
        regime = "dg" if family == "dg_chemical" else family
        if self.requests[f"{sid}:1"]["regime"] != regime:
            raise ValueError("independent cargo regime conflicts with source presentation")
        constraints = blueprint.ownership_data["freshPrivatePacking"]["constraints"]
        equal = [item for item in constraints if item["kind"] == "equal_container_loads"]
        mixed = [item for item in constraints if item["kind"] == "mixed_package_conservation"]
        if len(equal) + len(mixed) != len(constraints) or len(equal) > 1 or len(mixed) > 1:
            raise ValueError("source has unsupported or duplicate packing constraint")
        rows = len(blueprint.target["documentPatch"].get("containerInformation", []))
        if equal and rows < 1:
            raise ValueError("equal-container-load certificate has no container rows")
        quantity_multiple = rows if equal else 1
        return original.model_copy(
            update={
                "family": family,
                "allowed_hs_codes": (),
                "allowed_hs_headings": (),
                "allowed_package_categories": (),
                "source_load_accounting": None,
                "fixed_package_quantity": None,
                "quantity_multiple": quantity_multiple,
            }
        )

    @staticmethod
    def _equal_container_loads(draw: dict[str, Any]) -> None:
        total = draw["package"]["packageQuantity"]
        rows = draw["allocations"]
        if not rows or total % len(rows):
            raise CandidateIncompatibility(
                "certified shared container statement needs equal integral loads"
            )
        count = total // len(rows)
        for row in rows:
            row["packageQuantity"] = count
            for output, unit in (
                ("netKg", "unitNetKg"),
                ("grossKg", "unitGrossKg"),
                ("volumeM3", "unitVolumeM3"),
            ):
                row[output] = str(Decimal(draw["physical"][unit]) * count)

    def _scenario_from_draw(
        self,
        blueprint: SamplingBlueprint,
        capability: SourceCapabilities,
        draw: dict[str, Any],
        variant: int,
        sample_id: str,
        origin: ScenarioLocation,
        destination: ScenarioLocation,
        parties: dict[str, ScenarioLocation],
        geography: dict[str, Any],
        geography_receipt: dict[str, Any],
        ledger: dict[str, Any],
        retained_tares: list[dict[str, Any]],
        request: Mapping[str, Any],
    ) -> Plan:
        source = blueprint.target["documentPatch"]
        original_goods = source["goodsItemDetails"][0]
        goods = copy.deepcopy(original_goods)
        containers = copy.deepcopy(source.get("containerInformation", []))
        counted = goods["numberAndTypeOfPackages"][0]
        if "typeCategory" not in counted:
            raise ValueError("flat package source lacks a public type-category owner")
        counted.update(
            packageQuantity=draw["package"]["packageQuantity"],
            typeCategory=draw["internalCategory"],
        )
        for measure in ("grossWeight", "netWeight", "volume"):
            if measure not in goods:
                continue
            factor = Decimal(1) if measure == "volume" else _MASS_FACTORS[goods[measure]["unit"]]
            goods[measure]["value"] = float(Decimal(draw["physical"][measure]["value"]) / factor)
        identities = []
        for item in draw["identities"]:
            code = item["hs6"]
            if not isinstance(code, str) or re.fullmatch(r"[0-9]{6}", code) is None:
                raise ValueError("independent identity lacks a genuine registry HS6 join")
            identities.append(
                GoodsIdentity(
                    hs6=code,
                    phrase=item["phrase"],
                    phrase_role=item.get("phraseRole", "classification_context"),
                    classification=self.catalog._classification(code),
                )
            )
        goods["description"] = "; ".join(identity.phrase for identity in identities)
        if "hsCodes" in goods:
            goods["hsCodes"] = sample_source_hs_codes(
                original_goods["hsCodes"],
                [identity.hs6 for identity in identities],
                DeterministicStream(self.config["seed"], "independent-hs-render-v1", sample_id),
            )
        drawn_rows = draw["allocations"]
        if len(drawn_rows) != len(containers):
            raise ValueError("independent loads differ from printed equipment cardinality")
        by_identifier = {
            item["equipmentIdentifier"]: index for index, item in enumerate(containers)
        }
        for placement in goods.get("splitGoodsPlacement", []):
            if "packageQuantity" in placement:
                placement["packageQuantity"] = drawn_rows[
                    by_identifier[placement["equipmentIdentifier"]]
                ]["packageQuantity"]
        if "dangerousGoods" in goods:
            declaration = draw["dangerousGoods"]
            if declaration is None:
                raise ValueError("DG source received a non-DG independent declaration")
            for entry in goods["dangerousGoods"]:
                for field in entry:
                    if field not in declaration:
                        raise ValueError("independent DG declaration lacks printed field " + field)
                    if field == "flashPoint":
                        temperature = entry[field]["temperature"]
                        value = Decimal(declaration[field]["valueCelsius"])
                        if temperature["unit"] == "fahrenheit":
                            value = value * 9 / 5 + 32
                        temperature["value"] = float(value)
                    else:
                        entry[field] = copy.deepcopy(declaration[field])
        for index, container in enumerate(containers):
            equipment = draw["equipment"][index]
            for field in ("sizeCategory", "typeCategory"):
                if field in container:
                    container[field] = equipment[field]
            if "temperatureSetpoint" in container:
                thermal = Decimal(draw["thermal"]["temperatureCelsius"])
                if container["temperatureSetpoint"]["unit"] == "fahrenheit":
                    thermal = thermal * 9 / 5 + 32
                container["temperatureSetpoint"]["value"] = float(thermal)
        replacements = copy.deepcopy(geography)
        for key in goods.get("origin", {}):
            goods["origin"][key] = replacements[f"documentPatch.goodsItemDetails[0].origin.{key}"]
        for field, value in goods.items():
            if value != original_goods.get(field):
                replacements[f"documentPatch.goodsItemDetails[0].{field}"] = value
        for index, container in enumerate(containers):
            for field, value in container.items():
                if value != source["containerInformation"][index].get(field):
                    replacements[f"documentPatch.containerInformation[{index}].{field}"] = value
        quantity = draw["package"]["packageQuantity"]
        profile = RegistryLoadProfile(
            id=draw["profile"]["id"],
            hs_headings=tuple(dict.fromkeys(identity.hs6[:4] for identity in identities)),
            form=draw["profile"]["form"],
            package_category=draw["internalCategory"],
            unit_gross_kg=draw["physical"]["unitGrossKg"],
            unit_net_kg=draw["physical"]["unitNetKg"],
            unit_volume_m3=draw["physical"]["unitVolumeM3"],
            quantity_min=quantity,
            quantity_max=quantity,
            rationale="Full-registry independent flat V7 sampling; no commodity donor.",
        )
        physical_rows = []
        for index, container in enumerate(containers):
            capacity = equipment_capacity(draw["equipment"][index], self.catalog.limits)
            row_quantity = drawn_rows[index]["packageQuantity"]
            physical_rows.append(
                {
                    "equipmentIdentifier": container["equipmentIdentifier"],
                    "quantity": row_quantity,
                    "shareNumerator": row_quantity,
                    "shareDenominator": quantity,
                    "capacity": {
                        "payloadKg": str(capacity.payload_kg),
                        "volumeM3": (
                            str(capacity.volume_m3) if capacity.volume_m3 is not None else None
                        ),
                    },
                }
            )
            if capacity.family in {"twenty_open_top", "forty_open_top", "out_of_gauge"}:
                # The witness needs its explicitly unenclosed basis;
                # do not substitute the ordinary closed-box cube contract.
                physical_rows[-1]["capacity"]["family"] = capacity.family
        provenance = {
            "method": "independent_full_registry_flat_v1",
            "seed": self.config["seed"],
            "variant": variant,
            "physicalProfile": "independent_registry_form",
            "capabilities": capability.model_dump(mode="json"),
            "registryLoadProfile": profile.model_dump(mode="json"),
            "explicitPhysicalFacts": {
                key: {"value": str(unit * quantity), "unit": unit_name}
                for key, unit, unit_name in (
                    ("grossWeight", profile.unit_gross_kg, "kilogram"),
                    ("netWeight", profile.unit_net_kg, "kilogram"),
                    ("volume", profile.unit_volume_m3, "cubic_metre"),
                )
            },
            "physicalRows": physical_rows,
            "geographyEligibility": geography_receipt,
            "independentDraw": draw,
            "freshPrivatePacking": ledger,
            "equipmentRequestedMode": request["equipmentMode"],
            "retainedPublicUntypedHardwareTare": retained_tares,
            "publicEquipmentCategoryPresence": [
                {key: key in row for key in ("sizeCategory", "typeCategory")}
                for row in source.get("containerInformation", [])
            ],
        }
        if draw["thermal"] is not None:
            provenance["thermalCommodityContext"] = {
                "basis": "independent_full_hs_registry",
                "profile": request["regime"],
                "sampledHS6": [identity.hs6 for identity in identities],
                "commodityDescriptions": [identity.phrase for identity in identities],
                "productForm": profile.form,
                **draw["thermal"],
            }
            provenance["ventilationCbmPerHour"] = draw["thermal"]["ventilationCbmPerHour"]
        if draw["dangerousGoods"] is not None:
            declaration = draw["dangerousGoods"]
            provenance["dgHmtRecordId"] = declaration["recordId"]
            provenance["dgPrintedFacts"] = {
                "properShippingName": declaration["properShippingName"],
                "unNumber": declaration["unNumber"],
                "class": declaration["exactHazardClass"],
                "packingGroup": declaration["packingGroupCode"],
                "subsidiaryHazards": declaration["exactSubsidiaryHazards"],
            }
            if "flashPoint" in declaration:
                provenance["dgPrintedFacts"]["flashPointAuthority"] = self.sampler.flash[
                    declaration["recordId"]
                ].model_dump(mode="json")
        scenario = ShipmentScenario(
            source_id=blueprint.document_id,
            variant=variant,
            origin=origin,
            destination=destination,
            route_locations={
                field: ScenarioLocation.model_validate(geography_receipt["routeNodes"][node])
                for field, node in geography_receipt.get("routeFields", {}).items()
            },
            party_localities=parties,
            goods_identities=tuple(identities),
            replacements=replacements,
            cargo={"goodsItemDetails": [goods], "containerInformation": containers},
            provenance=provenance,
        )
        target = public_identity_updates(
            apply_replacements(blueprint.target, replacements),
            scenario,
            DeterministicStream(self.config["seed"], "full-current-v7", sample_id),
        )
        target = apply_dependent_text(blueprint, target, phase="scenario")
        return blueprint, scenario, sample_id, target

    def plan(self, sid: str, variant: int) -> Plan:
        identity = (sid, variant)
        override = self._plan_overrides.get(identity)
        if override is not None:
            return override
        with self._plan_lock_guard:
            slot_lock = self._plan_locks.setdefault(identity, threading.Lock())
        with slot_lock:
            if identity in self._planned:
                return self._planned[identity]
            if sid not in self.variant_counts or not 1 <= variant <= self.variant_counts[sid]:
                raise ValueError("independent source/variant is outside the selected campaign")
            key = f"{sid}:{variant}"
            request = self.requests[key]
            blueprint = self.blueprint(sid)
            capability = self._planning_capability(sid, blueprint)
            regime = "dg" if capability.family == "dg_chemical" else capability.family
            if request["regime"] != regime:
                raise ValueError("independent request regime conflicts with source presentation")
            private = self.private_plans[key]
            equipment_domain, retained_tares, equipment_slots = source_equipment_readiness(
                blueprint, self.physical_support, capability, regime
            )
            if "equipmentSlots" in request:
                if equipment_slots is not None and request["equipmentSlots"] != equipment_slots:
                    raise ValueError(
                        "frozen carrying hardware differs from the source-certified mix"
                    )
                equipment_slots = request["equipmentSlots"]
            sample_id = "syn_full_v7_" + digest([sid, self.config["seed"], variant])[:24]
            stream = DeterministicStream(
                self.config["seed"], "independent-flat-v1", f"{sid}:{variant}"
            )
            origin, destination, parties, geography, geography_receipt = self.catalog._geography(
                self.rows[sid], capability, stream.derive("geography")
            )
            failures: Counter[str] = Counter()
            for attempt in range(self.catalog.config.maximum_candidates):
                draw = self.sampler.draw(
                    request=request,
                    seed=self.config["seed"],
                    draw_key=f"{sid}:{variant}:{attempt}",
                    container_count=len(
                        blueprint.target["documentPatch"].get("containerInformation", [])
                    ),
                    selected_category=request["selectedCategory"],
                    counted_unit_context=request["countedUnitContext"],
                    quantity_multiple=capability.quantity_multiple,
                    equipment_domain=equipment_domain,
                    equipment_slots=equipment_slots,
                )
                try:
                    if capability.quantity_multiple > 1:
                        self._equal_container_loads(draw)
                    draw = self.numeric.prepare_draw(blueprint, draw, private, request)
                    ledger = self.numeric.build_ledger(
                        blueprint, draw, private, self.config["seed"], variant
                    )
                    if sid in self.wording_certificates:
                        overlap = set(ledger["surfaces"]) & set(phrase_surfaces(blueprint))
                        if overlap:
                            raise ValueError(
                                "wording phrase owner overlaps independent numeric owner: "
                                + str(sorted(overlap))
                            )
                        ledger["surfaces"].update(phrase_surfaces(blueprint))
                    result = self._scenario_from_draw(
                        blueprint,
                        capability,
                        draw,
                        variant,
                        sample_id,
                        origin,
                        destination,
                        parties,
                        geography,
                        geography_receipt,
                        ledger,
                        retained_tares,
                        request,
                    )
                    self.numeric.validate_composed_capacity(
                        draw, result[1].provenance["physicalRows"]
                    )
                    _validate_row_capacity(draw, result[1].provenance["physicalRows"])
                    self.physical_render(result[0], result[1], result[3])
                    prepare_phone_rendering(
                        blueprint,
                        result[3],
                        source_country_codes=self.source_phone_country_codes(blueprint),
                        target_country_codes={
                            role: location.country_code
                            for role, location in result[1].party_localities.items()
                        },
                        policy=self.phone_policy,
                        stream=DeterministicStream(
                            self.config["seed"], "phone-presentations-v1", sample_id
                        ),
                    )
                except CandidateIncompatibility as error:
                    failures[str(error)] += 1
                    continue
                if old_vessel := result[3]["documentPatch"].get("transport", {}).get("vesselName"):
                    candidates = tuple(v for v in self.vessels if v != old_vessel)
                    if not candidates:
                        raise ValueError("no distinct certified vessel name available")
                    result[3]["documentPatch"]["transport"]["vesselName"] = candidates[
                        DeterministicStream(self.config["seed"], "full-current-v7", sample_id)
                        .derive("vessel-name")
                        .randbelow(len(candidates))
                    ]
                scenario = result[1].model_copy(
                    update={
                        "provenance": {
                            **result[1].provenance,
                            "numericFitAttempts": attempt + 1,
                            "numericFitRejections": dict(failures),
                        }
                    }
                )
                admitted = result[0], scenario, result[2], result[3]
                self._planned[identity] = admitted
                self._plans_by_sample_id[sample_id] = identity
                return admitted
            raise ValueError(
                "independent draw could not fit source presentation in "
                f"{self.catalog.config.maximum_candidates} attempts: {dict(failures)}"
            )

    def physical_render(
        self, blueprint: SamplingBlueprint, scenario: ShipmentScenario, target: dict[str, Any]
    ) -> PhysicalRenderPlan:
        inventory = self.equipment_inventories.get(blueprint.document_id)
        if inventory is not None and inventory.source_target_sha256 != canonical_json_sha256(
            blueprint.target
        ):
            identity = blueprint.document_id, scenario.variant
            originals = self._projection_originals
            if identity not in originals:
                raise ValueError("projected equipment inventory has no original certified plan")
            original = originals[identity]
            current = recompile_projected_equipment_inventory(
                original[0], blueprint, original[3], target, inventory
            )
            view = copy.copy(self)
            view.equipment_inventories = {
                **self.equipment_inventories,
                blueprint.document_id: current,
            }
            base: PhysicalRenderPlan = Campaign.physical_render(  # type: ignore[no-untyped-call]
                view, blueprint, scenario, target
            )
        else:
            base = Campaign.physical_render(  # type: ignore[no-untyped-call]
                self, blueprint, scenario, target
            )
        retained = scenario.provenance["retainedPublicUntypedHardwareTare"]
        if retained:
            if base.receipt["tares"] != retained:
                raise ValueError("untyped aggregate tare receipt changed")
            original_rows = blueprint.target["documentPatch"].get("containerInformation", [])
            current_rows = base.target["documentPatch"].get("containerInformation", [])
            if len(original_rows) != len(current_rows) or any(
                key in row for row in current_rows for key in ("sizeCategory", "typeCategory")
            ):
                raise ValueError("untyped hardware branch introduced public equipment categories")
        ledger = scenario.provenance["freshPrivatePacking"]
        surfaces = copy.deepcopy(base.surface_values)
        changed = set(ledger["surfaces"])
        # A complete certified owner may replace a source recipe; every
        # replacement must still correspond to one compiled region.
        known = {region.key for region in blueprint.regions}
        if changed - known:
            raise ValueError("private ledger surface lacks an owned source region")
        surfaces.update(copy.deepcopy(ledger["surfaces"]))
        projected, handling_changes = project_handling_targets(blueprint, base.target, ledger)
        result = replace(base, target=projected, surface_values=surfaces)
        if (
            result.variable_values != base.variable_values
            or {
                key
                for key in set(base.surface_values) | set(surfaces)
                if base.surface_values.get(key) != surfaces.get(key)
            }
            - changed
        ):
            raise ValueError("private surface overlay changed variables or unrelated owner")
        return replace(
            result,
            receipt={
                **base.receipt,
                "freshPrivatePacking": ledger,
                "matchedPhysicalPreservation": {
                    "sameTarget": result.target == base.target,
                    **(
                        {"completeHandlingTargetChanges": handling_changes}
                        if handling_changes else {}
                    ),
                    "sameVariables": True,
                    "sameNonpackingSurfaces": True,
                },
            },
        )

    def wording_request(
        self,
        blueprint: SamplingBlueprint,
        scenario: ShipmentScenario,
        sample_id: str,
        target: dict[str, Any],
    ) -> WordingRequest:
        semantic = wording_scenario(scenario)
        request = Campaign.wording_request(self, blueprint, semantic, sample_id, target)
        draw = scenario.provenance["independentDraw"]
        ledger = scenario.provenance["freshPrivatePacking"]
        facts = {
            "countedCategory": draw["internalCategory"],
            "packageQuantity": draw["package"]["packageQuantity"],
            "privatePacking": semantic_private_packing(ledger),
        }
        request = replace(
            request,
            context=request.context
            + "\nAUTHORITATIVE COUNTED/PACKING FACTS\n"
            + yaml.safe_dump(facts, sort_keys=False)
            + "The host renders these facts. Do not contradict the counted unit or add an "
            "unrequested material/compression claim about it. Do not invent extra shipment "
            "totals, allocations, container counts, HS codes or regulated declarations. "
            "Keep genuine separate inner/support roles explicit; source private quantities "
            "are presentation examples, not facts for the new shipment. Preserve useful "
            "product details and the source-conditioned description scope.\n"
            + self.numeric.wording_context(ledger),
        )
        if blueprint.document_id in self.wording_certificates:
            request = prepare_wording_request(
                request,
                blueprint,
                thermal_state=self.requests[f"{blueprint.document_id}:{scenario.variant}"][
                    "regime"
                ],
            )
        request = self.geography.prepare_request(request, blueprint, scenario)
        localities = resolve_reviewed_locality_assemblies(
            blueprint,
            scenario,
            self.locality_owner_reviews.for_document(blueprint.document_id),
            geography=self.geography,
        )
        actors = resolve_reviewed_distinct_actors(
            blueprint, self.distinct_actor_reviews.for_document(blueprint.document_id)
        )
        return strengthen_presentation_request(
            request,
            locality_assemblies=localities,
            distinct_actors=actors,
            regime=self.requests[f"{blueprint.document_id}:{scenario.variant}"]["regime"],
        )

    def _check_independent_wording_values(
        self,
        blueprint: SamplingBlueprint,
        scenario: ShipmentScenario,
        values: Mapping[str, str],
    ) -> tuple[dict[str, str], list[Any]]:
        values, _ = self.numeric.project_wording(values, scenario.provenance["freshPrivatePacking"])
        findings: list[Any] = list(self.geography.validate_values(blueprint, values, scenario))
        sid = blueprint.document_id
        localities = resolve_reviewed_locality_assemblies(
            blueprint,
            scenario,
            self.locality_owner_reviews.for_document(sid),
            geography=self.geography,
        )
        findings.extend(validate_locality_owners(values, localities))
        actors = resolve_reviewed_distinct_actors(
            blueprint, self.distinct_actor_reviews.for_document(sid)
        )
        findings.extend(validate_distinct_actors(values, actors))
        return values, findings

    def _independent_wording_values(
        self,
        blueprint: SamplingBlueprint,
        scenario: ShipmentScenario,
        values: Mapping[str, str],
    ) -> dict[str, str]:
        values, findings = self._check_independent_wording_values(blueprint, scenario, values)
        if findings:
            raise ValueError(
                "independent wording violates reviewed owner authority: " + str(findings)
            )
        return values

    def _generated_wording_diagnostics(
        self, request: WordingRequest, values: dict[str, str]
    ) -> dict[str, Any]:
        if request.sample_id not in self._plans_by_sample_id:
            raise ValueError("independent wording lacks its planned sample identity")
        sid, variant = self._plans_by_sample_id[request.sample_id]
        blueprint, scenario, sample_id, target = self.plan(sid, variant)
        if request != self.wording_request(blueprint, scenario, sample_id, target):
            raise ValueError("independent wording request differs from current plan")
        checked, findings = self._check_independent_wording_values(blueprint, scenario, values)
        diagnostics: dict[str, Any] = {}
        if findings:
            diagnostics["reviewedOwnerAuthority"] = [str(finding) for finding in findings]
        postal_errors = postal_wording_errors(blueprint, scenario, target, checked)
        if postal_errors:
            diagnostics["postalSourceOwnership"] = postal_errors
        return diagnostics

    def _validated_wording(
        self, receipt: dict[str, Any], request: WordingRequest
    ) -> dict[str, str]:
        sid = receipt["sourceDocumentId"]
        variant = receipt["scenario"]["variant"]
        blueprint, scenario, sample_id, target = self.plan(sid, variant)
        current = self.wording_request(blueprint, scenario, sample_id, target)
        values = Campaign._validated_wording(self, receipt, current)
        return self._independent_wording_values(blueprint, scenario, values)

    def _current_wording(self, receipt: dict[str, Any], request: WordingRequest) -> dict[str, str]:
        return self._validated_wording(receipt, request)

    def review_private_packing_context(
        self, blueprint: SamplingBlueprint, candidate: dict[str, Any]
    ) -> dict[str, Any]:
        sid, variant = blueprint.document_id, candidate["variant"]
        request = self.requests[f"{sid}:{variant}"]
        private = self.private_plans[f"{sid}:{variant}"]
        _, scenario, sample_id, _ = self.plan(sid, variant)
        ledger = scenario.provenance["freshPrivatePacking"]
        if (
            candidate["documentId"] != sample_id
            or candidate["scenario"]["provenance"]["freshPrivatePacking"] != ledger
            or candidate["physical"]["freshPrivatePacking"] != ledger
            or ledger["sourceDocumentId"] != sid
            or ledger["certificateFileSha256"] != self.numeric.file_sha256
        ):
            raise ValueError("review private packing differs from the current certified plan")
        if any(
            candidate["surfaceValues"].get(key) != value
            for key, value in ledger["surfaces"].items()
        ):
            raise ValueError("review private packing differs from rendered numeric surfaces")
        counted = candidate["target"]["documentPatch"]["goodsItemDetails"][0][
            "numberAndTypeOfPackages"
        ][0]
        if (
            counted["typeCategory"] != ledger["countedCategory"]
            or counted["packageQuantity"] != ledger["publicQuantity"]
            or request["selectedCategory"] != ledger["countedCategory"]
            or private["privatePackingRelation"] != ledger["relation"]
            or private["relatedPackageType"] != ledger["relatedCategory"]
        ):
            raise ValueError("review private packing conflicts with frozen package authority")
        if private["origin"] in {"native_typed_private_plan", "native_mixed_accounting"}:
            private_evidence = private["nativePlanPacketSha256"]
        elif private["origin"] == "reviewed_no_private_relation":
            private_evidence = private["sourceCertificateSha256"]
        else:
            raise ValueError("review private packing has unknown plan authority")
        return {
            "authority": "reviewed independent numeric ledger and verified frozen package plans",
            "sampleId": sample_id,
            "declaredCountedUnit": {
                "category": ledger["countedCategory"],
                "quantity": ledger["publicQuantity"],
                "countedUnitContext": request["countedUnitContext"],
            },
            "privatePacking": {
                "relation": ledger["relation"],
                "relatedCategory": ledger["relatedCategory"],
                "relatedQuantity": ledger["relatedQuantity"],
            },
            "evidence": {
                "numericCertificateSha256": ledger["certificateSha256"],
                "nativePackageChoicePacketSha256": request["nativeChoiceEvidence"]["packetSha256"],
                "privatePlanOrigin": private["origin"],
                "privatePlanAuthoritySha256": private_evidence,
            },
        }

    def render(self, sid: str, variant: int, *, persist: bool = True) -> dict[str, Any]:
        candidate = Campaign.render(self, sid, variant, persist=False)
        original = self.plan(sid, variant)
        projected, projection = project_composite_locality_homonyms(original, candidate)
        geography = self.geography
        if projection is not None:
            identity = sid, variant
            view = copy.copy(self)
            view._plan_overrides = {identity: projected}
            view._projection_originals = {identity: original}
            geography = self.geography.locality_projection_view(original[0], projected[0])
            view.geography = geography
            candidate = Campaign.render(view, sid, variant, persist=False)
            candidate["compositeLocalityProjection"] = projection
        ledger = original[1].provenance["freshPrivatePacking"]
        receipt = json.loads((self.output / "wording" / f"{original[2]}.json").read_bytes())
        _, numeric_projection = self.numeric.project_wording(receipt["values"], ledger)
        candidate["numericWordingProjection"] = numeric_projection
        findings = geography.validate_rendered(projected[0], candidate, projected[1])
        if findings:
            raise ValueError("rendered postal assembly violates source authority: " + str(findings))
        if persist:
            path = self.output / "candidates" / f"{candidate['documentId']}.json"
            if path.exists() and json.loads(path.read_bytes()) != candidate:
                raise ValueError("existing independent candidate differs from deterministic replay")
            save(path, candidate)
        return candidate
