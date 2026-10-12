"""Registry-backed shipment scenarios for current, single-goods source templates.

Sampling is separate from text generation and rendering.  A scenario carries
current-V7 facts and provenance, never a legacy label projection. Quantities,
mass and volume use either train-only observed shipments or reviewed synthetic
form/load profiles. Registry identities and template domains constrain both.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import TransportCapacityConfig
from document_ocr.synthesis.country_registry import CountryRegistry, load_iso_country_registry
from document_ocr.synthesis.curated_equipment import PrintedEquipmentMix, capacity_assignment
from document_ocr.synthesis.curated_equipment_inventory import EquipmentInventoryContract
from document_ocr.synthesis.curated_goods import RegistryThermalPolicy
from document_ocr.synthesis.curated_load_profiles import (
    RegistryLoadProfile,
    RegistryLoadProfiles,
    weighted_choice,
)
from document_ocr.synthesis.curated_parties import PartyLocalityGroup, share_party_localities
from document_ocr.synthesis.curated_routes import GoodsOriginIdentifier, RouteTopology
from document_ocr.synthesis.dangerous_goods_registry import (
    DangerousGoodsHmtRecord,
    LoadedDangerousGoodsRegistry,
    _normalized_chemical_identity,
    load_dangerous_goods_registry,
)
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.hs_registry import (
    UkGlobalTariffRegistry,
    compile_uk_global_tariff_registry,
    load_ukgt_source_pin,
)
from document_ocr.synthesis.template_compiler.contact_values import _phone_prefix
from document_ocr.synthesis.template_compiler.dangerous_goods_packaging import profile
from document_ocr.synthesis.thermal_goods import classify_thermal_hs
from document_ocr.synthesis.transport_capacity import (
    capacity_limits,
    classify_equipment,
    equipment_capacity,
)

Family = Literal["ambient", "frozen", "chilled", "dg_chemical", "dg_vehicle", "vehicle"]
_MEASURES = ("grossWeight", "netWeight", "volume")
_MASS_FACTORS = {
    "kilogram": Decimal(1),
    "metric_tonne": Decimal(1000),
    "pound": Decimal("0.45359237"),
}
_REGISTRY_GP_PAIRS = tuple(
    size + "|GENERAL_PURPOSE"
    for size in (
        "TWENTY_FOOT_STANDARD_HEIGHT",
        "TWENTY_FOOT_HIGH_CUBE",
        "FORTY_FOOT_STANDARD_HEIGHT",
        "FORTY_FOOT_HIGH_CUBE",
        "FORTY_FIVE_FOOT_HIGH_CUBE",
    )
)


class CandidateIncompatibility(ValueError):
    """A sampled draw cannot fit its declared domain or source numeric grid.

    Only this error is eligible for bounded cargo resampling. Invalid source
    contracts, missing ownership, schema errors and unexpected exceptions are
    not candidate failures and must propagate without retry.
    """


def _party_signature(party: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Same ordered identity/address words; punctuation never hides a new site."""
    return tuple(
        tuple(re.findall(r"\w+", party.get(field, "").casefold()))
        for field in ("name", "addressLine")
    )


class FilePin(BaseModel):
    """Exact existing local registry content; not a mutable remote dependency."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: int | None = Field(default=None, gt=0)


class ObservedHSOverride(BaseModel):
    """Reviewed synthesis-only identity for a source printing a non-current HS code.

    This does not repair an extraction label. Exact label identity, old codes,
    and quoted source text are pinned before admitting the observed load bundle.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    from_hs6: tuple[str, ...] = Field(min_length=1)
    to_hs6: tuple[str, ...] = Field(min_length=1)
    source_text_literals: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)


InventoryMode = Literal["uniform", "mixed"]


class EquipmentInventorySampling(BaseModel):
    """Per-draw inventory distribution, requiring certified identifier ownership.

    ``source_ids`` is the explicit certified multi-container eligibility domain,
    not a promise that unrelated single-container or thermal sources are mixed.
    ``modes`` is an optional cyclic variant schedule for controlled campaigns;
    otherwise the mode is a seeded Bernoulli draw. Neither pins a source to a
    permanent mixed mode. Counts are drawn from the declared feasible intersection,
    never reduced after a draw; an empty intersection is a configuration error.
    Canonical category weights use ``ScenarioSamplingConfig.equipment_weights``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    source_ids: tuple[str, ...] = Field(min_length=1)
    mixed_probability: Decimal = Field(ge=0, le=1)
    category_count_weights: dict[
        Annotated[int, Field(strict=True, ge=2, le=len(_REGISTRY_GP_PAIRS))],
        Annotated[int, Field(strict=True, gt=0)],
    ] = Field(default_factory=lambda: {2: 1}, min_length=1)
    modes: tuple[InventoryMode, ...] | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )

    @field_validator("source_ids")
    @classmethod
    def exact_source_scope(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(not sid or sid.strip() != sid for sid in value):
            raise ValueError("inventory source scope requires distinct exact document IDs")
        return value

    @field_validator("category_count_weights", mode="before")
    @classmethod
    def canonical_count_keys(cls, value: Any) -> Any:
        # JSON object keys are strings. Accept only the exact integer spelling,
        # so persisted configuration replays without relaxing integer semantics.
        if isinstance(value, dict):
            normalized = {}
            for key, weight in value.items():
                key = int(key) if isinstance(key, str) and key in {"2", "3", "4", "5"} else key
                if key in normalized:
                    raise ValueError("duplicate inventory category-count weight")
                normalized[key] = weight
            return normalized
        return value

    def mode(self, variant: int, stream: DeterministicStream) -> InventoryMode:
        if variant < 1:
            raise ValueError("inventory sampling requires a positive variant")
        if self.modes is not None:
            return self.modes[(variant - 1) % len(self.modes)]
        numerator, denominator = self.mixed_probability.as_integer_ratio()
        return "mixed" if stream.randbelow(denominator) < numerator else "uniform"


class ScenarioSamplingConfig(BaseModel):
    """Pinned registries and explicit numeric policy for the current pilot."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    countries: FilePin
    locations: FilePin
    world_ports: FilePin
    localities: FilePin
    hs_manifest: FilePin
    hs_metadata: FilePin
    hs_report: FilePin
    commodity_phrases: FilePin
    hmt: FilePin
    ecics: FilePin
    transport_capacity: TransportCapacityConfig
    quantity_scale_min: Decimal = Field(default=Decimal("0.6"), gt=0, le=1)
    quantity_scale_max: Decimal = Field(default=Decimal("1.2"), ge=1, le=2)
    scale_steps: int = Field(default=60, ge=1)
    maximum_candidates: int = Field(default=512, ge=1)
    observed_hs_overrides: dict[str, ObservedHSOverride] = Field(default_factory=dict)
    thermal: RegistryThermalPolicy = Field(default_factory=RegistryThermalPolicy)
    load_profiles: FilePin | None = Field(default=None, exclude_if=lambda v: v is None)
    profile_weights: dict[str, Annotated[int, Field(strict=True, gt=0)]] = Field(
        default_factory=dict, exclude_if=lambda v: not v
    )
    package_weights: dict[str, Annotated[int, Field(strict=True, gt=0)]] = Field(
        default_factory=dict, exclude_if=lambda v: not v
    )
    equipment_weights: dict[str, Annotated[int, Field(strict=True, gt=0)]] = Field(
        default_factory=dict, exclude_if=lambda v: not v
    )
    equipment_inventory: EquipmentInventorySampling | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Earlier pipeline's ambient registry scope. Potentially hazardous chemical,
    # petroleum and perishable-food chapters require their dedicated profiles.
    ambient_hs_chapters: tuple[str, ...] = (
        "25",
        "39",
        "40",
        "42",
        "44",
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "70",
        "72",
        "73",
        "74",
        "75",
        "76",
        "78",
        "79",
        "80",
        "81",
        "82",
        "83",
        "84",
        "85",
        "86",
        "87",
        "88",
        "89",
        "90",
        "91",
        "92",
        "94",
        "95",
        "96",
    )


class ReviewedWholeUnitDonor(BaseModel):
    """Pinned training observation explicitly reviewed as whole vehicles, not parts.

    Package count is certified as the number of whole vehicles by the quoted
    source evidence. The receiving template retains its explicit packaging
    wording; mass, cube, equipment and exact HS identity come from this donor.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    whole_unit_count: int = Field(gt=0)
    source_text_literals: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class SourceCapabilities(BaseModel):
    """Adjudicated semantic and layout freedoms of one source template.

    These are semantic domains, not source-ID exceptions.  For example, a
    vehicle list can fix its count while allowing other tractors in HS8701.
    A printed national customs scheme can pin the destination country.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    family: Family
    identity_count: int = Field(default=1, ge=1)
    allowed_hs_headings: tuple[str, ...] = ()
    allowed_hs_codes: tuple[str, ...] = ()
    allowed_package_categories: tuple[str, ...] = ()
    allowed_equipment_pairs: tuple[str, ...] = ()
    origin_country: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    destination_country: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    fixed_package_quantity: int | None = Field(default=None, gt=0)
    quantity_multiple: int = Field(default=1, gt=0)
    required_private_measures: tuple[Literal["grossWeight", "netWeight", "volume"], ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
        description=(
            "Positive source-printed physical measures absent from public labels. "
            "Sample them jointly from the selected cargo authority and enforce equipment capacity; "
            "keep them private rather than adding target fields."
        ),
    )
    equipment_mix: PrintedEquipmentMix | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Fixed printed equipment mix without individual identifier classification.",
    )
    party_sides: dict[str, Literal["origin", "destination"]] = Field(default_factory=dict)
    party_locality_groups: tuple[PartyLocalityGroup, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    location_sides: dict[str, Literal["origin", "destination"]] = Field(default_factory=dict)
    route_topology: RouteTopology | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    goods_origin_identifiers: dict[str, GoodsOriginIdentifier] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    require_contact_support: bool = True
    physical_profile: Literal[
        "observed_train_bundle",
        "source_bundle",
        "source_whole_units",
        "reviewed_whole_units",
        "registry_form",
    ] = "observed_train_bundle"
    reviewed_whole_unit_donors: dict[str, ReviewedWholeUnitDonor] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_domains(self) -> SourceCapabilities:
        if self.physical_profile == "registry_form" and (
            self.family != "ambient" or self.equipment_mix is not None
        ):
            raise ValueError("registry-form sampling requires ambient, identifier-owned equipment")
        grouped_parties = [path for group in self.party_locality_groups for path in group.members]
        if len(set(grouped_parties)) != len(grouped_parties):
            raise ValueError("party locality certificates must use disjoint member groups")
        if any(re.fullmatch(r"[0-9]{4}", value) is None for value in self.allowed_hs_headings):
            raise ValueError("HS domains require exact four-digit headings")
        if any(re.fullmatch(r"[0-9]{6}", value) is None for value in self.allowed_hs_codes):
            raise ValueError("HS code domains require exact authoritative six-digit identities")
        if self.physical_profile in {
            "source_whole_units",
            "reviewed_whole_units",
        } and self.family not in {"vehicle", "dg_vehicle"}:
            raise ValueError("whole-unit source physics is restricted to enumerated vehicles")
        if self.physical_profile == "reviewed_whole_units":
            if not self.reviewed_whole_unit_donors or not self.allowed_hs_codes:
                raise ValueError(
                    "reviewed whole-unit physics requires pinned HS domains and donors"
                )
            if self.fixed_package_quantity is None:
                raise ValueError("reviewed whole-unit physics requires an explicit vehicle count")
        elif self.reviewed_whole_unit_donors:
            raise ValueError("reviewed donor pins require the reviewed whole-unit profile")
        if self.family in {"vehicle", "dg_vehicle"} and not self.allowed_hs_headings:
            raise ValueError("vehicle scenarios require an explicit source-owned HS heading")
        if self.family.startswith("dg_") and self.identity_count != 1:
            raise ValueError("this single-goods pilot requires one coherent DG identity")
        if self.route_topology is not None:
            for path, node in self.route_topology.party_nodes.items():
                if path in self.party_sides and self.party_sides[path] != node:
                    raise ValueError("party side conflicts with its explicit route node")
            for path, node in self.route_topology.location_nodes.items():
                if path in self.location_sides and self.location_sides[path] != node:
                    raise ValueError("location side conflicts with its explicit route node")
        nodes = {
            "origin",
            "destination",
            *(self.route_topology.nodes if self.route_topology else {}),
        }
        if any(v.node not in nodes for v in self.goods_origin_identifiers.values()):
            raise ValueError("goods-origin identifier refers to an undefined route node")
        return self


class ScenarioLocation(BaseModel):
    """Registry locality and country, shared by all text/label renderings."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    country_code: str
    country: str
    registry: Literal["unlocode_wpi", "geonames"]
    registry_id: str


class HSAlternative(BaseModel):
    """A different HS6; its path is relative to the shared heading, not a product option."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    hs6: str = Field(pattern=r"^[0-9]{6}$")
    description_path: tuple[str, ...]


class HSChildExample(BaseModel):
    """An immediate national branch within the selected HS6; illustrative, not mandatory."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    description: str


class HSClassificationContext(BaseModel):
    """Authoritative taxonomy context, not an additional shipment specification.

    National child descriptions supply concrete in-scope options without
    promoting one country's additional distinctions into global HS6 rules.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    authority: str
    chapter_code: str
    chapter_description: str
    heading_code: str
    heading_description: str
    description_path: tuple[str, ...]
    same_heading_alternatives: tuple[HSAlternative, ...]
    immediate_national_children: tuple[HSChildExample, ...]


class GoodsIdentity(BaseModel):
    """An HS6 reference or commercial identity with explicit wording authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    hs6: str = Field(pattern=r"^[0-9]{6}$")
    phrase: str
    phrase_role: Literal["commercial", "classification_context"] = Field(
        default="commercial",
        exclude_if=lambda value: value == "commercial",
        description=(
            "Commercial phrases may name the shipped goods. Classification context is "
            "authoritative HS nomenclature, not cargo prose; its exclusions and aggregate "
            "alternatives must not be copied as conflicting physical-form assertions."
        ),
    )
    observed_hs6: str | None = None
    classification: HSClassificationContext


class ShipmentScenario(BaseModel):
    """Private synthesis plan.  ``replacements`` updates only existing target fields.

    Names, address strings, identifiers and product prose are rendered by the
    caller from this plan.  ``cargo`` exposes the numeric/category plan for that
    call, while container IDs remain source handles until host regeneration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    variant: int
    origin: ScenarioLocation
    destination: ScenarioLocation
    # Inactive optional topology adds no metadata to direct-route receipts.
    route_locations: dict[str, ScenarioLocation] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    party_localities: dict[str, ScenarioLocation]
    goods_identities: tuple[GoodsIdentity, ...]
    replacements: dict[str, Any]
    cargo: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class CargoObservation:
    document_id: str
    goods: Mapping[str, Any]
    containers: tuple[Mapping[str, Any], ...]
    hs6: tuple[str, ...]
    family: Family
    package_category: str
    quantity: int


def _pin(root: Path, pin: FilePin) -> Path:
    path = root / pin.path
    if path.is_symlink() or not path.is_file() or sha256_file(path) != pin.sha256:
        raise ValueError(f"scenario registry pin mismatch: {pin.path}")
    return path


def _rows(root: Path, pin: FilePin) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in _pin(root, pin).read_text().splitlines()]
    if pin.records is not None and len(rows) != pin.records:
        raise ValueError(f"scenario registry row count mismatch: {pin.path}")
    return rows


def _hs6(goods: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.sub(r"\D", "", code)[:6] for code in goods.get("hsCodes", ())))


def _temperature(containers: Sequence[Mapping[str, Any]]) -> Decimal | None:
    settings = [c["temperatureSetpoint"] for c in containers if "temperatureSetpoint" in c]
    if not settings:
        return None
    values = set()
    for setting in settings:
        value = Decimal(str(setting["value"]))
        if setting["unit"] == "fahrenheit":
            value = (value - 32) * 5 / 9
        elif setting["unit"] != "celsius":
            raise ValueError("unsupported source temperature unit")
        values.add(value)
    if len(values) != 1:
        raise ValueError("one-goods donor has conflicting carrying temperatures")
    return values.pop()


def _ventilation_rate(goods: Mapping[str, Any]) -> str | None:
    """Read an explicitly quantified ventilation instruction, never infer a rate."""
    rates = set()
    for text in goods.get("handlingInstructions", ()):
        if re.search(r"\bVENT(?:IL|ILL)", text, re.I):
            rates.update(
                str(Decimal(m[1]))
                for m in re.finditer(
                    r"([0-9]+(?:\.[0-9]+)?)\s*(?:CBM|M3|M³)\s*(?:/\s*H(?:R)?|PER\s+HOUR)",
                    text,
                    re.I,
                )
            )
    if len(rates) > 1:
        raise ValueError("observed shipment has conflicting ventilation rates")
    return next(iter(rates), None)


def _observed_family(goods: Mapping[str, Any], containers: Sequence[Mapping[str, Any]]) -> Family:
    dg = goods.get("dangerousGoods", ())
    if dg:
        return (
            "dg_vehicle"
            if all(d.get("unNumber") in {"3166", "3171", "3556", "3557", "3558"} for d in dg)
            else "dg_chemical"
        )
    temperature = _temperature(containers)
    if temperature is not None:
        return "frozen" if temperature <= -18 else "chilled"
    if _hs6(goods) and all(
        code[:4] in {"8429", "8430", "8701", "8702", "8703", "8704", "8705"} for code in _hs6(goods)
    ):
        return "vehicle"
    return "ambient"


def _observation(row: Mapping[str, Any], family: Family | None = None) -> CargoObservation | None:
    patch = row["target"]["documentPatch"]
    groups = patch.get("goodsItemDetails", ())
    if len(groups) != 1:
        return None
    goods = groups[0]
    packages = goods.get("numberAndTypeOfPackages", ())
    if len(packages) != 1 or not {"packageQuantity", "typeCategory"} <= packages[0].keys():
        return None
    quantity = packages[0]["packageQuantity"]
    if isinstance(quantity, bool) or int(quantity) != quantity or quantity <= 0:
        return None
    containers = tuple(patch.get("containerInformation", ()))
    return CargoObservation(
        row["documentId"],
        goods,
        containers,
        _hs6(goods),
        family or _observed_family(goods, containers),
        packages[0]["typeCategory"],
        int(quantity),
    )


def _pick[T](values: Sequence[T], stream: DeterministicStream) -> T:
    if not values:
        raise CandidateIncompatibility("no compatible scenario candidates")
    return values[stream.randbelow(len(values))]


def _canonical_measure(goods: Mapping[str, Any], field: str) -> Decimal:
    measure = goods[field]
    factor = Decimal(1) if field == "volume" else _MASS_FACTORS[measure["unit"]]
    if field == "volume" and measure["unit"] != "cubic_metre":
        raise ValueError("unsupported source volume unit")
    return Decimal(str(measure["value"])) * factor


def _apportion(quantity: int, weights: Sequence[int]) -> list[int]:
    """Allocate a positive integer total, keeping every printed row non-empty.

    Largest remainder applies to the amount after one unit per row. Source
    weights preserve approximate proportions, not an unnecessary GCD lattice.
    """

    if quantity < len(weights) or not weights or any(weight <= 0 for weight in weights):
        raise ValueError("positive printed allocations cannot fit the sampled package total")
    total_weight = sum(weights)
    remaining = quantity - len(weights)
    quotient_remainder = [divmod(remaining * weight, total_weight) for weight in weights]
    result = [1 + quotient for quotient, _ in quotient_remainder]
    ranked = sorted(range(len(weights)), key=lambda i: (-quotient_remainder[i][1], i))
    for index in ranked[: quantity - sum(result)]:
        result[index] += 1
    return result


def _hazard_identity_index(registry: LoadedDangerousGoodsRegistry) -> dict[str, dict[str, Any]]:
    """Exact chemical names, never a blanket HS-based hazard classification.

    Conditional identities also block ordinary synthesis: an absent grade
    specification is not evidence that the non-dangerous grade was sampled.
    This eligibility gate does not modify historical source annotations.
    """
    by_un: dict[str, list[DangerousGoodsHmtRecord]] = defaultdict(list)
    names: dict[str, dict[str, Any]] = {}
    for record in registry.hmt_records:
        if record.maritime_eligible:
            by_un[record.un_number].append(record)

    def add(name: str, un_number: str) -> None:
        key = _normalized_chemical_identity(name)
        if not key or un_number not in by_un:
            return
        item = names.setdefault(key, {"names": set(), "unNumbers": set(), "hmtNames": set()})
        item["names"].add(name)
        item["unNumbers"].add(un_number)
        item["hmtNames"].update(r.proper_shipping_name for r in by_un[un_number])

    for records in by_un.values():
        for record in records:
            add(record.proper_shipping_name, record.un_number)
    for link in registry.ecics_links:
        for name in (link.name, link.iupac_description, *link.synonyms):
            if name:
                add(name, link.un_number)
    return {
        key: {field: sorted(values) for field, values in item.items()}
        for key, item in names.items()
    }


def _inventory_shares(
    containers: Sequence[Mapping[str, Any]], goods: Mapping[str, Any], quantity: int
) -> tuple[Decimal, ...]:
    """Per-ID shares, preserving printed placement order only for apportionment."""
    numbers = [row["equipmentIdentifier"] for row in containers]
    if not numbers or len(set(numbers)) != len(numbers):
        raise ValueError("inventory equipment identifiers must be present and unique")
    allocations = goods.get("splitGoodsPlacement", [])
    counts = {row["equipmentIdentifier"]: row.get("packageQuantity") for row in allocations}
    if len(counts) != len(allocations) or set(counts) - set(numbers):
        raise ValueError("inventory placement identifiers must be unique known equipment IDs")
    if any(value is not None for value in counts.values()):
        if (
            set(counts) != set(numbers)
            or any(value is None or value <= 0 for value in counts.values())
            or sum(counts.values()) != quantity
        ):
            raise ValueError("inventory quantity allocations must identify and sum over every row")
        return tuple(Decimal(counts[number]) / quantity for number in numbers)
    return (Decimal(1) / len(numbers),) * len(numbers)


class ScenarioCatalog:
    """Immutable train-fit and registry support; sampling never makes API calls."""

    def __init__(
        self,
        *,
        config: ScenarioSamplingConfig,
        countries: CountryRegistry,
        hs: UkGlobalTariffRegistry,
        phrases: Mapping[str, str],
        ports: Mapping[str, tuple[ScenarioLocation, ...]],
        localities: Mapping[str, tuple[ScenarioLocation, ...]],
        dg: LoadedDangerousGoodsRegistry,
        train_rows: Sequence[Mapping[str, Any]],
        validation_ids: frozenset[str],
        load_profiles: RegistryLoadProfiles | None = None,
    ) -> None:
        self.config, self.countries, self.hs, self.phrases = config, countries, hs, dict(phrases)
        self.load_profiles = load_profiles
        self.equipment_inventory: dict[str, EquipmentInventoryContract] = {}
        self._registry_profile_pool: dict[
            tuple[Any, ...], tuple[tuple[RegistryLoadProfile, tuple[str, ...]], ...]
        ] = {}
        if (load_profiles is None) != (config.load_profiles is None):
            raise ValueError("load-profile configuration and loaded registry differ")
        if load_profiles is not None:
            ids = {p.id for p in load_profiles.profiles}
            categories = {p.package_category for p in load_profiles.profiles}
            if set(config.profile_weights) - ids or set(config.package_weights) - categories:
                raise ValueError("sampling weights reference absent load-profile support")
            if set(config.equipment_weights) - set(_REGISTRY_GP_PAIRS):
                raise ValueError("registry-form equipment weights require supported GP pairs")
            headings = {code[:4] for code in self.phrases}
            for p in load_profiles.profiles:
                if missing := set(p.hs_headings) - headings:
                    raise ValueError(
                        f"profile {p.id} has absent authoritative HS registry headings: "
                        + ", ".join(sorted(missing))
                    )
        elif (
            config.profile_weights
            or config.package_weights
            or config.equipment_weights
            or config.equipment_inventory is not None
        ):
            raise ValueError("cargo distribution weights require an explicit load-profile registry")
        self.ports, self.localities, self.dg = dict(ports), dict(localities), dg
        self.contact_excluded_countries = {}
        for code in sorted(self.ports.keys() & self.localities.keys()):
            try:
                _phone_prefix(code)
            except ValueError as error:
                self.contact_excluded_countries[code] = str(error)
        self.train = {r["documentId"]: r for r in train_rows}
        if (
            len(self.train) != len(train_rows)
            or not self.train
            or self.train.keys() & validation_ids
        ):
            raise ValueError("scenario fitting requires unique, train-only document IDs")
        if config.equipment_inventory is not None and (
            set(config.equipment_inventory.source_ids) - self.train.keys()
        ):
            raise ValueError("configured inventory source scope must be entirely train-fit")
        self.fit_sha256 = sha256_bytes(
            canonical_json_bytes({sid: row["target"] for sid, row in sorted(self.train.items())})
        )
        observed_rows = dict(self.train)
        for sid, override in config.observed_hs_overrides.items():
            if sid not in self.train:
                raise ValueError("synthesis HS observation override is not train-only")
            row = self.train[sid]
            groups = row["target"]["documentPatch"].get("goodsItemDetails", [])
            if (
                len(groups) != 1
                or sha256_bytes(canonical_json_bytes(row["target"]))
                != override.source_target_sha256
                or _hs6(groups[0]) != override.from_hs6
                or any(text not in row["joinedRawText"] for text in override.source_text_literals)
                or any(code not in self.phrases for code in override.to_hs6)
                or {code[:4] for code in override.from_hs6}
                != {code[:4] for code in override.to_hs6}
            ):
                raise ValueError("synthesis HS observation override disagrees with pinned source")
            observed_rows[sid] = deepcopy(row)
            observed_rows[sid]["target"]["documentPatch"]["goodsItemDetails"][0]["hsCodes"] = list(
                override.to_hs6
            )
        self.observations = tuple(
            o for r in observed_rows.values() if (o := _observation(r)) is not None
        )
        self.hazard_identities = _hazard_identity_index(dg)
        self.heading_codes: dict[str, tuple[str, ...]] = {
            heading: tuple(code for code in sorted(self.phrases) if code[:4] == heading)
            for heading in {code[:4] for code in self.phrases}
        }
        self._classification_contexts: dict[str, HSClassificationContext] = {}
        self._hs_description_paths: dict[str, tuple[str, ...]] = {}
        self.limits = capacity_limits(config.transport_capacity)
        self.registry_sha256 = sha256_bytes(canonical_json_bytes(config.model_dump(mode="json")))

        chemical_links: dict[str, list[str]] = defaultdict(list)
        for link in self.dg.ecics_links:
            if (
                link.disposition == "eligible_unique_maritime_hmt"
                and link.hmt_record_id is not None
                and link.hs6 in self.phrases
            ):
                chemical_links[link.hmt_record_id].append(link.hs6)
        chemicals = []
        for record in self.dg.hmt_records:
            try:
                profile(record)
            except ValueError:
                continue
            chemicals.extend((record, code) for code in chemical_links[record.record_id])
        self.chemical_candidates = tuple(chemicals)
        self.thermal_codes = {"frozen": [], "chilled": []}
        self.ambient_codes = set()
        for code in sorted(self.phrases):
            row = self.hs.require_global(code, on_date=self.hs.receipt.snapshot_date)
            thermal = classify_thermal_hs(row)
            if thermal is None:
                if (
                    code[:2] in config.ambient_hs_chapters
                    and _normalized_chemical_identity(self.phrases[code])
                    not in self.hazard_identities
                ):
                    self.ambient_codes.add(code)
            else:
                family = thermal.lower()
                if code[:2] in getattr(config.thermal, family + "_chapters"):
                    self.thermal_codes[family].append(code)
        self._donor_cache: dict[tuple[str, str], tuple[CargoObservation, ...]] = {}
        self._identity_pool_cache: dict[tuple[str, str], tuple[str, ...]] = {}

    def bind_equipment_inventory(self, contracts: Mapping[str, EquipmentInventoryContract]) -> None:
        """Bind the exact configured source certificates before any scenario draws."""
        policy = self.config.equipment_inventory
        if policy is None or set(contracts) != set(policy.source_ids):
            raise ValueError(
                "inventory certificates must match the explicit configured source scope"
            )
        checked = {}
        for sid, contract in contracts.items():
            if not isinstance(contract, EquipmentInventoryContract) or sid not in self.train:
                raise ValueError("inventory certificate must belong to an exact train-fit source")
            row = self.train[sid]
            if (
                contract.source_id != sid
                or contract.source_sha256 != sha256_bytes(row["joinedRawText"].encode())
                or contract.source_target_sha256
                != sha256_bytes(canonical_json_bytes(row["target"]))
            ):
                raise ValueError("inventory certificate differs from the exact train-fit source")
            rows = row["target"]["documentPatch"].get("containerInformation", [])
            if (
                contract.row_count < 2
                or len(contract.groups) < 2
                or any(type(index) is not int for group in contract.groups for index in group)
                or sorted(index for group in contract.groups for index in group)
                != list(range(contract.row_count))
                or any(not group for group in contract.groups)
                or any(container["typeCategory"] != "GENERAL_PURPOSE" for container in rows)
            ):
                raise ValueError("configured inventory source lacks complete independent GP groups")
            contract.validate_rows(rows)
            checked[sid] = contract
        self.equipment_inventory = checked

    def validate_equipment_inventory_support(self, sid: str, cap: SourceCapabilities) -> None:
        """Validate configured source support without drawing cargo or calling a model."""
        policy = self.config.equipment_inventory
        if policy is None or sid not in policy.source_ids or sid not in self.equipment_inventory:
            raise ValueError("inventory support requires a declared, bound source certificate")
        if cap.family != "ambient" or cap.physical_profile != "registry_form":
            raise ValueError(
                "configured inventory source requires the ambient registry-form domain"
            )
        row = self.train[sid]
        contract = self.equipment_inventory[sid]
        if contract.source_sha256 != sha256_bytes(
            row["joinedRawText"].encode()
        ) or contract.source_target_sha256 != sha256_bytes(canonical_json_bytes(row["target"])):
            raise ValueError("inventory support differs from the bound exact source certificate")
        source = _observation(row, cap.family)
        if source is None or _observed_family(source.goods, source.containers) != "ambient":
            raise ValueError(
                "inventory registry forms require complete non-thermal, non-DG ambient goods"
            )
        contract.validate_rows(source.containers)
        profiles = self._eligible_registry_profiles(cap)
        categories = {profile.package_category for profile, _codes in profiles}
        if self.config.package_weights and not categories.intersection(self.config.package_weights):
            raise ValueError("inventory package-category weights have no eligible profile support")
        pairs = tuple(
            pair
            for pair in _REGISTRY_GP_PAIRS
            if not cap.allowed_equipment_pairs or pair in cap.allowed_equipment_pairs
        )
        self._inventory_category_support(contract, pairs)

    @classmethod
    def load(
        cls,
        root: Path,
        config: ScenarioSamplingConfig,
        train_rows: Sequence[Mapping[str, Any]],
        validation_ids: frozenset[str],
    ) -> ScenarioCatalog:
        countries = load_iso_country_registry(
            iso_path=_pin(root, config.countries), iso_sha256=config.countries.sha256
        )
        hs = compile_uk_global_tariff_registry(
            metadata_path=_pin(root, config.hs_metadata),
            report_path=_pin(root, config.hs_report),
            source=load_ukgt_source_pin(_pin(root, config.hs_manifest)),
        )
        phrases = {}
        for row in _rows(root, config.commodity_phrases):
            code = row["hs6"]
            if (
                code in phrases
                or code not in hs.global_codes
                or row["pathSha256"]
                != sha256_bytes(
                    json.dumps(hs.global_description_path(code), ensure_ascii=False).encode()
                )
            ):
                raise ValueError("commercial HS phrase identity/path is inconsistent")
            phrases[code] = row["phrase"]
        if set(phrases) != set(hs.global_codes):
            raise ValueError("commercial HS phrase registry is incomplete")
        whitelist = {row["locode"]: row["country_code"] for row in _rows(root, config.world_ports)}
        ports: dict[str, list[ScenarioLocation]] = defaultdict(list)
        for row in _rows(root, config.locations):
            if row["locode"] not in whitelist or "1" not in row["function_codes"]:
                continue
            code = row["country_code"]
            if whitelist[row["locode"]] != code:
                raise ValueError("UN/LOCODE and WPI countries disagree")
            country = countries.entry(code)
            ports[code].append(
                ScenarioLocation(
                    name=row["name_without_diacritics"].upper(),
                    country_code=code,
                    country=(country.common_name or country.name).upper(),
                    registry="unlocode_wpi",
                    registry_id=row["locode"],
                )
            )
        localities: dict[str, list[ScenarioLocation]] = defaultdict(list)
        for row in _rows(root, config.localities):
            code = row["country_code"]
            if code not in ports:
                continue
            country = countries.entry(code)
            localities[code].append(
                ScenarioLocation(
                    name=row["ascii_name"].upper(),
                    country_code=code,
                    country=(country.common_name or country.name).upper(),
                    registry="geonames",
                    registry_id=str(row["geoname_id"]),
                )
            )
        dg = load_dangerous_goods_registry(
            hmt_path=_pin(root, config.hmt),
            hmt_sha256=config.hmt.sha256,
            ecics_path=_pin(root, config.ecics),
            ecics_sha256=config.ecics.sha256,
        )
        return cls(
            config=config,
            countries=countries,
            hs=hs,
            phrases=phrases,
            ports={k: tuple(v) for k, v in ports.items()},
            localities={k: tuple(v) for k, v in localities.items()},
            dg=dg,
            train_rows=train_rows,
            validation_ids=validation_ids,
            load_profiles=RegistryLoadProfiles.model_validate_json(
                _pin(root, config.load_profiles).read_text()
            )
            if config.load_profiles is not None
            else None,
        )

    def _geography(
        self, row: Mapping[str, Any], capabilities: SourceCapabilities, stream: DeterministicStream
    ) -> tuple[
        ScenarioLocation, ScenarioLocation, dict[str, ScenarioLocation], dict[str, Any], dict
    ]:
        patch = row["target"]["documentPatch"]
        for group in capabilities.party_locality_groups:
            group.validate_source(row, self.countries)
        topology = capabilities.route_topology
        if topology is not None:
            topology.validate_source(patch)
        elif "transshipmentPort" in patch.get("route", {}):
            raise ValueError("transshipment sources require an explicit route topology")
        eligible = tuple(sorted(self.ports.keys() & self.localities.keys()))
        rejected = []
        requirements = {side: {"port": 1, "locality": 0} for side in ("origin", "destination")}
        if topology is not None:
            for spec in topology.nodes.values():
                if spec.country_from in requirements:
                    requirements[spec.country_from][spec.kind] += 1

        def incompatibility(code: str, side: str) -> str | None:
            if capabilities.require_contact_support and code in self.contact_excluded_countries:
                return self.contact_excluded_countries[code]
            for kind, needed in requirements[side].items():
                registry = self.ports if kind == "port" else self.localities
                if len(registry.get(code, ())) < needed:
                    return f"no distinct registry support: {needed} {kind} nodes required in {code}"
            return None

        for side, pin in (
            ("origin", capabilities.origin_country),
            ("destination", capabilities.destination_country),
        ):
            if pin is not None and pin not in eligible:
                raise ValueError(f"pinned geography lacks port/locality support: {pin}")
            if capabilities.require_contact_support and pin in self.contact_excluded_countries:
                raise ValueError(f"pinned geography lacks certified contact support: {pin}")
            if pin is not None and (reason := incompatibility(pin, side)) is not None:
                raise ValueError(f"pinned geography has {reason}")
        origin_country = capabilities.origin_country or _pick(
            tuple(c for c in eligible if c != capabilities.destination_country),
            stream.derive("origin-country"),
        )

        def compatible_country(code: str, side: str, other: str | None) -> str:
            # Keep the first draw unchanged for every already-supported plan.
            # Rejection streams avoid reindexing the complete country catalog.
            for attempt in range(self.config.maximum_candidates):
                reason = incompatibility(code, side)
                if reason is None:
                    return code
                rejected.append({"side": side, "country": code, "reason": reason})
                code = _pick(
                    tuple(c for c in eligible if c != other),
                    stream.derive(f"{side}-country-retry-{attempt + 1}"),
                )
            raise ValueError(f"compatible geography rejection sampling exhausted: {reason}")

        origin_country = compatible_country(
            origin_country, "origin", capabilities.destination_country
        )
        destination_country = capabilities.destination_country or _pick(
            tuple(c for c in eligible if c != origin_country), stream.derive("destination-country")
        )
        destination_country = compatible_country(destination_country, "destination", origin_country)
        if origin_country == destination_country:
            raise ValueError("international scenario has identical endpoint countries")
        origin = _pick(self.ports[origin_country], stream.derive("origin-port"))
        destination = _pick(self.ports[destination_country], stream.derive("destination-port"))
        nodes = {"origin": origin, "destination": destination}
        if topology is not None:
            party_nodes = set(topology.party_nodes.values())
            for name, spec in sorted(topology.nodes.items()):
                countries = (
                    (nodes[spec.country_from].country_code,)
                    if spec.country_from != "independent"
                    else tuple(
                        c
                        for c in sorted(self.ports)
                        if c not in {origin_country, destination_country}
                    )
                )
                registry = self.ports if spec.kind == "port" else self.localities
                if spec.port_ids:
                    known = {p.registry_id for values in self.ports.values() for p in values}
                    if not set(spec.port_ids) <= known:
                        raise ValueError("route node references unknown maritime port IDs")
                used = {(p.registry, p.registry_id) for p in nodes.values()}
                candidates = tuple(
                    p
                    for country in countries
                    for p in registry.get(country, ())
                    if (p.registry, p.registry_id) not in used
                    and (not spec.port_ids or p.registry_id in spec.port_ids)
                    and (
                        name not in party_nodes
                        or (
                            bool(self.localities.get(country))
                            and (
                                not capabilities.require_contact_support
                                or country not in self.contact_excluded_countries
                            )
                        )
                    )
                )
                if not candidates:
                    raise ValueError(f"route node has no distinct registry support: {name}")
                nodes[name] = _pick(candidates, stream.derive(f"route-node:{name}"))
        replacements: dict[str, Any] = {}
        for field, old in patch.get("route", {}).items():
            location = (
                nodes[topology.fields[field]]
                if topology
                else (origin if field in {"placeOfReceipt", "portOfLoading"} else destination)
            )
            for key in ("name", "country"):
                if key in old:
                    replacements[f"documentPatch.route.{field}.{key}"] = getattr(location, key)
        for key in ("name", "country"):
            if key in patch.get("placeOfIssue", {}):
                issue = nodes[topology.issue_node] if topology else origin
                replacements[f"documentPatch.placeOfIssue.{key}"] = getattr(issue, key)
        origin_identifier_paths = {
            f"documentPatch.goodsItemDetails[{index}].origin.identifier"
            for index, goods in enumerate(patch.get("goodsItemDetails", ()))
            if "identifier" in goods.get("origin", {})
        }
        if not capabilities.goods_origin_identifiers.keys() <= origin_identifier_paths:
            raise ValueError("goods-origin identifier contract refers to an absent target path")
        for index, goods in enumerate(patch.get("goodsItemDetails", ())):
            for key in goods.get("origin", {}):
                if key not in {"name", "identifier"}:
                    raise ValueError(
                        f"goods origin has an unsupported country representation: {key}"
                    )
                path = f"documentPatch.goodsItemDetails[{index}].origin.{key}"
                if key == "identifier":
                    presentation = capabilities.goods_origin_identifiers.get(
                        path, GoodsOriginIdentifier(kind="country_alpha2")
                    )
                    presentation.validate_source(goods["origin"][key])
                    location = nodes[presentation.node]
                    replacements[path] = presentation.render(
                        country_code=location.country_code,
                        registry=location.registry,
                        registry_id=location.registry_id,
                    )
                else:
                    replacements[path] = origin.country
        freight = patch.get("freight", {})
        if "paymentPlace" in freight:
            path = "documentPatch.freight.paymentPlace"
            side = (
                topology.location_nodes.get(path) if topology is not None else None
            ) or capabilities.location_sides.get(path)
            if side is None:
                side = {"prepaid": "origin", "collect": "destination"}.get(
                    freight.get("paymentArrangement")
                )
            if side is None:
                raise ValueError("freight payment place requires an explicit endpoint side")
            location = nodes[side]
            for key in ("name", "country"):
                if key in freight["paymentPlace"]:
                    replacements[f"{path}.{key}"] = getattr(location, key)
        parties: dict[str, ScenarioLocation] = {}
        party_country_assignments: dict[str, str] = {}
        repeated_parties: dict[tuple[tuple[str, ...], tuple[str, ...]], ScenarioLocation] = {}
        for role, values in patch.get("parties", {}).items():
            entries = values if isinstance(values, list) else [values]
            for i, party in enumerate(entries):
                path = f"documentPatch.parties.{role}" + (
                    f"[{i}]" if isinstance(values, list) else ""
                )
                if party.get("sameAs") is not None:
                    continue
                node = (
                    topology.party_nodes[path]
                    if topology is not None and path in topology.party_nodes
                    else capabilities.party_sides.get(
                        path, "origin" if role == "shipper" else "destination"
                    )
                )
                country = nodes[node].country_code
                party_country_assignments[path] = country
                location = _pick(self.localities[country], stream.derive(path))
                # Label punctuation and casing are not separate office identities.
                # Preserve the ordered tokens (including all address numbers),
                # so different sites are never joined merely by company name.
                signature = _party_signature(party)
                if all(signature):
                    if signature in repeated_parties:
                        if (
                            topology is not None
                            and topology.party_nodes
                            and repeated_parties[signature].country_code != country
                        ):
                            raise ValueError(
                                "repeated source party has conflicting route countries"
                            )
                        location = repeated_parties[signature]
                    else:
                        repeated_parties[signature] = location
                parties[path] = location
                if "country" in party:
                    replacements[f"{path}.country"] = location.country
        if capabilities.party_locality_groups:
            parties = share_party_localities(
                capabilities.party_locality_groups,
                patch,
                parties,
                party_country_assignments,
                _party_signature,
            )
        # A symbolic notify party may still print its own contact overrides.
        # These use the referenced party's geography, not a separately sampled
        # identity. Resolve after concrete parties so dict order cannot matter.
        for i, party in enumerate(patch.get("parties", {}).get("notifyParties", [])):
            if party.get("sameAs") is None or not party.get("contactDetails"):
                continue
            reference = f"documentPatch.parties.{party['sameAs']}"
            if reference not in parties:
                raise ValueError("notify contact reference lacks a sampled concrete party")
            parties[f"documentPatch.parties.notifyParties[{i}]"] = parties[reference]
        return (
            origin,
            destination,
            parties,
            replacements,
            {
                "contactSupportRequired": capabilities.require_contact_support,
                "excludedCountries": dict(self.contact_excluded_countries)
                if capabilities.require_contact_support
                else {},
                "rejectedDraws": rejected,
                **(
                    {
                        "partyLocalityGroups": [
                            sorted(group.members) for group in capabilities.party_locality_groups
                        ]
                    }
                    if capabilities.party_locality_groups
                    else {}
                ),
                **(
                    {
                        "routeNodes": {k: v.model_dump() for k, v in nodes.items()},
                        "routeFields": topology.fields,
                        **(
                            {"routePartyNodes": topology.party_nodes}
                            if topology.party_nodes
                            else {}
                        ),
                        **(
                            {"locationNodes": topology.location_nodes}
                            if topology.location_nodes
                            else {}
                        ),
                        "routePolicy": "synthetic_registry_topology_not_carrier_service",
                    }
                    if topology
                    else {}
                ),
            },
        )

    def _donors(
        self, source: CargoObservation, cap: SourceCapabilities
    ) -> tuple[CargoObservation, ...]:
        cache_key = (source.document_id, cap.model_dump_json())
        if cap.physical_profile != "reviewed_whole_units" and cache_key in self._donor_cache:
            return self._donor_cache[cache_key]
        candidates = []
        if cap.physical_profile == "reviewed_whole_units":
            reviewed = []
            for sid, pin in sorted(cap.reviewed_whole_unit_donors.items()):
                row = self.train.get(sid)
                donor = _observation(row, cap.family) if row is not None else None
                if (
                    row is None
                    or sha256_bytes(canonical_json_bytes(row["target"])) != pin.target_sha256
                    or any(text not in row["joinedRawText"] for text in pin.source_text_literals)
                    or donor is None
                    or donor.quantity != pin.whole_unit_count
                    or not donor.hs6
                    or len(donor.hs6) != cap.identity_count
                ):
                    raise ValueError(
                        "reviewed whole-unit donor disagrees with pinned training evidence"
                    )
                reviewed.append(donor)
            observations = tuple(reviewed)
        else:
            observations = (
                (source,)
                if cap.physical_profile in {"source_bundle", "source_whole_units"}
                else (*self.observations, source)
            )
        required = {f for f in _MEASURES if f in source.goods} | set(cap.required_private_measures)
        for donor in observations:
            thermal = cap.family in {"frozen", "chilled"}
            family_matches = (
                donor.family in {"frozen", "chilled"} if thermal else donor.family == cap.family
            )
            if not family_matches or not required <= donor.goods.keys():
                continue
            if bool(donor.containers) != bool(source.containers):
                continue
            if cap.family in {"vehicle", "dg_vehicle"} and re.match(
                r"(?:USED\s+)?(?:SPARE\s+)?PARTS\b", donor.goods.get("description", ""), re.I
            ):
                continue
            if (
                cap.allowed_package_categories
                and donor.package_category not in cap.allowed_package_categories
            ):
                continue
            if (
                not donor.hs6
                and cap.family not in {"vehicle", "dg_vehicle", "dg_chemical"}
                and not (
                    cap.family == "ambient"
                    and cap.physical_profile == "source_bundle"
                    and cap.allowed_hs_headings
                )
            ):
                continue
            if donor.hs6 and any(code not in self.phrases for code in donor.hs6):
                continue
            if (
                cap.allowed_hs_headings
                and donor.hs6
                and not all(code[:4] in cap.allowed_hs_headings for code in donor.hs6)
            ):
                continue
            if thermal:
                if (
                    _temperature(donor.containers) is None
                    or not donor.hs6
                    or any(
                        code[:2] not in {"02", "03", "07", "08", "16", "20"} for code in donor.hs6
                    )
                    or any(c.get("typeCategory") != "REFRIGERATED" for c in donor.containers)
                ):
                    continue
                # Registry meat/fish/frozen foods can share an observed food
                # carton/box load, but not a pharmaceutical or unreviewed crate.
                # Produce extensions keep their own exact physical bundle.
                if donor.package_category not in self.config.thermal.package_categories and (
                    donor.family != cap.family or not _ventilation_rate(donor.goods)
                ):
                    continue
            if donor.containers and any(
                not {"sizeCategory", "typeCategory"} <= c.keys() for c in donor.containers
            ):
                continue
            pairs = {(c.get("sizeCategory"), c.get("typeCategory")) for c in donor.containers}
            if len(pairs) > 1:
                continue  # One shared printed equipment declaration is one pair.
            if cap.allowed_equipment_pairs and any(
                f"{a}|{b}" not in cap.allowed_equipment_pairs for a, b in pairs
            ):
                continue
            candidates.append(donor)
        result = tuple({d.document_id: d for d in candidates}.values())
        self._donor_cache[cache_key] = result
        return result

    def _description_path(self, code: str) -> tuple[str, ...]:
        if code not in self._hs_description_paths:
            self._hs_description_paths[code] = self.hs.global_description_path(code)
        return self._hs_description_paths[code]

    def _classification(self, code: str) -> HSClassificationContext:
        if code not in self._classification_contexts:
            row = self.hs.require_global(code, on_date=self.hs.receipt.snapshot_date)
            children = {}
            for commodity in self.hs.uk_candidates(code, on_date=self.hs.receipt.snapshot_date):
                nodes = commodity.description_path
                indices = [
                    i
                    for i, node in enumerate(nodes)
                    if node.code == code + "0000" and node.suffix == "80"
                ]
                if len(indices) != 1:
                    raise ValueError("HS example lacks a unique global ancestor")
                index = indices[0] + 1
                if index < len(nodes):
                    node = nodes[index]
                    children[node.code, node.description] = HSChildExample(
                        code=node.code,
                        description=node.description,
                    )
            self._classification_contexts[code] = HSClassificationContext(
                authority=row.description_authority,
                chapter_code=row.chapter_code,
                chapter_description=row.chapter_description,
                heading_code=row.heading_code,
                heading_description=row.heading_description,
                description_path=self._description_path(code),
                same_heading_alternatives=tuple(
                    HSAlternative(hs6=sibling, description_path=self._description_path(sibling)[1:])
                    for sibling in self.heading_codes[code[:4]]
                    if sibling != code
                ),
                immediate_national_children=tuple(children[key] for key in sorted(children)),
            )
        return self._classification_contexts[code]

    def _identities(
        self,
        source: CargoObservation,
        donor: CargoObservation,
        cap: SourceCapabilities,
        stream: DeterministicStream,
    ) -> tuple[tuple[GoodsIdentity, ...], DangerousGoodsHmtRecord | None]:
        dg_record = None
        if cap.family == "dg_chemical":
            eligible = [
                (record, code)
                for record, code in self.chemical_candidates
                if (not cap.allowed_hs_headings or code[:4] in cap.allowed_hs_headings)
                and (not cap.allowed_hs_codes or code in cap.allowed_hs_codes)
            ]
            dg_record, code = _pick(eligible, stream.derive("chemical"))
            return (
                GoodsIdentity(
                    hs6=code,
                    phrase=dg_record.proper_shipping_name.upper(),
                    classification=self._classification(code),
                ),
            ), dg_record
        if cap.family == "dg_vehicle":
            un_numbers = {value["unNumber"] for value in source.goods["dangerousGoods"]}
            eligible = [
                r
                for r in self.dg.hmt_records
                if r.un_number in un_numbers
                and r.maritime_eligible
                and not r.technical_name_required
                and "flammable liquid" in r.proper_shipping_name.casefold()
            ]
            dg_record = _pick(eligible, stream.derive("vehicle-dg"))
        available = self._registry_pool(donor, cap)
        if cap.family == "dg_vehicle":
            # The pilot's vehicle declaration is liquid-fuel UN3166. An HS
            # heading also contains electric/gas cars; that broad heading alone
            # cannot authorize pairing them with this selected regulatory tuple.
            available = [
                code
                for code in available
                if re.search(r"\b(?:diesel|petrol|spark[ -]ignition)\b", self.phrases[code], re.I)
                and not re.search(r"\b(?:electric|hybrid)\b", self.phrases[code], re.I)
            ]
        selected = []
        for i in range(cap.identity_count):
            pool = [code for code in available if code not in selected]
            # Prefer a different identity, but record unavoidable singleton
            # domains honestly rather than inventing an HS classification.
            new = [code for code in pool if code not in source.hs6]
            code = _pick(new or pool, stream.derive(f"identity-{i}"))
            if cap.family == "ambient" and (
                hazard := self.hazard_identities.get(
                    _normalized_chemical_identity(self.phrases[code])
                )
            ):
                raise CandidateIncompatibility(
                    "ordinary source cannot emit exact regulated or grade-conditional chemical: "
                    f"{self.phrases[code]} / UN {','.join(hazard['unNumbers'])}"
                )
            selected.append(code)
        return tuple(
            GoodsIdentity(
                hs6=code,
                phrase=self.phrases[code].upper(),
                observed_hs6=next((c for c in donor.hs6 if c[:4] == code[:4]), None),
                classification=self._classification(code),
            )
            for code in selected
        ), dg_record

    def _registry_pool(self, donor: CargoObservation, cap: SourceCapabilities) -> tuple[str, ...]:
        key = (donor.document_id, cap.model_dump_json())
        if key in self._identity_pool_cache:
            return self._identity_pool_cache[key]
        headings = cap.allowed_hs_headings or tuple(dict.fromkeys(c[:4] for c in donor.hs6))
        if cap.family in {"frozen", "chilled"}:
            pool = (
                list(self.thermal_codes[cap.family])
                if donor.package_category in self.config.thermal.package_categories
                else []
            )
            # Extra fresh produce is supported by exact observed carriage data;
            # it does not narrow the separately available registry food domain.
            if donor.family == cap.family and all(c[:2] in {"07", "08"} for c in donor.hs6):
                pool.extend(c for c in donor.hs6 if c not in pool)
        else:
            pool = [c for h in headings for c in self.heading_codes.get(h, ())]
            if cap.family == "ambient":
                pool = [c for c in pool if c in self.ambient_codes]
        if cap.allowed_hs_headings:
            pool = [c for c in pool if c[:4] in cap.allowed_hs_headings]
        if cap.allowed_hs_codes:
            if any(c not in self.phrases for c in cap.allowed_hs_codes):
                raise ValueError("explicit HS code domain contains an unregistered identity")
            pool = [c for c in pool if c in cap.allowed_hs_codes]
        result = tuple(sorted(set(pool)))
        self._identity_pool_cache[key] = result
        return result

    def _thermal_settings(self, donor, cap, identities, stream):
        """Choose settings from sampled identity authority, never unrelated source prose."""
        if cap.family not in {"frozen", "chilled"}:
            return None
        codes = [i.hs6 for i in identities]
        empirical = [c for c in codes if c not in self.thermal_codes[cap.family]]
        if empirical:
            if len(codes) != 1 or codes[0] not in donor.hs6 or donor.family != cap.family:
                raise CandidateIncompatibility(
                    "empirical thermal extension requires its exact commodity bundle"
                )
            temperature = _temperature(donor.containers)
            if temperature is None:
                raise ValueError("observed produce identity lacks a carrying temperature")
            ventilation = _ventilation_rate(donor.goods)
            basis = "observed_produce_extension"
        else:
            temperature = getattr(self.config.thermal, cap.family).sample(stream.derive("setpoint"))
            ventilation = "0"
            basis = "hs_registry_food_profile"
        return {
            "basis": basis,
            "profile": cap.family,
            "sampledHS6": codes,
            "commodityDescriptions": [i.phrase for i in identities],
            "productForm": (
                "preserve the observed fresh produce form"
                if empirical
                else "frozen food"
                if cap.family == "frozen"
                else "chilled non-live food; choose this form over HS live/fresh alternatives"
            ),
            "temperatureCelsius": float(temperature),
            "ventilationCbmPerHour": ventilation,
            **({"observedDescription": donor.goods["description"]} if empirical else {}),
        }

    def _cargo(
        self,
        source: CargoObservation,
        donor: CargoObservation,
        cap: SourceCapabilities,
        identities: tuple[GoodsIdentity, ...],
        dg_record: DangerousGoodsHmtRecord | None,
        stream: DeterministicStream,
        *,
        quantity_override: int | None = None,
        inventory_pairs: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        goods = deepcopy(dict(source.goods))
        thermal_context = self._thermal_settings(donor, cap, identities, stream)
        if (
            thermal_context
            and thermal_context["ventilationCbmPerHour"] is None
            and any(
                re.search(r"\bVENT(?:IL|ILL)", t, re.I)
                for t in source.goods.get("handlingInstructions", ())
            )
        ):
            raise CandidateIncompatibility(
                "printed ventilation requires commodity-supported settings"
            )
        containers = [deepcopy(dict(c)) for c in source.containers]
        if inventory_pairs is not None and (
            cap.physical_profile != "registry_form"
            or cap.family != "ambient"
            or cap.equipment_mix is not None
            or len(inventory_pairs) != len(containers)
            or any(pair not in _REGISTRY_GP_PAIRS for pair in inventory_pairs)
        ):
            raise ValueError("per-ID inventory requires complete ambient registry-form GP rows")
        allocations = goods.get("splitGoodsPlacement", [])
        quantities = [int(a["packageQuantity"]) for a in allocations if "packageQuantity" in a]
        if quantities and (
            len(quantities) != len(allocations) or sum(quantities) != source.quantity
        ):
            raise ValueError("source printed package allocations are not a complete exact sum")
        multiple = cap.quantity_multiple
        scale = (
            self.config.quantity_scale_min
            + (self.config.quantity_scale_max - self.config.quantity_scale_min)
            * stream.derive("load-scale").randbelow(self.config.scale_steps + 1)
            / self.config.scale_steps
        )
        target_count = len(containers) or 1
        donor_count = len(donor.containers) or 1
        nominal = Decimal(donor.quantity) * target_count / donor_count * scale
        quantity = (
            quantity_override
            or cap.fixed_package_quantity
            or max(
                multiple,
                int((nominal / multiple).to_integral_value(rounding=ROUND_HALF_UP)) * multiple,
            )
        )
        if quantities and quantity < len(quantities) and cap.fixed_package_quantity is None:
            quantity = ((len(quantities) + multiple - 1) // multiple) * multiple
        if quantity % multiple:
            raise ValueError("fixed quantity conflicts with exact printed allocation topology")
        package_type = (
            source.package_category
            if cap.physical_profile == "reviewed_whole_units"
            else donor.package_category
        )
        if cap.family == "dg_chemical":
            package_type = _pick(
                cap.allowed_package_categories or ("PACKAGE_DRUM", "PACKAGE_CARTON", "PACKAGE_BOX"),
                stream.derive("chemical-packaging"),
            )
        goods["numberAndTypeOfPackages"][0].update(
            packageQuantity=quantity, typeCategory=package_type
        )
        if quantities:
            for allocation, count in zip(
                allocations, _apportion(quantity, quantities), strict=True
            ):
                allocation["packageQuantity"] = count
        factor = Decimal(quantity) / donor.quantity
        for field in _MEASURES:
            if field not in goods:
                continue
            canonical = _canonical_measure(donor.goods, field) * factor
            unit_factor = Decimal(1) if field == "volume" else _MASS_FACTORS[goods[field]["unit"]]
            # Use the source's printed scalar precision. The text compiler may
            # impose a coarser common lattice for repeated row-level totals.
            old = Decimal(str(goods[field]["value"]))
            quantum = Decimal(1).scaleb(min(old.as_tuple().exponent, 0))
            goods[field]["value"] = float(
                (canonical / unit_factor).quantize(quantum, rounding=ROUND_HALF_UP)
            )
            if goods[field]["value"] <= 0:
                raise CandidateIncompatibility(
                    "sampled physical measure rounds to zero at source precision"
                )
        # Missing public labels do not mean that a source template has no
        # printed mass/cube. Carry those facts from the same sampled donor,
        # not from an unrelated source commodity's per-package density.
        private_measures = {
            field: _canonical_measure(donor.goods, field) * factor
            for field in cap.required_private_measures
            if field not in goods
        }
        if any(value <= 0 for value in private_measures.values()):
            raise CandidateIncompatibility("sampled private physical measure must be positive")
        all_measures = {
            field: _canonical_measure(goods, field) for field in _MEASURES if field in goods
        } | private_measures
        if (
            "netWeight" in all_measures
            and "grossWeight" in all_measures
            and all_measures["netWeight"] > all_measures["grossWeight"]
        ):
            raise CandidateIncompatibility("sampled net weight exceeds gross weight")
        if "description" in goods:
            goods["description"] = "; ".join(i.phrase for i in identities)
        if "hsCodes" in goods:
            goods["hsCodes"] = [i.hs6 for i in identities]
        if dg_record is not None:
            for entry in goods["dangerousGoods"]:
                for key, value in (
                    ("unNumber", dg_record.un_number),
                    ("hazardCategory", dg_record.hazard_category),
                    ("packingGroupCategory", dg_record.packing_group_category),
                ):
                    if key in entry:
                        if value is None:
                            raise CandidateIncompatibility(
                                f"sampled DG tuple lacks printed source field {key}"
                            )
                        entry[key] = value
                if "subsidiaryHazardCategories" in entry:
                    if not dg_record.subsidiary_hazard_categories:
                        raise CandidateIncompatibility(
                            "sampled DG tuple lacks the source's subsidiary-hazard slot"
                        )
                    entry["subsidiaryHazardCategories"] = list(
                        dg_record.subsidiary_hazard_categories
                    )
        for index, container in enumerate(containers):
            sampled = donor.containers[0]
            if inventory_pairs is not None:
                size, kind = inventory_pairs[index].split("|")
                sampled = {"sizeCategory": size, "typeCategory": kind}
            for field in ("sizeCategory", "typeCategory"):
                if field in container:
                    container[field] = sampled[field]
            if "temperatureSetpoint" in container:
                if thermal_context is None:
                    raise CandidateIncompatibility(
                        "sampled cold-chain bundle has no carrying setpoint"
                    )
                temperature = Decimal(str(thermal_context["temperatureCelsius"]))
                value = (
                    temperature
                    if container["temperatureSetpoint"]["unit"] == "celsius"
                    else temperature * 9 / 5 + 32
                )
                container["temperatureSetpoint"]["value"] = float(value)
            if "verifiedGrossMass" in container:
                raise ValueError(
                    "verified gross mass requires an explicit tare/row-weight contract"
                )
        inventory_capacities = None
        if containers:
            if classify_equipment(donor.containers[0]) == "unsupported_reefer":
                raise CandidateIncompatibility(
                    "reefer size/type pair has no source-supported capacity contract"
                )
            capacity = equipment_capacity(donor.containers[0], self.limits)
            shares = (
                [Decimal(a["packageQuantity"]) / quantity for a in allocations]
                if quantities
                else [Decimal(1) / len(containers)] * len(containers)
            )
            peak = max(shares)
            if inventory_pairs is not None:
                shares = _inventory_shares(containers, goods, quantity)
                inventory_capacities = tuple(
                    equipment_capacity(container, self.limits) for container in containers
                )
                for share, row_capacity in zip(shares, inventory_capacities, strict=True):
                    if (
                        "grossWeight" in all_measures
                        and all_measures["grossWeight"] * share > row_capacity.payload_kg
                    ):
                        raise CandidateIncompatibility(
                            "sampled row exceeds assigned inventory payload capacity"
                        )
                    if (
                        "volume" in all_measures
                        and row_capacity.volume_m3 is not None
                        and all_measures["volume"] * share > row_capacity.volume_m3
                    ):
                        raise CandidateIncompatibility(
                            "sampled row exceeds assigned inventory enclosed volume"
                        )
            if cap.equipment_mix is None and (
                inventory_pairs is None
                and "grossWeight" in all_measures
                and all_measures["grossWeight"] * peak > capacity.payload_kg
            ):
                raise CandidateIncompatibility("sampled row exceeds equipment payload capacity")
            if cap.equipment_mix is None and (
                inventory_pairs is None
                and "volume" in all_measures
                and capacity.volume_m3 is not None
                and all_measures["volume"] * peak > capacity.volume_m3
            ):
                raise CandidateIncompatibility("sampled row exceeds equipment enclosed volume")
        if cap.family == "dg_chemical":
            if "grossWeight" in all_measures and all_measures["grossWeight"] / quantity > 400:
                raise CandidateIncompatibility(
                    "chemical package mass exceeds non-bulk synthesis envelope"
                )
            if "volume" in all_measures and all_measures["volume"] / quantity > Decimal(".45"):
                raise CandidateIncompatibility(
                    "chemical package cube exceeds non-bulk synthesis envelope"
                )
        mix_capacities = None
        mix_assignment = None
        if cap.equipment_mix is not None:
            mix_capacities = cap.equipment_mix.capacities(self.limits)
            loads = [
                {field: value * share for field, value in all_measures.items()} for share in shares
            ]
            mix_assignment = capacity_assignment(loads, mix_capacities)
            if mix_assignment is None:
                raise CandidateIncompatibility("sampled loads cannot fit the printed equipment mix")
        by_number = {a["equipmentIdentifier"]: a for a in allocations}
        physical_rows = []
        for index, container in enumerate(containers):
            number = container["equipmentIdentifier"]
            allocation = by_number.get(number, {})
            numerator = allocation.get("packageQuantity", 1)
            denominator = quantity if "packageQuantity" in allocation else len(containers)
            row_capacity = (
                inventory_capacities[index] if inventory_capacities is not None else capacity
            )
            physical_rows.append(
                {
                    "equipmentIdentifier": number,
                    "quantity": allocation.get("packageQuantity"),
                    "shareNumerator": numerator,
                    "shareDenominator": denominator,
                    "capacity": mix_capacities[mix_assignment[index]]
                    if mix_capacities is not None
                    else {
                        "payloadKg": str(row_capacity.payload_kg),
                        "volumeM3": str(row_capacity.volume_m3)
                        if row_capacity.volume_m3 is not None
                        else None,
                    },
                    "measures": {
                        field: {
                            "value": str(
                                Decimal(str(goods[field]["value"])) * numerator / denominator
                            ),
                            "unit": goods[field]["unit"],
                            "total": str(goods[field]["value"]),
                        }
                        for field in _MEASURES
                        if field in goods
                    },
                }
            )
        return {"goodsItemDetails": [goods], "containerInformation": containers}, {
            "donorDocumentId": donor.document_id,
            "physicalProfile": cap.physical_profile,
            **(
                {
                    "reviewedWholeUnitDonor": cap.reviewed_whole_unit_donors[
                        donor.document_id
                    ].model_dump(mode="json"),
                    "packageInterpretation": (
                        "source explicit packaging; donor measures per whole vehicle"
                    ),
                }
                if cap.physical_profile == "reviewed_whole_units"
                else {}
            ),
            "donorHS6": list(donor.hs6),
            "loadScale": str(scale),
            "goodsSamplingBasis": "registry_identity_conditioned_on_package_equipment_support",
            "ventilationCbmPerHour": thermal_context["ventilationCbmPerHour"]
            if thermal_context
            else None,
            **({"thermalCommodityContext": thermal_context} if thermal_context is not None else {}),
            "quantityMultiple": multiple,
            "physicalRows": physical_rows,
            **(
                {
                    "equipmentMix": {
                        "certificate": cap.equipment_mix.model_dump(mode="json"),
                        "capacities": mix_capacities,
                    }
                }
                if cap.equipment_mix is not None
                else {}
            ),
            **(
                {
                    "privatePhysicalMeasures": {
                        field: {
                            "value": str(value),
                            "unit": "cubic_metre" if field == "volume" else "kilogram",
                            "basis": "same_train_donor_per_package_measure",
                        }
                        for field, value in private_measures.items()
                    }
                }
                if private_measures
                else {}
            ),
            "dgHmtRecordId": dg_record.record_id if dg_record else None,
            "dgPrintedFacts": (
                {
                    "properShippingName": dg_record.proper_shipping_name,
                    "unNumber": dg_record.un_number,
                    "class": dg_record.exact_hazard_class,
                    "packingGroup": dg_record.packing_group_code,
                    "subsidiaryHazards": list(dg_record.exact_subsidiary_hazards),
                }
                if dg_record
                else None
            ),
            "sameHS6AsSource": tuple(i.hs6 for i in identities) == source.hs6,
        }

    def _eligible_registry_profiles(
        self, cap: SourceCapabilities
    ) -> tuple[tuple[RegistryLoadProfile, tuple[str, ...]], ...]:
        if self.load_profiles is None:
            raise ValueError("registry-form sampling requires a pinned load-profile catalog")
        key = (
            cap.identity_count,
            cap.allowed_package_categories,
            cap.allowed_hs_headings,
            cap.allowed_hs_codes,
        )
        if key not in self._registry_profile_pool:
            eligible = []
            for profile in self.load_profiles.profiles:
                if (
                    cap.allowed_package_categories
                    and profile.package_category not in cap.allowed_package_categories
                ):
                    continue
                codes = tuple(
                    code
                    for heading in profile.hs_headings
                    for code in self.heading_codes[heading]
                    if (not cap.allowed_hs_headings or code[:4] in cap.allowed_hs_headings)
                    and (not cap.allowed_hs_codes or code in cap.allowed_hs_codes)
                    and _normalized_chemical_identity(self.phrases[code])
                    not in self.hazard_identities
                )
                if len(codes) >= cap.identity_count and (
                    not self.config.profile_weights or profile.id in self.config.profile_weights
                ):
                    eligible.append((profile, codes))
            if not eligible:
                raise ValueError("source domains have no reviewed registry-form support")
            self._registry_profile_pool[key] = tuple(eligible)
        return self._registry_profile_pool[key]

    def _inventory_category_support(
        self, contract: EquipmentInventoryContract, pairs: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        weights = self.config.equipment_weights
        eligible_pairs = tuple(pair for pair in pairs if not weights or pair in weights)
        policy = self.config.equipment_inventory
        if policy is None:
            raise ValueError("inventory sampling requires its explicit distribution policy")
        eligible_counts = tuple(
            count
            for count in sorted(policy.category_count_weights)
            if count <= min(len(contract.groups), len(eligible_pairs))
        )
        if not eligible_counts:
            raise ValueError("inventory category-count distribution has no feasible source support")
        return eligible_pairs, eligible_counts

    def _draw_equipment_inventory(
        self,
        contract: EquipmentInventoryContract,
        mode: InventoryMode,
        pairs: tuple[str, ...],
        stream: DeterministicStream,
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        weights = self.config.equipment_weights
        eligible_pairs, eligible_counts = self._inventory_category_support(contract, pairs)
        policy = cast(EquipmentInventorySampling, self.config.equipment_inventory)
        if mode == "uniform":
            requested = (
                weighted_choice(pairs, weights, lambda value: value, stream.derive("equipment")),
            )
        else:
            count = weighted_choice(
                eligible_counts,
                {str(k): v for k, v in policy.category_count_weights.items()},
                str,
                stream.derive("inventory-category-count"),
            )
            selected = []
            for index in range(count):
                selected.append(
                    weighted_choice(
                        tuple(pair for pair in eligible_pairs if pair not in selected),
                        weights,
                        lambda value: value,
                        stream.derive(f"inventory-category-{index}"),
                    )
                )
            requested = tuple(selected)
        order = list(range(len(contract.groups)))
        for index in range(len(order) - 1, 0, -1):
            other = stream.derive(f"inventory-group-order-{index}").randbelow(index + 1)
            order[index], order[other] = order[other], order[index]
        assigned: list[str | None] = [None] * contract.row_count
        for position, group_index in enumerate(order):
            pair = (
                requested[position]
                if position < len(requested)
                else weighted_choice(
                    requested,
                    weights,
                    lambda value: value,
                    stream.derive(f"inventory-group-category-{position}"),
                )
            )
            for row in contract.groups[group_index]:
                assigned[row] = pair
        if any(pair is None for pair in assigned) or set(assigned) != set(requested):
            raise ValueError("inventory assignment did not cover its exact requested categories")
        result = cast(tuple[str, ...], tuple(assigned))
        return result, {
            "mode": mode,
            "ownerGroups": [list(group) for group in contract.groups],
            "sizes": [pair.split("|")[0] for pair in result],
            "requestedSizes": [pair.split("|")[0] for pair in requested],
            "eligiblePairs": list(eligible_pairs),
            "eligibleCategoryCounts": list(eligible_counts),
            "sourceSha256": contract.source_sha256,
            "sourceTargetSha256": contract.source_target_sha256,
        }

    def _registry_cargo(
        self,
        source: CargoObservation,
        cap: SourceCapabilities,
        stream: DeterministicStream,
        *,
        inventory_contract: EquipmentInventoryContract | None = None,
        inventory_mode: InventoryMode | None = None,
    ):
        """Select form, declared unit and equipment without an observed donor.

        Source counts and row ownership remain fixed. The selected profile is
        the numeric authority; existing physical compilation enforces print
        precision, allocation sums, private totals, tares and row capacities.
        """
        if self.load_profiles is None:
            raise ValueError("registry-form sampling requires a pinned load-profile catalog")
        if _observed_family(source.goods, source.containers) != "ambient":
            raise ValueError("registry-form source has non-ambient goods or thermal/DG settings")
        if not source.containers or any(
            c.get("typeCategory") != "GENERAL_PURPOSE" or not c.get("sizeCategory")
            for c in source.containers
        ):
            raise ValueError("registry-form source requires completely classified GP containers")
        if (
            inventory_contract is None
            and len({(c["sizeCategory"], c["typeCategory"]) for c in source.containers}) != 1
        ):
            raise ValueError("mixed-equipment source requires its separate inventory contract")
        eligible = self._eligible_registry_profiles(cap)
        category = weighted_choice(
            sorted({p.package_category for p, _ in eligible}),
            self.config.package_weights,
            lambda v: v,
            stream.derive("package-category"),
        )
        selected, codes = weighted_choice(
            [(p, codes) for p, codes in eligible if p.package_category == category],
            self.config.profile_weights,
            lambda v: v[0].id,
            stream.derive("form"),
        )
        pairs = tuple(
            p
            for p in _REGISTRY_GP_PAIRS
            if not cap.allowed_equipment_pairs or p in cap.allowed_equipment_pairs
        )
        inventory_pairs = None
        inventory_receipt = None
        if inventory_contract is None:
            pair = weighted_choice(
                pairs, self.config.equipment_weights, lambda v: v, stream.derive("equipment")
            )
        else:
            if inventory_mode is None:
                raise ValueError("inventory source has no explicit per-draw mode")
            inventory_contract.validate_rows(source.containers)
            inventory_pairs, inventory_receipt = self._draw_equipment_inventory(
                inventory_contract, inventory_mode, pairs, stream
            )
            pair = inventory_pairs[0]
        size, kind = pair.split("|")
        selected_codes = []
        for i in range(cap.identity_count):
            code = _pick(
                tuple(c for c in codes if c not in selected_codes), stream.derive(f"identity-{i}")
            )
            selected_codes.append(code)
        identities = tuple(
            GoodsIdentity(
                hs6=code,
                phrase=self.phrases[code].upper(),
                phrase_role="classification_context",
                classification=self._classification(code),
            )
            for code in selected_codes
        )
        quantity = cap.fixed_package_quantity
        if inventory_pairs is not None:
            shares = _inventory_shares(source.containers, source.goods, source.quantity)
            upper = selected.quantity_max
            for share, assigned_pair in zip(shares, inventory_pairs, strict=True):
                assigned_size, assigned_kind = assigned_pair.split("|")
                capacity = equipment_capacity(
                    {"sizeCategory": assigned_size, "typeCategory": assigned_kind}, self.limits
                )
                upper = min(upper, int(capacity.payload_kg / (selected.unit_gross_kg * share)))
                if capacity.volume_m3 is not None:
                    upper = min(upper, int(capacity.volume_m3 / (selected.unit_volume_m3 * share)))
            multiple = cap.quantity_multiple
            required = len(source.goods.get("splitGoodsPlacement", [])) or 1
            lo = (max(selected.quantity_min, required) + multiple - 1) // multiple
            hi = upper // multiple
            if lo > hi:
                raise CandidateIncompatibility(
                    "profile quantity interval cannot fit every assigned row"
                )
            if quantity is not None:
                if quantity % multiple or not lo * multiple <= quantity <= hi * multiple:
                    raise CandidateIncompatibility(
                        "fixed quantity cannot fit the profile/inventory interval"
                    )
            else:
                quantity = (
                    lo + stream.derive("declared-quantity").randbelow(hi - lo + 1)
                ) * multiple
            inventory_receipt["quantityUpperBound"] = upper
        elif quantity is None:
            multiple = cap.quantity_multiple
            required = len(source.goods.get("splitGoodsPlacement", [])) or 1
            lo = (max(selected.quantity_min, required) + multiple - 1) // multiple
            capacity = equipment_capacity({"sizeCategory": size, "typeCategory": kind}, self.limits)
            source_counts = [
                a.get("packageQuantity") for a in source.goods.get("splitGoodsPlacement", [])
            ]
            peak = (
                max(Decimal(q) / source.quantity for q in source_counts)
                if source_counts and all(q is not None for q in source_counts)
                else Decimal(1) / len(source.containers)
            )
            upper = min(
                selected.quantity_max, int(capacity.payload_kg / (selected.unit_gross_kg * peak))
            )
            if capacity.volume_m3 is not None:
                upper = min(upper, int(capacity.volume_m3 / (selected.unit_volume_m3 * peak)))
            hi = upper // multiple
            if lo > hi:
                raise CandidateIncompatibility(
                    "profile quantity interval cannot fit source multiples and equipment capacity"
                )
            quantity = (lo + stream.derive("declared-quantity").randbelow(hi - lo + 1)) * multiple
        donor = CargoObservation(
            document_id="registry-form:" + selected.id,
            goods={
                "numberAndTypeOfPackages": [
                    {"packageQuantity": 1, "typeCategory": selected.package_category}
                ],
                "grossWeight": {"value": float(selected.unit_gross_kg), "unit": "kilogram"},
                "netWeight": {"value": float(selected.unit_net_kg), "unit": "kilogram"},
                "volume": {"value": float(selected.unit_volume_m3), "unit": "cubic_metre"},
            },
            containers=({"sizeCategory": size, "typeCategory": kind},),
            family="ambient",
            hs6=tuple(selected_codes),
            package_category=selected.package_category,
            quantity=1,
        )
        cargo, receipt = self._cargo(
            source,
            donor,
            cap,
            identities,
            None,
            stream,
            quantity_override=quantity,
            inventory_pairs=inventory_pairs,
        )
        # This is not an invented train donor: the compiler receives the exact
        # selected profile and per-declared-unit ledger as its alternative authority.
        receipt.pop("donorDocumentId")
        for fact in receipt.get("privatePhysicalMeasures", {}).values():
            fact["basis"] = "reviewed_registry_form_per_declared_unit_measure"
        receipt["goodsSamplingBasis"] = "reviewed_registry_form_and_independent_equipment"
        receipt["registryLoadProfile"] = selected.model_dump(mode="json")
        receipt["explicitPhysicalFacts"] = {
            measure: {"value": str(unit * quantity), "unit": unit_name}
            for measure, unit, unit_name in (
                ("grossWeight", selected.unit_gross_kg, "kilogram"),
                ("netWeight", selected.unit_net_kg, "kilogram"),
                ("volume", selected.unit_volume_m3, "cubic_metre"),
            )
        }
        receipt["loadScale"] = None
        if inventory_receipt is not None:
            receipt["equipmentInventory"] = inventory_receipt
            receipt["goodsSamplingBasis"] = "reviewed_registry_form_and_identifier_owned_inventory"
        return identities, cargo, receipt

    def sample(
        self,
        source_row: Mapping[str, Any],
        *,
        seed: int,
        variant: int,
        capabilities: SourceCapabilities,
        validate_candidate: Callable[[ShipmentScenario], None] | None = None,
    ) -> ShipmentScenario:
        sid = source_row["documentId"]
        if capabilities.equipment_mix is not None:
            capabilities.equipment_mix.validate_source(
                source_row["joinedRawText"],
                source_row["target"]["documentPatch"].get("containerInformation", []),
            )
        if (
            variant < 1
            or sid not in self.train
            or canonical_json_bytes(source_row["target"])
            != canonical_json_bytes(self.train[sid]["target"])
        ):
            raise ValueError("scenario source is not the exact current train-fit label")
        inventory_contract = None
        inventory_mode = None
        inventory_source_target_sha256 = None
        policy = self.config.equipment_inventory
        if policy is not None and sid in policy.source_ids:
            if capabilities.physical_profile != "registry_form":
                raise ValueError(
                    "configured inventory source requires the ambient registry-form domain"
                )
            if sid not in self.equipment_inventory:
                raise ValueError("configured inventory source has no bound ownership certificate")
            inventory_contract = self.equipment_inventory[sid]
            inventory_source_target_sha256 = sha256_bytes(
                canonical_json_bytes(source_row["target"])
            )
            if (
                inventory_contract.source_sha256
                != sha256_bytes(source_row["joinedRawText"].encode())
                or inventory_contract.source_target_sha256 != inventory_source_target_sha256
            ):
                raise ValueError(
                    "inventory source OCR or labels differ from its bound ownership certificate"
                )
        source = _observation(source_row, capabilities.family)
        if source is None:
            raise ValueError(
                "current pilot requires one goods group and one positive typed package row"
            )
        stream = DeterministicStream(seed, "curated-v7-scenarios-v1", f"{sid}:{variant}")
        if inventory_contract is not None:
            inventory_mode = policy.mode(variant, stream.derive("inventory-mode"))
        origin, destination, parties, geography_replacements, geography_receipt = self._geography(
            source_row, capabilities, stream.derive("geography")
        )
        registry_form = capabilities.physical_profile == "registry_form"
        donors = () if registry_form else self._donors(source, capabilities)
        if not registry_form and not donors:
            raise ValueError(f"no complete train-only cargo donors for {sid}/{capabilities.family}")
        failures: dict[str, int] = defaultdict(int)
        for attempt in range(self.config.maximum_candidates):
            draw = stream.derive(f"cargo-{attempt}")
            donor = None if registry_form else _pick(donors, draw.derive("donor"))
            try:
                if registry_form:
                    if inventory_contract is None:
                        identities, cargo, receipt = self._registry_cargo(
                            source, capabilities, draw
                        )
                    else:
                        identities, cargo, receipt = self._registry_cargo(
                            source,
                            capabilities,
                            draw,
                            inventory_contract=inventory_contract,
                            inventory_mode=inventory_mode,
                        )
                else:
                    identities, dg_record = self._identities(source, donor, capabilities, draw)
                    cargo, receipt = self._cargo(
                        source, donor, capabilities, identities, dg_record, draw
                    )
            except CandidateIncompatibility as error:
                failures[str(error)] += 1
                continue
            # Rejected cargo is never an input to the next attempt. In particular,
            # a retry back to a source value must remove the rejected replacement.
            replacements = dict(geography_replacements)
            for index, goods in enumerate(cargo["goodsItemDetails"]):
                for key in goods.get("origin", {}):
                    goods["origin"][key] = replacements[
                        f"documentPatch.goodsItemDetails[{index}].origin.{key}"
                    ]
            for field, value in cargo["goodsItemDetails"][0].items():
                if field in source.goods and value != source.goods[field]:
                    replacements[f"documentPatch.goodsItemDetails[0].{field}"] = value
            for i, container in enumerate(cargo["containerInformation"]):
                for field, value in container.items():
                    if value != source.containers[i][field]:
                        replacements[f"documentPatch.containerInformation[{i}].{field}"] = value
            scenario = ShipmentScenario(
                source_id=sid,
                variant=variant,
                origin=origin,
                destination=destination,
                route_locations={
                    field: ScenarioLocation.model_validate(geography_receipt["routeNodes"][node])
                    for field, node in geography_receipt.get("routeFields", {}).items()
                },
                party_localities=parties,
                goods_identities=identities,
                replacements=replacements,
                cargo=cargo,
                provenance={
                    "method": "registry_goods_train_physical_support_geography_v2",
                    "fitSha256": self.fit_sha256,
                    "registryConfigSha256": self.registry_sha256,
                    "sourceTargetSha256": inventory_source_target_sha256
                    or sha256_bytes(canonical_json_bytes(source_row["target"])),
                    "seed": seed,
                    "variant": variant,
                    "capabilities": capabilities.model_dump(mode="json"),
                    "candidateAttempts": attempt + 1,
                    "geographyEligibility": geography_receipt,
                    "rejectedCandidates": dict(failures),
                    "observedHSOverride": (
                        self.config.observed_hs_overrides[donor.document_id].model_dump(mode="json")
                        if donor is not None
                        and donor.document_id in self.config.observed_hs_overrides
                        else None
                    ),
                    **receipt,
                },
            )
            if validate_candidate is not None:
                try:
                    validate_candidate(scenario)
                except CandidateIncompatibility as error:
                    failures[str(error)] += 1
                    continue
            return scenario
        raise ValueError(
            f"no physically compatible scenario after {self.config.maximum_candidates} draws: "
            f"{dict(failures)}"
        )
