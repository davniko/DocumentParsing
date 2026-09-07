"""Route-first, registry-projected shipment and party-locality scenarios.

The scenario is deliberately richer than the task label.  Commercial exporter
and importer countries are distinct from the physical loading/discharge
countries, and party roles are related to those entities through explicit
classes.  Concrete names and addresses are generated later; this stage owns
only geography, route topology, and freight semantics.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, cast

from document_ocr.synthesis.config import RouteScenarioOriginPriorConfig
from document_ocr.synthesis.country_registry import CountryRegistry
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.locality_registry import LocalityRecord
from document_ocr.synthesis.routes import RouteLocation, TradeFlowRecord, weighted_index
from document_ocr.synthesis.world_port_registry import WorldPortRecord

_MARITIME_FUNCTION = "1"

RouteSide = Literal["commercial_origin", "commercial_destination", "third_country", "missing"]
EndpointRelation = Literal["same_country", "other_country"]
LocalityMode = Literal["endpoint", "other_same_country", "missing"]
FreightArrangement = Literal["prepaid", "collect", "third_party", "payable_elsewhere"]
OriginPriorComponent = Literal["observed_exporter", "maritime_registry"]
OriginPriorWeighting = Literal[
    "train_isolated_shipper_country_document_count_v1",
    "uniform_route_feasible_iso_country_v1",
]

_PARTY_ROLES = (
    "shipper",
    "consignee",
    "notifyParties",
    "carrier",
    "forwardingAgent",
    "deliveryAgent",
    "consolidator",
)
_ROUTE_ROLES = (
    "placeOfReceipt",
    "portOfLoading",
    "portOfDischarge",
    "placeOfDelivery",
    "finalDestination",
)
_NON_ALPHANUMERIC = re.compile(r"[^A-Z0-9]+")
_PARTY_IDENTITY_FIELDS = ("name", "address", "city", "country")


def normalize_location_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("location name must be a non-empty string")
    normalized = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").upper()
    )
    normalized = _NON_ALPHANUMERIC.sub(" ", normalized).strip()
    if not normalized:
        raise ValueError("location name has no alphanumeric content")
    return normalized


@dataclass(frozen=True, slots=True)
class WeightedCategory:
    value: str
    count: int

    def __post_init__(self) -> None:
        if not self.value or self.count <= 0:
            raise ValueError("weighted categories require a non-empty value and positive count")


@dataclass(frozen=True, slots=True)
class SampledScenarioLocation:
    locode: str | None
    geoname_id: int | None
    name: str
    country_code: str
    country_name: str
    source: Literal[
        "observed_port",
        "registry_port_exploration",
        "registry_port_no_observed_support",
        "endpoint",
        "geonames_population_weighted",
    ]

    def __post_init__(self) -> None:
        if self.source in {
            "observed_port",
            "registry_port_exploration",
            "registry_port_no_observed_support",
            "endpoint",
        }:
            if self.locode is None or self.geoname_id is not None:
                raise ValueError("port/endpoint locations require only a UN/LOCODE identity")
        elif self.locode is not None or self.geoname_id is None:
            raise ValueError("GeoNames localities require only a geoname identity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "locode": self.locode,
            "geonameId": self.geoname_id,
            "name": self.name,
            "countryCode": self.country_code,
            "countryName": self.country_name,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ConcretePartyIdentitySource:
    """Source party whose explicitly printed identity is repeated verbatim.

    This is deliberately distinct from the task-schema ``sameAs`` relation.
    A concrete repeated notify block remains a concrete party in the target;
    this reference only prevents synthesis from assigning the two printed
    copies to different entities or geographies.
    """

    role: str
    occurrence: int

    def __post_init__(self) -> None:
        if self.role not in _PARTY_ROLES or self.occurrence < 0:
            raise ValueError("concrete party identity sources require a valid party reference")

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "occurrence": self.occurrence}


@dataclass(frozen=True, slots=True)
class PartyLocalityScenario:
    role: str
    occurrence: int
    same_as: str | None
    relation: RouteSide
    country_present: bool
    city_present: bool
    country_code: str | None
    country_name: str | None
    conditioning_country_code: str
    conditioning_country_name: str
    locality_mode: LocalityMode
    locality: SampledScenarioLocation | None
    concrete_identity_source: ConcretePartyIdentitySource | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "occurrence": self.occurrence,
            "sameAs": self.same_as,
            "relation": self.relation,
            "countryPresent": self.country_present,
            "cityPresent": self.city_present,
            "countryCode": self.country_code,
            "countryName": self.country_name,
            "conditioningCountryCode": self.conditioning_country_code,
            "conditioningCountryName": self.conditioning_country_name,
            "localityMode": self.locality_mode,
            "locality": self.locality.to_dict() if self.locality is not None else None,
            "concreteIdentitySource": (
                self.concrete_identity_source.to_dict()
                if self.concrete_identity_source is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class FreightScenario:
    arrangement: FreightArrangement | None
    payment_place_present: bool
    payment_side: RouteSide
    payment_place: SampledScenarioLocation | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arrangement": self.arrangement,
            "paymentPlacePresent": self.payment_place_present,
            "paymentSide": self.payment_side,
            "paymentPlace": (
                self.payment_place.to_dict() if self.payment_place is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ShipmentScenario:
    base_document_id: str
    commercial_origin_country_code: str
    commercial_destination_country_code: str
    loading_port: SampledScenarioLocation
    discharge_port: SampledScenarioLocation
    route_locations: Mapping[str, SampledScenarioLocation]
    party_localities: tuple[PartyLocalityScenario, ...]
    place_of_issue: SampledScenarioLocation | None
    freight: FreightScenario
    trade_flow_year: int
    trade_flow_weight: Decimal
    trade_flow_provider: Literal["world_bank_wits_latest_reporter_year_v1"]
    origin_prior_method: Literal["observed_exporter_plus_maritime_registry_mixture_v1"]
    origin_prior_component: OriginPriorComponent
    origin_prior_weighting: OriginPriorWeighting
    loading_country_method: Literal[
        "train_empirical_conditioned_on_origin_v1",
        "commercial_origin_identity_registry_v1",
    ]
    discharge_country_method: Literal[
        "train_empirical_conditioned_on_destination_v1",
        "commercial_destination_identity_when_unobserved_v1",
    ]
    transshipment_status: Literal["not_present"]
    vessel_status: Literal["not_present", "pending_synthetic_transport_identity"]

    def __post_init__(self) -> None:
        if not self.base_document_id:
            raise ValueError("scenario base_document_id cannot be empty")
        if self.loading_port.locode == self.discharge_port.locode:
            raise ValueError("loading and discharge ports must differ")
        if self.loading_port.country_code == self.discharge_port.country_code:
            raise ValueError("this scenario contract is international-only")
        if self.route_locations.get("portOfLoading") != self.loading_port or (
            self.route_locations.get("portOfDischarge") != self.discharge_port
        ):
            raise ValueError("route endpoints differ from route location projection")
        if self.trade_flow_year < 1900 or self.trade_flow_weight <= 0:
            raise ValueError("scenario trade-flow evidence is invalid")
        expected_weighting: Mapping[OriginPriorComponent, OriginPriorWeighting] = {
            "observed_exporter": "train_isolated_shipper_country_document_count_v1",
            "maritime_registry": "uniform_route_feasible_iso_country_v1",
        }
        if self.origin_prior_weighting != expected_weighting[self.origin_prior_component]:
            raise ValueError("commercial-origin component and weighting differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseDocumentId": self.base_document_id,
            "commercialOriginCountryCode": self.commercial_origin_country_code,
            "commercialDestinationCountryCode": self.commercial_destination_country_code,
            "commercialOriginPriorEvidence": {
                "method": self.origin_prior_method,
                "component": self.origin_prior_component,
                "weighting": self.origin_prior_weighting,
            },
            "loadingPort": self.loading_port.to_dict(),
            "dischargePort": self.discharge_port.to_dict(),
            "routeLocations": {
                role: location.to_dict() for role, location in sorted(self.route_locations.items())
            },
            "partyLocalities": [row.to_dict() for row in self.party_localities],
            "placeOfIssue": (
                self.place_of_issue.to_dict() if self.place_of_issue is not None else None
            ),
            "freight": self.freight.to_dict(),
            "tradeFlowEvidence": {
                "provider": self.trade_flow_provider,
                "year": self.trade_flow_year,
                "weight": str(self.trade_flow_weight),
            },
            "physicalEndpointCountryMethods": {
                "loading": self.loading_country_method,
                "discharge": self.discharge_country_method,
            },
            "transshipmentStatus": self.transshipment_status,
            "vesselStatus": self.vessel_status,
        }


@dataclass(frozen=True, slots=True)
class ScenarioCountryResolutionIssue:
    document_id: str
    path: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {
            "documentId": self.document_id,
            "path": self.path,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class ScenarioSupportAudit:
    input_documents: int
    country_clean_documents: int
    excluded_unresolved_country_documents: int
    unresolved_country_cells: int
    unresolved_country_issues: tuple[ScenarioCountryResolutionIssue, ...]
    usable_export_country_documents: int
    missing_export_country_documents: int
    export_country_without_maritime_port_documents: int
    export_country_without_locality_depth_documents: int
    export_country_without_loading_country_support_documents: int
    export_country_without_trade_flow_documents: int
    ambiguous_maritime_name_keys: int
    world_port_rows_input: int
    world_port_rows_excluded_non_maritime: int
    world_port_rows_eligible: int
    eligible_world_port_locodes: int
    observed_loading_ports_matched: int
    observed_loading_ports_unmatched: int
    observed_discharge_ports_matched: int
    observed_discharge_ports_unmatched: int
    excluded_transshipment_documents: int
    observed_export_countries: int
    maritime_registry_export_countries: int
    observed_registry_overlap_countries: int
    maritime_registry_only_export_countries: int
    observed_exporter_mixture_permyriad: int
    maritime_registry_mixture_permyriad: int

    def __post_init__(self) -> None:
        unresolved_documents = {issue.document_id for issue in self.unresolved_country_issues}
        if (
            self.country_clean_documents + self.excluded_unresolved_country_documents
            != self.input_documents
            or self.excluded_unresolved_country_documents != len(unresolved_documents)
            or self.unresolved_country_cells != len(self.unresolved_country_issues)
        ):
            raise ValueError("scenario-support country resolution audit does not balance")
        classified = (
            self.usable_export_country_documents
            + self.missing_export_country_documents
            + self.export_country_without_maritime_port_documents
            + self.export_country_without_locality_depth_documents
            + self.export_country_without_loading_country_support_documents
            + self.export_country_without_trade_flow_documents
            + self.excluded_transshipment_documents
        )
        if classified != self.country_clean_documents:
            raise ValueError("scenario-support document audit does not balance")
        if (
            self.observed_registry_overlap_countries != self.observed_export_countries
            or self.observed_registry_overlap_countries
            + self.maritime_registry_only_export_countries
            != self.maritime_registry_export_countries
            or self.observed_exporter_mixture_permyriad + self.maritime_registry_mixture_permyriad
            != 10_000
        ):
            raise ValueError("scenario-support commercial-origin mixture audit does not balance")
        if (
            self.world_port_rows_excluded_non_maritime + self.world_port_rows_eligible
            != self.world_port_rows_input
            or self.eligible_world_port_locodes > self.world_port_rows_eligible
        ):
            raise ValueError("scenario-support world-port audit does not balance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputDocuments": self.input_documents,
            "countryCleanDocuments": self.country_clean_documents,
            "excludedUnresolvedCountryDocuments": (self.excluded_unresolved_country_documents),
            "unresolvedCountryCells": self.unresolved_country_cells,
            "unresolvedCountryIssues": [
                issue.to_dict() for issue in self.unresolved_country_issues
            ],
            "usableExportCountryDocuments": self.usable_export_country_documents,
            "missingExportCountryDocuments": self.missing_export_country_documents,
            "exportCountryWithoutMaritimePortDocuments": (
                self.export_country_without_maritime_port_documents
            ),
            "exportCountryWithoutLocalityDepthDocuments": (
                self.export_country_without_locality_depth_documents
            ),
            "exportCountryWithoutLoadingCountrySupportDocuments": (
                self.export_country_without_loading_country_support_documents
            ),
            "exportCountryWithoutTradeFlowDocuments": (
                self.export_country_without_trade_flow_documents
            ),
            "observedLoadingPortsMatched": self.observed_loading_ports_matched,
            "observedLoadingPortsUnmatched": self.observed_loading_ports_unmatched,
            "observedDischargePortsMatched": self.observed_discharge_ports_matched,
            "observedDischargePortsUnmatched": self.observed_discharge_ports_unmatched,
            "excludedTransshipmentDocuments": self.excluded_transshipment_documents,
            "ambiguousMaritimeNameKeys": self.ambiguous_maritime_name_keys,
            "worldPortRowsInput": self.world_port_rows_input,
            "worldPortRowsExcludedNonMaritime": (
                self.world_port_rows_excluded_non_maritime
            ),
            "worldPortRowsEligible": self.world_port_rows_eligible,
            "eligibleWorldPortLocodes": self.eligible_world_port_locodes,
            "observedExportCountries": self.observed_export_countries,
            "maritimeRegistryExportCountries": self.maritime_registry_export_countries,
            "observedRegistryOverlapCountries": self.observed_registry_overlap_countries,
            "maritimeRegistryOnlyExportCountries": (self.maritime_registry_only_export_countries),
            "observedExporterMixturePermyriad": self.observed_exporter_mixture_permyriad,
            "maritimeRegistryMixturePermyriad": self.maritime_registry_mixture_permyriad,
        }


@dataclass(frozen=True, slots=True)
class TradeDestination:
    country_code: str
    year: int
    weight: Decimal

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Z]{2}", self.country_code) is None:
            raise ValueError("trade destination must use ISO alpha-2")
        if self.year < 1900 or not self.weight.is_finite() or self.weight <= 0:
            raise ValueError("trade destination requires a positive finite annual weight")


@dataclass(frozen=True, slots=True)
class ScenarioTradeFlowAudit:
    input_records: int
    eligible_records: int
    excluded_nonpositive_records: int
    excluded_domestic_records: int
    excluded_origin_without_maritime_port_records: int
    excluded_origin_without_locality_depth_records: int
    excluded_destination_without_maritime_port_records: int
    excluded_destination_without_locality_depth_records: int
    excluded_destination_without_distinct_physical_endpoint_records: int

    def __post_init__(self) -> None:
        classified = (
            self.eligible_records
            + self.excluded_nonpositive_records
            + self.excluded_domestic_records
            + self.excluded_origin_without_maritime_port_records
            + self.excluded_origin_without_locality_depth_records
            + self.excluded_destination_without_maritime_port_records
            + self.excluded_destination_without_locality_depth_records
            + self.excluded_destination_without_distinct_physical_endpoint_records
        )
        if classified != self.input_records:
            raise ValueError("scenario trade-flow audit does not balance")

    def to_dict(self) -> dict[str, int]:
        return {
            "inputRecords": self.input_records,
            "eligibleRecords": self.eligible_records,
            "excludedNonpositiveRecords": self.excluded_nonpositive_records,
            "excludedDomesticRecords": self.excluded_domestic_records,
            "excludedOriginWithoutMaritimePortRecords": (
                self.excluded_origin_without_maritime_port_records
            ),
            "excludedOriginWithoutLocalityDepthRecords": (
                self.excluded_origin_without_locality_depth_records
            ),
            "excludedDestinationWithoutMaritimePortRecords": (
                self.excluded_destination_without_maritime_port_records
            ),
            "excludedDestinationWithoutLocalityDepthRecords": (
                self.excluded_destination_without_locality_depth_records
            ),
            "excludedDestinationWithoutDistinctPhysicalEndpointRecords": (
                self.excluded_destination_without_distinct_physical_endpoint_records
            ),
        }


@dataclass(frozen=True, slots=True)
class ScenarioSupport:
    origin_prior: RouteScenarioOriginPriorConfig
    observed_export_countries: tuple[WeightedCategory, ...]
    maritime_registry_export_countries: tuple[WeightedCategory, ...]
    physical_loading_countries_by_origin: Mapping[str, tuple[WeightedCategory, ...]]
    loading_country_methods_by_origin: Mapping[
        str,
        Literal[
            "train_empirical_conditioned_on_origin_v1",
            "commercial_origin_identity_registry_v1",
        ],
    ]
    physical_discharge_countries_by_destination: Mapping[str, tuple[WeightedCategory, ...]]
    party_relations: Mapping[str, tuple[WeightedCategory, ...]]
    party_third_countries: Mapping[str, tuple[WeightedCategory, ...]]
    party_locality_modes: Mapping[tuple[str, str], tuple[WeightedCategory, ...]]
    issue_sides: tuple[WeightedCategory, ...]
    issue_third_countries: tuple[WeightedCategory, ...] | None
    freight_arrangements_by_payment_presence: Mapping[bool, tuple[WeightedCategory, ...]]
    freight_payment_sides: Mapping[str, tuple[WeightedCategory, ...]]
    freight_third_countries: Mapping[str, tuple[WeightedCategory, ...]]
    localities_by_country: Mapping[str, tuple[LocalityRecord, ...]]
    maritime_by_country: Mapping[str, tuple[RouteLocation, ...]]
    observed_loading_ports: Mapping[str, tuple[WeightedCategory, ...]]
    observed_discharge_ports: Mapping[str, tuple[WeightedCategory, ...]]
    trade_destinations_by_origin: Mapping[str, tuple[TradeDestination, ...]]
    audit: ScenarioSupportAudit
    trade_flow_audit: ScenarioTradeFlowAudit

    def __post_init__(self) -> None:
        observed = {row.value for row in self.observed_export_countries}
        registry = {row.value for row in self.maritime_registry_export_countries}
        route_origins = set(self.trade_destinations_by_origin)
        if not observed:
            raise ValueError("observed-exporter origin component has no route-feasible support")
        if registry != route_origins or any(
            row.count != 1 for row in self.maritime_registry_export_countries
        ):
            raise ValueError("maritime-registry origin support must uniformly cover route origins")
        if not observed <= registry:
            raise ValueError("observed-exporter origin support is outside registry route support")
        if route_origins != set(self.physical_loading_countries_by_origin):
            raise ValueError("route origins and physical-loading support differ")
        if route_origins != set(self.loading_country_methods_by_origin):
            raise ValueError("route origins and loading-country methods differ")
        for origin, destinations in self.trade_destinations_by_origin.items():
            if re.fullmatch(r"[A-Z]{2}", origin) is None:
                raise ValueError(f"commercial origin does not use ISO alpha-2: {origin}")
            loading_rows = self.physical_loading_countries_by_origin[origin]
            loading_method = self.loading_country_methods_by_origin[origin]
            if loading_method == "commercial_origin_identity_registry_v1" and loading_rows != (
                WeightedCategory(origin, 1),
            ):
                raise ValueError(
                    f"registry-identity loading support is not the commercial origin: {origin}"
                )
            if not destinations or any(
                not _feasible_physical_loading_rows(
                    self,
                    commercial_origin=origin,
                    commercial_destination=destination.country_code,
                )
                for destination in destinations
            ):
                raise ValueError(f"commercial origin has a non-feasible compiled route: {origin}")


@dataclass(frozen=True, slots=True)
class SampledCommercialOrigin:
    country_code: str
    component: OriginPriorComponent
    weighting: OriginPriorWeighting


@dataclass(frozen=True, slots=True)
class _PreeligibleDocument:
    document_id: str
    patch: Mapping[str, Any]
    route: Mapping[str, Any]
    parties: Mapping[str, Any]
    consignee: Mapping[str, Any]
    export_country: str
    loading_country: str


def _weighted_choice(
    rows: Sequence[WeightedCategory],
    *,
    stream: DeterministicStream,
    exclude: frozenset[str] = frozenset(),
) -> str:
    eligible = tuple(row for row in rows if row.value not in exclude)
    if not eligible:
        raise ValueError("weighted category support is empty after exclusions")
    index = weighted_index(
        tuple(Decimal(row.count) for row in eligible),
        stream=stream,
    )
    return eligible[index].value


def _categories(counter: Mapping[str, int], *, label: str) -> tuple[WeightedCategory, ...]:
    rows = tuple(
        WeightedCategory(value, count) for value, count in sorted(counter.items()) if count
    )
    if not rows:
        raise ValueError(f"scenario support has no observations for {label}")
    return rows


def _patch(target: Mapping[str, Any]) -> Mapping[str, Any]:
    patch = target.get("documentPatch")
    if target.get("schemaVersion") != "3.0.0-experimental" or not isinstance(patch, Mapping):
        raise ValueError("shipment scenarios require relation-v3 targets")
    return cast(Mapping[str, Any], patch)


def _modeled_country_cells(patch: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    cells: list[tuple[str, Any]] = []
    route = _location_dict(patch.get("route"))
    for role, location in sorted(route.items()):
        if isinstance(location, Mapping):
            cells.append((f"documentPatch.route.{role}.country", location.get("country")))
    issue = _location_dict(patch.get("placeOfIssue"))
    cells.append(("documentPatch.placeOfIssue.country", issue.get("country")))
    for role, occurrence, party in _party_rows(patch):
        cells.append(
            (
                f"documentPatch.parties.{role}[{occurrence}].country",
                party.get("country"),
            )
        )
    freight = _location_dict(patch.get("freight"))
    payment = _location_dict(freight.get("paymentPlace"))
    cells.append(("documentPatch.freight.paymentPlace.country", payment.get("country")))
    return tuple(cells)


def _party_rows(patch: Mapping[str, Any]) -> tuple[tuple[str, int, Mapping[str, Any]], ...]:
    parties = patch.get("parties")
    if not isinstance(parties, Mapping):
        return ()
    rows: list[tuple[str, int, Mapping[str, Any]]] = []
    for role in _PARTY_ROLES:
        value = parties.get(role)
        values = value if role == "notifyParties" and isinstance(value, list) else [value]
        for index, party in enumerate(values):
            if isinstance(party, Mapping):
                rows.append((role, index, cast(Mapping[str, Any], party)))
    return tuple(rows)


def _concrete_party_identity_signature(
    party: Mapping[str, Any],
) -> tuple[str | None, ...] | None:
    """Return a conservative exact signature for a concrete printed identity.

    A name plus at least one corroborating identity field is required.  Exact
    surfaces are used intentionally: fuzzy matching could collapse unrelated
    parties and silently corrupt a synthetic label.
    """

    if party.get("sameAs") is not None:
        return None
    values: list[str | None] = []
    for field in _PARTY_IDENTITY_FIELDS:
        value = party.get(field)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"party identity field must be text or null: {field}")
        values.append(value)
    name = values[0]
    if name is None or not name.strip() or not any(value and value.strip() for value in values[1:]):
        return None
    return tuple(values)


def _resolved_country(value: Any, registry: CountryRegistry) -> str | None:
    return registry.resolve(value) if isinstance(value, str) else None


def _relation(country: str | None, origin: str | None, destination: str | None) -> RouteSide:
    if country is None:
        return "missing"
    if origin is not None and country == origin:
        return "commercial_origin"
    if destination is not None and country == destination:
        return "commercial_destination"
    return "third_country"


def _location_dict(value: Any) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _unique_name_index(
    locations: Sequence[RouteLocation],
) -> tuple[dict[tuple[str, str], RouteLocation], frozenset[tuple[str, str]]]:
    candidates: dict[tuple[str, str], list[RouteLocation]] = defaultdict(list)
    for location in locations:
        candidates[(location.country_code, normalize_location_name(location.name))].append(location)
    ambiguous = frozenset(key for key, values in candidates.items() if len(values) != 1)
    return (
        {key: values[0] for key, values in candidates.items() if len(values) == 1},
        ambiguous,
    )


def _positive_locality_name_depth(
    localities: Mapping[str, Sequence[LocalityRecord]],
) -> dict[str, int]:
    """Count usable locality identities by country after display-name normalization.

    A commercial endpoint can itself have the same printed name as a GeoNames
    row.  Sampling another locality therefore requires two *distinct names*,
    not merely two source rows (which may be duplicate provider identities).
    """

    return {
        country: len(
            {normalize_location_name(row.ascii_name) for row in rows if row.population > 0}
        )
        for country, rows in sorted(localities.items())
    }


def build_scenario_support(
    *,
    source_targets: Mapping[str, Mapping[str, Any]],
    fit_document_ids: Sequence[str],
    country_registry: CountryRegistry,
    route_locations: Sequence[RouteLocation],
    world_ports: Sequence[WorldPortRecord],
    localities: Sequence[LocalityRecord],
    trade_flows: Sequence[TradeFlowRecord],
    origin_prior: RouteScenarioOriginPriorConfig,
) -> ScenarioSupport:
    """Fit explicit low-cardinality scenario priors on an already-isolated scope."""

    if country_registry.audit.observed_alias_records != 0:
        raise ValueError("shipment sampling requires an alias-free ISO country registry")
    if not fit_document_ids or len(fit_document_ids) != len(set(fit_document_ids)):
        raise ValueError("fit_document_ids must be non-empty and unique")
    missing = sorted(set(fit_document_ids) - set(source_targets))
    if missing:
        raise ValueError(f"fit scenario documents are absent from source targets: {missing}")
    country_issues: list[ScenarioCountryResolutionIssue] = []
    country_unclean_documents: set[str] = set()
    for document_id in sorted(fit_document_ids):
        patch = _patch(source_targets[document_id])
        for path, value in _modeled_country_cells(patch):
            if value is None:
                continue
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"malformed modeled country at {document_id}:{path}")
            if country_registry.resolve(value) is None:
                country_unclean_documents.add(document_id)
                country_issues.append(
                    ScenarioCountryResolutionIssue(
                        document_id=document_id,
                        path=path,
                        value=value,
                    )
                )
    country_clean_fit_ids = tuple(
        document_id
        for document_id in sorted(fit_document_ids)
        if document_id not in country_unclean_documents
    )
    if not country_clean_fit_ids:
        raise ValueError("no alias-free country-clean documents remain for scenario support")
    route_by_locode: dict[str, RouteLocation] = {}
    for location in sorted(route_locations, key=lambda row: row.locode):
        if location.locode in route_by_locode:
            raise ValueError(f"duplicate scenario registry locode: {location.locode}")
        route_by_locode[location.locode] = location

    wpi_locodes: set[str] = set()
    eligible_world_port_rows = 0
    excluded_non_maritime_world_port_rows = 0
    for port in world_ports:
        country_registry.entry(port.country_code)
        if port.locode not in route_by_locode:
            raise ValueError(f"world-port LOCODE is absent from route registry: {port.locode}")
        route_location = route_by_locode[port.locode]
        if route_location.country_code != port.country_code:
            raise ValueError(f"world-port and route country differ: {port.locode}")
        # WPI and UN/LOCODE are independently versioned.  A LOCODE that WPI still
        # assigns to a port can be reassigned to a non-maritime function in the
        # pinned UN/LOCODE release.  Since scenario labels use the current
        # UN/LOCODE name, such stale intersections must never enter port fields.
        if _MARITIME_FUNCTION not in route_location.function_codes:
            excluded_non_maritime_world_port_rows += 1
            continue
        eligible_world_port_rows += 1
        wpi_locodes.add(port.locode)
    if not wpi_locodes:
        raise ValueError("world-port whitelist cannot be empty")

    maritime: dict[str, list[RouteLocation]] = defaultdict(list)
    for location in route_by_locode.values():
        if location.locode in wpi_locodes:
            maritime[location.country_code].append(location)
    maritime_locations = tuple(row for country_rows in maritime.values() for row in country_rows)
    name_index, ambiguous_names = _unique_name_index(maritime_locations)

    locality_rows: dict[str, list[LocalityRecord]] = defaultdict(list)
    seen_geonames: set[int] = set()
    for locality in sorted(localities, key=lambda row: row.geoname_id):
        if locality.geoname_id in seen_geonames:
            raise ValueError(f"duplicate GeoNames identity: {locality.geoname_id}")
        seen_geonames.add(locality.geoname_id)
        country_registry.entry(locality.country_code)
        locality_rows[locality.country_code].append(locality)
    if not locality_rows:
        raise ValueError("GeoNames locality support cannot be empty")
    locality_name_depth = _positive_locality_name_depth(locality_rows)

    export = Counter[str]()
    loading_countries: dict[str, Counter[str]] = defaultdict(Counter)
    discharge_countries: dict[str, Counter[str]] = defaultdict(Counter)
    party_relation: dict[str, Counter[str]] = defaultdict(Counter)
    party_third_countries: dict[str, Counter[str]] = defaultdict(Counter)
    party_city_mode: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    issue_sides = Counter[str]()
    issue_third_countries = Counter[str]()
    freight_arrangements: dict[bool, Counter[str]] = defaultdict(Counter)
    freight_sides: dict[str, Counter[str]] = defaultdict(Counter)
    freight_third_countries: dict[str, Counter[str]] = defaultdict(Counter)
    loading_ports: dict[str, Counter[str]] = defaultdict(Counter)
    discharge_ports: dict[str, Counter[str]] = defaultdict(Counter)
    missing_export = 0
    no_port_export = 0
    no_locality_depth_export = 0
    no_loading_country_support_export = 0
    no_trade_export = 0
    excluded_transshipment = 0
    usable = 0
    loading_matched = loading_unmatched = discharge_matched = discharge_unmatched = 0

    preeligible: list[_PreeligibleDocument] = []

    for document_id in country_clean_fit_ids:
        patch = _patch(source_targets[document_id])
        route = _location_dict(patch.get("route"))
        if route.get("transshipmentPort") is not None:
            excluded_transshipment += 1
            continue
        parties = _location_dict(patch.get("parties"))
        shipper = _location_dict(parties.get("shipper"))
        consignee = _location_dict(parties.get("consignee"))
        raw_export_country = shipper.get("country")
        if not isinstance(raw_export_country, str):
            missing_export += 1
            continue
        export_country = _resolved_country(raw_export_country, country_registry)
        if export_country is None:
            raise RuntimeError("country-clean shipper country became unresolved")
        if not maritime.get(export_country):
            no_port_export += 1
            continue
        if locality_name_depth.get(export_country, 0) < 2:
            no_locality_depth_export += 1
            continue
        loading_value = _location_dict(route.get("portOfLoading"))
        loading_country = _resolved_country(loading_value.get("country"), country_registry)
        if loading_country is None or not maritime.get(loading_country):
            no_loading_country_support_export += 1
            continue
        preeligible.append(
            _PreeligibleDocument(
                document_id=document_id,
                patch=patch,
                route=route,
                parties=parties,
                consignee=consignee,
                export_country=export_country,
                loading_country=loading_country,
            )
        )

    for prepared in preeligible:
        loading_countries[prepared.export_country][prepared.loading_country] += 1
        discharge = _location_dict(prepared.route.get("portOfDischarge"))
        import_country = _resolved_country(prepared.consignee.get("country"), country_registry)
        discharge_country = _resolved_country(discharge.get("country"), country_registry)
        if (
            import_country is not None
            and discharge_country is not None
            and maritime.get(discharge_country)
        ):
            discharge_countries[import_country][discharge_country] += 1

    trade_dispositions = Counter[str]()
    trade_by_origin: dict[str, list[TradeDestination]] = defaultdict(list)
    trade_year_by_origin: dict[str, int] = {}
    seen_trade_pairs: set[tuple[str, str]] = set()
    for flow in trade_flows:
        pair = (flow.origin_country_code, flow.destination_country_code)
        if pair in seen_trade_pairs:
            raise ValueError(f"duplicate scenario trade-flow country pair: {pair}")
        seen_trade_pairs.add(pair)
        try:
            country_registry.entry(flow.origin_country_code)
            country_registry.entry(flow.destination_country_code)
        except KeyError as error:
            raise ValueError(
                f"trade-flow pair is outside the pinned ISO registry: {pair}"
            ) from error
        if flow.trade_value is None:
            raise ValueError("WITS scenario trade-flow rows must contain trade_value")
        if flow.trade_value <= 0:
            trade_dispositions["nonpositive"] += 1
            continue
        if flow.origin_country_code == flow.destination_country_code:
            trade_dispositions["domestic"] += 1
            continue
        if not maritime.get(flow.origin_country_code):
            trade_dispositions["origin_without_maritime_port"] += 1
            continue
        if locality_name_depth.get(flow.origin_country_code, 0) < 2:
            trade_dispositions["origin_without_locality_depth"] += 1
            continue
        if not maritime.get(flow.destination_country_code):
            trade_dispositions["destination_without_maritime_port"] += 1
            continue
        if locality_name_depth.get(flow.destination_country_code, 0) < 2:
            trade_dispositions["destination_without_locality_depth"] += 1
            continue
        empirical_loading_options = loading_countries.get(flow.origin_country_code)
        loading_options = (
            frozenset(empirical_loading_options)
            if empirical_loading_options
            else frozenset({flow.origin_country_code})
        ) - frozenset({flow.destination_country_code})
        discharge_options = frozenset(discharge_countries.get(flow.destination_country_code, ()))
        if not loading_options or (
            discharge_options
            and not any(
                loading_country != discharge_country
                for loading_country in loading_options
                for discharge_country in discharge_options
            )
        ):
            trade_dispositions["destination_without_distinct_physical_endpoint"] += 1
            continue
        previous_year = trade_year_by_origin.setdefault(flow.origin_country_code, flow.year)
        if previous_year != flow.year:
            raise ValueError(
                "WITS scenario support requires one latest observation year per origin"
            )
        trade_by_origin[flow.origin_country_code].append(
            TradeDestination(
                country_code=flow.destination_country_code,
                year=flow.year,
                weight=flow.trade_value,
            )
        )
        trade_dispositions["eligible"] += 1

    ordered_trade_by_origin = {
        origin: tuple(sorted(rows, key=lambda row: row.country_code))
        for origin, rows in sorted(trade_by_origin.items())
    }
    if not ordered_trade_by_origin:
        raise ValueError("maritime-registry origin component has no route-feasible support")
    trade_flow_audit = ScenarioTradeFlowAudit(
        input_records=len(trade_flows),
        eligible_records=trade_dispositions["eligible"],
        excluded_nonpositive_records=trade_dispositions["nonpositive"],
        excluded_domestic_records=trade_dispositions["domestic"],
        excluded_origin_without_maritime_port_records=trade_dispositions[
            "origin_without_maritime_port"
        ],
        excluded_origin_without_locality_depth_records=trade_dispositions[
            "origin_without_locality_depth"
        ],
        excluded_destination_without_maritime_port_records=trade_dispositions[
            "destination_without_maritime_port"
        ],
        excluded_destination_without_locality_depth_records=trade_dispositions[
            "destination_without_locality_depth"
        ],
        excluded_destination_without_distinct_physical_endpoint_records=trade_dispositions[
            "destination_without_distinct_physical_endpoint"
        ],
    )

    for prepared in preeligible:
        if prepared.export_country not in ordered_trade_by_origin:
            no_trade_export += 1
            continue
        patch = prepared.patch
        route = prepared.route
        parties = prepared.parties
        consignee = prepared.consignee
        export_country = prepared.export_country
        usable += 1
        export[export_country] += 1
        loading_value = _location_dict(route.get("portOfLoading"))
        discharge_value = _location_dict(route.get("portOfDischarge"))
        loading_country = prepared.loading_country
        import_country = _resolved_country(consignee.get("country"), country_registry)
        discharge_country = _resolved_country(discharge_value.get("country"), country_registry)
        for value, country, store, match_name in (
            (loading_value, loading_country, loading_ports, "loading"),
            (discharge_value, discharge_country, discharge_ports, "discharge"),
        ):
            name = value.get("name")
            if country is None or not isinstance(name, str):
                continue
            key = (country, normalize_location_name(name))
            matched = name_index.get(key)
            if matched is not None:
                store[country][matched.locode] += 1
                if match_name == "loading":
                    loading_matched += 1
                else:
                    discharge_matched += 1
            elif match_name == "loading":
                loading_unmatched += 1
            else:
                discharge_unmatched += 1

        for role, _occurrence, party in _party_rows(patch):
            if party.get("sameAs") is not None:
                continue
            country = _resolved_country(party.get("country"), country_registry)
            relation = _relation(country, export_country, import_country)
            # Country-field missingness is preserved from the selected template;
            # it is not a geographic side.  Fit latent side priors only from
            # parties whose printed country establishes that relationship.
            if country is None:
                continue
            if relation == "third_country":
                if locality_name_depth.get(country, 0) < 1:
                    continue
                party_third_countries[role][country] += 1
            party_relation[role][relation] += 1
            city = party.get("city")
            endpoint_name = None
            endpoint_country = None
            if relation == "commercial_origin" and country == loading_country:
                endpoint_name = loading_value.get("name")
                endpoint_country = loading_country
            elif relation == "commercial_destination" and country == discharge_country:
                endpoint_name = discharge_value.get("name")
                endpoint_country = discharge_country
            if not isinstance(city, str):
                continue
            if (
                endpoint_country == country
                and isinstance(endpoint_name, str)
                and normalize_location_name(city) == normalize_location_name(endpoint_name)
            ):
                mode = "endpoint"
            else:
                mode = "other_same_country"
            party_city_mode[(role, relation)][mode] += 1

        issue = _location_dict(patch.get("placeOfIssue"))
        issue_country = _resolved_country(issue.get("country"), country_registry)
        if issue_country is not None:
            issue_relation = _relation(issue_country, export_country, import_country)
            if issue_relation != "third_country" or locality_name_depth.get(issue_country, 0) >= 1:
                issue_sides[issue_relation] += 1
                if issue_relation == "third_country":
                    issue_third_countries[issue_country] += 1
        freight = _location_dict(patch.get("freight"))
        arrangement = freight.get("paymentArrangement")
        if isinstance(arrangement, str):
            payment = _location_dict(freight.get("paymentPlace"))
            payment_present = freight.get("paymentPlace") is not None
            freight_arrangements[payment_present][arrangement] += 1
            payment_country = _resolved_country(payment.get("country"), country_registry)
            if payment_country is not None:
                payment_relation = _relation(payment_country, export_country, import_country)
                if (
                    payment_relation != "third_country"
                    or locality_name_depth.get(payment_country, 0) >= 1
                ):
                    freight_sides[arrangement][payment_relation] += 1
                    if payment_relation == "third_country":
                        freight_third_countries[arrangement][payment_country] += 1

    observed_origin_codes = frozenset(export)
    registry_origin_codes = frozenset(ordered_trade_by_origin)
    audit = ScenarioSupportAudit(
        input_documents=len(fit_document_ids),
        country_clean_documents=len(country_clean_fit_ids),
        excluded_unresolved_country_documents=len(country_unclean_documents),
        unresolved_country_cells=len(country_issues),
        unresolved_country_issues=tuple(
            sorted(country_issues, key=lambda row: (row.document_id, row.path, row.value))
        ),
        usable_export_country_documents=usable,
        missing_export_country_documents=missing_export,
        export_country_without_maritime_port_documents=no_port_export,
        export_country_without_locality_depth_documents=no_locality_depth_export,
        export_country_without_loading_country_support_documents=(
            no_loading_country_support_export
        ),
        export_country_without_trade_flow_documents=no_trade_export,
        ambiguous_maritime_name_keys=len(ambiguous_names),
        world_port_rows_input=len(world_ports),
        world_port_rows_excluded_non_maritime=excluded_non_maritime_world_port_rows,
        world_port_rows_eligible=eligible_world_port_rows,
        eligible_world_port_locodes=len(wpi_locodes),
        observed_loading_ports_matched=loading_matched,
        observed_loading_ports_unmatched=loading_unmatched,
        observed_discharge_ports_matched=discharge_matched,
        observed_discharge_ports_unmatched=discharge_unmatched,
        excluded_transshipment_documents=excluded_transshipment,
        observed_export_countries=len(observed_origin_codes),
        maritime_registry_export_countries=len(registry_origin_codes),
        observed_registry_overlap_countries=len(observed_origin_codes & registry_origin_codes),
        maritime_registry_only_export_countries=len(registry_origin_codes - observed_origin_codes),
        observed_exporter_mixture_permyriad=(origin_prior.observed_exporter.mixture_permyriad),
        maritime_registry_mixture_permyriad=(origin_prior.maritime_registry.mixture_permyriad),
    )
    return ScenarioSupport(
        origin_prior=origin_prior,
        observed_export_countries=_categories(export, label="observed export countries"),
        maritime_registry_export_countries=_categories(
            {country: 1 for country in registry_origin_codes},
            label="maritime-registry export countries",
        ),
        physical_loading_countries_by_origin={
            country: (
                _categories(
                    loading_countries[country],
                    label=f"loading countries for {country}",
                )
                if loading_countries.get(country)
                else (WeightedCategory(country, 1),)
            )
            for country in sorted(registry_origin_codes)
        },
        loading_country_methods_by_origin={
            country: (
                "train_empirical_conditioned_on_origin_v1"
                if loading_countries.get(country)
                else "commercial_origin_identity_registry_v1"
            )
            for country in sorted(registry_origin_codes)
        },
        physical_discharge_countries_by_destination={
            country: _categories(rows, label=f"discharge countries for {country}")
            for country, rows in sorted(discharge_countries.items())
        },
        party_relations={
            role: _categories(party_relation[role], label=f"party relation {role}")
            for role in sorted(party_relation)
        },
        party_third_countries={
            role: _categories(rows, label=f"party third countries {role}")
            for role, rows in sorted(party_third_countries.items())
        },
        party_locality_modes={
            key: _categories(rows, label=f"party locality mode {key[0]}:{key[1]}")
            for key, rows in sorted(party_city_mode.items())
        },
        issue_sides=_categories(issue_sides, label="issue side"),
        issue_third_countries=(
            _categories(issue_third_countries, label="issue third countries")
            if issue_third_countries
            else None
        ),
        freight_arrangements_by_payment_presence={
            payment_present: _categories(
                rows,
                label=f"freight arrangements payment-place-present={payment_present}",
            )
            for payment_present, rows in sorted(freight_arrangements.items())
        },
        freight_payment_sides={
            arrangement: _categories(rows, label=f"freight side {arrangement}")
            for arrangement, rows in sorted(freight_sides.items())
        },
        freight_third_countries={
            arrangement: _categories(rows, label=f"freight third countries {arrangement}")
            for arrangement, rows in sorted(freight_third_countries.items())
        },
        localities_by_country={
            country: tuple(rows) for country, rows in sorted(locality_rows.items())
        },
        maritime_by_country={country: tuple(rows) for country, rows in sorted(maritime.items())},
        observed_loading_ports={
            country: _categories(rows, label=f"observed loading ports {country}")
            for country, rows in sorted(loading_ports.items())
        },
        observed_discharge_ports={
            country: _categories(rows, label=f"observed discharge ports {country}")
            for country, rows in sorted(discharge_ports.items())
        },
        trade_destinations_by_origin=ordered_trade_by_origin,
        audit=audit,
        trade_flow_audit=trade_flow_audit,
    )


def _sample_origin_country(
    support: ScenarioSupport,
    *,
    stream: DeterministicStream,
    exclude: frozenset[str] = frozenset(),
) -> SampledCommercialOrigin:
    observed_weight = support.origin_prior.observed_exporter.mixture_permyriad
    if stream.derive("component").randbelow(10_000) < observed_weight:
        component: OriginPriorComponent = "observed_exporter"
        weighting: OriginPriorWeighting = "train_isolated_shipper_country_document_count_v1"
        rows = support.observed_export_countries
    else:
        component = "maritime_registry"
        weighting = "uniform_route_feasible_iso_country_v1"
        rows = support.maritime_registry_export_countries
    return SampledCommercialOrigin(
        country_code=_weighted_choice(
            rows,
            stream=stream.derive("country"),
            exclude=exclude,
        ),
        component=component,
        weighting=weighting,
    )


def _feasible_physical_loading_rows(
    support: ScenarioSupport,
    *,
    commercial_origin: str,
    commercial_destination: str,
) -> tuple[WeightedCategory, ...]:
    loading_rows = support.physical_loading_countries_by_origin.get(commercial_origin, ())
    discharge_rows = support.physical_discharge_countries_by_destination.get(
        commercial_destination, ()
    )
    return tuple(
        row
        for row in loading_rows
        if row.value != commercial_destination
        and (not discharge_rows or any(other.value != row.value for other in discharge_rows))
    )


def _sample_trade_destination(
    support: ScenarioSupport,
    *,
    commercial_origin: str,
    stream: DeterministicStream,
) -> TradeDestination:
    rows = support.trade_destinations_by_origin.get(commercial_origin, ())
    if not rows:
        raise ValueError(
            f"commercial origin has no pinned bilateral trade-flow support: {commercial_origin}"
        )
    index = weighted_index(tuple(row.weight for row in rows), stream=stream)
    return rows[index]


def _port_location(
    value: RouteLocation,
    *,
    registry: CountryRegistry,
    source: Literal[
        "observed_port",
        "registry_port_exploration",
        "registry_port_no_observed_support",
        "endpoint",
    ],
) -> SampledScenarioLocation:
    return SampledScenarioLocation(
        locode=value.locode,
        geoname_id=None,
        name=value.name,
        country_code=value.country_code,
        country_name=registry.printable_name(value.country_code),
        source=source,
    )


def _sample_port(
    *,
    country: str,
    support: ScenarioSupport,
    country_registry: CountryRegistry,
    observed: Mapping[str, tuple[WeightedCategory, ...]],
    stream: DeterministicStream,
    registry_exploration_permyriad: int,
    excluded_locodes: frozenset[str] = frozenset(),
) -> SampledScenarioLocation:
    if not 0 <= registry_exploration_permyriad <= 10_000:
        raise ValueError("registry_exploration_permyriad must be between 0 and 10000")
    registry_rows = tuple(
        row
        for row in support.maritime_by_country.get(country, ())
        if row.locode not in excluded_locodes
    )
    if not registry_rows:
        raise ValueError(f"country has no eligible distinct maritime port: {country}")
    observed_rows = tuple(
        row for row in observed.get(country, ()) if row.value not in excluded_locodes
    )
    if not observed_rows:
        selected = registry_rows[stream.derive("registry-only").randbelow(len(registry_rows))]
        return _port_location(
            selected,
            registry=country_registry,
            source="registry_port_no_observed_support",
        )
    if stream.derive("mixture").randbelow(10_000) < registry_exploration_permyriad:
        selected = registry_rows[stream.derive("registry").randbelow(len(registry_rows))]
        return _port_location(
            selected, registry=country_registry, source="registry_port_exploration"
        )
    locode = _weighted_choice(observed_rows, stream=stream.derive("observed"))
    index = {row.locode: row for row in registry_rows}
    if locode not in index:
        raise RuntimeError("observed port support differs from registry support")
    return _port_location(index[locode], registry=country_registry, source="observed_port")


def _sample_locality(
    *,
    country: str,
    support: ScenarioSupport,
    country_registry: CountryRegistry,
    stream: DeterministicStream,
    endpoint: SampledScenarioLocation | None = None,
) -> SampledScenarioLocation:
    endpoint_key = (
        (endpoint.country_code, normalize_location_name(endpoint.name))
        if endpoint is not None
        else None
    )
    rows = tuple(
        row
        for row in support.localities_by_country.get(country, ())
        if row.population > 0 and (country, normalize_location_name(row.ascii_name)) != endpoint_key
    )
    if not rows:
        raise ValueError(f"country has no eligible non-endpoint locality: {country}")
    index = weighted_index(tuple(Decimal(row.population) for row in rows), stream=stream)
    selected = rows[index]
    return SampledScenarioLocation(
        locode=None,
        geoname_id=selected.geoname_id,
        name=selected.ascii_name,
        country_code=selected.country_code,
        country_name=country_registry.printable_name(selected.country_code),
        source="geonames_population_weighted",
    )


def _country_for_side(
    side: RouteSide,
    *,
    commercial_origin: str,
    commercial_destination: str,
    third_countries: Sequence[WeightedCategory] | None,
    stream: DeterministicStream,
) -> str | None:
    if side == "missing":
        return None
    if side == "commercial_origin":
        return commercial_origin
    if side == "commercial_destination":
        return commercial_destination
    if not third_countries:
        raise ValueError("third-country relation has no role-specific empirical support")
    return _weighted_choice(
        third_countries,
        stream=stream,
        exclude=frozenset({commercial_origin, commercial_destination}),
    )


def _feasible_side_categories(
    rows: Sequence[WeightedCategory],
    *,
    commercial_origin: str,
    commercial_destination: str,
    third_countries: Sequence[WeightedCategory] | None,
    permit_missing: bool,
) -> tuple[WeightedCategory, ...]:
    """Filter fitted relation classes to those realizable in this scenario.

    This is conditioning, not a fallback: weights are preserved for the subset
    whose required country support exists, and an empty subset is an error.
    """

    third_available = bool(
        third_countries
        and any(
            row.value not in {commercial_origin, commercial_destination} for row in third_countries
        )
    )
    eligible = tuple(
        row
        for row in rows
        if (permit_missing or row.value != "missing")
        and (row.value != "third_country" or third_available)
    )
    if not eligible:
        raise ValueError("no fitted geographic relation is feasible for the sampled route")
    return eligible


def sample_shipment_scenario(
    *,
    base_document_id: str,
    source_target: Mapping[str, Any],
    support: ScenarioSupport,
    country_registry: CountryRegistry,
    stream: DeterministicStream,
    registry_exploration_permyriad: int,
) -> ShipmentScenario:
    """Sample a complete direct-route geography while preserving source leaf topology."""

    patch = _patch(source_target)
    source_route = _location_dict(patch.get("route"))
    if source_route.get("transshipmentPort") is not None:
        raise ValueError("transshipment template requires a pinned connectivity provider")
    sampled_origin = _sample_origin_country(
        support,
        stream=stream.derive("commercial-origin"),
    )
    commercial_origin = sampled_origin.country_code
    trade_destination = _sample_trade_destination(
        support,
        commercial_origin=commercial_origin,
        stream=stream.derive("commercial-destination"),
    )
    commercial_destination = trade_destination.country_code
    loading_rows = _feasible_physical_loading_rows(
        support,
        commercial_origin=commercial_origin,
        commercial_destination=commercial_destination,
    )
    if not loading_rows:
        raise ValueError(
            f"commercial origin lacks empirical loading-country support: {commercial_origin}"
        )
    loading_country = _weighted_choice(
        loading_rows,
        stream=stream.derive("physical-loading-country"),
    )
    discharge_rows = support.physical_discharge_countries_by_destination.get(commercial_destination)
    if discharge_rows:
        discharge_country = _weighted_choice(
            discharge_rows,
            stream=stream.derive("physical-discharge-country"),
            exclude=frozenset({loading_country}),
        )
        discharge_country_method: Literal[
            "train_empirical_conditioned_on_destination_v1",
            "commercial_destination_identity_when_unobserved_v1",
        ] = "train_empirical_conditioned_on_destination_v1"
    else:
        if commercial_destination == loading_country:
            raise ValueError(
                "unobserved discharge country cannot use a non-distinct commercial destination"
            )
        discharge_country = commercial_destination
        discharge_country_method = "commercial_destination_identity_when_unobserved_v1"
    loading = _sample_port(
        country=loading_country,
        support=support,
        country_registry=country_registry,
        observed=support.observed_loading_ports,
        stream=stream.derive("loading-port"),
        registry_exploration_permyriad=registry_exploration_permyriad,
    )
    loading_locode = loading.locode
    if loading_locode is None:
        raise RuntimeError("sampled loading port has no UN/LOCODE identity")
    discharge = _sample_port(
        country=discharge_country,
        support=support,
        country_registry=country_registry,
        observed=support.observed_discharge_ports,
        stream=stream.derive("discharge-port"),
        registry_exploration_permyriad=registry_exploration_permyriad,
        excluded_locodes=frozenset({loading_locode}),
    )
    route_locations: dict[str, SampledScenarioLocation] = {}
    if source_route.get("portOfLoading") is not None:
        route_locations["portOfLoading"] = loading
    if source_route.get("portOfDischarge") is not None:
        route_locations["portOfDischarge"] = discharge
    for role, side, endpoint in (
        ("placeOfReceipt", "commercial_origin", loading),
        ("placeOfDelivery", "commercial_destination", discharge),
        ("finalDestination", "commercial_destination", discharge),
    ):
        if source_route.get(role) is None:
            continue
        source_value = _location_dict(source_route.get(role))
        endpoint_source = _location_dict(
            source_route.get("portOfLoading" if side == "commercial_origin" else "portOfDischarge")
        )
        source_country = _resolved_country(source_value.get("country"), country_registry)
        endpoint_source_country = _resolved_country(
            endpoint_source.get("country"), country_registry
        )
        same_endpoint = (
            source_country is not None
            and source_country == endpoint_source_country
            and isinstance(source_value.get("name"), str)
            and isinstance(endpoint_source.get("name"), str)
            and normalize_location_name(cast(str, source_value["name"]))
            == normalize_location_name(cast(str, endpoint_source["name"]))
        )
        if same_endpoint:
            route_locations[role] = SampledScenarioLocation(
                locode=endpoint.locode,
                geoname_id=None,
                name=endpoint.name,
                country_code=endpoint.country_code,
                country_name=endpoint.country_name,
                source="endpoint",
            )
        else:
            country = commercial_origin if side == "commercial_origin" else commercial_destination
            route_locations[role] = _sample_locality(
                country=country,
                support=support,
                country_registry=country_registry,
                stream=stream.derive(f"route-{role}"),
                endpoint=endpoint,
            )

    party_scenarios: list[PartyLocalityScenario] = []
    party_scenarios_by_reference: dict[tuple[str, int], PartyLocalityScenario] = {}
    concrete_party_identity_owners: dict[tuple[str | None, ...], tuple[str, int]] = {}
    for role, occurrence, party in _party_rows(patch):
        same_as = party.get("sameAs") if isinstance(party.get("sameAs"), str) else None
        country_present = isinstance(party.get("country"), str)
        city_present = isinstance(party.get("city"), str)
        signature = _concrete_party_identity_signature(party)
        concrete_identity_source: ConcretePartyIdentitySource | None = None
        identity_owner: PartyLocalityScenario | None = None
        if same_as is not None:
            if country_present or city_present:
                raise ValueError("sameAs parties cannot carry concrete country or city fields")
            identity_owner = party_scenarios_by_reference.get((same_as, 0))
            if identity_owner is None:
                raise ValueError(f"unsupported or unresolved sameAs party reference: {same_as}")
        elif role == "notifyParties" and signature is not None:
            owner_reference = concrete_party_identity_owners.get(signature)
            if owner_reference is not None:
                identity_owner = party_scenarios_by_reference[owner_reference]
                concrete_identity_source = ConcretePartyIdentitySource(
                    role=owner_reference[0], occurrence=owner_reference[1]
                )

        if identity_owner is not None:
            relation = identity_owner.relation
        elif role == "shipper":
            relation = "commercial_origin"
        elif role == "consignee":
            relation = "commercial_destination"
        else:
            relation_rows = support.party_relations.get(role)
            if not relation_rows:
                raise ValueError(f"party role lacks empirical relation support: {role}")
            feasible_relations: list[WeightedCategory] = []
            for row in relation_rows:
                relation_value = cast(RouteSide, row.value)
                if relation_value == "third_country":
                    third_rows = support.party_third_countries.get(role, ())
                    if not any(
                        value.value not in {commercial_origin, commercial_destination}
                        for value in third_rows
                    ):
                        continue
                if city_present:
                    mode_rows = support.party_locality_modes.get((role, relation_value), ())
                    endpoint_possible = (
                        relation_value == "commercial_origin"
                        and commercial_origin == loading.country_code
                    ) or (
                        relation_value == "commercial_destination"
                        and commercial_destination == discharge.country_code
                    )
                    if not any(
                        mode.value == "other_same_country"
                        or (mode.value == "endpoint" and endpoint_possible)
                        for mode in mode_rows
                    ):
                        continue
                feasible_relations.append(row)
            relation = cast(
                RouteSide,
                _weighted_choice(
                    tuple(feasible_relations),
                    stream=stream.derive(f"party-{role}-{occurrence}-relation"),
                ),
            )
        party_country = (
            identity_owner.conditioning_country_code
            if identity_owner is not None
            else _country_for_side(
                relation,
                commercial_origin=commercial_origin,
                commercial_destination=commercial_destination,
                third_countries=support.party_third_countries.get(role),
                stream=stream.derive(f"party-{role}-{occurrence}-country"),
            )
        )
        if party_country is None:
            raise ValueError("concrete party resolved to no conditioning country")
        if not city_present:
            locality_mode: LocalityMode = "missing"
            locality = None
        elif identity_owner is not None:
            if identity_owner.locality is None:
                raise RuntimeError("repeated concrete identity has no source locality")
            locality_mode = identity_owner.locality_mode
            locality = identity_owner.locality
        else:
            modes = support.party_locality_modes.get((role, relation))
            if modes is None:
                raise ValueError(
                    f"party role/relation lacks locality-mode support: {role}:{relation}"
                )
            party_endpoint: SampledScenarioLocation | None = None
            if relation == "commercial_origin" and party_country == loading.country_code:
                party_endpoint = loading
            elif relation == "commercial_destination" and party_country == discharge.country_code:
                party_endpoint = discharge
            locality_mode = cast(
                LocalityMode,
                _weighted_choice(
                    modes,
                    stream=stream.derive(f"party-{role}-{occurrence}-locality-mode"),
                    exclude=(
                        frozenset({"missing", "endpoint"})
                        if party_endpoint is None
                        else frozenset({"missing"})
                    ),
                ),
            )
            if locality_mode == "endpoint":
                if party_endpoint is None:
                    raise RuntimeError("endpoint mode was selected without an endpoint")
                locality = SampledScenarioLocation(
                    locode=party_endpoint.locode,
                    geoname_id=None,
                    name=party_endpoint.name,
                    country_code=party_endpoint.country_code,
                    country_name=party_endpoint.country_name,
                    source="endpoint",
                )
            else:
                locality = _sample_locality(
                    country=party_country,
                    support=support,
                    country_registry=country_registry,
                    stream=stream.derive(f"party-{role}-{occurrence}-locality"),
                    endpoint=party_endpoint,
                )
        scenario = PartyLocalityScenario(
            role=role,
            occurrence=occurrence,
            same_as=same_as,
            relation=relation,
            country_present=country_present,
            city_present=city_present,
            country_code=party_country if country_present else None,
            country_name=(
                country_registry.printable_name(party_country) if country_present else None
            ),
            conditioning_country_code=party_country,
            conditioning_country_name=country_registry.printable_name(party_country),
            locality_mode=locality_mode,
            locality=locality,
            concrete_identity_source=concrete_identity_source,
        )
        party_scenarios.append(scenario)
        party_scenarios_by_reference[(role, occurrence)] = scenario
        if role in {"shipper", "consignee"} and signature is not None:
            concrete_party_identity_owners.setdefault(signature, (role, occurrence))

    place_of_issue = None
    source_issue = _location_dict(patch.get("placeOfIssue"))
    if source_issue:
        issue_side = cast(
            RouteSide,
            _weighted_choice(
                _feasible_side_categories(
                    support.issue_sides,
                    commercial_origin=commercial_origin,
                    commercial_destination=commercial_destination,
                    third_countries=support.issue_third_countries,
                    permit_missing=False,
                ),
                stream=stream.derive("issue-side"),
            ),
        )
        issue_country = _country_for_side(
            issue_side,
            commercial_origin=commercial_origin,
            commercial_destination=commercial_destination,
            third_countries=support.issue_third_countries,
            stream=stream.derive("issue-country"),
        )
        if issue_country is None:
            raise ValueError("present issue place resolved to a missing side")
        place_of_issue = _sample_locality(
            country=issue_country,
            support=support,
            country_registry=country_registry,
            stream=stream.derive("issue-locality"),
        )

    source_freight = _location_dict(patch.get("freight"))
    arrangement: FreightArrangement | None = None
    payment_place = None
    payment_side: RouteSide = "missing"
    if source_freight:
        payment_place_present = source_freight.get("paymentPlace") is not None
        arrangement_rows = support.freight_arrangements_by_payment_presence.get(
            payment_place_present
        )
        if not arrangement_rows:
            raise ValueError(
                "freight arrangement support is absent for the template payment-place topology"
            )
        arrangement = cast(
            FreightArrangement,
            _weighted_choice(
                arrangement_rows,
                stream=stream.derive("freight-arrangement"),
            ),
        )
        if payment_place_present:
            side_rows = support.freight_payment_sides.get(arrangement)
            if not side_rows:
                raise ValueError(f"freight arrangement lacks payment-side support: {arrangement}")
            payment_side = cast(
                RouteSide,
                _weighted_choice(
                    _feasible_side_categories(
                        side_rows,
                        commercial_origin=commercial_origin,
                        commercial_destination=commercial_destination,
                        third_countries=support.freight_third_countries.get(arrangement),
                        permit_missing=False,
                    ),
                    stream=stream.derive("freight-payment-side"),
                ),
            )
            payment_country = _country_for_side(
                payment_side,
                commercial_origin=commercial_origin,
                commercial_destination=commercial_destination,
                third_countries=support.freight_third_countries.get(arrangement),
                stream=stream.derive("freight-payment-country"),
            )
            if payment_country is None:
                raise ValueError("present freight payment place resolved to a missing side")
            payment_place = _sample_locality(
                country=payment_country,
                support=support,
                country_registry=country_registry,
                stream=stream.derive("freight-payment-locality"),
            )
    transport = _location_dict(patch.get("transport"))
    vessel_present = any(
        transport.get(name) is not None
        for name in ("vesselName", "vesselImoNumber", "vesselFlagCountry")
    )
    return ShipmentScenario(
        base_document_id=base_document_id,
        commercial_origin_country_code=commercial_origin,
        commercial_destination_country_code=commercial_destination,
        loading_port=loading,
        discharge_port=discharge,
        route_locations=route_locations,
        party_localities=tuple(party_scenarios),
        place_of_issue=place_of_issue,
        freight=FreightScenario(
            arrangement=arrangement,
            payment_place_present=source_freight.get("paymentPlace") is not None,
            payment_side=payment_side,
            payment_place=payment_place,
        ),
        trade_flow_year=trade_destination.year,
        trade_flow_weight=trade_destination.weight,
        trade_flow_provider="world_bank_wits_latest_reporter_year_v1",
        origin_prior_method=support.origin_prior.method,
        origin_prior_component=sampled_origin.component,
        origin_prior_weighting=sampled_origin.weighting,
        loading_country_method=support.loading_country_methods_by_origin[commercial_origin],
        discharge_country_method=discharge_country_method,
        transshipment_status="not_present",
        vessel_status=(
            "pending_synthetic_transport_identity" if vessel_present else "not_present"
        ),
    )


def project_scenario_target(
    *, source_target: Mapping[str, Any], scenario: ShipmentScenario
) -> dict[str, Any]:
    """Project geography/freight into source-present leaves only.

    Names, addresses, and contacts remain untouched and consequently this
    target is not training-ready.  It is an auditable input to the later party
    entity and OCR-text realization stages.
    """

    target = cast(dict[str, Any], deepcopy(source_target))
    patch = cast(dict[str, Any], target["documentPatch"])
    source_route = cast(dict[str, Any], patch.get("route") or {})
    for role, location in scenario.route_locations.items():
        source_location = cast(dict[str, Any], source_route[role])
        projected: dict[str, str] = {}
        if source_location.get("name") is not None:
            projected["name"] = location.name
        if source_location.get("country") is not None:
            projected["country"] = location.country_name
        source_route[role] = projected
    if source_route:
        patch["route"] = source_route

    if patch.get("placeOfIssue") is not None and scenario.place_of_issue is not None:
        source_issue = cast(dict[str, Any], patch["placeOfIssue"])
        patch["placeOfIssue"] = {
            **(
                {"name": scenario.place_of_issue.name}
                if source_issue.get("name") is not None
                else {}
            ),
            **(
                {"country": scenario.place_of_issue.country_name}
                if source_issue.get("country") is not None
                else {}
            ),
        }

    parties = cast(dict[str, Any], patch.get("parties") or {})
    for assignment in scenario.party_localities:
        value = parties[assignment.role]
        party = (
            cast(list[dict[str, Any]], value)[assignment.occurrence]
            if assignment.role == "notifyParties"
            else cast(dict[str, Any], value)
        )
        if assignment.country_present:
            party["country"] = assignment.country_name
        if assignment.city_present:
            if assignment.locality is None:
                raise RuntimeError("present party city has no projected locality")
            party["city"] = assignment.locality.name
    if parties:
        patch["parties"] = parties

    if patch.get("freight") is not None:
        freight = cast(dict[str, Any], patch["freight"])
        if freight.get("paymentArrangement") is not None:
            freight["paymentArrangement"] = scenario.freight.arrangement
        if freight.get("paymentPlace") is not None and scenario.freight.payment_place is not None:
            source_payment = cast(dict[str, Any], freight["paymentPlace"])
            freight["paymentPlace"] = {
                **(
                    {"name": scenario.freight.payment_place.name}
                    if source_payment.get("name") is not None
                    else {}
                ),
                **(
                    {"country": scenario.freight.payment_place.country_name}
                    if source_payment.get("country") is not None
                    else {}
                ),
            }
    return target
