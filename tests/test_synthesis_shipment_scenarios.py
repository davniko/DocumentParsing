from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from typing import Any

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.config import RouteScenarioOriginPriorConfig
from document_ocr.synthesis.country_registry import CountryEntry, CountryRegistry
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.locality_registry import LocalityRecord
from document_ocr.synthesis.routes import RouteLocation, TradeFlowRecord
from document_ocr.synthesis.shipment_scenarios import (
    ScenarioSupport,
    _port_location,
    build_scenario_support,
    project_scenario_target,
    sample_shipment_scenario,
)
from document_ocr.synthesis.transshipment_routes import TransshipmentObservation, reviewed_chains
from document_ocr.synthesis.world_port_registry import WorldPortRecord


def _origin_prior(
    *, observed_permyriad: int = 7_500, registry_permyriad: int = 2_500
) -> RouteScenarioOriginPriorConfig:
    return RouteScenarioOriginPriorConfig.model_validate(
        {
            "method": "observed_exporter_plus_maritime_registry_mixture_v1",
            "observed_exporter": {
                "mixture_permyriad": observed_permyriad,
                "weighting": "train_isolated_shipper_country_document_count_v1",
            },
            "maritime_registry": {
                "mixture_permyriad": registry_permyriad,
                "weighting": "uniform_route_feasible_iso_country_v1",
            },
        },
        strict=True,
    )


@pytest.mark.parametrize(
    ("observed", "registry"),
    ((7_500, 2_499), (0, 10_000), (10_000, 0)),
)
def test_origin_prior_requires_two_positive_components_summing_to_10000(
    observed: int, registry: int
) -> None:
    with pytest.raises(ValueError):
        _origin_prior(observed_permyriad=observed, registry_permyriad=registry)


@pytest.mark.parametrize(
    ("component", "field", "value"),
    (
        ("observed_exporter", "weighting", "uniform_route_feasible_iso_country_v1"),
        ("maritime_registry", "weighting", "port_count_v1"),
        ("maritime_registry", "unrecognized", True),
    ),
)
def test_origin_prior_rejects_unknown_weighting_and_fields(
    component: str, field: str, value: object
) -> None:
    payload = _origin_prior().model_dump(mode="python")
    payload[component][field] = value

    with pytest.raises(ValueError):
        RouteScenarioOriginPriorConfig.model_validate(payload, strict=True)


def _country_registry() -> CountryRegistry:
    entries = []
    for code, alpha3, number, name in (
        ("BE", "BEL", "056", "Belgium"),
        ("BR", "BRA", "076", "Brazil"),
        ("EG", "EGY", "818", "Egypt"),
        ("NL", "NLD", "528", "Netherlands"),
        ("ZA", "ZAF", "710", "South Africa"),
    ):
        entries.append(
            CountryEntry.model_validate(
                {"alpha2": code, "alpha3": alpha3, "numeric": number, "name": name},
                strict=True,
            )
        )
    return CountryRegistry(
        entries=entries,
        observed_aliases={},
        iso_sha256="1" * 64,
        observed_aliases_sha256="2" * 64,
    )


def test_sampled_port_preserves_its_own_registry_subdivision_as_private_context():
    port = RouteLocation(
        locode="BRPNG",
        country_code="BR",
        name="Paranagua",
        subdivision_code="PR",
        function_codes=("1",),
    )
    sampled = _port_location(port, registry=_country_registry(), source="observed_port")
    assert sampled.subdivision_code == "PR"
    assert sampled.to_dict()["subdivisionCode"] == "PR"
    no_state = _port_location(
        port.model_copy(update={"subdivision_code": None}),
        registry=_country_registry(),
        source="observed_port",
    )
    assert no_state.to_dict()["subdivisionCode"] is None


def test_observed_country_aliases_are_rejected_at_the_sampling_boundary() -> None:
    base = _country_registry()
    registry = CountryRegistry(
        entries=(base.entry(code) for code in base.country_codes),
        observed_aliases={"A.R. EGYPT": "EG"},
        iso_sha256="1" * 64,
        observed_aliases_sha256="3" * 64,
    )

    with pytest.raises(ValueError, match="alias-free ISO country registry"):
        build_scenario_support(
            source_targets={},
            fit_document_ids=(),
            country_registry=registry,
            route_locations=(),
            world_ports=(),
            localities=(),
            trade_flows=(),
            origin_prior=_origin_prior(),
        )


def _location(locode: str, name: str, functions: tuple[str, ...] = ("1",)) -> RouteLocation:
    return RouteLocation.model_validate(
        {
            "locode": locode,
            "country_code": locode[:2],
            "name": name,
            "function_codes": functions,
            "status": "AA",
        },
        strict=True,
    )


def _flow(origin: str, destination: str, weight: str) -> TradeFlowRecord:
    return TradeFlowRecord.model_validate(
        {
            "origin_country_code": origin,
            "destination_country_code": destination,
            "year": 2024,
            "trade_value": Decimal(weight),
            "net_mass": None,
        },
        strict=True,
    )


def _world_port(location: RouteLocation, number: int) -> WorldPortRecord:
    return WorldPortRecord.model_validate(
        {
            "world_port_index_number": number,
            "source_oid": number,
            "locode": location.locode,
            "country_code": location.country_code,
            "port_name": location.name,
            "alternate_port_name": None,
            "latitude": "1.0",
            "longitude": "1.0",
        },
        strict=True,
    )


def _localities(countries: tuple[str, ...]) -> tuple[LocalityRecord, ...]:
    rows: list[LocalityRecord] = []
    for country_index, country in enumerate(countries, start=1):
        for occurrence in range(2):
            rows.append(
                LocalityRecord.model_validate(
                    {
                        "geoname_id": country_index * 100 + occurrence,
                        "canonical_name": f"City {country} {occurrence}",
                        "ascii_name": f"City {country} {occurrence}",
                        "country_code": country,
                        "feature_code": "PPL",
                        "population": 10_000 - occurrence,
                        "latitude": Decimal("1.0"),
                        "longitude": Decimal("1.0"),
                        "timezone": "Etc/UTC",
                    },
                    strict=True,
                )
            )
    return tuple(rows)


def _target(
    shipper_country: str, consignee_country: str, *, transshipment: bool = False
) -> dict[str, Any]:
    route = {
        "placeOfReceipt": {"name": "Antwerp", "country": "Belgium"},
        "portOfLoading": {"name": "Antwerp", "country": "Belgium"},
        "portOfDischarge": {"name": "Alexandria", "country": "Egypt"},
        "placeOfDelivery": {"name": "Cairo", "country": "Egypt"},
    }
    if transshipment:
        route["transshipmentPort"] = {"name": "Rotterdam", "country": "Netherlands"}
    return {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "billOfLadingNumber": "ABC-1",
            "route": route,
            "placeOfIssue": {"name": "Antwerp", "country": "Belgium"},
            "parties": {
                "shipper": {
                    "name": "SOURCE SHIPPER",
                    "address": "SOURCE ADDRESS",
                    "city": "Antwerp",
                    "country": shipper_country,
                },
                "consignee": {
                    "name": "SOURCE CONSIGNEE",
                    "city": "Cairo",
                    "country": consignee_country,
                },
                "notifyParties": [{"sameAs": "consignee"}],
                "deliveryAgent": {
                    "name": "SOURCE AGENT",
                    "city": "Alexandria",
                    "country": consignee_country,
                },
            },
            "freight": {
                "paymentArrangement": "prepaid",
                "paymentPlace": {"name": "Antwerp", "country": "Belgium"},
            },
            "transport": {"vesselName": "SOURCE VESSEL", "voyageNumber": "001A"},
        },
    }


def _support_with_registry_only_origin(
    *,
    origin_prior: RouteScenarioOriginPriorConfig,
    include_trade_flows: bool = True,
    include_transshipment: bool = False,
) -> tuple[CountryRegistry, dict[str, dict[str, Any]], ScenarioSupport]:
    registry = _country_registry()
    locations = (
        _location("BEANR", "Antwerpen"),
        _location("BEBRU", "Brussels", ("3",)),
        _location("BRSSZ", "Santos"),
        _location("EGALY", "Alexandria"),
        _location("NLRTM", "Rotterdam"),
        _location("ZACPT", "Cape Town"),
    )
    targets: dict[str, dict[str, Any]] = {
        "d1": _target("Belgium", "Egypt"),
        "d2": _target("Brazil", "Egypt"),
        "d3": _target("Netherlands", "Egypt"),
        "d4": _target("South Africa", "Egypt"),
        "d5": _target("Belgium", "Egypt"),
    }
    observations = ()
    if include_transshipment:
        targets["chain-evidence"] = _target("Belgium", "Egypt", transshipment=True)
        observations = (_observation("chain-evidence", targets["chain-evidence"]),)
    support = build_scenario_support(
        source_targets=targets,
        fit_document_ids=tuple(targets),
        country_registry=registry,
        route_locations=locations,
        world_ports=tuple(
            _world_port(location, number) for number, location in enumerate(locations, start=1)
        ),
        localities=_localities(("BE", "BR", "EG", "NL", "ZA")),
        trade_flows=(
            (
                _flow("BE", "EG", "10"),
                _flow("BR", "EG", "20"),
                _flow("EG", "ZA", "50"),
                _flow("NL", "EG", "30"),
                _flow("ZA", "EG", "40"),
            )
            if include_trade_flows
            else ()
        ),
        origin_prior=origin_prior,
        transshipment_observations=observations,
    )
    return registry, targets, support


def _observation(sid: str, target: dict[str, Any]) -> TransshipmentObservation:
    route = target["documentPatch"]["route"]
    payload = {
        "source_document_id": sid,
        "source_target_sha256": sha256_bytes(canonical_json_bytes(target)),
    }
    for field, role, locode in (
        ("port_of_loading", "portOfLoading", "BEANR"),
        ("transshipment_port", "transshipmentPort", "NLRTM"),
        ("port_of_discharge", "portOfDischarge", "EGALY"),
    ):
        payload[field] = dict(
            source_name=route[role]["name"],
            locode=locode,
            evidence="Fixture reviewed source role and registry alias",
        )
    return TransshipmentObservation.model_validate(payload, strict=True)


def test_transshipment_uses_reviewed_joint_chain_and_preserves_label_visibility():
    countries, targets, support = _support_with_registry_only_origin(
        origin_prior=_origin_prior(),
        include_transshipment=True,
    )
    source = deepcopy(targets["chain-evidence"])
    del source["documentPatch"]["route"]["transshipmentPort"]["country"]
    before = deepcopy(source)
    for seed in range(40):
        kwargs = dict(
            base_document_id="new-sample",
            source_target=source,
            support=support,
            country_registry=countries,
            stream=DeterministicStream(seed, "via", "x"),
            registry_exploration_permyriad=2500,
        )
        scenario = sample_shipment_scenario(**kwargs)
        assert scenario == sample_shipment_scenario(**kwargs)
        assert (
            scenario.loading_port.locode,
            scenario.route_locations["transshipmentPort"].locode,
            scenario.discharge_port.locode,
        ) == ("BEANR", "NLRTM", "EGALY")
        assert scenario.transshipment_evidence_document_id == "chain-evidence"
        assert scenario.transshipment_status == "train_observed_three_port_chain_v1"
        target = project_scenario_target(source_target=source, scenario=scenario)
        assert target["documentPatch"]["route"]["transshipmentPort"] == {"name": "Rotterdam"}
    assert source == before
    direct = sample_shipment_scenario(**{**kwargs, "source_target": targets["d1"]})
    assert direct.transshipment_status == "not_present"
    assert "transshipmentPort" not in direct.route_locations


def test_transshipment_missing_evidence_is_explicit_not_direct_fallback():
    countries, _, support = _support_with_registry_only_origin(origin_prior=_origin_prior())
    with pytest.raises(ValueError, match="pinned reviewed three-port"):
        sample_shipment_scenario(
            base_document_id="via",
            source_target=_target("Belgium", "Egypt", transshipment=True),
            support=support,
            country_registry=countries,
            stream=DeterministicStream(1, "via", "x"),
            registry_exploration_permyriad=2500,
        )


@pytest.mark.parametrize("role", ["forwardingAgent", "deliveryAgent", "consolidator"])
def test_repeated_secondary_party_reuses_complete_primary_identity_not_just_notify(role):
    countries, targets, support = _support_with_registry_only_origin(origin_prior=_origin_prior())
    source = deepcopy(targets["d1"])
    source["documentPatch"]["parties"][role] = deepcopy(
        source["documentPatch"]["parties"]["shipper"]
    )
    for seed in range(20):
        scenario = sample_shipment_scenario(
            base_document_id="same-entity",
            source_target=source,
            support=support,
            country_registry=countries,
            stream=DeterministicStream(seed, "shared-party", "x"),
            registry_exploration_permyriad=2500,
        )
        parties = {row.role: row for row in scenario.party_localities}
        assert (
            parties[role].conditioning_country_code == parties["shipper"].conditioning_country_code
        )
        assert parties[role].locality == parties["shipper"].locality
        assert parties[role].concrete_identity_source.role == "shipper"
        target = project_scenario_target(source_target=source, scenario=scenario)
        assert "sameAs" not in target["documentPatch"]["parties"][role]


@pytest.mark.parametrize(
    "corruption", ["fit_scope", "hash", "name", "country", "port", "duplicate"]
)
def test_transshipment_evidence_is_fail_closed(corruption):
    target = _target("Belgium", "Egypt", transshipment=True)
    observation = _observation("train", target)
    ports = {
        r.locode: r
        for r in (
            _location("BEANR", "Antwerpen"),
            _location("NLRTM", "Rotterdam"),
            _location("EGALY", "Alexandria"),
        )
    }
    fit = ["train"]
    payload = observation.model_dump()
    if corruption == "fit_scope":
        fit = ["elsewhere"]
    if corruption == "hash":
        payload["source_target_sha256"] = "0" * 64
    if corruption == "name":
        payload["transshipment_port"]["source_name"] = "Incorrect"
    if corruption == "country":
        target["documentPatch"]["route"]["transshipmentPort"]["country"] = "Belgium"
        payload["source_target_sha256"] = sha256_bytes(canonical_json_bytes(target))
    if corruption == "port":
        del ports["NLRTM"]
    observation = TransshipmentObservation.model_validate(payload, strict=True)
    rows = [observation, observation] if corruption == "duplicate" else [observation]
    with pytest.raises(ValueError):
        reviewed_chains(
            rows,
            source_targets={"train": target},
            fit_document_ids=fit,
            maritime_ports=ports,
            countries=_country_registry(),
        )


def test_transshipment_endpoints_cannot_be_reused_as_via_port():
    payload = _observation("train", _target("Belgium", "Egypt", transshipment=True)).model_dump()
    payload["transshipment_port"]["locode"] = "BEANR"
    with pytest.raises(ValueError, match="three distinct"):
        TransshipmentObservation.model_validate(payload, strict=True)


def test_positive_registry_component_fails_when_route_support_is_empty() -> None:
    with pytest.raises(ValueError, match="maritime-registry origin component"):
        _support_with_registry_only_origin(
            origin_prior=_origin_prior(),
            include_trade_flows=False,
        )


def test_declared_shared_port_and_delivery_sample_once_without_country_labels() -> None:
    registry, targets, support = _support_with_registry_only_origin(origin_prior=_origin_prior())
    source = deepcopy(targets["d1"])
    route = source["documentPatch"]["route"]
    route["portOfDischarge"] = {"name": "Alexandria"}
    route["placeOfDelivery"] = {"name": "Alexandria"}
    route["finalDestination"] = {"name": "Alexandria"}
    for seed in range(24):
        scenario = sample_shipment_scenario(
            base_document_id="d1",
            source_target=source,
            support=support,
            country_registry=registry,
            stream=DeterministicStream(seed, "shared-route", "d1"),
            registry_exploration_permyriad=2500,
            shared_route_locations=(
                ("portOfDischarge", "placeOfDelivery"),
                ("placeOfDelivery", "finalDestination"),
            ),
        )
        result = project_scenario_target(source_target=source, scenario=scenario)
        sampled = result["documentPatch"]["route"]
        assert (
            sampled["portOfDischarge"] == sampled["placeOfDelivery"] == sampled["finalDestination"]
        )
        assert set(sampled["placeOfDelivery"]) == {"name"}
    assert source["documentPatch"]["route"] == route


@pytest.mark.parametrize(
    "groups,patch,error",
    [
        ((("placeOfDelivery", "absent"),), {}, "absent or unsupported"),
        ((("placeOfDelivery", "placeOfDelivery"),), {}, "distinct roles"),
        ((("portOfLoading", "portOfDischarge"),), {}, "endpoints cannot share"),
        ((("portOfDischarge", "placeOfDelivery"),), {}, "source names"),
        (
            (("portOfDischarge", "placeOfDelivery"),),
            {"placeOfDelivery": {"name": "Alexandria", "country": "Belgium"}},
            "source countries",
        ),
        (
            (("portOfDischarge", "placeOfDelivery"),),
            {"placeOfDelivery": {"name": "Alexandria", "country": "UNRESOLVED"}},
            "unresolved",
        ),
        (
            (("portOfDischarge", "placeOfDelivery"),),
            {"placeOfDelivery": {"country": "Egypt"}},
            "lacks printed",
        ),
    ],
)
def test_shared_route_constraints_fail_before_sampling(groups, patch, error) -> None:
    from document_ocr.synthesis.shipment_scenarios import _shared_route_owners

    route = _target("Belgium", "Egypt")["documentPatch"]["route"]
    route.update(patch)
    with pytest.raises(ValueError, match=error):
        _shared_route_owners(route, groups, _country_registry())


def test_registry_origin_support_is_uniform_route_feasible_and_audited() -> None:
    _registry, _targets, support = _support_with_registry_only_origin(origin_prior=_origin_prior())

    assert [(row.value, row.count) for row in support.observed_export_countries] == [
        ("BE", 2),
        ("BR", 1),
        ("NL", 1),
        ("ZA", 1),
    ]
    assert [(row.value, row.count) for row in support.maritime_registry_export_countries] == [
        ("BE", 1),
        ("BR", 1),
        ("EG", 1),
        ("NL", 1),
        ("ZA", 1),
    ]
    assert support.loading_country_methods_by_origin["EG"] == (
        "commercial_origin_identity_registry_v1"
    )
    assert [
        (row.value, row.count) for row in support.physical_loading_countries_by_origin["EG"]
    ] == [("EG", 1)]
    assert support.audit.observed_export_countries == 4
    assert support.audit.maritime_registry_export_countries == 5
    assert support.audit.observed_registry_overlap_countries == 4
    assert support.audit.maritime_registry_only_export_countries == 1
    assert support.audit.world_port_rows_input == 6
    assert support.audit.world_port_rows_excluded_non_maritime == 1
    assert support.audit.world_port_rows_eligible == 5
    assert support.audit.eligible_world_port_locodes == 5
    assert [row.locode for row in support.maritime_by_country["BE"]] == ["BEANR"]


def test_origin_component_is_sampled_first_and_provenance_is_serialized() -> None:
    registry, targets, support = _support_with_registry_only_origin(
        origin_prior=_origin_prior(observed_permyriad=1, registry_permyriad=9_999)
    )

    registry_only = None
    for index in range(64):
        scenario = sample_shipment_scenario(
            base_document_id="d1",
            source_target=targets["d1"],
            support=support,
            country_registry=registry,
            stream=DeterministicStream(29, "origin-mixture-test", str(index)),
            registry_exploration_permyriad=10_000,
        )
        if scenario.commercial_origin_country_code == "EG":
            registry_only = scenario
            break

    assert registry_only is not None
    assert registry_only.origin_prior_component == "maritime_registry"
    assert registry_only.origin_prior_weighting == "uniform_route_feasible_iso_country_v1"
    assert registry_only.loading_country_method == "commercial_origin_identity_registry_v1"
    assert registry_only.loading_port.country_code == "EG"
    assert registry_only.to_dict()["commercialOriginPriorEvidence"] == {
        "method": "observed_exporter_plus_maritime_registry_mixture_v1",
        "component": "maritime_registry",
        "weighting": "uniform_route_feasible_iso_country_v1",
    }


def test_overlapping_origin_retains_both_component_identities() -> None:
    registry, targets, support = _support_with_registry_only_origin(
        origin_prior=_origin_prior(observed_permyriad=5_000, registry_permyriad=5_000)
    )
    be_components = set()

    for index in range(128):
        scenario = sample_shipment_scenario(
            base_document_id="d1",
            source_target=targets["d1"],
            support=support,
            country_registry=registry,
            stream=DeterministicStream(31, "origin-overlap-test", str(index)),
            registry_exploration_permyriad=10_000,
        )
        if scenario.commercial_origin_country_code == "BE":
            be_components.add(scenario.origin_prior_component)

    assert be_components == {"observed_exporter", "maritime_registry"}


@pytest.mark.parametrize("schema_version", ["3.0.0-experimental", "5.0.0-experimental"])
def test_route_first_scenario_is_deterministic_registry_backed_and_topology_preserving(
    schema_version: str,
) -> None:
    registry = _country_registry()
    locations = (
        _location("BEANR", "Antwerpen"),
        _location("BEBRU", "Brussels", ("3",)),
        _location("BRSSZ", "Santos"),
        _location("BRRIO", "Rio de Janeiro", ("3",)),
        _location("EGALY", "Alexandria"),
        _location("EGCAI", "Cairo", ("3",)),
        _location("NLRTM", "Rotterdam"),
        _location("NLAMS", "Amsterdam", ("3",)),
        _location("ZACPT", "Cape Town"),
        _location("ZAJNB", "Johannesburg", ("3",)),
    )
    targets = {
        "d1": _target("Belgium", "Egypt"),
        "d2": _target("Brazil", "Egypt"),
        "d3": _target("Netherlands", "Egypt"),
        "d4": _target("South Africa", "Egypt"),
    }
    for target in targets.values():
        target["schemaVersion"] = schema_version
    world_ports = tuple(
        _world_port(location, number)
        for number, location in enumerate(
            (locations[0], locations[2], locations[4], locations[6], locations[8]),
            start=1,
        )
    )
    localities = _localities(("BE", "BR", "EG", "NL", "ZA"))
    trade_flows = (
        _flow("BE", "EG", "10"),
        _flow("BR", "EG", "20"),
        _flow("NL", "EG", "30"),
        _flow("ZA", "EG", "40"),
    )
    support = build_scenario_support(
        source_targets=targets,
        fit_document_ids=tuple(targets),
        country_registry=registry,
        route_locations=locations,
        world_ports=world_ports,
        localities=localities,
        trade_flows=trade_flows,
        origin_prior=_origin_prior(),
    )
    assert support.audit.input_documents == 4
    assert support.audit.country_clean_documents == 4
    assert support.audit.usable_export_country_documents == 4

    targets_with_unresolved_party = deepcopy(targets)
    targets_with_unresolved_party["d4"]["documentPatch"]["parties"]["deliveryAgent"]["country"] = (
        "UNMAPPED COUNTRY"
    )
    clean_support = build_scenario_support(
        source_targets=targets_with_unresolved_party,
        fit_document_ids=tuple(targets_with_unresolved_party),
        country_registry=registry,
        route_locations=locations,
        world_ports=world_ports,
        localities=localities,
        trade_flows=trade_flows,
        origin_prior=_origin_prior(),
    )
    assert clean_support.audit.country_clean_documents == 3
    assert clean_support.audit.excluded_unresolved_country_documents == 1
    assert clean_support.audit.unresolved_country_cells == 1
    assert clean_support.audit.unresolved_country_issues[0].document_id == "d4"
    assert clean_support.audit.unresolved_country_issues[0].path == (
        "documentPatch.parties.deliveryAgent[0].country"
    )
    stream = DeterministicStream(17, "shipment-scenario-test", "synthetic-1")
    first = sample_shipment_scenario(
        base_document_id="d1",
        source_target=targets["d1"],
        support=support,
        country_registry=registry,
        stream=stream,
        registry_exploration_permyriad=10_000,
    )
    second = sample_shipment_scenario(
        base_document_id="d1",
        source_target=targets["d1"],
        support=support,
        country_registry=registry,
        stream=stream,
        registry_exploration_permyriad=10_000,
    )
    assert first == second
    assert first.loading_port.locode == "BEANR"
    assert first.discharge_port.locode == "EGALY"
    assert first.loading_port.locode != first.discharge_port.locode
    assert first.vessel_status == "pending_synthetic_transport_identity"
    assert first.transshipment_status == "not_present"
    assert first.commercial_destination_country_code == "EG"
    assert first.trade_flow_year == 2024

    projected = project_scenario_target(source_target=targets["d1"], scenario=first)
    assert projected["schemaVersion"] == schema_version
    patch = projected["documentPatch"]
    assert patch["parties"]["shipper"]["name"] == "SOURCE SHIPPER"
    assert patch["parties"]["shipper"]["address"] == "SOURCE ADDRESS"
    assert patch["route"]["portOfLoading"]["name"] == first.loading_port.name
    assert patch["route"]["portOfDischarge"]["country"] == first.discharge_port.country_name
    assert patch["parties"]["notifyParties"][0] == {"sameAs": "consignee"}


@pytest.mark.parametrize("route_present", [True, False])
def test_compiled_route_projection_preserves_carrier_and_missing_endpoint_labels(
    route_present: bool,
) -> None:
    registry, targets, support = _support_with_registry_only_origin(origin_prior=_origin_prior())
    target = deepcopy(targets["d1"])
    target["schemaVersion"] = "5.0.0-experimental"
    carrier = {"name": "Carrier", "country": "Belgium", "city": "Brussels"}
    target["documentPatch"]["parties"]["carrier"] = carrier
    if not route_present:
        target["documentPatch"].pop("route")
    scenario = sample_shipment_scenario(
        base_document_id="d1",
        source_target=target,
        support=support,
        country_registry=registry,
        stream=DeterministicStream(31, "compiled-route-projection", "sample"),
        registry_exploration_permyriad=2500,
    )
    projected = project_scenario_target(source_target=target, scenario=scenario)
    assert projected["documentPatch"]["parties"]["carrier"] == carrier
    assert ("route" in projected["documentPatch"]) is route_present
    assert scenario.loading_port.locode != scenario.discharge_port.locode
    assert not any(p.role == "carrier" for p in scenario.party_localities)


def test_concrete_notify_copy_remains_concrete_and_reuses_consignee_identity_geography() -> None:
    registry, targets, support = _support_with_registry_only_origin(origin_prior=_origin_prior())
    source_target = deepcopy(targets["d1"])
    consignee = source_target["documentPatch"]["parties"]["consignee"]
    consignee["address"] = "SOURCE CONSIGNEE ADDRESS"
    consignee["contactDetails"] = [{"type": "phone", "value": "+20 2 555 0100"}]
    notify = deepcopy(consignee)
    notify["contactDetails"] = [{"type": "email", "value": "notify@example.test"}]
    source_target["documentPatch"]["parties"]["notifyParties"] = [notify]

    scenario = sample_shipment_scenario(
        base_document_id="d1",
        source_target=source_target,
        support=support,
        country_registry=registry,
        stream=DeterministicStream(37, "concrete-notify-copy", "synthetic-1"),
        registry_exploration_permyriad=10_000,
    )
    assignments = {(row.role, row.occurrence): row for row in scenario.party_localities}
    consignee_assignment = assignments[("consignee", 0)]
    notify_assignment = assignments[("notifyParties", 0)]

    assert notify_assignment.same_as is None
    assert notify_assignment.concrete_identity_source is not None
    assert notify_assignment.concrete_identity_source.to_dict() == {
        "role": "consignee",
        "occurrence": 0,
    }
    assert notify_assignment.relation == consignee_assignment.relation
    assert notify_assignment.conditioning_country_code == (
        consignee_assignment.conditioning_country_code
    )
    assert notify_assignment.locality == consignee_assignment.locality

    projected = project_scenario_target(source_target=source_target, scenario=scenario)
    projected_consignee = projected["documentPatch"]["parties"]["consignee"]
    projected_notify = projected["documentPatch"]["parties"]["notifyParties"][0]
    assert "sameAs" not in projected_notify
    for field in ("name", "address", "city", "country"):
        assert projected_notify[field] == projected_consignee[field]
    assert projected_notify["contactDetails"] == [{"type": "email", "value": "notify@example.test"}]
    assert projected_consignee["contactDetails"] == [{"type": "phone", "value": "+20 2 555 0100"}]


def test_contact_only_concrete_party_copy_shares_country_without_unseen_labels():
    registry, targets, support = _support_with_registry_only_origin(origin_prior=_origin_prior())
    target = deepcopy(targets["d1"])
    party = {"name": "Example Cargo", "contactDetails": {"phoneNumbers": ["0101577330"]}}
    target["documentPatch"]["parties"]["consignee"] = party
    target["documentPatch"]["parties"]["notifyParties"] = [deepcopy(party)]
    for seed in range(5):
        scenario = sample_shipment_scenario(
            base_document_id="d1", source_target=target, support=support,
            country_registry=registry, stream=DeterministicStream(seed, "contact-copy", "s"),
            registry_exploration_permyriad=10_000,
        )
        assignments = {(r.role, r.occurrence): r for r in scenario.party_localities}
        consignee, notify = assignments[("consignee", 0)], assignments[("notifyParties", 0)]
        assert notify.concrete_identity_source.role == "consignee"
        assert notify.conditioning_country_code == consignee.conditioning_country_code
        projected = project_scenario_target(source_target=target, scenario=scenario)
        assert projected["documentPatch"]["parties"]["consignee"] == party
        assert projected["documentPatch"]["parties"]["notifyParties"] == [party]


def test_contact_identity_requires_exact_corresponding_evidence():
    from document_ocr.synthesis.shipment_scenarios import _concrete_party_identity_signature

    assert _concrete_party_identity_signature({"name": "Example"}) is None
    first = {"name": "Example", "contactDetails": {"phoneNumbers": ["0101577330"]}}
    second = {"name": "Example", "contactDetails": {"phoneNumbers": ["0101577331"]}}
    assert _concrete_party_identity_signature(first) != _concrete_party_identity_signature(second)


def test_transshipment_templates_are_fail_closed_and_audited() -> None:
    registry = _country_registry()
    locations = (
        _location("BEANR", "Antwerpen"),
        _location("BEBRU", "Brussels", ("3",)),
        _location("BRSSZ", "Santos"),
        _location("BRRIO", "Rio de Janeiro", ("3",)),
        _location("EGALY", "Alexandria"),
        _location("EGCAI", "Cairo", ("3",)),
        _location("NLRTM", "Rotterdam"),
        _location("NLAMS", "Amsterdam", ("3",)),
        _location("ZACPT", "Cape Town"),
        _location("ZAJNB", "Johannesburg", ("3",)),
    )
    targets = {
        "direct-1": _target("Belgium", "Egypt"),
        "direct-2": _target("Brazil", "Egypt"),
        "transshipment": _target("Netherlands", "Egypt", transshipment=True),
    }
    support = build_scenario_support(
        source_targets=targets,
        fit_document_ids=tuple(targets),
        country_registry=registry,
        route_locations=locations,
        world_ports=tuple(
            _world_port(location, number)
            for number, location in enumerate(
                (locations[0], locations[2], locations[4], locations[6], locations[8]),
                start=1,
            )
        ),
        localities=_localities(("BE", "BR", "EG", "NL", "ZA")),
        trade_flows=(
            _flow("BE", "EG", "10"),
            _flow("BR", "EG", "20"),
        ),
        origin_prior=_origin_prior(),
    )
    assert support.audit.excluded_transshipment_documents == 1
    assert support.audit.input_documents == 3


def test_commercial_countries_require_two_distinct_locality_names() -> None:
    registry = _country_registry()
    locations = (
        _location("BEANR", "Antwerpen"),
        _location("BRSSZ", "Santos"),
        _location("EGALY", "Alexandria"),
        _location("NLRTM", "Rotterdam"),
        _location("ZACPT", "Cape Town"),
    )
    targets = {
        "d1": _target("Belgium", "Egypt"),
        "d2": _target("Brazil", "Egypt"),
        "d3": _target("Netherlands", "Egypt"),
        "d4": _target("South Africa", "Egypt"),
    }
    localities = []
    for row in _localities(("BE", "BR", "EG", "NL", "ZA")):
        if row.country_code == "ZA":
            row = LocalityRecord.model_validate(
                {
                    **row.model_dump(mode="python"),
                    "canonical_name": "Same Place",
                    "ascii_name": "SAME-PLACE",
                },
                strict=True,
            )
        localities.append(row)

    support = build_scenario_support(
        source_targets=targets,
        fit_document_ids=tuple(targets),
        country_registry=registry,
        route_locations=locations,
        world_ports=tuple(
            _world_port(location, number) for number, location in enumerate(locations, start=1)
        ),
        localities=tuple(localities),
        trade_flows=(
            _flow("BE", "EG", "10"),
            _flow("BE", "ZA", "5"),
            _flow("BR", "EG", "20"),
            _flow("NL", "EG", "30"),
            _flow("ZA", "EG", "40"),
        ),
        origin_prior=_origin_prior(),
    )

    assert support.audit.export_country_without_locality_depth_documents == 1
    assert support.trade_flow_audit.eligible_records == 3
    assert support.trade_flow_audit.excluded_origin_without_locality_depth_records == 1
    assert support.trade_flow_audit.excluded_destination_without_locality_depth_records == 1
    assert (
        support.trade_flow_audit.excluded_destination_without_distinct_physical_endpoint_records
        == 0
    )
    assert support.trade_flow_audit.to_dict()["excludedDestinationWithoutLocalityDepthRecords"] == 1


def test_trade_destination_requires_distinct_physical_endpoint_support() -> None:
    registry = _country_registry()
    locations = (
        _location("BEANR", "Antwerpen"),
        _location("BRSSZ", "Santos"),
        _location("EGALY", "Alexandria"),
        _location("NLRTM", "Rotterdam"),
        _location("ZACPT", "Cape Town"),
    )
    targets = {
        "d1": _target("Belgium", "Egypt"),
        "d2": _target("Brazil", "Egypt"),
        "d3": _target("Netherlands", "Egypt"),
        "d4": _target("South Africa", "Egypt"),
    }
    targets["d1"]["documentPatch"]["route"]["portOfLoading"] = {
        "name": "Cape Town",
        "country": "South Africa",
    }

    support = build_scenario_support(
        source_targets=targets,
        fit_document_ids=tuple(targets),
        country_registry=registry,
        route_locations=locations,
        world_ports=tuple(
            _world_port(location, number) for number, location in enumerate(locations, start=1)
        ),
        localities=_localities(("BE", "BR", "EG", "NL", "ZA")),
        trade_flows=(
            _flow("BE", "EG", "10"),
            _flow("BE", "ZA", "5"),
            _flow("BR", "EG", "20"),
            _flow("NL", "EG", "30"),
            _flow("ZA", "EG", "40"),
        ),
        origin_prior=_origin_prior(),
    )

    assert (
        support.trade_flow_audit.excluded_destination_without_distinct_physical_endpoint_records
        == 1
    )
    assert [row.country_code for row in support.trade_destinations_by_origin["BE"]] == ["EG"]
