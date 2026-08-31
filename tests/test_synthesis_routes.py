from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.routes import (
    CountryRouteSupport,
    RouteLocation,
    SampledMaritimeRoute,
    TradeFlowRecord,
    build_route_support,
    load_pinned_route_locations,
    load_pinned_trade_flows,
    maritime_locations,
    sample_maritime_route,
    weighted_index,
)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _location(locode: str, name: str, *, functions: tuple[str, ...] = ("1",)) -> RouteLocation:
    return RouteLocation.model_validate(
        {
            "locode": locode,
            "country_code": locode[:2],
            "name": name,
            "function_codes": functions,
        },
        strict=True,
    )


def _flow(
    origin: str,
    destination: str,
    *,
    year: int = 2024,
    trade_value: Decimal | None = Decimal("1"),
    net_mass: Decimal | None = None,
) -> TradeFlowRecord:
    return TradeFlowRecord.model_validate(
        {
            "origin_country_code": origin,
            "destination_country_code": destination,
            "year": year,
            "trade_value": trade_value,
            "net_mass": net_mass,
        },
        strict=True,
    )


def test_pinned_loaders_validate_hash_count_and_strict_records(tmp_path: Path) -> None:
    flow_path = tmp_path / "flows.jsonl"
    flow_sha = _jsonl(
        flow_path,
        [
            {
                "origin_country_code": "BR",
                "destination_country_code": "ZA",
                "year": 2024,
                "trade_value": 12.5,
            }
        ],
    )
    location_path = tmp_path / "locations.jsonl"
    location_sha = _jsonl(
        location_path,
        [
            {
                "locode": "BRSSZ",
                "country_code": "BR",
                "name": "Santos",
                "subdivision_code": "SP",
                "function_codes": ["1", "3"],
                "status": "AA",
            }
        ],
    )

    flows = load_pinned_trade_flows(flow_path, expected_sha256=flow_sha, expected_records=1)
    locations = load_pinned_route_locations(
        location_path, expected_sha256=location_sha, expected_records=1
    )
    assert flows[0].trade_value == Decimal("12.5")
    assert locations[0].locode == "BRSSZ"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_pinned_trade_flows(flow_path, expected_sha256="0" * 64, expected_records=1)
    with pytest.raises(ValueError, match="record count mismatch"):
        load_pinned_route_locations(location_path, expected_sha256=location_sha, expected_records=2)

    invalid_path = tmp_path / "invalid.jsonl"
    invalid_sha = _jsonl(
        invalid_path,
        [
            {
                "origin_country_code": "BR",
                "destination_country_code": "ZA",
                "year": 2024,
                "trade_value": 1,
                "unexpected": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_pinned_trade_flows(invalid_path, expected_sha256=invalid_sha, expected_records=1)


def test_location_contract_and_maritime_filter_are_data_driven() -> None:
    locations = (
        _location("EGALY", "Alexandria", functions=("1", "4")),
        _location("EGCAI", "Cairo", functions=("2", "3", "4")),
        _location("ZACPT", "Cape Town"),
    )
    assert tuple(row.locode for row in maritime_locations(locations)) == (
        "EGALY",
        "ZACPT",
    )

    with pytest.raises(ValidationError, match="country differs"):
        RouteLocation.model_validate(
            {
                "locode": "EGALY",
                "country_code": "ZA",
                "name": "Alexandria",
                "function_codes": ("1",),
            },
            strict=True,
        )
    with pytest.raises(ValidationError, match="unique and sorted"):
        RouteLocation.model_validate(
            {
                "locode": "EGALY",
                "country_code": "EG",
                "name": "Alexandria",
                "function_codes": ("4", "1"),
            },
            strict=True,
        )
    with pytest.raises(ValueError, match="duplicate route location"):
        maritime_locations((locations[0], locations[0]))


def test_route_support_aggregates_flows_and_audits_every_exclusion() -> None:
    locations = (
        _location("BRSSZ", "Santos"),
        _location("ZACPT", "Cape Town"),
        _location("ZADUR", "Durban"),
        _location("EGCAI", "Cairo", functions=("2",)),
    )
    flows = (
        _flow("BR", "ZA", year=2023, trade_value=Decimal("10.25")),
        _flow("BR", "ZA", year=2024, trade_value=Decimal("4.75")),
        _flow("BR", "ZA", year=2020),
        _flow("BR", "ZA", trade_value=None, net_mass=Decimal("4")),
        _flow("BR", "ZA", trade_value=Decimal("0")),
        _flow("BR", "ZA", trade_value=Decimal("0.5")),
        _flow("ZA", "ZA", trade_value=Decimal("3")),
        _flow("BR", "EG", trade_value=Decimal("2")),
    )
    support = build_route_support(
        tuple(reversed(flows)),
        tuple(reversed(locations)),
        weight_field="trade_value",
        year_start=2022,
        year_end=2024,
        minimum_weight=Decimal("1"),
    )

    assert len(support.routes) == 1
    route = support.routes[0]
    assert (route.origin_country_code, route.destination_country_code) == ("BR", "ZA")
    assert route.weight == Decimal("15.00")
    assert route.observations == 2
    assert tuple(row.locode for row in route.destination_locations) == ("ZACPT", "ZADUR")
    assert support.audit.model_dump() == {
        "source_flow_records": 8,
        "source_location_records": 4,
        "maritime_location_records": 3,
        "included_flow_records": 2,
        "included_country_pairs": 1,
        "excluded_year_records": 1,
        "excluded_missing_weight_records": 1,
        "excluded_nonpositive_weight_records": 1,
        "excluded_below_minimum_weight_records": 1,
        "excluded_domestic_records": 1,
        "excluded_country_without_maritime_location_records": 1,
        "excluded_insufficient_domestic_ports_records": 0,
    }

    egypt_enabled = build_route_support(
        (_flow("BR", "EG", trade_value=Decimal("2")),),
        (locations[0], _location("EGALY", "Alexandria")),
        weight_field="trade_value",
        year_start=2024,
        year_end=2024,
    )
    assert egypt_enabled.routes[0].destination_country_code == "EG"


def test_domestic_support_requires_two_ports_and_never_samples_same_port() -> None:
    flow = _flow("ZA", "ZA", trade_value=Decimal("2"))
    with pytest.raises(ValueError, match="no eligible"):
        build_route_support(
            (flow,),
            (_location("ZACPT", "Cape Town"),),
            weight_field="trade_value",
            year_start=2024,
            year_end=2024,
            allow_domestic=True,
        )

    support = build_route_support(
        (flow,),
        (_location("ZACPT", "Cape Town"), _location("ZADUR", "Durban")),
        weight_field="trade_value",
        year_start=2024,
        year_end=2024,
        allow_domestic=True,
    )
    sampled = sample_maritime_route(support, stream=DeterministicStream(17, "routes", "domestic"))
    assert sampled.origin.locode != sampled.destination.locode

    only_port = _location("ZACPT", "Cape Town")
    with pytest.raises(ValidationError, match="at least two maritime locations"):
        CountryRouteSupport(
            origin_country_code="ZA",
            destination_country_code="ZA",
            weight=Decimal("1"),
            observations=1,
            origin_locations=(only_port,),
            destination_locations=(only_port,),
        )
    with pytest.raises(ValidationError, match="function-code-1"):
        SampledMaritimeRoute(
            origin=_location("EGCAI", "Cairo", functions=("2",)),
            destination=_location("ZACPT", "Cape Town"),
            country_pair_weight=Decimal("1"),
            observations=1,
        )


def test_hmac_weighted_sampling_is_exact_deterministic_and_order_independent() -> None:
    locations = (
        _location("BRSSZ", "Santos"),
        _location("ZACPT", "Cape Town"),
        _location("NLRTM", "Rotterdam"),
    )
    flows = (
        _flow("BR", "ZA", trade_value=Decimal("1.25")),
        _flow("BR", "NL", trade_value=Decimal("8.75")),
    )
    first = build_route_support(
        flows,
        locations,
        weight_field="trade_value",
        year_start=2024,
        year_end=2024,
    )
    second = build_route_support(
        tuple(reversed(flows)),
        tuple(reversed(locations)),
        weight_field="trade_value",
        year_start=2024,
        year_end=2024,
    )
    assert first == second

    stream = DeterministicStream(23, "routes", "sample-001")
    assert sample_maritime_route(first, stream=stream) == sample_maritime_route(
        second, stream=stream
    )
    selections = [
        sample_maritime_route(
            first,
            stream=DeterministicStream(23, "routes", f"sample-{index:03d}"),
        ).destination.country_code
        for index in range(200)
    ]
    assert selections.count("NL") > 150
    assert weighted_index(
        (Decimal("0.001"), Decimal("0.009")),
        stream=DeterministicStream(23, "routes", "decimal-weights"),
    ) in {0, 1}


def test_flow_aggregation_and_weight_conversion_do_not_use_decimal_context() -> None:
    # Python's default Decimal context has 28 digits.  Route weights are allowed
    # to exceed that precision after aggregation and must remain exact.
    value = Decimal("999999999999999999999999.123456")
    support = build_route_support(
        (_flow("BR", "ZA", trade_value=value), _flow("BR", "ZA", trade_value=value)),
        (_location("BRSSZ", "Santos"), _location("ZACPT", "Cape Town")),
        weight_field="trade_value",
        year_start=2024,
        year_end=2024,
    )
    assert support.routes[0].weight == Decimal("1999999999999999999999998.246912")
    assert (
        weighted_index(
            (support.routes[0].weight, Decimal("0.000001")),
            stream=DeterministicStream(31, "routes", "high-precision"),
        )
        == 0
    )


@pytest.mark.parametrize(
    "weights",
    ((), (Decimal("0"),), (Decimal("NaN"),), (Decimal("-1"),)),
)
def test_weighted_sampling_rejects_invalid_support(weights: tuple[Decimal, ...]) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        weighted_index(weights, stream=DeterministicStream(1, "routes", "invalid"))
