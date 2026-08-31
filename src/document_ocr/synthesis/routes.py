"""Pinned, deterministic route support for text-only document synthesis.

The provider consumes normalized local artifacts.  Adapting a raw trade or
UN/LOCODE release into these JSONL contracts is deliberately outside this
module: every generation run can therefore pin and audit the exact normalized
inputs without a network dependency or an implicit country allow-list.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal
from functools import reduce
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from document_ocr.synthesis.generators import DeterministicStream

CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
Locode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}[A-Z0-9]{3}$")]
FunctionCode = Annotated[str, StringConstraints(pattern=r"^[1-7B]$")]
NonNegativeWeight = Annotated[
    Decimal,
    Field(ge=0, max_digits=35, decimal_places=11, allow_inf_nan=False),
]
FlowWeightField = Literal["trade_value", "net_mass"]

_MARITIME_FUNCTION = "1"
_READ_CHUNK_SIZE = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_StrictModel = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class TradeFlowRecord(BaseModel):
    """One normalized bilateral flow observation."""

    model_config = _StrictModel

    origin_country_code: CountryCode
    destination_country_code: CountryCode
    year: Annotated[int, Field(ge=1900, le=9999)]
    trade_value: NonNegativeWeight | None = None
    net_mass: NonNegativeWeight | None = None

    @model_validator(mode="after")
    def contains_weight(self) -> TradeFlowRecord:
        if self.trade_value is None and self.net_mass is None:
            raise ValueError("trade flow must contain trade_value or net_mass")
        return self


class RouteLocation(BaseModel):
    """Normalized UN/LOCODE-like location metadata used by route synthesis."""

    model_config = _StrictModel

    locode: Locode
    country_code: CountryCode
    name: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    subdivision_code: (
        Annotated[
            str,
            StringConstraints(pattern=r"^[A-Z0-9][A-Z0-9-]{0,7}$"),
        ]
        | None
    ) = None
    function_codes: tuple[FunctionCode, ...] = Field(min_length=1)
    status: Annotated[str, StringConstraints(min_length=1, max_length=64)] | None = None

    @field_validator("name", "status")
    @classmethod
    def text_has_no_outer_whitespace(cls, value: str | None) -> str | None:
        if value is not None and value != value.strip():
            raise ValueError("location text must not contain outer whitespace")
        return value

    @model_validator(mode="after")
    def locode_and_functions_are_canonical(self) -> RouteLocation:
        if not self.locode.startswith(self.country_code):
            raise ValueError("location country differs from its UN/LOCODE prefix")
        if self.function_codes != tuple(sorted(set(self.function_codes))):
            raise ValueError("location function_codes must be unique and sorted")
        return self


class CountryRouteSupport(BaseModel):
    """One eligible country pair and its concrete maritime locations."""

    model_config = _StrictModel

    origin_country_code: CountryCode
    destination_country_code: CountryCode
    weight: Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
    observations: Annotated[int, Field(gt=0)]
    origin_locations: tuple[RouteLocation, ...] = Field(min_length=1)
    destination_locations: tuple[RouteLocation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def locations_match_route(self) -> CountryRouteSupport:
        if any(
            location.country_code != self.origin_country_code
            or _MARITIME_FUNCTION not in location.function_codes
            for location in self.origin_locations
        ):
            raise ValueError("origin route support contains an ineligible location")
        if any(
            location.country_code != self.destination_country_code
            or _MARITIME_FUNCTION not in location.function_codes
            for location in self.destination_locations
        ):
            raise ValueError("destination route support contains an ineligible location")
        origin_locodes = tuple(location.locode for location in self.origin_locations)
        destination_locodes = tuple(location.locode for location in self.destination_locations)
        if origin_locodes != tuple(sorted(set(origin_locodes))) or (
            destination_locodes != tuple(sorted(set(destination_locodes)))
        ):
            raise ValueError("route locations must be unique and sorted by UN/LOCODE")
        if self.origin_country_code == self.destination_country_code and (
            len(origin_locodes) < 2 or len(destination_locodes) < 2
        ):
            raise ValueError(
                "domestic route support requires at least two maritime locations for each endpoint"
            )
        return self


class RouteSupportAudit(BaseModel):
    """Mutually exclusive flow-filter accounting for one support build."""

    model_config = _StrictModel

    source_flow_records: Annotated[int, Field(ge=0)]
    source_location_records: Annotated[int, Field(ge=0)]
    maritime_location_records: Annotated[int, Field(ge=0)]
    included_flow_records: Annotated[int, Field(ge=0)]
    included_country_pairs: Annotated[int, Field(ge=0)]
    excluded_year_records: Annotated[int, Field(ge=0)]
    excluded_missing_weight_records: Annotated[int, Field(ge=0)]
    excluded_nonpositive_weight_records: Annotated[int, Field(ge=0)]
    excluded_below_minimum_weight_records: Annotated[int, Field(ge=0)]
    excluded_domestic_records: Annotated[int, Field(ge=0)]
    excluded_country_without_maritime_location_records: Annotated[int, Field(ge=0)]
    excluded_insufficient_domestic_ports_records: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def flow_counts_balance(self) -> RouteSupportAudit:
        classified = (
            self.included_flow_records
            + self.excluded_year_records
            + self.excluded_missing_weight_records
            + self.excluded_nonpositive_weight_records
            + self.excluded_below_minimum_weight_records
            + self.excluded_domestic_records
            + self.excluded_country_without_maritime_location_records
            + self.excluded_insufficient_domestic_ports_records
        )
        if classified != self.source_flow_records:
            raise ValueError("route-support flow audit does not balance")
        return self


class RouteSupport(BaseModel):
    """Stable weighted support used by deterministic route sampling."""

    model_config = _StrictModel

    weight_field: FlowWeightField
    year_start: Annotated[int, Field(ge=1900, le=9999)]
    year_end: Annotated[int, Field(ge=1900, le=9999)]
    allow_domestic: bool
    minimum_weight: Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
    routes: tuple[CountryRouteSupport, ...] = Field(min_length=1)
    audit: RouteSupportAudit

    @model_validator(mode="after")
    def routes_are_unique_and_sorted(self) -> RouteSupport:
        keys = tuple(
            (route.origin_country_code, route.destination_country_code) for route in self.routes
        )
        if keys != tuple(sorted(set(keys))):
            raise ValueError("route support must contain unique, sorted country pairs")
        if self.year_start > self.year_end:
            raise ValueError("route support year_start is after year_end")
        if self.audit.included_country_pairs != len(self.routes):
            raise ValueError("route-support pair audit differs from routes")
        return self


class SampledMaritimeRoute(BaseModel):
    """One fully resolved loading/discharge pair."""

    model_config = _StrictModel

    origin: RouteLocation
    destination: RouteLocation
    country_pair_weight: Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
    observations: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def ports_are_distinct(self) -> SampledMaritimeRoute:
        if self.origin.locode == self.destination.locode:
            raise ValueError("sampled route must use distinct locations")
        if _MARITIME_FUNCTION not in self.origin.function_codes or (
            _MARITIME_FUNCTION not in self.destination.function_codes
        ):
            raise ValueError("sampled route endpoints must be function-code-1 locations")
        return self


def _validate_pin(expected_sha256: str, expected_records: int) -> None:
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if expected_records < 0:
        raise ValueError("expected_records must be non-negative")


def _load_pinned_jsonl[RecordT: BaseModel](
    path: Path,
    *,
    expected_sha256: str,
    expected_records: int,
    model: type[RecordT],
    label: str,
) -> tuple[RecordT, ...]:
    _validate_pin(expected_sha256, expected_records)
    if path.is_symlink():
        raise ValueError(f"{label} path must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} path is not a regular file: {resolved}")
    digest = hashlib.sha256()
    rows: list[RecordT] = []
    with resolved.open("rb", buffering=_READ_CHUNK_SIZE) as stream:
        for line_number, encoded in enumerate(stream, 1):
            digest.update(encoded)
            if not encoded.strip():
                raise ValueError(f"{label}:{line_number}: blank JSONL rows are forbidden")
            try:
                rows.append(model.model_validate_json(encoded, strict=True))
            except ValidationError as error:
                raise ValueError(f"{label}:{line_number}: invalid record: {error}") from error
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )
    if len(rows) != expected_records:
        raise ValueError(
            f"{label} record count mismatch: expected {expected_records}, found {len(rows)}"
        )
    return tuple(rows)


def load_pinned_trade_flows(
    path: Path, *, expected_sha256: str, expected_records: int
) -> tuple[TradeFlowRecord, ...]:
    """Load a normalized trade-flow JSONL only when its pin and count match."""

    return _load_pinned_jsonl(
        path,
        expected_sha256=expected_sha256,
        expected_records=expected_records,
        model=TradeFlowRecord,
        label="trade-flow",
    )


def load_pinned_route_locations(
    path: Path, *, expected_sha256: str, expected_records: int
) -> tuple[RouteLocation, ...]:
    """Load normalized UN/LOCODE-like JSONL only when its pin and count match."""

    return _load_pinned_jsonl(
        path,
        expected_sha256=expected_sha256,
        expected_records=expected_records,
        model=RouteLocation,
        label="route-location",
    )


def maritime_locations(locations: Sequence[RouteLocation]) -> tuple[RouteLocation, ...]:
    """Return unique function-code-1 locations in stable UN/LOCODE order."""

    by_locode: dict[str, RouteLocation] = {}
    for location in locations:
        if location.locode in by_locode:
            raise ValueError(f"duplicate route location: {location.locode}")
        by_locode[location.locode] = location
    return tuple(
        by_locode[locode]
        for locode in sorted(by_locode)
        if _MARITIME_FUNCTION in by_locode[locode].function_codes
    )


def build_route_support(
    flows: Sequence[TradeFlowRecord],
    locations: Sequence[RouteLocation],
    *,
    weight_field: FlowWeightField,
    year_start: int,
    year_end: int,
    minimum_weight: Decimal = Decimal(0),
    allow_domestic: bool = False,
) -> RouteSupport:
    """Intersect positive bilateral flows with countries having maritime locations.

    Each source record receives exactly one audit disposition.  Duplicate
    country/year observations are intentionally aggregated into one country-pair
    weight; concrete locations remain uniformly selectable within each country.
    """

    if weight_field not in ("trade_value", "net_mass"):
        raise ValueError("weight_field must be trade_value or net_mass")
    if (
        isinstance(year_start, bool)
        or isinstance(year_end, bool)
        or not isinstance(year_start, int)
        or not isinstance(year_end, int)
        or year_start < 1900
        or year_end > 9999
        or year_start > year_end
    ):
        raise ValueError("route-support years must satisfy 1900 <= start <= end <= 9999")
    if not isinstance(minimum_weight, Decimal) or (
        not minimum_weight.is_finite() or minimum_weight < 0
    ):
        raise ValueError("minimum_weight must be finite and non-negative")
    eligible_locations = maritime_locations(locations)
    by_country: dict[str, list[RouteLocation]] = defaultdict(list)
    for location in eligible_locations:
        by_country[location.country_code].append(location)

    excluded: dict[str, int] = defaultdict(int)
    included_records = 0
    pair_weights: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    pair_observations: dict[tuple[str, str], int] = defaultdict(int)
    for flow in flows:
        if not year_start <= flow.year <= year_end:
            excluded["year"] += 1
            continue
        weight = getattr(flow, weight_field)
        if weight is None:
            excluded["missing_weight"] += 1
            continue
        if weight <= 0:
            excluded["nonpositive_weight"] += 1
            continue
        if weight < minimum_weight:
            excluded["below_minimum_weight"] += 1
            continue
        domestic = flow.origin_country_code == flow.destination_country_code
        if domestic and not allow_domestic:
            excluded["domestic"] += 1
            continue
        origin_locations = by_country.get(flow.origin_country_code, ())
        destination_locations = by_country.get(flow.destination_country_code, ())
        if not origin_locations or not destination_locations:
            excluded["country_without_maritime_location"] += 1
            continue
        if domestic and len(origin_locations) < 2:
            excluded["insufficient_domestic_ports"] += 1
            continue
        key = (flow.origin_country_code, flow.destination_country_code)
        pair_weights[key] = _add_decimals_exact(pair_weights[key], weight)
        pair_observations[key] += 1
        included_records += 1

    routes = tuple(
        CountryRouteSupport(
            origin_country_code=origin,
            destination_country_code=destination,
            weight=pair_weights[(origin, destination)],
            observations=pair_observations[(origin, destination)],
            origin_locations=tuple(by_country[origin]),
            destination_locations=tuple(by_country[destination]),
        )
        for origin, destination in sorted(pair_weights)
    )
    if not routes:
        raise ValueError("route support contains no eligible positive-flow country pairs")
    audit = RouteSupportAudit(
        source_flow_records=len(flows),
        source_location_records=len(locations),
        maritime_location_records=len(eligible_locations),
        included_flow_records=included_records,
        included_country_pairs=len(routes),
        excluded_year_records=excluded["year"],
        excluded_missing_weight_records=excluded["missing_weight"],
        excluded_nonpositive_weight_records=excluded["nonpositive_weight"],
        excluded_below_minimum_weight_records=excluded["below_minimum_weight"],
        excluded_domestic_records=excluded["domestic"],
        excluded_country_without_maritime_location_records=excluded[
            "country_without_maritime_location"
        ],
        excluded_insufficient_domestic_ports_records=excluded["insufficient_domestic_ports"],
    )
    return RouteSupport(
        weight_field=weight_field,
        year_start=year_start,
        year_end=year_end,
        allow_domestic=allow_domestic,
        minimum_weight=minimum_weight,
        routes=routes,
        audit=audit,
    )


def _decimal_coefficient_and_exponent(value: Decimal) -> tuple[int, int]:
    decimal_tuple = value.as_tuple()
    if not isinstance(decimal_tuple.exponent, int):
        raise ValueError("weighted sampling requires finite positive weights")
    coefficient = 0
    for digit in decimal_tuple.digits:
        coefficient = coefficient * 10 + digit
    if decimal_tuple.sign:
        coefficient = -coefficient
    exponent = decimal_tuple.exponent
    while coefficient and coefficient % 10 == 0:
        coefficient //= 10
        exponent += 1
    return coefficient, exponent


def _decimal_from_coefficient(coefficient: int, exponent: int) -> Decimal:
    absolute = str(abs(coefficient))
    return Decimal(
        (
            int(coefficient < 0),
            tuple(int(character) for character in absolute),
            exponent,
        )
    )


def _add_decimals_exact(left: Decimal, right: Decimal) -> Decimal:
    left_coefficient, left_exponent = _decimal_coefficient_and_exponent(left)
    right_coefficient, right_exponent = _decimal_coefficient_and_exponent(right)
    common_exponent = min(left_exponent, right_exponent)
    total = left_coefficient * 10 ** (left_exponent - common_exponent) + right_coefficient * 10 ** (
        right_exponent - common_exponent
    )
    return _decimal_from_coefficient(total, common_exponent)


def _integral_weights(weights: Sequence[Decimal]) -> tuple[int, ...]:
    if not weights or any(not value.is_finite() or value <= 0 for value in weights):
        raise ValueError("weighted sampling requires finite positive weights")
    parts = tuple(_decimal_coefficient_and_exponent(value) for value in weights)
    common_exponent = min(exponent for _, exponent in parts)
    integers = tuple(
        coefficient * 10 ** (exponent - common_exponent) for coefficient, exponent in parts
    )
    divisor = reduce(math.gcd, integers)
    return tuple(value // divisor for value in integers)


def weighted_index(weights: Sequence[Decimal], *, stream: DeterministicStream) -> int:
    """Select an index exactly in Decimal weight space using the HMAC stream."""

    integer_weights = _integral_weights(weights)
    draw = stream.randbelow(sum(integer_weights))
    cumulative = 0
    for index, weight in enumerate(integer_weights):
        cumulative += weight
        if draw < cumulative:
            return index
    raise RuntimeError("weighted sampling cumulative total is inconsistent")


def sample_maritime_route(
    support: RouteSupport, *, stream: DeterministicStream
) -> SampledMaritimeRoute:
    """Sample one country pair by flow weight, then concrete maritime locations."""

    route_index = weighted_index(
        tuple(route.weight for route in support.routes),
        stream=stream.derive("country-pair"),
    )
    route = support.routes[route_index]
    origin_index = stream.derive("origin-location").randbelow(len(route.origin_locations))
    origin = route.origin_locations[origin_index]
    destination_candidates = tuple(
        location for location in route.destination_locations if location.locode != origin.locode
    )
    if not destination_candidates:
        raise RuntimeError("route support cannot provide a distinct destination location")
    destination_index = stream.derive("destination-location").randbelow(len(destination_candidates))
    return SampledMaritimeRoute(
        origin=origin,
        destination=destination_candidates[destination_index],
        country_pair_weight=route.weight,
        observations=route.observations,
    )
