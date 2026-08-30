"""Deterministic, order-independent primitives for constrained B/L synthesis.

These functions do not choose business distributions.  They implement hard
identifiers, chronology, arithmetic, and relationship invariants after a caller
has explicitly selected source-supported parameters.
"""

from __future__ import annotations

import hashlib
import hmac
import string
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, cast

from document_ocr.label_schemas.mpci_bill_of_lading import (
    iso6346_expected_check_digit,
)

_CONTAINER_OWNER_ALPHABET = frozenset(string.ascii_uppercase)
_CONTAINER_CATEGORIES = frozenset("UJZ")


@dataclass(frozen=True, slots=True)
class DeterministicStream:
    """Counter-mode HMAC stream isolated by semantic field path."""

    seed: int
    namespace: str
    identity: str

    def derive(self, field_path: str) -> DeterministicStream:
        if not field_path:
            raise ValueError("field_path must not be empty")
        return DeterministicStream(
            seed=self.seed,
            namespace=self.namespace,
            identity=f"{self.identity}\x1f{field_path}",
        )

    def bytes(self, *, counter: int, length: int) -> bytes:
        if counter < 0 or length <= 0:
            raise ValueError("counter must be non-negative and length must be positive")
        key = hashlib.sha256(f"{self.namespace}\x00{self.seed}".encode()).digest()
        output = bytearray()
        block = counter
        while len(output) < length:
            message = f"{self.identity}\x00{block}".encode()
            output.extend(hmac.new(key, message, hashlib.sha256).digest())
            block += 1
        return bytes(output[:length])

    def randbelow(self, upper: int, *, counter: int = 0) -> int:
        """Uniform rejection sampling without global RNG or scheduling dependence."""

        if upper <= 0:
            raise ValueError("upper must be positive")
        width = max(1, (upper.bit_length() + 7) // 8)
        ceiling = 1 << (width * 8)
        limit = ceiling - ceiling % upper
        attempt = 0
        while True:
            value = int.from_bytes(self.bytes(counter=counter + attempt, length=width), "big")
            if value < limit:
                return value % upper
            attempt += 1


def iso6346_check_digit(prefix_and_serial: str) -> str:
    """Calculate the ISO 6346 check digit for four letters plus six digits."""

    if len(prefix_and_serial) != 10:
        raise ValueError("container body must contain four letters and six digits")
    owner, serial = prefix_and_serial[:4], prefix_and_serial[4:]
    if (
        any(character not in _CONTAINER_OWNER_ALPHABET for character in owner[:3])
        or owner[3] not in _CONTAINER_CATEGORIES
        or not serial.isascii()
        or not serial.isdigit()
    ):
        raise ValueError("container body must match AAA[UJZ]999999")
    return str(iso6346_expected_check_digit(prefix_and_serial))


def validate_container_number(value: str) -> bool:
    if len(value) != 11:
        return False
    try:
        return value[-1] == iso6346_check_digit(value[:-1])
    except ValueError:
        return False


def generate_container_number(
    *,
    owner_and_category: str,
    stream: DeterministicStream,
    excluded: Collection[str] = frozenset(),
) -> str:
    """Generate a valid number from an explicitly supplied, observed owner prefix."""

    if len(owner_and_category) != 4:
        raise ValueError("owner_and_category must contain three owner letters and U, J, or Z")
    # Validate the prefix through the same normative path used by the check digit.
    iso6346_check_digit(owner_and_category + "000000")
    for attempt in range(1_000_000):
        serial = f"{stream.randbelow(1_000_000, counter=attempt):06d}"
        body = owner_and_category + serial
        candidate = body + iso6346_check_digit(body)
        if candidate not in excluded:
            return candidate
    raise RuntimeError("container identifier space exhausted by exclusions")


def generate_unique_container_numbers(
    *,
    owner_by_identity: Mapping[str, str],
    stream: DeterministicStream,
    excluded: Collection[str] = frozenset(),
) -> dict[str, str]:
    """Allocate collision-free containers in stable identity order."""

    if not owner_by_identity or any(not identity for identity in owner_by_identity):
        raise ValueError("container requests require non-empty semantic identities")
    output: dict[str, str] = {}
    used: set[str] = set(excluded)
    for identity in sorted(owner_by_identity):
        candidate = generate_container_number(
            owner_and_category=owner_by_identity[identity],
            stream=stream.derive(identity),
            excluded=used,
        )
        output[identity] = candidate
        used.add(candidate)
    return output


def surface_pattern(value: str) -> str:
    """Encode character-level casing/digit/punctuation without retaining content."""

    if not value:
        raise ValueError("surface pattern source must not be empty")
    return "".join(
        "A"
        if character in string.ascii_uppercase
        else "a"
        if character in string.ascii_lowercase
        else "9"
        if character in string.digits
        else "\\" + character
        for character in value
    )


def generate_from_surface_pattern(
    *,
    pattern: str,
    stream: DeterministicStream,
    excluded: Collection[str] = frozenset(),
    additional_excluded: str | None = None,
) -> str:
    """Generate an identifier while preserving every case class and literal separator."""

    tokens: list[tuple[str, str]] = []
    index = 0
    while index < len(pattern):
        token = pattern[index]
        if token == "\\":
            if index + 1 >= len(pattern):
                raise ValueError("surface pattern ends with an escape")
            tokens.append(("literal", pattern[index + 1]))
            index += 2
        elif token in {"A", "a", "9"}:
            tokens.append((token, token))
            index += 1
        else:
            raise ValueError("surface pattern contains an unescaped literal")
    variable_positions = sum(kind != "literal" for kind, _ in tokens)
    literal_value = "".join(value for _, value in tokens)
    if variable_positions == 0 and (
        literal_value in excluded or literal_value == additional_excluded
    ):
        raise ValueError("literal-only surface pattern cannot avoid the excluded value")
    alphabets = {"A": string.ascii_uppercase, "a": string.ascii_lowercase, "9": string.digits}
    for attempt in range(10_000):
        output: list[str] = []
        variable_index = 0
        for kind, literal in tokens:
            if kind == "literal":
                output.append(literal)
                continue
            alphabet = alphabets[kind]
            derived = stream.derive(f"character-{variable_index}")
            output.append(alphabet[derived.randbelow(len(alphabet), counter=attempt)])
            variable_index += 1
        candidate = "".join(output)
        if candidate not in excluded and candidate != additional_excluded:
            return candidate
    raise RuntimeError("surface-pattern identifier space exhausted by exclusions")


def generate_seal_identifier(
    *, source: str, stream: DeterministicStream, excluded: Collection[str] = frozenset()
) -> str:
    """Generate one scalar seal while rejecting compound or whitespace-delimited evidence."""

    if not source.isascii() or any(character.isspace() for character in source):
        raise ValueError("simple seal generation requires one whitespace-free ASCII value")
    return generate_from_surface_pattern(
        pattern=surface_pattern(source),
        stream=stream,
        excluded=excluded,
        additional_excluded=source,
    )


def generate_unique_seal_identifiers(
    *,
    source_by_identity: Mapping[str, str],
    stream: DeterministicStream,
    excluded: Collection[str] = frozenset(),
) -> dict[str, str]:
    """Allocate collision-free scalar seals in stable identity order."""

    if not source_by_identity or any(not identity for identity in source_by_identity):
        raise ValueError("seal requests require non-empty semantic identities")
    output: dict[str, str] = {}
    used: set[str] = set(excluded) | set(source_by_identity.values())
    for identity in sorted(source_by_identity):
        candidate = generate_seal_identifier(
            source=source_by_identity[identity],
            stream=stream.derive(identity),
            excluded=used,
        )
        output[identity] = candidate
        used.add(candidate)
    return output


def shift_document_dates(
    *,
    issue_date: date | None,
    shipped_on_board_date: date | None,
    minimum: date,
    maximum: date,
    stream: DeterministicStream,
) -> tuple[date | None, date | None]:
    """Apply one bounded offset, preserving missingness and the exact date interval."""

    present = [value for value in (issue_date, shipped_on_board_date) if value is not None]
    if not present:
        return None, None
    if minimum > maximum:
        raise ValueError("minimum date is after maximum date")
    lower = max((minimum - value).days for value in present)
    upper = min((maximum - value).days for value in present)
    if lower > upper:
        raise ValueError("no shared date offset fits the requested window")
    offset = lower + stream.randbelow(upper - lower + 1)
    return (
        issue_date + timedelta(days=offset) if issue_date is not None else None,
        shipped_on_board_date + timedelta(days=offset)
        if shipped_on_board_date is not None
        else None,
    )


def largest_remainder_allocation(total: int, weights: Sequence[int]) -> tuple[int, ...]:
    """Apportion an integer exactly while preserving observed proportions."""

    if total < 0 or not weights or any(weight < 0 for weight in weights):
        raise ValueError("total and weights must be non-negative, with at least one weight")
    weight_total = sum(weights)
    if weight_total == 0:
        if total != 0:
            raise ValueError("positive total cannot be allocated from all-zero weights")
        return tuple(0 for _ in weights)
    numerators = [total * weight for weight in weights]
    bases = [numerator // weight_total for numerator in numerators]
    remainder = total - sum(bases)
    ranking = sorted(
        range(len(weights)), key=lambda index: (-(numerators[index] % weight_total), index)
    )
    for index in ranking[:remainder]:
        bases[index] += 1
    return tuple(bases)


def generate_quantity(
    *,
    minimum: int,
    maximum: int,
    stream: DeterministicStream,
    excluded: frozenset[int] = frozenset(),
) -> int:
    """Sample an integer from caller-approved support without a hidden distribution."""

    if minimum < 0 or maximum < minimum:
        raise ValueError("quantity bounds must be non-negative and ordered")
    support = maximum - minimum + 1
    if len({value for value in excluded if minimum <= value <= maximum}) >= support:
        raise ValueError("quantity support is exhausted by exclusions")
    for attempt in range(support * 2):
        candidate = minimum + stream.randbelow(support, counter=attempt)
        if candidate not in excluded:
            return candidate
    # Rejection can revisit values; a deterministic scan makes exhaustion/progress explicit.
    start = stream.randbelow(support, counter=support * 2)
    for offset in range(support):
        candidate = minimum + (start + offset) % support
        if candidate not in excluded:
            return candidate
    raise RuntimeError("unreachable quantity exhaustion")


def reconcile_allocation_group(
    *, packages: Sequence[Mapping[str, Any]], allocation_group: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild allocation quantities after package quantities have changed."""

    package_by_id = {cast(str, row["packageId"]): row for row in packages}
    output = deepcopy(dict(allocation_group))
    allocations = [
        deepcopy(dict(row)) for row in cast(Sequence[Mapping[str, Any]], output["allocations"])
    ]
    package_ids = cast(list[str], list(output.get("packageIds") or []))
    coverage = output["coverage"]
    if coverage == "one_to_one_package_allocations":
        if package_ids != [row.get("packageId") for row in allocations]:
            raise ValueError("one-to-one package and allocation order differs")
        for row in allocations:
            package = package_by_id.get(cast(str, row["packageId"]))
            if package is None or package.get("quantity") is None:
                raise ValueError("one-to-one allocation references a package without quantity")
            row["packageQuantity"] = package["quantity"]
    elif coverage in {"single_package_level", "all_package_levels_combined"}:
        if coverage == "single_package_level" and len(package_ids) != 1:
            raise ValueError("single-package coverage must reference exactly one package")
        if coverage == "all_package_levels_combined" and len(package_ids) < 2:
            raise ValueError("combined coverage must reference multiple packages")
        quantities = [package_by_id[package_id].get("quantity") for package_id in package_ids]
        if any(value is None for value in quantities):
            raise ValueError("covered package level lacks quantity")
        total = sum(cast(int, value) for value in quantities)
        weights = [cast(int, row.get("packageQuantity")) for row in allocations]
        if any(value is None for value in (row.get("packageQuantity") for row in allocations)):
            raise ValueError("covered allocation lacks its source proportion")
        values = largest_remainder_allocation(total, weights)
        for row, value in zip(allocations, values, strict=True):
            row["packageQuantity"] = value
    elif coverage == "unlinked_package_quantities":
        if package_ids or any(row.get("packageQuantity") is None for row in allocations):
            raise ValueError("unlinked allocation shape is invalid")
    elif coverage == "container_membership_only":
        if package_ids or any(
            row.get("packageQuantity") is not None or row.get("packageId") is not None
            for row in allocations
        ):
            raise ValueError("membership-only allocation contains package facts")
    else:
        raise ValueError(f"unsupported allocation coverage: {coverage!r}")
    output["allocations"] = allocations
    return output


def scale_cargo_measure(
    measure: Mapping[str, Any] | None,
    *,
    numerator: int,
    denominator: int,
    decimal_places: int,
) -> dict[str, Any] | None:
    """Scale a positive measure using decimal arithmetic and explicit precision."""

    if measure is None:
        return None
    if numerator <= 0 or denominator <= 0 or decimal_places < 0:
        raise ValueError("scale factors must be positive and decimal_places non-negative")
    value = Decimal(str(measure["value"])) * Decimal(numerator) / Decimal(denominator)
    quantum = Decimal(1).scaleb(-decimal_places)
    rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    if not rounded.is_finite() or rounded <= 0:
        raise ValueError("scaled measure is not finite and positive")
    return {"value": float(rounded), "unit": measure["unit"]}


def validate_mass_order(
    *, gross_weight: Mapping[str, Any] | None, net_weight: Mapping[str, Any] | None
) -> None:
    """Guard the comparable same-unit gross >= net physical invariant."""

    if gross_weight is None or net_weight is None:
        return
    factors = {
        "kilogram": Decimal("1"),
        "metric_tonne": Decimal("1000"),
        "pound": Decimal("0.45359237"),
    }
    try:
        gross = Decimal(str(gross_weight["value"])) * factors[cast(str, gross_weight["unit"])]
        net = Decimal(str(net_weight["value"])) * factors[cast(str, net_weight["unit"])]
    except KeyError as error:
        raise ValueError(f"unsupported mass unit: {error.args[0]!r}") from error
    if gross < net:
        raise ValueError("gross weight is below net weight")
