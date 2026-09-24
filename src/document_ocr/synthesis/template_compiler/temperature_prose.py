"""Project temperature prose only through a compiled text/setpoint dependency."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .models import CertifiedSemanticTemplate

_MENTION = re.compile(
    r"(?<![A-Za-z0-9.,])(?P<number>[-+]?\d+(?:[.,]\d+)?)\s*"
    r"(?:°\s*|DEGREES?\s+)?(?P<unit>CELSIUS|FAHRENHEIT|KELVIN|C|F|K)\b",
    re.I,
)
_UNITS = {
    "C": "celsius",
    "CELSIUS": "celsius",
    "F": "fahrenheit",
    "FAHRENHEIT": "fahrenheit",
    "K": "kelvin",
    "KELVIN": "kelvin",
}
_INTERVAL = re.compile(
    r"\s*(?:TEMPERATURE|TEMP\.?)\s*[:=]?\s*(?P<low>[+-]?\d+(?:\.\d+)?)"
    r"\s*(?:-|\u2013|TO)\s*(?P<high>[+-]?\d+(?:\.\d+)?)"
    r"\s*(?:°\s*)?(?P<unit>C|CELSIUS|F|FAHRENHEIT|K|KELVIN)?\s*",
    re.I,
)
_WORD_INTERVAL = re.compile(
    r"\s*(?:STOWED\s+IN\s+(?:A\s+)?REEFER\s+CONTAINER\s+AT\s+)?"
    r"(?:TEMPERATURE|TEMP\.?)\s*(?:OF\s+)?[:=]?\s*"
    r"(?P<low_sign>PLUS|MINUS)\s*(?P<low>\d+(?:[.,]\d+)?)\s*"
    r"(?:°\s*)?(?P<low_unit>CELSIUS|FAHRENHEIT|KELVIN|C|F|K)\s*"
    r"(?:TILL|TO)\s*(?P<high_sign>PLUS|MINUS)\s*"
    r"(?P<high>\d+(?:[.,]\d+)?)\s*(?:°\s*)?"
    r"(?P<high_unit>CELSIUS|FAHRENHEIT|KELVIN|C|F|K)\s*",
    re.I,
)
_YARN_CONSTRUCTION = re.compile(r"\b\d+(?:\.\d+)?(?:D|DTEX)/\d+F\b", re.I)
_SETTING_CAPTION = re.compile(
    r"\b(?:TEMP(?:ERATURE)?\.?|SET\s+AT|MAINTAIN\s+AT|"
    r"(?:GOODS|CARGO)\s+(?:(?:ARE|IS)\s+)?SHIPPED\s+AT)\b",
    re.I,
)
_OTHER_TEMPERATURE_PROPERTY = re.compile(
    r"\b(?:FLASH|MELTING|BOILING|IGNITION|FREEZING)\s*(?:POINT|TEMP(?:ERATURE)?)\b",
    re.I,
)
_CARRYING_SETTING = re.compile(
    r"\b(?:SET\s+TEMP(?:ERATURE)?|CARRYING\s+TEMPERATURE|REEFER\s+SETTINGS)"
    r"(?:\s+CONT\.\s+NOS\.?)?\s*(?:OF\s+)?[:=]?\s*"
    r"(?P<value>[+-]?\d+(?:[.,]\d+)?)\s*(?:°\s*|DEGREES?\s+)?"
    r"(?:CELSIUS|FAHRENHEIT|KELVIN|C|F|K)\b",
    re.I,
)


def require_source_setting_owners(
    source: bytes, template: CertifiedSemanticTemplate, source_target: Mapping[str, Any]
) -> None:
    """Do not treat a literal carrying setting as unrestricted flavor text.

    This checks explicit carrying captions, not flash points or operational
    gate measurements. Unlabelled settings require a represented private
    physical contract before admission; a source-fixed numeric slot alone is
    not such a contract. No units or setpoints are invented in training labels.
    """
    text = source.decode("utf-8")
    matches = tuple(_CARRYING_SETTING.finditer(text))
    if not matches:
        return
    prose_owners = {contract.path for contract in cargo_setpoint_contracts(template, source_target)}
    prose_owners.update(interval.path for interval in intervals(template, source_target))
    for match in matches:
        start = len(text[: match.start("value")].encode("utf-8"))
        end = len(text[: match.end("value")].encode("utf-8"))
        owners = [
            binding
            for binding in template.bindings
            if any(
                slot.byte_start <= start and end <= slot.byte_end for slot in binding.occurrences
            )
        ]
        if not any(
            any(".temperatureSetpoint.value" in p for p in (*b.target_paths, *b.dependency_paths))
            or bool(prose_owners.intersection(b.target_paths))
            for b in owners
        ):
            raise ValueError(
                "source-only carrying-temperature setting lacks physical ownership: "
                + match.group()
            )


def _mentions(text: str) -> tuple[re.Match[str], ...]:
    """Exclude typed yarn construction, not arbitrary unfamiliar unit suffixes."""
    yarn = tuple(m.span() for m in _YARN_CONSTRUCTION.finditer(text))
    return tuple(
        m
        for m in _MENTION.finditer(text)
        if not any(start <= m.start() and m.end() <= end for start, end in yarn)
    )


@dataclass(frozen=True)
class CargoSetpointContract:
    path: str
    text: str
    container_indices: tuple[int, ...]
    value: Decimal
    unit: str


def cargo_setpoint_contracts(
    template: CertifiedSemanticTemplate, source: Mapping[str, Any]
) -> tuple[CargoSetpointContract, ...]:
    """Join a cargo's explicit carrying instruction to its complete allocation.

    Text and setpoints must both be bound printed targets. Equal numbers alone
    do not prove ownership: the instruction must name temperature/setting, and
    every container carrying that cargo must expose the same value and unit.
    Storage ranges and already composite-bound instructions use their own rules.
    """
    from document_ocr.synthesis.semantic_completion_pipeline import _group_allocations

    patch = source["documentPatch"]
    groups, containers = patch.get("cargoGroups", []), patch.get("containers", [])
    if not any("temperatureSetpoint" in c for c in containers):
        return ()
    paths = {p for b in template.bindings for p in b.target_paths}
    explicit = {
        p
        for b in template.bindings
        if any(".temperatureSetpoint." in p for p in b.target_paths)
        for p in b.target_paths
    }
    allocation = _group_allocations(patch) if len(groups) > 1 else {}
    contracts = []
    for gi, group in enumerate(groups):
        numbers = allocation.get(group["groupId"], ())
        owners = tuple(
            i
            for i, c in enumerate(containers)
            if len(groups) == 1 or c.get("containerNumber") in numbers
        )
        if not owners:
            continue
        for field in ("handlingInstructions", "additionalInformation"):
            for index, text in enumerate(group.get(field, [])):
                path = f"documentPatch.cargoGroups[{gi}].{field}[{index}]"
                if (
                    path not in paths
                    or path in explicit
                    or not _SETTING_CAPTION.search(text)
                    or _OTHER_TEMPERATURE_PROPERTY.search(text)
                ):
                    continue
                if _INTERVAL.fullmatch(text) or re.search(
                    r"\b(?:PLUS|MINUS|TILL|BETWEEN)\b", text, re.I
                ):
                    continue
                mentions = _mentions(text)
                if not mentions:
                    continue
                setting = containers[owners[0]].get("temperatureSetpoint")
                if setting is None or any(
                    containers[i].get("temperatureSetpoint") != setting for i in owners
                ):
                    continue
                if any(
                    f"documentPatch.containers[{i}].temperatureSetpoint.{key}" not in paths
                    for i in owners
                    for key in ("value", "unit")
                ):
                    continue
                value = Decimal(str(setting["value"]))
                if any(
                    Decimal(m["number"].replace(",", ".")) != value
                    or _UNITS[m["unit"].upper()] != setting["unit"]
                    for m in mentions
                ):
                    continue
                contracts.append(CargoSetpointContract(path, text, owners, value, setting["unit"]))
    return tuple(contracts)


def shared_setpoint_owners(
    template: CertifiedSemanticTemplate, source: Mapping[str, Any]
) -> dict[int, tuple[int, ...]]:
    """Connected equality components, consumed before independent temperature draws."""
    components: list[set[int]] = []
    owners = [c.container_indices for c in cargo_setpoint_contracts(template, source)]
    owners.extend(summary_setpoint_owners(template, source))
    owners.extend(direct_shared_setpoint_owners(template, source))
    for indices in owners:
        current = set(indices)
        overlapping = [group for group in components if group & current]
        for group in overlapping:
            current.update(group)
            components.remove(group)
        components.append(current)
    return {i: tuple(sorted(group)) for group in components for i in group}


def direct_shared_setpoint_owners(
    template: CertifiedSemanticTemplate, source: Mapping[str, Any]
) -> tuple[tuple[int, ...], ...]:
    """Carry direct shared-value bindings into joint physical sampling.

    Rendering already rejects unequal target values for one shared binding, but
    the sampler needs the equality *before* it draws the container settings.
    """
    groups = set()
    printed = {path for binding in template.bindings for path in binding.target_paths}
    containers = source["documentPatch"]["containers"]
    for binding in template.bindings:
        if getattr(binding, "target_relationship", None) != "shared_value_equality":
            continue
        matches = tuple(
            re.fullmatch(r"documentPatch\.containers\[(\d+)\]\.temperatureSetpoint\.value", path)
            for path in binding.target_paths
        )
        if not any(matches):
            continue
        if not all(matches):
            raise ValueError("direct shared temperature binding mixes unrelated target paths")
        indices = tuple(sorted({int(match[1]) for match in matches if match is not None}))
        if len(indices) < 2 or any(index >= len(containers) for index in indices):
            raise ValueError("direct shared temperature binding has invalid owners")
        if any(
            f"documentPatch.containers[{index}].temperatureSetpoint.unit" not in printed
            for index in indices
        ):
            raise ValueError("direct shared temperature binding lacks printed units")
        settings = [containers[index].get("temperatureSetpoint") for index in indices]
        if not isinstance(settings[0], Mapping) or any(
            setting != settings[0] for setting in settings
        ):
            raise ValueError("direct shared temperature source setpoints disagree")
        if not Decimal(str(settings[0]["value"])).is_finite():
            raise ValueError("direct shared temperature source setpoint is non-finite")
        groups.add(indices)
    return tuple(sorted(groups))


def summary_setpoint_owners(
    template: CertifiedSemanticTemplate, source: Mapping[str, Any]
) -> tuple[tuple[int, ...], ...]:
    """Use explicit summary dependencies, never equality of unrelated numbers.

    A source-only scalar summary can represent several containers only when
    their complete printed source setpoints agree. Its equality constraint must
    reach the sampler before it chooses independent container temperatures.
    """
    from . import descendant as render

    groups = set()
    printed = {p for b in template.bindings for p in b.target_paths}
    for binding in template.bindings:
        if binding.derivation != "same_as_binding" or binding.target_paths:
            continue
        paths = binding.dependency_paths
        matches = [
            re.fullmatch(r"documentPatch\.containers\[(\d+)\]\.temperatureSetpoint\.value", p)
            for p in paths
        ]
        if not any(matches):
            continue
        if not all(matches):
            raise ValueError("temperature summary mixes incompatible dependency paths")
        indices = tuple(sorted({int(m[1]) for m in matches if m is not None}))
        if any(
            f"documentPatch.containers[{i}].temperatureSetpoint.{part}" not in printed
            for i in indices
            for part in ("value", "unit")
        ):
            raise ValueError("temperature summary requires complete printed setpoint owners")
        settings = [
            source["documentPatch"]["containers"][i]["temperatureSetpoint"] for i in indices
        ]
        if any(s != settings[0] for s in settings):
            raise ValueError("temperature summary source owners disagree")
        for path in paths:
            if not Decimal(str(render._resolve_path(source, path))).is_finite():
                raise ValueError("temperature summary requires finite setpoints")
        groups.add(indices)
    return tuple(sorted(groups))


def _render_number(text: str, mentions: Sequence[re.Match[str]], new: Decimal) -> str:
    output = text
    for match in reversed(mentions):
        raw = match["number"]
        places = max(
            len(raw.split(".")[-1]) if "." in raw else len(raw.split(",")[-1]) if "," in raw else 0,
            max(0, -int(new.normalize().as_tuple().exponent)),
        )
        value = format(new, f".{places}f")
        if raw.startswith("+") and new >= 0:
            value = "+" + value
        if "," in raw:
            value = value.replace(".", ",")
        output = output[: match.start("number")] + value + output[match.end("number") :]
    return output


@dataclass(frozen=True)
class TemperatureInterval:
    path: str
    text: str
    low_celsius: float
    high_celsius: float
    container_indices: tuple[int, ...]


def _celsius(value: float, unit: str) -> float:
    if unit == "celsius":
        return value
    if unit == "fahrenheit":
        return (value - 32) / 1.8
    if unit == "kelvin":
        return value - 273.15
    raise ValueError("temperature interval has an unsupported unit")


def _interval_values(text: str) -> tuple[float, float, str | None] | None:
    match = _INTERVAL.fullmatch(text)
    if match is not None:
        return float(match["low"]), float(match["high"]), match["unit"]
    # Words spelling a sign are not free-form linguistic instructions. Parse
    # only the complete carrying-temperature grammar and preserve its bounds,
    # including an equal-endpoint interval; never widen it to draw more goods.
    match = _WORD_INTERVAL.fullmatch(text)
    if match is None:
        return None
    if _UNITS[match["low_unit"].upper()] != _UNITS[match["high_unit"].upper()]:
        raise ValueError("temperature interval endpoints use different units")
    low, high = (
        float(match[key].replace(",", ".")) * (-1 if match[key + "_sign"].upper() == "MINUS" else 1)
        for key in ("low", "high")
    )
    return low, high, match["low_unit"]


def require_temperature_classification(bindings: Sequence[Any]) -> None:
    """An unowned numeric temperature must not become a random opaque code.

    Direct model numbers/marks retain their explicit target ownership. Ambiguous
    source-only cargo/equipment codes with this exact grammar require review,
    rather than treating a Celsius/Fahrenheit/Kelvin suffix as a random letter.
    """
    for binding in bindings:
        if binding.target_paths or binding.group_kind not in {"cargo", "equipment"}:
            continue
        if binding.value_kind not in {"operational_text", "identifier", "other_text"}:
            continue
        surfaces = (
            [s.source_text for s in binding.occurrences]
            if hasattr(binding, "occurrences")
            else [binding.source_text]
        )
        if any(_MENTION.fullmatch(text.strip()) for text in surfaces):
            raise ValueError(
                "source-only temperature-like surface requires typed physical review: "
                + binding.logical_key
            )


def intervals(
    template: CertifiedSemanticTemplate, source: Mapping[str, Any]
) -> tuple[TemperatureInterval, ...]:
    """Prove closed storage intervals through labelled cargo and its allocation.

    Unitless intervals require one unit shared by all owned printed setpoints.
    No missing setpoint or equipment label is inferred here.
    """
    from document_ocr.synthesis.semantic_completion_pipeline import _group_allocations

    patch = source["documentPatch"]
    groups = patch.get("cargoGroups", [])
    containers = patch.get("containers", [])
    paths = {p for binding in template.bindings for p in binding.target_paths}
    found = []
    for index, group in enumerate(groups):
        for field in ("handlingInstructions", "additionalInformation"):
            for item, text in enumerate(group.get(field, [])):
                parsed = _interval_values(text)
                path = f"documentPatch.cargoGroups[{index}].{field}[{item}]"
                if parsed is None or path not in paths:
                    continue
                if len(groups) == 1:
                    owners = tuple(range(len(containers)))
                else:
                    numbers = _group_allocations(patch).get(group["groupId"], ())
                    owners = tuple(
                        i for i, c in enumerate(containers) if c["containerNumber"] in numbers
                    )
                if not owners or any("temperatureSetpoint" not in containers[i] for i in owners):
                    raise ValueError(
                        "temperature interval lacks complete physical dependency "
                        "on printed setpoints"
                    )
                settings = [containers[i]["temperatureSetpoint"] for i in owners]
                units = {s["unit"] for s in settings}
                low_value, high_value, explicit_unit = parsed
                if explicit_unit:
                    unit = _UNITS[explicit_unit.upper()]
                elif len(units) == 1:
                    unit = next(iter(units))
                else:
                    raise ValueError("unitless temperature interval has ambiguous source units")
                low, high = (_celsius(value, unit) for value in (low_value, high_value))
                if low > high or any(
                    not low <= _celsius(s["value"], s["unit"]) <= high for s in settings
                ):
                    raise ValueError("source setpoint contradicts its printed temperature interval")
                found.append(TemperatureInterval(path, text, low, high, owners))
    return tuple(found)


def celsius_bounds(
    template: CertifiedSemanticTemplate, source: Mapping[str, Any]
) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    for interval in intervals(template, source):
        for index in interval.container_indices:
            low, high = result.get(index, (interval.low_celsius, interval.high_celsius))
            low, high = max(low, interval.low_celsius), min(high, interval.high_celsius)
            if low > high:
                raise ValueError("printed temperature intervals have empty intersection")
            result[index] = low, high
    return result


def require_contract(template: CertifiedSemanticTemplate, source: Mapping[str, Any]) -> None:
    """A printed temperature instruction cannot become unconstrained cargo prose.

    A refrigeration-capable box alone does not prove its setpoint or storage
    interval. In particular, a source without setpoint labels still needs an
    explicit physical dependency before a new goods scenario can replace it.
    """
    require_temperature_classification(template.bindings)
    represented = generate(template, source, source)
    for index, group in enumerate(source["documentPatch"].get("cargoGroups", [])):
        for field in ("handlingInstructions", "additionalInformation"):
            for item, text in enumerate(group.get(field, [])):
                stated = _mentions(text) or (
                    re.search(r"\b(?:TEMPERATURE|TEMP\.?|SET\s*POINT)\b", text, re.I)
                    and re.search(r"(?<!\w)[+-]?\d", text)
                )
                path = f"documentPatch.cargoGroups[{index}].{field}[{item}]"
                if stated and path not in represented:
                    raise ValueError(
                        "printed temperature instruction lacks a complete physical dependency: "
                        + path
                    )


def generate(
    template: CertifiedSemanticTemplate, source: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, str]:
    from . import descendant as render

    result: dict[str, str] = {}
    for owners in summary_setpoint_owners(template, source):
        settings = [target["documentPatch"]["containers"][i]["temperatureSetpoint"] for i in owners]
        if any(s != settings[0] for s in settings):
            raise ValueError("temperature summary requires one shared generated setpoint")
    for interval in intervals(template, source):
        for index in interval.container_indices:
            setting = target["documentPatch"]["containers"][index]["temperatureSetpoint"]
            if (
                not interval.low_celsius
                <= _celsius(setting["value"], setting["unit"])
                <= interval.high_celsius
            ):
                raise ValueError("sampled setpoint contradicts its printed temperature interval")
        result[interval.path] = interval.text
    for binding in template.bindings:
        paths = binding.target_paths
        text_paths = [
            p
            for p in paths
            if re.fullmatch(
                r"documentPatch\.cargoGroups\[\d+\]\.(?:handlingInstructions|additionalInformation)\[\d+\]",
                p,
            )
        ]
        values = [p for p in paths if p.endswith(".temperatureSetpoint.value")]
        if not text_paths or not values:
            continue
        if len(values) != 1 or values[0].removesuffix(".value") + ".unit" not in paths:
            raise ValueError("temperature prose lacks one complete compiled setpoint dependency")
        value_path = values[0]
        unit_path = value_path.removesuffix(".value") + ".unit"
        old = Decimal(str(render._resolve_path(source, value_path)))
        new = Decimal(str(render._resolve_path(target, value_path)))
        unit = render._resolve_path(source, unit_path)
        if not new.is_finite() or render._resolve_path(target, unit_path) != unit:
            raise ValueError("temperature prose requires a finite value in its source unit")
        for path in text_paths:
            text = render._resolve_path(source, path)
            mentions = _mentions(text)
            if not mentions or any(
                Decimal(m["number"].replace(",", ".")) != old or _UNITS[m["unit"].upper()] != unit
                for m in mentions
            ):
                raise ValueError("temperature prose source disagrees with its compiled setpoint")
            output = _render_number(text, mentions, new)
            if path in result and result[path] != output:
                raise ValueError("temperature prose has conflicting compiled owners")
            result[path] = output
    for contract in cargo_setpoint_contracts(template, source):
        settings = [
            target["documentPatch"]["containers"][i]["temperatureSetpoint"]
            for i in contract.container_indices
        ]
        if any(s["unit"] != contract.unit or s != settings[0] for s in settings):
            raise ValueError("cargo temperature instruction requires one shared generated setpoint")
        new = Decimal(str(settings[0]["value"]))
        if not new.is_finite():
            raise ValueError("cargo temperature instruction requires a finite setpoint")
        result[contract.path] = _render_number(contract.text, _mentions(contract.text), new)
    return result


def validate(
    template: CertifiedSemanticTemplate, source: Mapping[str, Any], target: Mapping[str, Any]
) -> None:
    from . import descendant as render

    for path, expected in generate(template, source, target).items():
        if render._resolve_path(target, path) != expected:
            raise ValueError("temperature instruction contradicts its sampled setpoint: " + path)
