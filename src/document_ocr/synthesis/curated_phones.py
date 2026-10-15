"""Typed phone presentation plans over reviewed source ownership.

The shipment sampler owns valid phone identities. This adapter owns only their
printed dialing/spacing presentation and the corresponding extraction labels.
It never searches or rewrites unowned document text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.synthesis.curated import digest, flat
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.phone_presentations import PhoneFamily, format_phone_presentation

if TYPE_CHECKING:
    from document_ocr.synthesis.curated_templates import RenderRegion, SamplingBlueprint

_PHONE_PATH = re.compile(r"^documentPatch\.parties\..+\.contactDetails\.phoneNumbers\[\d+\]$")


class PhoneFamilyWeight(BaseModel):
    """An explicitly weighted dialing family, independent of phone identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    family: PhoneFamily
    weight: int = Field(gt=0)


class PhoneRenderingPolicy(BaseModel):
    """Preserve source presentations by default; optionally sample families.

    Immutable literal dialing prefixes constrain eligible families rather than
    being silently erased. Grouping and physical line breaks remain source-led.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    family_weights: tuple[PhoneFamilyWeight, ...] = (PhoneFamilyWeight(family="source", weight=1),)

    @model_validator(mode="after")
    def distinct_families(self) -> PhoneRenderingPolicy:
        if not self.family_weights or len({item.family for item in self.family_weights}) != len(
            self.family_weights
        ):
            raise ValueError("phone families must be nonempty and distinct")
        return self

    def select(self, stream: DeterministicStream, literals: tuple[str, ...]) -> PhoneFamily:
        constrained = {value for value in literals if value}
        allowed = (
            {"source"}
            if len(constrained) > 1
            else (
                {"source", "plus"}
                if constrained == {"+"}
                else ({"source", "double_zero"} if constrained == {"00"} else None)
            )
        )
        eligible = [
            item for item in self.family_weights if allowed is None or item.family in allowed
        ]
        if not eligible:
            raise ValueError("configured phone families conflict with an unchanged dialing prefix")
        draw = stream.randbelow(sum(item.weight for item in eligible))
        for item in eligible:
            if draw < item.weight:
                return item.family
            draw -= item.weight
        raise AssertionError("phone family selection exhausted its exact weight total")


def phone_values(target: dict[str, Any]) -> dict[str, str]:
    """Return every current extraction phone leaf, never fax or unowned text."""
    result = {path: value for path, value in flat(target).items() if _PHONE_PATH.fullmatch(path)}
    if any(not isinstance(value, str) or not value for value in result.values()):
        raise ValueError("phone target leaves must be nonempty strings")
    return result


def _pattern(value: str) -> re.Pattern[str]:
    tokens = re.split(r"\s+", value.strip())
    return re.compile(r"(?<![A-Za-z0-9])" + r"\s*".join(map(re.escape, tokens)) + r"(?![0-9])")


def _region_paths(blueprint: SamplingBlueprint, region: RenderRegion) -> tuple[str, ...]:
    if region.curated_key is None:
        paths = set(region.target_paths)
        recipe = blueprint.historical_bindings.get(region.key, {}).get("auxiliary_recipe", {})
        if isinstance(recipe.get("target"), str) and _PHONE_PATH.fullmatch(recipe["target"]):
            paths.add(recipe["target"])
        return tuple(sorted(paths))
    placeholder = "{" + region.curated_key + "}"
    return tuple(
        binding.path for binding in blueprint.contract.targets if placeholder in binding.expression
    )


def _reviewed_target_source(blueprint: SamplingBlueprint, paths: list[str], original: str) -> str:
    """Resolve an explicitly reviewed literal target envelope, never infer one."""
    declarations = blueprint.ownership_data.get("phone_target_presentations", {})
    if not isinstance(declarations, dict):
        raise ValueError("phone target presentations require a path-keyed declaration")
    values = set()
    for path in paths:
        if path not in declarations:
            values.add(original)
            continue
        declaration = declarations[path]
        if (
            not isinstance(declaration, dict)
            or set(declaration) != {"key", "occurrence", "value"}
            or not isinstance(declaration["value"], str)
            or not declaration["value"]
            or type(declaration["occurrence"]) is not int
            or declaration["occurrence"] < 1
        ):
            raise ValueError("phone target presentation requires an exact owner and literal value")
        owners = [
            region
            for region in blueprint.regions
            if region.key == declaration["key"]
            and region.occurrence + 1 == declaration["occurrence"]
            and path in _region_paths(blueprint, region)
        ]
        if len(owners) != 1 or declaration["value"] not in owners[0].source:
            raise ValueError("reviewed phone target presentation lacks its exact owned source")
        values.add(declaration["value"])
    if len(values) != 1:
        raise ValueError("shared phone targets have conflicting reviewed presentation envelopes")
    return values.pop()


def _attested_newline_joins(source: str, target: str) -> tuple[int, ...] | None:
    """Identify source line breaks joined inside a reviewed target digit/token run.

    All non-whitespace characters must already agree exactly, including every
    punctuation mark. Only newline boundaries missing from that reviewed target
    may be joined in the new presentation.
    """
    if re.sub(r"\s", "", source) != re.sub(r"\s", "", target):
        return None
    target_gaps = {
        len(re.sub(r"\s", "", target[: match.start()])) for match in re.finditer(r"\s+", target)
    }
    joins, ordinal = [], 0
    for match in re.finditer(r"\s+", source):
        count = match[0].count("\n")
        if count:
            boundary = len(re.sub(r"\s", "", source[: match.start()]))
            if boundary not in target_gaps:
                joins.extend(range(ordinal, ordinal + count))
            ordinal += count
    return tuple(joins)


def grounded_phone_surface(surface: str, joined_newline_ordinals: tuple[int, ...]) -> str:
    """Apply only source-attested physical joins, then ordinary whitespace folding."""
    joins, ordinal, parts = set(joined_newline_ordinals), 0, []
    for character in surface:
        if character == "\n":
            if ordinal not in joins:
                parts.append(character)
            ordinal += 1
        else:
            parts.append(character)
    if joins and max(joins) >= ordinal:
        raise ValueError("attested phone newline is absent from its rendered occurrence")
    return " ".join("".join(parts).split())


@dataclass(frozen=True)
class _PhonePiece:
    path: str
    input_value: str
    surface: str


@dataclass(frozen=True)
class PhoneRenderPlan:
    """Source/input-pinned occurrence transformations and label projections."""

    source_sha256: str
    input_values: tuple[tuple[str, str], ...]
    projected_values: tuple[tuple[str, str], ...]
    pieces: tuple[tuple[str, int, _PhonePiece], ...]
    records: tuple[dict[str, Any], ...]
    source_country_codes: tuple[tuple[str, str], ...]
    target_country_codes: tuple[tuple[str, str], ...]
    policy: PhoneRenderingPolicy
    _region_index: Mapping[tuple[str, int], tuple[_PhonePiece, ...]] = field(
        init=False, repr=False, compare=False
    )
    _record_index: Mapping[tuple[str, int], tuple[dict[str, Any], ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        pieces: dict[tuple[str, int], list[_PhonePiece]] = {}
        records: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for key, occurrence, piece in self.pieces:
            pieces.setdefault((key, occurrence), []).append(piece)
        for record in self.records:
            records.setdefault((record["key"], record["occurrence"]), []).append(record)
        object.__setattr__(
            self,
            "_region_index",
            MappingProxyType({selector: tuple(values) for selector, values in pieces.items()}),
        )
        object.__setattr__(
            self,
            "_record_index",
            MappingProxyType({selector: tuple(values) for selector, values in records.items()}),
        )

    @property
    def target_values(self) -> dict[str, str]:
        return dict(self.projected_values)

    @property
    def region_surfaces(self) -> Mapping[tuple[str, int], tuple[_PhonePiece, ...]]:
        return self._region_index

    def validate_for(self, blueprint: SamplingBlueprint, target: dict[str, Any]) -> None:
        if self.source_sha256 != digest(blueprint.source.encode()) or self.input_values != tuple(
            sorted(phone_values(target).items())
        ):
            raise ValueError(
                "phone presentation plan differs from its pinned source or input phones"
            )

    def apply_region(self, region: RenderRegion, text: str) -> str:
        """Style phone pieces in the already assembled, independently owned text."""
        # Replace all pieces simultaneously: inserted digits cannot become a
        # later phone's match, and shared labels never duplicate an occurrence.
        pieces = self.region_surfaces.get((region.key, region.occurrence), ())
        replacements: dict[tuple[int, int], str] = {}
        for piece in pieces:
            matches = list(_pattern(piece.input_value).finditer(text))
            if not matches:
                raise ValueError(f"sampled phone absent from its assembled owner: {piece.path}")
            for match in matches:
                span = match.span()
                if span in replacements and replacements[span] != piece.surface:
                    raise ValueError("one printed phone occurrence has conflicting presentations")
                replacements[span] = piece.surface
        ordered = sorted(replacements.items())
        if any(a[0][1] > b[0][0] for a, b in pairwise(ordered)):
            raise ValueError("phone presentation pieces overlap")
        for (start, end), value in reversed(ordered):
            text = text[:start] + value + text[end:]
        return text

    def canonical_region(self, region: RenderRegion) -> str | None:
        """Assemble a phone-only owner before its presentation projection.

        Source separators and literal framing stay intact. Composite owners
        containing other fields continue through their declared renderer.
        """
        if any(not _PHONE_PATH.fullmatch(path) for path in region.target_paths):
            return None
        records = self._record_index.get((region.key, region.occurrence), ())
        if not records or any(record["basis"] == "reviewed_expression" for record in records):
            return None
        inputs = dict(self.input_values)
        edits: dict[tuple[int, int], str] = {}
        for record in records:
            source = record["source"]
            matches = list(re.finditer(re.escape(source), region.source))
            if not matches:
                raise ValueError("phone source piece is absent from its pinned region")
            for match in matches:
                replacement = inputs[record["targetPaths"][0]]
                if match.span() in edits and edits[match.span()] != replacement:
                    raise ValueError("one source phone piece has conflicting sampled identities")
                edits[match.span()] = replacement
        result = region.source
        ordered = sorted(edits.items())
        if any(a[0][1] > b[0][0] for a, b in pairwise(ordered)):
            raise ValueError("canonical phone source pieces overlap")
        for (start, end), value in reversed(ordered):
            result = result[:start] + value + result[end:]
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "version": 1,
            "sourceSha256": self.source_sha256,
            "inputValues": dict(self.input_values),
            "targetValues": dict(self.projected_values),
            "sourceCountryCodes": dict(self.source_country_codes),
            "targetCountryCodes": dict(self.target_country_codes),
            "policy": self.policy.model_dump(mode="json"),
            "regions": [dict(record) for record in self.records],
        }


def prepare_phone_rendering(
    blueprint: SamplingBlueprint,
    sampled_target: dict[str, Any],
    *,
    source_country_codes: Mapping[str, str],
    target_country_codes: Mapping[str, str],
    policy: PhoneRenderingPolicy | None = None,
    stream: DeterministicStream,
) -> PhoneRenderPlan:
    """Compile changed public phone presentations using exact reviewed owners."""
    policy = policy or PhoneRenderingPolicy()
    before, inputs = phone_values(blueprint.target), phone_values(sampled_target)
    if before.keys() != inputs.keys():
        raise ValueError("phone rendering cannot change extraction phone topology")
    changed = {path for path in before if before[path] != inputs[path]}
    groups: dict[tuple[str, str], list[str]] = {}
    for path in sorted(changed):
        role = path.split(".contactDetails.", 1)[0]
        if role not in target_country_codes:
            raise ValueError(f"sampled phone lacks explicit country authority: {role}")
        groups.setdefault((before[path], inputs[path]), []).append(path)
    projected, pieces, records = {}, [], []
    raw = blueprint.source.encode()
    active_regions = [
        (region, set(_region_paths(blueprint, region)))
        for region in blueprint.regions
        if set(_region_paths(blueprint, region)) & changed
    ]
    for (source_value, input_value), paths in sorted(groups.items()):
        target_source = _reviewed_target_source(blueprint, paths, source_value)
        owners = []
        for region, region_paths in active_regions:
            owned = region_paths & set(paths)
            if not owned:
                continue
            source_match = _pattern(source_value).search(region.source)
            declaration = blueprint.ownership_data.get("surfaces", {}).get(region.key, {})
            expression = declaration.get("expression", "")
            declared_paths = {path for path in owned if "{" + path + "}" in expression}
            if source_match is not None:
                source_piece = source_match[0]
                local_prefix = region.source[: source_match.start()]
                external_plus = (
                    not local_prefix and raw[max(0, region.start - 1) : region.start] == b"+"
                )
                literal = (
                    "+"
                    if not source_piece.startswith("+")
                    and (local_prefix.endswith("+") or external_plus)
                    else ""
                )
                owners.append((region, source_piece, literal, owned, "source_subspan"))
            elif declared_paths:
                # An explicit reviewed expression can relocate a contact into
                # its actual party block. The reviewed label supplies its style;
                # the source placeholder is not misrepresented as a phone.
                owners.append((region, source_value, "", declared_paths, "reviewed_expression"))
            elif len(set(_region_paths(blueprint, region))) == len(owned):
                recipe = blueprint.historical_bindings.get(region.key, {}).get(
                    "auxiliary_recipe", {}
                )
                if (
                    re.sub(r"\D", "", region.source) != re.sub(r"\D", "", source_value)
                    and recipe.get("target") not in owned
                ):
                    raise ValueError(
                        f"phone source owner is a partial or unrelated number: {region.key}"
                    )
                literal = "+" if raw[max(0, region.start - 1) : region.start] == b"+" else ""
                owners.append((region, region.source, literal, owned, "source_owner"))
        owned_paths = {path for _, _, _, selected, _ in owners for path in selected}
        if set(paths) != owned_paths:
            raise ValueError(
                f"changed phone lacks a current reviewed owner: {sorted(set(paths) - owned_paths)}"
            )
        roles = {path.split(".contactDetails.", 1)[0] for path in paths}
        target_codes = {target_country_codes[role] for role in roles}
        source_codes = {
            source_country_codes[role] for role in roles if role in source_country_codes
        }
        if len(target_codes) != 1 or len(source_codes) > 1:
            raise ValueError(
                "shared source phone has inconsistent source or sampled country authority"
            )
        literals = tuple(owner[2] for owner in owners)
        family = policy.select(stream.derive("phone-presentation:" + source_value), literals)
        frames = tuple(
            "("
            if target_source.startswith("(")
            and not source_piece.startswith("(")
            and raw[max(0, region.start - 1) : region.start] == b"("
            and not any(other.start <= region.start - 1 < other.end for other in blueprint.regions)
            else ""
            for region, source_piece, _, _, _ in owners
        )
        result = format_phone_presentation(
            source_value=target_source,
            source_surfaces=tuple(
                frame + owner[1] for frame, owner in zip(frames, owners, strict=True)
            ),
            replacement_identity=input_value,
            target_country_code=next(iter(target_codes)),
            source_country_code=next(iter(source_codes), None),
            family=family,
            literal_prefixes=literals,
        )
        supported: set[str] = set()
        for (region, source_piece, literal, owned, basis), frame, surface, observed in zip(
            owners, frames, result.surfaces, result.families, strict=True
        ):
            if frame:
                if not surface.startswith(frame):
                    raise ValueError(
                        "phone presentation conflicts with its immutable opening frame"
                    )
                surface = surface[len(frame) :]
            joins = _attested_newline_joins(frame + literal + source_piece, target_source)
            if (
                joins is not None
                and grounded_phone_surface(frame + literal + surface, joins) == result.target_value
            ):
                supported.update(owned)
            for path in sorted(owned):
                pieces.append(
                    (region.key, region.occurrence, _PhonePiece(path, input_value, surface))
                )
            records.append(
                {
                    "key": region.key,
                    "occurrence": region.occurrence,
                    "source": source_piece,
                    "rendered": surface,
                    "targetPaths": sorted(owned),
                    "family": observed,
                    "literalPrefix": literal,
                    "framePrefix": frame,
                    "joinedNewlineOrdinals": list(joins) if joins is not None else [],
                    "targetSourceValue": target_source,
                    "identity": result.identity,
                    "basis": basis,
                }
            )
        if supported != set(paths):
            raise ValueError(
                "projected phone target has no exact source-certified printed occurrence: "
                + repr(sorted(set(paths) - supported))
            )
        for path in paths:
            projected[path] = result.target_value
    return PhoneRenderPlan(
        digest(raw),
        tuple(sorted(inputs.items())),
        tuple(sorted(projected.items())),
        tuple(pieces),
        tuple(records),
        tuple(sorted(source_country_codes.items())),
        tuple(sorted(target_country_codes.items())),
        policy,
    )
