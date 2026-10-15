"""Rebase reviewed byte ownership onto the current extraction target.

Historical templates contribute locations, not historical labels.  Current
curated regions own lexical content; disjoint historical regions supply the
remaining shipment facts.  Every changed target leaf must have an owner.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import pairwise
from typing import Any

from document_ocr.synthesis.container_semantics import (
    canonical_equipment_surface,
    iso_equipment_surface,
)
from document_ocr.synthesis.curated import (
    Occurrence,
    SourceContract,
    TargetBinding,
    Variable,
    _number_style,
    assign,
    compile_contract,
    digest,
    flat,
    interpolate,
    layout_surface,
    locate,
)
from document_ocr.synthesis.curated_casing import RenderCasing, TargetCasing, case_owned_text
from document_ocr.synthesis.curated_container_ids import prepare_container_identifier_rendering
from document_ocr.synthesis.curated_descriptions import (
    CompiledDescriptionBlocks,
    DescriptionBlocks,
    compile_description_blocks,
    normalized_description,
    project_rendered_descriptions,
    validate_description_regions,
)
from document_ocr.synthesis.curated_hs import hs_digits, render_source_hs
from document_ocr.synthesis.curated_phones import PhoneRenderPlan, phone_values
from document_ocr.synthesis.package_registry import package_category_surface_present
from document_ocr.synthesis.template_compiler.descendant import (
    _number_to_words,
    _package_candidate,
    _render_date_surface,
)

_CANONICAL_MEASUREMENT_VALUE = re.compile(r"\.(?:grossWeight|netWeight|volume)\.value$")
_PHYSICAL_SUM_DERIVATIONS = frozenset({"sum_gross_weight", "sum_net_weight", "sum_volume"})


def validate_postal_country_owner(text: str, country: str, code: str | None, path: str) -> None:
    """Require the sampled country inside its certified printed postal owner."""
    postal_text = " ".join(text.split())
    terminal_code = bool(
        code and re.search(r"(?<!\w)" + re.escape(code) + r"[\s,;.)]*$", postal_text)
    )
    if (
        not re.search(r"(?<!\w)" + re.escape(country) + r"(?!\w)", postal_text, re.I)
        and not terminal_code
    ):
        raise ValueError(f"sampled country absent from owned postal text: {path}")


def current_path(path: str) -> str:
    """Translate structural names only; never resurrect discarded label fields."""
    path = path.replace(".cargoGroups[", ".goodsItemDetails[")
    path = path.replace(".containers", ".containerInformation")
    path = path.replace(".containerNumber", ".equipmentIdentifier")
    path = re.sub(
        r"\.cargoAllocationGroups\[0\]\.allocations\[(\d+)\]",
        r".goodsItemDetails[0].splitGoodsPlacement[\1]",
        path,
    )
    path = re.sub(
        r"\.cargoPackages\[(\d+)\]\.quantity$",
        r".goodsItemDetails[0].numberAndTypeOfPackages[\1].packageQuantity",
        path,
    )
    path = re.sub(
        r"\.cargoPackages\[(\d+)\]",
        r".goodsItemDetails[0].numberAndTypeOfPackages[\1]",
        path,
    )
    return path


def wrap_owned_text(source: str, value: str, *, product: bool = False) -> str:
    """Preserve line span approximately without detaching delimiters from words.

    Explicit generated lines take precedence. Products keep their natural
    generated line boundaries; a short source word is not a column-width
    measurement. Other fields retain the source-relative presentation.
    No token or address component is removed.
    """
    if not value.strip():
        raise ValueError("an owned region cannot be blank")
    if "\r" in value:
        raise ValueError("generated regions use LF newlines")
    if "\n" in value:
        if any(not line.strip() for line in value.splitlines()):
            raise ValueError("generated region contains an empty physical line")
        if any(re.match(r"\s*[,;:]", line) for line in value.splitlines()):
            raise ValueError("generated line starts with a detached delimiter")
        return value
    if re.match(r"\s*[,;:]", value):
        raise ValueError("generated region starts with a detached delimiter")
    rendered = value if product else layout_surface(source, value)
    if " ".join(rendered.split()) != " ".join(value.split()):
        raise ValueError("line wrapping changed generated tokens")
    return rendered


@dataclass(frozen=True)
class RenderRegion:
    """One exact, non-overlapping mutable source span."""

    key: str
    occurrence: int
    start: int
    end: int
    source: str
    target_paths: tuple[str, ...]
    kind: str
    curated_key: str | None = None
    presentation: str = "text"
    baseline: str | None = None
    allow_empty: bool = False


@dataclass(frozen=True)
class LexicalOwnership:
    """Audited complete lexical regions replacing narrower old fragment owners.

    Every displaced curated occurrence must be included; this prevents a
    changed product identity leaving a repeated source-only product behind.
    """

    key: str
    target_paths: tuple[str, ...]
    occurrences: tuple[Occurrence, ...]
    kind: str = "owned_text"


@dataclass(frozen=True)
class SamplingBlueprint:
    """Pinned source, current labels, executable regions and ownership inventory."""

    document_id: str
    source: str
    target: dict[str, Any]
    contract: SourceContract
    regions: tuple[RenderRegion, ...]
    historical_bindings: Mapping[str, dict[str, Any]]
    nested_bindings: Mapping[str, tuple[str, ...]]
    blocked_bindings: tuple[dict[str, Any], ...]
    ownership_data: Mapping[str, Any] = field(default_factory=dict)
    description_blocks: CompiledDescriptionBlocks | None = None

    def inventory(self) -> dict[str, Any]:
        owned = {p for region in self.regions for p in region.target_paths}
        owned.update(t.path for t in self.contract.targets)
        if self.description_blocks is not None:
            owned.update(self.description_blocks.paths)
        return {
            "documentId": self.document_id,
            "sourceSha256": digest(self.source.encode()),
            "regionCount": len(self.regions),
            "curatedRegionCount": sum(r.curated_key is not None for r in self.regions),
            "historicalRegionCount": sum(r.curated_key is None for r in self.regions),
            "nestedBindings": {key: list(paths) for key, paths in self.nested_bindings.items()},
            "blockedBindings": list(self.blocked_bindings),
            "unownedTargetPaths": sorted(set(flat(self.target)) - owned),
            "descriptionBlocksSha256": (
                self.description_blocks.contract_sha256 if self.description_blocks else None
            ),
        }


def _current_binding_paths(binding: dict, leaves: Mapping[str, Any]) -> tuple[str, ...]:
    paths: set[str] = set()
    for original in binding.get("target_paths", []):
        translated = current_path(original)
        if translated in leaves:
            paths.add(translated)
        # A historical equipment description now has two structured categories.
        if translated.endswith(".typeDescription") and ".containerInformation[" in translated:
            prefix = translated.rsplit(".", 1)[0]
            paths.update(
                path
                for path in (prefix + ".sizeCategory", prefix + ".typeCategory")
                if path in leaves
            )
    return tuple(sorted(paths))


def compile_sampling_blueprint(
    row: dict[str, Any],
    historical_template: dict[str, Any],
    contract: SourceContract,
    *,
    ownership_overrides: tuple[LexicalOwnership, ...] = (),
    description_blocks: DescriptionBlocks | dict | None = None,
) -> SamplingBlueprint:
    """Combine existing ownership without guessing or relocating historical spans.

    A historical region wholly inside a current lexical owner is role evidence
    for that owner. A partial overlap is inventoried and cannot independently
    authorize a changed target. All positions are verified against current OCR.
    """
    blocks = compile_description_blocks(row["joinedRawText"], row["target"], description_blocks)
    compile_contract(row, contract, description_blocks=blocks)
    raw = row["joinedRawText"].encode()
    if historical_template.get("document_id") != row["documentId"]:
        raise ValueError("historical template belongs to another source")
    leaves = flat(row["target"])
    regions = []
    for variable in contract.variables:
        paths = tuple(t.path for t in contract.targets if "{" + variable.key + "}" in t.expression)
        for index, occurrence in enumerate(variable.occurrences):
            start, end = locate(row["joinedRawText"], occurrence)
            regions.append(
                RenderRegion(
                    variable.key,
                    index,
                    start,
                    end,
                    occurrence.text,
                    paths,
                    variable.kind,
                    variable.key,
                    occurrence.presentation,
                    variable.value,
                )
            )
    overrides = []
    override_occurrences: dict[str, list[dict[str, Any]]] = {}
    removed_keys: set[str] = set()
    for override in ownership_overrides:
        if not override.target_paths or not set(override.target_paths) <= leaves.keys():
            raise ValueError(f"ownership override needs existing current leaves: {override.key}")
        for index, occurrence in enumerate(override.occurrences):
            start, end = locate(row["joinedRawText"], occurrence)
            override_occurrences.setdefault(override.key, []).append(
                dict(byte_start=start, byte_end=end, source_text=occurrence.text)
            )
            overlaps = [r for r in regions if start < r.end and r.start < end]
            if any(not (start <= r.start and r.end <= end) for r in overlaps):
                raise ValueError(
                    f"ownership override partially intersects another owner: {override.key}"
                )
            if any(not set(r.target_paths) <= set(override.target_paths) for r in overlaps):
                raise ValueError(
                    f"ownership override consumes another target's region: {override.key}"
                )
            removed_keys.update(r.curated_key for r in overlaps)
            overrides.append(
                RenderRegion(
                    override.key,
                    index,
                    start,
                    end,
                    occurrence.text,
                    override.target_paths,
                    override.kind,
                    override.key if override.kind in {"name", "postal", "product"} else None,
                    "text",
                    normalized_description(occurrence.text)
                    if override.kind == "product"
                    else str(leaves[override.target_paths[0]])
                    if override.kind in {"name", "postal", "product"}
                    else None,
                )
            )
    for region in regions:
        if region.curated_key in removed_keys and not any(
            replacement.start <= region.start and region.end <= replacement.end
            for replacement in overrides
        ):
            raise ValueError(
                f"ownership override omits a repeated occurrence: {region.curated_key}"
            )
    regions = [r for r in regions if r.curated_key not in removed_keys]
    regions.extend(overrides)
    curated_regions = tuple(regions)
    historical: dict[str, dict] = {}
    nested: dict[str, list[str]] = {}
    blocked = []
    for binding in historical_template["bindings"]:
        key = binding["logical_key"]
        if key in historical:
            raise ValueError(f"duplicate historical binding: {key}")
        historical[key] = deepcopy(binding)
        paths = _current_binding_paths(binding, leaves)
        for index, occurrence in enumerate(binding["occurrences"]):
            start, end, text = (
                occurrence["byte_start"],
                occurrence["byte_end"],
                occurrence["source_text"],
            )
            if not 0 <= start < end <= len(raw) or raw[start:end].decode() != text:
                raise ValueError(f"historical span differs from current OCR: {key}/{index}")
            overlaps = [r for r in curated_regions if start < r.end and r.start < end]
            if overlaps:
                owners = [r for r in overlaps if r.start <= start and end <= r.end]
                if len(owners) == 1:
                    owner = owners[0]
                    # A product's generated prose is not a rendering authority
                    # for a separately typed shipment measurement. Absorbing
                    # that owner would let wording delete its printed value
                    # while ownership coverage still certifies the target.
                    if (
                        owner.kind == "product"
                        and binding["value_kind"] in {"decimal_measurement", "integer"}
                        and (
                            any(_CANONICAL_MEASUREMENT_VALUE.search(path) for path in paths)
                            or binding.get("derivation") in _PHYSICAL_SUM_DERIVATIONS
                        )
                    ):
                        raise ValueError(
                            "product owner consumes a canonical measurement: "
                            f"{owner.key}/{key}; separate its numeric source owner"
                        )
                    nested.setdefault(key, []).append(owner.curated_key or owner.key)
                    # Add role provenance without creating a second physical edit.
                    position = (
                        regions.index(owner)
                        if owner in regions
                        else next(
                            i
                            for i, r in enumerate(regions)
                            if r.key == owner.key and r.occurrence == owner.occurrence
                        )
                    )
                    current = regions[position]
                    regions[position] = RenderRegion(
                        **{
                            **current.__dict__,
                            "target_paths": tuple(sorted(set(current.target_paths) | set(paths))),
                        }
                    )
                else:
                    blocked.append(
                        {
                            "key": key,
                            "occurrence": index,
                            "source": text,
                            "targetPaths": list(paths),
                            "reason": "partial_curated_overlap",
                            "owners": sorted({r.curated_key or r.key for r in overlaps}),
                        }
                    )
                continue
            regions.append(RenderRegion(key, index, start, end, text, paths, binding["value_kind"]))
    for override in ownership_overrides:
        if override.key in historical:
            raise ValueError(
                f"ownership override key duplicates a historical binding: {override.key}"
            )
        historical[override.key] = {
            "logical_key": override.key,
            "target_paths": list(override.target_paths),
            "value_kind": override.kind,
            "occurrences": override_occurrences[override.key],
        }
    lexical_overrides = [o for o in ownership_overrides if o.kind in {"name", "postal", "product"}]
    if lexical_overrides:
        replaced_paths = {p for o in lexical_overrides for p in o.target_paths}
        updated_variables = [v for v in contract.variables if v.key not in removed_keys]
        updated_targets = [t for t in contract.targets if t.path not in replaced_paths]
        for override in lexical_overrides:
            if len({str(leaves[p]) for p in override.target_paths}) != 1:
                raise ValueError(
                    f"lexical override combines distinct baseline facts: {override.key}"
                )
            baseline = str(leaves[override.target_paths[0]])
            if override.kind == "product":
                copies = {normalized_description(o.text) for o in override.occurrences}
                if len(copies) != 1:
                    raise ValueError(f"product override copies differ: {override.key}")
                baseline = copies.pop()
            updated_variables.append(
                Variable(
                    key=override.key,
                    kind=override.kind,
                    value=baseline,
                    meaning="Complete reviewed source-owned " + override.kind,
                    required_literals=[],
                    occurrences=list(override.occurrences),
                )
            )
            updated_targets.extend(
                TargetBinding(path=p, expression="{" + override.key + "}")
                for p in override.target_paths
            )
        contract = contract.model_copy(
            update={"variables": updated_variables, "targets": updated_targets}
        )
    regions.sort(key=lambda region: region.start)
    for previous, current in pairwise(regions):
        if previous.end > current.start:
            raise ValueError(f"overlapping historical regions: {previous.key}, {current.key}")
    validate_description_regions(blocks, tuple(regions))
    return SamplingBlueprint(
        row["documentId"],
        row["joinedRawText"],
        deepcopy(row["target"]),
        contract,
        tuple(regions),
        historical,
        {key: tuple(dict.fromkeys(owners)) for key, owners in nested.items()},
        tuple(blocked),
        description_blocks=blocks,
    )


def _simple_surface(region: RenderRegion, old: Any, new: Any) -> str:
    if old == new:
        return region.source
    if region.kind == "date":
        return _render_date_surface(region.source, old, new)
    if isinstance(old, (int, float)) and not isinstance(old, bool):
        return _number_style(region.source, str(old), str(new))
    if isinstance(new, str) and new.startswith("PACKAGE_"):
        return _package_candidate(region.source, new)
    if not isinstance(new, str):
        raise ValueError(f"explicit surface required for {region.key}: {type(new).__name__}")
    if region.kind == "owned_text":
        return wrap_owned_text(region.source, new)
    if any(".hsCodes[" in path for path in region.target_paths):
        return render_source_hs(region.source, hs_digits(new))
    if region.kind in {"phone", "identifier"}:
        stripped = re.sub(r"[\s/\-]", "", region.source)
        if stripped == str(old) and len(new) == len(stripped):
            chars = iter(new)
            return "".join(c if re.match(r"[\s/\-]", c) else next(chars) for c in region.source)
    # Composite surfaces (CHINA(CN), 3 X 40HC, etc.) require a supplied renderer.
    normalized_source = " ".join(region.source.split()).upper()
    if normalized_source != " ".join(str(old).split()).upper():
        raise ValueError(f"explicit composite surface required for {region.key}: {region.source!r}")
    return wrap_owned_text(region.source, new)


def _aligned_sample_leaves(before: dict, after: dict) -> dict:
    """Require every source fact slot, including distinct national extensions."""
    if set(before) != set(after):
        raise ValueError("sampling may not add/remove target fields or change array topology")
    return after


def _validate_curated_expressions(
    blueprint: SamplingBlueprint,
    values: Mapping[str, str],
    after: Mapping[str, Any],
    rendered_regions: list[tuple[RenderRegion, str]],
) -> None:
    before = flat(blueprint.target)
    replaced = {
        path
        for region, _ in rendered_regions
        if region.kind == "owned_text"
        for path in region.target_paths
    }
    if blueprint.description_blocks is not None:
        replaced.update(blueprint.description_blocks.paths)
    for binding in blueprint.contract.targets:
        if binding.path in replaced:
            continue
        expression = (
            blueprint.ownership_data.get("render_expressions", {}).get(
                binding.path, binding.expression
            )
            if after[binding.path] != before[binding.path]
            else binding.expression
        )
        # A separately printed country can be a fixed literal in the original
        # address expression. Substitute only its same-party owned source span,
        # before interpolation, so generated street words cannot be rewritten.
        if binding.path.endswith(".addressLine"):
            country_path = binding.path.rsplit(".", 1)[0] + ".country"
            substitutions = {
                (region.source, text)
                for region, text in rendered_regions
                if region.curated_key is None
                and country_path in region.target_paths
                and region.source != text
            }
            old_country = before.get(country_path)
            if old_country and substitutions:
                substitutions.add((str(old_country), str(after[country_path])))
            expression = substitute_expression_literals(expression, dict(substitutions))
        expected = " ".join(interpolate(expression, dict(values)).split())
        actual = after[binding.path]
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            expected_number = sum((Decimal(v.strip()) for v in expected.split("+")), Decimal(0))
            if expected_number != Decimal(str(actual)):
                raise ValueError(f"rendered numeric expression differs from target: {binding.path}")
        elif expected.upper() != " ".join(str(actual).split()).upper():
            raise ValueError(f"rendered lexical expression differs from target: {binding.path}")


def substitute_expression_literals(expression: str, substitutions: Mapping[str, str]) -> str:
    """Replace original literal tokens once, never reprocess inserted country names."""
    if not substitutions:
        return expression
    mapping = {
        " ".join(k.split()).casefold(): " ".join(v.split()) for k, v in substitutions.items()
    }
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
        + r")(?!\w)",
        re.I,
    )
    return "".join(
        chunk if chunk.startswith("{") else pattern.sub(lambda m: mapping[m[0].casefold()], chunk)
        for chunk in re.split(r"(\{[a-z][a-z0-9_]*\})", expression)
    )


def validate_party_field_separators(
    blueprint: SamplingBlueprint,
) -> tuple[tuple[RenderRegion, RenderRegion, str], ...]:
    """Resolve reviewed fused name/postal seams without changing source bytes.

    Occurrences are one-based within a region key's owned occurrences. Only
    explicitly selected, adjacent lexical fields of the same party qualify;
    ordinary zero-length source gaps never imply a separator automatically.
    """
    if "party_field_separators" not in blueprint.ownership_data:
        return ()
    declaration = blueprint.ownership_data["party_field_separators"]
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"source_sha256", "boundaries"}
        or declaration["source_sha256"] != digest(blueprint.source.encode())
        or not isinstance(declaration["boundaries"], list)
        or not declaration["boundaries"]
    ):
        raise ValueError("party-field separators require exact source pin and nonempty boundaries")
    by_selector = {(region.key, region.occurrence + 1): region for region in blueprint.regions}
    if len(by_selector) != len(blueprint.regions):
        raise ValueError("party-field separator source has duplicate region selectors")
    result, seen = [], set()
    for boundary in declaration["boundaries"]:
        if not isinstance(boundary, dict) or set(boundary) != {
            "party_path",
            "left",
            "right",
            "separator",
        }:
            raise ValueError(
                "party-field boundary requires exact role, two selectors and separator"
            )
        party = boundary["party_path"]
        if (
            not isinstance(party, str)
            or re.fullmatch(r"documentPatch\.parties\.[A-Za-z]\w*(?:\[(?:0|[1-9]\d*)\])?", party)
            is None
            or not isinstance(boundary["separator"], str)
            or boundary["separator"] not in {" ", "\n"}
        ):
            raise ValueError("party-field boundary has invalid role or whitespace separator")
        selectors = []
        for side in ("left", "right"):
            selector = boundary[side]
            if (
                not isinstance(selector, dict)
                or set(selector) != {"key", "occurrence"}
                or not isinstance(selector["key"], str)
                or not selector["key"]
                or not isinstance(selector["occurrence"], int)
                or isinstance(selector["occurrence"], bool)
                or selector["occurrence"] < 1
            ):
                raise ValueError(
                    "party-field boundary selector must be a key and one-based occurrence"
                )
            key = selector["key"], selector["occurrence"]
            if key not in by_selector:
                raise ValueError("party-field boundary selector is absent from source owners")
            selectors.append(key)
        left, right = (by_selector[key] for key in selectors)
        if (
            left.curated_key is None
            or right.curated_key is None
            or left.kind != "name"
            or right.kind != "postal"
            or party + ".name" not in left.target_paths
            or party + ".addressLine" not in right.target_paths
            or left.end != right.start
        ):
            raise ValueError(
                "party-field boundary needs adjacent lexical name/postal owners of one party"
            )
        pair = tuple(selectors)
        if pair in seen:
            raise ValueError("duplicate party-field boundary")
        seen.add(pair)
        result.append((left, right, boundary["separator"]))
    return tuple(result)


def _apply_party_field_separators(
    blueprint: SamplingBlueprint, rendered_regions: list[tuple[RenderRegion, str]]
) -> list[tuple[RenderRegion, str]]:
    boundaries = validate_party_field_separators(blueprint)
    if not boundaries:
        return rendered_regions
    indices = {(region.key, region.occurrence): i for i, (region, _) in enumerate(rendered_regions)}
    result = list(rendered_regions)
    for left, right, separator in boundaries:
        left_text = result[indices[left.key, left.occurrence]][1]
        right_index = indices[right.key, right.occurrence]
        right_text = result[right_index][1]
        if left_text != left.source or right_text != right.source:
            # Prefix the existing right owner, not an unowned insertion: its
            # ordinary byte-edit receipt remains the replay/geometry authority.
            result[right_index] = right, separator + right_text
    return result


def render_sampling_blueprint(
    blueprint: SamplingBlueprint,
    sampled_target: dict[str, Any],
    lexical_values: Mapping[str, str],
    *,
    surface_values: Mapping[str, str | list[str]] | None = None,
    certified_country_codes: Mapping[str, str] | None = None,
    render_casing: RenderCasing = "preserve",
    target_casing: TargetCasing = "uppercase",
    phone_rendering: PhoneRenderPlan | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Render a sampled current target with explicit coverage and byte receipts.

    ``lexical_values`` uses current contract variable keys. ``surface_values``
    uses historical logical keys; a list is indexed by original occurrence.
    Unchanged auxiliary surfaces remain unchanged, never silently regenerated.
    Unsupported changed fields or conflicting dependent values are errors.
    """
    surface_values = surface_values or {}
    certified_country_codes = certified_country_codes or {}
    variables = {v.key: v for v in blueprint.contract.variables}
    unknown = set(lexical_values) - set(variables)
    unknown_surfaces = set(surface_values) - set(blueprint.historical_bindings)
    if unknown or unknown_surfaces:
        raise ValueError(f"unknown render keys: {sorted(unknown | unknown_surfaces)}")
    before = flat(blueprint.target)
    changed_phones = {
        path for path, value in phone_values(sampled_target).items() if before.get(path) != value
    }
    if changed_phones and phone_rendering is None:
        raise ValueError("changed phones require an explicit source-conditioned presentation plan")
    if phone_rendering is not None:
        phone_rendering.validate_for(blueprint, sampled_target)
        if set(phone_rendering.target_values) != changed_phones:
            raise ValueError("phone presentation plan does not cover exactly the changed phones")
    expected_descriptions = {p for p in before if p.endswith(".description")}
    projected_paths = blueprint.description_blocks.paths if blueprint.description_blocks else set()
    if expected_descriptions != projected_paths:
        raise ValueError("description targets require complete reviewed description_blocks")
    validate_description_regions(blueprint.description_blocks, blueprint.regions)
    after = _aligned_sample_leaves(before, flat(sampled_target))
    container_rendering = prepare_container_identifier_rendering(blueprint, sampled_target)
    for path, code in certified_country_codes.items():
        if (
            path not in after
            or not path.endswith(".country")
            or not re.fullmatch(r"[A-Z]{2}", code)
        ):
            raise ValueError(f"invalid certified country-code mapping: {path}")
    requested_changes = {path for path in before if before[path] != after[path]}
    changed = requested_changes - projected_paths
    covered: set[str] = set()
    variable_values: dict[str, str] = {}
    direct = {
        key: [
            t.path
            for t in blueprint.contract.targets
            if t.expression == "{" + key + "}" and t.path not in projected_paths
        ]
        for key in variables
    }
    active_products = {r.curated_key for r in blueprint.regions if r.kind == "product"}
    for key, variable in variables.items():
        if (
            key in active_products
            and key not in lexical_values
            and any(before[path] != after[path] for path in projected_paths)
        ):
            raise ValueError(f"changed description requires explicit product wording: {key}")
        values = {str(after[path]) for path in direct[key]}
        if len(values) > 1:
            raise ValueError(f"shared variable has inconsistent sampled targets: {key}")
        value = lexical_values.get(key, next(iter(values)) if values else variable.value)
        # Numeric string forms such as 1200 vs 1200.0 express the same value.
        if (
            values
            and " ".join(value.split()).upper() != " ".join(next(iter(values)).split()).upper()
            and (
                variable.kind not in {"count", "mass", "volume"}
                or Decimal(value) != Decimal(next(iter(values)))
            )
        ):
            raise ValueError(f"lexical value differs from sampled target: {key}")
        variable_values[key] = value
    rendered_regions = []
    for region in blueprint.regions:
        changed_paths = set(region.target_paths) & (
            requested_changes if region.kind == "owned_text" else changed
        )
        phone_base = phone_rendering.canonical_region(region) if phone_rendering else None
        container_base = container_rendering.direct_region(region)
        if phone_base is not None:
            text = phone_base
            covered.update(changed_paths)
        elif container_base is not None and region.key not in surface_values:
            # The source-owned mask is the presentation contract, not ordinary
            # lexical wrapping. Explicit supplied composites are projected below.
            text = container_base
            covered.update(changed_paths)
        elif region.curated_key is not None:
            value = variable_values[region.curated_key]
            if value == region.baseline:
                text = region.source
            elif region.presentation == "number":
                text = _number_style(region.source, region.baseline, value)
            elif region.presentation == "words":
                number = Decimal(value)
                if number != int(number):
                    raise ValueError("spelled package count must be integral")
                text = _number_to_words(int(number)).upper()
            else:
                text = wrap_owned_text(region.source, value, product=region.kind == "product")
            # A nested country target must actually occur inside its postal
            # replacement; ownership alone does not prove a retained country.
            for path in region.target_paths if text != region.source else changed_paths:
                if ".parties." in path and path.endswith(".country"):
                    validate_postal_country_owner(
                        text, str(after[path]), certified_country_codes.get(path), path
                    )
            covered.update(changed_paths)
        elif region.key in surface_values:
            supplied = surface_values[region.key]
            if isinstance(supplied, list):
                expected = len(blueprint.historical_bindings[region.key]["occurrences"])
                if len(supplied) != expected:
                    raise ValueError(f"surface occurrence count differs: {region.key}")
                text = supplied[region.occurrence]
            else:
                text = supplied
            if not text.strip() and not (region.allow_empty and text == ""):
                raise ValueError(f"explicit surface cannot be blank: {region.key}")
            covered.update(changed_paths)
        elif changed_paths:
            paths = region.target_paths
            equipment = [
                p
                for p in paths
                if p.endswith((".sizeCategory", ".typeCategory")) and ".containerInformation[" in p
            ]
            if equipment:
                prefix = equipment[0].rsplit(".", 1)[0]
                size = after.get(prefix + ".sizeCategory")
                kind = after.get(prefix + ".typeCategory")
                if not size or not kind:
                    raise ValueError(f"explicit partial-equipment surface required: {region.key}")
                text = iso_equipment_surface(size, kind, region.source)
                if text is None:
                    # Operational suffixes are not equipment labels; callers must
                    # explicitly retain those rather than have this function erase them.
                    if re.search(r"FCL|LCL|STC|SAID|\b\d+\s*[Xx]", region.source):
                        raise ValueError(f"explicit equipment composite required: {region.key}")
                    text = canonical_equipment_surface(size, kind)
            else:
                pairs = {(str(before[path]), str(after[path])) for path in paths}
                if len(pairs) != 1:
                    raise ValueError(
                        f"shared historical surface has conflicting values: {region.key}"
                    )
                path = paths[0]
                text = _simple_surface(region, before[path], after[path])
            covered.update(changed_paths)
        else:
            text = region.source
        hs_paths = [path for path in region.target_paths if ".hsCodes[" in path]
        if hs_paths and any(hs_digits(text) != hs_digits(after[path]) for path in hs_paths):
            raise ValueError(f"printed HS digits differ from their complete target: {region.key}")
        rendered_regions.append((region, case_owned_text(text, region.target_paths, render_casing)))
    # Current contract paths, rather than stale historical array indices, own
    # identity and placement changes certified by the typed presentation plan.
    covered.update(container_rendering.covered_paths)
    if missing := changed - covered:
        raise ValueError(f"changed target fields have no rendered owner: {sorted(missing)}")
    for path in changed:
        if ".numberAndTypeOfPackages[" not in path or not path.endswith(".typeCategory"):
            continue
        category = after[path]
        supported = False
        for region, text in rendered_regions:
            if path not in region.target_paths or region.curated_key is not None:
                continue
            expected = " ".join(_package_candidate(region.source, category).split())
            if package_category_surface_present(text, category) or re.search(
                r"(?<!\w)" + re.escape(expected) + r"(?!\w)", " ".join(text.split()), re.I
            ):
                supported = True
                break
        if not supported:
            raise ValueError(f"changed package category lacks an owned printed noun: {path}")
    rendered_regions = _apply_party_field_separators(blueprint, rendered_regions)
    _validate_curated_expressions(blueprint, variable_values, after, rendered_regions)
    if phone_rendering is not None:
        rendered_regions = [
            (region, phone_rendering.apply_region(region, text))
            for region, text in rendered_regions
        ]
    rendered_regions = [
        (region, container_rendering.project_region(region, text))
        for region, text in rendered_regions
    ]
    raw, parts, cursor, edits = blueprint.source.encode(), [], 0, []
    for region, text in rendered_regions:
        parts.extend((raw[cursor : region.start], text.encode()))
        cursor = region.end
        if text != region.source:
            edits.append(
                {
                    "key": region.key,
                    "occurrence": region.occurrence,
                    "byteStart": region.start,
                    "byteEnd": region.end,
                    "before": region.source,
                    "after": text,
                    "targetPaths": list(region.target_paths),
                }
            )
    parts.append(raw[cursor:])
    result = b"".join(parts)
    replay = raw
    for edit in reversed(edits):
        if replay[edit["byteStart"] : edit["byteEnd"]].decode() != edit["before"]:
            raise ValueError("independent exact-edit replay failed")
        replay = replay[: edit["byteStart"]] + edit["after"].encode() + replay[edit["byteEnd"] :]
    if replay != result:
        raise ValueError("independent render/replay disagreement")
    projected, description_proof = project_rendered_descriptions(
        blueprint.description_blocks, raw, result, edits, casing=target_casing
    )
    target = deepcopy(sampled_target)
    covered.difference_update(projected_paths)
    for path, value in projected.items():
        assign(target, path, value)
        if before[path] != value:
            changed.add(path)
            covered.add(path)
    hs_presentations = {}
    for region, text in rendered_regions:
        for path in region.target_paths:
            if ".hsCodes[" in path:
                hs_presentations.setdefault(path, " ".join(text.split()))
    # HS values are projected from the printed presentation, just as descriptions
    # are projected from their owned passages. Intermediate digit-only values
    # must not leave formatting-only changes in an otherwise idempotent proof.
    for path, value in hs_presentations.items():
        assign(target, path, value)
        if before[path] != value:
            changed.add(path)
            covered.add(path)
        else:
            changed.discard(path)
            covered.discard(path)
    if phone_rendering is not None:
        for path, value in phone_rendering.target_values.items():
            assign(target, path, value)
            if before[path] != value:
                changed.add(path)
                covered.add(path)
            else:
                changed.discard(path)
                covered.discard(path)
    return (
        result.decode(),
        target,
        {
            "sourceSha256": digest(raw),
            "renderedSha256": digest(result),
            "changedTargetPaths": sorted(changed),
            "coveredTargetPaths": sorted(covered),
            "certifiedCountryCodes": dict(certified_country_codes),
            "edits": edits,
            "unchangedBytesPreserved": True,
            **({"hsPresentations": hs_presentations} if hs_presentations else {}),
            **({"descriptionProjection": description_proof} if description_proof else {}),
            **(
                {"phonePresentations": phone_rendering.receipt()}
                if phone_rendering is not None and changed_phones
                else {}
            ),
            **(
                {"containerIdentifierPresentations": container_rendering.receipt()}
                if container_rendering.records
                else {}
            ),
            **({"renderCasing": render_casing} if render_casing != "preserve" else {}),
        },
    )
