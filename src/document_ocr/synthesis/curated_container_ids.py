"""Source-shaped container identifiers over exact, current target ownership.

Only printed presentation changes. Canonical identifiers, placement links,
seals, equipment types and all surrounding text retain their existing owners.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.generators import validate_container_number

if TYPE_CHECKING:
    from document_ocr.synthesis.curated_templates import RenderRegion, SamplingBlueprint

_IDENTIFIER = re.compile(r"[A-Z]{3}[UJZ][0-9]{7}")
_SEPARATORS = re.compile(r"[\s./-]")
_IDENTIFIER_PATH = re.compile(
    r"^documentPatch\.(?:containerInformation\[\d+\]|"
    r"goodsItemDetails\[\d+\]\.splitGoodsPlacement\[\d+\])\.equipmentIdentifier$"
)


def format_container_identifier(
    source_surface: str, source_identifier: str, sampled_identifier: str
) -> str:
    """Replace eleven verified ID characters, preserving every source separator.

    Character aliases, missing digits and adjacent cargo/equipment/seal wording
    are not formatting. They must never pass by deleting arbitrary characters.
    """
    if not _IDENTIFIER.fullmatch(source_identifier):
        raise ValueError("source container identifier must be canonical ISO 6346 syntax")
    if not _IDENTIFIER.fullmatch(sampled_identifier) or not validate_container_number(
        sampled_identifier
    ):
        raise ValueError("sampled container identifier requires a valid ISO 6346 check digit")
    if _SEPARATORS.sub("", source_surface) != source_identifier:
        raise ValueError("source container presentation is not reversible to its canonical ID")
    replacement = iter(sampled_identifier)
    return "".join(
        character if _SEPARATORS.fullmatch(character) else next(replacement)
        for character in source_surface
    )


def container_identifier_values(target: dict[str, Any]) -> dict[str, str]:
    """Return current container identities and their explicit placement links."""
    patch = target.get("documentPatch", {})
    values = {}
    for index, equipment in enumerate(patch.get("containerInformation", ())):
        if "equipmentIdentifier" in equipment:
            values[f"documentPatch.containerInformation[{index}].equipmentIdentifier"] = equipment[
                "equipmentIdentifier"
            ]
    for goods_index, goods in enumerate(patch.get("goodsItemDetails", ())):
        for placement_index, placement in enumerate(goods.get("splitGoodsPlacement", ())):
            if "equipmentIdentifier" in placement:
                path = (
                    f"documentPatch.goodsItemDetails[{goods_index}]"
                    f".splitGoodsPlacement[{placement_index}].equipmentIdentifier"
                )
                values[path] = placement["equipmentIdentifier"]
    if any(
        not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) for value in values.values()
    ):
        raise ValueError("container identifier targets must contain canonical IDs")
    return values


def _current_paths(blueprint: SamplingBlueprint, region: RenderRegion) -> tuple[str, ...]:
    if region.curated_key is None:
        return region.target_paths
    placeholder = "{" + region.curated_key + "}"
    return tuple(
        binding.path for binding in blueprint.contract.targets if placeholder in binding.expression
    )


def _bounded_identifier(value: str) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])")


def project_container_identifier(
    rendered_owner: str,
    *,
    source_surface: str,
    source_identifier: str,
    sampled_identifier: str,
) -> str:
    """Project one owned ID inside supplied text without altering its neighbors.

    Exactly one canonical or already source-shaped occurrence is required.
    This deliberately rejects duplicate/ambiguous matches instead of replacing
    a seal or a second field that happens to contain the same character string.
    """
    styled = format_container_identifier(source_surface, source_identifier, sampled_identifier)
    shaped_matches = list(_bounded_identifier(styled).finditer(rendered_owner))
    canonical_matches = list(_bounded_identifier(sampled_identifier).finditer(rendered_owner))
    if styled == sampled_identifier:
        if len(canonical_matches) != 1:
            raise ValueError("container owner requires exactly one sampled identifier occurrence")
        return rendered_owner
    if shaped_matches:
        if len(shaped_matches) != 1 or canonical_matches:
            raise ValueError("container owner has ambiguous canonical/source-shaped occurrences")
        return rendered_owner
    if len(canonical_matches) != 1:
        raise ValueError("container owner requires exactly one sampled identifier occurrence")
    match = canonical_matches[0]
    return rendered_owner[: match.start()] + styled + rendered_owner[match.end() :]


@dataclass(frozen=True)
class ContainerIdentifierRenderPlan:
    """Pinned source masks and explicit aliases; canonical targets never change."""

    source_sha256: str
    input_values: tuple[tuple[str, str], ...]
    records: tuple[dict[str, Any], ...]
    _index: Mapping[tuple[str, int], dict[str, Any]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        index = {(record["key"], record["occurrence"]): record for record in self.records}
        if len(index) != len(self.records):
            raise ValueError("duplicate container identifier occurrence selectors")
        object.__setattr__(self, "_index", MappingProxyType(index))

    def validate_for(self, blueprint: SamplingBlueprint, target: dict[str, Any]) -> None:
        if self.source_sha256 != digest(blueprint.source.encode()) or self.input_values != tuple(
            sorted(container_identifier_values(target).items())
        ):
            raise ValueError(
                "container presentation plan differs from pinned source/input identities"
            )

    @property
    def covered_paths(self) -> frozenset[str]:
        return frozenset(path for record in self.records for path in record["targetPaths"])

    def direct_region(self, region: RenderRegion) -> str | None:
        """Return the complete source-shaped value for one pure ID source owner."""
        record = self._index.get((region.key, region.occurrence))
        return record["rendered"] if record is not None else None

    def project_region(self, region: RenderRegion, rendered_owner: str) -> str:
        record = self._index.get((region.key, region.occurrence))
        if record is None:
            return rendered_owner
        if rendered_owner == record["rendered"]:
            return rendered_owner
        if record["basis"] == "reviewed_identifier_alias":
            # A reviewed character-level OCR alias is not a separator mask.
            # Preserve its existing declared canonical rendering policy.
            matches = list(
                _bounded_identifier(record["sampledIdentifier"]).finditer(rendered_owner)
            )
            if len(matches) != 1:
                raise ValueError("reviewed container alias lacks exactly one sampled canonical ID")
            return rendered_owner
        return project_container_identifier(
            rendered_owner,
            source_surface=record["source"],
            source_identifier=record["sourceIdentifier"],
            sampled_identifier=record["sampledIdentifier"],
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "version": 1,
            "sourceSha256": self.source_sha256,
            "inputValues": dict(self.input_values),
            "records": [
                {**record, "targetPaths": list(record["targetPaths"])} for record in self.records
            ],
        }


def prepare_container_identifier_rendering(
    blueprint: SamplingBlueprint, sampled_target: dict[str, Any]
) -> ContainerIdentifierRenderPlan:
    """Compile all changed current-ID owners, rejecting missing or guessed mappings."""
    before, after = (
        container_identifier_values(blueprint.target),
        container_identifier_values(sampled_target),
    )
    if before.keys() != after.keys():
        raise ValueError("container presentation cannot change identifier target topology")
    identities: dict[str, str] = {}
    for path, original in before.items():
        sampled = after[path]
        if original in identities and identities[original] != sampled:
            raise ValueError(
                "one source container has inconsistent sampled placement/identity values"
            )
        identities[original] = sampled
    if len(set(identities.values())) != len(identities):
        raise ValueError("distinct source containers cannot collapse to one sampled identifier")
    changed = {path for path in before if before[path] != after[path]}
    records, covered = [], set()
    for region in blueprint.regions:
        paths = _current_paths(blueprint, region)
        owned = tuple(sorted(set(paths) & changed))
        if not owned:
            continue
        originals, sampled = {before[path] for path in owned}, {after[path] for path in owned}
        if len(originals) != 1 or len(sampled) != 1:
            raise ValueError("one container source occurrence owns conflicting current identities")
        original, replacement = originals.pop(), sampled.pop()
        if region.curated_key is None and any(
            not _IDENTIFIER_PATH.fullmatch(path) for path in paths
        ):
            raise ValueError("container source owner mixes ID and non-ID target fields")
        if _SEPARATORS.sub("", region.source) == original:
            value = format_container_identifier(region.source, original, replacement)
            basis = "source_identifier_mask"
        else:
            declaration = blueprint.ownership_data.get("surfaces", {}).get(region.key)
            expected = {"{" + path + "}" for path in owned}
            if (
                region.curated_key is not None
                or not _IDENTIFIER.fullmatch(region.source)
                or not isinstance(declaration, dict)
                or set(declaration) != {"expression"}
                or declaration["expression"] not in expected
            ):
                raise ValueError(
                    "non-reversible container occurrence lacks a reviewed alias policy"
                )
            if not validate_container_number(replacement):
                raise ValueError("reviewed container alias requires a valid sampled ID")
            value, basis = replacement, "reviewed_identifier_alias"
        records.append(
            {
                "key": region.key,
                "occurrence": region.occurrence,
                "sourceIdentifier": original,
                "sampledIdentifier": replacement,
                "source": region.source,
                "rendered": value,
                "targetPaths": owned,
                "basis": basis,
            }
        )
        covered.update(owned)
    if missing := changed - covered:
        raise ValueError(f"changed container identifiers lack reviewed owners: {sorted(missing)}")
    return ContainerIdentifierRenderPlan(
        digest(blueprint.source.encode()), tuple(sorted(after.items())), tuple(records)
    )
