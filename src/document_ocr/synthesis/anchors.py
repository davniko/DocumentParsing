"""Evidence-backed OCR anchors and source-format profiles for synthesis."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.generators import surface_pattern

_PAGE_HEADER = re.compile(r"(?m)^--- PAGE ([1-9][0-9]*) ---$")
_INDEX = re.compile(r"\[([0-9]+)\]")
_DATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("iso_ymd_hyphen", re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")),
    ("numeric_dmy_slash", re.compile(r"^[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4}$")),
    ("numeric_dmy_hyphen", re.compile(r"^[0-9]{1,2}-[0-9]{1,2}-[0-9]{2,4}$")),
    ("numeric_dmy_dot", re.compile(r"^[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{2,4}$")),
    ("month_name", re.compile(r"(?i)^(?:[0-9]{1,2}\W*)?[A-Z]{3,9}|^[A-Z]{3,9}\W*[0-9]")),
)
_INTEGER = re.compile(r"^[+-]?[0-9][0-9., ']*$")
_DECIMAL = re.compile(r"^[+-]?[0-9][0-9., ']*[.,][0-9]+(?:\s*[A-Za-z³]+)?$")
_CONTAINER_LIKE = re.compile(r"^[A-Z]{4}[0-9]{7}$")


def page_texts(joined_raw_text: str) -> tuple[str, ...]:
    matches = list(_PAGE_HEADER.finditer(joined_raw_text))
    numbers = [int(match.group(1)) for match in matches]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError("joined OCR page headers must be contiguous and start at page one")
    pages = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(joined_raw_text)
        pages.append(joined_raw_text[match.end() : end].strip("\n"))
    return tuple(pages)


def leaf_items(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaf_items(child, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from leaf_items(child, f"{path}[{index}]")
    else:
        yield path, value


def normalized_role_path(path: str) -> str:
    return _INDEX.sub("[]", path)


def _normal_path_for_relation(
    path: str, *, target: Mapping[str, Any], normal_target: Mapping[str, Any]
) -> str | None:
    if path == "schemaVersion" or path.endswith((".groupId", ".packageId", ".coverage")):
        return None
    if ".packageIds[" in path:
        return None
    package_match = re.match(r"documentPatch\.cargoPackages\[([0-9]+)\]\.(.+)$", path)
    if package_match:
        package_index = int(package_match.group(1))
        field = package_match.group(2)
        packages = cast(list[dict[str, Any]], target["documentPatch"].get("cargoPackages", []))
        package = packages[package_index]
        group_index = int(cast(str, package["groupId"])[1:]) - 1
        local_index = sum(
            prior["groupId"] == package["groupId"] for prior in packages[:package_index]
        )
        if field in {"packageId", "groupId"}:
            return None
        if field in {"typeCategory", "typeDescription"}:
            normal_packages = cast(
                list[dict[str, Any]],
                normal_target["documentPatch"]["goodsItems"][group_index].get("packages", []),
            )
            normal_field = "type" if "type" in normal_packages[local_index] else "typeCode"
            field = normal_field
        return f"documentPatch.goodsItems[{group_index}].packages[{local_index}].{field}"
    container_match = re.match(r"documentPatch\.containers\[([0-9]+)\]\.(.+)$", path)
    if container_match and container_match.group(2) == "typeCategory":
        container_index = int(container_match.group(1))
        normal_container = cast(
            dict[str, Any], normal_target["documentPatch"]["containers"][container_index]
        )
        normal_field = "typeDescription" if "typeDescription" in normal_container else "typeCode"
        return f"documentPatch.containers[{container_index}].{normal_field}"
    group_match = re.match(r"documentPatch\.cargoGroups\[([0-9]+)\]\.(.+)$", path)
    if group_match:
        group_index, field = int(group_match.group(1)), group_match.group(2)
        if field == "groupId":
            return None
        replacements = {
            "hazardCategory": "hazardClass",
            "subsidiaryHazardCategory": "subsidiaryHazard",
            "packingGroupCategory": "packingGroup",
        }
        for relation_name, normal_name in replacements.items():
            field = field.replace(relation_name, normal_name)
        return f"documentPatch.goodsItems[{group_index}].{field}"
    allocation_match = re.match(r"documentPatch\.cargoAllocationGroups\[([0-9]+)\]\.(.+)$", path)
    if allocation_match:
        allocation_index, field = int(allocation_match.group(1)), allocation_match.group(2)
        groups = cast(
            list[dict[str, Any]], target["documentPatch"].get("cargoAllocationGroups", [])
        )
        group_index = int(cast(str, groups[allocation_index]["groupId"])[1:]) - 1
        if field in {"groupId", "coverage"} or field.startswith("packageIds["):
            return None
        if field.endswith(".packageId"):
            return None
        return (
            f"documentPatch.goodsItems[{group_index}].containerAllocations"
            f"{field.removeprefix('allocations')}"
        )
    return path


def _surface_family(raw_value: str, target_value: Any, path: str) -> str:
    compact = raw_value.strip()
    if "Date" in path or path.endswith("date"):
        for name, pattern in _DATE_PATTERNS:
            if pattern.search(compact):
                return name
        return "date_other"
    if _CONTAINER_LIKE.fullmatch(compact):
        return "iso6346_compact"
    if isinstance(target_value, int) and _INTEGER.fullmatch(compact):
        return "integer"
    if isinstance(target_value, float) or _DECIMAL.fullmatch(compact):
        return "decimal_measure"
    if "\n" in compact:
        return "multiline_text"
    if compact.isupper():
        return "uppercase_text"
    if compact.istitle():
        return "titlecase_text"
    return "text"


def _locate(raw_value: str, excerpt: str, page: str) -> tuple[str, int | None, int | None, int]:
    starts = [match.start() for match in re.finditer(re.escape(raw_value), page)]
    if len(starts) == 1:
        return "exact_unique", starts[0], starts[0] + len(raw_value), 1
    if len(starts) > 1:
        excerpt_starts = [match.start() for match in re.finditer(re.escape(excerpt), page)]
        if len(excerpt_starts) == 1:
            local = excerpt.find(raw_value)
            if local >= 0:
                start = excerpt_starts[0] + local
                return "excerpt_scoped_unique", start, start + len(raw_value), len(starts)
        return "ambiguous_repeated", None, None, len(starts)
    return "not_found", None, None, 0


def build_document_anchors(
    *,
    document_id: str,
    joined_raw_text: str,
    target: Mapping[str, Any],
    normal_target: Mapping[str, Any],
    annotation: Mapping[str, Any],
    normal_path_aliases: Mapping[str, str] | None = None,
    patchable_locations: frozenset[str] = frozenset({"exact_unique", "excerpt_scoped_unique"}),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map audited normal-view evidence onto every relation target fact."""

    pages = page_texts(joined_raw_text)
    evidence_by_path: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for evidence in cast(Sequence[Mapping[str, Any]], annotation.get("evidence", [])):
        evidence_by_path[cast(str, evidence["targetPath"])].append(evidence)
    anchors: list[dict[str, Any]] = []
    expected_fact_count = 0
    covered_fact_count = 0
    patchable_fact_count = 0
    status_counts: Counter[str] = Counter()
    for relation_path, target_value in leaf_items(target):
        normal_path = _normal_path_for_relation(
            relation_path, target=target, normal_target=normal_target
        )
        if normal_path is None:
            continue
        expected_fact_count += 1
        evidence_path = (normal_path_aliases or {}).get(normal_path, normal_path)
        evidence_rows = evidence_by_path.get(evidence_path, [])
        if evidence_rows:
            covered_fact_count += 1
        fact_patchable = False
        for evidence_order, evidence in enumerate(evidence_rows):
            raw_rows = cast(Sequence[Mapping[str, Any]], evidence["rawOcrEvidence"])
            for anchor_order, raw in enumerate(raw_rows):
                page_number = cast(int, raw["pageNumber"])
                if page_number < 1 or page_number > len(pages):
                    raise ValueError(f"evidence page is outside OCR input: {document_id}")
                raw_value = cast(str, raw["rawValue"])
                excerpt = cast(str, raw["ocrExcerpt"])
                status, start, end, occurrences = _locate(
                    raw_value, excerpt, pages[page_number - 1]
                )
                status_counts[status] += 1
                patchable = status in patchable_locations
                fact_patchable |= patchable
                anchors.append(
                    {
                        "anchor_id": "anchor_"
                        + sha256_bytes(
                            canonical_json_bytes(
                                [document_id, relation_path, evidence_order, anchor_order]
                            )
                        )[:24],
                        "document_id": document_id,
                        "relation_target_path": relation_path,
                        "role_path": normalized_role_path(relation_path),
                        "normal_target_path": normal_path,
                        "annotation_evidence_path": evidence_path,
                        "target_value": target_value,
                        "evidence_kind": evidence["evidenceKind"],
                        "normalization_rule": evidence.get("normalizationRule"),
                        "page_number": page_number,
                        "raw_value": raw_value,
                        "ocr_excerpt": excerpt,
                        "location_status": status,
                        "page_start": start,
                        "page_end": end,
                        "page_occurrences": occurrences,
                        "patchable": patchable,
                        "surface_family": _surface_family(raw_value, target_value, relation_path),
                        "surface_pattern": surface_pattern(raw_value),
                    }
                )
        if fact_patchable:
            patchable_fact_count += 1
    return anchors, {
        "document_id": document_id,
        "expected_source_fact_leaves": expected_fact_count,
        "evidence_covered_fact_leaves": covered_fact_count,
        "patchable_fact_leaves": patchable_fact_count,
        "evidence_coverage": covered_fact_count / expected_fact_count,
        "patchable_coverage": patchable_fact_count / expected_fact_count,
        "anchor_status_counts": dict(sorted(status_counts.items())),
    }


def format_inventory(anchors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in anchors:
        key = (
            cast(str, row["role_path"]),
            cast(str, row["surface_family"]),
            cast(str, row["surface_pattern"]),
        )
        groups[key].append(row)
    output = []
    for (role_path, family, pattern), rows in groups.items():
        output.append(
            {
                "format_profile_id": "format_"
                + sha256_bytes(canonical_json_bytes([role_path, family, pattern]))[:20],
                "role_path": role_path,
                "surface_family": family,
                "surface_pattern": pattern,
                "anchor_count": len(rows),
                "document_count": len({row["document_id"] for row in rows}),
                "patchable_anchor_count": sum(bool(row["patchable"]) for row in rows),
            }
        )
    return sorted(output, key=lambda row: (-row["document_count"], row["format_profile_id"]))
