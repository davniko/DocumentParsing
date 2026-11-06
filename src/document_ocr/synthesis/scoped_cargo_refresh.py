"""Migrate a reviewed cargo proposal into a current, repaired synthetic record.

The native proposal is the authority only for owned cargo/equipment spans. All
other current text and labels are retained. Historical candidates supply span
provenance, never replacement private-party values. Ambiguous joins fail closed.
"""

from __future__ import annotations

import copy
import re
from difflib import SequenceMatcher
from functools import lru_cache

from document_ocr.synthesis.curated import digest, flat
from document_ocr.synthesis.curated_descriptions import (
    CompiledDescriptionBlocks,
    CompiledDescriptionField,
    CompiledDescriptionSpan,
    project_rendered_descriptions,
)


def reproject_scoped_descriptions(fresh, source, rendered, edits):
    """Replay certified description membership after retaining noncargo owners.

    An origin or identifier can be protected from refresh while belonging to a
    continuous description block. Copying the fresh description would then
    contradict the composed text. Use its exact certified spans, not a lexical
    replacement of country names that could also occur in product wording.
    """
    projection = fresh["proof"].get("descriptionProjection")
    if projection is None:
        return {}, None
    if projection["sourceSha256"] != digest(source):
        raise ValueError("scoped description source differs from certified projection")
    fields = []
    for field in projection["fields"]:
        spans = []
        previous_end = -1
        for span in field["spans"]:
            start, end = span["sourceByteStart"], span["sourceByteEnd"]
            if not 0 <= start < end <= len(source) or start < previous_end:
                raise ValueError("invalid scoped description span ordering")
            spans.append(CompiledDescriptionSpan(start, end, source[start:end].decode()))
            previous_end = end
        if not spans:
            raise ValueError("empty scoped description membership")
        fields.append(CompiledDescriptionField(field["path"], tuple(spans)))
    blocks = CompiledDescriptionBlocks(
        projection["sourceSha256"], projection["contractSha256"], tuple(fields)
    )
    expected = {
        key: value for key, value in flat(fresh["target"]).items()
        if re.fullmatch(r"documentPatch\.goodsItemDetails\[\d+\]\.description", key)
    }
    if len(fields) != len(blocks.paths) or blocks.paths != expected.keys():
        raise ValueError("scoped description projection membership differs from targets")
    values, verified = project_rendered_descriptions(
        blocks, source, fresh["joinedRawText"].encode(), fresh["proof"]["edits"],
        casing=projection["casing"],
    )
    if values != expected or verified != projection:
        raise ValueError("fresh description projection does not replay its authority")
    return project_rendered_descriptions(
        blocks, source, rendered, edits, casing=projection["casing"]
    )


@lru_cache(maxsize=4)
def _opcodes(before: bytes, after: bytes):
    # Every field of one document uses the same join. Bound the cache to a few
    # document pairs rather than retaining the campaign's texts.
    return SequenceMatcher(None, before, after, autojunk=False).get_opcodes()


def reconstruct_source(candidate: dict) -> bytes:
    raw = candidate["joinedRawText"].encode()
    if digest(raw) != candidate["proof"]["renderedSha256"]:
        raise ValueError("native rendered hash mismatch")
    parts, cursor, shift = [], 0, 0
    for edit in candidate["proof"]["edits"]:
        start = edit["byteStart"] + shift
        end = start + len(edit["after"].encode())
        if start < cursor or raw[start:end] != edit["after"].encode():
            raise ValueError("native edit receipt does not replay")
        parts.extend((raw[cursor:start], edit["before"].encode()))
        cursor = end
        shift += len(edit["after"].encode()) - (edit["byteEnd"] - edit["byteStart"])
    parts.append(raw[cursor:])
    source = b"".join(parts)
    if digest(source) != candidate["proof"]["sourceSha256"]:
        raise ValueError("native source reconstruction differs from receipt")
    return source


def equal_interval(before: bytes, after: bytes, start: int, end: int) -> tuple[int, int]:
    if before == after:
        return start, end
    for tag, a, b, c, _d in _opcodes(before, after):
        if tag == "equal" and a <= start < end <= b:
            return c + start - a, c + end - a
    raise ValueError("source owner changed across source authority revisions")


def rendered_interval(proof: dict, start: int, end: int) -> tuple[int, int]:
    """Map a complete source owner through native edits, including joined owners."""
    left_shift = right_shift = 0
    for edit in proof["edits"]:
        a, b = edit["byteStart"], edit["byteEnd"]
        if a < start < b or a < end < b:
            raise ValueError("new cargo owner cuts through an older complete owner")
        delta = len(edit["after"].encode()) - (b - a)
        if b <= start:
            left_shift += delta
        if b <= end:
            right_shift += delta
    return start + left_shift, end + right_shift


def current_interval(before: bytes, after: bytes, start: int, end: int) -> tuple[int, int]:
    """Transfer exact boundaries; changed content inside a complete owner is allowed."""
    if before == after:
        return start, end
    blocks = _opcodes(before, after)

    def boundary(position, right):
        # An insertion at an owned field edge (e.g. a later HS-format repair)
        # belongs to the complete mapped span, not to an obsolete prefix.
        insertions = [
            (c, d) for tag, a, b, c, d in blocks if tag == "insert" and a == b == position
        ]
        if insertions:
            return max(d for c, d in insertions) if right else min(c for c, d in insertions)
        # Prefer an equal run, including its exact boundary, over an adjacent
        # changed run. No endpoint may fall inside an untraceable replacement.
        for tag, a, b, c, _d in blocks:
            if tag == "equal" and a <= position <= b:
                return c + position - a
        for _tag, a, b, c, d in blocks:
            if position == a:
                return c
            if position == b:
                return d
        raise ValueError("current repair crosses cargo owner boundary")

    a, b = boundary(start, False), boundary(end, True)
    if a >= b:
        raise ValueError("current cargo owner disappeared")
    return a, b


def cargo_path(path: str) -> bool:
    if path.startswith("documentPatch.goodsItemDetails["):
        return ".origin" not in path and not path.endswith(".equipmentIdentifier")
    return path.startswith("documentPatch.containerInformation[") and path.endswith(
        (
            ".sizeCategory",
            ".typeCategory",
            ".tareWeight.value",
            ".temperatureSetpoint.value",
            ".temperatureSetpoint.unit",
        )
    )


def cargo_auxiliary_keys(bindings: dict) -> set[str]:
    """Follow explicit target dependencies, not lexical guesses about owner names."""
    return {
        key
        for key, recipe in bindings.items()
        if ("target" in recipe and cargo_path(recipe["target"])) or "dg_fact" in recipe
    }


def cargo_surface_keys(blueprint) -> set[str]:
    """Include source-only repetitions whose explicit authority is cargo.

    These surfaces have no public target path of their own. In particular, a
    repeated package noun must change with an aggregate multi-group allocation.
    Party aliases and expressions with any non-cargo dependency stay protected.
    """
    products = {v.key for v in blueprint.contract.variables if v.kind == "product"}
    result = set()
    for key, recipe in blueprint.ownership_data.get("surfaces", {}).items():
        paths = [recipe[name] for name in ("package_path", "measure_path") if name in recipe]
        paths.extend(re.findall(r"\{([^{}]+)\}", recipe.get("expression", "")))
        if (
            (paths and all(cargo_path(p) for p in paths))
            or recipe.get("lexical_key") in products
            or "product_reference" in recipe
            or "dg_fact" in recipe
            or "contained_quantity" in recipe
        ):
            result.add(key)
    # A compiled inventory summary owns no public scalar of its own, but its
    # explicit derivation depends on the equipment being refreshed.
    result.update(
        key
        for key, binding in blueprint.historical_bindings.items()
        if binding.get("derivation") == "equipment_receipt"
    )
    return result


def apply_edits(raw: bytes, edits: list[dict]) -> bytes:
    parts, cursor = [], 0
    for edit in sorted(edits, key=lambda e: e["byteStart"]):
        a, b = edit["byteStart"], edit["byteEnd"]
        if a < cursor or raw[a:b] != edit["before"].encode():
            raise ValueError("overlapping or stale current edit")
        parts.extend((raw[cursor:a], edit["after"].encode()))
        cursor = b
    return b"".join((*parts, raw[cursor:]))


def frozen_geometry_edits(source: bytes, current: bytes, offset: int, prefix: str) -> list[dict]:
    """Anchor retained-current changes to complete measured source line fragments."""
    a, b = source.splitlines(keepends=True), current.splitlines(keepends=True)
    ranges = []
    for tag, i, j, k, last in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if i == j:
            if i:
                i, k = i - 1, k - 1
            elif j < len(a):
                j, last = j + 1, last + 1
            else:
                raise ValueError("new frozen text has no source anchor")
        if ranges and i <= ranges[-1][1]:
            previous = ranges.pop()
            i, k = previous[0], previous[2]
            j, last = max(j, previous[1]), max(last, previous[3])
        ranges.append((i, j, k, last))
    offsets = [0]
    for line in a:
        offsets.append(offsets[-1] + len(line))
    return [
        dict(
            key=f"frozen-current:{prefix}:{n}",
            occurrence=0,
            byteStart=offset + offsets[i],
            byteEnd=offset + offsets[j],
            before=b"".join(a[i:j]).decode(),
            after=b"".join(b[k:last]).decode(),
            targetPaths=[],
        )
        for n, (i, j, k, last) in enumerate(ranges)
    ]


def complete_cargo_edits(native: dict, regions, private_keys: set[str]) -> list[dict]:
    """Include unchanged native owners: an older descendant may differ there."""
    source = reconstruct_source(native)
    edits = list(native["proof"]["edits"])
    for region in regions:
        if not (any(cargo_path(p) for p in region.target_paths) or region.key in private_keys):
            continue
        overlaps = [e for e in edits if e["byteStart"] < region.end and e["byteEnd"] > region.start]
        if overlaps:
            # A complete expression may replace several smaller source owners.
            if not any(
                e["byteStart"] <= region.start and e["byteEnd"] >= region.end for e in overlaps
            ):
                raise ValueError("native expression only partially covers cargo owner")
            continue
        if source[region.start : region.end].decode() != region.source:
            raise ValueError("unchanged cargo owner does not match source")
        edits.append(
            dict(
                key=region.key,
                occurrence=region.occurrence,
                byteStart=region.start,
                byteEnd=region.end,
                before=region.source,
                after=region.source,
                targetPaths=list(region.target_paths),
            )
        )
    return sorted(edits, key=lambda e: e["byteStart"])


def migrate(
    current: dict, historical: dict, native: dict, *, private_keys: set[str], regions
) -> tuple[dict, dict]:
    """Produce a staged raw/target record and two exact proofs; no positioning/write."""
    sid = current["sourceDocumentId"]
    if historical["sourceDocumentId"] != sid or native["sourceDocumentId"] != sid:
        raise ValueError("cargo migration changes source identity")
    old_source, new_source = reconstruct_source(historical), reconstruct_source(native)
    old_raw, now = historical["joinedRawText"].encode(), current["joinedRawText"].encode()
    if digest(now) != current["joinedRawTextSha256"]:
        raise ValueError("current input hash mismatch")
    selected, source_edits = [], []
    for edit in complete_cargo_edits(native, regions, private_keys):
        paths = edit.get("targetPaths", [])
        relevant = any(cargo_path(p) for p in paths) or edit["key"] in private_keys
        if not relevant:
            continue
        if any(not cargo_path(p) for p in paths):
            raise ValueError("cargo owner also owns protected noncargo facts: " + edit["key"])
        a, b = equal_interval(new_source, old_source, edit["byteStart"], edit["byteEnd"])
        a, b = rendered_interval(historical["proof"], a, b)
        a, b = current_interval(old_raw, now, a, b)
        selected.append({**edit, "byteStart": a, "byteEnd": b, "before": now[a:b].decode()})
        source_edits.append(edit)
    if not selected:
        raise ValueError("no owned cargo edits")
    result_raw = apply_edits(now, selected)
    result = copy.deepcopy(current)
    old_target, new_target = current["target"], native["target"]["documentPatch"]
    old_containers, new_containers = (
        old_target["containerInformation"],
        new_target["containerInformation"],
    )
    if len(old_containers) != len(new_containers):
        raise ValueError("cargo migration changes container count")
    identities = {
        new["equipmentIdentifier"]: old["equipmentIdentifier"]
        for old, new in zip(old_containers, new_containers, strict=True)
    }
    if len(identities) != len(old_containers):
        raise ValueError("duplicate proposal equipment identifier")
    for before, after, proposed in zip(
        old_containers, result["target"]["containerInformation"], new_containers, strict=True
    ):
        for key in ("sizeCategory", "typeCategory", "tareWeight", "temperatureSetpoint"):
            if (key in before) != (key in proposed):
                raise ValueError("cargo migration changes equipment field presence")
            if key in proposed:
                after[key] = copy.deepcopy(proposed[key])
    if len(old_target["goodsItemDetails"]) != 1 or len(new_target["goodsItemDetails"]) != 1:
        raise ValueError("bounded refresh requires one goods item")
    goods = copy.deepcopy(new_target["goodsItemDetails"][0])
    old_goods = old_target["goodsItemDetails"][0]
    goods.pop("origin", None)
    if "origin" in old_goods:
        goods["origin"] = copy.deepcopy(old_goods["origin"])
    for allocation in goods.get("splitGoodsPlacement", []):
        allocation["equipmentIdentifier"] = identities[allocation["equipmentIdentifier"]]
    if {a["equipmentIdentifier"] for a in goods.get("splitGoodsPlacement", [])} != {
        a["equipmentIdentifier"] for a in old_goods.get("splitGoodsPlacement", [])
    }:
        raise ValueError("cargo migration changes allocation visibility")
    result["target"]["goodsItemDetails"] = [goods]
    result.update(joinedRawText=result_raw.decode(), joinedRawTextSha256=digest(result_raw))

    # Preserve measured field boundaries for retained parties/identifiers too.
    # Merging a name and postal block into one generic diff can lose otherwise
    # valid anchors when one original line was unpositioned.
    geometry_pairs = list(zip(source_edits, selected, strict=True))
    for region in regions:
        if any(e["byteStart"] < region.end and e["byteEnd"] > region.start for e in source_edits):
            continue
        try:
            a, b = equal_interval(new_source, old_source, region.start, region.end)
            a, b = rendered_interval(historical["proof"], a, b)
            a, b = current_interval(old_raw, now, a, b)
        except ValueError:
            # This is frozen context, not a replacement. The exact intervening
            # current bytes remain in the anchored diff below.
            continue
        if any(e["byteStart"] < b and e["byteEnd"] > a for _, e in geometry_pairs):
            continue
        geometry_pairs.append(
            (
                dict(
                    key=region.key,
                    occurrence=region.occurrence,
                    byteStart=region.start,
                    byteEnd=region.end,
                    before=region.source,
                    after=now[a:b].decode(),
                    targetPaths=list(region.target_paths),
                ),
                dict(byteStart=a, byteEnd=b),
            )
        )
    geometry_pairs.sort(key=lambda pair: pair[0]["byteStart"])
    # Reconstruct geometry from the current text, not obsolete native parties.
    geometry_edits, source_cursor, current_cursor = [], 0, 0
    for source_edit, current_edit in geometry_pairs:
        sa, sb = source_edit["byteStart"], source_edit["byteEnd"]
        ca, cb = current_edit["byteStart"], current_edit["byteEnd"]
        if ca < current_cursor:
            raise ValueError("current mapped owner order differs from source")
        a, b = new_source[source_cursor:sa], now[current_cursor:ca]
        geometry_edits.extend(frozen_geometry_edits(a, b, source_cursor, str(len(geometry_edits))))
        geometry_edits.append(copy.deepcopy(source_edit))
        source_cursor, current_cursor = sb, cb
    a, b = new_source[source_cursor:], now[current_cursor:]
    geometry_edits.extend(frozen_geometry_edits(a, b, source_cursor, str(len(geometry_edits))))
    if apply_edits(new_source, geometry_edits) != result_raw:
        raise ValueError("hybrid geometry proof does not reproduce current-scoped text")
    geometry_proof = {
        **copy.deepcopy(native["proof"]),
        "renderedSha256": digest(result_raw),
        "edits": geometry_edits,
    }
    descriptions, description_proof = reproject_scoped_descriptions(
        native, new_source, result_raw, geometry_edits
    )
    if description_proof is not None:
        geometry_proof["descriptionProjection"] = description_proof
        for index, goods in enumerate(result["target"]["goodsItemDetails"]):
            path = f"documentPatch.goodsItemDetails[{index}].description"
            if path in descriptions:
                goods["description"] = descriptions[path]
    before_flat, after_flat = flat(old_target), flat(result["target"])
    changed = {
        key
        for key in before_flat.keys() | after_flat.keys()
        if before_flat.get(key) != after_flat.get(key)
    }
    if any(not cargo_path("documentPatch." + p) for p in changed):
        raise ValueError("cargo migration changed protected target")
    return result, dict(
        currentProof=dict(
            sourceSha256=digest(now), renderedSha256=digest(result_raw), edits=selected
        ),
        geometryProof=geometry_proof,
        targetChanges=sorted(changed),
        beforeRecordSha256=digest(current),
        nativeCandidateId=native["documentId"],
        historicalCandidateId=historical["documentId"],
    )
