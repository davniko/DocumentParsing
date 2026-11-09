"""Final-input casing projection for the R31 native Paddle target contract."""

from __future__ import annotations

import copy
import re

from document_ocr.labeling_agents.source_casing import (
    PrintedCaseIndex,
    case_key,
    recover_case,
    selected_fields,
)
from document_ocr.spatial_inputs.paddle import RegionSpan, pages_from_record
from document_ocr.synthesis.paddle_bindings import PaddleReplacement
from document_ocr.synthesis.paddle_render import (
    literal_owner_value,
    rendered_owner_projection,
    source_passage_projection,
    strip_postal_captions,
)

POLICY = "printed-party-and-goods-v1"


def target_pointer(path):
    return "/" + re.sub(r"\[(\d+)\]", r".\1", path.removeprefix("documentPatch.")).replace(".", "/")


def rendered_case_scopes(rendered, candidate, overlay, source):
    wanted = dict(selected_fields(rendered["target"]))
    paths = {
        f"{e['key']}@{e['byteStart']}:{e['byteEnd']}": set(
            map(target_pointer, e.get("targetPaths", []))
        )
        for e in candidate["proof"]["edits"]
    }
    for extra in overlay.get("extras", []):
        if "target" in extra and "owner" in extra:
            paths.setdefault(extra["owner"], set()).add(extra["target"])
    refs = {}
    for receipt in rendered["proof"]:
        owner = receipt["owner"].split(":frame:")[0]
        for path in paths.get(owner, []):
            refs.setdefault(path, set()).update((s["page"], s["region"]) for s in receipt["spans"])

    def receipt_refs(selectors, *, keys=False):
        return {
            (s["page"], s["region"])
            for receipt in rendered["proof"]
            if (receipt["owner"].split("@")[0] if keys else receipt["owner"].split(":frame:")[0])
            in selectors
            for s in receipt["spans"]
        }

    def rule_refs(rule):
        positions = set()
        for piece in rule.get("groups", [rule]):
            positions.update(tuple(r) for r in piece.get("regions", []))
            positions.update((p, i) for p, i, _ in piece.get("fragments", []))
            positions.update(tuple(s[:2]) for s in piece.get("spans", []))
        return positions

    # Follow the renderer's target-projection precedence. A reviewed projection
    # may deliberately replace a broader historical company/address owner.
    for action in overlay.get("targetFromOwners", []):
        refs[action["target"]] = receipt_refs(action["keys"], keys=True)
    for action in overlay.get("targetFromRenderedOwners", []):
        refs[action["target"]] = receipt_refs(action["owners"])
    for action in overlay.get("targetFromSourceSpans", []):
        refs[action["target"]] = rule_refs(action)
    for action in overlay.get("literalProjections", []):
        positions = set()
        for piece in action["pieces"]:
            if "fromKey" in piece:
                positions.update(receipt_refs([piece["fromKey"]], keys=True))
            for extra in overlay.get("extras", []):
                if any(
                    k in piece and extra.get(k) == piece[k] for k in ("wording", "scenarioText")
                ):
                    positions.update(rule_refs(extra))
        if positions:
            refs[action["target"]] = positions
    reviewed = overlay.get("sourceCaseRegions", {})
    for path, positions in reviewed.items():
        if path not in wanted:
            continue
        if not positions or len({tuple(r) for r in positions}) != len(positions):
            raise ValueError("primary casing region selection must be nonempty and unique")
        refs[path] = {tuple(r) for r in positions}
    scopes = {}
    actual_regions = [
        {(tuple(r["box"]), r["text"]) for r in page["regions"]} for page in rendered["pages"]
    ]
    for path, owned in refs.items():
        if path not in wanted:
            continue
        pieces = []
        for page in rendered["geometry"]["pages"]:
            page_i = page["page"]
            actual = actual_regions[page_i]
            for block in page["blocks"]:
                if not any((page_i, r) in owned for r in block["sourceRegions"]):
                    continue
                if any(
                    (tuple(f["box"]), f["text"]) not in actual for f in block["physicalFragments"]
                ):
                    raise ValueError("case scope refers to text absent from final Paddle input")
                fragments = "\n".join(f["text"] for f in block["physicalFragments"])
                if "".join(fragments.split()) != "".join(block["text"].split()):
                    raise ValueError("case scope differs from printed characters")
                pieces.append(fragments)
        if pieces:
            scopes[path] = PrintedCaseIndex("\n".join(pieces))
    # These are already reviewed character-level target projections. Reuse them
    # before uppercasing, including slices such as DRUMS in DRUMSHS:3302.90.
    # Region-level searching alone loses that explicit character boundary.
    for action in overlay.get("targetFromOwners", []):
        if action["target"] not in wanted or action["target"] in reviewed:
            continue
        literal = literal_owner_value(
            rendered["proof"],
            action["keys"],
            action["separator"],
            deduplicate_postal_components=action.get("deduplicatePostalComponents", False),
            postal_code_key=action.get("deduplicatePostalCodeKey"),
        )
        scopes[action["target"]] = PrintedCaseIndex(strip_postal_captions(literal, action))
    for action in overlay.get("targetFromRenderedOwners", []):
        if action["target"] not in wanted or action["target"] in reviewed:
            continue
        scopes[action["target"]] = PrintedCaseIndex(
            rendered_owner_projection(
                rendered["proof"], {k: v for k, v in action.items() if k != "targetCase"}
            )
        )
    if overlay.get("targetFromSourceSpans"):
        pages = pages_from_record(source)
        replacements = tuple(
            PaddleReplacement(r["owner"], tuple(RegionSpan(**s) for s in r["spans"]), r["after"])
            for r in rendered["proof"]
        )
        for action in overlay["targetFromSourceSpans"]:
            if action["target"] not in wanted or action["target"] in reviewed:
                continue
            literal = source_passage_projection(
                pages, replacements, {k: v for k, v in action.items() if k != "targetCase"}
            )
            if isinstance(literal, str):
                scopes[action["target"]] = PrintedCaseIndex(literal)
    return scopes


def assign(target, path, value):
    parts = path.strip("/").split("/")
    owner = target
    for part in parts[:-1]:
        owner = owner[int(part)] if isinstance(owner, list) else owner[part]
    owner[parts[-1]] = value


def project_source_casing(rendered, candidate, overlay, source):
    scopes = rendered_case_scopes(rendered, candidate, overlay, source)
    index = PrintedCaseIndex("\n".join(r["text"] for p in rendered["pages"] for r in p["regions"]))
    target, receipts, issues = copy.deepcopy(rendered["target"]), [], []
    for path, before in selected_fields(target):
        result = recover_case(before, index, owned=scopes.get(path))
        if "unresolved" in result:
            issues.append({"path": path, **result})
            continue
        after = result["value"]
        if case_key(after) != case_key(before):
            raise ValueError("printed case projection changed target content")
        assign(target, path, after)
        receipts.append(
            {"path": path, "before": before, "after": after, "method": result["method"]}
        )
    if issues:
        raise ValueError(f"printed casing requires explicit occurrence review: {issues}")
    return {**rendered, "target": target, "sourceCasing": {"policy": POLICY, "fields": receipts}}
