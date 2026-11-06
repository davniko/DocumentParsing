"""Keep actual retained edits separate from discovered, unchanged field owners."""

import copy

from document_ocr.synthesis.scoped_cargo_refresh import apply_edits, reconstruct_source


def discard_unused_noops(candidate, overlay):
    """Normalize unchanged receipt entries, never consumed or target-bearing owners.

    An exact no-op is not a synthetic value to transplant into a different OCR
    segmentation. Native text and all native acceptance gates remain authoritative.
    This operation alone does not authorize a rendered document or publication.
    """
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)

    selectors = {"key", "byteStart", "byteEnd", "spans", "regions", "fragments"}
    dependencies = set(strings({k: v for k, v in overlay.items() if k != "overrides"}))
    dependencies.update(strings([
        value for rule in overlay["overrides"] for key, value in rule.items() if key not in selectors
    ]))
    removed = []
    for edit in candidate["proof"]["edits"]:
        if edit["before"] != edit["after"] or edit.get("targetPaths") or edit["key"] in dependencies:
            continue
        rules = [rule for rule in overlay["overrides"] if rule["key"] == edit["key"]
                 and ("byteStart" not in rule or rule["byteStart"] == edit["byteStart"])
                 and ("byteEnd" not in rule or rule["byteEnd"] == edit["byteEnd"])]
        if not any(set(rule) - selectors for rule in rules):
            removed.append(edit)
    result = copy.deepcopy(candidate)
    result["proof"]["edits"] = [edit for edit in result["proof"]["edits"] if edit not in removed]
    source = reconstruct_source(candidate)
    if (reconstruct_source(result) != source
            or apply_edits(source, candidate["proof"]["edits"]).decode() != candidate["joinedRawText"]
            or apply_edits(source, result["proof"]["edits"]).decode() != candidate["joinedRawText"]):
        raise ValueError("unused no-op normalization changed retained bytes")
    return result, removed


def discard_added_noops(candidate, historical):
    """Drop only augmentation entries that do not modify a single source byte.

    Existing owners, including existing no-op owners used as explicit sampled
    authorities, remain intact. This normalization does not authorize native
    publication; rendering, semantic review and all native guards still apply.
    """
    source = reconstruct_source(historical)
    if candidate["sourceDocumentId"] != historical["sourceDocumentId"] or reconstruct_source(candidate) != source:
        raise ValueError("owner normalization changes source authority")
    if apply_edits(source, candidate["proof"]["edits"]).decode() != candidate["joinedRawText"]:
        raise ValueError("retained owners do not reconstruct their text")
    keys = {(e["key"], e["byteStart"], e["byteEnd"]) for e in historical["proof"]["edits"]}
    removed = [e for e in candidate["proof"]["edits"]
               if (e["key"], e["byteStart"], e["byteEnd"]) not in keys and e["before"] == e["after"]]
    result = copy.deepcopy(candidate)
    result["proof"]["edits"] = [e for e in result["proof"]["edits"] if e not in removed]
    if reconstruct_source(result) != source or apply_edits(source, result["proof"]["edits"]).decode() != candidate["joinedRawText"]:
        raise ValueError("no-op normalization changed retained bytes")
    return result, removed
