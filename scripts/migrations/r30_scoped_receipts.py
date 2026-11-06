"""Compose a fresh cargo section with an exact retained shipment receipt."""

import copy

from document_ocr.synthesis.curated import assign, digest, flat
from document_ocr.synthesis.scoped_cargo_refresh import (
    apply_edits,
    cargo_path,
    reconstruct_source,
    reproject_scoped_descriptions,
)
from document_ocr.synthesis.template_compiler.descendant import _number_word_value


def unchanged_cargo_authority(edit, fresh, fresh_targets):
    """Verify a source-equal value at its own occurrence, not other aliases."""
    values = []
    for field in ("surfaceValues", "renderValues"):
        if edit["key"] not in fresh.get(field, {}):
            continue
        value = fresh[field][edit["key"]]
        if isinstance(value, list):
            occurrence = edit["occurrence"]
            if type(occurrence) is not int or not 0 <= occurrence < len(value):
                raise ValueError("unchanged cargo occurrence lacks sampled authority")
            value = value[occurrence]
        values.append(value)
    paths = edit.get("targetPaths", [])
    if not values and paths and all(p in fresh_targets for p in paths):
        values = [str(fresh_targets[p]) for p in paths]
    if not values:
        return False
    if all(value == edit["before"] for value in values):
        return True
    # A number-word repetition is a presentation of an explicitly sampled count.
    # Never apply this interpretation to private identifiers or product prose.
    if paths and all(p.endswith(".packageQuantity") for p in paths):
        count = _number_word_value(edit["before"])
        return count is not None and all(str(value) == str(count) for value in values)
    return False


def compose_saved_scoped_stage(stage, historical, fresh):
    """Reuse the refresh's complete source proof and sampled cargo authority.

    This does not infer old private-owner boundaries by diffing descriptions.
    Both the source-to-final proof and the before-to-final proof must replay,
    and fresh cargo labels must agree with the saved refreshed labels exactly.
    """
    before, after, evidence = stage["before"], stage["after"], stage["evidence"]
    source = reconstruct_source(historical)
    if reconstruct_source(fresh) != source:
        raise ValueError("scoped stage changes source authority")
    if not (before["sourceDocumentId"] == after["sourceDocumentId"] == historical["sourceDocumentId"] == fresh["sourceDocumentId"]):
        raise ValueError("scoped stage changes source membership")
    if before["documentId"] != after["documentId"] or digest(before) != evidence["beforeRecordSha256"]:
        raise ValueError("scoped stage before-record identity differs")
    if fresh["documentId"] != evidence["nativeCandidateId"] or digest(fresh) != evidence["nativeSha256"]:
        raise ValueError("scoped stage fresh cargo receipt differs")
    current_proof, proof = evidence["currentProof"], evidence["geometryProof"]
    for raw, p in ((before["joinedRawText"].encode(), current_proof), (source, proof)):
        if p["sourceSha256"] != digest(raw) or p["renderedSha256"] != digest(after["joinedRawText"].encode()):
            raise ValueError("scoped stage proof fingerprints differ")
        if apply_edits(raw, p["edits"]).decode() != after["joinedRawText"]:
            raise ValueError("scoped stage does not replay every retained byte")
    a, b = flat(before["target"]), flat(after["target"])
    changed = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
    if any(not cargo_path("documentPatch." + k) for k in changed):
        raise ValueError("saved scoped refresh changes protected targets")
    old_inventory = before["target"]["containerInformation"]
    inventory = after["target"]["containerInformation"]
    fresh_target = fresh["target"]["documentPatch"]
    fresh_inventory = fresh_target["containerInformation"]
    if not len(old_inventory) == len(inventory) == len(fresh_inventory):
        raise ValueError("saved scoped refresh changes inventory length")
    ids = {n["equipmentIdentifier"]: o["equipmentIdentifier"]
           for o, n in zip(inventory, fresh_inventory, strict=True)}
    if len(ids) != len(inventory) or len(set(ids.values())) != len(inventory):
        raise ValueError("saved scoped refresh has duplicate equipment")
    for public, planned in zip(inventory, fresh_inventory, strict=True):
        for key in ("sizeCategory", "typeCategory", "tareWeight", "temperatureSetpoint"):
            if public.get(key) != planned.get(key):
                raise ValueError("saved equipment disagrees with fresh sampled cargo")
    if len(fresh_target["goodsItemDetails"]) != 1 or len(before["target"]["goodsItemDetails"]) != 1:
        raise ValueError("saved scoped refresh requires one goods item")
    goods = copy.deepcopy(fresh_target["goodsItemDetails"][0])
    descriptions, description_proof = reproject_scoped_descriptions(
        fresh, source, after["joinedRawText"].encode(), proof["edits"]
    )
    if descriptions:
        goods["description"] = descriptions["documentPatch.goodsItemDetails[0].description"]
    goods.pop("origin", None)
    if "origin" in before["target"]["goodsItemDetails"][0]:
        goods["origin"] = copy.deepcopy(before["target"]["goodsItemDetails"][0]["origin"])
    for allocation in goods.get("splitGoodsPlacement", []):
        allocation["equipmentIdentifier"] = ids[allocation["equipmentIdentifier"]]
    if [goods] != after["target"]["goodsItemDetails"]:
        raise ValueError("saved goods differ from the explicit fresh cargo receipt")
    result = copy.deepcopy(historical)
    result.update(documentId=after["documentId"], joinedRawText=after["joinedRawText"],
                  joinedRawTextSha256=digest(after["joinedRawText"].encode()), proof=copy.deepcopy(proof))
    result["target"]["documentPatch"] = copy.deepcopy(after["target"])
    if description_proof is not None:
        result["proof"]["descriptionProjection"] = description_proof
    result["physical"] = copy.deepcopy(fresh["physical"])
    old_source = historical["scenario"]["cargo"]["containerInformation"]
    new_source = fresh["scenario"]["cargo"]["containerInformation"]
    if [r["equipmentIdentifier"] for r in old_source] != [r["equipmentIdentifier"] for r in new_source]:
        raise ValueError("saved scoped refresh changes source equipment ownership")
    result["scenario"]["goods_identities"] = copy.deepcopy(fresh["scenario"]["goods_identities"])
    result["scenario"]["cargo"] = copy.deepcopy(fresh["scenario"]["cargo"])
    old_geo = result["scenario"]["provenance"].get("geographyEligibility")
    result["scenario"]["provenance"] = copy.deepcopy(fresh["scenario"]["provenance"])
    if old_geo is not None:
        result["scenario"]["provenance"]["geographyEligibility"] = old_geo
    source_goods = result["scenario"]["cargo"]["goodsItemDetails"][0]
    source_goods.pop("origin", None)
    if "origin" in goods:
        source_goods["origin"] = copy.deepcopy(goods["origin"])
    cargo_keys = set(evidence["privateOwnerKeys"]) | set(evidence.get("physicalPlanOwnerKeys", []))
    cargo_keys.update(e["key"] for e in fresh["proof"]["edits"] if any(cargo_path(p) for p in e.get("targetPaths", [])))
    for field in ("renderValues", "surfaceValues"):
        for key in cargo_keys:
            if key in fresh.get(field, {}):
                result.setdefault(field, {})[key] = copy.deepcopy(fresh[field][key])
        for key in result.get(field, {}):
            values = {e["after"] for e in proof["edits"] if e["key"] == key}
            if len(values) == 1 and not isinstance(result[field][key], list):
                result[field][key] = next(iter(values))
    for field in ("packagePresentations", "numericWordingProjection"):
        if field in fresh:
            result[field] = copy.deepcopy(fresh[field])
        else:
            result.pop(field, None)
    result["migrationSavedScopedProof"] = {
        "status": "EXACT_SAVED_SCOPED_PROOF_REQUIRES_NATIVE_REVIEW",
        "beforeRecordSha256": digest(before), "afterRecordSha256": digest(after),
        "freshCandidateId": fresh["documentId"], "targetChanges": sorted(changed),
    }
    if reconstruct_source(result) != source:
        raise ValueError("saved scoped receipt cannot reconstruct its source")
    return result


def compose_cargo(current, retained, fresh, private_keys, *, preserved_mixed_owners=()):
    """Replace explicit cargo owners, including split/compound owner topologies."""
    source = reconstruct_source(retained)
    if (
        reconstruct_source(fresh) != source
        or current["sourceDocumentId"] != fresh["sourceDocumentId"]
        or current["joinedRawText"] != retained["joinedRawText"]
    ):
        raise ValueError("scoped receipt requires the same source and exact retained text")
    protected_paths = set()
    preserved = set(preserved_mixed_owners)
    if len(preserved) != len(preserved_mixed_owners):
        raise ValueError('duplicate preserved mixed owner')
    for key in preserved:
        a = [e for e in retained['proof']['edits'] if e['key'] == key]
        b = [e for e in fresh['proof']['edits'] if e['key'] == key]
        def contract(edits):
            return [(e['byteStart'], e['byteEnd'], e['before'], sorted(e['targetPaths'])) for e in edits]
        if not a or contract(a) != contract(b):
            raise ValueError('preserved mixed owner requires identical source and target ownership')
        paths = {p for e in a for p in e['targetPaths']}
        cargo = {p for p in paths if cargo_path(p)}
        if not cargo or cargo == paths:
            raise ValueError('preserved owner must own both cargo and protected facts')
        protected_paths.update(cargo)
    selected = [
        e
        for e in fresh["proof"]["edits"]
        if e['key'] not in preserved
        and (any(cargo_path(p) for p in e.get("targetPaths", [])) or e["key"] in private_keys)
    ]
    if not selected:
        raise ValueError("no explicit cargo owners")
    fresh_targets = flat(fresh["target"])
    if any(any(not cargo_path(p) for p in e.get("targetPaths", [])) for e in selected):
        raise ValueError("cargo owner also contains protected facts")
    kept = []
    for edit in retained["proof"]["edits"]:
        overlaps = [
            e
            for e in selected
            if e["byteStart"] < edit["byteEnd"] and e["byteEnd"] > edit["byteStart"]
        ]
        if overlaps:
            if any(not cargo_path(p) for p in edit.get("targetPaths", [])) or (
                not edit.get("targetPaths") and edit["key"] not in private_keys
            ):
                raise ValueError("replacement crosses a protected retained owner")
            for e in overlaps:
                if not (
                    e["byteStart"] <= edit["byteStart"] < edit["byteEnd"] <= e["byteEnd"]
                    or edit["byteStart"] <= e["byteStart"] < e["byteEnd"] <= edit["byteEnd"]
                ):
                    raise ValueError("cargo owners have crossing, not nested, boundaries")
        elif edit['key'] not in preserved and (any(cargo_path(p) for p in edit.get("targetPaths", [])) or edit["key"] in private_keys):
            if any(not cargo_path(p) for p in edit.get("targetPaths", [])):
                raise ValueError("unchanged replacement also owns protected facts")
            # Proof edits omit values equal to source. That is not permission
            # to keep a different value from the previous shipment.
            if not unchanged_cargo_authority(edit, fresh, fresh_targets):
                raise ValueError("unedited cargo owner lacks exact unchanged sampled authority")
            unchanged = copy.deepcopy(edit)
            unchanged["after"] = edit["before"]
            selected.append(unchanged)
        else:
            kept.append(edit)
    edits = sorted(copy.deepcopy(kept + selected), key=lambda e: e["byteStart"])
    raw = apply_edits(source, edits).decode()
    before = current["target"]
    target = copy.deepcopy(before)
    new = fresh["target"]["documentPatch"]
    old_containers, new_containers = before["containerInformation"], new["containerInformation"]
    if len(old_containers) != len(new_containers):
        raise ValueError("scoped cargo changes container count")
    ids = {
        n["equipmentIdentifier"]: o["equipmentIdentifier"]
        for o, n in zip(old_containers, new_containers, strict=True)
    }
    if len(ids) != len(old_containers):
        raise ValueError("duplicate equipment identity")
    for output, proposal in zip(target["containerInformation"], new_containers, strict=True):
        for key in ("sizeCategory", "typeCategory", "tareWeight", "temperatureSetpoint"):
            if (key in output) != (key in proposal):
                raise ValueError("scoped cargo changes equipment field presence")
            if key in proposal:
                output[key] = copy.deepcopy(proposal[key])
    if len(before["goodsItemDetails"]) != 1 or len(new["goodsItemDetails"]) != 1:
        raise ValueError("scoped cargo requires one goods item")
    goods = copy.deepcopy(new["goodsItemDetails"][0])
    descriptions, description_proof = reproject_scoped_descriptions(
        fresh, source, raw.encode(), edits
    )
    if descriptions:
        goods["description"] = descriptions["documentPatch.goodsItemDetails[0].description"]
    goods.pop("origin", None)
    if "origin" in before["goodsItemDetails"][0]:
        goods["origin"] = copy.deepcopy(before["goodsItemDetails"][0]["origin"])
    for allocation in goods.get("splitGoodsPlacement", []):
        allocation["equipmentIdentifier"] = ids[allocation["equipmentIdentifier"]]
    if {a["equipmentIdentifier"] for a in goods.get("splitGoodsPlacement", [])} != {
        a["equipmentIdentifier"]
        for a in before["goodsItemDetails"][0].get("splitGoodsPlacement", [])
    }:
        raise ValueError("scoped cargo changes allocation visibility")
    target["goodsItemDetails"] = [goods]
    old_authority = flat({'documentPatch': before})
    for path in protected_paths:
        assign({'documentPatch': target}, path, copy.deepcopy(old_authority[path]))
    old_flat, new_flat = flat(before), flat(target)
    changed = sorted(
        k for k in old_flat.keys() | new_flat.keys() if old_flat.get(k) != new_flat.get(k)
    )
    if any(not cargo_path("documentPatch." + k) for k in changed):
        raise ValueError("scoped cargo changes protected targets")
    result = copy.deepcopy(retained)
    result.update(
        documentId=current["documentId"],
        joinedRawText=raw,
        joinedRawTextSha256=digest(raw.encode()),
    )
    result["proof"].update(edits=edits, renderedSha256=digest(raw.encode()))
    if description_proof is not None:
        result["proof"]["descriptionProjection"] = description_proof
    result["target"]["documentPatch"] = target
    result["physical"] = copy.deepcopy(fresh["physical"])
    result["scenario"]["goods_identities"] = copy.deepcopy(fresh["scenario"]["goods_identities"])
    # Planning cargo identifies the immutable source slots. Rendered/public IDs
    # belong to the target, not this lookup used by newly exposed repetitions.
    old_source = retained["scenario"]["cargo"]["containerInformation"]
    new_source = fresh["scenario"]["cargo"]["containerInformation"]
    source_ids = [c["equipmentIdentifier"] for c in old_source]
    if source_ids != [c["equipmentIdentifier"] for c in new_source] or len(source_ids) != len(ids):
        raise ValueError("scoped cargo changes source equipment ownership")
    result["scenario"]["cargo"] = copy.deepcopy(fresh["scenario"]["cargo"])
    source_goods = result["scenario"]["cargo"]["goodsItemDetails"][0]
    for path in protected_paths:
        assign({'documentPatch': result['scenario']['cargo']}, path, copy.deepcopy(old_authority[path]))
    source_goods.pop("origin", None)
    if "origin" in before["goodsItemDetails"][0]:
        source_goods["origin"] = copy.deepcopy(before["goodsItemDetails"][0]["origin"])
    old_geo = result["scenario"]["provenance"].get("geographyEligibility")
    result["scenario"]["provenance"] = copy.deepcopy(fresh["scenario"]["provenance"])
    if old_geo is not None:
        result["scenario"]["provenance"]["geographyEligibility"] = old_geo
    for field in ("renderValues", "surfaceValues"):
        for e in selected:
            if e["key"] in fresh.get(field, {}):
                result.setdefault(field, {})[e["key"]] = fresh[field][e["key"]]
    for field in ("packagePresentations", "numericWordingProjection"):
        if field in fresh:
            result[field] = copy.deepcopy(fresh[field])
        else:
            result.pop(field, None)
    result["validation"] = {"status": "SCOPED_CARGO_REQUIRES_NATIVE_REVIEW"}
    result["migrationCargoRefresh"] = {
        "preservedMixedOwners": sorted(preserved),
        "previousRecordSha256": digest(current),
        "targetChanges": changed,
        "freshCandidateId": fresh["documentId"],
        "privateOwnerKeys": sorted(private_keys),
        "retainedOwnerCount": len(kept),
        "freshCargoOwnerCount": len(selected),
    }
    if reconstruct_source(result) != source:
        raise ValueError("composed source replay differs")
    after = copy.deepcopy(current)
    after.update(target=target, joinedRawText=raw, joinedRawTextSha256=digest(raw.encode()))
    return result, after
