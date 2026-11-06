"""Stage a reviewed aggregate-only refresh when new tare rows contradict an old sum."""

import argparse
import copy
import json
from decimal import Decimal

from migrate_r30_synthetic import ART, DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.paddle_render import extra_value
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog
from document_ocr.synthesis.scoped_cargo_refresh import apply_edits, reconstruct_source


def replace_total(candidate, slot, key, old, new, provenance):
    """Change exactly one explicit source-only aggregate; preserve every public fact."""
    result = copy.deepcopy(candidate)
    if candidate["surfaceValues"][key] != [old] or old == new:
        raise ValueError("aggregate refresh needs one changed sampled surface")
    if slot["key"] != key or slot["before"] != old or slot["targetPaths"]:
        raise ValueError("aggregate must have an exact source-only owner")
    if any(e["key"] == key for e in candidate["proof"]["edits"]):
        raise ValueError("this refresh requires the reviewed unchanged original aggregate")
    tares = result["physical"]["tares"]
    if len(tares) != 1 or tares[0]["kg"] != old:
        raise ValueError("physical aggregate differs from its sampled surface")
    value = Decimal(new)
    if not value.is_finite() or value <= 0:
        raise ValueError("aggregate must be positive and finite")
    result["surfaceValues"][key] = [new]
    result["physical"]["tares"] = [
        {
            "owners": tares[0]["owners"],
            "kg": new,
            "ownershipBasis": "reviewed_equipment_pair_tare_sum",
            "previous": tares[0],
            "resamplingEvidence": provenance,
        }
    ]
    result["proof"]["edits"].append(
        {
            "key": key,
            "byteStart": slot["byteStart"],
            "byteEnd": slot["byteEnd"],
            "before": old,
            "after": new,
            "targetPaths": [],
        }
    )
    result["proof"]["edits"].sort(key=lambda e: (e["byteStart"], e["byteEnd"]))
    source = reconstruct_source(candidate)
    raw = apply_edits(source, result["proof"]["edits"]).decode()
    result.update(joinedRawText=raw, joinedRawTextSha256=digest(raw.encode()))
    result["proof"]["renderedSha256"] = digest(raw.encode())
    if reconstruct_source(result) != source or result["target"] != candidate["target"]:
        raise ValueError("tare-only refresh changed source or public facts")
    return result


def main(identity):
    inventory = read(OUT / "receipt-inventory.json")
    entry = next(e for e in inventory["records"] if e["id"] == identity)
    original_path = ROOT / entry["candidate"]
    candidate = read(original_path)
    current = next(
        r
        for r in map(json.loads, verify_pin(inventory["baseline"]).open())
        if r["documentId"] == identity
    )
    if candidate["joinedRawText"] != current["joinedRawText"]:
        raise ValueError("receipt is not the current baseline")
    catalog_pin = read(DATA / "manifest.json")["templateCatalog"]
    catalog = PaddleTemplateCatalog(ROOT, catalog_pin)
    template, _, transfer, _ = catalog._template(candidate["sourceDocumentId"])
    recipes = [
        e["equipmentTare"]
        for e in template["overlay"]["extras"]
        if "aggregate" in e.get("equipmentTare", {})
    ]
    if not recipes or any(
        r["aggregate"] != recipes[0]["aggregate"] or r["support"] != recipes[0]["support"]
        for r in recipes
    ):
        raise ValueError("requires one shared, fully owned tare aggregate")
    aggregate = recipes[0]["aggregate"]
    key = aggregate["surfaceKey"]
    (old,) = candidate["surfaceValues"][key]
    try:
        extra_value(
            {"owner": "tare-feasibility", "equipmentTare": recipes[0]},
            candidate,
            current,
            identity,
            catalog.equipment,
            project_root=ROOT,
        )
    except ValueError as exc:
        if str(exc) != "sampled equipment cannot satisfy the fixed tare aggregate":
            raise
    else:
        raise ValueError("existing aggregate is feasible; no refresh is justified")
    recipe = {k: v for k, v in recipes[0].items() if k not in ("aggregate", "index")}
    recipe["indices"] = aggregate["indices"]
    new, _ = extra_value(
        {"owner": "tare-total-refresh", "equipmentTare": recipe},
        candidate,
        current,
        identity,
        catalog.equipment,
        project_root=ROOT,
    )
    (slot,) = [s for s in transfer["slots"] if s["key"] == key]
    evidence = {
        "catalog": catalog_pin,
        "template": catalog.entries[candidate["sourceDocumentId"]]["template"],
        "indices": aggregate["indices"],
        "previousTotal": old,
        "newTotal": new,
        "reason": (
            "Old source-only aggregate is infeasible for retained mixed equipment; "
            "resample only related tares from reviewed equipment support."
        ),
    }
    result = replace_total(candidate, slot, key, old, new, evidence)
    result["migrationTareRefresh"] = {"previousCandidate": pin(original_path), **evidence}
    folder = ART / "retained-tare-refresh" / identity
    path = folder / "candidate.json"
    atomic_publish_json(path, result)
    replacement = {
        **current,
        "joinedRawText": result["joinedRawText"],
        "joinedRawTextSha256": result["joinedRawTextSha256"],
    }
    atomic_publish_json(folder / "replacement.json", replacement)
    if catalog.wording_specification(path) is not None:
        raise ValueError("new linguistic owners require separate reviewed migration")
    rendered = catalog.render(path, current=replacement)
    atomic_publish_json(folder / "rendered.json", rendered)
    proposal = {
        "status": "TARE_ONLY_REFRESH_REQUIRES_READBACK",
        "id": identity,
        "currentRecordSha256": digest(current),
        "candidate": pin(path),
        "replacementRecord": pin(folder / "replacement.json"),
        "rendered": pin(folder / "rendered.json"),
        "catalog": catalog_pin,
        "changes": evidence,
    }
    atomic_publish_json(folder / "proposal.json", proposal)
    print(json.dumps(proposal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identity")
    main(parser.parse_args().identity)
