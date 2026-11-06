"""Stage explicit private-support clarification without changing shipment facts."""

import argparse
import copy
import json

from migrate_r30_synthetic import ART, DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.declared_package_private import disjoint_support_wording
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog
from document_ocr.synthesis.scoped_cargo_refresh import apply_edits, reconstruct_source


def clarify(candidate):
    result = copy.deepcopy(candidate)
    ledger = result["physical"]["freshPrivatePacking"]
    if ledger != result["scenario"]["provenance"]["freshPrivatePacking"]:
        raise ValueError("private ledger authorities disagree")
    children = ledger["facts"]["declaredGroupLedgers"]
    if ledger["relation"] != "DISJOINT_GROUPS" or len(children) < 2:
        raise ValueError("requires disjoint declared groups")
    changes = {}
    for key, old in ledger["wordingAppends"].items():
        prior = [
            child["wordingAppends"][key] for child in children if key in child["wordingAppends"]
        ]
        if " ".join(prior) != old:
            raise ValueError("combined clause is not the exact ordered child projection")
        new = " ".join(
            disjoint_support_wording(child["wordingAppends"][key], child, len(children))
            for child in children
            if key in child["wordingAppends"]
        )
        if old == new:
            continue
        edits = [e for e in result["proof"]["edits"] if e["key"] == key]
        if not edits or any(e["after"].count(old) != 1 for e in edits):
            raise ValueError("private clause lacks exact unique ownership in each repetition")
        value = result["renderValues"][key]
        if not isinstance(value, str) or value.count(old) != 1:
            raise ValueError("wording value lacks the exact old clause")
        for edit in edits:
            edit["after"] = edit["after"].replace(old, new)
        result["renderValues"][key] = value.replace(old, new)
        ledger["wordingAppends"][key] = new
        changes[key] = {"before": old, "after": new}
    if not changes:
        raise ValueError("no ambiguous private support clause")
    goods = result["target"]["documentPatch"]["goodsItemDetails"]
    if len(goods) != 1:
        raise ValueError("requires a single description owner")
    for change in changes.values():
        old, new = (" ".join(change[k].split()).upper() for k in ("before", "after"))
        if goods[0]["description"].count(old) != 1:
            raise ValueError("description does not retain the exact private clause")
        goods[0]["description"] = goods[0]["description"].replace(old, new)
    result["scenario"]["provenance"]["freshPrivatePacking"] = copy.deepcopy(ledger)
    source = reconstruct_source(candidate)
    raw = apply_edits(source, result["proof"]["edits"]).decode()
    result.update(joinedRawText=raw, joinedRawTextSha256=digest(raw.encode()))
    result["proof"]["renderedSha256"] = digest(raw.encode())
    if reconstruct_source(result) != source:
        raise ValueError("source reconstruction changed")
    result["numericWordingProjection"] = {
        "previous": candidate["numericWordingProjection"],
        "ownedClauses": copy.deepcopy(ledger["wordingAppends"]),
        "clarifiedSupportScope": changes,
        "publicAccountingAndPhysicalFactsUnchanged": True,
    }
    return result, changes


def main(identity):
    inventory = read(OUT / "receipt-inventory.json")
    entry = next(e for e in inventory["records"] if e["id"] == identity)
    current = next(
        r
        for r in map(json.loads, verify_pin(inventory["baseline"]).open())
        if r["documentId"] == identity
    )
    original_path = ROOT / entry["candidate"]
    original = read(original_path)
    if original["joinedRawText"] != current["joinedRawText"]:
        raise ValueError("retained baseline differs")
    candidate, changes = clarify(original)
    candidate["documentId"] = identity
    candidate["migrationSupportClarification"] = {
        "previousCandidate": pin(original_path),
        "changes": changes,
        "reason": "Independent support units must not be mistaken for a disjoint declared group.",
    }
    replacement = copy.deepcopy(current)
    replacement.update(
        joinedRawText=candidate["joinedRawText"],
        joinedRawTextSha256=candidate["joinedRawTextSha256"],
        target=candidate["target"]["documentPatch"],
    )
    folder = ART / "disjoint-support-clarification" / identity
    destination = folder / "candidate.json"
    atomic_publish_json(destination, candidate)
    atomic_publish_json(folder / "replacement.json", replacement)
    catalog_pin = read(DATA / "manifest.json")["templateCatalog"]
    catalog = PaddleTemplateCatalog(ROOT, catalog_pin)
    specification = catalog.wording_specification(destination)
    if specification is not None and specification["requested"]:
        raise ValueError("this repair requires separately reviewed new linguistic owners")
    rendered = catalog.render(destination, current=replacement)
    atomic_publish_json(folder / "rendered.json", rendered)
    proposal = {
        "status": "SUPPORT_CLARIFICATION_REQUIRES_READBACK",
        "id": identity,
        "currentRecordSha256": digest(current),
        "candidate": pin(destination),
        "replacementRecord": pin(folder / "replacement.json"),
        "rendered": pin(folder / "rendered.json"),
        "catalog": catalog_pin,
        "changes": changes,
    }
    atomic_publish_json(folder / "proposal.json", proposal)
    print(json.dumps(proposal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("identity")
    main(parser.parse_args().identity)
