"""Clarify reviewed additive unit prefixes and group-owned mass summaries."""

import argparse
import copy
import json

from migrate_r30_synthetic import ART, DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.declared_package_private import join_unit_net_prefixes
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog
from document_ocr.synthesis.scoped_cargo_refresh import apply_edits, reconstruct_source


def repair(candidate, certificate, labels):
    result = copy.deepcopy(candidate)
    ledger = result["physical"]["freshPrivatePacking"]
    if ledger != result["scenario"]["provenance"]["freshPrivatePacking"]:
        raise ValueError("private ledger authorities differ")
    children = ledger["facts"]["declaredGroupLedgers"]
    if ledger["relation"] != "DISJOINT_GROUPS" or len(children) < 2 or len(labels) != len(children):
        raise ValueError("requires reviewed labels for every disjoint group")
    for label, child in zip(labels, children, strict=True):
        if not label.startswith(str(child["publicQuantity"]) + " "):
            raise ValueError("reviewed group label does not retain its count")
    changes = {}
    for index, region in enumerate(certificate["regions"]):
        role, key = region["role"], f"held-numeric:{index}"
        if role not in {"unit_net_prefix", "support_mass_breakdown", "component_mass_pair"}:
            continue
        active = [
            (i, child["surfaces"][key])
            for i, child in enumerate(children)
            if key in child["surfaces"]
        ]
        if len(active) < 2:
            continue
        if any(len(values) != 1 for _, values in active):
            raise ValueError("private owner has multiple unreviewed surfaces")
        terms = [values[0] for _, values in active]
        old = "\n".join(terms)
        if ledger["surfaces"][key] != [old]:
            raise ValueError("combined old surface is not the exact child projection")
        new = (
            join_unit_net_prefixes(terms)
            if role == "unit_net_prefix"
            else "\n".join(f"{labels[i]}: {values[0]}" for i, values in active)
        )
        edits = [e for e in result["proof"]["edits"] if e["key"] == key]
        allowed = (
            "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[",
            "documentPatch.goodsItemDetails[0].splitGoodsPlacement[",
            "documentPatch.goodsItemDetails[0].grossWeight.",
            "documentPatch.goodsItemDetails[0].netWeight.",
        )
        if not edits or any(
            e["after"] != old or any(not path.startswith(allowed) for path in e["targetPaths"])
            for e in edits
        ):
            raise ValueError("numeric surface has a different text or nonnumeric target owner")
        for edit in edits:
            edit["after"] = new
        for field in ("renderValues", "surfaceValues"):
            if key in result[field]:
                if result[field][key] not in (old, [old]):
                    raise ValueError("numeric value disagrees with its receipt")
                result[field][key] = [new] if isinstance(result[field][key], list) else new
        ledger["surfaces"][key] = [new]
        changes[key] = {"role": role, "before": old, "after": new}
    if not changes:
        raise ValueError("no reviewed disjoint surface requires correction")
    result["scenario"]["provenance"]["freshPrivatePacking"] = copy.deepcopy(ledger)
    source = reconstruct_source(candidate)
    raw = apply_edits(source, result["proof"]["edits"]).decode()
    result.update(joinedRawText=raw, joinedRawTextSha256=digest(raw.encode()))
    result["proof"]["renderedSha256"] = digest(raw.encode())
    if reconstruct_source(result) != source or result["target"] != candidate["target"]:
        raise ValueError("surface correction changed source or retained public facts")
    return result, changes


def main(review_path):
    review = read(review_path)
    if review["status"] != "MAIN_AGENT_REVIEWED_DISJOINT_NUMERIC_SURFACES":
        raise ValueError("explicit numeric-scope review required")
    identity = review["documentId"]
    inventory = read(OUT / "receipt-inventory.json")
    entry = next(e for e in inventory["records"] if e["id"] == identity)
    if review["candidate"] != {"path": entry["candidate"], "sha256": entry["candidateSha256"]}:
        raise ValueError("review belongs to another retained receipt")
    original = read(verify_pin(review["candidate"]))
    row = next(
        r
        for r in map(json.loads, verify_pin(inventory["baseline"]).open())
        if r["documentId"] == identity
    )
    if original["joinedRawText"] != row["joinedRawText"]:
        raise ValueError("retained current text differs")
    certificate = read(verify_pin(review["numericCertificates"]))["sources"][entry["source"]]
    if certificate["sourceSha256"] != original["proof"]["sourceSha256"]:
        raise ValueError("numeric certificate belongs to another source revision")
    candidate, changes = repair(original, certificate, review["groupLabels"])
    candidate["documentId"] = identity
    candidate["migrationNumericScopeRepair"] = {"review": pin(review_path), "changes": changes}
    replacement = copy.deepcopy(row)
    replacement.update(
        joinedRawText=candidate["joinedRawText"],
        joinedRawTextSha256=candidate["joinedRawTextSha256"],
    )
    folder = ART / "disjoint-numeric-repairs" / identity / digest(review)
    path = folder / "candidate.json"
    atomic_publish_json(path, candidate)
    atomic_publish_json(folder / "replacement.json", replacement)
    catalog = PaddleTemplateCatalog(ROOT, read(DATA / "manifest.json")["templateCatalog"])
    spec = catalog.wording_specification(path)
    wording = None
    if spec is not None:
        prior_path = verify_pin(review["candidate"])
        prior_spec = catalog.wording_specification(prior_path)
        if any(
            spec[key] != prior_spec[key]
            for key in ("context", "requested", "patterns", "sourceFingerprint")
        ):
            raise ValueError("changed context needs separate wording review")
        wording_path = verify_pin(review["wording"])
        wording = read(wording_path)
        if wording["candidateSha256"] != pin(prior_path)["sha256"]:
            renamed = ART / "identity-receipts" / (identity + ".json")
            normalized = read(renamed)
            if normalized != {
                **original,
                "documentId": identity,
                "migrationIdentity": {
                    "previousCandidate": pin(prior_path),
                    "currentRecordSha256": digest(row),
                },
            }:
                raise ValueError("wording identity normalization changed cargo")
            prior_path = renamed
        if (
            wording["status"] != "REVIEWED"
            or wording["candidateSha256"] != pin(prior_path)["sha256"]
        ):
            raise ValueError("wording is not reviewed for the exact retained context")
        wording.update(
            candidateSha256=pin(path)["sha256"],
            unchangedContextReview={
                "previousWording": pin(wording_path),
                "review": pin(review_path),
            },
        )
        atomic_publish_json(folder / "wording.json", wording)
    rendered = catalog.render(path, current=replacement, wording_receipt=wording)
    atomic_publish_json(folder / "rendered.json", rendered)
    atomic_publish_json(
        folder / "proposal.json",
        {
            "status": "NUMERIC_SCOPE_REPAIR_REQUIRES_FINAL_READBACK",
            "id": identity,
            "currentRecordSha256": digest(row),
            "candidate": pin(path),
            "replacementRecord": pin(folder / "replacement.json"),
            "rendered": pin(folder / "rendered.json"),
            "review": pin(review_path),
            "changes": changes,
            "wording": pin(folder / "wording.json") if wording is not None else None,
        },
    )
    print(folder / "proposal.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review")
    main(ROOT / parser.parse_args().review)
