"""Complete two manually reviewed inner ledgers without changing public shipment facts."""

import copy
import json
from decimal import Decimal

from migrate_r30_synthetic import ART, DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog


def main():
    identity = "syn_full_v7_129bf304c9c12a202b49a928"
    inventory = read(OUT / "receipt-inventory.json")
    entry = next(e for e in inventory["records"] if e["id"] == identity)
    row = next(r for r in map(json.loads, verify_pin(inventory["baseline"]).open())
               if r["documentId"] == identity)
    prior_path = ART / "identity-receipts" / (identity + ".json")
    candidate = read(prior_path)
    if (candidate["migrationIdentity"]["previousCandidate"] !=
        {"path": entry["candidate"], "sha256": entry["candidateSha256"]}):
        raise ValueError("retained identity receipt changed")
    prior = copy.deepcopy(candidate)
    goods = row["target"]["goodsItemDetails"][0]
    public = goods["numberAndTypeOfPackages"]
    expected = [{"packageQuantity": 11, "typeCategory": "PACKAGE_PALLET"},
                {"packageQuantity": 296, "typeCategory": "PACKAGE_BOX_FIBREBOARD"}]
    if public != expected or goods["description"] != "AGED CAMEL MILK CHEESE":
        raise ValueError("reviewed retained goods or package groups changed")
    provenance = candidate["scenario"]["provenance"]
    base = provenance["freshPrivatePacking"]
    physical = provenance["declaredPackages"]["groups"]
    packing = []
    for index, (cartons, pieces) in enumerate(((27, 6), (2, 2))):
        count = public[index]["packageQuantity"]
        facts = physical[index]
        related = count * cartons
        net, gross = Decimal(facts["netKg"]), Decimal(facts["grossKg"])
        if gross <= net or Decimal(facts["volumeM3"]) <= 0:
            raise ValueError("retained unit facts are not complete")
        packing.append({
            "countedCategory": public[index]["typeCategory"], "publicQuantity": count,
            "relation": "SUPPORTS" if index == 0 else "CONTAINS",
            "relatedCategory": "PACKAGE_CARTON", "relatedQuantity": related,
            "unitNetKg": str(net), "unitGrossKg": str(gross),
            "netKg": str(net * count), "grossKg": str(gross * count),
            "facts": {"externalPackingPresentation": {
                "owner": "native-inner-counts",
                "nativeFingerprint": "22102b0785e789bd73a22568fe02bcc9b66909a56b8a58a6717c8cbe78ee0b8a",
                "levels": 3, "contentsKind": "goods_pieces",
                "relatedQuantity": related, "relatedCategory": "PACKAGE_CARTON",
                "contents": {"packageSingular": "PIECE", "packagePlural": "PIECES",
                             "unitsPerRelated": pieces, "quantity": related * pieces},
                "publicPackageRowsAdded": False, "sourceCountRatioUsed": False,
            }},
            "reviewedRetainedPacking": {
                "cartonsPerCountedUnit": cartons, "netKgPerCarton": str(net / cartons),
                "netKgPerPiece": str(net / cartons / pieces),
                "packingKgPerCountedUnit": str(gross - net),
                "occupiedM3PerCartonEnvelope": str(Decimal(facts["volumeM3"]) / cartons),
                "basis": "Manually reviewed aged-cheese pieces. Pallet row: 27 cartons of "
                         "six approximately 4.51-kg pieces, within 1.5 m3 and 37.098 kg "
                         "existing packing budget. Box row: two inner cartons each holding "
                         "two 4.879-kg pieces, within 0.027 m3 and 1.165 kg packing budget. "
                         "No new mass added and no public or printed existing fact changed.",
            },
        })
    ledger = copy.deepcopy(base)
    ledger.update(relation="DISJOINT_GROUPS", publicQuantity=sum(r["packageQuantity"] for r in public),
                  countedCategory="PACKAGE_PACKAGE",
                  facts={"declaredGroupLedgers": packing})
    provenance["freshPrivatePacking"] = ledger
    candidate["physical"]["freshPrivatePacking"] = copy.deepcopy(ledger)
    candidate["migrationMissingPacking"] = {
        "previousCandidate": pin(prior_path), "currentRecordSha256": digest(row),
        "review": "Main-agent numeric and semantic review; incomplete provider proposal "
                  "was not accepted. Two explicit manual inner ledgers preserve all public facts.",
    }
    directory = ART / "retained-inner-completed" / identity
    destination = directory / "candidate.json"
    atomic_publish_json(destination, candidate)
    catalog = PaddleTemplateCatalog(ROOT, read(DATA / "manifest.json")["templateCatalog"])
    old_spec, new_spec = catalog.wording_specification(prior_path), catalog.wording_specification(destination)
    for field in ("context", "requested", "patterns", "sourceFingerprint"):
        if old_spec[field] != new_spec[field]:
            raise ValueError("missing inner ledgers changed linguistic context")
    wording_path = ART / "wording" / (identity + ".json")
    wording = read(wording_path)
    if wording["status"] != "REVIEWED" or wording["candidateSha256"] != pin(prior_path)["sha256"]:
        raise ValueError("wording receipt is not current")
    wording.update(candidateSha256=pin(destination)["sha256"],
                   unchangedWordingContext={"previous": pin(wording_path)})
    atomic_publish_json(directory / "wording.json", wording)
    for field in ("target", "joinedRawText", "proof"):
        if candidate[field] != prior[field]:
            raise ValueError("inner completion changed an existing public fact or text")
    rendered = catalog.render(destination, current=row, wording_receipt=wording)
    atomic_publish_json(directory / "rendered.json", rendered)
    atomic_publish_json(directory / "proposal.json", {
        "status": "INNER_LEDGER_COMPLETION_REQUIRES_FINAL_READBACK", "id": identity,
        "currentRecordSha256": digest(row), "candidate": pin(destination),
        "wording": pin(directory / "wording.json"), "rendered": pin(directory / "rendered.json"),
    })
    print(json.dumps({"id": identity, "innerLedgers": packing,
                      "target": rendered["target"]["goodsItemDetails"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
