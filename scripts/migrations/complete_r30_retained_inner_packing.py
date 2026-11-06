"""Complete reviewed private carton/piece counts without resampling retained cargo."""

import argparse
import copy
import json
from decimal import Decimal

from migrate_r30_synthetic import ART, DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin
from r30_missing_numeric import amount

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog


def complete_ledger(candidate, public, decision, native, authority):
    if "groups" in decision:
        return complete_disjoint_ledger(candidate, public, decision, native, authority)
    if len(public) != 1 or public[0] != decision["expectedPackage"]:
        raise ValueError("completion requires the exact reviewed single public group")
    counts = decision["cartonsPerUnit"], decision["piecesPerCarton"]
    if any(type(x) is not int or x <= 0 for x in counts):
        raise ValueError("reviewed inner counts must be positive integers")
    if decision["relation"] not in {"CONTAINS", "SUPPORTS"} or not decision["basis"].strip():
        raise ValueError("reviewed packing relation and basis required")
    quantity = public[0]["packageQuantity"]
    physical = candidate["physical"]
    gross_fact, volume_fact = physical["measures"]["grossWeight"], physical["measures"]["volume"]
    gross = amount(gross_fact["total"], gross_fact["unit"], "kilogram")
    volume = amount(volume_fact["total"], volume_fact["unit"], "cubic_metre")
    if gross <= 0 or volume <= 0 or quantity <= 0:
        raise ValueError("retained mass/volume/count must be positive")
    prior = candidate["scenario"]["provenance"].get("freshPrivatePacking")
    if prior is not None and (
        prior["relation"] != "NONE"
        or prior["facts"]
        or prior["publicQuantity"] != quantity
        or prior["countedCategory"] != public[0]["typeCategory"]
        or Decimal(prior["grossKg"]) != gross
    ):
        raise ValueError("completion would overwrite an existing relationship or physical fact")
    cartons, pieces = counts
    related = quantity * cartons
    ledger = {
        "sourceDocumentId": candidate["sourceDocumentId"],
        "countedCategory": public[0]["typeCategory"],
        "publicQuantity": quantity,
        "relation": decision["relation"],
        "relatedCategory": "PACKAGE_CARTON",
        "relatedQuantity": related,
        "grossKg": str(gross),
        "unitGrossKg": str(gross / quantity),
        "facts": {
            "externalPackingPresentation": {
                "owner": native["owner"],
                "nativeFingerprint": native["nestedPacking"]["nativeFingerprint"],
                "levels": 3,
                "contentsKind": "goods_pieces",
                "relatedQuantity": related,
                "relatedCategory": "PACKAGE_CARTON",
                "contents": {
                    "packageSingular": "PIECE",
                    "packagePlural": "PIECES",
                    "unitsPerRelated": pieces,
                    "quantity": related * pieces,
                },
                "publicPackageRowsAdded": False,
                "sourceCountRatioUsed": False,
            }
        },
        "reviewedRetainedPacking": {
            "authority": authority,
            "basis": decision["basis"],
            "cartonsPerCountedUnit": cartons,
            "averageGrossKgPerCarton": str(gross / related),
            "averageGrossKgPerPiece": str(gross / related / pieces),
            "occupiedM3PerCartonEnvelope": str(volume / related),
            "priorLedger": prior,
        },
    }
    if prior is not None:
        net = Decimal(prior["netKg"])
        if not 0 < net <= gross or Decimal(prior["unitNetKg"]) * quantity != net:
            raise ValueError("retained net mass is not coherent")
        ledger.update(netKg=str(net), unitNetKg=prior["unitNetKg"])
    return ledger


def complete_disjoint_ledger(candidate, public, decision, native, authority):
    """Use retained per-counted-unit facts for each ordered public package group."""
    provenance = candidate["scenario"]["provenance"]
    declared, prior = provenance["declaredPackages"], provenance["freshPrivatePacking"]
    choices = decision["groups"]
    if (
        len(public) < 2
        or declared["rows"] != public
        or len(choices) != len(public)
        or len(declared["groups"]) != len(public)
        or prior["relation"] != "NONE"
        or prior["facts"]
        or prior["publicQuantity"] != sum(r["packageQuantity"] for r in public)
    ):
        raise ValueError("completion needs exact ordered groups with no existing hierarchy")
    ledgers, sums = [], {"netKg": Decimal(0), "grossKg": Decimal(0), "volumeM3": Decimal(0)}
    for row, facts, choice in zip(public, declared["groups"], choices, strict=True):
        if facts["category"] != row["typeCategory"] or "groups" in choice:
            raise ValueError("group category or reviewed nesting scope differs")
        count = row["packageQuantity"]
        unit = {key: Decimal(facts[key]) for key in sums}
        if (
            any(not value.is_finite() or value <= 0 for value in unit.values())
            or unit["netKg"] >= unit["grossKg"]
        ):
            raise ValueError("group needs positive finite per-unit facts and packing headroom")
        total = {key: value * count for key, value in unit.items()}
        group_candidate = {
            "sourceDocumentId": candidate["sourceDocumentId"],
            "scenario": {"provenance": {}},
            "physical": {
                "measures": {
                    "grossWeight": {"total": str(total["grossKg"]), "unit": "kilogram"},
                    "volume": {"total": str(total["volumeM3"]), "unit": "cubic_metre"},
                }
            },
        }
        ledger = complete_ledger(group_candidate, [row], choice, native, authority)
        ledger.update(netKg=str(total["netKg"]), unitNetKg=str(unit["netKg"]))
        ledger["reviewedRetainedPacking"].update(
            netKgPerCarton=str(unit["netKg"] / choice["cartonsPerUnit"]),
            packingKgPerCountedUnit=str(unit["grossKg"] - unit["netKg"]),
        )
        ledgers.append(ledger)
        for key in sums:
            sums[key] += total[key]
    if any(sums[key] != Decimal(declared["totals"][key]) for key in sums) or (
        sums["grossKg"] != amount(candidate["physical"]["measures"]["grossWeight"]["total"], candidate["physical"]["measures"]["grossWeight"]["unit"], "kilogram")
        or sums["volumeM3"] != amount(candidate["physical"]["measures"]["volume"]["total"], candidate["physical"]["measures"]["volume"]["unit"], "cubic_metre")
        or sums["netKg"] != Decimal(prior["netKg"])
        or sums["grossKg"] != Decimal(prior["grossKg"])
    ):
        raise ValueError("group facts do not conserve retained shipment totals")
    result = copy.deepcopy(prior)
    result.update(relation="DISJOINT_GROUPS", facts={"declaredGroupLedgers": ledgers})
    return result


def main(review_path):
    review = read(review_path)
    if review["status"] != "MAIN_AGENT_REVIEWED_MISSING_INNER_COUNTS":
        raise ValueError("explicit missing-data review required")
    inventory = read(OUT / "receipt-inventory.json")
    entries = {e["id"]: e for e in inventory["records"]}
    rows = {
        r["documentId"]: r
        for r in map(json.loads, verify_pin(inventory["baseline"]).open())
        if r["documentId"] in review["decisions"]
    }
    if set(rows) != set(review["decisions"]):
        raise ValueError("review contains unknown current identities")
    catalog = PaddleTemplateCatalog(ROOT, read(DATA / "manifest.json")["templateCatalog"])
    for identity, decision in review["decisions"].items():
        proposal_path = ART / "retained-inner" / (identity + ".json")
        proposal = read(proposal_path)
        original_path = verify_pin(proposal["candidate"])
        original = read(original_path)
        entry, row = entries[identity], rows[identity]
        if (
            proposal["candidate"]
            != {"path": entry["candidate"], "sha256": entry["candidateSha256"]}
            or original["joinedRawText"] != row["joinedRawText"]
            or original["sourceDocumentId"] != row["sourceDocumentId"]
        ):
            raise ValueError("proposal does not describe the exact retained receipt")
        if original["documentId"] != identity:
            original_path = ART / "identity-receipts" / (identity + ".json")
            normalized = read(original_path)
            expected = {
                **original,
                "documentId": identity,
                "migrationIdentity": {
                    "previousCandidate": proposal["candidate"],
                    "currentRecordSha256": digest(row),
                },
            }
            if normalized != expected:
                raise ValueError("identity receipt changes more than the retained identity")
            original = normalized
        candidate = copy.deepcopy(original)
        template, _, _, _ = catalog.shipment(original_path)
        native = [e for e in template["overlay"]["extras"] if "nestedPacking" in e]
        if len(native) != 1:
            raise ValueError("missing unique certified native hierarchy owner")
        public = row["target"]["goodsItemDetails"][0]["numberAndTypeOfPackages"]
        ledger = complete_ledger(candidate, public, decision, native[0], pin(review_path))
        candidate["scenario"]["provenance"]["freshPrivatePacking"] = ledger
        candidate["physical"]["freshPrivatePacking"] = copy.deepcopy(ledger)
        candidate["migrationMissingPacking"] = {
            "previousCandidate": pin(original_path),
            "currentRecordSha256": digest(row),
            "review": pin(review_path),
            "proposal": pin(proposal_path),
        }
        directory = ART / "retained-inner-completed" / identity
        destination = directory / "candidate.json"
        atomic_publish_json(destination, candidate)
        old_spec = catalog.wording_specification(original_path)
        new_spec = catalog.wording_specification(destination)
        for field in ("context", "requested", "patterns", "sourceFingerprint"):
            if old_spec[field] != new_spec[field]:
                raise ValueError("inner completion changed linguistic context")
        wording_path = ART / "wording" / (identity + ".json")
        wording = read(wording_path)
        if (
            wording["status"] != "REVIEWED"
            or wording["candidateSha256"] != pin(original_path)["sha256"]
        ):
            raise ValueError("existing wording is not reviewed/current")
        wording.update(
            candidateSha256=pin(destination)["sha256"],
            unchangedWordingContext={"previous": pin(wording_path)},
        )
        atomic_publish_json(directory / "wording.json", wording)
        rendered = catalog.render(destination, current=row, wording_receipt=wording)
        atomic_publish_json(directory / "rendered.json", rendered)
        atomic_publish_json(
            directory / "proposal.json",
            {
                "status": "INNER_LEDGER_COMPLETION_REQUIRES_FINAL_READBACK",
                "id": identity,
                "currentRecordSha256": digest(row),
                "candidate": pin(destination),
                "wording": pin(directory / "wording.json"),
                "rendered": pin(directory / "rendered.json"),
                "review": pin(review_path),
            },
        )
        for field in ("target", "joinedRawText", "proof"):
            if candidate[field] != original[field]:
                raise ValueError("completion changed retained target/text/edits")
        groups = ledger["facts"].get("declaredGroupLedgers", [ledger])
        print(
            json.dumps(
                {
                    "id": identity,
                    "innerCartons": sum(g["relatedQuantity"] for g in groups),
                    "pieces": sum(
                        g["facts"]["externalPackingPresentation"]["contents"]["quantity"]
                        for g in groups
                    ),
                }
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review")
    args = parser.parse_args()
    main(ROOT / args.review)
