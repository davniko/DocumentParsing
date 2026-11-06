"""Attach reviewed, mass-conserving support facts without changing frozen cargo."""

import argparse
import copy
import json
from decimal import Decimal

from migrate_r30_synthetic import ART, DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog


def support_mass(quantity, count, unit_net, unit_gross, tare):
    """Keep support mass inside existing packing; no unprinted equal-fill constraint."""
    if type(count) is not int or type(quantity) is not int or not 0 < count <= quantity:
        raise ValueError("support count must cover positive counted packages")
    if not all(v.is_finite() for v in (unit_net, unit_gross, tare)):
        raise ValueError("support mass values must be finite")
    budget = quantity * (unit_gross - unit_net)
    if not 0 < tare * count < budget:
        raise ValueError("support mass must fit the existing complete packing budget")
    return budget


def main(proposal_path, review_path):
    evidence = read(proposal_path)
    proposal = evidence["proposal"]
    review = read(review_path)
    if (
        review["status"] != "MAIN_AGENT_REVIEWED_SUPPORT_PROPOSAL"
        or review["proposal"] != pin(proposal_path)
        or review["values"] != proposal
        or proposal["unknown"]
        or not review["reason"]
    ):
        raise ValueError("proposal differs from main-agent reviewed values")
    original = read(verify_pin(evidence["candidate"]))
    candidate = copy.deepcopy(original)
    identity, source = candidate["documentId"], candidate["sourceDocumentId"]
    catalog_pin = read(DATA / "manifest.json")["templateCatalog"]
    catalog = PaddleTemplateCatalog(ROOT, catalog_pin)
    template, _, _, _ = catalog._template(source)
    from document_ocr.synthesis.independent_numeric import IndependentNumericCertificates

    certificate = IndependentNumericCertificates.from_path(
        verify_pin(template["overlay"]["numericCertificate"]),
        expected_sha256=template["overlay"]["numericCertificate"]["sha256"],
    ).certificate(source)
    presentation = certificate["externalPackingPresentation"]
    if presentation["levels"] != 2 or presentation["contentsKind"] != "support_packages":
        raise ValueError("reviewed support proposal belongs to a different topology")
    provenance = candidate["scenario"]["provenance"]
    prior = provenance["freshPrivatePacking"]
    if prior["relation"] != "NONE":
        raise ValueError("support migration must not overwrite existing nesting")
    counts = proposal["palletCounts"]
    tare = Decimal(str(proposal["palletTareKg"]))
    unit_net, unit_gross = Decimal(prior["unitNetKg"]), Decimal(prior["unitGrossKg"])
    groups = []
    public_rows = candidate["target"]["documentPatch"]["goodsItemDetails"][0][
        "numberAndTypeOfPackages"
    ]
    if len(counts) != len(public_rows):
        raise ValueError("support proposal must cover each declared group exactly once")
    for row, count in zip(public_rows, counts, strict=True):
        quantity = row["packageQuantity"]
        if (
            row["typeCategory"] != prior["countedCategory"]
            or type(count) is not int
            or not 0 < count <= quantity
        ):
            raise ValueError("support proposal does not cover each declared group")
        budget = support_mass(quantity, count, unit_net, unit_gross, tare)
        group = {
            "countedCategory": row["typeCategory"],
            "publicQuantity": quantity,
            "relation": "ON_SUPPORT",
            "relatedCategory": "PACKAGE_PALLET",
            "relatedQuantity": count,
            "unitNetKg": str(unit_net),
            "unitGrossKg": str(unit_gross),
            "netKg": str(unit_net * quantity),
            "grossKg": str(unit_gross * quantity),
            "facts": {
                "supportEnvelope": {
                    "category": "PACKAGE_PALLET",
                    "quantity": count,
                    "unitTareKg": str(tare),
                    "totalTareKg": str(tare * count),
                    "packingBudgetKg": str(budget),
                    "authority": {
                        "reviewedProposal": pin(proposal_path),
                        "existingGrossIncludesSupports": True,
                    },
                },
                "externalPackingPresentation": {
                    **presentation,
                    "relatedQuantity": count,
                    "relatedCategory": "PACKAGE_PALLET",
                    "publicPackageRowsAdded": False,
                    "sourceCountRatioUsed": False,
                },
            },
        }
        groups.append(group)
    if sum(g["publicQuantity"] for g in groups) != prior["publicQuantity"]:
        raise ValueError("declared groups do not conserve the frozen shipment")
    ledger = copy.deepcopy(prior)
    ledger.update(
        relation="DISJOINT_GROUPS",
        relatedCategory="PACKAGE_PALLET",
        relatedQuantity=sum(g["relatedQuantity"] for g in groups),
    )
    ledger["facts"]["declaredGroupLedgers"] = groups
    ledger["certificateSha256"] = digest(certificate)
    ledger["certificateFileSha256"] = template["overlay"]["numericCertificate"]["sha256"]
    provenance["freshPrivatePacking"] = ledger
    candidate["migrationAdditionalPacking"] = {
        "previousCandidate": evidence["candidate"],
        "reviewedProposal": pin(proposal_path),
        "review": pin(review_path),
        "scope": "New Paddle-only support count. Reviewed support counts per declared group; tare "
        "allocated within frozen packing mass. All old source text, public cargo facts, numeric "
        "repetitions and equipment plans unchanged.",
    }
    if {
        k: v for k, v in candidate.items() if k not in {"scenario", "migrationAdditionalPacking"}
    } != {k: v for k, v in original.items() if k != "scenario"}:
        raise ValueError("targeted packing migration changed unrelated shipment content")
    output = proposal_path.parent / "candidate.json"
    atomic_publish_json(output, candidate)
    before_spec = catalog.wording_specification(verify_pin(evidence["candidate"]))
    after_spec = catalog.wording_specification(output)
    if {k: v for k, v in before_spec.items() if k != "candidateSha256"} != {
        k: v for k, v in after_spec.items() if k != "candidateSha256"
    }:
        raise ValueError("new support authority changed the linguistic context")
    wording_path = ART / "wording" / (identity + ".json")
    wording = read(wording_path)
    if (
        wording["candidateSha256"] != evidence["candidate"]["sha256"]
        or wording["status"] != "REVIEWED"
    ):
        raise ValueError("cannot transfer unreviewed wording")
    wording["candidateSha256"] = pin(output)["sha256"]
    wording["packingAuthorityReconciliation"] = {
        "previousWording": pin(wording_path),
        "sameWordingSpecification": True,
    }
    new_wording = proposal_path.parent / "wording.json"
    atomic_publish_json(new_wording, wording)
    inventory = read(OUT / "receipt-inventory.json")
    entry = next(r for r in inventory["records"] if r["id"] == identity)
    atomic_publish_json(
        ART / "reconciled" / (identity + ".json"),
        {
            "status": "REVIEWED_CURRENT_SHIPMENT_RECEIPT",
            "currentRecordSha256": entry["recordSha256"],
            "candidate": pin(output),
            "wording": pin(new_wording),
            "reviewedPacking": pin(proposal_path),
        },
    )
    print(
        json.dumps(
            {
                "id": identity,
                "supportPallets": ledger["relatedQuantity"],
                "existingPackingKg": str((unit_gross - unit_net) * prior["publicQuantity"]),
                "supportKg": str(tare * ledger["relatedQuantity"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal")
    parser.add_argument("review")
    args = parser.parse_args()
    main(ROOT / args.proposal, ROOT / args.review)
