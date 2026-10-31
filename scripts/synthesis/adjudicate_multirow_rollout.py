"""Serialize the main review of this rollout; never issue production acceptance.

The reasons below are manual decisions from text/target/ledger readback, not
automatic rules for approving future samples. Every decision pins this inventory's
exact candidate, geometry and native review. Unreviewed finding families fail.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.declared_packages import DeclaredPackages

DECISIONS = {
    "00388da5": "WOOL is a valid selected branch of the classification context WOOL OR FINE "
    "ANIMAL HAIR. A shipment need not repeat every alternative in an HS description.",
    "0d495914": "Reviewed against the settled detached-goods-contact policy: the CN>/NP> "
    "strips do not authorize promotion of the detached consignee contact. Preserve the "
    "certified target scope. M.Y.S. versus M.Y.S is the punctuation actually rendered in "
    "the separate address occurrences, not an inferred different country.",
    "23d02103": "TAX ID is a separately owned, generated source-only identifier. It is "
    "not included in the party name or postal-address target; no field is missing.",
    "276b6f10": "The review incorrectly equates container-load rows with package-group rows. "
    "The two-group matrix is [[1,1],[1,0]]: the second container's 26,316.986 kg is the "
    "bulk group, not a jumbo bag. The three-group load mixes coils, crates and steel "
    "drums; dividing its whole mass by the scalar count does not yield drum mass. "
    "Exact group multiplication, membership, mass, volume and capacities validate.",
    "2ae5fe6d": "The source itself repeats the delivery-agent identity before the postal "
    "block. Both generated repetitions belong to the same owner. The target address "
    "contains only postal components. This is retained source repetition, not an LLM "
    "company name injected into an address field.",
    "3f4147c4": "CNPJ is a retained source-only caption. The approved identifier policy "
    "requires deterministic format preservation, not country-specific identifier "
    "validity. Repeated exporter identifiers agree and do not contaminate party fields.",
    "4d278a8b": "The identity certificate explicitly permits organization_with_proprietor; "
    "the parenthesized person is therefore supported. Private reference/tax captions "
    "are outside goods targets. Disjoint package rows total 29 or 161 as appropriate; "
    "ONE CONTAINER counts equipment, not packages. The live-tuna proposal specifies "
    "aerated-water/insulated counted units and prints no conflicting thermal setting. "
    "This is a coherent synthetic presentation, not commercial loading approval.",
    "719e3e96": "The native finding has empty problem, evidence and correction; it is "
    "not an allegation of a defect. Text/target and deterministic replay passed.",
    "9a55d93a": "The native text explicitly reports no inconsistency. The sampled "
    "commodity and certified nested packing agree; inner cartons remain nonadditive.",
    "a286601b": "Only five per-container tare cells are printed as 3700. The sixth "
    "private tare is 3770, giving the printed aggregate 22270. The reviewer inferred "
    "an unprinted sixth 3700. Preserve the actual source visibility mask.",
    "a5265d59": "C.A. is an eligible pinned dotted ISO2 presentation of Canada. The "
    "country target remains CANADA; the address preserves the printed presentation.",
    "a915a411": "DG prose is generated commercial wording, not forced verbatim registry "
    "proper-shipping-name text. Printed UN/class/PG remain authoritative. The source "
    "has no flash-point slot; omission of a registry qualifier does not authorize "
    "inventing a numerical flash-point target.",
    "b0c96c12": "The 4700 kg tare is owned by the equipment planning authority and "
    "reconciles with the rendered calculations. A reviewer's typical tare range is "
    "not a mandatory limit or evidence that the supplied authority is absent.",
    "b2617901": "The source has eight containers but only seven visible rider gross "
    "weights. The omitted eighth weight remains private. Shipment totals use all "
    "eight loads; synthesis must not invent an eighth printed occurrence.",
    "b76bdefe": "CEILING is a valid selected alternative within CEILING OR WALL fittings. "
    "B.M.U is dotted ISO3 Bermuda, while BM is ISO2. Both presentations are eligible; "
    "the U is not a spurious letter and both targets identify the same country.",
    "bc52d98a": "The inspected campaign explicitly enables en/it/fr/es language "
    "presentations. EQUATEUR with its accent is the French Ecuador name; E.C.U. is "
    "dotted ISO3. These eligible rendered aliases retain the country target ECUADOR.",
    "cd979d32": "The authoritative form says OTHER THAN LIVE, FRESH OR CHILLED, not "
    "fresh/chilled. Cooked pasteurized ambient crab does not contradict that "
    "statement. Exact tariff classification is contextual, not the extraction goal.",
    "d274be39": "Only one VIN occurrence is owned/printed by the source. Multiple "
    "counted articles do not authorize adding unprinted identifiers. Counts and "
    "scalar allocation reconcile; the target does not claim that one VIN identifies "
    "every article.",
    "ecdbc2db": "Container membership is [[50,50],[40,50]] in the two-group example. "
    "Group nets 17100 and 20000 differ legitimately from container nets 19500 and "
    "17600, both totaling 37100. Separate same-category groups are allowed. "
    "Per-unit numeric captions/producer blocks are distinct owned source surfaces, "
    "not an instruction to extend the approved description boundary. Detached "
    "goods-area tax/phone footnotes retain their settled non-target scope.",
}

HOLDS = {
    "e18486c6": "PDF prints KOS on both GW and tare, not an OCR-only spelling. Private "
    "unit interpretation needs the user's decision before replacing the fixed "
    "7737/2030 values with independently sampled facts. Public weight labels remain "
    "absent. Structural passes do not certify this numerical surface.",
    "2763b891": "The one-digit Quantity cell is right-aligned immediately beside goods. "
    "A two/three-group list fails measured horizontal fit at its source font. "
    "Await the user's choice of an explicit scalar-total/list presentation rebase "
    "or single-row-only admission; do not invent space or shrink the font.",
}
SINGLE_ROW = {"bef02d48", "458d2fa0", "0dbc75a0", "2763b891"}


def selected_controls(source, prefix, holds):
    if prefix in SINGLE_ROW:
        controls = source["controls"][-3:]
        if len(controls) != 3 or any(c["groupCount"] != 1 for c in controls):
            raise ValueError("single-row presentation requires three distinct controls")
        return controls
    latest = {c["groupCount"]: c for c in source["controls"]}
    if prefix not in holds and set(latest) != {1, 2, 3}:
        raise ValueError("source is missing a required control")
    return [latest[n] for n in sorted(latest)]


def build(root, inventory_path, resolved):
    holds = {} if resolved else HOLDS
    inventory = json.loads(inventory_path.read_text())
    contracts = {}
    for path in sorted(root.glob("batch-*/source-contracts.json")):
        contracts.update(json.loads(path.read_text())["sources"])
    records = {}
    batches = Counter()
    finding_count = 0
    for sid, source in inventory["sources"].items():
        prefix = sid[4:12]
        controls = []
        for row in selected_controls(source, prefix, holds):
            paths = {name: Path(row[name]) for name in ("candidate", "positions", "review")}
            candidate, positions, review = (json.loads(p.read_text()) for p in paths.values())
            BillOfLadingExtractionV7Label.model_validate_json(json.dumps(candidate["target"]))
            if (
                sha256_file(paths["candidate"]) != row["candidateSha256"]
                or positions["candidateSha256"] != row["candidateSha256"]
                or positions["target"] != candidate["target"]
                or review["candidateHashes"][candidate["documentId"]] != digest(candidate)
            ):
                raise ValueError("candidate/review/geometry changed since main review")
            provenance = candidate["scenario"]["provenance"]
            receipt = provenance.get("declaredPackages")
            if receipt:
                ledger = DeclaredPackages.from_receipt(receipt)
                if provenance["declaredPackageContract"] != contracts[sid]:
                    raise ValueError("control used a different reviewed source contract")
                goods = candidate["target"]["documentPatch"]["goodsItemDetails"][0]
                if (
                    sum(p["packageQuantity"] for p in goods["numberAndTypeOfPackages"])
                    != ledger.total
                ):
                    raise ValueError("public declared counts do not conserve the private ledger")
            findings = [
                f for f in review["review"]["findings"] if f["sample_id"] == candidate["documentId"]
            ]
            if resolved and prefix in HOLDS:
                expected = "batch-39" if prefix == "e18486c6" else "batch-38"
                if row["batch"] != expected or findings:
                    raise ValueError("source resolution requires its reviewed clean fresh controls")
                goods = candidate["target"]["documentPatch"]["goodsItemDetails"][0]
                if prefix == "e18486c6":
                    if "grossWeight" in goods or candidate["joinedRawText"].count(" KOS") != 2:
                        raise ValueError("private KOS decision or public absence changed")
                    physical = candidate["physical"]
                    gross = physical["measures"]["grossWeight"]["total"]
                    from decimal import Decimal

                    if f"GW {Decimal(gross):,.2f} KOS" not in candidate["joinedRawText"]:
                        raise ValueError("private gross does not match the KOS rendering")
                    tare = physical["tares"][0]["kg"]
                    if f"CNTR TARE\n{tare} KOS" not in candidate["joinedRawText"]:
                        raise ValueError("private tare does not match the KOS rendering")
                elif len(goods["numberAndTypeOfPackages"]) != 1:
                    raise ValueError("vehicle user decision requires one public package row")
            if findings and prefix not in DECISIONS and prefix not in holds:
                raise ValueError("unreviewed native finding family: " + sid)
            reason = holds.get(prefix) if prefix in holds else DECISIONS.get(prefix)
            finding_count += len(findings)
            batches[row["batch"]] += 1
            controls.append(
                {
                    "sampleId": candidate["documentId"],
                    "groupCount": row["groupCount"],
                    "batch": row["batch"],
                    **{
                        name: {"path": str(path), "sha256": sha256_file(path)}
                        for name, path in paths.items()
                    },
                    "findingDecisions": {digest(f): reason for f in findings},
                }
            )
        records[sid] = {
            "status": "HELD" if prefix in holds else "PASS",
            "reason": holds.get(prefix)
            or (
                "Three reviewed single-row controls preserve this source's one counted "
                "article/installed-contents-lot presentation; extra rows would invent structure."
                if prefix in SINGLE_ROW
                else "Single/two/three-group controls passed synthesis, native text review, "
                "main adjudication, target validation and geometry replay. Complete count/type "
                "ownership, nested nonadditivity and the source's public visibility are retained."
            ),
            "controls": controls,
        }
    return {
        "status": "REVIEW_COMPLETE_NOT_PRODUCTION_ACCEPTANCE"
        if resolved
        else "REVIEW_COMPLETE_WITH_TWO_USER_DECISIONS_PENDING_NOT_PRODUCTION_ACCEPTANCE",
        "inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        "sources": records,
        "summary": {
            "registrySources": len(records),
            "reviewedPassingSources": sum(r["status"] == "PASS" for r in records.values()),
            "heldSources": sum(r["status"] == "HELD" for r in records.values()),
            "selectedControls": sum(len(r["controls"]) for r in records.values()),
            "selectedNativeFindings": finding_count,
            "selectedControlBatches": dict(sorted(batches.items())),
        },
        "reviewScope": "Source ownership; rendered cargo text and public target; private "
        "accounting and capacity; source visibility; retained identifiers/parties; "
        "coordinate-supported reflow. Geometry replay is not pixel review of every PDF. "
        "This is adjudicated acceptance evidence, not a first-draw quality estimate or "
        "a live regulatory filing certification.",
        "trainingDatasetMutation": False,
        "defaultConfigurationChanged": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--user-resolutions-reviewed", action="store_true")
    args = parser.parse_args()
    result = build(args.root, args.inventory, args.user_resolutions_reviewed)
    atomic_publish_json(args.output, result)
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
