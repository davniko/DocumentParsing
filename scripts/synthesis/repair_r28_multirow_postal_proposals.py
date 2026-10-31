"""Commit explicitly reviewed postal edits to rejected native wording proposals.

The exact before values, native proposal and all accepted values are retained.
This is an audit operation, not a production heuristic or automatic street edit.
"""

import argparse
import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.curated_wording import wording_output_type
from document_ocr.synthesis.curated_wording_receipts import wording_request_hash


def repair(config_path, edits_path):
    root = Path.cwd()
    campaign = create_campaign(root, json.loads(config_path.read_text()))
    if (campaign.output / "manifest.json").exists():
        raise ValueError("published campaign cannot receive proposal repairs")
    edits = json.loads(edits_path.read_text())
    for sid, changes in edits.items():
        proposals = []
        for path in (campaign.output / "calls").glob("wording-validation-repair-*.json"):
            data = json.loads(path.read_text())
            if data["identity"] == sid and data["output"] is not None:
                proposals.append((path, data["output"]))
        if len(proposals) != 1:
            raise ValueError("expected exactly one reviewed final native repair proposal")
        path, original = proposals[0]
        revised = copy.deepcopy(original)
        plans = campaign.plans(sid)
        requests = [campaign.wording_request(bp, sc, sample, t) for bp, sc, sample, t in plans]
        for change in changes:
            shipment, key = change["shipment"], change["key"]
            if change["before"] not in revised[shipment][key]:
                raise ValueError("reviewed before passage differs from native proposal")
            if revised[shipment][key].count(change["before"]) != 1:
                raise ValueError("reviewed correction must identify exactly one passage")
            variables = {v.key: v for v in plans[int(shipment[1:])][0].contract.variables}
            if variables[key].kind != "postal":
                raise ValueError("postal correction cannot change a nonpostal field")
            revised[shipment][key] = revised[shipment][key].replace(
                change["before"], change["after"]
            )
        output = wording_output_type(requests).model_validate(revised)
        accepted = campaign._unpack_generated_wording(output, requests)
        evidence = {
            "status": "MAIN_REVIEWED_REQUIRES_RENDER_NATIVE_REVIEW_GEOMETRY",
            "proposal": {"path": str(path.relative_to(root)), "sha256": sha256_file(path)},
            "edits": {"path": str(edits_path.relative_to(root)), "sha256": sha256_file(edits_path)},
            "reason": "Remove a duplicated sampled locality from fictional street/site wording; "
            "retain one exact locality occurrence, countries, line structure and other fields.",
            "changes": changes,
        }
        evidence_path = campaign.output / "main-proposal-repairs" / (sid + ".json")
        atomic_publish_json(evidence_path, evidence)
        for request, (bp, scenario, sample, target) in zip(requests, plans, strict=True):
            atomic_publish_json(
                campaign.output / "wording" / (sample + ".json"),
                {
                    "sourceDocumentId": sid,
                    "documentId": sample,
                    "requestSha256": wording_request_hash(request),
                    "scenario": scenario.model_dump(mode="json"),
                    "targetBeforeWording": target,
                    "values": accepted[sample],
                    "blueprint": bp.inventory(),
                    "mainProposalAdjudication": {
                        "path": str(evidence_path.relative_to(root)),
                        "sha256": sha256_file(evidence_path),
                    },
                },
            )
        print(json.dumps({"source": sid, "validatedProposalRecords": len(plans)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--edits", type=Path, required=True)
    args = parser.parse_args()
    repair(args.config.resolve(), args.edits.resolve())
