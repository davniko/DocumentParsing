"""Adjudicate one held native proposal with exact, recorded linguistic edits."""

import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.curated_campaign import create_campaign
from document_ocr.synthesis.curated_wording import wording_output_type
from document_ocr.synthesis.curated_wording_receipts import wording_request_hash


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    batch = root / "batch-37"
    campaign = create_campaign(Path.cwd(), json.loads((batch / "prepared-config.json").read_text()))
    sid = campaign.source_ids[0]
    plans = campaign.plans(sid)
    requests = [
        campaign.wording_request(bp, sc, sample, target) for bp, sc, sample, target in plans
    ]
    paths = list((campaign.output / "calls").glob("wording-validation-repair-*.json"))
    if len(paths) != 1:
        raise ValueError("expected one reviewed, held native proposal")
    proposal = json.loads(paths[0].read_text())["output"]
    revised = copy.deepcopy(proposal)
    for i, (_, scenario, _, _) in enumerate(plans):
        fields = revised[f"s{i}"]
        key = "audit_deliveryagent_addressline_1"
        old = fields[key]
        if len(old.splitlines()) != 2:
            raise ValueError("reviewed delivery address component changed")
        locality = scenario.party_localities["documentPatch.parties.deliveryAgent"].name
        fields[key] = old.replace("\n", ", " + locality.upper() + "\n", 1)
    if revised["s0"]["product_3"] != "CUE CHALK, FELT CLOTH AND NET REPAIR KITS":
        raise ValueError("reviewed third-HS product passage changed")
    revised["s0"]["product_3"] = "POLYESTER BED NETS, FINE MESH, INSECTICIDE TREATED"
    revised["s2"]["product_3"] = "REFINED LACTOSE POWDER FOR PHARMACEUTICAL USE"
    output = wording_output_type(requests).model_validate(revised)
    accepted = campaign._unpack_generated_wording(output, requests)
    evidence = {
        "status": "MAIN_REVIEWED_REQUIRES_RENDER_NATIVE_REVIEW_GEOMETRY",
        "proposal": {"path": str(paths[0]), "sha256": sha256_file(paths[0])},
        "reason": "Native repair removed both repeated postal-locality occurrences; restore "
        "exactly one in the country-owning postal component. Correct the two reviewed "
        "product continuations to their sampled third-HS identity without new packing claims.",
        "changes": {
            key: {
                name: {"before": proposal[key][name], "after": text}
                for name, text in fields.items()
                if text != proposal[key][name]
            }
            for key, fields in revised.items()
        },
    }
    path = batch / "main-proposal-adjudication.json"
    atomic_publish_json(path, evidence)
    for bp, scenario, sample, target in plans:
        request = next(q for q in requests if q.sample_id == sample)
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
                "mainProposalAdjudication": {"path": str(path), "sha256": sha256_file(path)},
            },
        )


if __name__ == "__main__":
    main()
