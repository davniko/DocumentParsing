"""Stage two incompatible retained cargoes using existing certified nested proposals."""

import argparse
import copy
import json

from migrate_r30_synthetic import ART, DATA, OUT, ROOT, read
from publish_r30_sources import pin, verify_pin
from r30_scoped_receipts import compose_cargo

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.paddle_templates import PaddleTemplateCatalog

SELECTION = {
    "syn_full_v7_0bb3c61af3538f999d9ecaef": "syn_full_v7_9c3a38738dbd7bd76324f9f3",
    "syn_full_v7_142dbc1480867d384e26200b": "syn_full_v7_becabf69293e4a010204fe0b",
}


def main(selection_path=None):
    decisions = read(selection_path) if selection_path else None
    selection = {k: v["freshId"] for k, v in decisions.items()} if decisions else SELECTION
    inventory = read(OUT / "receipt-inventory.json")
    rows = {
        r["documentId"]: r
        for r in map(json.loads, verify_pin(inventory["baseline"]).open())
        if r["documentId"] in selection
    }
    manifest = read(DATA / "manifest.json")
    catalog = PaddleTemplateCatalog(ROOT, manifest["templateCatalog"])
    controls = read(verify_pin(manifest["templateControls"]))
    for entry in inventory["records"]:
        identity = entry["id"]
        if identity not in selection:
            continue
        retained_path = ROOT / entry["candidate"]
        _, _, transfer, _ = catalog._template(entry["source"])
        fresh_choices = [d for d in transfer["descendants"] if d["id"] == selection[identity]]
        if len(fresh_choices) != 1:
            raise ValueError("fresh cargo is not a unique admitted control")
        fresh_path = ROOT / fresh_choices[0]["path"]
        prior_controls = [
            p
            for p in controls[entry["source"]]
            if read(verify_pin(p))["documentId"] == selection[identity]
        ]
        if len(prior_controls) != 1:
            raise ValueError("fresh cargo is not an exact certified control")
        candidate, replacement = compose_cargo(
            rows[identity],
            read(retained_path),
            read(fresh_path),
            set(decisions[identity]["privateOwnerKeys"])
            if decisions
            else {"agent:reviewed:shipment:gross_weight", "agent:reviewed:shipment:volume"},
        )
        candidate["migrationCargoRefresh"].update(
            previousCandidate=pin(retained_path),
            freshCandidate=pin(fresh_path),
            certifiedControl=prior_controls[0],
            reason=decisions[identity]["reason"]
            if decisions
            else "Newly visible three-level packing is incompatible with the retained cargo. "
            "Reuse unused certified donor-free cargo; retain every noncargo owner and target.",
        )
        directory = ART / "nested-refresh" / identity / digest(candidate)
        destination = directory / "candidate.json"
        atomic_publish_json(destination, candidate)
        atomic_publish_json(directory / "replacement.json", replacement)
        previous_wording = ART / "wording" / (identity + ".json")
        wording = read(previous_wording)
        wording_candidate = retained_path
        if wording["candidateSha256"] != pin(retained_path)["sha256"]:
            identity_receipt = ART / "identity-receipts" / (identity + ".json")
            renamed = read(identity_receipt)
            expected = {
                **read(retained_path),
                "documentId": identity,
                "migrationIdentity": renamed["migrationIdentity"],
            }
            if renamed != expected or renamed["migrationIdentity"]["previousCandidate"] != pin(
                retained_path
            ):
                raise ValueError("wording identity receipt changes shipment context")
            wording_candidate = identity_receipt
        old_spec = catalog.wording_specification(wording_candidate)
        new_spec = catalog.wording_specification(destination)
        expected_wording = (
            set(decisions[identity]["retainedWordingKeys"])
            if decisions
            else {"deliveryAddress", "deliveryName"}
        )
        if set(old_spec["requested"]) != expected_wording:
            raise ValueError("wording reuse is reviewed only for the unchanged delivery agency")
        if old_spec["requested"] != new_spec["requested"]:
            raise ValueError("wording instructions changed")
        for field in ("origin", "destination", "parties", "partyLocalities"):
            if old_spec["context"][field] != new_spec["context"][field]:
                raise ValueError("delivery-agency context changed")
        if (
            wording["status"] != "REVIEWED"
            or wording["candidateSha256"] != pin(wording_candidate)["sha256"]
        ):
            raise ValueError("retained wording is not reviewed for the exact old receipt")
        wording = copy.deepcopy(wording)
        wording["candidateSha256"] = pin(destination)["sha256"]
        wording["cargoRefreshReuse"] = {
            "previousWording": pin(previous_wording),
            "unchangedContextFields": ["origin", "destination", "parties", "partyLocalities"],
            "scope": "Only delivery-agency name/address. No goods-dependent wording reused.",
        }
        atomic_publish_json(directory / "wording.json", wording)
        rendered = catalog.render(destination, current=replacement, wording_receipt=wording)
        atomic_publish_json(directory / "rendered.json", rendered)
        atomic_publish_json(
            directory / "proposal.json",
            {
                "status": "SCOPED_CARGO_RENDER_REQUIRES_MAIN_AGENT_REVIEW",
                "id": identity,
                "currentRecordSha256": digest(rows[identity]),
                "candidate": pin(destination),
                "replacementRecord": pin(directory / "replacement.json"),
                "wording": pin(directory / "wording.json"),
                "rendered": pin(directory / "rendered.json"),
                "catalog": manifest["templateCatalog"],
            },
        )
        print(identity, json.dumps(rendered["target"]["goodsItemDetails"], ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection")
    args = parser.parse_args()
    main(ROOT / args.selection if args.selection else None)
