"""Apply exact main-reviewed lexical edits to unadmitted campaign receipts."""

import argparse
import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_campaign import create_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--edits", type=Path, required=True)
    parser.add_argument("--field-kind", choices=["postal", "name"], default="postal")
    args = parser.parse_args()
    campaign = create_campaign(Path.cwd(), json.loads(args.config.read_text()))
    if (campaign.output / "manifest.json").exists():
        raise ValueError("published samples are immutable")
    staged = []
    for change in json.loads(args.edits.read_text()):
        sample, key = change["sample"], change["key"]
        path = campaign.output / "wording" / (sample + ".json")
        data = path.read_bytes()
        receipt = json.loads(data)
        candidate_path = campaign.output / "candidates" / (sample + ".json")
        candidate_bytes = candidate_path.read_bytes()
        candidate = json.loads(candidate_bytes)
        bp, scenario, actual, target = campaign.plan(candidate["sourceDocumentId"], candidate["variant"])
        if actual != sample or candidate["wordingReceiptSha256"] != digest(receipt):
            raise ValueError("reviewed candidate/receipt differs")
        variable = next(v for v in bp.contract.variables if v.key == key)
        if variable.kind != args.field_kind or receipt["values"][key] != change["before"] or not change["reason"]:
            raise ValueError("only exact reviewed edits of the declared field kind are authorized")
        request = campaign.wording_request(bp, scenario, sample, target)
        campaign._validated_wording(receipt, request)
        changed = copy.deepcopy(receipt)
        changed["values"][key] = change["after"]
        campaign.validate_generated_wording([request], {sample: changed["values"]})
        archive = campaign.output / "wording-history" / (sample + "-" + digest(data) + ".json")
        changed.setdefault("corrections", []).append({
            "kind": "main_agent_scoped_lexical_review", **change,
            "beforeReceipt": {"path": str(archive), "sha256": digest(data)},
            "reviewedPlan": {"path": str(args.edits), "sha256": digest(args.edits.read_bytes())},
        })
        staged.append((path, data, changed, archive, candidate_path, candidate_bytes))
    rows = []
    for path, data, changed, archive, candidate_path, candidate_bytes in staged:
        atomic_publish_bytes(archive, data)
        candidate_archive = campaign.output / "candidate-history" / (path.stem + "-" + digest(candidate_bytes) + ".json")
        if candidate_archive.exists():
            raise ValueError("reviewed candidate archive exists")
        candidate_archive.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.rename(candidate_archive)
        atomic_write_json(path, changed)
        rows.append({"sample": path.stem, "wordingArchive": str(archive), "candidateArchive": str(candidate_archive)})
    atomic_publish_json(args.edits.with_suffix(".receipt.json"), {
        "status": "CORRECTED_REQUIRES_FRESH_RENDER_REVIEW_GEOMETRY", "corrections": rows,
    })


if __name__ == "__main__":
    main()
