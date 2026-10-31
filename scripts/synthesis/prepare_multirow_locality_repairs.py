"""Prepare reviewed complete-inline locality owners without mutating live authorities."""

import copy
import json
import re
from pathlib import Path

import yaml
from rebase_reviewed_locality_frames import frame_proposals, rebase_owner_frames

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign

REVIEWED = {
    "1b176673": {"NSICT, INDIA", "AIN SOKHNA PORT, EGYPT"},
    "22fa3887": {"DAMIETTA,EGYPT", "MUNDRA ,India"},
    "2d5864aa": {"MUNDRA, INDIA", "DAMIETTA, EGYPT"},
    "3929e40b": {"NINGBO, CHINA", "SOKHNA-EGYPT"},
}


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    config = json.loads(
        Path("configs/synthesis/mpci_bl_curated_v7_package_aliases.json").read_text()
    )
    config["source_ids"] = [s for s in config["source_ids"] if s[4:12] in REVIEWED]
    config["capabilities"] = {s: config["capabilities"][s] for s in config["source_ids"]}
    campaign = _SourceOwnershipCampaign(Path.cwd(), config)
    original = Path(config["ownership"])
    content = original.read_text()
    authority = yaml.safe_load(content)
    decisions = []
    for sid in config["source_ids"]:
        bp = campaign.blueprint(sid)
        proposals = frame_proposals(bp)
        observed = {o["text"] for p in proposals for o in p["occurrences"]}
        if observed != REVIEWED[sid[4:12]]:
            raise ValueError("reviewed inline frames changed: " + sid)
        replacement = rebase_owner_frames(authority["sources"][sid], proposals)
        block = (
            "\n".join(
                "  " + line if line else ""
                for line in yaml.safe_dump(
                    {sid: replacement}, allow_unicode=True, sort_keys=False, width=100
                ).splitlines()
            )
            + "\n"
        )
        pattern = r"^  " + re.escape(sid) + r":\n.*?(?=^  doc_|\Z)"
        content, count = re.subn(pattern, lambda _, block=block: block, content, flags=re.M | re.S)
        if count != 1:
            raise ValueError("source ownership block is not unique")
        decisions.append(
            {
                "sourceId": sid,
                "frames": proposals,
                "decision": "Pair only adjacent same-role locality/country scalars; "
                "country omission remains conditional on equal sampled identities. "
                "Source text, target and every cargo owner remain unchanged.",
            }
        )
    output = root / "authority-repairs/locality-frames-expanded"
    path = output / "ownership.yaml"
    atomic_publish_bytes(path, content.encode())
    revised = copy.deepcopy(config)
    revised["ownership"] = str(path)
    compiled = _SourceOwnershipCampaign(Path.cwd(), revised)
    for sid in config["source_ids"]:
        before, after = campaign.blueprint(sid), compiled.blueprint(sid)
        if before.source != after.source or before.target != after.target:
            raise ValueError("inline owner repair changed source bytes or labels")

        def cargo(bp):
            return [
                r
                for r in bp.regions
                if any(
                    "goodsItemDetails" in p or "containerInformation" in p for p in r.target_paths
                )
            ]

        if cargo(before) != cargo(after):
            raise ValueError("inline owner repair changed cargo ownership")
    atomic_publish_json(
        output / "review.json",
        {
            "status": "COMPILED_AWAITING_SYNTHESIS",
            "base": {"path": str(original), "sha256": sha256_file(original)},
            "ownership": {"path": str(path), "sha256": sha256_file(path)},
            "sources": decisions,
        },
    )
    for batch in (4, 5, 6, 7):
        directory = root / f"batch-{batch:02}"
        current = json.loads((directory / "prepared-config.json").read_text())
        current["ownership"] = str(path)
        atomic_publish_json(directory / "prepared-config-expanded-locality-repair.json", current)
    print(json.dumps({"sources": len(decisions), "sourceAndCargoUnchanged": True}))


if __name__ == "__main__":
    main()
