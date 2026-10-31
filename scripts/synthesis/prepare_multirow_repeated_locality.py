"""Bind all five reviewed Qingdao/China frames to their existing roles."""

import json
from pathlib import Path

import yaml
from prepare_multirow_marks_repair import replace_source
from rebase_reviewed_locality_frames import frame_proposals, rebase_owner_frames

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign


def main():
    root = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
    config = json.loads((root / "recovery-base-config.json").read_text())
    sid = next(s for s in config["source_ids"] if s.startswith("doc_3056615e"))
    config["source_ids"] = [sid]
    config["capabilities"] = {sid: config["capabilities"][sid]}
    source = _SourceOwnershipCampaign(Path.cwd(), config).blueprint(sid)
    proposals = frame_proposals(source)
    occurrences = [o for p in proposals for o in p["occurrences"]]
    if {(o["text"], o["occurrence"]) for o in occurrences} != {
        ("QINGDAO, CHINA", index) for index in range(1, 6)
    }:
        raise ValueError("reviewed repeated locality frames changed")
    original = Path(config["ownership"])
    content = original.read_text()
    declaration = yaml.safe_load(content)["sources"][sid]
    after = rebase_owner_frames(declaration, proposals)
    directory = root / "authority-repairs/repeated-locality"
    path = directory / "ownership.yaml"
    atomic_publish_bytes(path, replace_source(content, sid, after).encode())
    config["ownership"] = str(path)
    revised = _SourceOwnershipCampaign(Path.cwd(), config).blueprint(sid)
    if revised.source != source.source or revised.target != source.target:
        raise ValueError("locality frames changed source or target")
    atomic_publish_json(
        directory / "review.json",
        {
            "status": "COMPILED_AWAITING_SYNTHESIS",
            "sourceId": sid,
            "before": {"path": str(original), "sha256": sha256_file(original)},
            "after": {"path": str(path), "sha256": sha256_file(path)},
            "frames": proposals,
            "sourceAndTargetUnchanged": True,
            "reason": "Complete same-role inline frames preserve conditional country omission; "
            "five distinct source occurrences remain owned by their respective roles.",
        },
    )
    current = json.loads((root / "batch-27/prepared-config.json").read_text())
    current["ownership"] = str(path)
    atomic_publish_json(root / "batch-27/repeated-locality/config.json", current)


if __name__ == "__main__":
    main()
