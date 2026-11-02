"""Stage two reviewed complete route frames; preserve source text and targets."""

import copy
import json
from pathlib import Path

from rebase_reviewed_geography import validate_postal_preserving_rebase
from rebase_reviewed_locality_frames import frame_proposals, rebase_owner_frames

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.independent_geography import (
    IndependentGeographyGate,
    load_independent_geography_certificates,
)
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed"
OUTPUT = REGISTRY / "expansion-20261101-v19"
EXPECTED = {
    "48973902": ["Hamburg, Germany", "Alexandria Seaport, Egypt"],
    "35288b68": ["QINGDAO,CHINA", "SOKHNA,EGYPT", "Qingdao, China", "Qingdao, China"],
}


def main():
    parent = REGISTRY / "expansion-20261101-v17/complete-source-authority-config.json"
    config = json.loads(parent.read_text())
    owners = json.loads((ROOT / config["ownership"]).read_text())
    previous = _SourceOwnershipCampaign(ROOT, config)
    reviewed = {}
    for prefix, quotes in EXPECTED.items():
        sid = next(s for s in config["source_ids"] if s[4:12] == prefix)
        frames = frame_proposals(previous.blueprint(sid))
        if [o["text"] for f in frames for o in f["occurrences"]] != quotes:
            raise ValueError("reviewed locality frames changed: " + sid)
        owners["sources"][sid] = rebase_owner_frames(owners["sources"][sid], frames)
        reviewed[sid] = frames
    owner_path = OUTPUT / "ownership.json"
    atomic_publish_json(owner_path, owners)
    updated = copy.deepcopy(config)
    updated["ownership"] = str(owner_path.relative_to(ROOT))
    current = _SourceOwnershipCampaign(ROOT, updated)
    pin = config["independent"]["geography_certificates"]
    raw = (ROOT / pin["path"]).read_bytes()
    gate = IndependentGeographyGate(load_independent_geography_certificates(
        raw, expected_sha256=pin["sha256"]
    ))
    catalog = json.loads(raw)
    evidence = []
    for sid, frames in reviewed.items():
        before, after = previous.blueprint(sid), current.blueprint(sid)
        if before.source != after.source or before.target != after.target:
            raise ValueError("route frame rebase changed source text or target")
        if sid in gate.certificates:
            pins = validate_postal_preserving_rebase(gate, before, after)
            catalog["sources"][sid].update(effective_contract_sha256=pins[0], ownership_sha256=pins[1])
        else:
            # Sources without a private-postal exception need no invented
            # certificate. Verify that the ordinary gate still accepts both.
            gate._certificate(before, allow_prepared=False)
            gate._certificate(after, allow_prepared=False)
            touched = {k for f in frames for k in f["oldKeys"]} | {f["key"] for f in frames}
            protected_before = [r for r in before.regions if r.key not in touched]
            protected_after = [r for r in after.regions if r.key not in touched]
            if protected_before != protected_after:
                raise ValueError("unrelated region ownership changed")
        evidence.append({"sourceDocumentId": sid, "frames": frames,
                         "sourceAndTargetsUnchanged": True, "privatePostalOwnershipUnchanged": True})
    path = OUTPUT / "geography_certificates.json"
    atomic_publish_json(path, catalog)
    updated["independent"]["geography_certificates"] = {
        "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
    }
    atomic_publish_json(OUTPUT / "complete-source-authority-config.json", updated)
    atomic_publish_json(OUTPUT / "review.json", {
        "parent": {"path": str(parent.relative_to(ROOT)), "sha256": sha256_file(parent)},
        "status": "REQUIRES_FRESH_SYNTHESIS", "changes": evidence,
        "excludedProposal": "v18 mandatory exporter/shipper country equality was not established by the care-of source; do not promote it.",
    })
    print(OUTPUT.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
