"""Re-pin reviewed geography after a proven postal-preserving source rebase.

Reconstruct the exact previously inspected ownership, verify its old certificate,
and compare the protected name/postal state. No relaxed certificate matching or
new geography decision is introduced. Original catalogs remain immutable.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file
from document_ocr.synthesis.independent_geography import (
    IndependentGeographyGate,
    load_independent_geography_certificates,
)
from document_ocr.synthesis.independent_preparation import _SourceOwnershipCampaign


def validate_postal_preserving_rebase(gate, previous, current) -> tuple[str, str]:
    """Require an exact old certificate and unchanged protected actor ownership."""
    certificate = gate._certificate(previous, allow_prepared=False)
    if certificate is None:
        raise ValueError("source has no original private postal certificate")
    if (
        current.document_id != previous.document_id
        or current.source != previous.source
        or current.target != previous.target
    ):
        raise ValueError("source rebase changed source identity, OCR or extraction target")
    if gate._postal_state(current, certificate) != gate._postal_state(previous, certificate):
        raise ValueError("source rebase changed certified private actor/postal ownership")
    return gate._blueprint_pins(current)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    config = json.loads(args.config.read_text())
    review = json.loads(args.review.read_text())
    assert all(row["completeRead"] and row["decision"] for row in review["sources"])
    assert {row["prefix"] for row in review["sources"]} == {
        sid[4:12] for sid in config["source_ids"]
    }
    pin = config["independent"]["geography_certificates"]
    old_path = root / pin["path"]
    original = json.loads(old_path.read_bytes())
    gate = IndependentGeographyGate(
        load_independent_geography_certificates(
            old_path.read_bytes(), expected_sha256=pin["sha256"]
        )
    )
    source = _SourceOwnershipCampaign(root, config)
    updated = copy.deepcopy(original)
    evidence = []
    for sid in config["source_ids"]:
        current = source.blueprint(sid)
        if sid not in gate.certificates:
            gate._certificate(current)
            continue
        certificate = gate.certificates[sid]
        old_pins = (certificate.effective_contract_sha256, certificate.ownership_sha256)
        new_pins = gate._blueprint_pins(current)
        if new_pins == old_pins:
            gate._certificate(current)
            continue
        folder = args.inspection / sid
        assert (folder / "source.txt").read_bytes() == current.source.encode()
        assert json.loads((folder / "target.json").read_bytes()) == current.target
        ownership = json.loads((folder / "owners.json").read_bytes())["ownership"]
        old_owner_path = args.output / (sid + "-previous-ownership.json")
        atomic_publish_json(old_owner_path, {"version": 1, "sources": {sid: ownership}})
        old_config = copy.deepcopy(config)
        old_config.update(
            ownership=str(old_owner_path),
            source_ids=[sid],
            capabilities={sid: config["capabilities"][sid]},
        )
        previous = _SourceOwnershipCampaign(root, old_config).blueprint(sid)
        assert validate_postal_preserving_rebase(gate, previous, current) == new_pins
        updated["sources"][sid].update(
            effective_contract_sha256=new_pins[0], ownership_sha256=new_pins[1]
        )
        evidence.append(
            {
                "sourceDocumentId": sid,
                "oldPins": old_pins,
                "newPins": new_pins,
                "exactOriginalCertificateVerified": True,
                "protectedPostalStateUnchanged": True,
                "sourceAndPublicTargetUnchanged": True,
            }
        )
    target = args.output / "geography_certificates.json"
    atomic_publish_json(target, updated)
    final_gate = IndependentGeographyGate(
        load_independent_geography_certificates(
            target.read_bytes(), expected_sha256=sha256_file(target)
        )
    )
    for sid in config["source_ids"]:
        final_gate._certificate(source.blueprint(sid), allow_prepared=False)
    assert set(updated["sources"]) == set(original["sources"])
    for sid in set(original["sources"]) - {row["sourceDocumentId"] for row in evidence}:
        assert original["sources"][sid] == updated["sources"][sid]
    config["independent"]["geography_certificates"] = {
        "path": str(target),
        "sha256": sha256_file(target),
    }
    atomic_publish_json(args.output / "config.json", config)
    atomic_publish_json(
        args.output / "validation.json",
        {
            "beforeCatalog": pin,
            "afterCatalog": config["independent"]["geography_certificates"],
            "review": {"path": str(args.review), "sha256": sha256_file(args.review)},
            "sources": evidence,
            "otherCertificatesUnchanged": True,
            "activeSelectedSourcesVerified": len(config["source_ids"]),
            "apiCalls": 0,
        },
    )
    print(
        json.dumps(
            {
                "rebasedCertificates": len(evidence),
                "sourcesVerified": len(config["source_ids"]),
                "config": str(args.output / "config.json"),
                "apiCalls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
