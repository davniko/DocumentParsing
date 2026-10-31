"""Restore the verified utilization policy omitted by final configuration assembly."""

import copy
import json
from pathlib import Path

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file

ROOT = Path("artifacts/synthesis-templates/mpci-bl-v7-reviewed/multirow-rollout-20261031")
CONFIG = Path("configs/synthesis/mpci_bl_curated_v7_multirow.json")
MANIFEST = ROOT.parent / "manifest.json"


def read(path):
    return json.loads(path.read_text())


def pin(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def main():
    config = read(CONFIG)
    manifest = read(MANIFEST)
    publication = read(ROOT / "publication.json")
    original_pin = pin(CONFIG)
    if manifest["currentProductionConfig"] != original_pin or publication["config"] != original_pin:
        raise ValueError("production configuration changed outside the reviewed repair")
    if "utilization" in config["declared_packages"]:
        raise ValueError("repair is only for the confirmed missing setting")
    replay = read(ROOT / "final-authority-replay-v4/summary.json")
    configurations = []
    for batch in replay["batches"]:
        path = ROOT / "final-authority-replay-v4" / batch["batch"] / "config.json"
        if read(path)["declared_packages"]["utilization"] != [0.35, 0.75]:
            raise ValueError("certified cohort does not share the reviewed utilization envelope")
        configurations.append(pin(path))
    for path in (CONFIG, MANIFEST, ROOT / "publication.json"):
        atomic_publish_bytes(ROOT / "configuration-repair-before" / path.name, path.read_bytes())
    updated = copy.deepcopy(config)
    updated["declared_packages"]["utilization"] = [0.35, 0.75]
    atomic_write_json(CONFIG, updated)
    evidence_path = ROOT / "configuration-utilization-repair.json"
    atomic_publish_json(
        evidence_path,
        {
            "reason": (
                "Final configuration assembly omitted the explicit envelope already present "
                "in every accepted/replayed cohort. No sampler fallback was used."
            ),
            "before": original_pin,
            "after": pin(CONFIG),
            "changedKey": "declared_packages.utilization",
            "value": [0.35, 0.75],
            "verifiedReplayConfigurations": configurations,
            "sourceControlsUnchanged": True,
            "trainingDatasetMutation": False,
        },
    )
    manifest["currentProductionConfig"] = pin(CONFIG)
    manifest["multirowPackages20261031"]["config"] = pin(CONFIG)
    manifest["multirowPackages20261031"]["configurationRepair"] = pin(evidence_path)
    atomic_write_json(MANIFEST, manifest)
    publication["config"] = pin(CONFIG)
    publication["manifest"] = pin(MANIFEST)
    publication["configurationRepair"] = pin(evidence_path)
    atomic_write_json(ROOT / "publication.json", publication)


if __name__ == "__main__":
    main()
