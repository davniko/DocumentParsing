"""Withdraw superseded notify targets and refill exactly the same 18 scheduled slots."""

import json
from pathlib import Path

from expand_r28_2500 import selected_config
from stage_expansion_notify_authority import SOURCE
from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, atomic_write_json
from document_ocr.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/r28-expansion2500-20261101"
RUN = ROOT / "artifacts/kie-synthesis-production/r28-expansion2500-20261101"


def main():
    base = json.loads((ROOT / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/expansion-20261101-v21/complete-source-authority-config.json").read_text())
    original_groups = []
    for path in sorted(RUN.glob("batch-*/ordinary/config.json")):
        config = json.loads(path.read_text())
        if SOURCE in config["source_ids"]:
            original_groups.extend(config["declared_packages"]["source_group_counts"][SOURCE])
    if sorted(original_groups) != [1] * 15 + [2] * 3:
        raise ValueError("original notify-source campaign quotas changed")
    jobs = []
    for index in range(1, 5):
        control = index == 4
        path = RUN / ("notify-owner-controls" if control else f"notify-repair-{index:02}") / "config.json"
        groups = [1, 2, 3] if control else [1, 1, 1, 1, 1, 2]
        config = selected_config(base, {SOURCE: len(groups)}, path.parent / "campaign", 2026110210 + index, group_counts=groups)
        config["declared_packages"]["source_group_counts"] = {SOURCE: groups}
        config["independent_preparation"]["mixed_source_ids"] = []
        atomic_publish_json(path, config)
        jobs.append({"config": str(path.relative_to(ROOT)), "configSha256": sha256_file(path), "samples": len(groups), "role": "control" if control else "replacement"})
    withdrawal_path = OUT / "withdrawn-staged-samples.json"
    old_bytes = withdrawal_path.read_bytes()
    changed = json.loads(old_bytes)
    selected = []
    for path in sorted((OUT / "components").glob("*/certification.json")):
        receipt = json.loads(path.read_text())
        review = json.loads((ROOT / receipt["mainAudit"]["path"]).read_text())
        for sample, evidence in review["samples"].items():
            if evidence["sourceDocumentId"] == SOURCE:
                if sample in changed["samples"]:
                    raise ValueError("notify sample was already withdrawn")
                changed["samples"][sample] = "Shared ** footnote establishes notify phone/email, omitted by the earlier target. Preserve historical evidence; replace with freshly generated corrected source authority. Consignee exclusion is not changed."
                selected.append(sample)
    if len(selected) != 16:
        raise ValueError("expected sixteen published and two unpublished notify-source slots")
    atomic_publish_bytes(OUT / "notify-repair/withdrawals-before.json", old_bytes)
    atomic_publish_json(OUT / "redraw-plans/notify-repair.json", {"jobs": jobs, "replacementSamples": 18, "controlSamples": 3, "withdrawn": selected})
    atomic_write_json(withdrawal_path, changed)
    print(json.dumps({"withdrawnPublishedDrafts": 16, "replacements": 18, "separateControls": 3}), flush=True)


if __name__ == "__main__":
    main()
